"""Minimal stock-Wan foundation candidate, isolated from historical adapters.

Layout: anchor / up to K=8 history / up to six archive / optional four render / four target latents.
All appearance uses stock 48-channel patch embedding. Camera and render metadata
have distinct small-random projections; no reliability/init policy is attached.
Temporal positions enter RoPE only; there is no explicit age embedding.
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from gmlib.diffsynth.models.wan_video_dit import sinusoidal_embedding_1d


class Foundation(nn.Module):
    def __init__(self,dit):
        super().__init__();self.dit=dit;dim=dit.dim
        self.camera=nn.Conv3d(24,dim,(1,2,2),stride=(1,2,2))
        self.metadata=nn.Conv3d(2,dim,(1,2,2),stride=(1,2,2))
        self.roles=nn.Embedding(5,dim)
        # Lyra-2 cam/buffer encoders use truncated N(0, 1/dim), bounded at 3 std.
        # Apply that weight initialization to our new projections and role table;
        # pretrained Wan weights are already loaded and are never reinitialized.
        std=dim**-0.5
        self.condition_init=dict(distribution='truncated_normal',mean=0.,std=std,
            truncation_std=3.,bias='zero',modules=['camera','metadata','roles'])
        for layer in (self.camera,self.metadata,self.roles):
            nn.init.trunc_normal_(layer.weight,mean=0.,std=std,a=-3*std,b=3*std)
            if getattr(layer,'bias',None) is not None:nn.init.zeros_(layer.bias)

    def configure_trainables(self):
        for p in self.dit.parameters():p.requires_grad_(False)
        for name,p in self.dit.named_parameters():
            if '.self_attn.' in name or name.startswith('patch_embedding.'):
                p.requires_grad_(True);p.data=p.data.float()
        for module in (self.camera,self.metadata,self.roles):module.float()
        return [(n,p) for n,p in self.named_parameters() if p.requires_grad]

    def forward(self,latents,rays,metadata,roles,positions,text,u,context_zero=True,checkpoint_blocks=False,
                target_bias=None,camera_mask=None,causal_context=False,context_u=None,correspondence=None):
        dit=self.dit;b,_,frames,_,_=latents.shape
        counts=[int((roles==i).sum()) for i in range(5)]
        if (counts[0]!=1 or not 0<=counts[1]<=8 or not 0<=counts[2]<=6 or counts[3] not in (0,4)
                or counts[4]!=4 or frames!=sum(counts) or roles.numel()!=frames
                or latents.shape[1]!=48 or rays.shape[1]!=24 or positions.numel()!=frames):
            raise ValueError('requires anchor / 0..8 history / 0..6 archive / 0 or 4 render / 4 target')
        expected=torch.repeat_interleave(torch.arange(5,device=roles.device),torch.tensor(counts,device=roles.device))
        if not torch.equal(roles,expected):raise ValueError('context roles are out of order')
        if rays.shape[2:]!=latents.shape[2:] or metadata.shape[1:]!=(2,*latents.shape[2:]):
            raise ValueError('camera or metadata grid differs from appearance')
        if (positions<0).any():raise ValueError('negative temporal position')
        if not torch.equal(positions[-4:],torch.arange(int(positions[-4]),int(positions[-4])+4,device=positions.device)):
            raise ValueError('target positions must be consecutive')
        if (positions[roles<3]>=positions[-4]).any():raise ValueError('future appearance in context')
        if counts[3] and not torch.equal(positions[roles==3],positions[-4:]):raise ValueError('render/target position mismatch')
        camera=self.camera(rays)
        if camera_mask is not None:
            if camera_mask.shape!=(frames,) or camera_mask.dtype!=torch.bool:
                raise ValueError('camera_mask must be one boolean per latent slot')
            # Mask the complete projection, including its learned bias.
            camera=camera*camera_mask.to(camera.device)[None,None,:,None,None]
        x=dit.patch_embedding(latents)+camera+self.metadata(metadata)
        _,dim,f,h,w=x.shape
        x=x.permute(0,2,3,4,1)
        x=x+self.roles(roles).to(x.dtype)[None,:,None,None,:]
        x=x.reshape(b,f*h*w,dim)
        if target_bias is not None:
            if target_bias.shape!=(b,4*h*w,dim):raise ValueError('target endpoint conditioning grid mismatch')
            x=torch.cat([x[:,:-4*h*w],x[:,-4*h*w:]+target_bias.to(x.dtype)],dim=1)
        times=torch.full((f,),float(u)*1000,device=x.device,dtype=x.dtype)
        if context_zero:times[:-4]=0
        if context_u is not None:
            if (context_u.shape!=(f-4,) or not torch.isfinite(context_u).all()
                    or (context_u<0).any() or (context_u>1).any()):
                raise ValueError('one finite context noise level in [0,1] per context slot required')
            times[:-4]=context_u.to(times)*1000
        t=dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim,times).to(x.dtype))
        tm=dit.time_projection(t).reshape(f,6,dim).repeat_interleave(h*w,dim=0)[None]
        text=dit.text_embedding(text)
        p=positions.long()
        if int(p.max())<len(dit.freqs[0]):temporal=dit.freqs[0].to(x.device)[p]
        else:
            phase=p.double()[:,None]*torch.angle(dit.freqs[0][1]).to(x.device).double()[None]
            temporal=torch.polar(torch.ones_like(phase),phase).to(dit.freqs[0].dtype)
        frequencies=torch.cat([
            temporal.view(f,1,1,-1).expand(f,h,w,-1),
            dit.freqs[1].to(x.device)[:h].view(1,h,1,-1).expand(f,h,w,-1),
            dit.freqs[2].to(x.device)[:w].view(1,1,w,-1).expand(f,h,w,-1)],dim=-1).reshape(f*h*w,1,-1)
        attention_bias=None
        if causal_context:
            # Chunk causal: past states cannot read current render/noisy targets;
            # render states cannot read noisy targets. All four current targets
            # may attend each other. Later calls are never present in this input.
            levels=torch.where(roles<3,0,torch.where(roles==3,1,2)).repeat_interleave(h*w)
            allowed=levels[:,None]>=levels[None,:]
            attention_bias=x.new_zeros((1,1,f*h*w,f*h*w)).masked_fill(~allowed[None,None],-torch.inf)
        camera_tokens=camera.permute(0,2,3,4,1).reshape(b,f*h*w,dim)
        address_bias=None
        if getattr(self,'native_patch_address',False) and correspondence is not None:
            from joint_v9.correspondence import attention_prior
            address_bias=attention_prior(correspondence,roles,h*w,camera_mask,x.device,x.dtype)
        for index,block in enumerate(dit.blocks):
            qk_bias=None
            if getattr(self,'block_camera_qk',False):
                qk_bias=camera_tokens*self.camera_qk_gains[index].tanh().to(camera_tokens.dtype)
            block_bias=attention_bias
            if address_bias is not None:
                weighted=address_bias*self.archive_address_gains[index].tanh().to(address_bias.dtype)
                block_bias=weighted if attention_bias is None else attention_bias+weighted
            if checkpoint_blocks and torch.is_grad_enabled():
                if block_bias is None and qk_bias is None:x=checkpoint(block,x,text,tm,frequencies,use_reentrant=False)
                else:x=checkpoint(block,x,text,tm,frequencies,qk_bias,block_bias,use_reentrant=False)
            else:x=block(x,text,tm,frequencies,qk_bias=qk_bias,attn_logit_bias=block_bias)
        x=dit.head(x[:,-4*h*w:],t[-4:].repeat_interleave(h*w,dim=0)[None])
        return dit.unpatchify(x,(4,h,w))


def packed_rays(c2w,K,latent_positions,height=256,width=448):
    """Four RGB cameras per latent, unit rays at latent-cell centres.

Pose translations retain the provided common COLMAP gauge. This preflight does
not establish a metric scale or a deployment anchor-depth normalization.
"""
    # ReSplat and the point renderer use raster pixel centres x+0.5.
    # The mean centre of each 16-pixel latent cell is therefore 8, 24, ... .
    yy,xx=torch.meshgrid(torch.arange(height//16)*16+8.,torch.arange(width//16)*16+8.,indexing='ij')
    pixels=torch.stack([xx,yy,torch.ones_like(xx)]).float().reshape(3,-1)
    local=torch.linalg.inv(K.float())@pixels
    poses=torch.linalg.inv(c2w[0].float())[None]@c2w.float()
    frames=[]
    for pos in latent_positions:
        pos=int(pos);ids=[0]*4 if pos==0 else list(range(4*pos-3,4*pos+1))
        channels=[]
        for idx in ids:
            direction=poses[idx,:3,:3]@local
            direction=direction/direction.norm(dim=0,keepdim=True).clamp_min(1e-8)
            moment=torch.linalg.cross(poses[idx,:3,3,None].expand_as(direction),direction,dim=0)
            channels.append(torch.cat([direction,moment]).reshape(6,height//16,width//16))
        frames.append(torch.cat(channels))
    return torch.stack(frames,dim=1)[None]
