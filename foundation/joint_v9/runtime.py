"""A single-call runtime boundary shared by harvesting, training probes and eval."""
from dataclasses import dataclass
import copy
import hashlib
import numpy as np
import torch
from torch.nn import functional as F
from foundation_contract import build_condition, append_native
from foundation_depth import causal_depth
from .model import move_condition
from .sampling import sample
from .retrieval import select_archive, POLICY


@dataclass
class RuntimeState:
    prefix: torch.Tensor
    frames: np.ndarray
    past_c2w: torch.Tensor
    K: torch.Tensor
    anchor_depth: torch.Tensor
    anchor_mask: torch.Tensor
    gauge_id: str
    decoder_prefix: dict
    scene_id: str
    stream_id: str
    split: str
    calls: int=0
    source_pose_origin: str='commanded_baseline_writer'

    def validate(self):
        n=1+4*(self.prefix.shape[2]-1)
        if self.frames.shape!=(n,256,448,3) or self.past_c2w.shape!=(n,4,4):
            raise ValueError('runtime state native/RGB/camera timelines differ')
        if not self.gauge_id or not torch.isfinite(self.prefix).all():
            raise ValueError('invalid runtime state')
        if self.split not in ('train','development','calibration','test'):
            raise ValueError('unknown split')


class WorldRuntime:
    def __init__(self,model,vae,depth,renderer,steps=30,archive_max=6,*,residency='sequential',guidance=1.,negative_text=None):
        if residency not in ('sequential','generator_vae','pipeline'):raise ValueError('unknown runtime residency policy')
        self.model=model;self.vae=vae;self.depth=depth;self.renderer=renderer
        self.steps=steps;self.archive_max=archive_max;self.residency=residency
        self._past_geometry_cache=None
        self.guidance=guidance;self.negative_text=negative_text

    def keep_generator(self):
        # Gradient-bearing training probes have a larger envelope than sampled
        # rollouts. Preserve their gradients while retaining sequential placement.
        return self.residency in ('generator_vae','pipeline') and not any(p.grad is not None for p in self.model.parameters())

    def prepare_geometry_residency(self):
        if self.renderer is not None:self.renderer.retain_on_device=self.residency=='pipeline' and self.keep_generator()
        if self.keep_generator():self.model.cuda();self.vae.cuda()
        else:self.model.cpu();self.vae.cpu();torch.cuda.empty_cache()

    @torch.no_grad()
    def start(self,anchor_rgb,anchor,K,*,scene_id,stream_id,split):
        from gmlib.models.vae_stream import VaeStream
        self.prepare_geometry_residency();self.vae.cuda()
        rgb=torch.from_numpy(anchor_rgb.copy()).permute(2,0,1).float()/127.5-1
        prefix=VaeStream(self.vae).encode_chunk(rgb[None,:,None].cuda().to(torch.bfloat16)).float().cpu()
        decoder=VaeStream(self.vae,offload_device='cpu');decoder.decode_chunk(prefix.cuda().to(torch.bfloat16))
        state=RuntimeState(prefix,anchor_rgb[None].copy(),torch.eye(4)[None],K.cpu(),
            anchor.depth.cpu(),anchor.confidence.cpu(),anchor.gauge_id,decoder.snapshot(),
            scene_id,stream_id,split)
        state.validate();return state

    @torch.no_grad()
    def prepare(self,state,current_command,*,fixed_archive=None,appearance_error=False,geometry_error=None):
        """No later command, path identifier, cost or future RGB is accepted."""
        from gmlib.models.vae_stream import VaeStream
        from profile_physical_interventions import perturb
        state.validate();command=current_command.detach().float().cpu()
        if command.shape!=(16,4,4) or not torch.isfinite(command).all():
            raise ValueError('exactly the current 16-frame command is required')
        start=state.prefix.shape[2];boundary=len(state.frames)-1
        archive=(select_archive(state.past_c2w,command,state.K,start,self.archive_max)
                 if fixed_archive is None else list(fixed_archive))
        if len(set(archive))!=len(archive) or any(i<=0 or i>=max(1,start-8) for i in archive):
            raise ValueError('noncausal archive choice')
        ids=list(dict.fromkeys([0,*[4*i for i in archive],boundary,*range(max(0,boundary-12),boundary,4)]))
        self.prepare_geometry_residency()
        # Physical 2x2 branches share identical past geometry. Retain one CPU
        # result keyed by actual past content; branch perturbations are applied
        # afterwards and never enter this cache. No current/future RGB is keyed.
        signature=hashlib.sha256(state.frames[ids].tobytes())
        for tensor in (state.past_c2w[ids],state.K,state.anchor_depth,state.anchor_mask):
            signature.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        cache_key=(state.gauge_id,tuple(ids),signature.hexdigest())
        cached=self._past_geometry_cache
        if cached is not None and cached[0]==cache_key:
            depths,masks,depth_receipt=cached[1:]
        elif len(ids)>1:
            self.depth._load()
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(0)
                depths,masks,depth_receipt=causal_depth(self.depth,state.frames[ids],state.K,
                    state.past_c2w[ids],state.anchor_depth,state.anchor_mask,
                    keep_resident=self.residency=='pipeline' and self.keep_generator())
        else:
            depths=state.anchor_depth[None];masks=state.anchor_mask[None];depth_receipt={'anchor_only':True}
        self._past_geometry_cache=(cache_key,depths,masks,depth_receipt)
        sources=[]
        for j,i in enumerate(ids):
            sources.append(dict(frame_id=i,rgb=torch.from_numpy(state.frames[i].copy()).permute(2,0,1).float()/127.5-1,
                depth=state.anchor_depth if i==0 else depths[j],
                confidence=state.anchor_mask if i==0 else masks[j],
                c2w=state.past_c2w[i],gauge_id=state.gauge_id))
        archive_ids={4*i for i in archive}
        if geometry_error not in (None,'depth','source_pose'):raise ValueError('unknown physical error')
        branch=perturb(sources,archive_ids,int(appearance_error),int(geometry_error is not None),geometry_error or 'depth')
        rendered=self.renderer.render(branch,command,state.K,(256,448),seed=0,
            camera_units='anchor_median_depth',gauge_id=state.gauge_id,history_boundary=boundary)
        self.vae.cuda()
        video=rendered['rgb'].permute(1,0,2,3)[None];video=torch.cat([video[:,:,:1],video],2)
        rz=VaeStream(self.vae).encode_chunk(video.cuda().to(torch.bfloat16))[:,:,1:].float().cpu()
        meta=torch.stack([F.interpolate(rendered[k].reshape(4,4,1,256,448).mean(1),size=(16,28),mode='area')[:,0]
                          for k in ('valid','conf')])[None]
        prefix=state.prefix;poses=state.past_c2w.clone()
        if geometry_error=='source_pose':
            # A packed archive latent carries four subframe rays. Apply the same
            # physical pose error to every one of its represented subframes.
            for source,changed in zip(sources,branch):
                if source['frame_id'] in archive_ids:
                    delta=torch.linalg.inv(source['c2w'])@changed['c2w']
                    i=source['frame_id'];poses[i-3:i+1]=poses[i-3:i+1]@delta
        if appearance_error and archive:
            prefix=prefix.clone();pixels=torch.from_numpy(state.frames.copy()).permute(3,0,1,2).float()/127.5-1
            # Re-encode the actual causal RGB prefix with the same corruption on
            # all four frames represented by each archive latent. Only selected
            # archive slots replace native history; context/history stay fixed.
            for pos in archive:
                for i in range(4*pos-3,4*pos+1):
                    rgb=pixels[:,i].clone()
                    s=dict(sources[0],frame_id=4*pos,rgb=rgb)
                    corrupted=perturb([s],archive_ids,1,0,'depth')[0]['rgb']
                    pixels[:,i]=corrupted
            encoder=VaeStream(self.vae);parts=[]
            for lo,hi in [(0,1)]+[(i,min(i+32,pixels.shape[1])) for i in range(1,pixels.shape[1],32)]:
                frames=pixels[None,:,lo:hi]
                parts.append(encoder.encode_chunk(frames.cuda().to(torch.bfloat16)).float().cpu())
            encoded=torch.cat(parts,2);prefix[:,:,archive]=encoded[:,:,archive]
        condition=build_condition(prefix=prefix,observed_c2w=poses,requested_c2w=command,K=state.K,
            gauge_id=state.gauge_id,camera_units='anchor_median_depth',archive_indices=archive,
            render=rz,render_metadata=meta)
        if getattr(self.model,'native_patch_address',False):
            from .correspondence import attach_correspondence
            condition=attach_correspondence(condition,branch,command,state.K)
        return condition,dict(archive_indices=archive,source_ids=ids,retrieval_policy=POLICY,
            depth=depth_receipt,source_pose_origin=state.source_pose_origin,
            appearance_error=appearance_error,geometry_error=geometry_error,
            sources=branch,rendered=rendered)

    def generate(self,state,command,text,seed,*,forced_mode=None,grad=False,prepared=None):
        c,info=self.prepare(state,command) if prepared is None else prepared
        if not self.keep_generator():self.vae.cpu()
        if self.residency!='pipeline' or not self.keep_generator():
            self.depth.offload();self.renderer.release(force=True)
        if not self.keep_generator():torch.cuda.empty_cache()
        self.model.cuda();c=move_condition(c,'cuda');text=text.cuda()
        with torch.no_grad():condition,logits,mode=self.model.route(c,forced_mode)
        noise=torch.randn(1,48,4,16,28,device='cuda',generator=torch.Generator(device='cuda').manual_seed(seed))
        output=sample(self.model,condition,noise,text,self.steps,last_step_grad=grad,guidance=self.guidance,
                      negative_text=None if self.negative_text is None else self.negative_text.to(text.device))
        info.update(mode=mode,router_probabilities=logits.detach().softmax(-1).cpu().tolist())
        return output,info

    @torch.no_grad()
    def advance(self,state,command,generated):
        from gmlib.models.vae_stream import VaeStream
        self.prepare_geometry_residency();self.vae.cuda()
        decoder=VaeStream(self.vae,offload_device='cpu');decoder.restore(state.decoder_prefix)
        frames=decoder.decode_chunk(generated.detach().cuda().to(torch.bfloat16))[0]
        pixels=frames.permute(1,2,3,0).float().cpu().add(1).mul(127.5).clamp(0,255).byte().numpy()
        new=RuntimeState(append_native(state.prefix,generated.detach().cpu()),
            np.concatenate([state.frames,pixels]),torch.cat([state.past_c2w,command.detach().cpu()]),
            state.K,state.anchor_depth,state.anchor_mask,state.gauge_id,decoder.snapshot(),
            state.scene_id,state.stream_id,state.split,state.calls+1,
            'mixed_recorded_prefix_commanded_generated' if state.source_pose_origin=='offline_recorded_annotation' else state.source_pose_origin)
        new.validate();return new

    def two_calls(self,state,commands,text,seeds,*,first_mode=None):
        """Current-state rollout, detached first call, gradient only in second."""
        if len(commands)!=2 or len(seeds)!=2:raise ValueError('exactly two calls required')
        first,_=self.generate(state,commands[0],text,seeds[0],forced_mode=first_mode,grad=False)
        next_state=self.advance(state,commands[0],first)
        second,info=self.generate(next_state,commands[1],text,seeds[1],grad=True)
        return next_state,second,info
