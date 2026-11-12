"""Past-depth target-to-native-patch addresses, not learned target geometry.

Native video latents span four images. We use the stored endpoint RGB/depth
for each source latent, and vote over all four command poses per target latent.
This is an explicit approximate address, not exact temporal correspondence.
"""
from dataclasses import replace
import torch
from torch.nn import functional as F


@torch.no_grad()
def build_correspondence(condition,sources,command,K,*,stride=8,topk=4,minimum_confidence=.5):
    if command.shape!=(16,4,4) or not torch.isfinite(command).all():
        raise ValueError('only current 16 camera poses are accepted')
    h,w=condition.appearance.shape[-2:];ph,pw=h//2,w//2;patches=ph*pw
    height,width=h*16,w*16
    source_slots={4*int(position):i for i,position in enumerate(condition.positions.tolist())
                  if int(condition.roles[i])==2}
    if not source_slots:return condition.appearance.new_empty((0,3)).cpu()
    if any(int(s['frame_id'])>=4*int(condition.positions[-4])-3 for s in sources):
        raise ValueError('source geometry includes a current/future frame')
    if any(s.get('gauge_id')!=condition.gauge_id for s in sources):raise ValueError('mixed source gauge')
    yy,xx=torch.meshgrid(torch.arange(stride/2,height,stride),torch.arange(stride/2,width,stride),indexing='ij')
    pixels=torch.stack([xx,yy,torch.ones_like(xx)]).reshape(3,-1).float()
    grid=torch.stack([xx/width*2-1,yy/height*2-1],-1)[None]
    local=torch.linalg.inv(K.detach().float().cpu())@pixels
    source_patch=(yy.long()//32*pw+xx.long()//32).flatten()
    points=[];tokens=[];confidence=[]
    for source in sources:
        depth=source['depth'].detach().float().cpu()
        conf=source['confidence'].detach().float().cpu()
        if depth.shape!=(height,width) or conf.shape!=depth.shape:raise ValueError('source grid mismatch')
        z=F.grid_sample(depth[None,None],grid,align_corners=False)[0,0].flatten()
        c=F.grid_sample(conf[None,None],grid,align_corners=False)[0,0].flatten()
        valid=torch.isfinite(z)&torch.isfinite(c)&(z>0)&(c>=minimum_confidence)
        pose=source['c2w'].detach().float().cpu()
        points.append((pose[:3,:3]@(local*z)+pose[:3,3,None])[:,valid])
        slot=source_slots.get(int(source['frame_id']),-1)
        tokens.append((source_patch+slot*patches)[valid] if slot>=0 else torch.full((int(valid.sum()),),-1))
        confidence.append(c[valid].clamp(0,1))
    if not points:return torch.empty(0,3)
    world=torch.cat(points,1);token=torch.cat(tokens);conf=torch.cat(confidence)
    if not world.numel():return torch.empty(0,3)
    # All past surfaces compete for visibility; only native archive winners
    # create addresses. Near/anchor occluders must not be discarded first.
    edges=[];K=K.detach().float().cpu()
    for frame,pose in enumerate(command.detach().float().cpu()):
        camera=pose[:3,:3].T@(world-pose[:3,3,None]);depth=camera[2]
        projected=K@camera;uv=projected[:2]/depth.clamp_min(1e-8)
        valid=(depth>1e-4)&torch.isfinite(uv).all(0)&(uv[0]>=0)&(uv[0]<width)&(uv[1]>=0)&(uv[1]<height)
        index=(uv[1].long()//32*pw+uv[0].long()//32).clamp(0,patches-1)
        front=torch.full((patches,),torch.inf)
        front.scatter_reduce_(0,index[valid],depth[valid],reduce='amin',include_self=True)
        visible=valid&(depth<=front[index]*1.03)&(token>=0)
        if visible.any():
            edges.append(torch.stack([index[visible]+(frame//4)*patches,token[visible],conf[visible]],1))
    if not edges:return torch.empty(0,3)
    all_edges=torch.cat(edges);result=[]
    # At most four archive tokens per target patch; repeat frame/pixel hits do
    # not multiply logit strength. A source token gets its best confidence.
    for query in all_edges[:,0].unique(sorted=True):
        group=all_edges[all_edges[:,0]==query]
        ids,inverse=group[:,1].unique(sorted=True,return_inverse=True)
        weights=torch.zeros_like(ids).scatter_reduce_(0,inverse,group[:,2],reduce='amax',include_self=True)
        order=weights.argsort(descending=True,stable=True)[:topk]
        result.append(torch.stack([query.expand(len(order)),ids[order],weights[order]],1))
    return torch.cat(result).detach()


def attach_correspondence(condition,sources,command,K):
    return replace(condition,correspondence=build_correspondence(condition,sources,command,K))


def attention_prior(edges,roles,patches,camera_mask,device,dtype):
    """Sparse geometry -> target-to-archive bias, after strict index checks."""
    if edges.ndim!=2 or edges.shape[1]!=3 or not torch.isfinite(edges).all():raise ValueError('invalid address edges')
    if not len(edges):return None
    edges=edges.detach().to(device);q,k=edges[:,:2].long().unbind(1)
    if (not torch.equal(edges[:,:2],edges[:,:2].long().to(edges)) or
            (q<0).any() or (q>=4*patches).any() or (k<0).any() or (k>=(len(roles)-4)*patches).any() or
            (edges[:,2]<0).any() or (edges[:,2]>1).any()):raise ValueError('address index/weight outside contract')
    slots=k//patches
    if (roles[slots]!=2).any():raise ValueError('address source must be native archive')
    mask=torch.ones_like(slots,dtype=torch.bool) if camera_mask is None else camera_mask[slots]
    n=len(roles)*patches;prior=torch.zeros((1,1,n,n),device=device,dtype=dtype)
    prior[0,0,(len(roles)-4)*patches+q[mask],k[mask]]=edges[mask,2].to(dtype)
    return prior
