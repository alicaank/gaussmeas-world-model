"""Shared train/runtime archive selection using past poses and current command.

This is a deterministic camera overlap proxy, not a visible/static support label.
The depth band is fixed in anchor units; no whole-trajectory statistic is used.
"""
import torch

POLICY='past_current_anchor_frustum.v1'


def select_archive(observed_c2w, requested_c2w, K, prefix_latents, max_refs=6, size=(256,448)):
    boundary=4*(prefix_latents-1)
    if observed_c2w.shape!=(boundary+1,4,4) or requested_c2w.shape!=(16,4,4):
        raise ValueError('retrieval accepts exactly past cameras and current 16-frame command')
    if not 0<=max_refs<=6:raise ValueError('archive capacity must be 0..6')
    candidates=list(range(1,max(1,prefix_latents-8)))
    if not candidates or max_refs==0:return []
    h,w=size;device=K.device
    yy,xx=torch.meshgrid(torch.linspace(.1*h,.9*h,6,device=device),torch.linspace(.1*w,.9*w,8,device=device),indexing='ij')
    ray=torch.linalg.inv(K.float())@torch.stack([xx,yy,torch.ones_like(xx)]).reshape(3,-1)
    world=[]
    for target in requested_c2w[[3,7,11,15]]:
        for d in (.5,1.,2.,4.):
            world.append(target[:3,:3]@(ray*d)+target[:3,3,None])
    world=torch.cat(world,1)
    def covers(position):
        camera=torch.linalg.inv(observed_c2w[4*position])
        xyz=camera[:3,:3]@world+camera[:3,3,None];uv=K@xyz
        xy=uv[:2]/xyz[2].clamp_min(1e-8)
        return (xyz[2]>.1)&(xy[0]>=0)&(xy[0]<w)&(xy[1]>=0)&(xy[1]<h)
    covered=covers(0)
    for pos in range(max(1,prefix_latents-8),prefix_latents):covered|=covers(pos)
    selected=[];visibility={i:covers(i) for i in candidates}
    for _ in range(min(max_refs,len(candidates))):
        # Earlier index wins exact ties; no padding with redundant sources.
        scored=[(int((visibility[i]&~covered).sum()),-i,i) for i in candidates if i not in selected]
        gain,_,best=max(scored)
        if gain==0:break
        selected.append(best);covered|=visibility[best]
    return sorted(selected)
