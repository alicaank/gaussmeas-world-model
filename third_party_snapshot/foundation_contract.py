"""Shared causal conditioning for recorded training and native-latent inference.

All runtime camera translations and source depths are in frozen anchor-median
depth units. SfM-to-anchor conversion belongs to offline camera annotation;
it is never inferred from future RGB by the runtime condition builder.
"""
from dataclasses import dataclass
import math
import torch
from foundation_model import packed_rays

CONTRACT='foundation.anchor_depth.variable_context.v1'


def normalize_cameras(poses,anchor,units_per_anchor_depth):
    scale=float(units_per_anchor_depth)
    if not math.isfinite(scale) or scale<=0:raise ValueError('invalid camera depth unit')
    if poses.shape[-2:]!=(4,4) or anchor.shape!=(4,4) or not torch.isfinite(poses).all():
        raise ValueError('invalid camera poses')
    result=torch.linalg.inv(anchor.double())@poses.double()
    result[...,:3,3]/=scale
    return result.float()


def fit_registered_scale(registered_anchor_depth,normalized_anchor_depth,valid,min_pixels=64):
    """Offline camera label: registered depth units per normalized anchor unit."""
    a=registered_anchor_depth.double();b=normalized_anchor_depth.double()
    mask=valid.bool()&torch.isfinite(a)&torch.isfinite(b)&(a>0)&(b>0)
    if int(mask.sum())<min_pixels:raise ValueError('insufficient scale correspondence')
    # Robust multiplicative fit; never adjusts origin, rotation or path shape.
    ratios=a[mask]/b[mask];scale=float(ratios.median())
    if not math.isfinite(scale) or scale<=0:raise ValueError('invalid fitted scale')
    residual=float((ratios/scale).log().abs().median())
    return scale,dict(pixels=int(mask.sum()),median_abs_log_scale_residual=residual)


@dataclass
class Condition:
    appearance: torch.Tensor
    rays: torch.Tensor
    metadata: torch.Tensor
    roles: torch.Tensor
    positions: torch.Tensor
    gauge_id: str
    camera_mask: torch.Tensor | None = None
    context_u: torch.Tensor | None = None
    # Sparse (target-patch index, source-token index, confidence) triples.
    # Geometry is detached, past-only, and suppressed in appearance-only mode.
    correspondence: torch.Tensor | None = None

    def pack(self,noisy_target):
        if noisy_target.shape[1:]!=(48,4,*self.appearance.shape[-2:]):raise ValueError('target grid mismatch')
        result=dict(latents=torch.cat([self.appearance,noisy_target],dim=2),rays=self.rays,
            metadata=self.metadata,roles=self.roles,positions=self.positions)
        if self.camera_mask is not None:result['camera_mask']=self.camera_mask
        if self.context_u is not None:result['context_u']=self.context_u
        if self.correspondence is not None:result['correspondence']=self.correspondence
        return result


def build_condition(*,prefix,observed_c2w,requested_c2w,K,gauge_id,
                    camera_units,archive_indices=(),render=None,render_metadata=None):
    """No future RGB argument. Missing history/archive are absent, never copies.

`prefix` is recorded streaming latents in training or retained sampled native
latents at inference. Each continuation appends four native latents; no RGB
decode/re-encode is performed by this interface.
"""
    if camera_units!='anchor_median_depth' or not gauge_id:raise ValueError('explicit anchor gauge required')
    if prefix.ndim!=5 or prefix.shape[:2]!=(1,48) or prefix.shape[2]<1:raise ValueError('invalid prefix')
    start=prefix.shape[2];observed_frames=1+4*(start-1)
    if observed_c2w.shape!=(observed_frames,4,4) or requested_c2w.shape!=(16,4,4):
        raise ValueError('pose counts violate the causal VAE grid')
    if not torch.allclose(observed_c2w[0].float(),torch.eye(4,device=observed_c2w.device),atol=1e-5):
        raise ValueError('anchor pose must be canonical identity')
    history=list(range(max(1,start-8),start));archive=list(archive_indices)
    if len(archive)>6 or len(set(archive))!=len(archive) or any(i<=0 or i>=max(1,start-8) for i in archive):
        raise ValueError('archive must be unique, causal, and outside anchor/recent history')
    future=list(range(start,start+4));parts=[prefix[:,:,:1],prefix[:,:,history],prefix[:,:,archive]]
    roles=[0]+[1]*len(history)+[2]*len(archive);positions=[0,*history,*archive]
    if render is not None:
        if render.shape!=(1,48,4,*prefix.shape[-2:]):raise ValueError('render grid mismatch')
        if render_metadata is None or render_metadata.shape!=(1,2,4,*prefix.shape[-2:]):raise ValueError('render metadata required')
        parts.append(render);roles += [3]*4;positions += future
    elif render_metadata is not None:raise ValueError('metadata without render')
    roles += [4]*4;positions += future
    poses=torch.cat([observed_c2w,requested_c2w]).cpu()
    rays=packed_rays(poses,K.cpu(),positions,height=prefix.shape[-2]*16,width=prefix.shape[-1]*16).to(prefix.device)
    metadata=prefix.new_zeros((1,2,len(roles),*prefix.shape[-2:]))
    if render is not None:metadata[:,:,-8:-4]=render_metadata
    return Condition(torch.cat(parts,dim=2),rays,metadata,
        torch.tensor(roles,device=prefix.device),torch.tensor(positions,device=prefix.device),gauge_id)


def training_input(condition,target,noise,u):
    if target.shape!=noise.shape or not 0<=float(u)<=1:raise ValueError('invalid flow sample')
    return condition.pack((1-u)*target+u*noise),noise-target


def inference_input(condition,current_latents):
    return condition.pack(current_latents)


def append_native(prefix,generated):
    if generated.shape!=(1,48,4,*prefix.shape[-2:]):raise ValueError('each generated continuation has four latents')
    return torch.cat([prefix,generated.detach()],dim=2)
