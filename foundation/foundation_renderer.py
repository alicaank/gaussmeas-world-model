"""Published ReSplat weights, supplied per-view depths, canonical camera units.

One source is duplicated with identical RGB, depth, confidence and pose so that
ReSplat receives two views. Duplication does not create independent evidence.
Two or more sources are passed through unchanged. The same
policy is called during offline preparation and deployment. No adapted generator
checkpoint or old-model output is loaded.
"""
import torch
from torch import nn
from torch.nn import functional as F
from gmlib.memory.resplat_condition import ReSplatCondition
from gmlib.models.gate_omega_scaffold import render_point_source


def render_anchor(source,target_c2w,K,size):
    """Legacy point-render diagnostic; not used by FoundationRenderer."""
    depth=source['depth'][None].clone()
    depth[:,source['confidence']<=0]=0
    frames=[render_point_source(source['rgb'],depth,source['confidence'][None],
        source_c2w=source['c2w'],target_c2w=pose,source_intrinsics=K,target_intrinsics=K,output_size=size)
        for pose in target_c2w]
    return {key:torch.stack([frame[field] for frame in frames]) for key,field in
        (('rgb','values'),('depth','depth'),('valid','validity'),('conf','confidence'))}


class PerViewDepth(nn.Module):
    def __init__(self,predictor,sources):
        super().__init__();self.predictor=predictor;self.sources=sources

    def forward(self,*args,**kwargs):
        output=self.predictor(*args,**kwargs);estimates=[]
        for prediction in output['depth_preds']:
            batch,views,h,w=prediction.shape
            if batch!=1 or views!=len(self.sources):raise ValueError('source/depth count mismatch')
            seeded=[]
            for i,source in enumerate(self.sources):
                d=source['depth'].detach().to(prediction)[None,None]
                c=source['confidence'].detach().to(prediction)[None,None]
                d=F.interpolate(d,(h,w),mode='bilinear',align_corners=False)[0,0]
                c=F.interpolate(c,(h,w),mode='nearest')[0,0]
                valid=torch.isfinite(d)&(d>0)&(c>0)
                own=prediction[0,i];overlap=valid&torch.isfinite(own)&(own>0)
                ratio=torch.median(d[overlap]/own[overlap]) if overlap.any() else own.new_tensor(1.)
                seeded.append(torch.where(valid,d,own*ratio))
            estimates.append(torch.stack(seeded)[None])
        output['depth_preds']=estimates
        return output


class FoundationRenderer(ReSplatCondition):
    retain_on_device=False

    def release(self,*,force=False):
        if not self.retain_on_device or force:super().release()

    @torch.no_grad()
    def render(self,sources,target_c2w,K,size,seed=0,*,camera_units,gauge_id,history_boundary):
        if camera_units!='anchor_median_depth' or not gauge_id:raise ValueError('anchor gauge required')
        if not sources or any(s['frame_id']>history_boundary for s in sources):raise ValueError('noncausal sources')
        if any(s.get('gauge_id')!=gauge_id for s in sources):raise ValueError('mixed geometry gauges')
        # Keep the logical source list/IDs intact. No synthetic baseline or
        # camera perturbation: both views refer to the very same observation.
        render_sources=[sources[0],sources[0]] if len(sources)==1 else sources
        predictor=self.encoder.depth_predictor
        self.encoder.depth_predictor=PerViewDepth(predictor,render_sources)
        try:return super().render(render_sources,target_c2w,K,size,seed)
        finally:self.encoder.depth_predictor=predictor
