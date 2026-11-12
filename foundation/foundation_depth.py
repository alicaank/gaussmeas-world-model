"""Causal DA3 depths tied to the persistent observed anchor, including two views.

No Umeyama pose fit is attempted on short/stationary trajectories. Requested
poses remain unchanged. Only one depth multiplier is estimated from the same
anchor pixels; depth disagreement remains a measured limitation, not truth.
"""
import numpy as np
import torch
from gaussverse.training.gate_omega_recovery import _pad_depth,_resize_mask
from foundation_contract import fit_registered_scale


@torch.no_grad()
def causal_depth(predictor,images,K,c2w,anchor_depth,anchor_mask,*,keep_resident=False):
    if len(images)!=len(c2w) or len(images)<2:raise ValueError('need two or more causal source views')
    try:
        prediction=predictor._load().inference(image=list(images),
            extrinsics=torch.linalg.inv(c2w).float().numpy(),
            intrinsics=np.repeat(K.numpy()[None],len(images),0),
            align_to_input_extrinsics=False,infer_gs=False,process_res=max(images.shape[1:3]),
            process_res_method='upper_bound_resize',reorder_cam_token_by_reference=True,
            export_dir=None,export_format='mini_npz')
        size=tuple(images.shape[1:3]);depth=torch.stack([_pad_depth(d,size) for d in prediction.depth])
        sky=getattr(prediction,'sky',None)
        masks=torch.ones_like(depth) if sky is None else torch.stack([1-_resize_mask(v,size).clamp(0,1) for v in sky])
        scale,fit=fit_registered_scale(depth[0],anchor_depth,(masks[0]>.5)&(anchor_mask>.5))
        return depth/scale,masks,dict(depth_units_per_anchor_depth=scale,**fit)
    finally:
        if not keep_resident:predictor.offload()
