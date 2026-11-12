"""Frozen ReSplat RGB-D rendering for causal, matched-source memory studies."""
from __future__ import annotations

import gm_env  # machine paths: gm_env.py at the repository root (scripts/env.sh puts it on PYTHONPATH)
import os
import sys
from pathlib import Path

import torch

from gmlib.models.gate_omega_scaffold import render_point_source
from gmlib.memory.latent_splat import zbuffer_combine


class ReSplatCondition:
    def __init__(self, device="cuda", num_refine=4):
        root = Path(gm_env.RESPLAT)
        sys.path.insert(0, str(root))
        sys.path.insert(0, str(root / 'scripts'))
        # Reuse the locally built CUDA extension without altering the Python env.
        extension = root / 'src/model/encoder/pointops/build/lib.linux-x86_64-cpython-311'
        if extension.is_dir(): sys.path.insert(0, str(extension))
        import infer_colmap
        cwd = os.getcwd()
        try:
            os.chdir(root)
            self.encoder, self.decoder, self.shim = infer_colmap.build_model(
                experiment='dl3dv', checkpoint=str(root / 'pretrained/resplat-base-dl3dv-256x448-view8-1934a04c.pth'),
                num_refine=num_refine, image_shape=(256,448), overrides=[], device=device)
        finally:
            os.chdir(cwd)
        self.device, self.num_refine = device, num_refine
        self.release()

    @torch.no_grad()
    def render(self, sources, target_c2w, K, size, seed=0):
        if len(sources) < 2:
            return point_render(sources, target_c2w, K, size)
        h,w = size
        self.encoder.to(self.device).eval()
        self.decoder.to(self.device).eval()
        kn = K.float().clone(); kn[0] /= w; kn[1] /= h
        n,t = len(sources), len(target_c2w)
        context = dict(image=torch.stack([(s['rgb'].float()+1).clamp(0,2)/2 for s in sources])[None],
                       extrinsics=torch.stack([s['c2w'].float() for s in sources])[None],
                       intrinsics=kn[None,None].repeat(1,n,1,1),near=torch.full((1,n),.01),
                       far=torch.full((1,n),200.), index=torch.arange(n)[None])
        target = dict(image=torch.zeros(1,t,3,h,w),extrinsics=target_c2w.float()[None],
                      intrinsics=kn[None,None].repeat(1,t,1,1),near=torch.full((1,t),.01),
                      far=torch.full((1,t),200.),index=torch.arange(n,n+t)[None])
        def move(x):
            if isinstance(x,torch.Tensor): return x.to(self.device)
            if isinstance(x,dict): return {k:move(v) for k,v in x.items()}
            return x
        batch=self.shim(move(dict(context=context,target=target,scene=['causal-memory-study'])))
        try:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()] if self.device.startswith('cuda') else []):
                torch.manual_seed(seed)
                out=self.encoder(batch['context'],global_step=0,deterministic=False)
                gs=out['gaussians'] if isinstance(out,dict) else out
                cond=out.get('condition_features') if isinstance(out,dict) else None
                if self.num_refine>0 and cond is not None:
                    gs=self.encoder.forward_update(batch['context'],batch['target'],cond,gs,self.decoder,None)['gaussian'][-1]
                dec=self.decoder(gs,batch['target']['extrinsics'],batch['target']['intrinsics'],
                                 batch['target']['near'],batch['target']['far'],(h,w),depth_mode=None)
                rgb=dec.color[0].clamp(0,1)*2-1
                depth=dec.depth[0][:,None]
                alpha=dec.accumulated_alpha[0][:,None]
                valid=((alpha>.5)&(depth>0)&torch.isfinite(depth)&torch.isfinite(rgb).all(1,keepdim=True)).float()
                result=dict(rgb=(rgb*valid).float().cpu(),depth=(depth*valid).float().cpu(),
                            valid=valid.cpu(),conf=(alpha.clamp(0,1)*valid).float().cpu())
            return result
        finally:
            self.release()

    def release(self):
        self.encoder.cpu(); self.decoder.cpu()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def point_render(sources, target_c2w, K, size):
    result={k:[] for k in ('rgb','depth','valid','conf')}
    for pose in target_c2w:
        records=[render_point_source(s['rgb'],s['depth'],s['confidence'],
                 source_c2w=s['c2w'],target_c2w=pose,source_intrinsics=K,
                 target_intrinsics=K,output_size=size) for s in sources]
        if not records:
            h,w=size
            result['rgb'].append(torch.zeros(3,h,w))
            for k in ('depth','valid','conf'): result[k].append(torch.zeros(1,h,w))
            continue
        vals=torch.stack([x['values'] for x in records])
        depths=torch.stack([x['depth'] for x in records])
        masks=torch.stack([x['validity'] for x in records])
        confs=torch.stack([x['confidence'] for x in records])
        # zbuffer_combine expects N,C,H,W source tensors and returns C,H,W.
        fused,valid=zbuffer_combine(vals,masks,depths)
        depth,_=zbuffer_combine(depths,masks,depths)
        conf,_=zbuffer_combine(confs,masks,depths)
        for k,v in [('rgb',fused),('depth',depth),('valid',valid),('conf',conf)]: result[k].append(v[0])
    return {k:torch.stack(v) for k,v in result.items()}


class FixedWorldRender:
    """The existing condition builder's target renderer interface, without GT RGB."""
    def __init__(self, rendered, source_ids):
        self.rendered,self.source_ids=rendered,tuple(source_ids)

    def __call__(self, *, history_boundary, target_c2w, **kwargs):
        assert self.source_ids and max(self.source_ids)<=history_boundary
        assert len(target_c2w)==len(self.rendered['rgb'])
        return dict(frames=[{k:v[i:i+1] for k,v in self.rendered.items()} for i in range(len(target_c2w))],
                    source_frame_ids=self.source_ids,authority=0.0,
                    oldest_age=max(history_boundary-x for x in self.source_ids))


def outside_history_frusta(source, history_c2w, K, size):
    """Conservative recall eligibility: exclude points inside ANY history camera.

    Includes every RGB pose represented by the actual packed history, including
    the anchor. No occlusion test is needed: potentially visible points are
    excluded even if they might have been occluded. Geometry remains estimated.
    """
    h,w=size
    y,x=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
    pixels=torch.stack([x,y,torch.ones_like(x)]).reshape(3,-1).float()
    xyz=(torch.linalg.inv(K.float())@pixels)*source['depth'].reshape(1,-1)
    world=source['c2w'][:3,:3].float()@xyz+source['c2w'][:3,3:4].float()
    visible=torch.zeros(h*w,dtype=torch.bool)
    for pose in history_c2w:
        inv=torch.linalg.inv(pose.float())
        cam=inv[:3,:3]@world+inv[:3,3:4]
        proj=K.float()@cam
        uv=proj[:2]/proj[2:3].clamp_min(1e-8)
        visible|=(cam[2]>0)&(uv[0]>=-4)&(uv[0]<w+4)&(uv[1]>=-4)&(uv[1]<h+4)
    return (~visible).reshape(1,h,w).float()
