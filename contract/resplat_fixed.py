"""ReSplat renderers with the three wrapper errors of the 864 lineage removed (10 Nov 19:30; probe + code reading of cvg/resplat):
 1 resplat_condition.render decodes (540, 960) although the patch shim cropped context AND target to 512 rows and rescaled fy -> x1.055 vertical stretch;
 2 foundation_renderer.PerViewDepth interpolates the full 540-row DA3 depth onto the cropped 512-row grid -> depth squeezed against the image;
 3 the 16-view 540x960 weights were run with overrides=[] and num_refine=4; their preset (infer_colmap.MODEL_PRESETS['dl3dv_16v_540x960']) needs
   gaussian_adapter.gaussian_scale_max=3, depth_pred_half_res=true, num_refine=2 and the authors evaluate with no_crop_image=true.
 make('v8')   8-view weights, 512x960 (multiple of 64: no crop; preset = defaults)          -> (renderer, (512, 960))
 make('v16s') 16-view weights at 512x960, STOCK wrong settings (isolates errors 1-2)          -> (renderer, (512, 960))
 make('v16p') 16-view weights, official preset, no_crop_image=true, 540x960                   -> (renderer, (540, 960))"""
import os, sys
import renderer864 as R
from foundation_renderer import FoundationRenderer


class RendererPreset(FoundationRenderer):
    def __init__(self, preset='dl3dv_16v_540x960', image_shape=(540, 960), device='cuda', extra=('model.encoder.no_crop_image=true',)):
        root = R.ROOT; sys.path.insert(0, str(root)); sys.path.insert(0, str(root / 'scripts'))
        ext = root / 'src/model/encoder/pointops/build/lib.linux-x86_64-cpython-311'
        if ext.is_dir(): sys.path.insert(0, str(ext))
        import infer_colmap
        p = infer_colmap.MODEL_PRESETS[preset]; cwd = os.getcwd()
        try:
            os.chdir(root)
            self.encoder, self.decoder, self.shim = infer_colmap.build_model(experiment='dl3dv', checkpoint=str(root / p['checkpoint']), num_refine=p['num_refine'],
                                                                              image_shape=tuple(image_shape), overrides=[*p['overrides'], *extra], device=device)
        finally:
            os.chdir(cwd)
        self.device, self.num_refine, self.image_shape = device, p['num_refine'], tuple(image_shape); self.preset = dict(preset=preset, overrides=[*p['overrides'], *extra], num_refine=p['num_refine'])
        self.release()


def make(kind):
    if kind == 'v8': return R.Renderer864((512, 960)), (512, 960)
    if kind == 'v16s': return R.Renderer864((512, 960), ckpt_shape=(540, 960)), (512, 960)
    if kind == 'v16p': return RendererPreset(), (540, 960)
    if kind == 'v16p512': return RendererPreset(image_shape=(512, 960), extra=()), (512, 960)     # official preset, 512 rows (no crop needed; 540 without crop fails: 34 vs 33 rows)
    raise ValueError(kind)
