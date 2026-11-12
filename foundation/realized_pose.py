"""Command-free realized-pose estimate for generated frames, in our anchor-depth gauge.

DA3 (multi-view, intrinsics given, NO extrinsics given) on [anchor + a window of earlier frames + query frames].
Poses are re-expressed relative to the anchor view; the global scale is fixed once per call by matching DA3's anchor
depth to the stored anchor depth (median 1 = our gauge). No command, no trajectory-length normalization.
"""
import gm_env  # machine paths: gm_env.py at the repository root (scripts/env.sh puts it on PYTHONPATH)
import numpy as np, torch

DA3_SRC = gm_env.DA3_SRC
DA3_PATH = (f'{gm_env.HOME}/.cache/huggingface/hub/models--depth-anything--DA3NESTED-GIANT-LARGE-1.1/snapshots/'
            'b2359bdf726fb44ef62acca04d629dcf158053e7')


def load_da3(device='cuda'):
    import sys
    if DA3_SRC not in sys.path:
        sys.path.insert(0, DA3_SRC)
    from depth_anything_3.api import DepthAnything3
    return DepthAnything3.from_pretrained(DA3_PATH).to(device).eval()


def _pad44(e):
    e = np.asarray(e, dtype=np.float64)
    if e.shape[-2] == 3:
        out = np.tile(np.eye(4), (e.shape[0], 1, 1)); out[:, :3, :4] = e; return out
    return e


@torch.no_grad()
def realized_poses(da3, anchor_rgb, anchor_depth, frames, K, max_views=16, tail=4):
    """anchor_rgb/frames uint8 448x256; returns c2w (len(frames),4,4) in the anchor-depth gauge + diagnostics."""
    views = [anchor_rgb] + list(frames)
    if len(views) > max_views:          # keep anchor, the query tail, and evenly spaced earlier context
        tl = list(frames[-tail:]); ctx = list(frames[:-tail])
        pick = np.linspace(0, len(ctx) - 1, max_views - 1 - len(tl)).round().astype(int) if ctx and max_views - 1 - len(tl) > 0 else []
        keep_ctx = [ctx[i] for i in pick]
        views = [anchor_rgb] + keep_ctx + tl
        index_map = [None] * (len(frames) - tail) + list(range(len(views) - tail, len(views)))
    else:
        index_map = list(range(1, len(views)))
    Ks = np.repeat(np.asarray(K, dtype=np.float32)[None], len(views), 0)
    pred = da3.inference(image=views, intrinsics=Ks, extrinsics=None, align_to_input_extrinsics=False,
                         align_to_input_ext_scale=False, infer_gs=False, process_res=504, export_dir=None)
    w2c = _pad44(pred.extrinsics)
    rel = np.stack([w2c[0] @ np.linalg.inv(w2c[i]) for i in range(len(views))])     # c2w relative to anchor view
    d0 = np.asarray(pred.depth[0], dtype=np.float64)
    ours = torch.nn.functional.interpolate(torch.as_tensor(anchor_depth if anchor_depth is not None else d0)[None, None].float(), size=d0.shape,
                                           mode='bilinear', align_corners=False)[0, 0].numpy()
    ok = np.isfinite(d0) & (d0 > 0) & np.isfinite(ours) & (ours > 0)
    scale = float(np.median(ours[ok] / d0[ok])) if anchor_depth is not None else 1.0      # None: keep DA3's own (metric) units
    rel[:, :3, 3] *= scale
    out = [None if j is None else rel[j] for j in index_map]
    return out, dict(scale=scale, views=len(views))
