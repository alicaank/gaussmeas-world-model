"""Frontier (unobserved-region) mask for the ReSplat render memory (9 Nov, user: "model drifts the border instead of generating").

Diagnosis: ReSplat's `valid` is accumulated alpha > .5 and `conf` is alpha (gaussverse/memory/resplat_condition.py:66-70), so a
Gaussian stretched into newly revealed space is reported as valid, confident content; the packed model masks render tokens only
where pooled valid < .5 and otherwise follows the render (~2:1 in conflict), so at the frontier it extrudes the last known surface
instead of generating.

`observed` masks (per target pixel, 1 = some source view actually saw this surface):
- 'reproj' (default): back-project the RENDERED depth (ReSplat dec.depth, gauge units) to 3D, project into every source view; observed
  if it lands inside the source image (margin) and agrees with that source's DA3 depth within a relative tolerance. Dense; a stretched
  Gaussian lands outside every source (or behind/in front of what the source saw) and is marked unobserved.
- 'support': >= 1 source depth point lands on the target cell (renderer864.splat_depths). Sparse under forward motion (points spread
  apart), kept only as a diagnostic (smoke 9 Nov 13:45: speckled mask, 57% of cells dropped on a 5-call jump).
Where unobserved, (valid, conf) are zeroed; rgb is kept (render tokens with pooled valid < .5 are dropped from attention, so the latent
content there is irrelevant; cached training latents stay usable). install() swaps renderer864.render_chunk; render864_rows imports
it at call time, so rollouts pick it up. Training uses the same geometry test with DA3 depth of the recorded target frames
(build_frontier_cache.py).
"""
import json
import time
import numpy as np
import torch
from torch.nn import functional as F

STATE = dict(mode='off', close=1, log=None, calls=0, tol=0.15, margin=2)
LAST = {}


def closing(mask, r):
    """Binary closing (dilate then erode) with a (2r+1)^2 window on (N,h,w) float masks."""
    if r <= 0:
        return mask
    x = mask[:, None]
    x = F.max_pool2d(x, 2 * r + 1, stride=1, padding=r)
    x = -F.max_pool2d(-x, 2 * r + 1, stride=1, padding=r)
    return x[:, 0]


@torch.no_grad()
def observed_from_support(depths, K, c2w_src, target_c2w, size=(120, 216), close=1, out_size=(480, 864)):
    """(n_tgt, H, W) float mask: 1 where >= 1 source's depth points land on the target cell (after closing). Diagnostic."""
    from renderer864 import splat_depths
    D = splat_depths([d.cuda() if torch.cuda.is_available() else d for d in depths], K, c2w_src, target_c2w, size)   # (n_src, n_tgt, h, w)
    n = (D > 0).sum(0).float()
    obs = closing((n > 0).float(), close)
    return F.interpolate(obs[:, None], out_size, mode='nearest')[:, 0].cpu()


@torch.no_grad()
def observed_from_reprojection(depth_t, K, target_c2w, src_depths, c2w_src, tol=0.15, margin=2, out_size=(480, 864)):
    """depth_t (n_t,H,W) target-view depth (gauge units, <= 0 / non-finite = none) at the resolution of K; src_depths list of (H,W)
    source depths at the same resolution; cameras c2w (anchor gauge). Returns (n_t, *out_size) float: 1 where the target surface
    point projects inside some source view and matches that source's depth within tol (relative). Pixels without target depth -> 0."""
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    d = torch.as_tensor(np.asarray(depth_t), dtype=torch.float32, device=dev); n_t, H, W = d.shape
    Kt = torch.as_tensor(np.asarray(K, np.float64), dtype=torch.float32, device=dev); Kinv = torch.linalg.inv(Kt)
    ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32), torch.arange(W, device=dev, dtype=torch.float32), indexing='ij')
    pix = torch.stack([xs, ys, torch.ones_like(xs)]).reshape(3, -1)                    # (3,N)
    rays = Kinv @ pix
    T = torch.as_tensor(np.asarray(target_c2w, np.float64), dtype=torch.float32, device=dev)          # (n_t,4,4)
    S_inv = torch.linalg.inv(torch.as_tensor(np.asarray(c2w_src, np.float64), dtype=torch.float32, device=dev))   # (n_s,4,4)
    src = torch.stack([torch.as_tensor(np.asarray(s), dtype=torch.float32, device=dev) for s in src_depths])     # (n_s,H,W)
    ok_t = torch.isfinite(d) & (d > 0); dd = torch.where(ok_t, d, torch.ones_like(d)).reshape(n_t, 1, -1)
    world = T[:, :3, :3] @ (rays[None] * dd) + T[:, :3, 3:4]                            # (n_t,3,N)
    observed = torch.zeros(n_t, H * W, dtype=torch.bool, device=dev)
    for s in range(src.shape[0]):
        cam = S_inv[s, :3, :3] @ world + S_inv[s, :3, 3:4]                                  # (n_t,3,N)
        z = cam[:, 2]; uv = Kt @ cam; u = uv[:, 0] / uv[:, 2].clamp_min(1e-6); v = uv[:, 1] / uv[:, 2].clamp_min(1e-6)
        inside = (z > 1e-3) & (u >= margin) & (u <= W - 1 - margin) & (v >= margin) & (v <= H - 1 - margin)
        ui = u.round().long().clamp(0, W - 1); vi = v.round().long().clamp(0, H - 1)
        ds = src[s].reshape(-1)[vi * W + ui]                                               # (n_t,N)
        agree = torch.isfinite(ds) & (ds > 0) & ((z - ds).abs() <= tol * torch.maximum(ds, z))
        observed |= inside & agree
    obs = (observed & ok_t.reshape(n_t, -1)).float().reshape(n_t, 1, H, W)
    return F.interpolate(obs, out_size, mode='nearest')[:, 0].cpu()


def apply_mask(rgb, valid, conf, observed, zero_rgb=False):
    """valid/conf -> 0 where not observed (the packed model drops render tokens whose pooled valid < .5, so the token content
    is irrelevant there; partially observed cells keep their render with reduced valid/conf). zero_rgb additionally blanks the
    pixels (not used for training/eval parity: cached training latents are encoded from the unmasked render).
    Returns new tensors and the frontier fraction (valid but unobserved)."""
    obs = observed.to(valid.dtype)
    frontier = float(((valid > .5) & (obs < .5)).float().mean())
    return (rgb * obs[:, None] if zero_rgb else rgb), valid * obs, conf * obs, frontier


def pooled_cells(mask, hw=(30, 54)):
    """(16,H,W) pixel mask -> (4,h,w) per-latent cell fraction (mean over the 4 RGB frames of each latent, area pooled), as the
    training/inference metadata pooling in render864_rows."""
    return F.interpolate(mask.reshape(4, 4, 1, *mask.shape[-2:]).mean(1), size=hw, mode='area')[:, 0]


def frontier_render_chunk(renderer, da3, frames_864, c2w_src, K864, mpu, target_c2w, gauge_id, boundary, size=(540, 960), first_mode='dup2', features=False):
    """renderer864.render_chunk with the observed mask applied (same signature, same returns)."""
    import renderer864 as R
    H, W = size; src = R.upsize(frames_864, size); K = np.asarray(K864, np.float64).copy(); K[0] *= W / 864; K[1] *= H / 480
    depths, masks = R.da3_gauge_depths(da3, src, K, c2w_src, mpu)
    feat = R.render_features([d.cuda() for d in depths], K, c2w_src, target_c2w) if features else None
    sources = [dict(frame_id=i, rgb=torch.from_numpy(src[i].copy()).permute(2, 0, 1).float() / 127.5 - 1, depth=depths[i], confidence=masks[i], c2w=torch.as_tensor(c2w_src[i]).float(), gauge_id=gauge_id) for i in range(len(src))]
    if len(sources) == 1 and first_mode == 'dup4':
        sources = sources * 4
    if len(sources) == 1 and first_mode == 'points':
        pts = [R.warp_points(src[0], depths[0].numpy(), K, np.asarray(c2w_src[0], np.float64), np.asarray(t, np.float64)) for t in target_c2w]
        rgb = torch.from_numpy(np.stack([p[0] for p in pts])).permute(0, 3, 1, 2).float() / 127.5 - 1; val = torch.from_numpy(np.stack([p[1] for p in pts])).float()[:, None]
        rgb = F.interpolate(rgb, (480, 864), mode='bilinear', align_corners=False); val = F.interpolate(val, (480, 864), mode='nearest')
        return (rgb, val[:, 0], val[:, 0].clone()) + ((feat,) if features else ())
    out = renderer.render(sources, torch.as_tensor(target_c2w).float(), torch.as_tensor(K).float(), (H, W), seed=0, camera_units='anchor_median_depth', gauge_id=gauge_id, history_boundary=boundary)
    rgb = F.interpolate(out['rgb'], (480, 864), mode='bilinear', align_corners=False)
    valid = F.interpolate(out['valid'], (480, 864), mode='nearest')[:, 0]; conf = F.interpolate(out['conf'], (480, 864), mode='bilinear', align_corners=False)[:, 0]
    rec = dict(t=time.time(), sources=len(sources), boundary=int(boundary), valid_before=round(float(valid.mean()), 4), mode=STATE['mode'])
    if STATE['mode'] in ('support', 'reproj'):
        if STATE['mode'] == 'reproj':
            observed = observed_from_reprojection(out['depth'][:, 0], K, target_c2w, depths, c2w_src, tol=STATE['tol'], margin=STATE['margin'])
        else:
            observed = observed_from_support(depths, K, c2w_src, target_c2w, close=STATE['close'])
        rgb, valid, conf, frontier = apply_mask(rgb, valid, conf, observed)
        rec.update(observed=round(float(observed.mean()), 4), valid_after=round(float(valid.mean()), 4), frontier=round(frontier, 4),
                   cells_dropped=round(float((pooled_cells(observed) < .5).float().mean()), 4))
        LAST.update(observed=observed, valid_before=rec['valid_before'], depth=out['depth'][:, 0], src_depths=depths, K=K)
        if STATE.get('dump'):
            dump_call(STATE['dump'], STATE['calls'], rgb, out['valid'][:, 0] if out['valid'].ndim == 4 else out['valid'], observed, frames_864)
    STATE['calls'] += 1
    if STATE['log']:
        with open(STATE['log'], 'a') as f:
            f.write(json.dumps(rec) + '\n')
    return (rgb, valid, conf) + ((feat,) if features else ())


def dump_call(dump_dir, call, rgb, valid, observed, sources):
    """Diagnostic sheet per render call: rows = render | alpha valid | observed, columns = target frames 0/8/15; plus the source frames."""
    from pathlib import Path
    from PIL import Image
    d = Path(dump_dir); d.mkdir(parents=True, exist_ok=True); tw, th = 288, 160
    def img(x):
        if x.ndim == 3: return ((x.permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5).round().byte().numpy()
        return (x.clamp(0, 1) * 255).round().byte()[:, :, None].expand(-1, -1, 3).numpy()
    picks = [0, 8, 15]; vh = F.interpolate(valid[None].float(), (480, 864), mode='nearest')[0]
    sheet = Image.new('RGB', (tw * (len(picks) + 1), th * 3), 'black')
    for j, k in enumerate(picks):
        for i, x in enumerate((img(rgb[k]), img(vh[k]), img(observed[k]))):
            sheet.paste(Image.fromarray(x).resize((tw, th)), (j * tw, i * th))
    for i, s in enumerate(list(sources)[:3]):
        sheet.paste(Image.fromarray(np.asarray(s)).resize((tw, th)), (len(picks) * tw, i * th))
    sheet.save(d / f'call_{call:03d}.jpg', quality=85)


def install(mode='reproj', close=1, log=None, tol=0.15, margin=2, dump=None):
    import renderer864
    STATE.update(mode=mode, close=int(close), log=str(log) if log else None, tol=float(tol), margin=int(margin), dump=str(dump) if dump else None)
    renderer864.render_chunk = frontier_render_chunk
    return STATE
