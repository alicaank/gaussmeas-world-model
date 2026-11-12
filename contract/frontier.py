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

STATE = dict(mode='off', close=1, log=None, calls=0, tol=0.15, margin=2, alpha_min=0.5, conf_one=False, render_dump=None, alpha_gamma=0., comp=False, render_fix='', size=None, pad_to=0)
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
def support_density(depths, K, c2w_src, target_c2w, size=(120, 216)):
    """Support density (11 Nov critique: stretched splats are marked valid; the frustum test does not see them). Every source PIXEL (full source resolution) is
    dropped as a depth point into each target view; per target cell the number of landed points / the nominal number of source pixels per cell, MAX over the
    sources -> (n_tgt, h, w), ~1 on a surface seen at the same scale, << 1 where the render is filled by a few elongated Gaussians (nothing lands there).
    Occluded surfaces also land (no z-test): over-counts behind foreground, never under-counts a seen surface."""
    h, w = size; dev = 'cuda' if torch.cuda.is_available() else 'cpu'; n_t = len(target_c2w)
    T = torch.as_tensor(np.asarray(target_c2w, np.float64), dtype=torch.float32, device=dev); Tinv = torch.linalg.inv(T); best = torch.zeros((n_t, h * w), device=dev)
    for i, d in enumerate(depths):
        d = torch.as_tensor(np.asarray(d), dtype=torch.float32, device=dev); Hs, Ws = d.shape; Kf = torch.as_tensor(np.asarray(K, np.float64), dtype=torch.float32, device=dev)
        ys, xs = torch.meshgrid(torch.arange(Hs, device=dev, dtype=torch.float32), torch.arange(Ws, device=dev, dtype=torch.float32), indexing='ij')
        dd = d.reshape(-1); ok = torch.isfinite(dd) & (dd > 0); pts = (torch.linalg.inv(Kf) @ torch.stack([xs, ys, torch.ones_like(xs)]).reshape(3, -1)) * dd[None]
        S = torch.as_tensor(np.asarray(c2w_src[i], np.float64), dtype=torch.float32, device=dev); world = S[:3, :3] @ pts + S[:3, 3:4]; Kc = Kf.clone(); Kc[0] *= w / Ws; Kc[1] *= h / Hs
        for t in range(n_t):                                                                                   # per target: keeps the peak memory at one (3, Hs Ws) tensor
            cam = Tinv[t, :3, :3] @ world + Tinv[t, :3, 3:4]; z = cam[2]; uv = Kc @ cam; u = (uv[0] / z.clamp_min(1e-9)).floor(); v = (uv[1] / z.clamp_min(1e-9)).floor()
            m = ok & (z > 1e-3) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            cnt = torch.zeros(h * w, device=dev).scatter_add_(0, (v[m] * w + u[m]).long(), torch.ones(int(m.sum()), device=dev))
            best[t] = torch.maximum(best[t], cnt / (Hs * Ws / (h * w)))
    return best.reshape(n_t, h, w).cpu()


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


def lap_var(rgb_uint8):
    """Laplacian variance (sharpness) of an (H,W,3) uint8/float frame, as the eval's `sharp` metric."""
    from scipy.ndimage import laplace
    g = np.asarray(rgb_uint8, np.float32).mean(-1)
    return float(laplace(g).var())


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


def _lap_energy(x, m):
    """Mean squared Laplacian of the luminance of (3,H,W) in [-1,1] over the 5x5-eroded mask m (1,H,W)."""
    g = x.float().mean(0)[None, None]; k = torch.tensor([[0., 1, 0], [1, -4, 1], [0, 1, 0]], device=g.device)[None, None]
    e = -F.max_pool2d(-m.float()[None], 5, 1, 2)[0, 0] > .5
    return float(F.conv2d(g, k, padding=1)[0, 0][e].pow(2).mean()) if e.float().mean() > .05 else float('nan')


def composite_sources(rr, sources, fused):
    """Priority compositing instead of fusion (10 Nov evening: the fused render keeps .1-.45 of its source's Laplacian energy; a single real
    source keeps .7 at call 1; identity test: one source .3-.9, two sources .11-.51). Each source is rendered ALONE; the last generated
    latent (index 1) wins, holes are filled from s-2, then the anchor, then the bank view. valid = union, conf / depth from the winner."""
    order = [1] + list(range(2, min(3, len(sources)))) + [0] + list(range(3, len(sources))); outs = [rr([sources[i]]) for i in order]
    taken = torch.zeros_like(outs[0]['valid']); acc = {k: torch.zeros_like(outs[0][k]) for k in ('rgb', 'conf', 'depth')}; share = []
    for o in outs:
        add = (o['valid'] > .5).to(taken.dtype) * (1 - taken); share.append(round(float(add.mean()), 4)); taken = taken + add
        for k in acc: acc[k] = acc[k] + o[k] * add
    if STATE.get('comp') == 2:                                                                     # hybrid (critique 11 Nov): compositing where any single source covers, fused fill elsewhere
        add = (fused['valid'] > .5).to(taken.dtype) * (1 - taken); share.append(round(float(add.mean()), 4)); taken = taken + add
        for k in acc: acc[k] = acc[k] + fused[k] * add
    m = (taken[0] > .5) & (fused['valid'][0] > .5)
    info = dict(order=order, share=share, lap_fused=round(_lap_energy(fused['rgb'][0], m), 6), lap_comp=round(_lap_energy(acc['rgb'][0], m), 6), lap_source=round(_lap_energy(sources[1]['rgb'], m), 6),
                valid_fused=round(float(fused['valid'].mean()), 4), valid_comp=round(float(taken.mean()), 4))
    return dict(fused, rgb=acc['rgb'], valid=taken, conf=acc['conf'], depth=acc['depth']), info


def fix_render(out, ref, mode):
    """Render matched to its last source frame before the model sees it (critique 10 Nov): per-channel mean / std over the valid pixels of the
    first target (one frame after the source view) -> one affine map for the 16 targets ('tone'); 'full' adds an unsharp mask whose amount
    brings the first target's Laplacian energy to the source's (bisection, amount <= 3). Direct test of 'snap and darkening come from the
    soft dark render'; risk: the model was trained on soft renders - watch the sharpness ratio."""
    rgb = out['rgb'].float(); v = out['valid'].to(rgb.dtype); m = v[0] > .5; ref = ref.to(rgb.device).float(); info = dict(mode=mode)
    if float(m.float().mean()) < .05: return out, dict(info, skipped=True)
    mm = m[0]; mr = torch.stack([rgb[0, c][mm].mean() for c in range(3)]); sr = torch.stack([rgb[0, c][mm].std() for c in range(3)]).clamp_min(1e-3)
    ms = torch.stack([ref[c][mm].mean() for c in range(3)]); ss = torch.stack([ref[c][mm].std() for c in range(3)]); gain = (ss / sr).clamp(.8, 1.5)
    x = (rgb - mr[None, :, None, None]) * gain[None, :, None, None] + ms[None, :, None, None]
    info.update(gain=[round(float(g), 3) for g in gain], shift=[round(float(a - b) * 127.5, 1) for a, b in zip(ms, mr)], lap_before=round(_lap_energy(rgb[0], m), 6), lap_source=round(_lap_energy(ref, m), 6))
    if mode == 'full':
        hp = x - F.avg_pool2d(x, 5, 1, 2, count_include_pad=False); lo, hi = 0., 3.
        if _lap_energy((x + hi * hp)[0], m) < info['lap_source']: lo = hi
        else:
            for _ in range(12):
                mid = (lo + hi) / 2
                if _lap_energy((x + mid * hp)[0], m) < info['lap_source']: lo = mid
                else: hi = mid
        x = x + lo * hp; info['amount'] = round(lo, 3)
    x = x.clamp(-1, 1) * v; info['lap_after'] = round(_lap_energy(x[0], m), 6)
    return dict(out, rgb=x.to(out['rgb'].dtype)), info


def frontier_render_chunk(renderer, da3, frames_864, c2w_src, K864, mpu, target_c2w, gauge_id, boundary, size=(540, 960), first_mode='dup2', features=False):
    """renderer864.render_chunk with the observed mask applied (same signature, same returns)."""
    import renderer864 as R
    if STATE.get('size'): size = tuple(STATE['size'])                                              # 10 Nov evening: 512x960 for the 8-view weights
    H, W = size; src = R.upsize(frames_864, size); K = np.asarray(K864, np.float64).copy(); K[0] *= W / 864; K[1] *= H / 480
    depths, masks = R.da3_gauge_depths(da3, src, K, c2w_src, mpu)
    feat = R.render_features([d.cuda() for d in depths], K, c2w_src, target_c2w) if features else None
    sources = [dict(frame_id=i, rgb=torch.from_numpy(src[i].copy()).permute(2, 0, 1).float() / 127.5 - 1, depth=depths[i], confidence=masks[i], c2w=torch.as_tensor(c2w_src[i]).float(), gauge_id=gauge_id) for i in range(len(src))]
    if STATE.get('pad_to', 0) > len(sources):                                                      # 10 Nov evening (user): the weights expect 8 / 16 views; anchor copies
        sources = sources + [dict(sources[0]) for _ in range(STATE['pad_to'] - len(sources))]      # fill the list until enough generated views exist (same frame, depth, pose)
    if len(sources) == 1 and first_mode == 'dup4':
        sources = sources * 4
    if len(sources) == 1 and first_mode == 'points':
        pts = [R.warp_points(src[0], depths[0].numpy(), K, np.asarray(c2w_src[0], np.float64), np.asarray(t, np.float64)) for t in target_c2w]
        rgb = torch.from_numpy(np.stack([p[0] for p in pts])).permute(0, 3, 1, 2).float() / 127.5 - 1; val = torch.from_numpy(np.stack([p[1] for p in pts])).float()[:, None]
        rgb = F.interpolate(rgb, (480, 864), mode='bilinear', align_corners=False); val = F.interpolate(val, (480, 864), mode='nearest')
        return (rgb, val[:, 0], val[:, 0].clone()) + ((feat,) if features else ())
    rr = lambda ss: renderer.render(ss, torch.as_tensor(target_c2w).float(), torch.as_tensor(K).float(), (H, W), seed=0, camera_units='anchor_median_depth', gauge_id=gauge_id, history_boundary=boundary)
    out = rr(sources); extra = {}
    if STATE.get('comp') and len(sources) > 1:
        out, extra['comp'] = composite_sources(rr, sources, out)
    if STATE.get('render_fix'):
        out, extra['render_fix'] = fix_render(out, sources[1 if len(sources) > 1 else 0]['rgb'], STATE['render_fix'])
    if STATE.get('alpha_gamma', 0) > 0:                                                            # 10 Nov: alpha-normalised colour, c / alpha^gamma on valid pixels (gamma 1 =
        al = out['conf'].clamp(.5, 1.); col = ((out['rgb'] + 1) / 2 / al ** STATE['alpha_gamma']).clamp(0, 1)   # plain un-premultiplication; the identity render fits gamma ~ .45:
        out = dict(out, rgb=(col * 2 - 1) * out['valid'])                                           # alpha .6 -> x.80, alpha .8 -> x.92 of the source luminance)
    rgb = F.interpolate(out['rgb'], (480, 864), mode='bilinear', align_corners=False)
    valid = F.interpolate(out['valid'], (480, 864), mode='nearest')[:, 0]; conf = F.interpolate(out['conf'], (480, 864), mode='bilinear', align_corners=False)[:, 0]
    rec = dict(t=time.time(), sources=len(sources), boundary=int(boundary), valid_before=round(float(valid.mean()), 4), mode=STATE['mode'],
               render_sharp=[round(lap_var(((rgb[k].permute(1, 2, 0) + 1) * 127.5).numpy()), 1) for k in (0, 15)],              # 9 Nov diag: sharpness of the render (first/last target)
               source_sharp=[round(lap_var(np.asarray(fr)), 1) for fr in frames_864])                                               # and of each source frame fed to ReSplat
    rec['render_vs_source_mad'] = [round(float(np.abs(((rgb[0].permute(1, 2, 0) + 1) * 127.5).numpy()[::4, ::4].mean(-1) - np.asarray(fr, np.float32)[::4, ::4].mean(-1)).mean()), 1) for fr in frames_864]   # seam diag: first rendered target vs each source frame (the last generated frame is one of the sources)
    rec.update(extra)
    LAST.update(raw_valid=valid.clone(), raw_conf=conf.clone())                                    # before any mask (cache builders compare with the stored metadata)
    if STATE.get('alpha_min', .5) > .5:                                                            # 10 Nov mask v2: pixels with accumulated alpha below alpha_min are not
        keep = (conf >= STATE['alpha_min']).to(valid.dtype)                                        # valid (identity render: alpha .5-.7 is 18-42 luminance levels too dark);
        rec['alpha_dropped'] = round(float((valid * (1 - keep)).mean()), 4)                        # conf = alpha * valid, so this is a threshold on alpha itself
        valid = valid * keep; conf = conf * keep
    if STATE.get('conf_one'):                                                                      # 10 Nov test: confidence = 1 on valid pixels (alpha is low on
        conf = valid.clone(); rec['conf_one'] = True                                               # textureless surfaces and flickers frame to frame)
    if STATE['mode'] == 'density':
        dens = support_density(depths, K, c2w_src, target_c2w); up = lambda x: F.interpolate(x[:, None], (480, 864), mode='nearest')[:, 0]
        was = valid > .5; rec['density_dropped'] = {str(th): round(float(((up((dens < th).float()) > .5) & was).float().mean() / max(float(was.float().mean()), 1e-6)), 4) for th in (.05, .1, .15, .25, .5)}
        observed = up(closing((dens >= STATE.get('support_min', .15)).float(), STATE['close']))
        rgb, valid, conf, frontier = apply_mask(rgb, valid, conf, observed)
        rec.update(observed=round(float(observed.mean()), 4), valid_after=round(float(valid.mean()), 4), frontier=round(frontier, 4), support_min=STATE.get('support_min', .15),
                   density_median=round(float(dens.median()), 3), cells_dropped=round(float((pooled_cells(observed) < .5).float().mean()), 4))
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
    if STATE.get('render_dump'):                                                                   # render video for flicker analysis: 216x120 rgb + valid/conf
        from pathlib import Path
        d = Path(STATE['render_dump']); d.mkdir(parents=True, exist_ok=True); sm = lambda x: F.interpolate(x, (120, 216), mode='area')
        np.savez_compressed(d / f"call_{STATE['calls']:03d}.npz", rgb=((sm(rgb).clamp(-1, 1) + 1) * 127.5).round().byte().numpy(), valid=sm(valid[:, None].float())[:, 0].half().numpy(),
                            conf=sm(conf[:, None].float())[:, 0].half().numpy(), sources=len(sources))
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


def install(mode='reproj', close=1, log=None, tol=0.15, margin=2, dump=None, alpha_min=0.5, conf_one=False, render_dump=None, alpha_gamma=0., comp=False, render_fix='', size=None, pad_to=0, support_min=.15):
    import renderer864
    STATE.update(comp=int(comp), render_fix=str(render_fix or ''), size=tuple(size) if size else None, pad_to=int(pad_to), support_min=float(support_min))
    STATE.update(mode=mode, close=int(close), log=str(log) if log else None, tol=float(tol), margin=int(margin), dump=str(dump) if dump else None, alpha_min=float(alpha_min), conf_one=bool(conf_one), render_dump=str(render_dump) if render_dump else None, alpha_gamma=float(alpha_gamma))
    renderer864.render_chunk = frontier_render_chunk
    return STATE
