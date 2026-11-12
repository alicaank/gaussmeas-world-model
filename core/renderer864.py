"""ReSplat at 540x960 (zoo weights resplat-base-dl3dv-540x960-view16) for the 864x480 corpus: frames are resized back to
960x540 (the prep resized 960x533 -> 864x480), intrinsics scaled accordingly, per-view depths come from DA3 METRIC depth
divided by the stream's metres_per_anchor_unit (gauge units; no scale fit needed), fed through FoundationRenderer's
PerViewDepth seeding. Output rendered RGB/valid/conf at 864x480."""
import gm_env  # machine paths: gm_env.py at the repository root (scripts/env.sh puts it on PYTHONPATH)
import os, sys
from pathlib import Path
import numpy as np, torch
from torch.nn import functional as F
from gmlib.memory.resplat_condition import ReSplatCondition
from foundation_renderer import FoundationRenderer, PerViewDepth

ROOT = Path(gm_env.RESPLAT)
CKPTS = {(256, 448): 'resplat-base-dl3dv-256x448-view8-1934a04c.pth', (540, 960): 'resplat-base-dl3dv-540x960-view16-a72dc6d0.pth', (512, 960): 'resplat-base-dl3dv-512x960-view8-8179ed87.pth'}


class ReSplatAt(ReSplatCondition):
    """ReSplatCondition with a selectable checkpoint / image shape."""

    def __init__(self, image_shape=(540, 960), device='cuda', num_refine=4, ckpt_shape=None):
        ckpt_shape = tuple(ckpt_shape) if ckpt_shape else (tuple(image_shape) if tuple(image_shape) in CKPTS else (256, 448))   # low-res runs use the 256x448 weights
        sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
        ext = ROOT / 'src/model/encoder/pointops/build/lib.linux-x86_64-cpython-311'
        if ext.is_dir():
            sys.path.insert(0, str(ext))
        import infer_colmap
        cwd = os.getcwd()
        try:
            os.chdir(ROOT)
            self.encoder, self.decoder, self.shim = infer_colmap.build_model(experiment='dl3dv', checkpoint=str(ROOT / 'pretrained' / CKPTS[ckpt_shape]),
                                                                              num_refine=num_refine, image_shape=tuple(image_shape), overrides=[], device=device)
        finally:
            os.chdir(cwd)
        self.device, self.num_refine, self.image_shape = device, num_refine, tuple(image_shape); self.release()


class Renderer864(FoundationRenderer):
    """FoundationRenderer semantics (PerViewDepth seeding, single-source duplication) on ReSplatAt(540x960)."""

    def __init__(self, image_shape=(540, 960), num_refine=4, ckpt_shape=None):
        ReSplatAt.__init__(self, image_shape, num_refine=num_refine, ckpt_shape=ckpt_shape)


def da3_gauge_depths(da3, frames_uint8, K, c2w, mpu):
    """Metric DA3 depth of each frame (with the known cameras) divided by metres/unit -> gauge units. frames at the DA3 input size."""
    single = len(frames_uint8) < 2                                                 # DA3's Umeyama alignment needs >= 2 views; one view = its own metric depth
    kw = dict(image=list(frames_uint8), intrinsics=np.repeat(np.asarray(K, np.float32)[None], len(frames_uint8), 0), align_to_input_ext_scale=False,
              infer_gs=False, process_res=max(frames_uint8[0].shape[:2]), export_dir=None)
    try:
        pred = da3.inference(extrinsics=None if single else np.linalg.inv(np.asarray(c2w, np.float64)).astype(np.float32), align_to_input_extrinsics=not single, **kw)
    except Exception as e:                                                         # 10-22: near-coincident source cameras -> degenerate Umeyama; use metric depth unaligned
        if 'Umeyama' not in str(e) and 'Degenerate' not in str(e):
            raise
        pred = da3.inference(extrinsics=None, align_to_input_extrinsics=False, **kw)
    depths = [torch.as_tensor(np.asarray(d, np.float32)) for d in pred.depth]
    size = frames_uint8[0].shape[:2]
    depths = [F.interpolate(d[None, None], size, mode='bilinear', align_corners=False)[0, 0] / mpu if d.shape != size else d / mpu for d in depths]
    sky = getattr(pred, 'sky', None)
    masks = [torch.ones(size) if sky is None else 1 - F.interpolate(torch.as_tensor(np.asarray(s, np.float32))[None, None], size, mode='bilinear', align_corners=False)[0, 0].clamp(0, 1) for s in (sky if sky is not None else depths)]
    return depths, masks


def upsize(frames_864, size=(540, 960)):
    x = torch.from_numpy(np.asarray(frames_864)).permute(0, 3, 1, 2).float()
    return F.interpolate(x, size, mode='bicubic', align_corners=False).clamp(0, 255).round().byte().permute(0, 2, 3, 1).numpy()


def warp_points(rgb, depth, K, src_c2w, dst_c2w, radius=1):
    """z-buffered depth point-cloud splat of one view into dst_c2w (copy of memorizon_plugins/gv_memorizon2.warp_frame);
    uncovered pixels are zeroed. Returns (H,W,3) uint8 and the covered mask."""
    Hh, Ww = depth.shape; ys, xs = np.mgrid[0:Hh, 0:Ww]
    pix = np.stack([xs, ys, np.ones_like(xs)], 0).reshape(3, -1).astype(np.float64); pts = (np.linalg.inv(K) @ pix) * depth.reshape(1, -1)
    rel = np.linalg.inv(dst_c2w) @ src_c2w; cam = rel[:3, :3] @ pts + rel[:3, 3:4]; z = cam[2]; uv = K @ cam
    with np.errstate(divide='ignore', invalid='ignore'):
        projected = uv[:2] / uv[2:3]
    valid = (depth.reshape(-1) > 0) & np.isfinite(depth.reshape(-1)) & (z > 1e-3) & np.isfinite(z) & np.isfinite(projected).all(0)
    valid &= (projected[0] >= -radius - 1) & (projected[0] <= Ww + radius) & (projected[1] >= -radius - 1) & (projected[1] <= Hh + radius)
    source = np.flatnonzero(valid); u, v = np.rint(projected[:, source]).astype(np.int64); size = Hh * Ww; zbuf = np.full(size, np.inf); cands = []
    for du in range(-radius, radius + 1):
        for dv in range(-radius, radius + 1):
            uu, vv = u + du, v + dv; ok = (uu >= 0) & (uu < Ww) & (vv >= 0) & (vv < Hh); src = source[ok]; dst = vv[ok] * Ww + uu[ok]
            cands.append((src, dst, (uu[ok] - projected[0, src]) ** 2 + (vv[ok] - projected[1, src]) ** 2)); np.minimum.at(zbuf, dst, z[src])
    dbuf = np.full(size, np.inf)
    for src, dst, d in cands:
        near = z[src] == zbuf[dst]; np.minimum.at(dbuf, dst[near], d[near])
    ibuf = np.full(size, size, dtype=np.int64)
    for src, dst, d in cands:
        ch = (z[src] == zbuf[dst]) & (d == dbuf[dst]); np.minimum.at(ibuf, dst[ch], src[ch])
    cov = ibuf < size; img = np.zeros((size, 3), np.uint8); img[cov] = rgb.reshape(-1, 3)[ibuf[cov]]
    return img.reshape(Hh, Ww, 3), cov.reshape(Hh, Ww)


@torch.no_grad()
def splat_depths(depths, K, c2w_src, target_c2w, size=(120, 216)):
    """Per-source z-buffer depth of every source view in every target view: (n_src, n_tgt, h, w), 0 = not covered.
    depths: list of (Hs,Ws) gauge-unit tensors (any resolution, the K given is for that resolution); torch, any device."""
    h, w = size; n_t = len(target_c2w); dev = depths[0].device; K = torch.as_tensor(np.asarray(K, np.float64), dtype=torch.float64, device=dev)
    Hs, Ws = depths[0].shape; Kh = K.clone(); Kh[0] *= w / Ws; Kh[1] *= h / Hs                   # intrinsics at the feature resolution
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float64, device=dev), torch.arange(w, dtype=torch.float64, device=dev), indexing='ij')
    pix = torch.stack([xs, ys, torch.ones_like(xs)]).reshape(3, -1); rays = torch.linalg.inv(Kh) @ pix
    T = torch.as_tensor(np.asarray(target_c2w, np.float64), device=dev); Tinv = torch.linalg.inv(T)                  # (n_t,4,4)
    out = torch.zeros((len(depths), n_t, h * w + 1), dtype=torch.float64, device=dev)
    for i, d in enumerate(depths):
        d = F.interpolate(d[None, None].double(), (h, w), mode='area')[0, 0].reshape(-1); ok = torch.isfinite(d) & (d > 0)
        pts = rays * d[None]; S = torch.as_tensor(np.asarray(c2w_src[i], np.float64), device=dev)
        world = S[:3, :3] @ pts + S[:3, 3:4]
        cam = Tinv[:, :3, :3] @ world[None] + Tinv[:, :3, 3:4]                                                   # (n_t,3,N)
        z = cam[:, 2]; uv = Kh @ cam; u = (uv[:, 0] / uv[:, 2].clamp_min(1e-9)).round(); v = (uv[:, 1] / uv[:, 2].clamp_min(1e-9)).round()
        valid = ok[None] & (z > 1e-3) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        idx = torch.where(valid, (v * w + u).long(), torch.full_like(u, h * w, dtype=torch.long))
        buf = torch.full((n_t, h * w + 1), float('inf'), dtype=torch.float64, device=dev).scatter_reduce_(1, idx, torch.where(valid, z, torch.full_like(z, float('inf'))), 'amin')
        out[i] = torch.where(torch.isfinite(buf), buf, torch.zeros_like(buf))
    return out[:, :, :h * w].reshape(len(depths), n_t, h, w).float()


@torch.no_grad()
def render_features(depths, K, c2w_src, target_c2w, size=(120, 216)):
    """Gaussian-feature pilot: (n_tgt, 3, h, w) per target view from the source depths placed at their stored cameras:
    depth    = log1p of the fused (nearest-source) z-buffer depth, 0 where no source covers the pixel;
    conflict = relative SPREAD of the per-source z-buffer depths where >= 2 sources cover (std / mean, clipped at 1). Not an
               occlusion-aware contradiction: a consistent scene gives a nonzero spread wherever a source sees the background
               behind a foreground the target sees (review 10-26, D1); read it as 'depth spread'.
    support  = fraction of the sources whose points land on the pixel ('coverage count'), not verified matches."""
    D = splat_depths(depths, K, c2w_src, target_c2w, size); cov = D > 0; n = cov.sum(0).float()
    fused = torch.where(cov, D, torch.full_like(D, float('inf'))).amin(0); depth = torch.where(n > 0, torch.log1p(fused), torch.zeros_like(fused))
    mean = torch.where(n > 0, D.sum(0) / n.clamp_min(1), torch.zeros_like(n)); var = torch.where(cov, (D - mean[None]) ** 2, torch.zeros_like(D)).sum(0) / n.clamp_min(1)
    conflict = torch.where(n >= 2, (var.sqrt() / mean.clamp_min(1e-6)).clamp(0, 1), torch.zeros_like(n))
    return torch.stack([depth, conflict, n / len(depths)], 1)


def render_chunk(renderer, da3, frames_864, c2w_src, K864, mpu, target_c2w, gauge_id, boundary, size=(540, 960), first_mode='dup2', features=False):
    """frames_864: source frames uint8 (n,480,864,3) with cameras c2w_src (n,4,4) gauge units; target_c2w (16,4,4).
    Returns rgb (16,3,480,864) in [-1,1], valid/conf (16,480,864) [+ feat (16,3,120,216) with features=True].
    first_mode (single source only, i.e. the first step; idea "fix the first-step memory", render_single_test.py): 'dup2' = current
    (FoundationRenderer duplicates the one view), 'dup4' = the view given four times, 'points' = no ReSplat, z-buffered depth splat."""
    H, W = size; src = upsize(frames_864, size); K = np.asarray(K864, np.float64).copy(); K[0] *= W / 864; K[1] *= H / 480
    depths, masks = da3_gauge_depths(da3, src, K, c2w_src, mpu)
    feat = render_features([d.cuda() for d in depths], K, c2w_src, target_c2w) if features else None   # (16,3,120,216), appended to the return
    sources = [dict(frame_id=i, rgb=torch.from_numpy(src[i].copy()).permute(2, 0, 1).float() / 127.5 - 1, depth=depths[i], confidence=masks[i], c2w=torch.as_tensor(c2w_src[i]).float(), gauge_id=gauge_id) for i in range(len(src))]
    if len(sources) == 1 and first_mode == 'dup4':
        sources = sources * 4
    if len(sources) == 1 and first_mode == 'points':
        pts = [warp_points(src[0], depths[0].numpy(), K, np.asarray(c2w_src[0], np.float64), np.asarray(t, np.float64)) for t in target_c2w]
        rgb = torch.from_numpy(np.stack([p[0] for p in pts])).permute(0, 3, 1, 2).float() / 127.5 - 1; val = torch.from_numpy(np.stack([p[1] for p in pts])).float()[:, None]
        rgb = F.interpolate(rgb, (480, 864), mode='bilinear', align_corners=False); val = F.interpolate(val, (480, 864), mode='nearest')
        return (rgb, val[:, 0], val[:, 0].clone()) + ((feat,) if features else ())
    out = renderer.render(sources, torch.as_tensor(target_c2w).float(), torch.as_tensor(K).float(), (H, W), seed=0, camera_units='anchor_median_depth', gauge_id=gauge_id, history_boundary=boundary)
    rgb = F.interpolate(out['rgb'], (480, 864), mode='bilinear', align_corners=False)
    valid = F.interpolate(out['valid'], (480, 864), mode='nearest'); conf = F.interpolate(out['conf'], (480, 864), mode='bilinear', align_corners=False)
    return (rgb, valid[:, 0], conf[:, 0]) + ((feat,) if features else ())
