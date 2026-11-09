"""Multi-source point-splat renderer built on the verified warp primitives.

Convention (validated by a synthetic epipolar round-trip, ~1e-6 px error):
  * poses are stored as camera-to-world (c2w), 4x4.
  * `unproject_points` and `project_points` from memory.projection take **w2c**.
  * depth is z-depth (distance along the camera +Z axis), (N,1,H,W), >0 where valid.

`render_sources_to_target` mirrors GEN3C forward_warp's internals (unproject ->
project -> depth-weighted bilinear splat) but with the explicit w2c convention so
there is no ambiguity, and composites N sources into one target view by z-buffer
(nearest warped surface wins).
"""
from typing import Optional
import torch

from .projection import (
    unproject_points,
    project_points,
    bilinear_splat,
    coordinate_grid,
)


def invert_pose(T: torch.Tensor) -> torch.Tensor:
    """Invert a batch of 4x4 rigid transforms (c2w<->w2c).

    Inverted on CPU: these are 4x4 so the cost is negligible, and it avoids the GPU
    cuSOLVER path, whose handle creation fails intermittently
    (CUSOLVER_STATUS_INTERNAL_ERROR) when the NVIDIA driver has been updated without
    a reboot (kernel-module vs NVML-library version mismatch). Result is returned on
    the input device/dtype.
    """
    return torch.inverse(T.to("cpu", torch.float64)).to(T.device, T.dtype)


def make_intrinsics(fx, fy, cx, cy, device="cpu", dtype=torch.float32) -> torch.Tensor:
    K = torch.zeros(3, 3, device=device, dtype=dtype)
    K[0, 0], K[1, 1], K[0, 2], K[1, 2], K[2, 2] = fx, fy, cx, cy, 1.0
    return K


def _splat_one(rgb, depth, world, conf, tgt_w2c, tgt_K, *, source_valid=None):
    """Warp a single source (rgb, z-depth, world-xyz, conf) into the target view.

    Returns warped_rgb (b,3,H,W in [-1,1]), warped_depth (b,1,H,W), warped_world
    (b,3,H,W), warped_conf (b,1,H,W), and mask (b,1,H,W: 1 where a surface landed).

    `source_valid` (b,1,h,w) is which SOURCE pixels are allowed to contribute. Without
    it the only test is `trans_depth > 0`, which asks whether the warped point landed in
    front of the target camera and never whether the source pixel had geometry at all.
    `unproject_points` returns (0,0,0) for every pixel it masks out, so a source pixel
    with no depth is a point at the world origin -- and if the origin happens to sit in
    front of the target, it passes that test and becomes target geometry. Reproduced: a
    source with depth 0 and confidence 0 everywhere still yields valid target pixels
    carrying its RGB.
    """
    b, _, h, w = depth.shape
    # project cached world points into target camera
    proj, cam_pts = project_points(world, tgt_w2c, tgt_K, return_cam_points=True)  # (b,h,w,3,1),(b,h,w,3)
    trans_depth = cam_pts[..., 2].unsqueeze(1)  # (b,1,h,w) z in target cam
    coords = proj[..., :2, 0] / (proj[..., 2:3, 0] + 1e-7)  # (b,h,w,2)
    coords = coords.permute(0, 3, 1, 2)  # (b,2,h,w)
    grid = coordinate_grid(b, h, w, device=depth.device, dtype=depth.dtype)
    flow = coords - grid
    valid = (trans_depth > 0).to(depth.dtype)
    if source_valid is not None:
        valid = valid * source_valid.to(depth.dtype)

    warped_rgb, m = bilinear_splat(rgb, valid, trans_depth, flow, is_image=True)
    warped_depth = bilinear_splat(trans_depth, valid, trans_depth, flow)[0]
    warped_world = bilinear_splat(
        world.permute(0, 3, 1, 2), valid, trans_depth, flow
    )[0]
    warped_conf = bilinear_splat(conf, valid, trans_depth, flow)[0]
    return warped_rgb, warped_depth, warped_world, warped_conf, m


SOFT_Z_TAU = 0.05   # soft z-buffer falloff, as a FRACTION of the nearest depth


def render_sources_to_target(
    rgbs: torch.Tensor,      # (N,3,H,W) in [-1,1]
    depths: torch.Tensor,    # (N,1,H,W) z-depth, >0 valid
    src_c2ws: torch.Tensor,  # (N,4,4)
    src_Ks: torch.Tensor,    # (N,3,3)
    tgt_c2w: torch.Tensor,   # (4,4)
    tgt_K: torch.Tensor,     # (3,3)
    src_conf: Optional[torch.Tensor] = None,  # (N,1,H,W) per-source reliability weight in [0,1]
    fusion: str = "zbuffer",
    strict_validity: bool = False,
):
    """Composite N source frames into the target view.

    `strict_validity` decides which SOURCE pixels may contribute at all. With it off --
    the historical behaviour every existing checkpoint trained on, and therefore the
    default -- a pixel is admitted purely on `trans_depth > 0`, i.e. on having landed in
    front of the target camera. That test does not consult the source's own `depth > 0`
    or its confidence, and `unproject_points` gives every masked-out source pixel the
    world origin, so a depth-less source pixel becomes real target geometry whenever the
    origin is in front of the target. Measured on this code: a 4x4 source with depth 0
    and confidence 0 everywhere still produced valid target pixels carrying its RGB, and
    an unreliable near surface (conf 0) outranked a reliable far one (conf 1) under both
    fusion modes.

    With it on, a source pixel must additionally have `depth > 0` and non-zero
    confidence. Confidence is not decorative here: `KeyframeCache.add` defaults it to
    `reliable_depth_mask`, which is already zero at depth discontinuities and where depth
    is absent, so this is the mask the cache had computed and the splat was discarding.
    Pixels that no source covers reliably become holes, which the evidence layout already
    represents explicitly and neutral-fills.

    `fusion` selects how overlapping sources are combined:

      "zbuffer" (default)  nearest warped surface wins outright. Bit-identical to
                           the original behaviour -- every existing memory
                           checkpoint was trained on this.
      "soft"               confidence/visibility/angle/depth-weighted blend,
                           w_i = C_i * V_i * A_i * Z_i, with Z_i a soft z-buffer
                           exp(-(d_i - d_min) / (SOFT_Z_TAU * d_min)) so an
                           occluded surface is suppressed smoothly instead of
                           being discarded at a hard boundary.

    Returns a dict of target-space buffers:
      rgb    (1,3,H,W) in [-1,1], holes = -1
      depth  (1,1,H,W) z-depth, holes = 0
      world  (1,3,H,W) world xyz of the fused surface, holes = 0
      mask   (1,1,H,W) 1 where any source provided a surface
      conf   (1,1,H,W) reliability of the surface (0 in holes)
      count  (1,1,H,W) number of sources that hit each pixel (coverage redundancy)

    plus the DISAGREEMENT statistics, which exist because winner-take-all throws
    away the fact that the losers disagreed. `conf` alone cannot distinguish "one
    confident view" from "three views that contradict each other", and that is
    exactly where warped evidence should be overridden rather than trusted:
      rgb_var   (1,1,H,W) weighted variance of warped RGB across supporting views,
                          mean over channels; 0 where count <= 1
      depth_var (1,1,H,W) weighted variance of warped depth, normalised by the
                          fused depth so it is scale-free; 0 where count <= 1
      angle     (1,1,H,W) best view-angle quality among supporting views:
                          cos of the angle between the source and target viewing
                          rays at the surface, clamped to [0,1] (1 = co-located
                          viewpoint, 0 = 90 deg baseline)
    """
    assert fusion in ("zbuffer", "soft"), fusion
    N, _, H, Wd = rgbs.shape
    dev, dt = rgbs.device, rgbs.dtype
    tgt_w2c = invert_pose(tgt_c2w)[None]
    tgt_K_b = tgt_K[None]
    tgt_center = tgt_c2w[:3, 3].view(1, 3, 1, 1).float()

    # Pass 1: warp every source independently and keep the per-source buffers.
    # N is the retrieval budget (k=6), so this costs ~50 MB at 384x640 fp32 --
    # cheap enough to buy both fusion modes and the cross-view statistics.
    w_rgb, w_depth, w_world, w_conf, w_hit, w_angle = [], [], [], [], [], []
    for i in range(N):
        w2c_i = invert_pose(src_c2ws[i : i + 1])
        world_i = unproject_points(depths[i : i + 1], w2c_i, src_Ks[i : i + 1])  # (1,H,W,3)
        conf_i = src_conf[i : i + 1] if src_conf is not None else torch.ones_like(depths[i : i + 1])
        source_valid = None
        if strict_validity:
            source_valid = ((depths[i : i + 1] > 0) & (conf_i > 0)).to(depths.dtype)
        wr, wd, ww, wc, m = _splat_one(
            rgbs[i : i + 1], depths[i : i + 1], world_i, conf_i, tgt_w2c, tgt_K_b,
            source_valid=source_valid,
        )
        hit = (m > 0)
        # view-angle quality at the warped surface: how close the source's viewing
        # direction is to the target's. A grazing source is geometrically valid but
        # samples the surface poorly, so it should not outvote a head-on one.
        src_center = src_c2ws[i, :3, 3].view(1, 3, 1, 1).float()
        d_src = torch.nn.functional.normalize(ww.float() - src_center, dim=1, eps=1e-6)
        d_tgt = torch.nn.functional.normalize(ww.float() - tgt_center, dim=1, eps=1e-6)
        ang = (d_src * d_tgt).sum(dim=1, keepdim=True).clamp(0.0, 1.0) * hit.float()
        w_rgb.append(wr); w_depth.append(wd); w_world.append(ww)
        w_conf.append(wc); w_hit.append(hit); w_angle.append(ang)

    rgb_n = torch.cat(w_rgb, 0).float()                     # (N,3,H,W)
    depth_n = torch.cat(w_depth, 0).float()                 # (N,1,H,W)
    world_n = torch.cat(w_world, 0).float()                 # (N,3,H,W)
    conf_n = torch.cat(w_conf, 0).float()                   # (N,1,H,W)
    hit_n = torch.cat(w_hit, 0).float()                     # (N,1,H,W)
    angle_n = torch.cat(w_angle, 0).float()                 # (N,1,H,W)

    count = hit_n.sum(0, keepdim=True)                      # (1,1,H,W)
    any_hit = count > 0
    # nearest depth per pixel among the sources that actually hit it
    d_eff = torch.where(hit_n > 0, depth_n, torch.full_like(depth_n, float("inf")))
    d_min = d_eff.min(dim=0, keepdim=True).values                       # (1,1,H,W)
    d_min_safe = torch.where(any_hit, d_min, torch.ones_like(d_min))

    # soft z-buffer weight: 1 at the nearest surface, decaying for anything behind it
    z_w = torch.exp(-(depth_n - d_min_safe).clamp_min(0.0)
                    / (SOFT_Z_TAU * d_min_safe.abs().clamp_min(1e-6)))
    weights = conf_n.clamp_min(0.0) * hit_n * angle_n * z_w             # (N,1,H,W)
    w_sum = weights.sum(0, keepdim=True)
    # a pixel can be covered yet score ~0 (grazing + low conf); fall back to the
    # plain z-buffer weight there so coverage is never silently turned into a hole
    degenerate = any_hit & (w_sum < 1e-6)
    weights = torch.where(degenerate.expand_as(weights), hit_n * z_w, weights)
    w_sum = weights.sum(0, keepdim=True)
    w_norm = weights / w_sum.clamp_min(1e-6)

    if fusion == "soft":
        out_rgb = (w_norm * rgb_n).sum(0, keepdim=True)
        out_depth = (w_norm * depth_n).sum(0, keepdim=True)
        out_world = (w_norm * world_n).sum(0, keepdim=True)
        out_conf = (w_norm * conf_n).sum(0, keepdim=True)
        out_rgb = torch.where(any_hit.expand_as(out_rgb), out_rgb, torch.full_like(out_rgb, -1.0))
        out_depth = torch.where(any_hit, out_depth, torch.zeros_like(out_depth))
        out_world = torch.where(any_hit.expand_as(out_world), out_world, torch.zeros_like(out_world))
        out_conf = torch.where(any_hit, out_conf, torch.zeros_like(out_conf))
    else:
        # winner-take-all: index the nearest hitting source (ties -> lowest index,
        # matching the strict `<` of the original sequential z-buffer)
        win = d_eff.argmin(dim=0, keepdim=True)                          # (1,1,H,W)
        out_depth = torch.gather(depth_n, 0, win)
        out_conf = torch.gather(conf_n, 0, win)
        out_rgb = torch.gather(rgb_n, 0, win.expand(-1, 3, -1, -1))
        out_world = torch.gather(world_n, 0, win.expand(-1, 3, -1, -1))
        out_rgb = torch.where(any_hit.expand_as(out_rgb), out_rgb, torch.full_like(out_rgb, -1.0))
        out_depth = torch.where(any_hit, out_depth, torch.zeros_like(out_depth))
        out_world = torch.where(any_hit.expand_as(out_world), out_world, torch.zeros_like(out_world))
        out_conf = torch.where(any_hit, out_conf, torch.zeros_like(out_conf))

    out_mask = any_hit.to(dt)

    # Cross-view disagreement, weighted by the same w_i so an occluded surface
    # behind the visible one does not register as a contradiction.
    multi = (count > 1).float()
    rgb_mean = (w_norm * rgb_n).sum(0, keepdim=True)
    rgb_var = (w_norm * (rgb_n - rgb_mean).square()).sum(0, keepdim=True).mean(1, keepdim=True)
    depth_mean = (w_norm * depth_n).sum(0, keepdim=True)
    depth_var = (w_norm * (depth_n - depth_mean).square()).sum(0, keepdim=True)
    depth_var = depth_var / depth_mean.abs().clamp_min(1e-6).square()   # scale-free
    angle_best = (angle_n * hit_n).max(dim=0, keepdim=True).values

    return {
        "rgb": out_rgb.to(dt), "depth": out_depth.to(dt), "world": out_world.to(dt),
        "mask": out_mask, "conf": out_conf.to(dt), "count": count.to(dt),
        "rgb_var": (rgb_var * multi).to(dt),
        "depth_var": (depth_var * multi).to(dt),
        "angle": (angle_best * any_hit.float()).to(dt),
    }
