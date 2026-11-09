"""Assemble the canonical target-space conditioning buffers for the adapter.

Given the keyframe cache and a target camera, retrieve the spatially relevant
keyframes, splat them into the target view, and package the result as the buffers
the memory adapter consumes:

    rgb      (1,3,H,W)  warped RGB from established geometry, holes = -1
    depth    (1,1,H,W)  warped z-depth, holes = 0
    coords   (1,3,H,W)  world xyz normalized to ~[-1,1] by scene extent, holes = 0
    valid    (1,1,H,W)  1 where geometry was rendered
    hole     (1,1,H,W)  1 where the frontier is uncovered (= 1 - valid); the
                        region diffusion is responsible for inventing
    conf     (1,1,H,W)  reliability of the winning source (0 in holes)
    count    (1,1,H,W)  #sources hitting each pixel (verification redundancy)
    rgb_var  (1,1,H,W)  cross-view RGB disagreement, 0 where count <= 1
    depth_var(1,1,H,W)  cross-view depth disagreement (scale-free), 0 where count <= 1
    angle    (1,1,H,W)  best source/target view-angle quality, cos in [0,1]

`coords` gives the DiT a view-independent address for each surface: within one
denoising call the context and target sides share a normalization, so the same world
point receives the same coordinate on both sides and attention's coordinate-to-
coordinate term can match them.

It is NOT persistent across steps, and the earlier claim here that it was is wrong.
`scene_extent` derives (center, radius) from whichever camera centers are in the cache,
and the cache is rebuilt from the newest frames at every rollout step, so the address of
a FIXED world point moves as the window slides. Measured across the 60 pinned schedules
at 20 rollout steps each: the radius varies by a median factor of 1.84 within a single
rollout (max 348), and a fixed point's address shifts by a median 1.34 normalized units
between consecutive steps -- against a window that spans about 0.33 -- with a worst case
of 291. The radius is clamped at 1e-3, so a stationary context collapses onto the clamp
and the addresses blow up.

Pass `norm=` to impose a persistent frame instead (for instance the scene-level scale).
Doing so changes every coordinate the model has ever been trained on, so it is a
retrain-level change rather than an evaluation switch. `architecture.md` states the
consequence plainly: this is target-aligned evidence over a sliding window, not
long-term memory.
"""
import torch

from .geometry import render_sources_to_target
from .retrieval import greedy_cover


def scene_extent(cache):
    """Center and radius of the cached camera centers (world units)."""
    centers = torch.stack([f.c2w[:3, 3] for f in cache.frames])  # (N,3)
    center = centers.mean(0)
    radius = (centers - center).norm(dim=1).max().clamp(min=1e-3)
    # pad so nearby geometry (a few radii out) still lands in range
    return center, radius * 3.0 + 1e-3


def canonicalize_world(world: torch.Tensor, norm) -> torch.Tensor:
    """Map BCHW world coordinates through a legacy extent or explicit similarity.

    ``trajectory_v2`` freezes orientation, origin and scale.  Legacy checkpoints use
    only ``(center, radius)`` and keep world-axis orientation.  Supporting both here
    prevents an evaluator from approximating the V2 transform with the old two-value
    interface while preserving every historical call byte-for-byte.
    """
    if world.ndim != 4 or world.shape[1] != 3:
        raise ValueError("world coordinates must be Bx3xHxW")
    if hasattr(norm, "points") and hasattr(norm, "sha256"):
        points = world.permute(0, 2, 3, 1)
        return norm.points(points).permute(0, 3, 1, 2).to(world)
    if not isinstance(norm, (tuple, list)) or len(norm) != 2:
        raise TypeError("norm must be (center,radius) or a frozen similarity")
    center, radius = norm
    center = torch.as_tensor(center).view(1, 3, 1, 1).to(world)
    radius = torch.as_tensor(radius).to(world)
    if not bool(torch.isfinite(radius)) or float(radius) <= 0:
        raise ValueError("canonical radius must be finite and positive")
    return (world - center) / radius


def render_target_buffers(cache, tgt_c2w, tgt_K, k=6, retrieval=True, norm=None,
                          fusion="zbuffer", strict_validity=False):
    """Return the conditioning-buffer dict for one target camera.

    `norm` optionally overrides (center, radius) for coordinate normalization so a
    whole trajectory shares one canonical frame; otherwise it is derived from the
    cache each call.

    `fusion` is passed through to `render_sources_to_target` -- "zbuffer" (default,
    what every existing checkpoint trained on) or "soft".
    """
    if len(cache) == 0:
        raise ValueError("empty cache")

    if retrieval:
        idxs, coverage = greedy_cover(cache, tgt_c2w, tgt_K, k=k)
        if not idxs:
            idxs = list(range(min(k, len(cache))))
        sub = cache.subset(idxs)
    else:
        sub, idxs, coverage = cache, list(range(len(cache))), None

    rgbs, depths, c2ws, Ks, confs = sub.stack(device=tgt_c2w.device)
    out = render_sources_to_target(rgbs, depths, c2ws, Ks, tgt_c2w, tgt_K, src_conf=confs,
                                   fusion=fusion, strict_validity=strict_validity)

    norm_used = norm if norm is not None else scene_extent(cache)
    coords = canonicalize_world(out["world"], norm_used)
    coords = coords * out["mask"]  # zero in holes

    valid = out["mask"]
    return {
        "rgb": out["rgb"],
        "depth": out["depth"],
        "coords": coords,
        "valid": valid,
        "hole": 1.0 - valid,
        "conf": out["conf"],
        "count": out["count"],
        "rgb_var": out["rgb_var"],
        "depth_var": out["depth_var"],
        "angle": out["angle"],
        "selected": idxs,
        "coverage": coverage,
    }
