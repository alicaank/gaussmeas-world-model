"""Projection-based keyframe scoring — Lyra 2.0's retrieval, at rollout scale.

`retrieval.score_keyframes` answers "what can this keyframe explain at the target pose?"
by SPLAT-RENDERING the keyframe, once per candidate per query. That is affordable for a
six-frame window and hopeless for a store that grows with the rollout: a hundred cached
entries means a hundred renders to choose six.

`Sparse3DCache` avoids it. It keeps each frame as a coarse point cloud -- depth
subsampled by `downsample`, intrinsics scaled to match -- and answers the same question
with one batched matmul over every candidate at once: project the points into the target
camera, count the ones that land in frame. No rendering, no per-candidate loop.

Two details from the reference that matter and are easy to leave out:

* **Occlusion filtering.** Lyra keeps a point only when its depth is within
  `depth_threshold` of the frontmost depth seen at that pixel across ALL candidates
  (`keep = z_vals <= min_d_for_pts + depth_threshold`). Without it a frame behind a wall
  scores as if it could see through the wall, and set-cover happily picks it.
* **Coverage is counted on the downsampled grid**, so the score is a fraction of coarse
  cells, not pixels. That is the quantity used for selection only; the chosen frames are
  still warped at full resolution afterwards.

Our `depth_threshold` is expressed as a FRACTION OF THE SCENE RADIUS rather than in world
units, because the store already carries a frozen canonical radius and an absolute
tolerance would mean different things on different scenes.
"""
from __future__ import annotations

import torch

from .projection import unproject_points


def downsampled_world_points(depth, c2w, K, *, downsample: int = 4) -> torch.Tensor:
    """``(1,H,W)`` z-depth -> ``(h,w,3)`` world points on a subsampled grid.

    Mirrors `Sparse3DCache.add`: stride the depth, scale the intrinsics by the same
    factor, unproject. Invalid depth (<= 0) unprojects to the origin and is carried as a
    validity mask by the caller rather than dropped, so the grid stays rectangular.
    """
    if depth.ndim != 3 or depth.shape[0] != 1:
        raise ValueError(f"depth must be (1,H,W), got {tuple(depth.shape)}")
    step = max(1, int(downsample))
    depth_ds = depth[:, ::step, ::step]
    scale = 1.0 / float(step)
    K_ds = K.clone()
    K_ds[0, 0] *= scale
    K_ds[1, 1] *= scale
    K_ds[0, 2] *= scale
    K_ds[1, 2] *= scale
    w2c = torch.linalg.inv(c2w.double().cpu()).to(depth.dtype).to(depth.device)
    world = unproject_points(depth_ds[None], w2c[None], K_ds[None])   # (1,h,w,3)
    return world[0]


def projected_coverage(points, valid, tgt_c2w, tgt_K, target_hw, *, downsample: int = 4,
                       depth_threshold: float = 0.0):
    """Boolean coverage of the target's coarse grid, for every candidate at once.

    Args:
        points: list of ``(h,w,3)`` world-point grids, one per candidate.
        valid:  list of ``(h,w)`` bool grids marking which of those points are real.
        target_hw: ``(H, W)`` of the target image in FULL-resolution pixels.
        depth_threshold: occlusion tolerance in world units; 0 disables the test.

    Returns ``(masks, cells)`` with ``masks`` a ``(C, cells)`` bool tensor.
    """
    if len(points) == 0:
        return torch.zeros((0, 0), dtype=torch.bool), 0
    device = tgt_c2w.device
    step = max(1, int(downsample))
    height, width = int(target_hw[0]), int(target_hw[1])
    grid_h, grid_w = height // step, width // step
    cells = grid_h * grid_w

    pts = torch.stack([p.to(device=device, dtype=torch.float32) for p in points])  # (C,h,w,3)
    ok = torch.stack([v.to(device=device) for v in valid]).reshape(len(points), -1)  # (C,hw)
    count, _, _, _ = pts.shape
    flat = pts.reshape(count, -1, 3)

    w2c = torch.linalg.inv(tgt_c2w.double().cpu()).to(torch.float32).to(device)
    cam = flat @ w2c[:3, :3].T + w2c[:3, 3]                       # (C,hw,3)
    z = cam[..., 2]

    K = tgt_K.to(device=device, dtype=torch.float32).clone()
    scale = 1.0 / float(step)
    K[0, 0] *= scale; K[1, 1] *= scale; K[0, 2] *= scale; K[1, 2] *= scale
    x = (cam[..., 0] * K[0, 0] / z.clamp(min=1e-6)) + K[0, 2]
    y = (cam[..., 1] * K[1, 1] / z.clamp(min=1e-6)) + K[1, 2]
    col, row = x.round().long(), y.round().long()

    inside = ok & (z > 0) & (col >= 0) & (col < grid_w) & (row >= 0) & (row < grid_h)
    if not bool(inside.any()):
        return torch.zeros((count, cells), dtype=torch.bool, device=device), cells

    key = (row.clamp(0, grid_h - 1) * grid_w + col.clamp(0, grid_w - 1))

    if depth_threshold and depth_threshold > 0:
        # shared z-buffer across candidates: a point survives only if it is within
        # `depth_threshold` of the frontmost surface anyone reports at that cell
        front = torch.full((cells,), float("inf"), device=device)
        front.scatter_reduce_(0, key[inside], z[inside], reduce="amin", include_self=True)
        inside = inside & (z <= front[key] + float(depth_threshold))

    masks = torch.zeros((count, cells), dtype=torch.bool, device=device)
    rows = torch.arange(count, device=device).unsqueeze(1).expand_as(key)
    masks[rows[inside], key[inside]] = True
    return masks, cells


def downsample_region(region, target_hw, *, downsample: int = 4) -> torch.Tensor:
    """Reduce a full-resolution mask to the coarse grid used for scoring.

    A coarse cell counts as part of the region when ANY of its pixels do: for the
    hole-targeting case that is the conservative direction, since a cell holding even a
    little disocclusion is worth covering.
    """
    step = max(1, int(downsample))
    height, width = int(target_hw[0]), int(target_hw[1])
    grid_h, grid_w = height // step, width // step
    flat = region.reshape(1, 1, height, width).float()
    pooled = torch.nn.functional.max_pool2d(flat[:, :, :grid_h * step, :grid_w * step],
                                            kernel_size=step, stride=step)
    return (pooled.reshape(-1) > 0)
