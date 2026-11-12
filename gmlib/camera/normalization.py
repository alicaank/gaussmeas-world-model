"""Persistent-scene-scale pose normalization.

FrameCrafter's normalize_w2c_make_cam_last_origin rescales every sampled view
set so the mean camera-center distance to the target is 1 — the same world
point gets different coordinates depending on which views were sampled, which
breaks persistent memory. Here the rigid alignment (target camera -> identity)
is kept but the uniform scale is a per-scene constant computed once from ALL
cameras and reused for every sample, camera and (later) memory point.

Scale is per-scene, not metric: COLMAP scale still varies across scenes.
Cross-scene metric alignment (e.g. DA-V2 affine fit) is deferred past M1.
"""
from __future__ import annotations

import torch
from torch import Tensor


def compute_scene_scale(c2w_all: Tensor) -> float:
    """Fixed scene scale from ALL cameras: mean center distance to the centroid.

    Chosen (over e.g. median consecutive spacing) so that relative camera
    offsets within a scene land in O(1) units, matching the value range the
    per-set normalization produced — Plücker moments (o x d) stay bounded.
    """
    assert c2w_all.ndim == 3 and c2w_all.shape[-2:] == (4, 4)
    centers = c2w_all[:, :3, 3].double()
    s = (centers - centers.mean(dim=0)).norm(dim=-1).mean()
    return float(s.clamp_min(1e-6))


@torch.no_grad()
def relative_to_target(w2c: Tensor, s_scene: float, target_index: int = -1):
    """Rigid-align so camera `target_index` is at identity; scale by s_scene.

    Depends only on (frame, target) pairs — the same frame gets the same
    relative pose no matter which other views were sampled.

    Args:
        w2c: (N, 4, 4) world-to-camera, OpenCV convention.
    Returns:
        (w2c_rel, c2w_rel): both (N, 4, 4); c2w_rel[target_index] == I.
    """
    assert w2c.ndim == 3 and w2c.shape[-2:] == (4, 4)
    c2w = torch.linalg.inv(w2c.double())
    R = c2w[:, :3, :3]
    t = c2w[:, :3, 3]

    R0 = R[target_index]
    t0 = t[target_index]
    R_rel = R0.transpose(0, 1) @ R
    t_rel = (R0.transpose(0, 1) @ (t - t0).unsqueeze(-1)).squeeze(-1) / s_scene

    c2w_rel = torch.zeros_like(c2w)
    c2w_rel[:, :3, :3] = R_rel
    c2w_rel[:, :3, 3] = t_rel
    c2w_rel[:, 3, 3] = 1.0
    w2c_rel = torch.linalg.inv(c2w_rel)
    return w2c_rel.to(w2c.dtype), c2w_rel.to(w2c.dtype)
