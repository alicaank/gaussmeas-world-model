"""Target-view point rendering for GATE-ΩR scaffold evidence."""
from __future__ import annotations

import torch

from gmlib.training.gate_omega import fuse_scaffold_sources


def _validate_camera(c2w: torch.Tensor, intrinsics: torch.Tensor) -> None:
    if c2w.shape != (4, 4) or intrinsics.shape != (3, 3):
        raise ValueError("scaffold cameras require one 4x4 c2w and 3x3 intrinsics")
    if not torch.isfinite(c2w).all() or not torch.isfinite(intrinsics).all():
        raise ValueError("scaffold camera contains non-finite values")
    if abs(float(torch.linalg.det(intrinsics))) < 1e-12:
        raise ValueError("scaffold intrinsics are singular")


def render_point_source(values: torch.Tensor, depth: torch.Tensor,
                        confidence: torch.Tensor, *, source_c2w: torch.Tensor,
                        target_c2w: torch.Tensor, source_intrinsics: torch.Tensor,
                        target_intrinsics: torch.Tensor,
                        output_size: tuple[int, int] | None = None,
                        depth_tie_relative: float = 1e-4,
                        depth_tie_absolute: float = 1e-6) -> dict[str, torch.Tensor]:
    """Hard-render one RGB-D/feature source into one target camera.

    Points use pixel centres. Nearest positive target depth wins; confidence is
    consulted only for samples inside the explicit near-depth tie tolerance.
    """
    if values.ndim != 3 or depth.ndim != 3 or confidence.ndim != 3:
        raise ValueError("source values/depth/confidence must be CHW tensors")
    channels, source_h, source_w = values.shape
    if depth.shape != (1, source_h, source_w) \
            or confidence.shape != (1, source_h, source_w):
        raise ValueError("source depth/confidence grid does not match values")
    _validate_camera(source_c2w, source_intrinsics)
    _validate_camera(target_c2w, target_intrinsics)
    target_h, target_w = output_size or (source_h, source_w)
    if target_h < 1 or target_w < 1:
        raise ValueError("scaffold render size must be positive")
    device = values.device
    work_dtype = torch.float64
    yy, xx = torch.meshgrid(
        torch.arange(source_h, device=device, dtype=work_dtype) + 0.5,
        torch.arange(source_w, device=device, dtype=work_dtype) + 0.5,
        indexing="ij")
    pixels = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1).reshape(-1, 3)
    z_source = depth.reshape(-1).to(work_dtype)
    confidence_flat = confidence.reshape(-1).to(work_dtype)
    finite_value = torch.isfinite(values).all(dim=0).reshape(-1)
    legal = (torch.isfinite(z_source) & (z_source > 0)
             & torch.isfinite(confidence_flat)
             & (confidence_flat >= 0) & (confidence_flat <= 1)
             & finite_value)
    source_K_inv = torch.linalg.inv(source_intrinsics.to(device, work_dtype))
    camera_points = (pixels @ source_K_inv.T) * z_source[:, None]
    homogeneous = torch.cat([camera_points, torch.ones_like(z_source[:, None])], dim=1)
    world = homogeneous @ source_c2w.to(device, work_dtype).T
    target_camera = world @ torch.linalg.inv(
        target_c2w.to(device, work_dtype)).T
    target_z = target_camera[:, 2]
    projected = target_camera[:, :3] @ target_intrinsics.to(device, work_dtype).T
    projected_xy = projected[:, :2] / target_z[:, None].clamp_min(1e-12)
    target_x = torch.floor(projected_xy[:, 0]).long()
    target_y = torch.floor(projected_xy[:, 1]).long()
    legal &= (torch.isfinite(target_z) & (target_z > 0)
              & torch.isfinite(projected_xy).all(dim=1)
              & (target_x >= 0) & (target_x < target_w)
              & (target_y >= 0) & (target_y < target_h))

    output_values = values.new_zeros(channels, target_h, target_w)
    output_depth = depth.new_zeros(1, target_h, target_w)
    output_confidence = confidence.new_zeros(1, target_h, target_w)
    output_validity = depth.new_zeros(1, target_h, target_w)
    if not bool(legal.any()):
        return {"values": output_values, "depth": output_depth,
                "confidence": output_confidence, "validity": output_validity}

    original_index = torch.arange(source_h * source_w, device=device)[legal]
    pixel_index = (target_y[legal] * target_w + target_x[legal]).long()
    legal_z = target_z[legal]
    z_buffer = torch.full((target_h * target_w,), torch.inf,
                          device=device, dtype=work_dtype)
    z_buffer.scatter_reduce_(0, pixel_index, legal_z, reduce="amin", include_self=True)
    tolerance = depth_tie_absolute + depth_tie_relative * z_buffer[pixel_index].abs()
    tied = (legal_z - z_buffer[pixel_index]).abs() <= tolerance
    tied_conf = torch.where(tied, confidence_flat[legal],
                            torch.full_like(legal_z, -1))
    best_conf = torch.full((target_h * target_w,), -1.0,
                           device=device, dtype=work_dtype)
    best_conf.scatter_reduce_(0, pixel_index, tied_conf,
                              reduce="amax", include_self=True)
    best = tied & (tied_conf == best_conf[pixel_index])
    sentinel = source_h * source_w
    candidates = torch.where(best, original_index,
                             torch.full_like(original_index, sentinel))
    selected_index = torch.full((target_h * target_w,), sentinel,
                                device=device, dtype=torch.long)
    selected_index.scatter_reduce_(0, pixel_index, candidates,
                                   reduce="amin", include_self=True)
    valid_pixel = selected_index != sentinel
    selected = selected_index[valid_pixel]
    output_values.view(channels, -1)[:, valid_pixel] = values.reshape(channels, -1)[:, selected]
    output_depth.view(-1)[valid_pixel] = target_z[selected].to(output_depth.dtype)
    output_confidence.view(-1)[valid_pixel] = confidence_flat[selected].to(
        output_confidence.dtype)
    output_validity.view(-1)[valid_pixel] = 1
    return {"values": output_values, "depth": output_depth,
            "confidence": output_confidence, "validity": output_validity}


def render_and_fuse_sources(values: torch.Tensor, depth: torch.Tensor,
                            confidence: torch.Tensor, source_c2w: torch.Tensor, *,
                            target_c2w: torch.Tensor,
                            source_intrinsics: torch.Tensor,
                            target_intrinsics: torch.Tensor,
                            ages: torch.Tensor | None = None,
                            output_size: tuple[int, int] | None = None) -> dict:
    """Render one anchor and optionally one certified recent source, then fuse."""
    if values.ndim != 4 or values.shape[0] not in (1, 2):
        raise ValueError("scaffold accepts one anchor and at most one recent source")
    n = values.shape[0]
    if depth.shape[:2] != (n, 1) or confidence.shape[:2] != (n, 1) \
            or source_c2w.shape != (n, 4, 4):
        raise ValueError("scaffold source batch shapes disagree")
    if source_intrinsics.shape == (3, 3):
        source_intrinsics = source_intrinsics.expand(n, 3, 3)
    if source_intrinsics.shape != (n, 3, 3):
        raise ValueError("source intrinsics must be shared 3x3 or Nx3x3")
    rendered = [render_point_source(
        values[index], depth[index], confidence[index],
        source_c2w=source_c2w[index], target_c2w=target_c2w,
        source_intrinsics=source_intrinsics[index],
        target_intrinsics=target_intrinsics, output_size=output_size)
        for index in range(n)]
    return fuse_scaffold_sources(
        torch.stack([item["values"] for item in rendered]),
        torch.stack([item["depth"] for item in rendered]),
        torch.stack([item["validity"] for item in rendered]),
        torch.stack([item["confidence"] for item in rendered]),
        ages=ages)
