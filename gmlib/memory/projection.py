"""Projection and bilinear-splat primitives used by spatial memory.

The math is vendored from GEN3C's
``cosmos_predict1/diffusion/inference/forward_warp_utils_pytorch.py``. The
original module also contained unused high-level forward-warp, mesh, and
ray-tracing paths. Keeping only the primitives used by GaussMeas makes the
runtime dependency surface explicit.
"""

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional, Tuple

import torch


def _inverse_small_matrices(matrix: torch.Tensor) -> torch.Tensor:
    """Invert tiny camera matrices on CPU and return them to the input device.

    This avoids intermittent GPU cuSOLVER handle failures after a driver update;
    the matrices are only 3x3 or 4x4, so the transfer cost is negligible.
    """
    return torch.linalg.inv(matrix.to("cpu", torch.float32)).to(
        matrix.device, matrix.dtype
    )


def _max_exponent(dtype: torch.dtype) -> float:
    return {
        torch.bfloat16: 80.0,
        torch.float16: 10.0,
        torch.float32: 80.0,
        torch.float64: 700.0,
    }.get(dtype, 80.0)


def reliable_depth_mask(
    depth: torch.Tensor,
    window_size: int = 5,
    ratio_threshold: float = 0.05,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return locally smooth, positive depth pixels as a boolean mask.

    ``depth`` accepts ``(B,H,W)`` or ``(B,1,H,W)`` and the result always has a
    singleton channel dimension.
    """
    if window_size % 2 != 1:
        raise ValueError("window_size must be odd")
    if depth.ndim == 3:
        depth = depth.unsqueeze(1)
    elif depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError("depth must have shape (B,H,W) or (B,1,H,W)")

    pool = torch.nn.functional
    padding = window_size // 2
    local_max = pool.max_pool2d(depth, window_size, stride=1, padding=padding)
    local_min = -pool.max_pool2d(-depth, window_size, stride=1, padding=padding)
    local_mean = pool.avg_pool2d(depth, window_size, stride=1, padding=padding)
    ratio = (local_max - local_min) / (local_mean + eps)
    return (ratio < ratio_threshold) & (depth > 0)


def unproject_points(
    depth: torch.Tensor,
    w2c: torch.Tensor,
    intrinsic: torch.Tensor,
    is_depth: bool = True,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Unproject a z-depth image into a dense world-coordinate grid."""
    batch, _, height, width = depth.shape
    if mask is None:
        mask = depth > 0
    if mask.ndim == depth.ndim and mask.shape[1] == 1:
        mask = mask[:, 0]

    indices = torch.nonzero(mask)
    if indices.numel() == 0:
        return torch.zeros(
            (batch, height, width, 3), device=depth.device, dtype=depth.dtype
        )

    batch_idx, y_idx, x_idx = indices[:, 0], indices[:, 1], indices[:, 2]
    pixels = torch.stack(
        [
            x_idx.to(depth.dtype),
            y_idx.to(depth.dtype),
            torch.ones_like(x_idx, dtype=depth.dtype),
        ],
        dim=1,
    ).unsqueeze(-1)
    camera_rays = _inverse_small_matrices(intrinsic)[batch_idx] @ pixels

    valid_depth = depth[batch_idx, 0, y_idx, x_idx].view(-1, 1, 1)
    if is_depth:
        camera_points = valid_depth * camera_rays
    else:
        directions = camera_rays / camera_rays.norm(dim=1, keepdim=True).clamp_min(1e-8)
        camera_points = valid_depth * directions

    homogeneous = torch.cat(
        [
            camera_points,
            torch.ones(
                (camera_points.shape[0], 1, 1),
                device=depth.device,
                dtype=depth.dtype,
            ),
        ],
        dim=1,
    )
    world = _inverse_small_matrices(w2c)[batch_idx] @ homogeneous

    result = torch.zeros(
        (batch, height, width, 3), device=depth.device, dtype=depth.dtype
    )
    result[batch_idx, y_idx, x_idx] = world[:, :3, 0]
    return result


def project_points(
    world_points: torch.Tensor,
    w2c: torch.Tensor,
    intrinsic: torch.Tensor,
    return_cam_points: bool = False,
):
    """Project a dense world-coordinate grid into homogeneous image coordinates."""
    batch, height, width, _ = world_points.shape
    points = world_points.unsqueeze(-1)
    homogeneous = torch.cat(
        [
            points,
            torch.ones(
                (batch, height, width, 1, 1),
                device=points.device,
                dtype=points.dtype,
            ),
        ],
        dim=3,
    )
    camera = (w2c[:, None, None] @ homogeneous)[:, :, :, :3]
    projected = intrinsic[:, None, None] @ camera
    if return_cam_points:
        return projected, camera.squeeze(-1)
    return projected


def unproject_depth(
    depth: torch.Tensor,
    c2w: torch.Tensor,
    intrinsic: torch.Tensor,
) -> torch.Tensor:
    """Dense batched unprojection used by coverage diagnostics."""
    batch, channels, height, width = depth.shape
    if channels != 1:
        raise ValueError(f"expected one depth channel, got {depth.shape}")
    if c2w.shape != (batch, 4, 4):
        raise ValueError(f"invalid c2w shape: {c2w.shape}")
    if intrinsic.shape != (batch, 3, 3):
        raise ValueError(f"invalid intrinsic shape: {intrinsic.shape}")

    x = torch.arange(width, device=depth.device, dtype=depth.dtype)
    y = torch.arange(height, device=depth.device, dtype=depth.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    pixels = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
    pixels = pixels[None, :, :, :, None]

    rays = _inverse_small_matrices(intrinsic)[:, None, None] @ pixels
    camera_points = depth[:, 0, :, :, None, None] * rays
    homogeneous = torch.cat(
        [
            camera_points,
            torch.ones(
                (batch, height, width, 1, 1),
                device=depth.device,
                dtype=depth.dtype,
            ),
        ],
        dim=3,
    )
    return (c2w[:, None, None] @ homogeneous)[:, :, :, :3, 0]


def bilinear_splat(
    values: torch.Tensor,
    mask: Optional[torch.Tensor],
    depth: torch.Tensor,
    flow: torch.Tensor,
    flow_mask: Optional[torch.Tensor] = None,
    *,
    is_image: bool = False,
    n_views: int = 1,
    depth_weight_scale: float = 50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward-splat channel-first values with bilinear and depth weights."""
    batch, channels, height, width = values.shape
    device, dtype = values.device, values.dtype
    if mask is None:
        mask = torch.ones((batch, 1, height, width), device=device, dtype=dtype)
    if flow_mask is None:
        flow_mask = torch.ones((batch, 1, height, width), device=device, dtype=dtype)

    transformed = flow + coordinate_grid(batch, height, width, device, dtype)
    offset = transformed + 1
    floor = torch.floor(offset).long()
    ceil = torch.ceil(offset).long()
    offset = torch.stack(
        [
            offset[:, 0].clamp(0, width + 1),
            offset[:, 1].clamp(0, height + 1),
        ],
        dim=1,
    )
    floor = torch.stack(
        [
            floor[:, 0].clamp(0, width + 1),
            floor[:, 1].clamp(0, height + 1),
        ],
        dim=1,
    )
    ceil = torch.stack(
        [
            ceil[:, 0].clamp(0, width + 1),
            ceil[:, 1].clamp(0, height + 1),
        ],
        dim=1,
    )

    west = 1 - (offset[:, 0:1] - floor[:, 0:1])
    east = 1 - (ceil[:, 0:1] - offset[:, 0:1])
    north = 1 - (offset[:, 1:2] - floor[:, 1:2])
    south = 1 - (ceil[:, 1:2] - offset[:, 1:2])

    log_depth = torch.log1p(depth.clamp_min(0))
    exponent = log_depth / (log_depth.max() + 1e-7) * depth_weight_scale
    depth_weights = torch.exp(exponent.clamp(max=_max_exponent(dtype))) + 1e-7

    weights = [
        torch.moveaxis(
            proximity * mask * flow_mask / depth_weights,
            [0, 1, 2, 3],
            [0, 3, 1, 2],
        )
        for proximity in (north * west, south * west, north * east, south * east)
    ]
    values_cl = torch.moveaxis(values, [0, 1, 2, 3], [0, 3, 1, 2])
    output = torch.zeros(
        (batch, height + 2, width + 2, channels), dtype=dtype, device=device
    )
    output_weights = torch.zeros(
        (batch, height + 2, width + 2, 1), dtype=dtype, device=device
    )
    batch_idx = torch.arange(batch, device=device)[:, None, None]
    destinations = [
        (floor[:, 1], floor[:, 0]),
        (ceil[:, 1], floor[:, 0]),
        (floor[:, 1], ceil[:, 0]),
        (ceil[:, 1], ceil[:, 0]),
    ]
    for weight, (dest_y, dest_x) in zip(weights, destinations):
        output.index_put_(
            (batch_idx, dest_y, dest_x), values_cl * weight, accumulate=True
        )
        output_weights.index_put_(
            (batch_idx, dest_y, dest_x), weight, accumulate=True
        )

    if n_views > 1:
        output = output.reshape(
            batch // n_views, n_views, height + 2, width + 2, channels
        ).sum(1)
        output_weights = output_weights.reshape(
            batch // n_views, n_views, height + 2, width + 2, 1
        ).sum(1)

    output = torch.moveaxis(output, [0, 1, 2, 3], [0, 2, 3, 1])
    output_weights = torch.moveaxis(
        output_weights, [0, 1, 2, 3], [0, 2, 3, 1]
    )
    output = output[:, :, 1:-1, 1:-1]
    output_weights = torch.nan_to_num(output_weights[:, :, 1:-1, 1:-1], nan=1000.0)

    valid = output_weights > 0
    fill = torch.tensor(-1 if is_image else 0, dtype=dtype, device=device)
    result = torch.where(valid, output / output_weights, fill)
    if is_image:
        result = result.clamp(-1, 1)
    return result, valid.to(values)


def coordinate_grid(
    batch: int,
    height: int,
    width: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    """Return a dense ``(x,y)`` grid with shape ``(B,2,H,W)``."""
    x = torch.arange(width, device=device, dtype=dtype).view(1, 1, 1, width)
    y = torch.arange(height, device=device, dtype=dtype).view(1, 1, height, 1)
    return torch.cat(
        [
            x.expand(batch, 1, height, width),
            y.expand(batch, 1, height, width),
        ],
        dim=1,
    )
