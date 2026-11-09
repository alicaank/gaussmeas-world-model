"""Conservative target-aligned rescue from persistent source memory.

The newest local view remains authoritative wherever it projects valid evidence.
Older certified views may only fill local holes.  This makes the memory path an
explicit correction channel instead of a second unconstrained image generator.
"""
from __future__ import annotations

import torch

from gaussverse.memory.latent_splat import zbuffer_combine


def rescue_local_with_archive(
    local_warped: torch.Tensor,
    local_valid: torch.Tensor,
    local_depth: torch.Tensor,
    archive_warped: torch.Tensor,
    archive_valid: torch.Tensor,
    archive_depth: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Fill only local holes with the nearest valid certified archive source.

    Inputs are source-major ``N,C,H,W`` tensors produced by
    :func:`splat_latents_to_target`.  Exactly one local source is required; the archive
    can contain one or more sources.  The returned ``rescue_mask`` is the region where
    archive content is causally capable of changing the condition.
    """
    if local_warped.ndim != 4 or local_warped.shape[0] != 1:
        raise ValueError("target rescue requires exactly one local source")
    if archive_warped.ndim != 4 or archive_warped.shape[0] < 1:
        raise ValueError("target rescue requires at least one archive source")
    for name, value, reference in (
        ("local_valid", local_valid, local_warped),
        ("local_depth", local_depth, local_warped),
        ("archive_valid", archive_valid, archive_warped),
        ("archive_depth", archive_depth, archive_warped),
    ):
        if (value.ndim != 4 or value.shape[0] != reference.shape[0]
                or value.shape[1] != 1 or value.shape[-2:] != reference.shape[-2:]):
            raise ValueError(f"{name} does not match its source tensor")
    if (local_warped.shape[1:] != archive_warped.shape[1:]
            or any(not bool(torch.isfinite(value).all()) for value in (
                local_warped, local_valid, local_depth,
                archive_warped, archive_valid, archive_depth))):
        raise ValueError("local and archive rescue tensors must be finite and aligned")

    archive_fused, archive_any = zbuffer_combine(
        archive_warped, archive_valid, archive_depth)
    local_mask = (local_valid[:1] > 0).to(local_warped.dtype)
    archive_mask = (archive_any > 0).to(local_warped.dtype)
    rescue_mask = (1.0 - local_mask) * archive_mask
    fused = local_warped[:1] * local_mask + archive_fused * rescue_mask
    return {
        "features": fused,
        "local_mask": local_mask,
        "rescue_mask": rescue_mask,
        "valid": (local_mask + rescue_mask).clamp_max(1),
        "archive_features": archive_fused,
    }
