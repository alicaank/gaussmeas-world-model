"""Warp CLEAN VAE latents into a target view, per source, without an RGB round trip.

The current memory branch renders warped RGB in pixel space and then re-encodes it with
the VAE. That path costs the evidence two lossy steps the model can never undo:

  1. splatting in pixel space, where holes are filled with a neutral value the encoder
     then reads as content;
  2. a VAE encode of an image that is not a natural image -- seams, neutral-filled holes
     and resampling artifacts are exactly the statistics the encoder was not trained on.

Warping the SOURCE's clean latent instead skips both. The source latent is encoded once
from a real photograph, and geometry only decides where each latent cell lands.

This is not free of assumptions, and the honest one to state is that the VAE is not
shift-equivariant: a latent cell is a 16x16 receptive field with context, not an
independent 16x16 tile, so moving cells around a grid is an approximation of the warp the
image would have undergone. The screen in `bench/phase4_screen.py` exists precisely to
measure whether that approximation beats the RGB round trip, rather than assuming it.

Sources are kept SEPARATE. Fusing before the model sees them is what the current branch
does, and it destroys the per-source structure that any source-selection mechanism would
need to act on.
"""
from __future__ import annotations

import torch

from gmlib.memory.geometry import invert_pose
from gmlib.memory.projection import (
    bilinear_splat, coordinate_grid, project_points, unproject_points)


def scale_K_to_latent(K: torch.Tensor, factor: int) -> torch.Tensor:
    """Intrinsics for a `factor`x-downsampled grid.

    The half-pixel terms matter: pixel centres of the coarse grid do not coincide with
    pixel centres of the fine grid, and dropping the +/-0.5 puts the principal point half
    a latent cell off -- a systematic 8-pixel warp bias at factor 16, which is most of a
    cell and would show up as a spurious loss for the clean path.
    """
    out = K.clone().float()
    out[0, 0] = K[0, 0] / factor
    out[1, 1] = K[1, 1] / factor
    out[0, 2] = (K[0, 2] + 0.5) / factor - 0.5
    out[1, 2] = (K[1, 2] + 0.5) / factor - 0.5
    return out


def depth_to_latent_grid(depth: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """(1,1,H,W) pixel depth -> (1,1,h,w) at latent resolution.

    Uses NEAREST sampling, not area averaging. Averaging depth across a 16x16 cell that
    straddles a depth discontinuity invents a surface at a distance where nothing exists,
    and that invented surface then splats to a wrong location. Nearest keeps every value a
    real measurement.
    """
    return torch.nn.functional.interpolate(depth.float(), size=(h, w), mode="nearest")


def splat_latents_to_target(
    latents: torch.Tensor,     # (N, C, h, w) clean per-source VAE latents
    depths: torch.Tensor,      # (N, 1, H, W) pixel-resolution z-depth, >0 valid
    src_c2ws: torch.Tensor,    # (N, 4, 4)
    src_K: torch.Tensor,       # (3, 3) pixel-resolution intrinsics
    tgt_c2w: torch.Tensor,     # (4, 4)
    factor: int = 16,
):
    """Warp each source latent into the target view independently.

    Returns (warped, valid, warped_depth):
      warped        (N, C, h, w) source latents resampled into the target grid, 0 in holes
      valid         (N, 1, h, w) 1 where that source landed a surface
      warped_depth  (N, 1, h, w) target-frame z of the landed surface, 0 in holes

    No fusion happens here. `bench/phase4_screen.py` and any downstream adapter decide how
    to combine (or not combine) the sources.
    """
    N, C, h, w = latents.shape
    dev = latents.device
    K_lat = scale_K_to_latent(src_K, factor).to(dev)
    tgt_w2c = invert_pose(tgt_c2w.float().to(dev))[None]

    out = torch.zeros_like(latents)
    valid = torch.zeros((N, 1, h, w), device=dev, dtype=latents.dtype)
    wdepth = torch.zeros((N, 1, h, w), device=dev, dtype=latents.dtype)

    for i in range(N):
        d = depth_to_latent_grid(depths[i: i + 1].to(dev), h, w)
        w2c_i = invert_pose(src_c2ws[i: i + 1].float().to(dev))
        world = unproject_points(d, w2c_i, K_lat[None])                    # (1,h,w,3)
        proj, cam = project_points(world, tgt_w2c, K_lat[None], return_cam_points=True)
        z = cam[..., 2].unsqueeze(1)                                       # (1,1,h,w)
        coords = (proj[..., :2, 0] / (proj[..., 2:3, 0] + 1e-7)).permute(0, 3, 1, 2)
        flow = coords - coordinate_grid(1, h, w, device=dev, dtype=d.dtype)
        ok = (z > 0).to(d.dtype)

        # is_image=False is REQUIRED, not a default. `is_image=True` clamps the result to
        # [-1,1] and fills holes with -1, which is right for RGB and destroys a VAE latent:
        # Wan2.2 latents range well outside [-1,1], so the clamp silently truncates most
        # of the signal. Caught by test_identity_pose_round_trips_the_latent, which failed
        # with max error 3.1 on a warp that is mathematically the identity.
        v, m = bilinear_splat(latents[i: i + 1].float(), ok, z, flow, is_image=False)
        out[i] = v[0].to(latents.dtype)
        valid[i] = (m > 0).to(latents.dtype)[0]
        wdepth[i] = bilinear_splat(z, ok, z, flow)[0][0]

    return out, valid, wdepth


def zbuffer_combine(warped: torch.Tensor, valid: torch.Tensor, wdepth: torch.Tensor):
    """Nearest-surface-wins fusion, matching the existing branch's default.

    Provided as the FIXED-combination baseline the screen compares against a per-position
    source oracle. Holes (no source landed) stay zero and are excluded by the returned
    mask rather than being filled -- a filled hole is content the model cannot tell from
    evidence.
    """
    big = torch.finfo(wdepth.dtype).max
    d = torch.where(valid > 0, wdepth, torch.full_like(wdepth, big))
    win = d.argmin(dim=0, keepdim=True)                                    # (1,1,h,w)
    fused = torch.gather(warped, 0, win.expand(1, warped.shape[1], -1, -1))
    any_valid = (valid.sum(dim=0, keepdim=True) > 0).to(warped.dtype)
    return fused * any_valid, any_valid
