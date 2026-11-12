"""Sub-latent camera conditioning: four commanded poses per latent frame.

With native video latents one latent frame carries four RGB frames, but
`plucker_latent_rays` emits one ray plane per *pose*.  Conditioning a chunked model
on one pose per latent would drop the camera rate to a quarter of the frame rate,
which a keyboard interface cannot afford: a command must steer every frame it
generates, not every fourth.

This module keeps rays at the RGB frame rate and folds each group of four along the
channel axis, so a latent frame receives all four of its poses at full sub-pixel
resolution.  Nothing is averaged — pooling rays below the pixel level was the FC-1 v1
weak-camera failure and the same objection applies to pooling them in time.

Layout for one latent frame is sub-frame-major:

    [pose_0 (1536) | pose_1 (1536) | pose_2 (1536) | pose_3 (1536)]  = 6144 channels

The stream's first latent frame carries a single RGB frame (see `models.vae_stream`),
so its remaining three slots are filled with that pose repeated and flagged through
`sub_frame_validity`.  Repetition rather than zeros is deliberate: an all-zero ray
plane is not a camera and is out of distribution for a conv stem trained on unit
direction vectors — the same reason the binding battery uses graded real-camera lies
instead of zeroed rays.
"""
from __future__ import annotations

import torch
from torch import Tensor

from gaussverse.camera.rays import RAY_CHANNELS, VAE_STRIDE, plucker_latent_rays

FRAMES_PER_LATENT = 4
PACKED_RAY_CHANNELS = RAY_CHANNELS * FRAMES_PER_LATENT  # 6144


def latent_frame_count(rgb_frames: int, *, first_chunk: bool) -> int:
    """Latent frames produced by ``rgb_frames`` RGB frames, matching the VAE."""
    if rgb_frames < 1:
        raise ValueError("a chunk needs at least one RGB frame")
    if first_chunk:
        if (rgb_frames - 1) % FRAMES_PER_LATENT:
            raise ValueError(
                f"a first chunk needs 1 + 4k frames, got {rgb_frames}"
            )
        return 1 + (rgb_frames - 1) // FRAMES_PER_LATENT
    if rgb_frames % FRAMES_PER_LATENT:
        raise ValueError(f"a resumed chunk needs 4k frames, got {rgb_frames}")
    return rgb_frames // FRAMES_PER_LATENT


def group_poses(poses: Tensor, *, first_chunk: bool) -> Tensor:
    """``(T,4,4)`` per-frame poses -> ``(L,4,4,4)`` grouped by latent frame.

    The first latent of a stream owns one real pose; it is repeated to fill the group.
    """
    if poses.ndim != 3 or poses.shape[-2:] != (4, 4):
        raise ValueError(f"poses must be (T,4,4), got {tuple(poses.shape)}")
    frames = poses.shape[0]
    latents = latent_frame_count(frames, first_chunk=first_chunk)
    if not first_chunk:
        return poses.reshape(latents, FRAMES_PER_LATENT, 4, 4)
    head = poses[:1].expand(FRAMES_PER_LATENT, 4, 4)[None]
    if latents == 1:
        return head
    tail = poses[1:].reshape(latents - 1, FRAMES_PER_LATENT, 4, 4)
    return torch.cat([head, tail], dim=0)


def sub_frame_validity(latents: int, *, first_chunk: bool,
                       device=None, dtype=torch.float32) -> Tensor:
    """``(L, 4)`` mask: 1 where a slot holds a distinct commanded pose.

    Only the stream's very first latent has padded slots, so this is all ones except
    for that frame's trailing three entries.
    """
    if latents < 1:
        raise ValueError("a chunk needs at least one latent frame")
    validity = torch.ones(latents, FRAMES_PER_LATENT, device=device, dtype=dtype)
    if first_chunk:
        validity[0, 1:] = 0.0
    return validity


def packed_plucker_rays(poses: Tensor, K: Tensor, height: int, width: int,
                        *, first_chunk: bool, stride: int = VAE_STRIDE) -> Tensor:
    """Per-frame poses -> ``(L, 6144, height/stride, width/stride)``.

    ``poses`` are ``(T,4,4)`` camera-to-world in the target-relative gauge, one per RGB
    frame.  ``K`` is ``(3,3)`` or ``(T,3,3)`` at full resolution.
    """
    grouped = group_poses(poses, first_chunk=first_chunk)
    latents = grouped.shape[0]
    flat = grouped.reshape(latents * FRAMES_PER_LATENT, 4, 4)
    if K.dim() == 2:
        flat_K = K
    else:
        if K.shape[0] != poses.shape[0]:
            raise ValueError("per-frame intrinsics must match the pose count")
        grouped_K = group_poses(
            torch.nn.functional.pad(K, (0, 1, 0, 1)), first_chunk=first_chunk
        )[..., :3, :3]
        flat_K = grouped_K.reshape(latents * FRAMES_PER_LATENT, 3, 3)
    rays = plucker_latent_rays(flat, flat_K, height, width, stride)
    h_lat, w_lat = rays.shape[-2:]
    return rays.reshape(latents, FRAMES_PER_LATENT * RAY_CHANNELS, h_lat, w_lat)


def unpack_sub_frame(packed: Tensor, index: int) -> Tensor:
    """Recover one sub-frame's ``(L, 1536, h, w)`` ray plane from a packed tensor."""
    if not 0 <= index < FRAMES_PER_LATENT:
        raise ValueError(f"sub-frame index must lie in [0,{FRAMES_PER_LATENT - 1}]")
    if packed.shape[-3] != PACKED_RAY_CHANNELS:
        raise ValueError(
            f"packed rays must have {PACKED_RAY_CHANNELS} channels, "
            f"got {packed.shape[-3]}"
        )
    return packed[..., index * RAY_CHANNELS:(index + 1) * RAY_CHANNELS, :, :]
