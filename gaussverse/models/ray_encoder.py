"""Ray/context conditioning branch for the Wan2.2-5B frame renderer.

Adapted from FrameCrafter fb5b.ContextProjection with the GS-evidence path
removed. The doc's "compact ray encoder -> zero-init projection -> add to
stock patch tokens" — with the FC-1 v1 correction of keeping PixelUnshuffled
full-res rays (1536 ch) instead of a pooled 6-ch map.

Invariants carried over from fb5b (do not change casually):
- norm-free local stem: spatial norms leak information across the frame,
- zero-init patch projection => exact pretrained-backbone equivalence at init,
- gate initialized to 1.0: gate 0 on top of a zero projection deadlocks
  training (tanh(0) kills projection grads, zero projection kills gate grad).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from gaussverse.camera.rays import RAY_CHANNELS
from gaussverse.camera.packed_rays import FRAMES_PER_LATENT, PACKED_RAY_CHANNELS

LATENT_CHANNELS = 48
DIT_DIM = 3072
SPATIAL_PATCH = (2, 2)


class ResidualConvBlock(nn.Module):
    """Copied from FrameCrafter fb_adapter.ResidualConvBlock (norm-free)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(F.silu(x))
        x = self.conv2(F.silu(x))
        return x + residual


@dataclass(frozen=True)
class RayCondLayout:
    """Per-frame conditioning planes at latent resolution (NOT the DiT in_dim,
    which stays 48)."""

    context_latent: int = LATENT_CHANNELS  # clean VAE latent (zeros at target)
    condition_mask: int = 1                # 1 = context frame, 0 = target
    ray_features: int = RAY_CHANNELS       # 6·16² PixelUnshuffled Plücker rays
    view_role: int = 1                     # +1 context / -1 target

    @property
    def total(self) -> int:
        return self.context_latent + self.condition_mask + self.ray_features + self.view_role


LAYOUT = RayCondLayout()


class RayContextProjection(nn.Module):
    """Zero-initialized residual conditioning branch over ALL frames.

    Input:  (B, LAYOUT.total, F, H_lat, W_lat)
    Output: (B, F * H_lat/2 * W_lat/2, dim) in the DiT's frame-major order.
    """

    def __init__(self, channels: int = LAYOUT.total, hidden: int = 128, dim: int = DIT_DIM,
                 rayless: bool = False, init_scheme: str = "zero",
                 camera_span: bool = False):
        super().__init__()
        # `rayless` drops the 1536 Plücker planes from this branch's input, keeping
        # [48 context latents | 1 mask | 1 role] = 50 channels. Motivated by the camera-path
        # 2x2 (experiments.md 13.9): given CameraFiLM, this branch's ray planes contribute
        # +0.09 dB (n.s.) and nothing measurable to pose. The slice happens HERE so callers
        # keep passing the full 1586-ch cond and nothing else changes.
        self.rayless = rayless
        self.camera_span = bool(camera_span)
        if self.camera_span:
            # A resumed Wan latent represents four RGB frames.  The endpoint is
            # V3.1's historical ray plane; the three zero-initialized residual
            # gates let the preceding commanded poses enter without perturbing
            # an inherited checkpoint at step 0.  Gates are per ray feature, so
            # pixel-unshuffled spatial information is never pooled away.
            self.span_gates = nn.Parameter(torch.zeros(
                FRAMES_PER_LATENT - 1, RAY_CHANNELS
            ))
        if rayless:
            channels = LAYOUT.context_latent + LAYOUT.condition_mask + LAYOUT.view_role
        self.stem = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            ResidualConvBlock(hidden),
            ResidualConvBlock(hidden),
        )
        self.patch_projection = nn.Conv2d(hidden, dim, kernel_size=SPATIAL_PATCH, stride=SPATIAL_PATCH)
        # The output LayerNorm renormalises the residual to unit scale regardless of the
        # projection's magnitude, so the branch's strength is set entirely by
        # tanh(gate) ~ 0.76 and the initialisation std controls only its DIRECTION. With
        # zero-init that is harmless (zeros stay zeros) and the norm stabilises the branch
        # once it leaves zero. With scaled init it means the branch arrives at full
        # strength on step 0 rather than ramping -- the growworld finding, and the reason
        # Lyra has no output norm on the equivalent path. Under scaled init we therefore
        # drop it, so the learned projection scale governs magnitude the way it does in
        # Lyra's cam_encoder/buffer_encoder.
        self.output_norm = nn.LayerNorm(dim) if init_scheme == "zero" else nn.Identity()
        self.gate = nn.Parameter(torch.tensor(1.0))
        self.init_scheme = init_scheme
        if init_scheme == "zero":
            # Warm continuation: the branch is bitwise absent at step 0, so attaching it
            # to a converged checkpoint cannot disturb it.
            nn.init.zeros_(self.patch_projection.weight)
            nn.init.zeros_(self.patch_projection.bias)
        elif init_scheme == "scaled":
            # From-pretrained training: there is no converged camera behaviour to protect,
            # so zero-init only delays the branch becoming load-bearing while the backbone
            # is also moving -- the mechanism proposed for the lattice null in
            # experiments.md 29.10. Matches Lyra 2.0, which initialises its cam_encoder and
            # buffer_encoder with trunc_normal_(std=1/sqrt(dim)) rather than zeros
            # (`lyra_2/_src/networks/wan2pt1_lyra2.py`, WanAttentionBlock.init_weights).
            std = 1.0 / (dim ** 0.5)
            nn.init.trunc_normal_(self.patch_projection.weight, std=std, a=-3 * std, b=3 * std)
            nn.init.zeros_(self.patch_projection.bias)
        else:
            raise ValueError(f"init_scheme must be 'zero' or 'scaled', got {init_scheme!r}")

    def enable_camera_span(self) -> nn.Parameter:
        """Attach the V3.2 sub-latent camera branch to an old checkpoint."""
        if not self.camera_span:
            self.register_parameter(
                "span_gates",
                nn.Parameter(torch.zeros(
                    FRAMES_PER_LATENT - 1, RAY_CHANNELS,
                    device=self.gate.device, dtype=self.gate.dtype,
                )),
            )
            self.camera_span = True
        return self.span_gates

    def fuse_camera_span(self, rays: torch.Tensor) -> torch.Tensor:
        """Fold four exact RGB-frame ray planes into each latent ray plane.

        Ordinary V3.1 rays pass through unchanged.  Packed V3.2 rays are laid
        out sub-frame-major and reduce to the final-frame plane exactly while
        ``span_gates == 0``.
        """
        if rays.shape[1] == RAY_CHANNELS:
            return rays
        if rays.shape[1] != PACKED_RAY_CHANNELS:
            raise ValueError(
                f"camera rays need {RAY_CHANNELS} or {PACKED_RAY_CHANNELS} "
                f"channels, got {rays.shape[1]}"
            )
        if not self.camera_span:
            raise RuntimeError("packed camera rays require camera_span conditioning")
        b, _, f, h, w = rays.shape
        span = rays.reshape(b, FRAMES_PER_LATENT, RAY_CHANNELS, f, h, w)
        endpoint = span[:, -1]
        gates = torch.tanh(self.span_gates).to(
            device=rays.device, dtype=rays.dtype
        )[None, :, :, None, None, None]
        return endpoint + (gates * (span[:, :-1] - endpoint[:, None])).sum(dim=1)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        if self.rayless and cond.shape[1] == LAYOUT.total:
            # keep [ctx latents | mask] and the trailing role plane; drop the rays between
            keep = LAYOUT.context_latent + LAYOUT.condition_mask
            cond = torch.cat([cond[:, :keep], cond[:, -LAYOUT.view_role:]], dim=1)
        b, c, f, h, w = cond.shape
        x = cond.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
        x = self.stem(x)
        x = self.patch_projection(x)              # (B*F, dim, h/2, w/2)
        x = x.flatten(2).transpose(1, 2)          # (B*F, hw/4, dim)
        x = self.output_norm(x)
        x = x.reshape(b, -1, x.shape[-1])         # frame-major (f, h, w)
        return torch.tanh(self.gate) * x


def build_ray_cond(
    context_latents: torch.Tensor,   # (B, 48, M, H_lat, W_lat)
    rays: torch.Tensor,              # (B, RAY_CHANNELS, M+T, H_lat, W_lat) target-relative Plücker
    num_targets: int = 1,
) -> torch.Tensor:
    """Assemble the residual-branch input for M context + num_targets frames."""
    b, c_lat, m, h, w = context_latents.shape
    assert c_lat == LAYOUT.context_latent
    assert rays.shape[1] == LAYOUT.ray_features, rays.shape
    f = m + num_targets
    assert rays.shape[2] == f, (rays.shape, f)

    latent_plane = torch.zeros(b, LAYOUT.context_latent, f, h, w,
                               dtype=context_latents.dtype, device=context_latents.device)
    latent_plane[:, :, :m] = context_latents

    mask = torch.zeros(b, 1, f, h, w, dtype=context_latents.dtype, device=context_latents.device)
    mask[:, :, :m] = 1.0
    role = mask * 2 - 1  # +1 context / -1 target

    return torch.cat([latent_plane, mask, rays.to(context_latents.dtype), role], dim=1)
