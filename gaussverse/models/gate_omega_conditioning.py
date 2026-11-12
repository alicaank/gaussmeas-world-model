"""Sparse zero-output camera and scaffold routes for GATE-ΩR."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from gaussverse.camera.packed_rays import FRAMES_PER_LATENT, PACKED_RAY_CHANNELS
from gaussverse.camera.rays import RAY_CHANNELS
from gaussverse.models.ray_encoder import DIT_DIM, LATENT_CHANNELS, ResidualConvBlock
from gaussverse.models.trajectory_camera import COMMAND_CAMERA_FEATURE_DIM
from gaussverse.training.gate_omega import READER_BLOCKS


def _validate_blocks(blocks: tuple[int, ...]) -> tuple[int, ...]:
    blocks = tuple(int(value) for value in blocks)
    if not blocks or len(set(blocks)) != len(blocks) or min(blocks) < 0:
        raise ValueError("injection blocks must be unique nonnegative indices")
    return blocks


class _ZeroSparseImageProjection(nn.Module):
    """Shared full-rank output plus cheap block-specific low-rank corrections.

    A separate ``in_channels -> dim`` stride-two convolution at all thirty Wan
    blocks costs hundreds of millions of parameters.  Sharing the full-rank
    map preserves an all-block condition, while each zero-initialized down map
    gives its block an immediately trainable, distinct direction through the
    shared nonzero up map.  Both paths are exactly zero at initialization.
    """

    def __init__(self, in_channels: int, dim: int, blocks: tuple[int, ...],
                 rank: int = 16):
        super().__init__()
        self.blocks = _validate_blocks(blocks)
        self.rank = min(int(rank), int(in_channels), int(dim))
        if self.rank < 1:
            raise ValueError("block-specific condition rank must be positive")
        self.shared_projection = nn.Conv2d(in_channels, dim, 2, stride=2)
        self.projections = nn.ModuleDict({
            str(block): nn.Conv2d(in_channels, self.rank, 2, stride=2)
            for block in self.blocks
        })
        self.shared_up = nn.Linear(self.rank, dim, bias=False)
        nn.init.zeros_(self.shared_projection.weight)
        nn.init.zeros_(self.shared_projection.bias)
        for projection in self.projections.values():
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
        nn.init.trunc_normal_(self.shared_up.weight, std=dim ** -0.5)

    def forward(self, features: torch.Tensor, *, batch: int,
                frames: int) -> dict[int, torch.Tensor]:
        shared = self.shared_projection(features).flatten(2).transpose(1, 2)
        outputs = {}
        for block, projection in self.projections.items():
            low_rank = projection(features).flatten(2).transpose(1, 2)
            tokens = shared + self.shared_up(low_rank)
            outputs[int(block)] = tokens.reshape(batch, frames * tokens.shape[1], -1)
        return outputs


class OmegaOrderedRayStack(nn.Module):
    """Keep all four sub-RGB Plücker planes separate until the sparse Q/K output."""

    def __init__(self, *, ray_channels: int = RAY_CHANNELS, hidden: int = 64,
                 dim: int = DIT_DIM, blocks: tuple[int, ...] = READER_BLOCKS):
        super().__init__()
        self.ray_channels = int(ray_channels)
        self.hidden = int(hidden)
        self.blocks = _validate_blocks(blocks)
        self.plane_encoder = nn.Sequential(
            nn.Conv2d(self.ray_channels, hidden, 3, padding=1),
            ResidualConvBlock(hidden), ResidualConvBlock(hidden),
        )
        self.order_embedding = nn.Parameter(torch.empty(FRAMES_PER_LATENT, hidden))
        nn.init.trunc_normal_(self.order_embedding, std=hidden ** -0.5)
        self.output = _ZeroSparseImageProjection(
            FRAMES_PER_LATENT * hidden, dim, self.blocks)

    def encode(self, packed_rays: torch.Tensor,
               subframe_validity: torch.Tensor | None = None) -> torch.Tensor:
        if packed_rays.ndim != 5:
            raise ValueError("packed rays must be Bx(4R)xFxHxW")
        batch, channels, frames, height, width = packed_rays.shape
        if channels != FRAMES_PER_LATENT * self.ray_channels:
            raise ValueError(
                f"ordered rays require {FRAMES_PER_LATENT * self.ray_channels} channels")
        planes = packed_rays.reshape(
            batch, FRAMES_PER_LATENT, self.ray_channels, frames, height, width)
        planes = planes.permute(0, 3, 1, 2, 4, 5).reshape(
            batch * frames * FRAMES_PER_LATENT, self.ray_channels, height, width)
        encoded = self.plane_encoder(planes).reshape(
            batch, frames, FRAMES_PER_LATENT, self.hidden, height, width)
        order = self.order_embedding.to(encoded).view(
            1, 1, FRAMES_PER_LATENT, self.hidden, 1, 1)
        encoded = encoded + order
        if subframe_validity is not None:
            if subframe_validity.shape != (batch, frames, FRAMES_PER_LATENT):
                raise ValueError("subframe validity must be BxFx4")
            validity = subframe_validity.to(encoded).view(
                batch, frames, FRAMES_PER_LATENT, 1, 1, 1)
            encoded = encoded * validity
        return encoded.reshape(
            batch * frames, FRAMES_PER_LATENT * self.hidden, height, width)

    def forward(self, packed_rays: torch.Tensor,
                subframe_validity: torch.Tensor | None = None) -> dict[int, torch.Tensor]:
        batch, _, frames = packed_rays.shape[:3]
        return self.output(self.encode(packed_rays, subframe_validity),
                           batch=batch, frames=frames)


class OmegaCommandQKStack(nn.Module):
    """Sparse Q/K-only projection of the 45-D ordered command representation."""

    def __init__(self, *, feature_dim: int = COMMAND_CAMERA_FEATURE_DIM,
                 hidden: int = 256, dim: int = DIT_DIM,
                 blocks: tuple[int, ...] = READER_BLOCKS, rank: int = 16):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.blocks = _validate_blocks(blocks)
        self.rank = min(int(rank), int(hidden), int(dim))
        if self.rank < 1:
            raise ValueError("block-specific command rank must be positive")
        self.encoder = nn.Sequential(nn.Linear(feature_dim, hidden), nn.SiLU())
        self.shared_projection = nn.Linear(hidden, dim)
        self.projections = nn.ModuleDict({
            str(block): nn.Linear(hidden, self.rank) for block in self.blocks
        })
        self.shared_up = nn.Linear(self.rank, dim, bias=False)
        nn.init.zeros_(self.shared_projection.weight)
        nn.init.zeros_(self.shared_projection.bias)
        for projection in self.projections.values():
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
        nn.init.trunc_normal_(self.shared_up.weight, std=dim ** -0.5)

    def forward(self, features: torch.Tensor,
                tokens_per_frame: int) -> dict[int, torch.Tensor]:
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError("command features must be BxFx45")
        if tokens_per_frame < 1:
            raise ValueError("tokens_per_frame must be positive")
        hidden = self.encoder(features)
        shared = self.shared_projection(hidden)
        return {
            int(block): (shared + self.shared_up(projection(hidden))).repeat_interleave(
                tokens_per_frame, dim=1)
            for block, projection in self.projections.items()
        }


@dataclass(frozen=True)
class OmegaScaffoldLayout:
    latent_features: int = LATENT_CHANNELS
    inverse_depth: int = 1
    validity: int = 1
    confidence: int = 1
    disagreement: int = 1
    age: int = 1
    source_one_hot: int = 2
    unknown: int = 1
    sigma: int = 1

    @property
    def total(self) -> int:
        return sum(vars(self).values())


OMEGA_SCAFFOLD_LAYOUT = OmegaScaffoldLayout()


def build_scaffold_tensor(buffers: dict[str, torch.Tensor], sigma: torch.Tensor,
                          *, layout: OmegaScaffoldLayout = OMEGA_SCAFFOLD_LAYOUT) -> torch.Tensor:
    mapping = (
        ("features", layout.latent_features), ("inverse_depth", layout.inverse_depth),
        ("validity", layout.validity), ("confidence", layout.confidence),
        ("disagreement", layout.disagreement), ("age", layout.age),
        ("source_one_hot", layout.source_one_hot), ("unknown", layout.unknown),
    )
    tensors = []
    reference = buffers.get("features")
    if reference is None or reference.ndim != 5:
        raise ValueError("scaffold features must be BxCxTxHxW")
    batch, _, frames, height, width = reference.shape
    for name, channels in mapping:
        tensor = buffers.get(name)
        if tensor is None or tensor.shape != (batch, channels, frames, height, width):
            raise ValueError(f"scaffold field {name} has the wrong shape")
        tensors.append(tensor.to(reference))
    if sigma.ndim == 0:
        sigma = sigma.expand(batch, frames)
    if sigma.shape != (batch, frames) or not torch.isfinite(sigma).all() \
            or bool((sigma < 0).any()):
        raise ValueError("scaffold sigma must be a finite nonnegative BxT tensor")
    sigma_plane = sigma.to(reference).view(batch, 1, frames, 1, 1).expand(
        batch, 1, frames, height, width)
    result = torch.cat([*tensors, sigma_plane], dim=1)
    if result.shape[1] != layout.total:
        raise AssertionError("scaffold layout accounting error")
    return result


class OmegaScaffoldStack(nn.Module):
    """Target-aligned scaffold encoder with one zero projection at each injection site."""

    def __init__(self, *, layout: OmegaScaffoldLayout = OMEGA_SCAFFOLD_LAYOUT,
                 hidden: int = 128, dim: int = DIT_DIM,
                 blocks: tuple[int, ...] = READER_BLOCKS):
        super().__init__()
        self.layout = layout
        self.blocks = _validate_blocks(blocks)
        self.in_scale = nn.Parameter(torch.ones(layout.total))
        self.in_bias = nn.Parameter(torch.zeros(layout.total))
        self.encoder = nn.Sequential(
            nn.Conv2d(layout.total, hidden, 3, padding=1),
            ResidualConvBlock(hidden), ResidualConvBlock(hidden),
        )
        self.output = _ZeroSparseImageProjection(hidden, dim, self.blocks)

    def encode(self, scaffold: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        if scaffold.ndim != 5 or scaffold.shape[1] != self.layout.total:
            raise ValueError("scaffold must be BxExTxHxW with the contracted layout")
        batch, channels, frames, height, width = scaffold.shape
        values = scaffold.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, channels, height, width)
        rms = values.float().square().mean(dim=1, keepdim=True).add(1e-6).rsqrt()
        values = values * rms.to(values.dtype)
        values = values * self.in_scale.view(1, -1, 1, 1) \
            + self.in_bias.view(1, -1, 1, 1)
        return self.encoder(values), batch, frames

    def forward(self, scaffold: torch.Tensor) -> dict[int, torch.Tensor]:
        encoded, batch, frames = self.encode(scaffold)
        result = self.output(encoded, batch=batch, frames=frames)
        import os
        step = int(os.environ.get("GAUSSVERSE_AUDIT_STEP", "-1"))
        if (os.environ.get("GAUSSVERSE_MEMORY_AUDIT") and step >= 0 and step % 10 == 0
                and getattr(self, "_memory_audit_step", None) != step):
            from gaussverse.memory.audit import emit
            with torch.no_grad():
                emit("scaffold_injection", input_rms=float(scaffold.detach().float().square().mean().sqrt()),
                     residual_rms_by_block={str(k):float(v.detach().float().square().mean().sqrt())
                                            for k,v in result.items()},
                     meaning="nonzero injected scaffold residual; not proof of benefit")
            self._memory_audit_step=step
        return result


def merge_sparse_routes(*routes: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
    """Add independently zero-initialized sparse routes with strict shape agreement."""
    merged: dict[int, torch.Tensor] = {}
    for route in routes:
        for block, tokens in route.items():
            block = int(block)
            if block in merged and merged[block].shape != tokens.shape:
                raise ValueError(f"sparse route shape mismatch at block {block}")
            merged[block] = tokens if block not in merged else merged[block] + tokens
    return merged
