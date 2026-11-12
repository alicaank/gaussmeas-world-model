"""Heterogeneous history packing for the bidirectional Wan teacher.

The teacher remains bidirectional: this module does *not* turn Wan into a causal
student.  It changes the token budget presented to each self-attention block so that
important frames (the observed anchor, recent history, and supervised future) remain
spatially dense while older history is spatially pooled.  This is the useful part of
Lyra's FramePack recipe for teacher training; causal KV execution belongs to the later
student/distillation phase.

Packing happens after the stock patch embedding.  Consequently the pretrained 48-channel
input contract is unchanged and every target token can be unpatchified exactly.  Targets
must be the final, full-resolution entries in a plan; validation makes that invariant
explicit rather than relying on a fragile ordering convention.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


ROLE_NAMES = (
    "anchor_observed",
    "history_observed",
    "history_generated",
    "history_old",
    "history_recent",
    "retrieved_observed",
    "retrieved_generated",
    "target",
)
ROLE_TO_ID = {name: index for index, name in enumerate(ROLE_NAMES)}


@dataclass(frozen=True)
class FramePackEntry:
    """One input frame and its packed spatial resolution."""

    frame: int
    role: str
    spatial_stride: int = 1
    authority: float = 1.0

    def __post_init__(self) -> None:
        if self.frame < 0:
            raise ValueError("frame index must be nonnegative")
        if self.role not in ROLE_TO_ID:
            raise ValueError(f"unknown FramePack role {self.role!r}")
        if self.spatial_stride not in {1, 2, 4}:
            raise ValueError("FramePack spatial stride must be one of 1, 2, or 4")
        if not 0.0 <= self.authority <= 1.0:
            raise ValueError("FramePack authority must lie in [0,1]")


@dataclass(frozen=True)
class FramePackPlan:
    """Validated heterogeneous layout for one uniformly-patchified sample."""

    entries: tuple[FramePackEntry, ...]
    input_frames: int
    target_frames: int

    def __post_init__(self) -> None:
        if self.input_frames < 1 or not 1 <= self.target_frames <= self.input_frames:
            raise ValueError("invalid input/target frame count")
        if not self.entries:
            raise ValueError("FramePack plan cannot be empty")
        frames = [entry.frame for entry in self.entries]
        if len(set(frames)) != len(frames):
            raise ValueError("a FramePack plan cannot contain a frame twice")
        if any(frame >= self.input_frames for frame in frames):
            raise ValueError("FramePack entry is outside the input sequence")
        expected_targets = list(range(self.input_frames - self.target_frames,
                                      self.input_frames))
        target_entries = list(self.entries[-self.target_frames:])
        if [entry.frame for entry in target_entries] != expected_targets:
            raise ValueError("supervised targets must be the final FramePack entries")
        if any(entry.role != "target" or entry.spatial_stride != 1
               for entry in target_entries):
            raise ValueError("targets must use the target role at full spatial resolution")
        if any(entry.role == "target" for entry in self.entries[:-self.target_frames]):
            raise ValueError("only supervised trailing frames may use the target role")

    @property
    def target_entries(self) -> tuple[FramePackEntry, ...]:
        return self.entries[-self.target_frames:]

    def token_count(self, height: int, width: int) -> int:
        self._validate_grid(height, width)
        return sum((height // e.spatial_stride) * (width // e.spatial_stride)
                   for e in self.entries)

    def target_token_count(self, height: int, width: int) -> int:
        self._validate_grid(height, width)
        return self.target_frames * height * width

    def representative_indices(self, height: int, width: int, *,
                               device: torch.device | None = None) -> torch.Tensor:
        """Full-grid token index represented by each packed token."""
        self._validate_grid(height, width)
        indices = []
        for entry in self.entries:
            stride = entry.spatial_stride
            yy, xx = torch.meshgrid(
                torch.arange(stride // 2, height, stride, device=device),
                torch.arange(stride // 2, width, stride, device=device),
                indexing="ij",
            )
            indices.append(entry.frame * height * width + (yy * width + xx).reshape(-1))
        return torch.cat(indices)

    def packed_frame_ids(self, height: int, width: int, *,
                         device: torch.device | None = None) -> torch.Tensor:
        self._validate_grid(height, width)
        return torch.cat([
            torch.full(
                ((height // entry.spatial_stride) * (width // entry.spatial_stride),),
                entry.frame, dtype=torch.long, device=device,
            )
            for entry in self.entries
        ])

    def _validate_grid(self, height: int, width: int) -> None:
        if height < 1 or width < 1:
            raise ValueError("token grid must be positive")
        for entry in self.entries:
            if height % entry.spatial_stride or width % entry.spatial_stride:
                raise ValueError(
                    f"grid {(height, width)} is not divisible by stride "
                    f"{entry.spatial_stride} for frame {entry.frame}"
                )


def make_teacher_framepack(
    num_context: int,
    target_frames: int,
    *,
    generated_context: Iterable[int] = (),
    anchor_context: Iterable[int] | None = None,
    included_context: Iterable[int] | None = None,
    old_stride: int = 2,
    recent_frames: int = 3,
    context_authority: Iterable[float] | None = None,
) -> FramePackPlan:
    """Build the default static-world teacher layout.

    By default context position zero is the immutable observed anchor, preserving the
    training contract. ``anchor_context=()`` is the honest legacy-rollout diagnostic for
    a sliding window after the real anchor has left it. Generated history never inherits
    the authority or role of a real observation, regardless of tensor position.
    """
    if num_context < 1 or target_frames < 1 or recent_frames < 1:
        raise ValueError("context, targets, and recent_frames must be positive")
    generated = {int(index) for index in generated_context}
    anchors = ({0} if anchor_context is None
               else {int(index) for index in anchor_context})
    if any(index < 0 or index >= num_context for index in generated):
        raise ValueError("generated context index is outside the context prefix")
    if any(index < 0 or index >= num_context for index in anchors):
        raise ValueError("anchor context index is outside the context prefix")
    if generated & anchors:
        raise ValueError("a context slot cannot be both generated and an observed anchor")
    included = (set(range(num_context)) if included_context is None
                else {int(index) for index in included_context})
    authority_values = (None if context_authority is None
                        else [float(value) for value in context_authority])
    if authority_values is not None and (len(authority_values) != num_context or any(
            not 0.0 <= value <= 1.0 for value in authority_values)):
        raise ValueError("context authority must provide one [0,1] scalar per context")
    if not anchors.issubset(included):
        raise ValueError("an immutable observed anchor cannot be dropped")
    if any(index < 0 or index >= num_context for index in included):
        raise ValueError("included context index is outside the context prefix")
    entries: list[FramePackEntry] = []
    recent_start = max(1, num_context - recent_frames)
    for frame in range(num_context):
        if frame not in included:
            continue
        if frame in anchors:
            role, stride = "anchor_observed", 1
        elif frame in generated:
            role = "history_generated"
            stride = 1 if frame >= recent_start else old_stride
        elif frame >= recent_start:
            role, stride = "history_recent", 1
        else:
            role, stride = "history_old", old_stride
        authority = (authority_values[frame] if authority_values is not None
                     else 0.5 if frame in generated else 1.0)
        entries.append(FramePackEntry(frame, role, stride, authority))
    entries.extend(
        FramePackEntry(num_context + offset, "target", 1, 1.0)
        for offset in range(target_frames)
    )
    return FramePackPlan(tuple(entries), num_context + target_frames, target_frames)


class FramePackConditioner(nn.Module):
    """Role/provenance conditioning plus deterministic heterogeneous token packing."""

    def __init__(self, dim: int = 3072, in_dim: int = 48,
                 patch_size: tuple[int, int, int] = (1, 2, 2),
                 geometry_blocks: tuple[int, ...] = (7, 15, 23)) -> None:
        super().__init__()
        self.dim = int(dim)
        self.history_patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.role_embeddings = nn.Embedding(len(ROLE_NAMES), dim)
        self.authority_projection = nn.Sequential(
            nn.Linear(1, min(128, dim)), nn.SiLU(), nn.Linear(min(128, dim), dim)
        )
        self.geometry_gates = nn.ParameterDict({
            str(block): nn.Parameter(torch.tensor(0.0)) for block in geometry_blocks
        })
        # Exact continuation from a pretrained/unpacked checkpoint.  The layout changes
        # token count, but no arbitrary semantic offset is injected at initialization.
        nn.init.zeros_(self.role_embeddings.weight)
        nn.init.zeros_(self.authority_projection[-1].weight)
        nn.init.zeros_(self.authority_projection[-1].bias)

    @torch.no_grad()
    def initialize_history_patch(self, stock_patch_embedding: nn.Conv3d) -> None:
        """Copy Wan's pretrained patch projection into the clean-history branch."""
        if (self.history_patch_embedding.weight.shape != stock_patch_embedding.weight.shape
                or self.history_patch_embedding.bias.shape != stock_patch_embedding.bias.shape):
            raise ValueError("history and stock patch embeddings are shape-incompatible")
        self.history_patch_embedding.weight.copy_(stock_patch_embedding.weight)
        self.history_patch_embedding.bias.copy_(stock_patch_embedding.bias)

    def gated_geometry_bias(self, bias: torch.Tensor) -> dict[int, torch.Tensor]:
        """Sparse, exact-zero initialization for physical attention structure."""
        return {
            int(block): torch.tanh(gate).to(dtype=bias.dtype, device=bias.device) * bias
            for block, gate in self.geometry_gates.items()
        }

    @staticmethod
    def _pool_frame(frame: torch.Tensor, stride: int) -> torch.Tensor:
        """Pool BxHxWxD and return Bx(H/s)x(W/s)xD."""
        if stride == 1:
            return frame
        pooled = F.avg_pool2d(frame.permute(0, 3, 1, 2), stride, stride)
        return pooled.permute(0, 2, 3, 1).contiguous()

    def pack_grid(self, value: torch.Tensor, plan: FramePackPlan, *,
                  add_roles: bool = False) -> torch.Tensor:
        """Pack a BxFxHxWxD grid into BxSxD."""
        if value.ndim != 5 or value.shape[1] != plan.input_frames:
            raise ValueError("FramePack grid must be BxFxHxWxD and match the plan")
        if add_roles and value.shape[-1] != self.dim:
            raise ValueError("role embeddings require the configured hidden dimension")
        plan._validate_grid(value.shape[2], value.shape[3])
        packed = []
        for entry in plan.entries:
            tokens = self._pool_frame(value[:, entry.frame], entry.spatial_stride)
            tokens = tokens.reshape(value.shape[0], -1, value.shape[-1])
            if add_roles:
                role = self.role_embeddings.weight[ROLE_TO_ID[entry.role]].view(1, 1, -1)
                authority = value.new_full((value.shape[0], 1, 1), entry.authority)
                tokens = tokens + role.to(tokens.dtype) + self.authority_projection(
                    authority.float()
                ).to(tokens.dtype)
            packed.append(tokens)
        return torch.cat(packed, dim=1)

    def pack_tokens(self, value: torch.Tensor, plan: FramePackPlan,
                    height: int, width: int, *, add_roles: bool = False) -> torch.Tensor:
        if value.ndim != 3 or value.shape[1] != plan.input_frames * height * width:
            raise ValueError("flat token tensor does not match FramePack grid")
        grid = value.reshape(value.shape[0], plan.input_frames, height, width, value.shape[-1])
        return self.pack_grid(grid, plan, add_roles=add_roles)

    @staticmethod
    def pack_matrices(value: torch.Tensor, plan: FramePackPlan,
                      height: int, width: int) -> torch.Tensor:
        """Pack per-token rigid transforms without averaging rotation matrices."""
        if value.ndim != 4 or value.shape[1:] != (
                plan.input_frames * height * width, 4, 4):
            raise ValueError("matrix tensor does not match FramePack grid")
        plan._validate_grid(height, width)
        grid = value.reshape(value.shape[0], plan.input_frames, height, width, 4, 4)
        packed = []
        for entry in plan.entries:
            stride = entry.spatial_stride
            frame = grid[:, entry.frame, stride // 2::stride, stride // 2::stride]
            packed.append(frame.reshape(value.shape[0], -1, 4, 4))
        return torch.cat(packed, dim=1)

    @staticmethod
    def packed_freqs(dit, plan: FramePackPlan, frame_temporal_pos: torch.Tensor | None,
                     height: int, width: int, *, device: torch.device) -> torch.Tensor:
        """Build RoPE frequencies at the retained spatial sample centres."""
        plan._validate_grid(height, width)
        if frame_temporal_pos is not None and frame_temporal_pos.ndim == 2:
            return torch.stack([
                FramePackConditioner.packed_freqs(
                    dit, plan, positions, height, width, device=device
                )
                for positions in frame_temporal_pos
            ], dim=0)
        if frame_temporal_pos is None:
            frame_temporal_pos = torch.full((plan.input_frames,), -1, dtype=torch.long)
        if frame_temporal_pos.shape != (plan.input_frames,):
            raise ValueError("temporal positions do not match FramePack inputs")
        positive = frame_temporal_pos[frame_temporal_pos >= 0]
        if positive.numel() and int(positive.max()) >= len(dit.freqs[0]):
            raise ValueError("FramePack temporal position exceeds the model RoPE table")
        ident = torch.ones_like(dit.freqs[0][0])
        result = []
        for entry in plan.entries:
            position = int(frame_temporal_pos[entry.frame])
            temporal = dit.freqs[0][position] if position >= 0 else ident
            stride = entry.spatial_stride
            # Average pooling represents each cell at its integer centre sample.  This
            # keeps spatial addresses monotone and deterministic without inventing
            # interpolated complex rotations.
            y = torch.arange(stride // 2, height, stride, device=dit.freqs[1].device)
            x = torch.arange(stride // 2, width, stride, device=dit.freqs[2].device)
            if len(y) != height // stride or len(x) != width // stride:
                raise AssertionError("FramePack centre sampling disagrees with pooling")
            ft = temporal.view(1, 1, -1).expand(len(y), len(x), -1)
            fy = dit.freqs[1][y].view(len(y), 1, -1).expand(len(y), len(x), -1)
            fx = dit.freqs[2][x].view(1, len(x), -1).expand(len(y), len(x), -1)
            combined = torch.cat([ft, fy, fx], dim=-1)
            result.append(combined.reshape(-1, 1, combined.shape[-1]))
        return torch.cat(result, dim=0).to(device)
