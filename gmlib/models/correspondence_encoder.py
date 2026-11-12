"""Correspondence conditioning branch -- Proposal 1 of the memory/conditioning
ideas report (outputs/ideas_report.html).

Every existing adapter in this codebase (RayContextProjection,
MemoryEvidenceAdapter) adds its output to the hidden state `x` BEFORE it splits
into Q, K, and V -- so a wrong signal there can inject wrong CONTENT into the
generated pixels. That is the mechanism behind the one measured regression in
the memory-branch gate (a well-covered scene driven to 41 deg rotation error
under memory ON): warped evidence can be wrong, and today it has no way to be
merely ignored rather than actively misleading.

Lyra 2.0 (arXiv:2604.13036) splits this: retrieved canonical world coordinates
are injected ONLY into attention's queries and keys, never values --
`q=W_Q(x+p), k=W_K(x+p)`, content generation stays with the learned prior, and
only attention's ROUTING is geometry-biased. A wrong correspondence degrades to
uninformative routing (graceful); a wrong content residual actively hallucinates
pixels (not graceful). This module is that channel for GaussMeas: canonical
coordinates that `render_target_buffers` already computes and that
`data.evidence.build_target_evidence` has been silently discarding.

Same invariants as every other adapter here (do not relax casually):
  - norm-free local stem (spatial norms leak evidence across the whole frame),
  - zero-init projection => exact backbone equivalence at init: qk_bias=0 makes
    `x + qk_bias == x`, so SelfAttention.forward is bitwise unchanged,
  - gate initialized to 1.0, NOT 0 (zero gate on a zero projection deadlocks
    both gradients -- the double-zero rule),
  - no output LayerNorm, for the same reason MemoryEvidenceAdapter has none:
    this is bolted onto an already-converged camera checkpoint, so an output
    norm would turn the zero-init ramp into an immediate O(1) perturbation the
    instant the projection leaves zero.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from gmlib.models.ray_encoder import DIT_DIM, ResidualConvBlock, SPATIAL_PATCH


@dataclass(frozen=True)
class CorrespondenceLayout:
    """Per-target-frame correspondence planes at latent resolution.

    Deliberately thin compared to MemoryEvidenceAdapter's 51-channel content
    evidence: correspondence only needs to say WHERE a point is, not what it
    looks like, so there is no rendered-RGB-latent plane here at all.
    """

    canonical_coords: int = 3  # world xyz, normalized to scene extent, 0 in holes
    validity: int = 1          # 1 where the coordinate is real, 0 in holes

    @property
    def total(self) -> int:
        return self.canonical_coords + self.validity


CORR_LAYOUT = CorrespondenceLayout()


class CorrespondenceAdapter(nn.Module):
    """Zero-init: canonical-coordinate evidence -> per-token Q/K-only bias.

    Input:  (B, CORR_LAYOUT.total, T, H_lat, W_lat)
    Output: (B, T * H_lat/2 * W_lat/2, dim) in frame-major order -- consumed as
            `qk_bias` by model_fn, NOT added to the main token stream.
    """

    def __init__(self, channels: int = CORR_LAYOUT.total, hidden: int = 64, dim: int = DIT_DIM):
        super().__init__()
        self.channels = channels
        self.stem = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            ResidualConvBlock(hidden),
        )
        self.patch_projection = nn.Conv2d(hidden, dim, kernel_size=SPATIAL_PATCH, stride=SPATIAL_PATCH)
        self.gate = nn.Parameter(torch.tensor(1.0))
        nn.init.zeros_(self.patch_projection.weight)
        nn.init.zeros_(self.patch_projection.bias)

    def forward(self, corr: torch.Tensor) -> torch.Tensor:
        b, c, f, h, w = corr.shape
        assert c == self.channels, (c, self.channels)
        x = corr.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
        x = self.stem(x)
        x = self.patch_projection(x)              # (B*F, dim, h/2, w/2)
        x = x.flatten(2).transpose(1, 2)          # (B*F, hw/4, dim)
        x = x.reshape(b, -1, x.shape[-1])         # frame-major (f, h, w)
        return torch.tanh(self.gate) * x


def build_correspondence(coords: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Stack canonical coords + validity into the adapter's input tensor.

    Args:
        coords: (B, 3, T, H_lat, W_lat) canonical world xyz, 0 in holes.
        valid:  (B, 1, T, H_lat, W_lat).
    """
    assert coords.shape[1] == CORR_LAYOUT.canonical_coords, coords.shape
    assert valid.shape[1] == CORR_LAYOUT.validity, valid.shape
    return torch.cat([coords, valid.to(dtype=coords.dtype, device=coords.device)], dim=1)
