"""Causal trajectory sampling for world-model training and evaluation.

The original project sampler treats context images as an unordered view set and often
places them across the whole capture.  That is useful for novel-view reconstruction,
but it is not the deployment task: a walking agent observes a short, ordered history
and predicts the next point on the trajectory.

This module makes that distinction explicit.  Training tuples are always chronological,
use recent context, and put the target strictly after the observations.  Index spacing is
varied (to cover walking speed and prediction horizon), while a geometry-only continuity
gate rejects camera-track discontinuities.  It never looks at pixels or model outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


# Most training mass is one or two recorded frames ahead.  The longer tails teach the
# exact same model to tolerate missed/variable-rate observations without turning the
# task back into a widely-spread novel-view problem.
CONTEXT_STRIDES: tuple[int, ...] = (1, 2, 3)
CONTEXT_STRIDE_PROBS: tuple[float, ...] = (0.60, 0.30, 0.10)
TARGET_HORIZONS: tuple[int, ...] = (1, 2, 4, 8)
TARGET_HORIZON_PROBS: tuple[float, ...] = (0.55, 0.25, 0.15, 0.05)


@dataclass(frozen=True)
class TrajectorySpec:
    """One recent-history -> future-frame training tuple."""

    ctx: list[int]
    target: int
    context_stride: int
    horizon: int
    translation: float
    rotation_deg: float


@dataclass(frozen=True)
class ReplaySpec:
    """Clean prefix followed by generated steps and one clean supervision target."""

    ctx: list[int]
    rollout_targets: list[int]
    target: int
    stride: int


@dataclass(frozen=True)
class TrajectoryChunkSpec:
    """Recent observations followed by a contiguous multi-frame future chunk."""

    ctx: list[int]
    targets: list[int]
    context_stride: int
    target_stride: int


@dataclass(frozen=True)
class StaticTeacherChunkSpec:
    """Long anchor/old/recent history followed by one supervised future chunk."""

    ctx: list[int]
    targets: list[int]
    recent_start: int
    target_stride: int


def _adjacent_motion(c2w: torch.Tensor, scene_scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Translation and rotation for every consecutive camera pair.

    Translation is expressed in the same persistent scene-normalised units used by the
    rest of GaussMeas.  The arrays have length ``N-1`` and are cheap enough to cache once
    per scene in :class:`TrajectorySampler`.
    """
    c = c2w[:, :3, 3].double() / float(scene_scale)
    trans = torch.linalg.norm(c[1:] - c[:-1], dim=1)
    r = c2w[:, :3, :3].double()
    rel = r[:-1].transpose(1, 2) @ r[1:]
    trace = rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    rot = torch.rad2deg(torch.arccos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0)))
    return trans.cpu().numpy(), rot.cpu().numpy()


def _pair_motion(c2w: torch.Tensor, i: int, j: int, scene_scale: float) -> tuple[float, float]:
    c = c2w[[i, j], :3, 3].double() / float(scene_scale)
    trans = float(torch.linalg.norm(c[1] - c[0]))
    r = c2w[[i, j], :3, :3].double()
    rel = r[0].T @ r[1]
    trace = rel.diagonal().sum()
    rot = float(torch.rad2deg(torch.arccos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0))))
    return trans, rot


def select_observed_sources(
    scene,
    recent_ctx: list[int],
    target: int,
    *,
    k: int = 2,
    min_temporal_gap: int = 12,
    max_translation: float = 0.20,
    max_rotation_deg: float = 45.0,
    min_source_spacing: int = 6,
) -> list[int]:
    """Select immutable first-visit observations for a later target using cameras only.

    Sources must precede the recent causal window by ``min_temporal_gap`` recorded
    frames, so this cannot collapse into another copy of the three-frame context.  The
    score is target-relative translation plus normalised rotation and never consults
    pixels, targets, or model outputs.  Returning an empty list is intentional: ordinary
    forward motion has no justified persistent-memory read and should train the exact
    no-reader path.
    """
    if k < 1 or min_temporal_gap < 1 or min_source_spacing < 1:
        raise ValueError("k, temporal gap, and source spacing must be positive")
    if not recent_ctx or not 0 <= target < len(scene):
        raise ValueError("recent context and target must index the scene")
    last_source = min(recent_ctx) - min_temporal_gap
    if last_source < 0:
        return []
    scored = []
    for index in range(last_source + 1):
        translation, rotation = _pair_motion(scene.c2w, index, target, scene.s_scene)
        if (np.isfinite(translation) and np.isfinite(rotation)
                and translation <= max_translation and rotation <= max_rotation_deg):
            score = translation / max_translation + rotation / max_rotation_deg
            scored.append((score, translation, rotation, index))
    scored.sort()
    if len(scored) < k:
        return []
    # Consecutive video frames are almost identical keyframes. Without spacing, K=2
    # always selected adjacent frames on the held-out revisit audit (40/40 scenes),
    # wasting half of the reader's fixed token budget. Greedily keep the closest
    # target-compatible candidates that are temporally distinct, then relax only when
    # a short archive otherwise cannot provide K sources.
    selected = []
    for item in scored:
        index = item[-1]
        if all(abs(index - prior) >= min_source_spacing for prior in selected):
            selected.append(index)
            if len(selected) == k:
                return selected
    for item in scored:
        index = item[-1]
        if index not in selected:
            selected.append(index)
            if len(selected) == k:
                return selected
    return []


class TrajectorySampler:
    """Sample chronological recent-context tuples from one camera track.

    ``max_adjacent_*`` rejects capture cuts / tracking jumps by checking every recorded
    transition inside the tuple, including frames skipped by a stride.  The defaults are
    deliberately generous relative to measured DL3DV adjacent motion (median translation
    0.027 scene units, median rotation 1.37 degrees over 100 train scenes): they remove
    discontinuities, not fast walking or turns.
    """

    def __init__(
        self,
        scene,
        *,
        max_adjacent_translation: float = 0.20,
        max_adjacent_rotation_deg: float = 25.0,
    ) -> None:
        self.scene = scene
        self.max_adjacent_translation = float(max_adjacent_translation)
        self.max_adjacent_rotation_deg = float(max_adjacent_rotation_deg)
        self.adj_translation, self.adj_rotation = _adjacent_motion(scene.c2w, scene.s_scene)

    def _continuous(self, first: int, last: int) -> bool:
        if last <= first:
            return False
        sl = slice(first, last)
        return bool(
            np.all(np.isfinite(self.adj_translation[sl]))
            and np.all(np.isfinite(self.adj_rotation[sl]))
            and np.max(self.adj_translation[sl], initial=0.0) <= self.max_adjacent_translation
            and np.max(self.adj_rotation[sl], initial=0.0) <= self.max_adjacent_rotation_deg
        )

    def sample(
        self,
        num_ctx: int,
        rng: np.random.Generator,
        *,
        tries: int = 80,
    ) -> TrajectorySpec | None:
        if num_ctx < 1:
            raise ValueError("num_ctx must be positive")
        n = len(self.scene)
        for _ in range(tries):
            stride = int(rng.choice(CONTEXT_STRIDES, p=CONTEXT_STRIDE_PROBS))
            horizon = int(rng.choice(TARGET_HORIZONS, p=TARGET_HORIZON_PROBS))
            required = stride * (num_ctx - 1) + horizon
            if n <= required:
                continue
            first = int(rng.integers(0, n - required))
            ctx = [first + k * stride for k in range(num_ctx)]
            target = ctx[-1] + horizon
            if not self._continuous(first, target):
                continue
            trans, rot = _pair_motion(self.scene.c2w, ctx[-1], target, self.scene.s_scene)
            return TrajectorySpec(
                ctx=ctx,
                target=target,
                context_stride=stride,
                horizon=horizon,
                translation=trans,
                rotation_deg=rot,
            )
        return None

    def sample_replay(
        self,
        num_ctx: int,
        rng: np.random.Generator,
        *,
        rollout_steps: int = 2,
        tries: int = 80,
    ) -> ReplaySpec | None:
        """Sample a fixed-stride causal segment for detached rollout replay.

        ``rollout_targets`` are generated autoregressively. ``target`` is never
        generated and remains the clean flow-matching label. For a three-frame
        context and two rollout steps, training finally consumes
        ``[last_clean, generated_1, generated_2] -> clean_target``.
        """
        if num_ctx < 1 or rollout_steps < 1:
            raise ValueError("num_ctx and rollout_steps must be positive")
        n = len(self.scene)
        for _ in range(tries):
            stride = int(rng.choice(CONTEXT_STRIDES, p=CONTEXT_STRIDE_PROBS))
            intervals = num_ctx - 1 + rollout_steps + 1
            required = stride * intervals
            if n <= required:
                continue
            first = int(rng.integers(0, n - required))
            ctx = [first + k * stride for k in range(num_ctx)]
            future = [ctx[-1] + (k + 1) * stride for k in range(rollout_steps + 1)]
            if not self._continuous(first, future[-1]):
                continue
            return ReplaySpec(ctx, future[:-1], future[-1], stride)
        return None

    def sample_chunk(
        self,
        num_ctx: int,
        target_frames: int,
        rng: np.random.Generator,
        *,
        tries: int = 80,
    ) -> TrajectoryChunkSpec | None:
        """Sample a recent context and a jointly supervised future latent chunk.

        The future uses one fixed recorded-frame stride, so every slot is strictly
        causal and the chunk corresponds to a coherent camera segment.  A common
        coordinate frame (the final future camera) can then be used for all rays.
        """
        if num_ctx < 1 or target_frames < 1:
            raise ValueError("num_ctx and target_frames must be positive")
        n = len(self.scene)
        for _ in range(tries):
            context_stride = int(rng.choice(CONTEXT_STRIDES, p=CONTEXT_STRIDE_PROBS))
            target_stride = int(rng.choice(CONTEXT_STRIDES, p=CONTEXT_STRIDE_PROBS))
            required = context_stride * (num_ctx - 1) + target_stride * target_frames
            if n <= required:
                continue
            first = int(rng.integers(0, n - required))
            ctx = [first + k * context_stride for k in range(num_ctx)]
            targets = [ctx[-1] + (k + 1) * target_stride for k in range(target_frames)]
            if not self._continuous(first, targets[-1]):
                continue
            return TrajectoryChunkSpec(ctx, targets, context_stride, target_stride)
        return None

    def sample_static_teacher_chunk(
        self,
        num_ctx: int,
        target_frames: int,
        rng: np.random.Generator,
        *,
        recent_frames: int = 3,
        min_anchor_gap: int = 12,
        max_anchor_gap: int = 96,
        tries: int = 120,
    ) -> StaticTeacherChunkSpec | None:
        """Sample the heterogeneous history used by the static-world teacher.

        The first frame is a persistent observed anchor, middle frames span the old
        trajectory, and the final suffix is a dense recent window.  This avoids calling
        a three-adjacent-frame tuple "persistent memory" while remaining strictly causal.
        """
        if (num_ctx < 2 or target_frames < 1 or recent_frames < 1
                or min_anchor_gap < 1 or max_anchor_gap < min_anchor_gap):
            raise ValueError("invalid static-teacher sampling configuration")
        recent_frames = min(recent_frames, num_ctx - 1)
        old_frames = num_ctx - recent_frames - 1
        n = len(self.scene)
        for _ in range(tries):
            stride = int(rng.choice(CONTEXT_STRIDES, p=CONTEXT_STRIDE_PROBS))
            future_span = stride * target_frames
            recent_span = stride * recent_frames
            if n <= future_span + recent_span + min_anchor_gap:
                continue
            first_target = int(rng.integers(
                recent_span + min_anchor_gap, n - future_span + 1
            ))
            recent = [first_target - stride * (recent_frames - index)
                      for index in range(recent_frames)]
            latest_anchor = recent[0] - min_anchor_gap
            earliest_anchor = max(0, recent[0] - max_anchor_gap)
            if latest_anchor < earliest_anchor:
                continue
            anchor = int(rng.integers(earliest_anchor, latest_anchor + 1))
            if old_frames:
                # Interior samples are deterministic given endpoints and stay distinct.
                old = np.linspace(anchor, recent[0], old_frames + 2)[1:-1]
                old = [int(round(value)) for value in old]
                if len(set(old)) != len(old) or any(
                        not anchor < value < recent[0] for value in old):
                    continue
            else:
                old = []
            ctx = [anchor, *old, *recent]
            targets = [first_target + stride * index for index in range(target_frames)]
            if len(set(ctx + targets)) != len(ctx) + len(targets):
                continue
            if not self._continuous(anchor, targets[-1]):
                continue
            return StaticTeacherChunkSpec(ctx, targets, 1 + old_frames, stride)
        return None


def sample_legacy_sequential(scene, num_ctx: int, rng: np.random.Generator):
    """Historical full-FT tuple sampler retained for exact old-run reproduction.

    This is intentionally *not* used by the causal trajectory task: it sometimes chooses
    an interpolated target and does not continuity-gate the camera track.  Keeping the
    small compatibility function here lets the obsolete standalone temporal trainer be
    removed without changing the historical ``train_fullft --task viewset`` stream.
    """
    n = len(scene)
    d = int(rng.integers(2, 9))
    span = d * (num_ctx - 1)
    max_start = n - span - 4 * d - 1
    if max_start < 1:
        return None
    s0 = int(rng.integers(0, max_start))
    ctx = [s0 + k * d for k in range(num_ctx)]
    if rng.random() < 0.8:
        target = ctx[-1] + int(rng.integers(d, 4 * d + 1))
    else:
        target = int(rng.integers(ctx[0] + 1, ctx[-1]))
    return ctx, target


def evaluation_schedule(
    scene,
    *,
    context_window: int = 3,
    rollout_steps: int = 20,
    stride: int = 2,
    start_fraction: float = 0.10,
    max_adjacent_translation: float = 0.20,
    max_adjacent_rotation_deg: float = 25.0,
) -> tuple[list[int], list[int]] | None:
    """Fixed causal segment for teacher-forced and autoregressive evaluation.

    Candidate starts are searched deterministically from ``start_fraction`` onward until
    a geometry-continuous segment is found.  Both evaluation modes must consume the same
    returned indices; only the context *pixels* differ.
    """
    if context_window < 1 or rollout_steps < 1 or stride < 1:
        raise ValueError("context_window, rollout_steps and stride must be positive")
    sampler = TrajectorySampler(
        scene,
        max_adjacent_translation=max_adjacent_translation,
        max_adjacent_rotation_deg=max_adjacent_rotation_deg,
    )
    total_intervals = context_window - 1 + rollout_steps
    span = total_intervals * stride
    if len(scene) <= span:
        return None
    preferred = int(round((len(scene) - 1) * start_fraction))
    starts = list(range(preferred, len(scene) - span)) + list(range(0, preferred))
    for first in starts:
        last = first + span
        if sampler._continuous(first, last):
            ctx = [first + k * stride for k in range(context_window)]
            targets = [ctx[-1] + (k + 1) * stride for k in range(rollout_steps)]
            return ctx, targets
    return None


def trajectory_path_profile(scene, indices: list[int]) -> dict:
    """Geometry-only motion statistics and overlapping path-family labels."""
    if len(indices) < 2 or any(not 0 <= index < len(scene) for index in indices):
        raise ValueError("path profile requires at least two valid scene indices")
    poses = scene.c2w[indices].detach().double().cpu()
    centres = poses[:, :3, 3] / float(scene.s_scene)
    delta = centres[1:] - centres[:-1]
    speed = torch.linalg.norm(delta, dim=1)
    path_length = float(speed.sum())
    net_translation = float(torch.linalg.norm(centres[-1] - centres[0]))
    net_to_path = net_translation / max(path_length, 1e-9)
    speed_mean = float(speed.mean())
    speed_cv = float(speed.std(unbiased=False) / max(speed_mean, 1e-9))

    rotations = poses[:, :3, :3]
    relative = rotations[:-1].transpose(1, 2) @ rotations[1:]
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(-1)
    step_rotation = torch.rad2deg(torch.arccos(((trace - 1) / 2).clamp(-1, 1)))
    cumulative_rotation = float(step_rotation.sum())
    net_relative = rotations[0].T @ rotations[-1]
    net_rotation = float(torch.rad2deg(torch.arccos(
        ((net_relative.diagonal().sum() - 1) / 2).clamp(-1, 1)
    )))

    forward = rotations[:-1, :3, 2]
    axial = torch.abs((delta * forward).sum(-1))
    lateral = torch.linalg.norm(delta - (delta * forward).sum(-1, keepdim=True) * forward,
                                dim=1)
    lateral_fraction = float((lateral / speed.clamp_min(1e-9)).median())
    axial_fraction = float((axial / speed.clamp_min(1e-9)).median())
    families = {
        "forward_translation": path_length > 0.10 and axial_fraction >= 0.50,
        "turn_plus_translation": path_length > 0.10 and cumulative_rotation >= 15.0,
        "lateral_parallax": path_length > 0.10 and lateral_fraction >= 0.60,
        "out_and_back": path_length > 0.20 and net_to_path <= 0.50,
        "loop_closure": (
            path_length > 0.20 and net_to_path <= 0.20 and net_rotation <= 20.0
        ),
        "sparse_or_variable_speed": speed_cv >= 0.50,
    }
    return {
        "path_length": path_length, "net_translation": net_translation,
        "net_to_path_ratio": net_to_path, "cumulative_rotation_deg": cumulative_rotation,
        "net_rotation_deg": net_rotation, "speed_cv": speed_cv,
        "median_axial_fraction": axial_fraction,
        "median_lateral_fraction": lateral_fraction,
        "families": families,
    }


def corrupt_context_latents(
    z0: torch.Tensor,
    num_ctx: int,
    *,
    strength: float,
    dropout_prob: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Approximate rollout exposure while preserving the clean supervision target.

    Generated context differs from encoded ground truth by both persistent (shared) and
    frame-specific errors.  We blend context latents toward a 50/50 mixture of shared and
    independent Gaussian noise.  Older frames receive more corruption because they have
    survived more autoregressive transitions.  Optional whole-frame dropout teaches the
    bounded-context model to survive a failed generation.

    The target slice is cloned but never modified; tests pin this invariant because
    corrupting it would silently change the flow-matching label.
    """
    if z0.ndim != 5:
        raise ValueError(f"expected (B,C,F,H,W), got {tuple(z0.shape)}")
    if not 0 <= num_ctx < z0.shape[2]:
        raise ValueError("num_ctx must leave at least one target frame")
    if not 0.0 <= strength <= 1.0 or not 0.0 <= dropout_prob <= 1.0:
        raise ValueError("strength and dropout_prob must be in [0,1]")
    out = z0.clone()
    if num_ctx == 0 or (strength == 0.0 and dropout_prob == 0.0):
        return out

    shape = out[:, :, :num_ctx].shape
    shared = torch.randn(
        (shape[0], shape[1], 1, shape[3], shape[4]),
        dtype=out.dtype,
        device=out.device,
        generator=generator,
    )
    independent = torch.randn(shape, dtype=out.dtype, device=out.device, generator=generator)
    noise = (shared + independent) / np.sqrt(2.0)
    # oldest -> newest: full configured strength down to half strength
    age = torch.linspace(1.0, 0.5, num_ctx, dtype=out.dtype, device=out.device)
    alpha = (float(strength) * age).view(1, 1, num_ctx, 1, 1)
    out[:, :, :num_ctx] = (1.0 - alpha) * out[:, :, :num_ctx] + alpha * noise

    if dropout_prob > 0.0:
        drop = torch.rand(
            (shape[0], 1, num_ctx, 1, 1), device=out.device, generator=generator
        ) < dropout_prob
        out[:, :, :num_ctx] = torch.where(drop, torch.zeros_like(out[:, :, :num_ctx]),
                                                   out[:, :, :num_ctx])
    return out
