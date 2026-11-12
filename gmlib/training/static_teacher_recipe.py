"""Auditable recipe primitives for the GaussMeas static-world teacher.

This module deliberately contains policy, not a second model implementation.  The
training entry point composes these primitives with ``train_fullft``/``model_fn`` so the
teacher, evaluator, and later student use one renderer contract.
"""
from __future__ import annotations

import gm_env  # machine paths: gm_env.py at the repository root (scripts/env.sh puts it on PYTHONPATH)
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from gmlib.models.framepack import FramePackPlan, make_teacher_framepack
from gmlib.models.vae_stream import FRAMES_PER_LATENT, PRIME_FRAMES


DEFAULT_480P_ROOTS = tuple(
    Path(f"{gm_env.DATA_ROOT}/DL3DV-480P/{shard}")
    for shard in ("1K", "2K", "3K", "4K")
)
MIXED_SHARP_ROOTS = (
    Path((gm_env.DATA_ROOT + "/DL3DV-960P/1K")),
    *DEFAULT_480P_ROOTS[1:],
)


@dataclass(frozen=True)
class TeacherStage:
    name: str
    resolution: tuple[int, int]
    target_frames: int
    context_frames: int
    optimizer_steps: int
    old_stride: int
    retrieval_k: int
    self_aug_final_probability: float
    path_consistency_probability: float


TEACHER_STAGES = {
    "A": TeacherStage("A", (192, 320), 2, 6, 3000, 2, 0, 0.25, 0.10),
    "B": TeacherStage("B", (384, 640), 2, 8, 5000, 2, 2, 0.35, 0.15),
    "C": TeacherStage("C", (384, 640), 4, 10, 8000, 2, 3, 0.50, 0.20),
    # Stage D is a deliberate gate, not an automatic expense.  It exists so a passing
    # chunk-4 teacher can expand horizon without changing recipe semantics.
    "D": TeacherStage("D", (384, 640), 8, 12, 10000, 4, 4, 0.50, 0.25),
}


def validate_training_roots(roots: Iterable[Path], *, require_all_480p: bool = True) -> tuple[Path, ...]:
    roots = tuple(Path(root).resolve() for root in roots)
    if len(roots) != 4 or len(set(roots)) != 4:
        raise ValueError("teacher training requires four distinct DL3DV shards")
    missing = [str(root) for root in roots if not root.is_dir()]
    if missing:
        raise FileNotFoundError(f"missing DL3DV roots: {missing}")
    if require_all_480p and any("DL3DV-480P" not in str(root) for root in roots):
        raise ValueError("the selected recipe requires 480P for all four training shards")
    return roots


def optimizer_backend(*, trainable_parameters: int, gpu_bytes: int | None,
                      requested: str = "auto") -> str:
    """Choose AdamW state precision from a conservative peak-memory estimate.

    8-bit optimizer state is not a methodological requirement.  It is selected only
    when fp32 Adam moments plus master weights would leave less than 12 GiB for the 5B
    bf16 model and checkpointed activations.  On an 80-GiB accelerator this returns
    ordinary AdamW, matching Lyra's released optimizer; on the local 32-GiB 5090 it
    returns paged AdamW8bit for the ~1.1B selective tier.
    """
    if requested not in {"auto", "adamw", "adam8bit", "paged_adam8bit"}:
        raise ValueError(f"unknown optimizer request {requested!r}")
    if requested != "auto":
        return requested
    if gpu_bytes is None:
        return "adamw"
    # fp32 gradient + two moments + master copy is conservatively 16 bytes/parameter.
    optimizer_and_trainable = 16 * int(trainable_parameters)
    frozen_bf16_model = 11 * (1 << 30)  # Wan2.2-5B plus active small branches, rounded up
    activation_reserve = 12 * (1 << 30)
    return ("adamw" if optimizer_and_trainable + frozen_bf16_model + activation_reserve
            <= gpu_bytes else "paged_adam8bit")


def is_selective_backbone_parameter(name: str) -> bool:
    """Lyra-style teacher tier: self-attention and the stock patch embedding only."""
    return "self_attn" in name or name.startswith("patch_embedding.")


def set_selective_backbone_trainable(dit, *, dtype: torch.dtype = torch.float32):
    """Unfreeze the selective tier, holding the trainable weights in `dtype`.

    The default is fp32 and must stay fp32: a bf16 master weight cannot represent its own
    update. bf16 keeps 8 significand bits, so `w + d == w` whenever `|d|` is below half an
    ulp, i.e. below ~2^-9 of `|w|`. An AdamW step is ~lr by construction, reduced to
    ~0.23*lr by `m/sqrt(v)` under noisy gradients; at lr 3e-5 that is 7e-6, which clears
    half an ulp only for weights below ~3.6e-3. Measured on the 448x256 run: after 1000
    optimizer steps **88.17% of the 1.134B trainable parameters were bit-identical**, the
    RMSNorm gains (|w| ~ 1.33) were 100.00% frozen, and decoupled weight decay was a
    complete no-op. The rounding happens on write-back, so a discarded update is not
    deferred -- the weight is an absorbing state and never moves, at any step count.

    Compute is unaffected: the forward runs under `torch.autocast(bf16)`, which casts
    operands per-op, so fp32 masters cost memory but not speed. The extra memory is paid
    back by 8-bit optimizer state -- see `optimizer_backend`, whose estimate already
    assumes an fp32 master copy.
    """
    if dtype != torch.float32:
        raise ValueError(
            f"trainable backbone must be fp32, got {dtype}: a {dtype} master weight "
            "rounds away its own AdamW update (measured 88.17% of parameters frozen)")
    dit.requires_grad_(False)
    selected = []
    for name, parameter in dit.named_parameters():
        if is_selective_backbone_parameter(name):
            parameter.data = parameter.data.to(dtype)
            parameter.requires_grad_(True)
            selected.append(parameter)
    if not selected:
        raise RuntimeError("selective backbone policy matched no parameters")
    return selected


def assert_trainable_precision(named_parameters) -> int:
    """Every trainable parameter must be fp32. Returns how many were checked.

    A branch built in bf16 has exactly the same write-back problem as a bf16 backbone,
    and is easier to miss because the branches are small enough that nobody looks.
    """
    offenders = sorted(
        {f"{name} ({parameter.dtype})" for name, parameter in named_parameters
         if parameter.requires_grad and parameter.dtype != torch.float32}
    )
    if offenders:
        raise RuntimeError(
            "trainable parameters must be fp32 so AdamW updates survive write-back; "
            f"{len(offenders)} are not: {offenders[:6]}")
    return sum(1 for _, p in named_parameters if p.requires_grad)


def split_weight_decay(parameters_with_names, *, weight_decay: float = 1e-3):
    """Separate matrices from bias/norm/gate/embedding parameters."""
    decay, no_decay = [], []
    for name, parameter in parameters_with_names:
        if not parameter.requires_grad:
            continue
        lower = name.lower()
        target = (no_decay if parameter.ndim < 2 or any(
            marker in lower for marker in ("bias", "norm", "gate", "embedding")
        ) else decay)
        target.append(parameter)
    if len({id(p) for p in decay + no_decay}) != len(decay) + len(no_decay):
        raise ValueError("optimizer parameter groups contain duplicates")
    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def self_augmentation_probability(step: int, total_steps: int, final: float) -> float:
    """Anchor-safe on-policy exposure: 0 -> half strength -> full strength."""
    if step < 0 or total_steps < 1 or not 0.0 <= final <= 1.0:
        raise ValueError("invalid self-augmentation schedule")
    fraction = min(step / total_steps, 1.0)
    if fraction < 0.10:
        return 0.0
    if fraction < 0.35:
        return final * 0.5
    return final


@dataclass(frozen=True)
class ContextForcingPair:
    full: FramePackPlan
    sparse: FramePackPlan
    kept_context: tuple[int, ...]


def sample_sparse_context_pair(
    num_context: int,
    target_frames: int,
    rng: np.random.Generator,
    *,
    generated_context: Iterable[int] = (),
    old_stride: int = 2,
    drop_probability: float = 0.25,
) -> ContextForcingPair:
    """Two legal memory sets for the same target/noise (Wonder+Lyra forcing).

    The immutable first observation and newest observation can never disappear.  Other
    recent frames are dropped less often than old history.  The sparse arm omits tokens
    structurally; zeroing pixels would still let attention infer that a slot existed.
    """
    if num_context < 2 or not 0 <= drop_probability <= 1:
        raise ValueError("sparse forcing needs >=2 contexts and a probability in [0,1]")
    full = make_teacher_framepack(
        num_context, target_frames, generated_context=generated_context,
        old_stride=old_stride,
    )
    keep = {0, num_context - 1}
    recent_start = max(1, num_context - 3)
    for index in range(1, num_context - 1):
        probability = drop_probability * (0.5 if index >= recent_start else 1.0)
        if rng.random() >= probability:
            keep.add(index)
    sparse = make_teacher_framepack(
        num_context, target_frames, generated_context=generated_context,
        included_context=sorted(keep), old_stride=old_stride,
    )
    return ContextForcingPair(full, sparse, tuple(sorted(keep)))


@dataclass(frozen=True)
class SameTargetPaths:
    path_a: tuple[int, ...]
    path_b: tuple[int, ...]
    target: int


def sample_same_target_paths(scene, context_frames: int, target_frames: int,
                             rng: np.random.Generator, *, tries: int = 80) -> SameTargetPaths | None:
    """Choose two distinct causal observation subsets ending at the same target chunk."""
    if context_frames < 3 or target_frames < 1:
        raise ValueError("path consistency needs >=3 contexts and a positive target chunk")
    n = len(scene)
    for _ in range(tries):
        stride_a = int(rng.choice((1, 2)))
        stride_b = int(rng.choice((2, 3, 4)))
        span = max(stride_a, stride_b) * (context_frames - 1)
        if n <= span + target_frames:
            return None
        target = int(rng.integers(span, n - target_frames + 1))
        a = tuple(target - stride_a * (context_frames - index)
                  for index in range(context_frames))
        b = tuple(target - stride_b * (context_frames - index)
                  for index in range(context_frames))
        if a == b or min(a) < 0 or min(b) < 0:
            continue
        # Geometry-only continuity gate; no target appearance or model result is read.
        from gmlib.data.trajectory import TrajectorySampler
        sampler = TrajectorySampler(scene)
        if sampler._continuous(min(a + b), target + target_frames - 1):
            return SameTargetPaths(a, b, target)
    return None


@dataclass(frozen=True)
class NativeSameTargetPaths:
    """Two strided native-latent histories ending at one shared target chunk.

    Every entry is a *group start*: latent i covers RGB frames ``[start, start+4)`` and is
    encoded from the five-frame window ``[start-1, start+4)`` (see
    :func:`gmlib.models.vae_stream.encode_group`). The target is contiguous --
    ``target_frames`` latents covering ``4*target_frames`` frames from ``target`` -- and
    is encoded once, primed on ``[target-5, target)``, so both histories share it exactly.
    """
    a: tuple[int, ...]
    b: tuple[int, ...]
    target: int
    target_latents: int

    @property
    def target_frame_span(self) -> tuple[int, int]:
        return self.target, self.target + FRAMES_PER_LATENT * self.target_latents

    @property
    def prime_span(self) -> tuple[int, int]:
        return self.target - PRIME_FRAMES, self.target

    def group_window(self, start: int) -> tuple[int, int]:
        return start - 1, start + FRAMES_PER_LATENT

    @property
    def context_pose_frames(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return (tuple(latent_pose_frame(g) for g in self.a),
                tuple(latent_pose_frame(g) for g in self.b))

    @property
    def target_pose_frames(self) -> tuple[int, ...]:
        return tuple(latent_pose_frame(self.target + FRAMES_PER_LATENT * j)
                     for j in range(self.target_latents))


def latent_pose_frame(group_start: int) -> int:
    """The RGB frame whose pose represents a latent covering ``[start, start+4)``.

    Lyra 2.0 selects ``time_idx = [0] + list(range(4, v_len, 4))`` -- with latent k>=1
    spanning frames ``[4k-3, 4k]``, that is the LAST frame of each group, i.e. the pose
    the latent arrives at. This follows that rule rather than the four-poses-per-latent
    packing in ``camera.packed_rays``: Lyra conditions on ONE pose per latent
    (``cam_dim = 1536`` = 6*8*8 for its 8x VAE, exactly our 6*16*16 = 1536 at 16x), and
    matching it keeps ``RayContextProjection`` at its existing input width, so no
    trained ray branch is invalidated by moving to the native contract.
    """
    return group_start + FRAMES_PER_LATENT - 1


def native_target_pose_frames(chunk_start: int, target_latents: int) -> list[int]:
    """Pose frames for a native target chunk: one per latent, at each group's last frame."""
    if target_latents < 1:
        raise ValueError("a target chunk needs at least one latent")
    return [latent_pose_frame(chunk_start + FRAMES_PER_LATENT * j)
            for j in range(target_latents)]


def lyra_augment_sigma(shape, p_mean: float, p_std: float, multiplier: float,
                       *, device, generator=None) -> torch.Tensor:
    """Per-(batch, frame) corruption scale, exactly Lyra 2.0's GEN3C draw.

    ``draw_augment_sigma_and_epsilon_gen3c`` samples log-sigma from N(p_mean, p_std) and
    exponentiates, so sigma spans orders of magnitude: at Lyra's (-3.0, 2.0) the median is
    e^-3 = 0.050 and the 16-84% range is 0.007 to 0.368. The model therefore sees history
    ranging from almost clean to badly degraded, and is never told which: Lyra's
    ``augment_conditional_latent_frames`` returns ``(augment_latent, augment_sigma)`` but
    its caller writes back only the latent and drops the sigma, so it never reaches the
    network and robustness has to be learned blind.
    """
    if p_std <= 0 or multiplier <= 0:
        raise ValueError("augment sigma needs positive spread and multiplier")
    batch, frames = shape
    log_sigma = torch.randn(batch, frames, device=device, generator=generator) * p_std + p_mean
    return torch.exp(log_sigma) * multiplier


def apply_history_noise(context: torch.Tensor, *, p_mean: float, p_std: float,
                        multiplier: float = 1.0, fixed: float | None = None,
                        generator=None) -> torch.Tensor:
    """Additive conditioning-latent corruption: ``z + sigma * eps``, Lyra's exact form.

    Applied to HISTORY ONLY. The target slice is the flow-matching label and corrupting it
    would silently change what the model is regressing to -- a different bug entirely.

    ``fixed`` selects the inference path: Lyra trains with a sampled sigma and evaluates
    with a small constant (0.001), so train and test see the same *kind* of input.
    """
    if context.ndim != 5:
        raise ValueError(f"context must be (B,C,F,H,W), got {tuple(context.shape)}")
    batch, _, frames = context.shape[:3]
    if fixed is not None:
        sigma = torch.full((batch, frames), float(fixed), device=context.device)
    else:
        sigma = lyra_augment_sigma((batch, frames), p_mean, p_std, multiplier,
                                   device=context.device, generator=generator)
    noise = torch.randn(context.shape, device=context.device, dtype=context.dtype,
                        generator=generator)
    return context + noise * sigma.view(batch, 1, frames, 1, 1).to(context.dtype)


def sample_native_same_target_paths(
    scene, context_latents: int, target_latents: int, rng: np.random.Generator,
    *, tries: int = 80,
) -> NativeSameTargetPaths | None:
    """Native-contract analogue of :func:`sample_same_target_paths`.

    Strides are counted in *groups*, not frames, so a stride of 1 abuts consecutive
    four-frame groups and larger strides skip whole groups. Groups never overlap, which
    keeps each latent's five-frame encode window disjoint from its neighbours' and makes
    the history a genuine subsample of the video rather than a smear of it.
    """
    if context_latents < 3 or target_latents < 1:
        raise ValueError("path consistency needs >=3 context latents and a positive chunk")
    n = len(scene)
    span_frames = FRAMES_PER_LATENT * target_latents
    for _ in range(tries):
        stride_a = int(rng.choice((1, 2)))
        stride_b = int(rng.choice((2, 3, 4)))
        # earliest frame any encode window touches: first group start, minus its one-frame
        # predecessor. The target prime never reaches further back than that.
        reach = FRAMES_PER_LATENT * max(stride_a, stride_b) * context_latents + 1
        if n <= reach + span_frames:
            return None
        target = int(rng.integers(reach, n - span_frames + 1))
        def groups(stride: int) -> tuple[int, ...]:
            return tuple(target - FRAMES_PER_LATENT * stride * (context_latents - index)
                         for index in range(context_latents))
        a, b = groups(stride_a), groups(stride_b)
        if a == b or min(a) - 1 < 0:
            continue
        if target - PRIME_FRAMES < 0:
            continue
        from gmlib.data.trajectory import TrajectorySampler
        sampler = TrajectorySampler(scene)
        if sampler._continuous(min(a + b) - 1, target + span_frames - 1):
            return NativeSameTargetPaths(a, b, target, target_latents)
    return None


def supported_consistency_loss(prediction: torch.Tensor, teacher: torch.Tensor,
                               support: torch.Tensor | None, *, min_support: float = 0.02) -> torch.Tensor:
    """Path/memory consistency only where geometric evidence says the world is shared."""
    if prediction.shape != teacher.shape:
        raise ValueError("consistency predictions must have identical shapes")
    error = (prediction.float() - teacher.detach().float()) ** 2
    if support is None:
        return error.mean()
    if support.ndim != 5 or support.shape[0] != error.shape[0] \
            or support.shape[2:] != error.shape[2:]:
        raise ValueError("support must be Bx1xTxHxW on the prediction grid")
    mask = support.float().clamp(0, 1)
    if float(mask.mean()) < min_support:
        # Geometry abstains. Do not fabricate a consistency label in unseen regions.
        return error.sum() * 0.0
    return (error * mask).sum() / (mask.sum() * error.shape[1]).clamp_min(1.0)


def cosine_floor_schedule(step: int, total_steps: int, *, warmup: int = 100,
                          floor: float = 0.4) -> float:
    """Lyra-matched warmup and nonzero LR floor."""
    if total_steps < 1 or warmup < 0 or not 0 <= floor <= 1:
        raise ValueError("invalid LR schedule")
    if step < warmup:
        return (step + 1) / max(1, warmup)
    fraction = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    return floor + (1 - floor) * 0.5 * (1 + np.cos(np.pi * fraction))
