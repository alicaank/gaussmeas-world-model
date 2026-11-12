"""Stationary objective and optimizer engine for the single GATE-ΩR run.

Data/rendering code implements :class:`OmegaRowExecutor`; this module owns the parts
that must never vary with a dataset or optimizer step: role counts, long-depth draws,
forward accounting, objective weights, optimizer authority, LR policy, and clipping.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import gc
from typing import Protocol

import torch

from gmlib.training.gate_omega import (
    BINDING_WEIGHT,
    CONTRACT_ID,
    sample_history_depth,
    stationary_row_roles,
    symmetric_target_view_binding,
)
from gmlib.training.gate_omega_checkpoint import OmegaModules


OPTIMIZER_STEPS = 12_000
WARMUP_STEPS = 100
LR_CYCLE_STEPS = 400_000
LR_START_FRACTION = 1.0e-6
LR_MAX_FRACTION = 0.99
MIN_LR_FRACTION = 0.40
FAST_LR = 3e-5
SLOW_LR = 1e-5
BACKBONE_LR = 2e-5
WEIGHT_DECAY = 0.001
GRAD_CLIP = 0.1
POSITIVE_FORWARDS_PER_STEP = 9
CROSS_FORWARDS_PER_STEP = 2
SELF_AUG_ROLLOUT_FORWARDS = 30
TOTAL_FORWARDS_PER_STEP = 41
CPU_MASTER_OPTIMIZER = "cpu_master_adam8bit"
CPU_MASTER_STATE_SCHEMA = "gmlib.cpu_master_adam8bit.v1"


def trim_cpu_allocator() -> None:
    """Return released CPU staging pages to the OS before the next large teacher."""
    gc.collect()
    try:
        malloc_trim = ctypes.CDLL(None).malloc_trim
    except (AttributeError, OSError) as error:
        raise RuntimeError(
            "production CPU-master training requires glibc malloc_trim") from error
    malloc_trim.argtypes = [ctypes.c_size_t]
    malloc_trim.restype = ctypes.c_int
    malloc_trim(0)


@dataclass
class PositiveRowResult:
    flow_loss: torch.Tensor
    forward_count: int = 1
    history_depth: int = 0
    diagnostics: dict | None = None


@dataclass
class BindingRowResult:
    flow_a: torch.Tensor
    flow_b: torch.Tensor
    e_aa: torch.Tensor
    e_ab: torch.Tensor
    e_bb: torch.Tensor
    e_ba: torch.Tensor
    positive_forwards: int = 2
    cross_forwards: int = 2
    diagnostics: dict | None = None


@dataclass
class SelfAugRowResult(PositiveRowResult):
    rollout_forwards: int = SELF_AUG_ROLLOUT_FORWARDS
    abstained: bool = False
    attempts: int = 1


class OmegaRowExecutor(Protocol):
    """Heavy data/model boundary consumed by :func:`run_stationary_step`."""

    def positive(self, role: str, *, step: int, row: int,
                 history_depth: int) -> PositiveRowResult: ...

    def binding(self, *, step: int, row: int) -> BindingRowResult: ...

    def self_augment(self, *, step: int, row: int) -> SelfAugRowResult: ...


def _finite_scalar(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise RuntimeError(f"{name} must be a scalar tensor")
    if not bool(torch.isfinite(value.detach()).all()):
        raise FloatingPointError(f"{name} is non-finite")


def _backward(loss: torch.Tensor, name: str) -> None:
    _finite_scalar(loss, name)
    loss.backward()


def run_stationary_step(executor: OmegaRowExecutor, *, step: int,
                        expected_self_aug_forwards: int =
                        SELF_AUG_ROLLOUT_FORWARDS) -> dict:
    """Execute one contracted 8-row step with an explicit self-aug budget."""
    expected_rollout = int(expected_self_aug_forwards)
    if expected_rollout < 1:
        raise ValueError("expected self-aug forwards must be positive")
    roles = stationary_row_roles(step)
    flow_sum = 0.0
    binding_value = 0.0
    positive_forwards = cross_forwards = rollout_forwards = 0
    row_records = []
    # The 30-forward live rollout owns DA3/VAE scratch memory. Execute that
    # deterministic row before DiT gradient tensors exist; row seeds remain bound to
    # their original indices, so this changes only the memory schedule, not the loss.
    indexed_roles = list(enumerate(roles))
    execution = sorted(indexed_roles, key=lambda item: item[1] != "selfaug")
    for row, role in execution:
        if role == "binding":
            result = executor.binding(step=step, row=row)
            if result.positive_forwards != 2 or result.cross_forwards != 2:
                raise RuntimeError("binding row must execute two positive and two cross forwards")
            for name, value in (("flow_a", result.flow_a), ("flow_b", result.flow_b),
                                ("e_aa", result.e_aa), ("e_ab", result.e_ab),
                                ("e_bb", result.e_bb), ("e_ba", result.e_ba)):
                _finite_scalar(value, f"binding.{name}")
            bind, bind_diagnostics = symmetric_target_view_binding(
                result.e_aa, result.e_ab, result.e_bb, result.e_ba)
            loss = ((result.flow_a + result.flow_b) / POSITIVE_FORWARDS_PER_STEP
                    + BINDING_WEIGHT * bind)
            _backward(loss, "binding objective")
            positive_forwards += 2
            cross_forwards += 2
            flow_sum += float((result.flow_a + result.flow_b).detach())
            binding_value = float(bind.detach())
            row_records.append({"row": row, "role": role,
                                "binding": bind_diagnostics,
                                "producer": result.diagnostics or {}})
            continue

        if role == "selfaug":
            result = executor.self_augment(step=step, row=row)
            if result.rollout_forwards != expected_rollout:
                raise RuntimeError(
                    f"self-augmentation must execute exactly {expected_rollout} "
                    "no-grad rollout forwards")
            if result.attempts < 1:
                raise RuntimeError("self-augmentation attempt count must be positive")
            rollout_forwards += result.rollout_forwards
        else:
            depth = (sample_history_depth(
                f"{CONTRACT_ID}:step={step}:row={row}:role={role}")
                     if role == "deep" else 0)
            result = executor.positive(
                role, step=step, row=row, history_depth=depth)
            if result.history_depth != depth:
                raise RuntimeError(
                    f"producer history depth {result.history_depth} != contracted {depth}")
        if result.forward_count != 1:
            raise RuntimeError(f"{role} row must execute one loss-bearing forward")
        _backward(result.flow_loss / POSITIVE_FORWARDS_PER_STEP, f"{role} flow")
        positive_forwards += 1
        flow_sum += float(result.flow_loss.detach())
        row_records.append({
            "row": row, "role": role, "history_depth": result.history_depth,
            "selfaug_abstained": bool(getattr(result, "abstained", False)),
            "producer": result.diagnostics or {},
        })

    total_forwards = positive_forwards + cross_forwards + rollout_forwards
    expected_total = (POSITIVE_FORWARDS_PER_STEP + CROSS_FORWARDS_PER_STEP
                      + expected_rollout)
    if (positive_forwards, cross_forwards, rollout_forwards, total_forwards) != (
            POSITIVE_FORWARDS_PER_STEP, CROSS_FORWARDS_PER_STEP,
            expected_rollout, expected_total):
        raise RuntimeError(
            "stationary forward ledger violated: "
            f"positive={positive_forwards}, cross={cross_forwards}, "
            f"rollout={rollout_forwards}, total={total_forwards}")
    return {
        "step": int(step), "roles": list(roles),
        "execution_order": [row for row, _ in execution],
        "flow_loss": flow_sum / POSITIVE_FORWARDS_PER_STEP,
        "binding_loss": binding_value,
        "objective": flow_sum / POSITIVE_FORWARDS_PER_STEP
                     + BINDING_WEIGHT * binding_value,
        "positive_forwards": positive_forwards,
        "cross_forwards": cross_forwards,
        "rollout_forwards": rollout_forwards,
        "total_forwards": total_forwards,
        "rows": sorted(row_records, key=lambda item: item["row"]),
    }


def lr_fraction(completed_steps: int) -> float:
    """Exact released-Lyra 100-step warmup and 400k-cycle linear schedule."""
    if not 0 <= completed_steps <= OPTIMIZER_STEPS:
        raise ValueError(f"completed_steps must lie in [0,{OPTIMIZER_STEPS}]")
    if completed_steps < WARMUP_STEPS:
        return (LR_START_FRACTION
                + (LR_MAX_FRACTION - LR_START_FRACTION)
                * completed_steps / WARMUP_STEPS)
    return (MIN_LR_FRACTION
            + (LR_MAX_FRACTION - MIN_LR_FRACTION)
            * (LR_CYCLE_STEPS - completed_steps)
            / (LR_CYCLE_STEPS - WARMUP_STEPS))


def _is_slow_parameter(name: str) -> bool:
    return (name.startswith("omega_reader.")
            or "gate" in name
            or ("confidence" in name and "embedding" in name)
            or name.endswith("in_scale")
            or name.endswith("in_bias"))


def _uses_weight_decay(name: str, parameter: torch.Tensor) -> bool:
    return (parameter.ndim >= 2
            and "embedding" not in name
            and not name.endswith(".bias")
            and "gate" not in name
            and not name.endswith("in_scale"))


def optimizer_parameter_groups(modules: OmegaModules) -> tuple[list[dict], dict]:
    """Build stable Lyra-selective groups with a discriminative backbone LR."""
    modules.assert_parameter_authority()
    buckets = {
        "backbone_decay": [], "backbone_nodecay": [],
        "fast_decay": [], "fast_nodecay": [],
        "slow_decay": [], "slow_nodecay": [],
    }
    names = {key: [] for key in buckets}
    seen = set()
    for name, parameter in modules.trainable_named_parameters():
        if id(parameter) in seen:
            raise RuntimeError(f"optimizer parameter is aliased: {name}")
        seen.add(id(parameter))
        if name.startswith(("dit.", "framepack_patch.")):
            if parameter.dtype not in {torch.bfloat16, torch.float32}:
                raise RuntimeError(
                    f"trainable backbone parameter {name} has unsupported dtype "
                    f"{parameter.dtype}")
            speed = "backbone"
        else:
            if parameter.dtype != torch.float32:
                raise RuntimeError(f"trainable route parameter {name} is not fp32")
            speed = "slow" if _is_slow_parameter(name) else "fast"
        decay = "decay" if _uses_weight_decay(name, parameter) else "nodecay"
        key = f"{speed}_{decay}"
        buckets[key].append(parameter)
        names[key].append(name)
    groups = []
    for key, parameters in buckets.items():
        if not parameters:
            continue
        speed, decay = key.split("_", 1)
        groups.append({
            "params": parameters,
            "lr": ({"fast": FAST_LR, "slow": SLOW_LR,
                    "backbone": BACKBONE_LR}[speed]),
            "weight_decay": WEIGHT_DECAY if decay == "decay" else 0.0,
            "group_name": key,
        })
    expected = sum(1 for _ in modules.trainable_named_parameters())
    if sum(len(group["params"]) for group in groups) != expected:
        raise AssertionError("optimizer groups do not cover the trainable route surface")
    return groups, names


class CpuMasterAdamW8bit:
    """FP32 CPU masters with 8-bit Adam state and bf16 GPU compute shadows.

    Direct low-LR updates to bf16 model weights round to zero.  Keeping the optimizer
    authority in fp32 on CPU makes every Wan parameter genuinely trainable while the
    one-GPU forward/backward still uses its memory-efficient bf16 shadow.
    """

    def __init__(self, groups: list[dict], names: dict[str, list[str]]):
        try:
            import bitsandbytes as bnb
        except ImportError as error:
            raise RuntimeError("CPU-master AdamW8bit requires bitsandbytes") from error
        master_groups = []
        self._pairs: list[tuple[str, torch.nn.Parameter, torch.nn.Parameter]] = []
        for group in groups:
            group_name = str(group["group_name"])
            gpu_parameters = list(group["params"])
            group_names = list(names[group_name])
            if len(gpu_parameters) != len(group_names):
                raise RuntimeError(f"optimizer names drifted for {group_name}")
            masters = []
            for name, gpu_parameter in zip(group_names, gpu_parameters):
                master = torch.nn.Parameter(
                    gpu_parameter.detach().to(device="cpu", dtype=torch.float32),
                    requires_grad=True)
                # PyTorch 2.12 permits a lower-precision gradient buffer for an fp32
                # parameter.  This halves host RAM without changing Adam's fp32 update.
                master.grad_dtype = gpu_parameter.dtype
                masters.append(master)
                self._pairs.append((name, gpu_parameter, master))
            master_groups.append({
                key: value for key, value in group.items() if key != "params"
            } | {"params": masters})
        self._optimizer = bnb.optim.AdamW8bit(
            master_groups, betas=(0.9, 0.999), eps=1e-8)
        self.param_groups = self._optimizer.param_groups

    @property
    def state(self):
        return self._optimizer.state

    @property
    def master_parameter_bytes(self) -> int:
        return sum(master.numel() * master.element_size()
                   for _, _, master in self._pairs)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for _, gpu_parameter, _ in self._pairs:
            if set_to_none:
                gpu_parameter.grad = None
            elif gpu_parameter.grad is not None:
                gpu_parameter.grad.zero_()
        self._optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self) -> None:
        for _, gpu_parameter, master in self._pairs:
            master.grad = (None if gpu_parameter.grad is None else
                           gpu_parameter.grad.detach().to(
                               device="cpu", dtype=gpu_parameter.dtype))
        self._optimizer.step()
        for _, gpu_parameter, master in self._pairs:
            gpu_parameter.copy_(master, non_blocking=False)
            master.grad = None
        # The simultaneous bf16 CPU gradients are roughly 10 GiB for the full
        # teacher. They are dead after the optimizer update, but glibc otherwise
        # retains their arenas and the following DA3 load can trigger global OOM.
        trim_cpu_allocator()

    def allocate_state(self) -> None:
        """Materialize every CPU 8-bit moment without applying a parameter update."""
        optimizer = self._optimizer
        if not optimizer.initialized:
            optimizer.check_overrides()
            optimizer.to_gpu()
            optimizer.initialized = True
        for group_index, group in enumerate(optimizer.param_groups):
            for parameter_index, parameter in enumerate(group["params"]):
                if len(optimizer.state[parameter]) == 0:
                    optimizer.init_state(
                        group, parameter, group_index, parameter_index)
        if any(len(optimizer.state[p]) == 0 for group in optimizer.param_groups
               for p in group["params"]):
            raise RuntimeError("CPU-master optimizer state allocation is incomplete")

    def state_dict(self) -> dict:
        return {
            "schema": CPU_MASTER_STATE_SCHEMA,
            "parameter_names": [name for name, _, _ in self._pairs],
            "master_parameters": [master.detach() for _, _, master in self._pairs],
            "optimizer": self._optimizer.state_dict(),
        }

    @torch.no_grad()
    def load_state_dict(self, state: dict) -> None:
        if set(state) != {
                "schema", "parameter_names", "master_parameters", "optimizer"} \
                or state.get("schema") != CPU_MASTER_STATE_SCHEMA:
            raise RuntimeError("CPU-master optimizer resume schema is incompatible")
        names = [name for name, _, _ in self._pairs]
        values = state.get("master_parameters")
        if state.get("parameter_names") != names or not isinstance(values, list) \
                or len(values) != len(self._pairs):
            raise RuntimeError("CPU-master optimizer parameter topology changed")
        for value, (name, gpu_parameter, master) in zip(values, self._pairs):
            if (not isinstance(value, torch.Tensor)
                    or value.dtype != torch.float32
                    or value.shape != master.shape
                    or not bool(torch.isfinite(value).all())):
                raise RuntimeError(f"invalid CPU master tensor for {name}")
            master.copy_(value)
            gpu_parameter.copy_(master)
        self._optimizer.load_state_dict(state["optimizer"])


def make_optimizer(modules: OmegaModules, *, backend: str = CPU_MASTER_OPTIMIZER):
    groups, names = optimizer_parameter_groups(modules)
    if backend == "adamw":
        optimizer = torch.optim.AdamW(
            groups, betas=(0.9, 0.999), eps=1e-8, foreach=False)
    elif backend == "paged_adam8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as error:
            raise RuntimeError("PagedAdamW8bit requires bitsandbytes") from error
        optimizer = bnb.optim.PagedAdamW8bit(
            groups, betas=(0.9, 0.999), eps=1e-8)
    elif backend == CPU_MASTER_OPTIMIZER:
        optimizer = CpuMasterAdamW8bit(groups, names)
    else:
        raise ValueError(f"unsupported GATE-Ω optimizer backend {backend!r}")
    return optimizer, names


def apply_lr(optimizer, completed_steps: int) -> float:
    fraction = lr_fraction(completed_steps)
    for group in optimizer.param_groups:
        speed = group["group_name"].split("_", 1)[0]
        base = {"fast": FAST_LR, "slow": SLOW_LR,
                "backbone": BACKBONE_LR}[speed]
        group["lr"] = base * fraction
    return fraction


def finite_global_grad_norm(modules: OmegaModules, *, clip: float = GRAD_CLIP) -> float:
    parameters = [parameter for _, parameter in modules.trainable_named_parameters()]
    nonfinite = [name for name, parameter in modules.trainable_named_parameters()
                 if parameter.grad is not None
                 and not bool(torch.isfinite(parameter.grad).all())]
    if nonfinite:
        raise FloatingPointError(f"non-finite GATE-Ω gradients: {nonfinite[:8]}")
    norm = torch.nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
    return float(norm.detach())
