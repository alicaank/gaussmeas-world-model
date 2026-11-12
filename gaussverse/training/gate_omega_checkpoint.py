"""Strict construction and checkpointing for the Lyra-selective GATE-Ω teacher.

The checkpoint is intentionally independent of the historical final-teacher format.
Its tensor namespaces are the route manifest: there is no hand-maintained list that can
silently drift away from what was actually saved.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from gaussverse.models.framepack import FramePackConditioner
from gaussverse.models.gate_omega_conditioning import (
    OmegaCommandQKStack,
    OmegaOrderedRayStack,
    OmegaScaffoldStack,
)
from gaussverse.models.gate_omega_reader import OmegaSourceReader
from gaussverse.training.gate_omega import (
    CONDITION_BLOCKS,
    CONTRACT_ID,
    READER_BLOCKS,
)
from gaussverse.training.static_teacher_recipe import (
    is_selective_backbone_parameter,
)


CHECKPOINT_SCHEMA = "gaussverse.gate_omega_checkpoint.v3"
ROUTE_MANIFEST_SCHEMA = "gaussverse.gate_omega_route_manifest.v3"
ROUTE_PREFIXES = {
    "dit": "trainable_backbone",
    "framepack_patch": "trainable_input",
    "frozen": "frozen_input",
    "omega_framepack": "trainable_route",
    "omega_ray": "trainable_route",
    "omega_command": "trainable_route",
    "omega_scaffold": "trainable_route",
    "omega_reader": "trainable_route",
}
TRAINABLE_PREFIXES = frozenset(
    prefix for prefix, role in ROUTE_PREFIXES.items()
    if role.startswith("trainable_"))


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("utf-8")


def file_sha256(path: Path, *, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wan_source_fingerprint(root: Path) -> dict:
    """Hash every diffusion shard and the local index/config files, if present."""
    root = Path(root).resolve()
    shards = sorted(root.glob("diffusion_pytorch_model-*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no Wan diffusion shards under {root}")
    optional = [
        root / "diffusion_pytorch_model.safetensors.index.json",
        root / "config.json",
    ]
    paths = shards + [path for path in optional if path.is_file()]
    files = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)}
        for path in paths
    ]
    aggregate = hashlib.sha256(_canonical_json(files)).hexdigest()
    return {"schema": "gaussverse.wan_source_fingerprint.v1",
            "root": str(root), "files": files, "sha256": aggregate}


@dataclass
class OmegaModules:
    dit: torch.nn.Module
    framepack: FramePackConditioner
    omega_ray: OmegaOrderedRayStack
    omega_command: OmegaCommandQKStack
    omega_scaffold: OmegaScaffoldStack
    omega_reader: OmegaSourceReader

    def named_routes(self) -> tuple[tuple[str, torch.nn.Module], ...]:
        return (
            ("omega_ray", self.omega_ray),
            ("omega_command", self.omega_command),
            ("omega_scaffold", self.omega_scaffold),
            ("omega_reader", self.omega_reader),
        )

    def trainable_named_parameters(self):
        for name, parameter in self.dit.named_parameters():
            if parameter.requires_grad:
                yield f"dit.{name}", parameter
        for name, parameter in self.framepack.named_parameters():
            if parameter.requires_grad:
                prefix = ("framepack_patch" if
                          name.startswith("history_patch_embedding.")
                          else "omega_framepack")
                yield f"{prefix}.{name}", parameter
        for prefix, module in self.named_routes():
            for name, parameter in module.named_parameters():
                if parameter.requires_grad:
                    yield f"{prefix}.{name}", parameter

    def assert_parameter_authority(self) -> None:
        missing = [f"dit.{name}" for name, p in self.dit.named_parameters()
                   if is_selective_backbone_parameter(name)
                   and not p.requires_grad]
        excess = [f"dit.{name}" for name, p in self.dit.named_parameters()
                  if not is_selective_backbone_parameter(name)
                  and p.requires_grad]
        missing += [f"omega_framepack.{name}" for name, p in
                    self.framepack.named_parameters() if not p.requires_grad]
        missing += [f"{prefix}.{name}" for prefix, module in self.named_routes()
                    for name, p in module.named_parameters() if not p.requires_grad]
        non_fp32 = [name for name, p in self.trainable_named_parameters()
                    if p.dtype != torch.float32]
        if missing or excess or non_fp32:
            raise RuntimeError(
                "GATE-Ω Lyra-selective authority violation: "
                f"unexpectedly_frozen={missing[:8]}, "
                f"unexpectedly_trainable={excess[:8]}, "
                f"non_fp32_trainables={non_fp32[:8]}")


def apply_parameter_authority(modules: OmegaModules) -> None:
    """Apply Lyra's exact selective Wan tier plus all new Ω routes.

    Frozen Wan weights remain in their bf16 compute representation.  The selected
    self-attention and stock patch-embedding weights are fp32 masters updated in place,
    matching the resource-proven Lyra teacher recipe without a second CPU copy.
    """
    modules.dit.requires_grad_(False)
    selected = []
    for name, parameter in modules.dit.named_parameters():
        if is_selective_backbone_parameter(name):
            parameter.data = parameter.data.to(dtype=torch.float32)
            parameter.requires_grad_(True)
            selected.append(name)
    if not selected:
        raise RuntimeError("Lyra selective authority matched no Wan parameters")
    for parameter in modules.framepack.parameters():
        parameter.data = parameter.data.to(dtype=torch.float32)
        parameter.requires_grad_(True)
    for _, module in modules.named_routes():
        for parameter in module.parameters():
            parameter.data = parameter.data.to(dtype=torch.float32)
            parameter.requires_grad_(True)
    modules.assert_parameter_authority()


def build_omega_modules(dit: torch.nn.Module, *, dim: int | None = None,
                        ray_hidden: int = 64, command_hidden: int = 256,
                        scaffold_hidden: int = 128,
                        condition_blocks: tuple[int, ...] | None = None,
                        reader_blocks: tuple[int, ...] = READER_BLOCKS,
                        blocks: tuple[int, ...] | None = None,
                        main_keys: int = 1120, source_keys: int = 560) -> OmegaModules:
    """Attach zero-output Ω routes with Lyra-selective Wan authority."""
    # ``blocks`` keeps small focused fixtures concise. Production never uses it:
    # camera/buffer conditions cover all Wan blocks while the expensive archive
    # joint read remains at its separately budgeted three depths.
    if blocks is not None:
        if condition_blocks is not None or reader_blocks != READER_BLOCKS:
            raise ValueError("legacy blocks cannot be mixed with split block layouts")
        condition_blocks = reader_blocks = tuple(blocks)
    if condition_blocks is None:
        condition_blocks = CONDITION_BLOCKS
    dim = int(dim if dim is not None else dit.dim)
    invalid = sorted((set(condition_blocks) | set(reader_blocks))
                     - set(range(len(dit.blocks))))
    if invalid:
        raise ValueError(
            f"Ω blocks {invalid} are outside {len(dit.blocks)}-block parent")
    patch = dit.patch_embedding
    device = next(dit.parameters()).device
    patch_size = tuple(int(value) for value in patch.kernel_size)
    framepack = FramePackConditioner(
        dim=dim, in_dim=int(patch.in_channels), patch_size=patch_size,
        # ΩR has no separate FramePack geometry-bias route. Role and authority
        # residuals are the trainable zero-output surface.
        geometry_blocks=()).to(device=device, dtype=torch.float32)
    framepack.initialize_history_patch(patch)
    for parameter in framepack.history_patch_embedding.parameters():
        parameter.requires_grad_(True)

    modules = OmegaModules(
        dit=dit,
        framepack=framepack,
        omega_ray=OmegaOrderedRayStack(
            hidden=ray_hidden, dim=dim, blocks=condition_blocks).to(device=device),
        omega_command=OmegaCommandQKStack(
            hidden=command_hidden, dim=dim,
            blocks=condition_blocks).to(device=device),
        omega_scaffold=OmegaScaffoldStack(
            hidden=scaffold_hidden, dim=dim,
            blocks=condition_blocks).to(device=device),
        omega_reader=OmegaSourceReader(
            blocks=reader_blocks, dim=dim).to(device=device),
    )
    modules.omega_reader.initialize_beta(main_keys, source_keys)
    apply_parameter_authority(modules)
    return modules


def checkpoint_state(modules: OmegaModules,
                     frozen_text_context: torch.Tensor) -> dict[str, torch.Tensor]:
    if frozen_text_context.ndim != 3 or not frozen_text_context.is_floating_point():
        raise ValueError("frozen text context must be a floating BxLxD tensor")
    if not bool(torch.isfinite(frozen_text_context).all()):
        raise ValueError("frozen text context contains non-finite values")
    modules.assert_parameter_authority()
    state: dict[str, torch.Tensor] = {}
    sources = (("dit", modules.dit), *modules.named_routes())
    for prefix, module in sources:
        for name, tensor in module.state_dict().items():
            state[f"{prefix}.{name}"] = tensor.detach().cpu().contiguous()
    for name, tensor in modules.framepack.state_dict().items():
        prefix = ("framepack_patch" if name.startswith("history_patch_embedding.")
                  else "omega_framepack")
        state[f"{prefix}.{name}"] = tensor.detach().cpu().contiguous()
    state["frozen.text_context"] = frozen_text_context.detach().cpu().contiguous()
    return state


def _tensor_record(tensor: torch.Tensor) -> dict:
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "bytes": int(tensor.numel() * tensor.element_size())}


def route_layout_from_state(state: Mapping[str, torch.Tensor]) -> dict:
    unknown = sorted({key.split(".", 1)[0] for key in state} - set(ROUTE_PREFIXES))
    if unknown:
        raise RuntimeError(f"unknown checkpoint route prefixes: {unknown}")
    missing = sorted(set(ROUTE_PREFIXES) - {key.split(".", 1)[0] for key in state})
    if missing:
        raise RuntimeError(f"missing checkpoint route prefixes: {missing}")
    routes = {}
    for prefix, role in ROUTE_PREFIXES.items():
        tensors = {key: _tensor_record(value) for key, value in sorted(state.items())
                   if key.startswith(prefix + ".")}
        routes[prefix] = {
            "role": role,
            "tensor_count": len(tensors),
            "bytes": sum(item["bytes"] for item in tensors.values()),
            "tensors": tensors,
        }
    return {"schema": ROUTE_MANIFEST_SCHEMA, "routes": routes,
            "tensor_count": len(state),
            "bytes": sum(item["bytes"] for route in routes.values()
                         for item in route["tensors"].values())}


def route_layout_sha256(layout: dict) -> str:
    return hashlib.sha256(_canonical_json(layout)).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp",
                                     dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_omega_checkpoint(path: Path, modules: OmegaModules,
                          frozen_text_context: torch.Tensor, *,
                          parent_fingerprint: dict,
                          route_manifest_path: Path | None = None,
                          extra_metadata: Mapping[str, str] | None = None) -> dict:
    """Atomically save a full checkpoint and its content-derived route manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = checkpoint_state(modules, frozen_text_context)
    layout = route_layout_from_state(state)
    layout_sha = route_layout_sha256(layout)
    metadata = {
        "schema": CHECKPOINT_SCHEMA,
        "contract_id": CONTRACT_ID,
        "kind": "full",
        "parent_fingerprint_sha256": str(parent_fingerprint["sha256"]),
        "route_layout_sha256": layout_sha,
        "trainable_prefixes": ",".join(sorted(TRAINABLE_PREFIXES)),
    }
    if extra_metadata:
        overlap = set(metadata) & set(extra_metadata)
        mismatched = sorted(key for key in overlap
                            if str(extra_metadata[key]) != metadata[key])
        if mismatched:
            raise ValueError(
                f"conflicting reserved checkpoint metadata: {mismatched}")
        metadata.update({str(key): str(value) for key, value in extra_metadata.items()
                         if key not in overlap})
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp",
                                     dir=path.parent)
    os.close(fd)
    try:
        save_file(state, temporary, metadata=metadata)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    checkpoint_sha = file_sha256(path)
    manifest = {
        **layout,
        "contract_id": CONTRACT_ID,
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "route_layout_sha256": layout_sha,
        "parent_fingerprint": parent_fingerprint,
        "trainable_prefixes": sorted(TRAINABLE_PREFIXES),
    }
    manifest["manifest_payload_sha256"] = hashlib.sha256(
        _canonical_json(manifest)).hexdigest()
    route_manifest_path = (Path(route_manifest_path) if route_manifest_path is not None
                           else path.with_name("route_manifest.json"))
    _atomic_json(route_manifest_path, manifest)
    return manifest


def _expected_state_keys(modules: OmegaModules) -> set[str]:
    result = {"frozen.text_context"}
    for prefix, module in (("dit", modules.dit), *modules.named_routes()):
        result.update(f"{prefix}.{name}" for name in module.state_dict())
    for name in modules.framepack.state_dict():
        prefix = ("framepack_patch" if name.startswith("history_patch_embedding.")
                  else "omega_framepack")
        result.add(f"{prefix}.{name}")
    return result


def load_omega_checkpoint(path: Path, modules: OmegaModules, *,
                          expected_parent_sha256: str,
                          route_manifest_path: Path | None = None) -> tuple[torch.Tensor, dict]:
    """Strictly load a checkpoint; incompatibility is never downgraded to a warning."""
    path = Path(path)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        actual_keys = set(handle.keys())
    required_metadata = {
        "schema": CHECKPOINT_SCHEMA, "contract_id": CONTRACT_ID, "kind": "full",
        "parent_fingerprint_sha256": expected_parent_sha256,
        "trainable_prefixes": ",".join(sorted(TRAINABLE_PREFIXES)),
    }
    mismatch = {key: (metadata.get(key), value) for key, value in required_metadata.items()
                if metadata.get(key) != value}
    if mismatch:
        raise RuntimeError(f"GATE-Ω checkpoint metadata mismatch: {mismatch}")
    expected_keys = _expected_state_keys(modules)
    if actual_keys != expected_keys:
        raise RuntimeError(
            f"GATE-Ω tensor contract mismatch: missing={sorted(expected_keys-actual_keys)[:8]}, "
            f"unexpected={sorted(actual_keys-expected_keys)[:8]}")
    state = load_file(str(path), device="cpu")
    nonfinite = [key for key, value in state.items() if value.is_floating_point()
                 and not bool(torch.isfinite(value).all())]
    if nonfinite:
        raise RuntimeError(f"GATE-Ω checkpoint has non-finite tensors: {nonfinite[:8]}")
    layout = route_layout_from_state(state)
    layout_sha = route_layout_sha256(layout)
    if metadata.get("route_layout_sha256") != layout_sha:
        raise RuntimeError("GATE-Ω route-layout digest mismatch")

    if route_manifest_path is not None:
        manifest = json.loads(Path(route_manifest_path).read_text())
        recorded_payload_sha = manifest.pop("manifest_payload_sha256", None)
        computed_payload_sha = hashlib.sha256(_canonical_json(manifest)).hexdigest()
        manifest["manifest_payload_sha256"] = recorded_payload_sha
        if recorded_payload_sha != computed_payload_sha:
            raise RuntimeError("GATE-Ω route manifest payload digest mismatch")
        if manifest.get("route_layout_sha256") != layout_sha:
            raise RuntimeError("GATE-Ω route manifest describes another layout")
        if manifest.get("checkpoint_sha256") != file_sha256(path):
            raise RuntimeError("GATE-Ω route manifest describes another checkpoint")

    for prefix, module in (("dit", modules.dit), *modules.named_routes()):
        prefix_dot = prefix + "."
        module_state = {key[len(prefix_dot):]: value for key, value in state.items()
                        if key.startswith(prefix_dot)}
        module.load_state_dict(module_state, strict=True)
    framepack_state = {}
    for prefix in ("framepack_patch", "omega_framepack"):
        prefix_dot = prefix + "."
        framepack_state.update({
            key[len(prefix_dot):]: value for key, value in state.items()
            if key.startswith(prefix_dot)
        })
    modules.framepack.load_state_dict(framepack_state, strict=True)
    apply_parameter_authority(modules)
    return state["frozen.text_context"], metadata
