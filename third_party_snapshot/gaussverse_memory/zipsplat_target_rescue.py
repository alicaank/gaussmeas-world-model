"""Causal certified-view ZipSplat target renderer for GATE-Omega scaffolds."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping
import math

import torch

from gaussverse.rollout.zipsplat_stack import ZipSplatGeometry, central_square_mask
from gaussverse.training.gate_omega_condition_builder import (
    _archive_pose, _endpoint_rgb,
)


def rebase_c2w_to_first(poses: torch.Tensor
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ZipSplat-legal first-view-relative c2w and its world transform."""
    if (poses.ndim != 3 or poses.shape[-2:] != (4, 4) or len(poses) < 1
            or not bool(torch.isfinite(poses).all())):
        raise ValueError("ZipSplat source poses must be finite Nx4x4")
    reference_w2c = torch.linalg.inv(poses[0].double())
    relative = torch.einsum(
        "ij,njk->nik", reference_w2c, poses.double()).float()
    if not torch.allclose(
            relative[0], torch.eye(4, dtype=relative.dtype,
                                   device=relative.device),
            rtol=1e-5, atol=1e-5):
        raise RuntimeError("ZipSplat first-view rebase did not produce identity")
    return relative, reference_w2c.float()


def _rotation_deg(first: torch.Tensor, second: torch.Tensor) -> float:
    relative = first[:3, :3].double().T @ second[:3, :3].double()
    cosine = float(((torch.trace(relative) - 1) / 2).clamp(-1, 1))
    return math.degrees(math.acos(cosine))


def select_same_scene_wrong_span(scene, stream: list[dict],
                                 target_position: int
                                 ) -> tuple[tuple[int, ...], dict]:
    """Select one real causal two-latent camera negative from the same scene."""
    target_endpoint = int(stream[target_position]["pose_frame_id"])
    all_candidates = []
    for start in range(1, target_position - 1):
        endpoint = int(stream[start]["pose_frame_id"])
        translation = float(torch.linalg.vector_norm(
            scene.c2w[target_endpoint, :3, 3].double()
            - scene.c2w[endpoint, :3, 3].double())) / float(scene.s_scene)
        rotation = _rotation_deg(
            scene.c2w[target_endpoint], scene.c2w[endpoint])
        score = translation / 0.20 + rotation / 45.0
        all_candidates.append((score, start, translation, rotation))
    if not all_candidates:
        raise RuntimeError("scene has no causal same-scene wrong-camera span")
    in_band = [item for item in all_candidates
               if item[2] <= 0.20 and item[3] <= 45.0]
    if in_band:
        _, start, translation, rotation = max(in_band)
        within_requested_band = True
    else:
        _, start, translation, rotation = min(all_candidates)
        within_requested_band = False
    frame_ids = tuple(
        int(frame_id)
        for position in (start, start + 1)
        for frame_id in stream[position]["source_frame_ids"])
    if len(frame_ids) != 8 or len(set(frame_ids)) != 8:
        raise RuntimeError("same-scene wrong-camera span is not two RGB4 latents")
    return frame_ids, {
        "start_position": int(start),
        "endpoint_translation_scene_units": float(translation),
        "endpoint_rotation_deg": float(rotation),
        "within_requested_band": within_requested_band,
    }


class TargetPoseOverrideRenderer:
    """Render the same scheduled scene evidence at false target cameras."""

    def __init__(self, base, target_c2w: torch.Tensor):
        if (target_c2w.ndim != 3 or target_c2w.shape[-2:] != (4, 4)
                or not bool(torch.isfinite(target_c2w).all())):
            raise ValueError("wrong-camera override must be finite Tx4x4")
        self.base = base
        self.target_c2w = target_c2w.detach().float().cpu()

    def applies(self, memory, history_boundary: int) -> bool:
        applies = getattr(self.base, "applies", None)
        return True if applies is None else bool(applies(memory, history_boundary))

    def __call__(self, **kwargs):
        target = kwargs.get("target_c2w")
        if not isinstance(target, torch.Tensor) or target.shape != self.target_c2w.shape:
            raise RuntimeError("wrong-camera override target span differs")
        return self.base(**{**kwargs, "target_c2w": self.target_c2w.to(target)})


class CertifiedZipSplatTargetRescue:
    """Render a transient M- world; never commits generated Gaussian geometry."""

    def __init__(self, *, device: str = "cuda", view_budget: int = 8,
                 compression: float = 1.0, offload_after_call: bool = True,
                 source_schedule: Iterable[Mapping] | None = None):
        if view_budget < 1:
            raise ValueError("ZipSplat view budget must be positive")
        if str(device).startswith("cuda"):
            # gsplat silently installs a ``None`` backend when nvcc/CUDA_HOME is
            # absent, then fails only after the expensive local rollout arm has
            # finished.  Force the lazy extension check at construction so a
            # misconfigured launch fails before loading the carrier or writing
            # partial scientific artifacts.
            try:
                from gsplat.cuda._backend import _C
            except Exception as error:  # pragma: no cover - installation failure
                raise RuntimeError(
                    "ZipSplat CUDA backend import failed; bind CUDA_HOME and nvcc "
                    "before launching the probe") from error
            if _C is None:
                raise RuntimeError(
                    "ZipSplat CUDA backend is disabled; use CUDA_HOME=/home/hj/"
                    "miniconda3/envs/cuda128 and put its bin directory on PATH")
        self.device = device
        self.view_budget = int(view_budget)
        self.offload_after_call = bool(offload_after_call)
        self.geometry = ZipSplatGeometry(
            device=device, compression=float(compression))
        self._cache: OrderedDict[tuple[int, ...], object] = OrderedDict()
        self.source_schedule = tuple(dict(item) for item in (source_schedule or ()))

    def applies(self, memory, history_boundary: int) -> bool:
        """Whether a frozen exact-revisit intervention is deployable now.

        A clean trajectory schedule is only a request.  Generated endpoints may
        fail certification, so the exact same source set is not guaranteed to
        exist in the on-policy archive.  Missing evidence is an abstention (the
        caller keeps the Lyra local scaffold), never a late renderer exception
        or a silent substitution with recent frames.
        """
        if not self.source_schedule:
            return True
        candidates = [item for item in memory.archive_candidates()
                      if item.frame_id <= int(history_boundary)
                      and item.persistent_geometry and item.authority > 0]
        anchors = [item for item in candidates
                   if item.geometry_status == "anchor"]
        if len(anchors) != 1:
            return False
        anchor_id = int(anchors[0].frame_id)
        matches = [row for row in self.source_schedule
                   if int(row["history_boundary_pose_frame_id"])
                   == int(history_boundary)
                   and int(row["anchor_pose_frame_id"]) == anchor_id]
        if len(matches) != 1:
            return False
        requested = tuple(map(int, matches[0]["archive_pose_frame_ids"]))
        available = {int(item.frame_id) for item in candidates}
        return bool(requested) and all(frame_id in available
                                       for frame_id in requested)

    def _scheduled_sources(self, candidates, history_boundary: int):
        """Resolve a preregistered old-view set without falling back to recency."""
        if not self.source_schedule:
            return None
        anchors = [item for item in candidates if item.geometry_status == "anchor"]
        if len(anchors) != 1:
            raise RuntimeError("scheduled ZipSplat rescue requires one anchor")
        anchor_id = int(anchors[0].frame_id)
        matches = [row for row in self.source_schedule
                   if int(row["history_boundary_pose_frame_id"])
                   == int(history_boundary)
                   and int(row["anchor_pose_frame_id"]) == anchor_id]
        if len(matches) != 1:
            raise RuntimeError(
                "ZipSplat source schedule did not uniquely identify this history")
        requested = tuple(map(int, matches[0]["archive_pose_frame_ids"]))
        if not requested or len(requested) > self.view_budget:
            raise RuntimeError("scheduled ZipSplat source count exceeds its budget")
        by_id = {int(item.frame_id): item for item in candidates}
        missing = [frame_id for frame_id in requested if frame_id not in by_id]
        if missing:
            raise RuntimeError(
                f"scheduled ZipSplat sources are unavailable after certification: {missing}")
        return [by_id[frame_id] for frame_id in requested]

    def _sources(self, memory, history_boundary: int):
        candidates = [item for item in memory.archive_candidates()
                      if item.frame_id <= history_boundary
                      and item.persistent_geometry and item.authority > 0]
        if not candidates or candidates[0].geometry_status != "anchor":
            raise RuntimeError("ZipSplat rescue requires the immutable anchor")
        scheduled = self._scheduled_sources(candidates, history_boundary)
        if scheduled is not None:
            return scheduled
        if len(candidates) > self.view_budget:
            candidates = [candidates[0], *candidates[-(self.view_budget - 1):]] \
                if self.view_budget > 1 else [candidates[0]]
        return candidates

    def _scene(self, sources, gauge, image_size):
        key = tuple(item.frame_id for item in sources)
        scene = self._cache.get(key)
        if scene is not None:
            self._cache.move_to_end(key)
            return scene
        self.geometry.model.to(self.device).eval()
        source_poses = torch.stack([
            _archive_pose(item, gauge) for item in sources])
        relative_poses, reference_w2c = rebase_c2w_to_first(source_poses)
        gaussians = self.geometry.lift_frames(
            [((_endpoint_rgb(item, image_size) + 1) / 2).clamp(0, 1)
             for item in sources],
            list(relative_poses),
            [item.intrinsics.detach().float() for item in sources])
        scene = {"gaussians": gaussians,
                 "reference_w2c": reference_w2c}
        self._cache[key] = scene
        while len(self._cache) > 2:
            self._cache.popitem(last=False)
        return scene

    @torch.no_grad()
    def __call__(self, *, memory, gauge, target_c2w: torch.Tensor,
                 target_intrinsics: torch.Tensor, image_size: tuple[int, int],
                 history_boundary: int) -> dict:
        if (target_c2w.ndim != 3 or target_c2w.shape[-2:] != (4, 4)
                or target_intrinsics.shape != (len(target_c2w), 3, 3)):
            raise ValueError("ZipSplat target camera inventory is invalid")
        sources = self._sources(memory, int(history_boundary))
        scene = self._scene(sources, gauge, image_size)
        height, width = image_size
        center = central_square_mask(height, width, device=self.device)[None, None]
        frames = []
        for pose, K in zip(target_c2w, target_intrinsics):
            relative_pose = scene["reference_w2c"].to(pose) @ pose
            item = self.geometry.render(
                scene["gaussians"], K, relative_pose, height, width)
            valid = item["valid"] * center.to(item["valid"])
            frames.append({
                "rgb": (item["rgb"] * valid).float().cpu(),
                "depth": (item["depth"] * valid).float().cpu(),
                "valid": valid.float().cpu(),
                "conf": (item["conf"] * valid).float().cpu(),
            })
        if self.offload_after_call:
            # Cached Gaussian tensors cannot remain on CUDA if DA3 recovery follows.
            self._cache.clear()
            self.geometry.model.cpu()
            torch.cuda.empty_cache()
        return {
            "frames": frames,
            "source_frame_ids": tuple(item.frame_id for item in sources),
            "authority": min(float(item.authority) for item in sources),
            "oldest_age": max(int(history_boundary) - item.frame_id
                              for item in sources),
        }

    def release(self) -> None:
        self._cache.clear()
        self.geometry.model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
