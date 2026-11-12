"""Exact one-hop generated feedback path used by the stationary self-aug row."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Protocol

import torch

from gaussverse.memory.gate_omega_memory import EndpointEvidence, OmegaMemoryState
from gaussverse.training.gate_omega import (
    CertificationMeasurements,
    CertificationThresholds,
    DeploymentGauge,
    realization_residuals,
)
from gaussverse.training.gate_omega_trainer import SELF_AUG_ROLLOUT_FORWARDS


@dataclass(frozen=True)
class GeneratedChunk:
    decoded_pixels: torch.Tensor
    commanded_endpoint_c2w: torch.Tensor
    endpoint_intrinsics: torch.Tensor
    sampler_forwards: int
    frame_ids: tuple[int, ...]


@dataclass(frozen=True)
class PoseRecovery:
    realized_endpoint_c2w: torch.Tensor | None
    finite: bool
    nondegenerate: bool
    translation_residual: float
    rotation_residual_deg: float
    track_valid: bool
    track_translation_disagreement: float
    track_rotation_disagreement_deg: float
    recovery_id: str
    realized_frame_c2w: torch.Tensor | None = None


@dataclass(frozen=True)
class GeometryRecovery:
    depth: torch.Tensor
    confidence_map: torch.Tensor
    mean_confidence: float
    supported_fraction: float
    reprojection_error_px: float
    depth_log_error: float
    estimator_frame_ids: tuple[int, ...]
    recovery_id: str
    gauge_manifest_sha256: str
    depth_alignment_scale: float = 1.0


@dataclass(frozen=True)
class OneHopHandoff:
    generated: GeneratedChunk
    pose: PoseRecovery
    geometry: GeometryRecovery | None
    encoded_latent: torch.Tensor | None
    endpoint: EndpointEvidence | None
    next_relative_command: torch.Tensor | None
    abstained: bool
    fallback: str | None


class HopSampler(Protocol):
    def __call__(self, *, steps: int, seed: int) -> GeneratedChunk: ...


def _relative_command(gauge: DeploymentGauge, realized_c2w: torch.Tensor,
                      next_target_c2w: torch.Tensor) -> torch.Tensor:
    realized = gauge.normalize_c2w(realized_c2w)
    target = gauge.normalize_c2w(next_target_c2w)
    return torch.linalg.inv(realized) @ target


@torch.no_grad()
def execute_one_hop(*, sampler: HopSampler,
                    pose_backend: Callable[[torch.Tensor, tuple[int, ...]], PoseRecovery],
                    depth_backend: Callable[[torch.Tensor, tuple[int, ...]], GeometryRecovery],
                    vae_encoder: Callable[[torch.Tensor], torch.Tensor],
                    memory: OmegaMemoryState, gauge: DeploymentGauge,
                    next_target_c2w: torch.Tensor | None,
                    thresholds: CertificationThresholds,
                    seed: int, history_boundary: int,
                    generated_frame_id: int,
                    expected_sampler_forwards: int =
                    SELF_AUG_ROLLOUT_FORWARDS) -> OneHopHandoff:
    """Decode -> pose/depth estimate -> certify -> re-encode -> recompute command."""
    expected = int(expected_sampler_forwards)
    if expected < 1:
        raise ValueError("expected sampler forwards must be positive")
    generated = sampler(steps=expected, seed=int(seed))
    if generated.sampler_forwards != expected:
        raise RuntimeError(
            f"one-hop sampler executed {generated.sampler_forwards}, expected "
            f"{expected} forwards")
    pixels = generated.decoded_pixels
    if pixels.ndim not in (4, 5) or not pixels.is_floating_point() \
            or not bool(torch.isfinite(pixels).all()):
        raise RuntimeError("one-hop sampler did not return finite decoded pixels")
    if (not generated.frame_ids
            or tuple(sorted(set(generated.frame_ids))) != generated.frame_ids
            or generated.frame_ids[-1] != generated_frame_id
            or any(frame < 0 or frame > history_boundary
                   for frame in generated.frame_ids)):
        raise RuntimeError("generated frame IDs violate the causal handoff boundary")
    if generated.commanded_endpoint_c2w.shape != (4, 4) or not bool(
            torch.isfinite(generated.commanded_endpoint_c2w).all()):
        raise RuntimeError("one-hop sampler returned an invalid commanded endpoint")
    if generated.endpoint_intrinsics.shape != (3, 3) or not bool(
            torch.isfinite(generated.endpoint_intrinsics).all()) \
            or abs(float(torch.linalg.det(generated.endpoint_intrinsics.float()))) < 1e-12:
        raise RuntimeError("one-hop sampler returned invalid endpoint intrinsics")
    pose = pose_backend(pixels, generated.frame_ids)
    if not pose.finite or not pose.nondegenerate or pose.realized_endpoint_c2w is None:
        return OneHopHandoff(
            generated, pose, None, None, None, None, True,
            "clean_positive_pose_abstention")
    if pose.realized_endpoint_c2w.shape != (4, 4) or not torch.isfinite(
            pose.realized_endpoint_c2w).all():
        raise RuntimeError("pose backend marked a non-finite realized pose as valid")
    if (pose.realized_frame_c2w is None
            or pose.realized_frame_c2w.shape
                != (len(generated.frame_ids), 4, 4)
            or not bool(torch.isfinite(pose.realized_frame_c2w).all())
            or not torch.equal(
                pose.realized_frame_c2w[-1].to(pose.realized_endpoint_c2w),
                pose.realized_endpoint_c2w)):
        raise RuntimeError(
            "pose backend did not preserve every realized generated-frame pose")
    geometry = depth_backend(pixels, generated.frame_ids)
    if (not pose.recovery_id or pose.recovery_id != geometry.recovery_id):
        raise RuntimeError("pose and depth do not belong to one joint recovery")
    expected_gauge_sha256 = gauge.manifest()["payload_sha256"]
    if geometry.gauge_manifest_sha256 != expected_gauge_sha256:
        raise RuntimeError("geometry depth is not in the active deployment gauge")
    if geometry.depth.ndim < 2 or geometry.confidence_map.shape != geometry.depth.shape \
            or not bool(torch.isfinite(geometry.depth).all()) \
            or not bool(torch.isfinite(geometry.confidence_map).all()) \
            or bool((geometry.depth <= 0).all()) \
            or bool((geometry.confidence_map < 0).any()) \
            or bool((geometry.confidence_map > 1).any()) \
            or not math.isfinite(float(geometry.mean_confidence)) \
            or not math.isfinite(float(geometry.supported_fraction)):
        raise RuntimeError("deployment geometry backend returned invalid depth/confidence")
    # Re-encode exactly the pixels seen by the deployment estimators. No latent shortcut.
    encoded = vae_encoder(pixels)
    if not encoded.is_floating_point() or not bool(torch.isfinite(encoded).all()):
        raise RuntimeError("persistent VAE re-encode returned non-finite latents")
    translation_residual, rotation_residual_deg = realization_residuals(
        gauge, generated.commanded_endpoint_c2w, pose.realized_endpoint_c2w)
    measurement = CertificationMeasurements(
        pose_finite=pose.finite, pose_nondegenerate=pose.nondegenerate,
        translation_residual=translation_residual,
        rotation_residual_deg=rotation_residual_deg,
        track_valid=pose.track_valid,
        track_translation_disagreement=pose.track_translation_disagreement,
        track_rotation_disagreement_deg=pose.track_rotation_disagreement_deg,
        reprojection_error_px=geometry.reprojection_error_px,
        depth_log_error=geometry.depth_log_error,
        depth_confidence=geometry.mean_confidence,
        supported_fraction=geometry.supported_fraction,
        estimator_frame_ids=geometry.estimator_frame_ids,
        history_boundary=int(history_boundary),
    )
    endpoint = memory.commit_generated(
        frame_id=int(generated_frame_id),
        commanded_pose=generated.commanded_endpoint_c2w,
        intrinsics=generated.endpoint_intrinsics,
        gauge_manifest_sha256=geometry.gauge_manifest_sha256,
        realized_pose=pose.realized_endpoint_c2w,
        measurement=measurement, thresholds=thresholds,
        confidence=geometry.mean_confidence,
        latent=encoded, depth=geometry.depth,
        rgb=pixels,
        depth_confidence=geometry.confidence_map)
    next_command = (None if next_target_c2w is None else _relative_command(
        gauge, pose.realized_endpoint_c2w, next_target_c2w))
    return OneHopHandoff(
        generated, pose, geometry, encoded, endpoint, next_command,
        False, None)
