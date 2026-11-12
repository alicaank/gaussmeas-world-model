"""Executable invariants and pure kernels for the GATE-ΩR training contract.

The heavyweight trainer consumes this module; preflight tests exercise it without loading
Wan.  Keeping gauge, sampling, binding, causality, scaffold fusion, and certification in
one dependency-light module prevents training and deployment from reimplementing them
differently.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

from gaussverse.camera.commands import WALKING_MOTION


CONTRACT_ID = "gate_omega_r.2026-08-27.walking_local"
H_CAPACITY = 20
G_CHUNK = 2
READER_BLOCKS = (7, 15, 23)
CONDITION_BLOCKS = tuple(range(30))
LOCAL_HOP_MOTION_CONTRACT = "walking_local_metric_scale_v1"
# These are broad source-video continuity limits, not physical inference-command
# limits. Monocular gauge units are not centimetres; the deployment host supplies
# the metric scale separately. Training deliberately covers motions wider than the
# UI's typical 10--20 cm / 5--10 degree request.
LOCAL_MIN_TRANSLATION_GAUGE = 0.0
LOCAL_MAX_TRANSLATION_GAUGE = 0.75
LOCAL_MAX_ROTATION_DEG = 25.0
LOCAL_MAX_ADJACENT_TRANSLATION_GAUGE = LOCAL_MAX_TRANSLATION_GAUGE
LOCAL_MAX_ADJACENT_ROTATION_DEG = LOCAL_MAX_ROTATION_DEG
LOCAL_TRANSLATION_SPLIT = 0.05
LOCAL_ROTATION_SPLIT_DEG = 5.0
BINDING_WEIGHT = 0.10
BINDING_MIN_SUPPORT = 16
DEPTH_BANDS = ((1, 3, 0.20), (4, 9, 0.20), (10, 18, 0.25), (19, 27, 0.35))
ROW_ROLES = (
    "clean_translation", "clean_rotation", "clean_mixed", "clean_low_motion",
    "binding", "deep", "deep", "selfaug",
)


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def deterministic_uniform(material: object) -> float:
    """Stable U[0,1) draw independent of Python and torch RNG versions."""
    integer = int.from_bytes(hashlib.sha256(_canonical(material)).digest()[:8], "big")
    return integer / float(1 << 64)


def stationary_row_roles(step: int) -> tuple[str, ...]:
    """Fixed role counts with deterministic rotation to avoid row-position confounding."""
    if step < 0:
        raise ValueError("optimizer step must be nonnegative")
    offset = int(hashlib.sha256(f"{CONTRACT_ID}:roles:{step}".encode()).hexdigest()[:8], 16)
    offset %= len(ROW_ROLES)
    return ROW_ROLES[offset:] + ROW_ROLES[:offset]


def sample_history_depth(draw_id: str) -> int:
    """Draw the contracted 20/20/25/35 deep-history PMF deterministically."""
    if not isinstance(draw_id, str) or not draw_id:
        raise ValueError("draw_id must be a nonempty string")
    band_u = deterministic_uniform({"contract": CONTRACT_ID, "draw": draw_id,
                                    "purpose": "history_band"})
    cumulative = 0.0
    selected = DEPTH_BANDS[-1]
    for band in DEPTH_BANDS:
        cumulative += band[2]
        if band_u < cumulative:
            selected = band
            break
    lo, hi, _ = selected
    inside_u = deterministic_uniform({"contract": CONTRACT_ID, "draw": draw_id,
                                      "purpose": "history_inside", "band": [lo, hi]})
    return lo + min(int(inside_u * (hi - lo + 1)), hi - lo)


@dataclass(frozen=True)
class DeploymentGauge:
    """Anchor-observable similarity gauge shared by training and deployment."""

    anchor_rotation: torch.Tensor
    anchor_origin: torch.Tensor
    scale: torch.Tensor

    @classmethod
    def from_anchor(cls, anchor_c2w: torch.Tensor,
                    anchor_depth: torch.Tensor) -> "DeploymentGauge":
        if anchor_c2w.shape != (4, 4):
            raise ValueError("anchor_c2w must be 4x4")
        if not torch.isfinite(anchor_c2w).all():
            raise ValueError("anchor pose must be finite")
        depth = anchor_depth.detach().float()
        positive = depth[torch.isfinite(depth) & (depth > 0)]
        if positive.numel() < 1:
            raise ValueError("anchor depth has no finite positive support")
        scale = positive.median().to(anchor_c2w.device, torch.float64)
        if not bool(torch.isfinite(scale)) or float(scale) <= 0:
            raise ValueError("anchor gauge scale must be finite and positive")
        rotation = anchor_c2w[:3, :3].to(torch.float64)
        identity = torch.eye(3, device=rotation.device, dtype=rotation.dtype)
        if not torch.allclose(rotation.T @ rotation, identity, atol=1e-6, rtol=1e-6):
            raise ValueError("anchor rotation is not orthonormal")
        if float(torch.linalg.det(rotation)) <= 0:
            raise ValueError("anchor rotation must be proper")
        return cls(rotation, anchor_c2w[:3, 3].to(torch.float64), scale)

    @classmethod
    def from_manifest(cls, value: dict) -> "DeploymentGauge":
        """Restore an immutable cached gauge after verifying its content address."""
        required = {"contract", "source", "rotation", "origin", "scale",
                    "payload_sha256"}
        if set(value) != required or value.get("contract") != CONTRACT_ID \
                or value.get("source") != "anchor_rgb_deployment_estimator":
            raise ValueError("deployment gauge manifest has an incompatible schema")
        body = {key: item for key, item in value.items()
                if key != "payload_sha256"}
        if hashlib.sha256(_canonical(body)).hexdigest() != value["payload_sha256"]:
            raise ValueError("deployment gauge manifest digest is stale")
        rotation = torch.tensor(value["rotation"], dtype=torch.float64)
        origin = torch.tensor(value["origin"], dtype=torch.float64)
        scale = torch.tensor(value["scale"], dtype=torch.float64)
        if rotation.shape != (3, 3) or origin.shape != (3,) \
                or not bool(torch.isfinite(rotation).all()) \
                or not bool(torch.isfinite(origin).all()) \
                or not bool(torch.isfinite(scale)) or float(scale) <= 0:
            raise ValueError("deployment gauge manifest contains invalid tensors")
        identity = torch.eye(3, dtype=torch.float64)
        if not torch.allclose(rotation.T @ rotation, identity, atol=1e-6, rtol=1e-6) \
                or float(torch.linalg.det(rotation)) <= 0:
            raise ValueError("deployment gauge manifest rotation is invalid")
        result = cls(rotation, origin, scale)
        if result.manifest() != value:
            raise ValueError("deployment gauge manifest is not canonical")
        return result

    def normalize_points(self, points_world: torch.Tensor) -> torch.Tensor:
        points = points_world.to(device=self.anchor_origin.device, dtype=torch.float64)
        return torch.einsum(
            "ij,...j->...i", self.anchor_rotation.T,
            points - self.anchor_origin,
        ) / self.scale

    def normalize_c2w(self, c2w_world: torch.Tensor) -> torch.Tensor:
        if c2w_world.shape[-2:] != (4, 4):
            raise ValueError("camera poses must end in 4x4")
        pose = c2w_world.to(device=self.anchor_origin.device, dtype=torch.float64)
        out = torch.zeros_like(pose)
        out[..., :3, :3] = torch.einsum(
            "ij,...jk->...ik", self.anchor_rotation.T, pose[..., :3, :3])
        out[..., :3, 3] = self.normalize_points(pose[..., :3, 3])
        out[..., 3, 3] = 1
        return out

    def normalize_depth(self, depth: torch.Tensor) -> torch.Tensor:
        return depth.to(device=self.scale.device, dtype=torch.float64) / self.scale

    def manifest(self) -> dict:
        payload = {
            "contract": CONTRACT_ID,
            "source": "anchor_rgb_deployment_estimator",
            "rotation": self.anchor_rotation.cpu().tolist(),
            "origin": self.anchor_origin.cpu().tolist(),
            "scale": float(self.scale),
        }
        payload["payload_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
        return payload


def realization_residuals(gauge: DeploymentGauge, commanded_c2w: torch.Tensor,
                          realized_c2w: torch.Tensor) -> tuple[float, float]:
    """Commanded-versus-realized endpoint error in the sole deployment gauge."""
    commanded = gauge.normalize_c2w(commanded_c2w)
    realized = gauge.normalize_c2w(realized_c2w)
    translation = torch.linalg.vector_norm(
        commanded[:3, 3] - realized[:3, 3])
    relative_rotation = commanded[:3, :3].T @ realized[:3, :3]
    cosine = ((torch.trace(relative_rotation) - 1.0) / 2.0).clamp(-1.0, 1.0)
    rotation_deg = torch.rad2deg(torch.acos(cosine))
    return float(translation), float(rotation_deg)


def apply_global_similarity(c2w: torch.Tensor, depth: torch.Tensor, *,
                            rotation: torch.Tensor, translation: torch.Tensor,
                            scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Change the world coordinate frame by x' = scale R x + translation."""
    if scale <= 0 or not math.isfinite(scale):
        raise ValueError("similarity scale must be finite and positive")
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("similarity rotation/translation shapes are invalid")
    pose = c2w.to(torch.float64)
    R = rotation.to(device=pose.device, dtype=pose.dtype)
    t = translation.to(device=pose.device, dtype=pose.dtype)
    identity = torch.eye(3, device=pose.device, dtype=pose.dtype)
    if not torch.allclose(R.T @ R, identity, atol=1e-10, rtol=1e-10) \
            or float(torch.linalg.det(R)) <= 0:
        raise ValueError("similarity rotation must be proper and orthonormal")
    out = torch.zeros_like(pose)
    out[..., :3, :3] = torch.einsum("ij,...jk->...ik", R, pose[..., :3, :3])
    out[..., :3, 3] = scale * torch.einsum("ij,...j->...i", R, pose[..., :3, 3]) + t
    out[..., 3, 3] = 1
    return out, depth.to(torch.float64) * scale


def audit_prefix_causality(*, history_boundary: int,
                           conditioning_frame_ids: Iterable[int],
                           target_frame_ids: Iterable[int]) -> dict:
    conditioning = tuple(int(value) for value in conditioning_frame_ids)
    targets = tuple(int(value) for value in target_frame_ids)
    future = sorted({value for value in conditioning if value > history_boundary})
    leaked = sorted(set(conditioning) & set(targets))
    passed = not future and not leaked
    return {"pass": passed, "history_boundary": int(history_boundary),
            "conditioning_frame_ids": conditioning, "target_frame_ids": targets,
            "future_conditioning": future, "target_leak": leaked}


def validate_binding_pair(pair: dict) -> dict:
    """Validate the data identity constraints before any expensive binding forward."""
    required = {
        "prefix_frame_ids", "target_a_frame_ids", "target_b_frame_ids",
        "camera_a_id", "camera_b_id", "scaffold_a_camera_id",
        "scaffold_b_camera_id", "noise_id_a", "noise_id_b",
        "timestep_a", "timestep_b", "flow_pairs", "cross_pairs",
    }
    missing = sorted(required - set(pair))
    reasons = [f"missing field {name}" for name in missing]
    if not missing:
        prefix = set(map(int, pair["prefix_frame_ids"]))
        target_a = set(map(int, pair["target_a_frame_ids"]))
        target_b = set(map(int, pair["target_b_frame_ids"]))
        if prefix & (target_a | target_b):
            reasons.append("target frame occurs in the legal history prefix")
        if target_a & target_b:
            reasons.append("positive target spans overlap")
        if pair["camera_a_id"] == pair["camera_b_id"]:
            reasons.append("binding cameras are identical")
        if pair["scaffold_a_camera_id"] != pair["camera_a_id"]:
            reasons.append("scaffold A does not match camera A")
        if pair["scaffold_b_camera_id"] != pair["camera_b_id"]:
            reasons.append("scaffold B does not match camera B")
        if pair["noise_id_a"] != pair["noise_id_b"]:
            reasons.append("positive targets do not share epsilon")
        if pair["timestep_a"] != pair["timestep_b"]:
            reasons.append("positive targets do not share timestep")
        if {tuple(item) for item in pair["flow_pairs"]} != {("a", "a"), ("b", "b")}:
            reasons.append("flow supervision is not exactly the two positive pairs")
        if {tuple(item) for item in pair["cross_pairs"]} != {("a", "b"), ("b", "a")}:
            reasons.append("cross supervision is not exactly the two mismatched pairs")
    return {"pass": not reasons, "reasons": reasons}


def masked_flow_error(prediction: torch.Tensor, label: torch.Tensor,
                      mask: torch.Tensor, *,
                      min_support: int = BINDING_MIN_SUPPORT) -> torch.Tensor:
    if prediction.shape != label.shape:
        raise ValueError("prediction and label shapes differ")
    try:
        support = torch.broadcast_to(mask.to(device=prediction.device,
                                              dtype=prediction.dtype), prediction.shape)
    except RuntimeError as error:
        raise ValueError("binding mask is not broadcastable to flow tensors") from error
    spatial_support = torch.broadcast_to(mask.to(device=prediction.device).bool(),
                                         prediction.shape).any(dim=1)
    if int(spatial_support.sum()) < min_support:
        raise ValueError("binding comparison mask has insufficient support")
    denominator = support.sum().clamp_min(1.0)
    return ((prediction.float() - label.float()).square()
            * support.float()).sum() / denominator.float()


def symmetric_target_view_binding(e_aa: torch.Tensor, e_ab: torch.Tensor,
                                  e_bb: torch.Tensor, e_ba: torch.Tensor,
                                  *, epsilon: float = 1e-6) -> tuple[torch.Tensor, dict]:
    errors = tuple(value.float() for value in (e_aa, e_ab, e_bb, e_ba))
    if any(value.numel() != 1 or not bool(torch.isfinite(value).all())
           or bool((value < 0).any()) for value in errors):
        raise ValueError("binding errors must be finite nonnegative scalars")
    r_a = (errors[0] - errors[1]) / (
        0.5 * (errors[0] + errors[1]).detach() + epsilon)
    r_b = (errors[2] - errors[3]) / (
        0.5 * (errors[2] + errors[3]).detach() + epsilon)
    floor = F.softplus(torch.tensor(-2.0, device=r_a.device, dtype=r_a.dtype))
    loss = 0.5 * (F.softplus(r_a) + F.softplus(r_b)) - floor
    diagnostics = {
        "r_a": float(r_a.detach()), "r_b": float(r_b.detach()),
        "a_bound": bool(errors[0].detach() < errors[1].detach()),
        "b_bound": bool(errors[2].detach() < errors[3].detach()),
    }
    return loss, diagnostics


def fuse_scaffold_sources(values: torch.Tensor, depths: torch.Tensor,
                            validity: torch.Tensor, confidence: torch.Tensor, *,
                            ages: torch.Tensor | None = None,
                            tie_relative: float = 1e-4,
                            tie_absolute: float = 1e-6) -> dict[str, torch.Tensor]:
    """Fuse anchor/recent projections: nearest z wins, confidence breaks ties only."""
    if values.ndim != 4 or depths.ndim != 4 or validity.ndim != 4 or confidence.ndim != 4:
        raise ValueError("scaffold sources must be NCHW tensors")
    n, channels, height, width = values.shape
    expected = (n, 1, height, width)
    if n not in (1, 2) or depths.shape != expected or validity.shape != expected \
            or confidence.shape != expected:
        raise ValueError("scaffold supports one anchor and at most one recent source")
    finite = torch.isfinite(depths) & torch.isfinite(confidence) & torch.isfinite(values).all(
        dim=1, keepdim=True)
    valid = validity.bool() & finite & (depths > 0) & (confidence >= 0) & (confidence <= 1)
    infinity = torch.full_like(depths, torch.inf)
    candidate_depth = torch.where(valid, depths, infinity)
    nearest = candidate_depth.min(dim=0, keepdim=True).values
    tolerance = tie_absolute + tie_relative * nearest.abs()
    tied = valid & ((candidate_depth - nearest).abs() <= tolerance)
    tie_score = torch.where(tied, confidence, torch.full_like(confidence, -1))
    selected = tie_score.argmax(dim=0, keepdim=True)
    selected_values = torch.gather(values, 0, selected.expand(1, channels, height, width))
    selected_depth = torch.gather(depths, 0, selected)
    selected_confidence = torch.gather(confidence, 0, selected)
    any_valid = valid.any(dim=0, keepdim=True)
    selected_values = selected_values * any_valid.to(values.dtype)
    selected_depth = selected_depth * any_valid.to(depths.dtype)
    selected_confidence = selected_confidence * any_valid.to(confidence.dtype)

    valid_float = valid.to(values.dtype)
    count = valid_float.sum(dim=0, keepdim=True).clamp_min(1)
    mean = (values * valid_float).sum(dim=0, keepdim=True) / count
    rgb_disagreement = (((values - mean).square() * valid_float).sum(
        dim=0, keepdim=True) / count).mean(dim=1, keepdim=True)
    log_depth = torch.where(valid, depths.clamp_min(1e-8).log(), torch.zeros_like(depths))
    depth_mean = (log_depth * valid_float).sum(dim=0, keepdim=True) / count
    depth_disagreement = (((log_depth - depth_mean).square() * valid_float).sum(
        dim=0, keepdim=True) / count)
    disagreement = torch.sqrt(rgb_disagreement + depth_disagreement).to(values.dtype)
    disagreement = disagreement * (valid.sum(dim=0, keepdim=True) > 1).to(values.dtype)

    source_one_hot = torch.zeros((1, 2, height, width), device=values.device,
                                 dtype=values.dtype)
    source_one_hot.scatter_(1, selected.permute(1, 0, 2, 3), 1.0)
    source_one_hot *= any_valid.permute(1, 0, 2, 3).to(values.dtype)
    if ages is None:
        ages = torch.zeros(n, device=values.device, dtype=values.dtype)
    if ages.shape != (n,) or not torch.isfinite(ages).all() or bool((ages < 0).any()):
        raise ValueError("source ages must be a finite nonnegative N-vector")
    age_maps = ages.to(values).view(n, 1, 1, 1).expand(n, 1, height, width)
    selected_age = torch.gather(age_maps, 0, selected) * any_valid.to(values.dtype)
    inverse_depth = torch.where(selected_depth > 0, selected_depth.reciprocal(),
                                torch.zeros_like(selected_depth))
    return {
        "features": selected_values,
        "inverse_depth": inverse_depth,
        "validity": any_valid.to(values.dtype),
        "confidence": selected_confidence,
        "disagreement": disagreement,
        "age": selected_age,
        "source_one_hot": source_one_hot,
        "unknown": (~any_valid).to(values.dtype),
        "selected_index": selected,
    }


@dataclass(frozen=True)
class CertificationThresholds:
    max_translation_residual: float
    max_rotation_residual_deg: float
    max_track_translation_disagreement: float
    max_track_rotation_disagreement_deg: float
    max_reprojection_error_px: float
    max_depth_log_error: float
    min_depth_confidence: float
    min_supported_fraction: float

    def __post_init__(self):
        values = asdict(self)
        if any(not math.isfinite(float(value)) or float(value) < 0
               for value in values.values()):
            raise ValueError("certification thresholds must be finite and nonnegative")
        if not 0 <= self.min_depth_confidence <= 1 \
                or not 0 <= self.min_supported_fraction <= 1:
            raise ValueError("confidence and coverage thresholds must lie in [0,1]")


@dataclass(frozen=True)
class CertificationMeasurements:
    pose_finite: bool
    pose_nondegenerate: bool
    translation_residual: float
    rotation_residual_deg: float
    track_valid: bool
    track_translation_disagreement: float
    track_rotation_disagreement_deg: float
    reprojection_error_px: float
    depth_log_error: float
    depth_confidence: float
    supported_fraction: float
    estimator_frame_ids: tuple[int, ...]
    history_boundary: int


def certify_geometry(measurement: CertificationMeasurements,
                     thresholds: CertificationThresholds) -> dict:
    reasons = []
    error_numeric = [
        measurement.translation_residual, measurement.rotation_residual_deg,
        measurement.track_translation_disagreement,
        measurement.track_rotation_disagreement_deg,
        measurement.reprojection_error_px, measurement.depth_log_error,
    ]
    numeric = [*error_numeric, measurement.depth_confidence,
               measurement.supported_fraction]
    if not measurement.pose_finite or not measurement.pose_nondegenerate \
            or not all(math.isfinite(float(value)) for value in numeric):
        reasons.append("pose recovery is invalid or degenerate")
    if any(math.isfinite(float(value)) and float(value) < 0
           for value in error_numeric):
        reasons.append("certification error measurements must be nonnegative")
    if (not math.isfinite(float(measurement.depth_confidence))
            or not 0 <= float(measurement.depth_confidence) <= 1):
        reasons.append("depth confidence is outside [0,1]")
    if (not math.isfinite(float(measurement.supported_fraction))
            or not 0 <= float(measurement.supported_fraction) <= 1):
        reasons.append("supported fraction is outside [0,1]")
    if measurement.translation_residual > thresholds.max_translation_residual:
        reasons.append("commanded-realized translation residual exceeds threshold")
    if measurement.rotation_residual_deg > thresholds.max_rotation_residual_deg:
        reasons.append("commanded-realized rotation residual exceeds threshold")
    if not measurement.track_valid:
        reasons.append("feature-track/RANSAC confirmation is invalid")
    if measurement.track_translation_disagreement > thresholds.max_track_translation_disagreement:
        reasons.append("track/backend translation disagreement exceeds threshold")
    if measurement.track_rotation_disagreement_deg > thresholds.max_track_rotation_disagreement_deg:
        reasons.append("track/backend rotation disagreement exceeds threshold")
    if measurement.reprojection_error_px > thresholds.max_reprojection_error_px:
        reasons.append("reprojection error exceeds threshold")
    if measurement.depth_log_error > thresholds.max_depth_log_error:
        reasons.append("depth disagreement exceeds threshold")
    if measurement.depth_confidence < thresholds.min_depth_confidence:
        reasons.append("depth confidence is below threshold")
    if measurement.supported_fraction < thresholds.min_supported_fraction:
        reasons.append("covisible support is below threshold")
    recorded_ids = tuple(measurement.estimator_frame_ids)
    invalid_ids = sorted({repr(frame) for frame in recorded_ids
                          if (isinstance(frame, bool)
                              or not isinstance(frame, int)
                              or frame < 0)})
    if not recorded_ids:
        reasons.append("estimator input-frame IDs are not recorded")
    if measurement.history_boundary < 0 or invalid_ids:
        reasons.append("estimator input-frame IDs or history boundary are invalid")
    valid_ids = tuple(frame for frame in recorded_ids
                      if isinstance(frame, int) and not isinstance(frame, bool)
                      and frame >= 0)
    future = sorted({frame for frame in valid_ids
                     if frame > measurement.history_boundary})
    if future:
        reasons.append("estimator frame IDs cross the causal boundary")
    return {"pass": not reasons, "geometry_status": "certified" if not reasons else "rejected",
            "reasons": reasons, "future_estimator_frame_ids": future,
            "invalid_estimator_frame_ids": invalid_ids,
            "measurements": asdict(measurement), "thresholds": asdict(thresholds)}
