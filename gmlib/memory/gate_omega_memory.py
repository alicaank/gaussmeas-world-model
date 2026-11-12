"""Fail-closed generated-write state for GATE-ΩR."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

from gmlib.training.gate_omega import (
    CertificationMeasurements,
    CertificationThresholds,
    H_CAPACITY,
    certify_geometry,
)


GeometryStatus = Literal["anchor", "certified", "rejected"]
AppearanceStatus = Literal["persistent_anchor", "temporary", "rejected"]


@dataclass(frozen=True)
class EndpointEvidence:
    frame_id: int
    commanded_pose: torch.Tensor
    realized_pose: torch.Tensor | None
    intrinsics: torch.Tensor
    gauge_manifest_sha256: str
    pose_valid: bool
    realization_translation_residual: float
    realization_rotation_residual_deg: float
    content_origin: str
    geometry_status: GeometryStatus
    appearance_status: AppearanceStatus
    confidence: float
    authority: float
    estimator_frame_ids: tuple[int, ...]
    certification_reasons: tuple[str, ...] = ()
    latent: torch.Tensor | None = field(default=None, repr=False, compare=False)
    rgb: torch.Tensor | None = field(default=None, repr=False, compare=False)
    depth: torch.Tensor | None = field(default=None, repr=False, compare=False)
    depth_confidence: torch.Tensor | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.frame_id < 0 or not self.content_origin:
            raise ValueError("endpoint frame ID/origin is invalid")
        if not 0 <= self.confidence <= 1 or not 0 <= self.authority <= 1:
            raise ValueError("endpoint confidence and authority must lie in [0,1]")
        if self.commanded_pose.shape != (4, 4) or not torch.isfinite(
                self.commanded_pose).all():
            raise ValueError("endpoint commanded pose must be finite 4x4")
        if self.intrinsics.shape != (3, 3) or not torch.isfinite(
                self.intrinsics).all() or abs(float(torch.linalg.det(
                    self.intrinsics.float()))) < 1e-12:
            raise ValueError("endpoint intrinsics must be one finite nonsingular 3x3")
        if (not isinstance(self.gauge_manifest_sha256, str)
                or len(self.gauge_manifest_sha256) != 64):
            raise ValueError("endpoint requires its deployment-gauge SHA-256")
        if self.pose_valid and (self.realized_pose is None
                                or self.realized_pose.shape != (4, 4)
                                or not torch.isfinite(self.realized_pose).all()):
            raise ValueError("pose-valid endpoint requires one finite realized pose")
        if self.geometry_status == "certified" and not self.pose_valid:
            raise ValueError("geometry certification requires a valid realized pose")
        if self.depth_confidence is not None and (
                not self.depth_confidence.is_floating_point()
                or not bool(torch.isfinite(self.depth_confidence).all())
                or bool(((self.depth_confidence < 0)
                         | (self.depth_confidence > 1)).any())):
            raise ValueError("endpoint depth confidence must be finite in [0,1]")
        if self.rgb is not None and (
                not self.rgb.is_floating_point()
                or not bool(torch.isfinite(self.rgb).all())):
            raise ValueError("endpoint RGB must be a finite floating tensor")

    @property
    def persistent_geometry(self) -> bool:
        return self.geometry_status in {"anchor", "certified"}


class OmegaMemoryState:
    """Separate bounded appearance history from persistent certified geometry."""

    def __init__(self, anchor: EndpointEvidence, *, history_capacity: int = H_CAPACITY,
                 uncertified_authority_cap: float = 0.25,
                 provisional_retrieval_capacity: int = 64):
        if anchor.geometry_status != "anchor" or not anchor.persistent_geometry:
            raise ValueError("Omega memory requires one immutable certified anchor")
        if history_capacity != H_CAPACITY:
            raise ValueError(f"Omega temporal capacity is fixed at H={H_CAPACITY}")
        if not 0 <= uncertified_authority_cap <= 1:
            raise ValueError("uncertified authority cap must lie in [0,1]")
        if provisional_retrieval_capacity < 0:
            raise ValueError("provisional retrieval capacity must be nonnegative")
        self.anchor = anchor
        self.history_capacity = history_capacity
        self.uncertified_authority_cap = float(uncertified_authority_cap)
        self.provisional_retrieval_capacity = int(provisional_retrieval_capacity)
        self.history: list[EndpointEvidence] = [anchor]
        self.archive: list[EndpointEvidence] = [anchor]
        # Rejected writes must never become geometry/scaffold authority. Keep a
        # separate bounded bank solely for low-authority native appearance reads.
        self.provisional_retrieval: list[EndpointEvidence] = []
        self.recent_certified: EndpointEvidence | None = None

    def commit_generated(self, *, frame_id: int, commanded_pose: torch.Tensor,
                         intrinsics: torch.Tensor,
                         gauge_manifest_sha256: str,
                         realized_pose: torch.Tensor | None,
                         measurement: CertificationMeasurements,
                         thresholds: CertificationThresholds,
                         confidence: float, content_origin: str = "generated",
                         latent: torch.Tensor | None = None,
                         rgb: torch.Tensor | None = None,
                         depth: torch.Tensor | None = None,
                         depth_confidence: torch.Tensor | None = None) -> EndpointEvidence:
        if frame_id <= self.history[-1].frame_id:
            raise ValueError("generated endpoint frame IDs must increase monotonically")
        if bool(measurement.pose_finite) != bool(realized_pose is not None):
            raise ValueError("pose measurement and realized-pose presence disagree")
        result = certify_geometry(measurement, thresholds)
        certified = bool(result["pass"])
        authority = float(confidence) if certified else min(
            float(confidence), self.uncertified_authority_cap)
        endpoint = EndpointEvidence(
            frame_id=int(frame_id), commanded_pose=commanded_pose.detach().clone(),
            intrinsics=intrinsics.detach().clone(),
            gauge_manifest_sha256=gauge_manifest_sha256,
            realized_pose=(None if realized_pose is None
                           else realized_pose.detach().clone()),
            pose_valid=bool(measurement.pose_finite and measurement.pose_nondegenerate),
            realization_translation_residual=float(measurement.translation_residual),
            realization_rotation_residual_deg=float(measurement.rotation_residual_deg),
            content_origin=content_origin,
            geometry_status="certified" if certified else "rejected",
            appearance_status="temporary" if confidence > 0 else "rejected",
            confidence=float(confidence), authority=authority,
            estimator_frame_ids=tuple(map(int, measurement.estimator_frame_ids)),
            certification_reasons=tuple(result["reasons"]),
            latent=(None if latent is None else latent.detach().clone()),
            rgb=(None if rgb is None else rgb.detach().clone()),
            depth=(None if depth is None else depth.detach().clone()),
            depth_confidence=(None if depth_confidence is None
                              else depth_confidence.detach().clone()))
        evicted = ([self.history[1].frame_id] if len(self.history) >= self.history_capacity else [])
        self.history.append(endpoint)
        # The immutable anchor is never evicted. H counts total occupied temporal slots.
        while len(self.history) > self.history_capacity:
            del self.history[1]
        if certified:
            self.archive.append(endpoint)
            self.recent_certified = endpoint
        elif (self.provisional_retrieval_capacity > 0 and endpoint.pose_valid
              and endpoint.appearance_status == "temporary"
              and endpoint.authority > 0 and endpoint.latent is not None
              and endpoint.depth is not None
              and endpoint.depth_confidence is not None):
            self.provisional_retrieval.append(endpoint)
            del self.provisional_retrieval[
                :max(0, len(self.provisional_retrieval)
                     - self.provisional_retrieval_capacity)]
        from gmlib.memory.audit import emit
        emit("memory_write", frame_id=int(frame_id), gauge=gauge_manifest_sha256,
             history_added=True, archive_added=certified,
             geometry_status=endpoint.geometry_status,
             rejection_reasons=list(endpoint.certification_reasons),
             rgb_present=rgb is not None, latent_present=latent is not None,
             authority=authority, evicted_history_frame_ids=evicted,
             history_frame_ids=[x.frame_id for x in self.history],
             archive_frame_ids=[x.frame_id for x in self.archive])
        return endpoint

    def scaffold_sources(self) -> tuple[EndpointEvidence, ...]:
        """Return Lyra's local, causal RGB-D source for the next target warp.

        The source is the latest available history endpoint, regardless of
        whether it earned a *persistent archive* write.  This evidence is used
        transiently for exactly one subsequent call and cannot poison the
        archive.  Falling back to an old certified/anchor view here caused the
        measured 20% target coverage even when a near-current frame existed.
        """
        return (self.history[-1],)

    def certified_geometry_sources(self) -> tuple[EndpointEvidence, ...]:
        """Persistent geometry used by certification, never transient history."""
        return ((self.anchor,) if self.recent_certified is None
                else (self.anchor, self.recent_certified))

    def archive_candidates(self) -> tuple[EndpointEvidence, ...]:
        return tuple(self.archive)

    def native_retrieval_candidates(
            self, *, include_provisional: bool = False
            ) -> tuple[EndpointEvidence, ...]:
        """Return appearance candidates without expanding persistent authority."""
        if not include_provisional:
            return self.archive_candidates()
        by_frame = {item.frame_id: item for item in self.archive}
        by_frame.update({item.frame_id: item
                         for item in self.provisional_retrieval})
        return tuple(by_frame[key] for key in sorted(by_frame))

    def telemetry(self) -> dict:
        generated = [item for item in self.history if item.content_origin == "generated"]
        certified = [item for item in generated if item.geometry_status == "certified"]
        return {
            "history_occupancy": len(self.history),
            "archive_fill": len(self.archive),
            "provisional_retrieval_fill": len(self.provisional_retrieval),
            "generated_in_history": len(generated),
            "generated_geometry_certified": len(certified),
            "geometry_certification_rate": len(certified) / max(len(generated), 1),
            "recent_certified_frame_id": (
                None if self.recent_certified is None else self.recent_certified.frame_id),
            "local_scaffold_frame_id": self.history[-1].frame_id,
        }
