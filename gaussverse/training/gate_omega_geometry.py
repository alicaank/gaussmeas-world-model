"""Lyra-compatible seed-only MoGe alignment for ΩR deployment geometry.

Lyra estimates the seed depth with DA3, estimates the same RGB with MoGe, fits one
global scalar in inverse-depth space, and divides DA3 depth by that scalar.  MoGe is
not used again.  Subsequent multi-view DA3 calls are aligned through the pipeline
extrinsics and therefore already live in the pipeline's depth scale.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import torch

from gaussverse.training.gate_omega import CONTRACT_ID, DeploymentGauge


GEOMETRY_GAUGE_SCHEMA = "gaussverse.gate_omega_geometry_gauge.v2"
REGISTERED_BRIDGE_SCHEMA = "gaussverse.gate_omega_registered_bridge.v1"


def _canonical(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


@dataclass(frozen=True)
class FrozenDA3MoGeScale:
    """The exact seed-frame inverse-depth alignment used by Lyra."""

    inverse_depth_scale: float
    da3_to_moge: float
    anchor_median: float
    overlap_pixels: int
    gauge_manifest_sha256: str
    moge_weights_sha256: str
    da3_weights_sha256: str
    payload_sha256: str

    @classmethod
    def from_anchor(cls, *, gauge: DeploymentGauge,
                    moge_anchor_depth: torch.Tensor,
                    da3_anchor_depth: torch.Tensor,
                    moge_weights_sha256: str, da3_weights_sha256: str,
                    moge_mask: torch.Tensor | None = None,
                    da3_mask: torch.Tensor | None = None,
                    min_overlap: int = 11) -> "FrozenDA3MoGeScale":
        moge = moge_anchor_depth.detach().float()
        da3 = da3_anchor_depth.detach().float()
        if moge.shape != da3.shape or moge.ndim < 2:
            raise ValueError("DA3 and MoGe anchor depths must have the same image grid")
        valid = (torch.isfinite(moge) & (moge >= 0)
                 & torch.isfinite(da3) & (da3 >= 0))
        if moge_mask is not None:
            if moge_mask.shape != moge.shape:
                raise ValueError("MoGe seed mask differs from its depth grid")
            valid &= moge_mask.detach().to(valid.device) > 0.5
        if da3_mask is not None:
            if da3_mask.shape != da3.shape:
                raise ValueError("DA3 seed mask differs from its depth grid")
            valid &= da3_mask.detach().to(valid.device) > 0.5
        overlap = int(valid.sum())
        if overlap < int(min_overlap):
            raise ValueError("DA3/MoGe anchor alignment has insufficient overlap")
        inverse_da3 = 1.0 / (da3[valid] + 1e-6)
        inverse_moge = 1.0 / (moge[valid] + 1e-6)
        denominator = (inverse_da3 * inverse_da3).sum()
        if not bool(torch.isfinite(denominator)) or float(denominator) <= 1e-8:
            raise ValueError("DA3/MoGe inverse-depth denominator is degenerate")
        inverse_scale = float((inverse_da3 * inverse_moge).sum() / denominator)
        if not torch.isfinite(torch.tensor(inverse_scale)) or inverse_scale <= 1e-6:
            raise ValueError("DA3-to-MoGe anchor scale is invalid")
        ratio = 1.0 / inverse_scale
        aligned_seed = da3 / inverse_scale
        positive = aligned_seed[valid & (aligned_seed > 0)]
        if positive.numel() < 1:
            raise ValueError("aligned DA3 seed has no positive support")
        anchor_median = float(positive.median())
        if abs(anchor_median - float(gauge.scale)) > max(1e-6, 1e-6 * anchor_median):
            raise ValueError(
                "deployment gauge was not derived from the MoGe-aligned DA3 seed depth")
        for name, value in (("moge", moge_weights_sha256),
                            ("da3", da3_weights_sha256)):
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{name} weights require one SHA-256 digest")
        body = {
            "schema": GEOMETRY_GAUGE_SCHEMA, "contract_id": CONTRACT_ID,
            "policy": "lyra_seed_inverse_depth_ls_then_recurrent_da3_camera_alignment",
            "inverse_depth_scale": inverse_scale,
            "da3_to_moge": ratio, "anchor_median": anchor_median,
            "overlap_pixels": overlap,
            "gauge_manifest_sha256": gauge.manifest()["payload_sha256"],
            "moge_weights_sha256": moge_weights_sha256,
            "da3_weights_sha256": da3_weights_sha256,
            "moge_usage": "seed_rgb_only",
            "recurrent_da3_input_extrinsics": True,
            "recurrent_da3_align_to_input_extrinsics": True,
            "recurrent_da3_align_to_input_ext_scale": True,
            "later_moge_refit": False,
        }
        digest = hashlib.sha256(_canonical(body)).hexdigest()
        return cls(inverse_scale, ratio, anchor_median, overlap,
                   body["gauge_manifest_sha256"], moge_weights_sha256,
                   da3_weights_sha256, digest)

    @staticmethod
    def align_seed_depth(*, moge_anchor_depth: torch.Tensor,
                         da3_anchor_depth: torch.Tensor,
                         moge_mask: torch.Tensor | None = None,
                         da3_mask: torch.Tensor | None = None,
                         min_overlap: int = 11) -> torch.Tensor:
        """Apply Lyra's scalar fit before the deployment gauge is constructed."""
        moge = moge_anchor_depth.detach().float()
        da3 = da3_anchor_depth.detach().float()
        if moge.shape != da3.shape or moge.ndim < 2:
            raise ValueError("DA3 and MoGe anchor depths must have the same image grid")
        valid = (torch.isfinite(moge) & (moge >= 0)
                 & torch.isfinite(da3) & (da3 >= 0))
        for name, mask in (("MoGe", moge_mask), ("DA3", da3_mask)):
            if mask is not None:
                if mask.shape != da3.shape:
                    raise ValueError(f"{name} seed mask differs from its depth grid")
                valid &= mask.detach().to(valid.device) > 0.5
        if int(valid.sum()) < int(min_overlap):
            raise ValueError("DA3/MoGe anchor alignment has insufficient overlap")
        inverse_da3 = 1.0 / (da3[valid] + 1e-6)
        inverse_moge = 1.0 / (moge[valid] + 1e-6)
        denominator = (inverse_da3 * inverse_da3).sum()
        if not bool(torch.isfinite(denominator)) or float(denominator) <= 1e-8:
            raise ValueError("DA3/MoGe inverse-depth denominator is degenerate")
        inverse_scale = (inverse_da3 * inverse_moge).sum() / denominator
        if not bool(torch.isfinite(inverse_scale)) or float(inverse_scale) <= 1e-6:
            raise ValueError("DA3-to-MoGe anchor scale is invalid")
        return da3 / inverse_scale

    def normalize_da3_depth(self, depth: torch.Tensor) -> torch.Tensor:
        """Normalize a raw single-image DA3 seed after Lyra's MoGe fit."""
        value = depth.detach().float()
        valid = torch.isfinite(value) & (value > 0)
        result = torch.zeros_like(value)
        result[valid] = value[valid] * self.da3_to_moge / self.anchor_median
        return result

    @classmethod
    def from_manifest(cls, value: dict) -> "FrozenDA3MoGeScale":
        """Restore a frozen seed scalar only from its verified exact manifest."""
        recorded = value.get("payload_sha256")
        body = {key: item for key, item in value.items()
                if key != "payload_sha256"}
        if hashlib.sha256(_canonical(body)).hexdigest() != recorded:
            raise ValueError("frozen DA3/MoGe scale manifest digest is stale")
        if (value.get("schema") != GEOMETRY_GAUGE_SCHEMA
                or value.get("contract_id") != CONTRACT_ID
                or value.get("moge_usage") != "seed_rgb_only"
                or value.get("later_moge_refit") is not False
                or value.get("recurrent_da3_input_extrinsics") is not True
                or value.get("recurrent_da3_align_to_input_extrinsics") is not True
                or value.get("recurrent_da3_align_to_input_ext_scale") is not True):
            raise ValueError("frozen DA3/MoGe scale manifest changes Lyra semantics")
        result = cls(
            float(value["inverse_depth_scale"]), float(value["da3_to_moge"]),
            float(value["anchor_median"]), int(value["overlap_pixels"]),
            str(value["gauge_manifest_sha256"]),
            str(value["moge_weights_sha256"]),
            str(value["da3_weights_sha256"]), str(recorded))
        if result.manifest() != value:
            raise ValueError("frozen DA3/MoGe scale manifest is not canonical")
        return result

    def normalize_aligned_da3_depth(self, depth: torch.Tensor) -> torch.Tensor:
        """Normalize recurrent DA3 already aligned to Lyra's pipeline scale."""
        value = depth.detach().float()
        valid = torch.isfinite(value) & (value > 0)
        result = torch.zeros_like(value)
        result[valid] = value[valid] / self.anchor_median
        return result

    def manifest(self) -> dict:
        body = {
            "schema": GEOMETRY_GAUGE_SCHEMA, "contract_id": CONTRACT_ID,
            "policy": "lyra_seed_inverse_depth_ls_then_recurrent_da3_camera_alignment",
            "inverse_depth_scale": self.inverse_depth_scale,
            "da3_to_moge": self.da3_to_moge,
            "anchor_median": self.anchor_median,
            "overlap_pixels": self.overlap_pixels,
            "gauge_manifest_sha256": self.gauge_manifest_sha256,
            "moge_weights_sha256": self.moge_weights_sha256,
            "da3_weights_sha256": self.da3_weights_sha256,
            "moge_usage": "seed_rgb_only",
            "recurrent_da3_input_extrinsics": True,
            "recurrent_da3_align_to_input_extrinsics": True,
            "recurrent_da3_align_to_input_ext_scale": True,
            "later_moge_refit": False,
        }
        body["payload_sha256"] = self.payload_sha256
        if hashlib.sha256(_canonical({k: v for k, v in body.items()
                                      if k != "payload_sha256"})).hexdigest() \
                != self.payload_sha256:
            raise RuntimeError("frozen DA3/MoGe gauge digest is stale")
        return body


@dataclass(frozen=True)
class FrozenRegisteredTrajectoryBridge:
    """One causal similarity from registered camera units to the seed gauge.

    Lyra's generated trajectories are constructed in the DA3/MoGe seed-depth
    scale.  Registered training videos instead arrive in an arbitrary SfM
    scale.  A recurrent DA3 call on the *legal observed prefix* exposes that
    registered scale at the anchor; fitting it once to the already-frozen seed
    depth makes the registered poses deployment-equivalent without another
    MoGe call or any target-frame conditioning.
    """

    scale_to_seed: float
    overlap_pixels: int
    anchor_origin: torch.Tensor
    gauge_manifest_sha256: str
    seed_scale_sha256: str
    payload_sha256: str

    @classmethod
    def from_causal_anchor(cls, *, gauge: DeploymentGauge,
                           seed_scale: FrozenDA3MoGeScale,
                           seed_aligned_depth: torch.Tensor,
                           registered_da3_anchor_depth: torch.Tensor,
                           seed_mask: torch.Tensor | None = None,
                           registered_mask: torch.Tensor | None = None,
                           min_overlap: int = 11,
                           anchor_c2w: torch.Tensor) \
            -> "FrozenRegisteredTrajectoryBridge":
        if seed_scale.gauge_manifest_sha256 != gauge.manifest()["payload_sha256"]:
            raise ValueError("seed scale belongs to another deployment gauge")
        seed = seed_aligned_depth.detach().float()
        registered = registered_da3_anchor_depth.detach().float()
        if seed.shape != registered.shape or seed.ndim < 2:
            raise ValueError("seed and registered DA3 depths must share one grid")
        valid = (torch.isfinite(seed) & (seed > 0)
                 & torch.isfinite(registered) & (registered > 0))
        for name, mask in (("seed", seed_mask), ("registered", registered_mask)):
            if mask is not None:
                if mask.shape != seed.shape:
                    raise ValueError(f"{name} bridge mask differs from depth grid")
                valid &= mask.detach().to(valid.device) > 0.5
        overlap = int(valid.sum())
        if overlap < int(min_overlap):
            raise ValueError("registered trajectory bridge has insufficient overlap")
        inverse_registered = 1.0 / (registered[valid] + 1e-6)
        inverse_seed = 1.0 / (seed[valid] + 1e-6)
        denominator = inverse_registered.square().sum()
        if not bool(torch.isfinite(denominator)) or float(denominator) <= 1e-8:
            raise ValueError("registered trajectory bridge is degenerate")
        inverse_scale = (inverse_registered * inverse_seed).sum() / denominator
        scale_to_seed = float(1.0 / inverse_scale)
        if not torch.isfinite(torch.tensor(scale_to_seed)) or scale_to_seed <= 1e-6:
            raise ValueError("registered trajectory bridge scale is invalid")
        if anchor_c2w.shape != (4, 4) or not bool(torch.isfinite(anchor_c2w).all()):
            raise ValueError("registered trajectory bridge needs a finite anchor pose")
        origin = anchor_c2w[:3, 3].detach().double().cpu()
        if not torch.allclose(origin, gauge.anchor_origin.detach().double().cpu(),
                              atol=1e-7, rtol=1e-7):
            raise ValueError("registered bridge anchor differs from deployment gauge")
        body = {
            "schema": REGISTERED_BRIDGE_SCHEMA,
            "contract_id": CONTRACT_ID,
            "policy": "causal_prefix_recurrent_da3_anchor_to_frozen_seed_similarity",
            "scale_to_seed": scale_to_seed,
            "overlap_pixels": overlap,
            "anchor_origin": origin.tolist(),
            "gauge_manifest_sha256": gauge.manifest()["payload_sha256"],
            "seed_scale_sha256": seed_scale.payload_sha256,
            "moge_usage": "seed_rgb_only",
            "target_rgb_used": False,
        }
        digest = hashlib.sha256(_canonical(body)).hexdigest()
        return cls(scale_to_seed, overlap, origin, body["gauge_manifest_sha256"],
                   seed_scale.payload_sha256, digest)

    def canonicalize_c2w(self, c2w_registered: torch.Tensor) -> torch.Tensor:
        """Apply the frozen similarity about the immutable anchor origin."""
        if c2w_registered.shape[-2:] != (4, 4):
            raise ValueError("registered camera poses must end in 4x4")
        pose = c2w_registered.detach().to(torch.float64)
        if not bool(torch.isfinite(pose).all()):
            raise ValueError("registered camera poses must be finite")
        origin = self.anchor_origin.to(pose.device, pose.dtype)
        out = pose.clone()
        out[..., :3, 3] = origin + self.scale_to_seed * (
            pose[..., :3, 3] - origin)
        return out.to(c2w_registered.dtype)

    @classmethod
    def from_manifest(cls, value: dict) -> "FrozenRegisteredTrajectoryBridge":
        """Restore the exact registered-data bridge used to prepare cached rows."""
        recorded = value.get("payload_sha256")
        body = {key: item for key, item in value.items()
                if key != "payload_sha256"}
        if hashlib.sha256(_canonical(body)).hexdigest() != recorded:
            raise ValueError("registered trajectory bridge manifest digest is stale")
        if (value.get("schema") != REGISTERED_BRIDGE_SCHEMA
                or value.get("contract_id") != CONTRACT_ID
                or value.get("moge_usage") != "seed_rgb_only"
                or value.get("target_rgb_used") is not False):
            raise ValueError("registered trajectory bridge manifest changes policy")
        result = cls(
            float(value["scale_to_seed"]), int(value["overlap_pixels"]),
            torch.tensor(value["anchor_origin"], dtype=torch.float64),
            str(value["gauge_manifest_sha256"]),
            str(value["seed_scale_sha256"]), str(recorded))
        if result.anchor_origin.shape != (3,) or result.scale_to_seed <= 0:
            raise ValueError("registered trajectory bridge manifest is invalid")
        if result.manifest() != value:
            raise ValueError("registered trajectory bridge manifest is not canonical")
        return result

    def canonicalize_depth(self, depth_registered: torch.Tensor) -> torch.Tensor:
        value = depth_registered.detach().float()
        valid = torch.isfinite(value) & (value > 0)
        result = torch.zeros_like(value)
        result[valid] = value[valid] * self.scale_to_seed
        return result

    def manifest(self) -> dict:
        body = {
            "schema": REGISTERED_BRIDGE_SCHEMA,
            "contract_id": CONTRACT_ID,
            "policy": "causal_prefix_recurrent_da3_anchor_to_frozen_seed_similarity",
            "scale_to_seed": self.scale_to_seed,
            "overlap_pixels": self.overlap_pixels,
            "anchor_origin": self.anchor_origin.tolist(),
            "gauge_manifest_sha256": self.gauge_manifest_sha256,
            "seed_scale_sha256": self.seed_scale_sha256,
            "moge_usage": "seed_rgb_only",
            "target_rgb_used": False,
        }
        body["payload_sha256"] = self.payload_sha256
        if hashlib.sha256(_canonical({k: v for k, v in body.items()
                                      if k != "payload_sha256"})).hexdigest() \
                != self.payload_sha256:
            raise RuntimeError("registered trajectory bridge digest is stale")
        return body
