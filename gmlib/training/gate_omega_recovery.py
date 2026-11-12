"""Lyra-compatible MoGe/DA3 geometry with ΩR certification telemetry.

The seed RGB is estimated once by DA3 and MoGe and aligned with Lyra's inverse-depth
least-squares scalar.  Every recurrent DA3 window receives causal RGB plus pipeline
extrinsics/intrinsics with both DA3 alignment switches enabled.  MoGe never runs after
the seed.  ΩR retains its independent feature-track and cross-view consistency checks
around those Lyra geometry mechanics.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import sys
from typing import Callable, Protocol

import numpy as np
import torch
import torch.nn.functional as F

from gmlib.models.gate_omega_scaffold import render_point_source
from gmlib.training.gate_omega import DeploymentGauge
from gmlib.training.gate_omega_geometry import (
    FrozenDA3MoGeScale,
    FrozenRegisteredTrajectoryBridge,
)
from gmlib.training.gate_omega_handoff import GeometryRecovery, PoseRecovery


DA3_SOURCE = Path(os.environ.get(
    "GM_DA3_SRC",
    Path(__file__).resolve().parents[2] / "lyra" / "Lyra-2" /
    "lyra_2" / "_src" / "inference" / "depth_anything_3" / "src")).resolve()
DA3_MODEL = Path(os.environ.get(
    "GM_DA3_MODEL",
    Path.home() / ".cache" / "huggingface" / "hub" /
    "models--depth-anything--DA3NESTED-GIANT-LARGE-1.1" / "snapshots" /
    "b2359bdf726fb44ef62acca04d629dcf158053e7")).resolve()
MOGE_MODEL = Path(os.environ.get(
    "GM_MOGE_MODEL",
    Path.home() / ".cache" / "huggingface" / "hub" /
    "models--Ruicheng--moge-vitl" / "snapshots" /
    "ad326bfb61facd6c52b5a825bc1e34d7c97d9672" / "model.pt")).resolve()


@dataclass(frozen=True)
class CertifiedGeometryView:
    frame_id: int
    rgb: torch.Tensor
    depth: torch.Tensor
    confidence: torch.Tensor
    c2w: torch.Tensor
    intrinsics: torch.Tensor


@dataclass(frozen=True)
class PoseReferenceView:
    """Causal RGB/pose reference for the selected pose policy.

    Released-Lyra recurrence begins with the single observed anchor, then advances
    this window with generated RGB and commanded pipeline poses.  The legacy VIPE
    evaluation policy requires four references to fix its similarity gauge.
    """
    frame_id: int
    rgb: torch.Tensor
    c2w: torch.Tensor
    intrinsics: torch.Tensor


@dataclass(frozen=True)
class DA3Prediction:
    depths: torch.Tensor
    predicted_c2w: torch.Tensor
    valid_masks: torch.Tensor | None = None


@dataclass(frozen=True)
class SeedDepthPrediction:
    depth: torch.Tensor
    intrinsics: torch.Tensor | None
    mask: torch.Tensor


@dataclass(frozen=True)
class TrackConfirmation:
    valid: bool
    translation_disagreement: float
    rotation_disagreement_deg: float
    reprojection_error_px: float
    inliers: int
    mode: str = "essential"


class DA3Predictor(Protocol):
    def __call__(self, images_uint8: np.ndarray,
                 intrinsics: np.ndarray,
                 extrinsics_w2c: np.ndarray) -> DA3Prediction: ...


def _to_c2w(extrinsics: np.ndarray) -> torch.Tensor:
    w2c = torch.from_numpy(np.asarray(extrinsics)).double()
    if w2c.ndim != 3 or w2c.shape[-2:] not in {(3, 4), (4, 4)}:
        raise ValueError("DA3 extrinsics must be Nx3x4 or Nx4x4 w2c")
    if w2c.shape[-2:] == (3, 4):
        padded = torch.zeros((w2c.shape[0], 4, 4), dtype=torch.float64)
        padded[:, :3] = w2c
        padded[:, 3, 3] = 1
        w2c = padded
    return torch.linalg.inv(w2c).float()


def _pad_depth(value: np.ndarray, size: tuple[int, int]) -> torch.Tensor:
    depth = torch.from_numpy(np.asarray(value)).float()
    height, width = size
    if depth.shape == (height, width):
        return depth
    if depth.ndim != 2 or depth.shape[1] != width or depth.shape[0] > height:
        raise ValueError(f"DA3 depth {tuple(depth.shape)} cannot align to {size}")
    missing = height - depth.shape[0]
    return F.pad(depth[None, None],
                 (0, 0, missing // 2, missing - missing // 2),
                 mode="replicate")[0, 0]


def _resize_mask(value: np.ndarray, size: tuple[int, int]) -> torch.Tensor:
    mask = torch.from_numpy(np.asarray(value)).float()
    if mask.ndim != 2:
        raise ValueError("DA3 sky mask must be two-dimensional")
    if tuple(mask.shape) != size:
        mask = F.interpolate(mask[None, None], size=size, mode="nearest")[0, 0]
    return mask


def _lyra_alignment_offsets(extrinsics_w2c: np.ndarray, *,
                            variance_threshold: float = 1e-10,
                            offset_scale: float = 1e-6,
                            offset_freq_x: float = 0.00001,
                            offset_freq_y: float = 0.000013,
                            offset_freq_z: float = 0.000007) -> np.ndarray:
    """Copy Lyra's DA3 alignment guard for nearly identical camera centers."""
    extrinsics = np.asarray(extrinsics_w2c, dtype=np.float32)
    centers = np.stack([-ext[:3, :3].T @ ext[:3, 3] for ext in extrinsics])
    if float(np.var(centers, axis=0).sum()) >= variance_threshold:
        return extrinsics
    modified = extrinsics.copy()
    for index, ext in enumerate(modified):
        offset = np.array([
            offset_scale * np.sin(2 * np.pi * index * offset_freq_x),
            offset_scale * np.cos(2 * np.pi * index * offset_freq_y),
            offset_scale * np.sin(2 * np.pi * index * offset_freq_z),
        ])
        modified[index, :3, 3] = ext[:3, 3] - ext[:3, :3] @ offset
    return modified


class PinnedDA3Predictor:
    """Lazy pinned Lyra DA3NESTED-GIANT-LARGE-1.1 inference."""

    def __init__(self, device: str = "cuda", *, offload_after_call: bool = False):
        self.device = device
        self.offload_after_call = bool(offload_after_call)
        self._model = None

    def _load(self):
        if self._model is None:
            if not DA3_SOURCE.is_dir() or not DA3_MODEL.is_dir():
                raise FileNotFoundError("pinned DA3 source/model is not installed")
            if str(DA3_SOURCE) not in sys.path:
                sys.path.insert(0, str(DA3_SOURCE))
            from depth_anything_3.api import DepthAnything3
            self._model = DepthAnything3.from_pretrained(str(DA3_MODEL)).to(
                self.device).eval()
        if hasattr(self._model, "to"):
            self._model = self._model.to(self.device)
        if hasattr(self._model, "eval"):
            self._model = self._model.eval()
        return self._model

    def offload(self) -> None:
        if self._model is not None and hasattr(self._model, "cpu"):
            import gc
            self._model.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    @torch.no_grad()
    def __call__(self, images_uint8: np.ndarray,
                 intrinsics: np.ndarray,
                 extrinsics_w2c: np.ndarray) -> DA3Prediction:
        if images_uint8.ndim != 4 or images_uint8.shape[-1] != 3:
            raise ValueError("DA3 images must be NHWC uint8")
        if intrinsics.shape != (len(images_uint8), 3, 3):
            raise ValueError("DA3 intrinsics must be Nx3x3")
        if extrinsics_w2c.shape != (len(images_uint8), 4, 4):
            raise ValueError("DA3 pipeline extrinsics must be Nx4x4 w2c")
        try:
            prediction = self._load().inference(
                image=list(images_uint8), extrinsics=extrinsics_w2c.astype(np.float32),
                intrinsics=intrinsics.astype(np.float32),
                align_to_input_extrinsics=True, align_to_input_ext_scale=True,
                infer_gs=False, process_res=max(images_uint8.shape[1:3]),
                process_res_method="upper_bound_resize",
                reorder_cam_token_by_reference=True, export_dir=None,
                export_format="mini_npz")
            size = tuple(images_uint8.shape[1:3])
            depths = torch.stack([_pad_depth(value, size)
                                  for value in prediction.depth])
            sky = getattr(prediction, "sky", None)
            valid_masks = None if sky is None else torch.stack([
                1.0 - _resize_mask(value, size).clamp(0, 1) for value in sky])
            return DA3Prediction(
                depths, _to_c2w(prediction.extrinsics), valid_masks)
        finally:
            if self.offload_after_call:
                self.offload()

    @torch.no_grad()
    def seed(self, image_uint8: np.ndarray,
             target_size: tuple[int, int]) -> SeedDepthPrediction:
        """Run Lyra's single-image DA3 seed path without input cameras."""
        import cv2
        height, width = map(int, target_size)
        image = cv2.resize(np.asarray(image_uint8), (width, height),
                           interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        try:
            prediction = self._load().inference(
                image=[image], extrinsics=None, intrinsics=None,
                align_to_input_ext_scale=True, infer_gs=False,
                process_res=max(height, width),
                process_res_method="upper_bound_resize", export_dir=None,
                export_format="mini_npz")
            raw = torch.as_tensor(np.asarray(prediction.depth)[0]).float()
            source_size = tuple(raw.shape[-2:])
            depth = F.interpolate(
                raw[None, None], (height, width), mode="bilinear",
                align_corners=False)[0, 0]
            depth = torch.nan_to_num(depth, nan=1e4).clamp(0, 1e4)
            mask = (depth < 999.9).float()
            intrinsics = getattr(prediction, "intrinsics", None)
            K = None
            if intrinsics is not None:
                K = torch.as_tensor(np.asarray(intrinsics)[0]).float().clone()
                K[0, 0] *= width / float(source_size[1])
                K[0, 2] *= width / float(source_size[1])
                K[1, 1] *= height / float(source_size[0])
                K[1, 2] *= height / float(source_size[0])
            return SeedDepthPrediction(depth, K, mask)
        finally:
            if self.offload_after_call:
                self.offload()


class PinnedMoGePredictor:
    """Lazy seed-only MoGe-v1 predictor matching Lyra's preprocessing."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            if not MOGE_MODEL.is_file():
                raise FileNotFoundError("pinned MoGe-v1 model is not installed")
            import os
            os.environ["XFORMERS_DISABLED"] = "1"
            from moge.model.v1 import MoGeModel
            # Lyra also flips any already-imported MoGe backend switches after
            # import; the environment flag alone is not equivalent in a warm
            # Python process.
            for name, module in tuple(sys.modules.items()):
                if "moge" in name and hasattr(module, "XFORMERS_ENABLED"):
                    module.XFORMERS_ENABLED = False
            self._model = MoGeModel.from_pretrained(str(MOGE_MODEL)).to(
                self.device).eval()
        return self._model

    @torch.no_grad()
    def seed(self, image_uint8: np.ndarray,
             target_size: tuple[int, int]) -> SeedDepthPrediction:
        import cv2
        import gc
        height, width = map(int, target_size)
        image = cv2.resize(np.asarray(image_uint8), (width, height),
                           interpolation=cv2.INTER_LINEAR)
        model = self._load().to(self.device).eval()
        tensor = torch.tensor(image / 255.0, dtype=torch.float32,
                              device=self.device).permute(2, 0, 1)
        try:
            # This is Lyra's explicit seed-inference attention policy, not a
            # numerically looser accelerator-specific dispatch.
            with torch.nn.attention.sdpa_kernel(
                    [torch.nn.attention.SDPBackend.MATH]):
                output = model.infer(tensor)
            depth = torch.nan_to_num(output["depth"], nan=1e4).clamp(0, 1e4)
            mask = output["mask"]
            depth = torch.where(mask == 0, torch.tensor(
                1000.0, device=depth.device, dtype=depth.dtype), depth)
            normalized_K = output["intrinsics"].clone()
            normalized_K[0, 0] *= width
            normalized_K[0, 2] *= width
            normalized_K[1, 1] *= height
            normalized_K[1, 2] *= height
            result = SeedDepthPrediction(
                depth.cpu(), normalized_K.cpu(), mask.float().cpu())
        finally:
            # Lyra releases seed-only MoGe before recurrent generation. Keeping
            # it resident would waste production VRAM and blur its one-shot role.
            model.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        return result


@dataclass(frozen=True)
class LyraSeedGeometry:
    """Content needed to initialize ΩR in Lyra's seed depth scale."""

    aligned_depth: torch.Tensor
    intrinsics: torch.Tensor
    mask: torch.Tensor
    gauge: DeploymentGauge
    scale: FrozenDA3MoGeScale


@dataclass(frozen=True)
class RegisteredTrajectoryCalibration:
    """Causal registered-data conversion into Lyra's deployment gauge."""

    bridge: FrozenRegisteredTrajectoryBridge
    canonical_c2w: torch.Tensor
    normalized_depths: torch.Tensor
    confidences: torch.Tensor
    estimator_frame_ids: tuple[int, ...]
    calibration_id: str


def build_lyra_seed_geometry(*, image_uint8: np.ndarray,
                             anchor_c2w: torch.Tensor,
                             target_size: tuple[int, int],
                             da3_predictor,
                             moge_predictor,
                             da3_weights_sha256: str,
                             moge_weights_sha256: str) -> LyraSeedGeometry:
    """Run each seed estimator once and freeze Lyra's inverse-depth scalar."""
    da3 = da3_predictor.seed(image_uint8, target_size)
    moge = moge_predictor.seed(image_uint8, target_size)
    if da3.intrinsics is None:
        height, width = target_size
        focal = max(height, width) * 1.5
        intrinsics = torch.tensor(
            [[focal, 0.0, width / 2], [0.0, focal, height / 2],
             [0.0, 0.0, 1.0]], dtype=torch.float32)
    else:
        intrinsics = da3.intrinsics.detach().float().cpu()
    aligned = FrozenDA3MoGeScale.align_seed_depth(
        moge_anchor_depth=moge.depth, da3_anchor_depth=da3.depth,
        moge_mask=moge.mask, da3_mask=da3.mask)
    common_mask = (da3.mask > 0.5) & (moge.mask > 0.5)
    gauge = DeploymentGauge.from_anchor(
        anchor_c2w, torch.where(common_mask, aligned, torch.zeros_like(aligned)))
    scale = FrozenDA3MoGeScale.from_anchor(
        gauge=gauge, moge_anchor_depth=moge.depth,
        da3_anchor_depth=da3.depth, moge_mask=moge.mask,
        da3_mask=da3.mask, moge_weights_sha256=moge_weights_sha256,
        da3_weights_sha256=da3_weights_sha256)
    mask = common_mask.float()
    return LyraSeedGeometry(aligned, intrinsics, mask, gauge, scale)


@torch.no_grad()
def calibrate_registered_trajectory_prefix(
        *, images_uint8: np.ndarray, prefix_frame_ids: tuple[int, ...],
        history_boundary: int, registered_c2w: torch.Tensor,
        intrinsics: torch.Tensor, seed: LyraSeedGeometry,
        predictor: DA3Predictor, min_overlap: int = 11,
        anchor_frame_id: int | None = None,
        ) -> RegisteredTrajectoryCalibration:
    """Convert an arbitrary SfM prefix to the frozen seed gauge, once.

    This is an offline registered-data adapter, not a deployment estimator
    change.  Every RGB supplied here must already belong to the legal observed
    prefix.  Deployment trajectories are constructed in seed-depth units and
    therefore bypass this adapter.
    """
    if (images_uint8.ndim != 4 or images_uint8.shape[-1] != 3
            or len(images_uint8) < 2):
        raise ValueError("registered calibration needs at least two RGB frames")
    if len(prefix_frame_ids) != len(images_uint8):
        raise ValueError("registered calibration frame IDs differ from RGB")
    if tuple(sorted(set(map(int, prefix_frame_ids)))) != prefix_frame_ids:
        raise ValueError("registered calibration frame IDs must be strictly ordered")
    if any(int(value) > int(history_boundary) for value in prefix_frame_ids):
        raise ValueError("registered calibration contains a future frame")
    anchor_frame_id = (int(prefix_frame_ids[0]) if anchor_frame_id is None
                       else int(anchor_frame_id))
    if anchor_frame_id not in prefix_frame_ids:
        raise ValueError("registered calibration omits its seed anchor frame")
    anchor_index = prefix_frame_ids.index(anchor_frame_id)
    if (registered_c2w.shape != (len(images_uint8), 4, 4)
            or not bool(torch.isfinite(registered_c2w).all())):
        raise ValueError("registered calibration poses must be finite Nx4x4")
    K = (intrinsics[None].expand(len(images_uint8), -1, -1)
         if intrinsics.shape == (3, 3) else intrinsics)
    if K.shape != (len(images_uint8), 3, 3) or not bool(torch.isfinite(K).all()):
        raise ValueError("registered calibration intrinsics must be finite Nx3x3")
    w2c = torch.linalg.inv(registered_c2w.detach().double()).float()
    da3_w2c = _lyra_alignment_offsets(w2c.numpy())
    prediction = predictor(images_uint8, K.detach().float().cpu().numpy(), da3_w2c)
    height, width = images_uint8.shape[1:3]
    if (prediction.depths.shape != (len(images_uint8), height, width)
            or prediction.predicted_c2w.shape != (len(images_uint8), 4, 4)
            or not bool(torch.isfinite(prediction.depths).all())
            or not bool(torch.isfinite(prediction.predicted_c2w).all())
            or (prediction.valid_masks is not None
                and (prediction.valid_masks.shape != prediction.depths.shape
                     or not bool(torch.isfinite(prediction.valid_masks).all())))):
        raise RuntimeError("DA3 returned invalid registered-prefix calibration")
    registered_mask = (None if prediction.valid_masks is None
                       else prediction.valid_masks[anchor_index].float())
    bridge = FrozenRegisteredTrajectoryBridge.from_causal_anchor(
        gauge=seed.gauge, seed_scale=seed.scale,
        seed_aligned_depth=seed.aligned_depth,
        registered_da3_anchor_depth=prediction.depths[anchor_index],
        seed_mask=seed.mask, registered_mask=registered_mask,
        min_overlap=min_overlap, anchor_c2w=registered_c2w[anchor_index])
    canonical_c2w = bridge.canonicalize_c2w(registered_c2w)
    normalized_depths = torch.stack([
        seed.gauge.normalize_depth(bridge.canonicalize_depth(value)).float()
        for value in prediction.depths])
    confidences = torch.stack([_reliability(value) for value in normalized_depths])
    if prediction.valid_masks is not None:
        confidences *= prediction.valid_masks.detach().float().cpu()
    material = hashlib.sha256()
    material.update(images_uint8.tobytes())
    material.update(np.asarray(prefix_frame_ids, dtype=np.int64).tobytes())
    material.update(da3_w2c.tobytes())
    material.update(seed.scale.payload_sha256.encode())
    material.update(bridge.payload_sha256.encode())
    return RegisteredTrajectoryCalibration(
        bridge, canonical_c2w, normalized_depths, confidences,
        prefix_frame_ids, material.hexdigest())


def _as_frames(pixels: torch.Tensor) -> torch.Tensor:
    value = pixels.detach().float().cpu()
    if value.ndim == 3 and value.shape[0] == 3:
        value = value[None]
    if value.ndim == 5:
        if value.shape[0] != 1:
            raise ValueError("generated pixels must have batch size one")
        value = value[0]
    if value.ndim != 4:
        raise ValueError("generated pixels must be Tx3xHxW or 1x3xTxHxW")
    if value.shape[1] == 3:
        frames = value
    elif value.shape[0] == 3:
        frames = value.permute(1, 0, 2, 3)
    else:
        raise ValueError("generated pixels have no RGB channel dimension")
    if not bool(torch.isfinite(frames).all()):
        raise ValueError("generated pixels contain non-finite values")
    return frames


def _to_uint8(frames: torch.Tensor) -> np.ndarray:
    # Deployment decoder tensors use [-1,1]. Accept [0,1] only when all values
    # are nonnegative, preserving an unambiguous conversion.
    if float(frames.min()) >= 0 and float(frames.max()) <= 1:
        value = frames * 255
    else:
        value = (frames.clamp(-1, 1) + 1) * 127.5
    return value.round().byte().permute(0, 2, 3, 1).numpy()


def advance_pose_reference_window(
        references: tuple[PoseReferenceView, ...], *,
        pixels: torch.Tensor, frame_ids: tuple[int, ...],
        realized_c2w: torch.Tensor, intrinsics: torch.Tensor,
        capacity: int = 12) -> tuple[PoseReferenceView, ...]:
    """Advance the causal VIPE reference window after one finite handoff."""
    frames = _as_frames(pixels)
    poses = realized_c2w.detach().float().cpu()
    if (capacity < 1 or len(frames) != len(frame_ids)
            or poses.shape != (len(frame_ids), 4, 4)
            or tuple(sorted(set(frame_ids))) != frame_ids
            or not bool(torch.isfinite(poses).all())):
        raise ValueError("cannot advance VIPE references from an invalid handoff")
    Ks = (intrinsics.detach().float().cpu()[None].expand(len(frame_ids), -1, -1)
          if intrinsics.shape == (3, 3)
          else intrinsics.detach().float().cpu())
    if (Ks.shape != (len(frame_ids), 3, 3)
            or not bool(torch.isfinite(Ks).all())):
        raise ValueError("VIPE reference intrinsics do not match the handoff")
    appended = tuple(PoseReferenceView(
        int(frame_id), frames[index].clone(), poses[index].clone(),
        Ks[index].clone()) for index, frame_id in enumerate(frame_ids))
    result = (*references, *appended)[-capacity:]
    if not result or len({item.frame_id for item in result}) != len(result):
        raise RuntimeError("advanced VIPE reference window is degenerate")
    return tuple(result)


def _rotation_error_deg(first: torch.Tensor, second: torch.Tensor) -> float:
    cosine = ((torch.trace(first.double().T @ second.double()) - 1) / 2).clamp(-1, 1)
    return float(torch.rad2deg(torch.acos(cosine)))


def _effective_focal(first_intrinsics: np.ndarray,
                     second_intrinsics: np.ndarray) -> float:
    """Return the geometric mean focal length across both calibrated views."""
    product = (float(first_intrinsics[0, 0]) * float(first_intrinsics[1, 1])
               * float(second_intrinsics[0, 0])
               * float(second_intrinsics[1, 1]))
    return max(product, 1e-12) ** 0.25


class OpenCVFeatureTrackConfirmation:
    """Independent SIFT + essential-matrix/RANSAC confirmation."""

    def __init__(self, *, min_inliers: int = 8, ratio: float = 0.75,
                 ransac_px: float = 1.5,
                 pure_rotation_translation: float = 1e-6):
        self.min_inliers = int(min_inliers)
        self.ratio = float(ratio)
        self.ransac_px = float(ransac_px)
        self.pure_rotation_translation = float(pure_rotation_translation)
        if self.min_inliers < 8:
            raise ValueError("feature confirmation requires at least eight inliers")
        if (not 0 < self.ratio < 1 or self.ransac_px <= 0
                or self.pure_rotation_translation < 0):
            raise ValueError("feature confirmation thresholds are invalid")

    def __call__(self, first: np.ndarray, second: np.ndarray,
                 first_intrinsics: np.ndarray, second_intrinsics: np.ndarray,
                 da3_first_c2w: torch.Tensor,
                 da3_second_c2w: torch.Tensor) -> TrackConfirmation:
        import cv2
        gray_a = cv2.cvtColor(first, cv2.COLOR_RGB2GRAY)
        gray_b = cv2.cvtColor(second, cv2.COLOR_RGB2GRAY)
        sift = cv2.SIFT_create()
        key_a, desc_a = sift.detectAndCompute(gray_a, None)
        key_b, desc_b = sift.detectAndCompute(gray_b, None)
        if desc_a is None or desc_b is None:
            return TrackConfirmation(False, float("inf"), float("inf"),
                                     float("inf"), 0)
        matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_a, desc_b, k=2)
        good = [first_match for first_match, second_match in matches
                if first_match.distance < self.ratio * second_match.distance]
        if len(good) < max(8, self.min_inliers):
            return TrackConfirmation(False, float("inf"), float("inf"),
                                     float("inf"), len(good))
        points_a = np.float64([key_a[item.queryIdx].pt for item in good])
        points_b = np.float64([key_b[item.trainIdx].pt for item in good])
        normalized_a = cv2.undistortPoints(
            points_a[:, None], first_intrinsics, None)[:, 0]
        normalized_b = cv2.undistortPoints(
            points_b[:, None], second_intrinsics, None)[:, 0]
        da3_w2c_a = torch.linalg.inv(da3_first_c2w.double())
        da3_w2c_b = torch.linalg.inv(da3_second_c2w.double())
        relative = da3_w2c_b @ da3_first_c2w.double()
        da3_rotation = relative[:3, :3]
        da3_translation = relative[:3, 3]

        # Essential geometry is singular for a true rotation-only camera move.
        # Use an independent calibrated homography rotation estimate in exactly
        # that commanded regime; otherwise retain the stricter essential branch.
        if float(da3_translation.norm()) <= self.pure_rotation_translation:
            homography, mask = cv2.findHomography(
                points_a, points_b, cv2.RANSAC, self.ransac_px,
                confidence=0.999)
            if homography is None or mask is None:
                return TrackConfirmation(False, float("inf"), float("inf"),
                                         float("inf"), 0, "rotation_homography")
            inlier = mask.reshape(-1) > 0
            count = int(inlier.sum())
            if count < self.min_inliers:
                return TrackConfirmation(False, float("inf"), float("inf"),
                                         float("inf"), count,
                                         "rotation_homography")
            calibrated = (np.linalg.inv(second_intrinsics) @ homography
                          @ first_intrinsics)
            left, _, right = np.linalg.svd(calibrated)
            rotation = left @ right
            if np.linalg.det(rotation) < 0:
                left[:, -1] *= -1
                rotation = left @ right
            projected_b = cv2.perspectiveTransform(
                points_a[inlier, None].astype(np.float64), homography)[:, 0]
            inverse_h = np.linalg.inv(homography)
            projected_a = cv2.perspectiveTransform(
                points_b[inlier, None].astype(np.float64), inverse_h)[:, 0]
            reprojection = float(np.median(0.5 * (
                np.linalg.norm(projected_b - points_b[inlier], axis=1)
                + np.linalg.norm(projected_a - points_a[inlier], axis=1))))
            return TrackConfirmation(
                True, 0.0,
                _rotation_error_deg(torch.from_numpy(rotation), da3_rotation),
                reprojection, count, "rotation_homography")

        focal = _effective_focal(first_intrinsics, second_intrinsics)
        essential, mask = cv2.findEssentialMat(
            normalized_a, normalized_b, np.eye(3), method=cv2.RANSAC,
            prob=0.999, threshold=self.ransac_px / focal)
        if essential is None:
            return TrackConfirmation(False, float("inf"), float("inf"),
                                     float("inf"), 0)
        _, rotation, translation, pose_mask = cv2.recoverPose(
            essential, normalized_a, normalized_b, np.eye(3), mask=mask)
        inlier = pose_mask.reshape(-1) > 0
        count = int(inlier.sum())
        if count < self.min_inliers:
            return TrackConfirmation(False, float("inf"), float("inf"),
                                     float("inf"), count)
        track_rotation = torch.from_numpy(rotation).double()
        track_translation = torch.from_numpy(translation[:, 0]).double()
        rotation_disagreement = _rotation_error_deg(track_rotation, da3_rotation)
        if float(da3_translation.norm()) <= 1e-8:
            translation_disagreement = float("inf")
        else:
            translation_disagreement = float(torch.linalg.norm(
                track_translation / track_translation.norm()
                - da3_translation / da3_translation.norm()))
        first_K_inv = np.linalg.inv(first_intrinsics)
        second_K_inv = np.linalg.inv(second_intrinsics)
        xa = np.concatenate([points_a[inlier], np.ones((count, 1))], axis=1)
        xb = np.concatenate([points_b[inlier], np.ones((count, 1))], axis=1)
        fundamental = second_K_inv.T @ essential[:3, :3] @ first_K_inv
        lines_b = xa @ fundamental.T
        lines_a = xb @ fundamental
        numerator = np.abs(np.sum(xb * lines_b, axis=1))
        denominator = np.sqrt(
            lines_b[:, 0] ** 2 + lines_b[:, 1] ** 2
            + lines_a[:, 0] ** 2 + lines_a[:, 1] ** 2).clip(1e-12)
        reprojection = float(np.median(numerator / denominator))
        return TrackConfirmation(
            True, translation_disagreement, rotation_disagreement,
            reprojection, count, "essential")


def _path_nondegenerate(c2w: torch.Tensor, *, translation_epsilon: float = 1e-6,
                        rotation_epsilon_deg: float = 1e-3) -> bool:
    """Accept a causal path with real translation or real rotation."""
    if len(c2w) < 2:
        return False
    translation = float((c2w[1:, :3, 3] - c2w[:-1, :3, 3]).norm(
        dim=-1).sum())
    rotations = [
        _rotation_error_deg(c2w[index - 1, :3, :3], c2w[index, :3, :3])
        for index in range(1, len(c2w))]
    return translation > translation_epsilon or sum(rotations) > rotation_epsilon_deg


def register_causal_pose_similarity(
        raw_c2w: torch.Tensor, reference_c2w: torch.Tensor) -> torch.Tensor:
    """Fix monocular Sim(3) with Lyra's exact causal-pose alignment kernel.

    Only causal recovered/reference cameras enter the fit.  The implementation is
    Lyra's pinned DA3 ``align_poses_umeyama`` routine, including its deterministic
    tiny-offset retry for degenerate paths; target commands never enter the fit.
    """
    raw = raw_c2w.detach().double().cpu()
    reference = reference_c2w.detach().double().cpu()
    count = len(reference)
    if (count < 4 or raw.ndim != 3 or raw.shape[-2:] != (4, 4)
            or reference.shape != (count, 4, 4) or len(raw) <= count
            or not bool(torch.isfinite(raw).all())
            or not bool(torch.isfinite(reference).all())):
        raise RuntimeError("pose registration requires four causal reference cameras")
    if str(DA3_SOURCE) not in sys.path:
        sys.path.insert(0, str(DA3_SOURCE))
    from depth_anything_3.utils.pose_align import align_poses_umeyama

    raw_w2c = torch.linalg.inv(raw).numpy()
    reference_w2c = torch.linalg.inv(reference).numpy()
    try:
        rotation, translation, scale, aligned_reference_w2c = align_poses_umeyama(
            reference_w2c, raw_w2c[:count], return_aligned=True)
    except Exception as error:
        raise RuntimeError("Lyra causal reference pose alignment failed") from error
    if (not np.isfinite(rotation).all()
            or not np.isfinite(translation).all()
            or not np.isfinite(scale) or float(scale) <= 0
            or not np.isfinite(aligned_reference_w2c).all()):
        raise RuntimeError("Lyra causal reference pose alignment is invalid")

    # Apply the exact Sim(3) returned by Lyra to the complete raw trajectory.  This
    # is algebraically identical to DA3's apply_umeyama_alignment_to_ext helper,
    # while preserving 4x4 matrices for ΩR's recurrent interface.
    raw_c2w_np = raw.numpy()
    registered = raw_c2w_np.copy()
    registered[:, :3, :3] = np.einsum(
        "ij,njk->nik", rotation, raw_c2w_np[:, :3, :3])
    registered[:, :3, 3] = (
        rotation @ (float(scale) * raw_c2w_np[:, :3, 3].T)).T + translation
    registered[:, 3, :] = 0
    registered[:, 3, 3] = 1
    if not np.isfinite(registered).all():
        raise RuntimeError("Lyra registered trajectory is invalid")
    return torch.from_numpy(registered).float()


def _register_vipe_to_references(
        raw_c2w: torch.Tensor, reference_c2w: torch.Tensor,
        *, gauge: DeploymentGauge,
        residual_max: float = 0.08) -> torch.Tensor:
    """Register VIPE and reject a poor causal-reference fit."""
    registered = register_causal_pose_similarity(raw_c2w, reference_c2w)
    reference = reference_c2w.detach().double().cpu()
    count = len(reference)
    residual = torch.linalg.vector_norm(
        gauge.normalize_c2w(registered[:count])[:, :3, 3]
        - gauge.normalize_c2w(reference)[:, :3, 3], dim=-1).mean()
    if not bool(torch.isfinite(residual)) or float(residual) > residual_max:
        raise RuntimeError("VIPE causal reference registration residual is too large")
    return registered.float()


def _reliability(depth: torch.Tensor) -> torch.Tensor:
    """Continuous local depth reliability, with invalid pixels at exact zero.

    The former binary cutoff made the downstream mean-confidence predicate
    vacuous: support was restricted to positive confidence and every supported
    pixel therefore had confidence exactly one.  This monotone score retains the
    same 0.15 relative-range scale while exposing useful quality variation.
    """
    value = depth[None, None]
    local_max = F.max_pool2d(value, 5, 1, 2)
    local_min = -F.max_pool2d(-value, 5, 1, 2)
    local_mean = F.avg_pool2d(value, 5, 1, 2)
    relative_range = (local_max - local_min) / local_mean.clamp_min(1e-6)
    finite = (value > 0) & torch.isfinite(value) & torch.isfinite(relative_range)
    reliability = torch.exp(-relative_range.clamp_min(0) / 0.15).clamp(0, 1)
    return torch.where(finite, reliability, torch.zeros_like(reliability))[0, 0]


class DA3MoGeJointRecovery:
    """One Lyra-aligned causal DA3 window shared by ΩR handoff callbacks."""

    def __init__(self, *, predictor: DA3Predictor,
                 pose_predictor,
                 scale: FrozenDA3MoGeScale, gauge: DeploymentGauge,
                 certified_views: tuple[CertifiedGeometryView, ...],
                 pose_reference_views: tuple[PoseReferenceView, ...],
                 generated_intrinsics: torch.Tensor,
                 generated_pipeline_c2w: torch.Tensor,
                 tracker: Callable[..., TrackConfirmation] | None = None,
                 min_depth_overlap: int = 32,
                 pose_policy: str = "vipe_realized"):
        if not certified_views:
            raise ValueError("joint recovery requires the immutable anchor view")
        if scale.gauge_manifest_sha256 != gauge.manifest()["payload_sha256"]:
            raise ValueError("DA3/MoGe scale belongs to another deployment gauge")
        if generated_intrinsics.shape not in {(3, 3)} and not (
                generated_intrinsics.ndim == 3
                and generated_intrinsics.shape[-2:] == (3, 3)):
            raise ValueError("generated intrinsics must be 3x3 or Tx3x3")
        if (generated_pipeline_c2w.ndim != 3
                or generated_pipeline_c2w.shape[-2:] != (4, 4)
                or not bool(torch.isfinite(generated_pipeline_c2w).all())):
            raise ValueError("generated Lyra pipeline poses must be finite Tx4x4 c2w")
        self.predictor = predictor
        if pose_policy not in {"vipe_realized", "lyra_commanded"}:
            raise ValueError("unknown recurrent pose policy")
        if pose_policy == "vipe_realized" and not callable(pose_predictor):
            raise ValueError("joint recovery requires calibrated VIPE pose recovery")
        if pose_policy == "lyra_commanded" and pose_predictor is not None:
            raise ValueError("Lyra-commanded recovery cannot also invoke VIPE")
        minimum_references = 4 if pose_policy == "vipe_realized" else 1
        if (len(pose_reference_views) < minimum_references
                or len({item.frame_id for item in pose_reference_views})
                    != len(pose_reference_views)
                or any(item.c2w.shape != (4, 4)
                       or item.intrinsics.shape != (3, 3)
                       or not bool(torch.isfinite(item.c2w).all())
                       or not bool(torch.isfinite(item.intrinsics).all())
                       for item in pose_reference_views)):
            raise ValueError(
                "joint recovery lacks the causal references required by its pose policy")
        self.pose_predictor = pose_predictor
        self.pose_policy = pose_policy
        self.scale = scale
        self.gauge = gauge
        self.certified_views = certified_views
        self.pose_reference_views = pose_reference_views
        self.generated_intrinsics = generated_intrinsics.detach().float().cpu()
        self.generated_pipeline_c2w = generated_pipeline_c2w.detach().float().cpu()
        self.tracker = tracker or OpenCVFeatureTrackConfirmation()
        self.min_depth_overlap = int(min_depth_overlap)
        self._cache_key = None
        self._cache_value: tuple[PoseRecovery, GeometryRecovery] | None = None

    def _recover(self, pixels: torch.Tensor,
                 frame_ids: tuple[int, ...]) -> tuple[PoseRecovery, GeometryRecovery]:
        frames = _as_frames(pixels)
        if len(frame_ids) != len(frames):
            raise ValueError("generated frame IDs do not match decoded RGB frames")
        if tuple(sorted(set(frame_ids))) != frame_ids:
            raise ValueError("generated frame IDs must be strictly ordered")
        generated_u8 = _to_uint8(frames)
        # Endpoint evidence may retain the full decoded chunk for audit/re-encode.
        # Its pose and depth belong to the terminal frame, so recurrent DA3 must
        # consume that same terminal RGB rather than the chunk's first frame.
        certified_u8 = np.stack([_to_uint8(_as_frames(view.rgb))[-1]
                                 for view in self.certified_views])
        images = np.concatenate([certified_u8, generated_u8], axis=0)
        height, width = images.shape[1:3]
        if any(tuple(view.depth.squeeze().shape) != (height, width)
               for view in self.certified_views):
            raise ValueError("certified depth grid differs from recovery RGB")
        generated_K = (self.generated_intrinsics[None].expand(len(frames), -1, -1)
                       if self.generated_intrinsics.ndim == 2
                       else self.generated_intrinsics)
        if len(generated_K) != len(frames):
            raise ValueError("generated intrinsics do not match RGB frames")
        if len(self.generated_pipeline_c2w) != len(frames):
            raise ValueError("generated pipeline poses do not match RGB frames")
        K = torch.cat([torch.stack([view.intrinsics.float().cpu()
                                   for view in self.certified_views]),
                       generated_K], dim=0)
        pipeline_c2w = torch.cat([
            torch.stack([view.c2w.float().cpu() for view in self.certified_views]),
            self.generated_pipeline_c2w], dim=0)
        pipeline_w2c = torch.linalg.inv(pipeline_c2w.double()).float()
        material = hashlib.sha256()
        material.update(images.tobytes())
        pose_reference_u8 = np.stack([
            _to_uint8(_as_frames(view.rgb))[-1]
            for view in self.pose_reference_views])
        pose_reference_c2w = torch.stack([
            view.c2w.detach().float().cpu() for view in self.pose_reference_views])
        vipe_images = np.concatenate([pose_reference_u8, generated_u8], axis=0)
        material.update(vipe_images.tobytes())
        material.update(pose_reference_c2w.numpy().tobytes())
        material.update(np.asarray(frame_ids, dtype=np.int64).tobytes())
        da3_w2c = _lyra_alignment_offsets(pipeline_w2c.numpy())
        material.update(da3_w2c.tobytes())
        material.update(self.scale.payload_sha256.encode())
        key = material.hexdigest()
        if key == self._cache_key and self._cache_value is not None:
            return self._cache_value
        prediction = self.predictor(images, K.numpy(), da3_w2c)
        if (prediction.depths.shape != (len(images), height, width)
                or prediction.predicted_c2w.shape != (len(images), 4, 4)
                or not bool(torch.isfinite(prediction.depths).all())
                or not bool(torch.isfinite(prediction.predicted_c2w).all())
                or (prediction.valid_masks is not None
                    and (prediction.valid_masks.shape != prediction.depths.shape
                         or not bool(torch.isfinite(prediction.valid_masks).all())
                         or bool((prediction.valid_masks < 0).any())
                         or bool((prediction.valid_masks > 1).any())))):
            raise RuntimeError("DA3 returned an invalid joint depth/pose recovery")
        # Preserve Lyra's exact default DA3 depth semantics: its aligned pose output
        # is the supplied pipeline trajectory and is not treated as an independent
        # camera measurement.  Realized pose comes from Lyra's separately calibrated
        # VIPE wrapper, Sim(3)-registered only by causal observed input cameras.
        da3_mapped = pipeline_c2w
        if self.pose_policy == "lyra_commanded":
            # Released Lyra supplies the requested pipeline extrinsics to DA3 and
            # continues in that same camera gauge; it does not gate recurrence on
            # a separately recovered VIPE pose. Keep VIPE as an evaluation
            # instrument, not a deployment-only state transition dependency.
            realized_generated = self.generated_pipeline_c2w.clone()
            vipe_finite = True
        else:
            try:
                vipe_raw = self.pose_predictor(vipe_images, sequence_id=key)
                if (vipe_raw.shape != (len(vipe_images), 4, 4)
                        or not bool(torch.isfinite(vipe_raw).all())):
                    raise RuntimeError("VIPE returned invalid raw camera matrices")
                realized = _register_vipe_to_references(
                    vipe_raw, pose_reference_c2w, gauge=self.gauge)
                realized_generated = realized[len(self.pose_reference_views):]
                vipe_finite = True
            except Exception:
                realized_generated = None
                vipe_finite = False
        normalized_depths = torch.stack([
            self.scale.normalize_aligned_da3_depth(value)
            for value in prediction.depths])
        endpoint_index = len(images) - 1
        source_index = len(self.certified_views) - 1
        realized_endpoint = (None if realized_generated is None
                             else realized_generated[-1])
        tracker_endpoint = (self.generated_pipeline_c2w[-1]
                            if realized_endpoint is None else realized_endpoint)
        track_reference = self.pose_reference_views[-1]
        track = self.tracker(
            pose_reference_u8[-1], images[endpoint_index],
            track_reference.intrinsics.detach().float().cpu().numpy(),
            K[endpoint_index].numpy(), track_reference.c2w, tracker_endpoint)
        endpoint_depth = normalized_depths[endpoint_index]
        confidence = _reliability(endpoint_depth)
        if prediction.valid_masks is not None:
            confidence *= prediction.valid_masks[endpoint_index].float()
        source = self.certified_views[source_index]
        source_depth = source.depth.detach().float().squeeze()
        source_confidence = source.confidence.detach().float().squeeze()
        rendered = render_point_source(
            torch.zeros((1, height, width)), source_depth[None],
            source_confidence[None],
            source_c2w=self.gauge.normalize_c2w(source.c2w),
            target_c2w=self.gauge.normalize_c2w(tracker_endpoint),
            source_intrinsics=source.intrinsics,
            target_intrinsics=K[endpoint_index])
        support = (rendered["validity"][0] > 0) & (confidence > 0) \
            & (endpoint_depth > 0)
        overlap = int(support.sum())
        if overlap < self.min_depth_overlap:
            depth_error = float("inf")
        else:
            depth_error = float((
                rendered["depth"][0, support].clamp_min(1e-8).log()
                - endpoint_depth[support].clamp_min(1e-8).log()).abs().median())
        estimator_ids = tuple(dict.fromkeys((
            *(view.frame_id for view in self.certified_views),
            *(view.frame_id for view in self.pose_reference_views),
            *frame_ids)))
        nondegenerate = (False if realized_generated is None
                         else _path_nondegenerate(realized_generated))
        if realized_endpoint is None:
            translation_residual = rotation_residual = float("inf")
        else:
            translation_residual = float(torch.linalg.vector_norm(
                self.gauge.normalize_c2w(realized_endpoint)[:3, 3]
                - self.gauge.normalize_c2w(
                    self.generated_pipeline_c2w[-1])[:3, 3]))
            rotation_residual = _rotation_error_deg(
                self.generated_pipeline_c2w[-1, :3, :3],
                realized_endpoint[:3, :3])
        pose = PoseRecovery(
            realized_endpoint, vipe_finite, bool(nondegenerate),
            translation_residual, rotation_residual,
            track.valid, track.translation_disagreement,
            track.rotation_disagreement_deg, key, realized_generated)
        # Certification confidence must represent both sides of the cross-view
        # comparison.  Averaging only a target-side binary inclusion mask was
        # identically one and could never reject weak geometry.
        pair_confidence = torch.sqrt(
            rendered["confidence"][0].clamp(0, 1)
            * confidence.clamp(0, 1))
        mean_confidence = float(pair_confidence[support].mean()) if overlap else 0.0
        geometry = GeometryRecovery(
            endpoint_depth[None], confidence[None], mean_confidence,
            overlap / float(height * width), track.reprojection_error_px,
            depth_error, estimator_ids, key,
            self.gauge.manifest()["payload_sha256"])
        self._cache_key, self._cache_value = key, (pose, geometry)
        return pose, geometry

    def pose(self, pixels: torch.Tensor, frame_ids: tuple[int, ...]) -> PoseRecovery:
        return self._recover(pixels, frame_ids)[0]

    def depth(self, pixels: torch.Tensor,
              frame_ids: tuple[int, ...]) -> GeometryRecovery:
        return self._recover(pixels, frame_ids)[1]
