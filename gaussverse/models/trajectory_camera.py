"""Gauge-factored camera and motion encoding for causal trajectory generation.

Pluecker rays are an excellent dense spatial camera signal, but they make the network
rediscover simple global facts (relative rotation, translation velocity, acceleration,
intrinsics, and time spacing) from 1,536 pixel-unshuffled channels.  This module is a
strong complementary baseline: analytic target-relative SE(3) and motion features are
projected directly into a per-frame Q/K bias.  They never enter V.

The representation is gauge-factored: poses must already be expressed relative to the
final commanded target, so a global rigid transform of the reconstruction cannot change
the code.  It is also continuous in rotation through the six-dimensional representation
instead of Euler-angle discontinuities.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# translation(3), rotation-6D(6), normalized intrinsics(4),
# first translation difference(3), first rotation difference-6D(6),
# second translation difference(3), temporal delta(1), target role(1)
CAMERA_FEATURE_DIM = 27
# Four RGB frames are represented by one Wan latent.  The endpoint/global code
# above is necessary but cannot distinguish four different within-latent paths
# that end at the same camera.  Keep the three ordered pre-endpoint SE(3)
# residuals explicitly (translation in the endpoint basis + SO(3) log vector).
SUBRGB_RESIDUAL_DIM = 3 * 6
COMMAND_CAMERA_FEATURE_DIM = CAMERA_FEATURE_DIM + SUBRGB_RESIDUAL_DIM


def _invert_rigid(transform: torch.Tensor) -> torch.Tensor:
    """Analytic SE(3) inverse; avoids thousands of tiny solver calls per rollout."""
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    inverse = torch.zeros_like(transform)
    inverse[..., :3, :3] = rotation.transpose(-1, -2)
    inverse[..., :3, 3] = -torch.einsum(
        "...ij,...j->...i", rotation.transpose(-1, -2), translation
    )
    inverse[..., 3, 3] = 1
    return inverse


def _rotation_6d(rotation: torch.Tensor) -> torch.Tensor:
    """Continuous first-two-column representation, flattened column-major."""
    return rotation[..., :, :2].transpose(-1, -2).reshape(*rotation.shape[:-2], 6)


def _so3_log(rotation: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Stable SO(3) logarithm for small commanded inter-frame rotations.

    Camera commands in this project are smooth, but the implementation also
    handles the near-pi case without dividing by a vanishing sine.  Returning a
    three-vector, rather than Euler angles or a matrix sum, preserves direction
    and magnitude in a continuous local chart around the endpoint.
    """
    if rotation.shape[-2:] != (3, 3):
        raise ValueError("SO(3) logarithm requires ...x3x3 rotations")
    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    angle = torch.acos(cosine)
    vee = torch.stack([
        rotation[..., 2, 1] - rotation[..., 1, 2],
        rotation[..., 0, 2] - rotation[..., 2, 0],
        rotation[..., 1, 0] - rotation[..., 0, 1],
    ], dim=-1) * 0.5
    sine = torch.sin(angle)
    regular = angle / sine.clamp_min(eps)
    # sin(theta) changes sign only outside [0, pi], which acos never returns.
    # The first-order limit is one at identity.
    factor = torch.where(angle < 1e-4, torch.ones_like(angle), regular)
    value = factor[..., None] * vee
    # At exactly pi the skew part is numerically uninformative.  Commands this
    # discontinuous are rejected by the trajectory contract; fail visibly if
    # one reaches training instead of emitting an arbitrary axis.
    if bool((angle > torch.pi - 1e-3).any()):
        raise ValueError("within-latent camera rotation is too close to pi")
    return value


def trajectory_camera_features(
    c2w_relative: torch.Tensor,
    K: torch.Tensor,
    image_size: tuple[int, int],
    *,
    temporal_positions: torch.Tensor | None = None,
    num_targets: int = 1,
) -> torch.Tensor:
    """Build BxFx27 analytic camera features.

    ``c2w_relative`` is BxFx4x4 (or Fx4x4) with the final commanded target as
    identity and translations already divided by the persistent scene scale. ``K`` may
    be shared (3x3), per-batch (Bx3x3), or per-frame (BxFx3x3).
    """
    squeeze = c2w_relative.ndim == 3
    if squeeze:
        c2w_relative = c2w_relative[None]
    if c2w_relative.ndim != 4 or c2w_relative.shape[-2:] != (4, 4):
        raise ValueError("relative poses must be Fx4x4 or BxFx4x4")
    b, frames = c2w_relative.shape[:2]
    if not 1 <= num_targets <= frames:
        raise ValueError("num_targets must leave a valid frame sequence")
    if K.ndim == 2:
        K = K[None, None].expand(b, frames, 3, 3)
    elif K.ndim == 3:
        if K.shape[0] == b:
            K = K[:, None].expand(b, frames, 3, 3)
        elif b == 1 and K.shape[0] == frames:
            K = K[None]
        else:
            raise ValueError("ambiguous intrinsics shape")
    if K.shape != (b, frames, 3, 3):
        raise ValueError(f"intrinsics must broadcast to {(b, frames, 3, 3)}")
    height, width = image_size
    if height < 1 or width < 1:
        raise ValueError("image dimensions must be positive")

    pose = c2w_relative.float()
    rotation = pose[..., :3, :3]
    translation = pose[..., :3, 3]
    rotation6 = _rotation_6d(rotation)

    delta_t = torch.zeros_like(translation)
    delta_t[:, 1:] = translation[:, 1:] - translation[:, :-1]
    delta_r6 = torch.zeros_like(rotation6)
    if frames > 1:
        relative_rotation = rotation[:, :-1].transpose(-1, -2) @ rotation[:, 1:]
        delta_r6[:, 1:] = _rotation_6d(relative_rotation)
        # Identity is the meaningful zero-motion rotation feature, not six zeros.
        delta_r6[:, :1] = _rotation_6d(
            torch.eye(3, device=pose.device, dtype=pose.dtype).view(1, 1, 3, 3)
        )
    else:
        delta_r6[:] = _rotation_6d(
            torch.eye(3, device=pose.device, dtype=pose.dtype).view(1, 1, 3, 3)
        )
    accel_t = torch.zeros_like(translation)
    if frames > 2:
        accel_t[:, 2:] = delta_t[:, 2:] - delta_t[:, 1:-1]

    intrinsics = torch.stack([
        torch.log(K[..., 0, 0].float().clamp_min(1e-6) / width),
        torch.log(K[..., 1, 1].float().clamp_min(1e-6) / height),
        K[..., 0, 2].float() / width - 0.5,
        K[..., 1, 2].float() / height - 0.5,
    ], dim=-1)
    if temporal_positions is None:
        time = torch.arange(frames, device=pose.device, dtype=pose.dtype)[None].expand(b, -1)
    else:
        time = temporal_positions.to(device=pose.device, dtype=pose.dtype)
        if time.ndim == 1:
            time = time[None].expand(b, -1)
        if time.shape != (b, frames):
            raise ValueError(f"temporal positions must be {(b, frames)}")
    delta_time = torch.zeros_like(time)
    delta_time[:, 1:] = time[:, 1:] - time[:, :-1]
    # Retrieved/archive slots use the negative temporal sentinel. They retain their
    # target-relative pose, but they are not steps in the agent's motion trajectory.
    # Do not fabricate velocity/acceleration across an unordered->recent boundary.
    ordered = time >= 0
    transition_valid = ordered[:, 1:] & ordered[:, :-1]
    delta_t[:, 1:] *= transition_valid[..., None]
    identity6 = _rotation_6d(
        torch.eye(3, device=pose.device, dtype=pose.dtype).view(1, 1, 3, 3)
    )
    delta_r6[:, 1:] = torch.where(
        transition_valid[..., None], delta_r6[:, 1:], identity6,
    )
    if frames > 2:
        acceleration_valid = transition_valid[:, 1:] & transition_valid[:, :-1]
        accel_t[:, 2:] *= acceleration_valid[..., None]
    delta_time[:, 1:] *= transition_valid
    # Log compression retains variable-rate ordering without making a long archive
    # retrieval numerically dominate recent one-frame motion.
    delta_time = torch.sign(delta_time) * torch.log1p(delta_time.abs()) / math.log(9.0)
    target_role = torch.zeros(b, frames, device=pose.device, dtype=pose.dtype)
    target_role[:, -num_targets:] = 1

    features = torch.cat([
        translation, rotation6, intrinsics, delta_t, delta_r6, accel_t,
        delta_time[..., None], target_role[..., None],
    ], dim=-1)
    if features.shape[-1] != CAMERA_FEATURE_DIM or not torch.isfinite(features).all():
        raise RuntimeError("invalid analytic camera feature construction")
    return features[0] if squeeze else features


def command_resolved_camera_features(
    c2w_span_relative: torch.Tensor,
    K_span: torch.Tensor,
    image_size: tuple[int, int],
    *,
    temporal_positions: torch.Tensor | None = None,
    num_targets: int = 1,
) -> torch.Tensor:
    """Endpoint-global plus ordered sub-RGB camera code for each Wan latent.

    Args:
        c2w_span_relative: ``BxFx4x4x4`` (or ``Fx4x4x4``) camera-to-world
            transforms, already in the immutable anchor scale and expressed in
            the final commanded target gauge.
        K_span: matching intrinsics as ``BxFx4x3x3`` or ``Fx4x3x3``.

    The endpoint participates in the existing 27-D global trajectory code.
    Each of the first three RGB poses is then represented relative to its own
    endpoint, in order.  No summation, pooling, or learned scalar can erase one
    sub-frame before the network sees it.
    """
    squeeze = c2w_span_relative.ndim == 4
    if squeeze:
        c2w_span_relative = c2w_span_relative[None]
        K_span = K_span[None]
    if (c2w_span_relative.ndim != 5
            or c2w_span_relative.shape[-3:] != (4, 4, 4)):
        raise ValueError("camera spans must be Fx4x4x4 or BxFx4x4x4")
    b, frames = c2w_span_relative.shape[:2]
    if K_span.shape != (b, frames, 4, 3, 3):
        raise ValueError(f"intrinsic spans must be {(b, frames, 4, 3, 3)}")

    endpoint = c2w_span_relative[:, :, -1]
    endpoint_K = K_span[:, :, -1]
    global_features = trajectory_camera_features(
        endpoint, endpoint_K, image_size,
        temporal_positions=temporal_positions, num_targets=num_targets,
    )
    endpoint_rotation = endpoint[..., :3, :3]
    endpoint_translation = endpoint[..., :3, 3]
    earlier = c2w_span_relative[:, :, :3]
    # Translation expressed in the endpoint camera basis is gauge-invariant to
    # a global rigid transform while retaining the immutable scene scale.
    delta_world = earlier[..., :3, 3] - endpoint_translation[:, :, None]
    delta_translation = torch.einsum(
        "bfji,bfsj->bfsi", endpoint_rotation, delta_world
    )
    delta_rotation = (
        endpoint_rotation.transpose(-1, -2)[:, :, None]
        @ earlier[..., :3, :3]
    )
    delta_rotvec = _so3_log(delta_rotation)
    subrgb = torch.cat([delta_translation, delta_rotvec], dim=-1).reshape(
        b, frames, SUBRGB_RESIDUAL_DIM
    )
    features = torch.cat([global_features, subrgb], dim=-1)
    if (features.shape != (b, frames, COMMAND_CAMERA_FEATURE_DIM)
            or not bool(torch.isfinite(features).all())):
        raise RuntimeError("invalid command-resolved camera features")
    return features[0] if squeeze else features


class TrajectoryCameraQKAdapter(nn.Module):
    """Analytic global camera/motion features -> spatially expanded Q/K bias."""

    def __init__(self, dim: int = 3072, hidden: int = 512,
                 feature_dim: int = CAMERA_FEATURE_DIM):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )
        self.gate = nn.Parameter(torch.tensor(1.0))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor, tokens_per_frame: int) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError("camera features must be BxFxfeature_dim")
        if tokens_per_frame < 1:
            raise ValueError("tokens_per_frame must be positive")
        code = torch.tanh(self.gate) * self.net(features)
        return code.repeat_interleave(tokens_per_frame, dim=1)


class CommandResolvedCameraQKAdapter(TrajectoryCameraQKAdapter):
    """Q/K-only projection for endpoint + ordered four-RGB camera commands."""

    def __init__(self, dim: int = 3072, hidden: int = 512):
        super().__init__(dim=dim, hidden=hidden,
                         feature_dim=COMMAND_CAMERA_FEATURE_DIM)


def ray_local_transforms(
    c2w_relative: torch.Tensor,
    K: torch.Tensor,
    image_size: tuple[int, int],
    token_grid: tuple[int, int],
) -> torch.Tensor:
    """World-to-ray homogeneous transforms for every DiT token (B,F*Ht*Wt,4,4)."""
    if c2w_relative.ndim == 3:
        c2w_relative = c2w_relative[None]
    if c2w_relative.ndim != 4 or c2w_relative.shape[-2:] != (4, 4):
        raise ValueError("relative poses must be Fx4x4 or BxFx4x4")
    b, frames = c2w_relative.shape[:2]
    if K.ndim == 2:
        K = K[None, None].expand(b, frames, 3, 3)
    elif K.ndim == 3 and b == 1 and K.shape[0] == frames:
        K = K[None]
    elif K.ndim == 3 and K.shape[0] == b:
        K = K[:, None].expand(b, frames, 3, 3)
    if K.shape != (b, frames, 3, 3):
        raise ValueError("intrinsics do not match pose sequence")
    height, width = image_size
    grid_h, grid_w = token_grid
    if min(height, width, grid_h, grid_w) < 1:
        raise ValueError("image and token grids must be positive")

    yy, xx = torch.meshgrid(
        (torch.arange(grid_h, device=c2w_relative.device, dtype=torch.float32) + 0.5)
        * height / grid_h,
        (torch.arange(grid_w, device=c2w_relative.device, dtype=torch.float32) + 0.5)
        * width / grid_w,
        indexing="ij",
    )
    pixels = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
    camera_dirs = torch.einsum(
        "bfij,hwj->bfhwi", torch.linalg.inv(K.float()), pixels
    )
    camera_dirs = F.normalize(camera_dirs, dim=-1)
    rotation = c2w_relative[..., :3, :3].float()
    z_axis = F.normalize(torch.einsum("bfij,bfhwj->bfhwi", rotation, camera_dirs), dim=-1)
    vertical = rotation[..., :, 1][:, :, None, None].expand_as(z_axis)
    x_axis = F.normalize(torch.cross(vertical, z_axis, dim=-1), dim=-1)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=-1), dim=-1)
    ray_to_world = torch.zeros(
        b, frames, grid_h, grid_w, 4, 4,
        device=c2w_relative.device, dtype=torch.float32,
    )
    ray_to_world[..., :3, 0] = x_axis
    ray_to_world[..., :3, 1] = y_axis
    ray_to_world[..., :3, 2] = z_axis
    ray_to_world[..., :3, 3] = c2w_relative[..., :3, 3].float()[:, :, None, None]
    ray_to_world[..., 3, 3] = 1
    world_to_ray = _invert_rigid(ray_to_world)
    return world_to_ray.reshape(b, frames * grid_h * grid_w, 4, 4)


class RayLocalUCPEAttention(nn.Module):
    """Own-basis softmax camera attention with SANA-WM-style homogeneous UCPE.

    A configurable prefix of every head is divided into 4-D homogeneous groups.
    Query groups receive ``D^T`` (row-vector ``q @ D``), key/value groups receive
    ``D^-1``, and the attended output is mapped back by ``D``. Remaining channels are
    ordinary attention channels. This is materially stronger than adding a pose vector
    before a frozen Q/K projection because camera geometry structures the bilinear form.
    """

    def __init__(self, dim: int = 3072, cam_dim: int = 768, heads: int = 8,
                 geometric_channels: int = 32):
        super().__init__()
        if cam_dim % heads or geometric_channels % 4 or geometric_channels > cam_dim // heads:
            raise ValueError("UCPE dimensions must divide heads and homogeneous 4-D groups")
        self.cam_dim = cam_dim
        self.heads = heads
        self.head_dim = cam_dim // heads
        self.geometric_channels = geometric_channels
        self.q = nn.Linear(dim, cam_dim, bias=False)
        self.k = nn.Linear(dim, cam_dim, bias=False)
        self.v = nn.Linear(dim, cam_dim, bias=False)
        from gaussverse.diffsynth.models.wan_video_dit import RMSNorm
        self.norm_q = RMSNorm(cam_dim, eps=1e-6)
        self.norm_k = RMSNorm(cam_dim, eps=1e-6)
        self.out = nn.Linear(cam_dim, dim)
        self.gate = nn.Parameter(torch.tensor(1.0))
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @staticmethod
    def _row_transform(groups: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bhngc,bncd->bhngd", groups, matrix)

    def forward(self, hidden: torch.Tensor, world_to_ray: torch.Tensor) -> torch.Tensor:
        b, sequence, _ = hidden.shape
        if world_to_ray.shape != (b, sequence, 4, 4):
            raise ValueError(
                f"ray-local transforms must be {(b, sequence, 4, 4)}, "
                f"got {tuple(world_to_ray.shape)}"
            )
        matrix = world_to_ray.to(device=hidden.device, dtype=hidden.dtype)
        inverse_transpose = _invert_rigid(matrix.float()).transpose(-1, -2).to(hidden.dtype)

        def split(value):
            return value.view(b, sequence, self.heads, self.head_dim).transpose(1, 2)

        q, k, v = split(self.norm_q(self.q(hidden))), split(self.norm_k(self.k(hidden))), split(self.v(hidden))
        g = self.geometric_channels
        groups = g // 4
        q_geo = self._row_transform(q[..., :g].reshape(b, self.heads, sequence, groups, 4), matrix)
        k_geo = self._row_transform(k[..., :g].reshape(b, self.heads, sequence, groups, 4), inverse_transpose)
        v_geo = self._row_transform(v[..., :g].reshape(b, self.heads, sequence, groups, 4), inverse_transpose)
        q = torch.cat([q_geo.flatten(-2), q[..., g:]], dim=-1)
        k = torch.cat([k_geo.flatten(-2), k[..., g:]], dim=-1)
        v = torch.cat([v_geo.flatten(-2), v[..., g:]], dim=-1)
        attended = F.scaled_dot_product_attention(q, k, v)
        out_geo = self._row_transform(
            attended[..., :g].reshape(b, self.heads, sequence, groups, 4),
            matrix.transpose(-1, -2),
        ).flatten(-2)
        attended = torch.cat([out_geo, attended[..., g:]], dim=-1)
        attended = attended.transpose(1, 2).reshape(b, sequence, self.cam_dim)
        return torch.tanh(self.gate) * self.out(attended)


class RayLocalUCPEStack(nn.ModuleDict):
    """Sparse UCPE residual sites; checkpoint keys encode their block depths."""

    DEFAULT_BLOCKS = (7, 15, 23)

    def __init__(self, blocks=DEFAULT_BLOCKS, **kwargs):
        super().__init__({str(index): RayLocalUCPEAttention(**kwargs) for index in blocks})
        self.blocks = tuple(blocks)

    def site(self, index: int):
        return self[str(index)] if str(index) in self else None
