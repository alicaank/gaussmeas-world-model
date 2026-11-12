"""Keyboard commands as camera-local motions over a chunk of frames.

`demo/scripted_trajectory.apply_command` maps one keypress to one SE(3) jump, which
was right when the model emitted one frame per command.  A chunked model emits eight
RGB frames per command, so a keypress is a *motion held for half a second*, not an
instantaneous teleport: the eight frames must sweep the commanded delta rather than
all sit at its endpoint.

Conventions are the renderer's: OpenCV camera frame (x-right, y-down, z-forward),
camera-to-world matrices, translation in persistent scene-scale units.

    W/S  forward / back along camera-local z
    A/D  strafe left / right along camera-local x
    Q/E  yaw left / right about camera-local y  (Q = the canonical "10 degrees left")

Yaw sign check: with y down, right-multiplying by ``R_y(theta)`` sends the forward
axis toward ``+x`` (right) for positive ``theta``.  Q therefore uses ``-yaw``.
"""
from __future__ import annotations

from dataclasses import dataclass

import math

import torch
from torch import Tensor

COMMANDS = ("W", "A", "S", "D", "Q", "E")
_TRANSLATION = {"W": (0.0, 0.0, 1.0), "S": (0.0, 0.0, -1.0),
                "A": (-1.0, 0.0, 0.0), "D": (1.0, 0.0, 0.0)}
_YAW_SIGN = {"Q": -1.0, "E": 1.0}

DEFAULT_TRANSLATION = 0.01   # low-level persistent scene units; never call this cm
DEFAULT_YAW_DEGREES = 10.0


@dataclass(frozen=True)
class WalkingMotionContract:
    """Metric defaults for one eight-RGB walking-simulator call.

    Monocular MoGe/DA3 geometry has a similarity ambiguity: one normalized gauge
    unit has no observable physical length.  ``anchor_median_depth_meters`` is
    therefore a required host/simulator measurement, not an image-derived guess.
    It says how many physical metres correspond to the deployment gauge's unit
    median depth.  With that one scalar, metric commands convert exactly into the
    raw scene units consumed by :func:`command_latent_spans`.
    """

    min_translation_meters: float = 0.10
    max_translation_meters: float = 0.20
    min_rotation_degrees: float = 5.0
    max_rotation_degrees: float = 10.0
    default_translation_meters: float = 0.15
    default_rotation_degrees: float = 10.0
    min_anchor_median_depth_meters: float = 1.0
    max_anchor_median_depth_meters: float = 10.0

    def __post_init__(self) -> None:
        values = tuple(vars(self).values())
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("walking motion bounds must be finite and positive")
        if not (self.min_translation_meters <= self.default_translation_meters
                <= self.max_translation_meters):
            raise ValueError("default translation lies outside the walking range")
        if not (self.min_rotation_degrees <= self.default_rotation_degrees
                <= self.max_rotation_degrees):
            raise ValueError("default rotation lies outside the walking range")

    def validate_translation(self, meters: float) -> float:
        """Accept any positive request; min/max describe the typical UI range."""
        value = float(meters)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("one walking call needs a positive physical distance")
        return value

    def validate_rotation(self, degrees: float) -> float:
        """Accept any positive request; min/max describe the typical UI range."""
        value = float(degrees)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("one walking call needs a positive rotation")
        return value

    def scene_translation(self, *, translation_meters: float,
                          anchor_median_depth_meters: float,
                          gauge_scale: float) -> float:
        """Convert physical metres to the raw pose units used by the gauge."""
        meters = self.validate_translation(translation_meters)
        physical_gauge = float(anchor_median_depth_meters)
        raw_gauge = float(gauge_scale)
        if (not math.isfinite(physical_gauge) or physical_gauge <= 0
                or not math.isfinite(raw_gauge) or raw_gauge <= 0):
            raise ValueError(
                "metric commands require positive anchor_median_depth_meters "
                "and gauge_scale")
        return meters / physical_gauge * raw_gauge


WALKING_MOTION = WalkingMotionContract()


def parse_commands(value: str) -> list[str]:
    """Accept "W,W,Q" or "WWQ"; reject anything outside WASDQE."""
    tokens = [token.strip().upper() for token in value.replace(" ", ",").split(",")
              if token.strip()]
    if len(tokens) == 1 and len(tokens[0]) > 1:
        tokens = list(tokens[0])
    invalid = [token for token in tokens if token not in set(COMMANDS)]
    if not tokens or invalid:
        raise ValueError(f"commands must contain only W/A/S/D/Q/E; invalid={invalid}")
    return tokens


def _yaw_matrix(radians: float, *, dtype, device) -> Tensor:
    cosine, sine = math.cos(radians), math.sin(radians)
    return torch.tensor(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=dtype, device=device,
    )


def apply_command(c2w: Tensor, command: str, *, translation: float = DEFAULT_TRANSLATION,
                  yaw_degrees: float = DEFAULT_YAW_DEGREES,
                  fraction: float = 1.0) -> Tensor:
    """Apply ``fraction`` of one command to a 4x4 camera-to-world matrix.

    ``fraction`` exists so a chunk can sweep the motion; ``fraction=1`` reproduces the
    single-frame behaviour exactly.
    """
    if c2w.shape != (4, 4):
        raise ValueError(f"apply_command needs a 4x4 pose, got {tuple(c2w.shape)}")
    if command not in set(COMMANDS):
        raise ValueError(f"unknown command {command!r}; expected one of {COMMANDS}")
    if not math.isfinite(translation) or translation <= 0:
        raise ValueError("translation must be finite and positive")
    if not math.isfinite(yaw_degrees) or yaw_degrees <= 0:
        raise ValueError("yaw must be finite and positive")
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be finite and lie in [0,1]")

    out = c2w.double().clone()
    if command in _TRANSLATION:
        delta = torch.tensor(_TRANSLATION[command], dtype=out.dtype, device=out.device)
        out[:3, 3] += out[:3, :3] @ (translation * fraction * delta)
    else:
        radians = math.radians(yaw_degrees * fraction) * _YAW_SIGN[command]
        out[:3, :3] = out[:3, :3] @ _yaw_matrix(
            radians, dtype=out.dtype, device=out.device
        )
    return out.to(c2w.dtype)


def command_frame_poses(c2w: Tensor, command: str, frames: int, *,
                        translation: float = DEFAULT_TRANSLATION,
                        yaw_degrees: float = DEFAULT_YAW_DEGREES) -> Tensor:
    """One command over ``frames`` RGB frames -> ``(frames, 4, 4)``.

    The sweep is uniform in the command parameter and the final frame lands exactly on
    the full commanded delta, so chaining commands frame-by-frame agrees with chaining
    them chunk-by-chunk. The starting pose is *not* included: these are the ``frames``
    new poses the chunk must realise.
    """
    if frames < 1:
        raise ValueError("a command must cover at least one frame")
    fractions = [(index + 1) / frames for index in range(frames)]
    return torch.stack([
        apply_command(c2w, command, translation=translation,
                      yaw_degrees=yaw_degrees, fraction=fraction)
        for fraction in fractions
    ])


def command_sequence_poses(c2w: Tensor, commands: list[str], frames_per_command: int, *,
                           translation: float = DEFAULT_TRANSLATION,
                           yaw_degrees: float = DEFAULT_YAW_DEGREES) -> Tensor:
    """Chain commands, emitting ``frames_per_command`` poses for each.

    Returns ``(len(commands) * frames_per_command, 4, 4)``; the caller keeps the
    starting pose separately as the stream anchor.
    """
    poses, current = [], c2w
    for command in commands:
        block = command_frame_poses(
            current, command, frames_per_command,
            translation=translation, yaw_degrees=yaw_degrees,
        )
        poses.append(block)
        current = block[-1]
    return torch.cat(poses, dim=0)


def command_latent_spans(c2w: Tensor, command: str, latent_frames: int, *,
                         translation: float = DEFAULT_TRANSLATION,
                         yaw_degrees: float = DEFAULT_YAW_DEGREES) -> Tensor:
    """Plan one endpoint command as ``(L,4,4,4)`` exact RGB camera spans.

    The command is applied once over the whole generated chunk.  Consequently
    the final RGB pose (``result[-1, -1]``) is exactly the user-requested
    endpoint; earlier RGB poses are the deterministic motion into it.
    """
    if latent_frames < 1:
        raise ValueError("a command must generate at least one latent frame")
    poses = command_frame_poses(
        c2w, command, 4 * latent_frames,
        translation=translation, yaw_degrees=yaw_degrees,
    )
    return poses.reshape(latent_frames, 4, 4, 4)


def metric_command_latent_spans(
        c2w: Tensor, command: str, latent_frames: int, *,
        anchor_median_depth_meters: float, gauge_scale: float,
        translation_meters: float = WALKING_MOTION.default_translation_meters,
        yaw_degrees: float = WALKING_MOTION.default_rotation_degrees,
        contract: WalkingMotionContract = WALKING_MOTION) -> Tensor:
    """Metric, deployment-facing form of :func:`command_latent_spans`.

    W/A/S/D convert the requested endpoint displacement; 10--20 cm is the
    walking UI's typical range, not a rejection boundary. Q/E behave likewise
    for the typical 5--10 degree range. The unused modality is still accepted
    so one stable call signature can serve every keyboard command.
    """
    if command in _TRANSLATION:
        translation = contract.scene_translation(
            translation_meters=translation_meters,
            anchor_median_depth_meters=anchor_median_depth_meters,
            gauge_scale=gauge_scale)
        rotation = contract.default_rotation_degrees
    elif command in _YAW_SIGN:
        translation = contract.scene_translation(
            translation_meters=contract.default_translation_meters,
            anchor_median_depth_meters=anchor_median_depth_meters,
            gauge_scale=gauge_scale)
        rotation = contract.validate_rotation(yaw_degrees)
    else:
        raise ValueError(f"unknown command {command!r}; expected one of {COMMANDS}")
    return command_latent_spans(
        c2w, command, latent_frames,
        translation=translation, yaw_degrees=rotation)
