"""DL3DV-960P scene access: cameras (persistent scene scale) + frames + rays.

Scene layout: <root>/<hash>/images_4/frame_%05d.png + transforms.json
(OpenGL c2w per frame, frames sorted by file_path; intrinsics for the ORIGINAL
960P resolution — rescaled here with FrameCrafter's resize-to-cover +
center-crop convention so K matches the processed image exactly).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from gmlib import _paths
from gmlib.camera.normalization import compute_scene_scale, relative_to_target
from gmlib.camera.rays import opengl_c2w_to_opencv_w2c, plucker_latent_rays


# DL3DV ships several downsample levels: the 960P release has images_4 (960x540),
# the 480P release has images_8 (480x270). Intrinsics are rescaled from the ORIGINAL
# w,h in transforms.json, so any level works -- only the directory name differs.
IMAGE_DIR_CANDIDATES = ("images_4", "images_8", "images", "images_2")


def scene_image_dir(scene_dir: Path) -> Path | None:
    """The downsample level this scene actually ships, or None if it has no images."""
    for d in IMAGE_DIR_CANDIDATES:
        if (scene_dir / d).is_dir():
            return scene_dir / d
    return None


def list_scenes(root: Path | str = _paths.DL3DV_ROOT) -> list[Path]:
    root = Path(root)
    return sorted(
        p for p in root.iterdir()
        if (p / "transforms.json").exists() and scene_image_dir(p) is not None
    )


def scale_intrinsics(fx, fy, cx, cy, orig_w, orig_h, target_w, target_h) -> np.ndarray:
    """Resize-to-cover + center-crop intrinsics (FrameCrafter _scale_intrinsics).

    Crop offsets use floor division to match `crop_and_resize` EXACTLY. An image can
    only be cut at integer pixels, so the intrinsics must follow the image, not the
    other way round. The fractional `/ 2.0` this replaces (inherited from upstream
    FrameCrafter, which has the same mismatch against torchvision's integer
    center_crop) put the principal point half a pixel left of where the pixels
    actually are: every 16:9 source resizes to 341x192 or 683x384, giving crop_x
    10.5 vs the image's 10, and 21.5 vs 21. Small, but the whole contract here is
    exact cameras, and a constant cx bias is exactly the kind of error that
    reprojection and epipolar metrics absorb as apparent pose error.
    """
    scale = max(target_w / orig_w, target_h / orig_h)
    crop_x = float((round(orig_w * scale) - target_w) // 2)
    crop_y = float((round(orig_h * scale) - target_h) // 2)
    return np.array([
        [fx * scale, 0, cx * scale - crop_x],
        [0, fy * scale, cy * scale - crop_y],
        [0, 0, 1],
    ], dtype=np.float32)


def crop_and_resize(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Image-side counterpart of scale_intrinsics."""
    ow, oh = img.size
    scale = max(target_w / ow, target_h / oh)
    rw, rh = round(ow * scale), round(oh * scale)
    img = img.resize((rw, rh), Image.LANCZOS)
    left = (rw - target_w) // 2
    top = (rh - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


@dataclass
class Scene:
    """One DL3DV scene at a fixed working resolution."""

    scene_dir: Path
    height: int
    width: int
    w2c: torch.Tensor = field(init=False)      # (N, 4, 4) OpenCV
    c2w: torch.Tensor = field(init=False)      # (N, 4, 4) OpenCV
    K: torch.Tensor = field(init=False)        # (3, 3) at (height, width)
    image_paths: list[Path] = field(init=False)
    s_scene: float = field(init=False)

    def __post_init__(self):
        self.scene_dir = Path(self.scene_dir)
        td = json.loads((self.scene_dir / "transforms.json").read_text())
        frames = sorted(td["frames"], key=lambda f: f["file_path"])
        img_dir = scene_image_dir(self.scene_dir)
        assert img_dir is not None, f"no image dir in {self.scene_dir}"
        paths, mats = [], []
        for fd in frames:
            p = img_dir / Path(fd["file_path"]).name
            if p.exists():
                paths.append(p)
                mats.append(fd["transform_matrix"])
        assert len(paths) >= 2, f"scene too small: {self.scene_dir}"
        self.image_paths = paths
        c2w_gl = torch.tensor(np.array(mats, dtype=np.float32))
        self.w2c = opengl_c2w_to_opencv_w2c(c2w_gl)
        self.c2w = torch.linalg.inv(self.w2c)
        self.K = torch.from_numpy(scale_intrinsics(
            td["fl_x"], td["fl_y"], td["cx"], td["cy"],
            td["w"], td["h"], self.width, self.height,
        ))
        self.s_scene = compute_scene_scale(self.c2w)

    def __len__(self) -> int:
        return len(self.image_paths)

    @property
    def name(self) -> str:
        return self.scene_dir.name

    def load_frame(self, i: int) -> torch.Tensor:
        """(3, H, W) float32 in [-1, 1]."""
        img = Image.open(self.image_paths[i]).convert("RGB")
        img = crop_and_resize(img, self.width, self.height)
        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def load_frame_uint8(self, i: int) -> np.ndarray:
        """(H, W, 3) uint8 RGB (for depth backends / metrics on images)."""
        img = Image.open(self.image_paths[i]).convert("RGB")
        img = crop_and_resize(img, self.width, self.height)
        return np.asarray(img, dtype=np.uint8)

    def make_sample(self, ctx_indices: list[int], tgt_index: int,
                    *, load_frames: bool = True) -> dict:
        """Model-ready sample: frames + target-relative latent rays.

        Returns:
            frames: (M+1, 3, H, W) in [-1,1], context first, target LAST.
            rays:   (1, 1536, M+1, H/16, W/16) target-relative Plücker.
        """
        order = list(ctx_indices) + [tgt_index]
        frames = (torch.stack([self.load_frame(i) for i in order])
                  if load_frames else None)
        _, c2w_rel = relative_to_target(self.w2c[order], self.s_scene, target_index=-1)
        rays = plucker_latent_rays(c2w_rel.float(), self.K, self.height, self.width)
        rays = rays[None].permute(0, 2, 1, 3, 4)  # (1, C, F, h, w)
        return {"frames": frames, "rays": rays, "c2w_relative": c2w_rel,
                "K": self.K, "ctx": list(ctx_indices),
                "target": tgt_index, "scene": self.name}

    def make_camera_sample(self, ctx_indices: list[int], tgt_index: int) -> dict:
        """Target-relative cameras/rays without reopening images already held in a cache."""
        return self.make_sample(ctx_indices, tgt_index, load_frames=False)

    def make_chunk_sample(self, ctx_indices: list[int], target_indices: list[int],
                          *, load_frames: bool = True) -> dict:
        """Model-ready recent context plus a jointly denoised future latent chunk.

        All known camera rays share the final future camera as their coordinate frame.
        This avoids a contradictory per-target origin inside one self-attention sequence;
        target poses are commanded inputs, so using the last one as reference leaks no
        target appearance.
        """
        if not ctx_indices or not target_indices:
            raise ValueError("chunk samples require context and target frames")
        if max(ctx_indices) >= min(target_indices) or target_indices != sorted(target_indices):
            raise ValueError("chunk targets must be ordered strictly after all context")
        order = list(ctx_indices) + list(target_indices)
        frames = (torch.stack([self.load_frame(i) for i in order])
                  if load_frames else None)
        _, c2w_rel = relative_to_target(self.w2c[order], self.s_scene, target_index=-1)
        rays = plucker_latent_rays(c2w_rel.float(), self.K, self.height, self.width)
        rays = rays[None].permute(0, 2, 1, 3, 4)
        return {"frames": frames, "rays": rays, "c2w_relative": c2w_rel,
                "K": self.K, "ctx": list(ctx_indices),
                "targets": list(target_indices), "scene": self.name}
