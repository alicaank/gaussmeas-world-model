"""Plücker ray maps for the Wan2.2 latent grid.

Math is copied from FrameCrafter/camera_utils.py (proven in fb5b). The one
behavioral difference: Wan2.2's VAE has spatial stride 16, so rays are
PixelUnshuffled by 16 straight to the latent grid (6·16² = 1536 channels),
keeping all sub-patch ray gradients. Avg-pooled 6-ch rays were the FC-1 v1
weak-camera failure mode — do not "compact" the rays below the pixel level.

Conventions: OpenCV camera frame (x-right, y-down, z-forward); poses handled
as w2c externally, c2w only inside raymap construction.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

VAE_STRIDE = 16
RAY_CHANNELS = 6 * VAE_STRIDE * VAE_STRIDE  # 1536


def camera_to_raymap(Ks: Tensor, camtoworlds: Tensor, height: int, width: int) -> Tensor:
    """(..., 3, 3) intrinsics + (..., 4, 4) c2w -> (..., H, W, 6) [origins, dirs]."""
    assert Ks.shape[-2:] == (3, 3) and camtoworlds.shape[-2:] == (4, 4)
    dtype = Ks.dtype
    x, y = torch.meshgrid(
        torch.arange(width, device=Ks.device),
        torch.arange(height, device=Ks.device),
        indexing="xy",
    )
    coords = torch.stack([x + 0.5, y + 0.5, torch.ones_like(x)], dim=-1).to(dtype)
    dirs = torch.einsum("...ij,...hwj->...hwi", Ks.float().inverse().to(dtype), coords)
    dirs = torch.einsum("...ij,...hwj->...hwi", camtoworlds[..., :3, :3], dirs)
    dirs = F.normalize(dirs, p=2, dim=-1)
    origins = torch.broadcast_to(camtoworlds[..., None, None, :3, -1], dirs.shape)
    return torch.cat([origins, dirs], dim=-1)


def raymap_to_plucker(raymap: Tensor) -> Tensor:
    """(..., H, W, 6) [origins, dirs] -> (..., H, W, 6) [dirs, o x d]."""
    assert raymap.shape[-1] == 6
    origins, dirs = torch.split(raymap, [3, 3], dim=-1)
    dirs = F.normalize(dirs, p=2, dim=-1)
    return torch.cat([dirs, torch.cross(origins, dirs, dim=-1)], dim=-1)


def plucker_latent_rays(
    c2w: Tensor, K: Tensor, height: int, width: int, stride: int = VAE_STRIDE
) -> Tensor:
    """Full-res Plücker map PixelUnshuffled onto the latent grid.

    Args:
        c2w: (N, 4, 4) camera-to-world (OpenCV), already normalized/scaled.
        K:   (N, 3, 3) or (3, 3) intrinsics at (height, width) resolution.
    Returns:
        (N, 6·stride², height/stride, width/stride)
    """
    assert height % stride == 0 and width % stride == 0, (height, width, stride)
    if K.dim() == 2:
        K = K[None].expand(c2w.shape[0], 3, 3)
    plucker = raymap_to_plucker(camera_to_raymap(K, c2w, height, width))
    return F.pixel_unshuffle(plucker.permute(0, 3, 1, 2), stride)


def opengl_c2w_to_opencv_w2c(c2w_gl: Tensor) -> Tensor:
    """transforms.json stores OpenGL c2w; flip y/z rows of w2c to land in OpenCV.

    Same flip as FrameCrafter dataset.py lines 436-438.
    """
    w2c = torch.linalg.inv(c2w_gl)
    w2c = w2c.clone()
    w2c[..., [1, 2], :] *= -1
    return w2c
