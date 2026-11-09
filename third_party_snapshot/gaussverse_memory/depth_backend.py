"""Per-frame depth backend for the spatial memory.

Primary backend: Depth-Anything-V2 (vitl), which outputs affine-invariant
inverse depth (disparity, larger = nearer). Callers convert it to z-depth with
their calibrated affine-in-disparity parameters:

    z = 1 / (a * disp + b)

VGGT can be swapped in behind the same `predict_disparity` interface as an
independent benchmark verifier; it is intentionally NOT used in the inner loop.
"""
import sys
import numpy as np
import torch

_DAV2_ROOT = "/home/hj/Alican/Depth-Anything-V2"
_DAV2_CKPT = f"{_DAV2_ROOT}/checkpoints/depth_anything_v2_vitl.pth"
_CFG = {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]}


class DepthAnythingBackend:
    def __init__(self, device="cuda", ckpt=_DAV2_CKPT):
        if _DAV2_ROOT not in sys.path:
            sys.path.insert(0, _DAV2_ROOT)
        from depth_anything_v2.dpt import DepthAnythingV2

        self.device = device
        self.model = DepthAnythingV2(**_CFG)
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.model = self.model.to(device).eval()

    @torch.no_grad()
    def predict_disparity(self, rgb_uint8_bgr: np.ndarray) -> np.ndarray:
        """Affine-invariant inverse depth (H,W) float32; larger = nearer.

        Input is a HxWx3 uint8 BGR image (cv2 convention, as DA-V2 expects)."""
        d = self.model.infer_image(rgb_uint8_bgr)  # (H,W) float32
        return d.astype(np.float32)


def disparity_to_depth(disp: torch.Tensor, a: float, b: float, dmax=1e3) -> torch.Tensor:
    """z = 1/(a*disp + b), clamped positive and to dmax. disp: (...,H,W)."""
    denom = a * disp + b
    denom = torch.clamp(denom, min=1.0 / dmax)
    return torch.clamp(1.0 / denom, max=dmax)
