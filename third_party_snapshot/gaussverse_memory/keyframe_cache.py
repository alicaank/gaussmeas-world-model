"""Keyframe cache — the external spatial memory ("remember the world").

Adapted from GEN3C Cache3D_Buffer but simplified to a per-keyframe store (no
premature global point-cloud fusion, per the Lyra-style design). Each keyframe
holds RGB, z-depth, pose (c2w), intrinsics, a per-pixel confidence/validity, and
provenance (observed / provisional / verified). Depth of new frames is scale/shift
aligned against overlapping cached anchors before insertion so the cache stays in
one consistent scale.

snapshot()/revert() support the A->B->C->B->A protocol (revisit must see the world
as it was, not a corrupted later state).
"""
from copy import deepcopy
from dataclasses import dataclass, field
import torch

from .projection import reliable_depth_mask

OBSERVED, PROVISIONAL, VERIFIED = "observed", "provisional", "verified"


@dataclass
class Keyframe:
    rgb: torch.Tensor      # (3,H,W) in [-1,1]
    depth: torch.Tensor    # (1,H,W) z-depth, >0 valid
    c2w: torch.Tensor      # (4,4)
    K: torch.Tensor        # (3,3)
    conf: torch.Tensor     # (1,H,W) reliability in [0,1]
    fid: int               # frame id / generation order
    provenance: str = PROVISIONAL
    # Autoregressive steps between this frame and the last real observation it derives
    # from. 0 for an observed frame. Measured on the 448x256 teacher: content generated
    # from generated history loses matchability fast (48% of SIFT inliers survive in the
    # near field vs 67% far), and its depth comes from VGGT run on those same invented
    # pixels -- so age is a usable proxy for how much a cached frame should be trusted.
    age: int = 0


@dataclass
class KeyframeCache:
    frames: list[Keyframe] = field(default_factory=list)
    _next_id: int = 0

    def __len__(self):
        return len(self.frames)

    def add(self, rgb, depth, c2w, K, conf=None, provenance=PROVISIONAL, fid=None, age=0):
        """Insert a keyframe. `conf` defaults to a range-based reliability mask."""
        rgb = rgb.detach().float()
        depth = depth.detach().float()
        if conf is None:
            conf = reliable_depth_mask(depth[None])[0].float()  # (1,H,W)
        if fid is None:
            fid = self._next_id
            self._next_id += 1
        self.frames.append(Keyframe(rgb, depth, c2w.float(), K.float(), conf.float(), fid,
                                    provenance, int(age)))
        return fid

    def stack(self, device=None):
        """Return batched tensors (N,...) for the whole cache."""
        dev = device or self.frames[0].rgb.device
        rgbs = torch.stack([f.rgb for f in self.frames]).to(dev)
        depths = torch.stack([f.depth for f in self.frames]).to(dev)
        c2ws = torch.stack([f.c2w for f in self.frames]).to(dev)
        Ks = torch.stack([f.K for f in self.frames]).to(dev)
        confs = torch.stack([f.conf for f in self.frames]).to(dev)
        return rgbs, depths, c2ws, Ks, confs

    def subset(self, idxs):
        c = KeyframeCache()
        c.frames = [self.frames[i] for i in idxs]
        c._next_id = self._next_id
        return c

    # --- snapshot / revert for revisit protocols ---
    def snapshot(self):
        return (deepcopy(self.frames), self._next_id)

    def revert(self, snap):
        self.frames, self._next_id = deepcopy(snap[0]), snap[1]
