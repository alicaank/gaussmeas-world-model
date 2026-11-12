"""Cross-repo path shim.

gaussverse imports diffsynth (+ fb5b's CameraFiLM fallback) from FrameCrafter, and
VGGT from VGGT-World. The vendored memory projection stack and its perceptual metrics are
now VENDORED (gaussverse/memory/, gaussverse/bench/metrics.py), so growworld is no
longer a code dependency and is not put on sys.path. Import this module before any
cross-repo import:

    from gaussverse import _paths  # noqa: F401
    from gaussverse.memory.projection import unproject_points  # vendored from GEN3C
    from gaussverse.diffsynth.models.wan_video_dit import WanModel     # FrameCrafter

Note: run scripts as `python -m gaussverse.<pkg>.<mod>` so gaussverse's own packages
resolve as `gaussverse.*` (the gaussverse dir is never put on sys.path). `GROWWORLD`
is retained only as a DATA path (the VGGT checkpoint physically lives under it).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

GAUSSVERSE = Path(os.environ.get(
    "GAUSSVERSE_REPO", Path(__file__).resolve().parent)).resolve()
ALICAN = Path(os.environ.get(
    "GAUSSVERSE_WORKSPACE", GAUSSVERSE.parent)).resolve()
FRAMECRAFTER = Path(os.environ.get(
    "GAUSSVERSE_FRAMECRAFTER", ALICAN / "FrameCrafter")).resolve()
GROWWORLD = Path(os.environ.get(
    "GAUSSVERSE_GROWWORLD", ALICAN / "growworld")).resolve()
VGGT_WORLD = Path(os.environ.get(
    "GAUSSVERSE_VGGT_WORLD", ALICAN / "VGGT-World")).resolve()

DATA_ROOT = Path(os.environ.get(
    "GAUSSVERSE_DATA_ROOT", ALICAN / "data")).resolve()
DL3DV_ROOT = Path(os.environ.get(
    "GAUSSVERSE_DL3DV_ROOT", DATA_ROOT / "DL3DV-960P/1K")).resolve()
RE10K_ROOT = Path(os.environ.get(
    "GAUSSVERSE_RE10K_ROOT", ALICAN / "flash3d/data/RealEstate10K")).resolve()
WAN22_DIR = Path(os.environ.get(
    "GAUSSVERSE_WAN22_DIR",
    FRAMECRAFTER / "models/Wan-AI/Wan2.2-TI2V-5B")).resolve()
VGGT_CKPT_DIR = Path(os.environ.get(
    "GAUSSVERSE_VGGT_CKPT_DIR",
    GROWWORLD / "third_party/assets/vggt-1b")).resolve()

OUTPUTS = GAUSSVERSE / "outputs"
CACHE = GAUSSVERSE / "outputs/cache"


def resolve_manifest_scene_path(value: str | Path) -> Path:
    """Resolve immutable manifests after their data root moves workstation."""
    path = Path(value)
    if path.is_dir():
        return path.resolve()
    parts = path.parts
    if "data" in parts:
        relative = Path(*parts[parts.index("data") + 1:])
        relocated = DATA_ROOT / relative
        if relocated.is_dir():
            return relocated.resolve()
    raise FileNotFoundError(
        f"manifest scene is absent at {path}; set GAUSSVERSE_DATA_ROOT")

# growworld deliberately NOT on the path now (vendored). FrameCrafter for diffsynth;
# ALICAN so `import gaussverse` resolves when run as a module from the workspace root.
for _p in (ALICAN,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
