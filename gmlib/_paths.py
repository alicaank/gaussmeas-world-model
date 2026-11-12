"""Locations of external models and data for gmlib, read from GM_* environment variables (scripts/env.sh sets the
common ones). Only WAN22_DIR, DATA_ROOT and the workspace root are used by this repository's pipeline."""
from __future__ import annotations

import os
import sys
from pathlib import Path

GMLIB = Path(os.environ.get(
    "GM_LIB_DIR", Path(__file__).resolve().parent)).resolve()
WORKSPACE = Path(os.environ.get(
    "GM_WORKSPACE", GMLIB.parent)).resolve()
FRAMECRAFTER = Path(os.environ.get(
    "GM_FRAMECRAFTER", WORKSPACE / "FrameCrafter")).resolve()
GROWWORLD = Path(os.environ.get(
    "GM_GROWWORLD", WORKSPACE / "growworld")).resolve()
VGGT_WORLD = Path(os.environ.get(
    "GM_VGGT_WORLD", WORKSPACE / "VGGT-World")).resolve()

DATA_ROOT = Path(os.environ.get(
    "GM_DATA_ROOT", WORKSPACE / "data")).resolve()
DL3DV_ROOT = Path(os.environ.get(
    "GM_DL3DV_ROOT", DATA_ROOT / "DL3DV-960P/1K")).resolve()
RE10K_ROOT = Path(os.environ.get(
    "GM_RE10K_ROOT", WORKSPACE / "flash3d/data/RealEstate10K")).resolve()
WAN22_DIR = Path(os.environ.get(
    "GM_WAN22_DIR",
    FRAMECRAFTER / "models/Wan-AI/Wan2.2-TI2V-5B")).resolve()
VGGT_CKPT_DIR = Path(os.environ.get(
    "GM_VGGT_CKPT_DIR",
    GROWWORLD / "third_party/assets/vggt-1b")).resolve()

OUTPUTS = GMLIB / "outputs"
CACHE = GMLIB / "outputs/cache"


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
        f"manifest scene is absent at {path}; set GM_DATA_ROOT")

for _p in (WORKSPACE,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
