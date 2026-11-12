#!/usr/bin/env bash
# 864x480 frames (CPU + network): for training scenes of the 448 corpus with a stream of >= 41 latents (seeded order), download the scene zip from
# the Hugging Face dataset DL3DV/DL3DV-ALL-960P, centre-crop 960x540 to the 864:480 aspect, resize (LANCZOS), save JPEG q95, keep transforms.json.
# -> $GM_DATA_ROOT/DL3DV-960P-864x480/<subset>/<scene>/{frames_864x480/, transforms.json, crop.json, done}. Needs HF access to the DL3DV dataset.
# Our run: 1000 scenes selected, 958 completed before the disk floor. Usage: scripts/03_download_frames_864.sh [--n 1000] [--min-free-gb 30]
set -euo pipefail; source "$(dirname "$0")/env.sh"
[ $# -gt 0 ] || set -- --n 1000 --min-free-gb 30
python "$REPO/foundation/dl3dv960/stream_960p.py" --out "$GM_DATA_ROOT/DL3DV-960P-864x480" "$@"
