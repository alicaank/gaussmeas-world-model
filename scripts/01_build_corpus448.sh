#!/usr/bin/env bash
# The "448 corpus" (foundation/dataset_v3): scene / stream index of DL3DV, split, and per stream the VAE latents at 448x256, the anchor gauge
# (DA3 + MoGe), a caption (Qwen3-VL-2B) and its umT5 embedding. The 864 corpus reuses its stream windows, gauge and text.
#   index   CPU   foundation/prepare_all_scenes.py: every transforms.json under $GM_WORKSPACE/data/DL3DV-480P and DL3DV-960P -> streams = contiguous
#                 runs of >= 17 valid frames; 128 development scenes by stratified hashing (seed 20261009)
#   plan    CPU   render recipes (only the geometry stage needs them)
#   stages  GPU   vae, anchor, caption, text (geometry is not needed for the 864 corpus), then the audit
# STATUS: this is the code as it ran in October 2026; it has NOT been re-run from this repository layout. prepare_all_scenes.py also reads three
# legacy split inputs (docs/PIPELINE.md, "Known gaps"). Usage: scripts/01_build_corpus448.sh index|plan|vae|anchor|caption|text|audit [options]
set -euo pipefail; source "$(dirname "$0")/env.sh"; F="$REPO/foundation"; STAGE="${1:?index|plan|vae|anchor|caption|text|audit}"; shift
case "$STAGE" in
  index) python "$F/prepare_all_scenes.py" "$@";;
  audit) python "$F/audit_all_scenes.py" --verify-components --smoke-pairs 8 "$@";;
  plan|vae|anchor|caption|text|geometry) python "$F/materialize_all_scenes.py" "$STAGE" "$@";;
  *) echo "unknown stage $STAGE"; exit 2;;
esac
