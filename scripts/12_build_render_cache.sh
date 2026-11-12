#!/usr/bin/env bash
# Render cache v4 (GPU, resumable; 4,422 windows): ReSplat render of each window's 16 target views from the anchor, the two last context latents and
# the first bank frame, VAE-encoded, with raw (valid, conf) and support-density metadata; build_stats.jsonl has PSNR and density-dropped per window.
# Needs: core/data864, contract/data_contract/index_v4.json, ReSplat + DA3 (scripts/env.sh). Usage: scripts/12_build_render_cache.sh [--limit N]
set -euo pipefail; source "$(dirname "$0")/env.sh"; cd "$REPO/contract"; mkdir -p logs
python build_render_cache_v4.py "$@" 2>&1 | tee -a logs/cache_v4.log
