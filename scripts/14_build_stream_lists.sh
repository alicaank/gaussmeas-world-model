#!/usr/bin/env bash
# Stream exclusion lists from their rules (CPU): low-PSNR streams of the render cache and fast streams. Without arguments it only compares with the
# committed lists; `--write 1` rewrites them (use on new data). Needs contract/render_cache_v4/build_stats.jsonl and data_contract/windows.jsonl.
set -euo pipefail; source "$(dirname "$0")/env.sh"; python "$REPO/contract/data_contract/build_stream_lists.py" "$@"
