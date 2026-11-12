#!/usr/bin/env bash
# CPU-only checks (no weights, no data): outlier-row rejection, channel-render wrapper, PRoPE numerics. ~1 min.
set -euo pipefail; source "$(dirname "$0")/env.sh"; export CUDA_VISIBLE_DEVICES=''
(cd "$REPO/contract" && python tests_row_reject.py && python tests_channel_render.py)
(cd "$REPO/core" && python tests_numerics.py)
echo "CPU tests passed"
