#!/usr/bin/env bash
# CPU measurements on rollouts made with scripts/30_rollout.sh. Rollout directories are given as name=<dir with {scene}>, e.g.
#   scripts/40_metrics.sh fast   b_token=$PWD/contract/runs_eval/b_token_long/{scene}/baseline       (20-call fast metrics: regeneration, snap, DINO, sharpness, luminance)
#   scripts/40_metrics.sh first  b_token=...                                                           (calls 1-3 LPIPS + vertical / horizontal scale ratio)
#   scripts/40_metrics.sh return b_token=<out-and-back dirs>                                           (out-and-back self-consistency)
# Regeneration is a ratio to the recording: report |ratio - 1| (below 1 = standstill, above 1 = redrawing).
set -euo pipefail; source "$(dirname "$0")/env.sh"; WHAT="${1:?fast|first|return}"; shift; export CUDA_VISIBLE_DEVICES=''; cd "$REPO/contract"
case "$WHAT" in fast) python fast_metrics.py "$@" --n 321;; first) python gate_bp.py "$@" --out gate;; return) python diag_return.py "$@";; *) echo "fast|first|return"; exit 2;; esac
