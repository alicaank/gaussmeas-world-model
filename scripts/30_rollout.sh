#!/usr/bin/env bash
# Autoregressive rollout of a checkpoint on an evaluation scene, in the layout the model was trained for.
# Usage: scripts/30_rollout.sh <receipt.json> <scene> <out_dir> [fast|long|return] [extra run_arm_ckpt.py args]
#   fast    every 4th plan pose, 7 calls (112 generated frames): test speed          long   plan as recorded, 30 calls (480 frames)
#   return  out-and-back plan, 30 calls (self-consistency: contract/diag_return.py)
# A model trained WITH render (stage B / C) runs with the contract inference: corrected 8-view ReSplat render, support-density mask, contract bank.
# A stage-A model has no render rows: add `--no_render 1 --frontier off` after the mode (and drop the render flags by passing NORENDER=1).
set -euo pipefail; source "$(dirname "$0")/env.sh"; R="${1:?receipt}"; SC="${2:?scene}"; OUT="${3:?out dir}"; MODE="${4:-fast}"; shift $(( $# < 4 ? $# : 4 ))
case "$MODE" in fast) M=(--calls 7 --stride 4);; long) M=(--calls 30 --stride 1);; return) M=(--calls 30 --stride 1 --return_traj 1);; *) echo "mode: fast|long|return"; exit 2;; esac
if [ "${NORENDER:-0}" = 1 ]; then L=(--no_render 1 --bank contract --frontier off); else L=(--resplat v8 --frontier density --bank contract); fi
cd "$REPO/core"
python "$REPO/contract/sota_eval/run_arm_ckpt.py" --scene "$SC" --arm baseline --seed "${SEED:-7}" --out "$OUT" --receipt "$R" "${M[@]}" "${L[@]}" "$@"
