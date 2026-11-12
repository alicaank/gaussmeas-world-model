#!/usr/bin/env bash
# Stage C (PLANNED, not yet run in the contract line): on-policy pool rows on top of the winning stage-B arm, so the model sees renders built from
# its own flawed frames and still produces the recorded content. Decided recipe (docs/lab_notes/RECIPE_contract_20261111.md, 12 Nov):
#   pool      generated with the stage-B model, depth-1 items only, pose + content threshold
#   dose      exposures per pool item = updates x 16 rows x pool share / pool items <= 1.5   (100 updates x 16 x .5 / 600 = 1.33)
#   data      far_windows_v4.json excluded on top of the three stream lists; row rejection on
#   stop      stray first-generation artefacts gone AND cross-boundary regeneration <= 1.3 (|ratio - 1|); otherwise the previous checkpoint
# What exists today is the pool generator and trainer path of the earlier GM' line (pool rows = 50 % of the render rows); the commands below are
# that path with the contract settings. A pool generator in the contract INFERENCE layout (density mask, contract bank) is still to be written,
# and train_gmp.py has no window-level exclusion yet. Set GM_ALLOW_UNVALIDATED=1 to run anyway.
# Usage: scripts/22_train_stage_c.sh pool|train <stage_b_receipt> [run_name]
set -euo pipefail; source "$(dirname "$0")/env.sh"; WHAT="${1:?pool|train}"; B="${2:?stage-B receipt}"; RUN="${3:-c4_s1}"; C="$REPO/contract"; D="$C/data_contract"
[ "${GM_ALLOW_UNVALIDATED:-0}" = 1 ] || { echo "stage C is planned, not validated: read the header of $0"; exit 3; }
cd "$REPO/core"
if [ "$WHAT" = pool ]; then
  python "$C/gen_pool_v3.py" --out "$C/pool_c4" --resplat v8 --receipt "$B" --n_streams 300 --seed 20261024
else
  python "$C/train_gmp.py" --corrupt off --op_arm meas --op_q .5 --op_seed 7 --pool "$C/pool_c4" --meta density --bank contract --row_grad 1 --reject 50 \
    --exclude "$D/bad_streams_v3.json,$D/fast_streams_v4.json,$D/flag_streams_v4.json" --seed 20261114 --lr_new 2e-05 --init_receipt "$B" \
    --out "$C/runs/$RUN" --updates 100 --lr_backbone 2e-05 --wd 0.001 --clip 1.0 --text_dropout 0.15 --ckpt_every 25 --resume_every 25 \
    --min_free_gb 17.0 --prope 1 --max_hours 0.0 --keep_ckpts 0 --accum 16 --u_dist shift3 --lora_r 0 --render_p 0.9 --corpus 864 \
    --init memorizon50 --layout memorizon --scale_mult 2.0 --n_choices 2 --n_probs 1 --cam_flip_y 0 --max_bank 4 --warmup 50 \
    --render_cache "$C/render_cache_v4" 2>&1 | tee -a "$C/logs/train_$RUN.log"
fi
