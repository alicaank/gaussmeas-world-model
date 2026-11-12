#!/usr/bin/env bash
# Stage A: the Memorizon-50 teacher continued WITHOUT render on the contract windows (350 updates, ~4.6 h on one RTX 5090).
# Usage: scripts/20_train_stage_a.sh [run_name]        -> contract/runs/<run_name>/ckpt/trainables_00350.{json,pt}
set -euo pipefail; source "$(dirname "$0")/env.sh"; RUN="${1:-a4_s2}"; C="$REPO/contract"; mkdir -p "$C/logs"
cd "$REPO/core"
python "$C/train_a4.py" --out "$C/runs/$RUN" --updates 350 --seed 20261112 --lr_backbone 2e-05 --lr_new 0.0001 --wd 0.001 --clip 1.0 --text_dropout 0.15 \
  --ckpt_every 50 --resume_every 50 --min_free_gb 17.0 --prope 1 --max_hours 0.0 --keep_ckpts 0 --accum 16 --u_dist shift3 --lora_r 0 \
  --render_p 0.0 --corpus 864 --init memorizon50 --layout memorizon --scale_mult 2.0 --n_choices 2 --n_probs 1 --cam_flip_y 0 --max_bank 4 \
  --warmup 50 --index "$C/data_contract/index_v4.json" --exclude "$C/data_contract/bad_streams_v3.json" \
  $([ -f "$C/runs/$RUN/resume.json" ] && echo --resume 1) 2>&1 | tee -a "$C/logs/train_$RUN.log"
