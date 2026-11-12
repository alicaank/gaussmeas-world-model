#!/usr/bin/env bash
# Model-free dry run of a stage-B training's row sequence (CPU, ~1 min for 20 updates): draws the rows exactly as the trainer would and checks that every
# render-cache window exists. Run it before spending GPU hours. Usage: scripts/19_dry_run.sh token|channel [updates]
set -euo pipefail; source "$(dirname "$0")/env.sh"; MODE="${1:?token|channel}"; N="${2:-20}"; C="$REPO/contract"; D="$C/data_contract"; export CUDA_VISIBLE_DEVICES=''
cd "$REPO/core"
python "$C/dry_run_sampler.py" --corrupt off --op_arm C --meta density --bank contract --render_mode "$MODE" --row_grad 1 --reject 50 \
  --exclude "$D/bad_streams_v3.json,$D/fast_streams_v4.json,$D/flag_streams_v4.json" --seed 20261113 --lr_new 0.0001 \
  --init_receipt "$C/runs/a4_s2/ckpt/trainables_00350.json" --out "$C/runs/_dry_$MODE" --updates "$N" --lr_backbone 2e-05 --wd 0.001 --clip 1.0 \
  --text_dropout 0.15 --ckpt_every 50 --resume_every 50 --min_free_gb 17.0 --prope 1 --max_hours 0.0 --keep_ckpts 0 --accum 16 --u_dist shift3 \
  --lora_r 0 --render_p 0.9 --corpus 864 --init memorizon50 --layout memorizon --scale_mult 2.0 --n_choices 2 --n_probs 1 --cam_flip_y 0 \
  --max_bank 4 --warmup 50 --render_cache "$C/render_cache_v4"
