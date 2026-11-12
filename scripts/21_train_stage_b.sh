#!/usr/bin/env bash
# Stage B: render memory added to the stage-A model (250 updates). Two paired arms, same rows / noise / seed:
#   token    render as 4 extra sequence slots (role 3) + (valid, conf) metadata          (~65 s / update, 4.6 h)
#   channel  render concatenated to the target tokens through a zero-initialised conv    (~47 s / update, 3.3 h)
# Usage: scripts/21_train_stage_b.sh token|channel [stage_a_receipt] [run_name]
set -euo pipefail; source "$(dirname "$0")/env.sh"; MODE="${1:?token|channel}"; C="$REPO/contract"; D="$C/data_contract"
A="${2:-$C/runs/a4_s2/ckpt/trainables_00350.json}"; RUN="${3:-b4_${MODE}_s2}"; mkdir -p "$C/logs"
cd "$REPO/core"
python "$C/train_gmp.py" --corrupt off --op_arm C --meta density --bank contract --render_mode "$MODE" --row_grad 1 --reject 50 \
  --exclude "$D/bad_streams_v3.json,$D/fast_streams_v4.json,$D/flag_streams_v4.json" --seed 20261113 --lr_new 0.0001 --init_receipt "$A" \
  --out "$C/runs/$RUN" --updates 250 --lr_backbone 2e-05 --wd 0.001 --clip 1.0 --text_dropout 0.15 --ckpt_every 50 --resume_every 50 \
  --min_free_gb 17.0 --prope 1 --max_hours 0.0 --keep_ckpts 0 --accum 16 --u_dist shift3 --lora_r 0 --render_p 0.9 --corpus 864 \
  --init memorizon50 --layout memorizon --scale_mult 2.0 --n_choices 2 --n_probs 1 --cam_flip_y 0 --max_bank 4 --warmup 50 \
  --render_cache "$C/render_cache_v4" $([ -f "$C/runs/$RUN/resume.json" ] && echo --resume 1) 2>&1 | tee -a "$C/logs/train_$RUN.log"
python "$C/reject_report.py" "$C/runs/$RUN"
