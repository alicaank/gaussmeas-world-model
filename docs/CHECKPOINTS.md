# Checkpoints (not in git)

All receipts are `trainables_*.json` next to a `.pt` of the trainable tensors; the receipt records sha256, update, config and lineage.
Base: `fresh_teacher_20261021/memorizon50/` = Luffuly/memorizon-1280x704-50step (Wan2.2-TI2V-5B), fingerprint `memorizon50:e66808cdaa79`.

| Name | Receipt | sha256 (prefix) | Recipe |
|---|---|---|---|
| plain-350 | `fresh_teacher_20261021/run_plain350/ckpt/trainables_00350.json` | 8170d9a4 | stage A, no render, 350 updates from Memorizon-50 |
| mem (stage B) | `fresh_teacher_20261021/run_v864/mem/ckpt/trainables_00250.json` | 2b6d2ef7 | cached renders p .75, 250 updates from plain-350 |
| **Gauss-meas (GM)** | `fresh_teacher_20261021/run_op/meas/ckpt/trainables_00250.json` | 1dc0ba13 | 50 % on-policy measured-pose rows, 250 updates from mem; the "Ours" rows |
| GM-B | `robust_native_20261108/runs/gm_b_s1/ckpt/trainables_00250.json` | b7bd6e5c | GM + context corruption, 250 updates, seed 20261108 |
| GM-F | `robust_native_20261108/runs/gm_f_s1/ckpt/trainables_00250.json` | 184009fd | GM-B recipe + frontier mask + strip dropout, 250 updates |

Paths are relative to `/home/hj/Alican/analysis/plan_base_restart_20261008/` and `/home/hj/Alican/analysis/`.
Protected legacy models (AGENTS.md) are not part of this line.
