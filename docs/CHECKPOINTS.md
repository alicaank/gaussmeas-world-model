# Checkpoints (not in git)

A checkpoint is a receipt `trainables_<update>.json` next to a `.pt` of the trainable tensors; the receipt records the
sha256, update, config and lineage. Base: `core/memorizon50/` = `Luffuly/memorizon-1280x704-50step` (Wan2.2-TI2V-5B),
fingerprint `memorizon50:e66808cdaa79`.

Contract line (November 2026; `docs/PIPELINE.md`):

| Name | Receipt | sha256 (prefix) | Recipe |
|---|---|---|---|
| stage A | `contract/runs/a4_s2/ckpt/trainables_00350.json` | 3b9e0400 | no render, contract windows and bank, 350 updates from Memorizon-50 |
| stage B, token arm | `contract/runs/b4_token_s2/ckpt/trainables_00250.json` | be3bda8f | render as tokens + density mask, row rejection, 250 updates from stage A |
| stage B, channel arm | `contract/runs/b4_channel_s2/ckpt/` | (training) | render through input channels, same rows / noise / seed |

Earlier line, referenced in `docs/RESULTS.md`:

| Name | Receipt (original workspace) | sha256 (prefix) | Recipe |
|---|---|---|---|
| plain-350 | `fresh_teacher/run_plain350/ckpt/trainables_00350.json` | 8170d9a4 | Memorizon recipe, no render, 350 updates |
| Gauss-meas (GM) | `fresh_teacher/run_op/meas/ckpt/trainables_00250.json` | 1dc0ba13 | 50 % on-policy measured-pose rows on top of a render stage; the "Ours" rows of the cohort table |
| B′ | `robust_native/runs/bp3_s1/ckpt/trainables_00250.json` | | plain-350 + corrected render, 250 updates, no on-policy stage |
| GM′ | `robust_native/runs/gmp3_s1/ckpt/trainables_00250.json` | | B′ + on-policy pool (600 items, 50 % of the render rows), 250 updates |
| gmc3 | `robust_native/runs/gmc3_s1/ckpt/trainables_00250.json` | | GM′ + 250 more updates on the same pool, no corruption (control) |
