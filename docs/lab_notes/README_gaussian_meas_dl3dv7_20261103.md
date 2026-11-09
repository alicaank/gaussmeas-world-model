# Gaussian-meas model on the paper's 7-scene DL3DV cohort (queued 2026-11-03)

Runs our Memorizon-init Gaussian-memory checkpoint `fresh_teacher_20261021/run_op/meas` (250 on-policy updates on top of `run_v864/mem`).
It uses the scenes, plan and scorer that the closed-loop paper uses for Lyra-2 and Spatia.

- Batch: `continuous_eval_20261029/batch_continuous1009`. The scenes are 02267acf, a126efe8, 0238de14, c7418b7d, 61b27b61, 334ff324 and 924228bf. Scene 924228bf is stationary, so its position error is invalid.
- Rollout: 481 frames (30 calls of 16), 864x480, seed 7.
- Arms:
  - `baseline`: open loop.
  - `sham`: the observer runs and its output is discarded. Our render path is not repeatable, so this arm measures that noise floor.
  - `correction`: the paper's controller. It calls the shared `common.rejoin` and applies the memory-frame bridge; memory labels stay commanded.
  - `fprime`: measured writes plus the offset rejoin. This is the model's own F′ arm.
- Scoring: `runtime_repairs_20261031/revision2/score_reliability.py --style long --horizon 480 --model memorizon` (ViPE `vipe_native_map_mask_diagnostic_v3`, the `rejoin_v3` scorer of the paper's Lyra/Spatia rows). It reports horizons 160, 320 and 480.
- Inputs: `prepare.py` is GPU job 1. It encodes the umT5 captions, the VAE anchor latents and the DA3 metric anchor scale. Each step is first validated against training stream c2282d3f.
- Smoke test: a 3-call correction arm runs before the full set. A failure in the prep or smoke job stops the queue.
- CPU checks: `cpu_validation.json`. The bridge identity and the paper-rule equivalence hold to 1e-15. None of the scenes is in the model's corpus, and the checkpoint SHA matches.
- Queue: `run_queue.py` (PID in `status.json`).
  - It waits for v2 `status_A.json` and `status_B.json` to reach `finished`; phase B must be launched by hand after phase A.
  - It then takes the five shared queue locks, keeps 12 GiB free and never deletes anything.
- Cost: about 8–9 GPU-h. That is 28 arms at roughly 12–15 minutes each, plus 7 ViPE score jobs.
