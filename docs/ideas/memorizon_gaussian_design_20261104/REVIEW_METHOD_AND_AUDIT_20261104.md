# Review of METHOD_AND_TRAINING.md and PREFERENCE_AUDIT_20261105.md (4 Nov 2026, 23:30; re-evaluated 23:50 with the cited tables opened; session alican-55)

Reviewer context: I wrote the pilot code the audit inspects (PILOT_PLAN_CVPR_20261104.md, fresh_teacher_20261021/). The two documents
were written by another session between 22:25 and 22:46 the same evening. Verdict first, details after.

## Verdict

Adopt the contracts; do not adopt the training plan as written.

- **Adopt** (correct and cheap): P/U/M as three records with M-only memory labels; one versioned controller in collection and
  evaluation (the existing `rejoin_offset`, which is bit-identical to the paper's `common.rejoin`); recorded-target camera guard
  and honest labelling of pool rows as *recovery-to-recording* supervision; cache keys that include pose/gauge identity; an
  explicit single-view startup path; execution against U and tracking against P reported separately; failures kept in denominators.
- **Reject** for this deadline: a new training line from a **frozen** Memorizon base with adapter-only updates and a 6 + 24 + 6/6 h
  curriculum. Local evidence points the other way (below), it discards the one replicated gain (`run_op/meas`), and its 72 GPU-h
  cannot start before ~5 Nov 05:00 on a GPU that also has to run evaluations for 16 Nov.
- **Defer**: the GEAR-style restricted gathered reader with separately encoded source keyframes, and the reward-weighted FM
  pilot. The explicit execution-history embedding is worth one matched arm only if G / G+B finish by 9 Nov.

## Where the documents are right

1. `packed_model_corr.py` is an additive bias on dense attention, not a restricted reader (audit §Corrections, bullet 2). True. It
   is a soft preference the model can learn to use, and it keeps native attention intact; at gain → ∞ it concentrates on the
   corresponded keys. It is not GEAR and must not be described as such. The "GPU scalability" worry is moot at our sequence
   length: the model already materialises a dense (S,S) fp32 bias (S ≈ 8.5k, 290 MB); the correspondence term adds one more.
2. Pool rows query the recorded P, not the deployed U (bullet 3). True by construction and intended: the row supervises "produce
   the recorded chunk at its recorded cameras from a measured prefix". `rejoin_offset` ends every call at the planned end, so U
   and P agree at the chunk end and differ at interior latents by (1−a)·dt. Items store M, U and P ends (`meas_end`, `cmd_end`,
   `plan_end`); label this *recovery-to-recording*, and call the pool *replay* (one frozen producer), not on-policy training.
3. Single-view startup (bullet 6). Real bug: 2Xplat's normaliser divides by the camera extent, which is 0 for one view → NaN.
   Fixed tonight in `twoxplat_memory.py` (single view keeps the DA3 scale with the camera at the origin) and `profile_2x.py` now
   measures the call-1 render (coverage, PSNR) explicitly. Whether MVP gives a usable 1-view scene is a go-gate question.
4. Cache identity (bullet 5): the key now includes the gauge poses the scene was registered with, not only the frame indices.
5. `scale_mult = 2` is a lineage convention (bullet 7): agreed; every script reads `cam_scale` from the stream and lineage.
6. **GEAR's restricted-vs-dense ablation is verified (re-evaluation 23:50, arXiv HTML Table 2, DL3DV-Eval, same GCA module and
   parameter count):** dense attention over retrieved history LPIPS .5995 / ATE .2924 vs restricted .4459 / .0436; disturbance
   test (keep 2–3 of 9 valid matches, shift the rest 64–128 px) LPIPS .4513. Single runs, Wan2.1-I2V-14B frozen + LoRA r32 +
   262M GCA, 32 GPUs × 10k iters × batch 32, framewise VAE encoding of history, COMMANDED cameras. This is stronger than I
   allowed in the first pass: the audit's preference for a restricted reader is the best-supported design choice in both documents.
   Consequence: `packed_model_corr.py` now has `mode='restricted'` — a new narrow PRoPE branch every 2nd block whose target queries
   attend only to their corresponded anchor/bank/recent tokens (zero-init out_proj, output masked to tokens with a read); native
   attention untouched; identity at init; CPU-tested. Trainer arm `--op_arm gr`, evaluator auto-detects it. The soft bias (`gb`)
   stays as the cheap variant, and `gd` (the same reader without restriction) is the capacity-matched control, so `gr` vs `gd`
   isolates restriction the way GEAR's ablation does; `gr` vs `gb` does not (different capacity). Remaining gap vs GEAR: our bank
   tokens are temporal (4 frames) and addressed by the latent-end frame; GEAR encodes history framewise. Measure the
   correspondence hit rate on recorded data before claiming more.
8. **Evaluator bug (found by the other session, 23:22): the rollout never passed `corr`, so `gb`/`gr` checkpoints would have been
   evaluated as plain models.** Fixed: `LoopRoller2X.pack` builds the correspondence matrix from the scene (sources at measured
   poses, targets at the command's latent ends with rendered depth, same construction as the rows) and `denoise`/`velocity` pass it;
   `pk['corr_nnz']` is logged per call so an empty read is visible.
7. **Robust Dreamer Table 3 verified (ScanNet):** RGB-space memory 15.92 / .524 / LPIPS .402 / FID 24.85 vs full latent memory
   16.89 / .651 / .351 / 16.82 — RGB memory is worse, as the audit says; but RD writes memory at poses from a pose token
   PREDICTED from the frame (estimated, not commanded). My earlier "RD places memory at commanded cameras" note was wrong and the
   novelty table in RESEARCH_AND_DESIGN.md should not lean on it. Our delta vs RD: an explicit observer feeding a command
   correction, training on the states that loop produces, and global re-reconstruction with the observer as reconstructor.

## Where the documents are wrong or unsupported

1. ~~"Endpoint measurements are not full-rate measurements" has no effect on the model.~~ **Retracted (00:10, after the other
   session's verification):** `packed_rays` packs all four RGB cameras of each latent into the 24 Plücker channels, so the interior
   poses do enter the trained camera embedding (PRoPE uses the endpoint only). Pool rows mixed recorded interiors with measured
   ends while inference writes interpolated-measured interiors (`interp_rgb_poses`). Fixed in `rows_g.build_row_g`: interiors are
   now interpolated between measured ends exactly as the roller writes them (test added). The inherited `meas` arm has the same
   inconsistency; its result stands as reported but the G arms no longer share it.
2. **Frozen base + adapters (§2, §6).** Not refuted in general — GEAR did exactly frozen base + LoRA r32 + reader at ~320k
   samples — but at our budget (≈2k rows per arm) the closest local tests went the other way: zero-init widened geometric channels
   (`feat`) lost content 0/10 LPIPS; our LoRA/Plücker/roles stack lost to the plain recipe 10/10 at 600 steps; the two-seed gains
   came from training the Memorizon trainable set with memory rows. Continuing from `run_op/meas` is the budget-appropriate
   choice; adapter-only from a frozen base is a restart with no local evidence at this scale.
3. **Clean warm-up 6 h + common mixed FM 24 h from scratch (§9).** `run_op/meas` already is 250 memory updates + 250 measured
   on-policy updates on top of plain-350. Restarting throws away ~12 GPU-h of replicated progress and the ablations (C/cmd/meas,
   mcc/cmc/ccm, feat, R2) that make the measured-label claim attributable.
4. (withdrawn 4 Nov 23:40 — user: evaluation time and disk are not limits.) Plan by stages and gates; GPU-hours are not the
   binding constraint, the 16 Nov deadline is.
5. **Execution-history embedding as the primary hypothesis (§1, §4.3).** It is the least supported piece: no local or published
   evidence, and the measured prefix cameras + issued U already carry e_exec implicitly. As a matched arm it is fine; as the
   paper's main claim it is a bet with 12 days left. The replicated results (memory 8–9/10, measured labels 8/10) are the claim.
6. **"Native temporal tokens cannot be addressed as points" (audit row 3).** Correct in principle; in our rows the per-slot camera
   is the latent end and the token depth is the end frame's depth, i.e. a documented approximation. The remedy is to measure its
   effect (correspondence hit rate on recorded data is cheap), not to add a second encoder before the deadline.
7. Minor: both files are dated 5 November but were written on 4 November; "Memorizon's original latent bank retained" is already
   how every arm here works (`layout='memorizon'`), so it is not a design change.

## Reconciled plan (what runs)

1. 5 Nov: go-gate profile (`profile_2x.py`): VRAM at 8/16/32 views, measured-pose error on recorded prefixes, 1-view startup,
   render PSNR/LPIPS/coverage vs ReSplat. Fail → ReSplat stays the structural read; 2Xplat still supplies poses + visibility bank.
2. Pool round 2 with the loop active (`gen_onpolicy2.py`, controller = `rejoin_offset`), recorded-window renders, then
   `train_g --op_arm g`, `gr` (restricted reader, primary source-read arm after the GEAR check) and `gb` (soft bias) from
   `run_op/meas` (same rows, same op_rng). Controls: `meas` (round 1) and R2meas.
3. Evals: 160 f, 10 dev videos, seeds 7/107, open + F′; DL3DV-7 481 f with the paper's ViPE scorer.
4. The execution-history arm (MLP on {e_plan, e_exec, validity, age} → modulation of the corr gain and render metadata
   projection) with its matched no-feedback twin, and the R2meas control, scheduled after G / G+B (user 4 Nov: eval time is not
   a limit, so all arms run; order by information value).
5. Not before the deadline: restricted gathered reader, separately encoded source keyframes, reward-weighted FM, learned updater.

Everything above is CPU-tested where possible; nothing was launched on the GPU tonight (the v2 A-retry queue holds it until ~05:00).
