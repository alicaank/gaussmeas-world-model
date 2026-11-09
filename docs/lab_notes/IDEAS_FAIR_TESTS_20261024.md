# Fair tests of the status-page ideas (user 10-24: "continue to try ideas we have on artifact page, train them fairly")

Common protocol (fixed before any result):
- Base model: Memorizon recipe + 3D memory (run_v864/mem, 250 updates) unless stated. Eval: closed_loop_864, 160 frames (10 calls),
  the 10 dev videos, seeds 7 and 107, guidance 1, render 1; VGGT camera error at frame 160 (primary), DINO and LPIPS (content).
- Win = better on the primary metric in >= 7 of 10 videos on the two-seed per-video average AND content not worse on average
  (DINO and LPIPS). Training ideas: a matched control from the same init, same number of updates, same seed and the same base
  row stream (shared RNG, as in the measured-write test); only the idea differs. Inference ideas: same model, same seeds.
- Equal compute where an idea spends extra generation (early check / restart, resampling): compared against best-of-N plain
  samples with the same number of generated calls, selected by the same signal.

Order and cost (GPU):
1. DROPPED by the user 10-24 ("fikir1 onemli degil es gec onu"; code kept: closed_loop_864 --first_render). First-step memory (inference): first-call render dup2 (current) vs dup4 vs depth points (closed_loop_864 --first_render).
   6 runs ~4 h. If a variant wins: train +250 with that first-step render in the start=1 training rows vs control C (run_op/C).
2. Memory-contribution score (measurement): from the same state continue 4 calls with vs without step k in memory, 2 seeds;
   is the with/without difference larger than the seed-to-seed difference? ~1.5 h.
3. Early check and restart at equal compute: plain / best-of-2 plain (DA3 residual picks) / early check at step 4 + restart with
   the corrected command (gain from the model's measured response). 2 seeds, ~5 h.
4. Train longer: already answered (C = +250 steps: 160f median .230 vs base .221, no gain). Remaining variant: every training row
   has exactly one step with its own render (as at test time) vs C. ~2 h train + 2 h eval.
5. Measured-pose training (done 10-24, criterion met vs commanded poses): candidate base for later ideas if it holds on 1-4.
6. Pilots with a go/no-go by 1 Nov: two-view refiner (artificial errors on real videos first), Gaussian feature memory.

## Rev 10-26 (user: "Kuyruga bunlari ekle") -- designs fixed BEFORE any result; queue_ideas2.sh, log queue_ideas2.log
Base and criterion as above (run_v864/mem; closed_loop_864 10 calls, 10 dev videos, seeds 7/107, open and F' offset arms;
win = >= 7/10 videos on the two-seed VGGT camera error AND content not worse). Matched control for every training arm:
run_op/C (+250 from run_v864/mem, seed 20261025, accum 8, n_chunks 2, render p .75, same windows).
A. Idea 2, memory-contribution score (memory_contrib.py, eval864/contrib_mem): one state per video = the memory model's own
   4-call open-loop prefix; continuation of 4 calls with the full memory vs with chunk k (k = 1..3) excluded from the bank
   candidates and the render sources; continuation seeds 207 and 307. Contribution = metric(without k) - metric(with).
   Reliable <=> Spearman between the two seeds' contributions (pos_err, retrieved chunks) >= 0.5 AND |contribution| > the
   seed-to-seed gap of the 'with' arm for >= 60% of the retrieved (video, k) pairs; never-retrieved chunks are the zero control.
   Reliable -> the score becomes the memory-selection / reward signal (the reward line opens); unreliable -> the reward line stays closed.
B. Idea 4, one step per row with its own render (train_m864v --n_choices 1): S1 = accum 8 (half the target chunks of C),
   S1x = accum 16 (target-chunk budget matched to C). S1x vs C answers "own render for every target step"; S1 vs S1x isolates the
   budget; C's second (render-free, noisy-predecessor) chunk is the single/multi-block difference. 4 evals per arm.
C. Learned two-view refiner, pilot 1 (refiner_pilot.py, pilots/refiner1): real frames, controlled geometric errors in three
   bands, previous clean frame + corrupted frame -> flow + residual UNet (1.5M params, identity at init), 45 min. Go (all):
   medium-band LPIPS <= 0.6 x corrupted; clean pass-through PSNR >= 35 dB; on our c10 generated frames LPIPS vs recording not
   worse by > .005 with sharpness ratio >= .9. Go -> pilot 2 inside the loop; no-go -> line closed for this model.
D. Gaussian-feature memory pilot (precompute_feat_864.py, train_feat.py, run_op/feat): the render slots' metadata widened from
   (valid, conf) to + (fused z-buffer depth, per-source depth disagreement = contradictions, source coverage = matches) from
   the memory views at the target cameras; projection widened with zero weights (starts as C), same rows / windows / seed as C;
   at inference the features come from the model's own frames. Win criterion as above vs C (4 evals).
E. Measured-write follow-up: which part carries the gain. Arms on the same on-policy pool / rows / seed as cmd and meas:
   mcc = measured bank selection only, cmc = measured pose labels (PRoPE + rays) only, ccm = measured render placement only
   (train_onpolicy --op_arm). Compared against cmd and meas on the same 4 evals; the part(s) whose arm reaches meas carry the gain.
F. Measured-write validation: (i) new scenes = dev rows 10-19 (never evaluated before), C/cmd/meas, seed 7, open + F';
   (ii) training-seed replication: C/cmd/meas retrained with seed 20261027 / op_seed 8 (run_op/{C,cmd,meas}2), seed 7 evals.
   The 10-24 claim stands only if meas beats cmd and C on camera in both (i) and (ii) (>= 6/10 each) with content not worse.
Not queued: idea 3 restart/selection variants (closed 10-25), idea 1 (dropped by the user), more training / sampling steps.

### Rev 10-26b (after ideas2_review_20261026/review.json; queue_ideas3.sh replaces queue_ideas2.sh, log queue_ideas3.log)
A. + same-seed repeated 'with' arm (render/DA3 instability) in the noise floor; camera verdict on VGGT (DA3 alongside); criterion
   (iii) zero control: <= 25% of never-retrieved chunks above noise; scene-arms with an invalid pose dropped and counted; bank-token
   use and render-source use recorded separately; continuation frames saved for independent rescoring.
B. Replaced S1/S1x by R2 (train_r2.py): the control's rows, windows, targets, context, u and noise draws, plus a render group for
   chunk 1 (its own command, recorded prefix; precompute_renders_next.py) hidden from chunk 0 by the mask (chunk_index -(2+j));
   verified on CPU: single-group mask bit-identical to the old encoding, targets and context prefix identical to C's rows.
   R2 vs C isolates "own render for every target step". S1x kept as an optional RECIPE comparison (stage G, last).
C. Training includes uncorrupted identity examples (p .2); acceptance also requires beating copy-previous; no pose input, so the
   pilot tests appearance-level repair only; a no-go closes this architecture / error distribution, not the refiner idea.
D. Channels read as depth SPREAD and COVERAGE (not occlusion-aware contradiction / verified matches): the pilot adds projected
   depth cues to the RGB render memory; it does not replace the render with learned Gaussian features.
E. Sources fixed: pool_render_fixed.py adds rz_meas_srccmd / rz_cmd_srcmeas for the 45 items whose cmd/meas banks differ;
   ccm = cmd bank + cmd labels + cmd sources placed at measured poses; mcc = meas bank tokens + cmd labels + meas sources at
   commanded poses; cmc = cmd bank + measured labels + stock cmd render. Single-component arms test sufficiency, not necessity.
Q. Completion by check_eval.py (10 scenes, 10 calls, finite metrics, VGGT, receipt sha); STATUS lines -> queue_ideas3_status.json
   (done / done_before / failed / skipped_disk / skipped_no_model). Disk: ~8 checkpoints (19 GiB) + ~36 evals (13 GiB) exceed the
   33 GiB above the floor, so late stages (F ii, G) will be skipped unless space is freed; skipped != completed in every report.

### Rev 10-26c (user: "spatia ve vmemi fikir kuyrugu basina ekle") -- stage 0 of queue_ideas3c.sh (stage0_ext_loop.sh), before A
Why the correction failed there (common-set evidence): Spatia's official 121+81 rounds leave ONE correction point (frame 121; the
first 121 frames are shared by all arms); VMem over-executes the full every-4-frame correction (step ratio 1.06 -> 2.83) and its
broken frames make DA3 unreliable; Astronex does not follow commands (ratio .12) and is not retested.
0a Spatia, 41-frame rounds (non-official, disclosed): rounds [0,41) [32,73) [64,105) [96,137) [128,161) = 4 correction points;
   baseline / replan_measured / replan_commanded rerun with the same rounds (runs/spatia_r41, kind spatia_r41). replan_commanded
   separates the re-aim itself from the measurement (WorldPlay: .818 vs .809 -> its gain is not from the measurement).
0b VMem replan_damped: measured write + re-aim every 16 frames, gain .5, applied correction clamped to .1 scene units / 5 deg,
   blended over 16 frames; existing seed-7 baseline reused (unchanged code path).
Works on a model <=> VGGT mean camera error lower in >= 7/10 videos AND mean DINO not lower; WorldScore-5 reported alongside.

### Rev 10-27 (user: B "daha fazla eğitim mantıklı değil" -> on-policy render; C "daha iyi bir fikir olarak" -> geometry first) -- queue_chain_1027.sh
H. Idea C step 1, training-free re-anchor pass (closed_loop.LoopRoller.reanchor_chunk, closed_loop_864 --reanchor_u): after each
   measured call, the SAME chunk is re-denoised from level u0 (SDEdit, remaining schedule steps) with the camera labels of the
   PLANNED poses and a ReSplat render whose sources are anchor + previous prefix end + the chunk's own end frames 1 and 3 at their
   DA3-measured poses, rendered at the planned poses; the chunk is re-measured, written at the measured poses, and the next call
   replans from it. Arms u0 = .5 and .8, seed 7, F' offset, 10 dev videos, memory model run_v864/mem; control = eval864/c10_vmem_fp2.
   Go: VGGT camera error at frame 160 better on >= 7/10 videos vs the control AND mean LPIPS not worse by > .005 AND sharpness
   ratio (gen / recording) >= .9 x the control's. Known prior: the 10-22 probe showed re-noising at low u alone cannot move the
   camera; H adds the geometric target through the render. Caveat stated in advance: the render's source set (the target chunk's
   own frames) never occurred in training; extra compute = the u0 fraction of the steps + 1 render + 1 measurement per call, no
   equal-compute control in this pilot (a go would be followed by one: best-of-N plain at matched compute).
I. R2 on-policy (train_r2op.py, run_op/measR2): the measured-write arm (train_onpolicy --op_arm meas, same rows / pool draws / seed as
   run_op/meas) with every target chunk's own render group on both row kinds: base rows read the recorded chunk-1 cache (as R2),
   pool rows read a chunk-1 render made from the own prefix at MEASURED poses + the recorded chunk-0 frames (precompute_pool_next.py).
   Controls: run_op/meas (adds the per-chunk render) and run_op/R2 (adds the on-policy source). Win criterion as above (>= 7/10 on the
   two-seed VGGT camera error at frame 160 vs meas AND content not worse), 4 evals (seeds 7/107 x open/fp).
Execution: queue_chain_1027.sh stops queue_ideas3c.sh at the D/E boundary (exact PIDs; every stage is completion-checked), runs H,
restarts queue_ideas3c.sh (E-G), runs I, then queue_baselines_1027.sh and the RE10K Spatia queue, sequentially.
