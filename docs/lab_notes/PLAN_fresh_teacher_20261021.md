# Fresh packed teacher (stage A) — 2026-10-21

Merged plan: the other session's prep/learnability/control/render-policy items + packed chunk-causal multi-chunk rows,
a PRoPE side branch as the main camera design, uniform u, render off in stage A. Corrections accepted on 10-21:
the 10,250 recipe already had render on ~60% of rows (last 1,250 updates), history noise 39%, self-aug 28%, own-state
recovery 25%; 4e-4 was only the `camera_qk_gains` gradient; packing is a throughput hypothesis until profiled;
uniform(0.01,0.99) gives ~30% u<0.3; PRoPE is a design choice, not a proven necessity; LoRA does not guarantee DMD fits.

## Recipe (code in this directory)
- Init: stock Wan2.2-TI2V-5B (fingerprint = BASE_MODEL_AUDIT.json). Trainables: self_attn + patch_embedding (fp32 masters),
  camera/metadata/roles projections, PRoPE branches (dim/8, zero-init out_proj) — `packed_model.py`, `prope.py`.
- Sequence `[anchor | bank<=6 | recent<=4 | N chunks x 4]`, N in {2,4,6,8} p=(.15,.25,.30,.30); first-call rows 20%.
  Chunk-causal mask (context sees only context; chunk j sees context + chunks <= j). Context t=0; chunk u_j ~ U(.01,.99) i.i.d.
  RoPE positions fixed by slot (anchor 0, bank 1..6, recent 7..10, chunk j at 11+4j); true cameras in Plucker rays + PRoPE.
- Bank: frustum-coverage greedy top-6 over latents 1..start-5 for the chunk-0 command (`data.select_bank`, = joint_v9 retrieval
  with a 4-latent recent block). Inference re-selects per call from own latents at commanded poses.
- Loss: velocity MSE, uniform over all chunk tokens. Text dropout 0.2 -> empty embedding. AdamW8bit 3e-5 / 1e-4, wd 1e-3, clip 1.0.
- Data: dataset_v3 train split minus the 661 camera-quality-flagged streams (8,420 streams); dev = dev_monitor_10 (never trained).
- Stage A: no render (metadata channels zero). Render returns (stage B) only after the direction test shows the camera is
  executed from the rays/PRoPE: mean cos >= 0.8 and ratio in [0.6, 1.4] on forward/back/left/right, yaw ratio in [0.6, 1.4].

## Validation before the long run
1. `tests.py`: pose convention, row causality, no-future-leakage mask, per-chunk timestep labels, PRoPE invariance/sensitivity,
   gradient flow (out_proj first, q/k/v after one step), uniform loss. 2. `resources_profile.py`: sec/update and VRAM at N=2/4/8,
   checkpoint save/resume. 3. Learnability: 16 streams x 200 updates (loss must fall), then a fresh start for the pilot.
4. Pilot: 300-500 updates on the full train set; `eval_pilot.py` at 100/300/500: dev10 recorded-command DINO/LPIPS/DA3 per call
   + direction test. Learning-curve decision, not a single-number kill.

## Budget
Measured by the profiler; the main run is sized from chunks/hour. Disk floor 17 GiB; resume slots fall back to /dev/shm.
Protected models untouched (AGENTS.md).

## 2026-10-21 evening — audit corrections and recipe v2
- **Lineage error (found by the other session):** `run_pilot` and `run_overfit` shared `/dev/shm/fresh_teacher_resume/resume.pt`
  and `load_resume` did not verify the sha/run id. The "2000->3000+" continuation resumed at **600 from the overfit run**
  (`run_pilot_3000.log: resumed at 600`), and `trainables_02000` was overwritten (sha 7ee5… -> 98cf…). So evals at
  3000–8000 belong to the lineage stock -> 16-stream overfit 600 -> full data; only 100–2000 are the clean pilot.
  Fix: resume slot per run id, sha + run id + update verified on load, checkpoints immutable, lineage.json per run,
  eval.json records the checkpoint sha/run id.
- Probe now reports chunk-0 / chunk-1 / first-call rows separately with paired SE; first-call rows included.
- Recipe v2 (hygiene, not the hypothesis under test): accumulation 8, u ~ uniform t through shift 3 + Memorizon time
  weights (u>0.8 ≈ 43%, u<0.3 ≈ 12%), optional LoRA r=64 on FFN + cross-attention (self-attn stays fully trainable).
- Stage B implemented: `render_rows.py` (recorded-prefix DA3 + ReSplat render of chunk 0, VAE-encoded, no corruption),
  render slots (role 3, chunk-0 cameras/positions, context, invalid keys masked), roller render mode through
  joint_v9's WorldRuntime (DA3 on own frames + ReSplat), `eval_pilot.py --render 1`.
- Plan: clean lineage A2 (recipe v2, no render) -> verified checkpoint -> matched A/B pilot at equal rows:
  A2 continuation vs B (render p=0.75, clean). Success = direction cos >= 0.8 and ratio in [0.6,1.4] on the direction
  test with render on, calls 2–4 LPIPS not worse than A2; report separately whether rays/PRoPE learn without render
  (probe with render removed, direction test with render off).
- Resolution: source frames are 480x270 (DL3DV-480P images_8); 448x256 training is native. Higher resolution needs
  DL3DV-960P re-encoding (640x352: ~57 GB cache, 2.5x compute) — blocked by the 17 GiB disk floor.

## 2026-10-21 night — Memorizon-50 init at 864x480 (the camera-capable start)
- Released undistilled teacher Luffuly/memorizon-1280x704-50step (Wan2.2-TI2V-5B, train_backbone, PRoPE compress 8, shift 3)
  downloaded to memorizon50/ (10.3 GB, 1005 tensors mapped into PackedTeacher: backbone via diffsynth converter, prope_attn
  branches 1:1; our Plucker/metadata/roles projections zeroed). Layout 'memorizon' (A=0, bank=1, recent 2..5, chunk 6+).
- Zero-shot direction test, dev anchors from the other session's qualification inputs (calibrated metres/anchor-unit):
  448x256: fails (DINO 0.14, sharp 2700 — far from its training resolution). 864x480, CFG 1: scale x1 (metres/4) under-executes
  (fwd cos .81, ratio .48; 3-call z-ratio .48); **scale x2 (metres/2): fwd/right/left/back cos .97/.77/.74/.53, ratio .67–.85,
  still drift 0.06, 3-call z-ratio .86/.80/.78**. CFG 4 with the empty negative breaks it. Yaw magnitude right, sign mirrored
  (DA3/our convention verified on recorded streams: yaw_check.json) -> the weights' rotation convention; y-flip test queued.
- 864x480 corpus (prep_864.py -> data864/): 958 scenes = 989 streams of the 448 corpus (same windows), 74 flagged removed,
  gauge + text reused from the 448 components, metres/unit = median DA3 metric depth of the 864 anchor; 300 streams built
  (20 held-out dev), rest queued after the pilot (disk cleanup approved: 15.6 GB removed, cleanup_20261021/).
- Profile 864 + Memorizon init + LoRA 64: N=2 2.6 s/row 23 GiB, N=4 5.2 s/row 27.7 GiB; FM loss at step 0 already 0.14–0.42.
- Pilot: run_m864 = 100 steps x 8 rows (N in {2,4}, shift3, accum 8, LoRA 64, scale_mult 2) from Memorizon-50; evals on the
  20 held-out 864 streams (recorded commands) + direction test; compared with the zero-shot teacher.
- Stock-A2 control deferred (GPU given to the Memorizon route); stock A2 attempt died at step 200 on a full /dev/shm.

## 2026-10-21 23:10 — pilot gate passed (Memorizon-50 init, 864x480)
run_m864 (100 steps x 8 rows; N{2,3}, bank<=4, LoRA 64, shift3, scale x2; resumed from the step-50 slot after an N=4 OOM):
c1/c4 DINO .88/.42, pos err .10/.64, exec ratio c2-4 1.08, direction cos .92-.99 on all four axes, still drift .06
(gate_pilot.json). Transplant verified (single forward rel-L2 1.05%). Queued: 240x432 render baseline, CFG 4/2 with
Memorizon's negative prompt on the pilot checkpoint, full 864 corpus. Next (user decision): long run from the pilot
checkpoint with evals + the same gate every 250 steps; then stage B at 540x960 (render p=.75, camera-branch dropout).

## 2026-10-22 — camera-error ideas (user): literature + what changed since the 4-step-student experiments
User asked for two things: (1) a small refinement driven by the measured camera error, (2) a formulated camera-error loss for
training the generator. Earlier negative results (Oct 13–17) were all on the released 4-step DMD student and do NOT transfer:
FM fine-tune blur = mean regression on a few-step student (the 50-step teacher's native loss is FM; pilot-100 improved LPIPS);
latent pose head failed its gate on the STUDENT's own latents (Spearman .14); camera-ReFL never ran; latent D was not a
sharpness signal (refiner). Still valid on any model: DA3/VGGT on decoded frames is a trustworthy camera measurement
(recording sanity ratio .99–1.01; DA3–VGGT per-call Spearman .61), and the command is a saturating actuator on a distilled
model (switch gain .19 at step 0, ~0 from step 2).

### What others do (checked 2026-10-22, arXiv HTML)
| method | model / horizon | camera signal | optimiser | cost | result |
|---|---|---|---|---|---|
| CamVerse 2512.02870 | 2B DiT, single clip, 14 steps (3 SDE) | π³ poses, Umeyama-aligned, per-segment relative pose: e_t clipped L2 + e_R geodesic, confidence mask | GRPO, G=16 keep top/bottom 4, KL 1e-4, LoRA r64 all linears | 200 it × 343 s on 32 H200 | RE10K T2V trans .0395→.0293, rot .651→.514; I2V VQ drops 5.02→4.87 |
| Geo-Align 2605.23903 | ReDirector (Wan2.1-1.3B), 81 f, 25 steps | MapAnything METRIC poses, Σ w_i‖t−t̂‖ + geodesic, later frames weighted more, no alignment; +VideoAlign/HPSv3 | GRPO (MixGRPO window), G=12, no KL, self-attn only, LR 1e-4 | 140 it, 64 A800 × 130 h | DAVIS trans .0149→.0129, rot 1.46→1.36 |
| VGGRPO 2603.26599 | Wan2.1-1B / Wan2.2-5B, T_train 10 vs 40 | latent geometry model (Wan VAE → Any4D/VGGT stitched; no RGB decode): smoothness + reprojection consistency | GRPO, G=64, LoRA r32, KL .004 | ~1536 GPU-h | epipolar err .142→.093 (5B); test-time reward guidance every 20/50 steps = small gain |
| WorldCompass 2602.09022 | WorldPlay (HY-1.5-8B / Wan2.2-5B) AR, 16-frame clips, up to 16 clips | DA3/WorldMirror trajectory → discrete action accuracy (1° rot, 0.01–0.05 trans thresholds) + HPSv3 | DiffusionNFT (negative-aware FM targets), clip-level rollouts: prefix generated once, G=16 for clip n, top/bottom 3, 50% timesteps, Muon 1e-5, EMA anchor, no KL | 3 d × 64 H20 | composite-action acc 19.7→54.8 (381 f); IF-only reward → HPS −2.67 (degradation) |
| WorldCycle 2608.04964 | WorldPlay 8B AR | self-verifiable: action cycle must return to start (CoTracker displacement) + action following + HPSv3 | DiffusionNFT, G=16, only trailing chunk gets gradient, AdamW 1e-5 | 3 d × 8 H200 | drift −13–34% vs WorldCompass, acc .136→.553 |
| DreamX-World 1.0 2606.16993 | Wan2.2-5B student (after DMD) | clips cut from long AR rollouts, camera + quality reward | DiffusionNFT soft update | — | camera 73.75 |
| CamPilot 2601.16214 | CogVideoX-5B I2V | differentiable: latents → 3DGS decoder → render at GT poses vs GT frames (masked MSE + .5 LPIPS); needs GT | ReFL, t~N(.8,.075)∈[.6,1], 5k steps LR 1e-5; VAE decode would cost 43 GB → 3DGS decoder 8 GB | — | RotErr .023 / TransErr .152 (best) |
| PartiCam 2609.39504 | CogVideo / SVD, training-free | VGGT on decoded Tweedie x0 at steps 8–32 of 100; SMC K=2, N_r=2, restarts re-noise to σ_{t+1} | selection only, conditions never edited | larger denoising budget | ATE 1.56→.37 (static sweep) |
| JOG3R 2501.01409 | OpenSora, joint pose head | DUSt3R-style head on block 26 features, t∈[0,10] | L_gen + L_rec jointly | — | "no clear synergy": pose loss does not improve generation |
| Latent Bridge 2610.28521 | Wan2.1 16-ch VAE only (not our Wan2.2 48-ch) | bridge latents → frozen VGGT patch tokens | — | 1.5k GPU-h | geometry from latents; not a pose benchmark |
| RVM 2608.23664 / AWM 2509.25050 / FlowBP 2606.11075 | general | — | reward-weighted velocity matching on independently re-noised endpoints (no likelihoods, ODE rollouts fine; DiffusionNFT = special case); AWM G=24; FlowBP: evaluate reward on the sampled endpoint, K=3–5 active steps | RVM 2.3× cheaper than FlowGRPO | Wan2.1-1.3B VBench 84.1 vs DanceGRPO 79.5 |
Nobody backpropagates through VGGT/DA3; the differentiable routes are a 3DGS decoder (CamPilot, needs GT) or a latent
geometry model (VGGRPO, Wan2.1 VAE). Inference-time: PartiCam selects particles by camera reward, Latent-Reframe edits
latents, nobody corrects the command/condition from a measured error (still open, as in the 10-10 survey).

### Idea 1 — inference-time correction from the measured camera error
- Evidence that shapes it: on every model probed the camera is decided EARLY in the call (Memorizon: gain .19 at step 0,
  ~0 from step 2; ours 10,250: gain .95/.66/.13/.02 at u 1/.90/.75/.50). A re-noise refinement at u≈0.3 therefore cannot move
  the camera; it only re-renders appearance. decision_probe.py measures this on the new teacher (x0 pose per step, switch gain
  per step, re-noise gain per u_r ∈ {.3,.5,.75,.9}) — run on the pilot/350 checkpoint when the GPU is free (~45 min, 5 anchors).
- Design (no training): (a) F′ port — DA3 realized pose after each call (≈1 s), memory written at the measured pose, next
  command replanned toward the plan (SE(3) interpolation); won 6/6 on Memorizon, 5/6 on WorldPlay, 5/6 on VMem. (b) Early
  camera check — x0 pose at step 1–2 (usable: rel err 5–7% from step 1 on both models); if the executed/commanded error exceeds
  τ (e.g. ratio outside [.7,1.3] or direction cos < .9), restart the chunk from u=1 with the command corrected by the one-step
  inverse (command ← command + κ·error, κ from the probe's switch gain at step 0), not from u≈0.3. Cost: +2 forwards + DA3 per
  chunk, +1 chunk when triggered. (c) Only if the probe shows gain ≥ .5 at u_r ≤ .75: SDEdit-style re-noise refinement with
  corrected cameras becomes the cheap variant. Evaluate on the same 10 streams + direction test + gate (DA3 and VGGT).
- Needs no GPU for coding; evaluation after chain_long or in a gap (user decision; never pre-empt).

### Idea 2 — camera error as a training signal for the generator
- Not: FM on a few-step student (blur), latent pose head (failed gate), backprop through DA3 at 864×480 (no precedent; VAE
  decode alone ~43 GB in CamPilot's measurement), latent discriminator.
- Proposed arm (after the long run, from its best checkpoint, equal-lesson comparison): clip-level reward fine-tuning in the
  WorldCompass/WorldCycle form with the RVM/DiffusionNFT objective (reward-weighted FM on independently re-noised endpoints;
  no likelihoods, our Euler ODE sampler at 10–14 steps; LoRA + PRoPE + self-attn trainables as now).
  Rollout: prefix = recorded calls (50%) or own calls 1..n−1 generated once with F′ measured-pose writes (50%); group G=8
  candidates for call n with different noises; reward on decoded frames = −(|t_exec − t_cmd| / |t_cmd| in the anchor gauge
  [unaligned, metric via calibrated metres/unit] + e_R geodesic/15°) − LPIPS(gen, recorded chunk at the commanded poses)
  − sharpness penalty beyond 1.3× GT. Fidelity-to-recording is the quality regulariser (WorldCompass: IF-only reward degrades;
  CamVerse: VQ drop) and it is free because our commands come from recordings. Advantages z-scored per group; top/bottom 3.
- Budget on the single 32 GB GPU at 864×480: prefix (shared) + 8 rollouts × 12 steps + 8 decodes + DA3 ≈ 2 min, update
  (2 noised copies × N=1 row, 23 GB) ≈ 10 s → ~2.5 min/update; 100 updates ≈ 4 h. Gate = pilot gate + camera error per call
  (DA3, VGGT), KID/sharpness vs the parent; stop if LPIPS worsens.
- Differentiable alternative kept as an optional aux term for later: latent reprojection consistency (previous latent warped by
  DA3 depth of its frame, detached, under the commanded pose → masked L1 vs the chunk's one-step x0 at u∈[.6,.9]); cheap
  (no decode), but confounded with depth error — only after the reward arm.
Order: decision_probe (GPU gap) → F′ + early-check roller on the 350/600 checkpoint → reward arm from the best long-run
checkpoint. Long run (chain_long.sh) continues untouched.

## 2026-10-22 01:30 — "memory-verified closed loop": the novel package, readiness checks, and the queued experiments
User: include ReSplat, make the idea more novel, verify every metric/scale first, then start all experiments in order.
**Claim (what no one does):** the generator's own 3D memory (ReSplat Gaussians from its own frames, placed at DA3-MEASURED
poses) is used as a *verifier* in both loops — at inference (F' writes + replan + an early x0 camera check that restarts the
chunk with a corrected command) and in training (a reward arm whose signals are the realized pose in the calibrated anchor gauge,
agreement with the memory render at the commanded poses, and fidelity to the recorded chunk). Precedents each cover one piece:
WorldCompass/WorldCycle (reward RL on AR world models, discrete actions, no 3D memory), CamPilot (3DGS render reward vs GT, no
memory/AR), GeoStream/AlayaWorld/Uni3C (render as condition only), PartiCam/Latent-Reframe (inference selection/latent edits,
conditions never corrected). Pieces are not individually new; the closed loop through an explicit, measured-pose memory is.
**Code (sample.py/train.py untouched except one roller fix):** closed_loop.py (LoopRoller: measured write, rejoin replan, early
check + corrected restart, own-frame render context), closed_loop_864.py (arms; DA3 + VGGT; eval.json compatible with
gate_pilot), render864_rows.py (RenderBuilder864 for stage-B rows, OwnRender864 for rollouts; ReSplat 540x960 + DA3 gauge
depths + VAE), train_b864.py (stage B wrapper), reward_train.py (clip-level RVM/NFT reward arm; rewards above; reflected negative
target about the start weights; anchor FM on the recorded chunk), decision_probe.py, check_metrics_864.py, queue_ideas.sh.
Fixes: renderer864.da3_gauge_depths single-view (DA3 Umeyama needs >= 2 views); sample.py bank selection used the 448 frame
size at 864 (frustum test) -> stream size (affects chain_long evals from its first eval on, consistently).
**Readiness check (check_metrics_864.py, queued first; the 00:50 attempt OOMed next to prep):** DA3 and VGGT on recordings
(ratio ~1, pos err small), ReSplat 540x960 render at x0.5/x1/x2 target scale (x1 must win), metric self-consistency,
peak memory of the render path. The queue stops if it fails.
**Queue (queue_ideas.sh, started 01:3x, waits for chain_long to end; never pre-empts):** 0 readiness -> 1 decision probe (best
ckpt) -> 2 closed-loop arms open/write/F'/F'+early with gates vs open -> 3 stage B A/B from best (+250 render p=.75 vs +250
no render, equal lessons/seed; smoke first; evals with and without the own-frame render) -> 4 reward arm (smoke, 100 updates,
render on iff B beat A) -> evals (plain and with F'+early) + gate. Disk: smoke/resume slots of the queue's own runs are
removed after use; checkpoints kept (2.8 GB each).
- 01:50 ReSplat 540x960 memory (resplat_mem_864.json, next to prep): peak 9.4 GiB with 1 source / 15.4 GiB with 4 sources (DA3 +
  weights resident 6.3), 1.3–2.7 s; num_refine 1 vs 4 same memory. With the DiT (bf16 10 GB) + fp32 masters (5) + AdamW8bit
  (2.5) resident this would exceed 32 GB -> renders are capped at 4 sources (anchor + last two ends + best bank view) and
  reward_train/train_b864 move the masters + optimizer state to the host while a render runs (make_offload); evals keep VGGT on
  the host between scorings. Stage-B smoke in the queue measures the real training peak before the 250-step runs.

## 2026-10-22 02:00 — novelty assessment (user: measure it, compare with SOTA papers, CVPR fit)
Calibration — what accepted CVPR'26 / recent SOTA papers deliver: ONE clear mechanism + scale + benchmarks + ablations.
Spatia (CVPR'26): point-cloud memory updated by visual SLAM on GENERATED frames (= measured-pose write), dynamic/static split,
Wan2.2-5B at 1248x704, applications. WorldStereo (CVPR'26): two memory modules on Uni3C (global geometric memory from
feed-forward recon of generated frames + stereo-correspondence attention) + a new 3D-recon benchmark + DMD compatibility.
UCM (SIGGRAPH'26): PE warping unifying camera control + memory, 561k videos, 8 A100 x 4 d. World-R1: DA3->3DGS re-render
reward (1-LPIPS) + trajectory deviation + VLM meta-view, Flow-GRPO, single clip, no memory, 48-96 H200. WorldCompass /
WorldCycle: reward RL on AR world models (discrete actions / cycle closure), 64 H20 x 3 d.
| our piece | closest prior art | delta | novelty |
|---|---|---|---|
| memory written at the measured pose | Spatia (SLAM on generated frames), WorldStereo (recon of generated frames) | occupied; we only quantify its effect alone (mechanism test: residual -0.05, no appearance gain) | low (analysis value) |
| command replan from the measured pose (F') | none found (10-10 survey + 10-22 search); PartiCam/Latent-Reframe never edit the condition | new, simple ("one-step MPC"); works on Memorizon 6/6, VMem 5/6, not on WorldPlay at native speed cap | medium |
| early x0 camera check + corrected restart inside the call | PartiCam (reward-guided particle selection on x0, no condition edit) | new: measured x0 error edits the command where it still acts; needs the decision-step probe | medium |
| memory-render consistency reward (training) | World-R1 (3DGS re-render reward, single clip), VGGRPO (reprojection reward), CamPilot (render vs GT) | reward idea occupied in single-clip T2V; ours = persistent cross-call memory at measured poses + unaligned metric camera error + GT fidelity in AR rollouts | low-medium |
| same memory as verifier in BOTH loops (inference + training), 2x2 interaction | no paper couples them; WorldCompass admits no reliable drift/memory metric | framing + protocol (executed/commanded ratio, drift under DA3 AND VGGT, revisit) | medium (if the interaction replicates on >=3 generators incl. ours) |
Verdict: component novelty is moderate — no single piece is a breakthrough; the closed-loop framing and the cross-generator
analysis are the paper. SOTA papers bring one mechanism at 50-100x our compute; we cannot win on scale, so the contribution
must be (a) a result that holds across generators (ours + Memorizon + VMem; WorldPlay as the negative case explained by its
speed cap), (b) a closed-loop camera protocol with independent estimators, (c) the 2x2 + reward-term ablations, (d) a
reward arm that improves long-horizon camera error WITHOUT quality loss (WorldCompass/CamVerse report degradation).
CVPR fit (deadline 16 Nov, 25 days, one 32 GB GPU): submittable only if the queue results are positive by ~Oct 28 and the
evals scale to >=20 scenes x 4 arms x 3 generators by ~Nov 5; otherwise the honest target is a workshop or ICCV/ECCV 2027.
Do not claim "closed loop is empty" (Spatia, Closing-the-Loop, World SLAM Model, World-in-World exist); claim the measured-write
x replan interaction and the memory-verified reward.

## 2026-10-22 03:53 — stage 350 gate FAILED (chain_long stopped as designed); queue_ideas started from pilot-100
Stage 350 (250 more updates from pilot-100, loss .30 -> .28, 27 s/step, peak 27.7 GiB) vs pilot-100, same 10 streams (paired):
image side better — LPIPS c2–4 .557/.601/.643 vs .598/.670/.686 (6–7/10 scenes), late sharpness noise GONE (c3–4 sharp .94x GT
vs 1.67x; 8/10 scenes), DINO mixed (c4 .463 vs .427, 6/10); camera side worse on MEANS, better on MEDIANS: pos err median
c2/c3 .25/.36 vs .39/.52, exec ratio medians c1/c2 .95/.90 vs .68/.69 (pilot under-executed), but two streams blow up
(284b0dbc: c2 ratio 20, pos 2.26, DINO .39 — a runaway camera/scene change at call 2; 5f800e62 bad in both runs) and
e03b0bf0 over-executes (1.3–1.8). Gate (means): pos c4 .769 vs .675, ratio_dev .72, ratio_max 2.83 -> fail; dino/lpips/sharp/
direction/yaw/still pass. Reading: more SFT makes the camera stronger (under- -> full execution) and the images sharper, at the
cost of occasional runaway calls — exactly the failure a closed loop / camera reward should catch. Decision left to the user:
(a) keep the gate as is (queue runs from pilot-100, already started), or (b) accept 350 by medians + a blow-up count rule and
re-run the queue's stage B / reward arm from 350. Chain protected.json keeps pilot-100 + 350.

## 2026-10-22 05:15 — decision probe + closed-loop arms on pilot-100 (first positive closed-loop result on OUR generator)
Decision probe (5 anchors, fwd/right, 30 steps shift 3): x0 pose rel err .61/.33/.24/.07/.02 at steps 0/1/2/4/8 (usable from step 4);
command-switch gain .26–.47 at step 2, .11–.13 at step 4, ~.03 at step 8, ~0 at 12 -> camera decided by step ~8 (u~.85);
re-noise refinement gain at u .3/.5/.75/.9 = 0.00–0.02 for every command change -> **re-noising cannot move the camera**
(idea 1 in its original form is dead on this model too); the early check therefore taps x0 at step 4 and restarts from u=1.
Closed-loop arms (closed_loop_864, 10 streams x 4 calls, DA3 + independent VGGT, paired vs open loop):
- measured write: DA3 pos err wins 8/10 at c2 and c3 (median .32/.32 vs .39/.52), VGGT 9/10 c2, 7/10 c3; LPIPS 7/10 c2–4;
  DINO c2 7/10. Gate fails only ratio_max (c4 mean 2.3, one stream).
- F' (write + replan): VGGT pos err wins 9/8/8 at c2/c3/c4 (median c4 .41 vs .50), VGGT rot err c4 median 14 deg vs 31 (8/10),
  LPIPS c3/c4 8/10 (.634/.657 vs .669/.693), DINO c4 median .51 vs .38; translation over-executes on some streams (DA3 ratio
  mean 1.66 at c2, median .74). Gate fails only sharp (1.71 vs 1.67, marginal).
- F'+early crashed (dict merge bug) -> fixed, queue relaunched 05:16 (resumes at that arm; early step 4, tol .3, cos .9, kappa 1).
Same pattern as on Memorizon/VMem: write = position lever, replan = rotation + late-appearance lever; the combination wins on
the independent estimator on 8–9/10 held-out scenes. This is the evidence the paper needs (now on three generators).
- 05:27 F'+early (x0 check at step 4, tol .3, cos .9, kappa 1): DA3 pos err c2/c3/c4 .31/.33/.37 vs open .42/.58/.68 (c4 -46%),
  VGGT pos .33/.46/.72 vs .41/.64/.86, DINO c4 .503 vs .427, LPIPS c4 .639 vs .686 — best arm on camera and late appearance;
  but 36/40 calls restarted (the check fires almost always: x0 at step 4 + DA3 on blurry x0 frames is noisier than the probe's
  clean-anchor estimate, and tol .3 is tight) so it is effectively "generate twice with a one-step-inverse-corrected command";
  gate fails only sharp (1.75 vs 1.67). To tune later: tol .5 / step 6, kappa .5, restart statistics per stream.
- 05:28 stage-B smoke OOM at the render inside the accumulation loop: the offload moved masters + optimizer state but the
  accumulated fp32 GRADIENTS (5 GB) stayed -> 27.8 GB resident + ReSplat 3.75 GB alloc. Fixed: gradients offloaded too
  (make_offload), empty_cache on restore; queue relaunched.
- 05:36 second smoke OOM (update 2 draw, 30.45 GB allocated): make_offload kept Python references to the GPU tensors of the
  masters and optimizer states, so nothing was freed. Fixed (references dropped; CPU test ok); PYTORCH_CUDA_ALLOC_CONF=
  expandable_segments for all queue steps. Update 1 of the smoke had completed: 83 s, peak 29.1 GiB, 8/8 rows with render.
- 05:40 stage-B smoke PASSED: 2 updates, 84–87 s/update (3.1x the no-render 27 s), peak 26.1 / 29.0 GiB, 8/8 rows with render.
  run_b864/main (+250, render p=.75) started 05:41 (~6 h), then run_a864/main (+250, no render, ~2 h), evals, reward arm.
- 05:45 run_b864/main OOM at update 4 (28.0 GB allocated + 2 GB fragmentation during a row's forward/backward; N=3 + bank 4 +
  4 render slots is too big next to the resident render/DA3/VAE modules). A/B now both at N=2 (equal settings, equal lessons);
  resume slots removed right after each run (/dev/shm has room for one 7.7 GB slot next to run_m864's); reward arm keeps one
  checkpoint (ckpt_every 100); evals save generated frames only at quarter rate. Disk 26 GB (floor 17; writes need 20).
- 06:22 run_b864/main OOM again at update 11 (N=2; peak 25.9 at update 1 -> 29.0 from update 2): the render builder moved the
  training VAE (0.70B, 1.31 GB bf16) to the GPU for encoding and left it there. Fixed: the VAE returns to its original device
  after each encode (training passes a CPU VAE; the eval/reward rollers keep theirs on the GPU). Relaunched.
- 06:41 fourth B OOM (update 13, 26.7 GB allocated + 3.4 GB fragmented): moving ReSplat/DA3/VAE on and off the GPU eight
  times per update fragments the allocator next to the resident DiT + masters + grads. Decision: stage-B renders are
  PRECOMPUTED (precompute_renders_864.py: start 1 + 3 seeded random windows per train stream, same bank selection, stored
  bank checked at use; ~0.6 MB each, ~2.2 GB total) and train_b864.py reads them (CachedBuilder + a RowSampler.draw that
  picks a precomputed window for render rows). Training memory then equals stage A (N=2) and the 3x slowdown disappears.
  Reward arm: VAE + DA3 leave the GPU during its loss phase.
- 06:54 precompute crashed on a window with near-coincident source cameras (DA3 Umeyama "degenerate covariance rank");
  da3_gauge_depths now falls back to unaligned metric depth for such windows. Precompute resumes (existing files skipped).

## 2026-10-22 08:10 — audit (other session) accepted: 350 candidate branch, matched A/B, no automatic reward arm
Audit findings (verified): (1) child checkpoints (--init_receipt) carry lineage.init as a dict, so str(init).startswith('memorizon50')
was False -> evals/reward would have loaded the STOCK base and the Wan sampler for every A/B child; fixed with
model_io.resolve_init (walks the parents). (2) the reward parent choice ignored the gates vs the parent -> no automatic reward
arm; it starts only on the user's decision after the gates. (3) own-prefix rollouts paired with the recorded chunk as GT LPIPS +
anchor FM target are not valid once the history drifted -> first reward experiment uses recorded prefixes only (own_prefix_p 0);
the memory-render term alone does not verify the memory. (4) A/B rows were not matched (B drew from precomputed windows, A from
all windows) -> both arms now run through train_b864 with the same seed: identical (stream, start, n) sequences, B adds
renders on p=.75 (verified by the queue's matched-rows count). Corrections to my reading of 350: "camera better in median" holds
for c2/c3 only (c4 median .482 vs .474; wins 5/10 at every call); |ratio-1| medians .34/.35/.30/.42 vs .40/.34/.29/.46 — less
under-execution but some over-execution; the 284b0dbc ratio 20 is not a near-zero denominator (command .204 units) but needs an
independent measurement; "runaway count <=1" would pass both models — the new gate must separate median, tail magnitude and
visual collapse and must be fixed BEFORE the next evaluation, not after seeing 350. Closed loop: F'+early best on DA3 pos
(c4 .367) and DINO; F' best on VGGT pos (.716) and rotation (21.0 deg mean; 14 vs 31 median); early check costs 59 -> 75 s/scene.
Plan: queue_350.sh (after the render precompute): closed-loop arms on 350 (open/F'/F'+early, gates vs 350-open and vs
pilot-100-open) -> A/B from 350 with matched rows (run_a864_350, run_b864_350; evals a350 / b350_render / b350_norender;
gates vs 350 and vs pilot-100) -> stop. queue_ideas.sh (pilot-100 parent) stopped after its precompute step.

## 2026-10-22 08:50 — visual inspection of pilot-100 vs 350 (contact sheets, 10 streams x 8 frames x {rec, 100, 350})
Two different failure modes. Pilot-100: translucent white "plastic sheet"/fog overlays + speckle from call 2–3 on (c2282d3f,
0d26e9b3, e83f25bf, 284b0dbc, 5f800e62), then identity switches to a different scene (c2282d3f -> shop interior, e83f25bf ->
corridor). That is the 1.6x "sharpness" = noise. 350: texture and lighting stay photographic and the scene identity is mostly
kept (c2282d3f, 284b0dbc, bf5a1aba, c3cfc2ff), but GEOMETRY melts after call 2–3: boulders become lumps (e03b0bf0), the island
top turns into white sludge (0d26e9b3), the playground structure deforms early (e7523f6f), foliage floats (e83f25bf); the
camera also comes closer than the recording (over-execution, e03b0bf0 / c2282d3f end). 284b0dbc: pilot-100 fully collapsed
by frame 39 while 350 keeps the plaza with a sheared facade -> VGGT (350 closer) matches the visuals, DA3's "ratio 20" does not:
DA3 misreads the warped facade. 5f800e62 fails in both from frame 7 (low texture) -> exclude from gates. Clear 350 wins:
c2282d3f, 284b0dbc, bf5a1aba, c3cfc2ff; clear losses: e7523f6f, e03b0bf0; both fail: 0d26e9b3 (different modes), e83f25bf,
5f800e62; tie: edf87641. Neither model survives 4 calls on half the streams: the history-content collapse is still the
main problem; 350 fails more photo-realistically (inflates DINO/KID), pilot-100 fails more visibly.

## 2026-10-22 09:10 — assessment of the "measure → correct → distill → learned refinement" proposal (other session)
Framing accepted: the paper = "measured motion used in two places (memory write + next command) removes error accumulation,
without retraining, across generators" — known parts, new necessity/interaction evidence. Evaluation of the 3 steps vs our data:
STEP 1 (reliable correction pairs): supported. Our probes already give same-noise paired generation (decision_probe) and the
per-call gain: pilot executes .86–.95 of the command at call 1 and .79–.89 later; F'+early with a fixed kappa=1 over-corrects at
c3/c4 (ratio 1.12/1.27, VGGT c4 ratio 3.0 mean) = the audit's oscillation warning is already visible -> the gain must be
measured (running per-stream executed/commanded from the previous call; probe-based per-axis gains cost 2–4 extra generations
per call and are not worth it at first). Add rotation to the acceptance (early_check records rot_err but ignores it), bound the
correction (|delta| <= .5 |cmd|, confidence from the DA3/VGGT agreement), and replace the end-point rejoin with a waypoint-
preserving shift (command_k = P_k + (M - P_start)(1 - k/16)). Acceptance must use the memory render AT THE MEASURED pose
(camera error vs scene error separation) + a next-call check; measurement floor .04–.05 units sets the minimum gain.
STEP 2 (distil accepted corrections with the ORIGINAL command): the fastest practical gain and cheap (FM on stored pairs
mixed 1:1 with recorded rows). Not the 10-02 failure (GT future on own history) because the target is the model's own accepted
sample: no history conflict. Risks: (a) self-training on own samples reinforces the 350-style geometry melting unless the
memory-render check rejects it; (b) hindsight relabelling of own rollouts exists (noted 10-16) -> claim the closed-loop pairing,
not relabelling; (c) a per-state learned mapping is needed because a fixed command gain fails late (Memorizon cmd_gain 2/3.4
raised ratio but worsened call-10 error). Success metric: with the external corrector OFF, pos err / ratio / LPIPS / KID at
calls 1–4 improve and the number of restarts the corrector would trigger drops.
STEP 3 (error-conditioned residual refinement, ReSplat analogy): research-grade, highest risk. Constraint from the probe: the
camera is fixed by step ~8/30 and re-noise gains are 0 at u <= .9, so a refiner must act at u in [.85,.95] where the x0 pose
error is 7–24% and the x0 frames are blurry; "pose early, surfaces late" is the right split. Inputs (x_u, u, 6D camera error,
memory render at the measured pose + validity, diff map) overlap GeoStream (reprojection latents added to noised latents),
Lyra 2 (warped last frame), CameraNoise, Latent-Reframe, Cycle-World (reverse-model cycle guidance before writing history) ->
the delta is the error conditioning + partial-evidence masks; keep it as step 2 + extra inputs (zero-init residual branch), only
if step 2 plateaus. Not before Nov 16.
Cross-cutting: the visual inspection shows the dominant late failure is history-content collapse (fog / melting), not camera
error; the camera loop cannot fix it, but the memory-render acceptance can DETECT it in covered regions (34–40% coverage from
the anchor alone, 99% with 4 sources) -> the acceptance gate is the key new instrument; report its precision on the 10-stream
visual labels. Order: keep queue_350 (cl350 arms running, then A/B) OR reprioritise step 1 ahead of A/B (user decision).

## 2026-10-22 10:40 — closed loop on 350 PASSES the gate (both arms); 350 + F'+early is the best configuration so far
cl350 (10 streams, 4 calls, DA3 + VGGT): open DA3 pos .61/.80/.77 (c2-4), VGGT .28/.59/.90; F' DA3 .51/.51/.63, VGGT .31/.61/.71,
LPIPS c4 .599 vs .643, gate PASS; F'+early DA3 .24/.31/.36, VGGT .26/.30/.50, DINO c4 .511 vs .463, LPIPS c2-4 .549/.561/.598,
27/40 restarts (36 on pilot-100), gate PASS. Versus the same arm on pilot-100: VGGT c4 .50 vs .72, LPIPS c4 .598 vs .639.
Renders precomputed 10:02 (3543 windows); B(350) training started 10:38.

## 2026-10-22 13:40 — plain-Memorizon comparison arm (user request) + cleanup
User: train plain Memorizon 350 steps too and run plain-350 open / F' / F'+early for comparison. train_plain.py: Memorizon's recipe
as far as 32 GB allows (no LoRA, Plucker/metadata/roles frozen at zero via packed_model.PLAIN, prompt dropout 0, lr 2e-5/1e-4;
same rows N 2/3, bank <= 4, accum 8, shift-3 u, scale x2, seed, corpus 864; 1.275B trainables vs ours 1.390B; full-backbone
training does not fit). closed_loop_864 accepts --receipt none (zero-shot Memorizon-50) and plain.flag (same trainable scope).
Note: ours-350 saw the 300-stream partial corpus for its first 100 updates; plain-350 sees the full corpus throughout.
Disk: A(350) stopped at update 100 on the floor (19 GB); user approved deletion of 13.25 GiB (smoke checkpoints, unevaluated
trainables_00300, superseded eval frame dumps, invalid wp_u4) -> cleanup_20261022/receipt.json, 32 GB free.
Running: zero-shot closed-loop arms (run_zs_arms.sh) -> queue_350 (A resumes, evals, gates) -> queue_plain (plain-350 train,
its three arms, gates vs ours-350 F'+early). Logs: queue_350.log, queue_plain.log.
- 14:15 bug (mine): queue_350's cleanup "rm /dev/shm/fresh_teacher_resume/main-*" after B also matched A's slot (run ids are
  <dirname>-<hash>, both dirs named 'main') and ran again on the 13:31 restart -> A's update-100 slot was deleted; A restarts
  from 350 in run_a864_350/main2 (same seed, matched rows). Cleanups now use the exact model_io.run_id. Stale
  run_a864_350/main (trainables_00050, 2.6 GB) left in place pending approval. queue_plain was blocked by a waiter whose own
  command line matched its pgrep pattern -> fixed to 'bash ./queue_350.sh'. Zero-shot closed-loop arms finished (clzs_*).
- 14:30 audit accepted: A/B rows verified matched after the restart (248 rows identical incl. u and text dropout). Wording fix:
  zero-shot open -> ours-350 F'+early = 6.7x (DA3 pos) / 6.3x (VGGT pos) / -59% rotation / -20% LPIPS / DINO .380 -> .685;
  same F'+early arm on both models = 5.3–5.5x position. Zero-shot F' (all 10 streams): DA3 c4 1.072 -> .841 (9/10), VGGT
  1.261 -> 1.096 (7/10) but DINO .406 -> .354 (2/10) -> camera tracking alone does not preserve content. Early check on
  zero-shot: VGGT 1.096 -> 1.449, 39/40 restarts; cause not isolated (under-execution vs x0-pose reliability vs fixed kappa).
  plain-350 is a "32 GB Memorizon fine-tune reference", NOT a component ablation (confounds: no LoRA, lr 2e-5 vs 3e-5, prompt
  dropout 0 vs .2, full corpus vs partial first 100, extra branches frozen; not the official full-backbone recipe). After
  queue_350 + queue_plain: correction pairs on 350 from identical history (no new long training arms). queue_ideas.sh retired
  (exit 0) and its main-* cleanup glob replaced.
- 14:45 correction_pairs.py (step 1) + queue_pairs.sh (after queue_plain; smoke 2 states, then 60): own F' history on ours-350,
  gen1 at the planner command, per-call gain r = <e,c>/|c|^2 (clip .3–3), corrected command = along-track (c/r - c) + lateral
  (-(e - r c)/r) + yaw gap, ramped over the 16 waypoints, bounded (.5|c|, 10 deg); gen2 same noise; ReSplat memory L1 at each
  candidate's MEASURED poses (anchor + last two own frames; DiT on host during the render); next-call check; accepted pairs
  stored with the ORIGINAL command as the label. CPU unit test: under-execution .5 -> +.15 forward (bounded), 2x over -> -.15,
  yaw +8 -> -8 deg.
- 15:10 audit of correction_pairs.py accepted, all six fixed (correction_pairs_v2): (1) the next-call branch reuses the already
  measured e1/e2 (commit_measured) instead of re-measuring uncommitted frames (KeyError: 5); (2) missing evidence (DA3 failure,
  memory coverage < .2, next-call unmeasured) -> decision 'uncertain', never 'accepted'; (3) camera judged on the 4-end path
  (mean position + rotation) with a worsening bound on the other measure; next-call check includes rotation and memory L1 against
  the shared history; (4) yaw gap wrapped to [-180,180) (+179 target, -179 measured -> -2 deg); (5) every state stored (x1, x2,
  measured ends, commands, seed, checkpoint sha, scale, text, config) for accepted/rejected/uncertain; run_config.json refuses
  mixed settings; pairs are 'DA3-selected candidates' until an independent check; (6) queue checks the exit code and the state
  count. CPU tests: wrap, decision table (audit case -> uncertain; rot 1 -> 90 -> rejected), branch-commit keys. queue_pairs
  restarted (a running bash script must not be edited in place).
- 15:30 decide(): one evidence check for gen1/gen2/next1/next2 (finite pos/rot/mem/coverage, coverage >= .2) -> 'uncertain' otherwise (audit: g1/n1/n2 coverage and NaN next positions were accepted). Smoke now also asserts both next-call branches ran and every stored state reloads; acceptance not required in the smoke. After the 60 states: decision reasons + independent visual/VGGT check of the accepted pairs decide whether generator training follows.

## 2026-10-22 16:15 — STAGE B WINS BIG: 350 + 250 render rows, evaluated with the OWN-FRAME ReSplat render (causal memory)
A/B matched 250/250 updates (B render rows 1508/2000). Call-4 medians, 9 streams (5f800e62 excluded):
| arm | DA3 pos | VGGT pos | VGGT rot | LPIPS | DINO | sharp/ref |
| 350 open | .458 | .477 | 13.9 | .644 | .563 | .72 |
| 350 + F'+early | .163 | .166 | 7.7 | .537 | .685 | .63 |
| A (350+250, no render) | .459 | .558 | 16.6 | .631 | .646 | 1.16 |
| B (350+250 render) + own-frame render | .089 | .075 | 2.7 | .490 | .721 | 1.14 |
B beats 350-open on 9/9 (DA3), 8/9 (VGGT), 9/9 (LPIPS), 7/9 (DINO); gate vs 350 PASS; exec ratio ~1.0 at every call. A (no render)
does not improve over 350 (VGGT worse 8/9, c1 ratio median 1.63) -> the gain is the render path, not the extra updates.
Visual (frames 5..61, contact sheets): c2282d3f / 284b0dbc / e03b0bf0 track the recording almost frame by frame through call 4
(boulders keep their shape, facade intact); 0d26e9b3 keeps the island and cabinets (open/F' melt into sludge) but bends late;
e7523f6f keeps the playground structure longer but warps at calls 3–4; e83f25bf still degrades late (less than the others).
Render memory is causal: anchor + own generated frames at stored poses (OwnRender864, <= 4 sources). Pending: b350_norender
(same weights, no render at inference), gates, then plain-350; next: F' / F'+early ON TOP of B (measured writes feed the render).
- 16:25 queue_bcl.sh (B + F' / F'+early with own-frame render, gates vs B-open) inserted after queue_plain; queue_pairs now waits for it.

## 2026-10-22 16:50 — audit: pairs move to B with the render path ON; order B arms -> B pairs -> check -> (decision) distill; plain-350 last
correction_pairs v3: --render 1 (own-frame ReSplat render in EVERY generation: history, both candidates, next calls; the corrected
command gets its own render), --history open|fprime (queue picks F' only if B+F' passes its gate vs B-open; early check never
automatic), and the ORIGINAL command's training condition is stored (bank/recent ids, render latent + validity + source ids;
the corrected command's render kept separately). New: verify_pairs.py (independent VGGT path error of x1/x2 after decoding with
their own history + contact sheet), distill_pairs.py (step 2: FM on accepted pairs with the original-command condition, 1:1 with
cached-render recorded rows, init = pair parent; CPU-tested row construction), queue_distill.sh (verify + 2-update smoke only;
the 150-update training waits for the user). plain-350 paused at its update-50 resume slot, resumes last (queue_plain2.sh).
Wording: B's gain = render in training AND at inference (same-amount no-render training does not give it); b350_norender and an
A+render inference check separate the two. Known limit: greenhouse e83f25bf B c4 VGGT 2.09 (DA3 .136) with visible decay.
Chain: queue_bcl -> queue_pairs -> queue_distill -> queue_plain2.
- 16:55 user: priority to plain-350 (original Memorizon recipe) for the comparison -> pause cancelled, plain-350 trains through
  (queue_plain.sh: train + its three closed-loop arms + gates vs ours-350 F'+early), then queue_bcl -> queue_pairs -> queue_distill.
  queue_plain2.sh removed from the chain.

## 2026-10-22 19:50 — three-generator panel (call 4; medians over 9 streams | means over all 10 for DA3/VGGT)
| model / arm | DA3 | VGGT | rot | LPIPS | DINO | mean DA3 | mean VGGT |
| Memorizon-50 zero-shot open | 1.090 | 1.047 | 18.7 | .669 | .380 | 1.072 | 1.261 |
|   + F' | .690 | .748 | 16.0 | .679 | .322 | .841 | 1.096 |
| plain-350 (Memorizon recipe) open | .443 | .573 | 19.0 | .599 | .544 | .502 | .679 |
|   + F' | .079 | .119 | 9.2 | .560 | .655 | .284 | .513 |
|   + F'+early | .065 | .136 | 7.1 | .544 | .670 | .240 | .613 |
| ours-350 open | .458 | .477 | 13.9 | .644 | .563 | .769 | .903 |
|   + F' | .208 | .241 | 7.7 | .555 | .684 | .628 | .715 |
|   + F'+early | .163 | .166 | 7.7 | .537 | .685 | .362 | .495 |
| B (350+250 render) open, render on | .089 | .075 | 2.7 | .490 | .721 | .288 | .322 |
| B + F' (render on) | .037 | .055 | 1.6 | .475 | .651 | .135 | .164 |
Readings: (1) ours-350 vs plain-350 is a wash open loop (4–6/9 wins each way) -> the LoRA / Plucker / roles / text-dropout stack
is NOT supported as a contribution by this panel (plain-350 is a recipe reference, confounded; still no evidence for ours).
(2) F' helps every generator (zero-shot, plain, ours, B), on both estimators. (3) The render path (B) is the largest single
gain, and F' on top of B gives the best camera of all arms (VGGT c4 .055 median / .164 mean, rotation 1.6 deg) with the best LPIPS;
DINO .651 < B-open .721 -> check visually. (4) plain-350 + F' is better than ours-350 + F' on DA3 (.079 vs .208).

## 2026-10-22 20:30 — audit accepted; B pairs moved to hard motions at calls 4–6 with a content check
Audit (verified): b350_norender c4 VGGT .823 / LPIPS .676 / DINO .485 vs render on .075 / .490 / .721 -> geometric evidence from the
own history at inference is necessary (rays/PRoPE path not shown to work alone). B+F' DINO drop is real (7/10 streams: boulders
.778 -> .615, orchard .780 -> .651, beds .672 -> .603; kitchen island still bends) -> control accuracy vs content preservation
is scene-dependent. B+F' VGGT mean gain is dominated by the greenhouse rescue (2.093 -> .351); without it .125 -> .144; all 10:
VGGT pos 6/10, DA3 9/10, VGGT rot 9/10 -> "widespread rotation gain, large rescues of a few heavy position failures". All-10 c4
means: B-open VGGT .322 / rot 4.94 / LPIPS .508 / DINO .645; B+F' .164 / 2.90 / .498 / .630; B+F'+early .203 / 3.57 / .506 / .656
(7/40 restarts) -> F' is the main arm, early an option. Claims narrowed: training additions = "no evidence of contribution"
(not "useless"); generator-independence = "transfer across training/conditioning variants of ONE family (Memorizon-50)" until
another family is tested. Decisive next result: pair-distilled B beats an EQUAL-BUDGET recorded-data continuation of B
(distill_pairs.py --mix 0 control) under independent camera + content checks.
First B pairs (calls 2–3, F' history): errors already tiny (pos .01–.05) -> 1/8 accepted, the rest below the .05 margin;
thresholds NOT loosened. correction_pairs v4: hard-motion filter (planned call translation >= p75 .557 units or |yaw| >= p75
36.8 deg; calls 4–6), DINO content check (candidate vs memory render, reject if worse by > .02). Restarted as pairsB2 (smoke 2 ->
60); queue_distill points to pairsB2.
- 20:40 user: no distillation smoke; queue_distill = verify only.

## 2026-10-22 21:10 — audit: four fixes before any pair training; claims retracted; reward arm not queued
1 Verification now gates training: approve_pairs.py writes <pairs>/train_manifest.json = accepted at collection AND VGGT agrees
  AND visually approved by a named reviewer (keys listed explicitly) AND file SHA256 + parent receipt verified; distill_pairs.py
  reads ONLY this manifest (re-checks SHA + parent). Producing a contact sheet is not a visual check.
2 correction_pairs v5: DINO is required evidence (NaN -> uncertain, all four records), next-call DINO stored and compared;
  uncovered pixels are set to the same grey in candidate AND render (no candidate content in the reference); the score is an
  auxiliary memory-consistency measure, NOT a shape guarantee.
3 verify_pairs: VGGT uses the collection's OR logic (position gain with rotation kept, or rotation gain with position kept);
  margins fixed BEFORE any verification from VGGT on the recordings (floor c4 .034 units / .48 deg -> .04 units / 1.0 deg).
4 distill_pairs v2: matched control — every row one target chunk with render slots filled (pairs: stored original-command
  render; recorded: precomputed window render, render_p 1); both arms draw the same 8 recorded rows per update (same sampler
  seed and noise generator); the pair arm replaces slots 1,3,5,7 -> equal target chunks, equal render dose, shared recorded rows;
  GPU time logged per update. --target x1 = control for "own-history exposure vs verified correction".
Retracted: "a refiner can only work at u in [.85,.95]" and "if pair training fails the refiner fails" — the probe measured
command switching on pilot-100 (5 scenes, call 1, no render, fwd/right), not the capacity of a trained error-conditioned
branch; refinement is deferred for budget, not shown impossible. reward_train.py is an NFT-family variant (fixed start-weight
reference, candidates from the current policy, summed reward normalised jointly), not a faithful DiffusionNFT/WorldCompass port.
Pairs restarted as pairsB3 (v5 config); pairsB2's 5 states kept as a record. Order: pairs -> verify -> visual review ->
manifest -> (user) distill x2 vs matched control (and x1 control if x2 wins). Report unique scenes and error types.
- 22:05 audit fixes: verify_pairs records the file SHA256 at verification; approve_pairs accepts only if the current hash equals it
  (tamper test: replaced file -> dropped "file changed since VGGT verification"); distill also checks the file's own
  config.receipt_sha and record.key against the manifest; vggt_agrees returns None (uncertain) on non-finite values; manifest
  counts real scene_id (873 scenes) and streams separately. Thresholds .04 / 1 deg are pre-set from 5 recordings x 4 calls, not
  confidence bounds.
- 22:05 pairsB3 so far 20 states: 0 accepted (15 camera gain insufficient, 3 memory/DINO worse, 1 next call worse, 1 uncertain).
  B's first candidate on hard motions: pos median .069 (p75 .72, max 2.38), rot median 2.2 deg (max 71); the one-step corrected
  command improves position in only 11/20 and sometimes makes it much worse (-1.32 units, -44 deg) -> B's residual errors are
  not well modelled by a per-call gain; correction pairs may stay scarce on B.

## 2026-10-22 22:15 — what is actually known about refinement / reward / error-loss (user: are we sure they do not work?)
NOT shown to fail on the current models; most were never run on them:
- Learned error-conditioned refiner: never built.
- Reward arm (reward_train.py) and camera-ReFL: never run (ReFL on Memorizon stopped at the pose-head gate on 10-16; DMD v2 on
  10,250 OOMed). Pair distillation: no approved pairs yet (pairsB3 22 states: 21 rejected, 1 uncertain).
Shown to fail, in narrow settings only: (a) training-free re-noise at u .3-.9 does not move the camera (pilot-100, call 1, no
render, 5 anchors); (b) FM fine-tune / refiner training on the released 4-step DMD student blurs (Oct 13-16); (c) the one-step
corrected command on B helps position in only 11/20 hard states (pairs). (c) concerns the correction rule, not training.
Decisive cheap test queued (queue_spread.sh after the pair verification, ~1 h): seed_spread_B.py — 12 hard own-history states x
8 noises on B: if candidate camera errors differ by more than the measurement floor (.04 units / 1 deg) and the camera-best is
not the content-worst, a group-relative reward arm has a signal and is worth its ~4-5 h; if not, it cannot learn anything here.
- 22:45 host OOM (my 3 parallel CPU trajectory jobs, ~8 GB each) killed pairsB3 at 25/60 (22:23) and seed_spread_B (22:30). CPU jobs stopped; trajectories recomputed on the idle GPU in one process (2 min); queues restarted in order (pairs resume from 25 -> verify -> seed spread).
- 22:55 user spotted it on the page: in single-frame ReSplat renders the black (no-Gaussian) areas do not match what the frame sees.
  Measured: rendered from (almost) the SAME pose as the source frame, coverage is only .37/.70/.67/.49/.62 (5 dev streams) with one
  source, vs .995 with 4 sources (check_metrics_864). DA3 sky/confidence mask is 1.0 everywhere and gauge depth is sane (median
  .97), so the holes come from ReSplat itself when the single source is duplicated (multi-view matching degenerates). Affects the
  first call at inference (render_own uses only the anchor) and the start=1 training windows. Candidate fix: depth point-cloud
  splat (or a different single-view path) when only one frame exists; measure call-1 quality before/after.
- 23:30 fairness: the page's "Ours, no 3D memory" was ours-350 (350 steps) while the memory model B had 600. The matched
  600-step no-memory model A (same rows, render off) exists: open loop c4 VGGT median .558 / mean 1.39, LPIPS .631, DINO .646 ->
  not better than ours-350; B's gain is not the extra 250 steps. Queued (queue_10call_c.sh): A + closed-loop 4-call arm, A DA3
  paths, A open/closed-loop 10-call arms; the page will use A as the no-memory model once they finish.
- 23:30 reviewed the "Gaussian feature memory / persistent Gaussian world" proposal: facts verified (Gaussians carry means,
  covariances, harmonics, opacities; ReSplat returns final_state; incremental_map.py GaussianMapState is an untrained transfer
  experiment; 18 Sep hard persistence worsened boundary L1 +24%). Assessment on the page (ideas section): feature-read memory is
  close to Robust Dreamer / Mirage / WorldCrafter, so "no RGB" alone is not new; the persistent world updated by diagnosed errors is
  the strongest long-term direction but post-deadline unless a pilot works by 1 Nov.
- 9 Nov 20:05 user: evaluate the GM line on the SOTA-comparison scenes. sota_eval/: run_arm_ckpt.py = gaussian_meas_dl3dv7_20261103/run_arm.py with --receipt/--frontier (diff: receipt, sys.path, optional frontier.install tol 1e9; otherwise identical: batch_continuous1009 inputs, 30 calls x 16, seed 7, steps 30, guidance 1, max_bank 4, dup2 first render). queue_sota.py (waits for done_inside): phase 1 baseline arm for gm_b, gm_f_inside, gm_inside (control) on the 5 cohort-B scenes (02267acf 0238de14 334ff324 61b27b61 a126efe8) -> baseline-only scores (score_reliability --style long --horizon 480 --model memorizon) -> table_sota_baseline.md; phase 2 correction arm -> full scores -> table_sota.md (build_table_sota.py reuses comparison_table_20261105 metrics + rows). ~30 runs x ~13 min + scoring => ~7 h after the inside queue (~21:30) => ~04:30-05:00.
