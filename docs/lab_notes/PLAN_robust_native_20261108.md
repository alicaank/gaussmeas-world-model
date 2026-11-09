# Robust native Gaussian model: training plan (8 November 2026)

User decision (7 Nov, 23:30): the 72-hour window does not matter; what matters is
that the model trains well and that accuracy and long-rollout robustness improve.
This directory is separate from `analysis/gaussverse_recovery_20261106/` (another
session's campaign); it imports that code read-only and never writes there.

## Evidence this plan rests on

- Three generated-history continuations (pilot250 37.5% share / 8-bit, camera-rate100
  37.5% / FP32, support200 11% / coverage-gated) produced the same signature: command
  over-execution (exec ratio 1.3–3.7), independent camera error 3–10x worse, sharpness
  halved. The clean-history counterpart (clean100) improved the camera. Cause: generated
  context + measured labels + recorded targets conflict in content and in camera (lagging
  measured history versus the recorded target teaches a systematic catch-up prior).
  The old stage that worked (run_op/meas) used 1–2-call histories on 300 scenes, where
  both conflicts are small.
- Native250 (clean rows, every row rendered) depends on the render: gaussian-off is worse
  than the initializer. The earlier model had render_p 0.75.
- The early strength of the earlier model is reprojection + inpainting: projected anchor
  coverage 0.9→0.0 over calls 1–10; LPIPS gap to Lyra-2 opens only after frame 160;
  a history reset at 160 restores quality but re-drifts at the same per-call rate.

## Recipe

All stages: native Memorizon packing/retrieval/FM (10 query chunks where VRAM allows),
recorded targets, FP32 PRoPE moments, LoRA LR 1e-4 (2e-5 was inert), accum 4.
Stage B reuses the recovery render cache as is: 870 windows, one per stream (the "two windows
per scene" idea is not implemented; 250 updates x 4 rows cover the 870 windows once).

- **B0 control**: clean adaptation, render_p 0.75.
- **B robustness core** (`corruption.py`), on recorded context only, targets and labels
  unchanged, anchor never touched (rates aligned 8 Nov with Lyra 2.0 / GEAR / Robust Dreamer,
  see "Literature check" below):
  1. one-step self-augmentation, p 0.5, sigma ~ U(0.05, 0.5): noised context is pulled
     to its x0 estimate by the current model (context-only forward, no grad); the
     context becomes generated-like while poses stay recorded;
  2. otherwise context noise, p 0.3, sigma ~ U(0, 0.3) in the FlowMatch law x = (1-s) x0 + s eps;
  3. render perturbation follows the context sigma; validity/confidence scaled by (1-s);
  4. render dropout, p 0.25.
- **C short-age on-policy, refreshed**: pool from the B checkpoint, >=300 scenes, ages
  1–3 calls, measured writes during generation, measured labels; rows admitted only if
  the observer is valid AND the boundary error is small (|t| < 0.15 anchor units,
  rot < 10 deg); share 25%; pool regenerated from the current checkpoint every 100
  updates. Histories >= 5 calls are never paired with recorded targets.
- **D long-age context forcing** (after C): same weights, clean-context teacher without
  grad, generated-context student, loss on the velocity difference; no recorded target.
- **Inference**: measured writes + rejoin controller; render-off beyond a coverage
  threshold is a tested switch, not a default.

## Resolution: 864x480 (user decision, 7 Nov 23:50)

ReSplat has trained weights only up to 540x960 (local: 256x448, 540x960); at 864x480 the
render (540x960, downsampled) carries more detail than the latent grid, and every pipeline,
cache, reference model and the 60-scene test set already exist at 864. DA3 pose measurement
runs at process_res 504 regardless. 1280x704 would be teacher-native but needs upsampled
renders and 2.2x tokens (offload or 5-chunk rows); 960x544 adds 26% pixels for a full
re-preparation and loses comparability. The 2K download (`data/DL3DV-source2K-1280x704`,
identical COLMAP poses, verified) is kept as a sharper ReSplat source for a later,
separately qualified render variant (needs dev/test 2K frames for evaluation parity).
`prep1280.py` is retained but not scheduled.

## Gates (revised 8 Nov 13:00, user approval "dedigin sekilde duzelt")

Selection evaluation: 6 development scenes (>= 321 recorded frames, corpus order from offset 2,
includes the two review500 scenes) x 1 inference seed x 320 frames, sampler CFG 1/30 for every
arm (matches the earlier Gaussian model; beat CFG 1.5 and 4/50 on native500). Scene-paired late
LPIPS (calls 11-20) and independent VGGT position; tripwires: median exec ratio > 1.2 or sharpness
< 0.8x the control rejects; render-off arms for the dropout-trained models only. Second inference
seed and the 60-scene test set only for the final model. The second training seed is dropped:
scene variance dominates, and the GPU time goes to continuing the winning recipe from native500
(`queue_stage_b.py: decide()` applies the gate automatically and writes eval_s1/decision.json).

## Verification (8 Nov, user request "her şeyin doğru olduğuna emin ol")

Pose / resolution chain at 864x480, checked in code and data:
- K864 = transforms.json intrinsics scaled by the 960P crop (cx 432, cy 240); PRoPE normalizes K
  by camera_config 864/480 and uses 27x15 patches; retrieval frustum width/height = 2cx/2cy.
- c2w: OpenGL->OpenCV (diag(1,-1,-1,1)), anchor = identity, anchor-depth units; Memorizon gets
  metres/4 via metres_per_anchor_unit (DA3 metric depth of the 448 anchor over the gauge depth at
  the same resolution). One pose per latent = pose of RGB frame 4i, which is Memorizon's own
  semantics (pose index i = state after i x 0.25 s = frame 4i at 16 fps).
- Independent check: VGGT on the RECORDED 864 frames with our K and anchor gauge reproduces the
  labels to 0.007–0.03 anchor units and <1 deg (native_release_control/camera_control.json),
  so labels, K and scale are mutually consistent at this resolution.
- ReSplat: frames upsized to 540x960 with K rescaled, output downsized to 864x480; DA3 poses at
  process_res 504 (resolution-independent). 2K poses are byte-identical to the 960P poses.

What is established vs. hypothesis:
- Established: generated-history + measured-label + recorded-target rows cause over-execution and
  camera collapse (3 runs, clean counterpart does not); native250 depends on the render
  (gaussian-off worse than the initializer); LoRA at 2e-5 barely moved (relative ||dW|| ~1e-3 vs
  ~9e-2 for PRoPE, measured on adapter_00250); released Memorizon (teacher and 4-step student)
  drifts to ~1 anchor unit of position error within 96 frames on DL3DV while the earlier
  Gaussian-meas parent stays at 0.17 (same frames, VGGT): the ReSplat render channel is what
  executes commands on real scenes.
- Hypotheses the queue tests: render_p .75 removes the dependence (gaussian-off arms); the early
  gap is partly the sampler (CFG 1.5/30 eval); corruption training (B) improves late quality AND
  camera relative to B0 (two seeds); Gaussian channel lowers FM loss (probe_fm).
- Untested, optional: separating "generated content" from "measured labels" as the cause of
  over-execution needs a 2x2 training experiment; the plan avoids both factors, so it is not
  required for stage B. Caption format (ours: Qwen3-VL-2B object description without Memorizon's
  "First-person perspective" prefix) is a possible CFG-4 confound; low priority at CFG 1.5.

Literature check (fetched 8 Nov):
- Memorizon (2610.00544): UE5 synthetic only (380 episodes, 50 scenes, 95 h), 6k steps x batch 32,
  full backbone 2e-5 / camera 1e-4, spans 10–100 s, 20 steps CFG 4; retrieval raises Revisit, but
  quality falls with span and the paper attributes it to exposure bias; distillation effect on that
  gap "remains to be measured"; no real-footage training, no drift metric. Our domain adaptation on
  DL3DV is therefore necessary, and its weak command execution on DL3DV is expected.
- Lyra 2.0 (2604.13036): self-augmentation = noise history latent t~U(0,.5), ONE denoising step,
  p .7, recorded targets; removing it drops style consistency 85->78 and camera controllability
  64->54. GEAR (2609.34722): same mechanism, p .4, tau<=.3, enabled for the last 20% of training;
  without it ATE .044->.066, LPIPS .446->.476. Robust Dreamer (2605.30855): one-step deviation
  archive, p .9 with clean p .5, LoRA r128, recorded targets. All three corrupt RECORDED history
  with the model's own one-step error and keep recorded targets; none trains on long
  self-generated histories with recorded targets. Self-Forcing / DMD / History Forcing
  (2611.02782) need teacher + critic copies and are out of reach on one 32 GB GPU.

## Order

1. native500 evaluation (CFG-matched) + render-off diagnostic + Gaussian on/off FM probe.
2. 1280 corpus: latents, scale, captions (`prep1280.py`), then ReSplat-1280 qualification.
3. B0/B at 864, two training seeds (recipe selection).
4. 1280 render cache; B0/B at 1280, two seeds.
5. C, then D, then final evaluation.

## Run log (8 Nov)

- Queue lost the GPU to the recovery review500 at 00:32 and exited; relaunched 08:45 with retry-only waiting.
- Queue environment fixes: CUDA_HOME=cuda128 (gsplat otherwise disables its rasterizer: probe_fm failed),
  expandable_segments (profile OOM: self-aug forward leaves ~3 GiB fragmented cache; training peak 30.17 GiB of 31.35).
  Steps skip when their output exists; crashed training resumes from resume.pt.
- probe_fm (2 dev windows x 3 seeds, sigma .3 context noise): FM loss clean/noised, render on/off
  initializer .2604/.2829 (render inert), native250 .2324/.2609 on, .2412/.2714 off,
  native500 .2301/.2580 on, .2413/.2720 off. Corruption damage .023 (initializer) -> .028 (trained):
  adaptation made the model slightly more sensitive to corrupted context; the render offsets more of the loss under noise (.011 -> .014).
- profile_b_s1 (largest layout, forced self-aug): peak 30.17 GiB, 80–90 s/update.
- Added CFG 1 long-rollout evaluation (native500 first, then the B0/B checkpoints) in eval_cfg1/.
- Visual check of native500 (recovery review500 videos): kitchen keeps layout to frame 320 where the earlier model inserts
  foreign objects; native500 failure modes = semi-transparent doubling and invented content outside the anchor view;
  showroom fails at frames 160 and 320 for both models alike.
- eval_cfg1 (native500, CFG 1/30, same 2 scenes x 2 seeds x 20 calls as review500): versus CFG 4/50, LPIPS all calls .591 vs .609,
  VGGT position .336 vs .394, rotation 10.8 vs 13.7 deg, last-5 position .556 vs .702; DINO calls 11-20 .423 vs .459.
  Sampler-matched against the earlier model (CFG 1/30): earlier still better on LPIPS (.571) and DINO (.475), native500 2x better camera
  (earlier .764 / 22.5 deg). Gap is in the kitchen: native500 at the right pose but blurred (Laplacian sharpness 125 vs recorded 380;
  showroom 234 vs 216), earlier sharp but at the wrong pose (rotation 26–51 deg). Exec ratio median 1.04–1.12 (no over-execution).
- 12:55 restructure (user approved): old queue parent stopped (B training kept running, PID 1422262); eval_s1 = CFG 1/30, 6 scenes x 1 seed,
  arms initializer / native500 / b0_s1 / b_s1 (+ render-off for b0/b); eval_cfg1 and training seed 2 dropped; if B passes the gate vs B0,
  B continues from native500 for 250 updates (fresh optimizer, `--init-adapter`) and is evaluated under the same protocol.
  LoRA movement check: mean ||BA|| b0_s1 0.39 vs native500 0.14 vs native250 0.09 (LoRA at 1e-4 learns). Memorizon's training loss keeps
  context at the clean timestep, so stage-B corruption (latents corrupted, timestep clean) reproduces the inference-time condition.
- 16:35 gate fix: on the six-scene evaluation the DA3 path ratio of B0 and of native500 are both median 1.21 (observer jitter on
  short-baseline scenes), so the absolute 1.2 tripwire would reject any arm; changed to B <= max(1.2, B0 + 0.1). Queue parent restarted
  (evaluation child untouched). Partial (DA3 online camera, calls 1-20): B0 position .62 vs native500 .44, B0 late sharpness 411 vs 354;
  B0 render-off position 1.21 vs initializer 1.69 (render dropout removed the off-collapse native500 had, but off exec ratio 1.67).
- 17:34 eval_s1 complete (6 dev scenes x seed 7, CFG 1/30, 20 calls; VGGT on recorded frames reproduces labels to .01-.06 except e7523f6f .32).
  Means (late = calls 11-20): LPIPS 1-3 / late LPIPS / late DINO / VGGT pos / last-5 pos / rot deg / late sharpness / exec median:
  initializer .599/.783/.178/1.80/1.89/58.3/500/0.73; native500 .423/.609/.484/0.66/0.85/12.7/354/1.21;
  b0_s1 .450/.632/.393/1.20/1.30/26.6/411/1.21; b_s1 .430/.610/.478/0.89/0.84/23.6/480/1.21;
  render-off b0 .580/.726/.231/2.06/2.18/55.9/716/1.67, b .574/.724/.303/1.42/1.11/55.6/1101/1.69 (recorded sharpness mean 471).
  Gate B vs B0 PASSED (late LPIPS wins 5/6, VGGT wins 4/6, sharpness 480 vs 411, exec 1.21 vs 1.21). B at 250 updates from the initializer
  matches native500 (500 clean updates) on late LPIPS/DINO and is sharper (480 vs 354, recorded 471), but its camera is worse (pos .89 vs .66,
  rot 23.6 vs 12.7 deg; rotation loss sits in 0d26e9b3 35.7 vs 6.5 and 5f800e62 66.8 vs 37.2, a bright low-texture classroom: recorded sharpness 46). Render-off: dropout lifts B above
  the initializer (LPIPS .724 vs .783, pos 1.42 vs 1.80) but over-executes (1.69) and is over-sharp (noise); B0-off camera is worse than the initializer.
  Queue started b_from500_s1 (B recipe from native500, init sha 261565eb verified in its manifest).
- 18:20 user asked for a native250 comparison: native250 was never run under the eval_s1 protocol (dropped at 12:55). queue_followup.py (PID 1530547)
  waits for the main queue to exit, then adds native250 (render on) to eval_s1 with every other arm reused. Interim view on the two review scenes
  (seed 7), sampler-mismatched (native250 only exists at CFG 4/50): native250 late LPIPS .654, VGGT .62 / 23.4 deg; native500 at CFG4 .626 / .34 vs
  CFG1 .604 / .26 on the same scenes; B0 .642 / .67 / 33.6 deg, B .624 / .49 / 21.0 deg (CFG1).
  visual_compare.py: contact sheets (eval_s1/visual/sheet_*.jpg) and 2x2 videos (grid_*.mp4). Visual read: B0 dissolves into semi-transparent
  smears after call 5 in most scenes; B keeps solid, sharp structure but invents more of it (playground, kitchen call 20); native500 keeps layout
  best but blurs.
- 18:35 user: matched native250 evaluation not worth the GPU time; queue_followup.py (PID 1530547) stopped while still waiting (no GPU job started).
  native250 comparison stays the interim, sampler-mismatched one above.
- 21:15 b_from500_s1 (B recipe, 250 updates from native500, fresh optimizer) evaluated in eval_s1 (all arms one scorer pass).
  Means: LPIPS 1-3 .419 (native500 .423), late LPIPS .596 (.609), late DINO .495 (.484), late sharpness 488 (354), VGGT pos 1.04 (.66),
  median-scene pos .64 (.42), rot 14.4 (12.7), DA3 exec median 1.41 (1.21). Render-off: .725 LPIPS / 1.59 pos / exec 1.76.
  Against native500 it would FAIL the plan gate: late LPIPS wins 3/6, VGGT wins 2/6, exec 1.41 > 1.31. Camera loss concentrated in
  e83f25bf (greenhouse, pos 3.34 vs 1.30, exec 1.94) and 0d26e9b3 (kitchen, rot 16.2 vs 6.5, DA3 path 1.77 but net displacement .72 =
  zig-zag, not overshoot). Visual: sharper, same late failure modes (ghosted structures in the kitchen, generic plants in the greenhouse).
  Reading: every 250-update run with render dropout .25 + LoRA 1e-4 (B0, B, b_from500) has a worse camera than the render-always,
  LoRA-2e-5 native recipe (native500; native250 at CFG4 .62 on two scenes vs B0 .67 at CFG1); corruption helps appearance and is
  neutral-to-positive on camera relative to B0. Suspected camera cost: render dropout and/or render perturbation weaken render-following.
- 21:55 user: "is the problem really dropout?" probe_fm_render (same 2 dev windows x 3 seeds, sigma .3): render benefit clean/noised
  native500 .0111/.0140, b0_s1 .0066/.0082, b_s1 .0066/.0069, b_from500_s1 .0087/.0087; noised-context damage native500 .028 -> B .016.
  Dropout runs DO lean less on the render (-22..-40% FM benefit), but B0 and B have the same reliance and very different cameras, and the
  single-seed camera noise floor (native500 kitchen s7 vs s107: pos .28/.52, rot 6.5/23 deg) covers b_from500's regressions except the
  greenhouse. Dropout = plausible partial cause, not established. B-r (arm 'br', queue_br.py, tests 12/12) implemented but NOT launched.
- 22:10 user redirects to the earlier Gaussian-meas model (fresh_teacher run_op/meas: packed Memorizon-layout, full backbone 2e-5, render_p .75,
  n_chunks 2, max_bank 4, scale x2, 50% on-policy rows with 1-2-call own histories + measured poses + recorded targets; 250 updates = 1.9 h).
  Its late failure on the review scenes is NOT blur: per-call LPIPS beats every native arm at almost every call, but VGGT rotation has
  catastrophic excursions (kitchen s7 117/93/90/70 deg at poses 5/7/15/17; s107 120-172 deg at 16-19; native500 max 19 deg).
- 22:03 GM-B launched (user: "sadece GM-B 250 adim egit ve test et, egitilmis bozulmamis checkpointten"): queue_gm_b.py (PID 1595809) ->
  train_gm_b.py = run_op/meas config verbatim (train_onpolicy --op_arm meas q .5, lr 2e-5/1e-4, render_p .75, n_chunks 2, max_bank 4, accum 8),
  init run_op/meas trainables_00250 (sha 1dc0ba13), sampler seed 20261108, + gm_b_corruption.py on recorded rows only: self-aug p .5
  s~U(.05,.5) via an auxiliary packed row (model's own one-step x0 of latents start-4..start-1 from their recorded context; bank noised at s;
  rows with < 4 recent latents fall back to noise), else noise p .3 s<=.3; render slots noised at s, (valid,conf) x (1-s). Pool rows untouched.
  Tests test_gm_b.py 8/8. Then closed_loop_864.py (20 calls, CFG 1/30, write 1, render 1, replan 0, VGGT 1, offset 2 limit 8 -> the six
  eval_s1 scenes) for GM parent and GM-B -> compare_gm.py -> eval_gm/compare.json (late LPIPS/DINO, sharpness, DA3, VGGT pos/rot, rot > 30 deg count).
- 22:20 user standing order for the GM-B result (~02:15): if GM-B improves on GM (late LPIPS/DINO or camera wins >= 4/6 without a
  regression on the other, rotation excursions not worse), extend the same training until 08:00 (train_gm_b.py --resume 1, higher
  --updates, --max_hours to 07:45, ckpt every 250); if not, diagnose the failure in detail (per-scene/per-call, corruption records,
  render/camera traces), fix the recipe and train the fixed recipe until 08:00.
- 9 Nov 00:02 GM-B training complete (250 updates, final loss .255, peak 24.6 GiB, 24-33 s/update). eval_gm_s7 started 00:03 and its log
  stops at 00:11:31; the host was down until a reboot at 11:46 (kernel 7.0.0-31 -> -42). The overnight extension order could not run.
  11:55 queue relaunched (PID 5231; training skipped, partial GM eval set aside as eval_gm/gm_s7_partial_reboot).
- 9 Nov 12:30 FRONTIER FOCUS (user: "model drifts the border instead of generating" -> "buna odaklan, duzeltmemiz gerekiyor").
  Verified mechanism: ReSplat valid = accumulated alpha > .5, conf = alpha (gaussverse/memory/resplat_condition.py:66-70): a Gaussian
  stretched into newly revealed space is valid + confident; packed_model masks render tokens only where pooled valid < .5; the model
  follows the render (~2:1 in conflict) -> extrudes the last surface at the frontier instead of generating (02267acf frames 160-184:
  poster/stand stretched where the recording reveals the shelf corner). Training cache: only 11% of render cells valid < .5, border
  validity .870 vs interior .886 -> frontier regions rare in training, dominant at deployment after anchor coverage -> 0.
  Fix line: observed-support mask (frontier.py: >= 1 source's DA3 depth points land on the cell, renderer864.splat_depths, closing r=1,
  rgb/valid/conf zeroed where unobserved = renderer's own "no splat" convention). Step 1 = render-only smoke sheet + inference-only
  test on the GM parent (closed_loop_frontier.py, 6 scenes, same protocol) queued after the GM-B evaluation (queue_frontier.py).
  Step 2 (if the mask helps or is neutral): training with the mask + regional render dropout (needs the support channel in the render
  cache: re-render 870 windows ~3.6 h, or a conf-threshold proxy if the smoke shows conf separates frontier cells).
- 9 Nov 13:05 GM parent on the six scenes (closed_loop_864, CFG 1/30, write+render, VGGT): LPIPS 1-3 .367, late LPIPS .601, late DINO .403,
  late sharpness 262 (recorded 458), DA3 pos .533, exec 1.03, VGGT pos 1.04 (last-5 1.50), rot 20.1 deg, 3.5 poses/scene with rot > 30 deg.
  (native500 on the same scenes, eval_matched scorer: late LPIPS .609, DINO .484, VGGT pos .66, rot 12.7.)
  Frontier fix chain queued: queue_frontier.py (PID 34680: smoke + GM+mask inference test) -> queue_gm_f.py (PID 39062: decision, mask cache
  for all 3553 training windows (DA3 only), GM-F 250 updates from run_op/meas with corruption iff GM-B beat GM, eval with mask, compare).
  Design: mask multiplies (valid, conf) only (render tokens with pooled valid < .5 are dropped by packed_model; cached latents stay valid for
  partially observed cells) + regional strip dropout p .3 on recorded render rows. Tests: test_frontier 3, test_gm_f 5.
- 9 Nov 13:37 GM-B vs GM (6 scenes, closed_loop_864 CFG1/30 write+render VGGT): late DINO .442 vs .403 (5/6), late sharpness 379 vs 262
  (recorded 458, 5/6), VGGT pos .774 vs 1.036 (4/6), rot 17.3 vs 20.1, poses with rot > 30 deg 2.67 vs 3.50 per scene (kitchen 7 -> 2,
  playground 4 -> 1); but late LPIPS .623 vs .601 (worse 4/6; +.06 classroom, +.05 playground, +.04 kitchen); LPIPS 1-3 equal (.364/.367).
  Verdict: partial improvement (identity, sharpness, camera excursions) with an LPIPS cost -> not a clean win; the automatic GM-F rule
  therefore trains the frontier fix WITHOUT corruption (clean attribution of the mask); combination is the follow-up if both help.
- 9 Nov 14:20 frontier mask v2 = reprojection test (frontier.observed_from_reprojection: rendered depth -> 3D -> each source; observed if
  inside the source image (margin 2 px) and within 15% of that source's DA3 depth). Point-support v1 was too sparse under forward motion
  (smoke: speckled, 57% cells dropped on a 5-call jump; its inference run stopped and set aside as eval_gm/gm_fmask_support_partial).
  Smoke (c3cfc2ff, target latent 20): rollout-like sources (anchor + latents 17-19): alpha-valid .98, reproj observed .915 (7.7% cells
  dropped, frontier 6.6%), true-depth (DA3 of recorded targets, the training-cache rule) .934 (6.2%), reproj~true agreement 97.5%; the
  mask removes exactly the smeared left wall and the stretched foreground bed the alpha test passes. Far sources (latents 0-3, 5-call
  jump): render is a smeared tunnel, reproj drops 94% (render disagrees with the sources everywhere) while true-depth says 80% was seen:
  the two rules diverge only when the render geometry is wrong (not the deployment condition: renders are one call ahead).
  Tests 19/19 (frontier 6, gm_f 5, gm_b 8). Queues relaunched: queue_frontier (GM + mask inference eval) -> queue_gm_f (decision, cache,
  GM-F training without corruption, eval with mask, compare).
- 9 Nov 14:40 user overrules the "partial" verdict: GM-B leads on 7/9 metrics (DINO, sharpness, VGGT pos/last-5, rot, excursions, DA3) and
  the LPIPS +.022 is the sharpness cost -> GM-B counts as an improvement. GM-F therefore = GM-B recipe (corruption) + frontier mask +
  strip dropout, same init (run_op/meas) and 250 updates, compared against GM-B at equal budget; the winner is extended afterwards.
- 9 Nov 14:37 GM + mask (inference only, tol .15, no training) vs GM: means neutral (late LPIPS .603/.601, DINO .404/.403, VGGT pos 1.06/1.04),
  early LPIPS worse (.397 vs .367, 0/6: the untrained model loses render it relied on), kitchen much better (rot 33 -> 13 deg, excursions
  7 -> 3, DINO .39 -> .45), classroom/playground worse. Decision rule passed -> cache build started 14:36, GM-F training follows.
  FINDING: in rollouts the mask drops far more than in the smoke/training: cells dropped mean 42% (p90 81%), per scene 5-30% (storefront,
  showroom, kitchen) but 40-95% (greenhouse, playground, classroom) from call 2 on; training cache so far ~3-6%. Cause to establish: the
  ReSplat depth vs DA3 source depth disagreement on generated sources / thin structures at tol .15 (training uses DA3 vs DA3). Diagnostic
  planned after training (GPU): tolerance sweep .15/.35/inside-only on e7523f6f + e83f25bf with mask dumps; the GM-F evaluation tolerance
  is chosen from it (training is unaffected: it uses the true-depth cache).
- 9 Nov 15:42 frontier_cache done: 3553 windows, 62 min (one DA3 target frame per latent), tol .15, cells dropped mean .233 p90 .517, frontier (alpha-valid but unobserved) mean .178 p90 .386. train_gm_f_s1 started (GM-B recipe + frontier mask on recorded render rows + strip dropout p .3, corruption on by user decision, seed 20261108, from run_op/meas 250). First updates: loss .339/.294/.251 finite, grad .14, 28-33 s/update, peak 24.5 GB; rep['gmf'] shows mask on recorded render rows (cells_unobserved .06-.40), strips (left .24, bottom .43), on-policy rows untouched; gmb ops self_aug/noise/none as in GM-B. ETA ckpt ~17:50, eval (mask tol .15) ~19:10, then frontier_tol_diag.
- 9 Nov 15:50 visual_compare_gm.py (sheets/2x2 videos for the GM line from comparison_<sid>.mp4 + eval.json per-call LPIPS). Interim sheets eval_gm/visual_pre (GM, GM-B, GM+mask): kitchen 0d26e9b3: GM loses the room at call 5 (counter geometry warped, LPIPS .59) and shows a smeared blob at call 15 (.67); GM-B and GM+mask keep the room layout (chairs, wall, window) to call 20, both with slightly lower late LPIPS at this scene only on the mask arm (.63 vs .65). Greenhouse e83f25bf: all arms splat-textured (speckle, see-through foliage) from call 10; corruption does not fix texture here, mask neutral. Final sheets/videos after GM-F eval.
- 9 Nov 18:29 done_gm_f. GM-F (250 upd, corruption + frontier mask + strip) evaluated WITH the inference mask (tol .15): late LPIPS .621 (GM-B .623, GM .601), DINO .448 (best; GM-B .442), sharp 391 (GM-B 379), DA3 pos .542 (GM-B .471), VGGT pos .935 (GM-B .774), rot 20.8 (GM-B 17.3), rot>30 3.33/scene (GM-B 2.67); kitchen rotation back to 33 deg / 6 excursions (GM-B 11.5 / 2). Inference mask dropped 43.5% of cells on average (p90 75%), call 1 ~61% in every scene (single source) -> the inference mask starves the render memory; not a GM-B win. Next: frontier_tol_diag (GPU free now) + mask dumps to find why the rollout drop rate is 2-3x the true-depth cache rate, then re-evaluate GM-F at the chosen tolerance (and unmasked as a control).
- 9 Nov 18:51 frontier_tol_diag (GM parent, 8 calls, greenhouse e83f25bf off4 + playground e7523f6f off5, mask dumps in eval_gm/frontier_tol_diag/*/dump): the DEPTH-AGREEMENT test is what inflates the rollout drop rate, not the frustum test. Cells dropped per call: playground tol .15 .64 .57 .52 .52 .77 .75 .53 .87 | tol .35 .52 .26 .29 .25 .35 .53 .40 .40 | inside-only .23 .04 .07 .05 .06 .22 .09 .06; greenhouse .15 .69 .49 .56 .34 .58 .41 .51 .35 | .35 .61 .25 .34 .31 .26 .22 .23 .13 | inside .54 .06 .13 .11 .04 .11 .18 .10. Inside-only also gives the best LPIPS and position (playground call-1 LPIPS .19 vs .34 at .15, pos max .15 vs .43; greenhouse LPIPS equal, pos excursion similar). Cause (dumps): ReSplat's rendered depth is an alpha-blended soft depth that disagrees with DA3 depth on thin foliage/structures and in smeared regions where misregistered generated sources are fused, so a 15% relative test rejects correct render; the training cache (DA3 vs DA3) never had this disagreement (23% dropped). Call 1 drop (.5-.7) is mostly alpha-invalid already (single-source ReSplat valid .48 in the greenhouse: foreground basket missing from the first render) - a separate first-step-memory issue. Decision: inference mask = frustum-only (tol 1e9). queue_inside.py launched: eval GM-F and GM with the inside mask (6 scenes) -> compare_gm_inside.json. Visuals (GM, GM-B, GM-F masked .15) in eval_gm/visual.
