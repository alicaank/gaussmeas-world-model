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
- 9 Nov 20:35 done_inside. Frustum-only mask (tol 1e9), 6 dev scenes: GM + mask (inference only, NO training) is the best arm on every aggregate: late LPIPS .590 (GM .601, GM-B .623), DINO .451 (GM-B .442), DA3 pos .414 (GM-B .471), VGGT pos .884 (GM-B .774 better), rot 13.3 deg (GM 20.1, GM-B 17.3), rot>30 2.17/scene (GM-B 2.67); kitchen rot 33 -> 3.1 deg with 0 excursions, playground .677 -> .642 LPIPS, DINO .09 -> .28, rot 21.5 -> 5.0, 0 excursions. Mask drops 11% of cells (p90 26%), frontier 4.9%. GM-F + inside mask is the WORST arm (rot 30.6, kitchen 60.8 deg / 10 excursions, playground LPIPS .82): training with masked/stripped renders (23% true-depth cache + strips) and corruption lowered the model's render reliance, and the camera is executed through the render -> GM-F dropped. Conclusion: do not train the model to trust the render less; remove the misleading frontier tokens at inference. Next: SOTA cohort with gm_inside first, then gm_b and gm_b + inside mask (combination untested); GM-F removed from the SOTA queue.
- 9 Nov 21:38 GM-B + frustum mask (dev, 6 scenes): NOT additive. Late LPIPS .631 (GM-B .623, GM+mask .590), DINO .435, sharp 437, DA3 pos .654, VGGT pos .791, rot 17.9, excursions 3.17/scene. Playground e7523f6f collapses from call 9 to 16 (LPIPS .81-.87, rot 29 deg, 9 excursions); kitchen .654 / 13.6 deg; mattress store and greenhouse slightly better than GM-B (.502, .702 with 0 excursions). Reading: GM-B learned to lean on the temporal history under corruption; when the mask removes render tokens the history-only regime produces the grid collapse seen in GM-B's call 10. Winner stays GM (untrained) + frustum mask at inference. gm_b_inside removed from the SOTA queue (saves ~2.2 h); queue now gm_inside (running) -> gm_b, baseline then correction, scores, table.
- 9 Nov 22:40 mechanism diagnostic (diag_mechanism.py, CPU, eval_gm/diag_mechanism.txt) + user-forwarded critique. Findings: (1) GM+mask vs GM: camera POSITION gain is robust (DA3 -.21, VGGT -.31, 5/6 scenes, scene-bootstrap CI excludes 0); ROTATION gain comes from two scenes only (kitchen -31, playground -33 deg; others +-5); LPIPS -.011 (LOO all negative, CI touches 0); DINO +.048 (LOO all positive). (2) GM-B + mask vs GM-B: DA3 pos worse (+.31, CI > 0), VGGT pos / rot null, LPIPS +.008 -> "not additive" is true for DA3 position only; the playground collapse (calls 9-14, sharp < 60) occurs in GM-B WITHOUT the mask as well, so the mask did not cause it. (3) Sharpness confound rejected: corr(dsharp, dlpips) = -.01; GM+mask has the SAME sharpness as GM (.78 of the recording), GM-B arms are SHARPER than the recording (1.02-1.22 = grid/speckle artifacts, not detail); DINO (less blur-sensitive) also favours GM+mask. (4) DA3 vs VGGT position Spearman .77 over arms x scenes; gm_b_inside is the outlier (DA3 >> VGGT on c3cfc2ff: 1.14 vs .63), plausibly DA3 misled by over-sharp artifacts. (5) Mask-camera coupling: in the depth-test arms rho(cells dropped, rot) = .63-.68 but rho(drop_k, rot_{k+1}-rot_k) = -.25: heavy drops are a SYMPTOM of an already inconsistent state (misregistered sources), not the cause of the next error; in GM+frustum heavy drops are rare (3/114 calls) and coincide with 103 deg errors. (6) Two separate failure modes: (a) rotation excursions (GM kitchen 173/74/63 deg at calls 6-8 with frontier fraction only .05-.08 -> either a few frontier tokens suffice or the excursion is seed-chaotic: native500 kitchen s7 vs s107 gave rot 6.5 vs 23 deg); (b) blur collapse (sharpness < 15, playground calls 11-17) in EVERY arm, mask or not, while the frustum mask drops only 1-10% there: the render is "observed" (inside the frustum of generated sources) but built from already-blurred generated frames; sharpness returns at calls 18-20 when the camera re-enters anchor-era coverage. Also GM over-executes late (DA3 ratio 1.63/1.34 in calls 11-20, 31% of calls > 1.2). Decisive tests queued (queue_diag.py, SOTA paused): oracle memory (recorded frames + recorded poses as ReSplat sources, generator unchanged; oracle_memory.py) on kitchen + playground; seed replication s107/s207 for GM and GM+mask on both scenes; render/source sharpness now logged per call (frontier.py). SOTA queue relaunches automatically after.
- 9 Nov 23:30 SEAMS (user: "her inference'ta jump var, sahne sürekli değişiyor"). diag_seams.py (pixel MAD): boundary/within ratio GM 1.17, GM+mask 1.17, GM-B 1.33 (recording .97); no jump at latent boundaries (ratio ~.95) -> only the call boundary. diag_continuity.py (DINOv2 consecutive-frame similarity): boundary .746 vs within .899 (recording .963/.965) for GM; GM+mask .763/.902; GM-B .707/.899 -> drop .154/.140/.192 vs recording .002. Per scene drop: kitchen .20/.17/.28, classroom .30/.23/.33, greenhouse .21, shop .12, playground .08/.09/.16, mattress .02. 16-frame persistence .675 vs recording .762. Visual strips (eval_gm/seam_*.jpg): counter geometry, background and scene identity change at frames 64->65, 144->145, 224->225. Pixel seam ratio on the DL3DV cohort (stream_check_v2): ours 1.24 (29 boundaries), Lyra-2 1.10 (5), Spatia 1.16, MG-Distilled 1.07 (39), MG-Base 6.01 (hard cuts) -> DINO measure on all models running (diag_continuity_any.py). TELEPORT TEST (critique item 1): per-call boundary DINO drop vs the call's measured residual (roller.diag): Spearman +.67 (GM), +.57 (GM+mask), +.62 (GM-B); residual_deg +.61/.63/.67; drop .16 when residual < median vs .35 when >= median. Residual median .12-.14, p90 1.3-1.4. => the content jump at a call boundary is largely the camera: the history is written at the MEASURED pose while the next command is the PLANNED pose (--write 1 --replan 0), so the accumulated tracking error is executed in the first frame of the next call; the render at the planned pose shows the memory from there and the content snaps. Oracle objection accepted: the oracle run separates blur collapse, not the seam. Next GPU tests: F' (--replan 1 --replan_mode offset) for GM+mask on kitchen + playground (does the offset rejoin remove the seam?), and render sources restricted to the last two generated latents (generation mixing).
- 10 Nov 00:00 ORACLE MEMORY kitchen (recorded frames + poses as ReSplat sources, generator unchanged): late LPIPS .601 -> .515, DINO .39 -> .52, sharpness 128 -> 206, DA3 pos .35 -> .19; VGGT rot excursions remain (call 8: 120 deg with DA3 pos 1.01, both estimators; calls 18-19: 78/67 deg with DA3 .54/.26 -> estimators disagree). Frames (eval_gm/kitchen_excursion_frames.jpg): GM call 6 (VGGT 173 deg, DA3 pos .09) shows the SAME room (stripes wall, counter) with corrupted content, not a flipped view -> VGGT rotation on corrupted frames is partly a scorer artifact; the ">30 deg excursion" counts are not reliable on collapsed content. Render log: with recorded sources at sharpness 200-560 the ReSplat render's first target frame has sharpness 20-200 (3-5x blurrier) -> the render pipeline itself is a blur source; output sharpness tracks render sharpness. Critique accepted: teleport correlation is not causal proof (F' test queued), add open-loop (write 0) arm, u_release .75 arm (release_render.py), identity render test (identity_render.py), estimator check. queue_diag3.py running (then SOTA).
- 10 Nov 00:40 (critique round 3 applied). LOOP GAIN from the render logs (oracle runs, recorded sources): render/source sharpness .23 (kitchen) / .11 (playground); output/render 3.2 / 4.1; output/source .61 / .38 -> in a self-sourced loop sharpness decays geometrically per call (factor ~.4-.6) until the model's prior floor: explains the collapse speed order (playground fastest). Oracle ceiling caveat accepted: recorded sources carry unseen content, so -.26 is an upper bound, the reachable target is first-generation quality. NORMALISED seam (1 - boundary/within): GM .175, GM+mask .158, GM-B .219 (recording .005); oracle memory does not shrink it (kitchen .20 = GM) -> the seam is not memory content and (since DA3 pos halved under oracle without seam change) not simply teleport either; structural candidates: VAE dialect of the render clip, photometric offset, the model's own chunk-boundary behaviour. Queued (queue_diag4): dialect_prefix.py (render clip prefixed with the real previous frame instead of its own duplicated first frame) and oracle_chain.py (generation 2 and 3 from the oracle outputs at recorded poses: per-generation loop loss). CAMERA with estimator agreement (diag_camera_agree.py): DA3 and VGGT agree on only 52-59% of calls; on agreeing calls excursions fall to .3-1.7/scene, GM+mask keeps the position advantage (.33 vs .42-.53) but its rotation advantage disappears (11.6 vs GM 11.3, GM-B 8.5) -> camera ranking unresolved beyond position; appearance metrics carry the decision. Early render-off continuity (native b0, shop): render OFF seam .112 vs ON .040 -> the render reduces the seam in that arm; full table pending.
- 10 Nov 01:00 CONTINUITY ON THE DL3DV COHORT (diag_continuity_any.py, 30 videos): boundary DINO drop at our 16-frame boundaries .003 for ours open loop / + correction / F-prime, within .995; Lyra-2 .001, Spatia .000, MG-D .001 -> NO seam on the cohort for any model. Render-off vs on (native line, 6 dev scenes): render OFF seam .142-.162 vs ON .095 (within .825-.847 vs .872-.887, persistence .63-.66 vs .68-.69) -> the render REDUCES the seam and improves continuity; the seam is the model's own chunk-boundary behaviour, worse without memory. MOTION SPEED explains the cohort/dev gap: cohort plans move dt .010-.034 units and 0.7-2.6 deg per 16-frame call (near-static drift), dev scenes .19-.55 units and 10-34 deg per call (15-50x faster). The dev failures (seams, blur collapse, camera chaos) are a fast-motion regime that the SOTA cohort never enters; the cohort's "first-160 camera lead" is measured at tiny motions. Oracle continuity re-measured (separate file): kitchen drop .201, playground .094 (= GM) -> seam independent of memory content.
- 10 Nov 01:40 F' (offset rejoin, --replan 1) on GM+mask vs --replan 0: kitchen late LPIPS .575 vs .577 (=), DINO .444 vs .487 (worse), VGGT pos .35 vs .17, rot 10.0 vs 3.1 (worse), seam drop .128 vs .167 (-23%); playground late LPIPS .702 vs .642 (worse), residual p90 .62 vs .30 (the rejoin commands large catch-up moves after an error; calls 11-13 LPIPS .86/1.02/.85), seam .067 vs .088 (-24%). => teleport explains roughly a quarter of the boundary jump; F' as implemented (no per-call clamp, no rejection gate) hurts the rollout in the fast-motion regime. Gate 0a script written (gate0_latent_vs_rgb.py: pulled aligned latents vs pushed latents vs cached RGB-render latent vs true target latents on 40 recorded windows) and queued after the identity render (queue_diag5). SOTA still paused.
- 10 Nov 02:10 critique round 5 applied: queue reordered (queue_diag6): gate 0a first (recorded windows; + 'pull_src' = target depth z-buffered from SOURCE depths, the inference-realistic variant; motion-binned summary), gate 0a on GENERATED sources (oracle outputs at recorded poses, live ReSplat+VAE render latent), identity render, release, open loop, recent-only, dialect, chain g2/g3, first_render points (classroom/greenhouse first-call problem), damped F' (gain .5, clamp .1 units / 5 deg), seed 107 only (4 runs), then SOTA. DESIGN NOTE for the two-band plan: training-time target depth for the aligned-latent read must come from the SOURCES (z-buffer of source depths, as at inference), never from the recorded target frame (frontier-cache lesson: 23% vs 43%). Lyra-2 on the 6 dev scenes deferred: needs the driver's batch inputs prepared from dev streams (not in the queue yet).
- 10 Nov 02:20 OPEN LOOP (write 0, commanded labels) vs measured writes (write 1, replan 0), GM+mask: seam drop kitchen .167 -> .105, playground .088 -> .046 (largest seam reduction of any arm) while quality: kitchen LPIPS .589 vs .577, DINO .459 vs .487 (slightly worse); playground .781 vs .642, DINO .11 vs .28, sharp 24 vs 52 (much worse; camera DA3 .33 vs .14). => the measured-pose write is the teleport source (labels measured, command planned) and accounts for ~35-50% of the boundary jump, but it is what keeps the playground registered; the remaining ~50-65% is structural (persists with commanded labels and with oracle memory; render reduces it). F' reduces the seam less (-24%) and hurts via catch-up moves; the damped F' (gain .5, clamp .1/5 deg) is the queued compromise.
- 10 Nov 02:50 GATE 0a RESULT (gate0_latent_vs_rgb.py, 40 recorded train windows + 8+8 dev windows with GENERATED sources). Recorded sources, single-source first-call windows (n=29): RGB-render latent wins decisively (cos .676 vs pulled .513/.528, 1/29 reversals). Recorded sources, 3-4 source windows (n=11): pulled-with-source-depth latent beats the render in 8/11 (cos .647 vs .615). GENERATED sources (oracle outputs at recorded poses, the deployment case): kitchen pull_src cos .578 vs render .421 (mse .723 vs .880), playground .664 vs .540; the render's agreement with the true target latent decays with rollout depth (kitchen starts 21-29: .26-.37) while the pulled latent stays .55-.70. Coverage: render .93-.97, pull_src .62-.80. => Gate 0 passes where it matters: aligned clean latents are a better appearance carrier than the re-rendered RGB once the memory is built from generated frames and has several sources; the RGB render stays better for the single-source first call -> supports the two-band / first-call-RGB design. Open question for 0b: does the UNTRAINED GM read a pulled latent in the render slot (dialect of VAE-encoded render vs raw latent)? Gate 0b wrapper next. Paper-controller arm (commanded labels + bridge, paper_controller.py) inserted before damped F' (queue_diag7).
- 10 Nov 03:10 IDENTITY RENDER (identity_render.py: recorded frame rendered at its OWN pose): single source keeps .3-.9x of the source sharpness but with holes (valid .4-.85) and a photometric offset (kitchen dI -10..-20, i.e. the render is darker); TWO sources (frame, frame-4) fill the holes (valid .74-1.0) but sharpness falls to .11-.51x -> the dominant blur is the multi-source FUSION in DA3+ReSplat (depth inconsistency between views blends misaligned Gaussians), not the view change and not the single-view reconstruction. Implication: render per source without fusion (keep sources separate for the model) or render from the nearest source only; this also matches the latent_splat design note ("sources kept separate"). Gate 0b (two_band_inference.py: pulled clean latents in the render slots for u < .75, oracle and own sources, untrained GM) queued before damped F'.
- 10 Nov 03:50 kitchen, GM+mask variants: RENDER RELEASE (no render tokens for u < .75): late sharpness 134 -> 170 (the output IS pulled toward the render's blur) but LPIPS .577 -> .618, DINO .487 -> .382, VGGT pos .17 -> .78, rot 3.1 -> 19.9: the late-band render also carries content registration and camera; removing it costs more than the sharpness it returns. RECENT-ONLY sources (s-1, s-2, no anchor/bank): sharpness 183 but camera collapses (DA3 .74, VGGT pos 1.78, rot 65.6), LPIPS .665: the anchor/bank sources are what keep the rollout registered. => the early/late split must keep geometry+registration in the late band too (the pulled latent must carry it, or the late band keeps a geometry-only channel); Gate 0b (pulled clean latents with coverage metadata in the late band) is the direct test; nearest-single-source A/B next.
- 10 Nov 04:05 seams of the variants (kitchen): recent-only sources .249, render release .223 vs GM+mask .167 -> removing old sources or the late-band render INCREASES the boundary jump; the generation-mixing hypothesis (anchor/bank blended with drifted recent sources) is not supported for the seam. Critique applied: gate 0b now keeps the RGB render latent in uncovered cells (--fallback 1) and its reference is the release arm (same u < .75), not GM; the nearest-single-source arms were dropped (they would repeat the recent-only collapse) and replaced by the all-sources point render (pointrender_all.py: same source set, no Gaussian fusion). Queue v10: gate 0b x4 -> point render x2 -> identity v2 -> gate0 multi -> paper ctrl -> dialect -> damped F' -> recent playground -> chain -> points -> seeds -> SOTA. Training (two-band, from GM) starts after gate 0b + the render A/B fix the contract.
- 10 Nov 04:40 GATE 0b (two-band inference on the UNTRAINED GM+mask: u >= .75 RGB render latent, u < .75 pulled clean latents in covered cells, RGB fallback elsewhere; sources anchor + s-1 + s-2 + bank; coverage .55-1.0 per call). Reference = release arm (no render below .75). Kitchen, OWN sources: LPIPS .603 (release .618, RGB .577), DINO .418 (.382 / .487), camera DA3 .57 (release .45): partial recovery of appearance, none of camera. Kitchen, ORACLE sources: LPIPS .496 vs oracle-RGB .515, DINO .616 vs .524, sharp 227 vs 206, DA3 .159 vs .194, VGGT pos .377 vs .555, rot 12 vs 19 -> with CLEAN latents the untrained model reads the pulled latent and beats the RGB render on every metric (dialect is not a blocker). Playground, OWN: collapsed like release (DINO .06, sharp 24): own latents from collapsed frames carry the collapse. Playground, ORACLE: sharp 329 vs 205 (much sharper), LPIPS .442 vs .421 and DINO .446 vs .544 (slightly worse). => Gate 0b: the model can read aligned clean latents untrained; the value is bounded by SOURCE quality, so the write policy (keep clean/first-generation sources, do not overwrite them with later generations) is part of the mechanism, not an add-on. Point-render arm failed on tensor shapes (CHW), fixed, queue relaunched.
- 10 Nov 05:15 TWO-BAND STAGE B queued (queue_train_tb.py): build_latent_cache.py (pulled clean latents per render-cache window; target depth z-buffered from SOURCE depths, fusion s-1, s-2, anchor, bank; coverage) -> train_two_band.py (GM recipe verbatim from run_op/meas, 250 updates, seed 20261108; recorded render rows with chunk-0 u < .75 get the pulled latents in covered render cells + metadata 1, uncovered keep the RGB latent; on-policy rows unchanged; no corruption) -> dev eval two-band inference (own sources, mask, fallback) + RGB-only control of the same checkpoint -> compare_tb.json. Gate 1 (critique): calls 1-3 LPIPS within .02 of GM, exec ratio <= 1.2, and the two-band eval must beat the RGB-only control of the same weights (isolates the latent read from the extra updates). Point-render A/B, identity v2 and gate0-multi run first; the remaining diagnostics and SOTA after the training evals. Gate 0b seams unchanged (.18-.19 / .09-.10): the latent band does not touch the boundary jump.
- 10 Nov 05:40 POINT RENDER (all sources, z-buffer, no Gaussian fusion) on the untrained GM+mask: render sharpness 1000-4000 (point speckle/aliasing; the eval's own 'sharp' metric is inflated by it: 498 / 808), validity only .3-.9 (real holes), and the rollout collapses: kitchen LPIPS .669 (ref .577), rot 27.7 (3.1); playground LPIPS .756 (.642), DA3 .98 (.14). The model trained on dense ReSplat renders cannot use a sparse point render untrained (distribution shift, as warned) -> the fusion question cannot be settled at inference on this model; the early band stays ReSplat for stage B. Decision for the cache: ReSplat RGB render (existing cache) + pulled latents (new cache). Queue now: identity v2 -> gate0 multi -> latent cache -> training.
- 10 Nov 06:20 latent cache v1 built (3553 windows, 22 min, coverage .72) and two-band training started (loss .338/.293/.249, bands mixing), then ABORTED after gate0-multi (24 multi-source recorded windows): overall tie (render cos .605, pull .604, pull_src .599; common support render .622 vs .598) but PER ORIGIN the anchor-origin pulled cells are bad (cos .357 vs render .599 on the same cells) while s-2 (.657 vs .569), s-1 (.617 vs .606) and bank (.613 vs .576) origins beat the render -> the critique's anchor-dialect hypothesis holds (the first causal VAE latent encodes one frame, the others 4-frame groups). Fix: the anchor is excluded from the latent band (order s-1, s-2, bank; anchor cells fall back to the RGB render) in build_latent_cache.py and two_band_inference.py; cache rebuilt (v1 kept as latent_cache_v1_with_anchor), training restarted from scratch (aborted run kept as runs/tb_s1_v1_with_anchor_aborted). Motion bins: pulled latents gain relative to the render as motion grows (bin 2: pull .490 vs render .467).
- 10 Nov 08:15 OVERNIGHT RESULTS. (1) TWO-BAND STAGE B FAILED Gate 1: runs/tb_s1 (GM recipe + late-band pulled latents, 250 upd, latent share .20, loss .286 -> .280) on 6 dev scenes with the frustum mask: two-band inference late LPIPS .652 / DINO .384, RGB-only inference of the SAME weights .646 / .410, GM+mask .590 / .451; calls 1-3 unchanged (.372-.374 vs .373); exec frac>1.2 .31-.33 vs .24; playground tb_rgb sharp 451 with DINO .06 = grid artifact (as GM-B). => the 250 extra updates themselves hurt (as GM-B, GM-F: every 250-update continuation of GM so far raised late LPIPS by .02-.06), and the latent band does not beat the RGB control. (2) SEED NOISE: kitchen GM+mask s7 .577 / rot 3.1 vs s107 .629 / rot 47; GM s7 .601 vs s107 .623; playground GM s107 .632 beats GM+mask s107 .654. Two-seed means kitchen GM .612 vs GM+mask .603, playground .655 vs .648 -> the GM+mask advantage on these two scenes is within the seed floor; the kitchen "33 -> 3 deg" was seed luck. (3) ORACLE CHAIN (recorded poses, sources = previous generation): kitchen g1 .515 -> g2 .573 -> g3 .603 (= GM .601); playground .421 -> .584 -> .661 -> most of the loss happens at the FIRST re-sourcing; a write-once/first-generation policy would cap near g2, far from g1. (4) PAPER CONTROLLER (commanded labels + bridge): smallest seam (kitchen .120, playground .035) but worst quality (kitchen .662, playground .838 LPIPS) -> the measured-write benefit comes from memory PLACEMENT (supports the measured-pose memory claim); damped F' (gain .5, clamp .1/5 deg): kitchen .598 seam .185, playground .665 seam .110 -> no better than replan 0. (5) DIALECT PREFIX: kitchen .596 (seam .233), playground .671 / DINO .316 / camera best of all arms (.156 / .368 / 4.9 deg) -> mixed, one seed. (6) first_render points: no gain. (7) DARKENING CONFIRMED: identity render luminance error by alpha bin: alpha .5-.7 -18..-42, .7-.9 -5..-14, .9-1 ~+1 (all 3 scenes) -> ReSplat colour is not normalised by accumulated alpha; fix = divide by alpha (needs a re-cached training render to be used). SOTA queue (gm_inside -> gm_b, open loop then correction) relaunched at 07:50, running.
- 10 Nov 08:15 SOTA cohort, GM + frustum mask, open loop (5 DL3DV scenes, 480 frames, same scorer): pos .0539 / endpoint .164 / rot 2.65 / LPIPS .381 / DINO .853 / last-third LPIPS .562 / DINO .735 vs Ours (GM, no mask) .0505 / .154 / 2.54 / .378 / .855 / .552 / .743: neutral to slightly worse, 1/5 scenes better on LPIPS. The mask removes 5-11% of cells here (true frontier .5-2.8%): the cohort's near-static motion has almost no frontier, so the mask has nothing to fix. Not a SOTA improvement. GM-B and the correction arms follow.
- 10 Nov 08:40 critique "why does more training not help" (round 7) checked: no EMA, no LR warm-up, each continuation starts a fresh AdamW from the receipt (train.py: --init_receipt loads weights only) -> confirmed. Weight distances: |GM-B - GM| 20.1, |tb - GM| 20.3, |tb - GM-B| 20.9 -> two 250-update continuations moved in ~orthogonal directions (random-walk signature). LOOP GAIN from render logs (output sharpness / source sharpness per call): GM s107 kitchen .79-.84, playground .66-.69 (late median .05-.07 = collapsed); tb (two-band weights) 1.25-1.26 on 6 scenes (late 1.20-1.23) -> GM sits below 1 (blur collapse), the continuation above 1 (sharpening runaway / grid artifacts): the two failure modes are the two sides of an unstable loop, as predicted. Queued: interpolation checkpoints (interp_ckpt.py: GM/GM-B .5, GM/tb .5, soup GM-B/tb .5) and a HELD-OUT evaluation on dev scenes 10-16 (6 scenes never used for selection) for GM, GM-B, tb and the three interpolations (queue_heldout.py), then the two interpolations on the seen scenes; the SOTA queue pauses before its correction arm (sota_eval/PAUSE) and resumes after.
- 10 Nov 09:10 ROW AUDIT (audit_rows.py, 400 rows of the exact GM sampler, CPU): NO violations (shapes, roles, chunk index, render positions = chunk-0 positions, bank causal <= start-5, recent window, target/context/anchor latents identical to the stream, cameras = recorded poses in model units, on-policy context generated and labelled with measured poses, metadata only on render slots and in [0,1], finite tensors, text nonzero). DISTRIBUTION: plain 25% / recorded render 36% / on-policy 39%; on-policy rows exist only at start 5 and 9 (calls 2-3; pool 600 items made by the stage-B model, reused by every continuation); first-call 17.5%; start median 9, p90 59; bank empty in 68% of rows, bank 4 in 1.75%; call motion median .35 units (dev .19-.55: matched; the SOTA cohort .01-.03 is the outlier); render latent std .87 vs target .98, cos(render, target) .61 on valid cells; u (shift3): <.3 12%, .3-.75 38%, >.75 50%. Per update: 8 rows x 2 chunks = 16 chunks (64 target latents); Memorizon: 32 x 10 = 320 chunks per step at the same lr 2e-5, 10k steps (3.2M chunks vs our 4k per stage). Weight displacement per 250-update stage |d theta| ~ 20 over 1.27e9 trainables = 5.6e-4 rms per weight ~ 1.75 x lr*sqrt(T) (random-walk value 3.2e-4), far below lr*T = 5e-3 -> updates are noise-dominated. Queued before the held-out arms: grad_snr.py (cosine between independent accumulated gradients at A=8 -> per-row SNR -> accumulation needed; n_chunks 3/4 VRAM + time profile) and dfhist_inference.py (recent slots noised to u_h and LABELLED with u_h vs unlabelled, training-free on GM, kitchen + playground).
- 10 Nov 09:15 GRADIENT SNR at GM (grad_snr.py, 4 independent batches of 8 rows): cosine between batch gradients backbone .092, new modules .203, all .158 -> per-row SNR .013 / .032 / .024 -> rows per update for cos .5: 78 (backbone), 31 (new modules); predicted cos at accum 16 / 32 / 64 / 128: backbone .17 / .29 / .45 / .62. At accum 8 the backbone update is ~91% noise. Grad norm .11-.17 (clip 1.0 never active). PROFILE: n_chunks 3 = 4.8 s, 20.0 GB; n_chunks 4 = 6.2 s, 20.7 GB (n=2: 3.5 s) -> longer rows fit and cost LESS per chunk (1.55 vs 1.75 s); n = 6/8/10 profile queued. SOTA COHORT, GM-B open loop: pos .0521, endpoint .116, rot 2.44, LPIPS .377, DINO .896, last-third LPIPS .516, DINO .841 vs Ours (GM) .0505 / .154 / 2.54 / .378 / .855 / .552 / .743 -> GM-B better than GM on DINO, last-third LPIPS and last-third DINO in 5/5 scenes (LPIPS 3/5, pos 2/5); last-third DINO .841 passes Spatia (.837), Lyra-2 .880. On the slow cohort GM-B IS an improvement over GM (opposite to its late-LPIPS loss on the fast dev scenes).
- 10 Nov 10:00 USER: "GM-B solved the collapse on the SOTA cohort but it darkens; the mask must be in the training render records and GM-B must be in the new training". (1) DARKENING MEASURED (diag_luminance.py, frames every 16th, ratio to ground truth, thirds of the rollout, 5 scenes): GM-B luminance 1.01 / .965 / .90 (5/5 scenes fall, worst .79), contrast 1.035 / 1.088 / 1.108; GM (same renders) 1.06 / 1.085 / 1.06, saturation 1.02 / 1.11 / 1.19; Lyra-2 1.03 / 1.06 / 1.11; Spatia .99 / .96 / .95. -> the darkening is a property of the GM-B WEIGHTS, not of the render (GM brightens with the same renders); the alpha .5-.7 fringe of the render is dark (identity render ratio .80-.83) but dividing by alpha would over-correct (.6 -> 1.34). Mechanism candidate: the corruption is unlabelled (context t=0) and attenuating: (1-s) x + s eps scales the signal by (1-s), and the one-step x0 of the recent latents is low-contrast -> the model learns to amplify its context -> contrast gain > 1 per call. (2) FIXES BUILT: corruption v2 (gm_b_corruption.py amp_preserve: x + s/(1-s) eps, same SNR, signal coefficient 1; match_moments: self-augmented recent latents rescaled to the clean per-channel mean/std; x0_std_ratio logged per row); mask v2 (frontier.py alpha_min: valid iff alpha >= .7 AND rendered depth inside a source frustum; build_mask_cache.py re-renders all 3553 cached windows with the inference code path and stores only the metadata, render latents unchanged; train_gmp.py reads it); gen_pool_v2.py (deployment protocol: measured writes + mask-v2 render, depths 1-4 instead of 1-2, pool in this folder); LR warm-up 10 steps. Tests: test_gmp.py 11/11, test_gm_b 8/8, test_frontier 6/6. (3) ACCUMULATION CORRECTED: with Adam the per-step displacement is ~lr whatever the batch, so for a fixed number of rows accumulation A changes the signal displacement by 1/sqrt(A) and the noise displacement by 1/sqrt(A) too: the signal/noise ratio of the final weights depends only on the number of ROWS (sqrt(R) rho), not on A. "accum 64 x 40 updates" (said at 09:15) would have trained 2.8x less (the zero-initialised camera / metadata / role modules 6x less) for +14% ratio -> withdrawn. Setting: accum 16 x 250 updates (2x the rows of every earlier stage, ratio x1.41, 3.9 h per stage), lr 2e-5 / 1e-4 unchanged, 10-step warm-up. (4) LESSON 2 (noisy earlier chunks in Memorizon's memory pool): the labelled-history probe cannot run on this model (packed_model raises on t != 0 for context slots; logs/diag_dfhist_u10_label_kitchen.log) and labelling at inference is not what Memorizon does either (context labelled clean). Decision: the pool of noisy memory = corruption v2 on bank / recent / render (unlabelled, therefore amplitude-preserving) + deeper deployment-protocol pool rows; n_chunks stays 2 (profile: n=6 9.8 s / 22.4 GB, n=8 15.1 s / 24.8 GB, n=10 19.7 s / 26.7 GB fit, but only +13% chunks per second and per-chunk renders (R2) showed no win). (5) QUEUES: queue_heldout2 died at the probe (GPU idle 09:02-09:35); held-out tb / interpolation arms dropped; sota_eval/queue_sota_gmb_corr.py runs the GM-B "+ correction" arm (5 scenes) -> table_sota.md; queue_gmp.py: mask smoke -> mask cache -> B' (plain-350, recorded rows, mask v2 + corruption v2) -> gate vs old B (3 dev scenes, seed 107, stop if late LPIPS > +.05) -> pool v2 (B', 150 streams) -> GM' -> SOTA cohort open loop + dev 6 scenes. Expected end ~04:00 11 Nov.
- 10 Nov 10:50 SOTA COHORT, GM-B + correction (rejoin bridge, 5 scenes): pos .0522* / end .172* / rot 1.96 / LPIPS .360 / DINO .880 / last-third LPIPS .516 / DINO .773 vs GM-B open loop .0517 / .116 / 2.45 / .377 / .896 / .516 / .841; Ours + corr .0336 / .093 / .85 / .356 / .853 / .521 / .733. (*61b27b61: ViPE camera failed on the GM-B + corr video; pos/end over 4 scenes.) Correction helps GM-B on 02267acf / 0238de14 / 334ff324 (camera and LPIPS), hurts a126efe8 (end .18 -> .47) and 61b27b61 (DINO .78 -> .69) -> mixed; last-third DINO falls .841 -> .773. build_table_sota.py crashed on the None camera (comparison_table metrics() casts None); numbers above from a None-tolerant recomputation. MASK SMOKE v2: re-render vs cached metadata flips .19% of cells (p99 .57%), mean diff .003 -> ratio mask accepted; mask cache building (3553 windows).
- 10 Nov 11:00 WHY CORRECTION DOES NOT HELP GM-B (run.json boundaries + ViPE trajectories of the correction videos, 5 scenes): (1) DA3 per-call residuals of GM-B are 2-4x those of GM (median .022-.046 vs .011-.028 units) while the plan moves only .010-.036 units per call on this cohort -> the correction term is as large as or larger than the motion (commands up to 1.7-4.8x the plan step). (2) DA3 is not ViPE: after scale fitting the DA3 and ViPE error directions agree at cos .1-.65 (second half .08-.46); in 02267acf GM-B's ViPE error (.012) is below GM's (.036) although its DA3 residual is twice as large -> much of the residual is measurement noise on GM-B frames; gain-1 correction of noise = random walk. (3) a126efe8: residual grows monotonically .004 -> .20 despite correction every call; DA3 error .022 vs ViPE .058 -> the loop drifts. (4) 61b27b61: one DA3 jump at call 21 (.53 units, 23 deg), no outlier gate -> commands 12-35x the plan step -> runaway (3.9 units / 114 deg), ViPE fails. The correction arm has no gate / clamp / deadband. Fix candidates: reject residual > max(3x plan step, 5 deg); clamp correction <= 1x plan step; deadband at DA3 noise. GATE CHANGE (user): B' vs old B on the 5 SOTA-eval scenes only (open loop, 30 calls, seed 7, mask v2 for B'); dev evaluation removed from queue_gmp. Boundary-jump check on the SOTA videos running (diag_seams_sota.py).
- 10 Nov 12:05 BOUNDARY JUMPS + OWN-HISTORY IDENTITY ON THE SOTA COHORT (diag_seams_sota.py, 5 scenes x 8 videos, transition 16k -> 16k+1 vs all others; eval_gm/diag/seams_sota.json). Boundary / within ratio (DINO 1-cos, pixel MAD): GT .98 / 1.01, Ours .90 / .95, GM + mask .88 / .96, GM-B .58 / .81, GM-B + corr .65 / .88, Lyra-2 .86 / .96, Spatia 1.18 / 1.16 (Spatia is the only model with visible 16-frame jumps). Absolute: boundary change of GM-B = GT level (DINO .0035 vs .0027, MAD 2.98 vs 2.63, |dlum| .126 vs .125); WITHIN-call change of GM-B is 2x GT (DINO .0055 vs .0028, MAD 3.49 vs 2.58, |dlum| .31 vs .13) -> on this cohort GM-B changes the scene gradually inside the calls, not by jumps at the boundaries (unlike the fast dev scenes, seam 1.33). GM-B + corr: all changes 2-3x larger; a126efe8 boundary MAD doubles in the second half (3.8 -> 7.7) = the oversized correction commands. OWN-HISTORY IDENTITY (DINO cos frame t vs t+16 / t+48, second half): GT .972 / .953, Lyra-2 .969 / .947, Spatia .964 / .944, Ours .957 / .924, GM-B .951 / .922, GM + mask .949 / .914, GM-B + corr .910 / .861; a126efe8 p48 late GM-B .936 > GM .919 -> GM-B is as self-consistent as GM: the ceiling re-synthesis is slow drift AWAY FROM THE RECORDING, consistent with its own past, so an own-history gate cannot catch it (it is in queue_gmp's B' gate as a guard against flicker, not against identity drift). A recording-anchored observed-region metric is needed for that.
- 10 Nov 12:40 CORRECTION LOOP OF GM-B (critique: execution ratio + residual sign): per call, DA3 execution ratio (measured motion projected on the command from the measured start) GM-B + corr median .42 / .51 / .02 / .57 / .41 (5 scenes) vs Ours + corr .93 / .80 / .51 / .65 / .54; consecutive residuals point the SAME way (cos .93-.99, sign flips 0-4%) for both -> no oscillation: a steady lag, GM-B executes ~40% of each command by DA3. ViPE path-length ratio of GM-B open loop is 1.12-2.40 (GM .87-1.32) -> ViPE sees MORE path (inflated by the 2x within-call change), so the execution magnitude is estimator-dependent; the controller (tuned to GM's response) is not calibrated for GM-B -> GM-B + corr row withheld from tables until a gain-aware rule is calibrated. RECORDING-ANCHORED IDENTITY built (diag_anchor_identity.py): masks from the recording only (DA3 depth of recorded frame t + recorded camera, reprojected into the anchor, depth agreement 15%), every 8th frame, frames with coverage < 20% unscored; masked LPIPS + low-frequency colour / signed luminance vs the recording, curves + half means + second-half slope. queue_gmp: anchor masks before the interpolation; decision on render corruption = GM-Br keeps GM-B's last-third DINO (.02) AND anchored LPIPS (h2) not worse by > .01; B' gate adds anchored LPIPS (+.05) and within-call change ratio to the recording (+.5).
- 10 Nov 13:00 WITHIN-CALL PROFILE (diag_within_call.py; critique candidates 2-4): change per in-call position relative to the recording (5 scenes): GM-B DINO 4.08 / 2.26 / 1.68 / 2.13 / 1.49 ... 1.27 (positions 1..15), low-frequency 2.99 / 2.01 / 1.60 / 1.38 / 1.29 ... 1.16, high-frequency 1.45 -> 1.24; GM DINO 1.74 / 1.97 / 1.45 / 1.98 ... (2-frame alternation, no start peak), low 1.26 / 1.41 / 1.16 ...; Lyra-2 1.0-1.56 with a 4-frame period (latent boundaries). => candidate 2 (drift growing toward the END of the chunk) is rejected: GM-B's excess change sits at the START of each call, mostly the FIRST in-call transition (frame 16k+1 -> 16k+2) and is low-frequency (brightness / layout), i.e. a seam displaced by one frame - the boundary definition 16k -> 16k+1 used at 12:05 (and on the dev set) missed it; the 12:05 statement "no boundary jumps on the SOTA cohort" is corrected: GM-B snaps at every call start (4x the recording), GM barely (1.7x, = its average). Overall excess GM-B low x1.50 / high x1.23 (GM x1.21 / x1.12, Lyra-2 x1.29 / x1.05, Spatia x1.66 / x.85). ViPE step-length variability within calls: GT .04-.09, GM .16, GM-B .21 (camera jitter contributes). Render flicker: pending (gm_b_rdump run). New metric snap1 (first in-call transition / other in-call transitions) added to diag_seams_sota and the queue reports.
- 10 Nov 13:40 RECORDING-ANCHORED IDENTITY (diag_anchor_identity.py; anchor-visible regions from the recording's DA3 depth + cameras, coverage first/last sampled frame .89-.99 / .30-.76, every 8th frame): masked LPIPS first / second half, slope per 100 frames (h2), low-freq colour |diff|, signed luminance gen - rec (h1 / h2): GM .227 / .471, +.067, 16.8 / 35.5, +9.7 / +15.4; GM + mask .228 / .482, +.079, +9.6 / +17.0; GM-B .247 / .446, +.055, 15.8 / 32.8, -0.0 / -9.3; GM-B + corr .214 / .422; Ours + corr .206 / .442; Lyra-2 .219 / .435, +.063, +4.7 / +13.7; Spatia .244 / .443, +.086, -1.9 / -2.3. Conference hall a126efe8: GM .190 / .471 (slope +.044, dlum +5.8 / +2.6), GM-B .190 / .426 (slope +.081, dlum -3.6 / -20.2), Lyra-2 .265 / .464, Spatia .234 / .396. => on the anchor-visible regions GM-B is NOT worse than GM by masked LPIPS (better in the second half on average), its failure there is luminance (-9 mean, -20 in the hall, GM is +15 too bright) and a steeper late slope in the hall; every model reaches LPIPS .4-.5 in observed regions by the second half (camera offsets are included in this number).
- 10 Nov 12:10 (clock note: the entries above stamped 12:05-13:40 were written between 11:10 and 11:50.) CONFIDENCE = 1 TEST (GM-B, conference hall a126efe8, conf := valid, no training; runs/gm_b_conf1): snap1 3.84 vs GM-B 3.32, anchored luminance h2 -19.3 vs -20.2, anchored LPIPS .199 / .447 vs .190 / .426, within-call DINO .0052 vs .0053, persist48 late .939 vs .936; pos .068 vs .070, endpoint .128 vs .182 -> the confidence channel is NOT the cause of the call-start snap or the darkening (candidate 1 rejected). snap1 per model (5 scenes, first in-call transition / other in-call, from call 2 on): GT 1.04, GM 1.04, GM + mask .94, Lyra-2 1.00, Spatia .89, GM-B 2.81 (4.33 / 1.58 / 2.21 / 2.62 / 3.32), GM-B + corr 2.41 -> the snap is specific to the GM-B weights. Queue: GM-Br (near-latent corruption only) then GM-Bv2 (amplitude-preserving noise + moment-matched self-augmentation, render half per the GM-Br decision), both from GM with GM-B's settings, 5 SOTA scenes each; decision_corruption.json picks the recipe of B'/GM' (v2 iff last-third DINO within .02 of GM-B and snap1 lower by > .3 or |luminance drift| lower by > 2).
- 10 Nov 12:35 SNAP MECHANISM (diag_snap_mechanism.py on the GM-B rerun with render video + latents dumped, conference hall): (a) RUN-TO-RUN NOISE: same seed, same weights, frame MAD vs the original run .69 at frame 16, 10.4 at 160, 34.6 at 320, 40.5 at 480 -> runs are not reproducible beyond the first calls (ReSplat nondeterminism amplified by the loop); metric spread between the two GM-B runs: snap1 3.32 / 3.62, anchored LPIPS h2 .426 / .413, luminance -20.2 / -19.3, pos .070 / .053, endpoint .182 / .163 -> conf = 1 (3.84, .447, -19.3, .068, .128) is inside that spread for snap and luminance: confirmed no fix. (b) LATENT SEAM: |diff| of consecutive latents, boundary pair (last latent of a call -> first of the next) / in-call pairs = 1.25 (full), 1.29 (low-pass); first in-call latent pair 1.04 -> the step IS in latent space at the call boundary; the causal VAE decoder shows it one frame late (pixel boundary ratio .6-.7, first in-call transition 3.3-3.8). (c) DIRECTION: over valid cells, corr of the snap (frame 2 - frame 1 of the call, low-pass luminance) with (render - frame 1) = -.02, with (1 - conf) = .00 -> the snap does NOT move toward the render and is not tied to low-alpha regions ("settling onto the render" rejected); mean snap luminance -.34 per call (x29 calls ~ -10 = about half of the -20 drift); the render of own frames is 7.3 levels DARKER than the last output frame (valid cells). (d) RENDER FLICKER: in-call low-pass change of the render 1.28 vs output 2.00 -> the render is steadier than the output (candidate 3 rejected). Remaining explanation: GM-B does not continue its recent latents smoothly (the near-latent corruption taught it that the recent history is unreliable) -> GM-Br (near-latent corruption only) is the direct test; queue order unchanged.
- 10 Nov 13:15 USER: tests from now on ONLY on 02267acf (showroom) and 0238de14 (playground), fast results; "the memory seems to re-generate generated places every time instead of storing them" -> investigate. queue_gmp: qs.SCENES = these two for every arm (interp, GM-Br, GM-Bv2, gate, GM'). (1) CODE: the render memory is rebuilt every call from <= 4 source frames (OwnRender864.render_own): anchor, latents s-1 and s-2 (both from the immediately preceding call), best bank view; peak VRAM 26.6 of 31.35 GiB with 4 sources -> no room for more. Nothing generated is stored: a generated region is rendered from its most recent re-generation (photocopy chain); only anchor-visible content has a fixed source. (2) MEASURED (diag_region_persist.py: spatial LPIPS between frame t and t+16 / t+48 of the same video, ratio to the recording, A = anchor-visible cells, N = generated cells; first / second half): showroom GM A 1.06 / 1.06, N .83 / .83 (lag 48: 1.04, .79-.85) -> GM's generated regions change LESS than the recording (static smear / extrusion); GM-B A 1.37 / 1.46, N 1.49 / 1.85 (lag 48: A 1.10 / 1.28, N .98 / 1.33) -> GM-B re-draws, most in the generated regions of the second half; Lyra-2 A 1.05 / 1.07, N 1.10 / .92; Spatia 1.03 / 1.17, .97 / 1.19. Playground: GM 1.18 / 1.16, N 1.18 / 1.03; GM-B 1.29 / 1.16, N 1.29 / 1.09; Lyra-2 1.09 / 1.10, N .99 / .78 -> no region difference there. => the user's observation holds for GM-B on the showroom (generated regions x1.5-1.85 at lag 16) and is short-term (lag-48 ratios are lower); GM has the opposite defect. (3) TEST QUEUED (no training): source_policy.py 'firstseen' = write-once render sources: anchor + two keyframes (every 5th call end; each target frustum sample belongs to the oldest keyframe showing it; the two largest owners >= 3%) + s-1, on GM-B, two scenes -> snap1 / region persistence / anchored identity / scores vs GM-B.
- 10 Nov 13:45 FAST LOOP (pasted plan + user's two scenes): queue_gmp stopped (B'/pool/GM' lineage postponed); queue_fast.py: every arm = 2 scenes (02267acf showroom, 0238de14 playground) x 20 calls (321 frames), fast_metrics.py (snap1, last-third DINO, anchored luminance slope, sharpness ratio, p48, re-generation of generated cells inside / across calls; CPU, no ViPE; selection never by LPIPS). GM-Br (near-latent only, v1) was already training as an orphan of an earlier queue: its update-100 weights were extracted from the resume slot (runs/gm_br_s1/ckpt/trainables_00100), stopped at the update-150 slot (resumable). REFERENCE ROWS (321 frames, showroom / playground): snap1 GM 1.03 / 1.15, GM-B 4.48 / 1.78, Lyra-2 1.16 / 1.01, Spatia 1.25 / .89; last-third DINO GM .925 / .797, GM-B .902 / .819, Lyra-2 .906 / .907, Spatia .908 / .861; luminance slope per 100 frames GM +3.9 / +.9, GM-B +2.3 / -10.8 (end -22), Lyra-2 +4.1 / +.2, Spatia +6.9 / -4.5; sharpness ratio GM .96 / 1.33, GM-B 1.08 / 1.52, Lyra-2 .92 / .68, Spatia .44 / .24; p48 GM .944 / .896, GM-B .918 / .917. WITHIN vs ACROSS CALLS (change of generated cells over 8 frames / recording): GM-B inside a call 1.04 / 1.09, across a call boundary 2.73 / 1.52 (anchor-visible cells .95 / 1.13 inside, 1.96 / 1.40 across); GM .80 / 1.07 inside, .90 / 1.16 across; Lyra-2 1.16 / .95, 1.15 / .97 -> GM-B's re-generation happens ONLY across call boundaries (inside a call it is at the recording's level) => memory + call-start snap, not in-call model behaviour -> inference candidates first. PHASE I (GM-B, no training): rerun (noise), latent moment continuity (source_policy.install_moment), source swap s-2 -> oldest frame covering generated cells ('oldest'), bank by first-appearance bands ('unseen'; the stock select_bank discounts what the recent latents cover, so the bank is empty under forward motion), swap + bank, write-once keyframes ('firstseen'). PHASE II (100 updates from GM, GM-B settings): near-only @100 (free), v1 full @100 (calibration), v2, render-only (gm_b_corruption near=False).
- 10 Nov 14:00 critique applied to the fast loop: (1) anchor-visible cells also jump across calls (1.96 on the showroom) although their source is fixed -> most of the boundary change is the snap, the memory's share is the excess of generated cells (2.73 vs 1.96); (2) two Phase-I candidates added: gmb_ctxnoise (inference context corrupted as in GM-B's training: bank / recent / render (1-s) x + s eps at s = .2, render metadata x .8; source_policy.install_ctx_noise) and gmb_alphanorm (render colour / alpha^.45 on valid pixels, frontier alpha_gamma; plain division by alpha over-corrects: identity render ratio .80 at alpha .6); order rerun, moment, ctxnoise, alphanorm, oldest, bank, oldest + bank, firstseen; (3) Phase II runs the playground for 30 calls (481 frames) and fast_metrics reports it at N = 481 too (GM-B's collapse advantage is in frames 321-480: 320-frame last-third DINO GM .925 / .797 vs GM-B .902 / .819 is inside run noise); (4) guard: sharpness ratio > 1.6 flagged as grid suspect (GM-B 1.52 on the playground, GM 1.33). Bug fixed in run_arm_ckpt.py: an earlier replace had put sources / bank / moment_patch keywords into frontier.install (would have raised with --conf_one / --render_dump; no run was affected), run.json now records them.

### 10 Nov 15:50 - Phase I result (GM-B, inference only, 02267acf / 0238de14, 321 frames)
Noise reference (GM-B vs rerun): snap1 4.48/1.78 vs 3.85/1.76, last-third DINO .902/.819 vs .904/.831, lum slope +2.3/-10.8 vs +2.5/-10.8, regen_x 2.73/1.52 vs 2.47/1.54.
- moment: snap unchanged (3.89/2.12); playground darkening -10.8 -> -2.7 /100 fr (end -22 -> -3), last-third DINO .866, sharp 1.17.
- ctx noise .2: showroom snap 2.32 (only candidate outside noise), playground 1.89 (none); lum -4.4; DINO .916/.837; regen_x 1.62/1.55.
- alpha-norm: partial luminance (-6.8), snap none. oldest: playground breaks (snap 7.0, DINO .732). bank / oldest+bank / firstseen: within noise or worse.
=> source/bank selection at inference does not reduce regeneration; snap is a train/inference context mismatch in part (ctx noise halves it on
the showroom); darkening is removable by the moment patch. Follow-up queue_fast_b.py: ctx .2 + moment (playground 481), ctx .35.
481-frame playground reference: GM DINO_last .686 sharp 1.86 (grid), GM-B .816 sharp 2.32 (grid flag!) lum_end -24, Lyra-2 .894, Spatia .817.
Dev seams with snap1 (snap_recompute.json): GM 1.54, GM-B 1.65, GM+fmask 1.32, recording 1.07-1.09 (dev GM also snaps; SOTA GM 1.04). ViPE step by
in-call position: GM-B 1.53/1.38 at positions 1-2 (camera also jumps), GM flat (.89-1.16).

### 10 Nov 17:15 - visuals, yaw, Phase II so far
- Visuals: eval_gm/fast/visuals/<scene>_sheet.jpg + _grid.mp4 (fast_visuals.py). Critique: columns must show the snap (call end / first / second
  frame: 80/81/82, 160/161/162, 240/241/242) -> fixed; rows now GT, GM, GM-B, rerun, moment, ctx .2, ctx+moment, ctx .35, Lyra-2 (rebuilt after queue_fast_b).
- Priority: queue_fast (1046177) SIGSTOPped between steps by scratchpad/prio.sh while the f_v2 training child runs; after the training the GPU goes to
  queue_fast_b (ctx+moment, ctx .35), then queue_fast_c (ViPE horizon 320 on rerun/ctx/moment/ctx+moment/ctx35), then SIGCONT.
- Yaw (fast_camera.py, ViPE vs plan): commands are executed. Showroom plan yaw +18.9 at 320: GM +16.9, GM-B +17.8, Lyra-2 +18.7; translation exec 0.96 / 1.06.
  Playground plan +39.4: GM +42.3, GM-B +42.1 (1.07), Lyra-2 +39.5. The missing white wall is unseen content, not under-rotation.
- Phase II: near100 (near-latent only @100) and f_v1_100 (GM-B recipe @100): NO snap (1.17/1.13, 1.01/.99), NO darkening (lum slope +6.1/+1.1),
  regen_x .98/1.05 (GM-like). Playground 481: last-third DINO GM .686, near100 .695, f_v1@100 .757, GM-B@250 .816 -> collapse fix partly present at 100,
  snap + darkening appear between 100 and 250 updates. Calibration answer: 100 updates under-show the collapse fix; snap/darkening are late.
- queued queue_fast_d.py (pid 1118562): after queue_fast, f_v1 resumed to 200 (ckpts 150/200), both evaluated (playground 481).

### 10 Nov 17:35 - critique: length course to 250 for three arms
f_v1@100 is a new run, not GM-B's update 100 -> extend the same run to 250 (snap/darkening at 250 = length effect; none = GM-B's luck and
retraining is a fix). Brightness left to the moment patch (sweet spot not stable across scenes). Checkpoint chosen by snap + collapse DINO:
early unpatched vs late patched (ctx .2 + moment). queue_fast_d.py (pid 1128152, waits for queue_fast): f_v1 -> 250 (evals 150/200/250 + 250 patched),
f_v1@100 playground rerun (480 noise band), f_render -> 250 (same), f_v2 -> 250 (same). Winner then on the three untouched scenes.
Note: Lyra-2 translation exec on the showroom 1.35 (35% overshoot).

### 10 Nov 17:55 - CPU investigation while the arms train
1. Render softness: Laplacian variance of the render / of its source frame (frontier logs): showroom .45, hall .22-.29, playground .11 (h2: render 346 -> 135
   while the source frames sharpen 1239 -> 1433). The render carries low frequencies only; rendered at 540x960, so this is ReSplat, not resolution.
2. GM-B sawtooth (diag_incall_profile.py): showroom sharpness by in-call position .91 .90 .94 .95 | 1.01 1.03 1.04 (GM / GT / near100 / f_v1@100 flat;
   moment .95->1.01, ctx noise .97->1.02). Latents agree: first new latent std / last history latent std .949 (showroom moment log), HF energy x.953 at the
   boundary, x1.015 per in-call step (hall). Each call starts soft (render-like) and re-sharpens -> snap1 and re-synthesis of detail across the boundary
   (regen_x) are the same event: GM-B no longer carries high-frequency detail through the recent latents, only the soft render crosses the boundary.
   Tone follows the render too (render 7 levels darker than its sources; k ~ .15 per call gives the observed -1.1 levels / call) -> darkening; the moment
   patch forces k = 0.
3. On-policy pool (data864/onpolicy, 600 items of an OLDER model, depth 1-2 calls): generated context vs recorded at the same latents: recorded std x1.037
   (1.035 -> 1.061 with depth), HF energy x1.21 (-> 1.30), mean shift significant in 37/48 channels. GM-B boundary step is anti-aligned with the pool
   correction (cos -.42) and with render->target (-.44); in-call step +.19. Self-aug x0: std .95-.99 of clean (HF now logged: x0_hf_ratio).
4. Weights (GM -> X): |delta| 9.8 @50, 13.4 @100, 20.1 @250 (~sqrt t); cos(f_v1@100, GM-B@250) .355, cos(f_v1@50, f_v1@100) .75, cos(f_v1@100, near@100) .54.
   prope = 78 % of |delta|^2; relative change in 250 updates: camera .96, roles .55, metadata .50, prope .052, backbone self-attention .009-.010.
   The side modules (lr 1e-4) move at the scale of their own norm in every 250-update continuation.
5. Queued before the length course (queue_fast_d.py): module swaps, no training - swap_backbone (GM-B dit.* + GM side modules) and swap_side (GM dit.* +
   GM-B side modules), both scenes, playground 481, with latents/render dumps; then GM / GM-B dumps; all course evals dump latents + renders.

### 10 Nov 18:15 - why is the render soft? (user) + critique (render repair, side-module interpolation)
- Call 1 = one REAL source (anchor): render keeps .42-.54 of the source's Laplacian energy (full res), .72 at 4x scale (hall), and is 13 levels darker.
  With anchor + two generated latents: .35-.45 showroom (flat), hall .47 -> .38 (4x scale), playground .35 until call 6 then .07-.15.
- Training cache, latent scale: recorded sources render/target HF .76 (start >= 13), .98 first call; pool windows: recorded sources .80-.83, generated .72-.78.
  -> softness = fusion/ReSplat baseline + generated sources (pose/depth/content disagreement), growing with rollout depth; coverage is not the limit (valid .85-.99).
- frontier.py: composite_sources (each source rendered alone, s-1 wins, holes from s-2 / anchor / bank; logs lap fused / comp / source) and fix_render
  (tone affine, 'full' = + unsharp to the source's Laplacian energy); runner flags --render_comp, --render_fix tone|full. Unit-checked on CPU.
- queue_fast_d.py order: swap_backbone, swap_side, gmb_comp_tone, gmb_fix, f_v1 course (150/200/250 + patched), f_v1@100 rerun, side_a25/50/75 (if built), f_render
  course, f_v2 course. GM / GM-B plain dumps dropped.

### 10 Nov 18:35 - why ReSplat is soft and dark even from one real frame (user)
- By design: latent_downsample 4 -> one Gaussian per 4x4 pixels (135x240 = 32k per view at 540x960); detail below ~4 px cannot be represented.
  Matches: Laplacian ratio .42-.54 at full res, .72 at 4x scale (call 1, one real source).
- Out of its regime: checkpoint resplat-base-dl3dv-540x960-view16 was trained with 16 context views; we feed 1 (duplicated) to 3. Opacity is calibrated for
  many overlapping views -> accumulated alpha .76-.81 on valid pixels, half of them < .8; colour is composited on black.
- Darkness is alpha only (hall dump): luminance render/source by alpha bin: .5-.7 -> .75-.83, .7-.85 -> .90-.96, .85-.95 -> .99-1.00, > .95 -> 1.01-1.03.
- Also: DA3 depth replaces ReSplat's own depth (PerViewDepth), 864 -> 960 bicubic up and back down, generated views disagree (fusion).

### 10 Nov 18:35 - 8-view ReSplat (user) + follow-up results
- ctx .35: snap 2.51/2.63, DINO .889/.811 - no gain over .2. ctx .2 + moment at 481 (playground): last-third DINO .716, sharpness ratio 3.03 -> combined patch
  rejected (worse than GM-B .816). Yaw of all patched runs within 1-4 deg of plan (eval_gm/fast/camera/yaw_exec.json). Visual sheets rebuilt (80/81/82 ...).
- Downloaded resplat-base-dl3dv-512x960-view8-8179ed87.pth (sha prefix verified) to <workspace>/resplat/pretrained/.
- multi_view.py (source_ids recent/spread N, anchor copies fill the list; render_views with DA3 once per unique frame), probe_resplat_views.py v8|v16
  (offline re-render of the GM-B rerun: cur3 / recent8 / spread8 / recent16, generated and recorded sources, calls 1/3/7/13/19; lap, alpha, luminance,
  PSNR, peak VRAM) via queue_probe.py; queue_fast held (SIGSTOP) by scratchpad/hold2.sh until the probe ends.
- Runner: --sources recent4|recent8|spread8|recent16, --pad_views N (anchor copies), --resplat v8 (512x960 weights), --offload 1 (generator on CPU during the
  render; resplat_mem_864.json: 9.4 GiB for 1 source, 15.4 for 4 -> 8 views will not fit beside the generator).
- Lesson repeated: `pgrep -f` matched my own shell and killed it (exit 144); listing with ps + grep -v grep, kill by exact pid only.

### 10 Nov 19:05 - RENDER GEOMETRY BUG: every 540x960 ReSplat render is stretched vertically by 540/512 = 1.055
- Probe (probe_resplat_views.py, eval_gm/diag/resplat_views_{v8,v16}.json + sheets): recorded sources, first target vs the recorded frame:
  stock path (16-view weights, 540x960) PSNR 15.6-17.6 dB; 8-view weights at 512x960 26-31 dB. Rescaling the 540 render vertically about the centre:
  best factor .948 = 512/540 in 6/6 cases, PSNR 18 -> 24-32 dB. The 512 path is best at 1.000.
- Cause: ReSplat's patch shim crops context AND target to multiples of 64 rows (540 -> 512, 14 rows off top and bottom) and rescales fy;
  gaussverse/memory/resplat_condition.py render() still decodes (540, 960) with those intrinsics -> the centre 94.8 % of the vertical field of view is
  drawn over the whole height (up to 13 px off at the top / bottom of a 480-row frame). Affects every 864-lineage render: training caches
  (data864/renders, pool, mask cache) and inference. 256x448 (multiples of 64) was exact. The generator learned against the stretched render.
- Same probe, weights: with the stock 3 sources the 8-view weights give alpha .89-.95 (16-view .76-.88), valid .99 (.78-.97), 4x-scale Laplacian .73-.90
  (.46-.76). 8 sources: alpha .94-.99, valid 1.0, luminance ratio 1.00; sharpness about the same; PSNR (recorded) not better than 3 sources except at
  call 1 (anchor x8: 27.9 -> 30.7 dB, valid .95 -> 1.0). 8 views need 23.6 GiB (3 views 13.4) -> generator offloaded during the render. 16 views OOM.
- f_v2@100: snap .93/.97, playground 481 last-third DINO .792 (f_v1 .757, near .695, GM .686, GM-B@250 .816).
- Action: f_render 100-update training stopped at update 6 (dir set aside: runs/f_render_aborted_u6_20261110; redone in the course), queue_geo.py runs now:
  gmb_v16s (stretch removed only), gmb_v8, gmb_v8r8 (8 views + anchor copies + offload). Then queue_fast exits -> queue_fast_d: swaps, f_v1 course,
  f_v1@100 rerun, side interpolation (if built), f_render course (100..250), f_v2 course. Render repair tests (comp/fix) removed until redone on a correct render.

### 10 Nov 19:45 - three wrapper errors confirmed in code; outputs' vertical scale; corrected render at inference; lineage v3 queued
- Code (cvg/resplat, verified): (1) shim crops context + target to multiples of 64 rows, our render decodes 540; (2) PerViewDepth interpolates the full-height DA3
  depth onto the cropped grid (depth squeezed against the image); (3) MODEL_PRESETS['dl3dv_16v_540x960'] needs gaussian_scale_max=3, depth_pred_half_res=true,
  num_refine=2 and evaluation uses no_crop_image=true - we ran overrides=[] / num_refine=4. The 8-view 512x960 preset is the default (our v8 use is correct).
- diag_vscale.py (SIFT, sy/sx of generated vs recorded frame): showroom GM 1.002 @16 -> 1.008 @48 -> 1.016 @96 -> 1.030 @160; GM-B 1.007 @64 -> 1.019 @160;
  Lyra-2 .998-1.000, Spatia .993-.998, recording 1.000. Playground: none (.99-1.00). The stretch leaks ~.2-.3 % per call on the showroom.
- Corrected render at inference with GM-B (queue_geo; eval_gm/fast/gate_geo_inference.json): the model un-stretches a render that is now correct ->
  sy/sx .998 @16, .993 @32, .987 @48, .971 @96, .948 @160 (showroom; playground .991 -> .96). Compounding squeeze = inference-only fix invalid, retraining needed.
  Still: snap1 3.85-4.48 -> 2.21 (v16s) / 1.90 (v8) showroom, 1.76 -> 1.49 / 1.56 playground; LPIPS calls 1-3 .045 -> .030 / .028, call 3 .083 -> .046 / .043
  (showroom); regen_x 2.47 -> 1.71 / 1.59; playground last-third DINO (321) .831 -> .858 / .880. Luminance drift unchanged (-8 .. -10 / 100 fr).
- Stopped: queue_fast_d (swaps, length courses) - superseded. queue_v3.py (pid in queue_v3.pid) waits for queue_geo: probe v16p (official preset, no crop) ->
  renderer choice (rule in file) -> cache v3 for old B's 1335 windows -> B' (= old B recipe, corrected render) -> gate (gate_bp.py: LPIPS calls 1-3, vscale) ->
  cache for GM / GM-B windows -> pool v3 (B', frustum mask) -> GM' (frustum-mask metadata) -> GM-B' (v2 corruption, lr_new 2e-5, ckpts every 50) -> rollouts.
  New files: resplat_fixed.py, build_render_cache_v3.py, gate_bp.py, queue_v3.py; train_gmp.py --render_cache / --meta; gen_pool_v2.py --resplat.

### 10 Nov 19:55 - critique applied to queue_v3.py (restarted, pid in queue_v3.pid)
- GM' = GM's exact recipe: pool v3 = gen_pool_v3.py (original gen_onpolicy.py unchanged: 300 streams, seed 20261024, 2 calls, no mask; only the render and the
  output folder differ), raw metadata, no mask in training. Mask tried on GM' at inference only (gmp3_mask). GM-B' = v2 as tested (lr_new 1e-4).
- Checks: run_arm_ckpt.geometry_check before every rollout (recorded window through the rollout's own render object; corrected path must reach 21 dB, value in
  run.json); smoke tests right after the renderer choice (1-call rollouts corrected vs stock, pool wrapper on one stream); gate also stops if B' vertical scale at
  frame 48 is outside 1 +- .01 on both scenes (or +- .02 on one); pool check (>= 450 items, median end label gap <= 2x old pool).
- Renderer choice rule now also needs alpha(gen) >= v8 - .02. Single-source windows keep the stock duplication (x2) in cache and inference (anchor x8 gave
  +2.7 dB at call 1 but needs 23.6 GiB = generator offload; left for the next pass).
- Still missing in the gate: loss with / without render.

### 10 Nov 20:35 - lineage v3 under way
- gmb_v8r8 (8 sources, anchor copies, offload): snap 2.52/1.71, last-third DINO .895/.849, 481: .802 - not better than 3 sources (1.90/1.56, .906/.880, .814); +27 % time.
- Probe v16p (official preset, no_crop, 540x960): every render fails (tensor size 34 vs 33) -> renderer = v8 by rule. To do offline after the chain: preset at 512x960.
- Smoke: runner geometry check corrected 28.15 dB (16.55 if squeezed by .948), stock 16.12 dB (20.06 if squeezed); pool wrapper 2 items.
- Cache v3 (B set): 1335 windows in 1900 s; PSNR p5/p50/p95 18.2 / 25.2 / 32.1, 147 (11 %) below 20 dB, all in ~94 streams (pose-inconsistent recordings?);
  by source count: 1 -> 27.8 dB, 3 -> 24.6, 4 -> 23.6. lum .995, alpha .925, valid .95.
- B' (runs/bp3_s1) started 20:30; first 24 rows identical to old B's log (stream, start, u, render). Gate report gains lum slope / snap / DINO (gate_bp.py).
- 20:45 next-pass prep (critique): probe_fusion_same_windows.py (same recorded windows: fused / nearest / compositing) and v16 preset at 512x960 run after
  the chain (queue_after_v3.py waits for done_v3); train_gmp.py --min_psnr X drops cache windows whose first-target PSNR is below X (11 % of B's rows are
  below 20 dB = rows that teach "do not trust the render"). Not used in this pass (paired comparison with old B).

### 10 Nov 22:10 - training data audit (user: are the samples right now? can bad data be the problem?)
- Right-border zoom (eval_gm/fast/visuals/02267acf_right_zoom_geo.jpg): with the corrected render the large structure survives (wooden table, 5 calls) but the object
  on it is redrawn at every call boundary (box -> cup -> plant); stretch-only row: the whole border is rebuilt every call. My "stays in place" was too strong.
- LENS DISTORTION IGNORED IN THE TRAINING CORPUS (diag_distortion.py, eval_gm/diag/distortion.json): DL3DV transforms are OPENCV with k1, k2, p1, p2; prep_864.py keeps
  only fx, fy, cx, cy and the frames are the raw distorted photos (epipolar Sampson error of SIFT matches drops when keypoints are undistorted: k1 -.116 stream
  0-48: 2.91 -> .46 px, edge 5.16 -> .42; k1 .02-.05 streams edge .5-.9 -> .2-.3; low-k1 streams unchanged). Ignored displacement at 864x480 corners per train
  stream: < 1 px 69, 1-2 px 186, 2-4 px 261, 4-8 px 72, 8-16 px 251, > 16 px 52 (of 891; max 94 px, two streams with k1 -.116). Left/right edge p50 2.2, p90 6.8 px.
  The first-target PSNR check cannot see it (source and target share the distortion; the error grows with how far a point moves across the image).
  The EVALUATION scenes are undistorted (prepared.json: 'calibrated undistortion', distortion_removed) -> train/eval mismatch as well.
- Pose-inconsistent windows: 11 % of cache windows < 20 dB (94 streams); weakly related to distortion (corr -.15) -> separate cause.
- This pass unchanged (B' is paired with old B on the same data). Next pass: --min_psnr 20 and a distortion filter (corner displacement < 4 px keeps 516 streams),
  or undistort the corpus (frames -> VAE latents -> DA3 gauge -> render cache; ~half a day of GPU).

### 10 Nov 22:53 - B' GATE PASSED (eval_gm/fast/gate_bp.json), chain continues (cache for GM / GM-B windows -> render-loss probe -> pool v3 -> GM' -> GM-B')
              showroom B' / old B          playground B' / old B
LPIPS call 1  .015 / .023                  .038 / .067
LPIPS 1-3     .031 / .052 (-40 %)          .081 / .113 (-29 %)
LPIPS call 3  .044 / .082                  .118 / .151
vscale @48    1.001 / 1.024                .997 / 1.002      (@160: .995 / 1.054 showroom; old B stretches its OUTPUT by 5 % by frame 160)
lum slope     +6.2 / +5.8                  +0.1 / -8.8       (old B darkens the playground; B' does not -> darkening was learned from the dark render)
snap1         1.26 / 1.07                  1.29 / .78        (B' higher - watch in GM')
last-3rd DINO .856 / .880                  .747 / .724
Geometry check: B' 28.2 / 26.9 dB, old B (stock render) 16.1 / 17.1 dB.
- 23:00 (critique): GM' gate report added to queue_v3 (restarted; running cache step kept): gate_bp.py --out gate_gmp3 with gmp3 vs GM vs GM-B vs B' (LPIPS calls 1-3,
  vscale incl. frame 128 - frame 160 on the playground has < 25 SIFT inliers, lum slope, snap, 481-frame last-third DINO: GM .686) + render-loss probe GM' vs GM.
  Render-loss probe B' vs old B runs after this cache step (before the pool). Report only, never stops the chain.

### 11 Nov 08:00 - overnight: GM' FAILED at 02:46 (missing cache window), GPU idle ~5 h; fixed and restarted
- Cause: the base sampler renders a recorded window BEFORE the pool may replace the row, so windows that old GM replaced by pool rows are not in its log;
  the cache held only logged windows. Fix: queue_v3 builds the whole cache (3553 windows, ~45 min) before GM'; failed dir set aside (runs/gmp3_s1_failed_missing_cache_20261111).
  New ETA: GM' ~11:00, GM' gate ~11:30, GM-B' ~13:30, its 4 checkpoints evaluated ~15:00.
- Pool v3 ok: 600 items / 300 streams, median end label gap .027 (old pool .035).
- Render-loss probe (eval_gm/diag/render_loss_bp3_oldb_20261110.json, 48 unseen windows x 2 noise): B' gains .0096 +- .0034 from the corrected render (chunk 0:
  .280 -> .259), .0012 from the stock render; old B gains .0024 +- .0040 from its own stock render, .0012 from the corrected one. B' uses the render ~4x more.
- Fusion on the same recorded windows (60, >= 3 sources): first target, common pixels: fused 24.95 dB, nearest only 23.29, compositing 24.26 (nearest > fused in
  17/60); last target: fused 17.79, nearest 15.23, compositing 15.70; valid last: .98 / .67 / .88; Laplacian ratio last: .24 / .37 / .63.
  -> fusion is not worse in PSNR on recorded frames (it fills more and averages); compositing keeps far more detail (lap .63 vs .24) at -2 dB.
- 08:20 (critique): dry_run_sampler.py = model-free run of a training's full row sequence with a cache existence check; validated: B' args -> 2000 rows, 1508 render /
  1335 windows / 0 missing (= old B); GM' args on the partial cache -> 43 missing, first = 23cf8f5a start 55 (the window GM' died on). queue_v3 runs it before every
  training (restarted). Watchdog (background task) alerts on failure / end / GPU idle 8 min. Fusion probe v2 (after chain): + hybrid (compositing, fused fill:
  frontier comp=2) and LPIPS on the common valid region (outside = target); GM' + hybrid at inference added to the after-chain candidates.

## 2026-11-11 11:55 GM' gate (gate_bp.py crashed in-chain: `out = {}` had slipped into a comment; fixed, re-run on CPU -> eval_gm/fast/gate_gmp3.json)
| | showroom gmp3 / GM / GM-B / B' | playground gmp3 / GM / GM-B / B' |
|---|---|---|
| LPIPS calls 1-3 | .024 / .028 / .044 / .031 | .070 / .092 / .094 / .081 |
| vscale @160 | .996 / 1.029 / 1.018 / .995 | .965 / .992 / - / 1.024 (@128 .997 / .981 / .996 / .963) |
| lum slope / end | +7.6 / +21 (GM +3.9 / +13) | +1.9 / +13 (GM +0.9 / +10) |
| snap1 | 1.18 (GM 1.03) | 0.90 (GM 1.15) |
| last-third DINO 321 | .899 (GM .925) | .647 (GM .797) |
| playground 481 DINO / sharp | | **.462** (GM .686, GM-B .816, f_v2 .792) / 2.72 |
Render-loss gap (corrected render vs none): GM' .0072 +- .0053 (chunk 0 .282 -> .262), GM .0070 +- .0031 (stock .0044).
Reading: GM' wins the first calls everywhere (LPIPS 1-3 -14% / -24% vs GM), geometry flat, no snap; but the long playground rollout collapses EARLIER than GM
(slide is a green blob by 240, scene smeared by 400, pole/trunk by 480; visuals/0238de14_gmp3_vs_gm_481.jpg). Without corruption training the long-horizon
failure is unchanged or worse -> GM-B' (v2 corruption) is the arm that has to carry the collapse. Showroom brightening (+21 end) is a flag for GM-B'.
- 11 Nov 12:40 COPY-CHAIN CHECK (diag_copy_chain.py -> eval_gm/diag/copy_chain.json; per call newest-source / render / output Laplacian variance). Playground GM': src 880 -> 3076,
  render flat 450-620, output 1006 -> 3353 (call 28); GM output 1052 -> 2549; GM-B v8 output 986 -> 4060 (render 600-775). Showroom flat for all (output ~800-1000, render ~450).
  => the "output follows a blurring render down" prediction FAILS: the render never gets softer in absolute terms, the output climbs far above it (3-7x), and the
  climb is high-frequency smear/grain (sharp 2.72 = grid-suspect), not blur. GM-B with the same v8 render climbs even more yet keeps DINO .814 -> HF level does
  not separate collapse from no-collapse. Open: structural (low-freq) copying of the render cannot be judged without render dumps -> queued runs carry --render_dump.
- 11 Nov 12:45 queue_after_v3 re-armed (pid 1550158): after done_v3 FIRST gmp3_comp (priority compositing) + gmp3_oracle (new run_arm_ckpt --oracle 1:
  recorded frames at recorded plan poses as ReSplat sources, test-scene harness), both long + render dumps; then fusion probe v2, firstseen, firstseen+comp, hybrid.
  + gm_oracle (GM receipt, stock render) as same-harness reference; pid 1550468.
- 11 Nov 13:40 POOL CONTEXT vs RECORDING (critique: is the on-policy context soft/dark -> learned "sharpen/brighten" rule?). diag_pool_moments.py (latent, all 600 items/pool):
  generated context HF ratio .83 old / .834 v3, falling with depth (.86-.87 at latent 1-2 -> .78 at 6-8); per-channel std .97; |mean shift| .13 / .11 sd -> pools identical.
  diag_pool_pixels.py (CPU decode, deepest generated latent vs recorded, 12 s9 items/pool, same streams): Laplacian ratio .78 / .77, contrast 1.01 / .98,
  luminance median -9.8 old / -1.1 v3 (spread +-15). => the context IS ~22% softer than the target in both pools (sharpening rule plausible, shared by every
  pool-trained arm); the v3 pool is NOT darker -> the showroom +21 brightening of GM' is not explained by a dark pool context (old pool was darker, GM +13).
  v2 moment matching (gm_b_corruption.match_moments) applies ONLY to self-aug rows; on-policy pool rows return untouched (line 84) -> GM-B' inherits the pool's
  soft-context rule. Next training: match on-policy context to the recorded per-channel mean/std AND HF (moments alone leave HF at .83).
  Queue: + gmp3_moment (--moment_patch 1, v8, render dump) after gmp3_comp / gmp3_oracle (pid 1559408).
- 11 Nov 14:25 CRITIQUE: no proof the long-horizon collapse comes from the render (GM' fixed render, collapse worse; GM-B fixed collapse with the broken render;
  hardening passes through near latents; memory candidates changed nothing). Decisive test = clean each error path separately:
  free (own/own, have) | oracle render (recorded sources, own context; --oracle 1) | oracle CONTEXT (own render sources, recorded near+bank latents; NEW --oracle_ctx 1:
  recorded frames -> causal VaeStream as prep_864, roller.pack sees recorded latents 1.., anchor and render memory untouched). Both leak -> read the DIFFERENCE.
  queue_v3 tail reordered and restarted (pid 1568084, V3_WAIT_PID = training 1532260): gmbp3_250 -> gmp3_octx -> gmbp3_250_octx -> gmp3_oracle ->
  gmbp3_250_oracle -> gmp3_moment (all v8 + render dump, both scenes, playground 481) -> gmbp3_100/150/200 -> gmp3_mask -> done_v3 -> queue_after_v3
  (gmp3_comp, gm_oracle, fusion v2, firstseen, hybrid). No new render work until the three-way table is in. Done guards added to the two render-loss probes + gate_gmp3.
  Second critique: collapse <-> consistency may be one dial (trust in context); per GM-B' checkpoint read 481 collapse AND regen_x together; statistics-only
  corruption (HF / brightness / contrast both ways, content = recording) is the candidate if gmp3_moment delays the collapse.
- 11 Nov 13:10 GM-B' training done 13:05 (250 updates). Critique: collapse (stability, measure = playground 481 last-third DINO) and consistency (memory, measure = regen_x +
  region persistence) are SEPARATE problems (100-update arms fixed half the collapse without regen; corrected render improved regen without touching collapse).
  Working split: collapse <- training recipe; consistency <- memory/render (priority compositing candidate). Prediction to test with the queued runs: oracle context fixes
  collapse, oracle render fixes consistency. Checkpoint choice for GM-B': the one that fixes collapse without raising regen_x; consistency work then on top of it.
  (Clock note: my ETAs in the 12:40-13:00 replies were ~1.5 h late; actual order ends ~15:05.)
- 11 Nov 13:50 FIRST THREE-WAY RESULTS (playground 481 / showroom 321; visuals/0238de14_gmbp3_octx_481.jpg).
  GM' oracle CONTEXT (own render, recorded near+bank latents): playground 481 DINO .975 (free .462), sharp 1.04 (2.72), lum end +1.3; showroom DINO .983 (.899), lum +2.6 (+21).
  Visually perfect through 480. => with a clean near/bank context the collapse, the hardening AND the brightening vanish although the render is still built from
  the model's own frames -> the collapse is carried by the context path, not by the render. (Leak caveat: recorded latents up to s-1 sit right before the target;
  read against oracle render, still queued. regen_x/snap of octx not interpretable: the next call continues from the recording, not from the own last frame.)
  GM-B' @250 free: playground 481 DINO .787 (GM' .462, GM-B .816) but lum end -48.9 (GM-B -24), regen_x 1.95-2.0 (GM' 1.04-1.19), snap 3.0-3.3 -> collapse
  mostly fixed, consistency + darkening WORSE than GM-B; v2 amplitude preservation did not stop the darkening; visually the scene drifts dark/orange and the
  structure re-draws. Checkpoints 100/150/200 decide whether a point exists with collapse fixed and regen ~1.
- 11 Nov 14:05 CORRECTION after visual check (visuals/{0238de14,02267acf}_octx_check.jpg + LPIPS vs recording at mid-call frames t=8+16k):
  oracle-context output is CLOSER to the recording than the recording is to itself 8 frames earlier (playground .067/.085 vs GT(t-8) .091/.102; showroom .017/.032
  vs .045/.067) and reproduces recording artefacts (showroom floor structure at 240/320) -> the run is mostly LEAK (it continues the recording), DINO .975 says
  little about which path carries the collapse. Withdrawn: "collapse carried by the context path, not the render" (13:50). What it does show: from a clean recent
  context one call adds almost no error.
  GM-B' @250: LPIPS vs recording last third .562 playground / .280 showroom vs GM' .605 / .321 -> visually it does not melt, but by 320-480 it is a DIFFERENT
  playground (dark, orange slides, other layout) and the showroom gains a person / other display -> last-third DINO (.787) overstates the collapse fix; it measures
  plausibility, not fidelity. Read collapse with LPIPS-vs-recording next to DINO from now on.
  Leak-free version of the path test: oracle PREFIX then free (oracle context / oracle render for calls 1-10, own afterwards) -> compare drift rate per call after
  the switch; and oracle render vs oracle context only as a difference (both leak).
- 11 Nov 14:40 FIDELITY metric (diag_fidelity.py -> eval_gm/diag/fidelity.json: spatial LPIPS vs recording on ANCHOR-VISIBLE cells, every 8th frame, per call,
  thirds, slope; showroom compared over calls 1-20 for all). Playground first / last third: GM .282/.646, GM-B .264/.611, GM' .212/.642, GM-B' .211/.629 ->
  on the same-scene measure NO arm holds the playground (GM-B' "collapse fix" = plausibility only). Showroom calls 1-7 / 14-20: GM .081/.260, GM-B .104/.267,
  GM' .060/.238, GM-B' .067/.196. Oracle context: GM' .124/.141 (pg), .037/.063 (sr); GM-B' .174/.325, .058/.124 -> with the SAME clean recorded context GM-B'
  drifts away from the recording 2-3x more than GM' (leak identical for both): GM-B' discounts even clean context/anchor content (critique: corruption also
  teaches distrust of the anchor-derived render) -> recipe idea for next week: corrupt only what will be generated at inference (near latents + render cells
  from generated sources), keep anchor-derived cells clean with the confidence channel saying which is which (needs priority compositing in the cache).
  Queue (pid 1604370): gmp3_oracle (running) -> gmbp3_250_oracle -> gmbp3_250_moment -> gmp3_pfx10 -> gmbp3_250_pfx10 (new --oracle_prefix 10: full oracle
  calls 1-10, own afterwards; read the fidelity slope after the switch) -> gmbp3_100/150/200 -> gmp3_moment -> gmp3_mask. Checkpoints read with 3 measures:
  quality (DINO, sharp), fidelity (this metric), self-consistency (regen_x).
  Pool idea (critique, untested): reset the context to the recording every k calls while generating the pool (k = 3 -> grow), rows labelled by depth 1..k,
  measured-pose acceptance threshold, report acceptance by depth.
- 11 Nov 15:30 FIDELITY REFERENCE (same metric, released baselines): playground first/last third Lyra-2 .225/.571, Spatia .249/.555 (ours .21/.61-.65, oracle ctx .141);
  showroom calls 1-7 / 14-20 Lyra-2 .070/.367, Spatia .152/.278 (GM' .060/.238, GM-B' .067/.196). -> the last-third number is mostly scene difficulty + camera
  drift; on the playground we are .04-.09 worse than both baselines, on the showroom better than both. "No arm holds fidelity" stays true but no baseline does either.
- SELECTIVE TRUST built (critique 15:00; minimal form, no new cache): anchor_seen.py (DA3 anchor depth precompute -> anchor_depth/, z-buffered splat of the anchor
  into the 4 render-slot cameras, latent cells with >50% coverage = seen; 15 ms/row; plane test: seen .97 at start 1-5 -> .34 at start 41);
  train_gmp.py --selective 1: conf = 1 on seen render cells for ALL render rows (incl. pool rows, real stream id = OP id[3:]), corruption noise + (valid,conf) x (1-s)
  only on unseen cells (gm_b_corruption, row['_seen']); unit test: seen cells unchanged (0.0), unseen noised, near slots untouched. Inference: run_arm_ckpt --anchor_conf 1
  (same rule on the command cameras, anchor depth from DA3 of the anchor frame). Queue (pid 1610398): after gmbp3_250_oracle -> anchor_depth (GPU) -> 2-update
  training smoke -> gmp3_aconf (untrained GM' + conf rule, wiring check) -> rest. Launch of the 250-update run (init GM', v2 + --selective 1, seed 20261111) decided
  after the GM-B' checkpoints (~17:10).
- 11 Nov 14:50 THREE-WAY TABLE COMPLETE (fidelity = LPIPS vs recording on anchor-visible cells; playground last third | showroom calls 14-20; DINO 481 playground):
  GM'  : free .642 | .238 (DINO .462) ; oracle RENDER .508 | .156 (DINO .660, sharp 2.63 = still hardening) ; oracle CONTEXT .141 | .063 (DINO .975)
  GM-B': free .629 | .196 (DINO .787) ; oracle RENDER .639 | .146 (DINO .517, snap 14.9: render and own context fight) ; oracle CONTEXT .325 | .124 (DINO .932)
  Both oracles leak, so read the difference: a near-ground-truth RENDER of the target does not hold the playground (GM' .51, GM-B' .64) while a clean CONTEXT does
  (.14 / .33) -> the collapse travels through the near latents; the render can correct it only partly (showroom more than playground). Supports: labour goes to
  the context side (robustness to own history) - not to the render - for collapse; render/memory work is for self-consistency.
- USER CRITERION (11 Nov ~15:30): what the anchor shows is the given world and must be kept; newly generated content need not match the recording but must STAY
  once generated. -> measure = out-and-back plan (run_arm_ckpt --return_traj 1: calls/2 out on the recorded path, same poses back; diag_return.py: spatial LPIPS
  frame t vs frame 2n-t, anchor-seen vs generated cells, near/far revisit gap). Wanted model = copies like GM' but does not melt.
- BUILT: statistics-only corruption (train_gmp --corrupt stat; gm_b_corruption.stat_perturb: one coherent draw per row on bank+recent latents: HF gain logU(.65,1.6),
  per-channel std scale logU(.85,1.2), per-channel mean shift N(0,.25 sd); p .6; no one-step replacement, render untouched; test: content corr .94-.98, HF ratio .38-1.42).
  Inference: --unseen_conf c (multiplies conf on cells the anchor does not see, with --anchor_conf 1).
- QUEUE (pid 1631440): gmp3_aconf (wiring) -> gmp3_moment -> gmp3_ret, gmbp3_250_ret -> AUTOMATIC CHOICE: 'stat' iff playground-481 DINO(GM'+moment) >= DINO(GM' free)+.10
  and regen_x <= 1.3, else 'selective' -> ONE 250-update training from GM' (seeds as GM-B': --seed 20261108 --gmb_seed 1; runs/gmst3_s1 or runs/gmsel3_s1)
  -> new model: free (long), oracle context (trust tax), out-and-back (+ unseen_conf .8/.6 if selective) -> gmbp3_250_moment, pfx10 x2, GM-B' 100/150/200, gmp3_mask.
- SUCCESS CRITERIA for the new model (written before training): quality playground-481 DINO ~.78, no melt | fidelity playground last third toward .57 (Lyra-2 .571,
  Spatia .555; ours .63-.64) | self-consistency regen_x < 1.3 + out-and-back generated-cell LPIPS at GM' level | trust tax: oracle-context fidelity < .18
  (GM' .141, GM-B' .325) | brightness within +-10 (GM-B' -49). Rows to show next to it: GM', GM-B', untrained GM' + conf rule (gmp3_aconf).
- 11 Nov 15:10 CRITIQUE accepted: (1) render work drops to second priority for the collapse (a near-perfect render does not stop it; content travels through the
  near latents). (2) No automatic recipe choice: tonight's ONE training = statistics-only from GM' (runs/gmst3_s1, --corrupt stat --render_corrupt 0, seeds as GM-B');
  selective trust shelved (it corrupts exactly the generated cells where self-consistency is wanted). The moment-patch run only sets the expectation; if it is
  clearly worse (playground-481 DINO -.10) no training starts and the user is asked. (3) Anchor latent stays clean in stat mode (slots = bank + recent only; test).
  (4) Out-and-back measure gets a floor (same pose 2 and 8 frames apart at the turning point). Order (pid 1643576): gmp3_moment -> training (~2 h) -> gmp3_ret,
  gmbp3_250_ret (validate the measure: generated cells GM' << GM-B', anchor-seen cells close) -> gmst3 free / oracle context / out-and-back -> rest.
  Second round idea: separate draws for near vs bank latents (near drift > bank drift at inference).
- 11 Nov 15:15 MOMENT PATCH on GM' (statistics only, no training): playground 481 DINO .462 -> .683 (= GM .686), sharp 2.72 -> 0.90 (hardening gone), regen_x 1.16
  (free 1.19), lum end +11.9; 321: DINO .647 -> .800. Fidelity last third .642 -> .611. Showroom: DINO .893 (.899), lum +13 (+21), snap 2.1 (1.2, the patch's step).
  => the melt is at least partly a STATISTICS runaway: fixing only per-channel mean/std at each call start removes the hardening and half the DINO loss with
  no cost in regeneration. Supports the statistics-only recipe; training gmst3 started (dry run 15:15).
- 11 Nov 16:50 gmst3 training at update 190/250 (loss .25; logged rows ~1/3 stat, 1/3 clean, 1/3 pool). Critique: patched GM' = floor for the trained model (DINO .68,
  regen 1.2, snap at free-GM' level); if training lands below the patch, try patch ON the trained model. Added (pid 1676872): gmp3_moment_ret (does the patch's
  call-start step leave a trace on the way back?), gmst3_moment, and the patch REFERENCE test: source_policy.install_moment(ref=prev|anchor|mix) /
  run_arm_ckpt --moment_ref; the patch as run matches the first new latent to the PREVIOUS latent's per-channel mean/std -> drift is carried (lum +3.8 -> +11.9);
  'anchor' = anchor latent's moments, 'mix' = half way (view content changes, so pure anchor may be too rigid). Order after training: gmp3_ret, gmbp3_250_ret,
  gmp3_moment_ret -> gmst3 free / octx / ret / +moment -> gmp3_moment_anchor, gmp3_moment_mix -> gmbp3_250_moment, pfx10 x2, GM-B' 100/150/200, mask.
- 11 Nov 18:10 gmst3 trained (250 updates, done ~17:15). OUT-AND-BACK MEASURE VALIDATED (diag_return.py, eval_gm/diag/return.json, visuals/0238de14_return_check.jpg;
  spatial LPIPS frame t vs same pose on the way back; A = anchor-seen cells, N = generated cells; floor = same pose 2 / 8 frames apart at the turn):
  showroom (160 out): GM' all .080 (A .079, N .089; floor .014/.036) | GM'+moment .080 (A .078, N .093) | GM-B' .239 (A .237, N .205; floor .040/.101)
  playground (240 out): GM' all .365 (A .353, N .405; far gap N .470; floor .058/.086) | GM'+moment .332 (A .322, N .377) | GM-B' .516 (A .507, N .539; floor .151/.258)
  -> known difference reproduced (GM' << GM-B' on generated cells, x1.3 playground / x2.3 showroom) and visible (GM-B' returns to a scene without the buildings).
  Expectation "anchor-seen cells close for both" is NOT met: GM-B' re-draws anchor-visible content as well (x1.4 / x3), consistent with its trust tax (.325).
  GM-B' also changes 2-8 frames apart at the turn (floor .15 / .26 on the playground). The moment patch leaves no trace on the way back (slightly better on the playground).
  GM' on the playground is far from its own floor (.37 vs .06-.09): self-consistency over 480 frames is an open problem for GM' too.
- 11 Nov 18:20 USER: "still no consistency in any model". STRUCTURAL FINDING (code + run logs + offline replay of the selection on the out-and-back plans):
  data.select_bank picks bank latents by frustum-coverage GAIN over anchor + the 4 recent latents and stops at gain 0. On a revisit the recent latents already
  cover the view -> bank = [] in EVERY call of the way back (15/15 playground, 10/10 showroom), and render_own's sources are [anchor, s-1, s-2] (frontier_log
  sources = 3). The frames the model generated at these very poses are never retrieved, neither as context latents nor as render sources -> on a revisit the only
  memory is the anchor + the drifted last 16 frames; consistency of generated content is impossible by construction, whatever the training recipe.
  (Same rule in training rows: recorded DL3DV streams rarely revisit, so the model has also never been TRAINED to copy a same-pose bank latent.)
  FIX TEST (inference only): source_policy.bank_revisit = nearest-pose stored latent per target latent end (dt <= .25 gauge units, rot <= 15 deg, not among the last
  8 latents), remaining slots by the stock rule; replay: 55/60 and 35/40 exact same-pose hits on the way back; bank[0] (oldest) also becomes the 4th render source.
  Queued next: gmp3_ret_revisit (GM', --bank revisit), later gmst3_ret_revisit. If the model does not use the retrieved latent (it was never trained on
  same-pose bank rows): training rows from PALINDROME streams (recorded clip forward then backward: exact targets for free, bank holds the same-pose latent).
  RE10K test-split download: asked, user dismissed the question -> not started.
- 11 Nov 18:35 USER: "the scene keeps changing while going FORWARD too". Checked: the stock bank is empty on the FORWARD plans as well (0 of 29 calls playground,
  0 of 19 showroom; render sources = 3 in every call of the GM' runs). Every rollout evaluated so far ran with memory = anchor + last 16 frames (4 latents) +
  a render of [anchor, s-1, s-2]; no frame older than 16 frames is ever supplied. And the anchor, although it covers 69-100% of the target frustum samples on
  the playground through call 26, does not hold the content (fidelity on anchor-visible cells .21 -> .64) -> two separate gaps: (1) SUPPLY: nothing older is
  retrieved; (2) USE: the model weights the near latents over anchor / render (three-way table), because in recorded training rows the near latents are clean
  and adjacent to the target, so copying them is always right.
  Test (inference only, queued after gmp3_ret_revisit): gmp3_memory = forward plan with --bank memory (same-pose latents >= 16 latents old, then FIRST-SEEN
  owners of what the anchor does not show; replay: 4 bank latents from call 9 on). Read: fidelity + regen_x + DINO vs GM' free.
  If supply alone does not help -> training rows that teach USE: near latents content-drifted (self-aug), anchor + bank + render clean, bank chosen by the
  memory rule (GM-B noised bank and render as well -> taught distrust of everything).
- 11 Nov 18:55 CRITIQUE: "was the bank filled in TRAINING?" Row logs: bank > 0 in 61-71% of recorded rows with start >= 13 (plain-350 .61, old B .61, GM .64, B' .66,
  GM' .71, GM-B' / gmst3 .67), mean bank .5-.8; pool rows 10-13%. So the model WAS trained with a bank; at inference on the test plans it is empty in every call.
  WHY (new, checked on poses): the test plans are CONSECUTIVE 60 fps source frames (continuous_eval index: "no pose interpolation"), the training corpus is the
  DL3DV-960P frame set (328 frames per scene). Per-frame camera step, gauge units (anchor median depth = 1): playground .0020, showroom .0015 vs training
  median .0271 (p10 .0133, p90 .0494) -> 14x / 18x SLOWER than training, below the 10th percentile by 7-9x; rotation per 16-frame call 2.4 / 1.0 deg vs 20.4 deg.
  The whole 480-frame playground rollout (30 calls) covers a path of .94 units = what a training stream covers in ~35 frames (2 calls).
  Consequences: (1) bank gain is 0 because the recent latents cover everything; (2) the same content is re-generated 30 times where training re-generates it ~2
  times -> per-call error compounds 14x more per unit of camera motion; (3) the model has never seen 2-deg calls. This is a train/test mismatch of the same class
  as the render geometry and applies to every arm evaluated on this cohort (and to the SOTA table: Lyra-2 / Spatia are native-frame-rate video models).
  TEST (queued first, pid 1718970): run_arm_ckpt --stride 4 / 8 = every 4th / 8th recorded pose, 7 / 3 calls over the same path, GM' and GM; diag_stride.py
  compares fidelity and sharpness at the SAME recorded poses against the stride-1 runs. If stride-N holds the scene: generate at the training stride and fill
  in between (or train on consecutive-frame streams: data/DL3DV-source2K-1280x704 is on disk).
- 11 Nov 19:05 USER: "why are the last frames not enough? Lyra used the same training and eval set". Lyra-2 TRAINING.md (lyra/Lyra-2, on disk): trained on the
  DL3DV-10K SOURCE VIDEOS, every other frame (temporal stride 2), only videos > 1000 frames after subsampling, 80-frame chunks. Ours: the DL3DV-960P frame set
  (~328 frames per scene, ~14x the per-frame motion of consecutive frames), 16-frame calls. Same dataset name, different sampling: Lyra's training speed is ~2x
  the test speed (ours ~14x) and it covers 480 frames in 6 calls (ours 30 -> 5x more copy-of-a-copy hops). Lyra is in distribution on this cohort, we are not.
- 11 Nov 18:35 USER: "Memorizon itself does not have this problem on its own dataset; why do we?" Read from baselines/memorizon (README, configs/train_100s.yaml,
  data.py, retrieval.py): (1) RETRIEVAL: top-k per chunk by fov_overlap - 0.2 (|dt| + 4 angle), k = 6 per chunk, bank up to 60 frames, ALWAYS filled; ours =
  data.select_bank from the older joint_v9 line (coverage GAIN over anchor + recent, <= 4, empty on slow motion / revisits). (2) TRAINING: spans of 10-100 s,
  layout [A | bank | recent chunk | 10 target chunks]; everything older than the recent chunk reaches the model ONLY through the bank -> it must learn to read
  it; ours: 2 target chunks, bank mean .5-.8 frames. (3) DISTILLATION on its own rollouts with 0-20 s of SELF-GENERATED history in the memory; ours: pool depth
  1-2 calls, bank in 10-13% of pool rows. (4) SPEED: 16 fps, actions .25 m / 7.5 deg per 1-s chunk; our test plans are .10 m (playground) per 16-frame call =
  INSIDE Memorizon's native regime; our 850 fine-tuning updates on DL3DV-960P frames (1.5 m and 20 deg per call) moved the model away from it.
  => we kept Memorizon's architecture but replaced its memory protocol (rule, size, span, self-generated history) and its speed regime.
  Test added: source_policy.bank_memorizon = its exact rule (--bank memorizon, max 4 slots as our packing was trained); replay: forward -> the 4 latents before
  the recent chunk, way back -> the same-pose latents. Queued gmp3_ret_mz, gmp3_mz (pid 1724749) after the stride runs.
- 11 Nov 18:40 STRIDE TEST, GM (old weights, stock render; GM' runs failed on my geometry check - it paired strided poses with recorded frames 0/16/17 - fixed,
  re-running). Fidelity (LPIPS vs recording, anchor-visible cells) at the SAME recorded poses t = 32..448:
  playground: GM stride 1 (28 calls) mean .497 / end .660 | GM stride 4 (7 calls) .411 / .477 | Lyra-2 (6 calls) .427 / .575 | Spatia .417 / .575
  showroom:   GM stride 1 .232 / .346 | GM stride 4 .173 / .293 | Lyra-2 .302 / .510 | Spatia .261 / .402
  sharpness ratio at the end, playground: stride 1 2.91, stride 4 1.46 (at call 7 of stride 1 it was ~.85) -> hardening grows mainly with the NUMBER OF CALLS,
  partly with the path. => at a matched number of calls (7 vs Lyra-2's 6) GM is at or better than Lyra-2 on both scenes; the cohort gap is chain length.
  Caveats: one seed; stride-4 produces every 4th frame only (120 of 480); GM = old weights with the stretched render.
- Critique notes accepted: two regimes (cohort: slow, out of distribution, long copy chain | dev: fast, in distribution, hard new views; speed mismatch explains
  only the first); bank test expectations (way back: Memorizon rule should help; forward: its 4 latents are just the previous chunk -> first-seen rule is the
  hopeful one); the four Memorizon differences are hypotheses; bank size is VRAM-bound (24.5 GB with 4 slots + 2 chunks); do not write the 7-day recipe before
  tonight's three results. Added: run_memorizon_cohort.py = RELEASED Memorizon (official pipeline) on the two scenes, forward + out-and-back (queue pid 1730147).
  Open candidates (not started): mixed-stride training from the source videos; two chunks per call at inference; Lyra-2 on the fast dev scenes (mirror test).
  Queue restarts no longer recompute finished runs' CPU metrics (queue_fast.gen) - that had idled the GPU for minutes after every restart today.
- 11 Nov 18:55 STRIDE TEST, GM' (re-run after the geometry-check fix) + visuals (visuals/{0238de14,02267acf}_stride_check.jpg; rows recording / GM / GM stride 4 /
  GM' / GM' stride 4 / Lyra-2 at the same recorded poses). Fidelity mean / end: playground GM' stride 1 .443 / .672, stride 4 (7 calls) .353 / .455, stride 8
  (3 calls) .337 / .424 (Lyra-2 .427 / .575); showroom to frame 320: GM' stride 1 .155 / .271, stride 4 .127 / .224 (Lyra-2 .235 / .411). Sharpness at the end,
  playground: 3.71 -> 1.79 (stride 4) -> .84 (stride 8).
  Visually: playground GM' stride 4 holds the scene through 448 (clean buildings, structure, ground; the slide grows longer than in the recording), where GM'
  at every pose melts from ~256 and GM stride 4 (stretched render) ghosts; Lyra-2 is clean but also changes the slide. Showroom GM' stride 4 holds to 320, then
  a dark distorted object enters at the right edge at 384-448 (newly revealed region; recording shows a white wall) - also in GM stride 4 from 256.
- 11 Nov 19:00 USER: "02267acf sparse: still no consistency on the right side". Zoom (visuals/02267acf_stride4_right.jpg, right half, call by call): the display panel the
  anchor shows stays put; the region REVEALED to its right is re-drawn at every call: call 3 a cart leg, call 4 a shelf with objects, call 5 a tall dark smeared
  column, call 6 an orange object below it, call 7 a grey column (recording: white wall + radiator panels). Within a call it is stable. So the sparse plan removes
  the copy chain for existing content but NOT the user's criterion: generated content does not stay from one call to the next (and it is implausible).
  Candidates (not established): the render of that region is built from the previous call's generated edge strip with DA3 depth -> stretched splats (the dark column
  looks like one) and GM' copies the render; or the bank is empty and only s-1/s-2 carry it. Queued on the showroom, stride 4, 7 calls: render dump (what does
  the render show there), frustum mask at inference, Memorizon bank rule (queue pid 1738362).
- RELEASED MEMORIZON, forward (run_memorizon_cohort.py, 68-79 s per scene): fidelity first/last third playground .397 / .704, showroom .313 / .615 (ours .21 / .64,
  .06 / .24). Visual (visuals/*_memorizon_released.jpg): the picture hardly leaves the anchor view over 320 frames - the base does not execute this slow command
  (matches the 13-18 Oct finding "executes .32 of the command"); it is stable because it barely moves, so it is not evidence that the base keeps a scene under
  camera motion on this cohort. Out-and-back run in progress.
- 11 Nov 19:05 USER: released Memorizon is not a relevant reference here (never trained on this dataset) -> dropped from the reading; no further Memorizon-released runs.
- 11 Nov 19:15 RIGHT EDGE of the sparse showroom run, three diagnostics (visuals/02267acf_stride4_right_render.jpg; render dumps):
  (1) RENDER: calls 3-4 the revealed strip is a hole (valid 0); from call 5 the render shows a dark mass there and from call 6 the tall dark marbled column +
  orange patch - stretched splats built from the previous call's generated edge strip, marked VALID - and the output copies it. The column is a render artefact
  that GM' reproduces (GM' copies the render). (2) FRUSTUM MASK at inference: the column is gone in calls 5-6, other junk (dark object at the top, glassy /
  orange object bottom right) appears in 6-7 -> partial, not a fix. (3) MEMORIZON BANK rule: nothing changes in the new strip (no frame ever saw it), column stays.
  AGE SPLIT (critique; diag_boundary_age.py -> eval_gm/diag/boundary_age.json: adjacent-frame spatial LPIPS across the call boundary / mean inside the call;
  A = anchor-seen cells, G_old = render-valid but not anchor-seen = generated in earlier calls, NEW = render-invalid): stride 4: A 1.08, G_old median 1.08
  (calls 5 and 7: 1.67 / 1.68 = when the column grows), mask: G_old 1.20, NEW 1.9; Memorizon bank: G_old 1.15 (calls 5-7 1.23-1.34). Reference: GM' + moment
  at every pose 1.12 (pg) / 1.23 (sr); GM-B' oracle render 3.76 (A 2.68) -> the measure does see re-drawing where it exists.
  => content generated in an EARLIER call largely survives the boundary (same ratio as anchor content); what fails is the first generation of the newly
  revealed strip (implausible content) and its stretched-splat render one call later. Quality-at-the-frontier problem + render validity, not a general
  failure to keep generated content. Stride trade-off noted: stride 4 = 4x wider revealed strip per call.
- 11 Nov 19:10 SUPPORT-DENSITY MASK built (critique): frontier.support_density = every source pixel (full 512x960 resolution, DA3 depth) dropped as a point into
  each target view, per 120x216 cell landed points / nominal source pixels per cell, max over sources; observed = density >= support_min (.15) + closing 1;
  valid/conf zeroed elsewhere (render untouched); logs the fraction of valid pixels dropped at thresholds .05/.1/.15/.25/.5 (8 Nov's support mask dropped 43-57%
  on the broken geometry; healthy would be 10-15%). run_arm_ckpt --frontier density [--support_min]. Plane test: density ~1-2 on the seen surface, 0 in the
  revealed strip, 15.5% of cells below .15 at the last target. Note from the render dump: the column is first PAINTED by the model in call 5 where the render shows
  a stretched dark mass marked valid; from call 6 it is real content of s-1 -> the mask has to act in call 5.
  Queued (pid 1750183, after the running gmp3_ret_revisit): stride 4 + density (both scenes), stride 2 plain and + density (14 calls, both scenes), render dumps.
  Success criterion: showroom G_old boundary ratio of calls 5 / 7 (1.67 / 1.68) back to ~1.1 and no column. Permanent fix = the same mask function in the
  TRAINING rows (target = real content in invalid cells) -> first concrete item of the 7-day recipe.
- 11 Nov 19:15 critique, reading rules for the density runs: (a) zoom-in false rejects: if the dropped fraction jumps between thresholds .15 and .25 the mask is rejecting SEEN surfaces (density falls when the camera approaches) -> threshold relative to neighbouring cells instead; (b) two questions apart: does the mask remove the stretched content (call 5: dark mass invalid, boundary ratio -> 1.1) vs is what replaces it good (needs training); (c) calls 6-7: does the content generated in the masked strip STAY once it is real content of s-1 (invalid -> generate -> keep); (d) stride 2: sharpness ratio next to fidelity. diag_stride2.py written.
- 11 Nov 19:50 DENSITY MASK + STRIDE 2 RESULTS (diag_stride2.py -> eval_gm/diag/stride2.json; diag_boundary_age; visuals/02267acf_stride4_density.jpg).
  MASK BEHAVIOUR: drops 6.8% of the valid pixels on the showroom at stride 4 (1.4-7.6% per call), 3.9% playground; stride 2: 3.7% / 1.2%. Flat across thresholds
  (.066 at .05 -> .078 at .5; playground .022 -> .068) -> no zoom-in false rejects, the dropped region is a clean zero-density strip. In call 5 the stretched
  dark mass IS marked invalid. So question 1 (does the mask remove the stretched content from the input) = yes.
  WHAT REPLACES IT: the model still paints a grey marbled column + orange object in the invalid strip (call 5 last onward), slightly narrower. So the column is
  not only a copy of the stretched render: it is what GM' generates at the frontier (training rows had exactly such stretched content marked valid there).
  Question 2 = not solved at inference -> training (same mask in the rows, real content as target in invalid cells).
  PERSISTENCE: the column painted in call 5 stays in calls 6-7 (invalid -> generate -> keep works). G_old boundary ratio calls 5/6/7: no mask 1.67/1.07/1.68,
  density 1.31/1.44/1.20 (median 1.16 vs 1.08; A 1.04). Success criterion (1.1 and no column) NOT met. Fidelity unchanged (.164/.285 vs .162/.281; pg .358/.477 vs .353/.455).
  STRIDE (GM', fidelity mean / at 320 / at 448; sharpness at 320 / end):
    showroom:  stride 1 (20 calls to 320) .155 / .271 / -   ; 1.36      | stride 2 (14 calls) .157 / .221 / .271 ; 1.05 / 1.11 | stride 4 (7) .162 / .224 / .281 ; 1.05 / 1.24
    playground: stride 1 (30)             .443 / .596 / .672 ; 2.09 / 3.71 | stride 2 (14)      .400 / .522 / .518 ; 1.18 / 2.46 | stride 4 (7) .353 / .469 / .455 ; 1.04 / 1.79
  -> showroom: stride 2 = stride 4 in fidelity with less hardening at the end; playground: stride 2 sits between, hardening returns by call 14 (2.46).
  Boundary ratio at stride 2 is higher for ALL cells (A 1.39 showroom / 1.21 playground vs 1.08 / 1.06 at stride 4): the call-start step is the same size while
  the in-call change halves.
- 11 Nov 19:55 REVISIT RETRIEVAL on the out-and-back plan, GM', no training (diag_return): showroom all .080 -> .050 (A .079 -> .050, generated .089 -> .051; far gap
  generated .098 -> .056); playground .365 -> .311 (A .353 -> .299, generated .405 -> .352; far gap .470 -> .428). Floors: .014 / .058. -> with the same-pose latent
  in the bank (and as the 4th render source) the UNTRAINED model returns measurably closer to what it generated: supply was a real gap and the model can read it.
  Showroom -37%, playground -15% (still far from its floor). gmp3_ret_mz (Memorizon rule) running.
- Critique 20:00: (1) guidance: confirmed all runs use guidance 1.0 (LoopRoller(..., 1., empty)); text = the scene caption of the ANCHOR view (showroom: "A modern
  retail store displays various wall-mounted appliances ... A white table with chairs ..."), nothing about what lies to the right. Our stages trained with
  text_dropout 0 -> the empty-text branch was never trained by us (test is partly out of distribution). run_arm_ckpt --guidance added; queued showroom stride 4 +
  density mask at guidance 1.5 and 3 (pid 1771036). (2) Boundary ratio is not comparable across strides (denominator halves): ABSOLUTE boundary change, showroom:
  stride 4 A .051 / G_old .083, stride 2 A .032 / G_old .036 (playground stride 2 .079 / .088) -> stride 2 has the smaller absolute step. (3) Constant motion
  instead of constant stride: per call, showroom stride 2 = .048 units / ~2 deg, stride 4 = .096 / ~4 deg (equal fidelity); playground stride 4 = .128 units / ~10 deg
  (better than stride 2 = .064) -> rough target ~.1 gauge units per call; to be validated on other scenes. (4) GM' stride 4 on the three held-out cohort scenes:
  NOT run - the user's standing rule keeps them untouched until the final validation; asked.
- 11 Nov 20:00 USER: "test with guidance AND a summary starting text". encode_summary_text.py: whole-walk prompts (stock caption + what the recording shows beyond the
  anchor: showroom = white wall with radiator panels + low metal-grating platform to the right; playground = hedge, exercise equipment, benches, more buildings),
  local umT5-xxl on the CPU (stock caption reproduces the stored embedding, cos .9998) -> eval_gm/text/<scene>_summary.npz. They LEAK the recording: upper bound
  for "can text steer the frontier", not a fair arm. run_arm_ckpt --text_npz, --guidance. Queue (pid 1775541), showroom, stride 4, density mask, render dumps:
  stock text guidance 1.5 / 3 (running), summary text guidance 1 / 1.5 / 3. Then (critique) out-and-back at stride 4, 14 calls, with and without revisit
  retrieval, both scenes (diag_return takes name=dir:stride now). Caveat: text_dropout was 0 in our stages -> the empty-text branch is untrained by us.
  For the next trainings: text dropout 10-20%.
- 11 Nov 20:20 TEXT + GUIDANCE at the frontier (showroom, stride 4, density mask; visuals/02267acf_stride4_text_guidance.jpg; fidelity mean / end, sharp @320):
  stock caption g1 .164 / .285, 1.07 | summary text g1 .166 / .288, 1.04 | stock g1.5 .287 / .468, 5.40 | summary g1.5 .288 / .468, 5.31 | stock g3 .815 / 1.05 |
  summary g3 .814 / 1.02. Visual: summary text at guidance 1 = the stock run almost frame for frame (the column still appears from call 5, a little paler; no
  white wall / radiators); guidance 1.5 = blown whites, yellow floor, over-contrast everywhere, frontier = a black blob (stock) or a blue/brown panel (summary);
  guidance 3 = colour collapse. => on these weights the TEXT does not steer the frontier (even a prompt that names the content), and guidance is unusable
  (empty-text branch untrained by us: text_dropout 0 for 850 updates). No inference-only lever for the frontier content remains; it needs training:
  (1) density mask in the rows with real content as target in invalid cells, (2) text dropout 10-20% so guidance exists, (3) possibly prompts that describe
  more than the anchor view. Out-and-back at stride 4 (with / without revisit retrieval) running next.
- 11 Nov 20:25 CRITIQUE accepted: (1) the four inference-only levers for the frontier are closed (mask, bank, text, guidance); first generation at the frontier is a
  training matter, and the training signal is abundant (20 deg per call in the corpus: every row has a wide revealed region, so far filled with stretched splats
  marked valid). (2) 7-DAY RECIPE IN TWO GROUPS: A) INPUT CONTRACT, from stage B on: density mask in the rows (target = real content in invalid cells), revisit /
  nearest-pose retrieval rule (same function in training and inference), text dropout 10-20%; gate = frontier quality, first calls, out-and-back consistency.
  B) ROBUSTNESS, later stages: statistics-only corruption, reset pool; gate = melt and hardening on the long chain. Richer prompts only if guidance works.
  (3) Queue reordered (pid 1787548): after the running out-and-back stride-4 pair NOTHING goes before the statistics-only model: gmst3 free (long) -> oracle
  context -> out-and-back -> + moment -> out-and-back + revisit; bank forward tests (gmp3_ret_mz rest, gmp3_mz, gmp3_memory) after it. Released-Memorizon steps removed.
  Reading rules for gmst3: floor = patched GM' (playground 481 DINO .683, sharp .90, regen 1.16); trust tax = oracle-context fidelity < .18 (GM' .141, GM-B' .325).
- 11 Nov 20:45 COMBINED: out-and-back at stride 4 (112 strided frames = 448 source frames out, 14 calls), GM', with / without revisit retrieval (diag_return :4;
  visuals/{0238de14,02267acf}_stride4_return.jpg). Return LPIPS all / anchor-seen / generated, far gap anchor-seen / generated:
  playground: none .338 / .317 / .389, far .370 / .480 | revisit .258 / .248 / .300, far .267 / .368 (floor .085)  -> -24% (far-gap generated -23%)
  showroom:   none .197 / .167 / .233, far .200 / .242 | revisit .116 / .090 / .149, far .072 / .099 (floor .056-.074) -> -41% (far gap -59..-64%, at the floor)
  Visual (playground): without retrieval the structure comes back altered at pose 0 (stair panel replaced by a green slide, other ground layout); with retrieval
  the way back re-joins the original view closely. Sparse generation + retrieval add up; the far-gap showroom numbers reach the measure's floor.
  (Not comparable with the stride-1 out-and-back numbers: 448 vs 240 / 160 source frames out.)
- 11 Nov 20:55 USER: "inspect the visuals too". Zoomed out / back / difference sheets at pose 0 and pose 224 (visuals/*_stride4_return_pose{0,224}.jpg). CORRECTION of 20:45
  ("re-joins the original view closely" was too strong for the playground):
  showroom pose 0: with retrieval the way back is nearly the same image (difference map almost dark); without it the whole display wall is displaced and re-drawn.
  showroom pose 224: with retrieval the content matches but every edge is doubled in the difference map = the way-back view is SHIFTED (the executed camera is
  not at the way-out pose) -> the return LPIPS mixes camera execution error with content change at mid poses.
  playground pose 0: retrieval restores background, ground, bench, path; the play STRUCTURE is still altered - a green slide remains left of the stairs where the
  first view has the teal arch panel, palm and roof displaced. Without retrieval the stairs / rails are a different object altogether.
  playground pose 224: both arms differ strongly on the structure (slide shape, stairs, palm shifted); retrieval only darkens the background difference.
  => retrieval + short chain: solved for the showroom at the start pose, partial on the playground (background yes, main object no). The measure needs a pose
  term (measured camera of out vs back frames) before "content stays" can be claimed from it at mid poses.
- 11 Nov 21:10 CRITIQUE checks (CPU). (1) WAY-OUT halves of the two stride-4 out-and-back runs: fidelity / sharpness identical within noise (playground mean .3513 vs
  .3518, end .455 vs .453, sharp 1.79 vs 1.72; showroom .164 vs .168, end .288 vs .301). Replay of the rule on the way out: no nearest-pose hit at all (the few
  bank entries, distances .29-.55 > .25, come from the stock-rule fill, identical in both runs) -> the two runs had the SAME inputs on the way out; they already
  differ at the 4th decimal in call 1 = GPU nondeterminism, amplified later. The "crumpled slide" is run noise, not an effect of retrieval. The rule already
  returns nothing unless the pose is near (dt <= .25 units, 15 deg, >= 8 latents old).
  (2) CAMERA vs CONTENT (diag_return_align.py -> eval_gm/diag/return_align.json: SIFT + RANSAC homography way-back -> way-out; shift = median inlier displacement in
  px, inliers = geometrically consistent matches, LPIPS after warping). none -> revisit:
    showroom stride 4:   shift 5.8 -> 1.1 px | inliers 262 -> 452 (far gap 250 -> 647) | aligned LPIPS .202 -> .130 (far .195 -> .078)
    playground stride 4: shift 5.3 -> 2.2 px | inliers 43 -> 328                       | aligned .384 -> .292
    showroom stride 1:   shift 1.3 -> 0.9 px | inliers 551 -> 875                      | aligned .092 -> .059
    playground stride 1: shift 8.6 -> 16.8 px | inliers 65 -> 376                      | aligned .356 -> .251 (raw only .397 -> .351: the way back is OFFSET by ~17 px)
    GM-B' (stride 1, no retrieval): showroom shift 14 px, aligned .122 (raw .264: half of its difference is camera); playground shift 55 px, inliers 38.
  => the retrieval gain is mostly CONTENT (consistent matches x1.6-2.6 showroom, x6-8 playground); at stride 4 the pose is re-seated as well (5-6 px -> 1-2 px);
  at stride 1 on the playground the content returns but the camera lands 17 px off, so raw LPIPS under-states the gain. Inlier count = the cleaner
  "same content" number; report it next to LPIPS from now on.
- 11 Nov 21:00 STATISTICS-ONLY MODEL gmst3 (GM' + 250 updates, --corrupt stat), FREE runs: showroom 321: snap 2.64, regen_x 2.36 (GM' 1.04, GM-B' 2.00), DINO .903,
  lum end +2.8 (GM' +21), sharp 1.01; fidelity calls 1-7 / 14-20 .073 / .293 (GM' .060 / .238, GM-B' .067 / .196). Playground fidelity .282 / .642 (GM' .212 / .642).
  Visual (visuals/0238de14_gmst3_481.jpg): no melt, sharp frames, but the scene is RE-DRAWN: a person at 240, another play structure (red / black) at 320, red cars
  and different equipment at 400-480. Statistics and brightness are fixed; the user's criterion fails as with GM-B' - although this corruption never touched
  content. Open: is it the corruption (HF gain .65-1.6 teaches "context detail is unreliable"?) or the 250-update continuation itself (same seeds / same pool
  rows as GM-B'; earlier note: every 250-update continuation of GM raised late LPIPS)? The missing control = GM' + 250 updates with corruption OFF.
- 11 Nov 21:05 gmst3 playground 481: DINO .696 (floor .683), sharp 1.62, regen_x 2.20, snap 5.2, lum end -22.7 -> quality yes, self-consistency no (as GM-B').
  CRITIQUE accepted: (1) control training queued right after gmst3_octx (pid 1808556): runs/gmc3_s1 = GM' + 250 updates, --corrupt off, same seeds and pool rows
  (--seed 20261108, OP), then gmc3 free runs. If regen ~2 again -> second control = continuation WITHOUT pool rows. (2) 7-day group B (robustness training)
  on hold until the cause is known; group A (mask, retrieval rule, text dropout) unaffected. (3) What meets the user's criterion today is the UNTRAINED
  arrangement: GM' weights + stride for ~.1 units per call + nearest-pose retrieval (+ moment patch if needed).
  POOL CONTENT PROBE (diag_pool_content.py -> eval_gm/diag/pool_content.json; latent cosine between the deepest GENERATED context latent and the RECORDED latent
  of the same index, per-channel standardised; yardsticks: recorded latent vs its previous latent .48 median, vs two back .35):
    pool v3, depth 1 call (s=5, n 300): median .722 (p5 .361) ; worse than a one-latent shift 7% ; below .3: 3.7%
    pool v3, depth 2 calls (s=9, n 300): median .590 (p5 .148) ; worse than a one-latent shift 28% ; worse than a two-latent shift 11% ; below .3: 15%
    old pool: depth 1 .662 / 11% / 5.7% ; depth 2 .567 / 34% / 15%.  Correlation with the measured pose gap -.4.
  => a wide tail exists: at depth 2 more than a quarter of the pool rows pair a recorded target with a context that is further from the recording than a
  neighbouring latent, and ~15% with a near-unrelated image (pose error explains part: corr -.4; latent cosine cannot separate shifted from re-drawn content).
  Supports "pool rows teach: do not continue the context"; fix candidate = content acceptance threshold on pool items (cosine >= the one-latent yardstick),
  next to the pose threshold. Hypothesis until the control and a no-pool control answer.
- 11 Nov 21:10 critique notes: (a) magnitude caution: clearly contradicting pool items ~14% of the pool, pool rows ~1/3 of training -> ~5% of all rows; a mild mismatch
  exists in every pool row (median similarity .59-.72); the probe makes the hypothesis plausible, not proven. (b) control ORDER: 1 tonight corruption off + pool on;
  2 if re-drawing persists: pool OFF (separates the pool from the continuation itself: fresh Adam, side modules drifting); 3 only if the pool is the cause:
  content-filtered pool. (c) simplest first filter for step 3: depth-1 items only (s=5) + cosine >= .48 -> half the pool, 93% of it clean.
- 11 Nov 21:20 PROPOSAL (critique): rebuild the sampler with the right contract (speed mix from source videos, retrieval bank, long history, corrected render,
  density mask, text dropout; pool rows undecided) and restart from Memorizon weights (~850 updates). USER asked my view. Facts checked: (1) data/DL3DV-source2K-
  1280x704 is NOT video: it is the same sparse photo set at 2K (398 frames / scene, images/frame_%05d.png). Dense consecutive frames + dense poses exist only for
  the cohort scenes (continuous_eval_20261029/scenes: 10 dirs, 7 with dense_poses.npz), produced by download_videos.py -> prepare_rgb.py (timing match against the
  photos, calibrated undistortion) -> dense_pose.py (fit to the official sparse poses; 63 photos inside the 1009-frame window = one photo per ~16 video frames).
  => slower-than-photo-set training data needs video download + timing + dense pose per training scene: the real cost, unmeasured; existing frames can only be
  made SPARSER. My position: direction right; first measure the dense pipeline on ~10 training scenes; in parallel a contract pilot on existing frames
  (mask + retrieval rule + text dropout) and the sampler distribution report; restart-from-Memorizon decision after the data cost is known. Claims need 2 seeds.
- 11 Nov 21:15 gmst3 TRUST TAX (oracle context, fidelity first / last third): playground .186 / .161 (GM' .124 / .141, GM-B' .174 / .325) -> last third passes the
  < .18 bar, first third worse than GM'; showroom .061 / .123 (GM' .036 / .064, GM-B' .056 / .126) -> as high as GM-B'. Mixed: the statistics-only model
  discounts clean context on the showroom like GM-B', less so on the playground. Control training gmc3 (corruption off) started 21:13.
- 11 Nov 21:45 USER: "look at this plan, decide loss / lr etc., and do the data preparation accordingly". DONE (CPU): data_contract/ = windows.jsonl (67,547 windows:
  motion, bank, revisits, psnr), index_v4.json (4,423 windows; continuation rows in [.05,.15) units 11% -> 33%, >= 13 latents of history 62% -> 76%, true revisit
  -> 32%, contract bank filled 94%), memory_contract.py (one bank function: relative true-revisit test + first-seen owners; trainer --bank contract, roller
  --bank contract, cache builder), build_render_cache_v4.py (index v4, contract bank, meta_density_f16; queued after the gmc3 runs: smoke 4 windows, then all,
  ~1.75 h GPU), distribution_report.py (training draw vs test plans; caught an absolute-threshold bug in the first revisit test: 0 revisits on the test plans),
  RECIPE.md (all settings + reasons). Decisions: loss = FM velocity MSE, unchanged weighting, no masking; lr 2e-5 backbone, 1e-4 side modules in A / B then
  2e-5; warm-up 50; accum 8 x 2 rows; text dropout .15; min_psnr 20; no corruption; A 350 / B 250 / C <= 250 with 50-step checkpoints; no EMA; 2 inference seeds.
  Finding: the photo set HAS slow windows (13% < .15 units per call, 710 streams) and natural revisits (724 streams) -> no video needed for a first contract run;
  the test plans enter the training range only from stride 4 on (>= .13 units per call).
- 11 Nov 22:10 critique: stage D (optional, separate branch, <= 100 updates with 25-step checkpoints, three measures) added to RECIPE.md; long plans built (eval_gm/inputs_long, 1009 poses, run_arm_ckpt --inputs_dir); GM' long-plan reference queued after the cache (pid 1825215).
- 11 Nov 22:00 APPROVED ORDER (critique): stage A tonight from the Memorizon-50 teacher, then cache v4, morning A gate then B. Built: train_a4.py (plain scope,
  index v4 via the matched-window sampler, memory_contract.bank_contract_train, text dropout, warm-up, per-row motion / revisit log), dry_run_a4.py. Dry run of the
  exact stage-A arguments (2800 rows, seed 20261112): no failure; first call 19.9%, >= 13 latents of history 77.5%, translation p10 / p25 / p50 .089 / .13 / .21,
  34% in [.05,.15), bank filled 75.5% of continuation rows (after the 20% deterministic bank dropout; inference 67-83%), true revisit 33%.
  Queue (pid 1830043): gmc3 control (update 80 / 250) -> gmc3 free runs -> cache v4 smoke (4 windows) -> STAGE A (runs/a4_s1, 350 updates, ckpt every 50) ->
  cache v4 -> GM' long-plan reference -> rest.
  Checks: (1) bank fill mismatch 94% vs 67-83% -> 20% bank dropout (hash of the command poses, shared with the cache builder). (2) per-window PSNR filter IS biased
  by motion (sub-20 share 10% -> 36% from slow to > 1 unit) -> stream-level filter in B / C (119 of 888 streams hold 71% of the sub-20 windows). (3) step target =
  range .13-.2 units. render_p .90 in B / C (second critique). RECIPE.md updated.
- 11 Nov 22:20 critique: (1) the 119 flagged streams are excluded from STAGE A too (data_contract/bad_streams_v3.json: >= 2 cached continuation windows and >= half
  below 20 dB; 113 scenes; their median render PSNR 18.6 vs 24.6; 579 of the 4422 index windows) - train_a4 --exclude patches data.corpus864; dry run re-running.
  If the low PSNR came from depth (glass, sky) rather than poses this costs diversity only: accepted. (2) A gate measured at checkpoints 150 / 250 / 350, B starts
  from the checkpoint the two curves pick (not 350 by default): command execution at test speed (stride 4; run_arm_ckpt --no_render for stage-A models, camera
  measured with ViPE as fast_camera.py since the baseline arm does not measure) and probe_bank_swap.py (development windows with a full contract bank: loss with the
  own bank / no bank / another stream's latents in the bank slots, same noise; use and harm should grow with the updates).
- 11 Nov 22:15 USER: evaluation is not the problem; stay alert for FAILURES. failwatch.sh (scratchpad, background): alerts on queue dead / phase failed / 'failed|error|traceback' in the queue log / Traceback, OOM, RuntimeError, nonfinite, NaN in any active train / cache / eval / dry-run log / update loss > .8 (historical max .59) / GPU idle 20 min / disk < 20 GB.
