# Idea index: how the Gaussian memory and the camera signal are used (status as of 9 Nov 2026)

Documents in `docs/ideas/`, code in `src/ideas/`. Status words: **in the method** (part of Gauss-meas / GM-B / GM-F),
**tested, kept as component**, **tested, no win**, **built, not run**, **proposal**.

## Gaussian memory

| Idea | Where | Status |
|---|---|---|
| Measured-pose (realized) writes: memory labels are the DA3-measured cameras, not the commanded ones | `src/closed_loop.py`, `third_party_snapshot/realized_pose.py`, `docs/ideas/lit2_realized_pose_writes.md`, `PLAN_earned_memory_20261006.md` | in the method (on-policy rows of `run_op/meas`) |
| On-policy rows with measured history and recorded targets (train on the model's own 1–2-call rollouts) | `src/train_onpolicy.py`, `src/onpolicy_rows.py`, `src/gen_onpolicy.py` | in the method (the Gauss-meas step) |
| Context corruption with the model's own one-step x0 (GM-B) | `src/gm_b_corruption.py`, `src/train_gm_b.py` | in the method line; 7/9 dev metrics over GM (camera), LPIPS not |
| Frontier mask: render evidence only where a source actually saw the surface; generate elsewhere (GM-F) | `src/frontier.py`, `src/build_frontier_cache.py`, `src/train_gm_f.py` | trained; frustum-only inference test chosen 9 Nov, evaluation running |
| Per-chunk render groups (every target chunk gets the render of its own command, "R2") | `src/ideas/train_r2.py`, `train_r2op.py`, `precompute_renders_next.py` | tested 10-26..11-01, no win |
| Gaussian-feature memory (render metadata widened to depth / conflict / support) | `src/ideas/train_feat.py`, `renderer864.render_features` | tested, FAIL (1 Nov) |
| 2Xplat global re-reconstruction at measured poses + visibility bank + map-addressed attention bias | `src/ideas/twoxplat_memory.py`, `train_g.py`, `rows_g.py`, `gen_onpolicy2.py`, `docs/ideas/PILOT_PLAN_CVPR_20261104.md` | tested 4–5 Nov, stopped: loses on generated history |
| Earned memory: evidence channel, write-only-new, partitioned loss, loop closure (M1–M5) | `docs/ideas/PLAN_earned_memory_20261006.md` | proposal; M1/M2 partially realised by the realized-pose writes and the frontier mask |
| Selective visual memory: keep appearance of a view whose geometry is unreliable (omit / appearance-only / full) | `docs/ideas/CVPR_RESEARCH_PROPOSAL.md`, `cvpr_proposal/main.tex`, `IDEAS_v3.md` A/B/E/I/L | proposal (10 Oct) |
| Correspondence-transfer camera supervision through reliable memory correspondences, memory admission | `docs/ideas/CVPR_novel_ideas_memory_correspondence.md` | proposal with gates 1–3 |
| Learned two-view refiner (undo a controlled geometric error given the previous reliable frame) | `src/ideas/refiner_pilot.py`, `src/ideas/memorizon_refl/train_refiner.py` | tested on released Memorizon: blur; discriminator is not a sharpness signal (Oct 13–16) |

## Camera signal

| Idea | Where | Status |
|---|---|---|
| F′ closed loop: measure the realized pose after every chunk, offset-rejoin the plan, early x0 camera check | `src/closed_loop.py` (`replan_mode offset`), `IDEAS_v3.md` F, `lit2_C_camera_collapse_recipes.md` | in the method; wins 6/6 on released Memorizon (13 Oct) and is the "+ correction"/F′ arm of the tables |
| Memory-verified camera reward (RVM / DiffusionNFT reward-weighted velocity matching; rewards = DA3 realized pose vs command in the anchor gauge, LPIPS to recording, masked L1 to the model's own ReSplat render, sharpness penalty) | `src/ideas/reward_train.py`, `IDEAS_v2.md` Idea 2, `lit2_C_camera_collapse_recipes.md` (CamVerse / WorldCompass / WorldCycle / RVM table) | built with smoke mode, not run at scale (GPU went to the robustness line) |
| Camera-ReFL on the released 4-step Memorizon with a latent pose head as reward | `src/ideas/memorizon_refl/refl_camera.py`, `adapt_pose_head.py`, `diagnose_losses.py` | pose head failed its gate on the student's own latents (Spearman .14); ReFL never ran |
| Does the model read the camera? Teacher-forced loss with true vs still vs reversed commands | `src/ideas/probe_camera_loss.py` | diagnostic, used 10-22 |
| Self-predicted camera-execution error, scaffold-free warm-up, counterfactual render-alignment supervision | `IDEAS_v3.md` C/D/M, `DESIGN.md` | proposals |
| Camera-command injection into the released student (finetune line) | `src/ideas/memorizon_refl/inject.py`, `train_single.py`, `gv_memorizon2.py` | tested Oct 15–16 (Memorizon plug-in test), superseded by the fresh teacher |

Literature positioning: `docs/ideas/RELATED_METHODS.md`, `SOTA_COMPARISON.md`, `lit2_*.md`, `CVPR_STRATEGY_2026-09-08.md`.

## Reading the Gaussian world beyond the RGB render, and trusting geometry

| Idea | Where | Status |
|---|---|---|
| Trust channels injected with the render: (valid, conf) metadata per render token, invalid tokens masked from attention; frontier mask sets trust = 0 where no source saw the surface | `src/packed_model.py` (render slots), `src/render864_rows.py`, `src/frontier.py` | in the method (GM: alpha-based; GM-F: frustum-observed) |
| Geometric support channels (fused depth, cross-source conflict, source-coverage count) added to the render metadata | `src/ideas/train_feat.py`, `src/ideas/precompute_feat_864.py`, `renderer864.render_features` | tested (run_op/feat), FAIL 1 Nov |
| Structural read + original-source appearance read: a restricted cross-attention from noisy target tokens to the source VAE features that geometry says are visible at the same surface (zero added read when unsupported) | `docs/ideas/memorizon_gaussian_design_20261104/METHOD_AND_TRAINING.md` §4.1–4.2, `RESEARCH_AND_DESIGN.md` "beyond an RGB render", `DESIGN_AFTER_DRIFT_TESTS_20261105.md` "Restricted source reader" | design 4–5 Nov; implemented in `src/ideas/packed_model_corr.py` (`mode='restricted'`: zero-output PRoPE readers every second block over geometry-restricted keys, dense masks), run as arm G+B of the 4–5 Nov pilot, not qualified on ReSplat correspondences |
| Map-addressed attention bias (`gb`): correspondence matrix from the global 2Xplat map added to the attention logits with a zero-init per-block gain, so the generator learns how much to trust geometry-addressed source tokens | `src/ideas/packed_model_corr.py` (bias mode), `src/ideas/twoxplat_memory.py`, `src/ideas/train_g.py`, `src/ideas/rows_g.py`, `PILOT_PLAN_CVPR_20261104.md` | arms G / G+B run 4–5 Nov; 2Xplat line stopped (loses on generated history) |
| Feedback-conditioned read: encode log(M⁻¹P), log(M⁻¹U) (tracking and execution error in the calibrated gauge) into a zero-init modulation of the memory/camera adapter | `docs/ideas/memorizon_gaussian_design_20261104/FEEDBACK_CONDITIONING_20261105.md`, `METHOD_AND_TRAINING.md` §4.3 | proposal, unimplemented |
| Zero-gated host-weight archive reader (GATE-Ω): archive source tokens pass through the host blocks and are read by a tanh-gated joint attention at blocks 7/15/23, with paired-target binding loss | `docs/ideas/gaussverse_2026-07..09/GATE_OMEGA_CONTRACT.md` §3.5, `third_party_snapshot/gaussverse_memory/gate_omega_memory.py`, `docs/lab_notes/gaussverse_experiments_2026-07..09.md` | trained Aug–Sep (earlier Wan-based line): reader gates stayed ≈ 0, memory redundant at trained depths (postmortem 1 Sep) |
| Trusted-history interface C1/C2: route reliable projected RGB through the trusted history slots, or read original-source features through a small adapter keeping source pose / time / visibility explicit | `docs/ideas/research_plan_20260925/PLAN.md` Phase C/D | plan (Sep); C1 became the render slots of the fresh teacher |
| Persistent source Gaussians / latent splat / point retrieval (the earlier GaussVerse memory backends) | `third_party_snapshot/gaussverse_memory/` (`persistent_gaussians.py`, `latent_splat.py`, `point_retrieval.py`, `zipsplat_target_rescue.py`), `docs/ideas/gaussverse_2026-07..09/persistent_gaussians.md` | superseded by per-call ReSplat (ReSplat scaffold probe 5 Sep: .343 vs ZipSplat .448 LPIPS) |
| "Trust geometry, not appearance" writes (WorldWeaver), deviation archive on the scaffold input (Robust Dreamer), asymmetric history corruption; geometry-forcing / WorldForge camera-error supervision | `docs/ideas/literature_deep_20261008/B_memory_trust.md`, `D_geometry_forcing_worldforge.md`, `docs/ideas/gaussverse_2026-07..09/RECOMMENDATION_camera_error_supervision_20260929.md` | literature; the Robust-Dreamer-style deviation on the memory input became GM-B's self-augmentation |
| Memory use as an intervention (counterfactual memory: read/write actions chosen by causal effect on generation) | `docs/ideas/gaussverse_2026-07..09/PLAN_COUNTERFACTUAL_MEMORY.md` | Jul–Aug thesis framing; Gate 2 oracle gap ~1 % for warped-RGB memory |
