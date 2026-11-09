# lit2-B: Memory reliability, gating, source selection and geometric addressing vs our r(p) (2026-10-09)

V = verified in local code/paper text this pass; I = inferred; PS = previously surveyed only (literature_deep_20261008/B_memory_trust.md), not re-verifiable locally.
Our proposal (REDESIGN.md §2): one learned per-patch reliability map r(p) from render features + render-vs-history disagreement + rays, supervised by r*(p)=1[‖z_render−z*‖<ε], used for (a) per-patch init noise σ_init(p) and (b) scaffold-residual multiplier + attention logit bias +log r toward memory-source tokens, trained with 5 counterfactual families.

## Per-system findings

**Lyra 2.0 (V, code).** Carrier: native VAE latents of N_s=5 retrieved frames as "spatial slots" (4 at 2× subsample + 1 full), joint self-attention (lyra2.txt:405-411). Addressing: per-slot canonical coords (u,v,slot-z) + normalized warped depth, forward-warped with the slot's depth, pixelshuffled to 4·64 ch/slot (lyra2_model.py:48, 1733-1800), MLP-encoded and added to Q/K only as `kq_bias` in every block (wan2pt1_lyra2.py:175-196). Selection: `Sparse3DCache.retrieve` projects every cached frame's world points into the target views, z-buffers per pixel (depth_threshold 0.1), then greedy max-coverage at inference with the newest frame pre-covered (lyra2_model.py:2764-2815); training samples slots multinomially ∝ coverage score (2821-2832). Reliability: the only "validity" channel is a 1/0 slot-presence flag that zeroes the buffer embedding for padded slots (wan2pt1_lyra2.py:177-184, 734-740, 795-800); warp holes are simply zero coordinates; no confidence, no age, no agreement. Corruption: **correction to the old survey** — spatial slots ARE noised: `apply_corruption_to_spatial_region="noise_with_sigma"`, log-normal σ with p_mean=−3, p_std=2 (median σ≈0.05) (configs/experiment.py:41-44; lyra2_model.py:135-147, 392-401); temporal history gets 1-step self-aug (self_aug_steps=1, every_k=2, p=1.0; experiment.py:45-52); 10% rows drop 1..N selected slots (lyra2_model.py:2003-2025). Geometry (depth, correspondence bias) is never corrupted: wrong depth → wrong Q/K bias with no defense. Write: cache stores per-frame depth/w2c/K, no alignment or rejection (2640-2646); `spatial_memory_skip_recent=16`.

**VMem (V, baselines/vmem/modeling/pipeline.py).** Carrier: past frame latents (SEVA backbone) + Plücker. Addressing = **selection only**: surfels rendered at the *average* target pose with z-buffer and cos-to-normal (228-407); each hit surfel votes cos/(1+depth) for every timestep that observed it (`surfel_to_timestep`, 462-502); votes → frame counts → candidate list, then pose-distance NMS (700-758); the chosen latents enter the model as ordinary context with `cond_aug=0` (1260-1290). Nothing surfel-derived reaches attention. Write: `merge_surfels` attaches the new timestep to an existing surfel if within position threshold and normal·normal>0.7, else appends (775-830); CUT3R confidence is computed (extern/CUT3R/surfel_inference.py:203-242) but not used to gate. Only "prune" is undo of removed frames (1365-1400). So VMem is the closest *visibility-based source selection*, but it is frame-level, fixed, and never per-patch.

**SEVA (V, code).** Nearest-source by camera translation distance (seva/eval.py:492-500), `cond_aug=0.0` (1099), no confidence. Training-time input-view noising = PS.

**WorldMem (V, code).** Carrier: past frame latents in a dedicated memory-attention block per DiT block with adaLN gate (dit.py:265-319), Plücker + relative frame-index embeddings on Q/K. Selection: FOV overlap of 10k sphere points, `confidence = overlap_ratio − 0.2·age`, greedy with occupied-region removal (df_video.py:591-626) — fixed, frame-level. Memory held at fixed `stabilization_level` noise (456). No per-patch signal.

**LongLive 2.0 (V, code).** KV-cache sink + local window; `shot_clean_recache` zeroes KV at scene cuts (pipeline/causal_diffusion_inference.py:106, 632-633, 1036). LongLive-RAG is not local (PS). No reliability.

**Memorizon (V, code+paper).** Carrier: clean past latents in a bank, read in place via per-chunk attention mask under a shared positional index (layout.py:7-19; pipeline.py:139-144). Selection: `score = fov_overlap − 0.2·(‖dt‖+4·angle)`, top-K (retrieval.py:3-6, 100-104); fixed. Corruption: self-forcing/DMD on own rollouts (self_forcing.py:157-219); bank never corrupted; paper admits exposure bias (memorizon.txt:347, 667). Bank-replacement test (−83% revisit correlation, memorizon.txt:33) is a good read-out probe to copy.

**GEAR (V, paper; repo has no code).** Carrier: history latents + per-frame DA3 local meshes → face-ID patch correspondences; Invisible Octree accumulates visibility evidence and rejects projectable-but-occluded matches (gear.txt:171-172, 344-390, 431); GCA injects matched history features as a residual into target tokens (331, 431). Corruption: degraded history ẑ_H = sg[z_τ − τ·v_θ], τ~U(0,0.3), p=40% from 8k iters (483-500, 576-579); geometry never corrupted. Ablations: w/o degraded FVD 1166 vs 838; w/o octree 878; "w/ Corres. Disturbance" 842 (no gain) (635-646). Closest *binary* per-patch visibility gate on addressing — fixed, geometric, no appearance agreement.

**Robust Dreamer (V, paper).** Carrier: latent Gaussians; recall = alpha-composited splat of stored latents (Eq.7, lines 350-372); opacity is a learned Gaussian attribute, not a reliability estimate. Corruption: X̃_i = X_i + I_i·D_i, I_i~Bernoulli(p), injected into history **before** memory construction, anchor clean, target clean (Eq.2-4, 280-300); archive Z=500, ℓ2 dedup (991, 1103). Memory and predecessor are corrupted jointly (same deviations), never independently; no reliability read-out.

**GeoVerse (V, paper).** Confidence-weighted point set, generated depth scale-aligned, mask V_j = valid point support ∧ z-buffer visibility (geoverse.txt:373, 439-445); confidence is a fixed DA3-style channel; admits drift accumulates through memory updates (655-656).

**MoVerse (V, code+paper).** Panoramic 3DGS scaffold built once, never written (moverse.txt:21-23, 148); renderer = causal KV cache with t=0 re-cache of clean context (pipeline/causal_inference.py:299-327). No reliability.

**AlayaWorld v1.1 (V, code+paper).** Carrier: ViGeo point cache re-rendered along the plan, causally VAE-encoded with a real prefix (alaya.txt:62-70); tokens of invalid render regions are **removed** (70-71; inference/rollout_utils.py:100). Camera AdaLN removed (93-96). **New finding:** training corrupts the spatial (render) latent *and* the nearby latent with the same Helios corruptor (noise p=0.9 / downsample, corrupt_ratio 1/3, saturation) (trainer/rollout_trainer.py:715-737; fastvideo/rollout/drift_simulator.py `corrupt_history_latents_helios`) plus an ErrorBank deviation injection on the spatial latent (rollout_trainer.py:684-697); modes clean/drop/corrupt (5752-5770). Loss is up-weighted ×2 on *invalid* (uncovered) patches (862-872) — a fixed per-patch validity used in the loss, the opposite direction from r. Render-vs-target agreement is never computed.

**WorldPlay2 (V, paper).** TinyHistory-style 3D-conv compressor → memory tokens, teacher forcing with causal mask (worldplay2.txt:263-310), Stable Forcing; no geometric addressing, no reliability.

**WorldWeave (local paper, V).** Procedural metric world with validated/atomic commit of candidate additions (worldweave.txt:106-116, 184-193) — a symbolic write gate, not latent. The "re-noise RGB at t=0.8" claim from the old survey belongs to a different paper (PS).

**Curious (V, paper).** Persistent 3DGS forward model; reward = discrepancy between the render at the next pose and the observed frame (curious.txt:108-109, 125-126, 160-164). This is literally render-vs-observation disagreement as a learning signal, but per-frame, for an RL policy, not for generation.

**MeRoPE (V, paper).** Ray-local MinRot RoPE for history tokens (merope.txt:257-275); current-image dropout and lateral-conflict donor training (1232-1240); retrieved tokens tagged +100 in time (1242). Addressing is positional, no reliability.

**GEN3C (V, local /home/hj/Alican/GEN3C).** `update_cache` aligns new depth to the cache render, rigid or ARAP (cache_3d.py:246-274; camera_utils.py:273-312); edge filter `reliable_depth_mask_range_batch` ratio<0.05 (forward_warp_utils_pytorch.py:338-351; cache_3d.py:125, 296); `Cache3D_BufferSelector` top-K buffers by rendered-mask overlap (346-393); fixed `noise_aug_strength` on warped frames (gen3c_single_image.py:82-85). Age-scaled render noise: PS.

**ChronoWorld prune, TetherCache, Echo-Memory: PS only.**

**Our own prior (V, analysis/reliability_mask_20261006/README.md):** fixed conf<0.15 opacity mask on ReSplat sources: LPIPS 0.484→0.462 on 8 held-out scenes, late DINO unchanged, KID worse. This is already a B1-class fixed-threshold baseline we must beat.

## Axis × system

| System | Carrier | Addressing | Reliability signal | Memory corruption in training | Write gate |
|---|---|---|---|---|---|
| Lyra 2 | native latents | canonical-coord+depth Q/K bias; coverage-greedy select | slot-presence flag only | latent noise σ~LN(−3,2) on slots; 1-step self-aug on history; geometry clean | none |
| VMem | latents | surfel-visibility vote → frame selection | none (CUT3R conf unused) | none | position/normal merge |
| SEVA | latents | nearest camera | none | none (PS: suggested) | n/a |
| WorldMem | latents, mem-attn | FOV overlap − 0.2·age | fixed age term | fixed noise level on memory; 10% dropout | none |
| Memorizon | latents, bank | FOV overlap − pose dist | none | self-forcing only | none |
| GEAR | latents + mesh | face-ID correspondence + Invisible Octree | binary visibility | 1-step degraded history 40%; geometry clean | none |
| Robust Dreamer | latent Gaussians | splat (alpha) | opacity (learned, not reliability) | archive deviations, joint history+memory | voxel prune |
| GeoVerse | conf-weighted points | render + visibility mask | fixed conf | none | scale-align |
| AlayaWorld 1.1 | point-cache render | render tokens; invalid removed | binary validity; loss ×2 on invalid | Helios noise/downsample/saturation + ErrorBank on render AND history | scale-align |
| GEN3C | per-view clouds | render; top-K by mask overlap | edge mask, validity | none | depth align to cache render |
| MoVerse / WorldPlay2 / LongLive | scaffold / compressed / KV | fixed / positional | none | none / stable forcing | none |

## Verdict
"Learned per-patch reliability, supervised by render-vs-target agreement, used for both initial noise and addressing" exists nowhere in the surveyed set. Closest pieces: GEAR's Invisible Octree (fixed binary visibility on correspondence addressing); VMem's surfel-vote (fixed, frame-level selection, not attention); AlayaWorld's per-patch validity used as a loss weight and token removal (fixed, geometric, inverse direction); Robust Dreamer's deviation training (joint corruption, no read-out); Curious's render-vs-observation discrepancy (as RL reward). Nobody supervises a reliability estimator against the target, nobody schedules σ_init per patch, and nobody combines appearance disagreement with geometric validity. The novelty claim stands; it is narrow, so the control (B1 fixed-σ from confidence; our own conf<0.15 mask) must be reported head-to-head.

## Three strongest objections
1. **"r is just a confidence/validity relabel."** Every system with a signal uses a fixed one (GEN3C edge mask, GeoVerse conf, Alaya validity, our conf<0.15 mask). Our 5 families answer this only if the *memory-damaged* family keeps DA3 confidence high while depth is wrong (REDESIGN §5 D2 "conf korunarak depth sapması") so r*=0 at high confidence; report AUC(r) vs AUC(conf) per family, not pooled. Not answered: whether r generalises to deployment errors that the D2 corruption catalogue does not contain (Alaya's Helios corruptor and Robust Dreamer's archive both argue for self-generated deviations, which our families include only via the S1 one-step student).
2. **"r*=1[‖z_render−z*‖<ε] is a per-latent-patch threshold and the VAE latent norm is not perceptual; ε picks the answer."** Families share target and flow noise, so ε is at least constant across arms, but reviewers will ask for ε sweeps and a soft target (e.g. sigmoid of distance). Our data answers the first (same ε everywhere), not the second.
3. **"Logit bias +log r toward visible sources ignores occlusion/motion exactly where GEAR's octree matters; and Echo-Memory shows revisit gains can be regularisation."** The memory-missing family (10%) and the Memorizon-style bank-replacement probe (swap sources from another scene, expect collapse) answer the second; the first is unanswered — our visibility is render-derived (ReSplat z-buffer) and gets no occlusion accumulation over time.

## Two details to copy
1. **GEAR degraded history as one sg'd reverse-flow step, τ~U(0,0.3), p=0.4** (gear.txt:483-500, 576-579) — the same recipe as Lyra self-aug (lyra2_model.py:1233-1262) but with a published ablation; use it for the history-damaged family instead of inventing a new corruptor.
2. **AlayaWorld's symmetric corruptor on the render latent with per-latent-frame independent σ and keep_x0 control** (repos/alaya/alaya/trainer/rollout_trainer.py:715-737; fastvideo/rollout/drift_simulator.py) plus the invalid-patch loss weighting (rollout_trainer.py:862-872) — reuse the function for the memory-damaged family's appearance half (noise/downsample/saturation), and log loss split valid/invalid so r's effect on covered vs uncovered patches is visible from step 0.

Also lift Lyra's coverage-greedy retrieval with newest-frame pre-cover (lyra2_model.py:2764-2815) for the ≤2-source selection.
