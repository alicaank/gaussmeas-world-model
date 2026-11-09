# Lit2-C: camera injection, memory-write pose, anti-collapse recipes vs our redesign (2026-10-09)

Scope: local code/papers only. Marks: **V** = read in file at the cited line this session; **Vp** = verified in the 2026-10-08 survey (literature_deep_20261008/{A,B,C}*.md, reused, not re-read); **I** = inferred. Our design = plan_base_restart_20261008/REDESIGN.md + DESIGN.md (A3 camera = ReSplat render in all 30 blocks + zero-init ray fallback, 10% scaffold-dropout; A7 M-cmd vs M-real; B2-B4 foundation 2k + matched 1.5k arms with own-rollout rows and corrupted-history/clean-memory rows).

## 1. Per-system findings

**Lyra 2.0 (Wan2.1-14B)**
- Camera: depth-warped last frame concatenated into the denoising latent + Plücker MLP for uncovered pixels; canonical-coordinate warped maps enter Q/K as positional encoding (Vp, B_memory_trust:13). `cam_encoder` is a trained module (V, configs/experiment.py:62).
- Write pose: generated frames placed at the **commanded** trajectory; depth from DAv3 fine-tuned on 3,000 of its own generated videos, 10k iters (Vp, memory note lines 27-29). No online pose estimation.
- Anti-collapse: self-augmentation = one-step denoise of the **temporal history only** (`self_aug_steps=1`, `self_aug_every_k=2`, `self_aug_prob=1.0`, `self_aug_max_T=500` so t is scaled to <=500/1000 = sigma<=0.5; V, experiment.py:45-53, lyra2_model.py:1223-1262). Spatial (memory) slots stay clean; post-retrieval spatial memory dropout `spatial_memory_dropout_rate=0.1` (drop 1..N selected frames) and whole-cache drop `spatial_memory_drop_rate=0.1` (V, lyra2_model.py:85-94, 2003-2023). `ema=dict(enabled=False)` (V, experiment.py:38). No on-policy rollouts, no DMD (Vp).
- Trainable: `cam_encoder,buffer_encoder,self_attn,clean_patch_embeddings,patch_embedding` (V, experiment.py:62) — i.e. exactly our A5 set; lr 3e-5, wd 1e-3, warm-up 100 (V, experiment.py:68-72); 64 GB200, 7k iters bs64 (Vp). Ablations: w/o self-aug Cam Ctrl 53.9 vs 64.7, Style 78.0 vs 85.1; global fused cloud Cam Ctrl 49.9 vs 63.9 (Vp).

**Memorizon (Wan2.2-TI2V-5B, same base as ours)**
- Camera: PRoPE (cameras as relative positional encoding) through a **zero-initialised side attention branch `prope_attn` added to every block** (V, memorizon/camera.py:1,138,181-189). Camera branch LR 1e-4 vs backbone (V, configs/train_100s.yaml:6); `train_backbone: true` else only `prope_attn` (V, model.py:64). 4 nodes x 8 GPUs, 10k steps, bs 1/device (V, train_100s.yaml:2,29-30); 6k steps 32 H200 in paper (Vp).
- Memory: pose-only retrieval — "computed from camera poses alone, before any frame is generated", top-k 6 per chunk, score minus pose distance (V, retrieval.py:8,89-103; train_100s.yaml:23). Memory is a latent bank keyed by **commanded** poses; no estimator in the loop (V/I).
- Anti-collapse: separate stage = Self-Forcing + DMD 4-step distillation on own rollouts, 1k steps (V, self_forcing.py:1-15, configs/self_forcing_4step.yaml:11,35). Paper admits exposure-bias cost (Vp).

**VMem / SEVA (baselines/vmem, stable-virtual-camera)**
- Camera: SEVA conditions on Plücker rays (V, seva/geometry.py:119-163); input concat + FiLM in every ResBlock, 1.3B full training 700k iters (Vp, C_camera_eval). VMem keeps SEVA frozen-ish and selects context by surfel visibility (I from pipeline structure).
- Write pose: generated frames are appended with **`target_c2ws` = commanded pose** (V, vmem/modeling/pipeline.py:1298-1303) and surfels are built from CUT3R pointmaps (`cut3r_512_dpt_4_64.pth`, V, pipeline.py:86, 1036); CUT3R gives depth/pointmap, not the registration pose. So VMem = commanded pose + estimated depth, same as our M-cmd.

**Wan2.2-Fun-5B-Control-Camera (baselines/VideoX-Fun)**
- Camera: Plücker video, 4 frames stacked -> 24 ch per latent frame (`in_dim_control_adapter=24`, pixel-unshuffle 8) -> `SimpleAdapter` (conv + residual block, **not zero-init**) added once to patch embeddings before all blocks (V, wan_transformer3d.py:28,613-615,735-736; wan_camera_adapter.py:5-27). No camera training script ships for wan2_2_fun locally (V, scripts/ listing lacks it). Full fine-tune implied (I).

**AlayaWorld**: v1.1 removed the dedicated camera AdaLN branch; the planned trajectory only decides where the 3D point cache is re-rendered; no ablation (Vp, C_camera_eval). Hard history dropout rows (Vp, B_memory_trust:41).

**AC3D**: camera side blocks in the first 8 of 32 blocks; camera decided in the first ~10% of denoising, training t in [0.6,1]; all-block conditioning worsens FVD ~10% with no gain (Vp).

**CamCloneMaster**: no camera parameters; reference-motion video tokens frame-concatenated; frame-concat >> channel-concat >> additive ControlNet (Vp).

**Uni3C / ViewCrafter / GEN3C / Voyager** (Vp): render is the camera; Plücker-only ATE 0.402 vs render 0.091 vs both 0.102; Plücker helps only where the render is empty (360 turn 1.33 -> 1.01). GEN3C aligns new MoGe depth to the cache render before writing, 15% condition dropout, noise on renders by buffer age. Voyager trusts own RGB-D, no corruption.

**GeoVerse**: geometric latent built on DA3 geometry head + VGGT-style tokens (V, papers/geoverse.txt:99,176,290); admits accumulation of generated geometry (Vp). Pose for writes not separately estimated (I).

**MoVerse**: panorama -> persistent 3D Gaussian scaffold -> Gaussian-conditioned AR renderer; scaffold renderings along the trajectory are the camera signal (V, papers/moverse.txt:21-23,69-81). Memory is built once from the panorama, not from generated frames (V/I).

**WorldCrafter**: uses VGGT-Omega + DA3 for metric-scale geometry and for *evaluation* trajectories (V, papers/worldcrafter.txt:424-427,493-505); camera via rendered history; memory from generated frames at given poses (I).

**GEAR** (Vp): DA3 meshes -> token correspondences, Invisible Octree veto; degraded-history augmentation on 40% of samples (tau~U(0,0.3), one reverse-flow step); LoRA r32, 10k it, 32 GPUs; w/o degraded history FVD 1166 vs 838.

**Robust Dreamer** (Vp): latent Gaussian memory from generated latents at commanded poses; Deviation Archive of one-step deviations indexed by AR index x timestep, 500 slots, l2-dedup, injected into GT history **before** Gaussian construction p=0.9, clean-input rows 0.5, clean target; LoRA r128 on Wan2.1-I2V-14B, 8 H200 ~31 h/stage. ScanNet 80f SSIM: full 0.651 / clean memory 0.268 / Gaussian noise 0.275.

**WorldPlay / WorldPlay2** (Vp): PRoPE camera; memory context noise sampled in [0,0.2], targets [0,1]; Context Forcing = student rolls 4 chunks, bidirectional teacher with memory-aligned context, DMD; long-term PSNR 18.94 vs 16.27. WorldPlay2 Stable Forcing: PDD 4-NFE init, full-rollout replay with one cached intermediate step per chunk, per-clip DMD over 320-latent rollouts; RevisitBench LPIPS 0.318 vs 0.416.

**Wonder** (Vp): camera = VAE-encoded synthetic lattice + env-map render channel-concat; teacher extended 5->10->20 s; ODE init with sparse context forcing; on-policy DMD; GAN regulariser on low-frequency frozen-teacher features (camera drift).

**Causal Forcing / LongLive / Matrix-Game 2.0 / Self-Forcing(++)** (Vp): `context_noise: 0` in shipped configs; seamlessness learned via DMD from a bidirectional teacher on whole self-rollouts; LongLive LoRA r256, 64 H100 ~12 h, overlap 0-3 latents, sink 3; SF++ backward-noise init; Rolling Forcing window [1000,800,600,400,200] with 50/50 alternation, 3000 steps; LongLive: "attention sinks become effective once long-rollout collapse is addressed".

## 2. Table

| System | Camera signal / where / trainable | Write pose (est.?) | Anti-collapse | Budget |
|---|---|---|---|---|
| Lyra 2 | warped last frame concat + Plücker MLP fallback; canonical-coord PE in Q/K; self_attn+cam_encoder+patch_emb trained | commanded; DAv3 fine-tuned on own videos | one-step self-aug on temporal history only, every_k=2, t<=0.5; mem drop 0.1; no EMA | 64 GB200, 7k it, bs64, lr 3e-5 |
| Memorizon (5B) | PRoPE zero-init side attn in every block, lr 1e-4 | commanded (pose-only retrieval) | separate Self-Forcing+DMD 4-step, 1k steps | 32 GPUs, 10k steps |
| VMem/SEVA | Plücker concat + FiLM, SEVA full-trained 700k | commanded `target_c2ws`; CUT3R depth only | none (training-free memory) | — |
| Fun-5B-Camera | Plücker 24ch -> SimpleAdapter (not zero-init) at patch-embed once | n/a | n/a | not shipped |
| AlayaWorld 1.1 | render only; camera branch removed | commanded (point cache) | hard history dropout | — |
| Uni3C | render + Plücker side DiT; Plücker helps only empty render | commanded | 15% cond dropout (GEN3C) | 6k steps 64 H100 |
| GEAR | correspondences only | commanded + DA3 | 40% degraded history tau~U(0,0.3), LoRA r32 | 10k it 32 GPUs |
| Robust Dreamer | render (latent GS) | commanded | deviation archive p=0.9 into memory input, LoRA r128 | 8 H200 ~31 h/stage |
| WorldPlay(2) | PRoPE | commanded | context noise [0,0.2]; Context/Stable Forcing DMD | large (A14B teacher) |
| Wonder | lattice render concat | commanded | ODE init + on-policy DMD + GAN low-freq | large |
| LongLive/SF++ | n/a | n/a | DMD on self-rollouts, context_noise 0, overlap 0-3, sink 3, LoRA r256 | 64 H100 12 h |

## 3. Verdict

**(a) Render-as-camera-channel + ray fallback: standard, not novel.** It is the dominant pattern (Uni3C, GEN3C, Voyager, AlayaWorld 1.1, MoVerse, Robust Dreamer, Lyra 2's warped-frame + Plücker-for-holes is literally our A3). Our differences are implementation-level: ReSplat (vs point/pixel warp), 30-block residual (AC3D/CameraCtrl-II say early-block or once-at-patch-embed is enough; our g=0 ray finding is consistent with the ray route being redundant when the render is present), and the reliability map r feeding init. Claim nothing about the camera route; claim r.

**(b) Memory at an online-estimated pose: nobody does it.** VMem (`target_c2ws`, pipeline.py:1298-1303), Lyra 2, Robust Dreamer, GEAR, Memorizon, WorldPlay all write at the commanded pose; estimators (CUT3R, DA3, MoGe, VGGT) supply depth/scale or evaluation trajectories, never the registration pose. GEN3C's cache-render depth alignment is the closest "realized" step (geometry aligned to what the cache already says). So M-real is novel, with the known risk from our own 2026-09 probes (observer noise, 4x weaker anchor); it must be framed as a trained policy with the observer in the loop at train time, as A7 says, and compared against the GEN3C-style write-time alignment as the cheap control.

**(c) Strongest evidence per unit compute without DMD: asymmetric structured corruption of the *history* with clean memory, plus structured (own-deviation) corruption of the memory input.** Evidence: Lyra 2 self-aug (one forward step per every-2 updates; Cam Ctrl +10.8, Style +7.1), GEAR 40% degraded history (FVD 1166 -> 838, LoRA), Robust Dreamer archive (SSIM 0.268 -> 0.651, and Gaussian noise does *nothing*: 0.275). All three are augmentation-only, single-forward-step, LoRA/partial-set, and all worked on 8-32 GPUs with 7-10k steps. DMD/Context Forcing (WorldPlay, Wonder, LongLive) gives the best seams but needs teacher + fake-score + 64-GPU rollouts. Our own-rollout rows (recorded future as target) are the cheap on-policy analogue; Self-Forcing++ warns the target distribution must be the teacher's, and our 2026-10-02 FL own-rollout GT-regression failed for exactly that reason — keep own-rollout rows validity-filtered and <=1-5 calls, as planned.

**(d) Numbers to adopt**
1. History corruption: one-step regeneration of the temporal history at t~U(0,0.5) (`self_aug_max_T=500`), applied to ~50% of rows (`every_k=2`, prob 1.0), memory/scaffold clean, target clean (Lyra 2, experiment.py:45-53). Add GEAR-style 40% share if we prefer the lighter tau~U(0,0.3).
2. Memory-side corruption: own one-step deviations (not Gaussian noise) injected into endpoint frames before ReSplat on p=0.9 of corruption-family rows, with 0.5 of rows clean-input (Robust Dreamer); drop the whole cache with p=0.1 and 1..N sources with p=0.1 (Lyra 2, lyra2_model.py:85-94) — this replaces our flat 10% scaffold-dropout with a per-source variant.
3. Optimiser: backbone/self-attn lr 3e-5, wd 1e-3, 100-step warm-up, **no EMA** (Lyra 2, experiment.py:38,68-72); new modules (r head, ray projection) at 1e-4 (Memorizon camera branch, train_100s.yaml:6). History context noise at inference/training sigma in [0,0.2] only if we keep t=0 history (WorldPlay); shipped Self-Forcing configs use 0.

Caveats: five deeper code reads (AlayaWorld/CamCloneMaster/MoVerse injection points, Fun training recipe, Lyra config exhaustive) were dispatched but not awaited; Vp items were not re-read.
