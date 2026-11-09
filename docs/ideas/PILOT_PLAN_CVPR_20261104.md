# Measured global Gaussian memory on Memorizon: CVPR-2027 pilot plan (4 Nov 2026, evening)

User decisions 4 Nov: target CVPR 2027 anyway (deadline 16 Nov; only a pilot fits); scope = narrowed design + source-feature
read; init Memorizon-50; reconstruct with 2Xplat and take measured cameras from it; write memory at measured poses in training
and inference; keep camera correction in the loop. This file turns RESEARCH_AND_DESIGN.md into what one RTX 5090 (32 GB) can
train and evaluate before 16 Nov. The learned cause-sensitive map updater (its stage 1) is deferred.

## Method (what is trained)

State per rollout: anchor + accepted keyframes (latent ends, one RGB frame each), three camera records per frame: planned P,
issued U, measured M. Memory and labels use M only.

1. **Global re-reconstruction each call (frozen 2Xplat v2, DA3-Giant geometry expert).** All accepted keyframes (anchor + up to
   31 generated ends; farthest-point thinning by camera centre when more) go through 2Xplat as context views in one pass.
   Its DA3 pass returns raw poses + intrinsics + depth for every view (= the measured poses of the generated frames; the separate
   DA3 observer is dropped), its appearance expert returns one global Gaussian scene. Registration to the anchor gauge:
   Sim(3) by Umeyama on the camera centres of the frames already accepted (anchor = identity, gauge median depth 1); at call 1
   only the anchor exists, so the scale comes from the DA3 depth of the anchor (median ratio), as the current observer does.
   2Xplat's own normalisation (Gram-Schmidt canonical frame + max-|t| scale) is an internal frame; planned cameras are mapped
   gauge -> DA3-raw -> normalised before `_render_target_views`. Context-only inference is supported by the released code
   (`_run_da3_separate` with zero targets); EPA / pose_optimization is never used.
2. **Reads.** (a) Structural read, unchanged path: RGB render + (valid, conf) at the 16 planned cameras in the four render slots.
   (b) Visibility-addressed bank selection: `select_bank`'s greedy frustum coverage is replaced by coverage of the rendered
   target depth (the surfaces the target will actually see), computed from the global map; same max_bank 4, no new parameters.
   (c) Source-feature read = map-addressed attention bias: for every target token the map gives the bank/render token that sees
   the same surface (rendered target depth -> unproject -> project into each bank keyframe -> token index). A per-block scalar
   gain (zero-init) multiplies a sparse correspondence matrix added to the existing dense `attn_logit_bias` (packed_model.py
   already feeds a (1,1,S,S) additive bias to every block). Zero gain = the current model exactly; the generator learns how much
   to trust geometry-addressed source tokens. Occluded or uncovered target tokens get no entry.
3. **Training = measured on-policy rounds.** Pool: the current model rolls out train streams with the full loop (2Xplat memory
   at measured poses, measured bank/labels, rejoin correction); rows = own latents + measured labels + global render of the
   recorded next command from own frames + recorded target chunk (the recorded target expressed from the measured state IS the
   corrected command, so correction is implicit in every pool row; the loop only changes which states appear). Mixed with
   recorded rows as in train_onpolicy (q = .5, shared op_rng). +250 updates, Memorizon settings, from run_op/meas.

Claim to test (pilot): training on the states a measured-pose, globally re-reconstructed memory produces, with a
geometry-addressed read of the source tokens, improves long-horizon camera execution and content reuse over the same training
with per-call ReSplat memory (run_op/meas) and over the open-loop recipe. Not claimed: first Gaussian memory, first measured
writes, first correspondence attention.

## Arms (matched rows / seed / windows / budget)

| arm | memory at pool + train + eval | bank selection | attention bias | init |
|---|---|---|---|---|
| meas (exists) | ReSplat <=4 sources, measured poses | frustum | none | run_v864/mem |
| G | 2Xplat global, measured | visibility | none | run_op/meas |
| G+B | 2Xplat global, measured | visibility | map-addressed, zero-init | run_op/meas |
| R2meas (control, if time) | ReSplat, measured, round 2 from run_op/meas | frustum | none | run_op/meas |

R2meas separates "second round" from "global memory". Evals: closed_loop_864 10 dev videos x seeds 7/107 x open/F' at 160 f
(VGGT, DINO, LPIPS; win = >= 7/10 two-seed camera AND content not worse), plus the DL3DV-7 481-frame cohort with the paper's
ViPE scorer (gaussian_meas_dl3dv7 protocol, seed 7) for the cross-generator table.

## Costs and schedule (GPU free from ~5 Nov 14:00 after the v2-B / Gaussian-meas / A-retry chain)

| day | work | GPU h |
|---|---|---|
| 4 Nov (CPU now) | 2Xplat venv; adapter (`twoxplat_memory.py`: context-only forward, Sim(3) contract, planned-camera render, measured poses); visibility bank; bias module; pool generator switch; CPU tests (Sim(3) round trip 1e-6, bias zero-gain identity, bank selection on recorded depth) | 0 |
| 5 Nov pm | download v2_hr (7 GB) + DA3-GIANT-1.1 (~5 GB) once disk allows; profile 8/16/32 views at 960x540 next to nothing; cached-prefix comparison on the 10 dev prefixes vs ReSplat (pose err vs VGGT, render PSNR/LPIPS vs recording at the planned cameras, coverage, s/call, GiB) | 3 |
| 6 Nov | round-2 pool, 300 streams x 4 calls with the loop active (~45 s/item -> ~4 h) | 4 |
| 7 Nov | train G then G+B (+250 each, ~6 h each with offloaded renders precomputed from the pool) | 12 |
| 8-9 Nov | 160-f evals (8 runs/arm x 2 arms ~ 10 h) + 481-f DL3DV-7 (4 arms ~ 4 h) | 14 |
| 10-11 Nov | R2meas control if the go-gate passed; otherwise repeat the best arm with seed 2 | 10 |
| 12-16 Nov | writing; figures from eval864 + scores | 0 |

Go-gate after 5 Nov: 2Xplat fits with the DiT offloaded (<= 20 GiB for 32 views), measured poses within the DA3-vs-VGGT floor
on recordings (<= .06 units over 4 calls), render PSNR at planned cameras >= ReSplat's on >= 7/10 prefixes. If 32 views do not
fit, cap at 16 with farthest-point thinning; if the render is worse than ReSplat, keep ReSplat for the structural read and test
only the visibility bank + bias (the measured poses still come from 2Xplat's DA3 pass).

## Facts checked today

- 2Xplat: MIT, code github.com/HwasikJeong/2Xplat (commit 40e7a22, cloned to third_party/2Xplat), weights HF HwasikJeong/2Xplat:
  2xplat_v2_hr.pt 6.98 GiB (Aug 2026, multi-resolution appearance expert, 34M samples), 2xplat_dl3dv_hr.pt 11.48 GiB, lr 11.46,
  mvp_dl3dv_lr.pt 2.86; geometry expert = depth-anything/DA3-GIANT-1.1 (separate download; our cache holds DA3NESTED-GIANT-LARGE
  and DA3-BASE only). Inputs 960x540 at hr, 16-128 views; DA3 at 504x280. Scale normalised (max |t|), poses + intrinsics
  predicted, `camera_mode pred_pose_pred_intr`. gsplat fork only matters for training / EPA; inference uses standard
  `rasterization(packed=False, absgrad=False)` (framecrafter has gsplat 1.5.3). README env: torch 2.10 cu130; ours is 2.12 cu128
  on driver CUDA 13.2 -> try the existing torch first.
- packed_model.PackedTeacher.forward passes a dense (1,1,S,S) `attn_logit_bias` to every block -> the correspondence bias is an
  additive term, no attention rewrite.
- select_bank (data.py:86) scores candidates by frustum coverage of sample points at depths .5/1/2/4 along the target rays;
  the visibility version replaces those points by the map's rendered target depth.
- Disk 19 GiB free with the queues at the 12 GiB margin: weights cannot be downloaded before the chain ends or more space is
  freed. Never run the 2Xplat profile next to a queue GPU job.

## Code written 4 Nov evening (fresh_teacher_20261021/, all CPU tests pass; queues untouched)

| file | role | test |
|---|---|---|
| `map_geometry.py` | Sim(3) Umeyama / depth-scale registration, token unproject/project, map-addressed correspondences with occlusion test, visibility coverage + greedy bank | `tests_map_geometry.py` (5) |
| `packed_model_corr.py` | `PackedTeacherCorr`: per-block zero-init gain x sparse correspondence matrix added to the dense attn bias (main + PRoPE branch); `upgrade()` wraps a loaded PackedTeacher | `tests_packed_corr.py` (3, bit-exact identity at zero gain, gradient through checkpointed blocks, masked-token rejection) |
| `twoxplat_memory.py` | `TwoXplatScene` (keyframe thinning <= max_views, context-only 2Xplat pass, Sim(3) to the anchor gauge, cached scene, render RGB+depth+alpha at gauge cameras), `TwoXplatModel` (frozen 2Xplat v2 hr + DA3-Giant, 952x532 inputs, gsplat RGB+ED) | `tests_twoxplat_memory.py` (4, fake reconstructor) |
| `rows_g.py` | visibility bank, per-row correspondence matrix, round-2 pool rows (`build_row_g`), cached recorded windows (`CachedBuilder2X`) | `tests_rows_g.py` (3) |
| `closed_loop_2x.py` | `LoopRoller2X` (2Xplat pass = observer, registration on the measured history, visibility bank in `pack`), `TwoXRender864` | import smoke |
| `gen_onpolicy2.py` | round-2 pool: F' loop (measured writes + offset rejoin) with the 2Xplat memory, items at s = 5, 9, 13, 17 with M/U/P ends, render of the recorded next command, token depths | GPU |
| `precompute_renders_2x.py` | 2Xplat renders + visibility bank + token depths for the SAME recorded windows as data864/renders | GPU |
| `train_g.py` | arms `--op_arm g|gb` on train_m864v: cached 2Xplat recorded rows (cached visibility bank), pool replacement with a shared op_rng, PackedTeacherCorr + corr per row for gb | GPU |
| `profile_2x.py` | go-gate: VRAM/time at 8/16/32 views, measured-pose error on the 10 dev prefixes, render PSNR/LPIPS/coverage vs ReSplat | GPU |
| `closed_loop_864.py` | `--memory 2xplat --max_views 32` (LoopRoller2X), auto-upgrade for receipts with `corr_gain`; backup `closed_loop_864_pre2x_20261104.py` | compiled |
| `queue_2x_night.sh` | waits for the A-retry queue, an idle GPU and >= 24 GiB free; downloads v2_hr (7 GB) + DA3-GIANT-1.1 (5 GB); profile; precompute with the view cap the profile supports | log `queue_2x_night.log` |

Facts found while writing: DA3 patch embedding asserts H, W % 14 == 0 -> 952x532 inputs (cover resize + centre crop); 2Xplat's
`pose_regressor.prune_layers()` drops the depth head, so the call-1 scale comes from the rendered anchor depth; the v2 hr
checkpoint's architecture config is not published (load_ckpt is strict=False) -> inspect the state dict at profile time.
Disk: 19 GiB free at 22:30 with the queues holding a 12 GiB margin; the 12 GB download needs a cleanup first (asked).
