# GaussMeas: a Gaussian render memory for long camera-controlled video rollouts

Research code for a camera-controlled video world model. One image and a camera path go in; the model generates 16 frames
per call and keeps a **3D Gaussian memory of its own frames**: before each call the memory is rendered at the requested
cameras and the render conditions the generation. Base generator: Memorizon-50 (Wan2.2-TI2V-5B, 864×480). Memory: ReSplat
reconstruction with Depth-Anything-3 depth.

This repository holds the current training line (November 2026) end to end: dataset preparation, the training stages,
rollout and the measurements we judge it by. It is research code in progress, not a package.

## State of the line

| Part | State |
|---|---|
| Dataset: DL3DV → 864×480 corpus → window index → render cache | built; scripts included (`docs/PIPELINE.md`, section 2, with the known gaps) |
| Stage A: base model continued without render | trained (350 updates) |
| Stage B: render memory, two paired arms (render as tokens / as input channels) | token arm trained (250 updates); channel arm training |
| Stage C: on-policy rows | decided, not run |
| Comparison with Lyra-2 / Spatia on the cohort | not run for this line yet |

What stage B (token arm) does on the two development scenes, against the recording
(`docs/RESULTS.md` has every table and the open defects):

- It does not redraw the scene at call boundaries: change of generated cells 1.05 / 1.00 of the recording (a model trained
  twice on the same on-policy pool: 2.63 / 2.02).
- It executes slow commands: 1.07 / 1.02 of the planned displacement at frame 320. The same model without the render
  stands still (0.05 / 0.16).
- It returns to what it generated: out-and-back LPIPS 0.041 / 0.126, against 0.080 / 0.365 for the earlier model.
- Open: distant thin objects are enlarged wrongly on approach and then kept, frames darken over 30 calls, and the showroom
  loses sharpness.

## Quick start

```bash
git clone <this repository> && cd gaussmeas-world-model
# edit scripts/env.sh: where the gaussverse package, Wan2.2, ReSplat, Depth-Anything-3 and Memorizon live
source scripts/env.sh
python scripts/check_env.py          # what is present on this machine
scripts/run_cpu_tests.sh             # ~1 min, no weights or data needed
```

Then, in order (`docs/PIPELINE.md` explains every step, its inputs, outputs and how far it has been verified):

```bash
scripts/01_build_corpus448.sh index|plan|vae|anchor|caption|text|audit   # stream index, gauge, text
scripts/02_camera_quality.sh                                             # streams with broken camera labels
scripts/03_download_frames_864.sh                                        # 864x480 frames from DL3DV-960P
scripts/10_prepare_data864.sh                                            # the training corpus
scripts/11_build_window_index.sh                                         # training windows
scripts/12_build_render_cache.sh                                         # renders + validity metadata per window
scripts/14_build_stream_lists.sh; scripts/13_build_far_windows.sh        # exclusion lists

scripts/19_dry_run.sh token                                              # row sequence without the model
scripts/20_train_stage_a.sh                                              # 350 updates, 4.6 h on one RTX 5090
scripts/21_train_stage_b.sh token                                        # 250 updates, 4.6 h
scripts/21_train_stage_b.sh channel                                      # 250 updates, 3.3 h

scripts/30_rollout.sh contract/runs/b4_token_s2/ckpt/trainables_00250.json 0238de14 out/0238de14 long
scripts/40_metrics.sh fast b_token=$PWD/out/{scene}
```

## How it works

1. **Packed rows** (`core/packed_model.py`, `core/prope.py`, `core/data.py`). One sequence
   `[anchor | bank ≤ 4 | recent 4 | render 4 | 2 chunks × 4 latents]` with a chunk-causal attention mask; the context is
   clean, each chunk has its own flow-matching time. Cameras enter through a PRoPE side branch.
2. **Render memory** (`core/renderer864.py`, `contract/resplat_fixed.py`, `contract/build_render_cache_v4.py`). Sources =
   anchor, the last frames and one bank frame, with DA3 depth in the anchor gauge; ReSplat renders the 16 target views;
   the render is VAE-encoded.
3. **Validity** (`contract/frontier.py`). A render cell is trusted only where enough source pixels land in it
   (support density). The same function builds the training metadata and masks the render at inference, so the model
   learns to generate, not extrude, where the memory has no evidence.
4. **Bank** (`contract/memory_contract.py`). Retrieved frames: true revisits first (a nearby pose the camera left in
   between), then first-seen owners of the requested view. One function for the cache builder, the trainer and the rollout.
5. **Two ways to feed the render** (stage B). As four extra tokens with (valid, confidence) metadata, or concatenated to
   the target tokens through a zero-initialised convolution (`contract/channel_render.py`), because a render is
   pixel-aligned with the frame it conditions while retrieved frames are other viewpoints and stay tokens.
6. **Outlier-row rejection** (`contract/row_reject.py`). One row with a gradient norm far above the running median no
   longer sets the direction of a clipped update.

## Layout

- `core/` model, data, base trainer, rollout loop, corpus builder (`prep_864.py`).
- `contract/` the current line: window table and index (`data_contract/`), render cache, stage trainers (`train_a4.py`,
  `train_gmp.py`), rollout (`sota_eval/run_arm_ckpt.py`), measurements, and the committed stream / window lists.
- `foundation/` shared modules the above import: 448-corpus builder, renderer and depth contracts, realized-pose
  measurement, frame download, camera-quality check.
- `scripts/` one script per pipeline step; `gm_env.py` every machine path, overridable by `GM_*` environment variables.
- `docs/PIPELINE.md` the pipeline; `docs/RESULTS.md` measurements; `docs/CHECKPOINTS.md` checkpoint lineage;
  `docs/lab_notes/` the dated notes with every measurement and decision of the line.
- `SOURCE_PROVENANCE.json` for each code file: the hash of the working file it was copied from and how many machine
  paths were replaced. Nothing else was changed.

Checkpoints, data and third-party code (Wan2.2, Memorizon, ReSplat, Depth-Anything-3, the `gaussverse` package) are not in
this repository; `docs/PIPELINE.md`, section 1, lists them. Earlier experiments of the project (context corruption, frontier
mask variants, camera-reward arms, a global re-reconstruction memory) are in the git history before this cleanup. Related:
`alicaank/closeloopcorrection` (closed-loop camera correction across generators).
