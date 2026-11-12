# Pipeline: environment, dataset, training stages, rollout, measurements

Every step is one script in `scripts/`. They all start with `source scripts/env.sh`, which puts the code directories and
`gm_env.py` (the single place for machine paths) on `PYTHONPATH`. Run `python scripts/check_env.py` to see what your machine
has.

How far each step has been verified **from this repository layout** is stated per step:

- **verified**: run from a checkout of this repository on 12 Nov 2026.
- **as run**: the code that produced our data and checkpoints, copied with only the machine paths replaced
  (`SOURCE_PROVENANCE.json`); not re-run from the repository layout yet.
- **planned**: decided but not run.

## 1. Environment

One GPU with 32 GiB (RTX 5090; training peaks at 24.6 GiB), CUDA 12.8, Python 3.11, PyTorch, plus `safetensors einops
opencv-python lpips bitsandbytes scipy pillow tqdm transformers huggingface_hub imageio-ffmpeg`. `nvcc` must be on `PATH`
(`CUDA_HOME`) because ReSplat's rasterizer (gsplat) loads its extension only then.

External components, none of them in this repository. Point `scripts/env.sh` at them:

| Component | Variable | What it is |
|---|---|---|
| `gaussverse` package | `GM_WORKSPACE` (its parent directory) | Our base library: DiffSynth-derived Wan2.2 model code, camera rays, VAE streaming, ReSplat wrapper. Repository `alicaank/gaussverse`, branch `snapshot/research-state-2026-09-11` (commit `6ed16ba`); the 60 files this pipeline imports are tracked there. |
| Wan2.2-TI2V-5B | `GAUSSVERSE_WAN22_DIR` | DiT weights (fingerprint checked against `foundation/BASE_MODEL_AUDIT.json`), `Wan2.2_VAE.pth`, umT5-xxl encoder and tokenizer. |
| Memorizon-50 weights | `core/memorizon50/` | The released checkpoint `Luffuly/memorizon-1280x704-50step` as downloaded (config + 3 safetensors shards). It is not converted: `core/model_io.py` maps the keys at load time. |
| Memorizon code | `GM_MEMORIZON` | The released Memorizon source (its retrieval rule is one of the bank options; `core/encode_prompts_local.py` uses its text encoder wrapper). |
| ReSplat | `GM_RESPLAT` | Source tree with `pretrained/resplat-base-dl3dv-512x960-view8-8179ed87.pth` (the render cache and inference use this 8-view model) and `resplat-base-dl3dv-540x960-view16-a72dc6d0.pth` (loaded once by the default builder). |
| Depth-Anything-3 | `GAUSSVERSE_DA3_SOURCE` | Source tree; weights `DA3NESTED-GIANT-LARGE-1.1` in the Hugging Face cache. |
| MoGe, Qwen3-VL-2B | Hugging Face cache | Only for the 448 corpus (anchor gauge, captions). |
| ViPE scorer | not included | The ViPE-based camera scorer of our tables belongs to the closed-loop correction study and is not in this repository. `contract/gate_b4_camera.py` turns its trajectories into execution ratios. |

Data and outputs live next to the code and are git-ignored (`gm_env.py` lists them): `core/data864/`, `core/memorizon50/`,
`contract/render_cache_v4/`, `contract/runs/`, `foundation/dataset_v3/`. Put them there or symlink them.

Checks without weights or data (**verified**): `scripts/run_cpu_tests.sh` (outlier-row rejection, channel-render wrapper, PRoPE
numerics).

## 2. Dataset

The training set is 915 streams of DL3DV at 864×480 (895 train, 20 development), each a contiguous run of frames with
registered cameras, encoded by the Wan2.2 VAE. It is derived from an older 448×256 corpus that fixes the stream windows, the
anchor gauge and the text.

| Step | Script | Needs | Writes | Device | Status |
|---|---|---|---|---|---|
| D0 | (download) | Hugging Face `DL3DV/DL3DV-ALL-480P` | `$GM_WORKSPACE/data/DL3DV-480P/<1K..>/<scene>/{transforms.json, images_8/}` | network | not scripted |
| D1 | `01_build_corpus448.sh index`, then `plan` | D0 | `foundation/dataset_v3/{splits.json, corpus/streams.jsonl, …}` | CPU | as run |
| D2 | `01_build_corpus448.sh vae`, `anchor`, `caption`, `text`, `audit` | D1, Wan2.2 VAE, DA3, MoGe, Qwen3-VL, umT5 | `foundation/dataset_v3/components/<stream>/{vae,anchor,text}.npz, anchor.json, caption.json` | GPU | as run |
| D3 | `02_camera_quality.sh` | D2 | `foundation/camera_quality/manifest.json` | CPU | as run |
| D4 | `03_download_frames_864.sh` | D1, Hugging Face `DL3DV/DL3DV-ALL-960P` | `$GM_DATA_ROOT/DL3DV-960P-864x480/<subset>/<scene>/{frames_864x480/, transforms.json, crop.json, done}` | CPU + network | as run |
| D5 | `10_prepare_data864.sh` | D2–D4, Wan2.2 VAE, DA3 | `core/data864/{streams.jsonl, components/<stream>/{vae.npz, meta.json}}` | GPU | as run |
| D6 | `11_build_window_index.sh` | D5 | `contract/data_contract/{windows.jsonl, index_v4.json}` | CPU | as run (the committed `index_v4.json` is the one we trained on) |
| D7 | `12_build_render_cache.sh` | D5, D6, ReSplat, DA3 | `contract/render_cache_v4/<stream>/start_<s>.npz`, `build_stats.jsonl` | GPU, ~1.8 h | as run |
| D8 | `14_build_stream_lists.sh` | D6, D7 | `flag_streams_v4.json`, `fast_streams_v4.json` | CPU | **verified**: reproduces both committed lists exactly |
| D9 | `13_build_far_windows.sh` | D5, D6, D8 | `far_windows_v4.json` | CPU | **verified**: reproduces the committed list exactly |

What each step does:

- **D1, streams and split.** Frames of a scene are sorted; a frame is valid if its pose is finite and rigid and its image
  exists. The scene is cut at every invalid frame and every gap in the frame numbers; each contiguous run of at least 17
  frames is a stream (`stream_id` = digest of scene, start, end and the `transforms.json` hash, so the ids are the same on
  any machine with the same data). 128 development scenes are chosen by stratified hashing (seed 20261009). On our data:
  8,948 scenes, 9,337 streams.
- **D2, components.** `vae`: Wan2.2 latents of the 448×256 frames, intrinsics, DL3DV cameras converted to the OpenCV
  convention. `anchor`: depth of frame 0 (DA3 + MoGe) normalised to median 1, and `registered_units_per_anchor_depth`, the
  scale between DL3DV's camera units and that depth: the **anchor gauge**, in which all cameras are expressed. `caption`:
  Qwen3-VL-2B on frame 0, at most 96 tokens. `text`: its umT5-xxl embedding.
- **D3, camera quality.** In the anchor gauge, a stream is flagged for a per-latent step above 2, a frame-to-frame step above
  max(0.5, 20 × the median step), or non-finite poses. 661 of 9,337 streams on our data; they never enter the 864 corpus.
- **D4, frames.** Training scenes with a stream of at least 41 latents, in a seeded order. The 960×540 images are
  centre-cropped to the 864:480 aspect, resized with LANCZOS and saved as JPEG q95. `done` marks a completed scene.
- **D5, the 864 corpus.** Per stream: latents `[1, 48, L, 30, 54]` (4 frames per latent after the first), intrinsics at
  864×480, cameras in the anchor gauge, and `metres_per_anchor_unit` (median DA3 metric depth of the anchor over the gauge
  depth). The model's camera unit is `metres_per_anchor_unit × 2 / 4`. 20 streams (seeded shuffle, seed 864) are the
  development split.
- **D6, windows.** A training row is a window of 2 chunks × 4 latents at a start latent. `windows.jsonl` describes all
  67,547 candidate windows (per-call translation and rotation, anchor coverage, bank size, true revisits). `index_v4.json`
  keeps 5 per stream: the first-call window and four continuation windows drawn with weights that favour slow motion
  (the inference range), at least 13 latents of history and true revisits (seed 20261111).
- **D7, render cache.** For every indexed window: the bank chosen by the contract rule (`contract/memory_contract.py`: true
  revisits, then first-seen owners, deterministic 20 % dropout), a ReSplat render of the 16 target frames from the anchor,
  the two last context latents and the first bank frame (8-view weights, 512×960), VAE-encoded, with raw (valid, confidence)
  metadata and the **support-density** metadata the model is trained and run with (a cell is valid if at least 15 % of the
  nominal number of source pixels lands in it; `contract/frontier.py`). Our cache: 4,422 windows, PSNR p5 / p50 / p95 =
  17.9 / 24.6 / 32.0 dB.
- **D8, stream lists.** `flag_streams_v4`: at least 2 cached windows and at least half of them below 20 dB (the renders say
  the camera labels are inconsistent); 86 streams. `fast_streams_v4`: median per-call translation above 1.0 gauge units; 22
  streams. `bad_streams_v3` (119 streams) is the earlier list from an older cache and is kept as committed, because stage A
  excluded exactly that list. The three together remove 157 streams; 734 streams and 3,660 windows remain.
- **D9, far windows.** Windows whose cameras are more than 10 model units from the anchor (104 windows). They cause the
  gradient spikes of the trainer (section 3); stage B rejects them row by row, stage C excludes them.

### Known gaps in the dataset chain

- **D0 is not scripted**, and D1 reads three inputs from the original workspace that are not in git:
  `foundation/data_v2/locked_test.json` (committed), `<workspace>/analysis/plan_base_restart_20261008/scene_qa.jsonl`
  (per-scene glass / dynamic proxy scores used as split strata) and `foundation/data_v2/corpus/scenes.jsonl` (an earlier
  scene list). Without them D1 needs a small change to its strata.
- **D5 cannot be reproduced on our machine today**: two disk cleanups removed frames it reads (53 scenes of the 864 frames
  and the 448×256 anchor frames of 547 streams). They have to be downloaded again; the `transforms.json` hash is checked.
- 308 of the 915 rows of our `data864` come from two earlier versions of the scale calibration (`metres_per_anchor_unit`);
  the current `core/prep_864.py` calibrates in one pass.
- `bad_streams_v3.json` needs the older render cache v3, whose builder is not in this repository. On new data use the lists
  of D8 only.
- Rows of `core/data864/streams.jsonl` store absolute paths (`source`, `text`, `anchor_json`): a corpus cannot be moved
  without rewriting them.

## 3. Training stages

All stages start from the Memorizon-50 weights and use one trainer (`core/train.py`): flow-matching velocity loss on the 8
target latents, AdamW (8-bit), weight decay 1e-3, gradient clip 1.0, bf16, **16 rows accumulated per update** (one packed
row per forward), time sampling `shift3`, text dropout 0.15, a checkpoint every 50 updates. A checkpoint is a receipt
`trainables_<update>.json` (hash, config, lineage) next to a `.pt` of the trainable tensors.

Before any GPU run: `scripts/19_dry_run.sh token` draws the row sequence without the model and checks every cache window
(**verified**, 53 s for 20 updates).

| Stage | Script | From | Rows | Updates | Time | Status |
|---|---|---|---|---|---|---|
| A | `20_train_stage_a.sh` | Memorizon-50 | recorded windows, **no render**, contract bank; 119 streams excluded | 350 | 4.6 h | as run (`a4_s2`) |
| B, token arm | `21_train_stage_b.sh token` | A @ 350 | 90 % with render as 4 sequence slots + density metadata; 157 streams excluded; row rejection | 250 | 4.6 h | as run (`b4_token_s2`) |
| B, channel arm | `21_train_stage_b.sh channel` | A @ 350 | same rows, render concatenated to the target tokens through a zero-initialised conv | 250 | 3.3 h | as run (`b4_channel_s2`, training on 12 Nov) |
| C | `22_train_stage_c.sh` | winning B arm | + on-policy pool rows | 100 | | **planned** |

- **Stage A** teaches the input contract (window mix, bank rule, text dropout) without the render. On its own it executes
  fast commands but stands still on slow ones (section 5): the camera is executed through the render, which arrives in B.
- **Stage B** adds the render memory. The two arms are paired: same checkpoint, rows, noise and seed.
  *Token*: 4 render slots with their own role and (valid, confidence) metadata; invalid render tokens are masked as keys.
  *Channel* (`contract/channel_render.py`): the render latent × valid and (valid, confidence), 50 channels, enter the target
  tokens of the same chunk through a zero-initialised convolution, so the run starts exactly as the stage-A model; retrieved
  bank frames stay tokens. 17 instead of 21 slots per row.
- **Outlier-row rejection** (`contract/row_reject.py`, `--reject 50`). Without it 3–10 % of the updates of every run had a
  pre-clip gradient norm of 1e2–1e6 from a single row, and clipping then let that row set the direction of the whole step.
  A row whose gradient norm exceeds 50 × the running median of accepted rows is dropped and the accepted rows of that update
  are replayed with their saved random states, so the random stream, and with it the pairing of the arms, is unchanged. Cause
  of the outliers: windows far from the anchor. The anchor is in every row and the PRoPE camera branch carries the raw
  anchor-to-target translation into its attention logits. In the token arm all 12 rejected rows (of 4,000) were such windows.
  `contract/reject_report.py` lists rejections per run and marks those near the threshold.
- **Stage C** is decided but not run: `scripts/22_train_stage_c.sh` documents the recipe (pool from the stage-B model, at most
  1.5 exposures per pool item, stop rule) and refuses to run without `GM_ALLOW_UNVALIDATED=1`. Why it is needed and why the
  dose is bounded: `docs/RESULTS.md`.

## 4. Rollout

`scripts/30_rollout.sh <receipt> <scene> <out> fast|long|return` runs `contract/sota_eval/run_arm_ckpt.py`: one image (the
anchor), a camera plan, 16 frames per call. After each call the frames are written to the memory; the next call's render is
built from the anchor, the last frames and a bank frame.

- A model trained with render runs with the **contract inference**: `--resplat v8 --frontier density --bank contract` (the
  renderer, validity mask and bank rule of the training cache).
- A stage-A model has no render rows: `NORENDER=1 scripts/30_rollout.sh …`.
- `fast` = every 4th plan pose, 7 calls; `long` = the plan as recorded, 30 calls; `return` = out and back, 30 calls.

The evaluation scenes are described by `$GM_EVAL_BATCH/index.json`, `trajectories/<scene>.npz` (camera plan) and
`$GM_EVAL_INPUTS/inputs/<scene>.npz` (anchor latent, intrinsics, text). Our two development scenes (`02267acf` showroom,
`0238de14` playground) and the 5-scene comparison cohort were prepared for the closed-loop comparison; a builder for these
inputs is not in this repository. Status of the rollout script: as run.

## 5. Measurements

CPU, on rollout directories (`scripts/40_metrics.sh`):

- `fast` (`contract/fast_metrics.py`, 20 calls): **regeneration**, the change of generated cells over 8 frames inside a call
  and across a call boundary, as a ratio to the recording. Report `|ratio − 1|`: above 1 the model redraws the scene at call
  boundaries, below 1 it stands still. Also call-start snap, DINO to the recording, sharpness, luminance drift.
- `first` (`contract/gate_bp.py`): LPIPS over calls 1–3 and the vertical / horizontal scale ratio of the generated frames.
- `return` (`contract/diag_return.py`): out-and-back self-consistency, LPIPS between the way out and the way back at the
  same pose, split into cells the anchor sees and cells the model generated.
- `contract/gate_b4_camera.py name:stride …`: command execution from ViPE trajectories (projection of the measured
  displacement on the planned one, lateral part, yaw ratio).
- `contract/probe_bank_swap.py`: does the model use the bank (loss with own / no / swapped bank on recorded windows)?
