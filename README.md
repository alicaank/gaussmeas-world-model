# GaussMeas: measured-pose Gaussian memory for long camera-controlled rollouts

Research code for a camera-controlled video world model that keeps a **3D Gaussian memory addressed by measured (realized)
camera poses** instead of trusting commanded poses or the temporal past. Base generator: Memorizon-50 (Wan2.2-TI2V-5B, 864×480,
16-frame chunks). Memory: ReSplat reconstruction of the model's own frames with DA3 depth, rendered at the requested future views
and injected as VAE-encoded render tokens with (valid, confidence) metadata. The camera is executed from the render, the realized
pose is measured after every chunk and written back into the memory labels, and a closed-loop controller rejoins the plan.

This repository is the **working code of the current method line** (October–November 2026), copied byte-for-byte from the
analysis tree with hashes in `SOURCE_PROVENANCE.json`. It is organised for reading and reproduction on the original machine; it is
not yet a pip-installable package (see *Environment*).

## Method in one page

1. **Packed chunk-causal teacher** (`src/packed_model.py`, `src/prope.py`, `src/data.py`). One sequence
   `[anchor | bank ≤4 | recent 4 | render 4 | N chunks × 4 latents]` with a chunk-causal attention mask; context at t = 0, each chunk
   with its own flow-matching time. Cameras enter as Plücker rays and a PRoPE side branch (Memorizon layout). Render slots carry
   role `ROLE_RENDER`, the chunk-0 cameras and pooled (valid, conf) metadata; render tokens whose pooled validity < .5 are masked.
2. **Gaussian memory** (`src/renderer864.py`, `src/render864_rows.py`, `third_party_snapshot/resplat_condition.py`,
   `third_party_snapshot/foundation_renderer.py`). Sources = anchor + last prefix latents + the best bank view, DA3 depth in the
   anchor gauge (metres per anchor-depth unit), ReSplat render of the 16 target views at 540×960, downsized to 480×864 and encoded
   by the frozen Wan VAE. No learnable per-Gaussian features.
3. **Training lineage** (`src/train_plain.py` → `src/train_m864v.py` / `src/train_b864.py` → `src/train_onpolicy.py`).
   Stage A: stock Memorizon-50 init, no render. Stage B: cached renders on recorded rows (render_p .75). Stage C (**Gauss-meas**,
   `run_op/meas`): 50 % on-policy rows from the model's own 1–2-call rollouts with **measured** poses and recorded targets.
4. **Realized-pose measurement and closed loop** (`src/closed_loop.py`, `src/closed_loop_864.py`,
   `third_party_snapshot/realized_pose.py`). DA3 on the anchor plus generated frames, shared-pixel anchor-depth fit, failed
   measurements reject the write. Replan modes: `offset` rejoin (the model's own F′ arm) or the paper's controller
   (`src/run_arm_original_20261103.py`, `src/run_arm_ckpt.py`).
5. **Robustness line (9 Nov)** (`src/gm_b_corruption.py`, `src/frontier.py`, `src/train_gm_b.py`, `src/train_gm_f.py`).
   *GM-B*: context corruption during training (self-augmentation with the model's own one-step x0 of the recent latents, bank and
   render noised at the same strength). *GM-F*: GM-B plus a **frontier mask** on the render memory: a render pixel counts as
   observed only if its rendered 3D point projects inside some source image (frustum test); unobserved pixels get
   (valid, conf) = 0 so the model must generate instead of extruding the border. Training uses the same test with DA3 depth of
   the recorded target frames (`src/build_frontier_cache.py`) plus regional render dropout. The depth-agreement variant of the
   test was rejected on 9 Nov: ReSplat's alpha-blended depth disagrees with DA3 even on flat ground, so it discarded 50–87 % of
   correct render cells (`docs/lab_notes/PLAN_robust_native_20261108.md`).

## Results (seed 7, one scorer per table)

DL3DV cohort of the closed-loop paper, 5 scenes, 480 frames (30 calls), scorer `vipe_native_map_mask_diagnostic_v3`
(`docs/lab_notes/comparison_table_20261105.md`, "Ours" = Gauss-meas `run_op/meas`):

| Method | Position err ↓ | Endpoint ↓ | Rotation ° ↓ | LPIPS ↓ | DINO ↑ | LPIPS last⅓ ↓ | DINO last⅓ ↑ |
|---|---|---|---|---|---|---|---|
| Ours | 0.0505 | 0.154 | 2.54 | 0.378 | 0.855 | 0.552 | 0.743 |
| Ours + correction | 0.0336 | 0.093 | 0.85 | 0.356 | 0.853 | 0.521 | 0.733 |
| Lyra-2 | 0.0732 | 0.187 | 0.17 | 0.333 | 0.925 | 0.484 | 0.880 |
| Lyra-2 + correction | 0.0333 | 0.068 | 0.24 | 0.297 | 0.936 | 0.416 | 0.898 |
| Spatia | 0.0749 | 0.214 | 2.60 | 0.381 | 0.906 | 0.550 | 0.837 |
| Matrix-Game 3.5 Distilled | 0.1952 | 0.554 | 3.47 | 0.559 | 0.798 | 0.712 | 0.633 |

Ours leads every released baseline on camera position over the first 160 frames (0.0078 vs Lyra-2 0.0095) and loses on
appearance after ~160 frames; the robustness line above targets that drift. GM-B / GM-F rows on this cohort are being computed
(`src/run_arm_ckpt.py`, `src/build_table_sota.py`).

Development set (6 held-out scenes, 20 calls × 16 frames, `src/closed_loop_864.py --write 1 --render 1`, VGGT camera):

| Arm | LPIPS calls 11–20 ↓ | DINO 11–20 ↑ | sharpness | VGGT pos ↓ | VGGT rot ° ↓ | rot > 30° per scene ↓ |
|---|---|---|---|---|---|---|
| Gauss-meas (GM) | 0.601 | 0.403 | 262 | 1.036 | 20.1 | 3.5 |
| GM-B (corruption) | 0.623 | 0.442 | 379 | 0.774 | 17.3 | 2.7 |
| GM-F, inference mask tol .15 | 0.621 | 0.448 | 391 | 0.935 | 20.8 | 3.3 |
| GM-F, frustum-only mask | 0.636 | 0.418 | 358 | 0.917 | 30.6 | 5.8 |
| **GM + frustum-only mask (inference only, no training)** | **0.590** | **0.451** | 272 | 0.884 | **13.3** | **2.2** |

The frustum mask applied at inference to the untrained Gauss-meas model is the best arm on every aggregate (kitchen rotation
33° → 3.1° with no excursions). GM-F, trained on masked and stripped renders, is the worst: lowering the model's reliance on the
render costs camera execution, which runs through the render. GM-B + mask is being evaluated; GM-F is dropped.

## Ideas and their status

`docs/IDEAS_INDEX.md` lists every idea of the line for using the Gaussian memory (measured writes, on-policy rows,
context corruption, frontier mask, per-chunk renders, Gaussian features, 2Xplat global memory, earned memory, selective visual
memory, correspondence transfer) and the camera signal (F′ closed loop, memory-verified camera reward with reward-weighted
velocity matching, camera-ReFL with a latent pose head, camera-read probe), with what was tested and what remains a proposal.
Code under `src/ideas/`, documents under `docs/ideas/` (including the CVPR proposal source).

## Checkpoints

Not in git (sizes). Paths and SHA-256 of the receipts are listed in `docs/CHECKPOINTS.md`.

## Environment

The code runs from the analysis tree with `src/env.sh` (conda env `framecrafter`, CUDA 12.8, RTX 5090). External components,
all read-only here: Memorizon-50 weights (`Luffuly/memorizon-1280x704-50step`, loaded through DiffSynth's Wan2.2 converter),
the Wan2.2 VAE, ReSplat (`gaussverse/memory`), Depth-Anything-3 (Lyra-2 source tree), VGGT and ViPE scorers.
`third_party_snapshot/` holds byte copies of the in-house shared modules the method imports (realized pose, foundation renderer
and contract, VAE stream, DA3 predictor, scorer). Tests: `tests/` (`python -m unittest tests/test_frontier.py` etc. are CPU-only;
`tests/tests.py` needs the weights).

## Layout

- `src/` method, training, evaluation and the SOTA-cohort runner; `src/ideas/` the reward arm, per-chunk render, feature-memory,
  2Xplat and refiner experiments; `src/ideas/memorizon_refl/` the released-Memorizon ReFL / injection line. `docs/lab_notes/` dated plans with every measurement and
  decision of the line. `docs/CHECKPOINTS.md` checkpoint lineage. `SOURCE_PROVENANCE.json` source path and hash per file.
- Related repositories: `alicaank/closeloopcorrection` (the camera-correction controller and its cross-generator study).
