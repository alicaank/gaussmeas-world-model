# What we tried, what we found, where it stands

As of 12 November 2026. This is the short history of the line this repository implements: every approach we tried, what
it measured, and whether it stayed. Numbers are from two development scenes unless a cohort is named; "showroom /
playground" is the order of paired numbers. The dated notes with every measurement are in `docs/lab_notes/`.

## The problem

A camera-controlled video model generates 16 frames per call. Over many calls three things go wrong: the camera drifts
from the commanded path, the scene degrades (blur, collapse, darkening), and content the model generated earlier is not
there when the camera returns. Our approach: keep a 3D Gaussian memory of the model's own frames, render it at the
requested cameras, and condition the generation on that render.

## Where it stands

| Question | Answer today |
|---|---|
| Does the model execute the commanded camera? | Yes with the render (1.07 / 1.02 of the planned displacement at frame 320). Without the render it stands still on slow commands (0.05 / 0.16). |
| Does it redraw the scene at call boundaries? | No (change across a boundary 1.05 / 1.00 of the recording). |
| Does it return to what it generated? | Out-and-back LPIPS 0.041 / 0.126, two to three times better than the previous model. |
| Is the long rollout clean? | No. Distant thin objects are enlarged wrongly on approach and then kept; content the model generates at the edge is soft and stays soft; long chains at full frame rate degrade after ~320 frames. Brightness is stable against the anchor. |
| Does it beat released baselines? | Not measured for the current line. The previous model led on camera position over the first 160 frames and lost on appearance afterwards (`docs/RESULTS.md`). |
| What is trained? | Stage A (no render) and both stage-B arms; the render-as-tokens arm won. Stage C (on-policy rows) is running. |

## What we tried, in order

### 1. A packed chunk-causal teacher on Memorizon-50 (from 21 October) — kept, it is the base

One sequence holds the anchor, retrieved frames, recent frames, the render and the chunks being generated, with a
chunk-causal mask and a PRoPE camera branch. Three stages: no render, cached renders on recorded rows, then 50 % on-policy
rows (the model's own one-to-two-call rollouts with measured poses and recorded targets). The result, **Gauss-meas**, is the
"Ours" row of the cohort table: best camera position of all compared models over the first 160 frames, worse appearance
than Lyra-2 after that.

Finding that shaped everything after: **the model executes the camera through the render.** The ray inputs carry almost no
command signal; translation is read from the rendered memory. Released Memorizon, without such a render, executes about a
third of a commanded translation even on the first call.

### 2. Ideas around the memory and the camera signal (October – early November) — dropped

- **Learned per-Gaussian features, per-chunk renders, a refiner:** no measurable gain, or blur.
- **A latent pose head and a camera reward (ReFL-style):** failed; the pose head was not a usable reward.
- **A global re-reconstruction memory (2Xplat) at measured poses:** stopped; it loses once the history is generated frames.
- **Training rows with long generated histories and recorded targets:** failed on three runs. The model learns to catch up
  (executes 1.3–3.7 × the command) and blurs. Only short on-policy histories (1–2 calls) worked.
- **Render dropout / perturbing the render:** every such run lost camera accuracy. See the finding above.

The closed-loop controller that re-plans from measured poses is a separate study (`alicaank/closeloopcorrection`); in the
cohort table it improves camera position for our model and for Lyra-2.

### 3. Context corruption, "GM-B" (8–9 November) — partly worked, superseded

Noise and one-step self-augmentation on the context during training. On the 5-scene cohort it fixed the late collapse
(last-third DINO 0.841 against 0.743; Spatia 0.837, Lyra-2 0.880). But the frames darken (luminance 0.90 of the recording in
the last third), and every call starts with a visible snap (first in-call transition 2.8 × the others). Mechanism: the
corrupted model stops carrying detail through the recent latents and carries it only through the soft render, so it
re-synthesises at every boundary. Three corruption recipes at 250 updates all broke self-consistency this way; 100-update
doses did not.

### 4. Frontier masks (9 November) — one variant kept, at inference and in training

The renderer marks stretched splats at the image border as valid, and the model extrudes the border instead of generating.

- **Depth-agreement mask:** rejected. The renderer's blended depth disagrees with the depth estimator even on flat ground,
  so the test discarded 50–87 % of correct cells.
- **Training with masked and stripped renders ("GM-F"):** the worst arm. It lowered the model's reliance on the render, and
  the camera went with it.
- **Frustum-only mask at inference:** best development arm at first; a second seed showed the headline gain on one scene was
  seed luck.
- **Support-density mask** (a cell is valid if enough source pixels land in it): this is what the current line uses, with
  the same function in the training cache and at inference. With the mask off, an object that is not in the scene enters
  and existing structure tears.

### 5. Diagnostics of the long-rollout failure (10 November)

- With recorded frames as the memory's sources the collapse largely disappeared in that test, which points at memory
  built from generated frames (an upper bound: recorded sources also show content the model never saw).
- The render keeps only 11–45 % of the sources' high-frequency energy; a model that copies it loses detail every call.
- The comparison cohort moves 15–50 × slower per call than our early development scenes. Failures on the fast scenes are
  a different regime and say little about the cohort.
- Two pose estimators agreed on only 52–59 % of calls; counts of camera excursions on collapsed frames are unreliable.
- **A two-band render (latents pulled from memory for low noise levels, RGB render otherwise):** failed its gate.

### 6. A geometry bug in the render wrapper (10 November) — fixed, everything re-rendered

Every render of the earlier line was stretched vertically by 540 / 512 = 1.055: the renderer crops to multiples of 64 rows,
our wrapper decoded at 540. Recorded sources reproduced the recorded target at 18 dB as rendered and 24–32 dB after
undoing the stretch. Two more wrapper errors (depth squeezed onto the cropped grid; 16-view weights run with one to three
sources and without their preset) added misalignment and a dark render. The model had been trained against this, and un-stretched a
corrected render by itself. Fix: 8-view weights at 512 × 960 (`contract/resplat_fixed.py`), a new cache, retraining.
Conclusions of sections 3–5 that rest on the render were measured before this fix.

### 7. The corrected lineage, "GM′" (11 November) — better first calls, same long-horizon problem

With the corrected render: better first three calls (LPIPS 0.024 / 0.070 against 0.028 / 0.092) and no vertical drift, but
the playground still collapses by frame 480. Two findings from it:

- **The bank was empty on every revisit.** The retrieval rule selected frames by coverage gain over the anchor and the
  recent frames, so a frame the camera returns to was never retrieved. The contract bank retrieves true revisits first.
- **A control that only continues training on the same on-policy pool, with no corruption ("gmc3"), redraws the scene**
  (boundary change 2.63 / 2.02 of the recording, against 1.04 / 1.14 after the first pass). The corruption recipes were not
  the cause of the redrawing; the second pass over the pool is.

An "oracle context" experiment (recorded frames as context) first looked decisive and was withdrawn: its output was closer
to the recording than the recording's own neighbouring frames, i.e. it leaked.

### 8. The contract line (11–12 November) — current

One recipe in which training and inference use the same functions for the window mix, the bank, the render and its validity
mask (`docs/PIPELINE.md`).

- **Stage A** (no render): learns the new inputs. Alone it executes large steps but stands still on slow commands; a model
  trained with the stock recipe instead over-moves and collapses (boundary change 1.7–1.9 × the recording).
- **Gradient spikes explained.** In every run of the trainer 3–10 % of updates had a gradient norm of 1e2–1e6 from one row.
  The rows are windows far from the anchor: the anchor is in every row and the camera branch carries the raw
  anchor-to-target translation into its attention. Outlier-row rejection removes them (12 of 4,000 rows, all such windows).
- **Stage B, render as tokens:** the numbers in the table at the top. The bank helps identity and the density mask fixes the
  edge, each shown by switching it off.
- **Stage B, render as input channels:** the render is pixel-aligned with the frame it conditions, so it is concatenated to
  the target tokens while retrieved frames stay tokens. 28 % faster per update, but within 250 updates the zero-initialised
  input reads the render half as much as the attending tokens (paired loss gain 0.014 against 0.028). The render then
  cannot carry the camera: the arm keeps part of stage A's standstill on slow commands (execution 0.57 / 0.78) and is worse
  in the first calls and on return (out-and-back LPIPS 0.274 / 0.390 against 0.041 / 0.126). Dropped for this round.
- **The stray objects.** Thin objects far from the anchor are drawn as wire-frames when the camera approaches, then return
  through the render at full confidence. Not the bank (still there with the bank off), not the mask. The model before the
  on-policy stage has them too; the model after it draws a faint frame and drops it. Hence stage C.

## What the work taught us

- The camera is executed through the render. Do not train the model to trust the render less; remove misleading render
  cells with a validity mask the model was trained with.
- A model that does not move looks stable. Judge "the scene does not change" together with a measured execution ratio, and
  report regeneration as a distance from 1 in both directions.
- On-policy rows are needed (they teach the model to produce real content from a render of its own flawed frames), and
  they are dosed: one pass over the pool cleans, a second pass makes the model redraw. Bound the exposures per pool item.
- Runs with the same seed diverge after about 160 frames. Re-run before trusting a small gap; a win on one scene can be
  seed luck.
- Feature similarity of late frames measures plausibility, not fidelity. A model can produce a sharp, different scene.
  Pair it with a distance to the recording.
- Experiments that feed recorded frames into the context can leak the answer.
- With Adam, accumulating more rows per update does not buy signal for a fixed number of rows; more rows do.
- Check the renderer against recorded frames before believing anything that depends on it.

## Open problems and next steps

1. Long chains at full frame rate (30 calls at the recorded frame rate) degrade after ~320 frames; at the motion-targeted
   step (every 4th pose) they do not.
2. Stage C: a pool generator in the inference layout, at most 1.5 exposures per pool item, stop when the stray objects are
   gone and regeneration stays within 0.3 of 1.
3. Soft generation at the edge: masked cells are generated softer and the soft content returns through the render as valid.
   Measured against the anchor, the token arm does not darken (the recording's exposure moves). A moment patch at inference
   restores brightness and sharpness but adds a jump at call starts; text guidance sharpens but shifts brightness. Guidance
   applied only in masked cells is an untested option.
4. Windows far from the anchor: the anchor should leave the context (or be replaced by a near reference) when it is far and
   out of view, with the same rule in training and inference.
5. The confidence channel is the renderer's alpha, which stays high on an approached surface with few source pixels. A
   confidence derived from support density is the next contract change.
6. The training frames are not undistorted (303 of 891 training streams are off by more than 8 px at the corners); the
   evaluation scenes are.
7. Final validation on the 5-scene cohort against Lyra-2 and Spatia, two seeds.
