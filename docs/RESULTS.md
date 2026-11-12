# Results

## Scope of these numbers

- **Two development scenes only:** `02267acf` (showroom) and `0238de14` (playground). Paired numbers are always given in
  that order.
- **One run per number:** seed 7, one rollout per model and scene unless stated. Runs with the same seed diverge after about
  160 frames from GPU nondeterminism alone, so small differences are not established.
- **Held-out cohort untouched:** the three remaining scenes of the 5-scene comparison cohort (`334ff324`, `61b27b61`,
  `a126efe8`) have not been run with any model of this line. They are kept for the final validation, which requires two
  runs per scene.
- Every number is in `docs/lab_notes/RECIPE_contract_20261111.md` with the time it was measured.

Models: **stage A** (`a4_s2` @ 350, no render), **B token** (`b4_token_s2` @ 250, render as tokens), **B channel**
(`b4_channel_s2` @ 250, render as input channels; same rows, noise and seed as B token). References from the earlier
lineage: **plain-350** (Memorizon recipe, no render), **B′** (corrected render, no on-policy stage: the fair reference for
B token), **GM′** (B′ + on-policy pool), **gmc3** (GM′ + 250 more updates on the same pool).

## Stage B: the two arms

| Measure | B token | B channel |
|---|---|---|
| LPIPS, calls 1–3 | 0.021 / 0.064 | 0.084 / 0.113 |
| Command execution at frame 320, normal speed (ViPE) | 1.07 / 1.02 | 0.57 / 0.78 |
| Command execution at frame 112, every 4th pose | 1.08 / 0.98 | 1.06 / 1.11 |
| Change across a call boundary, relative to the recording | 1.05 / 1.00 | 0.54 / 0.87 |
| Out-and-back LPIPS, 30 calls | 0.041 / 0.126 | 0.274 / 0.390 |
| Out-and-back LPIPS, every 4th pose, 14 calls | 0.049 / 0.113 | 0.099 / 0.407 |
| Loss gain from the render (paired, 48 unseen windows × 2 noise seeds) | 0.028 ± 0.004 | 0.014 ± 0.003 |
| Bank use (loss without bank − with bank) | 0.0056 ± 0.0016 | 0.0073 ± 0.0013 |
| Training loss, last 50 updates | 0.254 | 0.261 |
| Seconds per update / peak memory | 65 / 24.6 GiB | 48 / 24.0 GiB |

**The token arm is the stage-B model.** Within this budget (250 updates) the zero-initialised channel input reads the
render half as much as the attending render tokens. The render is then too weak to carry the camera: the channel arm keeps
part of stage A's standstill on slow commands (execution below 1, regeneration below 1). Longer training could change this;
it was not tested.

## Does the model redraw the scene at call boundaries?

Change of generated cells relative to the recording, 20 calls at normal speed. 1 = as much change as the recording; above 1
the model redraws, below 1 it stands still. We report the distance from 1 because both directions are defects.

| Model | inside a call | across a boundary | boundary, distance from 1 |
|---|---|---|---|
| B token | 0.94 / 0.97 | 1.05 / 1.00 | 0.05 / 0.00 |
| B channel | 0.49 / 0.87 | 0.54 / 0.87 | 0.46 / 0.13 |
| B′ | 1.07 / 0.94 | 1.10 / 0.98 | 0.10 / 0.02 |
| GM′ | 0.91 / 1.08 | 1.04 / 1.14 | 0.04 / 0.14 |
| stage A (no render) | 0.29 / 0.68 | 0.32 / 0.58 | 0.68 / 0.42 |
| plain-350 (no render) | 1.67 / 1.52 | 1.91 / 1.70 | 0.91 / 0.70 |
| gmc3 | 1.17 / 1.17 | 2.63 / 2.02 | 1.63 / 1.02 |

A render-conditioned continuation without a pool (B token, B′) does not redraw. gmc3 does, at the boundary only: the damage
comes from a second pass over the same on-policy pool, which is why stage C bounds the exposures per pool item.

## Does it execute the command?

ViPE trajectories against the plan; execution = measured displacement projected on the planned one (1 = executed).

| Run | execution @ frame 160 | @ 320 | lateral @ 320 | yaw ratio @ 320 |
|---|---|---|---|---|
| B token, normal speed | 1.05 / 0.97 | 1.07 / 1.02 | 0.03 / 0.00 | 1.02 / 1.01 |
| B channel, normal speed | 0.36 / 0.67 | 0.57 / 0.78 | 0.02 / 0.03 | 0.67 / 0.94 |
| stage A, normal speed | 0.03 / 0.19 | 0.05 / 0.16 | 0.04 / 0.10 | 0.04 / 0.18 |
| B token, every 4th pose (@ 112) | | 1.08 / 0.98 | 0.01 / 0.01 | 1.02 / 1.02 |
| stage A, every 4th pose (@ 112) | | 0.98 / 1.35 | 0.09 / 0.11 | 0.90 / 0.92 |

Without the render the model executes large steps but stands still on slow commands (its low regeneration above is this).
With the render as tokens it follows the slow command.

## Does it return to what it generated?

Out-and-back plan: the camera follows the recorded path and retraces it; frames at the same pose on the way out and back are
compared. "Matches" = geometrically consistent SIFT matches between the two frames, divided by the matches between two
neighbouring frames of the same video (1 = as consistent as a neighbouring frame).

| Model, protocol | LPIPS | camera shift (px) | matches |
|---|---|---|---|
| B token, 30 calls | 0.041 / 0.126 | 0.7 / 2.5 | 0.69 / 0.45 |
| GM′, 30 calls | 0.080 / 0.365 | 1.3 / 8.6 | 0.45 / 0.03 |
| GM′ + revisit bank, 30 calls | 0.050 / 0.311 | 0.9 / 16.8 | 0.70 / 0.19 |
| Memorizon as released, 30 calls | 0.033 / 0.129 | 0.3 / 4.3 | 0.66 / 0.50 |
| B token, every 4th pose, 14 calls | 0.049 / 0.113 | 0.9 / 2.1 | 0.78 / 0.76 |
| GM′ + revisit bank, every 4th pose, 14 calls | 0.116 / 0.258 | 1.1 / 2.2 | 0.61 / 0.35 |

- Aligning the two frames does not lower B token's LPIPS: its consistency comes from the content, not from the camera
  landing at the same place. Its match ratio does not drop at the longest gaps (0.72 / 0.44); Memorizon's does (0.58 / 0.31).
- **B token really goes back.** Execution on the way out 1.05 / 0.99 and on the way back 1.05 / 0.99; at the end it is
  within 1 % of the path length from the start. Memorizon executes about a third of the command, so its return path is
  short and its consistency is flattered.

## First calls

LPIPS over calls 1–3: B token 0.021 / 0.064, GM′ 0.024 / 0.070, B′ 0.031 / 0.081, B channel 0.084 / 0.113, stage A
0.059 / 0.169, plain-350 0.154 / 0.198. Vertical / horizontal scale ratio of B token: 1.000–1.002 up to frame 128
(showroom), 0.978 at frame 160 (playground).

## Brightness and sharpness

**Brightness is measured against the anchor**, at SIFT-matched scene points (frame − anchor). The recording's own exposure
moves along the path (showroom −6, playground +12 against the anchor), so a comparison with the recording mixes the two.
Values are thirds of a 30-call run (the playground's last third has too few matches).

| Model, 30 calls | showroom | playground |
|---|---|---|
| recording | −1.0 / −3.9 / −5.9 | +3.1 / +11.9 |
| B token | −1.5 / −6.8 / −10.2 | +0.1 / −0.6 |
| B channel | −1.6 / −10.9 / −18.7 | −0.4 / −1.2 |
| B′ | +10 / +18 / +19 | +10 / +16 |
| GM′ | +6 / +11 / +19 | +15 / +25 |

B token is the photometrically most stable model; B′ and GM′ brighten by 15–25 levels.

**Sharpness by region** (Laplacian energy, generated / recorded at the same pose; cells the anchor sees | cells the model
generated; last third of a 30-call run): B token 0.79 | 0.44 (showroom), 0.98 | 0.99 (playground); B′ 1.00 | 0.88;
GM′ 1.31 | 1.07; B channel 1.92 | 1.62 (over-sharpened). B token's showroom blur sits in the generated cells and does not
recover, while B′ recovers from 0.39 to 0.88.

## What the contract changes do (playground, every 4th pose, 7 calls, one run each)

- **Bank off:** the slide breaks up more in the last two calls. The bank helps identity.
- **Density mask off:** an object that is not in the scene enters and the slide tears. The mask fixes the first generation at
  the edge; on the showroom the smeared column that earlier models drew at the right edge is gone. With the mask off the
  generated cells are somewhat sharper in the last third (0.88 against 0.75): masked cells are generated softer.

## Diagnostics on B token (showroom, one run each)

- **Moment patch** (statistics of the context matched to the anchor at inference): brightness follows the recording's own
  curve (+0.3 / −2.8 / −6.7) and generated-cell sharpness rises from 0.44 to 0.77. Part of the darkening and blur is a drift
  of moments carried by the context. The patch as applied adds a jump at call starts (1.81 against 0.94), so it is a
  diagnosis, not a fix.
- **Text guidance 1.5 and 2** (A and B were trained with 15 % text dropout): the image holds, generated-cell sharpness rises
  (0.64 → 0.77 → 1.03) without over-sharpening rendered cells, but brightness drifts by +16 / +31 against the anchor and
  saturated objects appear at the edges. Not usable as is.

## Open defects of B token

- **Wrongly enlarged distant content.** Thin objects far from the anchor (playground equipment a few pixels wide) are drawn
  as white wire-frames when the camera approaches, and then come back through the render at full confidence. It is not the
  bank and not the mask (both switched off: still there). B′ has the same objects; GM′ draws a faint frame and drops it. The
  difference is the on-policy stage, hence stage C.
- **Blur in generated cells** on the showroom (0.44 of the recording after 30 calls), not recovering.
- **Long chains at full frame rate.** In 30-call runs at the recorded frame rate the showroom floor turns brown after frame
  320 and coloured blocks appear on the right wall; the playground structure degrades after frame 320. Not at every 4th
  pose. This is a limit as long as the delivery layout is the motion-targeted step.
- **Windows far from the anchor.** Beyond ~10 model units the camera branch's logits grow with the distance. The test plans
  stay at or below 6 units.

## Stage C (running)

From B token @ 250: on-policy pool rows (the model's own call from a recorded history, at the contract windows, in the
inference layout), at most 1.5 exposures per pool item, 100 updates. Expected: the stray wire-frames gone and generated-cell
sharpness above 0.7. Stop rules: regeneration outside 1 ± 0.3, sharpness ratio above 1.3, worse return consistency,
brightness drifting against the anchor, execution away from 1.

## Training

| Run | updates | loss (50-update means) | s / update | peak GiB | rows rejected |
|---|---|---|---|---|---|
| stage A | 350 | 0.329 → 0.270 | 44 | 24.0 | (no rejection) |
| B token | 250 | 0.271 / 0.261 / 0.255 / 0.251 / 0.254 | 65 | 24.6 | 12 of 4,000, all far windows |
| B channel | 250 | 0.267 / 0.268 / 0.262 / 0.259 / 0.261 | 48 | 24.0 | 12 of 4,000, all far windows |

The channel arm started lower (first 6 updates 0.273 against 0.298 on the same rows: at its first update it is exactly
the stage-A model) and ended 0.007 higher.

## Earlier model on the comparison cohort

DL3DV cohort of the closed-loop study, 5 scenes, 480 frames, scorer `vipe_native_map_mask_diagnostic_v3`
(`docs/lab_notes/comparison_table_20261105.md`; "Ours" = Gauss-meas, the model before the contract line):

| Method | Position err ↓ | Endpoint ↓ | Rotation ° ↓ | LPIPS ↓ | DINO ↑ | LPIPS last⅓ ↓ | DINO last⅓ ↑ |
|---|---|---|---|---|---|---|---|
| Ours | 0.0505 | 0.154 | 2.54 | 0.378 | 0.855 | 0.552 | 0.743 |
| Ours + correction | 0.0336 | 0.093 | 0.85 | 0.356 | 0.853 | 0.521 | 0.733 |
| Lyra-2 | 0.0732 | 0.187 | 0.17 | 0.333 | 0.925 | 0.484 | 0.880 |
| Lyra-2 + correction | 0.0333 | 0.068 | 0.24 | 0.297 | 0.936 | 0.416 | 0.898 |
| Spatia | 0.0749 | 0.214 | 2.60 | 0.381 | 0.906 | 0.550 | 0.837 |
| Matrix-Game 3.5 Distilled | 0.1952 | 0.554 | 3.47 | 0.559 | 0.798 | 0.712 | 0.633 |

That model led on camera position over the first 160 frames and lost on appearance afterwards; the contract line targets
that drift. Cohort numbers for the contract line do not exist yet.
