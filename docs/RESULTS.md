# Results

Two development scenes (`02267acf` showroom, `0238de14` playground), seed 7, one run per number unless stated. The
5-scene comparison cohort is held back for the final validation; its last table (the earlier Gauss-meas model) is at the
end. Every number here is in `docs/lab_notes/RECIPE_contract_20261111.md` with the date it was measured.

Models: **stage A** (`a4_s2` @ 350, no render), **B token** (`b4_token_s2` @ 250), and for reference **plain-350**
(Memorizon recipe, no render), **B′** (corrected render, no on-policy stage: the fair reference for B token), **GM′**
(B′ + on-policy pool), **gmc3** (GM′ + 250 more updates on the same pool).

## Does the model redraw the scene at call boundaries?

Change of generated cells relative to the recording, 20 calls at normal speed, showroom / playground. 1 = as much change
as the recording; above 1 the model redraws, below 1 it stands still.

| Model | inside a call | across a boundary | boundary, distance from 1 |
|---|---|---|---|
| B token | 0.94 / 0.97 | 1.05 / 1.00 | 0.05 / 0.00 |
| B′ | 1.07 / 0.94 | 1.10 / 0.98 | 0.10 / 0.02 |
| GM′ | 0.91 / 1.08 | 1.04 / 1.14 | 0.04 / 0.14 |
| stage A (no render) | 0.29 / 0.68 | 0.32 / 0.58 | 0.68 / 0.42 |
| plain-350 (no render) | 1.67 / 1.52 | 1.91 / 1.70 | 0.91 / 0.70 |
| gmc3 | 1.17 / 1.17 | 2.63 / 2.02 | 1.63 / 1.02 |

A render-conditioned continuation without a pool (B token, B′) does not redraw. gmc3 does, at the boundary only: the damage
comes from the second pass over the same on-policy pool, which is why stage C bounds the exposures per pool item.

## Does it execute the command?

ViPE trajectories against the plan; execution = measured displacement projected on the planned one (1 = executed).

| Run | execution @ frame 160 | @ 320 | lateral @ 320 | yaw ratio @ 320 |
|---|---|---|---|---|
| B token, normal speed | 1.05 / 0.97 | 1.07 / 1.02 | 0.03 / 0.00 | 1.02 / 1.01 |
| stage A, normal speed | 0.03 / 0.19 | 0.05 / 0.16 | 0.04 / 0.10 | 0.04 / 0.18 |
| B token, every 4th pose (@ 112) | | 1.08 / 0.98 | 0.01 / 0.01 | 1.02 / 1.02 |
| stage A, every 4th pose (@ 112) | | 0.98 / 1.35 | 0.09 / 0.11 | 0.90 / 0.92 |

Without the render the model executes large steps but stands still on slow commands (its low regeneration above is this).
With the render it follows the slow command.

## Does it return to what it generated?

Out-and-back plan, 30 calls: LPIPS between the way out and the way back at the same pose (all cells / generated cells).

| Model | showroom | playground |
|---|---|---|
| B token | 0.041 / 0.045 | 0.126 / 0.158 |
| GM′ | 0.080 / 0.089 | 0.365 / 0.405 |
| GM′ + revisit bank | 0.050 / 0.051 | 0.311 / 0.352 |
| Memorizon as released | 0.033 / 0.021 | 0.129 / 0.124 |

Memorizon executes about a third of the command, which flatters this measure; B token executes all of it. Execution on the
way back was not measured.

## First calls

LPIPS over calls 1–3: B token 0.021 / 0.064, GM′ 0.024 / 0.070, B′ 0.031 / 0.081, stage A 0.059 / 0.169, plain-350
0.154 / 0.198. Vertical / horizontal scale ratio of B token: 1.000–1.002 up to frame 128 (showroom), 0.978 at frame 160
(playground).

## What the two contract changes do (playground, every 4th pose, 7 calls)

- Bank off: the slide breaks up more in the last two calls. The bank helps identity.
- Density mask off: an object that is not in the scene enters and the slide tears. The mask fixes the first generation at
  the edge; on the showroom the smeared column that earlier models drew at the right edge is gone.

## Open defects of B token

- **Wrongly enlarged distant content.** Thin objects far from the anchor (playground equipment a few pixels wide) are drawn
  as white wire-frames when the camera approaches, and then come back through the render at full confidence. B′ has the same
  objects; GM′ draws a faint frame and drops it. The difference is the on-policy stage, hence stage C.
- **Darkening.** Luminance in anchor-visible cells −5 / −12 after 20 calls, −6 / −20 after 30 (B′ and GM′ brighten).
- **Blur on the showroom.** Sharpness 0.69 of the recording after 20 calls.
- **Windows far from the anchor.** Beyond ~10 model units the camera branch's logits grow with the distance. The test plans
  stay at or below 6 units; one fast development scene of the earlier line reached 15.7 and collapsed there in every model. Cause
  not established for that scene (it is also the fastest, with the least consistent pose labels).

## Training

| Run | updates | loss (50-update means) | s / update | peak GiB | rows rejected |
|---|---|---|---|---|---|
| stage A | 350 | 0.329 → 0.270 | 44 | 24.0 | (no rejection) |
| B token | 250 | 0.271 / 0.261 / 0.255 / 0.251 / 0.254 | 65 | 24.6 | 12 of 4,000, all far windows |
| B channel | first 6 | 0.273 mean (token: 0.298 on the same rows) | 47 | 24.0 | 0 so far |

The channel arm starts lower because at its first update it is exactly the stage-A model without render; the token arm puts
four slots of a new role into the sequence. The arms are compared at the end of training.

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
that drift. Its cohort numbers do not exist yet.
