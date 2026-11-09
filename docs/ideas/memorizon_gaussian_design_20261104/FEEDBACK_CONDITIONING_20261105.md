# Proposed feedback-conditioned geometric memory read

**Current consolidated plan:** [METHOD_AND_TRAINING.md](METHOD_AND_TRAINING.md)
includes this proposal, the conditional reward comparison, training schedule and
novelty comparison. The [preference audit](PREFERENCE_AUDIT_20261105.md) qualifies
the representation, temporal addressing, capacity and implementation assumptions;
follow the consolidated design where this earlier proposal differs.

5 November 2026. Design extension requested by the user; unimplemented and
unverified. Retains the 72 GPU-hour budget, frozen 2Xplat, native Memorizon latent
bank and RGB/depth/coverage branch. The generator training objective is native
flow matching, not the historical final-output L1 pilot.

## Prioritize two small additions

### 1. Causal execution-history conditioning

At an accepted observation boundary, retain planned pose P, actually issued pose
U, and independently measured pose M at the SAME timestamp. With camera-to-world
transforms in the same calibrated gauge, encode log(M^-1 P) as task tracking error
and log(M^-1 U) as command execution error. Translation and rotation have separate
normalization; translation uses the established scene gauge, never a denominator
from nearly stationary planned motion. Include a short past motion/error history,
measurement validity, age and geometric support. Transport vectors to a common
local frame before aggregating across timestamps. Missing measurement is a mask,
not a fabricated zero error. Do not confuse tracking error with estimator error.

Feed the compact vector through an MLP to zero-initialized residual modulation
of the added memory/camera adapter. Freeze the backbone initially. U remains the
single authoritative camera request for the next chunk; feedback is context for
executing U, not another camera displacement to add to the controller's output.
Hold the last available feedback through chunk denoising. Never condition a
target on its own measured outcome, including during training.

Current measured history plus future commands already encode much of the
geometric residual. Explicit errors are an inductive bias, not automatically new
information. Past issued-versus-realized motion may reveal execution behavior,
but passive ratios are not identified causal gains. The comparison must include
the same measured poses and corrected commands without the feedback embedding.

The nonzero feedback branch needs actual generated-history recovery examples.
Clean recordings with zero execution error cannot train it adequately. Record
errors when generating each prefix; preserve camera-compatible next-chunk targets.
Never estimate feedback using the supervised future, invent realized motion by
changing labels on unchanged images, or align measurement scale to the commands.

### 2. Bounded geometry-guided cross-attention

Use noisy target tokens as queries and retained source appearance tokens as
keys/values. Project supported source surfaces into the requested cameras using
2Xplat geometry and measured source poses. This supplies correspondence,
visibility and support to a small cross-attention adapter. Values remain original
appearance features; no persistent latent-Gaussian representation is required.

Keep the native backbone attention unchanged. Use bounded source selection and
low-width projections in a few added blocks; do not attend densely to every
Gaussian. Following the GEAR ablation review, restrict the added read to supported
geometric candidates, with learned weights inside that set. Keep native attention
available and return zero from an empty added read. Uncertainty may justify a
bounded candidate footprint, not unrestricted history access. Temporal
VAE tokens cover multiple RGB frames: aggregate their projected footprints or
retain their temporal support instead of declaring each token a point sample.

Condition the residual read gate on execution history AND local geometric support.
The testable hypothesis is that the generator can learn when mapped structure is
helpful during recovery and when unreliable correspondence should yield to native
appearance context. A large tracking error alone must not imply low scene quality.
Keep RGB/depth conditioning as spatial evidence and native latent memory as the
appearance path. Do not add a Gaussian self-attention/updater network in this run.

## Flow matching and causality

For the convention z_t=(1-t)z+t*epsilon, regress epsilon-z with squared error and
the native timestep weights, on target tokens only. The implementation must call
the existing scheduler's noise/target/weight methods rather than assume signs or
timestep indexing. Preserve per-query-chunk noise and clean causal context.

Condition the velocity prediction on text, issued target cameras, measured prefix,
past feedback and geometric evidence. Gradients train the new reader/modulation;
detach observer measurements and reconstruction initially. There is no separate
pixel L1 objective and no per-noise-step pose-estimator reward in the first run.
Do not mask away uncovered target regions from the FM loss: the model must still
learn plausible completion there. Geometry support gates evidence, not whether
an output is scored. Flow-matching MSE is a velocity objective, not direct final
image regression. This recipe is for the undistilled teacher.

Correction changes issued cameras. Recorded pixels at different cameras remain
invalid training targets; use qualified camera-compatible supervision as required
by the main training decision. Flow matching does not remove that constraint.

## Related work and evidence needed

[WorldStereo](https://arxiv.org/abs/2603.02049) uses geometric correspondence to
guide memory attention; [Geometry as Address](https://arxiv.org/abs/2609.34722)
routes historical features to target patches; [GLIMPSE](https://arxiv.org/abs/2610.13706)
addresses uncertain projected memory locations. Thus generic geometric attention
or soft uncertainty weighting alone is not a new contribution.
[Robust Dreamer](https://arxiv.org/html/2605.30855) establishes corrupted-history
training with Gaussian latent memory. Our proposed combination remains a research
hypothesis, not a verified first claim.

The narrow candidate contribution is learning how measured execution history
conditions use of spatial memory during corrected generation. Establish value
against the same P/U/M inputs, same reconstruction, same controller and training
budget without the explicit feedback embedding. Also compare geometric versus
ordinary bounded memory attention. Zeroing or shuffling feedback at inference
checks reliance but is not a substitute for matched training controls.

Implementation priority is feedback modulation first, then bounded cross-attention
only if its GPU qualification fits the existing preparation allocation. No new
representation sweep or extra GPU budget is implied. Preserve the base teacher
and saved adapter states so an unsuccessful addition does not erase progress.
