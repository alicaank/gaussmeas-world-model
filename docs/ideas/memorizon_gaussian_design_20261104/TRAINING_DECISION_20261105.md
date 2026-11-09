# Deadline-constrained training decision

**Latest revision:** [design after the drift tests](DESIGN_AFTER_DRIFT_TESTS_20261105.md).
The 2Xplat-first and feedback-first allocations below are historical; the current
priority is DA3/ReSplat recovery training from the adapted checkpoint.

**Current consolidated plan:** [METHOD_AND_TRAINING.md](METHOD_AND_TRAINING.md).
Its Section 9 refines the same 72-hour total to include early imperfect-history
training, one matched feedback control and a conditional weighted/uniform replay
pilot. The [preference audit](PREFERENCE_AUDIT_20261105.md) explains the changes.
This note records the initial, superseded budget sequence.

5 November 2026. Recommendation following the user's request to concentrate
limited time on one strong model. This is a design decision, not a launched run
or a validated performance claim. The user confirmed approximately 72 GPU-hours
for this training effort. Existing evaluation jobs and protected checkpoints are
untouched. This is an allocation, not a claim that a run is queued or launched.

## Confirmed GPU allocation: 72 hours

| Allocation | GPU-hours | Purpose |
| --- | ---: | --- |
| Qualification and compact data preparation | 6 | Check checkpoint/resolution, causal 2Xplat outputs and units, zero-adapter parity, actual backward and save/reload; measure throughput |
| Main teacher adaptation | 42 | Native flow matching with measured-history Gaussian conditioning; adapter-first curriculum |
| Generated-history recovery | 12 | Collect short correction-enabled histories and train on camera-compatible, qualified supervision |
| Held-out evaluation | 8 | Fixed scenes/seeds; native reference, trained model, memory-use diagnostics and failures |
| Contingency | 4 | Failed-job recovery or a targeted fix; preserve evaluation allocation |

All GPU preprocessing, teacher sampling and validation count toward this budget.
CPU implementation time and waiting behind existing jobs are separate wall time;
72 GPU-hours do not imply delivery in 72 elapsed hours. Avoid running a second
large training architecture. Set update counts only after measuring step time
with the final resolution and context layout. Reserve evaluation and recovery
time rather than using the entire allowance on the initial loss curve.

If qualification exceeds its allocation, diagnose before committing the main
training budget. If correction-compatible targets are not qualified, do not
spend the recovery allocation on invalid targets or call the initial checkpoint
fully closed-loop-trained. Report the missing stage explicitly. Reallocation is
based on observed readiness and quality, not an automatic deadline shortcut.

## One architecture

The user's follow-up proposes past pose-error conditioning and 3D-guided attention.
The bounded extension is specified in [feedback conditioning](FEEDBACK_CONDITIONING_20261105.md):
prioritize a small execution-history modulation branch, and qualify a compact
source-memory cross-attention reader within the existing preparation budget.
This does not add persistent latent Gaussians or a learned scene updater. The
generator objective is native flow matching; final-output L1 is excluded.

Use an undistilled Memorizon teacher with its original latent bank, plus frozen
2Xplat reconstruction and an RGB/depth/coverage conditioning adapter. Both memory
paths use accepted measured cameras. Retain camera correction in the complete
rollout. The native bank retains original appearance latents; the Gaussian branch
provides spatially aligned evidence. This is not RGB-only memory.

This supersedes the earlier suggestion to prioritize a new feature-splat reader
for the first deadline-constrained training run. That reader and a learned
pose/map-repair network introduce separate unqualified learning problems. Keep
them as research options, not prerequisites for the first useful checkpoint.
Do not describe native camera-frustum retrieval as a new surface-token reader.

Keep 2Xplat weights frozen and retain source observations for reconstruction
refreshes. Improve state through measured registration, overlap with accepted
keyframes and validated updates. Do not promise that this already constitutes
learned persistent scene repair. Geometry must stay tied to its measured gauge;
no fit to commanded displacement and no silent scale-one fallback.

## Training objective and curriculum

1. Adapt the undistilled teacher using native flow matching with causal measured
   history and the Gaussian condition. Recorded target poses label recorded
   targets; measured history labels are estimated from prefix images only.
   Preserve native masks, noise conventions and VAE timestamps. Train the new
   zero-initialized residual adapters first, with the pretrained model frozen.
2. Sample genuine moving trajectories, departures and returns beyond recent
   context. Use bounded retrieved context rather than densely unrolling every
   intervening frame. Include partial/absent geometry so the pretrained image
   prior remains usable. Prioritize useful scene coverage over repeatedly
   optimizing a few stationary clips. Use only train scenes for adaptation.
3. Introduce short current-model histories with measured writes and correction
   active. Record the exact issued target cameras. Corrected targets require
   camera-compatible real/synthetic supervision or qualified teacher samples
   generated at those cameras from the same causal history. A frozen native
   teacher can provide model supervision for sparse recovery examples, but those
   targets require visual qualification and are not ground truth. Mix with real
   examples rather than replacing them. Preserve the target-camera rejection
   guard; an error archive cannot waive it.
4. Continue the same model only when held-out video supports it. Small backbone
   LoRA is an escalation if adapters underfit, not a second speculative branch.
   Defer four-step distillation until the teacher is useful. No final-output L1
   shortcut or three-network DMD assumption on the available 32 GB GPU.

Initial adapter optimizer values may reuse the existing pilot's 1e-4 learning
rate, BF16 base/FP32 trainable state and clipping at 1; these are starting settings,
not validated optima. Use one four-latent query and bounded native context for the
resource probe. Measure step time and memory before setting a run duration.

## Minimal decision evidence

Use the same held-out scene/seed set before and after training. Include motion,
return after context eviction, weak geometry and stationary controls. Judge
appearance, scene survival, revisit consistency and camera following together.
Do not choose a checkpoint from training loss or pose error alone.

Necessary controls are the unchanged native model and disabling the learned
Gaussian branch, with a matched data-adaptation control before a paper claim.
Memory-off inference is only a usage diagnostic, not a substitute for matched
training. This narrow evaluation is not a representation sweep.

## Readiness checked this turn

- The existing Gaussian adapter path is present, but its optimizer is a four-step
  final-output L1 pilot. The native FM entrypoint excludes Gaussian checkpoints.
  A Gaussian-enabled FM path and causal measured-history loader still need wiring.
- The local `memorizon50/config.json` identifies Memorizon, 50 sampling steps,
  1280x704, and the Wan2.2 TI2V-5B backbone. Three weight shards are listed locally;
  this review did not hash/load them or verify an adapted descendant checkpoint.
  The 480p recipe is therefore not yet qualified against this checkpoint.
- No 2Xplat backend is integrated in `measured-gaussian-memory`.
- Resource snapshot: RTX 5090, 32607 MiB total, 24838 MiB occupied, 100% utilization;
  filesystem approximately 19 GB free. Do not launch alongside that workload or
  allocate a full checkpoint/cache budget without a size estimate. Adapter-only
  saves must retain optimizer/RNG state and an immutable base identity for resume.

Immediate execution order: qualify the chosen checkpoint/resolution, integrate
the frozen 2Xplat adapter and Gaussian FM objective, check zero-adapter rollout
parity plus backward/save/reload, then spend the available GPU allocation on this
single curriculum. Do not label a native-only training job as Gaussian training.

References: [Memorizon](https://arxiv.org/html/2610.00544),
[Robust Dreamer](https://arxiv.org/html/2605.30855). Both motivate preserving
appearance memory and training on useful histories; neither validates our recipe.
