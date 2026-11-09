# GaussVerse research and execution plan

Status: 2026-07-29

This plan consolidates the implementation audit, the measured adapter and Q/K
fine-tuning results, the CameraFiLM/ray/correspondence studies, the warp-memory
failures, the recent-method comparison, and the autoregressive rollout diagnosis.

The order is intentionally evidence-gated. Do not build the entire final system
before establishing that each additional mechanism has a measurable oracle gap.

**Status update, 2026-07-30.** Phase 0 is complete. Phase 1 Gate 1 (memory harm)
and Gate 3 (utility is not geometric confidence) pass, but the cross-fitted Gate 2
oracle gap is only about 1% at its best timestep and therefore fails the usefulness
gate for warped-RGB v1 memory. Scene/action-level flow utility does predict
independently rendered PSNR and LPIPS-style VGG quality, so the failure is not
explained by an invalid denoising label. Do not train the immediate router on this
action space. Clean source-separated features may reopen Gate 2, but must repeat the
same oracle and label-validation protocol before router work resumes.

## 1. Research thesis

The paper should not claim geometric attention, clean feature memory, a
copy/route/imagine gate, or observed/provisional/verified memory as independent
novelties. Close variants of all of these now exist.

The primary claim should be:

> Memory use in a generative world model is an intervention. The correct read or
> write action is the one that causally improves generation while minimizing future
> memory contamination, not necessarily the one with the highest geometric
> confidence.

Working name: **Counterfactual Memory Transport**.

For a target region \(r\), diffusion time \(t\), and matched target/noise/backbone,
evaluate:

- \(a_0\): no memory;
- \(a_g\): geometry-only routing;
- \(a_a\): geometry plus clean appearance transport.

Define action risk and utility:

\[
R_a(r,t)=\lVert v_a(r,t)-v^*(r,t)\rVert^2,\qquad
\tau_a(r,t)=R_{a_0}(r,t)-R_a(r,t).
\]

The training-only oracle selects the action with the highest positive utility and
otherwise abstains. A target-free router learns to predict these utilities from
inference-time evidence. The long-horizon extension scores both the current frame
and the damage an action can introduce into future memory.

Supporting mechanisms:

- probabilistic point-to-epipolar-to-null geometric attention;
- source-separated clean feature transport;
- commanded-versus-realized camera state;
- fixed typed context;
- conservative, falsifiable persistent writes;
- scheduled beta-only CameraFiLM;
- short detached self-rollout training.

The supporting mechanisms matter, but they are not the headline novelty.

## 2. Current evidence and constraints

1. The current five-step rollout is visually coherent but geometrically inaccurate.
   It does not yet exhibit classic blur or collapse. The first failure to fix is
   pose-state correctness, not generic temporal smoothness.
2. A generated view can be sharp and self-consistent while being far from its
   commanded pose. Treating the command as the realized camera corrupts every later
   unprojection, ray map, retrieval, and write.
3. Warped evidence is often already broken before it reaches the model. More
   training on the same forced-copy objective can teach the model to reproduce the
   broken warp more faithfully.
4. Two-sided coordinate embeddings improve frontier fidelity but are not a clean
   camera-control win. They remain a baseline, not the research destination.
5. The input-level ray planes have a small but measured fidelity contribution.
   Do not simply remove them. Repurpose or ablate their role after a replacement is
   working.
6. Small 2--4 M adapters saturate around 3--4 k steps and can degrade afterward.
   This result must not be transferred to a 0.566 B Q/K fine-tuning tier.
7. Q/K unfreezing is the first backbone intervention that causally improves camera
   control and fidelity against a matched frozen control.
8. The present `qk_batch16` log reaches step 3000. Its fidelity tripwire rises above
   baseline early and remains positive; it does not show catastrophic forgetting.
   The run metadata records only `{"step": 3000}`, so its exact configuration must be
   reconstructed before using the directory name as scientific evidence.
9. Iterations are not comparable across global batch sizes. If `qk_batch16` truly
   used effective batch 16, 3000 steps are 48,000 samples--the same exposure as
   12,000 steps at effective batch 4.
10. Pose evaluation on the 480P held-out expansion has high SIFT/Sampson variance.
    Use the original 960P set for pose until the expanded scenes are available at
    equivalent resolution. Use all 60 for fidelity.

## 3. Phase 0 -- freeze a reproducible baseline

### Work

1. Create a run manifest for every candidate checkpoint:
   base checkpoint, git commit, trainable modules, optimizer, learning-rate
   schedule, global batch, scene roots, tuple mixture, seed, resolution, number of
   samples, and checkpoint step.
2. Treat any checkpoint whose configuration cannot be recovered as an exploratory
   result, not a paper result.
3. Evaluate the available Q/K checkpoints at matched protocol:
   `qk_batch16` step 2000 and 3000, the 12k small-batch run, the matched frozen
   control, and the camera-Q/K knockout.
4. Report:
   paired PSNR on 60 scenes; pose on the original 16 960P scenes; covered/frontier
   PSNR; camera binding; fidelity tripwire; and 5/10-step rollout behavior.
5. Save intermediate checkpoints every 500 steps in future runs. A training-loss
   trace alone is not a selection metric.

### Decision

- If step 3000 does not improve the task metrics over step 2000, stop or lower the
  learning rate; do not extend because the loss is noisy.
- If it still improves and fidelity is intact, continue in staged sample budgets,
  not directly to a monolithic 10x run.

### Deliverable

One checkpoint-selection table and one immutable baseline manifest used by every
later experiment.

## 4. Phase 1 -- falsify the counterfactual-utility hypothesis cheaply

Use the existing branches before changing the architecture.

### Work

1. Add a counterfactual evaluator that runs the same tuple, target latent, diffusion
   timestep, and noise under:
   no memory, correspondence only, memory only, and memory plus correspondence.
2. Produce token/patch utility labels from the flow-matching error. Also render a
   smaller final-image subset to verify that denoising utility predicts actual
   PSNR/LPIPS and not only the training objective.
3. Record the existing inference-time evidence beside each label:
   reprojection cycle error, depth uncertainty, depth discontinuity, source count,
   multi-view agreement, view angle, visibility, coverage, provenance, and camera
   baseline.
4. Add controlled hard negatives:
   pose perturbation, scale error, depth bias/noise, occlusion, moving objects,
   reflective cars, source corruption, and unsupported frontier regions.
5. Compare geometric confidence with counterfactual utility. The key evidence for
   the paper is a set of high-confidence but harmful reads and low-confidence but
   useful reads.

### Predeclared gates

- Memory must be measurably harmful on a nontrivial subset of patches/tuples;
  otherwise there is no selective-memory problem to solve.
- The oracle must beat the best fixed action by a useful margin on held-out scenes.
- Utility must not be reducible to a monotonic transform of existing confidence.
- Inference-only evidence should predict utility above a simple confidence gate
  before a larger router is trained.

If the oracle gap is small, stop. A learned router cannot exceed an uninteresting
oracle.

## 5. Phase 2 -- correct rollout state semantics

This phase is required before online writes.

### State carried by every frame

- `T_commanded`: requested camera, used for camera conditioning and binding loss;
- `T_realized`: estimated camera actually depicted by the generated image;
- pose covariance / observability;
- depth and depth uncertainty;
- source type: observed or generated;
- generation lineage and parent IDs;
- memory status: observed, provisional, verified, or rejected;
- timestamp and spatial support.

Initially estimate `T_realized` offline with joint multi-view VGGT or GLOMAP,
aligned to immutable observed anchors. A learned verifier can replace this later.

### Rules

1. Use `T_commanded` to tell the generator what to make.
2. Use `T_realized` for unprojection, target rays, retrieval, spatial indexing, and
   memory writes.
3. If the realized pose is unobservable or too far from the command, the frame may
   remain recent visual context but must not be spatially committed.
4. Never silently substitute the commanded pose when realized-pose estimation
   fails.

### Fixed typed context

Replace the current 3-to-8 growing rollout context with four fixed slots:

1. one immutable observed anchor or best observed revisit;
2. recent generated frame 1;
3. recent generated frame 2;
4. one nonredundant spatial retrieval.

Use explicit role embeddings. Recent slots receive ordered temporal RoPE; observed
anchors and spatial retrievals use the negative/identity temporal sentinel. A
missing role receives a learned null, not an extra generated frame.

### Gates

- Context memory and runtime remain constant from step 1 to step 50.
- Slot permutation behavior matches the declared roles.
- Realized-pose warping lowers reprojection error versus commanded-pose warping.
- A deliberately pose-wrong generated frame cannot enter spatial memory at the
  commanded location.

## 6. Phase 3 -- build the geometry compiler

Direct geometric attention is a transport mechanism, not the main novelty.

### Work

1. For each target token and each source frame, predict a geometric support
   distribution:

   \[
   p(j\mid i,G) \propto
   \exp(q_i k_j^\top/\sqrt d)\,p_{\rm geo}(j\mid i).
   \]

2. Let depth and pose uncertainty make \(p_{\rm geo}\) naturally become:
   a narrow point kernel, a broad epipolar kernel, or a null/unconstrained action.
3. Preserve source identity. Do not fuse candidates from different frames before
   the router sees them.
4. Return explicit source addresses:
   `(source_id, u, v, depth, uncertainty, visibility, provenance)`.
5. Implement sparse/top-K or block-sparse support so the feature does not require a
   dense all-token geometric matrix.
6. Reuse the Plücker/ray inputs in this compiler. Keep the current residual ray path
   until an ablation shows that the compiler replaces its measured fidelity value.

### Tests

- exact projection and epipolar tests on synthetic cameras;
- uncertainty smoothly widens point support into epipolar support;
- invalid geometry reduces to the declared null/free behavior;
- geometry changes Q/K routing but never injects appearance through V;
- source hypotheses remain distinct through routing;
- bounded runtime and memory at training resolution.

### Gate

The geometry-only action must improve correspondence accuracy or denoising utility
over two-sided coordinate embeddings without increasing negative-transfer rate.

## 7. Phase 4 -- clean, source-separated appearance memory

### Work

1. Encode each source frame once.
2. Store clean VAE spatial features or a selected encoder feature, with source
   address, view direction, uncertainty, pose state, and provenance.
3. Gather clean source neighborhoods at read time using the geometry compiler.
4. Keep candidates from different sources separate until routing.
5. Retain the current warped-RGB-to-VAE branch only as an explicit baseline.
6. Prevent one generated frame from appearing simultaneously as a recent context
   frame and a spatial retrieval in the same prediction.

### Gate

- The clean-appearance oracle must exceed the warped-RGB oracle.
- It must reduce the reflective-car, boundary, and disocclusion artifacts already
  observed.
- No-overlap/frontier regions must safely abstain rather than copy unrelated
  features.

If clean features do not raise the oracle ceiling, do not proceed to a large router.

## 8. Phase 5 -- train the immediate counterfactual router

### Router output

Predict per-source, per-region utilities for:

- abstain / imagine;
- geometry-only route;
- geometry plus clean appearance.

Predict utility and uncertainty, not only a three-class label. The policy selects
the highest lower-confidence-bound utility and abstains if none is positive.

### Inputs

Only inference-available features:
cycle error, pose/depth covariance, source agreement, support count, view angle,
occlusion, source age, observed/generated provenance, lineage depth, diffusion
timestep, layer, and current model-state features.

### Training

1. Cache counterfactual labels where practical.
2. Train with utility regression, pairwise ranking, and abstention/calibration
   losses.
3. Balance easy positive reads with corrupted and high-confidence harmful reads.
4. Start with the backbone frozen.
5. Co-adapt Q/K only after the frozen-router mechanism passes its oracle-gap gate.

### Gates

- fraction of oracle gap closed;
- negative-transfer rate versus a geometric-confidence router;
- AUROC/AUPRC for harmful reads;
- calibration and risk-coverage curve;
- held-out corruption generalization;
- covered/frontier fidelity and camera binding.

The principal baseline is not “memory always on.” It is a well-tuned
geometry-confidence gate. Counterfactual utility must beat it.

## 9. Phase 6 -- simplify and strengthen camera control

This is an auxiliary camera branch, not the memory-paper headline.

### Work

1. Produce a CameraFiLM layer-by-diffusion-time causal knockout map.
2. Test beta-only FiLM; retain gamma only if it has independent measured value.
3. Split the pose code into rotation and translation using
   \(\log(T_{\rm rel})=(\omega,v)\).
4. Activate camera conditioning only at layers/timesteps supported by the causal
   map.
5. Add a counterfactual binding margin:
   the true camera must have lower denoising loss than nearby hard-negative cameras.
6. Sample rotation, translation direction, translation scale, focal length,
   principal point, and context-role negatives separately.

### Gate

Improve graded camera binding and realized-pose error without losing the fidelity
gain of the existing model. If beta-only scheduled FiLM merely matches the current
branch, prefer it for simplicity.

## 10. Phase 7 -- rollout-aware training

### Dataset paths

Mine sequential trajectories rather than isolated tuples:

- forward extrapolation;
- backward motion;
- orbit;
- leave-and-revisit;
- figure-eight loop;
- branch-and-merge paths.

Use the exact inference slot policy during training.

### Curriculum

1. **Sequential teacher forcing.** Train clean ordered histories first.
2. **Cheap self-augmentation.** Ramp the probability from 0.25 to 0.75 of replacing
   a recent clean frame with a detached one-step \(x_0\), a replay-buffer generation,
   or a structured generation-like corruption. Build depth and memory from the
   replacement, not from hidden clean ground truth.
3. **Short true rollouts.** Train horizons \(H=2\), then \(H=4\):
   generate, detach, estimate realized pose/depth, apply the actual read/write and
   slot policy, and supervise every next target with the standard flow objective.
4. **Replay.** Maintain a refreshed buffer near 50% clean / 50% generated contexts.
   Do not backpropagate through 30 denoising steps times a long trajectory.

### Losses

- flow matching at every rollout step;
- camera-binding margin;
- commanded/realized pose discrepancy where the estimator is observable;
- router utility/ranking/calibration;
- revisit consistency;
- memory-write risk.

### Gates

Evaluate 5, 10, 20, and 50 generated views. Require bounded error growth, stable
fidelity, and better pose adherence. Smooth but wrong rollouts do not pass.

## 11. Phase 8 -- falsifiable persistent memory

### Write protocol

1. Observed memory is immutable.
2. Generated content enters only as provisional.
3. A generated frame cannot verify itself or another frame from the same lineage.
4. Promotion requires independent evidence: an observed revisit, a sufficiently
   independent branch, or multi-view agreement with bounded pose/depth uncertainty.
5. Contradictory surface hypotheses remain separate until evidence resolves them.
6. Confidence decays with lineage depth and time without independent support.
7. Rejected spatial writes may still be used briefly as recent context.

### Recovery controller

When predicted risk is high:

- reject the write;
- switch appearance memory to geometry-only;
- retrieve a stronger observed anchor;
- subdivide a large camera command with an SE(3) midpoint;
- spend more denoising steps or sample two candidates;
- select by binding, cycle, and revisit scores.

### Gates

- recovery after one deliberately corrupted generated frame;
- revisit accuracy after 10--40 intervening frames;
- false-memory half-life;
- observed-memory preservation;
- branch-and-merge path independence;
- contradiction handling.

## 12. Phase 9 -- long-horizon counterfactual value

Only start after the immediate router and provisional-write protocol work.

Let rollout state \(s_k\) contain typed context, memory, commanded/realized poses,
uncertainty, and provenance. Candidate actions include read mode and write mode.
Use short training-only branches to estimate:

\[
J_H(a\mid s_k)=
\sum_{h=0}^{H-1}\gamma^h
\left(L_{\rm flow}+\lambda_pL_{\rm pose}+\lambda_rL_{\rm revisit}\right)
+\lambda_M C(M_{k+H}),
\]

where \(C\) measures memory contamination against a later independent observation.
Distill the lowest-risk action into a target-free value model.

This is the strongest version of the contribution: a read can improve the current
frame and still be rejected because it poisons the next ten. Compare it directly
with the immediate-only router.

## 13. Training-budget policy

Use samples seen, trajectory targets seen, and counterfactual forward passes--not
optimizer steps alone.

| Component | Initial budget | Extension rule |
|---|---:|---|
| 2--4 M residual adapter | stop near 3--4 k steps | extend only if task A/B still improves; existing evidence says it usually degrades |
| 0.566 B Q/K tier | first compare at about 48 k samples | continue through staged 100 k, 200 k, 480 k sample gates if held-out task metrics improve |
| Geometry compiler/router | 50--100 k balanced oracle-labelled regions/tuples | extend based on oracle-gap closure and corruption generalization |
| Short self-rollout | 50 k targets at \(H=2\), then 100--200 k at \(H=4\) | extend only if 10/20-step metrics improve |
| Full 5 B fine-tune | deferred | use only if Q/K plus router lacks capacity; require low LR, EMA, and fidelity tripwire |

At effective batch 16, 30 k Q/K steps are roughly 480 k samples and are the honest
“10x” comparison to the current 48 k-sample exposure. Do not launch that run as one
blind job. Evaluate at 6 k, 12 k, 20 k, and 30 k, and stop when held-out task
improvement ceases.

Before scaling, diversify the training distribution. A useful target mixture is:

- 35% sequential/extrapolation;
- 25% wide-baseline/frontier;
- 20% revisit/loop;
- 10% branch-and-merge;
- 10% structured geometry/pose/memory corruption.

The exact weights should be tuned on coverage of behavior classes, not on the final
held-out score. Repeating one easy tuple distribution 10x is not equivalent to
training a robust rollout policy.

## 14. Evaluation matrix

### Paths

Forward, backward, orbit, large-baseline jump, leave/revisit, figure eight,
branch/merge, and deliberately corrupted history.

### Horizons

Single target and 5, 10, 20, 50-view autoregressive rollouts.

### Metrics

- covered/frontier/boundary PSNR, SSIM, and LPIPS;
- rotation and translation error reported separately;
- camera-binding hard-negative accuracy;
- realized-versus-commanded pose discrepancy;
- memory-harm / negative-transfer rate;
- oracle gap and fraction closed;
- utility calibration and risk-coverage;
- revisit consistency;
- path independence;
- false-memory half-life;
- recovery time after corruption;
- error growth \(E_{k+1}/E_k\);
- runtime and memory as horizon grows.

Do not use a composite camera metric that becomes zero when either its rotation or
translation term is zero. Keep the raw components visible.

### Ablation ladder

1. no memory;
2. current warped memory;
3. two-sided coordinate correspondence;
4. probabilistic geometric attention;
5. clean source features;
6. confidence router;
7. immediate counterfactual router;
8. commanded-only versus commanded/realized state;
9. teacher forcing versus self-augmentation versus short rollout;
10. immediate versus long-horizon value;
11. permissive versus provisional/falsifiable writes.

Use paired scenes/seeds, three training seeds for learned routers where affordable,
predeclared primary metrics, and confidence intervals. A visually chosen success
case is diagnostic evidence, not a headline result.

## 15. Paper go/no-go criteria

Proceed with the Counterfactual Memory Transport story only if:

1. memory harm exists and is not explained by ordinary geometric confidence;
2. the counterfactual oracle has a meaningful upper-bound advantage;
3. the learned router closes a substantial portion of that gap on held-out
   corruptions and scenes;
4. long-horizon value beats immediate utility on contamination, revisit, or
   recovery without losing current-frame quality;
5. commanded/realized pose bookkeeping measurably prevents spatial memory poison;
6. the complete system improves both camera accuracy and long-rollout stability
   against strong confidence-gated and recent-memory baselines.

If only geometric attention or clean memory wins, report it as a strong system
result but do not present the bundle as a new memory-learning principle.

## 16. Immediate execution order

1. Recover and freeze the exact `qk_batch16` configuration.
2. Evaluate its step-2000 and step-3000 checkpoints against the matched controls.
3. Build the counterfactual oracle using the existing memory/correspondence toggles.
4. Quantify oracle gap and confidence-versus-utility disagreement.
5. Implement fixed typed context and commanded/realized pose state.
6. Add source-address geometry support with separate source hypotheses.
7. Replace warped appearance with clean source-feature reads.
8. Train and gate the immediate utility router.
9. Add sequential self-augmentation and \(H=2\rightarrow4\) detached rollouts.
10. Enable provisional writes and corruption recovery.
11. Train the short-horizon read/write value model.
12. Scale Q/K or full-backbone training only where the gated results demonstrate a
    capacity rather than a data/objective problem.

The first decisive experiment is Step 3, not a larger training run. It determines
whether the proposed research question has enough headroom to justify the rest of
the system.
