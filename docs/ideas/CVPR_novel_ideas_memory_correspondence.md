# CVPR novel ideas

Revised proposal — 8 September 2026.

Status: reviewed proposal, not an implemented method or an established novelty claim. This document does not change the active Gaussian training run.

## 1. Supervise camera response through reliable memory correspondences

### Proposed research question

Can persistent Gaussian memory supervise the generator's response to camera commands without requiring the generator to imitate imperfect Gaussian renderings?

The intended benefit is more accurate camera execution and better preservation of previously observed content. This is a hypothesis to test, not an explanation already established for our recent evaluation regression.

Our current pipeline uses ReSplat-rendered Gaussian memory as a condition for a latent video generator. The generated video can depart from that condition. PixWorld instead supervises a predicted Gaussian scene through differentiable rendering. The adaptation proposed here keeps our video generator and asks the memory to provide geometric relationships between nearby commanded views.

### Main method: transfer correspondence

The main proposal is now correspondence supervision. Direct matching of generated and rendered feature derivatives is only an experimental baseline: those derivatives also transfer texture gradients, blur, and camera-dependent rendering artifacts.

Let `M` be causal memory, `h` fixed history, `c` the upcoming command, and `G_theta` the declared generation procedure. Construct two nearby commands with a local SE(3) or command parameterization:

```text
c_minus = perturb(c, -epsilon * d)
c_plus  = perturb(c, +epsilon * d)
```

Fix the starting pose and propagate the perturbation consistently across all eight output frames. Six command parameters describe a restricted trajectory family, not every possible eight-frame trajectory. Normalize translation by scene scale and declare the angular unit for rotation.

Keep memory, source selection, and admission decisions fixed. Recompute camera-dependent rays, command embeddings, scaffold renders, and archive geometry for each intervention. Otherwise discrete retrieval changes contaminate the comparison.

`xi` below denotes a declared shared-noise convention. Identical seeds reduce one source of variation; they do not prove that the outputs depict the same physical realization. Test this assumption on observed static content and compare alternative noise couplings.

Use Gaussian geometry to determine where an already supported surface should appear under another command. Compare the generator's own features at those corresponding locations, instead of using the Gaussian renderer's appearance as the target.

Let `T_minus_to_plus` transport features from the minus view to the plus view using memory-derived depth, the two camera trajectories, and visibility checks:

```text
F_minus = phi(G_theta(h, M, c_minus, xi))
F_plus  = phi(G_theta(h, M, c_plus, xi))

E_plus  = F_plus - T_minus_to_plus(F_minus)
E_minus = F_minus - T_plus_to_minus(F_plus)

L_transport = 0.5 * [weighted_mean(rho(E_plus), W_plus)
                    + weighted_mean(rho(E_minus), W_minus)]

L_total = L_existing + lambda * L_transport
```

Here `rho` is a declared robust feature residual. Normalize by valid support, require a minimum amount of support, and skip/log the auxiliary term when that requirement fails. Keep weights detached from the generator and independent of its current residual so it cannot reduce the objective by rejecting its own mistakes. Choose a small fixed `lambda` on validation data.

Initially compare corresponding frames from two command-conditioned clips; temporal transport within a clip is a separate extension. Target images remain ordinary supervised targets but cannot enter causal memory or supply future geometry to the correspondence teacher.

This transfers the renderer's correspondence rather than its RGB texture. It therefore addresses a specific weakness of the direct derivative formulation. It still inherits geometry and visibility errors, and generic warp consistency itself is not novel.

Implementation decisions to investigate:

- Derive transport from supported static surfaces, with depth consistency and occlusion handling. Do not treat blended depth from translucent or overlapping Gaussians as an exact surface.
- Start with low-resolution spatial features. Features must retain spatial information; a globally pooled semantic embedding can miss camera errors.
- Compare a symmetric loss with a stop-gradient target variant. Neither alone prevents collapse; normal ground-truth training is necessary.
- Do not constrain disoccluded areas, unsupported regions, moving objects, or strongly view-dependent effects as if their appearance were invariant.
- Audit whether the model satisfies transport by blurring details or ignoring stochastic variation.

### Make the evidence direction-specific

The same memory may support rotation accurately while providing weak evidence for translation. A distant or weakly textured region can make different motions difficult to distinguish.

For diagnosis, construct a local Jacobian `J` of spatial feature response induced by surface projection. Geometric displacement alone is insufficient: a featureless region can have large predicted displacement but no identifiable appearance response. A weighted matrix

```text
H = J^T W J
```

describes sensitivity under that representation. Under a specified linear observation model and noise covariance it has an information-matrix interpretation; without those assumptions, call it a sensitivity matrix rather than calibrated uncertainty.

Use its singular/eigen directions to identify ambiguous combinations of motion. Normalize translation by scene scale and rotation by a declared angular unit before inspecting the spectrum. Compare response magnitude against estimated teacher error: a large derivative from an unstable boundary is not useful evidence.

Possible reliability evidence includes source-subset reconstruction agreement, held-out reprojection error on prior observations, perturbation stability, and mutual visibility. Validate these signals against actual correspondence error before using them as weights. The method must report retained coverage so that apparent gains cannot come entirely from ignoring hard regions.

This is the potential organizing principle: **supervise camera behavior only where the memory has evidence for that behavior, and make the strength of supervision depend on the direction of motion.**

### Theoretical justification of the revised method

The assumption is feature consistency on the same visible static surface, not equality to Gaussian-rendered appearance.

Let `T_star` be true correspondence and `T_hat` the memory estimate. Suppose, on the retained region:

1. Correct surface features differ by at most `a` between views because of appearance changes or feature non-invariance.
2. Correspondence error is at most `delta` in feature-grid coordinates.
3. The compared feature field is locally spatially Lipschitz with constant `L`.

Adding and subtracting the feature sampled at the true corresponding location gives the local bound:

```text
|| F_plus_star - T_hat(F_minus_star) || <= a + L * delta
```

Interpolation error must be added when it is not negligible. This explains why correspondence accuracy, visibility, and feature stability should determine where supervision is applied. These assumptions must be validated; Gaussian opacity does not establish them.

A small transport residual does not prove correct pose or content. Constant, blurred, or jointly wrong views can agree under transport. Inferring pose error from feature residual requires local identifiability and an appearance/content anchor. Repeated patterns, textureless regions, incorrect geometry, and dynamics can invalidate that inference.

Keep ground-truth flow supervision and evaluate absolute pose and appearance. No long-horizon stability theorem is claimed: memory feedback changes the teacher over time. The earlier derivative bound does not apply automatically to this transport objective and is no longer its theoretical justification.

An auxiliary loss on a detached ReSplat condition alone cannot train Wan. The generated branch needs gradients through decoding and feature extraction. Frozen weights still require activation memory for input gradients; `no_grad` on that branch would disable the signal. Geometry, correspondence construction, and reliability weights can remain detached.

### Sampling and implementation contract

Define and measure the objective first on complete paired samples. A single clean prediction at a fixed noisy latent is a different conditional estimator, not automatically a valid surrogate for the deployed sampler.

Before training, compare complete sampling with one-step and truncated-sampling proxies at several noise levels. Test whether a small update using each proxy improves full-sampling camera accuracy. Measure noise-coupling sensitivity and restrict supervision to already supported static regions.

Start with one perturbation direction on occasional updates and profile backward memory and wall time. Report the actual surrogate used; do not describe it as full-rollout training if it is not.

### Practical experiment sequence

#### Gate 1: validate the diagnostic without training

Use synthetic scenes or reliable posed observations with known correspondences. Include deliberately wrong yaw, translation scale, mixed camera errors, texture changes, blur, missing geometry, and moving objects.

Ask:

- Does the diagnostic respond to camera mistakes more strongly than benign appearance changes?
- Does correspondence supervision tolerate texture degradation better than direct rendered-feature matching?
- Do uncertainty weights predict correspondence error, rather than just suppress difficult pixels?
- Does the method abstain on genuinely ambiguous or unsupported motion?
- How much usable supervision remains after masking?

Reject or revise the proposal if those distinctions fail.

#### Gate 2: validate a training surrogate

On a small scene set, compare the response of complete paired rollouts with one-step or truncated-sampling proxies at several noise levels. Measure agreement and verify the resulting gradient directions improve a short full-sampling test.

Do not assume that a cheap denoiser derivative transfers to the deployed sampler. Start with one sampled translation or rotation direction on occasional updates, not a full six-direction calculation at every update. Profile backward memory before scheduling a run.

#### Gate 3: controlled training comparison

Start from the same checkpoint, data schedule, and evaluation seeds. Compare:

1. Existing training.
2. Generic geometry-feature supervision inspired by PixWorld.
3. Direct rendered-feature response matching.
4. Geometry-derived feature transport.
5. Feature transport with calibrated reliability and direction selection, compared directly with visibility-only transport.

Report both matched-update and matched-compute comparisons where feasible. A simpler depth-warp teacher is essential: if it performs equally well, do not claim Gaussian-specific benefits.

Use independent camera evaluation, including absolute translation and rotation errors, image quality, detail preservation, revisit consistency, long-rollout degradation, runtime, and reliability-mask coverage. Reject a treatment that lowers its auxiliary loss by blurring outputs, reducing diversity, or discarding most challenging examples. Evaluate on held-out scenes and camera patterns, not only the perturbations used for training. If VGGT features supervise training, include another camera estimator or trusted geometry for evaluation to reduce evaluator-specific bias.

#### Later extension: memory admission

Only after the response experiment succeeds, investigate using geometric inconsistency to limit memory writes. This is a separate intervention requiring an ablation. Do not automatically substitute commanded poses for uncertain realized poses, and do not declare old geometry false merely because the newest generated frame disagrees with it. Response consistency alone cannot establish which observation is correct.

### Relationship to existing work

- **PixWorld:** motivates supervision through 3D structure and multi-view features. Our proposed transfer target is command response/correspondence from existing memory, while retaining a video generator. The released Wan adaptation differs from the paper model, and its released checkpoints omit the geometry-perception loss.
- **Sobolev training and Jacobian matching:** establish derivative supervision and knowledge transfer. The finite-difference equation is not a new mathematical contribution.
- **CamProbe:** already applies geometry-derived displacement fields during denoising. Our proposed distinction is training with memory-derived evidence and direction-specific reliability; this needs experimental and literature validation.
- **Memory-control methods:** camera-dependent retrieval and memory gating already exist. A generic confidence gate should not be presented as the contribution.

The potentially defensible contribution is a demonstrated method for transferring reliable camera behavior from imperfect persistent geometry, including a treatment of where that geometry cannot identify motion. It must outperform simpler geometry-loss and warping baselines to justify its complexity.

### Next action and decision criteria

Build the offline correspondence diagnostic first. Advance only if it distinguishes camera errors from appearance changes and its reliability estimates remain useful on generated histories. If visibility-only transport performs equally well, remove the extra direction-selection mechanism. If ordinary depth warping matches Gaussian memory, narrow the contribution accordingly. The highest-risk assumption is that memory-derived correspondences are accurate enough on the generated-history distribution to improve camera response without damaging appearance.

The current 4k run remains a baseline; this review made no training changes.

### References

1. [PixWorld paper](https://arxiv.org/html/2607.05373v1)
2. [PixWorld released training guide](https://github.com/SensenGao/PixWorld/blob/main/train/README.md)
3. [PixWorld geometry-loss implementation](https://github.com/SensenGao/PixWorld/blob/main/train/objectives/geometry_loss.py)
4. [Knowledge Transfer with Jacobian Matching, ICML 2018](https://proceedings.mlr.press/v80/srinivas18a.html)
5. [Sobolev Training for Neural Networks](https://arxiv.org/abs/1706.04859)
6. [CamProbe: Probing into Camera Control of Video Models](https://arxiv.org/abs/2605.14815)
7. [Memorize When Needed: Decoupled Memory Control](https://arxiv.org/abs/2604.18215)
8. [UCM: Unifying Camera Control and Memory](https://arxiv.org/abs/2602.22960)

## 2. Research priorities after inspecting our failures and newer methods

Review date: 8 September 2026. This section updates research priorities: first investigate whether the reconstruction path discards useful reliability information, before training the correspondence method in Section 1 against that geometry.

### Evidence from the current system

The completed 2k evaluation contains 14 scenes and 160 generated frames per scene. Its image metrics sample every eighth frame; the following are scene-averaged sampled LPIPS values, not averages over all 160 individual frames:

| Hop | 1 | 5 | 10 | 15 | 20 |
|---|---:|---:|---:|---:|---:|
| LPIPS | 0.232 | 0.518 | 0.617 | 0.690 | 0.742 |

These show increasing error with horizon, but larger viewpoint changes and disocclusions also increase difficulty. They do not isolate memory-induced drift.

The source and audit logs reveal a specific mismatch between stored reliability and reconstruction input:

- 233 of 280 generated endpoint writes failed geometry certification.
- 245 of 280 ReSplat reads included at least one rejected endpoint.
- The reader combines archive candidates with recent history and selects the anchor plus approximately evenly spaced sources, up to eight views. Rejected observations remain eligible through recent history.
- Memory caps rejected endpoints' authority at 0.25, but the ReSplat wrapper constructs reconstruction input from RGB, poses, intrinsics, and bounds. It does not apply that per-source authority or supplied depth confidence.
- Output confidence is derived from accumulated alpha and basic validity checks; it does not preserve the source's certification status.

Failure reasons include depth disagreement, pose-backend disagreement, and invalid confirmation. An endpoint rejected by these checks is not necessarily wrong: the estimator can fail. Therefore, excluding every rejected endpoint is a diagnostic baseline, not yet the recommended solution.

Reproducible counts and interpretation limits: [evidence.json](analysis/cvpr_failure_review_20260908/evidence.json). Source: the frozen runtime under `training_runs/gaussian_recovery500_prep_20260907/runtime/gaussverse/memory/`.

The earlier visual review concerns the previously reported benchmark model, not this 2k checkpoint. It documents object deformation, loss of small structures, and camera errors as distinct phenomena. In particular, an initially unseen bush cannot be labeled forgotten, and a high camera score does not establish preservation of a shop display. Keep those distinctions in the experiments.

### Candidate A — preserve evidence reliability through reconstruction and memory updates

**Priority: first diagnostic, and potentially the main method if the mechanism is confirmed.**

The question is whether unreliable generated observations become strong future conditioning merely because reconstruction produces an opaque surface. A renderer's ability to cover a pixel is different from evidence that the rendered surface is correct.

Maintain separate quantities for surface visibility, geometric support, pose uncertainty, and preservation of previously observed appearance. Carry a conservative summary of this evidence into reconstruction/source selection and the rendered condition. It is not sufficient to compute confidence after conflicting source information has already been fused.

Record which retained observations support a surface or patch, including whether they are external observations or generated descendants. Preserve references to earlier evidence when the latest generation changes an object. A generated view can add useful model-implied geometry, but repeated agreement among descendants is not independent physical confirmation.

One statistical motivation is correlated evidence. For `n` equal-variance measurements with common correlation `rho`, the variance of their mean is:

```text
Var(mean) = sigma^2 * [1 + (n - 1) * rho] / n
n_effective = n / [1 + (n - 1) * rho]
```

When `rho` is near one, many repeated observations contain roughly one measurement's information under this illustrative model. This is not a calibrated model of our generator. It motivates testing whether copying or repeatedly regenerating the same mistaken structure should increase confidence.

All generated observations ultimately share ancestry. A useful estimator must distinguish correlated repetition from complementary geometric constraints without calling either independent ground truth. Parent/source IDs alone do not estimate statistical correlation.

Concrete components to test, in order:

1. Preserve source status at reconstruction input: compare current selection, certified-only selection, and a coverage-preserving selection using validation-calibrated reliability. Do not attenuate RGB brightness as a substitute for confidence.
2. Compare reconstructions from available source subsets and reproject into earlier retained views that were not used in that particular reconstruction. This checks consistency with stored evidence, not truth independent of the generator.
3. Select the eight sources for supported target-visible coverage and complementary evidence, with a fixed budget. Compare simple pose/covisibility retrieval before introducing a learned policy.
4. Keep conflicting surface hypotheses or earlier appearance evidence available until the conflict can be resolved; measure the cost and avoid freezing an early hallucination permanently.

The exact integration of source reliability into the frozen ReSplat model remains open. Source selection can be tested first; an uncertainty-conditioned reconstructor would require separate training and validation. Simply multiplying output alpha by a scalar cannot undo incorrect fused geometry.

**Novelty boundary:** fixing ignored authority or adding a confidence threshold is an engineering correction. A paper-worthy contribution would require a validated treatment of correlated generated evidence, uncertainty propagation through reconstruction, and measurable improvement in persistent world state. Provenance metadata or generic confidence gating alone is insufficient.

**Reject this direction as the main explanation** if controlled source/reliability interventions leave the failure unchanged, or if the render is already correct and the generator is where objects disappear.

### Candidate B — train recovery to reduce inherited errors across memory updates

**Priority: second, conditional on demonstrating feedback amplification.**

Robust training on corrupted history is already established. A more specific hypothesis is that low next-frame loss can coexist with an update that amplifies a small inherited geometric error on later revisits.

Create two rollout states that differ by one controlled nuisance: a source-pose error, a local shape deformation, or a spurious occluder. Keep the intended scene, camera command, and noise coupling fixed. Compare the difference after generation, reconstruction, and memory writing, using matched visible surfaces and clean/reference targets where available during training.

For an idealized error process with uniform bound:

```text
||e_(t+1)|| <= q * ||e_t|| + b,  with 0 <= q < 1,

||e_t|| <= q^t * ||e_0|| + b * (1 - q^t) / (1 - q)
```

The bound motivates measuring error amplification, but a few paired samples cannot establish uniform contraction. Reprojection ambiguity, changing visibility, and discrete memory selection complicate the dynamics.

Only nuisance directions should contract. Differences caused by legitimate camera commands, newly exposed surfaces, or real object motion must remain. Apply the comparison after aligning the intended geometry; contracting arbitrary state differences would encourage frozen video or mode collapse.

Start with short paired rollouts and independent measurement of later error, not a new full differentiable renderer pipeline. Training through the current detached geometry path is not automatic. Candidate implementations include targets derived from cached intervention rollouts or a validated short differentiable generator surrogate; describe the chosen approximation explicitly.

**Novelty boundary:** distinguish error-amplification supervision from ordinary flow training with an error bank. Compare against our AHR/recovery recipe and Robust Dreamer-style deviation exposure at matched compute. Remove this direction if it merely improves its own residual score or suppresses valid motion.

### Candidate C — correspondence-based camera-response supervision

This is Section 1's proposal. It is the appropriate next step if supported geometry is correct but the video generator ignores its camera relationships. Use it as a separate treatment first. Combining it with a new memory writer immediately would make attribution difficult.

### Essential diagnostic before selecting the main contribution

At the same failing boundary, preserve source images, selected source IDs/status, source poses, scaffold RGB/depth/confidence, and generated output. Compare interventions with matched noise and the same source budget:

| Intervention | What it tests |
|---|---|
| Restore a useful earlier source already present in causal history | Retrieval omitted relevant evidence |
| Change source reliability handling | Uncertain geometry is becoming authoritative conditioning |
| Replace estimated source poses with trusted poses in a labeled diagnostic | Pose error corrupts reconstruction |
| Use clean-history memory in a labeled oracle experiment | Accumulated generated-history corruption |
| Keep memory fixed while advancing generation for a short diagnostic | Contribution of memory updates, with coverage loss reported |
| Correct scaffold in a labeled oracle experiment | Whether the generator can use accurate geometry at all |

Oracle variants are diagnosis only and must never be reported as deployable results. These interventions can identify which changes matter under controlled conditions, but interactions between retrieval, pose, and generation still require care.

### Current competitors and the claim we should target

As of this review, the comparison set must extend beyond Lyra 2.0:

| Relevant method | Capability already addressed |
|---|---|
| [Robust Dreamer](https://arxiv.org/abs/2605.30855) | Latent Gaussian memory and training with archived prediction deviations |
| [VMem](https://v-mem.github.io/) | Surfel-indexed retrieval of historical views |
| [UCM](https://arxiv.org/abs/2602.22960) | Camera control and memory alignment through positional-encoding warping |
| [WorldTrace](https://arxiv.org/abs/2608.07408) | Addressability of compressed long-horizon memories |
| [ReWorld](https://arxiv.org/abs/2608.23565) | Bounded pose-indexed landmark memory and action/recall evaluation |
| [SolarWM](https://arxiv.org/abs/2609.02886) | Large-scale open data and multi-backbone world-model training |

These papers use different protocols. Their reported scores do not form a directly comparable ranking with our 14-scene evaluation. Release accessibility, inference configuration, and common input information must be checked before selecting reproducible baselines.

The target claim should be narrow and measurable: improved preservation of previously established geometry and object appearance through occlusion and revisits, under accurate camera execution and a fixed memory budget. Distinguish this from reconstructing unseen ground-truth content, which is not uniquely determined by the input image.

Use the existing 14 scenes as a diagnostic/regression set, not the sole basis of a SOTA claim. Evaluate broader held-out scenes and trajectories, multiple seeds, paired scene-level uncertainty intervals, camera error, known-region identity, novel-region quality, and compute. Existing protocols such as [WorldRoamBench](https://arxiv.org/abs/2606.31672) and [WBench](https://github.com/meituan-longcat/WBench) should be considered before inventing another benchmark. Anti-shortcut controls should penalize stalled cameras, frozen scenes, and selective rejection of hard examples.

### Recommended research order

First isolate the source-reliability failure in the current pipeline. If supported, investigate Candidate A as the main contribution. If reliable rendering is being ignored, prioritize Candidate C instead. Pursue Candidate B only after directly measuring amplification through the generation–memory loop. One demonstrated failure mechanism and one effective remedy are stronger than an unexplained combination of three modules.

The review performed read-only code/log inspection and CPU aggregation. No new generation, GPU experiment, or training change was made.

## 3. BulletTime: implications for our novelty and diagnostics

Reviewed 8 September 2026: [BulletTime paper](https://arxiv.org/abs/2512.05076), [official repository](https://github.com/19reborn/BulletTime).

BulletTime already studies independent camera and world-time control with positional encoding, adaptive normalization, and disentangled data. Its video-conditioned, non-autoregressive setting differs from our persistent-memory rollout. Camera/time disentanglement alone is therefore not our novelty claim.

### Our inference: distinguish world time from computation and memory age

For our static-scene experiments, additional autoregressive calls should not themselves change a supported object's geometry. Four quantities must remain conceptually distinct:

- Camera pose: where the scene is viewed from.
- World time: the physical scene state being viewed.
- Memory age / rollout position: when evidence was retained relative to the computation.
- Diffusion timestep: the noise level within a generation call.

The existing relative memory-age encoding is not explicit world-time control. Its presence does not establish disentanglement.

Add a controlled diagnostic that holds scene state, requested pose, and causal geometric evidence fixed while varying history representation or its age metadata within valid ranges. Measure known-region feature and geometry changes across matched seeds and repeated samples. This tests sensitivity to the representation of history; it does not prove what happens after a naturally different long rollout.

A separate revisit test can compare outputs at the same physical pose after different detours. Restrict identity scoring to previously observed static content, account for differences in available evidence, and score camera execution independently. Newly revealed content need not match an unknown real-world continuation.

Dynamic scenes need their own negative control: legitimate physical evolution must remain possible. A method that freezes all appearance can pass a static invariance test while failing world modeling.

The potential extension to our memory work is preserving the distinction between changes justified by scene evidence and changes caused by repeatedly processing generated history. Treat this as a diagnostic principle first, not as an established new architecture or a guarantee of persistence.

The released simulator and data are candidates for controlled experiments after checking their conventions and task compatibility. No model download, inference, or training change was performed in this review.
