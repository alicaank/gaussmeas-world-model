# GATE-ΩR Contract — redesigned single-run final teacher

Date: 2026-08-26.

Status: **implemented; launch requires current preflight and full-population cache**.
This document is the executable scientific contract; training may not launch until every
preflight item in §10 passes for the exact runtime recipe and data transaction.
It supersedes both the staged GATE-Teacher proposal and the first GATE-Ω contract.

The method has one production optimization run. Preflight unit tests, step-zero
evaluation, instrument calibration, and a disposable two-update resource smoke are not
training stages and their weights are never reused. Nothing is unlocked, reweighted, or
reconfigured during the production run. A tripwire may stop the run; it may not alter it.

This contract distinguishes three kinds of statement:

- **Established mechanism:** supported by prior work and used only where the same causal
  role applies here.
- **Measured project fact:** supported by a named GaussVerse artifact or evaluator.
- **Hypothesis:** plausible but unproven until the pre-registered measurement passes.

No component is called novel merely because its exact code has not appeared elsewhere.
The possible contribution is the complete system and its measurement contract.

---

## 0. Decision and non-negotiable invariants

GATE-ΩR starts from the original pretrained Wan2.2-TI2V-5B carrier. The rejected
20k/21k/22k GaussVerse weights are evaluation baselines only. In particular,
`outputs/final_teacher_parent_all_on_step21000.safetensors` is not a parent: the previous
audit showed that it learned shallow generated recurrence, weak archive reading,
anchor-dominant authority, camera-rate compression, and a training/deployment feedback
mismatch. A short continuation cannot be assumed to unlearn those policies.

The following invariants define the method:

1. **Exact initialization, Lyra-selective optimization.** Original Wan tensors are exact
   at step zero and every new route has an exactly zero output. During the single run,
   Wan self-attention and the stock patch embedding, the copied FramePack history patch
   projection, and all Ω routes are trainable. The remaining Wan carrier is frozen.
   No rejected GaussVerse checkpoint is imported. Selected bf16 values are promoted
   exactly to fp32 storage before optimization; this changes representation, not value.
2. **One Lyra-compatible deployment gauge.** Seed DA3 is aligned once to seed MoGe by
   Lyra's inverse-depth least-squares scalar; recurrent DA3 is aligned to pipeline
   cameras. No privileged training-only scale enters model conditioning, supervision,
   memory admission, or evaluation.
3. **Positive supervision for every valid command.** No real camera command is trained
   only as a negative example.
4. **Stationary long-horizon exposure.** The mixture is fixed from optimizer step 1 and
   has support through deployment call 27, including saturated temporal memory and
   archive retrieval.
5. **Estimated-depth parity.** Model-facing scaffolds are built with the same estimator
   family and causal inputs used at deployment. GT depth is calibration/evaluation data,
   never a cleaner scaffold condition.
6. **Fail-closed persistent geometry.** Pose validity alone never certifies RGB-D as
   persistent geometric evidence.
7. **Measurement before claims.** Four-scene tests are tripwires only. Final claims use
   a powered held-out protocol, uncertainty intervals, and an independently checked
   camera instrument.

---

## 1. Parent, initialization, and provenance

### 1.1 Parent

- Backbone: the exact pretrained Wan2.2-TI2V-5B base used by the existing fresh-Wan
  initializer.
- The current `final_teacher_parent_all_on_wan22_base.safetensors` must not be reused
  blindly: it predates this architecture and contains live-initialized routes. A new
  initialization artifact is built after §10 is implemented.
- 20k/21k/22k checkpoints remain immutable external baselines. No tensor, optimizer
  moment, scheduler state, reader prior, or learned route is imported from them.
- The base fingerprint, source commit, patch hash, route manifest, data-manifest hash,
  gauge-manifest hash, and scaffold-calibration hash are stored in every checkpoint.

### 1.2 Trainable state

The optimizer follows Lyra's resource-proven selective tier: every Wan self-attention
parameter and the stock patch embedding are trainable, while all other Wan parameters
are frozen. This is full-rank selective training, not LoRA or an adapter-only ceiling.
The optimizer authority additionally includes:

- ordered four-ray encoders and their zero-output residual projections;
- the 45-D command-resolved Q/K residual;
- the target-view scaffold encoder and zero-output residual;
- the zero-gated host-weight archive delta reader;
- zero-output FramePack role/occupancy residuals; the history patch projection is copied
  exactly from Wan at initialization and is then trainable;
- scalar/channel gates and confidence embeddings belonging to those routes.

Text encoder, VAE, MoGe, DA3, VIPE, and evaluation-only VGGT remain frozen estimator or
tokenizer components and never enter the optimizer. Camera, scaffold, and archive
injections remain limited to blocks 7/15/23. Their locations are fixed before launch.
Carrier preservation is claimed only at initialization; post-update checkpoints contain
the complete deployable teacher, but only the selective tier and new routes have moved.

### 1.3 Step-0 equality

With all new gates present but disabled by their exact-zero outputs, deterministic
evaluation on a fixed batch must satisfy:

- tensor identity of the frozen parent state before and after initialization;
- maximum denoiser-output difference `<= 1e-7` in fp32 reference execution and exact
  equality in the production bf16 path where the same kernels are used;
- identical decoded pixels, LPIPS, and sampler trajectory under the same seed;
- every new output projection/gate exactly zero;
- nonzero, finite internal features behind every zero output;
- a nonzero gradient on each output gate/projection under its designated row type.

Camera dropout is not part of GATE-ΩR, so it cannot contaminate this comparison.

---

## 2. One gauge and geometric conventions

The sole similarity transform is fixed from the one deployment-observable anchor RGB.
Exactly as in Lyra, that RGB is processed once by both the pinned
`DA3NESTED-GIANT-LARGE-1.1` and MoGe-v1 checkpoints. On their common validity mask:

```text
u_D = 1 / (d_DA3 + 1e-6)
u_M = 1 / (d_MoGe + 1e-6)
alpha = sum(u_D * u_M) / sum(u_D * u_D)
d_anchor = d_DA3 / alpha
```

MoGe is released after this seed calculation and is never run on later generated RGB.
The anchor transform is then:

```text
R_g = R_anchor^T
o_g = t_anchor
s_g = median finite positive d_anchor
x_g = R_g (x_world - o_g) / s_g
```

Exactly this transform is used for:

- all commanded translations in the 45-D command code;
- every camera pose supplied to ray construction;
- Plücker moments, recomputed after transforming the ray origin and direction;
- source points and target cameras used by the scaffold renderer;
- commanded-versus-realized residuals;
- admission checks and metric-scale conversion.

For every later causal RGB window, DA3 receives the same calibrated pipeline intrinsics
and world-to-camera matrices used by Lyra, with both `align_to_input_extrinsics=True`
and `align_to_input_ext_scale=True`. Its recurrent depths are therefore already in the
pipeline camera scale and are divided only by `s_g` for model conditioning. Training
never multiplies MoGe/DA3 output by a dataset scene scalar. GT poses and depth may score
or calibrate this policy but cannot determine the model-facing gauge.

Required tests include train/deployment numerical equality and Sim(3) equivariance:
applying an arbitrary proper global similarity to a fixture must leave all normalized
commands, rays, scaffold projections, and admission decisions unchanged within tolerance.

---

## 3. Conditioning architecture

### 3.1 Ordered four-ray route

Each video latent represents four RGB targets. Their four six-channel Plücker planes
remain separate through the first camera encoder:

```text
[ray_0, ray_1, ray_2, ray_3] -> shared per-pose stem
                              -> ordered pose embeddings
                              -> joint token-local mixer
                              -> zero-output residual
```

The planned target cameras are all known before generation, so this route is called
**ordered**, not causal. It must not replace or reshape a parent activation; it adds a
zero-output residual at blocks 7/15/23. Sub-frame order is encoded
explicitly, and the order knockout is scored with a within-call pose-timing metric rather
than only an endpoint camera score.

### 3.2 Command-resolved Q/K route

The existing 45-D representation is retained after gauge unification: endpoint/global
motion, intrinsics, and three ordered earlier-to-endpoint SE(3) residuals. It produces a
Q/K-only residual at blocks 7/15/23. It never writes V content.

The ordered rays give spatial camera correspondence where image support exists. The
command code gives a global continuous instruction and remains defined in regions where
the scaffold has no pixels. Neither route is claimed sufficient by itself.

### 3.3 Target-view scaffold

The scaffold is the primary mechanism for making the commanded arrival geometrically
easy in already observed regions.

Eligible sources are:

1. the immutable anchor; and
2. the most recent **geometry-certified** endpoint, if one exists.

`pose_valid = 1` alone is insufficient. A generated endpoint is geometry-certified only
under §6. Until then it may occupy bounded temporal appearance history but cannot supply
persistent scaffold geometry or archive keys.

Each source RGB-D view is rendered independently to every target RGB camera using
depth-aware point splatting. The complete warped RGB sequence (eight RGB frames for
`G=2`) is then encoded through the same persistent Wan VAE path used at deployment;
latent-to-latent warping and endpoint-only camera shortcuts are forbidden. Composition
obeys target-camera visibility:

- nearest positive target depth wins;
- confidence breaks only near-depth ties;
- no unconditional anchor priority exists;
- disagreement, coverage, source age, and source identity remain explicit channels;
- holes remain holes and are never presented as valid black RGB.

The model-facing scaffold contains RGB latent features plus inverse depth, validity,
confidence, cross-source disagreement, and source age. A separate unknown-region mask is
preserved after VAE downsampling so invalid pixels cannot bleed into valid evidence
without being observable to the network.

The scaffold enters through a target-aligned zero-output residual. It is a hypothesis,
not a guarantee: its supported-region intervention must pass §11.

### 3.4 Deployment-matched scaffold noise

Every model-facing training scaffold is produced by the same estimator family and the
same causal RGB, pipeline-extrinsic, and calibrated-intrinsic input set used at
deployment. Seed MoGe participates only in the one Lyra scale alignment above; recurrent
geometry is DA3-only. GT-depth scaffolds are oracle diagnostics only.

Before launch, a calibration set disjoint from both production training rows and final
acceptance scenes measures estimator-versus-GT error after the anchor gauge, including:

- global log-scale residual;
- low-frequency surface warp;
- depth-edge displacement;
- thin-structure dropout;
- confidence-conditioned invalidation;
- dynamic/inconsistent-region dropout.

These statistics are diagnostics and certification inputs, not a second synthetic depth
domain. Model-facing training and inference both warp the raw causal deployment estimate;
neither substitutes GT depth nor applies an extra geometry corruption. This removes an
unvalidated train/deployment mismatch while keeping failures observable.

Latent noise augmentation follows Lyra's released `noise_with_sigma` path exactly:
independently per target latent frame, `log(sigma) ~ Normal(-3, 2)`, multiplier `1.0`.
Inference uses Lyra's fixed sigma `0.001`. Sigma is supplied as a condition to the
scaffold encoder.

### 3.5 Zero-gated host-weight archive delta reader

The old one-percent reader and the forced finite-mass step-0 reader are both removed.
At blocks 7/15/23, archive tokens use the host block's normalization and K/V projections.
For target rows the reader computes:

```text
A_main  = attention(Q, K_main,             V_main)
A_joint = attention(Q, [K_main, K_source], [V_main, V_source], source_bias)
H_main  = host_gate(h, host_out(A_main))
H_joint = host_gate(h, host_out(A_joint))
h       = H_main + tanh(g_reader) * (H_joint - H_main)
```

`g_reader` is exactly zero at initialization, so the parent function is unchanged even
though the internal joint attention may use a useful token-balanced source calibration.
Only the reader gate receives gradient initially; after it opens, source calibration and
source-role parameters receive gradient. This avoids both arbitrary one-percent
starvation and a step-0 function change.

Archive latents advance through the host blocks preceding each reader site under
the current diffusion timestep before their K/V are computed. Those states are not
cached across denoising timesteps. This preserves host-feature transfer rather than
asking a fresh source encoder to relearn the backbone representation.

The reference joint-attention calibration is 20% aggregate source mass at saturated
`H=20, G=2, K=5` equal-logit layout. This is an internal calibration, not a claim about
realized attention. Training and evaluation report the realized attention mass,
gate magnitude, source coverage, and source-removal effect separately.

---

## 4. Objective: symmetric positive target-view binding

The production objective has two terms with constant weights from step 1:

```text
L = L_flow + 0.10 * L_bind
```

There is no camera-route loss, path-pair consistency loss, support-masked low-frequency
loss, estimator reward, or self-gradient-forcing term in this run.

### 4.1 Flow supervision

Every positive target receives the ordinary rectified-flow loss. This remains the only
term that specifies what an image at a valid command should look like.

### 4.2 Symmetric paired-target binding

A binding draw selects two real target trajectories `a` and `b` from the same eligible
static scene and same legal history prefix. Each has its own command, target frames, and
correctly rendered scaffold:

```text
(c_a, x_a, scaffold_a)
(c_b, x_b, scaffold_b)
```

Both are positive flow examples. They share timestep and epsilon, but each target is
noised from its own clean latent. Define `e_ij` as the masked mean error between the
predicted flow for noisy target `x_i` under condition `c_j, scaffold_j` and the true flow
target for `x_i`. The mask contains static, confidence-qualified regions where the two
cameras induce a measurable projection difference. It does not claim supervision in
unknown/disoccluded pixels.

Pairs are built from one registered base frame and two distinct legal future target
spans; neither target span may appear in the history prefix. Reversed or synthetically
scaled cameras are excluded unless a corresponding real positive target exists. GT
geometry may construct the supervision-only comparison mask but cannot enter either
model condition.

For each target, normalize the positive-versus-cross error difference by the detached
mean magnitude of that pair. The binding loss is:

```text
r_a = (e_aa - e_ab) / stopgrad(0.5 * (e_aa + e_ab) + 1e-6)
r_b = (e_bb - e_ba) / stopgrad(0.5 * (e_bb + e_ba) + 1e-6)
L_bind = 0.5 * (softplus(r_a) + softplus(r_b)) - softplus(-2)
```

Because nonnegative errors keep each normalized ratio in `[-2, 2]`, subtracting the
constant `softplus(-2)` gives the objective an interpretable zero lower bound without
changing its gradients. A pair with empty or sub-threshold comparison support is rejected
and resampled; it is never converted into a zero-loss example.

This formulation has no unbounded margin and no command that is only negative: `c_a`
must correctly predict `x_a`, and `c_b` must correctly predict `x_b`. A route cannot win
globally by destroying reversed or scaled commands. Binding accuracy is the fraction of
pairs satisfying `e_aa < e_ab` and `e_bb < e_ba`; it is a diagnostic, never the camera
execution acceptance metric.

The contrast is deliberately named **target-view binding**, not proof of camera-only
binding: a model may distinguish the pair through the correctly camera-rendered scaffold,
the rays, the command code, or their interaction. Camera-specific evidence comes only from
the predeclared command/ray interventions with the scaffold regenerated consistently for
the intervened camera. Even those interventions establish route reliance, not the causal
effect of a separately trained component.

Unknown regions are learned through positive flow examples across many real cameras.
The contract makes no false claim that contrastive masking identifies a unique unseen
appearance.

---

## 5. Stationary batch and long-horizon coverage

Gradient accumulation is eight. The row-role counts and all sampling distributions are
identical for all 12,000 optimizer steps:

| Rows | Fixed role |
|---:|---|
| 4 | Clean positive rows: translation-hard, rotation-hard, mixed, and low-motion appearance reserve |
| 1 | Symmetric paired-target binding draw; two positive and two cross-conditioned forwards |
| 2 | Deep causal-prefix rows with pixel decode/re-encode and deployment roles |
| 1 | One-hop current-model self-augmentation row with an exact 30-step no-grad rollout |

“Stationary” means that the row-role sampler, depth PMF, attempt budget, admission rules,
and fallback kernel are fixed. It does not pretend that samples produced by an improving
model have a time-invariant content distribution.

The two deep rows draw history depth independently from this fixed distribution:

```text
band             probability       sampling inside band
1..3                 0.20          discrete uniform
4..9                 0.20          discrete uniform
10..18               0.25          discrete uniform
19..27               0.35          discrete uniform
```

Thus saturated `H=20` and retrieval-active states are ordinary training events, not late
curriculum stages. Deep rows use causally available real pixels marked explicitly as
teacher-forced but assigned the exact deployment recurrent-history role and neutral
authority. They execute deployment eviction, archive query, target-view scaffold
construction, and pixel decode/re-encode representation. They teach long-state operation
without pretending that clean history reproduces accumulated generation error.
Teacher-forced pixels do not receive automatic geometric authority: their model-facing
depth comes from the deployment estimator, and their simulated archive/scaffold admission
must pass the same certification predicates as a generated endpoint.

The self-augmentation row supplies the complementary generated-feedback exposure. It
runs one current-model call without gradient, decodes pixels, runs the Lyra-aligned DA3
window, checks feature-track and cross-view consistency, and re-encodes through the
persistent VAE path. The next supervision command is recomputed from the pipeline
endpoint to the next registered real target. If the DA3 window or confirmation checks
are invalid after a fixed attempt budget, the row becomes an ordinary clean positive row
and records an abstention. The attempt budget and fallback are fixed before step 1.

This is the declared approximation imposed by one-GPU compute: long state occupancy is
trained with causal real prefixes, while representation error is trained with one exact
generated handoff. The 30-call no-grad rollout is therefore a genuine test of a remaining
hypothesis, not something the document claims to have fully simulated in training.

Motion roles are sampled independently; no single fused score may erase pure rotation or
low-motion examples. The exact categorical PMFs and RNG derivation are serialized in the
recipe manifest.

---

## 6. Memory admission and the load-bearing chain

Every generated endpoint records separate fields for commanded pipeline pose, DA3
alignment health, feature-track agreement, content origin, geometry status, appearance
status, and confidence. Pipeline-pose agreement is not presented as an independent
realized-camera estimate: the Lyra DA3 call receives that pipeline pose as an alignment
input.

A generated endpoint becomes geometry-certified only if all conditions pass:

1. the joint Lyra-aligned DA3 window is finite and its pipeline path is nondegenerate;
2. DA3 alignment completes at the supplied camera scale without fallback or exception;
3. feature-track/RANSAC relative motion agrees with the supplied pipeline motion inside
   its calibrated interval;
4. reprojection and scale-invariant depth agree with at least one already certified view
   on sufficient covisible support;
5. depth confidence and supported area exceed fixed thresholds;
6. the estimator input-frame IDs are causal and recorded.

The feature-track check is algorithmically separate evidence; a second frame produced by
the same model is not called statistically independent. Failure on any condition leaves
the endpoint out of persistent geometry and archive K/V.

Uncertified generated output may remain in temporal appearance history for at most the
normal `H=20` lifetime with capped authority. It receives no persistent archive entry and
cannot become a scaffold geometry source. Geometry and appearance certification are
reported separately.

The actual dependency chain is stated honestly:

```text
anchor gauge
  -> rays + command + anchor scaffold
  -> generated endpoint
  -> aligned-DA3 and cross-view certification
  -> recent certified scaffold
  -> archive growth and long-horizon revisit support
```

Training self-augmentation and deployed recurrent inference call the same
`execute_omega_deployment_hop` implementation. That function owns the frozen 30-step
Wan Euler sampler, pixel decode, joint pose/depth recovery, certification, VAE re-encode,
memory commit, and realized-pose command replanning. A separate evaluator or demo loop is
not permitted to reimplement those transitions.

The immutable anchor is a floor, not proof that the chain cannot starve. Coverage,
certification rate, time-to-first-certified-write, archive fill, retrieval frequency,
reader mass, and supported/unknown-region quality are mandatory metrics.

---

## 7. Optimization

- Route parameters and Lyra-selected Wan weights are fp32 and updated in place. Frozen
  Wan weights remain bf16. There is no second full-model CPU master copy.
- Optimizer: fresh `PagedAdamW8bit`, betas `(0.9, 0.999)`, epsilon `1e-8`.
- Weight decay: Lyra-matched `0.001` on matrix weights; zero on biases, norms, gates,
  and embeddings.
- New residual routes: peak LR `3e-5`.
- Reader gate/calibration and confidence embeddings: peak LR `1e-5`.
- Selected Wan self-attention, stock patch embedding, and copied history patch
  projection: peak LR `2e-5`.
- Exact released-Lyra schedule: 100-step linear warmup from multiplier `1e-6` to
  `0.99`, then linear decay toward `0.4` over the released 400,000-step cycle. The
  ΩR run stops at step 12,000, where the multiplier is approximately `0.9724`; it does
  not compress Lyra's schedule into an artificial end-of-run cooldown.
- Lyra-matched global gradient clipping: `0.1`.
- Every loss-bearing microstep samples a fresh standard-normal epsilon and a fresh
  logit-normal training time on the 1,000-bin Wan shift-5 schedule, with the released
  Lyra timestep reweighting. The draw is keyed by contract, optimizer step, and row so
  exact resume replay is possible; a cached clean row never reuses corruption across
  optimizer steps. The two binding positives share the same freshly sampled epsilon,
  sigma, and timestep.
- Every conditioned row also resamples Lyra's per-frame `exp(N(-3, 2^2))`
  warped-latent sigma and standard-normal scaffold noise from clean cached features. Cached
  rows contain the clean features explicitly; pre-corrupted scaffold latents are never
  replayed at another optimizer step. Inference uses the frozen `0.001` sigma inside
  this training support.
- No averaged parameter copy is maintained. Every checkpoint contains the current
  trainable route weights plus the exact optimizer, step-counter, and RNG resume state.
- Exactly 12,000 optimizer steps; no loss ramp, module schedule, curriculum, or mid-run
  intervention.

Before source-stream work, one optimizer step contains nine positive loss-bearing
forwards, two cross-conditioned binding forwards, and one 30-forward no-grad rollout.
The resource projection must therefore price at least 41 DiT forwards per step plus
archive-source advancement; a clean-row timing extrapolation is not accepted.

The recipe manifest must bind the complete frozen eligible DL3DV training population.
Sparse cache rows memoize expensive deployment-parity geometry/VAE results; they may not
turn the old 28-scene resource-smoke population into production data. The recipe also
declares a numerical wall-time allocation and checkpoint-disk
allocation before the resource smoke. “Inside the declared allocation” is otherwise not
a test. After optimizer-state materialization and two complete updates, conservative
physical free VRAM must remain strictly positive. The smoke's exact fixed-shape workload,
allocator configuration, and peak reservation are recorded rather than replaced by an
arbitrary percentage threshold.

The fixed `0.10` binding weight is admissible only if the preflight gradient audit finds
its output-projection gradient norm between `0.1x` and `1.0x` the flow gradient norm on
the binding row. If it lies outside that range, the implementation is not launch-ready;
the weight is not silently tuned during training.

---

## 8. Camera instrument calibration

`execution_magnitude_ratio` is not trusted merely because it is monotonic. Calibration
uses at least 20 held-out registered static sequences and camera trajectories rendered or
selected at known `0.25x, 0.5x, 1.0x, 1.5x` translation magnitude with matched rotation.

The primary recovery is run in a common anchored reconstruction whenever the backend
permits it. Otherwise each recovery is converted through the same anchor-depth gauge.
The primary backend is the exact pinned VIPE wrapper released with Lyra, including
Lyra's own default overrides. Frozen VGGT is the independently implemented confirmation
instrument. Feature-track/RANSAC reconstruction is a third truth-control health check;
it is not substituted for either required recovery backend.

The magnitude instrument passes only if:

- Spearman rank correlation with commanded magnitude is at least `0.95`;
- median absolute log-ratio error is at most `log(1.10)`;
- 95th-percentile absolute log-ratio error is at most `log(1.25)`;
- fitted slope is in `[0.9, 1.1]` and intercept in `[-0.05, 0.05]`;
- all truth-control health criteria pass;
- the confirmation backend agrees on accept/reject for at least 90% of cases.

Thresholds in §11 are frozen only after this calibration passes. If it fails, the
instrument is redesigned and recalibrated before training; no model result is interpreted
through an unvalidated scale ratio.

---

## 9. Evaluation design

### 9.1 Sets and repetitions

- The executable held-out transaction contains at least 20 pose-reliable scenes,
  stratified translation, rotation, mixed, and low-motion commands. Every scene is run
  through all 28 deployment calls (224 generated RGB frames) with a fixed seed.
- Final comparison adds a second paired diffusion seed; pause tripwires keep the first
  fixed seed so they do not double the cost of every checkpoint decision.
- Historical rollouts remain qualitative/debug controls only and cannot satisfy the
  executable held-out population requirement.
- Scene IDs, commands, seeds, estimator versions, and checkpoint hashes are frozen before
  step 1.

Before freezing the set, a power analysis using paired variance from immutable baselines
must show at least 80% power for the `0.02` first-call LPIPS margin and a `0.10` execution-
magnitude difference. Scene or seed counts are increased if the stated minima do not
meet that requirement.

All model comparisons are paired on scene, command, target frame, and noise seed.
Reported differences include scene-block bootstrap 95% confidence intervals. A point
estimate alone cannot pass a gate.

### 9.2 Metrics

Primary camera metrics:

- calibrated execution magnitude ratio;
- executed progress fraction;
- stall and reversal fractions;
- rotation execution ratio and relative-rotation error;
- normalized Sampson/TSED-style multi-view consistency on eligible pairs.

Primary appearance and stability metrics:

- LPIPS, PSNR, and SSIM to available real targets;
- terminal sharpness and artifact energy;
- cross-call boundary excess;
- instance/feature persistence;
- supported-region, disagreement-region, and unknown/disocclusion-region splits;
- collapse rate over 28 calls.

Admission telemetry is never folded into one quality score.

### 9.3 Intervention tests

Evaluation-only interventions test route reliance, not training causality:

- scaffold off, reported separately in supported and unknown regions;
- command/ray shuffle with scaffold kept internally consistent with the shuffled camera;
- sub-frame order shuffle on trajectories with unequal within-latent motion;
- archive reader gate forced to zero after retrieval begins;
- recent certified source removed while retaining the anchor.

Each intervention has a predeclared target metric and must change it by at least 5% with
a paired 95% interval excluding zero. Failure means the route is unused, redundant, or
the metric is insensitive; it does not authorize an in-run weight change.

One production run can establish performance and route reliance for the integrated
system. It cannot establish the causal contribution of each training loss or module.
Any thesis claim of component causality requires separately trained ablations outside
this contract.

---

## 10. Blocking preflight

Training launch requires one content-addressed, human-approved JSON report with schema
`gaussverse.gate_omega_preflight.v1` in which every item below passes. Each requirement is
produced by a dedicated artifact carrying the contract ID, requirement ID, implementation
source hashes, and a canonical-payload SHA-256. The aggregate report also carries its own
payload hash, source commit, approver identity, and approval timestamp. This is an
integrity/approval record, not a claim of cryptographic signature unless an actual signing
key and verification command are named. The fail-closed aggregator is
`python -m gaussverse.bench.gate_omega_preflight`.

1. Fresh-Wan base fingerprint and exclusion of every 20k/21k/22k tensor.
2. Generated route metadata from actual saved tensor prefixes; no hand-written route
   list.
3. Step-0 equality and output-gradient tests from §1.3.
4. Unified-gauge equality and Sim(3)-equivariance tests from §2.
5. Ordered-ray shape/order tests and command-code gauge tests.
6. Scaffold visibility, hole-mask, depth-tie, and source-disagreement tests.
7. Frozen scaffold diagnostic manifest; raw deployment-estimator geometry and zero
   GT-depth conditions in the production dataloader.
8. Prefix-causality attestation: every geometry input frame ID is no later than the legal
   history boundary, and no target RGB/depth appears in conditioning.
9. Binding-pair bijection: both cameras occur as positives, each scaffold matches its
   camera, shared noise/timestep are exact, and cross pairs cannot enter `L_flow`.
10. Long-depth sampler frequency test over at least 100,000 offline RNG draws and an
    executed saturated `H=20, K=5` true-path smoke.
11. Generated one-hop pixel decode -> estimator -> certification -> re-encode true path.
12. Archive positive and negative admission fixtures, including estimator disagreement.
13. Camera instrument calibration from §8.
14. A deterministic operational collapse detector with executable synthetic clean,
    white, washed, and structural probes. Human legacy labels are an advisory empirical
    upgrade and are never fabricated as a launch prerequisite.
15. Gate repair: current-arm magnitude and degeneracy are binding fields.
16. Disposable two-update production-shape resource smoke after paged 8-bit state
    materialization: the exact stationary row mix and all 41 or more forwards execute,
    losses/gradients are finite, measured free VRAM satisfies §7, and projected wall time
    and checkpoint disk fit the numerical allocations frozen in the recipe manifest.
17. Full focused tests plus the existing repository regression suite.

The two-update smoke uses disposable weights and a separate RNG namespace. It is neither
a parent nor a hyperparameter search.

---

## 11. Tripwires and final acceptance

Tripwires use the content-addressed held-out population (at least 20 complete scenes)
and fixed seed. They may abort but never change the method. Thresholds below are
learning-stage safety checks, not claims that a partially trained model already meets
final SOTA acceptance.

| Point | Abort condition |
|---|---|
| Step 0 | Any §1.3 equality/gradient assertion fails; wrong base or manifest hash; nonzero new output; noncausal source; invalid instrument |
| 1k | Nonfinite training state; mean/terminal fidelity exceeds its independent bound; first-call LPIPS regression `> 0.04`; decoded collapse |
| 4k | Any 1k failure; VIPE/VGGT agreement or nondegeneracy fails; execution-magnitude median `< 0.45`; stall `> 0.40`; reversal `> 0.25` |
| 8k | Any 4k failure; execution-magnitude median `< 0.65`; stall does not improve strictly over step 0 |

Final acceptance is evaluated after training on §9's two-seed paired protocol and
requires all of:

- execution-magnitude median in `[0.80, 1.20]`;
- scene-level 10th percentile at least `0.65` and 90th percentile at most `1.35`;
- executed progress fraction at least `0.80`;
- stall fraction at most `0.15` and reversal fraction at most `0.10`;
- rotation execution median in `[0.85, 1.15]`;
- zero model-confirmed degenerate trajectories; estimator-only failures are adjudicated
  by the confirmation backend and reported, never silently dropped;
- zero white/washed or structural-collapse rollouts across the accepted evaluation set;
- first-call LPIPS noninferiority to fresh Wan with margin `0.02`, using an upper 95%
  confidence bound;
- full-rollout LPIPS paired difference upper 95% bound at most `0.03` versus the
  strongest eligible historical baseline;
- persistence retains at least 95% and terminal sharpness at least 90% of that baseline;
- cross-call boundary excess is no more than `0.05` worse than that baseline;
- every §9.3 intervention reaches its target metric effect;
- at least one geometry-certified generated write and one later successful retrieval per
  eligible rollout scene, or an explicit failure of the memory hypothesis;
- no provenance, causal-prefix, checkpoint, or evaluator-integrity violation.

The selected checkpoint is the best checkpoint satisfying every safety and
noninferiority constraint. Camera score may rank only among eligible checkpoints; it may
not trade away a failed constraint. If none passes, the result is a falsified integrated
hypothesis, not a near-success promoted through softer wording.

---

## 12. Pre-registered interpretation of failure

No fallback is silently substituted into this run. Post-run diagnoses are hypotheses for
separate experiments:

| Observed result | Supported interpretation | Separate next experiment |
|---|---|---|
| Scaffold intervention has no supported-region effect | Scaffold route unused or metric insensitive | Wiring/representation study before any loss change |
| Binding accuracy rises but raw execution does not | Conditional sensitivity without correct camera realization | Replace binding representation or add a separately validated geometric objective; do not increase `L_bind` |
| First-call execution holds but deep execution decays | Recurrence/certification feedback failure | Realized-residual feedback with its own zero-init contract |
| Supported quality improves but unknown quality regresses | Copy dependence or insufficient generative completion | Routing-only scaffold ablation and unknown-region data/objective study |
| Archive fills but reader intervention is null | Reader transfer/routing failure | Host-reader topology study |
| Archive does not fill | Certification starvation | Calibrate the failing certification component; never relax all gates together |
| Quality falls while camera execution improves | Full-teacher updates or Ω routes interfere with the prior | Separate discriminative-LR, selective-self-attention, and route-capacity ablations |

These rows do not claim their proposed experiments will work. They prevent a failed run
from being retroactively explained by an unfalsifiable story.

---

## 13. Grounding and novelty boundary

The following precedents justify mechanisms, not originality claims:

- [ControlNet](https://arxiv.org/abs/2302.05543): zero-initialized residual attachment;
  ΩR borrows the safe step-zero mechanism but deliberately trains the full carrier.
- [GEN3C](https://arxiv.org/abs/2503.03751) and Cosmos: target-view render conditioning,
  3D cache guidance, and sigma-conditioned latent corruption.
- CameraCtrl and [AC3D](https://arxiv.org/abs/2411.18673): Plücker/ray conditioning and
  selective camera injection.
- [Classifier-free guidance](https://arxiv.org/abs/2207.12598): conditional dropout and
  score combination; intentionally not included here because a complete
  camera-unconditional branch is outside this run.
- [SANA-WM](https://arxiv.org/abs/2605.15178) released implementation: zero-output camera
  branches in a pretrained video-transformer setting.

Potentially publishable contributions, contingent on results and a formal literature
review, are:

1. the ordered four-RGB-per-latent interface combined with one deployment-only gauge;
2. the zero-gated host-weight joint-minus-main archive reader;
3. symmetric positive target-view binding in which every counterfactual command is also a
   supervised positive, with camera-only causality left to separate interventions;
4. geometry-certified recurrent scaffolding with explicit supported/unknown evaluation;
5. the integrated stationary long-horizon single-run contract.

The paper may say “we propose” these items. It may not say “first,” “no precedent,” or
“state of the art” without a completed search and the corresponding experimental
comparison.

---

One sentence: start exactly from the clean Wan carrier, train the Lyra-selective Wan tier
and all Ω routes with deployment-matched certified scaffolds and positive camera binding
over the complete eligible population, expose saturated state from the beginning, and
let the shared deployment rollout plus calibrated held-out evaluation falsify the
integrated hypothesis.
