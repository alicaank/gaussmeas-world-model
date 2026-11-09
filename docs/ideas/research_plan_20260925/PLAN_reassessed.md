**GaussVerse research plan — reassessed against the full discussion, 25 September 2026**

The objective is faithful, camera-controlled scene exploration from the observed pair: preserve observed content, generate plausible new regions, retain the generated world's identity, and continue reliably through repeated calls. Improve the current model before committing to a new mapper or a paper claim.

The central working question is whether training and conditioning can preserve scene identity through the actual generation → history update → reconstruction → conditioning transition, while still responding correctly to camera changes and useful new evidence. Source retention is one candidate intervention. Persistent Gaussians are a separate representation choice. Neither is established as the principal solution or novelty.

This revision supersedes the priorities and contribution framing in [the earlier plan](PLAN_before_full_review.md). The original numerical snapshot remains [evidence_snapshot.json](evidence_snapshot.json); the revised work breakdown is [experiments.csv](experiments.csv). No new model experiment or training was launched by this review.

**1. Corrections to the earlier recommendation.**

First, the deployed system already stores source content. Endpoint records retain RGB and latent tensors, a bounded temporal history, and a persistent archive for certified endpoints. ReSplat selects views from archive plus history and reconstructs a fresh Gaussian scaffold. The meaningful distinction is between retained sources, successful use of their information, and persistent Gaussian primitives. Adding a source bank, confidence field or provenance record is not automatically a new capability.

This was rechecked in the deployed runtime's [memory state](../../training_runs/age_continuation500_pair_20260914/bounded/runtime/gaussverse/memory/gate_omega_memory.py) and [ReSplat source reader](../../training_runs/age_continuation500_pair_20260914/bounded/runtime/gaussverse/memory/resplat_scaffold.py). Retention is bounded or conditional on certification; it does not mean every past generated frame survives indefinitely, or that a certified generated view is true.

Second, the real-history oracle supplies new true past images throughout the trajectory. Its benefit includes newly available scene information. It does not establish that preserving the first photograph can reproduce that benefit. The initial anchor may have no useful local support later.

Third, the observed 0.1997 LPIPS with real history and memory is an imperfect pipeline reference. The real inputs do not make depth, registration, certification or causal decoder state perfect. It is not a per-call cost or irreducible floor.

Fourth, post-hoc alignment supports investigating appearance preservation but cannot isolate every geometric contribution. In particular, it does not measure content corruption that earlier geometry errors caused.

Fifth, the training and call-transition evidence deserves equal attention to memory readout. Recent continuation rows are overwhelmingly shallow and produced by a frozen parent using one-step approximations. Earlier audits already proposed valid continuation targets after the actual deployment handoff. The latest alignment result does not supersede that evidence.

Finally, neither source retention nor transition-aware training is established novelty. Mechanism tests and strong simple baselines must determine the contribution. Do not choose a paper slogan first and interpret every new result as supporting it.

**2. What is established, and what remains a hypothesis.**

| Evidence | Supported interpretation | Interpretation to avoid |
|---|---|---|
| Deployment LPIPS 0.4180; real memory 0.3297; real history 0.2611; both real 0.1997 | State quality and available evidence strongly affect continuation | Assigning independent percentages of blame to two modules |
| Early real-history refresh washes out within roughly 3–5 calls | Better initialization alone does not sustain the observed benefit | Assuming continual anchor conditioning solves the same problem |
| Real-history calls 16–20 approximately 0.2854 versus deployment 0.5287 | Continually refreshed true history substantially limits degradation in this protocol | Claiming this performance is reachable with the same sparse deployment observations |
| Late aligned LPIPS 0.4827 versus raw 0.5038 | This particular smooth warp repairs little of the existing output error | Calling 96% of error content-only |
| Observed scaffold validity approximately 58.8% late | Initial observations still have nominal support in these four trajectories | Guaranteed useful detail, or support on arbitrary longer trajectories |
| Camera-route probes weak; appearance route much stronger | Current control depends heavily on the scaffold in the tested states | The model has no camera control, or a new camera branch is necessarily required |
| Correct gauge and bit-exact same-source reconstruction | There is no per-call coordinate reset | Different reconstructions are equally correct |
| Sticky/covisible sources reduce render changes but not video error | Small reconstruction stabilisation alone has not helped | All geometric memory improvements are useless |
| Native history latents 0.424 versus RGB feedback 0.418 | This history VAE bypass did not fix our rollout | Our main problem is proven to be latent–RGB cycling |
| Fresh-Adam continuation 0.4433; restored-Adam 0.4307 | Optimizer reset contributes, but does not explain all regression | Restoring optimizer state makes the recipe successful |
| Persistent-map real replay fails quality | The untrained incremental transfer should not be deployed | Persistent mapping is fundamentally inferior |

Primary evidence: [camera and architecture report](../report_pose_resplat_20260924/README.md), [interventions](../report_pose_resplat_20260924/03_interventions.md), [training diagnosis](../aligned_history_long_20260923/DIAGNOSIS.md), [oracle audit](../oracle_full_review_20260921/OKUMA.md), [alignment review](../forward_walk_decomposition_20260924/REVIEW.md), [persistent trial](../incremental_resplat_20260924/README.md), and [completed optimizer controls](../long_eval_pilots_20260923/summary.json). Historical reports saying the optimizer control is running are stale; the completed results above take precedence.

**3. The architecture problem to investigate.**

The deployed loop generates eight RGB frames from two latent steps, feeds generated appearance into temporal history, estimates geometry from generated imagery, selects at most eight source views, reconstructs ReSplat, and renders conditioning for the next command. Those operations share one world gauge. They do not necessarily share an accurate account of the scene.

This creates three coupled research problems:

- **Conditional fidelity:** given available reliable evidence, does the next call use it without unnecessarily changing supported content?
- **Recurrent reliability:** after generated output passes through the real history, VAE, geometry, certification and reconstruction updates, does the next call preserve that content? Can errors be corrected rather than reinforced?
- **Controllability:** can the viewpoint change correctly while content stays stable, including where depth or initial-anchor coverage is poor?

The same drifted appearance can appear in both history and reconstructed memory. Their agreement is not independent evidence of correctness. Conversely, making all generated sources weak prevents exploration. The useful behavior is to preserve supported content, expand where information is absent, and revise only when evidence justifies it. This is a design requirement, not proof of a novel method.

Keep the present parent, world gauge, two-observation bootstrap, fresh ReSplat and eight-frame deployment interface as references. Test changes to training, evidence access and camera conditioning separately before combining them.

**4. Phase A — make the next comparisons trustworthy.**

A0: freeze parent weights, runtime, optimizer policy, frame IDs, noise, camera convention, source budgets and splits. Preserve immutable parent evaluation alongside each matched trained control.

A1: use the completed optimizer comparison to choose and document matched treatment of existing parameters. New adapters need explicitly initialized groups. Restoring by parameter identity is required when group layouts change.

A2: consume the already queued observed-history probe without duplicating or preempting it. Audit its unchanged arm against deployment. A full or low-frequency render substitution at inference tests that interface without training; a negative result does not reject a trained reader.

A3: retain raw LPIPS as primary and extend alignment calibration with valid support, border exclusion, occlusions and known-transform controls. Report warp magnitude, failure rate and scored area.

A4: qualify the oracle state intervention. The previous audit found generated pose-reference images being matched against real endpoints in real-memory arms, affecting certification and authority. Implement a separately labelled reference-matched oracle control before attributing the remaining error to backbone quality. Preserve the old results, and keep generated displayed outputs unchanged.

Deliverable: a compact reproducible state suite spanning early, middle and late calls, with clean and generated states, plus an explicit record of what each oracle replaces and what remains generated.

**5. Phase B — locate the approximately 0.20 residual.**

Run these on the fixed-state suite before large architectural changes:

| ID | Test | Decision it informs |
|---|---|---|
| B0 | Reproduce real-history/real-memory continuation with qualified state semantics | Clean-input reference |
| B1 | Matched causal VAE reconstruction and encoder/decoder-state controls | Whether representation or cache handling deserves attention |
| B2 | Compare reliable source evidence with generated output on shared visible surfaces | Whether usable content exists but is lost |
| B3 | Existing scaffold readout versus direct qualified source access at matched budget | Whether reconstruction/readout destroys useful information |
| B4 | Explicitly privileged target-view evidence through the candidate interface | Whether the route can express a strong correction |
| B5 | One larger sampling budget and a separate source-resolution control | Whether generation budget or source detail explains a useful fraction |

Reuse the [existing VAE factorial](../matched_target_alpha_20260919/README.md). A stateless reconstruction and the actual causal decoder are different experiments; VAE reconstruction is not a mathematical lower bound on generation error.

Use independent masks for reliable observed regions, uncertain/magnified regions, and unsupported regions. Compute regional LPIPS with documented spatial support or fixed crops; do not black-mask RGB and score artificial mask boundaries. Always report full-frame scores and area.

If source evidence is wrong, fix projection before stronger copying. If it is correct but ignored, test the interface. If errors largely concern unseen content, evaluate plausible generation separately from matching one recorded future. If the oracle route fails, inspect placement and feature compatibility before adding capacity.

Do not wait for a perfect clean-input model before testing recurrence. A repeatable actionable gap is enough to start a small matched experiment.

**6. Phase T — test the actual call transition as a training target.**

This branch runs independently of the new source reader; C is not its prerequisite. Restore the central idea from the [earlier integrated research design](../training_recipe_design_20260921/TUM_PROBLEMLER.md) and [concrete continuation proposal](../training_recipe_design_20260921/ONERI.md), while treating their proposed teacher and mechanism as unverified.

T0: keep a displayed prefix, available observations, requested camera and valid continuation target fixed. Measure supported content before and after the real deployment update. Test semantically equivalent source representations only on common reliable support. Reconstructing identical inputs is a determinism control, not evidence of a harmful transition. Changing available information or comparing a long bidirectional call with two causal calls is not a pure boundary intervention.

T1: establish compatible prefix/continuation pairs. Real or synthetic known-scene data can provide valid targets for controlled cases. A longer teacher can provide generated A/B pairs only after it demonstrates adequate content and motion on the relevant early and late states. Do not assume G4 is a superior teacher. It sees more future context and its prefix may depend on later commands.

T2: with the same valid A/B pairs, compare ordinary continuation training against continuation after passing A through the actual history, geometry, reconstruction and VAE update. B is a supervision target only; it cannot enter retrieval or reconstruction. Preserve causal target encoding. Match data exposure and report GPU cost; retain clean examples and a control with extra ordinary training.

If the learner later generates a different A, the previous B is not automatically a valid target. Recondition a qualified teacher, use registered supported supervision, or use an appropriately justified distributional objective. Arbitrary learner past paired with the recorded future repeats an already unsuccessful approach.

Reject this mechanism as the central contribution if the transition causes no meaningful supported-content loss, or if training through it does no better than the same synthetic data without that intervention. Earlier null source-selection results make this falsification especially important.

**7. Phase C/D — test better use and preservation of already stored content.**

C0 is the matched current-interface training control. C1 routes reliable projected RGB through the trusted history interface. C2 reads compatible original-source features through a small adapter, keeping source coordinates, time, pose, visibility and effective resolution explicit. Use the existing endpoint storage first rather than building a duplicate cache.

Compare at equal active tokens, source counts and adapter capacity. Retain recent temporal context for motion and unsupported regions. A target-aligned pseudo-frame needs the same position/time convention in training and deployment. ReSplat update features are not diffusion appearance latents.

Use standard flow matching first. Add correspondence-based preservation only where a static visible surface is supported. Supervised training can use accurate labels; the deployed mask must depend only on available evidence. A learned reliability router is justified only if oracle selection beats simple rules on held-out examples.

D0–D3 test generated sources as well as real ones: current source selection/readout, the same sources with the new reader, and longer-lived direct source reuse. This separates improved access from more storage or more context. Record eviction, ancestry, resolution and actual source use through paired source removal/swap tests.

A proposed write-policy ablation distinguishes newly generated support from repeated views of an already stored surface. Test whether avoiding unnecessary replacement of supported content helps, while allowing improved detail or corrected geometry when warranted. This is a hypothesis; hard freezing can preserve mistakes. The baseline already has confidence and certification, so compare against those existing rules.

D4 is required for a beyond-anchor claim. Use trajectories that actually leave initial useful support, not merely an anchor dropout while it remains visible. Label support independently using known geometry where possible. Proposed bins are above 50%, 5–50%, and below 5%; report known zero support separately. Require a sustained unsupported interval, provisionally five calls, and profile longer horizons before fixing them.

Measure consistency of generated surfaces since their first appearance, absolute quality, camera execution, newly exposed content, and revisits. Retention may reduce repeated changes to one surface; it cannot remove the dependence of newly invented surfaces on imperfect prior generated context.

Separate closed-loop correction of supported contradictions from observation-assisted correction using a new real image. The latter changes the input protocol. Without new evidence, no map can guarantee the true appearance of a never-observed room.

**8. Phase F — camera control remains an independent research branch.**

Camera evaluation begins immediately, and the conditioning experiment need not wait for source-reader success. The full discussion supports testing content preservation and camera control independently, then jointly. The alignment result does not justify postponing all camera work, but camera-only adjustments are not an established cure for fidelity.

First test valid paired camera futures on the current architecture. If that is insufficient, compare the N1 depth-based motion carrier and depth-independent relative pose/ray information under the same training. N1 is reliable only on limited support in difficult generated scenes. Do not invent dense accurate displacement where depth is unavailable.

Use zero-motion, translation magnitude, rotation and changing-command tests with valid corresponding targets. A deliberately wrong camera paired with an unchanged target is a sensitivity probe, not a valid control-learning target. Compare reader only, camera change only, and both if the individual arms show benefit.

Keep commanded pose and estimated content pose separate. Shared gauge does not make generated content correctly registered. DA3 aligned-input poses are not an independent observer. Previous measured-pose writes degraded motion; do not reinstate them without a valid observation/control design.

**9. Phase E — qualify longer learner-generated histories.**

After establishing valid targets in T1, test realistic history exposure independently of the interface. Compare current fixed-parent one-step histories with periodically refreshed learner-sampler states. Start with detached two-to-four-call histories plus a bounded cache of later states; balance early/middle/late exposure and record actual counts.

Keep clean-data fidelity checks. Do not require the new reader before testing this branch. If both branches help, use an interface × training factorial. Ordinary corruption training and ordinary short-rollout training are required controls.

For paired perturbation experiments, report absolute source-fidelity errors in both branches as well as their difference over later calls. A shrinking difference can mean both outputs became equally wrong. Recovery must preserve command response and useful source dependence.

Do not claim a contraction guarantee or unlimited stable rollout from finite empirical recovery curves. A better local generator and a less damaging recurrence are separate properties; both matter.

**10. Phase G — persistent Gaussians, if reconstruction remains limiting.**

The existing trial preserves primitive identities but substantially degrades next-view RGB L1: fresh 0.0373/0.0231 versus update-plus-retirement 0.1525/0.0812 on plaza/sign. It used estimated depth and transferred an update head without sequential training. It is not a reproduction of Robust Dreamer and not a decisive test of learned persistent mapping.

Reopen mapping if real replay demonstrates that reconstruction/association is a remaining bottleneck, or if better content access leaves that bottleneck intact. Success of a new source cache is not a logical prerequisite for all mapper research, but it is a useful cheaper alternative to compare.

Train sequential geometry/Gaussian proposals and fusion with held-out later views and retained-view checks. Audit gauge, association, covariance bases, scale, opacity, free-space contradictions and bounded growth. Require real-replay parity with fresh ReSplat before generated rollout integration. If direct reads and training suffice, a persistent Gaussian mapper can remain outside the first paper.

**11. Evaluation and decision criteria.**

The four existing scenes are development-only. Screen with paired seeds 7 and 11 at fixed states and 160 frames. Add sustained low/zero-anchor-support paths. Validate on at least 12 disjoint scenes and three inference seeds before final testing. Target at least 30 held-out scenes, multiple visual domains where feasible, three inference seeds and at least two training seeds for principal method/control; disclose actual scope if resources limit this.

The following are proposed planning thresholds, not proven significance or promised performance. Freeze them before new validation results:

| Gate | Proposed criterion |
|---|---|
| Clean-input fidelity | At least 5% relative full-frame LPIPS improvement, or 10% on prespecified reliable regions with no more than 2% full-frame regression; broader fidelity claim requires overall gain |
| Deployment | At least 5% relative improvement in both all-frame and calls 11–20 LPIPS over the better of parent and matched trained control |
| Control | No systematic worsening; provisional execution-ratio absolute-error margin +0.05, with separate stationary/rotation/return-path metrics |
| Measurement support | Gains cannot depend on dropping hard pairs; provisional maximum tracking-support loss five percentage points |
| Beyond-anchor claim | Improvement remains after useful initial support disappears, under matched memory and compute budgets |
| Mechanism | Beats the simple alternative that could explain the same gain, and source/transition interventions support the claimed cause |

Track raw LPIPS/PSNR/SSIM; qualified aligned LPIPS; reliable-region appearance; source resolution and coverage; absolute seam errors and interior motion; Sampson with match counts/failures; calibrated magnitude and rotation; revisit and correction behavior; latency, peak VRAM and storage.

Report fidelity to a recorded real scene separately from consistency of a plausible generated world. Neither stable mistakes nor visually plausible drift should pass as the other. Use paired scene-level confidence intervals with inference seeds nested in scenes, not frame-independent statistics.

Required ablations: parent versus extra training; source reader with identical sources; RGB versus feature readout; actual-transition versus ordinary continuation with the same targets; realistic versus surrogate histories; memory alone/camera alone/both; observed-only versus generated sources; recent versus retained sources at equal budget; existing certification versus any new write policy. A second memory backend is valuable if claiming a general transition mechanism.

**12. What could support a paper, and what already exists.**

The candidate scientific distinction is learning which changes should affect the generated world: valid camera changes must change the view; useful new observations should update content; a lossy representation refresh should not unnecessarily change a supported surface. These conditions need matched interventions. Invariance alone can be solved by ignoring memory, and continuity alone by reducing motion.

This is a research hypothesis, not an established novel method. Source-use tests, same-data training controls and a related-work audit decide whether the result warrants a broader contribution.

- [Robust Dreamer](https://arxiv.org/html/2605.30855v2) inherits generated latents into Gaussian memory and trains with stage/timestep-indexed deviations. Its benefits cannot be attributed to persistence alone, nor transferred to our system without matching representation and training.
- [Warp-as-History](https://arxiv.org/abs/2605.15182) already uses target-aligned warped pseudo-history and visibility selection. History routing alone is insufficient novelty.
- [World in World](https://arxiv.org/abs/2609.11548) already combines heterogeneous visual evidence, correspondence routing and retrieved generated states. Generic evidence routing is insufficient novelty.
- [Wonder](https://arxiv.org/abs/2607.26037) combines dense visual camera conditioning, sparse history retrieval and changes to training. A new control carrier alone is insufficient novelty.
- The [local implementation audit](../sota_implementation_audit_20260921/AUDIT.md) documents relevant Lyra, Alaya, SANA, EVOKE, GEN3C and VMem mechanisms. Reuse these pinned sources and distinguish released code from paper-only descriptions.

Our history-latent bypass result weakens latent–RGB cycling as a sufficient explanation here; it does not test every direct latent-memory design. Likewise, a negative untrained map does not overturn a trained competing system's reported results. Match or disclose backbone, initialization observations, resolution, call duration, sampler and compute in external comparisons.

**13. Execution order and stopping rules.**

First, complete A and the minimal B suite, including the oracle-reference qualification and queued probe. Profile one eight-frame call, one 160-frame rollout, one training row and 20 updates. No calendar estimate is credible before this profiling.

Second, run the cheap falsification tests for T, C and F on the same state suite. Select the smallest informative trained pilot from their results. Source readout, actual-transition training and camera conditioning are independently testable branches, not a mandatory chain of new modules.

Third, evaluate promising candidates over 160 frames and after anchor support disappears. Add realistic-history training and generated-source policy only with valid targets and matched controls. Combine individually useful changes, then expand held-out evaluation.

Last, pursue persistent Gaussian training only if the measured geometry bottleneck warrants it. Do not spend the next cycle on inference-only source-selection tweaks or another training run judged only at 16 frames.

Use the shared 32 GB GPU serially without preempting existing jobs. Prefer cached source features, frozen encoders, small adapters and detached rollouts. Use the pinned executing runtime, save exact manifests and atomic checkpoints, and preserve the immutable parent. Test state causality, cache restoration, source immutability, pose/gauge conventions, adapter identity initialization and save/load behavior.

Stop or narrow the claim if a simple baseline explains the gain; if only seams improve; if motion or source detail worsens; if benefits vanish beyond the anchor; if the teacher is not adequate; or if only the four development scenes improve. The immediate goal is a verified better system and an identified mechanism. A venue outcome cannot be guaranteed by an architecture proposal.
