# Memorizon with measured Gaussian memory: research and recommended design

**Consolidated specification:** [Method, training and novelty comparison](METHOD_AND_TRAINING.md)
is the current design. This file retains the broader research exploration.

**5 November execution priority:** the user's limited-time constraint narrows the
first training architecture to native latent memory plus RGB-rendered Gaussian
conditioning. See [the training decision](TRAINING_DECISION_20261105.md). The richer
feature reader and learned state-repair proposal below remain research options,
not required components of that first run.

4 November 2026. Design/research review, not a new implementation or training
result. User-selected scope: fine-tune Memorizon on our data, reconstruct with
2Xplat, obtain measured cameras from that reconstruction, write memory at measured
poses during training and inference, and include camera correction in training.
Concurrent methods inform design and comparisons; their existence does not by
itself disqualify this project. No paper-acceptance or exhaustive-novelty claim.

## Recommendation

Keep Memorizon's native latent bank and add a scene representation that can be
revised. Use that scene both to improve geometric reconstruction and to retrieve
the original appearance evidence for generation. The candidate contribution is a
joint read/write policy trained to distinguish camera-registration error, map
error, conflicting appearance and genuinely unseen content during generated
rollouts. Test whether that distinction improves reconstruction AND subsequent
video, beyond adding RGB or latent renders alone.

This is a research hypothesis. Similar ingredients exist in mapping and video
generation; the whole behavior and its controlled gains must carry the argument.

## Closest work and what it changes in our design

| Work | Relevant mechanism | Consequence for our design |
| --- | --- | --- |
| [Memorizon](https://arxiv.org/html/2610.00544) | Per-chunk camera-based retrieval, bounded packed training bank, own-history distillation | Preserve its trained retrieval/context semantics. Longer spans or generated history alone are not our new contribution. |
| [Robust Dreamer](https://arxiv.org/html/2605.30855) | Inherited latent Gaussian memory and archived model-error training | Directly relevant baseline; latent storage and corrupted-history training cannot establish our distinction. See the review below. |
| [L3G-World](https://arxiv.org/html/2610.40614) | Persistent Gaussians, learned spawn/refine updates, stored VAE features directly rendered for generation | Closest Gaussian-memory reference. Merely adding residual updates or latent rendering is not a distinct mechanism. Its updater training supplies GT cameras; our proposed measured-pose revision is a specific comparison axis, not proof it lacks every related capability. |
| [WorldStereo](https://arxiv.org/html/2603.02049) | Global point geometry plus spatially constrained source-image attention; overlapping point-cloud registration | Use geometry for correspondence as well as a rendered structural condition. Scene registration through shared views is established machinery. |
| [Lyra 2.0](https://research.nvidia.com/labs/sil/projects/lyra2/) | Per-frame geometry routes historical information; self-augmented training addresses drift | Keep high-quality original appearance available rather than forcing every detail through a reconstruction. |
| [Mirage](https://arxiv.org/abs/2606.09828) | Persistent 3D latent cache and direct latent-space warping | Avoiding RGB/VAE round trips is useful but already studied. |
| [MosaicMem](https://arxiv.org/abs/2603.17117) | 3D-localized patches and a patch-composition interface using native conditioning | A hybrid explicit/implicit memory is also established. |
| [GLIMPSE](https://arxiv.org/html/2610.13706) | Original-patch retrieval with geometry-dependent precision of projected addresses | Uncertainty-aware memory reads and retaining source-patch provenance are not new by themselves. Our proposal must show value from correcting the upstream state and training reads together with those revisions. |
| [OctWorld](https://microsoft.github.io/OctWorld/) | RGB-D generation, inter-chunk depth normalization, adaptive octree fusion | Compact incremental mapping is relevant; continuous per-frame Gaussian concatenation is an inadequate baseline design. |
| [World from Motion](https://research.nvidia.com/labs/amri/projects/world-from-motion/) | Generates views conditioned on a 4D scene and distills them back into the reconstruction | Generation can improve reconstruction, but treating generated views as new evidence is not itself original. Its dynamic-video setting differs from our static exploration task. |
| [BayesianGS-SLAM](https://arxiv.org/abs/2609.24140) | Predictive RGB-D uncertainty used for mapping, tracking and keyframe selection | Joint uncertainty-aware pose/map optimization is prior art. Generated observations have additional correlated, model-produced errors that our training must address. |

Source review depth: Memorizon local data/model/pipeline and prototype code;
L3G-World updater/training sections; WorldStereo memory sections; GLIMPSE method
overview; other papers' primary abstracts/project descriptions. This is a scoped
design search, not an assertion that each related method was exhaustively audited.

## Existing code and implementation gaps

### Robust Dreamer review added 4 November 2026

Reviewed Sections 3.1–3.3 and Appendices A–D/F of the
[paper](https://arxiv.org/html/2605.30855). It associates generated latents with
CUT3R-derived global pointmaps/Gaussians and splats them for conditioning. Its
training-only archive injects one-step model errors into histories, indexed by
rollout stage and noise time. Reconstruction heads are trained before adapting
Wan2.1-14B with LoRA. It acknowledges reconstruction failures and separate memory
and generator training. Section 3.2 explicitly predicts global pointmaps from
image and pose tokens and uses those points as Gaussian centers: this is
image-inferred geometry, not evidence of commanded-pose memory placement. We
cannot claim inferred memory placement as our difference from Robust Dreamer.
I found no explicit planned/issued/measured command-correction policy in those
sections; that narrower distinction still needs experimental support. The
[project page](https://hlinchen.github.io/projects/Robust_Dreamer/) currently
labels code as forthcoming, so this is a paper review, not a reproduction.

**Design implication (our proposal).** Add a matched latent-splat plus
deviation-training control before attributing improvements to explicit state
repair. Keep the reconstructor, observations, backbone and training budget fixed.
Then add measured pose/map revision with the same read, followed by the
source-evidence reader. Finally isolate command correction. This separates
training robustness, representation, state repair and control; a different
backbone or 2Xplat substitution alone is not the proposed contribution.

A cheap error archive is a useful candidate for early adaptation, but it cannot
replace our actual correction-enabled rollouts. For those rollouts, preserve
linked images, native latents, P/U/M, registration support and map versions.
Reconstruct corrupted histories consistently; perturbing feature values while
keeping pristine geometry does not exercise the full deployment failure.
Keep archives scene-disjoint from evaluation and refresh against current weights.

The decisive reconstruction test remains causal: given conflicting new evidence,
can the updater repair the responsible pose or surface without damaging supported
old views? Compare against both generic refinement and a generator trained to
tolerate the same errors. Evaluate subsequent video as well as withheld-view
reconstruction. This is an unverified hypothesis, not an established novelty or
performance claim. No implementation or training was launched by this review.

`baselines/memorizon` is pinned locally to
`432a49e01c19be8fbd60eb37427ed07dfcc2db16`.
Its inference rebuilds the context KV cache each chunk, with camera-conditioned
context computed before target denoising. Measured history poses can therefore
be supplied before this rebuild; do not edit labels after the cache is formed.
Bank retrieval, attention masks and context camera transforms must all see the
same measured-pose snapshot. Preserve native latent timestamp conventions.

The existing `measured-gaussian-memory` checkout already has native latent-bank
conditioning, an RGB/ReSplat branch, DA3 measured writes, and future-camera rejoin.
It is a working tree with substantial preexisting edits. It currently rebuilds
Gaussians from sources, rather than maintaining a fused revisable scene. The
observer still registers through the fixed anchor, with overlap checks/retries.
It has no 2Xplat backend or the proposed surface-evidence read/update mechanism.

Its current `MeasuredReplayWriter` rejects corrected requests that differ from
recorded target cameras. Its adapter pilot uses four-step final-output L1, with
command correction disabled. That route does not satisfy the new full training
requirement and must not be described as the recommended final objective.

Fine-tune an undistilled Memorizon teacher for the initial FM adaptation; verify
its actual checkpoint metadata/weights and compatible native recipe. A local
Memorizon-50 preparation is referenced in the October handoff, but that reference
is not a newly verified choice of the latest trained checkpoint. Do not substitute
the earlier stock-Wan 9k teacher. Distill only after the adapted teacher is useful.
Ordinary random-timestep FM is not automatically appropriate for a four-step
distilled checkpoint. Native distributed recipes do not establish 32 GB GPU fit.

## State: three camera meanings, two complementary memories

Maintain separate immutable records of:

- `P`: user-planned camera path, used to measure task tracking error.
- `U`: camera commands actually issued, including correction.
- `M`: independently estimated realized cameras and registration support.

Generated native-bank entries and Gaussian source entries use `M`, not `U`.
Recorded training images also receive causally estimated memory labels to match
deployment; reliable recorded poses remain supervision/evaluation references.
Missing measurements are missing labels, not permission to substitute commands.
Fit local-to-global scale/registration from shared observed geometry, never by
matching new measured motion to the commanded trajectory.

Persist a bounded appearance bank plus local Gaussian submaps with versioned
local-to-global transforms. Local coordinates allow rigid pose-graph corrections
without pretending that appearance coefficients are invariant to a world-axis
rotation. Refine shape within local submaps when evidence supports it. Retain
source identities and source-patch associations so geometry can be rebuilt or
revised without losing the original observations. Do not freeze every early
Gaussian forever, and do not blindly append duplicates on every generated frame.

Each surface neighborhood retains Gaussian geometry, source associations, source
view directions, observation-origin metadata, agreement/support summaries, and
features from a bounded selection of source observations. A learned source-feature
projection is preferable initially to training a complete 2Xplat replacement.
The native latent bank remains available as an appearance fallback.

## How to reconstruct better

1. **Propose locally.** Frozen 2Xplat reconstructs the new completed images with
   recent overlapping accepted keyframes. Bootstrap without reliable parallax
   leaves the new spatial branch unavailable and uses native image conditioning;
   duplicated identical poses must not manufacture a baseline or scale.
2. **Register and inspect.** Shared image/depth/feature evidence establishes the
   map connection. Include older keyframes on verified revisits. The initial
   anchor defines global coordinates, not a mandatory visual match at every call.
   Apply explicit local/anchor/native unit conversions; 2Xplat's normalized scale
   is not automatically Memorizon's metric/4 camera convention.
3. **Separate plausible causes.** Compare consistency of tracks, depth, source
   features and renders across views. A coherent reprojection shift can support
   a pose update; a stable pose with local cross-view disagreement can support a
   shape update; isolated appearance conflict can support withholding that write.
   These are evidence patterns, not guaranteed identifiable classes. Ambiguous
   cases preserve uncertainty or reject the update.
4. **Update selectively.** Refine accepted poses and supported Gaussian parameters,
   merge duplicates, and add primitives for uncovered surfaces. Use a bounded
   residual updater with a pose branch and a geometry/appearance branch. Train the
   update gates; keep a robust alternating pose/map optimizer as a simpler baseline.
   Protect established regions through reconstruction of previously accepted
   views, not by forbidding all updates. Appearance features may change without
   forcing geometry to deform.
5. **Validate before commit.** Check the update on past support views not used to
   fit it, where sufficient views exist. A lower fit loss on the same views is
   insufficient. Reject degraded old-view support or unobservable updates. A
   map revision invalidates dependent camera features and rendered caches.

Generated observations are correlated: many frames copying the same memory do
not provide that many independent confirmations. Track common source lineage and
view diversity; do not equate raw observation count or high opacity with truth.
In newly invented regions, the objective is coherent plausible completion, not
recovery of an unknowable hidden real scene. Scene-specific online optimization
updates memory state, not necessarily the pretrained reconstruction weights.

## How reconstruction helps generation beyond an RGB render

Use two reads with distinct jobs:

- **Structural read:** target-view depth, visibility/coverage, geometric support,
  and optionally RGB. It tells Memorizon what is mapped and where it projects.
- **Appearance read:** a target query identifies likely visible surfaces, then
  retrieves the associated original source patch/latent features. A small
  cross-attention adapter combines these with target tokens using relative camera,
  visibility, correspondence quality and cross-view agreement. Keep multiple
  supported source alternatives until attention, rather than forcing one blended
  RGB texture through the reconstructor.

Initialize new residual outputs at zero and retain native camera/retrieval paths.
Where geometry is uncertain, broaden or reduce its correspondence influence; do
not hard-force a false surface. Where support is absent, the video prior completes
the view. Reading surface features is related to WorldStereo, Lyra, Mirage,
MosaicMem and GLIMPSE; it is the useful mechanism, not a stand-alone novelty claim.

Memorizon's 3D VAE latents summarize temporal/spatial neighborhoods. Assigning all
48 channels to a single Gaussian/pixel assumes an equivariance the codec does not
guarantee. Start with retained native tokens plus geometric associations and a
learned projection. Direct learned feature lifting/rendering is a separate
comparison to the source-token read, not a requirement for the first model.

## Training that includes measured poses and command correction

**Stage 1: reconstruction/update learning.** On scene-disjoint training videos,
construct causal partial maps from past views and predict updates from the next
observed chunk. Fit held-out view RGB/features, geometric consistency and reliable
pose supervision where available. Older-view losses penalize forgetting. Masks
come from independent reference visibility for paired scoring, so predicting
low confidence cannot erase the training penalty. Missing real depth is not
silently replaced with ground-truth-labelled 2Xplat predictions.

Create four controlled examples: pose perturbation with unchanged scene; geometry
perturbation with fixed cameras; conflicting appearance in an incoming observation;
and genuine newly exposed content. Include combined/ambiguous errors and actual
rollout errors. Isolated synthetic perturbations supply branch supervision, but
the model must not learn only their artificial signatures. Do not expect RGB
residual magnitude alone to distinguish these causes.

**Stage 2: Memorizon adaptation on our data.** Fine-tune the undistilled teacher
with its native packed chunk-causal noise/loss conventions, plus the new memory
reads. Memory poses are causally measured; target-camera labels describe the
actual supervised target views. Train cases where old memory is informative,
redundant, or corrupted. Keep clean examples, explicit empty-memory startup,
wrong-memory controls and moving camera forks. Train adapters first, then qualified
LoRA/camera updates if needed; match trainable parameters in controls.

**Stage 3: actual closed-loop training.** Run the current model with measured
history writes AND camera correction enabled. At every boundary record `P,U,M`,
accepted/rejected map revisions and the source snapshot used for rendering. Train
on the states this full loop actually produces, including correction recovery and
registration failure. This stage is a required part of the requested method,
not an inference-only patch or a permanently disabled ablation.

There is a target-availability constraint: once correction issues `U` different
from a recording's cameras, those recorded pixels are not automatically valid
targets. Use either (a) genuine camera-compatible multiview/synthetic recovery
examples, or (b) a separately qualified teacher/distillation objective conditioned
on the actual issued cameras and compatible scene history. In option (b), the
teacher's score/targets are model supervision, not real ground truth. Do not copy
recorded target pixels and relabel them to `U`, and do not equate old final-step L1
replay with self-forcing/DMD. Reconstructed renders alone are also an insufficient
teacher for disoccluded or disputed surfaces.

**Stage 4: downstream refinement signal, if needed.** After both parts work, let
the generation loss train the source-feature reader and supported update features.
Initially detach measured poses and reconstruction across memory boundaries.
If later we connect generation loss to scene geometry through a differentiable
renderer, retain separate reconstruction/camera losses: otherwise geometry may
deform to become an easier conditioning signal. State which graph is detached;
frozen 2Xplat inference does not provide a gradient to earlier generated chunks.

Use compact ablations for gradient paths rather than naming every stage a separate
method. Report inference calls, jointly supervised target frames, retained history,
and gradient horizon separately. A distant sampled return can test memory after
history eviction without retaining the full generation graph.

## Candidate contribution and decisive experiments

The candidate claim is: **a scene memory revised using measured cameras and
multi-view disagreement, and read through its original source evidence, improves
both reconstruction and closed-loop generation when generated cameras/content
are imperfect.** Do not claim to be the first Gaussian memory, latent memory,
uncertainty gate, joint pose-map optimizer, or camera-correction method.

The specific test is whether learned cause-sensitive revision adds value over
generic refinement and uncertainty-weighted reads. If it does not, retain the
simpler method and describe the actual measured contribution; do not invent a
novelty label. Concurrent work should be cited and compared where feasible.

Use a shared measured-write/correction configuration to compare:

| Arm | What it isolates |
| --- | --- |
| Fine-tuned native Memorizon | Benefit beyond adapting the base to our data |
| + 2Xplat RGB target renders | Straightforward reconstruction-conditioning baseline |
| + geometry-addressed source features | Value beyond flattened RGB evidence |
| + generic pose/map refinement | Value of reconstruction improvement itself |
| + supervised selective pose/map/appearance revision | Incremental value of the proposed update policy |

For the strongest arm, additionally remove measured labels and remove correction
separately. A matched generic-refinement arm uses the same observations, read
budget and update compute. Include latent-render conditioning as the closest
representation comparison where feasible; external L3G-World/WorldStereo results
are informative but cannot substitute for these same-backbone tests.

The clearest diagnostic holds residual magnitude similar while changing its
cause: a wrong camera should not spawn a duplicate wall; a wrong wall should not
be hidden by rotating the camera; a bad generated texture should not overwrite
well-supported history. Evaluate both the update and later generated views.

Use departure beyond anchor visibility, motion/rotation, revisits after context
eviction, weak texture, occlusion and stationary controls. Report scene/seed
averages, paired effects, image sharpness/quality, camera errors, coverage,
reconstruction on withheld views, and resource cost. Preserve pose failures and
collapse in denominators. Evaluate with independent evidence where possible;
2Xplat must not be the sole scorer of its own corrected scene. Some scene failures
are acceptable; shared unit, timestamp, cache or target-label bugs are not.

## Immediate implementation order

1. Pin the intended undistilled Memorizon weights and source; adapt 2Xplat outputs
   to the existing measured-memory interface with camera/grid/unit provenance.
2. Establish identical native behavior when the added branches are zero and test
   measured labels before per-chunk context-cache reconstruction. Integrate local
   overlapping-view registration rather than relying solely on frame zero.
3. Implement the structural/source-feature reads; profile one training backward
   pass before specifying GPU-hours or scaling the run. Existing 32 GB claims in
   prototypes are not full-size qualification.
4. Train the teacher/read path and the small updater on controlled causal data,
   then build camera-compatible closed-loop training targets. Distill afterward.

No running queue, checkpoint, protected model, or existing training implementation
was changed by this review. The design supersedes the earlier stock-Wan teacher
initialization proposal, while retaining its coordinate and causality checks.
