# Related methods

Status: every external method discussed below, plus thesis-critical Wonder, was rechecked
against a primary arXiv or official CVF paper on **2026-08-04** and recorded in
`THESIS_EXTERNAL_SOURCES.json`. This is the concise comparison used
for GaussVerse design decisions.
For the measured temporal Q/K result and historical self-history experiments, see
[TEMPORAL_QK_ROLLOUT_REPORT.md](TEMPORAL_QK_ROLLOUT_REPORT.md).
For the higher-level SOTA positioning and research verdict, see
[SOTA_COMPARISON.md](SOTA_COMPARISON.md).

## Lyra 2.0

Primary source: [paper](https://arxiv.org/abs/2604.13036) and
[project page](https://nv-tlabs.github.io/Project-Lyra/).

Lyra separates two long-horizon failures:

- **Spatial forgetting:** each generated frame retains independent depth and camera
  geometry. A coarse point cloud is used to retrieve past frames with high target
  visibility. The selected RGB frames become spatial attention slots. Full-resolution
  depth forward-warps canonical pixel coordinates and depth to the target; the resulting
  correspondence embedding is added at every self-attention layer.
- **Temporal drift:** a clean history latent is noised at `t ~ U(0, 0.5)`, reconstructed
  with one model denoising step, detached, and used as conditioning while the next clean
  chunk remains the flow-matching target. This exposes the model to its own characteristic
  errors for one extra DiT forward.

Lyra explicitly does **not** warp RGB into the target. It argues that RGB warps carry
holes, stretching, and depth-boundary artifacts that the generator may copy. Geometry is
used for routing, while appearance comes from the retrieved source-frame tokens and the
generative prior. Earlier versions of this document incorrectly described Lyra as
channel-concatenating warped RGB and injecting correspondence only into Q/K; those claims
have been removed.

Transfer to GaussVerse: the one-step self-augmented history is the closest bounded
anti-drift experiment. Geometry retrieval is relevant later for location revisits, not as
a substitute for rollout recovery.

## SANA-WM

Primary source: [paper](https://arxiv.org/abs/2605.15178),
[project page](https://nvlabs.github.io/Sana/WM/), and
[repository](https://github.com/NVlabs/Sana).

SANA-WM is a 2.6B model natively trained on one-minute, 720p clips. Its principal
long-context change is architectural: frame-wise Gated DeltaNet recurrence uses learned
decay and delta correction, while every fourth block is replaced by standard softmax
attention for exact long-range recall. It combines this with dual-rate 6-DoF camera
conditioning. Earlier text converted “every fourth” into an unsupported absolute 15/20
block count; the paper does not state the total in that method passage.

The bidirectional minute model is subsequently fine-tuned into a chunk-causal model and
self-forcing-distilled. At deployment, local temporal windows and attention-sink tokens
bound softmax memory. A separately adapted long-video refiner improves the completed
minute-long latent sequence.

Transfer to GaussVerse: learned decay is a credible architectural escalation if realistic
history replay fails. It is not a fair first ablation: SANA-WM also changes the backbone,
training horizon, data scale, distillation, and refinement pipeline.

## ABot-World-0

Primary source: [paper](https://arxiv.org/abs/2607.19191).

ABot first full-parameter fine-tunes a bidirectional Wan2.2 teacher on long
action-conditioned videos. It then builds a causal model in three stages:

1. teacher forcing with clean history and a causal mask;
2. ODE distillation from the multi-step causal model to a few-step student;
3. LongForcing, which runs the student on its own long histories and aligns those rollout
   distributions with an extended-horizon bidirectional teacher using distribution
   matching.

Static character identity uses a separate read-only memory: canonical reference tokens
have negative temporal positions, video tokens may attend to them, and the reference
tokens do not absorb generated-video state. This memory stabilizes appearance; it is not
the method's anti-drift mechanism.

Transfer to GaussVerse: clean-history causal fine-tuning is only the initialization. The
long-rollout stage addresses the real deployment distribution. Full LongForcing is much
more expensive than detached two-step replay and should not be the first implementation.

## HorizonDrive

Primary source: [paper](https://arxiv.org/abs/2605.11596).

HorizonDrive cleanly separates a rollout-capable teacher from a fast student. Scheduled
Rollout Recovery first rolls the base teacher autoregressively, samples its degraded
history, and retains the clean ground-truth future as supervision. Local history-to-GT
blending smooths the transition boundary; a global schedule moves from severe late
rollout corruption toward earlier, more shared errors. Teacher Rollout DMD then transfers
fixed-window long rollouts to a short-chunk student.

Transfer to GaussVerse: this supports a staged design—first teach the existing model to
recover from its own errors, then consider distillation or inference acceleration.

## GEN3C

Primary source: [paper](https://arxiv.org/abs/2503.03751) and
[repository](https://github.com/nv-tlabs/GEN3C).

GEN3C maintains a 3D cache made from the initial and previously generated frames and
renders that cache at the requested camera as conditioning for video generation. This
reduces the burden on temporal attention to remember old geometry and helps synthesize
only genuinely unobserved areas.

Transfer to GaussVerse: this is the closest external analogue of the tested warped-memory
branch. It targets spatial persistence, especially revisits. It does not by itself close
the clean-history versus generated-history training gap.

## WorldStereo

Primary source: [CVPR 2026 paper](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_WorldStereo_Bridging_Camera-Guided_Video_Generation_and_Scene_Reconstruction_via_3D_CVPR_2026_paper.html).

WorldStereo combines a global geometric memory, which renders coarse structural control
from incrementally updated point clouds, with a spatial-stereo memory that constrains the
video model's attention receptive field using 3D correspondence. It is the closest
peer-reviewed collision with a proposed GaussVerse point/epipolar attention bias.

Transfer to GaussVerse: direct geometric attention remains a necessary baseline, but it is
not a novelty claim. A contribution must add something causally distinct, such as
query-dependent evidence authority or a static-scene path-independence objective.

## AlayaWorld

Primary source: [technical report](https://arxiv.org/abs/2607.18367) and
[repository](https://github.com/AlayaLab/AlayaWorld).

AlayaWorld uses a 15B video DiT with a bounded context containing a persistent sink,
compressed six-frame temporal history, a nearby frame, and geometry-aligned spatial
memory. It trains on corrupted histories and residuals collected from its own rollouts,
then combines distribution matching, self-forcing++, and consistency distillation to
reach four-step chunk generation.

Transfer to GaussVerse: sink + recent + compressed temporal + spatial memory is already a
complete occupied architecture. GaussVerse should exploit its narrower static-scene
contract instead of reproducing this system at smaller scale.

## CAMEO and UCPE

Primary sources: [CAMEO, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Kwon_Correspondence-Attention_Alignment_for_Multi-View_Diffusion_Models_CVPR_2026_paper.html)
and [UCPE, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Unified_Camera_Positional_Encoding_for_Controlled_Video_Generation_CVPR_2026_paper.html).

CAMEO directly supervises attention using geometric correspondences and reports that one
supervised layer is sufficient to improve multi-view convergence and quality. UCPE uses a
relative-ray encoding covering pose, intrinsics, and lens distortion, plus absolute pitch
and roll, through a lightweight spatial attention adapter.

Transfer to GaussVerse: CAMEO is the attention-supervision baseline for any learned
geometric routing claim. UCPE is the camera-conditioning baseline; unfreezing 0.604B Q/K
parameters should not be positioned as the only way to obtain camera control without
comparing against a modern small relative-ray adapter.

## GIM-World and Closing the Loop

Primary sources: [GIM-World](https://arxiv.org/abs/2606.02436) and
[Closing the Loop](https://arxiv.org/abs/2607.21848).

GIM-World compresses variable-length history into fixed-size memory tokens, trains a
camera-queryable geometry head using a frozen geometry teacher, and prunes history by
information value to keep cost bounded. Closing the Loop is training-free: temporal
correspondence retrieves pose-matched historical latent chunks into a bounded KV cache,
while camera/depth correspondence biases spatial attention on revisits.

Transfer to GaussVerse: neither bounded geometry memory nor pose-matched loop retrieval is
new. The proposed read-time authority must show value beyond ordinary correspondence
retrieval, and the paired two-path objective needs a broader novelty audit before it can be
positioned as a contribution.

## RELIC, MemLearner, MultiDiff, and Prisma-World

Primary sources: [RELIC](https://arxiv.org/abs/2512.04040),
[MemLearner](https://arxiv.org/abs/2606.31734),
[MultiDiff, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Muller_MultiDiff_Consistent_Novel_View_Synthesis_from_a_Single_Image_CVPR_2024_paper.html),
and [Prisma-World](https://arxiv.org/abs/2606.09507).

RELIC stores highly compressed historical latent tokens with relative actions and absolute
camera poses inside a bounded KV state, then self-forces a causal student over long
rollouts. MemLearner replaces hand-designed frame retrieval with learned query tokens that
bridge old context and predicted tokens. MultiDiff jointly predicts a camera trajectory
with structured noise rather than accepting one-frame autoregressive exposure. Prisma-
World jointly denoises multiple overlapping agent trajectories, using multi-agent RoPE and
relative-camera attention to enforce one shared world across views.

Transfer to GaussVerse: cacheable camera-aware history, learned memory queries, joint
chunks, and generic cross-trajectory consistency are all occupied. The remaining possible
research distinction is much narrower: two histories accumulate errors independently and
cannot communicate, yet the same target and same diffusion noise must yield the same
supported static state; provisional-source authority is calibrated from geometry,
provenance and lineage rather than serving as a generic learned retrieval score.

## Current design consequence

Static-scene camera trajectories have two separable problems:

1. **local rollout stability:** recent overlapping frames are repeatedly replaced by model
   outputs; solve this with ordered temporal conditioning and self-history recovery;
2. **spatial persistence on revisits:** an old location has left the recent window; solve
   this with selective geometry-grounded retrieval only if a revisit benchmark demands it.

The definitive full-backbone factorial establishes that fixed fused warp/two-sided
evidence improves n=16 teacher, rollout, revisit and cross-path fidelity, but its H20
interval crosses zero and its degradation slopes worsen. It is therefore retained as a
fidelity/consistency component, not an anti-drift mechanism. Both the matched H0/H1/H2/H3
replay curriculum and earlier one-step self-history formulations failed; this closes those
teacher-training recipes, not DMD/self-forcing generally.

The live causal intervention is now the matched L0/L1 persistent coordinate-field test
from the selected factorial-C parent. It tests one Wonder-aligned synthetic camera signal
under complete co-adaptation; it is not persistent appearance memory. No additional
reader, router, or recovery branch should be opened before its registered result. Sparse
native latent/KV state remains the empirically motivated future architecture because it
would avoid repeated depth estimation from degraded generated RGB.
