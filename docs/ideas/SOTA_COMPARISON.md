# GaussVerse versus current static-scene and long-rollout world models

Status: every external method in this table, plus thesis-critical Wonder, was rechecked
against a primary arXiv or official CVF paper on **2026-08-04** and recorded in
`THESIS_EXTERNAL_SOURCES.json`. Peer-reviewed venue papers and recent preprints are
labelled separately. Reported numbers are not directly comparable across datasets; this
document compares mechanisms, task coverage, and evidence rather than manufacturing a
single leaderboard.

The newly released Lyra 2.0 training implementation has also been inspected directly.
See `LYRA2_TRAINING_CODE_AUDIT.md` for the full-gradient constructor override,
coverage-based memory retrieval, Q/K correspondence injection, and self-augmentation
mixture.

## Bottom line

GaussVerse is not currently SOTA as a long-horizon world model. Its selected factorial-C
checkpoint is a full-5B single-GPU adaptation with exact camera rays, temporal RoPE,
fused target-warp evidence and two-sided correspondence. It has a matched 16-scene H20
RGB-feedback result, but no successful generated-history recovery stage, no selected
persistent-memory result, and 30 denoising steps per generated frame. Native retrieval,
geometric attention, latent feedback, chunks, cache-valid source K/V and stronger camera
branches are executable primitives/harnesses; none is a measured win. A matched
persistent coordinate-lattice experiment is training and is not yet part of the method.

The obvious integrated architecture—bounded recent frames, geometry-retrieved old frames,
correspondence-aware attention, and self-generated-history training—is no longer novel.
Lyra 2.0 and AlayaWorld combine most of those ingredients, while peer-reviewed WorldStereo
and CAMEO cover geometric memory and correspondence-guided attention.

The strongest open GaussVerse direction is therefore narrower: **path-independent static
scene generation with query-dependent authority for provisional generated observations**.
Even this is only a candidate: recent loop-closure and geometry-aware implicit-memory work
makes the neighborhood crowded.

## Mechanism comparison

| Method | Task and scale | Recent/temporal state | Spatial state | Rollout-error training | Consequence for GaussVerse |
|---|---|---|---|---|---|
| **GaussVerse selected** | Static DL3DV trajectories; one-frame AR; complete Wan2.2 5B DiT fine-tuned | Three ordered frames with temporal RoPE; RGB feedback | Latest-three fused target warp + two-sided correspondence; no persistent archive | Matched full-backbone H0/H1/H2/H3 replay curriculum failed against clean control | Controlled n=16 H20 baseline with improved fidelity/consistency; not minute-scale or persistent-memory SOTA |
| [Lyra 2.0](https://arxiv.org/abs/2604.13036) *(preprint)* | Long camera exploration and revisits; generated video lifted to 3D | FramePack compressed history | Per-frame geometry retrieves native RGB frames and establishes dense correspondence; geometry routes information | One-step self-augmented degraded histories | Closest task match; retrieval plus correspondence plus self-augmentation cannot be claimed as new |
| [WorldStereo](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_WorldStereo_Bridging_Camera-Guided_Video_Generation_and_Scene_Reconstruction_via_3D_CVPR_2026_paper.html) *(CVPR 2026)* | Camera-guided video and 3D reconstruction | Video diffusion backbone | Global point-memory control plus spatial-stereo memory restricting attention by 3D correspondence | Not its primary contribution | Direct geometric memory/attention is already peer-reviewed prior art |
| [AlayaWorld](https://arxiv.org/abs/2607.18367) *(preprint)* | 15B, 540/720p, short latent chunks, 24 fps | Sink frame, six-frame compressed history, nearby frame | Geometry-aligned spatial memory in a bounded context | Corrupted histories and prediction residuals from own rollouts; later 4-step distillation | The full recent+compressed+spatial+rollout-recovery stack is occupied |
| [SANA-WM](https://arxiv.org/abs/2605.15178) *(preprint)* | 2.6B, 720p, one minute; ~213k clips; large multi-GPU training | Frame-wise Gated DeltaNet with every fourth block replaced by softmax attention | Recurrent state rather than an explicit revisit archive | Chunk-causal conversion, self-forcing/distillation, long-video refinement | Learned recurrent decay is an escalation, not a fair first experiment on one GPU |
| [ABot-World-0](https://arxiv.org/abs/2607.19191) *(preprint)* | Dynamic action-conditioned worlds; few-step 720p desktop deployment | Bounded causal/KV context | Read-only character reference memory | Teacher forcing, ODE distillation, then LongForcing on long student self-rollouts | Supports the staged training principle; its dynamic/action problem and compute are much broader |
| [RELIC](https://arxiv.org/abs/2512.04040) *(preprint)* | 14B, 480×832, reported 16 FPS on 4×H100, up to 20 s | Camera/action-encoded compressed historical latents in KV state | Implicit camera-aware retrieval | Bidirectional 20-second teacher plus block-replayed self-forcing over long student rollouts | Cacheable compressed spatial history and long self-forcing are already occupied |
| [MemLearner](https://arxiv.org/abs/2606.31734) *(preprint)* | Long video with occlusion/dynamics | Learned query tokens bridge context and prediction | Adaptive learned context query rather than fixed pose rule | Multi-dataset long-video training | “Learn which memory to read” is occupied; authority must be more specific than a learned router |
| [HorizonDrive](https://arxiv.org/abs/2605.11596) *(preprint)* | Minute-scale driving rollout | Fixed bounded temporal window | No static-scene episodic archive | Scheduled Rollout Recovery, then teacher-rollout DMD | Best direct precedent for a matched, ramped recovery curriculum before distillation |
| [GEN3C](https://openaccess.thecvf.com/content/CVPR2025/html/Ren_GEN3C_3D-Informed_World-Consistent_Video_Generation_with_Precise_Camera_Control_CVPR_2025_paper.html) *(CVPR 2025)* | Camera-controlled video and sparse-view NVS | Video context | Separate per-view 3D caches rendered at the target pose | Spatial persistence, not a complete anti-drift objective | Closest precedent to GaussVerse's old warped evidence path |
| [Geometry-as-context](https://openaccess.thecvf.com/content/CVPR2026/html/Hu_Geometry-as-context_Modulating_Explicit_3D_in_Scene-consistent_Video_Generation_to_Geometry_CVPR_2026_paper.html) *(CVPR 2026)* | Autoregressive static-scene video | AR context | Interleaved geometry/RGB generation and reconstruction | Iterative geometry/RGB loop | Joint geometry prediction is occupied and can accumulate its own inference error |
| [CAMEO](https://openaccess.thecvf.com/content/CVPR2026/html/Kwon_Correspondence-Attention_Alignment_for_Multi-View_Diffusion_Models_CVPR_2026_paper.html) *(CVPR 2026)* | Multi-view diffusion/NVS | Not a long-rollout method | Ground-truth correspondence supervises an attention layer | No rollout recovery | Attention-correspondence supervision alone is not novel |
| [UCPE](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Unified_Camera_Positional_Encoding_for_Controlled_Video_Generation_CVPR_2026_paper.html) *(CVPR 2026)* | Camera-controlled video with diverse intrinsics/lenses | Video DiT | None | No long-horizon recovery | Relative-ray camera encoding with <1% adapters is a required camera baseline |
| [GIM-World](https://arxiv.org/abs/2606.02436) *(preprint)* | Long-horizon video world models | Fixed-size implicit memory tokens | Camera-queryable geometry head and information-guided bounded pruning | Not its main contribution | Bounded geometry-aware compressed memory is occupied |
| [Closing the Loop](https://arxiv.org/abs/2607.21848) *(preprint)* | AR generative rendering on loop-closure paths | Bounded KV cache plus retrieved historical chunks | Pose-matched latent retrieval and depth/camera correspondence bias | Training-free | Loop-closure retrieval plus geometric attention is explicitly occupied |
| [Seva](https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_Stable_Virtual_Camera_Generative_View_Synthesis_with_Diffusion_Models_ICCV_2025_paper.html) *(ICCV 2025)* | Flexible NVS and virtual-camera video | Joint diffusion view generation | No growing explicit memory | Optimized multi-view training/sampling | A strong camera/view-synthesis and seamless-loop baseline, even without explicit 3D memory |
| [MultiDiff](https://openaccess.thecvf.com/content/CVPR2024/html/Muller_MultiDiff_Consistent_Novel_View_Synthesis_from_a_Single_Image_CVPR_2024_paper.html) *(CVPR 2024)* | Long-trajectory NVS from one image | Joint multi-frame diffusion with structured noise | Depth-warp conditioning | Avoids one-frame AR exposure by joint prediction | Required precedent for GaussVerse's 4/8-frame chunk baseline |
| [Prisma-World](https://arxiv.org/abs/2606.09507) *(preprint)* | Camera-controlled multi-agent worlds | Joint full-attention denoising with multi-agent RoPE | Relative camera geometry couples overlapping agents | Overlap-decaying curriculum | Different paths/views of one world are already jointly consistency-trained; path consistency alone is not enough novelty |

## What GaussVerse has actually measured

The definitive full-backbone 10k A/B/C/D factorial uses 16 frozen scenes and selects C,
clean histories plus fixed fused warp/two-sided correspondence. Relative to matched
memory-off A, C improves teacher mean by `+0.805 dB [0.595,1.020]`, autoregressive mean
by `+0.467 dB [0.249,0.712]`, H1/H5/H10 by `+0.685/+0.378/+0.532 dB`, revisit fidelity
by `+0.672 dB`, and same-target cross-path PSNR by `+2.025 dB`. H20 remains only
`+0.173 dB [-0.144,0.472]`, and drift slopes do not improve, so the memory branch is a
fidelity/consistency component rather than an anti-drift result.

The matched H0/H1/H2/H3 replay curriculum is decisively negative: B−A autoregressive
mean is `-2.330 dB` and H20 is `-2.984 dB`. This closes the tested teacher formulations,
not self-forcing/DMD or all possible generated-history training.

The matched Q/K experiments support a fidelity gain over frozen controls on the old
single-target evaluation, but pose significance weakened across controls and subsets.
Camera-control superiority is therefore not a current headline.

## Where GaussVerse is stronger

- Exact 6-DoF target-relative full-resolution Plucker rays are already integrated.
- The task is deliberately static, allowing path-independence and same-pose loop tests
  that dynamic world models cannot enforce without ambiguity.
- Training and evaluation fit on one RTX 5090 and preserve a strong pretrained prior.
- The repository has unusually strict matched controls, immutable checkpoints,
  scene-bootstrap intervals, null-floor-corrected oracles, and a ground-truth-only pose
  validity gate.
- Negative results already rule out several seductive but non-causal memory proxies.

## Where it is behind

- Sixteen frozen trajectory scenes rather than a broad public benchmark.
- Twenty generated frames in the definitive factorial evaluation. The registered lattice
  chain reports H50 only through `trajectory_lattice_thesis_summary.json`; no H50 claim is
  made before that artifact exists.
- One-frame measured generation; 4/8-frame chunk code has no trained checkpoint.
- No successful exposure to realistic multi-step self-history.
- No compressed temporal state, sink, selected persistent retrieval, or minute-scale loop
  closure. Latest-three target-warp evidence does improve the measured revisit protocol.
- Thirty denoising steps in measured runs; direct latent feedback is implemented but unmeasured.
- Native context latents are cached exactly; the separate immutable-source K/V primitive
  is test-validated but not yet integrated or trained in the 5B renderer.
- UCPE-like ray-local attention and WorldStereo-like pairwise geometry are implemented,
  but no trained or powered baseline result exists; CAMEO-like loss is a primitive, not
  yet wired to selected-layer attention maps.

SANA-WM's camera ablation is especially relevant to budget selection: it reports the
attention-level PRoPE/UCPE comparison after `10k` fine-tuning steps, and its strongest
camera result combines UCPE with Pluecker mixing. This supports treating GaussVerse's old
3--4k saturation as objective-specific evidence, not as a universal full-backbone budget.

## Novelty collision analysis

| Candidate claim | Verdict in 2026 |
|---|---|
| Temporal RoPE + Q/K full adaptation | Useful baseline; not novel |
| Geometry-based keyframe retrieval | Occupied by Lyra 2.0, AlayaWorld, GEN3C |
| Point/epipolar attention bias | Occupied neighborhood: WorldStereo, Lyra 2.0, CAMEO |
| Pose-matched latent retrieval for loop closure | Occupied by Closing the Loop |
| Geometry-aware bounded implicit memory | Occupied by GIM-World |
| Learned adaptive memory query | Occupied by MemLearner |
| Camera-aware latent K/V memory | Occupied by RELIC |
| Joint multi-path/cross-agent consistency | Occupied neighborhood: Prisma-World and collaborative multi-video diffusion |
| Self-generated-history recovery | Occupied by Lyra 2.0, HorizonDrive, AlayaWorld, ABot |
| Sink + recent + compressed + spatial memory | Occupied by AlayaWorld |
| Hard pose-verified memory writes | Earlier GaussVerse gate failed; not currently viable |
| **Query-dependent authority for generated evidence** | Only plausibly differentiated when authority is explicitly calibrated from provenance, lineage and realized geometry; generic adaptive retrieval is occupied |
| **Lineage-independent evidence quorum** | Correlated descendants of one generated error cannot count as multiple votes. Executable as an opt-in retrieval control, but novelty and utility are unproven |
| **Two independent histories / same target / same-noise invariance** | Narrower than Prisma's joint agents, but heavily adjacent; needs a formal distinction and decisive causal result |

The proposed contribution should not be presented as “combining SANA-WM, Lyra, and
ABot.” A paper needs a mechanism those systems do not already supply and evidence that it
is necessary. The current candidate is:

> Under two independently corrupted causal histories, the same static target and the same
> diffusion noise should produce one supported state. Generated observations remain
> provisional, and their read authority is explicitly calibrated per target from realized
> geometry, cycle consistency, provenance, and lineage.

This combined statement is stronger than either part alone, but it is **not yet safe to
call CVPR-level novelty**. Prisma-World already couples overlapping trajectories during
joint denoising; RELIC and MemLearner already learn camera-aware/adaptive memory. The
remaining possible distinction is causal invariance across separately accumulated error
histories plus calibrated evidence authority, demonstrated on revisit/out-and-back tests.

## Required comparison protocol

External headline numbers use incompatible datasets, resolutions, actions, and metrics, so
GaussVerse needs mechanism-matched internal baselines:

1. recent-only temporal Q/K;
2. recent + time-only old frames;
3. recent + geometry-retrieved observed frames with free attention;
4. a GEN3C/fused-warp baseline;
5. a WorldStereo-like fixed correspondence mask;
6. a CAMEO-like attention-supervision baseline;
7. read-time authority and path-independence, each separately ablated;
8. correct, shuffled, and corrupted geometry controls.

Report trajectory AUC, horizon 20/50, teacher retention, drift, revisit fidelity,
same-camera path disagreement, pose on the frozen valid subset, latency, VRAM, and bounded
memory growth. No direct SOTA claim is defensible until the same model and benchmark beat
these internal baselines at `n >= 16`, followed by the 48-scene primary split.

## Immediate recommendation

Do not train another 12k clean continuation and do not build recurrent attention yet.
Finish the aligned replay cache, run both 50-step full-backbone mechanical smokes, then
advance the matched clean-versus-curriculum arms through the 250/500/750/1000 gates. The
configured 10k steps are a ceiling, not an entitlement. Only a recovery model that passes
teacher retention and improves rollout drift should become the parent for observed-token
retrieval and the later path-independence study.
