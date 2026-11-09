# IDEAS v3 (2026-10-10): novel additions grounded in our own measurements
Status: proposals, not decisions. Each has a kill test on an existing checkpoint/family; none needs new data beyond planned rounds.

**Superseding research review, 2026-10-10:** Both training queues are stopped at
the user's request to settle contributions before setup. Data preparation continues.
See [REDESIGN §12](REDESIGN.md#12-2026-10-10-katkı-kararı-ve-tek-rtx-5090-eğitim-hedefi).
The review closure and serial preparation schedule are in REDESIGN §14; the
[proposed contract](../../training_runs/base_restart_20261008/dataset_v3/review_20261010/contribution_contract.proposed.json)
specifies post-projection camera masking, memory substitution controls, calibrated
differentiable camera supervision and shared render dropout. It is not executable
and does not assert that these paths exist. Both legacy training configs are
explicitly disabled; their original copies and prototype code are preserved.
The six-action implementation below is a prototype, not the accepted final recipe.
The selected research problem is retaining useful memory appearance when its
estimated geometric placement is harmful. The user also requested a camera-specific
contribution: the selected hypothesis independently intervenes on requested camera
and erroneous render geometry, teaching correct command response and suppressing
camera drift induced by that erroneous geometry. Generic camera forks already have
close precedents in OnlineWM and ActionSplice; pose rewards in CamVerse/CamPilot.
The additional systems target is measured training accessibility on one RTX 5090,
not an inference-speed claim. Learned initialization,
long memory, utility supervision, and reliability filtering are not individually new.
There is no demonstrated novelty, quality gain, or full-run training speedup yet.
The original headline's assertion of no additional derived data/cost is superseded:
physical geometry interventions and new-student rollout validation require work.

**2026-10-10 joint round-0 decision:** User chose compatible core contributions,
trained together from published stock from update one; not all F–L mechanisms.
The baseline-only training service was stopped before training. The implemented
candidate in `training_runs/base_restart_20261008/joint_evidence.py` jointly learns
(1) a causal, chunk-level policy over render-read off/on × endpoint strength
0/.25/.5, and (2) an explicitly endpoint-conditioned flow model. Native
anchor/history/archive identities stay fixed across policy-label interventions.
The policy uses six matched four-step solver outputs and recorded target latent
reconstruction error, on one scene per optimizer update. This is short-solver
latent utility, NOT calibrated correctness, camera accuracy, long-rollout reward,
per-patch utility, or on-policy generated-history training. A 30-step comparison
on three fixed dev scenes measures transfer every 500 updates. Every FM update
includes explored evidence use (0.7), native-only (0.2), and memory-missing (0.1).
Policy and endpoint modules both train from the first update; no 6k foundation
prerequisite. Synthetic latent shift/channel-affine stress is not a wrong-depth
render or a substitute for own-state data. GPU qualification is queued after cache
completion; first-update gradient and exact resume gates precede broad training.

**GeoStream prior-art check (2026-10-10):**
[Paper](https://arxiv.org/html/2606.15162v1),
[project](https://a02983278.github.io/geostream.github.io/).
Published June 13, 2026; Wan2.2-5B, MoGe-2, self-generated geometry and on-policy
DMD already establish that own-state/S3b alone is not our contribution.
§3.1 adds projected/patchified render hidden states to video hidden states; this
is not a changed flow endpoint. Old explicit point geometry is discarded rather
than fused; native historical state is not entirely absent: w=3 past chunks plus
s=1 retained sink chunk. Main §3.1 describes chunk-end refresh, but Appendix B
names Every Frame as the default and Last Frame as an alternative. Do not claim
the implementation always uses only the preceding chunk's last image. No official
implementation link was found on the checked project page, so this is paper-level
verification. Training is 8k teacher + 5k causal warmup + 1k on-policy DMD.
Appendix A aligns camera scale during both data preparation and evaluation; this
does not establish our future-free anchor-only metric scale claim.

Comparison axes against GeoStream: long-term native retrieval, multi-source
ReSplat, native/render path separation, and learned evidence-use/endpoint policy.
These axes are not four independently established novel contributions. Lyra and
Memorizon cover memory precedents; Manifold4D/GeoFlow cover geometric endpoints;
FAR covers predictive-utility learning. The candidate claim needs matched
ablations showing useful old memory can be retained while harmful rendered
geometry is rejected. Current round-0 uses recorded histories and synthetic
stress, not GeoStream's on-policy DMD.

**2026-10-10 review superseding novelty/cost assumptions below:** See [REDESIGN §10](REDESIGN.md#10-2026-10-10-katkı-sınırı-müdahale-hedefi-ve-ayrılmış-deney-bütçesi).
These are unimplemented hypotheses, not verified novel methods. Counterfactual utility labels and paired sampler probes can
require additional derived data and GPU work beyond planned rounds. A: render support is not the boundary of causal influence;
lighting, geometry and occlusion cues can legitimately affect uncovered regions, so the proposed hard invariance is not adopted.
B: disagreement measures sensitivity, not whether evidence is right; beneficial novel evidence can cause large disagreement.
C: the claim that nobody predicts execution error is unverified and must not be used in a paper.
D: historical M5 g is measured command-response gain, not a learned gate magnitude. Render-only removal must retain the native
archive; current memory_missing removes both archive and render and therefore does not isolate this curriculum.
Manifold4D, GeoFlow and StableHand are close priors for initialization; native/render separation alone is also insufficient.

## A. Paired-condition consistency loss (evidence invariance outside support)
Round-0 already computes, per scene group, the same target/noise/time under two conditions (clean vs memory_missing);
round 1 adds E-corrupt. Add L_pair = mean over patches with render coverage m(p)=0 (or r*≈0 in round 1) of
||v(cond_with_evidence) - v(cond_without_or_wrong_evidence)||^2 (stop-grad on one side, alternate).
Teaches: evidence may change the output only where it covers; wrong/absent evidence must not change uncovered regions.
Cost: zero extra forwards (predictions already exist). Not in Lyra (dropout only), Memorizon (test-time bank swap),
Robust Dreamer (deviation corruption). Kill test: E-only family DINO/LPIPS and uncovered-region hallucination vs no L_pair.

## B. Disagreement-gated evidence init (training-free reliability)
At inference run the first k denoising steps with and without render; per-patch disagreement d(p) (normalised) where
render confidence is high = "surprising evidence". sigma_init(p) from d(p), or render weight 1-d(p). No head, no labels.
Builds on: evidence-CFG 1.5 fixed the seam (2026-09-18) — we use the difference as a gate instead of amplifying it.
Role: stronger B1 control that the learned r must beat; also a paper ablation. Kill test: seam arms on round-0 checkpoint.

## C. Self-predicted camera-execution error (free labels from own-state rows)
Own-state rows store commanded camera + estimated realized pose. Train a small head on the generated latent state to
predict per-call RPE (translation/rotation). Use: (i) write gate for M-real (write at realized pose only when predicted
error is low and estimate valid), (ii) analysis "the model knows when it did not execute the camera".
Related but different: CamVerse GRPO rewards geometry; nobody predicts own execution error as a state variable.
Kill test: AUROC of predicted vs measured RPE on dev rollouts > 0.75.

## D. Scaffold-free camera warm-up (curriculum, cheap)
Carriers closed (g≈0) when render exists. First N updates with render absent (memory_missing 100%), then introduce
render. Kill test: carrier gate magnitude and fork-RPE with render removed, vs no warm-up.

## E. Support-weighted FM loss (pose-only masks now, depth masks in round 1)
Up-weight target patches that only the archive sees (archive-exclusive support from archive_support_20261009/v2).
Forces memory use where it is the only source. Kill test: archive-removal probe delta on high-support dev rows.

## Rejected / not ours
Age-dependent memory noise (WorldPlay variant), learned history compressor (CaR), hierarchical memory (Infinite-World),
correspondence Q/K bias (Lyra): either occupied or needs a new trainable branch we cannot validate before 16 Nov.

# Part 2 (2026-10-10): mechanisms that should directly improve memory / camera / generation / long rollouts

## F. Closed-loop camera control (F′: final-checkpoint inference component)
**2026-10-10 correction and refinement; proposed, not implemented or added to the training queue.**
Scope is one final frozen C1+C2 checkpoint and one inference table, not a new DiT training arm or a cross-product with
all training ablations. Sequence: qualify runtime observer bias/proxy → pass the same-checkpoint C2 camera fork test →
fit the small response model → freeze controller settings → evaluate the ladder. No legacy gains transfer.
After a completed chunk, estimate pose from available RGB with a causal observer. Its output is an estimate, not true pose;
the old M-real and finite-prefix probes do not establish a working observer in the new single-anchor runtime.
Keep the user's desired trajectory, generator actuation trajectory, and estimated observation trajectory distinct.
Simple residual feedback remains a baseline. The preferred pilot replans the next desired waypoint from the estimated
current pose in SE(3), then uses a bounded, calibrated local command-response model to compensate under/over-response.
Apply corrections only in observable and controllable directions; do not invert near-zero camera response.
Correct the observer before fitting response: if m=A*u+b and measured m=S*m+d, naive fitting yields S*A and S*b+d.
Scale/direction errors contaminate A as well as b; offset subtraction alone is insufficient. Calibrate only supported,
well-conditioned corrections on scene-disjoint recorded fit/validation data, carry residual bias into uncertainty, and
qualify transfer to generated video. Do not use runtime future images or fit scale to the requested path.
Use a fixed anchor-depth gauge, no scale fit to requested motion, no future RGB, a calibrated deadband and bounded/smoothed
corrections. The primary pilot updates target-side rays and target render coherently; carrier-only is a separate ablation.
It cannot repair an already displayed chunk, and following future commands is allowed while seeing future images is not.

Memory proposal linked to C1: retain native appearance on pose failure, but defer its geometric registration rather than
silently calling the commanded pose measured truth. Reliable writes use estimated pose and geometrically consistent depth
in the same gauge. Confidence concerns pose observability, not small command error: a large but accurately measured miss
must remain usable for correction. Stored estimates and uncertainty must be distinguished from ground truth.
Default uncertainty proxy: two temporal subsets of the same completed chunk, with shared prefix/endpoints and fixed gauge;
combine disagreement with support/parallax and calibrated residual error. Agreement can hide common bias and is not covariance.
Two observer seeds are useful only for a stochastic observer. H is off by default; reuse a same-candidate observer result
if H is separately run, but two different generator seeds do not measure observer uncertainty. Extra subset passes are counted.

Evidence limits: the old 1.1–1.2 scaffold optimum was a denoising preference, not calibrated physical command gain
([review](../training_evidence_review_20260919/README.md)). Direct online source-pose replacement worsened mean boundary L1
0.058343→0.064009 in eight paired historical cases ([results](../online_memory_pose_20260919/comparison.md)); it does not
validate unconditional realized-pose writing. It intervened on scaffold source poses only and is not this complete controller.

Prior-art correction: [AlayaWorld §3.1](https://arxiv.org/html/2607.18367) uses the target/control trajectory as the shared
pose source; it does not establish independently estimated executed-pose writing. [WorldCrafter](https://arxiv.org/html/2609.24984)
compares pose-free readout but selects pose-guided readout. Generic inference-time feedback has broader precedents, including
[Feedback World Model](https://arxiv.org/abs/2605.15705) in robot control with actual environment observations. No “nobody does
feedback” claim. Our candidate question is whether estimated execution can jointly improve camera tracking and geometric
memory commitment while retaining appearance under uncertainty. Control theory itself is not the novelty.

After camera-path qualification, one table has at most six rows: open loop, static gain, proportional F, and F′ with the
baseline writer; then F′ with estimated-pose writing and its uncertainty gate off/on. The last two retain identical minimum
pose-validity checks. This separates pose-writing changes from gate effects without adding training arms. Independent camera
observer, fixed-gauge local/global error, boundary quality, oscillation and pose failures; count observer/calibration/re-render costs.
No arbitrary camera relabeling of the same recorded future as training GT. A lightweight controller can use detached
measured transitions; this does not replace the selected C2 training objective without a separate decision. See REDESIGN §15.

## G. Write-time colour registration
Before a generated latent is written to native memory / used for Gaussians, match its low-frequency latent statistics
(per-channel mean/var over covered patches) to the render of the same view from older memory. Stops the yellow/HSV-S
drift accumulating through the write loop. Inference-only, ~0 cost. SLAM-style exposure compensation; not in any
video world model we read.

## H. Verifier-guided sampling (test-time scaling)
Two seeds per call; pick by (a) agreement with render in covered regions, (b) realized-pose error, (c) colour drift vs
anchor. 2x inference. Positive-only change (allowed). Natural paper section "test-time scaling of memory world models".

## I. Source credit (earned reliability at inference)
Each archived source keeps a running reliability from how well its render agreed with later generated chunks in its
covered regions; render weight / r-init / retrieval score use it. Lightweight version of "earned memory" without a
persistent Gaussian map. Lyra's cache stores depth with no rejection; Memorizon pose-only.

## J. Boundary co-denoising
Re-noise the last history latent to sigma in [0,0.3] and denoise it jointly with the 4 target latents (loss on it too).
Base Wan prefers co-denoised history (T1 probe, 2026-09-01). Seam fix without warm-start blur. Diffusion-Forcing-like,
but applied only at the chunk boundary with native memory kept clean.

## K. Generator-internal depth for memory writes
Small DPT head on mid-block DiT features -> depth for the 4 generated latents, supervised by DA3 on recorded frames
(pseudo-labels already in pipeline). Memory write then uses geometry consistent with what the generator produced, with
no decode->DA3 pass; head variance gives per-patch depth uncertainty for r. Geometry-Forcing aligns features to VGGT
for quality; using the generator's own depth to write memory is different. Cost: training head + 1 extra loss.

## L. Two-pass retrieval
First pass by target frustum; after k denoising steps, find target patches still uncovered by render and retrieve
archive latents that see those directions. Denoising-aware set cover; Lyra covers once before sampling.

Suggested order by impact/cost: G, F, H (inference, testable on round-0 checkpoint) -> J, K (training) -> I, L.

# Part 3 (2026-10-10 evening)

## M. Memory-rendered counterfactual camera supervision (differential render-alignment)
Candidate C2 signal that needs no pose observer and no decode.
Pair: same history/archive/noise and the SAME input render (at command C, or absent); two commands C and C+delta;
one-step x0_hat each. Supervision direction comes from memory geometry: the scaffold rendered at C and at C+delta,
encoded to latent offline (cache +1 render per row). Loss on covered, reliability-weighted patches:
  L_M = 1 - cos( x0_hat(C+delta) - x0_hat(C),  z_render(C+delta) - z_render(C) )  [+ small magnitude term]
The model never sees the C+delta render as input, so it must move the camera from the command alone; the
"copy the render" shortcut is structurally closed. Differs from PosePilot (self warp, no counterfactual pair),
CamPilot/CamVerse (external reward, full generations), OnlineWM (simulator truth), GeoFlow/GEN3C (render as input).
C1 link: wrong-geometry sources are excluded by the reliability mask. Cost ~1.1x (two one-step forwards, no decode).
Limits: small delta (linearisation), covered regions only, does not fix absolute bias (recorded-path FM keeps that).
Kill test: matched 2k updates, 10 dev scenes, 3 seeds; render-absent fork RPE and command following under wrong render
vs render-dropout baseline. Same quantity as the C2 fork evaluation (signed differential gain).

## Recipe corrections applied to the restarted teacher (run "...20261010b")
Stopped the first teacher at update 176 (user decision b): GPU 15 % busy (per-row physical materialisation 5.5 s),
2 groups/update, Uniform(0.01,0.99) time, new-module LR 3e-5. New recipe: logit-normal(0,1) time mapped with the
same shift-5 map as sampling.schedule (engine.py), new-module LR 1e-4, accumulation 4 (profiled 35 s/update),
checkpoint every 500, teacher 6000 updates within the 72 h cap. Not changed: in-loop materialisation (prefetch or
dropout-before-materialise is the other session's call), explicit age flag.

# Part 4 (2026-10-11 night): why competitors survive their own history — implementation check vs our teacher
Evidence on our side (checkpoint 5000, recorded DL3DV path, 10 dev scenes): call-2 DINO 0.35 from own call-1 vs 0.73 from
recorded call-1; swapping only the history source recovers 75 % of the gap, only the render source 25 %; re-encoding the
generated frames through the VAE changes nothing (native-latent manifold is not the cause). Native archive adds ~0 to the FM
loss even where it has exclusive support (unreadable, not redundant).
| system | training context | per-call target | what makes it robust to its own history |
| Memorizon (model.py:70-100) | A + bank + recent CLEAN; 10 target chunks per sample, EACH chunk at its own random noise level (diffusion forcing), loss on targets only; later Self-Forcing on own rollouts with per-chunk teacher noise | 4 latents | every chunk is trained to continue from noisy/imperfect previous chunks; 10 continuation lessons per sample |
| Lyra 2 (experiment.py, lyra2_model.py) | temporal history teacher-forced + one-step self-aug every 2nd iter (t<=500), spatial slots log-normal noise, 10 % slot dropout | 20 latents = 80 frames | 2 boundaries in 160 frames instead of our 10; self-aug from the start |
| HY-WorldPlay | memory tokens noised t in [500,985] in training; context forcing; inference context timestep 14 | 16 frames | model never trusts context fully |
| GeoStream | teacher recorded; causal warmup; on-policy DMD 1k | 16 frames | distillation on own rollouts |
| ours (teacher b) | all context t=0 clean until update 5000; single 4-latent target; self-aug u<=0.5 on 50 % rows from 5000; native-only family 25 % | 4 latents | nothing before 5000 |
Gaps, cheapest first: (1) per-slot timestep input so history can be given noisy at a sampled level (diffusion forcing / context
forcing) — model.forward only supports one u for all target slots and 0 for context (pinned file, other session); (2) multi-chunk
joint training with causal mask (Memorizon) — bigger; (3) longer per-call target (Lyra 80 frames) — contract change, not now;
(4) deviation archive from recorded-command rollouts (Robust Dreamer) — needs hundreds of rollouts per refresh.
Decision rule: if 6000 shows call-2 DINO still < 0.5 with self-aug, request (1) from the other session as the next model change.
