# Online camera correction: diagnostic experiment

## Current implementation

For the subsequent causal moving-window observer and measurements after frame 80,
see [window_pose.md](window_pose.md). It is a separate opt-in output monitor;
denoising previews still use the frozen observer described below.

Two observers are available through `bootstrap_walking_session`:

- `pose_log_path`: asynchronous localization of decoded output against a frozen
  observed RGB-D anchor. Results include position/rotation errors, match support,
  reprojection residuals, and relative motion errors.
- `denoising_pose_log_path`: synchronous localization of predicted clean frames at
  sampler steps 10, 20, 25, and 30, followed by a separate final-output measurement.
  These previews add latency. They do not correct the sample.

By default both observers measure the first and last RGB frames of each eight-frame
command. `pose_every_frame=True` measures all eight. Every record contains the call
index and RGB frame ID; preview records additionally contain the denoising step,
sigma, stage, and decoding/localization timing. Missing localization is represented
by an explicit status and null pose errors, never by zero error.

The current reference map is the observed anchor with estimated seed depth. A
measurement is therefore conditional on that depth and calibration, not external
ground truth. Translation is measured in seed-depth world units. The metric column
uses the caller's `anchor_median_depth_meters`; it does not infer physical scale.

The ordinary online observer has a bounded CPU queue and can skip work when busy.
The denoising diagnostic runs synchronously and uses a separate localization object.
There is no claim of real-time performance for the denoising diagnostic.

## Run against an already loaded walking runtime

Run from `/home/hj/Alican` with the `framecrafter` Python environment. Here `runtime`,
`anchor_rgb`, and `anchor_depth_meters` are the existing loaded runtime, observed
3xHxW RGB tensor in [-1, 1], and externally supplied physical anchor-depth scale.
Choose new output paths for each run to keep repeated call/frame IDs unambiguous.

```python
from gaussverse.training.gate_omega_runtime import bootstrap_walking_session

session = bootstrap_walking_session(
    runtime,
    anchor_rgb=anchor_rgb,
    anchor_median_depth_meters=anchor_depth_meters,
    pose_log_path="results/pose_output.jsonl",
    denoising_pose_log_path="results/pose_denoising.jsonl",
    pose_every_frame=True,
)
try:
    result = session.command("Q", yaw_degrees=10.0, global_seed=7)
    for record in result.denoising_pose_records:
        print(record["stage"], record["denoising_step"], record["frame_id"],
              record["status"], record["rotation_error_deg"])
finally:
    session.close()
```

Do not silently substitute a frozen training-run snapshot for the main-tree runtime.
Record the checkpoint, conditioning configuration, scene, seed, and source revision
alongside these files. The workspace map explains why historical launcher names do
not identify the model/configuration used in reported results.

## Why the preview hook is before the scheduler update

The Wan flow sampler predicts velocity `v` at noisy latent `x_sigma`. Its clean
prediction is `x0 = x_sigma - sigma * v`. The hook receives this prediction before
the Euler update, using the scheduler's own `to_final=True` operation. Combining an
already-updated latent with the old sigma would produce an incorrect preview.

Each callback receives an independent tensor. Preview decoding snapshots and restores
the stream cache, including on failure. Only the ordinary final decode advances video
history. The final output is measured separately because postblend and dtype conversion
can distinguish it from the last clean preview. Temporal pose comparisons reset between
previews: denoising steps revisit the same video timestamps rather than advance time.

## Research hypothesis, not implemented behavior

The proposed experiment asks whether camera error can be corrected during generation
and then prevented from returning during subsequent denoising.

For a clean latent `h`, a proposed generative increment `d`, pose residual `e(h)`, and
local derivative `J = de/dh`, an idealized correction is:

```text
d_corrected = d - J.T @ inverse(J @ J.T + damping * I) @ (J @ d + gain * e)
```

With exact derivatives, full row rank, zero damping, and sufficiently small steps,
linearization gives `e_next ~= (1 - gain) * e`. This is a local statement, not a
convergence guarantee for a stochastic nonlinear video generator. Translation and
rotation must be normalized consistently. One frame contributes up to six observable
pose directions; an eight-frame chunk is not generally described by only six numbers.

The current SIFT/RANSAC observer is not differentiable and does not provide `J`.
Implementing a reliable derivative, or a geometric latent basis that approximates its
effect, is the central open work. A small pose-space solve does not make the required
preview decoding, geometry, or basis construction free.

Pose preservation also does not guarantee scene preservation. A generator could
deform geometry while satisfying the measured landmarks. Freeze reference geometry
during correction, limit corrections to supported directions, and score geometry and
appearance with information that the controller did not optimize against.

## First experiment and decision gates

1. Use short static-scene chunks with fixed histories, camera paths, and noise seeds.
   Begin with rotations and small translations that retain observed-scene overlap.
2. Log output pose error and localization success through denoising. Calibrate the
   instrument with real/known rendered controls before interpreting generated frames.
3. Check whether trusted localization becomes available early enough for later
   denoising steps to perform useful correction. If localization only succeeds on the
   final image, this observer cannot support the proposed inner loop as-is.
4. Once observability is established, compare the original sampler, one final
   rectification, selected-step correction, and selected-step correction with drift
   suppression. These corrective arms are not implemented yet.
5. Report pose error versus denoising step together with failure rate, image/geometry
   quality, and end-to-end latency. Use matched compute and an independent evaluator.
6. Only then extend to recurrent rollouts and measure whether accepting corrected
   chunks reduces boundary jumps and reconstruction inconsistency.

The synthetic CPU checks verify observer/sampler mathematics and cache isolation.
They do not establish early-preview observability, pose-correction efficacy, perceptual
quality, full-model latency, or publication-level novelty.

## First real-output measurements (2026-09-18)

The reproducible experiment and plots are in
[`analysis/online_pose_probe_20260918`](../../analysis/online_pose_probe_20260918/README.md).
Two predetermined scenes were measured against their two frozen observed prefix
views, followed by a one-anchor ablation matching the production reference count.

- All 16 first-call generated frames localized. Median rotation residuals were
  0.346° and 0.236°; real-target controls were 0.159° and 0.168°, respectively.
- Across the first 128 generated frames per scene, only 16/128 and 41/128 localized.
  Real controls localized in 35/128 and 92/128 frames and developed substantial
  pose bias as the view changed. Accepted PnP status is not calibrated accuracy.
- First-call CPU measurement took roughly 16–18 ms per frame at 448×256, excluding
  reference setup, decoding, queueing, and generation.
- A separate historical late-call panel localized 0/32 generated frames and 8/32
  real targets. Its successful real controls had about 4.46° median rotation error.
- With only the observed anchor, all 16 first-call generated frames localized;
  generated rotation medians were 0.300° and 0.286°, with real controls around
  0.114°. Overall availability was 50/256 generated frames, at roughly 13–15 ms
  per first-call frame. Adding a second reference does not resolve map limitations.

These results support testing early denoising in the first call. They do not support
using this frozen-anchor instrument as a long-rollout pose controller. Translation
is in fixed deployment-gauge units, not meters. Reference depth is estimated.
Map extension must avoid assigning generated references their requested poses:
that would introduce the answer into the measurement. Independent pose estimation,
uncertainty propagation, and relocalization remain necessary research components.

The subsequent full-model first-call probe succeeded on both scenes. All 16 frames
localized at steps 10, 20, 25, and 30. Median rotation residuals at step 10 were
0.358° and 0.386°, and final residuals were 0.288° and 0.245°. Fresh-run real-target
controls were around 0.170° and 0.166°. This establishes an early feedback window
in these examples, but leaves little margin above the instrument's error floor.
A useful correction test needs larger, confidently measured pose deviations.

The observer-enabled sampler produced latents exactly equal to the original frozen
sampler; decoded RGB also exactly matched the saved baseline in both scenes.
Four full previews increased sampler time from about 1.66 seconds to 2.99–3.14
seconds, plus a separate final diagnostic decode. These single-run timings include
diagnostic RGB capture and are not end-to-end real-time benchmarks. No pose
correction was applied.

## Related work to distinguish experimentally

- [WorldForge](https://arxiv.org/html/2509.15130v3): modifies intermediate clean
  predictions with trajectory guidance and motion-selective latent fusion.
- [NeoMap](https://arxiv.org/html/2607.01962v1): alternates generative and geometric
  projections for novel-view synthesis.
- [Feedback World Model](https://jingliangli.com/Feedback_World_Model/): corrects
  predictions using observations from executed robot actions.

The proposed distinction is feedback from independently measured realized camera pose
and suppression of camera drift introduced by subsequent generative updates. This is
a research direction, not an established first-of-its-kind claim.
