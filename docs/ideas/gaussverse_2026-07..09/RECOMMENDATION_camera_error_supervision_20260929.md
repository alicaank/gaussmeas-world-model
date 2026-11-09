# Using camera error to supervise or guide diffusion

29 September 2026. Static-scene setting. Research/code review only: no training or guidance experiment launched.

**Recommendation:** supervise geometry inside generation. Use Geometry Forcing as a training baseline and a WorldForge-inspired sampler as an inference baseline. The research candidate is a correction driven by independently observed geometric discrepancy, with fixed scale and explicit content/continuity constraints. A pose-error number alone is neither a differentiable loss nor sufficient supervision for a whole scene.

## 1. What the two methods actually do

| Method | Intervention | Useful baseline for us | What it does not establish |
|---|---|---|---|
| Geometry Forcing | Align diffusion hidden features with frozen VGGT features during training | Geometry-aware representation learning without an RGB decode on the student branch | Exact execution of a requested camera or a persistent map |
| WorldForge | Warp source content to requested views, then inject guidance during denoising using recursive refinement, selective latent fusion and dual-path guidance | A sampler that can use geometric evidence even when ordinary conditioning is weak | Independent pose-error feedback on our autoregressive runtime |

Geometry Forcing combines angular feature alignment with prediction of unnormalized geometric features from normalized student features. Its “scale” loss concerns feature magnitudes, not a guarantee of metric camera scale. The paper reports 2,500 camera-conditioned fine-tuning steps on 16-frame clips. Its geometry memory discussion is future work. [Paper v2](https://arxiv.org/html/2507.07982v2)

WorldForge changes the denoising process without training. Guidance comes from source-view warps; it is not a RoMa pose-loss loop. Its trajectory evaluation uses SfM followed by global Sim3 alignment and scale correction. We should reproduce that as a secondary literature-compatible metric, while retaining fixed-gauge execution error as our primary metric. Otherwise uniform under-travel can be removed by evaluation alignment. [Paper v3](https://arxiv.org/html/2509.15130v3)

## 2. Code findings that affect an implementation

### Geometry Forcing

Reviewed commit `f5dfc6c5c3cbd2c0dad0121c0f7b161d2886e683`.

- The training wrapper sends clean training frames to the geometry teacher, while diffusion produces the student hidden states.
- VGGT is frozen; teacher feature extraction uses `no_grad`. Student projection and the diffusion representation remain trainable.
- The released loss reshapes teacher tokens and normalizes flattened feature axes. It should not be copied assuming a simple per-patch channel cosine loss. Tensor axes and reduction scaling need an explicit unit check when porting to our DiT.
- The released training shell overrides the alignment coefficient to 0.1, while the base YAML says 0.5. The script/config/paper are not a single interchangeable recipe.

Sources: [training wrapper](https://github.com/CIntellifusion/GeometryForcing/blob/f5dfc6c5c3cbd2c0dad0121c0f7b161d2886e683/algorithms/dfot/dfot_video_pose.py), [alignment loss](https://github.com/CIntellifusion/GeometryForcing/blob/f5dfc6c5c3cbd2c0dad0121c0f7b161d2886e683/external/vggt_alignment_loss.py), [training script](https://github.com/CIntellifusion/GeometryForcing/blob/f5dfc6c5c3cbd2c0dad0121c0f7b161d2886e683/scripts/train_geometry_forcing.sh).

### WorldForge

Reviewed commit `ee573a051715a451b806a90e21462f23308faac4`, Wan implementation.

- `fuse_latents` decodes the predicted clean sample, blends warped RGB under a mask, re-encodes it, and selectively restores original latent channels. This is more than overwriting an arbitrary latent mask.
- The pipeline performs repeated denoiser calls and re-noising, then combines velocity predictions using a norm-adjusted orthogonal component. The extra forwards and VAE passes must count in an equal-compute comparison.
- Channel selection is implementation-specific. Its channel counts cannot be transferred directly to our 48-channel VAE.

Sources: [scheduler/fusion](https://github.com/Westlake-AGI-Lab/WorldForge/blob/ee573a051715a451b806a90e21462f23308faac4/wan_for_worldforge/utils/scheduling_unipc_multistep_clean.py), [denoising pipeline](https://github.com/Westlake-AGI-Lab/WorldForge/blob/ee573a051715a451b806a90e21462f23308faac4/wan_for_worldforge/utils/pipeline_wan_i2v_clean.py).

## 3. What our new evidence says

The independent RoMa run has completed all 576 solver records. Comparing abrupt repaired memory C with requested-pose rebuilding B on 31 common endpoints:

| Solver | Median paired rotation difference C−B | Median paired position difference C−B |
|---|---:|---:|
| PoseLib | +0.13643° | +0.00807 map units |
| OpenCV | +0.26725° | +0.00720 map units |

Thus the existing repair's LPIPS gain does not translate into improved measured camera execution. These remain estimator-based discrepancies; the two solvers share RoMa correspondences, so their agreement is not two independent image observers. D-related comparisons remain provisional while the strict replay repair waits for the active training pipeline.

This makes camera-specific supervision relevant. It does not show that fixing camera motion will fix texture/identity drift. Our prior post-hoc alignment tests also do not exclude geometry errors causing later content damage through feedback.

[RoMa report](../repair_roma_20260929/RESULTS.md) · [verified fidelity/transition pilot](../memory_repair_pilot_20260928/CHECKED_RESULTS.md)

## 4. Can we directly add the measured camera error to the loss?

Conceptually yes. For an independently inferred camera `E(I_hat, references) = (R_hat, C_hat)`, a candidate is

`L_pose = a ||Log(R_req^T R_hat)||² + b ||(C_hat − C_req)/s_ref||²`.

`s_ref` is fixed from the reference gauge, not fitted to the generated trajectory. It makes units comparable; it does not supply true metric scale. The estimator must infer pose from pixels/reference evidence without receiving the requested query pose.

**Our current measurement implementation cannot supply this gradient.** RoMa's `forward` and `match` use `torch.inference_mode()`. Matching is followed by detached NumPy coordinates and OpenCV/PoseLib RANSAC. Converting its scalar error back into a tensor does not reconnect it to the generator. Removing one decorator is not a complete differentiable-estimator implementation.

Alternatives are a frozen image-based differentiable pose network, differentiable correspondence/reprojection supervision, or black-box candidate scoring. A geometry network may have frozen weights while still retaining gradients with respect to its input images. Do not wrap that student-image branch in `no_grad`.

A trainable pose head attached to camera-conditioned hidden features can simply read the camera command. Low head error would not prove that generated pixels obey it. Keep output-image evaluation independent.

Local evidence: [RoMa implementation](../pose_estimator_upgrade_20260927/romav2/src/romav2/romav2.py), [matching and solvers](../pose_estimator_upgrade_20260927/run_roma.py).

## 5. Preferred direct objective: put supported landmarks where the requested camera predicts

For a static world point `X_i` reconstructed from an accessible source, project it using the requested camera-to-world transform:

`p_i* = project(K T_req^{-1} X_i)`.

Let `q_i(I_hat)` be a differentiable predicted location of the same source feature in the generated image, using frozen dense descriptors and a soft correspondence distribution. Then

`L_reprojection = sum_i w_i rho((q_i(I_hat) − p_i*) / image_diagonal) / (sum_i w_i + eps)`.

This says: keep the same visible landmark and place it where the camera command says it belongs. It supervises both motion and supported content without demanding the recorded GT appearance of every unseen pixel.

Implementation constraints:

- Freeze source coordinates, descriptors, scale and visibility weights during each optimization. Otherwise the system can reduce loss by moving the map or hiding inconvenient points.
- Use only causally available sources at deployment. The initial anchor supplies actual observed evidence; generated archive points provide self-consistency evidence, not newly observed truth.
- Visibility, occlusion and depth uncertainty affect continuous weights. Do not discard fitted evaluation poses because their error is large. Missing supervision is not a successful zero-error case.
- Soft matching must be checked for ambiguity. A soft-argmax can average two wrong matches into the requested location; descriptor contrast and spatial-support diagnostics are needed.
- Dense feature loss at projected positions is a simpler alternative, but it can also encourage duplicate texture. Independent geometric evaluation is essential.
- Unseen regions remain generative. This loss cannot recover unsupported real content.

A six-degree-of-freedom pose objective also permits nonrigid cheating: the model could put a handful of features in the right places while deforming the scene. Multi-region track constraints and source-identity checks reduce that freedom but do not mathematically guarantee complete geometry.

## 6. Where this enters our diffusion code

The active flow objective uses

`z_sigma = (1 − sigma) z_clean + sigma noise`, with target velocity `noise − z_clean`.

Therefore the predicted clean target is

`z0_hat = z_sigma − sigma v_theta`.

For training, decode `z0_hat` with the correct detached causal history cache and apply an image-space geometry term. Gradients must remain connected through the current decoded chunk. The old history can remain detached, avoiding full backpropagation through the rollout.

For inference, first test a bounded clean-latent correction at selected denoising times:

`z0_star = argmin_z [ ||z − stopgrad(z0_hat)||²/(2 eta) + lambda L_geometry(Decode(z)) ]`.

Then return the corrected clean prediction to a scheduler-consistent denoising step. For Euler flow, recompute a compatible velocity from the current noisy state and corrected clean prediction, with special handling near sigma zero. Do not change a clean prediction and continue using an incompatible cached velocity/solver state.

This can use gradients through the VAE and frozen feature network without differentiating through the full DiT: detach `z0_hat`, optimize that temporary latent, then resume denoising. That is a proposed proximal correction, not exact classifier guidance. It has to be compared with a no-correction extra-compute control.

The production sampler is wrapped in `no_grad` and has no geometry correction hook. The streaming VAE snapshots detach history, while its core current-chunk decode can potentially retain gradients. A separate experimental path must verify gradient flow, cache isolation and zero-strength identity before use. VAE encoder and decoder cache states must be cloned for each candidate; speculative candidates must not alter committed history. WorldForge's full-clip VAE calls cannot be pasted into this streaming path unchanged.

Sources: [active flow forward](../../training_runs/revisit_long_20260929/runtime/gaussverse/training/gate_omega_forward.py), [production sampler](../../training_runs/revisit_long_20260929/runtime/gaussverse/training/gate_omega_sampling.py), [streaming VAE](../../training_runs/revisit_long_20260929/runtime/gaussverse/models/vae_stream.py).

## 7. Geometry Forcing as the training baseline

Start with ordinary valid clean/compatible-history rows, a frozen geometry teacher on real training clips, and a small projection from selected DiT target tokens. Teacher GT future is training supervision, not deployment input. The projector must train with at least part of the generator; training only the probe cannot change output generation.

Keep the parent optimizer state and a matched extra-training control. Audit gradients from flow and geometry objectives separately before clipping: a small scalar loss can still dominate the clipped update, as our P3 experience showed. Weight scheduling is a tunable design choice, not a coefficient to copy from a different backbone.

Only after the objective works on compatible rows should it be combined with genuinely deployed history. Matching clean future features against a badly drifted history can reintroduce the target conflict seen in F. Direct source-supported reprojection offers a narrower target in such rows, but wrong source geometry remains a risk.

A 25/100-update run can establish functioning gradients or detect collapse. It cannot fairly reject representation learning for lack of a rollout gain. Use an adequately funded learning curve, sample/compute accounting and fresh validation scenes; training steps across architectures are not equivalent budgets.

## 8. A potentially stronger research direction

Under the static-scene assumption, separate a coherent camera discrepancy from residual inconsistency. For local residuals `r_i = q_i − p_i*`, compute the pose-projection Jacobian `J` of the fixed map and fit

`delta_xi = (J^T W J + mu I)^{-1} J^T W r`.

`J delta_xi` is the component explained by one small rigid camera change. The remainder is **not automatically content error**: it can include bad depth, occlusion, calibration and correspondence errors. Planar scenes, low parallax and poorly distributed points also make the fit ambiguous. Damping improves numerical conditioning but does not create observability.

The candidate idea is to use that observable rigid component to steer the current denoising chunk, while separate source-feature retention handles supported appearance. This differs from only matching hidden geometric features or copying a warped video. However, pose losses, reprojection constraints, visual servoing and diffusion guidance already have substantial prior art; this review of two methods does not establish novelty.

The contribution would need evidence that this separation gives better camera adherence, content fidelity and boundary continuity at matched compute and across long rollouts, without exploiting its own evaluator.

## 9. Smallest useful experimental sequence

1. **Gradient feasibility:** frozen generator; a few saved states spanning early and late calls. Verify nonzero, finite gradients from current decoded pixels to a temporary target latent. Check finite-difference direction and zero-strength identity. Verify all speculative VAE caches leave the original state unchanged.
2. **No-training output intervention:** parent, an equal-compute parent, a WorldForge-inspired masked fusion baseline, and the direct supported-geometry correction. Same source budget, commands and initial noise. Use original source RGB/features; do not treat a generated map as observed truth.
3. **Evaluate independently:** full RoMa fits/errors and missingness, real controls, a second image/video estimator where feasible, fixed-gauge rotation/translation, supported track residuals, LPIPS and boundary/intra-call motion. Report source coverage and long-rollout drift. No best-of-N claim without charging all candidates.
4. **Training baseline:** Geometry Forcing versus matched continued training, followed by reprojection supervision as a separate arm. Distill a successful sampling correction only after the correction itself works.

If previously displayed history is already at the wrong pose, immediately reaching the requested next pose can conflict with smooth motion. A bridge trajectory changes intermediate commands and must be reported as such. For the first experiment, correct each new chunk **before** it is shown and committed; do not retrofit old displayed frames or silently change requests.

All new single-image experiments should expose one observed initial anchor. Extra reference views in the legacy evaluation are a separate experimental setting and must not be silently reused as single-image input. At late times, when the anchor provides no support, a generated archive can enforce consistency but cannot certify true appearance or global pose. That remains a central limitation.
