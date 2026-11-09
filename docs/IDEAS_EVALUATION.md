# Evaluation of the ideas (9 Nov 2026)

Judged against what the line has measured, not against the literature alone. The measured bottleneck is appearance after
~160 frames, not camera: on the DL3DV cohort Gauss-meas leads every released baseline on camera position over the first 160
frames (0.0078 vs Lyra-2 0.0095) and executes 0.99 of the command through the render, while LPIPS per 80-frame bin opens from
.11 to .58 (Lyra-2 .08 → .51) and last-bin DINO is .70 vs .87. The visual failure after frame ~240 is splat-render texture
(speckle, see-through foliage) and border extrusion at the frontier; both come from a memory rebuilt from the model's own
degraded frames and read back as the dominant carrier (render ≈ 2:1 over the temporal history in conflicts).

## Verdicts

| Idea | Verdict | Reason |
|---|---|---|
| Measured-pose writes + on-policy measured rows (Gauss-meas) | **keep, core** | the only training step that moved both fidelity and camera; "Ours" rows of the SOTA table |
| F′ closed loop (offset rejoin, early x0 check) | **keep, core** | 6/6 wins on released Memorizon; "+ correction" rows; cheap at inference |
| GM-B context corruption | **keep** | camera/DINO/sharpness up on 6 dev scenes, LPIPS down on 4/6; user counts it as an improvement; it attacks exposure bias, not texture |
| Frontier mask at inference (frustum-only) | **keep, best dev arm** | 20:35 result: on the untrained GM it is the best arm on every aggregate (late LPIPS .590, rot 13.3°, 2.2 excursions/scene; kitchen 33° → 3.1°). The depth-agreement variant was the wrong instrument (ReSplat soft depth ≠ DA3) |
| GM-F (training with masked + stripped renders) | **drop** | worst arm with the same inference mask (rot 30.6°, kitchen 60.8°): lowering render reliance costs the camera, which is executed through the render. Lesson for every future arm: never train the model to trust the render less |
| Restricted original-source appearance read | **most promising untested idea** | directly targets the re-render/re-encode degradation: the generator would read the source VAE features that geometry says are visible instead of a texture that was rendered, downsized, VAE-encoded and re-rendered every call. GEAR's dense-vs-restricted ablation (.60 → .45 LPIPS) supports restriction. The G+B arm did not test it cleanly: it ran on the 2Xplat map with generated history, which lost for other reasons, and ReSplat/DA3 correspondences were never qualified. Cost: a qualification of the correspondence contract (CPU) + one 250-update arm (≈2.5 h) |
| Render VAE dialect (encode the render with real previous frames as causal prefix) | **cheap, do before the reader** | the render is encoded as a 17-frame clip with its own first frame duplicated, while every other latent is stream-causal; the κ-sweep report from the other session found the same mismatch. One cache rebuild + 250 updates |
| First-step memory (single-source ReSplat alpha-valid ≈ .48 in foliage scenes, foreground object missing) | **small, worth fixing** | every rollout starts from a half-empty render; `first_mode` 'dup4'/'points' exist in `renderer864.render_chunk` and were never evaluated on the 6 scenes |
| Write policy: do not overwrite supported content with re-generated views (source credit / earned reliability, D-ablation of the Sep plan) | **plausible, second tier** | WonderWorld is the one system where frozen old geometry works because nothing is regenerated over it; ours re-reconstructs from the latest frames each call. Needs the bank selection to prefer older recorded-quality sources; measurable with the existing closed loop |
| Map-addressed attention bias / 2Xplat global memory | **drop for now** | lost on generated history (5 Nov); the global map compounds registration error of generated frames |
| Gaussian-feature metadata (depth / conflict / support channels) | **drop** | FAIL on 1 Nov; adds geometry the model already gets from the render |
| Per-chunk render groups (R2) | **drop** | no win 10-26..11-01 |
| Memory-verified camera reward (RVM / NFT) | **defer** | camera is not the bottleneck; CamVerse / WorldCompass report quality loss under camera RL; the reward arm costs ≥ 3 GPU-h per 100 updates with G candidates. Revisit only if the SOTA re-evaluation shows camera regressions |
| Camera-ReFL with a latent pose head | **drop** | pose head failed its gate (Spearman .14 on the student's own latents); DA3 is the only reliable camera instrument |
| Feedback-conditioned read (explicit tracking / execution error) | **drop** | the design note itself concedes the measured history plus corrected commands already carry the residual; passive ratios are not causal gains; needs recovery examples we do not have |
| GATE-Ω zero-gated archive reader | **drop (historical)** | gates stayed ≈ 0 over a full run; memory redundant at trained depths |
| Counterfactual-memory framing | **drop as a claim** | Gate-2 oracle gap ≈ 1 % for warped-RGB memory; useful only as the evaluation discipline it introduced |
| Learned two-view refiner | **drop** | blur; the discriminator was not a sharpness signal |
| Selective visual memory (appearance-only vs full use of an unreliable view) | **absorb** | its practical form is the frontier mask (trust = 0 where unobserved) plus the appearance read above; no separate arm |

## Recommended order (GPU is serial; each arm ≈ 2.5 h train + 1.2 h dev eval)

1. Finish what is running: GM-F / GM with the frustum mask on the 6 dev scenes, then GM-B / GM-F / GM+mask on the 5-scene
   SOTA cohort with the paper's scorer. This decides the GM-B-vs-GM-F base for everything below.
2. First-step memory and render-dialect fixes on the winner (inference-only test first for `first_mode`; dialect needs a
   render-cache rebuild). Both are cheap and attack the memory quality the model reads, not the model.
3. Restricted appearance read: qualify ReSplat/DA3 correspondences on recorded motion (CPU, same reprojection code as the
   frontier mask), then one 250-update arm from the winner with `packed_model_corr` in restricted mode and matched rows.
   Success = last-third LPIPS/DINO on the SOTA cohort moving toward Lyra-2 without a camera regression.
4. Write policy (older recorded-quality sources preferred, re-generated views admitted only where unsupported) as an
   inference-only ablation on the same cohort; train only if the inference test moves the last third.
5. Camera reward only if a camera regression appears.

## What would change these verdicts

- The frontier mask (inference) is now the headline mechanism; the appearance read is its complement (generate where unobserved,
  read sources where observed). The SOTA-cohort runs (GM + mask, GM-B, GM-B + mask) decide the base model.
- If the correspondence qualification shows ReSplat depth cannot address source tokens reliably (as it could not pass the
  15 % depth test), the appearance read must use DA3 depth of the sources directly (frustum + visibility only), as the mask does.
