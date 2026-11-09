# lit2-A: chunk initialisation and chunk-to-chunk transitions vs. "Reliability-Scheduled Evidence Initialization" (2026-10-09)

V = verified in local file/paper; I = inferred; PS = previously surveyed (literature_deep_20261008/A,D), not re-verifiable locally.
**2026-10-09 correction:** The old per-patch σ_init proposal did not define a matching flow path/velocity. Current
[REDESIGN §2](REDESIGN.md) uses a global-clock conditional bridge: `b=(1-q)epsilon+q·render`,
`x_u=(1-u)target+u·b`, `v_star=b-target`; holes use q=0. Learned render-agreement r drives q, not a standard per-patch
diffusion clock. The source-system observations below remain literature notes; superseded proposal conclusions are corrected below.

## Per system

**Lyra 2.0** (`/home/hj/Alican/lyra/Lyra-2/lyra_2/_src/models/lyra2_model.py`; identical to the review copy)
1. New chunk: history = clean x0, generated region = pure `arch_invariant_rand` noise (`lyra2_model.py:543-555`; training epsilon `:1445-1450`) V. Memory enters only as attention tokens (spatial slots, K/Q-only coordinate injection, `:92`) V; the most recent frame's warped RGB is channel-concatenated (lyra2.txt ~L842) V, never used as the noisy state.
2. "Temporal self-augmentation" is training-only: history chunk noised to t~U(0,0.5) (`self_aug_max_T=500`), regenerated in `self_aug_steps=1`, decoded and re-encoded through the causal VAE cache (`:1246-1290`, `:2284-2330`; `experiment.py:45-53`), p_aug=0.7 V. At inference history is not re-noised (V by absence; `condition_video_augment_sigma_in_inference=0.001` only on spatial slots, `:344-390`); spatial slots get log-normal per-frame sigma at training (`experiment.py:41-44`) V.
3. Confidence: visibility score only for retrieval (L372-377) V; none on init.
4. w/o self-aug: Style-Consistency 85.07→77.98, Camera-Ctrl 63.87→53.92, LPIPS 0.552→0.568, FID 51.3→55.2, per-frame quality up 43.4→47.9 (Table 3) V.
5. Not done: no evidence-based init, no per-patch noise level, no learned reliability. Partial noise *corrupts history* at training, never *starts the target*.

**Self-Forcing lineage: Causal Forcing, LongLive, Matrix-3.5, WorldPlay2, Wonder, LongTake, Memorizon** (`open_source_world_models_20260927/sources/*`, `deep_world_model_review_20261008/repos/*`)
1. New block = `torch.randn` at the top timestep; past enters only as KV cache written by re-running the clean block at `context_noise`=0: `causal_forcing/pipeline/causal_inference.py:275-283`; `longlive/model/base.py:267,330`; `memorizon/memorizon/pipeline.py:153-175`; `worldplay2/worldplay2/pipeline.py:628,1119` V. Wonder/LongTake: sink + FIFO/top-k KV, target from noise (wonder.txt L527-544; longtake.txt L1172-1196) V.
2. Train=inference because the student is trained on its own rollouts (Self-Forcing/DMD, Stable Forcing, Memorizon §3.4, Wonder sparse context forcing) V. Training-only context corruption: LongLive `noise_augmentation_max_timestep` + error-recycling buffer (`model/diffusion.py:240-275`); Matrix-3.5 Gaussian noise (0.03, p 0.2) on anchor/context/valid mosaic latents (`wan_video.py:3640-3705`) V. WorldPlay v1: inference context timestep 14 (`stabilization_level−1`, `pipeline_wan_w_mem_relative_rope.py:680,881-888`), memory frames trained at timestep index U[500,985] (`ar_hunyuan_mem_training_pipeline.py:322-325`) V.
3. Confidence: none on init. Matrix-3.5 is the one per-patch *timestep* precedent: mosaic-memory hole patches (2×2 all-zero) are zeroed, masked out and given **timestep 1000**, valid patches timestep 0, same at train and inference (`wan_video.py:3430-3458,3818-3820`) V — binary, on condition tokens only.
4. Numbers: WorldPlay v1 w/o Context Forcing long PSNR 18.94→16.27, LPIPS 0.371→0.495 (README) V; WorldPlay2 RevisitBench LPIPS 0.318 vs 0.439 w/o Stable Forcing V; Memorizon GT-bank vs generated-bank Imaging 0.689 vs 0.627, "three quarters to four fifths of the gap" (L1579-1599) V; LongTake boundary DINO 0.9919 vs 0.9904 V. Rolling Forcing is present (`causal_forcing/long_video/pipeline/rolling_forcing_inference.py:190-295`: staggered-noise window, only newest block fresh noise, only cleanest written to KV) V; drift 0.01 vs SF 1.66, MAGI 5% context noise + w_prev 1.5 — PS.
5. Not done: no prior in the noisy state; seamlessness is learned by the rollout loss or by overlap co-denoising, never by an evidence prior.

**Warp-as-History** (`open_source_world_models_20260927/sources/warp_history`, mirrored under geometry_guidance_sources)
1. Warped frames are VAE-encoded and inserted as *history tokens* with the target's RoPE index (`pipeline.py:1295-1335`, `rope_alignment`); the target starts from the pipeline's own noise V. Holes: `warp_invisible_fill` mean_first_frame|black; **visible-token drop** removes tokens with pooled visibility < 0.1 (`defaults.py:21-22`, `pipeline.py:1301-1330`) V.
2. Warp latents noised per-frame σ~U[0.111,0.135] at inference (`pipeline.py:1028-1058`) and training (`training/core.py:275-300`) V — identical policy, but the Helios history recipe, not evidence-scheduled.
3. Binary visibility gate on tokens, not on noise level V. 4. WorldScore camera 61.3 vs 26.4 PS.
5. Not done: evidence never enters the noisy state; invisible regions are dropped, not re-generated from partial noise.

**WorldCrafter** (`sources/worldcrafter/worldcrafter/streaming.py`)
1. `prepare_latents(latents=None)` = pure noise each chunk (`:88-98`); 3D memory enters as `latents_history_long` tokens from the RepEncoder (`:73-80`) V; prefix latents noised σ~U[0.111,0.135] (`:50-58`) V.
2. Training memory = random GT frames, no corruption (worldcrafter.txt L331-333) V. 3. None. 4. Revisit LPIPS 0.255 vs context-memory 0.497 (Table 4) V. 5. Nothing touches init.

**MoVerse** (`repos/moverse/pipeline/causal_diffusion_inference.py:15-40,150-175`; moverse.txt)
1. Gaussian-scaffold render latent height-concatenated with the noisy block (`use_cond_latent`, frame_seq_length 3120) V; block from random noise, 2-step causal student re-noised between steps, `context_noise` 0 (Fig.5 L583) V.
2. Curriculum point-warp → GS → panorama renders; Self-Forcing + RAVEN (L636-650) V. 3. None (dense render) V. 4. No tables V.
5. Warns: Stage III "may render [scaffold errors] more convincingly rather than correct them" (L790), over-smoothing on degraded renders (L796) V — the failure r(p) must prevent.

**GeoVerse** (geoverse.txt; repo README only)
1. Gaussian-noise start in DA3 Level-1 feature space (L312-321); memory projection is a condition F = V ⊙ E(I), binary validity V from point confidence + z-buffer (L296-302, L439-446) V; Wan2.2-VACE prior via zero-init residual, SDEdit-like proxy at t=500 only for feature extraction (L333-338) V.
2. Training on generated rounds unspecified I. 3. Binary per-pixel validity on the *condition* — closest to r(p), not on noise level V.
4. w/o spatial memory PSNR 19.65→17.87, LPIPS 0.398→0.434 (Table 3) V; warns depth errors accumulate through memory updates (L655) V.

**AlayaWorld v1.1** (alaya.txt L60-90; `repos/alaya/alaya/inference/pipeline.py`)
1. Point-cache render causally encoded behind a real prefix; invalid-region tokens "removed rather than masked" (L70; coverage >0.5, `inference/spatial.py:418-422`) V; target `torch.randn` (`pipeline.py:335-341`) V; optional TTC re-noises x0 at σ {0.5,0.25} (`:56-66,398-403`) V.
2. Train≠inference: stage-2b history corruption σ~U(0,0.333), p 0.9, + SVI error bank (`configs/stage2b_arsft_vigeo.yaml:134-148`), off in stage-3 DMD; prefix σ=0 at inference V.
3. Binary coverage (token removal). 4. WBench consistency 89.5, no ablation V.

**Robust Dreamer** (robust_dreamer.txt)
1. "Predicts the next latent from noise" on concat(X0, X_i, splatted memory latent, c_i) (L273-277) V.
2. Deviation archive (one-step Euler error, Bernoulli p=0.9) corrupts history *before* memory writing, training only (L282-293, L397-402) V — closest precedent for our families, on conditioning not init.
3. Per-Gaussian confidence only for pruning (L887) V.
4. w/o deviation PSNR 16.89→14.10, FID 16.8→48.2; Gaussian-noise corruption 14.25/47.5 (Table 3) V — random noise does **not** substitute for realistic deviations.

**WorldForge IRR/FLF/DSG** (`geometry_guidance_sources/WorldForge__..._pipeline_wan_i2v_clean.py:619-690`, scheduler)
1. Pure noise start (`:322`); IRR = 20 steps × 3 resamples: x0-hat decoded, pixel-blended `ref*M + gen*(1−M)` with a soft 15-px mask (`infer_worldforge.py:105-150`), re-encoded, re-noised to the *current* sigma, step repeated (`:641-660`); FLF restores flow-disagreeing channels; DSG amplifies the orthogonal component, ω 1.8 (`:663-680`) V.
2. Training-free V. 3. Hand-set per-pixel mask + channel gate V. 4. FVD 93.2 vs 113.1 w/o FLF+DSG; CFG-formula control FID 120.9 vs 96.1 PS. 5. No learned reliability, no per-patch σ, no chunking.

**Geometry Forcing** (`GeometryForcing__algorithms__dfot__dfot_video_pose.py:134-190`): VGGT alignment is a training loss only; inference = DFoT with history at stabilization 0.02 V. Nothing touches init.

**Stock Wan2.2 TI2V-5B** (`repos/wan22/wan/textimage2video.py:549-551,573-598`): conditioned first latent hard-replaced every step, `latent = (1−mask2)·z + mask2·latent`, per-token timestep 0 via `mask2*timestep` V — the base model already has a *binary* per-token timestep; ours makes it continuous per patch.

**PS only:** WorldWarp (warped sites partial noise τ=0.8, holes σ≈1, trained, 5-frame overlap; RE10K frame-200 PSNR 17.1 vs 15.2), Time-to-Move (SDEdit init from noised warp, dual clock), h-control (hard replacement at observed sites + pseudo-Gibbs; global soft pull loses, FVD 157 vs 129; "faithfully tracks wrong depth"), MAGI, FramePack crossfade, CtrlCache, DFoT k=0.02.

## Mechanism × system

| System | New-chunk state | Evidence path | Per-patch modulation | Train=infer | Trained? |
|---|---|---|---|---|---|
| Lyra 2.0 | pure noise (V) | tokens, K/Q coords | none | history-aug only at train | yes |
| SF lineage (CF, LongLive, M3.5, WP1/2, Memorizon, Wonder, LongTake) | pure noise (V) | KV cache t=0 (WP1: t=14) | M3.5 binary hole timestep 0/1000 | rollout-trained | yes |
| Warp-as-History | pure noise (V) | warp as history tokens, σ≈0.12 | binary visible-token drop | yes | LoRA |
| WorldCrafter | pure noise (V) | memory tokens | none | GT memory | yes |
| MoVerse | pure noise (V) | render height-concat | none | curriculum+SF | yes |
| GeoVerse | pure noise (V) | V⊙features + depth | binary validity | ? | yes |
| Alaya v1.1 | pure noise (V) | render tokens, invalid removed | binary | unified | yes |
| Robust Dreamer | pure noise (V) | splatted latent concat | none | deviation archive | yes |
| WorldForge | pure noise + IRR x0 blend (V) | pixel blend, flow gate | hand mask + channel gate | n/a | no |
| Wan2.2 TI2V | noise + hard-replaced latent (V) | first latent | binary token timestep | n/a | base |
| WorldWarp / TTM / h-control (PS) | partial noise on warped sites | warp as x_t | binary warp/hole mask, fixed τ | WW yes | WW yes |
| **Proposed corrected bridge** | learned mixture endpoint b, holes Gaussian | prior in x_u; native read unchanged in first arm | continuous render-agreement r(p) | proposed, untested | proposed, untested |

## Verdict (blunt)
1. Candidate distinction, not established novelty: a learned per-patch conditional endpoint weight supervised by render agreement. This survey does not prove that no other system learns reliability. Predictive-utility retrieval (FAR) and adaptive confidence methods require comparison.
2. Noised priors/hard replacement have close precedents (WorldWarp/TTM/SDEdit/h-control). Trained B1 is our confidence-policy bridge control, not a full reproduction or the completed fixed-sigma H200 arm. AUC is auxiliary; actual quality/structure/motion at matched budget decides the claim.
3. The old history-in-holes proposal is removed from the default; holes remain Gaussian. Reliability-biased reads have close visibility/gating precedents. A query-common log r cancels in a separate memory softmax; architecture placement and read benefit require their own test.
4. The strongest *measured* lever everywhere is exposure to own/corrupted context (Lyra −7 style pts, RD FID 16.8→48.2, WP2 LPIPS 0.318→0.439, Memorizon 75–80% of gap). Our families inherit this; the init must be ablated alone (r init-only / read-only, as planned).
5. Failure A — **baking in render errors**: MoVerse "render[s] scaffold errors more convincingly rather than correct them" (L790); h-control "faithfully tracks wrong depth" (PS); Lyra: warped RGB "acts as a crutch", artifacts re-generated (L423-426). Low σ_init on a confidently-wrong glass patch is exactly this; the "high-conf wrong depth → r=0" family is load-bearing.
6. Failure B — **blur/detail restoration**: the completed H200 scaffold-init/replacement variants blur substantially. **The previous σ_lo≥0.11–0.14 advice is withdrawn:** Helios/WaH values apply to history-condition noise, not target init. In our old formula σ=0.11 means 89% prior. A trained bridge may restore detail, but neither a safe numerical floor nor impossibility of texture synthesis is established. Init weight requires a matched training pilot.
7. Failure C — **memory-loop propagation / planar bending**: the cited systems discuss accumulated geometry/appearance errors. A wrongly trusted patch can be written back and reused. Age/multiview agreement are candidate features, not truth guarantees. M-real registration is a separate calibrated extension, not required in the first bridge arm.
8. Predictor-based training/inference parity is necessary but insufficient: endpoint distribution, noised target, velocity label and solver must match. Corrected foundation starts Gaussian; bridge uses predicted frozen/detached r. Oracle r* is a labelled diagnostic, not the final policy.
9. Sampler-side blending and staggered/overlap denoising are relevant precedents; this survey is not an exhaustive “only systems” claim. Training-matched overlap co-denoising remains untested locally and separate from the first bridge pilot.
10. Candidate claim: learned endpoint choice beats training-matched confidence and Gaussian controls on seen-surface preservation with naturalness/camera/motion intact. Not yet achieved. A CPU sharpness-matched spatial blur control does not reproduce scaffold-init's seam gain, strengthening the pilot rationale without proving correct geometry; see [results](../seam_tests_20261008/RESULTS.md).
