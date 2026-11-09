# GaussVerse → CVPR 2027: failure analysis and the paper that can beat SOTA

Date: 2026-09-08. Deadline: CVPR 2027 papers 16 Nov 2026 AoE (supplementary 23 Nov). ~9.5 weeks.
Sources: thesis ch. 4–6, `outputs/gate_omega/*` postmortems, `SOTA_COMPARISON.md`, `RELATED_METHODS.md`, memory ledger, fair-pilot results, remote 14-scene numbers.

## 0. Verdict in four lines

1. The current system ("store posed images, rebuild ReSplat each read, render as condition") is a GEN3C variant with a better renderer. It will not clear the CVPR novelty bar: Robust Dreamer (2605.30855, Latent Gaussian Memory + deviation archive, DL3DV), Mirage/LSM-World (2606.09828, latent spatial cache, WorldScore SOTA, 10× faster), GEN3C, Lyra 2.0, WorldStereo, GIM-World and PERSIST all occupy "3D cache rendered as condition".
2. Our measured failures are not renderer failures. They are (a) camera under-execution, (b) commanded≠realized pose contamination of memory, (c) history-reliance fragility (noise cliff), (d) too many AR boundaries, (e) the generator *degrading* evidence it is given, (f) no demonstrated recall.
3. The paper that can win is not "Gaussian memory" but **"a world model that relies on a reliability-calibrated, realized-pose-registered 3D memory instead of its own temporal past"**, with an evaluation section nobody else has (execution magnitude with truth control, null/GT arms, out-and-back recall).
4. Realistic SOTA target: beat Lyra 2.0 on camera control *and* mean LPIPS at 80/160 frames on DL3DV held-out, at 5B / one GPU / 13× faster; match or beat on terminal LPIPS. Not: beat Lyra on everything at 784×448.

## 1. Failure ledger (measured, not suspected)

| # | Failure | Evidence | Root cause (best-supported) | Cost vs Lyra |
|---|---|---|---|---|
| F1 | Camera under-execution / stall / magnitude blindness | Executed fraction 0.84 but magnitude 0.2–0.3 (truth-controlled alignment scale 19–39 vs truth 6–7); roll 300–1000× less sensitive than position; ViPE camera 61.7 vs Lyra 87.7 (GT ceiling 91.6); rotation p90 70° at 12k; present at call 0 (pure I2V) | Learned conditional under-weighting of the camera signal: history/anchor shortcut explains most of the flow loss; pretrained I2V low-motion mode; guidance 1.0; nothing forces *arrival* | The single biggest gap (26 pts) |
| F2 | Memory registered at commanded pose, content at realized pose | Realized-pose writes fixed registration (+20 Camera Ctrl on frozen 22k) but not execution; thesis §4.4.1 admits configured policy stores commanded poses | Every generated source is placed too far along the trajectory → the Gaussian scene is systematically "ahead" of its content → render at the next commanded pose asks for *less* parallax than commanded → lag reinforces itself | Terminal drift, S9-type layout changes |
| F3 | History-reliance fragility ("noise cliff") | Base Wan never breaks up to s=0.8; every fine-tuned carrier (1k→8k, AHR, 12k) breaks at s≈0.65–0.75; reliance on the H=20 window, the cliff and the co-denoise attractor emerge *together* between 1.25k and 8k; deployment sits ~0.3σ below the cliff; 1/35 first-call collapse | Clean-context fine-tuning teaches the model to trust the temporal window; trust on generated context = fragility. Clean flow ↓ and tolerance ↓ are traded under the recipe | Frames 81–160 (Lyra 0.554 vs ours 0.562) |
| F4 | AR boundary count | G=2 → 20 rounds / 19 pixel decode→re-encode→depth cycles per 160 frames; Lyra 2, SANA 1, WorldPlay 11 | Each boundary is an injection point for VAE error + off-manifold context; the flow attractor lives at σ≈0.48–0.63 exactly there | Late-window drift |
| F5 | Generator degrades the evidence | Supported-region PSNR: render 14.27 → generated 14.15 (S4), 15.38 → 15.25 (S5); render LPIPS gain 0.225 transfers as 0.018 in output; old lineage: gates decline monotonically, model "learns to cope without evidence" | Evidence is advisory (additive features), dropped 10 % and hidden in binding → model learns robustness to absence, not fidelity to presence | Object erasure (shrub S12, cars fe4bd0d4), structure→texture collapse, CLIP-IQA excess +0.04 |
| F6 | No demonstrated recall | Archive sources at trained depths still inside H=20; anchor never evicted; source selection is temporal even-spacing, not visibility; reader gates ~0 in old lineage; 4 positive removal cases only | Memory is redundant by construction at trained depths (Lyra: skip_recent=16) | Supervisor Q3/Q5; reviewers will ask the same |
| F7 | Disocclusion | Error 2–2.5× larger inside warp holes at every step; evidence helps 31 % less there | Generative-prior problem, not memory | Nobody solves this; do not spend weeks on it |
| F8 | Statistical power / provenance | 14 scenes, 1 seed, run-to-run noise 0.005 mean / 0.016 terminal LPIPS; main column is a remote aggregate; ~50 % of the mean-LPIPS margin exists at call 1 (task-match fine-tune, not memory) | — | Reviewers will discount the tables |

Two facts that should drive the design: **the render is already better than the generated output where it has support (F5)**, and **the base model is more robust to corrupted context than any fine-tuned checkpoint (F3)**. We have been training away robustness and then ignoring the one input that is correct.

## 2. The paper thesis (three linked, falsifiable claims)

> Autoregressive video world models drift because fine-tuning teaches them to rely on their own temporal past, which is exactly the input that is corrupted at deployment. A static scene does not need a temporal past: it needs a **3D memory that is registered where the generator actually went, read with calibrated authority, and honored by the sampler where it is reliable.**

- C1 (analysis, new): reliance on the temporal window, the context-noise cliff and the co-denoise attractor co-emerge during clean-context fine-tuning; base Wan has none of them. Measured by the tolerance tripwire across a lineage. Nobody has published this.
- C2 (mechanism): a memory that (i) registers generated observations at *realized* pose, (ii) predicts per-pixel reliability from support, multi-view agreement, lineage depth and render–GT error at train time, and (iii) is enforced in supported regions during sampling, forces camera arrival and stops object erasure without copying artifacts.
- C3 (recipe): shifting long-range state from the temporal window (short, corruption-trained) to the geometric memory yields lower drift slope, higher context-noise tolerance and measurable recall beyond the window.

This differs from Robust Dreamer / Mirage (latent caches at commanded poses, memory advisory, no reliability, no execution measurement) and from Lyra (geometry routes attention; RGB never enforced; 14B, 64 GPUs).

## 3. Mechanisms, ranked by (expected gain × certainty) / cost

Cheap and training-free first. Every item has a 1–3 day kill test on the frozen 5k Gaussian checkpoint before any training.

### M1. Reliability-anchored sampling (training-free, week 1) — attacks F1, F5, erasure
At denoising step k, in pixels where the memory is reliable (support ∧ opacity>τ ∧ multi-source agreement ∧ lineage-depth bound), replace the predicted clean latent x̂0 with the render latent, for steps k ≤ K_anchor only ("anchor structure, free texture"); unsupported pixels are untouched. This is RePaint-style inpainting on the *memory* rather than conditioning.
- Why it should work: F5 shows the render is more accurate than the output where supported; anchoring at the commanded target camera makes parallax unavoidable → magnitude execution at call 0 (monocular ReSplat still encodes commanded parallax); objects in the render cannot vanish.
- Risk: ReSplat blur → LPIPS ↑ if anchored late. Hence K_anchor sweep (e.g., first 30–60 % of steps) and a hard rule to release texture steps.
- Kill test: 7 fair-pilot scenes × 2 seeds, 20 calls, arms {off, K=30 %, 50 %, 70 %} × τ. Pass = ViPE camera +8 pts and supported-region PSNR ≥ render PSNR with mean LPIPS not worse than noise floor. Two GPU-days.

### M2. Learned authority head (train, weeks 2–5) — makes M1 a contribution instead of a trick
A small head on the evidence encoder predicts per-latent-cell reliability r∈[0,1], supervised at training time by |render − GT| (we have GT during training; nobody uses it for this), plus lineage depth and agreement features as inputs. r gates (a) the additive evidence features, (b) M1's anchor mask, (c) archive admission weight. Ablate: hand-crafted mask (M1) vs learned r vs no gating.
- Kill: if learned r does not beat the hand mask on held-out scenes by more than noise, keep M1 and report the head as negative.

### M3. Realized-pose registration, closed loop (weeks 2–4) — attacks F2
Estimate the realized pose of each generated span (ViPE/VGGT joint window in the anchor gauge; existing `bench/executed_motion.py` + realized-write code), store observations at realized pose, keep the commanded pose as metadata, and issue the next command relative to the *realized* camera. Reject writes only on catastrophic residuals (existing gate).
- Why: memory registered where the content is → the render at the next commanded camera carries the *full* missing parallax → pulls execution instead of confirming the lag. With M1 this becomes a proper servo loop.
- Measure: registration error (Sampson of stored vs realized), execution magnitude ratio, drift slope. The old "replan arm" rejection (39.5) was on the stalled 22k lineage without anchoring; re-test, do not inherit the verdict.

### M4. Camera classifier-free guidance via camera dropout (train, weeks 2–5) — attacks F1
Train with p=0.1–0.15 camera dropout (drop rays + 45-D + target renders; keep history), sample with camera CFG 1.5–3. This was recorded as "launch-permanent, spec-or-skip" and never run. Every CameraCtrl-class result shows it fixes under-execution. Costs 2× sampling; we have 13× headroom over Lyra.
- Not novel; report as hygiene. Kill: if camera CFG >2 hurts fidelity beyond noise, keep 1.5.

### M5. Fewer AR boundaries: G=4–5 chunks (16–20 frames/call) + latent feedback (weeks 3–6) — attacks F4
21 GiB at G=2/H=20; G=5 adds ~3 latents of tokens, feasible. Latent feedback (skip decode→re-encode, code exists) removes the VAE error injection; keep pixel feedback as the ablation. 160 frames = 8 calls instead of 20.
- Kill: G=5 trained 1.5k steps from the 5k parent must not lose first-call LPIPS; compare drift slope per frame, not per call.

### M6. Short temporal window + corruption from step 0 (weeks 3–6) — attacks F3, tests C1/C3
Train from base with H=6–8 (anchor + last chunk or two) and all-non-anchor-slot corruption s∈[0.4, 0.65] from step 0 (the tolerance lineage that reached 0/24 attractor at 950 steps). Long-range state lives only in the memory. Arms: {H=20 clean (current)}, {H=20 corrupted}, {H=8 corrupted + memory}. Report tolerance tripwire, history-reliance ratio, drift slope, recall.
- This is the experiment that turns C1 into a paper section. Kill: if H=8+memory drifts more than H=20 at 160 frames, the claim becomes "corruption training" only; still publishable as analysis.

### M7. Visibility-based source selection + forced eviction in training (week 2, trivial) — attacks F6
Select the 8 reconstruction sources by target-frustum coverage/overlap (Lyra skip_recent logic), not temporal even-spacing; in training rows, evict the relevant old observation from the window so the memory is the only route. Required for any recall claim.

### Explicitly not doing
- Persistent Gaussian map with incremental merging (occupied: GEN3C, Mirage, Robust Dreamer, GSMem; contamination without retraction).
- New readers / routers / world tokens (all measured at gate ≈ 0).
- Re-running 30k lineages; DMD/self-forcing before the mechanism is fixed.
- Disocclusion (F7): report, do not chase.
- Chasing Lyra at 784×448 or 14B.

## 4. Evaluation contribution (this is half the paper)

1. **Execution metrics with truth control**: executed fraction, stall fraction, execution-magnitude ratio, roll sensitivity (all in `bench/executed_motion.py`, `score_v3_executed_motion.py`). Show WorldScore's camera score is stall-blind (global scale fit) — a finding, not a footnote.
2. **Null (static anchor) and GT reference arms** on every no-reference metric: exposes CLIP-IQA rewarding texture collapse and style-consistency ranking a constant image 90× above real video. Deviation-from-GT forms (`bench/gt_relative_metrics.py`).
3. **Out-and-back recall protocol**: first visit → excursion long enough that all relevant sources leave the packed window (log accessible source IDs) → return; score the return region against the reference; arms {memory on, memory off, wrong-pose memory}. Report end-to-end failures (missed admission) as failures.
4. **Context-noise tolerance tripwire** as a carrier-health metric across lineages and baselines (base Wan, ours, WorldPlay-5B which shares the backbone).
5. **Scale and seeds**: 60 genuinely held-out DL3DV scenes exist on disk; use ≥30 final scenes × 3 seeds, pre-registered, per-scene sign tests (n=30 gives real power). Lyra at 40 min/scene = 20 h; SANA, WorldPlay already scripted. Add WorldScore-static for public comparability with Mirage/Lyra numbers. Cite Robust Dreamer; run it if code exists (check first week).
6. Cost table: s/frame, peak VRAM, archive growth; we are ~13× faster than Lyra per scene.

## 5. Where we can realistically beat SOTA (honest odds)

| Metric (DL3DV held-out, 160 fr) | Now | Lever | Odds of beating Lyra 2.0 |
|---|---|---|---|
| Mean LPIPS / PSNR / SSIM | already best | keep | high, but must survive n=30 × 3 seeds |
| Camera (ViPE) 61.7 vs 87.7 | M1 + M3 + M4 | medium; closing to ≥80 is plausible, ≥88 needs luck |
| Terminal LPIPS 0.573 vs 0.572 | M1 + M5 + M6 | medium-high (gap is within noise; boundaries and anchoring both push it) |
| Style drift / KID | M1 anchoring keeps appearance | medium |
| Recall beyond window | M7 + benchmark | unknown; nobody reports this on DL3DV, so a clean positive at any magnitude is a result |
| Speed | 8 s/8 fr vs Lyra 40 min/scene | already; M5 improves further |

If M1 alone moves camera ≥ +8 and supported-region PSNR ≥ render, the paper exists. Everything else strengthens it.

## 6. Schedule (9.5 weeks, one GPU)

| Week | Work | Gate |
|---|---|---|
| 1 | M1 kill test on frozen 5k; M7 selection; tolerance tripwire on 5k model, WorldPlay-5B and base; check Robust Dreamer / Mirage code | M1 passes or paper pivots to analysis + M6 |
| 2 | M3 realized-pose loop on frozen model with M1; recall protocol built; pre-register 30-scene split | Registration error ↓, execution ratio ↑ |
| 3–5 | One training run from base: M4 dropout + M2 head + M6 corruption + G=4 (M5), ~5k updates (~2 days each; two runs max) | Must not lose first-call LPIPS; tolerance 0 attractor to s=0.8 |
| 6 | Ablation arms (1.5k steps each from the same parent): no head, no anchor, H=20, G=2 | |
| 7 | Final eval 30 scenes × 3 seeds all arms + baselines; WorldScore-static | |
| 8–9 | Writing, figures, supplementary video; freeze numbers by 9 Nov | |

Thesis submission runs in parallel; the paper reuses its baselines and boards but not its remote-only column.

## 7. Framing rules for the paper

- Lead with the analysis (C1) and the two measured facts (render beats output where supported; base beats fine-tuned on corrupted context). The mechanism follows from them.
- Position against Mirage / Robust Dreamer explicitly: latent caches at commanded poses are advisory; ours is registered, calibrated, enforced.
- Report camera under two instruments and always with the GT ceiling; report every no-reference metric as deviation from GT.
- No "first Gaussian memory" claim. No claim without per-scene sign tests at n≥30.

## 8. Two-stage program: accurate teacher first, then distill for speed (added 2026-09-08)

User direction: push accuracy and world quality first, then distill. Correct ordering (Lyra, AlayaWorld, HorizonDrive, ABot all do teacher → student). Two consequences from our own evidence:

1. **Distillation transfers the teacher's distribution; it cannot add accuracy.** A teacher that under-executes cameras produces a student that under-executes faster. So every accuracy lever (M1–M7) belongs in the teacher, and the teacher may be as slow as it likes: 30–50 steps, camera CFG 2, G=4–5 chunks with full attention inside the chunk. The student inherits CFG for free (DMD bakes guidance in) and can be chunk-causal with a KV cache (current model uses ordered temporal positions, `models/wan22_frame_renderer.py`, so the old "zero-temporal RoPE blocks caching" objection is obsolete).
2. **Context-noise tolerance is the distillability gate.** In rollout-DMD / LongForcing the teacher scores the student's *own* rollouts, i.e. corrupted context. A teacher that sits 0.3σ below the attractor cliff (every fine-tuned checkpoint so far) returns garbage scores on exactly those samples. The tolerance tripwire (0/24 attractor to s=0.8, base-level) must pass on the teacher before any student stage. This makes M6 (corruption from step 0) mandatory in the teacher, not optional.

### Stage 1 — teacher accuracy gates (all before any distillation)

| Gate | Threshold | Instrument |
|---|---|---|
| Camera execution | magnitude ratio ≥ 0.8 with truth control; ViPE camera ≥ 80 on held-out (Lyra 87.7, GT 91.6) | `bench/executed_motion.py`, ViPE |
| Evidence honored | supported-region PSNR of output ≥ render PSNR | regional scorer (thesis Table S4/S5 form) |
| Tolerance | 0/24 attractor to s = 0.8 | `probe_gate_omega_context_noise_tolerance.py` |
| Persistence | near-field object survival rate; out-and-back recall positive with memory on vs off | recall protocol (§4.3) |
| Drift | per-frame LPIPS slope over 81–160 ≤ Lyra 2.0 | horizon curves |
| Prior quality | CLIP-IQA excess vs GT ≈ 0 (no texture inflation) | `gt_relative_metrics.py` |

Teacher configuration to aim for: base Wan2.2-5B init, 256×448 (ReSplat checkpoint and DL3DV-480P GT ceiling both fix this; a 480p refiner is post-CVPR), G=4–5 latents per command with latent feedback, H short (6–8) with corruption from step 0, camera dropout 0.1–0.15, authority head, realized-pose registration, visibility-based source selection, training pool widened from 3,299 toward the full DL3DV-10K (cache cost only; Lyra uses ~10k). Sample at 30–50 steps, camera CFG 1.5–3, anchored sampling on.

### Stage 2 — distillation that fits one RTX 5090 (32 GB)

VRAM reality: 5B bf16 ≈ 10 GB per copy. Full DMD needs student + fake score + teacher = three copies plus optimizer state; does not fit. Feasible path:

- **2a. ODE distillation to 4 steps with an offline teacher-trajectory cache.** Run the frozen teacher once over the cached rows and rollout-bank states, store (z_s, s, z_0, conditioning ids) pairs; the student then trains alone (same footprint as today's training). ABot's stage 2. Keep anchored sampling in the student's first 1–2 steps (free). Teacher CFG is folded into the cached trajectories.
- **2b. Rollout self-forcing / DMD with LoRA roles.** One frozen bf16 base, LoRA student, LoRA fake-score, sequential checkpointed forwards; teacher = base + teacher LoRA if the teacher is itself LoRA-trained, else the full teacher offloaded and scored on a subsample. Student trains on its own rollouts with ReSplat memory in the loop (existing `OmegaOnPolicyTrainingState` capture). This is where remaining drift is fixed (HorizonDrive teacher-rollout DMD, ABot LongForcing).
- **2c. Runtime.** After 30→4 steps the DiT is no longer 99.7 % of the command: ReSplat rebuild (8 views, 4 refine), DA3 and VAE become the floor. Measure the non-DiT share first; expected ~1–1.5 s per 8-frame command → 5–8 FPS, ~10 FPS with FP8 + compile. Interactive, not 24 FPS; say so.

Must be re-measured after distillation (these decouple from PSNR): execution magnitude, Sampson, tolerance tripwire, distilled-with-memory vs distilled-without (few steps may not integrate evidence).

### What this does to the CVPR timeline

Teacher + evaluation is the paper. Stage 2a (offline ODE cache, 4 steps) is achievable by week 8 as a speed table if the teacher run finishes by week 6; 2b is post-deadline or supplementary. If the distilled student must be in the main paper, drop M2 (learned authority head; keep the hand mask), never the evaluation section.
