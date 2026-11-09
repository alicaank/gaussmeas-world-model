# GaussVerse experiments and results

This document records what has been measured, how trustworthy each number is,
and the methodology lessons paid for along the way. For system design see
[architecture.md](architecture.md); for the code map see
[implementation.md](implementation.md).

**How to read the numbers.** Camera-pose numbers are reported at the checkpoint's
own training resolution (never cross-resolution), on the pinned held-out set
(`data/heldout.py`), as a **per-scene median** with the frontier scenes bucketed
separately. Means over the raw scene set are not used: the score distribution is
bimodal, and a mean is dominated by a small tail of low-coverage collapses.

> ## ⚠️ Metric revision — read before any section below
>
> A metric audit **invalidated every camera-pose number produced by the 2-view
> VGGT estimator** (`bench/metrics.vggt_pose_err`). It runs VGGT on one anchor plus
> the generated frame — its least-constrained regime — with no confidence signal, and
> grossly *overstates* the camera-only baseline's error (reported 141°/60° on scenes
> whose true error was ~5–47°). That manufactured the large apparent improvements the
> older sections below report.
>
> **Metric trust ranking**, established by validation:
> 1. **Epipolar Sampson** — primary. Estimator-free; validated separation correct-pose
>    0.0003–0.006 vs wrong-pose 0.05–0.09.
> 2. **Registered VGGT** — joint reconstruct + Umeyama-align to GT cameras; recovers
>    metric scale but *understates* wrong-pose frames.
> 3. **2-view VGGT** — do not report.
>
> Sections **§5.3, §5.6, §5.9** reach conclusions that the current metrics reverse or
> nullify. They are kept for the record with correction notes; **§13 supersedes them.**
> The headline "OFF 35.4° → 10.0°" result was an artifact and does not survive.

---

## 1. Milestone-1 gates (feasibility)

| Gate | Question | Result | Status |
|---|---|---|---|
| M1-A | Does the frontier benchmark separate known from invented? | warp-oracle known 15.9 dB vs invented 7.7 dB, monotone across coverage; 205 tuples, 9/9 strata | PASS |
| M1-B one-step | Fits the GPU; adapter exactly inert at init? | bitwise zero-equivalence; 12.4 GB peak at both 192×320 and 384×640 | PASS |
| M1-B six-step | Does it train? | loss 0.78 → 0.34 | PASS |
| M1-B scenes100 | Is exact pose learnable on an overfit set? | learnable; permutation delta 0.7% | PASS |

The zero-init residual attachment means the model at initialization *is* the
pretrained Wan function; feasibility never required touching the backbone.

---

## 2. Camera binding — the mechanism reads the commanded pose

VGGT pose error alone cannot tell whether the model reads the commanded rays or
merely infers a plausible pose from context appearance. The binding battery
(`bench/camera_binding.py`) scores the model against ray fields recomputed about
*other real cameras* at graded geodesic distance ("graded lies"). All arms share
noise, sigma, and context; only the ray field changes.

Result (`ckpt_040000`, 40 tuples, 4 held-out scenes):

| arm | loss ÷ correct |
|---|---|
| correct | 1.000× |
| wrong &lt;15° | 1.159× |
| wrong 15–45° | 1.314× |
| wrong &gt;45° | 1.423× |

Loss rises monotonically with the size of the lie; every rung is
CI-significant (`large − small = +0.0475`, CI `[+0.034, +0.063]`). **Verdict:
BINDS.** A zero-lie control (wrong camera &lt;2°) produces a near-zero loss
change, confirming the metric tracks pose error specifically, not perturbation.

Note on method: an earlier bar keyed on `zero_rays` (all ray planes zeroed) was
dropped as confounded — zeroing 1536 channels is out-of-distribution for the
conv stem, so its loss is inflated for reasons unrelated to pose. The graded-lie
arms are all valid Plücker fields and are the trustworthy test.

---

## 3. Resolution ladder — camera precision rides on ray density

The single most consequential finding for deployment. A `384×640` checkpoint
must be evaluated at `384×640`; the Plücker ray field is four times denser than
at `192×320`, and the model learns to read sub-cell ray variation that is lost
on downshift.

The full 2×2 matrix, hardened eval on the pinned 16-scene held-out set (per-scene
median rot / trans-dir):

| median | @192 | @384 |
|---|---|---|
| r16 `ckpt_040000` (192-native) | **6.01° / 10.84°** | 5.75° / 13.51° |
| r32 `ckpt_030000` (384-native) | 8.03° / 21.15° | **3.75° / 6.74°** |

**Native-vs-native, r32@384 clearly beats r16@192** (~37% rotation, ~38%
translation, 3 vs 5 frontier collapses of 16) — deploy r32@384. Resolution
sensitivity is *asymmetric*: r32 craters run at 192 (it depends on the dense ray
field it trained on), but r16 is robust run at 384 (upsampling adds ray density
it can ignore). The rule "train at deploy resolution" therefore bites in the
*downsampling* direction specifically. The VGGT-resolution-inflation confound was
ruled out — the 192-trained r16 does not improve at 384.

Caveats on attribution: the r32 advantage conflates rank (16→32), native
resolution, data (200→1000), and steps, which these checkpoints cannot separate.
The held-out set is fully clean for r16 (it postdates r16 training) but only its
4 pure scenes are clean for r32; on the pure-4 subset the two tie (≈20°), though
that subset is small and frontier-dominated. Honest summary: r32@384 is
meaningfully better on typical scenes (easy-scene rot 2.73° vs 4.41°), the
magnitude is confounded, and the frontier collapse (§5) is shared by both.

Ladder cost: peak VRAM is `13.6 GB` at 192 / `15.0 GB` at 384 (batch 2, rank
32); throughput `7.4` vs `3.1` samples/s. The 384 rung is affordable on one 32 GB
card.

---

## 4. Gate decay is benign — FiLM and the ray adapter share load

Training the r32 run with CameraFiLM enabled from step 0 makes the two camera
pathways compete, and the ray-adapter gate decays monotonically
(`tanh(g)` 0.66 → 0.40 over 5k→25k). Because a zeroed ray adapter is
catastrophic elsewhere in the diagnosis, a falling gate that also lost binding
would be a red flag.

Binding sweep across checkpoints (`bench/binding_sweep.py`), same frozen tuple
set:

| step | gate `tanh(g)` | graded (large − small) | |
|---|---|---|---|
| 5k | 0.66 | +0.053 | BINDS |
| 15k | 0.51 | +0.054 | BINDS |
| 25k | 0.40 | +0.050 | BINDS |

Binding is flat as the gate falls. The decay is a redistribution of camera load
between the ray branch and FiLM that jointly preserves binding, not a loss of
control. This keeps the alternative (moving camera geometry into attention, a
UCPE/PRoPE-style port) parked rather than triggered.

---

## 5. Trustworthy pose baseline — and the bimodal reality

`bench/pose_eval_hardened.py`, pinned 16-scene held-out set, 8 targets/scene,
deterministic, `ckpt_030000` @384, 30 steps:

- **Median rotation 3.75°** (IQR `[2.63, 4.95]`), **median trans-dir 6.74°.**

The median is not the story — the *shape* is:

| bucket | n | rot median |
|---|---|---|
| controlled | 13 | **2.73°** |
| frontier | 3 | **68°** (36 / 68 / 78) |

Thirteen of sixteen scenes sit near 2.7° — camera control is genuinely tight.
Three scenes collapse to 36–78°, with individual targets reaching 170°. Even
within controlled scenes, only the widest-baseline targets spike. This is the
low-coverage invention problem, concentrated rather than uniform: on a frontier
target the model generates a coherent scene at the wrong orientation (see the
visual panels in §7). The failure is generative-geometry, not camera-mechanism —
consistent with binding (§2) and the earlier coverage-stratified rotation
diagnosis.

### 5.1 The far-target failure is extrapolation, not measurement

The visual export samples a far target at 0.9 of the trajectory with context at
0.25/0.5/0.75 — the target sits *outside* the context span, so the model must
extrapolate. Two diagnostics separate the causes.

First, is it a VGGT measurement artifact (a 2-frame reconstruction at wide
baseline)? No. `bench/vggt_protocol_test.py` measures the VGGT floor on
*ground-truth* far frames: median rot 1.4°, trans-dir 1.5° with the 2-frame
protocol. VGGT measures far poses reliably; the model's far failures are real. A
4-frame protocol is *worse* (unstable — it breaks one scene from 5° to 43°), so
adding context frames to the pose estimate is not a fix.

Second, is it extrapolation specifically? Yes. `bench/bracket_context_test.py`
regenerates each far target with context *bracketing* it (0.80/0.87/0.96) versus
behind it, measured against a reliable near anchor:

| far target, median | rot | trans-dir |
|---|---|---|
| extrapolated context | 7.50° | **42.35°** |
| bracketed context | 3.71° | **5.37°** |

Bracketing the context cuts translation-direction error ~8× (`fbb7` 136°→6°,
`ff11` 60°→3°). Translation is the casualty of extrapolation: the model guesses
rotation but cannot invent camera *position* beyond what it has observed — which
is exactly the "rotation right, translation wrong" pattern seen in individual
far panels. When the target's neighbourhood is observed, translation is accurate
to ~5°. The one exception (`fe30`, a texture-poor ceiling) stays hard even
bracketed.

`bench/export_eval_full.py` re-runs all three regimes with image metrics added
(PSNR, SSIM, LPIPS-VGG vs the GT target); the visual report renders them:

| regime | rot | trans-dir | PSNR | SSIM | LPIPS-VGG |
|---|---|---|---|---|---|
| near (interpolation control) | 4.52° | 5.89° | 13.01 | 0.333 | 0.0059 |
| far — extrapolated | 4.68° | **35.43°** | 11.58 | 0.338 | 0.0063 |
| far — bracketed | 3.26° | **4.56°** | 13.19 | 0.367 | 0.0060 |

Two readings. (a) **Translation-direction is the discriminative metric** (35°→4.6°
under bracketing); rotation is flat, PSNR follows weakly (+1.6 dB), SSIM and
LPIPS-VGG barely move — for this *comparison* the image metrics move little
because pose, not fidelity, is what changes. (b) **Consequence:** the intended
interpolation use is accurate today; the extrapolation stress case is the
frontier problem. Both regimes are reported by `export_eval_full`.

### 5.2 PSNR relative to FrameCrafter (~17)

FrameCrafter reports ~17 PSNR; our full-frame median is ~13–14, which prompted a
gap investigation. Findings:

- **Not an exposure/decode bug.** Brightness offset is 0.5%; matching per-image
  mean or per-channel mean+std does not move PSNR (13.55 → 13.59).
- **PSNR is bimodal, like pose.** Good/textured scenes reach **16–21 PSNR**
  (`fbb7` 21.1, `fb95` 16.5) — at or above FrameCrafter's 17 — while
  frontier/low-texture scenes sit at ~9–12 and drag the median down. The model
  is competitive on comparable scenes.
- **Matching FrameCrafter's 6-context protocol does not help us** — it slightly
  hurts (median 14.1 → 12.7), because the model was trained on 2–3 context views
  (`num_ctx_choices=(2,3)`); six views are out-of-distribution. FrameCrafter was
  trained on six.
- **Remaining legitimate gap factors:** FrameCrafter reports *region-masked*
  PSNR (known vs missing separately; known-region PSNR runs ~20–29 in related
  data), we report full-frame; plus 5B vs 14B, 384×640 vs 480×832, and harder
  targets. The apples-to-apples number is region-masked known-region PSNR on
  well-covered interpolation targets — not yet run.

So GaussVerse is not behind FrameCrafter on comparable scenes; the reported-median
gap is bimodal composition plus a full-frame-vs-region-masked protocol difference.

### 5.3 Memory-evidence branch — the frontier translation fix

> **CORRECTED (see §13).** This section's pose conclusion is an artifact of the 2-view
> VGGT metric. Under epipolar Sampson the memory branch has **no camera-pose effect at
> all** (MEM−OFF Δ +0.0001, 6/14 wins). It is a **content/fidelity** mechanism:
> +0.796 dB in warp-covered regions. The reasoning below about *why* warped geometry
> should help still stands; the claimed pose improvement does not.

The diagnosis (§5.1) says the frontier failure is the model inventing camera
*translation* beyond what context observed. The fix is to *give* it observed
geometry: the memory-evidence branch (warped context depth + canonical coords,
encoded to the Wan latent space) supplies the missing position anchor. It trains
on top of a frozen camera model (only the zero-init evidence adapter is
trainable), so OFF is bit-identical to the camera model and the ablation is a
clean paired A/B.

Early gate (`bench/gate_memory.py`, 4k steps — adapter still ramping, gate
≈0.77), 16 pinned held-out scenes, wide-baseline (0.9) targets, **DA-V2 depth**:

| | trans-dir median | rotation Δ | PSNR Δ |
|---|---|---|---|
| OFF (frozen camera) | 35.4° | — | — |
| ON (+ memory) | **19.6°** | +0.45° | −0.07 |

Mean Δ(ON−OFF) = **−19.25°**, CI95 **[−42.95, −0.34]** — entirely below zero, so
CI-significant even partly trained. Rotation and PSNR are unchanged; memory moves
*translation* specifically, exactly the diagnosed failure. The wins are the
frontier collapses: `fdd2` 141°→5°, `fbb7` 137°→27°, `ff11` 61°→8°. One honest
regression (`fd434230` 1.5°→41°, a well-covered scene) was traced to the
depth-scale caveat — poor DA-V2 depth on that scene feeds misleading evidence. As
§5.6 shows, swapping to VGGT depth removes this regression entirely, confirming the
diagnosis.

### 5.5 Warp quality — depth *estimation error*, not scale, is the limiter

The memory branch helped even with imperfect warps, but a direct check of the
warped evidence itself found it far from clean. Masked warp-PSNR on covered pixels
(warp vs GT), fixed scene/target:

| depth source | masked warp-PSNR |
|---|---|
| DA-V2 (per-frame, affine) | 9.06 dB |
| MoGe-2 (per-frame, metric) | 13.54 dB |
| MoGe-2 + cross-view scale fit | 14.59 dB |
| **VGGT (joint, Procrustes-registered)** | **15.74 dB** |

A well-aligned warp scores 20+ dB, so even the best is imperfect. The root cause
of cross-view fringing was isolated (`bench/moge_warp_test.py`,
`bench/vggt_warp_test.py`): per-frame monocular depth is estimated *independently*
per image, so the same surface lands at inconsistent depths across frames — visible
as rainbow fringing in a false-colour overlay of individually-warped frames.
Structural to monocular estimation. VGGT estimates all frames jointly (consistent
by construction) and, registered to the known GT poses, gives the best warp
(15.74 dB in that relative-frame measurement).

> **⚠ Scale bug (found & fixed 2026-07-23) — the 15.74 dB VGGT warp did not reach
> training.** `estimate_consistent_depths` originally registered VGGT to
> *target-relative* camera centers (divided by `scene.s_scene`), returning depths a
> factor `s_scene` too small, while `build_keyframe_cache` (the consumer, via
> `build_target_evidence`) pairs them with *absolute* `scene.c2w` poses. The 15.74 dB
> number was measured by `vggt_warp_test.py`, which used a self-consistent relative
> frame; the *training/gate* path used absolute poses and so ran the warp at the
> wrong scale — **cov 0.53 / 9.94 dB** vs the correct **cov 0.78 / 15.75 dB**
> (scene fe4bd0d4, s_scene 3.15, target 258). DA-V2 never had this problem (its
> `disparity_to_depth` is absolute-metric), so the VGGT swap silently *regressed*
> the warp to DA-V2 quality despite running VGGT. **The 8k-step memory+correspondence
> training and every gate below (§5.6) therefore ran on ~10 dB evidence, not
> 15.75 dB.** Fixed by registering to absolute `scene.c2w[ctx]` centers; verified
> the absolute evidence path now returns 15.75 dB. Consequence: §5.6's correspondence
> attribution is confounded by bad-warp and must be re-run on the fixed depth (§10).

Even correct joint depth leaves residual error (a moving car stays inconsistent),
which is the argument for treating the warp as an *uncertain correspondence hint*,
not trusted content (§5.6). VGGT is the depth backend for the memory trainer.

### 5.6 Correspondence branch (P1) — real but mixed; binding preserved

> **CORRECTED (see §13).** Pose numbers here come from the demoted 2-view metric. The
> branch's real, re-measured contribution is **fidelity by region**: BOTH−MEM is
> +0.067 dB covered and +0.094 dB frontier, both CI-significant — i.e. it helps
> disproportionately where there is no warped appearance to copy, exactly as the
> content/correspondence split predicts.

The memory branch's one regression (`fd434230`) is the failure mode its content
residual makes possible: a bad warp corrupts Q, K, *and* V, so it hallucinates
pixels. The correspondence branch (Proposal 1) injects the *same* warp's canonical
coordinates as a Q/K-only bias that leaves V untouched — a wrong correspondence can
only misroute attention, not invent content (see architecture.md). Both adapters
were trained *jointly* for 8k steps on VGGT depth; `bench/gate_correspondence.py`
scores three arms on the pinned 16-scene held-out set (OFF = frozen camera,
MEM = memory only, MEM+CORR = both), paired per scene:

| | trans-dir median |
|---|---|
| OFF | 35.4° |
| MEM | 21.6° |
| **MEM+CORR** | **11.6°** |

The median improves monotonically. But the mean-based bootstrap CI is
**inconclusive** — d(MEM+CORR − OFF) CI95 `[−35.2, +2.1]` spans zero — because two
scenes blow up the variance. `fbb7b6b3`: memory alone nearly fixed it (137°→23°)
and correspondence *undid* it (→147°, worse than OFF); `fc18ec31` similar
(52°→87°). Excluding those, 11 of 16 scenes show correspondence helping *further*
than memory alone (`fe0691c0` 62°→32°, `fc18f248` 22°→1.9°). Rotation and PSNR are
net-neutral. **Visual inspection is load-bearing here and corrected an over-generous
first read:** the `fbb7b6b3` failure is a *real* corruption (zoomed crops show
doubled/ghosted railing geometry), and a scene the metric scored as a clean win
(`fe4bd0d4`, 8°→5°) actually carries a new white-patch artifact on the car — smaller
than `fbb7`'s but the same family. So the honest read is not "one outlier, rest are
wins" but "correspondence introduces *some* visible artifact on a non-trivial
fraction of scenes, severity ranging from a small patch to a scene-wide corruption,
plausibly worst where the underlying VGGT warp is itself least reliable (the car
class VGGT already flagged as inconsistent)." At 8k steps with `gate_corr` still at
0.877 (unsettled), this reads as under-calibration, not a disproof of the approach.

**The original `fd434230` regression is fixed — but mostly by VGGT depth, not by
P1.** Under the VGGT-depth checkpoint, memory *alone* no longer regresses it
(1.5°→1.0°, not the DA-V2-era 41°); correspondence on top is roughly a wash there
(1.0°→5.2° trans, rotation improves 32°→25°). So the depth-consistency fix and the
injection-mechanism fix are partially redundant on that specific failure; P1's
distinguishable value is on the 11/16 scenes where it improves an already-VGGT-backed
memory branch.

**Binding is preserved with both branches active** (`bench/camera_binding.py`
extended with `--with-memory`, which keeps `mem_adapter`/`corr_adapter` live through
every arm of the graded-lie battery; evidence is built once from the honest
context/target so a ray-lie cannot be masked by an honest coordinate channel).
Camera-only baseline and MEM+CORR active are near-identical at matched rank-32 /
384×640 / 32 tuples:

| decisive check (graded large − small) | verdict |
|---|---|
| camera-only | +0.0168 CI `[+0.011, +0.024]` — BINDS |
| + MEM+CORR active | +0.0170 CI `[+0.011, +0.024]` — BINDS |

Every check (a)/(b)/(c) matches within noise; med−small is identical to four
decimals. The hypothesized risk — that the correspondence channel, which encodes
the *true* target pose via canonical coordinates, could leak pose information and
dilute the ray-binding signal — did **not** materialize. Binding is untouched.

Net: P1 is safe to keep training and using (camera control unaffected, typical case
improved, failure localized to specific hard/OOD inputs), but it is **not** the
clean win an aggregate-only read suggested. **[Superseded by §5.8 — this run was
confounded by the depth-scale bug of §5.5; on corrected evidence the result is a
CI-significant win and the ghosting resolves.]**

### 5.8 Fixed-depth retrain — the ghosting was the scale bug

Re-running §5.6 on the corrected VGGT depth (§5.5) is decisive: the confounded
result was largely an artifact of the wrong-scale warp, and on correct evidence the
branches produce a clean, CI-significant translation win.

Training-time signal first: the memory gate ended at **+0.883** on fixed depth,
versus **+0.587** (a steady decay from 1.0) on the confounded run. On a wrong-scale
warp the optimizer was *learning to distrust* the memory content residual; on a
correct warp it trusts it. That is direct confirmation the fix changed the evidence,
not just a downstream number.

Gate (`gate_correspondence`, same 16 pinned scenes, wide-baseline), fixed vs
confounded, trans-dir median:

| | OFF | MEM | MEM+CORR |
|---|---|---|---|
| confounded (~10 dB warp) | 35.4° | 21.6° | 11.6° |
| **fixed (15.75 dB warp)** | 35.4° | **12.1°** | **10.0°** |

The whole result moved from inconclusive to significant: **d(MEM+CORR − OFF) = −18.66°,
CI95 [−39.1, −4.1]** (was `[−35.2, +2.1]`, spanning zero). The two visually-confirmed
ghosting corruptions of §5.6 are **resolved**: `fbb7b6b3` 146.7°→**134.1°** (≈ OFF
136.5°, no longer corrupted worse than baseline; rotation 10.9°→3.3°), `fc18ec31`
87.0°→**52.1°** (now better than OFF). So the ghosting was the scale bug — not the
one-sided injection or the frozen backbone. The plan's H1/H2 question was confounded
by a third cause (H3, bad warp), now eliminated.

**Correspondence is genuinely additive on correct evidence.** Branch attribution
(n=16, bootstrap): MEM−OFF mean −22.1° `[−49.8, +0.2]` (large but frontier-composition
dominated, just spans 0); BOTH−OFF `[−39.1, −4.1]` significant; BOTH−MEM mean +3.43°
`[−12.9, +27.9]` spans 0 — but dominated by a *single* frontier outlier (`fd4dc277`,
MEM 13°→BOTH 177°). **Excluding it, BOTH−MEM = −7.29°, CI [−15.3, −1.2]: significant**,
and MEM+CORR beats MEM-alone on **12 of 16** scenes. The `fd4dc277` failure is a
*pose-selection* error, not corruption — both arms render clean, coherent buildings,
they simply choose different camera positions (MEM near GT at 13°, correspondence
biased to the wrong solution at 177°). That is the frontier extrapolation/invention
problem, orthogonal to the injection mechanism.

**Consequences.** (a) Frozen-base + memory + correspondence on *correct* depth
already delivers a CI-significant translation win with no visual corruption — the
fine-tune ladder (§10 Steps 2–3) is much less urgent; the frozen backbone is not the
blocker it appeared to be. (b) Two-sided correspondence (§10 Step 1) is now an
*optimization*, not a rescue — worth trying to convert frontier pose-blowups, but the
branch already earns its place. (c) The main remaining failure is frontier pose
ambiguity (`fd4dc277`, `fe30a865`, `fbb7b6b3` still hard) — the invention problem,
not evidence-reading. Checkpoint: `outputs/memory_vggtfix/ckpt_last.safetensors`.

### 5.9 Two-sided correspondence — ~~negative result~~ **REVERSED (see §13)**

> **This conclusion is withdrawn.** It was reached with the 2-view VGGT metric. Re-scored
> at matched step 4000 under region-split PSNR, two-sided **nearly doubles the frontier
> gain** (+0.258 → +0.457 dB; paired two-sided − one-sided +0.198, CI95 [+0.070, +0.350])
> at no significant cost in covered regions (−0.087, n.s.). **Two-sided is now the
> recommended default.** The reasoning below for why the matching term should help was
> right; the measurement that rejected it was not.

Proposal 1 injects the target's canonical coords into Q/K. Lyra also injects the
*context* frames' coords, so the attention logit carries a coordinate-to-coordinate
matching term (`pₜᵀ W p_c`) — a target query "I am at world point X" matching a
context key "I contain X". We implemented this two-sided variant (context tokens
carry their own coords; `data/evidence.py two_sided=True`), retrained 8k steps, and
gated it (`gate_correspondence --two-sided`):

| wide-baseline | median OFF→MEM→BOTH | d(BOTH−MEM) | BOTH beats MEM |
|---|---|---|---|
| one-sided | 35.4 → 12.1 → **10.0** | +3.4 CI[−12.9,+27.9] | 12/16 |
| two-sided | 35.4 → 12.1 → **16.4** | +1.3 CI[−4.7,+7.3] | 9/16 |

**Two-sided is slightly worse** — the median regresses (16.4 vs 10.0) and it helps
fewer scenes (9 vs 12). It is *more consistent* (tighter CI, fewer wild outliers),
but the matching-term hypothesis did not convert the frontier failures at 8k steps.
**One-sided stays the better config; two-sided is parked** (could be under-training —
it is a harder learning problem — but there is no evidence it helps). The full
mechanism is implemented and tested behind the `--two-sided` flag for future use.

### 5.10 Sequential (rollout) regime — easier for the base model, motivates temporal

The main gate uses context *spread* across the trajectory. The rollout regime is
different: clustered *early* context predicting a target *ahead*
(`bench/eval_sequential.py`, ctx `[0,3,6]` → target 18). A cheap warp-coverage pass
first showed this regime is easier for the memory branch — median coverage **0.88**
/ warp-PSNR **18.5 dB** (vs the wide-baseline ~0.78 / 15.75), because the target
overlaps the clustered context heavily.

The full gate (one-sided fixed-depth model, the *zero-temporal baseline*):

| sequential | median trans-dir |
|---|---|
| OFF | **17.5°** (vs wide-baseline 35.4° — the base model handles this regime far better) |
| MEM | 17.5° (median flat; d(MEM−OFF) mean −4.2°, MEM beats OFF 11/16 — helps on average) |
| +CORR | 20.9° (no median gain; one catastrophic outlier `fd4dc277` 4→170°) |

The base camera model is already good here (17.5°). Memory helps modestly on average
but the median is flat — partly because the memory adapter was trained on
wide-baseline tuples, so it is somewhat *out-of-distribution* for clustered-sequential
context. **The key finding: the sequential OFF failures are the large-forward-motion
scenes** (`fe4bd0d4` 143.8°, `fd434230` 148.7°, `fe0dbd7f` 127.0°) — exactly the cases
where the model must *extrapolate the trajectory trend*, which zero-temporal RoPE
prevents. So the data motivates temporal RoPE concretely: the failures are the
extrapolation cases temporal ordering targets. This 17.5° is the baseline it must beat.

### 5.11 Autoregressive rollout — stable, pose-imprecise, and "weird cars" = bad evidence

`bench/rollout_video.py` rolls the model forward feeding its *own* generations back as
context (ctx `[0,3,6]`→gen 18; ctx `[0,3,6,18]`→gen 26; … 5 steps to frame 50, context
grows 3→8), OFF vs ON, along the commanded GT trajectory. Outputs: an mp4
(`[GT | OFF | ON | trajectory]`), a comparison strip, a trajectory PNG (in
`outputs/rollout/<scene>/`).

Findings on `fe4bd0d4`:

1. **Stable — no collapse.** Both OFF and ON hold a coherent building across all 5
   steps, even with context growing to 8 frames (OOD — trained on 2–3). Feeding its
   own outputs back does not destabilise generation.
2. **Pose-imprecise.** The rendered viewpoint does not match the commanded forward
   position (step-1 = 143° trans error). The model renders a coherent building from a
   slightly-wrong-but-self-consistent viewpoint. Stable ≠ accurate; the error is
   per-step pose imprecision, not accumulating runaway drift (foreground sharpness
   stayed roughly flat over the 5 steps — a Laplacian-variance check *refuted* an
   initial visual impression that OFF blurred and memory resisted it; both are
   comparable and neither collapses within this horizon).
3. **The "weird car copying" is broken warped evidence, not training.** Rendering the
   warp *the model is fed* (`warp_car_zoom.png`) shows the cars are already severely
   distorted *before* the model runs — the Jeep stretched with a spray of hole-speckles,
   the sedan smeared. So the model faithfully renders broken input. **It is neither
   zero-init nor step count**: zero-init only governs training start (converged by 4k);
   more steps would inject the bad evidence *more* faithfully. Cars break in the warp
   because (a) VGGT depth fails on reflective glass / dark paint, (b) large foreground
   parallax magnifies depth error, (c) forward motion dis-occludes car angles the
   context never saw (the holes). This is the "trusting bad evidence" failure at the
   evidence level, and it is why a *frozen* adapter cannot fix it — see §12.

### 5.7 Why SOTA methods do better — and it is not (only) 8k iterations

The 8k-step correspondence run plateaued in training loss by ~4k (loss ~0.22 flat
to 8k), so the mixed result is not primarily step-starvation — more steps on a flat
objective converge artifacts, they do not remove them. Comparing to the methods
this line draws from surfaces the likely architectural gaps, ranked:

1. **One-sided vs two-sided correspondence.** Lyra 2.0 puts canonical coordinates
   into Q/K on *both* sides of attention (target *and* context tokens), so the logit
   carries an explicit coordinate-to-coordinate term `pₜᵀ W p_c` — "I am at world
   point X" matching "I contain world point X." Our `qk_bias` lands on target tokens
   only; context tokens carry no coordinate signal, so that matching term does not
   exist and the ghosting is exactly what mixing geometry-biased and appearance-based
   routing would produce. This is a genuine implementation gap, cheap to fix.
2. **Frozen backbone.** SOTA methods fine-tune the backbone (Lyra: Wan-14B;
   ABot-World-0: full-parameter FT of Wan2.2-**5B**, our identical backbone). Their
   backbone *learns to repair* imperfect evidence; ours is frozen + LoRA, asking
   2–3 M adapter params to teach a frozen stack to trust a noisy warp. The
   `fd434230` story fits: the fix came from cleaning the evidence (VGGT), not the
   model learning robustness.
3. **Evidence bandwidth.** GEN3C/Lyra channel-concat warped content into the noisy
   latent at full resolution, reprocessed by every layer; we compress 51 channels
   into one residual added at 2×2 patch granularity, once.
4. **Fighting the temporal prior.** Wan pretrained on ordered video; we feed an
   unordered set with zero-temporal RoPE (right for the permutation gate, but it
   disables the backbone's strongest asset). This is the Phase-5 item.
5. **No CFG.** Trained without conditioning dropout, so CFG > 1 collapses the image
   (§5.4); SOTA leans on CFG for conditioning adherence. Fixable only in a new run.
6. **Scale**, last — not step-limited given the 4k plateau, but 984 scenes and
   effective batch 8 is small next to the SOTA corpora.

Ruled out as the cause: camera binding (§2, §5.6), the depth backend (VGGT is best
available and fixed the worst failure, §5.5), and the eval protocol (§8). The
actionable order is §10.

### 5.4 Fidelity is not the gap — judge at full resolution

A detour worth recording. Downscaled report thumbnails and PSNR measured on hard
targets suggested the renderer was "soft / low fidelity" even on interpolation.
Full-resolution inspection (`bench/fidelity_diag.py`) refuted it: at `640×480`,
interpolation generations are close to ground truth — `fe30` stairwell PSNR 16.3,
`fe4b` temple 16.4 (ornate roof, signage, cars all reconstructed), `ff11`
boardwalk faithful *despite* SSIM 0.086 (foliage texture that looks right but
does not pixel-align tanks SSIM). The softness was a **visualization artifact**
(220 px thumbnails), not the model; thumbnails are now `440 px`. The CFG × steps
sweep is a dead end: 30→50 steps is flat (converged), and CFG > 1 collapses the
image (PSNR 14.7→5.7) because the model was trained without conditioning dropout,
so an unconditional pass is out-of-distribution. Conclusion: fidelity on
interpolation is competitive; the real gap is frontier *pose* (§5.1, §5.3), not
image quality. Lesson: inspect full resolution before judging generation quality.

---

## 6. Inference latency

`bench/infer_bench.py`, `ckpt_030000` @384, single target. Per-frame latency is
linear in denoising steps, as the compute is ~99.7% the repeated DiT forward:

| steps | sec/frame |
|---|---|
| 30 | 2.48 |
| 15 | 1.24 |
| 10 | 0.83 |
| 6 | 0.50 |
| 4 | 0.33 |

`torch.compile` (blocks, `mode="default"`) gives **1.14×** (2.51 → 2.19 s),
quality-neutral; `reduce-overhead` (CUDA graphs) crashes on the residual-stream
aliasing and must not be used. Caching the ray/FiLM adapters across denoising
steps was analysed and rejected: those stems are ~0.3% of per-frame compute, so
the only material latency levers are fewer steps (distillation) or cheaper steps
(fp8, kernels). The step-count quality cliff — how few steps the model tolerates
before camera adherence degrades — should be read with the hardened eval at
30/10/6/4 steps; it was not measurable earlier because the pose metric was too
noisy (see §8).

---

## 7. Visual results

`bench/export_eval_visuals.py` writes `[ctx0 ctx1 ctx2 | GT | GEN]` panels per
held-out scene (a near and a far target), with the generated frame labelled by
its VGGT error and tinted red when rot ≥ 10°. Panels and a rendered report live
under `outputs/eval_visuals/`. The visual read matches the numbers: controlled
scenes reproduce the target view closely; frontier targets show a coherent but
mis-oriented scene — the model's generative prior is intact where its pose
control fails.

---

## 8. Methodology lessons (paid for in this project)

Three corrections are recorded here so they are not re-learned:

1. **Held-out identity must be pinned by scene hash, not list position.** The
   trainer/eval selected `list_scenes()[-N:]`. When the DL3DV download grew the
   dataset 200 → 1000 scenes mid-experiment, that tail became four different
   scenes, so the *same* eval command returned rot 2.97° on one run and 10.87°
   on another — different held-out sets, both internally correct. `ckpt`
   bytes never changed. Fixed by `data/heldout.py`.

2. **The pipeline is bitwise-deterministic; do not attribute variance to
   "nondeterminism" without proof.** An isolation test
   (`bench/isolate_variance.py`, two separate process launches) showed identical
   latents, identical decoded frames, and identical VGGT output. The variance
   was entirely the moving split of lesson 1, not kernels or VGGT.

3. **Report medians, not means, on a bimodal metric.** With three of sixteen
   scenes in catastrophic collapse, a mean swings several degrees depending on
   which frontier scenes land in a small sample. The per-scene median with an
   explicit frontier bucket is stable and honest.

Consequence: earlier single-run pose means in the project history — including
the r16 "7.86°" baseline and a since-retracted "2.97°, 3× win" — are noisy point
estimates. Directional conclusions (r32 ladder ≥ r16 at matched resolution;
train-at-deploy-resolution; binding) survive because their gaps were large and
internally controlled; the precise magnitudes should be re-measured with the
hardened eval before being cited.

---

## 9. Open items

- ~~Wire `data.heldout.train_scenes()` into `train_view_set.py`~~ — **done**;
  the trainer now trains on the stable 984 and holds out the pinned set.
- ~~Run the hardened eval on r16 `ckpt_040000`~~ — **done** (§3, 2×2 matrix).
- Read the step-count quality cliff with the hardened eval at 30/10/6/4 steps.
- Phase-4 domain/frontier LoRA gate was borderline (n=12); re-gate with the
  larger pinned held-out set.
- ~~Train the memory-evidence branch~~ — **done** (§5.3, §5.6); trained jointly
  with correspondence for 8k steps on VGGT depth. Median trans-dir 35°→12°,
  binding preserved. `fd434230` regression resolved by the VGGT depth swap.
- ~~Full ON/OFF gate + binding re-check on the memory checkpoint~~ — **done**
  (§5.6); binding is untouched with both branches active.
- Region-masked known-region PSNR on interpolation targets, for a clean
  FrameCrafter comparison (§5.2).
- **Correspondence is mixed, not a clean win** (§5.6). Next: distinguish whether
  the ghosting is one-sided injection or frozen backbone — see §10.

---

## 10. Next architectural fork — injection shape vs backbone capacity

The correspondence result (§5.6) leaves one question that decides the next months
of work: is the ghosting caused by the **one-sided** coordinate injection, or by
the **frozen backbone**? The two hypotheses have separable, individually-affordable
experiments, laid out in full in `PLAN_correspondence_finetune.md`:

1. **Two-sided correspondence** — add canonical coordinates to *context* tokens too,
   restoring Lyra's `pₜᵀ W p_c` matching term. Cheap, keeps the zero-init invariant,
   fits the existing 32 GB card. If this alone kills the ghosting, the cause was
   injection shape and the backbone need never be touched.
2. **Attention-only fine-tune** — unfreeze `q,k,v,o` (~1.1 B params) with 8-bit
   Adam (~26 GB), targeting exactly the routing mechanism the artifacts live in.
   If this fixes it where (1) did not, the cause was the frozen attention stack.
3. **Full 5 B fine-tune** — the ABot regime; ~80 GB optimizer states, feasible on
   1× A100 80 GB with 8-bit Adam (≈60–65 GB). Only if (1) and (2) are
   insufficient, and paired with a held-out
   visual-fidelity tripwire because 984 scenes is small for full-FT of a 5 B model —
   the new risk is prior-forgetting, not VRAM.

These are diagnoses, not just fixes: running (1) before the fine-tunes tells us
whether we can avoid the fine-tuning compute and the forgetting risk entirely.

**Step 0 — re-run on the fixed depth first.** The §5.5 scale bug means §5.6's
correspondence result ran on ~10 dB warps, not 15.75 dB. Before attributing the
ghosting to injection shape (H1) or backbone capacity (H2), retrain
memory+correspondence on the fixed `depth_vggt` and re-gate — bad-warp is a third,
cheaper-to-eliminate candidate cause, and it must be ruled out first. No
architecture change; just a fresh training run with the corrected depth.

**Resolution so far (chronological):** Step 0 done (§5.8) — the fixed depth resolved
the confounded ghosting and made the result CI-significant; so the confounded ghosting
was largely the scale bug (H3), not H1/H2. Step 1 done (§5.9) — two-sided is *negative*,
one-sided is better, so injection shape is not the remaining lever. What actually
remains after the depth fix is (a) **frontier pose ambiguity / forward-extrapolation**
(the sequential OFF failures, §5.10) → addressed by **temporal RoPE** (§11), and (b)
**the frozen adapter faithfully copying broken warp evidence** (§5.11) → addressed by
**trust-gating (P2)** or, failing that, backbone fine-tune (Steps 2–3). The fine-tune
ladder is therefore *deferred*, not abandoned — temporal RoPE and trust-gating are the
cheaper, frozen-compatible moves that come first.

---

## 11. Temporal RoPE — the trajectory-order fix (in progress)

> **Historical pilot; superseded by §26.** This section describes the old r32/adapter-only
> run. Its “step 0 is the checkpoint” claim refers only to the `None` code path and is
> false once nonzero temporal positions are supplied; §26.1 measures the resulting ~4 dB
> zero-shot loss. The current run is full-Q/K and uses the causal sampler.

The rollout (§5.11) and sequential (§5.10) results converge on one gap: the model
cannot *extrapolate the forward trajectory* because it has no order signal. The
architecture inherited FrameCrafter's **zero-temporal RoPE** — context is an unordered
set, appropriate for a view-set task but not for a camera *trajectory*.

Fix (`models/wan22_frame_renderer.py`): `model_fn` gained `frame_temporal_pos` — one
integer temporal position per frame, indexing the DiT's own temporal RoPE table
(`dit.freqs[0]`). Positions are the real frame indices made relative to the tuple
minimum, so both *order* and *spacing* are preserved (ctx `[0,3,6]` + target 18 →
`[0,3,6,18]`). `frame_temporal_pos=None` reproduces the zero-temporal model *exactly*,
so every existing checkpoint stays valid; a `<0` sentinel keeps a frame on identity
(hybrid: ordered recent window + unordered/retrieved set-context). Tests
(`test_temporal_rope.py`) pin: `None` == zero-temporal; temporal positions change the
output; and — crucially — temporal ordering *breaks* the permutation-invariance the
zero-temporal model has (the permutation gate now flips: order must matter).

Training (`training/train_temporal.py`): fine-tune the camera trainables (LoRA + ray +
null + FiLM) from the r32 base with temporal RoPE ON, on *sequential-biased* tuples
(clustered ordered context + a mostly-forward target with varied offset, so the model
learns to use order as a prior while still reading the rays for the exact pose —
preserving binding). Since `None` == the base model, step 0 *is* the zero-temporal
checkpoint; temporal use ramps in. In progress (`outputs/temporal/`, 15k steps).

**To verify at completion:** re-run the exact rollout (§5.11) and sequential gate
(§5.10) — the 17.5° sequential OFF and the 143° rollout step-1 are the baselines to
beat — and re-run `bench/camera_binding.py` with a new `trend_break` arm (ordered
context trending one way, target commanded to violate it) to confirm the model still
obeys the commanded camera rather than shortcutting to trend-extrapolation.

---

## 12. Why the other world models don't have our problems (design-space)

A recurring question: Lyra, GEN3C, ABot-World, SANA-WM don't show the "faithfully
copy the broken warp" failure (§5.11) or the rollout pose-imprecision. Why? Because we
occupy a design-space corner none of them do — **explicit geometric warping AND a
frozen backbone**:

| method | warps geometry? | backbone | has our bad-warp failure? |
|---|---|---|---|
| GEN3C | yes (3D cache) | full fine-tune | no — trained backbone corrects the warp |
| Lyra 2.0 | yes (warp + coords) | full FT (14B) | no — trained backbone corrects the warp |
| ABot-World | no (actions + ref) | full FT (5B) | no — no warp to be wrong |
| SANA-WM | no (control) ⚠ | full FT | no — no warp to be wrong |
| **GaussVerse** | **yes** | **frozen + ~3M adapter** | **yes** |

Two of them (ABot, SANA-WM) don't warp at all — they learn forward dynamics from
actions/control, so the "bad warp" class cannot occur (cost: no metric/exact camera
control from geometry). The two that warp (GEN3C, Lyra) *fine-tune the whole backbone*
end-to-end on (imperfect-warp → clean-GT) pairs, so it internalises "the warp is an
approximate hint, correct it" — and they inject at full resolution through every layer,
not one compressed residual.

**Mechanistic crux — why our frozen adapter can't fix it.** The frozen backbone renders
whatever the tokens say; the adapter can only *perturb the tokens*, not change how the
backbone renders. To fix a broken car the model must recognise it's broken and
re-synthesise it — needing the backbone's generative machinery re-pointed, which only
unfreezing provides. What the frozen adapter *can* learn is to inject *nothing* where
the warp is bad (**distrust** → fall back to the generative prior), which it can't
currently do because it has no per-pixel trust mechanism. **Distrust is achievable
frozen; correction needs unfreezing.** That single fact is why **trust-gating (P2)** is
the right frozen-compatible next step, and full fine-tune the heavier alternative. Our
problems are the *cost* of two deliberate choices — geometric warping (for exact camera
control) + frozen backbone (for safety, cheapness, zero-init preservation) — not a flaw
in the idea. (Confident on GEN3C/Lyra/ABot; SANA-WM's exact conditioning ⚠ worth
verifying against its paper.)

---

## 13. Re-measurement under trustworthy metrics — the current results

Everything in this section uses **epipolar Sampson** for pose and **region-split
PSNR** (by the warp's validity mask) for fidelity, on the pinned 16-scene held-out
set at `384×640`, seed 7. It supersedes §5.3, §5.6 and §5.9 wherever they conflict.
The render path is deterministic: the frozen-base `OFF` arm reproduced bit-identical
PSNR (11.869 dB) and Sampson values across three independent eval processes, so these
runs are directly comparable.

### 13.0 Scoreboard

Every intervention measured under the current metrics, most decisive first. `*` = CI95
excludes zero. Camera effects are epipolar Sampson (negative = better); fidelity effects
are PSNR by region.

| # | intervention | type | camera (Sampson) | fidelity (PSNR) | verdict |
|---|---|---|---|---|---|
| 13.3 | **backbone Q/K unfrozen** vs matched control | addition | **median −0.0102 \*** (28/39, p=.005) | **+0.282 \*** | **works: pose AND fidelity** |
| 13.2 | CameraFiLM removed | removal | +0.0135 \* | −1.174 \* | **essential** |
| 13.2 | CameraFiLM given wrong-scene rays | removal | +0.0268 \* | −1.515 \* | worse than removal |
| 13.2 | CameraFiLM β only (γ:=0) | removal | +0.0015 n.s. | −0.073 n.s. | β carries it |
| 13.2 | CameraFiLM γ only (β:=0) | removal | +0.0097 n.s. | −1.075 \* | γ carries ~nothing |
| 13.8 | input-level ray planes given wrong rays | removal | +0.0051 n.s. | −0.341 \* | no pose role; small fidelity role |
| 13.4 | two-sided vs one-sided correspondence | addition | — | frontier **+0.198 \*** | **adopt** |
| 13.4 | memory (warped RGB) | addition | +0.0001 n.s. | covered +0.796 \* | fidelity only |
| 13.3 | Q/K-bias adapter v1 / v2 / rays-only | addition | −0.0063 / −0.0059 / +0.0009, all n.s. | — | null ×3 |
| 13.3 | own-basis camera attention | addition | +0.0062 n.s. | +0.249 \* | null (5th) |
| 13.3 | frozen-backbone control (extra training only) | addition | +0.0030 n.s. | +0.168 n.s. | null |
| 13.5 | warp evidence v2 | addition | — | frontier −0.026 n.s. | regression |
| 13.8 | adapter retrained WITHOUT ray planes | retrain | +0.0109 n.s. | −0.216 n.s. (11/16 neg) | underpowered; keep |

Two things this table is arranged to keep visible. **Removal and addition rows are not
comparable** — a knockout measures how much the trained model depends on a branch, an
addition measures marginal value. And **five of the twelve rows are nulls**, which is the
actual shape of the result: camera geometry could not be injected into a frozen backbone
by any means tried.

### 13.1 The headline pose result did not survive

`OFF 35.4° → 10.0°` was a 2-view VGGT artifact. Under Sampson, median
`OFF 0.0575 / MEM 0.0601 / BOTH 0.0472`; **nothing is significant** (MEM−OFF
Δ +0.0001, 6/14 wins — a coin flip). The memory branch gives **no camera-pose
benefit**.

### 13.2 CameraFiLM is causally essential — and it is geometric, not a stabilizer

Knockout (`--film-knockout`, zeroing the gates at eval; `tanh(0)=0` makes the update
exact identity, so this is surgical and needs no retraining):

| arm | PSNR | ΔPSNR vs `on` | Sampson | ΔSampson vs `on` |
|---|---|---|---|---|
| **on** | 11.869 | — | 0.0571 | — |
| off | 10.696 | −1.174 [−1.805, −0.614] * | 0.0677 | +0.01352 [+0.00242, +0.02375] * |
| **shuffled** | 10.354 | **−1.515** [−2.332, −0.743] * | 0.0736 | **+0.02676** [+0.01006, +0.04301] * |
| constant | 10.780 | −1.089 [−1.742, −0.494] * | 0.0753 | +0.01640 [+0.00192, +0.03069] * |
| gamma_only | 10.795 | −1.075 [−1.625, −0.575] * | 0.0732 | +0.00968 [−0.00356, +0.02256] |
| beta_only | 11.796 | −0.073 [−0.174, +0.030] | 0.0598 | +0.00154 [−0.00491, +0.00829] |

`shuffled` = rays from a *different scene* (valid statistics, wrong geometry);
`constant` = the target's own ray plane replicated to every frame (well-formed raymap,
relative pose destroyed); `gamma_only` = β forced to 0; `beta_only` = γ forced to 0.

Two conclusions:

1. **Geometric, not a stabilizer.** Wrong-but-valid rays are *worse than removing the
   branch entirely*. A branch that merely stabilized activations would be indifferent
   to which well-formed input it received. Ordering: correct > none ≈ uninformative >
   wrong.
2. **The mechanism is additive β, not multiplicative γ.** β alone is statistically
   indistinguishable from full FiLM; γ alone is as bad as deleting it. In
   `Δy ≈ J_l[γ⊙h + β]` the `γ⊙h` term does essentially nothing. "Camera-conditioned
   gain scheduling" is the wrong description.

### 13.3 Attention-level camera conditioning: five nulls, then one win

`CameraQKAdapter` biases `x` before the backbone's **frozen** W_Q/W_K. Δ vs OFF on
Sampson (negative = better):

| run | steps / scenes | ΔSampson | |
|---|---|---|---|
| v1 | 1 k / 984 | −0.0063 [−0.0123, +0.0001] | n.s. |
| v2 | 8 k / 2984 | −0.0059 [−0.0135, +0.0017] | n.s. |
| rays-only | 4 k / 2984 | +0.0009 [−0.0104, +0.0111] | n.s. |
| in frozen control | 4 k / 2984 | +0.00184 | n.s. |

The **rays-only** run tested the "signal purity" hypothesis (that the branch failed
because it saw the full 1586-ch cond while FiLM sees rays only). Restricting it to the
bare 1536 Plücker planes drove the effect to *zero* rather than sharpening it —
**hypothesis refuted.** That run also had the *lowest val_loss of any run* (0.1837)
and no pose effect whatsoever.

**Unfreezing the backbone's own `self_attn.q/k` is what works.** Against a matched
`--mode none` control (identical steps, data, conditioning updates, camqk co-training
and EMA; base frames bitwise identical across both A/Bs), paired per scene:

| metric | Q/K − control | |
|---|---|---|
| **Sampson** | **−0.01022** [−0.01897, −0.00238] | *significant*, 11/14 wins |
| PSNR | +0.147 [−0.051, +0.341] | n.s. |

The control produced nothing on pose (+0.00303, wrong direction, n.s.). Median Sampson
0.0560 → 0.0375, a 33 % reduction. Adding camqk on top of unfrozen Q/K changes nothing
(≤0.03 dB on every scene).

**Own-basis camera attention (2026-07-27) — the fifth null, and it refutes the
"own machinery" explanation.** `models/camera_attention.py`: a NEW attention module with
its own `W_Q^c`/`W_K^c`/`W_V^c` trained from scratch at blocks 7/15/23, camera term
applied inside the branch, 4000 steps on the same frozen base and data as the rays-only
run. Result: **Sampson +0.0062 CI95 [−0.0047, +0.0192] n.s., 7/15 wins** — point estimate
in the wrong direction. PSNR +0.249 CI95 [+0.029, +0.522] (significant), consistent with a
new attention op re-mixing existing content. Gates ended 0.81/0.76/0.75, i.e. the model
kept the branch OPEN and still got no camera benefit from it.
This was built and predicted to succeed on the reasoning that the four prior nulls failed
because they biased a *converged* basis rather than owning one. **That reasoning is wrong.**
Basis ownership is not the distinguishing variable.

(Implementation note: the first attempt diverged to NaN at step ~500 because the branch
omitted the QK normalization every Wan `SelfAttention` block applies — `norm_q`/`norm_k`.
q and k were unbounded sums of a projection of x and of a norm-free camera stem, and
`clip_grad_norm_` propagates NaN rather than sanitizing it, so one bad step poisoned the
weights permanently. Fixed with RMSNorm on q/k plus a non-finite-gradient skip guard;
`test_camera_attention.py` pins both.)

**Correct claim: unfreezing backbone Q/K buys camera control specifically, at no
fidelity cost.** The PSNR half was confounded with the extra 4000 steps on a larger
data pool and must not be claimed as a fidelity win.

### 13.4 Memory is a fidelity mechanism; correspondence is the frontier mechanism

Region split at matched step 4000, Δ vs OFF:

| arm | covered | frontier | whole |
|---|---|---|---|
| MEM (one-sided) | +0.796 * | +0.165 * | +0.526 * |
| BOTH (one-sided) | +0.863 * | +0.258 * | +0.600 * |
| BOTH − MEM | +0.067 * | **+0.094 *** | +0.074 * |
| **BOTH (two-sided)** | +0.776 * | **+0.457 *** | +0.620 * |

Paired two-sided − one-sided: **frontier +0.198 [+0.070, +0.350] ***, covered −0.087
(n.s.), whole +0.021 (n.s.). Two-sided nearly doubles the disocclusion gain at no
significant cost — **make it the default.**

Note the *shape*: the correspondence branch contributes more on the frontier (+0.094)
than in covered regions (+0.067), the opposite profile to the RGB warp. Geometry helps
where there is no appearance to copy. (Caveat: the paired two-sided − one-sided
difference in the branch's *marginal* contribution is +0.042, **n.s.** — the
configuration wins; isolating the mechanism is underpowered at n=16.)

Memory is **evidence-bound**: MEM output lands at the warp's own quality (mean
MEM − warp +0.06 dB), captures ~76 % of available headroom, and correlates with
coverage (+0.59). The lever is better depth and coverage, not more steps.

### 13.5 Warp evidence v2 — a regression, do not adopt

v2 = 55 planes (supporting-view count, cross-view RGB/depth disagreement, view-angle
quality) + soft `conf·visibility·angle·z` fusion + 15 % warp dropout. Matched to v1 at
step 4000 (same frozen base, same data, same steps):

| region | v1 | v2 |
|---|---|---|
| covered | +0.796 * | +0.765 * |
| **frontier** | **+0.165 *** | **−0.026 n.s.** |
| whole | +0.526 * | +0.370 * |

Covered is unchanged; the small significant frontier gain is **destroyed**. Prime
suspect is the warp dropout (the frontier is exactly where the model must rely on the
branch's learned integration rather than copyable pixels). Not decomposed — the
isolating arm is `--evidence-v2 --fusion soft --mem-dropout 0`.

⚠️ "15 % warp dropout" here means 15 % *content*-evidence dropout with correspondence
still supplied on every step — `--mem-dropout` now drops both branches, so this arm is
not reproducible by rerunning the same command.

### 13.6 Frontier: what moved disocclusion and what did not

Same base, same step, on the region the model must invent:

| intervention | frontier |
|---|---|
| warp evidence v2 (richer evidence channels) | −0.026 n.s. |
| one-sided correspondence | +0.258 * |
| **two-sided correspondence** | **+0.457 *** |

**Richer evidence channels did nothing for disocclusion; richer correspondence
geometry nearly doubled it.**

An earlier version of this section claimed a unifying pattern across §13.3 and §13.6 —
"richer inputs to frozen machinery fail; giving a mechanism its own machinery works" —
and used it to predict that an own-basis camera attention branch would succeed. **That
branch was built and is null (§13.3), so the unifying claim is withdrawn.** Report the two
findings separately:

- **Camera control**: five frozen-backbone interventions null; the only success unfroze
  the backbone's own `W_Q`/`W_K`. The operative variable is *whose* attention changes,
  not whether the branch owns projections.
- **Frontier fidelity**: two-sided correspondence nearly doubles the disocclusion gain.
  This is a change of *signal* (context frames carry their own canonical coordinates),
  not of machinery, and it is measured on region-split PSNR rather than camera pose. It
  is not an instance of the same pattern.

### 13.7 Methodology lessons added by this round

- **Gate magnitude, weight norm, and output magnitude do not predict causal
  importance.** FiLM's gates collapse to 0.001–0.063 and it is the strongest camera
  path in the model. Only intervention tells you.
- **Knockout-Δ and addition-Δ are not symmetric experiments.** Removing a branch
  measures co-adaptation dependency; adding one measures marginal value. Do not
  compare them numerically (an earlier "FiLM ≈ 2.3× UCPE" claim is retracted).
- **val_loss is decoupled from every metric that matters here** — demonstrated three
  times. Never early-stop on it or on gate values.
- **Frame caches must be keyed by run *and* step.** `memory_vggtfix/ckpt_004000` and
  `memory_v2/ckpt_004000` collided in one directory; unattributable cached frames
  produced a wrong "no frontier effect" reading that re-rendering reversed.
- **`load_ckpt` tolerates key mismatches by design**, so a wrongly-shaped adapter
  silently scores an *untrained* branch as trained. Eval must read the variant off the
  checkpoint and assert.
- **Copy the surrounding code's numerical conventions.** The own-basis camera attention
  branch diverged to NaN at step ~500 because it omitted the QK normalization that every
  Wan `SelfAttention` block applies (`norm_q`/`norm_k`). q and k were unbounded sums of a
  projection of x and of a norm-free camera stem. Two compounding factors made it fatal
  rather than noisy: `clip_grad_norm_` *propagates* non-finite gradients instead of
  sanitizing them, so one bad step poisoned the weights permanently; and NaNs made the run
  *faster* (2.2 → 3.3 samp/s after the fix, since RMSNorm adds work), so throughput looked
  fine. Guard added: skip any step whose gradient norm is non-finite.
- **Power is now the binding constraint.** At n=15–16 the CI half-width is roughly the
  effect size for most branches. The 480P **4K subset has never been trained on** and
  is the clean pool for expanding to n≈60.

### 13.8 Three camera paths: which one is the model actually reading? (2026-07-27)

`bench/camera_path_matrix.py` crosses the two TRAINED camera paths by swapping in Plücker
rays from a different scene — valid statistics, wrong geometry — for one path at a time.
(Zeroing is not available: the input adapter's tensor is `[48 ctx latents | 1 mask | 1536
rays | 1 role]`, so zeroing the branch deletes the context images, and zeroing only the ray
planes is out-of-distribution for a norm-free conv stem — see §2.)

| arm | PSNR | ΔPSNR vs both | Sampson | ΔSampson vs both |
|---|---|---|---|---|
| both correct | 11.869 | — | 0.0571 | — |
| ray adapter lied to | 11.528 | −0.341 [−0.593, −0.096] * | 0.0711 | +0.00506 **n.s.** |
| FiLM lied to | 10.354 | −1.515 [−2.332, −0.734] * | 0.0736 | +0.02676 * |
| both lied to | 10.262 | −1.607 [−2.361, −0.857] * | 0.0741 | +0.02435 * |

Marginal contribution, given the other path is already lied to:

```
ray adapter   +0.0921 dB [-0.1648,+0.3083]  n.s.   Sampson +0.0024 n.s.
CameraFiLM    +1.2662 dB [+0.5151,+2.0676]  *      Sampson -0.0215 *
```

**The input-level ray path carries no measurable CAMERA-POSE signal given FiLM.** (On fidelity it does carry a small significant effect — see 13.8b, which walks back the stronger 'redundant' reading.) FiLM-alone recovers
near-full performance and is statistically indistinguishable from both-correct on POSE
(+0.005, n.s.); ray-adapter-alone recovers essentially nothing.

**Unplanned cross-validation.** The `FiLM lied to` arm is by construction the same
intervention as `shuffled` in the FiLM matrix (§13.2), computed by an independent code path
(`FiLMProxy` vs `film_rays_override`) in a different bench: **−1.515 dB [−2.332, −0.743]**
vs **−1.515 dB [−2.332, −0.734]**. Identical to three decimals.

**Actionable, with a caveat.** The ray adapter also carries the 48 context-latent planes, so
the BRANCH is load-bearing — only its 1536 Plücker planes are redundant. The clean test is a
retrain with the adapter input cut 1586 → 50 channels, camera carried by FiLM alone.

**Harness bug worth recording.** The first smoke run printed four identical arms. Cause:
`--limit 1` truncated the scene list, so `donor = scenes[(i+1) % len(scenes)]` selected the
scene itself and the "wrong" rays were the correct rays. Read as a result that is
"all camera paths are redundant, the model ignores camera conditioning" — dramatic and
entirely false. Donors now come from the full list regardless of `--limit`, with assertions
that fail loudly if a donor equals its scene or the ray tensors match. **An intervention
that silently does nothing looks exactly like one that provably does nothing.**

### 13.8b Retraining without the ray planes — underpowered, and the sign says keep them

§13.8 was a knockout. This is the retrain it demanded: two matched arms, both freezing
LoRA/FiLM/null from the same base and training a FRESH zero-init adapter for 4000 steps,
differing only in input width (50 ch / 2.23 M vs 1586 ch / 4.00 M). Freezing FiLM held the
other camera path identical, so the only variable is the adapter's input.

```
ΔPSNR   (rayless − full)  −0.216  CI95 [−0.481, +0.013]   n.s.   rayless wins  5/16
ΔSampson(rayless − full)  +0.0109 CI95 [−0.0057, +0.0341] n.s.   rayless wins  7/16
per-scene ΔPSNR: 11 of 16 negative, median −0.178, sd 0.525
```

**Do not read this as "the ray planes are unnecessary."** The bench originally printed that
conclusion and it was wrong — absence of significance is not evidence of absence at the n
where power is the known binding constraint. What the data actually says:

- Both point estimates favour KEEPING the rays, and 11/16 scenes are negative.
- The PSNR CI misses zero by 0.013 dB.
- The Sampson point estimate (+0.0109) is the same magnitude as the entire Q/K-unfreezing
  effect (−0.0102), with a CI roughly four times wider.
- Power: SE 0.131 dB, CI half-width ~0.26 dB at n=16. Resolving a 0.10 dB effect needs
  **n ≈ 106**; even the planned n≈60 expansion would not fully settle it.

**Convergence with the knockout.** Lying to the ray planes cost −0.341 dB (significant);
training without them costs −0.216 dB (n.s.). Same sign, similar magnitude, two independent
experimental designs. The consistent reading across both:

> The input-level ray planes carry a **small fidelity contribution (~0.2–0.35 dB) and no
> measurable camera-pose contribution.** They are not redundant; they are cheap-but-real.

**Recommendation: keep them.** Dropping them saves 1.77 M params on a 5 B model — a rounding
error — for a probable small fidelity cost. The earlier "redundant" framing in this document
overstated the fidelity side; it was correct only about pose.

### 13.9 Open

- **Rayless adapter retrain (in flight).** §13.8 is a knockout — it shows the trained model
  does not read this branch's ray planes. Whether a model *trained without them* is as good
  is a separate question, since the paths co-adapted over 30k steps.
  `training/train_ray_adapter.py` runs the matched pair: both arms freeze LoRA/FiLM/null
  from the same base and train a FRESH zero-init adapter for 4000 steps, differing only in
  input width (50 vs 1586 ch). Freezing FiLM holds the other camera path identical. Compare
  arm-vs-arm only — both start from a fresh adapter, so both underperform the 30k base.
- `--mode attn` tier (q,k,v,o) now that `qk` is proven causal. Note it also lets the camera
  signal reach V, giving up the `qk_bias` safety property that the whole correspondence
  design was built around.
- v2 decomposition — isolate the warp dropout (`--evidence-v2 --fusion soft --mem-dropout 0`).
- **Power.** At n=15–16 the CI half-width is roughly the effect size for most branches. This
  is now the binding constraint, not training. The 480P **4K subset has never been trained
  on** and is the clean pool for expanding to n≈60.
- Never measured in isolation: the input-level adapter's *context-latent* planes. §13.8 only
  intervened on its rays. The branch may be load-bearing for reasons unrelated to camera.

### 13.10 Held-out expansion to n=60 — and why more scenes is not automatically more power

Power, not ideas, had become the binding constraint: at n=16 the PSNR CI half-width was
±0.26 dB and the Sampson half-width ±0.01, the size of most real effects here. The
held-out set was expanded 16 → 60 with scenes from DL3DV-480P/4K, verified disjoint from
all 3000 scenes ever trained on. (That check mattered: 960P/1K and 480P/1K are the SAME
1000 scenes at different resolution, so treating the four roots as distinct would have
leaked training data into the eval set.) The expansion is **opt-in** — the default is
still the original 16, which are a **prefix** of the 60, so every recorded number stays
comparable and per-scene comparisons are unchanged.

**Result 1 — the rayless adapter question resolved.**

| n | ΔPSNR (rayless − full) | |
|---|---|---|
| 16 | −0.216 [−0.481, **+0.013**] | n.s. |
| **60** | **−0.165 [−0.330, −0.006]** | **SIGNIFICANT** |

Dropping the 1536 Plücker planes from the input-level branch costs a small but **real**
fidelity loss. §13.8b's hedge ("the sign says keep them") is now a measurement. Keep them.

**Result 2 — the expansion helps PSNR and HURTS Sampson.** This was not anticipated.

```
metric    sd(original)  sd(expansion)  ratio   n=16 -> n=60
PSNR         0.525         0.685        1.3x   SE 0.131 -> 0.083   power GAINED
Sampson      0.0129        0.0349       2.7x   CI got WIDER        power LOST
```

camqk v1 at n=60: Δ −0.00505 [−0.01338, +0.00374], n.s. — the CI is *wider* than the
n=16 result (−0.00632 [−0.01233, +0.00013]) despite ~3× the scenes.

Cause: the expansion is 480P source, the originals 960P. Softer frames give fewer and
worse SIFT matches — **36% of expansion scenes abstain vs 6% of originals** — so each
epipolar estimate is noisier. PSNR degrades gracefully with source sharpness; SIFT-based
epipolar geometry does not.

**The error in the plan, stated plainly.** The expansion was proposed on the argument that
paired deltas cancel the source-resolution difference. That is true of *bias*, and it held
exactly — the original-16 subset reproduces its recorded numbers to five decimals inside
the 60. But bias-cancellation says nothing about *variance*, and variance is what power
depends on. **Adding scenes is not automatically more power; check sd on the new subset.**

**Protocol rule now encoded in `bench/evalkit.py`:**
- fidelity questions (PSNR, region splits) → `expanded=True`, n=60
- pose questions (Sampson) → the original 16, until the expansion scenes exist at 960P

Resolving camqk v1 therefore needs the same 44 scenes at 960P source, not more scenes.

### 13.11 Q/K unfreezing at n=60 — the claim revises in BOTH directions

The n=60 control was run so the headline could be stated as a paired comparison rather
than against base (which is confounded with the extra training). Base frames are identical
across the two runs (max|diff| 0.000000), so the pairing is exact.

**Paired Q/K − control, by subset:**

| metric | original 16 | expansion 44 | all 60 |
|---|---|---|---|
| PSNR | +0.147 n.s. | **+0.332 \*** | **+0.282 [+0.134, +0.458] \*** |
| Sampson | **−0.01022 \*** (11/14) | −0.00450 n.s. (17/25) | −0.00655 n.s. (28/39) |

**Fidelity — the earlier claim was too weak.** §13.3 said unfreezing buys camera control
"at no fidelity cost". At n=60 the fidelity gain is **+0.282 dB and significant**. The n=16
null (+0.147) was underpowered and was read as absence of effect — the same error this
document warns about in §13.7. It is a fidelity **gain**.

**Pose — the mean-based test is the wrong one here.**

```
bootstrap on the MEAN  : -0.00655 [-0.01724,+0.00470]   n.s.
bootstrap on the MEDIAN: -0.01016 [-0.01492,-0.00198]   SIGNIFICANT
sign test              : 28/39 wins, p = 0.0047
Wilcoxon signed-rank   : p = 0.0411
```

The median at n=60 (−0.01016) is almost exactly the n=16 mean (−0.01022): the typical scene
shows the same effect, and the mean is dragged by heavy-tailed 480P scenes (sd 0.0429 on the
expansion vs 0.0165 on the originals).

**On the test switch.** Choosing the median after seeing a null mean is how people fool
themselves. It is defensible here only because the heavy tail and its mechanism (SIFT
abstention on soft frames) were documented in §13.10 **before** this comparison ran. Report
mean, median and sign count together — `evalkit.robust_summary()` now returns all three so
the choice cannot be made silently after the fact.

**Revised claim:** unfreezing backbone Q/K improves **both** camera control (median −0.0102,
sign test p = 0.005) **and** fidelity (+0.282 dB), against a matched control.

**`--mode attn` OOM'd on the first attempt** at 30.94/31.35 GiB — 1.16 B trainable in fp32
plus two fp32 AdamW states plus a bf16 EMA shadow does not fit beside the frozen backbone.
Retried with `--optimizer adam8bit --no-ema`. That makes the optimizer differ from the `qk`
run, which must be stated whenever the two tiers are compared.

---

## 14. Training budgets of the reference systems (checked 2026-07-29)

Read from the papers, not from memory. Where a paper does not report a number it is
marked NOT REPORTED rather than estimated — a plausible-looking invented budget in a
comparison table is worse than an admitted gap.

| system | backbone | compute | data | steps / iterations |
|---|---|---|---|---|
| **SANA-WM** | 2.6 B | **64× H100, ~15 days** (+3.5 d VAE adaptation) | 212,975 clips | 30k + 30k + 31k + 10k over 4 stages; lr 5e-5 → 1e-5; global batch 32 |
| **Lyra 2.0** | Wan2.1-**14 B** | **64× GB200**, batch 64, bf16 | DL3DV-10K, 1000 frames/video | **7,000 iterations**, lr 3e-5, 832×480 |
| **FrameCrafter** | Wan2.1-**14 B** | **8× A6000 48 GB** (192×336, ZeRO-2) / **8× H100 80 GB** (480×832) | DL3DV-10K | epochs, not steps; lr 1e-4 / 5e-5 |
| **ABot-World-0** | Wan2.2-**5B** (our backbone) | **NOT REPORTED** | NOT REPORTED | NOT REPORTED |
| **GEN3C** | Cosmos 7 B | NOT REPORTED in abstract | — | — |

**ABot-World-0 is the awkward one**: it is the only published system on our exact
backbone, and it reports no training budget anywhere in the abstract or full HTML. It
describes three stages (teacher forcing → ODE distillation → LongForcing) with no
iteration counts, GPU count, learning rates or dataset size. Every hardware figure in
that paper is DEPLOYMENT (1× RTX 5090, 16 FPS, ~19 GiB peak).

### 14.1 What this says about our own budget

| | GPUs | samples seen |
|---|---|---|
| Lyra 2.0 (also DL3DV) | 64× GB200 | 7,000 × batch 64 ≈ **448,000** |
| gaussverse stage-1 `qk` @4k | **1** | 4,000 × 4 accum = **16,000** (5.4 passes over 2984 scenes) |
| gaussverse stage-1 `qk` @12k | 1 | 48,000 (16.1 passes) |
| gaussverse camera base r32 | 1 | 120,000 (122 passes over 984 scenes) |

Lyra 2.0 is the closest comparison — same dataset family, same class of task. Our
stage-1 backbone run at 4000 steps had seen roughly **28× fewer samples**, on 1 GPU
instead of 64.

**This is why "4000 steps plateaued" was the wrong call for the 0.5 B tier.** The
saturation prior in this document is real but belongs to the 2–4 M ADAPTERS, where flat
loss + declining gates + a measured 4k-vs-8k regression all agree. Carrying it across to
a tier 150× larger, at 5.4 passes over the data, was not warranted; the split-half test
on stage 1 (Welch p=0.036) says the loss was still falling. Extended to 12k.

**A structural observation worth carrying into any future compute request.** Lyra 2.0
reaches its result in only **7,000 iterations** — fewer optimizer steps than our own
extended run. Their budget is in the BATCH (64 across 64 GPUs), not the step count. On a
single card, more steps is the only lever available, but it is the weaker one: if
multi-GPU access appears, raising effective batch size is the better buy.

Sources: arXiv 2607.19191 (ABot-World-0), 2605.15178 (SANA-WM), 2604.13036 (Lyra 2.0),
2503.03751 (GEN3C); FrameCrafter from the local checkout's scripts and README.

---

## 15. How the reference systems evaluate (read from the papers, 2026-07-29)

Checked because our own metric audit (S13) invalidated a headline result, and the obvious
question is whether the systems we compare against would survive the same audit.

### 15.1 What each one measures

**Lyra 2.0** (DL3DV-Evaluation in-domain, Tanks & Temples out-of-domain)
- SSIM / LPIPS / FID of generated frames vs ground truth.
- **WorldScore** suite: subjective quality, style consistency ("visual drifting between
  the first and last frames"), and **camera controllability**.
- **Reprojection error** "computed by estimating per-frame depth with an off-the-shelf
  SLAM system [ViPE], to verify 3D consistency of the generated videos."
- 3D stage: **LPIPS-P** (rendered novel views vs the *generated* video frames -- isolates
  the video model's 3D consistency) and **LPIPS-G** (rendered vs *ground truth* --
  reconstruction quality). This P/G split is a genuinely good idea.
- Camera accuracy rides on WorldScore. Traced to the WorldScore paper (arXiv 2504.00983),
  "Camera Ctrl." is:
  - poses recovered from the generated video with **DROID-SLAM**;
  - `e_theta = arccos((tr(R_gt R^T) - 1)/2) * 180/pi`  (degrees);
  - `e_t = || t_gt - s*t ||_2`, where **s is a least-squares SCALE only** -- not a full
    Umeyama Sim(3) fit as SANA-WM uses, so trajectories are assumed pre-relativised to a
    common first frame;
  - `e_camera = sqrt(e_theta * e_t)`, then linearly normalised to 0-100 against empirical
    bounds, with a fixed-camera sequence as the low anchor.

**SANA-WM** (80 generated start images x 4 categories, Simple/Hard trajectory splits,
960 frames @16fps = 60 s per scene)
- Pose recovered from the generated video with **Pi3X**, ground-truth trajectory
  relativized to frame 0, then **Umeyama Sim(3) alignment**, then:
  - `RotErr = (1/T) sum (180/pi) acos(clip((tr(R^T R~)-1)/2, -1, 1))`
  - `TransErr = (1/T) sum ||t - t~||_2`
  - `CamMC = (1/T) sum ||P - P~||_F`  (full pose, rotation+translation, Frobenius)
- **VBench** 8 dimensions + aggregate; FVD in ablations.
- **Revisit memory**: find generated frame pairs at near-identical viewpoints
  (centres within 0.5 m, viewing angle < 20 deg), take the 5 best-scoring pairs, and
  compare the two GENERATED frames to each other with PSNR/SSIM/LPIPS. Measures loop
  closure without needing ground truth.
- **Delta-IQ**: VBench imaging quality of the first 10 s window minus the last 10 s
  window, over a 60 s rollout. A one-number drift metric.

**ABot-World-0** (WorldRoamBench)
- Strict Accuracy, Partial Accuracy, Trajectory Score, Aesthetic, Imaging, Mechanics,
  Memory -- benchmark-defined scores; the paper does not give their computation.
- LongForcing ablation over 60 s rollouts: **HPSv3** (learned human-preference model),
  high-saturation pixel ratio, perceptual blur, patch-repeat ratio.
- **No geometric camera evaluation anywhere.** No pose-estimation error, no trajectory
  ground-truth comparison. Control accuracy is assessed through benchmark scores and
  image-quality proxies only.

### 15.2 Where our protocol is stronger

**0. A fairness correction.** DROID-SLAM (WorldScore) and Pi3X (SANA-WM) both estimate
poses jointly over a whole generated sequence, so they are structurally comparable to our
*registered VGGT* -- our SECOND-choice metric -- not to the 2-view estimator we rejected.
Their numbers are not built on the tool we found broken. The ranking below is about ours
being estimator-free, not about theirs being naive.

**A real weakness in WorldScore's aggregation, though.** `e_camera = sqrt(e_theta * e_t)`
is ZERO whenever EITHER axis is zero: a model with perfect translation and 90 degrees of
rotation error scores a perfect 0 camera error. The geometric mean cannot distinguish
"good at both" from "perfect at one, catastrophic at the other". Pinned by
`tests/test_longhorizon.py::test_worldscore_geometric_mean_hides_a_pure_rotation_error`.
Report e_theta and e_t alongside it, never the combined score alone.

**1. We are the only one measuring camera accuracy without a learned pose estimator.**
Epipolar Sampson SIFT-matches the generated frame against a known anchor and tests those
correspondences against the **commanded** epipolar geometry. No estimator sits between
the model and the score.

This is not a stylistic preference -- it is the exact failure we already caught. S13.1:
the 2-view VGGT estimator manufactured a large fake improvement, and the project headline
did not survive replacing it. **SANA-WM's pipeline (learned pose estimator + Umeyama
Sim(3) alignment to GT) is structurally our "registered VGGT", which we measured and
DEMOTED to second choice** because it *understates* wrong-pose frames: good context
cameras drag the estimate toward a plausible pose. Their numbers are not wrong, but they
are computed with a tool we found to be the less sensitive of the two we tried.

**2. Nobody reports uncertainty.** All three publish point estimates. We report paired
bootstrap CI95 on every number, plus sign tests and medians where the tail is heavy
(`evalkit.robust_summary`). Several results in this document changed sign of conclusion
between n=16 and n=60 -- without intervals we would have shipped them.

**3. Region-split fidelity is ours.** Splitting PSNR by the warp's validity mask into
covered vs frontier is what separated "memory is a content mechanism" from "memory fixes
pose" (S13.4). No reference system splits by geometric region; they report whole-frame
quality, which averages the two effects together.

### 15.3 What we should steal

- **SANA-WM's revisit-memory protocol.** Pairs of GENERATED frames at near-identical
  viewpoints (< 0.5 m, < 20 deg), scored against each other. It needs no ground truth,
  and it measures exactly the loop-closure property our persistent-memory work is
  ultimately for -- which we currently have **no metric for at all**.
- **Delta-IQ**, or our own equivalent: first-window minus last-window quality over a long
  rollout. One number for drift, cheap to compute, and directly relevant to the
  autoregressive rollout regime.
- **Lyra's LPIPS-P / LPIPS-G split.** Rendered-vs-generated isolates the generator's own
  3D consistency; rendered-vs-GT measures reconstruction. Cleanly separates two things we
  currently conflate.

### 15.4 What we should not adopt

Learned preference models (HPSv3) and aggregate benchmark scores (VBench, WorldScore).
They are unfalsifiable at our scale: a single scalar with no CI, no per-scene breakdown,
and no way to tell a real effect from a 0.26 dB noise band. They are useful for
whole-system comparison against published baselines and useless for the mechanism
questions this project asks.

**Net:** on camera-accuracy rigour we are ahead of all three; on long-horizon and
revisit measurement we have nothing and SANA-WM has a protocol worth copying directly.

Sources: arXiv 2604.13036 (Lyra 2.0), 2605.15178 (SANA-WM), 2607.19191 (ABot-World-0).

## 16. Phase 0 — freezing a reproducible baseline (2026-07-29)

`PLAN_COUNTERFACTUAL_MEMORY.md` §3 requires that every candidate checkpoint carry a
manifest recovered from artifacts, and that any checkpoint whose configuration cannot be
recovered be demoted to an exploratory result. `bench/run_manifest.py` does the recovery.
It reads three channels — AdamW's saved `param_groups`, a tensor-level diff between two
checkpoints of the same run, and samples/s × s/step — and prints an explicit
`unrecovered` list so the manifest states its own limits instead of looking complete by
omission.

### 16.1 The fixed-compute batch experiment — null, and the pose effect went backwards

Hypothesis under test: the loss plateau at 4k steps is a batch-4 gradient-noise floor.
If so, spending the same compute at effective batch 16 should beat batch 4.

Both arms were recovered from artifacts, not from the launch command:

| | `qk_fullft_camqk` | `qk_batch16` |
|---|---|---|
| effective batch (derived) | **4.01 → 4** | **15.98 → 16** |
| steps | 12 000 | 3 000 |
| **samples** | **48 000** | **48 000** |
| lr at last step | 2e-5 | 4e-5 |
| trainable | 604.1 M / 285 tensors | 604.1 M / 285 tensors |
| optimizer | AdamW fp32, EMA on | AdamW fp32, EMA on |

The two arms are exactly matched in compute and in trainable set. The derived batch
figures are independent arithmetic confirmation that "batch16" in the directory name is
what actually ran.

Result at n=60 (PSNR) / n=40 (Sampson, original 960P set per §13.0):

```
batch-16 x 3k  vs  batch-4 x 12k   (same 48k samples)
  PSNR    +0.026 [-0.098,+0.161]                          n.s.
  Sampson mean +0.00604  median +0.00562  wins 17/40  p=0.866   nominally WORSE

batch-16  vs  matched frozen control
  PSNR    +0.407 [+0.226,+0.615]                          *SIG*
  Sampson mean -0.00258  median -0.00397  wins 21/39  p=0.375   n.s.

batch-4 @ 12k  vs  matched frozen control   (for comparison)
  PSNR    +0.381                                          *SIG*
  Sampson median -0.0100  wins 29/41  p=0.012             *SIG*
```

Base frames were bit-identical across arms (max|diff| 0.000000), so the pairing is exact.

**The hypothesis is not supported.** Larger batch at fixed compute did not improve the
task metric, and on pose it was nominally worse. So the plateau is not a batch-4 noise
floor.

Worse, batch-16 **bought fidelity and lost camera control**: its PSNR gain over the
control is the largest we have measured, while its pose effect collapsed from
29/41 (p=0.012) to a coin flip. This is the third independent instance in this project of
an intervention that improves pixels while degrading pose — see §13.9 (`--mode attn`) and
§13.11 (warp evidence v2). Fidelity and camera control are not the same axis and must
never be traded on a single number.

**Confound, stated plainly.** lr was scaled 2e-5 → 4e-5 with the batch. That was a
deliberate choice at launch — an unscaled lr would have handicapped the arm — but with a
negative pose result it is no longer free, and the tripwire's shape (peak 11.00 at step
750, monotone decline to 10.76 at 3000) is what too-high lr late in training looks like.
Batch and lr cannot be separated from this run.

**The fidelity tripwire disagreed with the A/B and should not have been read as a
result.** It showed batch-16 at +0.19 dB over batch-4 with a stable band. It was right
about fidelity, silent about pose, and was narrated in flight as though it settled the
question. Four pure held-out scenes, one seed, no CI: it is a divergence alarm, nothing
more. This is now the second time it has been over-read (§13.5).

### 16.2 The frozen control is NOT sample-matched — a threat to the headline

The manifest diff on `qk_control_frozen` recovered something the directory name hides:

| | trainable | steps | **samples** |
|---|---|---|---|
| `qk_control_frozen` | 25.8 M / 47 tensors | 4 000 | **16 000** |
| `qk_fullft_camqk` | 604.1 M / 285 tensors | 12 000 | **48 000** |
| `qk_batch16` | 604.1 M / 285 tensors | 3 000 | **48 000** |

**The control saw one third of the samples of the arms it is the control for.** Every
"vs matched frozen control" number in §13 — including the project's headline camera
result — compares a 48 000-sample model against a 16 000-sample one. "Matched" has been
doing work it did not earn.

Two facts decide how serious this is, and they point in opposite directions:

- *In defence:* the control is an adapter-tier run, and §13.2 measured 2–4 M adapters
  saturating at 3–4 k steps and degrading afterward. Stopping the control at 4 k was the
  right call **for that tier**.
- *Against:* this control is **25.8 M**, six to ten times larger than the adapters that
  result came from, and §2.6 of the plan explicitly forbids transferring a saturation
  result across a capacity tier. And its own tripwire was **still rising monotonically at
  its final step** — 10.19 → 10.24 → 10.30 → 10.30 → 10.34 → 10.34 → 10.35 → **10.38**,
  with no sign of a knee.

A control that had not stopped improving was stopped. The naive extrapolation is not
reassuring: +0.19 dB of tripwire over 3 500 steps, against a headline gap of +0.38 dB.

**Required:** rerun the frozen control to 12 000 steps at batch 4 (48 000 samples, ~7 h)
and re-derive the headline against it. Until that lands, the headline camera result
should be quoted as *"against a frozen control trained to its tier's expected saturation
point"*, not *"against a matched control"*. This is the correct kind of finding for
Phase 0 to produce, and it was invisible until the configurations were recovered from
artifacts.

### 16.3 What the manifests cannot recover

Present in every manifest's `unrecovered` list, because no artifact in a run directory
records them: seed, scene roots, tuple mixture, resolution, the identity of the base
checkpoint, the git commit at train time, and the shape of the lr schedule before the
final step. The last one is only partly lost — a cosine run that reached its final step
reports an lr near zero, so a constant-lr run is distinguishable from a completed decay;
an *interrupted* decay is not.

`bench/run_manifest.py` should be run at the end of every future training job, and §3.5
of the plan — checkpoint every 500 steps — should be adopted so the diff channel is always
available. `attn_fullft` has a single checkpoint and therefore no recoverable trainable
set; it is exploratory by the plan's own rule.

### 16.4 The checkpoint-selection table, and the §3 gate

`bench/checkpoint_table.py` re-scores every Q/K candidate from cached frames under one
protocol. All five arms already had rendered frames on disk, so this costs minutes and
introduces no seed or sampler drift. Deltas are against `base` = the frozen pre-training
camera checkpoint `view_set_r32` — **not** against `qk_control_frozen`, which is a
separately trained arm and gets its own row. Conflating the two is easy and silently
inflates or deflates every number.

```
arm                                  PSNR    dPSNR n=60             Sampson   dSampson n=15
base (frozen view_set_r32)          11.339                           0.0571
control frozen 4k (16k samples)     11.653  +0.315 [+0.178,+0.461]*  0.0585  -0.00200   8/15
qk batch4 12k  (48k samples)        12.035  +0.696 [+0.412,+0.991]*  0.0414  -0.00484   9/15
qk batch16 2k  (32k samples)        11.993  +0.654 [+0.426,+0.899]*  0.0484  -0.00364   9/15
qk batch16 3k  (48k samples)        12.060  +0.722 [+0.473,+0.983]*  0.0561  -0.00058   8/15
```

**§3 decision gate — batch-16 step 3000 minus step 2000:**

```
  PSNR    +0.068 [-0.006,+0.146]  n.s.   3000 better on 33/60
  Sampson +0.00838               n.s.    3000 better on  6/15
```

The final 1 000 steps — 16 000 samples, a third of the run — bought nothing measurable on
fidelity and moved pose the wrong way. The plan's rule ("if step 3000 does not improve the
task metrics over step 2000, stop or lower the learning rate") fires: **stop, or lower the
LR.** This is independent support for the lr-4e-5 reading of §16.1 rather than the batch
reading, since the damage is concentrated in the late steps where a too-high constant LR
does its worst.

Renders are **not** bit-reproducible across sessions. The three directories rendered on
07-27/28 are bit-identical to each other; the one rendered 07-29 from the same seed and
the same frozen checkpoint differs on 59/60 scenes (mean |diff| 0.40/255, max 118) from
non-deterministic kernel selection compounded over 30 denoising steps. It is worth
**0.00075 dB** of mean base PSNR and 0.018 dB on the worst single scene, so it does not
threaten any result here — but "identical base frames" can no longer be asserted, and
`_check_base_agreement` now guards the mean (bias) and the per-scene max (noise)
separately rather than demanding equal pixels.

### 16.5 The pose headline is subset-dependent — and our own rule picks the null

Recomputing the pose deltas on both held-out subsets:

| arm − control | original 16 (declared primary) | all 60 |
|---|---|---|
| qk batch4 12k | −0.00575 [−0.01298, +0.00097] n.s., 10/15, p=0.302 | **−0.00889 [−0.01650, −0.00134] \*SIG\*, 29/41, p=0.012** |
| qk batch16 3k | −0.00361 n.s., 8/15, p=1.000 | −0.00250 n.s., 21/39, p=0.749 |
| qk batch16 2k | −0.00675 n.s., 10/15, p=0.302 | −0.00678 n.s., 25/40, p=0.154 |

`evalkit.USE_ORIGINAL_FOR = ("sampson_norm",)` states that pose questions use the original
16, because §13.0 measured sd(Sampson) at 2.7× on the 480P expansion. **Under the rule
this project wrote down, the headline pose result is not significant.** It is significant
only on the set the rule excludes.

The mechanism is not the one §13.0 anticipated. Going to 60 does widen the CI, exactly as
predicted (0.0140 → 0.0152) — but the *effect* grows faster than the interval does
(−0.00575 → −0.00889), so significance arrives despite the noise penalty, not because of
extra power. The pose improvement is simply **larger on the 480P expansion scenes**. That
could be real headroom on softer inputs, or an artifact of SIFT behaving differently at
lower source resolution. With 15 measurable scenes on one side and 41 on the other, we
cannot currently tell.

**What this does and does not mean.** It does not mean the headline is wrong: the point
estimate has the same sign on both subsets, in every arm. It means the pose claim is
**underpowered and not robust to a protocol choice we made for unrelated reasons**, and
that only 15 of the 16 primary scenes yield a measurable epipolar estimate at all. No arm
in the table clears significance on the declared primary subset.

This promotes an item that was sitting in "open": **sourcing the 44 expansion scenes at
960P is now on the critical path for the paper's main camera claim**, not a
nice-to-have. Until then, the pose result should be reported with both subsets shown and
the disagreement stated, never as a single starred number.

## 17. Phase 1 — the counterfactual oracle (2026-07-29)

The plan's first decisive experiment. On `combined_qk_mem_corr/ckpt_merged`, 16 held-out
scenes × 5 diffusion timesteps × 76 800 latent positions, four actions per position
(abstain / correspondence / memory / both) with the target latent, timestep and noise
held identical across arms.

### 17.1 Two estimator bugs found before any result was read

**Pooling across timesteps hid the effect.** Risk at σ_id 949 is 4.5× the risk at 149,
so pooling raw squared errors weights the timesteps where memory does *nothing* about
five times more heavily than the ones where it works. The pooled oracle gap understated
the mean per-timestep gap by ~4×. `summarize` now reports both and says which to read.

**The naive oracle is optimistically biased, and the bias is measurable.** Selecting the
per-position argmin and scoring it on the same numbers beats the mean even when all four
actions are the *same* process. The size of that artifact is directly observable: at
σ_id 949 memory has no effect at all — the best fixed action is `abstain`, and the four
risks agree to five decimals — yet the naive oracle still reported a **0.94% gap**. That
is pure argmin-over-noise.

`honest_oracle_gap` fixes it by selecting on one noise draw and scoring on an independent
one, the held-out-split argument applied to the noise dimension. Effect on the headline:

```
                       naive     honest
  pooled oracle gap    2.05%      0.64%
```

**Two thirds of the apparent oracle gap was selection bias.** Had this been reported
naively it would have justified building a router on an artifact.

### 17.2 The gates

**GATE 1 — is memory ever harmful? PASS, strongly.**

```
  corr      harmful on 47.8% of positions   mean utility +0.00074
  mem       harmful on 45.5% of positions   mean utility +0.00225
  mem_corr  harmful on 45.6% of positions   mean utility +0.00282
```

Always-on memory is wrong on nearly half of all positions. There is a selective-memory
problem to describe.

**GATE 2 — does the oracle beat the best fixed action usefully? NO.**

σ_id 949 serves as a negative control: memory provably does nothing there, so whatever
the estimator reports at that timestep is its own noise floor (0.24%). Per-scene
bootstrap, each timestep minus that floor:

| σ_id | honest gap | minus null floor | |
|---|---|---|---|
| 149 | 1.37% [0.79, 2.02] | **+1.13% [+0.55, +1.77]** | *SIG* |
| 349 | −0.00% [−0.39, 0.35] | −0.25% [−0.62, +0.10] | n.s. |
| 549 | 0.14% [−0.07, 0.35] | −0.10% [−0.31, +0.10] | n.s. |
| 749 | 0.62% [0.47, 0.77] | **+0.38% [+0.23, +0.52]** | *SIG* |
| 949 | 0.24% | 0 by construction | — |

A real per-position routing gain exists and is statistically significant at two of five
timesteps. It is also **about 1% of the flow-matching risk at its best**, and that is an
ORACLE — an upper bound with perfect hindsight that no learned router can reach. The
plan's wording is "a useful margin"; 1% is not one.

**GATE 3 — is utility just geometric confidence? PASS, strongly.**

```
  channel      spearman   confident-but-harmful   unsure-but-useful
  validity      +0.087          40.8%                   53.2%
  confidence    +0.113          39.4%                   53.2%
  inv_depth     +0.054          42.3%                   54.8%
  view_count    +0.069          41.5%                   54.0%
  rgb_var       +0.049          40.4%                   54.8%
  depth_var     +0.005          43.9%                   56.3%
  view_angle    +0.064          41.0%                   54.0%
```

Every geometric confidence signal we have — including the multi-view disagreement
channels the model was never trained on — is **essentially uncorrelated with whether
memory helps**. Confidence barely shifts the harm rate: 39–44% of *high*-confidence
positions are harmful against a 45.6% base rate. A confidence gate is close to useless as
a router.

### 17.3 What this means

Gates 1 and 3 support the framing; Gate 2 undercuts the payoff. Read together they say
something sharper than any one of them: **memory is harmful about half the time, that harm
is invisible to geometric confidence, and it is also largely irreducible** — perfect
per-position hindsight recovers ~1% of the risk. The harm is close to symmetric noise
rather than a structured, learnable pattern.

On the plan's own predeclared criterion ("if the oracle gap is small, stop — a learned
router cannot exceed an uninteresting oracle"), **Counterfactual Memory Transport does not
currently clear its own bar as a headline claim.**

Three things must qualify that before it is treated as final:

1. **The label was initially unvalidated.** τ is a flow-matching residual. Plan §4.2
   requires showing that denoising utility predicts actual PSNR/LPIPS before it is
   trusted. Section 18 now performs that validation: the scene/action-level label is
   valid, but its downstream selection gain remains negligible.
2. **It measures this memory, not memory in general.** The arm under test is the warped-RGB
   v1 evidence branch. Plan §7 hypothesises that clean source-separated features raise the
   oracle ceiling; that hypothesis is untouched by this result and is now the cheapest way
   to move Gate 2.
3. **n = 16 scenes.** The per-timestep pattern is non-monotone (149 significant, 349/549
   null, 749 significant), which is not what a clean mechanism looks like and may be
   sampling noise.

## 18. Does denoising utility predict final-image quality? (2026-07-30)

Section 17 left one mandatory qualification open: its counterfactual label is a
flow-matching residual, while the project has repeatedly observed that denoising loss
and image quality can decouple. `bench/validate_counterfactual_label.py` now joins the
Phase-1 labels to independently sampled final-image renders.

### 18.1 Protocol

- Exact checkpoint match:
  `combined_qk_mem_corr/ckpt_merged.safetensors` in both artifacts.
- Same 16 held-out scenes and evaluation tuple.
- Flow labels use noise seed 11. Final images use the standard independent render seed
  7 and 30 denoising steps.
- Existing saved renders are reused, so this requires no new 5B-model or VGGT pass.
- The primary predictor was fixed before reading image metrics:

  \[
  \bar{\tau}_a =
  \frac{1}{4}\sum_{s\in\{149,349,549,749\}}
  \frac{R_{\rm none,s}-R_{a,s}}{R_{\rm none,s}}.
  \]

  Relative utility prevents high-risk timesteps from dominating. Sigma 949 is excluded
  because §17 established it as the no-effect negative control, but it is still reported.
- The unit of replication is the **scene**. The two tested actions (`mem`, `mem_corr`)
  are a correlated pair within each scene, so confidence intervals resample entire scene
  clusters and permutation tests permute scene pairs.
- Available final renders test `mem` and `mem_corr`; there is no saved `corr`-only render.
  This validates scene/action-level utility, not the spatial mixed-action oracle.
- “LPIPS” below is the repository's LPIPS-style normalized VGG feature distance.

### 18.2 Label validity

| downstream quality gain | Spearman ρ | scene-bootstrap CI95 | cluster permutation p | sign agreement |
|---|---:|---:|---:|---:|
| full PSNR | **+0.663** | **[+0.263, +0.893]** | **0.00125** | **87.5%** |
| SSIM | **+0.550** | **[+0.093, +0.863]** | **0.0131** | **84.4%** |
| LPIPS-style VGG | **+0.667** | **[+0.265, +0.862]** | **0.00230** | **84.4%** |
| covered-region PSNR | **+0.653** | **[+0.278, +0.862]** | **0.00230** | **87.5%** |
| frontier PSNR | +0.306 | [−0.115, +0.636] | 0.174 | 62.5% |

The label is valid at the scene/action level: when flow utility says the existing
memory action helps more, independently rendered full-image, covered-region, and
perceptual quality usually improve more. The lack of frontier association is also
mechanistically sensible: the warped-memory action has no observation to transport
there.

The negative control behaves oppositely. At sigma 949, where memory has effectively no
denoising leverage, correlation with final PSNR is −0.354 and sign agreement is only
18.8%. This confirms that including every timestep indiscriminately would make a worse
label, not a more complete one.

### 18.3 Valid label, negligible policy value

The proxy validation does **not** rescue Gate 2:

```
best fixed final-image action                         mem_corr
best fixed gain over abstain                          +0.4314 dB
flow-selected gain over that best fixed action        +0.00369 dB
scene-bootstrap CI95                                  [−0.00170, +0.01056]
flow choice agrees with image oracle                  68.8%
remaining image-oracle regret                         0.0514 dB
```

LPIPS says the same: flow selection minus the best fixed action is
−2.53e−6 [−2.05e−5, +9.25e−6], indistinguishable from zero.

This separates two questions that would otherwise be conflated:

1. **Is τ a meaningful image-quality proxy? Yes**, for whole-scene `mem` and
   `mem_corr` actions.
2. **Does routing those current actions offer useful headroom? No.** Always using
   `mem_corr` already captures essentially all of the available final-image gain.

The §17 no-go is therefore stronger, not weaker. Its ~1% honest flow-space oracle gap
was not small because the label was meaningless; the current warped-RGB action space
simply contains little useful selective advantage.

### 18.4 Decision

- Do not train the immediate counterfactual router on warped-RGB v1.
- Keep the cross-fitted oracle and downstream label validator as diagnostic tools.
- The cheapest experiment that can change Gate 2 is Phase 4 clean,
  source-separated feature transport. It creates a genuinely different appearance
  action rather than learning a policy over nearly equivalent noisy actions.
- Re-run all three oracle gates and this exact downstream validation before training
  a router on clean features.
- Corr-only and spatial mixed-action label validation remain explicit limitations;
  do not claim that this experiment validates per-pixel routing.

### 17.4 The label is valid — and §17's verdict survives it

§17.3 flagged that τ was an unvalidated proxy and made validating it the next experiment.
`bench/validate_counterfactual_label.py` runs plan §4.2 against the existing 30-step
render artifact for the same checkpoint (`n60_combined.json`), so no re-rendering was
needed. Design points that matter: the primary predictor — mean relative utility over
σ 149/349/549/749 — is **predeclared before any image metric is read**; utility is
RELATIVE because the raw residual scale moves ~5× across the schedule; σ 949 is held out
as the negative control; and the unit of replication is the **scene**, so CIs bootstrap
and permutations shuffle whole scene clusters rather than treating millions of latent
positions as independent.

```
metric      spearman   scene-boot CI95      perm p     sign     selection - best fixed
psnr         +0.663   [+0.263,+0.893]      0.00125    87.5%    +0.0037 dB [-0.0017,+0.0106]
lpips        +0.667   [+0.265,+0.862]      0.0023     84.4%    -2.5e-06   [-2.1e-05,+9.3e-06]
ssim         +0.550   [+0.093,+0.863]      0.0131     84.4%    +2.6e-05   [-1.4e-03,+1.1e-03]
psnr_cov     +0.653   [+0.278,+0.862]      0.0023     87.5%    -0.026 dB  [-0.175,+0.087]
psnr_fro     +0.306   [-0.115,+0.636]      0.1739     62.5%    +0.0098 dB [-0.025,+0.055]
```

**The label is valid.** τ predicts realised image quality at ρ ≈ 0.66 against both PSNR and
LPIPS, with cluster-permutation p ≈ 0.001–0.002 and 87.5% sign agreement. The concern that
motivated §17.3 — that this project has twice seen denoising loss decouple from image
quality — does not apply here.

**The negative control behaves.** At σ 949, where memory is a measured no-op, the
association is not merely absent but *negative* (ρ −0.35 to −0.46, sign agreement 19–28%).
A validator that reported a positive association there would be detecting its own
plumbing.

**Validity is region-dependent, in exactly the place §13.11 predicted.** τ tracks quality
in warp-COVERED regions (ρ +0.653, p=0.002) and not on the FRONTIER (ρ +0.306, CI crosses
zero, p=0.17, sign 62.5%). The memory branch transfers evidence it can see and contributes
nothing where it cannot, so there is nothing on the frontier for τ to be right about.

**And the selection gain is null in image space too.** On every metric, "flow selection
minus best fixed action" has a CI95 that crosses zero. The best fixed action is `mem_corr`
throughout, and choosing per scene with τ adds +0.0037 dB of PSNR — indistinguishable from
just leaving memory on. The image-space oracle's own regret is only 0.051 dB, so there is
very little to select *for* even with perfect hindsight.

**Net: §17's Gate-2 verdict stands, now on a validated label.** Flow space and image space
agree that per-scene action selection is worth nothing measurable. The two failure modes
that could have rescued the thesis — a bad label, or a real effect hidden by the wrong
metric space — are both eliminated.

**What is still NOT validated,** and the script says so in its own scope block: this
validates SCENE-LEVEL selection among {abstain, mem, mem_corr}. §17's Gate 2 measured a
per-POSITION mixed-action oracle, and the render artifact has no `corr`-only arm. So the
spatial oracle remains unvalidated against pixels. Given that the scene-level result is a
clean null and the spatial honest gap was ~1%, the burden of proof now sits firmly on
anyone claiming the spatial version is different.

## 18. Phase 4 — clean source-separated features (2026-07-30)

Run as the final falsification test that could reopen Counterfactual Memory Transport,
not as continued commitment to it. §17 killed the warped-RGB formulation; the one
hypothesis it left untouched is that clean per-source features change the action space
from three near-equivalent global interventions into genuinely different per-source
treatments.

### 18.1 VGGT amortisation (prerequisite)

`data/vggt_cache.py` caches joint geometry on
`scene + context indices + resolution + VGGT checkpoint fingerprint`. Verified
**bit-exact** against the uncached path (max|diff| = 0.0); warm forward 0.54 s → cache hit
0.002 s. All 60 held-out scenes warmed in 37 s, 177 MB.

Two correctness points are enforced rather than assumed. Context ORDER is in the key,
because the returned list is one depth per context frame in order. And
`estimate_consistent_depths(images=...)` — the rollout path where context frames are the
model's OWN generations — **bypasses the cache entirely**, because those pixels are not
implied by `(scene, ctx_indices)` and a hit would silently substitute ground truth for
generated content in exactly the experiment designed to measure the difference.

### 18.2 Representation screen — PASS, narrowly

No training. Warps each source's CLEAN VAE latent into the target grid
(`memory/latent_splat.py`) instead of splatting RGB in pixel space and re-encoding, then
scores against the ground-truth target latent. Arms are compared only where the RGB warp
AND at least one clean source landed a surface, so the coverage difference cannot flatter
either side.

```
arm            covered latent MSE      vs rgb_warp
rgb_warp              0.34886
clean_fixed           0.28414       -0.06472 [-0.10196,-0.03432] *
clean_oracle          0.23474       -0.11412 [-0.14843,-0.08491] *
clean_s0              0.30133       -0.04752 [-0.08393,-0.01203] *
clean_s1              0.33400       -0.01486 [-0.05766,+0.03130]
clean_s2              0.30797       -0.04089 [-0.08513,+0.00415]

error reduction: +18.46% [+10.31,+27.71]      scene wins: 13/16
coverage:        rgb 69.6%   clean 58.0%
```

**Predeclared gate (≥10% reduction AND ≥12/16 wins): PASS.** But narrowly — the CI lower
bound is 10.31% against a 10% bar, and 13 wins against a 12 bar. This is a screening
diagnostic on latent MSE, not a paper result.

**A clamp bug inflated the first run and was caught by a test, not by inspection.**
`bilinear_splat(is_image=True)` clamps its output to [-1,1] and fills holes with -1 —
correct for RGB, destructive for a VAE latent, whose values range well outside that
interval. The first screen reported +24.15% and 16/16. `is_image=False` gives the numbers
above. The bug was invisible in the aggregate (the clamped version still "beat" RGB and
looked like a clean win); what exposed it was
`test_identity_pose_round_trips_the_latent`, which warps a source into its OWN view and
demands the identity — it failed with max error 3.1 on a warp that is mathematically a
no-op. **Aggregate plausibility is not verification.**

### 18.3 What the screen does and does not license

The number that matters for reopening CMT is not the 18% headline but the **per-position
source-oracle headroom: clean_fixed 0.28414 → clean_oracle 0.23474, a further 17.4%
available from choosing the right SOURCE per position.** §17's honest oracle gap on three
global actions was ~1%. So the action space really did change, which is the premise the
Phase-4 test existed to check.

Two cautions carried into the adapter stage:

- **Clean evidence covers less** — 58.0% of positions against the RGB warp's 69.6%.
  Splatting a 24×40 grid leaves more holes than splatting 384×640 and pooling. The clean
  path gives better evidence where it lands and nothing in more places, so the per-source
  validity masks are load-bearing, not decoration.
- **Latent MSE is not the causal quantity.** §17 measured a real association between
  denoising utility and PSNR (ρ 0.66) and still found the selection gain null. A 17.4%
  representation-level headroom is not a 17.4% causal routing gain, and the predeclared
  reopening threshold (≥3% honest oracle gap after null-floor subtraction, plus ≥0.1 dB
  covered PSNR) remains the actual bar.

### 18.4 Phase-4 bounded adapter and the per-source oracle — FAIL

The adapter trained to step 3000 of a 4000-step budget: the run died at its final save
with ENOSPC. Cause was mine — I wired the new VGGT cache into `train_memory`, which draws a
fresh random context set every step, so ~16 000 unique tuples each wrote a 2.9 MB entry
that would never be read again and the cache reached **44 GB**. The cache is right for
evaluation (60 fixed tuples serving hundreds of runs) and pure cost for training.
`cached_consistent_depths(write=False)` plus an 8 GB cap now prevent it, both tested. The
step-3000 checkpoint was verified complete (1 492 tensors, last tensor readable and
finite), and val loss had been flat since step 1500 (0.1765 → 0.1771), so the missing
1 000 steps are unlikely to change the result — but it is a shortfall against the stated
budget and is reported as one.

**The oracle over the 8-action per-source space:**

```
 sigma   abstain  best fixed  naive %  honest %     bias  best action
   149   0.16418     0.16055    2.60%     1.15%  0.00232  clean_topk
   349   0.16084     0.15839    2.82%     0.46%  0.00374  clean_topk
   549   0.17997     0.17895    2.43%     0.51%  0.00343  clean_topk
   749   0.23259     0.23203    1.60%     0.52%  0.00249  clean_topk
   949   0.59362     0.59354    0.80%     0.25%  0.00330  clean_s0   <- negative control

 sigma        gap    minus null floor
   149      0.91%    +0.68% [+0.30,+1.06] *SIG*
   349      0.04%    -0.19% [-0.51,+0.15]  n.s.
   549      0.31%    +0.08% [-0.06,+0.21]  n.s.
   749      0.46%    +0.23% [+0.17,+0.29] *SIG*
```

| predeclared criterion | required | measured | |
|---|---|---|---|
| mean honest gap after null floor | ≥ 3.00% | **0.18%** | **FAIL** |
| CI95 excludes zero | yes | [+0.05, +0.36] | PASS |
| informative timesteps, same sign | ≥ 3/4 | 3/4 | PASS |

**Verdict: FAIL, by a factor of ~17 on the primary criterion.** The effect is real —
statistically detectable and consistently signed — and practically negligible.

Two details settle it beyond the arithmetic. **`clean_topk` is the best fixed action at
every informative timestep**: "use all the sources" is already optimal, and per-position
source selection adds essentially nothing on top. And the net honest gap here (0.18%) is
no larger than §17's warped-RGB version (~0.29%), so **expanding the action space from 3
global actions to 8 per-source ones did not increase causal routing headroom at all.**

### 18.5 The finding worth keeping

§18.2 measured a 17.4% per-position source-oracle headroom in latent MSE. §18.4 measured
0.18% in causal denoising utility. **Representation-level headroom did not transfer to
causal routing headroom**, and the gap between the two numbers is two orders of magnitude.

That is not a contradiction, it is the result: clean per-source features genuinely are
better evidence (§18.2, +18.5% covered latent MSE, 13/16 scenes), and the generator does
not need to be told which source to use. Given all of them it already does at least as
well as any oracle-selected subset. The right memory action, in this system, is "supply
the evidence" — not "choose among it".

Together with §17 and §17.4 the picture is consistent across three independent
measurements: memory is harmful about half the time (§17.2), that harm is invisible to
geometric confidence (§17.2) and to a validated denoising-utility label (§17.4), and it is
**not recoverable by selection** under either the warped-RGB action space (§17.2) or the
clean per-source one (§18.4). The harm behaves like symmetric noise, not a learnable
pattern.

**Counterfactual Memory Transport has now failed its own predeclared go/no-go twice, on
the formulation it was designed for and on the one hypothesis that could have rescued it.**
Per the decision rule, it stays closed. Clean source-separated memory should be retained
or dropped on ordinary fixed-component grounds — whether it beats the v1 fused branch on
quality — which is a separate, unmeasured question, and NOT on the routing claim.

### 18.6 Fixed-component comparison: checkpoint provenance blocks clean-vs-v1

Before rendering the proposed fixed-component A/B, an exact checkpoint guard found that
the two stage-2 runs do **not** share the same stage-1 snapshot:

```
v1 2k/4k     vs archived base in combined_qk_mem_corr/ckpt_merged: 637/637 exact
clean 2k/3k  vs current qk_fullft_camqk/ckpt_last:                 637/637 exact
v1 2k        vs current qk_fullft_camqk/ckpt_last:                 482/637 exact
```

The first mismatch is `adapter.output_norm.bias` (max |diff| 0.00256); 155 shared
incremental tensors differ in total. The timestamps explain the provenance break:
`combined_qk_mem_corr/ckpt_merged` was created at 23:06 on 07-28, while
`qk_fullft_camqk/ckpt_last` was overwritten at 04:14 on 07-29. The merged v1 artifact
preserves the old stage-1 snapshot, while Phase 4 trained against the later file at the
same pathname.

Therefore a direct clean-vs-v1 render on the current stage-1 file would apply the v1
geometry branch to a backbone it was not trained with and is **not** a
representation-only comparison. `bench/eval_source_memory_fixed.py` refuses that
comparison unless both incremental checkpoints and the selected full checkpoint agree
bit-for-bit on their copied tensors, and the two branches' full base snapshots agree
bit-for-bit on **every** non-memory/non-correspondence tensor (including trained base
Q/K weights that incremental files do not contain).

The immediate valid test is **clean vs OFF on its own current base**. If clean does not
improve full/covered image quality without a pose regression, drop it and stop. Only if
that paired test passes is a causal clean-vs-v1 comparison worth the cost: retrain the v1
branch for 2k steps from the current stage-1 snapshot (same tuples and exposure), then
run the guarded three-arm evaluator. Reusing the historical v1 checkpoint would leave a
base-by-representation interaction confounded.

Use `phase4_srcmem/ckpt_last.safetensors` as the immutable full base snapshot for that
test, not the mutable `qk_fullft_camqk/ckpt_last` pathname. The clean step-3000
incremental agrees with it on all 637 shared tensors, and loading the full snapshot
without memory/correspondence modules ignores its branch tensors before the evaluator
reattaches the selected incremental branches.

### 18.7 Fixed clean system vs OFF — PASS, attribution still open

The guarded n=16 render used the immutable Phase-4 full snapshot, verified all 637 copied
shared tensors exactly, and held the tuple, 30-step sampler, and seed 7 fixed.

| criterion | required | measured | |
|---|---:|---:|---|
| mean PSNR gain | ≥ +0.10 dB, CI excludes 0 | **+0.275 [+0.110,+0.471]** | PASS |
| LPIPS-style direction | improves | **−0.000123 [−0.000180,−0.000075]**, 16/16 | PASS |
| original-16 pose | non-inferior | Sampson +0.00114 [−0.00801,+0.01165], 8/15 | **UNRESOLVED** |
| coverage catastrophe | none | worst scene −0.195 dB, 13/16 wins | PASS |

Region PSNR also improves: v1-covered +0.223 [+0.067,+0.412], v1-frontier +0.361
[+0.128,+0.611], and common coverage +0.521 [+0.249,+0.827]. The distribution is
right-skewed (mean +0.275, median +0.148; largest win +1.326 dB), so this is a real but
heterogeneous component gain. LPIPS is perfectly signed and tiny in magnitude; it is
directional corroboration, not a perceptual headline.

The pose result shows **no detected regression**, but does not establish formal
non-inferiority. A nonsignificant difference is not a non-inferiority test: no margin was
predeclared, and the two-sided upper CI permits +0.01165 Sampson (about 22% of OFF's
0.05266 mean). It would not clear a +0.005 margin. This is exactly why sourcing the 44
expansion scenes at 960P remains on the critical path; the original-16 pose set is too
small to certify a tight safety margin.

**The fidelity/component gate passes; pose safety is provisionally reassuring but
underpowered, and the clean-latent mechanism is not yet identified.** In this
evaluator, `clean` enables BOTH the source-memory adapter and the jointly trained
two-sided correspondence adapter, while OFF disables both. Thus the result licenses
retaining the fixed geometry system and running the n=60 confirmation, but cannot assign
the gain to transported clean latents. This matters because clean evidence covers only
33.1% while the gain is larger on the v1 frontier than on its covered region.

`eval_source_memory_fixed --mechanism-arms` predeclares the decomposition before another
number is read:

```
OFF
correspondence only
geometry-only source memory + correspondence   (all clean-latent planes zero)
clean-latent source memory + correspondence
```

The decisive contrasts are correspondence−OFF, geometry+corr−correspondence, and
clean−geometry+corr. If the last is null, the useful component is geometric
conditioning/correspondence rather than clean latent transport, and retraining a matched
v1 memory branch is unnecessary.

## 19. The sample-matched control (2026-07-30)

§16.2 found the "matched frozen control" behind every headline number had seen 16 000
samples against the Q/K arm's 48 000, and that its tripwire was still rising when it was
stopped. This retrains it at 48 000 samples.

### 19.1 A control-composition error, caught before it produced a number

The first 48k control was launched WITHOUT `--cam-qk`. Recovered by diffing its own
checkpoints:

| | trainable | tensors | branches |
|---|---|---|---|
| control 4k (16k samples) | 25.8 M | 47 | adapter, **camqk**, film, null_context |
| control 12k (48k samples), first attempt | 21.9 M | 32 | adapter, film, null_context |
| Q/K arm | 604.1 M | 285 | + unfrozen self-attn Q/K, **camqk** |

The Q/K arm trains unfrozen Q/K *and* a camqk adapter, so a control lacking camqk does not
isolate Q/K unfreezing — it conflates two interventions, and it is a WEAKER control, which
inflates the gap in the flattering direction. Measured inflation: the headline moved
**+0.381 → +0.418 dB** when scored against the weaker control. Small, but in the direction
that survives review unchallenged. A matched control with `--cam-qk` is retraining; the
numbers below are provisional on the fidelity axis and the camqk-less arm is retained as a
labelled secondary.

### 19.2 Result: fidelity survives, pose weakens

```
arm                                PSNR    dPSNR vs base        Sampson  dSampson n<=16
base (frozen view_set_r32)       11.339                          0.0571
control frozen  4k (16k samples) 11.653  +0.315 [+0.178,+0.461]* 0.0585  -0.00200   8/15
control frozen 12k (48k samples) 11.617  +0.278 [+0.148,+0.420]* 0.0585  -0.00153  10/15
qk batch4 12k  (48k samples)     12.035  +0.696 [+0.412,+0.991]* 0.0414  -0.00484   9/15
```

**Tripling the control's training made it very slightly WORSE** (+0.278 vs +0.315), and its
tripwire was flat at 10.36 from step 6500 to 12000. §16.2's concern — that the control was
stopped while still improving — is not supported: it had converged, and the monotone
tripwire rise that prompted the concern did not continue.

Headline, against each control, with Sampson on both subsets:

```
qk batch4 12k  vs  control 4k (16k samples)
  PSNR    n=60  +0.381 [+0.168,+0.616] *SIG*  wins 42/60
  Sampson orig16 n=15  -0.00575 [-0.01298,+0.00097] n.s.  10/15  p=0.302
  Sampson all60  n=41  -0.00889 [-0.01650,-0.00134] *SIG* 29/41  p=0.012

qk batch4 12k  vs  control 12k (48k samples, no camqk)
  PSNR    n=60  +0.418 [+0.179,+0.677] *SIG*  wins 43/60
  Sampson orig16 n=15  -0.00151 [-0.01376,+0.01124] n.s.   8/15  p=1.000
  Sampson all60  n=40  -0.00917 [-0.01913,+0.00056] n.s.  27/40  p=0.038
```

**The fidelity headline survives a sample-matched control** — significant against both, and
the gap does not shrink when the control gets 3x the samples.

**The pose headline does not improve, and against the 48k control it degrades.** It was
already subset-dependent (§16.5: n.s. on the declared-primary original 16, significant only
on all 60). Against the 48k control the all-60 mean CI now crosses zero as well
([-0.01913, +0.00056]); only the sign test survives (27/40, p=0.038), and orig16 is a
coin flip (8/15, p=1.000). Two controls, two subsets, four tests: **one is significant.**

The honest reading is that the camera-control claim rests on a pose effect this evaluation
cannot reliably resolve at n=15 measurable scenes, and that sourcing the 44 expansion
scenes at 960P (§16.5) is not optional for the paper — it is the measurement that decides
whether the claim exists. The fidelity claim is solid and independent of it.

### 19.3 Final matched-camqk comparison protocol (frozen before its result)

The replacement control is `mode=none, cam_qk=true`, otherwise matched to the Q/K run:
48k samples, rank 32, AdamW at 2e-5, EMA, three contexts, and the same base checkpoint.
The evaluator renders each co-trained run with its camqk branch both disabled and enabled.
To prevent choosing whichever comparison looks best after the run:

1. **Primary mechanism comparison:** Q/K-trained with camqk OFF minus frozen-trained with
   camqk OFF. This is the existing `qk batch4` headline and asks what unfreezing backbone
   Q/K bought when the optional attention-coordinate branch is surgically knocked out in
   both co-trained systems.
2. **Secondary deployed-system comparison:** Q/K-trained with camqk ON minus
   frozen-trained with camqk ON.
3. The difference between those two effects is a descriptive Q/K×camqk interaction, not
   a third independent confirmation test.

Fidelity uses n=60 PSNR as primary. Pose uses the original 960P set as primary and all-60
as explicitly secondary until the 44 expansion scenes are sourced at 960P. Every pose
contrast uses the common jointly measurable scene subset and reports mean bootstrap,
median bootstrap, and sign count together. The generic evaluator names a loaded full
checkpoint's knockout arm `qk` even when its metadata says `mode=none`; the final table
must relabel those fields as control-camqk-OFF/ON before joining artifacts.

Operational note: `train_fullft --archive-every 0` disabled the disk preflight and delta
archives but still wrote an unbudgeted ~10 GB numbered FULL checkpoint every 4k steps.
The condition now honors zero and keeps only `ckpt_last`. The already-running process
loaded the old code and will still emit its 8k/12k archives; verify the final checkpoint,
then archive or remove redundant numbered copies before starting another full-snapshot
run.

## 20. 960P expansion source (2026-07-30)

All 44 expansion held-out scenes acquired at 960P from `DL3DV/DL3DV-ALL-960P` shard `4K`
(8.9 GB, 44/44, zero failures) via `data/fetch_960p_expansion.py`. Frame counts verified
per scene against the 480P copies and any mismatch rejected rather than accepted:
`evalkit.eval_tuple` derives its indices from `len(scene)`, so a differing frame set would
silently evaluate the same scene at different cameras.

The source is genuinely better. At the 384x640 working resolution the 480P copies were
**upsampled** (480 wide -> 640) and carry **1.89x less high-frequency energy** than the
960P copies downsampled to the same size.

### 20.1 But resolution was not the main cause — a correction to §13.0

§13.0 attributed the expansion set's 2.7x Sampson variance to "softer 480P frames give
fewer and worse SIFT matches". Measured directly (SIFT matches between the anchor and
target ground-truth frames of each eval tuple — an upper bound on what the pose metric can
use):

| set | median SIFT matches | below-15 rate |
|---|---|---|
| expansion @ 480P | 20 | 27.3% |
| expansion @ 960P | **28** | **22.7%** |
| originals @ 960P | **108** | 12.5% |

Resolution is worth a 40% lift in median matches and 4.6 points of abstention. It leaves
the expansion set at **roughly a quarter of the originals' match density** and nearly
double their abstention rate. **Most of the disparity is scene-intrinsic** — texture,
baseline and motion in those particular scenes — **not source resolution.**

This matters for what the upgrade licenses. The stated hope (§16.5, §19) was that 960P
would "collapse the two subsets into one properly-powered set" and settle the camera claim.
It does not obviously do that. The expansion scenes remain harder for epipolar geometry
after the fix, so a merged 60-scene pose statistic would still be dominated by a
high-variance majority. Whether the subsets may now be merged is an empirical question
about sd(Sampson) on re-rendered 960P frames, not something the acquisition settles.

### 20.2 What this invalidates

`_SEARCH_ROOTS` now prefers 960P/4K. The pixels for those 44 scenes change (mean |diff|
0.033 at working resolution), which changes BOTH the ground truth PSNR is measured against
AND the context frames the model generates from. **Every cached render and every recorded
n=60 fidelity number for the expansion scenes is superseded, not refreshed.** The original
16 are untouched and their numbers stand.

Re-rendering is therefore a prerequisite for any further n=60 claim, and the honest order
is: re-render, re-measure sd(Sampson) on the 960P expansion, and only then decide whether
`evalkit.USE_ORIGINAL_FOR` can be retired.

### 20.2 Ground-truth epipolar sanity exposes a calibration/scene-validity problem

Before either 960P model rerender finished, the Sampson metric was run with the REAL
target frame in place of a generated frame. A real target should satisfy the commanded
anchor→target fundamental matrix up to annotation/rolling-shutter/dynamic-scene noise.
It does on most originals and does not on most expansion scenes:

| 960P stratum | measurable GT | median GT Sampson | mean | sd |
|---|---:|---:|---:|---:|
| original | 14/16 | **0.000316** | 0.0122 | 0.0261 |
| expansion | 41/44 | **0.03225** | 0.0463 | 0.0487 |

This is stronger evidence than match density. The expansion stratum's real images
already violate the nominal epipolar geometry by about two orders of magnitude more at
the median. The 480P and 960P `transforms.json` files are byte-identical for checked
scenes, so this was not introduced by the new root; it reflects pose/intrinsic noise,
rolling shutter, dynamics, or other scene-intrinsic violations. The original set is not
perfect either: several of its GT targets also have large residuals.

The raw all-60 Sampson statistic is therefore not a valid camera-adherence headline,
regardless of its CI. Resolution alone cannot repair bad commanded geometry.

An outcome-independent validity diagnostic is frozen here, before reading either 960P
model arm:

```
GT anchor↔target SIFT matches >= 15
GT normalized Sampson <= 0.01
```

At 640×384, 0.01 is already a permissive median epipolar error of about 7.5 pixels. The
threshold also follows a gap in both observed distributions (original 0.00318→0.01969;
expansion 0.00764→0.02176), rather than being tuned to a model result. It retains 11/16
originals and 16/44 expansion scenes before requiring joint measurability in the model
arms. The exact per-scene diagnostic is saved in `outputs/gt_sampson_960p.json`.

Final pose reporting must now show:

1. the historically declared original-16 analysis, labelled underpowered and including
   its existing metric noise;
2. the new GT-valid subset, with original and expansion effects reported separately;
3. a pooled GT-valid effect only if the paired Q/K−control delta variance and sign are
   compatible across strata;
4. the retained/excluded counts. Exclusion is based only on GT metric validity, never on
   either model's outcome.

## 21. The final headline under the GT-validity protocol (2026-07-30)

Both arms re-rendered at 960P back-to-back in one session. Base-arm agreement is
**0.00000 dB mean and 0.00000 dB worst-case per scene** — the base frames are bit-identical,
so the pairing is exact rather than merely tolerable. This supersedes §19, whose expansion
scenes were 480P-derived and which predates the GT-validity gate.

Control composition and sample count both matched: 25.8 M trainable across
adapter/camqk/film/null_context, 48 000 samples, against the Q/K arm's 604.1 M and the same
48 000.

### 21.1 Fidelity

```
arm                                 PSNR    dPSNR vs base
base (frozen view_set_r32)        11.374
control matched 12k (48k)         11.565  +0.191 [+0.072,+0.317]*
qk batch4 12k  (48k)              11.935  +0.561 [+0.294,+0.838]*

HEADLINE  qk - control:  +0.370 [+0.149,+0.606]  *SIG*   wins 40/60
```

### 21.2 Pose, on GT-valid scenes only, reported per the frozen protocol

```
stratum     n    mean                          median                    sign
original   10   -0.01162 [-0.02232,-0.00014]*  -0.01494 [-0.02691,+0.00147]   7/10 p=0.344
expansion  14   -0.01577 [-0.03314,-0.00059]*  -0.00803 [-0.02508,+0.00552]  10/14 p=0.180
                          -> same sign, variance ratio <= 3: POOLING PERMITTED
pooled     24   -0.01404 [-0.02530,-0.00408]*  -0.01248 [-0.02065,-0.00137]* 17/24 p=0.064
```

Retained: 24 scored of 27 GT-valid of 60 held-out.

**This is the first pose result in the project that survives its own protocol.** Both
strata agree in sign, both mean CIs exclude zero, and the pooled mean AND median CIs both
exclude zero. Crucially the pooling precondition was frozen before any of these numbers
existed and was checked, not assumed — §19's subset flip is exactly what it exists to
catch, and here it does not fire.

**What is NOT claimed.** The sign test is marginal everywhere (p=0.344, 0.180, pooled
0.064) and reaches significance nowhere. Per-stratum medians individually cross zero; only
the pooled median clears. n=24 of 60 is a heavy power cost, imposed deliberately by the GT
gate. The honest statement is that the camera effect is **directionally consistent across
strata and significant in the mean and pooled median, with a sign test that does not
confirm it** — which is a real result under a pre-registered protocol, not a clean one.

Reporting all three statistics together is not optional here: quoting the mean alone would
overstate this, and quoting the sign test alone would understate it.

### 21.3 What changed the answer

Nothing about the model. §19 measured the same checkpoint and reported the pose claim as
significant on one subset and not the other, with no way to adjudicate. The difference is
in the measurement: excluding scenes whose GROUND TRUTH violates its own commanded
epipolar geometry by more than ~7.5 px median. On the excluded scenes the metric was
reporting a discrepancy an order of magnitude larger than the effect being measured.

**Stated at the strength the evidence supports.** This is a concrete, measured explanation
for the subset disagreement; it is not yet established that it ACCOUNTS for the flip. That
requires showing the flip disappears under the gate on comparable renders, which §21.2
suggests (both strata now agree in sign) but does not prove — the pre-gate and post-gate
numbers also differ in source resolution and render session.

**And on what the discrepancy IS:** the Sampson test cannot separate camera-metadata error,
intrinsics error, rolling shutter, or scene dynamics from genuine pose error. It shows only
that **the commanded geometry is unreliable** on those scenes. "The camera did not go where
it was told" is one candidate explanation among several and is not what was measured.

## 22. Clean-memory mechanism decomposition (n=16, 2026-07-30)

Run before any v1 retraining, per the locked sequencing: attribute the clean branch's gain
to its parts before spending a training run on a representation comparison. One backbone,
one context encoding, one geometry, one seed; arms differ only by masking.

```
contrast                                     dPSNR                 CI95        dLPIPS
correspondence          (corr - OFF)        +0.124 [+0.009,+0.294]*  -0.000060
source geometry     (geom+corr - corr)      -0.002 [-0.078,+0.076]   -0.000003
transported latent (clean - geom+corr)      +0.154 [+0.059,+0.258]*  -0.000059
clean total             (clean - OFF)       +0.275 [+0.110,+0.471]*  -0.000123
```

The parts sum to the whole (0.124 - 0.002 + 0.154 = 0.276 against a measured +0.275), so the
decomposition is internally consistent rather than three unrelated contrasts.

**Two findings, one of which changes the design.**

1. **The transported clean latent carries the gain**: +0.154 dB, CI excludes zero, and
   `clean - corr` (+0.151) is nearly identical — i.e. essentially all of the branch's value
   beyond correspondence is the warped clean CONTENT.
2. **The per-source geometry planes contribute nothing**: -0.002 dB with a CI tight around
   zero ([-0.078, +0.076]). Validity, inverse depth, confidence and view angle, delivered
   per source, add no measurable fidelity on top of the correspondence branch.

Finding 2 is worth stating carefully. It does NOT show geometry is useless — the
correspondence branch is itself geometric, and it contributes +0.124. It shows that ONCE
correspondence is present, the additional per-source geometry planes are redundant. The
untested inference is that the slot layout could drop those four planes without loss; that
is a plausible reading of a null, not a measured arm, and would need `latent-without-
geometry` to confirm.

**Gate outcome:** the final contrast is materially positive, so the locked plan's condition
for proceeding to an immutable-base v1-2k retrain and the guarded three-arm comparison is
met.

## 23. Guarded three-arm comparison: fused v1 beats clean source-separated memory

§22's gate passed, so fused v1 was retrained for 2 000 steps from the SAME immutable
stage-1 base as the clean branch (`qk_fullft_camqk/ckpt_last`, two-sided, `source_memory:
False`). §19's provenance finding made this mandatory: the historical v1 adapter was
trained against a stage-1 snapshot that differed from the current base in 155 of 637
tensors — precisely the trained Q/K LoRA weights — so a historical comparison would have
attributed 4 000 steps of backbone training to the memory representation.

Four guards passed before a single frame was rendered: clean verified against its stage-1
base, v1 against its base, both full base snapshots bit-compared excluding stage-2
branches, and the shared tensors of the two stage-2 states verified equal.

```
contrast              dPSNR                     dLPIPS
v1    - off          +0.455 [+0.240,+0.743]*   -0.000143 [-0.000224,-0.000077]*
clean - off          +0.275 [+0.110,+0.471]*   -0.000123 [-0.000180,-0.000075]*
clean - v1           -0.180 [-0.332,-0.037]*   +0.000020 [-0.000026,+0.000069]
```

**Clean source-separated memory is significantly WORSE than the fused v1 branch it was
built to replace on fidelity** (PSNR CI excludes zero). LPIPS reverses direction but is
indistinguishable from zero; it is not evidence of a perceptual regression. It does beat
no-memory — §22's +0.275 stands — but beating OFF was never the question.

The initially reported pose comparison used all 15 jointly measurable original scenes,
which predates §20.2's frozen ground-truth validity gate and is not admissible. Recomputed
on the 11 GT-valid and jointly measurable originals:

```
contrast          mean dSampson                 median                 wins
v1    - off      -0.00384 [-0.01424,+0.00617]  -0.00334 [-0.01080,+0.00158]  8/11
clean - off      +0.00057 [-0.00730,+0.00930]  -0.00130 [-0.01004,+0.00772]  6/11
clean - v1       +0.00440 [-0.00782,+0.01630]  +0.00585 [-0.01475,+0.02315]  3/11
```

Pose is directionally consistent with the fidelity ordering but statistically null. The
drop decision rests on the significant paired PSNR comparison, not pose.

The comparison is if anything generous to clean: it had **3 000 training steps to v1's
2 000**, on the same base, same tuples, same seed and sampler.

**Decision: drop the clean source-separated representation; retain fused v1.** Per the
locked plan's rule, the simpler system wins.

### 23.1 The pattern this completes

This is the second time in three days that a large latent-space representation advantage
failed to survive contact with a trained model:

| representation-level measurement | trained-model outcome |
|---|---|
| §18.2: clean latent evidence beats warped RGB by **+18.5%** covered latent MSE, 13/16 scenes | §23: the trained clean branch is **−0.180 dB WORSE** than the fused branch |
| §18.2: per-position source oracle offers a further **17.4%** headroom | §18.4: causal routing gap **0.18%** against a 3% bar |

The consistent lesson is that **evidence quality measured in latent space does not predict
trained-model quality.** A branch is not a lookup table: the model has to learn to use what
it is given, and a representation that reconstructs the target better in isolation can be
harder to exploit. Both §18.2 screens were run precisely because they were cheap proxies
for expensive experiments; both proxies pointed the wrong way. Future screens of this kind
should be treated as feasibility filters — they can kill a direction, but they cannot
endorse one.

### 23.2 Full-power clean-versus-OFF record

The first output named `fixed_clean_3k_n60` was invalid as an n=60 run: `--limit` still
defaulted to 16, and its recorded configuration (`expanded: true, limit: 16`) exposed the
silent truncation. The evaluator now defaults to `limit=0`, prints the actual scene count,
and records both `n_available` and `n_evaluated`.

The corrected run contains 60 unique scenes (`expanded: true, limit: 0`):

```
clean - OFF PSNR   +0.287 [+0.161,+0.430]*
clean - OFF SSIM   +0.0122 [+0.0087,+0.0160]*
clean - OFF LPIPS  -0.0000986 [-0.0001259,-0.0000734]*
```

Thus §22 generalises at higher power: clean memory genuinely improves over OFF. This is a
record, not a decision input, because the matched three-arm comparison already selected
the simpler fused v1 representation. The raw all-scene pose summary in this artifact is
not admissible. Under the frozen GT gate, clean−OFF is +0.00420
[-0.00455,+0.01234], median +0.00134 [-0.00604,+0.01372], 11/25 wins: null.

## 24. Full-power retained fused-v1 component

The corrected guarded three-arm evaluation was rerun on all 60 unique held-out scenes
(`expanded: true`, `limit: 0`, `n_available = n_evaluated = 60`). All arms were rendered
in one process from the same immutable stage-1 base; the v1 and clean incremental states
passed the exact shared-tensor guards before rendering.

```
contrast       mean dPSNR                 median dPSNR               wins      dLPIPS
v1 - OFF      +0.656 [+0.453,+0.887]*    +0.437 [+0.269,+0.535]*    51/60    -0.000200*
clean - OFF   +0.287 [+0.161,+0.430]*                                  43/60    -0.000099*
clean - v1    -0.369 [-0.542,-0.215]*    -0.227 [-0.342,-0.087]*    14/60    +0.000101*
```

The n=16 decision not only survives but strengthens: fused v1 is the retained memory
component, with a broad median gain rather than a mean carried by a few scenes. Clean is
now measurably worse than v1 on PSNR, SSIM and LPIPS at n=60.

Pose does not supply an additional claim. After the frozen GT gate and joint-measurability
filter, v1−OFF is +0.00329 normalized Sampson
[-0.00727,+0.01433], median −0.00306
[-0.01080,+0.01972], 14/25 wins (sign p=0.69): null.

## 25. Pose-verified generated writes fail the bounded oracle gate

The decisive experiment was run before building a learned verifier. The harness fixes the
recent context at three frames, keeps observed-frame memory immutable, uses the same
trained fused-v1 reader and sampler in every arm, and varies only the generated-write
ledger:

- `no_generated_writes`: retain observed memory; never commit generations.
- `commanded`: commit generated RGB/depth at its nominal camera.
- `corrupted_write`: commit the same evidence with a deterministic 5° rotation and
  translation equal to 5% of the observed-camera extent.

Thus `commanded - no_generated_writes` measures natural generated-write contamination.
`commanded - corrupted_write` is the controlled oracle recovery: nominal placement is the
known correct pose for the injected corruption. Six scenes were selected without model
outputs by cycling through original-valid, expansion-valid, original-invalid and
expansion-invalid strata in hash order. Four are GT-valid and two invalid. Horizon 1 is a
provable null because no generated write has yet been read; all three arms were exactly
equal on every scene.

```
horizon   commanded - no writes             correct - corrupted write
1         +0.000 [+0.000,+0.000]             +0.000 [+0.000,+0.000]
5         -0.087 [-0.248,+0.060]             +0.058 [-0.284,+0.381]
10        -0.435 [-0.910,+0.022]             -0.344 [-0.632,-0.064]*
20        -0.081 [-0.354,+0.297]             -0.267 [-0.674,+0.216]
```

All predeclared gates fail:

1. Natural commanded writes do not significantly degrade against observed-only memory.
   The h10 estimate is large and nearly significant, but it recedes at h20 rather than
   growing; at h20 commanded wins only 1/6 scenes, with a wide CI.
2. Correct placement does not recover the 0.20 dB material threshold. At h10 it is
   significantly **worse** than deliberately corrupted placement.
3. Recovery does not grow with horizon.

**Decision: close pose-verified writes and do not build the learned verifier.** Generated
evidence can harm long rollouts, but write-pose correctness is not the causal lever in
this experiment. The heterogeneous and sometimes inverted corruption response is more
consistent with unstable self-generated content/depth than with spatial misplacement.
`no_generated_writes` is a sensible conservative system default, but this n=6 screen does
not license a positive immutable-memory claim.

Architecturally, this is the measured motivation for sparse latent-KV memory rather than
merely a borrowed SOTA feature: the current write path must estimate depth from generated
pixels before it can place evidence, and correct pose did not repair that content/depth
failure. Wonder-style KV retention routes around this particular failure by storing
native latent attention state without a generated-pixel depth-estimation loop. That
motivates future work; it does not imply the unimplemented KV system would succeed here.

## 26. Primary-task correction: causal walking trajectories (2026-07-31)

The deployment task was clarified after §25: it is not primarily three poses spread over
a scene followed by an unrelated novel target. It is a temporal trajectory — a human or
agent walking through a world — with bounded recent context and autoregressive feedback.
The spread-view protocol therefore becomes a retention test, not the headline.

The repository already contained temporal RoPE and an old `train_temporal.py`, but that
run trained only the legacy ~57M camera adapters from `view_set_r32`; it did not continue
the successful 604M-parameter Q/K system. Its fixed `[0,3,6] -> 18` artifact also did not
show a temporal gain. It is treated as an obsolete pilot.

Implemented:

- `data/trajectory.py`: ordered recent-context sampler, future-only targets, 1/2/4/8
  horizon mix, geometry-only camera-track continuity gate, fixed rollout schedules and
  target-preserving context corruption;
- `training/train_trajectory.py`: one training core with `train_fullft`, continuing the
  full `qk_fullft_camqk` state in fp32, temporal RoPE ON, 12k/48k causal tuples, EMA,
  cosine decay, safe deltas and dual trajectory/retention tripwires;
- `training/train_trajectory_memory.py`: zero-init fused-v1 re-fit on the selected temporal
  base, rather than transplanting base-dependent stage-2 weights;
- `bench/eval_trajectory.py`: identical geometry schedule for teacher-forced and bounded
  autoregressive modes, temporal ON/OFF ablation, horizon curves and drift;
- primary fidelity split corrected to 48 scenes never seen by the inherited r32 base.

Measured dataset calibration on 100 train scenes: median per-scene adjacent translation
is 0.027 persistent scene units and median rotation 1.37 degrees. Sampler discontinuity
limits are 0.20 and 25 degrees, so they reject cuts rather than normal walking motion.

### 26.1 Zero-shot activation is strongly harmful

Before training, the current Q/K checkpoint was run on one fixed four-step walking
segment with the same context/targets and 10 denoising steps:

```
regime          temporal OFF    temporal ON      ON - OFF
teacher forced      23.239          19.105         -4.134 dB
autoregressive      20.775          16.243         -4.532 dB
```

Autoregressive OFF fell from 24.90 dB at step 1 to 18.40 at step 4; ON fell from 18.68 to
14.39. n=1 is a diagnostic, not a result, but the magnitude settles the implementation
question: temporal RoPE cannot simply be enabled at inference. The old note that “step 0
is the base checkpoint” was false in model-output terms. `None` is exact zero-temporal,
but supplying nonzero frequencies changes every attention block before the model learns
to interpret them.

The Stage-A run is launched as `outputs/trajectory_qk_12k`: 0.60B trainable parameters,
12k optimizer steps/48k tuples, cosine LR, context corruption warmup, EMA, 1k deltas. Its
predeclared gates and subsequent matched control are in `PLAN_TRAJECTORY_WORLD_MODEL.md`.

### 26.2 Temporal Q/K learns the task but does not clear the anti-drift gate

The run completed 12k steps. Its cheap held-out trajectory tripwire peaked at +2.63 dB
at step 4k, remained inside a 0.04 dB range through step 12k, and ended 0.03 dB below the
4k point. Training loss continued to decline. Thus 4k is an empirical early-stopping
ceiling for this adaptation objective, not a parameter-count claim: task validation had
saturated while the optimization surrogate kept moving.

The proper four-scene, 20-step bounded rollout diagnostic gave:

```
teacher-forced mean       +0.976 dB
autoregressive mean       +0.545 dB  CI crosses zero
horizon 20                +0.780 dB  [0.164,1.819], 4/4 wins
drift reduction              7.2%    FAIL against 20% bar
spread-view retention     +0.16 dB
```

Temporal Q/K is retained as the task-aligned base: it improves next-frame and late-horizon
quality. It is not called an anti-drift mechanism because the rollout-vs-teacher gap barely
shrinks.

### 26.3 Actual-image rollout boards and one-step self-history gate

`bench/visualize_trajectory_rollout.py` renders the exact bounded schedule with three
observed context images, GT targets, autoregressive generations at horizons 1/5/10/20,
and commanded camera motion. Translation is reported both as net displacement and path
length in persistent scene-normalised units; rotation is the net camera-frame angle.
Two non-selected pure-heldout boards are stored under `outputs/trajectory_qualitative/`:

- `fe4bd0d4_rollout_board.png`: frames 36/38/40 -> 42...80; h20 path 0.814,
  net translation 0.789, net rotation 6.13 degrees.
- `fff0ff50_rollout_board.png`: frames 42/44/46 -> 48...86; h20 path 1.202,
  net translation 1.091, net rotation 58.1 degrees.

The next bounded experiment implemented a causal Lyra-style approximation: on half of
tuples, select one non-initial context frame, noise it at sigma in [0.05,0.5], reconstruct
it with one detached model forward from its strictly earlier prefix, then supervise the
ordinary clean future using the reconstructed frame as history. The 4k-budget run paused
after its complete step-1k checkpoint before further spending.

Against the temporal-Q/K starting checkpoint on the same n=4, 20-step protocol:

```
metric                     self-history 1k - temporal Q/K
teacher-forced mean        +0.143 [+0.041,+0.225], 3/4 wins
autoregressive mean        +0.213 [-0.235,+0.801], 3/4 wins
horizon 20                 -0.443 [-1.437,+0.219], 1/4 wins
rollout drift              +0.199 [-0.257,+0.696] (higher is worse)
relative drift change      -3.7% (worse; required +20% reduction)
```

The qualitative warning was real: on `fe4bd0d4`, self-history reached 15.26 dB at h10
but collapsed to 11.57 dB at h20 versus temporal Q/K's 13.45. The full gate confirms that
one-step self-history improves clean-history prediction but does not improve closed-loop
stability. **Decision: stop at 1k; do not resume to 4k and do not run the equal-compute
control.** A control cannot rescue a treatment that already fails against its own starting
checkpoint.

## 27. Full-backbone recovery × observed-retrieval factorial (2026-08-02, running)

### 27.0 Lyra 2.0 release-code correction

The newly released local Lyra 2.0 training code was audited at commit `9fffc9a`; details are
in `LYRA2_TRAINING_CODE_AUDIT.md`. Its config declares a selective whitelist, but the base
constructor subsequently calls `self.net.requires_grad_(True)` with LoRA disabled and its
optimizer includes all gradient-enabled parameters. Thus the released executable recipe is
full-DiT training, although comments still name the intended source experiment K/Q-only.

Lyra's memory is also materially stronger than a generic history reader: five target-
coverage-retrieved native appearance slots enter main self-attention, while per-source
warped coordinate/depth fields bias Q/K and a recent RGB-D warp conditions the target.
Its recovery data is one 80-frame self-generated chunk followed by a clean target, used on
about 35% of iterations in expectation. These findings reinforce full-backbone B/D
co-adaptation through at least 4k. They do not license changing the live reader mid-run;
coverage retrieval plus geometry-only Q/K bias is the controlled fallback if B/D is null.

The earlier frozen/transplanted reader failures do **not** establish that persistent
memory is useless. The decisive experiment now jointly updates the full 5B DiT and uses
four arms from one immutable temporal parent: clean/recovery recent histories crossed
with an observed native-token reader OFF/ON. All arms instantiate the same K=2 sparse
source-cross-attention module; OFF is the exact scale-zero execution-graph control.
Target tuples, noise streams, optimizer recipe, 30/70 ordinary/revisit sampling,
gradient accumulation 4, and the 10k cosine schedule are matched.

Replay cache v3 contains aligned H0 `[GT,GT,GT]`, H1 `[GT,GT,G]`, H2 `[GT,G,G]`, and
H3 `[G,G,G]` contexts with one immutable clean target. It records the RGB feedback mode,
sampler-native generated latents, decoded audit images, per-slot PSNR/LPIPS/latent
statistics, seeds, content hashes, and machine-readable causal ancestry. The full
984-scene/30-step cache is materializing; incomplete manifests are refused by training.

The staged gate is now executable rather than nominal:

1. all arms pause at step 250;
2. the same four pure-heldout scenes run a mechanical-only smoke gate, with no scientific
   conclusion permitted and a deliberately broad 2 dB catastrophic threshold;
3. exact optimizer/RNG continuation resumes every passing arm to the step-1,000 pause;
4. at least 16 scenes measure provisional recovery, reader and interaction effects using scene-level
   bootstrap, including rollout mean, horizon 20, teacher retention, LPIPS, SSIM, raw and
   matched drift, and GT-headroom-filtered revisit/path consistency.

The step-1k result is explicitly provisional: it cannot reject the reader. Regardless of
the early sign, matched full-backbone arms B (recovery, reader OFF) and D (recovery, reader
ON) continue through at least step 4k on the unchanged 10k schedule. This corrects an
over-aggressive early-stop rule that would have treated short co-adaptation like the old
frozen/transplanted memory tests. A negative memory conclusion is permitted only from the
matched B/D trajectory and revisit comparison at 4k or later.

The archive opportunity screen was corrected before training. The first pilot measured
three frames near the first visit, whereas the implemented reader retrieves K=2 sources
from the full eligible immutable archive. The frozen screen now tries up to 20
camera-valid revisits per scene and selects by depth estimated from observed frames only;
model images and model scores cannot affect membership. Reader evaluation is forbidden
unless at least 16 scenes provide at least 10% exclusive support.

One runtime reliability risk was tested against the real CUDA optimizer, not only a
mock. A `PagedAdamW8bit` state was stepped, serialized, restored through the new managed
buffer path, and stepped again. Both uint8 moments were bit-exact after restoration,
retained `is_paged=True`, advanced from step 1 to 2, kept LR `2e-5`, and produced finite
parameters. This closes the resume-only OOM mechanism where bitsandbytes otherwise
materialized all full-backbone moments as ordinary CUDA tensors.

Two pre-training reader audits prevented unfairly weakening the memory arm. The original
closest-pose K=2 rule selected adjacent frames on every valid held-out revisit (40/40;
median temporal gap 1), so the second slot duplicated the first. A deterministic
six-frame source-keyframe spacing rule raises the median gap to 6 while retaining a
fallback for short valid archives. Also, archived patch embeddings previously bypassed
all positional encoding. The source K and target Q paths now use Wan's native spatial
RoPE; archive time remains the identity/unordered sentinel. This is still the declared
free reader—there is no depth, point, epipolar or target-warp bias—but source appearance
is no longer treated as a positionless bag of patches.

## 28. Proper full-backbone recovery × fused-memory experiment (2026-08-02, building)

Section 27's archive-reader experiment did not open: the frozen opportunity screen found
only 12 scenes with at least 10% archive-exclusive support, below the predeclared 16-scene
minimum. This is an evaluation-opportunity failure, not evidence that a native-token
archive is ineffective. It cannot be the next primary thesis experiment on this split.

The immediate experiment instead retains the two mechanisms that already have positive
GaussVerse evidence: fused-v1 warped appearance and two-sided correspondence. Historical
matched results were +0.526 dB for fused memory and +0.620 dB for two-sided correspondence
on the spread-view protocol; the latter improved frontier PSNR by +0.198 dB over the
one-sided variant. The failed counterfactual router, clean source-separated replacement,
and generated-write experiments do not invalidate this fixed branch.

The corrected 2x2 design is:

```text
A  H0 clean control       + identical mem/corr graph at scale 0
B  H0/H1/H2/H3 recovery  + identical mem/corr graph at scale 0
C  H0 clean control       + fused-v1/two-sided evidence at scale 1
D  H0/H1/H2/H3 recovery  + fused-v1/two-sided evidence at scale 1
```

Every arm starts from `trajectory_qk_12k/ckpt_last.safetensors`, unfreezes the complete
DiT, trains in bf16 with paged AdamW8bit, gradient accumulation 4, LR 5e-6 with 100-step
warmup and cosine decay to LR/10, no EMA, and receives 10,000 optimizer steps unless a
NaN/OOM or explicit catastrophic retention tripwire fires. Step-1k checkpoints are not
scientific stopping points. This directly corrects the project's repeated mistake of
drawing architecture conclusions from short/frozen co-adaptation.

Only `mem.*` and `corr.*` tensors are initialized from
`v1_fused_2k_matchedbase/ckpt_last.safetensors`; a strict prefix/shape/finite loader makes
it impossible for this checkpoint to overwrite the temporal backbone, camera branches,
FiLM, or null context. A scale-zero test proves that trained memory and correspondence
branches are exactly equivalent to OFF.

Replay-v3 is complete with 984 aligned RGB-feedback tuples. The mature recovery mixture
is 65/15/12/8 for H0/H1/H2/H3, i.e. 35% generated-history exposure—comparable to Lyra's
released self-augmentation rate—after conservative 90/7/2/1, 80/12/6/2, and 70/15/10/5
warmup stages. Unlike the earlier replay2 run, clean context remains the majority anchor.

Warp/correspondence evidence is being precomputed for every tier from the history the
model really consumes. Observed slots use observed RGB; generated slots are decoded from
the stored RGB-feedback latents. The clean target contributes camera/intrinsic metadata
only and is never an appearance source. The cache is bf16 (matching adapter input),
content-hashed, tied to the replay manifest, two-sided, z-buffer fused, atomic, and fully
validated before training. A first pilot exposed that decoding the re-encoded feedback
latent added a second VAE round trip absent from deployment. Cache v2 instead decodes the
stored sampler-native latent, matching the actual generated RGB memory path; clean evidence
remained bit-exact while every generated tier changed, as expected. A corrected CUDA pilot
passed for all four tiers. The complete 984-entry v2 build is active and chained directly
to training; training refuses to start until its manifest is complete. The full
implementation suite passes: 369 tests, 2 skipped.

The final comparison is scene-paired at n>=16 and measures teacher mean, rollout mean,
horizons 1/5/10/20, LPIPS proxy, RGB-Gaussian SSIM, raw/matched drift, two-path revisit
fidelity, cross-path consistency, pose on the frozen GT-valid subset, latency, and VRAM.
C-A and D-B estimate fixed memory under clean/recovery training; B-A and D-C estimate the
recovery curriculum; D-A is the integrated system effect.

### 28.1 First overnight status and tripwire correction

Cache v2 completed all 984 entries. Arm A completed 10,000 full-backbone updates with
finite checkpoints and no pause; its four-scene safety tripwire ended at trajectory
`+1.56 dB` and unordered retention `+0.25 dB` relative to its starting parent. These are
safety diagnostics, not the final n>=16 result.

Arm B was incorrectly auto-paused at step 1,000 by the inherited `0.30 dB` threshold:
trajectory was `-0.327 dB` and retention `-0.312 dB`. The corrected plan declared a broad
`2 dB` catastrophic-only threshold because short checkpoints cannot reject recovery. A
0.01--0.03 dB threshold crossing on a four-scene tripwire is not a catastrophic event.
The factorial launchers now default to `2.0 dB`. B's 1,499 checkpoint tensors were all
readable and finite; its paged optimizer plus NumPy/PyTorch RNG streams restored exactly,
and training resumed from step 1,000. The resume reports 5.08B backbone and 5.11B total
trainable parameters. A remains immutable and complete; C and D remain chained after B.

### 28.2 Full training complete; definitive evaluation launched

All four arms reached 10,000 full-backbone optimizer updates without a catastrophic
tripwire after the threshold correction. Their final four-scene safety diagnostics were:

```text
arm   trajectory delta   unordered-retention delta
A          +1.56 dB              +0.25 dB
B          -0.58 dB              +0.00 dB
C          +1.33 dB              +0.12 dB
D          -1.01 dB              -0.36 dB
```

This diagnostic suggests that the generated-history curriculum may trade away ordinary
quality, while fixed fused memory may impose a smaller additional cost. It is not the
factorial result: four scenes, one cheap tripwire protocol, no paired CI, and no H20
closed-loop measurement. The predeclared n=16 evaluation is now running with identical
schedules/seeds: teacher plus RGB-feedback H20 rollout, memory OFF for A/B and ON for C/D,
followed by two-path consistency, the scene-paired factorial gate, and the frozen pose gate.

### 28.3 Decision-gate audit during evaluation

The live evaluation exposed a latent protocol-guard bug before the factorial result was
produced. `eval_trajectory` records memory as the strings `"off"` and `"on"`; the gate
coerced this field with `bool(...)`, under which `bool("off")` is true. That would have
rejected the correct A/B artifacts (and could have obscured a genuinely inverted flag).
The gate now compares the recorded strings exactly.

The final decision is additionally bound to training provenance rather than trusting
filenames. Each arm must point to its expected run directory, have a readable checkpoint
and `meta.json` at exactly step 10,000, match the full-backbone/bf16/paged-AdamW8bit/
accumulation-4 recipe, use its declared replay and memory-scale factor, and share the same
immutable parent. Revisit artifacts must use the exact checkpoint evaluated by their
forward artifact. Evaluation flags and per-scene schedules are checked identically across
all arms. The gate also rejects intermediate, tripwire-stopped, or planned-pause
checkpoints even when their run directory has final-looking metadata. The complete
repository suite passes (372 passed, 2 skipped) while the GPU renderer continues.

### 28.4 Two-path evaluation recovery guard

The forward H20 evaluator already persisted every completed scene and validated a strict
schedule/config prefix on `--resume`. The chained two-path evaluator also persisted rows,
but had no resume mode: after an interruption the orchestrator would rerun it from zero
and overwrite partial work. It now resumes only after verifying the exact configuration,
geometry-selected scene/schedule prefix, two archive-source indices, finite core metrics,
and all LPIPS fields when requested. The factorial orchestrator passes `--resume` whenever
a partial revisit artifact exists. The complete suite passes (373 passed, 2 skipped).

### 28.5 Final-result completeness and pose binding

The factorial result now emits absolute arm summaries and paired contrasts at H1, H5,
H10, and H20, rather than retaining only H20. Every paired result includes the
predeclared mean/bootstrap CI and sign count plus median and 10%-trimmed mean; rollout
step latency, denoiser latency, and peak allocated VRAM are recorded descriptively and do
not enter selection.

The terminal pose evaluator previously trusted the supplied frame folders and did not
independently validate the four forward schedules or checkpoint identities. It now reuses
the exact forward-protocol and complete-10k provenance guards, requires 16 unique aligned
scene schedules, and fingerprints every H20 PNG actually scored. This changes no metric
or threshold; it prevents stale-frame or cross-checkpoint pose attribution.

The post-gate qualitative path now consumes those exact saved evaluator PNGs instead of
rerendering a checkpoint with a second stochastic pass. On three result-independent named
scenes (`fe4bd0d4`, `fe30a865`, `fff0ff50`) it emits observed-context/camera-path/
GT/generated boards, H20 videos, camera-motion JSON, and the 16-scene horizon curve. It
shows A versus the selected treatment, or all A/B/C/D arms if no treatment passes. The
complete suite passes (375 passed, 2 skipped).

A factorial-specific final manifest replaces the inappropriate historical artifact-
recovery manifest for these modern runs. It deterministically recomputes the entire gate,
checks H20 pose/frame binding again, embeds the four complete training metadata blocks,
hashes every decisive JSON/log/meta artifact, and resolves the selected checkpoint plus
matched A control. If no treatment passes, it records the immutable temporal parent as
the honest continuation rather than silently promoting a failed factorial arm. The
complete suite passes (376 passed, 2 skipped).

### 28.6 Definitive evaluation progress

Arm A's forward artifact completed the frozen 16-scene schedule. Independent validation
confirmed 16 unique scenes, three contexts plus 20 targets per scene, finite teacher and
rollout metrics, and all 1,280 aligned generated/ground-truth PNGs (teacher and rollout).
Absolute A values are teacher mean `19.552 dB`, RGB-feedback rollout mean `13.896 dB`,
and H20 `12.102 dB`. These are baseline descriptors, not treatment conclusions. The
chained all-60 geometry-selected two-path evaluation started immediately afterward.

Arm A's two-path artifact then completed all 40 geometry-selected scenes with two sources
per scene, exact protocol/checkpoint binding, and finite PSNR/MSE/LPIPS fields. Its
descriptive means are path-A `16.454 dB`, path-B `22.763 dB`, cross-path `16.886 dB`, and
cross-path LPIPS `0.00421`; these are within-arm absolute values and do not estimate a
treatment effect. Arm B forward evaluation launched automatically.

The first real revisit completion exposed a restart-integrity distinction: the frozen
schedule has 40 eligible scenes, while the scientific minimum is 16. The orchestrator's
generic `len(rows) >= 16` check could have treated an interrupted 16/40 artifact as fully
complete on restart. Revisit artifacts now persist `expected_rows` before model loading
and a `complete` flag after every row; both the orchestrator and final gate enforce exact
completion. This does not change the n>=16 inferential rule—it prevents silently dropping
eligible scheduled scenes. Focused orchestrator/long-horizon/factorial tests pass (26/26).
The complete repository suite passes (378 passed, 2 skipped).

Arm B then completed the matched 16-scene forward evaluation. All schedules and 1,280
PNGs align with A. Its absolute values are teacher `18.135 dB`, rollout `11.567 dB`, and
H20 `9.118 dB`; these are visibly below A but remain an interim main-effect observation
until the four-arm gate estimates recovery, memory, and their interaction together. B's
revisit run completed 40/40, with the new state transitioning from `complete=false` to
`true` only on the final row. Absolute B path-A/path-B/cross-path means are
`14.934/21.086/15.714 dB` and cross-path LPIPS is `0.00449`. Arm C (clean histories,
fused memory ON) launched automatically; its higher live VRAM footprint (~19.7 GB versus
~14.7 GB OFF) is expected from VGGT/warp evidence construction and will be reported from
the evaluator's recorded peak allocation.

Arm C completed and independently validated all 16 aligned forward scenes and 1,280 PNGs.
Its absolute teacher/rollout/H20 values are `20.356/14.363/12.275 dB`, descriptively above
A's `19.552/13.896/12.102`; paired C-A inference remains deferred to the complete gate.
C's revisit artifact completed all 40 rows with finite metrics and `complete=true`:
path-A/path-B/cross-path means `18.332/23.435/18.911 dB`, cross-path LPIPS `0.00331`.
Arm D (recovery curriculum plus fused memory) launched automatically.

### 28.7 Complete 10k factorial result — select C; recovery curriculum fails

All four forward artifacts contain the same 16 scenes/schedules and all four revisit
artifacts contain the same 40 geometry-selected schedules. Every checkpoint passed the
complete-10k/full-backbone/shared-parent provenance guard. The deterministic gate selects
**C (clean histories + fixed fused warp/two-sided correspondence)**:

```text
contrast / metric                 mean paired delta       95% scene-bootstrap CI   wins
C - A rollout mean                  +0.467 dB              [+0.249,+0.712]          14/16
C - A teacher mean                  +0.805 dB              [+0.595,+1.020]          16/16
C - A H1                            +0.685 dB              [+0.476,+0.913]          16/16
C - A H5                            +0.378 dB              [+0.066,+0.709]          13/16
C - A H10                           +0.532 dB              [+0.181,+0.948]          11/16
C - A H20                           +0.173 dB              [-0.144,+0.472]          11/16
C - A rollout LPIPS                -0.000152               improvement, CI excludes 0 14/16
C - A rollout RGB-Gaussian SSIM     +0.0152                [+0.0068,+0.0234]        14/16
C - A revisit path-B PSNR           +0.672 dB              [+0.299,+1.050]          32/40
C - A same-target cross-path PSNR   +2.025 dB              [+1.603,+2.463]          38/40
```

The central tendency is not driven by one tail: rollout mean/median/10%-trimmed mean are
`+0.467/+0.398/+0.433 dB`. The late-horizon effect narrows and H20 is not individually
significant, but the predeclared memory gate uses rollout mean plus revisit safety and
passes. Fixed recent-window fused evidence therefore survives proper 10k full-backbone
co-adaptation; this does **not** instantiate persistent archive memory.

The recovery result is decisive and negative:

```text
B - A rollout mean    -2.330 [-2.937,-1.803] dB,  0/16 wins
B - A teacher mean    -1.417 [-1.657,-1.176] dB,  0/16 wins
B - A H20             -2.984 [-3.748,-2.221] dB,  0/16 wins
D - B rollout mean    +0.039 [-0.244,+0.417] dB,  5/16 wins
D - A rollout mean    -2.290 [-2.737,-1.834] dB,  0/16 wins
D - A H20             -3.463 [-4.146,-2.790] dB,  0/16 wins
```

At H1 fixed memory still helps D-B (`+0.597 [+0.064,+1.083] dB`), but by H20 it hurts
(`-0.479 [-0.796,-0.161] dB`). The recovery×memory interaction on rollout mean is
negative (`-0.427 [-0.808,-0.018] dB`): the H0/H1/H2/H3 replay curriculum prevents the
model from retaining memory's long-horizon benefit. The cheap four-scene training
tripwire badly understated this failure (B ended at only `-0.58 dB` there versus
`-2.33 dB` on the definitive rollout), reaffirming that it was a safety trace only.

The recovery null is scoped to the two teacher-training formulations actually tested:
the H0/H1/H2/H3 replay curriculum and one-step self-history corruption, at this model,
data scale and optimizer recipe. It does not establish that every self-forcing-style
method fails, and it does not test Wonder's noisy-context exposure inside DMD distillation.

Two caveats constrain C's claim. First, every reported horizon remains at or above A, but
the benefit attenuates late and the fitted raw and color-matched slope statistics worsen
by `+0.335 [+0.090,+0.583]` and `+0.345 [+0.076,+0.636] dB`. The precise conclusion is
**memory raises the curve but is not an anti-drift mechanism**—not that memory makes any
reported horizon worse than baseline. Second, pose is unpowered: only 2 of 6
frozen-GT-valid scenes remain measurable
for C versus A, giving Sampson `-0.0042 [-0.0105,+0.0021]`, 1/2 wins. No camera-pose claim
is made.

Descriptive rollout cost is material: C averages `3.61 s/frame` versus A's `3.09`, and
peak allocated VRAM is `18.73 GB` versus `12.83 GB`. D is slower still (`4.83 s/frame`).
These are sequential-run measurements, not a controlled speed benchmark.

The final artifact manifest independently recomputed the gate and selected
`outputs/trajectory_factorial_c/ckpt_last.safetensors`, fingerprint
`07a31e99fd49db6c7c717a5c`, with A as its matched control. Exact-render boards/videos for
the three result-independent scenes and the 16-scene horizon curve are under
`outputs/trajectory_factorial_visuals/`.

### 28.8 Does observed-frame evidence explain memory-gain attenuation?

This post-result mechanism analysis uses no new rendering. In the bounded three-frame
RGB-feedback window, the exact fraction of original observations available before
H1/H2/H3/H4+ is `1, 2/3, 1/3, 0...`. The paired C−A curves give:

```text
mean gain while any observed frame remains (H1-H3)       +0.591 dB
mean gain after the window is all generated (H4-H20)     +0.445 dB
early-minus-late                                         +0.146 [-0.074,+0.362] dB
scene-level gain slope / observed fraction               +0.268 [-0.017,+0.545] dB
mean-curve Pearson / Spearman                            +0.398 / +0.310
```

The direction fits the hypothesis that warped generated content weakens memory's benefit,
but neither scene-level CI excludes zero. Horizon and observed fraction change together,
so this is not causal separation. Report it as **suggestive attenuation**, not an
established mechanism. Artifacts: `outputs/trajectory_factorial_memory_real_fraction.json`
and `outputs/trajectory_factorial_visuals/memory_gain_vs_real_fraction.png`.

## 29. Wonder-style persistent coordinate field — pre-registration (2026-08-04)

This section is frozen before renderer or injection implementation. The immutable parent
for both arms is selected factorial arm C,
`outputs/trajectory_factorial_c/ckpt_last.safetensors`, fingerprint
`07a31e99fd49db6c7c717a5c`. This is a controlled reproduction of Wonder's disclosed
conditioning family, not an exact reproduction of its undisclosed renderer parameters.

### 29.1 Coordinate field

Each trajectory has one persistent anchor: its **first observed context camera**. Camera
poses are rigidly expressed in that anchor's OpenCV frame and translations are divided by
the immutable scene scale. The anchor never moves when the target changes; resetting at
each target is forbidden because it would destroy cross-frame world correspondence.

The frozen scaffold is a rectilinear 3D grid in anchor coordinates:

```text
x: -3.0 ... +3.0, spacing 0.5     (camera-right)
y: -2.0 ... +2.0, spacing 0.5     (camera-down)
z: +0.5 ... +6.0, spacing 0.5     (camera-forward)
```

All adjacent x/y/z vertices are joined. Vertex/segment color is the direct affine
normalization of `(x,y,z)` over those fixed bounds, so the same world point has the same
RGB in every camera. Segments use depth-tested deterministic rasterization with a frozen
1.5-pixel radius. The background is an environment-direction field: each pixel's ray,
rotated into the anchor frame, maps to `0.5*(direction+1)`. Output is RGB in `[-1,1]` at
384x640. It depends only on camera, intrinsics, scene scale, anchor, and frozen constants;
RGB observations, target pixels, depth, VGGT, and model outputs are forbidden inputs.

The renderer manifest records all bounds/spacing/radius/color rules, anchor index,
resolution, code fingerprint, and per-render camera indices. The required visual board
shows three observed RGB contexts, their coordinate fields, H1/H5/H10/H20 coordinate
fields, the camera path, normalized net/path translation, and rotation.

### 29.2 Wonder-aligned injection and matched arms

The complete context+target coordinate-field video is encoded by the frozen Wan VAE to
the native 48-channel latent grid. A separate `Conv3d(48 -> dim)` uses the DiT's exact
patch kernel/stride and is summed with the ordinary video patch embedding before block 0:

```text
x = video_patch(latents) + lattice_scale * lattice_patch(lattice_latents)
```

`lattice_patch` weight and bias are zero-initialized. At scale 1 its output is initially
zero but gradients into its weights are live; at scale 0 it is bitwise branch-absent and
receives zero gradient. No adapter stem, target-only residual, synthetic correspondence
fill, or lattice attention bias is part of the primary study.

The sole expensive comparison is:

```text
L0  selected C parent + lattice module instantiated, scale 0
L1  same parent/initial module/RNG/data, lattice scale 1
```

Both retain C's fixed fused warp and two-sided correspondence at scale 1, use clean H0
`replay-control` histories, unfreeze the full 5B DiT, train bf16 with paged AdamW8bit,
gradient accumulation 4, LR 5e-6 (100-step warmup, cosine to LR/10), no EMA, and receive
10,000 optimizer updates. The tuple/noise/batch streams and parent are identical. Steps
250/1k are mechanical/safety checks only; a healthy arm is not scientifically stopped
before 10k.

Required implementation gates before launch:

- same scaffold point has bit-identical color across cameras;
- translation parallax has the analytically correct sign and depth ordering;
- pure rotation changes the environment field according to rotated ray direction;
- repeated calls are bitwise deterministic; permutation only permutes renders;
- a consistent world-gauge transform preserves the rendered video;
- invalid/behind-camera segments produce no NaN or out-of-bounds write;
- no image/depth argument exists on the renderer path;
- lattice scale 0 is bitwise output-identical to branch absence;
- zero-init scale 1 is output-identical at initialization but projection gradients live;
- L0/L1 train the complete backbone and checkpoint loading verifies the selected parent.

### 29.3 Frozen evaluation and decision

Trajectory evaluation is the same 16 scenes, contexts, targets, seed 7, RGB feedback,
384x640, 30 denoising steps, and H1/H5/H10/H20 schedule used in section 28. Scene is the
replication unit; report paired mean/bootstrap CI, median, 10%-trimmed mean, and sign count.

The lattice-specific binding test uses one frozen H10 tuple from each of those 16 scenes,
five sigma IDs `(100,300,500,700,900)`, and shared noise. It crosses:

```text
correct                    honest rays + honest lattice
lattice_lie_{small,med,large}  honest rays + lattice rendered at a wrong real camera
rays_lie_{small,med,large}     wrong-camera rays + honest lattice
consistent_lie_{small,med,large} wrong-camera rays + matching wrong-camera lattice
```

Wrong cameras are selected by geometry alone in the frozen rotation bands 0–15°, 15–45°,
and 45–180°. The direct binding statistic is `loss(lattice_lie_large)-loss(correct)`.
It must be exactly zero for L0 and positive with a scene-bootstrap CI excluding zero for
L1. The consistent large-minus-small slope and rays-vs-lattice conflict damage are always
reported; they diagnose control hierarchy but do not replace the direct intervention.

L1 is retained by either predeclared route:

1. **trajectory route:** L1-L0 rollout mean >= +0.10 dB with CI excluding zero, H20
   non-worse (mean >= 0), teacher mean >= -0.15 dB, and raw drift non-worse (mean >= 0
   in the gate's higher-is-better convention); or
2. **binding route:** the direct L1 lattice-lie statistic passes, while L1-L0 rollout
   mean and H20 are each >= -0.10 dB and teacher mean >= -0.15 dB.

Pose on the frozen GT-valid subset, LPIPS, SSIM, revisit/path consistency, latency, VRAM,
and path-family strata are mandatory secondary outputs. A blind pre-result amendment in
§29.8 makes pose descriptive and gives it no retain/drop weight; at least eight jointly
measurable scenes are required even for powered interpretation. Section 28 retained only
two. No synthetic
frontier fill, new memory reader, renderer retuning, or extra training arm is opened after
seeing L0/L1. A null/negative result is the controlled thesis finding.

### 29.4 Renderer/injection implementation gate and L0 launch

The renderer and Wonder-aligned input path are implemented. The retained renderer is
`persistent_anchor_coordinate_field_v1`, fingerprint
`5cfa0bc9a1a727b14d12ea6c`. Its visual gate is
`outputs/lattice_visual_sanity/fe30a865_rgb_lattice_camera_board.png`: the frozen
trajectory moves `0.331` scene-normalized units and rotates `167.9°` from anchor to H20;
the same coordinate colors persist, finite-grid lines show depth-ordered parallax, and
the directional environment remains defined after the scaffold leaves most of the view.

The structural gates pass: persistent colors/bounds, analytic rotation field, parallax
sign and near/far magnitude ordering, bitwise determinism and permutation equivariance,
world-gauge invariance, safe behind-camera handling, and an external original anchor for
sliding recent-context windows. The injection uses a separate zero-initialized
`Conv3d(48,3072,kernel=stride=(1,2,2))` before block 0 (0.59M parameters). Scale zero is
bitwise branch absence even with nonzero projection weights; scale one at zero init is
bitwise output-identical and has a nonzero weight gradient. Full/delta checkpoints treat
`lattice.*` as a strict named branch.

The aligned cache contains exactly 984 entries (`1x48x4x24x40`, bf16; 351 MB). An
independent pass read every file and verified shape, finiteness and SHA-256 content
digest. It binds renderer fingerprint `5cfa0bc9a1a727b14d12ea6c`, VAE fingerprint
`09ef0b249ab9ac22cc8a06cb`, and the exact replay manifest. Every entry uses
`initial_ctx[0]` as persistent anchor; zero of 984 reset to the sliding recent window.
This audit caught and replaced an earlier incomplete cache that incorrectly used
`clean_ctx[0]` after the three-step replay prefix.

The complete repository regression suite passes: **394 passed, 2 skipped**. L0 launched
from the guarded selected-C fingerprint `07a31e99fd49db6c7c717a5c`: full 5.08B backbone
unfrozen (5.11B total trainable), fused memory/correspondence scale 1, lattice scale 0,
clean replay-control, accumulation 4, LR 5e-6 cosine, 10k updates. Startup disk preflight
is 20.4 GB against 234.7 GB free. Baseline tripwires are trajectory `20.97 dB` and
unordered retention `10.91 dB`. These are safety traces, not result estimates.

### 29.5 Frozen post-training harness

The lattice evaluator now reads `lattice.*` strictly from the full checkpoint, defaults
to the scale recorded in `meta.json` (L0=0, L1=1), verifies the renderer source hash, and
renders every sliding context/target window in the original trajectory anchor. The exact
§28 forward protocol is pinned: first 16 `fidelity48` scenes in the same order, teacher +
RGB-feedback rollout, temporal on, fused memory on, 30 denoising steps, seed 7, H1/5/10/20,
LPIPS proxy, SSIM, raw/matched drift, latency, VRAM, and path-family strata.

`bench/lattice_binding.py` implements the registered H10 intervention with five shared
sigma IDs and shared target noise: honest, lattice-only lie, rays-only lie, and consistent
rays+lattice lie at small/medium/large wrong real cameras. It records actual rotation and
whether a scene truly contains a camera in each requested band; no absent >45° candidate
is silently described as one. L0 must give an exact-zero direct lattice intervention.

The queued secondary artifacts are H20 Sampson on the frozen GT-only validity subset and
the all-eligible two-path revisit protocol. Both paths in a revisit share path A's first
observation as lattice anchor; independent per-path anchors are forbidden because they
would change coordinate colors at the shared target. The paired revisit report orients
PSNR, MSE and LPIPS so positive always means L1 improves.

`bench/run_lattice_experiment.py` will run the two forward artifacts, both binding
artifacts, pose, revisits and `bench/lattice_gate.py` after the guarded L0→L1 training
chain completes. The gate implements the two pre-registered retention routes without
post-result rescue arms. Complete repository suite: **401 passed, 2 skipped**.

### 29.6 Pre-registered interactive follow-ups

These measurements run only after the primary L0/L1 gate. The gate selects L1 when
`retain_lattice=true`; otherwise L0 is the shipped checkpoint. Training loss and
intermediate checkpoints cannot select the arm.

The denoising-step cliff evaluates that one selected checkpoint at 30/15/8/4 steps on
the exact frozen 16-scene H20 protocol (teacher + RGB-feedback rollout, temporal and
fused memory on, seed 7). Every low-step statistic is the scene-paired low-minus-30
delta. A lower-step setting is non-inferior only when the bootstrap CI lower bound is
above **-0.25 dB** for rollout mean, above **-0.25 dB** at H20, and above **-0.15 dB**
for teacher history. The smallest setting passing all three is the recommended operating
point; if none pass, 30 steps remains the result. H1/H5/H10, LPIPS proxy, SSIM,
raw/matched drift and latency are mandatory secondary outputs.

The long-rollout evaluation runs both L0 and L1 at H50 on all 16 frozen scenes with
identical schedules, RGB feedback, 30 denoising steps, and H1/H5/H10/H20/H50 reports.
It reports scene-paired L1-minus-L0 for rollout/teacher fidelity, each horizon, perceptual
proxy, SSIM and drift. This is a matched durability measurement, not another lattice
retention gate. The selected arm additionally dumps aligned generated and ground-truth
frames for the thesis visual board. The harness refuses fewer than 16 eligible scenes;
geometry-only audit found all 16 eligible.

The guarded orchestration and both paired analyzers are covered by the complete
regression suite: **405 passed, 2 skipped**. The post-training watcher imports the current
orchestrator only after L1 exits, so these frozen follow-ups are part of the live queue.

### 29.7 Thesis visuals and scripted control demonstration

The post-training chain builds all figures from frozen artifacts without rerendering:
an L0/L1 H50 mean-PSNR curve, a paired 15/8/4-minus-30 cliff plot with the registered
non-inferiority margins, and boards for `fe30a865` and `fe4bd0d4` containing the three
real observations plus GT/selected-model H1/H5/H10/H20/H50 frames. Every target tile
reports actual translation divided by persistent scene scale and net camera rotation.
The manifest binds the figures to the gate, H50 JSONs, cliff JSON and scene list.

The scripted fallback is a genuine arbitrary-pose rollout rather than a renamed dataset
trajectory. `W/S` move along camera-local forward, `A/D` strafe, and `Q/E` yaw by 10°;
translation is 0.01 persistent scene-scale units per key. The frozen script is
`W,W,W,Q,Q,W,W,E,E,W,S,S` from `fe30a865`'s three observed H50-prefix cameras. At each
step the selected complete checkpoint:

- constructs target-relative Plücker rays from the commanded SE(3) pose;
- preserves the original first observation as coordinate-field lattice anchor;
- uses ordered temporal RoPE with the observed prefix's true frame spacing;
- rebuilds VGGT depth, fused warp and two-sided correspondence from the latest three
  RGB frames (including generated frames) at their commanded poses;
- feeds the generated RGB frame back and evicts the oldest recent frame.

The demo automatically uses the smallest denoising-step count accepted by the frozen
cliff rule. It emits every generated PNG, a path-annotated MP4 and a manifest containing
checkpoint fingerprint, commands, all commanded c2w matrices, conditioning branches,
renderer fingerprint and feedback policy. It makes no GT-fidelity or real-time claim.
The complete implementation regression suite passes: **410 passed, 2 skipped**.

### 29.8 Blind safeguards added while L0 trains

These amendments were frozen with L0 below step 1k and before any L1 weights or result
existed.

1. **Parent binding preflight.** Before L1 allocates the model, factorial C must pass a
   16-scene H10 graded-real-camera lie battery at 384x640. Temporal RoPE, fused memory and
   two-sided correspondence stay active; five sigma IDs and target noise are shared. The
   criterion is both `large lie - correct` and `large lie - small lie` with CI95 lower
   bound above zero. Failure blocks L1 because the rays/lattice conflict matrix would lack
   a bound parent reference.
2. **Pose policy.** H20 frozen-GT pose is always reported, needs at least eight jointly
   measurable scenes for powered interpretation, and never enters retain/drop. The
   decision is trajectory/teacher fidelity plus binding/revisit safety.
   Revisit safety is the same margin already used by the factorial: L1−L0 path-B revisit
   PSNR and same-target cross-path PSNR must each be at least −0.10 dB. The gate consumes
   the checkpoint-bound revisit artifact and applies this condition to both retention
   routes.
   The plan's previously non-numeric path-family safety clause is also made executable
   while blind to L1: a family enters retention only with at least eight frozen scenes,
   and its L1−L0 rollout mean must be at least −0.10 dB. The actual geometry-only counts
   are 16 turn-plus-translation, 14 lateral, 4 forward, 2 out-and-back, 2 variable-speed,
   and 0 loop-closure; the last four are reported but cannot gate.
3. **L0 continuation smoke.** The first five frozen scenes compare completed L0 against
   the already-rendered immutable C artifact. It is a mechanical guard, not efficacy
   inference; only a mean drop worse than 2 dB on rollout, teacher or H20 blocks L1. It
   also records the free extra-10k-continuation reference.
4. **Shared data stream.** Both wrappers use train seed 0, identical replay-control tuple
   order, replay-v3, warp-v2, lattice-v1, parent, optimizer, LR schedule and 10k budget.
   The preflight hashes these fields and refuses a mismatch. Only output directory and
   lattice scale differ.
5. **Within-L1 knockout.** The complete co-adapted L1 is evaluated with lattice scale
   zero. `L1 live - L1 scale0` measures inference-time dependence; it is secondary and
   cannot replace or overturn marginal L1−L0.

The full-backbone interpretation is also fixed. The earlier full-5B retirement was a
spread-view result under a different sampler. The trajectory factorial contains **no
frozen arm**: it establishes clean-vs-replay and memory-ON-vs-OFF among matched
full-backbone arms, not full-vs-frozen superiority in this exact regime. External
backbone-adaptation evidence remains the matched n=60 Q/K-control result (+0.370 dB, with
its qualified pose result). All A/B/C/D arms did run trajectory and unordered-retention
tripwires (§28.2); those four-scene traces were safety guards, never selection metrics.

L1 is a fair strong test at one documented operating point: full backbone, pixel-aligned
VAE coordinate field and train-in injection. A positive result is a matched controlled
effect with binding instrumentation absent from Wonder; a null is strong for this
renderer/scale/density/coloring, not a refutation of all coordinate-field designs.

Implementation audit after these safeguards: **423 passed, 2 skipped**. The live process
order remains L0 → preflight (parent binding + L0 smoke + stream manifest) → L1 → frozen
evaluation; no watcher restart or concurrent model allocation was introduced.

### 29.9 Thesis-delivery audit artifacts

The architecture record was collapsed to one measured system contract. The selected
checkpoint is factorial C (`07a31e99fd49db6c7c717a5c`): full Wan2.2-TI2V-5B fine-tuning,
latest-three RGB feedback, camera rays/FiLM/QK, temporal RoPE, fused target warp, and
two-sided correspondence. The persistent coordinate lattice is shown separately as a
live L0/L1 experiment and is not called part of the method before its registered gate.
Persistent archives/KV, chunk sampling, latent feedback and distillation are explicitly
unshipped.

`THESIS_RESULTS_LEDGER.json` initially bound 23 current headline and limitation values to exact
selectors in the frozen factorial and memory-attenuation JSON artifacts. It also binds
the selected checkpoint to its content-sensitive fingerprint. Running
`python -m gaussverse.bench.thesis_result_audit` produced
`outputs/thesis_result_audit.json`. The ledger was expanded with every completed
factorial statistic used in the working thesis results table plus the clean-screen and
three-arm representation comparison: **PASS (58 claims, 1 checkpoint)**. This is a drift
detector, not a substitute for the statistical protocols that produced those artifacts.

The publication system figure is generated by `bench/architecture_figure.py` as both
`outputs/thesis_figures/gaussverse_architecture_current.png` and `.svg`. Its manifest
records factorial C, the selected fingerprint, the live/unselected lattice status and
the SHA-256 of `architecture.md`. Solid paths are measured selected components; dashed
amber is the live experiment; gray is unshipped work.

After the architecture, ledger, figure, documentation reconciliation, post-gate thesis
compositor, full artifact validation, and release audit, the complete repository
regression suite passes: **455 passed, 2 skipped**. The only emitted messages are
dependency deprecation/user warnings; no test failed.

`THESIS_METHODS_RESULTS_DRAFT.md` now contains thesis-ready problem definition, method,
frozen methodology, completed factorial results, limitations, discussion and a
reproducibility map. Its lattice subsection is deliberately an enumerated pending block:
training loss and tripwires cannot fill it. L0 completed 10k with trajectory `21.25 dB`
(safety Δ `+0.28`) and retention `10.89 dB` (Δ `-0.03`), so no catastrophic-forgetting
tripwire fired. The final checkpoint, optimizer state and step-10k metadata were all
rewritten successfully; this remains a safety trace, not lattice efficacy evidence. The
guarded L1 process then entered the registered blind preflight before allocating its
training model.

That preflight passed completely. Factorial C retained graded camera binding on all 16
scenes: large-lie minus correct flow loss was `+0.01361`
`[+0.00927,+0.01893]`, and large-minus-small damage was `+0.00871`
`[+0.00538,+0.01265]`. The five-scene L0 continuation smoke was also safe: L0-minus-C
rollout mean `-0.0187 dB` `[-0.1024,+0.0700]`, teacher mean `+0.2843 dB`
`[+0.1810,+0.3975]`, with worst scene `-0.158 dB` against the frozen `-2 dB`
catastrophic threshold. The immutable preflight binds parent fingerprint
`07a31e99fd49db6c7c717a5c`, L0 fingerprint `c2cea5b8593a29fe2158b629`
and shared-stream signature `1df525f1f0e2294a4c64b88f`. These are command-binding and
mechanical-safety gates, not evidence for the lattice.

**L1 live safety trace (steps 1k--3k; non-selection).** The active-field run persisted
`lattice=True`, `lattice_scale=1.0`, fused memory ON, the same parent, stream seed and
accumulation-4 recipe at step 3k. Its cheap chronological tripwire was `20.04/19.93/19.90`
dB at 1k/2k/3k (deltas `-0.93/-1.04/-1.07`), while unordered retention was
`10.91/10.89/10.85` dB (deltas `-0.00/-0.03/-0.07`); no 2 dB auto-stop condition fired.
The trajectory decline is not uniquely an L1 effect: the matched scale-zero L0 trace at
the same updates was `20.04/19.93/19.99` dB (deltas `-0.93/-1.04/-0.98`), and the mean
of the last ten logged losses through 3k is virtually equal (`0.11231` L1 versus `0.11233`
L0). These four-scene diagnostics are retained only as safety evidence; the frozen
n=16 L1-minus-L0 rollout and revisit gate remains the efficacy decision.

The post-training watcher will also generate a fail-closed thesis result block after all
registered follow-ups. `bench/thesis_lattice_summary.py` composes the primary gate,
revisit, within-L1 knockout, H50 and denoising-cliff artifacts only when their selected
arm, n=16 scene schedule, checkpoint fingerprints and descriptive decision roles agree.
Revisit provenance is checked independently for both arms, and the denoising cliff must
bind to the gate-selected checkpoint. It
writes both `trajectory_lattice_thesis_summary.json` and `.md`; it cannot read loss logs
or change the retain/drop result. Immediately afterward,
`bench/bind_lattice_results.py` atomically adds 22 exact summary selectors and both
checkpoint fingerprints to `THESIS_RESULTS_LEDGER.json`; it is idempotent for the same
artifact and refuses to replace any previously bound value. The result audit runs before
optional visuals or the scripted demo. Orchestrator and binder tests cover that ordering.
The marker-bounded `bench/finalize_thesis_lattice_draft.py` then replaces only Section
4.4 from the audited summary, removes the two pending-state sentences, and records hashes
of the draft and both summary artifacts. Final release refuses a stale or hand-diverged
draft. It runs after ledger audit and before optional media.

The authoritative architecture has the same protection. Section 2 is marker-bounded;
`bench/finalize_architecture.py` replaces the provisional factorial-C selection with the
gate-selected L0 or L1 checkpoint and hashes the summary and Markdown. The architecture
figure consumes the same summary: retained L1 draws the coordinate field as a solid
selected path, while dropped L1 draws it inactive and selects L0. Final release requires
the selection manifest, diagram manifest and current Markdown hash to agree.

Finally, the plan's live-status block is marker-bounded. Only after selected-arm visual,
demo, draft, architecture and result-audit manifests exist does
`bench/finalize_thesis_plan_status.py` replace “running” with the registered decision and
completed deliverables. README and the SOTA limitations table use artifact-conditional
language so they remain truthful on both sides of that transition.

Two release-safety gaps were then closed. First, before any L0/L1 rendering the
orchestrator now runs `validate_training_artifact` over **every checkpoint and optimizer
tensor**, verifies finiteness and step/config agreement, requires the `lattice.*` branch
when configured, and binds the report to checkpoint fingerprint and byte size. A cheap
safetensors header/fingerprint check is no longer accepted as proof that a 10 GB artifact
is readable. The report is additionally bound to full-file SHA-256 digests for both the
checkpoint and optimizer state, so a changed middle tensor cannot pass via matching size
and endpoint fingerprint. Metadata and the complete training log are digest-bound as
well, preventing a stale validation report from surviving later edits. Second, the scientific lattice summary is emitted before optional
figure/video/demo generation, so an ffmpeg failure cannot hide a complete gate result.

`bench/thesis_release_audit.py` implements partial and final release modes. The current
partial audit passes: result ledger, architecture figure/hash, authoritative local links,
source secret-pattern scan, and stale-current-state phrase scan all pass. Final mode
fails closed until L0/L1 full artifact validations, the lattice result/ledger binding,
visual and scripted-demo manifests, and `THESIS_RELEASE_ATTESTATION.json` exist. The last
file is deliberately not generated by code: it must attest that the exposed Hugging Face
token was rotated and external paper claims were reverified.

The final audit parses rather than merely finds both media manifests. It binds the
visuals and scripted demo to the gate-selected checkpoint, complete protocol
(`384x640`, `fidelity48`, seed 7, RGB feedback), recommended denoising step count and
lattice scale; it also requires every listed figure and video to be nontrivial and remain
inside its declared output directory.

Before L1 results existed, the final failure-panel rule was frozen as the scene with the
minimum paired L1-minus-L0 rollout-mean PSNR among the same 16 primary scenes (stable
scene-order tie break). It is descriptive and cannot enter retention. The post-gate
visualizer consumes exact scored L0/L1 evaluator PNGs for that panel and also emits a
scene-paired revisit/cross-path CI plot; the historical July path matrix is not reused as
evidence for this trajectory experiment.

The thesis-critical external audit was rerun against primary arXiv HTML on 2026-08-04 and
recorded in `THESIS_EXTERNAL_SOURCES.json`. Wonder directly supports the pixel-space
lattice, full-resolution cached K/V with pooled retrieval, initial sink + `r=2` recent +
top-k middle chunks, progressive 5/10/20-second teacher, and three generators over four
student denoising steps. Lyra directly supports per-frame geometry retrieval, native RGB
spatial slots, coordinate/depth rather than RGB warping, FramePack and one-step
self-augmentation. SANA-WM supports frame-wise GDN with **every fourth** block replaced by
softmax, not the absolute “15/20” count previously inferred in our documents; that count
was removed. ABot directly supports clean-history teacher forcing → ODE distillation →
LongForcing and asymmetric negative-time reference memory. HorizonDrive supports
prediction-corrupted-history recovery with local transition and global boundary-decay
schedules, followed by bounded teacher-rollout DMD; its quantitative horizon is about
20 seconds and minute scale is qualitative. AlayaWorld supports the 15B, six-latent
compressed-history, persistent-sink, recent-frame, geometry-aligned spatial-memory and
four-step distillation description. GIM-World supports fixed-size implicit tokens,
training-only camera-queryable VGGT distillation and bounded information pruning. Closing
the Loop supports training-free pose-matched latent-KV retrieval plus camera/depth spatial
attention bias. RELIC supports camera/action-encoded compressed latent KV memory, a
20-second teacher and block-replayed full-self-rollout DMD; its 16 FPS claim uses four
H100s. MemLearner supports generator-native query tokens that bridge historical and
predicted tokens under occlusion/dynamics. Prisma-World supports joint full-attention
multi-agent denoising, synchronized agent-aware RoPE, relative-camera attention and an
overlap-decaying curriculum. WorldStereo, GEN3C, CAMEO, UCPE, Geometry-as-context,
Seva and MultiDiff were then checked against their peer-reviewed CVF papers: their
geometric-memory, explicit-cache, correspondence-supervision, camera-encoding,
interleaved-geometry, joint-view and structured-noise descriptions all match the table.
The audit records WorldStereo's 64-H20 memory-training scale and RELIC's four-H100 speed
claim so neither is presented as a single-GPU recipe. The partial release audit now
requires all eighteen primary-source records and passes.

### 29.10 Completed L0/L1 result and the persistence-first follow-up program

The matched full-backbone coordinate-field comparison is complete. L0 (the instantiated
coordinate-field branch at scale zero) and L1 (the identical branch at scale one) both
trained for 10k updates from factorial C on the same stream. The frozen n=16 primary result
is **L1 minus L0 rollout PSNR +0.024 dB [−0.053,+0.099]** and teacher PSNR
**−0.040 dB [−0.099,−0.006]**. Direct lattice-lie binding is null. The registered gate
therefore selected **L0** (`c2cea5b8593a29fe2158b629`) and dropped the lattice. Revisit
safety passed on 40 eligible pairs (path B +0.012 dB [−0.019,+0.044]); the frozen GT-valid
pose intersection has only three jointly measurable scenes and remains descriptive.

The selected L0 denoising cliff accepts 30 steps only. At H50, L1 minus L0 is −0.328 dB
[−1.119,+0.327], and selected L0 RGB-feedback rollout is −8.282 dB [−9.266,−7.207] below
its teacher-history counterpart. This is a measured persistence limitation, not evidence
that lattice is harmful.

The next program is persistence-first. It does not reopen routing, clean source-separated
replacement, teacher-stage replay recovery, or coordinate-field additions. Before training,
Phase A uses selected L0 and the frozen 384×640, `fidelity48`, seed-7 trajectory protocol
to measure immutable observed-frame warp evidence selected by target-view greedy coverage,
the VAE-cycle sharpness bound and rollout contraction, and executable inference-only
guidance. Any individually safe winners are recombined through H20/H50 with PSNR, LPIPS
and Laplacian sharpness.

Native latent feedback is not an open Phase-A arm. The completed matched n=16 gate selected
RGB feedback: latent minus RGB rollout PSNR is −0.225 dB [−0.410,−0.030], H20 is −0.559 dB
[−0.855,−0.207], and all adoption conditions failed
(`outputs/feedback_rgb_vs_latent_fidelity16.json`). It remains an historical negative
unless a later causal-student/chunk architecture changes the representation contract.

**Phase-A decision rule.** Compare the recombined recipe with pure RGB-feedback L0 on
identical scenes/schedules. It passes only if either the H50 plateau rises by at least
1.0 dB or H20 improves by at least 0.5 dB; report paired scene-bootstrap CIs, median and
sign count. The graded-lie binding battery remains the standing safety tripwire.

### 29.11 Phase-A implementation and registered component screens

The immutable-observation reader is now executable as `--evidence-source
observed_fallback`: it projects only the original three real context frames into a target
view when their target coverage is at least 0.10, and otherwise uses the historical recent
window. It never writes a generated frame into that cache. The paired H20 component result
is `outputs/phase_a_observed_evidence_gate_h20.json`: rollout mean is +0.181 dB
[−0.095,+0.461], with H20 −0.075 dB [−0.660,+0.502]. It is not a pass and cannot select a
recipe. The registered 16-scene H50 continuation is
`outputs/phase_a_observed_fallback_h50.json`, paired to the pre-existing selected-L0 H50
run; it is the component's plateau test.

The blur diagnostic is `bench/phase_a_blur_diag.py`, using the frozen H50 dumped rollout
frames and real held-out images. It reports generated-history sharpness contraction
alpha=0.974 over 752 three-frame-context pairs (initial 0.01009, final 0.00762), and VAE
real-frame sharpness ratios 0.870/0.801/0.764 after one/two/three encode-decode cycles
(`outputs/phase_a_blur_diagnostic.json`). Laplacian energy is recorded only beside PSNR and
LPIPS: it can increase under noise, so it is explicitly not a quality metric or a filtering
objective.

The remaining inference-only guidance screen is now implemented as PAG rather than an
invalid CFG surrogate. `--pag-scale` takes a second denoiser prediction in which explicit
selected self-attention layers use an identity map, and extrapolates the target flow away
from it; `--pag-scale 0` is exactly the ordinary sampler. The first registered screen is
teacher H10, frozen 16 scenes, selected L0, RGB feedback, 30 denoising steps, blocks
9,19,29, scales 0.25/0.50/0.75. Prior: low-to-moderate; information value is high because
it distinguishes sampler softness from learned contraction; it is an inference
substitution, not another conditioning addition. A screen winner may enter H20/H50 only if
its LPIPS does not worsen and its paired PSNR CI excludes zero; it cannot alone satisfy the
Phase-A decision rule.

The contraction stopgaps are similarly registered but not yet interpreted: RGB unsharp
amounts 0.25/0.50 and anchor-colour matching (moments of the first three *real* frames
only) are each H10 rollout component arms. A candidate is eligible for recombination only
if its paired LPIPS does not worsen, its PSNR CI excludes zero, and its Laplacian increase
does not occur alongside an LPIPS regression. They are output/feedback transforms with an
explicit no-op default, not claims that the underlying world model became sharper.

### 29.12 Immutable observed evidence H50 result — component fails

The completed frozen n=16 H50 run is
`outputs/phase_a_observed_fallback_h50.json`, paired against selected L0's prior RGB
recent-evidence dump by `outputs/phase_a_observed_evidence_gate_h50.json`. Observed-frame
fallback minus recent evidence is +0.069 dB rollout mean [−0.306,+0.447], LPIPS improvement
−0.000017 [−0.000093,+0.000045], and Laplacian sharpness +0.000206
[−0.001094,+0.001443]. H3 is +0.340 dB [+0.087,+0.638] and H5 +0.275 dB
[+0.013,+0.529], but H20 is −0.075 dB [−0.660,+0.502] and H50 **−0.267 dB
[−1.101,+0.430]**. Thus immutable evidence has a short transient benefit but no
demonstrated plateau benefit; it is not eligible for recombination and does not move Gate A.

This is not a reversal of the factorial memory result. The factorial's fused evidence is
still the selected stage-1 conditioning system; this Phase-A reader substitution says that
continually projecting the initial three observed frames is not a long-horizon persistence
mechanism. Generated history needs a native latent/KV subsystem rather than more RGB-space
warping.

---

## 30. Static teacher Stage A — gate PASS, and the evaluator that had to exist first (2026-08-08)

### 30.1 The evaluator was a prerequisite, not a formality

`bench/eval_trajectory.py` could not score this checkpoint. Its branch table enumerated
`full/camqk/memory/correspondence/trajectory_camera/ray_local_ucpe/history_repair/
source_reader/lattice` and had **no `framepack` entry**, so it would have loaded the
teacher, silently discarded every `framepack.*` tensor, and reported confident numbers
for a structurally different model. Three defects were fixed before any number was
produced:

1. `_checkpoint_branches` now detects `framepack.`;
2. `sample_target` accepts `framepack_plan`/`framepack_conditioner` — multi-step
   sampling of a FramePack teacher was previously impossible;
3. under a plan, `model_fn` returns **target-only** frames while `sample_target` sliced
   `pred[:, :, m:]` assuming full-sequence output. Every denoising step would have been
   mis-sliced. Fixed via `target_offset = 0 if framepack_plan is not None else m`,
   mirroring `flow_match_terms`, with a regression test.

A fourth was caught by the smoke run: `decode_target` decodes only the *last* latent, and
per-frame-encoded latents must be decoded individually, since decoding `L` together
yields `1 + 4(L-1)` frames.

`bench/eval_static_teacher.py` derives resolution and chunk width from the stage record
rather than from flags, and requires the matched control; `--skip-control` stamps
`interpretable_as_gate: false` into the artifact.

### 30.2 Result

Artifact `outputs/static_teacher_stage_a_eval.json`. Teacher
`static_teacher_stage_a/ckpt_last.safetensors`, step 3000, endpoint fingerprint
`69e6aca9fd067b6f6a304b0c`, initialised from `trajectory_lattice_l0_10k`. Control is that
same L0 parent under the identical protocol: n=8 frozen held-out scenes, 192x320, chunk
2, context 6, horizon 8, seed 7, 30 denoising steps. Every value below is oriented so
positive means the teacher improved.

| regime | metric | teacher | control | improvement | CI95 | wins |
|---|---|---|---:|---:|---|---:|
| teacher-forced | PSNR | 19.4282 | 16.8229 | +2.6053 | [+1.9144,+3.2721] | 8/8 |
| teacher-forced | SSIM | 0.5724 | 0.4355 | +0.1369 | [+0.1105,+0.1604] | 8/8 |
| teacher-forced | LPIPS | 0.0027 | 0.0040 | +0.0013 | [+0.0011,+0.0015] | 8/8 |
| rollout | PSNR | 16.9260 | 14.5157 | +2.4102 | [+1.7009,+3.3022] | 8/8 |
| rollout | SSIM | 0.4626 | 0.3596 | +0.1030 | [+0.0678,+0.1407] | 8/8 |
| rollout | LPIPS | 0.0039 | 0.0052 | +0.0012 | [+0.0008,+0.0017] | 8/8 |

A reporting defect was found and fixed in the summariser itself: `wins` counted
`delta > 0`, which is backwards for lower-is-better LPIPS and printed "wins 0/8" where
all eight scenes had improved. Every metric now carries `higher_is_better` and both
`raw_delta_mean` and `improvement_mean`.

### 30.3 What this does and does not establish

**Scope limit, stated before the run and unchanged by it.** The control is off
distribution: L0 was trained for single-frame targets with three dense context frames at
384x640, and is here asked for two-frame chunks with six FramePack slots at 192x320. The
+2.61 dB therefore conflates "the teacher improved" with "the control is handicapped".
No cleaner control exists, because evaluating L0 at its native interface would change
resolution *and* chunk width, which the protocol forbids comparing.

The gate answers the question it was registered to answer: **the Stage-A recipe adapted
the model to its new interface without breaking anything.** It is not an absolute quality
claim, and 192x320 values remain incomparable to every 384x640 headline in this document.

That rollout improvement (+2.41) nearly matches teacher-forced (+2.61) is the one
secondary observation worth keeping: the gain is not confined to teacher forcing.

### 30.4 From-base initialisation probe (running)

Wonder, ABot and Lyra all initialise their teacher from the pretrained foundation model;
Stage A initialises from an internal checkpoint. `--from-base` was added to
`train_static_teacher` and a matched arm launched under the protocol in
`bench/from_base_binding_probe.md`. The registered question is a *rate* question: does
graded-lie binding re-emerge from cold, against the parent's
`+0.01361 [+0.00927,+0.01893]`?

The launched arm uses the full 3000-step stage schedule rather than the 1500 steps first
drafted, because truncation would compress the cosine and confound initialisation with
learning-rate shape. Early matched-step dynamics (identical objective mixture at every
step, confirming a shared stream): loss `0.3073/0.1776/0.1477` at steps 25/100/500
against the warm arm's `0.1348/0.1211/0.1197`, with gnorm decaying `22.21 -> 8.97 -> 2.42`
against `0.237 -> 0.112 -> 0.105`. Converging, not diverging. Loss is not evidence of
camera control; the battery is.

**Confound recorded mid-run, before any binding number existed.** The arm inherits
`RayContextProjection`'s zero-initialised patch projection, so at step 0 it is exactly
pretrained Wan2.2-5B with no camera conditioning at all. That scheme exists to protect a
*converged* backbone when attaching a branch; from base there is nothing to protect, and
the branch must climb out of zero while the backbone also moves — the same mechanism
proposed for the lattice null in §29.10. This arm therefore measures from-base
initialisation *under a warm-start-shaped conditioning init*, not from-base in general. A
weak or null binding result does not close the question; the follow-up is a from-base arm
with ordinary (non-zero) branch initialisation.

Budget context: 1.216B trainable parameters at effective batch 32 over 3000 steps is
96,000 sample exposures, against roughly 320-450k for Lyra on 64 GPUs. Undertraining and
an intrinsic limit are not separable at this scale, and the write-up must say so.
Flow-matching loss plateaus early by construction — the from-base arm flattens about
0.019 above the warm arm from step 1000 — and carries little information about camera
control. `bench/teacher_binding.py` is the read.

### 30.5 From-base initialisation: decisively worse than the warm start (2026-08-09)

Three arms, all 3000 steps at stage A, same cache, seed, stream, schedule and interface.
Only initialisation regime differs.

| arm | init | lr | clip | parent |
|---|---|---|---|---|
| warm | zero | 5e-6 | 1.0 | `trajectory_lattice_l0_10k` |
| arm 1 | zero | 5e-6 | 1.0 | pretrained Wan2.2-5B |
| arm 2 | scaled | 3e-5 | 5.0 | pretrained Wan2.2-5B |

**Primary result — arm 2 minus warm** (`outputs/fid_arm2_vs_warm.json`, n=8, positive
means arm 2 better). Every metric negative, every interval clear of zero, **0/8 scenes**:

| metric | teacher-forced | rollout |
|---|---|---|
| PSNR | −2.878 [−3.533,−2.167] | −2.620 [−3.475,−1.683] |
| SSIM | −0.146 | −0.116 |
| LPIPS | −0.0014 | −0.0014 |
| Sampson | −0.0039 | −0.0159 |
| WorldScore camera | −7.16 [−10.13,−4.76] | −21.99 [−32.72,−13.24] |
| held-out val_loss | −0.0130 [−0.0143,−0.0117] | same |

**Secondary — arm 2 minus arm 1** (`outputs/fid_arm2_vs_arm1.json`), isolating the three
corrected defaults: teacher PSNR +0.898 [+0.526,+1.266] 7/8, rollout +0.586
[+0.150,+1.063] 7/8, val_loss +0.0061 8/8, WorldScore teacher +5.49 [+2.82,+8.94] 8/8.
Rollout camera is ambiguous: Sampson −0.0040 [−0.0111,+0.0025] and WorldScore +17.66
[−1.62,+41.32] both cross zero.

So the corrections recover about **24 % of the from-base deficit** (0.90 dB of 3.78) and
are worth keeping, but do not make from-base competitive at this budget.

**Methodological finding, and the more transferable one.** Training loss badly understated
the gap:

| quantity | arm 2 minus warm |
|---|---|
| training loss (mixture, training scenes) | 0.0047 |
| held-out val_loss (clean, fixed sigmas) | 0.0130 |
| PSNR | 2.9 dB |

A 68 % reduction in the *training*-loss gap corresponded to a model still 2.9 dB and
22 WorldScore points behind. Flow-matching loss over a shifting objective mixture on
training scenes is not a progress signal for this comparison; the held-out value is
nearly 3x larger and even that understates the fidelity gap. Selection must use the
task evaluation, per the standing rule "judge only by task A/B, never val_loss".

**Decision.** The warm start from `trajectory_lattice_l0_10k` is retained for the teacher
line. Wonder, ABot and Lyra initialise from the pretrained foundation model, but they
spend roughly 320-450k sample exposures on 32-64 GPUs; stage A spends 96,000 on one. The
measured justification for departing from that convention is this table.

**Scope.** This does not show camera control cannot be learned from base — arm 1's earlier
graded-lie battery bound at 16/16 with a monotone ladder. It shows that at 96k exposures
the resulting model is materially worse than a warm start on fidelity and camera control.

### 30.6 Graded-lie binding, all three arms, protocol-conformant (2026-08-09)

Fused memory and two-sided correspondence ACTIVE, per §29.8. Artifacts
`outputs/binding_{warm,arm1,arm2}.json`, n=16, shared sigma ladder and target noise.

| arm | correct | small lie | med lie | large lie | direct binding | gradedness |
|---|---:|---:|---:|---:|---|---|
| warm | 0.1355 | 0.4737 | 0.5355 | 0.5374 | +0.402 [+0.375,+0.428] 16/16 | +0.064 [−0.003,+0.142] **FAIL** |
| arm 1 | 0.1550 | 0.1856 | 0.2333 | 0.3789 | +0.224 [+0.173,+0.280] 16/16 | +0.193 [+0.133,+0.257] pass |
| arm 2 | 0.1437 | 0.2059 | 0.3189 | 0.7390 | +0.595 [+0.479,+0.702] 16/16 | +0.533 [+0.405,+0.654] pass |

**The warm arm saturates.** A 0-15 degree lie already costs +250 %; widening it to 45-180
degrees adds only +13 % more. Its gradedness interval touches zero. Arm 2 instead scales
steeply with lie size (+43 / +122 / +414 %). This reproduces the superseded run almost
exactly (+0.414 -> +0.402, graded +0.065 -> +0.064), so it is a property of the
checkpoint, not of the earlier protocol defect.

**Protocol effect was small.** Arm 1 non-conformant +0.237 vs conformant +0.224: turning
the evidence branches on changed binding by about 5 %. An earlier note in this session
claimed evidence *amplified* the contrast; that was an artifact of comparing arm 2 against
arm 1 and is retracted.

**Binding and following are different quantities, and they dissociate.** The battery asks
"does the model notice a lie about the camera"; Sampson and WorldScore ask "does the
generated frame sit at the commanded camera". Arm 2 wins the first decisively (+0.595 vs
+0.402, and properly graded) while warm wins the second decisively (WorldScore +7.2
teacher / +22.0 rollout, Sampson better in both regimes) and is 2.9 dB better on PSNR.

A strongly conditioned model can be brittle to conditioning error without being inaccurate
under correct conditioning. Nothing here overturns §30.5: warm remains the selected
teacher line. What it adds is that the warm checkpoint cannot rank camera-error severity,
which is worth knowing before any work that depends on graded pose sensitivity -- a
pose-verification or trust-router stage would be reading a signal this checkpoint does
not provide.

### 30.7 Arm 2 extended to 6000 steps: fidelity improves, binding does not (2026-08-09)

Arm 2 (`--from-base`, scaled init, lr 3e-5, clip 5.0) resumed from its own step-3000
checkpoint with optimizer state, to step 6000. The 3000-step checkpoint is preserved at
`static_teacher_from_base_fixed`; §30.5/§30.6 continue to bind to it. Artifacts
`outputs/fid_arm2_6k_vs_{arm2_3k,warm}.json`, `outputs/binding_arm2_6k.json`.

**Metadata defect found and fixed.** `write_meta` recorded `parent_checkpoint=str(args.init)`
unconditionally, but `--resume` reloads the run's own checkpoint and never reads `--init`.
The extension's `meta.json` therefore named the lattice parent and `from_base: false` --
the relaunch argv, not the lineage -- and this session misread the run as a warm restart
for roughly an hour before the byte-identical step-25 log line exposed it. Now records the
checkpoint actually loaded plus a `resumed` flag; regression test in
`tests/test_init_scheme.py::test_meta_records_the_checkpoint_actually_loaded`. The affected
`meta.json` carries a `metadata_correction` field rather than a silent rewrite.

#### More steps help, and have not saturated

| regime | PSNR | SSIM | WorldScore camera | val_loss |
|---|---|---|---|---|
| teacher | +1.003 [+0.662,+1.339] 8/8 | +0.047 8/8 | +2.78 [+1.60,+3.80] 7/8 | +0.0033 8/8 |
| rollout | +1.163 [+0.831,+1.536] 8/8 | +0.046 8/8 | +12.61 [+7.31,+17.66] 8/8 | +0.0033 8/8 |

Every metric improves on every scene. The gain is larger in rollout than under teacher
forcing (+1.163 vs +1.003 dB; WorldScore +12.6 vs +2.8), i.e. the extra steps helped most
in the regime that feeds on its own output. LPIPS improves 8/8 but by +0.0005 -- clean and
practically nil. What improved is geometric, not perceptual.

#### Arm 2 at 6k still loses to warm at 3k

| rollout | gap at 3k | gap at 6k | closed |
|---|---:|---:|---:|
| PSNR | -2.620 | -1.457 [-2.040,-0.812] | 44 % |
| SSIM | -0.116 | -0.070 [-0.094,-0.043] | 40 % |
| WorldScore camera | -21.99 | -9.38 [-16.32,-4.48] | 57 % |
| val_loss | -0.0130 | -0.0097 [-0.0113,-0.0080] | 25 % |

0/8 scenes on every metric (lone exception: rollout LPIPS 1/8). Consistency check: the two
independent runs agree exactly -- 2.6202 (3k gap) - 1.1631 (3k->6k gain) = 1.4571 = the
directly measured 6k gap.

Extrapolating the rollout PSNR gap geometrically (ratio 0.556 per doubling) puts arm 2
within 0.5 dB at about 24k steps, roughly 16 h at the observed 2.46 s/step. This is a
two-point extrapolation and should be held loosely.

**Compute accounting matters here.** Warm's parent `trajectory_lattice_l0_10k` had 10k
prior steps, so "warm@3k" is ~13k cumulative steps against arm 2's 6k. Arm 2 is 1.46 dB
behind on 46 % of the total training. At matched wall-clock on this task warm wins clearly;
at matched total compute the comparison is unresolved. Neither framing is obviously the
right one, and §30.5's selection of the warm line is unchanged by this.

#### Binding was already finished at 3000 steps

| arm | correct | small | med | large | direct binding | gradedness |
|---|---:|---:|---:|---:|---|---|
| warm@3k | 0.1355 | 0.4737 | 0.5355 | 0.5374 | +0.402 16/16 | +0.064 **FAIL** |
| arm 1@3k | 0.1550 | 0.1856 | 0.2333 | 0.3789 | +0.224 16/16 | +0.193 pass |
| arm 2@3k | 0.1437 | 0.2059 | 0.3189 | 0.7390 | +0.595 16/16 | +0.533 pass |
| arm 2@6k | 0.1387 | 0.2122 | 0.3392 | 0.7470 | +0.608 [+0.498,+0.710] 16/16 | +0.535 [+0.404,+0.659] pass |

Doubling the steps moved gradedness by +0.0016 (+0.533 -> +0.535) and direct binding by
+0.013, both far inside their intervals. Camera binding is complete by step 3000 and the
second 3000 steps bought none of it -- while buying +1.16 dB and +12.6 WorldScore.

This sharpens §30.6: binding and following are not merely dissociable across arms, they are
dissociable across training time within one arm. Fidelity and camera-following are still
improving at 6k; sensitivity to camera error stopped improving at 3k. Any future decision
that depends on graded pose sensitivity should be read off an early checkpoint just as
well as a late one, and cannot be bought with more steps.

#### Consequence for planning

Do not extend arm 2 to parity yet. The native-latent contract (4 poses per latent) changes
`RayContextProjection`'s input from 1536 to 6144 channels, which invalidates the ray branch
of every existing checkpoint. A 16-33 h from-base run started now would be discarded by
that change. Freeze the architecture first, then settle initialisation once.

### 30.8 Native latent contract, and three architecture fixes (2026-08-09)

Implements the outstanding list: drop dead LoRA, settle the self-attention double
parameterisation, route rays through Q/K only, and move the corpus onto the Wan native
latent contract (one latent = FOUR RGB frames, so a 2-latent chunk is 8 generated frames).
Suite 573 -> 613 tests.

#### One substring caused both LoRA defects

`is_selective_backbone_parameter` matches `"self_attn" in name`. Applied after PEFT
injection, that match hits `self_attn.q.base_layer.weight` AND
`self_attn.q.lora_A.default.weight`, while `cross_attn.*` and `ffn.*` adapters match
nothing. So the same line both starves two thirds of the adapters and trains
self-attention through two parameterisations with independent AdamW second moments.
Measured on the finished 6000-step arm 2: of 300 `lora_B` tensors, **cross_attn (120) and
ffn (60) are exactly zero; self_attn (120) are all nonzero**, sum|B| = 1.64e4.

`--lora off` is now the default. Because PEFT renames the wrapped weight, switching tiers
would make every existing checkpoint unloadable, so `merge_lora_into_base` folds
`W <- W + scaling*B@A` and unwraps -- the exact function LoRA computed, verified to 1e-5.
The trainer builds at the tier read from the checkpoint's own keys, loads, then merges.
Verified on the real arm-2 checkpoint: *"merged 300 'legacy' LoRA adapters into base
weights"*, then trained.

#### Ray routing is a checkpoint property, not a global default

`ray_qk_only` sends the ray adapter into the attention Q/K bias instead of the residual
stream, where it also reaches V. Lyra 2.0 keeps camera and buffer in `kq_bias` only.
Trainer default is on; the shared `flow_match_terms`/`sample_target` default is OFF and
the value is recorded in meta.json, because a True default there would silently re-route
every existing checkpoint at eval and void SS30.5-30.7.

#### Lyra uses ONE pose per latent, and that reverses a planned deviation

`camera.packed_rays` proposed four poses per latent (6144 ch) on the argument that a
command must steer every frame, not every fourth. Lyra 2.0 does not do this:

    cam_dim = 1536 if self.use_plucker_condition else 0        # networks/wan2pt1_lyra2.py
    time_idx = [0] + [i for i in range(4, v_len, frames_per_lat)]   # models/lyra2_model.py

One pose per latent, taken at the group's LAST frame. Their 1536 is 6*8*8 for an 8x VAE;
our 1536 is 6*16*16 at 16x -- the same quantity, not a coincidence. `latent_pose_frame`
reproduces their `time_idx` exactly ([0,4,8,12,16,20] on a 21-frame clip).

This matters beyond fidelity to the reference: keeping one pose per latent leaves
`RayContextProjection` at its existing input width, so **the native contract does NOT
invalidate any trained ray branch**. The earlier claim in this session that item 5 would
discard every checkpoint was wrong, and with it the argument for deferring further
from-base training. `packed_rays` is retained but unused.

#### The paired-target invariant under a stateful encoder

The cache guarantees two histories share one target byte-for-byte. A streaming VAE
breaks that: continuing each history's own stream into the target gives two different
targets. Resolution: encode the target ONCE, primed on the five true frames immediately
before the chunk -- a property of the scene and the chunk position, never of the sampled
history. `test_primed_target_equals_the_whole_sequence_encode` confirms on real weights
that priming then encoding 8 frames equals a single 13-frame encode at indices 2-3.

A four-frame group has no encoding of its own (T = 1 + 4(L-1) admits 1, 5, 9, ... but not
4), so `encode_group` defines it as the second latent of a five-frame window. That fixed
predecessor is what makes a *strided* history expressible in native latents at all.

#### Cache v2, and a validator gap found by composition

Latent counts are unchanged, so every tensor shape is identical to v1 and no shape check
can distinguish the contracts; the version is therefore load-bearing. v1 still loads --
verified against the real 4000-entry corpus, not a fixture.

An integration test caught a gap the unit tests could not: group start 0 needs frame -1,
and negative slicing returned an empty tensor rather than raising. The sampler already
refused it; the validator did not. Now it does. Its companion "prime fits" check was then
removed as provably unreachable -- `ctx[0] >= 1` plus `ctx[-1] + 4 <= target` force
`target >= 5`, checked exhaustively over context counts 1-7 and all legal spacings.

#### End-to-end verification

A real 3-entry native cache built and loaded: groups `[42,46,50,54,58,62]` (disjoint,
stride 4), target `[66]` as a single chunk start, context pose frames `[45,...,65]`,
target pose frames `[69,73]`, A/B targets byte-identical while contexts differ, shapes
unchanged from v1. Two real training steps consumed it at 1.192B trainable with no
adapter parameters.

**Remaining: build the full 4000-entry native cache (~4-6 h) and train.** No result in
this section; the corpus and the code are ready, the teacher is not yet trained on it.

### 30.9 Long-horizon stability: what four SOTA systems actually do (2026-08-09)

Prompted by the per-step rollout curves, which show every arm losing ~6 dB across 8
autoregressive steps at nearly the same rate. Read from source where available
(Lyra 2.0, GEN3C on disk) and from the papers otherwise (Wonder arXiv 2607.26037,
ABot-World-0 arXiv 2607.19191). SANA-WM could not be inspected: `growworld` now contains
only `third_party`.

| system | conditioning latents | on-policy in teacher | where drift is handled |
|---|---|---|---|
| GEN3C | corrupted, `augment_sigma` | -- | explicit 3D point cache, re-rendered per camera |
| Lyra 2.0 | corrupted, same draw | ~50 % (`every_k=2`, `prob=1.0`) | in the model itself |
| Wonder | **clean, explicitly** | -- | self-forcing distillation, mixture-of-students, GAN control reg |
| ABot-World-0 | clean | -- | **LongForcing**: DMD vs an extended-horizon teacher |
| GaussVerse (before) | clean | 24.8 % | nowhere |

Wonder states the choice outright: *"Noise is applied only to the target frames that need
to be generated, while the conditioning latents remain clean throughout both training and
inference."* Lyra's corruption is `noise_with_sigma`, marked *"Training only"* in
`lyra2_model.py`, with `sigma ~ exp(N(-3.0, 2.0))` per batch-and-frame and a fixed 0.001
at inference. Lyra's helper is named `draw_augment_sigma_and_epsilon_gen3c`: GEN3C
originated the recipe and Lyra inherited it. `condition_on_augment_sigma` is False, so the
network is never told how corrupted its history is.

**The split is not arbitrary, and the discriminating fact is what the trained model is
FOR.** Wonder's and ABot's teachers are bidirectional and never roll out; a causal student
does, and drift is handled there. Lyra ships no student -- `lyra2_ar_inference.py` rolls
out the trained model itself, so that model must survive its own outputs. GaussVerse's
teacher is currently in Lyra's position: it is the thing we roll out and measure.

**The finding that decided it was about our own configuration, not theirs.** We had
Lyra's self-augmentation at half its rate combined with Wonder's clean conditioning -- a
pairing neither reference uses, arrived at by accident rather than choice. The recipe now
follows Lyra coherently: `--history-noise lyra` (defaults matching theirs exactly;
measured sigma percentiles 0.0067 / 0.0494 / 0.3616 against analytic 0.0067 / 0.0498 /
0.3679) and `--self-aug-final 0.50`. Evaluation applies the recorded inference sigma, so a
clean-history checkpoint is never scored with noise added and vice versa. History only:
the target slice is the flow-matching label, and a test pins the concatenation.

**Three independent confirmations of choices already made.** Wonder uses *"a chunk size of
two latent frames"* -- the native contract of SS30.8. Wonder's sparse context forcing
(*"non-local historical edges ... randomly kept"*, stronger drop for longer horizons) is
what `sample_sparse_context_pair` already does at 16.9 % of steps. Wonder's memory --
initial chunk plus most recent `r=2` plus top-k middle history by query-key similarity, at
full KV resolution -- is the shape of `select_hybrid_context_sources`, which is built,
tested, and disabled at Stage A (`retrieval_k=0`). That is the next supported lever.

**An open confound, now sharper.** `trajectory_lattice_l0_10k`, the warm arm's parent, was
trained with context corruption (`context_corrupt_max 0.1`, `prob 0.5`, warmup 2000). The
from-base arms never saw corrupted history. Warm's rollout advantage may therefore be
partly inherited anti-drift exposure rather than extra optimizer steps -- untested, and it
predicts that the native run should close some of the gap for that reason alone.

**Cost.** This is the sixth simultaneous change relative to arm 2, so a win will not be
attributable. `--history-noise off` and `--self-aug-final 0.25` keep the ablation cheap,
but the debt is real and is recorded here rather than discovered later.

### 30.10 The reported "LPIPS" was never LPIPS (2026-08-09)

Found while reading Lyra-1's shipped evaluation harness (`src/eval/metrics.py`), which uses
`LPIPS(net="vgg")` -- the real package with its learned linear weights.

`bench/metrics.lpips_vgg`, which `eval_static_teacher` imports and reports as `mean_lpips`,
is a VGG16 relu3_3 normalised-feature L2 distance. Its own docstring says "LPIPS-style"; it
carries **none** of LPIPS's learned weights and its scale is unrelated. Measured on one
degraded pair: proxy **0.0018** against real LPIPS **0.1659**, roughly 90x. That is why every
recorded value sits near 1e-3 where LPIPS on visibly damaged frames reads 0.2-0.5.

**No conclusion changes.** The perceptual column was never load-bearing -- every reported
win rests on PSNR, SSIM, camera control and the binding battery, and the proxy is monotone
enough to have been read correctly as "essentially unchanged" (the 3k->6k gain of +0.0005 was
described in-session as statistically clean and practically nil, which is true of either
metric). What was wrong was the **name**, in a document intended as the standing reference
and in a comparison against Lyra that the metric could not support.

Fixed: `score_frame` now emits `vgg_feature_distance` (the proxy, honestly named) and `lpips`
(the real metric, `lpips_official`, vgg backbone, matching Lyra-1). Both carry orientation
entries. `SYSTEM.md` relabels the column *VGG feat dist* and states the non-comparability
where the number appears. Runs from the native teacher onward carry both; the four existing
arms have only the proxy, and no retrospective LPIPS is possible because frames were not
retained.

**Lyra's own reporting shape, for reference.** `compute_metrics_datasets.py` accumulates
`psnr_sum` as a `[T]` tensor -- per frame index, averaged across scenes, with per-frame std
and a `plot_average_metric_per_frame` curve. That is the same structure as SS30.8's per-step
table, arrived at independently. Lyra-1 ships no camera metric at all; the WorldScore figure
attributed to Lyra 2.0 comes from the paper, not from released code.

### 30.11 The rollout failure is camera lag, not blur (2026-08-09)

Diagnosis of `outputs/arm_step_boards/`: three held-out scenes, four arms, a row per
autoregressive step. Measured by cropping the boards and scoring each generated frame
against every ground-truth step.

**Method.** For generated frame at step k, report `argmax_j PSNR(gen_k, gt_j)`. On-track
gives j = k; j < k means the model has fallen behind the commanded trajectory.

| arm | mean lag (steps, n=3 scenes) | camera control | rollout PSNR@8 |
|---|---:|---:|---:|
| warm@3k | +0.21 | 94.3 | 14.59 |
| arm2@6k | -0.46 | 85.0 | 13.44 |
| arm2@3k | -1.33 | 72.3 | 11.95 |
| arm1@3k | -2.17 | 54.7 | 11.81 |

Correlation of lag with camera control **+0.995**, with PSNR@8 **+0.954**. On the hardest
scene warm matches `[1,2,3,4,5,6,7,8]` -- exact tracking at every step -- while arm2@3k
gives `[1,1,2,1,1,1,2,2]`, frozen near the opening view for the whole rollout.

**The ~6 dB rollout loss is a registration error, not an image-quality one.** Frames stay
sharp and plausible; they show the wrong place. Sharpness relative to ground truth is
roughly stable, and the apparent "over-sharpening" at late steps is an artifact of the
REFERENCE degrading: ground-truth sharpness in fe30a865 falls 11.1 -> 6.3 (0.57x) because
the real camera moves fast enough to motion-blur the footage. The models are sharper than
ground truth there, and pay PSNR for it.

**Mechanism, and why it matters for the recipe.** With clean conditioning, copying the
previous frame is a low-loss strategy: adjacent frames are similar, so under-motion is
barely penalised in training while compounding catastrophically at rollout. This is the
exposure-bias account with a measured mechanism rather than an analogy, and it is a
stronger argument for SS30.9's history corruption than "Lyra does it" -- corrupting the
history removes the copy shortcut and forces reliance on the camera signal.

**Falsifiable prediction for the native run.** History noise at sigma ~ exp(N(-3,2)) plus
self-augmentation at 0.50 should *reduce lag*; PSNR and camera control should follow. If
the native teacher reproduces the same lag profile, the corruption is not reaching this
mechanism and the next lever is the camera pathway itself, not the objective.

**Caveats.** The correlation is over four arms (n=4) and is suggestive, not established;
the per-frame lag measurement itself is direct. A second, distinct failure appears only in
the weakest arm: arm1@3k on fe4bd0d4 best-matches `[1,2,2,5,8,1,2,1]`, jumping back to the
opening frame -- collapse rather than lag.

### 30.12 Invented roll, and two opposite object failures (2026-08-09)

Follow-up on SS30.11, prompted by an observation that the generated frames look tilted and
that the cars merge. Scene fe4bd0d4 (temple + parking lot), `outputs/arm_step_boards/`.

**Roll is measurable and real.** Rotation of each generated frame relative to its own
ground-truth step, from a partial-affine fit over SIFT matches (RANSAC). The estimator is
calibrated on deliberately rotated ground truth and recovers +3 / -5 / +8 degrees to within
0.01, which two earlier estimators did not -- both returned a constant regardless of input,
and would have been reported as "no roll" had the calibration check been skipped.

| step | warm@3k | arm2@6k | arm2@3k | arm1@3k |
|---|---|---|---|---|
| 3 | +0.01 | +1.79 | +1.18 | +0.36 |
| 5 | +0.30 | +3.08 | +4.84 | n/a |
| 6 | +0.03 | +5.01 | -0.25 | n/a |
| 8 | -0.27 | +4.68 | n/a | n/a |

warm holds orientation throughout (|roll| <= 0.30 deg). arm2@6k accumulates monotonically
to ~5 deg. **Roll is not in the action space** -- W/A/S/D translate, Q/E yaw, and no command
produces roll -- so this is an invented degree of freedom.

**Orientation drift is independent of positional lag.** In this scene arm2@6k has near-zero
lag (-0.12 steps, SS30.11) and the worst roll. Position-following and orientation-following
therefore fail separately and need separate metrics; the WorldScore camera score folds both
into one number and cannot distinguish them.

**`n/a` is itself a result.** arm1@3k yields fewer than 8 RANSAC inliers against ground
truth from step 4 onward: the scene has diverged past correspondence entirely.

**Two opposite object failures.** Edge density in the lower (car) band, relative to ground
truth: warm 0.91 at step 8, arm2@6k 0.79, arm2@3k **0.55**, arm1@3k **1.07** (1.22 at step
6). arm2@3k *merges* -- the two cars fuse and the gap between them fills with car-coloured
material rather than showing background, a depth/occlusion failure in which foreground grows
into empty space. arm1@3k *fragments* -- more edges than the reference, spurious texture and
colour blobs. Same divergence, opposite signatures, so "degradation" is not one phenomenon.

**Caveat.** SIFT inlier counts fall with rollout depth (92 -> 25 for warm, 104 -> 8 for
arm2@6k), so late-step angles carry real uncertainty. The monotone trend across steps 3-8
for arm2@6k is more trustworthy than any single cell.

### 30.13 Closing two evaluation gaps found by looking at frames (2026-08-09)

SS30.11 and SS30.12 identified failures the protocol could not see. Both are now
measurable. Suite 613 -> 633.

**1. Every recorded rollout ran with the occlusion machinery switched off.**
`eval_static_teacher.generate_chunk` passed `mem_adapter=None, corr_adapter=None`. The
fused-warp memory (51 ch: warped latent + validity + confidence + inverse depth, z-buffer
fused) and the two-sided correspondence field are trained into every checkpoint -- `mem`
15 tensors, `corr` 9 -- and encode exactly "background is visible through this gap". They
were absent from the regime in which SS30.12 measured cars merging and gaps filling with
foreground.

`--evidence on` now builds them per chunk from the CURRENT context window. In rollout the
window's later entries are the model's own output, so depth is estimated from the frames
in hand via `cached_consistent_depths(images=...)`, which bypasses the per-scene cache for
this purpose; cameras still come from the commanded poses and never from pixels. GEN3C
keeps an explicit 3D cache and re-renders it per camera -- this is the same idea routed
through branches the checkpoint already has.

Default remains `off` and the setting is recorded as `evidence_branches` in the report, so
the existing numbers stay reproducible and a new run cannot be mistaken for the old
protocol.

**2. Roll was unobservable, in both the metric and the battery.**

`axis_rotation_errors` splits orientation error into yaw / pitch / roll, reported beside
the WorldScore scalar. Calibrated: pure yaw 5 -> (5.00, 0, 0), pure roll 4 -> (0, 0, 4.00),
mixed (3,1,2) recovered exactly.

The battery gains a `lie_roll` band: the TRUE target camera rolled 10 degrees about its own
optical axis. Pre-multiplying `w2c` by a camera-frame z-rotation leaves the centre
(`-R^T t`) and the view direction (`R^T e_z`) invariant, so the perturbation is pure roll --
verified to 1e-10 on both, and 10.000 deg recovered.

This band is deliberately SYNTHETIC where every other band uses a real camera from the
trajectory. That departure is the point: DL3DV is handheld panning footage, so any two real
cameras differ almost entirely in yaw, and a model with perfect yaw sensitivity and zero
roll sensitivity passes every existing rung. Roll is also outside the action space, so a
rolled camera is a physically valid pose the command set can never request. The graded
ladder is untouched -- `gradedness` still contrasts real-camera severities -- and
`roll_binding` is reported separately.

**Not done, deliberately.** Stage-A retrieval stays at `retrieval_k=0`; Stage B already
carries 2, which is where the Wonder-style memory belongs. Turning it on at Stage A would
have confounded the resolution change now in flight.

**Nothing is re-measured yet.** These are instruments, not results. The four existing arms
have no roll number and no evidence-on rollout; both require a re-run.

### 30.14 The warm-vs-from-base gap was largely a measurement artifact (2026-08-10)

The rollout evaluator had always passed `mem_adapter=None, corr_adapter=None` (§30.13).
Re-running the same comparison with those branches live changes the conclusion.

**Baseline reproduces bit-exactly.** `reeval_evoff.json` matches `fid_arm2_6k_vs_warm.json`
to 9 decimals on every metric, after roughly thirty code changes today (LoRA merge path,
Q/K routing, contract awareness, resolution override, evidence plumbing). The default path
is provably unperturbed, so anything that moves below is the intervention.

**arm2@6k vs warm@3k, rollout:**

| metric | evidence OFF | evidence ON |
|---|---|---|
| PSNR | **-1.457** [-2.04,-0.81] * | -0.092 [-0.54,+0.34] n.s. |
| SSIM | **-0.070** [-0.09,-0.04] * | -0.002 n.s. |
| camera control | **-9.378** [-16.3,-4.5] * | -0.590 n.s. |
| LPIPS (real) | **-0.054** [-0.09,-0.02] * | +0.000 n.s. |

Every significant gap becomes non-significant. **The §30.5 headline -- "warm decisively
beats from-base, 0/8 scenes" -- was measured in a regime that disadvantaged the from-base
arm**, and that framing is superseded rather than merely qualified.

**The effect is asymmetric, which is the mechanism.** Scene-paired, evidence on minus off:
arm2@6k gains **+2.404 dB** [+1.598,+3.154] and **+10.395** camera control [+6.022,+16.774];
warm gains only **+1.039 dB** [+0.676,+1.368] and +1.607. Warm had learned to cope without
evidence it was trained with; the from-base arm depends on it. Evaluating without it
measured that coping skill, not world modelling. Both arms train with evidence in ~93% of
steps (clean + sparse + path), so evidence-off was the off-distribution condition all along.

**What still stands from §30.5-30.7.** The binding dissociation is unaffected: warm still
fails gradedness (+0.064) while arm2 passes (+0.535), and the battery has always run WITH
evidence. The step-count result (§30.7, more steps buy fidelity not binding) is within-arm
and unaffected. What is retracted is the cross-arm fidelity ranking and any claim that
from-base is behind on world modelling.

**Roll: every arm is effectively blind.** New `lie_roll` band, the true camera rolled 10 deg
about its optical axis:

| arm | roll lie | large camera lie | ratio |
|---|---:|---:|---:|
| warm@3k | +0.0012 | +0.402 | 0.3 % |
| arm2@6k | +0.0005 | +0.608 | 0.08 % |
| arm2@3k | +0.0004 | +0.595 | 0.07 % |
| arm1@3k | +0.0004 | +0.224 | 0.2 % |

A 10 degree roll costs 300-1000x less than a wrong camera. Intervals exclude zero, so the
models are not literally blind, but the magnitude is negligible: **no gradient enforces
roll**, which is why arm2@6k drifts ~5 deg without penalty (§30.12). Warm is 2.4x more
roll-sensitive than arm2@6k and drifts ~16x less -- consistent, though n=4.

**Frame diagnosis is now a tool, not a notebook.** `bench/diagnose_boards.py` measures lag,
inlier-gated roll and object-structure density from the boards. Its first summary reported
a 41 deg roll for arm2@3k, which was RANSAC noise from late steps with 6-8 inliers; gating
on >=15 inliers and taking a median instead of a max gives warm 0.27 / arm2@6k 0.85 /
arm2@3k 1.98 / arm1@3k 0.77 deg. Evidence-on boards and an off-vs-on diagnosis are queued.

**Cost of an error.** The 480x256 corpus (3.5 h) was unusable: I validated resolution
divisibility by 32 (VAE stride x patch) but FramePack pools old context at `old_stride=2`,
so the TOKEN grid must be even too and the real divisor is 64. 480x256 gives tokens 8x15
and dies at `grid (8, 15) is not divisible by stride 2`. Rebuilt at 448x256 (tokens 8x14,
scale 0.948, still a genuine downsample). The guard now derives the divisor from the stage
rather than hardcoding, with a test reproducing the exact failure.

### 30.15 After evidence, the residual failure is disocclusion (2026-08-10)

Follow-up on §30.14, examining the region *between* the two cars in fe4bd0d4 after step 6 --
the place the merging was worst. Boards: `arm_step_boards` (off) and `_evon` (on).

**Evidence fixes merging but not this.** Edge density in the car band recovered from 0.55 to
0.99 (§30.14). Yet inter-car PSNR still falls from ~21 dB at step 1 to ~11 dB at step 8 with
evidence ON, for every arm.

**Because the ground truth is disoccluding.** Contrast inside the gap RISES over the
rollout: 50.9 / 39.4 / 35.5 / 39.8 / 48.0 / **61.8 / 72.1 / 73.6**. As the camera translates,
the gap widens and exposes background that no context frame observed.

**Evidence helps measurably less exactly there.** arm2@6k at step 8, evidence on minus off:

| region | OFF | ON | gain |
|---|---:|---:|---:|
| whole frame | 11.82 | 14.70 | +2.88 |
| inter-car gap | 8.74 | 10.73 | +1.99 |

31 % less. This is structural, not incidental: the fused warp carries a validity plane, and
in a disoccluded region validity is **zero by construction** -- there is no observation to
warp. The branch honestly reports "unknown"; the content has to be generated.

**It also flattens the arms.** In the gap at step 8: warm 10.86, arm2@6k 10.73, arm2@3k 9.40,
arm1 8.96 -- warm and arm2@6k are indistinguishable, where whole-frame they differ. No amount
of training tells a model what is behind a car it has never seen past.

**Consequence for the roadmap.** Memory, retrieval and warping all move *observed* content to
a new view; none can serve disocclusion. That is a generative-prior problem, and it is the
right place to look once the native teacher lands -- a stronger backbone or a longer-horizon
generative objective, not a better geometry branch. It also sets a ceiling on what §30.14's
result can be extrapolated to: evidence closes the arm gap on observed content, and observed
content is not the whole frame.

**Incidental.** With evidence ON, arm2@6k beats warm on this scene from step 3 onward
(18.22 vs 18.62 at step 3, then 16.36/16.12, 15.81/15.04, 15.56/14.95, 14.70/14.21). One
scene, no interval, so it is an observation and not a claim -- but it is the reverse of the
evidence-off ordering.

### 30.16 Disocclusion confirmed directly, and a coverage figure retracted (2026-08-10)

§30.15 argued from mechanism that the residual rollout failure is disocclusion.
`bench/viz_holes_vs_error.py` tests it: for arm2@6k on fe4bd0d4, compare mean absolute
error inside the warp's hole mask against outside, per rollout step. No checkpoint is
loaded -- generations are cropped from `arm_step_boards_evon`, holes come from the cached
depth for the fixed opening context window.

| step | target | coverage | err in holes | err outside | ratio |
|---|---:|---:|---:|---:|---:|
| 1 | 12 | 0.974 | 0.160 | 0.064 | **2.50x** |
| 4 | 18 | 0.893 | 0.154 | 0.100 | 1.54x |
| 8 | 26 | 0.829 | 0.230 | 0.111 | **2.07x** |

**Error is ~2x larger where the warp has no evidence**, at every step. The context window is
FIXED at [0,2,4,6,8,10] throughout, so the coverage decay 0.97 -> 0.83 is purely the camera
leaving the observed set. Notably the ratio is already 2.50 at step 1 with 97 % coverage:
the holes are a thin strip at the frame edge and are the hardest 3 % of the image.

**RETRACTED: the coverage figures in the prior session note.** I reported "33-67 % of every
target frame has no warped evidence" from `viz_correspondence`, which uses evalkit's tuple:
**3 contexts spaced 90 frames apart, target 54 frames later**. The boards and every rollout
number use Stage A's protocol: **6 contexts at stride 2, first target 2 frames later**. Under
the protocol that is actually evaluated, coverage is 0.97 at step 1 and 0.83 at step 8 --
3 % to 17 % invented, not 33-67 %. The disocclusion mechanism stands and is now measured
directly; the magnitude I attached to it was from the wrong experiment.

`viz_correspondence` also writes one filename per invocation, so rendering three scenes in
a loop left only the last. Fixed to `vis4_correspondence_<scene>.png`; boards exist for
fff0ff50, fb80d413 and fe4bd0d4.

**Note on cost.** Recomputing coverage on uncached context windows falls back to running
VGGT, which OOMed against the live training job (training survived; the probe process died).
Windows outside the 1328-entry depth cache should not be evaluated while training holds the
GPU.
