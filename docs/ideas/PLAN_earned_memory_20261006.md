# Plan: earned memory, honest geometry, live camera (2026-10-06)

Supersedes the mechanism list in `analysis/plan_base_restart_20261004/PLAN.md` (the base-restart decision, horizon curriculum,
latent history, 8-latent teacher -> 2-latent DMD student and the standing constraints stay as they are). Deadline CVPR 2027,
16 Nov; six weeks. Deployment is unchanged: single image, exact target cameras at train and inference, autoregressive 3D memory.

## 1. What we measured (the problems this plan answers)

| # | Failure | Evidence | Status |
|---|---|---|---|
| A | Invented content becomes memory fact: a hallucination is lifted by DA3, rendered back at full confidence, reinforced | bush scene 06cb0994 morphs call by call; source freezing stabilises identity but opens holes | open |
| B | Depth of generated frames: far content re-lifted near, thin poles smear | fd4dc277 facade; depth is DA3 WITH pipeline poses + frozen scale, then ReSplat refinement (not monocular-blind) | runtime arm queued |
| C | Camera channel dead: translation is read from the scaffold, not from rays/command | Sept 2026 decisive surface; path ratio 0.76-1.05 by scene | training-time fix |
| D | LPIPS rewards blur; late realism (DINO/KID/sharpness) is what G4-400 improved | 200-update checkpoints blurry with LPIPS gains, 400 sharp with LPIPS losses | metric protocol fixed |
| E | Realized-pose registration untrained hurts; offset control is an open-loop fix on a proportional under-executor | parent late DINO 0.412 -> 0.365; conf+realized 0.468 vs conf 0.463 (4/8) | needs training exposure |

Confirmed runtime gains so far (G4-400, 8 held-out scenes): reliability (conf) mask -0.016/-0.022 LPIPS on seeds 7/11 with late
DINO/KID held or better; coverage-driven sources +-0; coverage+mask = mask.

## 2. Thesis for the paper (one sentence)

Memory content should earn its authority by re-observation under exact cameras, and the generator should be trained to be
faithful where memory is earned and free where it is not; everything else (depth, camera, losses) serves this.

## 3. Mechanisms, in priority order

### M1. Evidence state channel ("earned memory")  [runtime + training]
Each Gaussian carries an evidence state: observed (from the real input image), generated-once, cross-view-consistent (a later
generated call from a different commanded pose lands content in the same 3D cell with consistent colour/depth). Rendered as a
channel next to validity/confidence. "Consistent" is NOT "confirmed": the commanded camera is exact but its execution by the
generator is not, and the model re-producing its own output is not an independent observation. Evidence from the real input and
self-consistency of generated content stay separate states. Demotion on cross-view contradiction; no heuristics on content class.

### M2. Write-only-new-content + depth alignment  [runtime, queued now]
`memory/novelty_write.py` (rev 2 after review) in the base-restart runtime, applied inside ReSplat per read, stateless. This is
READ-TIME SUPPRESSION (the joint build/refinement still saw every view), not a persistent write-only-new memory:
- align: generated view's seeded disparity fitted (trimmed affine) to the disparity the EARLIER KEPT views render at that camera;
  accepted only with disparity spread, small ABSOLUTE post-fit error that improves on the raw one, bounded a/b, few pixels moving
  > 2x; rebuild once.
- mask: sequential in time; per pixel against the kept earlier views' render: same surface (|log depth ratio| < 0.10) -> opacity 0;
  new foreground (nearer) -> kept; conflict (farther) -> kept as uncertain; uncovered -> kept. A suppressed view no longer covers
  later views (CPU test).
Arms: both / mask / align on G4-400 + conf mask, seed 7, 8 scenes (`reliability_mask_20261006/run_novelty.sh`).
Decision metrics: LPIPS by call, late DINO/KID, sharpness, motion, Sampson; visual check on fd4dc277 and 06cb0994.
Far layer (content below a disparity threshold placed at a fixed far depth, rotation-only parallax) only if the aligned arm still
shows far-near errors; it cannot fix content that DA3 calls near.

### M3. Coverage-partitioned objective  [training, teacher]
Per-token loss weight from the evidence state. Earned/observed tokens: full flow-matching weight. Unearned tokens: reduced
flow weight plus a distribution-level term (teacher: scaffold-conditioned feature-space discriminator on low-frequency
differences; student: DMD already is one). Targets failure D directly: "copy the scaffold, blur the rest" is no longer optimal.

### M4. Loop-closure loss  [training]
k calls forward on-policy without grad, one call back to the anchor pose with grad, target = the real input image. The only
continuation that ever improved the 20-call forward metric was the revisit-row arm RV200; this makes it a loss with a real target
at the end of an on-policy chain. Invented content is penalised when it contradicts the anchor on return.

### M5. Camera carriers that cannot be ignored  [training; first training comparison]
Correction (2026-10-06, code check): the "geometry-aware expected flow" carrier ALREADY EXISTS in the parent as
`models/pixel_motion.py` (commanded displacement from memory depth, three fixed depth hypotheses where depth is unknown; the N1
carrier validated 2026-09-24, rel EPE 0.06-0.15 from real-memory depth) and feeds `omega_scaffold.pixel_motion`. So the lever is
not a new carrier but training that makes the model unable to read translation from scaffold placement alone.
First training comparison (rev 2 after review; equal budget, same init = G4-400, conf mask in rollouts, NO realized-pose
histories, NO scaffold dropout yet):
| arm | pixel-motion route trainable | scaffold-camera-decoupled rows |
|---|---|---|
| A control | no | no |
| B implementation fix only | yes | no |
| C fix + training intervention | yes | yes (p = 0.3: scaffold rendered from P' = P*delta; the render camera P' and the target P are both explicit through a P'->P displacement channel computed from the scaffold's own depth, zero when P' = P) |
PRIOR (joint_motion_pair_20260922): the adapter was trained jointly for 300 updates (G2) on standard rows and evaluated with
correct / zero / wrong displacement fields: identical metrics (LPIPS 0.1730/0.1731/0.1731, direction 7.90/7.80/7.92 deg). So B has
in effect been run once at G2 and learned nothing usable; B is kept as the implementation-fix control but A and C are the decisive pair.
Rationale as reviewed: "normal rows give the gradient no reason to open the channel" was not yet shown then; B tests whether a useful flow signal is
learned under plain flow matching once the route can learn at all. Scaffold dropout becomes a later ablation if C wins.
Four checks before training: (1) the flow-route parameters are in the OPTIMIZER GROUP, not only requires_grad; (2) the final
projection has a nonzero gradient on the first backward and its weights change on the first update; (3) they survive checkpoint
save/load; (4) A/B/C share weights, optimizer init and sampling settings (hash-verified).
Pass criterion, two stages, on the command-response probe re-run with the NEW conditioning form (scaffold fixed at its true render
camera P'; the target P and the P'->P channel change together, so no unexplained contradiction is measured):
  1. Is the channel learned? On conclusive samples g leaves 0 with the right sign, varies monotonically with the command magnitude,
     off stays small.
  2. Does it help normal generation? The consistent arm keeps g ~ 1; near-object deformation, real-camera tracking (Sampson) and
     late DINO/KID improve or hold. g ~ 1 w.r.t. the scaffold does not mean the scene geometry is right (0446d11e: motion response
     present, structural error persists).
Diagnosis as measured: "in the tested late calls there is no scaffold-independent motion response to TRANSLATION commands";
camera rotations were not varied, so this does not generalise to all camera control.
BASELINE MEASURED (2026-10-06, analysis/command_response_g4_20261006): camera carriers g = 0.000 in all 24 samples (image changes
< 1.2 grey levels), consistent arm g ~ 1.0: the channel is completely closed and the scaffold alone sets the motion. The pixel-motion
route is zero-initialised and NOT trainable in this lineage (never trained); camera_flow == camera_only bit-exactly. M5a therefore
must (i) add omega_scaffold.pixel_motion to the trainable set and (ii) use scaffold-camera-decoupled rows; the probe is the pass test.

### M6. Second ring (if time): depth-free epipolar attention bias with an evidence-gated window; exact-pose epipolar consistency
loss on DiT features; generator-side depth latent. Not in the first teacher.

## 4. Experiment queue and gates

| step | what | gate to continue |
|---|---|---|
| done 06 Oct | closed-loop offset arm: FAILS (0.477 vs 0.463, integral windup on 0446d11e). M2 arms: nov-mask FAILS (late DINO 0.454 vs 0.494), both FAILS, nov-align borderline (LPIPS 6/8, DINO +0.005, KID -0.006); seed-11 replicate of align: LPIPS null (0.461 vs 0.462, 3/8), late DINO +0.029 and KID equal on s11 (s7: DINO +0.005, KID worse); candidate for a later rollout-training arm, not in the first comparison. See reliability_mask_20261006/README_novelty.md | NO M2 policy enters the teacher recipe; first training comparison = M5 three arms |
| next | short matched continuation on G4-400 (equal budget, same init): the ONE M2 intervention that passed, ON vs OFF in the training rollouts | ON improves late DINO/KID on dev scenes -> policy is in the teacher recipe |
| teacher | base restart Phase 1 with the passed runtime policy + M5 carriers + horizon 2->4->8 latents; M1 channel, M3 and M4 enter ONE AT A TIME as separate arms, not together | G1: first-call LPIPS <= 0.13 and camera <= parent, else parent + curriculum (unchanged fallback) |
| student | 2-latent DMD from the 8-latent teacher | stretch; teacher is the CVPR-safe system |

Evaluation protocol (unchanged): 160 frames, 8 held-out scenes, seeds 7 and 11; LPIPS reported, decisions on DINO/KID +
sharpness band + motion + Sampson; same KID subsets/seed across arms.

## 5. Timeline (six weeks)

- w1 (to 12 Oct): M2 arms, continuation ON/OFF, Phase-1 harness written (rows, evidence channel, M3/M4 losses, M5 carriers).
- w2-w3: teacher training from base, G=2 then 4; G1 gate at the end of w2.
- w4: G=8, M4 on; held-out evaluation, ablations (M1 channel off, M3 off, M4 off, M2 off).
- w5: DMD student if the teacher passes; figures, viewer, tables.
- w6: writing, replicate seed 11 for the main table.

## 5b. Review notes (2026-10-06)
- M3's reduced flow weight on unearned tokens does not follow directly from the blur finding (blur = LPIPS artefact at 200
  updates); it is a separate hypothesis and gets its own arm.
- The realized-pose arm (E) mixes a camera-free DA3 pose with a commanded-pose-conditioned DA3 depth
  (gate_omega_realized_pose.py); its loss can come from training mismatch, pose error, or that pose/depth inconsistency. It
  supports not deploying it; it does not identify the cause.

## 6. Risks
- M3's discriminator destabilises the teacher: fall back to reduced flow weight only (no adversarial term).
- M4 needs on-policy rollouts in the loop: 1-2 rollouts per step at G=8 is affordable on one GPU only with few-step denoising; use
  the one-step draw already in the self-aug path.
- M2 alignment can lock the generated view to a wrong earlier depth; trimmed fit + MIN_COVERED_FRACTION guard; 'mask' alone is
  the fallback.
- Realized-pose rows in training (M5) reintroduce DA3 measurement noise; keep the catastrophic gate and log the used fraction.

## 7. Diagnostics before the paper's main claim is chosen (2026-10-06, after two literature reviews)
Literature check (opened): AlayaWorld v1.1 (2608.13492) removes its camera AdaLN branch and controls viewpoint by re-rendering the
point cache along the planned trajectory; GEAR (2609.34722) keeps frame latents and uses geometry only to route attention, with an
appearance-free "Invisible Octree" for occlusion; WorldPlay2 (2609.35560) Stable Forcing (few-step init, full-rollout replay,
clip-wise score), WBench 83.1 / RevisitBench LPIPS 0.318, discrete moves + continuous pitch/yaw; CAER (2608.30897) reweights
supervision by the effect of removing the action under shared noise. Per the review (not opened here): Wonder sec 4.2.3 ties camera
drift during causal rollout to blur/collapse and adds camera-aware adversarial supervision; Robust Dreamer sec 3.3 already names the
write-recall contamination loop; Lyra 2 ablations report worse controllability with globally fused memory and without self-aug.
Consequences:
- "The explicit camera route is ignored" is an architectural diagnosis; full-pipeline control is not shown to fail (consistent arm
  g ~ 1). The candidate headline claims are (a) calibrated camera execution under self-generated history and (b) non-amplifying
  geometric memory. Neither is chosen yet; both overlap prior work and need an isolated failure first.
- Identifying the write-read loop is prior work (Robust Dreamer); a contribution must MEASURE its amplification and beat
  deviation-aware training.
Diagnostic program (GPU, after M5; all on saved states with bit-exact replay):
D1 Natural-rollout camera execution at several ages (branch at calls 1, 3, 5, 10; consistent arm as the main measure). Report
   per age: absolute execution error in a fixed gauge (did it reach the requested pose), intervention response g, executed
   displacement by magnitude, off-direction motion, stalls/reversals, tracking coverage and failed measurements. Rotation
   (yaw/pitch +-1.5 deg) as the depth-free reference next to translation.
D2 Same scene, pose, command and history layout with CONTROLLED state corruption (known geometry): missing coverage and wrong
   depth as separate interventions, measured errors (S1) not arbitrary noise. Establishes whether state corruption causes the
   deterioration D1 may show. Never swap an unrelated clean history into a drifted memory (FL lesson).
D3 Single-perturbation feedback probe: one localised memory error injected ONCE (depth shift / swapped source / thin-structure
   smear), identical commands and noise, A(j) = E(j)/E(0) on a FIXED persistent-surface set (lost surfaces = failures), error
   against an INDEPENDENT reference as well as branch disagreement. Matched control that blocks the perturbed content from
   entering later memory writes while keeping the camera path and coverage: separates write-loop amplification from
   temporal-history propagation.
D4 Call-length confound: G4 vs G2 over 160 frames differ in the number of memory write boundaries (10 vs 20); any claim about
   longer joint generation needs a matched-boundary comparison.
Training ideas held until D1-D3 read out: response-preserving control objective (camera intervention + state corruption from the
same scene, visibility-masked L1 on motion deltas, with a differentiable motion proxy; final eval on sampled videos with
independent measurement); amplification penalty max(0, E_next - rho * E_cur) only with a fixed/stop-grad threshold, an
external correctness anchor, and an explicit credit-assignment path (detached DA3/ReSplat gives no gradient through the loop).
Controls that any such arm must beat: camera-reward only; corrupted-history only; their plain combination; M5 decoupled rows.

## 8. Positioning after the October literature check (2026-10-06)
Training regimes of the closest systems (opened): WorldCrafter = teacher forcing with RANDOM real history frames, 4 stages,
joint encoder+DiT, pyramid DMD, no exposure-bias handling; MoVerse = teacher trained on a CURRICULUM OF CORRUPTED RENDERS
(warp proxies with holes, Gaussian floaters/aliasing), FM on target tokens only, student by Self Forcing + DMD with RAVEN
repacking, fixed scaffold; GeoVerse = 300k + 50k iters in DA3 feature space, FM + RGB + HF + depth losses, memory written
back with equal weight for observed/synthesized; WorldPlay2 = TF AR -> PDD -> Stable Forcing (full 320-latent student
rollouts replayed with DMD2), learned compressed memory; MeRoPE = 20k iters single clip on Wan2.2-5B. None trains the model
through an on-policy GEOMETRIC write-read loop; the one that writes back (GeoVerse) reports drift accumulation as its limit.
Our scale: 3.3-4.2k scenes, 10 rows/update, hundreds of updates on one GPU; theirs: 10^5-10^6 clips, 32-64 GPUs. No claim
can rest on scale or on a better memory encoder. The defensible contribution is what scale does not buy:
  (1) a controlled measurement of the geometric write-read loop (bit-exact state branching, response gain, feedback impulse
      response A(j) with a write-blocking control, fresh-vs-aged execution), and
  (2) a mechanism that makes the loop non-amplifying at matched budget (evidence state + partitioned loss, M1/M3), beaten
      against the two cheap in-house baselines that mirror the field's answers: NO WRITE-BACK (fixed scaffold after call 1 =
      our source-freeze code, MoVerse/WorldWeave) and DEVIATION-AWARE TRAINING (our EB error-bank arm, Robust Dreamer/Alaya),
      plus an implicit-memory reference (Lyra 2 numbers; WorldCrafter if weights appear).
M6 (epipolar bias) is withdrawn as a contribution: MeRoPE publishes it on our base model. Camera-execution retention is not the
headline unless D1/D2 show a loss that our Sampson data (G4-400 direction error flat late) does not show.

## 9. Low-budget contribution package (2026-10-06)
Budget facts: 1 GPU, ~2 min/update (G4, 10 rows), 200-update arm = 7 GPU-h, 160-frame validation = 1.2 GPU-h, probe = 0.8 GPU-h.
Six weeks ~ 900 GPU-h at best. The base restart (8-latent teacher + DMD student) does not fit: DROPPED for CVPR; the system is
the parent lineage (G4-400 + conf mask) plus whatever passes below. Contributions chosen so that cost scales with rollouts
and conditioning, not with data or parameters:
C1 Loop-closure self-supervision (M4): the input image is the only truth guaranteed at deployment; rows go k calls out
   (one-step draws, no grad) and one call back to the anchor camera (grad, target = input image). Cost = (k+1) calls per
   row. Prior: RV200 return rows were the only continuation that improved 20-call forward fidelity. Distinct from
   WorldPlay2/RevisitBench (revisits only evaluated) and Memorizon (supervised long spans with recorded video).
C2 Evidence-state memory + partitioned objective (M1 + M3): a conditioning channel and a per-token loss weight; trainable set
   unchanged. Distinct from Lyra/Mirage class filters, Alaya token dropout, GeoVerse equal-weight write-back.
C3 Learned memory-write gate (new, "M7"): a tiny per-pixel module (inputs: DA3 confidence, render validity, generated-vs-render
   disagreement, evidence state) outputs the write weight of each generated pixel into memory; trained with the GENERATOR
   FROZEN on the loop's own signal (next-call error on persistent surfaces / amplification A(1)) by a differentiable proxy
   through the next render, or bandit updates. First learned write policy for a generative 3D memory; every system today
   writes by hand rules. Cheap: generator frozen, module ~1e5 params.
C4 The loop audit (D3 + D1/D2) as the evaluation spine; the claim is NON-AMPLIFICATION at matched budget against no-write-back
   (source freeze) and deviation-aware (EB arm) baselines, Lyra 2 as implicit reference; sharpness/motion/camera bands enforced.
Order: M5 (running) -> D3 probe (1 day) -> C1 arm and C2 arm (7 GPU-h each + validation) -> C3 (2 days) -> replicate seed 11
-> paper. Base restart, M6, closed-loop offsets, read-time suppression: out.

## 10. Execution-aware re-rendering (camera-control contribution, 2026-10-06)
On-the-fly camera measurement (camera-free DA3 + anchor-rigid map; RoMa/ViPE for scoring) is NOT used as a reward (CamVerse,
CamPilot, WorldReward, PartiCam) nor as a command correction (our offset arm wound up) nor as a selector (chance-level in our
selection test). It is used as a CONDITION: after each call the rigid delta realized*inv(commanded) at the endpoint is applied to
the next call's commanded cameras to give the scaffold render cameras P'; the requested cameras P stay the targets and the
render-offset field P' -> P states the remaining motion. Memory is written at realized poses. Fresh delta per call (no integrator),
gated by the measurement's own gate and by |t| <= one call's displacement. Needs a model trained on decoupled rows (arm C).
Runtime: memory/execution_rerender.py, rollout(render_from_realized=True, pose_policy='da3_anchor_rigid', render_offset=True);
validate_g4.py env RENDER_FROM_REALIZED=1. Prior art re-renders along the PLANNED trajectory (AlayaWorld) or writes at the
commanded pose (Lyra 2, Robust Dreamer). Test at C200: C default / C re-rendered / C re-rendered gated / A re-rendered (untrained
control, expected to fail as before) on D1 absolute execution error + response gain and the late realism/sharpness/motion bands.

## 11. Camera control with online execution feedback (leading camera hypothesis, 2026-10-06, after review)
Claim candidate: a controller that uses the measured local response of the generator (J_t from the command-response probe) and
the observer's calibrated reliability to apply ONE bounded correction to the current chunk BEFORE it is committed to history and
memory. Distinct from camera rewards (CamPilot, CamVerse, WorldReward) and from our failed open-loop offset (accumulated, applied
to the command the generator does not read). Caution from our own data: the boardwalk probe showed the parent's x1.0 branch reads
0.838 progress while the REAL-frame control reads 0.859, i.e. most of the apparent translation under-execution was observer bias;
corrections must be relative to the observer's real-frame calibration. Yaw first (rolling ViPE: real 0.158 deg vs generated
1.046 deg median, above observer noise).
E1 (analysis/yaw_correction_20261006, queued after M5): 8 states (4 dev scenes x calls 3/8); baseline, +1 deg probe, bounded
response-calibrated correction (|delta| <= 2 deg, J >= 0.3), vs 3-sample camera-based selection at matched budget; in-loop observer
= camera-free DA3 + anchor-rigid; offline ViPE/RoMa corroboration; pass/stop pre-registered in PROTOCOL.md.
E2 only if E1 passes: commit each candidate and continue 5 calls without further correction; decision table: immediate+downstream
gain -> feedback coupled to memory; immediate only -> execution-accuracy contribution; downstream damage -> no integration; none ->
stop controller work. Later: the correction search becomes training data for a small correction module (generator frozen).
E1 RESULT (2026-10-07, yaw_correction_20261006/RESULTS.md): FAIL by the stop clause. In-loop gain 8/8 and better than
selection, but not corroborated by an independent two-view increment observer (corrected 0.86 vs baseline 0.43 deg median) or
by RoMa absolute heading vs the recorded endpoint (valid 4/8 states, corrected better 2/4, sign disagreement with the in-loop
observer 2/4). Per-call yaw execution at G4-400 is within observer noise. Controller work stops; E2 not run. The measurement
tooling (RoMa absolute heading, calibrated at 0.17 deg on real frames) stays for D1 at early calls.

## 12. M5 closed (2026-10-07, training_runs/m5_carriers_20261006/RESULTS.md)
Stage 1 fail for B and C (camera arms g = 0.000; C decoupled g 0.36/0.10/0.10, consistent g drops to 0.74-0.79). Stage 2: A/B
lower LPIPS (0.451/0.453 vs 0.463) but late DINO 0.459/0.464 vs 0.494 and over-sharp 1.17-1.18; C worse (0.501, collapse on
0446d11e). System stays G4-400 + conf mask. Camera-route line closed for CVPR (scope: additive sparse routes at this budget).
Remaining camera work: measurement only (D1 with RoMa early + rolling ViPE), no controller. Next GPU work: D3 feedback
amplification probe, then C1 loop-closure arm and C2 evidence-state arm.

## 13. D3 result (2026-10-07, feedback_d3_20261007/RESULTS.md; wording per review)
Single known-surface errors (depth x1.3, blur) injected once at call 3: no local amplification (A_out 0.8-1.3 over calls 5-9);
the injected difference persists, and replacing later records with clean content reduces that persistence (write-blocked
branch 0.2-0.7 over calls 5-9; the call-10 decay rule met in 2 of 3 measurable scenes). Global spread of the difference with no
clear fidelity loss shown by LPIPS (small LPIPS gaps do not rule out local structural damage). The pre-registered "growth
through the loop" rule is NOT met. D3 did not test the traced bush/vine mechanism (invented content in an uncovered region).
D3b (feedback_d3b_20261007, narrowed per review): not "does the invented object persist" but "does re-writing it through
generated views deform it" vs reading it from a FIXED geometric record; both branches accept the object; metrics = structure
preservation under known cameras (DINO/LPIPS/SIFT/edge IoU against the object's own record warped into each view) with the
surroundings' fidelity reported separately. Pilot on 06cb0994 + 2 dev scenes. No training arm opened on D3 alone.

## 14. D3b result (2026-10-07, feedback_d3b_20261007/RESULTS.md)
Invented objects exist at call 2 in 5 of 12 scenes (3-20% of the image). Fixed-record vs rewritten: pre-registered structure
condition met weakly (06cb0994 clearly, fc18f248 partly, fc18ec31 marginal, 0446d11e not on footprint scores, fb95b529
inconclusive). Unplanned: in the two largest-object scenes the region-limited fixed record PREVENTS the late global collapse
(06cb0994 staircase chain, 0446d11e smear) with no surroundings cost. Seed 11: replicates on 0446d11e (collapse prevented again),
06cb0994 does not collapse at seed 11 and the freeze is neutral/slightly worse there. Reading: a COLLAPSE GUARD (3/3 collapse
cases prevented), not a general gain. Next decision point: full-protocol test of the automated policy (8 held-out x 2 seeds vs
mask-only) before it is called a mechanism; then it is the first rule for C3.

## 15. Write-policy test (2026-10-08, write_policy_20261007/RESULTS.md) and the new priority
B (pin only) = A; C (pin + regional suppression) within seed spread (late DINO +0.009 s7, -0.011 s11; worst per-scene harm -0.073).
Rule 4: no mechanism, no deployment change, no gate motivation. Memory write/read policies are exhausted as a lever at this stage.
New priority (user, 2026-10-08): establish whether G4-400 is a good CONTINUATION generator at all: clean-continuation control
(analysis/clean_continuation_20261007, recorded consistent history + memory vs own state, 4 dev scenes, pre-registered table);
then train only the side it points to (clean continuation first, or robustness with target-valid controlled corruptions).
Clean-continuation control (2026-10-08, clean_continuation_20261007/RESULTS.md): clean late LPIPS 0.213 vs early 0.214, late DINO
0.849, KID 0.004; own state late 0.499 / DINO 0.544 / KID 0.076. Row 1: G4-400 continues well from a correct state; the failure is
robustness to its own degraded state. Before designing the robustness rows: a 2x2 split (recorded history + own memory, own history
+ recorded memory) to find which part of the state carries the damage.
Training comparison after the 2x2 (agreed 2026-10-08): ONE controlled arm vs an equal-budget control, both from G4-400; clean
continuation rows kept; corruption applied to the dominant path from the 2x2, on the SAME scene with a VALID target, with type and
strength matched to errors measured in real rollouts (not random noise). Decision needs BOTH: clean-condition ability preserved
AND long own-state rollouts improved (fixing one corrupted call is not stability when the fix is written back into memory).
2x2 (2026-10-08): recorded history alone closes 70/80% of the own->clean gap, recorded memory 35/32% -> temporal history dominant.
Calibration: training one-step history (cap 500) LPIPS 0.14-0.22 vs own deployment history 0.30-0.52; cap 950 matches (0.39).
DECISION: single training arm H = A + self-augmentation cap 950 (training_runs/history_robust_20261008, PROTOCOL.md), 200 updates,
control A200; pass needs own-state late DINO >= A200+0.02 with KID/sharp/motion held AND clean ability preserved (dev late DINO
>= 0.82). Launched 2026-10-08.
H200 RESULT (2026-10-08, history_robust_20261008/RESULTS.md): vs A200 late DINO 0.511 vs 0.459, KID 0.045 vs 0.062, motion 1.03,
clean ability kept (0.845), dev own-state 0.576 vs 0.544 - but sharpness 1.32 (band <= 1.3) -> condition 1 fails on sharpness only.
Vs G4-400+mask at seed 7: DINO +0.017, KID better, LPIPS 4/8, over-sharp 1.32 vs 0.97 -> not adopted; seed 11 pending.
