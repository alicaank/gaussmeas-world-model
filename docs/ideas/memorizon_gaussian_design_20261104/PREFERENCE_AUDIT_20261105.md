# Audit of the design preferences

**Later experimental evidence:** [design after the drift tests](DESIGN_AFTER_DRIFT_TESTS_20261105.md)
supersedes the 2Xplat and training priorities in this initial audit. It also records
that correspondence delivery has since been implemented for LoopRoller2X; the
ReSplat route still needs its own integration check.

5 November 2026. This audit separates published ablation evidence, local code
facts and our untested choices. It changes the consolidated design; it does not
launch experiments or modify the existing research implementation/queue.

**Follow-up:** [verification of the implementation review](REVIEW_VERIFICATION_20261105.md)
confirms the newly added restricted gr branch and passing CPU tests. The older
dense-only description below applies to gb. Evaluation currently bypasses both
branches because no corr reaches the sampling forward. The endpoint-only concern
is confirmed by the actual imported ray helper and saved camera weights; it is
not merely a terminology issue. Continue the adapted checkpoint for the pilot;
adapter-only restart and explicit feedback as a headline are not required.
The follow-up also finds a new single-view normalization branch and gauge-pose
cache signature. These supersede the corresponding original code descriptions
below; runtime startup, content/scene identity and registration remain to qualify.

## Evidence and decisions

| Previous preference | Evidence checked | Revised decision |
| --- | --- | --- |
| RGB conditioning as the best short-budget representation | RobustDreamer Table 3: RGB-memory LPIPS 0.402 versus latent-memory 0.351; L3G-World Table 3 also favors VAE lifting over rendered-RGB encoding | Retain RGB renderability and the existing structural path for implementation economy. Make direct source-feature reading the appearance route. Do not claim RGB is better or that this hybrid beats latent splatting. |
| Soft geometric attention / weak dense bias | GEAR Section 4.3 favors restricted candidates with matched parameter count | Keep restricted auxiliary attention and the native backbone fallback. Existing dense-bias code is a distinct baseline. |
| Native temporal tokens can be addressed as points using one camera | GEAR Appendix A.2 explicitly changes to framewise encoding to avoid temporal ambiguity | Preserve the main Memorizon codec. Prefer separately encoded selected source keyframes for the added reader; aggregate target subframe candidate support. Audit the adaptation; do not pretend it reproduces GEAR. |
| Thirty hours of clean-history adaptation before recovery | RobustDreamer Table 3: no-deviation FID 48.20, Gaussian-noise 47.50, full 16.82; GEAR also ablates degraded history | Include realistic imperfect histories after a short warm-up. Exactly how early to introduce them is our budget choice, not established by these tables. |
| Longer spans and more memory should help | Memorizon Section 4.3 finds span benefits saturate and longer spans can trade image quality for retrieval | Select spans by actual revisit coverage; cap the native bank and preserve scene diversity. |
| Append/retain a scene and expect better reconstruction | L3G-World Table 3 shows a refinement benefit; its Table 4 has mixed outcomes for some metrics | Do not claim a static Gaussian cache repairs geometry. Reconstruct/validate bounded keyframes for this pilot; a learned updater remains outside budget, not disproved. |
| A four-hour reward-weighted FM stage is a strong expected improvement | CamVerse and Geo-Align use actual GRPO; their results do not validate our weighted-imitation shortcut | Treat weighted FM as an unverified optional pilot, paired with uniform replay if run. Otherwise use the reserve for ordinary recovery FM. |
| Better pose reward implies better video | CamVerse Table 1 improves I2V translation 0.0337 to 0.0286 while VQ falls 5.02 to 4.87 | Keep motion and visual outcomes separate; no reward checkpoint promotion from pose scores alone. |
| 2Xplat gives usable native-scale camera poses automatically | 2Xplat normalizes translations and reports evaluation-time camera alignment; its Table 12 separates rendering quality from pose quality | Pin the released checkpoint/recipe and expose context-only poses. Native metric calibration, stationary startup and online registration are separate qualification tasks. |
| Feedback modulation is the most likely novel improvement | No reviewed ablation establishes our execution-history embedding | Keep one cheap, falsifiable hypothesis with a matched control. It is not an established advantage or sufficient novelty by itself. |
| Width 128, four insertions, adapter-only updates are proven adequate | GEAR uses a different backbone/codec, wider modules and LoRA; our values came from engineering judgment | Record these as capacity/profiling defaults. If adapting capacity, change both matched arms; no claimed optimality. |
| The new method must be built entirely in the clean repository | Existing research files include 2Xplat memory, Gaussian FM, pool collection and an adapted Memorizon-50 lineage | Audit and reuse this route before rebuilding. Clearly distinguish implemented pieces from the proposed restricted reader and feedback branch. |

Sources: [RobustDreamer, §4.3](https://arxiv.org/html/2605.30855),
[L3G-World, §4.4](https://arxiv.org/html/2610.40614),
[GEAR, §3.4/§4.3/Appendix A.2](https://arxiv.org/html/2609.34722),
[Memorizon, §4.3](https://arxiv.org/html/2610.00544),
[CamVerse, §4](https://arxiv.org/html/2512.02870),
[Geo-Align, §4.5](https://arxiv.org/html/2605.23903),
[2Xplat, Additional Details/Analyses](https://arxiv.org/html/2603.21064).
These are reported results from different settings, not comparable leaderboard
scores or replications. The review does not establish a universally best design.

## Corrections from inspecting the research implementation

The older [pilot plan](PILOT_PLAN_CVPR_20261104.md) points to
`analysis/plan_base_restart_20261008/fresh_teacher_20261021/`. It was missing from
the earlier readiness summary. Files inspected there establish the following:

- `train_g.py:step_loss_g` constructs noisy targets and trains velocity with
  squared error. There is already a Gaussian FM route. Native-scheduler and
  measured-input equivalence still require qualification.
- `packed_model_corr.py` originally only added correspondence bias to dense
  logits (gb). A subsequent gr branch hard-masks correspondence access in added
  readers, with zero-output initialization and empty-read masking. It still uses
  dense matrices, not gathered sparse computation. Zero-init identity is not
  evidence of correspondence quality, inference wiring or GPU scalability.
- `gen_onpolicy2.py` collects corrected histories but prepares the next supervised
  query at recorded P. `rows_g.py` pairs that query with recorded future latents.
  This can be camera-compatible recovery-to-recording supervision, subject to
  appearance compatibility. It is not proof that those query cameras equal the
  deployed controller's actual next U. Store both, check equality, and label
  changed-query replay separately. Do not call a one-time frozen pool on-policy.
- `rows_g.py` replaces history endpoint poses while constructing full-rate rays
  from the remaining recorded poses. Endpoint measurements do not make every
  subframe measured. Require full-rate observations or explicitly identified,
  validated interpolation/missing-data treatment before a full-rate measured claim.
- `twoxplat_memory.py` implements bounded global re-reconstruction; it is not the
  persistent local-submap system previously proposed. Its current cache identity
  is based on selected indices. Content/scene/gauge/pose revisions also need to
  invalidate it. Refit consistency for accepted old cameras must be checked.
- `LoopRoller2X.start` invokes reconstruction with the anchor alone. The upstream
  normalizer constructs a canonical frame and divides by camera extent; zero-
  baseline startup needs an explicit supported path. Duplicating the anchor or
  setting metric depth to one does not solve observability.
- `run_op/meas/ckpt/trainables_00250.json` identifies an existing adapted
  Memorizon-50 checkpoint, via `run_v864/mem` and `run_plain350`. Its recorded
  `scale_mult=2` affects the native camera contract. This audit inspected metadata
  and code, not all weight hashes or quality. Reuse is a candidate, not automatic
  promotion; do not replace its learned convention with another scale silently.

An anchor median-depth gauge of one is a coordinate choice, not a claim that the
scene has metric depth one. The conversion to the chosen checkpoint's native
units must have its own causal calibration provenance. Shared-view registration
maintains a gauge; it does not create metric information absent at initialization.

## Boundaries of this audit

No evidence establishes the optimum adapter capacity, history mixture, error
normalization, reward weights or GPU-hour split for our model. Those remain
declared starting choices. Full-size loading, backward and rollout checks have
not been performed by this documentation review. Existing queues, code, weights
and protected checkpoint dependencies are untouched.
