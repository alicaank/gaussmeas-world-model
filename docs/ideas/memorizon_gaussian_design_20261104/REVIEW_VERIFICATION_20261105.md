# Verification of the follow-up design review

5 November 2026. Scope: reopen the cited primary papers, inspect the current
research code in its declared environment, and run targeted CPU checks. No GPU
rollout, training launch or production-code modification was performed.

## Verdict

Accept the paper corrections and the existence of the restricted reader. Reject
the assertion that interior RGB poses cannot affect the model. Evaluator detection
exists, but correspondence delivery to the sampling forward is missing: the reader
is bypassed in the inspected evaluation path. Fix this before interpreting gr/gb
rollouts. Prefer continuing the adapted checkpoint over an unmotivated restart;
keep execution-history embedding as a secondary matched experiment.

## Paper checks

- [GEAR Table 2 and Section 4.3](https://arxiv.org/html/2609.34722#S4.T2):
  dense/restricted LPIPS 0.5995/0.4459, ATE 0.2924/0.0436. The dense control keeps
  the GCA module and parameter count. This is strong evidence in their setup,
  not a measured result for our reader. Their training freezes original weights
  while learning LoRA and GCA; 10,000 iterations at global batch 32 means 320,000
  sample presentations, not necessarily distinct examples. Framewise encoding
  differs from our temporal-token endpoint approximation.
- [Robust Dreamer Section 3.2 and Table 3](https://arxiv.org/html/2605.30855):
  image/pose tokens predict global pointmaps used as Gaussian centers. Calling
  this commanded-pose placement is unsupported. More precisely, the paper shows
  image-inferred global geometry, not necessarily our explicit measured-camera
  ledger. RGB-memory/full-latent LPIPS is 0.402/0.351. Its deviation training also
  means training with imperfect history alone is not our distinguishing claim.

The defensible proposed difference is the explicit observer-to-command-correction
loop, recovery training from its states, and the chosen reconstruction/update
policy. Global re-reconstruction is a mechanism difference, not an established
novelty or advantage. Explicit execution embeddings remain an ablation.

## Code findings

### 1. Evaluator loads the branch but never supplies its correspondences

`closed_loop_864.py:42` detects `reader.*` and selects restricted mode.
`LoopRoller2X.pack` returns no correspondence field; inherited
`closed_loop.py:denoise` calls `sample.py:Roller.velocity`, whose model forward
does not pass `corr`. `packed_model_corr.py:39` immediately returns the base
forward when `corr is None`. This bypass affects gb as well as gr.

Required fix: construct causal source/target correspondences for the actual issued
U from the accepted scene, pass them through pack/denoise and both CFG forwards,
and distinguish unavailable/empty geometry from missing wiring. Do not use actual
future RGB to create target depths. Add an integration check with nonzero reader
weights and a supported correspondence; zero-init parity cannot detect this bug.

### 2. Interior RGB poses do enter a trained camera branch

Using `fresh_teacher_20261021/env.sh`, Python resolves `foundation_model` to
`training_runs/joint_samplerfix_20261012/foundation_model.py`. Its `packed_rays`
uses frames 4p-3 through 4p for each latent p: four sets of six Pluecker channels.
PRoPE and the correspondence camera use the endpoint, but the additive ray branch
does not. `rows_g.py:65` replaces only endpoints in a recorded-pose array, then
passes that array to `packed_rays`; `packed_model.py:131` adds `self.camera(rays)`.

CPU check using mmap-loaded camera weights from
`run_op/meas/ckpt/trainables_00250.pt`:

| Quantity | Result |
| --- | ---: |
| Camera weight shape | 3072 x 24 x 1 x 2 x 2 |
| Maximum absolute camera weight | 0.00732421875 |
| Change only RGB frame 1 translation; hold endpoint frame 4 fixed | 0.1 native-coordinate units |
| Maximum packed-ray difference | 0.0975342914 |
| Maximum additive camera-embedding difference | 0.0016276594 |

This proves an input effect, not a measured video-quality regression or its
magnitude. The current pool mixes recorded interior poses with measured endpoints.
Choose full-rate measurement or an explicit interpolation convention consistent
between training and inference; do not call interiors irrelevant. The temporal
feature-address approximation is a separate issue and still needs qualification.

### 3. Restricted semantics are implemented, sparse computation is not

`mode='restricted'` creates zero-output PRoPE readers every second block for gr,
hard-masks unmatched keys and zeroes reads with no candidates. Native attention
is retained. `corr.to_dense()` and dense masked SDPA remain; this is restricted
attention implemented densely, not a gathered sparse kernel. Profile actual
forward/backward peak memory before dismissing the cost because other masks exist.

Also, gr versus gb is not GEAR's matched-capacity dense/restricted ablation:
gr adds attention projections while gb adds scalar biases. It is a useful recipe
comparison. To attribute gains specifically to restriction, use the same reader
with unrestricted historical keys as the matched control.

## Validation performed

- Existing `tests_packed_corr.py`: exit 0; identity/gradient with and without
  PRoPE, causal-mask rejection, geometry matrix ordering, restricted read checks.
- Additional CPU stand-in DiT: nonzero reader weights with absent corr produce
  exactly the base output; supplying corr changes output (max difference 0.32534).
- Interior-pose perturbation above used the actual saved camera projection, not
  randomly initialized weights. Full backbone loading and checkpoint hash
  re-verification were not part of this check.

These tests establish local behavior only. They do not qualify correspondence
accuracy, temporal alignment, full-size GPU memory or trained visual outcomes.
The reported earlier local win counts and approximately 2,000 rows per arm were
not independently recomputed in this verification.
