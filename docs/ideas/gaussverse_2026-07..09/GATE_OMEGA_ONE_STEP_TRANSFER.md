# Gate-Omega one-step treatment: transfer and launch

This is a bounded treatment from the cliff-free step-1200 carrier. It changes
one training factor: the 30-forward generated-history sampler becomes a
Lyra-style, one-forward, detached-x0 reconstruction of the complete G=2 chunk.
The logical eight-row objective and its weights are unchanged. It is not a
claim that the complete Lyra training recipe was reproduced.

The launch is fail-closed. Every 50 treatment updates it measures the fixed
eight-row all-history noise sweep and a parent-generated depth-at-least-20
state bank. It stops when clean flow, sigma-0.65 flow, or deep generated-state
flow regresses by more than 5%, or when any sigma-0.65 row enters the attractor.
A passing 250-step result is still only a candidate: the frozen 24-scene
Stress-8 and four-scene, three-seed, 27-hop evaluation remains blocking.

## Required artifacts

Transfer the repository at the recorded commit plus these non-Git artifacts:

- `outputs/recurrent_train_manifest_v1.json.gz`
- `outputs/text_context_probe/universal_context.safetensors`
- `outputs/gate_omega/init_wan22_walking_v3/`
- `outputs/gate_omega/row_cache_production_walking_v12/`
- `outputs/gate_omega/scaffold_calibration_walking_v2.json`
- `outputs/gate_omega/certification_acceptance_v5_commanded.json`
- `outputs/gate_omega/camera_instrument_calibration_v10_walking.json`
- `outputs/gate_omega/training_recipe_walking_v7_lyra_selective.json`
- `outputs/gate_omega/tolerance_matched_handoff_0950_1250_v1/best_step_01200.safetensors`
- `outputs/gate_omega/tolerance_matched_handoff_0950_1250_v1/best_step_01200.routes.json`
- `outputs/gate_omega/tolerance_matched_handoff_0950_1250_v1/best_step_01200.resume.pt`

The destination also needs DL3DV, FrameCrafter/Wan2.2, Lyra's vendored DA3
source, the pinned DA3 checkpoint, and the MoGe checkpoint. Paths no longer
assume `/home/hj/Alican`; set these variables when the layout differs:

```bash
export GAUSSVERSE_REPO=/workspace/gaussverse
export GAUSSVERSE_WORKSPACE=/workspace
export GAUSSVERSE_DATA_ROOT=/datasets
export GAUSSVERSE_FRAMECRAFTER=/workspace/FrameCrafter
export GAUSSVERSE_WAN22_DIR=/models/Wan2.2-TI2V-5B
export GAUSSVERSE_DA3_SOURCE=/workspace/lyra/Lyra-2/lyra_2/_src/inference/depth_anything_3/src
export GAUSSVERSE_DA3_MODEL=/models/DA3NESTED-GIANT-LARGE-1.1
export GAUSSVERSE_MOGE_MODEL=/models/moge-vitl/model.pt
export GAUSSVERSE_PYTHON=/opt/conda/envs/framecrafter/bin/python
```

Before copying, freeze the exact artifact inventory; after copying, run the
same command in verification mode. The manifest uses repository-relative paths,
so the destination directory can differ:

```bash
"$GAUSSVERSE_PYTHON" -m gaussverse.training.gate_omega_one_step_transfer \
  --root "$GAUSSVERSE_REPO" \
  --output "$GAUSSVERSE_REPO/outputs/gate_omega/one_step_transfer_manifest_v1.json"

"$GAUSSVERSE_PYTHON" -m gaussverse.training.gate_omega_one_step_transfer \
  --root "$GAUSSVERSE_REPO" \
  --verify "$GAUSSVERSE_REPO/outputs/gate_omega/one_step_transfer_manifest_v1.json"
```

The 512-scene shallow cache is intentionally rebuilt on the destination. It
encodes five latent slots per scene and does no deep recurrent recovery. The
existing 108-scene cache remains the source of the two deep rows and all
tripwire rows.

## Verification and launch

From the repository, first run the CPU contract suite:

```bash
export PYTHONPATH="$(dirname "$GAUSSVERSE_REPO")"
"$GAUSSVERSE_PYTHON" -m pytest -q \
  "$GAUSSVERSE_REPO/tests/test_gate_omega_handoff.py" \
  "$GAUSSVERSE_REPO/tests/test_gate_omega_sampling.py" \
  "$GAUSSVERSE_REPO/tests/test_gate_omega_selfaug.py" \
  "$GAUSSVERSE_REPO/tests/test_gate_omega_executor.py" \
  "$GAUSSVERSE_REPO/tests/test_gate_omega_trainer.py" \
  "$GAUSSVERSE_REPO/tests/test_gate_omega_checkpoint.py" \
  "$GAUSSVERSE_REPO/tests/test_gate_omega_one_step.py"
```

Then launch the treatment:

```bash
"$GAUSSVERSE_REPO/training/run_gate_omega_one_step_smoke.sh"
```

For the equal-compute clean-history control, use a new output path:

```bash
export GAUSSVERSE_ONE_STEP_ARM=one_step_clean_control
export GAUSSVERSE_ONE_STEP_OUT="$GAUSSVERSE_REPO/outputs/gate_omega/one_step_1200_clean_control_v1"
"$GAUSSVERSE_REPO/training/run_gate_omega_one_step_smoke.sh"
```

Do not run treatment and control concurrently on one GPU. The launcher refuses
to overwrite an existing output. The trainer also refuses to launch if any
treatment source differs from `HEAD`; unrelated thesis/analysis files do not
block it. `start.json` records the commit and hashes of the parent, both cache
manifests, and every source file that controls the treatment.
