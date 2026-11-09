# Persistent source Gaussians

`memory/persistent_gaussians.py` provides an opt-in adapter for the causal
ReSplat scaffold. It preserves the final Gaussian block assigned to an unchanged
source image across reconstruction calls. Existing renderer defaults are unchanged.

```python
from gaussverse.memory.persistent_gaussians import PersistentGaussianScaffold

# Construct a fresh adapter for each independent trajectory. model_id should
# identify the frozen ReSplat checkpoint and reconstruction configuration.
scaffold = PersistentGaussianScaffold(
    existing_resplat_scaffold,
    scene_id=trajectory_id,
    model_id=reconstruction_identity,
    max_sources=128,
)

# Pass scaffold as the existing zipsplat_scaffold argument of the rollout pool
# or build_omega_condition. The historical argument name also accepts ReSplat.
```

The normal source selection, reconstruction and four refinement iterations run
unchanged. Before final rendering, unchanged source blocks are replaced with
their cached final centers, covariance, opacity, scales, rotations and spherical
harmonics. New source images contribute newly reconstructed blocks. Source IDs
and their source-grid ordering identify primitives; different images are not
matched by array position.

Each entry fingerprints RGB, camera-to-world pose, depth, confidence, intrinsics
and image size. Changed inputs invalidate that source's block. A scene, gauge or
model namespace change clears the cache. Sources removed from the active view
set remain cached for later reuse. GPU offload leaves the CPU cache intact.
Render failures restore the original encoder method and do not commit staged
source entries. Exceeding the explicit source budget raises an error; it does
not silently discard persistent geometry.

The adapter assumes a frozen model, identified RGB-D views, one Gaussian batch,
source-major primitive ordering, and positive refinement count. Its shared
renderer is used serially. It deliberately rejects unsupported per-cloud
metadata rather than combining incompatible coordinate transforms.

This enforces parameter consistency for unchanged source blocks. It does not
ensure that independently added blocks represent mutually consistent surfaces,
and it can retain early reconstruction errors. New views can still change
occlusion and compositing. It also continues to compute fresh candidate blocks
for cached sources, so it should not be described as a reconstruction speedup.

Validation and paired video results are recorded in
`analysis/persistent_gaussians_20260918` at the workspace root. Lifecycle tests:

```sh
python -m pytest gaussverse/tests/test_persistent_gaussians.py -q
```
