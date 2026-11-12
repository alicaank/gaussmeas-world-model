# gmlib: the base library of this repository

Model code and geometry utilities the pipeline builds on. `core/`, `contract/` and `foundation/` import it as `gmlib.*`.

- `diffsynth/`: Wan2.1 / Wan2.2 model code, loaders and state-dict converters, derived from DiffSynth-Studio (ModelScope),
  which is released under the Apache License 2.0. Its loader resolves model classes from strings, so the subtree is kept
  whole.
- `camera/`: rays and camera commands. `data/`: DL3DV loading and trajectories. `models/`: VAE streaming, the Wan2.2
  loaders, point rendering. `memory/`: the ReSplat conditioning wrapper and projection utilities (`memory/projection.py` is
  vendored from GEN3C, NVIDIA, Apache-2.0; its SPDX header is kept). `training/`: geometry recovery with Depth-Anything-3
  and MoGe, model fingerprints.
- `_paths.py`: locations of external models, read from `GM_*` environment variables (`scripts/env.sh`).

Not included and needed at run time: the Wan2.2-TI2V-5B weights, ReSplat, Depth-Anything-3, MoGe and the Memorizon release
(`docs/PIPELINE.md`, section 1).
