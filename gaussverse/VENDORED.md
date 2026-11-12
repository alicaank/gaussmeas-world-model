# Vendored subset of the `gaussverse` package

The part of our base library that this pipeline imports, copied from `alicaank/gaussverse`, branch
`snapshot/research-state-2026-09-11` (commit `6ed16ba`), so that this repository runs on its own. Machine paths were replaced
by `gm_env` roots (`SOURCE_PROVENANCE.json`); nothing else was changed. The package still reads its own configuration from
`GAUSSVERSE_*` environment variables (`_paths.py`; `scripts/env.sh` sets them).

- `diffsynth/` (complete): Wan2.1 / Wan2.2 model code, loaders and state-dict converters, derived from DiffSynth-Studio
  (ModelScope), released under the Apache License 2.0. Its loader resolves model classes from strings, which is why the whole
  subtree is included.
- `camera/`, `data/`, `models/`, `memory/`, `training/`: only the modules reached from the pipeline's imports (rays and
  commands, DL3DV loading, VAE streaming, the ReSplat conditioning wrapper, geometry recovery with DA3 / MoGe).
  `memory/projection.py` is vendored from GEN3C (NVIDIA, Apache-2.0; its SPDX header is kept).

Not included and needed at run time: the Wan2.2-TI2V-5B weights, ReSplat, Depth-Anything-3, MoGe and the Memorizon release
(`docs/PIPELINE.md`, section 1).
