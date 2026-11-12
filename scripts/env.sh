# Source this file: `source scripts/env.sh`. Edit the block below for your machine (or export the variables before sourcing).
# See docs/PIPELINE.md, "Environment", for what each external component is.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${GM_WORKSPACE:=$(dirname "$REPO")}"            # default parent of the external checkouts and the data below
: "${GM_DATA_ROOT:=$GM_WORKSPACE/data}"            # contains DL3DV-960P-864x480/
: "${GM_RESPLAT:=$GM_WORKSPACE/resplat}"           # ReSplat source tree
: "${GM_MEMORIZON:=$GM_WORKSPACE/baselines/memorizon}"   # released Memorizon code
: "${GM_DA3_SRC:=$GM_WORKSPACE/depth_anything_3/src}"   # Depth-Anything-3 source tree
: "${GM_WAN22_DIR:=$GM_WORKSPACE/models/Wan2.2-TI2V-5B}"   # Wan2.2-TI2V-5B: DiT, Wan2.2_VAE.pth, umT5-xxl encoder + tokenizer
: "${GM_PYTHON_DEPS:=}"                            # optional: extra site dir with ReSplat's python dependencies
: "${CUDA_HOME:=${CONDA_PREFIX:-/usr/local/cuda}}" # gsplat (ReSplat rasterizer) loads its extension only when it finds nvcc
export GM_WORKSPACE GM_DATA_ROOT GM_RESPLAT GM_MEMORIZON GM_DA3_SRC CUDA_HOME GM_WAN22_DIR
export GM_LIB_DIR="$REPO/gmlib"                    # gmlib/_paths.py reads this and the GM_* variables above

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTORCH_ALLOC_CONF=expandable_segments:True   # training peaks near 25 GiB of a 32 GiB card
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$REPO:$REPO/core:$REPO/contract:$REPO/contract/sota_eval:$REPO/foundation:$GM_MEMORIZON${GM_PYTHON_DEPS:+:$GM_PYTHON_DEPS}${PYTHONPATH:+:$PYTHONPATH}"
export GM_REPO="$REPO"
