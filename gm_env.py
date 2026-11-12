"""Machine configuration: every location the code needs outside its own directory.

The code directories keep their data next to the code (all git-ignored):
  core/data864/            training corpus (core/prep_864.py)            core/memorizon50/   converted base weights
  core/run_*/              stage-A style runs                            contract/runs/      contract-line runs
  contract/render_cache_v4/ render cache                                 contract/eval_gm/   evaluation outputs
  foundation/dataset_v3/   the 448 corpus the 864 corpus is derived from
Put the data there or symlink it. Everything else is one of the roots below; override any of them with the environment
variable of the same name prefixed by GM_ (scripts/env.sh sets the common ones), e.g. GM_DATA_ROOT=/data/dl3dv.
"""
import os
from pathlib import Path

REPO = str(Path(__file__).resolve().parent)
HOME = os.path.expanduser('~')


def _root(name, default):
    return os.environ.get('GM_' + name, default)


# code directories of this repository
CORE = _root('CORE', REPO + '/core')                    # model, data, base trainer, closed loop
CONTRACT = _root('CONTRACT', REPO + '/contract')        # contract line: window index, render cache, stage trainers, rollout, evaluation
FOUNDATION = _root('FOUNDATION', REPO + '/foundation')  # shared in-house modules (dataset contract, renderer contract, realized pose, scorers)

PKG_ROOT = REPO                                         # the directory that contains the `gmlib` package

# external code (not in this repository; see docs/PIPELINE.md, "Environment")
WORKSPACE = _root('WORKSPACE', str(Path(REPO).parent))  # where the external checkouts and the data live by default
RESPLAT = _root('RESPLAT', WORKSPACE + '/resplat')      # ReSplat source tree (renderer of the Gaussian memory)
DA3_SRC = _root('DA3_SRC', WORKSPACE + '/depth_anything_3/src')   # Depth-Anything-3 source tree
BASELINES = _root('BASELINES', WORKSPACE + '/baselines')
MEMORIZON = _root('MEMORIZON', BASELINES + '/memorizon')   # released Memorizon code (base model layout + its retrieval rule)
FFMPEG = _root('FFMPEG', 'ffmpeg')

# data
DATA_ROOT = _root('DATA_ROOT', WORKSPACE + '/data')                         # contains DL3DV-960P-864x480/
CAMERA_QUALITY = _root('CAMERA_QUALITY', FOUNDATION + '/camera_quality')    # manifest.json: streams with broken camera labels
EVAL_BATCH = _root('EVAL_BATCH', CONTRACT + '/eval_inputs/batch_continuous1009')    # evaluation plans: index.json + trajectories/<scene>.npz
EVAL_INPUTS = _root('EVAL_INPUTS', CONTRACT + '/eval_inputs/cohort')                # inputs/<scene>.npz (anchor latent, K, plan, text) of the evaluation scenes
