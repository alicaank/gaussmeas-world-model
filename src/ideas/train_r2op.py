"""R2 on-policy (user 10-27): the measured-write arm (train_onpolicy.py --op_arm meas: on-policy pool rows, own frames at
MEASURED poses) with every target chunk's own render (data.PER_CHUNK_RENDER, as train_r2.py) on BOTH row kinds: base rows read
the recorded chunk-1 render cache (precompute_renders_next.py, _done_next), pool rows read rz_next_meas (precompute_pool_next.py:
own prefix at measured poses + recorded chunk 0). Same rows / windows / pool draws / seed as run_op/meas (control) and
run_op/R2 (teacher-forced per-chunk render); only the chunk-1 render groups are added. Lineage flag per_chunk_render=True.
Usage: train_r2op.py --op_arm meas <train_m864v.py args>"""
import os, sys
from pathlib import Path
import numpy as np
import data
data.PER_CHUNK_RENDER = True
import train_onpolicy                                   # pops --op_arm/--op_q/--op_seed, patches the sampler with the pool rows
import train_r2  # noqa: F401                           # train.save_trainables -> lineage per_chunk_render
import train_m864v, train
from train_onpolicy import ARM, Q, OP_SEED, ITEMS, POOL

if __name__ == '__main__':
    if ARM != 'meas':
        raise SystemExit('train_r2op.py: per-chunk pool renders exist for --op_arm meas only')
    if not (train_m864v.train_b864.RENDERS / '_done_next').exists():
        raise SystemExit('chunk-1 renders missing: run precompute_renders_next.py first')
    missing = []
    for f in ITEMS:
        with np.load(f) as z:
            if 'rz_next_meas_bf16' not in z.files:
                missing.append(f.name)
    if missing:
        raise SystemExit(f'{len(missing)} pool items without rz_next_meas: run precompute_pool_next.py first')
    out = Path(sys.argv[sys.argv.index('--out') + 1]); out.mkdir(parents=True, exist_ok=True)
    (out / 'onpolicy.flag').write_text(f'arm={ARM} q={Q} op_seed={OP_SEED} pool={len(ITEMS)} items from {POOL}\n'); (out / 'r2.flag').write_text('per_chunk_render=True (base + pool rows)\n')
    train.main(); os._exit(0)
