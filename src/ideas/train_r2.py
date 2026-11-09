"""Idea 4 (rev 10-26, review B1): every target chunk gets ITS OWN render group. Exactly the control's training (train_m864v recipe
from run_v864/mem, seed 20261025, n_chunks 2, accum 8, render p .75, the same windows -> the same targets, context, u and
noise draws) except that on a render row chunk 1 also receives the ReSplat render of its own command from the recorded prefix
(data.PER_CHUNK_RENDER; renders at start+4 precomputed by precompute_renders_next.py into the same cache; chunk_index -(2+j)
so chunk 1's render is invisible to chunk 0). Compared with run_op/C it isolates "own render for every target step".
Usage: train_r2.py <train_m864v.py args>"""
import os, sys
from pathlib import Path
import train_m864v  # noqa: F401
import train_b864, train, model_io, data

data.PER_CHUNK_RENDER = True


def save_trainables_r2(model, path, meta):
    meta = dict(meta); meta['lineage'] = dict(meta.get('lineage') or {}, per_chunk_render=True)
    return model_io.save_trainables(model, path, meta)


train.save_trainables = save_trainables_r2

if __name__ == '__main__':
    if not (train_b864.RENDERS / '_done_next').exists():
        raise SystemExit('chunk-1 renders missing: run precompute_renders_next.py first')
    out = Path(sys.argv[sys.argv.index('--out') + 1]); out.mkdir(parents=True, exist_ok=True); (out / 'r2.flag').write_text('per_chunk_render=True\n')
    train.main(); os._exit(0)
