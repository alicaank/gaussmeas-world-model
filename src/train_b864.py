"""Stage B at 864x480 with PRECOMPUTED ReSplat renders (data864/renders, precompute_renders_864.py): train.py is untouched;
render_rows.RenderBuilder is swapped for a cache reader and RowSampler.draw picks, for render rows, one of the precomputed
windows of the stream. No renderer, DA3 or VAE shuffling inside the training process (10-22: four OOMs from that).
Usage: train_b864.py <train.py args> --render_p .75 --corpus 864 ..."""
import json, os
import numpy as np, torch
import render_rows, data
from data import DATA864, build_row
import train

RENDERS = DATA864 / 'renders'
INDEX = {k: [int(s) for s in v] for k, v in json.load(open(RENDERS / 'index.json')).items()} if (RENDERS / 'index.json').exists() else {}   # windows for BOTH arms


class CachedBuilder:
    def __init__(self, vae):
        self.index = {k: [int(s) for s in v] for k, v in json.load(open(RENDERS / 'index.json')).items()}

    def render_chunk0(self, stream, start, bank):
        f = RENDERS / stream.row['stream_id'] / f'start_{start}.npz'
        with np.load(f) as z:
            if [int(b) for b in z['bank']] != [int(b) for b in bank]:
                raise RuntimeError(f'precomputed bank {list(z["bank"])} != row bank {list(bank)} for {stream.row["stream_id"][:8]} start {start}')
            rz = torch.from_numpy(z['rz_bf16'].copy()).view(torch.bfloat16).float()[None]; meta = torch.from_numpy(z['meta_f16'].astype(np.float32))[None]
        return rz, meta


def draw(self):
    """data.RowSampler.draw with the window of render rows taken from the precomputed set (same scene-balanced stream choice)."""
    if not self.pending:
        self.pending = list(self.scenes); self.rng.shuffle(self.pending)
    row = self.rng.choice(self.by_scene[self.pending.pop()]); sid = row['stream_id']
    if sid not in self.cache:
        if len(self.cache) >= self.cache_max:
            self.cache.pop(next(iter(self.cache)))
        self.cache[sid] = self.stream_cls(row)
    stream = self.cache[sid]; n = self.rng.choices(self.n_choices, self.n_probs)[0]
    starts = [s for s in INDEX.get(sid, []) if s + 4 * n <= stream.L]                    # train.py builds no render builder for A (render_p 0),
    start = self.rng.choice(starts) if starts else None                                 #  so the window set comes from the index: both arms draw
    u = self.rng.random(); render = bool(starts) and self.render_builder is not None and u < self.render_p   #  the SAME (stream, start, n) sequence
    if not render:
        self.rng.random()                                                              # build_row consumes one draw only when a builder is passed
    return build_row(stream, self.rng, n_chunks=n, start=start, render_builder=self.render_builder if render else None,
                     render_p=1. if render else 0., max_bank=self.max_bank)


render_rows.RenderBuilder = CachedBuilder; data.RowSampler.draw = draw

if __name__ == '__main__':
    if not (RENDERS / '_done').exists():
        raise SystemExit('precomputed renders missing: run precompute_renders_864.py first')
    train.main(); os._exit(0)
