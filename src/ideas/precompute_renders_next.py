"""Idea 4 (rev 10-26): for every precomputed window (data864/renders/index.json, start s, n_chunks 2) the render of CHUNK 1's
command at start s+4 from the recorded prefix (latents 0..s+3), exactly RenderBuilder864.render_chunk0's contract with the
bank selected at s+4 (max_bank 4, as the arms) -> data864/renders/<sid>/start_<s+4>.npz (same format; index.json untouched,
so the sampler's windows stay those of the control). _done_next is written only when every window's file is present.
Usage: precompute_renders_next.py [--limit 0]"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch
from data import corpus864, Stream864, select_bank, DATA864
from model_io import load_vae
from render864_rows import RenderBuilder864

RENDERS = DATA864 / 'renders'


def pack_bf16(t):
    return t.to(torch.bfloat16).view(torch.int16).numpy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--max_bank', type=int, default=4); ap.add_argument('--scale_mult', type=float, default=2.); a = ap.parse_args()
    index = json.load(open(RENDERS / 'index.json')); rows = {r['stream_id']: r for r in corpus864('train')[0]}
    rb = RenderBuilder864(load_vae(device='cpu').eval()); done = 0; t0 = time.time(); sids = [s for s in index if s in rows]; sids = sids[:a.limit] if a.limit else sids
    for i, sid in enumerate(sids):
        st = None
        for s0 in [int(x) for x in index[sid]]:
            start = s0 + 4; f = RENDERS / sid / f'start_{start}.npz'
            if f.exists():
                continue
            if st is None:
                st = Stream864(rows[sid], a.scale_mult); H, W = st.latents.shape[-2:]
            if start + 4 > st.L:
                continue
            command = st.c2w[4 * start - 3: 4 * start + 13]
            bank = select_bank(st.c2w[:4 * (start - 1) + 1], command, st.K, start, size=(16 * H, 16 * W), max_refs=a.max_bank)
            rz, meta = rb.render_chunk0(st, start, bank)
            np.savez_compressed(f, rz_bf16=pack_bf16(rz[0]), meta_f16=meta[0].to(torch.float16).numpy(), bank=np.array(bank, dtype=np.int64), start=start, max_bank=a.max_bank, next_of=s0)
            done += 1
        if i % 50 == 0:
            print(json.dumps(dict(streams=i, done=done, sec=round(time.time() - t0), per=round((time.time() - t0) / max(done, 1), 2))), flush=True)
    want = [(sid, int(x) + 4) for sid in index if sid in rows for x in index[sid]]
    have = sum((RENDERS / sid / f'start_{s}.npz').exists() for sid, s in want)
    if not a.limit and have == len(want):
        (RENDERS / '_done_next').write_text(f'{have}/{len(want)} chunk-1 renders, {done} new, {time.time() - t0:.0f} s\n')
    print('DONE', done, 'new;', have, '/', len(want), 'present;', round(time.time() - t0), 's', flush=True)


if __name__ == '__main__':
    main()
