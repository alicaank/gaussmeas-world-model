"""Stage B without the renderer inside the training process (10-22: four OOMs from module shuffling + fragmentation next to the
resident DiT/masters/grads): precompute the ReSplat 540x960 render of chunk 0 for a fixed set of windows per train stream,
exactly as RenderBuilder864.render_chunk0 would produce it for build_row (same bank selection -> stored and checked at use).
Output: data864/renders/<stream_id>/start_<s>.npz (rz bf16 (48,4,h,w), meta fp16 (2,4,h,w), bank). Windows: start 1 + K random
starts in [2, L-4*n_max] (seeded). Usage: precompute_renders_864.py [--starts 3] [--n_max 2] [--max_bank 4] [--limit 0]"""
import argparse, json, random, time
from pathlib import Path
import numpy as np, torch
from data import corpus864, Stream864, select_bank, DATA864
from model_io import load_vae
from render864_rows import RenderBuilder864


def pack_bf16(t):
    return t.to(torch.bfloat16).view(torch.int16).numpy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--starts', type=int, default=3); ap.add_argument('--n_max', type=int, default=2); ap.add_argument('--max_bank', type=int, default=4)
    ap.add_argument('--seed', type=int, default=20261022); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--scale_mult', type=float, default=2.); a = ap.parse_args()
    out = DATA864 / 'renders'; out.mkdir(exist_ok=True); rows, _ = corpus864('train'); rows = [r for r in rows if r.get('mpu_valid', True)]; rows = rows[:a.limit] if a.limit else rows
    rb = RenderBuilder864(load_vae(device='cpu').eval()); rng = random.Random(a.seed); done = 0; t0 = time.time(); index = {}
    for i, r in enumerate(rows):
        sid = r['stream_id']; d = out / sid; d.mkdir(exist_ok=True)
        try:
            s = Stream864(r, a.scale_mult)
        except Exception as e:
            print('skip', sid[:8], e, flush=True); continue
        hi = s.L - 4 * a.n_max; starts = [1] + (rng.sample(range(2, hi + 1), min(a.starts, max(0, hi - 1))) if hi >= 2 else [])
        H, W = s.latents.shape[-2:]; index[sid] = []
        for start in starts:
            f = d / f'start_{start}.npz'
            if f.exists():
                index[sid].append(start); continue
            command = s.c2w[4 * start - 3: 4 * start + 13]
            bank = select_bank(s.c2w[:4 * (start - 1) + 1], command, s.K, start, size=(16 * H, 16 * W), max_refs=a.max_bank) if start > 1 else []
            rz, meta = rb.render_chunk0(s, start, bank)
            np.savez_compressed(f, rz_bf16=pack_bf16(rz[0]), meta_f16=meta[0].to(torch.float16).numpy(), bank=np.array(bank, dtype=np.int64), start=start, max_bank=a.max_bank)
            index[sid].append(start); done += 1
        if i % 20 == 0:
            print(json.dumps(dict(i=i, streams=len(rows), rendered=done, sec=round(time.time() - t0), peak=round(torch.cuda.max_memory_allocated() / 2 ** 30, 1))), flush=True)
            json.dump(index, open(out / 'index.json', 'w'))
    json.dump(index, open(out / 'index.json', 'w')); (out / '_done').write_text(json.dumps(dict(rendered=done, streams=len(index), args=vars(a), sec=round(time.time() - t0))))
    print('done', done, 'renders for', len(index), 'streams in', round(time.time() - t0), 's', flush=True)


if __name__ == '__main__':
    main()
