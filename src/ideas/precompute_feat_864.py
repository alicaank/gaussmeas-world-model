"""Gaussian-feature memory pilot, training cache: for every precomputed render window (data864/renders/<sid>/start_<s>.npz,
the windows BOTH stage-B arms draw from) compute the three feature channels of renderer864.render_features from the SAME
sources RenderBuilder864.render_chunk0 used (anchor + last two prefix ends + best bank view; DA3 metric depth / mpu at
540x960) at the chunk-0 command, pooled to the latent grid like valid/conf -> data864/renders_feat/<sid>/start_<s>.npz
(feat_f16 (3,4,h,w): depth, conflict, support). The RGB render and (valid, conf) are reused from the existing cache, so the
feature arm sees exactly the control's renders plus the three channels. ~1 s per window.
Usage: precompute_feat_864.py [--limit 0]"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch
from data import corpus864, Stream864, DATA864
from render864_rows import rec_frames, pool_features
from renderer864 import da3_gauge_depths, upsize, render_features
from realized_pose import load_da3

RENDERS = DATA864 / 'renders'; OUT = DATA864 / 'renders_feat'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--scale_mult', type=float, default=2.); a = ap.parse_args()
    index = json.load(open(RENDERS / 'index.json')); rows = {r['stream_id']: r for r in corpus864('train')[0]}; OUT.mkdir(exist_ok=True)
    da3 = load_da3('cuda'); done = 0; t0 = time.time(); sids = list(index)[:a.limit] if a.limit else list(index); n_all = sum(len(index[s]) for s in sids)
    for sid in sids:
        if sid not in rows:
            continue
        d = OUT / sid; d.mkdir(exist_ok=True); st = None
        for start in [int(x) for x in index[sid]]:
            f = d / f'start_{start}.npz'
            if f.exists():
                continue
            if st is None:
                st = Stream864(rows[sid], a.scale_mult); H, W = st.latents.shape[-2:]
            with np.load(RENDERS / sid / f'start_{start}.npz') as z:
                bank = [int(b) for b in z['bank']]
            boundary = 4 * (start - 1); command = st.c2w[4 * start - 3: 4 * start + 13].numpy().astype(np.float64)
            ids = list(dict.fromkeys([0, boundary, max(0, boundary - 4), *[4 * i for i in bank[:1]]]))[:4]            # RenderBuilder864.render_chunk0's sources
            frames = upsize(rec_frames(st.row, ids)); c2w = st.c2w[ids].numpy().astype(np.float64); K = st.K.numpy().astype(np.float64).copy(); K[0] *= 960 / 864; K[1] *= 540 / 480
            depths, _ = da3_gauge_depths(da3, frames, K, c2w, st.metres_per_unit)
            feat = render_features([x.cuda() for x in depths], K, c2w, command)                                        # (16,3,120,216)
            pooled = pool_features(feat.cpu(), (H, W))[0]                                                              # (3,4,h,w)
            np.savez_compressed(f, feat_f16=pooled.to(torch.float16).numpy(), bank=np.array(bank, np.int64), start=start, sources=np.array(ids, np.int64),
                                cov=float((feat[:, 2] > 0).float().mean()), conflict_mean=float(feat[:, 1].mean()))
            done += 1
            if done % 100 == 0:
                print(json.dumps(dict(done=done, of=n_all, sec=round(time.time() - t0), per=round((time.time() - t0) / done, 2))), flush=True)
    have = sum((OUT / sid / f'start_{int(st)}.npz').exists() for sid in index if sid in rows for st in index[sid]); want = sum(len(index[sid]) for sid in index if sid in rows)
    if not a.limit and have == want:
        (OUT / '_done').write_text(f'{have}/{want} windows, {done} new, {time.time() - t0:.0f} s\n')
    print('DONE', done, 'new;', have, '/', want, 'windows present;', round(time.time() - t0), 's', flush=True)


if __name__ == '__main__':
    main()
