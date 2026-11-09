"""Observed masks for every window of the training render cache (fresh_teacher data864/renders/index.json), so training rows carry
the same frontier rule as inference. Per window: the training render's sources (RenderBuilder864.render_chunk0: anchor, last two
prefix latent ends, best bank view) with DA3 gauge depths at 540x960, the 16 RECORDED target frames with DA3 depths (the true
geometry of the chunk), frontier.observed_from_reprojection(target depth -> sources) pooled to the metadata grid (4,30,54).
Inference uses the rendered depth instead of the recorded one (same test). Output: frontier_cache/<stream>/start_<s>.npz
(obs_f16, frontier fraction vs the cached alpha-valid, sources). DA3 only, no ReSplat/VAE; resumable.
Usage: build_frontier_cache.py [--out DIR] [--limit N] [--tol .15]"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

FT = '/home/hj/Alican/analysis/plan_base_restart_20261008/fresh_teacher_20261021'
HERE = Path(__file__).resolve().parent
for p in (str(HERE), FT):
    if p not in sys.path:
        sys.path.insert(0, p)
import frontier  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, default=HERE / 'frontier_cache'); ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--tol', type=float, default=0.15); a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    if torch.cuda.mem_get_info()[0] / 2 ** 30 < 10: raise RuntimeError('GPU not free')
    from data import corpus864, Stream864
    from render864_rows import rec_frames
    from renderer864 import upsize, da3_gauge_depths
    from realized_pose import load_da3
    RENDERS = Path(FT) / 'data864/renders'; index = json.load(open(RENDERS / 'index.json')); by_id = {r['stream_id']: r for r in corpus864('train')[0]}
    da3 = load_da3('cuda'); todo = [(sid, int(s)) for sid, starts in index.items() for s in starts]
    if a.limit: todo = todo[:a.limit]
    t0 = time.time(); done = 0; fr = []; stats_path = a.out / 'build_stats.jsonl'
    for sid, s in todo:
        dest = a.out / sid / f'start_{s}.npz'
        if dest.exists(): done += 1; continue
        row = by_id.get(sid)
        if row is None: raise RuntimeError(f'stream {sid} not in the train corpus')
        with np.load(RENDERS / sid / f'start_{s}.npz') as z: bank = [int(b) for b in z['bank']]; meta = z['meta_f16'].astype(np.float32)
        st = Stream864(row, 2.); boundary = 4 * (s - 1); tgt_ids = list(range(4 * s - 3, 4 * s + 13)); command = st.c2w[tgt_ids].numpy().astype(np.float64)
        ids = list(dict.fromkeys([0, boundary, max(0, boundary - 4), *[4 * i for i in bank[:1]]]))[:4]
        frames = rec_frames(row, ids); c2w = st.c2w[ids].numpy().astype(np.float64)
        K = st.K.numpy().astype(np.float64).copy(); K[0] *= 960 / 864; K[1] *= 540 / 480
        src_depths, _ = da3_gauge_depths(da3, upsize(frames, (540, 960)), K, c2w, float(st.metres_per_unit))
        # one recorded target frame per latent (the latent's last RGB frame) instead of all 16: 4x fewer DA3 views (3553 windows x 16 was
        # ~3.5 h); the mask is pooled per latent anyway, so each latent's mask is repeated over its 4 frame slots
        sel = tgt_ids[3::4]
        tgt_depths, _ = da3_gauge_depths(da3, upsize(rec_frames(row, sel), (540, 960)), K, command[3::4], float(st.metres_per_unit))
        obs4 = frontier.observed_from_reprojection(torch.stack(tgt_depths), K, command[3::4], src_depths, c2w, tol=a.tol)   # (4,480,864)
        obs = obs4.repeat_interleave(4, dim=0)                                                                              # (16,480,864)
        cells = frontier.pooled_cells(obs).numpy()                                                                      # (4,30,54)
        valid = meta[0]; frontier_frac = float(((valid > .5) & (cells < .5)).mean()); dropped = float((cells < .5).mean())
        dest.parent.mkdir(exist_ok=True)
        np.savez_compressed(dest, obs_f16=cells.astype(np.float16), frontier=frontier_frac, dropped=dropped, sources=np.array(ids), start=s, tol=a.tol, mode='reproj_true_depth')
        rec = dict(stream=sid[:8], start=s, frontier=round(frontier_frac, 4), dropped=round(dropped, 4), valid_mean=round(float(valid.mean()), 4),
                   conf_frontier=round(float(meta[1][(valid > .5) & (cells < .5)].mean()), 4) if ((valid > .5) & (cells < .5)).any() else None,
                   conf_observed=round(float(meta[1][(valid > .5) & (cells >= .5)].mean()), 4) if ((valid > .5) & (cells >= .5)).any() else None)
        with stats_path.open('a') as f: f.write(json.dumps(rec) + '\n')
        fr.append(frontier_frac); done += 1
        if done % 100 == 0: print(f'{done}/{len(todo)} {time.time() - t0:.0f}s mean frontier {np.mean(fr):.4f}', flush=True)
    (a.out / '_done').write_text(json.dumps(dict(windows=len(todo), seconds=time.time() - t0, tol=a.tol, mode='reproj_true_depth')))
    print('done', len(todo), f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
