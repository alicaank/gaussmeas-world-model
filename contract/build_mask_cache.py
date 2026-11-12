"""Mask v2 for every window of the training render cache (10 Nov, user: "egitim orneklerindeki render kayitlarinda maskeyi de kullanmaliyiz").
The SAME rule and the SAME code as inference (frontier.frontier_render_chunk, installed into renderer864): a render pixel is valid iff
accumulated alpha >= --alpha_min (default .7: the alpha .5-.7 fringe is 18-42 luminance levels too dark in the identity render) AND the
RENDERED depth (source-derived, never the recorded target) back-projects inside some source frustum (reproj, tol 1e9 = no depth test).
Per window the cached render is reproduced (same sources as RenderBuilder864.render_chunk0, seed 0, no VAE: colours are unchanged, so the
cached rz latents stay valid) and only the (valid, conf) metadata is re-pooled exactly as render864_rows.Render864.render does.
Output mask_cache_v2/<stream>/start_<s>.npz: meta_f16 (2,4,30,54) = mask v2, meta_raw_f16 = the re-render without any mask (the trainer
applies the RATIO v2/raw to the cached metadata, so re-render noise (ReSplat refine is not bit-deterministic: max cell diff ~.15) cancels), meta_frustum_f16 = frustum only (alpha >= .5), plus the
max |pooled raw meta - cached meta| (determinism check of the re-render) and drop statistics. Resumable; DA3 + ReSplat only.
Usage: build_mask_cache.py [--out DIR] [--alpha_min .7] [--limit N]"""
import gm_env  # machine paths: gm_env.py at the repository root (scripts/env.sh puts it on PYTHONPATH)
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from torch.nn import functional as F

FT = gm_env.CORE; HERE = Path(__file__).resolve().parent
for p in (str(HERE), FT):
    if p not in sys.path: sys.path.insert(0, p)
import frontier  # noqa: E402


def pool_meta(valid, conf, hw=(30, 54)):
    """render864_rows.Render864.render's metadata pooling: mean over the 4 RGB frames of each latent, area-pooled. -> (2,4,h,w)"""
    return torch.stack([F.interpolate(x.float().reshape(4, 4, 1, 480, 864).mean(1), size=hw, mode='area')[:, 0] for x in (valid, conf)])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, default=HERE / 'mask_cache_v2'); ap.add_argument('--alpha_min', type=float, default=.7)
    ap.add_argument('--limit', type=int, default=0); a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    if torch.cuda.mem_get_info()[0] / 2 ** 30 < 14: raise RuntimeError('GPU not free')
    frontier.install('reproj', 1, None, tol=1e9, margin=2, alpha_min=a.alpha_min)
    from data import corpus864, Stream864
    from render864_rows import RenderBuilder864
    RENDERS = Path(FT) / 'data864/renders'; index = json.load(open(RENDERS / 'index.json')); by_id = {r['stream_id']: r for r in corpus864('train')[0]}
    rb = RenderBuilder864(None); rb.da3.to('cuda'); rb._own_da3 = False                              # DA3 stays on the GPU (no per-window transfers)
    todo = [(sid, int(s)) for sid, starts in index.items() for s in starts]
    if a.limit: todo = todo[:a.limit]
    t0 = time.time(); done = 0; new = 0; stats_path = a.out / 'build_stats.jsonl'; cur = (None, None)
    for sid, s in todo:
        dest = a.out / sid / f'start_{s}.npz'
        if dest.exists(): done += 1; continue
        if cur[0] != sid: cur = (sid, Stream864(by_id[sid], 2.))
        st = cur[1]
        with np.load(RENDERS / sid / f'start_{s}.npz') as z: bank = [int(b) for b in z['bank']]; cached = torch.from_numpy(z['meta_f16'].astype(np.float32))
        boundary = 4 * (s - 1); command = st.c2w[4 * s - 3: 4 * s + 13].numpy()
        ids = list(dict.fromkeys([0, boundary, max(0, boundary - 4), *[4 * i for i in bank[:1]]]))[:rb.MAX_SOURCES]   # = RenderBuilder864.render_chunk0
        frames = np.stack([rb.frame(st, i) for i in ids]); c2w = st.c2w[ids].numpy()
        _, _, raw = rb.render(frames, c2w, st.K.numpy(), st.metres_per_unit, command, st.gauge_id, boundary, encode=False)
        L = frontier.LAST; rv, rc, obs = L['raw_valid'].float().cpu(), L['raw_conf'].float().cpu(), L['observed'].float().cpu()
        m_raw = pool_meta(rv, rc); m_v2 = pool_meta(raw['valid'], raw['conf']); m_fr = pool_meta(rv * obs, rc * obs)
        diff = float((m_raw - cached).abs().max()); cell = lambda m: float((m[0] >= .5).float().mean())
        mad = float((m_raw - cached).abs().mean()); flips = float(((m_raw[0] >= .5) != (cached[0] >= .5)).float().mean())   # re-render vs cache: mean diff, binary flips
        dest.parent.mkdir(exist_ok=True)
        np.savez_compressed(dest, meta_f16=m_v2.numpy().astype(np.float16), meta_frustum_f16=m_fr.numpy().astype(np.float16), meta_raw_f16=m_raw.numpy().astype(np.float16), bank=np.array(bank, np.int64), start=s,
                            sources=np.array(ids), alpha_min=a.alpha_min, rerender_max_diff=diff)
        rec = dict(stream=sid[:8], start=s, n_src=len(ids), rerender_max_diff=round(diff, 4), rerender_mad=round(mad, 4), rerender_flips=round(flips, 4), cells_cached=round(cell(cached), 4), cells_frustum=round(cell(m_fr), 4), cells_v2=round(cell(m_v2), 4),
                   px_valid=round(float(rv.mean()), 4), px_alpha_drop=round(float((rv * (rc < a.alpha_min)).mean()), 4), px_frustum_drop=round(float((rv * (1 - obs)).mean()), 4), px_v2=round(float(raw['valid'].mean()), 4))
        with stats_path.open('a') as f: f.write(json.dumps(rec) + '\n')
        done += 1; new += 1
        if new <= 5 or new % 100 == 0: print(json.dumps(dict(done=done, of=len(todo), sec=round(time.time() - t0), **rec)), flush=True)
    rows = [json.loads(l) for l in stats_path.open()] if stats_path.exists() else []
    summ = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in ('cells_cached', 'cells_frustum', 'cells_v2', 'px_valid', 'px_alpha_drop', 'px_frustum_drop', 'px_v2')} if rows else {}
    summ.update({k + '_p50_p99': [round(float(v), 4) for v in np.percentile([r[k] for r in rows], [50, 99])] for k in ('rerender_mad', 'rerender_flips')} if rows else {})
    summ['rerender_diff_p50_p99_max'] = [round(float(v), 4) for v in np.percentile([r['rerender_max_diff'] for r in rows], [50, 99, 100])] if rows else None
    if not a.limit: (a.out / '_done').write_text(json.dumps(dict(windows=len(todo), seconds=time.time() - t0, alpha_min=a.alpha_min, rule='alpha>=alpha_min & reproj tol 1e9 (rendered depth)', summary=summ)))
    print('done', len(todo), f'{time.time() - t0:.0f}s', json.dumps(summ), flush=True)


if __name__ == '__main__':
    main()
