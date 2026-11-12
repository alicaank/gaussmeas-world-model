"""Render cache v4 = the input contract of the 11 Nov plan (data preparation). Differences from v3:
  windows   data_contract/index_v4.json (motion-weighted towards the inference range, >= 13 latents of history, true revisits up-weighted; 4423 windows)
  bank      memory_contract.bank_contract_train (true revisits, then first-seen owners; deterministic 20% bank dropout), computed here and stored; the trainer uses the SAME function
            (train_gmp --bank contract) and CachedBuilder refuses a row whose bank differs. Render sources as before: anchor, s-1, s-2, bank[0].
  validity  support-density mask (frontier.support_density, >= .15, closing 1) = the inference mask: meta_density_f16; meta_f16 stays the raw metadata.
  per window also: density_dropped (share of raw-valid pixels the mask removes), psnr_first, sources. 8-view weights at 512x960 (exact geometry) as v3.
Usage: build_render_cache_v4.py [--out DIR] [--limit N]   (GPU; resumable; `_done` at the end)"""
import gm_env  # machine paths: gm_env.py at the repository root (scripts/env.sh puts it on PYTHONPATH)
import argparse, json, sys, time, shutil
from pathlib import Path
import numpy as np, torch
from torch.nn import functional as F

FT = gm_env.CORE; HERE = Path(__file__).resolve().parent
for p in (str(HERE), FT):
    if p not in sys.path: sys.path.insert(0, p)
import frontier  # noqa: E402
from build_mask_cache import pool_meta  # noqa: E402


def drawn(logs, index):
    pre = {sid[:8]: sid for sid in index}; out = []
    for log in logs:
        for line in open(log):
            try: d = json.loads(line)
            except Exception: continue
            for r in d.get('rows', []):
                if r.get('render') and not str(r['stream']).startswith('OP'): out.append((pre[r['stream']], int(r['start'])))
    return list(dict.fromkeys(out))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, default=HERE / 'render_cache_v4'); ap.add_argument('--first', default=''); ap.add_argument('--only_first', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0); ap.add_argument('--renderer', default='v8', choices=('v8', 'v16p')); a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    if torch.cuda.mem_get_info()[0] / 2 ** 30 < 18: raise RuntimeError('GPU not free')
    import resplat_fixed; renderer, size = resplat_fixed.make(a.renderer)
    frontier.install('density', 1, None, alpha_min=.5, size=size, support_min=.15)
    import memory_contract
    from data import corpus864, Stream864
    from render864_rows import RenderBuilder864
    from renderer864 import Renderer864
    from model_io import load_vae
    SRC = HERE / 'data_contract/index_v4.json'; index = json.load(open(SRC)); rows_ = corpus864('train'); by_id = {r['stream_id']: r for r in (rows_[0] if isinstance(rows_, tuple) else rows_)}
    if not (a.out / 'index.json').exists(): shutil.copy(SRC, a.out / 'index.json')
    first = drawn([x for x in a.first.split(',') if x], index); depth = max(len(v) for v in index.values())
    rest = [(sid, int(v[k])) for k in range(depth) for sid, v in index.items() if k < len(v)]
    todo = list(dict.fromkeys(first + ([] if a.only_first else rest)))
    if a.limit: todo = todo[:a.limit]
    vae = load_vae(); vae.cuda(); rb = RenderBuilder864(vae); rb.renderer = renderer; rb.da3.to('cuda'); rb._own_da3 = False
    t0 = time.time(); done = new = 0; stats = a.out / 'build_stats.jsonl'; cur = (None, None)

    def mark(final):
        rows = [json.loads(l) for l in stats.open()] if stats.exists() else []; ps = [r['psnr'] for r in rows if r['psnr'] is not None]
        built = sum(1 for _ in a.out.glob('*/start_*.npz'))
        (a.out / '_done').write_text(json.dumps(dict(windows_built=built, of=sum(len(v) for v in index.values()), first_set=len(first), final=final, seconds=time.time() - t0,
                                                      renderer=a.renderer, psnr_p5_p50_p95=[round(float(x), 2) for x in np.percentile(ps, [5, 50, 95])] if ps else None,
                                                      psnr_below_20=int(sum(p < 20 for p in ps)), meta='meta_f16 raw | meta_density_f16 (support density >= .15)', bank='memory_contract.bank_contract')))

    for n, (sid, s) in enumerate(todo):
        dest = a.out / sid / f'start_{s}.npz'
        if not dest.exists():
            if cur[0] != sid: cur = (sid, Stream864(by_id[sid], 2.))
            st = cur[1]
            mb = 4; boundary = 4 * (s - 1); command = st.c2w[4 * s - 3: 4 * s + 13].numpy()
            bank = [int(b) for b in memory_contract.bank_contract_train(st.c2w[:boundary + 1], st.c2w[4 * s - 3: 4 * s + 13], st.K, s, max_refs=mb, size=(480, 864))] if s > 1 else []
            ids = list(dict.fromkeys([0, boundary, max(0, boundary - 4), *[4 * i for i in bank[:1]]]))[:rb.MAX_SOURCES]     # = RenderBuilder864.render_chunk0
            frames = np.stack([rb.frame(st, i) for i in ids]); c2w = st.c2w[ids].numpy()
            rz, meta, raw = rb.render(frames, c2w, st.K.numpy(), st.metres_per_unit, command, st.gauge_id, boundary)        # returned valid / conf carry mask v2
            L = frontier.LAST; rv, rc = L['raw_valid'].float().cpu(), L['raw_conf'].float().cpu()
            m_raw = pool_meta(rv, rc); m_den = pool_meta(raw['valid'], raw['conf']); dropped = round(float(((rv > .5) & (raw['valid'].float().cpu() < .5)).float().mean() / max(float((rv > .5).float().mean()), 1e-6)), 4)
            tgt = torch.from_numpy(rb.frame(st, 4 * s - 3)).permute(2, 0, 1).float() / 127.5 - 1; m = rv[0] > .5
            psnr = round(float(10 * torch.log10(4 / ((raw['rgb'][0] - tgt)[:, m] ** 2).mean())), 2) if float(m.float().mean()) > .05 else None
            lum = round(float((raw['rgb'][0].mean(0)[m] + 1).mean() / (tgt.mean(0)[m] + 1).mean()), 3) if psnr is not None else None
            dest.parent.mkdir(exist_ok=True); tmp = dest.with_suffix('.tmp.npz')
            np.savez_compressed(tmp, rz_bf16=rz[0].to(torch.bfloat16).view(torch.int16).numpy(), meta_f16=m_raw.numpy().astype(np.float16), meta_density_f16=m_den.numpy().astype(np.float16),
                                bank=np.array(bank, np.int64), start=s, max_bank=mb, sources=np.array(ids), psnr_first=-1. if psnr is None else psnr, density_dropped=dropped)
            tmp.rename(dest); cell = lambda x: round(float((x[0] >= .5).float().mean()), 4)
            rec = dict(stream=sid[:8], start=s, n_src=len(ids), psnr=psnr, lum=lum, px_valid=round(float(rv.mean()), 4), alpha=round(float(rc[rv > .5].mean()), 3) if float(rv.mean()) > 0 else None,
                       cells_raw=cell(m_raw), cells_density=cell(m_den), density_dropped=dropped, bank=bank)
            with stats.open('a') as f: f.write(json.dumps(rec) + '\n')
            new += 1
            if new <= 5 or new % 100 == 0: print(json.dumps(dict(n=n + 1, of=len(todo), new=new, sec=round(time.time() - t0), **rec)), flush=True)
        if first and n + 1 == len(first): mark(False); print('FIRST_SET_DONE', len(first), f'{time.time() - t0:.0f}s', flush=True)
    mark(not a.only_first and not a.limit); print('CACHE_OK', len(todo), 'new', new, f'{time.time() - t0:.0f}s', (a.out / '_done').read_text(), flush=True)


if __name__ == '__main__':
    main()
