"""Window table of the training corpus (11 Nov plan, data preparation): for EVERY candidate window (stream, start) of the 891 train streams:
per-call motion of chunk 0 (translation in gauge units, rotation in degrees, from the last context latent to the last target latent), anchor coverage of the
target frustum samples, bank size under the stock rule (data.select_bank) and under the contract rule (memory_contract.bank_contract), number of true revisits,
render PSNR of the first target where render_cache_v3 has the window. -> data_contract/windows.jsonl (one line per window) + windows_summary.json"""
import gm_env  # machine paths: gm_env.py at the repository root (scripts/env.sh puts it on PYTHONPATH)
import json, sys, os
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent.parent; FT = Path(gm_env.CORE)
for p in (str(HERE), str(FT), gm_env.PKG_ROOT):
    if p not in sys.path: sys.path.insert(0, p)
OUT = HERE / 'data_contract'


def work(r):
    import torch; torch.set_num_threads(1)
    from data import select_bank, DATA864
    import source_policy as sp, memory_contract as mc
    sid = r['stream_id']
    with np.load(DATA864 / 'components' / sid / 'vae.npz') as z: P = torch.from_numpy(z['c2w'].astype(np.float32)); K = torch.from_numpy(z['K'].astype(np.float32))
    L = (len(P) - 1) // 4 + 1; rows = []
    for s in range(1, L - 7):                                                # two target chunks: latents s .. s+7
        cmd = P[4 * s - 3: 4 * s + 13]; a, b = P[4 * (s - 1)], P[4 * (s + 3)]
        rot = float(np.degrees(np.arccos(np.clip((np.trace((a[:3, :3].T @ b[:3, :3]).numpy()) - 1) / 2, -1, 1)))); obs = P[:4 * (s - 1) + 1]
        w = sp.frustum_samples(cmd, K, (480, 864)); cov = float(sp.covers(P[0], w, K, (480, 864)).float().mean())
        stock = select_bank(obs, cmd, K, s, max_refs=4, size=(480, 864)) if s > 1 else []; rv = mc.revisits(obs, cmd, s) if s > 5 else []
        con = mc.bank_contract(obs, cmd, K, s, max_refs=4) if s > 1 else []
        f = HERE / 'render_cache_v3' / sid / f'start_{s}.npz'; psnr = None
        if f.exists():
            with np.load(f) as z: psnr = round(float(z['psnr_first']), 2)
        rows.append(dict(stream=sid, scene=r['scene_id'], start=s, L=L, trans=round(float((b[:3, 3] - a[:3, 3]).norm()), 4), rot=round(rot, 2), anchor_cov=round(cov, 3),
                         bank_stock=len(stock), bank_contract=len(con), revisits=len(rv), psnr=psnr))
    return rows


if __name__ == '__main__':
    from multiprocessing import Pool
    recs = [json.loads(l) for l in open(FT / 'data864/streams.jsonl')]; recs = [r for r in recs if r['split'] == 'train' and r['latent_count'] >= 9 and r.get('mpu_valid', True)]
    n = 0
    with Pool(14) as pool, open(OUT / 'windows.jsonl', 'w') as f:
        for rows in pool.imap_unordered(work, recs, chunksize=4):
            for x in rows: f.write(json.dumps(x) + '\n')
            n += 1
            if n % 100 == 0: print('streams', n, flush=True)
    print('WINDOWS_DONE', n, flush=True)
