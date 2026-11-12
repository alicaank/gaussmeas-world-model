"""Windows whose cameras are far from the anchor (12 Nov). The PRoPE branch multiplies q by P^T and k, v by P^-1, and the anchor slot is in every row, so
the raw anchor <-> target translation enters the attention logits: rows with max |t| >= 20 model units produced every gradient spike of the trainer
(pre-clip norm 1e2-1e6; AUC .999 on the stage-B rows). far = max |t| over the recent + target slots of the window, in model units
(metres_per_anchor_unit * scale_mult / 4, scale_mult 2). Stage B keeps these rows out with row rejection (train_gmp.py --reject 50); stage C excludes
the list written here. Usage: build_far_windows.py [--threshold 10] -> data_contract/far_windows_v4.json   (CPU)"""
import argparse, json
from pathlib import Path
import numpy as np
import gm_env

HERE = Path(__file__).resolve().parent; D = Path(gm_env.CORE) / 'data864'
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--threshold', type=float, default=10.); ap.add_argument('--index', default=str(HERE / 'index_v4.json'))
    ap.add_argument('--exclude', default=','.join(str(HERE / f'{n}.json') for n in ('bad_streams_v3', 'fast_streams_v4', 'flag_streams_v4'))); a = ap.parse_args()
    meta = {r['stream_id']: r for r in map(json.loads, (D / 'streams.jsonl').open())}; index = json.load(open(a.index))
    bad = {s[:8] for f in a.exclude.split(',') if f for s in json.load(open(f))['streams']}; out = []; n = 0
    for s, starts in index.items():
        if s[:8] in bad or s not in meta: continue
        with np.load(D / 'components' / s / 'vae.npz') as z: t = z['c2w'][:, :3, 3].astype(float) * meta[s]['metres_per_anchor_unit'] * 2 / 4
        for st in starts:
            n += 1; i = [4 * p for p in range(max(0, st - 4), st + 8) if 4 * p < len(t)]; far = float(np.linalg.norm(t[i], axis=1).max())
            if far > a.threshold: out.append(dict(stream=s, start=st, far=round(far, 2)))
    out.sort(key=lambda x: -x['far'])
    (HERE / 'far_windows_v4.json').write_text(json.dumps(dict(note='kept windows whose cameras are farther than the threshold (model units) from the anchor; PRoPE logits scale with this distance -> gradient spikes. Excluded from stage C.',
                                                         threshold=a.threshold, n_windows=n, windows=out), indent=0))
    print(f'{len(out)} of {n} windows above {a.threshold} ({len({o["stream"] for o in out})} streams)')
