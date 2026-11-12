"""Row-rejection report of a run (train.log of train_gmp.py --reject F): rebuilds the running threshold (F x median of the last 200 accepted row norms, as
row_reject.py) and marks every rejection CLEAR (>= 3 x threshold) or NEAR (< 3 x threshold: a row the rule may be cutting as an ordinary hard row), plus the
accepted rows in the half-threshold band and each row's distance from the anchor ('far', model units). Many NEAR rows = the rule is cutting ordinary rows.
Usage: reject_report.py runs/<run> [factor]"""
import gm_env  # machine paths: gm_env.py at the repository root (scripts/env.sh puts it on PYTHONPATH)
import json, sys
from pathlib import Path
import numpy as np
D = Path((gm_env.CORE + '/data864')); META = None; CAM = {}


def far(s, start, n=2):
    global META
    if META is None: META = {r['stream_id'][:8]: r for r in map(json.loads, (D / 'streams.jsonl').open())}
    if s not in CAM:
        r = META[s]
        with np.load(D / 'components' / r['stream_id'] / 'vae.npz') as z: CAM[s] = z['c2w'][:, :3, 3].astype(float) * r['metres_per_anchor_unit'] * 2 / 4
    t = CAM[s]; i = [4 * p for p in range(max(0, start - 4), start + 4 * n) if 4 * p < len(t)]; return float(np.linalg.norm(t[i], axis=1).max())


if __name__ == '__main__':
    run = Path(sys.argv[1]); F = float(sys.argv[2]) if len(sys.argv) > 2 else 50.
    L = [json.loads(l) for l in (run / 'train.log').open() if l.strip().startswith('{')]; hist = []; out = []; band = []
    for d in L:
        for r in d['rows']:
            if 'grow' not in r: continue
            med = float(np.median(hist[-200:])) if len(hist) >= 32 else max(.1, float(np.median(hist)) if hist else 0.); th = F * med; g = r['grow']
            if r.get('rejected'): out.append(dict(update=d['update'], stream=r['stream'], start=r['start'], u=r['u'], fm=r['fm'], grow=g, threshold=round(th, 3), ratio=round(g / th, 2),
                                                  kind='CLEAR' if g >= 3 * th else 'NEAR', far=round(far(r['stream'], r['start'], r['n']), 1)))
            else:
                hist.append(g)
                if g > .5 * th: band.append(dict(update=d['update'], stream=r['stream'], start=r['start'], grow=g, threshold=round(th, 3), far=round(far(r['stream'], r['start'], r['n']), 1)))
    n = sum('grow' in r for d in L for r in d['rows']); gr = np.array([d['grad'] for d in L])
    print(f'{run.name}: {len(L)} updates, {n} rows, rejected {len(out)} (CLEAR {sum(o["kind"] == "CLEAR" for o in out)}, NEAR {sum(o["kind"] == "NEAR" for o in out)}), accepted in the half-threshold band {len(band)}; '
          f'update norm median {np.median(gr):.3f} max {gr.max():.3g}; row norm median {np.median(hist):.4f}')
    for o in out: print(f'  {o["kind"]:5s} update {o["update"]:3d} {o["stream"]} start {o["start"]:3d} norm {o["grow"]:.4g} = {o["ratio"]} x threshold {o["threshold"]} far {o["far"]} u {o["u"]} fm {o["fm"]}')
    for o in band: print(f'  band  update {o["update"]:3d} {o["stream"]} start {o["start"]:3d} norm {o["grow"]:.4g} (threshold {o["threshold"]}) far {o["far"]}')
    (run / 'reject_report.json').write_text(json.dumps(dict(updates=len(L), rows=n, rejected=out, band=band), indent=1))
