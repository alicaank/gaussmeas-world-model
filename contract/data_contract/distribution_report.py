"""Sampler vs inference distributions (plan rule: test the sampler BEFORE training). Training side: the stream-balanced draw of index_v4.json (windows.jsonl stats).
Inference side: the test plans at strides 1 / 2 / 4 / 8, forward and out-and-back, with the SAME bank function (memory_contract.bank_contract).
Per-call translation / rotation, bank fill, true-revisit share. Mask ratio is added by the cache build (density_dropped). -> data_contract/distribution_report.json"""
import gm_env  # machine paths: gm_env.py at the repository root (scripts/env.sh puts it on PYTHONPATH)
import json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; FT = gm_env.CORE
for p in (str(ROOT), FT, gm_env.PKG_ROOT):
    if p not in sys.path: sys.path.insert(0, p)
import memory_contract as mc
W = {(w['stream'], w['start']): w for w in map(json.loads, open(HERE / 'windows.jsonl'))}; ix = json.load(open(HERE / 'index_v4.json'))
per = [(W[(sid, s)], 1. / len(st)) for sid, st in ix.items() for s in st if s > 1]; wt = np.array([p for _, p in per]); wt /= wt.sum()
def wq(v, qs=(.1, .25, .5, .75, .9)):
    v = np.array(v); o = np.argsort(v); c = np.cumsum(wt[o]); return [round(float(v[o][min(np.searchsorted(c, q), len(v) - 1)]), 3) for q in qs]
rep = dict(training_index_v4=dict(rows='continuation rows, stream-balanced', trans_p10_25_50_75_90=wq([w['trans'] for w, _ in per]), rot_p10_25_50_75_90=wq([w['rot'] for w, _ in per]),
                                  bank_filled=round(float(sum(p for (w, _), p in zip(per, wt) if w['bank_contract'] > 0)), 3), bank_mean=round(float(sum(p * w['bank_contract'] for (w, _), p in zip(per, wt))), 2),
                                  true_revisit=round(float(sum(p for (w, _), p in zip(per, wt) if w['revisits'] > 0)), 3)), inference={})
for scene in ('02267acf', '0238de14'):
    z = np.load(f'{gm_env.EVAL_INPUTS}/inputs/{scene}.npz'); K = torch.from_numpy(z['K'].astype(np.float32)); P0 = z['c2w']
    for st in (1, 2, 4, 8):
        for ret in (0, 1):
            P = P0[::st]; calls = (len(P) - 1) // 16
            if ret: n = 16 * (calls // 2); P = np.concatenate([P[:n + 1], P[n - 1::-1][:n]]); calls = 2 * (calls // 2)
            poses = torch.from_numpy(P).float(); tr, ro, bk, rv = [], [], [], []
            for c in range(1, calls):
                s = 1 + 4 * c; cmd = poses[1 + 16 * c: 17 + 16 * c]; a, b = poses[16 * c], poses[16 * c + 16]
                tr.append(float((b[:3, 3] - a[:3, 3]).norm())); ro.append(float(np.degrees(np.arccos(np.clip((np.trace((a[:3, :3].T @ b[:3, :3]).numpy()) - 1) / 2, -1, 1)))))
                bk.append(len(mc.bank_contract(poses[:16 * c + 1], cmd, K, s, max_refs=4))); rv.append(len(mc.revisits(poses[:16 * c + 1], cmd, s)) > 0)
            if tr: rep['inference'][f"{scene} stride {st} {'out-and-back' if ret else 'forward'}"] = dict(calls=calls, trans_median=round(float(np.median(tr)), 3), trans_min_max=[round(min(tr), 3), round(max(tr), 3)],
                    rot_median=round(float(np.median(ro)), 1), bank_filled=round(float(np.mean(np.array(bk) > 0)), 2), bank_mean=round(float(np.mean(bk)), 2), true_revisit=round(float(np.mean(rv)), 2))
(HERE / 'distribution_report.json').write_text(json.dumps(rep, indent=1))
print('TRAIN', rep['training_index_v4'])
for k, v in rep['inference'].items(): print(k, v)
