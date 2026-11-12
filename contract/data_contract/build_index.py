"""Window index v4 (11 Nov plan, data preparation). The base sampler draws a stream scene-balanced and then a window uniformly from index[stream], so the window
mix is decided HERE. Per stream K = 5 windows: one first-call window (start 1, = first_call_p .2) + four continuation windows drawn without replacement with
weights  w = speed(trans) x history(start) x revisit:
  speed:   per-call translation in [.05, .15) x30, [.15, .25) x8, [.25, .5) x1, >= .5 x.3, < .05 x2 (upper bound of the corpus = 48% in [.05,.15); 6/3/1/.5 gave 22%)  (inference target ~.1 gauge units per call; corpus median .35)
  history: start >= 14 x1 (>= 13 latents before the window, so the bank can fill), else x.15
  revisit: x2 if the contract finds a true revisit for this window
Windows at least 3 latents apart inside a stream. Seeded. -> data_contract/index_v4.json (same format as the render index) + index_v4_report.json with the
stream-balanced distributions of the OLD index and the new one (what the sampler will actually draw)."""
import json, random
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; K = 5; rng = random.Random(20261111)
W = [json.loads(l) for l in open(HERE / 'windows.jsonl')]; by = {}
for w in W: by.setdefault(w['stream'], {})[w['start']] = w
speed = lambda t: 2. if t < .05 else 30. if t < .15 else 8. if t < .25 else 1. if t < .5 else .3
index = {}
for sid, ws in by.items():
    pick = [1] if 1 in ws else []; cand = [s for s in ws if s > 1]
    while len(pick) < K and cand:
        wts = [speed(ws[s]['trans']) * (1. if s >= 14 else .15) * (2. if ws[s]['revisits'] else 1.) for s in cand]
        s = rng.choices(cand, wts)[0]; pick.append(s); cand = [c for c in cand if abs(c - s) >= 3]
    index[sid] = sorted(pick)
(HERE / 'index_v4.json').write_text(json.dumps(index))
_old = HERE.parent / 'render_cache_v3/index.json'; old = json.load(open(_old)) if _old.exists() else None   # the previous index: only for the comparison in the report
def report(ix):
    per = []                                                              # stream-balanced: every stream weighs the same, its windows share that weight
    for sid, starts in ix.items():
        for s in starts:
            w = by.get(sid, {}).get(int(s))
            if w: per.append((w, 1. / len(starts)))
    wt = np.array([p for _, p in per]); wt /= wt.sum(); f = lambda cond: round(float(sum(p for (w, _), p in zip(per, wt) if cond(w))), 3)
    cont = lambda w: w['start'] > 1
    tr = np.array([w['trans'] for w, _ in per]); order = np.argsort(tr); cdf = np.cumsum(wt[order]); qs = [round(float(tr[order][np.searchsorted(cdf, q)]), 3) for q in (.1, .25, .5, .75, .9)]
    return dict(windows=len(per), first_call=f(lambda w: w['start'] == 1), start_ge_14=f(lambda w: w['start'] >= 14), trans_p10_25_50_75_90=qs,
                trans_05_15=f(lambda w: cont(w) and .05 <= w['trans'] < .15), trans_15_25=f(lambda w: cont(w) and .15 <= w['trans'] < .25), trans_25_50=f(lambda w: cont(w) and .25 <= w['trans'] < .5),
                trans_ge_50=f(lambda w: cont(w) and w['trans'] >= .5), rot_le_10=f(lambda w: cont(w) and w['rot'] <= 10), bank_stock_filled=f(lambda w: cont(w) and w['bank_stock'] > 0),
                bank_contract_filled=f(lambda w: cont(w) and w['bank_contract'] > 0), true_revisit=f(lambda w: cont(w) and w['revisits'] > 0))
rep = dict(old_index=report(old) if old else None, index_v4=report(index), note='shares are fractions of ALL rows of a scene-balanced draw; continuation shares exclude start 1')
(HERE / 'index_v4_report.json').write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))
