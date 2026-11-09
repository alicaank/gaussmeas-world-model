"""Does the model use the camera? Teacher-forced FM loss on recorded rows with the TRUE cameras vs a STILL command
(all chunk cameras = last context camera; rays + PRoPE both altered) vs REVERSED translation. If the model reads the
camera, loss(true) < loss(still) on most rows. No sampling, so it is sensitive long before DA3 on samples is."""
import argparse, json, os, random
import numpy as np, torch
from data import corpus, Stream, build_row
from foundation_model import packed_rays
from model_io import fresh_model, load_trainables

ap = argparse.ArgumentParser(); ap.add_argument('--receipts', nargs='+', required=True); ap.add_argument('--rows', type=int, default=40); ap.add_argument('--u', type=float, nargs='+', default=[0.5, 0.9])
ap.add_argument('--split', default='train'); ap.add_argument('--out', default='probe_camera_loss.json'); ap.add_argument('--scenes', type=int, default=0); ap.add_argument('--seed', type=int, default=1); a = ap.parse_args()
rows, _ = corpus(a.split, exclude_flagged=(a.split == 'train'))
if a.scenes:                                                     # the same fixed subset train.py --scenes uses (overfit probe)
    rows = random.Random(a.seed).sample(rows, a.scenes)
rng = random.Random(123); picks = rng.sample(rows, min(len(rows), a.rows))
def lora_of(rec):
    try:
        return int(json.load(open(json.load(open(rec))['meta']['config'])).get('lora_r', 0))
    except Exception:
        return 0
model, params, fp = fresh_model(prope=True, lora_r=lora_of(a.receipts[0])); model.eval(); results = {}


def variant_cams(stream, row, kind):
    tp = [0, *row['bank'], *row['recent']] + list(range(row['start'], row['start'] + 4 * row['n_chunks'])); L = 4 * tp[-1] + 1
    c2w_all = stream.c2w[:L].clone(); b = 4 * (row['start'] - 1)                        # RGB frame of the last context latent
    if kind == 'still':
        c2w_all[b + 1:] = c2w_all[b][None]
    elif kind == 'reverse':
        rel = c2w_all[b + 1:, :3, 3] - c2w_all[b, :3, 3][None]; c2w_all[b + 1:, :3, 3] = c2w_all[b, :3, 3][None] - rel
    rays = packed_rays(c2w_all, stream.K, tp); c2w = torch.stack([c2w_all[0 if p == 0 else 4 * p] for p in tp]); return rays, c2w


for rec in a.receipts:
    r_ = load_trainables(model, rec); name = f"{r_['meta'].get('run_id', '?')}:{os.path.basename(rec)}:{r_['sha256'][:12]}"; per = {k: [] for k in ('true', 'still', 'reverse')}
    for i, r in enumerate(picks):
        s = Stream(r); row = build_row(s, random.Random(i), n_chunks=2, start=None, first_call_p=.2)      # first-call rows included
        g = torch.Generator().manual_seed(i); eps = torch.randn(row['target'].shape, generator=g)
        for u in a.u:
            uu = torch.full((1, 1, 8, 1, 1), u); noisy = (1 - uu) * row['target'] + uu * eps; label = (eps - row['target']).cuda()
            lat = torch.cat([row['context'], noisy], 2).cuda().bfloat16(); times = torch.cat([torch.zeros(row['context'].shape[2]), torch.full((8,), u)]).cuda()
            for kind in per:
                rays, c2w = variant_cams(s, row, kind)
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    pred = model(lat, rays.cuda().bfloat16(), row['roles'].cuda(), row['positions'].cuda(), times, s.text.cuda().bfloat16(), row['chunk_index'].cuda(), c2w.cuda(), row['K'].cuda()).float()
                e = (pred - label).square()
                per[kind].append(dict(u=u, loss=float(e.mean()), c0=float(e[:, :, :4].mean()), c1=float(e[:, :, 4:].mean()), first=row['start'] == 1,
                                      motion=float((s.c2w[4 * (row['start'] + 7), :3, 3] - s.c2w[4 * (row['start'] - 1), :3, 3]).norm())))
    out = {}
    for u in a.u:
        T = [x for x in per['true'] if x['u'] == u]; ST = [x for x in per['still'] if x['u'] == u]; RV = [x for x in per['reverse'] if x['u'] == u]
        def paired(key, sel=lambda x: True):
            t = np.array([x[key] for x in T if sel(x)]); st = np.array([x[key] for x in ST if sel(x)]); rv = np.array([x[key] for x in RV if sel(x)])
            if len(t) == 0:
                return None
            d = (st - t) / t.mean(); dr = (rv - t) / t.mean()
            return dict(n=int(len(t)), true=round(float(t.mean()), 4), gain_still=round(float(d.mean()), 4), se_still=round(float(d.std(ddof=1) / np.sqrt(len(d))), 4) if len(d) > 1 else None,
                        gain_reverse=round(float(dr.mean()), 4), se_reverse=round(float(dr.std(ddof=1) / np.sqrt(len(dr))), 4) if len(dr) > 1 else None, win_still=round(float((t < st).mean()), 2))
        out[f'u{u}'] = dict(all=paired('loss'), chunk0=paired('c0'), chunk1=paired('c1'), first_call_chunk0=paired('c0', lambda x: x['first']), continuation_chunk0=paired('c0', lambda x: not x['first']))
    results[name] = out; print(name, json.dumps(out), flush=True)
json.dump(results, open(a.out, 'w'), indent=1); os._exit(0)
