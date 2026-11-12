"""Does the model READ the bank? (stage-A gate, critique 11 Nov 22:40; measure at checkpoints 150 / 250 / 350.) Flow-matching loss on held-out DEVELOPMENT
windows with >= 13 latents of history and a full contract bank (memory_contract.bank_contract, no dropout), each window with THE SAME u and noise:
  own    the row as the sampler builds it          none   the same window without bank slots
  swap   the bank LATENTS replaced by latents of another stream (same slots, same poses / rays: only the content is wrong)
use = loss(none) - loss(own)  (what the bank is worth);  harm = loss(swap) - loss(own)  (what a wrong bank costs). Both should GROW with the updates if the
contract rule is being learned. No render rows (stage A). Usage (cwd = fresh_teacher): probe_bank_swap.py name=receipt.json ... [--n 40] [--out_name X]"""
import gm_env  # machine paths: gm_env.py at the repository root (scripts/env.sh puts it on PYTHONPATH)
import json, random, sys
from pathlib import Path
import numpy as np, torch
FT = Path(gm_env.CORE); HERE = Path(__file__).resolve().parent
for p in (str(HERE), str(FT)):
    if p not in sys.path: sys.path.insert(0, p)
args = sys.argv[1:]; N = int(args[args.index('--n') + 1]) if '--n' in args else 40; OUTN = args[args.index('--out_name') + 1] if '--out_name' in args else 'bank_swap'
specs = dict(a.split('=', 1) for a in args if '=' in a and not a.startswith('--'))
import packed_model, data  # noqa: E402
import memory_contract  # noqa: E402
from train import step_loss  # noqa: E402
from data import build_row, Stream864  # noqa: E402
from model_io import fresh_model, load_trainables, empty_text, resolve_init  # noqa: E402
data.select_bank = memory_contract.bank_contract
recs = [json.loads(l) for l in open(FT / 'data864/streams.jsonl')]; dev = [r for r in recs if r['split'] == 'development' and r['latent_count'] >= 30 and r.get('mpu_valid', True)]
rng = random.Random(0); rng.shuffle(dev); streams = {}; cand = []
for r in dev:
    st = Stream864(r, 2.); streams[r['stream_id']] = st; starts = list(range(14, st.L - 8)); rng.shuffle(starts)
    got = 0
    for s in starts[:12]:                                                   # only 20 development streams exist: up to two windows each
        b = memory_contract.bank_contract(st.c2w[:4 * (s - 1) + 1], st.c2w[4 * s - 3: 4 * s + 13], st.K, s, max_refs=4)
        if len(b) == 4: cand.append((r['stream_id'], s)); got += 1
        if got == 2: break
    if len(cand) >= N: break
print('windows', len(cand), 'from', len(dev), 'development streams', flush=True); empty = empty_text(); out = {}; ids = [c[0] for c in cand]


def rows_of(k):
    sid, s = cand[k]; st = streams[sid]; own = build_row(st, random.Random(7), n_chunks=2, start=s, render_builder=None, render_p=0., max_bank=4)
    none = build_row(st, random.Random(7), n_chunks=2, start=s, render_builder=None, render_p=0., max_bank=0)
    other = streams[ids[(k + 1) % len(ids)]]; nb = len(own['bank']); swap = dict(own); ctx = own['context'].clone(); j = random.Random(k).sample(range(1, other.L), nb)
    ctx[:, :, 1:1 + nb] = other.latents[:, :, j].to(ctx.dtype); swap['context'] = ctx
    return dict(own=own, none=none, swap=swap), s


for name, receipt in specs.items():
    rec = json.loads(Path(receipt).read_text()); lin = rec['meta'].get('lineage', {}); init = resolve_init(lin); packed_model.LAYOUT = lin.get('layout', 'memorizon')
    cfg = Path(rec['meta']['config']); cfg = cfg if cfg.is_absolute() else FT / cfg; packed_model.PLAIN = (cfg.parent / 'plain.flag').exists()
    model, _, _ = fresh_model(prope=True, lora_r=int(json.loads(cfg.read_text()).get('lora_r', 0)), init=init, md_ch=int(lin.get('md_ch', 2))); load_trainables(model, receipt); model.eval()
    L = {k: [] for k in ('own', 'none', 'swap')}
    for k in range(len(cand)):
        rows, s = rows_of(k)
        for seed in (0, 1):
            for kind, r in rows.items():
                g = torch.Generator().manual_seed(1000 * seed + s)
                with torch.no_grad(): _, rep = step_loss(model, r, g, 0., empty, u_dist='shift3')
                L[kind].append(np.mean(rep['fm']))
    A = {k: np.array(v) for k, v in L.items()}; st_ = lambda d: [round(float(d.mean()), 4), round(float(d.std() / np.sqrt(d.size)), 4)]
    out[name] = dict(loss={k: round(float(v.mean()), 4) for k, v in A.items()}, use=st_(A['none'] - A['own']), harm=st_(A['swap'] - A['own']), n=int(A['own'].size))
    print(name, json.dumps(out[name]), flush=True); del model; torch.cuda.empty_cache()
    (HERE / f'eval_gm/diag/{OUTN}.json').write_text(json.dumps(out, indent=1))
print('BANK_SWAP_OK', flush=True)
