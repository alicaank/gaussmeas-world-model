"""Stage A of the contract recipe (data_contract/RECIPE.md, 11 Nov): the Memorizon-50 teacher continued WITHOUT render, in the plain scope of plain-350
(packed_model.PLAIN: Memorizon's own trainables, our Plucker / metadata / role inputs frozen at zero), with the new input contract:
  windows  data_contract/index_v4.json through the matched-window sampler of train_b864 (motion-weighted towards the inference range, >= 13 latents of
           history, true revisits up-weighted)
  bank     memory_contract.bank_contract_train (true revisits, first-seen owners, deterministic 20% bank dropout) = the function of the cache v4 builder
  text     --text_dropout (0.15)
  warm-up  --warmup N optimizer steps (fresh Adam)
  streams  --exclude FILE: streams dropped from the corpus (data_contract/bad_streams_v3.json: renders say their poses are inconsistent)
Per logged row: per-call translation / rotation and true-revisit count from data_contract/windows.jsonl, bank size, text_dropped.
Usage: train_a4.py --index FILE --warmup 50 <train.py args: --out --updates --seed --lr_backbone --lr_new --text_dropout --accum --u_dist --init memorizon50 ...>"""
import gm_env  # machine paths: gm_env.py at the repository root (scripts/env.sh puts it on PYTHONPATH)
import json, os, sys
from pathlib import Path
import numpy as np, torch
FT = gm_env.CORE; HERE = Path(__file__).resolve().parent
for p in (str(HERE), FT):
    if p not in sys.path: sys.path.insert(0, p)


def pop_arg(name, default, cast=str):
    if name in sys.argv:
        i = sys.argv.index(name); v = cast(sys.argv[i + 1]); del sys.argv[i:i + 2]; return v
    return default


INDEX_FILE = pop_arg('--index', str(HERE / 'data_contract/index_v4.json')); WARMUP = pop_arg('--warmup', 50, int); PLAIN = pop_arg('--plain', 1, int); EXCLUDE = pop_arg('--exclude', '')
import packed_model  # noqa: E402
packed_model.PLAIN = bool(PLAIN)
import train_b864  # noqa: E402  (matched-window sampler: stream scene-balanced, window uniform from INDEX[stream])
import train, data  # noqa: E402
import memory_contract  # noqa: E402
train_b864.INDEX = {k: [int(x) for x in v] for k, v in json.load(open(INDEX_FILE)).items()}
data.select_bank = memory_contract.bank_contract_train
BAD = set(json.load(open(EXCLUDE))['streams']) if EXCLUDE else set()                # streams whose renders say the POSES are inconsistent: wrong camera labels for A as well
if BAD:
    _corpus = data.corpus864
    def corpus864(split='train'):
        r = _corpus(split); rows = r[0] if isinstance(r, tuple) else r; kept = [x for x in rows if x['stream_id'] not in BAD]
        print('corpus: excluded', len(rows) - len(kept), 'streams of', len(rows), flush=True); return (kept, r[1]) if isinstance(r, tuple) else kept
    data.corpus864 = corpus864
TABLE = {}
for line in open(HERE / 'data_contract/windows.jsonl'):
    w = json.loads(line); TABLE[(w['stream'][:8], w['start'])] = (w['trans'], w['rot'], w['revisits'])
COUNTER = {'step': 0}; _step = train.step_loss; _mk = train.make_optimizer; _draw = data.RowSampler.draw


def draw(self):
    row = _draw(self); row['_key'] = (str(row['stream_id'])[:8], int(row['start'])); return row


def step_loss(model, row, g, *args, **kwargs):
    key = row.pop('_key', None); loss, rep = _step(model, row, g, *args, **kwargs)
    if key in TABLE: rep['trans'], rep['rot'], rep['revisits'] = TABLE[key]
    return loss, rep


def make_optimizer(params, lr_backbone, lr_new, wd):
    opt = _mk(params, lr_backbone, lr_new, wd); base = [g['lr'] for g in opt.param_groups]; inner = opt.step
    def step(*a, **k):
        f = 1. if WARMUP <= 0 else min(1., (COUNTER['step'] + 1) / WARMUP)
        for g, b in zip(opt.param_groups, base): g['lr'] = b * f
        COUNTER['step'] += 1; return inner(*a, **k)
    opt.step = step; return opt


data.RowSampler.draw = draw; train.step_loss = step_loss; train.make_optimizer = make_optimizer

if __name__ == '__main__':
    arg = lambda n, d, c=str: c(sys.argv[sys.argv.index(n) + 1]) if n in sys.argv else d
    out = Path(arg('--out', None)); out.mkdir(parents=True, exist_ok=True)
    if arg('--render_p', 0., float) > 0: raise SystemExit('stage A has no render rows (use train_gmp.py for B / C)')
    if arg('--resume', 0, int) and (out / 'resume.json').exists(): COUNTER['step'] = int(json.loads((out / 'resume.json').read_text())['update'])
    if PLAIN: (out / 'plain.flag').write_text('contract stage A: plain scope (Memorizon trainables), index v4, contract bank, text dropout, warm-up\n')
    (out / 'contract.flag').write_text(json.dumps(dict(index=INDEX_FILE, windows=sum(len(v) for v in train_b864.INDEX.values()), bank='memory_contract.bank_contract_train', bank_drop=memory_contract.BANK_DROP,
                                                       warmup=WARMUP, plain=bool(PLAIN), exclude=EXCLUDE, excluded_streams=len(BAD), recipe='contract stage A (train_a4.py)'), indent=1))
    train.main(); os._exit(0)
