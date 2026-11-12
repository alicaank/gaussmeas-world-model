"""Model-free dry run of a train_gmp.py training's row sequence (critique 11 Nov: GM' died at its first row on a cache window that was not built).
Same arguments as train_gmp.py. Imports train_gmp as a module (its argument pops and sampler / cache patches apply), builds the RowSampler exactly as
train.main does, and draws updates x accum rows with the render loader replaced by an existence check on the cache file (shapes returned as zeros).
Pool rows are built for real (their keys are checked). Exit 1 with the list of missing windows / failures. Usage: dry_run_sampler.py <train_gmp args>"""
import gm_env  # machine paths: gm_env.py at the repository root (scripts/env.sh puts it on PYTHONPATH)
import json, sys, time
from pathlib import Path
import numpy as np, torch
FT = gm_env.CORE; HERE = Path(__file__).resolve().parent
for p in (str(HERE), FT):
    if p not in sys.path: sys.path.insert(0, p)
argv = list(sys.argv)
import train_gmp  # noqa: E402  (pops its own flags from sys.argv, patches the sampler)
arg = lambda n, d, c=str: c(sys.argv[sys.argv.index(n) + 1]) if n in sys.argv else d
import data, packed_model  # noqa: E402
from data import corpus864, Stream864  # noqa: E402
packed_model.LAYOUT = arg('--layout', 'memorizon'); tb = train_gmp.train_b864; missing = []; seen = set()


def check(self, stream, start, bank):
    f = tb.RENDERS / stream.row['stream_id'] / f'start_{int(start)}.npz'; seen.add((stream.row['stream_id'], int(start)))
    if not f.exists(): missing.append((stream.row['stream_id'][:8], int(start)))
    H, W = stream.latents.shape[-2:]; return torch.zeros(1, 48, 4, H, W), torch.zeros(1, 2, 4, H, W)


tb.CachedBuilder.render_chunk0 = check
rows, _ = corpus864('train'); sm = arg('--scale_mult', 2., float); flip = bool(arg('--cam_flip_y', 0, int))
sampler = data.RowSampler(rows, arg('--seed', 20261021, int), render_builder=tb.CachedBuilder(None), render_p=arg('--render_p', 0., float), stream_cls=lambda r: Stream864(r, sm, flip),
                          n_choices=[int(x) for x in arg('--n_choices', '2').split(',')], n_probs=[float(x) for x in arg('--n_probs', '1').split(',')], max_bank=arg('--max_bank', 6, int))
n = arg('--updates', 0, int) * arg('--accum', 1, int); t0 = time.time(); kinds = {}; fails = []
for i in range(n):
    try:
        row = sampler.draw(); k = 'pool' if str(row['stream_id']).startswith('OP') else ('render' if row.get('render') else 'plain'); kinds[k] = kinds.get(k, 0) + 1
    except Exception as e:
        fails.append((i, repr(e)[:200]))
        if len(fails) > 20: break
rep = dict(rows=n, drawn=sum(kinds.values()), kinds=kinds, windows_checked=len(seen), missing=len(missing), missing_first=missing[:10], failures=fails[:10], seconds=round(time.time() - t0, 1),
           cache=str(tb.RENDERS), args=argv[1:])
print('DRY_RUN', json.dumps(rep), flush=True)
sys.exit(0 if not missing and not fails else 1)
