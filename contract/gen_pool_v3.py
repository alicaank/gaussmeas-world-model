"""On-policy pool v3 = the ORIGINAL pool recipe (fresh_teacher gen_onpolicy.py: 300 streams, seed 20261024, 2 calls open loop along the recorded
commands, jointly measured latent ends, items at s = 5 and 9 with commanded and measured labelling, no mask) with only the render corrected
(resplat_fixed.py) and the output folder moved (critique 10 Nov 20:00: GM' must be GM's exact recipe, one change per stage).
gen_onpolicy.py itself is imported unchanged: its POOL and its OwnRender864 are replaced before main() runs.
Usage (cwd = fresh_teacher): gen_pool_v3.py --out DIR --resplat v8|v16p --receipt R [--n_streams 300] [--seed 20261024]"""
import gm_env  # machine paths: gm_env.py at the repository root (scripts/env.sh puts it on PYTHONPATH)
import json, os, sys
from pathlib import Path
import numpy as np
FT = gm_env.CORE; HERE = Path(__file__).resolve().parent
for p in (str(HERE), FT):
    if p not in sys.path: sys.path.insert(0, p)


def pop(name):
    i = sys.argv.index(name); v = sys.argv[i + 1]; del sys.argv[i:i + 2]; return v


OUT = Path(pop('--out')); KIND = pop('--resplat'); OUT.mkdir(parents=True, exist_ok=True)
import onpolicy_rows  # noqa: E402
onpolicy_rows.POOL = OUT
import frontier, resplat_fixed  # noqa: E402
frontier.install('off', 1, str(OUT / 'frontier_log.jsonl'), size=dict(v8=(512, 960), v16s=(512, 960), v16p=(540, 960))[KIND])
import gen_onpolicy  # noqa: E402
assert gen_onpolicy.POOL == OUT
_Own = gen_onpolicy.OwnRender864


def own(vae, da3):
    r = _Own(vae, da3); r.renderer, _ = resplat_fixed.make(KIND); return r


gen_onpolicy.OwnRender864 = own
if __name__ == '__main__':
    gen_onpolicy.main()
    items = sorted(OUT.glob('*_s*.npz')); gaps = []
    for f in items:
        with np.load(f) as z: gaps.append(float(z['label_gap'][-1]))
    (OUT / '_done').write_text(json.dumps(dict(items=len(items), streams=len({f.name.rsplit('_s', 1)[0] for f in items}), label_gap_end_median=float(np.median(gaps)) if gaps else None,
                                               label_gap_end_p90=float(np.percentile(gaps, 90)) if gaps else None, resplat=KIND, recipe='gen_onpolicy.py unchanged, corrected render')))
    os._exit(0)
