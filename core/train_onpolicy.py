"""Measured-write training test (user 10-23), three matched arms continued from the memory model (Memorizon settings +
our inputs + render rows, run_v864/mem) for the same number of updates with the same seed, hence the SAME base row stream:
  --op_arm C     control: the base rows only (recorded prefix renders, as the memory model was trained);
  --op_arm cmd   with probability --op_q a render row is replaced by an on-policy pool row, own frames at COMMANDED poses;
  --op_arm meas  the same replacements (same updates, same pool items), own frames at MEASURED poses.
The replacement decision and the pool index come from a separate RNG that every arm advances identically (C included), and
it is saved with the sampler state for resumes. Everything else is train_m864v.py (Memorizon lr / no LoRA / no prompt dropout).
Usage: train_onpolicy.py --op_arm C|cmd|meas|mcc|cmc|ccm [--op_q .5] [--op_seed 7] <train_m864v.py args>"""
import os, random, sys
from pathlib import Path
import numpy as np


def pop_arg(name, default, cast=str):
    if name in sys.argv:
        i = sys.argv.index(name); v = cast(sys.argv[i + 1]); del sys.argv[i:i + 2]; return v
    return default


ARM = pop_arg('--op_arm', None); Q = pop_arg('--op_q', .5, float); OP_SEED = pop_arg('--op_seed', 7, int)
if ARM != 'C':
    from onpolicy_rows import split_variant
    try:
        split_variant(ARM)                                                   # 'cmd' | 'meas' | 3-letter c/m code (bank, pose, render)
    except ValueError:
        raise SystemExit('--op_arm C|cmd|meas|<c/m code> required')
import train_m864v  # noqa: F401  (matched-window sampler, render cache, partial-init loader)
import train, data
from data import corpus864, Stream864
from onpolicy_rows import POOL, build_onpolicy_row

ITEMS = sorted(POOL.glob('*_s*.npz'))
if not ITEMS:
    raise SystemExit('on-policy pool is empty: run gen_onpolicy.py first')
BY_ID = {r['stream_id']: r for r in corpus864('train')[0]}; _streams = {}; _base_draw = data.RowSampler.draw


def _stream(sid):
    if sid not in _streams:
        if len(_streams) > 64:
            _streams.pop(next(iter(_streams)))
        _streams[sid] = Stream864(BY_ID[sid], 2.)
    return _streams[sid]


def draw(self):
    row = _base_draw(self)
    if not hasattr(self, 'op_rng'):
        self.op_rng = random.Random(OP_SEED)
    u = self.op_rng.random(); idx = self.op_rng.randrange(len(ITEMS))                      # advanced identically in every arm
    if ARM != 'C' and row.get('render') and u < Q:
        with np.load(ITEMS[idx]) as z:
            item = {k: z[k] for k in z.files}
        row = build_onpolicy_row(item, _stream(str(item['stream_id'])), ARM)
    return row


_sd, _lsd = data.RowSampler.state_dict, data.RowSampler.load_state_dict
def state_dict(self):
    d = _sd(self); d['op_rng'] = (self.op_rng if hasattr(self, 'op_rng') else random.Random(OP_SEED)).getstate(); return d
def load_state_dict(self, s):
    _lsd(self, s); self.op_rng = random.Random(OP_SEED)
    if 'op_rng' in s:
        st = s['op_rng']; self.op_rng.setstate((st[0], tuple(st[1]), st[2]) if isinstance(st, (list, tuple)) else st)


data.RowSampler.draw = draw; data.RowSampler.state_dict = state_dict; data.RowSampler.load_state_dict = load_state_dict

if __name__ == '__main__':
    out = Path(sys.argv[sys.argv.index('--out') + 1]); out.mkdir(parents=True, exist_ok=True)
    (out / 'onpolicy.flag').write_text(f'arm={ARM} q={Q} op_seed={OP_SEED} pool={len(ITEMS)} items from {POOL}\n')
    if float(sys.argv[sys.argv.index('--render_p') + 1]) > 0 and not (train_m864v.train_b864.RENDERS / '_done').exists():
        raise SystemExit('precomputed renders missing')
    train.main(); os._exit(0)
