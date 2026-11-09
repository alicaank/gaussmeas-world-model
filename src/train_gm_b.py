"""GM-B (8 Nov 22:30, user: "sadece GM-B 250 adim egit ve test et, egitilmis bozulmamis checkpointten"):
the Gaussian-meas recipe exactly (train_onpolicy.py --op_arm meas: Memorizon lr 2e-5/1e-4, no LoRA, render_p .75, 50% on-policy
pool rows at measured poses, n_chunks 2, max_bank 4, scale x2), continued from run_op/meas/ckpt/trainables_00250, with stage-B
context corruption applied to the RECORDED rows only (gm_b_corruption.py). Everything else runs through
train_onpolicy.py -> train_m864v.py -> train_b864.py -> train.py unchanged (imported read-only from fresh_teacher).
Usage: train_gm_b.py --gmb_seed 1 --op_arm meas --op_q .5 --op_seed 7 <train.py args incl. --out --init_receipt --accum>
"""
import json
import os
import random
import sys
from pathlib import Path

FT = '/home/hj/Alican/analysis/plan_base_restart_20261008/fresh_teacher_20261021'
HERE = str(Path(__file__).resolve().parent)
for p in (HERE, FT):
    if p not in sys.path:
        sys.path.insert(0, p)


def pop_arg(name, default, cast=str):
    if name in sys.argv:
        i = sys.argv.index(name); v = cast(sys.argv[i + 1]); del sys.argv[i:i + 2]; return v
    return default


def arg(name, default, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


GMB_SEED = pop_arg('--gmb_seed', 1, int)
import train_onpolicy  # noqa: E402,F401  pops --op_arm/--op_q/--op_seed, installs the on-policy sampler + m864v/b864 patches
import train  # noqa: E402
import data  # noqa: E402
from gm_b_corruption import GMBConfig, corrupt_packed  # noqa: E402

CFG = GMBConfig(seed=GMB_SEED, max_bank=arg('--max_bank', 4, int))
COUNTER = {'n': 0}
_draw = data.RowSampler.draw
_step = train.step_loss


def draw(self):
    """Recorded rows carry their stream (needed for the self-augmentation's auxiliary row); pool rows do not."""
    row = _draw(self); sid = row['stream_id']
    if not sid.startswith('OP') and sid in self.cache:
        row['_stream'] = self.cache[sid]
    return row


def step_loss(model, row, g, *args, **kwargs):
    stream = row.pop('_stream', None); i = COUNTER['n']; COUNTER['n'] += 1
    row, rec = corrupt_packed(row, stream, CFG, random.Random(f'gmb:{GMB_SEED}:{i}'), model=model)
    loss, rep = _step(model, row, g, *args, **kwargs); rep['gmb'] = rec
    return loss, rep


data.RowSampler.draw = draw; train.step_loss = step_loss

if __name__ == '__main__':
    out = Path(arg('--out', None)); out.mkdir(parents=True, exist_ok=True); accum = arg('--accum', 1, int)
    if arg('--resume', 0, int) and (out / 'resume.json').exists():
        COUNTER['n'] = int(json.loads((out / 'resume.json').read_text())['update']) * accum     # per-row corruption draws replay exactly
    (out / 'onpolicy.flag').write_text(f'arm={train_onpolicy.ARM} q={train_onpolicy.Q} op_seed={train_onpolicy.OP_SEED} pool={len(train_onpolicy.ITEMS)} items from {train_onpolicy.POOL}\n')
    (out / 'gmb.flag').write_text(json.dumps(dict(CFG.to_dict(), recipe='gm_b: stage-B context corruption on recorded rows (gm_b_corruption.py)'), indent=1))
    if arg('--render_p', 0., float) > 0 and not (train_onpolicy.train_m864v.train_b864.RENDERS / '_done').exists():
        raise SystemExit('precomputed renders missing')
    train.main(); os._exit(0)
