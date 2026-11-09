"""GM-F: the Gaussian-meas recipe with the FRONTIER fix in training (9 Nov, user: "duzgunce test et, calisiyorsa egitimi baslat").
Recorded render rows: (1) the observed-support mask of the window (frontier_cache, build_frontier_cache.py) multiplies the render
(valid, conf) metadata, exactly as closed_loop_frontier does at inference, so render tokens over unobserved space are dropped and the
model must generate there; (2) regional render dropout: with p --gmf_strip a contiguous strip (left/right/top/bottom, 20-50% of the
grid) of render cells is invalidated, so "render missing -> generate" is practised at the frontier geometry instead of only on
whole-render dropout; (3) optionally the GM-B context corruption (--gmf_corrupt 1). On-policy pool rows: untouched (their renders come
from generated frames; no mask cache). Everything else = train_gm_b.py -> train_onpolicy.py -> train.py.
Usage: train_gm_f.py --gmb_seed 1 --gmf_corrupt 0|1 [--gmf_strip .3] [--frontier_cache DIR] --op_arm meas ... <train.py args>"""
import json
import os
import random
import sys
from pathlib import Path
import numpy as np
import torch

FT = '/home/hj/Alican/analysis/plan_base_restart_20261008/fresh_teacher_20261021'
HERE = Path(__file__).resolve().parent
for p in (str(HERE), FT):
    if p not in sys.path:
        sys.path.insert(0, p)


def pop_arg(name, default, cast=str):
    if name in sys.argv:
        i = sys.argv.index(name); v = cast(sys.argv[i + 1]); del sys.argv[i:i + 2]; return v
    return default


def arg(name, default, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


GMB_SEED = pop_arg('--gmb_seed', 1, int); CORRUPT = pop_arg('--gmf_corrupt', 0, int); P_STRIP = pop_arg('--gmf_strip', 0.3, float)
CACHE = Path(pop_arg('--frontier_cache', str(HERE / 'frontier_cache')))
import train_onpolicy  # noqa: E402,F401
import train  # noqa: E402
import data  # noqa: E402
from gm_b_corruption import GMBConfig, corrupt_packed  # noqa: E402

CFG = GMBConfig(seed=GMB_SEED, max_bank=arg('--max_bank', 4, int))
COUNTER = {'n': 0}; STATS = dict(rows=0, masked=0, strips=0, cells_dropped=0.0)
_draw = data.RowSampler.draw; _step = train.step_loss


def load_mask(sid, start):
    p = CACHE / sid / f'start_{start}.npz'
    if not p.exists():
        raise FileNotFoundError(f'frontier mask missing for {sid[:8]} start {start}: run build_frontier_cache.py')
    with np.load(p) as z:
        return torch.from_numpy(z['obs_f16'].astype(np.float32))          # (4,h,w)


def strip_mask(h, w, rng):
    """Contiguous strip of render cells to invalidate: side and width drawn from rng. (h,w) float, 0 inside the strip."""
    m = torch.ones(h, w); side = rng.choice(['left', 'right', 'top', 'bottom']); frac = rng.uniform(0.2, 0.5)
    if side == 'left': m[:, :max(1, int(round(w * frac)))] = 0
    elif side == 'right': m[:, w - max(1, int(round(w * frac))):] = 0
    elif side == 'top': m[:max(1, int(round(h * frac)))] = 0
    else: m[h - max(1, int(round(h * frac))):] = 0
    return m, side, frac


def frontier_row(row, rng, p_strip=P_STRIP, cache=True):
    """Applies the window's observed mask (+ a random strip) to the render (valid, conf) metadata of a recorded render row."""
    rec = dict(frontier='none')
    if str(row['stream_id']).startswith('OP') or not row.get('render') or row.get('metadata') is None:
        return row, rec
    nctx = 1 + len(row['bank']) + len(row['recent']); md = row['metadata'].clone(); h, w = md.shape[-2:]
    m = load_mask(row['stream_id'], int(row['start'])) if cache else torch.ones(4, h, w)
    rec['frontier'] = 'mask'; rec['cells_unobserved'] = round(float((m < .5).float().mean()), 4)
    if rng.random() < p_strip:
        s, side, frac = strip_mask(h, w, rng); m = m * s[None]; rec.update(strip=side, strip_frac=round(frac, 3))
    md[:, :, nctx:nctx + 4] = md[:, :, nctx:nctx + 4] * m[None, None]
    out = dict(row); out['metadata'] = md
    return out, rec


def draw(self):
    row = _draw(self); sid = row['stream_id']
    if not sid.startswith('OP') and sid in self.cache:
        row['_stream'] = self.cache[sid]
    return row


def step_loss(model, row, g, *args, **kwargs):
    stream = row.pop('_stream', None); i = COUNTER['n']; COUNTER['n'] += 1
    rng = random.Random(f'gmf:{GMB_SEED}:{i}')
    row, frec = frontier_row(row, rng)
    if CORRUPT:
        row, crec = corrupt_packed(row, stream, CFG, random.Random(f'gmb:{GMB_SEED}:{i}'), model=model)
    else:
        crec = dict(op='off')
    loss, rep = _step(model, row, g, *args, **kwargs); rep['gmf'] = frec; rep['gmb'] = crec
    return loss, rep


data.RowSampler.draw = draw; train.step_loss = step_loss

if __name__ == '__main__':
    out = Path(arg('--out', None)); out.mkdir(parents=True, exist_ok=True); accum = arg('--accum', 1, int)
    if not (CACHE / '_done').exists(): raise SystemExit(f'frontier cache incomplete: {CACHE}')
    if arg('--resume', 0, int) and (out / 'resume.json').exists():
        COUNTER['n'] = int(json.loads((out / 'resume.json').read_text())['update']) * accum
    (out / 'onpolicy.flag').write_text(f'arm={train_onpolicy.ARM} q={train_onpolicy.Q} op_seed={train_onpolicy.OP_SEED} pool={len(train_onpolicy.ITEMS)} items from {train_onpolicy.POOL}\n')
    (out / 'gmf.flag').write_text(json.dumps(dict(frontier_cache=str(CACHE), p_strip=P_STRIP, corrupt=bool(CORRUPT), corruption=CFG.to_dict() if CORRUPT else None,
                                                  recipe='gm_f: observed-support render mask + regional render dropout on recorded rows (train_gm_f.py)'), indent=1))
    if arg('--render_p', 0., float) > 0 and not (train_onpolicy.train_m864v.train_b864.RENDERS / '_done').exists():
        raise SystemExit('precomputed renders missing')
    train.main(); os._exit(0)
