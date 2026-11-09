"""Arms G / G+B of PILOT_PLAN_CVPR_20261104.md: train_m864v (Memorizon settings + our inputs, matched windows) with
  - recorded render rows from the 2Xplat cache (rows_g.CachedBuilder2X, precompute_renders_2x.py; same windows as data864/renders),
    whose bank is the cached VISIBILITY bank;
  - round-2 on-policy pool rows (rows_g.build_row_g, gen_onpolicy2.py) replacing render rows with probability --op_q, drawn by a
    separate op_rng advanced identically whatever the arm (as train_onpolicy.py);
  - --op_arm gb: PackedTeacherCorr (map-addressed attention bias, zero-init gains at lr_new) and a correspondence matrix per row;
  - --op_arm gr: PackedTeacherCorr mode 'restricted' (GEAR-style new narrow PRoPE reader every 2nd block, target queries read only
    their corresponded context tokens, zero-init out_proj) with the same correspondence matrix.
Usage: train_g.py --op_arm g|gb [--op_q .5] [--op_seed 7] <train_m864v.py args, incl. --init_receipt run_op/meas/ckpt/trainables_00250.json --render_p .75>"""
import json, os, random, sys
from pathlib import Path
import numpy as np, torch


def pop_arg(name, default, cast=str):
    if name in sys.argv:
        i = sys.argv.index(name); v = cast(sys.argv[i + 1]); del sys.argv[i:i + 2]; return v
    return default


ARM = pop_arg('--op_arm', None); Q = pop_arg('--op_q', .5, float); OP_SEED = pop_arg('--op_seed', 7, int)
if ARM not in ('g', 'gb', 'gr', 'gd'):
    raise SystemExit('--op_arm g|gb|gr|gd required (gb = soft bias, gr = restricted reader every 2nd block, gd = the same reader unrestricted = capacity-matched control)')
import train_m864v  # noqa: F401  (matched-window sampler, load_init, Memorizon settings)
import train_b864, train, data, render_rows, model_io
from data import corpus864, Stream864
from rows_g import POOL2, RENDERS2X, CachedBuilder2X, build_row_g
import packed_model_corr

if not (RENDERS2X / '_done').exists():
    raise SystemExit('2Xplat renders of the recorded windows missing: run precompute_renders_2x.py first')
ITEMS = sorted(POOL2.glob('*_s*.npz'))
if not ITEMS:
    raise SystemExit('round-2 pool empty: run gen_onpolicy2.py first')
BUILDER = CachedBuilder2X()
train_b864.INDEX = {k: [int(s) for s in v] for k, v in json.load(open(RENDERS2X / 'index.json')).items()}      # same windows as renders/ once complete
render_rows.RenderBuilder = CachedBuilder2X
BY_ID = {r['stream_id']: r for r in corpus864('train')[0]}; _streams = {}; _CUR = {}
_select_bank = data.select_bank; _build_row = data.build_row; _base_draw = data.RowSampler.draw


def select_bank_cached(observed_c2w, requested_c2w, K, prefix_latents, *a, **k):
    """Recorded rows: the cached visibility bank of the current (stream, start) window when one exists."""
    st, start = _CUR.get('stream'), _CUR.get('start')
    if st is not None and start == prefix_latents and CachedBuilder2X.path(st, start).exists():
        return BUILDER.cached_bank(st, start)
    return _select_bank(observed_c2w, requested_c2w, K, prefix_latents, *a, **k)


def build_row_g_recorded(stream, rng, n_chunks=None, start=None, first_call_p=.2, render_builder=None, render_p=0., max_bank=6):
    _CUR['stream'], _CUR['start'] = stream, start
    try:
        row = _build_row(stream, rng, n_chunks=n_chunks, start=start, first_call_p=first_call_p, render_builder=render_builder, render_p=render_p, max_bank=max_bank)
    finally:
        _CUR.clear()
    if ARM in ('gb', 'gr', 'gd') and row.get('render') and isinstance(render_builder, CachedBuilder2X):
        row = render_builder.attach_corr(row, stream)
    return row


data.select_bank = select_bank_cached; train_b864.build_row = build_row_g_recorded


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
    u = self.op_rng.random(); idx = self.op_rng.randrange(len(ITEMS))
    if row.get('render') and u < Q:
        with np.load(ITEMS[idx]) as z:
            item = {k: z[k] for k in z.files}
        row = build_row_g(item, _stream(str(item['stream_id'])), with_corr=ARM in ('gb', 'gr', 'gd'))
    return row


_sd, _lsd = data.RowSampler.state_dict, data.RowSampler.load_state_dict


def state_dict(self):
    d = _sd(self); d['op_rng'] = (self.op_rng if hasattr(self, 'op_rng') else random.Random(OP_SEED)).getstate(); return d


def load_state_dict(self, s):
    _lsd(self, s)
    if 'op_rng' in s:
        self.op_rng = random.Random(OP_SEED); self.op_rng.setstate(s['op_rng'])


data.RowSampler.draw = draw; data.RowSampler.state_dict = state_dict; data.RowSampler.load_state_dict = load_state_dict

# ---- model: corr gains (gb) ----------------------------------------------------------------------------------------------
_fresh = train.fresh_model; _load_init = train.load_trainables


def fresh_model_g(*a, **k):
    model, params, fp = _fresh(*a, **k)
    if ARM in ('gb', 'gr', 'gd'):
        model = packed_model_corr.upgrade(model, mode={'gb': 'bias', 'gr': 'restricted', 'gd': 'dense'}[ARM], reader_every=2); params = model.configure_trainables()
    return model, params, fp


def load_init_g(model, receipt_path, strict_sha=True):
    """train_m864v.load_init but 'corr_gain' may be missing from the init checkpoint (zero = identity)."""
    receipt = json.loads(Path(receipt_path).read_text())
    if strict_sha and model_io.sha256(receipt['path']) != receipt['sha256']:
        raise ValueError('checkpoint sha mismatch: ' + receipt['path'])
    state = torch.load(receipt['path'], map_location='cpu')['trainables']
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    missing, extra = set(params) - set(state), set(state) - set(params)
    allowed = train_m864v.EXTRA + ('corr_gain', 'reader.')
    if extra or any(not n.startswith(allowed) for n in missing):
        raise ValueError(f'trainable scope differs beyond our input branches: missing {sorted(missing)[:4]} extra {sorted(extra)[:4]}')
    if any(float(params[n].abs().max()) != 0. for n in missing if not n.startswith('reader.') or n.startswith('reader.') and 'out_proj' in n):
        raise ValueError('new branches are not at their zero init')          # reader q/k/v are random by design; its out_proj must be zero
    with torch.no_grad():
        for n in state:
            params[n].copy_(state[n].to(params[n]))
    return receipt


_step = train.step_loss


def step_loss_g(model, row, g, text_dropout, empty, u_min=.01, u_max=.99, u_dist='uniform'):
    """train.step_loss + corr=row['corr'] for PackedTeacherCorr (dense (S,S) on the GPU inside the model)."""
    n = row['n_chunks']; x0 = row['target']
    u, w = train.draw_u(n, g, u_dist, u_min=u_min, u_max=u_max); eps = torch.randn(x0.shape, generator=g)
    uu = u.repeat_interleave(4).view(1, 1, 4 * n, 1, 1); noisy = (1 - uu) * x0 + uu * eps
    dropped = bool(torch.rand((), generator=g) < text_dropout); text = empty if dropped else row['text']
    dev = 'cuda'; lat = torch.cat([row['context'], noisy], 2).to(dev, torch.bfloat16)
    times = torch.cat([torch.zeros(row['context'].shape[2]), uu.flatten()]).to(dev)
    corr = row.get('corr'); extra = {} if corr is None or not isinstance(model, packed_model_corr.PackedTeacherCorr) else dict(corr=corr.to(dev))
    with torch.autocast('cuda', dtype=torch.bfloat16):
        md = row.get('metadata'); md = None if md is None else md.to(dev, torch.bfloat16)
        pred = model(lat, row['rays'].to(dev, torch.bfloat16), row['roles'].to(dev), row['positions'].to(dev), times, text.to(dev, torch.bfloat16),
                     row['chunk_index'].to(dev), row['c2w'].to(dev), row['K'].to(dev), metadata=md, checkpoint_blocks=True, **extra).float()
    label = (eps - x0).to(dev); err = (pred - label).square()
    loss = (err.mean(dim=(0, 1, 3, 4)).view(n, 4).mean(1) * w.to(dev)).mean()
    if not torch.isfinite(loss):
        raise RuntimeError('nonfinite loss')
    per_chunk = [round(float(err[:, :, 4 * j:4 * j + 4].detach().mean()), 4) for j in range(n)]
    rep = dict(n=n, start=row['start'], bank=len(row['bank']), recent=len(row['recent']), u=[round(float(x), 3) for x in u], w=[round(float(x), 2) for x in w],
               fm=per_chunk, text_dropped=dropped, render=bool(row.get('render')), stream=row['stream_id'][:8], corr_nnz=int(corr._nnz()) if corr is not None else 0)
    if isinstance(model, packed_model_corr.PackedTeacherCorr):
        rep['corr_gain'] = [round(float(x), 4) for x in model.corr_gain.detach().cpu()[:4]]
        if model.reader is not None:
            rep['reader_out'] = round(float(model.reader[0].out_proj.weight.detach().abs().mean()), 6)
    return loss, rep


train.fresh_model = fresh_model_g; train.load_trainables = load_init_g; train.step_loss = step_loss_g

if __name__ == '__main__':
    out = Path(sys.argv[sys.argv.index('--out') + 1]); out.mkdir(parents=True, exist_ok=True)
    (out / 'g.flag').write_text(json.dumps(dict(op_arm=ARM, op_q=Q, op_seed=OP_SEED, pool=str(POOL2), pool_items=len(ITEMS), pool_meta=json.load(open(POOL2 / 'meta.json')),
                                                renders=str(RENDERS2X), renders_done=json.loads((RENDERS2X / '_done').read_text()), memory='2xplat')))
    train.main(); os._exit(0)
