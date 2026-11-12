"""GM' lineage trainer (10 Nov, user: mask in the training render records + GM-B corruption in the new training; darkening fixed at the source).
The Gaussian-meas recipe (train_onpolicy.py -> train_m864v.py -> train_b864.py -> train.py, imported read-only from fresh_teacher) with
  --mask_cache DIR   recorded render rows read their (valid, conf) metadata from the mask-v2 cache (build_mask_cache.py: alpha >= .7 and the
                     rendered depth inside a source frustum; the same frontier.py rule runs at inference). Render latents are unchanged.
  --corrupt v2|v1|off  GM-B context corruption on the recorded rows (gm_b_corruption.py). v2 = amplitude-preserving: x + s/(1-s) eps instead
                     of (1-s) x + s eps on bank / recent / render slots, and the one-step self-augmented recent latents rescaled to the clean
                     per-channel mean/std (GM-B v1 darkens: luminance x.90 and contrast x1.11 by the last third of the SOTA cohort, GM does not).
  --render_corrupt 0|1  1 (GM-B): render slots noised with the context and (valid, conf) x (1-s); 0: render left intact (recent/bank only)
  --warmup N         linear LR warm-up over the first N optimizer steps (fresh AdamW: the first steps are sign steps of one noisy batch).
  --pool DIR         on-policy pool directory (gen_pool_v2.py); default = fresh_teacher data864/onpolicy.
  --op_arm C         stage B' (recorded rows only); --op_arm meas --op_q .5 = GM' (50% of the render rows replaced by pool rows).
Usage: train_gmp.py --gmb_seed 1 --corrupt v2 --mask_cache mask_cache_v2 --warmup 10 [--pool DIR] --op_arm C|meas --op_q .5 --op_seed 7 <train.py args>"""
import gm_env  # machine paths: gm_env.py at the repository root (scripts/env.sh puts it on PYTHONPATH)
import json, os, random, sys
from pathlib import Path
import numpy as np, torch

FT = gm_env.CORE; HERE = Path(__file__).resolve().parent
for p in (str(HERE), FT):
    if p not in sys.path: sys.path.insert(0, p)


def pop_arg(name, default, cast=str):
    if name in sys.argv:
        i = sys.argv.index(name); v = cast(sys.argv[i + 1]); del sys.argv[i:i + 2]; return v
    return default


def arg(name, default, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


GMB_SEED = pop_arg('--gmb_seed', 1, int); RENDER_CORRUPT = pop_arg('--render_corrupt', 1, int); NEAR_CORRUPT = pop_arg('--near_corrupt', 1, int); CORRUPT = pop_arg('--corrupt', 'v2'); MASK = pop_arg('--mask_cache', ''); WARMUP = pop_arg('--warmup', 10, int); POOL_DIR = pop_arg('--pool', '')
if CORRUPT not in ('v2', 'v1', 'off', 'stat'): raise SystemExit('--corrupt v2|v1|off|stat')
MASK = Path(MASK) if MASK else None
RCACHE = pop_arg('--render_cache', ''); META = pop_arg('--meta', 'raw')        # 10 Nov 19:20: render cache v3 (exact geometry, build_render_cache_v3.py); --meta raw|frustum|v2
if POOL_DIR:
    import onpolicy_rows  # noqa: E402
    onpolicy_rows.POOL = Path(POOL_DIR)                                   # train_onpolicy reads POOL at import
import train_onpolicy  # noqa: E402,F401
import train, data  # noqa: E402
train_b864 = train_onpolicy.train_m864v.train_b864
if RCACHE:
    RCACHE = Path(RCACHE); train_b864.RENDERS = RCACHE
    train_b864.INDEX = {k: [int(x) for x in v] for k, v in json.load(open(RCACHE / 'index.json')).items()}
    MIN_PSNR = pop_arg('--min_psnr', 0., float)                                        # next pass (critique 10 Nov 20:40): drop windows whose render is not aligned with the target
    if MIN_PSNR > 0:                                                                   # (first-target PSNR stored by build_render_cache_v3.py; changes the row sequence of a seed)
        def _ok(sid, x):
            f = RCACHE / sid / f'start_{x}.npz'
            if not f.exists(): return False
            with np.load(f) as z: return float(z['psnr_first']) >= MIN_PSNR
        n0 = sum(len(v) for v in train_b864.INDEX.values()); train_b864.INDEX = {k: [x for x in v if _ok(k, x)] for k, v in train_b864.INDEX.items()}
        print('render windows kept by --min_psnr', MIN_PSNR, ':', sum(len(v) for v in train_b864.INDEX.values()), 'of', n0, flush=True)
EXCLUDE = pop_arg('--exclude', '')                                                  # 12 Nov: comma-separated stream lists (bad_streams_v3, fast_streams_v4, flag_streams_v4); 8-char prefix match
if EXCLUDE:
    BAD = {s[:8] for f in EXCLUDE.split(',') for s in json.load(open(f))['streams']}
    _corpus = data.corpus864
    def corpus864(split='train'):
        r = _corpus(split); rows = r[0] if isinstance(r, tuple) else r; kept = [x for x in rows if x['stream_id'][:8] not in BAD]
        print('corpus: excluded', len(rows) - len(kept), 'streams of', len(rows), flush=True); return (kept, r[1]) if isinstance(r, tuple) else kept
    data.corpus864 = corpus864
    train_b864.INDEX = {k: v for k, v in train_b864.INDEX.items() if k[:8] not in BAD}
    print('render index after --exclude:', len(train_b864.INDEX), 'streams,', sum(len(v) for v in train_b864.INDEX.values()), 'windows', flush=True)
from gm_b_corruption import GMBConfig, corrupt_packed  # noqa: E402
RENDER_MODE = pop_arg('--render_mode', 'token')                                     # 12 Nov paired stage-B arm: 'channel' = channel_render.py (render on the target tokens, no render slots)
if RENDER_MODE == 'channel':
    import channel_render  # noqa: E402
    _fresh = train.fresh_model; train_onpolicy.train_m864v.EXTRA = (*train_onpolicy.train_m864v.EXTRA, 'render_in.')
    def fresh_model(*a_, **k_):
        model, params, fp = _fresh(*a_, **k_); channel_render.install(model)
        params += [(n, p_) for n, p_ in model.named_parameters() if n.startswith('render_in.')]; return model, params, fp
    train.fresh_model = fresh_model
elif RENDER_MODE != 'token': raise SystemExit('--render_mode token|channel')
REJECT = pop_arg('--reject', 0., float)                                              # 12 Nov critique: outlier-row rejection (row_reject.py): a row whose gradient norm estimate exceeds
ACCUM = arg('--accum', 1, int); RR = None                                           # REJECT x the running median never enters the update (it would set the direction of a clipped step)
ROW_GRAD = pop_arg('--row_grad', 0, int)                                            # 12 Nov: log the accumulated gradient norm BEFORE each row of an update ('gacc'; the trainer logs the final one):
                                                                                    # every run so far has 3-10% of updates with pre-clip norms of 1e2-1e6; this names the row

BANK = pop_arg('--bank', 'stock')                                                   # 11 Nov plan: 'contract' = memory_contract.bank_contract in the sampler (same function as the cache v4 builder and the inference roller)
if BANK == 'contract':
    import memory_contract  # noqa: E402
    data.select_bank = memory_contract.bank_contract_train
elif BANK != 'stock': raise SystemExit('--bank stock|contract')
SELECTIVE = pop_arg('--selective', 0, int)                                         # 11 Nov critique: selective trust (anchor_seen.py)
if SELECTIVE:
    import anchor_seen  # noqa: E402
    _C2W = {}
    def _c2w(sid):
        if sid not in _C2W:
            with np.load(data.DATA864 / 'components' / sid / 'vae.npz') as z: _C2W[sid] = z['c2w'].astype(np.float64)
        return _C2W[sid]
    def selective_mark(row, stream):
        sid = str(row['stream_id']); real = sid[3:] if sid.startswith('OP') else sid
        c2w = stream.c2w.numpy().astype(np.float64) if stream is not None else _c2w(real)
        depth, K = anchor_seen.load(real); st = int(row['start'])
        seen = torch.from_numpy(anchor_seen.seen_cells(depth, K, [c2w[0 if p == 0 else 4 * p] for p in range(st, st + 4)]))
        nctx = 1 + len(row['bank']) + len(row['recent']); md = row['metadata'].clone(); md[:, 1, nctx:nctx + 4] = torch.where(seen[None], torch.ones_like(md[:, 1, nctx:nctx + 4]), md[:, 1, nctx:nctx + 4])
        row = dict(row, metadata=md, _seen=seen); return row
CFG = GMBConfig(seed=GMB_SEED, max_bank=arg('--max_bank', 4, int), amp_preserve=CORRUPT == 'v2', match_moments=CORRUPT == 'v2',
                render_follows=bool(RENDER_CORRUPT), near=bool(NEAR_CORRUPT), stat_only=CORRUPT == 'stat')   # 10 Nov critique: render noise + (valid, conf) x (1-s) may teach distrust of CORRECT memory
COUNTER = {'n': 0, 'step': 0}; MASK_FLIP = 0.10; STATS = dict(mask_rows=0, cells_before=0., cells_after=0., raised=0)
_draw = data.RowSampler.draw; _step = train.step_loss; _rc0 = train_b864.CachedBuilder.render_chunk0; _mk = train.make_optimizer


def masked_meta(meta, sid, start, bank):
    """Mask-v2 metadata of a cached window: the cached (valid, conf) times keep = clamp(v2 / raw, 0, 1) per cell, where v2 and raw are the
    pooled validity of ONE re-render with and without the mask (build_mask_cache.py). The ratio cancels re-render noise (ReSplat is not
    bit-deterministic) and can only remove validity. A window whose re-render disagrees with the cache on > MASK_FLIP of the cells
    (binary valid >= .5) is not the same render -> error."""
    f = MASK / sid / f'start_{start}.npz'
    if not f.exists(): raise FileNotFoundError(f'mask v2 missing for {sid[:8]} start {start}: run build_mask_cache.py')
    with np.load(f) as z:
        if [int(b) for b in z['bank']] != [int(b) for b in bank]: raise RuntimeError(f'mask cache bank != row bank for {sid[:8]} start {start}')
        v2 = torch.from_numpy(z['meta_f16'][0].astype(np.float32)); raw = torch.from_numpy(z['meta_raw_f16'][0].astype(np.float32))
    if v2.shape != meta.shape[2:] or raw.shape != v2.shape: raise RuntimeError(f'mask shape {tuple(v2.shape)} != meta {tuple(meta.shape)}')
    flips = float(((raw >= .5) != (meta[0, 0] >= .5)).float().mean())
    if flips > MASK_FLIP: raise RuntimeError(f'mask v2 re-render differs from the cached render for {sid[:8]} start {start}: {flips:.3f} of cells flip')
    keep = torch.where(raw > 1e-3, v2 / raw.clamp_min(1e-3), torch.zeros_like(raw)).clamp(0, 1)
    out = meta * keep[None, None]
    STATS['mask_rows'] += 1; STATS['cells_before'] += float((meta[:, 0] >= .5).float().mean()); STATS['cells_after'] += float((out[:, 0] >= .5).float().mean())
    return out


def render_chunk0(self, stream, start, bank):
    rz, meta = _rc0(self, stream, start, bank)
    if RCACHE and META != 'raw':                                                        # mask from the SAME render as the latents (no ratio trick needed)
        with np.load(RCACHE / stream.row['stream_id'] / f'start_{int(start)}.npz') as z: meta = torch.from_numpy(z[f'meta_{META}_f16'].astype(np.float32))[None]
    return (rz, masked_meta(meta, stream.row['stream_id'], int(start), bank)) if MASK is not None else (rz, meta)


def draw(self):
    row = _draw(self); sid = row['stream_id']
    if not sid.startswith('OP') and sid in self.cache: row['_stream'] = self.cache[sid]
    return row


def step_loss(model, row, g, *args, **kwargs):
    stream = row.pop('_stream', None); i = COUNTER['n']; COUNTER['n'] += 1
    gacc = None; gstate = RR.before(model, g) if RR is not None else None            # judges the previous row of this update (its backward has run)
    if ROW_GRAD: gacc = row_reject.grad_norm(model)
    if SELECTIVE and row.get('render') and row.get('metadata') is not None: row = selective_mark(row, stream)
    if CORRUPT != 'off':
        row, rec = corrupt_packed(row, stream, CFG, random.Random(f'gmb:{GMB_SEED}:{i}'), model=model)
    else:
        rec = dict(op='off')
    seen = row.pop('_seen', None)
    loss, rep = _step(model, row, g, *args, **kwargs); rep['gmb'] = rec
    if seen is not None: rep['seen'] = round(float(seen.float().mean()), 3)
    if gacc is not None: rep['gacc'] = round(gacc, 4)
    if RR is not None: RR.after(row, rep, gstate, args, kwargs)
    if row.get('render') and row.get('metadata') is not None:
        nctx = 1 + len(row['bank']) + len(row['recent']); rep['render_cells'] = round(float((row['metadata'][:, 0, nctx:nctx + 4] >= .5).float().mean()), 3)
    return loss, rep


def warmup_factor(k, n):
    """LR factor of optimizer step k (0-based): linear from 1/n to 1 over the first n steps."""
    return 1. if n <= 0 else min(1., (k + 1) / n)


def make_optimizer(params, lr_backbone, lr_new, wd):
    opt = _mk(params, lr_backbone, lr_new, wd); base = [g['lr'] for g in opt.param_groups]; inner = opt.step
    def step(*a, **k):
        f = warmup_factor(COUNTER['step'], WARMUP)
        for g, b in zip(opt.param_groups, base): g['lr'] = b * f
        COUNTER['step'] += 1; return inner(*a, **k)
    opt.step = step; opt.base_lrs = base; return opt


import row_reject  # noqa: E402
if REJECT > 0:
    RR = row_reject.RowReject(REJECT, ACCUM, _step); _clip = torch.nn.utils.clip_grad_norm_
    def clip_grad_norm_(*a_, **k_):
        RR.finish(); return _clip(*a_, **k_)                                         # the last row is judged before the trainer clips
    torch.nn.utils.clip_grad_norm_ = clip_grad_norm_
train_b864.CachedBuilder.render_chunk0 = render_chunk0; data.RowSampler.draw = draw; train.step_loss = step_loss; train.make_optimizer = make_optimizer

if __name__ == '__main__':
    out = Path(arg('--out', None)); out.mkdir(parents=True, exist_ok=True); accum = arg('--accum', 1, int)
    if MASK is not None and not (MASK / '_done').exists(): raise SystemExit(f'mask cache incomplete: {MASK}')
    if arg('--resume', 0, int) and (out / 'resume.json').exists():
        u = int(json.loads((out / 'resume.json').read_text())['update']); COUNTER['n'] = u * accum; COUNTER['step'] = u   # corruption draws and warm-up replay exactly
    (out / 'onpolicy.flag').write_text(f'arm={train_onpolicy.ARM} q={train_onpolicy.Q} op_seed={train_onpolicy.OP_SEED} pool={len(train_onpolicy.ITEMS)} items from {train_onpolicy.POOL}\n')
    if RENDER_MODE == 'channel': (out / 'channel.flag').write_text('render as input channels on the target tokens (channel_render.py), zero-init render_in\n')
    (out / 'gmp.flag').write_text(json.dumps(dict(corrupt=CORRUPT, render_mode=RENDER_MODE, bank=BANK, exclude=EXCLUDE, reject=REJECT, corruption=CFG.to_dict() if CORRUPT != 'off' else None, mask_cache=str(MASK) if MASK else None, render_cache=str(RCACHE) if RCACHE else None, meta=META if RCACHE else None,
                                                  mask_rule=json.loads((MASK / '_done').read_text()) if MASK else None, warmup=WARMUP, pool=str(train_onpolicy.POOL),
                                                  recipe="gm': mask-v2 render metadata + amplitude-preserving GM-B corruption + LR warm-up (train_gmp.py)"), indent=1))
    if arg('--render_p', 0., float) > 0 and not (train_b864.RENDERS / '_done').exists(): raise SystemExit('precomputed renders missing')
    try:
        train.main()
    finally:
        if RR is not None: print('rejected rows', RR.rejected, 'of', len(RR.hist) + RR.rejected, 'median row norm', round(RR.median(), 4), flush=True)
        if STATS['mask_rows']: print('mask rows', STATS['mask_rows'], 'valid cells before', round(STATS['cells_before'] / STATS['mask_rows'], 4), 'after', round(STATS['cells_after'] / STATS['mask_rows'], 4), flush=True)
    os._exit(0)
