"""Fresh packed teacher, stage A (no render). One stream window per update, N chunks with independent
u ~ U(0.01, 0.99), chunk-causal attention, loss on every chunk; text dropout to the empty embedding."""
import argparse, json, os, random, resource, shutil, sys, time
from pathlib import Path
import torch
from data import corpus, RowSampler, Stream, build_row
from model_io import fresh_model, save_trainables, load_trainables, save_resume, load_resume, empty_text, run_id


def make_optimizer(params, lr_backbone, lr_new, wd):
    import bitsandbytes as bnb
    return bnb.optim.AdamW8bit([dict(params=[p for n, p in params if n.startswith('dit.')], lr=lr_backbone),
                                dict(params=[p for n, p in params if not n.startswith('dit.')], lr=lr_new)], weight_decay=wd)


_TABLE = {}


def time_table(shift=3., n=1000, sigma_min=0., sigma_max=1.):
    """Memorizon scheduler (memorizon/wan/scheduler.py:22-35) with training=True: sigmas = shift*s/(1+(shift-1)s) on a
    linear grid, Gaussian weight exp(-2((t-n/2)/n)^2) over the SHIFTED timesteps t=1000*sigma, min-subtracted and
    normalised to mean 1 on that table. One uniform index draws (u, weight) jointly."""
    key = (shift, n)
    if key not in _TABLE:
        s_ = torch.linspace(sigma_max, sigma_min, n + 1)[:-1]; sig = shift * s_ / (1 + (shift - 1) * s_); t = sig * 1000
        y = torch.exp(-2 * ((t - n / 2) / n) ** 2); y = y - y.min(); y = y * (n / y.sum())
        _TABLE[key] = (sig.clamp(1e-3, .999), y)
    return _TABLE[key]


def draw_u(n, g, dist='uniform', shift=3., u_min=.01, u_max=.99):
    """Per-chunk noise level and loss weight. 'uniform': U(u_min,u_max), weight 1. 'shift3': uniform index into the
    Memorizon shift-3 table (u>0.8 on ~43% of draws but ~20% of the total weight; ends get weight 0 as in the reference)."""
    if dist == 'uniform':
        return u_min + (u_max - u_min) * torch.rand(n, generator=g), torch.ones(n)
    if dist == 'shift3':
        sig, w = time_table(shift); idx = torch.randint(len(sig), (n,), generator=g); return sig[idx], w[idx]
    raise ValueError('unknown u distribution')


def step_loss(model, row, g, text_dropout, empty, u_min=.01, u_max=.99, u_dist='uniform'):
    """Returns (loss, report). Noise and u drawn from generator g (CPU)."""
    n = row['n_chunks']; x0 = row['target']
    u, w = draw_u(n, g, u_dist, u_min=u_min, u_max=u_max); eps = torch.randn(x0.shape, generator=g)
    uu = u.repeat_interleave(4).view(1, 1, 4 * n, 1, 1); noisy = (1 - uu) * x0 + uu * eps
    dropped = bool(torch.rand((), generator=g) < text_dropout); text = empty if dropped else row['text']
    dev = 'cuda'; lat = torch.cat([row['context'], noisy], 2).to(dev, torch.bfloat16)
    times = torch.cat([torch.zeros(row['context'].shape[2]), uu.flatten()]).to(dev)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        md = row.get('metadata'); md = None if md is None else md.to(dev, torch.bfloat16)
        pred = model(lat, row['rays'].to(dev, torch.bfloat16), row['roles'].to(dev), row['positions'].to(dev), times, text.to(dev, torch.bfloat16),
                     row['chunk_index'].to(dev), row['c2w'].to(dev), row['K'].to(dev), metadata=md, checkpoint_blocks=True).float()
    label = (eps - x0).to(dev); err = (pred - label).square()
    loss = (err.mean(dim=(0, 1, 3, 4)).view(n, 4).mean(1) * w.to(dev)).mean()                 # per-chunk MSE, time-weighted
    if not torch.isfinite(loss):
        raise RuntimeError('nonfinite loss')
    per_chunk = [round(float(err[:, :, 4 * j:4 * j + 4].detach().mean()), 4) for j in range(n)]
    return loss, dict(n=n, start=row['start'], bank=len(row['bank']), recent=len(row['recent']), u=[round(float(x), 3) for x in u], w=[round(float(x), 2) for x in w],
                      fm=per_chunk, text_dropped=dropped, render=bool(row.get('render')), stream=row['stream_id'][:8])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True); ap.add_argument('--updates', type=int, required=True); ap.add_argument('--seed', type=int, default=20261021)
    ap.add_argument('--scenes', type=int, default=0, help='learnability test: fixed subset of this many streams')
    ap.add_argument('--lr_backbone', type=float, default=3e-5); ap.add_argument('--lr_new', type=float, default=1e-4); ap.add_argument('--wd', type=float, default=1e-3)
    ap.add_argument('--clip', type=float, default=1.0); ap.add_argument('--text_dropout', type=float, default=0.2)
    ap.add_argument('--ckpt_every', type=int, default=100); ap.add_argument('--resume_every', type=int, default=250); ap.add_argument('--min_free_gb', type=float, default=17.)
    ap.add_argument('--prope', type=int, default=1); ap.add_argument('--resume', type=int, default=0); ap.add_argument('--max_hours', type=float, default=0.)
    ap.add_argument('--keep_ckpts', type=int, default=3, help='prune older trainables checkpoints (disk floor); 0 keeps all')
    ap.add_argument('--accum', type=int, default=1, help='rows per optimizer step (gradient accumulation)')
    ap.add_argument('--u_dist', default='uniform', choices=['uniform', 'shift3'])
    ap.add_argument('--lora_r', type=int, default=0, help='LoRA rank on FFN + cross-attention linears (0 = off)')
    ap.add_argument('--init_receipt', default='', help='start from a trainables receipt instead of stock (lineage recorded)')
    ap.add_argument('--render_p', type=float, default=0., help='stage B: probability that a row carries a ReSplat render of chunk 0')
    ap.add_argument('--corpus', default='448', choices=['448', '864']); ap.add_argument('--init', default='stock', choices=['stock', 'memorizon50'])
    ap.add_argument('--layout', default='fixed', choices=['fixed', 'memorizon']); ap.add_argument('--scale_mult', type=float, default=2.)
    ap.add_argument('--n_choices', default='2,4,6,8'); ap.add_argument('--n_probs', default='.15,.25,.30,.30'); ap.add_argument('--cam_flip_y', type=int, default=0); ap.add_argument('--max_bank', type=int, default=6)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    import packed_model; packed_model.LAYOUT = a.layout
    if a.corpus == '864':
        from data import corpus864, Stream864
        rows, flagged = corpus864('train'); stream_cls = lambda r: Stream864(r, a.scale_mult, bool(a.cam_flip_y))
    else:
        rows, flagged = corpus('train'); stream_cls = None
    if a.scenes:
        rows = random.Random(a.seed).sample(rows, a.scenes)
    (out / 'config.json').write_text(json.dumps(dict(vars(a), streams=len(rows), flagged_excluded=len(flagged), recipe='packed_chunk_causal_stageA_v1'), indent=1))
    model, params, fp = fresh_model(prope=bool(a.prope), lora_r=a.lora_r, init=a.init); opt = make_optimizer(params, a.lr_backbone, a.lr_new, a.wd)
    rid = run_id(out); lineage = dict(run_id=rid, init=fp, recipe='packed_chunk_causal_v2', corpus=a.corpus, layout=a.layout, scale_mult=a.scale_mult, cam_flip_y=bool(a.cam_flip_y))
    if a.init_receipt:
        r0 = load_trainables(model, a.init_receipt); lineage['init'] = dict(receipt=a.init_receipt, sha256=r0['sha256'], meta=r0.get('meta'))
    rb = None
    if a.render_p > 0:
        from render_rows import RenderBuilder
        from model_io import load_vae
        rb = RenderBuilder(load_vae(device='cpu').eval()); lineage['render_p'] = a.render_p
    sampler = RowSampler(rows, a.seed, render_builder=rb, render_p=a.render_p, stream_cls=stream_cls, n_choices=[int(x) for x in a.n_choices.split(',')],
                         n_probs=[float(x) for x in a.n_probs.split(',')], max_bank=a.max_bank); empty = empty_text(); update = 0; elapsed = 0.
    if a.resume and (out / 'resume.json').exists():
        update, extra = load_resume(model, opt, sampler, out); elapsed = extra.get('elapsed', 0.); print('resumed at', update, flush=True)
    elif (out / 'train.log').exists() and (out / 'train.log').stat().st_size:
        raise RuntimeError('run dir has a train.log but no resume slot was requested/found; refusing to restart silently')
    else:
        (out / 'lineage.json').write_text(json.dumps(lineage, indent=1))
    g = torch.Generator().manual_seed(a.seed + update); log = open(out / 'train.log', 'a'); model.train(); t_start = time.time() - elapsed
    if update and 'g_state' in extra:
        g.set_state(extra['g_state'])                                                         # bit-identical noise/u draws after a resume
    while update < a.updates:
        if a.max_hours and time.time() - t_start > 3600 * a.max_hours:
            print('wall budget reached', flush=True); break
        t0 = time.time(); reps = []; losses = []
        for _ in range(a.accum):
            row = sampler.draw(); loss, rep = step_loss(model, row, g, a.text_dropout, empty, u_dist=a.u_dist); (loss / a.accum).backward(); reps.append(rep); losses.append(float(loss))
        gn = float(torch.nn.utils.clip_grad_norm_([p for _, p in params], a.clip)); opt.step(); opt.zero_grad(set_to_none=True); update += 1
        rep = reps[0] if a.accum == 1 else dict(rows=reps, n=sum(r['n'] for r in reps), start=-1, bank=sum(r['bank'] > 0 for r in reps), recent=-1)
        rep.update(update=update, loss=round(sum(losses) / len(losses), 4), grad=round(gn, 4), sec=round(time.time() - t0, 1), peak=round(torch.cuda.max_memory_allocated() / 2 ** 30, 1),
                   rss=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 1), elapsed_h=round((time.time() - t_start) / 3600, 3))
        log.write(json.dumps(rep) + '\n'); log.flush()
        if update % 10 == 0:
            print(json.dumps(rep), flush=True)
        if update % a.ckpt_every == 0 or update == a.updates:
            if shutil.disk_usage(out).free < (a.min_free_gb + 3) * 2 ** 30:
                save_resume(model, opt, sampler, update, out, a.min_free_gb, dict(elapsed=time.time() - t_start, g_state=g.get_state()))
                raise RuntimeError('disk floor: cannot write the trainables checkpoint; resume slot saved, stopping')   # audit 10-21: never continue silently
            else:
                save_trainables(model, out / 'ckpt' / f'trainables_{update:05d}.pt', dict(update=update, recipe=lineage['recipe'], stock=fp, run_id=rid, lineage=lineage, config=str(out / 'config.json')))
                if a.keep_ckpts:
                    protected = set(json.load(open(out / 'protected.json'))) if (out / 'protected.json').exists() else set()
                    for old in sorted((out / 'ckpt').glob('trainables_*.pt'))[:-a.keep_ckpts]:
                        if int(old.stem.split('_')[1]) % 2000 and old.name not in protected:   # protected.json: pilot-100 / best checkpoints are never pruned
                            old.unlink(); old.with_suffix('.json').unlink(missing_ok=True)
        if update % a.resume_every == 0 or update == a.updates:
            save_resume(model, opt, sampler, update, out, a.min_free_gb, dict(elapsed=time.time() - t_start, g_state=g.get_state()))
    print('done', update, flush=True)


if __name__ == '__main__':
    main(); os._exit(0)
