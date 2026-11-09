"""Stage-B context corruption for the packed Gaussian-meas rows (fresh_teacher data.build_row layout).

Row context = [anchor | bank | recent | (4 render slots)], targets separate, context slots carry t=0 (packed_model forbids
noisy context timesteps), so the one-step self-augmentation of the RECENT latents is obtained from an auxiliary packed
sequence: the model's own one-step x0 of latents start-4..start-1 at noise s given their recorded predecessors
(data.build_row(start-4, n_chunks=1)) - the chunk the model would have produced one call earlier. Bank latents are noised at
the same s (not denoised: they are non-consecutive retrieved views). Rates = native stage B (train_robust.py ARMS['b']):
self-aug p .5, s ~ U(.05,.5); else noise p .3, s ~ U(0,.3); render slots noised at s, (valid, conf) metadata x (1-s).
Anchor, targets, labels and on-policy pool rows ('OP...' stream ids) are never touched.
"""
import random
from dataclasses import dataclass, asdict
import torch


@dataclass
class GMBConfig:
    p_self_aug: float = 0.5
    s_self: tuple = (0.05, 0.5)
    p_noise: float = 0.3
    s_max: float = 0.3
    render_follows: bool = True
    max_bank: int = 4
    seed: int = 1

    def to_dict(self):
        return asdict(self)


def noise_slots(x, slots, s, gen):
    """FlowMatch law x_s = (1-s) x0 + s eps on the given frame slots of a (1,C,F,h,w) tensor; returns a new tensor."""
    out = x.clone()
    if not slots or s <= 0:
        return out
    sub = out[:, :, slots].float(); eps = torch.randn(sub.shape, generator=gen)
    out[:, :, slots] = ((1.0 - s) * sub + s * eps).to(out.dtype)
    return out


def dit_velocity(model, aux, lat, times):
    """Production velocity call for the auxiliary packed sequence (no grad, bf16 autocast)."""
    dev = next(model.parameters()).device
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        v = model(lat.to(dev, torch.bfloat16), aux['rays'].to(dev, torch.bfloat16), aux['roles'].to(dev), aux['positions'].to(dev), times.to(dev),
                  aux['text'].to(dev, torch.bfloat16), aux['chunk_index'].to(dev), aux['c2w'].to(dev), aux['K'].to(dev), metadata=None)
    return v.float().cpu()


def default_aux_builder(stream, start, max_bank):
    import data
    return data.build_row(stream, random.Random(0), n_chunks=1, start=start, render_builder=None, render_p=0., max_bank=max_bank)


def one_step_recent(model, stream, row, s, gen, cfg, aux_builder=None, velocity=None):
    """One-step x0 of the four recent latents (start-4..start-1) at noise s from their recorded context. (1,48,4,h,w)."""
    aux = (aux_builder or default_aux_builder)(stream, row['start'] - 4, cfg.max_bank)
    x0 = aux['target'].float()
    n_recent = len(row['recent']); nctx = 1 + len(row['bank']) + n_recent
    if x0.shape[2] != 4 or not torch.equal(x0, row['context'][:, :, nctx - 4:nctx].float()):
        raise RuntimeError('auxiliary row does not reproduce the recent latents')
    eps = torch.randn(x0.shape, generator=gen); xs = (1.0 - s) * x0 + s * eps
    lat = torch.cat([aux['context'].float(), xs], 2); times = torch.cat([torch.zeros(aux['context'].shape[2]), torch.full((4,), float(s))])
    v = (velocity or dit_velocity)(model, aux, lat, times)
    if v.shape != xs.shape:
        raise RuntimeError('velocity shape mismatch')
    return xs - s * v


def corrupt_packed(row, stream, cfg, rng, model=None, aux_builder=None, velocity=None):
    """Returns (row', record). ``row`` is not modified; on-policy rows and first-call rows pass through."""
    sid = str(row['stream_id']); bank = list(row['bank']); recent = list(row['recent']); rec = dict(op='none', s=0.0, start=int(row['start']), render=bool(row.get('render')))
    if sid.startswith('OP'):
        rec['op'] = 'onpolicy'; return row, rec
    if recent == [0] or not recent:                                   # first-call layout: the recent slot is the anchor itself
        return row, rec
    nctx = 1 + len(bank) + len(recent); bank_slots = list(range(1, 1 + len(bank))); recent_slots = list(range(1 + len(bank), nctx))
    gen = torch.Generator().manual_seed(rng.getrandbits(62)); out = dict(row); s = 0.0
    if rng.random() < cfg.p_self_aug:
        s = rng.uniform(*cfg.s_self)
        if len(recent) == 4 and row['start'] >= 5 and stream is not None:
            x0 = one_step_recent(model, stream, row, s, gen, cfg, aux_builder, velocity)
            ctx = noise_slots(row['context'], bank_slots, s, gen); ctx[:, :, recent_slots] = x0.to(ctx.dtype); out['context'] = ctx; rec.update(op='self_aug', s=s)
        else:                                                         # fewer than four recent latents: no chunk to denoise, noise instead
            out['context'] = noise_slots(row['context'], bank_slots + recent_slots, s, gen); rec.update(op='self_aug_fallback_noise', s=s)
    elif rng.random() < cfg.p_noise:
        s = rng.uniform(0.0, cfg.s_max)
        out['context'] = noise_slots(row['context'], bank_slots + recent_slots, s, gen); rec.update(op='noise', s=s)
    if s > 0 and cfg.render_follows and row.get('render') and row.get('metadata') is not None:
        render_slots = list(range(nctx, nctx + 4))
        if out['context'].shape[2] < nctx + 4:
            raise RuntimeError('render row without render slots')
        out['context'] = noise_slots(out['context'], render_slots, s, gen)
        md = row['metadata'].clone(); md[:, :, nctx:nctx + 4] = md[:, :, nctx:nctx + 4] * (1.0 - s); out['metadata'] = md; rec['render_s'] = s
    return out, rec
