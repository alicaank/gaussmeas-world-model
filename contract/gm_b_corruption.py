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
    amp_preserve: bool = False     # v2 (10 Nov): x + s/(1-s) eps instead of (1-s) x + s eps: the same SNR without the (1-s) attenuation of the signal
    near: bool = True              # False (10 Nov, render-only arm): bank / recent slots are left clean, only the render corruption (noise + metadata x (1-s)) is applied
    match_moments: bool = False    # v2: the one-step x0 of the recent latents is rescaled to the per-channel mean/std of the clean recent latents
    stat_only: bool = False        # 11 Nov: statistics-only corruption of the near latents (stat_perturb): content untouched, no one-step replacement, render untouched
    p_stat: float = 0.6
    stat_hf: tuple = (0.65, 1.6)   # high-frequency gain, log-uniform (pool context is .78-.87 of the recording; rollouts harden x2-3 in Laplacian variance)
    stat_std: tuple = (0.85, 1.2)  # per-channel spatial std scale (contrast), log-uniform, one draw shared by the channels + .03 jitter per channel
    stat_shift: float = 0.25       # per-channel mean shift, N(0, stat_shift) in units of the channel's spatial std (pool context |shift| .11-.18)

    def to_dict(self):
        return asdict(self)


def noise_slots(x, slots, s, gen, preserve=False):
    """FlowMatch law x_s = (1-s) x0 + s eps on the given frame slots of a (1,C,F,h,w) tensor; returns a new tensor.
    preserve=True: x0 + s/(1-s) eps, i.e. x_s / (1-s): identical signal-to-noise ratio, signal amplitude untouched (the context slots
    carry t=0, so the model cannot be told the attenuation; an unlabelled (1-s) teaches it to amplify its context)."""
    out = x.clone()
    if not slots or s <= 0:
        return out
    sub = out[:, :, slots].float(); eps = torch.randn(sub.shape, generator=gen)
    out[:, :, slots] = (sub + s / (1.0 - s) * eps if preserve else (1.0 - s) * sub + s * eps).to(out.dtype)
    return out


def stat_perturb(x, rng, cfg):
    """Statistics-only drift of context latents x (1,C,F,h,w): ONE draw per row applied to every given slot (rollout drift is coherent in time):
    high-frequency gain (x_low + g (x - x_low), x_low = 3x3 box blur), per-channel spatial std scale about the channel mean, per-channel mean shift.
    Structure / content is unchanged. Returns (x', record)."""
    import math
    x = x.float(); g = math.exp(rng.uniform(math.log(cfg.stat_hf[0]), math.log(cfg.stat_hf[1]))); c = math.exp(rng.uniform(math.log(cfg.stat_std[0]), math.log(cfg.stat_std[1])))
    C = x.shape[1]; gen = torch.Generator().manual_seed(rng.getrandbits(62))
    cj = c * (1 + .03 * torch.randn(C, generator=gen)).view(1, C, 1, 1, 1); sh = cfg.stat_shift * torch.randn(C, generator=gen).view(1, C, 1, 1, 1)
    b, _, f, h, w = x.shape; flat = x.permute(0, 2, 1, 3, 4).reshape(b * f, C, h, w)
    low = torch.nn.functional.avg_pool2d(torch.nn.functional.pad(flat, (1, 1, 1, 1), mode='replicate'), 3, 1).reshape(b, f, C, h, w).permute(0, 2, 1, 3, 4)
    y = low + g * (x - low); m = y.mean((3, 4), keepdim=True); sd = x.std((0, 2, 3, 4), keepdim=True)
    y = m + cj * (y - m) + sh * sd
    return y, dict(hf_gain=round(g, 3), std_scale=round(c, 3), shift_abs=round(float(sh.abs().mean()), 3))


def match_channel_moments(x, ref):
    """x rescaled per channel (dim 1) to ref's mean / std over (frames, h, w): global luminance / contrast of ref, structure of x."""
    d = (0, 2, 3, 4); mx, sx = x.mean(d, keepdim=True), x.std(d, keepdim=True).clamp_min(1e-4); mr, sr = ref.mean(d, keepdim=True), ref.std(d, keepdim=True)
    return (x - mx) / sx * sr + mr


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
    if cfg.stat_only:                                                 # content stays the recording; only its statistics drift (bank + recent, one coherent draw)
        if rng.random() < cfg.p_stat:
            slots = bank_slots + recent_slots; ctx = row['context'].clone(); y, info = stat_perturb(ctx[:, :, slots], rng, cfg); ctx[:, :, slots] = y.to(ctx.dtype)
            out['context'] = ctx; rec.update(op='stat', **info)
        return out, rec
    if rng.random() < cfg.p_self_aug:
        s = rng.uniform(*cfg.s_self)
        if not cfg.near:
            rec.update(op='render_only', s=s)
        elif len(recent) == 4 and row['start'] >= 5 and stream is not None:
            x0 = one_step_recent(model, stream, row, s, gen, cfg, aux_builder, velocity)
            clean = row['context'][:, :, recent_slots].float(); sd = lambda t: float(t.std((0, 2, 3, 4)).mean())
            hf = lambda t: float((4 * t[..., 1:-1, 1:-1] - t[..., :-2, 1:-1] - t[..., 2:, 1:-1] - t[..., 1:-1, :-2] - t[..., 1:-1, 2:]).pow(2).mean())   # 10 Nov evening: is the self-augmented history soft?
            rec.update(x0_std_ratio=round(sd(x0) / max(sd(clean), 1e-6), 4), x0_mean_shift=round(float((x0 - clean).mean()), 4), x0_hf_ratio=round(hf(x0.float()) / max(hf(clean), 1e-9), 4))   # amplitude of the self-augmented history vs clean
            if cfg.match_moments:
                x0 = match_channel_moments(x0, clean)
            ctx = noise_slots(row['context'], bank_slots, s, gen, cfg.amp_preserve); ctx[:, :, recent_slots] = x0.to(ctx.dtype); out['context'] = ctx; rec.update(op='self_aug', s=s)
        else:                                                         # fewer than four recent latents: no chunk to denoise, noise instead
            out['context'] = noise_slots(row['context'], bank_slots + recent_slots, s, gen, cfg.amp_preserve); rec.update(op='self_aug_fallback_noise', s=s)
    elif rng.random() < cfg.p_noise:
        s = rng.uniform(0.0, cfg.s_max)
        if cfg.near: out['context'] = noise_slots(row['context'], bank_slots + recent_slots, s, gen, cfg.amp_preserve); rec.update(op='noise', s=s)
        else: rec.update(op='render_only', s=s)
    if s > 0 and cfg.render_follows and row.get('render') and row.get('metadata') is not None:
        render_slots = list(range(nctx, nctx + 4))
        if out['context'].shape[2] < nctx + 4:
            raise RuntimeError('render row without render slots')
        noised = noise_slots(out['context'], render_slots, s, gen, cfg.amp_preserve)
        md = row['metadata'].clone(); seen = row.get('_seen')
        if seen is None:
            out['context'] = noised; md[:, :, nctx:nctx + 4] = md[:, :, nctx:nctx + 4] * (1.0 - s)
        else:                                                             # 11 Nov selective trust: anchor-seen cells stay clean (render and metadata)
            keep = seen.to(noised.dtype)[None, None]                         # (1,1,4,h,w)
            out['context'] = noised.clone(); out['context'][:, :, nctx:nctx + 4] = (keep * row['context'][:, :, nctx:nctx + 4].float() + (1 - keep) * noised[:, :, nctx:nctx + 4].float()).to(noised.dtype)
            md[:, :, nctx:nctx + 4] = md[:, :, nctx:nctx + 4] * (keep + (1 - keep) * (1.0 - s)); rec['seen'] = round(float(seen.float().mean()), 3)
        out['metadata'] = md; rec['render_s'] = s
    return out, rec
