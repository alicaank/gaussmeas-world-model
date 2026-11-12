"""Which frames feed the ReSplat render memory (10 Nov, user: generated regions are re-generated every call instead of being stored).
Default (render864_rows.OwnRender864.render_own): anchor, latents s-1 and s-2 (BOTH from the immediately preceding call) and the best bank
view -> content generated earlier is rendered from its most recent re-generation (a photocopy chain); nothing is stored once.
  mode 'default'    unchanged, only logged (source latent ids + their age in calls)
  mode 'firstseen'  write-once memory: anchor + up to two KEYFRAMES + the last latent s-1. Every target frustum sample (data.select_bank's test, no
                    occlusion) outside the anchor is assigned to the OLDEST keyframe candidate (every 5th call end) that shows it;
                    keyframes = the two candidates owning the most samples (>= 3% each), so a region is rendered from the frame that first
                    showed it, not from its latest re-generation. Fewer than two keyframes -> filled with s-2 as in the default.
install(mode, log) patches OwnRender864.render_own. Log: one json line per render call."""
import gm_env  # machine paths: gm_env.py at the repository root (scripts/env.sh puts it on PYTHONPATH)
import json
import numpy as np, torch

STATE = dict(mode='default', log=None, gain=.03, grid=5)


def frustum_samples(command, K, size):
    h, w = size
    yy, xx = torch.meshgrid(torch.linspace(.1 * h, .9 * h, 6), torch.linspace(.1 * w, .9 * w, 8), indexing='ij')
    ray = torch.linalg.inv(K.float()) @ torch.stack([xx, yy, torch.ones_like(xx)]).reshape(3, -1); world = []
    for target in command[[3, 7, 11, 15]]:
        for d in (.5, 1., 2., 4.): world.append(target[:3, :3] @ (ray * d) + target[:3, 3, None])
    return torch.cat(world, 1)


def covers(c2w, world, K, size):
    h, w = size; cam = torch.linalg.inv(c2w); xyz = cam[:3, :3] @ world + cam[:3, 3, None]; uv = K.float() @ xyz; xy = uv[:2] / xyz[2].clamp_min(1e-8)
    return (xyz[2] > .1) & (xy[0] >= 0) & (xy[0] < w) & (xy[1] >= 0) & (xy[1] < h)


def choose(poses, s, command, K, size=(480, 864), gain=.03, max_keys=2, grid=5):
    """poses: RGB-rate list of c2w (anchor gauge); s = index of the first new latent. Returns (ids, record)."""
    P = lambda i: torch.as_tensor(np.asarray(poses[0 if i == 0 else 4 * i], np.float64)).float()
    K = torch.as_tensor(np.asarray(K, np.float64)).float(); world = frustum_samples(torch.as_tensor(np.asarray(command, np.float64)).float(), K, size)
    covered = covers(P(0), world, K, size); anchor_cov = float(covered.float().mean()); owner = {}
    for c in range(4 * grid, s - 1, 4 * grid):                             # every grid-th call end, oldest first (a new strip per call is 1-2% of the
                                                                           # view: too thin to own a source slot each; VRAM allows 4 sources); s-1 is always a source
        first = covers(P(c), world, K, size) & ~covered                    # samples whose FIRST appearance is frame c
        if first.any(): owner[c] = float(first.float().mean()); covered |= first
    top = sorted([c for c in owner if owner[c] >= gain], key=lambda c: -owner[c])[:max_keys]      # the frames that first showed the largest parts of the target
    keys = sorted(top); gains = [round(owner[c], 3) for c in keys]
    ids = [0, *keys, s - 1]
    if len(keys) < max_keys and s - 2 > 0: ids.append(s - 2)
    ids = list(dict.fromkeys(ids))[:4]
    return ids, dict(anchor_cov=round(anchor_cov, 3), keys=keys, gains=gains, covered=round(float(covered.float().mean()), 3))


def choose_oldest(poses, s, command, K, size=(480, 864), min_unseen=.03, frac=.5):
    """Source swap (critique 10 Nov): s-2 sees almost what s-1 sees; give its slot to the OLDEST latent (1..s-3) that covers the
    target's generated cells (frustum samples the anchor does not see) at least `frac` as well as the best candidate. Returns (id or None, record)."""
    P = lambda i: torch.as_tensor(np.asarray(poses[0 if i == 0 else 4 * i], np.float64)).float()
    K = torch.as_tensor(np.asarray(K, np.float64)).float(); world = frustum_samples(torch.as_tensor(np.asarray(command, np.float64)).float(), K, size)
    unseen = ~covers(P(0), world, K, size); u = float(unseen.float().mean())
    if u < min_unseen or s - 3 < 1: return None, dict(unseen=round(u, 3))
    cov = {c: float((covers(P(c), world, K, size) & unseen).float().sum() / unseen.float().sum()) for c in range(1, s - 2)}
    best = max(cov.values())
    if best <= 0: return None, dict(unseen=round(u, 3), best=0.)
    c = min(c for c, v in cov.items() if v >= frac * best)
    return c, dict(unseen=round(u, 3), best=round(best, 3), oldest=int(c), oldest_cov=round(cov[c], 3))


def bank_unseen(observed_c2w, requested_c2w, K, prefix_latents, recent=4, max_refs=6, size=(256, 448), exclude=()):
    """Bank selection for generated content (critique 10 Nov): data.select_bank discounts what the recent latents cover, so under forward
    motion the bank stays empty and generated regions are only present as their latest re-generation. Here: every target frustum sample
    the ANCHOR does not see gets its first-appearance latent (oldest candidate 1..prefix-recent-1 showing it); the samples are split into
    max_refs equal bands by first-appearance time and the latent closing each band is selected (the oldest latent that shows the whole
    band under forward motion). The bank latents are stored, never re-encoded: the lossless record of generated content."""
    candidates = [c for c in range(1, max(1, prefix_latents - recent)) if c not in exclude]
    if not candidates or max_refs == 0: return []
    K = K.float(); world = frustum_samples(requested_c2w.float(), K, size); unseen = ~covers(observed_c2w[0].float(), world, K, size)
    first = torch.full((world.shape[1],), -1, dtype=torch.long)
    for c in candidates:                                                                # oldest first
        v = covers(observed_c2w[4 * c].float(), world, K, size) & unseen & (first < 0); first[v] = c
    f = first[first > 0].sort().values
    if len(f) < .03 * world.shape[1]: return []                                         # the anchor still shows (almost) everything
    return sorted({int(f[min(len(f) - 1, int(round(j * len(f) / max_refs)) - 1)]) for j in range(1, max_refs + 1)})


def bank_revisit(observed_c2w, requested_c2w, K, prefix_latents, recent=4, max_refs=6, size=(256, 448), exclude=(), max_t=.25, max_deg=15., guard=8, fill=True):
    """Revisit retrieval (11 Nov, user: no model is consistent): data.select_bank scores candidates by coverage GAIN over anchor + recent, so when the
    recent latents already cover the view (any revisit, e.g. the way back of an out-and-back plan) the gain is 0 and the bank is EMPTY - the frames
    generated at these very poses are never looked up. Here: for each of the four target latent ends the stored latent (not among the last `guard`)
    with the NEAREST camera (translation in gauge units = anchor median depths, + rotation / 60 deg), accepted within max_t and max_deg; ties -> oldest.
    Remaining slots are filled by the stock rule. Sorted (oldest first: render_own takes bank[0] as its extra source)."""
    import data
    candidates = [c for c in range(1, max(1, prefix_latents - max(recent, guard))) if c not in exclude]
    if max_refs == 0: return []
    out = []
    if candidates:
        C = torch.stack([observed_c2w[4 * c].float() for c in candidates])
        for t in (3, 7, 11, 15):
            T = requested_c2w[t].float(); dt = (C[:, :3, 3] - T[:3, 3]).norm(dim=1)
            cos = ((torch.einsum('nij,ij->n', C[:, :3, :3], T[:3, :3]) - 1) / 2).clamp(-1, 1); deg = torch.rad2deg(torch.acos(cos))
            ok = (dt <= max_t) & (deg <= max_deg)
            if ok.any():
                score = dt + deg / 60.; score[~ok] = float('inf'); out.append(candidates[int(score.argmin())])
    out = sorted(set(out))[:max_refs]
    if fill and len(out) < max_refs:
        out = sorted(set(out) | set(data.select_bank(observed_c2w, requested_c2w, K, prefix_latents, recent=recent, max_refs=max_refs - len(out), size=size, exclude=tuple(exclude) + tuple(out))))
    return out


def bank_memory(observed_c2w, requested_c2w, K, prefix_latents, recent=4, max_refs=6, size=(256, 448), exclude=()):
    """Long-term memory for forward motion AND revisits (11 Nov, user: the scene also changes while going forward; the stock bank is empty in 29/29 forward
    calls too): same-pose latents first (bank_revisit without the stock fill), then the FIRST-SEEN owners of what the anchor does not show (bank_unseen)."""
    a = bank_revisit(observed_c2w, requested_c2w, K, prefix_latents, recent=recent, max_refs=max_refs, size=size, exclude=exclude, fill=False, guard=16)
    if len(a) < max_refs:
        a = sorted(set(a) | set(bank_unseen(observed_c2w, requested_c2w, K, prefix_latents, recent=recent, max_refs=max_refs - len(a), size=size, exclude=tuple(exclude) + tuple(a))))
    return a[:max_refs]


MPU = None                                                                 # metres per gauge unit of the running scene (set by run_arm_ckpt): Memorizon's rule works in metres / 4


def bank_memorizon(observed_c2w, requested_c2w, K, prefix_latents, recent=4, max_refs=6, size=(256, 448), exclude=()):
    """Memorizon's OWN retrieval (baselines/memorizon/memorizon/retrieval.py, the rule the base model was trained with): top-k stored latents by
    fov_overlap(chunk-end camera, candidate) - 0.2 (|dt| + 4 angle), poses in metres / 4, frustum depth band 2-60 m. Unlike data.select_bank (coverage
    GAIN over anchor + recent, empty when the recent latents cover the view) it always returns k frames: the nearest well-overlapping past views."""
    import importlib.util, sys
    if 'mz_retrieval' not in sys.modules:
        sp = importlib.util.spec_from_file_location('mz_retrieval', (gm_env.MEMORIZON + '/memorizon/retrieval.py')); m = importlib.util.module_from_spec(sp); sys.modules['mz_retrieval'] = m; sp.loader.exec_module(m)
    mz = sys.modules['mz_retrieval']
    cand = [c for c in range(1, max(1, prefix_latents - recent)) if c not in exclude]
    if not cand or max_refs == 0: return []
    if MPU is None: raise RuntimeError('source_policy.MPU not set')
    sc = float(MPU) / 4.; Kn = np.asarray(K, np.float64); fr = mz.Frustum.from_intrinsics((Kn[0, 0], Kn[1, 1], Kn[0, 2], Kn[1, 2]))
    conv = lambda P: (lambda A: (A.__setitem__((Ellipsis, slice(0, 3), 3), A[..., :3, 3] * sc), A)[1])(np.asarray(P, np.float64).copy())
    C = conv(torch.stack([observed_c2w[4 * c] for c in cand]).numpy()); q = conv(requested_c2w[15].numpy())
    return sorted(int(cand[i]) for i in mz.topk(q, C, fr, min(max_refs, len(cand))))


def install_bank(mode='unseen'):
    import closed_loop
    if mode == 'contract':
        import memory_contract; closed_loop.select_bank = memory_contract.bank_contract
    if mode == 'memorizon': closed_loop.select_bank = bank_memorizon
    if mode == 'memory': closed_loop.select_bank = bank_memory
    if mode == 'revisit': closed_loop.select_bank = bank_revisit
    if mode == 'unseen': closed_loop.select_bank = bank_unseen
    if mode == 'none': closed_loop.select_bank = lambda *args, **kw: []                       # 12 Nov: bank-off test (are the stray playground objects recalled?)


def install_ctx_noise(level=.2):
    """Inference context corrupted as in GM-B's training (critique 10 Nov: the only candidate aimed at the snap of fixed-source cells): from
    the second call on, bank / recent / render slots get (1-s) x + s eps (fixed eps per call) and the render (valid, conf) x (1-s), i.e.
    gm_b_corruption's 'noise' operation at a fixed level; the anchor stays clean."""
    from closed_loop import LoopRoller
    inner = LoopRoller.pack

    def pack(self, command, render_override=None):
        pk = inner(self, command, render_override); n = len(self.latents)
        if n > 1:
            d = dict(pk['dev']); ctx = d['context'].clone(); n1 = 1 + len(pk['bank']) + len(pk['recent'])
            eps = torch.randn(ctx[:, :, 1:].shape, generator=torch.Generator().manual_seed(9000 + n)).to(ctx.device, ctx.dtype)
            ctx[:, :, 1:] = ((1 - level) * ctx[:, :, 1:].float() + level * eps.float()).to(ctx.dtype); d['context'] = ctx
            if d.get('md') is not None:
                md = d['md'].clone(); md[:, :, n1:n1 + 4] = md[:, :, n1:n1 + 4] * (1 - level); d['md'] = md
            pk = dict(pk, dev=d)
        return pk

    LoopRoller.pack = pack


def install_moment(strength=1., log=None, ref='prev'):
    """Latent moment continuity at the call boundary (10 Nov: GM-B's call-start snap is a low-frequency step in latent space between the
    last latent of a call and the first of the next). After sampling a chunk, one per-channel affine map is applied to all four latents so
    that the FIRST new latent has the per-channel spatial mean / std of the last history latent; the chunk's own evolution is kept."""
    from closed_loop import LoopRoller
    inner = LoopRoller.denoise

    def denoise(self, x, times, pk, text, taps=()):
        x, x0s = inner(self, x, times, pk, text, taps)
        if len(self.latents) > 1 and len(times) > 8 and not taps:                      # full sampling pass of a continuation call
            prev = self.latents[-1].float().cpu(); d = (0, 2, 3, 4); x = x.float()
            mp, sp = prev.mean(d, keepdim=True), prev.std(d, keepdim=True)
            if ref != 'prev':                                                          # 11 Nov critique: prev carries the drift from call to call; the anchor is the only clean statistics reference
                an = self.latents[0].float().cpu(); ma, sa = an.mean(d, keepdim=True), an.std(d, keepdim=True)   # 'anchor' = anchor moments; 'mix' = half way (the view content does change)
                w = 1. if ref == 'anchor' else .5; mp = (1 - w) * mp + w * ma; sp = sp ** (1 - w) * sa ** w
            m1, s1 = x[:, :, :1].mean(d, keepdim=True), x[:, :, :1].std(d, keepdim=True).clamp_min(1e-4)
            scale = (sp / s1) ** strength; y = (x - m1) * scale + m1 + strength * (mp - m1)
            if log:
                with open(log, 'a') as f: f.write(json.dumps(dict(call=(len(self.latents) - 1) // 4 + 1, mean_step=round(float((m1 - mp).abs().mean()), 4), std_ratio=round(float((s1 / sp).mean()), 4))) + '\n')
            x = y
        return x, x0s

    LoopRoller.denoise = denoise


def install(mode='default', log=None, gain=.03, grid=5):
    from render864_rows import OwnRender864
    STATE.update(mode=mode, log=str(log) if log else None, gain=float(gain), grid=int(grid))

    def render_own(self, frames_by_latent, poses, bank, s, K, mpu, target_c2w, gauge_id):
        default = list(dict.fromkeys([0, s - 1, max(0, s - 2), *bank[:1]]))[:self.MAX_SOURCES]; rec = dict(s=int(s), call=(int(s) - 1) // 4 + 1, default=default, bank=[int(b) for b in bank])
        ids = default
        if STATE['mode'] == 'firstseen' and s > 5:
            ids, extra = choose(poses, s, target_c2w, K, gain=STATE['gain'], grid=STATE['grid']); rec.update(extra)
        if STATE['mode'].startswith(('recent', 'spread')):                                          # 10 Nov evening (user): 8 / 16 source views (multi_view.source_ids);
            import multi_view                                                                       # the anchor copies that fill the list are added by frontier (pad_to)
            m = STATE['mode'].rstrip('0123456789'); ids = list(dict.fromkeys(multi_view.source_ids(int(s), int(STATE['mode'][len(m):]), m)))
        elif STATE['mode'] == 'oldest' and s > 5:
            c, extra = choose_oldest(poses, s, target_c2w, K); rec.update(extra)
            if c is not None: ids = list(dict.fromkeys([0, s - 1, c, *bank[:1]]))[:self.MAX_SOURCES]
        rec.update(ids=[int(i) for i in ids], age_calls=[((int(s) - 1) - int(i) + 3) // 4 if i else None for i in ids])   # how many calls ago each source was generated
        if STATE['log']:
            with open(STATE['log'], 'a') as f: f.write(json.dumps(rec) + '\n')
        frames = np.stack([frames_by_latent[i] for i in ids]); c2w = np.stack([np.asarray(poses[0 if i == 0 else 4 * i], np.float64) for i in ids])
        return self.render(frames, c2w, np.asarray(K, np.float64), mpu, np.asarray(target_c2w, np.float64), gauge_id, boundary=4 * (s - 1))

    OwnRender864.render_own = render_own
    return STATE
