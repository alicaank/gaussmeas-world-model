"""Memory contract (11 Nov plan): ONE bank-selection function for the training sampler and the inference roller.
  1 TRUE REVISITS: per target latent end the nearest stored latent (translation <= MAX_T gauge units = anchor median depths, rotation <= MAX_DEG) that the
    camera LEFT in between: some latent between it and now is farther from the target than LEAVE x its own distance and by at least DELTA (pose distance =
    translation + rotation / 60 deg). RELATIVE, so it works at any speed and any excursion size: on a slow forward path every older latent is farther than
    the newer ones (never a revisit; the plain nearest-pose rule returned such neighbours), on a way back the same-pose latent has distance ~0 and everything
    after it was farther. (First version used absolute LEAVE thresholds of .5 units / 30 deg: the distribution report showed 0 revisits on the test plans,
    whose whole excursion is below that.)
  2 FIRST-SEEN OWNERS (source_policy.bank_unseen): the oldest latents showing what the anchor does not show, whatever the recent latents cover
    (data.select_bank scored the coverage GAIN over anchor + recent and was empty on every slow or revisiting call).
Latents of the recent block are never candidates. Returns sorted latent indices (oldest first: render_own takes bank[0] as its extra source)."""
import numpy as np, torch
import source_policy as sp
MAX_T, MAX_DEG, LEAVE, DELTA = .25, 15., 2., .05


def _dist(C, T):
    dt = (C[:, :3, 3] - T[:3, 3]).norm(dim=1); cos = ((torch.einsum('nij,ij->n', C[:, :3, :3], T[:3, :3]) - 1) / 2).clamp(-1, 1)
    return dt, torch.rad2deg(torch.acos(cos))


def revisits(observed_c2w, requested_c2w, prefix_latents, recent=4, exclude=()):
    cand = [c for c in range(1, max(1, prefix_latents - recent)) if c not in exclude]
    if not cand: return []
    lat = torch.stack([observed_c2w[4 * c].float() for c in range(prefix_latents)]); out = []
    for t in (3, 7, 11, 15):
        T = requested_c2w[t].float(); dt, deg = _dist(lat, T); near = (dt <= MAX_T) & (deg <= MAX_DEG); d = dt + deg / 60.
        later = torch.flip(torch.cummax(torch.flip(d, [0]), 0).values, [0])       # later[c] = max distance over latents c .. prefix-1
        best, score = None, float('inf')
        for c in cand:
            if bool(near[c]) and c + 1 < prefix_latents and float(later[c + 1]) >= max(LEAVE * float(d[c]), float(d[c]) + DELTA):   # near now, and left afterwards
                if float(d[c]) < score: best, score = c, float(d[c])
        if best is not None: out.append(best)
    return sorted(set(out))


def bank_contract(observed_c2w, requested_c2w, K, prefix_latents, recent=4, max_refs=4, size=(480, 864), exclude=()):
    if max_refs == 0 or prefix_latents - recent <= 1: return []
    a = revisits(observed_c2w, requested_c2w, prefix_latents, recent, exclude)[:max_refs]
    if len(a) < max_refs:
        a = sorted(set(a) | set(sp.bank_unseen(observed_c2w, requested_c2w, K, prefix_latents, recent=recent, max_refs=max_refs - len(a), size=size, exclude=tuple(exclude) + tuple(a))))
    return a[:max_refs]


BANK_DROP = .2      # training only: share of continuation windows whose bank is emptied (inference: first-seen fill starts late and 17-33% of calls have none)


def window_drop(requested_c2w, p=None):
    """Deterministic per window (hash of the command poses), so the cache builder and the trainer agree without sharing state."""
    import hashlib
    p = BANK_DROP if p is None else p
    h = hashlib.sha1(np.ascontiguousarray(np.asarray(requested_c2w, np.float32)).tobytes()).digest()
    return int.from_bytes(h[:4], 'little') / 2 ** 32 < p


def bank_contract_train(observed_c2w, requested_c2w, K, prefix_latents, recent=4, max_refs=4, size=(480, 864), exclude=()):
    """bank_contract with the deterministic bank dropout: the function of the TRAINING sampler and of the cache v4 builder (inference uses bank_contract)."""
    if window_drop(requested_c2w): return []
    return bank_contract(observed_c2w, requested_c2w, K, prefix_latents, recent=recent, max_refs=max_refs, size=size, exclude=exclude)
