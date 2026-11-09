"""Spatially-relevant keyframe retrieval (Lyra-style, time-independent).

Given a target camera, score each cached keyframe by how much of the target view
it can actually explain (valid pixels that land inside the target frustum with
positive depth), then greedily pick 4-6 keyframes that JOINTLY maximize target
coverage — set-cover, not top-K-by-score, so redundant near-duplicate views are
not all chosen. Generalizes GEN3C's Cache3D_BufferSelector overlap score.
"""
from typing import List
import torch

from .geometry import render_sources_to_target
from .keyframe_cache import OBSERVED


def _target_mask_for(frame, tgt_c2w, tgt_K):
    """Coverage mask (1,1,H,W) a single keyframe produces at the target pose."""
    out = render_sources_to_target(
        frame.rgb[None], frame.depth[None], frame.c2w[None], frame.K[None], tgt_c2w, tgt_K,
        src_conf=frame.conf[None],
    )
    return out["mask"], out["conf"]


def score_keyframes(cache, tgt_c2w, tgt_K):
    """Per-keyframe target coverage fraction (weighted by confidence)."""
    scores = []
    masks = []
    for f in cache.frames:
        m, c = _target_mask_for(f, tgt_c2w, tgt_K)
        masks.append(m)
        scores.append((m * c).mean().item())
    return scores, masks


def greedy_cover(cache, tgt_c2w, tgt_K, k=6, min_gain=1e-3):
    """Greedy set-cover selection of up to `k` keyframes maximizing joint coverage.

    Returns (selected_indices, final_coverage_fraction). Stops early when the next
    best frame adds less than `min_gain` new coverage.
    """
    if len(cache) == 0:
        return [], 0.0
    _, masks = score_keyframes(cache, tgt_c2w, tgt_K)
    masks = [(m > 0) for m in masks]  # binary coverage per keyframe
    H_W = masks[0].numel()
    covered = torch.zeros_like(masks[0])
    chosen: List[int] = []
    remaining = set(range(len(masks)))
    for _ in range(min(k, len(masks))):
        best_i, best_gain = -1, 0.0
        for i in remaining:
            gain = ((masks[i] & ~covered).float().sum() / H_W).item()
            if gain > best_gain:
                best_gain, best_i = gain, i
        if best_i < 0 or best_gain < min_gain:
            break
        chosen.append(best_i)
        remaining.discard(best_i)
        covered = covered | masks[best_i]
    coverage = covered.float().mean().item()
    return chosen, coverage


def greedy_cover_holes(cache, tgt_c2w, tgt_K, k=6, *, uncovered=None, skip_last_n=0,
                       prefer_trusted=True, min_gain=1e-3, tie_ratio=0.9,
                       sample=False, generator=None):
    """Set-cover selection that spends its budget on the pixels nothing else explains.

    `greedy_cover` maximizes coverage of the WHOLE target. In an autoregressive rollout
    the recent history window already explains most of the frame, so that objective keeps
    re-selecting near-duplicates of what is already known and spends nothing on the
    disocclusion holes -- which is where the error actually lives (measured: 2.04x error
    inside warp holes, and near-field matchability falling to 48% against 67% far). Pass
    the recent warp's `valid` plane and selection is scored against its complement.

    Lyra 2.0 does the same thing inside `Sparse3DCache.retrieve(max_coverage=True)`: it
    seeds `covered` with the temporally closest frame's mask "because its warping is
    already included in the network condition", excludes that frame, and set-covers the
    rest.

    Note this is only Lyra's INFERENCE-side exclusion, and only in the greedy branch --
    its `random=True` training branch does neither. Lyra keeps recent frames out of
    training retrieval somewhere else entirely: at cache CONSTRUCTION, where
    `spatial_memory_skip_recent=16` refuses to insert any frame within +/-16 of the
    generation window (and `spatial_memory_stride=8` subsamples the rest), so retrieval is
    forced to be genuinely long-range instead of covering the target with a temporal
    neighbour. `spatial_cache_skip_last_n` really is 0 everywhere and is not the mechanism.

    Ours is kept because our history window is several latents deep rather than one, so
    there is more than a single entry to hold out at query time. The construction-time
    equivalent of `skip_recent` is a Stage-B concern and is not implemented here.

    `prefer_trusted` breaks near-ties toward observed frames and, among those, the ones
    with the smallest `age`. Ties are common -- two views of the same wall cover the same
    holes -- and resolving them toward real observations is the whole point of tracking
    provenance. A candidate qualifies when its gain is at least `tie_ratio` of the best.

    Returns (selected_indices, covered_fraction_of_the_requested_region).
    """
    if len(cache) == 0:
        return [], 0.0
    available = len(cache) - max(0, int(skip_last_n))
    if available <= 0:
        return [], 0.0

    _, rendered = score_keyframes(cache, tgt_c2w, tgt_K)
    masks = torch.stack([(m > 0).reshape(-1) for m in rendered[:available]])
    if uncovered is None:
        target = torch.ones(masks.shape[1], dtype=torch.bool, device=masks.device)
    else:
        target = (uncovered.to(masks.device).reshape(-1) > 0.5)
    return select_from_masks(masks, target, k=k, frames=cache.frames[:available],
                             prefer_trusted=prefer_trusted, min_gain=min_gain,
                             tie_ratio=tie_ratio, sample=sample, generator=generator)


def select_from_masks(masks, target, *, k=6, frames=None, prefer_trusted=True,
                      min_gain=1e-3, tie_ratio=0.9, sample=False, generator=None):
    """Pick up to `k` rows of `masks` (C, cells) covering `target` (cells,).

    Shared by both scorers so the render path and the projected-point path cannot drift
    apart in their selection rule -- only in how the coverage masks were obtained.
    """
    if masks.numel() == 0 or masks.shape[0] == 0:
        return [], 0.0
    available = masks.shape[0]
    total = target.float().sum().clamp(min=1.0)

    def trust(i):
        frame = frames[i] if frames is not None else None
        # observed first, then least-aged, then oldest fid for determinism
        return (getattr(frame, "provenance", "") != OBSERVED,
                int(getattr(frame, "age", 0)), int(getattr(frame, "fid", i)))

    if sample:
        # Lyra retrieves stochastically while TRAINING (`random=bool(is_training)`) and
        # greedily at inference (`max_coverage=not is_training`). Without this the model
        # only ever sees the one set greedy cover happens to pick, and learns that
        # particular arbitration rather than how to use whatever it is handed.
        #
        # Weights are DISJOINT coverage, not raw coverage. Lyra's random branch scores by
        # z-buffer ownership -- each target cell is credited to exactly one candidate via
        # `amin` over candidate index -- so five near-duplicate views of one wall split
        # that wall's score instead of each claiming all of it. Scoring `(masks & target)`
        # directly gives every duplicate full credit and the multinomial spends the budget
        # on the same surface: measured on 5 duplicates + 3 unique views at k=3, raw
        # coverage picked >1 duplicate in 92.7% of draws and covered 0.598, against 15.3%
        # and 0.790 for ownership. The greedy branch below needs no such fix because
        # set-cover subtracts `covered` as it goes.
        #
        # Lyra breaks ownership ties by lowest candidate index, which in an append-only
        # cache means the oldest entry. We break by `trust` when `prefer_trusted`, which
        # resolves toward real observations first -- the same intent, with provenance.
        hit = masks & target
        order = sorted(range(available), key=trust) if prefer_trusted else list(range(available))
        owner = torch.full((hit.shape[1],), -1, dtype=torch.long, device=hit.device)
        for i in reversed(order):        # worst first, so the preferred owner wins
            owner[hit[i]] = i
        scores = torch.bincount(owner[owner >= 0], minlength=available).float().cpu()
        if float(scores.max()) <= 0:
            return [], 0.0
        weights = scores.clamp(min=0.0) + float(scores.max()) * 0.02
        picked = torch.multinomial(weights, num_samples=min(k, available),
                                   replacement=False, generator=generator).tolist()
        covered = torch.zeros_like(target)
        for i in picked:
            covered = covered | (masks[i] & target)
        return [int(i) for i in picked], (covered.float().sum() / total).item()

    covered = torch.zeros_like(target)
    chosen: List[int] = []
    remaining = set(range(available))
    for _ in range(min(k, available)):
        gains = {i: ((masks[i] & target & ~covered).float().sum() / total).item()
                 for i in remaining}
        best_gain = max(gains.values(), default=0.0)
        if best_gain < min_gain:
            break
        contenders = [i for i, g in gains.items() if g >= best_gain * tie_ratio]
        best_i = min(contenders, key=trust) if prefer_trusted else max(gains, key=gains.get)
        chosen.append(best_i)
        remaining.discard(best_i)
        covered = covered | (masks[best_i] & target)
    return chosen, (covered.float().sum() / total).item()
