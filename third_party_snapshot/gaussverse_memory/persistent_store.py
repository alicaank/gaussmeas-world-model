"""Append-only spatial memory spanning a whole autoregressive rollout (Stage B).

Stage A conditions on a SLIDING window: six latents, two evicted per chunk, refilled
with the model's own output. After three autoregressive iterations no ground-truth
observation is left in the conditioning at all, so the warp warps hallucinations and a
plausible-but-wrong texture gets re-supplied to itself until it hardens. That is visible
in the measurements -- near-field matchability falls to 48% of the teacher-forced value
against 67% far -- and invisible to PSNR, which showed no band difference whatsoever.

Both reference systems avoid this by never forgetting:

* GEN3C's `Cache3D_Buffer.update_cache` **concatenates** new points onto the cache and
  evicts nothing, then renders every cached frame into each new view.
* Lyra 2.0 pairs a recent warp buffer (`buffer_depth_latest`) with a `Sparse3DCache`
  retrieval store that is appended to and never pruned, queried by projecting its points
  into the target camera.

This is that store for our contract. Two departures, both earned from our own numbers:

1. **The canonical frame is frozen at construction.** `scene_extent` derives (center,
   radius) from whichever frames are in hand, which is self-consistent inside one forward
   pass but not across a rollout -- a frame retrieved from forty steps ago would be
   normalized differently from the current window, and the two-sided correspondence
   branch's "same colour means same 3D point" invariant would be false exactly where it
   is needed. Freezing it is a prerequisite, not a refinement.

2. **Entries carry provenance and age.** Neither reference tracks whether a cached frame
   is real, because in their pipelines it does not change what they do. For us it does:
   depth for generated entries comes from VGGT run on invented pixels, so old generated
   content is both wrong and confidently wrong. Selection and fusion can prefer truth.

Usage::

    store = PersistentEvidenceStore(center, radius)
    for frame, depth, c2w, K in observed_context:
        store.add_observed(frame, depth, c2w, K)
    ...
    idxs, cov = store.select(tgt_c2w, tgt_K, k=6, uncovered=1.0 - recent["valid"],
                             skip_last_n=len(history_window))
    buffers = render_target_buffers(store.working_cache(idxs, device="cuda"),
                                    tgt_c2w, tgt_K, retrieval=False, norm=store.norm)
"""
from __future__ import annotations

import torch

from .keyframe_cache import KeyframeCache, OBSERVED, PROVISIONAL
from .point_retrieval import (downsample_region, downsampled_world_points,
                              projected_coverage)
from .retrieval import greedy_cover_holes, select_from_masks


class PersistentEvidenceStore:
    """Rollout-scale keyframe memory: append-only, provenance-aware, CPU-resident.

    Storage is CPU-side by default, as Lyra offloads its cache (`store_device="cpu"` when
    `args.offload`), because the store grows with the rollout while VRAM does not. Measured
    at 256x448: 2.28 MiB per entry -- rgb 1.31, depth 0.44, conf 0.44, point index 0.09 --
    so a hundred-step rollout is about 228 MiB. That is still a long-horizon concern rather
    than a working one, and `max_generated` stays None by default, but note we cost roughly
    4x what `Sparse3DCache` does per entry: it indexes depth/w2c/K only and keeps RGB in a
    separate dict for the few frames that need it, while we carry RGB on every entry because
    our fusion path warps appearance rather than looking it up by frame id.
    """

    def __init__(self, center: torch.Tensor, radius: torch.Tensor, *,
                 store_device: str = "cpu", max_generated: int | None = None,
                 downsample: int = 4, depth_tolerance: float = 0.02):
        if float(radius) <= 0:
            raise ValueError(f"canonical radius must be positive, got {float(radius)}")
        self._center = torch.as_tensor(center).detach().float().cpu().reshape(3)
        self._radius = torch.as_tensor(radius).detach().float().cpu().reshape(())
        self.store_device = str(store_device)
        self.max_generated = None if max_generated is None else int(max_generated)
        # Retrieval scores on a coarse point grid rather than by rendering each candidate
        # (`Sparse3DCache` does the same at downsample=4). `depth_tolerance` is a fraction
        # of the frozen scene radius, so occlusion filtering means the same thing on every
        # scene; 0 disables it.
        self.downsample = max(1, int(downsample))
        self.depth_tolerance = float(depth_tolerance)
        self.cache = KeyframeCache()
        self._points: list[torch.Tensor] = []   # (h,w,3) world points per entry
        self._points_valid: list[torch.Tensor] = []

    # -- frozen canonical frame ------------------------------------------------

    @property
    def norm(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The (center, radius) every chunk must share. Fixed for the store's lifetime."""
        return self._center.clone(), self._radius.clone()

    # -- insertion -------------------------------------------------------------

    def _add(self, rgb, depth, c2w, K, conf, provenance, age, fid):
        to = lambda t: None if t is None else t.detach().float().to(self.store_device)
        depth = to(depth)
        c2w, K = to(c2w), to(K)
        fid = self.cache.add(to(rgb), depth, c2w, K, conf=to(conf),
                             provenance=provenance, fid=fid, age=age)
        # index once at insertion; queries then cost one batched matmul, not C renders
        points = downsampled_world_points(depth, c2w, K, downsample=self.downsample)
        self._points.append(points)
        self._points_valid.append(depth[:, ::self.downsample, ::self.downsample][0] > 0)
        self._evict()
        return fid

    def add_observed(self, rgb, depth, c2w, K, conf=None, fid=None) -> int:
        """A real frame. Never evicted, always preferred when selection ties."""
        return self._add(rgb, depth, c2w, K, conf, OBSERVED, 0, fid)

    def add_generated(self, rgb, depth, c2w, K, conf=None, age=1, fid=None) -> int:
        """A frame the model produced. `age` counts steps back to the last real frame."""
        if int(age) < 1:
            raise ValueError("a generated frame is at least one step from an observation")
        return self._add(rgb, depth, c2w, K, conf, PROVISIONAL, int(age), fid)

    def _evict(self) -> None:
        """Drop the oldest generated frames past the cap. Observed frames are permanent."""
        if self.max_generated is None:
            return
        generated = [i for i, f in enumerate(self.cache.frames) if f.provenance != OBSERVED]
        surplus = len(generated) - self.max_generated
        if surplus <= 0:
            return
        drop = set(generated[:surplus])
        keep = [i for i in range(len(self.cache.frames)) if i not in drop]
        self.cache.frames = [self.cache.frames[i] for i in keep]
        self._points = [self._points[i] for i in keep]
        self._points_valid = [self._points_valid[i] for i in keep]

    # -- query -----------------------------------------------------------------

    def select(self, tgt_c2w, tgt_K, *, k: int = 6, uncovered=None, skip_last_n: int = 0,
               prefer_trusted: bool = True, training: bool = False, generator=None,
               target_hw=None):
        """Choose up to `k` entries covering `uncovered` (the recent warp's holes).

        Pass `uncovered = 1 - recent["valid"]` so the budget goes to what the history
        window cannot already explain, and `skip_last_n` for the entries that window
        holds. Returns (indices into this store, covered fraction of the requested region).

        Give `target_hw` to score by projecting the coarse point index -- one batched
        matmul over every candidate, which is what makes an unbounded store queryable.
        Omitting it falls back to splat-rendering each candidate: exact, and O(cache)
        renders per query, so it is for small caches and for checking the fast path.

        `training=True` switches to score-weighted stochastic selection, which is what
        Lyra does (`random=bool(is_training)`, `max_coverage=not is_training`). Training
        and inference must differ here: a model trained only on greedy-cover's choices
        learns that arbitration rule rather than how to use an arbitrary retrieved set,
        and at Stage B the retrieved set is the whole point.
        """
        if len(self.cache) == 0:
            return [], 0.0
        if target_hw is None:
            return self._select_by_render(tgt_c2w, tgt_K, k, uncovered, skip_last_n,
                                          prefer_trusted, training, generator)

        available = len(self.cache) - max(0, int(skip_last_n))
        if available <= 0:
            return [], 0.0
        masks, cells = projected_coverage(
            self._points[:available], self._points_valid[:available], tgt_c2w, tgt_K,
            target_hw, downsample=self.downsample,
            depth_threshold=self.depth_tolerance * float(self._radius),
        )
        if cells == 0:
            return [], 0.0
        if uncovered is None:
            target = torch.ones(cells, dtype=torch.bool, device=masks.device)
        else:
            target = downsample_region(uncovered.to(masks.device), target_hw,
                                       downsample=self.downsample)
        return select_from_masks(
            masks, target, k=k, frames=self.cache.frames[:available],
            prefer_trusted=prefer_trusted, sample=bool(training), generator=generator)

    def _select_by_render(self, tgt_c2w, tgt_K, k, uncovered, skip_last_n,
                          prefer_trusted, training, generator):
        """Fallback that splat-renders each candidate. Exact, but O(cache) renders."""
        device = tgt_c2w.device
        working = KeyframeCache()
        working.frames = [self._on(f, device) for f in self.cache.frames]
        return greedy_cover_holes(working, tgt_c2w, tgt_K, k=k, uncovered=uncovered,
                                  skip_last_n=skip_last_n, prefer_trusted=prefer_trusted,
                                  sample=bool(training), generator=generator)

    def working_cache(self, idxs, device=None) -> KeyframeCache:
        """The selected entries as a normal `KeyframeCache`, on `device`.

        Feed this to `render_target_buffers(..., retrieval=False, norm=store.norm)`:
        selection already happened here, against the holes rather than the whole frame,
        and the frozen norm has to win over the per-call `scene_extent` fallback.
        """
        out = KeyframeCache()
        out.frames = [self._on(self.cache.frames[i], device) for i in idxs]
        out._next_id = self.cache._next_id
        return out

    def _on(self, frame, device):
        if device is None or str(device) == self.store_device:
            return frame
        moved = type(frame)(frame.rgb.to(device), frame.depth.to(device),
                            frame.c2w.to(device), frame.K.to(device),
                            frame.conf.to(device), frame.fid, frame.provenance, frame.age)
        return moved

    # -- bookkeeping -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.cache)

    @property
    def counts(self) -> dict[str, int]:
        observed = sum(1 for f in self.cache.frames if f.provenance == OBSERVED)
        return {"total": len(self.cache), "observed": observed,
                "generated": len(self.cache) - observed}

    def snapshot(self):
        """State for revisit protocols -- the world as it was, not as it later drifted.

        The point index is snapshotted alongside the frames; restoring one without the
        other would leave selection scoring a cache that no longer exists.
        """
        return (self.cache.snapshot(), list(self._points), list(self._points_valid))

    def revert(self, snap) -> None:
        cache_snap, points, valid = snap
        self.cache.revert(cache_snap)
        self._points, self._points_valid = list(points), list(valid)
        if len(self._points) != len(self.cache.frames):
            raise RuntimeError(
                f"point index ({len(self._points)}) and frames ({len(self.cache.frames)}) "
                "disagree after revert")
