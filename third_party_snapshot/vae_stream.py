"""Streaming Wan VAE: chunked encode/decode that agrees with a whole-sequence pass.

`WanVideoVAE38` clears its temporal feature cache on entry to both `encode` and
`decode`, so the first frame of any call is treated as a fresh one-frame group.  That
makes a naive sliding-window re-encode wrong: encoding frames 0..8 and separately
re-encoding 4..8 yields latents for the *same* RGB frames 5..8 that differ by ~25% of
signal scale.  Autoregressive rollout with native video latents needs the encode and
decode to continue the previous chunk instead of restarting.

This module carries that state explicitly, the way Lyra 2.0 does (`_clone_vae_cache`,
`_stage_a_vae_cache_T-1/T-2`, and the CPU-offloaded `enc_feat_cache`/`dec_feat_cache`
in its autoregressive inference path).

Frame arithmetic, measured on the real VAE:

- first chunk:      ``L`` latents  <->  ``1 + 4(L-1)`` RGB frames
- resumed chunk:    ``L`` latents  <->  ``4L`` RGB frames

So a two-latent command chunk emits 8 RGB frames once the stream is running.

Nothing here changes default VAE behaviour: `VaeStream` is the only thing that passes
`keep_cache`/`resume`, and both default to False everywhere else.
"""
from __future__ import annotations

import torch
from torch import Tensor

FRAMES_PER_LATENT = 4


def clone_cache(cache: list) -> list:
    """Detached deep copy of one `_feat_map` / `_enc_feat_map` list.

    Entries are usually tensors or ``None``, but `CausalConv3d` also parks the string
    sentinel ``'Rep'`` there to mark a replication-padded slot.  Non-tensor entries are
    passed through untouched; dropping or coercing them corrupts the stream.
    """
    return [entry.detach().clone() if isinstance(entry, Tensor) else entry
            for entry in cache]


def cache_to(cache: list, device) -> list:
    """Move a cloned cache between devices — CPU offload between rollout steps."""
    return [entry.to(device) if isinstance(entry, Tensor) else entry for entry in cache]


def first_chunk_frames(latents: int) -> int:
    if latents < 1:
        raise ValueError("a chunk needs at least one latent frame")
    return 1 + FRAMES_PER_LATENT * (latents - 1)


def resumed_chunk_frames(latents: int) -> int:
    if latents < 1:
        raise ValueError("a chunk needs at least one latent frame")
    return FRAMES_PER_LATENT * latents


PRIME_FRAMES = 1 + FRAMES_PER_LATENT      # smallest fresh encode that leaves a primed stream


def encode_group(vae, frames: Tensor) -> Tensor:
    """``(B,3,5,H,W)`` -> ``(B,48,1,h,w)``: the native latent for the LAST FOUR frames.

    A four-frame group has no encoding on its own -- a fresh encode of four frames is
    not even a legal length, since ``T = 1 + 4(L-1)`` admits 1, 5, 9, ... So the group
    is defined relative to a fixed predecessor: encode five frames from scratch and keep
    the second latent, which is exactly the resumed-chunk latent for frames 1..4.

    Fixing the predecessor is what makes a *strided* history expressible in native
    latents at all: each group carries its own one-frame context rather than depending
    on wherever the previous group happened to end.
    """
    if frames.ndim != 5 or frames.shape[2] != PRIME_FRAMES:
        raise ValueError(f"a group needs (B,3,{PRIME_FRAMES},H,W), got {tuple(frames.shape)}")
    latents = VaeStream(vae).encode_chunk(frames)
    if latents.shape[2] != 2:
        raise RuntimeError(f"group encode produced {latents.shape[2]} latents, expected 2")
    return latents[:, :, 1:2]


@torch.no_grad()
def decode_group_windows(vae, previous: Tensor, latents: Tensor) -> list[Tensor]:
    """Inverse of `encode_group`: each latent's own five-frame window.

    ``previous`` is the latent immediately before ``latents[..., 0]``; each latent is
    then decoded against its own predecessor, reproducing the fixed-predecessor window
    `encode_group` built it from. Measured against ground truth on three held-out scenes
    that beats the alternatives -- 25.73 dB, vs 25.31 for one shared prime across the
    chunk and 24.46 for decoding the whole run as one stream: the longer the decode
    context, the further it drifts from how the latents were made.

    Each returned window is ``(B,3,5,H,W)``: the predecessor's last frame, then the
    group's own four.
    """
    windows, previous = [], previous
    for index in range(latents.shape[2]):
        current = latents[:, :, index:index + 1]
        windows.append(VaeStream(vae).decode_chunk(torch.cat([previous, current], dim=2)))
        previous = current
    return windows


@torch.no_grad()
def reencode_groups(vae, windows: list[Tensor]) -> Tensor:
    """``decode_group_windows`` output -> ``(B,48,L,h,w)``, each latent remade by the encoder.

    Lyra 2.0's autoregressive history is never raw denoiser output: every generated
    segment is decoded to pixels and pushed back through the encoder before it becomes
    history (`lyra2_ar_inference.py:1037`, `gen_chunk_reencoded`), and its self-augmented
    training step does the same (`lyra2_model.py:1342-1397`: decode, splice into the
    video, then `get_data_and_condition` re-encodes it). Passing the denoiser's latent
    straight through skips that projection, so whatever it emits off the encoder's
    manifold is free to compound across a rollout.

    Round-trip cost, measured on ground-truth latents over seven groups: 17.5% mean
    absolute deviation at correlation 0.982 -- the encoder is a real contraction here,
    not an identity, which is the entire point and also the entire risk.
    """
    return torch.cat([encode_group(vae, window[:, :, :PRIME_FRAMES])
                      for window in windows], dim=2)


def encode_primed_chunk(vae, prime: Tensor, chunk: Tensor) -> Tensor:
    """``prime`` (5 frames) then ``chunk`` (4L frames) -> ``(B,48,L,h,w)``.

    Used for the supervised target. The prime frames are the true frames immediately
    preceding the chunk, so the result depends only on the scene and the chunk position
    -- NOT on which observation history was sampled. That is what keeps the two paired
    histories sharing one target byte-for-byte: under a stateful encoder, continuing
    each history's own stream into the target would give two different targets and
    silently destroy the invariant the whole paired design rests on.
    """
    if prime.ndim != 5 or prime.shape[2] != PRIME_FRAMES:
        raise ValueError(f"prime must be (B,3,{PRIME_FRAMES},H,W), got {tuple(prime.shape)}")
    if chunk.ndim != 5 or chunk.shape[2] % FRAMES_PER_LATENT:
        raise ValueError(f"chunk frames must be a multiple of {FRAMES_PER_LATENT}")
    stream = VaeStream(vae)
    stream.encode_chunk(prime)                       # discarded; establishes the state
    latents = stream.encode_chunk(chunk)
    expected = chunk.shape[2] // FRAMES_PER_LATENT
    if latents.shape[2] != expected:
        raise RuntimeError(f"primed chunk gave {latents.shape[2]} latents, expected {expected}")
    return latents


class VaeStream:
    """One autoregressive stream's VAE state.

    ``encode_chunk`` / ``decode_chunk`` continue the stream; the first call of each
    starts it.  Encode and decode keep independent caches, matching the underlying
    model's separate ``_enc_feat_map`` and ``_feat_map``.
    """

    def __init__(self, vae, *, offload_device: str | torch.device | None = None):
        self.vae = vae
        self.core = vae.model
        self.offload_device = offload_device
        self._enc_cache: list | None = None
        self._dec_cache: list | None = None

    # -- state ---------------------------------------------------------------

    @property
    def encoding(self) -> bool:
        return self._enc_cache is not None

    @property
    def decoding(self) -> bool:
        return self._dec_cache is not None

    def reset(self) -> None:
        """Forget the stream. The next chunk starts a new first-frame group."""
        self._enc_cache = None
        self._dec_cache = None
        self.core.clear_cache()

    def snapshot(self) -> dict:
        """Cloned state, for revisit protocols that must restore an earlier stream."""
        return {
            "enc": None if self._enc_cache is None else clone_cache(self._enc_cache),
            "dec": None if self._dec_cache is None else clone_cache(self._dec_cache),
        }

    def restore(self, state: dict) -> None:
        self._enc_cache = None if state["enc"] is None else clone_cache(state["enc"])
        self._dec_cache = None if state["dec"] is None else clone_cache(state["dec"])

    # -- internals -----------------------------------------------------------

    def _load(self, attribute: str, cache: list | None) -> None:
        if cache is not None:
            device = next(self.core.parameters()).device
            setattr(self.core, attribute, cache_to(cache, device))

    def _store(self, attribute: str) -> list:
        cache = clone_cache(getattr(self.core, attribute))
        if self.offload_device is not None:
            cache = cache_to(cache, self.offload_device)
        return cache

    # -- stream --------------------------------------------------------------

    def encode_chunk(self, frames: Tensor) -> Tensor:
        """``(B,3,T,H,W)`` in [-1,1] -> ``(B,48,L,h,w)``, continuing the stream.

        The first chunk consumes ``1 + 4(L-1)`` frames; every later chunk consumes a
        multiple of four.
        """
        if frames.ndim != 5:
            raise ValueError(f"frames must be (B,3,T,H,W), got {tuple(frames.shape)}")
        resume = self.encoding
        self._load("_enc_feat_map", self._enc_cache)
        latents = self.core.encode(
            frames.to(next(self.core.parameters()).dtype),
            self.vae.scale, keep_cache=True, resume=resume,
        )
        self._enc_cache = self._store("_enc_feat_map")
        self.core.clear_cache()
        return latents

    def decode_chunk(self, latents: Tensor) -> Tensor:
        """``(B,48,L,h,w)`` -> ``(B,3,T,H,W)`` in [-1,1], continuing the stream."""
        if latents.ndim != 5:
            raise ValueError(f"latents must be (B,48,L,h,w), got {tuple(latents.shape)}")
        resume = self.decoding
        self._load("_feat_map", self._dec_cache)
        frames = self.core.decode(
            latents.to(next(self.core.parameters()).dtype),
            self.vae.scale, keep_cache=True, resume=resume,
        )
        self._dec_cache = self._store("_feat_map")
        self.core.clear_cache()
        expected = (resumed_chunk_frames if resume else first_chunk_frames)(
            latents.shape[2]
        )
        if frames.shape[2] != expected:
            raise RuntimeError(
                f"streaming decode produced {frames.shape[2]} frames, expected {expected}"
            )
        return frames.clamp_(-1, 1)
