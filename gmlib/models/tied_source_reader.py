"""Tied native source reading — the corrected V3.1 reader (2026-08-18).

The transfer gap is structural: the previous reader owns fresh Q/K/V/out, so
generated-content competence trained in the temporal pathway never reaches it.
This module deletes the separate parameters instead of training them, with the
four review corrections applied:

1. **Inside the host block.** Wan's attention reads ``modulate(norm1(x))``, not
   raw ``x``.  Both the target rows and the archived source tokens pass through
   the block's OWN norm1/modulation before the block's OWN q/k/v/o weights touch
   them (weights used functionally — never re-registered under a second module).
2. **Joint target-row softmax.** Source keys are appended to the main keys for
   target query rows only: sources COMPETE with current temporal evidence and an
   irrelevant archive can receive ~zero probability — unlike a separate
   cross-attention, which forces a convex combination of source values.  The
   read calibration is a per-site logit offset ``beta`` (aggregate source
   probability is an explicit recipe choice), not an output
   gate: ``B_S = geometry + log(authority) + beta``.
3. **Timestep-matched source stream, honestly uncacheable.** The source tokens
   advance through blocks 0..site-1 as full native block forwards under the
   CURRENT timestep modulation — the best approximation to native insertion, and
   invalid to cache across sampling steps.  A cacheable canonical-timestep
   variant is a declared follow-up that must earn its keep in the diagnostic,
   not be assumed equivalent.
4. **RoPE is a choice, not a default.** ``neutral`` applies no rotation to
   source keys (projective correspondence lives in the geometry bias);
   ``source`` uses native spatial addresses (image-plane matching prior, valid
   for small motion).  Projected target-coordinate RoPE is the declared third
   variant.

New parameters per stack: one ``beta`` scalar per site and one source role
embedding.  Everything else belongs to the host block.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from gmlib.diffsynth.models.wan_video_dit import modulate, rope_apply

SOURCE_MASS_INIT = 0.01
ROPE_MODES = ("neutral", "source")


class TiedSourceReader(nn.Module):
    """Per-site logit offsets + source role; attention weights come from the host."""

    def __init__(self, blocks: tuple[int, ...] = (7, 15, 23), dim: int = 3072,
                 rope_mode: str = "neutral"):
        super().__init__()
        if rope_mode not in ROPE_MODES:
            raise ValueError(f"rope_mode must be one of {ROPE_MODES}")
        self.blocks = tuple(int(index) for index in blocks)
        self.rope_mode = rope_mode
        self.beta = nn.ParameterDict(
            {str(index): nn.Parameter(torch.zeros(())) for index in self.blocks})
        self.source_role = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta_initialized = False

    def site(self, index: int) -> bool:
        return int(index) in self.blocks

    @torch.no_grad()
    def initialize_beta(self, main_keys: int, source_keys: int,
                        mass: float = SOURCE_MASS_INIT) -> float:
        """Equal-logit calibration: aggregate source probability ~``mass`` at init.

        softmax over M main + S source keys at equal logit level gives the source
        block mass S*e^b/(M + S*e^b); solving for ``mass`` yields
        ``b = log(mass/(1-mass)) + log(M/S)``.  Realized mass on actual batches is
        what the diagnostic reports; this sets the starting point only.
        """
        if main_keys < 1 or source_keys < 1:
            raise ValueError("token counts must be positive")
        value = math.log(mass / (1.0 - mass)) + math.log(main_keys / source_keys)
        for parameter in self.beta.values():
            parameter.fill_(value)
        self.beta_initialized = True
        return value

    def source_column_bias(self, index: int, *, geometry_bias, key_authority,
                           batch: int, heads: int, query_tokens: int,
                           source_tokens: int, device, dtype) -> torch.Tensor:
        bias = torch.zeros(batch, heads, query_tokens, source_tokens,
                           device=device, dtype=dtype)
        if geometry_bias is not None:
            if geometry_bias.ndim == 3:
                geometry_bias = geometry_bias[:, None]
            if geometry_bias.shape[-2:] != (query_tokens, source_tokens):
                raise ValueError(
                    f"geometry bias {tuple(geometry_bias.shape)} does not match "
                    f"Q={query_tokens}, S={source_tokens}")
            bias = bias + geometry_bias.to(device=device, dtype=dtype)
        if key_authority is not None:
            if key_authority.shape != (batch, source_tokens):
                raise ValueError(
                    f"key authority must be {(batch, source_tokens)}")
            if not torch.isfinite(key_authority).all() or (
                    (key_authority < 0) | (key_authority > 1)).any():
                raise ValueError("key authority must be finite and in [0,1]")
            authority = torch.where(
                key_authority > 0,
                torch.log(key_authority.float().clamp_min(1e-6)),
                torch.full_like(key_authority.float(), float("-inf")),
            )
            bias = bias + authority[:, None, None, :].to(device=device,
                                                         dtype=dtype)
        return bias + self.beta[str(index)].to(dtype)


def tied_block_forward(block, reader: TiedSourceReader, index: int,
                       x: torch.Tensor, x_src: torch.Tensor, context,
                       t_mod: torch.Tensor, freqs, source_freqs, *,
                       n_target: int, geometry_bias=None, key_authority=None,
                       qk_bias=None, attn_logit_bias=None) -> torch.Tensor:
    """One host DiTBlock forward with a joint target-row read of archived sources.

    Replicates ``DiTBlock.forward`` exactly on the main sequence; target query
    rows additionally see the source keys/values (both computed with the block's
    own attention weights from the block's own normalized+modulated states).
    ``beta -> -inf`` recovers the unmodified block bit-for-bit up to kernel
    tolerance.
    """
    if t_mod.ndim != 3:
        raise ValueError("tied source reading requires a per-call timestep "
                         "(3D t_mod); per-token modulation is unsupported")
    if not 1 <= n_target <= x.shape[1]:
        raise ValueError(f"n_target must lie in [1, {x.shape[1]}]")
    modulation = (block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
                  + t_mod)
    (shift_msa, scale_msa, gate_msa,
     shift_mlp, scale_mlp, gate_mlp) = modulation.chunk(6, dim=1)
    attn = block.self_attn

    input_x = modulate(block.norm1(x), shift_msa, scale_msa)
    x_qk = input_x if qk_bias is None else input_x + qk_bias
    query = rope_apply(attn.norm_q(attn.q(x_qk)), freqs, attn.num_heads)
    key = rope_apply(attn.norm_k(attn.k(x_qk)), freqs, attn.num_heads)
    value = attn.v(input_x)

    input_src = modulate(block.norm1(x_src), shift_msa, scale_msa)
    src_kv_in = input_src + reader.source_role.to(dtype=input_src.dtype,
                                                  device=input_src.device)
    key_src = attn.norm_k(attn.k(src_kv_in))
    if reader.rope_mode == "source":
        key_src = rope_apply(key_src, source_freqs, attn.num_heads)
    value_src = attn.v(src_kv_in)

    main_out = attn.attn(query, key, value, attn_bias=attn_logit_bias)

    batch, heads = x.shape[0], attn.num_heads
    query_t = query[:, -n_target:]
    source_bias = reader.source_column_bias(
        index, geometry_bias=geometry_bias, key_authority=key_authority,
        batch=batch, heads=heads, query_tokens=n_target,
        source_tokens=key_src.shape[1], device=query.device, dtype=query.dtype)
    if attn_logit_bias is not None:
        main_part = attn_logit_bias[..., -n_target:, :].to(
            device=query.device, dtype=query.dtype)
        main_part = main_part.expand(batch, heads, n_target, key.shape[1])
    else:
        main_part = torch.zeros(batch, heads, n_target, key.shape[1],
                                device=query.device, dtype=query.dtype)
    joint_bias = torch.cat([main_part, source_bias], dim=-1)
    joint_t = attn.attn(query_t, torch.cat([key, key_src], dim=1),
                        torch.cat([value, value_src], dim=1),
                        attn_bias=joint_bias)

    attn_out = torch.cat([main_out[:, :-n_target], joint_t], dim=1)
    x = block.gate(x, gate_msa, attn.o(attn_out))
    x = x + block.cross_attn(block.norm3(x), context)
    input_x = modulate(block.norm2(x), shift_mlp, scale_mlp)
    x = block.gate(x, gate_mlp, block.ffn(input_x))
    return x
