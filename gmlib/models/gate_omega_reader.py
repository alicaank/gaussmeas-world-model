"""Zero-gated host-weight archive delta reader for GATE-ΩR.

Archive tokens reuse the host block's normalization and K/V/output weights.  The parent
main attention is computed exactly, and only the target-row ``joint - main`` delta is
multiplied by a zero-initialized scalar gate.  No host parameter is registered here.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from gmlib.diffsynth.models.wan_video_dit import modulate, rope_apply


OMEGA_SOURCE_MASS = 0.20
ROPE_MODES = ("neutral", "source")


class OmegaSourceReader(nn.Module):
    """Per-site zero gates and source-logit calibration; all projections are host-owned."""

    omega_delta = True

    def __init__(self, blocks: tuple[int, ...] = (7, 15, 23), dim: int = 3072,
                 rope_mode: str = "neutral", *, capture_mass: bool = False):
        super().__init__()
        if not blocks or len(set(blocks)) != len(blocks) or min(blocks) < 0:
            raise ValueError("reader blocks must be unique nonnegative indices")
        if rope_mode not in ROPE_MODES:
            raise ValueError(f"rope_mode must be one of {ROPE_MODES}")
        self.blocks = tuple(int(index) for index in blocks)
        self.rope_mode = rope_mode
        self.gates = nn.ParameterDict(
            {str(index): nn.Parameter(torch.zeros(())) for index in self.blocks})
        self.beta = nn.ParameterDict(
            {str(index): nn.Parameter(torch.zeros(())) for index in self.blocks})
        self.source_role = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta_initialized = False
        self.capture_mass = bool(capture_mass)
        self.last_source_mass: dict[int, float] = {}

    def site(self, index: int) -> bool:
        return int(index) in self.blocks

    @torch.no_grad()
    def initialize_beta(self, main_keys: int, source_keys: int,
                        mass: float = OMEGA_SOURCE_MASS) -> float:
        if main_keys < 1 or source_keys < 1:
            raise ValueError("token counts must be positive")
        if not 0 < mass < 1:
            raise ValueError("source mass must lie strictly between zero and one")
        value = math.log(mass / (1 - mass)) + math.log(main_keys / source_keys)
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
            if not torch.isfinite(geometry_bias).all():
                raise ValueError("geometry bias must be finite")
            bias = bias + geometry_bias.to(device=device, dtype=dtype)
        if key_authority is not None:
            if key_authority.shape != (batch, source_tokens):
                raise ValueError(f"key authority must be {(batch, source_tokens)}")
            if not torch.isfinite(key_authority).all() or bool(
                    ((key_authority < 0) | (key_authority > 1)).any()):
                raise ValueError("key authority must be finite and in [0,1]")
            authority = torch.where(
                key_authority > 0,
                torch.log(key_authority.float().clamp_min(1e-6)),
                torch.full_like(key_authority.float(), float("-inf")),
            )
            bias = bias + authority[:, None, None, :].to(device=device, dtype=dtype)
        return bias + self.beta[str(index)].to(dtype=dtype)

    @torch.no_grad()
    def record_realized_mass(self, index: int, query: torch.Tensor,
                             main_key: torch.Tensor, source_key: torch.Tensor,
                             joint_bias: torch.Tensor, heads: int) -> None:
        if not self.capture_mass:
            return
        batch, queries, width = query.shape
        head_dim = width // heads
        q = query.reshape(batch, queries, heads, head_dim).transpose(1, 2).float()
        key = torch.cat([main_key, source_key], dim=1)
        k = key.reshape(batch, key.shape[1], heads, head_dim).transpose(1, 2).float()
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
        logits = logits + joint_bias.float()
        mass = logits.softmax(dim=-1)[..., main_key.shape[1]:].sum(dim=-1).mean()
        self.last_source_mass[int(index)] = float(mass)


def omega_tied_block_forward(block, reader: OmegaSourceReader, index: int,
                             x: torch.Tensor, x_src: torch.Tensor, context,
                             t_mod: torch.Tensor, freqs, source_freqs, *,
                             n_target: int, geometry_bias=None, key_authority=None,
                             qk_bias=None, attn_logit_bias=None) -> torch.Tensor:
    """Host block forward with a zero-gated joint-minus-main target-row delta."""
    if t_mod.ndim != 3:
        raise ValueError("Omega reader requires one per-call timestep modulation")
    if not reader.site(index):
        raise ValueError(f"block {index} is not an Omega reader site")
    if not 1 <= n_target <= x.shape[1]:
        raise ValueError(f"n_target must lie in [1,{x.shape[1]}]")
    modulation = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
    (shift_msa, scale_msa, gate_msa,
     shift_mlp, scale_mlp, gate_mlp) = modulation.chunk(6, dim=1)
    attention = block.self_attn

    input_x = modulate(block.norm1(x), shift_msa, scale_msa)
    qk_input = input_x if qk_bias is None else input_x + qk_bias
    query = rope_apply(attention.norm_q(attention.q(qk_input)),
                       freqs, attention.num_heads)
    main_key = rope_apply(attention.norm_k(attention.k(qk_input)),
                          freqs, attention.num_heads)
    main_value = attention.v(input_x)

    input_source = modulate(block.norm1(x_src), shift_msa, scale_msa)
    source_input = input_source + reader.source_role.to(
        device=input_source.device, dtype=input_source.dtype)
    source_key = attention.norm_k(attention.k(source_input))
    if reader.rope_mode == "source":
        source_key = rope_apply(source_key, source_freqs, attention.num_heads)
    source_value = attention.v(source_input)

    main_attention = attention.attn(
        query, main_key, main_value, attn_bias=attn_logit_bias)
    target_query = query[:, -n_target:]
    batch, heads = x.shape[0], attention.num_heads
    source_bias = reader.source_column_bias(
        index, geometry_bias=geometry_bias, key_authority=key_authority,
        batch=batch, heads=heads, query_tokens=n_target,
        source_tokens=source_key.shape[1], device=query.device, dtype=query.dtype)
    if attn_logit_bias is None:
        main_bias = torch.zeros(batch, heads, n_target, main_key.shape[1],
                                device=query.device, dtype=query.dtype)
    else:
        main_bias = attn_logit_bias[..., -n_target:, :].to(
            device=query.device, dtype=query.dtype)
        main_bias = main_bias.expand(batch, heads, n_target, main_key.shape[1])
    joint_bias = torch.cat([main_bias, source_bias], dim=-1)
    joint_attention = attention.attn(
        target_query, torch.cat([main_key, source_key], dim=1),
        torch.cat([main_value, source_value], dim=1), attn_bias=joint_bias)
    reader.record_realized_mass(index, target_query, main_key, source_key,
                                joint_bias, heads)

    main_residual = attention.o(main_attention)
    delta = attention.o(joint_attention) - main_residual[:, -n_target:]
    scaled_delta = torch.tanh(reader.gates[str(index)]).to(delta.dtype) * delta
    # Sample at most once per block per audit step, including checkpoint
    # recomputation. Read-only diagnostics never modify attention or RNG.
    import os
    audit_step = int(os.environ.get("GM_AUDIT_STEP", "-1"))
    seen = getattr(reader, "_memory_audit_seen", {})
    if (os.environ.get("GM_MEMORY_AUDIT") and audit_step >= 0
            and audit_step % 10 == 0 and seen.get(index) != audit_step):
        with torch.no_grad():
            from gmlib.memory.audit import emit
            picks = torch.linspace(0, n_target-1, min(32,n_target),
                                   device=target_query.device).long()
            q = target_query[:,picks].float().reshape(batch,-1,heads,target_query.shape[-1]//heads).transpose(1,2)
            keys = torch.cat([main_key,source_key],1).float().reshape(batch,-1,heads,target_query.shape[-1]//heads).transpose(1,2)
            logits = (q @ keys.transpose(-1,-2)) / math.sqrt(target_query.shape[-1]//heads)
            weights = (logits + joint_bias[:,:,picks].float()).softmax(-1)
            mass = weights[:,:,:,-source_key.shape[1]:].sum(-1).mean()
            emit("reader_effect", block=int(index), sampled_queries=len(picks),
                 source_tokens=int(source_key.shape[1]),
                 source_attention_mass=float(mass),
                 gate_tanh=float(torch.tanh(reader.gates[str(index)])),
                 gated_delta_rms=float(scaled_delta.detach().float().square().mean().sqrt()),
                 main_residual_rms=float(main_residual[:,-n_target:].detach().float().square().mean().sqrt()),
                 authority_positive_tokens=(int((key_authority>0).sum())
                                            if key_authority is not None else None),
                 meaning="sampled internal contribution; not a quality-benefit claim")
        seen[index]=audit_step
        reader._memory_audit_seen=seen
    attention_residual = torch.cat([
        main_residual[:, :-n_target],
        main_residual[:, -n_target:] + scaled_delta,
    ], dim=1)
    hidden = block.gate(x, gate_msa, attention_residual)
    hidden = hidden + block.cross_attn(block.norm3(hidden), context)
    mlp_input = modulate(block.norm2(hidden), shift_mlp, scale_mlp)
    return block.gate(hidden, gate_mlp, block.ffn(mlp_input))
