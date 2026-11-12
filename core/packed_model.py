"""Packed chunk-causal teacher on the stock Wan2.2-TI2V-5B DiT.

Sequence: [anchor | bank (<=6, pose-retrieved past latents) | recent (<=4 latest latents) | N chunks x 4 target latents].
Context slots carry t=0 and attend only to context. Chunk j (its own noise level u_j, told to the model) attends to
context and chunks <= j, so chunk j>0 continues from a NOISY predecessor whose level is labelled. Loss on all chunks.
Camera: additive Plucker rays at the patch embedding (as the 10,250 lineage) + a PRoPE side branch after every block.
RoPE temporal positions are fixed by slot (anchor 0, bank 1..6, recent 7..10, chunk j at 11+4j), not by true time;
the true cameras carry time-free geometry. No render slots in stage A (metadata channels stay zero).
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from gaussverse.diffsynth.models.wan_video_dit import sinusoidal_embedding_1d
from prope import CameraBranch, CameraTransforms

ROLE_ANCHOR, ROLE_RECENT, ROLE_BANK, ROLE_RENDER, ROLE_TARGET = 0, 1, 2, 3, 4
WIDTH, HEIGHT = 448, 256


PLAIN = False                                                                         # see configure_trainables
LAYOUT = 'fixed'          # 'fixed': anchor 0, bank 1..6, recent 7..10, chunks 11+ | 'memorizon': anchor 0, bank all 1, recent 2..5, chunks 6+


def rope_positions(n_bank, n_recent, n_chunks, layout=None):
    """Fixed slot positions. 'memorizon' reproduces memorizon/layout.py (BANK_ROPE_INDEX=1, recent+targets consecutive)."""
    if n_bank > 6 or n_recent > 4:
        raise ValueError('bank <= 6 and recent <= 4 slots')
    layout = layout or LAYOUT
    if layout == 'memorizon':                                   # memorizon/layout.py rope_index: plain when no bank
        if n_bank == 0:
            recent = list(range(1, 1 + n_recent)); chunks = [1 + n_recent + i for i in range(4 * n_chunks)]; bank = []
        else:
            bank = [1] * n_bank; recent = list(range(2, 2 + n_recent)); chunks = [2 + n_recent + i for i in range(4 * n_chunks)]
    else:
        bank = list(range(1, 1 + n_bank)); recent = list(range(11 - n_recent, 11)); chunks = [11 + i for i in range(4 * n_chunks)]
    return torch.tensor([0, *bank, *recent, *chunks])


def chunk_causal_bias(chunk_index, tokens_per_slot, device, dtype=torch.float32):
    """chunk_index (F,): -1 for context slots, j for chunk j, -(2+j) for the render slots OF chunk j (per-chunk renders,
    idea 4 rev 10-26). Returns (1,1,S,S) additive logits (0 / -inf). Rules: context keys are visible to everyone; chunk keys
    to chunks >= them; the render keys of chunk j to chunks q >= j, to render slots of chunks >= j and (j = 0 only) to context
    queries -- so a row whose only render group is chunk 0's gets the SAME mask whether it is encoded -1 (old rows, inference)
    or -2, and a later chunk's render never reaches an earlier chunk (directly or through context tokens)."""
    ci = chunk_index.repeat_interleave(tokens_per_slot).to(device)
    q = ci[:, None]; k = ci[None, :]; rk = -k - 2; rq = -q - 2
    allowed = (k == -1) | ((k >= 0) & (q >= 0) & (k <= q)) | ((k <= -2) & (q >= 0) & (q >= rk)) | ((k <= -2) & (q <= -2) & (rk <= rq)) | ((k <= -2) & (q == -1) & (rk == 0))
    bias = torch.zeros(allowed.shape, device=device, dtype=dtype).masked_fill(~allowed, float('-inf'))
    return bias[None, None]


def reference_context_bias(chunk_index, roles, tokens_per_slot, device):
    """Opt-in released Memorizon context hierarchy; target/render causality stays intact.

    Anchor reads anchor only; bank reads anchor/bank; recent reads anchor/bank/recent.
    Render queries retain the existing causal rule and targets can still read renders.
    With one query chunk and no render this equals the released Layout mask.
    """
    bias = chunk_causal_bias(chunk_index, tokens_per_slot, device)
    r = roles.to(device)
    q, k = r[:, None], r[None, :]
    allowed = torch.ones((len(r), len(r)), dtype=torch.bool, device=device)
    allowed &= (q != ROLE_ANCHOR) | (k == ROLE_ANCHOR)
    allowed &= (q != ROLE_BANK) | (k == ROLE_ANCHOR) | (k == ROLE_BANK)
    allowed &= (q != ROLE_RECENT) | (k == ROLE_ANCHOR) | (k == ROLE_BANK) | (k == ROLE_RECENT)
    allowed = allowed.repeat_interleave(tokens_per_slot, 0).repeat_interleave(tokens_per_slot, 1)
    return bias.masked_fill(~allowed[None, None], float('-inf'))


class LoRALinear(nn.Module):
    """y = W x + (alpha/r) B A x with B zero-initialised; the frozen base weight stays bf16."""

    def __init__(self, base, r, alpha=None):
        super().__init__(); self.base = base; self.r = r; self.scale = (alpha or r) / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) / base.in_features ** .5); self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        return self.base(x) + (x.to(self.A.dtype) @ self.A.t() @ self.B.t()).to(x.dtype) * self.scale


def add_lora(dit, r):
    """FFN + cross-attention q/k/v/o of every block (self-attention is fully trainable already)."""
    n = 0
    for block in dit.blocks:
        for i in (0, 2):
            block.ffn[i] = LoRALinear(block.ffn[i], r); n += 1
        for name in ('q', 'k', 'v', 'o'):
            setattr(block.cross_attn, name, LoRALinear(getattr(block.cross_attn, name), r)); n += 1
    return n


class PackedTeacher(nn.Module):
    def __init__(self, dit, prope=True, compress=8, lora_r=0, md_ch=2):
        """md_ch: render-slot metadata channels: 2 = (valid, conf); 5 = + (depth, conflict, support), the Gaussian-feature pilot."""
        super().__init__(); self.dit = dit; dim = dit.dim
        self.lora_r = lora_r
        if lora_r:
            add_lora(dit, lora_r)
        self.camera = nn.Conv3d(24, dim, (1, 2, 2), stride=(1, 2, 2)); self.metadata = nn.Conv3d(md_ch, dim, (1, 2, 2), stride=(1, 2, 2))
        self.roles = nn.Embedding(5, dim); std = dim ** -.5
        for layer in (self.camera, self.metadata, self.roles):
            nn.init.trunc_normal_(layer.weight, mean=0., std=std, a=-3 * std, b=3 * std)
            if getattr(layer, 'bias', None) is not None:
                nn.init.zeros_(layer.bias)
        self.prope = nn.ModuleList([CameraBranch(dim, dit.num_heads, compress) for _ in dit.blocks]) if prope else None

    def configure_trainables(self):
        for p in self.dit.parameters():
            p.requires_grad_(False)
        for name, p in self.dit.named_parameters():
            if '.self_attn.' in name or name.startswith('patch_embedding.') or name.endswith('.A') or name.endswith('.B'):
                p.requires_grad_(True); p.data = p.data.float()
        for module in [self.camera, self.metadata, self.roles] + ([self.prope] if self.prope is not None else []):
            module.float()
        if PLAIN:                                                                          # 10-22 'plain Memorizon' arm: only Memorizon's own
            for module in (self.camera, self.metadata, self.roles):                        # branches train (self-attn/patch subset + PRoPE); our
                for p in module.parameters():                                              # Plucker/metadata/role projections stay at their
                    p.requires_grad_(False)                                                # zero init (no contribution)
        return [(n, p) for n, p in self.named_parameters() if p.requires_grad]

    def forward(self, latents, rays, roles, positions, times, text, chunk_index, c2w=None, K=None, metadata=None, checkpoint_blocks=False):
        """latents (1,48,F,h,w) context x0 + noisy chunks; rays (1,24,F,h,w); roles/positions/times/chunk_index (F,);
        text (1,L,4096); c2w (F,4,4) per-slot camera; K (3,3). Returns velocity for the chunk slots (1,48,4N,h,w)."""
        dit = self.dit; b, _, f, _, _ = latents.shape
        if b != 1 or rays.shape[2] != f or roles.numel() != f or positions.numel() != f or times.numel() != f or chunk_index.numel() != f:
            raise ValueError('one packed sequence with one entry per slot is required')
        n_chunks = int(chunk_index.max()) + 1
        if n_chunks < 1 or not torch.equal(chunk_index[-4 * n_chunks:], torch.arange(n_chunks, device=chunk_index.device).repeat_interleave(4)) \
                or (chunk_index[:-4 * n_chunks] >= 0).any():
            raise ValueError('chunks must be the trailing slots, four per chunk, in order')
        if (times[chunk_index < 0] != 0).any():
            raise ValueError('context slots must carry t=0')
        n_render = int((roles == ROLE_RENDER).sum()); n = 4 * n_chunks
        if n_render % 4 or n_render > n or (n_render and not (roles[-n - n_render:-n] == ROLE_RENDER).all()):
            raise ValueError('render slots: none, or 4 per rendered chunk (chunks 0..G-1) right before the chunks')
        if n_render and not torch.equal(positions[-n - n_render:-n], positions[-n:][:n_render]):
            raise ValueError('render group g must share chunk g positions')
        rci = chunk_index[-n - n_render:-n] if n_render else chunk_index[:0]
        if n_render and not (torch.equal(rci, torch.full_like(rci, -1)) or torch.equal(rci, -(2 + torch.arange(n_render // 4, device=rci.device)).repeat_interleave(4))):
            raise ValueError('render slots carry chunk_index -1 (shared) or -(2+g) for group g')
        if n_render == 4 and (rci == -1).all() and n_chunks > 1:
            pass                                                                             # chunk-0 render shared with later chunks: the 10-22 stage-B rows
        if metadata is not None and metadata.shape[1] != self.metadata.in_channels:
            raise ValueError(f'metadata has {metadata.shape[1]} channels, the model takes {self.metadata.in_channels}')
        if metadata is None:
            if n_render:
                raise ValueError('render slots need (valid, conf) metadata')
            metadata = latents.new_zeros((1, self.metadata.in_channels, f, *latents.shape[-2:]))
        x = dit.patch_embedding(latents) + self.camera(rays) + self.metadata(metadata)
        _, dim, f, h, w = x.shape
        x = x.permute(0, 2, 3, 4, 1) + self.roles(roles).to(x.dtype)[None, :, None, None, :]
        x = x.reshape(b, f * h * w, dim)
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, times.to(x.dtype) * 1000).to(x.dtype))
        tm = dit.time_projection(t).reshape(f, 6, dim).repeat_interleave(h * w, dim=0)[None]
        text = dit.text_embedding(text)
        p = positions.long()
        if int(p.max()) >= len(dit.freqs[0]):
            raise ValueError('temporal position outside the RoPE table')
        temporal = dit.freqs[0].to(x.device)[p]
        freqs = torch.cat([temporal.view(f, 1, 1, -1).expand(f, h, w, -1),
                           dit.freqs[1].to(x.device)[:h].view(1, h, 1, -1).expand(f, h, w, -1),
                           dit.freqs[2].to(x.device)[:w].view(1, 1, w, -1).expand(f, h, w, -1)], dim=-1).reshape(f * h * w, 1, -1)
        context_mode = getattr(self, 'context_mode', 'shared')
        if context_mode not in ('shared', 'reference'):
            raise ValueError('unknown context attention mode: ' + str(context_mode))
        bias = (reference_context_bias(chunk_index, roles, h * w, x.device) if context_mode == 'reference'
                else chunk_causal_bias(chunk_index, h * w, x.device))
        if n_render:                                                     # invalid render tokens are never attended (keys masked)
            valid = torch.nn.functional.avg_pool2d(metadata[:, 0].float(), 2)[0]                     # (f,h,w)
            invalid = ((valid < .5) & (roles == ROLE_RENDER).view(f, 1, 1).to(valid.device)).reshape(-1)
            bias = bias.masked_fill(invalid[None, None, None, :], float('-inf'))
        cams = None
        if self.prope is not None:
            if c2w is None or K is None:
                raise ValueError('PRoPE needs one camera per slot and the intrinsics')
            cams = CameraTransforms(c2w[None].to(x.device), K.to(x.device), self.prope[0].head_dim, w, h, 32 * w, 32 * h)   # pixels = 16 (VAE) x 2 (patch) per token

        def run(block, branch, x):
            x = block(x, text, tm, freqs, attn_logit_bias=bias)
            return x if branch is None else x + branch(x, cams, bias)

        for i, block in enumerate(dit.blocks):
            branch = None if self.prope is None else self.prope[i]
            if checkpoint_blocks and torch.is_grad_enabled():
                x = checkpoint(run, block, branch, x, use_reentrant=False)
            else:
                x = run(block, branch, x)
        n = 4 * n_chunks
        out = dit.head(x[:, -n * h * w:], t[-n:].repeat_interleave(h * w, dim=0)[None])
        return dit.unpatchify(out, (n, h, w))
