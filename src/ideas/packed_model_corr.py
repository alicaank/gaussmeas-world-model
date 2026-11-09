"""PackedTeacher + map-addressed attention bias (PILOT_PLAN_CVPR_20261104.md, read (c)).

A sparse 0/1 correspondence matrix C (S,S) in packed-token order says, for every target token, which anchor/bank/recent token sees
the same mapped surface point (map_geometry.correspondences). Block i adds gain_i * C to its attention logits (main attention and
the PRoPE branch alike); gain is a per-block scalar, zero at init, so the model is bit-identical to PackedTeacher until trained.
Kept as a subclass in its own file so the queues importing packed_model.py are unaffected (4 Nov: Gaussian-meas queue running).
forward(..., corr=None): corr = sparse or dense (S,S) float on any device; None -> plain PackedTeacher.forward."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from gaussverse.diffsynth.models.wan_video_dit import sinusoidal_embedding_1d
import packed_model
from packed_model import PackedTeacher, chunk_causal_bias, ROLE_RENDER
from prope import CameraTransforms, CameraBranch


class PackedTeacherCorr(PackedTeacher):
    def __init__(self, dit, prope=True, compress=8, lora_r=0, md_ch=2, mode='bias', reader_every=1):
        """mode 'bias': per-block zero-init gain x correspondence matrix added to the dense attention logits (soft preference).
        mode 'restricted': GEAR-style restricted read (arXiv 2609.34722 Table 2: same module, dense vs restricted LPIPS .60 vs .45):
        a NEW narrow PRoPE branch per block (CameraBranch, zero-init out_proj, every `reader_every`-th block) whose target queries
        attend ONLY to their corresponded anchor/bank/recent tokens; queries without a correspondence read nothing (output masked),
        native attention and the Memorizon PRoPE branch stay untouched. Identity at init in both modes."""
        super().__init__(dit, prope=prope, compress=compress, lora_r=lora_r, md_ch=md_ch)
        self.mode = mode; self.reader_every = int(reader_every)
        self.corr_gain = nn.Parameter(torch.zeros(len(dit.blocks)))          # zero-init: identity to PackedTeacher
        self.reader = nn.ModuleList([CameraBranch(dit.dim, dit.num_heads, compress) if i % self.reader_every == 0 else nn.Identity()
                                     for i in range(len(dit.blocks))]) if mode in ('restricted', 'dense') else None   # 'dense': same reader, no restriction (GEAR's control)

    def configure_trainables(self):
        params = super().configure_trainables()
        self.corr_gain.requires_grad_(self.mode == 'bias'); self.corr_gain.data = self.corr_gain.data.float()
        if self.reader is not None:
            self.reader.float()
            for p in self.reader.parameters():
                p.requires_grad_(True)
        return [(n, p) for n, p in self.named_parameters() if p.requires_grad]

    def forward(self, latents, rays, roles, positions, times, text, chunk_index, c2w=None, K=None, metadata=None, checkpoint_blocks=False, corr=None):
        if corr is None:
            return super().forward(latents, rays, roles, positions, times, text, chunk_index, c2w=c2w, K=K, metadata=metadata, checkpoint_blocks=checkpoint_blocks)
        dit = self.dit; b, _, f, _, _ = latents.shape
        if b != 1 or rays.shape[2] != f or roles.numel() != f or positions.numel() != f or times.numel() != f or chunk_index.numel() != f:
            raise ValueError('packed sequence shapes disagree')
        n_chunks = int(chunk_index.max()) + 1
        if n_chunks < 1 or not torch.equal(chunk_index[-4 * n_chunks:], torch.arange(n_chunks, device=chunk_index.device).repeat_interleave(4)) \
                or (chunk_index[:-4 * n_chunks] >= 0).any():
            raise ValueError('chunk_index must end with 0..N-1 (x4) and be negative before')
        if (times[chunk_index < 0] != 0).any():
            raise ValueError('context slots must be at t=0')
        n = 4 * n_chunks; n_render = int((roles == ROLE_RENDER).sum())
        if n_render % 4 or n_render > n or (n_render and not (roles[-n - n_render:-n] == ROLE_RENDER).all()):
            raise ValueError('render slots: none, or 4 per rendered chunk (chunks 0..G-1) right before the chunks')
        if n_render and not torch.equal(positions[-n - n_render:-n], positions[-n:][:n_render]):
            raise ValueError('render group g must share chunk g positions')
        rci = chunk_index[-n - n_render:-n] if n_render else chunk_index[:0]
        if n_render and not (torch.equal(rci, torch.full_like(rci, -1)) or torch.equal(rci, -(2 + torch.arange(n_render // 4, device=rci.device)).repeat_interleave(4))):
            raise ValueError('render slots carry chunk_index -1 (shared) or -(2+g) for group g')
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
        bias = chunk_causal_bias(chunk_index, h * w, x.device)
        if n_render:
            valid = torch.nn.functional.avg_pool2d(metadata[:, 0].float(), 2)[0]
            invalid = ((valid < .5) & (roles == ROLE_RENDER).view(f, 1, 1).to(valid.device)).reshape(-1)
            bias = bias.masked_fill(invalid[None, None, None, :], float('-inf'))
        S = f * h * w
        corr = corr.to_dense() if corr.is_sparse else corr
        if tuple(corr.shape) != (S, S):
            raise ValueError(f'corr is {tuple(corr.shape)}, the packed sequence has {S} tokens')
        corr = corr.to(x.device, bias.dtype)[None, None]
        if (corr[0, 0][bias[0, 0] == float('-inf')] != 0).any():
            raise ValueError('correspondences point at masked tokens')      # the map may only address visible context tokens
        cams = None
        if self.prope is not None or self.reader is not None:
            if c2w is None or K is None:
                raise ValueError('PRoPE needs one camera per slot and the intrinsics')
            hd = (self.prope[0] if self.prope is not None else next(m for m in self.reader if isinstance(m, CameraBranch))).head_dim
            cams = CameraTransforms(c2w[None].to(x.device), K.to(x.device), hd, w, h, 32 * w, 32 * h)
        rbias = None; has_corr = None
        if self.reader is not None and self.mode == 'restricted':             # restricted mask: allowed = corresponded keys only
            allowed = corr[0, 0] > 0; has_corr = allowed.any(1)                 # (S,) target tokens with at least one correspondence
            allowed = allowed | torch.diag(~has_corr)                          # empty rows attend to themselves (finite softmax), output masked below
            rbias = torch.zeros_like(bias).masked_fill(~allowed[None, None], float('-inf')); has_corr = has_corr.to(x.dtype)[None, :, None]
        elif self.reader is not None:                                         # dense control: the same reader over everything causality allows,
            rbias = bias; has_corr = (chunk_index.repeat_interleave(h * w) >= 0).to(x.dtype)[None, :, None]   # output on every target token

        def run(block, branch, reader, gain, x):
            bi = bias + gain * corr if self.mode == 'bias' else bias           # gain is a 0-d tensor: the graph reaches corr_gain
            x = block(x, text, tm, freqs, attn_logit_bias=bi)
            if branch is not None:
                x = x + branch(x, cams, bi)
            if reader is not None:
                x = x + reader(x, cams, rbias) * has_corr
            return x

        for i, block in enumerate(dit.blocks):
            branch = None if self.prope is None else self.prope[i]; gain = self.corr_gain[i].to(bias.dtype)
            reader = self.reader[i] if (self.reader is not None and isinstance(self.reader[i], CameraBranch)) else None
            if checkpoint_blocks and torch.is_grad_enabled():
                x = checkpoint(run, block, branch, reader, gain, x, use_reentrant=False)
            else:
                x = run(block, branch, reader, gain, x)
        out = dit.head(x[:, -n * h * w:], t[-n:].repeat_interleave(h * w, dim=0)[None])
        return dit.unpatchify(out, (n, h, w))


def upgrade(model, mode='bias', reader_every=1):
    """Wrap an existing PackedTeacher instance's modules into a PackedTeacherCorr without reloading the DiT (shares parameters)."""
    if isinstance(model, PackedTeacherCorr):
        return model
    dev = next(model.parameters()).device
    new = PackedTeacherCorr.__new__(PackedTeacherCorr); nn.Module.__init__(new)
    new.dit = model.dit; new.lora_r = model.lora_r; new.camera = model.camera; new.metadata = model.metadata; new.roles = model.roles
    new.prope = model.prope                                              # attribute assignment registers the shared modules
    new.mode = mode; new.reader_every = int(reader_every)
    new.corr_gain = nn.Parameter(torch.zeros(len(model.dit.blocks), device=dev))
    new.reader = nn.ModuleList([CameraBranch(model.dit.dim, model.dit.num_heads, 8).to(dev) if i % new.reader_every == 0 else nn.Identity()
                                for i in range(len(model.dit.blocks))]) if mode in ('restricted', 'dense') else None
    return new
