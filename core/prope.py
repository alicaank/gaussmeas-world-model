"""PRoPE side attention: cameras as relative positional encoding (arXiv:2507.10496).

Adapted from the Memorizon release (Apache-2.0, sota_repos/memorizon/memorizon/camera.py).
Differences: plain SDPA with an additive logit bias (our chunk-causal mask) instead of flex_attention,
one camera per latent slot, no KV cache. A zero-initialised output projection leaves the backbone
unchanged until the branch learns something.
Conventions: c2w 4x4 OpenCV axes in the anchor-depth gauge; K 3x3 in pixels of a width x height frame.
"""
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F


def invert_se3(m):
    rt = m[..., :3, :3].transpose(-1, -2); out = torch.zeros_like(m)
    out[..., :3, :3] = rt; out[..., :3, 3] = -torch.einsum('...ij,...j->...i', rt, m[..., :3, 3]); out[..., 3, 3] = 1.
    return out


def normalize_K(K, width, height):
    out = torch.zeros_like(K)
    out[..., 0, 0] = K[..., 0, 0] / width; out[..., 1, 1] = K[..., 1, 1] / height
    out[..., 0, 2] = K[..., 0, 2] / width - .5; out[..., 1, 2] = K[..., 1, 2] / height - .5; out[..., 2, 2] = 1.
    return out


def invert_K(K):
    out = torch.zeros_like(K)
    out[..., 0, 0] = 1. / K[..., 0, 0]; out[..., 1, 1] = 1. / K[..., 1, 1]
    out[..., 0, 2] = -K[..., 0, 2] / K[..., 0, 0]; out[..., 1, 2] = -K[..., 1, 2] / K[..., 1, 1]; out[..., 2, 2] = 1.
    return out


def lift(K):
    out = K.new_zeros(K.shape[:-2] + (4, 4)); out[..., :3, :3] = K; out[..., 3, 3] = 1.
    return out


def apply_projmat(feats, matrix):
    """feats (B, heads, cams*tokens, D) as D/4 homogeneous 4-vectors; matrix (B, cams, 4, 4)."""
    b, n, s, d = feats.shape; cams = matrix.shape[1]
    return torch.einsum('bcij,bncpkj->bncpki', matrix.to(feats.dtype), feats.reshape(b, n, cams, -1, d // 4, 4)).reshape(feats.shape)


def rope_coeffs(positions, base, scale, dim, dtype):
    n = dim // 2
    freqs = scale * (base ** (-torch.arange(n, device=positions.device)[None, None, None, :] / n))
    angles = positions[None, None, :, None] * freqs
    return torch.cos(angles).to(dtype), torch.sin(angles).to(dtype)


def apply_rope(feats, coeffs, inverse=False):
    cos, sin = coeffs
    if cos.shape[2] != feats.shape[2]:
        reps = feats.shape[2] // cos.shape[2]; cos, sin = cos.repeat(1, 1, reps, 1), sin.repeat(1, 1, reps, 1)
    cos, sin = cos.to(feats.dtype), sin.to(feats.dtype); half = feats.shape[-1] // 2
    x, y = feats[..., :half], feats[..., half:]
    if inverse:
        return torch.cat((cos * x - sin * y, sin * x + cos * y), dim=-1)
    return torch.cat((cos * x + sin * y, -sin * x + cos * y), dim=-1)


def block_diag(feats, funcs_and_sizes):
    funcs, sizes = zip(*funcs_and_sizes)
    return torch.cat([f(x) for f, x in zip(funcs, torch.split(feats, list(sizes), dim=-1))], dim=-1)


class CameraTransforms:
    """q / kv / output transforms of one forward, shared by every block.
    c2w (B, F, 4, 4) one camera per latent slot; K (3, 3) pixels; token order inside a slot is row-major (y, x)."""

    def __init__(self, c2w, K, head_dim, patches_x, patches_y, width, height, base=100., scale=1., dtype=torch.float32):
        Kn = normalize_K(K.float(), width, height)[None, None].expand(c2w.shape[0], c2w.shape[1], 3, 3)
        c2w = c2w.float()
        P = (lift(Kn) @ invert_se3(c2w)).to(dtype); P_inv = (c2w @ lift(invert_K(Kn))).to(dtype)
        pos_x = torch.tile(torch.arange(patches_x, device=c2w.device), (patches_y,)).float()
        pos_y = torch.repeat_interleave(torch.arange(patches_y, device=c2w.device), patches_x).float()
        rx = rope_coeffs(pos_x, base, scale, head_dim // 4, dtype); ry = rope_coeffs(pos_y, base, scale, head_dim // 4, dtype)
        half, quarter = head_dim // 2, head_dim // 4

        def make(matrix, inverse):
            return partial(block_diag, funcs_and_sizes=[(partial(apply_projmat, matrix=matrix), half),
                                                         (partial(apply_rope, coeffs=rx, inverse=inverse), quarter),
                                                         (partial(apply_rope, coeffs=ry, inverse=inverse), quarter)])
        self.apply_q = make(P.transpose(-1, -2), False); self.apply_kv = make(P_inv, False); self.apply_o = make(P, True)


class CameraBranch(nn.Module):
    """Narrow PRoPE side attention attached after one DiT block (residual, zero-init out_proj)."""

    def __init__(self, dim, num_heads, compress=8, eps=1e-6):
        super().__init__()
        self.attn_dim = dim // compress; self.num_heads = max(1, num_heads // compress); self.head_dim = self.attn_dim // self.num_heads
        if self.head_dim % 8:
            raise ValueError('PRoPE head dim must be a multiple of 8')
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.q_proj = nn.Linear(dim, self.attn_dim, bias=False); self.k_proj = nn.Linear(dim, self.attn_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.attn_dim, bias=False); self.out_proj = nn.Linear(self.attn_dim, dim, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x, camera, attn_bias=None):
        B, T, _ = x.shape; h = self.norm(x)
        dt = h.dtype; split = lambda proj: proj(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2).float()
        # camera transforms in float32: bf16 4x4 products lose the relative-pose signal (tests_numerics.py)
        q = camera.apply_q(split(self.q_proj)).to(dt); k = camera.apply_kv(split(self.k_proj)).to(dt); v = camera.apply_kv(split(self.v_proj)).to(dt)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None if attn_bias is None else attn_bias.to(q.dtype))
        out = camera.apply_o(out.float()).to(dt).transpose(1, 2).reshape(B, T, self.attn_dim)
        return self.out_proj(out)
