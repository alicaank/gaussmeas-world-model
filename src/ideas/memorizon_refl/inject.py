"""Zero-initialized geometric injection for Memorizon: a second patch embedding of a warp latent, added to the token
embedding of the frames it belongs to. With zero weights the released model is unchanged.

inject: [B, 48, F, h, w] (normalized VAE latent of the previous frame warped into each target pose; zeros where absent)
"""
import torch, torch.nn as nn
from memorizon.wan import transformer as T
from memorizon import model as M


def attach_inject(model):
    """Add `inject_embedding` to the transformer (idempotent) and enable the `inject` kwarg on forward paths."""
    dit = model.model
    if not hasattr(dit, 'inject_embedding'):
        pe = dit.patch_embedding
        conv = nn.Conv3d(pe.in_channels, pe.out_channels, kernel_size=pe.kernel_size, stride=pe.stride)
        nn.init.zeros_(conv.weight); nn.init.zeros_(conv.bias)
        dit.inject_embedding = conv.to(device=pe.weight.device, dtype=pe.weight.dtype)
    return model


_orig_forward = T.MemorizonTransformer.forward


def _forward(self, hidden_states, timestep, encoder_hidden_states, rope_index, attention_mask=None, camera=None,
             kv_cache=None, kv_cache_range=None, inject=None):
    if inject is None or not hasattr(self, 'inject_embedding'):
        return _orig_forward(self, hidden_states, timestep, encoder_hidden_states, rope_index, attention_mask, camera, kv_cache, kv_cache_range)
    orig = self._patch_embedding

    def patched(h):
        x = orig(h)
        return x + self.inject_embedding(inject.to(h.dtype)).flatten(2).transpose(1, 2).contiguous()
    self._patch_embedding = patched
    try:
        return _orig_forward(self, hidden_states, timestep, encoder_hidden_states, rope_index, attention_mask, camera, kv_cache, kv_cache_range)
    finally:
        self._patch_embedding = orig


T.MemorizonTransformer.forward = _forward


def trainable_parameters(model):
    out = []
    for n, p in model.named_parameters():
        want = ('prope_attn' in n) or ('inject_embedding' in n)
        p.requires_grad_(want)
        if want:
            out.append((n, p))
    return out


def save_trainable(model, path):
    torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}, path)


def load_trainable(model, path):
    sd = torch.load(path, map_location='cpu'); attach_inject(model)
    missing = model.load_state_dict(sd, strict=False)
    unexpected = [k for k in missing.unexpected_keys]
    if unexpected:
        raise KeyError(f'unexpected keys in trainable checkpoint: {unexpected[:5]}')
    return model
