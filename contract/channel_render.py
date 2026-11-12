"""Render as INPUT CHANNELS instead of tokens (stage-B paired arm, 12 Nov; a collaborator: a rendered prior is pixel-aligned with the frame it conditions ->
concatenate it to the noisy latent; retrieved frames are other viewpoints -> they stay tokens).

install(model) leaves packed_model untouched and changes the forward in one place:
  rows / the inference roller keep packing 4 render slots (role 3) + (valid, conf) metadata exactly as before; the wrapped forward REMOVES those slots from
  the sequence and feeds render_cat = (render latent x valid, valid, conf) = 50 channels on the target slots of the SAME chunk (group g -> chunk g) through a
  new zero-initialised Conv3d(50, dim, (1,2,2)) added to the patch embedding (a forward hook on model.metadata, next to the camera / metadata terms).
  Zero init -> the model starts exactly as its parent WITHOUT render. No render tokens -> 4 slots fewer in the sequence, no key masking, no attention
  competition: every target token reads its own render cell. Rows without render (render dropout, stage-A rows, CFG's render-free pass) are unchanged.
Trainable 'render_in.*' (lr_new group). A run writes channel.flag next to config.json; evaluators call maybe_install(model, config_dir)."""
from pathlib import Path
import torch
from torch import nn
from packed_model import ROLE_RENDER

CH = 50


def install(model):
    if hasattr(model, 'render_in'): return model
    ref = model.metadata.weight
    model.render_in = nn.Conv3d(CH, model.dit.dim, (1, 2, 2), stride=(1, 2, 2)).to(ref.device).float()
    nn.init.zeros_(model.render_in.weight); nn.init.zeros_(model.render_in.bias); model._rc = None; inner = model.forward

    def hook(mod, inp, out):
        rc = model._rc
        return out if rc is None else out + model.render_in(rc).to(out.dtype)
    model.metadata.register_forward_hook(hook)

    def forward(latents, rays, roles, positions, times, text, chunk_index, c2w=None, K=None, metadata=None, checkpoint_blocks=False):
        r = roles == ROLE_RENDER; n_r = int(r.sum()); model._rc = None
        if n_r:
            if metadata is None or metadata.shape[1] != 2: raise ValueError('channel render needs (valid, conf) metadata on the render slots')
            n = 4 * (int(chunk_index.max()) + 1)
            if n_r % 4 or n_r > n or not bool(r[-n - n_r:-n].all()): raise ValueError('render slots: 4 per rendered chunk, right before the chunks')
            if not torch.equal(positions[-n - n_r:-n], positions[-n:][:n_r]): raise ValueError('render group g must share chunk g positions')
            keep = ~r; f2 = int(keep.sum()); rd = r.to(latents.device); md = metadata[:, :, rd.to(metadata.device)]
            rc = md.new_zeros((1, CH, f2, *latents.shape[-2:]))
            rc[:, :48, f2 - n:f2 - n + n_r] = latents[:, :, rd].to(md.dtype) * md[:, :1]; rc[:, 48:, f2 - n:f2 - n + n_r] = md
            model._rc = rc; kd = keep.to(latents.device)
            latents = latents[:, :, kd]; rays = rays[:, :, keep.to(rays.device)]; metadata = None
            roles, positions, times, chunk_index = roles[keep], positions[keep.to(positions.device)], times[keep.to(times.device)], chunk_index[keep.to(chunk_index.device)]
            if c2w is not None: c2w = c2w[keep.to(c2w.device)]
        try:
            return inner(latents, rays, roles, positions, times, text, chunk_index, c2w, K, metadata=metadata, checkpoint_blocks=checkpoint_blocks)
        finally:
            model._rc = None
    model.forward = forward; return model


def maybe_install(model, config_dir):
    """Evaluators: call between fresh_model and load_trainables."""
    if (Path(config_dir) / 'channel.flag').exists(): install(model); return True
    return False
