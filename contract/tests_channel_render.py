"""CPU test of channel_render.install on a stub with PackedTeacher's interface. Usage (fresh_teacher env): tests_channel_render.py"""
import sys, types
import torch
from torch import nn
pm = types.ModuleType('packed_model'); pm.ROLE_RENDER = 3; sys.modules.setdefault('packed_model', pm)
import channel_render as cr


class Stub(nn.Module):
    def __init__(s):
        super().__init__(); s.dit = types.SimpleNamespace(dim=8); s.metadata = nn.Conv3d(2, 8, (1, 2, 2), stride=(1, 2, 2)); s.seen = None
    def forward(s, latents, rays, roles, positions, times, text, chunk_index, c2w=None, K=None, metadata=None, checkpoint_blocks=False):
        if metadata is None: metadata = latents.new_zeros((1, 2, latents.shape[2], *latents.shape[-2:]))
        s.seen = dict(latents=latents, rays=rays, roles=roles, positions=positions, times=times, chunk_index=chunk_index, c2w=c2w); return s.metadata(metadata)


torch.manual_seed(0); m = Stub(); cr.install(m); h = w = 4
roles = torch.tensor([0, 2, 1, 1, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4]); F = len(roles); ci = torch.tensor([-1] * 8 + [0] * 4 + [1] * 4)
pos = torch.tensor([0, 5, 9, 10, 11, 12, 13, 14, 11, 12, 13, 14, 15, 16, 17, 18]); times = torch.cat([torch.zeros(8), torch.full((8,), .5)])
lat = torch.randn(1, 48, F, h, w); rays = torch.randn(1, 24, F, h, w); c2w = torch.randn(F, 4, 4); md = torch.zeros(1, 2, F, h, w); md[:, :, 4:8] = torch.rand(1, 2, 4, h, w)
base = m(lat, rays, roles, pos, times, None, ci, c2w, None, metadata=md)
keep = roles != 3
assert base.shape[2] == 12 and torch.equal(m.seen['roles'], roles[keep]) and torch.equal(m.seen['positions'], pos[keep]) and torch.equal(m.seen['chunk_index'], ci[keep])
assert torch.equal(m.seen['latents'], lat[:, :, keep]) and torch.equal(m.seen['c2w'], c2w[keep]) and torch.equal(m.seen['times'], times[keep])
ref = m.metadata(torch.zeros(1, 2, 12, h, w)); assert torch.allclose(base, ref), 'zero init must equal the render-free forward'
nn.init.normal_(m.render_in.weight, std=.1); out = m(lat, rays, roles, pos, times, None, ci, c2w, None, metadata=md); d = (out - ref).abs().amax(dim=(0, 1, 3, 4))
assert (d[:4] == 0).all() and (d[4:8] > 0).all() and (d[8:] == 0).all(), d                            # only chunk 0's target slots carry the render
rc = torch.zeros(1, 50, 12, h, w); rc[:, :48, 4:8] = lat[:, :, 4:8] * md[:, :1, 4:8]; rc[:, 48:, 4:8] = md[:, :, 4:8]
assert torch.allclose(out, ref + m.render_in(rc), atol=1e-6)
nr = roles != 3; out2 = m(lat[:, :, nr], rays[:, :, nr], roles[nr], pos[nr], times[nr], None, ci[nr], c2w[nr], None, metadata=None); assert torch.allclose(out2, ref) and m._rc is None
md0 = md.clone(); md0[:, 0] = 0; out3 = m(lat, rays, roles, pos, times, None, ci, c2w, None, metadata=md0); rc0 = torch.zeros(1, 50, 12, h, w); rc0[:, 49, 4:8] = md[:, 1, 4:8]
assert torch.allclose(out3, ref + m.render_in(rc0), atol=1e-6)                                         # invalid cells: latent zeroed, conf still visible
print('channel_render tests passed')
