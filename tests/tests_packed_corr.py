"""CPU tests for packed_model_corr.py with a small stand-in DiT (same interface as the Wan DiT pieces the packed model touches)."""
import math, torch
from torch import nn
import packed_model
from packed_model import PackedTeacher, ROLE_ANCHOR, ROLE_BANK, ROLE_RECENT, ROLE_RENDER, ROLE_TARGET
from packed_model_corr import PackedTeacherCorr, upgrade
from prope import CameraBranch
import map_geometry as mg

torch.manual_seed(0)
DIM, HEADS, FREQ = 128, 2, 16
H_LAT, W_LAT = 4, 6                                   # latent grid -> 2x3 tokens per slot
TPS = (H_LAT // 2) * (W_LAT // 2)


class Block(nn.Module):
    def __init__(self):
        super().__init__(); self.qkv = nn.Linear(DIM, 3 * DIM); self.o = nn.Linear(DIM, DIM); self.ff = nn.Linear(DIM, DIM)

    def forward(self, x, text, tm, freqs, attn_logit_bias=None):
        B, T, _ = x.shape; q, k, v = self.qkv(x).view(B, T, 3, HEADS, DIM // HEADS).unbind(2)
        logits = torch.einsum('bqhd,bkhd->bhqk', q, k) / math.sqrt(DIM // HEADS)
        if attn_logit_bias is not None:
            logits = logits + attn_logit_bias
        a = logits.softmax(-1); out = torch.einsum('bhqk,bkhd->bqhd', a, v).reshape(B, T, DIM)
        return x + self.o(out) + self.ff(x).tanh() + tm[:, :, 0] * 0


class Head(nn.Module):
    def __init__(self):
        super().__init__(); self.lin = nn.Linear(DIM, 48 * 4)

    def forward(self, x, t):
        return self.lin(x)


class MiniDiT(nn.Module):
    def __init__(self, n_blocks=3):
        super().__init__(); self.dim = DIM; self.num_heads = HEADS; self.freq_dim = FREQ
        self.patch_embedding = nn.Conv3d(48, DIM, (1, 2, 2), stride=(1, 2, 2)); self.time_embedding = nn.Linear(FREQ, DIM)
        self.time_projection = nn.Linear(DIM, 6 * DIM); self.text_embedding = nn.Linear(8, DIM); self.head = Head()
        self.freqs = [torch.randn(64, 4, dtype=torch.cfloat), torch.randn(8, 2, dtype=torch.cfloat), torch.randn(8, 2, dtype=torch.cfloat)]
        self.blocks = nn.ModuleList([Block() for _ in range(n_blocks)])

    def unpatchify(self, out, shape):
        n, h, w = shape; return out.view(1, n, h, w, 48, 2, 2).permute(0, 4, 1, 2, 5, 3, 6).reshape(1, 48, n, 2 * h, 2 * w)


def make_inputs(n_slots_ctx=3, n_chunks=1, render=False):
    f = n_slots_ctx + (4 if render else 0) + 4 * n_chunks
    roles = torch.tensor([ROLE_ANCHOR, ROLE_BANK, ROLE_RECENT][:n_slots_ctx] + ([ROLE_RENDER] * 4 if render else []) + [ROLE_TARGET] * 4 * n_chunks)
    ci = torch.tensor([-1] * (n_slots_ctx + (4 if render else 0)) + [j for j in range(n_chunks) for _ in range(4)])
    times = torch.cat([torch.zeros(f - 4 * n_chunks), torch.full((4 * n_chunks,), .5)])
    latents = torch.randn(1, 48, f, H_LAT, W_LAT); rays = torch.randn(1, 24, f, H_LAT, W_LAT)
    pos_ctx = torch.arange(n_slots_ctx); pos_chunks = n_slots_ctx + torch.arange(4 * n_chunks)
    positions = torch.cat([pos_ctx, pos_chunks[:4] if render else pos_chunks[:0], pos_chunks])          # render slots share chunk-0 positions
    text = torch.randn(1, 5, 8); md = torch.ones(1, 2, f, H_LAT, W_LAT) if render else None
    K = torch.tensor([[50., 0, 3 * 16.], [0, 50., 2 * 16.], [0, 0, 1]]); c2w = torch.eye(4).repeat(f, 1, 1)
    return dict(latents=latents, rays=rays, roles=roles, positions=positions, times=times, text=text, chunk_index=ci, c2w=c2w, K=K, metadata=md)


def fwd(model, inp, corr=None, checkpoint_blocks=False):
    return model(inp['latents'], inp['rays'], inp['roles'], inp['positions'], inp['times'], inp['text'], inp['chunk_index'], c2w=inp['c2w'], K=inp['K'],
                 metadata=inp['metadata'], checkpoint_blocks=checkpoint_blocks, **({} if corr is None else dict(corr=corr)))


def corr_for(inp, pairs):
    """pairs: list of (target_token, source_token) in packed order."""
    S = inp['latents'].shape[2] * TPS; M = torch.zeros(S, S)
    for t, s in pairs:
        M[t, s] = 1
    return M.to_sparse()


def test_zero_gain_identity_and_gradient():
    for prope in (False, True):
        dit = MiniDiT(); base = PackedTeacher(dit, prope=prope); base.configure_trainables()
        model = upgrade(base); assert isinstance(model, PackedTeacherCorr) and model.dit is dit and (model.corr_gain == 0).all()
        inp = make_inputs(render=True); corr = corr_for(inp, [(7 * TPS + 1, 0 * TPS + 2), (8 * TPS, 1 * TPS + 3)])   # targets -> anchor/bank tokens
        ref = fwd(base, inp); out = fwd(model, inp, corr=corr); out_none = fwd(model, inp)
        assert torch.allclose(out, ref, atol=0, rtol=0) and torch.allclose(out_none, ref, atol=0, rtol=0), 'zero gain must be exact identity'
        with torch.no_grad():
            model.corr_gain.fill_(3.)
        out2 = fwd(model, inp, corr=corr); assert not torch.allclose(out2, ref), 'gain must change the output when correspondences exist'
        with torch.no_grad():
            model.corr_gain.zero_()
        loss = fwd(model, inp, corr=corr, checkpoint_blocks=True).square().mean(); loss.backward()
        assert model.corr_gain.grad is not None and model.corr_gain.grad.abs().sum() > 0, 'gain receives gradient through checkpointed blocks'
        names = dict(model.configure_trainables()); assert 'corr_gain' in names
        print('ok identity/gradient prope=%s' % prope)


def test_masked_target_rejected():
    dit = MiniDiT(); model = upgrade(PackedTeacher(dit, prope=False)); inp = make_inputs(n_chunks=2)
    bad = corr_for(inp, [(3 * TPS, 7 * TPS)])                                    # chunk 0 token -> chunk 1 token (masked by causality)
    try:
        fwd(model, inp, corr=bad); raise AssertionError('must reject correspondences into masked tokens')
    except ValueError as e:
        assert 'masked' in str(e)
    ok = corr_for(inp, [(7 * TPS, 3 * TPS)]); fwd(model, inp, corr=ok)         # chunk 1 -> chunk 0 is allowed
    print('ok masked rejection')


def test_matrix_from_geometry_matches_packed_order():
    inp = make_inputs(render=False); n_slots = inp['latents'].shape[2]
    corr = {(0, 0): -torch.ones(TPS, dtype=torch.long).numpy()}; corr[(0, 0)][4] = 1                              # target slot 3 token 4 -> anchor token 1
    M = mg.correspondence_matrix(n_slots, [3, 4, 5, 6], [0], corr, TPS); D = M.to_dense()
    assert D[3 * TPS + 4, 1] == 1 and D.sum() == 1
    fwd(upgrade(PackedTeacher(MiniDiT(), prope=False)), inp, corr=M); print('ok geometry matrix order')


def test_restricted_reader():
    dit = MiniDiT(n_blocks=1); base = PackedTeacher(dit, prope=True); base.configure_trainables(); model = upgrade(base, mode='restricted', reader_every=2)   # 1 block: later blocks would spread the read
    names = dict(model.configure_trainables()); assert 'reader.0.q_proj.weight' in names and 'corr_gain' not in names
    m2 = upgrade(PackedTeacher(MiniDiT(n_blocks=4), prope=True), mode='restricted', reader_every=2); n2 = dict(m2.configure_trainables())
    assert 'reader.0.q_proj.weight' in n2 and 'reader.2.q_proj.weight' in n2 and 'reader.1.q_proj.weight' not in n2
    inp = make_inputs(render=True); corr = corr_for(inp, [(7 * TPS + 1, 0 * TPS + 2), (7 * TPS + 1, 1 * TPS + 5), (8 * TPS, 1 * TPS + 3)])
    ref = fwd(base, inp); out = fwd(model, inp, corr=corr); assert torch.allclose(out, ref, atol=0, rtol=0), 'zero-init out_proj: identity'
    with torch.no_grad():
        for m in model.reader:
            if isinstance(m, CameraBranch): nn.init.normal_(m.out_proj.weight, std=.1)
    out2 = fwd(model, inp, corr=corr); d = (out2 - ref).abs()
    # only target latents carrying correspondences change (slots 7 and 8 -> latents 7, 8 of the 4-latent output = indices 0 and 1)
    assert d[:, :, 0].sum() > 0 and d[:, :, 1].sum() > 0 and d[:, :, 2:].sum() == 0, 'reader output is masked to tokens with correspondences'
    assert torch.isfinite(out2).all()
    empty = corr_for(inp, []); assert torch.allclose(fwd(model, inp, corr=empty), ref, atol=0, rtol=0), 'no correspondences -> no read'
    loss = fwd(model, inp, corr=corr, checkpoint_blocks=True).square().mean(); loss.backward()
    assert model.reader[0].q_proj.weight.grad is not None and model.reader[0].q_proj.weight.grad.abs().sum() > 0
    print('ok restricted reader')


def test_dense_control():
    dit = MiniDiT(n_blocks=1); base = PackedTeacher(dit, prope=True); base.configure_trainables(); model = upgrade(base, mode='dense', reader_every=1)
    inp = make_inputs(render=True); corr = corr_for(inp, [(7 * TPS + 1, 0 * TPS + 2)]); ref = fwd(base, inp)
    assert torch.allclose(fwd(model, inp, corr=corr), ref, atol=0, rtol=0)
    with torch.no_grad():
        nn.init.normal_(model.reader[0].out_proj.weight, std=.1)
    d = (fwd(model, inp, corr=corr) - ref).abs(); assert (d.sum(dim=(0, 1, 3, 4)) > 0).all(), 'dense control reads for every target token'
    assert torch.allclose(fwd(model, inp, corr=corr), fwd(model, inp, corr=corr_for(inp, [])), atol=0, rtol=0), 'dense control ignores the correspondences'
    print('ok dense control')


if __name__ == '__main__':
    test_zero_gain_identity_and_gradient(); test_masked_target_rejected(); test_matrix_from_geometry_matches_packed_order(); test_restricted_reader(); test_dense_control()
