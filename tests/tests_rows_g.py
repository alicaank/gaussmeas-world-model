"""CPU tests for rows_g.py (python tests_rows_g.py); no corpus access."""
import numpy as np, torch
import rows_g as rg
import map_geometry as mg

K = np.array([[700., 0, 432.], [0, 700., 240.], [0, 0, 1]])


def test_dtok():
    d = np.full((2, 480, 864), 2.); d[0, :, :16] = np.nan                                    # left half of the first token column is nan -> 50% covered
    t = rg.dtok_from_depth(d); assert t.shape == (2, rg.HT, rg.WT) and np.isfinite(t).all() and abs(t[0, 0, 0] - 2.) < 1e-12
    d[1, :32, :32] = np.nan; d[1, 32:64, :32] = np.nan                                        # 100% nan tokens -> nan
    t = rg.dtok_from_depth(d); assert np.isnan(t[1, 0, 0]) and np.isnan(t[1, 1, 0]) and np.isfinite(t[1, 2, 0])
    print('ok dtok')


def test_row_correspondences_shape_and_sources_only():
    F_ = 1 + 2 + 3 + 4 + 8; nctx = 6; roles = torch.tensor([0, 2, 2, 1, 1, 1] + [3] * 4 + [4] * 8); ci = torch.tensor([-1] * 10 + [0] * 4 + [1] * 4)
    c2w = np.stack([np.eye(4) for _ in range(F_)]); c2w[1, 0, 3] = .5                           # bank slot 1 shifted
    dt = np.full((rg.HT, rg.WT), 3.); slot_d = [dt] * nctx + [None] * 4 + [dt] * 8
    M = rg.row_correspondences(K, c2w, slot_d, roles, ci, nctx); S = F_ * rg.TPS; assert M.shape == (S, S)
    D = M.to_dense(); rows_with = (D.sum(1) > 0).nonzero().flatten()
    assert rows_with.min() >= nctx * rg.TPS + 4 * rg.TPS, 'only target tokens are rows'                   # render slots get nothing
    assert (D[:, nctx * rg.TPS:].sum() == 0), 'columns only in anchor/bank/recent'
    # identical cameras (slot 0 vs targets): token i -> token i; shifted bank slot: shifted by -3.6 tokens
    t0 = (nctx + 4) * rg.TPS; assert D[t0 + 100, 100] == 1 and D[t0 + 100, rg.TPS + 100 - 4] == 1
    slot_d2 = list(slot_d); slot_d2[nctx + 4] = None                                               # no depth at a target -> no rows for it
    D2 = rg.row_correspondences(K, c2w, slot_d2, roles, ci, nctx).to_dense(); assert D2[t0:t0 + rg.TPS].sum() == 0 and D2.sum() < D.sum()
    print('ok row correspondences')


def test_bank_by_visibility():
    cmd = np.stack([np.eye(4) for _ in range(16)]); dt = np.full((4, rg.HT, rg.WT), 3.)
    same = np.eye(4); far = np.eye(4); far[:3, 3] = [40, 0, 0]; half = np.eye(4); half[0, 3] = 1.8
    cand = [2, 5, 7]; cc = [far, same, half]; cd = [dt[0]] * 3
    assert rg.bank_by_visibility(K, cmd, dt, cand, cc, cd, [], 4) == [5]
    assert rg.bank_by_visibility(K, cmd, dt, cand, cc, cd, [(same, dt[0])], 4) == []                 # anchor already sees everything
    assert rg.bank_by_visibility(K, cmd, dt, [2, 7], [far, half], [dt[0]] * 2, [], 4) == [7]
    assert rg.bank_by_visibility(K, cmd, dt, [], [], [], [], 4) == []
    print('ok visibility bank')


def test_build_row_g_interpolates_interior_poses():
    from scipy.spatial.transform import Rotation
    from closed_loop import interp_rgb_poses
    from foundation_model import packed_rays
    rng = np.random.default_rng(3); L = 14; H, W = 4, 6

    class FakeStream:
        def __init__(self):
            self.L = L; self.latents = torch.randn(1, 48, L, H, W); self.K = torch.tensor([[50., 0, 3 * 16.], [0, 50., 2 * 16.], [0, 0, 1]]); self.text = torch.zeros(1, 4, 8)
            c2w = np.stack([np.eye(4) for _ in range(4 * (L - 1) + 1)])
            for i in range(1, len(c2w)):
                c2w[i, :3, 3] = [.01 * i, 0, .02 * i]
            self.c2w = torch.from_numpy(c2w).float(); self.cam_scale = .5; self.row = dict(stream_id='fakestream')
        def frame_index(self, p):
            return 0 if p == 0 else 4 * p
    st = FakeStream(); s_ = 5
    def pack(t):
        return t.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
    meas = np.stack([np.eye(4) for _ in range(s_)])
    for j in range(1, s_):
        meas[j, :3, :3] = Rotation.from_rotvec([0, .05 * j, 0]).as_matrix(); meas[j, :3, 3] = [.1 * j, .05 * j, .3 * j]
    item = dict(s=s_, gen_latents_bf16=pack(torch.randn(1, 48, s_, H, W)), meas_end=meas, bank_g=np.array([1]), rz_g_bf16=pack(torch.randn(1, 48, 4, H, W)), meta_g=np.ones((1, 2, 4, H, W), np.float32))
    row = rg.build_row_g(item, st, with_corr=False)
    exp = st.c2w.clone().double()
    for j in range(1, s_):
        exp[4 * j - 3:4 * j + 1] = torch.from_numpy(np.stack(interp_rgb_poses(meas[j - 1], [meas[j]])))
    exp = exp.float(); exp[:, :3, 3] *= st.cam_scale
    tp = [0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8, 9, 10, 11, 12]                               # anchor, bank 1, recent 1..4, render x4, chunks 5..12
    assert torch.allclose(row['rays'], packed_rays(exp[:4 * 12 + 1], st.K, tp, height=16 * H, width=16 * W)), 'rays use interpolated measured interiors'
    assert torch.allclose(row['c2w'][1], exp[4]) and torch.allclose(row['c2w'][5], exp[16]), 'slot cameras = measured ends x cam_scale'
    assert row['bank'] == [1] and row['recent'] == [1, 2, 3, 4] and row['start'] == s_ and row['render'] and 'corr' not in row
    print('ok build_row_g interior interpolation')


if __name__ == '__main__':
    test_dtok(); test_row_correspondences_shape_and_sources_only(); test_bank_by_visibility(); test_build_row_g_interpolates_interior_poses()
