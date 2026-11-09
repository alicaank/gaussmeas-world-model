"""CPU tests for twoxplat_memory.py with a fake reconstructor (python tests_twoxplat_memory.py)."""
import numpy as np, torch
from scipy.spatial.transform import Rotation
import twoxplat_memory as tx
import map_geometry as mg

rng = np.random.default_rng(1)
K864 = np.array([[700., 0, 432.], [0, 700., 240.], [0, 0, 1]])


def rand_pose(t=.6, deg=25):
    T = np.eye(4); T[:3, :3] = Rotation.from_rotvec(rng.normal(size=3) * np.deg2rad(deg) / np.sqrt(3)).as_matrix(); T[:3, 3] = rng.normal(size=3) * t; return T


class FakeWorld:
    """Ground-truth gauge poses per latent; the 'reconstructor' returns them in a hidden normalised frame (Sim(3) of the gauge),
    the 'renderer' returns a constant depth d_norm so the anchor depth median is known."""

    def __init__(self, n=40, d_gauge=1., noise=0.):
        self.truth = {0: np.eye(4), **{i: rand_pose() for i in range(1, n)}}
        self.T_hidden = mg.sim3_matrix(3.7, Rotation.from_rotvec([.3, -.2, .5]).as_matrix(), [2., -1., .5])   # gauge -> norm
        self.d_norm = d_gauge * 3.7; self.calls = 0; self.noise = noise

    def reconstructor(self, frames):
        self.calls += 1; ids = [int(f[0, 0, 0]) for f in frames]
        norm = np.stack([mg.apply_sim3_to_c2w(self.T_hidden, self.truth[i]) for i in ids])
        if self.noise:
            norm[:, :3, 3] += rng.normal(size=norm[:, :3, 3].shape) * self.noise
        return dict(norm_c2w=norm, fxfycxcy_norm=np.repeat(tx.K_to_fxfycxcy(K864)[None], len(ids), 0), scene=('scene', tuple(ids)))

    def renderer(self, scene, c2w_norm, fx, h, w):
        n = len(c2w_norm); return torch.zeros(n, 3, h, w), torch.full((n, h, w), self.d_norm), torch.ones(n, h, w)


def frame(i):
    f = np.zeros((480, 864, 3), np.uint8); f[0, 0, 0] = i; return f


def test_thinning_and_crop():
    c = rng.normal(size=(50, 3)); keep = tx.thin_keyframes(c, [0, 49], 10)
    assert len(keep) == 10 and 0 in keep and 49 in keep and keep == sorted(keep)
    assert tx.thin_keyframes(c[:5], [0], 10) == [0, 1, 2, 3, 4]
    (rh, rw, top, left), A = tx.cover_crop_params((480, 864), (532, 952))
    assert rh == 532 and rw == 958 and top == 0 and left == 3
    p = A @ np.array([432., 240., 1.]); assert abs(p[0] - (432 * rw / 864 - 3)) < 1e-9 and abs(p[1] - 240 * 532 / 480) < 1e-9
    x, A2 = tx.resize_crop_frames(np.zeros((2, 480, 864, 3), np.uint8)); assert tuple(x.shape) == (2, 3, 532, 952) and np.allclose(A, A2)
    Ti = tx.sim3_inverse(mg.sim3_matrix(2., Rotation.from_rotvec([.1, .2, .3]).as_matrix(), [1, 2, 3]))
    T = mg.sim3_matrix(2., Rotation.from_rotvec([.1, .2, .3]).as_matrix(), [1, 2, 3]); assert np.abs(T @ Ti - np.eye(4)).max() < 1e-12
    print('ok thinning/crop/inverse')


def test_call1_depth_scale_then_umeyama():
    w = FakeWorld(); scene = tx.TwoXplatScene(w.reconstructor, w.renderer, max_views=32)
    # call 1: anchor known, latents 1..4 unknown -> scale from the anchor depth median
    kf = {0: (frame(0), np.eye(4)), **{i: (frame(i), None) for i in range(1, 5)}}
    out = scene.update(kf, K864)
    assert scene.info['mode'] == 'depth_scale' and abs(scene.info["scale"] - 1 / 3.7) < 1e-6, scene.info
    for i in range(1, 5):
        assert np.abs(out[i] - w.truth[i]).max() < 1e-5
    # call 2: 1..4 known (measured), 5..8 unknown -> Umeyama on 5 known centres
    kf.update({i: (frame(i), out[i]) for i in range(1, 5)}); kf.update({i: (frame(i), None) for i in range(5, 9)})
    out2 = scene.update(kf, K864); assert scene.info['mode'] == 'umeyama' and scene.info['centre_residual_max'] < 1e-9
    for i in range(5, 9):
        assert np.abs(out2[i] - w.truth[i]).max() < 1e-5
    assert w.calls == 2
    again = scene.update(kf, K864); assert again.keys() == out2.keys() and all(np.allclose(again[i], out2[i]) for i in out2) and w.calls == 2   # cached scene
    # render: planned gauge cameras come back with depth in gauge units (constant d_norm * scale = 1)
    rgb, depth, alpha = scene.render(np.stack([w.truth[3], w.truth[7]]), K864, size=(120, 216))
    assert tuple(rgb.shape) == (2, 3, 120, 216) and abs(float(depth.mean()) - 1.) < 1e-5 and alpha.min() == 1
    print('ok call1 depth scale -> umeyama, caching, render units')


def test_thinning_keeps_new_and_latest():
    w = FakeWorld(n=60); scene = tx.TwoXplatScene(w.reconstructor, w.renderer, max_views=16)
    kf = {0: (frame(0), np.eye(4)), **{i: (frame(i), w.truth[i]) for i in range(1, 50)}, **{i: (frame(i), None) for i in range(50, 54)}}
    out = scene.update(kf, K864); used = scene.key[0]
    assert len(used) == 16 and 0 in used and all(i in used for i in range(50, 54)) and all(i in used for i in range(46, 50)), used
    assert scene.info['thinned'] == 54 - 16 and all(np.abs(out[i] - w.truth[i]).max() < 1e-5 for i in range(50, 54))
    print('ok thinning keeps anchor + new + latest known')


def test_noise_reported():
    w = FakeWorld(noise=.01); scene = tx.TwoXplatScene(w.reconstructor, w.renderer)
    kf = {0: (frame(0), np.eye(4)), **{i: (frame(i), w.truth[i]) for i in range(1, 9)}, 9: (frame(9), None)}
    out = scene.update(kf, K864); assert scene.info['centre_residual_max'] > 0 and np.abs(out[9] - w.truth[9]).max() < .05
    print('ok noisy registration reports residuals')


def test_render_intrinsics():
    """5 Nov regression: render() scaled K864 element-wise, zeroing the principal point (renders shifted by half a frame)."""
    w = FakeWorld(); seen = []
    def renderer(scene, c2w, fx, h, wd):
        seen.append(np.asarray(fx)); return w.renderer(scene, c2w, fx, h, wd)
    scene = tx.TwoXplatScene(w.reconstructor, renderer)
    scene.update({0: (frame(0), w.truth[0])}, K864); seen.clear()
    scene.render(np.stack([w.truth[0]]), K864, size=(480, 864)); assert np.allclose(seen[-1][0], [700, 700, 432, 240])
    scene.render(np.stack([w.truth[0]]), K864, size=(240, 432)); assert np.allclose(seen[-1][0], [350, 350, 216, 120])
    print('ok render intrinsics keep the principal point')


def test_seed_scale_stale_key_require_seed():
    """5 Nov scale audit: (a) with the seed depth set, call 1 takes its scale from seed/render pixels (FakeWorld: 1/3.7);
    (b) a different stream with the same indices + identity anchor re-reconstructs (pixel signature in the key);
    (c) require_seed=True refuses to register without a seed."""
    w = FakeWorld(); scene = tx.TwoXplatScene(w.reconstructor, w.renderer, max_views=32)
    Ks = np.array([[200., 0, 224.], [0, 200., 128.], [0, 0, 1]]); scene.set_anchor_gauge(np.ones((256, 448)), np.ones((256, 448)), Ks, tag='fake')
    kf = {0: (frame(0), np.eye(4)), 1: (frame(1), None), 2: (frame(2), None)}; out = scene.update(kf, K864)
    assert scene.info['mode'] == 'seed' and scene.info['scale_source'] == 'seed_pixel_ratio' and abs(scene.info['scale'] - 1 / 3.7) < 1e-6, scene.info
    for i in (1, 2):
        assert np.abs(out[i] - w.truth[i]).max() < 1e-5
    n = w.calls; f0 = frame(0); f0[40, 48, 1] = 9                                          # other stream, same indices / poses
    scene.update({0: (f0, np.eye(4)), 1: (frame(1), None), 2: (frame(2), None)}, K864); assert w.calls == n + 1, 'stale scene reused across streams'
    scene.update({0: (f0, np.eye(4)), 1: (frame(1), None), 2: (frame(2), None)}, K864); assert w.calls == n + 1, 'identical update must hit the cache'
    strict = tx.TwoXplatScene(w.reconstructor, w.renderer, require_seed=True)
    try:
        strict.update(kf, K864); raise AssertionError('require_seed must refuse')
    except RuntimeError:
        pass
    print('ok seed scale / stale key / require_seed')


if __name__ == '__main__':
    test_render_intrinsics(); test_thinning_and_crop(); test_call1_depth_scale_then_umeyama(); test_thinning_keeps_new_and_latest(); test_noise_reported(); test_seed_scale_stale_key_require_seed()
