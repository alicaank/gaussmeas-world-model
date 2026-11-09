"""2Xplat as the global scene memory + measured-pose observer (PILOT_PLAN_CVPR_20261104.md, step 1).

Per call: all accepted keyframes (anchor + generated latent ends, <= max_views after farthest-point thinning) go through frozen
2Xplat once. Its DA3 pass gives raw cameras for every view (the measured poses of the new frames), its appearance expert gives one
global Gaussian scene in 2Xplat's normalised frame. The normalised frame is registered to the anchor gauge with a Sim(3) fitted on
the frames whose gauge poses are already known (measured history; call 1: anchor only -> scale from the anchor's depth median,
gauge median = 1). Planned cameras are mapped gauge -> normalised for rendering; rendered depth is mapped back (x scale).

GPU parts (model, DA3, gsplat) are only touched inside TwoXplatScene.reconstruct / render; everything else is CPU-testable
(tests_twoxplat_memory.py uses a fake reconstructor). Input images for 2Xplat: 952x532 (DA3 patch 14, MVP patch 4) by cover
resize + centre crop of our 864x480 frames for DA3, 960x544 for the v2_hr appearance expert; intrinsics follow."""
import sys
from pathlib import Path
import numpy as np, torch
import map_geometry as mg

TWOX = Path('/home/hj/Alican/third_party/2Xplat'); WEIGHTS = TWOX / 'pretrained_weights'
DA_SIZE = (532, 952)             # (h, w) for DA3: multiples of 14, closest to the hr training size 960x540
APP_SIZE = (544, 960)            # (h, w) for the v2_hr appearance expert: multiples of 32 (patch 8 x two 2x merges); 5 Nov: the
                                 # published config is the lr one (patch 4, dims 256/512/1024); v2_hr's architecture is read from its
                                 # state dict (arch_from_state). Both crops are cover-crops of the same frame -> K_app = A_app A_da^-1 K_da.


# ---------------------------------------------------------------------------------------------------------- CPU helpers
def cover_crop_params(src_hw, dst_hw):
    """Cover-resize then centre-crop: returns (scaled_h, scaled_w, top, left) and the intrinsics map K' = A K (3x3)."""
    sh, sw = src_hw; dh, dw = dst_hw; s = max(dh / sh, dw / sw); rh, rw = round(sh * s), round(sw * s)
    top, left = (rh - dh) // 2, (rw - dw) // 2
    A = np.array([[rw / sw, 0, -left], [0, rh / sh, -top], [0, 0, 1.]])
    return (rh, rw, top, left), A


def resize_crop_frames(frames_uint8, dst_hw=DA_SIZE):
    """uint8 (n,H,W,3) -> float (n,3,dh,dw) in [0,1] (torch, cpu) + the 3x3 pixel map A."""
    x = torch.as_tensor(np.asarray(frames_uint8)).permute(0, 3, 1, 2).float() / 255.
    (rh, rw, top, left), A = cover_crop_params(x.shape[-2:], dst_hw)
    x = torch.nn.functional.interpolate(x, size=(rh, rw), mode='bilinear', align_corners=False, antialias=True)
    return x[:, :, top:top + dst_hw[0], left:left + dst_hw[1]].contiguous(), A


def K_to_fxfycxcy(K):
    K = np.asarray(K, np.float64); return np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])


def thin_keyframes(centres, always, max_views):
    """Farthest-point thinning on camera centres (n,3): keep `always` (indices) and add the farthest remaining until max_views.
    Returns sorted kept indices."""
    n = len(centres); always = sorted(set(int(i) for i in always))
    if n <= max_views:
        return list(range(n))
    if len(always) >= max_views:
        return always[:max_views]
    c = np.asarray(centres, np.float64); kept = list(always)
    d = np.min(np.linalg.norm(c[:, None] - c[kept][None], axis=-1), axis=1) if kept else np.full(n, np.inf)
    while len(kept) < max_views:
        d[kept] = -1; j = int(np.argmax(d))
        if d[j] <= 0 and np.isfinite(d[j]):
            break
        kept.append(j); d = np.minimum(d, np.linalg.norm(c - c[j], axis=1))
    return sorted(kept)


def sim3_inverse(T):
    s = np.cbrt(np.linalg.det(T[:3, :3])); R = T[:3, :3] / s; Ti = np.eye(4); Ti[:3, :3] = R.T / s; Ti[:3, 3] = -R.T @ T[:3, 3] / s; return Ti


# ---------------------------------------------------------------------------------------------------------- the scene
class TwoXplatScene:
    """Holds the current global reconstruction and its registration to the anchor gauge.

    reconstructor(frames_uint8 list) -> dict(norm_c2w (n,4,4) np, fxfycxcy_norm (n,4) np at DA_SIZE, scene=<opaque>) ; the
    default is TwoXplatModel.reconstruct. renderer(scene, c2w_norm (T,4,4), fxfycxcy (T,4), h, w) -> (rgb (T,3,h,w) [0,1],
    depth_norm (T,h,w), alpha (T,h,w)) float tensors on cpu."""

    def __init__(self, reconstructor, renderer, max_views=32, gauge_depth_median=1., require_seed=False, min_spread=.05):
        self.reconstructor = reconstructor; self.renderer = renderer; self.max_views = int(max_views); self.gauge_depth_median = float(gauge_depth_median)
        self.key = None; self.scene = None; self.T = None; self.info = None; self.norm_c2w = {}; self.fx = None
        self.seed = None; self.require_seed = bool(require_seed); self.min_spread = float(min_spread)   # 5 Nov scale audit

    def set_anchor_gauge(self, depth, confidence, K_seed, tag=None):
        """The anchor's seed depth in gauge units (anchor.npz: the depth that DEFINES the gauge, median 1) with its own K and
        resolution. Used for the scale whenever the cameras cannot fix it (call 1, tiny or collinear-free baselines). Call once per
        stream BEFORE its first update; also invalidates the cached reconstruction."""
        d = np.asarray(depth, np.float64); c = np.asarray(confidence, np.float64)
        if d.shape != c.shape or d.ndim != 2:
            raise ValueError('seed depth / confidence must be (h, w)')
        self.seed = dict(depth=d, conf=c, K=np.asarray(K_seed, np.float64), tag=tag); self.key = None

    def seed_scale(self, anchor_norm_c2w):
        """gauge / normalised scale = pixel median of seed_depth / rendered anchor depth over pixels valid in both (the mpu rule)."""
        h, w = self.seed['depth'].shape; fx = K_to_fxfycxcy(self.seed['K'])
        _, depth, alpha = self.renderer(self.scene, np.asarray(anchor_norm_c2w)[None], fx[None], h, w)
        dn = np.asarray(depth[0], np.float64); a = np.asarray(alpha[0], np.float64); g = self.seed['depth']
        ok = (a > .5) & np.isfinite(dn) & (dn > 0) & np.isfinite(g) & (g > 0) & (self.seed['conf'] > .5)
        if ok.sum() < 1000:
            raise RuntimeError(f'seed scale: only {int(ok.sum())} pixels valid in both the seed depth and the anchor render')
        r = g[ok] / dn[ok]; m = float(np.median(r))
        self.seed_diag = dict(pixels=int(ok.sum()), log_spread=round(float(np.median(np.abs(np.log(r / m)))), 4)); return m

    def update(self, keyframes, K864, anchor_latent=0):
        """keyframes: dict latent index -> (frame uint8 (480,864,3), gauge c2w or None). Frames with a known gauge pose register the
        reconstruction; frames with None get their measured gauge pose returned. Returns dict latent -> measured gauge c2w for the
        None entries (None if the view was thinned away or registration failed)."""
        idx = sorted(keyframes); unknown = [i for i in idx if keyframes[i][1] is None]; known = [i for i in idx if keyframes[i][1] is not None]
        if anchor_latent not in known:
            raise ValueError('the anchor must be a keyframe with a known (identity) gauge pose')
        centres = np.stack([np.asarray(keyframes[i][1], np.float64)[:3, 3] if keyframes[i][1] is not None else np.full(3, np.nan) for i in idx])
        for j, i in enumerate(idx):                                                 # unknown centres: assume near the latest known one for thinning
            if not np.isfinite(centres[j]).all():
                centres[j] = centres[[k for k, m in enumerate(idx) if keyframes[m][1] is not None]][-1]
        always = [idx.index(anchor_latent)] + [idx.index(i) for i in unknown] + [idx.index(i) for i in known[-4:]]
        keep = thin_keyframes(centres, always, self.max_views); used = [idx[k] for k in keep]
        known_sig = hash(tuple((i, np.asarray(keyframes[i][1], np.float64).round(9).tobytes()) for i in used if keyframes[i][1] is not None))
        pix_sig = hash(tuple(np.ascontiguousarray(np.asarray(keyframes[i][0])[::8, ::8]).tobytes() for i in used))   # 5 Nov: pixels too (stale-scene bug)
        key = (tuple(used), known_sig, pix_sig, None if self.seed is None else id(self.seed))   # frames + their pixels + gauge poses + seed
        if key != self.key:
            rec = self.reconstructor([keyframes[i][0] for i in used]); self.scene = rec['scene']; self.key = key
            self.norm_c2w = {i: np.asarray(rec['norm_c2w'][j], np.float64) for j, i in enumerate(used)}; self.fx = np.asarray(rec['fxfycxcy_norm'], np.float64)
            kn = [i for i in used if keyframes[i][1] is not None]
            local = np.stack([self.norm_c2w[i] for i in kn]); gauge = np.stack([np.asarray(keyframes[i][1], np.float64) for i in kn])
            if self.seed is not None:                                                 # 5 Nov scale audit: one rule, seed-defined fallback
                s_seed = self.seed_scale(self.norm_c2w[anchor_latent])
                self.T, self.info = mg.register_frames_gauge(local, gauge, seed_scale=s_seed, min_spread=self.min_spread)
                self.info.update(scale_source='cameras' if self.info['mode'] != 'seed' else 'seed_pixel_ratio', seed_tag=self.seed['tag'], seed_diag=self.seed_diag)
            elif self.require_seed:
                raise RuntimeError('TwoXplatScene(require_seed=True): call set_anchor_gauge(...) for this stream first')
            else:                                                                     # legacy (tests / old runs): median-depth fallback
                local_med = None
                if len(kn) < 3 or np.linalg.matrix_rank(local[:, :3, 3] - local[:, :3, 3].mean(0), tol=1e-6) < 2:
                    local_med = self.anchor_depth_median_norm(K864, used.index(anchor_latent), self.norm_c2w[anchor_latent])
                self.T, self.info = mg.register_frames(local, gauge, local_depth_median=local_med, gauge_depth_median=self.gauge_depth_median)
                self.info['scale_source'] = 'legacy'
            self.info['views'] = len(used); self.info['thinned'] = len(idx) - len(used); self.info['single_view'] = len(used) == 1
        out = {}
        for i in unknown:
            out[i] = mg.apply_sim3_to_c2w(self.T, self.norm_c2w[i]) if i in self.norm_c2w else None
        return out

    def anchor_depth_median_norm(self, K864, anchor_pos, anchor_norm_c2w, size=(120, 216)):
        """Median rendered depth of the anchor view in the normalised frame (used for the scale when < 3 non-collinear frames)."""
        fx = K_to_fxfycxcy(np.diag([size[1] / 864, size[0] / 480, 1.]) @ np.asarray(K864, np.float64))   # 5 Nov: was an element-wise product (cx = cy = 0)
        fx[2] *= 1; _, depth, alpha = self.renderer(self.scene, np.asarray(anchor_norm_c2w)[None], fx[None], size[0], size[1])
        d = np.asarray(depth[0], np.float64); a = np.asarray(alpha[0], np.float64); ok = (a > .5) & np.isfinite(d) & (d > 0)
        if ok.sum() < 100:
            raise RuntimeError('anchor render has no coverage; cannot set the scale')
        return float(np.median(d[ok]))

    def render(self, target_c2w_gauge, K864, size=(480, 864)):
        """rgb (T,3,h,w) in [0,1], depth in GAUGE units (T,h,w), alpha (T,h,w); cpu float tensors."""
        if self.scene is None:
            raise RuntimeError('no reconstruction yet')
        Ti = sim3_inverse(self.T); c2w_norm = mg.apply_sim3_to_c2w(Ti, np.asarray(target_c2w_gauge, np.float64))
        K = np.diag([size[1] / 864, size[0] / 480, 1.]) @ np.asarray(K864, np.float64)   # 5 Nov: was an element-wise product (cx = cy = 0); K[2, 2] = 1
        fx = np.repeat(K_to_fxfycxcy(K)[None], len(c2w_norm), 0)
        rgb, depth, alpha = self.renderer(self.scene, c2w_norm, fx, size[0], size[1])
        s = np.cbrt(np.linalg.det(self.T[:3, :3]))
        return rgb, depth * float(s), alpha


def arch_from_state(sd):
    """2Xplat appearance-expert architecture from its state dict (the v2_hr checkpoint ships without a config).
    decoder rows = patch^2 * (3 + 3(sh+1)^2 + 3 + 4 + (op+1)^2): (sh, op) solved over small degrees; must be unique."""
    d1, d2, d3 = sd['stage1.0.ln1.weight'].shape[0], sd['stage2.0.ln1.weight'].shape[0], sd['stage3.0.ln1.weight'].shape[0]
    n = {s: len({int(k.split('.')[1]) for k in sd if k.startswith(s + '.')}) for s in ('stage1', 'stage2', 'stage3')}
    tok = sd['image_tokenizer.1.weight']; in_ch = 12; p2 = tok.shape[1] // in_ch; p = int(round(p2 ** .5))
    if p * p * in_ch != tok.shape[1]:
        raise ValueError('tokenizer width is not in_channels x patch^2')
    rows = sd['gaussian_decoder.1.weight'].shape[0]
    sols = [(sh, op) for sh in range(4) for op in range(5) if p * p * (10 + 3 * (sh + 1) ** 2 + (op + 1) ** 2) == rows]
    if len(sols) != 1:
        raise ValueError(f'gaussian decoder width {rows} does not identify (sh, opacity) degrees: {sols}')
    hd = sd['stage1.0.attn.q_norm.weight'].shape[0]; mlp = sd['stage1.0.mlp.c_fc.weight'].shape[0] // d1
    return dict(dim1=d1, dim2=d2, dim3=d3, stage1_nlayer=n['stage1'], stage2_nlayer=n['stage2'], stage3_nlayer=n['stage3'], patch_size=p, in_channels=in_ch,
                head_dim=hd, inter_multi=mlp, num_register_tokens=sd['register_token_init'].shape[2], gaussians=dict(sh_degree=sols[0][0], opacity_degree=sols[0][1]))


# ---------------------------------------------------------------------------------------------------------- GPU model
class TwoXplatModel:
    """Frozen 2Xplat (TwoExpertModel) wrapper: reconstruct(frames) and render(scene, cams). Built lazily on first use."""

    def __init__(self, ckpt=WEIGHTS / '2xplat_dl3dv_hr.pt',   # 5 Nov: v2_hr renders garbage with the released code (v2 inference unpublished;
                 # self-render at its own input views ~yellow blur, depth 30x too far); dl3dv_hr self-render PSNR 34-39, LPIPS .049
                 da_weights=WEIGHTS / 'da3_giant.safetensors', device='cuda', amp=torch.bfloat16, size=APP_SIZE, da_size=DA_SIZE):
        self.ckpt = Path(ckpt); self.da_weights = Path(da_weights); self.device = device; self.amp = amp; self.size = tuple(size); self.da_size = tuple(da_size); self.model = None
        self.load_report = None

    def config(self, overrides=None):
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(TWOX / 'src/configs/inference.yaml')
        cfg.model.da_model_weights_path = str(self.da_weights); cfg.model.mvp_weights_path = ''
        cfg.data.resize_h, cfg.data.resize_w, cfg.data.square_crop = self.size[0], self.size[1], False
        cfg.inference.ckpt_path = str(self.ckpt); cfg.inference.pose_optimization = False; cfg.inference.camera_mode = 'pred_pose_pred_intr'
        cfg = OmegaConf.merge(cfg, OmegaConf.create(dict(model=self.arch)))
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
        return cfg

    def load(self, overrides=None):
        if self.model is not None:
            return self.model
        if str(TWOX) not in sys.path:
            sys.path.insert(0, str(TWOX))
        from src.model.model import TwoExpertModel
        sd = torch.load(str(self.ckpt), map_location='cpu', weights_only=True)['ema']
        self.arch = arch_from_state(sd)
        if self.size[0] % (4 * self.arch['patch_size']) or self.size[1] % (4 * self.arch['patch_size']) or self.da_size[0] % 14 or self.da_size[1] % 14:
            raise ValueError(f'input sizes {self.size} / {self.da_size} do not fit patch {self.arch["patch_size"]} x4 / DA3 patch 14')
        cfg = self.config(overrides); model = TwoExpertModel(cfg)
        res = model.load_state_dict(sd, strict=False); del sd                                   # strict=False as upstream, but checked:
        bad_missing = [k for k in res.missing_keys if not k.startswith(('pose_regressor.', 'loss_computer.'))]
        bad_extra = [k for k in res.unexpected_keys if not k.startswith(('loss_computer.', 'pose_regressor.'))]
        if bad_missing or bad_extra:
            raise RuntimeError(f'2Xplat checkpoint does not match the model: missing {bad_missing[:5]} unexpected {bad_extra[:5]}')
        self.load_report = dict(arch=self.arch, missing=len(res.missing_keys), unexpected=len(res.unexpected_keys),
                                missing_pose_regressor=sum(k.startswith('pose_regressor.') for k in res.missing_keys))
        model.to(self.device); model.eval(); self.model = model                     # upstream eval() returns None
        for p in self.model.parameters():
            p.requires_grad_(False)
        return self.model

    @torch.no_grad()
    def reconstruct(self, frames_uint8):
        """Context-only forward: DA3 cameras (raw + normalised) and the global Gaussians."""
        m = self.load(); x, A = resize_crop_frames(frames_uint8, self.size); x = x[None].to(self.device)            # (1,V,3,h,w) appearance input
        xd, Ad = resize_crop_frames(frames_uint8, self.da_size); xd = xd[None].to(self.device)                      # DA3 input
        M = torch.as_tensor(A @ np.linalg.inv(Ad), dtype=torch.float32, device=self.device)                       # DA3 pixels -> appearance pixels
        with torch.autocast('cuda', dtype=self.amp):
            with torch.autocast('cuda', enabled=False):
                out = m.geometry_expert.pose_regressor(xd.float())                                                  # raw DA3 cameras
                raw_c2w = out['extrinsics'][0].float(); fd = out['fxfycxcy'][0].float()
                fx_raw = torch.stack([fd[:, 0] * M[0, 0], fd[:, 1] * M[1, 1], fd[:, 2] * M[0, 0] + M[0, 2], fd[:, 3] * M[1, 1] + M[1, 2]], 1)
                if raw_c2w.shape[0] == 1:                                                                  # single view: 2Xplat's normaliser divides by the
                    norm_c2w = torch.linalg.inv(raw_c2w[0])[None] @ raw_c2w                                # camera extent (0) -> keep DA3 scale, camera at origin
                else:
                    norm_c2w = m.geometry_expert._shared_scene_normalization(raw_c2w, None, scene_scale=m.geometry_expert.scene_scale if hasattr(m.geometry_expert, 'scene_scale') else 1.)
            V = x.shape[1]; pad = V % m.appearance_expert.group_size                                       # stage 2 attends within view pairs: odd V ->
            xa = torch.cat([x, x[:, :1].expand(-1, pad, -1, -1, -1)], 1) if pad else x                         # append a copy of the anchor for the
            fx = torch.cat([fx_raw, fx_raw[:1].expand(pad, -1)])[None]; c2w = torch.cat([norm_c2w, norm_c2w[:1].expand(pad, -1, -1)])[None]   # appearance pass
            raymaps, Ks, w2c = m._build_raymap_input({'image': xa}, fx, c2w)                                 # only (DA3 saw the real views) and drop
            gauss = m.appearance_expert.predict_gaussians(raymaps, w2c, Ks, fx, c2w, None)                     # its Gaussians (view-major layout)
            if pad:
                n = gauss.xyz.shape[1] // (V + pad) * V
                for f in ('xyz', 'feature', 'scale', 'rotation', 'opacity'):
                    setattr(gauss, f, getattr(gauss, f)[:, :n])
        return dict(norm_c2w=norm_c2w.cpu().numpy(), raw_c2w=raw_c2w.cpu().numpy(), fxfycxcy_norm=fx_raw.cpu().numpy(), scene=gauss, pixel_map=A)

    @torch.no_grad()
    def render(self, scene, c2w_norm, fxfycxcy, h, w):
        """gsplat RGB + expected depth + alpha at (h, w) for cameras in the normalised frame."""
        from gsplat import rasterization
        from src.model.sh_eval import _spherical_harmonics
        m = self.load(); ae = m.appearance_expert
        xyz, feat, scale, rot, opa = scene.xyz[0], scene.feature[0], scene.scale[0], scene.rotation[0], scene.opacity[0]
        rgbs, deps, alps = [], [], []
        for c2w, fx in zip(np.asarray(c2w_norm, np.float64), np.asarray(fxfycxcy, np.float64)):
            c2w_t = torch.as_tensor(c2w, dtype=torch.float32, device=xyz.device); K = torch.tensor([[fx[0], 0, fx[2]], [0, fx[1], fx[3]], [0, 0, 1]], dtype=torch.float32, device=xyz.device)
            dirs = xyz - c2w_t[:3, 3][None]; o = _spherical_harmonics(ae.opacity_degree, dirs, opa).sigmoid().squeeze(-1)   # (N,3) x (N,K,1) -> (N,)
            colors, alpha, _ = rasterization(xyz, rot, scale.exp(), o, feat, torch.linalg.inv(c2w_t)[None], K[None], w, h, sh_degree=ae.sh_degree,
                                             near_plane=m.near_plane, far_plane=m.far_plane, packed=False, render_mode='RGB+ED',
                                             rasterize_mode='classic')                                   # white background composited below
            a_ = alpha[0, ..., :1]; rgb_ = colors[0, ..., :3] + (1 - a_)                                    # = upstream backgrounds=ones for RGB
            rgbs.append(rgb_.clamp(0, 1).permute(2, 0, 1).cpu()); deps.append(colors[0, ..., 3].cpu()); alps.append(a_[..., 0].cpu())
        return torch.stack(rgbs).float(), torch.stack(deps).float(), torch.stack(alps).float()


def make_scene(max_views=32, require_seed=True, **kw):
    """Real runs: require_seed=True -> every stream must call scene.set_anchor_gauge(*seed_gauge(stream_id)) first (5 Nov)."""
    model = TwoXplatModel(**kw); return TwoXplatScene(model.reconstruct, model.render, max_views=max_views, require_seed=require_seed), model


def seed_gauge(stream_id):
    """(depth, confidence, K, tag) of the stream's anchor seed depth in gauge units (dataset_v3 components/<sid>/anchor.npz)."""
    from all_scene_dataset import DATA
    f = DATA / 'components' / stream_id / 'anchor.npz'
    with np.load(f) as z:
        d, c, K = z['depth'].astype(np.float64), z['confidence'].astype(np.float64), z['K'].astype(np.float64)
    return d, c, K, f'{stream_id[:8]}:anchor.npz'
