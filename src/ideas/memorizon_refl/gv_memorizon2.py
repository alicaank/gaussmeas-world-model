"""Round 2 on released Memorizon: refinement, in-sampler error use, and reward headroom, all on top of (or against) F'.

Arms (json): base fields as gv_memorizon.py (write, replan, gate, bank) plus
  cfg      : camera classifier-free guidance on the query cameras: v = v0 + w (v - v0), v0 = same chunk with ZERO camera
             motion (every query pose = pose of the last context latent). w = number, or 'adaptive' = clip(1/g_prev, 1, 3)
             where g_prev = executed/commanded step of the previous call (first call: 1.5).
  redenoise: {'tau': anchor units, 'step': k} after the call, if the chunk-end residual > tau, re-noise the 4 latents to
             sigma_k of the 4-step schedule and denoise again with the over-corrected command c' = (c p^-1) c.
  reproject: {'step': k} after the call, DA3 depth+pose of every generated frame -> forward-warp each frame from its
             measured pose to its commanded pose (z-buffer), composite over the original, VAE-encode, re-noise to sigma_k,
             denoise with the original command. Latent-Reframe mechanism, but triggered by measurement, inside AR + memory.
  bestof   : N -> per call sample N seeds, keep the one with the smallest metric pose error (reward headroom, no training).
  gradguide: {'eta': float} at the last denoising step, x0 -> decode 4 frames -> DA3 -> position error -> gradient step on x0.
"""
import argparse, json, sys, time, math
from pathlib import Path
import numpy as np, torch
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path('/home/hj/Alican/baselines'); sys.path.insert(0, str(ROOT / 'memorizon'))
from memorizon import MemorizonPipeline
from memorizon.model import MemorizonModel
from memorizon.retrieval import Frustum, select_bank
from memorizon.layout import Layout, rope_index, attention_mask
from memorizon.camera import camera_transforms
from memorizon.wan.transformer import frame_mask_to_block_mask
sys.path.insert(0, str(Path(__file__).parent))
from realized_pose import load_da3, DA3_SRC
from gv_memorizon import se3_interp, StreamDecoder, small, BATCH, COMP, W, H, TRAJ_SCALE
sys.path.insert(0, str(Path(__file__).parent / 'finetune'))
from inject import attach_inject, load_trainable      # enables the `inject` kwarg on the transformer forward


def pad44(e):
    e = np.asarray(e, dtype=np.float64)
    if e.shape[-2] == 3:
        o = np.tile(np.eye(4), (e.shape[0], 1, 1)); o[:, :3, :4] = e; return o
    return e


@torch.no_grad()
def da3_window(da3, anchor448, anchor_depth, frames, K, max_views=20, with_depth=False, tail=16):
    """Command-free DA3 on [anchor + frames]; returns per-frame c2w in anchor-depth gauge (+ depth at 448x256)."""
    views = [anchor448] + list(frames)
    if len(views) > max_views:
        tl = list(frames[-tail:]); ctx = list(frames[:-tail])
        nctx = max(0, max_views - 1 - len(tl))
        pick = np.linspace(0, len(ctx) - 1, nctx).round().astype(int) if ctx and nctx else []
        views = [anchor448] + [ctx[i] for i in pick] + tl
        idx = [None] * (len(frames) - tail) + list(range(len(views) - tail, len(views)))
    else:
        idx = list(range(1, len(views)))
    Ks = np.repeat(np.asarray(K, dtype=np.float32)[None], len(views), 0)
    pred = da3.inference(image=views, intrinsics=Ks, extrinsics=None, align_to_input_extrinsics=False,
                         align_to_input_ext_scale=False, infer_gs=False, process_res=504, export_dir=None)
    w2c = pad44(pred.extrinsics); rel = np.stack([w2c[0] @ np.linalg.inv(w2c[i]) for i in range(len(views))])
    d0 = np.asarray(pred.depth[0], dtype=np.float64)
    ours = torch.nn.functional.interpolate(torch.as_tensor(anchor_depth)[None, None].float(), size=d0.shape, mode='bilinear', align_corners=False)[0, 0].numpy()
    ok = np.isfinite(d0) & (d0 > 0) & np.isfinite(ours) & (ours > 0); scale = float(np.median(ours[ok] / d0[ok]))
    rel[:, :3, 3] *= scale
    poses = [None if j is None else rel[j] for j in idx]
    out = dict(poses=poses, scale=scale)
    if with_depth:
        deps = []
        for j in idx:
            if j is None:
                deps.append(None); continue
            d = torch.as_tensor(np.asarray(pred.depth[j], dtype=np.float32))[None, None] * scale
            deps.append(torch.nn.functional.interpolate(d, size=(256, 448), mode='bilinear', align_corners=False)[0, 0].numpy())
        out['depth'] = deps
    return out


def interp_pose(a, b, t):
    d = b @ np.linalg.inv(a); return se3_interp(d, t) @ a


def warp_frame(rgb, depth, K, src_c2w, dst_c2w, radius=1):
    Hh, Ww = depth.shape
    ys, xs = np.mgrid[0:Hh, 0:Ww]
    pix = np.stack([xs, ys, np.ones_like(xs)], 0).reshape(3, -1).astype(np.float64)
    pts = (np.linalg.inv(K) @ pix) * depth.reshape(1, -1)
    rel = np.linalg.inv(dst_c2w) @ src_c2w
    cam = rel[:3, :3] @ pts + rel[:3, 3:4]
    z = cam[2]
    uv = K @ cam
    with np.errstate(divide='ignore', invalid='ignore'):
        projected = uv[:2] / uv[2:3]
    valid = ((depth.reshape(-1) > 0) & np.isfinite(depth.reshape(-1))
             & (z > 1e-3) & np.isfinite(z) & np.isfinite(projected).all(0))
    # Bound projected centers before converting to integer raster coordinates.
    valid &= ((projected[0] >= -radius - 1) & (projected[0] <= Ww + radius)
              & (projected[1] >= -radius - 1) & (projected[1] <= Hh + radius))
    source = np.flatnonzero(valid)
    u, v = np.rint(projected[:, source]).astype(np.int64)
    size = Hh * Ww
    zbuf = np.full(size, np.inf)
    candidates = []
    for du in range(-radius, radius + 1):
        for dv in range(-radius, radius + 1):
            uu, vv = u + du, v + dv
            ok = (uu >= 0) & (uu < Ww) & (vv >= 0) & (vv < Hh)
            src = source[ok]
            dst = vv[ok] * Ww + uu[ok]
            distance = ((uu[ok] - projected[0, src]) ** 2
                        + (vv[ok] - projected[1, src]) ** 2)
            candidates.append((src, dst, distance))
            np.minimum.at(zbuf, dst, z[src])
    distance_buf = np.full(size, np.inf)
    for src, dst, distance in candidates:
        nearest = z[src] == zbuf[dst]
        np.minimum.at(distance_buf, dst[nearest], distance[nearest])
    index_buf = np.full(size, size, dtype=np.int64)
    for src, dst, distance in candidates:
        chosen = (z[src] == zbuf[dst]) & (distance == distance_buf[dst])
        np.minimum.at(index_buf, dst[chosen], src[chosen])
    covered = index_buf < size
    image = rgb.copy().reshape(-1, 3)
    image[covered] = rgb.reshape(-1, 3)[index_buf[covered]]
    return image.reshape(Hh, Ww, 3), covered.reshape(Hh, Ww)


@torch.no_grad()
def encode_chunk(pipe, prev_frame448, frames448):
    """VAE-encode [prev + 16 frames] at 864x480 -> last 4 normalized latents."""
    ims = [np.asarray(Image.fromarray(f).resize((W, H), Image.BICUBIC)) for f in [prev_frame448] + list(frames448)]
    x = torch.from_numpy(np.stack(ims).astype(np.float32) / 127.5 - 1.0).permute(3, 0, 1, 2)[None].to(pipe.device)
    vae = pipe.vae; saved = (vae._feat_map, vae._conv_idx)
    vae.clear_cache()
    with torch.autocast('cuda', torch.bfloat16):
        z = vae.encode(x).latent_dist.mode().float()
    vae.clear_cache(); vae._feat_map, vae._conv_idx = saved
    return ((z - pipe.lat_mean) / pipe.lat_std).to(pipe.dtype)[:, :, -4:]


class Roller:
    @torch.no_grad()
    def __init__(self, pipe, da3, scene, arm, seed, memory_size=6, num_steps=4):
        self.pipe, self.da3, self.arm, self.seed = pipe, da3, arm, seed
        z = np.load(BATCH / 'trajectories' / f'{scene}.npz')
        self.plan = np.linalg.inv(z['w2c'].astype(np.float64))[0:161:4]; self.plan_f = np.linalg.inv(z['w2c'].astype(np.float64))[0:161]
        self.K = z['intrinsics'][0].astype(np.float64)
        stream = json.load(open(BATCH / 'index.json')); sid = [s for s in stream['scenes'] if s['name'] == scene][0]['stream_id']
        self.anchor_depth = np.load(COMP / sid / 'anchor.npz')['depth']
        self.anchor = np.asarray(Image.open(BATCH / 'images' / f'{scene}.png').convert('RGB'))
        self.intr = (self.K[0, 0] * W / 448, self.K[1, 1] * H / 256, self.K[0, 2] * W / 448, self.K[1, 2] * H / 256)
        self.text = pipe.encode_prompt(stream['prompt']).to(pipe.device, pipe.dtype)
        da3.to('cuda'); pm = da3.inference(image=[self.anchor], intrinsics=self.K[None].astype(np.float32), infer_gs=False, process_res=504, export_dir=None); da3.to('cpu'); torch.cuda.empty_cache()
        dm = np.asarray(pm.depth[0]); self.metric = float(np.median(dm[dm > 0])) if pm.is_metric else 1.0
        self.unit = self.metric / TRAJ_SCALE
        self.dit, self.C = pipe.model.model, pipe.model.config.chunk_size
        self.frustum = Frustum.from_intrinsics(list(self.intr)); self.memory_size = memory_size
        first = pipe.encode_image(Image.fromarray(self.anchor).resize((W, H), Image.BICUBIC))
        _, self.ch, _, self.h, self.w = first.shape; _, p_h, p_w = self.dit.config.patch_size; self.tpf = (self.h // p_h) * (self.w // p_w)
        self.sched = pipe.model.noise_scheduler; self.sched.set_timesteps(num_steps); self.timesteps = self.sched.timesteps.to(pipe.device)
        self.intr_b = torch.as_tensor(np.asarray(self.intr, dtype=np.float32))[None].to(pipe.device)
        self.gen = torch.zeros((1, self.ch, 41, self.h, self.w), device=pipe.device, dtype=pipe.dtype); self.gen[:, :, 0] = first[:, :, 0]
        self.rng = torch.Generator(device='cpu').manual_seed(seed)
        self.dec = StreamDecoder(pipe); self.frames = [small(f) for f in self.dec(self.gen[:, :, :1])]
        self.cmd = self.plan.copy(); self.mem = self.plan.copy(); self.measured = [None] * 41; self.measured[0] = np.eye(4)
        if arm.get('cmd_gain'):                            # fixed command amplification control: translations x k (rotation unchanged);
            self.cmd[:, :3, 3] *= float(arm['cmd_gain'])     # planned (scored) path stays unscaled
            self.mem = self.cmd.copy()
        self.residual = [None] * 41; self.excluded = set(); self.events = []; self.g_prev = None; self.last_depth = None

    # ---- one denoising pass of chunk latents [q_lo:q_hi) given cams (memory units), returns 4 latents
    @torch.no_grad()
    def denoise(self, ci, cams_np, x_init=None, start_step=0, noise=None, cfg_w=1.0, anchor_z=None, anchor_m=None, anchor_lam=0.0, init_mask=None, inject=None):
        C, q_lo, q_hi = self.C, 1 + ci * self.C, 1 + (ci + 1) * self.C
        k_mem = 0 if self.arm.get('bank') == 'drop' else self.memory_size
        n_rec = min(C, q_lo); recent = np.arange(q_lo - n_rec, q_lo)
        cand = np.array([i for i in range(1, max(1, q_lo - n_rec)) if i not in self.excluded], dtype=int)
        bank = select_bank(cams_np, cand, cams_np[q_hi - 1][None], self.frustum, k=k_mem, q0=q_lo, chunk=C, cap=max(k_mem, 1)) if k_mem else np.zeros(0, int)
        lay = Layout(n_bank=len(bank), n_recent=n_rec, chunk=C, n_query_chunks=1); F_, n_cond = lay.total, lay.n_cond
        rope = torch.as_tensor(rope_index(lay), dtype=torch.long, device=self.pipe.device)
        mask = (attention_mask(lay, bank, q_lo, cams_np[q_lo:q_hi], cams_np[bank], frustum=self.frustum, k=self.memory_size) if len(bank) else attention_mask(lay))
        mask = torch.as_tensor(mask, dtype=torch.bool, device=self.pipe.device); ctx_mask = frame_mask_to_block_mask(mask[:n_cond, :n_cond], self.tpf)
        idx = np.concatenate([[0], bank, recent]).astype(int)
        cams = torch.as_tensor(np.concatenate([cams_np[idx], cams_np[q_lo:q_hi]]), dtype=torch.float32, device=self.pipe.device)[None]
        cams0 = cams.clone(); cams0[0, n_cond:] = cams[0, n_cond - 1]            # zero-motion query cameras for camera CFG
        x = torch.randn((1, self.ch, F_, self.h, self.w), generator=self.rng).to(self.pipe.device, self.pipe.dtype) if noise is None else noise
        cond = self.gen[:, :, idx]; x[:, :, :n_cond] = cond
        if x_init is not None:
            sig = float(self.sched.sigmas[start_step]); eps = torch.randn(x_init.shape, generator=self.rng).to(x_init)
            init = (1 - sig) * x_init + sig * eps
            if init_mask is not None:                      # holes start from pure noise, not from a copied frame
                init = init_mask.to(init) * init + (1 - init_mask.to(init)) * eps
            x[:, :, n_cond:] = init
        cam_ctx = camera_transforms(self.dit, cams[:, :n_cond], self.intr_b); cam_q = camera_transforms(self.dit, cams[:, n_cond:], self.intr_b)
        cam_q0 = camera_transforms(self.dit, cams0[:, n_cond:], self.intr_b)
        kv = self.dit.init_kv_cache(1, F_, self.h, self.w)
        with torch.autocast('cuda', torch.bfloat16):
            self.dit(cond, torch.zeros((1, n_cond), device=self.pipe.device), self.text, rope_index=rope[:n_cond], attention_mask=ctx_mask, camera=cam_ctx, kv_cache=kv, kv_cache_range=(0, n_cond))
            for layer in kv:
                layer.pop('cross_attn_key', None); layer.pop('cross_attn_value', None)
            for si, t in enumerate(self.timesteps):
                if si < start_step:
                    continue
                ts = t.repeat(1, C).clone()
                pred = self.dit(x[:, :, n_cond:], ts, self.text, rope_index=rope, camera=cam_q, kv_cache=kv, kv_cache_range=(0, F_), inject=inject)
                if cfg_w != 1.0:
                    pred0 = self.dit(x[:, :, n_cond:], ts, self.text, rope_index=rope, camera=cam_q0, kv_cache=kv, kv_cache_range=(0, F_))
                    pred = pred0 + cfg_w * (pred - pred0)
                if anchor_z is not None and anchor_lam > 0:
                    sig = float(self.sched.sigmas[si]); xt = x[:, :, n_cond:].float()
                    x0 = xt - sig * pred.float()
                    x0 = x0 + anchor_lam * anchor_m * (anchor_z.float() - x0)
                    pred = ((xt - x0) / max(sig, 1e-4)).to(pred.dtype)
                last = si == len(self.timesteps) - 1
                if last and self.arm.get('gradguide'):
                    x0 = self.sched.step_diff_noise_level(pred, ts, x[:, :, n_cond:], to_final=True)
                    x0 = self.grad_guide(ci, x0.float()).to(x); x[:, :, n_cond:] = x0
                else:
                    x[:, :, n_cond:] = self.sched.step_diff_noise_level(pred, ts, x[:, :, n_cond:]).to(x)
        out = x[0:1, :, n_cond:].clone(); del kv, x, cond, cam_ctx, cam_q, cam_q0; torch.cuda.empty_cache()
        return out, bank

    def grad_guide(self, ci, x0):
        """One gradient step on the clean chunk latents toward the commanded poses (DA3 differentiable)."""
        q_lo = 1 + ci * self.C; eta = float(self.arm['gradguide']['eta'])
        x0 = x0.detach().requires_grad_(True)
        vae = self.pipe.vae; saved = (vae._feat_map, vae._conv_idx)
        with torch.enable_grad():
            z = vae.post_quant_conv(x0 * self.pipe.lat_std + self.pipe.lat_mean)
            vae.clear_cache(); frames = []
            for k in range(z.shape[2] - 2, z.shape[2]):                       # last two latents only (memory)
                vae._conv_idx = [0]
                o = vae.decoder(z[:, :, k:k + 1], feat_cache=vae._feat_map, feat_idx=vae._conv_idx, first_chunk=(k == z.shape[2] - 2))
                frames.append(o[0, :, -1])                                   # last frame of each latent, [3,H,W] in [-1,1]
            vae.clear_cache(); vae._feat_map, vae._conv_idx = saved
            imgs = torch.stack(frames)                                       # [4,3,H,W]
            imgs = torch.nn.functional.interpolate(imgs, size=(256, 448), mode='bilinear', align_corners=False)
            imgs01 = (imgs.clamp(-1, 1) + 1) / 2
            sys.path.insert(0, DA3_SRC)
            self.da3.to('cuda')
            anchor = torch.from_numpy(self.anchor).permute(2, 0, 1).float().to(imgs.device) / 255.
            batch = torch.cat([anchor[None], imgs01], 0)                     # [3,3,256,448]
            Ks = np.repeat(self.K[None].astype(np.float32), batch.shape[0], 0)
            pred = self.da3.inference(image=batch, intrinsics=Ks, extrinsics=None, align_to_input_extrinsics=False,
                                      align_to_input_ext_scale=False, infer_gs=False, process_res=224, export_dir=None)
            ext = pred.extrinsics
            if not torch.is_tensor(ext) or not ext.requires_grad:
                self.da3.to('cpu'); torch.cuda.empty_cache(); vae._feat_map, vae._conv_idx = saved; self.events.append(dict(kind='gradguide', chunk=ci, skipped='no_grad_path'))
                return x0.detach()
            w2c = ext if ext.shape[-2] == 4 else torch.cat([ext, torch.tensor([[[0, 0, 0, 1.]]], device=ext.device).repeat(ext.shape[0], 1, 1)], 1)
            rel = torch.stack([w2c[0] @ torch.linalg.inv(w2c[i]) for i in range(1, w2c.shape[0])])
            d0 = pred.depth[0]; ours = torch.nn.functional.interpolate(torch.as_tensor(self.anchor_depth)[None, None].float().to(d0.device), size=d0.shape, mode='bilinear', align_corners=False)[0, 0]
            ok = (d0 > 0) & (ours > 0); scale = torch.median(ours[ok] / d0[ok]).detach()
            tgt = torch.as_tensor(np.stack([self.cmd[j] for j in range(q_lo + 2, q_lo + 4)]), dtype=torch.float32, device=rel.device)
            loss = ((rel[:, :3, 3] * scale - tgt[:, :3, 3]) ** 2).sum()
            g, = torch.autograd.grad(loss, x0)
        self.da3.to('cpu'); torch.cuda.empty_cache()
        step = eta * x0.norm() * g / (g.norm() + 1e-8)
        self.events.append(dict(kind='gradguide', chunk=ci, loss=float(loss), step_ratio=float(step.norm() / x0.norm())))
        return (x0 - step).detach()

    @torch.no_grad()
    def measure(self, q_lo, q_hi, with_depth=False):
        lat_frames = [self.frames[4 * j] for j in range(1, q_hi)]
        torch.cuda.empty_cache(); self.da3.to('cuda')
        est = da3_window(self.da3, self.anchor, self.anchor_depth, lat_frames, self.K, max_views=16, tail=4, with_depth=True)
        self.da3.to('cpu'); torch.cuda.empty_cache()
        if est['depth'][-1] is not None:
            self.last_depth = est['depth'][-1]
        for i, j in enumerate(range(q_lo, q_hi)):
            p = est['poses'][j - 1]
            if p is not None and np.isfinite(p).all():
                self.measured[j] = p; self.residual[j] = float(np.linalg.norm(p[:3, 3] - self.cmd[j][:3, 3]))
        return est

    @torch.no_grad()
    def geo_warp(self, ci):
        """Warp the last generated frame (measured pose, DA3 depth) into the 16 commanded poses of chunk ci.
        Returns z_warp [1,48,4,h,w] (holes filled with the source frame), latent mask m [1,1,4,h,w], mean coverage."""
        q_lo, q_hi = 1 + ci * self.C, 1 + (ci + 1) * self.C
        src_idx = 4 * (q_lo - 1); src = self.frames[src_idx]
        src_pose = self.measured[q_lo - 1] if self.measured[q_lo - 1] is not None else self.cmd[q_lo - 1]
        depth = self.anchor_depth if ci == 0 or self.last_depth is None else self.last_depth
        warped, masks = [], []
        for fi in range(src_idx + 1, src_idx + 17):
            j = (fi - 1) // 4 + 1; t = ((fi - 1) % 4 + 1) / 4.0
            c_f = interp_pose(self.cmd[j - 1], self.cmd[j], t)
            wf, m = warp_frame(src, depth, self.K, src_pose, c_f); warped.append(wf); masks.append(m)
        z = encode_chunk(self.pipe, src, warped)
        m = torch.from_numpy(np.stack(masks).astype(np.float32))[None]                      # [1,16,256,448]
        m = torch.nn.functional.adaptive_avg_pool2d(m, (self.h, self.w))[0]                    # [16,h,w]
        m = m.view(4, 4, self.h, self.w).mean(1)[None, None].to(self.pipe.device)              # [1,1,4,h,w]
        return z, m, float(np.mean([mm.mean() for mm in masks]))

    @torch.no_grad()
    def run(self):
        arm, C = self.arm, self.C
        for ci in range(10):
            q_lo, q_hi = 1 + ci * C, 1 + (ci + 1) * C
            if arm.get('replan') and ci > 0 and self.measured[q_lo - 1] is not None:
                delta = self.measured[q_lo - 1] @ np.linalg.inv(self.cmd[q_lo - 1])
                for i, j in enumerate(range(q_lo, q_hi)):
                    self.cmd[j] = se3_interp(delta, 1. - (i + 1) / C) @ self.plan[j]
                self.events.append(dict(kind='replan', chunk=ci, position_error=float(np.linalg.norm(delta[:3, 3]))))
            self.mem[q_lo:q_hi] = self.cmd[q_lo:q_hi]
            cams_np = self.mem.copy(); cams_np[:, :3, 3] *= self.unit
            w = 1.0
            if arm.get('cfg') is not None:
                w = float(arm['cfg']) if arm['cfg'] != 'adaptive' else (1.5 if self.g_prev is None else float(np.clip(1.0 / max(self.g_prev, 1e-3), 1.0, 3.0)))
            # ---- best-of-N: sample N candidate chunks, keep the smallest metric pose error
            if arm.get('bestof'):
                cands = []
                for n in range(int(arm['bestof'])):
                    lat, bank = self.denoise(ci, cams_np, cfg_w=w)
                    saved_frames = len(self.frames)
                    new = [small(f) for f in self.dec(lat)]; self.frames += new
                    est = self.measure(q_lo, q_hi)
                    err = float(np.nanmean([self.residual[j] if self.residual[j] is not None else np.nan for j in range(q_lo, q_hi)]))
                    cands.append((err, lat, new, [self.measured[j] for j in range(q_lo, q_hi)], bank))
                    self.frames = self.frames[:saved_frames]
                    self.dec = StreamDecoder(self.pipe); self.dec(self.gen[:, :, :q_lo])        # rebuild decoder state deterministically
                err, lat, new, meas, bank = min(cands, key=lambda c: c[0])
                self.gen[:, :, q_lo:q_hi] = lat
                # decoder cache must contain the chosen chunk: rebuild through q_lo, decode the chosen latents, re-measure
                self.dec = StreamDecoder(self.pipe); self.dec(self.gen[:, :, :q_lo]); self.frames += [small(f) for f in self.dec(lat)]
                self.measure(q_lo, q_hi)
                self.events.append(dict(kind='bestof', chunk=ci, errors=[round(c[0], 3) for c in cands], chosen=round(err, 3)))
            elif arm.get('hybrid'):
                hy = arm['hybrid']; swap = getattr(self, '_gv_weights', None)
                if swap is not None: self.pipe.model.load_state_dict(swap['stock'], strict=False)
                lat, bank = self.denoise(ci, cams_np, cfg_w=w)
                self.gen[:, :, q_lo:q_hi] = lat; self.frames += [small(f) for f in self.dec(lat)]
                self.measure(q_lo, q_hi); before = self.residual[q_hi - 1]
                # refine: warp the just-generated chunk from its measured poses to the commanded poses, re-denoise with the
                # fine-tuned weights from sigma_k, injection = that warp (what the fine-tuned model was trained to follow)
                torch.cuda.empty_cache(); self.da3.to('cuda')
                est = da3_window(self.da3, self.anchor, self.anchor_depth, self.frames[1:4 * (q_hi - 1) + 1], self.K, max_views=20, with_depth=True)
                self.da3.to('cpu'); torch.cuda.empty_cache()
                f0 = 4 * (q_lo - 1) + 1; warped = []; cover = []
                for fi in range(f0, 4 * (q_hi - 1) + 1):
                    p = est['poses'][fi - 1]; d = est['depth'][fi - 1]; j = (fi - 1) // 4 + 1; t = ((fi - 1) % 4 + 1) / 4.0
                    c_f = interp_pose(self.cmd[j - 1], self.cmd[j], t)
                    if p is None or d is None:
                        warped.append(self.frames[fi]); cover.append(0.); continue
                    wf, cv = warp_frame(self.frames[fi], d, self.K, p, c_f); warped.append(wf); cover.append(float(cv.mean()))
                z_w = encode_chunk(self.pipe, self.frames[f0 - 1], warped)
                if swap is not None: self.pipe.model.load_state_dict(swap['ft'], strict=False)
                lat, _ = self.denoise(ci, cams_np, x_init=z_w, start_step=int(hy.get('step', 2)), cfg_w=w, inject=z_w)
                if swap is not None: self.pipe.model.load_state_dict(swap['stock'], strict=False)
                self.gen[:, :, q_lo:q_hi] = lat; self.frames = self.frames[:f0]
                self.dec = StreamDecoder(self.pipe); self.dec(self.gen[:, :, :q_lo]); self.frames += [small(f) for f in self.dec(lat)]
                self.measure(q_lo, q_hi)
                self.events.append(dict(kind='hybrid', chunk=ci, coverage=float(np.mean(cover)), residual_before=before, residual_after=self.residual[q_hi - 1]))
            elif arm.get('inject'):
                z_w, m_lat, cov = self.geo_warp(ci)
                lat, bank = self.denoise(ci, cams_np, cfg_w=w, inject=z_w)
                self.gen[:, :, q_lo:q_hi] = lat; self.frames += [small(f) for f in self.dec(lat)]
                self.measure(q_lo, q_hi)
                self.events.append(dict(kind='geo', chunk=ci, coverage=cov, residual_end=self.residual[q_hi - 1]))
            elif arm.get('geoinit') or arm.get('geosoft'):
                z_w, m_lat, cov = self.geo_warp(ci)
                gi = arm.get('geoinit') or {}; gs = arm.get('geosoft') or {}
                hard = (m_lat > float(gi.get('mask_thresh', 0.0))).float() if gi.get('holes') == 'noise' else None
                lat, bank = self.denoise(ci, cams_np, cfg_w=w, x_init=(z_w if gi else None), start_step=int(gi.get('step', 0)) if gi else 0,
                                         anchor_z=(z_w if gs else None), anchor_m=(m_lat if gs else None), anchor_lam=float(gs.get('lam', 0.0)) if gs else 0.0,
                                         init_mask=hard)
                self.gen[:, :, q_lo:q_hi] = lat; self.frames += [small(f) for f in self.dec(lat)]
                self.measure(q_lo, q_hi)
                self.events.append(dict(kind='geo', chunk=ci, coverage=cov, residual_end=self.residual[q_hi - 1]))
            else:
                lat, bank = self.denoise(ci, cams_np, cfg_w=w)
                self.gen[:, :, q_lo:q_hi] = lat; self.frames += [small(f) for f in self.dec(lat)]
                self.measure(q_lo, q_hi)
            # ---- refinement after the call
            if arm.get('redenoise') and self.residual[q_hi - 1] is not None and self.residual[q_hi - 1] > float(arm['redenoise']['tau']):
                corr = self.cmd[q_lo:q_hi].copy()
                for i, j in enumerate(range(q_lo, q_hi)):
                    if self.measured[j] is not None:
                        corr[i] = (self.cmd[j] @ np.linalg.inv(self.measured[j])) @ self.cmd[j]
                cams2 = cams_np.copy(); cams2[q_lo:q_hi] = corr; cams2[q_lo:q_hi, :3, 3] *= self.unit
                before = self.residual[q_hi - 1]
                lat, _ = self.denoise(ci, cams2, x_init=self.gen[:, :, q_lo:q_hi].clone(), start_step=int(arm['redenoise']['step']), cfg_w=w)
                self.gen[:, :, q_lo:q_hi] = lat; self.frames = self.frames[:4 * (q_lo - 1) + 1]
                self.dec = StreamDecoder(self.pipe); self.dec(self.gen[:, :, :q_lo]); self.frames += [small(f) for f in self.dec(lat)]
                self.measure(q_lo, q_hi)
                self.events.append(dict(kind='redenoise', chunk=ci, residual_before=before, residual_after=self.residual[q_hi - 1]))
            if arm.get('reproject'):
                torch.cuda.empty_cache(); self.da3.to('cuda')
                est = da3_window(self.da3, self.anchor, self.anchor_depth, self.frames[1:4 * (q_hi - 1) + 1], self.K, max_views=20, with_depth=True)
                self.da3.to('cpu'); torch.cuda.empty_cache()
                f0 = 4 * (q_lo - 1) + 1; warped = []; cover = []
                for fi in range(f0, 4 * (q_hi - 1) + 1):
                    p = est['poses'][fi - 1]; d = est['depth'][fi - 1]
                    j = (fi - 1) // 4 + 1; t = ((fi - 1) % 4 + 1) / 4.0
                    c_f = interp_pose(self.cmd[j - 1] if j - 1 >= 0 else np.eye(4), self.cmd[j], t)
                    if p is None or d is None:
                        warped.append(self.frames[fi]); cover.append(0.); continue
                    wf, cv = warp_frame(self.frames[fi], d, self.K, p, c_f); warped.append(wf); cover.append(float(cv.mean()))
                lat0 = encode_chunk(self.pipe, self.frames[f0 - 1], warped)
                before = self.residual[q_hi - 1]
                lat, _ = self.denoise(ci, cams_np, x_init=lat0, start_step=int(arm['reproject']['step']), cfg_w=w)
                self.gen[:, :, q_lo:q_hi] = lat; self.frames = self.frames[:f0]
                self.dec = StreamDecoder(self.pipe); self.dec(self.gen[:, :, :q_lo]); self.frames += [small(f) for f in self.dec(lat)]
                self.measure(q_lo, q_hi)
                self.events.append(dict(kind='reproject', chunk=ci, coverage=float(np.mean(cover)), residual_before=before, residual_after=self.residual[q_hi - 1]))
            # ---- write / gate / gain bookkeeping
            for j in range(q_lo, q_hi):
                if self.measured[j] is not None:
                    if arm.get('write') == 'measured':
                        self.mem[j] = self.measured[j]
                    if arm.get('gate') is not None and self.residual[j] is not None and self.residual[j] > float(arm['gate']):
                        self.excluded.add(j)
            a, b = self.measured[q_lo - 1], self.measured[q_hi - 1]
            if a is not None and b is not None:
                self.g_prev = float(np.linalg.norm(b[:3, 3] - a[:3, 3]) / max(np.linalg.norm(self.cmd[q_hi - 1][:3, 3] - self.cmd[q_lo - 1][:3, 3]), 1e-6))
            self.events.append(dict(kind='chunk', chunk=ci, cfg_w=w, g=self.g_prev, bank=[int(x) for x in bank],
                                    residual=[self.residual[j] for j in range(q_lo, q_hi)]))
        return np.stack(self.frames), dict(metric_anchor_depth_m=self.metric, events=self.events, commanded=self.cmd.tolist(),
                                            planned=self.plan.tolist(), measured=[None if m is None else m.tolist() for m in self.measured])


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True); p.add_argument('--arms', type=Path, required=True)
    p.add_argument('--scenes', nargs='+', default=['1740e916', '38dcc531']); p.add_argument('--seeds', nargs='+', type=int, default=[7, 11, 19])
    p.add_argument('--ckpt', default=None, help='trainable weights (camera branch + inject) from finetune/train_single.py')
    p.add_argument('--steps', type=int, default=4, help='sampler steps (Memorizon default 4)')
    p.add_argument('--save_latents', action='store_true', help='also save the generated latents [1,48,41,h,w] fp16 (<run>.latents.npy)'); a = p.parse_args()
    arms = json.load(open(a.arms)); a.out.mkdir(parents=True, exist_ok=True)
    pipe = MemorizonPipeline(MemorizonModel.from_pretrained(str(ROOT / 'models/memorizon_local'))); da3 = load_da3('cpu')
    weights = None
    if a.ckpt:
        attach_inject(pipe.model)
        stock = {n: p.detach().clone() for n, p in pipe.model.named_parameters() if ('prope_attn' in n or 'inject_embedding' in n)}
        load_trainable(pipe.model, a.ckpt); pipe.model.eval(); print('loaded trainable weights', a.ckpt, flush=True)
        ft = {n: p.detach().clone() for n, p in pipe.model.named_parameters() if ('prope_attn' in n or 'inject_embedding' in n)}
        weights = dict(stock=stock, ft=ft)
    for arm in arms:
        for scene in a.scenes:
            for seed in a.seeds:
                dst = a.out / arm['name'] / f'{scene}_s{seed}.npz'
                if dst.exists():
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True); t0 = time.time()
                try:
                    roller = Roller(pipe, da3, scene, arm, seed, num_steps=a.steps)
                    if weights is not None and arm.get('hybrid'):
                        roller._gv_weights = weights
                    elif weights is not None:
                        pipe.model.load_state_dict(weights['ft'], strict=False)
                    frames, log = roller.run()
                except Exception as e:
                    import traceback; tb = traceback.format_exc().strip().splitlines()
                    print(json.dumps(dict(arm=arm['name'], scene=scene, seed=seed, error=repr(e)[:200], where=[l.strip() for l in tb if 'gv_memorizon2.py' in l][-2:],
                                          peak_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 1))), flush=True); torch.cuda.empty_cache(); continue
                np.savez_compressed(dst, frames=frames[1:161])
                if a.save_latents:
                    np.save(dst.with_suffix('.latents.npy'), roller.gen.float().cpu().numpy().astype(np.float16))
                json.dump(dict(arm=arm, scene=scene, seed=seed, seconds=time.time() - t0, **log), open(dst.with_suffix('.json'), 'w'))
                res = [r for e in log['events'] if e['kind'] == 'chunk' for r in e['residual'] if r is not None]
                print(json.dumps(dict(arm=arm['name'], scene=scene, seed=seed, sec=round(time.time() - t0, 1), residual_end=round(res[-1], 3) if res else None,
                                      g_last=round(log['events'][-1].get('g') or 0, 2))), flush=True)


if __name__ == '__main__':
    main()
