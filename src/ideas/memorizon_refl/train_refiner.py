"""Trained refiner for released Memorizon: learn the refinement pass of the 'hybrid' arm (gv_memorizon2.py).

Inference (unchanged harness, arm {"hybrid": {"step": 2}}): stock weights generate a chunk; DA3 measures every generated
frame's pose + depth; each frame is warped from its measured pose to its commanded pose (holes keep the source pixels);
the composite is VAE-encoded (z_w), re-noised to sigma_2 of the 4-step schedule and denoised for the last 2 steps with
the REFINER weights (camera branch + zero-init inject_embedding, inject = z_w).
Training pairs from recordings ("delayed recorded frames"): for target frame f of a chunk, the source is the recorded
frame s(f) = f_prev + round(rho (f - f_prev)), rho ~ U(0.2, 1) per chunk (rho ~ 0.3 = Memorizon's measured execution),
warped from its recorded pose to f's recorded pose with its own DA3 depth, at 448x256 like inference. Target = recorded
latents of the chunk. Context = recorded anchor/bank/recent (LongSpanDataset, one query chunk).
Loss on the refiner's own 2-step output (last step with gradient): L1 to the recorded latents + adv * hinge latent
discriminator (refl_camera.Disc). No flow-matching loss (that blurred the student).
Init: --init (default finetune/run_inject/trainable_02000.pt, the FM-trained inject weights whose hybrid eval gave
pos_err 0.39 / ratio 0.92 but LPIPS 0.61).
"""
import argparse, json, random, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
HERE = Path(__file__).parent; PLUG = HERE.parent
sys.path.insert(0, '/home/hj/Alican/baselines/memorizon'); sys.path.insert(0, str(PLUG / 'finetune')); sys.path.insert(0, str(PLUG)); sys.path.insert(0, str(HERE))
from memorizon import MemorizonPipeline
from memorizon.model import MemorizonModel
from memorizon.data import LongSpanDataset
from memorizon.camera import camera_transforms
from memorizon.wan.scheduler import FlowMatchScheduler
from inject import attach_inject, load_trainable, save_trainable
from realized_pose import load_da3
from gv_memorizon2 import warp_frame, encode_chunk
from build_shards import crop_resize
from refl_camera import Disc, checkpoint_blocks
SHARDS = PLUG / 'finetune' / 'shards300'


class Pairs:
    def __init__(self, da3):
        self.da3, self.cache, self.depth = da3, {}, {}

    def aux(self, ep):
        if ep not in self.cache:
            z = np.load(SHARDS / 'aux' / f'{ep}.npz')
            self.cache[ep] = dict(c2w=z['c2w_frame'].astype(np.float64), K=z['K448'].astype(np.float64), frames=list(z['frames']),
                                  metric=float(z['metric_anchor_depth_m']))
        return self.cache[ep]

    def frame(self, a, i):
        return np.asarray(crop_resize(Image.open(a['frames'][i]).convert('RGB'), 448, 256)[0])

    @torch.no_grad()
    def depth_of(self, ep, a, i, img):
        key = (ep, i)
        if key not in self.depth:
            pm = self.da3.inference(image=[img], intrinsics=a['K'][None].astype(np.float32), infer_gs=False, process_res=504, export_dir=None)
            d = torch.nn.functional.interpolate(torch.as_tensor(np.asarray(pm.depth[0], np.float32))[None, None], size=(256, 448), mode='bilinear', align_corners=False)[0, 0].numpy()
            self.depth[key] = d / a['metric']                                      # metric -> anchor-depth gauge
            if len(self.depth) > 4000:
                self.depth.pop(next(iter(self.depth)))
        return self.depth[key]

    def __call__(self, pipe, ep, q0, rho):
        """Warp composite latent for target latents q0..q0+3 -> [1,48,4,h,w] normalized, plus mean coverage."""
        a = self.aux(ep); fprev = 4 * (q0 - 1); warped, cover, imgs = [], [], {}
        for f in range(fprev + 1, fprev + 17):
            s = fprev + int(round(rho * (f - fprev)))
            if s not in imgs:
                imgs[s] = self.frame(a, s)
            if s == f:
                warped.append(imgs[s]); cover.append(1.0); continue
            d = self.depth_of(ep, a, s, imgs[s])
            wf, cv = warp_frame(imgs[s], d, a['K'], a['c2w'][s], a['c2w'][f]); warped.append(wf); cover.append(float(cv.mean()))
        prev = imgs[fprev] if fprev in imgs else self.frame(a, fprev)
        return encode_chunk(pipe, prev, warped), float(np.mean(cover))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True); ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--lr', type=float, default=1e-5); ap.add_argument('--inject_lr', type=float, default=1e-4)
    ap.add_argument('--adv', type=float, default=0.05); ap.add_argument('--d_lr', type=float, default=1e-4); ap.add_argument('--start_step', type=int, default=2)
    ap.add_argument('--init', default=str(PLUG / 'finetune' / 'run_inject' / 'trainable_02000.pt')); ap.add_argument('--save_every', type=int, default=500)
    ap.add_argument('--l1_pool', type=int, default=1, help='L1 on k x k average-pooled latents: only low frequencies regressed (pixel-level blur source), detail left to the discriminator')
    ap.add_argument('--seed', type=int, default=0); a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    json.dump({k: str(v) for k, v in vars(a).items()}, open(a.out / 'args.json', 'w'), indent=1)
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    pipe = MemorizonPipeline(MemorizonModel.from_pretrained('/home/hj/Alican/baselines/models/memorizon_local'))
    pipe.vae.clear_cache()                                                     # creates vae._feat_map (encode_chunk saves/restores it)
    model = attach_inject(pipe.model)
    if a.init and a.init != 'none':
        load_trainable(model, a.init)
    with torch.no_grad():
        model.model.inject_embedding.bias.zero_()                              # eval loads weights only (bias stays 0 there)
    params = []
    for n, p in model.named_parameters():
        want = ('prope_attn' in n) or (n.endswith('inject_embedding.weight'))       # bias frozen at 0: zero inject on context adds nothing
        p.requires_grad_(want)
        if want:
            params.append((n, p))
    master = [p.detach().float().clone().requires_grad_(True) for _, p in params]
    opt = torch.optim.AdamW([dict(params=[m for (n, _), m in zip(params, master) if 'inject' not in n], lr=a.lr),
                             dict(params=[m for (n, _), m in zip(params, master) if 'inject' in n], lr=a.inject_lr)], betas=(0.9, 0.99), weight_decay=0.0)
    checkpoint_blocks(model.model); model.train()
    disc = Disc().cuda(); d_opt = torch.optim.AdamW(disc.parameters(), lr=a.d_lr, betas=(0.0, 0.99))
    sched = FlowMatchScheduler(shift=model.config.scheduler_shift, sigma_min=0.0, extra_one_step=True); sched.set_timesteps(4)
    da3 = load_da3('cuda'); pairs = Pairs(da3); C = 4
    ds = LongSpanDataset(str(SHARDS), chunk=C, n_query_chunks=1, m_min=1, m_max=9, topk_per_chunk=6, traj_scale=4.0, dtype=torch.bfloat16)
    man = json.load(open('/home/hj/Alican/analysis/plan_base_restart_20261008/camera_quality_20261015/manifest.json'))
    ds.episode_ids = [e for e in ds.episode_ids if e not in man['shards']]          # camera-quality manifest (window level)
    draw = ds._draw_span; last = {}
    def spy(*args, **kw):
        out = draw(*args, **kw); last['target'] = out[3]; return out
    ds._draw_span = spy
    print(f'[refiner] {len(ds)} episodes, trainable {sum(p.numel() for _, p in params) / 1e6:.1f}M, start_step={a.start_step}', flush=True)
    log = open(a.out / 'train.log', 'a'); t0 = time.time(); step = 0; skips = 0
    while step < a.steps:
        i = random.randrange(len(ds)); ep = ds.episode_ids[i]
        try:
            s = ds.getitem(i); q0 = int(last['target'][0]); rho = random.uniform(0.2, 1.0)
            z_w, cov = pairs(pipe, ep, q0, rho)
        except Exception as e:
            print('skip', ep, repr(e)[:160], flush=True); skips += 1
            if skips > 50:
                raise RuntimeError('more than 50 consecutive skipped samples') from e
            continue
        skips = 0
        lat = s['latents'][None].to('cuda', torch.bfloat16); n = int(s['n_cond']); Fn = lat.shape[2]
        c2w = s['c2w'][None].to('cuda', torch.float32); intr = s['intrinsics'][None].to('cuda', torch.float32)
        text = s['text_embedding'][None].to('cuda', torch.bfloat16); mask = s['attention_mask'].to('cuda'); rope = s['rope_index'].to('cuda')
        cam = camera_transforms(model.model, c2w, intr)
        inject = torch.zeros_like(lat); inject[:, :, n:] = z_w.to(lat)
        with torch.no_grad():
            for (_, p), m in zip(params, master):
                p.copy_(m.to(p.dtype))
            sig = float(sched.sigmas[a.start_step]); x = ((1 - sig) * z_w.float() + sig * torch.randn_like(z_w.float())).to(torch.bfloat16)
        for si in range(a.start_step, len(sched.timesteps)):
            ts = torch.zeros((1, Fn), device='cuda'); ts[:, n:] = float(sched.timesteps[si]); final = si == len(sched.timesteps) - 1
            with torch.set_grad_enabled(final), torch.autocast('cuda', torch.bfloat16):      # as Memorizon training (HF bf16 autocast)
                flow = model.model(torch.cat([lat[:, :, :n], x], 2), ts, text, rope_index=rope, attention_mask=mask, camera=cam, inject=inject)[:, :, n:]
                if final:
                    x0 = sched.step_diff_noise_level(flow.float(), ts[:, n:], x.float(), to_final=True)
                else:
                    x = sched.step_diff_noise_level(flow, ts[:, n:], x).to(torch.bfloat16)
        tgt = lat[:, :, n:].float(); fake = x0[0].permute(1, 0, 2, 3)
        d_loss = torch.zeros(())
        if a.adv > 0:
            with torch.autocast('cuda', torch.bfloat16):
                d_loss = F.relu(1 - disc(tgt[0].permute(1, 0, 2, 3))).mean() + F.relu(1 + disc(fake.detach())).mean()
            d_opt.zero_grad(); d_loss.backward(); d_opt.step()
        pool = lambda z: F.avg_pool2d(z[0].permute(1, 0, 2, 3), a.l1_pool) if a.l1_pool > 1 else z
        l1 = (pool(x0) - pool(tgt)).abs().mean(); loss = l1; g_adv = torch.zeros((), device='cuda')
        if a.adv > 0 and step >= 50:
            with torch.autocast('cuda', torch.bfloat16):
                g_adv = -disc(fake).float().mean()
            loss = loss + a.adv * g_adv
        loss.backward(); disc.zero_grad(set_to_none=True)
        with torch.no_grad():
            for (_, p), m in zip(params, master):
                m.grad = None if p.grad is None else p.grad.float(); p.grad = None
            l1_in = float((z_w.float() - tgt).abs().mean())                       # error of the warp composite itself
        gn = torch.nn.utils.clip_grad_norm_(master, 1.0); opt.step(); opt.zero_grad(set_to_none=True); step += 1
        rec = dict(step=step, ep=ep[:8], rho=round(rho, 2), cover=round(cov, 3), l1=round(float(l1), 4), l1_warp=round(l1_in, 4),
                   adv_g=round(float(g_adv), 3), d_loss=round(float(d_loss), 3), grad=round(float(gn), 4), F=Fn,
                   sec=round(time.time() - t0, 1), peak=round(torch.cuda.max_memory_allocated() / 2**30, 1))
        log.write(json.dumps(rec) + '\n'); log.flush()
        if step % 10 == 0 or step <= 3:
            print(json.dumps(rec), flush=True)
        if step % a.save_every == 0 or step == a.steps:
            with torch.no_grad():
                for (_, p), m in zip(params, master):
                    p.copy_(m.to(p.dtype))
            save_trainable(model, a.out / f'trainable_{step:05d}.pt'); torch.save(disc.state_dict(), a.out / 'disc.pt')


if __name__ == '__main__':
    main()
