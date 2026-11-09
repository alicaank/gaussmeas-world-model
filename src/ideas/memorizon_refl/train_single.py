"""Single-5090 fine-tune of Memorizon's camera branch + zero-init geometric injection (venv_memorizon).

Backbone frozen (bf16). Trainable: prope_attn (142M) + inject_embedding, fp32 master weights, AdamW, grad checkpointing
per block. Loss = Memorizon's diffusion-forcing FM loss on the target chunks. Sample = Memorizon LongSpanDataset on our
shards (4 target chunks = 16 frames, bank <= 6, recent 4), cameras anchor-relative in metres/4.
Injection per target latent: the RECORDED frame ending the previous latent, DA3 depth, dense z-buffer warp into the 4
frames of the target latent (per-frame recorded poses), VAE-encoded -> warp latent (zeros when --inject off).
Checkpoints: trainable weights only.
"""
import argparse, json, random, sys, time, math
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, '/home/hj/Alican/baselines/memorizon'); sys.path.insert(0, str(Path(__file__).parent.parent)); sys.path.insert(0, str(Path(__file__).parent))
from memorizon import MemorizonPipeline
from memorizon.model import MemorizonModel
from memorizon.data import LongSpanDataset
from memorizon.camera import camera_transforms
from memorizon.wan import transformer as T
from inject import attach_inject, trainable_parameters, save_trainable, load_trainable
from realized_pose import load_da3
from gv_memorizon2 import warp_frame
from build_shards import crop_resize
W, H = 864, 480


class Warper:
    """Recorded-frame warp latents for target latents (teacher-forced rows)."""
    def __init__(self, pipe, da3, shards):
        self.pipe, self.da3, self.shards = pipe, da3, Path(shards); self.cache = {}

    def aux(self, ep):
        if ep not in self.cache:
            z = np.load(self.shards / 'aux' / f'{ep}.npz'); self.cache[ep] = dict(c2w=z['c2w_frame'], K864=z['K864'], frames=list(z['frames']), metric=float(z['metric_anchor_depth_m']))
        return self.cache[ep]

    @torch.no_grad()
    def __call__(self, ep, target_latents):
        """target_latents: absolute latent indices (chunks of 4). For each chunk, the source is the recorded frame that
        ends the latent BEFORE the chunk (frame 4*(j0-1)), warped into all 16 frames of the chunk with their recorded
        poses, then VAE-encoded causally [src + 16] -> last 4 latents. Identical to inference-time geo_warp, which only
        has the last generated frame before the chunk. Returns [48, len, h, w] normalized warp latents."""
        a = self.aux(ep); out = []
        Kx = a['K864']; K = np.array([[Kx[0], 0, Kx[2]], [0, Kx[1], Kx[3]], [0, 0, 1]], np.float64)
        for k in range(0, len(target_latents), 4):
            j0 = target_latents[k]; src_f = 4 * (j0 - 1)
            src = np.asarray(crop_resize(Image.open(a['frames'][src_f]).convert('RGB'))[0])
            pm = self.da3.inference(image=[src], intrinsics=K[None].astype(np.float32), infer_gs=False, process_res=504, export_dir=None)
            d = torch.nn.functional.interpolate(torch.as_tensor(np.asarray(pm.depth[0], np.float32))[None, None], size=(H, W), mode='bilinear', align_corners=False)[0, 0].numpy()
            d = d / a['metric']                                                            # DA3 metric -> gauge units (anchor depth = 1)
            warped = [warp_frame(src, d, K, a['c2w'][src_f], a['c2w'][f])[0] for f in range(src_f + 1, src_f + 17)]
            x = torch.from_numpy(np.stack([src] + warped).astype(np.float32) / 127.5 - 1).permute(3, 0, 1, 2)[None].to('cuda')
            vae = self.pipe.vae; vae.clear_cache()
            with torch.autocast('cuda', torch.bfloat16):
                z = vae.encode(x).latent_dist.mode().float()
            vae.clear_cache()
            out.append(((z - self.pipe.lat_mean) / self.pipe.lat_std)[0, :, -4:])
        return torch.cat(out, 1)


def checkpoint_blocks(dit):
    import torch.utils.checkpoint as cp
    for blk in dit.blocks:
        f = blk.forward
        def wrapped(*args, _f=f, **kw):
            return cp.checkpoint(lambda *a: _f(*a, **kw), *args, use_reentrant=False)
        blk.forward = wrapped


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--shards', required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=2000); ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--inject', type=int, default=1)
    ap.add_argument('--save_every', type=int, default=250); ap.add_argument('--resume', default=None); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_query_chunks', type=int, default=4); ap.add_argument('--inject_dropout', type=float, default=0.0)
    ap.add_argument('--inject_lr', type=float, default=None); a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); random.seed(a.seed)
    pipe = MemorizonPipeline(MemorizonModel.from_pretrained('/home/hj/Alican/baselines/models/memorizon_local'))
    model = attach_inject(pipe.model); model.train()
    if a.resume:
        load_trainable(model, a.resume)
    params = trainable_parameters(model); names = [n for n, _ in params]
    master = [p.detach().float().clone().requires_grad_(True) for _, p in params]
    groups = [dict(params=[m for (n, _), m in zip(params, master) if 'inject' not in n], lr=a.lr),
              dict(params=[m for (n, _), m in zip(params, master) if 'inject' in n], lr=a.inject_lr if a.inject_lr is not None else a.lr)]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.01)
    checkpoint_blocks(model.model)
    da3 = load_da3('cuda'); warper = Warper(pipe, da3, a.shards)
    ds = LongSpanDataset(a.shards, chunk=4, n_query_chunks=a.n_query_chunks, m_min=a.n_query_chunks - 1, m_max=9, topk_per_chunk=6, traj_scale=4.0, dtype=torch.bfloat16)
    print(f'[data] {len(ds)} episodes; trainable {sum(p.numel() for _, p in params)/1e6:.1f}M', flush=True)
    log = open(a.out / 'train.log', 'a'); step = 0; t0 = time.time(); ema = None
    if a.resume and (a.out / 'state.json').exists():
        step = json.load(open(a.out / 'state.json'))['step']
    while step < a.steps:
        i = random.randrange(len(ds)); ep = ds.episode_ids[i]
        try:
            s = ds.getitem(i)
        except Exception as e:
            print('skip', ep, repr(e)[:120], flush=True); continue
        lat = s['latents'][None].to('cuda'); c2w = s['c2w'][None].to('cuda'); intr = s['intrinsics'][None].to('cuda')
        n_cond = int(s['n_cond']); Tn = lat.shape[2]
        inject = torch.zeros((1, 48, Tn, lat.shape[3], lat.shape[4]), device='cuda')
        if a.inject and random.random() >= a.inject_dropout:
            # absolute latent indices of the targets: the span starts at 0 (cond_stride = episode), targets are the last 4*nq
            d = torch.load(Path(a.shards) / 'latents' / f'{ep}.pt', map_location='cpu'); n_all = d['latents'].shape[1]
            tgt_abs = list(range(n_all - (Tn - n_cond), n_all)) if False else None
            # recover absolute indices from the sample's cameras: match c2w of targets against the episode's latent cameras
            ep_c2w = d['c2w_latent'].numpy().astype(np.float64); anchor_inv = None
            tgt_rel = s['c2w'][n_cond:].numpy().astype(np.float64)
            # the sample's cameras are relative to the first target; find the episode index whose relative chain matches
            best = None
            for q0 in range(1, n_all - (Tn - n_cond) + 1):
                rel = np.linalg.inv(ep_c2w[q0]) @ ep_c2w[q0:q0 + (Tn - n_cond)]
                err = np.abs(rel - tgt_rel).max()
                if best is None or err < best[0]:
                    best = (err, q0)
            q0 = best[1]
            try:
                inj = warper(ep, list(range(q0, q0 + (Tn - n_cond))))
                inject[:, :, n_cond:] = inj[None].to('cuda')
            except Exception as e:
                print('warp failed', ep, repr(e)[:120], flush=True)
        # copy master -> model params (bf16)
        with torch.no_grad():
            for (_, p), m in zip(params, master):
                p.copy_(m.to(p.dtype))
        # forward: Memorizon's loss with inject threaded through the transformer
        B, _, Tn, _, _ = lat.shape; cs = 4; n_chunks = (Tn - n_cond) // cs; clean = model.config.num_train_timesteps
        ids = torch.randint(clean, size=(B, n_chunks)).repeat_interleave(cs, dim=-1)
        ids = torch.cat([torch.full((B, n_cond), clean), ids], dim=1).to('cuda')
        noise = torch.randn_like(lat.float()); timesteps = model.timesteps[ids]
        is_target = (ids < clean).view(B, 1, Tn, 1, 1)
        noisy = model.noise_scheduler.batch_add_frame_noise(lat.float(), noise, timesteps)
        noisy = torch.where(is_target, noisy, lat.float()).to(model.dtype)
        target = model.noise_scheduler.training_target(lat.float(), noise, timesteps)
        weights = model.noise_scheduler.batch_frame_training_weight(timesteps).to('cuda')
        with torch.autocast('cuda', torch.bfloat16):
            pred = model.model(noisy, timesteps, s['text_embedding'][None].to('cuda', model.dtype), rope_index=s['rope_index'].to('cuda'),
                               attention_mask=s['attention_mask'].to('cuda'), camera=camera_transforms(model.model, c2w, intr), inject=inject)
        loss = (pred.float() - target) ** 2
        loss = (weights.view(B, 1, Tn, 1, 1) * loss)[is_target.broadcast_to(loss.shape)].mean()
        loss.backward()
        with torch.no_grad():
            for (_, p), m in zip(params, master):
                m.grad = None if p.grad is None else p.grad.float(); p.grad = None
        gn = torch.nn.utils.clip_grad_norm_(master, 1.0); opt.step(); opt.zero_grad(set_to_none=True)
        step += 1; v = float(loss); ema = v if ema is None else 0.98 * ema + 0.02 * v
        rec = dict(step=step, loss=round(v, 5), ema=round(ema, 5), grad=round(float(gn), 4), T=Tn, n_cond=n_cond, sec=round(time.time() - t0, 1), peak=round(torch.cuda.max_memory_allocated() / 2**30, 1))
        log.write(json.dumps(rec) + '\n'); log.flush()
        if step % 10 == 0:
            print(json.dumps(rec), flush=True)
        if step % a.save_every == 0 or step == a.steps:
            with torch.no_grad():
                for (_, p), m in zip(params, master):
                    p.copy_(m.to(p.dtype))
            save_trainable(model, a.out / f'trainable_{step:05d}.pt'); json.dump(dict(step=step), open(a.out / 'state.json', 'w'))


if __name__ == '__main__':
    main()
