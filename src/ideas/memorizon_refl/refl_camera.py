"""Camera-ReFL on the released 4-step Memorizon student (venv_memorizon, one 5090).

Why not flow matching: FM regresses the conditional mean at a random t and blurred the DMD student (no-inject control,
20-step eval, dropout + low-LR run). Here every loss is computed on the student's OWN few-step output, so the few-step
behaviour is what gets optimised (the Self-Forcing / DMD principle).

Rollout: Memorizon's own Self-Forcing rollout (memorizon/self_forcing.py, reused unchanged): a LongSpanDataset sample
(recorded anchor + bank + recent chunk), then n_query_chunks target chunks, each generated from the chunks generated
before it (own history from chunk 2 on), always the full 4-step schedule. The last step of
every chunk is recomputed with gradient (_generator_x0, one block-diagonal forward) -> x0 with grad.
Losses on x0 (per target latent):
  camera  frozen latent pose head (refl/head864/pose_head864.pt) on (previous latent, generated latent) vs the commanded
          relative pose, anchor-depth gauge: |t_pred - t_cmd| + geodesic(R_pred, R_cmd) [rad]
  adv     (--adv > 0) unconditional latent discriminator, hinge; real = recorded target latents of the same sample.
          Keeps outputs on the sharp data manifold without pinning them to one output (APT/LADD-style), so it does not
          fight the camera change the way same-noise regression would. D trains alone for the first 50 steps.
Trainable: camera branch only (prope_attn, 142M), fp32 masters, AdamW. Checkpoints load with gv_memorizon2.py --ckpt.
"""
import argparse, json, random, sys, time, types
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
HERE = Path(__file__).parent; PLUG = HERE.parent
sys.path.insert(0, '/home/hj/Alican/baselines/memorizon'); sys.path.insert(0, str(PLUG / 'finetune'))
sys.path.insert(0, '/home/hj/Alican/analysis/plan_base_restart_20261008/cfg_res_20261015')
from memorizon.model import MemorizonModel
from memorizon.data import LongSpanDataset
from memorizon.self_forcing import SelfForcingModel
from memorizon.wan.scheduler import FlowMatchScheduler
from inject import trainable_parameters, save_trainable
from train_pose_head import Head, geodesic
SHARDS = PLUG / 'finetune' / 'shards300'


def checkpoint_blocks(dit):
    import torch.utils.checkpoint as cp
    for blk in dit.blocks:
        f = blk.forward
        def wrapped(*args, _f=f, **kw):
            return cp.checkpoint(lambda *x: _f(*x, **kw), *args, use_reentrant=False)
        blk.forward = wrapped


class Disc(nn.Module):
    def __init__(self, c=48, w=192):
        super().__init__()
        sn = nn.utils.parametrizations.spectral_norm
        blk = lambda i, o, s: nn.Sequential(sn(nn.Conv2d(i, o, 3, s, 1)), nn.LeakyReLU(0.2))
        self.net = nn.Sequential(blk(c, w, 1), blk(w, w, 2), blk(w, 2 * w, 2), blk(2 * w, 2 * w, 2))
        self.out = sn(nn.Conv2d(2 * w, 1, 1))

    def forward(self, x):                      # [N,48,h,w] -> patch logits mean [N]
        return self.out(self.net(x)).mean((1, 2, 3))


def make_rollout(model, chunk=4, topk=6, traj_scale=4.0, steps=4):
    """A minimal object exposing Memorizon's Self-Forcing rollout methods for a single generator."""
    sf = types.SimpleNamespace()
    sf.config = types.SimpleNamespace(chunk=chunk, topk_per_chunk=topk, traj_scale=traj_scale, generator_timesteps=steps)
    sf.generator = model
    sf.gen_sched = FlowMatchScheduler(shift=model.config.scheduler_shift, sigma_min=0.0, extra_one_step=True)
    sf.gen_sched.set_timesteps(steps, training=True)
    for name in ('_flow', '_gen_chunk', '_self_history', '_rollout', '_generator_x0'):
        setattr(sf, name, types.MethodType(getattr(SelfForcingModel, name), sf))
    return sf


def rel_targets(c2w, prev_idx, q_idx, units_to_gauge):
    """Commanded relative poses prev -> q (sample cameras, metres/4) in the anchor-depth gauge."""
    T = torch.linalg.inv(c2w[prev_idx]) @ c2w[q_idx]
    return T[:, :3, :3], T[:, :3, 3] * units_to_gauge


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True); ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--lr', type=float, default=1e-5); ap.add_argument('--adv', type=float, default=0.05); ap.add_argument('--d_lr', type=float, default=1e-4)
    ap.add_argument('--n_query_chunks', type=int, default=3); ap.add_argument('--save_every', type=int, default=250)
    ap.add_argument('--head', default=str(HERE / 'head864' / 'pose_head864.pt')); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max_steps_debug', type=int, default=0); a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(a) | dict(out=str(a.out)), open(a.out / 'args.json', 'w'), indent=1)
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    model = MemorizonModel.from_pretrained('/home/hj/Alican/baselines/models/memorizon_local').to('cuda', torch.bfloat16)
    params = trainable_parameters(model); assert params and all('prope_attn' in n for n, _ in params)
    master = [p.detach().float().clone().requires_grad_(True) for _, p in params]
    opt = torch.optim.AdamW(master, lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0)
    checkpoint_blocks(model.model); model.train()
    sf = make_rollout(model)
    head = Head().cuda(); head.load_state_dict(torch.load(a.head, map_location='cuda')); head.eval().requires_grad_(False)
    disc = Disc().cuda(); d_opt = torch.optim.AdamW(disc.parameters(), lr=a.d_lr, betas=(0.0, 0.99))
    nq = a.n_query_chunks; C = 4
    ds = LongSpanDataset(str(SHARDS), chunk=C, n_query_chunks=nq, m_min=nq - 1, m_max=9, topk_per_chunk=6, traj_scale=4.0, dtype=torch.bfloat16)
    man = json.load(open('/home/hj/Alican/analysis/plan_base_restart_20261008/camera_quality_20261015/manifest.json'))
    ds.episode_ids = [e for e in ds.episode_ids if e not in man['shards']]          # camera-quality manifest (window level)
    metric = {}
    print(f'[refl] {len(ds)} episodes, trainable {sum(p.numel() for _, p in params) / 1e6:.1f}M, nq={nq}, adv={a.adv}', flush=True)
    log = open(a.out / 'train.log', 'a'); t0 = time.time(); step = 0; skips = 0
    while step < a.steps:
        i = random.randrange(len(ds)); ep = ds.episode_ids[i]
        try:
            s = ds.getitem(i)
        except Exception as e:
            print('skip', ep, repr(e)[:120], flush=True); skips += 1
            if skips > 50:
                raise RuntimeError('more than 50 consecutive skipped samples') from e
            continue
        skips = 0
        if ep not in metric:
            metric[ep] = float(np.load(SHARDS / 'aux' / f'{ep}.npz')['metric_anchor_depth_m'])
        u2g = 4.0 / metric[ep]; Kx = s['intrinsics']; kn = torch.tensor([[float(Kx[0]) / 864, float(Kx[1]) / 480]], device='cuda')                                                     # metres/4 -> anchor-depth gauge
        lat = s['latents'][None].to('cuda', torch.bfloat16); c2w = s['c2w'][None].to('cuda', torch.float32)
        intr = s['intrinsics'][None].to('cuda', torch.float32); text = s['text_embedding'][None].to('cuda', torch.bfloat16)
        n = int(s['n_cond']); n_bank = int(s['n_bank']); nch = (lat.shape[2] - n) // C
        with torch.no_grad():
            for (_, p), m in zip(params, master):
                p.copy_(m.to(p.dtype))
        exit_steps = [3] * nch                     # ReFL on the real 4-step output only (head is validated on final outputs, not intermediate x0)
        with torch.autocast('cuda', torch.bfloat16):                               # as Memorizon training (HF bf16 autocast)
            x0_roll, steps_in = sf._rollout(lat, c2w, intr, text, n, n_bank, exit_steps)
        # ---- discriminator step (detached fakes)
        real = lat[0, :, n:].permute(1, 0, 2, 3).float(); fake = x0_roll[0].permute(1, 0, 2, 3).float()
        if a.adv > 0:
            with torch.autocast('cuda', torch.bfloat16):
                d_loss = F.relu(1 - disc(real)).mean() + F.relu(1 + disc(fake)).mean()
            d_opt.zero_grad(); d_loss.backward(); d_opt.step()
        # ---- generator step: exit steps recomputed with gradient
        with torch.autocast('cuda', torch.bfloat16):
            x0 = sf._generator_x0(steps_in, text, intr).float()                        # [1,48,4*nch,h,w]
        cams = c2w[0]; prev_lat, prev_idx, q_idx = [], [], []
        for j in range(nch):
            pl = lat[0, :, n - 1] if j == 0 else x0_roll[0, :, C * j - 1]          # what the chunk was generated after
            for k in range(C):
                prev_lat.append(pl); prev_idx.append(n - 1 if j == 0 else n + C * j - 1); q_idx.append(n + C * j + k)
        Rc, tc = rel_targets(cams, torch.tensor(prev_idx, device='cuda'), torch.tensor(q_idx, device='cuda'), u2g)
        with torch.autocast('cuda', torch.bfloat16):
            NQ = len(prev_lat)
            Rp, tp = head(lat[0, :, 0].float()[None].expand(NQ, -1, -1, -1), torch.stack(prev_lat).float(), x0[0].permute(1, 0, 2, 3), kn.expand(NQ, -1))
        l_t = (tp.float() - tc).norm(dim=-1).mean(); l_r = geodesic(Rp.float(), Rc).mean(); loss = l_t + l_r
        g_adv = torch.zeros((), device='cuda')
        if a.adv > 0 and step >= 50:
            with torch.autocast('cuda', torch.bfloat16):
                g_adv = -disc(x0[0].permute(1, 0, 2, 3)).float().mean()
            loss = loss + a.adv * g_adv
        loss.backward(); disc.zero_grad(set_to_none=True)
        with torch.no_grad():
            for (_, p), m in zip(params, master):
                m.grad = None if p.grad is None else p.grad.float(); p.grad = None
        gn = torch.nn.utils.clip_grad_norm_(master, 1.0); opt.step(); opt.zero_grad(set_to_none=True)
        step += 1
        with torch.no_grad():
            last = torch.arange(C - 1, len(q_idx), C)                              # chunk-end latents
            ratio = (tp.float()[last].norm(dim=-1) / tc[last].norm(dim=-1).clamp_min(1e-3)).median()
        rec = dict(step=step, ep=ep[:8], loss=round(float(loss), 4), cam_t=round(float(l_t), 4), cam_r_deg=round(float(l_r) * 57.3, 3),
                   ratio_head=round(float(ratio), 3), adv_g=round(float(g_adv), 3), d_loss=round(float(d_loss), 3) if a.adv > 0 else None,
                   grad=round(float(gn), 4), exits=exit_steps, T=lat.shape[2], sec=round(time.time() - t0, 1),
                   peak=round(torch.cuda.max_memory_allocated() / 2**30, 1))
        log.write(json.dumps(rec) + '\n'); log.flush()
        if step % 10 == 0 or step <= 3:
            print(json.dumps(rec), flush=True)
        if step % a.save_every == 0 or step == a.steps:
            with torch.no_grad():
                for (_, p), m in zip(params, master):
                    p.copy_(m.to(p.dtype))
            save_trainable(model, a.out / f'trainable_{step:05d}.pt'); torch.save(disc.state_dict(), a.out / 'disc.pt')
        if a.max_steps_debug and step >= a.max_steps_debug:
            break


if __name__ == '__main__':
    main()
