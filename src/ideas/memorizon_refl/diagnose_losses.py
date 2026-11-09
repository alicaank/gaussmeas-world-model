"""Loss-balance and discriminator diagnostics for a refiner checkpoint (venv_memorizon, GPU), before any v5 recipe.

On N fixed samples (seeded LongSpanDataset draws, same warp pairs as training):
  grads  per parameter group (prope_attn camera branch, inject_embedding): ||g_L1||, ||lambda g_adv||, cosine, their norm
         ratio, and the projection decomposition of the summed gradient (shares sum to 1). One global clip coefficient (both
         terms scale together). Pre-optimizer: AdamW rescales per coordinate, so this is not the applied update split.
  disc   discriminator sensitivity: logits on recorded target latents, the same latents 4x4-box-blurred (latent space),
         decoded->re-encoded (VAE round-trip control), decoded->blurred->re-encoded (pixel blur, sigma 1.5 px; compare
         against the round-trip, not against the raw latent), the warp composite, and the refiner output.
         A discriminator that does not separate sharp from blurred recorded latents cannot be a sharpness signal.
         Condition-mismatch test is N/A: the current D is unconditional (noted in the output).
Output: <out>.json
"""
import argparse, json, random, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import torch._functorch.config; torch._functorch.config.donated_buffer = False                # two autograd.grad calls on one graph (compiled backward)
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE))
import train_refiner as TR
from refl_camera import Disc


def flat(grads):
    return torch.cat([g.reshape(-1) for g in grads if g is not None]) if any(g is not None for g in grads) else torch.zeros(1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ckpt', required=True); ap.add_argument('--disc', required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--adv', type=float, default=0.05); ap.add_argument('--l1_pool', type=int, default=1); ap.add_argument('--n', type=int, default=8); ap.add_argument('--seed', type=int, default=123)
    a = ap.parse_args(); torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    pipe = TR.MemorizonPipeline(TR.MemorizonModel.from_pretrained('/home/hj/Alican/baselines/models/memorizon_local')); pipe.vae.clear_cache()
    model = TR.attach_inject(pipe.model); TR.load_trainable(model, a.ckpt); model.train()
    groups = {'prope_attn': [], 'inject': []}
    for n, p in model.named_parameters():
        want = ('prope_attn' in n) or n.endswith('inject_embedding.weight'); p.requires_grad_(want)
        if want:
            groups['inject' if 'inject' in n else 'prope_attn'].append(p)
    TR.checkpoint_blocks(model.model)
    disc = Disc().cuda(); disc.load_state_dict(torch.load(a.disc, map_location='cuda')); disc.eval().requires_grad_(False)
    sched = TR.FlowMatchScheduler(shift=model.config.scheduler_shift, sigma_min=0.0, extra_one_step=True); sched.set_timesteps(4)
    da3 = TR.load_da3('cuda'); pairs = TR.Pairs(da3); C = 4
    ds = TR.LongSpanDataset(str(TR.SHARDS), chunk=C, n_query_chunks=1, m_min=1, m_max=9, topk_per_chunk=6, traj_scale=4.0, dtype=torch.bfloat16)
    man = json.load(open('/home/hj/Alican/analysis/plan_base_restart_20261008/camera_quality_20261015/manifest.json')); ds.episode_ids = [e for e in ds.episode_ids if e not in man['shards']]
    draw = ds._draw_span; last = {}
    def spy(*args, **kw):
        out = draw(*args, **kw); last['target'] = out[3]; return out
    ds._draw_span = spy
    pool = lambda z: F.avg_pool2d(z[0].permute(1, 0, 2, 3), a.l1_pool) if a.l1_pool > 1 else z
    rows, logits = [], dict(real=[], real_latent_blur=[], real_vae_roundtrip=[], real_pixel_blur=[], warp=[], refiner=[])
    k = 0
    while k < a.n:
        i = random.randrange(len(ds)); ep = ds.episode_ids[i]
        try:
            s = ds.getitem(i); q0 = int(last['target'][0]); rho = random.uniform(0.2, 1.0); z_w, cov = pairs(pipe, ep, q0, rho)
        except Exception as e:
            print('skip', ep, repr(e)[:120], flush=True); continue
        lat = s['latents'][None].to('cuda', torch.bfloat16); n = int(s['n_cond']); Fn = lat.shape[2]
        c2w = s['c2w'][None].to('cuda', torch.float32); intr = s['intrinsics'][None].to('cuda', torch.float32)
        text = s['text_embedding'][None].to('cuda', torch.bfloat16); mask = s['attention_mask'].to('cuda'); rope = s['rope_index'].to('cuda')
        cam = TR.camera_transforms(model.model, c2w, intr); inject = torch.zeros_like(lat); inject[:, :, n:] = z_w.to(lat)
        sig = float(sched.sigmas[2]); x = ((1 - sig) * z_w.float() + sig * torch.randn_like(z_w.float())).to(torch.bfloat16)
        for si in range(2, 4):
            ts = torch.zeros((1, Fn), device='cuda'); ts[:, n:] = float(sched.timesteps[si]); final = si == 3
            with torch.set_grad_enabled(final), torch.autocast('cuda', torch.bfloat16):
                flow = model.model(torch.cat([lat[:, :, :n], x], 2), ts, text, rope_index=rope, attention_mask=mask, camera=cam, inject=inject)[:, :, n:]
                x0 = sched.step_diff_noise_level(flow.float(), ts[:, n:], x.float(), to_final=True) if final else None
                if not final:
                    x = sched.step_diff_noise_level(flow, ts[:, n:], x).to(torch.bfloat16)
        tgt = lat[:, :, n:].float(); fake = x0[0].permute(1, 0, 2, 3)
        fake_d = fake.detach().clone()
        l1 = (pool(x0) - pool(tgt)).abs().mean()
        with torch.autocast('cuda', torch.bfloat16):
            g_adv = -disc(fake).float().mean()
        rec = dict(ep=ep[:8], rho=round(rho, 2), cover=round(cov, 3), l1=float(l1), adv=float(g_adv))
        params = groups['prope_attn'] + groups['inject']
        gl1 = torch.autograd.grad(l1, params, retain_graph=True, allow_unused=True); gad = torch.autograd.grad(g_adv, params, allow_unused=True)
        A1, A2 = flat(gl1).float(), flat(gad).float(); clip = min(1.0, 1.0 / max((A1 + a.adv * A2).norm().item(), 1e-9))   # ONE global clip coefficient
        for name, idx in (('prope_attn', slice(0, len(groups['prope_attn']))), ('inject', slice(len(groups['prope_attn']), None)), ('all', slice(None))):
            a1, a2 = flat(gl1[idx]).float(), a.adv * flat(gad[idx]).float(); gt = a1 + a2; n1, n2, nt = a1.norm().item(), a2.norm().item(), gt.norm().item()
            cos = F.cosine_similarity(a1, a2, dim=0).item() if a1.numel() > 1 and n1 > 0 and n2 > 0 else None
            # projection decomposition of the summed gradient: share_l1 + share_adv = 1 (can be negative when the terms oppose)
            rec[name] = dict(norm_l1=n1, norm_adv_weighted=n2, norm_adv_raw=n2 / max(a.adv, 1e-12), cosine=cos, norm_total=nt,
                             norm_ratio_adv_over_l1=n2 / max(n1, 1e-12), proj_share_adv=float((a2 @ gt) / max(nt ** 2, 1e-24)),
                             proj_share_l1=float((a1 @ gt) / max(nt ** 2, 1e-24)), global_clip_coeff=clip,
                             note='pre-optimizer gradients; AdamW rescales per coordinate, so this is not the applied update split')
        # discriminator sensitivity (no grad); DiT + DA3 offloaded so the full-sequence VAE decode/encode fits
        del gl1, gad, x0, flow; model.zero_grad(set_to_none=True); model.model.to('cpu'); da3.to('cpu'); torch.cuda.empty_cache()
        with torch.no_grad(), torch.autocast('cuda', torch.bfloat16):
            real = tgt[0].permute(1, 0, 2, 3)
            blur_lat = F.avg_pool2d(F.pad(real, (2, 1, 2, 1), mode='replicate'), 4, stride=1)
            # pixel blur: decode the 4 target latents (with the recorded prefix latent for the causal VAE), blur, re-encode
            vae = pipe.vae; vae.clear_cache(); seq = lat[0:1, :, n - 1:].float()
            dec = vae.decode((seq * pipe.lat_std + pipe.lat_mean).to(torch.float32)).sample.clamp(-1, 1)           # [1,3,1+16,H,W]
            vae.clear_cache(); fr = dec[0].permute(1, 0, 2, 3)
            kx = torch.exp(-torch.arange(-3, 4, device='cuda', dtype=torch.float32) ** 2 / (2 * 1.5 ** 2)); kx = kx / kx.sum()
            blur = F.conv2d(F.pad(fr, (3, 3, 0, 0), mode='replicate'), kx.view(1, 1, 1, 7).repeat(3, 1, 1, 1), groups=3)
            blur = F.conv2d(F.pad(blur, (0, 0, 3, 3), mode='replicate'), kx.view(1, 1, 7, 1).repeat(3, 1, 1, 1), groups=3)
            vae.clear_cache(); zb = vae.encode(blur.permute(1, 0, 2, 3)[None]).latent_dist.mode().float(); vae.clear_cache()
            zb = ((zb - pipe.lat_mean) / pipe.lat_std)[0, :, 1:].permute(1, 0, 2, 3)
            zr = vae.encode(fr.permute(1, 0, 2, 3)[None]).latent_dist.mode().float(); vae.clear_cache()                 # same path, no blur
            zr = ((zr - pipe.lat_mean) / pipe.lat_std)[0, :, 1:].permute(1, 0, 2, 3)
            for key, z in (('real', real), ('real_latent_blur', blur_lat), ('real_vae_roundtrip', zr), ('real_pixel_blur', zb), ('warp', z_w[0].permute(1, 0, 2, 3).float()), ('refiner', fake_d)):
                logits[key].append(float(disc(z).float().mean()))
        model.model.to('cuda'); da3.to('cuda'); torch.cuda.empty_cache()
        rows.append(rec); k += 1; print(json.dumps({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in rec.items() if kk in ('ep', 'l1', 'adv')} | {'all': {kk: round(v, 4) for kk, v in rec['all'].items() if isinstance(v, float)}}), flush=True)
        model.zero_grad(set_to_none=True); torch.cuda.empty_cache()
    summ = dict(n=len(rows), adv_lambda=a.adv, l1_pool=a.l1_pool, ckpt=a.ckpt,
                grads={g: {k: float(np.mean([r[g][k] for r in rows if isinstance(r[g][k], float)])) for k in rows[0][g] if k != 'note'} for g in ('prope_attn', 'inject', 'all')},
                grads_note=rows[0]['all']['note'],
                disc_logit_mean={k: float(np.mean(v)) for k, v in logits.items()}, disc_logit_std={k: float(np.std(v)) for k, v in logits.items()},
                disc_separates_sharp_from_blur=dict(latent_blur=float(np.mean(np.array(logits['real']) > np.array(logits['real_latent_blur']))),
                                                    pixel_blur_vs_roundtrip=float(np.mean(np.array(logits['real_vae_roundtrip']) > np.array(logits['real_pixel_blur']))),
                                                    roundtrip_vs_real=float(np.mean(np.array(logits['real']) > np.array(logits['real_vae_roundtrip'])))),
                condition_mismatch_test='not applicable: discriminator is unconditional', rows=rows)
    json.dump(summ, open(a.out, 'w'), indent=1); print(json.dumps({k: v for k, v in summ.items() if k != 'rows'}, indent=1))


if __name__ == '__main__':
    main()
