"""Learned two-view refiner, pilot 1 (status page / user 10-26: "gercek videolara kontrollu hata ekleyip geri duzeltebilme").

Question: given the previous RELIABLE frame and a new frame with a geometric error, can a small network undo the error?
Data: recorded 864x480 train frames at half resolution (240x432); pair = (frame t-4 clean, frame t clean); the input frame t
is corrupted by a controlled GEOMETRIC error of one of three bands (camera-like homography K R K^-1 with yaw/pitch/roll ~
N(0, sigma_r), scale 1+N(0, sigma_s), shift N(0, sigma_px), plus a smooth elastic field of amplitude sigma_e px on a 6x10
grid; reflection padding, no appearance change):
  band  sigma_r(deg)  sigma_s  shift(px)  elastic(px)
  small   0.5          .01       3          1.5
  medium  1.5          .03       9          4.5
  large   3.0          .06      18          9
Model (~2M params): UNet on [prev, corrupted] (6 ch) -> backward flow (2 ch, zero-init -> identity at start) + residual
(3 ch); out = warp(corrupted, flow) + residual. Loss L1 + 0.5 L1 on image gradients. AdamW 2e-4, batch 4, --minutes wall.
Eval (dev streams, held out): per band PSNR / LPIPS of corrupted, refined and copy-previous vs the clean frame; clean
pass-through PSNR (band none); sharpness ratio refined / clean. Real test: our c10 rollouts' generated frames (frames_*.npz,
quarter rate) refined with the previous generated frame as 'reliable' -> LPIPS vs the recording before / after.
Training draws band 'none' (uncorrupted input, identity target) with p .2 next to the three bands (review 10-26, C1).
Pre-registered go (all): (1) medium band: refined LPIPS <= 0.6 x corrupted LPIPS AND < the copy-previous LPIPS; (2) clean
pass-through PSNR >= 35 dB; (3) generated frames: mean LPIPS vs recording after <= before + 0.005 and sharpness ratio >= 0.9.
Go -> pilot 2 (the refiner inside the loop, with the target pose / DA3 geometry as conditioning, camera progress measured);
no-go closes THIS architecture on THIS synthetic-error distribution, not the learned-refiner idea. The pilot has no pose
input, so it cannot correct camera progress; it tests appearance-level geometric repair only.
Usage: refiner_pilot.py --out DIR [--minutes 45 --steps 6000 --batch 4 --gen_runs eval864/c10_vmem_open]"""
import argparse, json, math, os, random, time
from pathlib import Path
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from PIL import Image
from data import corpus864
from render864_rows import rec_frames

H, W = 240, 432
BANDS = dict(small=(0.5, .01, 3., 1.5), medium=(1.5, .03, 9., 4.5), large=(3.0, .06, 18., 9.))


def to_t(frames):                                                                      # uint8 (n,480,864,3) -> (n,3,H,W) in [-1,1]
    x = torch.from_numpy(np.asarray(frames)).permute(0, 3, 1, 2).float() / 127.5 - 1
    return F.interpolate(x, (H, W), mode='area')


def to_u8(x):
    return ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()


def corrupt_grid(band, K, rng, device):
    """Backward-sampling grid (1,H,W,2) in normalised coords for one corruption draw; None = identity."""
    sr, ss, spx, se = BANDS[band]
    ang = np.radians(rng.normal(0, sr, 3)); cx, cy, cz = np.cos(ang); sx_, sy_, sz = np.sin(ang)
    Rx = np.array([[1, 0, 0], [0, cx, -sx_], [0, sx_, cx]]); Ry = np.array([[cy, 0, sy_], [0, 1, 0], [-sy_, 0, cy]]); Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Hm = K @ (Rz @ Ry @ Rx) @ np.linalg.inv(K); s = 1 + rng.normal(0, ss); t = rng.normal(0, spx, 2)
    S = np.array([[s, 0, (1 - s) * W / 2 + t[0]], [0, s, (1 - s) * H / 2 + t[1]], [0, 0, 1]]); Hm = S @ Hm
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float64), torch.arange(W, dtype=torch.float64), indexing='ij')
    pix = torch.stack([xs, ys, torch.ones_like(xs)]).reshape(3, -1); src = torch.from_numpy(np.linalg.inv(Hm)) @ pix; src = src[:2] / src[2:3]
    el = torch.from_numpy(rng.normal(0, se, (1, 2, 6, 10))).float(); el = F.interpolate(el, (H, W), mode='bicubic', align_corners=True)[0]
    src = src.float().reshape(2, H, W) + el
    grid = torch.stack([src[0] / (W - 1) * 2 - 1, src[1] / (H - 1) * 2 - 1], -1)[None]
    return grid.to(device)


def warp(x, grid):
    return F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection', align_corners=True)


class Block(nn.Module):
    def __init__(self, i, o):
        super().__init__(); self.c = nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GELU(), nn.Conv2d(o, o, 3, padding=1), nn.GELU())

    def forward(self, x):
        return self.c(x)


class Refiner(nn.Module):
    def __init__(self, ch=(32, 64, 128, 192)):
        super().__init__(); self.enc = nn.ModuleList(); self.dec = nn.ModuleList(); c_in = 6
        for c in ch:
            self.enc.append(Block(c_in, c)); c_in = c
        for c_skip, c in zip(ch[-2::-1], ch[-1:0:-1]):
            self.dec.append(Block(c + c_skip, c_skip))
        self.head = nn.Conv2d(ch[0], 5, 3, padding=1); nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)   # identity at init
        self.register_buffer('base', torch.stack(torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')[::-1], -1)[None])

    def forward(self, prev, cur):
        x = torch.cat([prev, cur], 1); skips = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else F.avg_pool2d(x, 2)); skips.append(x)
        for d, s in zip(self.dec, skips[-2::-1]):
            x = d(torch.cat([F.interpolate(x, size=s.shape[-2:], mode='bilinear', align_corners=False), s], 1))
        o = self.head(x); flow = 0.1 * torch.tanh(o[:, :2]).permute(0, 2, 3, 1); res = o[:, 2:]
        return warp(cur, self.base + flow) + res, flow


def grad_l1(a, b):
    return (a[..., 1:, :] - a[..., :-1, :] - b[..., 1:, :] + b[..., :-1, :]).abs().mean() + (a[..., :, 1:] - a[..., :, :-1] - b[..., :, 1:] + b[..., :, :-1]).abs().mean()


def psnr(a, b):
    return float(10 * torch.log10(4. / ((a - b) ** 2).mean()))


def sharp(x):
    from scipy.ndimage import laplace
    return float(np.mean([laplace(f.astype(np.float32).mean(-1)).var() for f in to_u8(x)]))


class Pairs:
    def __init__(self, rows, rng):
        self.rows = rows; self.rng = rng; self.K = {}

    def draw(self):
        r = self.rng.choice(self.rows); n = 1 + 4 * (r['latent_count'] - 1); t = self.rng.randrange(4, n)
        fr = rec_frames(r, [t - 4, t]); x = to_t(fr)
        with np.load(Path('data864/components') / r['stream_id'] / 'vae.npz') as z:
            K = z['K'].astype(np.float64).copy()
        K[0] *= W / 864; K[1] *= H / 480
        return x[:1], x[1:], K


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--minutes', type=float, default=45); ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--batch', type=int, default=4); ap.add_argument('--seed', type=int, default=20261026); ap.add_argument('--gen_runs', default='eval864/c10_vmem_open')
    ap.add_argument('--eval_pairs', type=int, default=8); ap.add_argument('--eval_only', type=int, default=0); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); torch.manual_seed(a.seed); rng = random.Random(a.seed); nrng = np.random.default_rng(a.seed); dev = 'cuda'
    train_rows, _ = corpus864('train'); dev_rows, _ = corpus864('development'); dev_rows = dev_rows[:10]
    net = Refiner().to(dev); print('params', sum(p.numel() for p in net.parameters()), flush=True)
    if a.eval_only and (out / 'refiner.pt').exists():
        net.load_state_dict(torch.load(out / 'refiner.pt', map_location=dev))
    else:
        opt = torch.optim.AdamW(net.parameters(), 2e-4, weight_decay=1e-4); pairs = Pairs(train_rows, rng); t0 = time.time(); log = open(out / 'train.log', 'a'); step = 0
        while step < a.steps and (time.time() - t0) / 60 < a.minutes:
            P, C, X = [], [], []
            for _ in range(a.batch):
                prev, cur, K = pairs.draw(); band = 'none' if rng.random() < .2 else rng.choice(list(BANDS))
                P.append(prev); C.append(cur); X.append(cur if band == 'none' else warp(cur, corrupt_grid(band, K, nrng, 'cpu')))
            prev, cur, cor = (torch.cat(v).to(dev) for v in (P, C, X))
            lr = 2e-4 * 0.5 * (1 + math.cos(math.pi * min(1., step / a.steps)))
            for gph in opt.param_groups:
                gph['lr'] = lr
            with torch.autocast('cuda', dtype=torch.bfloat16):
                ref, flow = net(prev, cor)
            ref = ref.float(); loss = (ref - cur).abs().mean() + 0.5 * grad_l1(ref, cur)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.); opt.step(); step += 1
            if step % 50 == 0:
                rep = dict(step=step, loss=round(float(loss), 4), l1_in=round(float((cor - cur).abs().mean()), 4), lr=round(lr, 6), min=round((time.time() - t0) / 60, 1)); log.write(json.dumps(rep) + '\n'); log.flush()
                if step % 500 == 0:
                    print(json.dumps(rep), flush=True)
        torch.save(net.state_dict(), out / 'refiner.pt'); print('trained', step, 'steps', round((time.time() - t0) / 60, 1), 'min', flush=True)
    # ---- evaluation (dev streams, held out) ------------------------------------------------------------------------------
    net.eval(); from evaluate_recorded import metrics_models; _, lp = metrics_models(); erng = np.random.default_rng(a.seed + 1); res = {b: [] for b in list(BANDS) + ['none']}; board = []
    with torch.no_grad():
        for r in dev_rows:
            n = 1 + 4 * (r['latent_count'] - 1); ts = np.linspace(8, n - 1, a.eval_pairs).round().astype(int)
            with np.load(Path('data864/components') / r['stream_id'] / 'vae.npz') as z:
                K = z['K'].astype(np.float64).copy()
            K[0] *= W / 864; K[1] *= H / 480
            for t in ts:
                x = to_t(rec_frames(r, [t - 4, t])); prev, cur = x[:1].to(dev), x[1:].to(dev)
                for b in res:
                    cor = cur if b == 'none' else warp(cur, corrupt_grid(b, K, erng, dev))
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        ref, _ = net(prev, cor)
                    ref = ref.float(); u = dict(cur=to_u8(cur), cor=to_u8(cor), ref=to_u8(ref), prev=to_u8(prev))
                    res[b].append(dict(psnr_cor=psnr(cor, cur), psnr_ref=psnr(ref, cur), psnr_prev=psnr(prev, cur), lp_cor=float(lp(u['cor'], u['cur'])[0]), lp_ref=float(lp(u['ref'], u['cur'])[0]),
                                       lp_prev=float(lp(u['prev'], u['cur'])[0]), sharp_ref=sharp(ref), sharp_cur=sharp(cur)))
                    if b == 'medium' and len(board) < 4 and t == ts[len(ts) // 2]:
                        board.append(np.concatenate([u['prev'][0], u['cor'][0], u['ref'][0], u['cur'][0]], 1))
        gen = {}
        for run in [Path(p) for p in a.gen_runs.split(',') if p]:
            rows_by = {r['stream_id'][:8]: r for r in dev_rows}; before, after, sh = [], [], []
            for f in sorted(run.glob('frames_*.npz')):
                sid = f.stem.replace('frames_', '')
                if sid not in rows_by:
                    continue
                g = np.load(f)['gen']; idx = [1 + 4 * i for i in range(len(g))]; rec = to_t(rec_frames(rows_by[sid], idx)); gt = to_t(g)
                for i in range(1, len(g)):
                    prev, cur, tgt = gt[i - 1:i].to(dev), gt[i:i + 1].to(dev), rec[i:i + 1].to(dev)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        ref, _ = net(prev, cur)
                    ref = ref.float(); before.append(float(lp(to_u8(cur), to_u8(tgt))[0])); after.append(float(lp(to_u8(ref), to_u8(tgt))[0])); sh.append(sharp(ref) / max(sharp(cur), 1e-6))
            gen[run.name] = dict(n=len(before), lpips_before=round(float(np.mean(before)), 4), lpips_after=round(float(np.mean(after)), 4), sharp_ratio=round(float(np.mean(sh)), 3))
    S = {b: {k: round(float(np.mean([x[k] for x in v])), 4) for k in v[0]} for b, v in res.items() if v}
    for b in S:
        S[b]['sharp_ratio'] = round(S[b]['sharp_ref'] / max(S[b]['sharp_cur'], 1e-6), 3)
    g0 = next(iter(gen.values())) if gen else None
    go = dict(c1_medium_lpips_ratio=round(S['medium']['lp_ref'] / max(S['medium']['lp_cor'], 1e-6), 3), c2_clean_passthrough_psnr=S['none']['psnr_ref'],
              c3_gen_lpips_delta=None if g0 is None else round(g0['lpips_after'] - g0['lpips_before'], 4), c3_gen_sharp_ratio=None if g0 is None else g0['sharp_ratio'])
    go['c1_medium_lpips_ref_vs_prev'] = (S['medium']['lp_ref'], S['medium']['lp_prev'])
    checks = dict(c1=go['c1_medium_lpips_ratio'] <= 0.6 and S['medium']['lp_ref'] < S['medium']['lp_prev'], c2=go['c2_clean_passthrough_psnr'] >= 35, c3=g0 is not None and go['c3_gen_lpips_delta'] <= 0.005 and go['c3_gen_sharp_ratio'] >= 0.9)
    verdict = dict(go=all(checks.values()), checks=checks, values=go)
    json.dump(dict(args=vars(a), bands=S, generated=gen, verdict=verdict, per_pair={b: v for b, v in res.items()}), open(out / 'refiner_pilot.json', 'w'), indent=1)
    if board:
        Image.fromarray(np.concatenate(board, 0)).save(out / 'board_medium.jpg', quality=85)
    print('SUMMARY', json.dumps(dict(bands=S, generated=gen, verdict=verdict)), flush=True)


if __name__ == '__main__':
    main(); os._exit(0)
