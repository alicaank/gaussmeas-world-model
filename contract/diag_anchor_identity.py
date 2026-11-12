"""Recording-anchored identity on the SOTA cohort (10 Nov critique: GM-B re-synthesises the observed ceiling of the conference hall while staying
self-consistent; an own-history gate cannot see it). Masks come from the RECORDING only (identical for every model):
  --masks (GPU, DA3): every 8th recorded frame t (8..480) with the anchor, DA3 metric depth / mpu_da3 aligned to the recorded cameras
  (inputs/<scene>.npz c2w, groups of the anchor + 6 frames), mask_t = recorded pixel of frame t back-projected with its depth lands inside the
  anchor and agrees with the anchor depth within 15% (frontier.observed_from_reprojection) -> eval_gm/diag/anchor_masks/<scene>.npz.
  metrics (CPU): per sampled frame, inside mask_t: LPIPS (alex, spatial map averaged over the mask) of generated vs recorded frame t, and the
  low-frequency colour difference (both frames area-pooled to 30x54, mean |RGB diff| and signed luminance gen - rec over masked cells).
  Frames whose mask covers < 20% are not scored. Reports curves + first/second half means + second-half slope per 100 frames.
Caveat: a camera offset counts as identity loss here (read together with the position error).
Usage: diag_anchor_identity.py --masks | diag_anchor_identity.py [extra name=src_dir_with_{scene} ...]  -> eval_gm/diag/anchor_identity.json"""
import gm_env  # machine paths: gm_env.py at the repository root (scripts/env.sh puts it on PYTHONPATH)
import json, sys
from pathlib import Path
import numpy as np, torch
from torch.nn import functional as F
HERE = Path(__file__).resolve().parent; FT = gm_env.CORE
INP = Path((gm_env.EVAL_INPUTS + '/inputs')); MASKS = HERE / 'eval_gm/diag/anchor_masks'; OUT = HERE / 'eval_gm/diag/anchor_identity.json'
SC = ['02267acf', '0238de14', '334ff324', '61b27b61', 'a126efe8']; FR = list(range(8, 481, 8)); MIN_COV = .2
for p in (str(HERE), FT):
    if p not in sys.path: sys.path.insert(0, p)


def build_masks():
    import frontier
    from renderer864 import da3_gauge_depths
    from realized_pose import load_da3
    sys.argv = sys.argv[:1]; import diag_seams_sota as ds                                            # frames_of (recording reader)
    MASKS.mkdir(parents=True, exist_ok=True); da3 = load_da3('cuda')
    for scene in SC:
        if (MASKS / f'{scene}.npz').exists(): continue
        with np.load(INP / f'{scene}.npz') as z: c2w = z['c2w'].astype(np.float64); K = z['K'].astype(np.float64); mpu = float(z['mpu_da3'])
        rec = {t: f for t, f in enumerate(ds.frames_of('GT', scene)) if t == 0 or t in FR}
        masks, depth_ok = [], []
        for i in range(0, len(FR), 6):
            ids = [0] + FR[i:i + 6]; frames = np.stack([rec[t] for t in ids])
            depths, _ = da3_gauge_depths(da3, frames, K, c2w[ids], mpu)
            obs = frontier.observed_from_reprojection(torch.stack(depths[1:]), K, c2w[ids[1:]], [depths[0]], c2w[[0]], tol=.15, margin=2)
            masks.append(obs.numpy() > .5)
        m = np.concatenate(masks); np.savez_compressed(MASKS / f'{scene}.npz', frames=np.array(FR), mask_bits=np.packbits(m, axis=-1), shape=np.array(m.shape))
        print(scene, 'coverage', [round(float(x), 2) for x in m.reshape(len(FR), -1).mean(1)[::6]], flush=True)


def load_mask(scene):
    with np.load(MASKS / f'{scene}.npz') as z: return np.unpackbits(z['mask_bits'], axis=-1)[..., :int(z['shape'][2])].astype(bool)


def metrics(extra):
    import lpips
    sys.argv = sys.argv[:1]; import diag_seams_sota as ds
    lp = lpips.LPIPS(net='alex', spatial=True, verbose=False).eval(); torch.set_num_threads(6)
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for scene in SC:
        M = load_mask(scene); cov = M.reshape(len(FR), -1).mean(1); rec = {t: f for t, f in enumerate(ds.frames_of('GT', scene)) if t in FR}
        srcs = ds.sources(scene); srcs.pop('GT')
        for spec in extra:
            n, pth = spec.split('=', 1); pth = pth.format(scene=scene)
            if Path(pth).exists(): srcs[n] = pth
        for name, src in srcs.items():
            key = f'{scene}|{name}'
            if key in res: continue
            gen = {t: f for t, f in enumerate(ds.frames_of(src, scene)) if t in FR}; L, C, Y = [], [], []
            for j, t in enumerate(FR):
                if cov[j] < MIN_COV: L.append(None); C.append(None); Y.append(None); continue
                a = torch.from_numpy(np.ascontiguousarray(gen[t])).permute(2, 0, 1)[None].float(); b = torch.from_numpy(np.ascontiguousarray(rec[t])).permute(2, 0, 1)[None].float()
                if a.shape[-2:] != b.shape[-2:]: a = F.interpolate(a, b.shape[-2:], mode='area')
                m = torch.from_numpy(M[j])[None, None].float()
                with torch.no_grad(): d = lp(a / 127.5 - 1, b / 127.5 - 1)
                L.append(float((d * m).sum() / m.sum()))
                pa, pb, pm = F.adaptive_avg_pool2d(a, (30, 54)), F.adaptive_avg_pool2d(b, (30, 54)), F.adaptive_avg_pool2d(m, (30, 54)) > .5
                if pm.sum() == 0: C.append(None); Y.append(None); continue
                C.append(float((pa - pb).abs().mean(1, keepdim=True)[pm].mean())); Y.append(float((pa - pb).mean(1, keepdim=True)[pm].mean()))
            res[key] = summarize(L, C, Y, cov); OUT.write_text(json.dumps(res, indent=1))
            s = res[key]; print(f"{scene} {name:16s} LPIPSm {s['lpips_h1']}/{s['lpips_h2']} slope {s['lpips_slope']}  colour {s['col_h1']}/{s['col_h2']}  dlum {s['dlum_h1']}/{s['dlum_h2']}  scored {s['scored']}", flush=True)


def summarize(L, C, Y, cov):
    fr = np.array(FR); h = fr > 240
    def half(v, sel):
        x = [a for a, k in zip(v, sel) if k and a is not None]; return round(float(np.mean(x)), 4) if x else None
    ok = [i for i, v in enumerate(L) if v is not None and h[i]]
    slope = round(float(np.polyfit(fr[ok], np.array(L)[ok], 1)[0] * 100), 4) if len(ok) >= 5 else None
    return dict(frames=FR, coverage=[round(float(c), 3) for c in cov], lpips=L, colour=C, dlum=Y, scored=sum(v is not None for v in L),
                lpips_h1=half(L, ~h), lpips_h2=half(L, h), lpips_slope=slope, col_h1=half(C, ~h), col_h2=half(C, h), dlum_h1=half(Y, ~h), dlum_h2=half(Y, h))


if __name__ == '__main__':
    if '--masks' in sys.argv: build_masks()
    else: metrics([a for a in sys.argv[1:] if '=' in a])
