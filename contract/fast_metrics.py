"""Fast decision metrics for the short-arm loop (10 Nov): two scenes (02267acf showroom, 0238de14 playground), first N = 321 frames
(20 calls), CPU only, no ViPE. Per video:
  snap1        first in-call transition / other in-call transitions (DINO change), from call 2 on        [call-start snap; GT ~1.0]
  dino / dino_last   DINO cos to the recorded frame, all frames / last third                              [collapse; higher = better]
  lum_slope    slope per 100 frames of (generated - recorded) luminance in anchor-visible cells; lum_end = mean of the last third
  sharp        Laplacian-variance ratio generated / recorded, last third (median)                         [<< 1 blur collapse, >> 1 grid]
  p48          DINO cos(frame t, t+48), second half                                                       [own-history identity; GT ~.95]
  regen_in / regen_x   change of GENERATED cells (not anchor-visible) over 8 frames, ratio to the recording: pairs inside one call
               (16k+8 -> 16k+16) / pairs across a call boundary (16k+16 -> 16k+24); vis_in / vis_x the same on anchor-visible cells
Selection uses snap1, lum_slope, dino_last, sharp; never LPIPS. Usage: fast_metrics.py [name=run_dir_with_{scene} ...] [--n 321]
Output eval_gm/fast/metrics.json (cached per name / scene / n) + table."""
import json, sys
from pathlib import Path
import numpy as np, torch, cv2
from PIL import Image
from torch.nn import functional as F
HERE = Path(__file__).resolve().parent; OUT = HERE / 'eval_gm/fast/metrics.json'; OUT.parent.mkdir(parents=True, exist_ok=True)
args = sys.argv[1:]; N = int(args[args.index('--n') + 1]) if '--n' in args else 321; extra = [a for a in args if '=' in a]; SCENES = args[args.index('--scenes') + 1].split(',') if '--scenes' in args else None; sys.argv = sys.argv[:1]
import diag_seams_sota as ds
import diag_anchor_identity as ai
SC = SCENES or ['02267acf', '0238de14']; torch.set_num_threads(8)
REF = {'GT': None, 'GM': 'Ours', 'GM-B': 'GM-B', 'Lyra-2': 'Lyra-2', 'Spatia': 'Spatia'}


def frames(src, scene):
    if src == 'GT' or not (Path(src) / 'review_full.mp4').exists():
        for t, f in enumerate(ds.frames_of(src, scene)):
            if t >= N: break
            yield f if f.shape[:2] == (480, 864) else cv2.resize(f, (864, 480), interpolation=cv2.INTER_AREA)
        return
    cap = cv2.VideoCapture(str(Path(src) / 'review_full.mp4'))
    for t in range(N):
        ok, im = cap.read()
        if not ok: raise ValueError(f'{src}: {t} frames < {N}')
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB); yield im if im.shape[:2] == (480, 864) else cv2.resize(im, (864, 480), interpolation=cv2.INTER_AREA)
    cap.release()


@torch.no_grad()
def features(src, scene):
    E, buf, low, sharp, keep = [], [], [], [], {}
    for t, f in enumerate(frames(src, scene)):
        g = f.astype(np.float32).mean(-1); low.append(cv2.resize(g, (54, 30), interpolation=cv2.INTER_AREA)); sharp.append(float(cv2.Laplacian(g, cv2.CV_32F).var()))
        if t % 8 == 0: keep[t] = torch.from_numpy(np.ascontiguousarray(f)).permute(2, 0, 1)[None].float() / 127.5 - 1
        buf.append(ds.pre(Image.fromarray(f)))
        if len(buf) == 32: E.append(F.normalize(ds.dino(torch.stack(buf)), dim=-1)); buf = []
    if buf: E.append(F.normalize(ds.dino(torch.stack(buf)), dim=-1))
    return dict(E=torch.cat(E).numpy(), low=np.stack(low), sharp=np.array(sharp), keep=keep)


def region_change(keep, M, idx, lp):
    """LPIPS(frame t, t+8) on generated (N) and anchor-visible (A) cells, pairs inside a call and across a call boundary."""
    out = {k: [] for k in ('N_in', 'N_x', 'A_in', 'A_x')}
    for t in range(8, N - 8, 8):
        if t + 8 not in keep or t not in idx or t + 8 not in idx: continue
        with torch.no_grad(): d = lp(keep[t], keep[t + 8])[0, 0].numpy()
        A = M[idx[t]] & M[idx[t + 8]]; Nn = ~M[idx[t]] & ~M[idx[t + 8]]; tag = 'in' if t % 16 == 8 else 'x'
        if Nn.mean() > .05: out[f'N_{tag}'].append((t, float(d[Nn].mean())))
        if A.mean() > .05: out[f'A_{tag}'].append((t, float(d[A].mean())))
    return out


def metrics(ft, gt, M, idx, rc, rc_gt):
    E, G = ft['E'], gt['E']; n = len(E); d = 1 - (E[:-1] * E[1:]).sum(-1); third = slice(2 * n // 3, n)
    cos = (E * G[:n]).sum(-1); p48 = (E[:-48] * E[48:]).sum(-1)
    dl = [(t, float((ft['low'][t] - gt['low'][t])[cell].mean())) for t in range(8, n, 8) if t in idx for cell in [F.adaptive_avg_pool2d(torch.from_numpy(M[idx[t]])[None, None].float(), (30, 54))[0, 0].numpy() > .5] if cell.mean() > .2]
    tt = np.array([a for a, _ in dl]); yy = np.array([b for _, b in dl])
    def ratio(key):
        g = dict(rc_gt[key]); v = [b / g[t] for t, b in rc[key] if t in g and g[t] > 0]; return round(float(np.mean(v)), 2) if v else None
    return dict(snap1=ds.snap(d), dino=round(float(cos[1:].mean()), 3), dino_last=round(float(cos[third].mean()), 3), lum_slope=round(float(np.polyfit(tt, yy, 1)[0] * 100), 2), lum_end=round(float(yy[tt >= 2 * n // 3].mean()), 1),
                sharp=round(float(np.median(ft['sharp'][third] / gt['sharp'][third])), 2), p48=round(float(p48[len(p48) // 2:].mean()), 3),
                regen_in=ratio('N_in'), regen_x=ratio('N_x'), vis_in=ratio('A_in'), vis_x=ratio('A_x'))


if __name__ == '__main__':
    import lpips
    lp = lpips.LPIPS(net='alex', spatial=True, verbose=False).eval(); res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for scene in SC:
        M = ai.load_mask(scene); idx = {t: j for j, t in enumerate(ai.FR)}; srcs = ds.sources(scene); todo = {k: srcs[v] for k, v in REF.items() if v and v in srcs}
        for spec in extra:
            name, pth = spec.split('=', 1); pth = pth.format(scene=scene)
            if (Path(pth) / 'review_full.mp4').exists(): todo[name] = pth
        todo = {k: v for k, v in todo.items() if f'{scene}|{k}|{N}' not in res}
        if not todo: continue
        gt = features('GT', scene); rc_gt = region_change(gt['keep'], M, idx, lp)
        for name, src in todo.items():
            ft = features(src, scene); rc = region_change(ft['keep'], M, idx, lp)
            res[f'{scene}|{name}|{N}'] = metrics(ft, gt, M, idx, rc, rc_gt); OUT.write_text(json.dumps(res, indent=1)); print(scene, name, res[f'{scene}|{name}|{N}'], flush=True)
    cols = ['snap1', 'dino_last', 'dino', 'lum_slope', 'lum_end', 'sharp', 'p48', 'regen_in', 'regen_x', 'vis_in', 'vis_x']
    names = list(dict.fromkeys(k.split('|')[1] for k in res if k.endswith(f'|{N}')))
    print(f'\\n{N} frames, ' + ' / '.join(SC)); print('model'.ljust(22) + ''.join(c.rjust(14) for c in cols))
    for nme in names:
        row = [res.get(f'{s}|{nme}|{N}') for s in SC]
        if all(r is None for r in row): continue
        flag = '  << sharp > 1.6: grid suspect' if any(r is not None and r['sharp'] > 1.6 for r in row) else ''      # dev-set grid artefact direction
        print(nme.ljust(22) + ''.join('/'.join('-' if r is None else str(r[c]) for r in row).rjust(14) for c in cols) + flag)
