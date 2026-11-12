"""Call-boundary jumps on the SOTA cohort (10 Nov, user: "sinirlardaki atlamayi da kontrol et sota sahnelerde"). Per video (481 frames,
streamed): per-transition DINO dissimilarity (1 - cos) and pixel MAD (luminance, 1/4 resolution) and signed luminance step.
Ours / GM-B / GM + mask: boundary = transition 16k -> 16k+1 (k >= 1, 29 per video) vs all other transitions -> ratio boundary / within.
Lyra-2 / Spatia (own chunking unknown here): the period P in 8..120 and phase with the largest mean-jump ratio is reported (their own
boundaries if they have visible ones), plus the 16-frame ratio for reference. Ground truth = the same statistics on the recorded frames.
Identity (critique 10 Nov: measure against the model's OWN history, not the recording): persist16 = mean DINO cos(frame t, frame t+16),
persist48 (3 calls), and the same on the ground truth of the scene (ratio gen / GT); late = second half. Embeddings are kept in
eval_gm/diag/seams_sota_emb/<scene>_<name>.npy so further identity metrics need no re-embedding.
Usage: diag_seams_sota.py [extra name=src_dir ...]  (extra: an additional model video for every scene, src_dir with {scene})
Output eval_gm/diag/seams_sota.json."""
import gm_env  # machine paths: gm_env.py at the repository root (scripts/env.sh puts it on PYTHONPATH)
import json, sys, numpy as np, torch, cv2
from pathlib import Path
from PIL import Image
import importlib.util
torch.set_num_threads(4)
import torchvision.transforms as T
V2 = Path(gm_env.FOUNDATION); HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(V2)); import visual_compare480_20261105 as vc
ct = vc.ct if hasattr(vc, 'ct') else None
spec = importlib.util.spec_from_file_location('ct2', V2 / 'comparison_table_20261105.py'); ct = importlib.util.module_from_spec(spec); spec.loader.exec_module(ct)
dino = torch.hub.load((gm_env.HOME + '/.cache/torch/hub/facebookresearch_dinov2_main'), 'dinov2_vits14', source='local', pretrained=False)
dino.load_state_dict(torch.load((gm_env.HOME + '/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth'), map_location='cpu')); dino.eval()
pre = T.Compose([T.Resize((224, 392)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
SC = ['02267acf', '0238de14', '334ff324', '61b27b61', 'a126efe8']; N = 481


def frames_of(src, scene):
    src = Path(src)
    if src == Path('GT'):
        row = vc.GT_ROWS[scene]
        for t in range(N): yield np.asarray(Image.open(Path(row['frames_dir']) / f"frame_{row['original_start'] + t + 1:05d}.jpg").convert('RGB'))
        return
    if (src / 'frames.npz').exists():
        with np.load(src / 'frames.npz', mmap_mode='r') as z:
            for t in range(N): yield np.array(z['frames'][t])
        return
    if (src / f'{scene}_frames').exists():
        for t in range(N): yield np.asarray(Image.open(src / f'{scene}_frames' / f'frame_{t:05d}.png').convert('RGB'))
        return
    cap = cv2.VideoCapture(str(src if src.is_file() else src / 'review_full.mp4'))
    for t in range(N):
        ok, im = cap.read()
        if not ok: raise ValueError(f'{src}: missing frame {t}')
        yield cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    cap.release()


@torch.no_grad()
def series(src, scene):
    E, L, prev = [], [], None; buf = []; mad = []; dlum = []
    for f in frames_of(src, scene):
        if f.shape[:2] != (480, 864): f = cv2.resize(f, (864, 480), interpolation=cv2.INTER_AREA)
        g = cv2.resize(f, (216, 120), interpolation=cv2.INTER_AREA).astype(np.float32).mean(-1)
        if prev is not None: mad.append(float(np.abs(g - prev).mean())); dlum.append(float(g.mean() - prev.mean()))
        prev = g; buf.append(pre(Image.fromarray(f)))
        if len(buf) == 32: E.append(torch.nn.functional.normalize(dino(torch.stack(buf)), dim=-1)); buf = []
    if buf: E.append(torch.nn.functional.normalize(dino(torch.stack(buf)), dim=-1))
    E = torch.cat(E).numpy(); d = 1 - (E[:-1] * E[1:]).sum(-1)
    return np.asarray(d), np.asarray(mad), np.asarray(dlum), E


def ratio16(x):
    i = np.arange(len(x)); b = ((i + 1) % 16 == 0) & (i >= 16)                     # transition i: frame i -> i+1; boundary 16k -> 16k+1, k >= 1
    return float(x[b].mean() / max(x[~b].mean(), 1e-9)), float(x[b].mean()), float(x[~b].mean())


def snap(x):
    """10 Nov: the call seam shows one frame AFTER the boundary (first -> second frame of the call, transition 16k+1 -> 16k+2): its mean / the
    mean of the other in-call transitions."""
    i = np.arange(len(x)); p = (i - 1) % 16 + 1; first = (p == 1) & (i >= 17); inner = (p > 1) & (p < 16) & (i >= 1)
    return round(float(x[first].mean() / max(x[inner].mean(), 1e-9)), 3)


def best_period(x):
    best = (1., 0, 0)
    for P in range(8, 121):
        for ph in range(P):
            i = np.arange(len(x)); b = (i % P) == ph
            if b.sum() < 3: continue
            r = float(x[b].mean() / max(x[~b].mean(), 1e-9))
            if r > best[0]: best = (r, P, ph)
    return dict(ratio=round(best[0], 2), period=best[1], phase=best[2])


def sources(scene):
    out = {'GT': 'GT'}
    for name, model, arm in (('Ours', 'ours', 'baseline'), ('Ours + corr', 'ours', 'correction'), ('Lyra-2', 'lyra', 'baseline'), ('Spatia', 'spatia', 'baseline')):
        _, d = ct.score_file(model, scene)
        if d is not None and arm in d['arms']: out[name] = d['arms'][arm]['source']
    for name, f, arm in (('GM + mask', f'gm_inside_{scene}_baseline.json', 'baseline'), ('GM-B', f'gm_b_{scene}.json', 'baseline'), ('GM-B + corr', f'gm_b_{scene}.json', 'correction')):
        p = HERE / 'sota_eval/scores' / f
        if p.exists():
            d = json.load(open(p))
            if arm in d['arms']: out[name] = d['arms'][arm]['source']
    return out


if __name__ == '__main__':
    OUT = HERE / 'eval_gm/diag/seams_sota.json'; EMB = HERE / 'eval_gm/diag/seams_sota_emb'; res = json.load(open(OUT)) if OUT.exists() else {}
    for key, r in res.items():                                                                    # backfill snap1 (added 10 Nov) from the saved embeddings
        f = EMB / (key.replace('|', '_').replace(' ', '_').replace('+', 'p') + '.npy')
        if f.exists():
            E = np.load(f).astype(np.float32); r['snap1'] = snap(1 - (E[:-1] * E[1:]).sum(-1))
    json.dump(res, open(OUT, 'w'), indent=1)
    for scene in SC:
        src_all = sources(scene)
        for spec in sys.argv[1:]:
            n, pth = spec.split('=', 1); pth = pth.format(scene=scene)
            if Path(pth).exists(): src_all[n] = pth
        for name, src in src_all.items():
            key = f'{scene}|{name}'
            if key in res: continue
            d, mad, dl, E = series(src, scene); EMB.mkdir(exist_ok=True); np.save(EMB / f"{scene}_{name.replace(' ', '_').replace('+', 'p')}.npy", E.astype(np.float16))
            per = lambda k, late=False: float((E[:-k] * E[k:]).sum(-1)[(len(E) - k) // 2 if late else 0:].mean())
            r = dict(dino=ratio16(d), mad=ratio16(mad), lum_step_boundary=round(float(np.abs(dl[[i for i in range(len(dl)) if (i + 1) % 16 == 0 and i >= 16]]).mean()), 3),
                     lum_step_within=round(float(np.abs(dl[[i for i in range(len(dl)) if not ((i + 1) % 16 == 0 and i >= 16)]]).mean()), 3),
                     best_dino=best_period(d), best_mad=best_period(mad), mad_by_boundary=[round(float(mad[i]), 2) for i in range(len(mad)) if (i + 1) % 16 == 0 and i >= 16], snap1=snap(d), snap1_low=None, persist16=round(per(16), 4), persist16_late=round(per(16, True), 4), persist48=round(per(48), 4), persist48_late=round(per(48, True), 4))
            res[key] = r; json.dump(res, open(OUT, 'w'), indent=1)
            print(f"{scene} {name:12s} DINO b/w {r['dino'][0]:.2f} ({r['dino'][1]:.4f}/{r['dino'][2]:.4f})  MAD b/w {r['mad'][0]:.2f} ({r['mad'][1]:.2f}/{r['mad'][2]:.2f})  "
                  f"|dlum| {r['lum_step_boundary']}/{r['lum_step_within']}  persist16 {r['persist16']:.3f}/{r['persist16_late']:.3f} persist48 {r['persist48']:.3f}/{r['persist48_late']:.3f}", flush=True)
