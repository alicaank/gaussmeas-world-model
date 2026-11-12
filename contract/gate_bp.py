"""Gate for B' against old stage B (critique 10 Nov 19:10): if the render geometry error mattered, the difference shows in the first calls.
Per scene and run: LPIPS vs the recording over calls 1-3 (frames 1-48) and call 1 alone, vertical/horizontal scale ratio of the generated
frames (diag_vscale.fit) at frames 16 / 32 / 48 / 96 / 160, plus the fast metrics already computed by queue_fast.gen.
Usage: gate_bp.py name=run_dir_with_{scene} ... -> eval_gm/fast/gate_bp.json"""
import json, sys
from pathlib import Path
import numpy as np, torch, cv2, lpips
specs = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a and not a.startswith('--')); OUTNAME = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else 'gate_bp'; sys.argv = sys.argv[:1]
import diag_seams_sota as ds
import importlib.util
HERE = Path(__file__).parent; torch.set_num_threads(8); net = lpips.LPIPS(net='alex').eval()
sp = importlib.util.spec_from_file_location('vs', HERE / 'diag_vscale.py')
src = (HERE / 'diag_vscale.py').read_text().split("res = json.loads(OUT.read_text())")[0]; vs = dict(__file__=str(HERE / 'diag_vscale.py')); exec(compile(src, 'diag_vscale_head', 'exec'), vs)   # fit() only
T = [16, 32, 48, 96, 128, 160]; out = {}   # 128 added: at 160 the playground often has < 25 SIFT inliers (content diverged)
to = lambda f: torch.from_numpy(cv2.resize(f, (432, 240), interpolation=cv2.INTER_AREA)).permute(2, 0, 1)[None].float() / 127.5 - 1
for scene in ('02267acf', '0238de14'):
    gt = []
    for t, f in enumerate(ds.frames_of('GT', scene)):
        if t > 160: break
        gt.append(f if f.shape[:2] == (480, 864) else cv2.resize(f, (864, 480), interpolation=cv2.INTER_AREA))
    out[scene] = {}
    for name, spec in specs.items():
        cap = cv2.VideoCapture(str(Path(spec.format(scene=scene)) / 'review_full.mp4')); fr = []
        for t in range(161):
            ok, im = cap.read()
            if not ok: break
            fr.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        with torch.no_grad(): lp = [float(net(to(fr[t]), to(gt[t]))) for t in range(1, 49)]
        rng = np.random.default_rng(0); sc = {}
        for t in T:
            r = vs['fit'](gt[t], fr[t], rng) if t < len(fr) else None; sc[str(t)] = None if r is None else round(r[1] / r[0], 4)
        out[scene][name] = dict(lpips_1=round(float(np.mean(lp[:16])), 4), lpips_1_3=round(float(np.mean(lp)), 4), lpips_3=round(float(np.mean(lp[32:])), 4), vscale=sc)
        print(scene, name, out[scene][name], flush=True)
fm = json.loads((HERE / 'eval_gm/fast/metrics.json').read_text()) if (HERE / 'eval_gm/fast/metrics.json').exists() else {}   # fast metrics of the same runs (queue_fast.gen)
for scene in out:
    for name in out[scene]:
        m = fm.get(f'{scene}|{name}|321', {}); m481 = fm.get(f'{scene}|{name}|481', {}); out[scene][name].update({k: m.get(k) for k in ('lum_slope', 'lum_end', 'snap1', 'dino_last', 'sharp', 'regen_x')}, dino_last_481=m481.get('dino_last'), sharp_481=m481.get('sharp'))   # critique: brightness slope in the gate
        print(scene, name, {k: out[scene][name][k] for k in ('lum_slope', 'lum_end', 'snap1', 'dino_last')}, flush=True)
(HERE / f"eval_gm/fast/{OUTNAME}.json").write_text(json.dumps(out, indent=1))
