"""Self-consistency on an out-and-back plan (run_arm_ckpt --return_traj 1): frame t (way out) and frame 2 n_out - t (way back) share a pose.
Spatial LPIPS between the two, every 8th frame, split by A = cells the anchor sees at that pose (given world) and N = cells it does not (content the
model generated; "once generated it must stay"), and by revisit gap (frames between the two visits). No recording needed (masks use the recorded
geometry of the way-out poses only). Usage: diag_return.py name=dir_with_{scene} ... -> eval_gm/diag/return.json (merged)"""
import json, sys
from pathlib import Path
import numpy as np, torch, cv2
HERE = Path(__file__).resolve().parent; OUT = HERE / 'eval_gm/diag/return.json'
specs = [a for a in sys.argv[1:] if '=' in a]; sys.argv = sys.argv[:1]
import diag_anchor_identity as ai
import lpips
torch.set_num_threads(8); lp = lpips.LPIPS(net='alex', spatial=True, verbose=False).eval(); res = json.loads(OUT.read_text()) if OUT.exists() else {}
to = lambda f: torch.from_numpy(np.ascontiguousarray(f)).permute(2, 0, 1)[None].float() / 127.5 - 1
for scene in ('02267acf', '0238de14'):
    M = ai.load_mask(scene); idx = {t: j for j, t in enumerate(ai.FR)}
    for spec in specs:
        name, pth = spec.split('=', 1); st = int(pth.rsplit(':', 1)[1]) if ':' in pth else 1; pth = Path(pth.rsplit(':', 1)[0].format(scene=scene))   # name=dir[:stride]
        if not (pth / 'review_full.mp4').exists(): continue
        cap = cv2.VideoCapture(str(pth / 'review_full.mp4')); fr = []
        while True:
            ok, im = cap.read()
            if not ok: break
            fr.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        n_out = (len(fr) - 1) // 2; rows = []
        for t in range(8, n_out, 8):
            if st * t not in idx: continue
            with torch.no_grad(): d = lp(to(fr[t]), to(fr[2 * n_out - t]))[0, 0].numpy()
            A = M[idx[st * t]]; rows.append((2 * (n_out - t), float(d[A].mean()) if A.mean() > .05 else None, float(d[~A].mean()) if (~A).mean() > .05 else None, float(d.mean()), float((~A).mean())))
        floor = {}                                                                                  # same pose, 2 / 8 frames apart at the turning point: what "perfect" scores on this measure
        for k in (1, 4):
            with torch.no_grad(): d = lp(to(fr[n_out - k]), to(fr[n_out + k]))[0, 0].numpy()
            A = M[idx[st * n_out]] if st * n_out in idx else np.ones_like(d, bool)
            floor[f'gap{2 * k}'] = dict(all=round(float(d.mean()), 4), anchor_seen=round(float(d[A].mean()), 4) if A.mean() > .05 else None, generated=round(float(d[~A].mean()), 4) if (~A).mean() > .05 else None)
        mean = lambda xs: round(float(np.mean([x for x in xs if x is not None])), 4) if any(x is not None for x in xs) else None
        far = [r for r in rows if r[0] >= n_out]; near = [r for r in rows if r[0] < n_out]
        res[f'{scene}|{name}'] = dict(n_out=n_out, all=mean([r[3] for r in rows]), anchor_seen=mean([r[1] for r in rows]), generated=mean([r[2] for r in rows]),
                                      near_gap=dict(all=mean([r[3] for r in near]), anchor_seen=mean([r[1] for r in near]), generated=mean([r[2] for r in near])),
                                      far_gap=dict(all=mean([r[3] for r in far]), anchor_seen=mean([r[1] for r in far]), generated=mean([r[2] for r in far])),
                                      generated_share=mean([r[4] for r in rows]), floor=floor, rows=[[r[0]] + [None if x is None else round(x, 4) for x in r[1:4]] for r in rows])
        print(scene, name, {k: v for k, v in res[f'{scene}|{name}'].items() if k != 'rows'}, flush=True)
OUT.write_text(json.dumps(res, indent=1))
