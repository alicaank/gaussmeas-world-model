"""Command execution from ViPE trajectories (queue_between3.py / the stage-B gate): generated frame i <-> plan pose stride * i. exec = projection of the measured
displacement on the planned one / |planned|^2 (1 = executed; << 1 = standstill), lat = perpendicular part / |planned|, yaw ratio = measured / planned yaw.
Same definitions as gate_a4_camera.py. Usage: gate_b4_camera.py name:stride ...   (trajectories eval_gm/fast/camera_b4/<name>_<scene>_trajectories/baseline.npz;
names not found there are looked up in camera_a4) -> eval_gm/fast/camera_b4/exec.json"""
import gm_env  # machine paths: gm_env.py at the repository root (scripts/env.sh puts it on PYTHONPATH)
import json, sys
from pathlib import Path
import numpy as np
B = Path((gm_env.EVAL_BATCH + '/trajectories'))
HERE = Path(__file__).parent; C = HERE / 'eval_gm/fast/camera_b4'; C4 = HERE / 'eval_gm/fast/camera_a4'


def yaw(c2w):
    R = np.einsum('ji,njk->nik', c2w[0, :3, :3], c2w[:, :3, :3]); return np.degrees(np.arctan2(R[:, 0, 2], R[:, 2, 2]))


def rel_t(c2w): return np.einsum('ji,nj->ni', c2w[0, :3, :3], c2w[:, :3, 3] - c2w[0, :3, 3])


out = json.loads((C / 'exec.json').read_text()) if (C / 'exec.json').exists() else {}
for s in ('02267acf', '0238de14'):
    with np.load(B / f'{s}.npz') as z: full = np.linalg.inv(z['w2c_overrun'].astype(float))
    print(f'\n{s}')
    for spec in sys.argv[1:]:
        n, st = spec.split(':'); st = int(st); f = next((d / f'{n}_{s}_trajectories/baseline.npz' for d in (C, C4) if (d / f'{n}_{s}_trajectories/baseline.npz').exists()), None)
        if f is None: print(f'{n:22s} missing'); continue
        with np.load(f) as z: m = z['c2w'].astype(float)
        plan = full[::st][:len(m)]; py, pt, my, mt = yaw(plan), rel_t(plan), yaw(m), rel_t(m); T = [t for t in (48, 112, 160, 320) if t < len(m)]; r = {'frames': len(m), 'stride': st}
        for t in T:
            d = pt[t]; e = float(mt[t] @ d / max(1e-9, d @ d)); r[f'exec{t}'] = round(e, 3); r[f'lat{t}'] = round(float(np.linalg.norm(mt[t] - e * d) / max(1e-9, np.linalg.norm(d))), 3)
            r[f'yaw_ratio{t}'] = round(float(my[t] / py[t]), 3) if abs(py[t]) > 1 else None
        seg = [float(np.linalg.norm(mt[a + 16] - mt[a])) / max(1e-9, float(np.linalg.norm(pt[a + 16] - pt[a]))) for a in range(0, len(m) - 16, 16)]   # path length per call, measured / planned
        r['step_ratio_median'] = round(float(np.median(seg)), 3); out[f'{s}|{n}'] = r
        print(f'{n:22s} stride {st} ' + ' '.join(f'@{t}: exec {r[f"exec{t}"]:.2f} lat {r[f"lat{t}"]:.2f} yaw {r[f"yaw_ratio{t}"] if r[f"yaw_ratio{t}"] is not None else float("nan"):.2f} |' for t in T) + f' per-call step ratio median {r["step_ratio_median"]:.2f}')
(C / 'exec.json').write_text(json.dumps(out, indent=1))
