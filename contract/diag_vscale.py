"""Vertical scale of the generated frames against the recording (critique 10 Nov 19:10). Every 540-row render was stretched vertically by
540/512; did the outputs inherit a vertical scale, and does a corrected render get 'un-stretched' again (compounding squeeze)?
Per frame pair (generated t, recorded t): SIFT matches -> robust fit of x' = sx x + tx, y' = sy y + ty (no rotation / shear, RANSAC) mapping
the recording onto the generated frame; sy / sx is the anisotropy (isotropic zoom from a forward-position error cancels; camera error and
parallax add noise -> Lyra-2 / Spatia / a second recording frame are the floor). > 1 = generated content is vertically stretched.
Usage: diag_vscale.py [name=run_dir_with_{scene} ...] -> eval_gm/diag/vscale.json"""
import json, sys
from pathlib import Path
import numpy as np, cv2
extra = [a for a in sys.argv[1:] if '=' in a]; sys.argv = sys.argv[:1]
import diag_seams_sota as ds
HERE = Path(__file__).parent; OUT = HERE / 'eval_gm/diag/vscale.json'; T = [1, 4, 8, 16, 32, 48, 64, 96, 128, 160, 240, 320]
sift = cv2.SIFT_create(4000); bf = cv2.BFMatcher()


def fit(a, b, rng):
    """a = recording frame, b = generated (uint8 RGB 480x864). Returns sx, sy, inliers."""
    ka, da = sift.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), None); kb, db = sift.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_RGB2GRAY), None)
    if da is None or db is None or len(ka) < 20 or len(kb) < 20: return None
    m = [x for x, y in bf.knnMatch(da, db, k=2) if x.distance < .75 * y.distance]
    if len(m) < 30: return None
    p = np.float64([ka[x.queryIdx].pt for x in m]); q = np.float64([kb[x.trainIdx].pt for x in m]); best = None
    for _ in range(400):
        i, j = rng.choice(len(m), 2, replace=False); dx, dy = p[j] - p[i]
        if abs(dx) < 40 or abs(dy) < 40: continue
        sx = (q[j, 0] - q[i, 0]) / dx; sy = (q[j, 1] - q[i, 1]) / dy
        if not (.7 < sx < 1.4 and .7 < sy < 1.4): continue
        r = np.hypot(sx * (p[:, 0] - p[i, 0]) + q[i, 0] - q[:, 0], sy * (p[:, 1] - p[i, 1]) + q[i, 1] - q[:, 1]); inl = r < 4
        if best is None or inl.sum() > best.sum(): best = inl
    if best is None or best.sum() < 25: return None
    A = np.stack([p[best, 0], np.ones(best.sum())], 1); sx = np.linalg.lstsq(A, q[best, 0], rcond=None)[0][0]
    A = np.stack([p[best, 1], np.ones(best.sum())], 1); sy = np.linalg.lstsq(A, q[best, 1], rcond=None)[0][0]
    return float(sx), float(sy), int(best.sum())


res = json.loads(OUT.read_text()) if OUT.exists() else {}
for scene in ('02267acf', '0238de14'):
    src = ds.sources(scene); todo = {k: src[v] for k, v in (('GM', 'Ours'), ('GM-B', 'GM-B'), ('Lyra-2', 'Lyra-2'), ('Spatia', 'Spatia')) if v in src}
    todo.update({a.split('=')[0]: a.split('=', 1)[1].format(scene=scene) for a in extra})
    gt = {}
    for t, f in enumerate(ds.frames_of('GT', scene)):
        if t > 322: break
        if t in T or t - 2 in T: gt[t] = f if f.shape[:2] == (480, 864) else cv2.resize(f, (864, 480), interpolation=cv2.INTER_AREA)
    todo['recording t+2'] = None
    print(f'\n{scene}: sy/sx (inliers) at frames ' + ' '.join(f'{t:>5d}' for t in T))
    for name, d in todo.items():
        key = f'{scene}|{name}'
        if key not in res:
            rng = np.random.default_rng(0); row = {}
            if d is None: fr = {t: gt[t + 2] for t in T}
            else:
                fr = {}
                for t, f in enumerate(ds.frames_of(d, scene)):
                    if t > 320: break
                    if t in T: fr[t] = f if f.shape[:2] == (480, 864) else cv2.resize(f, (864, 480), interpolation=cv2.INTER_AREA)
            for t in T:
                if t in fr: r = fit(gt[t], fr[t], rng); row[str(t)] = None if r is None else dict(sx=round(r[0], 4), sy=round(r[1], 4), ratio=round(r[1] / r[0], 4), n=r[2])
            res[key] = row; OUT.write_text(json.dumps(res))
        row = res[key]; print(f'{name:14s} ' + ' '.join((f"{row[str(t)]['ratio']:.3f}" if row.get(str(t)) else '  -  ') for t in T) + '   n: ' + ' '.join((str(row[str(t)]['n']) if row.get(str(t)) else '-') for t in T))
