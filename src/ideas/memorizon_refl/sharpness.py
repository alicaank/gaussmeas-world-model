"""Laplacian-variance sharpness of generated frames per arm (calls 1-5 / 6-10); usage: sharpness.py <runs_root>."""
import sys, glob, json
from pathlib import Path
import numpy as np
from scipy.ndimage import laplace
root = Path(sys.argv[1]); out = {}
for d in sorted(p for p in root.iterdir() if p.is_dir()):
    v = [[laplace(fr[i].astype(np.float32).mean(-1)).var() for i in range(0, 160, 8)] for fr in (np.load(f)['frames'] for f in sorted(d.glob('*_s*.npz')))]
    if v:
        v = np.array(v); out[d.name] = dict(n=len(v), calls1_5=float(v[:, :10].mean()), calls6_10=float(v[:, 10:].mean()))
json.dump(out, open(root / 'sharpness.json', 'w'), indent=1); [print(k, {a: round(b, 1) for a, b in e.items()}) for k, e in out.items()]
