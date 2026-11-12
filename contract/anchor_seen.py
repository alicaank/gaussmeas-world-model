"""Selective trust (critique 11 Nov 15:00): which render cells the ANCHOR sees. The anchor is clean at inference (recorded image), everything else in
the memory is generated; corrupt only the cells the anchor does not see, and tell the model which is which through the conf channel (conf = 1 on
anchor-seen cells, at training AND inference).
  precompute (GPU, DA3):  anchor_seen.py precompute  -> anchor_depth/<stream_id>.npz  (gauge-unit depth of the anchor at 120x216, K at that size)
  seen_cells(depth, K, targets): z-buffered splat (renderer864.warp_points) of the anchor into each target camera, fraction covered per latent cell
  (30x54, 4x4 pixels at 120x216) -> bool (n_t, 30, 54) with frac > .5."""
import gm_env  # machine paths: gm_env.py at the repository root (scripts/env.sh puts it on PYTHONPATH)
import json, sys
from pathlib import Path
import numpy as np, torch
FT = Path(gm_env.CORE)
for p in (str(FT), gm_env.WORKSPACE):
    if p not in sys.path: sys.path.insert(0, p)
HERE = Path(__file__).resolve().parent; OUT = HERE / 'anchor_depth'; SIZE = (120, 216)


def scaled_K(K864):
    K = np.asarray(K864, np.float64).copy(); K[0] *= SIZE[1] / 864; K[1] *= SIZE[0] / 480; return K


def anchor_depth(da3, rgb864, K864, mpu):
    """Gauge-unit depth of one 864x480 frame at SIZE (same DA3 call as the renderer, single view = its own metric depth)."""
    from renderer864 import da3_gauge_depths
    import cv2
    d, _ = da3_gauge_depths(da3, np.asarray(rgb864)[None], np.asarray(K864, np.float64), np.eye(4)[None], float(mpu))
    return cv2.resize(d[0].numpy().astype(np.float32), SIZE[::-1], interpolation=cv2.INTER_NEAREST)


def seen_cells(depth, K, targets, thr=.5):
    from renderer864 import warp_points
    rgb = np.zeros((*SIZE, 3), np.uint8); out = []
    for t in targets:
        _, cov = warp_points(rgb, depth, K, np.eye(4), np.asarray(t, np.float64), radius=1)
        out.append(cov.reshape(30, 4, 54, 4).mean((1, 3)) > thr)
    return np.stack(out)


_cache = {}
def load(stream_id):
    if stream_id not in _cache:
        f = OUT / f'{stream_id}.npz'
        if not f.exists(): raise FileNotFoundError(f'no anchor depth for {stream_id}: run anchor_seen.py precompute')
        with np.load(f) as z: _cache[stream_id] = (z['depth'].astype(np.float32), z['K'])
        if len(_cache) > 2000: _cache.pop(next(iter(_cache)))
    return _cache[stream_id]


def row_seen(stream, row):
    """(4,30,54) bool for the four render slots of a training row (chunk-0 cameras = recorded c2w of latents start..start+3, gauge units)."""
    depth, K = load(stream.row['stream_id'])
    return seen_cells(depth, K, [stream.c2w[stream.frame_index(p)].numpy() for p in range(row['start'], row['start'] + 4)])


if __name__ == '__main__' and sys.argv[1:2] == ['precompute']:
    from PIL import Image
    from realized_pose import load_da3
    from data import corpus864
    OUT.mkdir(exist_ok=True); da3 = load_da3('cuda'); rows = corpus864('train'); rows = rows[0] if isinstance(rows, tuple) else rows; n = 0; fr = []
    for r in rows:
        f = OUT / f"{r['stream_id']}.npz"
        if f.exists(): continue
        img = np.array(Image.open(Path(r['source']) / 'frames_864x480' / f"frame_{r['original_start'] + 1:05d}.jpg").convert('RGB'))
        d = anchor_depth(da3, img, r['K'], r['metres_per_anchor_unit'])
        np.savez_compressed(f, depth=d.astype(np.float16), K=scaled_K(r['K']), median=np.float32(np.median(d[d > 0])))
        fr.append(float(np.median(d[d > 0]))); n += 1
        if n % 100 == 0: print('done', n, 'median depth (gauge, ~1 expected)', round(float(np.median(fr)), 3), flush=True)
    (OUT / '_done').write_text(json.dumps(dict(streams=len(rows), new=n, median_depth=float(np.median(fr)) if fr else None)))
    print('ANCHOR_DEPTH_DONE', n, flush=True)
