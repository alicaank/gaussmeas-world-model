"""Build the 864x480 training corpus from data/DL3DV-960P-864x480 (958 scenes = 989 streams of the 448 corpus, same frame
windows): per stream vae.npz (bf16 latents [1,48,L,30,54], K864, canonical c2w in the anchor-depth gauge reused from the
448 anchor.json, metres_per_anchor_unit = median DA3 metric depth of the 864 anchor), text reused from the 448 corpus.
Output: data864/components/<stream_id>/{vae.npz, meta.json}, data864/streams.jsonl (20 held-out dev streams)."""
import argparse, json, os, random, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
from all_scene_dataset import DATA, pack_bf16
from foundation_contract import normalize_cameras
from gaussverse.models.vae_stream import VaeStream
from gaussverse.data.dl3dv import scale_intrinsics

ROOT = Path('/home/hj/Alican/data/DL3DV-960P-864x480'); OUT = Path(__file__).parent / 'data864'
ap = argparse.ArgumentParser(); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--skip_flagged', type=int, default=1); ap.add_argument('--dev', type=int, default=20); a = ap.parse_args()
(OUT / 'components').mkdir(parents=True, exist_ok=True)
rows = [json.loads(l) for l in (DATA / 'corpus/streams.jsonl').open()]
flag = set(json.load(open('/home/hj/Alican/analysis/plan_base_restart_20261008/camera_quality_20261015/manifest.json'))['streams'])
done = {d.name: d for s in ROOT.glob('*') if s.is_dir() for d in s.glob('*') if (d / 'done').exists()}
rows = [r for r in rows if r['scene_id'] in done and (not a.skip_flagged or r['stream_id'] not in flag)]
rows.sort(key=lambda r: r['stream_id']); random.Random(864).shuffle(rows); dev_ids = {r['stream_id'] for r in rows[:a.dev]}
rows = rows[:a.limit] if a.limit else rows
from model_io import load_vae
vae = load_vae().eval()
from realized_pose import load_da3
da3 = load_da3('cuda')


def intrinsics(td, crop):
    (sw, sh), (x0, y0, x1, y1), (ow, oh) = crop['source_size'], crop['crop_box'], crop['out_size']
    s1 = sw / td['w']; sx = ow / (x1 - x0); sy = oh / (y1 - y0)
    return np.array([[td['fl_x'] * s1 * sx, 0, (td['cx'] * s1 - x0) * sx], [0, td['fl_y'] * s1 * sy, (td['cy'] * s1 - y0) * sy], [0, 0, 1]], np.float32)


def calibrate(sid, r):
    """metres per anchor-gauge unit, the qualification way: DA3 metric depth of the 448x256 anchor / stored gauge depth,
    pixel median over confident pixels. Checks: metric prediction, >= 10k valid pixels, finite positive scale."""
    from all_scene_dataset import RecordedStream
    za = np.load(DATA / 'components' / sid / 'anchor.npz'); gauge = za['depth']; conf = za['confidence']; K448 = za['K'].astype(np.float32)
    with torch.no_grad():
        pred = da3.inference(image=[RecordedStream(r).frame(0)], intrinsics=K448[None], extrinsics=None, align_to_input_extrinsics=False, align_to_input_ext_scale=False, infer_gs=False, process_res=504, export_dir=None)
    is_metric = bool(getattr(pred, 'is_metric', True))
    metric = np.asarray(pred.depth[0], np.float64)
    if metric.shape != gauge.shape:
        metric = torch.nn.functional.interpolate(torch.as_tensor(metric)[None, None].float(), size=gauge.shape, mode='bilinear', align_corners=False)[0, 0].numpy()
    ok = np.isfinite(metric) & (metric > 0) & np.isfinite(gauge) & (gauge > 0) & (conf > .5); ratio = metric[ok] / gauge[ok]
    mpu = float(np.median(ratio)) if ok.sum() else float('nan'); spread = float(np.median(np.abs(np.log(ratio / mpu)))) if ok.sum() else float('nan')
    valid = bool(is_metric and ok.sum() >= 10000 and np.isfinite(mpu) and mpu > 0)
    return dict(metres_per_anchor_unit=mpu, mpu_method='da3_metric_over_gauge_pixel_median', mpu_spread=spread, mpu_valid_pixels=int(ok.sum()),
                mpu_is_metric=is_metric, mpu_valid=valid, gauge_median=float(np.median(gauge[ok])) if ok.sum() else float('nan'))


log = open(OUT / 'prep.log', 'a'); manifest = []
for i, r in enumerate(rows):
    sid = r['stream_id']; d = done[r['scene_id']]; dst = OUT / 'components' / sid; t0 = time.time()
    if (dst / 'vae.npz').exists():
        m = json.load(open(dst / 'meta.json'))
        if m.get('mpu_method') == 'da3_metric_over_gauge_pixel_median' or 'mpu_median_metric_depth' in m:   # calibrated (prep or fix_mpu)
            manifest.append(m); continue
        cal = calibrate(sid, r)                                              # uncalibrated record: fix the metadata only, keep the latents
        m.update(mpu_median_metric_depth=m['metres_per_anchor_unit'], **cal); json.dump(m, open(dst / 'meta.json', 'w')); manifest.append(m); continue
    td = json.loads((d / 'transforms.json').read_text()); crop = json.loads((d / 'crop.json').read_text())
    frames = sorted(td['frames'], key=lambda f: f['file_path'])[r['original_start']:r['original_end_exclusive']]
    n_img = len(sorted((d / 'frames_864x480').glob('*.jpg')))
    if len(frames) != r['raw_frames'] or n_img < r['original_end_exclusive']:
        log.write(json.dumps(dict(stream=sid, skip='frame count mismatch', frames=len(frames), imgs=n_img)) + '\n'); continue
    gl = np.asarray([f['transform_matrix'] for f in frames], np.float32); c2w_raw = torch.from_numpy(gl @ np.diag([1., -1., -1., 1.]).astype(np.float32))
    ann = json.loads((DATA / 'components' / sid / 'anchor.json').read_text())
    c2w = normalize_cameras(c2w_raw, c2w_raw[0], ann['registered_units_per_anchor_depth']); K = intrinsics(td, crop)
    L = (len(frames) - 1) // 4 + 1; T = 1 + 4 * (L - 1)
    def img(j):
        return np.array(Image.open(d / 'frames_864x480' / f'frame_{r["original_start"] + j + 1:05d}.jpg').convert('RGB'))
    with torch.no_grad():
        enc = VaeStream(vae); parts = []
        for lo, hi in [(0, min(33, T))] + [(s, min(s + 32, T)) for s in range(33, T, 32)]:
            x = torch.from_numpy(np.stack([img(j) for j in range(lo, hi)])).permute(3, 0, 1, 2).float()[None] / 127.5 - 1
            parts.append(enc.encode_chunk(x.cuda().bfloat16()).float().cpu())
        lat = torch.cat(parts, 2)
        if lat.shape[2] != L:
            raise RuntimeError(f'{sid}: {lat.shape[2]} latents for {T} frames (expected {L})')
        cal = calibrate(sid, r)
    meta = dict(stream_id=sid, scene_id=r['scene_id'], split='development' if sid in dev_ids else 'train', latent_count=L, frames=T, **cal,
                gauge=ann['registered_units_per_anchor_depth'], gauge_id=ann['gauge_id'], K=K.tolist(), source=str(d), text=str(DATA / 'components' / sid / 'text.npz'),
                anchor_json=str(DATA / 'components' / sid / 'anchor.json'), original_start=r['original_start'], flagged=sid in flag)
    np.savez_compressed(dst.mkdir(parents=True, exist_ok=True) or dst / 'vae.npz', latents_bf16=pack_bf16(lat.bfloat16().float().numpy()), K=K, c2w=c2w.numpy().astype(np.float32))
    (dst / 'meta.json').write_text(json.dumps(meta)); manifest.append(meta)
    log.write(json.dumps(dict(stream=sid, L=L, mpu=round(cal['metres_per_anchor_unit'], 3), valid=cal['mpu_valid'], sec=round(time.time() - t0, 1))) + '\n'); log.flush()
    if i % 20 == 0:
        print(i, len(rows), sid[:8], 'L', L, 'mpu', round(cal['metres_per_anchor_unit'], 2), round(time.time() - t0, 1), 's', flush=True)
with open(OUT / 'streams.jsonl', 'w') as f:
    for m in manifest:
        f.write(json.dumps(m) + '\n')
print('done', len(manifest), 'streams; dev', sum(m['split'] == 'development' for m in manifest)); os._exit(0)
