"""Streamed DL3DV-960P subset: download one scene zip, keep frames as 864x480 JPEG (q95) + transforms.json, delete the PNGs.

Scenes = training scenes of our corpus (dataset_v3 streams, split=train), in a fixed seeded order, so 480P and 960P
can be compared on the same scenes. Development / held-out scenes are never selected.
Output: <out>/<split>/<scene>/{frames_864x480/frame_XXXXX.jpg, transforms.json, crop.json}
Writes progress.jsonl (per scene: download MB, seconds, output MB) and aborts below --min-free-gb.
"""
import gm_env  # machine paths: gm_env.py at the repository root (scripts/env.sh puts it on PYTHONPATH)
import argparse, json, os, random, shutil, time, zipfile
from pathlib import Path
from PIL import Image
from huggingface_hub import hf_hub_download

REPO = 'DL3DV/DL3DV-ALL-960P'
DATA = Path((gm_env.FOUNDATION + '/dataset_v3'))
W, H = 864, 480


def free_gb(p):
    s = os.statvfs(p); return s.f_bavail * s.f_frsize / 2**30


def crop_box(iw, ih, w=W, h=H):
    aspect = w / h
    if iw / ih > aspect:
        nw = int(ih * aspect); return ((iw - nw) // 2, 0, (iw - nw) // 2 + nw, ih)
    nh = int(iw / aspect); return (0, (ih - nh) // 2, iw, (ih - nh) // 2 + nh)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True); ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--seed', type=int, default=20261016); ap.add_argument('--min-free-gb', type=float, default=40.0); a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True); tmp = a.out / '_tmp'; tmp.mkdir(exist_ok=True)
    rows = [json.loads(l) for l in open(DATA / 'corpus/streams.jsonl')]
    held = {r['scene_id'] for r in rows if r['split'] != 'train'}
    scenes = {}
    for r in rows:
        if r['split'] == 'train' and r['scene_id'] not in held and r['latent_count'] >= 41:
            split = Path(r['scene_path']).parent.name; scenes.setdefault(r['scene_id'], split)
    order = sorted(scenes); random.Random(a.seed).shuffle(order); order = order[:a.n]
    json.dump(dict(scenes=[dict(scene_id=s, split=scenes[s]) for s in order], seed=a.seed), open(a.out / 'selection.json', 'w'), indent=1)
    log = open(a.out / 'progress.jsonl', 'a')
    for i, sid in enumerate(order):
        split = scenes[sid]; dst = a.out / split / sid
        if (dst / 'done').exists():
            continue
        if free_gb(a.out) < a.min_free_gb:
            print('disk floor reached', free_gb(a.out)); break
        t0 = time.time()
        try:
            z = hf_hub_download(REPO, f'{split}/{sid}.zip', repo_type='dataset', local_dir=tmp)
            zmb = os.path.getsize(z) / 2**20; t1 = time.time()
            ex = tmp / 'ex'; shutil.rmtree(ex, ignore_errors=True)
            with zipfile.ZipFile(z) as zf:
                zf.extractall(ex)
            os.remove(z)
            src = next(ex.rglob('transforms.json')).parent
            pngs = sorted((src / 'images_4').glob('*.png')) if (src / 'images_4').is_dir() else sorted(src.rglob('*.png'))
            if len(pngs) < 2:
                raise ValueError('archive without frames')
            (dst / 'frames_864x480').mkdir(parents=True, exist_ok=True)
            iw, ih = Image.open(pngs[0]).size; box = crop_box(iw, ih)
            for p in pngs:
                Image.open(p).convert('RGB').crop(box).resize((W, H), Image.LANCZOS).save(dst / 'frames_864x480' / (p.stem + '.jpg'), quality=95)
            shutil.copy(src / 'transforms.json', dst / 'transforms.json')
            json.dump(dict(source_size=[iw, ih], crop_box=box, out_size=[W, H], frames=len(pngs)), open(dst / 'crop.json', 'w'))
            shutil.rmtree(ex, ignore_errors=True); (dst / 'done').touch()
            out_mb = sum(f.stat().st_size for f in dst.rglob('*') if f.is_file()) / 2**20
            rec = dict(i=i, scene=sid, split=split, frames=len(pngs), source=[iw, ih], zip_MB=round(zmb, 1), download_s=round(t1 - t0, 1),
                       total_s=round(time.time() - t0, 1), out_MB=round(out_mb, 1), free_GB=round(free_gb(a.out), 1))
        except Exception as e:
            rec = dict(i=i, scene=sid, split=split, error=repr(e)[:200])
            shutil.rmtree(tmp / 'ex', ignore_errors=True)
        log.write(json.dumps(rec) + '\n'); log.flush(); print(json.dumps(rec), flush=True)


if __name__ == '__main__':
    main()
