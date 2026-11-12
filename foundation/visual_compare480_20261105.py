"""Deep visual check, 480 frames (5 Nov). Three sheet types, all read from the exact videos the v3 score files name (scorer's
reader order: frames.npz, <scene>_frames/, mp4), keeping only sampled frames (host RAM shared with a running Lyra job).
  strips480/<scene>_strip480.jpg  : GT + baseline/correction of ours, Lyra-2, Spatia, Matrix-Game at frames 0..480 step 80
  detail/<scene>_detail.jpg       : GT / ours corr / Lyra-2 / Spatia / Matrix-Game as released at 240, 360, 480, 640x356 tiles
  reset/<scene>_reset.jpg         : GT / ours corr / reset160 at 160, 176, 240, 320, 400, 480 (reset = recorded history installed at 160)
Per-row captions carry the score-file numbers for the same arm so what is seen can be matched with what was scored."""
import gm_env  # machine paths: gm_env.py at the repository root (scripts/env.sh puts it on PYTHONPATH)
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import comparison_table_20261105 as ct

HERE = Path(__file__).resolve().parent; OUT = HERE / 'visual_compare480_20261105'
H = 480; TW, TH = 432, 240
FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 18)
FONT_S = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
GT_ROWS = {r['name']: r for r in json.load(open((gm_env.EVAL_BATCH + '/index.json')))['scenes']}
RESET = Path(gm_env.EVAL_INPUTS)


def sampled(src, scene, steps):
    import cv2
    src = Path(src)
    if (src / 'frames.npz').exists():
        with np.load(src / 'frames.npz') as z:
            fr = z['frames']
            if len(fr) < max(steps) + 1 or fr.dtype != np.uint8:
                raise ValueError(f'{src}: {fr.shape} {fr.dtype}')
            return [fr[t].copy() for t in steps]
    if (src / f'{scene}_frames').exists():
        return [np.asarray(Image.open(src / f'{scene}_frames' / f'frame_{t:05d}.png').convert('RGB')) for t in steps]
    path = src if src.is_file() else src / 'review_full.mp4'; cap = cv2.VideoCapture(str(path)); keep = {}
    for i in range(max(steps) + 1):
        ok, im = cap.read()
        if not ok:
            raise ValueError(f'{path}: missing frame {i}')
        if i in steps:
            keep[i] = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    cap.release(); return [keep[t] for t in steps]


def gt(scene, steps):
    row = GT_ROWS[scene]
    return [np.asarray(Image.open(Path(row['frames_dir']) / f"frame_{row['original_start'] + t + 1:05d}.jpg").convert('RGB')) for t in steps]


def tile(im, w, h):
    img = Image.fromarray(im); s = min(w / img.width, h / img.height); img = img.resize((round(img.width * s), round(img.height * s)), Image.BICUBIC)
    c = Image.new('RGB', (w, h)); c.paste(img, ((w - img.width) // 2, (h - img.height) // 2)); return c


def sheet(scene, rows, steps, tw, th, path, lw=300):
    """rows: list of (name, caption, frames)."""
    s = Image.new('RGB', (lw + tw * len(steps), 34 + th * len(rows)), (255, 255, 255)); d = ImageDraw.Draw(s)
    d.text((8, 6), scene, fill=(0, 0, 0), font=FONT)
    for c, t in enumerate(steps):
        d.text((lw + c * tw + tw // 2 - 30, 6), f'frame {t}', fill=(0, 0, 0), font=FONT)
    for r, (name, cap, fr) in enumerate(rows):
        y = 34 + r * th; d.text((8, y + th // 2 - 24), name, fill=(0, 0, 0), font=FONT)
        if cap:
            for k, line in enumerate(cap.split('\n')):
                d.text((8, y + th // 2 + 2 + 20 * k), line, fill=(90, 90, 90), font=FONT_S)
        for c, im in enumerate(fr):
            s.paste(tile(im, tw, th), (lw + c * tw, y))
    s.save(path, quality=88)


def arm(model, scene, a):
    f, d = ct.score_file(model, scene)
    if d is None or a not in d['arms']:
        return None
    x = d['arms'][a]
    fm = lambda v, n: '—' if v is None else f'{v:.{n}f}'
    cap = f"pos {fm(x['camera_mean'], 4)} rot {fm(x['rotation_mean'], 2)}\nLPIPS {x['lpips']:.3f} DINO {x['dino']:.3f} plr {fm(None if x.get('path_length_ratio') in (None, 'None') else float(x['path_length_ratio']), 2)}"
    return x['source'], cap


def main(which):
    MODELS = [('Ours', 'ours'), ('Lyra-2', 'lyra'), ('Spatia', 'spatia'), ('Matrix-Game 3.5 Distilled', 'matrixgame'), ('Matrix-Game 3.5 Base', 'matrixgame_base')]
    if 'strips' in which:
        (OUT / 'strips480').mkdir(parents=True, exist_ok=True); steps = [0, 80, 160, 240, 320, 400, 480]
        for scene in ['02267acf', '0238de14', '334ff324', '61b27b61', 'a126efe8', '924228bf']:
            rows = [('Ground truth', '', gt(scene, steps))]
            for label, m in MODELS:
                for a in ('baseline', 'correction'):
                    r = arm(m, scene, a)
                    if r is not None:
                        rows.append((f'{label}{" + corr" if a == "correction" else ""}', r[1], sampled(r[0], scene, steps)))
            sheet(scene, rows, steps, TW, TH, OUT / 'strips480' / f'{scene}_strip480.jpg'); print('strip', scene, len(rows), flush=True)
    if 'detail' in which:
        (OUT / 'detail').mkdir(parents=True, exist_ok=True); steps = [240, 360, 480]
        for scene in ['02267acf', '0238de14', '334ff324', '61b27b61', 'a126efe8']:
            rows = [('Ground truth', '', gt(scene, steps))]
            for label, m in MODELS:                         # user 5 Nov: compare ours + correction with the other models as released
                a = 'correction' if m == 'ours' else 'baseline'; r = arm(m, scene, a)
                if r is not None:
                    rows.append((f'{label}{" + corr" if a == "correction" else ""}', r[1], sampled(r[0], scene, steps)))
            sheet(scene, rows, steps, 640, 356, OUT / 'detail' / f'{scene}_detail.jpg', lw=260); print('detail', scene, flush=True)
    if 'reset' in which:
        (OUT / 'reset').mkdir(parents=True, exist_ok=True); steps = [160, 176, 240, 320, 400, 480]
        for f in sorted((RESET / 'scores_reset160').glob('*.json')):
            scene = f.stem
            if scene == 'analysis':
                continue
            d = json.load(open(f)); rows = [('Ground truth', '', gt(scene, steps))]
            for a, label in (('correction', 'Ours corr (original)'), ('reset160', 'Reset at 160')):
                x = d['arms'][a]; cap = f"LPIPS {x['lpips']:.3f} DINO {x['dino']:.3f}\npos {x['camera_mean'] if x['camera_mean'] is None else round(x['camera_mean'], 4)}"
                src = Path(x['source']); src = src if src.is_absolute() else RESET / src
                rows.append((label, cap, sampled(src, scene, steps)))
            sheet(scene, rows, steps, 432, 240, OUT / 'reset' / f'{scene}_reset.jpg'); print('reset', scene, flush=True)


if __name__ == '__main__':
    main(sys.argv[1:] or ['strips', 'detail', 'reset'])
