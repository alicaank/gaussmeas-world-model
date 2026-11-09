"""Visual comparison of the Gauss-meas line (eval_gm/<arm>/comparison_<sid>.mp4 from closed_loop_864: left recorded, right
generated, 32-px label band, all 321 frames). Contact sheets (frames 16/80/160/240/320, per-call LPIPS from eval.json) and 2x2
videos (recorded | arm1 / arm2 | arm3), streamed frame by frame. CPU only.
Usage: visual_compare_gm.py --arms gm_s7 gm_b_s7 gm_f_fmask_s7 [--out eval_gm/visual]"""
import argparse
import json
from pathlib import Path
import numpy as np
import imageio.v2 as iio
from PIL import Image, ImageDraw
from visual_compare import font

HERE = Path(__file__).resolve().parent
LABEL = {'gm_s7': 'Gauss-meas (GM)', 'gm_b_s7': 'GM-B (bağlam bozulması)', 'gm_fmask_s7': 'GM + sınır maskesi (eğitimsiz)',
         'gm_f_fmask_s7': 'GM-F (bozulma + sınır maskesi eğitimi)'}
COLS = [16, 80, 160, 240, 320]


def frames_at(path, ks):
    r = iio.get_reader(path); got = {}
    try:
        for k, fr in enumerate(r):
            if k in ks: got[k] = fr
            if len(got) == len(ks): break
    finally: r.close()
    return got


def lpips_by_call(ev, arm, sid):
    e = json.loads((ev / arm / 'eval.json').read_text())
    s = next(x for x in e['scenes'] if x['stream'].startswith(sid[:8]))
    return {i + 1: p['lpips'] for i, p in enumerate(s['per_call'])}


def sheet(ev, sid, arms, out):
    tw, th, lw, hh = 432, 240, 250, 34; ks = set(c - 1 for c in COLS)
    rows = []
    for i, a in enumerate(arms):
        got = frames_at(ev / a / f'comparison_{sid[:8]}.mp4', ks)
        if i == 0: rows.append(('Kayıt (gerçek)', [got[c - 1][32:, :864] for c in COLS], None))
        rows.append((LABEL.get(a, a), [got[c - 1][32:, 864:] for c in COLS], lpips_by_call(ev, a, sid)))
    img = Image.new('RGB', (lw + tw * len(COLS), hh + th * len(rows)), 'white'); dr = ImageDraw.Draw(img); f, fs = font(17), font(14)
    for j, c in enumerate(COLS): dr.text((lw + j * tw + 6, 8), f'RGB {c}  (çağrı {max(1, (c + 15) // 16)})', fill='black', font=f)
    for i, (name, frames, lp) in enumerate(rows):
        y = hh + i * th; dr.multiline_text((8, y + th // 2 - 20), name.replace(' (', '\n('), fill='black', font=f)
        for j, (c, fr) in enumerate(zip(COLS, frames)):
            img.paste(Image.fromarray(fr).resize((tw, th), Image.BILINEAR), (lw + j * tw, y))
            call = max(1, (c + 15) // 16)
            if lp and call in lp:
                dr.rectangle((lw + j * tw + 4, y + th - 24, lw + j * tw + 112, y + th - 4), fill='black'); dr.text((lw + j * tw + 8, y + th - 23), f'LPIPS {lp[call]:.2f}', fill='white', font=fs)
    img.save(out, quality=88)


def video(ev, sid, arms, out):
    readers = [iio.get_reader(ev / a / f'comparison_{sid[:8]}.mp4') for a in arms]
    names = ['Kayıt (gerçek)'] + [LABEL.get(a, a) for a in arms]; f = font(22)
    w = iio.get_writer(out, fps=16, codec='libx264', quality=7, macro_block_size=16, ffmpeg_params=['-pix_fmt', 'yuv420p'])
    try:
        for k in range(321):
            fr = [r.get_data(k) for r in readers]
            panels = [fr[0][32:, :864]] + [x[32:, 864:] for x in fr]
            grid = np.concatenate([np.concatenate(panels[:2], 1), np.concatenate(panels[2:], 1)], 0)
            im = Image.fromarray(grid); dr = ImageDraw.Draw(im)
            for p, n in enumerate(names):
                x, y = (p % 2) * 864, (p // 2) * 480; dr.rectangle((x, y, x + 12 + 13 * len(n), y + 32), fill='black'); dr.text((x + 6, y + 3), n, fill='white', font=f)
            dr.rectangle((1728 - 230, 0, 1728, 32), fill='black'); dr.text((1728 - 224, 3), f'RGB {k}  çağrı {max(1, (k + 15) // 16)}', fill='yellow', font=f)
            w.append_data(np.asarray(im))
    finally:
        w.close(); [r.close() for r in readers]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--arms', nargs='+', default=['gm_s7', 'gm_b_s7', 'gm_f_fmask_s7'])
    ap.add_argument('--eval', type=Path, default=HERE / 'eval_gm'); ap.add_argument('--out', type=Path, default=None); ap.add_argument('--no-video', action='store_true')
    a = ap.parse_args(); out = a.out or a.eval / 'visual'; out.mkdir(parents=True, exist_ok=True)
    sids = [p.stem.split('_', 1)[1] for p in sorted((a.eval / a.arms[0]).glob('comparison_*.mp4'))]
    for sid in sids:
        sheet(a.eval, sid, a.arms, out / f'sheet_{sid}.jpg'); print('sheet', sid, flush=True)
        if not a.no_video and len(a.arms) == 3: video(a.eval, sid, a.arms, out / f'grid_{sid}.mp4'); print('video', sid, flush=True)


if __name__ == '__main__':
    main()
