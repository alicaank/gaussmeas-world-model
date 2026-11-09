"""Visual check of the frontier masks on recorded frames (no DiT): sources = anchor + recorded latents 1..3, target = a recorded
command --target_latent..+3 so new space is revealed. Rows: recorded target | ReSplat render | ReSplat valid (alpha) | observed
'support' (points) | observed 'reproj' (render depth vs sources) | observed 'true' (DA3 depth of the recorded targets vs sources,
the training-cache definition) | render tokens kept under 'reproj'. Usage: frontier_smoke.py --out DIR [--scene_offset 2] [--target_latent 20]"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw

FT = '/home/hj/Alican/analysis/plan_base_restart_20261008/fresh_teacher_20261021'
HERE = str(Path(__file__).resolve().parent)
for p in (HERE, FT):
    if p not in sys.path:
        sys.path.insert(0, p)
import frontier  # noqa: E402


def to_img(x):
    if x.ndim == 3:
        return ((x.permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5).round().byte().numpy()
    return (x.clamp(0, 1) * 255).round().byte()[:, :, None].expand(-1, -1, 3).numpy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True); ap.add_argument('--scene_offset', type=int, default=2)
    ap.add_argument('--target_latent', type=int, default=20); ap.add_argument('--close', type=int, default=1); ap.add_argument('--tol', type=float, default=0.15)
    ap.add_argument('--src_latents', default='0,1,2,3', help='source latents (anchor + prefix views); rollout-like: 0,17,18,19 for target 20'); a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    from data import corpus864, Stream864
    from render864_rows import Render864, rec_frames
    from renderer864 import upsize, da3_gauge_depths
    row = corpus864('development')[0][a.scene_offset]; st = Stream864(row, 1.)
    s = a.target_latent; src_lat = [int(x) for x in a.src_latents.split(',')]; frames = rec_frames(row, [4 * i for i in src_lat]); c2w_src = st.c2w[[4 * i for i in src_lat]].numpy().astype(np.float64)
    target = st.c2w[4 * s - 3: 4 * s + 13].numpy().astype(np.float64); gt = rec_frames(row, range(4 * s - 3, 4 * s + 13))
    r = Render864(vae=None)
    frontier.install('off'); _, _, raw0 = r.render(frames, c2w_src, st.K.numpy(), st.metres_per_unit, target, st.gauge_id, boundary=12, encode=False)
    frontier.install('support', a.close); r.render(frames, c2w_src, st.K.numpy(), st.metres_per_unit, target, st.gauge_id, boundary=12, encode=False); obs_sup = frontier.LAST['observed']
    frontier.install('reproj', tol=a.tol); r.render(frames, c2w_src, st.K.numpy(), st.metres_per_unit, target, st.gauge_id, boundary=12, encode=False); obs_rep = frontier.LAST['observed']
    K540 = frontier.LAST['K']; src_depths = frontier.LAST['src_depths']
    r.da3.to('cuda'); tdepths, _ = da3_gauge_depths(r.da3, upsize(gt, (540, 960)), K540, target, float(st.metres_per_unit)); r.da3.to('cpu')
    obs_true = frontier.observed_from_reprojection(torch.stack(tdepths), K540, target, src_depths, c2w_src, tol=a.tol)
    valid = raw0['valid']; stats = dict(scene=row['stream_id'][:8], target_latent=s, valid_mean=float(valid.mean()))
    for name, obs in (('support', obs_sup), ('reproj', obs_rep), ('true', obs_true)):
        cells = frontier.pooled_cells(obs)
        stats[name] = dict(observed_mean=float(obs.mean()), frontier_frac=float(((valid > .5) & (obs < .5)).float().mean()), cells_dropped=float((cells < .5).float().mean()),
                           conf_frontier=float(raw0['conf'][(valid > .5) & (obs < .5)].mean()) if ((valid > .5) & (obs < .5)).any() else None,
                           conf_observed=float(raw0['conf'][(valid > .5) & (obs >= .5)].mean()) if ((valid > .5) & (obs >= .5)).any() else None)
    stats['reproj_vs_true_agreement'] = float(((obs_rep >= .5) == (obs_true >= .5)).float().mean())
    print(json.dumps(stats, indent=1), flush=True); (a.out / f'smoke_stats_s{s}_src{a.src_latents.replace(",", "-")}.json').write_text(json.dumps(stats, indent=1))
    keep = torch.nn.functional.interpolate((frontier.pooled_cells(obs_rep) >= .5).float()[:, None], (480, 864), mode='nearest')[:, 0]
    picks = [0, 8, 15]; tw, th = 432, 240; names = ['recorded', 'render', 'valid (alpha)', 'observed: support', 'observed: reproj', 'observed: true (DA3 target)', 'tokens kept (reproj)']
    sheet = Image.new('RGB', (tw * len(picks), th * len(names)), 'black'); d = ImageDraw.Draw(sheet)
    for j, k in enumerate(picks):
        cols = [gt[k], to_img(raw0['rgb'][k]), to_img(valid[k]), to_img(obs_sup[k]), to_img(obs_rep[k]), to_img(obs_true[k]), to_img(raw0['rgb'][k] * keep[k // 4][None])]
        for i, im in enumerate(cols):
            sheet.paste(Image.fromarray(im).resize((tw, th)), (j * tw, i * th))
            d.rectangle((j * tw, i * th, j * tw + 190, i * th + 14), fill='black'); d.text((j * tw + 3, i * th + 1), f'{names[i]}  f{4 * s - 3 + k}', fill='yellow')
    sheet.save(a.out / f'smoke_{row["stream_id"][:8]}_s{s}_src{a.src_latents.replace(",", "-")}.jpg', quality=90); print('sheet written', flush=True)


if __name__ == '__main__':
    main()
