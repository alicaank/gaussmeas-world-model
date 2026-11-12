"""Comparison table (5 Nov): ours best arm (gaussian_meas correction) vs Lyra-2, Spatia, Matrix-Game, each without / with the
paper's correction controller. ONE scorer for every row: vipe_native_map_mask_diagnostic_v3 (score_reliability --style long,
horizon 480, seed 7). Sources: ours gaussian_meas_dl3dv7_20261103/scores; Lyra / Matrix-Game v2_rerun scores; Spatia and Lyra
0238de14 / 61b27b61 scores_v3_extra_20261105 (same scorer on the same videos the v2_rerun manifest lists).
Metrics: cumulative mean over frames 1..480 (camera position error in anchor-depth units, rotation deg, LPIPS, DINO), camera
endpoint at 480, and LPIPS / DINO over the final third = mean of the image samples with frame id in 321..480 (fixed 5 Nov after
review: the earlier 3*H480 - 2*H320 assumed equal sample counts per third, which the 41 image samples do not have).
Cohorts: A = the 3 scenes every model (incl. Matrix-Game 3.5 Base) has; B = all 5 scenes (every model except Matrix-Game 3.5 Base).
Output comparison_table_20261105.md + .json."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; AN = HERE.parent
A = ['02267acf', '0238de14', '334ff324']; B = A + ['61b27b61', 'a126efe8']
PROTO = 'vipe_native_map_mask_diagnostic_v3'


def score_file(model, scene):
    c = {'ours': [AN / 'gaussian_meas_dl3dv7_20261103/scores' / f'{scene}.json'],
         'lyra': [HERE / 'scores' / f'lyra_dl3dv480_{scene}_s7.json', HERE / 'scores_v3_extra_20261105' / f'lyra_{scene}_s7.json'],
         'spatia': [HERE / 'scores_v3_extra_20261105' / f'spatia_{scene}_s7.json'],
         'matrixgame': [AN / 'mg35_distilled_20261105/scores' / f'matrixgame_distilled_{scene}_s7.json'],   # 6 Nov: Matrix-Game-3.5-Distilled (user)
         'matrixgame_base': [HERE / 'scores' / f'matrixgame_dl3dv480_{scene}_s7.json']}[model]              # 6 Nov: base back alongside (user), 3 scenes
    for f in c:
        if f.exists():
            d = json.load(open(f))
            if d['evaluation_protocol'] != PROTO or int(d['horizon']) != 480:
                raise ValueError(f'{f}: protocol {d["evaluation_protocol"]} horizon {d["horizon"]}')
            return f, d
    return None, None


def window(arm, key, lo, hi):
    ids = np.asarray(arm['image_frame_ids']); v = np.asarray(arm[key], float); m = (ids >= lo) & (ids <= hi)
    assert m.sum() >= 5, (key, lo, hi, ids)
    return float(v[m].mean())


def metrics(arm):
    f = lambda x: float(x)
    return dict(pos=f(arm['camera_mean']), end=f(arm['camera_endpoint']), rot=f(arm['rotation_mean']), lpips=f(arm['lpips']), dino=f(arm['dino']),
                lpips_last=window(arm, 'lpips_by_frame', 321, 480), dino_last=window(arm, 'dino_by_frame', 321, 480))


ROWS = [('Ours', 'ours', 'baseline'), ('Ours + correction', 'ours', 'correction'), ('Lyra-2', 'lyra', 'baseline'), ('Lyra-2 + correction', 'lyra', 'correction'),
        ('Spatia', 'spatia', 'baseline'), ('Spatia + correction', 'spatia', 'correction'), ('Matrix-Game 3.5 Distilled', 'matrixgame', 'baseline'),
        ('Matrix-Game 3.5 Distilled + correction', 'matrixgame', 'correction'), ('Matrix-Game 3.5 Base', 'matrixgame_base', 'baseline'),
        ('Matrix-Game 3.5 Base + correction', 'matrixgame_base', 'correction')]
COLS = [('pos', 'Position err ↓', 4), ('end', 'Endpoint ↓', 3), ('rot', 'Rotation ° ↓', 2), ('lpips', 'LPIPS ↓', 3), ('dino', 'DINO ↑', 3),
        ('lpips_last', 'LPIPS last⅓ ↓', 3), ('dino_last', 'DINO last⅓ ↑', 3)]


def table(scenes, rows):
    out = []; missing = []
    for name, model, arm in rows:
        per = {}
        for s in scenes:
            f, d = score_file(model, s)
            if d is None or arm not in d['arms']:
                missing.append(f'{name} {s}'); continue
            per[s] = metrics(d['arms'][arm])
        if len(per) == len(scenes):
            out.append(dict(name=name, mean={k: float(np.mean([per[s][k] for s in scenes])) for k, _, _ in COLS}, per_scene=per))
    return out, missing


def md(rows, title, scenes):
    best = {k: (min if '↓' in lab else max)(r['mean'][k] for r in rows) for k, lab, _ in COLS}
    s = [f'### {title}', f'Scenes: {", ".join(scenes)} (seed 7, 480 frames, one scorer: `{PROTO}`)', '',
         '| Method | ' + ' | '.join(lab for _, lab, _ in COLS) + ' |', '|---|' + '---|' * len(COLS)]
    for r in rows:
        cells = []
        for k, _, nd in COLS:
            v = f'{r["mean"][k]:.{nd}f}'; cells.append(f'**{v}**' if abs(r['mean'][k] - best[k]) < 10 ** -(nd + 1) else v)
        s.append(f'| {r["name"]} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(s)


def main():
    ta, ma = table(A, ROWS); tb, mb = table(B, ROWS)                     # 6 Nov: distilled Matrix-Game runs on all 5 scenes
    text = ['# Ours vs Lyra-2 / Spatia / Matrix-Game (DL3DV, 480 frames)', '',
            'Position error and endpoint in anchor-depth units; rotation in degrees; columns are means over frames 1–480 except the last-third '
            'columns (frames 321–480). "+ correction" = the paper\'s closed-loop controller wrapped around the released model. Bold = best.', '',
            md(ta, 'A. Scenes shared by all four models', A), '', md(tb, 'B. Scenes shared by ours, Lyra-2 and Spatia', B)]
    if ma or mb:
        text += ['', 'Missing (row dropped if any scene missing): ' + '; '.join(sorted(set(ma + mb)))]
    (HERE / 'comparison_table_20261105.md').write_text('\n'.join(text) + '\n')
    json.dump(dict(A=ta, B=tb, missing=ma + mb), open(HERE / 'comparison_table_20261105.json', 'w'), indent=1)
    print('\n'.join(text))


if __name__ == '__main__':
    main()
