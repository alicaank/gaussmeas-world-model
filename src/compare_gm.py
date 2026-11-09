"""GM vs GM-B on the closed_loop_864 evaluation (eval.json per arm): late appearance, sharpness, DA3 camera, independent VGGT
camera with the rotation-excursion count (> 30 deg) that characterises the earlier model's failures. Usage: compare_gm.py OUT.json A=eval.json B=eval.json ..."""
import json
import statistics as st
import sys
from pathlib import Path


def per_scene(ev):
    out = {}
    for sc in ev['scenes']:
        pc = sc['per_call']; vg = sc.get('vggt') or []
        late = [p for p in pc if p['call'] >= 11]
        out[sc['stream']] = dict(lpips_1_3=st.mean(p['lpips'] for p in pc if p['call'] <= 3), lpips_late=st.mean(p['lpips'] for p in late), dino_late=st.mean(p['dino'] for p in late),
                                 sharp_late=st.mean(p['sharp'] for p in late), sharp_ref_late=st.mean(p['sharp_ref'] for p in late),
                                 da3_pos=st.mean(p['pos_err'] for p in pc), exec_median=st.median(p['exec_ratio'] for p in pc),
                                 vggt_pos=st.mean(v['pos_err'] for v in vg) if vg else None, vggt_pos_last5=st.mean(v['pos_err'] for v in vg[-5:]) if vg else None,
                                 vggt_rot=st.mean(v['rot_err_deg'] for v in vg) if vg else None, rot_gt30=sum(v['rot_err_deg'] > 30 for v in vg) if vg else None,
                                 restarts=sc.get('restarts'))
    return out


def main():
    out = Path(sys.argv[1]); arms = {k: json.loads(Path(v).read_text()) for k, v in (x.split('=', 1) for x in sys.argv[2:])}
    per = {k: per_scene(v) for k, v in arms.items()}; scenes = sorted(set.intersection(*(set(p) for p in per.values())))
    keys = ['lpips_1_3', 'lpips_late', 'dino_late', 'sharp_late', 'da3_pos', 'exec_median', 'vggt_pos', 'vggt_pos_last5', 'vggt_rot', 'rot_gt30']
    means = {k: {m: (st.mean(per[k][s][m] for s in scenes) if all(per[k][s][m] is not None for s in scenes) else None) for m in keys} for k in per}
    names = list(per); wins = {}
    if len(names) == 2:
        a, b = names; lower = {'lpips_1_3', 'lpips_late', 'da3_pos', 'vggt_pos', 'vggt_pos_last5', 'vggt_rot', 'rot_gt30'}
        wins = {m: sum((per[b][s][m] < per[a][s][m]) if m in lower else (per[b][s][m] > per[a][s][m]) for s in scenes) for m in keys if m != 'exec_median' and all(per[x][s][m] is not None for x in names for s in scenes)}
    res = dict(scenes=scenes, per_scene=per, means=means, wins_second_over_first=wins, order=names, sharp_ref_late=st.mean(per[names[0]][s]['sharp_ref_late'] for s in scenes))
    out.write_text(json.dumps(res, indent=2))
    print(f"{'arm':10s} " + ' '.join(f'{m:>12s}' for m in keys))
    for k in names:
        print(f"{k:10s} " + ' '.join(f"{means[k][m]:12.3f}" if means[k][m] is not None else f"{'-':>12s}" for m in keys))
    print('recorded late sharpness', round(res['sharp_ref_late']), '| wins (second over first, of', len(scenes), 'scenes):', wins)


if __name__ == '__main__':
    main()
