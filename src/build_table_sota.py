"""Comparison table with the GM-line rows added: reuses comparison_table_20261105 (same metrics(), same baseline score files for Lyra-2,
Spatia, Matrix-Game and the "Ours" = run_op/meas rows) and appends GM-B, GM-F (+frustum mask) and GM + frustum mask (control) from
sota_eval/scores/<model>_<scene><tag>.json. Output table_sota<tag>.md / .json. Usage: build_table_sota.py [--tag _baseline]"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = Path('/home/hj/Alican/analysis/v2_rerun_20261103')
spec = importlib.util.spec_from_file_location('ct', V2 / 'comparison_table_20261105.py'); ct = importlib.util.module_from_spec(spec); spec.loader.exec_module(ct)
NEW = [('GM-B (corruption)', 'gm_b'), ('GM-F (corruption + frontier mask) + frustum mask', 'gm_f_inside'), ('GM (run_op/meas) + frustum mask', 'gm_inside')]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', default=''); a = ap.parse_args()
    orig = ct.score_file

    def score_file(model, scene):
        if model in dict((m, n) for n, m in NEW):
            f = HERE / 'scores' / f'{model}_{scene}{a.tag}.json'
            if not f.exists(): return None, None
            d = json.load(open(f))
            if d['evaluation_protocol'] != ct.PROTO or int(d['horizon']) != 480: raise ValueError(str(f))
            return f, d
        return orig(model, scene)
    ct.score_file = score_file
    rows = list(ct.ROWS)
    for name, model in NEW:
        rows.append((name, model, 'baseline'))
        if not a.tag: rows.append((name + ' + correction', model, 'correction'))
    tb, mb = ct.table(ct.B, rows); ta, ma = ct.table(ct.A, rows)
    text = ['# GM line vs Ours (run_op/meas) / Lyra-2 / Spatia / Matrix-Game (DL3DV, 480 frames, seed 7, one scorer)', '',
            f'tag={a.tag or "full"}; GM-B = runs/gm_b_s1, GM-F = runs/gm_f_s1, frustum mask = frontier.py tol 1e9 (9 Nov decision).', '',
            ct.md(ta, 'A. Scenes shared by all four models', ct.A), '', ct.md(tb, 'B. Scenes shared by ours, Lyra-2 and Spatia', ct.B)]
    if ma or mb: text += ['', 'Missing (row dropped if any scene missing): ' + '; '.join(sorted(set(ma + mb)))]
    (HERE / f'table_sota{a.tag}.md').write_text('\n'.join(text) + '\n')
    (HERE / f'table_sota{a.tag}.json').write_text(json.dumps(dict(A=ta, B=tb), indent=1, default=str))
    print('\n'.join(text))


if __name__ == '__main__':
    main()
