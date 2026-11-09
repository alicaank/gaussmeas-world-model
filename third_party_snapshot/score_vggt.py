"""Independent camera scoring with VGGT-1B (F' and score_memorizon.py both use DA3; this scorer shares no network with them).

Per run: VGGT jointly on [anchor + the 40 latent-end frames] (448x256 -> 518x294). Poses re-expressed relative to the
anchor view; ONE scale per run from VGGT's own anchor depth vs the stored anchor depth (median ratio; the anchor depth
defines the gauge, it is not a measurement of the generated frames). Same camera metrics as score_memorizon.camera():
position error vs the planned path, rotation error, executed/commanded step per call. Also scores the recordings of the
scenes (sanity: ratio ~1, small error) and reports per-call agreement with the DA3 measurement logged by the harness.
Usage (framecrafter env, recorded_eval/run_eval_python.sh): score_vggt.py <runs_root> [<runs_root> ...]
Output: <first root>/vggt_summary.json
"""
import json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr
from PIL import Image
sys.path.insert(0, '/home/hj/Alican/VGGT-World')                      # VGGT-1B weights, vanilla forward path
sys.path.insert(0, '/home/hj/Alican/training_runs/base_restart_20261008')
from all_scene_dataset import DATA, RecordedStream
BATCH = Path('/home/hj/Alican/analysis/plan_base_restart_20261008/lyra2_plugins/batch_dev10')
CKPT = '/home/hj/Alican/growworld/third_party/assets/vggt-1b/model.safetensors'


def load_vggt():
    from vggt.models.vggt import VGGT
    from safetensors.torch import load_file
    m = VGGT(enable_track=False, enable_point=False); res = m.load_state_dict(load_file(CKPT), strict=False)   # VGGT-World class (needs CUDA)
    miss = [k for k in res.missing_keys if not k.startswith('fm.')]; assert not miss, f'missing VGGT keys: {miss[:5]}'
    return m.cuda().eval()                                                     # vanilla VGGT-1B path: forward_vggt (the fm module is unused)


@torch.no_grad()
def vggt_poses(model, views, anchor_depth):
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    x = torch.from_numpy(np.stack(views)).float().permute(0, 3, 1, 2).cuda() / 255.
    x = F.interpolate(x, size=(294, 518), mode='bilinear', align_corners=False)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        pred = model.forward_vggt(x[None])
    ext, _ = pose_encoding_to_extri_intri(pred['pose_enc'], x.shape[-2:])
    w2c = torch.eye(4, dtype=torch.float64).repeat(len(views), 1, 1); w2c[:, :3, :] = ext[0].double().cpu()
    rel = (w2c[0] @ torch.linalg.inv(w2c)).numpy()                                   # c2w relative to the anchor view
    d0 = pred['depth'][0, 0, :, :, 0].float()
    ours = F.interpolate(torch.as_tensor(anchor_depth).float()[None, None].cuda(), size=d0.shape, mode='bilinear', align_corners=False)[0, 0]
    ok = (d0 > 0) & (ours > 0) & torch.isfinite(d0) & torch.isfinite(ours); s = float(torch.median(ours[ok] / d0[ok]))
    rel[:, :3, 3] *= s
    return rel, s


def camera(M, P):
    pos = [float(np.linalg.norm(M[j][:3, 3] - P[j][:3, 3])) for j in range(1, 41)]
    rot = [float(np.degrees(Rotation.from_matrix(M[j][:3, :3].T @ P[j][:3, :3]).magnitude())) for j in range(1, 41)]
    ratio = [float(np.linalg.norm(M[4 * c + 4][:3, 3] - M[4 * c][:3, 3]) / max(np.linalg.norm(P[4 * c + 4][:3, 3] - P[4 * c][:3, 3]), 1e-6)) for c in range(10)]
    return dict(pos_err_per_call=[float(np.mean(pos[4 * c:4 * c + 4])) for c in range(10)],
                rot_err_per_call=[float(np.mean(rot[4 * c:4 * c + 4])) for c in range(10)], step_ratio_per_call=ratio)


def da3_ratio(log):
    P = np.array(log['planned']); M = [None if m is None else np.array(m) for m in log['measured']]; M[0] = np.eye(4)
    return [float(np.linalg.norm(M[4 * c + 4][:3, 3] - M[4 * c][:3, 3]) / max(np.linalg.norm(P[4 * c + 4][:3, 3] - P[4 * c][:3, 3]), 1e-6))
            if M[4 * c] is not None and M[4 * c + 4] is not None else np.nan for c in range(10)]


def main():
    roots = [Path(p) for p in sys.argv[1:]]; model = load_vggt()
    idx = {s['name']: s for s in json.load(open(BATCH / 'index.json'))['scenes']}
    streams = {json.loads(l)['stream_id']: json.loads(l) for l in (DATA / 'corpus/streams.jsonl').open()}
    rows, seen = [], set()
    for root in roots:
        for npz in sorted(root.glob('*/*_s*.npz')):
            arm = npz.parent.name; scene, seed = npz.stem.split('_s')
            if (arm, scene, seed) in seen:
                continue
            seen.add((arm, scene, seed)); log = json.load(open(npz.with_suffix('.json')))
            P = np.array(log['planned']); g = np.load(npz)['frames']
            anchor = np.asarray(Image.open(BATCH / 'images' / f'{scene}.png').convert('RGB'))
            adepth = np.load(DATA / 'components' / idx[scene]['stream_id'] / 'anchor.npz')['depth']
            M, s = vggt_poses(model, [anchor] + [g[4 * j - 1] for j in range(1, 41)], adepth)
            cam = camera(M, P)
            rows.append(dict(arm=arm, scene=scene, seed=int(seed), scale=s, da3_ratio=da3_ratio(log), **cam))
            print(arm, scene, seed, 'pos_err %.3f ratio %.2f' % (np.mean(cam['pos_err_per_call']), np.median(cam['step_ratio_per_call'])), flush=True)
    for scene in sorted({r['scene'] for r in rows}):                                  # recordings: sanity check of the scorer
        st = RecordedStream(streams[idx[scene]['stream_id']]); fr = [st.frame(4 * j) for j in range(0, 41)]
        P = np.array(json.load(open(next(r for r in sorted(roots[0].glob(f'*/{scene}_s*.json'))))) ['planned'])
        adepth = np.load(DATA / 'components' / idx[scene]['stream_id'] / 'anchor.npz')['depth']
        M, s = vggt_poses(model, fr, adepth); cam = camera(M, P)
        rows.append(dict(arm='recording', scene=scene, seed=0, scale=s, da3_ratio=[np.nan] * 10, **cam))
    key = lambda r: (r['scene'], r['seed'])
    base = {key(r): r for r in rows if r['arm'] == 'baseline'}
    met = {'pos_err_mean': lambda r: np.mean(r['pos_err_per_call']), 'pos_err_call10': lambda r: r['pos_err_per_call'][-1],
           'rot_err_mean': lambda r: np.mean(r['rot_err_per_call']), 'step_ratio_median': lambda r: np.median(r['step_ratio_per_call'])}
    arms = {}
    for arm in sorted({r['arm'] for r in rows}):
        rs = [r for r in rows if r['arm'] == arm]; e = {m: float(np.mean([f(r) for r in rs])) for m, f in met.items()}; e['n'] = len(rs)
        if arm not in ('baseline', 'recording'):
            for m, f in met.items():
                d = [f(r) - f(base[key(r)]) for r in rs if key(r) in base]
                if d:
                    e['delta_' + m] = float(np.mean(d)); e['delta_' + m + '_same_sign'] = f'{sum(x > 0 for x in d)}+/{sum(x < 0 for x in d)}-'
        arms[arm] = e
    pv = [(a, b) for r in rows for a, b in zip(r['step_ratio_per_call'], r['da3_ratio']) if np.isfinite(b)]
    agree = dict(n=len(pv), spearman_step_ratio_vggt_vs_da3=float(spearmanr([a for a, _ in pv], [b for _, b in pv]).correlation),
                 median_ratio_vggt=float(np.median([a for a, _ in pv])), median_ratio_da3=float(np.median([b for _, b in pv])))
    out = dict(arms=arms, agreement_with_da3=agree, rows=rows)
    json.dump(out, open(roots[0] / 'vggt_summary.json', 'w'), indent=1, default=float)
    print(json.dumps(agree)); [print(a, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items()}) for a, e in arms.items()]


if __name__ == '__main__':
    main()
