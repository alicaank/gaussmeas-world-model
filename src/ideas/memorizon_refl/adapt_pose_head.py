"""Stage 0 of camera-ReFL on Memorizon: adapt the latent pose head to 864x480 latents and gate it as a reward.

Init: cfg_res_20261015/pose_head/pose_head.pt (trained on dataset_v3 448x256 latents; inputs anchor + pair + focal). Fine-tune on a 50/50 mix of
Memorizon shards (864x480, same Wan2.2 VAE + normalization) and dataset_v3 (448x256), targets in the anchor-depth gauge.
Data: camera_quality_20261015/manifest.json applied to both pools (shard windows, dataset_v3 streams).
Validation (written to head_gate.json):
  heldout864 20 development-split scenes (no training pool contains them; not the dev monitor scenes), encoded at 864:
             gap 1 and 4 relative translation error, direction cosine, rotation error
  generated  the released generator's OWN latents (gate_runs/, 4 dev scenes that are not the ReFL eval scenes x 2 seeds,
             saved by gv_memorizon2 --save_latents; no decode/re-encode) vs the command-free DA3 poses of the same run,
             per call: step magnitude (ratio, rank), translation direction, rotation.
Gate (pass -> ReFL may start): heldout gap-4 rel err <= 0.25, dir cos >= 0.9, rot err <= 1.5 deg; generated |ratio head -
DA3| <= 0.15, Spearman >= 0.5, dir cos >= 0.7, rot err <= max(1.5 deg, half the DA3 rotation), >= 6 runs.
"""
import json, random, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
from scipy.stats import spearmanr
HERE = Path(__file__).parent; PLUG = HERE.parent
sys.path.insert(0, '/home/hj/Alican/analysis/plan_base_restart_20261008/cfg_res_20261015')
sys.path.insert(0, '/home/hj/Alican/baselines/memorizon'); sys.path.insert(0, str(PLUG / 'finetune'))
from train_pose_head import Head, load_stream, rel, geodesic, DATA
from build_shards import stream_frames, crop_resize
SHARDS = PLUG / 'finetune' / 'shards300'; BASE_HEAD = Path('/home/hj/Alican/analysis/plan_base_restart_20261008/cfg_res_20261015/pose_head/pose_head.pt')
OUT = HERE / 'head864'; OUT.mkdir(exist_ok=True)
W, H = 864, 480


def load_shard(ep):
    d = torch.load(SHARDS / 'latents' / f'{ep}.pt', map_location='cpu')
    a = np.load(SHARDS / 'aux' / f'{ep}.npz'); K = a['K864']
    kn = np.array([K[0] / 864, K[1] / 480], np.float32)
    return d['latents'].float(), a['c2w_frame'].astype(np.float64), 1.0, kn     # c2w_frame: anchor-relative, anchor-depth gauge


def pair_batch(src, n, rng, cache, loader, max_gap=8):
    A, B, R, t, A0, KN = [], [], [], [], [], []
    for _ in range(n):
        k = rng.choice(src)
        if k not in cache:
            if len(cache) > 300:
                cache.pop(next(iter(cache)))
            cache[k] = loader(k)
        lat, c2w, u, kn = cache[k]
        Tm = min(lat.shape[1] - 1, (len(c2w) - 1) // 4); i = rng.randrange(0, Tm); j = min(Tm, i + rng.randint(1, max_gap))
        A.append(lat[:, i]); B.append(lat[:, j]); A0.append(lat[:, 0]); KN.append(kn); M = rel(c2w, i, j, u); R.append(M[:3, :3]); t.append(M[:3, 3])
    f = lambda x: torch.tensor(np.stack(x), dtype=torch.float32).cuda()
    return torch.stack(A0).cuda(), torch.stack(A).cuda(), torch.stack(B).cuda(), f(KN), f(R), f(t)


@torch.no_grad()
def predict(head, a0, a, b, kn):
    with torch.autocast('cuda', torch.bfloat16):
        R, t = head(a0[None].cuda(), a[None].cuda(), b[None].cuda(), torch.as_tensor(kn)[None].cuda())
    return R.float()[0].cpu().numpy(), t.float()[0].cpu().numpy()


@torch.no_grad()
def encode(pipe, frames_uint8):
    """[N,H,W,3] uint8 (any size) -> normalized latents [48, 1+(N-1)/4, 30, 54], causal VAE, Memorizon encode path."""
    ims = [np.asarray(Image.fromarray(f).resize((W, H), Image.BICUBIC)) if f.shape[:2] != (H, W) else f for f in frames_uint8]
    x = torch.from_numpy(np.stack(ims).astype(np.float32) / 127.5 - 1).permute(3, 0, 1, 2)[None].cuda()
    pipe.vae.clear_cache()
    with torch.autocast('cuda', torch.bfloat16):
        z = pipe.vae.encode(x).latent_dist.mode().float()
    pipe.vae.clear_cache()
    return ((z - pipe.lat_mean) / pipe.lat_std)[0].cpu()


MANIFEST = Path('/home/hj/Alican/analysis/plan_base_restart_20261008/camera_quality_20261015/manifest.json')
DEV10 = Path('/home/hj/Alican/analysis/plan_base_restart_20261008/lyra2_plugins/batch_dev10')
GATE_RUNS = HERE / 'gate_runs' / 'baseline_lat'          # gv_memorizon2 --save_latents, 4 non-eval dev scenes x 2 seeds


def rot_deg(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def heldout_streams(rows, man, n=20):
    """Development-split streams (never in any training pool), not the 10 dev monitor scenes, camera-quality clean."""
    dev10 = {s['stream_id'] for s in json.load(open(DEV10 / 'index.json'))['scenes']}
    c = [r for r in rows if r['split'] != 'train' and r['latent_count'] >= 41 and r['stream_id'] not in dev10 and r['stream_id'] not in man['streams']]
    c = sorted(c, key=lambda r: r['stream_id']); random.Random(2).shuffle(c)
    out, seen = [], set()
    for r in c:                                            # one stream per scene
        if r['scene_id'] not in seen:
            out.append(r); seen.add(r['scene_id'])
        if len(out) == n:
            break
    return out


def main():
    man = json.load(open(MANIFEST))
    eps = [e for e in json.load(open(SHARDS / 'index.json'))['episode_ids'] if e not in man['shards']]   # window-level camera quality
    rows = [json.loads(l) for l in open(DATA / 'corpus/streams.jsonl')]
    v3 = [r['stream_id'] for r in rows if r['split'] == 'train' and r['latent_count'] >= 12 and r['stream_id'] not in man['streams'] and (DATA / 'components' / r['stream_id'] / 'vae.npz').exists()]
    print(f'shards {len(eps)} (excluded {len(man["shards"])}), v3 {len(v3)}', flush=True)
    head = Head().cuda(); head.load_state_dict(torch.load(BASE_HEAD, map_location='cuda'))
    steps = 4000; opt = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-4, total_steps=steps, pct_start=0.05)
    rng = random.Random(1); c864, c448 = {}, {}; log = open(OUT / 'train.log', 'w'); t0 = time.time()
    for step in range(1, steps + 1):
        if step % 2:
            a0, a, b, kn, R, t = pair_batch(eps, 16, rng, c864, load_shard)
        else:
            a0, a, b, kn, R, t = pair_batch(v3, 16, rng, c448, load_stream)
        with torch.autocast('cuda', torch.bfloat16):
            Rp, tp = head(a0, a, b, kn)
        lr_ = geodesic(Rp.float(), R).mean(); lt = (tp.float() - t).norm(dim=-1).mean(); loss = lr_ + lt
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step(); sched.step()
        if step % 200 == 0:
            rec = dict(step=step, res='864' if step % 2 else '448', rot_deg=float(lr_) * 57.3, trans=float(lt), sec=round(time.time() - t0))
            log.write(json.dumps(rec) + '\n'); log.flush(); print(rec, flush=True)
    torch.save(head.state_dict(), OUT / 'pose_head864.pt'); head.eval()
    del c864, c448

    from memorizon import MemorizonPipeline
    from memorizon.model import MemorizonModel
    from build_shards import intrinsics_after
    pipe = MemorizonPipeline(MemorizonModel.from_pretrained('/home/hj/Alican/baselines/models/memorizon_local'))
    pipe.model.to('cpu'); torch.cuda.empty_cache()                       # VAE only

    # --- validation 1: held-out SCENES (development split, never trained on by the base head or here), encoded at 864
    real = {1: [], 4: []}
    for r in heldout_streams(rows, man):
        paths, c2w_f, tdk = stream_frames(r); u = json.loads((DATA / 'components' / r['stream_id'] / 'anchor.json').read_text())['registered_units_per_anchor_depth']
        c2w = np.linalg.inv(c2w_f[0]) @ c2w_f[:161]; c2w[:, :3, 3] /= u
        ims = [crop_resize(Image.open(p).convert('RGB')) for p in paths[:161]]; K = intrinsics_after(tdk, ims[0][1], ims[0][2])
        kn = np.array([K[0] / 864, K[1] / 480], np.float32); lat = encode(pipe, [np.asarray(im[0]) for im in ims])
        for g in real:
            for i in range(0, 40 - g, 3):
                M = rel(c2w, i, i + g, 1.0); Rp, tp = predict(head, lat[:, 0], lat[:, i], lat[:, i + g], kn); tn = np.linalg.norm(M[:3, 3])
                real[g].append(dict(rel=float(np.linalg.norm(tp - M[:3, 3]) / max(tn, 1e-3)), ratio=float(np.linalg.norm(tp) / max(tn, 1e-3)),
                                    cos=float(tp @ M[:3, 3] / max(np.linalg.norm(tp) * tn, 1e-9)), rot=rot_deg(Rp, M[:3, :3]), rot_mag=rot_deg(np.eye(3), M[:3, :3])))
    summ = lambda v: dict(n=len(v), rel_t_err_median=float(np.median([x['rel'] for x in v])), ratio_median=float(np.median([x['ratio'] for x in v])),
                          dir_cos_median=float(np.median([x['cos'] for x in v])), rot_err_median=float(np.median([x['rot'] for x in v])),
                          rot_mag_median=float(np.median([x['rot_mag'] for x in v])))
    res = dict(heldout864={g: summ(v) for g, v in real.items()})

    # --- validation 2: the generator's OWN latents (no decode / re-encode), vs the command-free DA3 poses of the same run
    gen = []
    for npz in sorted(GATE_RUNS.glob('*_s*.npz')):
        scene = npz.stem.split('_s')[0]; lg = json.load(open(npz.with_suffix('.json')))
        z = torch.from_numpy(np.load(npz.with_suffix('.latents.npy')).astype(np.float32))[0]      # [48,41,h,w]
        P = np.array(lg['planned']); Mm = [None if m is None else np.array(m) for m in lg['measured']]; Mm[0] = np.eye(4)
        Kb = np.load(DEV10 / 'trajectories' / f'{scene}.npz')['intrinsics'][0]; kn = np.array([Kb[0, 0] / 448, Kb[1, 1] / 256], np.float32)
        for c in range(10):
            a_, b_ = 4 * c, 4 * c + 4
            if Mm[a_] is None or Mm[b_] is None:
                continue
            Rp, tp = predict(head, z[:, 0], z[:, a_], z[:, b_], kn); D = np.linalg.inv(Mm[a_]) @ Mm[b_]; cmd = np.linalg.inv(P[a_]) @ P[b_]
            td = D[:3, 3]; nd = np.linalg.norm(td)
            gen.append(dict(run=npz.stem, call=c + 1, head=float(np.linalg.norm(tp)), da3=float(nd), cmd=float(np.linalg.norm(cmd[:3, 3])),
                            cos=float(tp @ td / max(np.linalg.norm(tp) * nd, 1e-9)), rot=rot_deg(Rp, D[:3, :3]), rot_mag=rot_deg(np.eye(3), D[:3, :3])))
    g = [r for r in gen if r['cmd'] > 1e-3 and r['da3'] > 1e-3]
    rh = float(np.median([r['head'] / r['cmd'] for r in g])); rd = float(np.median([r['da3'] / r['cmd'] for r in g]))
    rho = float(spearmanr([r['head'] for r in g], [r['da3'] for r in g]).correlation)
    res['generated'] = dict(n=len(g), runs=len({r['run'] for r in g}), ratio_head_median=rh, ratio_da3_median=rd, spearman_head_vs_da3=rho,
                            dir_cos_median=float(np.median([r['cos'] for r in g])), rot_err_median=float(np.median([r['rot'] for r in g])),
                            rot_mag_median=float(np.median([r['rot_mag'] for r in g])), rows=gen)
    G = res['generated']; Hh = res['heldout864'][4]
    gate = dict(heldout_gap4_rel=Hh['rel_t_err_median'] <= 0.25, heldout_dir=Hh['dir_cos_median'] >= 0.9, heldout_rot=Hh['rot_err_median'] <= 1.5,
                gen_ratio_agree=abs(rh - rd) <= 0.15, gen_rank=rho >= 0.5, gen_dir=G['dir_cos_median'] >= 0.7,
                gen_rot=G['rot_err_median'] <= max(1.5, 0.5 * G['rot_mag_median']), gen_n=G['runs'] >= 6)
    res['gate'] = dict(**gate, passed=bool(all(gate.values())))
    json.dump(res, open(OUT / 'head_gate.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'generated'}, indent=1))
    print(json.dumps({k: v for k, v in res['generated'].items() if k != 'rows'}, indent=1))


if __name__ == '__main__':
    main()
