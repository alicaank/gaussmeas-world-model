"""Pool for the measured-write training test (see onpolicy_rows.py). Per TRAIN stream (one per scene, L >= 17 latents): the
current memory model (--receipt) rolls out 2 calls open loop along the recorded commands (seeds seed, seed+1), exactly as
closed_loop_864 does; DA3 measures all own latent ends jointly (tail 8). For k = 1, 2 (s = 1 + 4k) one item is written with
the own latents 0..s-1, commanded and measured latent-end poses, the bank chosen under each labelling, and the ReSplat render
of the recorded next command from the own frames under each labelling (OwnRender864, as at inference).
Usage: gen_onpolicy.py --receipt R [--n_streams 300] [--seed 20261024]"""
import argparse, json, os, random, time
from pathlib import Path
import numpy as np, torch
from data import corpus864, Stream864, select_bank
from model_io import fresh_model, load_trainables, load_vae, empty_text, resolve_init
from closed_loop import LoopRoller
from render864_rows import OwnRender864
from realized_pose import load_da3
from onpolicy_rows import POOL
import packed_model


def pack_bf16(t):
    # all_scene_dataset.unpack_bf16 expects unsigned storage of the BF16 bits.
    return t.detach().to(device='cpu', dtype=torch.bfloat16).view(torch.int16).numpy().view(np.uint16)


def rgb_rate(ends):
    """latent-end poses (s,4,4) -> RGB-rate list where index 4j holds ends[j] (only latent ends are read)."""
    out = [np.eye(4)] * (4 * (len(ends) - 1) + 1)
    for j, e in enumerate(ends):
        out[4 * j] = np.asarray(e, np.float64)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--receipt', required=True); ap.add_argument('--n_streams', type=int, default=300)
    ap.add_argument('--seed', type=int, default=20261024); ap.add_argument('--steps', type=int, default=30); a = ap.parse_args()
    POOL.mkdir(exist_ok=True)
    rec = json.load(open(a.receipt)); lin = rec['meta'].get('lineage', {}); init = resolve_init(lin); packed_model.LAYOUT = lin.get('layout', 'memorizon')
    lora_r = int(json.load(open(rec['meta']['config'])).get('lora_r', 0)); scale_mult = float(lin.get('scale_mult', 2.))
    packed_model.PLAIN = (Path(rec['meta']['config']).parent / 'plain.flag').exists()
    model, _, fp = fresh_model(prope=True, lora_r=lora_r, init=init); load_trainables(model, a.receipt); model.eval()
    vae = load_vae(); da3 = load_da3('cpu'); empty = empty_text(); render = OwnRender864(vae, da3)
    rows, _ = corpus864('train'); rows = [r for r in rows if r.get('mpu_valid', True) and r['latent_count'] >= 17]
    rng = random.Random(a.seed); by = {}
    for r in rows:
        by.setdefault(r['scene_id'], []).append(r)
    scenes = sorted(by); rng.shuffle(scenes); pick = [rng.choice(by[sc]) for sc in scenes][:a.n_streams]
    meta = dict(receipt=a.receipt, ckpt_sha256=rec['sha256'], seed=a.seed, steps=a.steps, n_streams=len(pick)); json.dump(meta, open(POOL / 'meta.json', 'w'))
    done = 0; t_all = time.time()
    for r in pick:
        sid = r['stream_id']
        if all((POOL / f'{sid}_s{1 + 4 * k}.npz').exists() for k in (1, 2)):
            continue
        t0 = time.time(); st = Stream864(r, scale_mult); P = st.c2w.numpy().astype(np.float64); K = st.K.numpy(); mpu = st.metres_per_unit
        da3.to('cuda')
        roller = LoopRoller(model, vae, a.steps, 3. if init == 'memorizon50' else 5., 1., empty, cam_scale=st.cam_scale, mpu=mpu, da3=da3, write_measured=False,
                            replan=False, early=None, render=render, gauge_id=st.gauge_id, schedule_mode='memorizon' if init == 'memorizon50' else 'wan',
                            max_bank=4).start(st.latents[:, :, :1], st.K)
        for c in range(2):
            roller.call_plan(P[1 + 16 * c: 17 + 16 * c], st.text, a.seed + c)
        ps = roller.measure(tail=8)                                                         # anchor gauge, latent ends 1..8
        if not all(p is not None and np.isfinite(np.asarray(p)).all() for p in ps[-8:]):
            print(json.dumps(dict(stream=sid[:8], skipped='measurement invalid')), flush=True); continue
        meas_all = [np.eye(4)] + [np.asarray(p, np.float64) for p in ps[-8:]]
        gen = torch.cat(roller.latents, 2); fbl = roller.frames_by_latent; H, W = st.latents.shape[-2:]
        for k in (1, 2):
            s = 1 + 4 * k; command = P[4 * s - 3: 4 * s + 13]
            ends = dict(cmd=[P[4 * j] for j in range(s)], meas=meas_all[:s]); item = dict(stream_id=sid, s=s, gen_latents_bf16=pack_bf16(gen[:, :, :s]))
            for v, e in ends.items():
                obs = torch.from_numpy(np.stack(rgb_rate(e))).float()
                bank = select_bank(obs, torch.from_numpy(command).float(), st.K, s, size=(16 * H, 16 * W), max_refs=4) if s > 1 else []
                rz, mt, raw = render.render_own(fbl, rgb_rate(e), bank, s, K, mpu, command, st.gauge_id)
                item.update({f'{v}_end': np.stack(e), f'bank_{v}': np.asarray(bank, np.int64), f'rz_{v}_bf16': pack_bf16(rz), f'meta_{v}': mt.numpy(),
                             f'coverage_{v}': float(raw['valid'].mean())})
            item['label_gap'] = np.asarray([np.linalg.norm(ends['meas'][j][:3, 3] - ends['cmd'][j][:3, 3]) for j in range(s)])
            np.savez_compressed(POOL / f'{sid}_s{s}.npz', **item)
        done += 1; da3.to('cpu'); torch.cuda.empty_cache()
        print(json.dumps(dict(stream=sid[:8], gap_end=round(float(np.linalg.norm(meas_all[8][:3, 3] - P[32][:3, 3])), 3), cov=[round(item['coverage_cmd'], 3), round(item['coverage_meas'], 3)],
                              sec=round(time.time() - t0, 1), done=done)), flush=True)
    print('DONE', done, 'streams', round((time.time() - t_all) / 3600, 2), 'h')


if __name__ == '__main__':
    main(); os._exit(0)
