"""Round-2 on-policy pool with the full loop active (PILOT_PLAN_CVPR_20261104.md, step 3): the current model (--receipt, run_op/meas)
rolls out --calls calls per TRAIN stream with the 2Xplat global memory, measured writes and the offset rejoin (F'); the 2Xplat pass
is the observer. For every k in 1..calls (s = 1 + 4k) one item: own latents 0..s-1, measured latent-end poses (M), issued (U) and
planned (P) ends, the visibility bank for the RECORDED next command P[4s-3:4s+13] (= the corrected command from the measured
state), the 2Xplat render of that command (rz, meta) from the scene registered on the measured history, token depths at the
recorded latent ends s..s+7 (dtok_tgt) and at the measured poses of latents 0..s-1 (dtok_src), coverage.
Output: data864/onpolicy2/<sid>_s<s>.npz. Usage: gen_onpolicy2.py --receipt R [--n_streams 300] [--calls 4] [--seed 20261105]"""
import argparse, json, os, random, time
from pathlib import Path
import numpy as np, torch
from data import corpus864, Stream864
from model_io import fresh_model, load_trainables, load_vae, empty_text, resolve_init
from closed_loop_2x import LoopRoller2X, TwoXRender864
from twoxplat_memory import make_scene, seed_gauge
from rows_g import POOL2, bank_by_visibility, dtok_from_depth
import packed_model


def pack_bf16(t):
    return t.detach().to(device='cpu', dtype=torch.bfloat16).view(torch.int16).numpy().view(np.uint16)


def rgb_rate(ends):
    out = [np.eye(4)] * (4 * (len(ends) - 1) + 1)
    for j, e in enumerate(ends):
        out[4 * j] = np.asarray(e, np.float64)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--receipt', required=True); ap.add_argument('--n_streams', type=int, default=300); ap.add_argument('--calls', type=int, default=4)
    ap.add_argument('--seed', type=int, default=20261105); ap.add_argument('--steps', type=int, default=30); ap.add_argument('--max_views', type=int, default=32); ap.add_argument('--max_bank', type=int, default=4)
    a = ap.parse_args(); POOL2.mkdir(exist_ok=True)
    rec = json.load(open(a.receipt)); lin = rec['meta'].get('lineage', {}); init = resolve_init(lin); packed_model.LAYOUT = lin.get('layout', 'memorizon')
    lora_r = int(json.load(open(rec['meta']['config'])).get('lora_r', 0)); scale_mult = float(lin.get('scale_mult', 2.))
    packed_model.PLAIN = (Path(rec['meta']['config']).parent / 'plain.flag').exists()
    model, _, fp = fresh_model(prope=True, lora_r=lora_r, init=init); load_trainables(model, a.receipt); model.eval()
    vae = load_vae(); empty = empty_text(); scene, twox = make_scene(max_views=a.max_views); render = TwoXRender864(vae, scene)
    rows, _ = corpus864('train'); rows = [r for r in rows if r.get('mpu_valid', True) and r['latent_count'] >= 1 + 4 * a.calls + 8]
    rng = random.Random(a.seed); by = {}
    for r in rows:
        by.setdefault(r['scene_id'], []).append(r)
    scenes = sorted(by); rng.shuffle(scenes); pick = [rng.choice(by[sc]) for sc in scenes][:a.n_streams]
    json.dump(dict(receipt=a.receipt, ckpt_sha256=rec['sha256'], seed=a.seed, steps=a.steps, calls=a.calls, n_streams=len(pick), max_views=a.max_views,
                   loop='write_measured + replan offset', memory='2xplat', twox_ckpt=str(twox.ckpt)), open(POOL2 / 'meta.json', 'w'))
    done = 0; t_all = time.time()
    for r in pick:
        sid = r['stream_id']
        if all((POOL2 / f'{sid}_s{1 + 4 * k}.npz').exists() for k in range(1, a.calls + 1)):
            continue
        scene.set_anchor_gauge(*seed_gauge(r['stream_id']))                          # 5 Nov scale audit
        t0 = time.time(); st = Stream864(r, scale_mult); P = st.c2w.numpy().astype(np.float64); K = st.K.numpy(); mpu = st.metres_per_unit
        roller = LoopRoller2X(model, vae, a.steps, 3. if init == 'memorizon50' else 5., 1., empty, cam_scale=st.cam_scale, mpu=mpu, da3=None, write_measured=True,
                              replan=True, replan_mode='offset', early=None, render=render, gauge_id=st.gauge_id, schedule_mode='memorizon' if init == 'memorizon50' else 'wan',
                              max_bank=a.max_bank, scene=scene).start(st.latents[:, :, :1], st.K)
        items = {}; ok = True
        for c in range(a.calls):
            roller.call_plan(P[1 + 16 * c: 17 + 16 * c], st.text, a.seed + c)
            s = 1 + 4 * (c + 1)
            if any(roller.meas_hist.get(j) is None for j in range(1, s)):
                ok = False; break
            # the scene now holds anchor + all own latent ends at measured poses: render the RECORDED next command (the target of the row)
            command = P[4 * s - 3: 4 * s + 13]; nxt = P[4 * (s + 4) - 3: 4 * (s + 4) + 13]
            rgb, depth, alpha = scene.render(np.concatenate([command, nxt]), K, size=(480, 864)); valid = (alpha > .5)
            dtok_tgt = dtok_from_depth(np.where(valid.numpy(), depth.numpy(), np.nan))[[3, 7, 11, 15, 19, 23, 27, 31]]
            meas = [np.eye(4)] + [roller.meas_hist[j] for j in range(1, s)]
            _, dsrc, asrc = scene.render(np.stack(meas), K, size=(480, 864)); dtok_src = dtok_from_depth(np.where((asrc > .5).numpy(), dsrc.numpy(), np.nan))
            recent = list(range(max(1, s - 4), s)); cands = [j for j in range(1, max(1, s - 4))]
            bank = bank_by_visibility(K, command, dtok_tgt[:4], cands, [meas[j] for j in cands], [dtok_src[j] for j in cands],
                                      [(meas[j], dtok_src[j]) for j in [0, *recent]], a.max_bank)
            rz, mt, raw = render.render(None, None, K, mpu, command, st.gauge_id, boundary=4 * (s - 1))
            gen = torch.cat(roller.latents[:s], 2)
            items[s] = dict(stream_id=sid, s=s, gen_latents_bf16=pack_bf16(gen), meas_end=np.stack(meas), cmd_end=np.stack([np.asarray(roller.poses_cmd[0 if j == 0 else 4 * j], np.float64) for j in range(s)]),
                            plan_end=np.stack([P[4 * j] for j in range(s)]), bank_g=np.asarray(bank, np.int64), rz_g_bf16=pack_bf16(rz), meta_g=mt.numpy(),
                            dtok_tgt=dtok_tgt.astype(np.float32), dtok_src=dtok_src.astype(np.float32), coverage_g=float(raw['valid'].mean()),
                            scene_info=json.dumps(roller.diag_scene), residual=np.array([np.nan if d.get('residual') is None else d['residual'] for d in roller.diag], np.float32))
        if not ok:
            print(json.dumps(dict(stream=sid[:8], skipped='measurement invalid')), flush=True); continue
        for s, it in items.items():
            np.savez_compressed(POOL2 / f'{sid}_s{s}.npz', **it)
        done += 1; torch.cuda.empty_cache(); last = items[max(items)]
        print(json.dumps(dict(stream=sid[:8], gap_end=round(float(np.linalg.norm(last['meas_end'][-1][:3, 3] - last['plan_end'][-1][:3, 3])), 3), cov=round(last['coverage_g'], 3),
                              views=json.loads(last['scene_info']).get('views'), sec=round(time.time() - t0, 1), done=done)), flush=True)
    print('DONE', done, 'streams', round((time.time() - t_all) / 3600, 2), 'h')


if __name__ == '__main__':
    main(); os._exit(0)
