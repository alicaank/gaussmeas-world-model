"""Closed-loop evaluation on the 864 held-out dev streams (recorded commands, C calls): arms = open loop / measured write /
F' (write + replan) / F' + early check / any of these + own-frame ReSplat render. Same metrics and eval.json layout as
eval_864.py (DINO/LPIPS/sharpness vs the recording, DA3 position error in anchor units and executed/commanded ratio measured
INDEPENDENTLY on the final frames) + optional VGGT-1B camera scoring (--vggt 1) + the loop's own diagnostics."""
import argparse, json, os, time
from pathlib import Path
import numpy as np, torch
from scipy.ndimage import laplace
from evaluate_recorded import metrics_models
from realized_pose import load_da3, realized_poses
from data import corpus864, Stream864
from model_io import fresh_model, load_trainables, load_vae, empty_text
from closed_loop import LoopRoller
from render864_rows import rec_frames
import packed_model


def sharp(frames):
    return float(np.mean([laplace(f.astype(np.float32).mean(-1)).var() for f in frames]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--receipt', required=True); ap.add_argument('--out', required=True); ap.add_argument('--calls', type=int, default=4)
    ap.add_argument('--steps', type=int, default=30); ap.add_argument('--guidance', type=float, default=1.); ap.add_argument('--seed', type=int, default=7); ap.add_argument('--limit', type=int, default=10); ap.add_argument('--offset', type=int, default=0, help='first dev row (10 = the second, untouched ten)')
    ap.add_argument('--write', type=int, default=0); ap.add_argument('--replan', type=int, default=0); ap.add_argument('--early', type=int, default=0); ap.add_argument('--render', type=int, default=0)
    ap.add_argument('--early_step', type=int, default=4); ap.add_argument('--early_tol', type=float, default=.3); ap.add_argument('--early_cos', type=float, default=.9); ap.add_argument('--kappa', type=float, default=1.)
    ap.add_argument('--vggt', type=int, default=1); ap.add_argument('--max_bank', type=int, default=4); ap.add_argument('--first_render', default='dup2', choices=['dup2', 'dup4', 'points']); ap.add_argument('--bestof', type=int, default=1)
    ap.add_argument('--context_mode', choices=['shared','reference'], default=None)
    ap.add_argument('--schedule_mode', choices=['wan','memorizon','memorizon_native'], default=None)
    ap.add_argument('--observer_seed', type=int, default=None)
    ap.add_argument('--save_video', type=int, default=1, help='save full-rate recorded/generated comparison for visual and seam review')
    ap.add_argument('--replan_mode', default='line', choices=['line', 'offset']); ap.add_argument('--deadband_t', type=float, default=0.); ap.add_argument('--deadband_r', type=float, default=0.)
    ap.add_argument('--memory', default='resplat', choices=['resplat', '2xplat'], help='2xplat: global 2Xplat scene as memory + observer (closed_loop_2x, PILOT_PLAN_CVPR_20261104)'); ap.add_argument('--max_views', type=int, default=32)
    ap.add_argument('--reanchor_u', type=float, default=0., help='idea C step 1: re-denoise every measured chunk from this level toward the planned poses (0 = off; needs --render 1 --write 1)'); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lin = {}
    if a.receipt == 'none':                                                            # zero-shot Memorizon-50 (no fine-tuning)
        init, lora_r, scale_mult = 'memorizon50', 0, 2.; packed_model.LAYOUT = 'memorizon'
        model, params, fp = fresh_model(prope=True, lora_r=0, init=init); model.eval()
    else:
        rec = json.load(open(a.receipt)); lin = rec['meta'].get('lineage', {})
        from model_io import resolve_init; init = resolve_init(lin); packed_model.LAYOUT = lin.get('layout', 'memorizon')
        lora_r = int(json.load(open(rec['meta']['config'])).get('lora_r', 0)); scale_mult = float(lin.get('scale_mult', 2.))
        packed_model.PLAIN = (Path(rec['meta']['config']).parent / 'plain.flag').exists()   # plain-Memorizon arm: same trainable scope
        model, params, fp = fresh_model(prope=True, lora_r=lora_r, init=init, md_ch=int(lin.get('md_ch', 2)))
        _keys = torch.load(rec['path'], map_location='cpu')['trainables'].keys(); _gflag = Path(rec['meta']['config']).parent / 'g.flag'
        if 'corr_gain' in _keys or any(k.startswith('reader.') for k in _keys):                         # gb / gr / gd arms (train_g.py): mode from g.flag
            _mode = {'gb': 'bias', 'gr': 'restricted', 'gd': 'dense'}[json.load(open(_gflag))['op_arm']] if _gflag.exists() else ('restricted' if any(k.startswith('reader.') for k in _keys) else 'bias')
            import packed_model_corr; model = packed_model_corr.upgrade(model, mode=_mode, reader_every=2); model.configure_trainables()
        load_trainables(model, a.receipt); model.eval()
    model.context_mode = a.context_mode or lin.get('context_mode','shared')
    chosen_schedule = a.schedule_mode or lin.get('schedule_mode', 'memorizon' if init == 'memorizon50' else 'wan')
    observer_seed = a.observer_seed if a.observer_seed is not None else lin.get('observer_rng_seed')
    vae = load_vae(); emb, lp = metrics_models(); da3 = load_da3('cpu'); empty = empty_text(); render = None
    if observer_seed is not None:
        from deterministic_observer import isolate_observer_rng
        da3 = isolate_observer_rng(da3, observer_seed)
    scene = None
    if a.render and a.memory == '2xplat':
        from closed_loop_2x import TwoXRender864, LoopRoller2X
        from twoxplat_memory import make_scene
        scene, twox = make_scene(max_views=a.max_views); render = TwoXRender864(vae, scene)
    elif a.render:
        from render864_rows import OwnRender864; render = OwnRender864(vae, da3); render.first_mode = a.first_render
        render.features = model.metadata.in_channels > 2                      # Gaussian-feature pilot: (depth, conflict, support) metadata
    vggt = None
    if a.vggt:
        from check_metrics_864 import vggt_relposes, cam_errors
        from score_vggt import load_vggt; vggt = load_vggt().cpu()
    rows, _ = corpus864('development'); rows = rows[a.offset:a.offset + a.limit]; results = []
    for r in rows:
        t0 = time.time(); s = Stream864(r, scale_mult); n = 1 + 16 * a.calls; torch.cuda.reset_peak_memory_stats()
        if s.L < 1 + 4 * a.calls:
            continue
        rec_fr = rec_frames(r, range(n)); P = s.c2w.numpy(); mpu = s.metres_per_unit; K = s.K.numpy(); da3.to('cuda')
        if scene is not None:
            from twoxplat_memory import seed_gauge; scene.set_anchor_gauge(*seed_gauge(r['stream_id']))   # 5 Nov scale audit
        roller = (LoopRoller2X if scene is not None else LoopRoller)(model, vae, a.steps, 3. if init == 'memorizon50' else 5., a.guidance, empty, cam_scale=s.cam_scale, mpu=mpu, da3=da3, write_measured=bool(a.write), replan=bool(a.replan), **({'scene': scene} if scene is not None else {}),
                            early=dict(step=a.early_step, tol=a.early_tol, cos_min=a.early_cos, kappa=a.kappa) if a.early else None, render=render, gauge_id=s.gauge_id,
                            schedule_mode=chosen_schedule, max_bank=a.max_bank, replan_mode=a.replan_mode,
                            deadband=(a.deadband_t, a.deadband_r), bestof=a.bestof, reanchor=dict(u0=a.reanchor_u) if a.reanchor_u > 0 else None).start(s.latents[:, :, :1], s.K)
        for c in range(a.calls):
            roller.call_plan(P[1 + 16 * c: 17 + 16 * c], s.text, a.seed + c)
        gen = roller.all_frames(); M = {}; Mprev = {}; peak_gen = torch.cuda.max_memory_allocated() / 2**30   # before scoring (VGGT/DA3 eval)
        T = {k: sum(d.get('time', {}).get(k, 0.) for d in roller.diag) for k in ('pack_render', 'denoise', 'decode', 'measure')}; gsec = sum(T.values())
        import resource as _rs; res_info = dict(peak_vram_gb=round(peak_gen, 2), gen_seconds=round(gsec, 1), fps=round(16 * a.calls / max(gsec, 1e-6), 3),
                                                time=T and {k: round(v, 1) for k, v in T.items()}, host_maxrss_gb=round(_rs.getrusage(_rs.RUSAGE_SELF).ru_maxrss / 2**20, 1))
        for call in range(a.calls):                                                       # independent final measurement (eval_864 protocol)
            lat = [gen[4 * j] for j in range(1, 4 * (call + 1) + 1)]
            ps, _ = realized_poses(da3, gen[0], None, lat, K, max_views=16, tail=5 if call else 4)
            for i, j in enumerate(range(4 * call + 1, 4 * (call + 1) + 1)):
                M[j] = ps[len(lat) - 4 + i]
            Mprev[call] = ps[len(lat) - 5] if call else np.eye(4)
        da3.to('cpu'); torch.cuda.empty_cache(); per = []
        for c in range(a.calls):
            idx = list(range(1 + 16 * c, 1 + 16 * (c + 1), 2)); ends = [4 * j for j in range(4 * c + 1, 4 * c + 5)]
            pe = [float(np.linalg.norm(M[j // 4][:3, 3] / mpu - P[j][:3, 3])) for j in ends]
            a_, b_ = Mprev[c], M[4 * c + 4]
            ratio = float(np.linalg.norm(b_[:3, 3] - a_[:3, 3]) / mpu / max(np.linalg.norm(P[16 * c + 16][:3, 3] - P[16 * c][:3, 3]), 1e-6))
            per.append(dict(call=c + 1, dino=float((emb(gen[idx]) * emb(rec_fr[idx])).sum(-1).mean()), lpips=float(lp(gen[idx], rec_fr[idx]).mean()),
                            sharp=sharp(gen[idx]), sharp_ref=sharp(rec_fr[idx]), pos_err=float(np.mean(pe)), exec_ratio=ratio))
        still = float((emb(np.repeat(gen[:1], 8, 0)) * emb(rec_fr[1:17:2])).sum(-1).mean())
        res = dict(stream=r['stream_id'][:8], resources=res_info, per_call=per, still_dino=still, mpu=mpu, diag=roller.diag, restarts=sum(bool(d.get('restart')) for d in roller.diag), sec=round(time.time() - t0, 1))
        if vggt is not None:
            from renderer864 import da3_gauge_depths
            da3.to('cuda'); depths, _ = da3_gauge_depths(da3, [gen[0]], K, P[:1], mpu); da3.to('cpu')
            model.cpu(); torch.cuda.empty_cache(); vggt.cuda()                                   # 10-call runs: VGGT sees 41 views -> DiT off the GPU meanwhile
            rel, sv = vggt_relposes(vggt, [gen[0]] + [gen[4 * j] for j in range(1, 4 * a.calls + 1)], depths[0].numpy()); vggt.cpu(); torch.cuda.empty_cache(); model.cuda()
            res['vggt'] = cam_errors({j: rel[j] for j in range(1, 4 * a.calls + 1)}, P, 1.0, a.calls); res['vggt_scale'] = sv
        if a.save_video:
            from visual_evidence import save_comparison
            res['visual_evidence'] = save_comparison(out / f"comparison_{r['stream_id'][:8]}.mp4", rec_fr, gen)
        results.append(res); np.savez_compressed(out / f"frames_{r['stream_id'][:8]}.npz", gen=gen[1::4])   # metrics artifact remains quarter rate; video retains every frame
        print(json.dumps(dict(stream=res['stream'], dino=[round(p['dino'], 3) for p in per], lpips=[round(p['lpips'], 3) for p in per], pos=[round(p['pos_err'], 3) for p in per],
                              ratio=[round(p['exec_ratio'], 2) for p in per], sharp=[round(p['sharp']) for p in per], restarts=res['restarts'],
                              vggt_pos=[round(p['pos_err'], 3) for p in res['vggt']] if 'vggt' in res else None, sec=res['sec'])), flush=True)
        json.dump(dict(receipt=a.receipt, init=fp, args=vars(a), scenes=results), open(out / 'eval.json', 'w'), indent=1)
    S = dict(n=len(results), still_dino=round(float(np.mean([r['still_dino'] for r in results])), 4), restarts=int(sum(r['restarts'] for r in results)))
    for k in ('dino', 'lpips', 'sharp', 'sharp_ref', 'pos_err', 'exec_ratio'):
        S[k] = [round(float(np.mean([r['per_call'][c][k] for r in results])), 4) for c in range(a.calls)]
    if vggt is not None:
        for k in ('pos_err', 'exec_ratio', 'rot_err_deg'):
            S['vggt_' + k] = [round(float(np.mean([r['vggt'][c][k] for r in results])), 4) for c in range(a.calls)]
    json.dump(dict(receipt=a.receipt, init=fp, args=vars(a), summary=S, scenes=results), open(out / 'eval.json', 'w'), indent=1); print('SUMMARY', json.dumps(S))


if __name__ == '__main__':
    main(); os._exit(0)
