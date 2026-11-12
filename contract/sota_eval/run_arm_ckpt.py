"""run_arm.py of gaussian_meas_dl3dv7_20261103 (the SOTA-comparison protocol: batch_continuous1009 scenes, 30 calls of 16 at 864x480,
seed 7, arms baseline / sham / correction / fprime, same inputs/<scene>.npz) parametrised by checkpoint: --receipt <trainables json>
and --frontier off|inside (frustum-only render mask, robust_native_20261108/frontier.py tol 1e9, decided 9 Nov). Everything else is
byte-identical to the original so GM-B / GM-F rows are comparable with the "Ours" (run_op/meas) rows of compare_page_20261105.
The original docstring follows."""

import gm_env  # machine paths: gm_env.py at the repository root (scripts/env.sh puts it on PYTHONPATH)
import argparse, hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np, torch

HERE = Path(gm_env.EVAL_INPUTS)   # inputs/<scene>.npz read-only
RN = Path(gm_env.CONTRACT)
FT = Path(gm_env.CORE)
NM = Path((gm_env.FOUNDATION + '/new_models_loop'))
FC = Path(gm_env.FOUNDATION)
RECEIPT = None   # set from --receipt in main()
sys.path[:0] = [str(FT), str(FC), str(RN)]
spec = importlib.util.spec_from_file_location('driver_common', NM / 'common.py'); driver_common = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver_common)
ARMS = ('baseline', 'sham', 'correction', 'fprime')


def bridge(L_b, M_b, D):
    """Memory-frame bridge: label-frame commands whose motion from the latest history label equals the desired motion from
    the measured pose, L_b^-1 U_j = M_b^-1 D_j."""
    return np.einsum('ij,njk->nik', np.asarray(L_b, np.float64) @ np.linalg.inv(np.asarray(M_b, np.float64)), np.asarray(D, np.float64))


def plan_command(arm, planned, meas_end, measured_ok, plan_end, label_end):
    """Command for one call. planned: (16,4,4) plan; meas_end: measured end of the previous call (world = anchor frame);
    plan_end: planned end of the previous call; label_end: memory label of the latest frame."""
    from closed_loop import rejoin_offset
    if arm in ('baseline', 'sham') or meas_end is None:
        return planned, 'plan'
    if arm == 'correction':
        if not measured_ok:
            return planned, 'plan_measurement_failed'
        return bridge(label_end, meas_end, driver_common.rejoin(np.asarray(meas_end, np.float64), np.asarray(plan_end, np.float64), planned)), 'rejoin_bridge'
    return rejoin_offset(meas_end, planned, plan_end), 'fprime_offset'


def geometry_check(render, scene, P, K, mpu, gauge_id):
    """One RECORDED window through the very render object the rollout uses: sources = recorded frames 0 and 16 at their plan poses, target = the
    next 16 plan poses; PSNR / best vertical rescale of the first rendered target against recorded frame 17 on valid pixels. The stock 540-row
    path scores ~18 dB with a best vertical scale of .948; a correct render >= 24 dB at 1.0."""
    import cv2
    from PIL import Image
    from torch.nn import functional as F
    row = next(r for r in json.load(open((gm_env.EVAL_BATCH + '/index.json')))['scenes'] if r['name'] == scene)
    fr = lambda t: cv2.resize(np.asarray(Image.open(Path(row['frames_dir']) / f"frame_{row['original_start'] + t + 1:05d}.jpg").convert('RGB')), (864, 480), interpolation=cv2.INTER_AREA)
    _, _, raw = render.render(np.stack([fr(0), fr(16)]), np.stack([P[0], P[16]]).astype(np.float64), np.asarray(K, np.float64), mpu, np.asarray(P[17:33], np.float64), gauge_id, 16, encode=False)
    rgb, v = raw['rgb'][:1].float(), raw['valid'][:1].float()[:, None]; tgt = torch.from_numpy(fr(17)).permute(2, 0, 1)[None].float() / 127.5 - 1; out = {}
    for f in (.948, 1.0):
        th = torch.tensor([[[1., 0, 0], [0, 1 / f, 0]]]); g = F.affine_grid(th, rgb.shape, align_corners=False)
        w = F.grid_sample(rgb, g, align_corners=False); m = (F.grid_sample(v, g, align_corners=False) > .99).expand(-1, 3, -1, -1)
        out[f] = round(float(10 * torch.log10(4 / ((w - tgt)[m] ** 2).mean())), 2) if float(m.float().mean()) > .05 else None
    return dict(psnr=out[1.0], psnr_if_squeezed_948=out[.948], valid=round(float(v.mean()), 3), alpha=round(float(raw['conf'][:1][raw['valid'][:1] > .5].mean()), 3))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--scene', required=True); ap.add_argument('--arm', required=True, choices=ARMS)
    ap.add_argument('--out', required=True); ap.add_argument('--calls', type=int, default=30); ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--receipt', required=True); ap.add_argument('--frontier', default='off', choices=('off', 'inside', 'density')); ap.add_argument('--support_min', type=float, default=.15); ap.add_argument('--alpha_min', type=float, default=0.5)
    ap.add_argument('--sources', default='', choices=('', 'default', 'firstseen', 'oldest', 'recent4', 'recent8', 'spread8', 'recent16')); ap.add_argument('--bank', default='', choices=('', 'unseen', 'revisit', 'memory', 'memorizon', 'contract', 'none')); ap.add_argument('--moment_patch', type=float, default=0.); ap.add_argument('--render_comp', type=int, default=0); ap.add_argument('--resplat', default='', choices=('', 'v8', 'v16s', 'v16p')); ap.add_argument('--pad_views', type=int, default=0); ap.add_argument('--offload', type=int, default=0); ap.add_argument('--render_fix', default='', choices=('', 'tone', 'full')); ap.add_argument('--ctx_noise', type=float, default=0.); ap.add_argument('--alpha_gamma', type=float, default=0.); ap.add_argument('--conf_one', type=int, default=0); ap.add_argument('--render_dump', type=int, default=0); ap.add_argument('--oracle', type=int, default=0); ap.add_argument('--oracle_ctx', type=int, default=0); ap.add_argument('--oracle_prefix', type=int, default=0); ap.add_argument('--anchor_conf', type=int, default=0); ap.add_argument('--moment_ref', default='prev', choices=('prev', 'anchor', 'mix')); ap.add_argument('--return_traj', type=int, default=0); ap.add_argument('--stride', type=int, default=1); ap.add_argument('--guidance', type=float, default=1.); ap.add_argument('--no_render', type=int, default=0); ap.add_argument('--inputs_dir', default=''); ap.add_argument('--text_npz', default=''); ap.add_argument('--unseen_conf', type=float, default=1.); a = ap.parse_args()
    if a.oracle_prefix: a.oracle = a.oracle_ctx = 1                                                  # 11 Nov: full oracle (render sources + context) for calls 1..N, own afterwards
    global RECEIPT; RECEIPT = Path(a.receipt)
    out = Path(a.out)
    if (out / 'run.json').exists():
        raise FileExistsError('runs are immutable: ' + str(out))
    out.mkdir(parents=True, exist_ok=True)
    if a.frontier != 'off' or a.conf_one or a.render_dump or a.alpha_gamma or a.render_comp or a.render_fix or a.resplat or a.pad_views:                                     # 10 Nov: conf=1 test / render video dump (mask only with 'inside')
        import frontier; frontier.install({'inside': 'reproj', 'density': 'density'}.get(a.frontier, 'off'), 1, str(out / 'frontier_log.jsonl'), tol=1e9, margin=2, alpha_min=a.alpha_min,
                                          conf_one=bool(a.conf_one), render_dump=str(out / 'render') if a.render_dump else None, alpha_gamma=a.alpha_gamma, comp=int(a.render_comp), render_fix=a.render_fix, size=((540, 960) if a.resplat == 'v16p' else (512, 960)) if a.resplat else None, pad_to=a.pad_views, support_min=a.support_min)
    if a.sources:                                                                                 # 10 Nov: render-memory source policy (source_policy.py), logged
        import source_policy; source_policy.install(a.sources, out / 'sources_log.jsonl')
    if a.bank:
        import source_policy; source_policy.install_bank(a.bank)
    if a.ctx_noise:
        import source_policy; source_policy.install_ctx_noise(a.ctx_noise)
    if a.moment_patch:
        import source_policy; source_policy.install_moment(a.moment_patch, out / 'moment_log.jsonl', ref=a.moment_ref)
    import packed_model
    from data import pad_text
    from model_io import fresh_model, load_trainables, load_vae, empty_text, resolve_init
    from realized_pose import load_da3
    from closed_loop import LoopRoller
    from render864_rows import OwnRender864
    from fair_common import write_mp4
    t0 = time.time()
    with np.load((Path(a.inputs_dir) if a.inputs_dir else HERE / 'inputs') / f'{a.scene}.npz') as z:   # --inputs_dir eval_gm/inputs_long = the full 1009-pose plans (same anchor, text, K)
        inp = {k: z[k] for k in z.files}
    rec = json.loads(RECEIPT.read_text()); lin = rec['meta'].get('lineage', {})
    init = resolve_init(lin); packed_model.LAYOUT = lin.get('layout', 'memorizon')
    cfg = FT / rec['meta']['config'] if not Path(rec['meta']['config']).is_absolute() else Path(rec['meta']['config'])
    lora_r = int(json.loads(cfg.read_text()).get('lora_r', 0)); scale_mult = float(lin.get('scale_mult', 2.))
    packed_model.PLAIN = (cfg.parent / 'plain.flag').exists()
    model, params, fp = fresh_model(prope=True, lora_r=lora_r, init=init, md_ch=int(lin.get('md_ch', 2)))
    import channel_render; channel = channel_render.maybe_install(model, cfg.parent)                 # 12 Nov: stage-B channel arm (render on the target tokens)
    load_trainables(model, RECEIPT); model.eval()
    vae = load_vae(); da3 = load_da3('cpu'); empty = empty_text()
    render = OwnRender864(vae, da3); render.first_mode = 'dup2'; render.features = model.metadata.in_channels > 2
    if a.sources.startswith(('recent', 'spread')): render.MAX_SOURCES = 16
    if a.resplat:                                                                                  # 10 Nov: corrected renderers (resplat_fixed.py): v8 | v16s | v16p
        sys.path.insert(0, gm_env.CONTRACT); import resplat_fixed; render.renderer, _ = resplat_fixed.make(a.resplat)
    if a.offload:                                                                                  # 8+ views of ReSplat do not fit beside the generator: it waits on the CPU during the render
        from contextlib import contextmanager

        @contextmanager
        def _offload():
            model.to('cpu'); torch.cuda.empty_cache()
            try: yield
            finally: model.to('cuda')
        render.offload = _offload
    mpu = float(inp['mpu_da3']); P = inp['c2w']; K = torch.from_numpy(inp['K'].astype(np.float32)); text = pad_text(torch.from_numpy(inp['text']).float())
    if a.stride > 1: P = P[::a.stride]                                                             # 11 Nov: every N-th recorded pose (test plans are consecutive 60 fps frames: ~14x slower per frame than the training streams)
    if a.return_traj:                                                                              # 11 Nov: out-and-back plan (calls/2 forward on the recorded path, then the same poses backwards):
        n_out = 16 * (a.calls // 2); P = np.concatenate([P[:n_out + 1], P[n_out - 1::-1][:n_out]])     # frame t and frame 2 n_out - t share a pose -> self-consistency without the recording
    if a.bank: sys.modules['source_policy'].MPU = mpu
    if a.text_npz:                                                                                 # 11 Nov: another prompt (scene summary) instead of the stock anchor caption
        with np.load(a.text_npz) as zt: text = pad_text(torch.from_numpy(zt['text']).float()); print('TEXT', str(zt['caption'])[:200], flush=True)
    if len(P) < 1 + 16 * a.calls:
        raise ValueError('plan shorter than the rollout')
    torch.cuda.reset_peak_memory_stats(); da3.to('cuda')
    if a.no_render: render = None                                                                    # stage-A models (no render rows in training): the roller packs no render slots
    geom = dict(psnr=None, skipped='no_render') if a.no_render else geometry_check(render, a.scene, inp['c2w'], inp['K'], mpu, str(inp['scene_id']))                   # 10 Nov 20:00: the inference path must render like the cache (critique)
    print('GEOM_CHECK', json.dumps(geom), flush=True)
    if 'frontier' in sys.modules: sys.modules['frontier'].STATE['calls'] = 0                        # the check's render must not shift the per-call dump numbering
    if a.resplat and (geom['psnr'] is None or geom['psnr'] < 21): raise RuntimeError(f'inference render fails the geometry check: {geom}')
    if a.oracle:                                                                                   # 11 Nov critique: oracle memory on the test scenes = recorded frames at recorded (plan) poses as ReSplat sources
        import cv2
        from PIL import Image
        row = next(r for r in json.load(open((gm_env.EVAL_BATCH + '/index.json')))['scenes'] if r['name'] == a.scene)
        recf = lambda t: cv2.resize(np.asarray(Image.open(Path(row['frames_dir']) / f"frame_{row['original_start'] + t + 1:05d}.jpg").convert('RGB')), (864, 480), interpolation=cv2.INTER_AREA)
        _own = render.render_own
        def _oracle_own(frames_by_latent, poses, *rest):
            if a.oracle_prefix and (rest[1] - 1) // 4 >= a.oracle_prefix: return _own(frames_by_latent, poses, *rest)
            return _own({i: recf(0 if i == 0 else 4 * i) for i in frames_by_latent}, [np.asarray(P[j], np.float64) for j in range(len(poses))], *rest)
        render.render_own = _oracle_own; print('ORACLE MEMORY: recorded frames + recorded poses as sources', flush=True)
    roller = LoopRoller(model, vae, 30, 3. if init == 'memorizon50' else 5., a.guidance, empty, cam_scale=mpu * scale_mult / 4., mpu=mpu, da3=da3,
                        write_measured=a.arm == 'fprime', replan=a.arm != 'baseline', early=None, render=render, gauge_id=str(inp['scene_id']),
                        schedule_mode='memorizon' if init == 'memorizon50' else 'wan', max_bank=4, replan_mode='offset',
                        deadband=(0., 0.), bestof=1, reanchor=None).start(torch.from_numpy(inp['anchor_lat']).float(), K)
    if a.oracle_ctx:                                                                               # 11 Nov critique: oracle CONTEXT = near + bank latents from the recording (causal VAE stream of the
        import cv2                                                                                 # recorded frames, as prep_864), render memory still from the model's OWN frames; anchor latent unchanged
        from PIL import Image
        from gmlib.models.vae_stream import VaeStream
        row = next(r for r in json.load(open((gm_env.EVAL_BATCH + '/index.json')))['scenes'] if r['name'] == a.scene)
        recf = lambda t: cv2.resize(np.asarray(Image.open(Path(row['frames_dir']) / f"frame_{row['original_start'] + t + 1:05d}.jpg").convert('RGB')), (864, 480), interpolation=cv2.INTER_AREA)
        T = 1 + 16 * a.calls; parts = []
        with torch.no_grad():
            enc = VaeStream(vae)
            for lo, hi in [(0, min(33, T))] + [(t, min(t + 32, T)) for t in range(33, T, 32)]:
                x = torch.from_numpy(np.stack([recf(j) for j in range(lo, hi)])).permute(3, 0, 1, 2).float()[None] / 127.5 - 1
                parts.append(enc.encode_chunk(x.cuda().bfloat16()).float().cpu())
            del enc
        rec_lat = torch.cat(parts, 2); anchor_rel = float((rec_lat[:, :, :1] - roller.latents[0].float()).norm() / roller.latents[0].float().norm())
        if rec_lat.shape[2] != 1 + 4 * a.calls: raise RuntimeError(f'oracle context: {rec_lat.shape[2]} recorded latents for {T} frames')
        print('ORACLE CONTEXT: recorded latents as near/bank context, own frames in the render; anchor rel l2', round(anchor_rel, 4), flush=True)
        _pack = roller.pack
        def _oracle_pack(command, render_override=None):
            if a.oracle_prefix and (len(roller.latents) - 1) // 4 >= a.oracle_prefix: return _pack(command, render_override)
            own = roller.latents; roller.latents = [own[0]] + [rec_lat[:, :, i:i + 1].to(own[0].dtype) for i in range(1, len(own))]
            try: return _pack(command, render_override)
            finally: roller.latents = own
        roller.pack = _oracle_pack
    if a.anchor_conf:                                                                              # 11 Nov selective trust: conf = 1 on render cells the anchor sees (as train_gmp --selective)
        sys.path.insert(0, gm_env.CONTRACT); import anchor_seen
        a_depth = anchor_seen.anchor_depth(da3, roller.frames[0][0], inp['K'], mpu); a_K = anchor_seen.scaled_K(inp['K'])
        _own2 = render.render_own; SEEN = []
        def _anchor_conf_own(frames_by_latent, poses, bank, s_, K_, mpu_, target_c2w, gauge_id):
            rz, meta, raw = _own2(frames_by_latent, poses, bank, s_, K_, mpu_, target_c2w, gauge_id)
            seen = torch.from_numpy(anchor_seen.seen_cells(a_depth, a_K, [np.asarray(target_c2w)[4 * j + 3] for j in range(4)]))
            meta = meta.clone(); meta[:, 1] = torch.where(seen[None], torch.ones_like(meta[:, 1]), meta[:, 1] * a.unseen_conf); SEEN.append(round(float(seen.float().mean()), 3))
            return rz, meta, raw
        render.render_own = _anchor_conf_own; print('ANCHOR CONF: median anchor depth', round(float(np.median(a_depth[a_depth > 0])), 3), flush=True)
    log = []
    for c in range(a.calls):
        planned = np.asarray(P[1 + 16 * c: 17 + 16 * c], np.float64)
        first = c == 0
        meas_end = None if first else np.asarray(roller.meas_end, np.float64)
        ok = (not first) and roller.last_ends is not None
        label_end = np.asarray(roller.poses[-1], np.float64)
        command, rule = plan_command(a.arm, planned, meas_end, ok, roller.plan_end, label_end)
        roller.plan_end = planned[-1].copy()
        _, _, _, info = roller.sample_chunk(np.asarray(command, np.float64), text, a.seed + c, commit=True)
        log.append(dict(call=c + 1, seed=a.seed + c, rule=rule, measured_ok=bool(ok),
                        measured_start=None if meas_end is None else meas_end[:3, 3].round(5).tolist(),
                        plan_end=planned[-1][:3, 3].round(5).tolist(), command_first=np.asarray(command)[0][:3, 3].round(5).tolist(),
                        command_end=np.asarray(command)[-1][:3, 3].round(5).tolist(), label_end_before=label_end[:3, 3].round(5).tolist(),
                        command_max_step=float(np.linalg.norm(np.diff(np.concatenate([label_end[None], np.asarray(command)])[:, :3, 3], axis=0), axis=1).max()),
                        plan_max_step=float(np.linalg.norm(np.diff(np.asarray(P[16 * c: 17 + 16 * c])[:, :3, 3], axis=0), axis=1).max()),
                        residual=info.get('residual'), residual_deg=info.get('residual_deg'), time=info.get('time')))
        print(json.dumps(log[-1]), flush=True)
    gen = roller.all_frames(); peak = torch.cuda.max_memory_allocated() / 2 ** 30
    if gen.shape != (1 + 16 * a.calls, 480, 864, 3) or gen.dtype != np.uint8:
        raise RuntimeError(f'unexpected frames {gen.shape} {gen.dtype}')
    write_mp4(out / 'review_full.mp4', gen)
    if a.render_dump:                                                                              # 10 Nov: latents for the latent-level seam check
        lat = torch.cat([x.cpu() for x in roller.latents], 2).to(torch.bfloat16).view(torch.int16).numpy(); np.savez_compressed(out / 'latents.npz', latents_bf16=lat)
    run = dict(model='gaussian_meas', render_mode='channel' if channel else 'token', frontier=a.frontier, support_min=a.support_min, alpha_min=a.alpha_min, conf_one=bool(a.conf_one), sources=a.sources or 'default', bank=a.bank or 'default', moment_patch=a.moment_patch, moment_ref=a.moment_ref, ctx_noise=a.ctx_noise, render_comp=int(a.render_comp), oracle=int(a.oracle), oracle_ctx=int(a.oracle_ctx), oracle_prefix=int(a.oracle_prefix), anchor_conf=int(a.anchor_conf), return_traj=int(a.return_traj), stride=int(a.stride), guidance=float(a.guidance), no_render=int(a.no_render), inputs_dir=a.inputs_dir, text_npz=a.text_npz, unseen_conf=float(a.unseen_conf), render_fix=a.render_fix, geometry_check=geom, resplat=a.resplat or 'v16', pad_views=a.pad_views, offload=bool(a.offload), alpha_gamma=a.alpha_gamma, receipt=str(RECEIPT), receipt_sha256=rec['sha256'], init=fp, scene=a.scene, scene_id=str(inp['scene_id']), arm=a.arm,
               seed=a.seed, calls=a.calls, frames=int(len(gen)), mpu_da3=mpu, cam_scale=mpu * scale_mult / 4., scale_mult=scale_mult,
               settings=dict(steps=30, guidance=1., max_bank=4, render=1, first_render='dup2', replan_mode='offset', deadband=[0, 0]),
               input_sha256=hashlib.sha256((HERE / 'inputs' / f'{a.scene}.npz').read_bytes()).hexdigest(),
               frame_sha1=[hashlib.sha1(f.tobytes()).hexdigest() for f in gen], peak_vram_gib=round(peak, 2),
               seconds=round(time.time() - t0, 1), boundaries=log, diag=roller.diag)
    with (out / 'run.json').open('x') as f:
        json.dump(run, f, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    print('RUN_OK', json.dumps(dict(scene=a.scene, arm=a.arm, frames=len(gen), seconds=run['seconds'], peak_vram_gib=run['peak_vram_gib'])), flush=True)


if __name__ == '__main__':
    main()
