"""One arm of the 7-scene DL3DV cohort (batch_continuous1009, 481 frames = 30 calls of 16) for the Memorizon-init
Gaussian-memory checkpoint run_op/meas (meas-3cf73fde, 250 on-policy updates on top of run_v864/mem).

Arms (identical seeds per call, identical inputs from inputs/<scene>.npz):
  baseline    open loop, plan sent unchanged
  sham        observer runs at every boundary (same DA3 calls as correction), output discarded, plan sent
  correction  the paper's controller: shared driver rejoin common.rejoin(M_b, P_b, plan) + memory-frame bridge
              U_j = L_b M_b^-1 D_j; memory labels stay commanded (no measured writes). A failed measurement sends the plan.
  fprime      the model's own F' arm (closed_loop_864 --write 1 --replan 1 --replan_mode offset): measured writes into the
              Gaussian memory labels + offset rejoin; labels are measured, so no bridge is needed.
Settings copied from eval864/mw_meas_*_s7: 30 steps, guidance 1, max_bank 4, own-frame ReSplat render, first_render dup2.
Output: review_full.mp4 (H.264 crf 12, the fair_common writer; frame 0 = decoded anchor) + run.json with per-boundary log."""
import argparse, hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
FT = Path('/home/hj/Alican/analysis/plan_base_restart_20261008/fresh_teacher_20261021')
NM = Path('/home/hj/Alican/analysis/plan_base_restart_20261008/new_models_loop_20261028')
FC = Path('/home/hj/Alican/analysis/plan_base_restart_20261008/fair_common_20261025')
RECEIPT = FT / 'run_op/meas/ckpt/trainables_00250.json'
sys.path[:0] = [str(FT), str(FC)]
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


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--scene', required=True); ap.add_argument('--arm', required=True, choices=ARMS)
    ap.add_argument('--out', required=True); ap.add_argument('--calls', type=int, default=30); ap.add_argument('--seed', type=int, default=7); a = ap.parse_args()
    out = Path(a.out)
    if (out / 'run.json').exists():
        raise FileExistsError('runs are immutable: ' + str(out))
    out.mkdir(parents=True, exist_ok=True)
    import packed_model
    from data import pad_text
    from model_io import fresh_model, load_trainables, load_vae, empty_text, resolve_init
    from realized_pose import load_da3
    from closed_loop import LoopRoller
    from render864_rows import OwnRender864
    from fair_common import write_mp4
    t0 = time.time()
    with np.load(HERE / 'inputs' / f'{a.scene}.npz') as z:
        inp = {k: z[k] for k in z.files}
    rec = json.loads(RECEIPT.read_text()); lin = rec['meta'].get('lineage', {})
    init = resolve_init(lin); packed_model.LAYOUT = lin.get('layout', 'memorizon')
    cfg = FT / rec['meta']['config'] if not Path(rec['meta']['config']).is_absolute() else Path(rec['meta']['config'])
    lora_r = int(json.loads(cfg.read_text()).get('lora_r', 0)); scale_mult = float(lin.get('scale_mult', 2.))
    packed_model.PLAIN = (cfg.parent / 'plain.flag').exists()
    model, params, fp = fresh_model(prope=True, lora_r=lora_r, init=init, md_ch=int(lin.get('md_ch', 2))); load_trainables(model, RECEIPT); model.eval()
    vae = load_vae(); da3 = load_da3('cpu'); empty = empty_text()
    render = OwnRender864(vae, da3); render.first_mode = 'dup2'; render.features = model.metadata.in_channels > 2
    mpu = float(inp['mpu_da3']); P = inp['c2w']; K = torch.from_numpy(inp['K'].astype(np.float32)); text = pad_text(torch.from_numpy(inp['text']).float())
    if len(P) < 1 + 16 * a.calls:
        raise ValueError('plan shorter than the rollout')
    torch.cuda.reset_peak_memory_stats(); da3.to('cuda')
    roller = LoopRoller(model, vae, 30, 3. if init == 'memorizon50' else 5., 1., empty, cam_scale=mpu * scale_mult / 4., mpu=mpu, da3=da3,
                        write_measured=a.arm == 'fprime', replan=a.arm != 'baseline', early=None, render=render, gauge_id=str(inp['scene_id']),
                        schedule_mode='memorizon' if init == 'memorizon50' else 'wan', max_bank=4, replan_mode='offset',
                        deadband=(0., 0.), bestof=1, reanchor=None).start(torch.from_numpy(inp['anchor_lat']).float(), K)
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
    run = dict(model='gaussian_meas', receipt=str(RECEIPT), receipt_sha256=rec['sha256'], init=fp, scene=a.scene, scene_id=str(inp['scene_id']), arm=a.arm,
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
