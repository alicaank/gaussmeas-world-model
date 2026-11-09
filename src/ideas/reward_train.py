"""Idea 2 (2026-10-22): memory-verified camera reward arm for the packed 864 teacher. Clip-level rollouts in the WorldCompass /
WorldCycle form with a reward-weighted velocity-matching objective (RVM / DiffusionNFT family: no likelihoods, our Euler ODE
sampler at few steps), rewards computed on decoded frames without gradients:
  camera   : DA3 realized pose of the chunk end vs the commanded end, UNALIGNED in the anchor gauge (calibrated metres/unit):
             e_t = |t_exec - t_cmd| / max(|command vector|, .05), e_R = geodesic(R_exec, R_cmd) / 15 deg;
  fidelity : LPIPS between the generated chunk and the RECORDED chunk at the commanded poses (commands come from recordings);
  memory   : masked L1 between the generated frames and the ReSplat render of the model's OWN memory at the commanded poses
             (self-verifiable, no GT; --render 1 also feeds that render as role-3 context);
  sharpness: penalty above 1.3x the recording's Laplacian sharpness.
Per update: one train stream, call n ~ U{1..calls}, prefix = recorded latents (teacher-forced) or own rollouts with measured-pose
writes (p = own_prefix_p); G candidates for call n (different noises, `steps` Euler steps); group z-scored rewards, top-k pushed
toward their own sample (FM at a shift-3 u), bottom-k pushed away through the reflected target 2 v_ref - (eps - x0) with v_ref
from the START weights (swapped in, no grad); anchor = FM on the recorded chunk with the same context (lambda_anchor).
Checkpoints via save_trainables (immutable), lineage records the parent receipt. --smoke 1: 2 tiny updates, nothing saved."""
import argparse, json, os, random, resource, time
from contextlib import contextmanager
from pathlib import Path
import numpy as np, torch
from scipy.ndimage import laplace
from evaluate_recorded import metrics_models
from realized_pose import load_da3
from data import corpus864, Stream864
from model_io import fresh_model, load_trainables, save_trainables, load_vae, empty_text, run_id
from train import make_optimizer, draw_u
from closed_loop import LoopRoller, geodesic_deg
from render864_rows import rec_frames
import packed_model


def sharp(frames):
    return float(np.mean([laplace(f.astype(np.float32).mean(-1)).var() for f in frames]))


class RefWeights:
    """Start-of-arm trainables kept on the host; swapped into the model for reference velocities."""

    def __init__(self, params):
        self.params = params; self.ref = {n: p.detach().cpu().clone() for n, p in params}

    @contextmanager
    def swapped(self):
        cur = {n: p.detach().cpu().clone() for n, p in self.params}
        with torch.no_grad():
            for n, p in self.params:
                p.copy_(self.ref[n].to(p.device))
        try:
            yield
        finally:
            with torch.no_grad():
                for n, p in self.params:
                    p.copy_(cur[n].to(p.device))


def make_offload(params, opt):
    """Context: trainable fp32 masters + optimizer state to the host while the ReSplat render runs (DiT bf16 stays)."""
    @contextmanager
    def ctx():
        moved = []; grads = []
        for n, p in params:                                                            # keep NO reference to the GPU tensors (10-22: a kept
            if p.data.is_cuda:                                                         #  reference left masters + states resident -> OOM)
                p.data = p.data.cpu(); moved.append(p)
            if p.grad is not None and p.grad.is_cuda:                                  # accumulated gradients (5 GB) also leave the GPU
                grads.append((p, p.grad.cpu())); p.grad = None
        states = []
        for st in opt.state.values():
            for k in list(st.keys()):
                v = st[k]
                if torch.is_tensor(v) and v.is_cuda:
                    st[k] = v.cpu(); states.append((st, k)); del v
        torch.cuda.empty_cache()
        try:
            yield
        finally:
            for p in moved:
                p.data = p.data.cuda()
            for p, g in grads:
                p.grad = g.cuda()
            for st, k in states:
                st[k] = st[k].cuda()
            torch.cuda.empty_cache()
    return ctx


def velocity_at(model, pk, noisy, u, text, K, grad):
    d = pk['dev']; lat = torch.cat([d['context'], noisy.to('cuda', torch.bfloat16)], 2)
    times = torch.cat([torch.zeros(pk['nctx']), torch.full((4,), float(u))]).cuda()
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx, torch.autocast('cuda', dtype=torch.bfloat16):
        return model(lat, d['rays'], d['roles'], d['positions'], times, text.to('cuda', torch.bfloat16), d['ci'], d['c2w'], K.cuda(), metadata=d['md'], checkpoint_blocks=grad).float()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--receipt', required=True); ap.add_argument('--out', required=True); ap.add_argument('--updates', type=int, default=100)
    ap.add_argument('--group', type=int, default=8); ap.add_argument('--topk', type=int, default=3); ap.add_argument('--steps', type=int, default=12); ap.add_argument('--calls', type=int, default=4)
    ap.add_argument('--own_prefix_p', type=float, default=0., help='10-22 audit: first experiment uses recorded prefixes only (GT LPIPS + anchor FM are only valid in the recorded context)'); ap.add_argument('--render', type=int, default=0, help='1: own-frame ReSplat render as context AND memory reward')
    ap.add_argument('--lambda_cam', type=float, default=1.); ap.add_argument('--lambda_lp', type=float, default=1.); ap.add_argument('--lambda_mem', type=float, default=.5); ap.add_argument('--lambda_sharp', type=float, default=.5)
    ap.add_argument('--lambda_anchor', type=float, default=1.); ap.add_argument('--lambda_neg', type=float, default=.5); ap.add_argument('--lr_backbone', type=float, default=1e-5); ap.add_argument('--lr_new', type=float, default=3e-5)
    ap.add_argument('--wd', type=float, default=1e-3); ap.add_argument('--clip', type=float, default=1.); ap.add_argument('--seed', type=int, default=20261022); ap.add_argument('--ckpt_every', type=int, default=25)
    ap.add_argument('--max_bank', type=int, default=4); ap.add_argument('--smoke', type=int, default=0); ap.add_argument('--min_free_gb', type=float, default=17.); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rec = json.load(open(a.receipt)); lin = rec['meta'].get('lineage', {})
    from model_io import resolve_init; init = resolve_init(lin); packed_model.LAYOUT = lin.get('layout', 'memorizon')
    lora_r = int(json.load(open(rec['meta']['config'])).get('lora_r', 0)); scale_mult = float(lin.get('scale_mult', 2.)); shift = 3. if init == 'memorizon50' else 5.
    if a.smoke:
        a.updates, a.group, a.topk, a.steps = 2, 2, 1, 4
    model, params, fp = fresh_model(prope=True, lora_r=lora_r, init=init); r0 = load_trainables(model, a.receipt); model.train()
    opt = make_optimizer(params, a.lr_backbone, a.lr_new, a.wd); ref = RefWeights(params)
    vae = load_vae(); emb, lp = metrics_models(); da3 = load_da3('cpu'); empty = empty_text(); render = None
    if a.render:
        from render864_rows import OwnRender864; render = OwnRender864(vae, da3, offload=make_offload(params, opt))
    rows, _ = corpus864('train'); rows = [r for r in rows if r.get('mpu_valid', True)]; rng = random.Random(a.seed); g = torch.Generator().manual_seed(a.seed)
    rid = run_id(out); lineage = dict(run_id=rid, init=dict(receipt=a.receipt, sha256=r0['sha256']), recipe='reward_arm_v1', corpus='864', layout=packed_model.LAYOUT, scale_mult=scale_mult, render=bool(a.render), args=vars(a))
    (out / 'config.json').write_text(json.dumps(dict(vars(a), lora_r=lora_r, streams=len(rows)), indent=1)); (out / 'lineage.json').write_text(json.dumps(lineage, indent=1))
    log = open(out / 'train.log', 'a'); t_start = time.time()
    for update in range(1, a.updates + 1):
        t0 = time.time(); torch.cuda.reset_peak_memory_stats()
        # ---- stream, call index, prefix ---------------------------------------------------------------------------
        while True:
            r = rng.choice(rows); s = Stream864(r, scale_mult); n_call = rng.randint(1, a.calls)
            if s.L >= 1 + 4 * n_call:
                break
        P = s.c2w.numpy(); mpu = s.metres_per_unit; K = s.K; da3.to('cuda'); vae.model.cuda()
        roller = LoopRoller(model, vae, a.steps, shift, 1., empty, cam_scale=s.cam_scale, mpu=mpu, da3=da3, write_measured=True, replan=False, early=None, render=render,
                            gauge_id=s.gauge_id, schedule_mode='memorizon' if init == 'memorizon50' else 'wan', max_bank=a.max_bank).start(s.latents[:, :, :1], K)
        own_prefix = n_call > 1 and rng.random() < a.own_prefix_p; model.eval()
        with torch.no_grad():
            if n_call > 1 and not own_prefix:
                npre = 4 * (n_call - 1) + 1; fr = rec_frames(r, [4 * i for i in range(npre)])
                roller.set_prefix(s.latents[:, :, :npre], P[:4 * (npre - 1) + 1], {i: fr[i] for i in range(npre)})
            elif n_call > 1:
                for c in range(n_call - 1):
                    roller.call_plan(P[1 + 16 * c: 17 + 16 * c], s.text, a.seed + 1000 * update + c)
            command = P[1 + 16 * (n_call - 1): 17 + 16 * (n_call - 1)]; rec_chunk = rec_frames(r, range(1 + 16 * (n_call - 1), 1 + 16 * n_call)); sharp_ref = sharp(rec_chunk[1::2])
            # ---- candidates + rewards --------------------------------------------------------------------------------
            cands = []
            for i in range(a.group):
                x, frames, pk, _ = roller.sample_chunk(command, s.text, a.seed + 7919 * update + i, commit=False)
                ends = [frames[4 * j + 3] for j in range(4)]; ps = roller.measure(extra_frames=ends, tail=4)
                start = ps[-5] if len(ps) >= 5 and ps[-5] is not None else np.asarray(roller.poses[-1], np.float64)
                cmd_vec = command[-1, :3, 3] - np.asarray(roller.poses[-1])[:3, 3]
                e_t = float(np.linalg.norm(ps[-1][:3, 3] - command[-1, :3, 3]) / max(np.linalg.norm(cmd_vec), .05)); e_R = geodesic_deg(ps[-1], command[-1]) / 15.
                lpv = float(lp(frames[1::2], rec_chunk[1::2]).mean()); sh = sharp(frames[1::2]); sp = max(0., sh / sharp_ref - 1.3); mem = 0.
                if pk['raw'] is not None:
                    gen = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 127.5 - 1; m = (pk['raw']['valid'] > .5).float()[:, None]
                    mem = float(((gen - pk['raw']['rgb']).abs() * m).sum() / (m.sum() * 3 + 1e-6))
                reward = -a.lambda_cam * (e_t + e_R) - a.lambda_lp * lpv - a.lambda_mem * mem - a.lambda_sharp * sp
                cands.append(dict(x=x, pk=pk, reward=reward, e_t=e_t, e_R=e_R, lpips=lpv, sharp=sh, mem=mem, exec_ratio=float(np.linalg.norm(ps[-1][:3, 3] - start[:3, 3]) / max(np.linalg.norm(cmd_vec), 1e-6))))
        da3.to('cpu'); vae.model.cpu(); torch.cuda.empty_cache(); model.train()                 # 10-22: VAE (1.3 GB) + DA3 leave the GPU for the loss phase
        rw = np.array([c['reward'] for c in cands]); z = (rw - rw.mean()) / (rw.std() + 1e-6); order = np.argsort(-z)
        pos = [int(i) for i in order[:a.topk] if z[i] > 0]; neg = [int(i) for i in order[::-1][:a.topk] if z[i] < 0]
        # ---- losses ---------------------------------------------------------------------------------------------------
        parts = dict(pos=0., neg=0., anchor=0.); text = s.text
        for i in pos:                                                                      # toward the good samples (reward-weighted FM)
            c = cands[i]; u, w = draw_u(1, g, 'shift3'); eps = torch.randn(c['x'].shape, generator=g); noisy = (1 - u) * c['x'] + u * eps
            v = velocity_at(model, c['pk'], noisy, u, text, K, grad=True); loss = float(min(z[i], 1.)) * float(w) * (v - (eps - c['x']).cuda()).square().mean()
            loss.backward(); parts['pos'] += float(loss); del v, loss
        if neg and a.lambda_neg > 0:                                                       # away from the bad samples, reflected about the start model's velocity
            draws = []
            for i in neg:
                c = cands[i]; u, w = draw_u(1, g, 'shift3'); eps = torch.randn(c['x'].shape, generator=g); draws.append((i, u, w, eps, (1 - u) * c['x'] + u * eps))
            with ref.swapped():
                vref = [velocity_at(model, cands[i]['pk'], noisy, u, text, K, grad=False).cpu() for i, u, w, eps, noisy in draws]
            for (i, u, w, eps, noisy), vr in zip(draws, vref):
                c = cands[i]; target = (2 * vr - (eps - c['x'])).cuda()
                v = velocity_at(model, c['pk'], noisy, u, text, K, grad=True); loss = a.lambda_neg * float(min(-z[i], 1.)) * float(w) * (v - target).square().mean()
                loss.backward(); parts['neg'] += float(loss); del v, loss
        if a.lambda_anchor > 0:                                                            # stay on the data: FM on the recorded chunk with the same context
            pk = cands[0]['pk']; x0 = s.latents[:, :, 4 * n_call - 3: 4 * n_call + 1].clone(); u, w = draw_u(1, g, 'shift3'); eps = torch.randn(x0.shape, generator=g); noisy = (1 - u) * x0 + u * eps
            v = velocity_at(model, pk, noisy, u, text, K, grad=True); loss = a.lambda_anchor * float(w) * (v - (eps - x0).cuda()).square().mean(); loss.backward(); parts['anchor'] = float(loss); del v, loss
        gn = float(torch.nn.utils.clip_grad_norm_([p for _, p in params], a.clip)); opt.step(); opt.zero_grad(set_to_none=True)
        rep = dict(update=update, stream=r['stream_id'][:8], call=n_call, own_prefix=own_prefix, reward=[round(float(x), 3) for x in rw], e_t=[round(c['e_t'], 3) for c in cands], e_R=[round(c['e_R'], 2) for c in cands],
                   lpips=[round(c['lpips'], 3) for c in cands], mem=[round(c['mem'], 3) for c in cands], ratio=[round(c['exec_ratio'], 2) for c in cands], sharp=[round(c['sharp']) for c in cands], sharp_ref=round(sharp_ref),
                   pos=pos, neg=neg, loss=parts, grad=round(gn, 4), sec=round(time.time() - t0, 1), peak=round(torch.cuda.max_memory_allocated() / 2 ** 30, 1),
                   rss=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 1), elapsed_h=round((time.time() - t_start) / 3600, 3))
        log.write(json.dumps(rep) + '\n'); log.flush(); print(json.dumps(rep), flush=True)
        del cands; torch.cuda.empty_cache()
        if not a.smoke and (update % a.ckpt_every == 0 or update == a.updates):
            import shutil
            if shutil.disk_usage(out).free < (a.min_free_gb + 3) * 2 ** 30:
                raise RuntimeError('disk floor: cannot write the trainables checkpoint')
            save_trainables(model, out / 'ckpt' / f'trainables_{update:05d}.pt', dict(update=update, recipe=lineage['recipe'], stock=fp, run_id=rid, lineage=lineage, config=str(out / 'config.json')))
    if a.smoke:
        (out / 'smoke.json').write_text(json.dumps(dict(ok=True, updates=a.updates)))
    print('done', a.updates, flush=True)


if __name__ == '__main__':
    main(); os._exit(0)
