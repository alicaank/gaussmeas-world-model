"""Closed-loop roller for the packed 864 teacher (idea 1, 2026-10-22): memory written at the DA3-MEASURED pose, next command
replanned from the measured pose toward the planned end (F'), an EARLY camera check inside the call (one-step x0 at step k ->
DA3 pose -> if the executed vector misses the command, restart the chunk from u=1 with the command corrected by the one-step
inverse; re-noising at low u cannot move a camera that is decided in the first steps), and an optional ReSplat render of the
model's OWN frames at the commanded poses as role-3 context (OwnRender864). sample.py is untouched."""
import time
import numpy as np, torch
from scipy.spatial.transform import Rotation, Slerp
from foundation_model import packed_rays
from data import select_bank, assemble
from sample import Roller, schedule
from realized_pose import realized_poses
import packed_model


def rejoin(M, planned):
    """Command from the measured pose M toward the planned end of the call: translation linear, rotation slerp (F' replan)."""
    P = np.asarray(planned[-1], np.float64); M = np.asarray(M, np.float64); n = len(planned); out = np.tile(np.eye(4), (n, 1, 1))
    sl = Slerp([0, 1], Rotation.from_matrix(np.stack([M[:3, :3], P[:3, :3]])))
    for k in range(n):
        a = (k + 1) / n; out[k, :3, 3] = M[:3, 3] + a * (P[:3, 3] - M[:3, 3]); out[k, :3, :3] = sl(a).as_matrix()
    return out


def rejoin_offset(M, planned, prev_end):
    """10-23 audit: rejoin() throws away the plan's intermediate poses (with an EXACT start pose it still bends the classroom
    call 4 by .288 units / 13.2 deg). Here the plan is kept and only the start offset (measured M vs the plan's previous end)
    is removed linearly over the call: M == prev_end returns the plan unchanged, the last pose is always the planned end."""
    M = np.asarray(M, np.float64); E = np.asarray(prev_end, np.float64); planned = np.asarray(planned, np.float64); n = len(planned)
    dt = M[:3, 3] - E[:3, 3]; dR = Rotation.from_matrix(M[:3, :3] @ E[:3, :3].T)
    sl = Slerp([0, 1], Rotation.concatenate([dR, Rotation.identity()])); out = planned.copy()
    for k in range(n):
        a = (k + 1) / n; out[k, :3, 3] = planned[k, :3, 3] + (1 - a) * dt; out[k, :3, :3] = sl(a).as_matrix() @ planned[k, :3, :3]
    return out


def interp_rgb_poses(prev, ends):
    """16 RGB-rate poses (4 per latent) from the previous latent end to each measured latent end."""
    out = []; cur = np.asarray(prev, np.float64)
    for e in ends:
        e = np.asarray(e, np.float64); sl = Slerp([0, 1], Rotation.from_matrix(np.stack([cur[:3, :3], e[:3, :3]])))
        for k in range(4):
            a = (k + 1) / 4; T = np.eye(4); T[:3, 3] = cur[:3, 3] + a * (e[:3, 3] - cur[:3, 3]); T[:3, :3] = sl(a).as_matrix(); out.append(T)
        cur = e
    return out


def geodesic_deg(Ra, Rb):
    return float(np.degrees(Rotation.from_matrix(np.asarray(Ra)[:3, :3].T @ np.asarray(Rb)[:3, :3]).magnitude()))


class LoopRoller(Roller):
    def __init__(self, model, vae, steps, shift, guidance, empty, *, cam_scale, mpu, da3=None, write_measured=False, replan=False, early=None,
                 render=None, gauge_id=None, schedule_mode='memorizon', max_bank=6, replan_mode='line', deadband=(0., 0.), bestof=1, reanchor=None):
        super().__init__(model, vae, steps, shift, guidance, empty, schedule_mode=schedule_mode, cam_scale=cam_scale)
        self.reanchor = reanchor   # idea C step 1 (10-27): {'u0': level} -> reanchor_chunk after every measured call; None = off
        self.da3 = da3; self.mpu = float(mpu); self.write_measured = write_measured; self.replan = replan; self.render = render; self.gauge_id = gauge_id
        self.replan_mode = replan_mode; self.deadband = (float(deadband[0]), float(deadband[1]))   # (units, deg): residual inside -> no write, no replan
        self.bestof = int(bestof)   # idea 3 equal-compute control: N full samples per call (candidate 0 = the plain seed), DA3 residual picks
        self.early = {**dict(step=4, tol=0.3, cos_min=0.9, kappa=1.0), **early} if early is not None else None; self.max_bank = max_bank   # probe 10-22: x0 pose err .07 at step 4, restart gain 1

    def start(self, anchor_latent, K):
        super().start(anchor_latent, K); self.snap = self.decoder.snapshot(); self.poses_cmd = [torch.eye(4)]
        self.frames_by_latent = {0: self.frames[0][0]}; self.diag = []; self.poses = [np.eye(4)]
        self.meas_end = np.eye(4); self.plan_end = np.eye(4); return self                 # measured pose kept apart from the memory's pose labels

    # -- state from a recorded prefix (teacher-forced rollouts) -------------------------------------------------------
    def set_prefix(self, latents, poses_rgb, frames_by_latent):
        """latents (1,48,P,h,w) incl. the anchor; poses_rgb (1+4(P-1),4,4) anchor gauge; frames for latent ends 0..P-1."""
        self.latents = [latents[:, :, i:i + 1].clone() for i in range(latents.shape[2])]; self.poses = [np.asarray(p, np.float64) for p in poses_rgb]
        self.poses_cmd = [torch.as_tensor(p).float() for p in poses_rgb]; self.frames_by_latent = dict(frames_by_latent)
        self.decoder.restore(self.snap)
        if latents.shape[2] > 1:
            self.decoder.decode_chunk(latents[:, :, 1:].cuda().bfloat16())                      # advance the causal VAE stream through the prefix
        self.snap = self.decoder.snapshot(); return self

    # -- measurement --------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def measure(self, extra_frames=None, tail=4):
        """DA3 c2w (anchor gauge) of the own latent ends (+ extra frames), relative to the anchor; None for dropped context views."""
        lat = [self.frames_by_latent[i] for i in range(1, len(self.latents))] + (list(extra_frames) if extra_frames else [])
        ps, _ = realized_poses(self.da3, self.frames[0][0], None, lat, self.K.numpy(), max_views=16, tail=tail); out = []
        for p in ps:
            if p is None:
                out.append(None); continue
            q = np.asarray(p, np.float64).copy(); q[:3, 3] /= self.mpu; out.append(q)
        return out

    def decode(self, x):
        self.decoder.restore(self.snap)
        return self._uint8(self.decoder.decode_chunk(x.cuda().bfloat16())[0])

    # -- one call -----------------------------------------------------------------------------------------------------
    def pack(self, command, render_override=None):
        """render_override (rz, meta): use this render instead of render_own (reanchor_chunk)."""
        s = len(self.latents); poses = torch.stack([torch.as_tensor(p).float() for p in self.poses]); command = torch.as_tensor(np.asarray(command)).float()
        if poses.shape[0] != 1 + 4 * (s - 1):
            raise RuntimeError('pose bookkeeping broken')
        recent = list(range(max(1, s - 4), s))
        if s == 1 and packed_model.LAYOUT == 'memorizon':
            recent = [0]
        H, W = self.latents[0].shape[-2:]
        bank = select_bank(poses, command, self.K, s, max_refs=self.max_bank, size=(16 * H, 16 * W)) if s > 1 else []
        render = self.render is not None
        true_positions, roles, positions, ci = assemble(None, bank, recent, [s, s + 1, s + 2, s + 3], 1, render=render)
        all_poses = torch.cat([poses, command]).clone(); all_poses[:, :3, 3] *= self.cam_scale
        rays = packed_rays(all_poses, self.K, true_positions, height=16 * H, width=16 * W)
        c2w = torch.stack([all_poses[0 if p == 0 else 4 * p] for p in true_positions])
        context = torch.cat([self.latents[i] for i in [0, *bank, *recent]], 2); md = None; raw = None
        if render:
            if render_override is not None:
                rz, meta = render_override
            else:
                rz, meta, raw = self.render.render_own(self.frames_by_latent, self.poses, bank, s, self.K.numpy(), self.mpu, command.numpy(), self.gauge_id)
            md = torch.zeros((1, meta.shape[1], len(true_positions), *rz.shape[-2:])); n0 = 1 + len(bank) + len(recent); md[:, :, n0:n0 + 4] = meta; context = torch.cat([context, rz], 2)
        dev = 'cuda'
        return dict(context=context, rays=rays, roles=roles, positions=positions, ci=ci, c2w=c2w, md=md, nctx=context.shape[2], bank=bank, recent=recent, raw=raw, command=command,
                    dev=dict(context=context.to(dev, torch.bfloat16), rays=rays.to(dev, torch.bfloat16), roles=roles.to(dev), positions=positions.to(dev), ci=ci.to(dev),
                             c2w=c2w.to(dev), md=None if md is None else md.to(dev, torch.bfloat16)))

    @torch.no_grad()
    def denoise(self, x, times, pk, text, taps=()):
        d = pk['dev']; x = x.to('cuda').float(); x0s = {}
        for i, (u, v) in enumerate(zip(times[:-1], times[1:])):
            tt = torch.cat([torch.zeros(pk['nctx']), torch.full((4,), u)]).cuda()
            vel = self.velocity(torch.cat([d['context'], x.bfloat16()], 2), d['rays'], d['roles'], d['positions'], tt, text, d['ci'], d['c2w'], d['md'])[:, :, -4:]
            if i in taps:
                x0s[i] = (x - u * vel).cpu().float()
            x = x + (v - u) * vel
        if not torch.isfinite(x).all():
            raise RuntimeError('nonfinite sample')
        return x.cpu().float(), x0s

    def early_check(self, x0, command):
        """Executed vector of the one-step x0 (DA3) vs the commanded vector; returns (ok, info, corrected command)."""
        fr = self.decode(x0); ends = [fr[4 * j + 3] for j in range(4)]; ps = self.measure(extra_frames=ends, tail=4)
        start = ps[-5] if len(ps) >= 5 and ps[-5] is not None else np.asarray(self.poses[-1], np.float64)
        exec_vec = ps[-1][:3, 3] - start[:3, 3]; cmd = np.asarray(command, np.float64); cmd_vec = cmd[-1, :3, 3] - np.asarray(self.poses[-1])[:3, 3]
        nc = float(np.linalg.norm(cmd_vec)); ne = float(np.linalg.norm(exec_vec)); e = self.early
        if nc < 0.02:
            ok = ne <= 0.1; ratio = ne / 0.02; cos = 1.
        else:
            ratio = ne / nc; cos = float(exec_vec @ cmd_vec / (ne * nc + 1e-9)); ok = (1 - e['tol'] <= ratio <= 1 + e['tol']) and cos >= e['cos_min']
        info = dict(ratio=round(ratio, 3), cos=round(cos, 3), ok=bool(ok), rot_err=round(geodesic_deg(ps[-1], cmd[-1]), 2))
        if ok:
            return True, info, command
        corr = cmd.copy(); delta = e['kappa'] * (cmd_vec - exec_vec)
        for k in range(len(corr)):
            corr[k, :3, 3] += (k + 1) / len(corr) * delta
        info['delta'] = [round(float(x), 3) for x in delta]
        return False, info, corr

    def sample_chunk(self, command, text, seed, commit=True):
        """One chunk at `command` (16 absolute c2w, anchor gauge). Returns (x latents (1,48,4,h,w) cpu, frames (16,) uint8, pack, info)."""
        s = len(self.latents); H, W = self.latents[0].shape[-2:]; g = torch.Generator().manual_seed(seed); noise = torch.randn((1, 48, 4, H, W), generator=g)
        tp = time.perf_counter(); command = np.asarray(command, np.float64); pk = self.pack(command); t_pack = time.perf_counter() - tp   # pack includes the memory render
        times = schedule(self.steps, self.shift, self.schedule_mode); info = dict(call=s, restart=False); td = time.perf_counter()
        if self.early is not None:
            k = self.early['step']; x, x0s = self.denoise(noise, times[:k + 2], pk, text, taps={k})
            ok, chk, corr = self.early_check(x0s[k], command); info['check'] = chk
            if ok:
                x, _ = self.denoise(x, times[k + 1:], pk, text)
            else:
                command = corr; pk = self.pack(command); x, _ = self.denoise(noise, times, pk, text); info['restart'] = True
        elif self.bestof > 1:
            best = None; scores = []
            for i in range(self.bestof):
                nz = noise if i == 0 else torch.randn((1, 48, 4, H, W), generator=torch.Generator().manual_seed(seed + 1000 * i))
                xi, _ = self.denoise(nz, times, pk, text); fi = self.decode(xi)
                ends = self.measure(extra_frames=[fi[3], fi[7], fi[11], fi[15]], tail=5 if s > 1 else 4)[-4:]
                if ends[-1] is None:
                    sc = float('inf')
                else:
                    sc = float(np.linalg.norm(ends[-1][:3, 3] - command[-1, :3, 3])) / .04 + geodesic_deg(ends[-1], command[-1])   # 0.04 units ~ 1 deg (F' deadband)
                scores.append(round(sc, 3))
                if best is None or sc < best[0]:
                    best = (sc, i, xi)
            x = best[2]; info['bestof'] = dict(scores=scores, chosen=best[1])
        else:
            x, _ = self.denoise(noise, times, pk, text)
        torch.cuda.synchronize(); t_den = time.perf_counter() - td; tc = time.perf_counter(); frames = self.decode(x); t_dec = time.perf_counter() - tc; t_meas = 0.
        if commit:
            self.latents += [x[:, :, i:i + 1].clone() for i in range(4)]; self.frames.append(frames); self.poses_cmd += [torch.as_tensor(c).float() for c in command]
            self.poses += [np.asarray(c, np.float64) for c in command]
            for j in range(4):
                self.frames_by_latent[s + j] = frames[4 * j + 3]
            self.snap_prev = self.snap; self.snap = self.decoder.snapshot()                # decode(x) left the VAE stream after this chunk
            self.meas_end = np.asarray(command[-1], np.float64); self.last_ends = None
            if self.write_measured or self.replan:
                prev = np.asarray(self.poses[4 * (s - 1)], np.float64)                     # memory label of the previous end (before this write)
                tm = time.perf_counter(); ps = self.measure(tail=5 if s > 1 else 4); ends = ps[-4:]; t_meas = time.perf_counter() - tm
                if all(e is not None for e in ends):
                    rt = float(np.linalg.norm(ends[-1][:3, 3] - command[-1, :3, 3])); rr = geodesic_deg(ends[-1], command[-1])
                    inside = rt <= self.deadband[0] and rr <= self.deadband[1]
                    info['measured_end'] = [round(float(v), 3) for v in ends[-1][:3, 3]]; info['residual'] = round(rt, 3); info['residual_deg'] = round(rr, 2)
                    info['deadband'] = bool(inside); self.last_ends = list(ends)
                    if not inside:
                        self.meas_end = np.asarray(ends[-1], np.float64)
                        if self.write_measured:
                            self.poses[-16:] = interp_rgb_poses(prev, ends)
            info['time'] = dict(pack_render=round(t_pack, 3), denoise=round(t_den, 3), decode=round(t_dec, 3), measure=round(t_meas, 3))
            self.diag.append(info)
        return x, frames, pk, info

    def call_plan(self, planned, text, seed):
        """planned: 16 absolute poses of the plan for this call; replan from the measured pose if enabled."""
        planned = np.asarray(planned, np.float64)
        if self.replan and len(self.latents) > 1:
            command = rejoin(self.meas_end, planned) if self.replan_mode == 'line' else rejoin_offset(self.meas_end, planned, self.plan_end)
        else:
            command = planned
        self.plan_end = planned[-1].copy()
        out = self.sample_chunk(command, text, seed, commit=True)
        if self.reanchor is not None:
            self.reanchor_chunk(planned, text, seed)
        return out

    @torch.no_grad()
    def reanchor_chunk(self, planned, text, seed):
        """Idea C step 1 (user 10-27, training-free "re-anchoring"): after the call is measured, re-denoise the SAME chunk from
        level u0 (SDEdit: x_u0 = (1-u0) x + u0 eps, the remaining schedule steps) with the camera labels of the PLANNED poses and a
        ReSplat render whose sources are the anchor, the previous prefix end (at its memory label) and the chunk's own end
        frames 1 and 3 placed at their DA3-MEASURED poses, rendered at the planned poses: the generated content shown where it
        should be. The chunk's latents / frames / labels are replaced, it is measured again (written at the measured poses when
        --write 1) and the new measured end drives the next replan. Note the 10-22 probe: re-noising at low u alone could not
        move the camera; this test adds the geometric target through the render. Training renders never contained the target
        chunk's own frames (out-of-distribution source set, stated in the pre-registration)."""
        u0 = float(self.reanchor['u0']); ends = getattr(self, 'last_ends', None); s0 = len(self.latents) - 4; info = self.diag[-1]
        if ends is None or any(e is None for e in ends) or self.render is None or s0 < 1 or info.get('deadband'):
            info['reanchor'] = dict(skipped=True); return
        planned = np.asarray(planned, np.float64); t0 = time.perf_counter()
        ids = list(dict.fromkeys([0, s0 - 1, s0 + 1, s0 + 3]))[:4]
        src_pose = {0: self.poses[0], s0 - 1: self.poses[4 * (s0 - 1)]}
        for j in range(4):
            src_pose[s0 + j] = ends[j]
        frames = np.stack([self.frames_by_latent[i] for i in ids]); c2w = np.stack([np.asarray(src_pose[i], np.float64) for i in ids])
        rz, meta, raw = self.render.render(frames, c2w, self.K.numpy().astype(np.float64), self.mpu, planned, self.gauge_id, boundary=len(ids))  # frame_id = source index; the chunk's own frames are deliberate here
        x_old = torch.cat(self.latents[s0:s0 + 4], 2); self.latents = self.latents[:s0]; self.poses = self.poses[:-16]; self.poses_cmd = self.poses_cmd[:-16]
        self.frames.pop()
        for j in range(4):
            del self.frames_by_latent[s0 + j]
        self.snap = self.snap_prev                                                         # VAE stream before this chunk
        pk = self.pack(planned, render_override=(rz, meta)); t_pack = time.perf_counter() - t0
        eps = torch.randn(x_old.shape, generator=torch.Generator().manual_seed(seed + 500))
        times = [u0] + [t for t in schedule(self.steps, self.shift, self.schedule_mode) if t < u0]
        td = time.perf_counter(); x, _ = self.denoise((1 - u0) * x_old + u0 * eps, times, pk, text); torch.cuda.synchronize(); t_den = time.perf_counter() - td
        tc = time.perf_counter(); fr = self.decode(x); t_dec = time.perf_counter() - tc
        self.latents += [x[:, :, i:i + 1].clone() for i in range(4)]; self.frames.append(fr); self.poses_cmd += [torch.as_tensor(c).float() for c in planned]
        self.poses += [np.asarray(c, np.float64) for c in planned]
        for j in range(4):
            self.frames_by_latent[s0 + j] = fr[4 * j + 3]
        self.snap = self.decoder.snapshot(); self.meas_end = np.asarray(planned[-1], np.float64)
        prev = np.asarray(self.poses[4 * (s0 - 1)], np.float64); tm = time.perf_counter(); ps = self.measure(tail=5 if s0 > 1 else 4); ends2 = ps[-4:]; t_meas = time.perf_counter() - tm
        r = dict(u0=u0, steps=len(times) - 1, sources=ids, render_valid=round(float(raw['valid'].mean()), 3), residual_before=info.get('residual'), residual_deg_before=info.get('residual_deg'))
        if all(e is not None for e in ends2):
            rt = float(np.linalg.norm(ends2[-1][:3, 3] - planned[-1, :3, 3])); rr = geodesic_deg(ends2[-1], planned[-1])
            r.update(residual=round(rt, 3), residual_deg=round(rr, 2)); self.meas_end = np.asarray(ends2[-1], np.float64); self.last_ends = list(ends2)
            if self.write_measured:
                self.poses[-16:] = interp_rgb_poses(prev, ends2)
        r['time'] = dict(pack_render=round(t_pack, 3), denoise=round(t_den, 3), decode=round(t_dec, 3), measure=round(t_meas, 3)); info['reanchor'] = r
        for k, v in r['time'].items():
            info['time'][k] = round(info['time'].get(k, 0.) + v, 3)
