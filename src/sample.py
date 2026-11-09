"""Inference roller for the packed teacher: one chunk per call (4 latents, 16 RGB poses), recent = last <=4 own latents, bank = pose-retrieved
own latents (commanded poses for now), Euler over the shifted schedule, optional text CFG with the empty embedding.
Generated latents are kept raw (no decode/re-encode); frames are stream-decoded for metrics only."""
import numpy as np
import torch
from gaussverse.models.vae_stream import VaeStream
from foundation_model import packed_rays
from data import select_bank, assemble
import packed_model


def schedule(steps, shift=5., mode='wan'):
    """'wan': linspace(1,0,steps+1) through the shift map (ends at 0 after `steps` Euler steps).
    'memorizon': FlowMatchScheduler(shift, sigma_min=0.003/1.002): `steps` shifted sigmas then a final step to 0."""
    if mode == 'memorizon_native':
        # Released MemorizonModel constructs sigma_min=0, extra_one_step=True.
        # Preserve the older grid as 'memorizon' for checkpoint/experiment replay.
        t = torch.linspace(1., 0., steps + 1)[:-1]
        return (shift * t / (1 + (shift - 1) * t)).tolist() + [0.]
    if mode == 'memorizon':
        t = torch.linspace(1, 0.003 / 1.002, steps, dtype=torch.float64); return (shift * t / (1 + (shift - 1) * t)).tolist() + [0.]
    t = torch.linspace(1, 0, steps + 1, dtype=torch.float64)
    return (shift * t / (1 + (shift - 1) * t)).tolist()


class _NoModel(torch.nn.Module):
    """Stands in for the DiT inside joint_v9's WorldRuntime: only residency bookkeeping touches it."""
    native_patch_address = False

    def __init__(self):
        super().__init__(); self.p = torch.nn.Parameter(torch.zeros(1))


def make_runtime(vae):
    """joint_v9 geometry pipeline (DA3 on own frames + ReSplat render + VAE stream decode) without the DiT."""
    from joint_v9.runtime import WorldRuntime
    from gaussverse.training.gate_omega_recovery import PinnedDA3Predictor
    from foundation_renderer import FoundationRenderer
    return WorldRuntime(_NoModel(), vae, PinnedDA3Predictor(offload_after_call=True), FoundationRenderer(), residency='sequential')


class Roller:
    def __init__(self, model, vae, steps=30, shift=5., guidance=1., empty=None, history='recent', u_ctx=0.2, n_prev=2, runtime=None, cam_scale=1., cam_flip_y=False, rot_transpose=False, schedule_mode='wan'):
        """history='recent': own latents as clean recent slots (t=0). history='chunks': the last n_prev own chunks enter as
        NOISED predecessor chunks at level u_ctx (fresh noise, labelled u_ctx), the new chunk is the last chunk -- the
        configuration every training row with N>1 teaches; recent slots stay empty (they only ever held clean recordings)."""
        self.model = model; self.vae = vae; self.steps = steps; self.shift = shift; self.guidance = guidance; self.empty = empty
        self.history = history; self.u_ctx = u_ctx; self.n_prev = n_prev; self.runtime = runtime        # runtime != None: stage-B render per call
        self.cam_scale = cam_scale                                                                     # translation unit conversion for the model's camera inputs only
        self.cam_flip_y = cam_flip_y                                                                   # conjugate by diag(1,-1,1,1): a NO-OP for pure yaw and y-free translations (audit 10-21)
        self.rot_transpose = rot_transpose                                                             # rotation sense test: R -> R^T with translation unchanged
        self.schedule_mode = schedule_mode
        if guidance != 1. and empty is None:
            raise ValueError('CFG needs the empty text embedding')

    def start(self, anchor_latent, K, anchor=None):
        """anchor (render mode): object with .rgb uint8 (256,448,3), .depth, .confidence (dataset gauge), .gauge_id, .scene_id, .stream_id."""
        self.K = K.cpu().float()
        if self.runtime is not None:
            self.state = self.runtime.start(anchor.rgb, anchor, self.K, scene_id=anchor.scene_id, stream_id=anchor.stream_id, split='development')
            self.latents = [self.state.prefix[:, :, :1].clone()]; self.poses = [torch.eye(4)]; self.frames = [self.state.frames[:1].copy()]
            return self
        self.latents = [anchor_latent.cpu().float()]; self.poses = [torch.eye(4)]
        self.decoder = VaeStream(self.vae, offload_device='cpu')
        frames = self.decoder.decode_chunk(anchor_latent.cuda().bfloat16())[0]
        self.frames = [self._uint8(frames)]
        return self

    @staticmethod
    def _uint8(frames):                               # (3,T,H,W) in [-1,1] -> (T,H,W,3) uint8
        return ((frames.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 3, 0).cpu().numpy()

    @torch.no_grad()
    def velocity(self, lat, rays, roles, positions, times, text, ci, c2w, metadata=None):
        dev = 'cuda'
        def f(t):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                return self.model(lat, rays, roles, positions, times, t.to(dev, torch.bfloat16), ci, c2w, self.K.to(dev), metadata=metadata).float()
        v = f(text)
        if self.guidance == 1.:
            return v
        u = f(self.empty); return u + self.guidance * (v - u)

    @torch.no_grad()
    def call(self, command, text, seed, noise=None):
        """command (16,4,4) c2w in the anchor gauge for the 16 new RGB frames."""
        s = len(self.latents); poses = torch.stack(self.poses)                           # RGB-rate: anchor + 16 per past call
        if poses.shape[0] != 1 + 4 * (s - 1):                                        # four RGB poses per latent
            raise RuntimeError('pose bookkeeping broken')
        command = command.cpu().float(); g = torch.Generator().manual_seed(seed)
        if self.history == 'chunks':
            n_prev = min(self.n_prev, (s - 1) // 4); recent = []
            prev = list(range(s - 4 * n_prev, s))                                        # previous own chunks, oldest first
            bank_limit = s - 4 * n_prev                                                   # bank candidates before the predecessor chunks
            bank = select_bank(poses[:4 * (bank_limit - 1) + 1], command, self.K, bank_limit, recent=0, size=(16 * self.latents[0].shape[-2], 16 * self.latents[0].shape[-1])) if bank_limit > 1 else []
        else:
            n_prev = 0; prev = []; recent = list(range(max(1, s - 4), s))
            if s == 1 and packed_model.LAYOUT == 'memorizon':
                recent = [0]                                                              # Memorizon's first call: A also fills the recent slot
            bank = select_bank(poses, command, self.K, s, size=(16 * self.latents[0].shape[-2], 16 * self.latents[0].shape[-1])) if s > 1 else []   # 10-22: frustum test at the stream's size (was the 448 default at 864)
        render = self.runtime is not None
        if render and n_prev:
            raise ValueError('render mode uses the recent layout (render slots sit at the new chunk; predecessor chunks would shift them)')
        chunks = [*prev, s, s + 1, s + 2, s + 3]; true_positions, roles, positions, ci = assemble(None, bank, recent, chunks, 1 + n_prev, render=render)
        H, W = self.latents[0].shape[-2:]; all_poses = torch.cat([poses, command]).clone(); all_poses[:, :3, 3] *= self.cam_scale
        if self.cam_flip_y:
            Fy = torch.diag(torch.tensor([1., -1., 1., 1.])); all_poses = Fy @ all_poses @ Fy
        if self.rot_transpose:
            all_poses = all_poses.clone(); all_poses[:, :3, :3] = all_poses[:, :3, :3].transpose(1, 2)
        rays = packed_rays(all_poses, self.K, true_positions, height=16 * H, width=16 * W)
        c2w = torch.stack([all_poses[0 if p == 0 else 4 * p] for p in true_positions])
        context = torch.cat([self.latents[i] for i in [0, *bank, *recent]], 2); metadata = None
        if render:                                                                        # ReSplat render from OWN frames at the commanded poses
            c, _ = self.runtime.prepare(self.state, command); rr = c.roles
            rz = c.appearance[:, :, (rr[:-4] == 3)].float().cpu(); meta = c.metadata[:, :, rr == 3].float().cpu()
            metadata = torch.zeros((1, 2, len(true_positions), *rz.shape[-2:])); nctx0 = 1 + len(bank) + len(recent); metadata[:, :, nctx0:nctx0 + 4] = meta
            context = torch.cat([context, rz], 2); self.last_render_valid = float(meta[0, 0].mean())
        x = torch.randn((1, 48, 4, *context.shape[-2:]), generator=g) if noise is None else noise.clone().float()
        if n_prev:                                                                        # noised, labelled predecessor chunks
            x0p = torch.cat([self.latents[i] for i in prev], 2); eps = torch.randn(x0p.shape, generator=g)
            prev_noisy = (1 - self.u_ctx) * x0p + self.u_ctx * eps; context = torch.cat([context, prev_noisy], 2)
        dev = 'cuda'; context = context.to(dev, torch.bfloat16); x = x.to(dev); rays = rays.to(dev, torch.bfloat16)
        roles, positions, ci, c2w = roles.to(dev), positions.to(dev), ci.to(dev), c2w.to(dev); nctx = context.shape[2] - 4 * n_prev
        md = None if metadata is None else metadata.to(dev, torch.bfloat16)
        times = schedule(self.steps, self.shift, self.schedule_mode)
        for u, v in zip(times[:-1], times[1:]):
            tt = torch.cat([torch.zeros(nctx), torch.full((4 * n_prev,), float(self.u_ctx)), torch.full((4,), u)]).to(dev)
            vel = self.velocity(torch.cat([context, x.bfloat16()], 2), rays, roles, positions, tt, text, ci, c2w, md)[:, :, -4:]
            x = x + (v - u) * vel
        if not torch.isfinite(x).all():
            raise RuntimeError('nonfinite sample')
        if render:
            self.state = self.runtime.advance(self.state, command, x.cpu().float()); self.latents += [self.state.prefix[:, :, i:i + 1].clone() for i in range(s, s + 4)]
            self.poses += list(command); self.frames.append(self.state.frames[-16:].copy()); return dict(bank=bank, recent=recent, prev=prev, render_valid=self.last_render_valid)
        self.latents += [x[:, :, i:i + 1].cpu().float() for i in range(4)]; self.poses += list(command)   # one entry per latent
        frames = self.decoder.decode_chunk(x.bfloat16())[0]; self.frames.append(self._uint8(frames))
        return dict(bank=bank, recent=recent, prev=prev)

    def all_frames(self):
        return np.concatenate(self.frames)
