"""ReSplat (540x960 zoo weights) render memory for the 864x480 teacher: training rows from the RECORDED causal prefix
(stage B, RenderBuilder864.render_chunk0 == render_rows.RenderBuilder's contract) and renders from the model's OWN frames at
inference / in reward rollouts (OwnRender864). Geometry: DA3 metric depth of each source frame with its known camera, divided
by metres_per_anchor_unit -> anchor gauge; cameras are the stored (commanded or DA3-measured) poses in the anchor gauge, i.e.
the same units as the command before the model's cam_scale. Output: render latents (1,48,4,30,54) VAE-encoded with the first
render frame as the stream prime, metadata (valid, conf) area-pooled to (1,2,4,30,54), plus the raw rgb/valid for rewards."""
from pathlib import Path
import numpy as np, torch
from torch.nn import functional as F
from PIL import Image
from gaussverse.models.vae_stream import VaeStream


def rec_frames(row, idxs):
    """Recorded 864x480 jpgs of a data864 row at RGB indices idxs (0 = anchor)."""
    src = Path(row['source']) / 'frames_864x480'
    return np.stack([np.array(Image.open(src / f"frame_{row['original_start'] + int(j) + 1:05d}.jpg").convert('RGB')) for j in idxs])


def pool_features(feat, hw):
    """(16,3,fh,fw) target-view features -> (1,3,4,h,w): mean over the 4 RGB frames of each latent, area-pooled (as valid/conf)."""
    n, c = feat.shape[:2]; x = feat.reshape(n // 4, 4, c, *feat.shape[-2:]).mean(1)
    return F.interpolate(x, size=hw, mode='area').permute(1, 0, 2, 3)[None].float()


class Render864:
    MAX_SOURCES = 4            # resplat_mem_864.json: 540x960 peak 9.4 GiB (1 source) / 15.4 GiB (4 sources) incl. DA3

    def __init__(self, vae, da3=None, offload=None):
        from renderer864 import Renderer864
        from realized_pose import load_da3
        self.vae = vae; self.da3 = da3 if da3 is not None else load_da3('cpu'); self.renderer = Renderer864(); self._own_da3 = da3 is None
        self.offload = offload     # optional context-manager factory: frees GPU memory (optimizer / fp32 masters) while the render runs
        self.first_mode = 'dup2'   # single-source (first-step) render mode, see renderer864.render_chunk
        self.features = False      # Gaussian-feature pilot: metadata = (valid, conf) + (depth, conflict, support), see renderer864.render_features

    @torch.no_grad()
    def render(self, frames, c2w_src, K, mpu, target_c2w, gauge_id, boundary, encode=True):
        """frames uint8 (n,480,864,3), c2w_src (n,4,4) and target_c2w (16,4,4) in the anchor gauge, K 864 intrinsics."""
        from renderer864 import render_chunk
        from contextlib import nullcontext
        self.da3.to('cuda')
        try:
          with (self.offload() if self.offload is not None else nullcontext()):
            rgb, valid, conf, *feat = render_chunk(self.renderer, self.da3, np.asarray(frames), np.asarray(c2w_src, np.float64), np.asarray(K, np.float64), float(mpu),
                                                   np.asarray(target_c2w, np.float64), gauge_id, boundary, first_mode=self.first_mode, features=self.features)
        finally:
            self.renderer.release(force=True)
            if self._own_da3:
                self.da3.to('cpu')
        raw = dict(rgb=rgb.float().cpu(), valid=valid.float().cpu(), conf=conf.float().cpu(), feat=feat[0].float().cpu() if feat else None)
        if not encode:
            return None, None, raw
        dev0 = next(self.vae.model.parameters()).device                                  # training passes a CPU VAE: encode on the GPU, then
        self.vae.cuda(); video = rgb.permute(1, 0, 2, 3)[None]; video = torch.cat([video[:, :, :1], video], 2)   # send it back (1.3 GB; 10-22 OOM)
        rz = VaeStream(self.vae).encode_chunk(video.cuda().to(torch.bfloat16))[:, :, 1:].float().cpu()
        if dev0.type == 'cpu':
            self.vae.cpu(); torch.cuda.empty_cache()
        h, w = rz.shape[-2:]
        meta = torch.stack([F.interpolate(raw[k].reshape(4, 4, 1, 480, 864).mean(1), size=(h, w), mode='area')[:, 0] for k in ('valid', 'conf')])[None].float()
        if raw['feat'] is not None:
            meta = torch.cat([meta, pool_features(raw['feat'], (h, w))], 1)                   # (1,5,4,h,w)
        return rz, meta, raw


class RenderBuilder864(Render864):
    """Training rows: render of the chunk-0 command from the recorded prefix (anchor + bank + last 4 prefix latent ends)."""

    def __init__(self, vae):
        super().__init__(vae); self._frames = {}

    def frame(self, stream, idx):
        key = (stream.row['stream_id'], int(idx))
        if key not in self._frames:
            if len(self._frames) > 512:
                self._frames.pop(next(iter(self._frames)))
            self._frames[key] = rec_frames(stream.row, [idx])[0]
        return self._frames[key]

    @torch.no_grad()
    def render_chunk0(self, stream, start, bank):
        boundary = 4 * (start - 1); command = stream.c2w[4 * start - 3: 4 * start + 13].numpy()
        ids = list(dict.fromkeys([0, boundary, max(0, boundary - 4), *[4 * i for i in bank[:1]]]))[:self.MAX_SOURCES]   # anchor + last two prefix ends + best bank view
        frames = np.stack([self.frame(stream, i) for i in ids]); c2w = stream.c2w[ids].numpy()
        rz, meta, _ = self.render(frames, c2w, stream.K.numpy(), stream.metres_per_unit, command, stream.gauge_id, boundary)
        return rz, meta


class OwnRender864(Render864):
    """Inference / rollouts: render of the next command from the model's own frames at their stored poses."""

    @torch.no_grad()
    def render_own(self, frames_by_latent, poses, bank, s, K, mpu, target_c2w, gauge_id):
        """frames_by_latent: dict latent index -> uint8 frame (anchor = 0, latent i = frame 4i); poses: RGB-rate list of c2w
        (anchor gauge, commanded or measured); s = index of the first new latent."""
        ids = list(dict.fromkeys([0, s - 1, max(0, s - 2), *bank[:1]]))[:self.MAX_SOURCES]
        frames = np.stack([frames_by_latent[i] for i in ids]); c2w = np.stack([np.asarray(poses[0 if i == 0 else 4 * i], np.float64) for i in ids])
        return self.render(frames, c2w, np.asarray(K, np.float64), mpu, np.asarray(target_c2w, np.float64), gauge_id, boundary=4 * (s - 1))
