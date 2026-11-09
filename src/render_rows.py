"""Stage B: ReSplat render of the chunk-0 command from the RECORDED causal prefix (training rows), reproducing
joint_v9.runtime.prepare's geometry path without the DiT: DA3 depths of anchor + bank frames + the last 12 prefix
frames (anchor depth from anchor.npz, dataset gauge), FoundationRenderer at the 16 commanded poses, VAE-encoded with
the previous recorded frame as the stream prime. No corruption (pilot). Returns render latents (1,48,4,16,28) and
metadata (valid, conf) (1,2,4,16,28)."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from all_scene_dataset import DATA, RecordedStream
from foundation_depth import causal_depth
from gaussverse.models.vae_stream import VaeStream


class RenderBuilder:
    def __init__(self, vae):
        from gaussverse.training.gate_omega_recovery import PinnedDA3Predictor
        from foundation_renderer import FoundationRenderer
        self.vae = vae; self.depth = PinnedDA3Predictor(offload_after_call=True); self.renderer = FoundationRenderer(); self._rec = {}

    def recorded(self, stream):
        sid = stream.row['stream_id']
        if sid not in self._rec:
            if len(self._rec) > 64:
                self._rec.pop(next(iter(self._rec)))
            d = DATA / 'components' / sid
            with np.load(d / 'anchor.npz') as z:
                depth = torch.from_numpy(z['depth'].copy()); mask = torch.from_numpy(z['confidence'].copy())
            self._rec[sid] = (RecordedStream(stream.row), depth, mask)
        return self._rec[sid]

    @torch.no_grad()
    def render_chunk0(self, stream, start, bank):
        rec, adepth, amask = self.recorded(stream); boundary = 4 * (start - 1); K = stream.K; gauge = stream.gauge_id
        command = stream.c2w[boundary + 1: boundary + 17]
        ids = list(dict.fromkeys([0, *[4 * i for i in bank], boundary, *range(max(0, boundary - 12), boundary, 4)]))
        frames = np.stack([rec.frame(i) for i in ids]); c2w = stream.c2w[ids]
        if len(ids) > 1:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(0); depths, masks, _ = causal_depth(self.depth, frames, K, c2w, adepth, amask)
        else:
            depths = adepth[None]; masks = amask[None]
        sources = [dict(frame_id=i, rgb=torch.from_numpy(frames[j].copy()).permute(2, 0, 1).float() / 127.5 - 1,
                        depth=adepth if i == 0 else depths[j], confidence=amask if i == 0 else masks[j], c2w=c2w[j], gauge_id=gauge) for j, i in enumerate(ids)]
        rendered = self.renderer.render(sources, command, K, (256, 448), seed=0, camera_units='anchor_median_depth', gauge_id=gauge, history_boundary=boundary)
        self.renderer.release(force=True)
        self.vae.cuda(); video = rendered['rgb'].permute(1, 0, 2, 3)[None]; video = torch.cat([video[:, :, :1], video], 2)
        rz = VaeStream(self.vae).encode_chunk(video.cuda().to(torch.bfloat16))[:, :, 1:].float().cpu()
        meta = torch.stack([F.interpolate(rendered[k].reshape(4, 4, 1, 256, 448).mean(1), size=(16, 28), mode='area')[:, 0] for k in ('valid', 'conf')])[None].float().cpu()
        return rz, meta
