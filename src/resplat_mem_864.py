"""Peak GPU memory / time of the ReSplat render path alone (no DiT): 540x960 weights at num_refine 4 and 1, 1 and 4 source
frames, 512x960 view8 weights, and the 256x448 weights at 240x432 -- decides whether stage B / the reward arm can keep the
renderer next to the resident DiT + optimizer (~17.5 GB) on the 32 GB card. Can run next to prep (11 GB)."""
import json, sys, time
import numpy as np, torch
from realized_pose import load_da3
from renderer864 import Renderer864, render_chunk
from data import corpus864, Stream864
from render864_rows import rec_frames
rows, _ = corpus864('development'); r = next(x for x in rows if x.get('mpu_valid', True)); s = Stream864(r); P = s.c2w.numpy(); K = s.K.numpy(); mpu = s.metres_per_unit
fr = rec_frames(r, [0, 4, 8, 12]); da3 = load_da3('cuda'); out = []
for shape, ck, nr in (((540, 960), None, 4), ((540, 960), None, 1), ((512, 960), None, 4), ((240, 432), (256, 448), 4)):
    ren = Renderer864(shape, num_refine=nr, ckpt_shape=ck)
    for nsrc in (1, 4):
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated() / 2 ** 30; t0 = time.time()
        rgb, valid, conf = render_chunk(ren, da3, fr[:nsrc], P[[0, 4, 8, 12][:nsrc]], K, mpu, P[1:17], s.gauge_id, boundary=0 if nsrc == 1 else 12, size=shape)
        ren.release(force=True); torch.cuda.synchronize()
        rec = dict(shape=shape, ckpt=ck or shape, num_refine=nr, sources=nsrc, peak_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), resident_before=round(base, 2), sec=round(time.time() - t0, 1), coverage=round(float((valid > .5).float().mean()), 3))
        out.append(rec); print(json.dumps(rec), flush=True)
    del ren; torch.cuda.empty_cache()
json.dump(out, open('resplat_mem_864.json', 'w'), indent=1)
