"""(a) float64 CPU: PRoPE branch is exactly invariant to a global rigid transform of all cameras and sensitive to relative
pose; (b) GPU (optional, --gpu): distribution of the chunk-0 velocity difference between an N=1 and an N=3 row, and under a
change of chunk-1 input at equal length (bf16 chaos vs real leakage)."""
import argparse, json, random
import numpy as np, torch
from prope import CameraBranch, CameraTransforms
from packed_model import chunk_causal_bias

ap = argparse.ArgumentParser(); ap.add_argument('--gpu', type=int, default=0); a = ap.parse_args()
torch.manual_seed(0); dim, heads, F, h, w = 128, 4, 5, 4, 7
br = CameraBranch(dim, heads, compress=2).double()
with torch.no_grad():
    br.out_proj.weight.normal_(0, .1)
x = torch.randn(1, F * h * w, dim, dtype=torch.float64)
c2w = torch.eye(4).repeat(F, 1, 1).double()
for i in range(1, F):
    ang = .3 * i; c2w[i, :3, :3] = torch.tensor([[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]]); c2w[i, :3, 3] = torch.tensor([.2 * i, -.1 * i, .4 * i])
K = torch.tensor([[400., 0, 224.], [0, 400., 128.], [0, 0, 1]]).double()
ci = torch.tensor([-1, -1, 0, 0, 1]); bias = chunk_causal_bias(ci, h * w, 'cpu', torch.float64)
cam = lambda c: CameraTransforms(c[None], K, br.head_dim, w, h, 448, 256, dtype=torch.float64)
y = br(x, cam(c2w), bias)
G = torch.eye(4).double(); ang = 1.1; G[:3, :3] = torch.tensor([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]]); G[:3, 3] = torch.tensor([1., -2., .5])
yG = br(x, cam(G[None] @ c2w), bias); c2 = c2w.clone(); c2[2:, :3, 3] += .3
yR = br(x, cam(c2), bias)
r = dict(global_transform_maxdiff=float((yG - y).abs().max()), relative_pose_maxdiff=float((yR - y).abs().max()), out_scale=float(y.abs().mean()))
r['prope_math_invariant'] = r['global_transform_maxdiff'] < 1e-6 and r['relative_pose_maxdiff'] > 1e-3
print(json.dumps(r))
if a.gpu:
    from data import corpus, Stream, build_row
    from model_io import fresh_model
    rows, _ = corpus('train'); s = Stream(rows[0]); model, params, fp = fresh_model(prope=True); model.eval()
    with torch.no_grad():
        for b_ in model.prope:
            b_.out_proj.weight.normal_(0, 1e-3)

    def fwd(row, noisy, u, c2w=None):
        lat = torch.cat([row['context'], noisy], 2).cuda().bfloat16(); times = torch.cat([torch.zeros(row['context'].shape[2]), torch.as_tensor(u).repeat_interleave(4)]).cuda()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            return model(lat, row['rays'].cuda().bfloat16(), row['roles'].cuda(), row['positions'].cuda(), times, s.text.cuda().bfloat16(), row['chunk_index'].cuda(),
                         (row['c2w'] if c2w is None else c2w).cuda(), row['K'].cuda()).float()
    row = build_row(s, random.Random(2), n_chunks=3, start=12); x0 = row['target']; g = torch.Generator().manual_seed(0); eps = torch.randn(x0.shape, generator=g); u = [.7, .4, .9]
    uu = torch.tensor(u).repeat_interleave(4).view(1, 1, 12, 1, 1); noisy = (1 - uu) * x0 + uu * eps
    v3 = fwd(row, noisy, u); row1 = build_row(s, random.Random(2), n_chunks=1, start=12); v1 = fwd(row1, noisy[:, :, :4], u[:1])
    nb = noisy.clone(); nb[:, :, 4:8] = torch.randn_like(nb[:, :, 4:8]); vb = fwd(row, nb, u)
    v3b = fwd(row, noisy, u)                                                   # repeat: determinism
    G = torch.eye(4); G[:3, 3] = torch.tensor([.3, -.2, .5]); vG = fwd(row, noisy, u, c2w=G[None] @ row['c2w'])
    def dist(d, ref):
        d = d.abs().flatten(); ref = ref.abs().mean().item()
        return dict(mean=round(d.mean().item() / ref, 4), p50=round(d.median().item() / ref, 4), p99=round(d.quantile(.99).item() / ref, 4), max=round(d.max().item() / ref, 4))
    out = dict(repeat_same_input=dist(v3b - v3, v3), chunk0_N1_vs_N3=dist(v3[:, :, :4] - v1, v1), chunk0_under_chunk1_change=dist(vb[:, :, :4] - v3[:, :, :4], v3),
               chunk1_under_chunk1_change=dist(vb[:, :, 4:8] - v3[:, :, 4:8], v3), global_transform=dist(vG - v3, v3))
    print(json.dumps(out)); json.dump(dict(cpu=r, gpu=out), open('tests_numerics_result.json', 'w'), indent=1)
import os; os._exit(0)
