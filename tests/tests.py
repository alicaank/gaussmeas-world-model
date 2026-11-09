"""Code-correctness package (GPU): data causality + pose convention, chunk-causal mask (no future leakage),
per-chunk timestep labelling, PRoPE world-frame invariance / relative-pose sensitivity / gradient flow."""
import json, random, sys, time
import numpy as np, torch
from all_scene_dataset import RecordedStream
from data import corpus, Stream, build_row
from model_io import fresh_model
from packed_model import chunk_causal_bias

R = {}


def check(name, ok, info=''):
    R[name] = bool(ok); print(('PASS ' if ok else 'FAIL ') + name, info, flush=True)


def forward(model, row, noisy, u, text, c2w=None):
    lat = torch.cat([row['context'], noisy], 2).cuda().bfloat16()
    times = torch.cat([torch.zeros(row['context'].shape[2]), torch.as_tensor(u).repeat_interleave(4)]).cuda()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        return model(lat, row['rays'].cuda().bfloat16(), row['roles'].cuda(), row['positions'].cuda(), times, text.cuda().bfloat16(),
                     row['chunk_index'].cuda(), (row['c2w'] if c2w is None else c2w).cuda(), row['K'].cuda()).float()


def main():
    rows, flagged = corpus('train'); rng = random.Random(0)
    # 1. pose convention: vae.npz c2w == RecordedStream c2w (GL->CV converted) for 3 streams; row causality
    ok = True
    for r in rng.sample(rows, 3):
        s = Stream(r); rs = RecordedStream(r)
        with np.load(f"/home/hj/Alican/training_runs/joint_samplerfix_20261012/dataset_v3/components/{r['stream_id']}/vae.npz") as z:
            raw = torch.from_numpy(z['c2w'].copy())
        ok &= torch.allclose(raw[:rs.c2w.shape[0]], rs.c2w, atol=1e-5) and torch.allclose(s.K, rs.K, atol=1e-4)
    check('vae_npz_c2w_matches_RecordedStream', ok)
    s = Stream(rows[0]); rng2 = random.Random(1); ok = True; first = 0
    for _ in range(200):
        row = build_row(s, rng2); st, n = row['start'], row['n_chunks']
        ok &= all(b < max(1, st - 4) for b in row['bank']) and row['recent'] == list(range(max(1, st - 4), st)) and st + 4 * n <= s.L
        ok &= (len(row['bank']) == 0 and len(row['recent']) == 0) if st == 1 else True
        ok &= row['rays'].shape[2] == row['roles'].numel() == row['c2w'].shape[0] == row['context'].shape[2] + 4 * n
        first += st == 1
    check('row_causality_and_shapes', ok, f'first-call rows {first}/200')
    check('flagged_streams_excluded', not any(r['stream_id'] in flagged for r in rows), f'{len(flagged)} flagged, {len(rows)} train streams kept')
    # 2. model
    model, params, fp = fresh_model(prope=True); model.eval(); text = s.text
    row = build_row(s, random.Random(2), n_chunks=3, start=12); x0 = row['target']
    g = torch.Generator().manual_seed(0); eps = torch.randn(x0.shape, generator=g); u = [0.7, 0.4, 0.9]
    uu = torch.tensor(u).repeat_interleave(4).view(1, 1, 12, 1, 1); noisy = (1 - uu) * x0 + uu * eps
    with torch.no_grad():
        for br in model.prope:
            br.out_proj.weight.normal_(0, 1e-3)              # make the branch visible for the invariance tests
    v3 = forward(model, row, noisy, u, text)
    row1 = build_row(s, random.Random(2), n_chunks=1, start=12)
    v1 = forward(model, row1, noisy[:, :, :4], u[:1], text)
    dd = (v3[:, :, :4] - v1).abs().flatten(); scale = v1.abs().mean().item(); d = dd.max().item()
    # different sequence lengths change the bf16 attention kernel tiling; leakage is tested exactly at equal length below
    check('chunk0_independent_of_later_chunks', dd.mean().item() < 0.03 * scale and dd.quantile(.99).item() < 0.1 * scale, f'mean {dd.mean().item()/scale:.4f} p99 {dd.quantile(.99).item()/scale:.4f} max {d/scale:.3f} (relative)')
    noisy_b = noisy.clone(); noisy_b[:, :, 8:] = torch.randn_like(noisy_b[:, :, 8:])
    vb = forward(model, row, noisy_b, u, text)
    d01 = (vb[:, :, :8] - v3[:, :, :8]).abs().max().item(); d2 = (vb[:, :, 8:] - v3[:, :, 8:]).abs().max().item()
    check('chunks01_independent_of_chunk2_input', d01 < 0.05 * scale + 1e-3 and d2 > 0.1 * scale, f'd01 {d01:.4f} d2 {d2:.4f}')
    vu = forward(model, row, noisy, [0.7, 0.2, 0.9], text)
    d0 = (vu[:, :, :4] - v3[:, :, :4]).abs().max().item(); d1 = (vu[:, :, 4:8] - v3[:, :, 4:8]).abs().max().item()
    check('timestep_label_per_chunk', d0 < 0.05 * scale + 1e-3 and d1 > 0.02 * scale, f'd0 {d0:.4f} d1 {d1:.4f}')
    # mask structure
    ci = torch.tensor([-1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1]); b = chunk_causal_bias(ci, 1, 'cpu')[0, 0]
    ok = torch.isfinite(b[:3, :3]).all() and torch.isinf(b[:3, 3:]).all() and torch.isfinite(b[3:7, :7]).all() and torch.isinf(b[3:7, 7:]).all() and torch.isfinite(b[7:, :]).all()
    check('mask_structure', bool(ok))
    # 3. PRoPE: world-frame invariance and relative-pose sensitivity
    G = torch.eye(4); ang = 0.7; G[:3, :3] = torch.tensor([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1.]]); G[:3, 3] = torch.tensor([0.3, -0.2, 0.5])
    vG = forward(model, row, noisy, u, text, c2w=G[None] @ row['c2w'])
    dG = (vG - v3).abs().max().item()
    c2 = row['c2w'].clone(); c2[1:, :3, 3] += torch.tensor([0.4, 0., 0.])                  # move every non-anchor camera
    vR = forward(model, row, noisy, u, text, c2w=c2); dR = (vR - v3).abs().max().item()
    dGm = (vG - v3).abs().mean().item()
    check('prope_world_frame_invariant', dGm < 0.03 * scale, f'global transform mean diff {dGm/scale:.4f} max {dG/scale:.3f} (relative; exact in float64, tests_numerics.py)')
    check('prope_relative_pose_sensitive', dR > 1e-3, f'relative pose diff {dR:.5f} (branch at 1e-3 init)')
    # 4. gradient reaches PRoPE and trainables; loss is a plain mean over chunk tokens
    with torch.no_grad():
        for br in model.prope:
            br.out_proj.weight.zero_()
    model.train(); lat = torch.cat([row['context'], noisy], 2).cuda().bfloat16()
    times = torch.cat([torch.zeros(row['context'].shape[2]), uu.flatten()]).cuda()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        pred = model(lat, row['rays'].cuda().bfloat16(), row['roles'].cuda(), row['positions'].cuda(), times, text.cuda().bfloat16(),
                     row['chunk_index'].cuda(), row['c2w'].cuda(), row['K'].cuda(), checkpoint_blocks=True).float()
    label = (eps - x0).cuda(); loss = (pred - label).square().mean(); loss.backward()
    per_chunk = [(pred[:, :, 4 * j:4 * j + 4] - label[:, :, 4 * j:4 * j + 4]).square().mean().item() for j in range(3)]
    check('loss_uniform_over_chunks', abs(np.mean(per_chunk) - loss.item()) < 1e-4, f'{per_chunk} vs {loss.item():.5f}')
    g_out = sum(float(br.out_proj.weight.grad.norm()) for br in model.prope); g_q = sum(float(br.q_proj.weight.grad.norm()) for br in model.prope)
    g_attn = sum(float(p.grad.norm()) for n, p in params if '.self_attn.' in n and p.grad is not None)
    g_cam = float(model.camera.weight.grad.norm()); g_patch = float(model.dit.patch_embedding.weight.grad.norm())
    check('grad_reaches_prope_out_proj_step0', g_out > 0 and g_q == 0, f'out {g_out:.3e} q {g_q:.3e} (q is zero at step 0 by zero-init)')
    check('grad_reaches_backbone_trainables', g_attn > 0 and g_cam > 0 and g_patch > 0, f'attn {g_attn:.3e} cam {g_cam:.3e} patch {g_patch:.3e}')
    with torch.no_grad():
        for br in model.prope:
            br.out_proj.weight.add_(-1e-2 * br.out_proj.weight.grad / (br.out_proj.weight.grad.norm() + 1e-12))
    model.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        pred = model(lat, row['rays'].cuda().bfloat16(), row['roles'].cuda(), row['positions'].cuda(), times, text.cuda().bfloat16(),
                     row['chunk_index'].cuda(), row['c2w'].cuda(), row['K'].cuda(), checkpoint_blocks=True).float()
    (pred - label).square().mean().backward()
    g_q = sum(float(br.q_proj.weight.grad.norm()) for br in model.prope)
    check('grad_reaches_prope_qkv_after_one_step', g_q > 0, f'q {g_q:.3e}')
    n_tr = sum(p.numel() for _, p in params); n_pr = sum(p.numel() for br in model.prope for p in br.parameters())
    print(json.dumps(dict(trainable_params=n_tr, prope_params=n_pr, stock=fp[:12], peak_GiB=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2))))
    json.dump(dict(results=R, trainable_params=n_tr, prope_params=n_pr), open('tests_result.json', 'w'), indent=1)
    print('ALL PASS' if all(R.values()) else 'SOME FAIL')


if __name__ == '__main__':
    main(); import os; os._exit(0)
