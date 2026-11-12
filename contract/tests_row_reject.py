"""CPU test of row_reject.RowReject on a tiny model: a spiking row is removed exactly, the generator stream is unchanged. Usage: tests_row_reject.py"""
import torch
from torch import nn
from row_reject import RowReject, grad_norm


def step(model, row, g, scale=1.):
    eps = torch.randn(row['x'].shape, generator=g); return ((model(row['x'] + .1 * eps) - row['y']) ** 2).mean() * row['k'] * scale, {}


def run(factor, spike_at, accum=8, updates=6):
    torch.manual_seed(0); model = nn.Linear(16, 4); g = torch.Generator().manual_seed(5); rr = RowReject(factor, accum, step, warm=4); d = torch.Generator().manual_seed(9); out = []
    for u in range(updates):
        reps = []
        for k in range(accum):
            row = dict(x=torch.randn(32, 16, generator=d), y=torch.randn(32, 4, generator=d), k=1e5 if (u, k) in spike_at else 1.)
            gs = rr.before(model, g); loss, rep = step(model, row, g); rr.after(row, rep, gs, (), {}); (loss / accum).backward(); reps.append(rep)
        rr.finish(); out.append((torch.cat([p.grad.flatten() for p in model.parameters()]).clone(), reps, g.get_state().clone()))
        for p in model.parameters(): p.grad = None
    return out, rr


spikes = {(2, 3), (4, 7), (5, 0)}; a, ra = run(50., spikes); b, rb = run(0., set())
assert ra.rejected == 3 and rb.rejected == 0, ra.rejected
for u, ((ga, repa, sa), (gb, repb, sb)) in enumerate(zip(a, b)):
    assert torch.equal(sa, sb), 'generator stream changed'                                             # same random stream with / without rejections
    rej = [k for k, r in enumerate(repa) if r.get('rejected')]; assert rej == sorted(k for (uu, k) in spikes if uu == u), (u, rej)
    assert all('grow' in r for r in repa)
# a rejected update == the same update accumulated without that row (rebuilt by hand from the clean run's rows)
torch.manual_seed(0); model = nn.Linear(16, 4); g = torch.Generator().manual_seed(5); d = torch.Generator().manual_seed(9)
for u in range(6):
    for k in range(8):
        row = dict(x=torch.randn(32, 16, generator=d), y=torch.randn(32, 4, generator=d), k=1.); loss, _ = step(model, row, g)
        if (u, k) not in spikes: (loss / 8).backward()
    ref = torch.cat([p.grad.flatten() for p in model.parameters()]); assert torch.allclose(ref, a[u][0], atol=1e-6), u
    for p in model.parameters(): p.grad = None
print('row_reject tests passed; clean-run norms', [round(float(x[0].norm()), 3) for x in b][:3])
