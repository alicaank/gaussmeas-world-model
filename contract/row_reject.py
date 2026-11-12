"""Outlier-row rejection for gradient accumulation (critique 12 Nov 09:00). Every run of the trainer has 3-10% of updates where ONE row's gradient has a
pre-clip norm of 1e2-1e6 (median update norm .16): global clipping then scales the whole accumulated gradient, so that row alone sets the direction of a
unit-norm step and the other 15 rows are erased. Here such a row never enters the update.

The trainer accumulates into .grad (a second copy of the 1.3e9 trainables does not fit next to the 24.6 GB peak), so a row is judged AFTER its backward:
  row norm estimate  r = sqrt(max(|acc_after|^2 - |acc_before|^2, 0))     (a row is near-orthogonal to the sum of the others in this dimension)
  reject iff r > factor x running median of the accepted rows' r (last 200; fallback .1 until 32 rows are seen)
On a rejection the accumulated gradient is cleared and the ACCEPTED rows of this update are replayed with their saved generator states (same u / noise /
text-dropout draws), then the generator is put back where it was -> the random stream of the run is unchanged by a rejection (paired arms stay paired).
Rare (a few % of updates), costs one partial update of compute. The update then averages 15 rows with the same 1/accum scale. Logged per row: 'grow' (the
estimate) and 'rejected'. Usage: r = RowReject(factor, accum, step); r.before(model, g) ... r.after(row, rep, gstate, args, kwargs); r.finish() before clipping."""
import numpy as np
import torch


def grad_norm(model):
    with torch.no_grad():
        s = [(p.grad.float() ** 2).sum() for p in model.parameters() if p.requires_grad and p.grad is not None]
        return float(torch.stack(s).sum() ** .5) if s else 0.


class RowReject:
    def __init__(self, factor, accum, step, window=200, warm=32, fallback=.1):
        self.factor, self.accum, self.step, self.window, self.warm, self.fallback = factor, accum, step, window, warm, fallback
        self.rows = []; self.prev = 0.; self.hist = []; self.rejected = 0; self.model = None; self.g = None

    def median(self):
        return float(np.median(self.hist[-self.window:])) if len(self.hist) >= self.warm else max(self.fallback, float(np.median(self.hist)) if self.hist else 0.)

    def settle(self):
        """Judge the row whose backward ran last (if any)."""
        if not self.rows or self.rows[-1].get('settled'): return
        last = self.rows[-1]; last['settled'] = True; b = grad_norm(self.model); r = max(b * b - self.prev ** 2, 0.) ** .5; last['rep']['grow'] = round(r, 4)
        if self.factor > 0 and r > self.factor * self.median():
            last['rep']['rejected'] = True; self.rejected += 1; self.rows.pop(); state = self.g.get_state()
            for p in self.model.parameters(): p.grad = None
            for it in self.rows:
                self.g.set_state(it['g']); loss, _ = self.step(self.model, it['row'], self.g, *it['args'], **it['kwargs']); (loss / self.accum).backward()
            self.g.set_state(state); self.prev = grad_norm(self.model)
        else:
            self.hist.append(r); self.prev = b

    def before(self, model, g):
        self.model, self.g = model, g; self.settle(); return g.get_state()

    def after(self, row, rep, gstate, args, kwargs):
        self.rows.append(dict(row=row, rep=rep, g=gstate, args=args, kwargs=kwargs))

    def finish(self):
        """Before clipping / the optimizer step: judge the last row, then start a new update."""
        self.settle(); self.rows = []; self.prev = 0.
