"""CPU tests for the packed-row corruption (gm_b_corruption.py): scope, rates, exact one-step recovery, determinism."""
import random
import unittest
import torch
from gm_b_corruption import GMBConfig, corrupt_packed, noise_slots


def packed_row(bank=(2, 3), recent=(5, 6, 7, 8), n_chunks=2, render=True, sid='abc', start=9, seed=0):
    g = torch.Generator().manual_seed(seed); nctx = 1 + len(bank) + len(recent); h, w = 2, 3
    ctx = torch.randn(1, 48, nctx + (4 if render else 0), h, w, generator=g); tgt = torch.randn(1, 48, 4 * n_chunks, h, w, generator=g)
    md = None
    if render:
        md = torch.zeros(1, 2, nctx + 4 + 4 * n_chunks, h, w); md[:, :, nctx:nctx + 4] = 1.0
    return dict(context=ctx, target=tgt, metadata=md, render=render, stream_id=sid, start=start, bank=list(bank), recent=list(recent), n_chunks=n_chunks)


class Rng:
    """random() returns the scripted values; uniform() the midpoint; getrandbits a constant."""
    def __init__(self, draws, bits=12345): self.draws = list(draws); self.bits = bits
    def random(self): return self.draws.pop(0)
    def uniform(self, a, b): return (a + b) / 2
    def getrandbits(self, n): return self.bits


def exact_velocity_from(x0_recent):
    def velocity(model, aux, lat, times):
        xs = lat[:, :, -4:]; s = float(times[-1]); eps = (xs - (1 - s) * x0_recent) / s
        return eps - x0_recent
    return velocity


def aux_from(row):
    def build(stream, start, max_bank):
        nctx = 1 + len(row['bank']) + len(row['recent'])
        return dict(target=row['context'][:, :, nctx - 4:nctx].clone(), context=row['context'][:, :, :1].clone(), rays=None, roles=None, positions=None, text=None, chunk_index=None, c2w=None, K=None)
    return build


class Corruption(unittest.TestCase):
    cfg = GMBConfig()

    def test_onpolicy_and_first_call_rows_pass_through(self):
        row = packed_row(sid='OPmabc'); out, rec = corrupt_packed(row, None, self.cfg, Rng([0.0, 0.0]))
        self.assertIs(out, row); self.assertEqual(rec['op'], 'onpolicy')
        row = packed_row(bank=(), recent=(0,), start=1); out, rec = corrupt_packed(row, None, self.cfg, Rng([0.0, 0.0]))
        self.assertIs(out, row); self.assertEqual(rec['op'], 'none')

    def test_noise_branch_touches_bank_recent_and_render_only(self):
        row = packed_row(); out, rec = corrupt_packed(row, object(), self.cfg, Rng([0.9, 0.1]))     # skip self-aug, take noise
        self.assertEqual(rec['op'], 'noise'); s = rec['s']; self.assertAlmostEqual(s, 0.15); self.assertAlmostEqual(rec['render_s'], s)
        nctx = 7
        self.assertTrue(torch.equal(out['context'][:, :, 0], row['context'][:, :, 0]))                 # anchor
        self.assertTrue(torch.equal(out['target'], row['target']))                                     # targets / labels
        for k in range(1, nctx + 4): self.assertFalse(torch.equal(out['context'][:, :, k], row['context'][:, :, k]))
        self.assertTrue(torch.allclose(out['metadata'][:, :, nctx:nctx + 4], torch.full((1, 2, 4, 2, 3), 1 - s)))
        self.assertTrue(torch.equal(out['metadata'][:, :, :nctx], row['metadata'][:, :, :nctx])); self.assertTrue(torch.equal(out['metadata'][:, :, nctx + 4:], row['metadata'][:, :, nctx + 4:]))
        self.assertTrue(torch.equal(row['metadata'][:, :, nctx:nctx + 4], torch.ones(1, 2, 4, 2, 3)))  # input row untouched

    def test_noise_law_statistics(self):
        x = torch.zeros(1, 48, 6, 8, 8); y = noise_slots(x, [1, 2], 0.3, torch.Generator().manual_seed(0))
        self.assertAlmostEqual(float(y[:, :, [1, 2]].std()), 0.3, places=2); self.assertTrue(torch.equal(y[:, :, [0, 3, 4, 5]], x[:, :, [0, 3, 4, 5]]))

    def test_self_aug_with_exact_velocity_recovers_recent_and_noises_bank(self):
        row = packed_row(); nctx = 7; clean_recent = row['context'][:, :, 3:7].clone()
        out, rec = corrupt_packed(row, object(), self.cfg, Rng([0.1]), model=None, aux_builder=aux_from(row), velocity=exact_velocity_from(clean_recent))
        self.assertEqual(rec['op'], 'self_aug'); self.assertAlmostEqual(rec['s'], 0.275)
        self.assertTrue(torch.allclose(out['context'][:, :, 3:7], clean_recent, atol=1e-5))         # exact velocity -> exact x0
        for k in (1, 2): self.assertFalse(torch.equal(out['context'][:, :, k], row['context'][:, :, k]))  # bank noised at s
        self.assertTrue(torch.equal(out['context'][:, :, 0], row['context'][:, :, 0])); self.assertTrue(torch.equal(out['target'], row['target']))
        for k in range(nctx, nctx + 4): self.assertFalse(torch.equal(out['context'][:, :, k], row['context'][:, :, k]))  # render follows

    def test_self_aug_falls_back_to_noise_with_short_recent(self):
        row = packed_row(bank=(), recent=(1, 2), start=3); out, rec = corrupt_packed(row, object(), self.cfg, Rng([0.1]))
        self.assertEqual(rec['op'], 'self_aug_fallback_noise'); self.assertAlmostEqual(rec['s'], 0.275)
        for k in (1, 2): self.assertFalse(torch.equal(out['context'][:, :, k], row['context'][:, :, k]))
        self.assertTrue(torch.equal(out['context'][:, :, 0], row['context'][:, :, 0]))

    def test_auxiliary_row_mismatch_is_rejected(self):
        row = packed_row(); bad = aux_from(packed_row(seed=1))
        with self.assertRaises(RuntimeError):
            corrupt_packed(row, object(), self.cfg, Rng([0.1]), aux_builder=bad, velocity=exact_velocity_from(row['context'][:, :, 3:7]))

    def test_deterministic_per_seed(self):
        row = packed_row(); kw = dict(aux_builder=aux_from(row), velocity=exact_velocity_from(row['context'][:, :, 3:7]))
        a, _ = corrupt_packed(row, object(), self.cfg, random.Random('gmb:1:5'), **kw); b, _ = corrupt_packed(row, object(), self.cfg, random.Random('gmb:1:5'), **kw)
        c, _ = corrupt_packed(row, object(), self.cfg, random.Random('gmb:1:6'), **kw)
        self.assertTrue(torch.equal(a['context'], b['context'])); self.assertFalse(torch.equal(a['context'], c['context']))

    def test_rates(self):
        row = packed_row(bank=(), recent=(1, 2), start=3); ops = {}
        for i in range(3000):
            _, rec = corrupt_packed(row, object(), self.cfg, random.Random(i)); ops[rec['op']] = ops.get(rec['op'], 0) + 1
        self.assertAlmostEqual(ops['self_aug_fallback_noise'] / 3000, .5, delta=.03); self.assertAlmostEqual(ops['noise'] / 3000, .15, delta=.03)
        self.assertAlmostEqual(ops.get('none', 0) / 3000, .35, delta=.03)


if __name__ == '__main__':
    unittest.main()
