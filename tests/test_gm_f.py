"""CPU tests for the GM-F training hook (train_gm_f.frontier_row / strip_mask): scope, strip geometry, determinism, pass-through."""
import random
import sys
import unittest
from pathlib import Path
import torch

sys.argv = [sys.argv[0]]                      # train_gm_f pops its options at import; keep the test runner's argv clean
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
import importlib.util
spec = importlib.util.spec_from_file_location('gmf_parts', HERE / 'train_gm_f.py')
src = (HERE / 'train_gm_f.py').read_text()
# load only the pure functions (the module's imports need the training corpus); exec the defs of strip_mask / frontier_row
ns = {'torch': torch, 'random': random, 'np': __import__('numpy'), 'Path': Path, 'P_STRIP': 0.3}
start = src.index('def strip_mask'); end = src.index('def draw(')
exec(src[start:end], ns)


def row(render=True, sid='abc', bank=(2, 3), recent=(5, 6, 7, 8), n_chunks=2):
    nctx = 1 + len(bank) + len(recent); F_ = nctx + (4 if render else 0) + 4 * n_chunks; md = None
    if render:
        md = torch.zeros(1, 2, F_, 30, 54); md[:, 0, nctx:nctx + 4] = 1.0; md[:, 1, nctx:nctx + 4] = 0.8
    return dict(stream_id=sid, render=render, metadata=md, bank=list(bank), recent=list(recent), start=9)


class GMF(unittest.TestCase):
    def test_strip_is_contiguous_side_band(self):
        m, side, frac = ns['strip_mask'](30, 54, random.Random(3))
        self.assertIn(side, ('left', 'right', 'top', 'bottom')); self.assertTrue(0.2 <= frac <= 0.5)
        zero = (m == 0); rows = zero.any(1).nonzero().flatten(); cols = zero.any(0).nonzero().flatten()
        if side in ('left', 'right'): self.assertEqual(len(rows), 30); self.assertEqual(len(cols), int(cols.max() - cols.min()) + 1)
        else: self.assertEqual(len(cols), 54); self.assertEqual(len(rows), int(rows.max() - rows.min()) + 1)

    def test_mask_touches_render_slots_only_and_strip_applies(self):
        r = row(); nctx = 7
        out, rec = ns['frontier_row'](r, random.Random(0), p_strip=1.0, cache=False)      # cache=False -> all-ones observed mask, strip forced
        self.assertEqual(rec['frontier'], 'mask'); self.assertIn('strip', rec)
        md = out['metadata']; self.assertTrue(torch.equal(md[:, :, :nctx], r['metadata'][:, :, :nctx])); self.assertTrue(torch.equal(md[:, :, nctx + 4:], r['metadata'][:, :, nctx + 4:]))
        v = md[0, 0, nctx:nctx + 4]; self.assertGreater(float((v == 0).float().mean()), 0.15); self.assertLess(float((v == 0).float().mean()), 0.55)
        self.assertTrue(torch.equal(md[0, 1, nctx:nctx + 4] == 0, v == 0))                   # conf zeroed on the same cells
        self.assertTrue(torch.equal(r['metadata'][0, 0, nctx:nctx + 4], torch.ones(4, 30, 54)))  # input untouched

    def test_no_strip_leaves_metadata_unchanged_with_all_observed(self):
        r = row(); out, rec = ns['frontier_row'](r, random.Random(0), p_strip=0.0, cache=False)
        self.assertTrue(torch.equal(out['metadata'], r['metadata'])); self.assertNotIn('strip', rec)

    def test_pass_through_for_pool_and_no_render_rows(self):
        r = row(sid='OPmabc'); out, rec = ns['frontier_row'](r, random.Random(0), cache=False); self.assertIs(out, r); self.assertEqual(rec['frontier'], 'none')
        r = row(render=False); out, rec = ns['frontier_row'](r, random.Random(0), cache=False); self.assertIs(out, r)

    def test_deterministic(self):
        r = row(); a, _ = ns['frontier_row'](r, random.Random('gmf:1:5'), p_strip=1.0, cache=False); b, _ = ns['frontier_row'](r, random.Random('gmf:1:5'), p_strip=1.0, cache=False)
        self.assertTrue(torch.equal(a['metadata'], b['metadata']))


if __name__ == '__main__':
    unittest.main()
