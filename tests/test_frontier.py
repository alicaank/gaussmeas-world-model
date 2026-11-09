"""CPU tests for the frontier mask: closing, masking arithmetic, and observed-support geometry on a synthetic camera pair."""
import sys
import unittest
from pathlib import Path
import numpy as np
import torch

FT = '/home/hj/Alican/analysis/plan_base_restart_20261008/fresh_teacher_20261021'
for p in (str(Path(__file__).resolve().parent), FT):
    if p not in sys.path:
        sys.path.insert(0, p)
import frontier


class Frontier(unittest.TestCase):
    def test_closing_fills_point_gaps_and_keeps_empty_regions(self):
        m = torch.zeros(1, 9, 9); m[0, 2:7:2, 2:7:2] = 1        # sparse points in a 5x5 block, empty border
        c = frontier.closing(m, 1)
        self.assertEqual(float(c[0, 2:7, 2:7].mean()), 1.0); self.assertEqual(float(c[0, :1].sum() + c[0, -1:].sum()), 0.0)

    def test_apply_mask_zeroes_unobserved_only(self):
        rgb = torch.ones(2, 3, 4, 6); valid = torch.ones(2, 4, 6); conf = torch.full((2, 4, 6), .7); obs = torch.ones(2, 4, 6); obs[:, :, 4:] = 0
        r, v, c, fr = frontier.apply_mask(rgb, valid, conf, obs)
        self.assertTrue(torch.equal(r, rgb)); self.assertEqual(float(v[:, :, :4].mean()), 1.0); self.assertEqual(float(v[:, :, 4:].sum() + c[:, :, 4:].sum()), 0.0)
        self.assertAlmostEqual(fr, 2 / 6)
        r2, *_ = frontier.apply_mask(rgb, valid, conf, obs, zero_rgb=True); self.assertEqual(float(r2[:, :, :, 4:].abs().sum()), 0.0)
        cells = frontier.pooled_cells(torch.ones(16, 480, 864)); self.assertEqual(tuple(cells.shape), (4, 30, 54)); self.assertEqual(float(cells.mean()), 1.0)

    def test_observed_support_marks_revealed_side_unobserved(self):
        H, W = 60, 108; K = np.array([[80., 0, W / 2], [0, 80., H / 2], [0, 0, 1]]); depth = [torch.full((H, W), 4.0)]
        src = np.eye(4)[None]; same = np.eye(4)[None]
        shifted = np.eye(4)[None].copy(); shifted[0, 0, 3] = 3.0                                   # camera moves 3 units right: new space on the right
        obs_same = frontier.observed_from_support(depth, K, src, same, size=(15, 27), close=1, out_size=(H, W))
        obs_shift = frontier.observed_from_support(depth, K, src, shifted, size=(15, 27), close=1, out_size=(H, W))
        self.assertGreater(float(obs_same.mean()), .95)
        self.assertGreater(float(obs_shift[0, :, :W // 4].mean()), .9); self.assertLess(float(obs_shift[0, :, -W // 4:].mean()), .1)


class Reprojection(unittest.TestCase):
    H, W = 60, 108; K = np.array([[80., 0, 54.], [0, 80., 30.], [0, 0, 1]])

    def test_same_camera_same_depth_is_fully_observed(self):
        plane = torch.full((1, self.H, self.W), 4.0)
        obs = frontier.observed_from_reprojection(plane, self.K, np.eye(4)[None], [plane[0]], np.eye(4)[None], out_size=(self.H, self.W))
        self.assertGreater(float(obs[0, 3:-3, 3:-3].mean()), .99)                                   # margin strips excluded

    def test_revealed_side_and_depth_disagreement_are_unobserved(self):
        plane = torch.full((1, self.H, self.W), 4.0); T = np.eye(4)[None].copy(); T[0, 0, 3] = 3.0      # target moved right by 3 units
        obs = frontier.observed_from_reprojection(plane, self.K, T, [plane[0]], np.eye(4)[None], out_size=(self.H, self.W))
        self.assertGreater(float(obs[0, :, 5:self.W // 4].mean()), .9); self.assertLess(float(obs[0, :, -self.W // 4:].mean()), .05)
        near = torch.full((1, self.H, self.W), 2.0)                                                  # surface claimed in front of what the source saw
        obs2 = frontier.observed_from_reprojection(near, self.K, np.eye(4)[None], [plane[0]], np.eye(4)[None], out_size=(self.H, self.W))
        self.assertLess(float(obs2.mean()), .01)
        obs3 = frontier.observed_from_reprojection(plane * 1.08, self.K, np.eye(4)[None], [plane[0]], np.eye(4)[None], tol=.15, out_size=(self.H, self.W))
        self.assertGreater(float(obs3[0, 3:-3, 3:-3].mean()), .99)                                     # 8% depth noise within tolerance

    def test_invalid_target_depth_is_unobserved(self):
        plane = torch.full((1, self.H, self.W), 4.0); plane[0, :, : self.W // 2] = 0
        obs = frontier.observed_from_reprojection(plane, self.K, np.eye(4)[None], [torch.full((self.H, self.W), 4.0)], np.eye(4)[None], out_size=(self.H, self.W))
        self.assertEqual(float(obs[0, :, : self.W // 2].sum()), 0.0); self.assertGreater(float(obs[0, 3:-3, self.W // 2 + 3: -3].mean()), .99)


if __name__ == '__main__':
    unittest.main()
