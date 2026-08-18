from __future__ import annotations

import pathlib
import sys
import unittest

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gift_chroma_master.core.spill import linear_luminance, suppress_spill_v5  # noqa: E402


class ChromaSpillTests(unittest.TestCase):
    def setUp(self):
        self.alpha = torch.full((1, 1, 16, 16), 0.5)
        self.screen = torch.tensor([0.02, 0.80, 0.02]).view(1, 3, 1, 1)

    def test_zero_amount_is_exact_identity(self):
        image = torch.rand(1, 3, 16, 16)
        output, spill_map, _ = suppress_spill_v5(image, self.alpha, self.screen, amount=0.0)
        self.assertIs(output, image)
        self.assertEqual(float(spill_map.max()), 0.0)

    def test_linear_luminance_is_preserved(self):
        image = torch.zeros(1, 3, 16, 16)
        image[:, 0] = 0.22
        image[:, 1] = 0.44
        image[:, 2] = 0.18
        output, _, _ = suppress_spill_v5(
            image, self.alpha, self.screen,
            amount=1.0, range=0.8, spill=1.0, luma_restore=1.0,
            skin_protection=0.0, key_color_protection=0.0,
        )
        error = (linear_luminance(output) - linear_luminance(image)).abs().max()
        self.assertLess(float(error), 1.0e-5)

    def test_opponent_screen_projection_is_reduced(self):
        image = torch.zeros(1, 3, 16, 16)
        image[:, 0] = 0.18
        image[:, 1] = 0.55
        image[:, 2] = 0.12
        output, spill_map, _ = suppress_spill_v5(
            image, self.alpha, self.screen,
            amount=1.0, range=0.8, spill=1.0,
            skin_protection=0.0, highlight_protection=0.0, key_color_protection=0.0,
        )
        before = image[:, 1:2] - 0.5 * (image[:, 0:1] + image[:, 2:3])
        after = output[:, 1:2] - 0.5 * (output[:, 0:1] + output[:, 2:3])
        self.assertGreater(float(spill_map.mean()), 0.0)
        self.assertLess(float(after.mean()), float(before.mean()))


if __name__ == "__main__":
    unittest.main()
