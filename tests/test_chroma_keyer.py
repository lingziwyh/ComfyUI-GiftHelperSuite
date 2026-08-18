from __future__ import annotations

import pathlib
import sys
import unittest

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gift_chroma_master.core.keyer import (  # noqa: E402
    estimate_screen_color,
    finish_matte,
    run_v5_keyer,
)


class ChromaKeyerTests(unittest.TestCase):
    def test_manual_arbitrary_keys_do_not_fall_back_to_a_primary_channel(self):
        key_colors = (
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 0.35, 0.0),
        )
        for color in key_colors:
            with self.subTest(color=color):
                key = torch.tensor(color).view(1, 3, 1, 1)
                image = key.expand(1, 3, 12, 13).contiguous()
                result = run_v5_keyer(
                    image,
                    screen_mode="manual",
                    manual_screen_color=color,
                    screen_balance=0.5,
                )
                self.assertLess(float(result["alpha"].max()), 1.0e-5)

    def test_foreground_color_unrelated_to_yellow_screen_stays_solid(self):
        image = torch.zeros(1, 3, 12, 13)
        image[:, 0] = 1.0
        result = run_v5_keyer(
            image,
            screen_mode="manual",
            manual_screen_color=(1.0, 1.0, 0.0),
        )
        self.assertGreater(float(result["alpha"].min()), 0.999)

    def test_constant_matte_does_not_darkening_at_image_border(self):
        alpha = torch.ones(1, 1, 3, 4)
        result = finish_matte(alpha, shrink_grow_px=-2.5, softness_px=3.0)
        self.assertGreater(float(result.min()), 0.999)

    def test_smoothed_screen_sampling_resets_on_screen_colour_cut(self):
        image = torch.zeros(6, 3, 20, 24)
        image[:3, 1] = 1.0
        image[3:, 2] = 1.0
        sampled, _ = estimate_screen_color(
            image,
            screen_mode="auto",
            temporal_mode="per_frame_smoothed",
            temporal_smoothing=0.82,
        )
        first_blue = sampled[3, :, 0, 0]
        self.assertGreater(float(first_blue[2]), 0.99)
        self.assertLess(float(first_blue[1]), 0.01)

    def test_smoothed_screen_sampling_tracks_large_exposure_change(self):
        image = torch.zeros(6, 3, 20, 24)
        image[:3, 1] = 0.90
        image[3:, 1] = 0.45
        sampled, _ = estimate_screen_color(
            image,
            screen_mode="green",
            temporal_mode="per_frame_smoothed",
            temporal_smoothing=0.82,
        )
        self.assertLess(abs(float(sampled[3, 1, 0, 0]) - 0.45), 0.15)


if __name__ == "__main__":
    unittest.main()
