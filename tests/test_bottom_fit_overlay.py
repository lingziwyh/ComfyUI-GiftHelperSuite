from __future__ import annotations

import pathlib
import sys
import unittest

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fast_bottom_fit_overlay import FastBottomFitOverlay  # noqa: E402


class FastBottomFitOverlayTests(unittest.TestCase):
    def test_new_controls_are_optional_for_legacy_api_workflows(self):
        input_types = FastBottomFitOverlay.INPUT_TYPES()
        optional = input_types["optional"]

        self.assertIn("packed_size_mode", optional)
        self.assertIn("enable_rounded_rect_fade", optional)
        self.assertIn("rounded_rect_fade_ratio", optional)
        self.assertIn("rounded_corner_radius", optional)

    def test_rounded_rect_follows_layer_aspect_ratio(self):
        background = torch.zeros(1, 80, 40, 3)
        foreground = torch.ones(1, 80, 40, 3)

        image, mask, packed = FastBottomFitOverlay().composite(
            background,
            foreground,
            enable_rounded_rect_fade=True,
            rounded_rect_fade_ratio=0.16,
            rounded_corner_radius=0.30,
        )

        self.assertEqual(tuple(packed.shape), (1, 80, 80, 3))
        self.assertEqual(float(mask[0, 0, 0]), 0.0)
        self.assertLess(float(mask[0, 0, 20]), 0.2)
        self.assertLess(float(mask[0, 40, 0]), 0.2)
        self.assertEqual(float(mask[0, 10, 5]), 1.0)
        self.assertEqual(float(mask[0, 10, 20]), 1.0)
        self.assertEqual(float(mask[0, 40, 20]), 1.0)
        self.assertEqual(float(image[0, 0, 0, 0]), 0.0)
        self.assertEqual(float(image[0, 10, 20, 0]), 1.0)

    def test_max_corner_radius_becomes_an_inscribed_ellipse(self):
        node = FastBottomFitOverlay()
        mask = node._make_rounded_rect_fade_mask(
            1,
            80,
            40,
            torch.device("cpu"),
            torch.float32,
            0.0,
            1.0,
        )[0, 0]

        self.assertEqual(float(mask[10, 5]), 0.0)
        self.assertEqual(float(mask[10, 20]), 1.0)
        self.assertEqual(float(mask[40, 20]), 1.0)

    def test_top_and_rounded_rect_fades_are_mutually_exclusive(self):
        background = torch.zeros(1, 8, 8, 3)
        foreground = torch.ones(1, 4, 8, 3)

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            FastBottomFitOverlay().composite(
                background,
                foreground,
                enable_top_fade=True,
                enable_rounded_rect_fade=True,
            )

    def test_rounded_rect_fade_is_applied_to_fixed_canvas_packed_output(self):
        background = torch.zeros(1, 8, 6, 3)
        foreground = torch.ones(1, 2, 3, 3)

        packed = FastBottomFitOverlay().composite(
            background,
            foreground,
            packed_size_mode="Fixed Canvas (1440x1280)",
            enable_rounded_rect_fade=True,
        )[2]

        self.assertEqual(tuple(packed.shape), (1, 1280, 1440, 3))
        self.assertEqual(float(packed[0, 800, 0, 0]), 0.0)
        self.assertEqual(float(packed[0, 1040, 360, 0]), 1.0)
        self.assertEqual(float(packed[0, 1040, 1080, 0]), 1.0)

    def test_legacy_positional_layer_mask_argument_still_works(self):
        background = torch.zeros(1, 8, 6, 3)
        foreground = torch.ones(1, 2, 3, 3)
        layer_mask = torch.zeros(1, 2, 3)

        packed = FastBottomFitOverlay().composite(
            background,
            foreground,
            1.0,
            True,
            False,
            0.08,
            True,
            layer_mask,
        )[2]

        self.assertEqual(tuple(packed.shape), (1, 4, 12, 3))
        self.assertEqual(float(packed.max()), 0.0)

    def test_fit_content_keeps_dynamic_packed_height(self):
        background = torch.zeros(1, 8, 6, 3)
        foreground = torch.ones(1, 2, 3, 3)

        packed = FastBottomFitOverlay().composite(
            background,
            foreground,
            packed_size_mode="Fit Content (Dynamic Height)",
        )[2]

        self.assertEqual(tuple(packed.shape), (1, 4, 12, 3))
        self.assertTrue(torch.equal(packed, torch.ones_like(packed)))

    def test_fixed_canvas_is_1440x1280_with_black_top_padding(self):
        background = torch.zeros(1, 8, 6, 3)
        foreground = torch.ones(1, 2, 3, 3)

        packed = FastBottomFitOverlay().composite(
            background,
            foreground,
            opacity=0.5,
            packed_size_mode="Fixed Canvas (1440x1280)",
        )[2]

        self.assertEqual(tuple(packed.shape), (1, 1280, 1440, 3))
        self.assertEqual(float(packed[:, :800].max()), 0.0)
        self.assertTrue(torch.equal(packed[:, 800:], torch.full_like(packed[:, 800:], 0.5)))

    def test_fixed_canvas_crops_overflow_from_the_top(self):
        background = torch.zeros(1, 8, 6, 3)
        foreground = torch.ones(1, 2, 1, 3)

        packed = FastBottomFitOverlay().composite(
            background,
            foreground,
            packed_size_mode="Fixed Canvas (1440x1280)",
        )[2]

        self.assertEqual(tuple(packed.shape), (1, 1280, 1440, 3))
        self.assertTrue(torch.equal(packed, torch.ones_like(packed)))


if __name__ == "__main__":
    unittest.main()
