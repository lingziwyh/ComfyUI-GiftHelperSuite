from __future__ import annotations

import pathlib
import sys
import unittest

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fast_bottom_fit_overlay import FastBottomFitOverlay  # noqa: E402


class FastBottomFitOverlayTests(unittest.TestCase):
    def test_half_scale_keeps_fitted_video_center_and_packed_panels_aligned(self):
        bg = torch.full((1, 12, 8, 3), 0.2)
        fg = torch.ones(2, 4, 8, 3)
        image, mask, packed = FastBottomFitOverlay().composite(bg, fg, center_scale=0.5)
        expected = torch.zeros(2, 12, 8)
        expected[:, 9:11, 2:6] = 1
        self.assertTrue(torch.equal(mask, expected))
        self.assertTrue(torch.allclose(image, bg.expand(2, -1, -1, -1) * (1 - expected[..., None]) + expected[..., None]))
        self.assertEqual(tuple(packed.shape), (2, 4, 16, 3))
        self.assertTrue(torch.equal(packed[:, :, :8, 0], expected[:, 8:]))
        self.assertTrue(torch.equal(packed[:, :, :8], packed[:, :, 8:]))

    def test_fixed_canvas_scale_has_correct_center_and_zero_endpoint(self):
        bg = torch.full((1, 1280, 720, 3), 0.2)
        fg = torch.ones(1, 4, 3, 3)
        node = FastBottomFitOverlay()
        for scale in (0.5, 0.0):
            with self.subTest(scale=scale):
                image, mask, packed = node.composite(
                    bg, fg, center_scale=scale, packed_size_mode="Fixed Canvas (1440x1280)"
                )
                expected = torch.zeros(1, 1280, 720)
                if scale:
                    expected[:, 560:1040, 180:540] = 1
                self.assertEqual(tuple(packed.shape), (1, 1280, 1440, 3))
                self.assertTrue(torch.equal(mask, expected))
                self.assertTrue(torch.equal(packed[:, :, :720, 0], expected))
                self.assertTrue(torch.equal(packed[:, :, :720], packed[:, :, 720:]))
                if not scale:
                    self.assertTrue(torch.equal(image, bg))

    def test_scale_uses_center_of_visible_video_after_top_crop(self):
        bg = torch.zeros(1, 8, 8, 3)
        fg = torch.ones(1, 16, 8, 3)
        _, mask, packed = FastBottomFitOverlay().composite(bg, fg, center_scale=0.5)
        expected = torch.zeros_like(mask)
        expected[:, 2:6, 2:6] = 1
        self.assertTrue(torch.equal(mask, expected))
        self.assertEqual(tuple(packed.shape), (1, 16, 16, 3))

    def test_scaled_feather_uses_premultiplied_color_and_retains_dtype(self):
        for dtype in (torch.float32, torch.float16):
            with self.subTest(dtype=dtype):
                bg = torch.zeros(1, 80, 40, 3, dtype=dtype)
                fg = torch.ones_like(bg)
                image, mask, packed = FastBottomFitOverlay().composite(
                    bg, fg, center_scale=0.5, opacity=0.7,
                    enable_rounded_rect_fade=True,
                )
                self.assertEqual(image.dtype, dtype)
                self.assertTrue(torch.equal(image[..., 0], mask))
                self.assertTrue(torch.equal(packed[:, :, :40, 0], mask))
                self.assertTrue(torch.equal(packed[:, :, 40:], image))
                self.assertTrue(bool(((mask > 0) & (mask < 0.7)).any()))

    def test_new_controls_are_optional_for_legacy_api_workflows(self):
        input_types = FastBottomFitOverlay.INPUT_TYPES()
        optional = input_types["optional"]

        self.assertIn("packed_size_mode", optional)
        self.assertIn("enable_rounded_rect_fade", optional)
        self.assertIn("rounded_rect_fade_ratio", optional)
        self.assertIn("rounded_corner_radius", optional)
        self.assertEqual(optional["center_scale"][1]["default"], 1.0)

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
