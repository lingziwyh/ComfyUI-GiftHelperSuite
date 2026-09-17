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
                    fade_mode="Rounded Rect",
                )
                self.assertEqual(image.dtype, dtype)
                self.assertTrue(torch.equal(image[..., 0], mask))
                self.assertTrue(torch.equal(packed[:, :, :40, 0], mask))
                self.assertTrue(torch.equal(packed[:, :, 40:], image))
                self.assertTrue(bool(((mask > 0) & (mask < 0.7)).any()))

    def test_controls_follow_global_then_preset_ui_order(self):
        input_types = FastBottomFitOverlay.INPUT_TYPES()
        required = input_types["required"]
        optional = input_types["optional"]

        self.assertEqual(
            list(required),
            [
                "background_image",
                "layer_image",
                "opacity",
                "clip_if_too_tall",
                "enable_packed_output",
            ],
        )
        self.assertNotIn("fade_mode", required)
        self.assertNotIn("top_fade_ratio", required)
        self.assertTrue(required["clip_if_too_tall"][1]["advanced"])
        self.assertIn("packed_size_mode", optional)
        self.assertIn("fade_mode", optional)
        self.assertEqual(optional["fade_mode"][0], ["None", "Top Fade", "Rounded Rect"])
        self.assertIn("rounded_rect_fade_ratio", optional)
        self.assertIn("rounded_corner_radius", optional)
        self.assertIn("foreground_image", optional)
        self.assertIn("foreground_mask", optional)
        self.assertIn("fade_frames", optional)
        self.assertEqual(optional["preset"][0], ["Custom", "Low Coins", "Standard", "Naked-Eye 3D"])
        self.assertEqual(optional["background_fade_ratio"][1]["default"], 0.0)
        self.assertEqual(optional["center_scale"][1]["default"], 1.0)
        self.assertEqual(optional["fade_frames"][1]["default"], 0)
        optional_order = list(optional)
        self.assertLess(optional_order.index("fade_frames"), optional_order.index("preset"))
        self.assertLess(optional_order.index("packed_size_mode"), optional_order.index("preset"))
        self.assertLess(optional_order.index("preset"), optional_order.index("fade_mode"))
        self.assertLess(optional_order.index("preset"), optional_order.index("center_scale"))

    def test_fade_in_out_affects_complete_stack_but_not_background(self):
        background = torch.full((5, 2, 2, 3), 0.2)
        layer = torch.zeros(5, 2, 2, 3)
        layer[..., 0] = 1.0
        foreground = torch.zeros(5, 2, 2, 3)
        foreground[..., 1] = 1.0
        foreground_mask = torch.zeros(5, 2, 2)
        foreground_mask[:, :, 0] = 1.0

        image, mask, packed = FastBottomFitOverlay().composite(
            background,
            layer,
            foreground_image=foreground,
            foreground_mask=foreground_mask,
            fade_frames=3,
        )

        self.assertTrue(torch.equal(image[0], background[0]))
        self.assertTrue(torch.equal(image[-1], background[-1]))
        self.assertTrue(torch.equal(mask[:, 0, 0], torch.tensor([0.0, 0.5, 1.0, 0.5, 0.0])))
        self.assertTrue(torch.equal(mask[:, 0, 1], torch.tensor([0.0, 0.5, 1.0, 0.5, 0.0])))
        self.assertTrue(torch.allclose(image[1, 0, 0], torch.tensor([0.1, 0.6, 0.1])))
        self.assertTrue(torch.allclose(image[1, 0, 1], torch.tensor([0.6, 0.1, 0.1])))
        self.assertTrue(torch.equal(image[2, 0, 0], torch.tensor([0.0, 1.0, 0.0])))
        self.assertTrue(torch.equal(image[2, 0, 1], torch.tensor([1.0, 0.0, 0.0])))
        self.assertTrue(torch.equal(packed[:, :, :2, 0], mask))
        self.assertTrue(torch.equal(packed[0], torch.zeros_like(packed[0])))
        self.assertTrue(torch.equal(packed[-1], torch.zeros_like(packed[-1])))

    def test_production_presets_match_the_requested_fade_values(self):
        self.assertEqual(
            FastBottomFitOverlay.PRESET_VALUES,
            {
                "Low Coins": {
                    "fade_mode": "Rounded Rect",
                    "background_fade_ratio": 0.0,
                    "rounded_rect_fade_ratio": 0.255,
                    "rounded_corner_radius": 0.90,
                    "center_scale": 0.85,
                },
                "Standard": {
                    "fade_mode": "Top Fade",
                    "top_fade_ratio": 0.08,
                    "background_fade_ratio": 0.0,
                },
                "Naked-Eye 3D": {
                    "fade_mode": "Top Fade",
                    "top_fade_ratio": 0.045,
                    "background_fade_ratio": 0.52,
                },
            },
        )

    def test_low_coins_preset_overrides_center_scale(self):
        background = torch.zeros(1, 10, 10, 3)
        layer = torch.ones(1, 10, 10, 3)

        _, mask, _ = FastBottomFitOverlay().composite(
            background,
            layer,
            preset="Low Coins",
            center_scale=1.0,
        )

        self.assertEqual(float(mask[:, :, 0].max()), 0.0)
        self.assertGreater(float(mask[:, :, 1:-1].max()), 0.0)

    def test_background_fade_affects_layer_but_not_restored_foreground(self):
        background = torch.zeros(1, 4, 4, 3)
        layer = torch.zeros(1, 4, 4, 3)
        layer[..., 0] = 1.0
        foreground = torch.zeros(1, 4, 4, 3)
        foreground[..., 1] = 1.0
        foreground_mask = torch.ones(1, 4, 4)

        image, mask, packed = FastBottomFitOverlay().composite(
            background,
            layer,
            background_fade_ratio=0.5,
            foreground_image=foreground,
            foreground_mask=foreground_mask,
        )

        self.assertTrue(torch.equal(image, foreground))
        self.assertTrue(torch.equal(mask, torch.ones_like(mask)))
        self.assertTrue(torch.equal(packed[:, :, :4], torch.ones_like(packed[:, :, :4])))
        self.assertTrue(torch.equal(packed[:, :, 4:, 1], torch.ones_like(packed[:, :, 4:, 1])))

    def test_naked_eye_3d_preset_applies_large_background_and_short_global_fades(self):
        background = torch.zeros(1, 100, 4, 3)
        layer = torch.ones(1, 100, 4, 3)

        _, mask, _ = FastBottomFitOverlay().composite(
            background,
            layer,
            preset="Naked-Eye 3D",
        )

        self.assertEqual(float(mask[0, 0, 0]), 0.0)
        self.assertGreater(float(mask[0, 2, 0]), 0.0)
        self.assertLess(float(mask[0, 50, 0]), 1.0)
        self.assertEqual(float(mask[0, 52, 0]), 1.0)

    def test_top_fade_applies_to_restored_foreground_and_layer(self):
        background = torch.zeros(1, 4, 4, 3)
        layer = torch.zeros(1, 4, 4, 3)
        layer[..., 0] = 1.0
        foreground = torch.zeros(1, 4, 4, 3)
        foreground[..., 1] = 1.0
        foreground_mask = torch.zeros(1, 4, 4)
        foreground_mask[:, 0:2, 1:3] = 1.0

        image, mask, packed = FastBottomFitOverlay().composite(
            background,
            layer,
            fade_mode="Top Fade",
            top_fade_ratio=0.5,
            foreground_image=foreground,
            foreground_mask=foreground_mask,
        )

        self.assertTrue(torch.equal(image[0, 0], torch.zeros_like(image[0, 0])))
        self.assertTrue(torch.equal(mask[0, 0], torch.zeros_like(mask[0, 0])))
        self.assertTrue(torch.equal(image[0, 1, 1:3], torch.tensor([[0.0, 1.0, 0.0]]).expand(2, -1)))
        self.assertTrue(torch.equal(mask[0, 1, 1:3], torch.ones(2)))
        self.assertTrue(torch.equal(packed[:, :, :4, 0], mask))
        self.assertTrue(torch.equal(packed[0, 0, 4:8], torch.zeros_like(packed[0, 0, 4:8])))
        self.assertTrue(torch.equal(packed[0, 1, 5:7], torch.tensor([[0.0, 1.0, 0.0]]).expand(2, -1)))

    def test_top_fade_applies_to_restored_foreground_in_fixed_canvas_packed_output(self):
        background = torch.zeros(1, 8, 6, 3)
        layer = torch.zeros(1, 2, 3, 3)
        layer_mask = torch.zeros(1, 2, 3)
        foreground = torch.zeros(1, 2, 3, 3)
        foreground[..., 1] = 1.0
        foreground_mask = torch.ones(1, 2, 3)

        packed = FastBottomFitOverlay().composite(
            background,
            layer,
            layer_mask=layer_mask,
            fade_mode="Top Fade",
            top_fade_ratio=0.5,
            foreground_image=foreground,
            foreground_mask=foreground_mask,
            packed_size_mode="Fixed Canvas (1440x1280)",
        )[2]

        self.assertEqual(tuple(packed.shape), (1, 1280, 1440, 3))
        self.assertEqual(float(packed[0, 800, 360, 0]), 0.0)
        self.assertEqual(float(packed[0, 800, 1080, 1]), 0.0)
        self.assertEqual(float(packed[0, 1040, 360, 0]), 1.0)
        self.assertEqual(float(packed[0, 1040, 1080, 1]), 1.0)

    def test_optional_foreground_uses_source_over_alpha_and_broadcasts_batch(self):
        background = torch.zeros(1, 2, 2, 3)
        layer = torch.zeros(1, 2, 2, 3)
        layer[..., 0] = 1.0
        layer_mask = torch.full((1, 2, 2), 0.5)
        foreground = torch.zeros(2, 2, 2, 3)
        foreground[..., 2] = 1.0
        foreground_mask = torch.full((2, 2, 2), 0.5)

        image, mask, _ = FastBottomFitOverlay().composite(
            background,
            layer,
            layer_mask=layer_mask,
            foreground_image=foreground,
            foreground_mask=foreground_mask,
        )

        self.assertEqual(tuple(image.shape), (2, 2, 2, 3))
        self.assertTrue(torch.allclose(mask, torch.full_like(mask, 0.75)))
        expected = torch.tensor([0.25, 0.0, 0.5])
        self.assertTrue(torch.allclose(image, expected.view(1, 1, 1, 3).expand_as(image)))

    def test_omitting_optional_foreground_keeps_legacy_result_bit_exact(self):
        torch.manual_seed(7)
        background = torch.rand(2, 8, 6, 3)
        layer = torch.rand(1, 4, 3, 3)
        layer_mask = torch.rand(1, 4, 3)
        node = FastBottomFitOverlay()

        legacy = node.composite(background, layer, layer_mask=layer_mask)
        explicit_none = node.composite(
            background,
            layer,
            layer_mask=layer_mask,
            foreground_image=None,
            foreground_mask=None,
        )

        for old, new in zip(legacy, explicit_none):
            self.assertTrue(torch.equal(old, new))

    def test_rounded_rect_follows_layer_aspect_ratio(self):
        background = torch.zeros(1, 80, 40, 3)
        foreground = torch.ones(1, 80, 40, 3)

        image, mask, packed = FastBottomFitOverlay().composite(
            background,
            foreground,
            fade_mode="Rounded Rect",
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

    def test_unknown_fade_mode_is_rejected(self):
        background = torch.zeros(1, 8, 8, 3)
        foreground = torch.ones(1, 4, 8, 3)

        with self.assertRaisesRegex(ValueError, "Unknown fade_mode"):
            FastBottomFitOverlay().composite(
                background,
                foreground,
                fade_mode="Both",
            )

    def test_rounded_rect_fade_is_applied_to_fixed_canvas_packed_output(self):
        background = torch.zeros(1, 8, 6, 3)
        foreground = torch.ones(1, 2, 3, 3)

        packed = FastBottomFitOverlay().composite(
            background,
            foreground,
            packed_size_mode="Fixed Canvas (1440x1280)",
            fade_mode="Rounded Rect",
        )[2]

        self.assertEqual(tuple(packed.shape), (1, 1280, 1440, 3))
        self.assertEqual(float(packed[0, 800, 0, 0]), 0.0)
        self.assertEqual(float(packed[0, 1040, 360, 0]), 1.0)
        self.assertEqual(float(packed[0, 1040, 1080, 0]), 1.0)

    def test_layer_mask_controls_packed_output(self):
        background = torch.zeros(1, 8, 6, 3)
        foreground = torch.ones(1, 2, 3, 3)
        layer_mask = torch.zeros(1, 2, 3)

        packed = FastBottomFitOverlay().composite(
            background,
            foreground,
            layer_mask=layer_mask,
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
