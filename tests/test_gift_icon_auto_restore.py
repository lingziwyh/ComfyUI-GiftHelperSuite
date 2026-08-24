from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import unittest
from unittest import mock

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE_NAME = "gift_helper_suite_icon_test"


def load_suite():
    if PACKAGE_NAME in sys.modules:
        return sys.modules[PACKAGE_NAME]
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = module
    with mock.patch.dict(os.environ, {"GIFT_HELPER_SKIP_EXAMPLE_ASSETS": "1"}):
        spec.loader.exec_module(module)
    return module


class GiftIconAutoRestoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.node_class = load_suite().NODE_CLASS_MAPPINGS["GiftIconAutoRestore"]

    @staticmethod
    def make_inputs():
        height, width = 48, 64
        subject = torch.zeros(1, height, width, 4)
        subject[:, 14:38, 24:40, 0] = 0.85
        subject[:, 14:38, 24:40, 1] = 0.35
        subject[:, 14:38, 24:40, 2] = 0.10
        subject[:, 14:38, 24:40, 3] = 1.0
        mask = subject[..., 3].clone()

        source = torch.zeros(1, height, width, 3)
        source[:, 14:38, 24:40] = subject[:, 14:38, 24:40, :3]
        # A soft blue glow lives outside the solid subject mask.
        source[:, 8:44, 18:46, 2] = torch.maximum(
            source[:, 8:44, 18:46, 2],
            torch.full((1, 36, 28), 0.18),
        )
        return subject, mask, source

    def run_node(self, **overrides):
        subject, mask, source = self.make_inputs()
        options = {
            "fx_strength": 1.0,
            "unmult_black_point": 0.105,
            "fx_reach": 0.6,
            "fx_edge_feather": 2,
            "alpha_cutoff": 0.001,
            "enable_edge_fade": True,
            "edge_guard_percent": 0.0,
            "edge_feather_percent": 2.5,
            "min_effect_visibility": 0.004,
            "crop_padding": 0,
            "output_padding": 0,
            "target_size": 96,
            "thumbnail_size": 24,
            "preview_scale": 1.0,
            "preview_canvas_size": 96,
            "performance_mode": "cpu",
        }
        options.update(overrides)
        return self.node_class().restore_and_export(subject, mask, source, **options)

    def test_outputs_have_standard_square_shapes_and_rgba(self):
        preview, icon, thumbnail = self.run_node()
        self.assertEqual(tuple(preview.shape), (1, 96, 96, 3))
        self.assertEqual(tuple(icon.shape), (1, 96, 96, 4))
        self.assertEqual(tuple(thumbnail.shape), (1, 24, 24, 4))
        self.assertTrue(preview.is_contiguous())
        self.assertTrue(icon.is_contiguous())
        self.assertTrue(thumbnail.is_contiguous())

    def test_schema_keeps_production_defaults(self):
        required = self.node_class.INPUT_TYPES()["required"]
        self.assertEqual(required["unmult_black_point"][1]["default"], 0.105)
        self.assertEqual(required["output_padding"][1]["default"], 8)
        self.assertEqual(required["preview_canvas_size"][1]["default"], 1680)
        self.assertEqual(required["performance_mode"][1]["default"], "auto")

    def test_final_alpha_includes_glow_beyond_the_subject_mask(self):
        _, icon, _ = self.run_node()
        alpha = icon[0, ..., 3]
        nonzero = torch.nonzero(alpha > 0.001, as_tuple=False)
        alpha_height = int(nonzero[:, 0].max() - nonzero[:, 0].min() + 1)
        alpha_width = int(nonzero[:, 1].max() - nonzero[:, 1].min() + 1)
        # The source glow is wider than the 16x24 subject and must affect the crop.
        self.assertGreater(alpha_width / alpha_height, 16 / 24)

    def test_long_alpha_edge_touches_canvas_and_short_edge_is_transparent(self):
        _, icon, thumbnail = self.run_node()
        for image, last in ((icon, 95), (thumbnail, 23)):
            alpha = image[0, ..., 3]
            nonzero = torch.nonzero(alpha > (0.5 / 255.0), as_tuple=False)
            y_min, x_min = nonzero.min(dim=0).values.tolist()
            y_max, x_max = nonzero.max(dim=0).values.tolist()
            self.assertTrue(
                (y_min == 0 and y_max == last) or (x_min == 0 and x_max == last)
            )
            self.assertTrue(
                bool(torch.all(alpha[:, 0] == 0))
                or bool(torch.all(alpha[0, :] == 0))
            )

    def test_preview_background_is_used_without_changing_rgba_outputs(self):
        background = torch.full((1, 120, 140, 3), 0.25)
        preview, icon, thumbnail = self.run_node(preview_background=background)
        self.assertEqual(tuple(preview.shape), (1, 120, 140, 3))
        self.assertEqual(tuple(icon.shape), (1, 96, 96, 4))
        self.assertEqual(tuple(thumbnail.shape), (1, 24, 24, 4))
        self.assertTrue(torch.allclose(preview[0, 0, 0], torch.full((3,), 0.25)))

    def test_lifted_black_noise_does_not_become_valid_crop_content(self):
        subject, mask, source = self.make_inputs()
        torch.manual_seed(7)
        noisy_source = (source + torch.rand_like(source) * 0.004).clamp(0.0, 1.0)
        _, icon, _ = self.node_class().restore_and_export(
            subject,
            mask,
            noisy_source,
            fx_strength=1.0,
            unmult_black_point=0.105,
            fx_reach=0.8,
            fx_edge_feather=2,
            alpha_cutoff=0.001,
            enable_edge_fade=True,
            edge_guard_percent=0.0,
            edge_feather_percent=2.5,
            min_effect_visibility=0.006,
            crop_padding=0,
            output_padding=0,
            target_size=96,
            thumbnail_size=24,
            preview_scale=1.0,
            preview_canvas_size=96,
            performance_mode="cpu",
        )
        alpha = icon[0, ..., 3]
        # Background lift is present everywhere in the source, but the exported
        # icon still has true transparent padding instead of a full-canvas haze.
        self.assertTrue(bool(torch.all(alpha[:, 0] == 0)))
        self.assertTrue(bool(torch.all(alpha[:, -1] == 0)))

    def test_edge_guard_softens_a_subject_cut_by_the_source_canvas(self):
        height, width = 40, 64
        subject = torch.zeros(1, height, width, 4)
        subject[:, 10:30, :, :3] = 0.8
        subject[:, 10:30, :, 3] = 1.0
        mask = subject[..., 3].clone()
        source = subject[..., :3].clone()
        common = dict(
            fx_strength=0.0,
            unmult_black_point=0.105,
            fx_reach=0.2,
            fx_edge_feather=2,
            alpha_cutoff=0.001,
            min_effect_visibility=0.006,
            crop_padding=0,
            output_padding=0,
            target_size=64,
            thumbnail_size=16,
            preview_scale=1.0,
            preview_canvas_size=64,
            performance_mode="cpu",
        )
        _, hard, _ = self.node_class().restore_and_export(
            subject,
            mask,
            source,
            enable_edge_fade=False,
            edge_guard_percent=0.0,
            edge_feather_percent=20.0,
            **common,
        )
        _, soft, _ = self.node_class().restore_and_export(
            subject,
            mask,
            source,
            enable_edge_fade=True,
            edge_guard_percent=0.0,
            edge_feather_percent=20.0,
            **common,
        )
        hard_alpha = hard[0, ..., 3]
        soft_alpha = soft[0, ..., 3]
        self.assertGreater(float(hard_alpha[:, 0].max()), 0.95)
        self.assertLess(float(soft_alpha[:, 0].max()), 0.15)
        self.assertGreater(float(soft_alpha.max()), 0.95)

    def test_custom_edge_guard_overrides_generated_guard(self):
        subject, mask, source = self.make_inputs()
        custom = torch.ones(1, 24, 32)
        custom[:, :, :8] = 0.0
        _, icon, _ = self.node_class().restore_and_export(
            subject,
            mask,
            source,
            fx_strength=0.0,
            unmult_black_point=0.105,
            enable_edge_fade=False,
            edge_guard_percent=0.0,
            edge_feather_percent=1.0,
            edge_guard_mask=custom,
            output_padding=0,
            target_size=64,
            thumbnail_size=16,
            preview_canvas_size=64,
            performance_mode="cpu",
        )
        self.assertEqual(tuple(icon.shape), (1, 64, 64, 4))
        self.assertEqual(float(icon[..., 3].min()), 0.0)

    def test_output_padding_uses_target_pixels_and_scales_for_thumbnail(self):
        _, icon, thumbnail = self.run_node(output_padding=4)
        for image, expected_padding in ((icon, 4), (thumbnail, 1)):
            alpha = image[0, ..., 3]
            nonzero = torch.nonzero(alpha > (0.5 / 255.0), as_tuple=False)
            y_min, x_min = nonzero.min(dim=0).values.tolist()
            y_max, x_max = nonzero.max(dim=0).values.tolist()
            size = int(alpha.shape[0])
            self.assertTrue(
                (
                    y_min == expected_padding
                    and y_max == size - 1 - expected_padding
                )
                or (
                    x_min == expected_padding
                    and x_max == size - 1 - expected_padding
                )
            )

    def test_empty_subject_mask_fails_clearly(self):
        subject, mask, source = self.make_inputs()
        with self.assertRaisesRegex(ValueError, "subject_mask.*empty"):
            self.node_class().restore_and_export(
                subject,
                torch.zeros_like(mask),
                source,
                target_size=64,
                thumbnail_size=16,
                performance_mode="cpu",
            )


if __name__ == "__main__":
    unittest.main()
