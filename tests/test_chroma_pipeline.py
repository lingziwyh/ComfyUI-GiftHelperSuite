from __future__ import annotations

import pathlib
import sys
import unittest

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gift_chroma_master.core.pipeline import (  # noqa: E402
    cleaner_stage,
    diagnostics_image,
    run_trio,
    screen_key_stage,
    spill_stage,
)
from gift_chroma_master.nodes import (  # noqa: E402
    GiftChromaMaster,
    GiftChromaMasterCleaner,
    GiftChromaMasterDespill,
    GiftChromaMasterKeyer,
)


def required_defaults(node, *, skip=()):
    result = {}
    for name, spec in node.INPUT_TYPES()["required"].items():
        if name in skip:
            continue
        options = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
        if "default" in options:
            result[name] = options["default"]
        elif isinstance(spec[0], list):
            result[name] = spec[0][0]
        else:
            raise AssertionError(f"required input {name!r} has no default")
    return result


class ChromaPipelineTests(unittest.TestCase):
    def test_daily_and_expert_default_chains_are_equivalent(self):
        image = torch.zeros(3, 20, 24, 3)
        image[..., 1] = 1.0
        image[:, 5:16, 7:18] = torch.tensor([0.78, 0.16, 0.10])

        daily = GiftChromaMaster()
        daily_rgb, daily_alpha = daily.apply(
            image=image,
            **required_defaults(GiftChromaMaster, skip={"image"}),
        )
        keyer = GiftChromaMasterKeyer()
        state, _ = keyer.apply(
            image=image,
            **required_defaults(GiftChromaMasterKeyer, skip={"image"}),
        )
        cleaner = GiftChromaMasterCleaner()
        state, _ = cleaner.apply(
            keyer_state=state,
            **required_defaults(GiftChromaMasterCleaner, skip={"keyer_state"}),
        )
        spill = GiftChromaMasterDespill()
        expert_rgb, expert_alpha, _ = spill.apply(
            cleaner_state=state,
            **required_defaults(GiftChromaMasterDespill, skip={"cleaner_state"}),
        )
        # The daily node may use CUDA automatically while the expert stages
        # follow their CPU input.  The algorithms remain equivalent, but
        # cross-device kernels are not guaranteed to be bit-identical.
        self.assertTrue(torch.allclose(daily_rgb, expert_rgb, atol=2.0e-5, rtol=1.0e-5))
        self.assertTrue(torch.allclose(daily_alpha, expert_alpha, atol=2.0e-5, rtol=1.0e-5))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_chunking_preserves_video_continuity(self):
        frames, height, width = 7, 28, 36
        image = torch.zeros(frames, height, width, 3)
        for index in range(frames):
            exposure = 0.72 + 0.04 * index
            image[index, ..., 1] = exposure
            left = 8 + index
            image[index, 7:23, left:left + 12] = torch.tensor([0.82, 0.14, 0.09])
        transparency = torch.zeros(frames, height, width)
        defaults = required_defaults(GiftChromaMaster, skip={"image"})
        cpu_rgb, cpu_alpha = GiftChromaMaster().apply(
            image=image,
            source_mask=transparency,
            performance_mode="cpu",
            gpu_chunk_size=2,
            **defaults,
        )
        gpu_rgb, gpu_alpha = GiftChromaMaster().apply(
            image=image,
            source_mask=transparency,
            performance_mode="cuda",
            gpu_chunk_size=2,
            **defaults,
        )
        rgb_error = (gpu_rgb - cpu_rgb).abs()
        alpha_error = (gpu_alpha - cpu_alpha).abs()
        self.assertLess(float(rgb_error.mean()), 1.0e-4)
        self.assertLess(float(rgb_error.max()), 1.0e-3)
        self.assertLess(float(alpha_error.mean()), 1.0e-4)
        self.assertLess(float(alpha_error.max()), 1.0e-3)

    def test_complete_pipeline_has_stable_minimal_outputs(self):
        image = torch.zeros(2, 24, 32, 3)
        image[..., 1] = 1.0
        image[:, 6:18, 8:24] = torch.tensor([0.8, 0.15, 0.1])
        foreground, alpha = run_trio(
            image,
            keyer={"screen_mode": "manual", "screen_color": "#00ff00"},
            cleaner={"edge_radius": 2.0, "strength": 0.6},
            spill={"amount": 0.8},
        )
        self.assertEqual(tuple(foreground.shape), (2, 24, 32, 3))
        self.assertEqual(tuple(alpha.shape), (2, 24, 32))
        self.assertTrue(torch.isfinite(foreground).all())
        self.assertTrue(torch.isfinite(alpha).all())

    def test_load_image_transparency_mask_polarity_is_explicit(self):
        image = torch.zeros(1, 8, 9, 3)
        image[..., 0] = 1.0
        transparency = torch.ones(1, 8, 9)
        _, alpha = screen_key_stage(
            image,
            screen_mode="manual",
            screen_color="#00ff00",
            source_alpha_mode="multiply",
            source_mask_polarity="transparency",
            source_mask=transparency,
        )
        self.assertEqual(float(alpha.max()), 0.0)

    def test_rgb_load_image_placeholder_mask_is_treated_as_absent(self):
        image = torch.zeros(1, 80, 96, 3)
        image[..., 0] = 1.0
        placeholder = torch.zeros(1, 64, 64)
        _, alpha = screen_key_stage(
            image,
            screen_mode="manual",
            screen_color="#00ff00",
            source_alpha_mode="multiply",
            source_mask_polarity="transparency",
            source_mask=placeholder,
        )
        self.assertGreater(float(alpha.min()), 0.999)

    def test_expert_state_moves_through_three_named_stages(self):
        image = torch.zeros(1, 16, 20, 3)
        image[..., 1] = 1.0
        image[:, 4:12, 5:15] = torch.tensor([0.75, 0.20, 0.12])
        keyed, _ = screen_key_stage(image, screen_mode="manual", screen_color="#00ff00")
        self.assertIsInstance(keyed, tuple)
        self.assertNotIsInstance(keyed, dict)
        self.assertEqual(keyed["stage"], "keyed")
        cleaned, _ = cleaner_stage(keyed)
        self.assertEqual(cleaned["stage"], "cleaned")
        final, foreground, alpha = spill_stage(cleaned)
        self.assertEqual(final["stage"], "spilled")
        self.assertEqual(tuple(diagnostics_image(final, "spill_map").shape), (1, 16, 20, 3))
        self.assertEqual(tuple(foreground.shape[-1:]), (3,))
        self.assertEqual(tuple(alpha.shape), (1, 16, 20))

    def test_inside_and_outside_constraints_remain_hard_after_cleaner(self):
        image = torch.zeros(1, 16, 24, 3)
        image[..., 1] = 1.0
        inside = torch.zeros(1, 16, 24)
        inside[:, :, :6] = 1.0
        outside = torch.zeros(1, 16, 24)
        outside[:, :, 18:] = 1.0
        keyed, _ = screen_key_stage(
            image,
            screen_mode="manual",
            screen_color="#00ff00",
            inside_mask=inside,
            outside_mask=outside,
        )
        _, cleaned = cleaner_stage(
            keyed,
            edge_radius=4.0,
            strength=1.0,
            detail_recovery=0.0,
        )
        self.assertEqual(float(cleaned[:, :, :6].min()), 1.0)
        self.assertEqual(float(cleaned[:, :, 18:].max()), 0.0)

    def test_source_alpha_is_applied_once_after_cleaner(self):
        image = torch.zeros(1, 12, 14, 3)
        image[..., 0] = 1.0
        source_alpha = torch.full((1, 12, 14), 0.5)
        keyed, keyed_alpha = screen_key_stage(
            image,
            screen_mode="manual",
            screen_color="#00ff00",
            source_alpha_mode="multiply",
            source_mask_polarity="alpha",
            source_mask=source_alpha,
        )
        _, cleaned_alpha = cleaner_stage(
            keyed,
            edge_radius=3.0,
            strength=1.0,
        )
        self.assertTrue(torch.allclose(keyed_alpha, torch.full_like(keyed_alpha, 0.5)))
        self.assertTrue(torch.allclose(cleaned_alpha, torch.full_like(cleaned_alpha, 0.5)))

    def test_spill_mask_and_zero_amount_bypass_edge_recovery(self):
        height, width = 4, 5
        source = torch.tensor((0.40, 0.50, 0.20)).view(1, 3, 1, 1)
        source = source.expand(1, 3, height, width).contiguous()
        alpha = torch.full((1, 1, height, width), 0.5)
        state = {
            "schema": "gift_chroma_master_v1",
            "stage": "keyed",
            "rgb_source_srgb": source,
            "rgb_source_linear": source,
            "screen_srgb": torch.tensor((0.0, 1.0, 0.0)).view(1, 3, 1, 1),
            "screen_linear": torch.tensor((0.0, 1.0, 0.0)).view(1, 3, 1, 1),
            "screen_confidence": torch.ones(1, 1, 1, 1),
            "screen_mix": torch.full_like(alpha, 0.5),
            "alpha_raw": alpha,
            "alpha_base": alpha,
            "alpha_keyed": alpha,
            "alpha_clean": alpha,
            "rgb_clean_linear": source,
            "spill_map": torch.zeros_like(alpha),
        }
        masked, _, _ = spill_stage(
            state,
            amount=1.0,
            edge_recovery=1.0,
            effect_mask=torch.zeros(1, height, width),
        )
        disabled, _, _ = spill_stage(state, amount=0.0, edge_recovery=1.0)
        self.assertTrue(torch.equal(masked["rgb_clean_linear"], source))
        self.assertTrue(torch.equal(disabled["rgb_clean_linear"], source))


if __name__ == "__main__":
    unittest.main()
