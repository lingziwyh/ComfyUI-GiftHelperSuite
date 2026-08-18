from __future__ import annotations

import pathlib
import sys
import unittest

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gift_chroma_master.core.cleaner import clean_alpha_v5  # noqa: E402


class ChromaCleanerTests(unittest.TestCase):
    def test_solid_matte_has_no_dark_frame(self):
        rgb = torch.rand(1, 3, 5, 7)
        alpha = torch.ones(1, 1, 5, 7)
        clean, _ = clean_alpha_v5(rgb, alpha, edge_radius=4.5, strength=1.0)
        self.assertGreater(float(clean.min()), 0.99999)

    def test_zero_effect_mask_is_exact_bypass(self):
        rgb = torch.rand(2, 3, 24, 24)
        alpha = torch.rand(2, 1, 24, 24)
        clean, _ = clean_alpha_v5(
            rgb, alpha, edge_radius=3.0, strength=1.0,
            effect_mask=torch.zeros(1, 1, 24, 24),
        )
        self.assertTrue(torch.equal(clean, alpha))

    def test_static_video_chatter_is_reduced(self):
        torch.manual_seed(7)
        frames = 7
        rgb = torch.full((frames, 3, 32, 32), 0.4)
        base = torch.zeros(1, 1, 32, 32)
        base[:, :, :, 16:] = 1.0
        noise = torch.randn(frames, 1, 32, 32) * 0.045
        alpha = (base + noise).clamp(0.0, 1.0)
        clean, diagnostics = clean_alpha_v5(
            rgb, alpha,
            edge_radius=2.0, strength=0.8, detail_recovery=0.0,
            temporal_mode="fast", reduce_chatter=0.9,
        )
        before = alpha[:, :, :, 15:18].std(dim=0).mean()
        after = clean[:, :, :, 15:18].std(dim=0).mean()
        self.assertLess(float(after), float(before))
        self.assertGreater(float(diagnostics["temporal_gate"].mean()), 0.0)

    def test_moving_silhouette_does_not_borrow_a_stale_edge(self):
        frames, height, width = 8, 32, 48
        rgb = torch.zeros(frames, 3, height, width)
        rgb[:, 1] = 0.8
        alpha = torch.zeros(frames, 1, height, width)
        for index in range(frames):
            edge = 10 + index * 3
            alpha[index, :, :, edge:] = 1.0
            rgb[index, 0, :, edge:] = 0.8
            rgb[index, 1, :, edge:] = 0.1

        spatial, _ = clean_alpha_v5(
            rgb, alpha,
            edge_radius=2.0, strength=0.8, detail_recovery=0.0,
            temporal_mode="off", reduce_chatter=1.0,
        )
        temporal, diagnostics = clean_alpha_v5(
            rgb, alpha,
            edge_radius=2.0, strength=0.8, detail_recovery=0.0,
            temporal_mode="fast", reduce_chatter=1.0,
        )
        self.assertLess(float((temporal - spatial).abs().max()), 1.0e-6)
        self.assertEqual(float(diagnostics["temporal_gate"].max()), 0.0)

    def test_scene_cut_matches_processing_each_shot_separately(self):
        torch.manual_seed(17)
        height, width = 24, 28
        first_rgb = torch.full((3, 3, height, width), 0.12)
        second_rgb = torch.full((3, 3, height, width), 0.88)
        first_alpha = (torch.full((3, 1, height, width), 0.45)
                       + torch.randn(3, 1, height, width) * 0.025).clamp(0.0, 1.0)
        second_alpha = (torch.full((3, 1, height, width), 0.55)
                        + torch.randn(3, 1, height, width) * 0.025).clamp(0.0, 1.0)
        rgb = torch.cat((first_rgb, second_rgb), dim=0)
        alpha = torch.cat((first_alpha, second_alpha), dim=0)
        options = dict(
            edge_radius=2.0, strength=0.8, detail_recovery=0.0,
            temporal_mode="fast", reduce_chatter=0.9,
        )

        together, _ = clean_alpha_v5(rgb, alpha, **options)
        first, _ = clean_alpha_v5(first_rgb, first_alpha, **options)
        second, _ = clean_alpha_v5(second_rgb, second_alpha, **options)
        segmented = torch.cat((first, second), dim=0)
        self.assertTrue(torch.equal(together, segmented))

    def test_fast_temporal_does_not_trail_a_moving_edge(self):
        height, width = 24, 56
        x = torch.arange(width).view(1, 1, 1, width).float()
        positions = torch.tensor((40.0, 34.0, 28.0)).view(3, 1, 1, 1)
        alpha = torch.sigmoid((positions - x) / 1.2).expand(3, 1, height, width)
        foreground = torch.tensor((0.8, 0.1, 0.1)).view(1, 3, 1, 1)
        screen = torch.tensor((0.05, 0.75, 0.08)).view(1, 3, 1, 1)
        rgb = foreground * alpha + screen * (1.0 - alpha)

        spatial, _ = clean_alpha_v5(
            rgb, alpha,
            edge_radius=2.0, strength=0.8, detail_recovery=0.0,
            temporal_mode="off", reduce_chatter=0.0,
        )
        temporal, diagnostics = clean_alpha_v5(
            rgb, alpha,
            edge_radius=2.0, strength=0.8, detail_recovery=0.0,
            temporal_mode="fast", reduce_chatter=0.9,
        )
        disoccluded = (alpha[:-1] > 0.7) & (alpha[1:] < 0.3)
        self.assertTrue(bool(disoccluded.any()))
        self.assertLess(
            float((temporal[1:] - spatial[1:])[disoccluded].abs().max()),
            1.0e-6,
        )
        self.assertLess(float(diagnostics["temporal_gate"][1:][disoccluded].max()), 1.0e-6)

    def test_fast_temporal_rejects_a_scene_cut(self):
        torch.manual_seed(11)
        height, width = 20, 28
        alpha = torch.rand(2, 1, height, width)
        rgb = torch.empty(2, 3, height, width)
        rgb[0] = torch.tensor((0.05, 0.75, 0.08)).view(3, 1, 1)
        rgb[1] = torch.tensor((0.08, 0.10, 0.75)).view(3, 1, 1)

        spatial, _ = clean_alpha_v5(
            rgb, alpha,
            edge_radius=2.0, strength=0.8, detail_recovery=0.0,
            temporal_mode="off", reduce_chatter=0.0,
        )
        temporal, diagnostics = clean_alpha_v5(
            rgb, alpha,
            edge_radius=2.0, strength=0.8, detail_recovery=0.0,
            temporal_mode="fast", reduce_chatter=0.9,
        )
        self.assertTrue(torch.equal(temporal, spatial))
        self.assertEqual(float(diagnostics["temporal_gate"].max()), 0.0)


if __name__ == "__main__":
    unittest.main()
