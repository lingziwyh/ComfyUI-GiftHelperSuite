from __future__ import annotations

import pathlib
import sys
import unittest

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gift_chroma_master.core.utils import (  # noqa: E402
    image_bhwc_to_bchw,
    mask_to_bchw,
    parse_color,
)


class ChromaUtilityTests(unittest.TestCase):
    def test_bhwc_contract_is_not_ambiguous_when_height_is_three(self):
        image = torch.rand(1, 3, 4, 3)
        rgb, alpha = image_bhwc_to_bchw(image)
        self.assertEqual(tuple(rgb.shape), (1, 3, 3, 4))
        self.assertIsNone(alpha)

    def test_image_boundary_rejects_integer_and_nonfinite_data(self):
        with self.assertRaises(TypeError):
            image_bhwc_to_bchw(torch.zeros(1, 2, 2, 3, dtype=torch.uint8))
        invalid = torch.zeros(1, 2, 2, 3)
        invalid[0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            image_bhwc_to_bchw(invalid)

    def test_rgb255_triplets_are_normalized(self):
        color = parse_color([255, 128, 0])
        expected = torch.tensor([1.0, 128.0 / 255.0, 0.0])
        self.assertTrue(torch.allclose(color[0, :, 0, 0], expected))

    def test_bad_color_is_not_silently_replaced_with_green(self):
        with self.assertRaises(ValueError):
            parse_color("not-a-color")

    def test_precomputed_per_frame_color_tensor_is_supported_strictly(self):
        colors = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.2, 1.0]])
        result = parse_color(colors, batch=2)
        self.assertEqual(tuple(result.shape), (2, 3, 1, 1))
        self.assertTrue(torch.equal(result[:, :, 0, 0], colors))
        with self.assertRaises(ValueError):
            parse_color(colors * 256.0, batch=2)

    def test_single_mask_broadcasts_across_batch(self):
        mask = torch.ones(1, 5, 7)
        result = mask_to_bchw(mask, batch=3, height=5, width=7, device=torch.device("cpu"), dtype=torch.float32)
        self.assertEqual(tuple(result.shape), (3, 1, 5, 7))
        self.assertEqual(result.stride(0), 0)

    def test_height_one_bhwc_mask_is_not_misread_as_bchw(self):
        mask = torch.arange(10, dtype=torch.float32).reshape(2, 1, 5, 1) / 9.0
        result = mask_to_bchw(
            mask,
            batch=2,
            height=1,
            width=5,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertEqual(tuple(result.shape), (2, 1, 1, 5))
        self.assertTrue(torch.equal(result, mask.permute(0, 3, 1, 2)))


if __name__ == "__main__":
    unittest.main()
