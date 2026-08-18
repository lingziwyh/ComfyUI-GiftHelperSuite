from __future__ import annotations

import pathlib
import sys
import unittest

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gift_mask_blend import (  # noqa: E402
    GiftFrameSlice,
    GiftMaskBlend,
    GiftMaskFadeInOut,
    GiftMaskRamp,
)


class GiftMaskBlendTests(unittest.TestCase):
    def test_ramp_stays_white_after_end_frame(self):
        images = torch.zeros(5, 2, 3, 3)
        _, mask = GiftMaskRamp().generate(images, 1, 3)
        expected = torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0])
        self.assertTrue(torch.equal(mask[:, 0, 0], expected))

    def test_ramp_handles_reversed_and_single_frame_ranges(self):
        images = torch.zeros(4, 1, 1, 3)
        _, reversed_mask = GiftMaskRamp().generate(images, 3, 1)
        self.assertTrue(
            torch.equal(reversed_mask[:, 0, 0], torch.tensor([0.0, 0.0, 0.5, 1.0]))
        )
        _, step_mask = GiftMaskRamp().generate(images, 2, 2)
        self.assertTrue(
            torch.equal(step_mask[:, 0, 0], torch.tensor([0.0, 0.0, 1.0, 1.0]))
        )

    def test_one_frame_fade_is_visible_and_does_not_modify_input(self):
        images = torch.zeros(4, 2, 2, 3)
        source = torch.ones(5, 2, 2)
        before = source.clone()
        _, mask = GiftMaskFadeInOut().generate(images, 1, source)
        self.assertTrue(torch.equal(source, before))
        self.assertTrue(
            torch.equal(mask[:, 0, 0], torch.tensor([0.0, 1.0, 1.0, 0.0]))
        )

    def test_fade_broadcasts_and_resizes_a_single_mask(self):
        images = torch.zeros(5, 4, 6, 3)
        source = torch.full((1, 2, 3), 0.5)
        _, mask = GiftMaskFadeInOut().generate(images, 3, source)
        self.assertEqual(tuple(mask.shape), (5, 4, 6))
        self.assertTrue(
            torch.allclose(
                mask[:, 0, 0],
                torch.tensor([0.0, 0.25, 0.5, 0.25, 0.0]),
            )
        )

    def test_frame_slice_is_inclusive_and_accepts_reversed_range(self):
        images = torch.arange(6, dtype=torch.float32).view(6, 1, 1, 1)
        sliced = GiftFrameSlice().slice(images, 4, 2)[0]
        self.assertTrue(torch.equal(sliced[:, 0, 0, 0], torch.tensor([2.0, 3.0, 4.0])))

    def test_sequence_blend_repeats_short_mask_without_python_frame_loop(self):
        first = torch.ones(3, 2, 2, 3)
        second = torch.zeros(2, 2, 2, 3)
        short_mask = torch.full((1, 2, 2), 0.5)
        output = GiftMaskBlend().blend(first, second, short_mask, 1)[0]
        self.assertEqual(tuple(output.shape), (3, 2, 2, 3))
        self.assertTrue(torch.equal(output[0], first[0]))
        self.assertTrue(torch.equal(output[1:], torch.full_like(output[1:], 0.5)))

    def test_sequence_blend_aligns_resolution_and_preserves_long_background_tail(self):
        first = torch.ones(1, 4, 6, 3)
        second = torch.zeros(3, 2, 3, 3)
        output = GiftMaskBlend().blend(first, second, torch.zeros(1, 1, 1), 0)[0]
        self.assertEqual(tuple(output.shape), (3, 4, 6, 3))
        self.assertEqual(float(output.max()), 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_mask_nodes_match_on_cuda(self):
        images = torch.rand(7, 8, 9, 3)
        source = torch.rand(3, 5, 6)
        cpu = GiftMaskFadeInOut().generate(images, 3, source)[1]
        gpu = GiftMaskFadeInOut().generate(images.cuda(), 3, source.cuda())[1].cpu()
        self.assertTrue(torch.allclose(cpu, gpu, atol=1.0e-6, rtol=1.0e-6))


if __name__ == "__main__":
    unittest.main()
