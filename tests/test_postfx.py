from __future__ import annotations

import pathlib
import sys
import unittest

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import fast_gift_postfx as postfx  # noqa: E402


OPTIONS = {
    "enable_bloom": True,
    "bloom_threshold": 0.65,
    "bloom_intensity": 0.35,
    "bloom_radius": 8,
    "bloom_downsample": 4,
    "enable_ca": True,
    "ca_amount": 2.0,
    "ca_angle_deg": 17.0,
    "ca_highlight_only": True,
    "enable_sharpen": True,
    "sharpen_amount": 0.2,
    "sharpen_radius": 1,
    "natural_saturation": 0.15,
    "saturation": 1.1,
    "contrast": 1.05,
    "brightness": 0.01,
}


class FastGiftPostFXTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.image = torch.rand(5, 31, 37, 3)

    def test_existing_required_contract_is_unchanged(self):
        schema = postfx.FastGiftPostFX.INPUT_TYPES()
        self.assertEqual(
            list(schema["required"]),
            [
                "image", "enable_bloom", "bloom_threshold", "bloom_intensity",
                "bloom_radius", "bloom_downsample", "enable_ca", "ca_amount",
                "ca_angle_deg", "ca_highlight_only", "enable_sharpen",
                "sharpen_amount", "sharpen_radius", "natural_saturation",
                "saturation", "contrast", "brightness",
            ],
        )
        self.assertEqual(set(schema["optional"]), {"performance_mode", "gpu_chunk_size"})

    def test_cpu_chunking_is_bit_exact_with_full_batch(self):
        expected = postfx._apply_fx_batch(self.image, **OPTIONS)
        actual = postfx._run_chunked(
            self.image,
            processing_device=torch.device("cpu"),
            output_device=torch.device("cpu"),
            chunk_size=2,
            options=OPTIONS,
        )
        self.assertTrue(torch.equal(expected, actual))

    def test_public_cpu_path_preserves_shape_device_and_float_output(self):
        output = postfx.FastGiftPostFX().apply_fx(
            self.image,
            **OPTIONS,
            performance_mode="cpu",
            gpu_chunk_size=2,
        )[0]
        self.assertEqual(tuple(output.shape), tuple(self.image.shape))
        self.assertEqual(output.device, self.image.device)
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(output.is_contiguous())
        self.assertGreaterEqual(float(output.min()), 0.0)
        self.assertLessEqual(float(output.max()), 1.0)

    def test_input_is_not_modified(self):
        before = self.image.clone()
        postfx.FastGiftPostFX().apply_fx(
            self.image,
            **OPTIONS,
            performance_mode="cpu",
            gpu_chunk_size=1,
        )
        self.assertTrue(torch.equal(self.image, before))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_matches_cpu_and_returns_to_input_device(self):
        cpu = postfx.FastGiftPostFX().apply_fx(
            self.image,
            **OPTIONS,
            performance_mode="cpu",
            gpu_chunk_size=2,
        )[0]
        gpu = postfx.FastGiftPostFX().apply_fx(
            self.image,
            **OPTIONS,
            performance_mode="cuda",
            gpu_chunk_size=2,
        )[0]
        self.assertEqual(gpu.device, self.image.device)
        self.assertTrue(torch.allclose(cpu, gpu, atol=1.0e-5, rtol=1.0e-5))


if __name__ == "__main__":
    unittest.main()
