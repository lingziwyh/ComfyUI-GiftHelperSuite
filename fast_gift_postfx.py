
import math
from typing import Tuple

import torch
import torch.nn.functional as F


def _to_bchw(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 4:
        raise ValueError(f"Expected image tensor with 4 dims [B,H,W,C], got shape {tuple(image.shape)}")
    return image.permute(0, 3, 1, 2).contiguous()


def _to_bhwc(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 4:
        raise ValueError(f"Expected image tensor with 4 dims [B,C,H,W], got shape {tuple(image.shape)}")
    return image.permute(0, 2, 3, 1).contiguous()


def _match_batch(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    ba, bb = a.shape[0], b.shape[0]
    if ba == bb:
        return a, b
    if ba == 1:
        return a.expand(bb, *a.shape[1:]), b
    if bb == 1:
        return a, b.expand(ba, *b.shape[1:])
    raise ValueError(f"Batch mismatch: {ba} vs {bb}. One input batch must be 1 or both batches equal.")


def _box_blur(x: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return x
    k = radius * 2 + 1
    # horizontal
    x = F.avg_pool2d(x, kernel_size=(1, k), stride=1, padding=(0, radius), count_include_pad=False)
    # vertical
    x = F.avg_pool2d(x, kernel_size=(k, 1), stride=1, padding=(radius, 0), count_include_pad=False)
    return x


def _approx_gaussian_blur(x: torch.Tensor, radius: int, passes: int = 2) -> torch.Tensor:
    if radius <= 0:
        return x
    out = x
    for _ in range(max(1, passes)):
        out = _box_blur(out, radius)
    return out


def _rgb_to_luma(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] < 3:
        return x.mean(dim=1, keepdim=True)
    r = x[:, 0:1]
    g = x[:, 1:2]
    b = x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def _make_highlight_mask(x: torch.Tensor, threshold: float, knee: float = 0.1) -> torch.Tensor:
    luma = _rgb_to_luma(x)
    lo = max(0.0, threshold - knee)
    hi = min(1.0, threshold + knee)
    if hi <= lo:
        return (luma >= threshold).to(x.dtype)
    return ((luma - lo) / (hi - lo)).clamp(0.0, 1.0)


def _screen(base: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    return 1.0 - (1.0 - base) * (1.0 - blend)


def _shift_channel_2d(channel: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    # channel: [B,1,H,W]
    if dx == 0 and dy == 0:
        return channel
    b, c, h, w = channel.shape
    pad_left = max(dx, 0)
    pad_right = max(-dx, 0)
    pad_top = max(dy, 0)
    pad_bottom = max(-dy, 0)
    padded = F.pad(channel, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")
    x0 = pad_right
    y0 = pad_bottom
    return padded[:, :, y0:y0+h, x0:x0+w]


class FastGiftPostFX:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "enable_bloom": ("BOOLEAN", {"default": True}),
                "bloom_threshold": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bloom_intensity": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 3.0, "step": 0.01}),
                "bloom_radius": ("INT", {"default": 8, "min": 0, "max": 64, "step": 1}),
                "bloom_downsample": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1}),
                "enable_ca": ("BOOLEAN", {"default": True}),
                "ca_amount": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "ca_angle_deg": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "ca_highlight_only": ("BOOLEAN", {"default": False}),
                "enable_sharpen": ("BOOLEAN", {"default": True}),
                "sharpen_amount": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 2.0, "step": 0.01}),
                "sharpen_radius": ("INT", {"default": 1, "min": 0, "max": 8, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_fx"
    CATEGORY = "image/postprocessing"

    def apply_fx(
        self,
        image: torch.Tensor,
        enable_bloom: bool,
        bloom_threshold: float,
        bloom_intensity: float,
        bloom_radius: int,
        bloom_downsample: int,
        enable_ca: bool,
        ca_amount: float,
        ca_angle_deg: float,
        ca_highlight_only: bool,
        enable_sharpen: bool,
        sharpen_amount: float,
        sharpen_radius: int,
    ):
        if image.ndim != 4:
            raise ValueError(f"Expected IMAGE tensor [B,H,W,C], got {tuple(image.shape)}")

        x = _to_bchw(image).float().clamp(0.0, 1.0)
        original = x

        # 1) Bloom: threshold -> downsample -> blur -> upsample -> screen/add-ish blend
        if enable_bloom and bloom_intensity > 0.0 and bloom_radius > 0:
            highlight = _make_highlight_mask(x, bloom_threshold, knee=0.08)
            bloom_src = x * highlight

            ds = max(1, int(bloom_downsample))
            if ds > 1:
                h, w = x.shape[-2:]
                small_h = max(1, round(h / ds))
                small_w = max(1, round(w / ds))
                bloom_small = F.interpolate(bloom_src, size=(small_h, small_w), mode="bilinear", align_corners=False)
                small_radius = max(1, round(bloom_radius / ds))
            else:
                bloom_small = bloom_src
                small_radius = bloom_radius

            bloom_small = _approx_gaussian_blur(bloom_small, small_radius, passes=2)

            if ds > 1:
                bloom = F.interpolate(bloom_small, size=x.shape[-2:], mode="bilinear", align_corners=False)
            else:
                bloom = bloom_small

            # Mixed blend: screen keeps highlights pleasant, intensity scales effect.
            x = _screen(x, bloom * bloom_intensity)
            x = x.clamp(0.0, 1.0)

        # 2) Sharpen: unsharp mask
        if enable_sharpen and sharpen_amount > 0.0 and sharpen_radius > 0:
            blur = _approx_gaussian_blur(x, sharpen_radius, passes=1)
            high = x - blur
            x = (x + high * sharpen_amount).clamp(0.0, 1.0)

        # 3) Chromatic aberration: fast channel shift
        if enable_ca and ca_amount > 0.0 and x.shape[1] >= 3:
            rad = math.radians(ca_angle_deg)
            dx = int(round(math.cos(rad) * ca_amount))
            dy = int(round(math.sin(rad) * ca_amount))

            r = _shift_channel_2d(x[:, 0:1], dx, dy)
            g = x[:, 1:2]
            b = _shift_channel_2d(x[:, 2:3], -dx, -dy)

            ca = torch.cat([r, g, b], dim=1)
            if x.shape[1] > 3:
                ca = torch.cat([ca, x[:, 3:]], dim=1)

            if ca_highlight_only:
                mask = _make_highlight_mask(original, max(0.35, bloom_threshold if enable_bloom else 0.6), knee=0.12)
                x = x * (1.0 - mask) + ca * mask
            else:
                x = ca
            x = x.clamp(0.0, 1.0)

        return (_to_bhwc(x),)


NODE_CLASS_MAPPINGS = {
    "FastGiftPostFX": FastGiftPostFX,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FastGiftPostFX": "Fast Gift PostFX",
}
