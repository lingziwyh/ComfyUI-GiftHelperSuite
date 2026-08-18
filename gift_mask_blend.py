"""Vectorized mask and image-sequence helpers for GiftHelperSuite."""

from __future__ import annotations

import torch
import torch.nn.functional as F


MASK_CATEGORY = "GiftHelperSuite/Mask"
SEQUENCE_CATEGORY = "GiftHelperSuite/Sequence"


def _validate_image(images: torch.Tensor, name: str) -> tuple[int, int, int, int]:
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        shape = getattr(images, "shape", None)
        raise ValueError(f"{name} must be an IMAGE tensor [B,H,W,C], got {shape}")
    batch, height, width, channels = images.shape
    if batch < 1 or height < 1 or width < 1 or channels < 1:
        raise ValueError(f"{name} cannot contain an empty dimension: {tuple(images.shape)}")
    return batch, height, width, channels


def _frame_range(batch: int, start_frame: int, end_frame: int) -> tuple[int, int]:
    start = max(0, min(int(start_frame), batch - 1))
    end = max(0, min(int(end_frame), batch - 1))
    return (start, end) if start <= end else (end, start)


def _mask_to_bhw(
    mask: torch.Tensor,
    *,
    batch: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise ValueError("mask must be a torch.Tensor")
    if mask.ndim == 2:
        value = mask.unsqueeze(0)
    elif mask.ndim == 3:
        value = mask
    elif mask.ndim == 4 and mask.shape[1] == 1:
        value = mask[:, 0]
    elif mask.ndim == 4 and mask.shape[-1] == 1:
        value = mask[..., 0]
    else:
        raise ValueError(
            "mask must have shape [H,W], [B,H,W], [B,1,H,W], or [B,H,W,1]; "
            f"got {tuple(mask.shape)}"
        )
    if value.shape[0] < 1:
        raise ValueError("mask batch cannot be empty")

    work_dtype = dtype if dtype.is_floating_point else torch.float32
    value = value.to(device=device, dtype=work_dtype)
    if value.shape[0] < batch:
        tail = value[-1:].expand(batch - value.shape[0], -1, -1)
        value = torch.cat((value, tail), dim=0)
    elif value.shape[0] > batch:
        value = value[:batch]

    if value.shape[-2:] != (height, width):
        value = F.interpolate(
            value.unsqueeze(1),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
    return value.clamp(0.0, 1.0)


def _align_image_like(images: torch.Tensor, reference: torch.Tensor, name: str) -> torch.Tensor:
    _, target_height, target_width, target_channels = _validate_image(reference, "reference")
    _, height, width, channels = _validate_image(images, name)
    dtype = reference.dtype if reference.dtype.is_floating_point else torch.float32
    value = images.to(device=reference.device, dtype=dtype)

    if channels != target_channels:
        if channels == 4 and target_channels == 3:
            value = value[..., :3]
        elif channels == 3 and target_channels == 4:
            alpha = torch.ones((*value.shape[:-1], 1), device=value.device, dtype=value.dtype)
            value = torch.cat((value, alpha), dim=-1)
        elif channels == 1 and target_channels in (3, 4):
            rgb = value.expand(-1, -1, -1, 3)
            if target_channels == 4:
                alpha = torch.ones((*rgb.shape[:-1], 1), device=rgb.device, dtype=rgb.dtype)
                value = torch.cat((rgb, alpha), dim=-1)
            else:
                value = rgb
        else:
            raise ValueError(
                f"{name} has {channels} channels but the first sequence has "
                f"{target_channels}; only 1/3/4-channel alignment is supported"
            )

    if (height, width) != (target_height, target_width):
        value = F.interpolate(
            value.permute(0, 3, 1, 2),
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1).contiguous()
    return value


class GiftMaskRamp:
    """Generate a temporal 0-to-1 mask ramp for an image sequence."""

    CATEGORY = MASK_CATEGORY
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "mask")
    DESCRIPTION = "开始帧前为 0，起止帧间渐变，结束帧后持续为 1。"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "end_frame": ("INT", {"default": 10, "min": 0, "max": 100000, "step": 1}),
            }
        }

    def generate(self, images, start_frame, end_frame):
        batch, height, width, _ = _validate_image(images, "images")
        start, end = _frame_range(batch, start_frame, end_frame)
        frame = torch.arange(batch, device=images.device, dtype=torch.float32)
        if start == end:
            levels = (frame >= start).to(torch.float32)
        else:
            levels = ((frame - start) / float(end - start)).clamp(0.0, 1.0)
        mask = levels.view(batch, 1, 1).expand(batch, height, width)
        return images, mask


class GiftMaskFadeInOut:
    """Apply a symmetric temporal fade to an optional input mask."""

    CATEGORY = MASK_CATEGORY
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "mask")
    DESCRIPTION = "为序列首尾生成对称透明过渡；输入 Mask 不会被原地修改。"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fade_frames": ("INT", {"default": 10, "min": 0, "max": 100000, "step": 1}),
            },
            "optional": {"input_mask": ("MASK",)},
        }

    def generate(self, images, fade_frames, input_mask=None):
        batch, height, width, _ = _validate_image(images, "images")
        if input_mask is None:
            mask = torch.ones((batch, height, width), device=images.device, dtype=torch.float32)
        else:
            mask = _mask_to_bhw(
                input_mask,
                batch=batch,
                height=height,
                width=width,
                device=images.device,
                dtype=torch.float32,
            )

        count = max(0, min(int(fade_frames), (batch + 1) // 2))
        if count == 0:
            return images, mask

        frame = torch.arange(batch, device=images.device, dtype=torch.float32)
        denominator = float(max(1, count - 1))
        fade_in = (frame / denominator).clamp(0.0, 1.0)
        fade_out = ((batch - 1 - frame) / denominator).clamp(0.0, 1.0)
        envelope = torch.minimum(fade_in, fade_out)
        return images, mask * envelope.view(batch, 1, 1)


class GiftFrameSlice:
    """Take an inclusive frame range from an IMAGE batch."""

    CATEGORY = SEQUENCE_CATEGORY
    FUNCTION = "slice"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "end_frame": ("INT", {"default": 10, "min": 0, "max": 100000, "step": 1}),
            }
        }

    def slice(self, images, start_frame, end_frame):
        batch, _, _, _ = _validate_image(images, "images")
        start, end = _frame_range(batch, start_frame, end_frame)
        return (images[start : end + 1],)


class GiftMaskBlend:
    """Blend two image sequences using the first sequence's temporal mask."""

    CATEGORY = SEQUENCE_CATEGORY
    FUNCTION = "blend"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    DESCRIPTION = (
        "第一段在指定帧后与第二段从头融合；短 Mask 自动重复末帧，"
        "第二段会自动对齐到第一段尺寸。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "first_images": ("IMAGE",),
                "second_images": ("IMAGE",),
                "mask": ("MASK",),
                "blend_start": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
            }
        }

    def blend(self, first_images, second_images, mask, blend_start):
        first_len, height, width, _ = _validate_image(first_images, "first_images")
        second_len, _, _, _ = _validate_image(second_images, "second_images")
        second = _align_image_like(second_images, first_images, "second_images")
        work_dtype = first_images.dtype if first_images.dtype.is_floating_point else torch.float32
        first = first_images.to(dtype=work_dtype)
        alpha = _mask_to_bhw(
            mask,
            batch=first_len,
            height=height,
            width=width,
            device=first.device,
            dtype=work_dtype,
        ).unsqueeze(-1)

        start = max(0, min(int(blend_start), first_len))
        first_remaining = first_len - start
        overlap = min(first_remaining, second_len)
        parts = []
        if start:
            parts.append(first[:start])
        if overlap:
            foreground = first[start : start + overlap]
            background = second[:overlap]
            opacity = alpha[start : start + overlap]
            parts.append(foreground * opacity + background * (1.0 - opacity))
        if first_remaining > second_len:
            foreground = first[start + overlap :]
            opacity = alpha[start + overlap :]
            parts.append(foreground * opacity)
        elif second_len > first_remaining:
            parts.append(second[overlap:])
        return (torch.cat(parts, dim=0),)


NODE_CLASS_MAPPINGS = {
    "GiftMaskRamp": GiftMaskRamp,
    "GiftMaskFadeInOut": GiftMaskFadeInOut,
    "GiftFrameSlice": GiftFrameSlice,
    "GiftMaskBlend": GiftMaskBlend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GiftMaskRamp": "Gift Mask Ramp · 遮罩渐变",
    "GiftMaskFadeInOut": "Gift Mask Fade In-Out · 首尾淡入淡出",
    "GiftFrameSlice": "Gift Frame Slice · 帧切片",
    "GiftMaskBlend": "Gift Mask Blend · 序列融合",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
