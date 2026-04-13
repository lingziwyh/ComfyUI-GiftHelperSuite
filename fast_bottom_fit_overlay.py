import torch
import torch.nn.functional as F


class FastBottomFitOverlay:
    """
    Fast ComfyUI node:
    - Resize layer_image to match background width exactly
    - Keep aspect ratio
    - Bottom-align the resized layer onto the background
    - Center horizontally
    - Support IMAGE batches efficiently with torch ops
    - Optional MASK for alpha compositing
    - Optional built-in top feather fade to soften top-edge cutoffs
    - Optional packed output:
        left  = final mask (RGB)
        right = masked foreground on black background
      Packed output keeps the resized layer aspect ratio and does NOT pad to
      the background height. Its size is [B, resized_h, background_w * 2, 3].

    IMAGE format in ComfyUI is expected to be [B, H, W, C], float32/float16 in [0, 1]
    MASK format is expected to be [B, H, W] or [H, W]
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background_image": ("IMAGE",),
                "layer_image": ("IMAGE",),
                "opacity": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "clip_if_too_tall": (
                    "BOOLEAN",
                    {"default": True},
                ),
                "enable_top_fade": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "top_fade_ratio": (
                    "FLOAT",
                    {
                        "default": 0.08,
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.005,
                    },
                ),
                "enable_packed_output": (
                    "BOOLEAN",
                    {"default": True},
                ),
            },
            "optional": {
                "layer_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("image", "mask", "packed_image")
    FUNCTION = "composite"
    CATEGORY = "image/composite"

    def _ensure_image(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() != 4:
            raise ValueError(f"Expected IMAGE tensor with shape [B,H,W,C], got {tuple(image.shape)}")
        return image

    def _ensure_mask(self, mask: torch.Tensor, batch: int, height: int, width: int, device, dtype) -> torch.Tensor:
        if mask is None:
            return torch.ones((batch, 1, height, width), device=device, dtype=dtype)

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        elif mask.dim() != 3:
            raise ValueError(f"Expected MASK tensor with shape [H,W] or [B,H,W], got {tuple(mask.shape)}")

        if mask.shape[0] == 1 and batch > 1:
            mask = mask.expand(batch, -1, -1)
        elif mask.shape[0] != batch:
            raise ValueError(
                f"Mask batch size {mask.shape[0]} does not match resolved image batch size {batch}. "
                f"Use mask batch 1 or the same batch size as the images."
            )

        mask = mask.unsqueeze(1).to(device=device, dtype=dtype)
        if mask.shape[-2:] != (height, width):
            mask = F.interpolate(mask, size=(height, width), mode="bilinear", align_corners=False)
        return mask.clamp(0.0, 1.0)

    def _broadcast_batch(self, bg: torch.Tensor, fg: torch.Tensor):
        b_bg = bg.shape[0]
        b_fg = fg.shape[0]

        if b_bg == b_fg:
            return bg, fg, b_bg
        if b_bg == 1 and b_fg > 1:
            return bg.expand(b_fg, -1, -1, -1), fg, b_fg
        if b_fg == 1 and b_bg > 1:
            return bg, fg.expand(b_bg, -1, -1, -1), b_bg

        raise ValueError(
            f"Incompatible batch sizes: background={b_bg}, layer={b_fg}. "
            f"One input must have batch 1, or both must match."
        )

    def _make_top_fade_mask(self, batch: int, height: int, width: int, device, dtype, fade_ratio: float) -> torch.Tensor:
        fade_ratio = float(max(0.0, min(0.5, fade_ratio)))
        if fade_ratio <= 0.0 or height <= 1:
            return torch.ones((batch, 1, height, width), device=device, dtype=dtype)

        fade_h = max(1, int(round(height * fade_ratio)))
        fade_h = min(fade_h, height)

        ramp = torch.ones((height,), device=device, dtype=dtype)
        ramp[:fade_h] = torch.linspace(0.0, 1.0, steps=fade_h, device=device, dtype=dtype)

        return ramp.view(1, 1, height, 1).expand(batch, 1, height, width)

    def composite(
        self,
        background_image,
        layer_image,
        opacity=1.0,
        clip_if_too_tall=True,
        enable_top_fade=False,
        top_fade_ratio=0.08,
        enable_packed_output=True,
        layer_mask=None,
    ):
        bg = self._ensure_image(background_image)
        fg = self._ensure_image(layer_image)

        bg, fg, batch = self._broadcast_batch(bg, fg)

        device = bg.device
        dtype = bg.dtype

        bg_h, bg_w = bg.shape[1], bg.shape[2]
        fg_h, fg_w = fg.shape[1], fg.shape[2]

        if fg_w <= 0 or fg_h <= 0 or bg_w <= 0 or bg_h <= 0:
            raise ValueError("Invalid image dimensions.")

        # Resize layer to exactly match background width, preserving aspect ratio.
        packed_w = bg_w
        scale = float(packed_w) / float(fg_w)
        resized_h = max(1, int(round(fg_h * scale)))

        fg_bchw = fg.permute(0, 3, 1, 2).contiguous()
        fg_resized = F.interpolate(fg_bchw, size=(resized_h, packed_w), mode="bilinear", align_corners=False)

        alpha = self._ensure_mask(layer_mask, batch, fg_h, fg_w, device, dtype)
        alpha = F.interpolate(alpha, size=(resized_h, packed_w), mode="bilinear", align_corners=False)

        if enable_top_fade:
            top_fade = self._make_top_fade_mask(batch, resized_h, packed_w, device, dtype, top_fade_ratio)
            alpha = alpha * top_fade

        alpha = (alpha * float(opacity)).clamp(0.0, 1.0)

        # Packed output keeps the resized layer aspect ratio and does not pad to background height.
        if enable_packed_output:
            packed_mask = alpha[:, 0:1, :, :].repeat(1, 3, 1, 1).permute(0, 2, 3, 1).contiguous()
            packed_fg = (fg_resized * alpha).permute(0, 2, 3, 1).contiguous()
            packed = torch.cat([packed_mask, packed_fg], dim=2).clamp(0.0, 1.0)
        else:
            packed = bg.clone()

        # Preview outputs remain constrained by background canvas.
        preview_h = resized_h
        fg_preview = fg_resized
        alpha_preview = alpha

        if preview_h > bg_h:
            if not clip_if_too_tall:
                raise ValueError(
                    f"Resized layer height ({preview_h}) exceeds background height ({bg_h}). "
                    f"Enable clip_if_too_tall to crop from the top automatically."
                )
            fg_preview = fg_preview[:, :, preview_h - bg_h :, :]
            alpha_preview = alpha_preview[:, :, preview_h - bg_h :, :]
            preview_h = bg_h

        y0 = bg_h - preview_h
        y1 = bg_h
        x0 = 0
        x1 = bg_w

        out = bg.clone()
        out_region = out[:, y0:y1, x0:x1, :]

        fg_region = fg_preview.permute(0, 2, 3, 1).contiguous()
        a_region = alpha_preview.permute(0, 2, 3, 1).contiguous()

        out[:, y0:y1, x0:x1, :] = fg_region * a_region + out_region * (1.0 - a_region)

        full_mask = torch.zeros((batch, bg_h, bg_w), device=device, dtype=dtype)
        full_mask[:, y0:y1, x0:x1] = alpha_preview[:, 0, :, :]

        return (out.clamp(0.0, 1.0), full_mask.clamp(0.0, 1.0), packed)


NODE_CLASS_MAPPINGS = {
    "FastBottomFitOverlay": FastBottomFitOverlay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FastBottomFitOverlay": "Fast Bottom Fit Overlay (Packed)",
}
