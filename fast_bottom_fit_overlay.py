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
    - Optional rounded-rectangle feather mask that follows the resized layer
      aspect ratio, becomes an inscribed ellipse at maximum corner radius,
      and is mutually exclusive with the top feather fade
    - Optional packed output:
        left  = final mask (RGB)
        right = masked foreground on black background
      Fit Content keeps the resized layer aspect ratio and does NOT pad to the
      background height. Its size is [B, resized_h, background_w * 2, 3].
      Fixed Canvas (1440x1280) uses two 720px-wide panels, bottom-aligns the
      packed content, and pads the top with black pixels.

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
                "packed_size_mode": (
                    ["Fit Content (Dynamic Height)", "Fixed Canvas (1440x1280)"],
                    {"default": "Fit Content (Dynamic Height)"},
                ),
                "enable_rounded_rect_fade": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "rounded_rect_fade_ratio": (
                    "FLOAT",
                    {
                        "default": 0.16,
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.005,
                        "tooltip": "Inward feather width; 0.16 is about 8% of the layer's full width and height.",
                    },
                ),
                "rounded_corner_radius": (
                    "FLOAT",
                    {
                        "default": 0.30,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "0 is a rectangle; 1 becomes an ellipse fitted to all four layer edges.",
                    },
                ),
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

    def _make_rounded_rect_fade_mask(
        self,
        batch: int,
        height: int,
        width: int,
        device,
        dtype,
        fade_ratio: float,
        corner_radius_ratio: float,
    ) -> torch.Tensor:
        """Return a rounded rectangle that continuously becomes an ellipse."""
        fade_ratio = float(max(0.0, min(0.5, fade_ratio)))
        corner_radius_ratio = float(max(0.0, min(1.0, corner_radius_ratio)))
        radius_y = float(height) * 0.5
        radius_x = float(width) * 0.5

        y = (torch.arange(height, device=device, dtype=dtype) + 0.5 - radius_y) / radius_y
        x = (torch.arange(width, device=device, dtype=dtype) + 0.5 - radius_x) / radius_x
        abs_y = y[:, None].abs()
        abs_x = x[None, :].abs()

        # Signed distance in normalized layer coordinates. At radius 0 this is
        # a rectangle; at radius 1 the straight sections vanish into an ellipse.
        straight_extent = 1.0 - corner_radius_ratio
        q_y = abs_y - straight_extent
        q_x = abs_x - straight_extent
        outside_distance = torch.sqrt(q_y.clamp_min(0.0).square() + q_x.clamp_min(0.0).square())
        inside_distance = torch.minimum(torch.maximum(q_y, q_x), torch.zeros_like(q_y + q_x))
        signed_distance = outside_distance + inside_distance - corner_radius_ratio

        if fade_ratio <= 0.0:
            rounded_rect = (signed_distance <= 0.0).to(dtype=dtype)
        else:
            rounded_rect = (-signed_distance / fade_ratio).clamp(0.0, 1.0)

        return rounded_rect.view(1, 1, height, width).expand(batch, 1, height, width)

    def _apply_fade_mask(
        self,
        alpha: torch.Tensor,
        enable_top_fade: bool,
        top_fade_ratio: float,
        enable_rounded_rect_fade: bool,
        rounded_rect_fade_ratio: float,
        rounded_corner_radius: float,
    ) -> torch.Tensor:
        batch, _, height, width = alpha.shape
        if enable_top_fade:
            return alpha * self._make_top_fade_mask(
                batch,
                height,
                width,
                alpha.device,
                alpha.dtype,
                top_fade_ratio,
            )
        if enable_rounded_rect_fade:
            return alpha * self._make_rounded_rect_fade_mask(
                batch,
                height,
                width,
                alpha.device,
                alpha.dtype,
                rounded_rect_fade_ratio,
                rounded_corner_radius,
            )
        return alpha

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
        packed_size_mode="Fit Content (Dynamic Height)",
        enable_rounded_rect_fade=False,
        rounded_rect_fade_ratio=0.16,
        rounded_corner_radius=0.30,
    ):
        bg = self._ensure_image(background_image)
        fg = self._ensure_image(layer_image)

        if enable_top_fade and enable_rounded_rect_fade:
            raise ValueError(
                "Top fade and rounded-rectangle fade are mutually exclusive. Enable only one fade mode."
            )

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

        alpha = self._apply_fade_mask(
            alpha,
            enable_top_fade,
            top_fade_ratio,
            enable_rounded_rect_fade,
            rounded_rect_fade_ratio,
            rounded_corner_radius,
        )

        alpha = (alpha * float(opacity)).clamp(0.0, 1.0)

        # Packed output is either content-sized or bottom-aligned on a fixed canvas.
        if enable_packed_output:
            packed_fg_resized = fg_resized
            packed_alpha = alpha

            if packed_size_mode == "Fixed Canvas (1440x1280)" and packed_w != 720:
                fixed_scale = 720.0 / float(fg_w)
                fixed_h = max(1, int(round(fg_h * fixed_scale)))
                packed_fg_resized = F.interpolate(
                    fg_bchw,
                    size=(fixed_h, 720),
                    mode="bilinear",
                    align_corners=False,
                )
                packed_alpha = self._ensure_mask(layer_mask, batch, fg_h, fg_w, device, dtype)
                packed_alpha = F.interpolate(
                    packed_alpha,
                    size=(fixed_h, 720),
                    mode="bilinear",
                    align_corners=False,
                )
                packed_alpha = self._apply_fade_mask(
                    packed_alpha,
                    enable_top_fade,
                    top_fade_ratio,
                    enable_rounded_rect_fade,
                    rounded_rect_fade_ratio,
                    rounded_corner_radius,
                )
                packed_alpha = (packed_alpha * float(opacity)).clamp(0.0, 1.0)

            packed_mask = (
                packed_alpha[:, 0:1, :, :]
                .repeat(1, 3, 1, 1)
                .permute(0, 2, 3, 1)
                .contiguous()
            )
            packed_fg = (packed_fg_resized * packed_alpha).permute(0, 2, 3, 1).contiguous()
            packed = torch.cat([packed_mask, packed_fg], dim=2).clamp(0.0, 1.0)

            if packed_size_mode == "Fixed Canvas (1440x1280)":
                if packed.shape[1] > 1280:
                    packed = packed[:, packed.shape[1] - 1280 :, :, :]
                elif packed.shape[1] < 1280:
                    fixed_canvas = packed.new_zeros((batch, 1280, 1440, 3))
                    fixed_canvas[:, 1280 - packed.shape[1] :, :, :] = packed
                    packed = fixed_canvas
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
