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
    - Optional restored foreground IMAGE + MASK composited above the faded layer
    - Optional symmetric temporal fade applied to the complete composited stack;
      background_image is never faded
    - Optional built-in top feather fade applied to the complete restored stack
      to soften top-edge cutoffs
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

    PRESET_OPTIONS = (
        "Custom",
        "Low Coins",
        "Standard",
        "Hybrid Naked-Eye 3D",
        "Naked-Eye 3D",
    )
    FADE_MODE_OPTIONS = ("None", "Top Fade", "Rounded Rect")
    PRESET_VALUES = {
        "Low Coins": {
            "fade_mode": "Rounded Rect",
            "background_fade_ratio": 0.0,
            "rounded_rect_fade_ratio": 0.255,
            "rounded_corner_radius": 0.90,
        },
        "Standard": {
            "fade_mode": "Top Fade",
            "top_fade_ratio": 0.08,
            "background_fade_ratio": 0.0,
        },
        "Hybrid Naked-Eye 3D": {
            "fade_mode": "None",
            "background_fade_ratio": 0.0,
        },
        "Naked-Eye 3D": {
            "fade_mode": "Top Fade",
            "top_fade_ratio": 0.045,
            "background_fade_ratio": 0.52,
        },
    }

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
                    {
                        "default": True,
                        "advanced": True,
                        "tooltip": "安全裁切：图层适配后高于背景时从顶部裁掉溢出部分。通常保持开启。",
                    },
                ),
                "enable_packed_output": (
                    "BOOLEAN",
                    {"default": True},
                ),
            },
            "optional": {
                "layer_mask": ("MASK",),
                "foreground_image": ("IMAGE",),
                "foreground_mask": ("MASK",),
                "fade_frames": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 100000,
                        "step": 1,
                        "tooltip": "首尾对称淡入淡出帧数；作用于除 background_image 外的完整合成结果，0 为关闭。",
                    },
                ),
                "packed_size_mode": (
                    ["Fit Content (Dynamic Height)", "Fixed Canvas (1440x1280)"],
                    {"default": "Fit Content (Dynamic Height)"},
                ),
                "preset": (
                    list(cls.PRESET_OPTIONS),
                    {
                        "default": "Custom",
                        "tooltip": "Production presets override their configured controls; Custom uses the manual values below.",
                    },
                ),
                "fade_mode": (
                    list(cls.FADE_MODE_OPTIONS),
                    {
                        "default": "None",
                        "tooltip": "空间羽化类型；Top Fade 与 Rounded Rect 互斥。",
                    },
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
                "background_fade_ratio": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Large top-to-bottom fade applied only to layer_image; 0 disables it.",
                    },
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
                "center_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "中心缩放比例：以贴底后视频区域的中心缩放；1 为原大小，0 为完全消失。输出画布尺寸不变。",
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

    def _broadcast_foreground(self, bg: torch.Tensor, layer: torch.Tensor, foreground: torch.Tensor):
        target_batch = max(bg.shape[0], layer.shape[0], foreground.shape[0])
        images = []
        for name, image in (("background", bg), ("layer", layer), ("foreground", foreground)):
            if image.shape[0] == target_batch:
                images.append(image)
            elif image.shape[0] == 1:
                images.append(image.expand(target_batch, -1, -1, -1))
            else:
                raise ValueError(
                    f"Incompatible batch sizes: background={bg.shape[0]}, layer={layer.shape[0]}, "
                    f"foreground={foreground.shape[0]}. Each input must have batch 1 or {target_batch}."
                )
        return images[0], images[1], images[2], target_batch

    def _combine_premultiplied_layers(
        self,
        layer_rgb: torch.Tensor,
        layer_alpha: torch.Tensor,
        foreground_rgb: torch.Tensor = None,
        foreground_alpha: torch.Tensor = None,
    ):
        """Return premultiplied RGB and alpha, restoring foreground above layer."""
        layer_premultiplied = layer_rgb * layer_alpha
        if foreground_rgb is None or foreground_alpha is None:
            return layer_premultiplied, layer_alpha

        inverse_foreground = 1.0 - foreground_alpha
        combined_rgb = foreground_rgb * foreground_alpha + layer_premultiplied * inverse_foreground
        combined_alpha = foreground_alpha + layer_alpha * inverse_foreground
        return combined_rgb, combined_alpha.clamp(0.0, 1.0)

    def _make_top_fade_mask(self, batch: int, height: int, width: int, device, dtype, fade_ratio: float) -> torch.Tensor:
        fade_ratio = float(max(0.0, min(1.0, fade_ratio)))
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
        signed_distance = self._make_normalized_rounded_rect_signed_distance(
            height,
            width,
            device,
            dtype,
            corner_radius_ratio,
        )

        if fade_ratio <= 0.0:
            rounded_rect = (signed_distance <= 0.0).to(dtype=dtype)
        else:
            rounded_rect = (-signed_distance / fade_ratio).clamp(0.0, 1.0)

        return rounded_rect.view(1, 1, height, width).expand(batch, 1, height, width)

    def _make_normalized_rounded_rect_signed_distance(
        self,
        height: int,
        width: int,
        device,
        dtype,
        corner_radius_ratio,
    ) -> torch.Tensor:
        """Return an aspect-ratio-aware rounded-rectangle distance field."""
        radius_y = float(height) * 0.5
        radius_x = float(width) * 0.5

        y = (torch.arange(height, device=device, dtype=dtype) + 0.5 - radius_y) / radius_y
        x = (torch.arange(width, device=device, dtype=dtype) + 0.5 - radius_x) / radius_x
        abs_y = y[:, None].abs()
        abs_x = x[None, :].abs()

        # Signed distance in normalized layer coordinates. At radius 0 this is
        # a rectangle; at radius 1 the straight sections vanish into an ellipse.
        corner_radius_ratio = torch.as_tensor(
            corner_radius_ratio,
            device=device,
            dtype=dtype,
        ).clamp(0.0, 1.0)
        straight_extent = 1.0 - corner_radius_ratio
        q_y = abs_y - straight_extent
        q_x = abs_x - straight_extent
        outside_distance = torch.sqrt(q_y.clamp_min(0.0).square() + q_x.clamp_min(0.0).square())
        inside_distance = torch.minimum(torch.maximum(q_y, q_x), torch.zeros_like(q_y + q_x))
        return outside_distance + inside_distance - corner_radius_ratio

    def _apply_fade_mask(
        self,
        alpha: torch.Tensor,
        fade_mode: str,
        top_fade_ratio: float,
        rounded_rect_fade_ratio: float,
        rounded_corner_radius: float,
    ) -> torch.Tensor:
        batch, _, height, width = alpha.shape
        if fade_mode == "Top Fade":
            return alpha * self._make_top_fade_mask(
                batch,
                height,
                width,
                alpha.device,
                alpha.dtype,
                top_fade_ratio,
            )
        if fade_mode == "Rounded Rect":
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

    def _make_hybrid_scene_mask(
        self,
        batch: int,
        height: int,
        width: int,
        device,
        dtype,
    ) -> torch.Tensor:
        """Return the broad rounded-rectangle scene window used by the hybrid preset."""
        radius_y = float(height) * 0.5
        radius_x = float(width) * 0.5
        y = (torch.arange(height, device=device, dtype=dtype) + 0.5 - radius_y) / radius_y
        x = (torch.arange(width, device=device, dtype=dtype) + 0.5 - radius_x) / radius_x

        # A fourth-order superellipse closely matches a half-radius rounded
        # rectangle while keeping every centre-to-edge contour smooth. Unlike
        # an inward feather band, it never creates a broad solid-white plateau.
        rounded_distance = (
            y[:, None].abs().pow(4.0) + x[None, :].abs().pow(4.0)
        ).pow(0.25)
        scene = (1.0 - rounded_distance.pow(1.6)).clamp_min(0.0).pow(1.15)
        return scene.view(1, 1, height, width).expand(batch, 1, height, width)

    def _make_hybrid_foreground_guard_mask(
        self,
        batch: int,
        height: int,
        width: int,
        device,
        dtype,
    ) -> torch.Tensor:
        """Return the broad asymmetric guard used around the restored foreground."""
        radius_y = float(height) * 0.5
        y = (torch.arange(height, device=device, dtype=dtype) + 0.5 - radius_y) / radius_y

        # The upper half behaves like a lightly rounded card while the lower
        # half uses a broader half-radius rounded rectangle. This preserves
        # more of wide and tall videos than the former lower half-ellipse.
        corner_radius = torch.where(
            y[:, None] < 0.0,
            torch.full((height, 1), 0.25, device=device, dtype=dtype),
            torch.full((height, 1), 0.5, device=device, dtype=dtype),
        )
        signed_distance = self._make_normalized_rounded_rect_signed_distance(
            height,
            width,
            device,
            dtype,
            corner_radius,
        )
        guard = (-signed_distance / 0.24).clamp(0.0, 1.0)
        return guard.view(1, 1, height, width).expand(batch, 1, height, width)

    def _make_hybrid_bottom_fill_mask(
        self,
        batch: int,
        height: int,
        width: int,
        device,
        dtype,
    ) -> torch.Tensor:
        """Fill the lower frame so imperfect subject mattes cannot punch holes through it."""
        y = (torch.arange(height, device=device, dtype=dtype) + 0.5) / float(height)
        fill = ((y - 0.40) / 0.20).clamp(0.0, 1.0)
        fill = fill.pow(3.0) * (fill * (fill * 6.0 - 15.0) + 10.0)
        return fill.view(1, 1, height, 1).expand(batch, 1, height, width)

    def _feather_hybrid_foreground_boundary(self, alpha: torch.Tensor) -> torch.Tensor:
        """Apply a subtle resolution-aware feather to the final hybrid foreground edge."""
        height, width = alpha.shape[-2:]
        sigma = max(0.75, min(4.0, float(min(height, width)) * 0.0025))
        radius = max(1, int(round(sigma * 2.0)))
        coordinates = torch.arange(-radius, radius + 1, device=alpha.device, dtype=alpha.dtype)
        kernel = torch.exp(-0.5 * (coordinates / sigma).square())
        kernel = kernel / kernel.sum()

        horizontal = kernel.view(1, 1, 1, -1)
        vertical = kernel.view(1, 1, -1, 1)
        feathered = F.conv2d(F.pad(alpha, (radius, radius, 0, 0)), horizontal)
        feathered = F.conv2d(F.pad(feathered, (0, 0, radius, radius)), vertical)
        return feathered.clamp(0.0, 1.0)

    def _apply_layer_spatial_masks(
        self,
        alpha: torch.Tensor,
        hybrid_mode: bool,
        fade_mode: str,
        top_fade_ratio: float,
        background_fade_ratio: float,
        rounded_rect_fade_ratio: float,
        rounded_corner_radius: float,
    ) -> torch.Tensor:
        batch, _, height, width = alpha.shape
        if hybrid_mode:
            return alpha * self._make_hybrid_scene_mask(
                batch, height, width, alpha.device, alpha.dtype
            )
        if background_fade_ratio > 0.0:
            alpha = alpha * self._make_top_fade_mask(
                batch,
                height,
                width,
                alpha.device,
                alpha.dtype,
                background_fade_ratio,
            )
        return self._apply_fade_mask(
            alpha,
            fade_mode,
            top_fade_ratio,
            rounded_rect_fade_ratio,
            rounded_corner_radius,
        )

    def _apply_foreground_spatial_mask(
        self,
        alpha: torch.Tensor,
        hybrid_mode: bool,
        fade_mode: str,
        top_fade_ratio: float,
    ) -> torch.Tensor:
        batch, _, height, width = alpha.shape
        if hybrid_mode:
            bottom_fill = self._make_hybrid_bottom_fill_mask(
                batch, height, width, alpha.device, alpha.dtype
            )
            guard = self._make_hybrid_foreground_guard_mask(
                batch, height, width, alpha.device, alpha.dtype
            )
            combined = torch.maximum(alpha, bottom_fill) * guard
            return self._feather_hybrid_foreground_boundary(combined)
        if fade_mode == "Top Fade":
            return alpha * self._make_top_fade_mask(
                batch,
                height,
                width,
                alpha.device,
                alpha.dtype,
                top_fade_ratio,
            )
        return alpha

    def _scale_about_center(self, tensor: torch.Tensor, scale: float) -> torch.Tensor:
        """Scale BCHW content inside its existing canvas, leaving empty margins."""
        if scale == 1.0:
            return tensor
        if scale == 0.0:
            return torch.zeros_like(tensor)
        height, width = tensor.shape[-2:]
        scaled_h = max(1, int(round(height * scale)))
        scaled_w = max(1, int(round(width * scale)))
        resized = F.interpolate(tensor, size=(scaled_h, scaled_w), mode="bilinear", align_corners=False)
        top = (height - scaled_h) // 2
        left = (width - scaled_w) // 2
        return F.pad(resized, (left, width - scaled_w - left, top, height - scaled_h - top))

    def _make_temporal_fade_envelope(
        self,
        batch: int,
        device,
        dtype,
        fade_frames: int,
    ) -> torch.Tensor:
        """Match Gift Mask Fade In-Out and return a broadcastable BCHW envelope."""
        count = max(0, min(int(fade_frames), (batch + 1) // 2))
        if count == 0:
            return torch.ones((batch, 1, 1, 1), device=device, dtype=dtype)

        frame = torch.arange(batch, device=device, dtype=dtype)
        denominator = float(max(1, count - 1))
        fade_in = (frame / denominator).clamp(0.0, 1.0)
        fade_out = ((batch - 1 - frame) / denominator).clamp(0.0, 1.0)
        return torch.minimum(fade_in, fade_out).view(batch, 1, 1, 1)

    def composite(
        self,
        background_image,
        layer_image,
        opacity=1.0,
        clip_if_too_tall=True,
        fade_mode="None",
        top_fade_ratio=0.08,
        enable_packed_output=True,
        layer_mask=None,
        packed_size_mode="Fit Content (Dynamic Height)",
        rounded_rect_fade_ratio=0.16,
        rounded_corner_radius=0.30,
        center_scale=1.0,
        foreground_image=None,
        foreground_mask=None,
        preset="Custom",
        background_fade_ratio=0.0,
        fade_frames=0,
    ):
        if preset not in self.PRESET_OPTIONS:
            raise ValueError(f"Unknown preset: {preset!r}. Expected one of {self.PRESET_OPTIONS}.")
        hybrid_mode = preset == "Hybrid Naked-Eye 3D"
        if preset != "Custom":
            preset_values = self.PRESET_VALUES[preset]
            fade_mode = preset_values["fade_mode"]
            top_fade_ratio = preset_values.get("top_fade_ratio", top_fade_ratio)
            background_fade_ratio = preset_values.get("background_fade_ratio", background_fade_ratio)
            rounded_rect_fade_ratio = preset_values.get("rounded_rect_fade_ratio", rounded_rect_fade_ratio)
            rounded_corner_radius = preset_values.get("rounded_corner_radius", rounded_corner_radius)

        if fade_mode not in self.FADE_MODE_OPTIONS:
            raise ValueError(f"Unknown fade_mode: {fade_mode!r}. Expected one of {self.FADE_MODE_OPTIONS}.")

        center_scale = float(center_scale)
        if not 0.0 <= center_scale <= 1.0:
            raise ValueError("center_scale must be between 0 and 1.")
        bg = self._ensure_image(background_image)
        fg = self._ensure_image(layer_image)
        restored_fg = self._ensure_image(foreground_image) if foreground_image is not None else None

        if restored_fg is None:
            bg, fg, batch = self._broadcast_batch(bg, fg)
        else:
            bg, fg, restored_fg, batch = self._broadcast_foreground(bg, fg, restored_fg)

        device = bg.device
        dtype = bg.dtype
        temporal_fade = self._make_temporal_fade_envelope(
            batch,
            device,
            dtype,
            fade_frames,
        )

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

        restored_fg_bchw = None
        restored_alpha = None
        if restored_fg is not None:
            restored_h, restored_w = restored_fg.shape[1], restored_fg.shape[2]
            restored_fg_bchw = restored_fg.permute(0, 3, 1, 2).contiguous()
            restored_fg_bchw = F.interpolate(
                restored_fg_bchw,
                size=(resized_h, packed_w),
                mode="bilinear",
                align_corners=False,
            )
            restored_alpha = self._ensure_mask(
                foreground_mask,
                batch,
                restored_h,
                restored_w,
                device,
                dtype,
            )
            restored_alpha = F.interpolate(
                restored_alpha,
                size=(resized_h, packed_w),
                mode="bilinear",
                align_corners=False,
            )

        alpha = self._ensure_mask(layer_mask, batch, fg_h, fg_w, device, dtype)
        alpha = F.interpolate(alpha, size=(resized_h, packed_w), mode="bilinear", align_corners=False)

        alpha = self._apply_layer_spatial_masks(
            alpha,
            hybrid_mode,
            fade_mode,
            top_fade_ratio,
            background_fade_ratio,
            rounded_rect_fade_ratio,
            rounded_corner_radius,
        )

        alpha = (alpha * float(opacity)).clamp(0.0, 1.0)
        if restored_alpha is not None:
            restored_alpha = self._apply_foreground_spatial_mask(
                restored_alpha,
                hybrid_mode,
                fade_mode,
                top_fade_ratio,
            )
            restored_alpha = (restored_alpha * float(opacity)).clamp(0.0, 1.0)

        # Packed output is either content-sized or bottom-aligned on a fixed canvas.
        if enable_packed_output:
            packed_fg_resized = fg_resized
            packed_alpha = alpha
            packed_restored_fg = restored_fg_bchw
            packed_restored_alpha = restored_alpha

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
                packed_alpha = self._apply_layer_spatial_masks(
                    packed_alpha,
                    hybrid_mode,
                    fade_mode,
                    top_fade_ratio,
                    background_fade_ratio,
                    rounded_rect_fade_ratio,
                    rounded_corner_radius,
                )
                packed_alpha = (packed_alpha * float(opacity)).clamp(0.0, 1.0)

                if restored_fg is not None:
                    packed_restored_fg = F.interpolate(
                        restored_fg.permute(0, 3, 1, 2).contiguous(),
                        size=(fixed_h, 720),
                        mode="bilinear",
                        align_corners=False,
                    )
                    packed_restored_alpha = self._ensure_mask(
                        foreground_mask,
                        batch,
                        restored_fg.shape[1],
                        restored_fg.shape[2],
                        device,
                        dtype,
                    )
                    packed_restored_alpha = F.interpolate(
                        packed_restored_alpha,
                        size=(fixed_h, 720),
                        mode="bilinear",
                        align_corners=False,
                    )
                    packed_restored_alpha = self._apply_foreground_spatial_mask(
                        packed_restored_alpha,
                        hybrid_mode,
                        fade_mode,
                        top_fade_ratio,
                    )
                    packed_restored_alpha = (
                        packed_restored_alpha * float(opacity)
                    ).clamp(0.0, 1.0)

            # Scale each panel around the fitted video center, before top padding.
            # Premultiply before resampling to keep transparent-edge colors clean.
            packed_rgb, packed_alpha = self._combine_premultiplied_layers(
                packed_fg_resized,
                packed_alpha,
                packed_restored_fg,
                packed_restored_alpha,
            )
            packed_rgb = packed_rgb * temporal_fade
            packed_alpha = packed_alpha * temporal_fade
            if center_scale != 1.0:
                if packed_size_mode == "Fixed Canvas (1440x1280)":
                    packed_rgb = packed_rgb[:, :, -1280:, :]
                    packed_alpha = packed_alpha[:, :, -1280:, :]
                packed_rgb = self._scale_about_center(packed_rgb, center_scale)
                packed_alpha = self._scale_about_center(packed_alpha, center_scale)
            packed_mask = (
                packed_alpha[:, 0:1, :, :]
                .repeat(1, 3, 1, 1)
                .permute(0, 2, 3, 1)
                .contiguous()
            )
            packed_fg = packed_rgb.permute(0, 2, 3, 1).contiguous()
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
        restored_fg_preview = restored_fg_bchw
        restored_alpha_preview = restored_alpha

        if preview_h > bg_h:
            if not clip_if_too_tall:
                raise ValueError(
                    f"Resized layer height ({preview_h}) exceeds background height ({bg_h}). "
                    f"Enable clip_if_too_tall to crop from the top automatically."
                )
            fg_preview = fg_preview[:, :, preview_h - bg_h :, :]
            alpha_preview = alpha_preview[:, :, preview_h - bg_h :, :]
            if restored_fg_preview is not None:
                restored_fg_preview = restored_fg_preview[:, :, preview_h - bg_h :, :]
                restored_alpha_preview = restored_alpha_preview[:, :, preview_h - bg_h :, :]
            preview_h = bg_h

        y0 = bg_h - preview_h
        y1 = bg_h
        x0 = 0
        x1 = bg_w

        out = bg.clone()
        out_region = out[:, y0:y1, x0:x1, :]

        combined_rgb, alpha_preview = self._combine_premultiplied_layers(
            fg_preview,
            alpha_preview,
            restored_fg_preview,
            restored_alpha_preview,
        )
        combined_rgb = combined_rgb * temporal_fade
        alpha_preview = alpha_preview * temporal_fade
        a_region = alpha_preview.permute(0, 2, 3, 1).contiguous()

        if center_scale == 1.0:
            out[:, y0:y1, x0:x1, :] = (
                combined_rgb.permute(0, 2, 3, 1).contiguous()
                + out_region * (1.0 - a_region)
            )
        else:
            scaled_rgb = self._scale_about_center(combined_rgb, center_scale)
            alpha_preview = self._scale_about_center(alpha_preview, center_scale)
            a_region = alpha_preview.permute(0, 2, 3, 1)
            out[:, y0:y1, x0:x1, :] = scaled_rgb.permute(0, 2, 3, 1) + out_region * (1.0 - a_region)

        full_mask = torch.zeros((batch, bg_h, bg_w), device=device, dtype=dtype)
        full_mask[:, y0:y1, x0:x1] = alpha_preview[:, 0, :, :]

        return (out.clamp(0.0, 1.0), full_mask.clamp(0.0, 1.0), packed)


NODE_CLASS_MAPPINGS = {
    "FastBottomFitOverlay": FastBottomFitOverlay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FastBottomFitOverlay": "Fast Bottom Fit Overlay (Packed)",
}
