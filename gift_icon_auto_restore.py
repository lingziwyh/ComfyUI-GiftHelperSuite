"""One-node gift ICON restoration, tight crop, and standard exports.

The node is designed for the production path used by Gift Chroma Master:

    keyed subject RGBA + foreground alpha + original black-background render
        -> restore useful glow/translucency
        -> crop from the final alpha (not the solid subject mask)
        -> premultiplied-alpha resize and transparent square padding
        -> preview + 1280 RGBA + 168 RGBA

All public IMAGE values use ComfyUI's BHWC convention. Internal image work uses
BCHW tensors so PyTorch interpolation behaves consistently for RGB and alpha.
"""

from __future__ import annotations

import math
import pathlib
from typing import Optional

import torch
import torch.nn.functional as F

from .gift_chroma_master.core.utils import image_bhwc_to_bchw, mask_to_bchw


CATEGORY = "GiftHelperSuite/Icon"
_PREVIEW_BACKGROUND_CPU: Optional[torch.Tensor] = None


def _float(default, minimum, maximum, step=0.01, *, advanced=False, tooltip=None):
    options = {"default": default, "min": minimum, "max": maximum, "step": step}
    if advanced:
        options["advanced"] = True
    if tooltip:
        options["tooltip"] = tooltip
    return ("FLOAT", options)


def _int(default, minimum, maximum, step=1, *, advanced=False, tooltip=None):
    options = {"default": default, "min": minimum, "max": maximum, "step": step}
    if advanced:
        options["advanced"] = True
    if tooltip:
        options["tooltip"] = tooltip
    return ("INT", options)


def _match_batch(tensor: torch.Tensor, batch: int, name: str) -> torch.Tensor:
    if int(tensor.shape[0]) == batch:
        return tensor
    if int(tensor.shape[0]) == 1:
        return tensor.expand(batch, *tensor.shape[1:])
    raise ValueError(f"{name} batch must be 1 or {batch}; got {tensor.shape[0]}")


def _resolve_processing_device(image: torch.Tensor, mode: str) -> torch.device:
    policy = str(mode).strip().lower()
    if policy not in ("auto", "cuda", "cpu"):
        raise ValueError("performance_mode must be auto, cuda, or cpu")
    if policy == "cpu":
        return torch.device("cpu")
    if policy == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA acceleration was requested but CUDA is unavailable")
        return image.device if image.device.type == "cuda" else torch.device(
            "cuda", torch.cuda.current_device()
        )
    if image.device.type == "cuda":
        return image.device
    try:
        import comfy.model_management as model_management

        comfy_device = torch.device(model_management.get_torch_device())
        return comfy_device if comfy_device.type == "cuda" else torch.device("cpu")
    except (ImportError, AttributeError, RuntimeError, TypeError):
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")


def _interpolate(
    tensor: torch.Tensor,
    size: tuple[int, int],
    *,
    mode: str = "bilinear",
) -> torch.Tensor:
    if tuple(tensor.shape[-2:]) == tuple(size):
        return tensor
    kwargs = {"size": size, "mode": mode}
    if mode in ("bilinear", "bicubic"):
        kwargs["align_corners"] = False
        # Antialiasing is useful for reduction but expensive and unnecessary
        # for enlargement. The old implementation enabled it for every 1280
        # and preview upscale, which dominated CPU execution time.
        if size[0] < tensor.shape[-2] or size[1] < tensor.shape[-1]:
            kwargs["antialias"] = True
    try:
        return F.interpolate(tensor, **kwargs)
    except TypeError:
        kwargs.pop("antialias", None)
        return F.interpolate(tensor, **kwargs)


def _estimate_black_statistics(intensity: torch.Tensor) -> tuple[float, float]:
    """Estimate the black floor and visible noise amplitude from the canvas border."""
    height, width = intensity.shape
    band = max(2, int(round(min(height, width) * 0.03)))
    border = torch.cat(
        (
            intensity[:band].reshape(-1),
            intensity[-band:].reshape(-1),
            intensity[:, :band].reshape(-1),
            intensity[:, -band:].reshape(-1),
        )
    ).float()
    median = torch.quantile(border, 0.50)
    high = torch.quantile(border, 0.90)
    mad = torch.quantile(torch.abs(border - median), 0.50) * 1.4826
    noise = torch.maximum(mad, (high - median).clamp_min(0.0))
    # A small automatic offset handles quantization around pure black without
    # imposing the old fixed 0.01 floor that could erase legitimate faint glow.
    floor = median + torch.maximum(noise * 0.50, median.new_tensor(1.0 / 1024.0))
    return (
        float(floor.clamp(0.0, 0.20).item()),
        float(noise.clamp(0.0, 0.20).item()),
    )


def _smoothstep(value: torch.Tensor, low: float, high: float) -> torch.Tensor:
    span = max(float(high) - float(low), 1.0e-6)
    t = ((value - float(low)) / span).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _procedural_edge_guard(
    height: int,
    width: int,
    margin_percent: float,
    feather_percent: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create a soft rounded-rectangle keep mask: 1=center, 0=canvas edge."""
    reference = float(min(height, width))
    margin = reference * max(0.0, float(margin_percent)) / 100.0
    feather = max(1.0, reference * max(0.0, float(feather_percent)) / 100.0)
    y = torch.arange(height, device=device, dtype=dtype)
    x = torch.arange(width, device=device, dtype=dtype)
    y_distance = torch.minimum(y, (height - 1) - y)
    x_distance = torch.minimum(x, (width - 1) - x)
    y_ramp = _smoothstep(y_distance, margin, margin + feather)
    x_ramp = _smoothstep(x_distance, margin, margin + feather)
    # Multiplication makes both axes participate at a corner, producing the
    # rounded soft shape used by the former hand-authored safety matte.
    return (y_ramp[:, None] * x_ramp[None, :]).clamp(0.0, 1.0)


def _optional_mask_to_bchw(
    mask: torch.Tensor,
    *,
    batch: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    """Normalize an optional mask and resize it to the working canvas."""
    if not isinstance(mask, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    value = mask
    if value.ndim == 2:
        value = value[None, None]
    elif value.ndim == 3:
        value = value[:, None]
    elif value.ndim == 4 and value.shape[-1] == 1:
        value = value.permute(0, 3, 1, 2)
    elif value.ndim == 4 and value.shape[1] == 1:
        pass
    else:
        raise ValueError(
            f"{name} must be [H,W], [B,H,W], [B,1,H,W], or [B,H,W,1]"
        )
    value = value.to(device=device, dtype=dtype)
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains NaN or infinite values")
    value = _match_batch(value, batch, name)
    return _interpolate(value, (height, width)).clamp(0.0, 1.0)


def _soft_expand(mask: torch.Tensor, expand_pixels: float, feather_pixels: float) -> torch.Tensor:
    """Expand a 2D mask on a bounded working grid, then feather its boundary."""
    height, width = mask.shape
    scale = min(1.0, 384.0 / max(height, width))
    small_h = max(16, int(round(height * scale)))
    small_w = max(16, int(round(width * scale)))
    small = _interpolate(mask[None, None], (small_h, small_w))[0, 0]

    radius = max(0, int(round(float(expand_pixels) * scale)))
    max_radius = max(0, (min(small_h, small_w) - 1) // 2)
    radius = min(radius, max_radius)
    if radius:
        expanded = F.max_pool2d(
            small[None, None],
            kernel_size=radius * 2 + 1,
            stride=1,
            padding=radius,
        )[0, 0]
    else:
        expanded = small

    feather = max(0, int(round(float(feather_pixels) * scale)))
    feather = min(feather, max_radius)
    if feather:
        softened = F.avg_pool2d(
            expanded[None, None],
            kernel_size=feather * 2 + 1,
            stride=1,
            padding=feather,
        )[0, 0]
        expanded = torch.maximum(expanded, softened)
    return _interpolate(expanded[None, None], (height, width))[0, 0].clamp(0.0, 1.0)


def _bounds(mask: torch.Tensor, threshold: float) -> Optional[tuple[int, int, int, int]]:
    coordinates = torch.nonzero(mask > float(threshold), as_tuple=False)
    if coordinates.numel() == 0:
        return None
    return (
        int(coordinates[:, 0].min().item()),
        int(coordinates[:, 0].max().item()) + 1,
        int(coordinates[:, 1].min().item()),
        int(coordinates[:, 1].max().item()) + 1,
    )


def _normal_over(
    subject_rgb: torch.Tensor,
    subject_alpha: torch.Tensor,
    effect_rgb: torch.Tensor,
    effect_alpha: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """AE Alpha Over compatible normal blend for two straight-alpha CHW layers."""
    subject_a = subject_alpha.unsqueeze(0)
    effect_a = effect_alpha.unsqueeze(0)
    output_alpha = effect_a + subject_a - effect_a * subject_a
    output_premult = (
        effect_rgb * effect_a
        + subject_rgb * subject_a * (1.0 - effect_a)
    )
    output_rgb = torch.where(
        output_alpha > 1.0e-6,
        output_premult / output_alpha.clamp_min(1.0e-6),
        torch.zeros_like(output_premult),
    )
    return output_rgb.clamp(0.0, 1.0), output_alpha[0].clamp(0.0, 1.0)


def _resize_rgba_exact(rgba: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize CHW straight RGBA through premultiplied color to prevent dark fringes."""
    if tuple(rgba.shape[-2:]) == (height, width):
        return rgba
    alpha = rgba[3:4].clamp(0.0, 1.0)
    premultiplied = rgba[:3].clamp(0.0, 1.0) * alpha
    reducing = height < rgba.shape[-2] or width < rgba.shape[-1]
    color_mode = "area" if reducing else "bicubic"
    alpha_mode = "area" if reducing else "bilinear"
    resized_premult = _interpolate(
        premultiplied[None], (height, width), mode=color_mode
    )[0].clamp(0.0, 1.0)
    resized_alpha = _interpolate(alpha[None], (height, width), mode=alpha_mode)[0].clamp(0.0, 1.0)
    resized_rgb = torch.where(
        resized_alpha > 1.0e-6,
        resized_premult / resized_alpha.clamp_min(1.0e-6),
        torch.zeros_like(resized_premult),
    )
    return torch.cat((resized_rgb.clamp(0.0, 1.0), resized_alpha), dim=0)


def _fit_rgba_to_square(rgba: torch.Tensor, size: int, padding: int) -> torch.Tensor:
    """Scale the long side into a transparent square and center the short side."""
    usable = int(size) - 2 * int(padding)
    if usable <= 0:
        raise ValueError("icon padding must leave at least one usable pixel")
    height, width = map(int, rgba.shape[-2:])
    scale = usable / float(max(height, width))
    new_h = max(1, min(usable, int(round(height * scale))))
    new_w = max(1, min(usable, int(round(width * scale))))
    resized = _resize_rgba_exact(rgba, new_h, new_w)
    canvas = torch.zeros((4, int(size), int(size)), device=rgba.device, dtype=rgba.dtype)
    top = (int(size) - new_h) // 2
    left = (int(size) - new_w) // 2
    canvas[:, top : top + new_h, left : left + new_w] = resized
    return canvas


def _refit_visible_rgba(
    rgba: torch.Tensor,
    size: int,
    visibility_threshold: float,
    padding: int,
) -> torch.Tensor:
    """Re-fit after resampling so 8-bit-invisible tails cannot define the edge."""
    result = rgba
    for _ in range(2):
        visible = _bounds(result[3], float(visibility_threshold))
        if visible is None:
            return result
        y_min, y_max, x_min, x_max = visible
        long_edge_is_fitted = (
            (y_min == int(padding) and y_max == int(size) - int(padding))
            or (x_min == int(padding) and x_max == int(size) - int(padding))
        )
        if long_edge_is_fitted:
            return result
        result = _fit_rgba_to_square(
            result[:, y_min:y_max, x_min:x_max],
            int(size),
            int(padding),
        )
    return result


def _checkerboard(
    batch: int,
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    tile = max(8, min(height, width) // 32)
    y = torch.arange(height, device=device)[:, None] // tile
    x = torch.arange(width, device=device)[None, :] // tile
    pattern = ((x + y) % 2).to(dtype=dtype)
    low = torch.full_like(pattern, 0.18)
    high = torch.full_like(pattern, 0.32)
    rgb = torch.where(pattern.bool(), high, low)[None, None].expand(batch, 3, -1, -1)
    return rgb.contiguous()


def _builtin_preview_background(
    size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Load the bundled 1680 guide once, then reuse its cached CPU tensor."""
    global _PREVIEW_BACKGROUND_CPU
    if _PREVIEW_BACKGROUND_CPU is None:
        asset = pathlib.Path(__file__).resolve().parent / "assets" / "gift_icon_preview_guide.png"
        try:
            import numpy as np
            from PIL import Image

            with Image.open(asset) as image:
                array = np.asarray(image.convert("RGB")).copy()
            _PREVIEW_BACKGROUND_CPU = (
                torch.from_numpy(array)
                .to(dtype=torch.float32)
                .div_(255.0)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .contiguous()
            )
        except (FileNotFoundError, ImportError, OSError, ValueError):
            _PREVIEW_BACKGROUND_CPU = torch.empty(0)
    if _PREVIEW_BACKGROUND_CPU.numel() == 0:
        return _checkerboard(1, int(size), int(size), device=device, dtype=dtype)
    background = _PREVIEW_BACKGROUND_CPU.to(device=device, dtype=dtype)
    if tuple(background.shape[-2:]) != (int(size), int(size)):
        background = _interpolate(background, (int(size), int(size)))
    return background


def _preview_composite(
    icons: torch.Tensor,
    preview_background: Optional[torch.Tensor],
    preview_scale: float,
    preview_canvas_size: int,
) -> torch.Tensor:
    """Center RGBA icons over a user background or a generated checkerboard."""
    batch, _, icon_h, icon_w = icons.shape
    if preview_background is None:
        background = _builtin_preview_background(
            int(preview_canvas_size),
            device=icons.device,
            dtype=icons.dtype,
        )
        background = background.expand(batch, -1, -1, -1)
    else:
        background, background_alpha = image_bhwc_to_bchw(preview_background)
        background = _match_batch(background, batch, "preview_background").to(
            device=icons.device, dtype=icons.dtype
        )
        if background_alpha is not None:
            background_alpha = _match_batch(
                background_alpha, batch, "preview_background alpha"
            ).to(device=icons.device, dtype=icons.dtype)
            background = background * background_alpha

    out = background.clone()
    background_h, background_w = map(int, out.shape[-2:])
    requested = max(1, int(round(max(icon_h, icon_w) * float(preview_scale))))
    display_size = min(requested, background_h, background_w)
    for index in range(batch):
        icon = _resize_rgba_exact(icons[index], display_size, display_size)
        top = (background_h - display_size) // 2
        left = (background_w - display_size) // 2
        alpha = icon[3:4]
        region = out[index, :, top : top + display_size, left : left + display_size]
        out[index, :, top : top + display_size, left : left + display_size] = (
            icon[:3] * alpha + region * (1.0 - alpha)
        )
    return out.clamp(0.0, 1.0).permute(0, 2, 3, 1).contiguous()


class GiftIconAutoRestore:
    """Restore black-background FX and emit production-ready gift ICON files."""

    CATEGORY = CATEGORY
    FUNCTION = "restore_and_export"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("preview", "icon_1280_rgba", "icon_168_rgba")
    OUTPUT_TOOLTIPS = (
        "ICON composited over preview_background; checkerboard when no background is connected.",
        "Tight-cropped RGBA, long side fitted to target_size, transparent short-side padding.",
        "Premultiplied-alpha thumbnail derived from the standard ICON.",
    )
    DESCRIPTION = (
        "Restores glow/translucency from the original black-background render, crops from "
        "the combined final alpha, and exports standard transparent gift ICON sizes."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "subject_rgba": (
                    "IMAGE",
                    {"tooltip": "Klein + Gift Chroma Master Pack RGBA output at the original canvas size."},
                ),
                "subject_mask": (
                    "MASK",
                    {"tooltip": "Gift Chroma Master foreground alpha: 1=subject, 0=transparent."},
                ),
                "original_black_image": (
                    "IMAGE",
                    {"tooltip": "Original black-background subject + glow image before Klein."},
                ),
                "fx_strength": _float(
                    0.75,
                    0.0,
                    2.0,
                    0.05,
                    tooltip="Opacity of the automatically recovered glow/effect layer.",
                ),
                "unmult_black_point": _float(
                    0.105,
                    0.0,
                    0.50,
                    0.001,
                    tooltip=(
                        "AE Unmult black point. Default 0.105 matches the former #380 chain "
                        "and keeps glow edges continuous without revealing black-floor noise."
                    ),
                ),
                "fx_reach": _float(
                    0.35,
                    0.05,
                    1.0,
                    0.01,
                    advanced=True,
                    tooltip="Maximum effect search distance relative to subject size.",
                ),
                "fx_edge_feather": _int(
                    32,
                    0,
                    256,
                    advanced=True,
                    tooltip="Softens the subject-shaped effect search region.",
                ),
                "alpha_cutoff": _float(
                    0.002,
                    0.0,
                    0.20,
                    0.001,
                    advanced=True,
                    tooltip="Final alpha at or below this value is treated as invalid edge haze.",
                ),
                "enable_edge_fade": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Fade subject and effects before the source canvas edge so cropped "
                            "arms/close-ups cannot leave a straight hard boundary."
                        ),
                    },
                ),
                "edge_guard_percent": _float(
                    0.0,
                    0.0,
                    25.0,
                    0.1,
                    advanced=True,
                    tooltip=(
                        "Fully rejected border width as a percentage of the short canvas side. "
                        "Default 0 keeps every non-edge pixel available."
                    ),
                ),
                "edge_feather_percent": _float(
                    2.5,
                    0.1,
                    40.0,
                    0.1,
                    advanced=True,
                    tooltip=(
                        "Narrow soft transition at the outer canvas edge. Default 2.5% is "
                        "about 32 px on a 1280 image, leaving the central 95% untouched."
                    ),
                ),
                "min_effect_visibility": _float(
                    0.006,
                    0.0,
                    0.20,
                    0.001,
                    advanced=True,
                    tooltip=(
                        "Minimum premultiplied RGB contribution considered visible. "
                        "The automatic border-noise estimate can raise this threshold."
                    ),
                ),
                "crop_padding": _int(
                    0,
                    0,
                    256,
                    advanced=True,
                    tooltip="Optional pixels kept around the valid final-alpha bounds.",
                ),
                "output_padding": _int(
                    8,
                    0,
                    512,
                    advanced=True,
                    tooltip=(
                        "Transparent padding around the target-size ICON. The thumbnail "
                        "padding is scaled proportionally from this value."
                    ),
                ),
                "target_size": _int(1280, 64, 4096, advanced=True),
                "thumbnail_size": _int(168, 16, 1024, advanced=True),
                "preview_scale": _float(
                    1.04,
                    0.05,
                    4.0,
                    0.01,
                    advanced=True,
                    tooltip="Preview ICON size relative to target_size before centering.",
                ),
                "preview_canvas_size": _int(
                    1680,
                    256,
                    4096,
                    advanced=True,
                    tooltip="Size of the bundled square guide background when no custom background is connected.",
                ),
                "performance_mode": (
                    ["auto", "cuda", "cpu"],
                    {
                        "default": "auto",
                        "advanced": True,
                        "tooltip": "Auto uses ComfyUI's CUDA device and returns outputs to the input device.",
                    },
                ),
            },
            "optional": {
                "preview_background": (
                    "IMAGE",
                    {"tooltip": "Optional background used only for the preview output."},
                ),
                "edge_guard_mask": (
                    "MASK",
                    {
                        "tooltip": (
                            "Optional custom safety matte: 1=keep, 0=fade. "
                            "Overrides the generated rounded edge guard."
                        )
                    },
                ),
            },
        }

    @torch.inference_mode()
    def restore_and_export(
        self,
        subject_rgba,
        subject_mask,
        original_black_image,
        fx_strength=0.75,
        unmult_black_point=0.105,
        fx_reach=0.35,
        fx_edge_feather=32,
        alpha_cutoff=0.002,
        enable_edge_fade=True,
        edge_guard_percent=0.0,
        edge_feather_percent=2.5,
        min_effect_visibility=0.006,
        crop_padding=0,
        output_padding=8,
        target_size=1280,
        thumbnail_size=168,
        preview_scale=1.04,
        preview_canvas_size=1680,
        performance_mode="auto",
        preview_background=None,
        edge_guard_mask=None,
    ):
        subject_rgb, embedded_alpha = image_bhwc_to_bchw(subject_rgba)
        output_device = subject_rgb.device
        process_device = _resolve_processing_device(subject_rgb, str(performance_mode))
        subject_rgb = subject_rgb.to(device=process_device)
        if embedded_alpha is None:
            embedded_alpha = torch.ones_like(subject_rgb[:, :1])
        else:
            embedded_alpha = embedded_alpha.to(device=process_device)
        batch, _, height, width = subject_rgb.shape
        device, dtype = subject_rgb.device, subject_rgb.dtype
        foreground = mask_to_bchw(
            subject_mask,
            batch=batch,
            height=height,
            width=width,
            device=device,
            dtype=dtype,
            name="subject_mask",
        )
        if foreground is None:
            raise ValueError("subject_mask is required")

        source_rgb, _ = image_bhwc_to_bchw(original_black_image)
        source_rgb = _match_batch(source_rgb, batch, "original_black_image").to(
            device=device, dtype=dtype
        )
        source_rgb = _interpolate(source_rgb, (height, width)).clamp(0.0, 1.0)
        embedded_alpha = _match_batch(embedded_alpha, batch, "subject_rgba alpha")
        if edge_guard_mask is not None:
            edge_guard = _optional_mask_to_bchw(
                edge_guard_mask,
                batch=batch,
                height=height,
                width=width,
                device=device,
                dtype=dtype,
                name="edge_guard_mask",
            )
        elif bool(enable_edge_fade):
            generated_guard = _procedural_edge_guard(
                height,
                width,
                float(edge_guard_percent),
                float(edge_feather_percent),
                device=device,
                dtype=dtype,
            )
            edge_guard = generated_guard[None, None].expand(batch, 1, -1, -1)
        else:
            edge_guard = torch.ones(
                (batch, 1, height, width),
                device=device,
                dtype=dtype,
            )

        subject_alpha = (
            torch.minimum(embedded_alpha, foreground) * edge_guard
        ).clamp(0.0, 1.0)
        subject_alpha = torch.where(
            subject_alpha > max(float(alpha_cutoff), 1.0e-6),
            subject_alpha,
            torch.zeros_like(subject_alpha),
        )

        icons = []
        thumbnails = []
        for index in range(batch):
            mask = foreground[index, 0] * edge_guard[index, 0]
            subject_bounds = _bounds(mask, max(float(alpha_cutoff), 0.002))
            if subject_bounds is None:
                raise ValueError(f"subject_mask for batch item {index} is empty")
            y_min, y_max, x_min, x_max = subject_bounds
            subject_scale = math.sqrt(max(1, (y_max - y_min) * (x_max - x_min)))

            source = source_rgb[index]
            intensity = source.amax(dim=0)
            black_floor, border_noise = _estimate_black_statistics(intensity)
            effective_black_point = max(
                float(unmult_black_point),
                black_floor + max(float(min_effect_visibility), border_noise * 2.5),
            )
            effective_black_point = min(effective_black_point, 0.95)
            extracted_alpha = (
                (intensity - effective_black_point)
                / max(1.0 - effective_black_point, 1.0e-6)
            ).clamp(0.0, 1.0)
            effect_rgb = torch.where(
                extracted_alpha.unsqueeze(0) > 1.0e-6,
                source / extracted_alpha.unsqueeze(0).clamp_min(1.0e-6),
                torch.zeros_like(source),
            ).clamp(0.0, 1.0)

            outer = _soft_expand(
                mask,
                subject_scale * float(fx_reach),
                float(fx_edge_feather),
            )
            # Keep the recovered alpha continuous. The previous implementation
            # multiplied it by a per-pixel confidence gate; small source noise
            # then became a jagged contour after tight-crop enlargement. A
            # single AE-style black-point remap removes the noise floor while
            # this smooth subject-shaped region only limits distant artifacts.
            effect_region = outer.clamp(0.0, 1.0)
            effect_keep = edge_guard[index, 0]
            effect_alpha = (
                extracted_alpha
                * effect_region
                * effect_keep
                * float(fx_strength)
            ).clamp(0.0, 1.0)
            # The former MaskImage node multiplied every RGBA channel by the
            # soft safety matte. Matching that behavior keeps the transition
            # visually identical to the old #380 chain near the canvas edge.
            effect_rgb = effect_rgb * effect_keep.unsqueeze(0)
            effect_alpha = torch.where(
                effect_alpha > float(alpha_cutoff),
                effect_alpha,
                torch.zeros_like(effect_alpha),
            )

            output_rgb, output_alpha = _normal_over(
                subject_rgb[index],
                subject_alpha[index, 0],
                effect_rgb,
                effect_alpha,
            )
            output_alpha = torch.where(
                output_alpha > float(alpha_cutoff),
                output_alpha,
                torch.zeros_like(output_alpha),
            )
            output_rgb = torch.where(
                output_alpha.unsqueeze(0) > 0,
                output_rgb,
                torch.zeros_like(output_rgb),
            )
            final_bounds = _bounds(output_alpha, float(alpha_cutoff))
            if final_bounds is None:
                raise ValueError(f"final alpha for batch item {index} is empty")
            fg_y_min, fg_y_max, fg_x_min, fg_x_max = final_bounds
            padding = int(crop_padding)
            fg_y_min = max(0, fg_y_min - padding)
            fg_y_max = min(height, fg_y_max + padding)
            fg_x_min = max(0, fg_x_min - padding)
            fg_x_max = min(width, fg_x_max + padding)
            cropped = torch.cat((output_rgb, output_alpha.unsqueeze(0)), dim=0)[
                :, fg_y_min:fg_y_max, fg_x_min:fg_x_max
            ]
            standard_padding = min(
                int(output_padding),
                max(0, (int(target_size) - 1) // 2),
            )
            thumbnail_padding = min(
                int(round(int(output_padding) * int(thumbnail_size) / int(target_size))),
                max(0, (int(thumbnail_size) - 1) // 2),
            )
            standard_icon = _fit_rgba_to_square(
                cropped,
                int(target_size),
                standard_padding,
            )
            thumbnail_icon = _fit_rgba_to_square(
                cropped,
                int(thumbnail_size),
                thumbnail_padding,
            )
            if int(crop_padding) == 0:
                # PNG alpha is normally exported at 8-bit precision. Values
                # below half a quantization step round to zero and must not be
                # allowed to leave an apparently empty border at either size.
                export_visibility = max(float(alpha_cutoff), 0.5 / 255.0)
                standard_icon = _refit_visible_rgba(
                    standard_icon,
                    int(target_size),
                    export_visibility,
                    standard_padding,
                )
                thumbnail_icon = _refit_visible_rgba(
                    thumbnail_icon,
                    int(thumbnail_size),
                    export_visibility,
                    thumbnail_padding,
                )
            icons.append(standard_icon)
            # Derive the thumbnail from the tight source crop instead of from
            # the padded 1280 canvas. This avoids a second resampling pass and
            # keeps the thumbnail's own long edge fitted to its canvas.
            thumbnails.append(thumbnail_icon)

        icon_rgba = torch.stack(icons).clamp(0.0, 1.0)
        thumbnail = torch.stack(thumbnails).clamp(0.0, 1.0)
        preview = _preview_composite(
            icon_rgba,
            preview_background,
            float(preview_scale),
            int(preview_canvas_size),
        )
        icon_image = icon_rgba.permute(0, 2, 3, 1).contiguous()
        thumbnail_image = thumbnail.permute(0, 2, 3, 1).contiguous()
        return (
            preview.to(device=output_device),
            icon_image.to(device=output_device),
            thumbnail_image.to(device=output_device),
        )


NODE_CLASS_MAPPINGS = {"GiftIconAutoRestore": GiftIconAutoRestore}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GiftIconAutoRestore": "Gift Icon Auto Restore & Export",
}


__all__ = ["GiftIconAutoRestore"]
