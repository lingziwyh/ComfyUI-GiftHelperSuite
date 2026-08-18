import math

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


def _box_blur(x: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return x
    k = radius * 2 + 1
    x = F.avg_pool2d(x, kernel_size=(1, k), stride=1, padding=(0, radius), count_include_pad=False)
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
    if dx == 0 and dy == 0:
        return channel
    b, c, h, w = channel.shape
    pad_left = max(dx, 0)
    pad_right = max(-dx, 0)
    pad_top = max(dy, 0)
    pad_bottom = max(-dy, 0)
    padded = F.pad(channel, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')
    x0 = pad_right
    y0 = pad_bottom
    return padded[:, :, y0:y0+h, x0:x0+w]


def _apply_color_correction(
    x: torch.Tensor,
    natural_saturation: float,
    saturation: float,
    contrast: float,
    brightness: float,
) -> torch.Tensor:

    if x.shape[1] < 3:
        if abs(contrast - 1.0) > 1e-6:
            x = (x - 0.5) * contrast + 0.5
        if abs(brightness - 0.0) > 1e-6:
            x = x + brightness
        return x.clamp(0.0, 1.0)

    # Natural saturation
    if abs(natural_saturation) > 1e-6:
        luma = _rgb_to_luma(x)
        maxc = x.max(dim=1, keepdim=True).values
        minc = x.min(dim=1, keepdim=True).values
        sat_map = (maxc - minc).clamp(0.0, 1.0)
        boost = 1.0 + natural_saturation * (1.0 - sat_map)
        x = luma + (x - luma) * boost

    # Global saturation
    if abs(saturation - 1.0) > 1e-6:
        luma = _rgb_to_luma(x)
        x = luma + (x - luma) * saturation

    # Contrast
    if abs(contrast - 1.0) > 1e-6:
        x = (x - 0.5) * contrast + 0.5

    # Brightness (linear offset)
    if abs(brightness) > 1e-6:
        x = x + brightness

    return x.clamp(0.0, 1.0)


def _resolve_processing_device(image: torch.Tensor, mode: str) -> torch.device:
    policy = str(mode).strip().lower()
    if policy not in ("auto", "cuda", "cpu"):
        raise ValueError("performance_mode must be auto, cuda, or cpu")
    if policy == "cpu":
        return torch.device("cpu")
    if policy == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA acceleration was requested but CUDA is unavailable")
        if image.device.type == "cuda":
            return image.device
        return torch.device("cuda", torch.cuda.current_device())
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


def _apply_fx_batch(
    image: torch.Tensor,
    *,
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
    natural_saturation: float,
    saturation: float,
    contrast: float,
    brightness: float,
) -> torch.Tensor:
    """Process one independent frame chunk on its current device."""
    x = _to_bchw(image).float().clamp(0.0, 1.0)
    original = x if enable_ca and ca_highlight_only else None

    if enable_bloom and bloom_intensity > 0.0 and bloom_radius > 0:
        highlight = _make_highlight_mask(x, bloom_threshold, knee=0.08)
        bloom_src = x * highlight
        ds = max(1, int(bloom_downsample))
        if ds > 1:
            height, width = x.shape[-2:]
            small_height = max(1, round(height / ds))
            small_width = max(1, round(width / ds))
            bloom_small = F.interpolate(
                bloom_src,
                size=(small_height, small_width),
                mode="bilinear",
                align_corners=False,
            )
            small_radius = max(1, round(bloom_radius / ds))
        else:
            bloom_small = bloom_src
            small_radius = bloom_radius
        bloom_small = _approx_gaussian_blur(bloom_small, small_radius, passes=2)
        if ds > 1:
            bloom = F.interpolate(
                bloom_small,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        else:
            bloom = bloom_small
        x = _screen(x, bloom * bloom_intensity).clamp(0.0, 1.0)

    if enable_sharpen and sharpen_amount > 0.0 and sharpen_radius > 0:
        blur = _approx_gaussian_blur(x, sharpen_radius, passes=1)
        x = (x + (x - blur) * sharpen_amount).clamp(0.0, 1.0)

    if enable_ca and ca_amount > 0.0 and x.shape[1] >= 3:
        radians = math.radians(ca_angle_deg)
        dx = int(round(math.cos(radians) * ca_amount))
        dy = int(round(math.sin(radians) * ca_amount))
        red = _shift_channel_2d(x[:, 0:1], dx, dy)
        green = x[:, 1:2]
        blue = _shift_channel_2d(x[:, 2:3], -dx, -dy)
        shifted = torch.cat((red, green, blue), dim=1)
        if x.shape[1] > 3:
            shifted = torch.cat((shifted, x[:, 3:]), dim=1)
        if ca_highlight_only:
            highlight_source = original if original is not None else x
            mask = _make_highlight_mask(
                highlight_source,
                max(0.35, bloom_threshold if enable_bloom else 0.6),
                knee=0.12,
            )
            x = x * (1.0 - mask) + shifted * mask
        else:
            x = shifted
        x = x.clamp(0.0, 1.0)

    x = _apply_color_correction(
        x,
        natural_saturation,
        saturation,
        contrast,
        brightness,
    )
    return _to_bhwc(x)


def _run_chunked(
    image: torch.Tensor,
    *,
    processing_device: torch.device,
    output_device: torch.device,
    chunk_size: int,
    options: dict,
) -> torch.Tensor:
    output = torch.empty(
        image.shape,
        device=output_device,
        dtype=torch.float32,
        memory_format=torch.contiguous_format,
    )
    for start in range(0, image.shape[0], chunk_size):
        end = min(start + chunk_size, image.shape[0])
        chunk = image[start:end].to(
            device=processing_device,
            non_blocking=processing_device.type == "cuda",
        )
        result = _apply_fx_batch(chunk, **options)
        output[start:end].copy_(
            result,
            non_blocking=False,
        )
        del chunk, result
    return output


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
                "natural_saturation": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 2.0, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "brightness": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "performance_mode": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "advanced": True,
                    "tooltip": "auto 会遵守 ComfyUI 设备设置，并在可用时自动使用 CUDA。",
                }),
                "gpu_chunk_size": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 32,
                    "step": 1,
                    "advanced": True,
                    "tooltip": "每次送入显卡的帧数；720p 推荐 2–4，显存不足可降低。",
                }),
            },
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
        natural_saturation: float,
        saturation: float,
        contrast: float,
        brightness: float,
        performance_mode: str = "auto",
        gpu_chunk_size: int = 4,
    ):
        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            shape = getattr(image, "shape", None)
            raise ValueError(f"Expected IMAGE tensor [B,H,W,C], got {shape}")
        if image.shape[0] < 1:
            raise ValueError("IMAGE batch cannot be empty")

        options = {
            "enable_bloom": bool(enable_bloom),
            "bloom_threshold": float(bloom_threshold),
            "bloom_intensity": float(bloom_intensity),
            "bloom_radius": int(bloom_radius),
            "bloom_downsample": int(bloom_downsample),
            "enable_ca": bool(enable_ca),
            "ca_amount": float(ca_amount),
            "ca_angle_deg": float(ca_angle_deg),
            "ca_highlight_only": bool(ca_highlight_only),
            "enable_sharpen": bool(enable_sharpen),
            "sharpen_amount": float(sharpen_amount),
            "sharpen_radius": int(sharpen_radius),
            "natural_saturation": float(natural_saturation),
            "saturation": float(saturation),
            "contrast": float(contrast),
            "brightness": float(brightness),
        }
        requested_mode = str(performance_mode).strip().lower()
        processing_device = _resolve_processing_device(image, requested_mode)
        output_device = image.device
        chunk_size = max(1, min(int(gpu_chunk_size), image.shape[0]))
        if processing_device.type == "cpu":
            pixel_count = max(1, int(image.shape[1]) * int(image.shape[2]))
            cpu_cache_chunk = max(1, 1_000_000 // pixel_count)
            chunk_size = min(chunk_size, cpu_cache_chunk)

        while True:
            try:
                with torch.inference_mode():
                    output = _run_chunked(
                        image,
                        processing_device=processing_device,
                        output_device=output_device,
                        chunk_size=chunk_size,
                        options=options,
                    )
                return (output,)
            except torch.OutOfMemoryError:
                if processing_device.type == "cuda":
                    with torch.cuda.device(processing_device):
                        torch.cuda.empty_cache()
                if processing_device.type == "cuda" and chunk_size > 1:
                    chunk_size = max(1, chunk_size // 2)
                    continue
                if requested_mode == "auto" and processing_device.type == "cuda":
                    processing_device = torch.device("cpu")
                    continue
                raise RuntimeError(
                    "Fast Gift PostFX ran out of memory at one frame per chunk; "
                    "use performance_mode=cpu or free GPU memory"
                ) from None


NODE_CLASS_MAPPINGS = {
    "FastGiftPostFX": FastGiftPostFX,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FastGiftPostFX": "Fast Gift PostFX",
}
