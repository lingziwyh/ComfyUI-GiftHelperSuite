"""Shared contracts and color utilities for Gift Chroma Master.

ComfyUI IMAGE tensors are accepted only at the node boundary as BHWC. Every
processing function uses BCHW internally.  Keeping that contract explicit avoids
the ambiguous H=1/3/4 shape bug present in older versions.
"""

from __future__ import annotations

import re
from typing import Any, NamedTuple, Optional, Tuple

import torch


_HEX_COLOR = re.compile(r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


class AlphaConstraints(NamedTuple):
    """Prepared matte constraints kept in a cache-visible tuple."""

    source_alpha: Optional[torch.Tensor] = None
    source_alpha_mode: str = "ignore"
    inside_mask: Optional[torch.Tensor] = None
    outside_mask: Optional[torch.Tensor] = None


class CleanerDiagnostics(NamedTuple):
    """Full-resolution cleaner maps stored without hiding tensors in a dict."""

    edge_map: Optional[torch.Tensor] = None
    repair_map: Optional[torch.Tensor] = None
    detail_gate: Optional[torch.Tensor] = None
    temporal_gate: Optional[torch.Tensor] = None

    def get(self, key: str, default: Any = None) -> Any:
        value = getattr(self, key, default)
        return default if value is None else value


class SpillDiagnostics(NamedTuple):
    """Small aggregate spill statistics in a cache-visible tuple."""

    detected_mean: Optional[torch.Tensor] = None
    applied_mean: Optional[torch.Tensor] = None
    protected_mean: Optional[torch.Tensor] = None
    luma_error_mean: Optional[torch.Tensor] = None
    luma_error_max: Optional[torch.Tensor] = None


class GiftChromaMasterState(NamedTuple):
    """Opaque expert-pipeline state whose tensors ComfyUI can account for.

    ComfyUI's cache scanner recursively visits tuples/lists but not dictionaries.
    Keeping the state as a NamedTuple lets RAM/VRAM pressure eviction see every
    nested tensor while retaining readable named fields for the implementation.
    """

    schema: str = "gift_chroma_master_v1"
    stage: str = ""
    rgb_source_srgb: Optional[torch.Tensor] = None
    rgb_source_linear: Optional[torch.Tensor] = None
    screen_srgb: Optional[torch.Tensor] = None
    screen_linear: Optional[torch.Tensor] = None
    screen_confidence: Optional[torch.Tensor] = None
    screen_mix: Optional[torch.Tensor] = None
    alpha_raw: Optional[torch.Tensor] = None
    alpha_base: Optional[torch.Tensor] = None
    alpha_constraints: Optional[AlphaConstraints] = None
    alpha_keyed: Optional[torch.Tensor] = None
    alpha_clean: Optional[torch.Tensor] = None
    rgb_clean_linear: Optional[torch.Tensor] = None
    spill_map: Optional[torch.Tensor] = None
    cleaner_diagnostics: Optional[CleanerDiagnostics] = None
    spill_diagnostics: Optional[SpillDiagnostics] = None

    def __getitem__(self, key):
        if isinstance(key, str):
            return getattr(self, key)
        return tuple.__getitem__(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        value = getattr(self, key, default)
        return default if value is None else value


def srgb_to_linear(value: torch.Tensor) -> torch.Tensor:
    """Convert normalized sRGB to linear-light RGB without changing layout."""
    value = value.clamp(0.0, 1.0)
    return torch.where(
        value <= 0.04045,
        value / 12.92,
        ((value + 0.055) / 1.055).pow(2.4),
    )


def linear_to_srgb(value: torch.Tensor) -> torch.Tensor:
    """Convert normalized linear-light RGB to sRGB without changing layout."""
    value = value.clamp(0.0, 1.0)
    return torch.where(
        value <= 0.0031308,
        value * 12.92,
        1.055 * value.clamp_min(0.0).pow(1.0 / 2.4) - 0.055,
    )


def image_bhwc_to_bchw(image: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Validate a ComfyUI IMAGE and return RGB plus any embedded alpha in BCHW."""
    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        raise ValueError("image must be a ComfyUI IMAGE tensor with shape [B,H,W,C]")
    if not image.is_floating_point():
        raise TypeError(
            f"image must use normalized floating-point values, got {image.dtype}"
        )
    if not bool(torch.isfinite(image).all()):
        raise ValueError("image contains NaN or infinite values")
    channels = int(image.shape[-1])
    if channels not in (1, 3, 4):
        raise ValueError(
            f"image must use BHWC layout with 1, 3, or 4 channels; got {tuple(image.shape)}"
        )
    work = image.to(dtype=torch.float32).clamp(0.0, 1.0)
    if channels == 1:
        rgb = work.expand(*work.shape[:-1], 3)
        alpha = None
    else:
        rgb = work[..., :3]
        alpha = work[..., 3:4] if channels == 4 else None
    rgb_bchw = rgb.permute(0, 3, 1, 2).contiguous()
    alpha_bchw = None if alpha is None else alpha.permute(0, 3, 1, 2).contiguous()
    return rgb_bchw, alpha_bchw


def bchw_to_image(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a strict BCHW tensor to a contiguous ComfyUI BHWC IMAGE."""
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
        raise ValueError("tensor must have shape [B,C,H,W]")
    if int(tensor.shape[1]) not in (1, 3, 4):
        raise ValueError(f"BCHW tensor has unsupported channel count: {tuple(tensor.shape)}")
    return tensor.permute(0, 2, 3, 1).contiguous()


def mask_to_bchw(
    mask: Optional[torch.Tensor],
    *,
    batch: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str = "mask",
) -> Optional[torch.Tensor]:
    """Normalize common ComfyUI MASK layouts to [B,1,H,W], with safe broadcasting."""
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    value = mask
    if value.ndim == 2:
        value = value.unsqueeze(0).unsqueeze(0)
    elif value.ndim == 3:
        value = value.unsqueeze(1)
    elif value.ndim == 4:
        # Resolve layout against the expected spatial dimensions.  Merely
        # checking shape[1] first is ambiguous for legal BHWC masks whose image
        # height is one, e.g. [B,1,W,1].
        is_bchw = value.shape[1] == 1 and tuple(value.shape[-2:]) == (height, width)
        is_bhwc = value.shape[-1] == 1 and tuple(value.shape[1:3]) == (height, width)
        if is_bchw:
            pass
        elif is_bhwc:
            value = value.permute(0, 3, 1, 2)
        else:
            raise ValueError(
                f"{name} 4D layout/size does not match [B,1,{height},{width}] "
                f"or [B,{height},{width},1]; got {tuple(value.shape)}"
            )
    else:
        raise ValueError(f"{name} must be [H,W], [B,H,W], [B,1,H,W], or [B,H,W,1]")
    if tuple(value.shape[-2:]) != (height, width):
        raise ValueError(
            f"{name} spatial size {tuple(value.shape[-2:])} does not match image {(height, width)}"
        )
    if value.shape[0] not in (1, batch):
        raise ValueError(f"{name} batch {value.shape[0]} does not match image batch {batch}")
    value = value.to(device=device, dtype=dtype)
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains NaN or infinite values")
    value = value.clamp(0.0, 1.0)
    if value.shape[0] == 1 and batch > 1:
        value = value.expand(batch, 1, height, width)
    return value


def parse_color(
    value: Any,
    *,
    batch: int = 1,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    name: str = "color",
) -> torch.Tensor:
    """Strictly parse a color and return [B,3,1,1] in normalized sRGB.

    Accepted forms are #RGB/#RRGGBB, 0xRRGGBB integers, or three numeric values.
    Numeric triplets may consistently use either 0..1 or 0..255 ranges.
    """
    if isinstance(value, torch.Tensor):
        result = value.to(device=device, dtype=dtype)
        if result.ndim == 1 and result.shape[0] == 3:
            result = result.view(1, 3, 1, 1)
        elif result.ndim == 2 and result.shape[1] == 3:
            result = result[:, :, None, None]
        elif result.ndim == 4 and tuple(result.shape[1:]) == (3, 1, 1):
            pass
        else:
            raise ValueError(
                f"{name} tensor must be [3], [B,3], or [B,3,1,1]; "
                f"got {tuple(result.shape)}"
            )
        if result.shape[0] not in (1, int(batch)):
            raise ValueError(
                f"{name} tensor batch must be 1 or match image batch {batch}; "
                f"got {result.shape[0]}"
            )
        if not bool(torch.isfinite(result).all()):
            raise ValueError(f"{name} tensor contains NaN or infinite values")
        if bool((result < 0.0).any()) or bool((result > 1.0).any()):
            raise ValueError(f"{name} tensor channels must use normalized 0..1 values")
        if result.shape[0] == 1 and int(batch) > 1:
            result = result.expand(int(batch), -1, -1, -1)
        return result.contiguous()

    channels: list[float]
    if isinstance(value, bool):
        raise ValueError(f"{name} cannot be a boolean")
    if isinstance(value, int):
        if value < 0 or value > 0xFFFFFF:
            raise ValueError(f"{name} integer must be in 0x000000..0xFFFFFF")
        channels = [
            float((value >> 16) & 0xFF) / 255.0,
            float((value >> 8) & 0xFF) / 255.0,
            float(value & 0xFF) / 255.0,
        ]
    elif isinstance(value, str):
        text = value.strip()
        match = _HEX_COLOR.fullmatch(text)
        if match:
            digits = match.group(1)
            if len(digits) == 3:
                digits = "".join(char * 2 for char in digits)
            channels = [int(digits[index:index + 2], 16) / 255.0 for index in (0, 2, 4)]
        else:
            try:
                parts = [float(part.strip()) for part in text.split(",")]
            except ValueError as exc:
                raise ValueError(f"{name} must be #RGB, #RRGGBB, or r,g,b") from exc
            if len(parts) != 3:
                raise ValueError(f"{name} must contain exactly three channels")
            channels = _normalize_triplet(parts, name)
    elif isinstance(value, (tuple, list)) and len(value) == 3:
        try:
            channels = _normalize_triplet([float(channel) for channel in value], name)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain three numeric channels") from exc
    else:
        raise ValueError(f"unsupported {name} value: {value!r}")
    color = torch.tensor(channels, device=device, dtype=dtype).view(1, 3, 1, 1)
    return color.expand(int(batch), 3, 1, 1).contiguous()


def _normalize_triplet(channels: list[float], name: str) -> list[float]:
    values = torch.tensor(channels, dtype=torch.float64)
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} channels must be finite")
    low = float(values.min())
    high = float(values.max())
    if low < 0.0:
        raise ValueError(f"{name} channels cannot be negative")
    if high <= 1.0:
        return [float(channel) for channel in channels]
    if high <= 255.0 and all(float(channel).is_integer() for channel in channels):
        return [float(channel) / 255.0 for channel in channels]
    raise ValueError(f"{name} channels must consistently use 0..1 or integer 0..255 values")


def linear_composite(
    foreground_srgb: torch.Tensor,
    alpha: torch.Tensor,
    background_srgb: torch.Tensor,
) -> torch.Tensor:
    """Composite BCHW sRGB tensors in linear light and return sRGB."""
    fg_linear = srgb_to_linear(foreground_srgb)
    bg_linear = srgb_to_linear(background_srgb)
    return linear_to_srgb(fg_linear * alpha + bg_linear * (1.0 - alpha)).clamp(0.0, 1.0)


def ensure_state(state: Any, stage: Optional[str] = None) -> GiftChromaMasterState:
    """Validate the lightweight expert-node state container."""
    if isinstance(state, dict):
        if state.get("schema") != "gift_chroma_master_v1":
            raise ValueError("expected a Gift Chroma Master state")
        values = {
            name: state.get(name)
            for name in GiftChromaMasterState._fields
            if name in state
        }
        constraints = values.get("alpha_constraints")
        if isinstance(constraints, dict):
            values["alpha_constraints"] = AlphaConstraints(**constraints)
        cleaner = values.get("cleaner_diagnostics")
        if isinstance(cleaner, dict):
            values["cleaner_diagnostics"] = CleanerDiagnostics(**cleaner)
        spill = values.get("spill_diagnostics")
        if isinstance(spill, dict):
            values["spill_diagnostics"] = SpillDiagnostics(**spill)
        state = GiftChromaMasterState(**values)
    if (
        not isinstance(state, GiftChromaMasterState)
        or state.schema != "gift_chroma_master_v1"
    ):
        raise ValueError("expected a Gift Chroma Master state")
    valid_stages = {"keyed", "cleaned", "spilled"}
    if state.stage not in valid_stages:
        raise ValueError(
            "Gift Chroma Master state has invalid stage "
            f"{state.stage!r}; expected one of {sorted(valid_stages)}"
        )
    required = (
        "rgb_source_srgb",
        "rgb_source_linear",
        "screen_srgb",
        "screen_linear",
        "screen_confidence",
        "screen_mix",
        "alpha_raw",
        "alpha_base",
        "alpha_keyed",
        "alpha_clean",
        "rgb_clean_linear",
    )
    missing = [name for name in required if getattr(state, name) is None]
    if missing:
        raise ValueError(
            "Gift Chroma Master state stage "
            f"{state.stage!r} is missing required field(s): {', '.join(missing)}"
        )
    if stage is not None and state.stage != stage:
        raise ValueError(
            f"expected Gift Chroma Master state stage {stage!r}, got {state.stage!r}"
        )
    return state


def copy_state(state: Any, **updates: Any) -> GiftChromaMasterState:
    """Copy only the small state tuple while reusing tensor references."""
    current = ensure_state(state)
    unknown = set(updates).difference(GiftChromaMasterState._fields)
    if unknown:
        raise ValueError(f"unknown Gift Chroma Master state field(s): {sorted(unknown)}")
    return current._replace(**updates)
