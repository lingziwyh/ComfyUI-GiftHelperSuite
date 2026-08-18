"""Gift Chroma Master matte analysis.

This module intentionally has no ComfyUI dependency.  Images enter as strict
``[batch, channels, height, width]`` tensors, and all public functions return
the same layout.  The code keeps matte classification (``alpha_raw``) separate
from the estimated physical screen contribution (``screen_mix``), because
those signals have different jobs later in a key/clean/despill pipeline.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F


Tensor = torch.Tensor
ColorLike = Union[Tensor, Sequence[float]]

_SCREEN_MODES = {"manual", "green", "blue", "auto"}
_TEMPORAL_MODES = {"first_frame_locked", "batch_shared", "per_frame_smoothed"}


def srgb_to_linear(value: Tensor) -> Tensor:
    """Convert normalized sRGB values to scene-linear RGB.

    Values are clamped to the display-referred 0..1 domain expected by this
    keyer.  Linear light is used for mixture/projection maths; user-facing
    colors and returned sampled colors remain sRGB.
    """

    value = value.clamp(0.0, 1.0)
    return torch.where(
        value <= 0.04045,
        value / 12.92,
        ((value + 0.055) / 1.055).pow(2.4),
    )


def _require_bchw(
    value: Tensor,
    name: str,
    channels: Optional[int] = None,
) -> Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor in BCHW layout.")
    if value.ndim != 4:
        raise ValueError(
            f"{name} must have four dimensions [B,C,H,W]; got {tuple(value.shape)}."
        )
    if value.shape[0] < 1 or value.shape[2] < 1 or value.shape[3] < 1:
        raise ValueError(f"{name} must have non-empty batch and spatial dimensions.")
    if channels is not None and value.shape[1] != channels:
        raise ValueError(
            f"{name} must have exactly {channels} channel(s) in dimension 1; "
            f"got shape {tuple(value.shape)}. No BHWC/BCHW guessing is performed."
        )
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype; got {value.dtype}.")
    return value


def _validate_unit_float(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]; got {value!r}.")
    return result


def _working_float(value: Tensor) -> Tuple[Tensor, torch.dtype]:
    """Use float32 for numerically sensitive chroma maths, preserving output type."""

    original_dtype = value.dtype
    if value.dtype in (torch.float16, torch.bfloat16):
        return value.float(), original_dtype
    return value, original_dtype


def _prepare_color(
    color: ColorLike,
    batch: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> Tensor:
    """Return a color as ``[B,3,1,1]`` without implicit 0..255 normalization."""

    if isinstance(color, torch.Tensor):
        result = color.to(device=device, dtype=dtype)
    else:
        result = torch.as_tensor(color, device=device, dtype=dtype)

    if result.ndim == 1 and result.shape[0] == 3:
        result = result.view(1, 3, 1, 1)
    elif result.ndim == 2 and result.shape[1] == 3:
        result = result[:, :, None, None]
    elif result.ndim == 4 and result.shape[1:] == (3, 1, 1):
        pass
    else:
        raise ValueError(
            f"{name} must be [3], [B,3], or [B,3,1,1] with normalized sRGB values; "
            f"got shape {tuple(result.shape)}."
        )

    if result.shape[0] not in (1, batch):
        raise ValueError(
            f"{name} batch must be 1 or match the image batch ({batch}); "
            f"got {result.shape[0]}."
        )
    if result.shape[0] == 1 and batch > 1:
        result = result.expand(batch, -1, -1, -1)
    return result.clamp(0.0, 1.0)


def _safe_separable_gaussian(value: Tensor, sigma_px: float) -> Tensor:
    """Gaussian blur with continuous sigma and non-darkening image borders."""

    sigma = float(sigma_px)
    if not math.isfinite(sigma) or sigma < 0.0:
        raise ValueError(f"sigma_px must be finite and >= 0; got {sigma_px!r}.")
    if sigma < 1.0e-3:
        return value
    if sigma > 128.0:
        raise ValueError("sigma_px above 128 is not supported (would allocate excessive padding).")

    radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=value.device, dtype=value.dtype)
    kernel = torch.exp(-0.5 * (coords / max(sigma, 1.0e-3)).square())
    kernel = kernel / kernel.sum().clamp_min(torch.finfo(value.dtype).eps)
    channels = value.shape[1]

    pad_mode_x = "reflect" if value.shape[3] > radius else "replicate"
    horizontal = F.pad(value, (radius, radius, 0, 0), mode=pad_mode_x)
    horizontal = F.conv2d(
        horizontal,
        kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1),
        groups=channels,
    )

    pad_mode_y = "reflect" if value.shape[2] > radius else "replicate"
    vertical = F.pad(horizontal, (0, 0, radius, radius), mode=pad_mode_y)
    return F.conv2d(
        vertical,
        kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1),
        groups=channels,
    )


def gaussian_blur_bchw(value: Tensor, sigma_px: float) -> Tensor:
    """Blur any floating BCHW tensor without zero-padding its borders."""

    _require_bchw(value, "value")
    work, output_dtype = _working_float(value)
    return _safe_separable_gaussian(work, sigma_px).to(output_dtype)


def _border_pixels(rgb_srgb: Tensor, border_fraction: float) -> Tensor:
    fraction = float(border_fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction <= 0.5:
        raise ValueError(
            f"border_fraction must be finite and in (0, 0.5]; got {border_fraction!r}."
        )
    _, _, height, width = rgb_srgb.shape
    border = max(1, int(math.ceil(min(height, width) * fraction)))
    yy = torch.arange(height, device=rgb_srgb.device)[:, None]
    xx = torch.arange(width, device=rgb_srgb.device)[None, :]
    mask = (yy < border) | (yy >= height - border) | (xx < border) | (xx >= width - border)
    return rgb_srgb[:, :, mask]


def _chromaticity(rgb_linear: Tensor, dim: int) -> Tensor:
    eps = torch.finfo(rgb_linear.dtype).eps
    total = rgb_linear.sum(dim=dim, keepdim=True)
    neutral = torch.full_like(rgb_linear, 1.0 / 3.0)
    return torch.where(total > eps * 8.0, rgb_linear / total.clamp_min(eps), neutral)


def _cluster_auto_border(border_srgb: Tensor, max_cluster_samples: int) -> Tuple[Tensor, Tensor]:
    """Vectorized dominant-chroma clustering for arbitrary screen hues."""

    batch, _, samples = border_srgb.shape
    if max_cluster_samples < 16:
        raise ValueError("max_cluster_samples must be at least 16.")
    sample_count = min(samples, int(max_cluster_samples))
    sample_idx = torch.linspace(
        0,
        samples - 1,
        steps=sample_count,
        device=border_srgb.device,
    ).round().long()

    border_linear = srgb_to_linear(border_srgb)
    chroma = _chromaticity(border_linear, dim=1)
    sampled_chroma = chroma.index_select(2, sample_idx).transpose(1, 2)
    sampled_rgb = border_srgb.index_select(2, sample_idx)

    neutral = torch.full_like(sampled_chroma, 1.0 / 3.0)
    sampled_purity = torch.linalg.vector_norm(sampled_chroma - neutral, dim=2)
    sampled_brightness = sampled_rgb.amax(dim=1)
    sampled_quality = sampled_purity.square() * sampled_brightness.clamp_min(0.01)

    # A small, vectorized medoid search finds the densest saturated hue cluster.
    # 384 samples cost ~0.6 MiB/frame in float32 and avoid GPU-synchronizing loops.
    distances = torch.cdist(sampled_chroma, sampled_chroma, p=2)
    bandwidth = 0.085
    affinity = torch.exp(-0.5 * (distances / bandwidth).square())
    density = (affinity * sampled_quality[:, None, :]).sum(dim=2)
    seed_idx = density.argmax(dim=1)
    gather_idx = seed_idx[:, None, None].expand(batch, 1, 3)
    seed = sampled_chroma.gather(1, gather_idx).transpose(1, 2)

    full_distance = torch.linalg.vector_norm(chroma - seed, dim=1)
    membership = torch.exp(-0.5 * (full_distance / bandwidth).square())
    purity = torch.linalg.vector_norm(
        chroma - torch.full_like(chroma, 1.0 / 3.0), dim=1
    )
    brightness = border_srgb.amax(dim=1)
    quality = purity.square() * brightness.clamp_min(0.01)
    weights = membership * (quality + 0.01 * brightness)
    denominator = weights.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    color = (border_srgb * weights[:, None, :]).sum(dim=2) / denominator

    quality_total = quality.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    support = (membership * quality).sum(dim=1, keepdim=True) / quality_total
    coherence = (membership.square() * weights).sum(dim=1, keepdim=True) / denominator
    sampled_color_linear = srgb_to_linear(color)
    sampled_color_chroma = _chromaticity(sampled_color_linear, dim=1)
    color_purity = torch.linalg.vector_norm(
        sampled_color_chroma - torch.full_like(sampled_color_chroma, 1.0 / 3.0), dim=1
    ).unsqueeze(1)
    purity_confidence = (color_purity / 0.45).clamp(0.0, 1.0)
    confidence = (support * coherence).sqrt() * purity_confidence
    return color, confidence.clamp(0.0, 1.0)


def _cluster_primary_border(border_srgb: Tensor, channel: int) -> Tuple[Tensor, Tensor]:
    """Robustly sample a requested green/blue hue from border pixels."""

    border_linear = srgb_to_linear(border_srgb)
    chroma = _chromaticity(border_linear, dim=1)
    selected = chroma[:, channel, :]
    others = (chroma.sum(dim=1) - selected) * 0.5
    dominance = (selected - others).clamp_min(0.0)
    brightness = border_srgb.amax(dim=1)
    weights = dominance.square() * brightness.clamp_min(0.01)
    denominator = weights.sum(dim=1, keepdim=True)
    safe_denominator = denominator.clamp_min(1.0e-8)
    sampled = (border_srgb * weights[:, None, :]).sum(dim=2) / safe_denominator

    canonical = torch.zeros_like(sampled)
    canonical[:, channel] = 1.0
    has_evidence = denominator > 1.0e-7
    color = torch.where(has_evidence.expand_as(sampled), sampled, canonical)

    chroma_energy = torch.linalg.vector_norm(
        chroma - torch.full_like(chroma, 1.0 / 3.0), dim=1
    ).sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    support = dominance.sum(dim=1, keepdim=True) / chroma_energy
    mean_dominance = (weights * dominance).sum(dim=1, keepdim=True) / safe_denominator
    confidence = support.clamp(0.0, 1.0) * (mean_dominance / 0.45).clamp(0.0, 1.0)
    confidence = torch.where(has_evidence, confidence, torch.zeros_like(confidence))
    return color, confidence.clamp(0.0, 1.0)


def _apply_temporal_policy(
    colors: Tensor,
    confidence: Tensor,
    temporal_mode: str,
    temporal_smoothing: float,
) -> Tuple[Tensor, Tensor]:
    batch = colors.shape[0]
    if temporal_mode == "first_frame_locked":
        return colors[0:1].expand(batch, -1), confidence[0:1].expand(batch, -1)
    if temporal_mode == "batch_shared":
        weights = confidence.clamp_min(0.025)
        shared = (colors * weights).sum(dim=0, keepdim=True) / weights.sum(
            dim=0, keepdim=True
        ).clamp_min(1.0e-8)
        shared_confidence = (confidence * weights).sum(dim=0, keepdim=True) / weights.sum(
            dim=0, keepdim=True
        ).clamp_min(1.0e-8)
        return shared.expand(batch, -1), shared_confidence.expand(batch, -1)

    smoothing = _validate_unit_float(temporal_smoothing, "temporal_smoothing")
    if batch == 1 or smoothing <= 0.0:
        return colors, confidence

    # Confidence-weighted causal smoothing with two safeguards for real clips:
    # a large chromaticity jump resets the history at a shot/screen change, and
    # a large exposure residual temporarily shortens the smoothing memory.  The
    # latter tracks lighting changes instead of lagging several frames behind.
    evidence = confidence.clamp_min(0.025)
    color_chroma = _chromaticity(
        srgb_to_linear(colors[:, :, None, None]), dim=1
    ).squeeze(-1).squeeze(-1)
    weighted_color = colors[0:1] * evidence[0:1]
    weight_sum = evidence[0:1]
    weighted_confidence = confidence[0:1] * evidence[0:1]
    output_colors = [colors[0:1]]
    output_confidence = [confidence[0:1]]

    base_decay = colors.new_tensor(smoothing).view(1, 1)
    for index in range(1, batch):
        previous_color = weighted_color / weight_sum.clamp_min(1.0e-8)
        previous_chroma = _chromaticity(
            srgb_to_linear(previous_color[:, :, None, None]), dim=1
        ).squeeze(-1).squeeze(-1)
        hue_delta = torch.linalg.vector_norm(
            color_chroma[index : index + 1] - previous_chroma,
            dim=1,
            keepdim=True,
        )
        reliable_cut = (
            (confidence[index : index + 1] > 0.12)
            & (output_confidence[-1] > 0.12)
            & (hue_delta > 0.28)
        )

        residual = (
            colors[index : index + 1] - previous_color
        ).abs().amax(dim=1, keepdim=True)
        change = ((residual - 0.025) / 0.15).clamp(0.0, 1.0)
        change = change * change * (3.0 - 2.0 * change)
        current_reliability = 0.25 + 0.75 * confidence[index : index + 1].clamp(0.0, 1.0)
        adaptive_decay = base_decay * (1.0 - 0.85 * change * current_reliability)
        adaptive_decay = torch.where(
            reliable_cut,
            torch.zeros_like(adaptive_decay),
            adaptive_decay,
        )

        current_evidence = evidence[index : index + 1]
        weighted_color = (
            colors[index : index + 1] * current_evidence
            + adaptive_decay * weighted_color
        )
        weight_sum = current_evidence + adaptive_decay * weight_sum
        weighted_confidence = (
            confidence[index : index + 1] * current_evidence
            + adaptive_decay * weighted_confidence
        )
        filtered_color = weighted_color / weight_sum.clamp_min(1.0e-8)
        filtered_confidence = (
            weighted_confidence / weight_sum.clamp_min(1.0e-8)
        ).clamp(0.0, 1.0)
        output_colors.append(filtered_color)
        output_confidence.append(filtered_confidence)

    return torch.cat(output_colors, dim=0), torch.cat(output_confidence, dim=0)


def apply_screen_temporal_policy(
    colors: Tensor,
    confidence: Tensor,
    *,
    temporal_mode: str,
    temporal_smoothing: float,
) -> Tuple[Tensor, Tensor]:
    """Apply the public clip-level screen-colour policy to precomputed candidates.

    This small-tensor entry point lets the accelerated daily node estimate raw
    per-frame candidates in bounded GPU chunks, then perform the exact causal,
    first-frame, or batch-shared policy once across the complete clip.
    """
    color_4d = colors.ndim == 4
    confidence_4d = confidence.ndim == 4
    color_values = colors.squeeze(-1).squeeze(-1) if color_4d else colors
    confidence_values = (
        confidence.squeeze(-1).squeeze(-1) if confidence_4d else confidence
    )
    if color_values.ndim != 2 or color_values.shape[1] != 3:
        raise ValueError("colors must be [B,3] or [B,3,1,1]")
    if confidence_values.ndim != 2 or confidence_values.shape[1] != 1:
        raise ValueError("confidence must be [B,1] or [B,1,1,1]")
    if confidence_values.shape[0] != color_values.shape[0]:
        raise ValueError("colors and confidence must have matching batches")
    if confidence_values.device != color_values.device:
        confidence_values = confidence_values.to(color_values.device)
    confidence_values = confidence_values.to(dtype=color_values.dtype)
    if not bool(torch.isfinite(color_values).all()) or not bool(
        torch.isfinite(confidence_values).all()
    ):
        raise ValueError("screen color candidates must be finite")
    mode = str(temporal_mode).lower()
    if mode not in _TEMPORAL_MODES:
        raise ValueError(
            f"temporal_mode must be one of {sorted(_TEMPORAL_MODES)}; "
            f"got {temporal_mode!r}."
        )
    filtered_color, filtered_confidence = _apply_temporal_policy(
        color_values.clamp(0.0, 1.0),
        confidence_values.clamp(0.0, 1.0),
        mode,
        temporal_smoothing,
    )
    if color_4d:
        filtered_color = filtered_color[:, :, None, None]
    if confidence_4d:
        filtered_confidence = filtered_confidence[:, :, None, None]
    return filtered_color, filtered_confidence


def estimate_screen_color(
    rgb_srgb: Tensor,
    *,
    screen_mode: str = "auto",
    manual_screen_color: Optional[ColorLike] = None,
    temporal_mode: str = "first_frame_locked",
    border_fraction: float = 0.08,
    temporal_smoothing: float = 0.80,
    max_cluster_samples: int = 384,
) -> Tuple[Tensor, Tensor]:
    """Estimate a screen color from image borders.

    Args:
        rgb_srgb: Normalized sRGB image in strict ``[B,3,H,W]`` layout.
        screen_mode: ``manual``, ``green``, ``blue``, or arbitrary-hue ``auto``.
        manual_screen_color: Normalized sRGB color used by manual mode.
        temporal_mode: Treat the batch as a clip and use the first estimate,
            one shared estimate, or a causal confidence-weighted smooth estimate.
        border_fraction: Fraction of the shorter image side sampled at all edges.
        temporal_smoothing: Prior-frame weight for ``per_frame_smoothed``.

    Returns:
        ``(screen_color, confidence)`` with shapes ``[B,3,1,1]`` and
        ``[B,1,1,1]``.  Sampling is vectorized over B; adaptive causal
        smoothing uses a lightweight ordered recurrence so cuts can reset it.
    """

    _require_bchw(rgb_srgb, "rgb_srgb", channels=3)
    mode = str(screen_mode).lower()
    temporal = str(temporal_mode).lower()
    if mode not in _SCREEN_MODES:
        raise ValueError(f"screen_mode must be one of {sorted(_SCREEN_MODES)}; got {screen_mode!r}.")
    if temporal not in _TEMPORAL_MODES:
        raise ValueError(
            f"temporal_mode must be one of {sorted(_TEMPORAL_MODES)}; got {temporal_mode!r}."
        )
    _validate_unit_float(temporal_smoothing, "temporal_smoothing")

    work, output_dtype = _working_float(rgb_srgb)
    work = work.clamp(0.0, 1.0)
    batch = work.shape[0]

    if mode == "manual":
        if manual_screen_color is None:
            raise ValueError("manual_screen_color is required when screen_mode='manual'.")
        color = _prepare_color(
            manual_screen_color,
            batch,
            device=work.device,
            dtype=work.dtype,
            name="manual_screen_color",
        ).squeeze(-1).squeeze(-1)
        confidence = torch.ones((batch, 1), device=work.device, dtype=work.dtype)
    else:
        sampling_work = work[0:1] if temporal == "first_frame_locked" else work
        border = _border_pixels(sampling_work, border_fraction)
        if mode == "green":
            color, confidence = _cluster_primary_border(border, channel=1)
        elif mode == "blue":
            color, confidence = _cluster_primary_border(border, channel=2)
        else:
            color, confidence = _cluster_auto_border(border, int(max_cluster_samples))

    if temporal == "first_frame_locked" and color.shape[0] == 1 and batch > 1:
        color = color.expand(batch, -1)
        confidence = confidence.expand(batch, -1)
    else:
        color, confidence = _apply_temporal_policy(
            color,
            confidence,
            temporal,
            temporal_smoothing,
        )
    return (
        color[:, :, None, None].to(output_dtype),
        confidence[:, :, None, None].to(output_dtype),
    )


def compute_raw_matte(
    rgb_srgb: Tensor,
    screen_color_srgb: ColorLike,
    *,
    screen_gain: float = 1.0,
    screen_balance: float = 0.50,
    alpha_bias_color: Optional[ColorLike] = None,
    preblur_px: float = 0.0,
    return_diagnostics: bool = False,
) -> Dict[str, Tensor]:
    """Compute unclipped alpha and a separate physical screen-mixture estimate.

    The analysis is performed in linear RGB.  A full opponent vector runs from
    ``alpha_bias_color`` (neutral gray by default) to the sampled screen color,
    so yellow, cyan, magenta, red, and other manual keys do not collapse to a
    single max-channel vote.

    ``screen_balance`` blends conservative linear-mixture evidence (0) with
    illumination-invariant chromaticity evidence (1).  ``screen_mix`` remains
    the conservative, gain-independent mixture estimate for later recovery and
    despill stages; it is deliberately not the same tensor as ``1-alpha_raw``.
    """

    _require_bchw(rgb_srgb, "rgb_srgb", channels=3)
    gain = float(screen_gain)
    if not math.isfinite(gain) or gain < 0.0:
        raise ValueError(f"screen_gain must be finite and >= 0; got {screen_gain!r}.")
    balance = _validate_unit_float(screen_balance, "screen_balance")

    work, output_dtype = _working_float(rgb_srgb)
    work = work.clamp(0.0, 1.0)
    batch = work.shape[0]
    key_srgb = _prepare_color(
        screen_color_srgb,
        batch,
        device=work.device,
        dtype=work.dtype,
        name="screen_color_srgb",
    )

    image_linear = srgb_to_linear(work)
    analysis_linear = _safe_separable_gaussian(image_linear, preblur_px)
    key_linear = srgb_to_linear(key_srgb)

    if alpha_bias_color is None:
        # Match the key's linear energy so the default neutral reference does
        # not inject an arbitrary exposure into the physical projection.
        bias_linear = key_linear.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)
    else:
        bias_srgb = _prepare_color(
            alpha_bias_color,
            batch,
            device=work.device,
            dtype=work.dtype,
            name="alpha_bias_color",
        )
        bias_linear = srgb_to_linear(bias_srgb)

    key_chroma = _chromaticity(key_linear, dim=1)
    bias_chroma = _chromaticity(bias_linear, dim=1)
    image_chroma = _chromaticity(analysis_linear, dim=1)

    chroma_axis = key_chroma - bias_chroma
    chroma_denominator = chroma_axis.square().sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    relative_chroma = image_chroma - bias_chroma
    chroma_coordinate = (
        relative_chroma * chroma_axis
    ).sum(dim=1, keepdim=True) / chroma_denominator
    perpendicular = relative_chroma - chroma_coordinate * chroma_axis
    normalized_perpendicular = torch.linalg.vector_norm(perpendicular, dim=1, keepdim=True) / (
        chroma_denominator.sqrt() + 1.0e-8
    )

    # Direction tolerance is intentionally wider near the neutral/bias end and
    # tighter near full screen, preserving gray foreground while accepting
    # compression/noise on a saturated screen.
    coordinate_unit = chroma_coordinate.clamp(0.0, 1.0)
    tolerance = 0.42 - 0.18 * coordinate_unit
    direction_match = torch.exp(-0.5 * (normalized_perpendicular / tolerance).square())
    chromatic_screen = coordinate_unit * direction_match

    linear_axis = key_linear - bias_linear
    linear_denominator = linear_axis.square().sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    linear_coordinate = (
        (analysis_linear - bias_linear) * linear_axis
    ).sum(dim=1, keepdim=True) / linear_denominator
    screen_mix = linear_coordinate.clamp(0.0, 1.0) * direction_match

    background_evidence = torch.lerp(screen_mix, chromatic_screen, balance)
    background_evidence = (background_evidence * gain).clamp(0.0, 1.0)
    alpha_raw = (1.0 - background_evidence).clamp(0.0, 1.0)

    result: Dict[str, Tensor] = {
        "alpha_raw": alpha_raw.to(output_dtype),
        "screen_mix": screen_mix.clamp(0.0, 1.0).to(output_dtype),
    }
    if return_diagnostics:
        result.update(
            {
                "chromatic_screen": chromatic_screen.to(output_dtype),
                "background_evidence": background_evidence.to(output_dtype),
                "direction_match": direction_match.to(output_dtype),
                "chroma_coordinate": chroma_coordinate.to(output_dtype),
                "perpendicular_error": normalized_perpendicular.to(output_dtype),
                "analysis_rgb_linear": analysis_linear.to(output_dtype),
                "opponent_vector": chroma_axis.to(output_dtype),
            }
        )
    return result


def _morph_integer(alpha: Tensor, radius: int, grow: bool) -> Tensor:
    if radius <= 0:
        return alpha
    if radius > 256:
        raise ValueError("Absolute shrink/grow above 256 px is not supported.")
    padded = F.pad(alpha, (radius, radius, radius, radius), mode="replicate")
    kernel = 2 * radius + 1
    def square_max(value: Tensor) -> Tensor:
        # A square max filter is exactly separable.  Two 1D pools preserve the
        # same morphology while avoiding the very costly KxK CPU kernel.
        horizontal = F.max_pool2d(value, kernel_size=(1, kernel), stride=1)
        return F.max_pool2d(horizontal, kernel_size=(kernel, 1), stride=1)
    if grow:
        return square_max(padded)
    return 1.0 - square_max(1.0 - padded)


def subpixel_morph(alpha: Tensor, shrink_grow_px: float) -> Tensor:
    """Subpixel grayscale morphology; positive grows foreground, negative shrinks."""

    _require_bchw(alpha, "alpha", channels=1)
    amount = float(shrink_grow_px)
    if not math.isfinite(amount):
        raise ValueError(f"shrink_grow_px must be finite; got {shrink_grow_px!r}.")
    magnitude = abs(amount)
    if magnitude < 1.0e-6:
        return alpha
    if magnitude > 256.0:
        raise ValueError("Absolute shrink_grow_px above 256 is not supported.")
    lower_radius = int(math.floor(magnitude))
    fraction = magnitude - lower_radius
    grow = amount > 0.0
    lower = _morph_integer(alpha, lower_radius, grow)
    if fraction <= 1.0e-7:
        return lower
    upper = _morph_integer(alpha, lower_radius + 1, grow)
    return torch.lerp(lower, upper, fraction)


def finish_matte(
    alpha_raw: Tensor,
    *,
    clip_black: float = 0.0,
    clip_white: float = 1.0,
    clip_rollback: float = 0.0,
    shrink_grow_px: float = 0.0,
    softness_px: float = 0.0,
    return_diagnostics: bool = False,
) -> Union[Tensor, Dict[str, Tensor]]:
    """Finish a raw matte with clipping, rollback, morphology, and softness.

    ``clip_rollback`` restores unclipped values only inside the transition band,
    retaining exact black/white endpoints.  Morphology is linearly interpolated
    between integer radii, so fractional pixel controls are continuous.  All
    spatial operations use reflected or replicated borders and cannot create a
    dark frame around a constant matte.
    """

    _require_bchw(alpha_raw, "alpha_raw", channels=1)
    black = _validate_unit_float(clip_black, "clip_black")
    white = _validate_unit_float(clip_white, "clip_white")
    rollback = _validate_unit_float(clip_rollback, "clip_rollback")
    if white <= black:
        raise ValueError(
            f"clip_white must be greater than clip_black; got {white} <= {black}."
        )

    work, output_dtype = _working_float(alpha_raw)
    raw = work.clamp(0.0, 1.0)
    clipped = ((raw - black) / (white - black)).clamp(0.0, 1.0)
    transition = (4.0 * clipped * (1.0 - clipped)).clamp(0.0, 1.0)
    rolled = torch.lerp(clipped, raw, rollback * transition).clamp(0.0, 1.0)
    morphed = subpixel_morph(rolled, shrink_grow_px).clamp(0.0, 1.0)
    softened = _safe_separable_gaussian(morphed, softness_px).clamp(0.0, 1.0)

    if not return_diagnostics:
        return softened.to(output_dtype)
    return {
        "alpha": softened.to(output_dtype),
        "alpha_clipped": clipped.to(output_dtype),
        "alpha_rollback": rolled.to(output_dtype),
        "alpha_morphed": morphed.to(output_dtype),
    }


def run_v5_keyer(
    rgb_srgb: Tensor,
    *,
    screen_mode: str = "auto",
    manual_screen_color: Optional[ColorLike] = None,
    temporal_mode: str = "first_frame_locked",
    border_fraction: float = 0.08,
    temporal_smoothing: float = 0.80,
    max_cluster_samples: int = 384,
    screen_gain: float = 1.0,
    screen_balance: float = 0.50,
    alpha_bias_color: Optional[ColorLike] = None,
    preblur_px: float = 0.0,
    clip_black: float = 0.0,
    clip_white: float = 1.0,
    clip_rollback: float = 0.0,
    shrink_grow_px: float = 0.0,
    softness_px: float = 0.0,
    return_diagnostics: bool = False,
) -> Dict[str, Tensor]:
    """Run screen sampling, raw key analysis, and matte finishing.

    The compact result always contains ``alpha``, ``alpha_raw``, ``screen_mix``,
    ``screen_color``, and ``screen_confidence``.  Setting
    ``return_diagnostics=True`` opts into additional full-resolution maps.
    """

    screen_color, confidence = estimate_screen_color(
        rgb_srgb,
        screen_mode=screen_mode,
        manual_screen_color=manual_screen_color,
        temporal_mode=temporal_mode,
        border_fraction=border_fraction,
        temporal_smoothing=temporal_smoothing,
        max_cluster_samples=max_cluster_samples,
    )
    raw = compute_raw_matte(
        rgb_srgb,
        screen_color,
        screen_gain=screen_gain,
        screen_balance=screen_balance,
        alpha_bias_color=alpha_bias_color,
        preblur_px=preblur_px,
        return_diagnostics=return_diagnostics,
    )
    finished = finish_matte(
        raw["alpha_raw"],
        clip_black=clip_black,
        clip_white=clip_white,
        clip_rollback=clip_rollback,
        shrink_grow_px=shrink_grow_px,
        softness_px=softness_px,
        return_diagnostics=return_diagnostics,
    )

    if return_diagnostics:
        assert isinstance(finished, dict)
        result = dict(raw)
        result.update(finished)
    else:
        assert isinstance(finished, torch.Tensor)
        result = {
            "alpha": finished,
            "alpha_raw": raw["alpha_raw"],
            "screen_mix": raw["screen_mix"],
        }
    result["screen_color"] = screen_color
    result["screen_confidence"] = confidence
    return result


__all__ = [
    "apply_screen_temporal_policy",
    "compute_raw_matte",
    "estimate_screen_color",
    "finish_matte",
    "gaussian_blur_bchw",
    "run_v5_keyer",
    "srgb_to_linear",
    "subpixel_morph",
]
