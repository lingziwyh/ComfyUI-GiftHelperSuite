"""Edge-local matte cleanup for Gift Chroma Master.

This module deliberately has no knowledge of ComfyUI tensor layouts.  Its public
contract is strict BCHW: RGB is ``[B, 3, H, W]`` and alpha is
``[B, 1, H, W]``.  Temporal cleanup treats the batch dimension as an ordered
sequence only when ``temporal_mode="fast"`` is selected explicitly.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


Tensor = torch.Tensor


def _clamp_float(value: float, low: float, high: float, name: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return max(low, min(high, value))


def _validate_inputs(rgb: Tensor, alpha: Tensor, effect_mask: Optional[Tensor]) -> None:
    if not torch.is_tensor(rgb) or not torch.is_tensor(alpha):
        raise TypeError("rgb and alpha must be torch tensors")
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("rgb must use strict BCHW layout [B, 3, H, W]")
    if alpha.ndim != 4 or alpha.shape[1] != 1:
        raise ValueError("alpha must use strict BCHW layout [B, 1, H, W]")
    if rgb.shape[0] != alpha.shape[0] or rgb.shape[-2:] != alpha.shape[-2:]:
        raise ValueError("rgb and alpha must have matching batch and spatial dimensions")
    if rgb.device != alpha.device:
        raise ValueError("rgb and alpha must be on the same device")
    if not rgb.dtype.is_floating_point or not alpha.dtype.is_floating_point:
        raise TypeError("rgb and alpha must use floating-point dtypes")
    if rgb.shape[0] < 1 or rgb.shape[-2] < 1 or rgb.shape[-1] < 1:
        raise ValueError("rgb and alpha may not contain empty dimensions")

    if effect_mask is None:
        return
    if not torch.is_tensor(effect_mask):
        raise TypeError("effect_mask must be a torch tensor when provided")
    if effect_mask.ndim != 4 or effect_mask.shape[1] != 1:
        raise ValueError("effect_mask must use BCHW layout [B, 1, H, W]")
    if effect_mask.shape[0] not in (1, alpha.shape[0]):
        raise ValueError("effect_mask batch must be 1 or match alpha batch")
    if effect_mask.shape[-2:] != alpha.shape[-2:]:
        raise ValueError("effect_mask must match alpha spatial dimensions")
    if effect_mask.device != alpha.device:
        raise ValueError("effect_mask and alpha must be on the same device")
    if not effect_mask.dtype.is_floating_point:
        raise TypeError("effect_mask must use a floating-point dtype")


def _safe_pad(x: Tensor, left: int, right: int, top: int, bottom: int) -> Tensor:
    """Pad without introducing zero-valued image borders.

    Reflection is preferred for normal images.  Replication is the safe fallback
    for one-pixel dimensions and for radii too large for PyTorch reflection pad.
    """

    if left == right == top == bottom == 0:
        return x
    can_reflect = (
        x.shape[-1] > max(left, right)
        and x.shape[-2] > max(top, bottom)
        and x.shape[-1] > 1
        and x.shape[-2] > 1
    )
    mode = "reflect" if can_reflect else "replicate"
    return F.pad(x, (left, right, top, bottom), mode=mode)


def _box_filter_integer(x: Tensor, radius: int) -> Tensor:
    if radius <= 0:
        return x
    padded = _safe_pad(x, radius, radius, radius, radius)
    return F.avg_pool2d(padded, kernel_size=2 * radius + 1, stride=1)


def _box_filter(x: Tensor, radius: float) -> Tensor:
    """Box filter with linear interpolation between integer support radii."""

    radius = max(0.0, float(radius))
    lower = int(math.floor(radius))
    upper = int(math.ceil(radius))
    low_value = _box_filter_integer(x, lower)
    if lower == upper:
        return low_value
    high_value = _box_filter_integer(x, upper)
    return low_value.lerp(high_value, radius - lower)


def _gaussian_blur(x: Tensor, sigma: float) -> Tensor:
    sigma = max(0.0, float(sigma))
    if sigma < 1.0e-4:
        return x
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum().clamp_min(torch.finfo(x.dtype).eps)

    channels = x.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    result = F.conv2d(_safe_pad(x, radius, radius, 0, 0), horizontal, groups=channels)
    result = F.conv2d(_safe_pad(result, 0, 0, radius, radius), vertical, groups=channels)
    return result


def _morph_integer(alpha: Tensor, radius: int, dilate: bool) -> Tensor:
    if radius <= 0:
        return alpha
    padded = _safe_pad(alpha, radius, radius, radius, radius)
    kernel = 2 * radius + 1
    def square_max(value: Tensor) -> Tensor:
        horizontal = F.max_pool2d(value, kernel_size=(1, kernel), stride=1)
        return F.max_pool2d(horizontal, kernel_size=(kernel, 1), stride=1)
    if dilate:
        return square_max(padded)
    return -square_max(-padded)


def _morph(alpha: Tensor, radius: float, dilate: bool) -> Tensor:
    """Morphology with a continuously variable effective radius."""

    radius = max(0.0, float(radius))
    lower = int(math.floor(radius))
    upper = int(math.ceil(radius))
    low_value = _morph_integer(alpha, lower, dilate)
    if lower == upper:
        return low_value
    high_value = _morph_integer(alpha, upper, dilate)
    return low_value.lerp(high_value, radius - lower)


def _smoothstep(low: float, high: float, value: Tensor) -> Tensor:
    position = ((value - low) / max(high - low, 1.0e-6)).clamp(0.0, 1.0)
    return position * position * (3.0 - 2.0 * position)


def _luminance(rgb: Tensor) -> Tensor:
    return (
        rgb[:, 0:1] * 0.2126
        + rgb[:, 1:2] * 0.7152
        + rgb[:, 2:3] * 0.0722
    )


def _joint_guided_filter_integer(
    rgb: Tensor,
    alpha: Tensor,
    radius: int,
    epsilon: float,
) -> Tensor:
    if radius <= 0:
        return alpha

    # Evaluate the four scalar guidance models one at a time.  Their maths is
    # independent, so accumulating the weighted predictions preserves the
    # exact result while avoiding several simultaneous four-channel temporary
    # tensors.  This materially lowers RAM/VRAM pressure and memory bandwidth.
    mean_alpha = _box_filter_integer(alpha, radius)
    weighted_prediction = torch.zeros_like(alpha)
    total_weight = torch.zeros_like(alpha)
    guidance_channels = (
        (_luminance(rgb), 1.5),
        (rgb[:, 0:1], 1.0),
        (rgb[:, 1:2], 1.0),
        (rgb[:, 2:3], 1.0),
    )
    for guidance, priority in guidance_channels:
        mean_guidance = _box_filter_integer(guidance, radius)
        correlation_guidance = _box_filter_integer(guidance * guidance, radius)
        correlation_cross = _box_filter_integer(guidance * alpha, radius)
        variance = (
            correlation_guidance - mean_guidance * mean_guidance
        ).clamp_min(0.0)
        covariance = correlation_cross - mean_guidance * mean_alpha
        slope = covariance / (variance + epsilon)
        intercept = mean_alpha - slope * mean_guidance
        prediction = (
            _box_filter_integer(slope, radius) * guidance
            + _box_filter_integer(intercept, radius)
        )
        reliability = variance / (variance + epsilon)
        weights = (0.04 + reliability) * priority
        weighted_prediction.add_(prediction * weights)
        total_weight.add_(weights)

    return weighted_prediction / total_weight.clamp_min(torch.finfo(alpha.dtype).eps)


def _joint_guided_filter(
    rgb: Tensor,
    alpha: Tensor,
    radius: float,
    epsilon: float,
) -> Tensor:
    radius = max(0.0, float(radius))
    lower = int(math.floor(radius))
    upper = int(math.ceil(radius))
    low_value = _joint_guided_filter_integer(rgb, alpha, lower, epsilon)
    if lower == upper:
        return low_value
    high_value = _joint_guided_filter_integer(rgb, alpha, upper, epsilon)
    return low_value.lerp(high_value, radius - lower)


def _sobel(x: Tensor) -> Tuple[Tensor, Tensor]:
    kernel_x = x.new_tensor(
        ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0))
    ).view(1, 1, 3, 3) / 8.0
    kernel_y = kernel_x.transpose(-1, -2).contiguous()
    channels = x.shape[1]
    padded = _safe_pad(x, 1, 1, 1, 1)
    # Keep each channel's X/Y kernels adjacent so a single grouped convolution
    # produces exactly the same gradients with half the launch/dispatch work.
    kernels = torch.cat((kernel_x, kernel_y), dim=0).repeat(channels, 1, 1, 1)
    gradients = F.conv2d(padded, kernels, groups=channels)
    return gradients[:, 0::2], gradients[:, 1::2]


def _detail_correlation(rgb: Tensor, alpha: Tensor, edge_band: Tensor) -> Tensor:
    """Return alpha/image edge agreement used to protect supported fine detail."""

    combined_x, combined_y = _sobel(torch.cat((alpha, rgb), dim=1))
    alpha_x, rgb_x = combined_x[:, :1], combined_x[:, 1:]
    alpha_y, rgb_y = combined_y[:, :1], combined_y[:, 1:]
    epsilon = torch.finfo(alpha.dtype).eps

    alpha_magnitude = torch.sqrt(alpha_x * alpha_x + alpha_y * alpha_y + epsilon)
    rgb_magnitude = torch.sqrt(rgb_x * rgb_x + rgb_y * rgb_y + epsilon)
    directional_dot = torch.abs(rgb_x * alpha_x + rgb_y * alpha_y)
    alignment = directional_dot / (rgb_magnitude * alpha_magnitude + 1.0e-6)
    alignment = alignment.amax(dim=1, keepdim=True).clamp(0.0, 1.0)

    strongest_colour_edge = rgb_magnitude.amax(dim=1, keepdim=True)
    image_support = _smoothstep(0.003, 0.055, strongest_colour_edge)
    matte_support = _smoothstep(0.001, 0.060, alpha_magnitude)
    return (alignment * image_support * matte_support * edge_band).clamp(0.0, 1.0)


def _contrast_alpha(alpha: Tensor, amount: float) -> Tensor:
    if abs(amount) < 1.0e-6:
        return alpha
    epsilon = 1.0e-5
    scale = 2.0 ** (2.0 * amount)
    safe = alpha.clamp(epsilon, 1.0 - epsilon)
    logits = torch.log(safe) - torch.log1p(-safe)
    contrasted = torch.sigmoid(logits * scale)
    # Keep exact solid regions exact.  This also prevents endpoint drift after
    # repeated application in a video workflow.
    contrasted = torch.where(alpha <= 0.0, torch.zeros_like(contrasted), contrasted)
    return torch.where(alpha >= 1.0, torch.ones_like(contrasted), contrasted)


def _temporal_neighbor_weight(
    current_rgb: Tensor,
    neighbour_rgb: Tensor,
    current_alpha: Tensor,
    neighbour_alpha: Tensor,
    motion_threshold: float,
    scene_cut_threshold: float,
    structural_radius: int,
) -> Tensor:
    colour_delta = (current_rgb - neighbour_rgb).abs().mean(dim=1, keepdim=True)
    scene_delta = colour_delta.mean(dim=(2, 3), keepdim=True)

    local_gate = 1.0 - _smoothstep(0.25 * motion_threshold, motion_threshold, colour_delta)
    scene_gate = 1.0 - _smoothstep(
        0.55 * scene_cut_threshold, scene_cut_threshold, scene_delta
    )
    # A moving silhouette can cross a uniform screen with little RGB evidence at
    # isolated pixels.  Matte disagreement is therefore a second, conservative
    # motion test that prevents a previous edge from trailing behind.
    matte_delta = (current_alpha - neighbour_alpha).abs()
    # Expand disagreement by one pixel so a translating hard contour is rejected
    # even at a nearby pixel whose binary alpha happens to match in both frames.
    # Moderate same-pixel fluctuations remain eligible for chatter reduction.
    structural_delta = _morph_integer(matte_delta, structural_radius, dilate=True)
    matte_gate = 1.0 - _smoothstep(0.04, 0.40, structural_delta)
    return (local_gate * scene_gate * matte_gate).clamp(0.0, 1.0)


def _reduce_chatter_fast(
    rgb: Tensor,
    alpha: Tensor,
    motion_alpha: Tensor,
    repair_map: Tensor,
    amount: float,
    motion_threshold: float,
    scene_cut_threshold: float,
    structural_radius: int,
) -> Tuple[Tensor, Tensor]:
    batch = alpha.shape[0]
    if batch < 2 or amount <= 0.0:
        return alpha, torch.zeros_like(alpha)

    # Adjacent-pair weights are symmetric because every gate is built from
    # absolute differences.  Evaluate each pair once for the whole batch,
    # then scatter it to both participating frames.  This is numerically
    # identical to the former per-frame loop while avoiding duplicate guided
    # motion analysis and many small GPU/CPU kernel launches.
    pair_weight = _temporal_neighbor_weight(
        rgb[1:],
        rgb[:-1],
        motion_alpha[1:],
        motion_alpha[:-1],
        motion_threshold,
        scene_cut_threshold,
        structural_radius,
    )

    accumulated = alpha.clone()
    total_weight = torch.ones_like(alpha)
    accepted_weight = torch.zeros_like(alpha)

    accumulated[1:] = accumulated[1:] + alpha[:-1] * pair_weight
    accumulated[:-1] = accumulated[:-1] + alpha[1:] * pair_weight
    total_weight[1:] = total_weight[1:] + pair_weight
    total_weight[:-1] = total_weight[:-1] + pair_weight
    accepted_weight[1:] = accepted_weight[1:] + pair_weight
    accepted_weight[:-1] = accepted_weight[:-1] + pair_weight

    temporal_target = accumulated / total_weight.clamp_min(
        torch.finfo(alpha.dtype).eps
    )
    blend = (amount * repair_map).clamp(0.0, 1.0)
    neighbour_count = torch.full(
        (batch, 1, 1, 1),
        2.0,
        device=alpha.device,
        dtype=alpha.dtype,
    )
    neighbour_count[0] = 1.0
    neighbour_count[-1] = 1.0
    temporal_gate = blend * (accepted_weight / neighbour_count).clamp(0.0, 1.0)
    return alpha.lerp(temporal_target, blend), temporal_gate


def clean_alpha_v5(
    rgb: Tensor,
    alpha: Tensor,
    *,
    edge_radius: float = 3.0,
    strength: float = 0.75,
    alpha_contrast: float = 0.10,
    detail_recovery: float = 0.55,
    temporal_mode: str = "off",
    reduce_chatter: float = 0.0,
    effect_mask: Optional[Tensor] = None,
    guide_epsilon: float = 1.0e-3,
    motion_threshold: float = 0.08,
    scene_cut_threshold: float = 0.18,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Clean a matte locally around its boundary.

    Args:
        rgb: Straight RGB guidance in ``[0, 1]``, strict ``[B, 3, H, W]``.
        alpha: Input coverage matte in ``[0, 1]``, strict ``[B, 1, H, W]``.
        edge_radius: Spatial repair radius in pixels. Fractional values are valid.
        strength: Blend amount for spatial repair, in ``[0, 1]``.
        alpha_contrast: Edge contrast in ``[-1, 1]``; zero is neutral.
        detail_recovery: Restore source-matte detail only where image and matte
            edges correlate, in ``[0, 1]``.
        temporal_mode: ``"off"`` or ``"fast"``. Fast mode interprets ``B`` as
            ordered adjacent frames and uses a symmetric one-frame window.
        reduce_chatter: Temporal blend amount in ``[0, 1]``.
        effect_mask: Optional ``[B, 1, H, W]`` mask; zero bypasses cleanup.
            A one-frame mask may be broadcast across the batch.
        guide_epsilon: Regularisation for the RGB/luma guided filter.
        motion_threshold: Local RGB difference that fully rejects a neighbour.
        scene_cut_threshold: Frame-average RGB difference that rejects a cut.

    Returns:
        ``(alpha_clean, diagnostics)``.  Diagnostics contains four one-channel
        maps: ``edge_map``, ``repair_map``, ``detail_gate`` and
        ``temporal_gate``.  No RGB-sized debug copies are materialised.
    """

    _validate_inputs(rgb, alpha, effect_mask)
    mode = str(temporal_mode).strip().lower()
    if mode not in ("off", "fast"):
        raise ValueError('temporal_mode must be either "off" or "fast"')

    radius = _clamp_float(edge_radius, 0.0, 64.0, "edge_radius")
    spatial_strength = _clamp_float(strength, 0.0, 1.0, "strength")
    contrast = _clamp_float(alpha_contrast, -1.0, 1.0, "alpha_contrast")
    detail_amount = _clamp_float(detail_recovery, 0.0, 1.0, "detail_recovery")
    chatter_amount = _clamp_float(reduce_chatter, 0.0, 1.0, "reduce_chatter")
    epsilon = _clamp_float(guide_epsilon, 1.0e-6, 0.25, "guide_epsilon")
    local_motion = _clamp_float(motion_threshold, 1.0e-4, 1.0, "motion_threshold")
    scene_cut = _clamp_float(
        scene_cut_threshold, 1.0e-4, 1.0, "scene_cut_threshold"
    )

    output_dtype = alpha.dtype
    work_dtype = (
        torch.float32
        if alpha.dtype in (torch.float16, torch.bfloat16)
        else alpha.dtype
    )
    work_rgb = rgb.to(dtype=work_dtype).clamp(0.0, 1.0)
    work_alpha = alpha.to(dtype=work_dtype).clamp(0.0, 1.0)

    if effect_mask is None:
        mask = torch.ones_like(work_alpha)
    else:
        mask = effect_mask.to(dtype=work_dtype).clamp(0.0, 1.0)
        if mask.shape[0] == 1 and work_alpha.shape[0] > 1:
            mask = mask.expand(work_alpha.shape[0], -1, -1, -1)

    dilated = _morph(work_alpha, radius, dilate=True)
    eroded = _morph(work_alpha, radius, dilate=False)
    morphology_band = (dilated - eroded).clamp(0.0, 1.0)
    soft_transition = (4.0 * work_alpha * (1.0 - work_alpha)).clamp(0.0, 1.0)
    edge_map = torch.maximum(morphology_band, soft_transition)
    if radius > 0.0:
        feathered_band = _gaussian_blur(edge_map, min(2.0, 0.30 + radius * 0.12))
        edge_map = torch.maximum(edge_map, feathered_band * 0.85).clamp(0.0, 1.0)

    repair_map = (edge_map * mask * spatial_strength).clamp(0.0, 1.0)
    guided = _joint_guided_filter(
        work_rgb, work_alpha, radius, epsilon
    ).clamp(0.0, 1.0)
    detail_gate = _detail_correlation(work_rgb, work_alpha, edge_map)
    detail_preserved = guided.lerp(work_alpha, detail_gate * detail_amount)
    spatial_target = _contrast_alpha(detail_preserved.clamp(0.0, 1.0), contrast)
    spatial_clean = work_alpha.lerp(spatial_target, repair_map).clamp(0.0, 1.0)

    if mode == "fast":
        clean, temporal_gate = _reduce_chatter_fast(
            work_rgb,
            spatial_clean,
            work_alpha,
            repair_map,
            chatter_amount,
            local_motion,
            scene_cut,
            # A guided filter has roughly twice its nominal radius of influence.
            # Reject temporal borrowing across that whole footprint when a
            # silhouette moves, not merely at the changed binary pixel.
            max(1, int(math.ceil(2.0 * radius + 1.0))),
        )
    else:
        clean = spatial_clean
        temporal_gate = torch.zeros_like(work_alpha)

    clean = clean.clamp(0.0, 1.0).to(dtype=output_dtype)
    diagnostics = {
        "edge_map": edge_map.to(dtype=output_dtype),
        "repair_map": repair_map.to(dtype=output_dtype),
        "detail_gate": detail_gate.to(dtype=output_dtype),
        "temporal_gate": temporal_gate.to(dtype=output_dtype),
    }
    return clean, diagnostics


# Readable alias for callers that do not need the implementation version in the
# function name.  Keeping a single function object also keeps signatures equal.
clean_alpha = clean_alpha_v5


__all__ = ["clean_alpha_v5", "clean_alpha"]
