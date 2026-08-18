"""Linear-light, arbitrary-screen-colour spill suppression for Gift Chroma Master.

The functions in this module deliberately operate on an internal BCHW contract.
They do not perform layout guessing or transfer-function conversion.  Callers must
provide straight (un-premultiplied) linear RGB in the [0, 1] range.

This is an independent opponent-colour implementation.  It is not intended to be
a bit-for-bit reproduction of any proprietary keyer or spill suppressor.
"""

import torch


_LUMA = (0.2126, 0.7152, 0.0722)


def linear_luminance(rgb: torch.Tensor) -> torch.Tensor:
    """Return Rec.709 luminance for linear RGB, preserving BCHW dimensions."""
    weights = rgb.new_tensor(_LUMA).view(1, 3, 1, 1)
    return (rgb * weights).sum(dim=1, keepdim=True)


def _smoothstep(edge0, edge1, value: torch.Tensor) -> torch.Tensor:
    width = max(float(edge1) - float(edge0), 1.0e-6)
    t = ((value - float(edge0)) / width).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _zero_luma_chroma(rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    y = linear_luminance(rgb)
    return rgb - y, y


def _normalise_chroma(chroma: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    length = torch.linalg.vector_norm(chroma, dim=1, keepdim=True)
    return chroma / length.clamp_min(eps), length


def _bounded_luma_match(
    rgb: torch.Tensor,
    target_luma: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Match luminance without creating values outside [0, 1].

    Darkening scales toward black.  Brightening moves toward white.  Because
    linear luminance is affine and the luma weights sum to one, both branches hit
    the requested target exactly apart from floating-point error.
    """
    x = rgb.clamp(0.0, 1.0)
    current = linear_luminance(x)
    target = target_luma.clamp(0.0, 1.0)

    dark_fraction = ((current - target) / current.clamp_min(eps)).clamp(0.0, 1.0)
    darkened = x * (1.0 - dark_fraction)

    bright_fraction = ((target - current) / (1.0 - current).clamp_min(eps)).clamp(0.0, 1.0)
    brightened = x + (1.0 - x) * bright_fraction
    return torch.where(target >= current, brightened, darkened).clamp(0.0, 1.0)


def _validate_unit_value(name: str, value) -> float:
    result = float(value)
    if not (0.0 <= result <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value!r}.")
    return result


def _validate_inputs(
    rgb_linear: torch.Tensor,
    alpha: torch.Tensor,
    screen_color_linear: torch.Tensor,
    effect_mask: torch.Tensor | None,
) -> None:
    if not isinstance(rgb_linear, torch.Tensor) or rgb_linear.ndim != 4 or rgb_linear.shape[1] != 3:
        raise ValueError("rgb_linear must be a BCHW tensor with shape [B, 3, H, W].")
    if not rgb_linear.is_floating_point():
        raise TypeError("rgb_linear must use a floating-point dtype.")

    expected_alpha = (rgb_linear.shape[0], 1, rgb_linear.shape[2], rgb_linear.shape[3])
    if not isinstance(alpha, torch.Tensor) or tuple(alpha.shape) != expected_alpha:
        raise ValueError(f"alpha must have shape {expected_alpha}, got {getattr(alpha, 'shape', None)}.")
    if not alpha.is_floating_point():
        raise TypeError("alpha must use a floating-point dtype.")

    expected_screen = (rgb_linear.shape[0], 3, 1, 1)
    if not isinstance(screen_color_linear, torch.Tensor) or tuple(screen_color_linear.shape) != expected_screen:
        raise ValueError(
            f"screen_color_linear must have shape {expected_screen}, "
            f"got {getattr(screen_color_linear, 'shape', None)}."
        )
    if not screen_color_linear.is_floating_point():
        raise TypeError("screen_color_linear must use a floating-point dtype.")

    for name, tensor in (("alpha", alpha), ("screen_color_linear", screen_color_linear)):
        if tensor.device != rgb_linear.device:
            raise ValueError(f"{name} must be on the same device as rgb_linear.")
        if tensor.dtype != rgb_linear.dtype:
            raise ValueError(f"{name} must use the same dtype as rgb_linear.")

    if effect_mask is not None:
        if not isinstance(effect_mask, torch.Tensor) or tuple(effect_mask.shape) != expected_alpha:
            raise ValueError(
                f"effect_mask must have shape {expected_alpha}, got {getattr(effect_mask, 'shape', None)}."
            )
        if not effect_mask.is_floating_point():
            raise TypeError("effect_mask must use a floating-point dtype.")
        if effect_mask.device != rgb_linear.device or effect_mask.dtype != rgb_linear.dtype:
            raise ValueError("effect_mask must use the same device and dtype as rgb_linear.")


def _validate_optional_colour(
    name: str,
    colour: torch.Tensor | None,
    rgb_linear: torch.Tensor,
) -> None:
    if colour is None:
        return
    expected = (rgb_linear.shape[0], 3, 1, 1)
    if not isinstance(colour, torch.Tensor) or tuple(colour.shape) != expected:
        raise ValueError(f"{name} must have shape {expected}, got {getattr(colour, 'shape', None)}.")
    if not colour.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype.")
    if colour.device != rgb_linear.device or colour.dtype != rgb_linear.dtype:
        raise ValueError(f"{name} must use the same device and dtype as rgb_linear.")


def _colour_direction(colour: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    chroma, _ = _zero_luma_chroma(colour)
    direction, length = _normalise_chroma(chroma, eps)
    # A neutral colour has no meaningful hue.  The continuous validity factor
    # makes it a safe no-op instead of inventing a primary-channel direction.
    validity = (length / (length + 1.0e-4)).clamp(0.0, 1.0)
    return direction, validity


def _skin_hue_match(chroma_direction: torch.Tensor, chroma_presence: torch.Tensor) -> torch.Tensor:
    """Conservative warm-hue protection used mainly for red/magenta screens."""
    # Two common skin reflectance directions expressed directly in linear RGB.
    # Multiple anchors avoid treating a single skin colour as universal.
    anchors = chroma_direction.new_tensor(
        (
            (0.5356, 0.1470, 0.0452),
            (0.3515, 0.0782, 0.0252),
        )
    ).view(2, 3, 1, 1)
    anchors = anchors - linear_luminance(anchors)
    anchors = anchors / torch.linalg.vector_norm(anchors, dim=1, keepdim=True).clamp_min(1.0e-6)
    match = (chroma_direction.unsqueeze(1) * anchors.unsqueeze(0)).sum(dim=2).amax(dim=1, keepdim=True)
    return _smoothstep(0.72, 0.95, match) * chroma_presence


def suppress_spill_v5(
    rgb_linear: torch.Tensor,
    alpha: torch.Tensor,
    screen_color_linear: torch.Tensor,
    *,
    mode: str = "standard",
    amount: float = 1.0,
    range: float = 0.50,
    desaturate: float = 0.20,
    spill: float = 1.0,
    luma_restore: float = 1.0,
    edge_influence: float = 1.0,
    translucent_influence: float = 0.85,
    opaque_influence: float = 0.18,
    despill_bias: torch.Tensor | None = None,
    protect_color: torch.Tensor | None = None,
    protect_strength: float = 0.85,
    skin_protection: float = 0.55,
    highlight_protection: float = 0.50,
    key_color_protection: float = 0.80,
    effect_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Suppress screen-colour contamination in straight linear RGB.

    Args:
        rgb_linear: Strict [B, 3, H, W] straight linear RGB in [0, 1].
        alpha: Strict [B, 1, H, W] foreground coverage in [0, 1].
        screen_color_linear: Strict [B, 3, 1, 1] linear screen colour.
        mode: ``"standard"`` for conservative vector removal, or ``"ultra"``
            for a wider, smoother hue selection and stronger local desaturation.
        amount: Master blend.  It is applied exactly once, at the final blend.
        range: Width of selected screen-like hues; 0 is narrow, 1 is wide.
        desaturate: Neutralises residual chroma in proportion to removed spill.
        spill: Fraction of the positive screen-direction chroma to remove.
        luma_restore: 0 keeps the candidate's post-clamp luminance; 1 restores
            original linear-light luminance exactly (within floating precision).
        edge_influence: Strength in the matte transition band.
        translucent_influence: Strength inside low-coverage hair/glass/smoke.
        opaque_influence: Small allowance for spill just inside solid foreground.
        despill_bias: Optional [B, 3, 1, 1] replacement hue in linear RGB.
            It only steers colour recovered from removed screen chroma.
        protect_color: Optional [B, 3, 1, 1] foreground hue to preserve.
        protect_strength: Strength of the explicit protect-colour match.
        skin_protection: Automatic warm skin-hue protection.
        highlight_protection: Protection for bright foreground highlights.
        key_color_protection: Protects highly opaque, saturated intentional
            foreground that happens to share the screen hue.
        effect_mask: Optional [B, 1, H, W] multiplier in [0, 1].

    Returns:
        ``(rgb_clean, spill_map, diagnostics)``.  ``spill_map`` is the actual
        final blend weight, additionally scaled by ``spill`` so a zero-spill
        candidate reports no effect.  Diagnostics contain only per-frame scalar
        tensors, avoiding hidden full-resolution debug allocations.
    """
    _validate_inputs(rgb_linear, alpha, screen_color_linear, effect_mask)
    _validate_optional_colour("despill_bias", despill_bias, rgb_linear)
    _validate_optional_colour("protect_color", protect_color, rgb_linear)

    mode_name = str(mode).strip().lower().replace("-", "_")
    if mode_name in ("ultra_like", "advanced"):
        mode_name = "ultra"
    if mode_name not in ("standard", "ultra"):
        raise ValueError(f"mode must be 'standard' or 'ultra', got {mode!r}.")

    amount_value = _validate_unit_value("amount", amount)
    range_value = _validate_unit_value("range", range)
    desaturate_value = _validate_unit_value("desaturate", desaturate)
    spill_value = _validate_unit_value("spill", spill)
    luma_restore_value = _validate_unit_value("luma_restore", luma_restore)
    edge_value = _validate_unit_value("edge_influence", edge_influence)
    translucent_value = _validate_unit_value("translucent_influence", translucent_influence)
    opaque_value = _validate_unit_value("opaque_influence", opaque_influence)
    protect_value = _validate_unit_value("protect_strength", protect_strength)
    skin_value = _validate_unit_value("skin_protection", skin_protection)
    highlight_value = _validate_unit_value("highlight_protection", highlight_protection)
    key_protect_value = _validate_unit_value("key_color_protection", key_color_protection)

    batch = rgb_linear.shape[0]
    scalar_shape = (batch, 1, 1, 1)
    if amount_value == 0.0 or spill_value == 0.0:
        zero_map = torch.zeros_like(alpha)
        zero_scalar = rgb_linear.new_zeros(scalar_shape)
        diagnostics = {
            "detected_mean": zero_scalar,
            "applied_mean": zero_scalar,
            "protected_mean": zero_scalar,
            "luma_error_mean": zero_scalar,
            "luma_error_max": zero_scalar,
        }
        # Return the exact source tensor: amount=0 is a bit-exact identity path.
        return rgb_linear, zero_map, diagnostics

    eps = max(float(torch.finfo(rgb_linear.dtype).eps) * 8.0, 1.0e-7)
    x = rgb_linear
    a = alpha.clamp(0.0, 1.0)
    screen_direction, screen_validity = _colour_direction(screen_color_linear, eps)

    chroma, source_luma = _zero_luma_chroma(x)
    chroma_direction, chroma_length = _normalise_chroma(chroma, eps)
    projection = (chroma * screen_direction).sum(dim=1, keepdim=True)
    positive_projection = projection.clamp_min(0.0)
    hue_cosine = projection / chroma_length.clamp_min(eps)

    # range=0 selects only hues tightly clustered around the screen.  range=1
    # approaches the complete positive screen-opponent hemisphere.
    if mode_name == "standard":
        hue_floor = 0.94 - 0.84 * range_value
        feather = 0.08 + 0.10 * range_value
        chroma_floor = 0.0075
        desaturation_scale = 0.55
    else:
        hue_floor = 0.90 - 1.02 * range_value
        feather = 0.12 + 0.16 * range_value
        chroma_floor = 0.0045
        desaturation_scale = 1.0

    hue_gate = _smoothstep(hue_floor, min(1.0, hue_floor + feather), hue_cosine)
    chroma_presence = chroma_length / (chroma_length + chroma_floor)
    detected = (hue_gate * chroma_presence * screen_validity).clamp(0.0, 1.0)

    # Matte routing: edge and translucent-interior terms are deliberately
    # independent.  The latter keeps hair, smoke and glass from receiving only a
    # one-pixel contour treatment.  Fully transparent background is still a no-op.
    edge_band = (4.0 * a * (1.0 - a)).clamp(0.0, 1.0)
    foreground_presence = _smoothstep(0.005, 0.08, a)
    translucent_body = foreground_presence * torch.sqrt((1.0 - a).clamp_min(0.0))
    matte_gate = torch.maximum(edge_band * edge_value, translucent_body * translucent_value)
    matte_gate = torch.maximum(matte_gate, a * opaque_value).clamp(0.0, 1.0)

    # Protection is a probabilistic union: several moderate protections combine
    # smoothly without any one heuristic producing a hard, visible boundary.
    saturation = chroma_length / (source_luma.abs() + chroma_length + eps)
    opaque_gate = _smoothstep(0.72, 0.97, a)
    strong_chroma = _smoothstep(0.08, 0.34, saturation)

    skin_match = _skin_hue_match(chroma_direction, chroma_presence)
    skin_term = (skin_match * _smoothstep(0.38, 0.90, a) * skin_value).clamp(0.0, 0.98)

    highlight_match = _smoothstep(0.52, 0.92, source_luma) * _smoothstep(0.30, 0.95, a)
    highlight_term = (highlight_match * highlight_value).clamp(0.0, 0.98)

    intentional_key = hue_gate * strong_chroma * opaque_gate
    key_term = (intentional_key * key_protect_value).clamp(0.0, 0.98)

    protection_remaining = (1.0 - skin_term) * (1.0 - highlight_term) * (1.0 - key_term)

    if protect_color is not None and protect_value > 0.0:
        protect_direction, protect_validity = _colour_direction(protect_color, eps)
        protect_cosine = (chroma_direction * protect_direction).sum(dim=1, keepdim=True)
        explicit_match = _smoothstep(0.72, 0.96, protect_cosine)
        explicit_match = explicit_match * chroma_presence * protect_validity
        explicit_term = (
            explicit_match * _smoothstep(0.22, 0.82, a) * protect_value
        ).clamp(0.0, 0.98)
        protection_remaining = protection_remaining * (1.0 - explicit_term)

    protection = (1.0 - protection_remaining).clamp(0.0, 0.98)

    region_weight = detected * matte_gate * protection_remaining
    if effect_mask is not None:
        region_weight = region_weight * effect_mask.clamp(0.0, 1.0)

    # Build exactly one full-strength candidate.  Master amount is intentionally
    # absent from every operation below and appears only in the final blend.
    removed_projection = positive_projection * spill_value
    candidate = x - removed_projection * screen_direction

    if despill_bias is not None:
        bias_direction, bias_validity = _colour_direction(despill_bias, eps)
        # A bias steers at most half of the removed chroma; desaturate=1 requests
        # a neutral result and therefore disables hue replacement.
        bias_gain = 0.50 * (1.0 - desaturate_value)
        candidate = candidate + removed_projection * bias_direction * bias_validity * bias_gain

    removal_ratio = (removed_projection / chroma_length.clamp_min(eps)).clamp(0.0, 1.0)
    local_desaturation = (desaturate_value * desaturation_scale * removal_ratio).clamp(0.0, 1.0)
    neutral = source_luma.expand_as(candidate)
    candidate = candidate + (neutral - candidate) * local_desaturation
    candidate = candidate.clamp(0.0, 1.0)

    if luma_restore_value > 0.0:
        candidate_luma = linear_luminance(candidate)
        target_luma = candidate_luma + (source_luma - candidate_luma) * luma_restore_value
        candidate = _bounded_luma_match(candidate, target_luma, eps)

    # This is the only application of amount.  The candidate is never partially
    # applied earlier, preventing the squared-strength behaviour of the old path.
    final_weight = (region_weight * amount_value).clamp(0.0, 1.0)
    rgb_clean = x + (candidate - x) * final_weight
    rgb_clean = rgb_clean.clamp(0.0, 1.0)

    # Report the actual routed effect.  Including spill makes spill=0 intuitive,
    # while amount remains represented exactly once in both image and map.
    spill_map = (final_weight * spill_value).clamp(0.0, 1.0)
    luma_error = (linear_luminance(rgb_clean) - source_luma).abs()
    diagnostics = {
        "detected_mean": detected.mean(dim=(2, 3), keepdim=True),
        "applied_mean": spill_map.mean(dim=(2, 3), keepdim=True),
        "protected_mean": protection.mean(dim=(2, 3), keepdim=True),
        "luma_error_mean": luma_error.mean(dim=(2, 3), keepdim=True),
        "luma_error_max": luma_error.amax(dim=(2, 3), keepdim=True),
    }
    return rgb_clean, spill_map, diagnostics


# Descriptive alias for callers that prefer the full effect name.
advanced_spill_suppressor_v5 = suppress_spill_v5


__all__ = [
    "advanced_spill_suppressor_v5",
    "linear_luminance",
    "suppress_spill_v5",
]
