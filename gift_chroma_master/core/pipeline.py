"""Orchestration for the Gift Chroma Master processing stages."""

from __future__ import annotations

from typing import Any, Optional

import torch

from .cleaner import clean_alpha_v5
from .keyer import (
    apply_screen_temporal_policy,
    estimate_screen_color,
    run_v5_keyer,
)
from .spill import suppress_spill_v5
from .utils import (
    AlphaConstraints,
    CleanerDiagnostics,
    GiftChromaMasterState,
    SpillDiagnostics,
    bchw_to_image,
    copy_state,
    ensure_state,
    image_bhwc_to_bchw,
    linear_composite,
    linear_to_srgb,
    mask_to_bchw,
    parse_color,
    srgb_to_linear,
)


STATE_SCHEMA = "gift_chroma_master_v1"


def _smoothstep(low: float, high: float, value: torch.Tensor) -> torch.Tensor:
    t = ((value - low) / max(high - low, 1.0e-6)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _normalize_optional_mask(
    value: Optional[torch.Tensor],
    reference: torch.Tensor,
    name: str,
) -> Optional[torch.Tensor]:
    batch, _, height, width = reference.shape
    return mask_to_bchw(
        value,
        batch=batch,
        height=height,
        width=width,
        device=reference.device,
        dtype=reference.dtype,
        name=name,
    )


def _prepare_alpha_constraints(
    reference: torch.Tensor,
    *,
    embedded_alpha: Optional[torch.Tensor],
    source_mask: Optional[torch.Tensor],
    source_alpha_mode: str,
    source_mask_polarity: str,
    inside_mask: Optional[torch.Tensor],
    outside_mask: Optional[torch.Tensor],
) -> AlphaConstraints:
    """Normalize source alpha, holdout, and garbage mattes once."""
    source_parts: list[torch.Tensor] = []
    if embedded_alpha is not None:
        source_parts.append(
            embedded_alpha.to(reference.device, reference.dtype).clamp(0.0, 1.0)
        )

    # ComfyUI LoadImage emits an all-zero 64x64 placeholder MASK for ordinary
    # RGB files regardless of image resolution.  Treat only that exact sentinel
    # as "no source alpha"; every real mismatched mask still raises an error.
    if source_mask is not None and tuple(source_mask.shape[-2:]) == (64, 64):
        target_size = tuple(reference.shape[-2:])
        if target_size != (64, 64) and int(torch.count_nonzero(source_mask).item()) == 0:
            source_mask = None
    normalized_source = _normalize_optional_mask(source_mask, reference, "source_mask")
    polarity = str(source_mask_polarity).strip().lower()
    if polarity not in ("alpha", "transparency"):
        raise ValueError("source_mask_polarity must be 'alpha' or 'transparency'")
    if normalized_source is not None:
        if polarity == "transparency":
            normalized_source = 1.0 - normalized_source
        source_parts.append(normalized_source)

    source_alpha = None
    for part in source_parts:
        source_alpha = part if source_alpha is None else source_alpha * part

    mode = str(source_alpha_mode).strip().lower()
    if mode not in ("ignore", "multiply", "add_inside"):
        raise ValueError("source_alpha_mode must be 'ignore', 'multiply', or 'add_inside'")
    inside = _normalize_optional_mask(inside_mask, reference, "inside_mask")
    outside = _normalize_optional_mask(outside_mask, reference, "outside_mask")
    return AlphaConstraints(source_alpha, mode, inside, outside)


def _apply_alpha_constraints(
    alpha: torch.Tensor,
    constraints: Optional[AlphaConstraints],
) -> torch.Tensor:
    """Apply prepared constraints exactly once to a candidate matte."""
    result = alpha.clamp(0.0, 1.0)
    if constraints is None:
        return result
    source_alpha = constraints.source_alpha
    if source_alpha is not None and constraints.source_alpha_mode == "multiply":
        result = result * source_alpha
    elif source_alpha is not None and constraints.source_alpha_mode == "add_inside":
        result = torch.maximum(result, source_alpha)
    if constraints.inside_mask is not None:
        result = torch.maximum(result, constraints.inside_mask)
    if constraints.outside_mask is not None:
        result = result * (1.0 - constraints.outside_mask)
    return result.clamp(0.0, 1.0)


def screen_key_stage(
    image: torch.Tensor,
    *,
    screen_mode: str = "auto",
    screen_color: Any = "#00ff00",
    sampling_mode: str = "first_frame_locked",
    border_fraction: float = 0.08,
    temporal_smoothing: float = 0.80,
    screen_gain: float = 1.0,
    screen_balance: float = 0.50,
    alpha_bias_color: Any | None = None,
    preblur_px: float = 0.0,
    clip_black: float = 0.0,
    clip_white: float = 1.0,
    clip_rollback: float = 0.0,
    shrink_grow_px: float = 0.0,
    softness_px: float = 0.0,
    source_alpha_mode: str = "multiply",
    source_mask_polarity: str = "transparency",
    source_mask: Optional[torch.Tensor] = None,
    inside_mask: Optional[torch.Tensor] = None,
    outside_mask: Optional[torch.Tensor] = None,
) -> tuple[GiftChromaMasterState, torch.Tensor]:
    """Run screen analysis and matte finishing from a ComfyUI BHWC IMAGE."""
    rgb_srgb, embedded_alpha = image_bhwc_to_bchw(image)
    batch = rgb_srgb.shape[0]
    manual_color = parse_color(
        screen_color,
        batch=batch,
        device=rgb_srgb.device,
        dtype=rgb_srgb.dtype,
        name="screen_color",
    )
    bias_color = None
    if alpha_bias_color is not None:
        bias_color = parse_color(
            alpha_bias_color,
            batch=batch,
            device=rgb_srgb.device,
            dtype=rgb_srgb.dtype,
            name="alpha_bias_color",
        )

    key = run_v5_keyer(
        rgb_srgb,
        screen_mode=str(screen_mode),
        manual_screen_color=manual_color,
        temporal_mode=str(sampling_mode),
        border_fraction=float(border_fraction),
        temporal_smoothing=float(temporal_smoothing),
        screen_gain=float(screen_gain),
        screen_balance=float(screen_balance),
        alpha_bias_color=bias_color,
        preblur_px=float(preblur_px),
        clip_black=float(clip_black),
        clip_white=float(clip_white),
        clip_rollback=float(clip_rollback),
        shrink_grow_px=float(shrink_grow_px),
        softness_px=float(softness_px),
        return_diagnostics=False,
    )
    constraints = _prepare_alpha_constraints(
        key["alpha"],
        embedded_alpha=embedded_alpha,
        source_mask=source_mask,
        source_alpha_mode=source_alpha_mode,
        source_mask_polarity=source_mask_polarity,
        inside_mask=inside_mask,
        outside_mask=outside_mask,
    )
    alpha_keyed = _apply_alpha_constraints(key["alpha"], constraints)
    rgb_linear = srgb_to_linear(rgb_srgb)
    screen_linear = srgb_to_linear(key["screen_color"])
    state = GiftChromaMasterState(
        schema=STATE_SCHEMA,
        stage="keyed",
        rgb_source_srgb=rgb_srgb,
        rgb_source_linear=rgb_linear,
        screen_srgb=key["screen_color"],
        screen_linear=screen_linear,
        screen_confidence=key["screen_confidence"],
        screen_mix=key["screen_mix"],
        alpha_raw=key["alpha_raw"],
        alpha_base=key["alpha"],
        alpha_constraints=constraints,
        alpha_keyed=alpha_keyed,
        alpha_clean=alpha_keyed,
        rgb_clean_linear=rgb_linear,
        spill_map=torch.zeros_like(alpha_keyed),
    )
    return state, alpha_keyed[:, 0].contiguous()


def cleaner_stage(
    state: Any,
    *,
    edge_radius: float = 3.0,
    strength: float = 0.75,
    alpha_contrast: float = 0.10,
    detail_recovery: float = 0.55,
    temporal_mode: str = "off",
    reduce_chatter: float = 0.0,
    motion_threshold: float = 0.08,
    scene_cut_threshold: float = 0.18,
    effect_mask: Optional[torch.Tensor] = None,
    retain_diagnostics: bool = True,
) -> tuple[GiftChromaMasterState, torch.Tensor]:
    """Run edge-local spatial/temporal matte cleanup on a master state."""
    current = ensure_state(state)
    if current.get("stage") != "keyed":
        raise ValueError(
            "cleaner expects a keyed Gift Chroma Master state, "
            f"got {current.get('stage')!r}"
        )
    alpha = current.get("alpha_base", current["alpha_keyed"])
    normalized_effect = _normalize_optional_mask(effect_mask, alpha, "cleaner_effect_mask")
    alpha_unconstrained, diagnostics = clean_alpha_v5(
        current["rgb_source_srgb"],
        alpha,
        edge_radius=float(edge_radius),
        strength=float(strength),
        alpha_contrast=float(alpha_contrast),
        detail_recovery=float(detail_recovery),
        temporal_mode=str(temporal_mode),
        reduce_chatter=float(reduce_chatter),
        effect_mask=normalized_effect,
        motion_threshold=float(motion_threshold),
        scene_cut_threshold=float(scene_cut_threshold),
    )
    alpha_clean = _apply_alpha_constraints(
        alpha_unconstrained,
        current.get("alpha_constraints"),
    )
    updated = copy_state(
        current,
        stage="cleaned",
        alpha_clean=alpha_clean,
        cleaner_diagnostics=(
            CleanerDiagnostics(**diagnostics) if retain_diagnostics else None
        ),
    )
    return updated, alpha_clean[:, 0].contiguous()


def _recover_screen_mixture(
    rgb_linear: torch.Tensor,
    screen_linear: torch.Tensor,
    screen_mix: torch.Tensor,
    alpha_raw: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    """Conservatively undo screen mixture using the unmodified raw signals."""
    amount = max(0.0, min(1.0, float(strength)))
    if amount <= 0.0:
        return rgb_linear
    mix = screen_mix.clamp(0.0, 0.92)
    candidate = ((rgb_linear - screen_linear * mix) / (1.0 - mix).clamp_min(0.08)).clamp(0.0, 1.0)
    edge = (4.0 * alpha_raw * (1.0 - alpha_raw)).clamp(0.0, 1.0)
    foreground_gate = _smoothstep(0.015, 0.20, alpha_raw)
    weight = (amount * edge * foreground_gate).clamp(0.0, 1.0)
    return rgb_linear.lerp(candidate, weight)


def spill_stage(
    state: Any,
    *,
    mode: str = "standard",
    amount: float = 0.80,
    range: float = 0.55,
    desaturate: float = 0.20,
    spill: float = 1.0,
    luma_restore: float = 1.0,
    edge_recovery: float = 0.20,
    despill_bias: Any | None = None,
    protect_color: Any | None = None,
    protect_strength: float = 0.85,
    skin_protection: float = 0.55,
    highlight_protection: float = 0.50,
    key_color_protection: float = 0.80,
    effect_mask: Optional[torch.Tensor] = None,
    retain_diagnostics: bool = True,
) -> tuple[GiftChromaMasterState, torch.Tensor, torch.Tensor]:
    """Run arbitrary-hue, linear-light spill suppression on a keyed state."""
    current = ensure_state(state)
    if current.get("stage") not in ("keyed", "cleaned"):
        raise ValueError(
            "spill suppressor expects a keyed or cleaned Gift Chroma Master "
            f"state, got {current.get('stage')!r}"
        )
    alpha = current.get("alpha_clean", current["alpha_keyed"])
    batch = alpha.shape[0]
    normalized_effect = _normalize_optional_mask(effect_mask, alpha, "spill_effect_mask")
    source_linear = current["rgb_source_linear"]
    recovered_linear = _recover_screen_mixture(
        source_linear,
        current["screen_linear"],
        current["screen_mix"],
        current["alpha_raw"],
        float(edge_recovery) * max(0.0, min(1.0, float(amount))),
    )
    # The spill mask and master amount govern the complete colour stage,
    # including edge colour recovery.  A zero mask/amount is therefore a true
    # bypass instead of silently changing RGB before despill is evaluated.
    rgb_linear = recovered_linear
    if normalized_effect is not None:
        rgb_linear = source_linear.lerp(recovered_linear, normalized_effect)
    bias_linear = None
    if despill_bias is not None:
        bias_linear = srgb_to_linear(parse_color(
            despill_bias,
            batch=batch,
            device=rgb_linear.device,
            dtype=rgb_linear.dtype,
            name="despill_bias",
        ))
    protect_linear = None
    if protect_color is not None:
        protect_linear = srgb_to_linear(parse_color(
            protect_color,
            batch=batch,
            device=rgb_linear.device,
            dtype=rgb_linear.dtype,
            name="protect_color",
        ))
    rgb_clean, spill_map, diagnostics = suppress_spill_v5(
        rgb_linear,
        alpha,
        current["screen_linear"],
        mode=str(mode),
        amount=float(amount),
        range=float(range),
        desaturate=float(desaturate),
        spill=float(spill),
        luma_restore=float(luma_restore),
        despill_bias=bias_linear,
        protect_color=protect_linear,
        protect_strength=float(protect_strength),
        skin_protection=float(skin_protection),
        highlight_protection=float(highlight_protection),
        key_color_protection=float(key_color_protection),
        effect_mask=normalized_effect,
    )
    updated = copy_state(
        current,
        stage="spilled",
        rgb_clean_linear=rgb_clean,
        spill_map=spill_map if retain_diagnostics else None,
        spill_diagnostics=(
            SpillDiagnostics(**diagnostics) if retain_diagnostics else None
        ),
    )
    image = bchw_to_image(linear_to_srgb(rgb_clean).clamp(0.0, 1.0))
    return updated, image, alpha[:, 0].contiguous()


def run_trio(
    image: torch.Tensor,
    *,
    keyer: dict[str, Any],
    cleaner: dict[str, Any],
    spill: dict[str, Any],
    source_mask: Optional[torch.Tensor] = None,
    inside_mask: Optional[torch.Tensor] = None,
    outside_mask: Optional[torch.Tensor] = None,
    cleaner_mask: Optional[torch.Tensor] = None,
    spill_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the complete master pipeline and return straight RGB plus MASK."""
    keyed, _ = screen_key_stage(
        image,
        source_mask=source_mask,
        inside_mask=inside_mask,
        outside_mask=outside_mask,
        **keyer,
    )
    cleaned, _ = cleaner_stage(
        keyed,
        effect_mask=cleaner_mask,
        retain_diagnostics=False,
        **cleaner,
    )
    _, foreground, alpha = spill_stage(
        cleaned,
        effect_mask=spill_mask,
        retain_diagnostics=False,
        **spill,
    )
    return foreground, alpha


def _slice_video_tensor(
    value: Optional[torch.Tensor],
    *,
    start: int,
    end: int,
    batch: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Slice a per-frame optional tensor while preserving broadcast masks."""
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise ValueError("optional masks must be torch.Tensor values")
    sliced = value
    # Comfy MASK may be HW, BHW, BCHW, or BHWC.  Only layouts with an explicit
    # full-batch leading dimension should be sliced; one-frame masks remain
    # broadcast views and LoadImage's 64x64 sentinel remains intact.
    if value.ndim >= 3 and int(value.shape[0]) == int(batch):
        sliced = value[start:end]
    return sliced.to(device=device, non_blocking=device.type == "cuda")


def _estimate_batch_screen_colors(
    image: torch.Tensor,
    keyer: dict[str, Any],
    *,
    processing_device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    """Estimate the clip's screen colours once before GPU chunking.

    Raw per-frame candidates are sampled in bounded accelerator chunks, then
    the tiny color/confidence tensors are filtered once across the full clip.
    This preserves causal stable-video, first-frame-lock, and batch-shared
    semantics without making a full BCHW copy of the video on the CPU.
    """
    batch = int(image.shape[0])
    manual_color = parse_color(
        keyer.get("screen_color", "#00ff00"),
        batch=batch,
        device=image.device,
        dtype=torch.float32,
        name="screen_color",
    )
    screen_mode = str(keyer.get("screen_mode", "auto"))
    temporal_mode = str(keyer.get("sampling_mode", "first_frame_locked"))
    smoothing = float(keyer.get("temporal_smoothing", 0.80))

    if screen_mode == "manual":
        confidence = torch.ones(
            (batch, 1, 1, 1), device=image.device, dtype=torch.float32
        )
        colors, _ = apply_screen_temporal_policy(
            manual_color,
            confidence,
            temporal_mode=temporal_mode,
            temporal_smoothing=smoothing,
        )
        return colors

    sample_count = 1 if temporal_mode == "first_frame_locked" else batch
    color_parts: list[torch.Tensor] = []
    confidence_parts: list[torch.Tensor] = []
    for start in range(0, sample_count, int(chunk_size)):
        end = min(sample_count, start + int(chunk_size))
        image_part = image[start:end].to(
            device=processing_device,
            non_blocking=processing_device.type == "cuda",
        )
        rgb_part, _ = image_bhwc_to_bchw(image_part)
        colors, confidence = estimate_screen_color(
            rgb_part,
            screen_mode=screen_mode,
            manual_screen_color=manual_color[start:end].to(
                device=processing_device,
                non_blocking=processing_device.type == "cuda",
            ),
            temporal_mode="per_frame_smoothed",
            border_fraction=float(keyer.get("border_fraction", 0.08)),
            temporal_smoothing=0.0,
            max_cluster_samples=int(keyer.get("max_cluster_samples", 384)),
        )
        color_parts.append(colors.to(device=image.device))
        confidence_parts.append(confidence.to(device=image.device))
        del image_part, rgb_part, colors, confidence

    raw_colors = torch.cat(color_parts, dim=0)
    raw_confidence = torch.cat(confidence_parts, dim=0)
    if temporal_mode == "first_frame_locked":
        return raw_colors[0:1].expand(batch, -1, -1, -1).contiguous()
    filtered, _ = apply_screen_temporal_policy(
        raw_colors,
        raw_confidence,
        temporal_mode=temporal_mode,
        temporal_smoothing=smoothing,
    )
    return filtered


def run_trio_chunked(
    image: torch.Tensor,
    *,
    keyer: dict[str, Any],
    cleaner: dict[str, Any],
    spill: dict[str, Any],
    processing_device: torch.device | str,
    chunk_size: int = 8,
    output_device: torch.device | str | None = None,
    source_mask: Optional[torch.Tensor] = None,
    inside_mask: Optional[torch.Tensor] = None,
    outside_mask: Optional[torch.Tensor] = None,
    cleaner_mask: Optional[torch.Tensor] = None,
    spill_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the daily trio in bounded chunks on an accelerator.

    Screen colours are estimated once for the complete clip so stable-video,
    first-frame-lock, and batch-shared policies do not reset at chunk edges.
    The cleaner receives one context frame on either side, making its symmetric
    adjacent-frame chatter reduction equivalent at internal chunk boundaries.
    Outputs return to the input device by default, so a CPU video loader does
    not leave a long clip occupying VRAM after the node finishes.
    """
    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        raise ValueError("image must be a ComfyUI IMAGE tensor with shape [B,H,W,C]")
    batch, height, width = (int(image.shape[0]), int(image.shape[1]), int(image.shape[2]))
    if batch < 1 or height < 1 or width < 1:
        raise ValueError("image must have non-empty batch and spatial dimensions")
    size = int(chunk_size)
    if size < 1:
        raise ValueError("chunk_size must be at least 1")
    process_device = torch.device(processing_device)
    destination = image.device if output_device is None else torch.device(output_device)

    # CPU compatibility remains the original single-call path.  Chunking is an
    # accelerator memory-control mechanism, not a new approximation mode.
    if process_device.type == "cpu" and image.device.type == "cpu":
        return run_trio(
            image,
            keyer=keyer,
            cleaner=cleaner,
            spill=spill,
            source_mask=source_mask,
            inside_mask=inside_mask,
            outside_mask=outside_mask,
            cleaner_mask=cleaner_mask,
            spill_mask=spill_mask,
        )

    screen_colors = _estimate_batch_screen_colors(
        image,
        keyer,
        processing_device=process_device,
        chunk_size=size,
    )
    foreground_out = torch.empty(
        (batch, height, width, 3),
        device=destination,
        dtype=torch.float32,
    )
    alpha_out = torch.empty(
        (batch, height, width),
        device=destination,
        dtype=torch.float32,
    )

    temporal_cleaner = (
        str(cleaner.get("temporal_mode", "off")) == "fast"
        and float(cleaner.get("reduce_chatter", 0.0)) > 0.0
        and batch > 1
    )
    overlap = 1 if temporal_cleaner else 0

    for start in range(0, batch, size):
        end = min(batch, start + size)
        context_start = max(0, start - overlap)
        context_end = min(batch, end + overlap)
        image_chunk = image[context_start:context_end].to(
            device=process_device,
            non_blocking=process_device.type == "cuda",
        )
        chunk_keyer = dict(keyer)
        # The already-filtered per-frame colours must pass through unchanged.
        chunk_keyer.update({
            "screen_mode": "manual",
            "screen_color": screen_colors[context_start:context_end].to(
                device=process_device,
                non_blocking=process_device.type == "cuda",
            ),
            "sampling_mode": "per_frame_smoothed",
            "temporal_smoothing": 0.0,
        })
        foreground, alpha = run_trio(
            image_chunk,
            keyer=chunk_keyer,
            cleaner=cleaner,
            spill=spill,
            source_mask=_slice_video_tensor(
                source_mask,
                start=context_start,
                end=context_end,
                batch=batch,
                device=process_device,
            ),
            inside_mask=_slice_video_tensor(
                inside_mask,
                start=context_start,
                end=context_end,
                batch=batch,
                device=process_device,
            ),
            outside_mask=_slice_video_tensor(
                outside_mask,
                start=context_start,
                end=context_end,
                batch=batch,
                device=process_device,
            ),
            cleaner_mask=_slice_video_tensor(
                cleaner_mask,
                start=context_start,
                end=context_end,
                batch=batch,
                device=process_device,
            ),
            spill_mask=_slice_video_tensor(
                spill_mask,
                start=context_start,
                end=context_end,
                batch=batch,
                device=process_device,
            ),
        )
        local_start = start - context_start
        local_end = local_start + (end - start)
        foreground_out[start:end].copy_(
            foreground[local_start:local_end], non_blocking=False
        )
        alpha_out[start:end].copy_(alpha[local_start:local_end], non_blocking=False)
        del image_chunk, foreground, alpha

    return foreground_out.contiguous(), alpha_out.contiguous()


def diagnostics_image(state: Any, view: str) -> torch.Tensor:
    """Materialize one requested debug view; no fixed debug outputs are cached."""
    current = ensure_state(state)
    name = str(view).strip().lower()
    alpha_clean = current.get("alpha_clean", current["alpha_keyed"])
    maps: dict[str, torch.Tensor] = {
        "raw_matte": current["alpha_raw"],
        "screen_matte": 1.0 - current["alpha_raw"],
        "screen_mix": current["screen_mix"],
        "keyed_matte": current["alpha_keyed"],
        "clean_matte": alpha_clean,
        "spill_map": current.get("spill_map", torch.zeros_like(alpha_clean)),
    }
    cleaner = current.get("cleaner_diagnostics", {})
    maps.update({
        "edge_map": cleaner.get("edge_map", torch.zeros_like(alpha_clean)),
        "repair_map": cleaner.get("repair_map", torch.zeros_like(alpha_clean)),
        "temporal_gate": cleaner.get("temporal_gate", torch.zeros_like(alpha_clean)),
    })
    if name == "screen_color":
        _, _, height, width = alpha_clean.shape
        return bchw_to_image(current["screen_srgb"].expand(-1, 3, height, width))
    if name not in maps:
        raise ValueError(f"unknown diagnostics view: {view!r}")
    return bchw_to_image(maps[name].clamp(0.0, 1.0).expand(-1, 3, -1, -1))


def preview_composite(
    image: torch.Tensor,
    mask: torch.Tensor,
    *,
    background: str = "checker",
    background_color: Any = "#202020",
) -> torch.Tensor:
    """Composite straight RGB and alpha in linear light for inspection."""
    rgb, _ = image_bhwc_to_bchw(image)
    alpha = _normalize_optional_mask(mask, rgb[:, :1], "mask")
    assert alpha is not None
    batch, _, height, width = rgb.shape
    mode = str(background).strip().lower()
    if mode == "black":
        bg = torch.zeros_like(rgb)
    elif mode == "white":
        bg = torch.ones_like(rgb)
    elif mode == "color":
        bg = parse_color(
            background_color,
            batch=batch,
            device=rgb.device,
            dtype=rgb.dtype,
            name="background_color",
        ).expand(batch, 3, height, width)
    elif mode == "checker":
        yy = torch.arange(height, device=rgb.device).view(1, 1, height, 1)
        xx = torch.arange(width, device=rgb.device).view(1, 1, 1, width)
        cell = max(8, min(height, width) // 32)
        pattern = (((xx // cell) + (yy // cell)) % 2).to(rgb.dtype)
        bg = (0.36 + 0.24 * pattern).expand(batch, 3, height, width)
    else:
        raise ValueError("background must be checker, black, white, or color")
    return bchw_to_image(linear_composite(rgb, alpha, bg))


def pack_rgba(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pack straight RGB plus alpha as an explicit, opt-in RGBA IMAGE."""
    rgb, _ = image_bhwc_to_bchw(image)
    alpha = _normalize_optional_mask(mask, rgb[:, :1], "mask")
    assert alpha is not None
    return bchw_to_image(torch.cat((rgb, alpha), dim=1).clamp(0.0, 1.0))


__all__ = [
    "STATE_SCHEMA",
    "cleaner_stage",
    "diagnostics_image",
    "pack_rgba",
    "preview_composite",
    "run_trio",
    "run_trio_chunked",
    "screen_key_stage",
    "spill_stage",
]
