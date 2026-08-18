"""ComfyUI node wrappers for Gift Chroma Master."""

from __future__ import annotations

import torch

from .core.pipeline import (
    cleaner_stage,
    diagnostics_image,
    pack_rgba,
    preview_composite,
    run_trio,
    run_trio_chunked,
    screen_key_stage,
    spill_stage,
)


CATEGORY = "GiftHelperSuite/Chroma"
STATE_TYPE = "GIFT_CHROMA_MASTER_STATE"


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
        # Respect ComfyUI's --cpu / device policy instead of bypassing it merely
        # because the Python build can see a CUDA driver.
        return comfy_device if comfy_device.type == "cuda" else torch.device("cpu")
    except (ImportError, AttributeError, RuntimeError, TypeError):
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")


def _sampling_policy(name: str) -> tuple[str, float]:
    policies = {
        "stable_video": ("per_frame_smoothed", 0.82),
        "lock_first_frame": ("first_frame_locked", 0.80),
        "batch_shared": ("batch_shared", 0.80),
        "per_image": ("per_frame_smoothed", 0.0),
    }
    if name not in policies:
        raise ValueError(f"unknown screen sampling policy: {name!r}")
    return policies[name]


class GiftChromaMaster:
    """Daily all-in-one: screen key -> edge cleaner -> spill suppressor."""

    CATEGORY = CATEGORY
    FUNCTION = "apply"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("foreground_rgb", "foreground_alpha")
    OUTPUT_TOOLTIPS = (
        "Straight / 未预乘的前景 RGB。",
        "前景不透明度：1=前景，0=透明。不要把它当作 Load Image 的透明度 MASK。",
    )
    DESCRIPTION = "独立实现的专业三段式色键：屏幕键控、边缘清理、线性光去溢色。"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "screen_mode": (["auto", "green", "blue", "manual"], {"default": "auto", "tooltip": "自动、约束为绿/蓝，或使用手动颜色。"}),
                "screen_color": ("COLOR", {"default": "#00ff00", "tooltip": "仅在 manual 模式使用。"}),
                "batch_mode": (["ordered_video", "independent_images"], {"default": "ordered_video", "tooltip": "视频批次会启用安全的时域稳定；独立图片不会互相借帧。"}),
                "screen_sampling": (["stable_video", "lock_first_frame", "batch_shared", "per_image"], {"default": "stable_video", "tooltip": "稳定视频会平滑取色，并在明显换镜/换幕色时自动重置。"}),
                "screen_gain": _float(1.10, 0.0, 3.0, 0.01, tooltip="越高抠除越多。"),
                "screen_balance": _float(0.80, 0.0, 1.0, 0.01, advanced=True),
                "preblur_px": _float(0.60, 0.0, 12.0, 0.05, advanced=True),
                "clip_black": _float(0.060, 0.0, 0.95, 0.001),
                "clip_white": _float(0.985, 0.05, 1.0, 0.001),
                "clip_rollback": _float(0.25, 0.0, 1.0, 0.01, advanced=True),
                "shrink_grow_px": _float(-0.20, -12.0, 12.0, 0.05, tooltip="正值扩张前景，负值向内收缩。"),
                "softness_px": _float(0.45, 0.0, 12.0, 0.05),
                "cleaner_strength": _float(0.72, 0.0, 1.0, 0.01),
                "edge_radius": _float(3.0, 0.0, 32.0, 0.1),
                "alpha_contrast": _float(0.04, -1.0, 1.0, 0.01, advanced=True),
                "detail_recovery": _float(0.58, 0.0, 1.0, 0.01),
                "reduce_chatter": _float(0.35, 0.0, 1.0, 0.01, tooltip="仅 ordered_video 生效。"),
                "spill_mode": (["standard", "ultra"], {"default": "standard"}),
                "despill_amount": _float(0.82, 0.0, 1.0, 0.01),
                "spill_range": _float(0.55, 0.0, 1.0, 0.01),
                "desaturate": _float(0.20, 0.0, 1.0, 0.01, advanced=True),
                "luma_restore": _float(1.00, 0.0, 1.0, 0.01, advanced=True),
                "edge_recovery": _float(0.35, 0.0, 1.0, 0.01, advanced=True),
                "source_alpha_mode": (["multiply", "ignore", "add_inside"], {"default": "multiply", "advanced": True}),
                "source_mask_polarity": (["transparency", "alpha"], {"default": "transparency", "advanced": True, "tooltip": "ComfyUI Load Image 的 MASK 通常是 transparency。"}),
            },
            "optional": {
                "source_mask": ("MASK",),
                "inside_mask": ("MASK",),
                "outside_mask": ("MASK",),
                "cleaner_mask": ("MASK",),
                "spill_mask": ("MASK",),
                "performance_mode": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "advanced": True,
                    "tooltip": "auto 会在 ComfyUI 允许时使用 CUDA，并把长视频分块以限制显存。",
                }),
                "gpu_chunk_size": _int(
                    8, 1, 32, 1, advanced=True,
                    tooltip="每次送入显卡的帧数；720p 下 8 约占 2.6GB 临时显存。显存紧张可改 4。",
                ),
            },
        }

    def apply(
        self, image, screen_mode, screen_color, batch_mode, screen_sampling,
        screen_gain, screen_balance, preblur_px, clip_black, clip_white,
        clip_rollback, shrink_grow_px, softness_px, cleaner_strength,
        edge_radius, alpha_contrast, detail_recovery, reduce_chatter,
        spill_mode, despill_amount, spill_range, desaturate, luma_restore,
        edge_recovery, source_alpha_mode, source_mask_polarity,
        source_mask=None, inside_mask=None, outside_mask=None,
        cleaner_mask=None, spill_mask=None, performance_mode="auto",
        gpu_chunk_size=8,
    ):
        sampling_mode, smoothing = _sampling_policy(str(screen_sampling))
        if str(batch_mode) == "independent_images" and str(screen_sampling) == "stable_video":
            sampling_mode, smoothing = "per_frame_smoothed", 0.0
        keyer_options = {
            "screen_mode": str(screen_mode),
            "screen_color": screen_color,
            "sampling_mode": sampling_mode,
            "temporal_smoothing": smoothing,
            "screen_gain": float(screen_gain),
            "screen_balance": float(screen_balance),
            "preblur_px": float(preblur_px),
            "clip_black": float(clip_black),
            "clip_white": float(clip_white),
            "clip_rollback": float(clip_rollback),
            "shrink_grow_px": float(shrink_grow_px),
            "softness_px": float(softness_px),
            "source_alpha_mode": str(source_alpha_mode),
            "source_mask_polarity": str(source_mask_polarity),
        }
        cleaner_options = {
            "edge_radius": float(edge_radius),
            "strength": float(cleaner_strength),
            "alpha_contrast": float(alpha_contrast),
            "detail_recovery": float(detail_recovery),
            "temporal_mode": "fast" if str(batch_mode) == "ordered_video" else "off",
            "reduce_chatter": float(reduce_chatter),
        }
        spill_options = {
            "mode": str(spill_mode),
            "amount": float(despill_amount),
            "range": float(spill_range),
            "desaturate": float(desaturate),
            "luma_restore": float(luma_restore),
            "edge_recovery": float(edge_recovery),
        }
        process_device = _resolve_processing_device(image, str(performance_mode))
        chunk = max(1, min(32, int(gpu_chunk_size)))
        while True:
            try:
                with torch.inference_mode():
                    if process_device.type == "cuda":
                        foreground, alpha = run_trio_chunked(
                            image,
                            keyer=keyer_options,
                            cleaner=cleaner_options,
                            spill=spill_options,
                            processing_device=process_device,
                            chunk_size=chunk,
                            output_device=torch.device("cpu"),
                            source_mask=source_mask,
                            inside_mask=inside_mask,
                            outside_mask=outside_mask,
                            cleaner_mask=cleaner_mask,
                            spill_mask=spill_mask,
                        )
                    else:
                        cpu_image = image.to(device="cpu")
                        cpu_masks = [
                            None if value is None else value.to(device="cpu")
                            for value in (
                                source_mask, inside_mask, outside_mask,
                                cleaner_mask, spill_mask,
                            )
                        ]
                        foreground, alpha = run_trio(
                            cpu_image,
                            keyer=keyer_options,
                            cleaner=cleaner_options,
                            spill=spill_options,
                            source_mask=cpu_masks[0],
                            inside_mask=cpu_masks[1],
                            outside_mask=cpu_masks[2],
                            cleaner_mask=cpu_masks[3],
                            spill_mask=cpu_masks[4],
                        )
                break
            except torch.OutOfMemoryError:
                if process_device.type == "cuda":
                    with torch.cuda.device(process_device):
                        torch.cuda.empty_cache()
                if process_device.type == "cuda" and chunk > 1:
                    chunk = max(1, chunk // 2)
                    continue
                if str(performance_mode).strip().lower() == "auto":
                    process_device = torch.device("cpu")
                    continue
                raise RuntimeError(
                    "Gift Chroma Master CUDA processing ran out of memory even at "
                    "one frame per chunk; "
                    "use performance_mode=cpu or free GPU memory"
                ) from None
        return foreground, alpha


class GiftChromaMasterKeyer:
    CATEGORY = CATEGORY
    FUNCTION = "apply"
    RETURN_TYPES = (STATE_TYPE, "MASK")
    RETURN_NAMES = ("keyer_state", "keyed_matte")
    OUTPUT_TOOLTIPS = (
        "传给 Gift Chroma Master Cleaner/诊断节点的轻量状态。",
        "1=前景的基础不透明度遮罩。",
    )
    DESCRIPTION = "专家模式：生成基础屏幕遮罩并保留原始分析状态。"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "screen_mode": (["auto", "green", "blue", "manual"], {"default": "auto"}),
                "screen_color": ("COLOR", {"default": "#00ff00"}),
                "sampling_mode": (["first_frame_locked", "batch_shared", "per_frame_smoothed"], {"default": "per_frame_smoothed", "tooltip": "per_frame_smoothed 对连续镜头稳定取色并自动检测明显幕色切换。"}),
                "border_fraction": _float(0.08, 0.01, 0.50, 0.005),
                "temporal_smoothing": _float(0.82, 0.0, 1.0, 0.01),
                "screen_gain": _float(1.10, 0.0, 3.0, 0.01),
                "screen_balance": _float(0.80, 0.0, 1.0, 0.01),
                "alpha_bias_color": ("COLOR", {"default": "#808080", "advanced": True}),
                "use_alpha_bias": ("BOOLEAN", {"default": False, "advanced": True}),
                "preblur_px": _float(0.60, 0.0, 12.0, 0.05),
                "clip_black": _float(0.060, 0.0, 0.95, 0.001),
                "clip_white": _float(0.985, 0.05, 1.0, 0.001),
                "clip_rollback": _float(0.25, 0.0, 1.0, 0.01),
                "shrink_grow_px": _float(-0.20, -32.0, 32.0, 0.05),
                "softness_px": _float(0.45, 0.0, 32.0, 0.05),
                "source_alpha_mode": (["multiply", "ignore", "add_inside"], {"default": "multiply"}),
                "source_mask_polarity": (["transparency", "alpha"], {"default": "transparency"}),
            },
            "optional": {
                "source_mask": ("MASK",),
                "inside_mask": ("MASK",),
                "outside_mask": ("MASK",),
            },
        }

    def apply(
        self, image, screen_mode, screen_color, sampling_mode, border_fraction,
        temporal_smoothing, screen_gain, screen_balance, alpha_bias_color,
        use_alpha_bias, preblur_px, clip_black, clip_white, clip_rollback,
        shrink_grow_px, softness_px, source_alpha_mode, source_mask_polarity,
        source_mask=None, inside_mask=None, outside_mask=None,
    ):
        return screen_key_stage(
            image,
            screen_mode=str(screen_mode),
            screen_color=screen_color,
            sampling_mode=str(sampling_mode),
            border_fraction=float(border_fraction),
            temporal_smoothing=float(temporal_smoothing),
            screen_gain=float(screen_gain),
            screen_balance=float(screen_balance),
            alpha_bias_color=alpha_bias_color if bool(use_alpha_bias) else None,
            preblur_px=float(preblur_px),
            clip_black=float(clip_black),
            clip_white=float(clip_white),
            clip_rollback=float(clip_rollback),
            shrink_grow_px=float(shrink_grow_px),
            softness_px=float(softness_px),
            source_alpha_mode=str(source_alpha_mode),
            source_mask_polarity=str(source_mask_polarity),
            source_mask=source_mask,
            inside_mask=inside_mask,
            outside_mask=outside_mask,
        )


class GiftChromaMasterCleaner:
    CATEGORY = CATEGORY
    FUNCTION = "apply"
    RETURN_TYPES = (STATE_TYPE, "MASK")
    RETURN_NAMES = ("cleaner_state", "clean_matte")
    OUTPUT_TOOLTIPS = (
        "传给 Gift Chroma Master Despill/诊断节点的清理状态。",
        "1=前景的清理后不透明度遮罩。",
    )
    DESCRIPTION = "专家模式：只在边缘带恢复细节并抑制视频遮罩抖动。"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keyer_state": (STATE_TYPE,),
                "edge_radius": _float(3.0, 0.0, 64.0, 0.1),
                "strength": _float(0.72, 0.0, 1.0, 0.01),
                "alpha_contrast": _float(0.04, -1.0, 1.0, 0.01),
                "detail_recovery": _float(0.58, 0.0, 1.0, 0.01),
                "temporal_mode": (["off", "fast"], {"default": "fast"}),
                "reduce_chatter": _float(0.35, 0.0, 1.0, 0.01),
                "motion_threshold": _float(0.08, 0.001, 1.0, 0.005, advanced=True),
                "scene_cut_threshold": _float(0.18, 0.001, 1.0, 0.005, advanced=True),
            },
            "optional": {"effect_mask": ("MASK",)},
        }

    def apply(
        self, keyer_state, edge_radius, strength, alpha_contrast,
        detail_recovery, temporal_mode, reduce_chatter, motion_threshold,
        scene_cut_threshold, effect_mask=None,
    ):
        return cleaner_stage(
            keyer_state,
            edge_radius=float(edge_radius),
            strength=float(strength),
            alpha_contrast=float(alpha_contrast),
            detail_recovery=float(detail_recovery),
            temporal_mode=str(temporal_mode),
            reduce_chatter=float(reduce_chatter),
            motion_threshold=float(motion_threshold),
            scene_cut_threshold=float(scene_cut_threshold),
            effect_mask=effect_mask,
        )


class GiftChromaMasterDespill:
    CATEGORY = CATEGORY
    FUNCTION = "apply"
    RETURN_TYPES = ("IMAGE", "MASK", STATE_TYPE)
    RETURN_NAMES = ("foreground_rgb", "foreground_alpha", "final_state")
    OUTPUT_TOOLTIPS = (
        "Straight / 未预乘的去溢色前景 RGB。",
        "前景不透明度：1=前景，0=透明。",
        "仅在需要诊断视图时连接；批量视频优先使用日用一体化节点。",
    )
    DESCRIPTION = "专家模式：线性光、任意键色方向的高级去溢色。"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cleaner_state": (STATE_TYPE,),
                "mode": (["standard", "ultra"], {"default": "standard"}),
                "amount": _float(0.82, 0.0, 1.0, 0.01),
                "range": _float(0.55, 0.0, 1.0, 0.01),
                "desaturate": _float(0.20, 0.0, 1.0, 0.01),
                "spill": _float(1.00, 0.0, 1.0, 0.01),
                "luma_restore": _float(1.00, 0.0, 1.0, 0.01),
                "edge_recovery": _float(0.35, 0.0, 1.0, 0.01),
                "despill_bias": ("COLOR", {"default": "#c08060", "advanced": True}),
                "use_despill_bias": ("BOOLEAN", {"default": False, "advanced": True}),
                "protect_color": ("COLOR", {"default": "#c08060", "advanced": True}),
                "use_protect_color": ("BOOLEAN", {"default": False, "advanced": True}),
                "protect_strength": _float(0.85, 0.0, 1.0, 0.01, advanced=True),
                "skin_protection": _float(0.55, 0.0, 1.0, 0.01, advanced=True),
                "highlight_protection": _float(0.50, 0.0, 1.0, 0.01, advanced=True),
                "key_color_protection": _float(0.80, 0.0, 1.0, 0.01, advanced=True),
            },
            "optional": {"effect_mask": ("MASK",)},
        }

    def apply(
        self, cleaner_state, mode, amount, range, desaturate, spill,
        luma_restore, edge_recovery, despill_bias, use_despill_bias,
        protect_color, use_protect_color, protect_strength, skin_protection,
        highlight_protection, key_color_protection, effect_mask=None,
    ):
        state, image, mask = spill_stage(
            cleaner_state,
            mode=str(mode),
            amount=float(amount),
            range=float(range),
            desaturate=float(desaturate),
            spill=float(spill),
            luma_restore=float(luma_restore),
            edge_recovery=float(edge_recovery),
            despill_bias=despill_bias if bool(use_despill_bias) else None,
            protect_color=protect_color if bool(use_protect_color) else None,
            protect_strength=float(protect_strength),
            skin_protection=float(skin_protection),
            highlight_protection=float(highlight_protection),
            key_color_protection=float(key_color_protection),
            effect_mask=effect_mask,
        )
        return image, mask, state


class GiftChromaMasterDiagnostics:
    CATEGORY = CATEGORY
    FUNCTION = "render"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("diagnostic",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "state": (STATE_TYPE,),
            "view": ([
                "raw_matte", "screen_matte", "screen_mix", "keyed_matte",
                "clean_matte", "edge_map", "repair_map", "temporal_gate",
                "spill_map", "screen_color",
            ], {"default": "clean_matte"}),
        }}

    def render(self, state, view):
        return (diagnostics_image(state, str(view)),)


class GiftChromaMasterPreview:
    CATEGORY = CATEGORY
    FUNCTION = "render"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "background": (["checker", "black", "white", "color"], {"default": "checker"}),
            "background_color": ("COLOR", {"default": "#202020"}),
        }}

    def render(self, image, mask, background, background_color):
        return (preview_composite(image, mask, background=str(background), background_color=background_color),)


class GiftChromaMasterPackRGBA:
    CATEGORY = CATEGORY
    FUNCTION = "pack"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_rgba",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "mask": ("MASK",)}}

    def pack(self, image, mask):
        return (pack_rgba(image, mask),)


__all__ = [
    "GiftChromaMaster",
    "GiftChromaMasterCleaner",
    "GiftChromaMasterDespill",
    "GiftChromaMasterDiagnostics",
    "GiftChromaMasterKeyer",
    "GiftChromaMasterPackRGBA",
    "GiftChromaMasterPreview",
]
