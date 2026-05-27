import torch
import torch.nn.functional as F


class AEUnmultRGBA:
    """
    AE-style UnMult for ComfyUI.

    Input:  IMAGE tensor in ComfyUI format [B, H, W, C], RGB or RGBA, float 0-1.
    Output: IMAGE tensor [B, H, W, 4], RGBA with black removed.

    Core idea, same as classic AE UnMult workflows:
    - infer alpha from pixel luminance / intensity against black
    - unpremultiply RGB by that alpha
    - keep additive glow / particles as translucent RGBA instead of crushing them
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["max_rgb", "luma_rec709", "luma_rec601", "average_rgb"], {
                    "default": "max_rgb"
                }),
                "black_point": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "display": "slider",
                }),
                "white_point": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.001,
                    "max": 1.0,
                    "step": 0.001,
                    "display": "slider",
                }),
                "alpha_gamma": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.05,
                    "max": 5.0,
                    "step": 0.01,
                    "display": "slider",
                }),
                "alpha_gain": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 4.0,
                    "step": 0.01,
                    "display": "slider",
                }),
                "despill_black": ("BOOLEAN", {"default": False}),
                "preserve_input_alpha": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rgba",)
    FUNCTION = "unmult"
    CATEGORY = "image/alpha"

    def _alpha_from_rgb(self, rgb: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "luma_rec709":
            alpha = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
        elif mode == "luma_rec601":
            alpha = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
        elif mode == "average_rgb":
            alpha = rgb.mean(dim=-1)
        else:
            # The common AE UnMult behavior for glows/fire/particles is closest to max channel.
            # It avoids desaturating colored light edges and preserves the strongest color channel.
            alpha = rgb.max(dim=-1).values
        return alpha.unsqueeze(-1)

    def unmult(
        self,
        image: torch.Tensor,
        mode: str = "max_rgb",
        black_point: float = 0.0,
        white_point: float = 1.0,
        alpha_gamma: float = 1.0,
        alpha_gain: float = 1.0,
        despill_black: bool = False,
        preserve_input_alpha: bool = True,
    ):
        if image is None:
            raise ValueError("AE Unmult RGBA: image input is required.")

        img = image.float().clamp(0.0, 1.0)

        if img.shape[-1] < 3:
            raise ValueError("AE Unmult RGBA: input IMAGE must have at least 3 channels.")

        rgb = img[..., :3]
        input_alpha = img[..., 3:4] if img.shape[-1] >= 4 else None

        # Infer matte from brightness above black.
        raw_alpha = self._alpha_from_rgb(rgb, mode)

        # Level-like alpha remap: black point / white point.
        denom = max(float(white_point) - float(black_point), 1e-6)
        alpha = ((raw_alpha - float(black_point)) / denom).clamp(0.0, 1.0)

        # Gamma and gain allow matching AE layer/extract behavior more closely on real footage.
        alpha = torch.pow(alpha.clamp(0.0, 1.0), float(alpha_gamma))
        alpha = (alpha * float(alpha_gain)).clamp(0.0, 1.0)

        if preserve_input_alpha and input_alpha is not None:
            alpha = alpha * input_alpha.clamp(0.0, 1.0)

        eps = 1e-6
        unpremult_rgb = rgb / torch.clamp(alpha, min=eps)
        unpremult_rgb = unpremult_rgb.clamp(0.0, 1.0)

        # Optional: reduce residual dark contamination left on semi-transparent pixels.
        # Kept off by default because AE-style UnMult should preserve glow softness.
        if despill_black:
            lift = torch.clamp(alpha, min=eps)
            unpremult_rgb = torch.where(alpha > eps, unpremult_rgb / torch.sqrt(lift), unpremult_rgb)
            unpremult_rgb = unpremult_rgb.clamp(0.0, 1.0)

        rgba = torch.cat([unpremult_rgb, alpha], dim=-1).clamp(0.0, 1.0)
        return (rgba,)


class AEAlphaOverRGBA:
    """
    AE/Photoshop-style RGBA layer compositing for two IMAGE tensors.

    Inputs:  background IMAGE + foreground IMAGE, RGB or RGBA, float 0-1.
    Output:  RGBA IMAGE [B, H, W, 4].

    v5 adds foreground_rgb_mode and output_rgb_mode:
    - foreground_rgb_mode controls how the foreground RGB is interpreted before blending.
      premultiply_by_alpha_ps_like is useful for matching Photoshop PNG/normal compositing with UnMult glow layers.
    - output_rgb_mode controls how the final RGBA RGB is encoded.
    """

    BLEND_MODES = [
        "normal",
        "multiply",
        "screen",
        "overlay",
        "soft_light",
        "hard_light",
        "color_dodge",
        "color_burn",
        "linear_dodge_add",
        "linear_burn",
        "darken",
        "lighten",
        "difference",
        "exclusion",
        "subtract",
        "divide",
    ]

    FOREGROUND_RGB_MODES = [
        "premultiply_by_alpha_ps_like",
        "straight_rgb_standard",
    ]

    OUTPUT_RGB_MODES = [
        "premultiplied_rgb_ps_like",
        "straight_alpha_standard",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background": ("IMAGE",),
                "foreground": ("IMAGE",),
                "blend_mode": (cls.BLEND_MODES, {"default": "normal"}),
                "foreground_opacity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                }),
                "background_opacity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                }),
                "resize_foreground_to_background": (["none", "bilinear", "nearest"], {
                    "default": "none"
                }),
                "foreground_rgb_mode": (cls.FOREGROUND_RGB_MODES, {
                    "default": "premultiply_by_alpha_ps_like"
                }),
                "output_rgb_mode": (cls.OUTPUT_RGB_MODES, {
                    "default": "premultiplied_rgb_ps_like"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rgba",)
    FUNCTION = "alpha_over"
    CATEGORY = "image/alpha"

    @staticmethod
    def _to_rgba(image: torch.Tensor, name: str) -> torch.Tensor:
        if image is None:
            raise ValueError(f"AE Alpha Over RGBA: {name} input is required.")

        img = image.float().clamp(0.0, 1.0)

        if img.ndim != 4:
            raise ValueError(
                f"AE Alpha Over RGBA: {name} must be a ComfyUI IMAGE tensor with shape [B, H, W, C]."
            )

        if img.shape[-1] < 3:
            raise ValueError(f"AE Alpha Over RGBA: {name} must have at least 3 channels.")

        rgb = img[..., :3]
        alpha = img[..., 3:4] if img.shape[-1] >= 4 else torch.ones_like(rgb[..., :1])
        return torch.cat([rgb, alpha], dim=-1).clamp(0.0, 1.0)

    @staticmethod
    def _resize_like(image: torch.Tensor, height: int, width: int, mode: str) -> torch.Tensor:
        if mode == "none":
            return image

        x = image.permute(0, 3, 1, 2)
        if mode == "nearest":
            x = F.interpolate(x, size=(height, width), mode="nearest")
        else:
            x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
        return x.permute(0, 2, 3, 1).clamp(0.0, 1.0)

    @staticmethod
    def _match_batch(background: torch.Tensor, foreground: torch.Tensor):
        bg_b = background.shape[0]
        fg_b = foreground.shape[0]

        if bg_b == fg_b:
            return background, foreground
        if bg_b == 1:
            return background.expand(fg_b, -1, -1, -1), foreground
        if fg_b == 1:
            return background, foreground.expand(bg_b, -1, -1, -1)

        raise ValueError(
            "AE Alpha Over RGBA: batch sizes must match, or one input batch must be 1. "
            f"Got background batch {bg_b} and foreground batch {fg_b}."
        )

    @staticmethod
    def _blend_rgb(bg_rgb: torch.Tensor, fg_rgb: torch.Tensor, blend_mode: str) -> torch.Tensor:
        """
        Photoshop-like straight RGB blend functions.

        bg_rgb: backdrop color, 0-1
        fg_rgb: source/layer color, 0-1
        """
        bg = bg_rgb.clamp(0.0, 1.0)
        fg = fg_rgb.clamp(0.0, 1.0)
        eps = 1e-6

        if blend_mode == "multiply":
            out = bg * fg
        elif blend_mode == "screen":
            out = 1.0 - (1.0 - bg) * (1.0 - fg)
        elif blend_mode == "overlay":
            out = torch.where(bg <= 0.5, 2.0 * bg * fg, 1.0 - 2.0 * (1.0 - bg) * (1.0 - fg))
        elif blend_mode == "soft_light":
            # W3C/Photoshop-like soft light approximation.
            d = torch.where(bg <= 0.25, ((16.0 * bg - 12.0) * bg + 4.0) * bg, torch.sqrt(torch.clamp(bg, min=0.0)))
            out = torch.where(
                fg <= 0.5,
                bg - (1.0 - 2.0 * fg) * bg * (1.0 - bg),
                bg + (2.0 * fg - 1.0) * (d - bg),
            )
        elif blend_mode == "hard_light":
            out = torch.where(fg <= 0.5, 2.0 * bg * fg, 1.0 - 2.0 * (1.0 - bg) * (1.0 - fg))
        elif blend_mode == "color_dodge":
            out = torch.where(fg >= 1.0 - eps, torch.ones_like(bg), torch.clamp(bg / torch.clamp(1.0 - fg, min=eps), 0.0, 1.0))
        elif blend_mode == "color_burn":
            out = torch.where(fg <= eps, torch.zeros_like(bg), 1.0 - torch.clamp((1.0 - bg) / torch.clamp(fg, min=eps), 0.0, 1.0))
        elif blend_mode == "linear_dodge_add":
            out = bg + fg
        elif blend_mode == "linear_burn":
            out = bg + fg - 1.0
        elif blend_mode == "darken":
            out = torch.minimum(bg, fg)
        elif blend_mode == "lighten":
            out = torch.maximum(bg, fg)
        elif blend_mode == "difference":
            out = torch.abs(bg - fg)
        elif blend_mode == "exclusion":
            out = bg + fg - 2.0 * bg * fg
        elif blend_mode == "subtract":
            out = bg - fg
        elif blend_mode == "divide":
            out = bg / torch.clamp(fg, min=eps)
        else:
            out = fg

        return out.clamp(0.0, 1.0)

    def alpha_over(
        self,
        background: torch.Tensor,
        foreground: torch.Tensor,
        blend_mode: str = "normal",
        foreground_opacity: float = 1.0,
        background_opacity: float = 1.0,
        resize_foreground_to_background: str = "none",
        foreground_rgb_mode: str = "premultiply_by_alpha_ps_like",
        output_rgb_mode: str = "premultiplied_rgb_ps_like",
    ):
        bg = self._to_rgba(background, "background")
        fg = self._to_rgba(foreground, "foreground")

        bg_h, bg_w = bg.shape[1], bg.shape[2]
        fg_h, fg_w = fg.shape[1], fg.shape[2]

        if (bg_h != fg_h or bg_w != fg_w):
            if resize_foreground_to_background == "none":
                raise ValueError(
                    "AE Alpha Over RGBA: image sizes must match. "
                    f"Got background {bg_w}x{bg_h}, foreground {fg_w}x{fg_h}. "
                    "Set resize_foreground_to_background to bilinear or nearest if you want automatic matching."
                )
            fg = self._resize_like(fg, bg_h, bg_w, resize_foreground_to_background)

        bg, fg = self._match_batch(bg, fg)

        bg_rgb = bg[..., :3]
        bg_a = (bg[..., 3:4] * float(background_opacity)).clamp(0.0, 1.0)
        fg_rgb = fg[..., :3]
        fg_a = (fg[..., 3:4] * float(foreground_opacity)).clamp(0.0, 1.0)

        # Photoshop-like PNG/layer behavior for many UnMult glow layers is closer to using
        # the visually premultiplied foreground color before the normal layer blend.
        # In practice this means the foreground color contribution is effectively weighted by alpha twice
        # while the final alpha still follows normal source-over composition.
        # This avoids low-alpha/high-RGB UnMult haze from washing out the solid subject.
        if foreground_rgb_mode == "premultiply_by_alpha_ps_like":
            fg_rgb_for_blend = (fg_rgb * fg_a).clamp(0.0, 1.0)
        else:
            fg_rgb_for_blend = fg_rgb

        blended_rgb = self._blend_rgb(bg_rgb, fg_rgb_for_blend, blend_mode)

        # Correct handling for semi-transparent backdrops:
        # if the background pixel is fully transparent, non-normal blend modes should not darken/alter the foreground.
        # if it is fully opaque, the foreground receives the full selected blend-mode result.
        effective_fg_rgb = fg_rgb_for_blend * (1.0 - bg_a) + blended_rgb * bg_a

        inv_fg_a = 1.0 - fg_a
        out_a = fg_a + bg_a * inv_fg_a

        # Premultiplied result is the actual visual layer result.
        # With foreground_rgb_mode=premultiply_by_alpha_ps_like, this matches Photoshop-style normal stacking
        # much more closely for UnMult glow layers.
        premult_rgb = effective_fg_rgb * fg_a + bg_rgb * bg_a * inv_fg_a

        eps = 1e-6
        if output_rgb_mode == "straight_alpha_standard":
            # Mathematically standard straight-alpha RGBA.
            # Good when the result will continue through tools that explicitly expect straight alpha.
            out_rgb = torch.where(
                out_a > eps,
                premult_rgb / torch.clamp(out_a, min=eps),
                torch.zeros_like(premult_rgb),
            )
        else:
            # PS/AE-preview-like visual RGB.
            # This prevents low-alpha/high-RGB UnMult pixels from looking overly bright or dirty in ComfyUI previews/exports.
            out_rgb = premult_rgb

        out = torch.cat([out_rgb, out_a], dim=-1).clamp(0.0, 1.0)
        return (out,)


NODE_CLASS_MAPPINGS = {
    "AEUnmultRGBA": AEUnmultRGBA,
    "AEAlphaOverRGBA": AEAlphaOverRGBA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AEUnmultRGBA": "AE Unmult RGBA",
    "AEAlphaOverRGBA": "AE Alpha Over RGBA",
}
