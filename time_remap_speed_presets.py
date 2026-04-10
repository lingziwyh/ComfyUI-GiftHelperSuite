import math
import torch


def _clampf(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def _resample(frames: torch.Tensor, src_index: float, mode: str) -> torch.Tensor:
    # frames: [N,H,W,C] float 0..1
    n = int(frames.shape[0])
    if n <= 1:
        return frames[0]

    if mode == "nearest":
        idx = int(round(src_index))
        idx = max(0, min(n - 1, idx))
        return frames[idx]

    # linear
    i0 = int(math.floor(src_index))
    i1 = min(i0 + 1, n - 1)
    i0 = max(0, min(n - 1, i0))
    w = float(src_index - i0)
    if i0 == i1:
        return frames[i0]
    return frames[i0] * (1.0 - w) + frames[i1] * w


def _speed_base(preset: str, u: float) -> float:
    # u in [0,1], return unnormalized positive speed
    # NOTE: final mapping is normalized by integral, so absolute scale doesn't matter.
    if preset == "linear":
        return 1.0

    if preset == "ease_in":          # slow -> fast
        return 0.25 + 1.75 * (u * u)

    if preset == "ease_out":         # fast -> slow
        t = (1.0 - u)
        return 0.25 + 1.75 * (t * t)

    if preset == "ease_in_out":      # slow ends, fast middle
        # sin^2(pi*u) gives 0 at ends, 1 at middle; keep a floor
        s = math.sin(math.pi * u)
        return 0.20 + 1.80 * (s * s)

    if preset == "ramp_up":          # linear up
        return 0.20 + 1.80 * u

    if preset == "ramp_down":        # linear down
        return 0.20 + 1.80 * (1.0 - u)

    if preset == "hold_then_burst":  # hold then speed up near end
        return 0.35 if u < 0.65 else 2.60

    if preset == "burst_then_hold":  # speed up then hold
        return 2.60 if u < 0.35 else 0.35

    if preset == "double_burst":     # two peaks
        s = math.sin(2.0 * math.pi * u)
        return 0.20 + 1.80 * (s * s)

    if preset == "pulse":            # rhythmic pulses (clamped)
        s = 1.0 + 0.85 * math.sin(4.0 * math.pi * u)
        return max(0.15, s)

    if preset == "slow_mid":         # fast ends, slow middle
        c = math.cos(math.pi * u)
        return 0.20 + 1.80 * (c * c)

    if preset == "fast_mid":         # very fast mid spike
        s = math.sin(math.pi * u)
        return 0.15 + 2.60 * (s * s)

    # fallback
    return 1.0


class VideoTimeRemapSpeedPresets:
    """
    AE-like Time Remap driven by SPEED presets (dt_in/dt_out), no curve editor UI.

    Input:  IMAGE batch (frames)
    Output: IMAGE batch (frames)

    Guarantees:
    - First output frame maps to first input frame.
    - Last output frame maps strictly to last input frame.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "target_seconds": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 120.0, "step": 0.1}),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 240.0, "step": 1.0}),
                "speed_preset": ([
                    "linear",
                    "ease_in",
                    "ease_out",
                    "ease_in_out",
                    "ramp_up",
                    "ramp_down",
                    "hold_then_burst",
                    "burst_then_hold",
                    "double_burst",
                    "pulse",
                    "slow_mid",
                    "fast_mid"
                ], {"default": "ease_in_out"}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.5, "step": 0.05}),
                "min_speed": ("FLOAT", {"default": 0.05, "min": 0.001, "max": 2.0, "step": 0.01}),
                "resample_mode": (["linear", "nearest"], {"default": "linear"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "video/time"

    def run(self, images, target_seconds, fps, speed_preset, intensity, min_speed, resample_mode):
        frames = images
        if not isinstance(frames, torch.Tensor):
            raise ValueError("images must be a torch Tensor (IMAGE batch)")

        n_in = int(frames.shape[0])
        if n_in <= 0:
            raise ValueError("Empty input frames")

        n_out = max(1, int(round(float(target_seconds) * float(fps))))

        if n_out == 1 or n_in == 1:
            return (frames[:1],)

        # Sample speed curve at each output frame time u
        s = [0.0] * n_out
        for i in range(n_out):
            u = i / (n_out - 1)
            base = _speed_base(speed_preset, u)
            # intensity blends between flat 1.0 and preset base
            spd = (1.0 - float(intensity)) * 1.0 + float(intensity) * base
            spd = max(float(min_speed), float(spd))
            s[i] = spd

        # Integrate speed to get time-map v(u) with v(0)=0, v(1)=1 (strict alignment)
        cum = [0.0] * n_out
        for i in range(1, n_out):
            cum[i] = cum[i - 1] + 0.5 * (s[i - 1] + s[i])  # trapezoidal integration

        total = cum[-1]
        if total <= 1e-12:
            # fallback to linear mapping
            cum = [i for i in range(n_out)]
            total = cum[-1]

        out_list = []
        for i in range(n_out):
            v = cum[i] / total  # normalized [0..1]
            # strict: v(n_out-1)=1 -> src=n_in-1
            src = v * (n_in - 1)
            src = _clampf(src, 0.0, float(n_in - 1))
            out_list.append(_resample(frames, float(src), resample_mode))

        out = torch.stack(out_list, dim=0)
        return (out,)


NODE_CLASS_MAPPINGS = {
    "VideoTimeRemapSpeedPresets": VideoTimeRemapSpeedPresets
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoTimeRemapSpeedPresets": "Video Time Remap (Speed Presets)"
}
