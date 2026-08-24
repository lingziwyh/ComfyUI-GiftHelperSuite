# Gift Icon Auto Restore & Export

Node ID: `GiftIconAutoRestore`
Display name: `Gift Icon Auto Restore & Export`
Category: `GiftHelperSuite/Icon`

This node produces the three outputs used by the gift ICON workflow:

1. preview composited over the optional background;
2. square 1280 RGBA ICON with the valid long edge fitted to the canvas;
3. square 168 RGBA ICON independently derived from the tight source crop.

## Core inputs

- `subject_rgba`: the Klein result packed by `GiftChromaMasterPackRGBA`, kept at the
  original canvas resolution;
- `subject_mask`: `GiftChromaMaster.foreground_alpha`, where 1 means foreground;
- `original_black_image`: the original subject + light effects on black, before Klein;
- `preview_background` (optional): used only for the preview output. When it is not
  connected, the bundled 1680 px gift ICON guide background is used automatically.

## Effective-pixel policy

Non-zero Unmult alpha is not automatically valid content. The node follows the old
`#380` AE Unmult + normal Alpha Over behavior while controlling lifted black:

- estimates black level and noise amplitude from a narrow canvas-border sample;
- raises the effective Unmult black point only when the measured border noise requires it;
- converts max-RGB intensity into one continuous AE-style alpha curve, without a
  per-pixel visibility gate that could turn faint glow into a jagged contour;
- limits distant artifacts with a soft subject-shaped region while retaining continuous
  weak glow inside that region;
- composites the recovered effect with straight-alpha normal Alpha Over, matching the
  former AE node chain;
- evaluates the final edge again at 1280 and 168 using the half-step threshold of an
  8-bit PNG alpha channel.

`unmult_black_point = 0.105` matches the old #380 setup. `min_effect_visibility` is the
manual noise margin above the measured black floor. The automatic border-noise estimate
may raise the effective black point for noisy/compressed sources. Lower the black point
only when a genuinely useful dark effect disappears; raise it when faint haze expands
the crop.

## Canvas-edge safety fade

`enable_edge_fade` prevents a close-up arm, face, or object clipped by the source canvas
from becoming a straight hard ICON boundary.

The built-in matte is 1 in the main image area and smoothly reaches 0 only at the outer
canvas edge. Defaults:

- `edge_guard_percent = 0`: no fully rejected border band;
- `edge_feather_percent = 2.5`: only the outer 2.5% is softened (about 32 px at 1280);
- the central 95% is exactly unchanged.

Connect `edge_guard_mask` to override the generated matte. Its semantics are `1=keep`,
`0=fade`; it is automatically resized to the working canvas.

## Crop and resize behavior

- The crop bounds come from the cleaned combined alpha of subject + restored effects.
- `crop_padding = 0` produces a true tight crop.
- `output_padding = 8` leaves an 8 px transparent margin on the 1280 output; the 168
  margin is scaled proportionally (normally 1 px). Set it to 0 for a true edge fit.
- RGB is resized through premultiplied alpha to prevent black/grey color fringing.
- The long side is fitted to `target_size` and the short side is centered with transparent
  pixels.
- The 168 output is generated from the tight source crop, not by shrinking the padded
  1280 canvas, so its own valid long edge remains fitted.

## Preview and performance

- The supplied 1680 x 1680 guide image is bundled in `assets` and cached after its first
  load. Connect `preview_background` only when a different preview background is needed.
- `preview_canvas_size = 1680` generates the standard guide preview.
- `performance_mode = auto` uses ComfyUI's CUDA device when available and otherwise uses
  CPU. Use `cpu` only for troubleshooting or low-VRAM systems.
- Resize antialiasing is enabled only for downscaling; transparent upscales avoid the
  expensive filter without changing the recovered-alpha curve.

## Recommended starting values

```text
fx_strength: 0.75
unmult_black_point: 0.105
fx_reach: 0.35
fx_edge_feather: 32
alpha_cutoff: 0.002
min_effect_visibility: 0.006
enable_edge_fade: true
edge_guard_percent: 0.0
edge_feather_percent: 2.5
crop_padding: 0
output_padding: 8
target_size: 1280
thumbnail_size: 168
preview_scale: 1.04
preview_canvas_size: 1680
performance_mode: auto
```
