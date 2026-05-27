# AE RGBA Tools for ComfyUI-GiftHelperSuite

This merge adds two AE/Photoshop-style RGBA compositing nodes into `ComfyUI-GiftHelperSuite`:

```text
AE Unmult RGBA
AE Alpha Over RGBA
```

Category:

```text
image/alpha
```

## Install / Merge

Copy these files into your existing `ComfyUI/custom_nodes/ComfyUI-GiftHelperSuite/` folder:

```text
ae_rgba_tools.py
__init__.py
```

Then restart ComfyUI.

If you previously installed the standalone `ComfyUI_Unmult_AE` package, remove or disable that old folder to avoid duplicate node names.

---

## AE Unmult RGBA

Performs AE-style UnMult / black background removal and outputs RGBA `IMAGE`.

### Inputs

- `image`: ComfyUI `IMAGE`, RGB or RGBA.
- `mode`:
  - `max_rgb`: recommended default for glow, fire, particles, light FX.
  - `luma_rec709`: perceptual HD luminance.
  - `luma_rec601`: classic SD luminance.
  - `average_rgb`: simple channel average.
- `black_point`: raises the cutoff for what counts as black.
- `white_point`: lowers the level that becomes fully opaque.
- `alpha_gamma`: adjusts alpha curve.
- `alpha_gain`: multiplies alpha after gamma.
- `despill_black`: optional extra cleanup for dark contamination.
- `preserve_input_alpha`: if the input already has alpha, multiply the generated alpha by the input alpha.

### Output

- `rgba`: ComfyUI `IMAGE`, shape `[B, H, W, 4]`.

Recommended for black-background glow / particle / atmosphere overlays:

```text
mode: max_rgb
black_point: 0.0–0.02
white_point: 1.0
alpha_gamma: 0.8–1.0
alpha_gain: 1.0
despill_black: false
```

---

## AE Alpha Over RGBA

Performs Photoshop-like layer compositing for two RGBA images, then outputs a new RGBA `IMAGE`.

```text
background RGBA + foreground RGBA
→ blend mode + layer opacity + alpha composite
→ output RGBA
```

### Inputs

- `background`: bottom image layer, RGB or RGBA.
- `foreground`: top image layer, RGB or RGBA.
- `blend_mode`: Photoshop-like blend mode dropdown.
- `foreground_opacity`: opacity multiplier for the top layer, slider `0.0–1.0`.
- `background_opacity`: opacity multiplier for the bottom layer, slider `0.0–1.0`.
- `resize_foreground_to_background`:
  - `none`: sizes must match exactly.
  - `bilinear`: resize foreground to background size with smooth interpolation.
  - `nearest`: resize foreground to background size with hard-pixel interpolation.

### Blend modes

```text
normal
multiply
screen
overlay
soft_light
hard_light
color_dodge
color_burn
linear_dodge_add
linear_burn
darken
lighten
difference
exclusion
subtract
divide
```

### Output

- `rgba`: ComfyUI `IMAGE`, shape `[B, H, W, 4]`, composited with alpha preserved.

For `normal`, it behaves like standard source-over compositing:

```text
out_a = fg_a + bg_a * (1 - fg_a)
out_rgb = (fg_rgb * fg_a + bg_rgb * bg_a * (1 - fg_a)) / out_a
```

For non-normal blend modes, the node first calculates a blended foreground color using the selected blend mode, then composites it with proper alpha handling.
