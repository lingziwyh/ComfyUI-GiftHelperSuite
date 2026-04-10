# ComfyUI-GiftHelperSuite

A small high-speed helper suite for gift / livestream asset workflows in ComfyUI.

## Included nodes

- **Fast Bottom Fit Overlay**
  - Resize layer image to match background width
  - Keep aspect ratio
  - Bottom-align automatically
  - Optional top fade to soften top clipping
  - Supports batch image compositing

- **Fast Gift PostFX**
  - Fast batch-friendly post-processing for short image sequences
  - Includes Bloom, Chromatic Aberration, and Sharpen

- **Video Time Remap (Speed Presets)**
  - AE-style time remap based on speed presets
  - Works on IMAGE batches
  - Preserves first and last frame alignment

## Install

Copy the whole `ComfyUI-GiftHelperSuite` folder into:

`ComfyUI/custom_nodes/`

Then restart ComfyUI.

## Notes

This suite is optimized for short gift / effect sequences and fixed-layout compositing workflows.
