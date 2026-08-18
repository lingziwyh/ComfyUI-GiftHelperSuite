# Merge notes · v0.4 local integration

This working tree integrates three related changes:

1. `Gift Chroma Master`
   - Extracted only the GPU-accelerated V5 production chain.
   - Excludes V3/V4, Args nodes, legacy engines, legacy tests and `web/colorWidget.js`.
   - Uses seven new `GiftChromaMaster*` IDs and an isolated state schema.
2. `Gift Mask & Sequence`
   - Reimplements the four standalone Mask Blend behaviors under collision-safe IDs.
   - Vectorizes the frame paths and fixes one-frame fade, input mutation, short-mask and size-alignment issues.
3. `Fast Gift PostFX`
   - Keeps the existing `FastGiftPostFX` ID and all existing required inputs.
   - Adds optional automatic CUDA processing, frame chunking and OOM fallback.

The old standalone plugins were not modified or deleted. They can remain installed during migration because the suite uses new IDs. The two old `ComfyUI-mask-blend*` folders still conflict with each other and should be disabled after workflow migration.

No Git commit or push is included in this local integration.
