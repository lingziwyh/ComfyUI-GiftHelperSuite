from .fast_bottom_fit_overlay import (
    NODE_CLASS_MAPPINGS as OVERLAY_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as OVERLAY_NODE_DISPLAY_NAME_MAPPINGS,
)
from .fast_gift_postfx import (
    NODE_CLASS_MAPPINGS as POSTFX_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as POSTFX_NODE_DISPLAY_NAME_MAPPINGS,
)
from .time_remap_speed_presets import (
    NODE_CLASS_MAPPINGS as TIMEREMAP_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as TIMEREMAP_NODE_DISPLAY_NAME_MAPPINGS,
)
from .ae_rgba_tools import (
    NODE_CLASS_MAPPINGS as AE_RGBA_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as AE_RGBA_NODE_DISPLAY_NAME_MAPPINGS,
)
from .gift_mask_blend import (
    NODE_CLASS_MAPPINGS as MASK_BLEND_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as MASK_BLEND_NODE_DISPLAY_NAME_MAPPINGS,
)
from .gift_chroma_master import (
    NODE_CLASS_MAPPINGS as CHROMA_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as CHROMA_NODE_DISPLAY_NAME_MAPPINGS,
)
from .example_assets import install_example_assets


def _merge_mappings(label, *groups):
    merged = {}
    for group in groups:
        duplicates = set(merged).intersection(group)
        if duplicates:
            names = ", ".join(sorted(duplicates))
            raise RuntimeError(f"duplicate {label} registration(s): {names}")
        merged.update(group)
    return merged


NODE_CLASS_MAPPINGS = _merge_mappings(
    "node",
    OVERLAY_NODE_CLASS_MAPPINGS,
    POSTFX_NODE_CLASS_MAPPINGS,
    TIMEREMAP_NODE_CLASS_MAPPINGS,
    AE_RGBA_NODE_CLASS_MAPPINGS,
    MASK_BLEND_NODE_CLASS_MAPPINGS,
    CHROMA_NODE_CLASS_MAPPINGS,
)

NODE_DISPLAY_NAME_MAPPINGS = _merge_mappings(
    "display-name",
    OVERLAY_NODE_DISPLAY_NAME_MAPPINGS,
    POSTFX_NODE_DISPLAY_NAME_MAPPINGS,
    TIMEREMAP_NODE_DISPLAY_NAME_MAPPINGS,
    AE_RGBA_NODE_DISPLAY_NAME_MAPPINGS,
    MASK_BLEND_NODE_DISPLAY_NAME_MAPPINGS,
    CHROMA_NODE_DISPLAY_NAME_MAPPINGS,
)

if set(NODE_CLASS_MAPPINGS) != set(NODE_DISPLAY_NAME_MAPPINGS):
    missing_names = set(NODE_CLASS_MAPPINGS).difference(NODE_DISPLAY_NAME_MAPPINGS)
    unknown_names = set(NODE_DISPLAY_NAME_MAPPINGS).difference(NODE_CLASS_MAPPINGS)
    raise RuntimeError(
        "node/display mapping mismatch: "
        f"missing={sorted(missing_names)}, unknown={sorted(unknown_names)}"
    )

try:
    _EXAMPLE_ASSET_STATUS = install_example_assets()
except Exception:
    # Example media is optional and must never prevent node registration.
    import logging as _logging

    _logging.getLogger(__name__).exception("GiftHelperSuite example asset installation failed")
    _EXAMPLE_ASSET_STATUS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
