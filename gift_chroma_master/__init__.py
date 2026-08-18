"""Gift Chroma Master node registration."""

from .nodes import (
    GiftChromaMaster,
    GiftChromaMasterCleaner,
    GiftChromaMasterDespill,
    GiftChromaMasterDiagnostics,
    GiftChromaMasterKeyer,
    GiftChromaMasterPackRGBA,
    GiftChromaMasterPreview,
)


NODE_CLASS_MAPPINGS = {
    "GiftChromaMaster": GiftChromaMaster,
    "GiftChromaMasterKeyer": GiftChromaMasterKeyer,
    "GiftChromaMasterCleaner": GiftChromaMasterCleaner,
    "GiftChromaMasterDespill": GiftChromaMasterDespill,
    "GiftChromaMasterDiagnostics": GiftChromaMasterDiagnostics,
    "GiftChromaMasterPreview": GiftChromaMasterPreview,
    "GiftChromaMasterPackRGBA": GiftChromaMasterPackRGBA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GiftChromaMaster": "Gift Chroma Master · 一键专业抠像",
    "GiftChromaMasterKeyer": "Gift Chroma Master · 屏幕键控",
    "GiftChromaMasterCleaner": "Gift Chroma Master · 边缘清理",
    "GiftChromaMasterDespill": "Gift Chroma Master · 高级去溢色",
    "GiftChromaMasterDiagnostics": "Gift Chroma Master · 诊断视图",
    "GiftChromaMasterPreview": "Gift Chroma Master · 合成预览",
    "GiftChromaMasterPackRGBA": "Gift Chroma Master · 打包 RGBA",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
