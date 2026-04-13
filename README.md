ComfyUI-GiftHelperSuite

Fast utility nodes for ComfyUI gift / livestream animation pipelines.

Designed for high-speed processing of short video frame sequences (5–10s) used in AIGC gift effects, livestream animations, and realtime compositing workflows.

🚀 Features / 功能

This plugin provides several optimized nodes designed specifically for fast batch processing of image sequences.

本插件提供了一组专为 序列帧高速处理 优化的 ComfyUI 节点，适用于 AI 礼物动效、直播特效制作等场景。

4.13 Update：- 支持生产用 packed 输出：
  左侧为最终 mask，右侧为应用同一份 mask 的黑底前景
- packed 输出保持 layer 缩放后的实际画幅比例，不补齐到 background 高度

Included nodes:

Node	Description
Fast Bottom Fit Overlay	Auto scale foreground to background width and bottom-align
Fast Gift PostFX	Ultra-fast Bloom / Chromatic Aberration / Sharpen
Time Remap (Speed Presets)	Fast frame sequence retiming tool
📦 Included Nodes / 节点说明
1️⃣ Fast Bottom Fit Overlay

Automatically composits a foreground layer onto a background image.

自动将前景图叠加到底图，并进行尺寸自适应。

Key features:

Auto scale layer width to background width
Keep aspect ratio
Bottom alignment
Batch optimized
Optional top feather fade to soften cut edges

核心功能：

自动将叠加层宽度缩放至背景宽度
保持纵横比
自动底部对齐
支持批量序列帧处理
内置 顶部柔和渐变遮罩，避免硬切边

Use case:

Gift animation compositing
Livestream effects
Character + background merging
Short video overlays

Code reference:

2️⃣ Fast Gift PostFX

A high-performance post-processing node optimized for short frame sequences.

专为 5–10 秒视频序列帧设计的高速后处理节点。

Included effects:

Bloom
Chromatic Aberration
Sharpen

Optimizations:

Fully batch processed
Low resolution bloom pipeline
GPU tensor operations
Avoids slow per-frame Python loops

优化特点：

全 Batch GPU 处理
低分辨率 Bloom 提升性能
使用 Torch 张量计算
避免逐帧 Python 运算

Typical usage:

Magic effects
Neon glow
Fireworks
Highlight enhancement
Anime-style lighting

Code reference:

3️⃣ Time Remap (Speed Presets)

Fast frame sequence retiming tool.

快速序列帧时间重映射节点。

Typical uses:

Speed up animation
Slow motion
Beat matching
Motion timing adjustment

Useful for:

AI generated video
Gift animations
Motion experiments
⚡ Performance

These nodes are designed to be significantly faster than many existing ComfyUI nodes when processing frame sequences.

相比常规 ComfyUI 节点，这些节点在 100+ 帧序列处理 时速度更快。


📥 Installation

Clone into ComfyUI custom_nodes folder.

将插件克隆至 ComfyUI 的 custom_nodes 目录。

cd ComfyUI/custom_nodes
git clone https://github.com/YOURNAME/ComfyUI-GiftHelperSuite

Restart ComfyUI.

重启 ComfyUI。
