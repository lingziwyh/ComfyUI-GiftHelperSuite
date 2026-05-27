# ComfyUI-GiftHelperSuite

A lightweight ComfyUI utility node suite for **AIGC gift effects, livestream animation compositing, short video frame processing, and RGBA layer workflows**.

这是一个面向 **AI 礼物动效、直播特效、序列帧合成、RGBA 光效处理** 的 ComfyUI 自定义节点工具包。

它的目标不是做“大而全”的通用插件，而是解决礼物生产链路里最常见、最重复、最影响效率的几个问题：

- 序列帧快速叠加
- 前景层自动贴底合成
- 礼物动效快速后处理
- 时间重映射 / 变速处理
- 黑底光效 Unmult 去黑转 RGBA
- RGBA 图层按 Photoshop / After Effects 逻辑合成

---

## Features / 功能概览

| Node | Description | 用途 |
|---|---|---|
| **Fast Bottom Fit Overlay** | Auto scale foreground to background width and bottom-align it | 前景层自动缩放到底图宽度，并贴底合成 |
| **Fast Gift PostFX** | Fast Bloom / Chromatic Aberration / Sharpen / Color adjustment | 礼物动效快速泛光、色散、锐化、调色 |
| **Time Remap Speed Presets** | Retiming frame sequences with common speed presets | 序列帧时间重映射、加速、减速 |
| **AE Unmult RGBA** | Remove black background like AE UnMult and output RGBA | 黑底光效去黑，输出带透明通道的 RGBA |
| **AE Alpha Over RGBA** | Composite two RGBA images with blend modes and opacity controls | 两张 RGBA 图层按正常图层逻辑合成，并继续输出 RGBA |

---

## Installation / 安装方式

Clone this repository into your ComfyUI `custom_nodes` folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lingziwyh/ComfyUI-GiftHelperSuite.git
```

Then restart ComfyUI.

将本仓库克隆到 ComfyUI 的 `custom_nodes` 目录下：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lingziwyh/ComfyUI-GiftHelperSuite.git
```

然后重启 ComfyUI。

---

## Included Nodes / 节点说明

### 1. Fast Bottom Fit Overlay

A fast compositing node designed for gift animation and livestream layout workflows.

It automatically scales the foreground layer to match the background width, keeps the original aspect ratio, and aligns the layer to the bottom of the background.

适用于礼物动画、直播间布局层、前景氛围层的快速合成。

它会自动将前景图宽度缩放到底图宽度，保持原始比例，并将前景层贴到底部。

#### Key Features

- Auto scale foreground to background width
- Keep aspect ratio
- Bottom alignment
- Batch sequence support
- Optional top feather fade
- Useful for character / atmosphere / overlay compositing

#### 核心功能

- 前景层自动适配底图宽度
- 保持原始宽高比
- 自动底部贴合
- 支持序列帧批处理
- 支持顶部柔和过渡遮罩
- 适合人物层、氛围层、礼物动效层叠加

#### Typical Use Cases

- Gift animation compositing
- Livestream effect overlays
- Character + background merging
- Layout animation layers
- Black-background effect preparation

---

### 2. Fast Gift PostFX

A high-performance post-processing node optimized for short video frame sequences, especially 5–10 second AIGC gift animations.

一个专为 5–10 秒短视频序列帧优化的高速后处理节点，适用于 AI 礼物动效与直播特效制作。

#### Effects

- Bloom
- Chromatic Aberration
- Sharpen
- Natural Saturation
- Saturation
- Contrast
- Brightness

#### 支持效果

- 泛光
- 色散
- 锐化
- 自然饱和度
- 饱和度
- 对比度
- 亮度

#### Design Goal

This node is designed for fast batch processing.  
It avoids slow per-frame Python loops where possible and uses tensor-based operations for better performance.

该节点的设计目标是 **高速批处理序列帧**。  
尽量避免低效的逐帧 Python 循环，适合礼物动效生产中的快速视觉增强。

#### Typical Use Cases

- Magic effects
- Fireworks
- Glow enhancement
- Neon lighting
- Anime-style highlights
- Gift animation polishing
- Livestream effect post-processing

---

### 3. Time Remap Speed Presets

A simple frame sequence retiming node for speed changes and timing adjustment.

用于序列帧时间重映射、节奏调整、加速和减速。

#### Typical Use Cases

- Speed up animation
- Slow motion
- Beat matching
- Motion timing adjustment
- AI video frame sequence retiming

#### 常见用途

- 动画加速
- 慢动作
- 卡点节奏调整
- 礼物动效时间重排
- AI 视频序列帧变速

---

### 4. AE Unmult RGBA

A black-background removal node inspired by the UnMult workflow in After Effects.

It converts black-background light effects, particles, glows, and atmosphere layers into RGBA images with transparency.

这是一个参考 After Effects UnMult 工作流设计的去黑节点。

它可以将黑底光效、粒子、烟花、辉光、氛围层转换为带透明通道的 RGBA 图像。

#### Input

| Input | Type | Description |
|---|---|---|
| `image` | IMAGE | Input image or frame sequence |

#### Output

| Output | Type | Description |
|---|---|---|
| `image` | IMAGE | RGBA image with black removed |

#### Best For

- Black background glow effects
- Particle layers
- Fireworks
- Light streaks
- Magic effects
- Atmosphere overlays
- Video layers prepared for alpha compositing

#### 适合处理

- 黑底光效
- 黑底粒子
- 烟花
- 魔法特效
- 氛围光
- 发光线条
- 需要转透明通道的动效层

---

### 5. AE Alpha Over RGBA

A RGBA compositing node designed to work like a normal layer stack in Photoshop / After Effects.

It supports blend modes, foreground opacity, background opacity, and outputs RGBA so the result can continue to be used in transparent compositing workflows.

这是一个用于 RGBA 图层合成的节点，逻辑接近 Photoshop / After Effects 中的图层叠加。

它支持混合模式、前景透明度、背景透明度，并继续输出 RGBA，适合透明通道链路中的多层合成。

#### Inputs

| Input | Type | Description |
|---|---|---|
| `background` | IMAGE | Background RGBA image |
| `foreground` | IMAGE | Foreground RGBA image |
| `blend_mode` | Dropdown | Layer blend mode |
| `foreground_opacity` | Float Slider | Foreground opacity |
| `background_opacity` | Float Slider | Background opacity |
| `resize_foreground_to_background` | Boolean | Resize foreground to background size |

#### Output

| Output | Type | Description |
|---|---|---|
| `image` | IMAGE | Composited RGBA image |

#### Supported Blend Modes

- `normal`
- `multiply`
- `screen`
- `overlay`
- `soft_light`
- `hard_light`
- `color_dodge`
- `color_burn`
- `linear_dodge_add`
- `linear_burn`
- `darken`
- `lighten`
- `difference`
- `exclusion`
- `subtract`
- `divide`

#### Typical Use Cases

- RGBA layer compositing
- Overlaying multiple transparent effects
- Combining Unmult light effects
- Stacking particles, glow, smoke, fireworks
- Preparing transparent foreground animation layers
- Livestream layout effect compositing

#### 推荐工作流

```text
Black-background effect
        ↓
AE Unmult RGBA
        ↓
AE Alpha Over RGBA
        ↓
Final RGBA output
```

例如：

```text
黑底粒子 / 黑底光效
        ↓
AE Unmult RGBA 去黑
        ↓
AE Alpha Over RGBA 多层合成
        ↓
输出透明 RGBA 动效层
```

---

## Recommended Workflow / 推荐使用方式

### Black Background Effect to Transparent Overlay

```text
Input black-background effect
        ↓
AE Unmult RGBA
        ↓
AE Alpha Over RGBA
        ↓
Final transparent effect layer
```

适合将黑底光效转成透明通道，再叠加到其他画面上。

---

### Gift Animation Post Processing

```text
Generated frame sequence
        ↓
Fast Gift PostFX
        ↓
AE Unmult RGBA / AE Alpha Over RGBA
        ↓
Final composited output
```

适合 AI 礼物动效的快速质感增强与透明层合成。

---

### Livestream Layout Overlay

```text
Background / livestream frame
        ↓
Fast Bottom Fit Overlay
        ↓
Fast Gift PostFX
        ↓
Final layout animation
```

适合直播间布局层、前景氛围层、人物层、动效层的快速合成。

---

## Why This Suite Exists / 为什么做这个插件

AIGC gift animation and livestream effect production often requires many small but repetitive operations:

- Resize foreground layers
- Align overlays
- Process short frame sequences
- Add quick glow and sharpening
- Remove black backgrounds from light effects
- Composite RGBA layers repeatedly
- Keep alpha channels through the whole pipeline

Many existing nodes can do part of this, but they are often too general, too slow, or not optimized for gift / livestream animation workflows.

This suite focuses on production efficiency: fewer nodes, clearer logic, faster iteration.

---

AI 礼物动效和直播特效生产中，经常会遇到大量重复但又很影响效率的小操作：

- 前景层缩放
- 图层贴底
- 序列帧批处理
- 快速泛光和锐化
- 黑底光效去黑
- 多层 RGBA 合成
- 透明通道在整个链路中保持稳定

很多已有节点可以完成其中一部分，但往往过于通用、链路较长，或者不够适合礼物动效生产。

这个工具包的目标是让这些高频操作更直接、更稳定、更适合生产。

---

## Notes / 注意事项

- For best results, use image sequences with consistent resolution.
- When working with RGBA images, make sure the alpha channel is preserved by upstream nodes.
- If another custom node package has the same node names, remove duplicates to avoid registration conflicts.
- After installing or updating the plugin, restart ComfyUI.

---

- 建议输入序列帧保持统一分辨率。
- 使用 RGBA 工作流时，请确认上游节点没有丢失 alpha 通道。
- 如果你之前安装过单独版本的 `ComfyUI_Unmult_AE`，建议删除，避免同名节点重复注册。
- 安装或更新插件后，需要重启 ComfyUI。

---

## Update Log / 更新记录

### v0.3

Added AE-style RGBA tools:

- Added `AE Unmult RGBA`
- Added `AE Alpha Over RGBA`
- Added blend mode dropdown
- Added foreground opacity slider
- Added background opacity slider
- Added RGBA normal/source-over compositing workflow

---

### v0.2

Added fast post-processing tools:

- Bloom
- Chromatic Aberration
- Sharpen
- Saturation / Contrast / Brightness controls

---

### v0.1

Initial utility nodes:

- Fast Bottom Fit Overlay
- Time Remap Speed Presets

---

## License / 许可

MIT License.

---

## Author / 作者

Created by [lingziwyh](https://github.com/lingziwyh)

Designed for ComfyUI-based AIGC gift animation and livestream effect production workflows.
