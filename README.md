# ComfyUI-GiftHelperSuite

A lightweight ComfyUI utility node suite for **AIGC gift effects, livestream animation compositing, short video frame processing, and RGBA layer workflows**.

这是一个面向 **AI 礼物动效、直播特效、序列帧合成、RGBA 光效处理** 的 ComfyUI 自定义节点工具包。

它的目标不是做“大而全”的通用插件，而是解决礼物生产链路里最常见、最重复、最影响效率的几个问题：

- 序列帧快速叠加
- 前景层自动贴底合成
- 礼物动效快速后处理
- 绿幕 / 蓝幕专业色键、边缘清理与去溢色
- 遮罩渐变、首尾淡入淡出、帧切片与序列融合
- 时间重映射 / 变速处理
- 黑底光效 Unmult 去黑转 RGBA
- RGBA 图层按 Photoshop / After Effects 逻辑合成

---

## Features / 功能概览

| Node | Description | 用途 |
|---|---|---|
| **Fast Bottom Fit Overlay** | Auto scale foreground to background width and bottom-align it | 前景层自动缩放到底图宽度，并贴底合成 |
| **Fast Gift PostFX** | Fast Bloom / Chromatic Aberration / Sharpen / Color adjustment | 礼物动效快速泛光、色散、锐化、调色 |
| **Gift Chroma Master** | GPU chroma key + edge cleaner + advanced despill | 自动 GPU 分块的专业色键、边缘清理和去溢色 |
| **Gift Mask & Sequence** | Temporal mask ramps, fade, frame slice and sequence blend | 遮罩渐变、首尾淡入淡出、帧切片和序列融合 |
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

No extra pip dependencies are required. The suite uses the PyTorch bundled with ComfyUI.

无需额外安装 pip 依赖，节点直接使用 ComfyUI 自带的 PyTorch。

---

## Example Workflows / 示例工作流

- [Gift PostFX Example](example_workflows/Gift_PostFX_Example.json)
- [Gift Chroma Master Example](example_workflows/Gift_Chroma_Master_Example.json)

Both workflows include their real demo videos and background image under `example_workflows/assets`. On the first ComfyUI startup after installation, GiftHelperSuite atomically copies the three content-hashed demo files into the root of `ComfyUI/input`, so the workflows can be dragged into ComfyUI without manually reconnecting local files. Existing files are never overwritten; a different same-name file is kept and reported as a conflict. Set `GIFT_HELPER_SKIP_EXAMPLE_ASSETS=1` before starting ComfyUI to opt out.

The two video examples also require [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) for video loading and encoding. No models are required.

两个示例工作流已内置真实视频和直播背景图。安装或更新组件并重启 ComfyUI 后，组件会把三个带 `GiftHelperSuite_` 前缀和内容哈希的示例素材原子复制到 `ComfyUI/input` 根目录，因此可直接拖入工作流，无需重新选择本机文件。组件不会覆盖任何同名文件；如需禁用自动安装示例素材，可在启动前设置 `GIFT_HELPER_SKIP_EXAMPLE_ASSETS=1`。视频加载与导出需要安装 `ComfyUI-VideoHelperSuite`，不需要任何模型。

Uninstalling the plugin does not delete copied input files. They can be removed manually if no longer needed: `GiftHelperSuite_PostFX_Source_bebfe94a.mp4`, `GiftHelperSuite_Chroma_Source_92532474.mp4`, and `GiftHelperSuite_Example_Background_213ee241.png`. If installation reports a conflict or the opt-out flag is enabled, copy these files from `example_workflows/assets` to `ComfyUI/input` manually before running the templates.

卸载组件时不会自动删除已复制到 `ComfyUI/input` 的三份素材；不再需要时可按上面的文件名手动清理。如果日志提示同名冲突，或启用了跳过开关，请先从 `example_workflows/assets` 手动复制素材，再运行示例工作流。

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
- Optional rounded-rectangle feather mask that follows the layer aspect ratio
- Adjustable corner radius from a rectangle to a fully inscribed ellipse
- Top feather fade and rounded-rectangle feather mask are mutually exclusive
- Packed size modes: dynamic content height or a fixed 1440x1280 canvas with black top padding
- Useful for character / atmosphere / overlay compositing

#### 核心功能

- 前景层自动适配底图宽度
- 保持原始宽高比
- 自动底部贴合
- 支持序列帧批处理
- 支持顶部柔和过渡遮罩
- 支持跟随图层宽高比动态变化的圆角矩形柔化遮罩
- 圆角半径可从直角矩形调节到四边贴边的内切椭圆
- 顶部羽化与圆角矩形羽化互斥，只能启用其中一种
- Packed 输出支持动态内容高度，或固定 1440x1280 画布并在顶部补黑
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

This node is designed for fast batch processing. It automatically follows ComfyUI's device policy, uses CUDA when available, processes long clips in small frame chunks, and returns the result to the input device. Existing workflows keep the same `FastGiftPostFX` node ID and required inputs.

该节点的设计目标是 **高速批处理序列帧**。新版会遵守 ComfyUI 的设备设置，在可用时自动使用 CUDA，并把长序列按帧分块处理；旧工作流的节点 ID 和原有必填参数保持不变。

Advanced performance options / 高级性能选项：

- `performance_mode=auto`：推荐；CUDA 可用时自动加速，显存不足时按 `4 → 2 → 1` 帧降低分块，最后安全回退 CPU。
- `performance_mode=cuda`：强制 CUDA；单帧仍显存不足时明确报错。
- `performance_mode=cpu`：强制 CPU；仍会分块以降低内存峰值。
- `gpu_chunk_size=4`：720p 的默认平衡值；显存较小时可改为 `2` 或 `1`。

Reference on the development machine (RTX 5880 Ada, 32 × 720p, excluding decode/save): old CPU path about `0.85 s`; new CPU path about `0.42 s`; new auto CUDA path about `0.09 s`. Actual speed depends on resolution, enabled effects and hardware.

开发机参考（RTX 5880 Ada，32 帧 720p，不含解码和保存）：旧 CPU 路径约 `0.85 秒`，新版 CPU 路径约 `0.42 秒`，新版自动 CUDA 约 `0.09 秒`。实际速度会随分辨率、效果参数和硬件变化。

#### Typical Use Cases

- Magic effects
- Fireworks
- Glow enhancement
- Neon lighting
- Anime-style highlights
- Gift animation polishing
- Livestream effect post-processing

---

### 3. Gift Chroma Master

`GiftChromaMaster` combines a professional three-stage chroma workflow in one production node:

```text
Screen key → edge cleaner → linear-light despill
```

`GiftChromaMaster` 将常用的专业三段式色键流程整合成一个生产节点：

```text
屏幕键控 → 边缘清理 → 线性光去溢色
```

It is an independent implementation inspired by the working method of Keylight + Key Cleaner + Advanced Spill Suppressor. It contains no Adobe code and does not claim pixel-identical After Effects output.

这是独立算法实现，目标是复现类似组合拳的工作方式与观感，不包含 Adobe 代码，也不承诺与 After Effects 像素级一致。

| Node ID | 用途 |
|---|---|
| `GiftChromaMaster` | 推荐的一体化生产节点；自动 CUDA 分块，输出 straight RGB + 前景 Alpha |
| `GiftChromaMasterKeyer` | 专家链：幕色估计与基础 Matte |
| `GiftChromaMasterCleaner` | 专家链：边缘清理、细节恢复、视频降抖 |
| `GiftChromaMasterDespill` | 专家链：线性光高级去溢色与亮度恢复 |
| `GiftChromaMasterDiagnostics` | 查看 Matte、边缘、修复、时域与溢色诊断图 |
| `GiftChromaMasterPreview` | 棋盘、黑、白或自定义背景合成预览 |
| `GiftChromaMasterPackRGBA` | 把 straight RGB 和前景 Alpha 打包为 RGBA IMAGE |

For normal production, connect the complete ordered video batch to `GiftChromaMaster`. The node uses `performance_mode=auto` and `gpu_chunk_size=8` by default, keeps temporal context across chunks, retries `8 → 4 → 2 → 1` on CUDA OOM, and finally falls back to CPU in auto mode. Expert nodes are intended for short diagnostic batches and follow the tensor's existing device.

日常生产优先把完整连续视频批次接入 `GiftChromaMaster`。主节点默认自动使用 CUDA 和 8 帧分块，跨块保留视频上下文；显存不足会自动尝试 `8 → 4 → 2 → 1`，最后在 auto 模式回退 CPU。专家链适合短批次调试，其设备跟随输入张量。

`foreground_alpha` uses foreground-opacity semantics: `1=foreground`, `0=transparent`. It is not the inverted transparency MASK expected by some built-in ComfyUI nodes; use `GiftChromaMasterPackRGBA` when in doubt. See [README_GIFT_CHROMA_MASTER.md](README_GIFT_CHROMA_MASTER.md) for tuning guidance.

### 4. Gift Mask & Sequence

The former standalone Mask Blend functionality is reimplemented as four vectorized, collision-safe nodes:

| Node ID | 用途 |
|---|---|
| `GiftMaskRamp` | 开始帧前为 0，区间内 0→1，结束帧后保持 1 |
| `GiftMaskFadeInOut` | 首尾对称淡入淡出；可叠加输入 Mask，不会修改上游数据 |
| `GiftFrameSlice` | 含结束帧的序列切片，自动处理越界和倒序 |
| `GiftMaskBlend` | 两段序列按 Mask 融合；短 Mask 重复末帧，第二段自动对齐尺寸 |

The new IDs deliberately do not register the generic legacy names `MaskGradientNode`, `FrameSliceNode`, `MaskTransparentInOutNode`, or `SequenceOverlayNode`, so the suite can coexist during migration. Existing JSON workflows can be migrated with this mapping:

```text
MaskGradientNode          → GiftMaskRamp
MaskTransparentInOutNode → GiftMaskFadeInOut
FrameSliceNode            → GiftFrameSlice
SequenceOverlayNode       → GiftMaskBlend
```

### 5. Time Remap Speed Presets

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

### 6. AE Unmult RGBA

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

### 7. AE Alpha Over RGBA

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

### Chroma Key Video / 视频色键

```text
Load Image / video frame batch
        ↓ IMAGE
Gift Chroma Master
        ├─ foreground_rgb ───→ downstream composite / encoder
        └─ foreground_alpha ─→ Gift Chroma Master Pack RGBA
```

正式生产时不要把完整长视频都接到棋盘格 Preview；预览节点适合抽帧检查，主分支直接交给合成或编码，可节省额外时间和内存。

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
- The standalone `KeylightChromaKeyHub` can coexist temporarily because Gift Chroma Master uses new IDs. Remove it after migrating workflows if you only want the suite version.
- The two old `ComfyUI-mask-blend*` folders register duplicate generic IDs with each other. Disable or remove them after migrating to the new `Gift*` IDs.
- After installing or updating the plugin, restart ComfyUI.

---

- 建议输入序列帧保持统一分辨率。
- 使用 RGBA 工作流时，请确认上游节点没有丢失 alpha 通道。
- 如果你之前安装过单独版本的 `ComfyUI_Unmult_AE`，建议删除，避免同名节点重复注册。
- `KeylightChromaKeyHub` 因为节点 ID 不同，可以在迁移期间与本套件共存；工作流迁移完后如果只保留组件包版，可再移出旧目录。
- 两个旧 `ComfyUI-mask-blend*` 目录彼此会注册重复的通用 ID；迁移到新 `Gift*` 节点后建议禁用或移除。
- 安装或更新插件后，需要重启 ComfyUI。

---

## Update Log / 更新记录

### v0.5

- Added portable Gift PostFX and Gift Chroma Master video example workflows.
- Bundled two demo videos and one livestream background image with verified hashes.
- Added conflict-safe, idempotent example asset installation for drag-and-run workflows.
- Migrated the Chroma example from legacy `ProChroma*V5` IDs to current `GiftChromaMaster*` IDs and removed local temp paths.

### v0.4

- Added `Gift Chroma Master` with one production node and six expert/helper nodes.
- Includes only the GPU-accelerated chroma implementation; V3/V4/Args and legacy web code are not included.
- Added four vectorized Gift Mask & Sequence nodes and fixed one-frame fade, upstream-mask mutation, short-mask and size-alignment issues.
- Refactored `Fast Gift PostFX` for automatic CUDA frame chunking, OOM fallback, lower memory use and a faster CPU path while preserving its original node ID and required inputs.
- Added duplicate-registration checks and a full regression test suite.

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

---

## Gift Icon Auto Restore & Export

`GiftIconAutoRestore` turns the Klein + Gift Chroma Master preparation chain into three
production outputs: a background preview, a tight 1280 RGBA ICON, and a tight 168 RGBA
ICON. It restores visible glow from the original black-background image, rejects
lifted-black/compression noise, applies a narrow canvas-edge safety fade, and performs
premultiplied-alpha resizing with transparent short-side padding.

The built-in edge guard only affects the outer 2.5% of the short canvas side by default
(about 32 px at 1280); the central 95% is exactly unchanged. A custom `edge_guard_mask`
can override the generated guard when needed.

See [README_GIFT_ICON_AUTO_RESTORE.md](README_GIFT_ICON_AUTO_RESTORE.md) for the complete
input, output, effective-pixel, and tuning reference.

A complete Klein -> Gift Chroma Master -> auto-restore production template is included
as [Gift_Icon_Auto_Restore_Production.json](example_workflows/Gift_Icon_Auto_Restore_Production.json).
Replace its placeholder `GiftHelperSuite_Icon_Source.png` with the source image in ComfyUI.
