# Gift Chroma Master

Gift Chroma Master is the V5-generation chroma pipeline extracted into `ComfyUI-GiftHelperSuite` under new, collision-safe node IDs. No V3, V4, Args nodes, legacy engine, legacy tests, or legacy web widget are included.

它把常见的专业三段式色键流程整合为：

```text
屏幕键控 → 边缘清理 → 线性光去溢色
```

目标是在 ComfyUI 中获得接近 Keylight + Key Cleaner + Advanced Spill Suppressor 组合的工作方式和观感。它是独立实现，不包含 Adobe 代码，也不承诺与 After Effects 像素级一致。

## 推荐入口

日常生产使用：

```text
Load Image / 连续视频帧批次
        ↓ IMAGE
Gift Chroma Master · 一键专业抠像
        ├─ foreground_rgb ─┐
        └─ foreground_alpha ─→ Gift Chroma Master · 打包 RGBA
```

- `foreground_rgb` 是去溢色后的 straight / 未预乘 RGB。
- `foreground_alpha` 是前景不透明度：`1=前景`、`0=透明`。
- 透明区域的 RGB 不会被强制清零，必须和 Alpha 配合使用。
- 只检查少量帧时，可接 `GiftChromaMasterPreview`；长视频主分支不建议生成整段棋盘预览。

## 节点

| Node ID | 显示名 | 用途 |
|---|---|---|
| `GiftChromaMaster` | 一键专业抠像 | 推荐的生产节点，输出最精简并自动使用 CUDA 分块 |
| `GiftChromaMasterKeyer` | 屏幕键控 | 专家链第一段：取幕色并生成基础 Matte |
| `GiftChromaMasterCleaner` | 边缘清理 | 专家链第二段：恢复边缘细节并抑制视频抖动 |
| `GiftChromaMasterDespill` | 高级去溢色 | 专家链第三段：线性光去溢色、颜色保护与亮度恢复 |
| `GiftChromaMasterDiagnostics` | 诊断视图 | 查看 raw/clean Matte、边缘、修复、时域、spill 等图 |
| `GiftChromaMasterPreview` | 合成预览 | 合成到棋盘、黑、白或自定义颜色背景 |
| `GiftChromaMasterPackRGBA` | 打包 RGBA | 把 RGB 与前景 Alpha 打包为四通道 IMAGE |

主节点与三段专家链的默认参数对齐。专家链会保留较多分析状态，适合短批次调试；完整长视频优先使用一体化主节点。

## 推荐调节顺序

1. 先确认幕色：`auto` 不稳定时改为 `green`、`blue` 或 `manual`。
2. 调基础 Matte：优先使用 `screen_gain`、`clip_black`、`clip_white`。
3. 调轮廓：小幅调整 `shrink_grow_px`、`softness_px`、`edge_radius`。
4. 最后调颜色：使用 `despill_amount`、`spill_range`、`edge_recovery`。

背景仍有残留时可尝试：

```text
screen_gain      1.15 ~ 1.35
clip_black       0.07 ~ 0.12
shrink_grow_px  -0.30 ~ -0.80
```

发丝、烟雾、玻璃或发光粒子被吃掉时可尝试：

```text
clip_black       0.015 ~ 0.040
clip_rollback    0.35  ~ 0.65
detail_recovery  0.70  ~ 0.90
shrink_grow_px   0.00  ~ -0.20
```

轮廓仍偏绿或偏蓝时可尝试：

```text
despill_amount  0.90 ~ 1.00
spill_range     0.65 ~ 0.85
spill_mode      ultra
edge_recovery   0.25 ~ 0.45
```

参数过高会损失发丝、半透明细节、肤色或主体中接近幕色的颜色，建议逐项小幅调整并查看实际合成背景。

## 视频与性能

- 连续视频使用 `batch_mode=ordered_video`、`screen_sampling=stable_video`。
- 多张互不相关的图片使用 `batch_mode=independent_images`。
- 一体化节点默认 `performance_mode=auto`，遵守 ComfyUI 的 `--cpu` 与设备策略。
- CUDA 可用时默认按 `gpu_chunk_size=8` 处理，屏幕色与 Cleaner 相邻帧上下文会跨块保留。
- CUDA 显存不足会重试 `8 → 4 → 2 → 1`；auto 模式最后回退 CPU，显式 cuda 模式则给出明确错误。
- 输出会回到 CPU，避免长视频结果持续占用显存。

开发机参考：Ryzen 9 9950X、RTX 5880 Ada 48GB、PyTorch 2.9.1；不含视频解码、编码和保存。

| 输入 | 运行方式 | 参考耗时 |
|---|---|---:|
| `96 × 1280×720` | 自动 CUDA，8 帧分块 | 约 `1.9 s`，约 `51 fps` |
| 同上 | 临时显存峰值 | 约 `2.62 GiB` |

## Mask 与 Alpha 极性

Gift Chroma Master 的输出 Alpha、inside/outside/effect masks 均采用：

```text
1 = 生效 / 前景 / 不透明
0 = 不生效 / 背景 / 透明
```

- `inside_mask=1` 强制保留前景。
- `outside_mask=1` 强制删除前景。
- `cleaner_mask=0` 或 `spill_mask=0` 会严格旁路对应阶段。
- ComfyUI 内置 `Load Image` 的 MASK 通常相反，为 `1=透明`；连接它时保留默认 `source_mask_polarity=transparency`。
- 不要把 `foreground_alpha` 直接接给期望“透明度 MASK”的节点；优先使用 `GiftChromaMasterPackRGBA`。

## 自动取色边界

以下情况建议手动指定幕色或补充 Mask：

- 主体占满画面边缘，四周几乎没有幕布；
- 边缘存在大面积彩灯、字幕或前景物体；
- 背景低饱和、严重欠曝或接近无彩色；
- 主体本身与幕色相同；
- 严重运动模糊、反射玻璃或复杂半透明体。

## 从独立插件迁移

组件包使用全新 ID，不注册任何 `ProChroma*V5`、`KeylightCoreV3/V4` 或 Args alias，因此可以与 `KeylightChromaKeyHub` 暂时并存。旧 V5 工作流按下表替换 `nodes[].type` 和 `properties["Node name for S&R"]`：

```text
ProChromaTrioV5            → GiftChromaMaster
ProChromaScreenKeyerV5     → GiftChromaMasterKeyer
ProChromaKeyCleanerV5      → GiftChromaMasterCleaner
ProChromaSpillSuppressorV5 → GiftChromaMasterDespill
ProChromaDiagnosticsV5     → GiftChromaMasterDiagnostics
ProChromaPreviewV5         → GiftChromaMasterPreview
ProChromaPackRGBAV5        → GiftChromaMasterPackRGBA
```

确认新工作流保存、重载与执行正常后，如只需要组件包版，再把旧独立目录移出 `custom_nodes` 并重启 ComfyUI。

## 数据约定

- 输入为浮点 `IMAGE`，范围 `[0,1]`，布局 `[B,H,W,C]`。
- 内部使用 BCHW；颜色混合、前景恢复和去溢色使用线性光。
- 当前假设输入为有限范围 sRGB / Rec.709 显示编码。
- Log、ACES、HDR、负值或已线性化素材应先转换到合适的工作空间。
- NaN、Inf 与整数 IMAGE 会明确报错。
