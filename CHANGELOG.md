# Changelog

版本号采用 `MAJOR.MINOR.PATCH`。以下 v0.1–v0.5 保留原 README 的历史记录；不为过去的提交补造发布日期。

## v0.6.0 — 2026-09-16

整合 v0.5 之后已落地的功能，并统一 README、代码版本号和 Git 标签。

### Added

- 自动视频抠像：`GiftAutoShotSplit`、`GiftMaskCheckFrames`、`GiftAdaptiveMatting`；保持已验证的 V2 稀疏纠错算法，未合入其他实验策略。
- 执行节点时自动下载缺失的 TransNetV2 / MatAnyone2 权重，支持旧路径复用、SHA-256 校验、临时文件保护和离线开关。
- 内置 MIT 授权的 TransNetV2 推理代码，移除对外部切镜插件安装路径的依赖。
- 自动抠像可移植示例、依赖安装指南和第三方许可说明。
- `GiftIconAutoRestore`：光效恢复、边缘安全羽化、紧边裁切，以及预览 / 1280 RGBA / 168 RGBA 输出；附完整 ICON 工作流。

### Changed

- `FastBottomFitOverlay` 增加可选前景图像与遮罩贴回，Top Fade 同时作用于礼物底层与贴回前景，背景大渐变只作用于礼物底层。
- 增加 Custom / Low Coins / Standard / Naked-Eye 3D 预设、圆角矩形羽化、动态或固定尺寸 Packed 输出，以及中心缩放。
- README 按安装、示例、合成、抠像、节点索引和迁移重新组织；明确按需下载权重与安装插件依赖的区别。
- 入口导出 `__version__ = "0.6.0"`；新增文档链接、节点覆盖和版本一致性检查。

### Compatibility and verification

- 旧合成节点 ID 保持不变；不连接可选前景输入仍可用于普通合成。
- 三个新抠像节点使用正式 ID；不注册重复 Research 别名、不自动卸载旧节点包或覆盖旧工作流。
- MatAnyone2 / RMBG 等代码依赖仍需独立安装；模型授权不随本仓库 MIT 许可改变。
- 抠像合并时通过 104 项单元测试；空模型目录真实下载与关闭下载后的复用通过。
- 本次文档与版本整理后通过 107 项单元测试，包含节点索引、相对链接和版本一致性检查。
- 同参数 124 帧、512×512 新旧链路对照中，导出的 8-bit 遮罩 PNG 逐像素一致。这不是全量素材质量保证或速度基准。

## Earlier versions

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
