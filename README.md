# ComfyUI-GiftHelperSuite

**v0.6.0 · 2026-09-16** · [更新记录](CHANGELOG.md) · [示例工作流](example_workflows/README.md) · [第三方许可](THIRD_PARTY_NOTICES.md)

面向 **AI 礼物动效、直播间合成、自动视频抠像和 RGBA ICON** 的 ComfyUI 工具包。

Tools for AI gift effects, livestream compositing, adaptive video matting and RGBA ICON preparation.

原有合成、色键和序列工具直接使用 ComfyUI 的 PyTorch；视频抠像通过外部 MatAnyone2 / RMBG 节点完成模型推理。**自动下载权重不等于自动安装插件或 Python 依赖。**

## 功能与文档

| 模块 | 能做什么 | 入口 |
| --- | --- | --- |
| 裸眼 3D / 贴底合成 | 自动适配宽度、前景贴回、背景渐变、顶部羽化、圆角和 Packed 输出 | 下方「合成与预设」 |
| 自动视频抠像 | 自动切镜、稀疏检查帧、MatAnyone2 传播与分歧纠错 | [视频抠像指南](README_VIDEO_MATTING.md) |
| Fast Gift PostFX | Bloom、色散、锐化、自然饱和度、饱和度、对比度和亮度 | 下方「后处理、色键与 ICON」 |
| Gift Chroma Master | 绿幕 / 蓝幕键控、边缘清理、视频降抖和去溢色 | [色键指南](README_GIFT_CHROMA_MASTER.md) |
| Gift Icon Auto Restore | 恢复黑底光效，裁切并输出预览、1280 RGBA 和 168 RGBA ICON | [ICON 指南](README_GIFT_ICON_AUTO_RESTORE.md) |
| 遮罩与序列 | 遮罩渐变、首尾淡入淡出、帧切片、序列融合和变速 | 下方「节点索引」 |
| AE RGBA Tools | Unmult 去黑、带 Alpha 的多层混合 | [RGBA 指南](README_AE_RGBA_TOOLS.md) |

## 安装与更新

在 ComfyUI 的 `custom_nodes` 下安装：

```bash
git clone https://github.com/lingziwyh/ComfyUI-GiftHelperSuite.git
```

更新已有安装：

```bash
cd ComfyUI-GiftHelperSuite
git pull --ff-only
```

**若本地有自己的修改，先备份并处理差异，不要强制覆盖。** 安装完成后重启 ComfyUI、刷新前端。已配置热重载的环境，确认后台节点重载成功后可直接刷新；本包不负责安装或启用热重载插件。

### 按用途安装依赖

| 使用场景 | 额外代码 / 模型依赖 |
| --- | --- |
| 本包合成、PostFX、色键、遮罩、RGBA 工具和 ICON 恢复节点本身 | 不需要下载推理模型 |
| 视频读取与编码 | [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) |
| 自动视频抠像完整示例 | [ComfyUI-MatAnyone](https://github.com/FuouM/ComfyUI-MatAnyone)（需支持 MatAnyone2）、[ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG)、VideoHelperSuite、[KJNodes](https://github.com/kijai/ComfyUI-KJNodes) |
| 完整 ICON 示例 | 模板中指定的 Klein 模型及上游节点；不是仅安装 ICON 恢复节点就能运行整条流 |

外部插件的 requirements 应使用**启动 ComfyUI 的同一个 Python 环境**安装。本包不会在执行节点时静默 git clone、pip install 或升级 torch/CUDA。

### 缺失权重自动下载

| 模型 | 何时下载 | 新下载位置 |
| --- | --- | --- |
| TransNetV2 | 首次执行自动切镜且找不到有效权重时 | `models/GiftHelperSuite/transnetv2-pytorch-weights.pth` |
| MatAnyone2 | 首次执行自适应视频抠像且找不到有效权重时 | `models/GiftHelperSuite/matanyone2.pth` |
| InSPyReNet / RMBG 等 | 由外部 RMBG 节点按所选模型处理 | 通常为 `models/RMBG/...` |

本包两种权重支持已有路径复用、固定 SHA-256 校验和不覆盖式原子发布；损坏文件会报错并保留，不会静默覆盖。安装、启动、打开工作流不会触发这些模型下载。

- 单节点关闭：`auto_download=false`。
- 全局关闭本包下载：启动前设置 `GIFT_HELPER_AUTO_DOWNLOAD=0`。
- 上述开关**不控制外部 RMBG 插件**。
- 网络不可达时，错误信息提供手工下载地址及目标路径。

下载源、旧缓存路径和模型授权见[视频抠像指南](README_VIDEO_MATTING.md)及[第三方许可](THIRD_PARTY_NOTICES.md)。

## 示例工作流

将对应 JSON 拖入 ComfyUI；先按上表安装该示例所需依赖。

| 工作流 | 内容 | 素材与模型 |
| --- | --- | --- |
| [Gift PostFX Example](example_workflows/Gift_PostFX_Example.json) | 视频后处理与贴底合成 | 内置示例视频 / 背景；不需要推理模型 |
| [Gift Chroma Master Example](example_workflows/Gift_Chroma_Master_Example.json) | 色键、去溢色与透明合成 | 内置示例视频 / 背景；不需要推理模型 |
| [Gift Auto Video Matting](example_workflows/Gift_Auto_Video_Matting.json) | 自动前景遮罩与裸眼 3D 合成 | 复用内置视频 / 背景；需要抠像相关模型 |
| [Gift Icon Auto Restore Production](example_workflows/Gift_Icon_Auto_Restore_Production.json) | Klein → 色键 → 光效恢复 → ICON 输出 | 自行上传源图、准备模板指定的模型 |

前三个视频示例引用 `example_workflows/assets` 中的素材。插件启动时会把两段视频、一张背景图安全复制到 `ComfyUI/input` 根目录；已有同名但内容不同的文件会保留并提示冲突。

- 设置 `GIFT_HELPER_SKIP_EXAMPLE_ASSETS=1` 可关闭素材复制；这与模型下载开关无关。
- 如复制被禁用或发生冲突，可从[素材目录](example_workflows/assets)手工放置文件。
- ICON 示例中的 `GiftHelperSuite_Icon_Source.png` 是需要替换的占位文件名，不是内置素材。
- 卸载插件不会自动删除已复制的素材或已下载的权重。

## 合成与预设

`FastBottomFitOverlay` 将礼物图层按直播背景宽度缩放、保持比例并贴底，支持批量视频帧。

### 图层含义

| 输入 / 控制 | 作用 |
| --- | --- |
| `background_image` | 直播间底图 |
| `layer_image` | 礼物原图 / 视频底层 |
| `layer_mask` | 可选外部底层遮罩；与内置底层渐变叠乘 |
| `foreground_image` + `foreground_mask` | 可选贴回前景；不接入前景图像时仍可走普通合成 |
| `background_fade_ratio` | 背景大渐变，仅作用于礼物底层；0 关闭 |
| `enable_top_fade` + `top_fade_ratio` | 顶部短羽化，同时影响礼物底层和贴回前景，不改变直播间底图 |
| `center_scale` | 以适配后视频区域中心缩放；画布尺寸不变，1 为原大小、0 为消失 |

前景遮罩使用 `1=保留前景、0=透明` 的含义。贴回前景时应同时连接图像与遮罩；只接前景图像会按不透明图层处理。

**背景大渐变和顶部短羽化不是同一件事。** 前者让礼物环境融入直播间，后者缓和主体出上边框时的硬切口。

### 预设

| preset | Top Fade | 背景大渐变 | 圆角矩形羽化 | 圆角半径 |
| --- | --- | --- | --- | --- |
| `Custom` | 使用手动值 | 使用手动值 | 使用手动值 | 使用手动值 |
| `Low Coins` | 关闭 | 关闭 | 开启，ratio = 0.5 | 0.5 |
| `Standard` | 开启，ratio = 0.080 | 关闭 | 关闭 | 不启用 |
| `Naked-Eye 3D` | 开启，ratio = 0.030 | ratio = 0.52 | 关闭 | 不启用 |

预设在执行时覆盖相应羽化参数，不会自动调整其他参数。**要手调渐变，先把 preset 设为 `Custom`。** 手动模式下 Top Fade 与圆角矩形羽化互斥。

### 输出

- `image`：叠在直播间背景上的预览。
- `mask`：最终礼物图层遮罩。
- `packed_image`：左侧为 RGB 遮罩，右侧为黑底预乘 RGB 内容；不是 RGBA 图像。

Packed 支持两种尺寸：

- `Fit Content (Dynamic Height)`：保持适配后礼物高度；宽度为背景宽度的两倍。
- `Fixed Canvas (1440x1280)`：左右各 720 像素，内容贴底并在上方补黑。

`center_scale` 对预览及 Packed 输出的 RGB / Alpha 同步生效；超过可见画布的内容按既有裁切规则处理。

## 自动视频抠像

```text
输入视频 → 自动切镜 → 每镜头抽取检查帧 → InSPyReNet / RMBG 等生成检查遮罩
                └──────── 原镜头帧 + 检查遮罩 / 帧号 → MatAnyone2 自适应抠像
                                                      ↓
                                             MASK 列表合并 → 渐变合成
```

三个节点位于 `GiftHelperSuite/Video Matting`：

| 节点 ID | 功能 |
| --- | --- |
| `GiftAutoShotSplit` | TransNetV2 自动切镜及局部硬切点对齐，输出镜头列表 |
| `GiftMaskCheckFrames` | 按间隔抽帧，包含每镜头首帧和尾帧 |
| `GiftAdaptiveMatting` | 时序传播、稀疏分歧判断、重新初始化与局部反向回补 |

主要默认值：`interval=12`、`disagreement_threshold=0.025`、`focus_ratio=0.6`、`n_warmup=10`、`max_internal_size=1024`。

- `focus_ratio` 控制上方区域的分歧判断，不会裁掉下方遮罩。
- 检查更频繁、阈值更低会增加纠错与耗时，不保证视觉效果更好。
- 检查帧抠图模型决定前景归属；MatAnyone2 不保证补回初始遗漏的物体。
- 透明球体、粒子、复杂建筑和新增主体仍可能漏选或闪烁，不宣称全自动万能抠像。
- 新 ID 对应既有 V2 算法，并未把尚未统一验证的其他实验策略混入。
- 换素材时同步设置 Video Combine 的 `frame_rate`；内置视频为 24 FPS，`force_rate=0` 时按源帧率读取。

完整接法、依赖和缓存说明见[视频抠像指南](README_VIDEO_MATTING.md)。

## 后处理、色键与 ICON

### Fast Gift PostFX

`FastGiftPostFX` 支持 Bloom、色散、锐化和颜色调整，按小批次处理视频帧。

`performance_mode=auto` 遵循 ComfyUI 设备选择，可用时使用 CUDA；显存不足时缩小分块，必要时回退 CPU。`cuda` 模式不能处理单帧时会明确报错，`cpu` 模式用于排查或 CPU 运行。默认 `gpu_chunk_size=4`；实际耗时取决于分辨率、效果和设备，不将局部测试数字当作整条链路速度保证。

### Gift Chroma Master

推荐使用一体化 `GiftChromaMaster`：幕色键控 → 边缘清理 → 线性光去溢色。默认自动 CUDA、8 帧分块，并保留跨块时域上下文；专家节点用于分阶段调整。

输出 `foreground_alpha` 为 `1=前景、0=透明`，与某些节点采用的反向 MASK 含义不同。需要 RGBA 时使用 `GiftChromaMasterPackRGBA`。它是独立算法实现，不含 Adobe 代码、不保证与 After Effects 像素一致。[详细说明](README_GIFT_CHROMA_MASTER.md)

### Gift Icon Auto Restore

`GiftIconAutoRestore` 接收已分离的主体及原始黑底图，恢复光效、抑制抬黑噪声、窄边羽化、紧边裁切，并输出：

- `preview`：背景预览，未连接背景时使用内置参考底图。
- `icon_1280_rgba`：默认 1280 方形 RGBA ICON。
- `icon_168_rgba`：从紧边源图独立缩放的默认 168 方形 RGBA ICON。

恢复节点本身不运行 Klein；Klein 位于完整示例的上游。默认边缘羽化只覆盖画布短边外侧 2.5%，中心区域不变。[参数与有效像素说明](README_GIFT_ICON_AUTO_RESTORE.md)

### 序列与 RGBA 工具

`VideoTimeRemapSpeedPresets` 用于帧序列变速与时间重映射；不要把它等同于音频同步或生成式补帧。

`AEUnmultRGBA` 将黑底光效转成 RGBA；`AEAlphaOverRGBA` 支持 normal、screen、multiply、overlay 等混合模式和图层透明度。透明内容需由后续节点及导出格式继续保留 Alpha；普通 H.264 MP4 并不承载 RGBA 透明通道。[RGBA 参数说明](README_AE_RGBA_TOOLS.md)

## 节点索引

当前共 **20 个节点**。下表为可在工作流 JSON 中使用的正式 ID。

| 节点 ID | 用途 |
| --- | --- |
| `FastBottomFitOverlay` | 贴底、渐变、前景贴回及 Packed 合成 |
| `FastGiftPostFX` | 批量视频后处理 |
| `GiftAutoShotSplit` | 自动切镜 |
| `GiftMaskCheckFrames` | 稀疏检查帧 |
| `GiftAdaptiveMatting` | 自适应视频抠像 |
| `GiftChromaMaster` | 一体化色键 |
| `GiftChromaMasterKeyer` | 基础幕色键控 |
| `GiftChromaMasterCleaner` | 边缘清理与时域处理 |
| `GiftChromaMasterDespill` | 去溢色 |
| `GiftChromaMasterDiagnostics` | 遮罩与边缘诊断 |
| `GiftChromaMasterPreview` | 色键结果预览 |
| `GiftChromaMasterPackRGBA` | RGB 与 Alpha 打包 |
| `GiftIconAutoRestore` | 光效恢复与 ICON 输出 |
| `GiftMaskRamp` | 指定区间内遮罩 0→1 渐变 |
| `GiftMaskFadeInOut` | 首尾淡入淡出 |
| `GiftFrameSlice` | 含结束帧的序列切片 |
| `GiftMaskBlend` | 遮罩控制的两段序列融合 |
| `VideoTimeRemapSpeedPresets` | 序列变速 |
| `AEUnmultRGBA` | 黑底去黑转 RGBA |
| `AEAlphaOverRGBA` | RGBA 多层合成 |

## 兼容与迁移

- 旧合成节点 ID 保持不变；新增前景输入是可选项，普通非裸眼 3D 流程不需要接入。
- 自动抠像不再依赖 `ComfyUI-Video-Segmentation` 或旧研究包，但仍需要独立的 MatAnyone2 / RMBG 推理代码。
- 不会自动覆盖旧工作流、卸载旧插件或注册重复的 Research 别名。请在副本上迁移：

| 旧节点 ID | 新节点 ID |
| --- | --- |
| `GiftResearchShotSplit` | `GiftAutoShotSplit` |
| `GiftResearchCheckFrames` | `GiftMaskCheckFrames` |
| `GiftResearchAdaptiveMatAnyone2` | `GiftAdaptiveMatting` |
| `MaskGradientNode` | `GiftMaskRamp` |
| `MaskTransparentInOutNode` | `GiftMaskFadeInOut` |
| `FrameSliceNode` | `GiftFrameSlice` |
| `SequenceOverlayNode` | `GiftMaskBlend` |

前三项可保留节点 ID 与连线，仅替换节点类型和 `Node name for S&R`。旧遮罩 / 序列节点应同时核对参数和接口。其他 V3 / Lucida 实验节点不是同一算法，不要直接替换名称。

若独立的 AE / 遮罩插件造成同名注册冲突，先迁移和备份工作流，再按需禁用旧包。

## 测试与版本

在仓库目录使用 ComfyUI 的 Python：

```bash
python -m unittest discover -s tests -p "test_*.py"
```

单元测试不下载模型。视频抠像功能合并时另做了空模型目录下载、离线复用，以及同参数 124 帧迁移对照；详见[更新记录](CHANGELOG.md)。短片对照不代表所有素材质量都已通过。

版本号由入口 `__version__` 导出；README 与 CHANGELOG 同步维护。此版本为 **v0.6.0**，不意味着已发布到 Comfy Registry。

## 许可与作者

本仓库自有代码使用 [MIT License](LICENSE)。内置 TransNetV2 代码保留其 MIT 许可；MatAnyone2、RMBG、Klein 等外部模型与插件各自遵循上游条款，**不因本仓库 MIT 许可而获得统一商用授权**。[第三方说明](THIRD_PARTY_NOTICES.md)

作者：[lingziwyh](https://github.com/lingziwyh)。
