# 自动视频前景遮罩 / Automatic video matting

三个正式节点位于 `GiftHelperSuite/Video Matting`：

| 节点 ID | 显示名 | 职责 |
| --- | --- | --- |
| `GiftAutoShotSplit` | Gift Auto Shot Split / 自动切镜 | TransNetV2 切镜及局部硬切点对齐，输出镜头列表 |
| `GiftMaskCheckFrames` | Gift Mask Check Frames / 遮罩检查帧 | 每镜头间隔采样，始终包含首帧和尾帧 |
| `GiftAdaptiveMatting` | Gift Adaptive Matting / 自适应视频抠像 | MatAnyone2 时序传播、稀疏分歧纠错和局部反向回补 |

链路：视频 → 自动切镜 → 检查帧 → 自动抠图 → 自适应视频抠像 → MASK 列表转批次 → 原渐变合成。
这次发布保留已运行的 V2 时序算法，不混入其他尚未统一验证的实验策略。
改名不代表已消除闪烁、透明物体实心化或前景漏选。

## 新环境安装

安装本包后，视频抠像还需要以下节点包及各自的 requirements：

- [FuouM/ComfyUI-MatAnyone](https://github.com/FuouM/ComfyUI-MatAnyone)：需支持 `MatAnyone2`；已测试 2.1.4 / commit `87cbce38c03bb359471bd23704ebd1720e98a842`。
- [1038lab/ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG)：提供 InSPyReNet / RMBG 等检查帧遮罩。
- [VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)：示例的视频读取和导出。
- [KJNodes](https://github.com/kijai/ComfyUI-KJNodes)：示例的 `MaskListToMaskBatch`。

通过 ComfyUI Manager 或对应仓库安装；依赖请使用启动 ComfyUI 的同一个 Python。
**不会在运行中自动 git clone 或 pip install，不会升级 torch/CUDA。**
缺少 MatAnyone2 代码时，节点给出安装提示；不影响本包其他工具的注册。
切镜模型代码已经内置，不再依赖 ComfyUI-Video-Segmentation，也不需要 TensorFlow。

## 缺失权重自动下载

只在执行相应节点时检查；安装、启动、打开工作流和读取参数列表不会触发模型下载。

| 模型 | 新下载的保存位置 | 下载来源 |
| --- | --- | --- |
| TransNetV2 | `ComfyUI/models/GiftHelperSuite/transnetv2-pytorch-weights.pth` | [MiaoshouAI 转换权重](https://huggingface.co/MiaoshouAI/transnetv2-pytorch-weights)（固定 revision） |
| MatAnyone2 | `ComfyUI/models/GiftHelperSuite/matanyone2.pth` | [官方 v1.0.0](https://github.com/pq-yang/MatAnyone2/releases/tag/v1.0.0) |
| InSPyReNet / RMBG | `ComfyUI/models/RMBG/...` | 由外部 RMBG 节点按所选模型自行下载 |

TransNetV2 / MatAnyone2 会复用旧路径的同一权重，不会重复下载：

- `models/VLM/transnetv2-pytorch-weights/transnetv2-pytorch-weights.pth`
- `models/MatAnyone2/matanyone2.pth` 或 `models/matanyone2/matanyone2.pth`
- 已加载的 ComfyUI-MatAnyone 插件目录中的 `checkpoint/matanyone2.pth`

本包下载采用临时文件、固定 SHA-256 校验和不覆盖式原子发布。下载中断、HTML 错误页或校验失败不会留下可误加载的正式权重；重新运行即可重试。已有不匹配文件会报错并保留，不会静默覆盖。
HTTP(S) 代理遵循 Python/操作系统代理设置。网络不可达时，错误信息提供直链和手工放置路径。

节点的 `auto_download=false` 可关闭该节点下载；也可在启动前设置 `GIFT_HELPER_AUTO_DOWNLOAD=0`。
**这个开关只管本包两种权重，不控制外部 RMBG 节点的下载。**
模型授权分别遵循上游条款，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 工作流与参数

[自动视频抠像示例](example_workflows/Gift_Auto_Video_Matting.json) 使用本包已有示例素材，不含个人盘符路径。
默认 InSPyReNet 只是可替换的示例选择，不是所有礼物的最佳模型；也可选 RMBG-2.0。

- `interval=12`：检查帧间隔，越小检查成本越高。
- `disagreement_threshold=0.025`：触发纠错阈值；调低会增加重新初始化，不保证画面更稳定。
- `focus_ratio=0.6`：分歧判断只关注上方 60%，不是裁掉下方遮罩。
- `n_warmup=10` / `max_internal_size=1024`：保留现有运行参数。
- **换视频时同步设置 Video Combine 的 frame_rate**，输入 force_rate=0 保留源帧率。
- 合成节点用 `Naked-Eye 3D` 预设：Top Fade 0.03、背景渐变 0.52；手调渐变时先选 `Custom`，否则预设会覆盖手动值。

新工作流使用正式 ID，不注册重复的 Research 别名，也不卸载旧研究包。旧工作流可继续使用旧包。
迁移旧 JSON 时保留节点 ID 和连线，只按上表替换这三个节点类型（并更新 `Node name for S&R`）。
不要把其他 V3/Lucida 实验节点直接改成这些 ID，它们不是同一算法。

## 验证

使用 ComfyUI Python 在本仓库运行：

```text
python -m unittest discover -s tests -p "test_*.py"
```

单元测试使用小型本地夹具，不会下载模型。真实下载和 GPU 回归须另行执行。
