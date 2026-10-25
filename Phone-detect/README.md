# [PhoneGuard-P2：考场手机检测](https://github.com/swj-stone/phone-detect)

**中文** | [English](README_EN.md)

PhoneGuard-P2 是一个面向考场图片与视频帧的单类别手机检测项目。项目以 YOLOv8 为基线，围绕小目标检测、复杂场景鲁棒性、可复现训练和严格的数据边界，构建从安全预处理到训练、评估、恢复与提交推理的完整工程流程。

> 本仓库的目标是提供可审计、可复现的研究与工程实现，而不是承诺某个排行榜分数。

## 项目目标

- 精确定位画面中的手机，并输出边界框与置信度。
- 改善小目标、遮挡、低光、模糊、压缩、多尺度和多角度场景下的检测效果。
- 减少遥控器、耳机盒、充电宝等相似物体造成的误检。
- 在精度、召回率和推理速度之间取得适合实时监考的平衡。
- 保持训练、评估、checkpoint 选择和恢复过程可追踪、可验证。

比赛指标与本地评估保持一致：

```text
score = 100 × (0.5 × mAP@0.5 + 0.3 × Precision + 0.2 × Recall)
```

## 方法概览

PhoneGuard-P2 在常规 P3/P4/P5 检测尺度之外增加 P2 小目标检测头。系统包含两个训练入口和两项工程控制机制：

- **Baseline**：用于快速建立 YOLOv8 对照结果；其 `--pretrained` 模式可能下载权重，严格离线运行应显式选择 `--scratch`。
- **PhoneGuard-P2**：采用 P2/P3/P4/P5 四尺度检测、定制增强和受控训练流水线。
- **Robust → Refine 课程训练**：第一阶段覆盖低光、模糊、压缩、有限遮挡和尺度变化；第二阶段关闭强 Mosaic/MixUp、降低学习率，并通过质量门拒绝退化结果。
- **可复现与可恢复**：数据指纹、checkpoint binding、追加式运行证据和经过认证的恢复入口共同保护实验身份。

两阶段质量门用于降低退化与过拟合风险，但不能保证完全不过拟合，也不保证一定提升平台分数。

## 工作流与数据边界

```text
本地训练归档 ──→ 安全预处理与确定性 train/val 划分 ──→ 训练与内部验证
                                                              ↓
本地测试归档 ──→ 安全预处理并隔离为 processed/test ──→ 最终预测 ──→ submission JSON
```

预处理器可以在本地校验并准备隔离的测试集，但训练、验证、阈值调整和 checkpoint 选择绝不消费测试样本；只有最终预测模块可以读取 `processed/test`。schema-v2 的正式训练身份只依赖 train、val 与训练归档，不包含测试集路径、文件名、大小、哈希或成员列表。

官方数据、私有测试集、生成的权重、运行结果、凭据和 submission 均应保留在本地，不得提交或重新分发。竞赛任务与数据约束应以主办方提供的原始本地材料为准。

## 仓库结构

| 路径 | 作用 |
| --- | --- |
| `project/processdata/` | 安全解包、校验、确定性划分、原子替换和训练专用视图 |
| `project/baseline/` | YOLOv8 baseline 的训练、验证和提交推理 |
| `project/experiments/phoneguard_p2/` | 四尺度模型、增强、单阶段训练、课程训练、评估、选择与恢复 |
| `tests/` | 合成测试、单元测试和安全回归测试 |
| `phoneguard-p2-size-aware-cancellation-20260828/` | 独立的 train-only 尺度感知采样与协作式中断恢复交接包 |
| `requirements.txt` | 主项目 Python/CUDA 依赖 |

## 环境准备

推荐在 Windows PowerShell 中运行。真实 GPU 训练需要可用的 NVIDIA GPU；`requirements.txt` 当前使用 PyTorch CUDA 12.8 构建。

```powershell
git clone https://github.com/swj-stone/phone-detect.git
Set-Location phone-detect

$RepoRoot = (Get-Location).Path
$EnvRoot = "${RepoRoot}-venv"
python --version  # 确认输出为 Python 3.13.x
python -m venv $EnvRoot
& "$EnvRoot\Scripts\Activate.ps1"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

依赖安装可能受 Python、CUDA、显卡驱动和 PyTorch wheel 可用性影响。开始长训练前，请先确认 `torch.cuda.is_available()` 返回 `True`。

## 快速验证

从仓库根目录运行主项目测试：

```powershell
python -u -m unittest discover -s tests -p "test_*.py" -v
```

纯代码 checkout 缺少本地验收元数据时，相关测试会按设计跳过。随后可运行不需要正式数据或 GPU 的 tiny smoke：

```powershell
Set-Location project
python -u -m experiments.phoneguard_p2.smoke --tiny --device cpu
```

tiny smoke 只验证代码路径和工件契约，不代表真实模型质量。

## 主要入口

以下命令用于查看经过实现支持的参数。预处理器和课程训练器提供 `--dry-run`；其他入口应先查看 `--help`，不要假定它们支持只读预检：

```powershell
Set-Location project

python -u -m processdata.preprocess --help
python -u -m baseline.train --help
python -u -m baseline.evaluate --help
python -u -m baseline.predict --help

python -u -m experiments.phoneguard_p2.train --help
python -u -m experiments.phoneguard_p2.curriculum --help
python -u -m experiments.phoneguard_p2.evaluate --help
python -u -m experiments.phoneguard_p2.smoke --help
```

课程训练中断后，使用对应 pipeline 根目录的不可变 manifest 恢复：

```powershell
python -u -m experiments.phoneguard_p2.curriculum `
  --resume-pipeline "<results-root>/<pipeline-id>/pipeline_manifest.json"
```

恢复过程会重新核对数据指纹、模型身份、事件链和 checkpoint 字节，不会覆盖已有 execution。

## 输出与可复现性

本地运行工件默认位于 `project/results/`。正式 PhoneGuard-P2 pipeline 通常包含：

- 不可变 pipeline manifest 和追加式事件记录；
- robust/refine 各阶段的 checkpoint、binding 与独立 execution；
- 本地验证指标、checkpoint 排名和质量门决定；
- 最终选中权重及其决策证据。

这些目录可能包含私有数据衍生信息或大文件，应保持未跟踪状态。PhoneGuard-P2 正式流程只接受可信本地资源和经过确认的 `.pt` 文件，不会自动下载；PyTorch checkpoint 可能包含 pickle 载荷。Baseline 的 `--pretrained` 模式是独立入口，可能联网获取权重。

## 代码导航

- `project/experiments/phoneguard_p2/`：受控训练、恢复、评估与工件契约。
- `project/processdata/`：安全预处理与训练视图实现。
- `project/baseline/`：YOLOv8 对照流程。
- `phoneguard-p2-size-aware-cancellation-20260828/`：独立的 train-only 实验交接包。

仓库内可能保留历史工程说明；在共享或引用其中的分数、路径、摘要和哈希前，应先完成隐私审查。

## 限制

- 模型效果依赖数据质量、划分策略、硬件、随机性和训练预算。
- 本地验证分数与排行榜分数可能存在分布差异。
- 质量门可以拒绝明显退化的候选模型，但不能消除所有过拟合风险。
- 本仓库未声明开源许可证；使用代码和数据前请确认相应授权与竞赛规则。
