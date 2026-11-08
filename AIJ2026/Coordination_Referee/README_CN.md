# Coordination Referee｜多智能体协作故障诊断

[English](README.md) · **简体中文**

给定一次已经结束的多智能体协作，判断三个问题：**哪个环节出了错？团队最终是否完成目标？故障从哪一轮开始？** 本文件夹收录了最终保留的 **v84b** 提交，包括原始代码、两套训练好的 ModernBERT 及其分词器、候选集模型权重、公开演示数据和运行脚本。可单独复制整个文件夹使用，无需原项目目录。

## 官方服务器成绩

| Score | Macro f1 | Robustness f1 | Success f1 | Fault turn hit2 |
|---:|---:|---:|---:|---:|
| **0.94328** | 0.94406 | 0.94326 | 0.94827 | 0.93194 |

以上为参赛者提供的[服务器显示值](official_result.json)。随包演示测试集没有真值，也没有官方用于计算 Robustness 的样本标记；运行示例可以复现预测流程，但不能重算这张成绩表。

## 比赛任务

每条 CSV 记录是一次模拟的多智能体运行，包含原始目标、智能体及其角色、通信拓扑、有序消息、共享状态和产物。正式训练集有 10,000 条带标签记录，测试集有 4,000 条无标签记录，公开评测与私有评测各占 2,000 条且混排。一次交接可以失败，随后团队仍可能补救成功，因此**协作状态**和**任务是否成功**是两个独立的预测目标。[主办方入门仓库](https://github.com/dmayudin/coordination-referee/tree/690fbc90f611f79b0d21222b087f33a24705965f)另外提供了相同格式的公开演示数据。

| 输出列 | 含义 |
|---|---|
| `label` | 一个根因类别：正常 `clean`、交接遗漏 `dropped_handoff`、重复工作 `duplicated_work`、死锁 `deadlock`、冲突 `conflict`、目标偏移 `goal_drift`、失控循环 `runaway_loop`。 |
| `success` | 原始任务目标是否达成：达成为 `1`，否则为 `0`。 |
| `fault_turn` | 根因首次出现的消息轮次；正常运行填 `-1`。 |

[官方评分规则](https://github.com/dmayudin/coordination-referee/blob/690fbc90f611f79b0d21222b087f33a24705965f/docs/metrics.md)：总分由七分类 Macro F1（50%）、隐藏分布偏移子集上的 Macro F1，即 Robustness F1（25%）、任务成功的二分类 F1（15%）和故障定位 hit@2（10%）组成。定位以**所有真实故障运行**为分母，只有**故障类别正确，且预测起始轮次与标注相差不超过两轮**才算命中。

## 整体解题思路

1. **先找证据，再判断整次运行。** 从消息中解析任务分派、交接、交付、确认、重复操作和状态冲突，为每种故障提出可能的起始轮次。LightGBM 按运行分组做五折训练，给候选轮次评分，再把最高分、分差和位置汇总为整次运行的特征。参见[消息解析](src/parse.py)和[候选生成](src/feats.py)。
2. **用不同模型判断根因。** LightGBM 捕捉结构特征，时序 CNN 建模消息先后关系；五折 CatBoost 在这一阶段的分类概率融合中占一半。程序可将置信度达到 0.90 的无标签测试样本加入第二轮训练，因此改变测试批次可能改变预测。[故障仲裁模块](src/arbiter.py)进一步区分同一次运行中的根因和次生症状。
3. **分别处理成功判断和故障定位。** [候选集合 Transformer](src/csetnn.py)为每类故障保留最多八个候选轮次和一个空候选，用零初始化的周期数值嵌入残差表示位置与候选分数。五个成员在对数几率空间融合成功概率，并汇总候选轮次的分数。定位时选择前后两轮内累计概率最大的中心轮次，与比赛的 ±2 容差一致。
4. **让语言模型参与有分歧的分类决策。** [两个 ModernBERT 编码器](src/lm/lm_infer.py)读取有序日志，每个最多处理 8,192 个 token。先对两个模型的类别概率取几何平均，再按 `log P_stage2 + 0.2 × log P_LM` 融合。仅当融合改变第二阶段概率的最高分类别时才更新既有决策，类别更换后重新定位。语言模型的 Success 权重为 **0**，其定位指针不直接用于最终预测。最后根据[消息和状态证据](src/resid_post.py)修正特定类别或轮次。完整流程见 [pipeline.py](src/pipeline.py)。

## 如何复现

以下命令均在**本文件夹内**执行。重跑已验证的小样本推荐使用 **Python 3.13.9**（Windows x86-64）；下方也保留了历史 Python 3.12.10 的环境说明。校验与准备数据只需要 Python 标准库；运行完整模型还需要支持 bfloat16 的 NVIDIA CUDA 显卡、适配的驱动及下列依赖。离线运行前先装好环境。建议为交付文件、解压后的模型和 CSV 预留约 **2 GB**，另给 Python 环境和训练临时文件预留空间。

### 1. 校验提交包并准备环境

```sh
python reproduce.py verify
python reproduce.py prepare
python -m venv .venv
```

激活环境：Linux 用 `source .venv/bin/activate`；Windows PowerShell 用 `.\.venv\Scripts\Activate.ps1`。然后先安装参考 CUDA 版 PyTorch：

```sh
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r environment/requirements-smoke.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.is_bf16_supported())"
```

后两个结果应为 `True`。[小样本环境清单](environment/requirements-smoke.txt)列出了下方 GPU 实测时使用的版本。如需还原早期研究环境，请另建 **Python 3.12.10** 环境，从同一个 cu128 源安装 PyTorch `2.7.1`，再按[历史依赖清单](requirements.txt)运行 `python -m pip install -r requirements.txt`。两套环境均非比赛服务器镜像。`prepare` 将原包的 32 个文件解压至 `runtime/v84b/`；[src/](src/) 是方便阅读的源码副本。执行时使用解压后的原始入口，以便正确加载权重和分词器。

### 2. 重跑已验证的 GPU 小样本（560 条训练 / 35 条推理）

```sh
python examples/make_smoke.py --out work/smoke
python reproduce.py run --train work/smoke/train.csv --test work/smoke/test.csv --output work/smoke/predictions.csv --jobs 4 --lm-budget 8192
python tools/score_local.py --truth work/smoke/truth.csv --pred work/smoke/predictions.csv
```

此例生成 560 条训练记录和 35 条单独的推理记录。务必使用默认 `supplied-only` 模式：包内的演示数据也含这些留出记录。`truth.csv` 只传给评分脚本。输出包括 Macro F1、Success F1、joint hit@2 及**本地已知加权分 K**（`0.50 × Macro F1 + 0.15 × Success F1 + 0.10 × joint hit@2`，上限 0.75），不估算 Robustness。若高置信伪标签少于 20 条，原算法会在第一阶段结束。此时启动脚本会发现缺少语言模型融合日志，不会把这一输出保存为完整的 v84b 预测。

[GPU 实测记录](examples/reference/run.json)显示：两个语言模型均执行，22 条伪标签进入第二阶段，五折 CatBoost 和五个候选集模型均参与，最终得到 [35 行参考预测](examples/reference/gpu_predictions.csv)。不同显卡、依赖版本和批大小可能造成预测细微差异。本次也校验了下方全量演示输入，但没有重新执行 4,000 条的完整推理。

**没有 GPU** 也可以只用标准库重新生成划分，并重算[参考预测的本地指标](examples/reference/local_metrics.json)：

```sh
python examples/make_smoke.py --out work/smoke
python tools/score_local.py --truth work/smoke/truth.csv --pred examples/reference/gpu_predictions.csv
```

### 3. 为完整的 4,000 条公开演示数据生成预测

```sh
python tools/prepare_demo.py
python reproduce.py run --train work/demo/train.csv --test work/demo/test.csv --output work/demo/predictions.csv --jobs 4
python reproduce.py check --test work/demo/test.csv --output work/demo/predictions.csv
```

不需要另行下载数据。准备脚本从提交包内的 `extra_train.csv.gz` 恢复 10,000 条公开演示训练记录，从 `assets/demo-test.csv.gz` 恢复 4,000 条无标签演示测试记录，同时校验两个 CSV 的哈希。默认 `supplied-only` 模式只以传入 CSV 作为有标签训练来源，关闭额外演示数据合并（`CR_NO_EXTRA=1`），在本次训练数据上重新训练候选集模型（`CR_CSET_MODE=train`），直接加载两套语言模型权重。输出须选择新的文件名；预测每个测试 ID 恰好一行，列为 `run_id,label,success,fault_turn`。

显存较小时，可在 `run` 命令末尾加 `--lm-budget 8192`，先用较小的批次执行两套语言模型，再运行其余模型。省略该参数时沿用原始启动方式：语言模型与其他阶段并行，批次预算为 65,536 个 token。不同批大小可能造成末位数值差异。

### 4. 换用比赛平台文件

如已取得[比赛平台数据](https://dsworks.ru/en/champ/aij26-referee/data)，将路径替换为实际的训练和测试 CSV。若要保留**原提交包默认的数据加载行为**，运行：

```sh
python reproduce.py run --profile archive-default --train platform/train.csv --test platform/test.csv --output work/platform_predictions.csv --jobs 4
```

`archive-default` 允许原包合并去重后的公开演示训练行，并保留候选集模型的自动模式。只希望使用传入训练 CSV 时，省略 `--profile archive-default`。本文件夹没有官方测试真值、服务器实际作业配置，也没有两个语言模型从头训练的完整配方或样本清单；可以重跑现有模型，无法保证服务器分数逐位一致，也不能据此声称本地留出样本对语言模型是独立的。

## 随文件夹一起保留的文件

| 文件或目录 | 用途 |
|---|---|
| [原始提交包](assets/submission_v84b_v81_lm2.zip) | 完整的 v84b：入口 `solution.py`、候选集模型权重、两套 ModernBERT 权重及分词器、压缩演示训练集。SHA-256：`918ae44f7fbb01756594c165e19dd8969e7d599ec3dfbe1dea53952129bbd8c4`。 |
| [演示测试集](assets/demo-test.csv.gz) | 来自[主办方仓库的固定版本](https://github.com/dmayudin/coordination-referee/tree/690fbc90f611f79b0d21222b087f33a24705965f/data)，未修改，无标签。 |
| [运行脚本](reproduce.py) · [依赖清单](requirements.txt) | 校验原包、解压、运行模型、检查输出。 |
| [数据准备](tools/prepare_demo.py) · [小样本划分](examples/make_smoke.py) · [本地评分](tools/score_local.py) | 均可离线运行，不依赖上级工作区。 |

若在 GitHub 分发，建议通过 Release 附件或 Git LFS 提供原始 ZIP。从 Release 下载后，将其放在 `assets/submission_v84b_v81_lm2.zip`。若拿到的只是 LFS 指针文件而非实际压缩包，`verify` 会明确报错。
