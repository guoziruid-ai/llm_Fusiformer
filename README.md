# llm_Fusiformer

## 当前工程入口（2026-10-27）

本仓库同时保存当前 V2.1 ModernBERT 工作流与旧 Qwen/MatSciBERT 研究程序，不能混用模型、缓存或权重。正式入口位于 `scripts/train/` 与 `scripts/evaluate/`；请用文下的 `python -m` 规范模块路径。无 dropout scratch 入口为 `scripts/train/train_fusiformer_v2_modernbert.py`；`scripts/train/train_fusiformer_v2_modernbert_independent.py` 是 dropout/LR 默认值不同的实验。验证与独立测试分别由 `scripts/evaluate/validate_fusiformer_v2_modernbert.py`、`scripts/evaluate/test_fusiformer_v2_modernbert.py` 执行。根目录的两个训练入口及 validation、test 同名转发文件已移除，旧路径不再受支持；旧 ModernBERT 健康检查转发也已移除。运行前阅读 [使用指南](docs/USER_GUIDE.md)，给每次评估使用新的输出目录。

此前独立验证任务使用 A800、无 dropout epoch 46 权重完成验证/测试各 14,326 条，见 [历史远端报告](docs/REMOTE_VALIDATION_REPORT.md)。本轮工程重构在**新的** AutoDL 独立目录重新运行了完整验证和测试，指标及 28,652 行预测数值与历史基线一致，并使 `sample_id` 正确读取 `mp_id`；具体命令、退出状态和结果路径见 [本轮验收报告](docs/REFACTOR_TEST_REPORT.md)。[文件资产清单](docs/FILE_ASSET_INVENTORY.md) 和 [移动日志](docs/FILE_CLEANUP_LOG.md) 记录历史脚本去向。下文 Qwen 内容是旧工作流。

从项目根目录、已激活的现有 `fusiformer` 环境运行以下命令；默认配置读取 `data/`、`external/ModernBERT-base` 和 `checkpoints/best_model_weights.pt`，这些大资源不包含在仓库中，需先按 [使用指南](docs/USER_GUIDE.md) 放置或修改入口顶部配置。`environment.yml` 是已验证 Linux 环境的版本快照；`requirements-dev.txt` 单列测试依赖，本轮未安装/升级依赖。

```bash
python -m scripts.diagnostics.check_environment
python -m scripts.data.check_v2_assets
python -m scripts.train.train_fusiformer_v2_modernbert --check-config
python -u -m scripts.train.train_fusiformer_v2_modernbert  # 仅在准备启动新训练时
python -m scripts.evaluate.validate_fusiformer_v2_modernbert
python -m scripts.evaluate.test_fusiformer_v2_modernbert
python -m pytest -q
```

[项目结构](docs/PROJECT_STRUCTURE.md) 解释实际目录与调用关系；[开发约定](docs/DEVELOPMENT.md) 说明扩展位置、资源边界和测试方法。以下内容保留旧 Qwen 链的历史说明，不是当前 V2.1 的默认运行命令。

本仓库是在 `GengCongcong/llm_Fusiformer` 基础上修改得到的实验版本。当前版本的主要目标是：在原有 Fusiformer 晶体图神经网络框架中引入 Qwen 文本语义信息，使模型能够同时利用晶体结构图信息与材料文本描述信息进行性质回归预测。

默认任务为材料形成能预测：

```text
目标属性：e_form
数据路径：./data/my_dataset.json(默认), ./data/my_dataset_wiki_prompt.json(针对修改)
默认本地 Qwen 路径：/root/models/Qwen2.5-1.5B-Instruct
```

## 项目改动概述

与原始仓库相比，本版本主要进行了以下修改：

### 1. 引入 Qwen 文本编码能力

模型 `models/TextGuidedFusiformer.py` 中引入了 Qwen 作为文本编码器。文本经过 Qwen 得到句向量表示后，再通过 `text_proj` 投影到 Fusiformer 的图特征空间中。

当前默认使用本地 Qwen 模型：

```python
qwen_model_name = "/root/models/Qwen2.5-1.5B-Instruct"
```

### 2. 支持离线加载 Qwen

当前版本倾向于从服务器本地目录加载 Qwen，而不是每次从 Hugging Face 下载：

```python
local_files_only=True
```

因此，在运行模型前，需要保证本地已经存在完整的 Qwen 模型文件，例如：

```text
/root/models/Qwen2.5-1.5B-Instruct
```

可以先运行：

```bash
python test_qwen_local.py
```

### 3. 修正 Qwen hidden size 与 text_dim 的匹配问题

Qwen2.5 不同规模模型的隐藏层维度不同，例如：

```text
Qwen2.5-1.5B-Instruct: hidden_size = 1536
Qwen2.5-3B-Instruct:   hidden_size = 2048
Qwen2.5-7B-Instruct:   hidden_size = 3584
```


### 4. 调整数据加载与文本传递流程

当前文本数据加载流程为：

```text
JSON 原始数据
  ↓
get_dataset()
  ↓
CrystalGraphDataset.__getitem__()
  ↓
读取 description 文本字段
  ↓
Qwen tokenizer 分词
  ↓
crystal["text"]
  ↓
GraphCollecting
  ↓
inputs["text"]
  ↓
DataLoader.collect()
  ↓
batch_text = {
    "input_ids": [B, L],
    "attention_mask": [B, L]
}
  ↓
engine.to_device()
  ↓
TextGuidedFusiformer.forward()
  ↓
encode_text()
  ↓
Qwen 输出 last_hidden_state
  ↓
attention_mask 加权平均池化
  ↓
text_proj 投影到 embed_dim
  ↓
TextCrossAttention 引导节点/边特征
  ↓
图池化 + 回归预测
```

### 5. 增强 engine.to_device()

当前版本在 `engine.py` 中增加了递归式 `to_device()`，用于统一移动：

```text
dict
list
tuple
DGLGraph
Tensor
```

典型用法：

```python
from engine import to_device

inputs = to_device(inputs, device)
targets = targets.to(device, non_blocking=True)
```

## 推荐调试顺序


```bash
python test_qwen_local.py
python debug_dataset_one.py
python -m scripts.diagnostics.legacy.debug_bad_text_samples
python debug_dataloader_batch.py
python debug_engine_to_device.py
python debug_model_forward.py
```

以上是旧 Qwen 工作流命令，不能据此判断当前 V2.1 诊断状态。`debug_dataset_one.py` 仍使用字面路径 `ROOT/data/my_dataset.json`；文本扫描器规范实现仍指向相邻旧仓目录，运行前需核实输入路径。其旧根转发 `debug_bad_text_samples.py` 已移除。当前正式预检以文首 `scripts/` 命令为准。

每一步的作用如下：

| 顺序 | 脚本                          | 作用                                                |
| -- | --------------------------- | ------------------------------------------------- |
| 1  | `test_qwen_local.py`        | 检查本地 Qwen 是否可以离线加载                                |
| 2  | `debug_dataset_one.py`      | 检查单条样本是否能读取                                       |
| 3  | `debug_bad_text_samples.py` | 检查文本字段是否为空、缺失或异常                                  |
| 4  | `debug_dataloader_batch.py` | 检查 DataLoader 是否能正确组 batch                        |
| 5  | `debug_engine_to_device.py` | 检查 graph、line_graph、text、target 是否能移动到 GPU        |
| 6  | `debug_model_forward.py`    | 检查 Qwen 文本嵌入与 TextGuidedFusiformer forward 是否完整跑通 |

## 小样本训练验证

建议先运行小样本过拟合测试：

```bash
python test_dataset_effect_with_logging.py
```

该脚本默认用于验证：

```text
数据读取正常
Qwen tokenizer 正常
Qwen forward 正常
TextGuidedFusiformer forward 正常
loss 能下降
反向传播正常
日志与可视化文件能够保存
```

训练结束后，可查看：

```text
runs/dataset_effect/<timestamp>/train.log
runs/dataset_effect/<timestamp>/epoch_metrics.csv
runs/dataset_effect/<timestamp>/step_metrics.csv
runs/dataset_effect/<timestamp>/*.png
```

只重新绘图而不重新训练：

```bash
python test_dataset_effect_with_logging.py --plot-only --log-dir ./runs/dataset_effect/某次实验目录
```
