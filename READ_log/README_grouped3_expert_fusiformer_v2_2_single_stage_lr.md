# Grouped-3 Expert Fusiformer v2.2 — 单阶段 Base-Compatible Warmup + Cosine

配套程序：

```text
grouped3_expert_fusiformer_v2_2_single_stage_lr.py
```

本版本直接基于你已经跑通的 **v2.1 Diagnostics** 程序修改。Router、KMeans、difficulty-aware 3-Expert 分组、无 test leakage、Expert 微调层、Global-FT 对照、诊断 CSV/JSON、可视化和 deployment bundle 均保持原逻辑；本次重点只调整 **Expert / Global-FT 的学习率机制**。

---

## 1. 本次修改目标

不使用两阶段学习率训练。

整个微调只保留原 Base 训练文件的第一阶段机制：

```text
warmup_initial_lr
        │
        │ linear warmup
        ▼
    peak_lr
        │
        │ half-cosine decay
        ▼
    minimum_lr
```

程序中不再存在：

```text
Stage-2
第二次 warmup
LR reheating
第二段 cosine
additional_epochs
stage2_peak_lr
stage2_min_lr
```

---

## 2. 与 Base 第一阶段保持一致的核心机制

学习率按照 **optimizer update step** 调整，而不是按照 epoch 粗粒度调整。

Warmup 阶段：

```text
lr(t) = warmup_initial_lr
      + t / warmup_steps
      × (peak_lr - warmup_initial_lr)
```

Warmup 完成后使用 half-cosine：

```text
lr(t) = min_lr
      + 0.5 × (peak_lr - min_lr)
      × [1 + cos(pi × progress)]
```

最终衰减到 `minimum_lr`。

---

## 3. 当前正式推荐参数

程序顶部默认设置为：

```python
BATCH_SIZE = 64
GRADIENT_ACCUMULATION_STEPS = 1
FINE_TUNE_EPOCHS = 30

SCALE_LR_WITH_ACCUMULATION = False
ABSOLUTE_LR = 1e-5
BASE_LR = 1e-3

MINIMUM_LR = 1e-7
MINIMUM_LR_RATIO = None

WARMUP_RATIO = 0.04
WARMUP_STEPS_OVERRIDE = None
WARMUP_INITIAL_LR = 1e-7
```

因此默认学习率轨迹为：

```text
1e-7
  │
  │ 前 4% optimizer update steps 线性 warmup
  ▼
1e-5
  │
  │ 后 96% optimizer update steps half-cosine decay
  ▼
1e-7
```

`WARMUP_RATIO=0.04` 与原 Base Stage-1 设置保持一致。

---

## 4. 为什么峰值 LR 不直接照搬 Base scratch training

Base scratch training 可以使用更大的 peak LR，而当前 Expert 是从 `best_model_weights.pt` 开始进行局部微调。因此本版本只保持 **调度机制一致**，而将微调峰值显式设为：

```python
ABSOLUTE_LR = 1e-5
```

即：

```text
Base Stage-1 scheduler shape  保持一致
Expert fine-tuning LR scale   适当降低
```

这是为了避免已经训练好的高层表示在 Expert 微调初期被过大的学习率破坏。

---

## 5. Peak LR 的解析规则

程序保留与 Base 类似的两种配置方式。

### 推荐：固定 Expert peak LR

```python
ABSOLUTE_LR = 1e-5
```

此时：

```text
actual_peak_lr = 1e-5
```

### 可选：恢复 Base 的 batch-size scaling

如果设置：

```python
ABSOLUTE_LR = None
```

则：

```text
actual_peak_lr
= BASE_LR × lr_scaling_batch_size / 256
```

当前：

```python
SCALE_LR_WITH_ACCUMULATION = False
```

所以 scaling 使用 micro batch size，而不是 effective batch size。

正式 Expert 主实验不建议把 `ABSOLUTE_LR` 改为 `None`，因为 batch size 64、`BASE_LR=1e-3` 时会得到约 `2.5e-4`，这更接近 scratch training 的量级。

---

## 6. Minimum LR 的设置规则

下面两个参数必须恰好一个非 `None`：

```python
MINIMUM_LR = 1e-7
MINIMUM_LR_RATIO = None
```

当前使用固定 floor：

```text
minimum_lr = 1e-7
```

如果以后需要按比例控制，可改成例如：

```python
MINIMUM_LR = None
MINIMUM_LR_RATIO = 0.01
```

此时：

```text
minimum_lr = actual_peak_lr × 0.01
```

---

## 7. Warmup 按真实 optimizer step 计算

程序先计算：

```text
updates_per_epoch
= ceil(num_batches_per_epoch / gradient_accumulation_steps)
```

再计算：

```text
total_update_steps
= updates_per_epoch × FINE_TUNE_EPOCHS
```

默认：

```text
warmup_steps
= round(total_update_steps × 0.04)
```

如果未来提高 gradient accumulation，学习率仍然按真实 optimizer update 前进。

---

## 8. Module LR scale

为了让第一轮正式 Expert 实验更贴近 Base 的统一学习率策略，本版本默认：

```python
USE_MODULE_LR_SCALES = False

BACKBONE_LR_SCALE = 1.0
TEXT_MODULE_LR_SCALE = 1.0
REGRESSION_HEAD_LR_SCALE = 1.0
```

因此所有当前可训练参数共享同一条：

```text
1e-7 → 1e-5 → 1e-7
```

学习率曲线。

如果后续要研究“Regression Head 是否需要更大的 LR”，可以作为单独消融实验再开启：

```python
USE_MODULE_LR_SCALES = True
REGRESSION_HEAD_LR_SCALE = 5.0
```

但不建议与当前主实验同时改变。

---

## 9. Expert / Global-FT 公平性

以下模型全部使用同一套单阶段学习率机制：

```text
Expert-A shared3
Expert-B hardest
Expert-C second-hardest
Global-FT control
```

全部保持：

```text
同一 peak LR
同一 minimum LR
同一 warmup ratio
同一 epoch budget
同一 optimizer
同一 loss
```

它们的数据量不同，所以各自的：

```text
updates_per_epoch
total_update_steps
warmup_steps
```

会根据自己的 DataLoader 自动计算。

这样每个模型都在“自身完整 30 epoch 微调预算”上执行同形的 warmup + cosine 曲线。

---

## 10. Early stopping

当前仍保留：

```python
EARLY_STOPPING_PATIENCE = 8
```

所以 scheduler 虽然按照 30 epoch 的完整预算定义，如果 validation MAE 连续 8 epoch 不再改善，训练可以提前停止。

这不属于第二阶段训练，只是提前终止当前单阶段 schedule。

如果未来需要严格让每个模型跑满 30 epoch，可以设置：

```python
EARLY_STOPPING_PATIENCE = 0
```

但第一轮正式实验建议先保留 early stopping。

---

## 11. 新增学习率日志文件

每个 Expert 和 Global-FT 目录都会新增：

```text
lr_schedule_config.json
```

其中记录：

```text
scheduler
stage2_enabled = false
epochs_budget
micro_batch_size
gradient_accumulation_steps
effective_batch_size
scale_lr_with_accumulation
absolute_lr_requested
base_lr
actual_peak_lr
minimum_lr
warmup_ratio
warmup_steps
total_update_steps
cosine_steps
module lr scales
```

已有：

```text
epoch_metrics.csv
figures/lr_curve.png
```

仍然保留。

---

## 12. 训练日志中的学习率确认

每个 Expert 开始时会输出类似：

```text
expert_B_hardest LR |
SINGLE STAGE warmup+cosine |
peak=1e-05 |
min=1e-07 |
warmup_init=1e-07 |
warmup=xxx/xxxx update steps (4.00%) |
updates/epoch=xxx |
module_scales=False
```

看到这一行即可确认正式训练使用的是新的单阶段 Base-compatible LR。

---

## 13. Router / KMeans 逻辑不变

本次修改不改变：

```text
TRAIN graph_out
    ↓
L2 normalize
    ↓
KMeans.fit(TRAIN only)
    ↓
VAL predict
    ↓
VAL Base difficulty
    ↓
hardest        → Expert-B
second hardest → Expert-C
remaining 3    → Expert-A
```

TEST 仍然直到 Expert 与 Global-FT 的训练和模型选择全部完成后才进入最终评价。

因此原来的无 test leakage 设计完全保留。

---

## 14. 微调网络结构不变

当前仍然：

```python
FINETUNE_MODE = "last2"
TRAIN_TEXT_PROJ = False
```

对于当前 depth=6 Base：

```text
Block1-4       frozen
Block5-6       trainable
RegressionHead trainable
text_proj      frozen
MatSciBERT     frozen
```

三个 Experts 均从同一个 Base checkpoint 初始化。

---

## 15. 正式运行前建议检查

```bash
cd ~/llm_fusiformer
conda activate fusiformer

python -m py_compile grouped3_expert_fusiformer_v2_2_single_stage_lr.py
```

然后：

```bash
grep -nE \
"BASE_WEIGHTS_PATH|FINE_TUNE_EPOCHS|ABSOLUTE_LR|MINIMUM_LR|WARMUP_RATIO|USE_MODULE_LR_SCALES|SMOKE_TEST" \
grouped3_expert_fusiformer_v2_2_single_stage_lr.py
```

正式实验建议确认：

```python
SMOKE_TEST = False
FINE_TUNE_EPOCHS = 30
ABSOLUTE_LR = 1e-5
MINIMUM_LR = 1e-7
WARMUP_RATIO = 0.04
USE_MODULE_LR_SCALES = False
RUN_GLOBAL_FT_CONTROL = True
```

---

## 16. 后台正式运行

```bash
nohup python -u grouped3_expert_fusiformer_v2_2_single_stage_lr.py \
> grouped3_expert_v2_2_full.log 2>&1 &
```

查看日志：

```bash
tail -f grouped3_expert_v2_2_full.log
```

GPU：

```bash
watch -n 2 nvidia-smi
```

---

## 17. v2.1 → v2.2 的学习率变化

| 项目 | v2.1 | v2.2 |
|---|---|---|
| Router | train-only KMeans | 不变 |
| Embedded Router | 有 | 不变 |
| Difficulty-aware grouping | 有 | 不变 |
| Fine-tune layers | last2 + head | 不变 |
| Loss | MSE | 不变 |
| Scheduler | 单阶段 warmup+cosine | **改为 Base Stage-1 同构实现** |
| Warmup ratio | 0.05 | **0.04** |
| Peak LR | 1e-5 | **1e-5** |
| Min LR | 1e-7 | **1e-7** |
| LR scaling machinery | 简化 | **支持 Base 风格 absolute/base LR 解析** |
| Warmup step override | 无 | **支持** |
| Minimum LR ratio | 无 | **支持** |
| Module LR scales | 默认开启，Head 5× | **主实验默认关闭** |
| Stage-2 | 无 | **明确不存在** |
| LR 配置记录 | 基础 | **新增 `lr_schedule_config.json`** |

---

## 18. 当前建议固定的主实验配置

```python
FINETUNE_MODE = "last2"
TRAIN_TEXT_PROJ = False

FINE_TUNE_EPOCHS = 30

SCALE_LR_WITH_ACCUMULATION = False
ABSOLUTE_LR = 1e-5
BASE_LR = 1e-3
MINIMUM_LR = 1e-7
MINIMUM_LR_RATIO = None
WARMUP_RATIO = 0.04
WARMUP_STEPS_OVERRIDE = None
WARMUP_INITIAL_LR = 1e-7

USE_MODULE_LR_SCALES = False

LOSS_NAME = "mse"
RUN_GLOBAL_FT_CONTROL = True
SMOKE_TEST = False
```

这套配置的目标是尽量只研究：

```text
latent-region specialization
```

本身，而不同时引入第二阶段 scheduler、不同模块 LR 倍率或其他优化策略变化。
