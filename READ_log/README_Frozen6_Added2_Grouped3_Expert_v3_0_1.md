# Frozen-6 + Added-2 Grouped-3 Expert Fusiformer v3.0

本版本用于验证新的主实验方案：

> **保留已经训练完成的 depth=6 Fusiformer 作为冻结 Base，在原 6 个 Fusiformer 图网络 Block 后继续增加 2 个可训练图网络 Block，并为三个 latent Expert 分别训练 Added Block7/8 + 独立 regression head。Router 仍由冻结 Base 的 graph_out 完成。**

配套文件：

```text
Added2ExpertFusiformer.py
train_frozen6_added2_grouped3_expert_v3.py
README_Frozen6_Added2_Grouped3_Expert_v3.md
```

建议把前两个 Python 文件都放在项目根目录：

```text
/root/llm_fusiformer/
├── Added2ExpertFusiformer.py
├── train_frozen6_added2_grouped3_expert_v3.py
├── models/
├── loader/
├── data/
└── runs/
```

---

## 1. 研究目标

已有 Base：

```text
Crystal graph + cached MatSciBERT
        ↓
Embedding
        ↓
Fusiformer Block1
        ↓
...
        ↓
Fusiformer Block6
        ↓
readout
        ↓
graph_out
        ↓
regression_head
        ↓
e_form
```

新的 Expert 不再修改已经训练好的 Block5/6，而是：

```text
Frozen Base Block1-6
        ↓
Added Block7
        ↓
Added Block8
        ↓
readout
        ↓
Expert regression_head
        ↓
e_form
```

其中：

```text
Block1-6        frozen
Block7-8        trainable
regression_head trainable
MatSciBERT       frozen
text projection  frozen
```

三个 Expert 拥有各自独立的 Block7/8 与 regression head。

---

## 2. 为什么程序使用 depth=8 Expert branch

本项目原始 `TextGuidedFusiformer` 已经通过 `depth` 控制 `blocks` 数量。

因此本版本不手工重写 Fusiformer Block 的内部 forward，而是：

```python
Base:
    depth = 6

Expert:
    depth = 8
```

随后：

```text
Base checkpoint
    ↓
复制所有能够匹配的 Base 参数
    ↓
Expert Block1-6 = Base Block1-6
Expert head     = Base head
    ↓
新增 Block7/8
```

这样新增 Block7/8 会由项目原有的模型 forward 自动位于：

```text
Block6
↓
Block7
↓
Block8
↓
readout
↓
regression_head
```

而不是错误地放在已经池化完成的 `graph_out` 后面。

---

## 3. Router 完全保留当前严格无泄漏方案

Router 仍然只使用 **冻结 depth=6 Base**：

```text
TRAIN
  ↓
Frozen Base depth=6
  ↓
graph-only graph_out
  ↓
L2 normalize
  ↓
KMeans.fit(TRAIN only, K=5)
```

Validation：

```text
VAL
 ↓
Frozen Base graph_out
 ↓
KMeans.predict
 ↓
Base validation error by raw cluster
 ↓
hardest cluster        → Expert-B
second hardest cluster → Expert-C
remaining three        → Expert-A
```

Test 在三个 Expert 与 Global-Added2 全部训练完成之前不会参与 Router 或 Expert grouping。

---

## 4. 三个 Expert

最终得到：

```text
Expert-A
  ├─ shared 3 raw KMeans clusters
  ├─ Added Block7-A
  ├─ Added Block8-A
  └─ Head-A

Expert-B
  ├─ hardest cluster
  ├─ Added Block7-B
  ├─ Added Block8-B
  └─ Head-B

Expert-C
  ├─ second-hardest cluster
  ├─ Added Block7-C
  ├─ Added Block8-C
  └─ Head-C
```

Base Block1-6 不参与梯度更新。

---

## 5. Added Block7/8 初始化

默认：

```python
ADDED_BLOCK_INIT = "clone_last"
```

即：

```text
Added Block7 ← copy trained Base Block6
Added Block8 ← copy trained Base Block6
```

而不是完全随机初始化。

原因是新增层直接随机初始化可能在训练初期严重破坏已经学习好的表示。

可用于消融的另一种方式：

```python
ADDED_BLOCK_INIT = "random"
```

主实验建议保持：

```python
ADDED_BLOCK_INIT = "clone_last"
```

---

## 6. Residual prediction gate

为了进一步避免新增两层在训练初期破坏 Base，默认开启：

```python
USE_RESIDUAL_PREDICTION_GATE = True
RESIDUAL_GATE_INIT = 0.05
```

最终预测：

```text
y_final
=
y_base
+
gate × (y_expert - y_base)
```

其中：

```text
gate = sigmoid(residual_logit)
```

初始：

```text
gate ≈ 0.05
```

因此训练开始时：

```text
y_final ≈ 95% Base + 5% Added2 Expert
```

但 gate 非零，因此 Block7/8 从第一个 optimizer step 就能获得梯度。

随着训练：

```text
gate
```

可以自动学习 Expert correction 应该占多大比例。

每个 epoch 都会保存：

```text
residual_gate
```

并生成：

```text
figures/residual_gate_curve.png
```

---

## 7. 为什么不把 gate 初始化成 0

如果：

```text
gate = 0
```

则：

```text
y_final = y_base
```

虽然初始预测完全等于 Base，但：

```text
∂y_final / ∂Expert = gate = 0
```

新 Block7/8 在初始阶段拿不到有效梯度。

因此默认使用：

```text
0.05
```

作为“接近 Base、但仍能立即训练”的折中。

---

## 8. Global-Added2 control

本版本不再使用以前的 Global-FT last2 对照。

新的关键对照是：

```text
Global-Added2
```

它使用与三个 Expert **完全相同的模型容量**：

```text
Frozen Block1-6
+
Added Block7
+
Added Block8
+
trainable copied head
+
same residual gate
```

区别只有：

```text
Global-Added2:
    使用整个 TRAIN

Grouped-3 Added2 Experts:
    各自只使用 routed TRAIN 子集
```

因此最终最关键的比较是：

```text
Base
vs
Global-Added2
vs
Routed-Added2-3Expert
```

如果：

```text
Routed-Added2-3Expert < Global-Added2
```

才更有力地说明提升来自：

```text
latent-region specialization
```

而不仅仅是：

```text
模型多了两层
```

---

## 9. 推荐正式学习率

与之前“微调已有 Block5/6”不同，Added Block7/8 是新的可训练容量，因此不能继续使用 `1e-8` 量级的 peak LR。

主实验默认：

```python
ABSOLUTE_LR = 5e-5

MINIMUM_LR = 5e-7
WARMUP_INITIAL_LR = 1e-7
WARMUP_RATIO = 0.04
```

仍然只有一个阶段：

```text
1e-7
  ↓
4% optimizer-step linear warmup
  ↓
5e-5
  ↓
half-cosine decay
  ↓
5e-7
```

没有 Stage-2。

---

## 10. Differential LR

本方案同时训练：

```text
Added Block7/8
Base-copied regression_head
residual gate
```

默认：

```python
USE_MODULE_LR_SCALES = True

ADDED_BLOCK_LR_SCALE = 1.0
EXPERT_HEAD_LR_SCALE = 0.2
RESIDUAL_GATE_LR_SCALE = 1.0
```

所以当基础 peak LR 为：

```text
5e-5
```

实际：

```text
Added Block7/8 peak LR = 5e-5
Head peak LR           = 1e-5
Gate peak LR           = 5e-5
```

原因：

- Block7/8 是新加入层，需要更积极学习；
- regression head 已从 Base 拷贝，不宜与新层同等幅度更新；
- gate 只有一个很小的标量参数，可以与新增层使用同量级调度。

推荐后续 LR 消融：

```text
1e-5
2.5e-5
5e-5
```

但第一轮只跑 `5e-5` 即可。

---

## 11. 单个 Expert 的参数保存

为了避免三个 Expert checkpoint 各自重复保存整个 Base，`best_model.pt` 只保存：

```text
Added Block7
Added Block8
regression_head
residual_logit
```

即：

```text
compact_expert_state_dict
```

Base checkpoint 仍然通过：

```python
BASE_WEIGHTS_PATH
```

单独读取。

这比保存三个完整 depth=8 模型更适合当前 shared-backbone Expert 思路。

---

## 12. Deployment bundle

最终：

```text
deployment/
└── frozen6_added2_grouped3_expert_bundle.pt
```

包含：

```text
Base depth-6 weights
Embedded KMeans router
cluster → expert mapping

Expert-A compact state
Expert-B compact state
Expert-C compact state
```

因此保存结构相当于：

```text
1 × shared Base
+
3 × (Block7 + Block8 + Head + gate)
```

而不是保存三个完整 Base。

---

## 13. 训练日志

沿用 v2.3 Detailed Epoch Logging。

每个 epoch 输出：

```text
TRAIN:
MSE / MAE / RMSE / throughput

VAL:
MSE / MAE / RMSE
bias
median AE
P90
P95
P99
max AE
R²

LR:
warmup / cosine phase
optimizer step
group LR range

MODEL:
residual gate

TIME:
train
validation
epoch
elapsed
ETA

GPU:
allocated
reserved
peak allocated
peak reserved
```

并保存：

```text
epoch_metrics.csv
latest_epoch_status.json
tensorboard/
figures/
```

---

## 14. 当前训练参数

第一次正式实验建议：

```python
MODEL_DEPTH = 6
EXPERT_EXTRA_DEPTH = 2

ADDED_BLOCK_INIT = "clone_last"

USE_RESIDUAL_PREDICTION_GATE = True
RESIDUAL_GATE_INIT = 0.05

RUN_GLOBAL_ADDED2_CONTROL = True

FINE_TUNE_EPOCHS = 30

ABSOLUTE_LR = 5e-5
MINIMUM_LR = 5e-7
WARMUP_RATIO = 0.04
WARMUP_INITIAL_LR = 1e-7

USE_MODULE_LR_SCALES = True
ADDED_BLOCK_LR_SCALE = 1.0
EXPERT_HEAD_LR_SCALE = 0.2
RESIDUAL_GATE_LR_SCALE = 1.0

EARLY_STOPPING_PATIENCE = 8
LOSS_NAME = "mse"

SMOKE_TEST = False
```

---

## 15. Base checkpoint

正式实验必须固定：

```python
BASE_WEIGHTS_PATH = (
    "runs/fusiformer_production_matscibert_mlp64/"
    "你的正式Base目录/checkpoints/best_model_weights.pt"
)
```

该 checkpoint 必须是：

```text
depth = 6
```

程序仍会做 architecture validation。

---

## 16. Smoke Test

第一次部署到服务器建议：

```python
SMOKE_TEST = True
```

此时：

```text
每个 Expert 2 epochs
跳过 Global-Added2
```

检查：

```text
Router
Expert-A/B/C 构造
Block7/8 是否可训练
head 是否可训练
Base Block1-6 是否冻结
residual gate 是否变化
checkpoint 是否可保存/加载
final evaluation
```

全部跑通后：

```python
SMOKE_TEST = False
```

开始正式实验。

---

## 17. 启动命令

```bash
cd /root/llm_fusiformer
conda activate fusiformer

python -m py_compile \
    Added2ExpertFusiformer.py \
    train_frozen6_added2_grouped3_expert_v3.py
```

Smoke：

```bash
python -u train_frozen6_added2_grouped3_expert_v3.py
```

正式后台运行：

```bash
nohup python -u train_frozen6_added2_grouped3_expert_v3.py \
> frozen6_added2_grouped3_v3_full.log 2>&1 &
```

查看：

```bash
tail -f frozen6_added2_grouped3_v3_full.log
```

---

## 18. 正常启动后应看到

每个 Expert 初始化时应出现类似：

```text
expert_B_hardest architecture |
frozen Base Blocks1-6 |
train Added Blocks7-8 + head |
init=clone_last |
residual_gate=True |
gate_init=0.0500
```

以及：

```text
effective peak LR |
Added Block7/8=5.000e-05 |
head=1.000e-05 |
gate=5.000e-05
```

`trainable_parameters.json` 中应看到可训练名称只属于：

```text
expert_model.blocks.6.*
expert_model.blocks.7.*
expert_model.regression_head.*
residual_logit
```

如果看到：

```text
expert_model.blocks.0.*
...
expert_model.blocks.5.*
```

出现在 trainable list，说明冻结失败，应停止正式实验。

---

## 19. 结果判断

第一层：

```text
Base
vs
Global-Added2
vs
Routed-Added2-3Expert
```

关注：

```text
MAE
RMSE
P95
P99
catastrophic error rate
```

第二层：

```text
per_cluster_improvement.csv
```

检查困难簇是否真正改善。

第三层：

```text
Expert-A shared three clusters
```

检查是否存在某个 easy cluster 被共享 Expert 伤害。

第四层：

```text
residual_gate_curve.png
```

解释 Expert 是否真正依赖新增 Block7/8。

例如：

```text
gate 0.05 → 0.25
```

说明训练逐渐增加 Expert correction。

如果一直：

```text
≈0.05
```

且指标没有改善，则应检查新增层梯度/LR。

---

## 20. 建议的完整实验矩阵

为了论文解释，建议最终至少保留：

| Experiment | Base Block1-6 | Added Block7/8 | Routing | Trainable Head |
|---|---|---|---|---|
| Base | 原模型 | 无 | 无 | 否 |
| Old Last2 Expert | B5/6 微调 | 无 | 3 Expert | 是 |
| Global-Added2 | 冻结 | 共享新增2层 | 无 | 是 |
| **Grouped3-Added2** | **冻结** | **每 Expert 独立2层** | **3 Expert** | **是** |

核心结论应由：

```text
Grouped3-Added2 vs Global-Added2
```

支持。

---

## 21. 一个工程上的重要说明

为了不修改你项目中 `TextGuidedFusiformerCached.py` 的内部 forward，本版本采用：

```text
depth=8 Expert model
```

来保证新 Block7/8 真正进入原模型的 Block 序列。

Residual gate 开启时，还会保留一个冻结 depth=6 Base forward 作为 prediction anchor，因此训练/推理计算量会比单个 depth=8 forward 更高。

这是当前版本为了：

```text
兼容现有代码
+
保证结构位置正确
+
训练初期稳定
```

做的工程折中。

如果本实验确认有效，下一阶段可以再修改核心模型，做成真正的：

```text
一次共享 Block1-6 forward
        ↓
Router
        ↓
只执行被选中的 Expert Block7/8
```

这样能够显著降低推理计算重复，但那属于效率优化，不应该与本轮“Added2 Expert 是否有效”的方法验证同时改变。

---

## 22. 当前版本的科学问题

这套实验最终回答两个相互独立的问题：

### 问题 A

```text
在冻结一个已经训练好的 depth=6 Base 后，
继续增加两层局部图网络容量是否有价值？
```

由：

```text
Base vs Global-Added2
```

回答。

### 问题 B

```text
相同新增容量按 Base latent region 做专家化，
是否优于全局统一新增两层？
```

由：

```text
Global-Added2 vs Routed-Added2-3Expert
```

回答。

如果两者都成立，则能够形成较完整的方法论：

```text
Base latent representation
        ↓
difficulty-aware routing
        ↓
region-specific depth expansion
        ↓
improved e_form prediction
```


---

## v3.0.1 Bugfix（2026-09-13）

修复首次进入 `train_specialist()` 时的：

```text
KeyError: 'total_parameter_count'
```

原因是 `Added2ExpertFusiformer.trainable_summary()` 在 v3.0 中返回：

```text
total_parameter_count_runtime
trainable_fraction_runtime
```

而训练程序日志仍读取旧字段：

```text
total_parameter_count
trainable_fraction
```

v3.0.1 同时做了两层兼容：

1. 模型 summary 同时返回 canonical 与 runtime alias；
2. 训练程序日志使用兼容 fallback。

该问题发生在 optimizer/epoch 开始之前，因此不会造成模型权重被错误更新。

另外修复了 residual Base anchor 的输入复制顺序：现在在 Expert forward 之前复制 Base 输入，避免项目图网络 forward 若写入临时 DGL node/edge feature 时污染冻结 Base anchor 的输入。
