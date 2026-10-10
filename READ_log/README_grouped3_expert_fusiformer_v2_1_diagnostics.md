# Grouped-3 Expert Fusiformer v2.1 Diagnostics

配套程序：

```text
grouped3_expert_fusiformer_v2_1_diagnostics.py
```

本版本在 v2 的 **严格 train-only KMeans Router + Embedded Router + Grouped-3 Expert** 基础上，补充了完整的 **KMeans/Router 数值诊断、潜在空间可视化、Expert 学习曲线、Base/Global-FT/Expert 最终误差可视化**。训练逻辑和无泄漏设计不变，新增内容只用于分析与结果展示。

---

## 1. 总体实验结构

正式流程：

```text
Original dataset
      │
      ├── TRAIN
      ├── VAL
      └── TEST

TRAIN crystal
      ↓
Frozen Base Fusiformer
      ↓
graph-only graph_out
      ↓
L2 normalization
      ↓
KMeans.fit(TRAIN only, K=5)
      ↓
R0 ... R4

VAL crystal
      ↓
KMeans.predict
      ↓
Base prediction difficulty on VAL
      ↓
Hardest cluster       → Expert-B
Second hardest        → Expert-C
Remaining 3 clusters  → Expert-A

Expert-A/B/C
      ↓
all initialized from the SAME Base best checkpoint
      ↓
default fine-tuning: Block5 + Block6 + regression_head
      ↓
validation early stopping

TEST is touched only after all training/model selection is complete
      ↓
EmbeddedKMeansRouter
      ↓
Expert-A/B/C
      ↓
e_form prediction
```

因此该方法是：

```text
5-cluster train-only Router
        ↓
3 difficulty-aware Experts
```

不是 K=3 聚类，也不依赖 `assignments.csv` 查表。

---

## 2. 无泄漏设计

v2.1 保持以下规则：

```text
KMeans.fit            → TRAIN only
hard-cluster selection → VAL only
Expert early stopping  → VAL only
TEST                    → final evaluation only
old assignments.csv     → not used
```

默认还启用：

```python
USE_DISJOINT_VALIDATION_ROLES = True
DIFFICULTY_SELECTION_FRACTION = 0.50
```

即把每个 Router cluster 内的 validation 样本拆成两部分：

```text
50% → difficulty selection
50% → Expert early stopping
```

避免同一批 VAL 样本同时承担“决定困难簇”和“选择最佳 epoch”两个角色。

---

## 3. 与 Base 模型保持一致的结构

程序继续使用你的 Base 配置：

```python
MODEL_DEPTH = 6
EDGE_INPUT_DIM = 80
TRIPLET_INPUT_DIM = 40
EMBED_DIM = 128
TEXT_PROJ_DIM = 64
TEXT_PROJ_HIDDEN_DIMS = (256, 128)
TEXT_PROJ_DROPOUT = 0.1
NUM_HEADS = 4
```

文本部分保持：

```text
MatSciBERT
mean pooling
precomputed text embeddings
freeze text encoder
```

Router 使用 graph-only `graph_out`；Expert prediction 继续使用 cached MatSciBERT embedding，因此不会把“去掉文本”与“Expert specialization”混在一起。

---

## 4. 默认 Expert 微调参数

主配置：

```python
FINETUNE_MODE = "last2"
TRAIN_TEXT_PROJ = False
```

在 depth=6 下对应：

```text
Embedding        frozen
Block1           frozen
Block2           frozen
Block3           frozen
Block4           frozen
Block5           trainable
Block6           trainable
Regression Head  trainable
Text projection  frozen
MatSciBERT        frozen
```

Python 层编号对应：

```text
blocks.4
blocks.5
regression_head
```

优化参数默认：

```python
BATCH_SIZE = 64
FINE_TUNE_EPOCHS = 30
ABSOLUTE_LR = 1e-5
MINIMUM_LR = 1e-7
WARMUP_RATIO = 0.05
REGRESSION_HEAD_LR_SCALE = 5.0
EARLY_STOPPING_PATIENCE = 8
LOSS_NAME = "mse"
```

MSE 保持与 Base 一致，用于控制变量。

---

# 5. v2.1 新增：KMeans / Router 数值分析

程序会在：

```text
<run>/router/
```

新增以下文件。

### `kmeans_quality_metrics.json`

只基于 TRAIN Router features 计算：

```text
K
full-train inertia
sample silhouette
sample Calinski-Harabasz
sample Davies-Bouldin
```

默认：

```python
ROUTER_QUALITY_SAMPLE_SIZE = 5000
```

Silhouette 为避免对十万级样本做昂贵的 O(N²) 距离计算，默认只在固定随机种子抽取的 5000 个 TRAIN 样本上计算。

这些指标用于 **诊断正式 train-only Router 的聚类质量**，不是重新选择 K。

---

### `router_cluster_statistics.csv`

按 TRAIN / VAL、R0~R4 输出：

```text
sample count
sample fraction
mean / median centroid distance
mean / median routing confidence
mean / std / median e_form
Q25 / Q75 e_form
min / max e_form
```

如果 loader 无法暴露 original dataset index，TRAIN 的 `e_form` 统计可能为空；VAL 的 `e_form` 始终可从 DataLoader prediction targets 获得。

---

### `centroid_pairwise_distances.csv`

保存 5 个 KMeans centers 之间的两两欧氏距离矩阵，用于判断不同 latent region 的中心间几何关系。

---

### Router sample-level diagnostics

```text
train_router_diagnostics.csv
val_router_diagnostics.csv
```

每个样本包含：

```text
cluster_id
expert_id
nearest_dist2
second_dist2
margin
confidence
e_form (when available)
```

这些结果不依赖 sample lookup 做路由，而是由 `EmbeddedKMeansRouter` 实际计算得到。

---

# 6. v2.1 新增：Router PCA 可视化

PCA **只用于可视化，不参与 KMeans 拟合和路由**。

默认：

```python
ROUTER_PCA_FIT_SAMPLE_SIZE = 30000
ROUTER_MAX_PLOT_POINTS = 30000
```

PCA 只在 TRAIN Router features 上 fit：

```text
TRAIN graph_out
   ↓
L2
   ↓
PCA.fit(TRAIN sample)
```

VAL 仅 transform。

输出：

```text
router/pca_visualization_train_only.joblib
router/pca_visualization_meta.json
router/kmeans_centers_pca2d.csv
router/train_pca_coordinates.csv
router/val_pca_coordinates.csv
```

`pca_visualization_meta.json` 会记录 PC1/PC2 explained variance ratio。

---

## 7. Router 图像输出

目录：

```text
router/figures/
```

### `router_train_pca_clusters.png`

TRAIN latent PCA，按 R0~R4 着色，并标记 KMeans centroids。

用途：检查 train-only KMeans 在 latent manifold 上是否仍形成局部连续区域。

### `router_val_pca_clusters.png`

把 VAL 投影到 TRAIN-fitted PCA，并使用 train-only KMeans 路由。

用途：检查 TRAIN Router 边界在 VAL 上是否具有连续性。

### `router_train_pca_eform.png`

如果可以通过 original indices 对齐 raw JSON，则绘制 TRAIN latent PCA 的 `e_form` 色图。

### `router_val_pca_eform.png`

VAL latent PCA 的 `e_form` 色图。

用途：重新验证正式无泄漏 Router 是否仍具有与 formation energy 相关的潜在梯度。

### `router_cluster_sizes.png`

TRAIN / VAL 每个 Router cluster 的样本量对比。

用途：发现过小、过度不平衡的 cluster。

### `router_val_eform_by_cluster_boxplot.png`

R0~R4 的 validation `e_form` 分布。

用途：判断新 train-only Router 的 property-region 分层。

### `validation_cluster_difficulty.png`

直接绘制用于分配 Expert 的 VAL Base difficulty。

默认 difficulty metric：

```python
DIFFICULTY_METRIC = "mae"
```

因此这张图可以直接看出：

```text
哪个 R cluster → Expert-B
哪个 R cluster → Expert-C
哪些 cluster → Expert-A
```

### `router_distance_by_cluster.png`

TRAIN 每个 cluster 的 centroid squared-distance 箱线图。

### `router_confidence_by_cluster.png`

TRAIN 每个 cluster 的 routing confidence 箱线图。

### `router_train_pca_confidence.png`

在 latent PCA 上显示 routing confidence 空间分布。

### `router_train_pca_centroid_distance.png`

在 latent PCA 上显示 centroid distance。

这些图可以继续承接之前“距离 cluster center 越远，Base prediction risk 越高”的误差分析。

---

# 8. v2.1 新增：Expert 学习曲线

每个：

```text
expert_A_shared3/
expert_B_hardest/
expert_C_second_hardest/
global_ft_control/
```

目录下都会生成：

```text
figures/
├── mae_curve.png
├── rmse_curve.png
├── mse_curve.png
└── lr_curve.png
```

同时继续保存：

```text
epoch_metrics.csv
trainable_parameters.json
best_result.json
checkpoints/best_model.pt
tensorboard/
```

学习曲线用于检查：

```text
train/val gap
过拟合
收敛速度
最佳 epoch
不同 Expert 是否具有不同优化难度
```

---

# 9. v2.1 新增：最终 Base / Global-FT / Expert 可视化

最终数值结果继续保存：

```text
final_evaluation/
├── overall_comparison.csv
├── per_cluster_comparison.csv
├── per_cluster_improvement.csv
├── per_expert_group_comparison.csv
├── catastrophic_error_comparison.csv
├── sample_predictions_paired.csv
└── test_routes_from_embedded_router.csv
```

图像目录：

```text
final_evaluation/figures/
```

新增图像如下。

### 总体表现

```text
overall_mae_comparison.png
overall_rmse_comparison.png
overall_abs_error_boxplot.png
```

直接比较：

```text
Base
Global-FT
Routed-3Expert
```

---

### Cluster-wise 表现

```text
per_cluster_mae_comparison.png
per_cluster_rmse_comparison.png
per_cluster_p95_comparison.png
per_cluster_p99_comparison.png
per_cluster_mae_improvement.png
```

其中：

```text
per_cluster_mae_improvement.png
```

绘制：

```text
Base MAE - Routed Expert MAE
```

所以：

```text
> 0  → Expert 改善
< 0  → Expert 发生负迁移
```

它尤其适合判断共享 Expert-A 中的三个 cluster 是否都获益。

---

### Expert-group 级别

```text
per_expert_group_mae_comparison.png
```

比较：

```text
Expert-A shared3
Expert-B hardest
Expert-C second-hardest
```

各自对应区域的 Base vs Routed-3Expert MAE。

---

### 长尾 / catastrophic error

```text
catastrophic_error_rate_comparison.png
```

默认阈值：

```python
CATASTROPHIC_ERROR_THRESHOLDS = [0.1, 0.2, 0.5, 1.0]
```

比较三个模型中：

```text
|error| >= 0.1
|error| >= 0.2
|error| >= 0.5
|error| >= 1.0
```

的 test sample proportion。

这对于判断 Expert-B 是否真正压缩 catastrophic tail 很重要。

---

### True vs Predicted

分别生成：

```text
true_vs_pred_base.png
true_vs_pred_global_ft.png       # 如果 Global-FT 开启
true_vs_pred_routed_3expert.png
```

---

### 样本级 Base → Expert 改善

```text
paired_abs_error_base_vs_routed.png
```

横轴：

```text
Base |error|
```

纵轴：

```text
Routed Expert |error|
```

参考线：

```text
y = x
```

因此：

```text
点在 y < x → Expert 改善
点在 y > x → Expert 退化
```

这张图可以直接回答：

> Expert 是否真正修正了原来 Base 的高误差样本，而不是只改善总体平均值。

---

### TEST Router diagnostics

最终评估阶段还会生成：

```text
test_router_confidence_by_cluster.png
test_router_distance_by_cluster.png
```

这些 TEST 图只用于最终结果描述，不参与任何模型选择。

---

# 10. 关键参数区

运行前重点检查：

```python
DATA_PATH
BASE_WEIGHTS_PATH
TEXT_MODEL_NAME
TEXT_EMBEDDING_CACHE_PATH
```

正式运行推荐显式设置 Base 权重：

```python
BASE_WEIGHTS_PATH = (
    "./runs/fusiformer_production_matscibert_mlp64/"
    "YOUR_BASE_RUN/checkpoints/best_model_weights.pt"
)
```

不要依赖自动 newest lookup 完成最终论文实验。

---

# 11. Router diagnostics 参数

可调：

```python
ROUTER_DIAGNOSTICS_ENABLED = True
ROUTER_QUALITY_SAMPLE_SIZE = 5000
ROUTER_PCA_FIT_SAMPLE_SIZE = 30000
ROUTER_MAX_PLOT_POINTS = 30000
ROUTER_DIAGNOSTICS_RANDOM_STATE = 20260910
ROUTER_SAVE_PCA_COORDINATES = True
```

如果内存不足，可优先减少：

```text
ROUTER_PCA_FIT_SAMPLE_SIZE
ROUTER_MAX_PLOT_POINTS
ROUTER_QUALITY_SAMPLE_SIZE
```

这些只影响分析速度与图像点数，不改变 KMeans Router 本身。

---

# 12. 首次服务器运行

放置到项目根目录：

```text
~/llm_fusiformer/
├── grouped3_expert_fusiformer_v2_1_diagnostics.py
├── loader/
├── models/
├── data/
└── runs/
```

进入环境：

```bash
cd ~/llm_fusiformer
conda activate fusiformer
```

先检查：

```bash
python -m py_compile grouped3_expert_fusiformer_v2_1_diagnostics.py
```

第一次推荐：

```python
SMOKE_TEST = True
```

然后：

```bash
python -u grouped3_expert_fusiformer_v2_1_diagnostics.py
```

Smoke test 仍会产生 Router diagnostics 和 Expert 学习曲线，但 Expert 只训练 2 epochs，并跳过 Global-FT。

确认完整流程后：

```python
SMOKE_TEST = False
```

正式运行：

```bash
nohup python -u grouped3_expert_fusiformer_v2_1_diagnostics.py \
  > grouped3_expert_v2_1.nohup.log 2>&1 &
```

查看：

```bash
tail -f grouped3_expert_v2_1.nohup.log
```

---

# 13. 输出目录

默认：

```text
runs/
└── grouped3_expert_embedded_router_v2_1_diagnostics_matscibert/
    ├── router_embedding_cache/
    │   ├── train_graph_out.npy
    │   ├── val_graph_out.npy
    │   └── test_graph_out.npy
    │
    └── <timestamp>/
        ├── train.log
        ├── experiment_config_and_summary.json
        ├── training_summaries.json
        │
        ├── router/
        │   ├── kmeans_train_only_diagnostic.joblib
        │   ├── kmeans_centers.csv
        │   ├── kmeans_quality_metrics.json
        │   ├── router_cluster_statistics.csv
        │   ├── centroid_pairwise_distances.csv
        │   ├── train_router_diagnostics.csv
        │   ├── val_router_diagnostics.csv
        │   ├── validation_cluster_difficulty.csv
        │   ├── difficulty_aware_grouping.json
        │   ├── embedded_router_state.pt
        │   ├── pca_visualization_train_only.joblib
        │   ├── pca_visualization_meta.json
        │   ├── train_pca_coordinates.csv
        │   ├── val_pca_coordinates.csv
        │   └── figures/
        │       └── ... Router figures ...
        │
        ├── expert_A_shared3/
        │   ├── epoch_metrics.csv
        │   ├── figures/
        │   └── checkpoints/
        │
        ├── expert_B_hardest/
        │   └── ...
        │
        ├── expert_C_second_hardest/
        │   └── ...
        │
        ├── global_ft_control/
        │   └── ...
        │
        ├── final_evaluation/
        │   ├── overall_comparison.csv
        │   ├── per_cluster_comparison.csv
        │   ├── per_cluster_improvement.csv
        │   ├── per_expert_group_comparison.csv
        │   ├── catastrophic_error_comparison.csv
        │   ├── sample_predictions_paired.csv
        │   └── figures/
        │       └── ... final figures ...
        │
        └── deployment/
            └── grouped3_expert_embedded_router_bundle.pt
```

---

# 14. 正式结果建议优先上传的文件

Router 阶段首先看：

```text
router/kmeans_quality_metrics.json
router/router_cluster_statistics.csv
router/validation_cluster_difficulty.csv
router/difficulty_aware_grouping.json
router/figures/router_train_pca_clusters.png
router/figures/router_val_pca_eform.png
router/figures/validation_cluster_difficulty.png
```

训练阶段：

```text
expert_A_shared3/epoch_metrics.csv
expert_B_hardest/epoch_metrics.csv
expert_C_second_hardest/epoch_metrics.csv
```

最终：

```text
final_evaluation/overall_comparison.csv
final_evaluation/per_cluster_improvement.csv
final_evaluation/per_expert_group_comparison.csv
final_evaluation/catastrophic_error_comparison.csv
```

图像重点：

```text
per_cluster_mae_improvement.png
per_cluster_p95_comparison.png
per_cluster_p99_comparison.png
catastrophic_error_rate_comparison.png
paired_abs_error_base_vs_routed.png
```

---

# 15. 结果判定

第一层：

```text
Routed-3Expert MAE < Base MAE ?
```

第二层：

```text
Routed-3Expert MAE < Global-FT MAE ?
```

第三层：

```text
Hardest cluster 是否明显改善？
Second-hardest cluster 是否明显改善？
Shared Expert-A 的三个 cluster 是否没有明显负迁移？
```

第四层：

```text
P95 / P99 / catastrophic error rate 是否下降？
```

第五层：

```text
paired_abs_error_base_vs_routed 中，原 Base 高误差样本是否大部分落到 y<x？
```

如果这些条件同时成立，才可以比较有力地支持：

> Fusiformer latent space 中的局部预测困难度具有结构性差异，而 difficulty-aware grouped Expert specialization 能够提供超出普通 continued fine-tuning 的收益。

---

# 16. 注意事项

1. 当前 `G024/G1/G3` 的历史结论只用于形成研究假设；v2.1 正式 Router 不读取旧 assignments。
2. 新 Router cluster 使用 `R0~R4`，编号本身无物理意义。
3. PCA 只用于展示，不能把二维 U-shape 等形状直接解释成物理坐标。
4. TEST Router 图和 TEST error 图属于最终 evaluation diagnostics，不能根据这些图再返回修改 Expert grouping 并继续把同一 test 作为最终成绩。
5. 如果本轮结果用于架构开发，最终论文建议锁定架构后在 fresh split / independent test 上做 confirmatory run。
