# Fusiformer 潜在空间—形成能聚类分析 README

配套程序：

```text
fusiformer_latent_eform_clustering.py
```

本程序用于研究：

> **晶体经过已训练 Fusiformer 图编码器后，在模型潜在图表示空间中的聚类，与形成能 `e_form` 自身聚类之间具有多大程度的对应、重合与统计关联。**

当前阶段仅做 **聚类分析与表示空间分析**，不会训练、微调或修改 Fusiformer 权重。

---

# 1. 实验目标

本实验希望回答以下几个问题：

1. 晶体经过已经训练好的 Fusiformer 后，其图模态潜在表示 `graph_out` 是否形成清晰的自然簇？
2. Fusiformer 潜在空间的自然簇，与形成能 `e_form` 的自然簇是否具有对应关系？
3. 在相同聚类数 `K` 下，二者的一致程度如何随 `K` 改变？
4. 不同 Fusiformer 潜在簇的形成能分布是否显著不同？
5. 是否存在某些结构潜在簇，对低形成能、高形成能或某个形成能区间表现出显著富集？
6. 哪一个 `K` 更适合后续把 Fusiformer 拆分成若干 cluster-specific 子模型 / experts？

因此，本程序既用于当前的表示空间研究，也为后续：

```text
Fusiformer latent cluster
        ↓
Cluster Router
        ↓
Expert-0 / Expert-1 / Expert-2 / ...
        ↓
cluster-specific fine-tuning
```

提供分簇依据。

---

# 2. 与之前 raw graph descriptor 聚类程序的区别

之前的聚类实验使用的是人工固定结构描述符：

```text
Crystal
  ↓
Fusiformer-compatible graph construction
  ↓
composition histogram
distance RBF mean/std
angle RBF mean/std
graph statistics
  ↓
StandardScaler
  ↓
PCA
  ↓
KMeans
```

该方法研究的是：

> **不经过模型学习时，晶体原始结构统计特征与形成能之间是否存在天然聚类关系。**

而本程序使用的是：

```text
Crystal
  ↓
Fusiformer graph construction
  ↓
Atom embedding
  ↓
Edge / angle embedding
  ↓
Fusiformer Blocks
  ↓
AvgPooling
  ↓
graph_out
  ↓
L2 normalization
  ↓
KMeans
```

本程序研究的是：

> **经过 Base Fusiformer 表示学习以后，模型内部如何组织晶体，以及这种表示与形成能空间有多大程度的对应。**

两类实验应当保留为前后对照：

| 实验 | 表示空间 | 是否使用训练权重 | 研究目的 |
|---|---|---:|---|
| Raw structural baseline | 人工固定结构描述符 | 否 | 原始结构天然分布 |
| Fusiformer latent clustering | `graph_out` | 是 | 模型学习后的潜在空间 |

---

# 3. 程序提取的 Fusiformer 表示位置

程序不使用最终预测值作为 embedding。

它通过 forward hook 截取：

```python
model.readout
```

对应 Fusiformer 中：

```text
Fusiformer Blocks
        ↓
AvgPooling / readout
        ↓
graph_out
        ↓
regression_head
        ↓
e_form prediction
```

因此真正用于聚类的是：

```text
graph_out
```

而不是：

```text
prediction
```

在默认模型配置中，通常：

```text
graph_out.shape = [N, 128]
```

即每个晶体对应一个约 128 维的 Fusiformer 图模态潜在向量。

具体维度由 checkpoint 中模型配置决定，程序会自动尝试恢复。

---

# 4. 当前默认使用 graph-only 模式

程序默认：

```python
LATENT_MODE = "graph_only"
```

在推理阶段实际设置：

```python
inputs["text_mode"] = "bypass"
```

即：

```text
Crystal Graph
     ↓
Fusiformer graph branch
     ↓
graph_out
```

不输入 `description` 文本，也不加载 Qwen。

## 为什么当前推荐 graph-only？

因为后续计划将潜在簇作为 expert router。

对于一个未来未知晶体，最稳定的 router 输入应该是：

```text
crystal structure
```

而不是依赖：

```text
crystal structure + description text
```

因此当前建议保持：

```python
LATENT_MODE = "graph_only"
```

---

# 5. 可选：完整多模态潜在空间

程序也支持：

```python
LATENT_MODE = "normal_cached_text"
```

这时需要提供已经生成好的 Qwen pooled embedding：

```python
TEXT_EMBEDDING_CACHE_PATH = ...
```

例如：

```python
TEXT_EMBEDDING_CACHE_PATH = (
    PROJECT_ROOT
    / "data"
    / "qwen_cache"
    / "qwen2p5_1p5b_description_meanpool_f32.npy"
)
```

此模式表示：

```text
Crystal Graph + cached text embedding
                ↓
             Fusiformer
                ↓
            graph_out
```

注意：

- 不会现场加载 Qwen；
- 文本 embedding 数量必须与当前数据集严格对齐；
- 当前 cluster-router 研究阶段不建议优先使用此模式。

---

# 6. 文件放置位置

推荐把：

```text
fusiformer_latent_eform_clustering.py
```

直接放在 `llm_Fusiformer` 项目根目录，例如：

```text
llm_Fusiformer/
│
├── fusiformer_latent_eform_clustering.py
├── models/
├── loader/
├── data/
├── runs/
└── ...
```

这是因为程序直接导入：

```python
from loader.loader import CrystalGraphDataset
from loader.transforms import ...
from models.TextGuidedFusiformerCached import TextGuidedFusiformer
```

如果程序不在项目根目录，可能出现模块导入失败。

---

# 7. 运行环境

本程序需要你当前 Fusiformer 训练环境中的主要依赖，包括：

```text
Python
PyTorch
NumPy
Pandas
SciPy
scikit-learn
Matplotlib
DGL / JARVIS 等 Fusiformer 项目本身所需依赖
```

最推荐直接进入你训练 Fusiformer 的现有 Conda 环境运行，例如：

```bash
conda activate fusiformer
```

然后：

```bash
cd /path/to/llm_Fusiformer
python fusiformer_latent_eform_clustering.py
```

Windows 示例：

```bat
cd /d E:\Python项目\llm_Fusiformer\llm_Fusiformer-master
python fusiformer_latent_eform_clustering.py
```

Linux / AutoDL 示例：

```bash
cd ~/llm_fusiformer
python fusiformer_latent_eform_clustering.py
```

---

# 8. 必须修改的两个路径

运行前最重要的是修改：

```python
DATA_PATH
```

和：

```python
CHECKPOINT_PATH
```

---

## 8.1 DATA_PATH

默认：

```python
DATA_PATH = PROJECT_ROOT / "data" / "my_dataset.json"
```

应改为你的完整数据集路径。

例如：

```python
DATA_PATH = PROJECT_ROOT / "data" / "my_dataset_robocrys.json"
```

或者 Windows 绝对路径：

```python
DATA_PATH = Path(
    r"E:\Python项目\llm_Fusiformer\llm_Fusiformer-master\data\my_dataset.json"
)
```

数据集顶层应为 JSON list，例如：

```json
[
  {
    "mp_id": "mp-xxxx",
    "formula": "AB2",
    "atoms": {...},
    "e_form": -1.23,
    "description": "..."
  },
  ...
]
```

本程序至少要求每条数据含：

```text
atoms
e_form
```

推荐同时含：

```text
mp_id
formula
```

以方便输出 assignments。

---

## 8.2 CHECKPOINT_PATH

默认是占位路径：

```python
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "runs"
    / "fusiformer_production"
    / "CHANGE_ME"
    / "checkpoints"
    / "best_model_weights.pt"
)
```

必须替换为真实训练结果。

例如：

```python
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "runs"
    / "fusiformer_production"
    / "20260811-134939"
    / "checkpoints"
    / "best_model_weights.pt"
)
```

推荐优先使用：

```text
best_model_weights.pt
```

或者：

```text
best_checkpoint.pt
```

如果想研究训练最后一个 epoch 的模型表示，也可使用：

```text
last_checkpoint.pt
```

但正式实验一般优先使用最佳验证性能对应的 checkpoint。

---

# 9. checkpoint 的作用

本程序会加载 Base Fusiformer 的：

```text
atom embedding
edge embedding
angle embedding
Fusiformer Blocks
readout
...
```

训练权重。

但程序会：

```python
model.eval()
```

并在：

```python
torch.inference_mode()
```

下执行。

因此：

```text
不会反向传播
不会计算梯度
不会创建 optimizer
不会训练
不会修改 checkpoint
不会进行 fine-tuning
```

checkpoint 在这里只负责定义：

> **Fusiformer 已经学习得到的晶体潜在空间。**

---

# 10. 模型结构参数恢复

程序优先从 checkpoint 中读取：

```python
training_config
```

然后尝试自动推断：

```text
depth
embed_dim
edge_input_dim
triplet_input_dim
text_dim
```

相关配置。

程序顶部提供：

```python
DEPTH_OVERRIDE = None
NUM_HEADS_OVERRIDE = None
EDGE_INPUT_DIM_OVERRIDE = None
TRIPLET_INPUT_DIM_OVERRIDE = None
EMBED_DIM_OVERRIDE = None
TEXT_DIM_OVERRIDE = None
```

通常全部保持：

```python
None
```

即可。

只有当 checkpoint 与当前仓库代码结构不同、自动恢复失败时，才手动填写。

例如：

```python
DEPTH_OVERRIDE = 6
EMBED_DIM_OVERRIDE = 128
NUM_HEADS_OVERRIDE = 4
```

---

# 11. 图构建参数

当前程序默认：

```python
GRAPH_CUTOFF = 8.0
GRAPH_MAX_NEIGHBORS = 12
```

它们应尽可能与训练 Base Fusiformer 时保持一致。

当前阶段不建议为了提高聚类结果而修改这些值。

原因是本实验研究：

> **已经训练好的 Fusiformer 如何组织晶体。**

如果 checkpoint 是在：

```text
cutoff = 8 Å
max_neighbors = 12
```

下训练的，那么推理聚类也应该保持相同图构建规则。

否则会产生：

```text
训练时输入分布 != 聚类分析时输入分布
```

从而削弱实验解释。

---

# 12. 推理参数

默认：

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INFERENCE_BATCH_SIZE = 32
NUM_WORKERS = 4
PIN_MEMORY = True
RANDOM_SEED = 42
```

---

## 12.1 DEVICE

自动选择：

```text
GPU存在 → cuda
否则 → cpu
```

也可以强制：

```python
DEVICE = "cuda"
```

或：

```python
DEVICE = "cpu"
```

144,484 个晶体建议使用 GPU。

---

## 12.2 INFERENCE_BATCH_SIZE

默认：

```python
INFERENCE_BATCH_SIZE = 32
```

如果显存充足，可以尝试：

```python
64
```

或：

```python
128
```

如果出现 CUDA OOM：

```text
CUDA out of memory
```

则改小：

```python
16
```

甚至：

```python
8
```

此参数只影响 embedding 提取速度和显存占用，不应改变聚类结果。

---

## 12.3 NUM_WORKERS

默认：

```python
NUM_WORKERS = 4
```

Linux / AutoDL 可以尝试：

```python
8
```

Windows 如果出现 DataLoader multiprocessing 错误，可以改为：

```python
NUM_WORKERS = 0
```

这是 Windows 环境下最常见的兼容性设置。

---

# 13. graph_out embedding 缓存

第一次运行时，对 144,484 个晶体进行 Fusiformer 前向推理会花费较多时间。

程序会生成：

```text
runs/
└── fusiformer_latent_eform_clustering/
    └── embedding_cache/
        ├── fusiformer_graph_out.npy
        └── fusiformer_graph_out.meta.json
```

默认：

```python
REUSE_EMBEDDING_CACHE = True
SAVE_EMBEDDING_CACHE = True
```

第一次：

```text
dataset
  ↓
Fusiformer
  ↓
graph_out
  ↓
保存 cache
```

后续：

```text
读取 graph_out cache
  ↓
直接聚类
```

不再重新经过 Fusiformer。

这对于反复研究：

```text
不同 K
不同聚类算法
不同 preprocessing
Experiment A / B / Cross-K
```

非常重要。

---

# 14. cache 自动有效性检查

程序不会仅仅根据文件名判断 cache。

它还会检查 metadata，包括：

```text
数据集路径
数据集文件大小
数据集修改时间
checkpoint 路径
checkpoint 文件大小
checkpoint 修改时间
样本数
embedding dimension
LATENT_MODE
GRAPH_CUTOFF
GRAPH_MAX_NEIGHBORS
```

只有完全一致时才复用 cache。

因此更换：

```text
dataset
checkpoint
graph-only / multimodal mode
cutoff
max_neighbors
```

后，旧 cache 默认不会被错误复用。

---

# 15. 哪些参数改变后不需要重新提取 embedding？

已经有有效 cache 后，以下参数可直接修改：

```text
K_VALUES
MAIN_SHARED_K
CLUSTER_ALGORITHM
KMEANS_N_INIT
KMEANS_MAX_ITER
KMEANS_TOL
MINIBATCH_BATCH_SIZE
QUALITY_SAMPLE_SIZE
K_SELECTION_METRIC
LATENT_PREPROCESS
RUN_EXPERIMENT_A
RUN_CROSS_K_SCAN
RUN_MAIN_SHARED_K
VIS_SAMPLE_SIZE
N_PERMUTATIONS
```

因为这些操作都发生在：

```text
graph_out 提取之后
```

---

# 16. 哪些修改会导致重新提取 embedding？

修改下列内容后，通常应重新生成：

```text
DATA_PATH
CHECKPOINT_PATH
LATENT_MODE
TEXT_EMBEDDING_CACHE_PATH
GRAPH_CUTOFF
GRAPH_MAX_NEIGHBORS
模型结构 / checkpoint
```

因为它们会改变：

```text
graph_out
```

本身。

---

# 17. 潜在空间预处理：LATENT_PREPROCESS

默认：

```python
LATENT_PREPROCESS = "l2"
```

可选：

```text
l2
standardize
none
```

---

## 17.1 `l2` —— 当前主实验推荐

```python
LATENT_PREPROCESS = "l2"
```

执行：

\[
h_i'=\frac{h_i}{||h_i||_2}
\]

然后：

```text
L2-normalized graph_out
        ↓
KMeans
```

当前建议将其作为主实验。

原因是：

- `graph_out` 已是模型学习到的紧凑表示；
- L2 normalization 可以减弱单纯向量模长对欧氏距离的支配；
- 更强调 embedding 方向上的相似性；
- 与基于 cosine similarity 的表示比较具有较好的对应。

---

## 17.2 `none`

```python
LATENT_PREPROCESS = "none"
```

直接：

```text
raw graph_out
   ↓
KMeans
```

可作为敏感性实验。

---

## 17.3 `standardize`

```python
LATENT_PREPROCESS = "standardize"
```

对 128 个维度分别做：

\[
z_{ij}=
\frac{x_{ij}-\mu_j}{\sigma_j}
\]

然后聚类。

同样建议作为 robustness test，而非当前首选主实验。

---

# 18. 为什么主聚类不再使用 PCA？

旧的人工 descriptor 需要：

```text
StandardScaler
  ↓
PCA
  ↓
KMeans
```

本程序的 `graph_out` 本身已经是 Fusiformer 学习得到的 latent representation。

如果研究目标是：

> **尽可能保留 Fusiformer 内部真实表示空间**

那么当前主实验直接：

```text
graph_out
  ↓
L2 normalize
  ↓
KMeans
```

更合适。

所以程序明确：

```text
PCA 不参与正式 clustering
```

---

# 19. PCA 在本程序中的用途

PCA 只用于二维结果可视化：

```python
SAVE_LATENT_2D_PLOTS = True
```

最终生成：

```text
latent_pca2_colored_by_graph_cluster.png
latent_pca2_colored_by_eform.png
```

这些图片用于观察：

```text
Fusiformer latent distribution
```

但不用于产生 cluster labels。

即：

```text
正式聚类：
128D graph_out → KMeans

二维图：
128D graph_out → PCA-2D → visualization
```

不要根据二维 PCA 图重新聚类并替代正式结果。

---

# 20. 形成能空间处理

形成能：

```python
TARGET_FIELD = "e_form"
```

程序读取：

\[
e_{form}
\]

然后：

```python
StandardScaler()
```

得到标准化一维形成能：

\[
e'_i=
\frac{e_i-\mu}{\sigma}
\]

再使用 KMeans。

因为形成能只有 1 维，不需要 PCA。

形成能 cluster 会按照中心从低到高重新编号，因此：

```text
E0 = 最低形成能簇
E1 = 次低形成能簇
...
E(K-1) = 最高形成能簇
```

这一点便于后续解释。

---

# 21. 聚类算法

默认：

```python
CLUSTER_ALGORITHM = "minibatch"
```

可选：

```text
minibatch
kmeans
```

---

## 21.1 MiniBatchKMeans

```python
CLUSTER_ALGORITHM = "minibatch"
```

适合当前：

```text
N = 144,484
K = 2~12 扫描
```

优点：

- 速度快；
- 内存压力较小；
- 适合 Cross-K；
- 适合第一次探索。

默认：

```python
MINIBATCH_BATCH_SIZE = 4096
```

---

## 21.2 KMeans

最终确认某个 K 后可以改成：

```python
CLUSTER_ALGORITHM = "kmeans"
```

做标准 KMeans 复核。

推荐实验流程：

```text
探索阶段：
MiniBatchKMeans

最终 shared-K 复核：
KMeans
```

如果 MiniBatch 与完整 KMeans 结论一致，则更能说明结果稳定。

---

# 22. KMeans 参数

当前：

```python
KMEANS_N_INIT = 20
KMEANS_MAX_ITER = 500
KMEANS_TOL = 1e-4
```

---

## 22.1 KMEANS_N_INIT

```python
KMEANS_N_INIT = 20
```

代表使用多个初始化，降低局部最优和随机初始化影响。

正式科学分析不建议只设为：

```python
1
```

当前：

```text
20
```

是合理的稳定性设置。

---

## 22.2 KMEANS_MAX_ITER

```python
KMEANS_MAX_ITER = 500
```

允许聚类迭代最多 500 次。

一般不需要主动修改。

---

## 22.3 KMEANS_TOL

```python
KMEANS_TOL = 1e-4
```

控制收敛阈值。

建议保持默认。

---

# 23. 聚类 K 范围

当前：

```python
K_VALUES = list(range(2, 13))
```

即：

\[
K=2,3,\ldots,12
\]

注意：

> 之前 raw descriptor 实验中 `K=5` 的结果不能直接继承到 Fusiformer latent space。

因为现在 embedding 已经完全改变。

必须重新扫描：

```text
Fusiformer graph_out
        ↓
K=2~12
```

再选择适合后续 expert 划分的 K。

---

# 24. 程序包含三类实验

当前有：

```python
RUN_EXPERIMENT_A = True
RUN_CROSS_K_SCAN = True
RUN_MAIN_SHARED_K = True
MAIN_SHARED_K = 5
```

分别对应：

1. Experiment A：独立最佳 K
2. Cross-K：同 K 全范围扫描
3. Experiment B：指定 shared K 的详细分析

---

# 25. Experiment A：两个空间独立选择 K

开启：

```python
RUN_EXPERIMENT_A = True
```

Experiment A 的逻辑：

```text
Fusiformer latent
       ↓
K = 2~12
       ↓
根据聚类质量选 K_graph*

e_form
       ↓
K = 2~12
       ↓
根据聚类质量选 K_eform*
```

因此：

\[
K_{graph}
\]

和：

\[
K_{eform}
\]

可以不同。

它回答：

> **两个空间在各自最自然的聚类尺度下，是否形成相似的样本划分？**

---

# 26. Experiment A 如何选择最佳 K？

由：

```python
K_SELECTION_METRIC = "silhouette"
```

控制。

可选：

```text
silhouette
calinski_harabasz
davies_bouldin
```

---

## 26.1 Silhouette

越高越好。

大致衡量：

> 同簇样本是否更靠近，而不同簇是否更分离。

当前主推荐：

```python
K_SELECTION_METRIC = "silhouette"
```

但正式报告时不要只看一个指标。

---

## 26.2 Calinski-Harabasz

越高越好。

关注：

```text
簇间分离
vs
簇内紧凑
```

---

## 26.3 Davies-Bouldin

越低越好。

---

# 27. K 质量评价使用固定抽样

当前：

```python
QUALITY_SAMPLE_SIZE = 10000
```

非常重要：

> 聚类模型是在完整数据集上 fit 的。

即：

```text
144,484 samples
       ↓
KMeans / MiniBatchKMeans fit
```

只是 Silhouette / CH / DB 评价时抽：

```text
10,000 samples
```

这样可以避免 Silhouette 全样本计算过慢。

如果服务器资源充足，可改：

```python
QUALITY_SAMPLE_SIZE = 20000
```

如果内存压力较大，可改：

```python
5000
```

但不同 K 必须使用同一批固定抽样样本，这一点程序已经保证。

---

# 28. Cross-K Scan

开启：

```python
RUN_CROSS_K_SCAN = True
```

程序会对：

```text
K=2
K=3
...
K=12
```

逐个执行：

```text
Fusiformer graph_out KMeans(K)
e_form KMeans(K)
        ↓
correspondence metrics
```

最终得到：

```text
cross_k_metrics.csv
```

这是当前最重要的探索文件之一。

---

# 29. Cross-K 用于回答什么？

主要回答：

> **Fusiformer 潜在表示与形成能之间的对应关系，在什么聚类尺度上最明显？**

关注：

```text
ARI
NMI
AMI
Cramér's V
eta_squared_eform_by_graph_cluster
graph silhouette
cluster size
```

不要仅根据：

```text
Hungarian overlap
```

选择 K。

---

# 30. Experiment B：指定相同 K 的详细分析

开启：

```python
RUN_MAIN_SHARED_K = True
```

指定：

```python
MAIN_SHARED_K = 5
```

表示：

```text
Fusiformer latent → K=5
e_form            → K=5
```

然后做完整详细分析。

注意：

```python
MAIN_SHARED_K = 5
```

当前只是首次测试值。

正式结果应先跑 Cross-K，再根据 Fusiformer latent space 的真实结果决定：

```text
K=3 / 4 / 5 / ...
```

---

# 31. 只运行 Experiment B

如果已经拥有：

```text
embedding_cache/fusiformer_graph_out.npy
```

并且只想测试某个 K，例如：

```text
K=4
```

设置：

```python
RUN_EXPERIMENT_A = False
RUN_CROSS_K_SCAN = False

RUN_MAIN_SHARED_K = True
MAIN_SHARED_K = 4

REUSE_EMBEDDING_CACHE = True
```

此时程序只会：

```text
读取 graph_out cache
       ↓
latent preprocessing
       ↓
K=4 clustering
       ↓
详细分析
```

不会重新扫描 K=2~12。

---

# 32. 推荐的第一轮正式探索配置

建议：

```python
LATENT_MODE = "graph_only"

LATENT_PREPROCESS = "l2"

K_VALUES = list(range(2, 13))

CLUSTER_ALGORITHM = "minibatch"

KMEANS_N_INIT = 20
KMEANS_MAX_ITER = 500

QUALITY_SAMPLE_SIZE = 10000

RUN_EXPERIMENT_A = True
RUN_CROSS_K_SCAN = True

RUN_MAIN_SHARED_K = True
MAIN_SHARED_K = 5

N_PERMUTATIONS = 0
```

第一轮重点获得：

```text
cross_k_metrics.csv
Experiment A selected K
K=5 detailed analysis
```

然后再决定真正的 shared-K。

---

# 33. 推荐的第二轮 Experiment B 正式复核

假设第一轮显示：

```text
K=4
```

最值得研究。

建议：

```python
RUN_EXPERIMENT_A = False
RUN_CROSS_K_SCAN = False

RUN_MAIN_SHARED_K = True
MAIN_SHARED_K = 4

CLUSTER_ALGORITHM = "kmeans"

REUSE_EMBEDDING_CACHE = True
```

然后获得完整 K=4 结果。

可以再对：

```text
K=3
K=5
```

做邻近敏感性分析。

---

# 34. 输出目录

根目录：

```text
runs/
└── fusiformer_latent_eform_clustering/
```

embedding cache：

```text
embedding_cache/
├── fusiformer_graph_out.npy
└── fusiformer_graph_out.meta.json
```

每次聚类分析会创建时间戳目录，例如：

```text
20260905-183000/
```

---

# 35. 主目录输出

例如：

```text
runs/fusiformer_latent_eform_clustering/20260905-183000/
│
├── analysis_config.json
├── eform_distribution_summary.csv
├── cross_k_metrics.csv
├── correspondence_vs_k.png
├── silhouette_vs_k.png
│
├── experiment_A_independent_K/
│   ├── correspondence_metrics.csv
│   └── experiment_A_selected_K.json
│
└── experiment_B_same_K_05/
    ├── correspondence_metrics.csv
    ├── graph_vs_eform_contingency.csv
    ├── contingency_heatmap.png
    ├── graph_to_eform_row_fraction.csv
    ├── contingency_row_fraction.png
    ├── graph_to_eform_lift.csv
    ├── enrichment_lift_heatmap.png
    ├── graph_cluster_eform_statistics.csv
    ├── graph_cluster_eform_centers.csv
    ├── eform_cluster_centers.csv
    ├── centroid_distance_standardized.csv
    ├── centroid_distance_heatmap.png
    ├── centroid_hungarian_matching.csv
    ├── assignments.csv
    ├── e_form_by_graph_cluster_boxplot.png
    ├── latent_pca2_colored_by_graph_cluster.png
    └── latent_pca2_colored_by_eform.png
```

---

# 36. analysis_config.json

用于记录当前实验配置，包括：

```text
dataset
checkpoint
target
latent_mode
latent_preprocess
graph_cutoff
graph_max_neighbors
n_samples
embedding_shape
cluster_algorithm
k_values
quality_sample_size
k_selection_metric
main_shared_k
random_seed
model_info
```

正式实验建议保留该文件，因为它相当于：

> **该次聚类实验的可复现配置记录。**

---

# 37. eform_distribution_summary.csv

记录形成能：

```text
N
min
P1
P5
P25
median
mean
P75
P95
P99
max
std
```

用于判断：

- 是否长尾；
- 是否存在明显极端值；
- KMeans 是否可能受到高形成能少数样本影响。

---

# 38. cross_k_metrics.csv

这是 Cross-K 的核心表。

每一行对应一个 K，例如：

| K | graph_silhouette | eform_silhouette | ARI | NMI | AMI | cramers_v | eta² | min cluster fraction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | ... | ... | ... | ... | ... | ... | ... | ... |
| 3 | ... | ... | ... | ... | ... | ... | ... | ... |
| ... | | | | | | | | |

用于综合选择后续 expert 数量。

---

# 39. ARI

Adjusted Rand Index：

\[
ARI
\]

衡量：

> 两个聚类对样本对关系的一致程度。

大致：

```text
ARI ≈ 0
```

表示接近随机一致。

```text
ARI → 1
```

表示高度一致。

ARI 对 cluster label 编号交换不敏感。

对于：

> 图簇和形成能簇是否“一一重合”

ARI 是重要指标。

---

# 40. NMI

Normalized Mutual Information：

\[
NMI
\]

衡量：

> 知道 Fusiformer cluster 后，可以获得多少关于形成能 cluster 的信息。

通常：

```text
0 → 信息很少
1 → 完全一致
```

---

# 41. AMI

Adjusted Mutual Information：

\[
AMI
\]

是经过随机一致性修正的 mutual information 指标。

正式报告中建议与 ARI 一起优先使用。

---

# 42. Hungarian overlap

程序通过 Hungarian matching 寻找：

```text
G_i ↔ E_j
```

的一一最优配对，然后计算：

\[
Overlap=
\frac{\text{匹配后正确样本数}}{N}
\]

它直观，但：

> 不适合单独用来跨不同 K 选择最佳 K。

因为 cluster 数量与簇大小会影响 overlap 的基线。

---

# 43. Graph → e_form purity

计算：

> 每个 Fusiformer graph cluster 内选择占比最高的 e_form cluster 后，总体能覆盖多少样本。

同样适合 fixed-K 解释，不应作为唯一 K 选择标准。

---

# 44. Cramér's V

程序计算：

\[
V
\]

基于 contingency table 的总体类别关联。

它与 ARI 的关注点不同。

ARI 强调：

```text
两个 partition 是否相似
```

Cramér's V 强调：

```text
两个类别变量是否存在统计关联
```

因此可能出现：

```text
ARI不高
但 Cramér's V 明显大于0
```

这意味着：

> 图簇与形成能簇不是一一对应，但二者仍然存在显著多对一 / 多对多关联。

这对材料结构—性质分析很重要。

---

# 45. eta²：本实验非常重要的指标

程序计算：

```text
eta_squared_eform_by_graph_cluster
```

即：

\[
\eta^2=
\frac{SS_{between}}
{SS_{total}}
\]

表示：

> Fusiformer graph clusters 之间的形成能均值差异，占总形成能变异的比例。

例如：

```text
eta² = 0.40
```

可以理解为：

> graph cluster 分组与形成能之间存在较明显的组间效应。

但注意：

```text
eta² 不是模型预测 R²
```

不能说：

> “cluster 可以预测 40% 的形成能”。

更准确是：

> **约 40% 的形成能总变异可由这些簇之间的组均值差异描述。**

---

# 46. correspondence_vs_k.png

同时绘制：

```text
ARI
NMI
AMI
Cramér's V
Hungarian overlap
```

随 K 的变化。

用于寻找：

```text
稳定对应区间
```

而不是机械选择某一个最高点。

例如：

```text
K=4~6 指标整体形成平台
```

通常比：

```text
K=5 比 K=4 高 0.01
```

更值得解释。

---

# 47. silhouette_vs_k.png

比较：

```text
Fusiformer latent silhouette
e_form silhouette
```

随 K 的变化。

注意：

> 如果某个 K silhouette 很高，但 cluster 极不平衡，应谨慎解释。

因此必须结合：

```text
min_cluster_fraction
max_cluster_fraction
cluster_size_entropy_norm
```

一起判断。

---

# 48. cluster size 指标

Cross-K 输出包含：

```text
graph_min_cluster_size
graph_max_cluster_size
graph_min_cluster_fraction
graph_max_cluster_fraction
graph_cluster_size_entropy_norm
```

以及形成能对应指标。

对后续 expert 方案尤其重要。

例如：

```text
K=5
最小 cluster = 800 samples
```

即使 ARI 较高，也不一定适合直接训练 5 个完整 experts。

后续实际拆子模型时建议额外关注：

\[
N_{min}
\]

以及：

\[
\frac{N_{min}}{N}
\]

通常希望每个 expert 都拥有足够的训练样本。

---

# 49. graph_vs_eform_contingency.csv

原始计数矩阵：

```text
              E0     E1     E2
G0           ...
G1           ...
G2           ...
```

回答：

> 每一个潜在图簇中有多少晶体进入各形成能簇。

由于受到 cluster size 影响，不能单独看颜色深浅判断富集。

---

# 50. graph_to_eform_row_fraction.csv

将每个 Fusiformer cluster 行归一化：

\[
P(E_j|G_i)
\]

回答：

> 已知晶体属于 `G_i`，其属于各形成能簇的概率是多少？

这是详细分析中最重要的矩阵之一。

---

# 51. graph_to_eform_lift.csv

程序计算：

\[
Lift_{ij}
=
\frac{P(E_j|G_i)}
{P(E_j)}
\]

解释：

```text
Lift = 1
```

没有富集。

```text
Lift > 1
```

富集。

```text
Lift >> 1
```

强富集。

例如：

```text
G2 → E4 = 8.5
```

表示：

> G2 中出现 E4 的概率，是完整数据集中 E4 基线概率的 8.5 倍。

对于寻找：

```text
特殊高形成能潜在簇
```

非常有价值。

---

# 52. graph_cluster_eform_statistics.csv

每个 Fusiformer latent cluster 会输出：

```text
n
fraction
mean_e_form
median_e_form
std_e_form
min
q05
q25
q75
q95
max
```

这是判断：

> 不同潜在结构簇是否具有不同形成能分布

的核心文件。

例如如果：

```text
G0 mean = -2.1
G1 mean = -1.3
G2 mean = +0.7
G3 mean = +2.0
```

就说明即使 ARI 不是非常高，潜在 cluster 仍然具有明显形成能分层。

---

# 53. centroid distance

程序不会直接比较：

```text
128D graph centroid
```

和：

```text
1D e_form centroid
```

因为二者不在同一个空间。

正确做法是：

> 将每个 graph cluster 对应样本的平均形成能作为其在 e_form space 中的中心。

然后计算：

\[
D_{ij}
=
|\mu_{G_i}^{eform}-\mu_{E_j}^{eform}|
\]

输出：

```text
centroid_distance_standardized.csv
centroid_distance_heatmap.png
```

该结果用于判断：

> 某个 Fusiformer latent cluster 的平均形成能最接近哪个形成能簇中心。

---

# 54. centroid_hungarian_matching.csv

使用 centroid distance 做一一最优匹配。

注意：

> 一个真实材料问题中可能存在多个 graph clusters 对应同一个形成能区域。

因此 Hungarian matching 只是辅助分析，不代表真实关系必须一一对应。

---

# 55. assignments.csv

如果：

```python
SAVE_ASSIGNMENTS = True
```

程序保存：

```text
index
mp_id
formula
e_form
fusiformer_cluster
eform_cluster
```

这是后续非常重要的文件。

未来可以直接用于：

```text
按照 cluster 划分数据
分析每个 cluster 的公式组成
寻找代表晶体
训练 cluster-specific expert
```

因此正式实验建议保留。

---

# 56. e_form_by_graph_cluster_boxplot.png

比较：

```text
G0
G1
G2
...
```

各自形成能分布。

该图用于回答：

> Fusiformer 潜在空间不同簇是否对应不同的形成能水平或分布宽度。

如果 ARI 不高，但 boxplot 显著分离，说明关系可能是：

```text
连续性质分布差异
```

而不是：

```text
完全一一离散标签对应
```

---

# 57. latent_pca2_colored_by_graph_cluster.png

将 Fusiformer latent embedding：

```text
128D
 ↓
PCA 2D
```

只用于绘图。

颜色：

```text
fusiformer_cluster
```

用于观察 cluster 的二维投影分布。

---

# 58. latent_pca2_colored_by_eform.png

同样的二维坐标，但颜色改成：

```text
e_form
```

如果出现明显连续颜色梯度，说明：

> Fusiformer latent space 可能按照形成能方向连续组织，而不一定天然形成清晰的一一离散簇。

这一点对解释低 ARI + 高 eta² 的情况尤其重要。

---

# 59. N_PERMUTATIONS

默认：

```python
N_PERMUTATIONS = 0
```

表示关闭随机标签置换。

正式最终实验可以设置：

```python
N_PERMUTATIONS = 100
```

程序会随机打乱形成能 cluster labels，然后建立：

```text
random ARI
random NMI
random AMI
```

基线。

由于样本量约 144k，第一次探索不建议立即运行大量 permutation。

---

# 60. 当前阶段 K 的选择原则

后续目标不是单纯获得“最大的 ARI”，而是准备拆成若干 expert。

因此 K 应综合考虑四类因素：

## 60.1 Fusiformer latent cluster quality

关注：

```text
graph_silhouette
graph_calinski_harabasz
graph_davies_bouldin
```

---

## 60.2 Cluster balance

关注：

```text
min_cluster_size
min_cluster_fraction
cluster_size_entropy_norm
```

特别小的 cluster 不利于后续独立微调完整子模型。

---

## 60.3 Formation-energy association

关注：

```text
ARI
AMI
NMI
Cramér's V
eta²
Lift
```

---

## 60.4 后续 Expert 可训练性

例如：

```text
K=6
ARI最高
但两个 cluster 只有 500~1000 个样本
```

可能并不适合直接拆 6 个完整模型。

而：

```text
K=4
ARI略低
但所有 cluster > 10,000
```

可能更适合作为 expert 划分。

因此：

> **后续 Expert 数量不应只根据 ARI 最大值确定。**

---

# 61. 推荐的 K 选择表

建议最终整理：

| K | Graph Silhouette | ARI | AMI | Cramér V | eta² | min cluster | min fraction |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | | | | | | | |
| 3 | | | | | | | |
| 4 | | | | | | | |
| 5 | | | | | | | |
| ... | | | | | | | |

然后综合选择：

```text
3~5 个
```

规模稳定的候选 cluster 数。

---

# 62. 推荐实验顺序

## Stage 1：首次 embedding 提取 + 全 K 扫描

```python
RUN_EXPERIMENT_A = True
RUN_CROSS_K_SCAN = True

RUN_MAIN_SHARED_K = True
MAIN_SHARED_K = 5
```

目的：

```text
生成 graph_out cache
扫描 K
观察整体趋势
```

---

## Stage 2：选择候选 shared K

根据：

```text
Cross-K
cluster quality
cluster balance
ARI/AMI
Cramér V
eta²
```

例如选：

```text
K = 4
```

---

## Stage 3：正式 K=4 详细分析

```python
RUN_EXPERIMENT_A = False
RUN_CROSS_K_SCAN = False

RUN_MAIN_SHARED_K = True
MAIN_SHARED_K = 4

CLUSTER_ALGORITHM = "kmeans"
```

---

## Stage 4：邻近 K 稳健性

分别：

```text
K=3
K=4
K=5
```

比较：

```text
cluster membership
形成能分布
ARI/AMI
eta²
cluster size
```

---

## Stage 5：latent preprocessing 稳健性

主实验：

```python
LATENT_PREPROCESS = "l2"
```

再测试：

```python
LATENT_PREPROCESS = "none"
```

以及：

```python
LATENT_PREPROCESS = "standardize"
```

如果主要 cluster 结构和形成能关联仍保持，则结论更加可靠。

---

# 63. 后续 Expert 模型研究时的关键原则

当前程序暂时只做 clustering，但后续若进入子模型微调，需要注意：

> **聚类器必须只在训练集表示上 fit。**

正式 expert 实验不应：

```text
144k全部样本
  ↓
先cluster
  ↓
再split train/val/test
```

而应：

```text
先固定 train / val / test
        ↓
Base Fusiformer 在 train 上训练
        ↓
提取 train graph_out
        ↓
KMeans.fit(train)
        ↓
Val/Test 只 predict cluster
```

这是避免 test leakage 的关键。

当前全数据聚类分析的目标是：

> 研究整体潜在空间结构与确定后续候选 K。

它与最终 expert 训练的严格数据流程要区分。

---

# 64. 常见问题：程序提示 dataset not found

例如：

```text
FileNotFoundError: dataset not found
```

检查：

```python
DATA_PATH
```

是否为真实路径。

Windows 推荐：

```python
Path(r"E:\...\dataset.json")
```

---

# 65. checkpoint not found

检查：

```python
CHECKPOINT_PATH
```

特别注意：

```text
best_model_weights.pt
best_checkpoint.pt
```

实际文件名是否存在。

---

# 66. checkpoint/model mismatch

如果出现：

```text
Checkpoint/model mismatch
Missing keys
Unexpected keys
```

说明：

> 当前 `TextGuidedFusiformerCached.py` 的模型结构与保存 checkpoint 时使用的结构不完全一致。

首先确认：

```text
当前仓库版本
训练脚本版本
模型 depth
embed_dim
num_heads
```

是否一致。

必要时使用：

```python
DEPTH_OVERRIDE
NUM_HEADS_OVERRIDE
EMBED_DIM_OVERRIDE
...
```

手动指定。

---

# 67. Windows DataLoader 报错

优先改：

```python
NUM_WORKERS = 0
```

如果仍有问题再确认脚本由：

```python
if __name__ == "__main__":
    main()
```

保护；当前程序已经包含该结构。

---

# 68. CUDA OOM

降低：

```python
INFERENCE_BATCH_SIZE
```

例如：

```text
32 → 16 → 8
```

embedding 提取完成后聚类主要在 CPU / NumPy / sklearn 中进行，因此 GPU 显存压力主要来自 Fusiformer forward。

---

# 69. 已有 cache 但仍重新编码

程序会校验 cache metadata。

如果以下任一发生变化：

```text
dataset
checkpoint
latent mode
cutoff
max neighbors
```

程序会认为旧 cache 无效。

查看：

```text
fusiformer_graph_out.meta.json
```

即可确认具体配置。

---

# 70. 为什么形成能 KMeans 会把 cluster 重新排序？

普通 KMeans 的 cluster label：

```text
0,1,2,...
```

本身没有物理顺序。

为了方便解释，程序会按照形成能中心从低到高重新标记：

```text
E0 = lowest
E1 = ...
E(K-1) = highest
```

因此形成能簇具有明确物理顺序。

Fusiformer latent cluster 的：

```text
G0,G1,G2,...
```

仍然只是编号，没有高低意义。

---

# 71. 为什么 graph cluster 不按平均形成能重新编号？

因为 graph cluster 是根据：

```text
Fusiformer latent geometry
```

产生的。

如果根据形成能重新编号，会把 target information 人为引入 graph cluster 的命名过程。

因此：

```text
G0,G1,G2...
```

保持 KMeans 原始 cluster identity。

后续可以单独按照：

```text
mean_e_form
```

排序展示，但不要修改原始 cluster assignment。

---

# 72. 如何判断实验结果是否有价值？

不要只问：

> ARI 是否非常高？

材料结构与形成能通常并不是严格一一映射。

一个有价值的结果可能表现为：

```text
ARI 中等
AMI 中等
Cramér V 明显 > 0
eta² 较高
某些 G_i 对高/低能量簇显著富集
不同 G_i 形成能均值明显分离
```

这意味着：

> Fusiformer 潜在空间中存在明显 structure-property organization，但关系可能是多对一、多对多或连续梯度，而不是完全离散一一对应。

---

# 73. 对未来 Expert Router 最重要的结果

当前实验最终要帮助确定：

```text
多少个 cluster
每个 cluster 多大
cluster 是否稳定
不同 cluster 是否具有不同形成能映射
```

因此尤其关注：

```text
cluster size
eta²
Lift
Cramér V
每簇 formation-energy distribution
```

而不是只关注：

```text
ARI maximum
```

---

# 74. 建议最终保留的研究结论结构

最终可以按照三层结论组织：

### 结论 1：latent cluster 是否存在

依据：

```text
Silhouette
CH
DB
cluster balance
PCA visualization
```

### 结论 2：latent cluster 是否与 e_form 有关

依据：

```text
ARI
AMI
NMI
Cramér V
eta²
```

### 结论 3：具体哪些 cluster 对应哪些形成能区域

依据：

```text
row fraction
Lift
boxplot
cluster e_form statistics
centroid distance
```

这比只报告一个“重合率”更加完整。

---

# 75. 推荐第一轮实际配置汇总

```python
# Dataset / checkpoint
DATA_PATH = ...
CHECKPOINT_PATH = ...

# Latent extraction
LATENT_MODE = "graph_only"

GRAPH_CUTOFF = 8.0
GRAPH_MAX_NEIGHBORS = 12

DEVICE = "cuda"
INFERENCE_BATCH_SIZE = 32
NUM_WORKERS = 4

# Cache
REUSE_EMBEDDING_CACHE = True
SAVE_EMBEDDING_CACHE = True

# Latent preprocessing
LATENT_PREPROCESS = "l2"

# Clustering
K_VALUES = list(range(2, 13))
CLUSTER_ALGORITHM = "minibatch"

KMEANS_N_INIT = 20
KMEANS_MAX_ITER = 500
KMEANS_TOL = 1e-4
MINIBATCH_BATCH_SIZE = 4096

QUALITY_SAMPLE_SIZE = 10000
K_SELECTION_METRIC = "silhouette"

# Experiments
RUN_EXPERIMENT_A = True
RUN_CROSS_K_SCAN = True

RUN_MAIN_SHARED_K = True
MAIN_SHARED_K = 5

# Statistical robustness
N_PERMUTATIONS = 0
```

---

# 76. 推荐后续正式 Experiment B 配置

当 Cross-K 确定例如：

```text
K=4
```

后：

```python
RUN_EXPERIMENT_A = False
RUN_CROSS_K_SCAN = False

RUN_MAIN_SHARED_K = True
MAIN_SHARED_K = 4

REUSE_EMBEDDING_CACHE = True

CLUSTER_ALGORITHM = "kmeans"

KMEANS_N_INIT = 20
```

这样只读取现有 graph_out cache 并执行 K=4 的详细分析。

---

# 77. 当前实验的一句话定义

可以在报告中将本实验命名为：

> **Fusiformer 图模态潜在表示空间与形成能空间的无监督聚类对应性分析**

英文可写为：

> **Unsupervised Cluster Correspondence Analysis between Fusiformer Graph Latent Space and Formation-Energy Space**

---

# 78. 当前实验核心假设

可以定义：

\[
H_1:
\text{Fusiformer graph latent representations exhibit cluster structures associated with formation energy.}
\]

更具体：

\[
H_{1a}:
P(e_{form}|G_i)
\neq
P(e_{form}|G_j)
\]

即不同 Fusiformer latent clusters 的形成能分布不同。

以及：

\[
H_{1b}:
\text{Fusiformer latent clusters and e_form clusters are statistically dependent.}
\]

最后为未来 expert 模型提供假设：

\[
H_{1c}:
\text{cluster-specific local structure-property mappings may benefit from specialized fine-tuning.}
\]

---

# 79. 当前阶段不做的事情

本程序不会：

```text
训练 Fusiformer
修改 checkpoint
微调 expert
重新训练 Qwen
使用真实 e_form 作为 graph embedding 输入
根据形成能直接生成 graph cluster
```

形成能只用于：

```text
独立 clustering
cluster correspondence evaluation
post-hoc statistical analysis
```

因此 graph cluster 本身来自：

```text
Fusiformer graph_out latent geometry
```

---

# 80. 推荐保存的实验记录

每次正式实验建议记录：

```text
checkpoint 文件名
checkpoint epoch
dataset 版本
LATENT_MODE
LATENT_PREPROCESS
K
cluster algorithm
random seed
cross_k_metrics.csv
analysis_config.json
assignments.csv
```

这些信息后续在：

```text
Raw descriptor
vs
Base Fusiformer
vs
Expert Fusiformer
```

三阶段对照时都会用到。

---

## 总结

当前推荐主流程为：

```text
144,484 crystals
        ↓
Fusiformer-compatible crystal graph
        ↓
trained Base Fusiformer
        ↓
graph_out latent embedding
        ↓
L2 normalization
        ↓
KMeans / MiniBatchKMeans
        ↓
Fusiformer latent clusters
        │
        ├──────── compare ────────┐
        │                         │
        ↓                         ↓
cluster statistics         e_form KMeans
        │                         │
        └──────────┬──────────────┘
                   ↓
        ARI / NMI / AMI
          Cramér's V
              eta²
        Hungarian overlap
             Lift
       centroid distance
                   ↓
        select candidate K
                   ↓
     future Expert-Fusiformer
```

当前阶段首先完成：

1. `graph_out` cache；
2. K=2~12 Cross-K；
3. Experiment A；
4. 一个候选 shared-K 的详细分析；
5. 综合聚类质量、cluster balance、形成能关联程度选择后续 expert 数量。

后续再进入：

```text
cluster-specific fine-tuning
```

阶段。
