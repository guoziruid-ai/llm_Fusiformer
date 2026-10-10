# Fusiformer 原始图空间 vs 形成能聚类实验 README

## 1. 实验目的

本阶段只研究一个问题：

> **晶体原子图结构形成的无监督聚类，与形成能 `e_form` 的无监督聚类之间是否存在稳定、显著、可解释的对应关系？**

本实验不加载任何已经训练好的 Fusiformer 模型，也不加载 checkpoint。程序只沿用当前项目中的晶体原始图构建方式，并把原始图中的确定性物理信息编码为 fixed graph embedding，再进行无监督聚类。

当前数据集共有 144,484 个晶体，`e_form` 对全部样本均为有效数值，因此本阶段不需要因为形成能缺失而删除样本。实际进入分析的样本数只可能因为个别晶体构图失败而略有减少。

---

## 2. 程序文件

主程序：

```text
fusiformer_eform_clustering_experiments.py
```

运行：

```bash
python fusiformer_eform_clustering_experiments.py
```

运行前首先修改：

```python
DATA_PATH = Path("data/my_dataset.json")
```

例如 Windows 下可以使用：

```python
DATA_PATH = Path(r"E:\Python项目\llm_Fusiformer\llm_Fusiformer-master\dataset.json")
```

输出目录默认：

```python
OUTPUT_DIR = Path("runs/eform_clustering_experiments")
```

---

# 3. 三类实验的区别

程序将实验明确拆成三种。

## 3.1 Experiment A：两个空间分别独立选择最佳 K

设置：

```python
RUN_EXPERIMENT_A = True
```

默认：

```python
EXPERIMENT_A_GRAPH_K = None
EXPERIMENT_A_EFORM_K = None
```

`None` 表示：

- 图空间在 `K_MIN ~ K_MAX` 中独立寻找最佳 K；
- 形成能空间也在 `K_MIN ~ K_MAX` 中独立寻找最佳 K；
- 两个 K **不要求相同**。

例如最终可能得到：

```text
Graph best K = 6
Formation-energy best K = 4
```

然后比较：

```text
Graph clusters (6 clusters)
        VS
Formation-energy clusters (4 clusters)
```

ARI、NMI、AMI 都允许两个聚类的簇数不同。

### Experiment A 回答的问题

> 图结构空间和形成能空间各自最自然的聚类尺度是什么？在各自最自然的尺度下，两种划分之间是否仍然具有对应关系？

因此 Experiment A 是本研究中最接近“自然聚类分析”的实验。

### 如果想人为固定 Experiment A 的 K

例如：

```python
EXPERIMENT_A_GRAPH_K = 6
EXPERIMENT_A_EFORM_K = 4
```

此时程序不再自动选 K。

通常建议保持：

```python
EXPERIMENT_A_GRAPH_K = None
EXPERIMENT_A_EFORM_K = None
```

---

## 3.2 Experiment B：两个空间固定相同 K

设置：

```python
RUN_EXPERIMENT_B = True
EXPERIMENT_B_SHARED_K = 4
```

程序会强制：

```text
Graph K = 4
Formation-energy K = 4
```

然后比较两种四分类结果。

### Experiment B 回答的问题

> 当图空间和形成能空间使用完全相同的聚类粒度时，它们的样本划分具有多大程度的对应？

Experiment B 更适合：

- contingency heatmap；
- Hungarian matching；
- cluster-to-cluster center comparison；
- 图簇形成能分布；
- 直接展示 `G0 -> E2`、`G1 -> E0` 等对应关系。

### `EXPERIMENT_B_SHARED_K` 应该如何确定

不建议只因为 Cross-K 中某个 K 的 ARI 最大，就直接宣布该 K 是“真实最佳 K”。这会产生明显的结果导向选择。

更合理的依据是综合：

1. Experiment A 得到的图空间最佳 K；
2. Experiment A 得到的形成能空间最佳 K；
3. Cross-K 中指标是否在某个 K 区间形成稳定高值或平台；
4. Silhouette / CH / DB 的聚类质量；
5. 每个簇的样本量是否足够；
6. 形成能簇中心是否具有明确物理解释；
7. 图簇是否存在明显极小簇。

例如，如果：

```text
Experiment A:
Graph best K = 5
E_form best K = 4

Cross-K:
K=4、5 的 ARI/NMI 接近，且 K=4 更易解释
```

可以把正式 Experiment B 设为：

```python
EXPERIMENT_B_SHARED_K = 4
```

然后把 K=5 作为敏感性结果保留。

---

## 3.3 Cross-K Scan：相同 K 的连续扫描

设置：

```python
RUN_CROSS_K_SCAN = True
CROSS_K_VALUES = list(range(2, 13))
```

等价于依次做：

```text
K = 2
K = 3
K = 4
...
K = 12
```

每一次都执行：

```text
Graph -> K clusters
E_form -> K clusters
```

然后计算：

```text
ARI
NMI
AMI
Hungarian overlap
Graph silhouette
Graph Calinski-Harabasz
Graph Davies-Bouldin
E_form silhouette
E_form Calinski-Harabasz
E_form Davies-Bouldin
```

### Cross-K 回答的问题

> 结构空间与形成能空间的对应关系是只在某一个 K 上偶然出现，还是在一段聚类尺度范围内都稳定存在？

它属于**敏感性/尺度分析**，而不是用来代替 Experiment A 的独立 K 选择。

最重要输出：

```text
cross_K_scan/cross_k_metrics.csv
cross_K_scan/correspondence_vs_k.png
```

---

# 4. 推荐的实际实验顺序

## 阶段 1：探索自然 K + 跨 K 稳定性

推荐先这样设置：

```python
RUN_EXPERIMENT_A = True
RUN_EXPERIMENT_B = False
RUN_CROSS_K_SCAN = True

K_MIN = 2
K_MAX = 12
CROSS_K_VALUES = list(range(2, 13))

EXPERIMENT_A_GRAPH_K = None
EXPERIMENT_A_EFORM_K = None
```

先观察：

```text
experiment_A_independent_K/experiment_A_selected_K.json
experiment_A_independent_K/graph_k_selection.csv
experiment_A_independent_K/eform_k_selection.csv
cross_K_scan/cross_k_metrics.csv
```

这一轮主要用来回答：

- 图空间自然 K 是多少；
- 形成能空间自然 K 是多少；
- K=2~12 时结构-形成能对应关系是否稳定。

---

## 阶段 2：选择一个代表性 K 做正式同尺度比较

假设最终决定用：

```text
K = 4
```

则设置：

```python
RUN_EXPERIMENT_A = False
RUN_EXPERIMENT_B = True
RUN_CROSS_K_SCAN = False

EXPERIMENT_B_SHARED_K = 4
```

此时重点查看：

```text
experiment_B_same_K_04/
```

中的：

```text
correspondence_metrics.csv
graph_vs_eform_contingency.csv
contingency_heatmap.png
graph_to_eform_row_fraction.csv
centroid_distance_standardized.csv
centroid_distance_heatmap.png
graph_cluster_eform_statistics.csv
eform_cluster_centers.csv
assignments.csv
e_form_by_graph_cluster_boxplot.png
```

---

## 阶段 3：最终精确复核

探索阶段推荐：

```python
CLUSTER_ALGORITHM = "minibatch"
```

如果已经选定最终 K，可以把 Experiment B 改为：

```python
CLUSTER_ALGORITHM = "kmeans"
RUN_EXPERIMENT_A = False
RUN_EXPERIMENT_B = True
RUN_CROSS_K_SCAN = False
```

这样只对最终 K 使用完整 KMeans 做一次精确复核。

---

# 5. 图构建参数

当前程序保持 Fusiformer 原始图定义：

```python
GRAPH_CUTOFF = 8.0
GRAPH_MAX_NEIGHBORS = 12
USE_CANONIZE = True
ATOM_Z_MAX = 95
DIST_RBF_BINS = 80
ANGLE_RBF_BINS = 40
```

## `GRAPH_CUTOFF = 8.0`

表示构图时考虑的邻域截断尺度。

当前阶段不要为了提高聚类结果而随意修改，因为实验目的不是寻找“最优图构建参数”，而是分析：

> Fusiformer 当前采用的原始图表示是否天然包含与形成能相关的聚类结构。

后续如果需要做图构建敏感性分析，可以单独比较：

```text
cutoff = 6 Å
cutoff = 8 Å
cutoff = 10 Å
```

但不要与第一阶段主实验混合。

## `GRAPH_MAX_NEIGHBORS = 12`

表示邻居数设置。

同样建议当前保持 12。

## `DIST_RBF_BINS = 80`

距离 RBF 描述维度。

## `ANGLE_RBF_BINS = 40`

键角余弦 RBF 描述维度。

两者当前都建议固定，不作为 KMeans 超参数调整。

---

# 6. 固定图 embedding 的含义

程序不是把晶体送入一个随机初始化的 Fusiformer 网络。

它从原始图中构造确定性的描述符，包括：

```text
原子序数组成分数
距离 RBF mean/std
角度 RBF mean/std
节点数
边数
平均度
距离统计
角度统计
元素种类数
组成熵
...
```

因此：

```text
同一个晶体 + 同一套参数
```

总会得到相同 embedding。

这非常重要，因为本实验研究的是原始图结构的自然聚类，而不是训练监督已经改变过的隐空间。

---

# 7. 图空间标准化与 PCA

完整流程：

```text
Raw graph descriptor
       ↓
StandardScaler
       ↓
PCA
       ↓
KMeans / MiniBatchKMeans
```

## 为什么必须 StandardScaler

图描述符中同时存在：

```text
composition fraction
RBF values
num_nodes
num_edges
distance statistics
...
```

不同维度尺度不同。如果不标准化，数值范围大的特征可能不合理地主导欧氏距离。

程序固定启用 StandardScaler。

---

## `GRAPH_PCA_COMPONENTS`

默认：

```python
GRAPH_PCA_COMPONENTS = 0.95
```

表示：

> PCA 自动保留能够解释 95% 累计方差的主成分。

这是推荐主实验。

也可以设置：

```python
GRAPH_PCA_COMPONENTS = 32
```

或：

```python
GRAPH_PCA_COMPONENTS = 64
```

用于 PCA 维度敏感性分析。

如果设置：

```python
GRAPH_PCA_COMPONENTS = None
```

表示标准化后直接在完整 fixed graph embedding 中聚类。

### 推荐

主实验：

```python
GRAPH_PCA_COMPONENTS = 0.95
```

稳健性分析：

```text
32
64
0.95
```

分别比较最终 ARI/NMI/AMI 是否基本稳定。

---

# 8. K 搜索范围

默认：

```python
K_MIN = 2
K_MAX = 12
```

即 Experiment A 搜索：

```text
2, 3, 4, ..., 12
```

为什么第一阶段推荐 2~12：

- K 太小只能看到非常粗粒度划分；
- K 太大容易出现大量小簇；
- 形成能是一维连续变量，切成几十个簇通常缺乏解释价值；
- 你的目标是寻找宏观结构-性质对应，而不是人为把数据切得非常细。

如果观察到 K=12 附近仍然明显没有达到稳定区间，可以再扩大为：

```python
K_MAX = 20
CROSS_K_VALUES = list(range(2, 21))
```

但第一轮不建议直接搜索到 30~50。

---

# 9. Experiment A 的 K 选择指标

参数：

```python
K_SELECTION_METRIC = "silhouette"
```

支持：

```python
"silhouette"
"calinski_harabasz"
"davies_bouldin"
```

## Silhouette Score

越大越好。

它衡量：

```text
簇内是否紧密
簇间是否分离
```

默认推荐：

```python
K_SELECTION_METRIC = "silhouette"
```

但不要只看一个指标做科学解释。

---

## Calinski-Harabasz Score

越大越好。

反映簇间离散度相对于簇内离散度的比例。

---

## Davies-Bouldin Score

越小越好。

反映不同簇之间的相似性。

---

## 推荐判断方式

程序会同时保存三种指标，而 `K_SELECTION_METRIC` 只负责自动确定一个 K。

例如：

```text
K  Silhouette  CH       DB
2  0.41        58000    1.02
3  0.38        63000    0.91
4  0.35        71000    0.82
```

如果 Silhouette 推荐 K=2，但 CH/DB、簇中心分布和物理解释更支持 K=4，则报告中应明确说明，而不是机械地认为 K=2 一定是唯一正确结果。

---

# 10. `K_SELECTION_SAMPLE`

默认：

```python
K_SELECTION_SAMPLE = 5000
```

这个参数只影响：

- Silhouette；
- Calinski-Harabasz；
- Davies-Bouldin；
- Experiment A 的候选 K 评价；
- Cross-K 中聚类质量的抽样估计。

它**不影响最终 KMeans 使用多少样本**。

最终聚类仍然使用全部有效晶体。

为什么不是直接对 144,484 个样本完整计算 Silhouette：

Silhouette 涉及大量样本间距离，完整计算代价很高。

推荐：

探索：

```python
K_SELECTION_SAMPLE = 5000
```

需要更高精度复核时可以：

```python
K_SELECTION_SAMPLE = 10000
```

通常没有必要为了第一阶段直接使用全部 144k 样本计算 Silhouette。

---

# 11. KMeans / MiniBatchKMeans

## `CLUSTER_ALGORITHM`

默认：

```python
CLUSTER_ALGORITHM = "minibatch"
```

适合你的 144,484 个晶体进行 K 扫描。

如果设置：

```python
CLUSTER_ALGORITHM = "kmeans"
```

则使用标准 KMeans。

### 推荐策略

探索 A + Cross-K：

```python
CLUSTER_ALGORITHM = "minibatch"
```

最终选定 K 后的 Experiment B 复核：

```python
CLUSTER_ALGORITHM = "kmeans"
```

---

## `KMEANS_N_INIT`

默认：

```python
KMEANS_N_INIT = 20
```

含义：使用多次不同初始化，最终保留较优结果。

比单次初始化更加稳定。

建议保持 20。

---

## `KMEANS_MAX_ITER`

默认：

```python
KMEANS_MAX_ITER = 500
```

通常足够。

如果模型已经收敛，提高到 1000 不会自动提高结果质量。

---

## `KMEANS_TOL`

默认：

```python
KMEANS_TOL = 1e-4
```

建议保持默认。

---

## `MINIBATCH_BATCH_SIZE`

默认：

```python
MINIBATCH_BATCH_SIZE = 4096
```

只在：

```python
CLUSTER_ALGORITHM = "minibatch"
```

时生效。

对于 144k 样本是较合理的起点。

---

# 12. 随机种子

默认：

```python
RANDOM_STATE = 42
```

用于：

- KMeans 初始化；
- MiniBatchKMeans；
- K 选择抽样；
- UMAP；
- permutation test。

正式报告必须记录该值，以保证实验可复现。

后续稳健性分析可以重复：

```text
0
10
20
30
42
```

检查结果是否严重依赖随机初始化。

---

# 13. 形成能空间如何聚类

形成能是一维变量：

```text
[e_form]
```

程序先做：

```text
StandardScaler
```

然后进行 KMeans。

因为一维 StandardScaler 只是线性缩放，所以不会改变形成能的排序结构，但可让数值尺度与算法配置保持一致。

最终结果会重新输出为 `e_form` 的原始单位，而不是只报告标准化坐标。

程序不会对形成能做 log 变换，也不会默认删除高形成能样本。

当前阶段建议保留完整数据分布。

---

# 14. Cross-K 的详细文件开关

默认：

```python
SAVE_DETAILED_FIXED_K_OUTPUT = False
```

原因是你的数据有约 144k 样本。

如果 K=2~12 每一个 K 都保存完整 assignments，则会生成 11 份大型 CSV。

默认 Cross-K 只保存：

```text
cross_k_metrics.csv
correspondence_vs_k.png
各类 clustering-quality 曲线
cross_k_peak_summary.json
```

如果确实想查看每个 K 的逐样本标签，再改为：

```python
SAVE_DETAILED_FIXED_K_OUTPUT = True
```

Experiment A 和 Experiment B 的详细 assignments 不受这个开关影响，始终保存。

---

# 15. 图 embedding 缓存

默认：

```python
REUSE_GRAPH_EMBEDDING_CACHE = True
```

第一次运行时程序会生成：

```text
graph_cache/raw_graph_embeddings.npz
```

以后只修改：

```text
Experiment A/B/Cross-K 开关
K
聚类算法参数
```

时，不需要重新从 144,484 个晶体构图。

如果你修改了以下参数：

```text
GRAPH_CUTOFF
GRAPH_MAX_NEIGHBORS
DIST_RBF_BINS
ANGLE_RBF_BINS
ATOM_Z_MAX
```

则必须删除旧的：

```text
graph_cache/raw_graph_embeddings.npz
```

或者设置：

```python
REUSE_GRAPH_EMBEDDING_CACHE = False
```

重新提取图描述符。

这是非常重要的实验一致性要求。

---

# 16. UMAP 参数

默认：

```python
USE_UMAP_IF_AVAILABLE = True
VIS_SAMPLE_MAX = 30000
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.1
```

UMAP **只用于可视化**。

正式聚类发生在：

```text
StandardScaler -> PCA space
```

而不是 UMAP 二维坐标中。

因此不要用二维 UMAP 图上看起来是否“分得开”来替代 ARI/NMI 等定量指标。

如果没有安装 `umap-learn`，程序会自动退回二维 PCA 可视化。

---

# 17. `VIS_SAMPLE_MAX`

默认：

```python
VIS_SAMPLE_MAX = 30000
```

只影响 UMAP/PCA2 可视化点数。

不影响聚类实际使用的样本数。

因此：

```text
Clustering: 全部有效样本
Visualization: 最多 30,000 个随机样本
```

这是正常设置。

---

# 18. 聚类对应性指标

## ARI

Adjusted Rand Index。

它关注样本对是否在两套划分中同时属于同簇或不同簇，并对随机一致性进行修正。

一般：

```text
接近 1：高度一致
接近 0：接近随机一致水平
```

不要把负的小数值简单解释为“负相关”，它表示一致程度低于某种随机期望。

---

## NMI

Normalized Mutual Information。

回答：

> 知道图簇标签以后，对形成能簇标签能获得多少信息？

通常范围为 0~1，越大表示关联更强。

---

## AMI

Adjusted Mutual Information。

是对随机一致性做过修正的信息论指标。

建议和 ARI 一起作为主要指标。

---

## Hungarian overlap

程序构建：

```text
Graph cluster × Formation-energy cluster
```

列联表，然后使用 Hungarian matching 找到最优一一对应。

例如：

```text
G0 -> E2
G1 -> E0
G2 -> E3
G3 -> E1
```

最后计算最优匹配后覆盖的样本比例。

这个指标非常直观，但它不像 ARI/AMI 那样完整修正随机一致性，因此应与 ARI/NMI/AMI 一起报告。

---

# 19. 簇中心对应分析

图簇中心和形成能簇中心不能直接在原始图 embedding 空间中比较，因为它们维度不同。

程序采用正确的方式：

1. 对每一个 Graph Cluster，计算其中样本的平均形成能；
2. 得到 Graph Cluster 在形成能空间中的中心；
3. 与 Formation-energy KMeans 的形成能中心比较；
4. 计算标准化形成能空间中的距离矩阵；
5. 再使用 Hungarian matching 找最佳中心对应。

输出：

```text
centroid_distance_standardized.csv
centroid_distance_heatmap.png
centroid_hungarian_matching.csv
```

它直接回答：

> 每个结构簇所对应的平均形成能位置，是否靠近形成能自身形成的某个自然簇中心？

---

# 20. 图簇形成能统计

Experiment A/B 会输出：

```text
graph_cluster_eform_statistics.csv
```

包含：

```text
Graph cluster
样本数
mean e_form
median e_form
std e_form
min e_form
max e_form
```

这张表非常重要。

即使 ARI 不是很高，只要不同 Graph Cluster 的形成能均值/分布明显不同，也可能说明图结构与形成能存在连续性的统计联系，只是这种关系未必表现为严格的一一离散簇对应。

因此不要把“ARI 不高”直接解释成“图结构与形成能完全无关”。

---

# 21. Permutation Test

默认：

```python
N_PERMUTATIONS = 0
```

表示关闭。

探索阶段建议关闭。

正式报告时可以：

```python
N_PERMUTATIONS = 100
```

或：

```python
N_PERMUTATIONS = 500
```

程序会随机打乱形成能聚类标签，建立随机对应的 null distribution，再比较真实：

```text
ARI
NMI
AMI
Hungarian overlap
```

是否显著高于随机水平。

Permutation test 当前主要用于 Experiment A/B 的正式结果，不建议在 Cross-K 的每个 K 上都开启大量置换，否则计算成本会明显增加，而且容易产生多重比较问题。

---

# 22. 推荐参数：第一轮正式探索

建议直接使用：

```python
PROPERTY_FIELD = "e_form"

RUN_EXPERIMENT_A = True
RUN_EXPERIMENT_B = False
RUN_CROSS_K_SCAN = True

K_MIN = 2
K_MAX = 12
CROSS_K_VALUES = list(range(2, 13))

EXPERIMENT_A_GRAPH_K = None
EXPERIMENT_A_EFORM_K = None
K_SELECTION_METRIC = "silhouette"
K_SELECTION_SAMPLE = 5000

GRAPH_PCA_COMPONENTS = 0.95

CLUSTER_ALGORITHM = "minibatch"
KMEANS_N_INIT = 20
KMEANS_MAX_ITER = 500
KMEANS_TOL = 1e-4
MINIBATCH_BATCH_SIZE = 4096

RANDOM_STATE = 42
N_PERMUTATIONS = 0

SAVE_DETAILED_FIXED_K_OUTPUT = False
```

这组参数的目的不是“追求最高 ARI”，而是先获得一套可解释、可复现的无监督基线。

---

# 23. 推荐参数：选定 K 后的正式 Experiment B

假设前一步综合判断选择：

```text
K = 4
```

则：

```python
RUN_EXPERIMENT_A = False
RUN_EXPERIMENT_B = True
RUN_CROSS_K_SCAN = False

EXPERIMENT_B_SHARED_K = 4

GRAPH_PCA_COMPONENTS = 0.95
CLUSTER_ALGORITHM = "kmeans"
KMEANS_N_INIT = 20
KMEANS_MAX_ITER = 500
RANDOM_STATE = 42

N_PERMUTATIONS = 100
```

此配置用于生成最终详细图表和显著性结果。

---

# 24. 推荐参数：PCA 稳健性实验

保持 K 与其他参数完全不变，只改变：

```python
GRAPH_PCA_COMPONENTS = 32
```

然后：

```python
GRAPH_PCA_COMPONENTS = 64
```

然后：

```python
GRAPH_PCA_COMPONENTS = 0.95
```

分别比较：

```text
ARI
NMI
AMI
Hungarian overlap
Graph cluster e_form statistics
```

如果结论相近，可说明对应关系不是由某一个人为 PCA 维度造成。

---

# 25. 推荐参数：随机种子稳健性实验

保持全部参数固定，只修改：

```python
RANDOM_STATE = 0
RANDOM_STATE = 10
RANDOM_STATE = 20
RANDOM_STATE = 30
RANDOM_STATE = 42
```

分别运行最终 Experiment B。

如果结果基本稳定，说明聚类不强依赖单次随机初始化。

---

# 26. 输出目录结构

默认大致为：

```text
runs/eform_clustering_experiments/
│
├── analysis_config.json
├── run_summary.json
├── main_experiment_metrics.csv
│
├── graph_cache/
│   ├── raw_graph_embeddings.npz
│   └── failed_graph_samples.csv        # 仅有失败样本时生成
│
├── graph_preprocess/
│   ├── graph_scaler.joblib
│   ├── graph_pca.joblib
│   ├── graph_pca_explained_variance.csv
│   └── graph_preprocess_summary.json
│
├── data_diagnostics/
│   ├── e_form_histogram.png
│   ├── e_form_percentiles.csv
│   ├── e_form_summary.json
│   └── e_form_scaler.joblib
│
├── experiment_A_independent_K/
│   ├── graph_k_selection.csv
│   ├── eform_k_selection.csv
│   ├── experiment_A_selected_K.json
│   ├── correspondence_metrics.csv
│   ├── graph_vs_eform_contingency.csv
│   ├── contingency_heatmap.png
│   ├── centroid_distance_standardized.csv
│   ├── graph_cluster_eform_statistics.csv
│   ├── eform_cluster_centers.csv
│   └── assignments.csv
│
├── experiment_B_same_K_04/
│   ├── correspondence_metrics.csv
│   ├── graph_vs_eform_contingency.csv
│   ├── contingency_heatmap.png
│   ├── graph_to_eform_row_fraction.csv
│   ├── centroid_distance_standardized.csv
│   ├── centroid_distance_heatmap.png
│   ├── graph_cluster_eform_statistics.csv
│   ├── eform_cluster_centers.csv
│   └── assignments.csv
│
└── cross_K_scan/
    ├── cross_k_metrics.csv
    ├── correspondence_vs_k.png
    ├── graph_silhouette_vs_k.png
    ├── graph_calinski_harabasz_vs_k.png
    ├── graph_davies_bouldin_vs_k.png
    ├── eform_silhouette_vs_k.png
    ├── eform_calinski_harabasz_vs_k.png
    ├── eform_davies_bouldin_vs_k.png
    └── cross_k_peak_summary.json
```

---

# 27. 最重要的结果应该如何阅读

## 第一步：先看 Experiment A

查看：

```text
experiment_A_selected_K.json
```

记录：

```text
Graph K*
E_form K*
```

然后查看：

```text
correspondence_metrics.csv
```

判断两个自然聚类尺度之间是否存在对应。

---

## 第二步：看 Cross-K

查看：

```text
cross_k_metrics.csv
correspondence_vs_k.png
```

重点不是单独寻找“最高的一个点”，而是观察：

- ARI/NMI/AMI 是否整体大于随机水平；
- 是否存在稳定高值区间；
- K 增大后指标是否快速崩溃；
- 某个 K 的高值是否只是孤立峰；
- Graph clustering quality 与 e_form clustering quality 是否同时合理。

---

## 第三步：确定 Experiment B 的代表性 K

选定 K 后，看：

```text
contingency_heatmap.png
graph_to_eform_row_fraction.csv
graph_cluster_eform_statistics.csv
centroid_distance_heatmap.png
```

如果出现：

```text
一个 Graph Cluster 主要映射到一个 Formation-energy Cluster
```

并且：

```text
不同 Graph Cluster 的 mean/median e_form 明显不同
```

则说明晶体图结构自然分区与形成能具有可解释的对应。

---

# 28. 关于“对应程度”不能只看一个指标

建议最终至少同时报告：

```text
ARI
NMI
AMI
Hungarian overlap
```

再配合：

```text
contingency heatmap
graph-cluster formation-energy statistics
centroid distance heatmap
boxplot
```

原因是可能存在下面这种情况：

```text
Graph cluster 1 -> 低形成能区
Graph cluster 2 -> 中低形成能区
Graph cluster 3 -> 中高形成能区
```

但形成能本身是连续变量，因此离散 KMeans 标签未必达到非常高的 ARI。

此时图簇之间形成能分布的统计差异仍然可能很有研究价值。

---

# 29. 不建议的实验设置

## 不建议 1：用 UMAP 2D 结果直接聚类

错误流程：

```text
Graph embedding -> UMAP 2D -> KMeans
```

当前正式实验不要这样做。

---

## 不建议 2：为了让 ARI 最大反复调 K

Cross-K 是诊断工具，不应该被当成“优化目标”。

否则实验会从：

```text
检验结构-形成能对应
```

变成：

```text
寻找最容易产生对应的超参数
```

两者科学意义不同。

---

## 不建议 3：把图空间 centroid 和形成能 centroid 直接做欧氏距离

两者维度不同，没有直接比较意义。

程序已经把 Graph Cluster 投影回形成能空间后再比较中心。

---

## 不建议 4：看到 ARI 低就直接否定结构-形成能关系

形成能是连续性质，结构与形成能之间可能是：

```text
连续趋势
多对一关系
一对多关系
局部关联
```

因此还要看图簇形成能分布、中心差异和热力图。

---

# 30. 当前推荐的主实验结论框架

最终可以把实验组织为：

### Experiment A：Natural-cluster correspondence

```text
Graph embedding -> independently selected K_G
E_form -> independently selected K_E
```

回答：两个空间各自自然分群是否一致。

### Experiment B：Same-resolution correspondence

```text
Graph K = E_form K = K_shared
```

回答：同一聚类粒度下是否存在清晰的一一或近一一映射。

### Cross-K：Resolution sensitivity

```text
K = 2...12
```

回答：对应关系是否跨聚类尺度稳定存在。

三者结合，比只选择一个 K 跑一次 KMeans 更完整，也更适合作为后续 Fusiformer 图模态解释性研究的无监督结构基线。
