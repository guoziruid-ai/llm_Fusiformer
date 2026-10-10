#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fusiformer 原始图模态 vs 形成能（e_form）聚类对应性实验
===========================================================

本脚本专门用于当前阶段的 formation-energy 聚类研究：

- 不加载任何已训练 Fusiformer 模型或 checkpoint。
- 沿用项目原始晶体图构建方式：cutoff=8 Å、max_neighbors=12、原子序数、
  距离以及 line-graph 键角信息。
- 使用确定性的 fixed raw-graph embedding，避免随机初始化神经网络 embedding
  对无监督聚类造成不可解释的随机影响。
- 只研究 e_form，不运行其他性质或联合性质聚类。

包含三类实验：

Experiment A
    图空间与 e_form 空间分别独立选择各自最佳 K，然后比较两种聚类。

Experiment B
    图空间与 e_form 空间固定使用同一个指定 K，进行同尺度详细比较。

Cross-K scan
    对 K=2...12（可编辑）逐一扫描相同 K，观察 ARI/NMI/AMI/Hungarian overlap
    以及两个空间各自聚类质量随 K 的变化。

运行：
    python fusiformer_eform_clustering_experiments.py

请先修改下方“可编辑运行配置”，详细参数解释见配套 README。
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# 防止已有 shell 环境中 OMP_NUM_THREADS 被设置成空字符串等非法值。
if not str(os.environ.get("OMP_NUM_THREADS", "1")).strip().isdigit():
    os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
import dgl
import matplotlib.pyplot as plt

from jarvis.core.atoms import Atoms
from jarvis.core.graphs import nearest_neighbor_edges, build_undirected_edgedata

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.preprocessing import StandardScaler
import joblib

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


# =============================================================================
# 可编辑运行配置：形成能聚类专用版
# =============================================================================

# -----------------------------------------------------------------------------
# 0) 数据与输出
# -----------------------------------------------------------------------------
DATA_PATH = Path("data/my_dataset.json")
OUTPUT_DIR = Path("runs/eform_clustering_experiments")
PROPERTY_FIELD = "e_form"

# -----------------------------------------------------------------------------
# 1) 三类实验开关
# -----------------------------------------------------------------------------
# Experiment A：图空间与形成能空间分别独立选择各自最佳 K。
# 目的：研究两个空间各自的“自然聚类尺度”及其对应程度。
RUN_EXPERIMENT_A = True

# Experiment B：图空间和形成能空间强制使用同一个指定 K。
# 目的：在同一粒度下做可直接比较的聚类对应分析。
RUN_EXPERIMENT_B = True
EXPERIMENT_B_SHARED_K = 4

# Cross-K：对相同 K = 2...12 逐一扫描，并输出 ARI/NMI/AMI/Overlap 随 K 的变化。
# 目的：判断结构-形成能对应关系在哪个聚类尺度最强、是否稳定。
RUN_CROSS_K_SCAN = True
CROSS_K_VALUES = list(range(2, 13))

# -----------------------------------------------------------------------------
# 2) Fusiformer 原始图参数 —— 与项目原始构图保持一致
# -----------------------------------------------------------------------------
GRAPH_CUTOFF = 8.0
GRAPH_MAX_NEIGHBORS = 12
USE_CANONIZE = True
ATOM_Z_MAX = 95
DIST_RBF_BINS = 80
ANGLE_RBF_BINS = 40

# -----------------------------------------------------------------------------
# 3) 图特征提取
# -----------------------------------------------------------------------------
NUM_WORKERS = 1
PROCESS_CHUNKSIZE = 32

# 如果输出目录中已经存在 raw_graph_embeddings.npz，可直接复用，避免每次重新构图。
REUSE_GRAPH_EMBEDDING_CACHE = True

# -----------------------------------------------------------------------------
# 4) 图空间预处理
# -----------------------------------------------------------------------------
# fixed graph embedding -> StandardScaler -> PCA -> clustering
# 推荐主实验使用 0.95（保留 95% 累计解释方差）。
# 也可设置整数 32 / 64 做敏感性实验；None = 不做 PCA。
GRAPH_PCA_COMPONENTS: Optional[Any] = 0.95

# -----------------------------------------------------------------------------
# 5) K 选择参数（Experiment A）
# -----------------------------------------------------------------------------
K_MIN = 2
K_MAX = 12

# Experiment A 的最佳 K 依据。可选：
#   "silhouette"         -> 最大值最好（默认）
#   "calinski_harabasz" -> 最大值最好
#   "davies_bouldin"    -> 最小值最好
K_SELECTION_METRIC = "silhouette"

# 只在抽样数据上进行 K 选择指标计算，避免 144k 数据 silhouette 代价过高。
K_SELECTION_SAMPLE = 5000

# 若想跳过自动选择，可指定整数；None = 自动选择。
EXPERIMENT_A_GRAPH_K: Optional[int] = None
EXPERIMENT_A_EFORM_K: Optional[int] = None

# -----------------------------------------------------------------------------
# 6) KMeans / MiniBatchKMeans 参数
# -----------------------------------------------------------------------------
RANDOM_STATE = 42

# "minibatch"：推荐用于 144,484 样本，速度快。
# "kmeans"：更适合作为最终精确复核，但计算更慢。
CLUSTER_ALGORITHM = "minibatch"
KMEANS_N_INIT = 20
KMEANS_MAX_ITER = 500
KMEANS_TOL = 1e-4
MINIBATCH_BATCH_SIZE = 4096
MINIBATCH_REASSIGNMENT_RATIO = 0.01

# -----------------------------------------------------------------------------
# 7) 可视化
# -----------------------------------------------------------------------------
USE_UMAP_IF_AVAILABLE = True
VIS_SAMPLE_MAX = 30000
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.1
DPI = 180

# -----------------------------------------------------------------------------
# 8) 显著性与稳健性
# -----------------------------------------------------------------------------
# permutation test：0 = 关闭；正式论文可用 100~500。
N_PERMUTATIONS = 0

# Cross-K 是否为每一个 K 额外保存 144k 级 assignments/contingency。
# 正式扫描通常保持 False；Experiment A/B 的详细结果始终会保存。
SAVE_DETAILED_FIXED_K_OUTPUT = False

# =============================================================================
# 工具函数
# =============================================================================


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_float(v: Any) -> float:
    """尽量把 JSON 标量转成 float；无效值返回 NaN。"""
    if v is None or isinstance(v, bool):
        return float("nan")
    if isinstance(v, (int, float, np.integer, np.floating)):
        x = float(v)
        return x if np.isfinite(x) else float("nan")
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"", "na", "n/a", "nan", "none", "null", "-"}:
            return float("nan")
        try:
            x = float(s)
            return x if np.isfinite(x) else float("nan")
        except Exception:
            return float("nan")
    return float("nan")


def get_by_path(obj: Dict[str, Any], path: str) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def infer_sample_id(record: Dict[str, Any], index: int) -> str:
    for key in ["jid", "material_id", "mpid", "mp_id", "id"]:
        if key in record and record[key] not in (None, ""):
            return str(record[key])
    return f"sample_{index:07d}"


def infer_composition(record: Dict[str, Any]) -> str:
    for key in ["composition", "formula", "pretty_formula"]:
        if key in record and record[key] not in (None, ""):
            return str(record[key])
    try:
        atoms = Atoms.from_dict(record["atoms"])
        comp = getattr(atoms, "composition", None)
        if comp is not None:
            for attr in ["reduced_formula", "formula"]:
                if hasattr(comp, attr):
                    return str(getattr(comp, attr))
    except Exception:
        pass
    return ""


def _candidate_scalar_paths(record: Dict[str, Any]) -> Iterable[str]:
    """候选性质：顶层标量 + properties 字典内标量。"""
    for k, v in record.items():
        if k == "properties" and isinstance(v, dict):
            for kk, vv in v.items():
                if not isinstance(vv, (dict, list, tuple)):
                    yield f"properties.{kk}"
        elif not isinstance(v, (dict, list, tuple)):
            yield k


def _is_excluded_property_name(path: str) -> bool:
    leaf = path.split(".")[-1].lower()
    if path in EXCLUDE_PROPERTY_FIELDS or leaf in EXCLUDE_PROPERTY_FIELDS:
        return True
    # 结构/身份类字段默认不作为“物化性质”。
    patterns = [
        r"(^|_)id$", r"^id_", r"index", r"(^|_)idx$",
        r"num_atoms", r"n_atoms", r"natoms", r"nsites", r"num_sites",
        r"space.?group", r"spg", r"atomic_number",
    ]
    return any(re.search(p, leaf) for p in patterns)


def detect_property_fields(records: Sequence[Dict[str, Any]]) -> Tuple[List[str], pd.DataFrame]:
    counts = Counter()
    valid_counts = Counter()

    for rec in records:
        seen = set(_candidate_scalar_paths(rec))
        for path in seen:
            if _is_excluded_property_name(path):
                continue
            counts[path] += 1
            if np.isfinite(safe_float(get_by_path(rec, path))):
                valid_counts[path] += 1

    rows = []
    n = len(records)
    for path in sorted(counts):
        valid = valid_counts[path]
        coverage = valid / max(n, 1)
        rows.append({
            "field": path,
            "valid_count": valid,
            "total_count": n,
            "coverage": coverage,
            "selected": bool(valid >= 2 and coverage >= MIN_PROPERTY_COVERAGE),
        })
    coverage_df = pd.DataFrame(rows).sort_values(
        ["selected", "coverage", "field"], ascending=[False, False, True]
    )
    selected = coverage_df.loc[coverage_df["selected"], "field"].tolist()
    return selected, coverage_df


def sanitize_name(name: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z_\-]+", "_", name)
    return s.strip("_") or "property"


def rbf_expand_np(values: np.ndarray, vmin: float, vmax: float, bins: int) -> np.ndarray:
    """
    复现项目 models/layers.py::RBFExpansion：
      centers = linspace(vmin, vmax, bins)
      lengthscale = mean(diff(centers))
      gamma = 1 / lengthscale
      exp(-gamma * (x-center)^2)
    """
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    centers = np.linspace(vmin, vmax, bins, dtype=np.float32)
    if bins <= 1:
        gamma = 1.0
    else:
        lengthscale = float(np.diff(centers).mean())
        gamma = 1.0 / lengthscale
    return np.exp(-gamma * (values[:, None] - centers[None, :]) ** 2).astype(np.float32)


def mean_std_or_zeros(x: np.ndarray, dim: int) -> Tuple[np.ndarray, np.ndarray]:
    if x.size == 0:
        z = np.zeros(dim, dtype=np.float32)
        return z.copy(), z.copy()
    return x.mean(axis=0).astype(np.float32), x.std(axis=0).astype(np.float32)


def scalar_stats_or_zeros(x: np.ndarray) -> Tuple[float, float, float, float]:
    if x.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    x = np.asarray(x, dtype=np.float32)
    return float(x.mean()), float(x.std()), float(x.min()), float(x.max())


def graph_feature_names() -> List[str]:
    names = [f"atom_fraction_Z{z}" for z in range(1, ATOM_Z_MAX + 1)]
    names += [f"distance_rbf_mean_{i:02d}" for i in range(DIST_RBF_BINS)]
    names += [f"distance_rbf_std_{i:02d}" for i in range(DIST_RBF_BINS)]
    names += [f"angle_rbf_mean_{i:02d}" for i in range(ANGLE_RBF_BINS)]
    names += [f"angle_rbf_std_{i:02d}" for i in range(ANGLE_RBF_BINS)]
    names += [
        "num_nodes", "num_edges", "edges_per_node",
        "degree_mean", "degree_std", "degree_min", "degree_max",
        "distance_mean", "distance_std", "distance_min", "distance_max",
        "angle_cos_mean", "angle_cos_std", "angle_cos_min", "angle_cos_max",
        "atomic_Z_mean", "atomic_Z_std", "atomic_Z_min", "atomic_Z_max",
        "element_count", "composition_entropy",
    ]
    return names


def build_graph_embedding_from_record(record: Dict[str, Any]) -> np.ndarray:
    """单个晶体 -> deterministic raw graph embedding。"""
    if "atoms" not in record:
        raise KeyError("record 缺少 'atoms' 字段")

    atoms = Atoms.from_dict(record["atoms"])
    atomic_numbers = np.asarray(atoms.atomic_numbers, dtype=np.int64).reshape(-1)
    if atomic_numbers.size == 0:
        raise ValueError("晶体没有原子")
    if atomic_numbers.min() < 1 or atomic_numbers.max() > ATOM_Z_MAX:
        raise ValueError(
            f"atomic number 超出 Fusiformer atom embedding 支持范围 1..{ATOM_Z_MAX}: "
            f"min={atomic_numbers.min()}, max={atomic_numbers.max()}"
        )

    # 与 loader/loader.py 一致的邻居图构建。
    edges = nearest_neighbor_edges(
        atoms=atoms,
        cutoff=GRAPH_CUTOFF,
        max_neighbors=GRAPH_MAX_NEIGHBORS,
        use_canonize=USE_CANONIZE,
    )
    u, v, r = build_undirected_edgedata(atoms, edges)
    g = dgl.graph((u, v))
    offset = torch.as_tensor(r, dtype=torch.float32)
    g.edata["offset"] = offset

    distances = torch.norm(offset, dim=1).detach().cpu().numpy().astype(np.float32)

    # 与 loader/transforms.py + loader/utils.py 的 line-graph angle cosine 等价。
    lg = g.line_graph(shared=True)
    if lg.num_edges() > 0:
        src, dst = lg.edges()
        bond_vec = lg.ndata["offset"]
        r1 = -bond_vec[src]
        r2 = bond_vec[dst]
        denom = torch.norm(r1, dim=1) * torch.norm(r2, dim=1) + 1e-6
        angles = (torch.sum(r1 * r2, dim=1) / denom).clamp(-1, 1)
        angle_cos = angles.detach().cpu().numpy().astype(np.float32)
    else:
        angle_cos = np.empty((0,), dtype=np.float32)

    # 1) 原子序数组成分数（替代不可用的“未训练 nn.Embedding 随机权重”）。
    atom_hist = np.bincount(atomic_numbers, minlength=ATOM_Z_MAX + 1)[1:ATOM_Z_MAX + 1].astype(np.float32)
    atom_fraction = atom_hist / max(float(atom_hist.sum()), 1.0)

    # 2) 与 Fusiformer 初始边/角输入相同参数的固定 RBF。
    dist_rbf = rbf_expand_np(distances, 0.0, GRAPH_CUTOFF, DIST_RBF_BINS)
    angle_rbf = rbf_expand_np(angle_cos, -1.0, 1.0, ANGLE_RBF_BINS)
    dist_mean, dist_std = mean_std_or_zeros(dist_rbf, DIST_RBF_BINS)
    angle_mean, angle_std = mean_std_or_zeros(angle_rbf, ANGLE_RBF_BINS)

    # 3) 图统计量。
    degree = g.in_degrees().detach().cpu().numpy().astype(np.float32)
    deg_mean, deg_std, deg_min, deg_max = scalar_stats_or_zeros(degree)
    d_mean, d_std, d_min, d_max = scalar_stats_or_zeros(distances)
    a_mean, a_std, a_min, a_max = scalar_stats_or_zeros(angle_cos)
    z_mean, z_std, z_min, z_max = scalar_stats_or_zeros(atomic_numbers.astype(np.float32))

    nonzero = atom_fraction[atom_fraction > 0]
    entropy = float(-(nonzero * np.log(nonzero + 1e-12)).sum()) if nonzero.size else 0.0

    stats = np.asarray([
        float(g.num_nodes()),
        float(g.num_edges()),
        float(g.num_edges()) / max(float(g.num_nodes()), 1.0),
        deg_mean, deg_std, deg_min, deg_max,
        d_mean, d_std, d_min, d_max,
        a_mean, a_std, a_min, a_max,
        z_mean, z_std, z_min, z_max,
        float((atom_hist > 0).sum()),
        entropy,
    ], dtype=np.float32)

    vec = np.concatenate([
        atom_fraction,
        dist_mean, dist_std,
        angle_mean, angle_std,
        stats,
    ]).astype(np.float32)

    expected = len(graph_feature_names())
    if vec.shape[0] != expected:
        raise RuntimeError(f"graph embedding dim mismatch: {vec.shape[0]} != {expected}")
    if not np.all(np.isfinite(vec)):
        raise ValueError("graph embedding 中出现 NaN/Inf")
    return vec


def _graph_worker(payload: Tuple[int, Dict[str, Any]]) -> Tuple[int, Optional[np.ndarray], Optional[str]]:
    idx, record = payload
    try:
        return idx, build_graph_embedding_from_record(record), None
    except Exception as exc:
        return idx, None, f"{type(exc).__name__}: {exc}"


def extract_graph_embeddings(records: Sequence[Dict[str, Any]], out_dir: Path):
    names = graph_feature_names()
    vectors: List[np.ndarray] = []
    kept_indices: List[int] = []
    failures: List[Dict[str, Any]] = []

    payloads = list(enumerate(records))
    if NUM_WORKERS and NUM_WORKERS > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=NUM_WORKERS, mp_context=ctx) as ex:
            iterator = ex.map(_graph_worker, payloads, chunksize=PROCESS_CHUNKSIZE)
            iterator = tqdm(iterator, total=len(records), desc="Graph embedding")
            for idx, vec, err in iterator:
                if err is None and vec is not None:
                    kept_indices.append(idx)
                    vectors.append(vec)
                else:
                    failures.append({"sample_index": idx, "sample_id": infer_sample_id(records[idx], idx), "error": err})
    else:
        for payload in tqdm(payloads, total=len(records), desc="Graph embedding"):
            idx, vec, err = _graph_worker(payload)
            if err is None and vec is not None:
                kept_indices.append(idx)
                vectors.append(vec)
            else:
                failures.append({"sample_index": idx, "sample_id": infer_sample_id(records[idx], idx), "error": err})

    if failures:
        pd.DataFrame(failures).to_csv(out_dir / "failed_graph_samples.csv", index=False)
    if not vectors:
        raise RuntimeError("没有任何晶体成功生成 graph embedding。")

    X = np.stack(vectors, axis=0).astype(np.float32)
    kept = np.asarray(kept_indices, dtype=np.int64)
    np.savez_compressed(
        out_dir / "raw_graph_embeddings.npz",
        embeddings=X,
        sample_indices=kept,
        feature_names=np.asarray(names, dtype=object),
    )
    return X, kept, names, failures


def sample_indices_for_clustering(n: int, max_n: int, seed: int) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_n, replace=False))


def choose_k_by_quality(
    X: np.ndarray,
    fixed_k: Optional[int],
    label: str,
    out_csv: Optional[Path] = None,
) -> Tuple[int, pd.DataFrame]:
    """
    Experiment A 使用：在 K_MIN..K_MAX 中独立选择最佳 K。

    为控制 144k 数据的计算量，只在最多 K_SELECTION_SAMPLE 个样本上评估。
    同时输出 silhouette / Calinski-Harabasz / Davies-Bouldin / inertia。
    最终采用 K_SELECTION_METRIC 指定的指标选 K。
    """
    n = len(X)
    if n < 3:
        raise ValueError(f"{label}: 样本数不足 ({n})")

    if fixed_k is not None:
        if fixed_k < 2 or fixed_k >= n:
            raise ValueError(f"{label}: fixed K={fixed_k} 必须满足 2 <= K < N={n}")
        df = pd.DataFrame([{
            "k": fixed_k,
            "silhouette": np.nan,
            "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan,
            "inertia": np.nan,
            "mode": "fixed",
        }])
        if out_csv:
            df.to_csv(out_csv, index=False)
        return fixed_k, df

    idx = sample_indices_for_clustering(n, K_SELECTION_SAMPLE, RANDOM_STATE)
    Xs = X[idx]
    rows = []
    max_k = min(K_MAX, len(Xs) - 1)
    min_k = min(K_MIN, max_k)

    for k in range(min_k, max_k + 1):
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            random_state=RANDOM_STATE,
            n_init=KMEANS_N_INIT,
            max_iter=KMEANS_MAX_ITER,
            tol=KMEANS_TOL,
        )
        labels = model.fit_predict(Xs)
        if len(np.unique(labels)) < 2:
            sil = ch = db = float("nan")
        else:
            sil = float(silhouette_score(Xs, labels, metric="euclidean"))
            ch = float(calinski_harabasz_score(Xs, labels))
            db = float(davies_bouldin_score(Xs, labels))
        rows.append({
            "k": k,
            "silhouette": sil,
            "calinski_harabasz": ch,
            "davies_bouldin": db,
            "inertia": float(model.inertia_),
            "mode": "auto",
        })

    df = pd.DataFrame(rows)
    metric = K_SELECTION_METRIC.strip().lower()
    if metric not in {"silhouette", "calinski_harabasz", "davies_bouldin"}:
        raise ValueError(
            "K_SELECTION_METRIC 必须是 silhouette / calinski_harabasz / davies_bouldin"
        )
    valid = df[np.isfinite(df[metric])].copy()
    if valid.empty:
        best_k = max(2, min(4, n - 1))
    elif metric == "davies_bouldin":
        best_k = int(valid.sort_values([metric, "k"], ascending=[True, True]).iloc[0]["k"])
    else:
        best_k = int(valid.sort_values([metric, "k"], ascending=[False, True]).iloc[0]["k"])

    df["selected"] = df["k"] == best_k
    df["selection_metric"] = metric
    if out_csv:
        df.to_csv(out_csv, index=False)
    return best_k, df


def plot_k_quality_curves(df: pd.DataFrame, title: str, out_dir: Path):
    ensure_dir(out_dir)
    specs = [
        ("silhouette", "Silhouette score"),
        ("calinski_harabasz", "Calinski-Harabasz score"),
        ("davies_bouldin", "Davies-Bouldin score"),
        ("inertia", "KMeans inertia"),
    ]
    for col, ylabel in specs:
        if col not in df.columns:
            continue
        valid = df[np.isfinite(df[col])]
        if valid.empty:
            continue
        plt.figure(figsize=(6, 4))
        plt.plot(valid["k"], valid[col], marker="o")
        plt.xlabel("K")
        plt.ylabel(ylabel)
        plt.title(f"{title}: {ylabel}")
        plt.tight_layout()
        plt.savefig(out_dir / f"k_selection_{col}.png", dpi=DPI)
        plt.close()


def fit_cluster_model(X: np.ndarray, k: int, seed: Optional[int] = None):
    """统一聚类入口；Experiment A/B/Cross-K 使用完全相同的算法参数。"""
    seed = RANDOM_STATE if seed is None else int(seed)
    algo = CLUSTER_ALGORITHM.strip().lower()
    if algo == "kmeans":
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            random_state=seed,
            n_init=KMEANS_N_INIT,
            max_iter=KMEANS_MAX_ITER,
            tol=KMEANS_TOL,
        )
    elif algo == "minibatch":
        model = MiniBatchKMeans(
            n_clusters=k,
            init="k-means++",
            random_state=seed,
            batch_size=min(MINIBATCH_BATCH_SIZE, max(256, len(X))),
            n_init=KMEANS_N_INIT,
            max_iter=KMEANS_MAX_ITER,
            tol=KMEANS_TOL,
            reassignment_ratio=MINIBATCH_REASSIGNMENT_RATIO,
        )
    else:
        raise ValueError("CLUSTER_ALGORITHM 必须为 'minibatch' 或 'kmeans'")
    model.fit(X)
    return model


# 兼容旧函数名，避免下面未使用的辅助函数报错。
def fit_minibatch_kmeans(X: np.ndarray, k: int):
    return fit_cluster_model(X, k)


def reduce_for_visualization(X: np.ndarray, method_name: str, out_dir: Path) -> Tuple[np.ndarray, str]:
    idx = sample_indices_for_clustering(len(X), VIS_SAMPLE_MAX, RANDOM_STATE)
    Xs = X[idx]

    if USE_UMAP_IF_AVAILABLE:
        try:
            import umap
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=UMAP_N_NEIGHBORS,
                min_dist=UMAP_MIN_DIST,
                metric="euclidean",
                random_state=RANDOM_STATE,
            )
            coords = reducer.fit_transform(Xs)
            return np.c_[idx, coords], "UMAP"
        except Exception as exc:
            warnings.warn(f"UMAP 不可用，回退 PCA2: {exc}")

    pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca2.fit_transform(Xs)
    return np.c_[idx, coords], "PCA"


def plot_embedding_scatter(coords_with_idx: np.ndarray, labels: np.ndarray, method: str, title: str, path: Path):
    idx = coords_with_idx[:, 0].astype(int)
    coords = coords_with_idx[:, 1:3]
    sub_labels = labels[idx]
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=sub_labels, s=7, alpha=0.7, cmap="tab20")
    plt.xlabel(f"{method}-1")
    plt.ylabel(f"{method}-2")
    plt.title(title)
    plt.colorbar(sc, label="cluster")
    plt.tight_layout()
    plt.savefig(path, dpi=DPI)
    plt.close()


def plot_property_on_embedding(
    coords_with_idx: np.ndarray,
    values: np.ndarray,
    method: str,
    title: str,
    path: Path,
):
    idx = coords_with_idx[:, 0].astype(int)
    coords = coords_with_idx[:, 1:3]
    v = values[idx]
    finite = np.isfinite(v)
    if finite.sum() < 2:
        return
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(coords[finite, 0], coords[finite, 1], c=v[finite], s=7, alpha=0.7, cmap="viridis")
    plt.xlabel(f"{method}-1")
    plt.ylabel(f"{method}-2")
    plt.title(title)
    plt.colorbar(sc, label=title)
    plt.tight_layout()
    plt.savefig(path, dpi=DPI)
    plt.close()


def contingency_table(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized contingency table; suitable for 1e5+ samples and permutation tests."""
    a = np.asarray(a)
    b = np.asarray(b)
    ua, ai = np.unique(a, return_inverse=True)
    ub, bi = np.unique(b, return_inverse=True)
    flat = ai.astype(np.int64) * len(ub) + bi.astype(np.int64)
    table = np.bincount(flat, minlength=len(ua) * len(ub)).reshape(len(ua), len(ub))
    return table.astype(np.int64, copy=False), ua, ub


def cluster_correspondence_metrics(a: np.ndarray, b: np.ndarray) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    a = np.asarray(a)
    b = np.asarray(b)
    if len(a) != len(b):
        raise ValueError("label lengths differ")
    table, ua, ub = contingency_table(a, b)

    row_ind, col_ind = linear_sum_assignment(-table)
    overlap = float(table[row_ind, col_ind].sum() / max(table.sum(), 1))
    purity = float(table.max(axis=1).sum() / max(table.sum(), 1))

    metrics = {
        "n_samples": int(len(a)),
        "graph_k": int(len(ua)),
        "property_k": int(len(ub)),
        "ARI": float(adjusted_rand_score(a, b)),
        "NMI": float(normalized_mutual_info_score(a, b)),
        "AMI": float(adjusted_mutual_info_score(a, b)),
        "hungarian_overlap": overlap,
        "graph_to_property_purity": purity,
    }

    table_df = pd.DataFrame(table, index=[f"G{int(x)}" for x in ua], columns=[f"P{int(x)}" for x in ub])
    match_df = pd.DataFrame({
        "graph_cluster": ua[row_ind],
        "property_cluster": ub[col_ind],
        "overlap_count": table[row_ind, col_ind],
    })
    return metrics, table_df, match_df


def permutation_significance(a: np.ndarray, b: np.ndarray, n_perm: int) -> Dict[str, float]:
    if n_perm <= 0:
        return {}
    obs, _, _ = cluster_correspondence_metrics(a, b)
    rng = np.random.default_rng(RANDOM_STATE + 11)
    null = {"ARI": [], "NMI": [], "AMI": [], "hungarian_overlap": []}
    for _ in range(n_perm):
        bp = rng.permutation(b)
        m, _, _ = cluster_correspondence_metrics(a, bp)
        for key in null:
            null[key].append(m[key])
    out = {}
    for key, vals in null.items():
        arr = np.asarray(vals, dtype=float)
        p = (1.0 + float((arr >= obs[key]).sum())) / (1.0 + n_perm)
        out[f"{key}_perm_p"] = p
        out[f"{key}_perm_mean"] = float(arr.mean())
        out[f"{key}_perm_std"] = float(arr.std())
    return out


def plot_matrix(df: pd.DataFrame, title: str, path: Path, fmt: str = "d"):
    arr = df.values
    plt.figure(figsize=(max(6, 0.65 * df.shape[1] + 2), max(5, 0.55 * df.shape[0] + 2)))
    im = plt.imshow(arr, aspect="auto", interpolation="nearest", cmap="viridis")
    plt.xticks(range(df.shape[1]), df.columns, rotation=45, ha="right")
    plt.yticks(range(df.shape[0]), df.index)
    plt.title(title)
    plt.colorbar(im)
    if df.shape[0] <= 12 and df.shape[1] <= 12:
        threshold = np.nanmax(arr) * 0.55 if arr.size else 0
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                val = arr[i, j]
                if fmt == "d":
                    txt = f"{int(val)}"
                else:
                    txt = format(float(val), fmt)
                plt.text(j, i, txt, ha="center", va="center",
                         color="white" if val < threshold else "black", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=DPI)
    plt.close()


def plot_graph_cluster_boxplot(values: np.ndarray, graph_labels: np.ndarray, title: str, path: Path):
    groups = []
    names = []
    for g in np.unique(graph_labels):
        v = values[graph_labels == g]
        v = v[np.isfinite(v)]
        if len(v):
            groups.append(v)
            names.append(f"G{g}")
    if len(groups) < 1:
        return
    plt.figure(figsize=(max(7, 0.8 * len(groups) + 2), 5))
    plt.boxplot(groups, labels=names, showfliers=False)
    plt.xlabel("Graph cluster")
    plt.ylabel(title)
    plt.title(f"{title} distribution by graph cluster")
    plt.tight_layout()
    plt.savefig(path, dpi=DPI)
    plt.close()


def centroid_distance_single_property(
    z_values: np.ndarray,
    graph_labels: np.ndarray,
    property_model: MiniBatchKMeans,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ug = np.unique(graph_labels)
    graph_centers = np.asarray([z_values[graph_labels == g].mean() for g in ug], dtype=float).reshape(-1, 1)
    prop_centers = property_model.cluster_centers_.reshape(-1, 1)
    dist = cdist(graph_centers, prop_centers, metric="euclidean")
    df = pd.DataFrame(dist, index=[f"G{g}" for g in ug], columns=[f"P{p}" for p in range(len(prop_centers))])
    r, c = linear_sum_assignment(dist)
    match = pd.DataFrame({
        "graph_cluster": ug[r],
        "property_cluster": c,
        "standardized_centroid_distance": dist[r, c],
    })
    return df, match


def centroid_distance_joint(
    X_prop: np.ndarray,
    graph_labels: np.ndarray,
    property_model: MiniBatchKMeans,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ug = np.unique(graph_labels)
    graph_centers = np.vstack([X_prop[graph_labels == g].mean(axis=0) for g in ug])
    prop_centers = property_model.cluster_centers_
    dist = cdist(graph_centers, prop_centers, metric="euclidean")
    df = pd.DataFrame(dist, index=[f"G{g}" for g in ug], columns=[f"P{p}" for p in range(len(prop_centers))])
    r, c = linear_sum_assignment(dist)
    match = pd.DataFrame({
        "graph_cluster": ug[r],
        "property_cluster": c,
        "standardized_centroid_distance": dist[r, c],
    })
    return df, match


# =============================================================================
# 形成能专用实验辅助函数
# =============================================================================


def load_or_extract_graph_embeddings(records: Sequence[Dict[str, Any]], graph_cache_dir: Path):
    """复用 raw graph embedding，避免每次 Experiment A/B/Cross-K 都重新构图。"""
    ensure_dir(graph_cache_dir)
    cache = graph_cache_dir / "raw_graph_embeddings.npz"
    if REUSE_GRAPH_EMBEDDING_CACHE and cache.exists():
        print(f"Reusing graph embedding cache: {cache}")
        z = np.load(cache, allow_pickle=True)
        X = z["embeddings"].astype(np.float32)
        kept = z["sample_indices"].astype(np.int64)
        names = z["feature_names"].tolist() if "feature_names" in z else graph_feature_names()
        return X, kept, names, []
    return extract_graph_embeddings(records, graph_cache_dir)


def preprocess_graph_space(X_raw: np.ndarray, out_dir: Path):
    """固定图描述符 -> StandardScaler -> 可选 PCA。只做一次，三类实验共享。"""
    ensure_dir(out_dir)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_raw).astype(np.float32)
    joblib.dump(scaler, out_dir / "graph_scaler.joblib")

    pca = None
    Xc = Xs
    spec = GRAPH_PCA_COMPONENTS
    if spec is not None:
        if isinstance(spec, float) and 0.0 < spec < 1.0:
            # sklearn 的方差比例模式要求 full SVD。
            pca = PCA(n_components=spec, svd_solver="full", random_state=RANDOM_STATE)
            Xc = pca.fit_transform(Xs).astype(np.float32)
        elif isinstance(spec, (int, np.integer)):
            n_comp = min(int(spec), Xs.shape[1], len(Xs) - 1)
            if n_comp < Xs.shape[1]:
                pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=RANDOM_STATE)
                Xc = pca.fit_transform(Xs).astype(np.float32)
        else:
            raise ValueError("GRAPH_PCA_COMPONENTS 必须为 None、0~1 之间的小数或正整数。")

    if pca is not None:
        pd.DataFrame({
            "component": np.arange(1, len(pca.explained_variance_ratio_) + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative": np.cumsum(pca.explained_variance_ratio_),
        }).to_csv(out_dir / "graph_pca_explained_variance.csv", index=False)
        joblib.dump(pca, out_dir / "graph_pca.joblib")

    meta = {
        "raw_dim": int(X_raw.shape[1]),
        "cluster_dim": int(Xc.shape[1]),
        "pca_components_setting": GRAPH_PCA_COMPONENTS,
        "pca_cumulative_explained_variance": (
            float(np.sum(pca.explained_variance_ratio_)) if pca is not None else None
        ),
    }
    with open(out_dir / "graph_preprocess_summary.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return Xc, scaler, pca


def summarize_eform(values: np.ndarray, out_dir: Path):
    ensure_dir(out_dir)
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    qs = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    qv = np.percentile(values, qs)
    pd.DataFrame({"percentile": qs, "e_form": qv}).to_csv(
        out_dir / "e_form_percentiles.csv", index=False
    )
    summary = {
        "n": int(len(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
    }
    with open(out_dir / "e_form_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    plt.figure(figsize=(7, 5))
    plt.hist(values, bins=100)
    plt.xlabel("e_form")
    plt.ylabel("Count")
    plt.title("Formation energy distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "e_form_histogram.png", dpi=DPI)
    plt.close()


def make_eform_standardized(values_on_graph: np.ndarray):
    valid = np.isfinite(values_on_graph)
    if valid.sum() < 10:
        raise ValueError("e_form 有效样本不足。")
    scaler = StandardScaler()
    z = scaler.fit_transform(values_on_graph[valid].reshape(-1, 1)).astype(np.float32)
    return valid, z, scaler


def sampled_cluster_quality(X: np.ndarray, model) -> Dict[str, float]:
    idx = sample_indices_for_clustering(len(X), K_SELECTION_SAMPLE, RANDOM_STATE)
    Xs = X[idx]
    labels = model.predict(Xs)
    if len(np.unique(labels)) < 2:
        return {
            "silhouette": float("nan"),
            "calinski_harabasz": float("nan"),
            "davies_bouldin": float("nan"),
        }
    return {
        "silhouette": float(silhouette_score(Xs, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(Xs, labels)),
        "davies_bouldin": float(davies_bouldin_score(Xs, labels)),
    }


def save_fixed_pair_outputs(
    X_graph: np.ndarray,
    e_z: np.ndarray,
    e_original: np.ndarray,
    global_indices: np.ndarray,
    records: Sequence[Dict[str, Any]],
    graph_model,
    eform_model,
    out_dir: Path,
    experiment_name: str,
):
    """保存 Experiment A/B 共用的详细对应性结果。"""
    ensure_dir(out_dir)
    g_labels = graph_model.labels_.astype(int)
    p_labels = eform_model.labels_.astype(int)

    metrics, table_df, match_df = cluster_correspondence_metrics(g_labels, p_labels)
    metrics.update(permutation_significance(g_labels, p_labels, N_PERMUTATIONS))
    metrics["experiment"] = experiment_name
    metrics["graph_algorithm"] = CLUSTER_ALGORITHM
    metrics["graph_inertia"] = float(graph_model.inertia_)
    metrics["eform_inertia"] = float(eform_model.inertia_)

    gq = sampled_cluster_quality(X_graph, graph_model)
    pq = sampled_cluster_quality(e_z, eform_model)
    for k, v in gq.items():
        metrics[f"graph_{k}"] = v
    for k, v in pq.items():
        metrics[f"eform_{k}"] = v

    pd.DataFrame([metrics]).to_csv(out_dir / "correspondence_metrics.csv", index=False)
    table_df.to_csv(out_dir / "graph_vs_eform_contingency.csv")
    match_df.to_csv(out_dir / "hungarian_overlap_matching.csv", index=False)
    plot_matrix(table_df, f"{experiment_name}: Graph vs e_form", out_dir / "contingency_heatmap.png", fmt="d")

    row_sum = table_df.sum(axis=1).replace(0, np.nan)
    row_norm = table_df.div(row_sum, axis=0).fillna(0.0)
    row_norm.to_csv(out_dir / "graph_to_eform_row_fraction.csv")
    plot_matrix(
        row_norm,
        f"{experiment_name}: P(e_form cluster | graph cluster)",
        out_dir / "contingency_row_fraction.png",
        fmt=".2f",
    )

    # 在同一个标准化形成能空间中比较 graph-cluster centroid 与 e_form-cluster centroid。
    dist_df, centroid_match = centroid_distance_single_property(
        e_z.reshape(-1), g_labels, eform_model
    )
    dist_df.to_csv(out_dir / "centroid_distance_standardized.csv")
    centroid_match.to_csv(out_dir / "centroid_hungarian_matching.csv", index=False)
    plot_matrix(
        dist_df,
        f"{experiment_name}: centroid distance",
        out_dir / "centroid_distance_heatmap.png",
        fmt=".2f",
    )

    # 形成能簇中心恢复为 eV/atom 原始单位。
    e_scaler = StandardScaler().fit(e_original.reshape(-1, 1))
    # 注意：模型训练的 e_z 来自同一批 e_original 的 StandardScaler，下面直接由标准化均值/方差恢复。
    # 为避免重新拟合存在数值差异，这里用训练数据的均值/标准差公式恢复。
    mean = float(np.mean(e_original))
    scale = float(np.std(e_original))
    centers_original = eform_model.cluster_centers_.reshape(-1) * scale + mean
    pd.DataFrame({
        "eform_cluster": np.arange(len(centers_original)),
        "center_e_form_original_unit": centers_original,
        "count": np.bincount(p_labels, minlength=len(centers_original)),
    }).sort_values("center_e_form_original_unit").to_csv(out_dir / "eform_cluster_centers.csv", index=False)

    graph_stats = []
    for g in np.unique(g_labels):
        vals = e_original[g_labels == g]
        graph_stats.append({
            "graph_cluster": int(g),
            "count": int(len(vals)),
            "mean_e_form": float(np.mean(vals)),
            "median_e_form": float(np.median(vals)),
            "std_e_form": float(np.std(vals)),
            "min_e_form": float(np.min(vals)),
            "max_e_form": float(np.max(vals)),
        })
    pd.DataFrame(graph_stats).to_csv(out_dir / "graph_cluster_eform_statistics.csv", index=False)

    pd.DataFrame({
        "sample_index": global_indices,
        "sample_id": [infer_sample_id(records[int(i)], int(i)) for i in global_indices],
        "formula": [infer_composition(records[int(i)]) for i in global_indices],
        "e_form": e_original,
        "graph_cluster": g_labels,
        "eform_cluster": p_labels,
    }).to_csv(out_dir / "assignments.csv", index=False)

    plot_graph_cluster_boxplot(
        e_original, g_labels, "e_form", out_dir / "e_form_by_graph_cluster_boxplot.png"
    )
    joblib.dump(graph_model, out_dir / "graph_cluster_model.joblib")
    joblib.dump(eform_model, out_dir / "eform_cluster_model.joblib")
    return metrics


def run_experiment_a(
    X_graph: np.ndarray,
    e_z: np.ndarray,
    e_original: np.ndarray,
    global_indices: np.ndarray,
    records: Sequence[Dict[str, Any]],
    out_dir: Path,
):
    """Experiment A：两个空间分别独立自动/指定最佳 K。"""
    ensure_dir(out_dir)
    graph_k, graph_curve = choose_k_by_quality(
        X_graph, EXPERIMENT_A_GRAPH_K, "graph",
        out_dir / "graph_k_selection.csv",
    )
    eform_k, eform_curve = choose_k_by_quality(
        e_z, EXPERIMENT_A_EFORM_K, "e_form",
        out_dir / "eform_k_selection.csv",
    )
    plot_k_quality_curves(graph_curve, "Experiment A graph", out_dir / "graph_k_quality")
    plot_k_quality_curves(eform_curve, "Experiment A e_form", out_dir / "eform_k_quality")

    graph_model = fit_cluster_model(X_graph, graph_k)
    eform_model = fit_cluster_model(e_z, eform_k)
    metrics = save_fixed_pair_outputs(
        X_graph, e_z, e_original, global_indices, records,
        graph_model, eform_model, out_dir, "Experiment_A_independent_K",
    )
    with open(out_dir / "experiment_A_selected_K.json", "w", encoding="utf-8") as f:
        json.dump({
            "graph_k": int(graph_k),
            "eform_k": int(eform_k),
            "selection_metric": K_SELECTION_METRIC,
        }, f, ensure_ascii=False, indent=2)
    return metrics, graph_model, eform_model


def run_experiment_b(
    X_graph: np.ndarray,
    e_z: np.ndarray,
    e_original: np.ndarray,
    global_indices: np.ndarray,
    records: Sequence[Dict[str, Any]],
    out_dir: Path,
):
    """Experiment B：两个空间固定使用相同 K。"""
    k = int(EXPERIMENT_B_SHARED_K)
    if k < 2 or k >= min(len(X_graph), len(e_z)):
        raise ValueError(f"EXPERIMENT_B_SHARED_K={k} 非法。")
    graph_model = fit_cluster_model(X_graph, k)
    eform_model = fit_cluster_model(e_z, k)
    metrics = save_fixed_pair_outputs(
        X_graph, e_z, e_original, global_indices, records,
        graph_model, eform_model, out_dir, f"Experiment_B_same_K_{k}",
    )
    return metrics, graph_model, eform_model


def plot_cross_k_metrics(df: pd.DataFrame, out_dir: Path):
    ensure_dir(out_dir)
    cols = ["ARI", "NMI", "AMI", "hungarian_overlap"]
    plt.figure(figsize=(8, 5))
    for col in cols:
        if col in df.columns:
            plt.plot(df["k"], df[col], marker="o", label=col)
    plt.xlabel("Shared K")
    plt.ylabel("Correspondence score")
    plt.title("Graph–formation-energy correspondence across K")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "correspondence_vs_k.png", dpi=DPI)
    plt.close()

    # 两个空间各自的聚类质量曲线分别绘制，避免与对应性指标混淆。
    for prefix, title in [("graph", "Graph"), ("eform", "e_form")]:
        for metric in ["silhouette", "calinski_harabasz", "davies_bouldin"]:
            col = f"{prefix}_{metric}"
            if col not in df.columns:
                continue
            plt.figure(figsize=(6, 4))
            plt.plot(df["k"], df[col], marker="o")
            plt.xlabel("Shared K")
            plt.ylabel(metric)
            plt.title(f"{title} clustering quality across K: {metric}")
            plt.tight_layout()
            plt.savefig(out_dir / f"{prefix}_{metric}_vs_k.png", dpi=DPI)
            plt.close()


def run_cross_k_scan(
    X_graph: np.ndarray,
    e_z: np.ndarray,
    e_original: np.ndarray,
    global_indices: np.ndarray,
    records: Sequence[Dict[str, Any]],
    out_dir: Path,
):
    """相同 K 扫描：K=2...12，每个 K 都重新聚类两个空间并计算对应指标。"""
    ensure_dir(out_dir)
    rows = []
    valid_k_values = []
    for k in CROSS_K_VALUES:
        k = int(k)
        if k < 2 or k >= min(len(X_graph), len(e_z)):
            continue
        valid_k_values.append(k)
        print(f"    Cross-K: K={k}")
        graph_model = fit_cluster_model(X_graph, k)
        eform_model = fit_cluster_model(e_z, k)
        m, _, _ = cluster_correspondence_metrics(graph_model.labels_, eform_model.labels_)
        gq = sampled_cluster_quality(X_graph, graph_model)
        pq = sampled_cluster_quality(e_z, eform_model)
        row = {"k": k, **m}
        for key, val in gq.items():
            row[f"graph_{key}"] = val
        for key, val in pq.items():
            row[f"eform_{key}"] = val
        row["graph_inertia"] = float(graph_model.inertia_)
        row["eform_inertia"] = float(eform_model.inertia_)
        rows.append(row)

        # 可选：为每个 K 保存详细结果。默认只保存汇总，避免产生大量文件。
        if SAVE_DETAILED_FIXED_K_OUTPUT:
            k_dir = ensure_dir(out_dir / f"K_{k:02d}")
            pd.DataFrame({
                "sample_index": global_indices,
                "e_form": e_original,
                "graph_cluster": graph_model.labels_.astype(int),
                "eform_cluster": eform_model.labels_.astype(int),
            }).to_csv(k_dir / "assignments.csv", index=False)
            table_df = cluster_correspondence_metrics(
                graph_model.labels_, eform_model.labels_
            )[1]
            table_df.to_csv(k_dir / "contingency.csv")

    if not rows:
        raise RuntimeError("CROSS_K_VALUES 中没有合法 K。")
    df = pd.DataFrame(rows).sort_values("k")
    df.to_csv(out_dir / "cross_k_metrics.csv", index=False)
    plot_cross_k_metrics(df, out_dir)

    # 只作为诊断摘要，不自动把对应性最高 K 当成“真实最佳 K”。
    summary = {}
    for col in ["ARI", "NMI", "AMI", "hungarian_overlap"]:
        if col in df.columns and np.isfinite(df[col]).any():
            i = df[col].idxmax()
            summary[f"max_{col}"] = float(df.loc[i, col])
            summary[f"k_at_max_{col}"] = int(df.loc[i, "k"])
    with open(out_dir / "cross_k_peak_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return df


# =============================================================================
# 主程序：仅研究 formation energy (e_form)
# =============================================================================


def main():
    t0 = time.time()
    project_root = Path(__file__).resolve().parent
    data_path = DATA_PATH if DATA_PATH.is_absolute() else project_root / DATA_PATH
    output_dir = OUTPUT_DIR if OUTPUT_DIR.is_absolute() else project_root / OUTPUT_DIR
    ensure_dir(output_dir)
    graph_cache_dir = ensure_dir(output_dir / "graph_cache")
    preprocess_dir = ensure_dir(output_dir / "graph_preprocess")
    diagnostics_dir = ensure_dir(output_dir / "data_diagnostics")

    print("=" * 88)
    print("Fusiformer raw graph vs formation-energy clustering experiments")
    print("NO trained model / NO checkpoint")
    print("data:", data_path)
    print("output:", output_dir)
    print("Experiment A:", RUN_EXPERIMENT_A)
    print("Experiment B:", RUN_EXPERIMENT_B, "shared K =", EXPERIMENT_B_SHARED_K)
    print("Cross-K:", RUN_CROSS_K_SCAN, CROSS_K_VALUES)
    print("=" * 88)

    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list) or not records:
        raise ValueError("数据 JSON 顶层必须是非空 list。")
    print(f"Loaded samples: {len(records):,}")

    # e_form 全数据检查。
    e_all = np.asarray([safe_float(get_by_path(r, PROPERTY_FIELD)) for r in records], dtype=float)
    n_valid_all = int(np.isfinite(e_all).sum())
    print(f"Valid {PROPERTY_FIELD}: {n_valid_all:,}/{len(records):,}")
    if n_valid_all < 10:
        raise RuntimeError(f"{PROPERTY_FIELD} 有效数值不足。")
    summarize_eform(e_all, diagnostics_dir)

    config_snapshot = {
        "data_path": str(data_path),
        "property_field": PROPERTY_FIELD,
        "run_experiment_A": RUN_EXPERIMENT_A,
        "run_experiment_B": RUN_EXPERIMENT_B,
        "experiment_B_shared_k": EXPERIMENT_B_SHARED_K,
        "run_cross_k_scan": RUN_CROSS_K_SCAN,
        "cross_k_values": CROSS_K_VALUES,
        "graph_cutoff": GRAPH_CUTOFF,
        "graph_max_neighbors": GRAPH_MAX_NEIGHBORS,
        "distance_rbf_bins": DIST_RBF_BINS,
        "angle_rbf_bins": ANGLE_RBF_BINS,
        "graph_pca_components": GRAPH_PCA_COMPONENTS,
        "k_min": K_MIN,
        "k_max": K_MAX,
        "k_selection_metric": K_SELECTION_METRIC,
        "k_selection_sample": K_SELECTION_SAMPLE,
        "cluster_algorithm": CLUSTER_ALGORITHM,
        "kmeans_n_init": KMEANS_N_INIT,
        "kmeans_max_iter": KMEANS_MAX_ITER,
        "kmeans_tol": KMEANS_TOL,
        "minibatch_batch_size": MINIBATCH_BATCH_SIZE,
        "random_state": RANDOM_STATE,
        "n_permutations": N_PERMUTATIONS,
        "num_workers": NUM_WORKERS,
        "reuse_graph_embedding_cache": REUSE_GRAPH_EMBEDDING_CACHE,
    }
    with open(output_dir / "analysis_config.json", "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, ensure_ascii=False, indent=2)

    print("\n[1/3] Preparing deterministic raw graph embeddings...")
    X_raw, kept_indices, feature_names, failures = load_or_extract_graph_embeddings(
        records, graph_cache_dir
    )
    print(f"Graph embeddings: {X_raw.shape}; failures={len(failures)}")

    print("\n[2/3] Preprocessing graph space...")
    X_graph, graph_scaler, graph_pca = preprocess_graph_space(X_raw, preprocess_dir)
    print(f"Graph clustering space: {X_graph.shape}")

    # 三类实验只比较“成功构图 + e_form 有效”的完全相同样本集合。
    e_on_graph = e_all[kept_indices]
    valid, e_z, e_scaler = make_eform_standardized(e_on_graph)
    X_graph_valid = X_graph[valid]
    e_original = e_on_graph[valid].astype(float)
    global_indices = kept_indices[valid]
    joblib.dump(e_scaler, diagnostics_dir / "e_form_scaler.joblib")

    if len(X_graph_valid) != len(e_z):
        raise RuntimeError("graph 与 e_form 样本对齐失败。")
    print(f"Aligned analysis samples: {len(e_z):,}")

    # 共用的二维图空间坐标，只用于可视化，不参与聚类。
    vis_coords, vis_method = reduce_for_visualization(X_graph_valid, "graph", preprocess_dir)

    metric_rows = []
    print("\n[3/3] Running clustering experiments...")

    if RUN_EXPERIMENT_A:
        print("  - Experiment A: independent K selection")
        a_dir = ensure_dir(output_dir / "experiment_A_independent_K")
        m, gm, pm = run_experiment_a(
            X_graph_valid, e_z, e_original, global_indices, records, a_dir
        )
        metric_rows.append(m)
        plot_embedding_scatter(
            vis_coords, gm.labels_.astype(int), vis_method,
            f"Experiment A graph clusters (K={gm.n_clusters})",
            a_dir / "graph_space_clusters.png",
        )
        plot_property_on_embedding(
            vis_coords, e_original, vis_method, "e_form",
            a_dir / "e_form_on_graph_embedding.png",
        )

    if RUN_EXPERIMENT_B:
        print(f"  - Experiment B: same K = {EXPERIMENT_B_SHARED_K}")
        b_dir = ensure_dir(output_dir / f"experiment_B_same_K_{int(EXPERIMENT_B_SHARED_K):02d}")
        m, gm, pm = run_experiment_b(
            X_graph_valid, e_z, e_original, global_indices, records, b_dir
        )
        metric_rows.append(m)
        plot_embedding_scatter(
            vis_coords, gm.labels_.astype(int), vis_method,
            f"Experiment B graph clusters (K={gm.n_clusters})",
            b_dir / "graph_space_clusters.png",
        )

    if RUN_CROSS_K_SCAN:
        print("  - Cross-K scan")
        cross_dir = ensure_dir(output_dir / "cross_K_scan")
        run_cross_k_scan(
            X_graph_valid, e_z, e_original, global_indices, records, cross_dir
        )

    if metric_rows:
        pd.DataFrame(metric_rows).to_csv(output_dir / "main_experiment_metrics.csv", index=False)

    run_summary = {
        "n_input_samples": int(len(records)),
        "n_graph_success": int(len(kept_indices)),
        "n_graph_failures": int(len(failures)),
        "n_aligned_analysis_samples": int(len(e_z)),
        "raw_graph_embedding_dim": int(X_raw.shape[1]),
        "graph_cluster_space_dim": int(X_graph_valid.shape[1]),
        "elapsed_seconds": float(time.time() - t0),
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print("Results:", output_dir)
    print(f"Elapsed: {(time.time() - t0) / 60:.1f} min")
    if RUN_CROSS_K_SCAN:
        print("Cross-K table:", output_dir / "cross_K_scan" / "cross_k_metrics.csv")


if __name__ == "__main__":
    main()
