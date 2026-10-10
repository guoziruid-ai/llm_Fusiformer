#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fusiformer latent-space vs formation-energy clustering analysis
===============================================================

Purpose
-------
This script performs clustering ANALYSIS ONLY. It does not train or fine-tune
Fusiformer.

Compared with the earlier raw/handcrafted graph-descriptor clustering script,
this version:

    crystal -> Fusiformer graph construction
            -> trained Fusiformer graph encoder
            -> AvgPooling / graph_out latent vector
            -> optional L2 normalization
            -> KMeans/MiniBatchKMeans

The formation-energy side is clustered independently in 1D, then the two
clusterings are compared with ARI / NMI / AMI / Hungarian overlap / Cramer's V,
cluster enrichment, centroid distances, and e_form distribution statistics.

IMPORTANT
---------
1. Put this file in the llm_Fusiformer project root.
2. Edit DATA_PATH and CHECKPOINT_PATH below.
3. Default LATENT_MODE="graph_only" uses the model's text_mode="bypass" path.
   This is recommended if the future cluster router should depend only on
   crystal structure.
4. LATENT_MODE="normal_cached_text" is also supported if you want the actual
   multimodal latent space and already have a pooled-Qwen embedding cache.
5. The model is always model.eval() + torch.inference_mode(); no gradients and
   no optimizer are created.

Repository assumptions
----------------------
The script reuses the project's current:
    loader.loader.CrystalGraphDataset
    loader.loader_cached.CachedCrystalGraphDataset (optional)
    loader.transforms.AddNodeFeature/AddEdgeFeature/AddAngleFeature
    models.TextGuidedFusiformerCached.TextGuidedFusiformer

The readout output is captured through a forward hook on model.readout, so the
project model source does not need to be modified.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler, normalize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project imports. Run this script from / place it in the project root.
from loader.loader import CrystalGraphDataset
from loader.transforms import (
    AddAngleFeature,
    AddEdgeFeature,
    AddNodeFeature,
    Compose,
    GraphCollecting,
)

try:
    from loader.loader_cached import CachedCrystalGraphDataset, GraphCollectingCached
except Exception:
    CachedCrystalGraphDataset = None
    GraphCollectingCached = None

try:
    from models.TextGuidedFusiformerCached import TextGuidedFusiformer
except Exception as exc:
    raise ImportError(
        "Cannot import models.TextGuidedFusiformerCached.TextGuidedFusiformer. "
        "Please place this script in the current llm_Fusiformer project root."
    ) from exc


# =============================================================================
# 0. EDITABLE CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# -----------------------------------------------------------------------------
# Dataset / checkpoint
# -----------------------------------------------------------------------------
DATA_PATH = PROJECT_ROOT / "data" / "my_dataset.json"

# Recommended: best_model_weights.pt or best_checkpoint.pt from the Base model.
# Example:
# CHECKPOINT_PATH = PROJECT_ROOT / "runs" / "fusiformer_production" / \
#     "20260811-134939" / "checkpoints" / "best_model_weights.pt"
CHECKPOINT_PATH = PROJECT_ROOT / "runs" / "fusiformer_production" / \
    "CHANGE_ME" / "checkpoints" / "best_model_weights.pt"

TARGET_FIELD = "e_form"
ID_FIELD = "mp_id"
FORMULA_FIELD = "formula"

# -----------------------------------------------------------------------------
# Which Fusiformer latent space?
# -----------------------------------------------------------------------------
# "graph_only":
#     inputs["text_mode"] = "bypass"
#     No Qwen/tokenizer/text cache is required.
#     Recommended for a future structure-only router / expert selection.
#
# "normal_cached_text":
#     Uses the model's normal text-guided path but obtains pooled Qwen vectors
#     from the existing cache; Qwen itself is still NOT loaded.
LATENT_MODE = "graph_only"

# Required only when LATENT_MODE == "normal_cached_text".
TEXT_EMBEDDING_CACHE_PATH: Optional[Path] = None
# Example:
# TEXT_EMBEDDING_CACHE_PATH = PROJECT_ROOT / "data" / "qwen_cache" / \
#     "qwen2p5_1p5b_description_meanpool_f32.npy"

# -----------------------------------------------------------------------------
# Graph construction: keep identical to the current project loader defaults.
# -----------------------------------------------------------------------------
GRAPH_CUTOFF = 8.0
GRAPH_MAX_NEIGHBORS = 12

# -----------------------------------------------------------------------------
# Model architecture overrides
# -----------------------------------------------------------------------------
# Normally keep None. The script tries checkpoint["training_config"] first,
# then infers dimensions from state_dict when possible.
DEPTH_OVERRIDE: Optional[int] = None
NUM_HEADS_OVERRIDE: Optional[int] = None
EDGE_INPUT_DIM_OVERRIDE: Optional[int] = None
TRIPLET_INPUT_DIM_OVERRIDE: Optional[int] = None
EMBED_DIM_OVERRIDE: Optional[int] = None
TEXT_DIM_OVERRIDE: Optional[int] = None

# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INFERENCE_BATCH_SIZE = 32
NUM_WORKERS = 4
PIN_MEMORY = True
RANDOM_SEED = 42

# -----------------------------------------------------------------------------
# Embedding cache
# -----------------------------------------------------------------------------
OUTPUT_ROOT = PROJECT_ROOT / "runs" / "fusiformer_latent_eform_clustering"
CACHE_DIR = OUTPUT_ROOT / "embedding_cache"
REUSE_EMBEDDING_CACHE = True
SAVE_EMBEDDING_CACHE = True

# Cache names. Metadata is checked against dataset/checkpoint/mode before reuse.
EMBEDDING_CACHE_FILE = CACHE_DIR / "fusiformer_graph_out.npy"
EMBEDDING_CACHE_META = CACHE_DIR / "fusiformer_graph_out.meta.json"

# -----------------------------------------------------------------------------
# Latent-space preprocessing BEFORE clustering
# -----------------------------------------------------------------------------
# Recommended primary experiment: "l2"
#   graph_out -> L2 normalize -> clustering
# Options: "l2", "none", "standardize"
LATENT_PREPROCESS = "l2"

# NOTE: PCA is NOT used in the primary clustering pipeline.
# PCA below is visualization only.

# -----------------------------------------------------------------------------
# Clustering
# -----------------------------------------------------------------------------
K_VALUES = list(range(2, 13))
CLUSTER_ALGORITHM = "minibatch"   # "minibatch" or "kmeans"
KMEANS_N_INIT = 20
KMEANS_MAX_ITER = 500
KMEANS_TOL = 1e-4
MINIBATCH_BATCH_SIZE = 4096

# K-selection quality metrics are calculated on a fixed subsample AFTER the
# clustering model has been fitted on the FULL dataset.
QUALITY_SAMPLE_SIZE = 10000
K_SELECTION_METRIC = "silhouette"  # silhouette / calinski_harabasz / davies_bouldin

# -----------------------------------------------------------------------------
# Experiment switches
# -----------------------------------------------------------------------------
# A: latent space and e_form independently choose their own best K.
RUN_EXPERIMENT_A = True

# Cross-K: enforce the same K on both spaces for K in K_VALUES.
RUN_CROSS_K_SCAN = True

# Detailed shared-K experiment. 5 is a useful first setting; do NOT assume it
# remains optimal after changing from raw descriptors to Fusiformer graph_out.
RUN_MAIN_SHARED_K = True
MAIN_SHARED_K = 5

# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
VIS_SAMPLE_SIZE = 30000
SAVE_LATENT_2D_PLOTS = True
SAVE_EFORM_BOXPLOT = True
SAVE_ASSIGNMENTS = True

# Optional statistical random-label baseline. 0 disables it.
# For 144k samples, start with 0; use 100+ only for final robustness analysis.
N_PERMUTATIONS = 0


# =============================================================================
# 1. GENERAL UTILITIES
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_existing(path: Path, name: str) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    return path


def file_signature(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(to_device(v, device) for v in value)
    if hasattr(value, "to"):
        try:
            return value.to(device, non_blocking=True)
        except TypeError:
            return value.to(device)
    return value


def load_torch_checkpoint(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_model_state(checkpoint: Any) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dict/state_dict.")

    training_config = checkpoint.get("training_config") or checkpoint.get("run_config") or {}

    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        state = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        state = checkpoint
    else:
        raise KeyError(
            "Checkpoint does not contain model_state_dict/model/state_dict."
        )

    if state and all(str(k).startswith("module.") for k in state.keys()):
        state = {str(k)[len("module."):]: v for k, v in state.items()}

    return dict(state), dict(training_config)


def infer_depth_from_state(state: Dict[str, torch.Tensor]) -> Optional[int]:
    indices = []
    for key in state:
        if key.startswith("blocks."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                indices.append(int(parts[1]))
    return max(indices) + 1 if indices else None


def infer_model_config(
    state: Dict[str, torch.Tensor],
    training_config: Dict[str, Any],
) -> Dict[str, int]:
    def cfg_int(name: str, override: Optional[int], fallback: Optional[int]) -> int:
        if override is not None:
            return int(override)
        if name in training_config and training_config[name] is not None:
            return int(training_config[name])
        if fallback is not None:
            return int(fallback)
        raise ValueError(
            f"Cannot infer model parameter {name!r}. Set the corresponding override."
        )

    atom_w = state.get("atom_embedding.weight")
    edge_w = state.get("edge_embedding.1.weight")
    angle_w = state.get("angle_embedding.1.weight")
    text_w = state.get("text_proj.0.weight")

    inferred_embed = int(atom_w.shape[1]) if atom_w is not None else None
    inferred_edge_bins = int(edge_w.shape[1]) if edge_w is not None else None
    inferred_angle_bins = int(angle_w.shape[1]) if angle_w is not None else None
    inferred_text_dim = int(text_w.shape[1]) if text_w is not None else None
    inferred_depth = infer_depth_from_state(state)

    # num_heads cannot be safely inferred from linear weight shapes.
    # Prefer checkpoint training_config; current project default is 4.
    default_heads = 4

    return {
        "depth": cfg_int("depth", DEPTH_OVERRIDE, inferred_depth or 4),
        "edge_input_dim": cfg_int(
            "edge_input_dim", EDGE_INPUT_DIM_OVERRIDE, inferred_edge_bins or 80
        ),
        "triplet_input_dim": cfg_int(
            "triplet_input_dim", TRIPLET_INPUT_DIM_OVERRIDE, inferred_angle_bins or 40
        ),
        "embed_dim": cfg_int("embed_dim", EMBED_DIM_OVERRIDE, inferred_embed or 128),
        "num_heads": cfg_int("num_heads", NUM_HEADS_OVERRIDE, default_heads),
        "text_dim": cfg_int("text_dim", TEXT_DIM_OVERRIDE, inferred_text_dim or 1536),
    }


def build_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    checkpoint = load_torch_checkpoint(checkpoint_path)
    state, training_config = extract_model_state(checkpoint)
    model_cfg = infer_model_config(state, training_config)

    print("\n[Model configuration]")
    for k, v in model_cfg.items():
        print(f"  {k}: {v}")

    model = TextGuidedFusiformer(
        targets=[TARGET_FIELD],
        depth=model_cfg["depth"],
        edge_input_dim=model_cfg["edge_input_dim"],
        triplet_input_dim=model_cfg["triplet_input_dim"],
        embed_dim=model_cfg["embed_dim"],
        num_heads=model_cfg["num_heads"],
        text_dim=model_cfg["text_dim"],
        qwen_model_name="unused_in_graph_only_or_cached_mode",
        freeze_qwen=True,
        load_qwen=False,
    )

    # We deliberately do not load Qwen. Filter possible qwen.* keys from older
    # checkpoints; all graph/text-projection/downstream trained tensors remain.
    filtered_state = {
        k: v for k, v in state.items()
        if not k.startswith("qwen.")
    }

    incompatible = model.load_state_dict(filtered_state, strict=False)
    missing = [k for k in incompatible.missing_keys if not k.startswith("qwen.")]
    unexpected = [k for k in incompatible.unexpected_keys if not k.startswith("qwen.")]
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/model mismatch.\n"
            f"Missing keys: {missing[:30]}\n"
            f"Unexpected keys: {unexpected[:30]}\n"
            "If your trained architecture differs from the current repository, "
            "set the architecture overrides at the top of this script."
        )

    model = model.to(device)
    model.eval()
    return model, {
        "training_config": training_config,
        "model_config": model_cfg,
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "best_epoch": checkpoint.get("best_epoch") if isinstance(checkpoint, dict) else None,
        "best_metric": checkpoint.get("best_metric") if isinstance(checkpoint, dict) else None,
    }


# =============================================================================
# 2. DATASET + FUSIFORMER LATENT EXTRACTION
# =============================================================================

def load_json_dataset(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError("Expected top-level JSON list.")
    print(f"Loaded dataset: {len(data):,} samples")

    bad = []
    for i, item in enumerate(data):
        if "atoms" not in item:
            bad.append((i, "atoms missing"))
        value = item.get(TARGET_FIELD)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value):
            bad.append((i, f"invalid {TARGET_FIELD}: {value!r}"))
        if len(bad) >= 10:
            break
    if bad:
        raise ValueError(f"Invalid dataset samples (first examples): {bad}")
    return data


def build_full_dataset(data: List[Dict[str, Any]]):
    if LATENT_MODE == "graph_only":
        transform = Compose([
            AddNodeFeature(),
            AddEdgeFeature(),
            AddAngleFeature(),
            GraphCollecting(["graph", "line_graph"], [TARGET_FIELD]),
        ])
        dataset = CrystalGraphDataset(
            data=data,
            transforms=transform,
            cutoff=GRAPH_CUTOFF,
            max_neighbors=GRAPH_MAX_NEIGHBORS,
            line_graph=True,
            text_tokenizer=None,
        )
        return dataset

    if LATENT_MODE == "normal_cached_text":
        if CachedCrystalGraphDataset is None or GraphCollectingCached is None:
            raise ImportError(
                "loader.loader_cached is required for LATENT_MODE='normal_cached_text'."
            )
        if TEXT_EMBEDDING_CACHE_PATH is None:
            raise ValueError(
                "Set TEXT_EMBEDDING_CACHE_PATH for LATENT_MODE='normal_cached_text'."
            )
        cache_path = resolve_existing(Path(TEXT_EMBEDDING_CACHE_PATH), "text embedding cache")
        arr = np.load(cache_path, mmap_mode="r")
        if arr.ndim != 2 or arr.shape[0] != len(data):
            raise ValueError(
                f"Text cache shape {arr.shape} is not aligned with dataset N={len(data)}."
            )
        del arr

        transform = Compose([
            AddNodeFeature(),
            AddEdgeFeature(),
            AddAngleFeature(),
            GraphCollectingCached(["graph", "line_graph"], [TARGET_FIELD]),
        ])
        dataset = CachedCrystalGraphDataset(
            data=data,
            transforms=transform,
            original_indices=list(range(len(data))),
            text_embedding_cache_path=cache_path,
            cutoff=GRAPH_CUTOFF,
            max_neighbors=GRAPH_MAX_NEIGHBORS,
            line_graph=True,
            text_tokenizer=None,
        )
        return dataset

    raise ValueError(
        f"Unknown LATENT_MODE={LATENT_MODE!r}; use 'graph_only' or 'normal_cached_text'."
    )


def expected_cache_meta(
    data_path: Path,
    checkpoint_path: Path,
    num_samples: int,
    embed_dim: int,
) -> Dict[str, Any]:
    meta = {
        "data": file_signature(data_path),
        "checkpoint": file_signature(checkpoint_path),
        "num_samples": int(num_samples),
        "embed_dim": int(embed_dim),
        "latent_mode": LATENT_MODE,
        "graph_cutoff": float(GRAPH_CUTOFF),
        "graph_max_neighbors": int(GRAPH_MAX_NEIGHBORS),
    }
    if LATENT_MODE == "normal_cached_text" and TEXT_EMBEDDING_CACHE_PATH is not None:
        meta["text_embedding_cache"] = file_signature(
            resolve_existing(Path(TEXT_EMBEDDING_CACHE_PATH), "text embedding cache")
        )
    return meta


def cache_is_valid(expected: Dict[str, Any]) -> bool:
    if not EMBEDDING_CACHE_FILE.is_file() or not EMBEDDING_CACHE_META.is_file():
        return False
    try:
        with EMBEDDING_CACHE_META.open("r", encoding="utf-8") as f:
            actual = json.load(f)
        if actual != expected:
            return False
        arr = np.load(EMBEDDING_CACHE_FILE, mmap_mode="r")
        ok = arr.shape == (expected["num_samples"], expected["embed_dim"])
        del arr
        return bool(ok)
    except Exception:
        return False


def extract_graph_out_embeddings(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    expected_meta: Dict[str, Any],
) -> np.ndarray:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if REUSE_EMBEDDING_CACHE and cache_is_valid(expected_meta):
        print(f"\nReusing Fusiformer graph_out cache: {EMBEDDING_CACHE_FILE}")
        return np.load(EMBEDDING_CACHE_FILE)

    print("\nExtracting Fusiformer graph_out embeddings ...")
    print(f"  latent mode: {LATENT_MODE}")
    print(f"  device: {device}")
    print(f"  batch size: {INFERENCE_BATCH_SIZE}")

    loader_kwargs = dict(
        batch_size=INFERENCE_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and device.type == "cuda",
        collate_fn=dataset.collect(),
    )
    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True

    loader = torch.utils.data.DataLoader(dataset, **loader_kwargs)

    captured: Dict[str, torch.Tensor] = {}

    def readout_hook(_module, _inputs, output):
        captured["graph_out"] = output.detach().float().cpu()

    handle = model.readout.register_forward_hook(readout_hook)
    chunks: List[np.ndarray] = []

    try:
        with torch.inference_mode():
            for batch_idx, (inputs, _targets) in enumerate(loader):
                if LATENT_MODE == "graph_only":
                    inputs["text_mode"] = "bypass"
                    # Explicitly clear any accidental text fields.
                    inputs["text"] = None
                    inputs["text_emb"] = None
                else:
                    inputs["text_mode"] = "normal"

                inputs = to_device(inputs, device)
                captured.clear()
                _ = model(inputs)
                if "graph_out" not in captured:
                    raise RuntimeError("readout hook did not capture graph_out")
                chunks.append(captured["graph_out"].numpy())

                if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(loader):
                    done = min((batch_idx + 1) * INFERENCE_BATCH_SIZE, len(dataset))
                    print(f"  encoded {done:,}/{len(dataset):,}")
    finally:
        handle.remove()

    embeddings = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    if embeddings.shape[0] != len(dataset):
        raise RuntimeError(
            f"Embedding count mismatch: {embeddings.shape[0]} != {len(dataset)}"
        )

    if SAVE_EMBEDDING_CACHE:
        np.save(EMBEDDING_CACHE_FILE, embeddings)
        json_dump(EMBEDDING_CACHE_META, expected_meta)
        print(f"Saved embedding cache: {EMBEDDING_CACHE_FILE}")

    return embeddings


# =============================================================================
# 3. CLUSTERING UTILITIES
# =============================================================================

def preprocess_latent(X: np.ndarray) -> np.ndarray:
    mode = LATENT_PREPROCESS.strip().lower()
    if mode == "l2":
        return normalize(X, norm="l2", axis=1, copy=True).astype(np.float32)
    if mode == "standardize":
        return StandardScaler().fit_transform(X).astype(np.float32)
    if mode == "none":
        return np.asarray(X, dtype=np.float32)
    raise ValueError(
        f"Unknown LATENT_PREPROCESS={LATENT_PREPROCESS!r}; use l2/standardize/none."
    )


def make_clusterer(k: int, seed: int):
    algo = CLUSTER_ALGORITHM.strip().lower()
    if algo == "kmeans":
        return KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=KMEANS_N_INIT,
            max_iter=KMEANS_MAX_ITER,
            tol=KMEANS_TOL,
            random_state=seed,
        )
    if algo == "minibatch":
        return MiniBatchKMeans(
            n_clusters=k,
            init="k-means++",
            n_init=KMEANS_N_INIT,
            max_iter=KMEANS_MAX_ITER,
            tol=KMEANS_TOL,
            batch_size=MINIBATCH_BATCH_SIZE,
            reassignment_ratio=0.01,
            random_state=seed,
        )
    raise ValueError("CLUSTER_ALGORITHM must be 'kmeans' or 'minibatch'.")


def relabel_1d_by_center(labels: np.ndarray, centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    centers = np.asarray(centers).reshape(-1)
    order = np.argsort(centers)
    mapping = {int(old): int(new) for new, old in enumerate(order)}
    relabeled = np.array([mapping[int(x)] for x in labels], dtype=np.int32)
    sorted_centers = centers[order]
    return relabeled, sorted_centers


def cluster_full(X: np.ndarray, k: int, seed: int = RANDOM_SEED):
    model = make_clusterer(k, seed)
    labels = model.fit_predict(X).astype(np.int32)
    return model, labels


def quality_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    sample_indices: np.ndarray,
) -> Dict[str, float]:
    Xs = X[sample_indices]
    ls = labels[sample_indices]
    unique = np.unique(ls)
    if len(unique) < 2:
        return {
            "silhouette": float("nan"),
            "calinski_harabasz": float("nan"),
            "davies_bouldin": float("nan"),
        }
    return {
        "silhouette": float(silhouette_score(Xs, ls, metric="euclidean")),
        "calinski_harabasz": float(calinski_harabasz_score(Xs, ls)),
        "davies_bouldin": float(davies_bouldin_score(Xs, ls)),
    }


def cluster_size_metrics(labels: np.ndarray, k: int) -> Dict[str, float]:
    counts = np.bincount(labels, minlength=k).astype(float)
    probs = counts / counts.sum()
    nz = probs[probs > 0]
    entropy = -np.sum(nz * np.log(nz))
    normalized_entropy = entropy / math.log(k) if k > 1 else 0.0
    return {
        "min_cluster_size": int(counts.min()),
        "max_cluster_size": int(counts.max()),
        "min_cluster_fraction": float(probs.min()),
        "max_cluster_fraction": float(probs.max()),
        "cluster_size_entropy_norm": float(normalized_entropy),
    }


def contingency_counts(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_values = np.unique(a)
    b_values = np.unique(b)
    mat = np.zeros((len(a_values), len(b_values)), dtype=np.int64)
    a_map = {int(v): i for i, v in enumerate(a_values)}
    b_map = {int(v): i for i, v in enumerate(b_values)}
    for x, y in zip(a, b):
        mat[a_map[int(x)], b_map[int(y)]] += 1
    return mat


def correspondence_metrics(
    graph_labels: np.ndarray,
    eform_labels: np.ndarray,
) -> Dict[str, float]:
    n = len(graph_labels)
    mat = contingency_counts(graph_labels, eform_labels)

    ari = adjusted_rand_score(graph_labels, eform_labels)
    nmi = normalized_mutual_info_score(graph_labels, eform_labels)
    ami = adjusted_mutual_info_score(graph_labels, eform_labels)

    row_ind, col_ind = linear_sum_assignment(-mat)
    hungarian_overlap = mat[row_ind, col_ind].sum() / n
    purity = mat.max(axis=1).sum() / n

    chi2, chi2_p, _, _ = chi2_contingency(mat, correction=False)
    denom = n * min(mat.shape[0] - 1, mat.shape[1] - 1)
    cramers_v = math.sqrt(chi2 / denom) if denom > 0 else float("nan")

    return {
        "n_samples": int(n),
        "graph_k": int(len(np.unique(graph_labels))),
        "eform_k": int(len(np.unique(eform_labels))),
        "ARI": float(ari),
        "NMI": float(nmi),
        "AMI": float(ami),
        "hungarian_overlap": float(hungarian_overlap),
        "graph_to_eform_purity": float(purity),
        "chi2": float(chi2),
        "chi2_p_value": float(chi2_p),
        "cramers_v": float(cramers_v),
    }


def eta_squared(labels: np.ndarray, y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    grand = float(y.mean())
    ss_total = float(np.sum((y - grand) ** 2))
    if ss_total <= 0:
        return float("nan")
    ss_between = 0.0
    for c in np.unique(labels):
        vals = y[labels == c]
        ss_between += len(vals) * (float(vals.mean()) - grand) ** 2
    return float(ss_between / ss_total)


def select_best_k(table: pd.DataFrame, prefix: str) -> int:
    metric = K_SELECTION_METRIC.strip().lower()
    col = f"{prefix}_{metric}"
    if col not in table.columns:
        raise KeyError(f"Missing K-selection column: {col}")
    if metric == "davies_bouldin":
        idx = table[col].astype(float).idxmin()
    else:
        idx = table[col].astype(float).idxmax()
    return int(table.loc[idx, "K"])


def permutation_baseline(
    graph_labels: np.ndarray,
    eform_labels: np.ndarray,
    n_perm: int,
    seed: int,
) -> Dict[str, float]:
    if n_perm <= 0:
        return {}
    rng = np.random.default_rng(seed)
    ari_vals, nmi_vals, ami_vals = [], [], []
    for _ in range(n_perm):
        shuffled = rng.permutation(eform_labels)
        ari_vals.append(adjusted_rand_score(graph_labels, shuffled))
        nmi_vals.append(normalized_mutual_info_score(graph_labels, shuffled))
        ami_vals.append(adjusted_mutual_info_score(graph_labels, shuffled))
    return {
        "perm_n": int(n_perm),
        "perm_ari_mean": float(np.mean(ari_vals)),
        "perm_ari_std": float(np.std(ari_vals)),
        "perm_nmi_mean": float(np.mean(nmi_vals)),
        "perm_nmi_std": float(np.std(nmi_vals)),
        "perm_ami_mean": float(np.mean(ami_vals)),
        "perm_ami_std": float(np.std(ami_vals)),
    }


# =============================================================================
# 4. PLOTS / DETAILED OUTPUTS
# =============================================================================

def save_matrix_csv_and_plot(
    matrix: np.ndarray,
    path_csv: Path,
    path_png: Path,
    title: str,
    fmt: str,
    row_prefix: str = "G",
    col_prefix: str = "E",
) -> None:
    df = pd.DataFrame(
        matrix,
        index=[f"{row_prefix}{i}" for i in range(matrix.shape[0])],
        columns=[f"{col_prefix}{j}" for j in range(matrix.shape[1])],
    )
    df.to_csv(path_csv, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(max(6, matrix.shape[1] * 1.1), max(5, matrix.shape[0] * 0.9)))
    im = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticklabels(df.columns)
    ax.set_yticklabels(df.index)
    ax.set_xlabel("Formation-energy cluster")
    ax.set_ylabel("Fusiformer latent cluster")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)

    threshold = float(np.nanmax(matrix)) * 0.55 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = format(value, fmt)
            ax.text(j, i, text, ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def detailed_shared_k_outputs(
    out_dir: Path,
    data: List[Dict[str, Any]],
    X_latent: np.ndarray,
    eform_raw: np.ndarray,
    eform_scaled: np.ndarray,
    graph_labels: np.ndarray,
    eform_labels: np.ndarray,
    eform_centers_scaled: np.ndarray,
    eform_scaler: StandardScaler,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    k_g = len(np.unique(graph_labels))
    k_e = len(np.unique(eform_labels))

    metrics = correspondence_metrics(graph_labels, eform_labels)
    metrics["eta_squared_eform_by_graph_cluster"] = eta_squared(graph_labels, eform_raw)
    metrics.update(permutation_baseline(
        graph_labels, eform_labels, N_PERMUTATIONS, RANDOM_SEED
    ))
    pd.DataFrame([metrics]).to_csv(
        out_dir / "correspondence_metrics.csv", index=False, encoding="utf-8-sig"
    )

    counts = contingency_counts(graph_labels, eform_labels)
    row_sums = counts.sum(axis=1, keepdims=True)
    row_frac = np.divide(
        counts,
        row_sums,
        out=np.zeros_like(counts, dtype=float),
        where=row_sums > 0,
    )
    priors = counts.sum(axis=0, keepdims=True) / counts.sum()
    lift = np.divide(
        row_frac,
        priors,
        out=np.zeros_like(row_frac, dtype=float),
        where=priors > 0,
    )

    save_matrix_csv_and_plot(
        counts,
        out_dir / "graph_vs_eform_contingency.csv",
        out_dir / "contingency_heatmap.png",
        "Fusiformer latent cluster vs e_form cluster: counts",
        "d",
    )
    save_matrix_csv_and_plot(
        row_frac,
        out_dir / "graph_to_eform_row_fraction.csv",
        out_dir / "contingency_row_fraction.png",
        "P(e_form cluster | Fusiformer latent cluster)",
        ".2f",
    )
    save_matrix_csv_and_plot(
        lift,
        out_dir / "graph_to_eform_lift.csv",
        out_dir / "enrichment_lift_heatmap.png",
        "Enrichment lift = P(Ej|Gi) / P(Ej)",
        ".2f",
    )

    # Graph-cluster e_form statistics.
    rows = []
    graph_centers_scaled = []
    for g in range(k_g):
        vals = eform_raw[graph_labels == g]
        vals_scaled = eform_scaled[graph_labels == g, 0]
        graph_centers_scaled.append(float(vals_scaled.mean()))
        rows.append({
            "graph_cluster": g,
            "n": int(len(vals)),
            "fraction": float(len(vals) / len(eform_raw)),
            "mean_e_form": float(np.mean(vals)),
            "median_e_form": float(np.median(vals)),
            "std_e_form": float(np.std(vals)),
            "min_e_form": float(np.min(vals)),
            "q05_e_form": float(np.percentile(vals, 5)),
            "q25_e_form": float(np.percentile(vals, 25)),
            "q75_e_form": float(np.percentile(vals, 75)),
            "q95_e_form": float(np.percentile(vals, 95)),
            "max_e_form": float(np.max(vals)),
        })
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(
        out_dir / "graph_cluster_eform_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    eform_centers_original = eform_scaler.inverse_transform(
        np.asarray(eform_centers_scaled).reshape(-1, 1)
    ).reshape(-1)
    pd.DataFrame({
        "eform_cluster": np.arange(k_e),
        "center_scaled": eform_centers_scaled,
        "center_eV_per_atom": eform_centers_original,
        "n": np.bincount(eform_labels, minlength=k_e),
    }).to_csv(
        out_dir / "eform_cluster_centers.csv", index=False, encoding="utf-8-sig"
    )

    graph_centers_scaled = np.asarray(graph_centers_scaled)
    graph_centers_original = eform_scaler.inverse_transform(
        graph_centers_scaled.reshape(-1, 1)
    ).reshape(-1)
    pd.DataFrame({
        "graph_cluster": np.arange(k_g),
        "center_scaled_in_eform_space": graph_centers_scaled,
        "mean_e_form_eV_per_atom": graph_centers_original,
        "n": np.bincount(graph_labels, minlength=k_g),
    }).to_csv(
        out_dir / "graph_cluster_eform_centers.csv",
        index=False,
        encoding="utf-8-sig",
    )

    centroid_dist = np.abs(
        graph_centers_scaled[:, None] - np.asarray(eform_centers_scaled)[None, :]
    )
    save_matrix_csv_and_plot(
        centroid_dist,
        out_dir / "centroid_distance_standardized.csv",
        out_dir / "centroid_distance_heatmap.png",
        "Centroid distance in standardized e_form space",
        ".2f",
    )
    r, c = linear_sum_assignment(centroid_dist)
    pd.DataFrame({
        "graph_cluster": r,
        "eform_cluster": c,
        "distance_standardized": centroid_dist[r, c],
    }).to_csv(
        out_dir / "centroid_hungarian_matching.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if SAVE_ASSIGNMENTS:
        assignment = pd.DataFrame({
            "index": np.arange(len(data)),
            "mp_id": [item.get(ID_FIELD, "") for item in data],
            "formula": [item.get(FORMULA_FIELD, "") for item in data],
            "e_form": eform_raw,
            "fusiformer_cluster": graph_labels,
            "eform_cluster": eform_labels,
        })
        assignment.to_csv(
            out_dir / "assignments.csv", index=False, encoding="utf-8-sig"
        )

    if SAVE_EFORM_BOXPLOT:
        groups = [eform_raw[graph_labels == g] for g in range(k_g)]
        fig, ax = plt.subplots(figsize=(max(7, k_g * 1.1), 5.5))
        ax.boxplot(groups, labels=[f"G{g}" for g in range(k_g)], showfliers=False)
        ax.set_xlabel("Fusiformer latent cluster")
        ax.set_ylabel("e_form (eV/atom)")
        ax.set_title("Formation-energy distribution by Fusiformer latent cluster")
        fig.tight_layout()
        fig.savefig(out_dir / "e_form_by_graph_cluster_boxplot.png", dpi=220)
        plt.close(fig)

    if SAVE_LATENT_2D_PLOTS:
        n_vis = min(VIS_SAMPLE_SIZE, len(X_latent))
        rng = np.random.default_rng(RANDOM_SEED)
        idx = np.sort(rng.choice(len(X_latent), size=n_vis, replace=False))
        X2 = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(X_latent[idx])

        fig, ax = plt.subplots(figsize=(7.2, 6))
        sc = ax.scatter(X2[:, 0], X2[:, 1], c=graph_labels[idx], s=5, alpha=0.6)
        ax.set_xlabel("PCA-1 (visualization only)")
        ax.set_ylabel("PCA-2 (visualization only)")
        ax.set_title("Fusiformer graph_out colored by latent cluster")
        plt.colorbar(sc, ax=ax, label="latent cluster")
        fig.tight_layout()
        fig.savefig(out_dir / "latent_pca2_colored_by_graph_cluster.png", dpi=220)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.2, 6))
        sc = ax.scatter(X2[:, 0], X2[:, 1], c=eform_raw[idx], s=5, alpha=0.6)
        ax.set_xlabel("PCA-1 (visualization only)")
        ax.set_ylabel("PCA-2 (visualization only)")
        ax.set_title("Fusiformer graph_out colored by e_form")
        plt.colorbar(sc, ax=ax, label="e_form (eV/atom)")
        fig.tight_layout()
        fig.savefig(out_dir / "latent_pca2_colored_by_eform.png", dpi=220)
        plt.close(fig)


# =============================================================================
# 5. MAIN EXPERIMENT
# =============================================================================

def main() -> None:
    set_seed(RANDOM_SEED)

    data_path = resolve_existing(Path(DATA_PATH), "dataset")
    checkpoint_path = resolve_existing(Path(CHECKPOINT_PATH), "checkpoint")

    if MAIN_SHARED_K not in K_VALUES and RUN_MAIN_SHARED_K:
        raise ValueError("MAIN_SHARED_K must be included in K_VALUES.")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)

    print("=" * 80)
    print("Fusiformer latent-space vs formation-energy clustering")
    print("=" * 80)
    print(f"Dataset:    {data_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Run dir:    {run_dir}")
    print(f"Latent mode:{LATENT_MODE}")

    data = load_json_dataset(data_path)
    eform_raw = np.asarray([float(x[TARGET_FIELD]) for x in data], dtype=np.float64)

    device = torch.device(DEVICE)
    model, model_info = build_model_from_checkpoint(checkpoint_path, device)
    embed_dim = int(model.atom_embedding.embedding_dim)

    dataset = build_full_dataset(data)
    expected_meta = expected_cache_meta(
        data_path, checkpoint_path, len(data), embed_dim
    )
    embeddings = extract_graph_out_embeddings(
        model, dataset, device, expected_meta
    )

    # Free GPU memory before clustering.
    del dataset
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not np.all(np.isfinite(embeddings)):
        raise ValueError("Fusiformer embeddings contain NaN/Inf.")

    X_latent = preprocess_latent(embeddings)
    eform_scaler = StandardScaler()
    X_eform = eform_scaler.fit_transform(eform_raw.reshape(-1, 1)).astype(np.float32)

    # Save experiment configuration.
    run_config = {
        "dataset": file_signature(data_path),
        "checkpoint": file_signature(checkpoint_path),
        "target": TARGET_FIELD,
        "latent_mode": LATENT_MODE,
        "latent_preprocess": LATENT_PREPROCESS,
        "graph_cutoff": GRAPH_CUTOFF,
        "graph_max_neighbors": GRAPH_MAX_NEIGHBORS,
        "n_samples": len(data),
        "embedding_shape": list(embeddings.shape),
        "cluster_algorithm": CLUSTER_ALGORITHM,
        "k_values": K_VALUES,
        "quality_sample_size": QUALITY_SAMPLE_SIZE,
        "k_selection_metric": K_SELECTION_METRIC,
        "main_shared_k": MAIN_SHARED_K,
        "random_seed": RANDOM_SEED,
        "model_info": model_info,
    }
    json_dump(run_dir / "analysis_config.json", run_config)

    pd.DataFrame({
        "statistic": ["n", "min", "p01", "p05", "p25", "median", "mean", "p75", "p95", "p99", "max", "std"],
        "value": [
            len(eform_raw),
            np.min(eform_raw),
            np.percentile(eform_raw, 1),
            np.percentile(eform_raw, 5),
            np.percentile(eform_raw, 25),
            np.median(eform_raw),
            np.mean(eform_raw),
            np.percentile(eform_raw, 75),
            np.percentile(eform_raw, 95),
            np.percentile(eform_raw, 99),
            np.max(eform_raw),
            np.std(eform_raw),
        ],
    }).to_csv(run_dir / "eform_distribution_summary.csv", index=False, encoding="utf-8-sig")

    # Fixed quality sample for fair K-to-K comparison.
    rng = np.random.default_rng(RANDOM_SEED)
    n_q = min(QUALITY_SAMPLE_SIZE, len(data))
    quality_idx = np.sort(rng.choice(len(data), size=n_q, replace=False))

    graph_labels_by_k: Dict[int, np.ndarray] = {}
    eform_labels_by_k: Dict[int, np.ndarray] = {}
    eform_centers_by_k: Dict[int, np.ndarray] = {}
    scan_rows: List[Dict[str, Any]] = []

    # Experiment A or Cross-K needs the complete K range. If both are disabled,
    # a main shared-K run only fits the requested K, which is much faster when
    # reusing the graph_out cache.
    if RUN_EXPERIMENT_A or RUN_CROSS_K_SCAN:
        active_k_values = list(K_VALUES)
    elif RUN_MAIN_SHARED_K:
        active_k_values = [int(MAIN_SHARED_K)]
    else:
        raise ValueError("At least one experiment switch must be True.")

    print("\nRunning clustering for K values:", active_k_values)
    for k in active_k_values:
        print(f"  K={k}")

        graph_model, graph_labels = cluster_full(X_latent, k)
        eform_model, eform_labels_raw = cluster_full(X_eform, k)
        eform_labels, eform_centers = relabel_1d_by_center(
            eform_labels_raw,
            eform_model.cluster_centers_.reshape(-1),
        )

        graph_labels_by_k[k] = graph_labels
        eform_labels_by_k[k] = eform_labels
        eform_centers_by_k[k] = eform_centers

        qg = quality_metrics(X_latent, graph_labels, quality_idx)
        qe = quality_metrics(X_eform, eform_labels, quality_idx)
        sg = cluster_size_metrics(graph_labels, k)
        se = cluster_size_metrics(eform_labels, k)
        corr = correspondence_metrics(graph_labels, eform_labels)

        row = {"K": k}
        row.update({f"graph_{name}": val for name, val in qg.items()})
        row.update({f"eform_{name}": val for name, val in qe.items()})
        row.update({f"graph_{name}": val for name, val in sg.items()})
        row.update({f"eform_{name}": val for name, val in se.items()})
        row.update({name: val for name, val in corr.items() if name not in {"graph_k", "eform_k"}})
        row["eta_squared_eform_by_graph_cluster"] = eta_squared(graph_labels, eform_raw)
        scan_rows.append(row)

    scan_df = pd.DataFrame(scan_rows)
    metrics_name = "cross_k_metrics.csv" if len(active_k_values) > 1 else "shared_k_metrics.csv"
    scan_df.to_csv(run_dir / metrics_name, index=False, encoding="utf-8-sig")

    # Cross-K plots are meaningful only when multiple K values were evaluated.
    if len(active_k_values) > 1:
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        for col in ["ARI", "NMI", "AMI", "cramers_v", "hungarian_overlap"]:
            ax.plot(scan_df["K"], scan_df[col], marker="o", label=col)
        ax.set_xlabel("Shared K")
        ax.set_ylabel("Score")
        ax.set_title("Fusiformer latent vs e_form correspondence across K")
        ax.set_xticks(active_k_values)
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / "correspondence_vs_k.png", dpi=220)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ax.plot(scan_df["K"], scan_df["graph_silhouette"], marker="o", label="graph_out")
        ax.plot(scan_df["K"], scan_df["eform_silhouette"], marker="o", label="e_form")
        ax.set_xlabel("K")
        ax.set_ylabel("Silhouette")
        ax.set_title("Cluster quality across K")
        ax.set_xticks(active_k_values)
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / "silhouette_vs_k.png", dpi=220)
        plt.close(fig)

    # Experiment A: each space chooses its own K using exactly the same FULL-data
    # fitting protocol as the Cross-K scan.
    if RUN_EXPERIMENT_A:
        graph_k = select_best_k(scan_df, "graph")
        eform_k = select_best_k(scan_df, "eform")
        graph_labels = graph_labels_by_k[graph_k]
        eform_labels = eform_labels_by_k[eform_k]

        expa_dir = run_dir / "experiment_A_independent_K"
        expa_dir.mkdir(parents=True, exist_ok=True)
        metrics = correspondence_metrics(graph_labels, eform_labels)
        metrics["graph_selected_k"] = graph_k
        metrics["eform_selected_k"] = eform_k
        metrics["selection_metric"] = K_SELECTION_METRIC
        metrics["eta_squared_eform_by_graph_cluster"] = eta_squared(graph_labels, eform_raw)
        pd.DataFrame([metrics]).to_csv(
            expa_dir / "correspondence_metrics.csv", index=False, encoding="utf-8-sig"
        )
        json_dump(expa_dir / "experiment_A_selected_K.json", metrics)
        print(
            f"\nExperiment A: graph K={graph_k}, e_form K={eform_k}, "
            f"ARI={metrics['ARI']:.4f}, AMI={metrics['AMI']:.4f}, "
            f"Cramer's V={metrics['cramers_v']:.4f}"
        )

    if RUN_MAIN_SHARED_K:
        k = int(MAIN_SHARED_K)
        detail_dir = run_dir / f"experiment_B_same_K_{k:02d}"
        detailed_shared_k_outputs(
            detail_dir,
            data,
            X_latent,
            eform_raw,
            X_eform,
            graph_labels_by_k[k],
            eform_labels_by_k[k],
            eform_centers_by_k[k],
            eform_scaler,
        )
        m = correspondence_metrics(graph_labels_by_k[k], eform_labels_by_k[k])
        eta2 = eta_squared(graph_labels_by_k[k], eform_raw)
        print(
            f"\nShared K={k}: ARI={m['ARI']:.4f}, NMI={m['NMI']:.4f}, "
            f"AMI={m['AMI']:.4f}, Cramer's V={m['cramers_v']:.4f}, "
            f"eta^2={eta2:.4f}"
        )

    if not RUN_CROSS_K_SCAN and RUN_EXPERIMENT_A:
        (run_dir / "cross_k_scan_disabled_note.txt").write_text(
            "RUN_CROSS_K_SCAN=False, but the K range was evaluated internally "
            "because Experiment A needs a consistent full-data K-selection protocol.\n",
            encoding="utf-8",
        )

    print("\nDone.")
    print(f"Results: {run_dir}")
    print(f"Embedding cache: {EMBEDDING_CACHE_FILE}")


if __name__ == "__main__":
    main()
