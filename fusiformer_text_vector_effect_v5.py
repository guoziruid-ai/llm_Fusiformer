#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fusiformer 文本效用诊断 v5
==========================

目标
----
针对当前 MatSciBERT + MLP 投影 + Text-Guided Fusiformer 模型，使用同一个已训练
checkpoint 做后处理诊断，回答以下问题：

1. 正确匹配文本是否真正改善预测，而不只是让预测发生变化？
2. normal -> wrong-text 的变化是否显著高于 wrong-text A -> wrong-text B 的随机背景？
3. Projected Integrated Gradients (IG) 是否满足 completeness，并在积分步数增加时收敛？
4. 64D（或模型实际维度）投影通道的作用是否稳定、是否存在明显正负抵消？
5. 高归因通道做 Top-K 遮挡时，是否比随机通道引起更大预测变化（faithfulness）？
6. 当前模型的真实结构、维度和参数量是否与训练配置/checkpoint一致？

本程序不训练模型、不修改 checkpoint。

当前已知 Fusiformer v6 基线结构（程序会运行时自动检测并覆盖记录）：
    Crystal graph + line graph
      atom number -> Embedding(95 -> graph_dim)
      distance    -> RBF(80) -> Linear -> LayerNorm -> ReLU
      angle       -> RBF(40) -> Linear -> LayerNorm -> ReLU

    MatSciBERT pooled embedding (通常 768D, 离线 cache)
      -> Linear 768->256 -> LN -> GELU -> Dropout
      -> Linear 256->128 -> LN -> GELU -> Dropout
      -> Linear 128->64  -> LN

    Fusiformer Block x depth
      每个 block: line-graph update -> main-graph update
      每个 update: text guidance + graph cross-attention + edge-conditioned attention + MLP

    AvgPooling(graph nodes) -> regression head 128->64->1

运行方式
--------
1. 将本文件放在 /root/llm_fusiformer/ 项目根目录。
2. 建议显式填写 CHECKPOINT_PATH。
3. 运行：
       python fusiformer_text_effect_diagnostics_v3.py

依赖
----
仅使用项目已有依赖：PyTorch / NumPy / matplotlib / DGL（由训练模块导入）。
无需 scipy / captum。
"""

from __future__ import annotations

import csv
import importlib
import json
import logging
import math
import os
import random
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from types import MethodType
from typing import Any, Iterable, Optional

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:128",
)

import numpy as np
import torch

try:
    from engine import to_device as project_to_device
except Exception:
    project_to_device = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# 0. 可编辑配置
# =============================================================================
BASE_PATH = Path(__file__).resolve().parent

# 优先使用 v6；若服务器之后把正式训练入口改名，可在此修改。
TRAINING_MODULE_NAME = "train_fusiformer_production_editor_v6"

# 强烈建议显式指定本次解释的 best checkpoint。
# None 时从训练模块 output_root 下寻找最近修改的 best_checkpoint.pt。
CHECKPOINT_PATH: Optional[str] = (
    "./runs/fusiformer_production/20260811-134939/checkpoints/best_checkpoint.pt"
)
# 示例：
# CHECKPOINT_PATH = (
#     "./runs/fusiformer_production_matscibert_mlp64/"
#     "20260823-134654/checkpoints/best_checkpoint.pt"
# )

OUTPUT_ROOT = "./runs/text_vector_effect"
EXPERIMENT_NAME: Optional[str] = None

# 数据 split：test | validation | val
EVAL_SPLIT = "test"
DEVICE_NAME = "cuda"
SCAN_BATCH_SIZE = 32
NUM_WORKERS_OVERRIDE: Optional[int] = 4
PIN_MEMORY = True
RANDOM_SEED = 42

# 推理可按训练设置开 AMP；IG 建议始终 FP32。
USE_AMP_FOR_PREDICTION = False
USE_AMP_FOR_IG = False

# -----------------------------------------------------------------------------
# 1. 全集文本效用：normal / bypass / repeated wrong-text / null
# -----------------------------------------------------------------------------
RUN_PREDICTION_UTILITY = True
PREDICTION_SHUFFLE_REPEATS = 20

# global:
#   从整个当前 split 随机错配，保证 donor != 自身。
# different_composition:
#   优先保证 donor composition != 当前样本，减少“随机错配但语义仍近似”的情况。
# within_composition:
#   只在同 composition 内错配（组内不足会 fallback 到 global），属于更严格的受控实验。
SHUFFLE_MODE = "global"
SHUFFLE_RANDOM_SEED = 2026
NULL_RANDOM_SEED = 12026

# 对 MAE improvement 做 paired bootstrap CI 和 sign-flip permutation test。
RUN_STATISTICAL_INFERENCE = True
BOOTSTRAP_REPEATS = 3000
PERMUTATION_REPEATS = 5000
CONFIDENCE_LEVEL = 0.95

# -----------------------------------------------------------------------------
# 2. Projected differential IG（主诊断）
# -----------------------------------------------------------------------------
RUN_PROJECTED_IG = True

# stratified_signal：覆盖从低到高文本响应区间，避免只挑 Top 样本导致选择偏差。
# top_prediction_signal / top_text_help / random / manual
SAMPLE_SELECTION_MODE = "stratified_signal"
NUM_SAMPLES_TO_ANALYZE = 24
MANUAL_SAMPLE_INDICES = [0, 100, 1000]

# 每个样本使用多少组 wrong A / wrong B。
IG_SHUFFLE_REPEATS = 5

# gauss_legendre 通常比相同步数 midpoint 更快收敛。
IG_INTEGRATION_METHOD = "gauss_legendre"  # gauss_legendre | midpoint
IG_STEPS = 64
IG_CONVERGENCE_STEPS = [16, 32, 64, 128]
IG_CONVERGENCE_SAMPLES = 4

# prediction delta 很小时，相对 completeness error 会失真；仍保存绝对误差。
COMPLETENESS_MIN_DELTA = 1e-6

# raw MatSciBERT 768D IG 成本更高且不直接对应融合通道，默认关闭。
RUN_RAW_IG = False

# -----------------------------------------------------------------------------
# 3. 归因 faithfulness / 路径检查
# -----------------------------------------------------------------------------
RUN_TOPK_FAITHFULNESS = True
PROJECTED_TOPK = [1, 2, 4, 8, 16, 32]
RANDOM_OCCLUSION_REPEATS = 20

RUN_INTERPOLATION_PATH = True
INTERPOLATION_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]

# -----------------------------------------------------------------------------
# 4. 输出
# -----------------------------------------------------------------------------
SAVE_PLOTS = True
SAVE_IG_STACKS = True
HEATMAP_DPI = 300
HEATMAP_MAX_SAMPLES = 80
LOG_INTERVAL = 50
EPS = 1e-12


# =============================================================================
# 1. 配置对象
# =============================================================================
@dataclass
class Config:
    training_module_name: str
    checkpoint_path: Optional[Path]
    output_root: Path
    experiment_name: Optional[str]
    eval_split: str
    device_name: str
    scan_batch_size: int
    num_workers_override: Optional[int]
    pin_memory: bool
    seed: int
    use_amp_prediction: bool
    use_amp_ig: bool

    run_prediction_utility: bool
    prediction_shuffle_repeats: int
    shuffle_mode: str
    shuffle_seed: int
    null_seed: int
    run_statistical_inference: bool
    bootstrap_repeats: int
    permutation_repeats: int
    confidence_level: float

    run_projected_ig: bool
    sample_selection_mode: str
    num_samples_to_analyze: int
    manual_sample_indices: list[int]
    ig_shuffle_repeats: int
    ig_integration_method: str
    ig_steps: int
    ig_convergence_steps: list[int]
    ig_convergence_samples: int
    completeness_min_delta: float
    run_raw_ig: bool

    run_topk_faithfulness: bool
    projected_topk: list[int]
    random_occlusion_repeats: int
    run_interpolation_path: bool
    interpolation_alphas: list[float]

    save_plots: bool
    save_ig_stacks: bool
    heatmap_dpi: int
    heatmap_max_samples: int
    log_interval: int


def resolve_path(value: str | Path | None) -> Optional[Path]:
    if value is None:
        return None
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = BASE_PATH / p
    return p.resolve()


def build_config() -> Config:
    return Config(
        training_module_name=str(TRAINING_MODULE_NAME),
        checkpoint_path=resolve_path(CHECKPOINT_PATH),
        output_root=resolve_path(OUTPUT_ROOT),
        experiment_name=EXPERIMENT_NAME,
        eval_split=str(EVAL_SPLIT).strip().lower(),
        device_name=str(DEVICE_NAME),
        scan_batch_size=int(SCAN_BATCH_SIZE),
        num_workers_override=(
            None if NUM_WORKERS_OVERRIDE is None else int(NUM_WORKERS_OVERRIDE)
        ),
        pin_memory=bool(PIN_MEMORY),
        seed=int(RANDOM_SEED),
        use_amp_prediction=bool(USE_AMP_FOR_PREDICTION),
        use_amp_ig=bool(USE_AMP_FOR_IG),
        run_prediction_utility=bool(RUN_PREDICTION_UTILITY),
        prediction_shuffle_repeats=int(PREDICTION_SHUFFLE_REPEATS),
        shuffle_mode=str(SHUFFLE_MODE).strip().lower(),
        shuffle_seed=int(SHUFFLE_RANDOM_SEED),
        null_seed=int(NULL_RANDOM_SEED),
        run_statistical_inference=bool(RUN_STATISTICAL_INFERENCE),
        bootstrap_repeats=int(BOOTSTRAP_REPEATS),
        permutation_repeats=int(PERMUTATION_REPEATS),
        confidence_level=float(CONFIDENCE_LEVEL),
        run_projected_ig=bool(RUN_PROJECTED_IG),
        sample_selection_mode=str(SAMPLE_SELECTION_MODE).strip().lower(),
        num_samples_to_analyze=int(NUM_SAMPLES_TO_ANALYZE),
        manual_sample_indices=[int(x) for x in MANUAL_SAMPLE_INDICES],
        ig_shuffle_repeats=int(IG_SHUFFLE_REPEATS),
        ig_integration_method=str(IG_INTEGRATION_METHOD).strip().lower(),
        ig_steps=int(IG_STEPS),
        ig_convergence_steps=[int(x) for x in IG_CONVERGENCE_STEPS],
        ig_convergence_samples=int(IG_CONVERGENCE_SAMPLES),
        completeness_min_delta=float(COMPLETENESS_MIN_DELTA),
        run_raw_ig=bool(RUN_RAW_IG),
        run_topk_faithfulness=bool(RUN_TOPK_FAITHFULNESS),
        projected_topk=[int(x) for x in PROJECTED_TOPK],
        random_occlusion_repeats=int(RANDOM_OCCLUSION_REPEATS),
        run_interpolation_path=bool(RUN_INTERPOLATION_PATH),
        interpolation_alphas=[float(x) for x in INTERPOLATION_ALPHAS],
        save_plots=bool(SAVE_PLOTS),
        save_ig_stacks=bool(SAVE_IG_STACKS),
        heatmap_dpi=int(HEATMAP_DPI),
        heatmap_max_samples=int(HEATMAP_MAX_SAMPLES),
        log_interval=int(LOG_INTERVAL),
    )


def validate_config(cfg: Config) -> None:
    if cfg.eval_split not in {"test", "validation", "val"}:
        raise ValueError("EVAL_SPLIT must be test / validation / val")
    if cfg.shuffle_mode not in {"global", "different_composition", "within_composition"}:
        raise ValueError(
            "SHUFFLE_MODE must be global / different_composition / within_composition"
        )
    if cfg.prediction_shuffle_repeats < 2:
        raise ValueError("PREDICTION_SHUFFLE_REPEATS must be >= 2")
    if cfg.ig_shuffle_repeats < 2:
        raise ValueError("IG_SHUFFLE_REPEATS must be >= 2")
    if cfg.ig_integration_method not in {"gauss_legendre", "midpoint"}:
        raise ValueError("IG_INTEGRATION_METHOD must be gauss_legendre or midpoint")
    if cfg.ig_steps < 1:
        raise ValueError("IG_STEPS must be positive")
    if not cfg.ig_convergence_steps or min(cfg.ig_convergence_steps) < 1:
        raise ValueError("IG_CONVERGENCE_STEPS must contain positive values")
    if not (0 < cfg.confidence_level < 1):
        raise ValueError("CONFIDENCE_LEVEL must be in (0,1)")
    if cfg.bootstrap_repeats < 100:
        raise ValueError("BOOTSTRAP_REPEATS should be >= 100")
    if cfg.permutation_repeats < 100:
        raise ValueError("PERMUTATION_REPEATS should be >= 100")
    if cfg.completeness_min_delta <= 0:
        raise ValueError("COMPLETENESS_MIN_DELTA must be > 0")
    valid_selection = {
        "stratified_signal",
        "top_prediction_signal",
        "top_text_help",
        "random",
        "manual",
    }
    if cfg.sample_selection_mode not in valid_selection:
        raise ValueError(
            f"SAMPLE_SELECTION_MODE must be one of {sorted(valid_selection)}"
        )


# =============================================================================
# 2. 基础工具
# =============================================================================
def setup_logger(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("fusiformer_text_effect_diagnostics_v3")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(run_dir / "diagnostic.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: Optional[list[str]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(value: Any, device: torch.device) -> Any:
    if project_to_device is not None:
        try:
            return project_to_device(value, device)
        except TypeError:
            pass
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(to_device(v, device) for v in value)
    if hasattr(value, "to"):
        return value.to(device, non_blocking=True)
    return value


def normalize_outputs(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, dict):
        for key in ("prediction", "pred", "output"):
            if key in outputs:
                return outputs[key]
        raise KeyError("model returned dict but no prediction/pred/output key found")
    if isinstance(outputs, (list, tuple)):
        return torch.cat(list(outputs), dim=-1)
    return outputs


def normalize_targets(targets: Any, device: torch.device) -> torch.Tensor:
    if isinstance(targets, list):
        targets = torch.cat(targets, dim=-1)
    elif isinstance(targets, tuple):
        targets = torch.cat(list(targets), dim=-1)
    return targets.to(device, non_blocking=True)


def selected_output(outputs: Any, target_index: int = 0) -> torch.Tensor:
    y = normalize_outputs(outputs)
    if y.ndim == 0:
        return y.reshape(1)
    if y.ndim == 1:
        return y
    return y[:, target_index]


def scalar_selected_output(outputs: Any, target_index: int = 0) -> torch.Tensor:
    y = selected_output(outputs, target_index)
    if y.numel() != 1:
        raise ValueError(f"expected scalar output, got shape={tuple(y.shape)}")
    return y.reshape(())


def safe_get_id(info: dict[str, Any]) -> str:
    for key in ["mp_id", "mp-id", "material_id", "jid", "id", "task_id"]:
        if key in info and info[key] is not None:
            return str(info[key])
    return "unknown"


def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def clone_inputs_with_mode(inputs: dict[str, Any], mode: str) -> dict[str, Any]:
    copied = dict(inputs)
    copied["text_mode"] = mode
    return copied


def clone_inputs_with_text_emb(
    inputs: dict[str, Any], text_emb: torch.Tensor
) -> dict[str, Any]:
    copied = dict(inputs)
    copied["text_emb"] = text_emb
    copied["text_mode"] = "normal"
    return copied


def make_eval_loader(dataset: Any, cfg: Config, device: torch.device):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.scan_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.num_workers_override or 0,
        pin_memory=cfg.pin_memory and device.type == "cuda",
        collate_fn=dataset.collect(),
        persistent_workers=(cfg.num_workers_override or 0) > 0,
    )


def get_single_batch(dataset: Any, sample_idx: int, device: torch.device):
    collate = dataset.collect()
    inputs, targets = collate([dataset[sample_idx]])
    return to_device(inputs, device), normalize_targets(targets, device)


# =============================================================================
# 3. 训练模块 / checkpoint / cache / 模型结构
# =============================================================================
def resolve_training_relative(path_like: str | Path, train_module: Any) -> Path:
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p.resolve()
    module_base = Path(getattr(train_module, "BASE_PATH", BASE_PATH))
    return (module_base / p).resolve()


def choose_checkpoint(cfg: Config, train_module: Any, train_args: Any) -> Path:
    if cfg.checkpoint_path is not None:
        if not cfg.checkpoint_path.is_file():
            raise FileNotFoundError(f"CHECKPOINT_PATH does not exist: {cfg.checkpoint_path}")
        return cfg.checkpoint_path

    root_value = getattr(train_args, "output_root", None)
    if root_value is None:
        root_value = getattr(train_module, "output_root", None)
    if root_value is None:
        raise ValueError("cannot infer output_root; please set CHECKPOINT_PATH")
    root = resolve_training_relative(root_value, train_module)
    candidates = list(root.glob("**/checkpoints/best_checkpoint.pt"))
    if not candidates:
        candidates = list(root.glob("**/best_checkpoint.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"no best_checkpoint.pt found below {root}; set CHECKPOINT_PATH"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_checkpoint_file(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_model_state(checkpoint: Any) -> tuple[dict[str, Any], list[str]]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
        excluded = list(checkpoint.get("excluded_model_prefixes", []))
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
        excluded = list(checkpoint.get("excluded_model_prefixes", []))
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
        excluded = list(checkpoint.get("excluded_model_prefixes", []))
    elif isinstance(checkpoint, dict) and checkpoint and all(
        torch.is_tensor(v) for v in checkpoint.values()
    ):
        state = checkpoint
        excluded = []
    else:
        raise KeyError("checkpoint does not contain model parameters")

    if state and all(str(k).startswith("module.") for k in state):
        state = {str(k)[len("module."):]: v for k, v in state.items()}
    return state, excluded


def load_model_state(model: torch.nn.Module, path: Path, logger: logging.Logger):
    checkpoint = load_checkpoint_file(path)
    state, excluded = extract_model_state(checkpoint)
    incompatible = model.load_state_dict(state, strict=False)
    missing = [
        x for x in incompatible.missing_keys
        if not any(x.startswith(prefix) for prefix in excluded)
    ]
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            "checkpoint/model mismatch; "
            f"missing={missing[:50]}, unexpected={unexpected[:50]}"
        )
    logger.info("loaded model tensors: %d", len(state))
    if excluded:
        logger.info("checkpoint excluded prefixes: %s", excluded)
    return checkpoint if isinstance(checkpoint, dict) else {}


def classify_parameter(name: str) -> str:
    if name.startswith("text_encoder.") or name.startswith("qwen."):
        return "text_encoder"
    if name.startswith("text_proj."):
        return "text_projection"
    if name.startswith("regression_head."):
        return "regression_head"
    if name.startswith(("atom_embedding.", "edge_embedding.", "angle_embedding.")):
        return "graph_input_embedding"
    if name.startswith("blocks."):
        if any(
            marker in name
            for marker in (
                "text_cross_attn",
                "norm_text_node",
                "norm_text_edge",
            )
        ):
            return "text_graph_fusion"
        return "graph_backbone"
    return "other"


def summarize_model(
    model: torch.nn.Module,
    train_args: Any,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    cache_np: np.ndarray,
    run_dir: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    groups: dict[str, dict[str, int]] = {}
    total = 0
    trainable = 0
    for name, p in model.named_parameters():
        count = int(p.numel())
        group = classify_parameter(name)
        total += count
        if p.requires_grad:
            trainable += count
        g = groups.setdefault(group, {"total": 0, "trainable": 0})
        g["total"] += count
        if p.requires_grad:
            g["trainable"] += count
        rows.append(
            {
                "name": name,
                "group": group,
                "shape": "x".join(str(x) for x in p.shape),
                "numel": count,
                "requires_grad": bool(p.requires_grad),
            }
        )

    write_csv(run_dir / "model_parameters.csv", rows)
    group_rows = [
        {
            "group": group,
            "total_params": values["total"],
            "trainable_params": values["trainable"],
            "total_M": values["total"] / 1e6,
            "trainable_M": values["trainable"] / 1e6,
        }
        for group, values in sorted(groups.items())
    ]
    write_csv(run_dir / "model_parameter_groups.csv", group_rows)

    blocks = getattr(model, "blocks", None)
    depth = len(blocks) if blocks is not None else getattr(train_args, "depth", None)
    text_proj = getattr(model, "text_proj", None)
    text_proj_repr = repr(text_proj) if text_proj is not None else None
    raw_dim = int(getattr(model, "text_dim", cache_np.shape[1]))
    proj_dim = int(getattr(model, "text_proj_dim", 0) or 0)
    if proj_dim <= 0 and text_proj is not None:
        # 运行时后面还会用真实 probe 覆盖。
        for module in reversed(list(text_proj.modules())):
            if isinstance(module, torch.nn.Linear):
                proj_dim = int(module.out_features)
                break

    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "total_params": total,
        "trainable_params_flagged": trainable,
        "parameter_groups": groups,
        "depth": depth,
        "graph_embed_dim": int(getattr(model, "embed_dim", getattr(train_args, "embed_dim", 0))),
        "raw_text_dim": raw_dim,
        "projected_text_dim": proj_dim,
        "text_proj_hidden_dims": list(getattr(model, "text_proj_hidden_dims", [])),
        "text_proj_dropout": getattr(model, "text_proj_dropout", None),
        "text_projection_repr": text_proj_repr,
        "text_encoder_name": getattr(model, "text_encoder_name", getattr(train_args, "text_encoder_name", None)),
        "text_model_name": getattr(model, "text_model_name", getattr(train_args, "text_model_name", None)),
        "text_pooling": getattr(model, "text_pooling", getattr(train_args, "text_pooling", None)),
        "text_encoder_loaded": getattr(model, "text_encoder", None) is not None,
        "cache_shape": list(cache_np.shape),
        "readout": type(getattr(model, "readout", None)).__name__,
        "regression_head": repr(getattr(model, "regression_head", None)),
    }
    save_json(run_dir / "model_structure_summary.json", summary)

    structure_lines = [
        "Fusiformer current model structure",
        "=" * 80,
        f"Graph embedding dimension : {summary['graph_embed_dim']}",
        f"Fusiformer depth          : {summary['depth']}",
        f"Raw text dimension        : {summary['raw_text_dim']}",
        f"Projected text dimension  : {summary['projected_text_dim']}",
        f"Text projection hidden    : {summary['text_proj_hidden_dims']}",
        f"Text encoder loaded       : {summary['text_encoder_loaded']}",
        f"Readout                   : {summary['readout']}",
        f"Total parameters          : {summary['total_params']:,}",
        "",
        "Graph path:",
        "  atomic number -> atom_embedding",
        "  distance -> RBF -> edge_embedding",
        "  angle -> RBF -> angle_embedding",
        "  [line-graph update -> main-graph update] x depth",
        "  AvgPooling -> regression head",
        "",
        "Text path:",
        f"  cached pooled text [{summary['raw_text_dim']}D]",
        f"  -> text_proj -> [{summary['projected_text_dim']}D]",
        "  -> text guidance inside line-graph/main-graph update layers",
        "",
        "Exact text_proj:",
        str(text_proj_repr),
        "",
        "Parameter groups:",
    ]
    for r in group_rows:
        structure_lines.append(
            f"  {r['group']:<24s} {r['total_params']:>12,d} params"
        )
    (run_dir / "model_structure.txt").write_text(
        "\n".join(structure_lines), encoding="utf-8"
    )

    logger.info(
        "model | depth=%s graph_dim=%s raw_text=%s projected=%s total=%.3fM",
        summary["depth"],
        summary["graph_embed_dim"],
        summary["raw_text_dim"],
        summary["projected_text_dim"],
        total / 1e6,
    )
    return summary


def build_project_components(cfg: Config, device: torch.device, logger: logging.Logger):
    train_module = importlib.import_module(cfg.training_module_name)
    for fn in ("build_runtime_config", "build_model", "build_dataloaders"):
        if not hasattr(train_module, fn):
            raise AttributeError(f"{cfg.training_module_name} does not expose {fn}()")

    train_args = train_module.build_runtime_config()
    if not bool(getattr(train_args, "use_precomputed_text_embeddings", False)):
        raise RuntimeError("diagnostic requires use_precomputed_text_embeddings=True")

    # 只改变后处理 loader 设置，不改变 split / 模型超参数。
    if hasattr(train_args, "batch_size"):
        train_args.batch_size = cfg.scan_batch_size
    if cfg.num_workers_override is not None and hasattr(train_args, "num_workers"):
        train_args.num_workers = cfg.num_workers_override
    if hasattr(train_args, "pin_mem"):
        train_args.pin_mem = cfg.pin_memory

    bundle = train_module.build_dataloaders(train_args, device, logger)
    if len(bundle) < 3:
        raise RuntimeError("build_dataloaders() must return train/val/test datasets")
    train_dataset, val_dataset, test_dataset = bundle[:3]
    eval_dataset = test_dataset if cfg.eval_split == "test" else val_dataset
    if eval_dataset is None or len(eval_dataset) == 0:
        raise RuntimeError(f"{cfg.eval_split} dataset is empty")

    model = train_module.build_model(train_args, device)
    checkpoint_path = choose_checkpoint(cfg, train_module, train_args)
    checkpoint = load_model_state(model, checkpoint_path, logger)
    model.eval()

    cache_value = getattr(train_args, "text_embedding_cache_path", None)
    if cache_value is None:
        raise ValueError("training config does not provide text_embedding_cache_path")
    cache_path = resolve_training_relative(cache_value, train_module)
    if not cache_path.is_file():
        raise FileNotFoundError(f"text cache missing: {cache_path}")
    cache_np = np.load(cache_path, mmap_mode="r")

    logger.info("training module : %s", cfg.training_module_name)
    logger.info("checkpoint      : %s", checkpoint_path)
    logger.info("text cache      : %s", cache_path)
    logger.info("cache shape     : %s dtype=%s", cache_np.shape, cache_np.dtype)
    logger.info("eval split      : %s samples=%d", cfg.eval_split, len(eval_dataset))

    return (
        train_module,
        train_args,
        model,
        train_dataset,
        val_dataset,
        test_dataset,
        eval_dataset,
        cache_np,
        cache_path,
        checkpoint_path,
        checkpoint,
    )


def validate_data_and_cache(
    datasets: list[tuple[str, Any]],
    cache_np: np.ndarray,
    model: torch.nn.Module,
    run_dir: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    split_indices: dict[str, np.ndarray] = {}
    for name, ds in datasets:
        if ds is None:
            continue
        idx = np.asarray(ds.original_indices, dtype=np.int64)
        split_indices[name] = idx
        if idx.size and (idx.min() < 0 or idx.max() >= cache_np.shape[0]):
            raise RuntimeError(
                f"{name} original_indices exceed text cache rows; cache/dataset mismatch"
            )

    overlaps = {}
    names = list(split_indices)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = np.intersect1d(split_indices[a], split_indices[b])
            overlaps[f"{a}_{b}"] = int(inter.size)

    model_raw_dim = int(getattr(model, "text_dim", cache_np.shape[1]))
    if int(cache_np.shape[1]) != model_raw_dim:
        raise RuntimeError(
            f"cache dim {cache_np.shape[1]} != model text_dim {model_raw_dim}"
        )

    # 分块检查所有 split 实际会使用到的 cache rows 是否 finite。
    all_used = np.unique(np.concatenate(list(split_indices.values())))
    finite = True
    chunk = 4096
    for start in range(0, len(all_used), chunk):
        arr = np.asarray(cache_np[all_used[start:start + chunk]])
        if not np.isfinite(arr).all():
            finite = False
            break
    if not finite:
        raise RuntimeError("text embedding cache contains NaN/Inf on used rows")

    summary = {
        "cache_shape": list(cache_np.shape),
        "cache_dtype": str(cache_np.dtype),
        "model_raw_text_dim": model_raw_dim,
        "split_sizes": {k: int(len(v)) for k, v in split_indices.items()},
        "split_original_index_overlaps": overlaps,
        "used_cache_rows": int(len(all_used)),
        "used_cache_all_finite": finite,
    }
    save_json(run_dir / "data_cache_integrity.json", summary)
    logger.info("data/cache integrity: PASS | split overlaps=%s", overlaps)
    return summary


# =============================================================================
# 4. 文本 donor / shuffle 构造
# =============================================================================
def get_composition_key(info: dict[str, Any]) -> str:
    for key in [
        "composition",
        "formula",
        "formula_pretty",
        "pretty_formula",
        "reduced_formula",
        "full_formula",
        "anonymous_formula",
    ]:
        if key in info and info[key] not in (None, ""):
            value = info[key]
            if isinstance(value, dict):
                return json.dumps(value, sort_keys=True, ensure_ascii=False)
            return str(value)
    for key in ["elements", "atom_types", "species"]:
        if key in info and info[key] not in (None, ""):
            value = info[key]
            if isinstance(value, (list, tuple, set)):
                return "|".join(sorted(str(v) for v in value))
            return str(value)
    return "__unknown__"


def deranged_positions(n: int, rng: np.random.Generator) -> np.ndarray:
    base = np.arange(n, dtype=np.int64)
    if n <= 1:
        return base
    for _ in range(100):
        p = rng.permutation(n)
        if not np.any(p == base):
            return p
    return np.roll(base, 1)


def build_wrong_maps(dataset: Any, repeats: int, mode: str, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    original = np.asarray(dataset.original_indices, dtype=np.int64)
    n = len(dataset)
    comp = np.asarray([get_composition_key(x) for x in dataset.data], dtype=object)
    maps: list[np.ndarray] = []

    if n < 2:
        raise RuntimeError("shuffle diagnostic needs at least 2 samples")

    groups: dict[str, np.ndarray] = {}
    for key in np.unique(comp):
        groups[str(key)] = np.where(comp == key)[0]

    for _ in range(repeats):
        if mode == "global":
            donor_pos = deranged_positions(n, rng)

        elif mode == "different_composition":
            donor_pos = np.empty(n, dtype=np.int64)
            all_pos = np.arange(n, dtype=np.int64)
            for i in range(n):
                valid = all_pos[(comp != comp[i]) & (all_pos != i)]
                if valid.size == 0:
                    valid = all_pos[all_pos != i]
                donor_pos[i] = int(rng.choice(valid))

        elif mode == "within_composition":
            donor_pos = deranged_positions(n, rng)  # fallback
            for positions in groups.values():
                if len(positions) < 2:
                    continue
                local = deranged_positions(len(positions), rng)
                donor_pos[positions] = positions[local]
        else:
            raise ValueError(f"unsupported shuffle mode: {mode}")

        mapped = original[donor_pos]
        if np.any(mapped == original):
            raise RuntimeError("wrong-text map unexpectedly contains self donor")
        maps.append(mapped)
    return maps


def repair_null_maps(
    dataset: Any,
    maps_a: list[np.ndarray],
    maps_b: list[np.ndarray],
    mode: str,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    original = np.asarray(dataset.original_indices, dtype=np.int64)
    comp = np.asarray([get_composition_key(x) for x in dataset.data], dtype=object)
    original_to_pos = {int(v): i for i, v in enumerate(original)}
    all_pos = np.arange(len(dataset), dtype=np.int64)
    out_maps: list[np.ndarray] = []

    for a, b in zip(maps_a, maps_b):
        out = b.copy()
        for i in range(len(dataset)):
            if int(out[i]) != int(a[i]) and int(out[i]) != int(original[i]):
                continue
            valid = all_pos[all_pos != i]
            if mode == "different_composition":
                filtered = valid[comp[valid] != comp[i]]
                if filtered.size:
                    valid = filtered
            elif mode == "within_composition":
                filtered = valid[comp[valid] == comp[i]]
                if filtered.size:
                    valid = filtered
            # 还必须与 wrong A donor 不同。
            a_pos = original_to_pos.get(int(a[i]), -1)
            valid = valid[valid != a_pos]
            if valid.size == 0:
                valid = all_pos[(all_pos != i) & (all_pos != a_pos)]
            if valid.size == 0:
                raise RuntimeError("cannot construct distinct null donor")
            out[i] = int(original[int(rng.choice(valid))])
        out_maps.append(out)
    return out_maps


def save_donor_maps(
    dataset: Any,
    maps_a: list[np.ndarray],
    maps_b: list[np.ndarray],
    run_dir: Path,
) -> None:
    original = np.asarray(dataset.original_indices, dtype=np.int64)
    rows = []
    max_r = min(len(maps_a), 20)
    for i in range(len(dataset)):
        row: dict[str, Any] = {
            "sample_idx": i,
            "original_idx": int(original[i]),
            "material_id": safe_get_id(dataset.data[i]),
            "composition_key": get_composition_key(dataset.data[i]),
        }
        for r in range(max_r):
            row[f"wrong_a_r{r}"] = int(maps_a[r][i])
            row[f"wrong_b_r{r}"] = int(maps_b[r][i])
        rows.append(row)
    write_csv(run_dir / "wrong_text_donor_maps.csv", rows)


def cache_batch_to_tensor(
    cache_np: np.ndarray,
    original_indices: Iterable[int],
    device: torch.device,
) -> torch.Tensor:
    idx = np.asarray(list(original_indices), dtype=np.int64)
    arr = np.asarray(cache_np[idx], dtype=np.float32)
    return torch.from_numpy(arr).to(device=device, dtype=torch.float32)


# =============================================================================
# 5. 推理 / projected override / 自检
# =============================================================================
@torch.no_grad()
def predict_batch_mode(
    model: torch.nn.Module,
    inputs: dict[str, Any],
    mode: str,
    device: torch.device,
    amp: bool,
) -> torch.Tensor:
    model.eval()
    run_inputs = clone_inputs_with_mode(inputs, mode)
    with autocast_context(device, amp):
        y = selected_output(model(run_inputs), 0)
    return y.detach().float().cpu()


@torch.no_grad()
def predict_batch_text_emb(
    model: torch.nn.Module,
    inputs: dict[str, Any],
    text_emb: torch.Tensor,
    device: torch.device,
    amp: bool,
) -> torch.Tensor:
    model.eval()
    run_inputs = clone_inputs_with_text_emb(inputs, text_emb)
    with autocast_context(device, amp):
        y = selected_output(model(run_inputs), 0)
    return y.detach().float().cpu()


@torch.no_grad()
def project_text(model: torch.nn.Module, text_emb: torch.Tensor) -> torch.Tensor:
    model.eval()
    if not hasattr(model, "project_cached_text"):
        raise AttributeError("model has no project_cached_text()")
    return model.project_cached_text(text_emb)


@contextmanager
def override_project_cached_text(model: torch.nn.Module, projected_text: torch.Tensor):
    if not hasattr(model, "project_cached_text"):
        raise AttributeError("model has no project_cached_text()")
    original_method = model.project_cached_text

    def patched(self, pooled_text_emb: torch.Tensor):
        if projected_text.shape[0] == pooled_text_emb.shape[0]:
            return projected_text
        if projected_text.shape[0] == 1:
            return projected_text.expand(pooled_text_emb.shape[0], -1)
        raise ValueError(
            "projected text batch mismatch: "
            f"projected={tuple(projected_text.shape)}, raw={tuple(pooled_text_emb.shape)}"
        )

    model.project_cached_text = MethodType(patched, model)
    try:
        yield
    finally:
        model.project_cached_text = original_method


@torch.no_grad()
def predict_single_projected(
    model: torch.nn.Module,
    inputs: dict[str, Any],
    u: torch.Tensor,
    device: torch.device,
    amp: bool,
) -> float:
    model.eval()
    with override_project_cached_text(model, u):
        run_inputs = clone_inputs_with_mode(inputs, "normal")
        with autocast_context(device, amp):
            y = scalar_selected_output(model(run_inputs), 0)
    return float(y.detach().float().cpu().item())


def run_interface_self_checks(
    model: torch.nn.Module,
    dataset: Any,
    cache_np: np.ndarray,
    maps_a: list[np.ndarray],
    device: torch.device,
    cfg: Config,
    run_dir: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    inputs, _ = get_single_batch(dataset, 0, device)
    idx = int(dataset.original_indices[0])
    wrong_idx = int(maps_a[0][0])
    z = cache_batch_to_tensor(cache_np, [idx], device)
    z_wrong = cache_batch_to_tensor(cache_np, [wrong_idx], device)

    pred_raw = float(
        predict_batch_text_emb(model, inputs, z, device, cfg.use_amp_prediction)[0]
    )
    u = project_text(model, z)
    pred_projected = predict_single_projected(
        model, inputs, u, device, cfg.use_amp_prediction
    )
    equivalence_diff = abs(pred_raw - pred_projected)

    # bypass 条件应与输入 text_emb 无关。
    bypass_1 = float(
        predict_batch_mode(model, clone_inputs_with_text_emb(inputs, z), "bypass", device, cfg.use_amp_prediction)[0]
    )
    bypass_2 = float(
        predict_batch_mode(model, clone_inputs_with_text_emb(inputs, z_wrong), "bypass", device, cfg.use_amp_prediction)[0]
    )
    bypass_diff = abs(bypass_1 - bypass_2)

    summary = {
        "projected_forward_equivalence_abs_diff": equivalence_diff,
        "bypass_text_invariance_abs_diff": bypass_diff,
        "projected_forward_pass": bool(equivalence_diff <= 1e-5),
        "strict_bypass_pass": bool(bypass_diff <= 1e-6),
    }
    save_json(run_dir / "interface_self_checks.json", summary)
    if not summary["projected_forward_pass"]:
        raise RuntimeError(
            "projected override does not reproduce normal forward: "
            f"abs diff={equivalence_diff:.6g}"
        )
    if not summary["strict_bypass_pass"]:
        raise RuntimeError(
            "bypass prediction depends on text embedding; strict bypass is broken: "
            f"abs diff={bypass_diff:.6g}"
        )
    logger.info(
        "interface checks: PASS | projected diff=%.3g | bypass diff=%.3g",
        equivalence_diff,
        bypass_diff,
    )
    return summary


# =============================================================================
# 6. 统计推断
# =============================================================================
def bootstrap_mean_ci(
    values: np.ndarray,
    repeats: int,
    confidence: float,
    seed: int,
) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=np.float64)
    # chunk 避免一次创建 repeats x N 的超大数组。
    chunk = 128
    done = 0
    while done < repeats:
        m = min(chunk, repeats - done)
        idx = rng.integers(0, len(v), size=(m, len(v)))
        means[done:done + m] = v[idx].mean(axis=1)
        done += m
    alpha = 1.0 - confidence
    return {
        "mean": float(v.mean()),
        "ci_low": float(np.quantile(means, alpha / 2.0)),
        "ci_high": float(np.quantile(means, 1.0 - alpha / 2.0)),
    }


def sign_flip_pvalue(
    values: np.ndarray,
    repeats: int,
    seed: int,
    alternative: str = "greater",
) -> float:
    """Paired random sign-flip test for mean(values)."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.size == 0:
        return float("nan")
    observed = float(v.mean())
    rng = np.random.default_rng(seed)
    extreme = 0
    chunk = 128
    done = 0
    while done < repeats:
        m = min(chunk, repeats - done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(m, len(v)))
        means = (signs * v[None, :]).mean(axis=1)
        if alternative == "greater":
            extreme += int(np.sum(means >= observed))
        elif alternative == "two-sided":
            extreme += int(np.sum(np.abs(means) >= abs(observed)))
        else:
            raise ValueError("alternative must be greater or two-sided")
        done += m
    return float((extreme + 1) / (repeats + 1))


def paired_effect_summary(
    diff: np.ndarray,
    cfg: Config,
    seed_offset: int,
) -> dict[str, Any]:
    d = np.asarray(diff, dtype=np.float64)
    out: dict[str, Any] = {
        "mean_improvement": float(d.mean()),
        "median_improvement": float(np.median(d)),
        "fraction_improved": float(np.mean(d > 0)),
        "fraction_hurt": float(np.mean(d < 0)),
        "std": float(d.std()),
        "standardized_mean_effect": float(d.mean() / (d.std() + EPS)),
    }
    if cfg.run_statistical_inference:
        out["bootstrap"] = bootstrap_mean_ci(
            d,
            cfg.bootstrap_repeats,
            cfg.confidence_level,
            cfg.seed + seed_offset,
        )
        out["sign_flip_p_one_sided_mean_gt_0"] = sign_flip_pvalue(
            d,
            cfg.permutation_repeats,
            cfg.seed + 1000 + seed_offset,
            alternative="greater",
        )
    return out


# =============================================================================
# 7. Experiment A: 全集文本效用 + repeated wrong text + null
# =============================================================================
def run_prediction_utility(
    model: torch.nn.Module,
    dataset: Any,
    cache_np: np.ndarray,
    maps_a: list[np.ndarray],
    maps_b: list[np.ndarray],
    cfg: Config,
    device: torch.device,
    run_dir: Path,
    logger: logging.Logger,
):
    logger.info("========== A. Prediction-level text utility ==========")
    loader = make_eval_loader(dataset, cfg, device)
    sample_rows: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []

    ys, normals, bypasses = [], [], []
    wrong_a_all, wrong_b_all = [], []
    offset = 0
    t0 = time.time()

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = to_device(inputs, device)
        targets = normalize_targets(targets, device)
        y = targets[:, 0].detach().float().cpu().numpy()
        normal = predict_batch_mode(
            model, inputs, "normal", device, cfg.use_amp_prediction
        ).numpy()
        bypass = predict_batch_mode(
            model, inputs, "bypass", device, cfg.use_amp_prediction
        ).numpy()

        bsz = len(y)
        positions = np.arange(offset, offset + bsz, dtype=np.int64)
        a_preds, b_preds = [], []
        for r in range(cfg.prediction_shuffle_repeats):
            za = cache_batch_to_tensor(cache_np, maps_a[r][positions], device)
            zb = cache_batch_to_tensor(cache_np, maps_b[r][positions], device)
            pa = predict_batch_text_emb(
                model, inputs, za, device, cfg.use_amp_prediction
            ).numpy()
            pb = predict_batch_text_emb(
                model, inputs, zb, device, cfg.use_amp_prediction
            ).numpy()
            a_preds.append(pa)
            b_preds.append(pb)
        pa = np.stack(a_preds, axis=1)  # B,R
        pb = np.stack(b_preds, axis=1)

        normal_err = np.abs(normal - y)
        bypass_err = np.abs(bypass - y)
        wrong_err = np.abs(pa - y[:, None])
        pred_signal = np.abs(normal[:, None] - pa)
        pred_null = np.abs(pa - pb)

        for i in range(bsz):
            sample_idx = offset + i
            signal_mean = float(pred_signal[i].mean())
            null_mean = float(pred_null[i].mean())
            wrong_error_mean = float(wrong_err[i].mean())
            row = {
                "sample_idx": sample_idx,
                "original_idx": int(dataset.original_indices[sample_idx]),
                "material_id": safe_get_id(dataset.data[sample_idx]),
                "composition_key": get_composition_key(dataset.data[sample_idx]),
                "target": float(y[i]),
                "pred_normal": float(normal[i]),
                "pred_bypass": float(bypass[i]),
                "error_normal": float(normal_err[i]),
                "error_bypass": float(bypass_err[i]),
                "text_help_vs_bypass": float(bypass_err[i] - normal_err[i]),
                "pred_wrong_mean": float(pa[i].mean()),
                "pred_wrong_std": float(pa[i].std()),
                "wrong_error_mean": wrong_error_mean,
                "wrong_error_std": float(wrong_err[i].std()),
                "matched_text_advantage_vs_wrong": float(wrong_error_mean - normal_err[i]),
                "prediction_signal_mean": signal_mean,
                "prediction_signal_std": float(pred_signal[i].std()),
                "prediction_null_mean": null_mean,
                "prediction_null_std": float(pred_null[i].std()),
                "prediction_signal_null_ratio": float(signal_mean / (null_mean + EPS)),
            }
            sample_rows.append(row)
            for r in range(cfg.prediction_shuffle_repeats):
                repeat_rows.append(
                    {
                        "sample_idx": sample_idx,
                        "repeat": r,
                        "wrong_a_original_idx": int(maps_a[r][sample_idx]),
                        "wrong_b_original_idx": int(maps_b[r][sample_idx]),
                        "pred_normal": float(normal[i]),
                        "pred_wrong_a": float(pa[i, r]),
                        "pred_wrong_b": float(pb[i, r]),
                        "wrong_error": float(wrong_err[i, r]),
                        "matched_advantage": float(wrong_err[i, r] - normal_err[i]),
                        "prediction_signal_abs": float(pred_signal[i, r]),
                        "prediction_null_abs": float(pred_null[i, r]),
                    }
                )

        ys.append(y)
        normals.append(normal)
        bypasses.append(bypass)
        wrong_a_all.append(pa)
        wrong_b_all.append(pb)
        offset += bsz
        if (batch_idx + 1) % cfg.log_interval == 0 or batch_idx == len(loader) - 1:
            logger.info("prediction batches: %d/%d", batch_idx + 1, len(loader))

    y = np.concatenate(ys)
    normal = np.concatenate(normals)
    bypass = np.concatenate(bypasses)
    wrong_a = np.concatenate(wrong_a_all, axis=0)
    wrong_b = np.concatenate(wrong_b_all, axis=0)

    normal_err = np.abs(normal - y)
    bypass_err = np.abs(bypass - y)
    wrong_err = np.abs(wrong_a - y[:, None])
    wrong_err_sample_mean = wrong_err.mean(axis=1)
    signal = np.abs(normal[:, None] - wrong_a)
    null = np.abs(wrong_a - wrong_b)
    mae_wrong_repeats = wrong_err.mean(axis=0)

    summary = {
        "num_samples": int(len(y)),
        "shuffle_mode": cfg.shuffle_mode,
        "num_shuffle_repeats": cfg.prediction_shuffle_repeats,
        "mae_normal": float(normal_err.mean()),
        "mae_bypass": float(bypass_err.mean()),
        "mae_wrong_mean_over_repeats": float(mae_wrong_repeats.mean()),
        "mae_wrong_std_over_repeats": float(mae_wrong_repeats.std()),
        "utility_vs_bypass": paired_effect_summary(
            bypass_err - normal_err, cfg, seed_offset=11
        ),
        "utility_vs_wrong_text": paired_effect_summary(
            wrong_err_sample_mean - normal_err, cfg, seed_offset=23
        ),
        "prediction_signal_mean": float(signal.mean()),
        "prediction_null_mean": float(null.mean()),
        "prediction_signal_null_ratio_global": float(
            signal.mean() / (null.mean() + EPS)
        ),
        "fraction_samples_signal_gt_null": float(
            np.mean(signal.mean(axis=1) > null.mean(axis=1))
        ),
        "elapsed_seconds": float(time.time() - t0),
    }

    write_csv(run_dir / "prediction_utility_samples.csv", sample_rows)
    write_csv(run_dir / "prediction_utility_repeats.csv", repeat_rows)
    save_json(run_dir / "prediction_utility_summary.json", summary)

    logger.info(
        "MAE normal=%.8f bypass=%.8f wrong=%.8f | matched advantage=%.8f | signal/null=%.3f",
        summary["mae_normal"],
        summary["mae_bypass"],
        summary["mae_wrong_mean_over_repeats"],
        summary["utility_vs_wrong_text"]["mean_improvement"],
        summary["prediction_signal_null_ratio_global"],
    )
    if cfg.save_plots:
        plot_prediction_utility(sample_rows, run_dir, cfg)
    return sample_rows, summary


# =============================================================================
# 8. IG 数值工具
# =============================================================================
def integration_nodes_weights(steps: int, method: str) -> tuple[np.ndarray, np.ndarray]:
    if method == "midpoint":
        alpha = (np.arange(steps, dtype=np.float64) + 0.5) / float(steps)
        weights = np.full(steps, 1.0 / steps, dtype=np.float64)
        return alpha, weights
    if method == "gauss_legendre":
        x, w = np.polynomial.legendre.leggauss(steps)
        alpha = (x + 1.0) / 2.0
        weights = w / 2.0
        return alpha.astype(np.float64), weights.astype(np.float64)
    raise ValueError(method)


def attribution_stats(v: np.ndarray) -> dict[str, float]:
    x = np.asarray(v, dtype=np.float64)
    abs_x = np.abs(x)
    l1 = float(abs_x.sum())
    signed_sum = float(x.sum())
    p = abs_x / (l1 + EPS)
    entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum()) if l1 > 0 else 0.0
    effective_channels = float(math.exp(entropy)) if l1 > 0 else 0.0

    order = np.argsort(-abs_x)
    def top_share(k: int) -> float:
        if l1 <= 0:
            return 0.0
        kk = min(k, len(x))
        return float(abs_x[order[:kk]].sum() / l1)

    return {
        "signed_sum": signed_sum,
        "abs_sum_l1": l1,
        "l2": float(np.linalg.norm(x)),
        "max_abs": float(abs_x.max()) if x.size else 0.0,
        "cancellation_ratio": float(1.0 - abs(signed_sum) / (l1 + EPS)) if l1 > 0 else 0.0,
        "top1_l1_share": top_share(1),
        "top5_l1_share": top_share(5),
        "top10_l1_share": top_share(10),
        "effective_channels": effective_channels,
    }


def completeness_metrics(
    attribution: np.ndarray,
    pred_from: float,
    pred_to: float,
    min_delta: float,
) -> dict[str, Any]:
    delta = float(pred_to - pred_from)
    attr_sum = float(np.asarray(attribution, dtype=np.float64).sum())
    abs_error = abs(delta - attr_sum)
    return {
        "prediction_delta": delta,
        "ig_signed_sum": attr_sum,
        "completeness_abs_error": float(abs_error),
        "completeness_rel_error": float(abs_error / max(abs(delta), min_delta)),
        "tiny_prediction_delta": bool(abs(delta) < min_delta),
    }


def projected_differential_ig(
    model: torch.nn.Module,
    inputs: dict[str, Any],
    u_from: torch.Tensor,
    u_to: torch.Tensor,
    steps: int,
    method: str,
    device: torch.device,
    amp: bool,
) -> np.ndarray:
    model.eval()
    u0 = u_from.detach().clone().to(device=device, dtype=torch.float32)
    u1 = u_to.detach().clone().to(device=device, dtype=torch.float32)
    if u0.ndim == 1:
        u0 = u0.unsqueeze(0)
    if u1.ndim == 1:
        u1 = u1.unsqueeze(0)
    delta = u1 - u0
    weighted_grad = torch.zeros_like(u1)
    alphas, weights = integration_nodes_weights(steps, method)

    for alpha, weight in zip(alphas, weights):
        u_alpha = (u0 + float(alpha) * delta).detach().clone().requires_grad_(True)
        with override_project_cached_text(model, u_alpha):
            run_inputs = clone_inputs_with_mode(inputs, "normal")
            with autocast_context(device, amp):
                y = scalar_selected_output(model(run_inputs), 0)
        grad = torch.autograd.grad(
            y,
            u_alpha,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
        weighted_grad += float(weight) * grad.detach()

    ig = delta * weighted_grad
    return ig.squeeze(0).detach().float().cpu().numpy()


def raw_differential_ig(
    model: torch.nn.Module,
    inputs: dict[str, Any],
    z_from: torch.Tensor,
    z_to: torch.Tensor,
    steps: int,
    method: str,
    device: torch.device,
    amp: bool,
) -> np.ndarray:
    model.eval()
    z0 = z_from.detach().clone().to(device=device, dtype=torch.float32)
    z1 = z_to.detach().clone().to(device=device, dtype=torch.float32)
    if z0.ndim == 1:
        z0 = z0.unsqueeze(0)
    if z1.ndim == 1:
        z1 = z1.unsqueeze(0)
    delta = z1 - z0
    weighted_grad = torch.zeros_like(z1)
    alphas, weights = integration_nodes_weights(steps, method)

    for alpha, weight in zip(alphas, weights):
        z_alpha = (z0 + float(alpha) * delta).detach().clone().requires_grad_(True)
        run_inputs = clone_inputs_with_text_emb(inputs, z_alpha)
        with autocast_context(device, amp):
            y = scalar_selected_output(model(run_inputs), 0)
        grad = torch.autograd.grad(
            y,
            z_alpha,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
        weighted_grad += float(weight) * grad.detach()

    ig = delta * weighted_grad
    return ig.squeeze(0).detach().float().cpu().numpy()


# =============================================================================
# 9. 样本选择
# =============================================================================
def select_samples(
    prediction_rows: list[dict[str, Any]],
    dataset: Any,
    cfg: Config,
    run_dir: Path,
) -> list[int]:
    n = len(dataset)
    k = min(cfg.num_samples_to_analyze, n)
    rng = np.random.default_rng(cfg.seed + 77)

    if cfg.sample_selection_mode == "manual":
        selected = [x for x in cfg.manual_sample_indices if 0 <= x < n]

    elif cfg.sample_selection_mode == "random" or not prediction_rows:
        selected = rng.choice(n, size=k, replace=False).tolist()

    else:
        by_idx = {int(r["sample_idx"]): r for r in prediction_rows}
        if cfg.sample_selection_mode == "top_prediction_signal":
            selected = sorted(
                range(n),
                key=lambda i: float(by_idx[i]["prediction_signal_mean"]),
                reverse=True,
            )[:k]
        elif cfg.sample_selection_mode == "top_text_help":
            selected = sorted(
                range(n),
                key=lambda i: float(by_idx[i]["matched_text_advantage_vs_wrong"]),
                reverse=True,
            )[:k]
        elif cfg.sample_selection_mode == "stratified_signal":
            ordered = sorted(
                range(n), key=lambda i: float(by_idx[i]["prediction_signal_mean"])
            )
            # 等分布抽取，覆盖低/中/高响应，避免 Top-only selection bias。
            positions = np.linspace(0, n - 1, num=k, dtype=int)
            selected = [ordered[int(p)] for p in positions]
        else:
            raise ValueError(cfg.sample_selection_mode)

    selected = list(dict.fromkeys(int(x) for x in selected))[:k]
    rows = []
    pred_lookup = {int(r["sample_idx"]): r for r in prediction_rows}
    for rank, idx in enumerate(selected):
        r = pred_lookup.get(idx, {})
        rows.append(
            {
                "selection_rank": rank,
                "sample_idx": idx,
                "original_idx": int(dataset.original_indices[idx]),
                "material_id": safe_get_id(dataset.data[idx]),
                "prediction_signal_mean": r.get("prediction_signal_mean"),
                "matched_text_advantage_vs_wrong": r.get("matched_text_advantage_vs_wrong"),
            }
        )
    write_csv(run_dir / "selected_ig_samples.csv", rows)
    return selected


# =============================================================================
# 10. Top-K faithfulness 与 interpolation path
# =============================================================================
def run_topk_occlusion_for_sample(
    model: torch.nn.Module,
    inputs: dict[str, Any],
    u_correct: torch.Tensor,
    u_wrong_mean: torch.Tensor,
    importance: np.ndarray,
    sample_idx: int,
    cfg: Config,
    device: torch.device,
) -> list[dict[str, Any]]:
    base_pred = predict_single_projected(
        model, inputs, u_correct, device, cfg.use_amp_prediction
    )
    c = int(u_correct.shape[-1])
    order = np.argsort(-np.abs(importance))
    rng = np.random.default_rng(cfg.seed + 3000 + sample_idx)
    rows: list[dict[str, Any]] = []

    for k in cfg.projected_topk:
        k_eff = min(int(k), c)
        if k_eff <= 0:
            continue
        top_idx = order[:k_eff]
        u_top = u_correct.clone()
        u_top[:, top_idx] = u_wrong_mean[:, top_idx]
        p_top = predict_single_projected(
            model, inputs, u_top, device, cfg.use_amp_prediction
        )
        top_change = abs(base_pred - p_top)

        random_changes = []
        for _ in range(cfg.random_occlusion_repeats):
            idx = rng.choice(c, size=k_eff, replace=False)
            u_rand = u_correct.clone()
            u_rand[:, idx] = u_wrong_mean[:, idx]
            p_rand = predict_single_projected(
                model, inputs, u_rand, device, cfg.use_amp_prediction
            )
            random_changes.append(abs(base_pred - p_rand))
        rand_mean = float(np.mean(random_changes))
        rows.append(
            {
                "sample_idx": sample_idx,
                "k": k_eff,
                "topk_abs_prediction_change": float(top_change),
                "random_abs_prediction_change_mean": rand_mean,
                "random_abs_prediction_change_std": float(np.std(random_changes)),
                "topk_random_ratio": float(top_change / (rand_mean + EPS)),
            }
        )
    return rows


def run_interpolation_for_sample(
    model: torch.nn.Module,
    inputs: dict[str, Any],
    u_wrong: torch.Tensor,
    u_correct: torch.Tensor,
    sample_idx: int,
    cfg: Config,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows = []
    p0 = predict_single_projected(model, inputs, u_wrong, device, cfg.use_amp_prediction)
    p1 = predict_single_projected(model, inputs, u_correct, device, cfg.use_amp_prediction)
    for alpha in cfg.interpolation_alphas:
        u = u_wrong + float(alpha) * (u_correct - u_wrong)
        pred = predict_single_projected(model, inputs, u, device, cfg.use_amp_prediction)
        linear_pred = p0 + float(alpha) * (p1 - p0)
        rows.append(
            {
                "sample_idx": sample_idx,
                "alpha": float(alpha),
                "prediction": pred,
                "linear_interpolation_prediction": linear_pred,
                "nonlinearity_abs": abs(pred - linear_pred),
            }
        )
    return rows


# =============================================================================
# 11. Experiment B: Projected IG + null + stability + faithfulness
# =============================================================================
def run_projected_ig_diagnostic(
    model: torch.nn.Module,
    dataset: Any,
    cache_np: np.ndarray,
    maps_a: list[np.ndarray],
    maps_b: list[np.ndarray],
    prediction_rows: list[dict[str, Any]],
    cfg: Config,
    device: torch.device,
    run_dir: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    logger.info("========== B. Projected differential IG diagnostic ==========")
    selected = select_samples(prediction_rows, dataset, cfg, run_dir)
    if not selected:
        raise RuntimeError("no samples selected for IG")

    signal_stack: list[np.ndarray] = []  # S,R,C
    null_stack: list[np.ndarray] = []
    raw_signal_stack: list[np.ndarray] = []
    raw_null_stack: list[np.ndarray] = []
    repeat_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    faithfulness_rows: list[dict[str, Any]] = []
    interpolation_rows: list[dict[str, Any]] = []

    for order_idx, sample_idx in enumerate(selected):
        inputs, targets = get_single_batch(dataset, sample_idx, device)
        target = float(targets[:, 0].detach().cpu().item())
        correct_idx = int(dataset.original_indices[sample_idx])
        z_correct = cache_batch_to_tensor(cache_np, [correct_idx], device)
        u_correct = project_text(model, z_correct).detach()
        pred_correct = predict_single_projected(
            model, inputs, u_correct, device, cfg.use_amp_prediction
        )

        sample_signal, sample_null = [], []
        sample_raw_signal, sample_raw_null = [], []
        sample_comp, null_comp = [], []
        wrong_projected: list[torch.Tensor] = []

        for r in range(cfg.ig_shuffle_repeats):
            ia = int(maps_a[r][sample_idx])
            ib = int(maps_b[r][sample_idx])
            za = cache_batch_to_tensor(cache_np, [ia], device)
            zb = cache_batch_to_tensor(cache_np, [ib], device)
            ua = project_text(model, za).detach()
            ub = project_text(model, zb).detach()
            wrong_projected.append(ua)

            pred_a = predict_single_projected(
                model, inputs, ua, device, cfg.use_amp_prediction
            )
            pred_b = predict_single_projected(
                model, inputs, ub, device, cfg.use_amp_prediction
            )

            sig = projected_differential_ig(
                model,
                inputs,
                ua,
                u_correct,
                cfg.ig_steps,
                cfg.ig_integration_method,
                device,
                cfg.use_amp_ig,
            )
            nul = projected_differential_ig(
                model,
                inputs,
                ua,
                ub,
                cfg.ig_steps,
                cfg.ig_integration_method,
                device,
                cfg.use_amp_ig,
            )
            sample_signal.append(sig)
            sample_null.append(nul)

            sig_stats = attribution_stats(sig)
            nul_stats = attribution_stats(nul)
            sig_comp = completeness_metrics(
                sig, pred_a, pred_correct, cfg.completeness_min_delta
            )
            nul_comp = completeness_metrics(
                nul, pred_a, pred_b, cfg.completeness_min_delta
            )
            sample_comp.append(float(sig_comp["completeness_rel_error"]))
            null_comp.append(float(nul_comp["completeness_rel_error"]))

            if cfg.run_raw_ig:
                raw_sig = raw_differential_ig(
                    model,
                    inputs,
                    za,
                    z_correct,
                    cfg.ig_steps,
                    cfg.ig_integration_method,
                    device,
                    cfg.use_amp_ig,
                )
                raw_nul = raw_differential_ig(
                    model,
                    inputs,
                    za,
                    zb,
                    cfg.ig_steps,
                    cfg.ig_integration_method,
                    device,
                    cfg.use_amp_ig,
                )
                sample_raw_signal.append(raw_sig)
                sample_raw_null.append(raw_nul)

            repeat_rows.append(
                {
                    "sample_idx": sample_idx,
                    "repeat": r,
                    "wrong_a_original_idx": ia,
                    "wrong_b_original_idx": ib,
                    "signal_prediction_delta": sig_comp["prediction_delta"],
                    **{f"signal_{k}": v for k, v in sig_stats.items()},
                    "signal_completeness_abs_error": sig_comp["completeness_abs_error"],
                    "signal_completeness_rel_error": sig_comp["completeness_rel_error"],
                    "signal_tiny_prediction_delta": sig_comp["tiny_prediction_delta"],
                    "null_prediction_delta": nul_comp["prediction_delta"],
                    **{f"null_{k}": v for k, v in nul_stats.items()},
                    "null_completeness_abs_error": nul_comp["completeness_abs_error"],
                    "null_completeness_rel_error": nul_comp["completeness_rel_error"],
                    "null_tiny_prediction_delta": nul_comp["tiny_prediction_delta"],
                    "ig_l1_signal_null_ratio": float(
                        sig_stats["abs_sum_l1"] / (nul_stats["abs_sum_l1"] + EPS)
                    ),
                }
            )

        sig_arr = np.stack(sample_signal, axis=0)
        nul_arr = np.stack(sample_null, axis=0)
        signal_stack.append(sig_arr)
        null_stack.append(nul_arr)
        if sample_raw_signal:
            raw_signal_stack.append(np.stack(sample_raw_signal, axis=0))
            raw_null_stack.append(np.stack(sample_raw_null, axis=0))

        sig_mean = sig_arr.mean(axis=0)
        per_repeat_stats = [attribution_stats(x) for x in sig_arr]
        within_mean = sig_arr.mean(axis=0)
        within_std = sig_arr.std(axis=0)
        within_snr = np.abs(within_mean) / (within_std + EPS)
        pos_rate = (sig_arr > 0).mean(axis=0)
        neg_rate = (sig_arr < 0).mean(axis=0)
        within_sign_consistency = np.maximum(pos_rate, neg_rate)
        sig_l1 = np.abs(sig_arr).sum(axis=1)
        nul_l1 = np.abs(nul_arr).sum(axis=1)

        sample_rows.append(
            {
                "sample_idx": sample_idx,
                "original_idx": correct_idx,
                "material_id": safe_get_id(dataset.data[sample_idx]),
                "target": target,
                "pred_correct": pred_correct,
                "projected_dim": int(sig_arr.shape[-1]),
                "ig_repeats": int(sig_arr.shape[0]),
                "signal_l1_mean": float(sig_l1.mean()),
                "null_l1_mean": float(nul_l1.mean()),
                "signal_null_ratio_l1": float(sig_l1.mean() / (nul_l1.mean() + EPS)),
                "signed_sum_mean": float(sig_arr.sum(axis=1).mean()),
                "cancellation_ratio_mean": float(np.mean([x["cancellation_ratio"] for x in per_repeat_stats])),
                "top5_l1_share_mean": float(np.mean([x["top5_l1_share"] for x in per_repeat_stats])),
                "top10_l1_share_mean": float(np.mean([x["top10_l1_share"] for x in per_repeat_stats])),
                "effective_channels_mean": float(np.mean([x["effective_channels"] for x in per_repeat_stats])),
                "mean_within_sample_channel_snr": float(within_snr.mean()),
                "median_within_sample_channel_snr": float(np.median(within_snr)),
                "mean_within_sample_sign_consistency": float(within_sign_consistency.mean()),
                "mean_signal_completeness_rel_error": float(np.mean(sample_comp)),
                "median_signal_completeness_rel_error": float(np.median(sample_comp)),
                "mean_null_completeness_rel_error": float(np.mean(null_comp)),
            }
        )

        u_wrong_mean = torch.stack(wrong_projected, dim=0).mean(dim=0)
        if cfg.run_topk_faithfulness:
            faithfulness_rows.extend(
                run_topk_occlusion_for_sample(
                    model,
                    inputs,
                    u_correct,
                    u_wrong_mean,
                    sig_mean,
                    sample_idx,
                    cfg,
                    device,
                )
            )
        if cfg.run_interpolation_path:
            interpolation_rows.extend(
                run_interpolation_for_sample(
                    model,
                    inputs,
                    wrong_projected[0],
                    u_correct,
                    sample_idx,
                    cfg,
                    device,
                )
            )

        logger.info(
            "IG %d/%d sample=%d | L1 signal=%.5g null=%.5g ratio=%.3f | comp median=%.4g",
            order_idx + 1,
            len(selected),
            sample_idx,
            sig_l1.mean(),
            nul_l1.mean(),
            sig_l1.mean() / (nul_l1.mean() + EPS),
            np.median(sample_comp),
        )

    signal = np.stack(signal_stack, axis=0)  # S,R,C
    null = np.stack(null_stack, axis=0)
    s_count, r_count, c_count = signal.shape

    # Channel statistics: 先在每个样本内部看 repeat 稳定性，再跨样本汇总。
    sample_mean = signal.mean(axis=1)  # S,C
    sample_std = signal.std(axis=1)
    sample_snr = np.abs(sample_mean) / (sample_std + EPS)
    sample_pos_rate = (signal > 0).mean(axis=1)
    sample_neg_rate = (signal < 0).mean(axis=1)
    sample_sign_consistency = np.maximum(sample_pos_rate, sample_neg_rate)

    signal_abs_mean = np.abs(signal).mean(axis=(0, 1))
    null_abs_mean = np.abs(null).mean(axis=(0, 1))
    channel_rows = []
    for c in range(c_count):
        channel_rows.append(
            {
                "channel": c,
                "signal_mean_signed_all": float(signal[:, :, c].mean()),
                "signal_mean_abs_all": float(signal_abs_mean[c]),
                "null_mean_abs_all": float(null_abs_mean[c]),
                "signal_null_ratio": float(signal_abs_mean[c] / (null_abs_mean[c] + EPS)),
                "mean_within_sample_signed_snr": float(sample_snr[:, c].mean()),
                "median_within_sample_signed_snr": float(np.median(sample_snr[:, c])),
                "mean_within_sample_sign_consistency": float(sample_sign_consistency[:, c].mean()),
                "across_sample_mean_abs_of_repeat_mean": float(np.abs(sample_mean[:, c]).mean()),
                "across_sample_std_of_repeat_mean": float(sample_mean[:, c].std()),
            }
        )

    sig_l1 = np.abs(signal).sum(axis=-1)
    nul_l1 = np.abs(null).sum(axis=-1)
    repeat_comp = np.asarray(
        [float(r["signal_completeness_rel_error"]) for r in repeat_rows], dtype=np.float64
    )
    repeat_comp_non_tiny = np.asarray(
        [
            float(r["signal_completeness_rel_error"])
            for r in repeat_rows
            if not bool(r["signal_tiny_prediction_delta"])
        ],
        dtype=np.float64,
    )
    if repeat_comp_non_tiny.size == 0:
        repeat_comp_non_tiny = repeat_comp

    faithfulness_summary = summarize_faithfulness(faithfulness_rows)
    interpolation_summary = summarize_interpolation(interpolation_rows)

    summary = {
        "num_samples": s_count,
        "ig_repeats": r_count,
        "projected_text_dim": c_count,
        "ig_steps": cfg.ig_steps,
        "integration_method": cfg.ig_integration_method,
        "global_signal_l1_mean": float(sig_l1.mean()),
        "global_null_l1_mean": float(nul_l1.mean()),
        "global_ig_signal_null_ratio": float(sig_l1.mean() / (nul_l1.mean() + EPS)),
        "median_completeness_rel_error_non_tiny": float(np.median(repeat_comp_non_tiny)),
        "p90_completeness_rel_error_non_tiny": float(np.percentile(repeat_comp_non_tiny, 90)),
        "mean_sample_cancellation_ratio": float(np.mean([r["cancellation_ratio_mean"] for r in sample_rows])),
        "mean_sample_top5_l1_share": float(np.mean([r["top5_l1_share_mean"] for r in sample_rows])),
        "mean_sample_top10_l1_share": float(np.mean([r["top10_l1_share_mean"] for r in sample_rows])),
        "mean_effective_channels": float(np.mean([r["effective_channels_mean"] for r in sample_rows])),
        "mean_within_sample_channel_snr": float(sample_snr.mean()),
        "median_within_sample_channel_snr": float(np.median(sample_snr)),
        "mean_within_sample_sign_consistency": float(sample_sign_consistency.mean()),
        "mean_channel_signal_null_ratio": float(np.mean([r["signal_null_ratio"] for r in channel_rows])),
        "median_channel_signal_null_ratio": float(np.median([r["signal_null_ratio"] for r in channel_rows])),
        "faithfulness": faithfulness_summary,
        "interpolation": interpolation_summary,
    }

    write_csv(run_dir / "ig_repeat_metrics.csv", repeat_rows)
    write_csv(run_dir / "ig_sample_summary.csv", sample_rows)
    write_csv(run_dir / "ig_channel_summary.csv", channel_rows)
    if faithfulness_rows:
        write_csv(run_dir / "topk_faithfulness.csv", faithfulness_rows)
        save_json(run_dir / "topk_faithfulness_summary.json", faithfulness_summary)
    if interpolation_rows:
        write_csv(run_dir / "interpolation_path.csv", interpolation_rows)
        save_json(run_dir / "interpolation_summary.json", interpolation_summary)

    if cfg.save_ig_stacks:
        np.save(run_dir / "ig_signal_all.npy", signal.astype(np.float32))
        np.save(run_dir / "ig_null_all.npy", null.astype(np.float32))
        np.save(run_dir / "ig_signal_mean_per_sample.npy", sample_mean.astype(np.float32))
        if raw_signal_stack:
            np.save(
                run_dir / "raw_ig_signal_all.npy",
                np.stack(raw_signal_stack, axis=0).astype(np.float32),
            )
            np.save(
                run_dir / "raw_ig_null_all.npy",
                np.stack(raw_null_stack, axis=0).astype(np.float32),
            )

    convergence = run_ig_convergence(
        model,
        dataset,
        cache_np,
        maps_a,
        selected,
        cfg,
        device,
        run_dir,
        logger,
    )
    summary["convergence"] = convergence
    save_json(run_dir / "ig_summary.json", summary)

    if cfg.save_plots:
        plot_ig_diagnostics(
            signal,
            null,
            selected,
            dataset,
            repeat_rows,
            channel_rows,
            faithfulness_rows,
            run_dir,
            cfg,
        )
        if interpolation_rows:
            plot_interpolation(interpolation_rows, run_dir, cfg)

    return summary


def summarize_faithfulness(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    by_k: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        by_k.setdefault(int(r["k"]), []).append(r)
    summary = {}
    for k, values in sorted(by_k.items()):
        top = np.asarray([v["topk_abs_prediction_change"] for v in values], dtype=float)
        rnd = np.asarray([v["random_abs_prediction_change_mean"] for v in values], dtype=float)
        summary[str(k)] = {
            "topk_change_mean": float(top.mean()),
            "random_change_mean": float(rnd.mean()),
            "topk_random_ratio": float(top.mean() / (rnd.mean() + EPS)),
            "fraction_samples_top_gt_random": float(np.mean(top > rnd)),
        }
    return summary


def summarize_interpolation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    nonlin = np.asarray([r["nonlinearity_abs"] for r in rows], dtype=float)
    return {
        "mean_abs_nonlinearity_from_linear_path": float(nonlin.mean()),
        "max_abs_nonlinearity_from_linear_path": float(nonlin.max()),
    }


# =============================================================================
# 12. IG convergence
# =============================================================================
def run_ig_convergence(
    model: torch.nn.Module,
    dataset: Any,
    cache_np: np.ndarray,
    maps_a: list[np.ndarray],
    selected: list[int],
    cfg: Config,
    device: torch.device,
    run_dir: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    steps_list = sorted(set(cfg.ig_convergence_steps))
    reference_steps = max(steps_list)
    n = min(cfg.ig_convergence_samples, len(selected))
    rows: list[dict[str, Any]] = []

    for sample_idx in selected[:n]:
        inputs, _ = get_single_batch(dataset, sample_idx, device)
        correct_idx = int(dataset.original_indices[sample_idx])
        wrong_idx = int(maps_a[0][sample_idx])
        z0 = cache_batch_to_tensor(cache_np, [wrong_idx], device)
        z1 = cache_batch_to_tensor(cache_np, [correct_idx], device)
        u0 = project_text(model, z0).detach()
        u1 = project_text(model, z1).detach()
        p0 = predict_single_projected(model, inputs, u0, device, cfg.use_amp_prediction)
        p1 = predict_single_projected(model, inputs, u1, device, cfg.use_amp_prediction)

        by_steps = {}
        for steps in steps_list:
            by_steps[steps] = projected_differential_ig(
                model,
                inputs,
                u0,
                u1,
                steps,
                cfg.ig_integration_method,
                device,
                cfg.use_amp_ig,
            )
        ref = by_steps[reference_steps]
        ref_norm = float(np.linalg.norm(ref))
        for steps in steps_list:
            ig = by_steps[steps]
            comp = completeness_metrics(ig, p0, p1, cfg.completeness_min_delta)
            rows.append(
                {
                    "sample_idx": sample_idx,
                    "steps": steps,
                    "reference_steps": reference_steps,
                    "relative_l2_to_reference": float(
                        np.linalg.norm(ig - ref) / (ref_norm + EPS)
                    ),
                    "completeness_abs_error": comp["completeness_abs_error"],
                    "completeness_rel_error": comp["completeness_rel_error"],
                }
            )

    write_csv(run_dir / "ig_convergence.csv", rows)
    by_steps_rel: dict[int, list[float]] = {}
    by_steps_comp: dict[int, list[float]] = {}
    for r in rows:
        s = int(r["steps"])
        by_steps_rel.setdefault(s, []).append(float(r["relative_l2_to_reference"]))
        by_steps_comp.setdefault(s, []).append(float(r["completeness_rel_error"]))
    summary = {
        str(s): {
            "mean_relative_l2_to_reference": float(np.mean(by_steps_rel[s])),
            "mean_completeness_rel_error": float(np.mean(by_steps_comp[s])),
        }
        for s in sorted(by_steps_rel)
    }
    save_json(run_dir / "ig_convergence_summary.json", summary)
    if cfg.save_plots:
        plot_convergence(rows, run_dir, cfg)
    logger.info("IG convergence completed on %d samples", n)
    return summary


# =============================================================================
# 13. 绘图
# =============================================================================
def plot_prediction_utility(rows: list[dict[str, Any]], run_dir: Path, cfg: Config) -> None:
    normal_err = np.asarray([r["error_normal"] for r in rows], dtype=float)
    bypass_err = np.asarray([r["error_bypass"] for r in rows], dtype=float)
    wrong_err = np.asarray([r["wrong_error_mean"] for r in rows], dtype=float)
    signal = np.asarray([r["prediction_signal_mean"] for r in rows], dtype=float)
    null = np.asarray([r["prediction_null_mean"] for r in rows], dtype=float)
    benefit = wrong_err - normal_err

    plt.figure(figsize=(7, 5))
    plt.hist(benefit, bins=60)
    plt.axvline(0.0, linestyle="--")
    plt.xlabel("wrong-text error - matched-text error")
    plt.ylabel("count")
    plt.title("Matched-text utility distribution")
    plt.tight_layout()
    plt.savefig(run_dir / "matched_text_utility_distribution.png", dpi=cfg.heatmap_dpi)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(null, signal, s=10, alpha=0.5)
    vmax = float(max(signal.max(initial=0), null.max(initial=0), EPS))
    plt.plot([0, vmax], [0, vmax], linestyle="--")
    plt.xlabel("wrong A -> wrong B prediction change")
    plt.ylabel("matched -> wrong prediction change")
    plt.title("Prediction signal vs null")
    plt.tight_layout()
    plt.savefig(run_dir / "prediction_signal_vs_null.png", dpi=cfg.heatmap_dpi)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(wrong_err, normal_err, s=10, alpha=0.5)
    vmax = float(max(wrong_err.max(initial=0), normal_err.max(initial=0), EPS))
    plt.plot([0, vmax], [0, vmax], linestyle="--")
    plt.xlabel("mean wrong-text absolute error")
    plt.ylabel("matched-text absolute error")
    plt.title("Matched text vs wrong text error")
    plt.tight_layout()
    plt.savefig(run_dir / "matched_vs_wrong_error.png", dpi=cfg.heatmap_dpi)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(bypass_err, normal_err, s=10, alpha=0.5)
    vmax = float(max(bypass_err.max(initial=0), normal_err.max(initial=0), EPS))
    plt.plot([0, vmax], [0, vmax], linestyle="--")
    plt.xlabel("bypass absolute error")
    plt.ylabel("matched-text absolute error")
    plt.title("Matched text vs strict bypass")
    plt.tight_layout()
    plt.savefig(run_dir / "matched_vs_bypass_error.png", dpi=cfg.heatmap_dpi)
    plt.close()


def plot_heatmap(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    path: Path,
    cfg: Config,
    center_zero: bool,
    cbar_label: str,
    channel_order: Optional[np.ndarray] = None,
) -> None:
    mat = np.asarray(matrix, dtype=np.float64)
    if channel_order is not None:
        mat = mat[:, channel_order]
    n_show = min(len(labels), cfg.heatmap_max_samples)
    mat = mat[:n_show]
    labels = labels[:n_show]
    if mat.size == 0:
        return

    h = min(max(5.0, 0.30 * mat.shape[0] + 2.5), 26.0)
    w = min(max(10.0, 0.18 * mat.shape[1] + 4.0), 22.0)
    finite = mat[np.isfinite(mat)]
    if center_zero:
        vmax = float(np.percentile(np.abs(finite), 98)) if finite.size else 1.0
        vmax = max(vmax, EPS)
        vmin = -vmax
        cmap = "coolwarm"
    else:
        vmax = float(np.percentile(finite, 98)) if finite.size else 1.0
        vmax = max(vmax, EPS)
        vmin = 0.0
        cmap = "viridis"

    plt.figure(figsize=(w, h))
    im = plt.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.xlabel("projected text channel")
    plt.ylabel("sample")
    step = max(1, int(math.ceil(mat.shape[1] / 32)))
    xticks = np.arange(0, mat.shape[1], step)
    if channel_order is None:
        xtick_labels = [f"c{i}" for i in xticks]
    else:
        xtick_labels = [f"c{int(channel_order[i])}" for i in xticks]
    plt.xticks(xticks, xtick_labels, rotation=90, fontsize=7)
    if len(labels) <= 80:
        plt.yticks(np.arange(len(labels)), labels, fontsize=7)
    cbar = plt.colorbar(im)
    cbar.set_label(cbar_label)
    plt.tight_layout()
    plt.savefig(path, dpi=cfg.heatmap_dpi)
    plt.close()


def plot_ig_diagnostics(
    signal: np.ndarray,
    null: np.ndarray,
    selected: list[int],
    dataset: Any,
    repeat_rows: list[dict[str, Any]],
    channel_rows: list[dict[str, Any]],
    faithfulness_rows: list[dict[str, Any]],
    run_dir: Path,
    cfg: Config,
) -> None:
    mean_per_sample = signal.mean(axis=1)
    null_abs_per_sample = np.abs(null).mean(axis=1)
    labels = [
        f"s{idx:05d}_{safe_get_id(dataset.data[idx])[:12]}"
        for idx in selected
    ]
    plot_heatmap(
        mean_per_sample,
        labels,
        "Repeated wrong-text mean projected IG: wrong -> matched",
        run_dir / "ig_signal_mean_heatmap.png",
        cfg,
        True,
        "mean signed IG",
    )
    plot_heatmap(
        null_abs_per_sample,
        labels,
        "Projected IG null floor: wrong A -> wrong B",
        run_dir / "ig_null_mean_abs_heatmap.png",
        cfg,
        False,
        "mean |IG|",
    )

    importance_order = np.argsort(
        -np.asarray([r["signal_mean_abs_all"] for r in channel_rows], dtype=float)
    )
    plot_heatmap(
        mean_per_sample,
        labels,
        "Projected IG, channels ranked by global mean |IG|",
        run_dir / "ig_signal_heatmap_ranked_channels.png",
        cfg,
        True,
        "mean signed IG",
        channel_order=importance_order,
    )

    ch = np.arange(len(channel_rows))
    ratio = np.asarray([r["signal_null_ratio"] for r in channel_rows], dtype=float)
    snr = np.asarray([r["mean_within_sample_signed_snr"] for r in channel_rows], dtype=float)
    sign_consistency = np.asarray(
        [r["mean_within_sample_sign_consistency"] for r in channel_rows], dtype=float
    )

    plt.figure(figsize=(12, 5))
    plt.plot(ch, ratio)
    plt.axhline(1.0, linestyle="--")
    plt.xlabel("projected channel")
    plt.ylabel("mean |IG| signal / null")
    plt.title("Channel signal-to-null ratio")
    plt.tight_layout()
    plt.savefig(run_dir / "ig_channel_signal_null_ratio.png", dpi=cfg.heatmap_dpi)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(ch, snr, label="within-sample signed SNR")
    plt.plot(ch, sign_consistency, label="within-sample sign consistency")
    plt.xlabel("projected channel")
    plt.title("Repeated-shuffle channel stability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "ig_channel_stability.png", dpi=cfg.heatmap_dpi)
    plt.close()

    delta = np.asarray([r["signal_prediction_delta"] for r in repeat_rows], dtype=float)
    ig_sum = np.asarray([r["signal_signed_sum"] for r in repeat_rows], dtype=float)
    plt.figure(figsize=(6, 6))
    plt.scatter(delta, ig_sum, s=12, alpha=0.5)
    lo = float(min(delta.min(initial=0), ig_sum.min(initial=0)))
    hi = float(max(delta.max(initial=0), ig_sum.max(initial=0)))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("F(matched) - F(wrong)")
    plt.ylabel("sum(projected IG)")
    plt.title("IG completeness")
    plt.tight_layout()
    plt.savefig(run_dir / "ig_completeness_scatter.png", dpi=cfg.heatmap_dpi)
    plt.close()

    sig_l1 = np.asarray([r["signal_abs_sum_l1"] for r in repeat_rows], dtype=float)
    nul_l1 = np.asarray([r["null_abs_sum_l1"] for r in repeat_rows], dtype=float)
    plt.figure(figsize=(6, 6))
    plt.scatter(nul_l1, sig_l1, s=12, alpha=0.5)
    vmax = float(max(sig_l1.max(initial=0), nul_l1.max(initial=0), EPS))
    plt.plot([0, vmax], [0, vmax], linestyle="--")
    plt.xlabel("null IG L1")
    plt.ylabel("signal IG L1")
    plt.title("Attribution signal vs null")
    plt.tight_layout()
    plt.savefig(run_dir / "ig_signal_vs_null.png", dpi=cfg.heatmap_dpi)
    plt.close()

    if faithfulness_rows:
        by_k: dict[int, list[dict[str, Any]]] = {}
        for r in faithfulness_rows:
            by_k.setdefault(int(r["k"]), []).append(r)
        ks = sorted(by_k)
        top = [np.mean([x["topk_abs_prediction_change"] for x in by_k[k]]) for k in ks]
        rnd = [np.mean([x["random_abs_prediction_change_mean"] for x in by_k[k]]) for k in ks]
        plt.figure(figsize=(7, 5))
        plt.plot(ks, top, marker="o", label="Top-K IG channels")
        plt.plot(ks, rnd, marker="o", label="Random-K channels")
        plt.xlabel("masked projected channels")
        plt.ylabel("mean absolute prediction change")
        plt.title("IG faithfulness: Top-K vs random occlusion")
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / "ig_topk_faithfulness.png", dpi=cfg.heatmap_dpi)
        plt.close()


def plot_interpolation(rows: list[dict[str, Any]], run_dir: Path, cfg: Config) -> None:
    by_sample: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        by_sample.setdefault(int(r["sample_idx"]), []).append(r)
    plt.figure(figsize=(8, 6))
    for sample_idx, values in list(sorted(by_sample.items()))[:12]:
        values = sorted(values, key=lambda x: float(x["alpha"]))
        plt.plot(
            [float(x["alpha"]) for x in values],
            [float(x["prediction"]) for x in values],
            marker="o",
            alpha=0.6,
            label=str(sample_idx) if len(by_sample) <= 8 else None,
        )
    plt.xlabel("alpha: wrong text -> matched text")
    plt.ylabel("prediction")
    plt.title("Projected-text interpolation paths")
    if len(by_sample) <= 8:
        plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "projected_text_interpolation_paths.png", dpi=cfg.heatmap_dpi)
    plt.close()


def plot_convergence(rows: list[dict[str, Any]], run_dir: Path, cfg: Config) -> None:
    by_steps: dict[int, list[float]] = {}
    comp_by_steps: dict[int, list[float]] = {}
    for r in rows:
        s = int(r["steps"])
        by_steps.setdefault(s, []).append(float(r["relative_l2_to_reference"]))
        comp_by_steps.setdefault(s, []).append(float(r["completeness_rel_error"]))
    steps = sorted(by_steps)
    rel = [float(np.mean(by_steps[s])) for s in steps]
    comp = [float(np.mean(comp_by_steps[s])) for s in steps]

    plt.figure(figsize=(7, 5))
    plt.plot(steps, rel, marker="o")
    plt.xlabel("IG steps")
    plt.ylabel("mean relative L2 to max-step reference")
    plt.title("IG numerical convergence")
    plt.tight_layout()
    plt.savefig(run_dir / "ig_convergence.png", dpi=cfg.heatmap_dpi)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(steps, comp, marker="o")
    plt.xlabel("IG steps")
    plt.ylabel("mean completeness relative error")
    plt.title("IG completeness vs integration steps")
    plt.tight_layout()
    plt.savefig(run_dir / "ig_completeness_vs_steps.png", dpi=cfg.heatmap_dpi)
    plt.close()


# =============================================================================
# 14. 自动诊断摘要（启发式，不替代统计判断）
# =============================================================================
def make_final_report(
    model_summary: dict[str, Any],
    utility: dict[str, Any],
    ig: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    notes = []
    flags: dict[str, Optional[bool]] = {}

    if utility:
        wrong = utility.get("utility_vs_wrong_text", {})
        flags["matched_text_improves_over_wrong"] = bool(
            wrong.get("mean_improvement", 0.0) > 0
        )
        boot = wrong.get("bootstrap", {}) if isinstance(wrong, dict) else {}
        ci_low = boot.get("ci_low")
        flags["matched_text_improvement_ci_above_zero"] = (
            None if ci_low is None else bool(ci_low > 0)
        )
        ratio = utility.get("prediction_signal_null_ratio_global")
        flags["prediction_signal_above_null"] = (
            None if ratio is None else bool(ratio > 1.0)
        )

    if ig:
        ratio = ig.get("global_ig_signal_null_ratio")
        flags["ig_signal_above_null"] = None if ratio is None else bool(ratio > 1.0)
        comp = ig.get("median_completeness_rel_error_non_tiny")
        flags["ig_median_completeness_below_5pct"] = (
            None if comp is None else bool(comp < 0.05)
        )
        cancellation = ig.get("mean_sample_cancellation_ratio")
        if cancellation is not None and cancellation > 0.7:
            notes.append(
                "Signed IG sum is strongly affected by positive/negative cancellation; "
                "use L1/L2 and completeness together, not signed sum alone."
            )
        faith = ig.get("faithfulness", {})
        if faith:
            ratios = [v.get("topk_random_ratio", 0.0) for v in faith.values()]
            flags["topk_occlusion_above_random_on_average"] = bool(
                len(ratios) > 0 and np.mean(ratios) > 1.0
            )

    report = {
        "model": {
            "depth": model_summary.get("depth"),
            "graph_embed_dim": model_summary.get("graph_embed_dim"),
            "raw_text_dim": model_summary.get("raw_text_dim"),
            "projected_text_dim": model_summary.get("projected_text_dim"),
            "total_params": model_summary.get("total_params"),
        },
        "heuristic_flags": flags,
        "notes": notes,
        "interpretation_warning": (
            "Flags are engineering heuristics, not universal statistical thresholds. "
            "Primary evidence is the paired utility CI/p-value, signal-vs-null controls, "
            "IG completeness/convergence, and Top-K faithfulness results."
        ),
    }
    save_json(run_dir / "diagnostic_summary.json", report)

    lines = [
        "Fusiformer text utility diagnostics v3 - final checklist",
        "=" * 80,
        f"Model depth              : {model_summary.get('depth')}",
        f"Graph embedding dim      : {model_summary.get('graph_embed_dim')}",
        f"Raw / projected text dim : {model_summary.get('raw_text_dim')} / {model_summary.get('projected_text_dim')}",
        f"Total params             : {model_summary.get('total_params'):,}",
        "",
    ]
    if utility:
        lines += [
            "Prediction utility",
            "-" * 80,
            f"MAE normal              : {utility.get('mae_normal')}",
            f"MAE bypass              : {utility.get('mae_bypass')}",
            f"MAE wrong-text          : {utility.get('mae_wrong_mean_over_repeats')}",
            f"Matched advantage       : {utility.get('utility_vs_wrong_text', {}).get('mean_improvement')}",
            f"Prediction signal/null  : {utility.get('prediction_signal_null_ratio_global')}",
            "",
        ]
    if ig:
        lines += [
            "Projected IG",
            "-" * 80,
            f"IG signal/null L1       : {ig.get('global_ig_signal_null_ratio')}",
            f"Median completeness err : {ig.get('median_completeness_rel_error_non_tiny')}",
            f"Mean cancellation ratio : {ig.get('mean_sample_cancellation_ratio')}",
            f"Mean effective channels : {ig.get('mean_effective_channels')}",
            f"Mean within-sample SNR  : {ig.get('mean_within_sample_channel_snr')}",
            "",
        ]
    lines.append("Heuristic flags")
    lines.append("-" * 80)
    for k, v in flags.items():
        lines.append(f"{k:<45s}: {v}")
    if notes:
        lines += ["", "Notes", "-" * 80, *notes]
    (run_dir / "diagnostic_report.txt").write_text("\n".join(lines), encoding="utf-8")
    return report


# =============================================================================
# 15. main
# =============================================================================
def main() -> None:
    cfg = build_config()
    validate_config(cfg)
    seed_everything(cfg.seed)

    exp_name = cfg.experiment_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = cfg.output_root / exp_name
    run_dir.mkdir(parents=True, exist_ok=False)
    logger = setup_logger(run_dir)

    if cfg.device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device(cfg.device_name)

    save_json(
        run_dir / "diagnostic_config.json",
        {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
    )
    logger.info("========== Fusiformer Text Effect Diagnostics v3 ==========")
    logger.info("run dir: %s", run_dir)
    logger.info("device : %s", device)

    (
        train_module,
        train_args,
        model,
        train_dataset,
        val_dataset,
        test_dataset,
        eval_dataset,
        cache_np,
        cache_path,
        checkpoint_path,
        checkpoint,
    ) = build_project_components(cfg, device, logger)

    validate_data_and_cache(
        [("train", train_dataset), ("val", val_dataset), ("test", test_dataset)],
        cache_np,
        model,
        run_dir,
        logger,
    )
    model_summary = summarize_model(
        model,
        train_args,
        checkpoint_path,
        checkpoint,
        cache_np,
        run_dir,
        logger,
    )

    # 用真实 forward probe 覆盖 projected dimension，避免依赖配置字段。
    probe_idx = int(eval_dataset.original_indices[0])
    probe_z = cache_batch_to_tensor(cache_np, [probe_idx], device)
    probe_u = project_text(model, probe_z)
    model_summary["raw_text_dim"] = int(probe_z.shape[-1])
    model_summary["projected_text_dim"] = int(probe_u.shape[-1])
    save_json(run_dir / "model_structure_summary.json", model_summary)

    needed_repeats = max(cfg.prediction_shuffle_repeats, cfg.ig_shuffle_repeats)
    maps_a = build_wrong_maps(
        eval_dataset, needed_repeats, cfg.shuffle_mode, cfg.shuffle_seed
    )
    maps_b = build_wrong_maps(
        eval_dataset, needed_repeats, cfg.shuffle_mode, cfg.null_seed
    )
    maps_b = repair_null_maps(
        eval_dataset, maps_a, maps_b, cfg.shuffle_mode, cfg.null_seed + 999
    )
    save_donor_maps(eval_dataset, maps_a, maps_b, run_dir)

    run_interface_self_checks(
        model, eval_dataset, cache_np, maps_a, device, cfg, run_dir, logger
    )

    prediction_rows: list[dict[str, Any]] = []
    utility_summary: dict[str, Any] = {}
    if cfg.run_prediction_utility:
        prediction_rows, utility_summary = run_prediction_utility(
            model,
            eval_dataset,
            cache_np,
            maps_a,
            maps_b,
            cfg,
            device,
            run_dir,
            logger,
        )

    ig_summary: dict[str, Any] = {}
    if cfg.run_projected_ig:
        ig_summary = run_projected_ig_diagnostic(
            model,
            eval_dataset,
            cache_np,
            maps_a,
            maps_b,
            prediction_rows,
            cfg,
            device,
            run_dir,
            logger,
        )

    make_final_report(model_summary, utility_summary, ig_summary, run_dir)
    logger.info("all done: %s", run_dir)


if __name__ == "__main__":
    main()
