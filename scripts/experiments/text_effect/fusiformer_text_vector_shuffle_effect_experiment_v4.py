#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fusiformer 文本向量影响权重 + Shuffle 对照实验程序（文件内配置版）
=====================================================

用途
----
本程序用于已经训练完成的 llm_Fusiformer / TextGuidedFusiformerCached 模型，
直接分析 Qwen mean-pool 后的文本向量 inputs["text_emb"] 对预测结果的影响，并加入 shuffle 错配文本向量对照。

它不分析 token / 单词级别，不加载 Qwen，不使用 tokenizer，只使用预计算的
qwen2p5_1p5b_description_meanpool_f32.npy 缓存向量。

新增 shuffle 实验
----------------
1. 全测试集 normal / bypass / shuffle 三模式对比：
   - normal：当前晶体图 + 正确匹配文本向量
   - bypass：当前晶体图 + 关闭文本向量
   - shuffle：当前晶体图 + 随机错配文本向量

2. 单样本 shuffle→normal differential integrated gradients：
   分析从“错配文本向量”替换为“正确文本向量”时，1536 维 raw text_emb
   和 128 维 text_proj channel 对预测变化的贡献。

主要输出
--------
1. test_text_effect_summary.csv
   全测试集 normal vs bypass 对比，衡量文本向量整体贡献。

2. 每个样本目录：
   - sample_summary.json
   - raw_text_vector_importance_1536.csv
   - projected_text_channel_importance_128.csv
   - vector_scaling_curve.csv
   - raw_topk_occlusion.csv
   - projected_topk_occlusion.csv
   - shuffle_predictions.csv
   - differential_raw_1536_integrated_gradients.csv
   - differential_projected_128_integrated_gradients.csv
   - 若干 PNG 图

运行方式
--------
1. 将本文件放在 Fusiformer 项目根目录，例如：
       /root/llm_fusiformer/fusiformer_text_vector_shuffle_effect_experiment.py

2. 修改下面“可编辑运行配置”。

3. 直接运行：
       python fusiformer_text_vector_shuffle_effect_experiment.py

不需要命令行参数。
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import random
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, Iterable, Optional

# 缓解 CUDA 显存碎片；必须在导入 torch 前设置。
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:128",
)

import numpy as np
import torch
from transformers import AutoConfig

try:
    from engine import to_device as project_to_device
except Exception:
    project_to_device = None

from loader.loader_cached import get_dataset, GraphCollectingCached
from loader.transforms import AddAngleFeature, AddEdgeFeature, AddNodeFeature, Compose
from models.TextGuidedFusiformerCached import TextGuidedFusiformer

# matplotlib 只用于保存图片；服务器无显示环境时使用 Agg 后端。
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# 可编辑运行配置
# =============================================================================
# 所有相对路径都相对于本脚本所在项目根目录解析。
BASE_PATH = Path(__file__).resolve().parents[3]

# -----------------------
# 1. 路径配置
# -----------------------
DATA_PATH = "./data/dataset_updated_only.json"
TARGET_NAME = "e_form"

QWEN_MODEL_NAME = "/root/models/Qwen2.5-1.5B-Instruct"
ALLOW_REMOTE_MODEL = False
TEXT_EMBEDDING_CACHE_PATH = (
    "./data/qwen_cache/qwen2p5_1p5b_description_meanpool_f32.npy"
)
TEXT_EMBEDDING_META_PATH = None
TEXT_FIELD = "description"
TEXT_MAX_LENGTH = 512

# 建议使用 best_checkpoint.pt；只有研究最后一个 epoch 时才改为 last_checkpoint.pt。
CHECKPOINT_PATH = (
    "./runs/fusiformer_production/20260811-134939/checkpoints/best_checkpoint.pt"
)

OUTPUT_ROOT = "./runs/text_vector_effect"
EXPERIMENT_NAME = None  # None 表示自动用时间戳创建目录。

# -----------------------
# 2. 数据划分配置：必须与训练时一致
# -----------------------
DATA_SPLIT_RATIO = [0.8, 0.1, 0.1]
DATA_SPLIT_COUNTS = None  # 例如 [64832, 8104, 8104]；不用则设为 None。

# -----------------------
# 3. 模型结构配置：必须与训练 checkpoint 一致
# -----------------------
MODEL_DEPTH = 4
EDGE_INPUT_DIM = 80
TRIPLET_INPUT_DIM = 40
EMBED_DIM = 128
NUM_HEADS = 4
FREEZE_QWEN = True
USE_PRECOMPUTED_TEXT_EMBEDDINGS = True

# -----------------------
# 4. 运行环境
# -----------------------
DEVICE_NAME = "cuda"  # "cuda" 或 "cpu"
RANDOM_SEED = 42
NUM_WORKERS = 4
PIN_MEMORY = True
USE_AMP = False

# 全测试集 normal/bypass 扫描 batch size。
SCAN_BATCH_SIZE = 32

# 单样本解释 batch size 固定为 1，不需要配置。

# -----------------------
# 5. 实验开关
# -----------------------
RUN_TEST_TEXT_EFFECT_SCAN = True
RUN_SAMPLE_ANALYSIS = True

# 样本选择方式：
# - "manual"：使用 MANUAL_TEST_SAMPLE_INDICES
# - "top_abs_delta"：从 test_text_effect_summary.csv 中选择 abs_delta_text 最大的样本
# - "top_text_help"：选择 text_help 最大，即文本最有帮助的样本
# - "top_text_hurt"：选择 text_help 最小，即文本最可能误导的样本
SAMPLE_SELECTION_MODE = "top_abs_delta"
MANUAL_TEST_SAMPLE_INDICES = [0, 100, 1000]
NUM_SAMPLES_TO_ANALYZE = 20

# -----------------------
# 6. 权重/归因配置
# -----------------------
TARGET_INDEX = 0

RUN_RAW_GRAD_X_INPUT = True
RUN_RAW_INTEGRATED_GRADIENTS = True

RUN_PROJECTED_GRAD_X_INPUT = True
RUN_PROJECTED_INTEGRATED_GRADIENTS = True

# IG baseline:
# - "zero"：全零文本向量
# - "train_mean"：训练集文本向量均值，推荐论文主实验使用
IG_BASELINE = "train_mean"
IG_STEPS = 32

# 文本向量强度缩放曲线。alpha=0 表示 baseline，alpha=1 表示真实文本向量。
SCALING_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
SCALING_BASELINE = "zero"  # "zero" 或 "train_mean"

# Top-K 遮挡验证：把高权重维度/通道替换成 baseline 对应值。
RUN_TOPK_OCCLUSION = True
RAW_TOPK_DIMS = [10, 20, 50, 100, 200]
PROJECTED_TOPK_CHANNELS = [5, 10, 20, 40, 80]
RANDOM_OCCLUSION_REPEATS = 5

# -----------------------
# 7. Shuffle 文本向量对照实验
# -----------------------
# 目标：比较“正确匹配文本向量”和“随机错配文本向量”的影响。
RUN_SHUFFLE_TEXT_EXPERIMENT = True

# 每个样本重复多少次 shuffle；正式实验建议 5~10，冒烟测试可设为 2。
NUM_SHUFFLE_REPEATS = 5

# shuffle 方式：
# - "global"：在测试集全局随机错配文本向量，主实验推荐先用这个。
# - "composition"：优先在相同 composition / formula 组内错配；组内不足时退化为 global。
SHUFFLE_MODE = "global"

# 是否在单样本分析中计算 shuffle→normal differential IG。
RUN_SHUFFLE_ATTRIBUTION = True
RUN_DIFFERENTIAL_IG = True
DIFFERENTIAL_IG_STEPS = 32

# 单样本归因时最多使用多少个 shuffle baseline 做平均。设为 1 更快，设为 3~5 更稳。
SAMPLE_SHUFFLE_BASELINES = 3

# shuffle 随机种子独立于训练/采样随机种子。
SHUFFLE_RANDOM_SEED = 2026

# 重要性排序依据：
# - "ig" 优先用 Integrated Gradients
# - "gxi" 用 Gradient × Input
IMPORTANCE_RANKING_METHOD = "ig"

# 是否保存 PNG 图。
SAVE_PLOTS = True

# -----------------------
# 8. 额外热力图可视化
# -----------------------
# 目标：把文本向量权重实验结果进一步整理成论文/汇报更直观的热力图。
RUN_ADDITIONAL_HEATMAPS = True

# heatmaps/ 中 normal/bypass/shuffle 误差热力图展示多少个样本。
HEATMAP_TOP_N_MODE_ERROR = 80

# 多样本 sample × channel 热力图最多展示多少个已分析样本。
# 建议不要超过 80，否则图会过高。
HEATMAP_MAX_SAMPLES = 80

# 在每个单样本目录中额外生成 8×16 的 128 维投影通道热力图；
# 只给前 N 个样本生成，避免文件过多。
HEATMAP_SINGLE_SAMPLE_LIMIT = 40

# 多样本通道热力图默认使用 signed_weight；如果改为 True，则使用 abs_weight。
HEATMAP_USE_ABS_ATTRIBUTION = False

# 保存热力图分辨率。
HEATMAP_DPI = 300

# 统一热力图纵轴尺寸：
# - True：一次实验内所有“样本 × ...”热力图使用相同的样本顺序、相同行数、相同图像高度；
#         便于 mode_error / IG / differential IG / Top-K occlusion 之间逐行比较。
# - False：保留旧版按内容自适应高度的行为。
HEATMAP_UNIFIED_Y_AXIS = True

# 每个样本行在纵轴方向的目标高度，单位 inch。数值越大，每一行越高。
HEATMAP_ROW_HEIGHT_INCH = 0.28

# 统一样本热力图的额外高度，预留给标题、x/y 标签和 colorbar。
HEATMAP_EXTRA_HEIGHT_INCH = 2.8

# 统一样本热力图的高度下限 / 上限，避免样本很少时过矮或样本很多时过高。
HEATMAP_MIN_HEIGHT_INCH = 6.0
HEATMAP_MAX_HEIGHT_INCH = 28.0

# 单样本 8×16 projected channel block 热力图也固定使用同一尺寸。
HEATMAP_SINGLE_BLOCK_WIDTH_INCH = 10.5
HEATMAP_SINGLE_BLOCK_HEIGHT_INCH = 5.0

# 日志间隔。
SCAN_LOG_INTERVAL = 50
SAMPLE_LOG_INTERVAL = 1


# =============================================================================
# 工具函数
# =============================================================================
def resolve_path(path_like: str | Path | None) -> Optional[Path]:
    if path_like is None:
        return None
    p = Path(path_like).expanduser()
    if not p.is_absolute():
        p = BASE_PATH / p
    return p.resolve()


def setup_logger(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("text_vector_effect")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(run_dir / "experiment.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


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
    if isinstance(outputs, (list, tuple)):
        return torch.cat(list(outputs), dim=-1)
    return outputs


def normalize_targets(targets: Any, device: torch.device) -> torch.Tensor:
    if isinstance(targets, list):
        targets = torch.cat(targets, dim=-1)
    elif isinstance(targets, tuple):
        targets = torch.cat(list(targets), dim=-1)
    return targets.to(device, non_blocking=True)


def selected_output(outputs: Any, target_index: int) -> torch.Tensor:
    y = normalize_outputs(outputs)
    if y.ndim == 0:
        return y.reshape(1)[0]
    if y.ndim == 1:
        if y.numel() == 1:
            return y.reshape(1)[0]
        return y[target_index]
    return y[:, target_index]


def scalar_selected_output(outputs: Any, target_index: int) -> torch.Tensor:
    y = selected_output(outputs, target_index)
    if y.numel() != 1:
        raise ValueError(f"expected single-sample scalar output, got shape={tuple(y.shape)}")
    return y.reshape(())


def clone_inputs_with_text_emb(inputs: dict[str, Any], text_emb: torch.Tensor) -> dict[str, Any]:
    copied = dict(inputs)
    copied["text_emb"] = text_emb
    copied["text_mode"] = "normal"
    return copied


def clone_inputs_with_mode(inputs: dict[str, Any], mode: str) -> dict[str, Any]:
    copied = dict(inputs)
    copied["text_mode"] = mode
    return copied


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Optional[list[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def as_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0])
    return float(value)


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


@dataclass
class RuntimeConfig:
    data_path: Path
    target_name: str
    qwen_model_name: str
    allow_remote_model: bool
    text_embedding_cache_path: Path
    text_embedding_meta_path: Optional[Path]
    text_field: str
    text_max_length: int
    checkpoint_path: Path
    output_root: Path
    experiment_name: Optional[str]
    split_ratio: Optional[list[float]]
    split_counts: Optional[list[int]]
    depth: int
    edge_input_dim: int
    triplet_input_dim: int
    embed_dim: int
    num_heads: int
    freeze_qwen: bool
    use_precomputed_text_embeddings: bool
    device_name: str
    seed: int
    num_workers: int
    pin_memory: bool
    use_amp: bool
    scan_batch_size: int
    run_test_text_effect_scan: bool
    run_sample_analysis: bool
    sample_selection_mode: str
    manual_test_sample_indices: list[int]
    num_samples_to_analyze: int
    target_index: int
    run_raw_grad_x_input: bool
    run_raw_integrated_gradients: bool
    run_projected_grad_x_input: bool
    run_projected_integrated_gradients: bool
    ig_baseline: str
    ig_steps: int
    scaling_alphas: list[float]
    scaling_baseline: str
    run_topk_occlusion: bool
    raw_topk_dims: list[int]
    projected_topk_channels: list[int]
    random_occlusion_repeats: int
    run_shuffle_text_experiment: bool
    num_shuffle_repeats: int
    shuffle_mode: str
    run_shuffle_attribution: bool
    run_differential_ig: bool
    differential_ig_steps: int
    sample_shuffle_baselines: int
    shuffle_random_seed: int
    importance_ranking_method: str
    save_plots: bool
    run_additional_heatmaps: bool
    heatmap_top_n_mode_error: int
    heatmap_max_samples: int
    heatmap_single_sample_limit: int
    heatmap_use_abs_attribution: bool
    heatmap_dpi: int
    heatmap_unified_y_axis: bool
    heatmap_row_height_inch: float
    heatmap_extra_height_inch: float
    heatmap_min_height_inch: float
    heatmap_max_height_inch: float
    heatmap_single_block_width_inch: float
    heatmap_single_block_height_inch: float


def build_config() -> RuntimeConfig:
    return RuntimeConfig(
        data_path=resolve_path(DATA_PATH),
        target_name=TARGET_NAME,
        qwen_model_name=QWEN_MODEL_NAME,
        allow_remote_model=bool(ALLOW_REMOTE_MODEL),
        text_embedding_cache_path=resolve_path(TEXT_EMBEDDING_CACHE_PATH),
        text_embedding_meta_path=resolve_path(TEXT_EMBEDDING_META_PATH),
        text_field=TEXT_FIELD,
        text_max_length=int(TEXT_MAX_LENGTH),
        checkpoint_path=resolve_path(CHECKPOINT_PATH),
        output_root=resolve_path(OUTPUT_ROOT),
        experiment_name=EXPERIMENT_NAME,
        split_ratio=None if DATA_SPLIT_COUNTS is not None else list(DATA_SPLIT_RATIO),
        split_counts=None if DATA_SPLIT_COUNTS is None else list(DATA_SPLIT_COUNTS),
        depth=int(MODEL_DEPTH),
        edge_input_dim=int(EDGE_INPUT_DIM),
        triplet_input_dim=int(TRIPLET_INPUT_DIM),
        embed_dim=int(EMBED_DIM),
        num_heads=int(NUM_HEADS),
        freeze_qwen=bool(FREEZE_QWEN),
        use_precomputed_text_embeddings=bool(USE_PRECOMPUTED_TEXT_EMBEDDINGS),
        device_name=DEVICE_NAME,
        seed=int(RANDOM_SEED),
        num_workers=int(NUM_WORKERS),
        pin_memory=bool(PIN_MEMORY),
        use_amp=bool(USE_AMP),
        scan_batch_size=int(SCAN_BATCH_SIZE),
        run_test_text_effect_scan=bool(RUN_TEST_TEXT_EFFECT_SCAN),
        run_sample_analysis=bool(RUN_SAMPLE_ANALYSIS),
        sample_selection_mode=str(SAMPLE_SELECTION_MODE),
        manual_test_sample_indices=list(MANUAL_TEST_SAMPLE_INDICES),
        num_samples_to_analyze=int(NUM_SAMPLES_TO_ANALYZE),
        target_index=int(TARGET_INDEX),
        run_raw_grad_x_input=bool(RUN_RAW_GRAD_X_INPUT),
        run_raw_integrated_gradients=bool(RUN_RAW_INTEGRATED_GRADIENTS),
        run_projected_grad_x_input=bool(RUN_PROJECTED_GRAD_X_INPUT),
        run_projected_integrated_gradients=bool(RUN_PROJECTED_INTEGRATED_GRADIENTS),
        ig_baseline=str(IG_BASELINE),
        ig_steps=int(IG_STEPS),
        scaling_alphas=list(SCALING_ALPHAS),
        scaling_baseline=str(SCALING_BASELINE),
        run_topk_occlusion=bool(RUN_TOPK_OCCLUSION),
        raw_topk_dims=list(RAW_TOPK_DIMS),
        projected_topk_channels=list(PROJECTED_TOPK_CHANNELS),
        random_occlusion_repeats=int(RANDOM_OCCLUSION_REPEATS),
        run_shuffle_text_experiment=bool(RUN_SHUFFLE_TEXT_EXPERIMENT),
        num_shuffle_repeats=int(NUM_SHUFFLE_REPEATS),
        shuffle_mode=str(SHUFFLE_MODE),
        run_shuffle_attribution=bool(RUN_SHUFFLE_ATTRIBUTION),
        run_differential_ig=bool(RUN_DIFFERENTIAL_IG),
        differential_ig_steps=int(DIFFERENTIAL_IG_STEPS),
        sample_shuffle_baselines=int(SAMPLE_SHUFFLE_BASELINES),
        shuffle_random_seed=int(SHUFFLE_RANDOM_SEED),
        importance_ranking_method=str(IMPORTANCE_RANKING_METHOD),
        save_plots=bool(SAVE_PLOTS),
        run_additional_heatmaps=bool(RUN_ADDITIONAL_HEATMAPS),
        heatmap_top_n_mode_error=int(HEATMAP_TOP_N_MODE_ERROR),
        heatmap_max_samples=int(HEATMAP_MAX_SAMPLES),
        heatmap_single_sample_limit=int(HEATMAP_SINGLE_SAMPLE_LIMIT),
        heatmap_use_abs_attribution=bool(HEATMAP_USE_ABS_ATTRIBUTION),
        heatmap_dpi=int(HEATMAP_DPI),
        heatmap_unified_y_axis=bool(HEATMAP_UNIFIED_Y_AXIS),
        heatmap_row_height_inch=float(HEATMAP_ROW_HEIGHT_INCH),
        heatmap_extra_height_inch=float(HEATMAP_EXTRA_HEIGHT_INCH),
        heatmap_min_height_inch=float(HEATMAP_MIN_HEIGHT_INCH),
        heatmap_max_height_inch=float(HEATMAP_MAX_HEIGHT_INCH),
        heatmap_single_block_width_inch=float(HEATMAP_SINGLE_BLOCK_WIDTH_INCH),
        heatmap_single_block_height_inch=float(HEATMAP_SINGLE_BLOCK_HEIGHT_INCH),
    )


def validate_config(cfg: RuntimeConfig) -> None:
    for p, name in [
        (cfg.data_path, "DATA_PATH"),
        (cfg.text_embedding_cache_path, "TEXT_EMBEDDING_CACHE_PATH"),
        (cfg.checkpoint_path, "CHECKPOINT_PATH"),
    ]:
        if p is None or not p.is_file():
            raise FileNotFoundError(f"{name} does not exist: {p}")
    if cfg.ig_baseline not in {"zero", "train_mean"}:
        raise ValueError("IG_BASELINE must be 'zero' or 'train_mean'")
    if cfg.scaling_baseline not in {"zero", "train_mean"}:
        raise ValueError("SCALING_BASELINE must be 'zero' or 'train_mean'")
    if cfg.sample_selection_mode not in {
        "manual",
        "top_abs_delta",
        "top_text_help",
        "top_text_hurt",
        "top_abs_delta_shuffle",
        "top_text_help_shuffle",
        "top_shuffle_hurt",
    }:
        raise ValueError("unsupported SAMPLE_SELECTION_MODE")
    if cfg.shuffle_mode not in {"global", "composition"}:
        raise ValueError("SHUFFLE_MODE must be 'global' or 'composition'")
    if cfg.heatmap_top_n_mode_error <= 0:
        raise ValueError("HEATMAP_TOP_N_MODE_ERROR must be positive")
    if cfg.heatmap_max_samples <= 0:
        raise ValueError("HEATMAP_MAX_SAMPLES must be positive")
    if cfg.heatmap_single_sample_limit < 0:
        raise ValueError("HEATMAP_SINGLE_SAMPLE_LIMIT must be non-negative")
    if cfg.heatmap_dpi <= 0:
        raise ValueError("HEATMAP_DPI must be positive")
    if cfg.heatmap_row_height_inch <= 0:
        raise ValueError("HEATMAP_ROW_HEIGHT_INCH must be positive")
    if cfg.heatmap_extra_height_inch < 0:
        raise ValueError("HEATMAP_EXTRA_HEIGHT_INCH must be non-negative")
    if cfg.heatmap_min_height_inch <= 0 or cfg.heatmap_max_height_inch <= 0:
        raise ValueError("HEATMAP_MIN_HEIGHT_INCH and HEATMAP_MAX_HEIGHT_INCH must be positive")
    if cfg.heatmap_min_height_inch > cfg.heatmap_max_height_inch:
        raise ValueError("HEATMAP_MIN_HEIGHT_INCH cannot be larger than HEATMAP_MAX_HEIGHT_INCH")
    if cfg.heatmap_single_block_width_inch <= 0 or cfg.heatmap_single_block_height_inch <= 0:
        raise ValueError("single block heatmap size must be positive")
    if cfg.num_shuffle_repeats < 1:
        raise ValueError("NUM_SHUFFLE_REPEATS must be >= 1")
    if cfg.differential_ig_steps < 1:
        raise ValueError("DIFFERENTIAL_IG_STEPS must be >= 1")
    if cfg.sample_shuffle_baselines < 1:
        raise ValueError("SAMPLE_SHUFFLE_BASELINES must be >= 1")
    if cfg.importance_ranking_method not in {"ig", "gxi"}:
        raise ValueError("IMPORTANCE_RANKING_METHOD must be 'ig' or 'gxi'")
    if cfg.ig_steps < 1:
        raise ValueError("IG_STEPS must be >= 1")


# =============================================================================
# 数据、模型、checkpoint
# =============================================================================
def build_datasets(cfg: RuntimeConfig, logger: logging.Logger):
    transform = Compose(
        [
            AddNodeFeature(),
            AddEdgeFeature(),
            AddAngleFeature(),
            GraphCollectingCached(["graph", "line_graph"], [cfg.target_name]),
        ]
    )
    train_dataset, val_dataset, test_dataset = get_dataset(
        data_path=str(cfg.data_path),
        ratio_train_val_test=cfg.split_ratio,
        n_train_val_test=cfg.split_counts,
        transforms=transform,
        graph=True,
        line_graph=True,
        qwen_model_name=cfg.qwen_model_name,
        text_max_length=cfg.text_max_length,
        text_field=cfg.text_field,
        text_embedding_cache_path=str(cfg.text_embedding_cache_path),
        text_embedding_meta_path=(
            str(cfg.text_embedding_meta_path) if cfg.text_embedding_meta_path is not None else None
        ),
    )
    logger.info("train samples: %d", len(train_dataset))
    logger.info("val samples: %d", 0 if val_dataset is None else len(val_dataset))
    logger.info("test samples: %d", 0 if test_dataset is None else len(test_dataset))
    return train_dataset, val_dataset, test_dataset


def build_model(cfg: RuntimeConfig, device: torch.device) -> torch.nn.Module:
    qwen_config = AutoConfig.from_pretrained(
        cfg.qwen_model_name,
        local_files_only=not cfg.allow_remote_model,
        trust_remote_code=True,
    )
    qwen_hidden_size = int(qwen_config.hidden_size)
    model = TextGuidedFusiformer(
        targets=[cfg.target_name],
        depth=cfg.depth,
        edge_input_dim=cfg.edge_input_dim,
        triplet_input_dim=cfg.triplet_input_dim,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        text_dim=qwen_hidden_size,
        qwen_model_name=cfg.qwen_model_name,
        freeze_qwen=cfg.freeze_qwen,
        load_qwen=False,
    )
    model.use_precomputed_text_embeddings = True
    model.freeze_qwen = cfg.freeze_qwen
    return model.to(device)


def extract_model_state(checkpoint: Any) -> tuple[dict[str, torch.Tensor], list[str]]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
        excluded = checkpoint.get("excluded_model_prefixes", [])
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
        excluded = checkpoint.get("excluded_model_prefixes", [])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
        excluded = checkpoint.get("excluded_model_prefixes", [])
    elif isinstance(checkpoint, dict) and checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        state = checkpoint
        excluded = []
    else:
        raise KeyError("checkpoint does not contain model parameters")
    if state and all(str(name).startswith("module.") for name in state.keys()):
        state = {name[len("module."):]: value for name, value in state.items()}
    return state, list(excluded)


def load_checkpoint_file(checkpoint_path: Path) -> Any:
    """
    Load a trusted local training checkpoint.

    PyTorch 2.6 changed torch.load's default weights_only value to True,
    which can reject checkpoints that contain optimizer states / numpy objects.
    The checkpoints used here are produced by this project itself, so we
    explicitly set weights_only=False. For older PyTorch versions that do not
    accept this keyword, fall back to the old call signature.
    """
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def load_model_state(model: torch.nn.Module, checkpoint_path: Path, logger: logging.Logger) -> None:
    checkpoint = load_checkpoint_file(checkpoint_path)
    state, excluded = extract_model_state(checkpoint)
    incompatible = model.load_state_dict(state, strict=False)
    missing = [
        name
        for name in incompatible.missing_keys
        if not any(name.startswith(prefix) for prefix in excluded)
    ]
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            "model state loading incomplete; "
            f"missing={missing[:30]}, unexpected={unexpected[:30]}"
        )
    logger.info("loaded model tensors: %d", len(state))
    if excluded:
        logger.info("checkpoint excluded prefixes: %s", excluded)


def make_eval_loader(dataset: Any, cfg: RuntimeConfig, device: torch.device) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.scan_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and device.type == "cuda",
        collate_fn=dataset.collect(),
        persistent_workers=cfg.num_workers > 0,
    )


def get_single_batch(dataset: Any, sample_idx: int, device: torch.device) -> tuple[dict[str, Any], torch.Tensor]:
    collate_fn = dataset.collect()
    inputs, targets = collate_fn([dataset[sample_idx]])
    return to_device(inputs, device), normalize_targets(targets, device)


# =============================================================================
# 预测与扫描
# =============================================================================
@torch.no_grad()
def predict_batch(
    model: torch.nn.Module,
    inputs: dict[str, Any],
    cfg: RuntimeConfig,
    mode: str,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    run_inputs = clone_inputs_with_mode(inputs, mode)
    with autocast_context(device, cfg.use_amp):
        outputs = model(run_inputs)
    return selected_output(outputs, cfg.target_index).detach().float().cpu()


@torch.no_grad()
def predict_single_with_text_emb(
    model: torch.nn.Module,
    inputs: dict[str, Any],
    text_emb: torch.Tensor,
    cfg: RuntimeConfig,
    device: torch.device,
) -> float:
    model.eval()
    run_inputs = clone_inputs_with_text_emb(inputs, text_emb)
    with autocast_context(device, cfg.use_amp):
        outputs = model(run_inputs)
    return as_float(scalar_selected_output(outputs, cfg.target_index))


@torch.no_grad()
def predict_single_mode(
    model: torch.nn.Module,
    inputs: dict[str, Any],
    cfg: RuntimeConfig,
    mode: str,
    device: torch.device,
) -> float:
    model.eval()
    run_inputs = clone_inputs_with_mode(inputs, mode)
    with autocast_context(device, cfg.use_amp):
        outputs = model(run_inputs)
    return as_float(scalar_selected_output(outputs, cfg.target_index))


def run_text_effect_scan(
    model: torch.nn.Module,
    test_dataset: Any,
    cfg: RuntimeConfig,
    device: torch.device,
    run_dir: Path,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    logger.info("========== Experiment 1: test-set normal vs bypass scan ==========")
    loader = make_eval_loader(test_dataset, cfg, device)
    rows: list[dict[str, Any]] = []
    sample_offset = 0
    t0 = time.time()

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = to_device(inputs, device)
        targets = normalize_targets(targets, device)
        y_true = targets[:, cfg.target_index].detach().float().cpu()

        pred_normal = predict_batch(model, inputs, cfg, "normal", device)
        pred_bypass = predict_batch(model, inputs, cfg, "bypass", device)

        batch_size = int(y_true.shape[0])
        for i in range(batch_size):
            test_idx = sample_offset + i
            info = test_dataset.data[test_idx]
            original_idx = int(test_dataset.original_indices[test_idx])
            true_i = float(y_true[i])
            normal_i = float(pred_normal[i])
            bypass_i = float(pred_bypass[i])
            delta = normal_i - bypass_i
            err_normal = abs(normal_i - true_i)
            err_bypass = abs(bypass_i - true_i)
            rows.append(
                {
                    "test_sample_idx": test_idx,
                    "original_idx": original_idx,
                    "material_id": safe_get_id(info),
                    "true_y": true_i,
                    "pred_normal": normal_i,
                    "pred_bypass": bypass_i,
                    "delta_text": delta,
                    "abs_delta_text": abs(delta),
                    "error_normal": err_normal,
                    "error_bypass": err_bypass,
                    "text_help": err_bypass - err_normal,
                }
            )
        sample_offset += batch_size

        if (batch_idx + 1) % SCAN_LOG_INTERVAL == 0:
            logger.info("scanned batches: %d/%d", batch_idx + 1, len(loader))

    out_csv = run_dir / "test_text_effect_summary.csv"
    write_csv(out_csv, rows)

    y_true_all = np.array([r["true_y"] for r in rows], dtype=np.float64)
    normal_all = np.array([r["pred_normal"] for r in rows], dtype=np.float64)
    bypass_all = np.array([r["pred_bypass"] for r in rows], dtype=np.float64)
    delta_all = normal_all - bypass_all
    metrics = {
        "num_samples": int(len(rows)),
        "mae_normal": float(np.mean(np.abs(normal_all - y_true_all))),
        "mae_bypass": float(np.mean(np.abs(bypass_all - y_true_all))),
        "rmse_normal": float(np.sqrt(np.mean((normal_all - y_true_all) ** 2))),
        "rmse_bypass": float(np.sqrt(np.mean((bypass_all - y_true_all) ** 2))),
        "mean_delta_text": float(np.mean(delta_all)),
        "mean_abs_delta_text": float(np.mean(np.abs(delta_all))),
        "median_abs_delta_text": float(np.median(np.abs(delta_all))),
        "text_help_mean": float(np.mean(np.abs(bypass_all - y_true_all) - np.abs(normal_all - y_true_all))),
        "elapsed_seconds": float(time.time() - t0),
    }
    save_json(run_dir / "test_text_effect_metrics.json", metrics)
    logger.info("normal MAE: %.6f | bypass MAE: %.6f | mean |delta_text|: %.6f",
                metrics["mae_normal"], metrics["mae_bypass"], metrics["mean_abs_delta_text"])

    if cfg.save_plots:
        plot_scan_distributions(rows, run_dir)
    return rows


def get_composition_key(info: dict[str, Any]) -> str:
    """尽量从数据字典中提取 composition/formula，用于 composition-controlled shuffle。"""
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
    # 有些数据可能保存 elements / atom_types；退化为元素集合键。
    for key in ["elements", "atom_types", "species"]:
        if key in info and info[key] not in (None, ""):
            value = info[key]
            if isinstance(value, (list, tuple, set)):
                return "|".join(sorted(str(v) for v in value))
            return str(value)
    return "__unknown__"


def _make_deranged_permutation(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """生成尽量不与自身相同的 permutation；长度为 1 时只能返回自身。"""
    n = int(len(values))
    if n <= 1:
        return values.copy()
    for _ in range(100):
        perm = values[rng.permutation(n)]
        if not np.any(perm == values):
            return perm
    # 兜底：循环平移保证无自身匹配。
    return np.roll(values, 1)


def build_shuffle_maps(
    test_dataset: Any,
    cfg: RuntimeConfig,
    run_dir: Path,
    logger: logging.Logger,
) -> list[np.ndarray]:
    """
    返回长度为 NUM_SHUFFLE_REPEATS 的列表；每个元素 shape=[len(test_dataset)]，
    其中第 i 个值是 sample i 使用的错配文本向量在原始 JSON/cache 中的 original_idx。
    """
    rng = np.random.default_rng(cfg.shuffle_random_seed)
    n = len(test_dataset)
    original_indices = np.array(test_dataset.original_indices, dtype=np.int64)
    maps: list[np.ndarray] = []

    if cfg.shuffle_mode == "global":
        for _ in range(cfg.num_shuffle_repeats):
            maps.append(_make_deranged_permutation(original_indices, rng))
    elif cfg.shuffle_mode == "composition":
        # 在相同 composition/formula 组内优先错配；组内只有一个样本时，用 global fallback。
        groups: dict[str, list[int]] = {}
        for test_idx, info in enumerate(test_dataset.data):
            groups.setdefault(get_composition_key(info), []).append(test_idx)
        fallback_template = _make_deranged_permutation(original_indices, rng)
        for _ in range(cfg.num_shuffle_repeats):
            mapped = fallback_template.copy()
            for _, test_positions in groups.items():
                if len(test_positions) < 2:
                    continue
                pos = np.array(test_positions, dtype=np.int64)
                vals = original_indices[pos]
                mapped[pos] = _make_deranged_permutation(vals, rng)
            maps.append(mapped.copy())
    else:
        raise ValueError(f"unsupported shuffle_mode={cfg.shuffle_mode}")

    # 保存映射，便于复现。
    rows = []
    max_repeats_to_write = min(cfg.num_shuffle_repeats, 20)
    for test_idx in range(n):
        row = {
            "test_sample_idx": int(test_idx),
            "original_idx": int(original_indices[test_idx]),
            "material_id": safe_get_id(test_dataset.data[test_idx]),
        }
        for r in range(max_repeats_to_write):
            row[f"shuffle_original_idx_r{r}"] = int(maps[r][test_idx])
        rows.append(row)
    write_csv(run_dir / "shuffle_index_map.csv", rows)
    logger.info(
        "built %d shuffle maps | mode=%s | samples=%d",
        len(maps),
        cfg.shuffle_mode,
        n,
    )
    return maps


def load_cache_memmap(cache_path: Path) -> np.ndarray:
    return np.load(cache_path, mmap_mode="r")


def cache_batch_to_tensor(cache_np: np.ndarray, original_indices: Iterable[int], device: torch.device) -> torch.Tensor:
    arr = np.asarray(cache_np[np.array(list(original_indices), dtype=np.int64)], dtype=np.float32)
    return torch.from_numpy(arr).to(device=device, dtype=torch.float32)


def run_text_effect_shuffle_scan(
    model: torch.nn.Module,
    test_dataset: Any,
    cache_np: np.ndarray,
    shuffle_maps: list[np.ndarray],
    cfg: RuntimeConfig,
    device: torch.device,
    run_dir: Path,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    """全测试集 normal / bypass / shuffle 联合扫描。"""
    logger.info("========== Experiment 1: test-set normal / bypass / shuffle scan ==========")
    loader = make_eval_loader(test_dataset, cfg, device)
    rows: list[dict[str, Any]] = []
    sample_offset = 0
    t0 = time.time()

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = to_device(inputs, device)
        targets = normalize_targets(targets, device)
        y_true = targets[:, cfg.target_index].detach().float().cpu()

        pred_normal = predict_batch(model, inputs, cfg, "normal", device)
        pred_bypass = predict_batch(model, inputs, cfg, "bypass", device)
        batch_size = int(y_true.shape[0])
        batch_test_indices = np.arange(sample_offset, sample_offset + batch_size, dtype=np.int64)

        shuffle_pred_stack = []
        for r, shuffle_map in enumerate(shuffle_maps):
            shuf_orig = shuffle_map[batch_test_indices]
            shuf_text = cache_batch_to_tensor(cache_np, shuf_orig, device)
            run_inputs = clone_inputs_with_text_emb(inputs, shuf_text)
            with torch.no_grad():
                with autocast_context(device, cfg.use_amp):
                    pred_shuffle = selected_output(model(run_inputs), cfg.target_index).detach().float().cpu()
            shuffle_pred_stack.append(pred_shuffle.numpy())
        shuffle_pred_arr = np.stack(shuffle_pred_stack, axis=0)  # [R, B]
        shuffle_mean = shuffle_pred_arr.mean(axis=0)
        shuffle_std = shuffle_pred_arr.std(axis=0)

        for i in range(batch_size):
            test_idx = sample_offset + i
            info = test_dataset.data[test_idx]
            original_idx = int(test_dataset.original_indices[test_idx])
            true_i = float(y_true[i])
            normal_i = float(pred_normal[i])
            bypass_i = float(pred_bypass[i])
            shuffle_i = float(shuffle_mean[i])
            shuffle_std_i = float(shuffle_std[i])
            delta = normal_i - bypass_i
            delta_shuffle = normal_i - shuffle_i
            err_normal = abs(normal_i - true_i)
            err_bypass = abs(bypass_i - true_i)
            err_shuffle = abs(shuffle_i - true_i)
            rows.append(
                {
                    "test_sample_idx": test_idx,
                    "original_idx": original_idx,
                    "material_id": safe_get_id(info),
                    "true_y": true_i,
                    "pred_normal": normal_i,
                    "pred_bypass": bypass_i,
                    "pred_shuffle_mean": shuffle_i,
                    "pred_shuffle_std": shuffle_std_i,
                    "delta_text": delta,
                    "abs_delta_text": abs(delta),
                    "delta_shuffle": delta_shuffle,
                    "abs_delta_shuffle": abs(delta_shuffle),
                    "error_normal": err_normal,
                    "error_bypass": err_bypass,
                    "error_shuffle_mean": err_shuffle,
                    "text_help": err_bypass - err_normal,
                    "text_help_shuffle": err_shuffle - err_normal,
                    "shuffle_vs_bypass_help": err_shuffle - err_bypass,
                }
            )
        sample_offset += batch_size

        if (batch_idx + 1) % SCAN_LOG_INTERVAL == 0:
            logger.info("scanned batches: %d/%d", batch_idx + 1, len(loader))

    write_csv(run_dir / "test_text_effect_summary.csv", rows)
    write_csv(run_dir / "test_shuffle_effect_summary.csv", rows)

    y_true_all = np.array([r["true_y"] for r in rows], dtype=np.float64)
    normal_all = np.array([r["pred_normal"] for r in rows], dtype=np.float64)
    bypass_all = np.array([r["pred_bypass"] for r in rows], dtype=np.float64)
    shuffle_all = np.array([r["pred_shuffle_mean"] for r in rows], dtype=np.float64)
    metrics = {
        "num_samples": int(len(rows)),
        "num_shuffle_repeats": int(cfg.num_shuffle_repeats),
        "shuffle_mode": cfg.shuffle_mode,
        "mae_normal": float(np.mean(np.abs(normal_all - y_true_all))),
        "mae_bypass": float(np.mean(np.abs(bypass_all - y_true_all))),
        "mae_shuffle_mean": float(np.mean(np.abs(shuffle_all - y_true_all))),
        "rmse_normal": float(np.sqrt(np.mean((normal_all - y_true_all) ** 2))),
        "rmse_bypass": float(np.sqrt(np.mean((bypass_all - y_true_all) ** 2))),
        "rmse_shuffle_mean": float(np.sqrt(np.mean((shuffle_all - y_true_all) ** 2))),
        "mean_abs_delta_text": float(np.mean(np.abs(normal_all - bypass_all))),
        "mean_abs_delta_shuffle": float(np.mean(np.abs(normal_all - shuffle_all))),
        "median_abs_delta_shuffle": float(np.median(np.abs(normal_all - shuffle_all))),
        "text_help_bypass_mean": float(np.mean(np.abs(bypass_all - y_true_all) - np.abs(normal_all - y_true_all))),
        "text_help_shuffle_mean": float(np.mean(np.abs(shuffle_all - y_true_all) - np.abs(normal_all - y_true_all))),
        "elapsed_seconds": float(time.time() - t0),
    }
    save_json(run_dir / "test_text_effect_metrics.json", metrics)
    save_json(run_dir / "shuffle_metrics.json", metrics)
    logger.info(
        "MAE | normal=%.6f | bypass=%.6f | shuffle=%.6f | mean |normal-shuffle|=%.6f",
        metrics["mae_normal"],
        metrics["mae_bypass"],
        metrics["mae_shuffle_mean"],
        metrics["mean_abs_delta_shuffle"],
    )

    if cfg.save_plots:
        plot_scan_distributions(rows, run_dir)
        plot_shuffle_distributions(rows, run_dir)
    return rows


# =============================================================================
# 额外热力图可视化
# =============================================================================
def _csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file as list of dictionaries. Return [] if file is missing."""
    path = Path(path)
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(value: Any, default: float = np.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _safe_percentile(values: np.ndarray, percentile: float, default: float = 1.0) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return default
    val = float(np.nanpercentile(arr, percentile))
    if not np.isfinite(val) or val <= 0:
        return default
    return val


def save_matrix_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    output_path: Path,
    *,
    center_zero: bool,
    colorbar_label: str,
    xlabel: str,
    ylabel: str,
    dpi: int = 300,
    max_xticks: int = 32,
    max_yticks: int = 80,
    fig_width: Optional[float] = None,
    fig_height: Optional[float] = None,
) -> None:
    """
    Save a general 2D heatmap using matplotlib only.

    fig_height 可由 generate_additional_heatmaps 统一传入。这样同一次实验中
    mode_error / projected IG / differential IG / occlusion 等“样本 × ...”
    热力图具有相同的纵轴物理尺寸，便于逐行比较样本。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2 or mat.size == 0:
        raise ValueError(f"matrix must be a non-empty 2D array, got shape={mat.shape}")

    if center_zero:
        vmax = _safe_percentile(np.abs(mat), 98, default=1.0)
        vmin = -vmax
        cmap = "coolwarm"
    else:
        vmin = 0.0
        vmax = _safe_percentile(mat, 98, default=1.0)
        cmap = "viridis"

    fig_w = fig_width if fig_width is not None else max(7.0, min(20.0, 0.16 * mat.shape[1] + 4.0))
    fig_h = fig_height if fig_height is not None else max(3.8, min(18.0, 0.24 * mat.shape[0] + 2.5))
    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if mat.shape[1] <= max_xticks:
        xticks = np.arange(mat.shape[1])
    else:
        step = int(np.ceil(mat.shape[1] / max_xticks))
        xticks = np.arange(0, mat.shape[1], step)
    plt.xticks(xticks, [col_labels[int(i)] for i in xticks], rotation=90, fontsize=7)

    if mat.shape[0] <= max_yticks:
        yticks = np.arange(mat.shape[0])
    else:
        step = int(np.ceil(mat.shape[0] / max_yticks))
        yticks = np.arange(0, mat.shape[0], step)
    plt.yticks(yticks, [row_labels[int(i)] for i in yticks], fontsize=7)

    cbar = plt.colorbar(im)
    cbar.set_label(colorbar_label)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def load_attribution_vector_from_csv(
    csv_path: Path,
    expected_size: int,
    value_col: str,
    id_cols: tuple[str, ...] = ("channel", "dim"),
) -> Optional[np.ndarray]:
    """Load an attribution vector from a per-sample CSV and restore natural channel/dim order."""
    rows = _csv_rows(csv_path)
    if not rows:
        return None

    id_col = None
    for candidate in id_cols:
        if candidate in rows[0]:
            id_col = candidate
            break
    if id_col is None or value_col not in rows[0]:
        return None

    values = np.full(expected_size, np.nan, dtype=np.float64)
    for r in rows:
        idx_val = r.get(id_col, "")
        try:
            idx = int(float(idx_val))
        except Exception:
            continue
        if 0 <= idx < expected_size:
            values[idx] = _to_float(r.get(value_col), default=np.nan)

    if not np.isfinite(values).any():
        return None
    values = np.nan_to_num(values, nan=0.0)
    return values


def compute_unified_heatmap_height(n_rows: int, cfg: RuntimeConfig) -> float:
    """
    Compute the shared vertical size for all sample-level heatmaps in one run.

    统一规则：同一次实验内，只要是“样本 × ...”热力图，就使用相同的 fig_height。
    这样 mode_error / projected IG / differential IG / occlusion 可以并排逐行比较。
    """
    n = max(int(n_rows), 1)
    height = cfg.heatmap_extra_height_inch + cfg.heatmap_row_height_inch * n
    height = max(cfg.heatmap_min_height_inch, height)
    height = min(cfg.heatmap_max_height_inch, height)
    return float(height)


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _parse_test_sample_idx_from_dirname(name: str) -> Optional[int]:
    # sample dir name generated by this script: test_000123_orig_4567_xxx
    parts = name.split("_")
    if len(parts) >= 2 and parts[0] == "test":
        try:
            return int(parts[1])
        except Exception:
            return None
    return None


def collect_canonical_heatmap_samples(
    samples_root: Path,
    cfg: RuntimeConfig,
) -> list[dict[str, Any]]:
    """
    Build the canonical sample axis used by all sample-level heatmaps.

    All sample × ... heatmaps must use this same sample list and order. If a
    particular CSV is missing for one sample, that row is kept and filled with
    NaN rather than being skipped, so the y-axis remains identical across plots.
    """
    samples_root = Path(samples_root)
    if not samples_root.exists():
        return []
    sample_dirs = sorted([p for p in samples_root.iterdir() if p.is_dir()])[: cfg.heatmap_max_samples]
    canonical: list[dict[str, Any]] = []
    for sample_dir in sample_dirs:
        summary = _read_json_if_exists(sample_dir / "sample_summary.json")
        sample_idx = summary.get("test_sample_idx")
        if sample_idx is None:
            sample_idx = _parse_test_sample_idx_from_dirname(sample_dir.name)
        original_idx = summary.get("original_idx")
        material_id = str(summary.get("material_id", ""))
        try:
            sample_idx_int = int(sample_idx) if sample_idx is not None else None
        except Exception:
            sample_idx_int = None
        label = sample_dir.name[:36]
        if sample_idx_int is not None:
            label = f"test_{sample_idx_int:06d}"
            if material_id and material_id.lower() != "nan":
                label += f"_{material_id[:12]}"
        canonical.append(
            {
                "sample_dir": sample_dir,
                "test_sample_idx": sample_idx_int,
                "original_idx": original_idx,
                "material_id": material_id,
                "label": label,
            }
        )
    return canonical


def _build_summary_lookup(csv_path: Path) -> dict[int, dict[str, str]]:
    rows = _csv_rows(csv_path)
    lookup: dict[int, dict[str, str]] = {}
    for r in rows:
        key = r.get("test_sample_idx", r.get("sample_idx", ""))
        try:
            idx = int(float(key))
        except Exception:
            continue
        lookup[idx] = r
    return lookup


def save_single_projected_128_block_heatmap(
    csv_path: Path,
    output_path: Path,
    title: str,
    cfg: RuntimeConfig,
    value_col: str = "signed_weight",
) -> None:
    """
    Save one 8×16 heatmap for a single sample's 128 projected text channels.

    注意：8×16 只是可视化排列，不表示通道之间存在真实二维拓扑关系。
    本版本固定所有单样本 block heatmap 的宽高，便于样本内和样本间比较。
    """
    vec = load_attribution_vector_from_csv(csv_path, expected_size=128, value_col=value_col)
    if vec is None:
        return
    mat = vec.reshape(8, 16)
    vmax = _safe_percentile(np.abs(mat), 98, default=1.0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(cfg.heatmap_single_block_width_inch, cfg.heatmap_single_block_height_inch))
    im = plt.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    plt.title(title)
    plt.xlabel("visualized channel block column")
    plt.ylabel("visualized channel block row")

    # 标注 channel 编号，方便回到 CSV 定位。
    for r in range(8):
        for c in range(16):
            ch = r * 16 + c
            plt.text(c, r, str(ch), ha="center", va="center", fontsize=5)

    cbar = plt.colorbar(im)
    cbar.set_label(value_col)
    plt.tight_layout()
    plt.savefig(output_path, dpi=cfg.heatmap_dpi)
    plt.close()


def save_mode_error_heatmap(
    run_dir: Path,
    cfg: RuntimeConfig,
    logger: logging.Logger,
    *,
    canonical_samples: Optional[list[dict[str, Any]]] = None,
    common_fig_height: Optional[float] = None,
) -> None:
    """Save sample × {normal,bypass,shuffle} absolute-error heatmap."""
    csv_path = run_dir / "test_shuffle_effect_summary.csv"
    rows = _csv_rows(csv_path)
    if not rows:
        logger.warning("skip mode error heatmap; missing or empty: %s", csv_path)
        return

    if canonical_samples:
        lookup = _build_summary_lookup(csv_path)
        matrix_rows = []
        row_labels = []
        for item in canonical_samples:
            idx = item.get("test_sample_idx")
            r = lookup.get(idx) if idx is not None else None
            if r is None:
                matrix_rows.append([np.nan, np.nan, np.nan])
            else:
                matrix_rows.append(
                    [
                        _to_float(r.get("error_normal"), default=np.nan),
                        _to_float(r.get("error_bypass"), default=np.nan),
                        _to_float(r.get("error_shuffle_mean"), default=np.nan),
                    ]
                )
            row_labels.append(str(item.get("label", "sample")))
        matrix = np.asarray(matrix_rows, dtype=np.float64)
    else:
        # Fallback when no sample analysis was run: show top-N rows by text_help_shuffle.
        sort_key = "text_help_shuffle" if "text_help_shuffle" in rows[0] else "abs_delta_shuffle"
        rows = sorted(rows, key=lambda r: _to_float(r.get(sort_key), default=0.0), reverse=True)
        rows = rows[: cfg.heatmap_top_n_mode_error]
        matrix = np.array(
            [
                [
                    _to_float(r.get("error_normal"), default=np.nan),
                    _to_float(r.get("error_bypass"), default=np.nan),
                    _to_float(r.get("error_shuffle_mean"), default=np.nan),
                ]
                for r in rows
            ],
            dtype=np.float64,
        )
        row_labels = []
        for r in rows:
            sample = r.get("test_sample_idx", r.get("sample_idx", "?"))
            mid = str(r.get("material_id", ""))
            label = f"test_{sample}"
            if mid and mid != "nan":
                label += f"_{mid[:10]}"
            row_labels.append(label)
        if cfg.heatmap_unified_y_axis and common_fig_height is None:
            common_fig_height = compute_unified_heatmap_height(len(row_labels), cfg)

    save_matrix_heatmap(
        matrix=matrix,
        row_labels=row_labels,
        col_labels=["normal", "bypass", "shuffle"],
        title="Prediction error under normal / bypass / shuffle modes",
        output_path=run_dir / "heatmaps" / "mode_error_heatmap.png",
        center_zero=False,
        colorbar_label="absolute error",
        xlabel="mode",
        ylabel="sample",
        dpi=cfg.heatmap_dpi,
        max_xticks=3,
        max_yticks=max(len(row_labels), 1),
        fig_width=8.0,
        fig_height=common_fig_height,
    )


def save_multi_sample_projected_heatmap(
    samples_root: Path,
    output_path: Path,
    cfg: RuntimeConfig,
    *,
    filename: str,
    title: str,
    value_col: str = "signed_weight",
    canonical_samples: Optional[list[dict[str, Any]]] = None,
    common_fig_height: Optional[float] = None,
) -> None:
    """Save sample × 128 projected channel attribution heatmap."""
    if not samples_root.exists():
        return
    if canonical_samples is None:
        canonical_samples = collect_canonical_heatmap_samples(samples_root, cfg)
    if not canonical_samples:
        return

    rows = []
    row_labels = []
    for item in canonical_samples:
        sample_dir = Path(item["sample_dir"])
        vec = load_attribution_vector_from_csv(
            sample_dir / filename,
            expected_size=128,
            value_col=value_col,
            id_cols=("channel", "dim"),
        )
        if vec is None:
            vec = np.full(128, np.nan, dtype=np.float64)
        rows.append(np.abs(vec) if cfg.heatmap_use_abs_attribution else vec)
        row_labels.append(str(item.get("label", sample_dir.name[:36])))

    matrix = np.stack(rows, axis=0)
    center_zero = not cfg.heatmap_use_abs_attribution
    save_matrix_heatmap(
        matrix=matrix,
        row_labels=row_labels,
        col_labels=[f"c{i}" for i in range(128)],
        title=title,
        output_path=output_path,
        center_zero=center_zero,
        colorbar_label=("abs attribution" if cfg.heatmap_use_abs_attribution else value_col),
        xlabel="projected text channel",
        ylabel="sample",
        dpi=cfg.heatmap_dpi,
        max_xticks=32,
        max_yticks=max(len(row_labels), 1),
        fig_width=20.0,
        fig_height=common_fig_height,
    )


def save_topk_occlusion_heatmap(
    samples_root: Path,
    output_path: Path,
    cfg: RuntimeConfig,
    *,
    filename: str = "projected_topk_occlusion.csv",
    mask_type: str = "top_importance",
    title: str,
    canonical_samples: Optional[list[dict[str, Any]]] = None,
    common_fig_height: Optional[float] = None,
) -> None:
    """Save sample × K heatmap for Top-K or random projected-channel occlusion."""
    if not samples_root.exists():
        return
    if canonical_samples is None:
        canonical_samples = collect_canonical_heatmap_samples(samples_root, cfg)
    if not canonical_samples:
        return

    k_ref = list(cfg.projected_topk_channels)
    rows = []
    row_labels = []
    for item in canonical_samples:
        sample_dir = Path(item["sample_dir"])
        csv_rows = _csv_rows(sample_dir / filename)
        selected = [r for r in csv_rows if r.get("mask_type") == mask_type]
        by_k = {}
        for r in selected:
            try:
                k = int(_to_float(r.get("k"), default=np.nan))
            except Exception:
                continue
            by_k[k] = _to_float(r.get("abs_delta_from_original"), default=np.nan)
        values = np.array([by_k.get(k, np.nan) for k in k_ref], dtype=np.float64)
        rows.append(values)
        row_labels.append(str(item.get("label", sample_dir.name[:36])))

    matrix = np.stack(rows, axis=0)
    save_matrix_heatmap(
        matrix=matrix,
        row_labels=row_labels,
        col_labels=[f"K={k}" for k in k_ref],
        title=title,
        output_path=output_path,
        center_zero=False,
        colorbar_label="absolute prediction change",
        xlabel="masked projected channels",
        ylabel="sample",
        dpi=cfg.heatmap_dpi,
        max_xticks=len(k_ref),
        max_yticks=max(len(row_labels), 1),
        fig_width=8.5,
        fig_height=common_fig_height,
    )


def generate_additional_heatmaps(run_dir: Path, cfg: RuntimeConfig, logger: logging.Logger) -> None:
    """Generate additional heatmap visualizations after numeric experiment files are saved."""
    if not cfg.save_plots or not cfg.run_additional_heatmaps:
        return

    heatmap_dir = run_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    samples_root = run_dir / "samples"

    canonical_samples = collect_canonical_heatmap_samples(samples_root, cfg)
    if canonical_samples:
        write_csv(
            heatmap_dir / "heatmap_canonical_sample_axis.csv",
            [
                {
                    "row_index": i,
                    "test_sample_idx": item.get("test_sample_idx"),
                    "original_idx": item.get("original_idx"),
                    "material_id": item.get("material_id"),
                    "row_label": item.get("label"),
                    "sample_dir": str(item.get("sample_dir")),
                }
                for i, item in enumerate(canonical_samples)
            ],
        )
    common_fig_height: Optional[float] = None
    if cfg.heatmap_unified_y_axis:
        n_rows_for_height = len(canonical_samples) if canonical_samples else cfg.heatmap_top_n_mode_error
        common_fig_height = compute_unified_heatmap_height(n_rows_for_height, cfg)
        logger.info(
            "unified heatmap y-axis enabled | rows=%d | fig_height=%.2f inch | row_height=%.3f inch",
            n_rows_for_height,
            common_fig_height,
            cfg.heatmap_row_height_inch,
        )

    # 1. The mode-error heatmap is aligned to canonical sample dirs when detailed sample analysis exists.
    save_mode_error_heatmap(
        run_dir,
        cfg,
        logger,
        canonical_samples=canonical_samples,
        common_fig_height=common_fig_height,
    )

    # 2. All sample × channel / sample × K heatmaps use the exact same y-axis sample list.
    save_multi_sample_projected_heatmap(
        samples_root=samples_root,
        output_path=heatmap_dir / "sample_projected_128_ig_matrix_heatmap.png",
        cfg=cfg,
        filename="projected_text_channel_importance_128_integrated_gradients.csv",
        title="Sample × projected 128D text-channel IG",
        value_col="signed_weight",
        canonical_samples=canonical_samples,
        common_fig_height=common_fig_height,
    )
    save_multi_sample_projected_heatmap(
        samples_root=samples_root,
        output_path=heatmap_dir / "sample_differential_projected_128_ig_matrix_heatmap.png",
        cfg=cfg,
        filename="differential_projected_128_integrated_gradients.csv",
        title="Sample × differential projected 128D IG: shuffle → normal",
        value_col="signed_weight",
        canonical_samples=canonical_samples,
        common_fig_height=common_fig_height,
    )

    save_topk_occlusion_heatmap(
        samples_root=samples_root,
        output_path=heatmap_dir / "topk_projected_occlusion_heatmap.png",
        cfg=cfg,
        mask_type="top_importance",
        title="Top-K projected-channel occlusion effect",
        canonical_samples=canonical_samples,
        common_fig_height=common_fig_height,
    )
    save_topk_occlusion_heatmap(
        samples_root=samples_root,
        output_path=heatmap_dir / "random_projected_occlusion_heatmap.png",
        cfg=cfg,
        mask_type="random_mean",
        title="Random-K projected-channel occlusion effect",
        canonical_samples=canonical_samples,
        common_fig_height=common_fig_height,
    )

    if samples_root.exists() and cfg.heatmap_single_sample_limit > 0:
        sample_dirs = [Path(item["sample_dir"]) for item in canonical_samples[: cfg.heatmap_single_sample_limit]]
        for sample_dir in sample_dirs:
            normal_csv = sample_dir / "projected_text_channel_importance_128_integrated_gradients.csv"
            if normal_csv.exists():
                save_single_projected_128_block_heatmap(
                    csv_path=normal_csv,
                    output_path=sample_dir / "projected_128_ig_block_heatmap.png",
                    title=f"Projected 128D IG - {sample_dir.name[:32]}",
                    cfg=cfg,
                    value_col="signed_weight",
                )
            diff_csv = sample_dir / "differential_projected_128_integrated_gradients.csv"
            if diff_csv.exists():
                save_single_projected_128_block_heatmap(
                    csv_path=diff_csv,
                    output_path=sample_dir / "differential_projected_128_ig_block_heatmap.png",
                    title=f"Differential projected 128D IG - {sample_dir.name[:32]}",
                    cfg=cfg,
                    value_col="signed_weight",
                )

    logger.info("additional heatmaps saved to: %s", heatmap_dir)


# =============================================================================
# Baseline 与投影 override
# =============================================================================
def compute_train_text_mean(train_dataset: Any, cache_path: Path, logger: logging.Logger) -> np.ndarray:
    logger.info("computing train-set mean text embedding from cache...")
    cache = np.load(cache_path, mmap_mode="r")
    indices = np.array(train_dataset.original_indices, dtype=np.int64)
    # 直接索引通常可行；若内存不足，可以改成分块。
    mean = np.asarray(cache[indices], dtype=np.float64).mean(axis=0).astype(np.float32)
    logger.info("train mean text embedding shape: %s", mean.shape)
    return mean


def make_baseline_text_emb(
    z: torch.Tensor,
    baseline_name: str,
    train_mean: Optional[torch.Tensor],
) -> torch.Tensor:
    if baseline_name == "zero":
        return torch.zeros_like(z)
    if baseline_name == "train_mean":
        if train_mean is None:
            raise ValueError("train_mean baseline requested but train_mean is None")
        if train_mean.ndim == 1:
            train_mean = train_mean.unsqueeze(0)
        return train_mean.to(device=z.device, dtype=z.dtype).expand_as(z).clone()
    raise ValueError(f"unsupported baseline_name={baseline_name}")


@contextmanager
def override_project_cached_text(model: torch.nn.Module, projected_text: torch.Tensor):
    """临时覆盖 model.project_cached_text，让 forward 使用给定的 128维投影文本特征。"""
    original_method = model.project_cached_text

    def patched(self, pooled_text_emb: torch.Tensor):
        if projected_text.shape[0] == pooled_text_emb.shape[0]:
            return projected_text
        if projected_text.shape[0] == 1:
            return projected_text.expand(pooled_text_emb.shape[0], -1)
        raise ValueError(
            "projected_text batch size mismatch: "
            f"projected={tuple(projected_text.shape)}, pooled={tuple(pooled_text_emb.shape)}"
        )

    model.project_cached_text = MethodType(patched, model)
    try:
        yield
    finally:
        model.project_cached_text = original_method


@torch.no_grad()
def project_text(model: torch.nn.Module, text_emb: torch.Tensor) -> torch.Tensor:
    model.eval()
    return model.project_cached_text(text_emb)


# =============================================================================
# 归因方法
# =============================================================================
def raw_gradient_x_input(
    model: torch.nn.Module,
    inputs: dict[str, Any],
    cfg: RuntimeConfig,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    z = inputs["text_emb"].detach().clone().to(device).requires_grad_(True)
    run_inputs = clone_inputs_with_text_emb(inputs, z)
    model.zero_grad(set_to_none=True)
    with autocast_context(device, cfg.use_amp):
        y = scalar_selected_output(model(run_inputs), cfg.target_index)
    y.backward()
    grad = z.grad.detach()
    score = (z.detach() * grad).squeeze(0).detach().float().cpu().numpy()
    model.zero_grad(set_to_none=True)
    return score


def raw_integrated_gradients(
    model: torch.nn.Module,
    inputs: dict[str, Any],
    baseline: torch.Tensor,
    cfg: RuntimeConfig,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    z = inputs["text_emb"].detach().clone().to(device)
    z0 = baseline.detach().clone().to(device=device, dtype=z.dtype)
    if z0.ndim == 1:
        z0 = z0.unsqueeze(0)
    z0 = z0.expand_as(z)
    delta = z - z0
    grad_sum = torch.zeros_like(z)

    # midpoint Riemann approximation: alpha = (k + 0.5) / steps
    for k in range(cfg.ig_steps):
        alpha = (k + 0.5) / float(cfg.ig_steps)
        z_alpha = (z0 + alpha * delta).detach().clone().requires_grad_(True)
        run_inputs = clone_inputs_with_text_emb(inputs, z_alpha)
        model.zero_grad(set_to_none=True)
        with autocast_context(device, cfg.use_amp):
            y = scalar_selected_output(model(run_inputs), cfg.target_index)
        y.backward()
        grad_sum += z_alpha.grad.detach()

    avg_grad = grad_sum / float(cfg.ig_steps)
    ig = (delta * avg_grad).squeeze(0).detach().float().cpu().numpy()
    model.zero_grad(set_to_none=True)
    return ig


def projected_gradient_x_input(
    model: torch.nn.Module,
    inputs: dict[str, Any],
    cfg: RuntimeConfig,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    capture: dict[str, torch.Tensor] = {}

    def hook(_module, _args, output):
        output.retain_grad()
        capture["u"] = output

    handle = model.text_proj.register_forward_hook(hook)
    model.zero_grad(set_to_none=True)
    try:
        run_inputs = clone_inputs_with_mode(inputs, "normal")
        with autocast_context(device, cfg.use_amp):
            y = scalar_selected_output(model(run_inputs), cfg.target_index)
        y.backward()
        u = capture["u"]
        grad = u.grad.detach()
        score = (u.detach() * grad).squeeze(0).detach().float().cpu().numpy()
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
    return score


def projected_integrated_gradients(
    model: torch.nn.Module,
    inputs: dict[str, Any],
    raw_baseline: torch.Tensor,
    cfg: RuntimeConfig,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    z = inputs["text_emb"].detach().clone().to(device)
    z0 = raw_baseline.detach().clone().to(device=device, dtype=z.dtype)
    if z0.ndim == 1:
        z0 = z0.unsqueeze(0)
    z0 = z0.expand_as(z)

    with torch.no_grad():
        u = project_text(model, z).detach()
        u0 = project_text(model, z0).detach()
    delta_u = u - u0
    grad_sum = torch.zeros_like(u)

    for k in range(cfg.ig_steps):
        alpha = (k + 0.5) / float(cfg.ig_steps)
        u_alpha = (u0 + alpha * delta_u).detach().clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        with override_project_cached_text(model, u_alpha):
            run_inputs = clone_inputs_with_mode(inputs, "normal")
            with autocast_context(device, cfg.use_amp):
                y = scalar_selected_output(model(run_inputs), cfg.target_index)
        y.backward()
        grad_sum += u_alpha.grad.detach()

    avg_grad = grad_sum / float(cfg.ig_steps)
    ig = (delta_u * avg_grad).squeeze(0).detach().float().cpu().numpy()
    model.zero_grad(set_to_none=True)
    return ig


def raw_differential_integrated_gradients(
    model: torch.nn.Module,
    inputs: dict[str, Any],
    z_from: torch.Tensor,
    z_to: torch.Tensor,
    cfg: RuntimeConfig,
    device: torch.device,
) -> np.ndarray:
    """Raw 1536D differential IG: baseline=shuffle text, target=normal text."""
    model.eval()
    z0 = z_from.detach().clone().to(device=device, dtype=torch.float32)
    z1 = z_to.detach().clone().to(device=device, dtype=torch.float32)
    if z0.ndim == 1:
        z0 = z0.unsqueeze(0)
    if z1.ndim == 1:
        z1 = z1.unsqueeze(0)
    z0 = z0.expand_as(z1)
    delta = z1 - z0
    grad_sum = torch.zeros_like(z1)

    for k in range(cfg.differential_ig_steps):
        alpha = (k + 0.5) / float(cfg.differential_ig_steps)
        z_alpha = (z0 + alpha * delta).detach().clone().requires_grad_(True)
        run_inputs = clone_inputs_with_text_emb(inputs, z_alpha)
        model.zero_grad(set_to_none=True)
        with autocast_context(device, cfg.use_amp):
            y = scalar_selected_output(model(run_inputs), cfg.target_index)
        y.backward()
        grad_sum += z_alpha.grad.detach()

    avg_grad = grad_sum / float(cfg.differential_ig_steps)
    ig = (delta * avg_grad).squeeze(0).detach().float().cpu().numpy()
    model.zero_grad(set_to_none=True)
    return ig


def projected_differential_integrated_gradients(
    model: torch.nn.Module,
    inputs: dict[str, Any],
    z_from: torch.Tensor,
    z_to: torch.Tensor,
    cfg: RuntimeConfig,
    device: torch.device,
) -> np.ndarray:
    """Projected 128D differential IG: baseline=project(shuffle), target=project(normal)."""
    model.eval()
    z0 = z_from.detach().clone().to(device=device, dtype=torch.float32)
    z1 = z_to.detach().clone().to(device=device, dtype=torch.float32)
    if z0.ndim == 1:
        z0 = z0.unsqueeze(0)
    if z1.ndim == 1:
        z1 = z1.unsqueeze(0)
    z0 = z0.expand_as(z1)
    with torch.no_grad():
        u0 = project_text(model, z0).detach()
        u1 = project_text(model, z1).detach()
    delta_u = u1 - u0
    grad_sum = torch.zeros_like(u1)

    for k in range(cfg.differential_ig_steps):
        alpha = (k + 0.5) / float(cfg.differential_ig_steps)
        u_alpha = (u0 + alpha * delta_u).detach().clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        with override_project_cached_text(model, u_alpha):
            run_inputs = clone_inputs_with_mode(inputs, "normal")
            with autocast_context(device, cfg.use_amp):
                y = scalar_selected_output(model(run_inputs), cfg.target_index)
        y.backward()
        grad_sum += u_alpha.grad.detach()

    avg_grad = grad_sum / float(cfg.differential_ig_steps)
    ig = (delta_u * avg_grad).squeeze(0).detach().float().cpu().numpy()
    model.zero_grad(set_to_none=True)
    return ig


def vector_scaling_curve(
    model: torch.nn.Module,
    inputs: dict[str, Any],
    baseline: torch.Tensor,
    cfg: RuntimeConfig,
    device: torch.device,
) -> list[dict[str, Any]]:
    z = inputs["text_emb"].detach().clone().to(device)
    z0 = baseline.detach().clone().to(device=device, dtype=z.dtype)
    if z0.ndim == 1:
        z0 = z0.unsqueeze(0)
    z0 = z0.expand_as(z)
    rows: list[dict[str, Any]] = []
    for alpha in cfg.scaling_alphas:
        z_alpha = z0 + float(alpha) * (z - z0)
        pred = predict_single_with_text_emb(model, inputs, z_alpha, cfg, device)
        rows.append({"alpha": float(alpha), "prediction": pred})
    return rows


def topk_raw_occlusion(
    model: torch.nn.Module,
    inputs: dict[str, Any],
    baseline: torch.Tensor,
    importance: np.ndarray,
    cfg: RuntimeConfig,
    device: torch.device,
) -> list[dict[str, Any]]:
    z = inputs["text_emb"].detach().clone().to(device)
    z0 = baseline.detach().clone().to(device=device, dtype=z.dtype)
    if z0.ndim == 1:
        z0 = z0.unsqueeze(0)
    z0 = z0.expand_as(z)

    baseline_pred = predict_single_with_text_emb(model, inputs, z, cfg, device)
    order = np.argsort(-np.abs(importance))
    rows: list[dict[str, Any]] = []
    dim_count = int(z.shape[-1])

    rng = np.random.default_rng(cfg.seed)
    for k in cfg.raw_topk_dims:
        k_eff = min(int(k), dim_count)
        top_idx = order[:k_eff]
        z_mask = z.clone()
        z_mask[:, top_idx] = z0[:, top_idx]
        pred = predict_single_with_text_emb(model, inputs, z_mask, cfg, device)
        rows.append(
            {
                "mask_type": "top_importance",
                "k": k_eff,
                "prediction": pred,
                "delta_from_original": baseline_pred - pred,
                "abs_delta_from_original": abs(baseline_pred - pred),
            }
        )

        random_abs = []
        random_delta = []
        for repeat in range(cfg.random_occlusion_repeats):
            rand_idx = rng.choice(dim_count, size=k_eff, replace=False)
            z_rand = z.clone()
            z_rand[:, rand_idx] = z0[:, rand_idx]
            pred_rand = predict_single_with_text_emb(model, inputs, z_rand, cfg, device)
            random_delta.append(baseline_pred - pred_rand)
            random_abs.append(abs(baseline_pred - pred_rand))
        rows.append(
            {
                "mask_type": "random_mean",
                "k": k_eff,
                "prediction": "",
                "delta_from_original": float(np.mean(random_delta)),
                "abs_delta_from_original": float(np.mean(random_abs)),
            }
        )
    return rows


def predict_single_with_projected(
    model: torch.nn.Module,
    inputs: dict[str, Any],
    projected_text: torch.Tensor,
    cfg: RuntimeConfig,
    device: torch.device,
) -> float:
    model.eval()
    with torch.no_grad():
        with override_project_cached_text(model, projected_text):
            run_inputs = clone_inputs_with_mode(inputs, "normal")
            with autocast_context(device, cfg.use_amp):
                outputs = model(run_inputs)
    return as_float(scalar_selected_output(outputs, cfg.target_index))


def topk_projected_occlusion(
    model: torch.nn.Module,
    inputs: dict[str, Any],
    raw_baseline: torch.Tensor,
    importance: np.ndarray,
    cfg: RuntimeConfig,
    device: torch.device,
) -> list[dict[str, Any]]:
    z = inputs["text_emb"].detach().clone().to(device)
    z0 = raw_baseline.detach().clone().to(device=device, dtype=z.dtype)
    if z0.ndim == 1:
        z0 = z0.unsqueeze(0)
    z0 = z0.expand_as(z)
    with torch.no_grad():
        u = project_text(model, z).detach()
        u0 = project_text(model, z0).detach()

    baseline_pred = predict_single_with_projected(model, inputs, u, cfg, device)
    order = np.argsort(-np.abs(importance))
    channel_count = int(u.shape[-1])
    rng = np.random.default_rng(cfg.seed)
    rows: list[dict[str, Any]] = []

    for k in cfg.projected_topk_channels:
        k_eff = min(int(k), channel_count)
        top_idx = order[:k_eff]
        u_mask = u.clone()
        u_mask[:, top_idx] = u0[:, top_idx]
        pred = predict_single_with_projected(model, inputs, u_mask, cfg, device)
        rows.append(
            {
                "mask_type": "top_importance",
                "k": k_eff,
                "prediction": pred,
                "delta_from_original": baseline_pred - pred,
                "abs_delta_from_original": abs(baseline_pred - pred),
            }
        )

        random_abs = []
        random_delta = []
        for repeat in range(cfg.random_occlusion_repeats):
            rand_idx = rng.choice(channel_count, size=k_eff, replace=False)
            u_rand = u.clone()
            u_rand[:, rand_idx] = u0[:, rand_idx]
            pred_rand = predict_single_with_projected(model, inputs, u_rand, cfg, device)
            random_delta.append(baseline_pred - pred_rand)
            random_abs.append(abs(baseline_pred - pred_rand))
        rows.append(
            {
                "mask_type": "random_mean",
                "k": k_eff,
                "prediction": "",
                "delta_from_original": float(np.mean(random_delta)),
                "abs_delta_from_original": float(np.mean(random_abs)),
            }
        )
    return rows


# =============================================================================
# 保存与可视化
# =============================================================================
def normalized_abs_weights(values: np.ndarray) -> np.ndarray:
    abs_values = np.abs(values.astype(np.float64))
    denom = float(abs_values.sum())
    if denom <= 0:
        return np.zeros_like(abs_values)
    return abs_values / denom


def make_importance_rows(prefix: str, values: np.ndarray) -> list[dict[str, Any]]:
    values = np.asarray(values, dtype=np.float64)
    norm = normalized_abs_weights(values)
    rows = []
    for i, v in enumerate(values):
        rows.append(
            {
                prefix: i,
                "signed_weight": float(v),
                "abs_weight": float(abs(v)),
                "normalized_weight": float(norm[i]),
            }
        )
    rows.sort(key=lambda r: r["abs_weight"], reverse=True)
    return rows


def plot_importance_bar(
    path: Path,
    values: np.ndarray,
    title: str,
    xlabel: str,
    top_k: int = 40,
) -> None:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(-np.abs(values))[: min(top_k, len(values))]
    y = values[order]
    labels = [str(i) for i in order]
    plt.figure(figsize=(12, 5))
    plt.bar(np.arange(len(y)), y)
    plt.xticks(np.arange(len(y)), labels, rotation=90)
    plt.xlabel(xlabel)
    plt.ylabel("signed attribution")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_importance_heatmap(path: Path, values: np.ndarray, title: str) -> None:
    values = np.asarray(values, dtype=np.float64).reshape(1, -1)
    plt.figure(figsize=(14, 2.4))
    plt.imshow(values, aspect="auto")
    plt.yticks([0], ["weight"])
    plt.xlabel("dimension / channel")
    plt.colorbar(label="signed attribution")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_scaling_curve(path: Path, rows: list[dict[str, Any]]) -> None:
    xs = [float(r["alpha"]) for r in rows]
    ys = [float(r["prediction"]) for r in rows]
    plt.figure(figsize=(7, 5))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("text vector scaling alpha")
    plt.ylabel("prediction")
    plt.title("Prediction vs text-vector scaling")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_occlusion_curve(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    top = [r for r in rows if r["mask_type"] == "top_importance"]
    rnd = [r for r in rows if r["mask_type"] == "random_mean"]
    plt.figure(figsize=(7, 5))
    if top:
        plt.plot(
            [int(r["k"]) for r in top],
            [float(r["abs_delta_from_original"]) for r in top],
            marker="o",
            label="top importance",
        )
    if rnd:
        plt.plot(
            [int(r["k"]) for r in rnd],
            [float(r["abs_delta_from_original"]) for r in rnd],
            marker="o",
            label="random mean",
        )
    plt.xlabel("number of masked dimensions / channels")
    plt.ylabel("absolute prediction change")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_scan_distributions(rows: list[dict[str, Any]], run_dir: Path) -> None:
    abs_delta = np.array([r["abs_delta_text"] for r in rows], dtype=np.float64)
    text_help = np.array([r["text_help"] for r in rows], dtype=np.float64)
    err_normal = np.array([r["error_normal"] for r in rows], dtype=np.float64)
    err_bypass = np.array([r["error_bypass"] for r in rows], dtype=np.float64)

    plt.figure(figsize=(7, 5))
    plt.hist(abs_delta, bins=60)
    plt.xlabel("|pred_normal - pred_bypass|")
    plt.ylabel("count")
    plt.title("Distribution of absolute text-vector effect")
    plt.tight_layout()
    plt.savefig(run_dir / "abs_delta_text_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.hist(text_help, bins=60)
    plt.xlabel("text_help = error_bypass - error_normal")
    plt.ylabel("count")
    plt.title("Distribution of text help")
    plt.tight_layout()
    plt.savefig(run_dir / "text_help_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(err_bypass, err_normal, s=8, alpha=0.5)
    max_v = float(max(err_bypass.max(), err_normal.max())) if len(rows) else 1.0
    plt.plot([0, max_v], [0, max_v], linestyle="--")
    plt.xlabel("bypass absolute error")
    plt.ylabel("normal absolute error")
    plt.title("Normal vs bypass error")
    plt.tight_layout()
    plt.savefig(run_dir / "normal_vs_bypass_error_scatter.png", dpi=200)
    plt.close()


def plot_shuffle_distributions(rows: list[dict[str, Any]], run_dir: Path) -> None:
    if not rows or "pred_shuffle_mean" not in rows[0]:
        return
    delta_shuffle = np.array([r["delta_shuffle"] for r in rows], dtype=np.float64)
    abs_delta_shuffle = np.abs(delta_shuffle)
    text_help_shuffle = np.array([r["text_help_shuffle"] for r in rows], dtype=np.float64)
    err_normal = np.array([r["error_normal"] for r in rows], dtype=np.float64)
    err_shuffle = np.array([r["error_shuffle_mean"] for r in rows], dtype=np.float64)

    plt.figure(figsize=(7, 5))
    plt.hist(abs_delta_shuffle, bins=60)
    plt.xlabel("|pred_normal - pred_shuffle_mean|")
    plt.ylabel("count")
    plt.title("Distribution of matched-vs-shuffled text effect")
    plt.tight_layout()
    plt.savefig(run_dir / "abs_delta_shuffle_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.hist(text_help_shuffle, bins=60)
    plt.xlabel("text_help_shuffle = error_shuffle - error_normal")
    plt.ylabel("count")
    plt.title("Distribution of matched-text advantage over shuffle")
    plt.tight_layout()
    plt.savefig(run_dir / "text_help_shuffle_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(err_shuffle, err_normal, s=8, alpha=0.5)
    max_v = float(max(err_shuffle.max(), err_normal.max())) if len(rows) else 1.0
    plt.plot([0, max_v], [0, max_v], linestyle="--")
    plt.xlabel("shuffle absolute error")
    plt.ylabel("normal absolute error")
    plt.title("Normal vs shuffle error")
    plt.tight_layout()
    plt.savefig(run_dir / "normal_vs_shuffle_error_scatter.png", dpi=200)
    plt.close()


def save_sample_outputs(
    sample_dir: Path,
    summary: dict[str, Any],
    raw_gxi: Optional[np.ndarray],
    raw_ig: Optional[np.ndarray],
    proj_gxi: Optional[np.ndarray],
    proj_ig: Optional[np.ndarray],
    scaling_rows: list[dict[str, Any]],
    raw_occ_rows: list[dict[str, Any]],
    proj_occ_rows: list[dict[str, Any]],
    cfg: RuntimeConfig,
) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    save_json(sample_dir / "sample_summary.json", summary)

    # 保存原始数组，便于后续重新画图或统计。
    np.savez_compressed(
        sample_dir / "attribution_arrays.npz",
        raw_grad_x_input=np.array([]) if raw_gxi is None else raw_gxi,
        raw_integrated_gradients=np.array([]) if raw_ig is None else raw_ig,
        projected_grad_x_input=np.array([]) if proj_gxi is None else proj_gxi,
        projected_integrated_gradients=np.array([]) if proj_ig is None else proj_ig,
    )

    if raw_gxi is not None:
        write_csv(
            sample_dir / "raw_text_vector_importance_1536_grad_x_input.csv",
            make_importance_rows("dim", raw_gxi),
        )
    if raw_ig is not None:
        write_csv(
            sample_dir / "raw_text_vector_importance_1536_integrated_gradients.csv",
            make_importance_rows("dim", raw_ig),
        )
    if proj_gxi is not None:
        write_csv(
            sample_dir / "projected_text_channel_importance_128_grad_x_input.csv",
            make_importance_rows("channel", proj_gxi),
        )
    if proj_ig is not None:
        write_csv(
            sample_dir / "projected_text_channel_importance_128_integrated_gradients.csv",
            make_importance_rows("channel", proj_ig),
        )

    write_csv(sample_dir / "vector_scaling_curve.csv", scaling_rows)
    if raw_occ_rows:
        write_csv(sample_dir / "raw_topk_occlusion.csv", raw_occ_rows)
    if proj_occ_rows:
        write_csv(sample_dir / "projected_topk_occlusion.csv", proj_occ_rows)

    if cfg.save_plots:
        if raw_gxi is not None:
            plot_importance_bar(
                sample_dir / "raw_1536_grad_x_input_top_dims.png",
                raw_gxi,
                "Raw 1536D text vector attribution: Gradient × Input",
                "raw text dimension",
            )
        if raw_ig is not None:
            plot_importance_bar(
                sample_dir / "raw_1536_integrated_gradients_top_dims.png",
                raw_ig,
                "Raw 1536D text vector attribution: Integrated Gradients",
                "raw text dimension",
            )
            plot_importance_heatmap(
                sample_dir / "raw_1536_integrated_gradients_heatmap.png",
                raw_ig,
                "Raw 1536D Integrated Gradients heatmap",
            )
        if proj_gxi is not None:
            plot_importance_bar(
                sample_dir / "projected_128_grad_x_input_top_channels.png",
                proj_gxi,
                "Projected 128D text channel attribution: Gradient × Input",
                "projected channel",
            )
        if proj_ig is not None:
            plot_importance_bar(
                sample_dir / "projected_128_integrated_gradients_top_channels.png",
                proj_ig,
                "Projected 128D text channel attribution: Integrated Gradients",
                "projected channel",
            )
            plot_importance_heatmap(
                sample_dir / "projected_128_integrated_gradients_heatmap.png",
                proj_ig,
                "Projected 128D Integrated Gradients heatmap",
            )
        plot_scaling_curve(sample_dir / "vector_scaling_curve.png", scaling_rows)
        if raw_occ_rows:
            plot_occlusion_curve(
                sample_dir / "raw_topk_occlusion_curve.png",
                raw_occ_rows,
                "Raw text-vector Top-K occlusion",
            )
        if proj_occ_rows:
            plot_occlusion_curve(
                sample_dir / "projected_topk_occlusion_curve.png",
                proj_occ_rows,
                "Projected text-channel Top-K occlusion",
            )


def save_shuffle_sample_outputs(
    sample_dir: Path,
    shuffle_rows: list[dict[str, Any]],
    diff_raw_mean: Optional[np.ndarray],
    diff_proj_mean: Optional[np.ndarray],
    diff_raw_stack: list[np.ndarray],
    diff_proj_stack: list[np.ndarray],
    cfg: RuntimeConfig,
) -> None:
    if not shuffle_rows and diff_raw_mean is None and diff_proj_mean is None:
        return
    if shuffle_rows:
        write_csv(sample_dir / "shuffle_predictions.csv", shuffle_rows)
    np.savez_compressed(
        sample_dir / "shuffle_attribution_arrays.npz",
        differential_raw_integrated_gradients=np.array([]) if diff_raw_mean is None else diff_raw_mean,
        differential_projected_integrated_gradients=np.array([]) if diff_proj_mean is None else diff_proj_mean,
        differential_raw_integrated_gradients_stack=(
            np.array([]) if not diff_raw_stack else np.stack(diff_raw_stack, axis=0)
        ),
        differential_projected_integrated_gradients_stack=(
            np.array([]) if not diff_proj_stack else np.stack(diff_proj_stack, axis=0)
        ),
    )
    if diff_raw_mean is not None:
        write_csv(
            sample_dir / "differential_raw_1536_integrated_gradients.csv",
            make_importance_rows("dim", diff_raw_mean),
        )
    if diff_proj_mean is not None:
        write_csv(
            sample_dir / "differential_projected_128_integrated_gradients.csv",
            make_importance_rows("channel", diff_proj_mean),
        )
    if cfg.save_plots:
        if diff_raw_mean is not None:
            plot_importance_bar(
                sample_dir / "differential_raw_1536_integrated_gradients_top_dims.png",
                diff_raw_mean,
                "Differential IG: shuffled raw text vector → matched raw text vector",
                "raw text dimension",
            )
            plot_importance_heatmap(
                sample_dir / "differential_raw_1536_integrated_gradients_heatmap.png",
                diff_raw_mean,
                "Differential raw 1536D IG heatmap",
            )
        if diff_proj_mean is not None:
            plot_importance_bar(
                sample_dir / "differential_projected_128_integrated_gradients_top_channels.png",
                diff_proj_mean,
                "Differential IG: shuffled projected text → matched projected text",
                "projected channel",
            )
            plot_importance_heatmap(
                sample_dir / "differential_projected_128_integrated_gradients_heatmap.png",
                diff_proj_mean,
                "Differential projected 128D IG heatmap",
            )


# =============================================================================
# 样本选择与单样本实验
# =============================================================================
def select_sample_indices(
    scan_rows: list[dict[str, Any]],
    test_dataset: Any,
    cfg: RuntimeConfig,
    run_dir: Path,
    logger: logging.Logger,
) -> list[int]:
    if cfg.sample_selection_mode == "manual" or not scan_rows:
        indices = [int(i) for i in cfg.manual_test_sample_indices]
    else:
        if cfg.sample_selection_mode == "top_abs_delta":
            sorted_rows = sorted(scan_rows, key=lambda r: float(r["abs_delta_text"]), reverse=True)
        elif cfg.sample_selection_mode == "top_text_help":
            sorted_rows = sorted(scan_rows, key=lambda r: float(r["text_help"]), reverse=True)
        elif cfg.sample_selection_mode == "top_text_hurt":
            sorted_rows = sorted(scan_rows, key=lambda r: float(r["text_help"]))
        elif cfg.sample_selection_mode == "top_abs_delta_shuffle":
            sorted_rows = sorted(scan_rows, key=lambda r: float(r.get("abs_delta_shuffle", 0.0)), reverse=True)
        elif cfg.sample_selection_mode == "top_text_help_shuffle":
            sorted_rows = sorted(scan_rows, key=lambda r: float(r.get("text_help_shuffle", 0.0)), reverse=True)
        elif cfg.sample_selection_mode == "top_shuffle_hurt":
            sorted_rows = sorted(scan_rows, key=lambda r: float(r.get("text_help_shuffle", 0.0)))
        else:
            raise ValueError(f"unsupported sample selection mode: {cfg.sample_selection_mode}")
        indices = [int(r["test_sample_idx"]) for r in sorted_rows[: cfg.num_samples_to_analyze]]

    indices = [i for i in indices if 0 <= i < len(test_dataset)]
    indices = list(dict.fromkeys(indices))[: cfg.num_samples_to_analyze]
    write_csv(
        run_dir / "selected_samples.csv",
        [{"test_sample_idx": i, "original_idx": int(test_dataset.original_indices[i]), "material_id": safe_get_id(test_dataset.data[i])} for i in indices],
    )
    logger.info("selected %d samples: %s", len(indices), indices[:20])
    return indices


def choose_importance(primary_ig: Optional[np.ndarray], gxi: Optional[np.ndarray], cfg: RuntimeConfig) -> Optional[np.ndarray]:
    if cfg.importance_ranking_method == "ig":
        if primary_ig is not None and primary_ig.size:
            return primary_ig
        return gxi
    if gxi is not None and gxi.size:
        return gxi
    return primary_ig


def analyze_one_sample(
    model: torch.nn.Module,
    test_dataset: Any,
    sample_idx: int,
    train_mean_t: Optional[torch.Tensor],
    cache_np: Optional[np.ndarray],
    shuffle_maps: Optional[list[np.ndarray]],
    cfg: RuntimeConfig,
    device: torch.device,
    samples_root: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    info = test_dataset.data[sample_idx]
    original_idx = int(test_dataset.original_indices[sample_idx])
    material_id = safe_get_id(info)
    safe_id = material_id.replace("/", "_").replace("\\", "_").replace(":", "_")
    sample_dir = samples_root / f"test_{sample_idx:06d}_orig_{original_idx}_{safe_id}"

    inputs, targets = get_single_batch(test_dataset, sample_idx, device)
    true_y = as_float(targets[:, cfg.target_index])
    pred_normal = predict_single_mode(model, inputs, cfg, "normal", device)
    pred_bypass = predict_single_mode(model, inputs, cfg, "bypass", device)
    delta_text = pred_normal - pred_bypass

    shuffle_rows: list[dict[str, Any]] = []
    diff_raw_stack: list[np.ndarray] = []
    diff_proj_stack: list[np.ndarray] = []
    diff_raw_mean: Optional[np.ndarray] = None
    diff_proj_mean: Optional[np.ndarray] = None

    z = inputs["text_emb"].detach().clone().to(device)
    ig_baseline = make_baseline_text_emb(z, cfg.ig_baseline, train_mean_t)
    scaling_baseline = make_baseline_text_emb(z, cfg.scaling_baseline, train_mean_t)

    raw_gxi = raw_ig = proj_gxi = proj_ig = None
    if cfg.run_raw_grad_x_input:
        raw_gxi = raw_gradient_x_input(model, inputs, cfg, device)
    if cfg.run_raw_integrated_gradients:
        raw_ig = raw_integrated_gradients(model, inputs, ig_baseline, cfg, device)
    if cfg.run_projected_grad_x_input:
        proj_gxi = projected_gradient_x_input(model, inputs, cfg, device)
    if cfg.run_projected_integrated_gradients:
        proj_ig = projected_integrated_gradients(model, inputs, ig_baseline, cfg, device)

    if (
        cfg.run_shuffle_text_experiment
        and cfg.run_shuffle_attribution
        and cfg.run_differential_ig
        and cache_np is not None
        and shuffle_maps
    ):
        n_baselines = min(cfg.sample_shuffle_baselines, len(shuffle_maps))
        for r in range(n_baselines):
            shuffle_original_idx = int(shuffle_maps[r][sample_idx])
            z_shuffle = cache_batch_to_tensor(cache_np, [shuffle_original_idx], device)
            pred_shuffle = predict_single_with_text_emb(model, inputs, z_shuffle, cfg, device)
            shuffle_rows.append(
                {
                    "repeat": int(r),
                    "shuffle_original_idx": shuffle_original_idx,
                    "pred_shuffle": pred_shuffle,
                    "delta_normal_minus_shuffle": pred_normal - pred_shuffle,
                    "abs_delta_normal_minus_shuffle": abs(pred_normal - pred_shuffle),
                    "error_shuffle": abs(pred_shuffle - true_y),
                    "text_help_shuffle": abs(pred_shuffle - true_y) - abs(pred_normal - true_y),
                }
            )
            diff_raw_stack.append(
                raw_differential_integrated_gradients(
                    model=model,
                    inputs=inputs,
                    z_from=z_shuffle,
                    z_to=z,
                    cfg=cfg,
                    device=device,
                )
            )
            diff_proj_stack.append(
                projected_differential_integrated_gradients(
                    model=model,
                    inputs=inputs,
                    z_from=z_shuffle,
                    z_to=z,
                    cfg=cfg,
                    device=device,
                )
            )
        if diff_raw_stack:
            diff_raw_mean = np.mean(np.stack(diff_raw_stack, axis=0), axis=0)
        if diff_proj_stack:
            diff_proj_mean = np.mean(np.stack(diff_proj_stack, axis=0), axis=0)

    scaling_rows = vector_scaling_curve(model, inputs, scaling_baseline, cfg, device)

    raw_occ_rows: list[dict[str, Any]] = []
    proj_occ_rows: list[dict[str, Any]] = []
    if cfg.run_topk_occlusion:
        raw_importance = choose_importance(raw_ig, raw_gxi, cfg)
        if raw_importance is not None:
            raw_occ_rows = topk_raw_occlusion(model, inputs, ig_baseline, raw_importance, cfg, device)
        proj_importance = choose_importance(proj_ig, proj_gxi, cfg)
        if proj_importance is not None:
            proj_occ_rows = topk_projected_occlusion(model, inputs, ig_baseline, proj_importance, cfg, device)

    raw_main = choose_importance(raw_ig, raw_gxi, cfg)
    proj_main = choose_importance(proj_ig, proj_gxi, cfg)
    summary = {
        "test_sample_idx": int(sample_idx),
        "original_idx": int(original_idx),
        "material_id": material_id,
        "target_name": cfg.target_name,
        "target_index": int(cfg.target_index),
        "true_y": true_y,
        "pred_normal": pred_normal,
        "pred_bypass": pred_bypass,
        "delta_text": delta_text,
        "abs_delta_text": abs(delta_text),
        "error_normal": abs(pred_normal - true_y),
        "error_bypass": abs(pred_bypass - true_y),
        "text_help": abs(pred_bypass - true_y) - abs(pred_normal - true_y),
        "pred_shuffle_mean": (
            None if not shuffle_rows else float(np.mean([r["pred_shuffle"] for r in shuffle_rows]))
        ),
        "pred_shuffle_std": (
            None if not shuffle_rows else float(np.std([r["pred_shuffle"] for r in shuffle_rows]))
        ),
        "delta_shuffle_mean": (
            None if not shuffle_rows else float(pred_normal - np.mean([r["pred_shuffle"] for r in shuffle_rows]))
        ),
        "text_help_shuffle_mean": (
            None if not shuffle_rows else float(np.mean([r["text_help_shuffle"] for r in shuffle_rows]))
        ),
        "shuffle_mode": cfg.shuffle_mode if cfg.run_shuffle_text_experiment else None,
        "shuffle_baselines_used": len(shuffle_rows),
        "differential_ig_steps": int(cfg.differential_ig_steps),
        "top_raw_dims_by_abs_differential_weight": (
            [] if diff_raw_mean is None else [int(i) for i in np.argsort(-np.abs(diff_raw_mean))[:20]]
        ),
        "top_projected_channels_by_abs_differential_weight": (
            [] if diff_proj_mean is None else [int(i) for i in np.argsort(-np.abs(diff_proj_mean))[:20]]
        ),
        "text_emb_shape": list(z.shape),
        "ig_baseline": cfg.ig_baseline,
        "ig_steps": int(cfg.ig_steps),
        "top_raw_dims_by_abs_weight": (
            [] if raw_main is None else [int(i) for i in np.argsort(-np.abs(raw_main))[:20]]
        ),
        "top_projected_channels_by_abs_weight": (
            [] if proj_main is None else [int(i) for i in np.argsort(-np.abs(proj_main))[:20]]
        ),
    }
    save_sample_outputs(
        sample_dir=sample_dir,
        summary=summary,
        raw_gxi=raw_gxi,
        raw_ig=raw_ig,
        proj_gxi=proj_gxi,
        proj_ig=proj_ig,
        scaling_rows=scaling_rows,
        raw_occ_rows=raw_occ_rows,
        proj_occ_rows=proj_occ_rows,
        cfg=cfg,
    )
    save_shuffle_sample_outputs(
        sample_dir=sample_dir,
        shuffle_rows=shuffle_rows,
        diff_raw_mean=diff_raw_mean,
        diff_proj_mean=diff_proj_mean,
        diff_raw_stack=diff_raw_stack,
        diff_proj_stack=diff_proj_stack,
        cfg=cfg,
    )
    logger.info(
        "sample %d | id=%s | true=%.6f | normal=%.6f | bypass=%.6f | |delta|=%.6f",
        sample_idx,
        material_id,
        true_y,
        pred_normal,
        pred_bypass,
        abs(delta_text),
    )
    return summary


def aggregate_sample_attributions(samples_root: Path, run_dir: Path, logger: logging.Logger) -> None:
    raw_igs = []
    proj_igs = []
    raw_gxis = []
    proj_gxis = []
    for npz_path in samples_root.glob("*/attribution_arrays.npz"):
        arr = np.load(npz_path)
        if "raw_integrated_gradients" in arr and arr["raw_integrated_gradients"].size:
            raw_igs.append(arr["raw_integrated_gradients"])
        if "projected_integrated_gradients" in arr and arr["projected_integrated_gradients"].size:
            proj_igs.append(arr["projected_integrated_gradients"])
        if "raw_grad_x_input" in arr and arr["raw_grad_x_input"].size:
            raw_gxis.append(arr["raw_grad_x_input"])
        if "projected_grad_x_input" in arr and arr["projected_grad_x_input"].size:
            proj_gxis.append(arr["projected_grad_x_input"])

    def save_agg(name: str, values_list: list[np.ndarray], prefix: str):
        if not values_list:
            return
        values = np.stack(values_list, axis=0)
        mean_signed = values.mean(axis=0)
        mean_abs = np.abs(values).mean(axis=0)
        norm = mean_abs / max(float(mean_abs.sum()), 1e-12)
        rows = []
        for i in range(values.shape[1]):
            rows.append(
                {
                    prefix: i,
                    "mean_signed_weight": float(mean_signed[i]),
                    "mean_abs_weight": float(mean_abs[i]),
                    "normalized_mean_abs_weight": float(norm[i]),
                }
            )
        rows.sort(key=lambda r: r["mean_abs_weight"], reverse=True)
        write_csv(run_dir / f"mean_{name}.csv", rows)
        plot_importance_bar(
            run_dir / f"mean_{name}_top.png",
            mean_signed,
            f"Mean {name}",
            prefix,
        )

    save_agg("raw_1536_integrated_gradients", raw_igs, "dim")
    save_agg("projected_128_integrated_gradients", proj_igs, "channel")
    save_agg("raw_1536_grad_x_input", raw_gxis, "dim")
    save_agg("projected_128_grad_x_input", proj_gxis, "channel")
    logger.info("aggregated attribution files saved")


def aggregate_shuffle_attributions(samples_root: Path, run_dir: Path, logger: logging.Logger) -> None:
    raw_diffs = []
    proj_diffs = []
    for npz_path in samples_root.glob("*/shuffle_attribution_arrays.npz"):
        arr = np.load(npz_path)
        if (
            "differential_raw_integrated_gradients" in arr
            and arr["differential_raw_integrated_gradients"].size
        ):
            raw_diffs.append(arr["differential_raw_integrated_gradients"])
        if (
            "differential_projected_integrated_gradients" in arr
            and arr["differential_projected_integrated_gradients"].size
        ):
            proj_diffs.append(arr["differential_projected_integrated_gradients"])

    def save_agg(name: str, values_list: list[np.ndarray], prefix: str):
        if not values_list:
            return
        values = np.stack(values_list, axis=0)
        mean_signed = values.mean(axis=0)
        mean_abs = np.abs(values).mean(axis=0)
        norm = mean_abs / max(float(mean_abs.sum()), 1e-12)
        rows = []
        for i in range(values.shape[1]):
            rows.append(
                {
                    prefix: i,
                    "mean_signed_weight": float(mean_signed[i]),
                    "mean_abs_weight": float(mean_abs[i]),
                    "normalized_mean_abs_weight": float(norm[i]),
                }
            )
        rows.sort(key=lambda r: r["mean_abs_weight"], reverse=True)
        write_csv(run_dir / f"mean_{name}.csv", rows)
        plot_importance_bar(
            run_dir / f"mean_{name}_top.png",
            mean_signed,
            f"Mean {name}",
            prefix,
        )

    save_agg("differential_raw_1536_integrated_gradients", raw_diffs, "dim")
    save_agg("differential_projected_128_integrated_gradients", proj_diffs, "channel")
    logger.info("aggregated shuffle differential attribution files saved")


# =============================================================================
# 主程序
# =============================================================================
def main() -> None:
    cfg = build_config()
    validate_config(cfg)
    seed_everything(cfg.seed)

    exp_name = cfg.experiment_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = cfg.output_root / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(run_dir)

    logger.info("========== Fusiformer Text Vector Effect Experiment ==========")
    logger.info("base path: %s", BASE_PATH)
    logger.info("run dir: %s", run_dir)
    save_json(run_dir / "run_config.json", {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()})

    if cfg.device_name == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(cfg.device_name)
    logger.info("device: %s", device)

    train_dataset, val_dataset, test_dataset = build_datasets(cfg, logger)
    if test_dataset is None or len(test_dataset) == 0:
        raise RuntimeError("test dataset is empty; cannot run text vector effect experiment")

    cache_np: Optional[np.ndarray] = None
    shuffle_maps: Optional[list[np.ndarray]] = None
    if cfg.run_shuffle_text_experiment:
        cache_np = load_cache_memmap(cfg.text_embedding_cache_path)
        shuffle_maps = build_shuffle_maps(test_dataset, cfg, run_dir, logger)

    model = build_model(cfg, device)
    load_model_state(model, cfg.checkpoint_path, logger)
    model.eval()

    train_mean_np: Optional[np.ndarray] = None
    train_mean_t: Optional[torch.Tensor] = None
    needs_train_mean = (
        cfg.ig_baseline == "train_mean"
        or cfg.scaling_baseline == "train_mean"
    )
    if needs_train_mean:
        train_mean_np = compute_train_text_mean(train_dataset, cfg.text_embedding_cache_path, logger)
        np.save(run_dir / "train_text_embedding_mean.npy", train_mean_np)
        train_mean_t = torch.from_numpy(train_mean_np).to(device)

    scan_rows: list[dict[str, Any]] = []
    scan_csv = run_dir / "test_text_effect_summary.csv"
    if cfg.run_test_text_effect_scan:
        if cfg.run_shuffle_text_experiment and cache_np is not None and shuffle_maps is not None:
            scan_rows = run_text_effect_shuffle_scan(
                model=model,
                test_dataset=test_dataset,
                cache_np=cache_np,
                shuffle_maps=shuffle_maps,
                cfg=cfg,
                device=device,
                run_dir=run_dir,
                logger=logger,
            )
        else:
            scan_rows = run_text_effect_scan(model, test_dataset, cfg, device, run_dir, logger)
    elif scan_csv.is_file():
        with scan_csv.open("r", encoding="utf-8") as f:
            scan_rows = list(csv.DictReader(f))

    if cfg.run_sample_analysis:
        samples_root = run_dir / "samples"
        sample_indices = select_sample_indices(scan_rows, test_dataset, cfg, run_dir, logger)
        sample_summaries = []
        for n, sample_idx in enumerate(sample_indices, start=1):
            summary = analyze_one_sample(
                model=model,
                test_dataset=test_dataset,
                sample_idx=sample_idx,
                train_mean_t=train_mean_t,
                cache_np=cache_np,
                shuffle_maps=shuffle_maps,
                cfg=cfg,
                device=device,
                samples_root=samples_root,
                logger=logger,
            )
            sample_summaries.append(summary)
            if n % SAMPLE_LOG_INTERVAL == 0:
                logger.info("analyzed samples: %d/%d", n, len(sample_indices))
        write_csv(run_dir / "sample_analysis_summary.csv", sample_summaries)
        aggregate_sample_attributions(samples_root, run_dir, logger)
        if cfg.run_shuffle_text_experiment and cfg.run_shuffle_attribution:
            aggregate_shuffle_attributions(samples_root, run_dir, logger)

    generate_additional_heatmaps(run_dir, cfg, logger)

    logger.info("all done. outputs saved to: %s", run_dir)


if __name__ == "__main__":
    main()
