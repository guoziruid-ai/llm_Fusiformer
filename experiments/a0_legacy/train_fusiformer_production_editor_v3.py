#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fusiformer 正式训练程序（直接运行版）
====================================

使用方式
--------
1. 只编辑本文件顶部“可编辑运行配置”中的变量。
2. 将本文件放在 Fusiformer 项目根目录。
3. 直接运行：

       python train_fusiformer_production_editor_v3.py

不需要也不会读取命令行训练参数。

训练设计
--------
- AdamW，betas=(0.9, 0.999)
- 一维参数与 bias 不使用 weight decay
- 未设置绝对学习率时：lr = blr * effective_batch_size / 256
- effective_batch_size 会计入梯度累积步数
- warmup 与 half-cycle cosine 均按真实 optimizer.step() 次数推进
- AMP 梯度溢出导致更新跳过时，学习率进度不会前移
- 支持 scratch、resume、weights、eval、plot_only 五种模式
- 保存完整续训检查点、最佳轻量权重、CSV、日志、TensorBoard 和曲线
- 所有相对路径都相对于本脚本所在的项目根目录解析
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os

# 缓解 CUDA 长时间训练中的显存碎片；必须在导入 torch 前设置。
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:128",
)

import random
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Optional

SCRIPT_VERSION = "2026-08-11-qwen-cache-extend-after50-v1"

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from transformers import AutoConfig

from loader.loader_cached import get_dataset, GraphCollectingCached
from loader.transforms import (
    AddAngleFeature,
    AddEdgeFeature,
    AddNodeFeature,
    Compose,
    GraphCollecting,
)

try:
    from engine import to_device
except ImportError:
    def to_device(value: Any, device: torch.device) -> Any:
        """engine.to_device 不可用时的递归兼容实现。"""
        if isinstance(value, dict):
            return {k: to_device(v, device) for k, v in value.items()}
        if isinstance(value, list):
            return [to_device(v, device) for v in value]
        if isinstance(value, tuple):
            return tuple(to_device(v, device) for v in value)
        if hasattr(value, "to"):
            return value.to(device, non_blocking=True)
        return value

try:
    from models.TextGuidedFusiformerCached import TextGuidedFusiformer
except ImportError:
    from TextGuidedFusiformerCached import TextGuidedFusiformer

try:
    # 优先使用原项目实现。
    from tools.weight_decay import add_weight_decay as project_add_weight_decay
except ImportError:
    project_add_weight_decay = None

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


# 统一路径基准：脚本从任何工作目录启动，都以脚本所在目录作为项目根目录。
base_path = Path(__file__).resolve().parents[2]
BASE_PATH = base_path
PROJECT_ROOT = BASE_PATH  # 保留旧名称，兼容日志和检查点配置。
RUNS_PATH = BASE_PATH / "runs"

# DEFAULT_DATA_PATH = Path("data") / "my_dataset.json"
# DEFAULT_OUTPUT_ROOT = Path("runs") / "fusiformer_production"
# DEFAULT_QWEN_MODEL = "/root/models/Qwen2.5-1.5B-Instruct"

# =============================================================================
# 可编辑运行配置
# =============================================================================
# 修改本区域后，直接运行：
#     python train_fusiformer_production_editor.py
#
# 所有相对路径均相对于本脚本所在目录。训练产物强制保存在 ./runs/。

# -----------------------
# 1. 启动模式与检查点
# -----------------------
# 可选："scratch"、"resume"、"weights"、"eval"、"plot_only"
training_mode = "resume"

# resume：必须指向包含 optimizer_state_dict 的完整检查点。
resume_checkpoint_path = "./runs/fusiformer_production/20260811-134939/checkpoints/last_checkpoint.pt"

# weights：只加载模型参数，优化器、epoch 和最佳指标重新初始化。
weights_checkpoint_path = None

# eval：只评估指定检查点，不训练。
eval_checkpoint_path = None

# 手动指定实验目录：
# - None：scratch/weights/eval 自动创建时间戳目录；resume 自动沿用原实验目录。
# - plot_only：必须填写已有实验目录。
run_log_dir = None

# 新实验日志根目录。
output_root = RUNS_PATH / "fusiformer_production"


# -----------------------
# 2. 数据配置
# -----------------------
data_path = "./data/my_dataset_wiki_prompt_robocrys.json"
target_name = "e_form"

# 二选一:
# - 按比例划分：data_split_counts = None，并设置 data_split_ratio。
data_split_ratio = [0.8, 0.1, 0.1]
# - 按数量划分：设置 data_split_counts=[训练数, 验证数, 测试数]。
data_split_counts = None


# -----------------------
# 3. 模型配置
# -----------------------
qwen_model_name = "/root/models/Qwen2.5-1.5B-Instruct"
allow_remote_model = False

# True: 训练阶段不再加载/执行 Qwen，直接读取预计算的 pooled embedding。
# cache 保存的是 text_proj 之前的 Qwen 表示，因此 text_proj 仍参与训练。
use_precomputed_text_embeddings = True
text_embedding_cache_path = (
    "./data/qwen_cache/qwen2p5_1p5b_description_meanpool_f32.npy"
)
text_embedding_meta_path = None
text_field = "description"
text_max_length = 512

model_depth = 4
edge_input_dim = 80
triplet_input_dim = 40
embed_dim = 128
num_heads = 4

freeze_qwen = True
use_text = True


# -----------------------
# 4. 正式训练与学习率配置
# -----------------------
# batch_size 是单次前向/反向使用的 micro-batch 大小。
batch_size = 32

# 每累计多少个 micro-batch 执行一次 optimizer.step()。
# 有效 batch size = batch_size * gradient_accumulation_steps。
gradient_accumulation_steps = 2

# ------------------------------------------------------------------
# 50 epoch 完成后的“延长训练”配置
# ------------------------------------------------------------------
# 原实验已经完成的学习率计划长度。不要修改，否则严格 resume 会改变原 cosine 曲线。
lr_schedule_total_epochs = 50

# 想在原 50 epoch 基础上再训练多少个 epoch。
# 例如 20 表示继续训练 Epoch 51~70。
additional_epochs = 20

# 训练循环的最终总 epoch 数。
total_epochs = lr_schedule_total_epochs + additional_epochs

# True：自动学习率缩放会计入梯度累积后的有效 batch size。
# False：只按单次 micro-batch 大小缩放，适合希望保持旧实验峰值 lr 的情况。
scale_lr_with_accumulation = True

# absolute_lr=None 时：
#     peak_lr = base_lr * lr_scaling_batch_size / 256
# 默认 batch_size=8、accumulation=8、base_lr=1e-3 时有效 batch=64，peak_lr=2.5e-4。
absolute_lr = None
base_lr = 1e-3

# minimum_lr 与 minimum_lr_ratio 必须且只能设置一个。
# 推荐使用比例，使最小学习率始终和峰值学习率匹配。
minimum_lr = None
minimum_lr_ratio = 0.01

# warmup 按 optimizer 更新次数定义，而不是按 epoch 定义。
# warmup_steps_override=None 时：warmup_steps = total_update_steps * warmup_ratio。
warmup_ratio = 0.05
warmup_steps_override = None

# 第一次参数更新使用非零学习率，避免第一个 batch 完全不更新。
warmup_initial_lr = 1e-7

# 完整续训时，若学习率调度相关配置发生变化：
# True 直接报错，保证严格续训；False 仅警告并按当前配置重新计算曲线。
strict_resume_schedule = True

weight_decay = 0.05
adam_beta1 = 0.9
adam_beta2 = 0.999

# 可选的模块差分学习率。默认关闭以保持旧检查点的优化器参数组兼容。
# 解冻 Qwen 时建议开启，并让 qwen_lr_scale 明显小于 1。
use_module_lr_scales = False
backbone_lr_scale = 1.0
text_module_lr_scale = 1.0
regression_head_lr_scale = 1.0
qwen_lr_scale = 0.1

# None 表示不裁剪梯度；正数表示在真正 optimizer.step() 前裁剪。
clip_grad_norm = None


# -----------------------
# 5. 运行环境
# -----------------------
device_name = "cuda"
random_seed = 42
num_workers = 8
pin_memory = True
drop_last = False

# 原项目入口默认未开启 AMP；确认模型兼容后可改为 True。
use_amp = False
use_deterministic_algorithms = False


# -----------------------
# 6. 日志、模型选择与保存
# -----------------------
# auto：有验证集时使用 val_mae，否则使用 train_mae。
best_model_monitor = "auto"

save_interval = 10
log_interval = 25

# 0 表示关闭提前停止。
early_stopping_patience = 10

enable_tensorboard = True
exclude_frozen_qwen_from_checkpoint = True

# 正式实验建议保持 False，仅在训练结束时用最佳模型评估测试集。
evaluate_test_each_epoch = False


STEP_FIELDS = [
    "epoch",
    "step",
    "micro_step",
    "global_step",
    "accumulation_position",
    "update_applied",
    "progress_epoch",
    "loss",
    "lr",
    "min_group_lr",
    "grad_norm",
    "amp_scale",
]
EPOCH_FIELDS = [
    "epoch",
    "train_mse",
    "train_mae",
    "train_rmse",
    "val_mse",
    "val_mae",
    "val_rmse",
    "optimizer_steps_end",
    "micro_steps_end",
    "lr_end",
    "epoch_seconds",
]


# -----------------------------------------------------------------------------
# 配置构建与路径
# -----------------------------------------------------------------------------

def build_runtime_config() -> SimpleNamespace:
    """把文件顶部的可编辑变量整理成运行配置。"""
    ratio = None if data_split_counts is not None else list(data_split_ratio)
    counts = None if data_split_counts is None else list(data_split_counts)

    return SimpleNamespace(
        mode=str(training_mode).strip().lower(),
        data_path=data_path,
        target=target_name,
        output_root=output_root,
        run_dir=run_log_dir,
        resume=resume_checkpoint_path,
        weights=weights_checkpoint_path,
        eval_checkpoint=eval_checkpoint_path,
        split_ratio=ratio,
        split_counts=counts,
        qwen_model_name=qwen_model_name,
        allow_remote_model=bool(allow_remote_model),
        use_precomputed_text_embeddings=bool(use_precomputed_text_embeddings),
        text_embedding_cache_path=text_embedding_cache_path,
        text_embedding_meta_path=text_embedding_meta_path,
        text_field=str(text_field),
        text_max_length=int(text_max_length),
        depth=int(model_depth),
        edge_input_dim=int(edge_input_dim),
        triplet_input_dim=int(triplet_input_dim),
        embed_dim=int(embed_dim),
        num_heads=int(num_heads),
        freeze_qwen=bool(freeze_qwen),
        use_text=bool(use_text),
        batch_size=int(batch_size),
        accumulation_steps=int(gradient_accumulation_steps),
        epochs=int(total_epochs),
        lr_schedule_epochs=int(lr_schedule_total_epochs),
        scale_lr_with_accumulation=bool(scale_lr_with_accumulation),
        lr=None if absolute_lr is None else float(absolute_lr),
        blr=float(base_lr),
        min_lr=None if minimum_lr is None else float(minimum_lr),
        min_lr_ratio=(
            None if minimum_lr_ratio is None else float(minimum_lr_ratio)
        ),
        warmup_ratio=float(warmup_ratio),
        warmup_steps_override=(
            None if warmup_steps_override is None else int(warmup_steps_override)
        ),
        warmup_init_lr=float(warmup_initial_lr),
        strict_resume_schedule=bool(strict_resume_schedule),
        weight_decay=float(weight_decay),
        beta1=float(adam_beta1),
        beta2=float(adam_beta2),
        use_module_lr_scales=bool(use_module_lr_scales),
        backbone_lr_scale=float(backbone_lr_scale),
        text_lr_scale=float(text_module_lr_scale),
        head_lr_scale=float(regression_head_lr_scale),
        qwen_lr_scale=float(qwen_lr_scale),
        clip_grad=None if clip_grad_norm is None else float(clip_grad_norm),
        device=str(device_name),
        seed=int(random_seed),
        num_workers=int(num_workers),
        pin_mem=bool(pin_memory),
        drop_last=bool(drop_last),
        amp=bool(use_amp),
        deterministic=bool(use_deterministic_algorithms),
        monitor=str(best_model_monitor).strip().lower(),
        save_interval=int(save_interval),
        log_interval=int(log_interval),
        early_stopping_patience=int(early_stopping_patience),
        tensorboard=bool(enable_tensorboard),
        exclude_frozen_qwen_from_checkpoint=bool(
            exclude_frozen_qwen_from_checkpoint
        ),
        evaluate_test_each_epoch=bool(evaluate_test_each_epoch),
    )



def validate_config(args: SimpleNamespace) -> None:
    valid_modes = {"scratch", "resume", "weights", "eval", "plot_only"}
    if args.mode not in valid_modes:
        raise ValueError(
            f"training_mode={args.mode!r} 无效，可选值为 {sorted(valid_modes)}"
        )

    valid_monitors = {
        "auto",
        "train_mse",
        "train_mae",
        "train_rmse",
        "val_mse",
        "val_mae",
        "val_rmse",
    }
    if args.monitor not in valid_monitors:
        raise ValueError(
            f"best_model_monitor={args.monitor!r} 无效，可选值为 {sorted(valid_monitors)}"
        )

    if args.epochs < 1:
        raise ValueError("total_epochs 必须大于等于 1")
    if args.lr_schedule_epochs < 1:
        raise ValueError("lr_schedule_total_epochs 必须大于等于 1")
    if args.epochs < args.lr_schedule_epochs:
        raise ValueError(
            "total_epochs 不能小于 lr_schedule_total_epochs；"
            "延长训练时 total_epochs 应大于原学习率计划长度"
        )
    if args.batch_size < 1:
        raise ValueError("batch_size 必须大于等于 1")
    if args.accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps 必须大于等于 1")
    if args.num_workers < 0:
        raise ValueError("num_workers 不能小于 0")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("warmup_ratio 必须满足 0 <= warmup_ratio < 1")
    if args.warmup_steps_override is not None and args.warmup_steps_override < 0:
        raise ValueError("warmup_steps_override 不能小于 0")
    if args.warmup_init_lr <= 0:
        raise ValueError("warmup_initial_lr 必须大于 0")

    min_lr_values_set = int(args.min_lr is not None) + int(
        args.min_lr_ratio is not None
    )
    if min_lr_values_set != 1:
        raise ValueError(
            "minimum_lr 与 minimum_lr_ratio 必须且只能设置一个，另一个设为 None"
        )
    if args.min_lr is not None and args.min_lr < 0:
        raise ValueError("minimum_lr 不能小于 0")
    if args.min_lr_ratio is not None and not 0.0 <= args.min_lr_ratio < 1.0:
        raise ValueError("minimum_lr_ratio 必须满足 0 <= ratio < 1")

    if args.lr is not None and args.lr <= 0:
        raise ValueError("absolute_lr 必须大于 0，或设置为 None")
    if args.blr <= 0:
        raise ValueError("base_lr 必须大于 0")
    if args.weight_decay < 0:
        raise ValueError("weight_decay 不能小于 0")
    for name, value in {
        "backbone_lr_scale": args.backbone_lr_scale,
        "text_module_lr_scale": args.text_lr_scale,
        "regression_head_lr_scale": args.head_lr_scale,
        "qwen_lr_scale": args.qwen_lr_scale,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} 必须大于 0")
    if args.clip_grad is not None and args.clip_grad <= 0:
        raise ValueError("clip_grad_norm 必须为正数，或设置为 None")
    if args.save_interval < 1:
        raise ValueError("save_interval 必须大于等于 1")
    if args.log_interval < 1:
        raise ValueError("log_interval 必须大于等于 1")
    if args.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience 不能小于 0")

    if args.text_max_length < 1:
        raise ValueError("text_max_length 必须大于等于 1")
    if args.use_precomputed_text_embeddings:
        if not args.freeze_qwen:
            raise ValueError(
                "use_precomputed_text_embeddings=True 仅适用于 freeze_qwen=True"
            )
        if not args.text_embedding_cache_path:
            raise ValueError(
                "启用预计算文本 embedding 时必须设置 text_embedding_cache_path"
            )

    if args.split_ratio is not None:
        if len(args.split_ratio) != 3:
            raise ValueError("data_split_ratio 必须包含训练、验证、测试三个比例")
        if any(value < 0 for value in args.split_ratio):
            raise ValueError("data_split_ratio 不能包含负数")
        if not math.isclose(
            sum(args.split_ratio), 1.0, rel_tol=1e-6, abs_tol=1e-6
        ):
            raise ValueError("data_split_ratio 三项之和必须为 1")

    if args.split_counts is not None:
        if len(args.split_counts) != 3:
            raise ValueError("data_split_counts 必须包含训练、验证、测试三个数量")
        if any(value < 0 for value in args.split_counts):
            raise ValueError("data_split_counts 不能包含负数")

    required_paths = {
        "resume": args.resume,
        "weights": args.weights,
        "eval": args.eval_checkpoint,
    }
    if args.mode in required_paths and not required_paths[args.mode]:
        required_name = {
            "resume": "resume_checkpoint_path",
            "weights": "weights_checkpoint_path",
            "eval": "eval_checkpoint_path",
        }[args.mode]
        raise ValueError(
            f'training_mode="{args.mode}" 时必须设置 {required_name}'
        )

    if args.mode == "plot_only" and not args.run_dir:
        raise ValueError('training_mode="plot_only" 时必须设置 run_log_dir')


def _coerce_path_value(
    path_value: str | os.PathLike[str] | tuple[Any, ...] | list[Any] | None,
) -> str | os.PathLike[str] | None:
    """兼容配置中误写成单元素 tuple/list 的路径。"""
    if path_value is None:
        return None
    if isinstance(path_value, (tuple, list)):
        if len(path_value) != 1:
            raise TypeError(f"路径配置只能包含一个值，当前为：{path_value!r}")
        path_value = path_value[0]
    if not isinstance(path_value, (str, os.PathLike)):
        raise TypeError(
            "路径配置必须是字符串、PathLike 或 None，"
            f"当前类型={type(path_value).__name__}，值={path_value!r}"
        )
    return path_value


def resolve_path(
    path_value: str | os.PathLike[str] | tuple[Any, ...] | list[Any] | None,
) -> Optional[Path]:
    """
    解析数据、模型等普通项目路径。

    相对路径始终相对于 BASE_PATH，而不是终端当前工作目录。
    对旧配置中重复出现的项目目录名进行兼容处理。
    """
    normalized = _coerce_path_value(path_value)
    if normalized is None:
        return None

    path = Path(normalized).expanduser()
    if path.is_absolute():
        return path.resolve()

    parts = list(path.parts)
    if BASE_PATH.name in parts:
        index = len(parts) - 1 - parts[::-1].index(BASE_PATH.name)
        path = Path(*parts[index + 1:]) if index + 1 < len(parts) else Path()

    return (BASE_PATH / path).resolve()


def resolve_run_path(
    path_value: str | os.PathLike[str] | tuple[Any, ...] | list[Any] | None,
) -> Optional[Path]:
    """
    解析日志、检查点、绘图目录等训练产物路径。

    所有训练产物均被归一化到 BASE_PATH/runs 下。支持：
      - ./runs/fusiformer_production/...
      - fusiformer_production/...
      - 旧项目绝对路径 .../runs/fusiformer_production/...
    """
    normalized = _coerce_path_value(path_value)
    if normalized is None:
        return None

    path = Path(normalized).expanduser()
    parts = list(path.parts)

    if "runs" in parts:
        index = len(parts) - 1 - parts[::-1].index("runs")
        relative_tail = Path(*parts[index + 1:]) if index + 1 < len(parts) else Path()
        resolved = RUNS_PATH / relative_tail
    elif path.is_absolute():
        raise ValueError(
            "日志、结果和检查点必须位于当前脚本目录下的 runs/ 中；"
            f"当前路径为：{path}"
        )
    else:
        resolved = RUNS_PATH / path

    resolved = resolved.resolve()
    runs_resolved = RUNS_PATH.resolve()
    try:
        resolved.relative_to(runs_resolved)
    except ValueError as exc:
        raise ValueError(f"训练路径越出了 runs/：{resolved}") from exc
    return resolved




def resolve_checkpoint_path(
    path_value: str | os.PathLike[str] | tuple[Any, ...] | list[Any] | None,
) -> Optional[Path]:
    """
    专门解析 checkpoint 路径。

    与普通 runs 路径不同，checkpoint 需要兼容历史实验目录变化，例如：
        runs/old_name/exp/checkpoints/last_checkpoint.pt
    实际保存：
        runs/new_name/exp/checkpoints/last_checkpoint.pt

    策略：
      1. 原路径存在：直接使用。
      2. 原路径不存在：根据实验目录名 + checkpoint 文件名搜索。
      3. 只有唯一候选时自动修复，否则报错。
    """
    normalized = _coerce_path_value(path_value)
    if normalized is None:
        return None

    original = Path(normalized).expanduser()
    resolved = resolve_run_path(original)

    if resolved is not None and resolved.is_file():
        return resolved

    if resolved is None:
        return None

    filename = resolved.name
    if not filename.endswith(".pt"):
        return resolved

    parts = list(resolved.parts)
    experiment_name = None
    if "checkpoints" in parts:
        idx = parts.index("checkpoints")
        if idx > 0:
            experiment_name = parts[idx - 1]

    candidates = []
    for candidate in RUNS_PATH.rglob(filename):
        if not candidate.is_file():
            continue
        if experiment_name is None:
            candidates.append(candidate)
        elif candidate.parent.parent.name == experiment_name:
            candidates.append(candidate)

    if len(candidates) == 1:
        return candidates[0].resolve()

    if len(candidates) > 1:
        raise FileNotFoundError(
            "checkpoint 路径不存在，且发现多个可能候选，请手动指定：\n"
            + "\n".join(str(x) for x in candidates)
        )

    return resolved

def path_for_record(
    path_value: str | os.PathLike[str] | Path | None,
) -> Optional[str]:
    """写入 JSON 时优先保存相对于 BASE_PATH 的可迁移路径。"""
    if path_value is None:
        return None
    path = Path(path_value).expanduser().resolve()
    try:
        relative = path.relative_to(BASE_PATH.resolve())
        return f"./{relative.as_posix()}"
    except ValueError:
        return str(path)


def make_run_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = output_root / timestamp
    run_dir = base
    suffix = 1
    while run_dir.exists():
        run_dir = Path(f"{base}-{suffix:02d}")
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def select_run_dir(args: SimpleNamespace, resume_path: Optional[Path]) -> Path:
    if args.run_dir:
        run_dir = resolve_run_path(args.run_dir)
        assert run_dir is not None
        if not run_dir.exists():
            raise FileNotFoundError(
                f"指定日志目录不存在，禁止自动创建续训目录: {run_dir}"
            )
        return run_dir

    if args.mode == "resume" and resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(
                f"resume checkpoint 不存在: {resume_path}"
            )
        if resume_path.parent.name == "checkpoints":
            return resume_path.parent.parent
        return resume_path.parent

    output_root = resolve_run_path(args.output_root)
    assert output_root is not None
    return make_run_dir(output_root)


# -----------------------------------------------------------------------------
# 日志、CSV、TensorBoard 与绘图
# -----------------------------------------------------------------------------
def setup_logger(run_dir: Path, append: bool) -> logging.Logger:
    logger = logging.getLogger("fusiformer_production")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(
        run_dir / "train.log",
        mode="a" if append else "w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def log_section(logger: logging.Logger, title: str) -> None:
    logger.info("")
    logger.info("========== %s ==========", title)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(temp, path)



def create_csv_writer(
    path: Path,
    fieldnames: list[str],
    append: bool,
) -> tuple[Any, csv.DictWriter]:
    """创建 CSV writer；续训时自动迁移旧表头，避免列错位。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    should_append = append and path.exists() and path.stat().st_size > 0

    if should_append:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_fields = list(reader.fieldnames or [])
            existing_rows = list(reader)
        if existing_fields != fieldnames:
            temp = path.with_suffix(path.suffix + ".schema.tmp")
            with temp.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in existing_rows:
                    writer.writerow({key: row.get(key, "") for key in fieldnames})
            os.replace(temp, path)

    file_obj = path.open(
        "a" if should_append else "w",
        newline="",
        encoding="utf-8",
    )
    writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
    if not should_append:
        writer.writeheader()
        file_obj.flush()
    return file_obj, writer


def safe_float(value: Any) -> Optional[float]:
    if value in (None, "", "None", "nan"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None



def plot_metrics_from_csv(run_dir: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    saved: list[str] = []
    epoch_csv = run_dir / "epoch_metrics.csv"
    step_csv = run_dir / "step_metrics.csv"

    if epoch_csv.exists():
        with epoch_csv.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            x = [int(row["epoch"]) for row in rows]
            series_groups = [
                ("mse_curve.png", "MSE", ["train_mse", "val_mse"]),
                ("mae_curve.png", "MAE", ["train_mae", "val_mae"]),
                ("rmse_curve.png", "RMSE", ["train_rmse", "val_rmse"]),
            ]
            for filename, ylabel, keys in series_groups:
                plt.figure()
                plotted = False
                for key in keys:
                    values = [safe_float(row.get(key)) for row in rows]
                    if any(v is not None for v in values):
                        plt.plot(x, values, marker="o", label=key)
                        plotted = True
                if plotted:
                    plt.xlabel("Epoch")
                    plt.ylabel(ylabel)
                    plt.title(f"{ylabel} curve")
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    plt.tight_layout()
                    output = run_dir / filename
                    plt.savefig(output, dpi=200)
                    saved.append(str(output))
                plt.close()

            lr_end = [safe_float(row.get("lr_end")) for row in rows]
            if any(v is not None for v in lr_end):
                plt.figure()
                plt.plot(x, lr_end, marker="o", label="lr_end")
                plt.xlabel("Epoch")
                plt.ylabel("Learning rate")
                plt.title("Epoch-end learning rate")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                output = run_dir / "epoch_lr_curve.png"
                plt.savefig(output, dpi=200)
                plt.close()
                saved.append(str(output))

    if step_csv.exists():
        with step_csv.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            loss_x = [
                int(row.get("micro_step") or row.get("global_step") or 0)
                for row in rows
            ]
            losses = [safe_float(row.get("loss")) for row in rows]
            loss_pairs = [(x, y) for x, y in zip(loss_x, losses) if y is not None]
            if loss_pairs:
                plt.figure()
                plt.plot(
                    [item[0] for item in loss_pairs],
                    [item[1] for item in loss_pairs],
                    label="step_loss",
                )
                plt.xlabel("Micro step")
                plt.ylabel("Loss")
                plt.title("Step loss")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                output = run_dir / "step_loss_curve.png"
                plt.savefig(output, dpi=200)
                plt.close()
                saved.append(str(output))

            update_rows = [
                row
                for row in rows
                if str(row.get("update_applied", "1")).lower()
                not in {"", "0", "false", "none"}
            ]
            lr_pairs = [
                (int(row.get("global_step") or 0), safe_float(row.get("lr")))
                for row in update_rows
            ]
            lr_pairs = [(x, y) for x, y in lr_pairs if y is not None]
            if lr_pairs:
                plt.figure()
                plt.plot(
                    [item[0] for item in lr_pairs],
                    [item[1] for item in lr_pairs],
                    label="learning_rate",
                )
                plt.xlabel("Optimizer step")
                plt.ylabel("Learning rate")
                plt.title("Step-based warmup and cosine schedule")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                output = run_dir / "step_lr_curve.png"
                plt.savefig(output, dpi=200)
                plt.close()
                saved.append(str(output))

    return saved


# -----------------------------------------------------------------------------
# 随机性、优化器与学习率
# -----------------------------------------------------------------------------
def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    else:
        cudnn.benchmark = torch.cuda.is_available()
        cudnn.deterministic = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Optional[dict[str, Any]], logger: logging.Logger) -> None:
    if not state:
        logger.warning("checkpoint does not contain RNG state")
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        try:
            torch.cuda.set_rng_state_all(state["cuda"])
        except RuntimeError as exc:
            logger.warning("CUDA RNG state could not be fully restored: %s", exc)
    logger.info("random number generator states restored")



def fallback_add_weight_decay(
    model: torch.nn.Module,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """与原项目意图一致：1D 参数和 bias 不使用 weight decay。"""
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {
            "params": no_decay,
            "weight_decay": 0.0,
            "lr_scale": 1.0,
            "group_name": "no_decay",
        },
        {
            "params": decay,
            "weight_decay": weight_decay,
            "lr_scale": 1.0,
            "group_name": "decay",
        },
    ]


def parameter_lr_category(name: str) -> str:
    if name.startswith("qwen."):
        return "qwen"
    if (
        name.startswith("text_proj.")
        or ".text_cross_attn_" in name
        or ".norm_text_" in name
    ):
        return "text"
    if name.startswith("regression_head."):
        return "head"
    return "backbone"


def build_module_lr_param_groups(
    model: torch.nn.Module,
    args: SimpleNamespace,
) -> list[dict[str, Any]]:
    scales = {
        "backbone": args.backbone_lr_scale,
        "text": args.text_lr_scale,
        "head": args.head_lr_scale,
        "qwen": args.qwen_lr_scale,
    }
    grouped: dict[tuple[str, bool], dict[str, Any]] = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        category = parameter_lr_category(name)
        use_decay = not (param.ndim <= 1 or name.endswith(".bias"))
        key = (category, use_decay)
        if key not in grouped:
            grouped[key] = {
                "params": [],
                "weight_decay": args.weight_decay if use_decay else 0.0,
                "lr_scale": float(scales[category]),
                "group_name": f"{category}_{'decay' if use_decay else 'no_decay'}",
            }
        grouped[key]["params"].append(param)

    return [group for group in grouped.values() if group["params"]]



def build_optimizer(
    model: torch.nn.Module,
    args: SimpleNamespace,
    actual_lr: float,
) -> torch.optim.AdamW:
    if args.use_module_lr_scales:
        param_groups = build_module_lr_param_groups(model, args)
    elif project_add_weight_decay is not None:
        param_groups = project_add_weight_decay(model, args.weight_decay)
        for index, group in enumerate(param_groups):
            group.setdefault("lr_scale", 1.0)
            group.setdefault("group_name", f"project_group_{index}")
    else:
        param_groups = fallback_add_weight_decay(model, args.weight_decay)

    if not param_groups:
        raise RuntimeError("没有可训练参数，无法创建优化器")

    return torch.optim.AdamW(
        param_groups,
        lr=actual_lr,
        betas=(args.beta1, args.beta2),
    )


def log_optimizer_groups(
    optimizer: torch.optim.Optimizer,
    logger: logging.Logger,
) -> None:
    for index, group in enumerate(optimizer.param_groups):
        parameter_count = sum(param.numel() for param in group["params"])
        logger.info(
            "optimizer group %d | name=%s | params=%d | lr_scale=%.6g | weight_decay=%.6g",
            index,
            group.get("group_name", f"group_{index}"),
            parameter_count,
            float(group.get("lr_scale", 1.0)),
            float(group.get("weight_decay", 0.0)),
        )



def resolve_schedule_steps(
    *,
    num_batches_per_epoch: int,
    total_epochs: int,
    accumulation_steps: int,
    warmup_ratio: float,
    warmup_steps_override: Optional[int],
) -> tuple[int, int, int]:
    updates_per_epoch = math.ceil(num_batches_per_epoch / accumulation_steps)
    total_update_steps = updates_per_epoch * total_epochs
    if total_update_steps < 1:
        raise ValueError("total_update_steps 必须大于 0")

    if warmup_steps_override is None:
        warmup_steps = int(round(total_update_steps * warmup_ratio))
        if warmup_ratio > 0 and warmup_steps == 0 and total_update_steps > 1:
            warmup_steps = 1
    else:
        warmup_steps = int(warmup_steps_override)

    if warmup_steps < 0:
        raise ValueError("warmup_steps 不能小于 0")
    if warmup_steps >= total_update_steps:
        raise ValueError(
            "warmup_steps 必须小于 total_update_steps；"
            f"当前 warmup_steps={warmup_steps}, total_update_steps={total_update_steps}"
        )
    return updates_per_epoch, total_update_steps, warmup_steps


def resolve_minimum_lr(args: SimpleNamespace, peak_lr: float) -> float:
    if args.min_lr is not None:
        return float(args.min_lr)
    assert args.min_lr_ratio is not None
    return float(peak_lr) * float(args.min_lr_ratio)


def scheduled_learning_rate(
    update_step: int,
    *,
    peak_lr: float,
    min_lr: float,
    warmup_init_lr: float,
    warmup_steps: int,
    total_update_steps: int,
) -> float:
    """返回第 update_step 次真实优化器更新所使用的基础学习率。"""
    if total_update_steps <= 0:
        raise ValueError("total_update_steps 必须大于 0")
    if warmup_steps < 0 or warmup_steps >= total_update_steps:
        raise ValueError("warmup_steps 必须满足 0 <= warmup_steps < total_update_steps")
    if update_step < 0:
        raise ValueError("update_step 不能小于 0")
    if update_step >= total_update_steps:
        return min_lr

    if warmup_steps > 0 and update_step < warmup_steps:
        progress = update_step / max(1, warmup_steps)
        return warmup_init_lr + progress * (peak_lr - warmup_init_lr)

    decay_steps = total_update_steps - warmup_steps
    decay_index = update_step - warmup_steps
    if decay_steps <= 1:
        cosine_progress = 1.0
    else:
        cosine_progress = decay_index / (decay_steps - 1)
    cosine_progress = min(max(cosine_progress, 0.0), 1.0)
    return min_lr + 0.5 * (peak_lr - min_lr) * (
        1.0 + math.cos(math.pi * cosine_progress)
    )


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    update_step: int,
    *,
    peak_lr: float,
    min_lr: float,
    warmup_init_lr: float,
    warmup_steps: int,
    total_update_steps: int,
) -> float:
    """仅按成功的 optimizer.step() 计数更新学习率。"""
    lr = scheduled_learning_rate(
        update_step,
        peak_lr=peak_lr,
        min_lr=min_lr,
        warmup_init_lr=warmup_init_lr,
        warmup_steps=warmup_steps,
        total_update_steps=total_update_steps,
    )
    for group in optimizer.param_groups:
        scale = float(group.get("lr_scale", 1.0))
        group["lr"] = lr * scale
    return lr


def get_lr_range(optimizer: torch.optim.Optimizer) -> tuple[float, float]:
    values = [float(group["lr"]) for group in optimizer.param_groups]
    return min(values), max(values)



def get_optimizer_internal_step(optimizer: torch.optim.Optimizer) -> float:
    """读取 AdamW 已完成的最大内部 step，用于可靠判断 AMP 是否跳过更新。"""
    maximum = 0.0
    for state in optimizer.state.values():
        value = state.get("step")
        if value is None:
            continue
        if torch.is_tensor(value):
            value = value.detach().item()
        maximum = max(maximum, float(value))
    return maximum


def log_cuda_memory(
    device: torch.device,
    logger: logging.Logger,
    stage: str,
) -> None:
    if device.type != "cuda":
        return
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    gib = 1024 ** 3
    logger.info(
        "CUDA memory [%s] | free %.2f/%.2f GiB | allocated %.2f GiB | reserved %.2f GiB",
        stage,
        free_bytes / gib,
        total_bytes / gib,
        allocated / gib,
        reserved / gib,
    )


def build_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


# -----------------------------------------------------------------------------
# 模型、数据与指标
# -----------------------------------------------------------------------------
def normalize_targets(targets: Any, device: torch.device) -> torch.Tensor:
    if isinstance(targets, list):
        targets = torch.cat(targets, dim=-1)
    elif isinstance(targets, tuple):
        targets = torch.cat(list(targets), dim=-1)
    return targets.to(device, non_blocking=True)


def normalize_outputs(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, (list, tuple)):
        return torch.cat(list(outputs), dim=-1)
    return outputs


def clone_inputs_without_text(inputs: Any) -> Any:
    if not isinstance(inputs, dict):
        return inputs
    copied = dict(inputs)
    copied["text"] = None
    copied["text_emb"] = None
    return copied


def set_qwen_eval_if_frozen(model: torch.nn.Module) -> None:
    module = model.module if hasattr(model, "module") else model
    if (
        getattr(module, "freeze_qwen", False)
        and hasattr(module, "qwen")
        and module.qwen is not None
    ):
        module.qwen.eval()



def build_model(args: SimpleNamespace, device: torch.device) -> torch.nn.Module:
    config = AutoConfig.from_pretrained(
        args.qwen_model_name,
        local_files_only=not args.allow_remote_model,
        trust_remote_code=True,
    )
    qwen_hidden_size = int(config.hidden_size)

    model = TextGuidedFusiformer(
        targets=[args.target],
        depth=args.depth,
        edge_input_dim=args.edge_input_dim,
        triplet_input_dim=args.triplet_input_dim,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        text_dim=qwen_hidden_size,
        qwen_model_name=args.qwen_model_name,
        freeze_qwen=args.freeze_qwen,
        load_qwen=not args.use_precomputed_text_embeddings,
    )
    model.use_precomputed_text_embeddings = args.use_precomputed_text_embeddings
    # 兼容旧模型类没有保存该属性的情况；训练、检查点排除逻辑依赖它。
    model.freeze_qwen = args.freeze_qwen
    return model.to(device)


def log_model_summary(model: torch.nn.Module, logger: logging.Logger) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("model total params: %d (%.2f M)", total, total / 1e6)
    logger.info("model trainable params: %d (%.2f M)", trainable, trainable / 1e6)

    if hasattr(model, "qwen") and model.qwen is not None:
        qwen_total = sum(p.numel() for p in model.qwen.parameters())
        qwen_trainable = sum(
            p.numel() for p in model.qwen.parameters() if p.requires_grad
        )
        logger.info("qwen total params: %d", qwen_total)
        logger.info("qwen trainable params: %d", qwen_trainable)
        logger.info("qwen frozen: %s", qwen_trainable == 0)
    elif getattr(model, "use_precomputed_text_embeddings", False):
        logger.info("qwen runtime model: NOT LOADED (precomputed text embeddings)")


def build_dataloaders(
    args: SimpleNamespace,
    device: torch.device,
    logger: logging.Logger,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    transform = Compose(
        [
            AddNodeFeature(),
            AddEdgeFeature(),
            AddAngleFeature(),
            GraphCollectingCached(["graph", "line_graph"], [args.target]),
        ]
    )

    ratio = None if args.split_counts is not None else args.split_ratio
    counts = args.split_counts

    train_dataset, val_dataset, test_dataset = get_dataset(
        data_path=args.data_path,
        ratio_train_val_test=ratio,
        n_train_val_test=counts,
        transforms=transform,
        graph=True,
        line_graph=True,
        qwen_model_name=args.qwen_model_name,
        text_max_length=args.text_max_length,
        text_field=args.text_field,
        text_embedding_cache_path=(
            args.text_embedding_cache_path
            if args.use_precomputed_text_embeddings
            else None
        ),
        text_embedding_meta_path=args.text_embedding_meta_path,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_mem and device.type == "cuda",
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if args.num_workers > 0:
        common["persistent_workers"] = True

    effective_drop_last = args.drop_last and len(train_dataset) >= args.batch_size
    if args.drop_last and not effective_drop_last:
        logger.warning(
            "train dataset is smaller than batch size; drop_last is automatically disabled"
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=effective_drop_last,
        collate_fn=train_dataset.collect(),
        **common,
    )

    def make_eval_loader(dataset: Any):
        if dataset is None or len(dataset) == 0:
            return None
        return torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            drop_last=False,
            collate_fn=dataset.collect(),
            **common,
        )

    val_loader = make_eval_loader(val_dataset)
    test_loader = make_eval_loader(test_dataset)

    logger.info("train samples: %d", len(train_dataset))
    logger.info("validation samples: %d", 0 if val_dataset is None else len(val_dataset))
    logger.info("test samples: %d", 0 if test_dataset is None else len(test_dataset))
    logger.info("train batches: %d", len(train_loader))
    logger.info("batch size: %d", args.batch_size)

    if len(train_loader) == 0:
        raise RuntimeError("训练 DataLoader 为空，请检查数据划分与 batch size")

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    )


def empty_metric_accumulator() -> dict[str, float]:
    return {
        "squared_error_sum": 0.0,
        "absolute_error_sum": 0.0,
        "element_count": 0.0,
    }


def update_metric_accumulator(
    accumulator: dict[str, float],
    outputs: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    error = outputs.detach().float() - targets.detach().float()
    accumulator["squared_error_sum"] += torch.sum(error * error).item()
    accumulator["absolute_error_sum"] += torch.sum(torch.abs(error)).item()
    accumulator["element_count"] += float(error.numel())


def finalize_metrics(accumulator: dict[str, float]) -> dict[str, float]:
    count = max(accumulator["element_count"], 1.0)
    mse = accumulator["squared_error_sum"] / count
    mae = accumulator["absolute_error_sum"] / count
    return {"mse": mse, "mae": mae, "rmse": math.sqrt(mse)}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    use_text: bool,
    amp_enabled: bool,
) -> Optional[dict[str, float]]:
    if loader is None:
        return None

    model.eval()
    set_qwen_eval_if_frozen(model)
    accumulator = empty_metric_accumulator()

    for inputs, targets in loader:
        inputs = to_device(inputs, device)
        targets = normalize_targets(targets, device)
        if not use_text:
            inputs = clone_inputs_without_text(inputs)

        with autocast_context(device, amp_enabled):
            outputs = normalize_outputs(model(inputs))
        update_metric_accumulator(accumulator, outputs, targets)

    return finalize_metrics(accumulator)



def train_one_epoch(
    *,
    model: torch.nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    epoch_index: int,
    global_step: int,
    micro_step: int,
    args: SimpleNamespace,
    peak_lr: float,
    min_lr: float,
    warmup_steps: int,
    total_update_steps: int,
    logger: logging.Logger,
    step_writer: csv.DictWriter,
    step_file: Any,
    tensorboard_writer: Any,
) -> tuple[dict[str, float], int, int, float]:
    model.train()
    set_qwen_eval_if_frozen(model)

    criterion = torch.nn.MSELoss()
    accumulator = empty_metric_accumulator()
    num_steps = len(loader)
    accumulation_steps = args.accumulation_steps
    last_lr = get_lr_range(optimizer)[1]
    optimizer.zero_grad(set_to_none=True)

    for step, (inputs, targets) in enumerate(loader):
        window_start = (step // accumulation_steps) * accumulation_steps
        window_end = min(window_start + accumulation_steps, num_steps)
        window_size = window_end - window_start
        accumulation_position = step - window_start + 1
        is_update_boundary = step + 1 == window_end

        if accumulation_position == 1:
            last_lr = adjust_learning_rate(
                optimizer,
                global_step,
                peak_lr=peak_lr,
                min_lr=min_lr,
                warmup_init_lr=args.warmup_init_lr,
                warmup_steps=warmup_steps,
                total_update_steps=total_update_steps,
            )

        inputs = to_device(inputs, device)
        targets = normalize_targets(targets, device)
        if not args.use_text:
            inputs = clone_inputs_without_text(inputs)

        with autocast_context(device, args.amp and device.type == "cuda"):
            outputs = normalize_outputs(model(inputs))
            loss = criterion(outputs, targets)
            loss_for_backward = loss / window_size

        loss_value = float(loss.detach().item())
        if not math.isfinite(loss_value):
            raise RuntimeError(
                f"invalid loss at epoch={epoch_index + 1}, step={step}: {loss_value}"
            )

        scaler.scale(loss_for_backward).backward()

        grad_norm_value: Optional[float] = None
        update_applied = False
        if is_update_boundary:
            if args.clip_grad is not None:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=args.clip_grad,
                )
                grad_norm_value = float(grad_norm.detach().item())

            internal_step_before = get_optimizer_internal_step(optimizer)
            scaler.step(optimizer)
            scaler.update()
            internal_step_after = get_optimizer_internal_step(optimizer)
            update_applied = internal_step_after > internal_step_before
            optimizer.zero_grad(set_to_none=True)

            if update_applied:
                global_step += 1
            else:
                logger.warning(
                    "AMP overflow: optimizer update skipped at epoch=%d step=%d; "
                    "scheduler step remains %d",
                    epoch_index + 1,
                    step,
                    global_step,
                )

        update_metric_accumulator(accumulator, outputs, targets)
        micro_step += 1
        min_group_lr, max_group_lr = get_lr_range(optimizer)
        progress_epoch = epoch_index + (step + 1) / num_steps
        amp_scale = float(scaler.get_scale())

        step_writer.writerow(
            {
                "epoch": epoch_index + 1,
                "step": step,
                "micro_step": micro_step,
                "global_step": global_step,
                "accumulation_position": f"{accumulation_position}/{window_size}",
                "update_applied": int(update_applied),
                "progress_epoch": progress_epoch,
                "loss": loss_value,
                "lr": max_group_lr,
                "min_group_lr": min_group_lr,
                "grad_norm": "" if grad_norm_value is None else grad_norm_value,
                "amp_scale": amp_scale,
            }
        )
        step_file.flush()

        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar("train/step_loss", loss_value, micro_step)
            if update_applied:
                tensorboard_writer.add_scalar("train/lr", max_group_lr, global_step)
                tensorboard_writer.add_scalar(
                    "train/amp_scale", amp_scale, global_step
                )

        if step % args.log_interval == 0 or step == num_steps - 1:
            logger.info(
                "epoch %03d | micro %04d/%04d | accum %d/%d | "
                "optimizer_step %07d | update=%s | loss %.6f | lr %.8g",
                epoch_index + 1,
                step,
                num_steps - 1,
                accumulation_position,
                window_size,
                global_step,
                update_applied,
                loss_value,
                max_group_lr,
            )

    return finalize_metrics(accumulator), global_step, micro_step, last_lr


# -----------------------------------------------------------------------------
# 检查点
# -----------------------------------------------------------------------------
def atomic_torch_save(obj: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(obj, temp_path)
    os.replace(temp_path, output_path)


def load_torch_file(path: Path, map_location: Any) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def get_checkpoint_model_state(
    model: torch.nn.Module,
    exclude_frozen_qwen: bool,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    module = model.module if hasattr(model, "module") else model
    excluded_prefixes: list[str] = []
    if (
        exclude_frozen_qwen
        and getattr(module, "freeze_qwen", False)
        and hasattr(module, "qwen")
        and module.qwen is not None
    ):
        excluded_prefixes.append("qwen.")

    state: dict[str, torch.Tensor] = {}
    for name, value in module.state_dict().items():
        if any(name.startswith(prefix) for prefix in excluded_prefixes):
            continue
        state[name] = value.detach().cpu()
    return state, excluded_prefixes


def extract_model_state(checkpoint: Any) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must be a dict or state_dict")

    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
        excluded = checkpoint.get("excluded_model_prefixes", [])
    elif "model" in checkpoint:
        # 兼容原项目 misc.save_model 生成的检查点。
        state = checkpoint["model"]
        excluded = checkpoint.get("excluded_model_prefixes", [])
    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
        excluded = checkpoint.get("excluded_model_prefixes", [])
    elif checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        state = checkpoint
        excluded = []
    else:
        raise KeyError("checkpoint does not contain model parameters")

    if state and all(name.startswith("module.") for name in state):
        state = {name[len("module."):]: value for name, value in state.items()}
    return state, list(excluded)


def load_model_state(
    model: torch.nn.Module,
    checkpoint: Any,
    logger: logging.Logger,
) -> None:
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
            "model state loading is incomplete; "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}"
        )

    logger.info("loaded model tensors: %d", len(state))
    if excluded:
        logger.info("checkpoint excluded prefixes: %s", excluded)



def build_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    completed_epoch: int,
    global_step: int,
    micro_step: int,
    scheduler_state: dict[str, Any],
    best_metric: float,
    best_epoch: int,
    best_metrics: Optional[dict[str, Any]],
    epochs_without_improvement: int,
    monitor_name: str,
    training_config: dict[str, Any],
    args: SimpleNamespace,
) -> dict[str, Any]:
    model_state, excluded = get_checkpoint_model_state(
        model,
        args.exclude_frozen_qwen_from_checkpoint,
    )
    state = dict(scheduler_state)
    state["optimizer_step"] = global_step
    state["micro_step"] = micro_step
    state["current_group_lrs"] = [
        float(group["lr"]) for group in optimizer.param_groups
    ]
    return {
        "format_version": 4,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "epoch": completed_epoch,
        # global_step 保留为兼容字段；其含义从本版本起明确为成功的优化器更新数。
        "global_step": global_step,
        "optimizer_step": global_step,
        "micro_step": micro_step,
        "scheduler_state": state,
        "model_state_dict": model_state,
        "excluded_model_prefixes": excluded,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "epochs_without_improvement": epochs_without_improvement,
        "monitor": monitor_name,
        "rng_state": capture_rng_state(),
        "training_config": training_config,
    }


def save_lightweight_weights(path: Path, checkpoint: dict[str, Any]) -> None:
    light = {
        key: checkpoint.get(key)
        for key in [
            "format_version",
            "saved_at",
            "epoch",
            "global_step",
            "optimizer_step",
            "micro_step",
            "scheduler_state",
            "model_state_dict",
            "excluded_model_prefixes",
            "best_metric",
            "best_epoch",
            "best_metrics",
            "monitor",
            "training_config",
        ]
    }
    atomic_torch_save(light, path)




def check_resume_compatibility(
    checkpoint: dict[str, Any],
    current_config: dict[str, Any],
    logger: logging.Logger,
    strict_schedule: bool,
) -> None:
    previous_raw = checkpoint.get("training_config") or checkpoint.get("run_config") or {}
    previous = dict(previous_raw)
    keys = [
        "target",
        "qwen_model_name",
        "depth",
        "edge_input_dim",
        "triplet_input_dim",
        "embed_dim",
        "num_heads",
        "freeze_qwen",
    ]
    mismatches = []
    for key in keys:
        if key in previous and key in current_config and previous[key] != current_config[key]:
            mismatches.append((key, previous[key], current_config[key]))
    if mismatches:
        details = "; ".join(f"{k}: {old!r} -> {new!r}" for k, old, new in mismatches)
        raise ValueError(f"resume model configuration mismatch: {details}")

    # 把旧版按 epoch 调度的配置转换为可比较的近似 optimizer-step 配置。
    # 旧脚本每个 batch 都执行 optimizer.step()，因此 accumulation 默认为 1。
    legacy_schedule = not checkpoint.get("scheduler_state")
    if legacy_schedule:
        previous.setdefault("gradient_accumulation_steps", 1)
        previous.setdefault("use_module_lr_scales", False)
        previous.setdefault("backbone_lr_scale", 1.0)
        previous.setdefault("text_lr_scale", 1.0)
        previous.setdefault("head_lr_scale", 1.0)
        previous.setdefault("qwen_lr_scale", 1.0)
        if "min_lr" in previous and "resolved_min_lr" not in previous:
            previous["resolved_min_lr"] = previous["min_lr"]
        if "epochs" in previous and "total_update_steps" not in previous:
            previous["total_update_steps"] = (
                int(previous["epochs"]) * int(current_config["updates_per_epoch"])
            )
        if "warmup_epochs" in previous and "warmup_steps" not in previous:
            previous["warmup_steps"] = int(
                round(
                    float(previous["warmup_epochs"])
                    * int(current_config["updates_per_epoch"])
                )
            )
        logger.warning(
            "legacy checkpoint detected: epoch-based schedule is converted to an "
            "optimizer-step approximation for compatibility checking"
        )

    schedule_keys = [
        "actual_lr",
        "resolved_min_lr",
        "warmup_steps",
        "total_update_steps",
        "gradient_accumulation_steps",
        "use_module_lr_scales",
    ]
    if not legacy_schedule:
        schedule_keys.append("warmup_init_lr")
    if previous.get("use_module_lr_scales") or current_config.get(
        "use_module_lr_scales"
    ):
        schedule_keys.extend(
            [
                "backbone_lr_scale",
                "text_lr_scale",
                "head_lr_scale",
                "qwen_lr_scale",
            ]
        )
    schedule_changes = []
    for key in schedule_keys:
        if key in previous and key in current_config and previous[key] != current_config[key]:
            schedule_changes.append((key, previous[key], current_config[key]))

    if schedule_changes:
        details = "; ".join(
            f"{k}: {old!r} -> {new!r}" for k, old, new in schedule_changes
        )
        if strict_schedule:
            raise ValueError(
                "严格续训模式下学习率调度配置不能改变：" + details
            )
        logger.warning(
            "learning-rate schedule configuration changed during resume: %s",
            details,
        )

    if legacy_schedule:
        logger.warning(
            "legacy checkpoint does not contain scheduler_state; "
            "optimizer_step is inferred from global_step. For exact reproducibility, "
            "start a new run or use a version-4 checkpoint."
        )


# -----------------------------------------------------------------------------
# 主流程辅助
# -----------------------------------------------------------------------------
def choose_monitor_name(requested: str, has_validation: bool) -> str:
    if requested == "auto":
        return "val_mae" if has_validation else "train_mae"
    if requested.startswith("val_") and not has_validation:
        raise ValueError(f"monitor={requested} requires a validation set")
    return requested


def build_epoch_metrics(
    train_metrics: dict[str, float],
    val_metrics: Optional[dict[str, float]],
) -> dict[str, Optional[float]]:
    return {
        "train_mse": float(train_metrics["mse"]),
        "train_mae": float(train_metrics["mae"]),
        "train_rmse": float(train_metrics["rmse"]),
        "val_mse": None if val_metrics is None else float(val_metrics["mse"]),
        "val_mae": None if val_metrics is None else float(val_metrics["mae"]),
        "val_rmse": None if val_metrics is None else float(val_metrics["rmse"]),
    }


def get_monitor_value(
    monitor_name: str,
    current_metrics: dict[str, Optional[float]],
) -> float:
    value = current_metrics.get(monitor_name)
    if value is None:
        raise ValueError(f"monitor value is unavailable: {monitor_name}")
    return float(value)


def metrics_to_log_text(prefix: str, metrics: Optional[dict[str, float]]) -> str:
    if metrics is None:
        return f"{prefix}: unavailable"
    return (
        f"{prefix} MSE {metrics['mse']:.6f} | "
        f"{prefix} MAE {metrics['mae']:.6f} | "
        f"{prefix} RMSE {metrics['rmse']:.6f}"
    )



def build_training_config(
    args: SimpleNamespace,
    actual_lr: float,
    resolved_min_lr: float,
    effective_batch_size: int,
    lr_scaling_batch_size: int,
    updates_per_epoch: int,
    total_update_steps: int,
    resolved_warmup_steps: int,
    device: torch.device,
    run_dir: Path,
) -> dict[str, Any]:
    return {
        "mode": args.mode,
        "base_path": ".",
        "project_root": ".",
        "run_dir": path_for_record(run_dir),
        "data_path": path_for_record(args.data_path),
        "target": args.target,
        "split_ratio": args.split_ratio,
        "split_counts": args.split_counts,
        "qwen_model_name": args.qwen_model_name,
        "use_precomputed_text_embeddings": args.use_precomputed_text_embeddings,
        "text_embedding_cache_path": path_for_record(
            args.text_embedding_cache_path
        ) if args.text_embedding_cache_path else None,
        "text_embedding_meta_path": path_for_record(
            args.text_embedding_meta_path
        ) if args.text_embedding_meta_path else None,
        "text_field": args.text_field,
        "text_max_length": args.text_max_length,
        "depth": args.depth,
        "edge_input_dim": args.edge_input_dim,
        "triplet_input_dim": args.triplet_input_dim,
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
        "freeze_qwen": args.freeze_qwen,
        "use_text": args.use_text,
        "device": str(device),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "scale_lr_with_accumulation": args.scale_lr_with_accumulation,
        "lr_scaling_batch_size": lr_scaling_batch_size,
        "epochs": args.epochs,
        "lr_schedule_epochs": args.lr_schedule_epochs,
        "updates_per_epoch": updates_per_epoch,
        "total_update_steps": total_update_steps,
        "absolute_lr_requested": args.lr,
        "blr": args.blr,
        "actual_lr": actual_lr,
        "minimum_lr_requested": args.min_lr,
        "minimum_lr_ratio": args.min_lr_ratio,
        "resolved_min_lr": resolved_min_lr,
        "warmup_ratio": args.warmup_ratio,
        "warmup_steps_override": args.warmup_steps_override,
        "warmup_steps": resolved_warmup_steps,
        "warmup_init_lr": args.warmup_init_lr,
        "strict_resume_schedule": args.strict_resume_schedule,
        "weight_decay": args.weight_decay,
        "betas": [args.beta1, args.beta2],
        "use_module_lr_scales": args.use_module_lr_scales,
        "backbone_lr_scale": args.backbone_lr_scale,
        "text_lr_scale": args.text_lr_scale,
        "head_lr_scale": args.head_lr_scale,
        "qwen_lr_scale": args.qwen_lr_scale,
        "clip_grad": args.clip_grad,
        "amp": args.amp,
        "num_workers": args.num_workers,
        "pin_mem": args.pin_mem,
        "drop_last": args.drop_last,
        "monitor_requested": args.monitor,
        "save_interval": args.save_interval,
        "early_stopping_patience": args.early_stopping_patience,
        "resume": args.resume,
        "weights": args.weights,
        "eval_checkpoint": args.eval_checkpoint,
    }


def write_tensorboard_epoch(
    writer: Any,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: Optional[dict[str, float]],
    test_metrics: Optional[dict[str, float]],
) -> None:
    if writer is None:
        return
    for name, value in train_metrics.items():
        writer.add_scalar(f"train/{name}", value, epoch)
    if val_metrics is not None:
        for name, value in val_metrics.items():
            writer.add_scalar(f"val/{name}", value, epoch)
    if test_metrics is not None:
        for name, value in test_metrics.items():
            writer.add_scalar(f"test/{name}", value, epoch)
    writer.flush()


# -----------------------------------------------------------------------------
# 主程序
# -----------------------------------------------------------------------------

def main() -> None:
    RUNS_PATH.mkdir(parents=True, exist_ok=True)
    args = build_runtime_config()
    validate_config(args)

    if args.mode == "plot_only":
        run_dir = resolve_run_path(args.run_dir)
        assert run_dir is not None
        figures = plot_metrics_from_csv(run_dir)
        print("Generated figures:")
        for item in figures:
            print(item)
        return

    args.data_path = str(resolve_path(args.data_path))
    if not Path(args.data_path).is_file():
        raise FileNotFoundError(f"dataset does not exist: {args.data_path}")

    if args.use_precomputed_text_embeddings:
        cache_path = resolve_path(args.text_embedding_cache_path)
        assert cache_path is not None
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"text embedding cache does not exist: {cache_path}"
            )
        args.text_embedding_cache_path = str(cache_path)

        if args.text_embedding_meta_path is not None:
            meta_path = resolve_path(args.text_embedding_meta_path)
            assert meta_path is not None
            args.text_embedding_meta_path = str(meta_path)

    qwen_path = Path(args.qwen_model_name).expanduser()
    if not qwen_path.is_absolute() and not args.allow_remote_model:
        args.qwen_model_name = str(resolve_path(args.qwen_model_name))

    resume_path = (
        resolve_checkpoint_path(args.resume)
        if args.resume is not None
        else None
    )
    weights_path = (
        resolve_checkpoint_path(args.weights)
        if args.weights is not None
        else None
    )
    eval_path = (
        resolve_checkpoint_path(args.eval_checkpoint)
        if args.eval_checkpoint is not None
        else None
    )
    args.resume = path_for_record(resume_path)
    args.weights = path_for_record(weights_path)
    args.eval_checkpoint = path_for_record(eval_path)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    device = torch.device(args.device)
    amp_enabled = args.amp and device.type == "cuda"
    if args.amp and not amp_enabled:
        args.amp = False

    set_seed(args.seed, args.deterministic)
    run_dir = select_run_dir(args, resume_path)
    append = args.mode == "resume" and (run_dir / "train.log").exists()
    logger = setup_logger(run_dir, append=append)

    if append:
        logger.info("")
        logger.info(
            "========== Resume session %s ==========",
            datetime.now().isoformat(timespec="seconds"),
        )

    log_section(logger, "Runtime")
    logger.info("script version: %s", SCRIPT_VERSION)
    logger.info("mode: %s", args.mode)
    logger.info("base path: %s", BASE_PATH)
    logger.info("runs path: %s", RUNS_PATH)
    logger.info("run dir: %s", run_dir)
    logger.info("device: %s", device)
    logger.info("torch version: %s", torch.__version__)
    if device.type == "cuda":
        logger.info("CUDA device: %s", torch.cuda.get_device_name(device))
        torch.cuda.empty_cache()
        log_cuda_memory(device, logger, "startup")
    logger.info("AMP enabled: %s", amp_enabled)
    logger.info(
        "precomputed text embeddings: %s",
        args.use_precomputed_text_embeddings,
    )
    if args.use_precomputed_text_embeddings:
        logger.info(
            "text embedding cache: %s",
            args.text_embedding_cache_path,
        )

    log_section(logger, "Dataset")
    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    ) = build_dataloaders(args, device, logger)

    has_validation = val_loader is not None
    monitor_name = choose_monitor_name(args.monitor, has_validation)

    effective_batch_size = args.batch_size * args.accumulation_steps
    lr_scaling_batch_size = (
        effective_batch_size
        if args.scale_lr_with_accumulation
        else args.batch_size
    )
    actual_lr = (
        float(args.lr)
        if args.lr is not None
        else float(args.blr) * lr_scaling_batch_size / 256.0
    )
    resolved_min_lr = resolve_minimum_lr(args, actual_lr)
    if resolved_min_lr > actual_lr:
        raise ValueError(
            f"resolved min_lr ({resolved_min_lr}) cannot exceed peak lr ({actual_lr})"
        )
    updates_per_epoch, total_update_steps, resolved_warmup_steps = (
        resolve_schedule_steps(
            num_batches_per_epoch=len(train_loader),
            total_epochs=args.lr_schedule_epochs,
            accumulation_steps=args.accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            warmup_steps_override=args.warmup_steps_override,
        )
    )
    if resolved_warmup_steps > 0 and args.warmup_init_lr > actual_lr:
        raise ValueError(
            f"warmup_initial_lr ({args.warmup_init_lr}) cannot exceed peak lr ({actual_lr})"
        )

    training_config = build_training_config(
        args,
        actual_lr,
        resolved_min_lr,
        effective_batch_size,
        lr_scaling_batch_size,
        updates_per_epoch,
        total_update_steps,
        resolved_warmup_steps,
        device,
        run_dir,
    )
    training_config["monitor"] = monitor_name
    config_name = (
        f"resume_config_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        if append
        else "config.json"
    )
    save_json(run_dir / config_name, training_config)

    log_section(logger, "Learning rate and optimizer design")
    logger.info("micro batch size: %d", args.batch_size)
    logger.info("gradient accumulation steps: %d", args.accumulation_steps)
    logger.info("effective batch size: %d", effective_batch_size)
    logger.info("lr scaling batch size: %d", lr_scaling_batch_size)
    logger.info("base lr normalized to batch 256: %.8g", args.blr)
    logger.info("actual peak lr: %.8g", actual_lr)
    logger.info("warmup initial lr: %.8g", args.warmup_init_lr)
    logger.info("updates per epoch: %d", updates_per_epoch)
    logger.info("training target epoch: %d", args.epochs)
    logger.info("LR schedule epoch horizon: %d", args.lr_schedule_epochs)
    logger.info("total scheduled optimizer update steps: %d", total_update_steps)
    if args.epochs > args.lr_schedule_epochs:
        logger.info(
            "extension training: epochs %d-%d use the schedule floor lr %.8g",
            args.lr_schedule_epochs + 1,
            args.epochs,
            resolved_min_lr,
        )
    logger.info("warmup ratio: %.6f", args.warmup_ratio)
    logger.info("resolved warmup steps: %d", resolved_warmup_steps)
    logger.info("cosine minimum lr: %.8g", resolved_min_lr)
    logger.info("weight decay: %.8g", args.weight_decay)
    logger.info("AdamW betas: (%.4f, %.4f)", args.beta1, args.beta2)
    logger.info("module lr scales enabled: %s", args.use_module_lr_scales)
    logger.info("best model monitor: %s (lower is better)", monitor_name)

    log_section(logger, "Model")
    model = build_model(args, device)
    set_qwen_eval_if_frozen(model)
    log_model_summary(model, logger)
    log_cuda_memory(device, logger, "after model build")

    optimizer = build_optimizer(model, args, actual_lr)
    log_optimizer_groups(optimizer, logger)
    scaler = build_grad_scaler(amp_enabled)

    # 在任何训练开始前，把优化器学习率设为 schedule 的第 0 个真实更新值。
    adjust_learning_rate(
        optimizer,
        0,
        peak_lr=actual_lr,
        min_lr=resolved_min_lr,
        warmup_init_lr=args.warmup_init_lr,
        warmup_steps=resolved_warmup_steps,
        total_update_steps=total_update_steps,
    )

    scheduler_state_base = {
        "schedule_type": "linear_warmup_half_cosine_by_optimizer_step",
        "peak_lr": actual_lr,
        "min_lr": resolved_min_lr,
        "warmup_init_lr": args.warmup_init_lr,
        "warmup_steps": resolved_warmup_steps,
        "updates_per_epoch": updates_per_epoch,
        "total_update_steps": total_update_steps,
        "gradient_accumulation_steps": args.accumulation_steps,
    }

    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint_path = checkpoints_dir / "last_checkpoint.pt"
    best_checkpoint_path = checkpoints_dir / "best_checkpoint.pt"
    best_weights_path = checkpoints_dir / "best_model_weights.pt"
    best_result_path = run_dir / "best_result.json"
    final_test_path = run_dir / "final_test_result.json"

    start_epoch_index = 0
    global_step = 0
    micro_step = 0
    best_metric = math.inf
    best_epoch = 0
    best_metrics: Optional[dict[str, Any]] = None
    epochs_without_improvement = 0

    if args.mode == "resume":
        assert resume_path is not None
        log_section(logger, "Load full checkpoint")
        checkpoint = load_torch_file(resume_path, map_location=device)
        if not isinstance(checkpoint, dict) or "optimizer_state_dict" not in checkpoint:
            raise KeyError(
                "resume 模式需要包含 optimizer_state_dict 的完整检查点。"
                "若文件只有模型权重，请把 training_mode 改为 weights。"
            )
        check_resume_compatibility(
            checkpoint,
            training_config,
            logger,
            strict_schedule=args.strict_resume_schedule,
        )
        load_model_state(model, checkpoint, logger)
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except ValueError as exc:
            raise ValueError(
                "optimizer 参数组与检查点不兼容。若旧检查点未使用模块差分学习率，"
                "请保持 use_module_lr_scales=False；若只需加载模型，请改用 weights 模式。"
            ) from exc
        if checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        checkpoint_monitor = checkpoint.get("monitor")
        restore_best_state = True
        if checkpoint_monitor and args.monitor == "auto":
            monitor_name = choose_monitor_name(checkpoint_monitor, has_validation)
            training_config["monitor"] = monitor_name
            save_json(run_dir / config_name, training_config)
            logger.info("auto monitor inherited from checkpoint: %s", monitor_name)
        elif checkpoint_monitor and checkpoint_monitor != monitor_name:
            restore_best_state = False
            logger.warning(
                "checkpoint monitor=%s differs from current monitor=%s; "
                "best metric history is reset",
                checkpoint_monitor,
                monitor_name,
            )

        start_epoch_index = int(checkpoint.get("epoch", 0))
        global_step = int(
            checkpoint.get("optimizer_step", checkpoint.get("global_step", 0))
        )
        micro_step = int(
            checkpoint.get(
                "micro_step",
                start_epoch_index * len(train_loader),
            )
        )
        if restore_best_state:
            best_metric = float(checkpoint.get("best_metric", math.inf))
            best_epoch = int(checkpoint.get("best_epoch", 0))
            best_metrics = checkpoint.get("best_metrics")
            epochs_without_improvement = int(
                checkpoint.get("epochs_without_improvement", 0)
            )
        else:
            best_metric = math.inf
            best_epoch = 0
            best_metrics = None
            epochs_without_improvement = 0
        restore_rng_state(checkpoint.get("rng_state"), logger)

        # optimizer_state_dict 会恢复保存时的 lr；这里按恢复后的 optimizer_step
        # 再计算一次，确保 scheduler 与步数严格对齐。
        adjust_learning_rate(
            optimizer,
            global_step,
            peak_lr=actual_lr,
            min_lr=resolved_min_lr,
            warmup_init_lr=args.warmup_init_lr,
            warmup_steps=resolved_warmup_steps,
            total_update_steps=total_update_steps,
        )
        logger.info("resume checkpoint: %s", resume_path)
        logger.info("next epoch: %d", start_epoch_index + 1)
        logger.info("restored optimizer step: %d", global_step)
        logger.info("restored micro step: %d", micro_step)
        logger.info("restored best epoch: %d", best_epoch)
        logger.info("restored best %s: %s", monitor_name, best_metric)

    elif args.mode == "weights":
        assert weights_path is not None
        log_section(logger, "Load model weights only")
        checkpoint = load_torch_file(weights_path, map_location=device)
        load_model_state(model, checkpoint, logger)
        logger.info("weights checkpoint: %s", weights_path)
        logger.info("optimizer, scaler, epoch and best metrics start from scratch")

    elif args.mode == "eval":
        assert eval_path is not None
        log_section(logger, "Evaluation")
        checkpoint = load_torch_file(eval_path, map_location=device)
        load_model_state(model, checkpoint, logger)
        val_metrics = evaluate(model, val_loader, device, args.use_text, amp_enabled)
        test_metrics = evaluate(model, test_loader, device, args.use_text, amp_enabled)
        result = {
            "checkpoint": str(eval_path),
            "validation": val_metrics,
            "test": test_metrics,
            "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        }
        save_json(run_dir / "eval_result.json", result)
        logger.info(metrics_to_log_text("val", val_metrics))
        logger.info(metrics_to_log_text("test", test_metrics))
        logger.info("evaluation result: %s", run_dir / "eval_result.json")
        return

    else:
        log_section(logger, "Initialize from scratch")
        logger.info("training starts from newly initialized trainable parameters")

    append_csv = args.mode == "resume"
    step_file, step_writer = create_csv_writer(
        run_dir / "step_metrics.csv", STEP_FIELDS, append_csv
    )
    epoch_file, epoch_writer = create_csv_writer(
        run_dir / "epoch_metrics.csv", EPOCH_FIELDS, append_csv
    )

    tensorboard_writer = None
    if args.tensorboard:
        if SummaryWriter is None:
            logger.warning("TensorBoard is unavailable; install tensorboard to enable it")
        else:
            tensorboard_writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    training_started = time.time()
    stopped_early = False
    last_completed_epoch = start_epoch_index

    try:
        log_section(logger, "Training")
        if start_epoch_index >= args.epochs:
            logger.warning(
                "checkpoint has completed %d epochs; target epochs=%d, no new training is needed",
                start_epoch_index,
                args.epochs,
            )

        for epoch_index in range(start_epoch_index, args.epochs):
            epoch_started = time.time()
            logger.info("")
            logger.info("========== Epoch %d/%d ==========", epoch_index + 1, args.epochs)

            train_metrics, global_step, micro_step, lr_end = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                epoch_index=epoch_index,
                global_step=global_step,
                micro_step=micro_step,
                args=args,
                peak_lr=actual_lr,
                min_lr=resolved_min_lr,
                warmup_steps=resolved_warmup_steps,
                total_update_steps=total_update_steps,
                logger=logger,
                step_writer=step_writer,
                step_file=step_file,
                tensorboard_writer=tensorboard_writer,
            )

            val_metrics = evaluate(
                model, val_loader, device, args.use_text, amp_enabled
            )
            test_metrics = None
            if args.evaluate_test_each_epoch:
                test_metrics = evaluate(
                    model, test_loader, device, args.use_text, amp_enabled
                )

            epoch_seconds = time.time() - epoch_started
            logger.info(metrics_to_log_text("train", train_metrics))
            logger.info(metrics_to_log_text("val", val_metrics))
            if test_metrics is not None:
                logger.info(metrics_to_log_text("test", test_metrics))
            logger.info("epoch time: %s", timedelta(seconds=int(epoch_seconds)))

            current_metrics = build_epoch_metrics(train_metrics, val_metrics)
            current_metric = get_monitor_value(monitor_name, current_metrics)
            is_best = current_metric < best_metric

            if is_best:
                best_metric = current_metric
                best_epoch = epoch_index + 1
                best_metrics = current_metrics
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            epoch_writer.writerow(
                {
                    "epoch": epoch_index + 1,
                    **{
                        key: "" if value is None else value
                        for key, value in current_metrics.items()
                    },
                    "optimizer_steps_end": global_step,
                    "micro_steps_end": micro_step,
                    "lr_end": get_lr_range(optimizer)[1],
                    "epoch_seconds": epoch_seconds,
                }
            )
            epoch_file.flush()

            write_tensorboard_epoch(
                tensorboard_writer,
                epoch_index + 1,
                train_metrics,
                val_metrics,
                test_metrics,
            )

            checkpoint = build_checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                completed_epoch=epoch_index + 1,
                global_step=global_step,
                micro_step=micro_step,
                scheduler_state=scheduler_state_base,
                best_metric=best_metric,
                best_epoch=best_epoch,
                best_metrics=best_metrics,
                epochs_without_improvement=epochs_without_improvement,
                monitor_name=monitor_name,
                training_config=training_config,
                args=args,
            )
            atomic_torch_save(checkpoint, last_checkpoint_path)
            last_completed_epoch = epoch_index + 1
            logger.info("saved latest checkpoint: %s", last_checkpoint_path)

            if (epoch_index + 1) % args.save_interval == 0:
                periodic = checkpoints_dir / f"checkpoint_epoch_{epoch_index + 1:04d}.pt"
                atomic_torch_save(checkpoint, periodic)
                logger.info("saved periodic checkpoint: %s", periodic)

            if is_best:
                atomic_torch_save(checkpoint, best_checkpoint_path)
                save_lightweight_weights(best_weights_path, checkpoint)
                save_json(
                    best_result_path,
                    {
                        "best_epoch": best_epoch,
                        "monitor": monitor_name,
                        "best_metric": best_metric,
                        "metrics": best_metrics,
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                    },
                )
                logger.info(
                    "NEW BEST | epoch %d | %s %.8f",
                    best_epoch,
                    monitor_name,
                    best_metric,
                )

            patience = args.early_stopping_patience
            if patience > 0 and epochs_without_improvement >= patience:
                logger.info(
                    "early stopping: no improvement for %d epochs",
                    epochs_without_improvement,
                )
                stopped_early = True
                break

        log_section(logger, "Final evaluation with best model")
        if best_checkpoint_path.exists():
            best_checkpoint = load_torch_file(best_checkpoint_path, map_location=device)
            load_model_state(model, best_checkpoint, logger)
        else:
            logger.warning("best checkpoint was not created; evaluating current model")

        final_val_metrics = evaluate(
            model, val_loader, device, args.use_text, amp_enabled
        )
        final_test_metrics = evaluate(
            model, test_loader, device, args.use_text, amp_enabled
        )
        save_json(
            final_test_path,
            {
                "best_epoch": best_epoch,
                "monitor": monitor_name,
                "best_metric": None if math.isinf(best_metric) else best_metric,
                "validation": final_val_metrics,
                "test": final_test_metrics,
                "stopped_early": stopped_early,
                "optimizer_step": global_step,
                "micro_step": micro_step,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        logger.info(metrics_to_log_text("best val", final_val_metrics))
        logger.info(metrics_to_log_text("best test", final_test_metrics))

        figures = plot_metrics_from_csv(run_dir)
        for path in figures:
            logger.info("saved figure: %s", path)

        elapsed = time.time() - training_started
        logger.info("training completed in %s", timedelta(seconds=int(elapsed)))
        logger.info("run directory: %s", run_dir)
        logger.info("best checkpoint: %s", best_checkpoint_path)
        logger.info("best lightweight weights: %s", best_weights_path)
        logger.info("final evaluation: %s", final_test_path)

    except torch.OutOfMemoryError as exc:
        if device.type == "cuda":
            log_cuda_memory(device, logger, "OOM")
            torch.cuda.empty_cache()
        logger.error(
            "CUDA 显存不足：micro-batch=%d，梯度累积=%d，有效 batch=%d。"
            "请在顶部配置区继续减小 batch_size，并按比例增大 "
            "gradient_accumulation_steps。",
            args.batch_size,
            args.accumulation_steps,
            args.batch_size * args.accumulation_steps,
        )
        raise RuntimeError(
            "CUDA 显存不足。可尝试 batch_size=4、"
            "gradient_accumulation_steps=16，并检查其他 GPU 进程。"
        ) from exc

    except KeyboardInterrupt:
        logger.warning("training interrupted by user")
        logger.warning(
            "interrupted checkpoint may contain partial-epoch updates; "
            "exact mid-epoch sampler continuation is not guaranteed"
        )
        emergency = checkpoints_dir / "interrupted_checkpoint.pt"
        checkpoint = build_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            completed_epoch=last_completed_epoch,
            global_step=global_step,
            micro_step=micro_step,
            scheduler_state={
                **scheduler_state_base,
                "interrupted_mid_epoch": True,
            },
            best_metric=best_metric,
            best_epoch=best_epoch,
            best_metrics=best_metrics,
            epochs_without_improvement=epochs_without_improvement,
            monitor_name=monitor_name,
            training_config=training_config,
            args=args,
        )
        atomic_torch_save(checkpoint, emergency)
        logger.warning("saved emergency checkpoint: %s", emergency)
        raise
    finally:
        step_file.close()
        epoch_file.close()
        if tensorboard_writer is not None:
            tensorboard_writer.flush()
            tensorboard_writer.close()


if __name__ == "__main__":
    main()
