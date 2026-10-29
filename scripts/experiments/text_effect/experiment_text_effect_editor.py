#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fusiformer 文本作用量化实验程序（直接运行版）
================================================

用途
----
在不修改正式训练脚本的前提下，单独训练和评估以下文本条件：

1. normal
   晶体结构使用与其正确匹配的文本；正常执行 Qwen、文本投影和文本交叉注意力。

2. bypass
   完全跳过 Qwen、文本投影和文本交叉注意力，作为严格 graph-only 对照。

3. shuffle
   在每个 batch 内循环错配文本，保留文本分支但破坏晶体—文本语义对应关系。

重要前提
--------
models/TextGuidedFusiformer.py 必须已经按照文本消融设计完成修改：

- TextGuidedFusiformer.forward(inputs) 读取 inputs["text_mode"]；
- text_mode="normal" 时保持原始计算；
- text_mode="bypass" 时完全跳过文本编码与文本交叉注意力；
- TextGuidedFusiformerBlock.forward(...) 和
  TextGuidedFusiformerLayer.forward(...) 接收 enable_text_guidance 参数。

本程序会在正式运行前检查模型是否具备严格 bypass 接口。若模型仍是旧版，
程序会直接报错，而不会把“传入 text=None 后使用零向量”误认为 graph-only。

使用方式
--------
1. 将本文件放在 Fusiformer 项目根目录。
2. 只编辑下方“可编辑运行配置”。
3. 直接运行：

       python experiment_text_effect_editor.py

不需要命令行参数。

推荐实验顺序
------------
保持其他参数完全相同，只依次修改：

    text_train_condition = "normal"
    experiment_tag = "normal_seed0"

    text_train_condition = "bypass"
    experiment_tag = "bypass_seed0"

    text_train_condition = "shuffle"
    experiment_tag = "shuffle_seed0"

每次训练完成后，程序会自动对最佳模型执行 normal、bypass、shuffle 三种测试，
并输出总体指标、单样本结果和图像。
"""

from __future__ import annotations

import csv
import inspect
import json
import logging
import math
import os

# 缓解长时间训练中的 CUDA 显存碎片；必须在首次导入/初始化 CUDA 前设置。
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:128",
)
import random
import sys
import time
from collections.abc import Mapping
from contextlib import nullcontext
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from transformers import AutoConfig

from loader.loader import get_dataset
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
            return {key: to_device(item, device) for key, item in value.items()}
        if isinstance(value, list):
            return [to_device(item, device) for item in value]
        if isinstance(value, tuple):
            return tuple(to_device(item, device) for item in value)
        if hasattr(value, "to"):
            return value.to(device, non_blocking=True)
        return value

try:
    from models.TextGuidedFusiformer import TextGuidedFusiformer
except ImportError:
    from TextGuidedFusiformer import TextGuidedFusiformer

try:
    from tools.weight_decay import add_weight_decay as project_add_weight_decay
except ImportError:
    project_add_weight_decay = None

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


# 统一路径基准：无论从哪个工作目录启动，均以当前脚本所在目录为基准。
base_path = Path(__file__).resolve().parents[3]
BASE_PATH = base_path
PROJECT_ROOT = BASE_PATH  # 保留旧名称，兼容已有日志与辅助代码。
RUNS_PATH = BASE_PATH / "runs"

# DEFAULT_DATA_PATH = Path("data") / "my_dataset.json"
# DEFAULT_OUTPUT_ROOT = Path("runs") / "text_effect"
# DEFAULT_QWEN_MODEL = "/root/models/Qwen2.5-1.5B-Instruct"
VALID_TEXT_CONDITIONS = {"normal", "bypass", "shuffle"}


# =============================================================================
# 可编辑运行配置
# =============================================================================
# 修改本区域后直接运行：
#     python experiment_text_effect_editor.py
# 所有相对路径均相对于本脚本所在的项目根目录解析。

# -----------------------
# 1. 启动模式与检查点
# -----------------------
# 可选："scratch"、"resume"、"weights"、"eval"、"plot_only"
training_mode = "scratch"

# resume：完整续训检查点，必须包含 optimizer_state_dict。
resume_checkpoint_path = None

# weights：只加载模型参数，优化器、epoch、最佳指标重新初始化。
weights_checkpoint_path = None

# eval：只评估指定检查点，不训练。
eval_checkpoint_path = None

# 手动指定实验目录：
# - None：scratch/weights/eval 自动在 output_root 下创建目录；
# - resume：None 时自动沿用检查点所在实验目录；
# - plot_only：必须填写已有实验目录。
run_log_dir = None

# 实验输出根目录，与正式训练目录完全分离。
output_root = RUNS_PATH / "text_effect"

# 实验标签。自动创建目录时，目录名为：标签_时间戳。
experiment_tag = "bypass_seed42"


# -----------------------
# 2. 文本作用实验配置
# -----------------------
# 训练条件："normal"、"bypass"、"shuffle"
text_train_condition = "bypass"

# 训练结束后，在同一个最佳模型上执行的推理条件。
text_test_conditions = ["normal", "bypass", "shuffle"]

# 单样本 MAE gain 的中性阈值：
# gain > threshold 视为文本有益；gain < -threshold 视为文本有害。
text_effect_neutral_threshold = 1e-6

# 是否在第一个训练 step 后记录文本相关参数的梯度情况。
verify_text_gradients_first_step = True

# shuffle 使用 batch 内循环移动。batch size=1 时无法错配：
# - "keep"：保留原文本，并在结果中记录 singleton 数量；
# - "error"：遇到单样本 batch 直接报错。
shuffle_singleton_policy = "keep"


# -----------------------
# 3. 数据配置
# -----------------------
data_path = "./data/my_dataset_wiki_success.json"
target_name = "e_form"

# 二选一：
# - 按比例：data_split_counts = None，并设置 data_split_ratio；
# - 按数量：设置 data_split_counts=[训练数, 验证数, 测试数]。
data_split_ratio = [0.6, 0.2, 0.2]
data_split_counts = [2400, 800, 800]


# -----------------------
# 4. 模型配置
# -----------------------
qwen_model_name = "/root/models/Qwen2.5-1.5B-Instruct"
allow_remote_model = False

model_depth = 4
edge_input_dim = 80
triplet_input_dim = 40
embed_dim = 128
num_heads = 4

freeze_qwen = True


# -----------------------
# 5. 训练配置
# -----------------------
# micro-batch：一次真正送入 GPU 的样本数。图规模差异较大，默认使用 8 降低峰值显存。
batch_size = 32

# 梯度累积：8 个 micro-batch 更新一次参数，有效 batch size = 8 × 8 = 64。
gradient_accumulation_steps = 1

total_epochs = 80

# absolute_lr=None 时：实际峰值学习率 = base_lr * 有效batch_size / 256。
absolute_lr = None
base_lr = 1e-3
minimum_lr = 1e-6
warmup_epochs = 5.0

weight_decay = 0.05
adam_beta1 = 0.9
adam_beta2 = 0.999

# None：不裁剪梯度；正数：启用梯度裁剪。
clip_grad_norm = None


# -----------------------
# 6. 运行环境
# -----------------------
device_name = "cuda"
random_seed = 42
num_workers = 4
pin_memory = True
drop_last = False

# CUDA 上默认启用自动混合精度，显著降低激活显存。
use_amp = True
use_deterministic_algorithms = False


# -----------------------
# 7. 日志、模型选择与保存
# -----------------------
# auto：有验证集时使用 val_mae，否则使用 train_mae。
best_model_monitor = "auto"

save_interval = 10
log_interval = 20

# 0 表示关闭提前停止。
early_stopping_patience = 0

enable_tensorboard = True
exclude_frozen_qwen_from_checkpoint = True

# 建议保持 False，仅在训练结束后用最佳模型评估测试集。
evaluate_test_each_epoch = False


STEP_FIELDS = [
    "epoch",
    "step",
    "global_step",
    "progress_epoch",
    "loss",
    "lr",
    "min_group_lr",
    "grad_norm",
]

EPOCH_FIELDS = [
    "epoch",
    "train_mse",
    "train_mae",
    "train_rmse",
    "val_mse",
    "val_mae",
    "val_rmse",
    "lr_end",
    "epoch_seconds",
]


# =============================================================================
# 配置、路径与验证
# =============================================================================
def build_runtime_config() -> SimpleNamespace:
    ratio = None if data_split_counts is not None else list(data_split_ratio)
    counts = None if data_split_counts is None else list(data_split_counts)

    return SimpleNamespace(
        mode=str(training_mode).strip().lower(),
        data_path=data_path,
        target=target_name,
        output_root=output_root,
        run_dir=run_log_dir,
        experiment_tag=str(experiment_tag).strip(),
        resume=resume_checkpoint_path,
        weights=weights_checkpoint_path,
        eval_checkpoint=eval_checkpoint_path,
        split_ratio=ratio,
        split_counts=counts,
        qwen_model_name=qwen_model_name,
        allow_remote_model=bool(allow_remote_model),
        depth=int(model_depth),
        edge_input_dim=int(edge_input_dim),
        triplet_input_dim=int(triplet_input_dim),
        embed_dim=int(embed_dim),
        num_heads=int(num_heads),
        freeze_qwen=bool(freeze_qwen),
        text_train_condition=str(text_train_condition).strip().lower(),
        text_test_conditions=[
            str(item).strip().lower() for item in text_test_conditions
        ],
        text_effect_neutral_threshold=float(text_effect_neutral_threshold),
        verify_text_gradients_first_step=bool(verify_text_gradients_first_step),
        shuffle_singleton_policy=str(shuffle_singleton_policy).strip().lower(),
        batch_size=int(batch_size),
        gradient_accumulation_steps=int(gradient_accumulation_steps),
        epochs=int(total_epochs),
        lr=None if absolute_lr is None else float(absolute_lr),
        blr=float(base_lr),
        min_lr=float(minimum_lr),
        warmup_epochs=float(warmup_epochs),
        weight_decay=float(weight_decay),
        beta1=float(adam_beta1),
        beta2=float(adam_beta2),
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
            f"best_model_monitor={args.monitor!r} 无效，"
            f"可选值为 {sorted(valid_monitors)}"
        )

    if args.text_train_condition not in VALID_TEXT_CONDITIONS:
        raise ValueError(
            f"text_train_condition={args.text_train_condition!r} 无效，"
            f"可选值为 {sorted(VALID_TEXT_CONDITIONS)}"
        )

    if not args.text_test_conditions:
        raise ValueError("text_test_conditions 不能为空")

    duplicated = len(args.text_test_conditions) != len(set(args.text_test_conditions))
    if duplicated:
        raise ValueError("text_test_conditions 不能包含重复条件")

    invalid_test_conditions = [
        item for item in args.text_test_conditions
        if item not in VALID_TEXT_CONDITIONS
    ]
    if invalid_test_conditions:
        raise ValueError(
            f"text_test_conditions 包含无效值：{invalid_test_conditions}"
        )

    if args.shuffle_singleton_policy not in {"keep", "error"}:
        raise ValueError(
            "shuffle_singleton_policy 只能是 'keep' 或 'error'"
        )

    if args.text_effect_neutral_threshold < 0:
        raise ValueError("text_effect_neutral_threshold 不能小于 0")

    if not args.experiment_tag:
        raise ValueError("experiment_tag 不能为空")

    if args.epochs < 1:
        raise ValueError("total_epochs 必须大于等于 1")
    if args.batch_size < 1:
        raise ValueError("batch_size 必须大于等于 1")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps 必须大于等于 1")
    if args.num_workers < 0:
        raise ValueError("num_workers 不能小于 0")
    if args.warmup_epochs < 0:
        raise ValueError("warmup_epochs 不能小于 0")
    if args.warmup_epochs > args.epochs:
        raise ValueError("warmup_epochs 不能大于 total_epochs")
    if args.min_lr < 0:
        raise ValueError("minimum_lr 不能小于 0")
    if args.lr is not None and args.lr <= 0:
        raise ValueError("absolute_lr 必须大于 0，或设置为 None")
    if args.blr <= 0:
        raise ValueError("base_lr 必须大于 0")
    if args.weight_decay < 0:
        raise ValueError("weight_decay 不能小于 0")
    if args.clip_grad is not None and args.clip_grad <= 0:
        raise ValueError("clip_grad_norm 必须为正数，或设置为 None")
    if args.save_interval < 1:
        raise ValueError("save_interval 必须大于等于 1")
    if args.log_interval < 1:
        raise ValueError("log_interval 必须大于等于 1")
    if args.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience 不能小于 0")

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
    """规范化配置路径，并兼容误写成 ("path",) 的单元素元组。"""
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
    解析普通项目路径。

    相对路径始终相对于 BASE_PATH（当前脚本所在目录），而不是终端工作目录。
    对误写为 data/.../<项目目录名>/... 的旧路径会自动去掉重复项。
    """
    normalized = _coerce_path_value(path_value)
    if normalized is None:
        return None

    path = Path(normalized).expanduser()
    if path.is_absolute():
        return path.resolve()

    parts = list(path.parts)
    if BASE_PATH.name in parts:
        # 修复类似 ./data/llm_Fusiformer-master/data/my_dataset.json 的重复路径。
        index = len(parts) - 1 - parts[::-1].index(BASE_PATH.name)
        path = Path(*parts[index + 1:]) if index + 1 < len(parts) else Path()

    return (BASE_PATH / path).resolve()


def resolve_run_path(
    path_value: str | os.PathLike[str] | tuple[Any, ...] | list[Any] | None,
) -> Optional[Path]:
    """
    解析训练产物路径，并强制其位于 BASE_PATH/runs 内。

    支持以下写法：
      - ./runs/text_effect/...
      - text_effect/...
      - 旧项目的绝对路径 .../runs/text_effect/...

    即使从 /root、/tmp 或其他目录启动，所有日志、CSV、图像和检查点都只会
    写入当前脚本目录下的 runs/。
    """
    normalized = _coerce_path_value(path_value)
    if normalized is None:
        return None

    path = Path(normalized).expanduser()
    parts = list(path.parts)

    if "runs" in parts:
        # 只保留最后一个 runs 后面的部分，自动修复重复项目路径或旧绝对路径。
        index = len(parts) - 1 - parts[::-1].index("runs")
        relative_tail = Path(*parts[index + 1:]) if index + 1 < len(parts) else Path()
        resolved = RUNS_PATH / relative_tail
    elif path.is_absolute():
        raise ValueError(
            "训练输出、日志和检查点必须位于当前脚本目录下的 runs/ 中；"
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


def path_for_record(path_value: str | os.PathLike[str] | Path | None) -> Optional[str]:
    """写入 JSON/日志时优先保存相对于 BASE_PATH 的可迁移路径。"""
    if path_value is None:
        return None
    path = Path(path_value).expanduser().resolve()
    try:
        relative = path.relative_to(BASE_PATH.resolve())
        return f"./{relative.as_posix()}"
    except ValueError:
        return str(path)


def make_run_dir(output_root_path: Path, tag: str) -> Path:
    output_root_path.mkdir(parents=True, exist_ok=True)
    safe_tag = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in tag
    ).strip("._")
    if not safe_tag:
        safe_tag = "text_effect"

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = output_root_path / f"{safe_tag}_{timestamp}"
    run_dir_path = base
    suffix = 1
    while run_dir_path.exists():
        run_dir_path = Path(f"{base}-{suffix:02d}")
        suffix += 1
    run_dir_path.mkdir(parents=True, exist_ok=False)
    return run_dir_path


def select_run_dir(
    args: SimpleNamespace,
    resume_path: Optional[Path],
) -> Path:
    if args.run_dir:
        selected = resolve_run_path(args.run_dir)
        assert selected is not None
        selected.mkdir(parents=True, exist_ok=True)
        return selected

    if args.mode == "resume" and resume_path is not None:
        if resume_path.parent.name == "checkpoints":
            selected = resume_path.parent.parent
        else:
            selected = resume_path.parent
        selected.mkdir(parents=True, exist_ok=True)
        return selected

    root = resolve_run_path(args.output_root)
    assert root is not None
    return make_run_dir(root, args.experiment_tag)


# =============================================================================
# 日志、JSON、CSV 与绘图
# =============================================================================
def setup_logger(run_dir: Path, append: bool) -> logging.Logger:
    logger = logging.getLogger("fusiformer_text_effect")
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
    with temp.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    os.replace(temp, path)


def create_csv_writer(
    path: Path,
    fieldnames: list[str],
    append: bool,
) -> tuple[Any, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    should_append = append and path.exists() and path.stat().st_size > 0
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


def plot_training_metrics(run_dir: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    saved: list[str] = []
    epoch_csv = run_dir / "epoch_metrics.csv"
    step_csv = run_dir / "step_metrics.csv"

    if epoch_csv.exists():
        with epoch_csv.open("r", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        if rows:
            x = [int(row["epoch"]) for row in rows]
            groups = [
                ("mse_curve.png", "MSE", ["train_mse", "val_mse"]),
                ("mae_curve.png", "MAE", ["train_mae", "val_mae"]),
                ("rmse_curve.png", "RMSE", ["train_rmse", "val_rmse"]),
            ]
            for filename, ylabel, keys in groups:
                plt.figure()
                plotted = False
                for key in keys:
                    values = [safe_float(row.get(key)) for row in rows]
                    if any(value is not None for value in values):
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
                    saved.append(path_for_record(output) or str(output))
                plt.close()

            lr_end = [safe_float(row.get("lr_end")) for row in rows]
            if any(value is not None for value in lr_end):
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
                saved.append(path_for_record(output) or str(output))

    if step_csv.exists():
        with step_csv.open("r", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        if rows:
            steps = [int(row["global_step"]) for row in rows]
            losses = [safe_float(row.get("loss")) for row in rows]
            lrs = [safe_float(row.get("lr")) for row in rows]

            plt.figure()
            plt.plot(steps, losses, label="step_loss")
            plt.xlabel("Global step")
            plt.ylabel("Loss")
            plt.title("Step loss")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            output = run_dir / "step_loss_curve.png"
            plt.savefig(output, dpi=200)
            plt.close()
            saved.append(path_for_record(output) or str(output))

            plt.figure()
            plt.plot(steps, lrs, label="learning_rate")
            plt.xlabel("Global step")
            plt.ylabel("Learning rate")
            plt.title("Warmup and cosine learning-rate schedule")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            output = run_dir / "step_lr_curve.png"
            plt.savefig(output, dpi=200)
            plt.close()
            saved.append(path_for_record(output) or str(output))

    return saved


def plot_text_effect_results(result_dir: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    metrics_path = result_dir / "text_condition_metrics.json"
    samples_path = result_dir / "text_condition_samples.csv"
    if not metrics_path.exists():
        return []

    with metrics_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)

    conditions_data = summary.get("conditions", {})
    conditions = list(conditions_data.keys())
    saved: list[str] = []

    if conditions:
        mae_values = [conditions_data[item]["mae"] for item in conditions]
        rmse_values = [conditions_data[item]["rmse"] for item in conditions]

        plt.figure()
        positions = np.arange(len(conditions))
        width = 0.35
        plt.bar(positions - width / 2, mae_values, width, label="MAE")
        plt.bar(positions + width / 2, rmse_values, width, label="RMSE")
        plt.xticks(positions, conditions)
        plt.ylabel("Metric")
        plt.title("Text-condition metrics")
        plt.legend()
        plt.tight_layout()
        output = result_dir / "text_condition_metrics.png"
        plt.savefig(output, dpi=200)
        plt.close()
        saved.append(path_for_record(output) or str(output))

    if not samples_path.exists():
        return saved

    with samples_path.open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        return saved

    if all(
        key in rows[0]
        for key in (
            "absolute_error_normal",
            "absolute_error_bypass",
            "mae_gain_from_text",
            "prediction_shift",
        )
    ):
        error_normal = [float(row["absolute_error_normal"]) for row in rows]
        error_bypass = [float(row["absolute_error_bypass"]) for row in rows]
        gain = [float(row["mae_gain_from_text"]) for row in rows]
        shift = [float(row["prediction_shift"]) for row in rows]

        max_error = max(error_normal + error_bypass + [1e-12])
        plt.figure()
        plt.scatter(error_bypass, error_normal, s=16, alpha=0.7)
        plt.plot([0, max_error], [0, max_error], linestyle="--")
        plt.xlabel("Absolute error: bypass")
        plt.ylabel("Absolute error: normal")
        plt.title("Correct text vs strict graph-only")
        plt.tight_layout()
        output = result_dir / "normal_vs_bypass_error_scatter.png"
        plt.savefig(output, dpi=200)
        plt.close()
        saved.append(path_for_record(output) or str(output))

        plt.figure()
        plt.hist(gain, bins=40)
        plt.axvline(0.0, linestyle="--")
        plt.xlabel("MAE gain from text")
        plt.ylabel("Count")
        plt.title("Distribution of per-element text gain")
        plt.tight_layout()
        output = result_dir / "text_gain_histogram.png"
        plt.savefig(output, dpi=200)
        plt.close()
        saved.append(path_for_record(output) or str(output))

        plt.figure()
        plt.scatter(shift, gain, s=16, alpha=0.7)
        plt.axhline(0.0, linestyle="--")
        plt.xlabel("Prediction shift |normal - bypass|")
        plt.ylabel("MAE gain from text")
        plt.title("Text dependence and usefulness")
        plt.tight_layout()
        output = result_dir / "dependency_vs_gain.png"
        plt.savefig(output, dpi=200)
        plt.close()
        saved.append(path_for_record(output) or str(output))

    return saved


def plot_all_available(run_dir: Path) -> list[str]:
    saved = plot_training_metrics(run_dir)
    for subdirectory_name in ("text_effect_validation", "text_effect_test"):
        subdirectory = run_dir / subdirectory_name
        if subdirectory.exists():
            saved.extend(plot_text_effect_results(subdirectory))
    return saved


# =============================================================================
# 随机性、优化器与学习率
# =============================================================================
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


def restore_rng_state(
    state: Optional[dict[str, Any]],
    logger: logging.Logger,
) -> None:
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
    weight_decay_value: float,
) -> list[dict[str, Any]]:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)

    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay_value},
    ]


def build_optimizer(
    model: torch.nn.Module,
    args: SimpleNamespace,
    actual_lr: float,
) -> torch.optim.AdamW:
    if project_add_weight_decay is not None:
        parameter_groups = project_add_weight_decay(model, args.weight_decay)
    else:
        parameter_groups = fallback_add_weight_decay(model, args.weight_decay)

    return torch.optim.AdamW(
        parameter_groups,
        lr=actual_lr,
        betas=(args.beta1, args.beta2),
    )


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    progress_epoch: float,
    *,
    base_lr_value: float,
    min_lr_value: float,
    warmup_epochs_value: float,
    total_epochs_value: int,
) -> float:
    if warmup_epochs_value > 0 and progress_epoch < warmup_epochs_value:
        learning_rate = base_lr_value * progress_epoch / warmup_epochs_value
    elif total_epochs_value <= warmup_epochs_value:
        learning_rate = base_lr_value
    else:
        cosine_progress = (progress_epoch - warmup_epochs_value) / (
            total_epochs_value - warmup_epochs_value
        )
        cosine_progress = min(max(cosine_progress, 0.0), 1.0)
        learning_rate = min_lr_value + (base_lr_value - min_lr_value) * 0.5 * (
            1.0 + math.cos(math.pi * cosine_progress)
        )

    for group in optimizer.param_groups:
        scale = float(group.get("lr_scale", 1.0))
        group["lr"] = learning_rate * scale
    return learning_rate


def get_lr_range(
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    values = [float(group["lr"]) for group in optimizer.param_groups]
    return min(values), max(values)


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


def log_cuda_memory(
    device: torch.device,
    logger: logging.Logger,
    stage: str,
) -> None:
    """记录当前进程和整张 GPU 的显存状态，便于提前发现其他进程占用。"""
    if device.type != "cuda":
        return
    with torch.cuda.device(device):
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
    gib = 1024 ** 3
    logger.info(
        "CUDA memory [%s] | free %.2f GiB / total %.2f GiB | "
        "process allocated %.2f GiB | process reserved %.2f GiB",
        stage,
        free_bytes / gib,
        total_bytes / gib,
        allocated / gib,
        reserved / gib,
    )


def move_optimizer_state_to_cpu(
    optimizer: torch.optim.Optimizer,
    logger: logging.Logger,
) -> None:
    """
    训练结束后把 AdamW 动量状态移到 CPU。

    最终验证/测试不再需要优化器，把 exp_avg/exp_avg_sq 留在 GPU 会额外占用
    数 GB 显存，尤其容易在多条件评估时触发 OOM。
    """
    moved_tensors = 0
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value) and value.device.type == "cuda":
                state[key] = value.detach().cpu()
                moved_tensors += 1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info(
        "moved optimizer state tensors to CPU before final evaluation: %d",
        moved_tensors,
    )


# =============================================================================
# 文本条件控制
# =============================================================================
def _find_batch_size_and_device(
    text_tokens: Mapping[str, Any],
) -> tuple[int, torch.device]:
    for value in text_tokens.values():
        if torch.is_tensor(value) and value.ndim >= 1:
            return int(value.shape[0]), value.device
    raise ValueError("无法从 inputs['text'] 中确定 batch size")


def cyclic_shift_text_tokens(
    text_tokens: Any,
    *,
    singleton_policy: str,
) -> tuple[Any, bool]:
    """
    对 tokenizer 输出沿 batch 维循环移动一位。

    返回
    ----
    shifted_text, singleton_kept
    """
    if text_tokens is None:
        raise ValueError(
            "shuffle 条件需要 inputs['text']，但当前 batch 的文本为 None"
        )
    if not isinstance(text_tokens, Mapping):
        raise TypeError(
            "shuffle 条件要求 inputs['text'] 是 tokenizer 返回的映射对象"
        )

    batch_size_value, device = _find_batch_size_and_device(text_tokens)
    if batch_size_value <= 1:
        if singleton_policy == "error":
            raise RuntimeError(
                "shuffle 条件遇到 batch size=1，无法构造错配文本。"
                "请增大 batch_size、调整数据划分，或将 "
                "shuffle_singleton_policy 设置为 'keep'。"
            )
        return dict(text_tokens), True

    permutation = torch.roll(
        torch.arange(batch_size_value, device=device),
        shifts=1,
        dims=0,
    )

    shifted: dict[str, Any] = {}
    for key, value in text_tokens.items():
        if (
            torch.is_tensor(value)
            and value.ndim >= 1
            and int(value.shape[0]) == batch_size_value
        ):
            shifted[key] = value.index_select(0, permutation)
        else:
            shifted[key] = value
    return shifted, False


def apply_text_condition(
    inputs: Any,
    condition: str,
    *,
    singleton_policy: str,
) -> tuple[Any, bool]:
    """
    构造文本实验输入，不深拷贝 DGL graph。

    返回
    ----
    prepared_inputs, singleton_shuffle_kept
    """
    if not isinstance(inputs, dict):
        raise TypeError("模型输入必须是字典")

    condition = str(condition).strip().lower()
    if condition not in VALID_TEXT_CONDITIONS:
        raise ValueError(f"不支持的文本条件：{condition!r}")

    prepared = dict(inputs)
    singleton_kept = False

    if condition == "normal":
        if prepared.get("text") is None:
            raise ValueError("normal 条件需要非空 inputs['text']")
        prepared["text_mode"] = "normal"

    elif condition == "bypass":
        prepared["text"] = None
        prepared["text_mode"] = "bypass"

    elif condition == "shuffle":
        shifted, singleton_kept = cyclic_shift_text_tokens(
            prepared.get("text"),
            singleton_policy=singleton_policy,
        )
        prepared["text"] = shifted
        prepared["text_mode"] = "normal"

    return prepared, singleton_kept


def verify_strict_bypass_support(model: torch.nn.Module) -> None:
    """防止旧模型静默忽略 text_mode，把零文本误当严格 graph-only。"""
    module = model.module if hasattr(model, "module") else model

    blocks = getattr(module, "blocks", None)
    if blocks is None or len(blocks) == 0:
        raise RuntimeError("模型中未找到 blocks，无法验证文本 bypass 接口")

    first_block = blocks[0]
    block_signature = inspect.signature(first_block.forward)
    if "enable_text_guidance" not in block_signature.parameters:
        raise RuntimeError(
            "当前 models/TextGuidedFusiformer.py 仍未支持严格文本旁路："
            "TextGuidedFusiformerBlock.forward 缺少 enable_text_guidance 参数。"
            "请先完成模型文件修改，再运行本实验脚本。"
        )

    for layer_name in ("g_update", "lg_update"):
        layer = getattr(first_block, layer_name, None)
        if layer is None:
            raise RuntimeError(f"模型 block 中缺少 {layer_name}")
        layer_signature = inspect.signature(layer.forward)
        if "enable_text_guidance" not in layer_signature.parameters:
            raise RuntimeError(
                "当前 models/TextGuidedFusiformer.py 仍未支持严格文本旁路："
                f"{layer_name}.forward 缺少 enable_text_guidance 参数。"
            )

    try:
        top_source = inspect.getsource(module.forward)
    except (OSError, TypeError):
        top_source = ""
    if top_source and "text_mode" not in top_source:
        raise RuntimeError(
            "顶层 TextGuidedFusiformer.forward 未读取 text_mode。"
            "请先完成模型文件修改。"
        )


def log_text_module_gradients(
    model: torch.nn.Module,
    logger: logging.Logger,
    condition: str,
) -> None:
    module = model.module if hasattr(model, "module") else model
    text_parameters = 0
    parameters_with_gradient = 0
    parameters_with_nonzero_gradient = 0
    total_gradient_norm = 0.0

    text_markers = (
        "text_proj",
        "text_cross_attn",
        "norm_text_node",
        "norm_text_edge",
    )

    for name, parameter in module.named_parameters():
        if not any(marker in name for marker in text_markers):
            continue
        text_parameters += 1
        if parameter.grad is None:
            continue
        parameters_with_gradient += 1
        gradient_norm = float(parameter.grad.detach().float().norm().item())
        total_gradient_norm += gradient_norm
        if gradient_norm > 0:
            parameters_with_nonzero_gradient += 1

    logger.info(
        "text gradient check | condition=%s | parameters=%d | "
        "grad_not_none=%d | grad_nonzero=%d | summed_grad_norm=%.8g",
        condition,
        text_parameters,
        parameters_with_gradient,
        parameters_with_nonzero_gradient,
        total_gradient_norm,
    )

    if condition == "bypass" and parameters_with_nonzero_gradient > 0:
        raise RuntimeError(
            "bypass 条件下文本相关参数出现非零梯度，说明文本分支没有被完全旁路"
        )


# =============================================================================
# 模型、数据与指标
# =============================================================================
def normalize_targets(targets: Any, device: torch.device) -> torch.Tensor:
    if isinstance(targets, list):
        targets = torch.cat(targets, dim=-1)
    elif isinstance(targets, tuple):
        targets = torch.cat(list(targets), dim=-1)
    return targets.to(device, non_blocking=True)


def normalize_outputs(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, dict):
        if "prediction" not in outputs:
            raise KeyError("模型输出字典中缺少 'prediction'")
        return outputs["prediction"]
    if isinstance(outputs, (list, tuple)):
        return torch.cat(list(outputs), dim=-1)
    return outputs


def set_qwen_eval_if_frozen(model: torch.nn.Module) -> None:
    module = model.module if hasattr(model, "module") else model
    qwen = getattr(module, "qwen", None)
    if qwen is None:
        return

    qwen_trainable = any(parameter.requires_grad for parameter in qwen.parameters())
    if not qwen_trainable:
        qwen.eval()


def build_model(
    args: SimpleNamespace,
    device: torch.device,
) -> torch.nn.Module:
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
    )

    # 兼容旧模型类未保存 freeze_qwen 属性的情况。
    model.freeze_qwen = args.freeze_qwen
    model = model.to(device)
    set_qwen_eval_if_frozen(model)
    return model


def log_model_summary(
    model: torch.nn.Module,
    logger: logging.Logger,
) -> None:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    logger.info("model total params: %d (%.2f M)", total, total / 1e6)
    logger.info(
        "model trainable params: %d (%.2f M)",
        trainable,
        trainable / 1e6,
    )

    module = model.module if hasattr(model, "module") else model
    qwen = getattr(module, "qwen", None)
    if qwen is not None:
        qwen_total = sum(parameter.numel() for parameter in qwen.parameters())
        qwen_trainable = sum(
            parameter.numel()
            for parameter in qwen.parameters()
            if parameter.requires_grad
        )
        logger.info("qwen total params: %d", qwen_total)
        logger.info("qwen trainable params: %d", qwen_trainable)
        logger.info("qwen frozen: %s", qwen_trainable == 0)


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
            GraphCollecting(["graph", "line_graph"], [args.target]),
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
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)

    evaluation_generator = torch.Generator()
    evaluation_generator.manual_seed(args.seed)

    common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_mem and device.type == "cuda",
        "worker_init_fn": seed_worker,
    }
    if args.num_workers > 0:
        common["persistent_workers"] = True

    effective_drop_last = args.drop_last and len(train_dataset) >= args.batch_size
    if args.drop_last and not effective_drop_last:
        logger.warning(
            "train dataset is smaller than batch size; drop_last is disabled"
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=effective_drop_last,
        collate_fn=train_dataset.collect(),
        generator=train_generator,
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
            generator=evaluation_generator,
            **common,
        )

    val_loader = make_eval_loader(val_dataset)
    test_loader = make_eval_loader(test_dataset)

    logger.info("train samples: %d", len(train_dataset))
    logger.info(
        "validation samples: %d",
        0 if val_dataset is None else len(val_dataset),
    )
    logger.info(
        "test samples: %d",
        0 if test_dataset is None else len(test_dataset),
    )
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


def finalize_metrics(
    accumulator: dict[str, float],
) -> dict[str, float]:
    count = max(accumulator["element_count"], 1.0)
    mse_value = accumulator["squared_error_sum"] / count
    mae_value = accumulator["absolute_error_sum"] / count
    return {
        "mse": mse_value,
        "mae": mae_value,
        "rmse": math.sqrt(mse_value),
    }


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    text_condition: str,
    amp_enabled: bool,
    singleton_policy: str,
) -> Optional[dict[str, float]]:
    if loader is None:
        return None

    model.eval()
    set_qwen_eval_if_frozen(model)
    accumulator = empty_metric_accumulator()
    singleton_batches = 0

    for inputs, targets in loader:
        inputs = to_device(inputs, device)
        targets = normalize_targets(targets, device)
        prepared_inputs, singleton_kept = apply_text_condition(
            inputs,
            text_condition,
            singleton_policy=singleton_policy,
        )
        singleton_batches += int(singleton_kept)

        with autocast_context(device, amp_enabled):
            outputs = normalize_outputs(model(prepared_inputs))
        update_metric_accumulator(accumulator, outputs, targets)

    result = finalize_metrics(accumulator)
    result["shuffle_singleton_batches"] = float(singleton_batches)
    return result


def train_one_epoch(
    *,
    model: torch.nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    epoch_index: int,
    global_step: int,
    args: SimpleNamespace,
    actual_lr: float,
    logger: logging.Logger,
    step_writer: csv.DictWriter,
    step_file: Any,
    tensorboard_writer: Any,
) -> tuple[dict[str, float], int, float]:
    """
    训练一个 epoch。

    batch_size 是单次前向传播的 micro-batch；通过梯度累积保持较大的有效
    batch size，从而降低图注意力和 line-graph 分支的峰值显存。
    """
    model.train()
    set_qwen_eval_if_frozen(model)

    criterion = torch.nn.MSELoss()
    accumulator = empty_metric_accumulator()
    num_steps = len(loader)
    accumulation_steps = args.gradient_accumulation_steps
    last_lr = actual_lr
    singleton_batches = 0
    first_gradient_check_done = False

    optimizer.zero_grad(set_to_none=True)

    for step, (inputs, targets) in enumerate(loader):
        progress_epoch = epoch_index + step / num_steps
        last_lr = adjust_learning_rate(
            optimizer,
            progress_epoch,
            base_lr_value=actual_lr,
            min_lr_value=args.min_lr,
            warmup_epochs_value=args.warmup_epochs,
            total_epochs_value=args.epochs,
        )

        inputs = to_device(inputs, device)
        targets = normalize_targets(targets, device)
        prepared_inputs, singleton_kept = apply_text_condition(
            inputs,
            args.text_train_condition,
            singleton_policy=args.shuffle_singleton_policy,
        )
        singleton_batches += int(singleton_kept)

        window_start = (step // accumulation_steps) * accumulation_steps
        window_size = min(accumulation_steps, num_steps - window_start)
        accumulation_index = step - window_start + 1
        should_update = accumulation_index == window_size

        try:
            with autocast_context(device, args.amp and device.type == "cuda"):
                outputs = normalize_outputs(model(prepared_inputs))
                loss = criterion(outputs, targets)
        except torch.cuda.OutOfMemoryError as exc:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA 显存不足。当前 micro-batch="
                f"{args.batch_size}，梯度累积={accumulation_steps}。"
                "请先确认没有其他进程占满 GPU；仍不足时继续减小 batch_size，"
                "并按比例增大 gradient_accumulation_steps。"
            ) from exc

        loss_value = float(loss.detach().item())
        if not math.isfinite(loss_value):
            raise RuntimeError(
                f"invalid loss at epoch={epoch_index + 1}, "
                f"step={step}: {loss_value}"
            )

        # 最后一个不足 accumulation_steps 的窗口使用实际窗口大小归一化，
        # 避免该窗口的梯度被额外缩小。
        backward_loss = loss / float(window_size)
        scaler.scale(backward_loss).backward()

        grad_norm_value: Optional[float] = None
        if should_update:
            need_gradient_check = (
                args.verify_text_gradients_first_step
                and epoch_index == 0
                and not first_gradient_check_done
            )
            need_unscale = (
                args.amp
                and device.type == "cuda"
                and (need_gradient_check or args.clip_grad is not None)
            )
            if need_unscale:
                scaler.unscale_(optimizer)

            if need_gradient_check:
                log_text_module_gradients(
                    model,
                    logger,
                    args.text_train_condition,
                )
                first_gradient_check_done = True

            if args.clip_grad is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ],
                    max_norm=args.clip_grad,
                )
                grad_norm_value = float(grad_norm.detach().item())

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        update_metric_accumulator(accumulator, outputs, targets)
        global_step += 1
        min_group_lr, max_group_lr = get_lr_range(optimizer)

        step_writer.writerow(
            {
                "epoch": epoch_index + 1,
                "step": step,
                "global_step": global_step,
                "progress_epoch": progress_epoch,
                "loss": loss_value,
                "lr": max_group_lr,
                "min_group_lr": min_group_lr,
                "grad_norm": "" if grad_norm_value is None else grad_norm_value,
            }
        )
        step_file.flush()

        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar(
                "train/step_loss", loss_value, global_step
            )
            tensorboard_writer.add_scalar(
                "train/lr", max_group_lr, global_step
            )

        if step % args.log_interval == 0 or step == num_steps - 1:
            logger.info(
                "epoch %03d | step %04d/%04d | global_step %07d | "
                "condition %s | loss %.6f | lr %.8g | accumulation %d/%d%s",
                epoch_index + 1,
                step,
                num_steps - 1,
                global_step,
                args.text_train_condition,
                loss_value,
                max_group_lr,
                accumulation_index,
                window_size,
                " | optimizer_step" if should_update else "",
            )

        # 显式删除大图相关局部引用，降低下一 micro-batch 前的引用滞留风险。
        del backward_loss, loss, outputs, prepared_inputs, targets, inputs

    metrics = finalize_metrics(accumulator)
    metrics["shuffle_singleton_batches"] = float(singleton_batches)
    return metrics, global_step, last_lr


# =============================================================================
# 多条件反事实测试
# =============================================================================
@torch.inference_mode()
def evaluate_text_conditions(
    *,
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    conditions: list[str],
    amp_enabled: bool,
    output_dir: Path,
    neutral_threshold: float,
    singleton_policy: str,
    split_name: str,
) -> Optional[dict[str, Any]]:
    if loader is None:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    set_qwen_eval_if_frozen(model)

    target_chunks: list[torch.Tensor] = []
    prediction_chunks: dict[str, list[torch.Tensor]] = {
        condition: [] for condition in conditions
    }
    singleton_counts = {condition: 0 for condition in conditions}

    for inputs, targets in loader:
        inputs = to_device(inputs, device)
        targets = normalize_targets(targets, device)
        target_chunks.append(targets.detach().float().cpu().reshape(-1))

        for condition in conditions:
            prepared_inputs, singleton_kept = apply_text_condition(
                inputs,
                condition,
                singleton_policy=singleton_policy,
            )
            singleton_counts[condition] += int(singleton_kept)
            with autocast_context(device, amp_enabled):
                outputs = normalize_outputs(model(prepared_inputs))
            prediction_chunks[condition].append(
                outputs.detach().float().cpu().reshape(-1)
            )

    targets_all = torch.cat(target_chunks, dim=0)
    predictions_all = {
        condition: torch.cat(chunks, dim=0)
        for condition, chunks in prediction_chunks.items()
    }

    condition_metrics: dict[str, dict[str, float]] = {}
    for condition, predictions in predictions_all.items():
        error = predictions - targets_all
        mse_value = float(torch.mean(error * error).item())
        mae_value = float(torch.mean(torch.abs(error)).item())
        condition_metrics[condition] = {
            "mse": mse_value,
            "mae": mae_value,
            "rmse": math.sqrt(mse_value),
            "shuffle_singleton_batches": float(singleton_counts[condition]),
        }

    summary: dict[str, Any] = {
        "split": split_name,
        "num_elements": int(targets_all.numel()),
        "conditions": condition_metrics,
        "neutral_threshold": neutral_threshold,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }

    target_std = float(targets_all.std(unbiased=False).item())
    target_std = max(target_std, 1e-12)

    if "normal" in predictions_all and "bypass" in predictions_all:
        normal_prediction = predictions_all["normal"]
        bypass_prediction = predictions_all["bypass"]
        normal_error = torch.abs(normal_prediction - targets_all)
        bypass_error = torch.abs(bypass_prediction - targets_all)
        gain = bypass_error - normal_error
        shift = torch.abs(normal_prediction - bypass_prediction)

        mae_normal = condition_metrics["normal"]["mae"]
        mae_bypass = condition_metrics["bypass"]["mae"]

        benefit_rate = float(
            torch.mean((gain > neutral_threshold).float()).item()
        )
        harm_rate = float(
            torch.mean((gain < -neutral_threshold).float()).item()
        )
        neutral_rate = float(
            torch.mean((torch.abs(gain) <= neutral_threshold).float()).item()
        )

        summary["normal_vs_bypass"] = {
            "absolute_mae_gain": mae_bypass - mae_normal,
            "relative_text_gain": (
                (mae_bypass - mae_normal) / max(mae_bypass, 1e-12)
            ),
            "mean_prediction_shift": float(shift.mean().item()),
            "median_prediction_shift": float(shift.median().item()),
            "mean_text_dependency_index": float(
                (shift / target_std).mean().item()
            ),
            "benefit_rate": benefit_rate,
            "harm_rate": harm_rate,
            "neutral_rate": neutral_rate,
        }

    if "normal" in predictions_all and "shuffle" in predictions_all:
        mae_normal = condition_metrics["normal"]["mae"]
        mae_shuffle = condition_metrics["shuffle"]["mae"]
        summary["normal_vs_shuffle"] = {
            "absolute_semantic_mae_gain": mae_shuffle - mae_normal,
            "relative_semantic_gain": (
                (mae_shuffle - mae_normal) / max(mae_shuffle, 1e-12)
            ),
            "mean_prediction_shift": float(
                torch.abs(
                    predictions_all["normal"] - predictions_all["shuffle"]
                ).mean().item()
            ),
        }

    save_json(output_dir / "text_condition_metrics.json", summary)

    fieldnames = ["element_index", "target"]
    for condition in conditions:
        fieldnames.extend(
            [
                f"prediction_{condition}",
                f"absolute_error_{condition}",
            ]
        )

    if "normal" in predictions_all and "bypass" in predictions_all:
        fieldnames.extend(
            [
                "prediction_shift",
                "signed_prediction_shift",
                "mae_gain_from_text",
                "text_dependency_index",
                "text_effect_class",
            ]
        )

    if "normal" in predictions_all and "shuffle" in predictions_all:
        fieldnames.extend(
            [
                "semantic_prediction_shift",
                "mae_gain_correct_vs_shuffle",
            ]
        )

    sample_path = output_dir / "text_condition_samples.csv"
    with sample_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for index in range(targets_all.numel()):
            target_value = float(targets_all[index].item())
            row: dict[str, Any] = {
                "element_index": index,
                "target": target_value,
            }

            for condition in conditions:
                prediction_value = float(predictions_all[condition][index].item())
                row[f"prediction_{condition}"] = prediction_value
                row[f"absolute_error_{condition}"] = abs(
                    prediction_value - target_value
                )

            if "normal" in predictions_all and "bypass" in predictions_all:
                normal_value = float(predictions_all["normal"][index].item())
                bypass_value = float(predictions_all["bypass"][index].item())
                normal_error_value = abs(normal_value - target_value)
                bypass_error_value = abs(bypass_value - target_value)
                gain_value = bypass_error_value - normal_error_value
                shift_value = abs(normal_value - bypass_value)

                if gain_value > neutral_threshold:
                    effect_class = "benefit"
                elif gain_value < -neutral_threshold:
                    effect_class = "harm"
                else:
                    effect_class = "neutral"

                row["prediction_shift"] = shift_value
                row["signed_prediction_shift"] = normal_value - bypass_value
                row["mae_gain_from_text"] = gain_value
                row["text_dependency_index"] = shift_value / target_std
                row["text_effect_class"] = effect_class

            if "normal" in predictions_all and "shuffle" in predictions_all:
                normal_value = float(predictions_all["normal"][index].item())
                shuffle_value = float(predictions_all["shuffle"][index].item())
                row["semantic_prediction_shift"] = abs(
                    normal_value - shuffle_value
                )
                row["mae_gain_correct_vs_shuffle"] = (
                    abs(shuffle_value - target_value)
                    - abs(normal_value - target_value)
                )

            writer.writerow(row)

    figures = plot_text_effect_results(output_dir)
    summary["figures"] = figures
    save_json(output_dir / "text_condition_metrics.json", summary)
    return summary


# =============================================================================
# 检查点
# =============================================================================
def atomic_torch_save(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, temp)
    os.replace(temp, path)


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
    full_state = module.state_dict()
    excluded_prefixes: list[str] = []

    qwen = getattr(module, "qwen", None)
    qwen_is_frozen = (
        qwen is not None
        and not any(parameter.requires_grad for parameter in qwen.parameters())
    )

    if exclude_frozen_qwen and qwen_is_frozen:
        excluded_prefixes.append("qwen.")

    state: dict[str, torch.Tensor] = {}
    for name, tensor in full_state.items():
        if any(name.startswith(prefix) for prefix in excluded_prefixes):
            continue
        state[name] = tensor.detach().cpu()
    return state, excluded_prefixes


def extract_model_state(checkpoint: Any) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint 必须是字典")

    if "model_state_dict" in checkpoint:
        model_state = checkpoint["model_state_dict"]
        excluded_prefixes = list(checkpoint.get("excluded_model_prefixes", []))
    else:
        model_state = checkpoint
        excluded_prefixes = []

    if model_state and all(name.startswith("module.") for name in model_state):
        model_state = {
            name[len("module."):]: value
            for name, value in model_state.items()
        }

    return model_state, excluded_prefixes


def load_model_state(
    model: torch.nn.Module,
    checkpoint: Any,
    logger: logging.Logger,
) -> None:
    module = model.module if hasattr(model, "module") else model
    model_state, excluded_prefixes = extract_model_state(checkpoint)

    incompatible = module.load_state_dict(model_state, strict=False)
    unexpected_keys = list(incompatible.unexpected_keys)
    missing_keys = [
        name
        for name in incompatible.missing_keys
        if not any(name.startswith(prefix) for prefix in excluded_prefixes)
    ]

    if unexpected_keys or missing_keys:
        raise RuntimeError(
            "模型参数加载不完整。"
            f" missing_keys={missing_keys[:20]},"
            f" unexpected_keys={unexpected_keys[:20]}"
        )

    logger.info("loaded model parameters: %d tensors", len(model_state))
    if excluded_prefixes:
        logger.info(
            "checkpoint excluded frozen prefixes: %s",
            excluded_prefixes,
        )


def build_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    completed_epoch: int,
    global_step: int,
    best_metric: float,
    best_epoch: int,
    best_metrics: Optional[dict[str, Any]],
    epochs_without_improvement: int,
    monitor_name: str,
    training_config: dict[str, Any],
    args: SimpleNamespace,
) -> dict[str, Any]:
    model_state, excluded_prefixes = get_checkpoint_model_state(
        model,
        args.exclude_frozen_qwen_from_checkpoint,
    )
    return {
        "format_version": 3,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "epoch": completed_epoch,
        "global_step": global_step,
        "model_state_dict": model_state,
        "excluded_model_prefixes": excluded_prefixes,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "epochs_without_improvement": epochs_without_improvement,
        "monitor": monitor_name,
        "rng_state": capture_rng_state(),
        "training_config": training_config,
        "run_config": training_config,
        "text_train_condition": args.text_train_condition,
        "text_test_conditions": list(args.text_test_conditions),
    }


def save_best_model_weights(
    path: Path,
    checkpoint: dict[str, Any],
) -> None:
    weights_object = {
        "format_version": checkpoint.get("format_version", 3),
        "saved_at": checkpoint.get("saved_at"),
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "model_state_dict": checkpoint["model_state_dict"],
        "excluded_model_prefixes": checkpoint.get(
            "excluded_model_prefixes", []
        ),
        "best_metric": checkpoint.get("best_metric"),
        "best_epoch": checkpoint.get("best_epoch"),
        "best_metrics": checkpoint.get("best_metrics"),
        "monitor": checkpoint.get("monitor"),
        "training_config": checkpoint.get("training_config"),
        "run_config": checkpoint.get("run_config"),
        "text_train_condition": checkpoint.get("text_train_condition"),
        "text_test_conditions": checkpoint.get("text_test_conditions"),
    }
    atomic_torch_save(weights_object, path)


# =============================================================================
# 训练结果与配置辅助函数
# =============================================================================
def choose_monitor_name(
    requested_monitor: str,
    has_validation: bool,
) -> str:
    if requested_monitor == "auto":
        return "val_mae" if has_validation else "train_mae"
    if requested_monitor.startswith("val_") and not has_validation:
        raise ValueError(
            f"monitor={requested_monitor} 需要非空验证集"
        )
    return requested_monitor


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
    epoch_metrics: dict[str, Optional[float]],
) -> float:
    value = epoch_metrics.get(monitor_name)
    if value is None:
        raise ValueError(f"无法计算监控指标 {monitor_name}")
    return float(value)


def metrics_to_log_text(
    name: str,
    metrics: Optional[dict[str, float]],
) -> str:
    if metrics is None:
        return f"{name}: unavailable"
    return (
        f"{name}: mse={metrics['mse']:.8f} | "
        f"mae={metrics['mae']:.8f} | "
        f"rmse={metrics['rmse']:.8f}"
    )


def build_training_config(
    args: SimpleNamespace,
    actual_lr: float,
    device: torch.device,
    run_dir: Path,
) -> dict[str, Any]:
    return {
        "program": Path(__file__).name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode,
        "experiment_tag": args.experiment_tag,
        "base_path": ".",
        "runs_path": "./runs",
        "run_dir": path_for_record(run_dir),
        "project_root": ".",
        "data_path": path_for_record(args.data_path),
        "target": args.target,
        "split_ratio": args.split_ratio,
        "split_counts": args.split_counts,
        "qwen_model_name": args.qwen_model_name,
        "allow_remote_model": args.allow_remote_model,
        "depth": args.depth,
        "edge_input_dim": args.edge_input_dim,
        "triplet_input_dim": args.triplet_input_dim,
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
        "freeze_qwen": args.freeze_qwen,
        "text_train_condition": args.text_train_condition,
        "text_test_conditions": list(args.text_test_conditions),
        "text_effect_neutral_threshold": args.text_effect_neutral_threshold,
        "shuffle_singleton_policy": args.shuffle_singleton_policy,
        "device": str(device),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": (
            args.batch_size * args.gradient_accumulation_steps
        ),
        "epochs": args.epochs,
        "absolute_lr_requested": args.lr,
        "blr": args.blr,
        "actual_lr": actual_lr,
        "min_lr": args.min_lr,
        "warmup_epochs": args.warmup_epochs,
        "weight_decay": args.weight_decay,
        "betas": [args.beta1, args.beta2],
        "clip_grad": args.clip_grad,
        "amp": args.amp,
        "num_workers": args.num_workers,
        "pin_mem": args.pin_mem,
        "drop_last": args.drop_last,
        "monitor_requested": args.monitor,
        "save_interval": args.save_interval,
        "early_stopping_patience": args.early_stopping_patience,
        "resume": path_for_record(args.resume),
        "weights": path_for_record(args.weights),
        "eval_checkpoint": path_for_record(args.eval_checkpoint),
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
        if isinstance(value, (int, float)):
            writer.add_scalar(f"train/{name}", value, epoch)
    if val_metrics is not None:
        for name, value in val_metrics.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(f"val/{name}", value, epoch)
    if test_metrics is not None:
        for name, value in test_metrics.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(f"test/{name}", value, epoch)
    writer.flush()


def validate_checkpoint_text_condition(
    checkpoint: Any,
    current_condition: str,
    *,
    mode: str,
    logger: logging.Logger,
) -> None:
    if not isinstance(checkpoint, dict):
        return
    checkpoint_condition = checkpoint.get("text_train_condition")
    if checkpoint_condition is None:
        config = checkpoint.get("training_config") or checkpoint.get("run_config")
        if isinstance(config, dict):
            checkpoint_condition = config.get("text_train_condition")

    if checkpoint_condition is None:
        logger.warning("checkpoint does not record text_train_condition")
        return

    checkpoint_condition = str(checkpoint_condition).strip().lower()
    if mode == "resume" and checkpoint_condition != current_condition:
        raise ValueError(
            "resume 检查点的文本训练条件与当前配置不一致："
            f"checkpoint={checkpoint_condition}, current={current_condition}。"
            "完整续训必须保持同一文本训练条件。"
        )
    if checkpoint_condition != current_condition:
        logger.warning(
            "checkpoint text condition=%s, current configured condition=%s",
            checkpoint_condition,
            current_condition,
        )


# =============================================================================
# 主程序
# =============================================================================
def main() -> None:
    RUNS_PATH.mkdir(parents=True, exist_ok=True)
    args = build_runtime_config()
    validate_config(args)

    if args.mode == "plot_only":
        run_dir = resolve_run_path(args.run_dir)
        assert run_dir is not None
        figures = plot_all_available(run_dir)
        print("Generated figures:")
        for item in figures:
            print(item)
        return

    resolved_data_path = resolve_path(args.data_path)
    assert resolved_data_path is not None
    args.data_path = str(resolved_data_path)
    if not Path(args.data_path).is_file():
        raise FileNotFoundError(f"dataset does not exist: {args.data_path}")

    qwen_path = Path(args.qwen_model_name).expanduser()
    if not qwen_path.is_absolute() and not args.allow_remote_model:
        resolved_qwen = resolve_path(args.qwen_model_name)
        assert resolved_qwen is not None
        args.qwen_model_name = str(resolved_qwen)

    resume_path = resolve_run_path(args.resume)
    weights_path = resolve_run_path(args.weights)
    eval_path = resolve_run_path(args.eval_checkpoint)
    args.resume = resume_path
    args.weights = weights_path
    args.eval_checkpoint = eval_path

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False"
        )
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
    logger.info("mode: %s", args.mode)
    logger.info("experiment tag: %s", args.experiment_tag)
    logger.info("project root: %s", PROJECT_ROOT)
    logger.info("run dir: %s", run_dir)
    logger.info("device: %s", device)
    logger.info("torch version: %s", torch.__version__)
    if device.type == "cuda":
        logger.info("CUDA device: %s", torch.cuda.get_device_name(device))
    logger.info("AMP enabled: %s", amp_enabled)
    logger.info("text train condition: %s", args.text_train_condition)
    logger.info("text test conditions: %s", args.text_test_conditions)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        log_cuda_memory(device, logger, "startup")

    log_section(logger, "Dataset")
    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    ) = build_dataloaders(args, device, logger)

    del train_dataset, val_dataset, test_dataset

    has_validation = val_loader is not None
    monitor_name = choose_monitor_name(args.monitor, has_validation)

    effective_batch_size = (
        args.batch_size * args.gradient_accumulation_steps
    )
    actual_lr = (
        float(args.lr)
        if args.lr is not None
        else float(args.blr) * effective_batch_size / 256.0
    )
    if args.min_lr > actual_lr:
        raise ValueError(
            f"min_lr ({args.min_lr}) cannot exceed actual_lr ({actual_lr})"
        )

    training_config = build_training_config(args, actual_lr, device, run_dir)
    training_config["monitor"] = monitor_name
    config_name = (
        f"resume_config_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        if append
        else "config.json"
    )
    save_json(run_dir / config_name, training_config)

    log_section(logger, "Model")
    model = build_model(args, device)
    verify_strict_bypass_support(model)
    logger.info("strict text bypass interface: PASS")
    log_model_summary(model, logger)
    log_cuda_memory(device, logger, "after model build")

    optimizer = build_optimizer(model, args, actual_lr)
    scaler = build_grad_scaler(amp_enabled)

    logger.info("micro-batch size: %d", args.batch_size)
    logger.info(
        "gradient accumulation steps: %d",
        args.gradient_accumulation_steps,
    )
    logger.info("effective batch size: %d", effective_batch_size)
    logger.info("actual peak learning rate: %.8g", actual_lr)
    logger.info("minimum learning rate: %.8g", args.min_lr)
    logger.info("warmup epochs: %.4f", args.warmup_epochs)
    logger.info("monitor: %s (lower is better)", monitor_name)

    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint_path = checkpoints_dir / "last_checkpoint.pt"
    best_checkpoint_path = checkpoints_dir / "best_checkpoint.pt"
    best_weights_path = checkpoints_dir / "best_model_weights.pt"

    start_epoch_index = 0
    global_step = 0
    best_metric = math.inf
    best_epoch = 0
    best_metrics: Optional[dict[str, Any]] = None
    epochs_without_improvement = 0

    if args.mode == "resume":
        assert resume_path is not None
        log_section(logger, "Resume")
        checkpoint = load_torch_file(resume_path, map_location=device)
        validate_checkpoint_text_condition(
            checkpoint,
            args.text_train_condition,
            mode="resume",
            logger=logger,
        )
        load_model_state(model, checkpoint, logger)

        if "optimizer_state_dict" not in checkpoint:
            raise KeyError(
                "该文件不包含 optimizer_state_dict，无法完整续训；"
                "请将 training_mode 设为 'weights'，并设置 "
                "weights_checkpoint_path。"
            )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        checkpoint_monitor = checkpoint.get("monitor")
        restore_best_state = True
        if checkpoint_monitor and args.monitor == "auto":
            monitor_name = choose_monitor_name(
                str(checkpoint_monitor),
                has_validation,
            )
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
        global_step = int(checkpoint.get("global_step", 0))
        if restore_best_state:
            best_metric = float(checkpoint.get("best_metric", math.inf))
            best_epoch = int(checkpoint.get("best_epoch", 0))
            best_metrics = checkpoint.get("best_metrics")
            epochs_without_improvement = int(
                checkpoint.get("epochs_without_improvement", 0)
            )
        restore_rng_state(checkpoint.get("rng_state"), logger)
        logger.info("resume checkpoint: %s", resume_path)
        logger.info("next epoch: %d", start_epoch_index + 1)
        logger.info("restored global step: %d", global_step)
        logger.info("restored best epoch: %d", best_epoch)
        logger.info("restored best %s: %s", monitor_name, best_metric)

    elif args.mode == "weights":
        assert weights_path is not None
        log_section(logger, "Load model weights only")
        checkpoint = load_torch_file(weights_path, map_location=device)
        validate_checkpoint_text_condition(
            checkpoint,
            args.text_train_condition,
            mode="weights",
            logger=logger,
        )
        load_model_state(model, checkpoint, logger)
        logger.info("weights checkpoint: %s", weights_path)
        logger.info("optimizer, scaler, epoch and best metrics start from scratch")

    elif args.mode == "eval":
        assert eval_path is not None
        log_section(logger, "Evaluation")
        checkpoint = load_torch_file(eval_path, map_location=device)
        load_model_state(model, checkpoint, logger)

        val_summary = evaluate_text_conditions(
            model=model,
            loader=val_loader,
            device=device,
            conditions=args.text_test_conditions,
            amp_enabled=amp_enabled,
            output_dir=run_dir / "text_effect_validation",
            neutral_threshold=args.text_effect_neutral_threshold,
            singleton_policy=args.shuffle_singleton_policy,
            split_name="validation",
        )
        test_summary = evaluate_text_conditions(
            model=model,
            loader=test_loader,
            device=device,
            conditions=args.text_test_conditions,
            amp_enabled=amp_enabled,
            output_dir=run_dir / "text_effect_test",
            neutral_threshold=args.text_effect_neutral_threshold,
            singleton_policy=args.shuffle_singleton_policy,
            split_name="test",
        )
        result = {
            "checkpoint": path_for_record(eval_path),
            "validation": val_summary,
            "test": test_summary,
            "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        }
        save_json(run_dir / "eval_result.json", result)
        logger.info("evaluation result: %s", run_dir / "eval_result.json")
        return

    else:
        log_section(logger, "Initialize from scratch")
        logger.info("training starts from newly initialized trainable parameters")

    append_csv = args.mode == "resume"
    step_file, step_writer = create_csv_writer(
        run_dir / "step_metrics.csv",
        STEP_FIELDS,
        append_csv,
    )
    epoch_file, epoch_writer = create_csv_writer(
        run_dir / "epoch_metrics.csv",
        EPOCH_FIELDS,
        append_csv,
    )

    tensorboard_writer = None
    if args.tensorboard:
        if SummaryWriter is None:
            logger.warning(
                "TensorBoard is unavailable; install tensorboard to enable it"
            )
        else:
            tensorboard_writer = SummaryWriter(
                log_dir=str(run_dir / "tensorboard")
            )

    training_started = time.time()
    stopped_early = False
    last_completed_epoch = start_epoch_index

    try:
        log_section(logger, "Training")
        if start_epoch_index >= args.epochs:
            logger.warning(
                "checkpoint has completed %d epochs; target epochs=%d, "
                "no new training is needed",
                start_epoch_index,
                args.epochs,
            )

        for epoch_index in range(start_epoch_index, args.epochs):
            epoch_started = time.time()
            logger.info("")
            logger.info(
                "========== Epoch %d/%d ==========",
                epoch_index + 1,
                args.epochs,
            )

            train_metrics, global_step, lr_end = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                epoch_index=epoch_index,
                global_step=global_step,
                args=args,
                actual_lr=actual_lr,
                logger=logger,
                step_writer=step_writer,
                step_file=step_file,
                tensorboard_writer=tensorboard_writer,
            )

            val_metrics = evaluate(
                model,
                val_loader,
                device,
                args.text_train_condition,
                amp_enabled,
                args.shuffle_singleton_policy,
            )
            test_metrics = None
            if args.evaluate_test_each_epoch:
                test_metrics = evaluate(
                    model,
                    test_loader,
                    device,
                    args.text_train_condition,
                    amp_enabled,
                    args.shuffle_singleton_policy,
                )

            epoch_seconds = time.time() - epoch_started
            logger.info(metrics_to_log_text("train", train_metrics))
            logger.info(metrics_to_log_text("val", val_metrics))
            if test_metrics is not None:
                logger.info(metrics_to_log_text("test", test_metrics))
            logger.info(
                "epoch time: %s",
                timedelta(seconds=int(epoch_seconds)),
            )

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
                    "lr_end": lr_end,
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

            if is_best:
                atomic_torch_save(checkpoint, best_checkpoint_path)
                save_best_model_weights(best_weights_path, checkpoint)
                save_json(
                    run_dir / "best_result.json",
                    {
                        "best_epoch": best_epoch,
                        "monitor": monitor_name,
                        "best_metric": best_metric,
                        "metrics": best_metrics,
                        "text_train_condition": args.text_train_condition,
                        "checkpoint": path_for_record(best_checkpoint_path),
                        "weights": path_for_record(best_weights_path),
                        "updated_at": datetime.now().isoformat(
                            timespec="seconds"
                        ),
                    },
                )
                logger.info(
                    "NEW BEST | epoch %03d | %s %.8f | %s",
                    best_epoch,
                    monitor_name,
                    best_metric,
                    best_checkpoint_path,
                )

            if (
                args.save_interval > 0
                and (epoch_index + 1) % args.save_interval == 0
            ):
                interval_path = checkpoints_dir / (
                    f"checkpoint_epoch_{epoch_index + 1:04d}.pt"
                )
                atomic_torch_save(checkpoint, interval_path)
                logger.info("saved interval checkpoint: %s", interval_path)

            if (
                args.early_stopping_patience > 0
                and epochs_without_improvement
                >= args.early_stopping_patience
            ):
                stopped_early = True
                logger.info(
                    "early stopping: no improvement for %d epochs",
                    epochs_without_improvement,
                )
                break

    finally:
        step_file.close()
        epoch_file.close()
        if tensorboard_writer is not None:
            tensorboard_writer.close()

    move_optimizer_state_to_cpu(optimizer, logger)
    log_cuda_memory(device, logger, "before final evaluation")

    log_section(logger, "Final evaluation")
    selected_checkpoint_path = (
        best_checkpoint_path
        if best_checkpoint_path.exists()
        else last_checkpoint_path
    )

    if selected_checkpoint_path.exists():
        selected_checkpoint = load_torch_file(
            selected_checkpoint_path,
            map_location=device,
        )
        load_model_state(model, selected_checkpoint, logger)
        logger.info("selected checkpoint: %s", selected_checkpoint_path)
    else:
        logger.warning("no checkpoint found; evaluating current in-memory model")

    final_val_training_condition = evaluate(
        model,
        val_loader,
        device,
        args.text_train_condition,
        amp_enabled,
        args.shuffle_singleton_policy,
    )
    final_test_training_condition = evaluate(
        model,
        test_loader,
        device,
        args.text_train_condition,
        amp_enabled,
        args.shuffle_singleton_policy,
    )

    val_text_effect = evaluate_text_conditions(
        model=model,
        loader=val_loader,
        device=device,
        conditions=args.text_test_conditions,
        amp_enabled=amp_enabled,
        output_dir=run_dir / "text_effect_validation",
        neutral_threshold=args.text_effect_neutral_threshold,
        singleton_policy=args.shuffle_singleton_policy,
        split_name="validation",
    )
    test_text_effect = evaluate_text_conditions(
        model=model,
        loader=test_loader,
        device=device,
        conditions=args.text_test_conditions,
        amp_enabled=amp_enabled,
        output_dir=run_dir / "text_effect_test",
        neutral_threshold=args.text_effect_neutral_threshold,
        singleton_policy=args.shuffle_singleton_policy,
        split_name="test",
    )

    total_seconds = time.time() - training_started
    final_result = {
        "experiment_tag": args.experiment_tag,
        "text_train_condition": args.text_train_condition,
        "text_test_conditions": args.text_test_conditions,
        "selected_checkpoint": (
            path_for_record(selected_checkpoint_path)
            if selected_checkpoint_path.exists()
            else None
        ),
        "last_completed_epoch": last_completed_epoch,
        "best_epoch": best_epoch,
        "monitor": monitor_name,
        "best_metric": None if math.isinf(best_metric) else best_metric,
        "best_metrics": best_metrics,
        "stopped_early": stopped_early,
        "training_seconds": total_seconds,
        "validation_at_training_condition": final_val_training_condition,
        "test_at_training_condition": final_test_training_condition,
        "validation_text_effect": val_text_effect,
        "test_text_effect": test_text_effect,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(run_dir / "final_result.json", final_result)

    figures = plot_all_available(run_dir)
    for figure in figures:
        logger.info("saved figure: %s", figure)

    logger.info("")
    logger.info("text-effect experiment completed")
    logger.info("run dir: %s", run_dir)
    logger.info("best epoch: %s", best_epoch)
    logger.info("best %s: %s", monitor_name, best_metric)
    logger.info("final result: %s", run_dir / "final_result.json")
    logger.info("test text-effect result: %s", run_dir / "text_effect_test")
    logger.info(
        "total time: %s",
        timedelta(seconds=int(total_seconds)),
    )


if __name__ == "__main__":
    main()
