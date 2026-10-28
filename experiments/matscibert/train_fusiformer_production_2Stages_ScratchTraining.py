#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fusiformer standalone two-stage scratch training with MatSciBERT cached embeddings.

Standalone version:
- DOES NOT import train_fusiformer_production_editor_v4.py
- Uses the project's existing model / loader modules only
- AdamW + gradient accumulation
- step-based linear warmup + half-cosine decay
- optional Stage-2 warmup + half-cosine refinement
- checkpoint / best weights / CSV / TensorBoard / plots
- supports precomputed MatSciBERT embeddings
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:128",
)

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from loader.loader_cached import get_dataset, GraphCollectingCached
from loader.transforms import AddAngleFeature, AddEdgeFeature, AddNodeFeature, Compose
from models.TextGuidedFusiformerCached import TextGuidedFusiformer
from models.text_encoders import get_text_hidden_size

try:
    from engine import to_device
except ImportError:
    def to_device(value: Any, device: torch.device) -> Any:
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
    from tools.weight_decay import add_weight_decay as project_add_weight_decay
except ImportError:
    project_add_weight_decay = None

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


SCRIPT_VERSION = "2026-09-01-standalone-matscibert-2stage-scratch-v1"
BASE_PATH = Path(__file__).resolve().parents[2]
RUNS_PATH = BASE_PATH / "runs"


# =============================================================================
# Editable configuration
# =============================================================================
# 1. Mode / output
training_mode = "scratch"
resume_checkpoint_path = None
weights_checkpoint_path = None
eval_checkpoint_path = None
run_log_dir = None
output_root = RUNS_PATH / "fusiformer_production_matscibert_mlp64"

# 2. Dataset
data_path = "./data/dataset_final_merged_with_robocrys.json"
target_name = "e_form"
data_split_ratio = [0.8, 0.1, 0.1]
data_split_counts = None

# 3. MatSciBERT text encoder
text_encoder_name = "matscibert"
text_model_name = "/root/models/matscibert"
text_pooling = "mean"
text_preprocessing = "matscibert"
text_field = "description"
text_max_length = 512
freeze_text_encoder = True
allow_remote_model = False

use_precomputed_text_embeddings = True
text_embedding_cache_path = (
    "./data/matscibert_cache/matscibert_description_meanpool_f32.npy"
)
text_embedding_meta_path = None

# 4. Fusiformer architecture
model_depth = 6
edge_input_dim = 80
triplet_input_dim = 40
embed_dim = 128
text_proj_dim = 64
text_proj_hidden_dims = (256, 128)
text_proj_dropout = 0.1
num_heads = 4
use_text = True

# 5. Stage-1 + Stage-2 training
batch_size = 64
gradient_accumulation_steps = 1

# Stage 1: 70 epochs
lr_schedule_total_epochs = 100
# Stage 2: 30 additional epochs
additional_epochs = 30
total_epochs = lr_schedule_total_epochs + additional_epochs

# With absolute_lr=None and scale=False:
# peak_lr = base_lr * batch_size / 256 = 1e-3 * 64 / 256 = 2.5e-4
scale_lr_with_accumulation = False
absolute_lr = None
base_lr = 1e-3

# Exactly ONE of these must be non-None.
minimum_lr = 1e-7
minimum_lr_ratio = None

warmup_ratio = 0.04
warmup_steps_override = None
warmup_initial_lr = 1e-7

# Stage 2 is intentionally enabled in this standalone 2-stage file.
# It starts continuously from Stage-1 floor 1e-7,
# warms up to 2.5e-7, then cosine-decays to 1e-9.
use_stage2_dynamic_lr = True
stage2_peak_lr = 2.5e-7
stage2_min_lr = 1e-9
stage2_warmup_epochs = 3

weight_decay = 0.05
adam_beta1 = 0.9
adam_beta2 = 0.999

use_module_lr_scales = False
backbone_lr_scale = 1.0
text_module_lr_scale = 1.0
regression_head_lr_scale = 1.0
text_encoder_lr_scale = 0.1
clip_grad_norm = None

# 6. Runtime / logging
device_name = "cuda"
random_seed = 42
num_workers = 8
pin_memory = True
drop_last = False
use_amp = False
use_deterministic_algorithms = False

best_model_monitor = "auto"
save_interval = 20
log_interval = 250

# Disabled so Stage 2 is guaranteed to be reached.
# After validating the full 2-stage schedule, you can re-enable it.
early_stopping_patience = 20

enable_tensorboard = True
exclude_frozen_text_encoder_from_checkpoint = True
evaluate_test_each_epoch = False


STEP_FIELDS = [
    "epoch", "step", "micro_step", "global_step",
    "accumulation_position", "update_applied",
    "progress_epoch", "loss", "lr", "min_group_lr",
    "grad_norm", "amp_scale",
]

EPOCH_FIELDS = [
    "epoch", "train_mse", "train_mae", "train_rmse",
    "val_mse", "val_mae", "val_rmse",
    "optimizer_steps_end", "micro_steps_end",
    "lr_end", "epoch_seconds",
]


# =============================================================================
# Configuration / paths
# =============================================================================
def build_runtime_config() -> SimpleNamespace:
    ratio = None if data_split_counts is not None else list(data_split_ratio)
    counts = None if data_split_counts is None else list(data_split_counts)

    return SimpleNamespace(
        mode=str(training_mode).strip().lower(),
        resume=resume_checkpoint_path,
        weights=weights_checkpoint_path,
        eval_checkpoint=eval_checkpoint_path,
        run_dir=run_log_dir,
        output_root=output_root,

        data_path=data_path,
        target=target_name,
        split_ratio=ratio,
        split_counts=counts,

        text_encoder_name=str(text_encoder_name),
        text_model_name=str(text_model_name),
        text_pooling=str(text_pooling),
        text_preprocessing=str(text_preprocessing),
        text_field=str(text_field),
        text_max_length=int(text_max_length),
        freeze_text_encoder=bool(freeze_text_encoder),
        allow_remote_model=bool(allow_remote_model),
        use_precomputed_text_embeddings=bool(use_precomputed_text_embeddings),
        text_embedding_cache_path=text_embedding_cache_path,
        text_embedding_meta_path=text_embedding_meta_path,

        depth=int(model_depth),
        edge_input_dim=int(edge_input_dim),
        triplet_input_dim=int(triplet_input_dim),
        embed_dim=int(embed_dim),
        num_heads=int(num_heads),
        text_proj_dim=int(text_proj_dim),
        text_proj_hidden_dims=tuple(int(x) for x in text_proj_hidden_dims),
        text_proj_dropout=float(text_proj_dropout),
        use_text=bool(use_text),

        batch_size=int(batch_size),
        accumulation_steps=int(gradient_accumulation_steps),
        epochs=int(total_epochs),
        lr_schedule_epochs=int(lr_schedule_total_epochs),

        use_stage2_dynamic_lr=bool(use_stage2_dynamic_lr),
        stage2_peak_lr=float(stage2_peak_lr),
        stage2_min_lr=float(stage2_min_lr),
        stage2_warmup_epochs=int(stage2_warmup_epochs),

        scale_lr_with_accumulation=bool(scale_lr_with_accumulation),
        lr=None if absolute_lr is None else float(absolute_lr),
        blr=float(base_lr),
        min_lr=None if minimum_lr is None else float(minimum_lr),
        min_lr_ratio=None if minimum_lr_ratio is None else float(minimum_lr_ratio),
        warmup_ratio=float(warmup_ratio),
        warmup_steps_override=(
            None if warmup_steps_override is None else int(warmup_steps_override)
        ),
        warmup_init_lr=float(warmup_initial_lr),

        weight_decay=float(weight_decay),
        beta1=float(adam_beta1),
        beta2=float(adam_beta2),

        use_module_lr_scales=bool(use_module_lr_scales),
        backbone_lr_scale=float(backbone_lr_scale),
        text_lr_scale=float(text_module_lr_scale),
        head_lr_scale=float(regression_head_lr_scale),
        text_encoder_lr_scale=float(text_encoder_lr_scale),
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
        exclude_frozen_text_encoder_from_checkpoint=bool(
            exclude_frozen_text_encoder_from_checkpoint
        ),
        evaluate_test_each_epoch=bool(evaluate_test_each_epoch),
    )


def validate_config(args: SimpleNamespace) -> None:
    valid_modes = {"scratch", "weights", "eval"}
    if args.mode not in valid_modes:
        raise ValueError(
            f"training_mode={args.mode!r} invalid; standalone file supports {sorted(valid_modes)}"
        )

    if args.mode == "weights" and not args.weights:
        raise ValueError(
            'training_mode="weights" requires weights_checkpoint_path'
        )
    if args.mode == "eval" and not args.eval_checkpoint:
        raise ValueError(
            'training_mode="eval" requires eval_checkpoint_path'
        )

    if args.epochs < 1:
        raise ValueError("total_epochs must be >= 1")
    if args.lr_schedule_epochs < 1:
        raise ValueError("lr_schedule_total_epochs must be >= 1")
    if args.epochs < args.lr_schedule_epochs:
        raise ValueError("total_epochs cannot be smaller than lr_schedule_total_epochs")

    if args.use_stage2_dynamic_lr:
        if args.epochs <= args.lr_schedule_epochs:
            raise ValueError(
                "Stage-2 requires additional_epochs > 0"
            )
        if args.stage2_peak_lr <= 0:
            raise ValueError("stage2_peak_lr must be > 0")
        if args.stage2_min_lr < 0:
            raise ValueError("stage2_min_lr must be >= 0")
        if args.stage2_min_lr >= args.stage2_peak_lr:
            raise ValueError("stage2_min_lr must be < stage2_peak_lr")
        if args.stage2_warmup_epochs < 0:
            raise ValueError("stage2_warmup_epochs cannot be negative")
        if args.stage2_warmup_epochs >= (args.epochs - args.lr_schedule_epochs):
            raise ValueError("stage2_warmup_epochs must be < additional_epochs")

    if args.batch_size < 1 or args.accumulation_steps < 1:
        raise ValueError("batch size and accumulation steps must be >= 1")

    min_lr_set = int(args.min_lr is not None) + int(args.min_lr_ratio is not None)
    if min_lr_set != 1:
        raise ValueError(
            "minimum_lr and minimum_lr_ratio: set exactly one, set the other to None"
        )
    if args.min_lr is not None and args.min_lr < 0:
        raise ValueError("minimum_lr must be >= 0")
    if args.min_lr_ratio is not None and not (0 <= args.min_lr_ratio < 1):
        raise ValueError("minimum_lr_ratio must satisfy 0 <= ratio < 1")

    if args.lr is not None and args.lr <= 0:
        raise ValueError("absolute_lr must be > 0 or None")
    if args.blr <= 0:
        raise ValueError("base_lr must be > 0")
    if not (0 <= args.warmup_ratio < 1):
        raise ValueError("warmup_ratio must satisfy 0 <= ratio < 1")
    if args.warmup_init_lr <= 0:
        raise ValueError("warmup_initial_lr must be > 0")

    if args.use_precomputed_text_embeddings:
        if not args.freeze_text_encoder:
            raise ValueError(
                "precomputed text embeddings require freeze_text_encoder=True"
            )
        if not args.text_embedding_cache_path:
            raise ValueError(
                "text_embedding_cache_path is required"
            )

    if args.split_ratio is not None:
        if len(args.split_ratio) != 3 or not math.isclose(
            sum(args.split_ratio), 1.0, rel_tol=1e-6, abs_tol=1e-6
        ):
            raise ValueError("data_split_ratio must contain 3 values summing to 1")

    if args.clip_grad is not None and args.clip_grad <= 0:
        raise ValueError("clip_grad_norm must be > 0 or None")


def resolve_path(value: str | os.PathLike[str] | None) -> Optional[Path]:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (BASE_PATH / path).resolve()


def make_run_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_root / stamp
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"{stamp}-{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


# =============================================================================
# Logging / serialization / plotting
# =============================================================================
def setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("fusiformer_standalone")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    fh = logging.FileHandler(run_dir / "train.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def create_csv_writer(path: Path, fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    f.flush()
    return f, writer


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
            for filename, ylabel, keys in [
                ("mse_curve.png", "MSE", ["train_mse", "val_mse"]),
                ("mae_curve.png", "MAE", ["train_mae", "val_mae"]),
                ("rmse_curve.png", "RMSE", ["train_rmse", "val_rmse"]),
            ]:
                plt.figure()
                plotted = False
                for key in keys:
                    vals = [safe_float(row.get(key)) for row in rows]
                    if any(v is not None for v in vals):
                        plt.plot(x, vals, marker="o", label=key)
                        plotted = True
                if plotted:
                    plt.xlabel("Epoch")
                    plt.ylabel(ylabel)
                    plt.title(f"{ylabel} curve")
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    plt.tight_layout()
                    out = run_dir / filename
                    plt.savefig(out, dpi=200)
                    saved.append(str(out))
                plt.close()

            lr_vals = [safe_float(row.get("lr_end")) for row in rows]
            if any(v is not None for v in lr_vals):
                plt.figure()
                plt.plot(x, lr_vals, marker="o")
                plt.xlabel("Epoch")
                plt.ylabel("Learning rate")
                plt.title("Epoch-end learning rate")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                out = run_dir / "epoch_lr_curve.png"
                plt.savefig(out, dpi=200)
                plt.close()
                saved.append(str(out))

    if step_csv.exists():
        with step_csv.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            pairs = [
                (int(row.get("global_step") or 0), safe_float(row.get("lr")))
                for row in rows
                if str(row.get("update_applied", "0")).lower() not in {"", "0", "false"}
            ]
            pairs = [(x, y) for x, y in pairs if y is not None]
            if pairs:
                plt.figure()
                plt.plot([x for x, _ in pairs], [y for _, y in pairs])
                plt.xlabel("Optimizer step")
                plt.ylabel("Learning rate")
                plt.title("Step learning-rate curve")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                out = run_dir / "step_lr_curve.png"
                plt.savefig(out, dpi=200)
                plt.close()
                saved.append(str(out))

    return saved


# =============================================================================
# Seed / optimizer / LR scheduler
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


def fallback_add_weight_decay(
    model: torch.nn.Module,
    weight_decay_value: float,
) -> list[dict[str, Any]]:
    decay, no_decay = [], []
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
            "weight_decay": weight_decay_value,
            "lr_scale": 1.0,
            "group_name": "decay",
        },
    ]


def parameter_lr_category(name: str) -> str:
    if name.startswith("text_encoder."):
        return "text_encoder"
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
        "text_encoder": args.text_encoder_lr_scale,
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

    return [g for g in grouped.values() if g["params"]]


def build_optimizer(
    model: torch.nn.Module,
    args: SimpleNamespace,
    actual_lr: float,
) -> torch.optim.AdamW:
    if args.use_module_lr_scales:
        groups = build_module_lr_param_groups(model, args)
    elif project_add_weight_decay is not None:
        groups = project_add_weight_decay(model, args.weight_decay)
        for i, group in enumerate(groups):
            group.setdefault("lr_scale", 1.0)
            group.setdefault("group_name", f"project_group_{i}")
    else:
        groups = fallback_add_weight_decay(model, args.weight_decay)

    if not groups:
        raise RuntimeError("No trainable parameters found")

    return torch.optim.AdamW(
        groups,
        lr=actual_lr,
        betas=(args.beta1, args.beta2),
    )


def resolve_schedule_steps(
    num_batches_per_epoch: int,
    total_epochs_value: int,
    accumulation_steps: int,
    warmup_ratio_value: float,
    warmup_steps_override_value: Optional[int],
) -> tuple[int, int, int]:
    updates_per_epoch = math.ceil(num_batches_per_epoch / accumulation_steps)
    total_update_steps = updates_per_epoch * total_epochs_value
    if total_update_steps < 1:
        raise ValueError("total_update_steps must be > 0")

    if warmup_steps_override_value is None:
        warmup_steps = int(round(total_update_steps * warmup_ratio_value))
        if warmup_ratio_value > 0 and warmup_steps == 0 and total_update_steps > 1:
            warmup_steps = 1
    else:
        warmup_steps = int(warmup_steps_override_value)

    if warmup_steps < 0 or warmup_steps >= total_update_steps:
        raise ValueError(
            f"Invalid warmup_steps={warmup_steps}; total_update_steps={total_update_steps}"
        )
    return updates_per_epoch, total_update_steps, warmup_steps


def resolve_minimum_lr(args: SimpleNamespace, peak_lr: float) -> float:
    if args.min_lr is not None:
        return float(args.min_lr)
    assert args.min_lr_ratio is not None
    return float(peak_lr) * float(args.min_lr_ratio)


def scheduled_learning_rate(
    update_step: int,
    peak_lr: float,
    min_lr: float,
    warmup_init_lr: float,
    warmup_steps: int,
    total_update_steps: int,
) -> float:
    if update_step >= total_update_steps:
        return min_lr

    if warmup_steps > 0 and update_step < warmup_steps:
        progress = update_step / max(1, warmup_steps)
        return warmup_init_lr + progress * (peak_lr - warmup_init_lr)

    decay_steps = total_update_steps - warmup_steps
    decay_index = update_step - warmup_steps
    if decay_steps <= 1:
        progress = 1.0
    else:
        progress = decay_index / (decay_steps - 1)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (peak_lr - min_lr) * (
        1.0 + math.cos(math.pi * progress)
    )


def stage2_dynamic_learning_rate(
    extension_step: int,
    start_lr: float,
    peak_lr: float,
    min_lr: float,
    warmup_steps: int,
    total_steps: int,
) -> float:
    if total_steps <= 0 or extension_step >= total_steps:
        return min_lr

    if warmup_steps > 0 and extension_step < warmup_steps:
        if warmup_steps == 1:
            return peak_lr
        progress = extension_step / float(warmup_steps)
        return start_lr + progress * (peak_lr - start_lr)

    decay_steps = total_steps - warmup_steps
    if decay_steps <= 1:
        return min_lr

    decay_index = extension_step - warmup_steps
    progress = min(max(decay_index / float(decay_steps - 1), 0.0), 1.0)
    return min_lr + 0.5 * (peak_lr - min_lr) * (
        1.0 + math.cos(math.pi * progress)
    )


def adjust_learning_rate_extended(
    optimizer: torch.optim.Optimizer,
    update_step: int,
    peak_lr: float,
    min_lr: float,
    warmup_init_lr: float,
    warmup_steps: int,
    total_update_steps: int,
    use_stage2: bool,
    stage2_peak: float,
    stage2_min: float,
    stage2_warmup_steps: int,
    stage2_total_steps: int,
) -> float:
    if use_stage2 and update_step >= total_update_steps:
        lr = stage2_dynamic_learning_rate(
            extension_step=update_step - total_update_steps,
            start_lr=min_lr,
            peak_lr=stage2_peak,
            min_lr=stage2_min,
            warmup_steps=stage2_warmup_steps,
            total_steps=stage2_total_steps,
        )
    else:
        lr = scheduled_learning_rate(
            update_step=update_step,
            peak_lr=peak_lr,
            min_lr=min_lr,
            warmup_init_lr=warmup_init_lr,
            warmup_steps=warmup_steps,
            total_update_steps=total_update_steps,
        )

    for group in optimizer.param_groups:
        group["lr"] = lr * float(group.get("lr_scale", 1.0))
    return lr


def get_lr_range(optimizer: torch.optim.Optimizer) -> tuple[float, float]:
    values = [float(group["lr"]) for group in optimizer.param_groups]
    return min(values), max(values)


def get_optimizer_internal_step(optimizer: torch.optim.Optimizer) -> float:
    maximum = 0.0
    for state in optimizer.state.values():
        value = state.get("step")
        if value is None:
            continue
        if torch.is_tensor(value):
            value = value.detach().item()
        maximum = max(maximum, float(value))
    return maximum


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


# =============================================================================
# Model / data / metrics
# =============================================================================
def build_model(args: SimpleNamespace, device: torch.device) -> torch.nn.Module:
    hidden_size = get_text_hidden_size(
        args.text_model_name,
        local_files_only=not args.allow_remote_model,
    )

    if args.use_precomputed_text_embeddings:
        cache = np.load(args.text_embedding_cache_path, mmap_mode="r")
        if cache.ndim != 2:
            raise ValueError(f"text embedding cache must be [N,D], got {cache.shape}")
        if int(cache.shape[1]) != int(hidden_size):
            raise ValueError(
                "text cache/model dimension mismatch: "
                f"cache D={cache.shape[1]}, model hidden_size={hidden_size}"
            )
        del cache

    model = TextGuidedFusiformer(
        targets=[args.target],
        depth=args.depth,
        edge_input_dim=args.edge_input_dim,
        triplet_input_dim=args.triplet_input_dim,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        text_dim=hidden_size,
        text_proj_dim=args.text_proj_dim,
        text_proj_hidden_dims=args.text_proj_hidden_dims,
        text_proj_dropout=args.text_proj_dropout,
        text_encoder_name=args.text_encoder_name,
        text_model_name=args.text_model_name,
        text_pooling=args.text_pooling,
        text_preprocessing=args.text_preprocessing,
        freeze_text_encoder=args.freeze_text_encoder,
        load_text_encoder=not args.use_precomputed_text_embeddings,
        allow_remote_model=args.allow_remote_model,
    )
    model.use_precomputed_text_embeddings = args.use_precomputed_text_embeddings
    return model.to(device)


def set_text_encoder_eval_if_frozen(model: torch.nn.Module) -> None:
    module = model.module if hasattr(model, "module") else model
    text_encoder = getattr(module, "text_encoder", None)
    if getattr(module, "freeze_text_encoder", False) and text_encoder is not None:
        text_encoder.eval()


def log_model_summary(model: torch.nn.Module, logger: logging.Logger) -> None:
    module = model.module if hasattr(model, "module") else model
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    logger.info("model total params: %d (%.2f M)", total, total / 1e6)
    logger.info("model trainable params: %d (%.2f M)", trainable, trainable / 1e6)

    encoder = getattr(module, "text_encoder", None)
    if encoder is not None:
        enc_total = sum(p.numel() for p in encoder.parameters())
        enc_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        logger.info("text encoder total params: %d", enc_total)
        logger.info("text encoder trainable params: %d", enc_trainable)
    elif getattr(module, "use_precomputed_text_embeddings", False):
        logger.info("text encoder runtime model: NOT LOADED (precomputed embeddings)")

    logger.info(
        "text projection: %s -> %s -> %s",
        getattr(module, "text_dim", "?"),
        " -> ".join(str(x) for x in args_for_summary.text_proj_hidden_dims),
        args_for_summary.text_proj_dim,
    )


def build_dataloaders(
    args: SimpleNamespace,
    device: torch.device,
    logger: logging.Logger,
):
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
        text_encoder_name=args.text_encoder_name,
        text_model_name=args.text_model_name,
        text_pooling=args.text_pooling,
        text_preprocessing=args.text_preprocessing,
        allow_remote_model=args.allow_remote_model,
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

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=effective_drop_last,
        collate_fn=train_dataset.collect(),
        **common,
    )

    def make_eval_loader(dataset):
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

    if len(train_loader) == 0:
        raise RuntimeError("Training DataLoader is empty")

    return (
        train_dataset, val_dataset, test_dataset,
        train_loader, val_loader, test_loader,
    )


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
    use_text_value: bool,
    amp_enabled: bool,
) -> Optional[dict[str, float]]:
    if loader is None:
        return None

    model.eval()
    set_text_encoder_eval_if_frozen(model)
    acc = empty_metric_accumulator()

    for inputs, targets in loader:
        inputs = to_device(inputs, device)
        targets = normalize_targets(targets, device)
        if not use_text_value:
            inputs = clone_inputs_without_text(inputs)

        with autocast_context(device, amp_enabled):
            outputs = normalize_outputs(model(inputs))
        update_metric_accumulator(acc, outputs, targets)

    return finalize_metrics(acc)


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
    stage2_warmup_steps: int,
    stage2_total_steps: int,
    logger: logging.Logger,
    step_writer: csv.DictWriter,
    step_file: Any,
    tensorboard_writer: Any,
) -> tuple[dict[str, float], int, int, float]:
    model.train()
    set_text_encoder_eval_if_frozen(model)

    criterion = torch.nn.MSELoss()
    acc = empty_metric_accumulator()
    num_steps = len(loader)
    accumulation_steps = args.accumulation_steps
    optimizer.zero_grad(set_to_none=True)
    last_lr = get_lr_range(optimizer)[1]

    for step, (inputs, targets) in enumerate(loader):
        window_start = (step // accumulation_steps) * accumulation_steps
        window_end = min(window_start + accumulation_steps, num_steps)
        window_size = window_end - window_start
        accumulation_position = step - window_start + 1
        is_update_boundary = step + 1 == window_end

        if accumulation_position == 1:
            last_lr = adjust_learning_rate_extended(
                optimizer=optimizer,
                update_step=global_step,
                peak_lr=peak_lr,
                min_lr=min_lr,
                warmup_init_lr=args.warmup_init_lr,
                warmup_steps=warmup_steps,
                total_update_steps=total_update_steps,
                use_stage2=args.use_stage2_dynamic_lr,
                stage2_peak=args.stage2_peak_lr,
                stage2_min=args.stage2_min_lr,
                stage2_warmup_steps=stage2_warmup_steps,
                stage2_total_steps=stage2_total_steps,
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

            internal_before = get_optimizer_internal_step(optimizer)
            scaler.step(optimizer)
            scaler.update()
            internal_after = get_optimizer_internal_step(optimizer)

            if not args.amp:
                update_applied = True
            else:
                update_applied = internal_after > internal_before

            optimizer.zero_grad(set_to_none=True)

            if update_applied:
                global_step += 1

        update_metric_accumulator(acc, outputs, targets)
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

        if step % args.log_interval == 0 or step == num_steps - 1:
            logger.info(
                "epoch %03d | micro %04d/%04d | accum %d/%d | "
                "optimizer_step %07d | loss %.6f | lr %.8g",
                epoch_index + 1,
                step,
                num_steps - 1,
                accumulation_position,
                window_size,
                global_step,
                loss_value,
                max_group_lr,
            )

    return finalize_metrics(acc), global_step, micro_step, last_lr


# =============================================================================
# Checkpoints
# =============================================================================
def atomic_torch_save(obj: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, output_path)


def load_torch_file(path: Path, map_location: Any) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def get_checkpoint_model_state(
    model: torch.nn.Module,
    exclude_frozen_text_encoder: bool,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    module = model.module if hasattr(model, "module") else model
    excluded_prefixes: list[str] = []

    text_encoder = getattr(module, "text_encoder", None)
    if (
        exclude_frozen_text_encoder
        and getattr(module, "freeze_text_encoder", False)
        and text_encoder is not None
    ):
        excluded_prefixes.append("text_encoder.")

    state: dict[str, torch.Tensor] = {}
    for name, value in module.state_dict().items():
        if any(name.startswith(prefix) for prefix in excluded_prefixes):
            continue
        state[name] = value.detach().cpu()
    return state, excluded_prefixes


def extract_model_state(checkpoint: Any) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must be a dict/state_dict")

    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
        excluded = checkpoint.get("excluded_model_prefixes", [])
    elif "model" in checkpoint:
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
            "model state loading incomplete; "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}"
        )

    logger.info("loaded model tensors: %d", len(state))


def build_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    completed_epoch: int,
    global_step: int,
    micro_step: int,
    best_metric: float,
    best_epoch: int,
    best_metrics: Optional[dict[str, Any]],
    monitor_name: str,
    training_config: dict[str, Any],
    args: SimpleNamespace,
) -> dict[str, Any]:
    model_state, excluded = get_checkpoint_model_state(
        model,
        args.exclude_frozen_text_encoder_from_checkpoint,
    )
    return {
        "format_version": 5,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "script_version": SCRIPT_VERSION,
        "epoch": completed_epoch,
        "global_step": global_step,
        "optimizer_step": global_step,
        "micro_step": micro_step,
        "model_state_dict": model_state,
        "excluded_model_prefixes": excluded,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "monitor": monitor_name,
        "training_config": training_config,
    }


def save_lightweight_weights(path: Path, checkpoint: dict[str, Any]) -> None:
    light = {
        key: checkpoint.get(key)
        for key in [
            "format_version", "saved_at", "script_version",
            "epoch", "global_step", "optimizer_step", "micro_step",
            "model_state_dict", "excluded_model_prefixes",
            "best_metric", "best_epoch", "best_metrics",
            "monitor", "training_config",
        ]
    }
    atomic_torch_save(light, path)


# =============================================================================
# Main
# =============================================================================
def choose_monitor_name(requested: str, has_validation: bool) -> str:
    if requested == "auto":
        return "val_mae" if has_validation else "train_mae"
    if requested.startswith("val_") and not has_validation:
        raise ValueError(f"monitor={requested} requires validation data")
    return requested


def metrics_to_log_text(prefix: str, metrics: Optional[dict[str, float]]) -> str:
    if metrics is None:
        return f"{prefix}: unavailable"
    return (
        f"{prefix} MSE {metrics['mse']:.6f} | "
        f"{prefix} MAE {metrics['mae']:.6f} | "
        f"{prefix} RMSE {metrics['rmse']:.6f}"
    )


def main() -> None:
    global args_for_summary

    RUNS_PATH.mkdir(parents=True, exist_ok=True)
    args = build_runtime_config()
    args_for_summary = args
    validate_config(args)

    args.data_path = str(resolve_path(args.data_path))
    if not Path(args.data_path).is_file():
        raise FileNotFoundError(f"dataset does not exist: {args.data_path}")

    if args.use_precomputed_text_embeddings:
        cache_path = resolve_path(args.text_embedding_cache_path)
        if cache_path is None or not cache_path.is_file():
            raise FileNotFoundError(
                f"text embedding cache does not exist: {cache_path}"
            )
        args.text_embedding_cache_path = str(cache_path)

        if args.text_embedding_meta_path is not None:
            meta_path = resolve_path(args.text_embedding_meta_path)
            args.text_embedding_meta_path = str(meta_path)

    if not Path(args.text_model_name).expanduser().is_absolute() and not args.allow_remote_model:
        args.text_model_name = str(resolve_path(args.text_model_name))

    output_root_path = resolve_path(args.output_root)
    assert output_root_path is not None
    run_dir = make_run_dir(output_root_path)
    logger = setup_logger(run_dir)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device(args.device)
    amp_enabled = args.amp and device.type == "cuda"

    set_seed(args.seed, args.deterministic)

    logger.info("script version: %s", SCRIPT_VERSION)
    logger.info("mode: %s", args.mode)
    logger.info("run dir: %s", run_dir)
    logger.info("device: %s", device)

    (
        train_dataset, val_dataset, test_dataset,
        train_loader, val_loader, test_loader,
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
            f"minimum lr {resolved_min_lr} exceeds peak lr {actual_lr}"
        )
    if args.warmup_init_lr > actual_lr and args.warmup_ratio > 0:
        raise ValueError(
            f"warmup_initial_lr {args.warmup_init_lr} exceeds peak lr {actual_lr}"
        )

    updates_per_epoch, total_update_steps, warmup_steps = resolve_schedule_steps(
        num_batches_per_epoch=len(train_loader),
        total_epochs_value=args.lr_schedule_epochs,
        accumulation_steps=args.accumulation_steps,
        warmup_ratio_value=args.warmup_ratio,
        warmup_steps_override_value=args.warmup_steps_override,
    )

    extension_epochs = max(0, args.epochs - args.lr_schedule_epochs)
    stage2_total_steps = updates_per_epoch * extension_epochs
    stage2_warmup_steps = updates_per_epoch * args.stage2_warmup_epochs

    if args.use_stage2_dynamic_lr and args.stage2_peak_lr < resolved_min_lr:
        raise ValueError(
            "stage2_peak_lr should not be lower than the Stage-1 ending LR; "
            f"got stage2_peak_lr={args.stage2_peak_lr}, Stage1 min={resolved_min_lr}"
        )

    training_config = {
        "script_version": SCRIPT_VERSION,
        "mode": args.mode,
        "data_path": args.data_path,
        "target": args.target,
        "split_ratio": args.split_ratio,
        "text_encoder_name": args.text_encoder_name,
        "text_model_name": args.text_model_name,
        "text_pooling": args.text_pooling,
        "text_preprocessing": args.text_preprocessing,
        "text_field": args.text_field,
        "freeze_text_encoder": args.freeze_text_encoder,
        "use_precomputed_text_embeddings": args.use_precomputed_text_embeddings,
        "text_embedding_cache_path": args.text_embedding_cache_path,
        "depth": args.depth,
        "edge_input_dim": args.edge_input_dim,
        "triplet_input_dim": args.triplet_input_dim,
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
        "text_proj_dim": args.text_proj_dim,
        "text_proj_hidden_dims": list(args.text_proj_hidden_dims),
        "text_proj_dropout": args.text_proj_dropout,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "scale_lr_with_accumulation": args.scale_lr_with_accumulation,
        "absolute_lr_requested": args.lr,
        "base_lr": args.blr,
        "actual_peak_lr": actual_lr,
        "minimum_lr": resolved_min_lr,
        "warmup_initial_lr": args.warmup_init_lr,
        "warmup_steps": warmup_steps,
        "lr_schedule_epochs": args.lr_schedule_epochs,
        "total_epochs": args.epochs,
        "use_stage2_dynamic_lr": args.use_stage2_dynamic_lr,
        "stage2_peak_lr": args.stage2_peak_lr,
        "stage2_min_lr": args.stage2_min_lr,
        "stage2_warmup_epochs": args.stage2_warmup_epochs,
        "weight_decay": args.weight_decay,
        "betas": [args.beta1, args.beta2],
        "clip_grad_norm": args.clip_grad,
        "monitor": monitor_name,
    }
    save_json(run_dir / "config.json", training_config)

    logger.info("micro batch size: %d", args.batch_size)
    logger.info("gradient accumulation: %d", args.accumulation_steps)
    logger.info("effective batch size: %d", effective_batch_size)
    logger.info("actual peak lr: %.8g", actual_lr)
    logger.info("Stage-1 min lr: %.8g", resolved_min_lr)
    logger.info("Stage-1 epochs: %d", args.lr_schedule_epochs)
    if args.use_stage2_dynamic_lr:
        logger.info(
            "Stage-2 epochs: %d | start %.8g -> peak %.8g -> min %.8g",
            extension_epochs,
            resolved_min_lr,
            args.stage2_peak_lr,
            args.stage2_min_lr,
        )

    model = build_model(args, device)
    set_text_encoder_eval_if_frozen(model)
    log_model_summary(model, logger)

    optimizer = build_optimizer(model, args, actual_lr)
    scaler = build_grad_scaler(amp_enabled)

    adjust_learning_rate_extended(
        optimizer=optimizer,
        update_step=0,
        peak_lr=actual_lr,
        min_lr=resolved_min_lr,
        warmup_init_lr=args.warmup_init_lr,
        warmup_steps=warmup_steps,
        total_update_steps=total_update_steps,
        use_stage2=args.use_stage2_dynamic_lr,
        stage2_peak=args.stage2_peak_lr,
        stage2_min=args.stage2_min_lr,
        stage2_warmup_steps=stage2_warmup_steps,
        stage2_total_steps=stage2_total_steps,
    )

    if args.mode == "weights":
        weights_path = resolve_path(args.weights)
        assert weights_path is not None
        logger.info("loading MODEL WEIGHTS ONLY from: %s", weights_path)
        checkpoint = load_torch_file(weights_path, map_location=device)
        load_model_state(model, checkpoint, logger)
        logger.info("optimizer / scheduler / epoch / best metric are newly initialized")

    elif args.mode == "eval":
        eval_path = resolve_path(args.eval_checkpoint)
        assert eval_path is not None
        checkpoint = load_torch_file(eval_path, map_location=device)
        load_model_state(model, checkpoint, logger)
        val_metrics = evaluate(model, val_loader, device, args.use_text, amp_enabled)
        test_metrics = evaluate(model, test_loader, device, args.use_text, amp_enabled)
        save_json(
            run_dir / "eval_result.json",
            {"validation": val_metrics, "test": test_metrics},
        )
        logger.info(metrics_to_log_text("val", val_metrics))
        logger.info(metrics_to_log_text("test", test_metrics))
        return

    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint_path = checkpoints_dir / "last_checkpoint.pt"
    best_checkpoint_path = checkpoints_dir / "best_checkpoint.pt"
    best_weights_path = checkpoints_dir / "best_model_weights.pt"

    step_file, step_writer = create_csv_writer(
        run_dir / "step_metrics.csv", STEP_FIELDS
    )
    epoch_file, epoch_writer = create_csv_writer(
        run_dir / "epoch_metrics.csv", EPOCH_FIELDS
    )

    tb_writer = None
    if args.tensorboard and SummaryWriter is not None:
        tb_writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    global_step = 0
    micro_step = 0
    best_metric = math.inf
    best_epoch = 0
    best_metrics: Optional[dict[str, Any]] = None
    epochs_without_improvement = 0
    training_started = time.time()
    stopped_early = False

    try:
        for epoch_index in range(args.epochs):
            epoch_started = time.time()
            logger.info("")
            logger.info("========== Epoch %d/%d ==========", epoch_index + 1, args.epochs)

            train_metrics, global_step, micro_step, _ = train_one_epoch(
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
                warmup_steps=warmup_steps,
                total_update_steps=total_update_steps,
                stage2_warmup_steps=stage2_warmup_steps,
                stage2_total_steps=stage2_total_steps,
                logger=logger,
                step_writer=step_writer,
                step_file=step_file,
                tensorboard_writer=tb_writer,
            )

            val_metrics = evaluate(
                model, val_loader, device, args.use_text, amp_enabled
            )
            test_metrics = None
            if args.evaluate_test_each_epoch:
                test_metrics = evaluate(
                    model, test_loader, device, args.use_text, amp_enabled
                )

            current = {
                "train_mse": train_metrics["mse"],
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "val_mse": None if val_metrics is None else val_metrics["mse"],
                "val_mae": None if val_metrics is None else val_metrics["mae"],
                "val_rmse": None if val_metrics is None else val_metrics["rmse"],
            }
            current_metric = float(current[monitor_name])
            is_best = current_metric < best_metric

            if is_best:
                best_metric = current_metric
                best_epoch = epoch_index + 1
                best_metrics = dict(current)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            epoch_seconds = time.time() - epoch_started
            _, lr_end = get_lr_range(optimizer)

            epoch_writer.writerow(
                {
                    "epoch": epoch_index + 1,
                    **{k: "" if v is None else v for k, v in current.items()},
                    "optimizer_steps_end": global_step,
                    "micro_steps_end": micro_step,
                    "lr_end": lr_end,
                    "epoch_seconds": epoch_seconds,
                }
            )
            epoch_file.flush()

            logger.info(metrics_to_log_text("train", train_metrics))
            logger.info(metrics_to_log_text("val", val_metrics))
            logger.info("epoch-end lr: %.8g", lr_end)
            logger.info("epoch time: %s", timedelta(seconds=int(epoch_seconds)))

            if tb_writer is not None:
                for name, value in train_metrics.items():
                    tb_writer.add_scalar(f"train/{name}", value, epoch_index + 1)
                if val_metrics is not None:
                    for name, value in val_metrics.items():
                        tb_writer.add_scalar(f"val/{name}", value, epoch_index + 1)
                tb_writer.flush()

            checkpoint = build_checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                completed_epoch=epoch_index + 1,
                global_step=global_step,
                micro_step=micro_step,
                best_metric=best_metric,
                best_epoch=best_epoch,
                best_metrics=best_metrics,
                monitor_name=monitor_name,
                training_config=training_config,
                args=args,
            )
            atomic_torch_save(checkpoint, last_checkpoint_path)

            if (epoch_index + 1) % args.save_interval == 0:
                atomic_torch_save(
                    checkpoint,
                    checkpoints_dir / f"checkpoint_epoch_{epoch_index + 1:04d}.pt",
                )

            if is_best:
                atomic_torch_save(checkpoint, best_checkpoint_path)
                save_lightweight_weights(best_weights_path, checkpoint)
                save_json(
                    run_dir / "best_result.json",
                    {
                        "best_epoch": best_epoch,
                        "monitor": monitor_name,
                        "best_metric": best_metric,
                        "metrics": best_metrics,
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

        if best_checkpoint_path.exists():
            best_checkpoint = load_torch_file(best_checkpoint_path, map_location=device)
            load_model_state(model, best_checkpoint, logger)

        final_val = evaluate(model, val_loader, device, args.use_text, amp_enabled)
        final_test = evaluate(model, test_loader, device, args.use_text, amp_enabled)

        save_json(
            run_dir / "final_test_result.json",
            {
                "best_epoch": best_epoch,
                "monitor": monitor_name,
                "best_metric": None if math.isinf(best_metric) else best_metric,
                "validation": final_val,
                "test": final_test,
                "stopped_early": stopped_early,
                "optimizer_step": global_step,
                "micro_step": micro_step,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
            },
        )

        logger.info(metrics_to_log_text("best val", final_val))
        logger.info(metrics_to_log_text("best test", final_test))

        for figure in plot_metrics_from_csv(run_dir):
            logger.info("saved figure: %s", figure)

        logger.info(
            "training completed in %s",
            timedelta(seconds=int(time.time() - training_started)),
        )
        logger.info("run directory: %s", run_dir)
        logger.info("best checkpoint: %s", best_checkpoint_path)
        logger.info("best lightweight weights: %s", best_weights_path)

    finally:
        step_file.close()
        epoch_file.close()
        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()


args_for_summary = None

if __name__ == "__main__":
    main()
