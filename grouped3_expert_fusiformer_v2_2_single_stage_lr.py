#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Difficulty-aware Grouped-3 Expert Fusiformer v2.2 Diagnostics
=================================================

Strict no-test-leakage router + embedded KMeans routing parameters + full diagnostics.
Fine-tuning LR: Base-compatible SINGLE-STAGE step-wise linear warmup + half-cosine decay.

Core pipeline
-------------
1. Reproduce Base train/val/test with loader_cached.get_dataset().
2. Load the Base best weights.
3. Extract TRAIN/VAL graph-only graph_out using the frozen Base model.
4. Fit KMeans(K=5) on TRAIN graph_out only (after optional L2 normalization).
5. Evaluate Base prediction difficulty on VAL only; select the two hardest
   latent clusters by VAL MAE. They receive independent experts. The other
   three clusters share Expert-A.
6. Embed KMeans centers + cluster->expert mapping into an nn.Module using
   register_buffer(). No assignments.csv / mp_id / index lookup is required.
7. Fine-tune three experts from the same Base checkpoint. Default: only the
   last two Fusiformer blocks (Block5/6 for depth=6) + regression_head.
8. Optionally train a Global-FT control with the same trainable layers and
   optimization budget.
9. Only after all model-selection/training is complete, extract TEST graph_out,
   route TEST with the frozen embedded router, and evaluate Base / Global-FT /
   Routed-3Expert.
10. Save a deployment bundle containing Base-router weights, embedded router
    buffers, and compact expert overrides.

This file is designed from the user's uploaded Base training program:
MatSciBERT cached embeddings + TextGuidedFusiformerCached, depth=6.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:128",
)
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import joblib
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    r2_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.preprocessing import normalize as sklearn_l2_normalize

from loader.loader_cached import get_dataset, GraphCollectingCached
from loader.transforms import AddAngleFeature, AddEdgeFeature, AddNodeFeature, Compose
from models.TextGuidedFusiformerCached import TextGuidedFusiformer
from models.text_encoders import get_text_hidden_size

try:
    import dgl
except ImportError:
    dgl = None

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
            try:
                return value.to(device, non_blocking=True)
            except TypeError:
                return value.to(device)
        return value

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


# =============================================================================
# 0. EDITABLE EXPERIMENT PARAMETERS
#    For normal use, edit this section only.
# =============================================================================

SCRIPT_VERSION = "2026-09-10-grouped3-expert-embedded-router-v2.2-single-stage-base-lr"
BASE_PATH = Path(__file__).resolve().parent
RUNS_PATH = BASE_PATH / "runs"

# -----------------------------------------------------------------------------
# A. Dataset / Base checkpoint
# -----------------------------------------------------------------------------
DATA_PATH = "./data/dataset_final_merged_with_robocrys.json"
TARGET_NAME = "e_form"
DATA_SPLIT_RATIO = [0.8, 0.1, 0.1]
DATA_SPLIT_COUNTS = None

# Recommended for formal runs: set this to the exact Base best weights path.
# If None, the newest matching file is used.
BASE_WEIGHTS_PATH = "runs/fusiformer_production_matscibert_mlp64/20260902-215103/checkpoints/best_model_weights.pt"
# BASE_WEIGHTS_GLOB = (
#     "runs/fusiformer_production_matscibert_mlp64/20260902-215103/checkpoints/best_model_weights.pt"
# )

OUTPUT_ROOT = RUNS_PATH / "grouped3_expert_matscibert_v2_2_single_stage_lr"

# -----------------------------------------------------------------------------
# B. MatSciBERT settings -- kept consistent with the uploaded Base program
# -----------------------------------------------------------------------------
TEXT_ENCODER_NAME = "matscibert"
TEXT_MODEL_NAME = "/root/models/matscibert"
TEXT_POOLING = "mean"
TEXT_PREPROCESSING = "matscibert"
TEXT_FIELD = "description"
TEXT_MAX_LENGTH = 512
FREEZE_TEXT_ENCODER = True
ALLOW_REMOTE_MODEL = False

USE_PRECOMPUTED_TEXT_EMBEDDINGS = True
TEXT_EMBEDDING_CACHE_PATH = (
    "./data/matscibert_cache/matscibert_description_meanpool_f32.npy"
)
TEXT_EMBEDDING_META_PATH = None

# Expert prediction remains multimodal, matching Base evaluation.
USE_TEXT_FOR_EXPERT_PREDICTION = True

# -----------------------------------------------------------------------------
# C. Base Fusiformer architecture -- uploaded Base program uses depth=6
# -----------------------------------------------------------------------------
MODEL_DEPTH = 8
EDGE_INPUT_DIM = 80
TRIPLET_INPUT_DIM = 40
EMBED_DIM = 128
TEXT_PROJ_DIM = 64
TEXT_PROJ_HIDDEN_DIMS = (256, 128)
TEXT_PROJ_DROPOUT = 0.1
NUM_HEADS = 4

# -----------------------------------------------------------------------------
# D. Strict train-only Router
# -----------------------------------------------------------------------------
ROUTER_K = 5
ROUTER_RANDOM_STATE = 42
ROUTER_N_INIT = 20
ROUTER_MAX_ITER = 500
ROUTER_TOL = 1e-4
ROUTER_L2_NORMALIZE = True
ROUTER_BATCH_SIZE = 64
REUSE_ROUTER_EMBEDDING_CACHE = True
VERIFY_EMBEDDED_ROUTER = True

# NO old assignments.csv is used anywhere in v2.
# The two hardest new train-only clusters are selected using validation error.
DIFFICULTY_METRIC = "mae"  # currently mae or rmse
N_STANDALONE_HARD_EXPERTS = 2
MIN_DIFFICULTY_VAL_SAMPLES_PER_CLUSTER = 100

# Stronger model-selection hygiene: split Base validation samples, stratified by
# router cluster, into two roles. Half selects hard clusters; the other half is
# used for Expert early stopping. This is still the Base validation split, but
# avoids using the exact same rows for both decisions.
USE_DISJOINT_VALIDATION_ROLES = True
DIFFICULTY_SELECTION_FRACTION = 0.50
VALIDATION_ROLE_SEED = 20260910

# Expert IDs are stable even though raw KMeans cluster IDs are arbitrary.
EXPERT_ID_TO_NAME = {
    0: "expert_A_shared3",
    1: "expert_B_hardest",
    2: "expert_C_second_hardest",
}

# -----------------------------------------------------------------------------
# E. Selective fine-tuning
# -----------------------------------------------------------------------------
# depth=6 main experiment:
# last2 => blocks.4 + blocks.5 + regression_head
FINETUNE_MODE = "last2"  # head | last1 | last2 | last3 | all
TRAIN_TEXT_PROJ = False

RUN_GLOBAL_FT_CONTROL = True

# -----------------------------------------------------------------------------
# F. Fine-tuning optimizer / SINGLE-STAGE Base-compatible LR schedule
# -----------------------------------------------------------------------------
BATCH_SIZE = 64
GRADIENT_ACCUMULATION_STEPS = 1
FINE_TUNE_EPOCHS = 30

# Only the Base training file's Stage-1 mechanism is retained:
#   optimizer-step-based linear warmup -> half-cosine decay -> minimum LR
# There is NO Stage-2 / LR reheating / second cosine stage in this program.
#
# Expert fine-tuning uses a smaller absolute peak LR than Base scratch training.
# If ABSOLUTE_LR is not None, it is used directly.
# If ABSOLUTE_LR is None, the Base rule is used:
#   peak_lr = BASE_LR * lr_scaling_batch_size / 256
SCALE_LR_WITH_ACCUMULATION = False
ABSOLUTE_LR: Optional[float] = 1e-6
BASE_LR = 1e-7

# Exactly one of MINIMUM_LR / MINIMUM_LR_RATIO must be non-None.
MINIMUM_LR: Optional[float] = 1e-10
MINIMUM_LR_RATIO: Optional[float] = None

# Match the original Base Stage-1 defaults.
WARMUP_RATIO = 0.04
WARMUP_STEPS_OVERRIDE: Optional[int] = None
WARMUP_INITIAL_LR = 1e-8

WEIGHT_DECAY = 0.05
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999

# Main controlled run defaults to the Base behavior: one shared LR schedule for
# all trainable parameters. Module scales remain available for later ablations.
USE_MODULE_LR_SCALES = False
BACKBONE_LR_SCALE = 1.0
TEXT_MODULE_LR_SCALE = 1.0
REGRESSION_HEAD_LR_SCALE = 1.0
TEXT_ENCODER_LR_SCALE = 0.1

CLIP_GRAD_NORM = None
EARLY_STOPPING_PATIENCE = 8
LOSS_NAME = "mse"  # main controlled experiment keeps the Base loss

# -----------------------------------------------------------------------------
# G. Runtime
# -----------------------------------------------------------------------------
DEVICE_NAME = "cuda"
RANDOM_SEED = 42
NUM_WORKERS = 8
PIN_MEMORY = True
DROP_LAST = False
USE_AMP = False
USE_DETERMINISTIC_ALGORITHMS = False
LOG_INTERVAL = 200
ENABLE_TENSORBOARD = True

# -----------------------------------------------------------------------------
# H. Router diagnostics / visualization
# -----------------------------------------------------------------------------
# These diagnostics DO NOT change KMeans, grouping, or training. They only
# analyze the already-fitted train-only router. PCA is visualization only.
ROUTER_DIAGNOSTICS_ENABLED = True
ROUTER_QUALITY_SAMPLE_SIZE = 5000
ROUTER_PCA_FIT_SAMPLE_SIZE = 30000
ROUTER_MAX_PLOT_POINTS = 30000
ROUTER_DIAGNOSTICS_RANDOM_STATE = 20260910
ROUTER_SAVE_PCA_COORDINATES = True

# -----------------------------------------------------------------------------
# I. Expert/final diagnostics
# -----------------------------------------------------------------------------
CATASTROPHIC_ERROR_THRESHOLDS = [0.1, 0.2, 0.5, 1.0]
SAVE_SAMPLE_PREDICTIONS = True
SAVE_PLOTS = True
PLOT_TRAINING_CURVES = True
FINAL_SCATTER_MAX_POINTS = 30000
PLOT_DPI = 220

# Sanity check only. A warning is emitted if a different Base checkpoint/split
# produces a substantially different test MAE.
EXPECTED_BASE_TEST_MAE: Optional[float] = 0.03423114123425691
EXPECTED_BASE_TEST_MAE_TOLERANCE = 0.005

# Save compact deployment bundle:
# Base state + embedded KMeans buffers + expert changed-submodule states.
SAVE_DEPLOYMENT_BUNDLE = True

# -----------------------------------------------------------------------------
# J. Smoke test
# -----------------------------------------------------------------------------
# True -> 2 epochs per Expert, skips Global-FT. Only for pipeline validation.
SMOKE_TEST = False


# =============================================================================
# 1. Utilities
# =============================================================================

def resolve_path(value: str | os.PathLike[str] | Path | None) -> Optional[Path]:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (BASE_PATH / path).resolve()


def make_run_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = root / stamp
    suffix = 1
    while run_dir.exists():
        run_dir = root / f"{stamp}-{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("grouped3_expert_v2_1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(run_dir / "train.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


def atomic_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


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


def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def build_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def load_torch_file(path: Path, map_location: Any) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def resolve_base_weights() -> Path:
    if BASE_WEIGHTS_PATH is not None:
        path = resolve_path(BASE_WEIGHTS_PATH)
        assert path is not None
        if not path.is_file():
            raise FileNotFoundError(f"BASE_WEIGHTS_PATH not found: {path}")
        return path
    candidates = sorted(
        BASE_PATH.glob(BASE_WEIGHTS_GLOB),
        key=lambda p: p.stat().st_mtime_ns,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No Base best_model_weights.pt matched: {BASE_WEIGHTS_GLOB}"
        )
    return candidates[0].resolve()


# =============================================================================
# 2. Config / Base-compatible model construction
# =============================================================================

def build_args() -> SimpleNamespace:
    epochs = 2 if SMOKE_TEST else int(FINE_TUNE_EPOCHS)
    text_model_path = Path(TEXT_MODEL_NAME).expanduser()
    if not text_model_path.is_absolute() and not ALLOW_REMOTE_MODEL:
        resolved_text_model = resolve_path(TEXT_MODEL_NAME)
        assert resolved_text_model is not None
        text_model_name = str(resolved_text_model)
    else:
        text_model_name = str(text_model_path)

    cache_path = resolve_path(TEXT_EMBEDDING_CACHE_PATH)
    return SimpleNamespace(
        data_path=str(resolve_path(DATA_PATH)),
        target=TARGET_NAME,
        split_ratio=list(DATA_SPLIT_RATIO) if DATA_SPLIT_COUNTS is None else None,
        split_counts=None if DATA_SPLIT_COUNTS is None else list(DATA_SPLIT_COUNTS),
        text_encoder_name=TEXT_ENCODER_NAME,
        text_model_name=text_model_name,
        text_pooling=TEXT_POOLING,
        text_preprocessing=TEXT_PREPROCESSING,
        text_field=TEXT_FIELD,
        text_max_length=int(TEXT_MAX_LENGTH),
        freeze_text_encoder=bool(FREEZE_TEXT_ENCODER),
        allow_remote_model=bool(ALLOW_REMOTE_MODEL),
        use_precomputed_text_embeddings=bool(USE_PRECOMPUTED_TEXT_EMBEDDINGS),
        text_embedding_cache_path=None if cache_path is None else str(cache_path),
        text_embedding_meta_path=(
            None if TEXT_EMBEDDING_META_PATH is None
            else str(resolve_path(TEXT_EMBEDDING_META_PATH))
        ),
        use_text=bool(USE_TEXT_FOR_EXPERT_PREDICTION),
        depth=int(MODEL_DEPTH),
        edge_input_dim=int(EDGE_INPUT_DIM),
        triplet_input_dim=int(TRIPLET_INPUT_DIM),
        embed_dim=int(EMBED_DIM),
        text_proj_dim=int(TEXT_PROJ_DIM),
        text_proj_hidden_dims=tuple(int(x) for x in TEXT_PROJ_HIDDEN_DIMS),
        text_proj_dropout=float(TEXT_PROJ_DROPOUT),
        num_heads=int(NUM_HEADS),
        batch_size=int(BATCH_SIZE),
        accumulation_steps=int(GRADIENT_ACCUMULATION_STEPS),
        epochs=epochs,
        scale_lr_with_accumulation=bool(SCALE_LR_WITH_ACCUMULATION),
        lr=None if ABSOLUTE_LR is None else float(ABSOLUTE_LR),
        blr=float(BASE_LR),
        min_lr=None if MINIMUM_LR is None else float(MINIMUM_LR),
        min_lr_ratio=None if MINIMUM_LR_RATIO is None else float(MINIMUM_LR_RATIO),
        warmup_ratio=float(WARMUP_RATIO),
        warmup_steps_override=(
            None if WARMUP_STEPS_OVERRIDE is None else int(WARMUP_STEPS_OVERRIDE)
        ),
        warmup_init_lr=float(WARMUP_INITIAL_LR),
        weight_decay=float(WEIGHT_DECAY),
        beta1=float(ADAM_BETA1),
        beta2=float(ADAM_BETA2),
        backbone_lr_scale=float(BACKBONE_LR_SCALE),
        text_lr_scale=float(TEXT_MODULE_LR_SCALE),
        head_lr_scale=float(REGRESSION_HEAD_LR_SCALE),
        text_encoder_lr_scale=float(TEXT_ENCODER_LR_SCALE),
        clip_grad=None if CLIP_GRAD_NORM is None else float(CLIP_GRAD_NORM),
        patience=int(EARLY_STOPPING_PATIENCE),
        device=str(DEVICE_NAME),
        seed=int(RANDOM_SEED),
        num_workers=int(NUM_WORKERS),
        pin_mem=bool(PIN_MEMORY),
        drop_last=bool(DROP_LAST),
        amp=bool(USE_AMP),
        deterministic=bool(USE_DETERMINISTIC_ALGORITHMS),
        log_interval=int(LOG_INTERVAL),
        tensorboard=bool(ENABLE_TENSORBOARD),
    )


def validate_args(args: SimpleNamespace) -> None:
    if not Path(args.data_path).is_file():
        raise FileNotFoundError(f"dataset not found: {args.data_path}")
    if args.use_precomputed_text_embeddings:
        if args.text_embedding_cache_path is None or not Path(args.text_embedding_cache_path).is_file():
            raise FileNotFoundError(
                f"text embedding cache not found: {args.text_embedding_cache_path}"
            )
        if not args.freeze_text_encoder:
            raise ValueError("precomputed embeddings require FREEZE_TEXT_ENCODER=True")
    if args.depth != 6:
        raise ValueError(
            f"This v2 is designed around the uploaded depth=6 Base; got depth={args.depth}"
        )
    if args.epochs < 1:
        raise ValueError("FINE_TUNE_EPOCHS must be >= 1")
    if args.batch_size < 1 or args.accumulation_steps < 1:
        raise ValueError("BATCH_SIZE and GRADIENT_ACCUMULATION_STEPS must be >= 1")

    min_lr_set = int(args.min_lr is not None) + int(args.min_lr_ratio is not None)
    if min_lr_set != 1:
        raise ValueError(
            "Set exactly one of MINIMUM_LR and MINIMUM_LR_RATIO; set the other to None"
        )
    if args.min_lr is not None and args.min_lr < 0:
        raise ValueError("MINIMUM_LR must be >= 0")
    if args.min_lr_ratio is not None and not (0 <= args.min_lr_ratio < 1):
        raise ValueError("MINIMUM_LR_RATIO must satisfy 0 <= ratio < 1")
    if args.lr is not None and args.lr <= 0:
        raise ValueError("ABSOLUTE_LR must be > 0 or None")
    if args.blr <= 0:
        raise ValueError("BASE_LR must be > 0")
    if not (0 <= args.warmup_ratio < 1):
        raise ValueError("WARMUP_RATIO must satisfy 0 <= ratio < 1")
    if args.warmup_steps_override is not None and args.warmup_steps_override < 0:
        raise ValueError("WARMUP_STEPS_OVERRIDE must be >= 0 or None")
    if args.warmup_init_lr <= 0:
        raise ValueError("WARMUP_INITIAL_LR must be > 0")
    if LOSS_NAME.lower() != "mse":
        raise ValueError("Controlled v2 main experiment currently requires LOSS_NAME='mse'")
    if ROUTER_K < 3:
        raise ValueError("ROUTER_K must be >= 3 for Grouped-3 experts")
    if N_STANDALONE_HARD_EXPERTS != 2:
        raise ValueError("This Grouped-3 implementation expects exactly two hard standalone experts")
    if USE_DISJOINT_VALIDATION_ROLES and not (0 < DIFFICULTY_SELECTION_FRACTION < 1):
        raise ValueError("DIFFICULTY_SELECTION_FRACTION must be in (0,1)")


def get_text_hidden(args: SimpleNamespace) -> int:
    return int(get_text_hidden_size(
        args.text_model_name,
        local_files_only=not args.allow_remote_model,
    ))


def build_model(
    args: SimpleNamespace,
    device: torch.device,
    *,
    text_hidden_size: Optional[int] = None,
) -> torch.nn.Module:
    hidden = get_text_hidden(args) if text_hidden_size is None else int(text_hidden_size)
    if args.use_precomputed_text_embeddings and args.text_embedding_cache_path:
        cache = np.load(args.text_embedding_cache_path, mmap_mode="r")
        if cache.ndim != 2:
            raise ValueError(f"text embedding cache must be [N,D], got {cache.shape}")
        if int(cache.shape[1]) != hidden:
            raise ValueError(
                f"text cache D={cache.shape[1]} != text hidden size={hidden}"
            )
        del cache

    model = TextGuidedFusiformer(
        targets=[args.target],
        depth=args.depth,
        edge_input_dim=args.edge_input_dim,
        triplet_input_dim=args.triplet_input_dim,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        text_dim=hidden,
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
    logger: Optional[logging.Logger] = None,
) -> None:
    state, excluded = extract_model_state(checkpoint)
    incompatible = model.load_state_dict(state, strict=False)
    missing = [
        name for name in incompatible.missing_keys
        if not any(name.startswith(prefix) for prefix in excluded)
    ]
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            "checkpoint/model mismatch; "
            f"missing={missing[:30]}, unexpected={unexpected[:30]}"
        )
    if logger:
        logger.info("loaded model tensors: %d", len(state))


def validate_checkpoint_architecture(
    checkpoint: Any,
    args: SimpleNamespace,
    logger: logging.Logger,
) -> None:
    if not isinstance(checkpoint, dict):
        return
    cfg = checkpoint.get("training_config") or {}
    expected = {
        "depth": args.depth,
        "edge_input_dim": args.edge_input_dim,
        "triplet_input_dim": args.triplet_input_dim,
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
        "text_proj_dim": args.text_proj_dim,
    }
    mismatch = []
    for key, value in expected.items():
        if key in cfg and cfg[key] is not None and int(cfg[key]) != int(value):
            mismatch.append((key, cfg[key], value))
    if mismatch:
        raise RuntimeError(f"Base checkpoint architecture mismatch: {mismatch}")
    logger.info("Base checkpoint architecture validation passed")


# =============================================================================
# 3. Dataset / DataLoader -- mirrors Base training program
# =============================================================================

def build_datasets(args: SimpleNamespace, logger: logging.Logger):
    transform = Compose([
        AddNodeFeature(),
        AddEdgeFeature(),
        AddAngleFeature(),
        GraphCollectingCached(["graph", "line_graph"], [args.target]),
    ])
    train_dataset, val_dataset, test_dataset = get_dataset(
        data_path=args.data_path,
        ratio_train_val_test=args.split_ratio,
        n_train_val_test=args.split_counts,
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
            if args.use_precomputed_text_embeddings else None
        ),
        text_embedding_meta_path=args.text_embedding_meta_path,
    )
    if val_dataset is None or test_dataset is None:
        raise RuntimeError("This Expert experiment requires train/val/test datasets")
    logger.info(
        "Base split sizes | train=%d val=%d test=%d",
        len(train_dataset), len(val_dataset), len(test_dataset),
    )
    return train_dataset, val_dataset, test_dataset


def make_loader(
    dataset: Any,
    args: SimpleNamespace,
    *,
    shuffle: bool,
    batch_size: Optional[int] = None,
    collect_source: Any = None,
    seed: Optional[int] = None,
):
    bs = int(batch_size or args.batch_size)
    generator = torch.Generator()
    generator.manual_seed(args.seed if seed is None else int(seed))
    kwargs = {
        "batch_size": bs,
        "shuffle": shuffle,
        "drop_last": bool(args.drop_last and shuffle and len(dataset) >= bs),
        "num_workers": args.num_workers,
        "pin_memory": args.pin_mem and args.device.startswith("cuda"),
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
    collate_owner = collect_source if collect_source is not None else dataset
    return torch.utils.data.DataLoader(
        dataset,
        collate_fn=collate_owner.collect(),
        **kwargs,
    )


def make_subset(dataset: Any, positions: np.ndarray | list[int]):
    return torch.utils.data.Subset(dataset, [int(x) for x in positions])


def try_original_indices(dataset: Any) -> Optional[np.ndarray]:
    for attr in ("original_indices", "indices", "original_index"):
        if hasattr(dataset, attr):
            value = getattr(dataset, attr)
            if value is None:
                continue
            arr = np.asarray(value, dtype=int).reshape(-1)
            if len(arr) == len(dataset):
                return arr
    return None


# =============================================================================
# 4. graph_out extraction and embedded KMeans Router
# =============================================================================

def clone_inputs_without_text(inputs: Any) -> Any:
    if not isinstance(inputs, dict):
        return inputs
    copied = dict(inputs)
    copied["text"] = None
    copied["text_emb"] = None
    return copied


def graph_only_inputs(inputs: Any) -> Any:
    if not isinstance(inputs, dict):
        return inputs
    copied = dict(inputs)
    copied["text"] = None
    copied["text_emb"] = None
    copied["text_mode"] = "bypass"
    return copied


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


def _first_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (list, tuple)):
        for item in output:
            if torch.is_tensor(item):
                return item
    raise TypeError(f"Unsupported readout hook output type: {type(output)}")


@torch.no_grad()
def extract_graph_out(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    amp_enabled: bool,
    logger: logging.Logger,
    split_name: str,
) -> np.ndarray:
    module = model.module if hasattr(model, "module") else model
    if not hasattr(module, "readout"):
        raise AttributeError("TextGuidedFusiformer has no readout module")

    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        captured.append(_first_tensor(output).detach().float().cpu())

    handle = module.readout.register_forward_hook(hook)
    chunks: list[np.ndarray] = []
    model.eval()
    try:
        for batch_i, (inputs, _targets) in enumerate(loader):
            captured.clear()
            inputs = to_device(inputs, device)
            inputs = graph_only_inputs(inputs)
            with autocast_context(device, amp_enabled):
                _ = model(inputs)
            if len(captured) != 1:
                raise RuntimeError(
                    f"Expected one readout hook output; got {len(captured)}"
                )
            chunks.append(captured[0].numpy())
            if (batch_i + 1) % 100 == 0 or batch_i + 1 == len(loader):
                logger.info(
                    "graph_out %s | batch %d/%d",
                    split_name, batch_i + 1, len(loader),
                )
    finally:
        handle.remove()

    graph_out = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    if graph_out.ndim != 2:
        raise RuntimeError(f"graph_out must be [N,D], got {graph_out.shape}")
    logger.info("graph_out %s shape=%s", split_name, graph_out.shape)
    return graph_out


class EmbeddedKMeansRouter(torch.nn.Module):
    """PyTorch-only frozen KMeans router.

    centers and cluster_to_expert are buffers: saved in state_dict, moved by
    .to(device), and never optimized by AdamW.
    """

    def __init__(
        self,
        centers: np.ndarray | torch.Tensor,
        cluster_to_expert: list[int] | np.ndarray | torch.Tensor,
        *,
        l2_normalize: bool = True,
    ) -> None:
        super().__init__()
        centers_t = torch.as_tensor(centers, dtype=torch.float32)
        mapping_t = torch.as_tensor(cluster_to_expert, dtype=torch.long)
        if centers_t.ndim != 2:
            raise ValueError(f"centers must be [K,D], got {tuple(centers_t.shape)}")
        if mapping_t.ndim != 1 or mapping_t.numel() != centers_t.shape[0]:
            raise ValueError("cluster_to_expert must have one entry per cluster")
        self.register_buffer("centers", centers_t)
        self.register_buffer("cluster_to_expert", mapping_t)
        self.l2_normalize = bool(l2_normalize)

    def forward(self, graph_out: torch.Tensor) -> dict[str, torch.Tensor]:
        z = graph_out.float()
        if self.l2_normalize:
            z = F.normalize(z, p=2, dim=-1, eps=1e-12)
        # squared Euclidean distance in the same feature space as sklearn KMeans
        dist2 = (
            (z * z).sum(dim=1, keepdim=True)
            - 2.0 * z @ self.centers.t()
            + (self.centers * self.centers).sum(dim=1).unsqueeze(0)
        )
        cluster_id = torch.argmin(dist2, dim=1)
        expert_id = self.cluster_to_expert[cluster_id]
        two = torch.topk(dist2, k=min(2, dist2.shape[1]), largest=False, dim=1)
        nearest_dist2 = two.values[:, 0]
        if dist2.shape[1] >= 2:
            second_dist2 = two.values[:, 1]
            margin = second_dist2 - nearest_dist2
            confidence = margin / (second_dist2.abs() + 1e-12)
        else:
            second_dist2 = nearest_dist2.clone()
            margin = torch.zeros_like(nearest_dist2)
            confidence = torch.ones_like(nearest_dist2)
        return {
            "cluster_id": cluster_id,
            "expert_id": expert_id,
            "dist2": dist2,
            "nearest_dist2": nearest_dist2,
            "second_dist2": second_dist2,
            "margin": margin,
            "confidence": confidence,
        }


def router_cache_dir(args: SimpleNamespace, base_weights_path: Path) -> Path:
    root = resolve_path(OUTPUT_ROOT)
    assert root is not None
    cache = root / "router_embedding_cache"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def cached_or_extract_graph_out(
    *,
    split_name: str,
    dataset: Any,
    base_model: torch.nn.Module,
    args: SimpleNamespace,
    device: torch.device,
    base_weights_path: Path,
    logger: logging.Logger,
) -> np.ndarray:
    cache_dir = router_cache_dir(args, base_weights_path)
    array_path = cache_dir / f"{split_name}_graph_out.npy"
    meta_path = cache_dir / f"{split_name}_meta.json"
    expected_meta = {
        "script_family": "grouped3_expert_v2_router_cache",
        "data": file_signature(Path(args.data_path)),
        "base_weights": file_signature(base_weights_path),
        "split_name": split_name,
        "split_size": len(dataset),
        "random_seed": args.seed,
        "router_mode": "graph_only_readout",
    }
    if REUSE_ROUTER_EMBEDDING_CACHE and array_path.is_file() and meta_path.is_file():
        try:
            old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if old_meta == expected_meta:
                logger.info("reuse %s graph_out cache: %s", split_name, array_path)
                return np.load(array_path)
        except Exception:
            pass

    loader = make_loader(
        dataset,
        args,
        shuffle=False,
        batch_size=ROUTER_BATCH_SIZE,
        seed=args.seed,
    )
    graph_out = extract_graph_out(
        base_model,
        loader,
        device,
        args.amp and device.type == "cuda",
        logger,
        split_name,
    )
    np.save(array_path, graph_out)
    save_json(meta_path, expected_meta)
    original = try_original_indices(dataset)
    if original is not None:
        np.save(cache_dir / f"{split_name}_original_indices.npy", original)
    return graph_out


def prepare_router_features(graph_out: np.ndarray) -> np.ndarray:
    if ROUTER_L2_NORMALIZE:
        return sklearn_l2_normalize(graph_out, norm="l2", axis=1).astype(np.float32)
    return np.asarray(graph_out, dtype=np.float32)


def fit_train_only_kmeans(
    train_graph_out: np.ndarray,
    run_dir: Path,
    logger: logging.Logger,
) -> tuple[KMeans, np.ndarray]:
    X_train = prepare_router_features(train_graph_out)
    kmeans = KMeans(
        n_clusters=ROUTER_K,
        n_init=ROUTER_N_INIT,
        max_iter=ROUTER_MAX_ITER,
        tol=ROUTER_TOL,
        random_state=ROUTER_RANDOM_STATE,
    )
    train_cluster = kmeans.fit_predict(X_train).astype(int)
    router_dir = run_dir / "router"
    router_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(kmeans, router_dir / "kmeans_train_only_diagnostic.joblib")
    pd.DataFrame(kmeans.cluster_centers_).to_csv(
        router_dir / "kmeans_centers.csv",
        index_label="cluster_id",
        encoding="utf-8-sig",
    )
    logger.info("TRAIN-only KMeans fitted | K=%d inertia=%.6f", ROUTER_K, kmeans.inertia_)
    return kmeans, train_cluster


def predict_sklearn_cluster(kmeans: KMeans, graph_out: np.ndarray) -> np.ndarray:
    return kmeans.predict(prepare_router_features(graph_out)).astype(int)



# =============================================================================
# 4B. Router diagnostics / visualization (analysis only; no model selection)
# =============================================================================

def deterministic_sample_positions(n: int, max_points: int, seed: int) -> np.ndarray:
    n = int(n)
    max_points = int(max_points)
    if n <= max_points:
        return np.arange(n, dtype=int)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_points, replace=False).astype(int))


def targets_from_raw_json_for_dataset(
    dataset: Any,
    raw_data: Optional[list[dict[str, Any]]],
    target_name: str,
) -> Optional[np.ndarray]:
    if raw_data is None:
        return None
    original = try_original_indices(dataset)
    if original is None:
        return None
    try:
        values = [float(raw_data[int(i)][target_name]) for i in original]
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    return np.asarray(values, dtype=float)


def route_graph_out_numpy(
    router: EmbeddedKMeansRouter,
    graph_out: np.ndarray,
    device: torch.device,
    batch_size: int = 8192,
) -> dict[str, np.ndarray]:
    chunks: dict[str, list[np.ndarray]] = {
        "cluster_id": [],
        "expert_id": [],
        "nearest_dist2": [],
        "second_dist2": [],
        "margin": [],
        "confidence": [],
    }
    router = router.to(device)
    router.eval()
    with torch.no_grad():
        for start in range(0, len(graph_out), batch_size):
            x = torch.from_numpy(np.asarray(graph_out[start:start + batch_size], dtype=np.float32)).to(device)
            out = router(x)
            for key in chunks:
                chunks[key].append(out[key].detach().cpu().numpy())
    return {key: np.concatenate(parts) for key, parts in chunks.items()}


def compute_router_quality_metrics(
    train_graph_out: np.ndarray,
    train_cluster: np.ndarray,
    kmeans: KMeans,
    run_dir: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    X = prepare_router_features(train_graph_out)
    pos = deterministic_sample_positions(
        len(X),
        min(ROUTER_QUALITY_SAMPLE_SIZE, len(X)),
        ROUTER_DIAGNOSTICS_RANDOM_STATE,
    )
    Xs = X[pos]
    ys = np.asarray(train_cluster, dtype=int)[pos]
    metrics: dict[str, Any] = {
        "fit_split": "train_only",
        "K": int(ROUTER_K),
        "n_train": int(len(X)),
        "quality_sample_size": int(len(Xs)),
        "inertia_full_train": float(kmeans.inertia_),
        "l2_normalize": bool(ROUTER_L2_NORMALIZE),
    }
    if len(np.unique(ys)) >= 2 and len(Xs) > len(np.unique(ys)):
        metrics["silhouette_sample"] = float(silhouette_score(Xs, ys, metric="euclidean"))
        metrics["calinski_harabasz_sample"] = float(calinski_harabasz_score(Xs, ys))
        metrics["davies_bouldin_sample"] = float(davies_bouldin_score(Xs, ys))
    else:
        metrics["silhouette_sample"] = None
        metrics["calinski_harabasz_sample"] = None
        metrics["davies_bouldin_sample"] = None
    save_json(run_dir / "router" / "kmeans_quality_metrics.json", metrics)
    logger.info(
        "router quality | sample=%d | silhouette=%s | CH=%s | DB=%s | inertia=%.6f",
        len(Xs),
        "NA" if metrics["silhouette_sample"] is None else f"{metrics['silhouette_sample']:.6f}",
        "NA" if metrics["calinski_harabasz_sample"] is None else f"{metrics['calinski_harabasz_sample']:.3f}",
        "NA" if metrics["davies_bouldin_sample"] is None else f"{metrics['davies_bouldin_sample']:.6f}",
        float(kmeans.inertia_),
    )
    return metrics


def _safe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def run_router_diagnostics(
    *,
    train_graph_out: np.ndarray,
    val_graph_out: np.ndarray,
    train_cluster: np.ndarray,
    val_cluster: np.ndarray,
    y_val: np.ndarray,
    train_dataset: Any,
    val_dataset: Any,
    raw_data: Optional[list[dict[str, Any]]],
    kmeans: KMeans,
    embedded_router: EmbeddedKMeansRouter,
    cluster_to_expert: np.ndarray,
    difficulty_df: pd.DataFrame,
    run_dir: Path,
    device: torch.device,
    logger: logging.Logger,
) -> dict[str, Any]:
    if not ROUTER_DIAGNOSTICS_ENABLED:
        return {"enabled": False, "figures": []}

    router_dir = run_dir / "router"
    fig_dir = router_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    quality = compute_router_quality_metrics(
        train_graph_out, train_cluster, kmeans, run_dir, logger
    )
    train_route = route_graph_out_numpy(embedded_router, train_graph_out, device)
    val_route = route_graph_out_numpy(embedded_router, val_graph_out, device)
    if not np.array_equal(train_route["cluster_id"].astype(int), train_cluster.astype(int)):
        raise RuntimeError("router diagnostics train labels differ from KMeans")
    if not np.array_equal(val_route["cluster_id"].astype(int), val_cluster.astype(int)):
        raise RuntimeError("router diagnostics val labels differ from KMeans")

    y_train = targets_from_raw_json_for_dataset(train_dataset, raw_data, TARGET_NAME)
    y_val_json = targets_from_raw_json_for_dataset(val_dataset, raw_data, TARGET_NAME)
    if y_val_json is not None and len(y_val_json) == len(y_val):
        max_diff = float(np.max(np.abs(y_val_json - y_val)))
        if max_diff > 1e-5:
            logger.warning("VAL target/raw-json alignment max diff = %.6g", max_diff)

    def build_diag_frame(split: str, clusters: np.ndarray, route: dict[str, np.ndarray], y: Optional[np.ndarray]):
        frame = pd.DataFrame({
            "position": np.arange(len(clusters), dtype=int),
            "cluster_id": clusters.astype(int),
            "cluster": [f"R{x}" for x in clusters],
            "expert_id": cluster_to_expert[clusters].astype(int),
            "expert_name": [EXPERT_ID_TO_NAME[int(x)] for x in cluster_to_expert[clusters]],
            "nearest_dist2": route["nearest_dist2"].astype(float),
            "second_dist2": route["second_dist2"].astype(float),
            "margin": route["margin"].astype(float),
            "confidence": route["confidence"].astype(float),
        })
        if y is not None and len(y) == len(frame):
            frame[TARGET_NAME] = np.asarray(y, dtype=float)
        frame.to_csv(router_dir / f"{split}_router_diagnostics.csv", index=False, encoding="utf-8-sig")
        return frame

    train_df = build_diag_frame("train", train_cluster, train_route, y_train)
    val_df = build_diag_frame("val", val_cluster, val_route, y_val)

    # Numerical router-cluster statistics for downstream analysis/reporting.
    cluster_stat_rows = []
    for split_name, frame in (("train", train_df), ("val", val_df)):
        for g in range(ROUTER_K):
            sub = frame[frame["cluster_id"] == g]
            row = {
                "split": split_name,
                "cluster_id": g,
                "cluster": f"R{g}",
                "expert_id": int(cluster_to_expert[g]),
                "expert_name": EXPERT_ID_TO_NAME[int(cluster_to_expert[g])],
                "n": int(len(sub)),
                "fraction": float(len(sub) / max(len(frame), 1)),
                "nearest_dist2_mean": float(sub["nearest_dist2"].mean()),
                "nearest_dist2_median": float(sub["nearest_dist2"].median()),
                "confidence_mean": float(sub["confidence"].mean()),
                "confidence_median": float(sub["confidence"].median()),
            }
            if TARGET_NAME in sub.columns and len(sub):
                values = sub[TARGET_NAME].to_numpy(float)
                row.update({
                    "e_form_mean": float(np.mean(values)),
                    "e_form_std": float(np.std(values)),
                    "e_form_median": float(np.median(values)),
                    "e_form_q25": float(np.percentile(values, 25)),
                    "e_form_q75": float(np.percentile(values, 75)),
                    "e_form_min": float(np.min(values)),
                    "e_form_max": float(np.max(values)),
                })
            cluster_stat_rows.append(row)
    pd.DataFrame(cluster_stat_rows).to_csv(
        router_dir / "router_cluster_statistics.csv", index=False, encoding="utf-8-sig"
    )

    centers = np.asarray(kmeans.cluster_centers_, dtype=np.float64)
    center_diff = centers[:, None, :] - centers[None, :, :]
    center_dist = np.sqrt(np.sum(center_diff * center_diff, axis=-1))
    pd.DataFrame(
        center_dist,
        index=[f"R{g}" for g in range(ROUTER_K)],
        columns=[f"R{g}" for g in range(ROUTER_K)],
    ).to_csv(router_dir / "centroid_pairwise_distances.csv", encoding="utf-8-sig")

    X_train = prepare_router_features(train_graph_out)
    X_val = prepare_router_features(val_graph_out)
    fit_pos = deterministic_sample_positions(
        len(X_train),
        min(ROUTER_PCA_FIT_SAMPLE_SIZE, len(X_train)),
        ROUTER_DIAGNOSTICS_RANDOM_STATE + 1,
    )
    pca = PCA(n_components=2, svd_solver="randomized", random_state=ROUTER_DIAGNOSTICS_RANDOM_STATE)
    pca.fit(X_train[fit_pos])
    train_plot_pos = deterministic_sample_positions(
        len(X_train), ROUTER_MAX_PLOT_POINTS, ROUTER_DIAGNOSTICS_RANDOM_STATE + 2
    )
    val_plot_pos = deterministic_sample_positions(
        len(X_val), ROUTER_MAX_PLOT_POINTS, ROUTER_DIAGNOSTICS_RANDOM_STATE + 3
    )
    train_xy = pca.transform(X_train[train_plot_pos])
    val_xy = pca.transform(X_val[val_plot_pos])
    center_xy = pca.transform(np.asarray(kmeans.cluster_centers_, dtype=np.float32))

    pca_meta = {
        "fit_split": "train_only",
        "fit_sample_size": int(len(fit_pos)),
        "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "explained_variance_ratio_sum_2d": float(np.sum(pca.explained_variance_ratio_)),
        "visualization_only": True,
    }
    save_json(router_dir / "pca_visualization_meta.json", pca_meta)
    joblib.dump(pca, router_dir / "pca_visualization_train_only.joblib")
    pd.DataFrame({
        "cluster_id": np.arange(ROUTER_K, dtype=int),
        "PCA1": center_xy[:, 0],
        "PCA2": center_xy[:, 1],
    }).to_csv(router_dir / "kmeans_centers_pca2d.csv", index=False, encoding="utf-8-sig")

    if ROUTER_SAVE_PCA_COORDINATES:
        tcoord = train_df.iloc[train_plot_pos].copy()
        tcoord["PCA1"] = train_xy[:, 0]
        tcoord["PCA2"] = train_xy[:, 1]
        tcoord.to_csv(router_dir / "train_pca_coordinates.csv", index=False, encoding="utf-8-sig")
        vcoord = val_df.iloc[val_plot_pos].copy()
        vcoord["PCA1"] = val_xy[:, 0]
        vcoord["PCA2"] = val_xy[:, 1]
        vcoord.to_csv(router_dir / "val_pca_coordinates.csv", index=False, encoding="utf-8-sig")

    plt = _safe_import_matplotlib()
    saved: list[str] = []
    if plt is not None and SAVE_PLOTS:
        def savefig(fig, name: str):
            path = fig_dir / name
            fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
            plt.close(fig)
            saved.append(str(path))

        # Train PCA clusters + centroids.
        fig, ax = plt.subplots(figsize=(8.5, 6.3))
        sc = ax.scatter(train_xy[:, 0], train_xy[:, 1], c=train_cluster[train_plot_pos], s=7, alpha=0.45)
        ax.scatter(center_xy[:, 0], center_xy[:, 1], marker="X", s=150)
        for g, (x, y) in enumerate(center_xy):
            ax.annotate(f"R{g}", (x, y), xytext=(5, 5), textcoords="offset points")
        ax.set_xlabel("PCA1")
        ax.set_ylabel("PCA2")
        ax.set_title("TRAIN-only router latent PCA by KMeans cluster")
        fig.colorbar(sc, ax=ax, label="Router cluster ID")
        fig.tight_layout()
        savefig(fig, "router_train_pca_clusters.png")

        # Validation projection under train-fitted PCA/router.
        fig, ax = plt.subplots(figsize=(8.5, 6.3))
        sc = ax.scatter(val_xy[:, 0], val_xy[:, 1], c=val_cluster[val_plot_pos], s=9, alpha=0.55)
        ax.scatter(center_xy[:, 0], center_xy[:, 1], marker="X", s=150)
        ax.set_xlabel("PCA1")
        ax.set_ylabel("PCA2")
        ax.set_title("Validation samples routed by TRAIN-only KMeans")
        fig.colorbar(sc, ax=ax, label="Router cluster ID")
        fig.tight_layout()
        savefig(fig, "router_val_pca_clusters.png")

        # Validation e_form on same PCA.
        fig, ax = plt.subplots(figsize=(8.5, 6.3))
        sc = ax.scatter(val_xy[:, 0], val_xy[:, 1], c=np.asarray(y_val)[val_plot_pos], s=9, alpha=0.55)
        ax.set_xlabel("PCA1")
        ax.set_ylabel("PCA2")
        ax.set_title("Validation latent PCA colored by e_form")
        fig.colorbar(sc, ax=ax, label="e_form (eV/atom)")
        fig.tight_layout()
        savefig(fig, "router_val_pca_eform.png")

        if y_train is not None and len(y_train) == len(train_cluster):
            fig, ax = plt.subplots(figsize=(8.5, 6.3))
            sc = ax.scatter(train_xy[:, 0], train_xy[:, 1], c=y_train[train_plot_pos], s=7, alpha=0.45)
            ax.set_xlabel("PCA1")
            ax.set_ylabel("PCA2")
            ax.set_title("TRAIN latent PCA colored by e_form")
            fig.colorbar(sc, ax=ax, label="e_form (eV/atom)")
            fig.tight_layout()
            savefig(fig, "router_train_pca_eform.png")

        # Cluster sizes.
        train_counts = np.asarray([np.sum(train_cluster == g) for g in range(ROUTER_K)])
        val_counts = np.asarray([np.sum(val_cluster == g) for g in range(ROUTER_K)])
        x = np.arange(ROUTER_K)
        width = 0.38
        fig, ax = plt.subplots(figsize=(8.0, 5.2))
        ax.bar(x - width / 2, train_counts, width, label="Train")
        ax.bar(x + width / 2, val_counts, width, label="Validation")
        ax.set_xticks(x, [f"R{g}" for g in range(ROUTER_K)])
        ax.set_ylabel("Samples")
        ax.set_title("Router cluster size distribution")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        savefig(fig, "router_cluster_sizes.png")

        # e_form distribution by cluster, validation only (always available).
        grouped_y = [np.asarray(y_val)[val_cluster == g] for g in range(ROUTER_K)]
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ax.boxplot(grouped_y, labels=[f"R{g}" for g in range(ROUTER_K)], showfliers=False)
        ax.set_xlabel("Router cluster")
        ax.set_ylabel("e_form (eV/atom)")
        ax.set_title("Validation e_form distribution by TRAIN-only router cluster")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        savefig(fig, "router_val_eform_by_cluster_boxplot.png")

        # Validation difficulty ranking used to allocate standalone Experts.
        ddf = difficulty_df.sort_values("cluster_id")
        fig, ax = plt.subplots(figsize=(8.0, 5.2))
        ax.bar(ddf["cluster"], ddf[DIFFICULTY_METRIC])
        ax.set_ylabel(DIFFICULTY_METRIC.upper() + " (eV/atom)")
        ax.set_title("Validation Base difficulty used for Expert grouping")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        savefig(fig, "validation_cluster_difficulty.png")

        # Router distance / confidence distributions.
        train_distance = [train_df.loc[train_df.cluster_id == g, "nearest_dist2"].to_numpy() for g in range(ROUTER_K)]
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ax.boxplot(train_distance, labels=[f"R{g}" for g in range(ROUTER_K)], showfliers=False)
        ax.set_xlabel("Router cluster")
        ax.set_ylabel("Squared distance to assigned centroid")
        ax.set_title("TRAIN routing distance by cluster")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        savefig(fig, "router_distance_by_cluster.png")

        train_conf = [train_df.loc[train_df.cluster_id == g, "confidence"].to_numpy() for g in range(ROUTER_K)]
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ax.boxplot(train_conf, labels=[f"R{g}" for g in range(ROUTER_K)], showfliers=False)
        ax.set_xlabel("Router cluster")
        ax.set_ylabel("Routing confidence")
        ax.set_title("TRAIN routing confidence by cluster")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        savefig(fig, "router_confidence_by_cluster.png")

        fig, ax = plt.subplots(figsize=(8.5, 6.3))
        sc = ax.scatter(train_xy[:, 0], train_xy[:, 1], c=train_df["confidence"].to_numpy()[train_plot_pos], s=7, alpha=0.45)
        ax.set_xlabel("PCA1")
        ax.set_ylabel("PCA2")
        ax.set_title("TRAIN latent PCA colored by routing confidence")
        fig.colorbar(sc, ax=ax, label="Routing confidence")
        fig.tight_layout()
        savefig(fig, "router_train_pca_confidence.png")

        fig, ax = plt.subplots(figsize=(8.5, 6.3))
        sc = ax.scatter(train_xy[:, 0], train_xy[:, 1], c=train_df["nearest_dist2"].to_numpy()[train_plot_pos], s=7, alpha=0.45)
        ax.set_xlabel("PCA1")
        ax.set_ylabel("PCA2")
        ax.set_title("TRAIN latent PCA colored by centroid distance")
        fig.colorbar(sc, ax=ax, label="Squared distance to assigned centroid")
        fig.tight_layout()
        savefig(fig, "router_train_pca_centroid_distance.png")

    summary = {
        "enabled": True,
        "quality_metrics": quality,
        "pca": pca_meta,
        "train_distance_mean": float(train_df["nearest_dist2"].mean()),
        "train_confidence_mean": float(train_df["confidence"].mean()),
        "val_distance_mean": float(val_df["nearest_dist2"].mean()),
        "val_confidence_mean": float(val_df["confidence"].mean()),
        "figures": saved,
    }
    save_json(router_dir / "router_diagnostics_summary.json", summary)
    for path in saved:
        logger.info("saved router diagnostic figure: %s", path)
    return summary

# =============================================================================
# 5. Metrics / validation difficulty selection
# =============================================================================

def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(y_true) == 0:
        raise ValueError("metric_dict received zero samples")
    err = y_pred - y_true
    ae = np.abs(err)
    mse = float(np.mean(err * err))
    return {
        "n": int(len(y_true)),
        "mse": mse,
        "mae": float(np.mean(ae)),
        "rmse": float(math.sqrt(mse)),
        "bias": float(np.mean(err)),
        "median_abs_error": float(np.median(ae)),
        "p90_abs_error": float(np.percentile(ae, 90)),
        "p95_abs_error": float(np.percentile(ae, 95)),
        "p99_abs_error": float(np.percentile(ae, 99)),
        "max_abs_error": float(np.max(ae)),
        "r2": (
            float(r2_score(y_true, y_pred))
            if len(y_true) > 1 and np.var(y_true) > 1e-15
            else float("nan")
        ),
    }


@torch.no_grad()
def predict_loader(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    use_text: bool,
    amp_enabled: bool,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for inputs, targets in loader:
        inputs = to_device(inputs, device)
        targets = normalize_targets(targets, device)
        if not use_text:
            inputs = clone_inputs_without_text(inputs)
        with autocast_context(device, amp_enabled):
            outputs = normalize_outputs(model(inputs))
        ys.append(targets.detach().float().cpu().numpy().reshape(-1))
        ps.append(outputs.detach().float().cpu().numpy().reshape(-1))
    return np.concatenate(ys), np.concatenate(ps)


def split_validation_roles(
    val_cluster: np.ndarray,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    n = len(val_cluster)
    all_positions = np.arange(n, dtype=int)
    if not USE_DISJOINT_VALIDATION_ROLES:
        role_df = pd.DataFrame({
            "val_position": all_positions,
            "cluster_id": val_cluster,
            "difficulty_selection": 1,
            "expert_early_stop": 1,
        })
        return all_positions, all_positions, role_df

    rng = np.random.default_rng(VALIDATION_ROLE_SEED)
    difficulty_positions: list[int] = []
    early_stop_positions: list[int] = []
    role_rows = []
    for cluster_id in range(ROUTER_K):
        positions = np.where(val_cluster == cluster_id)[0].astype(int)
        if len(positions) < 2:
            raise RuntimeError(
                f"Validation cluster R{cluster_id} has only {len(positions)} sample(s); "
                "cannot split validation roles."
            )
        shuffled = positions.copy()
        rng.shuffle(shuffled)
        n_select = int(round(len(shuffled) * DIFFICULTY_SELECTION_FRACTION))
        n_select = min(max(n_select, 1), len(shuffled) - 1)
        select_set = set(int(x) for x in shuffled[:n_select])
        stop_set = set(int(x) for x in shuffled[n_select:])
        difficulty_positions.extend(sorted(select_set))
        early_stop_positions.extend(sorted(stop_set))
        for pos in positions:
            role_rows.append({
                "val_position": int(pos),
                "cluster_id": int(cluster_id),
                "difficulty_selection": int(int(pos) in select_set),
                "expert_early_stop": int(int(pos) in stop_set),
            })
    difficulty = np.asarray(sorted(difficulty_positions), dtype=int)
    early = np.asarray(sorted(early_stop_positions), dtype=int)
    if set(difficulty.tolist()) & set(early.tolist()):
        raise RuntimeError("validation role split overlaps")
    if len(difficulty) + len(early) != n:
        raise RuntimeError("validation role split does not cover all validation samples")
    logger.info(
        "validation roles | difficulty_selection=%d expert_early_stop=%d",
        len(difficulty), len(early),
    )
    return difficulty, early, pd.DataFrame(role_rows).sort_values("val_position")


def select_expert_mapping_from_validation(
    *,
    val_cluster: np.ndarray,
    y_val: np.ndarray,
    p_val_base: np.ndarray,
    difficulty_positions: np.ndarray,
    run_dir: Path,
    logger: logging.Logger,
) -> tuple[np.ndarray, dict[int, list[int]], pd.DataFrame]:
    rows = []
    difficulty_set = np.zeros(len(val_cluster), dtype=bool)
    difficulty_set[difficulty_positions] = True
    for cluster_id in range(ROUTER_K):
        positions = np.where((val_cluster == cluster_id) & difficulty_set)[0]
        if len(positions) == 0:
            raise RuntimeError(f"R{cluster_id} has no difficulty-selection validation samples")
        metrics = metric_dict(y_val[positions], p_val_base[positions])
        rows.append({"cluster_id": cluster_id, "cluster": f"R{cluster_id}", **metrics})

    difficulty_df = pd.DataFrame(rows)
    difficulty_df.to_csv(
        run_dir / "router" / "validation_cluster_difficulty.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metric_name = DIFFICULTY_METRIC.lower().strip()
    if metric_name not in {"mae", "rmse"}:
        raise ValueError("DIFFICULTY_METRIC must be 'mae' or 'rmse'")
    eligible = difficulty_df[
        difficulty_df["n"] >= MIN_DIFFICULTY_VAL_SAMPLES_PER_CLUSTER
    ].copy()
    if len(eligible) < 2:
        raise RuntimeError(
            "Fewer than 2 router clusters satisfy MIN_DIFFICULTY_VAL_SAMPLES_PER_CLUSTER="
            f"{MIN_DIFFICULTY_VAL_SAMPLES_PER_CLUSTER}."
        )
    eligible = eligible.sort_values(
        [metric_name, "rmse", "cluster_id"],
        ascending=[False, False, True],
    )
    hardest = int(eligible.iloc[0]["cluster_id"])
    second = int(eligible.iloc[1]["cluster_id"])
    shared = [g for g in range(ROUTER_K) if g not in {hardest, second}]

    # cluster id -> expert id: shared->0, hardest->1, second hardest->2
    cluster_to_expert = np.zeros(ROUTER_K, dtype=np.int64)
    cluster_to_expert[hardest] = 1
    cluster_to_expert[second] = 2
    expert_to_clusters = {
        0: shared,
        1: [hardest],
        2: [second],
    }

    logger.info(
        "difficulty-aware grouping | metric=%s | Expert-B=R%d | Expert-C=R%d | Expert-A=%s",
        metric_name, hardest, second, [f"R{x}" for x in shared],
    )
    save_json(
        run_dir / "router" / "difficulty_aware_grouping.json",
        {
            "selection_source": "validation_only",
            "difficulty_metric": metric_name,
            "use_disjoint_validation_roles": USE_DISJOINT_VALIDATION_ROLES,
            "difficulty_selection_fraction": DIFFICULTY_SELECTION_FRACTION,
            "hardest_cluster": hardest,
            "second_hardest_cluster": second,
            "shared_clusters": shared,
            "cluster_to_expert_id": cluster_to_expert.tolist(),
            "expert_id_to_name": EXPERT_ID_TO_NAME,
            "expert_to_clusters": expert_to_clusters,
        },
    )
    return cluster_to_expert, expert_to_clusters, difficulty_df


def verify_embedded_router(
    router: EmbeddedKMeansRouter,
    graph_out: np.ndarray,
    sklearn_labels: np.ndarray,
    device: torch.device,
    logger: logging.Logger,
    name: str,
) -> None:
    if not VERIFY_EMBEDDED_ROUTER:
        return
    router = router.to(device)
    predicted = []
    bs = 8192
    with torch.no_grad():
        for start in range(0, len(graph_out), bs):
            tensor = torch.from_numpy(graph_out[start:start + bs]).to(device)
            out = router(tensor)
            predicted.append(out["cluster_id"].cpu().numpy())
    pred = np.concatenate(predicted).astype(int)
    mismatch = int(np.sum(pred != sklearn_labels))
    rate = mismatch / max(len(pred), 1)
    logger.info(
        "embedded-router verification %s | mismatch=%d/%d (%.6f)",
        name, mismatch, len(pred), rate,
    )
    if rate > 1e-6:
        raise RuntimeError(
            f"Embedded router does not reproduce sklearn KMeans for {name}: rate={rate}"
        )


# =============================================================================
# 6. Selective fine-tuning / optimizer / schedule
# =============================================================================

def trainable_block_indices(depth: int, mode: str) -> list[int]:
    mode = mode.lower().strip()
    if mode == "head":
        return []
    if mode == "last1":
        return [depth - 1]
    if mode == "last2":
        return [depth - 2, depth - 1]
    if mode == "last3":
        return [depth - 3, depth - 2, depth - 1]
    if mode == "all":
        return list(range(depth))
    raise ValueError("FINETUNE_MODE must be head/last1/last2/last3/all")


def configure_finetune(
    model: torch.nn.Module,
    args: SimpleNamespace,
) -> dict[str, Any]:
    module = model.module if hasattr(model, "module") else model
    for p in module.parameters():
        p.requires_grad = False

    if not hasattr(module, "blocks"):
        raise AttributeError("Model has no 'blocks' attribute")
    if len(module.blocks) != args.depth:
        raise RuntimeError(f"expected {args.depth} blocks, found {len(module.blocks)}")

    block_ids = trainable_block_indices(args.depth, FINETUNE_MODE)
    if FINETUNE_MODE.lower() == "all":
        for p in module.parameters():
            p.requires_grad = True
        encoder = getattr(module, "text_encoder", None)
        if encoder is not None and args.freeze_text_encoder:
            for p in encoder.parameters():
                p.requires_grad = False
    else:
        for idx in block_ids:
            for p in module.blocks[idx].parameters():
                p.requires_grad = True
        if not hasattr(module, "regression_head"):
            raise AttributeError("Model has no regression_head")
        for p in module.regression_head.parameters():
            p.requires_grad = True
        if TRAIN_TEXT_PROJ:
            text_proj = getattr(module, "text_proj", None)
            if text_proj is None:
                raise AttributeError("TRAIN_TEXT_PROJ=True but model has no text_proj")
            for p in text_proj.parameters():
                p.requires_grad = True

    trainable_names = [name for name, p in module.named_parameters() if p.requires_grad]
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {
        "finetune_mode": FINETUNE_MODE,
        "trainable_block_indices_zero_based": block_ids,
        "trainable_block_numbers_one_based": [i + 1 for i in block_ids],
        "train_text_proj": TRAIN_TEXT_PROJ,
        "trainable_parameter_count": int(trainable),
        "total_parameter_count": int(total),
        "trainable_fraction": float(trainable / max(total, 1)),
        "trainable_names": trainable_names,
    }


def set_selective_train_mode(model: torch.nn.Module, args: SimpleNamespace) -> None:
    # Frozen modules stay eval() so dropout in frozen modules cannot inject noise.
    module = model.module if hasattr(model, "module") else model
    module.eval()
    if FINETUNE_MODE.lower() == "all":
        module.train()
        encoder = getattr(module, "text_encoder", None)
        if encoder is not None and args.freeze_text_encoder:
            encoder.eval()
        return
    for idx in trainable_block_indices(args.depth, FINETUNE_MODE):
        module.blocks[idx].train()
    module.regression_head.train()
    if TRAIN_TEXT_PROJ:
        module.text_proj.train()


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


def build_optimizer(
    model: torch.nn.Module,
    args: SimpleNamespace,
    actual_peak_lr: float,
) -> torch.optim.AdamW:
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
                "lr_scale": float(scales[category]) if USE_MODULE_LR_SCALES else 1.0,
                "group_name": f"{category}_{'decay' if use_decay else 'no_decay'}",
            }
        grouped[key]["params"].append(param)
    groups = list(grouped.values())
    if not groups:
        raise RuntimeError("No trainable parameters found")
    return torch.optim.AdamW(
        groups,
        lr=float(actual_peak_lr),
        betas=(args.beta1, args.beta2),
    )


def resolve_actual_peak_lr(args: SimpleNamespace) -> float:
    """Resolve peak LR using the same Stage-1 rule as the Base trainer."""
    effective_batch_size = args.batch_size * args.accumulation_steps
    lr_scaling_batch_size = (
        effective_batch_size
        if args.scale_lr_with_accumulation
        else args.batch_size
    )
    actual_peak_lr = (
        float(args.lr)
        if args.lr is not None
        else float(args.blr) * lr_scaling_batch_size / 256.0
    )
    return float(actual_peak_lr)


def resolve_minimum_lr(args: SimpleNamespace, peak_lr: float) -> float:
    """Resolve the Stage-1 LR floor exactly as in the Base trainer."""
    if args.min_lr is not None:
        return float(args.min_lr)
    assert args.min_lr_ratio is not None
    return float(peak_lr) * float(args.min_lr_ratio)


def resolve_schedule_steps(
    num_batches_per_epoch: int,
    total_epochs_value: int,
    accumulation_steps: int,
    warmup_ratio_value: float,
    warmup_steps_override_value: Optional[int],
) -> tuple[int, int, int]:
    """Convert epoch settings to optimizer-update steps, matching Base training."""
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
            f"Invalid warmup_steps={warmup_steps}; "
            f"total_update_steps={total_update_steps}"
        )
    return updates_per_epoch, total_update_steps, warmup_steps


def scheduled_learning_rate(
    update_step: int,
    peak_lr: float,
    min_lr: float,
    warmup_init_lr: float,
    warmup_steps: int,
    total_update_steps: int,
) -> float:
    """Single-stage Base LR: linear warmup followed by half-cosine decay."""
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


def apply_lr(
    optimizer: torch.optim.Optimizer,
    update_step: int,
    peak_lr: float,
    min_lr: float,
    warmup_init_lr: float,
    warmup_steps: int,
    total_update_steps: int,
) -> float:
    """Apply the single Stage-1 LR to all optimizer groups with optional scales."""
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


def save_lr_schedule_config(
    expert_dir: Path,
    *,
    args: SimpleNamespace,
    actual_peak_lr: float,
    resolved_min_lr: float,
    updates_per_epoch: int,
    total_update_steps: int,
    warmup_steps: int,
) -> None:
    """Save the exact single-stage LR definition for reproducibility."""
    save_json(
        expert_dir / "lr_schedule_config.json",
        {
            "scheduler": "single_stage_linear_warmup_half_cosine",
            "base_training_stage_matched": "Stage-1",
            "stage2_enabled": False,
            "epochs_budget": int(args.epochs),
            "micro_batch_size": int(args.batch_size),
            "gradient_accumulation_steps": int(args.accumulation_steps),
            "effective_batch_size": int(args.batch_size * args.accumulation_steps),
            "scale_lr_with_accumulation": bool(args.scale_lr_with_accumulation),
            "absolute_lr_requested": args.lr,
            "base_lr": float(args.blr),
            "actual_peak_lr": float(actual_peak_lr),
            "minimum_lr": float(resolved_min_lr),
            "minimum_lr_requested": args.min_lr,
            "minimum_lr_ratio_requested": args.min_lr_ratio,
            "warmup_ratio": float(args.warmup_ratio),
            "warmup_steps_override": args.warmup_steps_override,
            "warmup_initial_lr": float(args.warmup_init_lr),
            "updates_per_epoch": int(updates_per_epoch),
            "total_update_steps": int(total_update_steps),
            "warmup_steps": int(warmup_steps),
            "cosine_steps": int(total_update_steps - warmup_steps),
            "use_module_lr_scales": bool(USE_MODULE_LR_SCALES),
            "backbone_lr_scale": float(args.backbone_lr_scale),
            "text_lr_scale": float(args.text_lr_scale),
            "head_lr_scale": float(args.head_lr_scale),
            "text_encoder_lr_scale": float(args.text_encoder_lr_scale),
        },
    )


def train_one_epoch(
    *,
    model: torch.nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    args: SimpleNamespace,
    epoch: int,
    global_step: int,
    peak_lr: float,
    min_lr: float,
    warmup_steps: int,
    total_steps: int,
    logger: logging.Logger,
) -> tuple[dict[str, float], int, float]:
    set_selective_train_mode(model, args)
    criterion = torch.nn.MSELoss()
    accumulation = args.accumulation_steps
    optimizer.zero_grad(set_to_none=True)
    y_chunks, p_chunks = [], []
    last_lr = float(peak_lr)

    for step, (inputs, targets) in enumerate(loader):
        window_start = (step // accumulation) * accumulation
        window_end = min(window_start + accumulation, len(loader))
        window_size = window_end - window_start
        accumulation_position = step - window_start + 1
        update_boundary = (step + 1 == window_end)

        if accumulation_position == 1:
            last_lr = apply_lr(
                optimizer=optimizer,
                update_step=global_step,
                peak_lr=peak_lr,
                min_lr=min_lr,
                warmup_init_lr=args.warmup_init_lr,
                warmup_steps=warmup_steps,
                total_update_steps=total_steps,
            )

        inputs = to_device(inputs, device)
        targets = normalize_targets(targets, device)
        if not args.use_text:
            inputs = clone_inputs_without_text(inputs)

        with autocast_context(device, args.amp and device.type == "cuda"):
            outputs = normalize_outputs(model(inputs))
            loss = criterion(outputs, targets)
            backward_loss = loss / window_size

        loss_value = float(loss.detach().item())
        if not math.isfinite(loss_value):
            raise RuntimeError(f"non-finite loss at epoch={epoch}, step={step}")
        scaler.scale(backward_loss).backward()

        if update_boundary:
            if args.clip_grad is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=args.clip_grad,
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        y_chunks.append(targets.detach().float().cpu().numpy().reshape(-1))
        p_chunks.append(outputs.detach().float().cpu().numpy().reshape(-1))

        if step % args.log_interval == 0 or step + 1 == len(loader):
            logger.info(
                "epoch %03d | batch %04d/%04d | loss %.7f | base_lr %.3e",
                epoch, step + 1, len(loader), loss_value, last_lr,
            )

    return metric_dict(np.concatenate(y_chunks), np.concatenate(p_chunks)), global_step, last_lr


def override_state_prefixes(args: SimpleNamespace) -> list[str]:
    if FINETUNE_MODE.lower() == "all":
        return [""]
    prefixes = [f"blocks.{idx}." for idx in trainable_block_indices(args.depth, FINETUNE_MODE)]
    prefixes.append("regression_head.")
    if TRAIN_TEXT_PROJ:
        prefixes.append("text_proj.")
    return prefixes


def extract_override_state(model: torch.nn.Module, args: SimpleNamespace) -> dict[str, torch.Tensor]:
    module = model.module if hasattr(model, "module") else model
    prefixes = override_state_prefixes(args)
    state = {}
    for name, value in module.state_dict().items():
        if prefixes == [""] or any(name.startswith(prefix) for prefix in prefixes):
            state[name] = value.detach().cpu()
    return state


def specialist_checkpoint(
    model: torch.nn.Module,
    *,
    expert_name: str,
    expert_id: Optional[int],
    clusters: list[int],
    epoch: int,
    best_val_mae: float,
    trainable_info: dict[str, Any],
    args: SimpleNamespace,
) -> dict[str, Any]:
    module = model.module if hasattr(model, "module") else model
    return {
        "format_version": 2,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "script_version": SCRIPT_VERSION,
        "expert_name": expert_name,
        "expert_id": expert_id,
        "clusters": [int(x) for x in clusters],
        "epoch": int(epoch),
        "best_val_mae": float(best_val_mae),
        "model_state_dict": {k: v.detach().cpu() for k, v in module.state_dict().items()},
        "override_state_dict": extract_override_state(module, args),
        "trainable_info": trainable_info,
    }


def train_specialist(
    *,
    expert_name: str,
    expert_id: Optional[int],
    clusters: list[int],
    base_checkpoint: Any,
    train_dataset: Any,
    val_dataset: Any,
    train_positions: np.ndarray,
    val_positions: np.ndarray,
    args: SimpleNamespace,
    device: torch.device,
    text_hidden_size: int,
    run_dir: Path,
    logger: logging.Logger,
) -> tuple[Path, dict[str, Any]]:
    if len(train_positions) == 0 or len(val_positions) == 0:
        raise RuntimeError(
            f"{expert_name}: empty train/val subset: train={len(train_positions)} val={len(val_positions)}"
        )
    expert_dir = run_dir / expert_name
    checkpoint_dir = expert_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best_model.pt"

    set_seed(args.seed, args.deterministic)
    model = build_model(args, device, text_hidden_size=text_hidden_size)
    load_model_state(model, base_checkpoint, logger)
    trainable_info = configure_finetune(model, args)
    save_json(expert_dir / "trainable_parameters.json", trainable_info)

    train_subset = make_subset(train_dataset, train_positions)
    val_subset = make_subset(val_dataset, val_positions)
    train_loader = make_loader(
        train_subset,
        args,
        shuffle=True,
        collect_source=train_dataset,
        seed=args.seed,
    )
    val_loader = make_loader(
        val_subset,
        args,
        shuffle=False,
        collect_source=val_dataset,
        seed=args.seed,
    )

    actual_peak_lr = resolve_actual_peak_lr(args)
    resolved_min_lr = resolve_minimum_lr(args, actual_peak_lr)
    if resolved_min_lr > actual_peak_lr:
        raise ValueError(
            f"{expert_name}: minimum LR {resolved_min_lr} exceeds "
            f"peak LR {actual_peak_lr}"
        )
    if args.warmup_init_lr > actual_peak_lr and args.warmup_ratio > 0:
        raise ValueError(
            f"{expert_name}: warmup initial LR {args.warmup_init_lr} exceeds "
            f"peak LR {actual_peak_lr}"
        )

    updates_per_epoch, total_steps, warmup_steps = resolve_schedule_steps(
        num_batches_per_epoch=len(train_loader),
        total_epochs_value=args.epochs,
        accumulation_steps=args.accumulation_steps,
        warmup_ratio_value=args.warmup_ratio,
        warmup_steps_override_value=args.warmup_steps_override,
    )

    optimizer = build_optimizer(model, args, actual_peak_lr)
    scaler = build_grad_scaler(args.amp and device.type == "cuda")

    # Match the Base trainer: initialize optimizer groups at Stage-1 step 0.
    apply_lr(
        optimizer=optimizer,
        update_step=0,
        peak_lr=actual_peak_lr,
        min_lr=resolved_min_lr,
        warmup_init_lr=args.warmup_init_lr,
        warmup_steps=warmup_steps,
        total_update_steps=total_steps,
    )

    save_lr_schedule_config(
        expert_dir,
        args=args,
        actual_peak_lr=actual_peak_lr,
        resolved_min_lr=resolved_min_lr,
        updates_per_epoch=updates_per_epoch,
        total_update_steps=total_steps,
        warmup_steps=warmup_steps,
    )

    tb = None
    if args.tensorboard and SummaryWriter is not None:
        tb = SummaryWriter(log_dir=str(expert_dir / "tensorboard"))

    history = []
    best_mae = math.inf
    best_epoch = 0
    no_improve = 0
    global_step = 0
    started = time.time()

    logger.info(
        "%s | clusters=%s | train=%d val=%d | trainable=%d/%d (%.2f%%)",
        expert_name,
        [f"R{x}" for x in clusters],
        len(train_subset), len(val_subset),
        trainable_info["trainable_parameter_count"],
        trainable_info["total_parameter_count"],
        100.0 * trainable_info["trainable_fraction"],
    )
    logger.info(
        "%s LR | SINGLE STAGE warmup+cosine | peak=%.8g min=%.8g "
        "| warmup_init=%.8g | warmup=%d/%d update steps (%.2f%%) "
        "| updates/epoch=%d | module_scales=%s",
        expert_name,
        actual_peak_lr,
        resolved_min_lr,
        args.warmup_init_lr,
        warmup_steps,
        total_steps,
        100.0 * warmup_steps / max(total_steps, 1),
        updates_per_epoch,
        USE_MODULE_LR_SCALES,
    )

    try:
        for epoch in range(1, args.epochs + 1):
            epoch_started = time.time()
            logger.info("========== %s epoch %d/%d ==========", expert_name, epoch, args.epochs)
            train_metrics, global_step, last_lr = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                args=args,
                epoch=epoch,
                global_step=global_step,
                peak_lr=actual_peak_lr,
                min_lr=resolved_min_lr,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                logger=logger,
            )
            yv, pv = predict_loader(
                model,
                val_loader,
                device,
                args.use_text,
                args.amp and device.type == "cuda",
            )
            val_metrics = metric_dict(yv, pv)
            row = {
                "epoch": epoch,
                "train_mse": train_metrics["mse"],
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "val_mse": val_metrics["mse"],
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_bias": val_metrics["bias"],
                "lr_base": last_lr,
                "optimizer_steps": global_step,
                "epoch_seconds": time.time() - epoch_started,
            }
            history.append(row)
            pd.DataFrame(history).to_csv(
                expert_dir / "epoch_metrics.csv",
                index=False,
                encoding="utf-8-sig",
            )

            logger.info(
                "%s | train MAE %.6f RMSE %.6f | val MAE %.6f RMSE %.6f",
                expert_name,
                train_metrics["mae"], train_metrics["rmse"],
                val_metrics["mae"], val_metrics["rmse"],
            )
            if tb is not None:
                for key in ("mse", "mae", "rmse"):
                    tb.add_scalar(f"train/{key}", train_metrics[key], epoch)
                    tb.add_scalar(f"val/{key}", val_metrics[key], epoch)
                tb.add_scalar("train/lr_base", last_lr, epoch)
                tb.flush()

            if val_metrics["mae"] < best_mae:
                best_mae = val_metrics["mae"]
                best_epoch = epoch
                no_improve = 0
                atomic_torch_save(
                    specialist_checkpoint(
                        model,
                        expert_name=expert_name,
                        expert_id=expert_id,
                        clusters=clusters,
                        epoch=epoch,
                        best_val_mae=best_mae,
                        trainable_info=trainable_info,
                        args=args,
                    ),
                    best_path,
                )
                logger.info(
                    "NEW BEST %s | epoch=%d val_mae=%.8f",
                    expert_name, epoch, best_mae,
                )
            else:
                no_improve += 1

            if args.patience > 0 and no_improve >= args.patience:
                logger.info(
                    "early stopping %s after %d non-improving epochs",
                    expert_name, args.patience,
                )
                break
    finally:
        if tb is not None:
            tb.flush()
            tb.close()

    if not best_path.is_file():
        raise RuntimeError(f"No best checkpoint generated for {expert_name}")
    learning_curve_figures = plot_expert_learning_curves(expert_dir, history, logger)
    summary = {
        "expert_name": expert_name,
        "expert_id": expert_id,
        "clusters": clusters,
        "best_epoch": best_epoch,
        "best_val_mae": best_mae,
        "train_samples": int(len(train_positions)),
        "val_samples": int(len(val_positions)),
        "elapsed_seconds": time.time() - started,
        "best_checkpoint": str(best_path),
        "trainable_info": trainable_info,
        "learning_curve_figures": learning_curve_figures,
        "lr_schedule": {
            "scheduler": "single_stage_linear_warmup_half_cosine",
            "stage2_enabled": False,
            "actual_peak_lr": float(actual_peak_lr),
            "minimum_lr": float(resolved_min_lr),
            "warmup_initial_lr": float(args.warmup_init_lr),
            "warmup_ratio": float(args.warmup_ratio),
            "warmup_steps": int(warmup_steps),
            "total_update_steps": int(total_steps),
            "updates_per_epoch": int(updates_per_epoch),
        },
    }
    save_json(expert_dir / "best_result.json", summary)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_path, summary


def load_full_specialist_model(
    checkpoint_path: Path,
    args: SimpleNamespace,
    device: torch.device,
    text_hidden_size: int,
) -> torch.nn.Module:
    checkpoint = load_torch_file(checkpoint_path, map_location=device)
    model = build_model(args, device, text_hidden_size=text_hidden_size)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model



# =============================================================================
# 6B. Expert learning-curve visualization
# =============================================================================

def plot_expert_learning_curves(
    expert_dir: Path,
    history: list[dict[str, Any]],
    logger: logging.Logger,
) -> list[str]:
    if not (SAVE_PLOTS and PLOT_TRAINING_CURVES) or not history:
        return []
    plt = _safe_import_matplotlib()
    if plt is None:
        return []
    df = pd.DataFrame(history)
    fig_dir = expert_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    def pair_plot(train_key: str, val_key: str, ylabel: str, filename: str):
        fig, ax = plt.subplots(figsize=(7.6, 5.2))
        ax.plot(df["epoch"], df[train_key], marker="o", label=train_key)
        ax.plot(df["epoch"], df[val_key], marker="o", label=val_key)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{expert_dir.name}: {ylabel} learning curve")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = fig_dir / filename
        fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(path))

    pair_plot("train_mae", "val_mae", "MAE (eV/atom)", "mae_curve.png")
    pair_plot("train_rmse", "val_rmse", "RMSE (eV/atom)", "rmse_curve.png")
    pair_plot("train_mse", "val_mse", "MSE", "mse_curve.png")

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.plot(df["epoch"], df["lr_base"], marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Base learning rate")
    ax.set_title(f"{expert_dir.name}: single-stage warmup + cosine LR")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = fig_dir / "lr_curve.png"
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(path))

    save_json(expert_dir / "figures_manifest.json", {"figures": saved})
    for path in saved:
        logger.info("saved training curve: %s", path)
    return saved

# =============================================================================
# 7. Deployment bundle and optional integrated runtime
# =============================================================================

def build_deployment_bundle(
    *,
    base_checkpoint: Any,
    router: EmbeddedKMeansRouter,
    cluster_to_expert: np.ndarray,
    expert_to_clusters: dict[int, list[int]],
    expert_best_paths: dict[int, Path],
    args: SimpleNamespace,
    text_hidden_size: int,
    run_dir: Path,
) -> Path:
    base_state, excluded = extract_model_state(base_checkpoint)
    expert_overrides = {}
    expert_best_meta = {}
    for expert_id, path in expert_best_paths.items():
        checkpoint = load_torch_file(path, map_location="cpu")
        expert_overrides[int(expert_id)] = checkpoint["override_state_dict"]
        expert_best_meta[int(expert_id)] = {
            "name": checkpoint["expert_name"],
            "clusters": checkpoint["clusters"],
            "best_val_mae": checkpoint["best_val_mae"],
            "epoch": checkpoint["epoch"],
        }

    bundle = {
        "format_version": 2,
        "bundle_type": "Grouped3ExpertFusiformerEmbeddedRouter",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script_version": SCRIPT_VERSION,
        "base_model_state_dict": {k: v.detach().cpu() for k, v in base_state.items()},
        "base_excluded_model_prefixes": excluded,
        "router_state_dict": {k: v.detach().cpu() for k, v in router.state_dict().items()},
        "router_l2_normalize": bool(router.l2_normalize),
        "cluster_to_expert": cluster_to_expert.astype(int).tolist(),
        "expert_id_to_name": EXPERT_ID_TO_NAME,
        "expert_to_clusters": expert_to_clusters,
        "expert_override_state_dicts": expert_overrides,
        "expert_best_meta": expert_best_meta,
        "model_config": {
            "target": args.target,
            "depth": args.depth,
            "edge_input_dim": args.edge_input_dim,
            "triplet_input_dim": args.triplet_input_dim,
            "embed_dim": args.embed_dim,
            "num_heads": args.num_heads,
            "text_hidden_size": text_hidden_size,
            "text_proj_dim": args.text_proj_dim,
            "text_proj_hidden_dims": list(args.text_proj_hidden_dims),
            "text_proj_dropout": args.text_proj_dropout,
            "text_encoder_name": args.text_encoder_name,
            "text_pooling": args.text_pooling,
            "text_preprocessing": args.text_preprocessing,
            "use_precomputed_text_embeddings": args.use_precomputed_text_embeddings,
            "use_text_for_prediction": args.use_text,
            "finetune_mode": FINETUNE_MODE,
            "train_text_proj": TRAIN_TEXT_PROJ,
        },
    }
    deploy_dir = run_dir / "deployment"
    deploy_dir.mkdir(parents=True, exist_ok=True)
    path = deploy_dir / "grouped3_expert_embedded_router_bundle.pt"
    atomic_torch_save(bundle, path)
    return path


def apply_override_state(model: torch.nn.Module, override: dict[str, torch.Tensor]) -> None:
    state = model.state_dict()
    missing_override_keys = [k for k in override if k not in state]
    if missing_override_keys:
        raise KeyError(f"Override has unknown keys: {missing_override_keys[:20]}")
    state.update(override)
    model.load_state_dict(state, strict=True)


def build_runtime_from_bundle(
    bundle_path: str | Path,
    args: SimpleNamespace,
    device: torch.device,
) -> tuple[torch.nn.Module, EmbeddedKMeansRouter, dict[int, torch.nn.Module]]:
    """Load Base-router model + embedded router + all Experts.

    This does not use assignments.csv or any sample ID lookup. It is provided as
    a deployment helper. Keeping all full experts on GPU may require more VRAM;
    the training/evaluation pipeline itself loads specialists sequentially.
    """
    bundle = load_torch_file(Path(bundle_path), map_location="cpu")
    cfg = bundle["model_config"]
    hidden = int(cfg["text_hidden_size"])

    base_model = build_model(args, device, text_hidden_size=hidden)
    base_model.load_state_dict(bundle["base_model_state_dict"], strict=False)
    base_model.eval()

    router_state = bundle["router_state_dict"]
    centers = router_state["centers"]
    mapping = router_state["cluster_to_expert"]
    router = EmbeddedKMeansRouter(
        centers,
        mapping,
        l2_normalize=bool(bundle["router_l2_normalize"]),
    ).to(device)
    router.load_state_dict(router_state, strict=True)
    router.eval()

    experts = {}
    for expert_id_raw, override in bundle["expert_override_state_dicts"].items():
        expert_id = int(expert_id_raw)
        model = build_model(args, device, text_hidden_size=hidden)
        model.load_state_dict(bundle["base_model_state_dict"], strict=False)
        apply_override_state(model, override)
        model.eval()
        experts[expert_id] = model
    return base_model, router, experts


def _clone_graph_value(value: Any) -> Any:
    if dgl is not None and hasattr(value, "ndata") and hasattr(value, "edata") and hasattr(value, "num_nodes"):
        try:
            return value.clone()
        except Exception:
            return value
    return value


def clone_router_inputs(inputs: Any) -> Any:
    if not isinstance(inputs, dict):
        return inputs
    copied = {k: _clone_graph_value(v) for k, v in inputs.items()}
    copied["text"] = None
    copied["text_emb"] = None
    copied["text_mode"] = "bypass"
    return copied


@torch.no_grad()
def graph_out_from_single_batch(
    base_model: torch.nn.Module,
    inputs: dict[str, Any],
) -> torch.Tensor:
    module = base_model.module if hasattr(base_model, "module") else base_model
    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        captured.append(_first_tensor(output))

    handle = module.readout.register_forward_hook(hook)
    try:
        _ = base_model(clone_router_inputs(inputs))
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"single-batch readout hook captured {len(captured)} tensors")
    return captured[0]


def _select_dgl_batch(graph: Any, indices: list[int]) -> Any:
    if dgl is None:
        raise RuntimeError("DGL is required to split a mixed routed batch")
    graphs = dgl.unbatch(graph)
    return dgl.batch([graphs[i] for i in indices])


def select_batch_inputs(inputs: Any, indices: list[int], batch_size: int) -> Any:
    """Best-effort selector for GraphCollectingCached batched inputs."""
    if isinstance(inputs, dict):
        return {k: select_batch_inputs(v, indices, batch_size) for k, v in inputs.items()}
    if dgl is not None and hasattr(inputs, "batch_size") and hasattr(inputs, "ndata"):
        try:
            if int(inputs.batch_size) == batch_size:
                return _select_dgl_batch(inputs, indices)
        except Exception:
            pass
        return inputs
    if torch.is_tensor(inputs):
        if inputs.ndim >= 1 and inputs.shape[0] == batch_size:
            index_tensor = torch.as_tensor(indices, dtype=torch.long, device=inputs.device)
            return inputs.index_select(0, index_tensor)
        return inputs
    if isinstance(inputs, list) and len(inputs) == batch_size:
        return [inputs[i] for i in indices]
    if isinstance(inputs, tuple) and len(inputs) == batch_size:
        return tuple(inputs[i] for i in indices)
    return inputs


class GroupedExpertFusiformerRuntime(torch.nn.Module):
    """Optional end-to-end runtime for an already-collated batch.

    It performs:
        inputs -> frozen Base graph_out -> embedded router -> selected Expert(s)

    The default offline experiment uses subset DataLoaders for lower memory use;
    this class demonstrates that routing no longer requires lookup tables.
    """

    def __init__(
        self,
        base_router_model: torch.nn.Module,
        router: EmbeddedKMeansRouter,
        experts: dict[int, torch.nn.Module],
    ) -> None:
        super().__init__()
        self.base_router_model = base_router_model
        self.router = router
        self.experts = torch.nn.ModuleDict({str(int(k)): v for k, v in experts.items()})
        for p in self.base_router_model.parameters():
            p.requires_grad = False
        self.base_router_model.eval()
        self.router.eval()
        self.experts.eval()

    @torch.no_grad()
    def forward(self, inputs: dict[str, Any]):
        graph_out = graph_out_from_single_batch(self.base_router_model, inputs)
        routing = self.router(graph_out)
        expert_ids = routing["expert_id"]
        batch_size = int(expert_ids.numel())
        final_output: Optional[torch.Tensor] = None

        for expert_id in torch.unique(expert_ids).tolist():
            positions = torch.where(expert_ids == int(expert_id))[0].tolist()
            sub_inputs = select_batch_inputs(inputs, positions, batch_size)
            sub_output = normalize_outputs(self.experts[str(int(expert_id))](sub_inputs))
            if final_output is None:
                shape = (batch_size,) + tuple(sub_output.shape[1:])
                final_output = torch.empty(shape, dtype=sub_output.dtype, device=sub_output.device)
            final_output[torch.as_tensor(positions, device=sub_output.device)] = sub_output

        if final_output is None:
            raise RuntimeError("empty routed batch")
        return final_output, routing


# =============================================================================
# 8. Final evaluation helpers
# =============================================================================

def metadata_for_positions(dataset: Any, positions: np.ndarray, raw_data: Optional[list[dict[str, Any]]]):
    original = try_original_indices(dataset)
    rows = []
    for pos in positions:
        pos_i = int(pos)
        original_i = int(original[pos_i]) if original is not None else None
        item = None
        if raw_data is not None and original_i is not None and 0 <= original_i < len(raw_data):
            item = raw_data[original_i]
        rows.append({
            "dataset_position": pos_i,
            "original_index": original_i,
            "mp_id": "" if item is None else item.get("mp_id", ""),
            "formula": "" if item is None else item.get("formula", ""),
        })
    return rows


def prediction_frame(
    *,
    dataset: Any,
    positions: np.ndarray,
    cluster_ids: np.ndarray,
    expert_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    raw_data: Optional[list[dict[str, Any]]],
) -> pd.DataFrame:
    meta = metadata_for_positions(dataset, positions, raw_data)
    rows = []
    for m, cluster, expert, yt, yp in zip(meta, cluster_ids, expert_ids, y_true, y_pred):
        rows.append({
            **m,
            "cluster_id": int(cluster),
            "cluster": f"R{int(cluster)}",
            "expert_id": int(expert),
            "expert_name": EXPERT_ID_TO_NAME[int(expert)],
            "model": model_name,
            "y_true": float(yt),
            "y_pred": float(yp),
            "error": float(yp - yt),
            "abs_error": float(abs(yp - yt)),
        })
    return pd.DataFrame(rows)


def catastrophic_summary(df: pd.DataFrame, model_name: str) -> list[dict[str, Any]]:
    ae = df["abs_error"].to_numpy(float)
    rows = []
    for threshold in CATASTROPHIC_ERROR_THRESHOLDS:
        count = int(np.sum(ae >= threshold))
        rows.append({
            "model": model_name,
            "threshold_abs_error": float(threshold),
            "count": count,
            "fraction": float(count / max(len(ae), 1)),
        })
    return rows


def make_plots(
    *,
    overall_df: pd.DataFrame,
    per_cluster_df: pd.DataFrame,
    improvement_df: pd.DataFrame,
    catastrophe_df: pd.DataFrame,
    expert_group_df: pd.DataFrame,
    base_df: pd.DataFrame,
    routed_df: pd.DataFrame,
    global_df: Optional[pd.DataFrame],
    run_dir: Path,
) -> list[str]:
    if not SAVE_PLOTS:
        return []
    plt = _safe_import_matplotlib()
    if plt is None:
        return []
    figure_dir = run_dir / "final_evaluation" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    def savefig(fig, filename: str):
        path = figure_dir / filename
        fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(path))

    # Overall MAE / RMSE.
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.bar(overall_df["model"], overall_df["mae"])
    ax.set_ylabel("MAE (eV/atom)")
    ax.set_title("Overall test MAE")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    savefig(fig, "overall_mae_comparison.png")

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.bar(overall_df["model"], overall_df["rmse"])
    ax.set_ylabel("RMSE (eV/atom)")
    ax.set_title("Overall test RMSE")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    savefig(fig, "overall_rmse_comparison.png")

    # Per-cluster metrics.
    def clustered_bar(metric: str, ylabel: str, title: str, filename: str):
        pivot = per_cluster_df.pivot(index="cluster", columns="model", values=metric)
        fig, ax = plt.subplots(figsize=(9.0, 5.5))
        x = np.arange(len(pivot.index))
        names = list(pivot.columns)
        width = 0.8 / max(len(names), 1)
        for i, name in enumerate(names):
            ax.bar(
                x + (i - (len(names) - 1) / 2) * width,
                pivot[name].to_numpy(),
                width,
                label=name,
            )
        ax.set_xticks(x, pivot.index)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        savefig(fig, filename)

    clustered_bar("mae", "MAE (eV/atom)", "Test MAE by train-only router cluster", "per_cluster_mae_comparison.png")
    clustered_bar("rmse", "RMSE (eV/atom)", "Test RMSE by train-only router cluster", "per_cluster_rmse_comparison.png")
    clustered_bar("p95_abs_error", "P95 absolute error (eV/atom)", "Test P95 absolute error by router cluster", "per_cluster_p95_comparison.png")
    clustered_bar("p99_abs_error", "P99 absolute error (eV/atom)", "Test P99 absolute error by router cluster", "per_cluster_p99_comparison.png")

    # Base -> Routed MAE improvement; positive is better.
    order = improvement_df.sort_values("cluster_id")
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.bar(order["cluster"], order["delta_mae_base_minus_routed"])
    ax.axhline(0.0, linewidth=1.0)
    ax.set_ylabel("Base MAE - Routed MAE (eV/atom)")
    ax.set_title("Per-cluster MAE improvement from Routed Experts")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    savefig(fig, "per_cluster_mae_improvement.png")

    # Catastrophic error rate.
    thresholds = sorted(catastrophe_df["threshold_abs_error"].unique())
    models = list(catastrophe_df["model"].drop_duplicates())
    x = np.arange(len(thresholds))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    for i, model_name in enumerate(models):
        sub = catastrophe_df[catastrophe_df["model"] == model_name].set_index("threshold_abs_error")
        values = [100.0 * float(sub.loc[t, "fraction"]) for t in thresholds]
        ax.bar(x + (i - (len(models) - 1) / 2) * width, values, width, label=model_name)
    ax.set_xticks(x, [f"≥ {t:g}" for t in thresholds])
    ax.set_xlabel("Absolute-error threshold (eV/atom)")
    ax.set_ylabel("Test samples (%)")
    ax.set_title("Catastrophic-error rate comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    savefig(fig, "catastrophic_error_rate_comparison.png")

    # Per Expert-group MAE: directly shows the shared-vs-hard specialist effect.
    eg_pivot = expert_group_df.pivot(index="expert_name", columns="model", values="mae")
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    x = np.arange(len(eg_pivot.index))
    eg_models = list(eg_pivot.columns)
    width = 0.8 / max(len(eg_models), 1)
    for i, model_name in enumerate(eg_models):
        ax.bar(
            x + (i - (len(eg_models) - 1) / 2) * width,
            eg_pivot[model_name].to_numpy(),
            width,
            label=model_name,
        )
    ax.set_xticks(x, eg_pivot.index, rotation=12)
    ax.set_ylabel("MAE (eV/atom)")
    ax.set_title("Test MAE by Expert group")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    savefig(fig, "per_expert_group_mae_comparison.png")

    # Overall absolute-error distribution, robust view without plotting outliers.
    err_groups = [base_df["abs_error"].to_numpy()]
    err_labels = ["Base"]
    if global_df is not None:
        err_groups.append(global_df["abs_error"].to_numpy())
        err_labels.append("Global-FT")
    err_groups.append(routed_df["abs_error"].to_numpy())
    err_labels.append("Routed-3Expert")
    fig, ax = plt.subplots(figsize=(7.8, 5.5))
    ax.boxplot(err_groups, labels=err_labels, showfliers=False)
    ax.set_ylabel("Absolute error (eV/atom)")
    ax.set_title("Overall test absolute-error distribution")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    savefig(fig, "overall_abs_error_boxplot.png")

    # True-vs-pred scatter for each final model.
    frame_list: list[tuple[str, pd.DataFrame]] = [("Base", base_df)]
    if global_df is not None:
        frame_list.append(("Global-FT", global_df))
    frame_list.append(("Routed-3Expert", routed_df))
    for model_name, frame in frame_list:
        pos = deterministic_sample_positions(
            len(frame), FINAL_SCATTER_MAX_POINTS,
            ROUTER_DIAGNOSTICS_RANDOM_STATE + 50 + len(saved),
        )
        y = frame["y_true"].to_numpy(float)[pos]
        p = frame["y_pred"].to_numpy(float)[pos]
        lo = float(min(y.min(), p.min()))
        hi = float(max(y.max(), p.max()))
        fig, ax = plt.subplots(figsize=(6.5, 6.2))
        ax.scatter(y, p, s=9, alpha=0.45)
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0)
        ax.set_xlabel("True e_form (eV/atom)")
        ax.set_ylabel("Predicted e_form (eV/atom)")
        ax.set_title(f"{model_name}: true vs predicted")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        safe_name = model_name.lower().replace("-", "_")
        savefig(fig, f"true_vs_pred_{safe_name}.png")

    # Paired absolute-error change: below y=x means Routed Expert improved.
    merged = base_df[["dataset_position", "abs_error"]].rename(columns={"abs_error": "base_abs"}).merge(
        routed_df[["dataset_position", "abs_error"]].rename(columns={"abs_error": "routed_abs"}),
        on="dataset_position",
        how="inner",
        validate="one_to_one",
    )
    pos = deterministic_sample_positions(
        len(merged), FINAL_SCATTER_MAX_POINTS, ROUTER_DIAGNOSTICS_RANDOM_STATE + 99
    )
    xbase = merged["base_abs"].to_numpy(float)[pos]
    yrouted = merged["routed_abs"].to_numpy(float)[pos]
    lim = float(max(np.max(xbase), np.max(yrouted), 1e-6))
    fig, ax = plt.subplots(figsize=(6.5, 6.2))
    ax.scatter(xbase, yrouted, s=9, alpha=0.4)
    ax.plot([0, lim], [0, lim], linestyle="--", linewidth=1.0)
    ax.set_xlabel("Base absolute error (eV/atom)")
    ax.set_ylabel("Routed Expert absolute error (eV/atom)")
    ax.set_title("Paired sample error: Base vs Routed Expert")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    savefig(fig, "paired_abs_error_base_vs_routed.png")

    # Router confidence / distance on final test samples, descriptive only.
    if "router_confidence" in base_df.columns:
        grouped = [base_df.loc[base_df.cluster_id == g, "router_confidence"].to_numpy() for g in range(ROUTER_K)]
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ax.boxplot(grouped, labels=[f"R{g}" for g in range(ROUTER_K)], showfliers=False)
        ax.set_xlabel("Router cluster")
        ax.set_ylabel("Routing confidence")
        ax.set_title("TEST routing confidence by cluster (evaluation only)")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        savefig(fig, "test_router_confidence_by_cluster.png")

    if "router_nearest_dist2" in base_df.columns:
        grouped = [base_df.loc[base_df.cluster_id == g, "router_nearest_dist2"].to_numpy() for g in range(ROUTER_K)]
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ax.boxplot(grouped, labels=[f"R{g}" for g in range(ROUTER_K)], showfliers=False)
        ax.set_xlabel("Router cluster")
        ax.set_ylabel("Squared distance to assigned centroid")
        ax.set_title("TEST routing distance by cluster (evaluation only)")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        savefig(fig, "test_router_distance_by_cluster.png")

    save_json(figure_dir / "figures_manifest.json", {"figures": saved})
    return saved


# =============================================================================
# 9. Main experiment
# =============================================================================

def main() -> None:
    RUNS_PATH.mkdir(parents=True, exist_ok=True)
    args = build_args()
    validate_args(args)
    output_root = resolve_path(OUTPUT_ROOT)
    assert output_root is not None
    run_dir = make_run_dir(output_root)
    logger = setup_logger(run_dir)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device(args.device)
    set_seed(args.seed, args.deterministic)

    base_weights_path = resolve_base_weights()
    base_checkpoint = load_torch_file(base_weights_path, map_location=device)
    validate_checkpoint_architecture(base_checkpoint, args, logger)
    text_hidden_size = get_text_hidden(args)

    logger.info("script version: %s", SCRIPT_VERSION)
    logger.info("run dir: %s", run_dir)
    logger.info("Base weights: %s", base_weights_path)
    logger.info("device: %s", device)
    logger.info("router: TRAIN-only KMeans K=%d, no assignments.csv", ROUTER_K)
    logger.info("difficulty metric: %s", DIFFICULTY_METRIC)
    logger.info("finetune mode: %s", FINETUNE_MODE)

    # Raw JSON is optional metadata only; routing/training never needs it.
    raw_data: Optional[list[dict[str, Any]]] = None
    try:
        with Path(args.data_path).open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, list):
            raw_data = loaded
    except Exception as exc:
        logger.warning("Could not load raw JSON metadata: %s", exc)

    train_dataset, val_dataset, test_dataset = build_datasets(args, logger)

    # -------------------------------------------------------------------------
    # Phase 1: TRAIN + VAL only. TEST is not iterated here.
    # -------------------------------------------------------------------------
    base_model = build_model(args, device, text_hidden_size=text_hidden_size)
    load_model_state(base_model, base_checkpoint, logger)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    train_graph_out = cached_or_extract_graph_out(
        split_name="train",
        dataset=train_dataset,
        base_model=base_model,
        args=args,
        device=device,
        base_weights_path=base_weights_path,
        logger=logger,
    )
    val_graph_out = cached_or_extract_graph_out(
        split_name="val",
        dataset=val_dataset,
        base_model=base_model,
        args=args,
        device=device,
        base_weights_path=base_weights_path,
        logger=logger,
    )

    kmeans, train_cluster = fit_train_only_kmeans(train_graph_out, run_dir, logger)
    val_cluster = predict_sklearn_cluster(kmeans, val_graph_out)

    # Base validation predictions, used only for difficulty-aware model selection.
    val_loader = make_loader(val_dataset, args, shuffle=False, seed=args.seed)
    y_val, p_val_base = predict_loader(
        base_model,
        val_loader,
        device,
        args.use_text,
        args.amp and device.type == "cuda",
    )
    if len(y_val) != len(val_cluster):
        raise RuntimeError("validation prediction / router-label length mismatch")

    difficulty_positions, early_stop_positions, role_df = split_validation_roles(
        val_cluster, logger
    )
    role_df.to_csv(
        run_dir / "router" / "validation_roles.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cluster_to_expert, expert_to_clusters, difficulty_df = select_expert_mapping_from_validation(
        val_cluster=val_cluster,
        y_val=y_val,
        p_val_base=p_val_base,
        difficulty_positions=difficulty_positions,
        run_dir=run_dir,
        logger=logger,
    )

    embedded_router = EmbeddedKMeansRouter(
        kmeans.cluster_centers_,
        cluster_to_expert,
        l2_normalize=ROUTER_L2_NORMALIZE,
    ).to(device)
    embedded_router.eval()
    verify_embedded_router(
        embedded_router, train_graph_out, train_cluster, device, logger, "train"
    )
    verify_embedded_router(
        embedded_router, val_graph_out, val_cluster, device, logger, "val"
    )

    router_state_path = run_dir / "router" / "embedded_router_state.pt"
    atomic_torch_save(
        {
            "format_version": 2,
            "script_version": SCRIPT_VERSION,
            "router_state_dict": {
                k: v.detach().cpu() for k, v in embedded_router.state_dict().items()
            },
            "l2_normalize": ROUTER_L2_NORMALIZE,
            "cluster_to_expert": cluster_to_expert.tolist(),
            "expert_id_to_name": EXPERT_ID_TO_NAME,
            "expert_to_clusters": expert_to_clusters,
            "fit_split": "train_only",
            "difficulty_selection_split": "validation_only",
            "test_used_for_router_or_group_selection": False,
        },
        router_state_path,
    )

    # Router assignments are computed by the router itself, not lookup tables.
    pd.DataFrame({
        "train_position": np.arange(len(train_cluster)),
        "cluster_id": train_cluster,
        "cluster": [f"R{x}" for x in train_cluster],
        "expert_id": cluster_to_expert[train_cluster],
        "expert_name": [EXPERT_ID_TO_NAME[int(x)] for x in cluster_to_expert[train_cluster]],
    }).to_csv(run_dir / "router" / "train_routes.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "val_position": np.arange(len(val_cluster)),
        "cluster_id": val_cluster,
        "cluster": [f"R{x}" for x in val_cluster],
        "expert_id": cluster_to_expert[val_cluster],
        "expert_name": [EXPERT_ID_TO_NAME[int(x)] for x in cluster_to_expert[val_cluster]],
    }).to_csv(run_dir / "router" / "val_routes.csv", index=False, encoding="utf-8-sig")

    save_json(
        run_dir / "router" / "router_summary.json",
        {
            "fit_split": "train_only",
            "K": ROUTER_K,
            "l2_normalize": ROUTER_L2_NORMALIZE,
            "cluster_to_expert": cluster_to_expert.tolist(),
            "expert_to_clusters": expert_to_clusters,
            "difficulty_metric": DIFFICULTY_METRIC,
            "test_used_before_training_complete": False,
            "train_cluster_counts": {
                f"R{g}": int(np.sum(train_cluster == g)) for g in range(ROUTER_K)
            },
            "val_cluster_counts": {
                f"R{g}": int(np.sum(val_cluster == g)) for g in range(ROUTER_K)
            },
        },
    )

    # Full Router/KMeans diagnostics. These use TRAIN/VAL only and do not alter
    # any router/grouping/training decision.
    router_diagnostics = run_router_diagnostics(
        train_graph_out=train_graph_out,
        val_graph_out=val_graph_out,
        train_cluster=train_cluster,
        val_cluster=val_cluster,
        y_val=y_val,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        raw_data=raw_data,
        kmeans=kmeans,
        embedded_router=embedded_router,
        cluster_to_expert=cluster_to_expert,
        difficulty_df=difficulty_df,
        run_dir=run_dir,
        device=device,
        logger=logger,
    )

    # -------------------------------------------------------------------------
    # Phase 2: train Experts using TRAIN and expert-early-stop VAL only.
    # -------------------------------------------------------------------------
    early_stop_mask = np.zeros(len(val_dataset), dtype=bool)
    early_stop_mask[early_stop_positions] = True

    expert_best_paths: dict[int, Path] = {}
    training_summaries = []
    for expert_id in (0, 1, 2):
        clusters = [int(x) for x in expert_to_clusters[expert_id]]
        expert_name = EXPERT_ID_TO_NAME[expert_id]
        train_positions = np.where(np.isin(train_cluster, clusters))[0]
        val_positions = np.where(np.isin(val_cluster, clusters) & early_stop_mask)[0]
        best_path, summary = train_specialist(
            expert_name=expert_name,
            expert_id=expert_id,
            clusters=clusters,
            base_checkpoint=base_checkpoint,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_positions=train_positions,
            val_positions=val_positions,
            args=args,
            device=device,
            text_hidden_size=text_hidden_size,
            run_dir=run_dir,
            logger=logger,
        )
        expert_best_paths[expert_id] = best_path
        training_summaries.append(summary)

    global_best_path: Optional[Path] = None
    if RUN_GLOBAL_FT_CONTROL and not SMOKE_TEST:
        global_best_path, global_summary = train_specialist(
            expert_name="global_ft_control",
            expert_id=None,
            clusters=list(range(ROUTER_K)),
            base_checkpoint=base_checkpoint,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_positions=np.arange(len(train_dataset), dtype=int),
            val_positions=early_stop_positions,
            args=args,
            device=device,
            text_hidden_size=text_hidden_size,
            run_dir=run_dir,
            logger=logger,
        )
        training_summaries.append(global_summary)

    save_json(run_dir / "training_summaries.json", training_summaries)

    # -------------------------------------------------------------------------
    # Phase 3: TEST is touched only now.
    # -------------------------------------------------------------------------
    test_graph_out = cached_or_extract_graph_out(
        split_name="test",
        dataset=test_dataset,
        base_model=base_model,
        args=args,
        device=device,
        base_weights_path=base_weights_path,
        logger=logger,
    )
    test_sklearn_cluster = predict_sklearn_cluster(kmeans, test_graph_out)
    verify_embedded_router(
        embedded_router,
        test_graph_out,
        test_sklearn_cluster,
        device,
        logger,
        "test",
    )
    with torch.no_grad():
        routing_out = embedded_router(torch.from_numpy(test_graph_out).to(device))
    test_cluster = routing_out["cluster_id"].cpu().numpy().astype(int)
    test_expert = routing_out["expert_id"].cpu().numpy().astype(int)
    test_confidence = routing_out["confidence"].cpu().numpy().astype(float)
    test_nearest_dist2 = routing_out["nearest_dist2"].cpu().numpy().astype(float)

    if not np.array_equal(test_cluster, test_sklearn_cluster):
        raise RuntimeError("embedded router test labels differ from sklearn KMeans")

    # Base test once, after model selection is complete.
    test_loader = make_loader(test_dataset, args, shuffle=False, seed=args.seed)
    y_test, p_base = predict_loader(
        base_model,
        test_loader,
        device,
        args.use_text,
        args.amp and device.type == "cuda",
    )
    base_metrics = metric_dict(y_test, p_base)
    logger.info(
        "BASE TEST | MAE %.6f RMSE %.6f P95 %.6f P99 %.6f",
        base_metrics["mae"], base_metrics["rmse"],
        base_metrics["p95_abs_error"], base_metrics["p99_abs_error"],
    )
    if EXPECTED_BASE_TEST_MAE is not None:
        delta = abs(base_metrics["mae"] - EXPECTED_BASE_TEST_MAE)
        if delta > EXPECTED_BASE_TEST_MAE_TOLERANCE:
            logger.warning(
                "Base test MAE sanity check differs: actual=%.6f expected=%.6f delta=%.6f. "
                "Verify the selected Base checkpoint and split before interpretation.",
                base_metrics["mae"], EXPECTED_BASE_TEST_MAE, delta,
            )

    all_test_positions = np.arange(len(test_dataset), dtype=int)
    base_df = prediction_frame(
        dataset=test_dataset,
        positions=all_test_positions,
        cluster_ids=test_cluster,
        expert_ids=test_expert,
        y_true=y_test,
        y_pred=p_base,
        model_name="Base",
        raw_data=raw_data,
    )
    base_df["router_confidence"] = test_confidence
    base_df["router_nearest_dist2"] = test_nearest_dist2

    # Routed experts: route by embedded center distances, then evaluate subset.
    routed_parts = []
    for expert_id in (0, 1, 2):
        positions = np.where(test_expert == expert_id)[0]
        if len(positions) == 0:
            continue
        model = load_full_specialist_model(
            expert_best_paths[expert_id], args, device, text_hidden_size
        )
        loader = make_loader(
            make_subset(test_dataset, positions),
            args,
            shuffle=False,
            collect_source=test_dataset,
            seed=args.seed,
        )
        yt, yp = predict_loader(
            model,
            loader,
            device,
            args.use_text,
            args.amp and device.type == "cuda",
        )
        part = prediction_frame(
            dataset=test_dataset,
            positions=positions,
            cluster_ids=test_cluster[positions],
            expert_ids=test_expert[positions],
            y_true=yt,
            y_pred=yp,
            model_name="Routed-3Expert",
            raw_data=raw_data,
        )
        part["router_confidence"] = test_confidence[positions]
        part["router_nearest_dist2"] = test_nearest_dist2[positions]
        routed_parts.append(part)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    routed_df = pd.concat(routed_parts, ignore_index=True).sort_values("dataset_position")
    routed_df = routed_df.reset_index(drop=True)
    if not np.array_equal(routed_df["dataset_position"].to_numpy(), all_test_positions):
        raise RuntimeError("Routed output does not cover test set exactly once")
    routed_metrics = metric_dict(
        routed_df["y_true"].to_numpy(), routed_df["y_pred"].to_numpy()
    )
    logger.info(
        "ROUTED-3EXPERT TEST | MAE %.6f RMSE %.6f P95 %.6f P99 %.6f",
        routed_metrics["mae"], routed_metrics["rmse"],
        routed_metrics["p95_abs_error"], routed_metrics["p99_abs_error"],
    )

    global_df = None
    global_metrics = None
    if global_best_path is not None:
        global_model = load_full_specialist_model(
            global_best_path, args, device, text_hidden_size
        )
        yg, pg = predict_loader(
            global_model,
            test_loader,
            device,
            args.use_text,
            args.amp and device.type == "cuda",
        )
        global_df = prediction_frame(
            dataset=test_dataset,
            positions=all_test_positions,
            cluster_ids=test_cluster,
            expert_ids=test_expert,
            y_true=yg,
            y_pred=pg,
            model_name="Global-FT",
            raw_data=raw_data,
        )
        global_df["router_confidence"] = test_confidence
        global_df["router_nearest_dist2"] = test_nearest_dist2
        global_metrics = metric_dict(yg, pg)
        logger.info(
            "GLOBAL-FT TEST | MAE %.6f RMSE %.6f",
            global_metrics["mae"], global_metrics["rmse"],
        )
        del global_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_dir = run_dir / "final_evaluation"
    final_dir.mkdir(parents=True, exist_ok=True)

    overall_rows = [{"model": "Base", **base_metrics}]
    if global_metrics is not None:
        overall_rows.append({"model": "Global-FT", **global_metrics})
    overall_rows.append({"model": "Routed-3Expert", **routed_metrics})
    overall_df = pd.DataFrame(overall_rows)
    overall_df.to_csv(final_dir / "overall_comparison.csv", index=False, encoding="utf-8-sig")

    # Per-router-cluster metrics.
    per_cluster_rows = []
    model_frames = [("Base", base_df), ("Routed-3Expert", routed_df)]
    if global_df is not None:
        model_frames.insert(1, ("Global-FT", global_df))
    for model_name, frame in model_frames:
        for cluster_id in range(ROUTER_K):
            sub = frame[frame["cluster_id"] == cluster_id]
            metrics = metric_dict(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy())
            per_cluster_rows.append({
                "model": model_name,
                "cluster_id": cluster_id,
                "cluster": f"R{cluster_id}",
                "expert_id": int(cluster_to_expert[cluster_id]),
                "expert_name": EXPERT_ID_TO_NAME[int(cluster_to_expert[cluster_id])],
                **metrics,
            })
    per_cluster_df = pd.DataFrame(per_cluster_rows)
    per_cluster_df.to_csv(
        final_dir / "per_cluster_comparison.csv", index=False, encoding="utf-8-sig"
    )

    # Paired Base -> Routed improvement per cluster.
    improvement_rows = []
    for cluster_id in range(ROUTER_K):
        b = base_df[base_df["cluster_id"] == cluster_id]
        r = routed_df[routed_df["cluster_id"] == cluster_id]
        bm = metric_dict(b["y_true"].to_numpy(), b["y_pred"].to_numpy())
        rm = metric_dict(r["y_true"].to_numpy(), r["y_pred"].to_numpy())
        improvement_rows.append({
            "cluster_id": cluster_id,
            "cluster": f"R{cluster_id}",
            "expert_id": int(cluster_to_expert[cluster_id]),
            "expert_name": EXPERT_ID_TO_NAME[int(cluster_to_expert[cluster_id])],
            "n": bm["n"],
            "base_mae": bm["mae"],
            "routed_mae": rm["mae"],
            "delta_mae_base_minus_routed": bm["mae"] - rm["mae"],
            "relative_mae_improvement": (
                bm["mae"] - rm["mae"]
            ) / max(bm["mae"], 1e-12),
            "base_rmse": bm["rmse"],
            "routed_rmse": rm["rmse"],
            "base_p95": bm["p95_abs_error"],
            "routed_p95": rm["p95_abs_error"],
            "base_p99": bm["p99_abs_error"],
            "routed_p99": rm["p99_abs_error"],
        })
    improvement_df = pd.DataFrame(improvement_rows)
    improvement_df.to_csv(
        final_dir / "per_cluster_improvement.csv", index=False, encoding="utf-8-sig"
    )

    # Expert-group metrics.
    expert_group_rows = []
    for expert_id, clusters in expert_to_clusters.items():
        for model_name, frame in [("Base", base_df), ("Routed-3Expert", routed_df)]:
            sub = frame[np.isin(frame["cluster_id"], clusters)]
            metrics = metric_dict(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy())
            expert_group_rows.append({
                "expert_id": expert_id,
                "expert_name": EXPERT_ID_TO_NAME[expert_id],
                "clusters": ",".join(f"R{x}" for x in clusters),
                "model": model_name,
                **metrics,
            })
    expert_group_df = pd.DataFrame(expert_group_rows)
    expert_group_df.to_csv(
        final_dir / "per_expert_group_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    catastrophe_rows = catastrophic_summary(base_df, "Base")
    if global_df is not None:
        catastrophe_rows.extend(catastrophic_summary(global_df, "Global-FT"))
    catastrophe_rows.extend(catastrophic_summary(routed_df, "Routed-3Expert"))
    catastrophe_df = pd.DataFrame(catastrophe_rows)
    catastrophe_df.to_csv(
        final_dir / "catastrophic_error_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if SAVE_SAMPLE_PREDICTIONS:
        paired = base_df[[
            "dataset_position", "original_index", "mp_id", "formula",
            "cluster_id", "cluster", "expert_id", "expert_name",
            "router_confidence", "router_nearest_dist2",
            "y_true", "y_pred", "error", "abs_error",
        ]].rename(columns={
            "y_pred": "base_pred",
            "error": "base_error",
            "abs_error": "base_abs_error",
        })
        rcols = routed_df[[
            "dataset_position", "y_pred", "error", "abs_error"
        ]].rename(columns={
            "y_pred": "routed_pred",
            "error": "routed_error",
            "abs_error": "routed_abs_error",
        })
        paired = paired.merge(rcols, on="dataset_position", how="inner", validate="one_to_one")
        paired["abs_error_reduction"] = paired["base_abs_error"] - paired["routed_abs_error"]
        if global_df is not None:
            gcols = global_df[["dataset_position", "y_pred", "abs_error"]].rename(columns={
                "y_pred": "global_ft_pred",
                "abs_error": "global_ft_abs_error",
            })
            paired = paired.merge(gcols, on="dataset_position", how="left", validate="one_to_one")
        paired.to_csv(
            final_dir / "sample_predictions_paired.csv",
            index=False,
            encoding="utf-8-sig",
        )

    pd.DataFrame({
        "test_position": all_test_positions,
        "cluster_id": test_cluster,
        "cluster": [f"R{x}" for x in test_cluster],
        "expert_id": test_expert,
        "expert_name": [EXPERT_ID_TO_NAME[int(x)] for x in test_expert],
        "router_confidence": test_confidence,
        "router_nearest_dist2": test_nearest_dist2,
    }).to_csv(final_dir / "test_routes_from_embedded_router.csv", index=False, encoding="utf-8-sig")

    figures = make_plots(
        overall_df=overall_df,
        per_cluster_df=per_cluster_df,
        improvement_df=improvement_df,
        catastrophe_df=catastrophe_df,
        expert_group_df=expert_group_df,
        base_df=base_df,
        routed_df=routed_df,
        global_df=global_df,
        run_dir=run_dir,
    )
    for path in figures:
        logger.info("saved final evaluation figure: %s", path)

    bundle_path = None
    if SAVE_DEPLOYMENT_BUNDLE:
        bundle_path = build_deployment_bundle(
            base_checkpoint=base_checkpoint,
            router=embedded_router,
            cluster_to_expert=cluster_to_expert,
            expert_to_clusters=expert_to_clusters,
            expert_best_paths=expert_best_paths,
            args=args,
            text_hidden_size=text_hidden_size,
            run_dir=run_dir,
        )
        logger.info("saved deployment bundle: %s", bundle_path)

    summary = {
        "script_version": SCRIPT_VERSION,
        "data_path": args.data_path,
        "base_weights_path": str(base_weights_path),
        "strict_no_test_leakage_design": {
            "kmeans_fit": "train_only",
            "difficulty_group_selection": "validation_only",
            "expert_early_stopping": "validation_only",
            "disjoint_validation_roles": USE_DISJOINT_VALIDATION_ROLES,
            "old_assignments_csv_used": False,
            "test_used_before_training_complete": False,
        },
        "router": {
            "K": ROUTER_K,
            "l2_normalize": ROUTER_L2_NORMALIZE,
            "cluster_to_expert": cluster_to_expert.tolist(),
            "expert_to_clusters": expert_to_clusters,
            "expert_id_to_name": EXPERT_ID_TO_NAME,
        },
        "router_diagnostics": router_diagnostics,
        "finetune": {
            "mode": FINETUNE_MODE,
            "train_text_proj": TRAIN_TEXT_PROJ,
            "epochs": args.epochs,
            "lr_scheduler": "single_stage_linear_warmup_half_cosine",
            "stage2_enabled": False,
            "scale_lr_with_accumulation": args.scale_lr_with_accumulation,
            "absolute_lr_requested": args.lr,
            "base_lr": args.blr,
            "actual_peak_lr": resolve_actual_peak_lr(args),
            "minimum_lr": resolve_minimum_lr(args, resolve_actual_peak_lr(args)),
            "minimum_lr_ratio_requested": args.min_lr_ratio,
            "warmup_ratio": args.warmup_ratio,
            "warmup_steps_override": args.warmup_steps_override,
            "warmup_initial_lr": args.warmup_init_lr,
            "use_module_lr_scales": USE_MODULE_LR_SCALES,
            "backbone_lr_scale": args.backbone_lr_scale,
            "text_lr_scale": args.text_lr_scale,
            "head_lr_scale": args.head_lr_scale,
            "loss": LOSS_NAME,
        },
        "base_test_metrics": base_metrics,
        "global_ft_test_metrics": global_metrics,
        "routed_test_metrics": routed_metrics,
        "deployment_bundle": None if bundle_path is None else str(bundle_path),
        "figures": figures,
    }
    save_json(run_dir / "experiment_config_and_summary.json", summary)

    logger.info("")
    logger.info("================ FINAL SUMMARY ================")
    logger.info("Base MAE:           %.8f", base_metrics["mae"])
    if global_metrics is not None:
        logger.info("Global-FT MAE:      %.8f", global_metrics["mae"])
    logger.info("Routed-3Expert MAE: %.8f", routed_metrics["mae"])
    logger.info(
        "Routed improvement vs Base: %.2f%%",
        100.0 * (base_metrics["mae"] - routed_metrics["mae"]) / max(base_metrics["mae"], 1e-12),
    )
    logger.info("run directory: %s", run_dir)
    logger.info("Inspect first:")
    logger.info("  %s", run_dir / "router" / "validation_cluster_difficulty.csv")
    logger.info("  %s", run_dir / "router" / "difficulty_aware_grouping.json")
    logger.info("  %s", run_dir / "router" / "kmeans_quality_metrics.json")
    logger.info("  %s", run_dir / "router" / "router_diagnostics_summary.json")
    logger.info("  %s", final_dir / "overall_comparison.csv")
    logger.info("  %s", final_dir / "per_cluster_improvement.csv")
    logger.info("  %s", final_dir / "catastrophic_error_comparison.csv")


if __name__ == "__main__":
    main()
