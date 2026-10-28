#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fusiformer-V2.1 独立训练入口：Frozen ModernBERT + Late Chunking。

在任意项目根目录运行：
    python -m scripts.train.train_fusiformer_v2_modernbert_independent --check-config
    python -u -m scripts.train.train_fusiformer_v2_modernbert_independent

只需修改下方 Config 的字段。该文件**不导入** train_fusiformer_v2.py 或
train_fusiformer_v2_modernbert.py；直接组装 V2 模型/数据加载器，并复用已验证
的 v6/v4 优化器、DGL 图处理、训练循环、CSV/TensorBoard、checkpoint 和绘图引擎。

ModernBERT 权重不会在训练期间加载；仅使用已经预计算的 [N,M,768] 语义缓存。
默认仅支持从头训练；A0 checkpoint 与 V2 不兼容，避免误续训。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fusiformer_v2.modernbert_runtime import (
    inspect_cache,
    setup_training as _setup_training,
    validate_user_config,
)

ROOT = Path(__file__).resolve().parents[2]


# =============================================================================
#                        用户统一配置区（修改这里即可）
# =============================================================================
@dataclass(frozen=True)
class Config:
    # 1. 任务 / 数据 / 运行目录
    experiment_name: str = "fusiformer_v21_a2_eform_modernbert_late"
    data_path: str = str(ROOT / "data/dataset_final_robocrys.json")
    text_cache_path: str = str(ROOT / "data/v2/modernbert_late_robocrys_segments.npy")
    text_model_path: str = str(ROOT / "external/ModernBERT-base")  # 仅校验缓存来源；训练不加载权重
    output_root: str = str(ROOT / "runs/fusiformer_v21_a2_eform_modernbert_late")
    target: str = "e_form"
    text_field: str = "description"
    data_split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1)
    # None = 与当前 V2 baseline 一致的划分；允许原实现因取整少 1 条。
    data_split_counts: tuple[int, int, int] | None = None

    # 2. V2 阶段与融合结构
    stage: str = "a2"                      # a1 / a2 / a3
    num_queries: int = 16                  # A3 生效
    cross_dropout: float = 0.1             # A2 / A3
    model_depth: int = 6
    edge_input_dim: int = 80
    triplet_input_dim: int = 40
    embed_dim: int = 128
    num_heads: int = 4
    text_proj_dim: int = 64
    text_proj_hidden_dims: tuple[int, ...] = (256, 128)
    text_proj_dropout: float = 0.1

    # 3. Batch / Epoch / 早停
    batch_size: int = 32
    gradient_accumulation_steps: int = 2     # effective_batch = 64
    lr_schedule_total_epochs: int = 100
    additional_epochs: int = 0
    early_stopping_patience: int = 10        # 0 = 不启用早停
    random_seed: int = 42
    best_model_monitor: str = "val_mae"

    # 4. 优化器和学习率（保持 v6/v4 原有计算范式）
    # absolute_lr != None: 用固定学习率；否则实际 LR = base_lr * scaling_batch/256
    absolute_lr: float | None = None
    base_lr: float = 1e-3
    scale_lr_with_accumulation: bool = True
    minimum_lr: float | None = None
    minimum_lr_ratio: float | None = 1e-4
    warmup_ratio: float = 0.04
    warmup_steps_override: int | None = None
    warmup_initial_lr: float = 1e-7
    weight_decay: float = 0.05
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    clip_grad_norm: float | None = None

    # 5. 模块差异化学习率（默认关闭，与此前 baseline 保持一致）
    use_module_lr_scales: bool = False
    backbone_lr_scale: float = 1.0
    text_module_lr_scale: float = 1.0
    regression_head_lr_scale: float = 1.0
    text_encoder_lr_scale: float = 0.1    # 编码器冻结，不会进入优化器

    # 6. 可选延长训练（默认关闭）
    use_stage2_dynamic_lr: bool = False
    stage2_peak_lr: float = 2.5e-5
    stage2_min_lr: float = 1e-7
    stage2_warmup_epochs: int = 3

    # 7. 运行 / 日志 / 保存策略
    device_name: str = "cuda"
    num_workers: int = 8
    pin_memory: bool = True
    drop_last: bool = False
    use_amp: bool = False              # v4 AMP 范式，首次训练保留 False
    use_deterministic_algorithms: bool = False
    log_interval: int = 100            # 每 100 micro-batch 输出训练进度
    save_interval: int = 25            # 每 25 Epoch 保存额外周期 checkpoint
    enable_tensorboard: bool = True
    evaluate_test_each_epoch: bool = False


CFG = Config()
VERSION = "fusiformer-v2.1-independent-modernbert-late-20261017"


# =============================================================================
#                               参数检查
# =============================================================================
# =============================================================================
#         仅本脚本安装 V2 训练钩子；不依赖另外两个 V2 训练启动文件
# =============================================================================
def setup_training(cfg: Config):
    return _setup_training(cfg, version=VERSION, script_name=Path(__file__).name)


def report_configuration(cfg: Config, runtime, cache_info: dict[str, Any]) -> None:
    print("=" * 76)
    print(f"Fusiformer-V2.1 | {cfg.experiment_name} | Frozen ModernBERT Late Chunking")
    print("=" * 76)
    print(f"data_path          : {runtime.data_path}")
    print(f"semantic_cache     : {runtime.text_embedding_cache_path}")
    print(f"cache_shape        : {cache_info['cache_shape']} / {cache_info['cache_dtype']}")
    print(f"output_root        : {runtime.output_root}")
    print(f"target / stage     : {runtime.target} / {runtime.v2_stage}")
    print(f"text encoder       : {runtime.text_encoder_name} (FROZEN; not loaded)")
    print(f"coGN depth / width : {runtime.depth} / {runtime.embed_dim}")
    print(f"attention heads    : {runtime.num_heads}; queries={runtime.v2_num_queries}")
    print(f"batch / accumulate : {runtime.batch_size} / {runtime.accumulation_steps}")
    effective = runtime.batch_size * runtime.accumulation_steps
    scaling = effective if runtime.scale_lr_with_accumulation else runtime.batch_size
    peak = runtime.lr if runtime.lr is not None else runtime.blr * scaling / 256
    print(f"effective batch    : {effective}")
    print(f"peak LR            : {peak:.8g}")
    print(f"epochs / patience  : {runtime.epochs} / {runtime.early_stopping_patience}")
    print(f"seed / workers     : {runtime.seed} / {runtime.num_workers}")
    print(f"AMP / GPU          : {runtime.amp} / {runtime.device}")
    print("SHA256 note        : JSON text/order hash is checked again at Dataset load")
    print("Outputs            : config.json, train.log, step_metrics.csv, epoch_metrics.csv,")
    print("                     checkpoints/, best_result.json, final_test_result.json,")
    print("                     tensorboard/, training figures (if plotting succeeds)")
    print("=" * 76, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-config", action="store_true",
                        help="检查本文件配置、缓存格式、训练引擎映射；不构建图或启动训练")
    ns = parser.parse_args()
    validate_user_config(CFG)
    cache_info = inspect_cache(CFG)
    engine = setup_training(CFG)
    runtime = engine.build_runtime_config()
    engine.validate_config(runtime)
    report_configuration(CFG, runtime, cache_info)
    if ns.check_config:
        print("[PASS] Configuration and cache metadata verified; no training performed.")
        return
    print("[INFO] Starting production engine; all paths/metrics/checkpoints follow v4/v6.", flush=True)
    engine.main()


if __name__ == "__main__":
    main()
