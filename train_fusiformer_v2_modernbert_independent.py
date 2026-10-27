#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fusiformer-V2.1 独立训练入口：Frozen ModernBERT + Late Chunking。

使用方式（放在 /root/llm_fusiformer/ 项目根目录）：
    python train_fusiformer_v2_modernbert_independent.py --check-config
    python -u train_fusiformer_v2_modernbert_independent.py

只需修改下方 Config 的字段。该文件**不导入** train_fusiformer_v2.py 或
train_fusiformer_v2_modernbert.py；直接组装 V2 模型/数据加载器，并复用已验证
的 v6/v4 优化器、DGL 图处理、训练循环、CSV/TensorBoard、checkpoint 和绘图引擎。

ModernBERT 权重不会在训练期间加载；仅使用已经预计算的 [N,M,768] 语义缓存。
默认仅支持从头训练；A0 checkpoint 与 V2 不兼容，避免误续训。
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent


# =============================================================================
#                        用户统一配置区（修改这里即可）
# =============================================================================
@dataclass(frozen=True)
class Config:
    # 1. 任务 / 数据 / 运行目录
    experiment_name: str = "fusiformer_v21_a2_eform_modernbert_late"
    data_path: str = str(ROOT / "data/dataset_final_robocrys.json")
    text_cache_path: str = str(ROOT / "data/v2/modernbert_late_robocrys_segments.npy")
    text_model_path: str = "/root/models/ModernBERT-base"   # 仅校验缓存来源；训练不加载权重
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
def validate_user_config(cfg: Config) -> None:
    if cfg.stage not in {"a1", "a2", "a3"}:
        raise ValueError("stage 只支持 a1、a2、a3")
    if not cfg.target.strip():
        raise ValueError("target 不能为空")
    if cfg.num_queries < 1 or cfg.num_heads < 1:
        raise ValueError("num_queries 与 num_heads 必须 >= 1")
    if not (0 <= cfg.cross_dropout < 1) or not (0 <= cfg.text_proj_dropout < 1):
        raise ValueError("dropout 必须满足 0 <= p < 1")
    if cfg.embed_dim % cfg.num_heads:
        raise ValueError("embed_dim 必须能被 num_heads 整除")
    if cfg.batch_size < 1 or cfg.gradient_accumulation_steps < 1:
        raise ValueError("batch_size / accumulation_steps 必须 >= 1")
    if cfg.lr_schedule_total_epochs < 1 or cfg.additional_epochs < 0:
        raise ValueError("epoch 配置无效")
    if len(cfg.data_split_ratio) != 3 or abs(sum(cfg.data_split_ratio) - 1.0) > 1e-6:
        raise ValueError("data_split_ratio 必须为三个总和为 1 的数")
    if cfg.data_split_counts is not None:
        if len(cfg.data_split_counts) != 3 or min(cfg.data_split_counts) < 0:
            raise ValueError("data_split_counts 必须为三个非负整数")
    if not cfg.experiment_name.strip():
        raise ValueError("experiment_name 不能为空")
    if cfg.minimum_lr is not None and cfg.minimum_lr_ratio is not None:
        raise ValueError("minimum_lr / minimum_lr_ratio 只能二选一")
    if cfg.minimum_lr is None and cfg.minimum_lr_ratio is None:
        raise ValueError("minimum_lr / minimum_lr_ratio 必须至少填写一个")
    if cfg.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience 必须 >= 0")
    if cfg.use_stage2_dynamic_lr and cfg.additional_epochs < 1:
        raise ValueError("开启 stage2 LR 必须把 additional_epochs 设为正数")


def inspect_cache(cfg: Config) -> dict[str, Any]:
    """廉价预检；完整文本顺序 SHA256 会由 Dataset 在真实训练时再次严格检查。"""
    import numpy as np

    data_path = Path(cfg.data_path)
    cache_path = Path(cfg.text_cache_path)
    model_path = Path(cfg.text_model_path)
    mask_path = cache_path.with_name(cache_path.stem + "_mask.npy")
    meta_path = cache_path.with_suffix(cache_path.suffix + ".meta.json")
    required = [data_path, cache_path, mask_path, meta_path, model_path / "config.json"]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("必要文件不存在:\n  " + "\n  ".join(missing))

    with meta_path.open(encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("cache_format") != "fusiformer-v2-segments-v1":
        raise ValueError("错误的 V2 缓存格式")
    if meta.get("encoder_name") != "modernbert":
        raise ValueError("缓存编码器不是 modernbert")
    if meta.get("text_field") != cfg.text_field:
        raise ValueError("缓存 text_field 与本次训练配置不一致")
    if Path(str(meta.get("model_name", ""))).name != model_path.name:
        raise ValueError("模型文件夹与缓存记录的编码器文件夹不一致")
    if not meta.get("text_sha256"):
        raise ValueError("缓存元数据缺少 text_sha256")
    arr = np.load(cache_path, mmap_mode="r")
    mask = np.load(mask_path, mmap_mode="r")
    if arr.ndim != 3 or arr.shape[-1] != 768:
        raise ValueError(f"期待 ModernBERT [N,M,768]，实际 {arr.shape}")
    if arr.dtype not in (np.float16, np.float32):
        raise ValueError(f"不支持的缓存 dtype: {arr.dtype}")
    if mask.dtype != np.bool_ or mask.shape != arr.shape[:2]:
        raise ValueError("语义 Mask 形状或 dtype 错误")
    if list(meta.get("shape", [])) != list(arr.shape):
        raise ValueError("缓存 metadata.shape 不一致")
    if int(meta.get("hidden_size", -1)) != 768:
        raise ValueError("缓存元数据 hidden_size != 768")
    if cfg.data_split_counts is not None and sum(cfg.data_split_counts) > arr.shape[0]:
        raise ValueError("自定义 data_split_counts 总数超过缓存行数")

    result = {"cache_shape": tuple(arr.shape), "cache_dtype": str(arr.dtype),
              "mask_shape": tuple(mask.shape), "data_path": str(data_path),
              "text_sha256": meta["text_sha256"], "cache_path": str(cache_path)}
    del arr, mask
    return result


# =============================================================================
#         仅本脚本安装 V2 训练钩子；不依赖另外两个 V2 训练启动文件
# =============================================================================
def setup_training(cfg: Config):
    import numpy as np
    import train_fusiformer_production_editor_v6 as baseline
    from fusiformer_v2.dataset import get_dataset_v2
    from fusiformer_v2.model import FusiformerV2

    engine = baseline.engine

    # 将顶层可编辑配置准确映射到现有 v6/v4 引擎的同名变量。
    overrides = {
        "training_mode": "scratch",
        "resume_checkpoint_path": None,
        "weights_checkpoint_path": None,
        "eval_checkpoint_path": None,
        "run_log_dir": None,
        "output_root": Path(cfg.output_root),
        "data_path": cfg.data_path,
        "target_name": cfg.target,
        "data_split_ratio": list(cfg.data_split_ratio),
        "data_split_counts": None if cfg.data_split_counts is None else list(cfg.data_split_counts),
        "text_encoder_name": "modernbert",
        "text_model_name": cfg.text_model_path,
        "text_pooling": "mean",                  # 缓存已完成 Late Chunking
        "text_preprocessing": "none",
        "text_field": cfg.text_field,
        "text_max_length": 8192,
        "freeze_text_encoder": True,
        "allow_remote_model": False,
        "use_precomputed_text_embeddings": True,
        "text_embedding_cache_path": cfg.text_cache_path,
        "text_embedding_meta_path": None,
        "exclude_frozen_text_encoder_from_checkpoint": True,
        "use_text": True,
        "model_depth": cfg.model_depth,
        "edge_input_dim": cfg.edge_input_dim,
        "triplet_input_dim": cfg.triplet_input_dim,
        "embed_dim": cfg.embed_dim,
        "num_heads": cfg.num_heads,
        "text_proj_dim": cfg.text_proj_dim,
        "text_proj_hidden_dims": tuple(cfg.text_proj_hidden_dims),
        "text_proj_dropout": cfg.text_proj_dropout,
        "batch_size": cfg.batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "lr_schedule_total_epochs": cfg.lr_schedule_total_epochs,
        "additional_epochs": cfg.additional_epochs,
        "total_epochs": cfg.lr_schedule_total_epochs + cfg.additional_epochs,
        "use_stage2_dynamic_lr": cfg.use_stage2_dynamic_lr,
        "stage2_peak_lr": cfg.stage2_peak_lr,
        "stage2_min_lr": cfg.stage2_min_lr,
        "stage2_warmup_epochs": cfg.stage2_warmup_epochs,
        "scale_lr_with_accumulation": cfg.scale_lr_with_accumulation,
        "absolute_lr": cfg.absolute_lr,
        "base_lr": cfg.base_lr,
        "minimum_lr": cfg.minimum_lr,
        "minimum_lr_ratio": cfg.minimum_lr_ratio,
        "warmup_ratio": cfg.warmup_ratio,
        "warmup_steps_override": cfg.warmup_steps_override,
        "warmup_initial_lr": cfg.warmup_initial_lr,
        "strict_resume_schedule": True,
        "weight_decay": cfg.weight_decay,
        "adam_beta1": cfg.adam_beta1,
        "adam_beta2": cfg.adam_beta2,
        "use_module_lr_scales": cfg.use_module_lr_scales,
        "backbone_lr_scale": cfg.backbone_lr_scale,
        "text_module_lr_scale": cfg.text_module_lr_scale,
        "regression_head_lr_scale": cfg.regression_head_lr_scale,
        "text_encoder_lr_scale": cfg.text_encoder_lr_scale,
        "clip_grad_norm": cfg.clip_grad_norm,
        "device_name": cfg.device_name,
        "random_seed": cfg.random_seed,
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.pin_memory,
        "drop_last": cfg.drop_last,
        "use_amp": cfg.use_amp,
        "use_deterministic_algorithms": cfg.use_deterministic_algorithms,
        "best_model_monitor": cfg.best_model_monitor,
        "save_interval": cfg.save_interval,
        "log_interval": cfg.log_interval,
        "early_stopping_patience": cfg.early_stopping_patience,
        "enable_tensorboard": cfg.enable_tensorboard,
        "evaluate_test_each_epoch": cfg.evaluate_test_each_epoch,
        "SCRIPT_VERSION": VERSION,
    }
    for name, value in overrides.items():
        if not hasattr(baseline, name):
            raise AttributeError(f"v6 训练引擎缺少预期配置字段 {name!r}；请核对代码版本")
        setattr(baseline, name, value)

    baseline.get_dataset = get_dataset_v2
    runtime_parent = baseline.build_runtime_config
    config_parent = baseline.build_training_config
    clone_parent = engine.clone_inputs_without_text

    def build_runtime_config():
        args = runtime_parent()
        args.v2_stage = cfg.stage
        args.v2_num_queries = int(cfg.num_queries)
        args.v2_cross_dropout = float(cfg.cross_dropout)
        if args.mode != "scratch":
            raise ValueError("当前独立训练入口仅支持 scratch，禁止加载不兼容的 A0 权重")
        if args.text_encoder_name != "modernbert" or not args.freeze_text_encoder:
            raise ValueError("ModernBERT 必须保持冻结")
        if not args.use_precomputed_text_embeddings:
            raise ValueError("必须启用 rank-3 语义缓存")
        return args

    def build_model(args, device):
        arr = np.load(args.text_embedding_cache_path, mmap_mode="r")
        if arr.ndim != 3 or arr.shape[-1] != 768:
            raise ValueError(f"ModernBERT 缓存必须为 [N,M,768]，实际 {arr.shape}")
        model = FusiformerV2(
            targets=[args.target],
            depth=args.depth,
            edge_input_dim=args.edge_input_dim,
            triplet_input_dim=args.triplet_input_dim,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            text_dim=int(arr.shape[-1]),
            text_proj_dim=args.text_proj_dim,
            text_proj_hidden_dims=args.text_proj_hidden_dims,
            text_proj_dropout=args.text_proj_dropout,
            stage=args.v2_stage,
            num_queries=args.v2_num_queries,
            cross_dropout=args.v2_cross_dropout,
            text_encoder_name=args.text_encoder_name,
            text_model_name=args.text_model_name,
            text_pooling=args.text_pooling,
            text_preprocessing=args.text_preprocessing,
            load_text_encoder=False,
            freeze_text_encoder=True,
        )
        model.use_precomputed_text_embeddings = True
        return model.to(device)

    def clone_inputs_without_text(inputs):
        cloned = clone_parent(inputs)
        if isinstance(cloned, dict):
            cloned["text_segments"] = None
            cloned["text_segment_mask"] = None
            cloned["text_mode"] = "bypass"
        return cloned

    def parameter_lr_category(name: str) -> str:
        if name.startswith(("cross_modal.", "v2_text_to_graph.", "material_queries.")):
            return "text"
        return baseline.parameter_lr_category(name)

    def build_training_config(*args, **kwargs):
        payload = config_parent(*args, **kwargs)
        payload.update({
            "fusiformer_version": "V2.1",
            "v2_stage": cfg.stage,
            "v2_num_queries": cfg.num_queries,
            "v2_cross_dropout": cfg.cross_dropout,
            "v2_semantic_cache": cfg.text_cache_path,
            "text_encoder_runtime_loaded": False,
            "independent_script": Path(__file__).name,
            "independent_script_version": VERSION,
            "experiment_name": cfg.experiment_name,
            "independent_user_config": asdict(cfg),
        })
        return payload

    # v6 自带的 build_dataloaders 创建原版 DGL graph/line_graph transforms；
    # 仅 get_dataset 使用 V2 缓存语义图 Dataset，保持输出与 baseline 一致。
    engine.build_runtime_config = build_runtime_config
    engine.build_model = build_model
    engine.build_dataloaders = baseline.build_dataloaders
    engine.clone_inputs_without_text = clone_inputs_without_text
    engine.parameter_lr_category = parameter_lr_category
    engine.build_training_config = build_training_config
    engine.SCRIPT_VERSION = VERSION
    return engine


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
