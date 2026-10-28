"""Shared validation for the two ModernBERT V2.1 launchers.

The training launchers retain their separate defaults and engine hooks.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any


def validate_user_config(cfg: Any) -> None:
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


def inspect_cache(cfg: Any) -> dict[str, Any]:
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


def setup_training(cfg: Any, *, version: str, script_name: str):
    """Bind a ModernBERT V2 run to the existing v6/v4 training engine."""
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
        "SCRIPT_VERSION": version,
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
            "independent_script": script_name,
            "independent_script_version": version,
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
    engine.SCRIPT_VERSION = version
    return engine
