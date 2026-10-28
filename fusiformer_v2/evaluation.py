"""Checkpoint-only, per-sample evaluation for the existing ModernBERT V2 model."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


MODEL_KEYS = (
    "model_name", "target", "stage", "num_queries", "cross_dropout", "model_depth",
    "edge_input_dim", "triplet_input_dim", "embed_dim", "num_heads",
    "text_dim", "text_proj_dim", "text_proj_hidden_dims", "text_proj_dropout",
    "text_encoder_name", "text_model_path", "text_pooling", "text_preprocessing",
    "text_mode",
)
DATA_KEYS = (
    "data_path", "text_cache_path", "text_field", "data_split_ratio",
    "data_split_counts", "batch_size", "num_workers", "device_name",
    "pin_memory", "use_amp",
)


def validate_settings(split: str, model_cfg: dict, data_cfg: dict,
                      checkpoint_cfg: dict, output_cfg: dict) -> tuple[Path, Path]:
    """Fail before importing optional graph/HF dependencies or writing results."""
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    for name in MODEL_KEYS:
        if name not in model_cfg:
            raise ValueError(f"MODEL_CONFIG.{name} is required")
    for name in DATA_KEYS:
        if name not in data_cfg:
            raise ValueError(f"DATA_CONFIG.{name} is required")
    if model_cfg["stage"] not in {"a1", "a2", "a3"}:
        raise ValueError("MODEL_CONFIG.stage must be a1, a2, or a3")
    if model_cfg["model_name"] != "FusiformerV2.1-ModernBERT":
        raise ValueError("MODEL_CONFIG.model_name must identify the supported V2.1 ModernBERT model")
    if (model_cfg["text_encoder_name"], model_cfg["text_pooling"], model_cfg["text_preprocessing"]) != ("modernbert", "mean", "none"):
        raise ValueError("MODEL_CONFIG text encoder settings must match the V2.1 ModernBERT cached training path")
    if model_cfg["text_mode"] not in {"normal", "bypass"}:
        raise ValueError("MODEL_CONFIG.text_mode must be normal or bypass")
    if model_cfg["num_heads"] < 1 or model_cfg["embed_dim"] < 1 or model_cfg["embed_dim"] % model_cfg["num_heads"]:
        raise ValueError("MODEL_CONFIG.embed_dim must be divisible by num_heads")
    if not model_cfg["target"] or data_cfg["batch_size"] < 1 or data_cfg["num_workers"] < 0:
        raise ValueError("target/batch_size/num_workers configuration is invalid")
    ratios, counts = data_cfg["data_split_ratio"], data_cfg["data_split_counts"]
    if counts is None:
        if len(ratios) != 3 or any(x < 0 for x in ratios) or abs(sum(ratios) - 1) > 1e-6:
            raise ValueError("DATA_CONFIG.data_split_ratio must contain three fractions summing to 1")
    elif len(counts) != 3 or any(x < 0 for x in counts):
        raise ValueError("DATA_CONFIG.data_split_counts must contain three non-negative integers")
    checkpoint_path = checkpoint_cfg.get("path")
    if not checkpoint_path:
        raise ValueError("Set CHECKPOINT_CONFIG['path'] to an explicit V2.1 checkpoint file")
    checkpoint_path = Path(checkpoint_path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"CHECKPOINT_CONFIG.path does not exist: {checkpoint_path}")
    if checkpoint_cfg.get("weight_key") != "model_state_dict":
        raise ValueError("CHECKPOINT_CONFIG.weight_key must be 'model_state_dict' for v4 V2.1 checkpoints")
    if checkpoint_cfg.get("expected_format_version") != 4:
        raise ValueError("CHECKPOINT_CONFIG.expected_format_version must be 4")
    if checkpoint_cfg.get("expected_model_version") != "V2.1":
        raise ValueError("CHECKPOINT_CONFIG.expected_model_version must be 'V2.1'")
    if checkpoint_cfg.get("strict") is not True:
        raise ValueError("CHECKPOINT_CONFIG.strict must be True")
    if not data_cfg["data_path"] or not Path(data_cfg["data_path"]).is_file():
        raise FileNotFoundError(f"DATA_CONFIG.data_path does not exist: {data_cfg['data_path']}")
    if not data_cfg["text_cache_path"] or not Path(data_cfg["text_cache_path"]).is_file():
        raise FileNotFoundError(f"DATA_CONFIG.text_cache_path does not exist: {data_cfg['text_cache_path']}")
    output_dir = Path(output_cfg["directory"]).expanduser()
    names = [output_cfg[key] for key in ("metrics_filename", "predictions_filename", "metadata_filename")]
    if len(set(names)) != 3 or any(Path(name).name != name or name in {"", ".", ".."} for name in names):
        raise ValueError("OUTPUT_CONFIG filenames must be three distinct simple filenames")
    for name in names:
        if (output_dir / name).exists():
            raise FileExistsError(f"Refusing to overwrite existing evaluation result: {output_dir / name}")
    return checkpoint_path, output_dir


def _check_checkpoint(checkpoint: Any, model_cfg: dict, data_cfg: dict,
                      checkpoint_cfg: dict) -> dict:
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a v4 dictionary, not a bare tensor/object")
    if checkpoint.get("format_version") != checkpoint_cfg["expected_format_version"]:
        raise ValueError(f"Checkpoint format_version={checkpoint.get('format_version')!r}; expected 4; fields={list(checkpoint)}")
    key = checkpoint_cfg["weight_key"]
    state = checkpoint.get(key)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Checkpoint lacks nonempty {key!r}; fields={list(checkpoint)}")
    if checkpoint.get("excluded_model_prefixes"):
        raise ValueError("Checkpoint excludes model parameters; strict V2.1 evaluation requires a complete state_dict")
    saved = checkpoint.get("training_config")
    if not isinstance(saved, dict) or saved.get("fusiformer_version") != checkpoint_cfg["expected_model_version"]:
        raise ValueError("Checkpoint lacks V2.1 training_config; refusing unverified A0/legacy weights")
    original = saved.get("independent_user_config")
    if not isinstance(original, dict):
        raise ValueError("Checkpoint lacks independent_user_config; cannot verify V2.1 architecture/split")
    same_fields = (
        "target", "stage", "num_queries", "cross_dropout", "model_depth",
        "edge_input_dim", "triplet_input_dim", "embed_dim", "num_heads",
        "text_proj_dim", "text_proj_hidden_dims", "text_proj_dropout",
    )
    for name in same_fields:
        current = model_cfg[name]
        previous = original.get(name)
        if isinstance(current, (tuple, list)):
            current, previous = list(current), list(previous) if isinstance(previous, (tuple, list)) else previous
        if current != previous:
            raise ValueError(f"Checkpoint architecture mismatch at {name}: saved={previous!r}, configured={current!r}")
    for name in ("text_field", "data_split_ratio", "data_split_counts"):
        current = data_cfg[name]
        previous = original.get(name)
        if isinstance(current, (tuple, list)):
            current, previous = list(current), list(previous) if isinstance(previous, (tuple, list)) else previous
        if current != previous:
            raise ValueError(f"Checkpoint data/split mismatch at {name}: saved={previous!r}, configured={current!r}")
    if Path(str(original.get("text_model_path", ""))).name != Path(str(model_cfg["text_model_path"])).name:
        raise ValueError("Checkpoint text_model_path basename differs from MODEL_CONFIG.text_model_path")
    return state


def select_dataset(datasets: tuple[Any, Any, Any], split: str) -> Any:
    """Select the existing loader's validation/test slot, never a fallback."""
    if len(datasets) != 3 or split not in {"validation", "test"}:
        raise ValueError("Expected train/validation/test datasets and an explicit evaluation split")
    selected = datasets[1 if split == "validation" else 2]
    if selected is None or len(selected) == 0:
        raise ValueError(f"Selected {split} split is empty; no other split will be used")
    return selected


def evaluate_loader(model: Any, loader: Any, dataset: Any, device: Any,
                    *, text_mode: str, amp_enabled: bool) -> tuple[dict[str, float], list[dict]]:
    """Use v4's element-weighted metric implementation and preserve split order."""
    import torch
    from train_fusiformer_production_editor_v4 import (
        autocast_context, empty_metric_accumulator, finalize_metrics,
        normalize_outputs, normalize_targets, to_device, update_metric_accumulator,
    )

    if loader is None or dataset is None or len(dataset) == 0:
        raise ValueError("Selected evaluation split is empty")
    if not isinstance(loader.sampler, torch.utils.data.SequentialSampler) or loader.drop_last:
        raise ValueError("Evaluation loader must be sequential and drop_last=False")
    if len(dataset.original_indices) != len(dataset.data) or len(dataset.data) != len(dataset):
        raise ValueError("Dataset original indices and records are not aligned")
    model.eval()
    accumulator = empty_metric_accumulator()
    rows: list[dict] = []
    offset = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = to_device(inputs, device)
            inputs["text_mode"] = text_mode
            targets = normalize_targets(targets, device)
            with autocast_context(device, amp_enabled):
                outputs = normalize_outputs(model(inputs))
            if outputs.numel() != targets.numel() or outputs.numel() != targets.shape[0]:
                raise ValueError(f"Expected one prediction and label per sample; outputs={tuple(outputs.shape)}, targets={tuple(targets.shape)}")
            if not torch.isfinite(outputs).all() or not torch.isfinite(targets).all():
                raise ValueError("Non-finite prediction or label encountered; no results saved")
            outputs, targets = outputs.reshape(-1, 1), targets.reshape(-1, 1)
            update_metric_accumulator(accumulator, outputs, targets)
            predictions = outputs.detach().float().reshape(-1).cpu().tolist()
            labels = targets.detach().float().reshape(-1).cpu().tolist()
            for label, prediction in zip(labels, predictions):
                if offset >= len(dataset):
                    raise ValueError("More predictions than selected dataset records")
                record = dataset.data[offset]
                sample_id = next((record[key] for key in ("jid", "id", "material_id", "mp_id") if record.get(key) is not None), "")
                difference = prediction - label
                rows.append({
                    "original_index": int(dataset.original_indices[offset]),
                    "sample_id": str(sample_id),
                    "target": label,
                    "prediction": prediction,
                    "absolute_error": abs(difference),
                    "squared_error": difference * difference,
                })
                offset += 1
    if offset != len(dataset) or len({row["original_index"] for row in rows}) != len(dataset):
        raise ValueError("Prediction count or original sample index alignment failed")
    metrics = finalize_metrics(accumulator)
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("Non-finite aggregate metric encountered")
    return metrics, rows


def run_evaluation(split: str, model_cfg: dict, data_cfg: dict,
                   checkpoint_cfg: dict, output_cfg: dict) -> dict:
    """Create only the requested split loader; never invoke a training entry."""
    checkpoint_path, output_dir = validate_settings(
        split, model_cfg, data_cfg, checkpoint_cfg, output_cfg)
    import numpy as np
    import torch
    from fusiformer_v2.dataset import get_dataset_v2
    from fusiformer_v2.model import FusiformerV2
    from loader.loader_cached import GraphCollectingCached
    from loader.transforms import AddAngleFeature, AddEdgeFeature, AddNodeFeature, Compose
    from train_fusiformer_production_editor_v4 import load_torch_file

    device = torch.device(data_cfg["device_name"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DATA_CONFIG.device_name requests CUDA, but CUDA is unavailable")
    if data_cfg["use_amp"] and device.type != "cuda":
        raise ValueError("DATA_CONFIG.use_amp=True requires a CUDA device")
    transform = Compose([
        AddNodeFeature(), AddEdgeFeature(), AddAngleFeature(),
        GraphCollectingCached(["graph", "line_graph"], [model_cfg["target"]]),
    ])
    ratio = data_cfg["data_split_ratio"] if data_cfg["data_split_counts"] is None else None
    cache = np.load(data_cfg["text_cache_path"], mmap_mode="r")
    if cache.ndim != 3 or int(cache.shape[-1]) != model_cfg["text_dim"]:
        raise ValueError(f"Text cache shape {cache.shape} is incompatible with MODEL_CONFIG.text_dim={model_cfg['text_dim']}")
    counts = data_cfg["data_split_counts"]
    if counts is not None and sum(counts) > int(cache.shape[0]):
        raise ValueError("DATA_CONFIG.data_split_counts total exceeds the dataset/cache length")
    datasets = get_dataset_v2(
        data_path=data_cfg["data_path"], ratio_train_val_test=ratio,
        n_train_val_test=data_cfg["data_split_counts"], transforms=transform,
        graph=True, line_graph=True, text_embedding_cache_path=data_cfg["text_cache_path"],
        text_field=data_cfg["text_field"], text_encoder_name=model_cfg["text_encoder_name"],
        text_model_name=model_cfg["text_model_path"],
    )
    dataset = select_dataset(datasets, split)
    for original_index, record in zip(dataset.original_indices, dataset.data):
        raw_label = record.get(model_cfg["target"])
        try:
            valid_label = raw_label is not None and math.isfinite(float(raw_label))
        except (TypeError, ValueError):
            valid_label = False
        if not valid_label:
            raise ValueError(
                f"Selected {split} split has a missing/non-numeric/non-finite "
                f"{model_cfg['target']!r} label at original_index={original_index}")
    checkpoint = load_torch_file(checkpoint_path, map_location="cpu")
    state = _check_checkpoint(checkpoint, model_cfg, data_cfg, checkpoint_cfg)
    model = FusiformerV2(
        targets=[model_cfg["target"]], depth=model_cfg["model_depth"],
        edge_input_dim=model_cfg["edge_input_dim"],
        triplet_input_dim=model_cfg["triplet_input_dim"],
        embed_dim=model_cfg["embed_dim"], num_heads=model_cfg["num_heads"],
        text_dim=model_cfg["text_dim"], text_proj_dim=model_cfg["text_proj_dim"],
        text_proj_hidden_dims=model_cfg["text_proj_hidden_dims"],
        text_proj_dropout=model_cfg["text_proj_dropout"],
        stage=model_cfg["stage"], num_queries=model_cfg["num_queries"],
        cross_dropout=model_cfg["cross_dropout"], text_encoder_name=model_cfg["text_encoder_name"],
        text_model_name=model_cfg["text_model_path"], text_pooling=model_cfg["text_pooling"],
        text_preprocessing=model_cfg["text_preprocessing"],
        load_text_encoder=False, freeze_text_encoder=True,
    )
    model.use_precomputed_text_embeddings = True
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"Checkpoint model_state_dict is incompatible with configured FusiformerV2: {exc}") from exc
    model = model.to(device)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=data_cfg["batch_size"], num_workers=data_cfg["num_workers"],
        pin_memory=bool(data_cfg["pin_memory"]) and device.type == "cuda",
        shuffle=False, drop_last=False, collate_fn=dataset.collect(),
    )
    metrics, rows = evaluate_loader(
        model, loader, dataset, device, text_mode=model_cfg["text_mode"],
        amp_enabled=bool(data_cfg["use_amp"]))
    summary = {"split": split, "sample_count": len(rows), **metrics}
    metadata = {
        "split": split, "model_config": {k: str(v) if isinstance(v, Path) else v for k, v in model_cfg.items()},
        "data_config": {k: str(v) if isinstance(v, Path) else v for k, v in data_cfg.items()},
        "checkpoint_path": str(checkpoint_path.resolve()), "checkpoint_format_version": checkpoint["format_version"],
        "checkpoint_epoch": checkpoint.get("epoch"), "checkpoint_saved_at": checkpoint.get("saved_at"),
        "checkpoint_script": checkpoint["training_config"].get("independent_script"),
        "split_seed": 7, "sample_count": len(rows), "metrics": summary,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {key: output_dir / output_cfg[key] for key in
             ("metrics_filename", "predictions_filename", "metadata_filename")}
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing evaluation result: {path}")
    with paths["metrics_filename"].open("x", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    with paths["predictions_filename"].open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=(
            "original_index", "sample_id", "target", "prediction", "absolute_error", "squared_error"))
        writer.writeheader()
        writer.writerows(rows)
    with paths["metadata_filename"].open("x", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2, default=str)
    return {"metrics": summary, "paths": {key: str(path) for key, path in paths.items()}}
