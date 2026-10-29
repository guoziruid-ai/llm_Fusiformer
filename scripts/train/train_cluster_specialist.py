#!/usr/bin/env python3
"""Train one independent tensor specialist from an index manifest.

This reference entry is for cached deployable features. A real graph/text V2
specialist uses the same checkpoint/adapter protocol but constructs
`FusiformerV2` through `build_v2_specialist` and the existing V2 loaders.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from scripts.experiments.cluster_specialists.checkpoints import (
    checkpoint_payload, manifest_fingerprint, save_checkpoint)
from scripts.experiments.cluster_specialists.config import HistoricalMultiSourceConfig, HistoricalSourceConfig
from scripts.experiments.cluster_specialists.alignment import load_aligned_array
from scripts.experiments.cluster_specialists.manifests import read_manifest
from scripts.experiments.cluster_specialists.models import TensorMLPExpert
from scripts.experiments.cluster_specialists.training import train_tensor_expert

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    cluster_id: str = "A"
    prediction_mode: str = "direct"  # direct | residual
    manifest_path: str = str(
        ROOT / "runs/cluster_specialists/manifests_seed42/cluster_manifest.csv")
    feature_path: str = str(ROOT / "data/specialist_features.npy")
    feature_index_path: str = str(ROOT / "data/specialist_features.index.csv")
    target_path: str = str(ROOT / "data/specialist_targets.npy")
    target_index_path: str = str(ROOT / "data/specialist_targets.index.csv")
    baseline_prediction_path: str | None = None
    baseline_prediction_index_path: str | None = None
    output_checkpoint: str = str(
        ROOT / "runs/cluster_specialists/checkpoints/A/best.pt")
    input_dim: int = 128
    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.0
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    loss_name: str = "mse"
    random_seed: int = 42
    device: str = "cuda"
    # Opt-in only. Legacy callers retain the original manifest workflow.
    historical_source: HistoricalSourceConfig | HistoricalMultiSourceConfig | None = None


CFG = Config()


def validate_config(cfg: Config) -> None:
    if cfg.cluster_id not in {"A", "B", "C"}:
        raise ValueError("cluster_id must be A, B or C")
    if cfg.prediction_mode not in {"direct", "residual"}:
        raise ValueError("prediction_mode must be direct or residual")
    for name in ("manifest_path", "feature_path", "feature_index_path",
                 "target_path", "target_index_path"):
        if not Path(getattr(cfg, name)).is_file():
            raise FileNotFoundError(f"{name} not found: {getattr(cfg, name)}")
    if cfg.prediction_mode == "residual" and (
            not cfg.baseline_prediction_path
            or not Path(cfg.baseline_prediction_path).is_file()):
        raise FileNotFoundError(
            "residual mode requires baseline_prediction_path")
    if cfg.baseline_prediction_path and (
            not cfg.baseline_prediction_index_path
            or not Path(cfg.baseline_prediction_index_path).is_file()):
        raise FileNotFoundError(
            "baseline predictions require baseline_prediction_index_path")
    if cfg.historical_source is not None and Path(cfg.output_checkpoint).exists():
        raise FileExistsError(f"refusing to overwrite specialist checkpoint: {cfg.output_checkpoint}")


def _training_rows(cfg: Config):
    rows = read_manifest(cfg.manifest_path)
    if cfg.historical_source is None:
        return rows
    historical = cfg.historical_source.load()
    if manifest_fingerprint(historical) != manifest_fingerprint(rows):
        raise ValueError("historical labels, identities or splits differ from the complete training manifest")
    return historical


def _loader(features: np.ndarray, targets: np.ndarray,
            baseline: np.ndarray | None, positions: list[int],
            batch_size: int, shuffle: bool) -> torch.utils.data.DataLoader:
    tensors = [
        torch.as_tensor(features[positions], dtype=torch.float32),
        torch.as_tensor(targets[positions], dtype=torch.float32)]
    if baseline is not None:
        tensors.append(torch.as_tensor(
            baseline[positions], dtype=torch.float32))
    dataset = torch.utils.data.TensorDataset(*tensors)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def run(cfg: Config, *, check_config: bool = False) -> None:
    validate_config(cfg)
    rows = _training_rows(cfg)
    features = load_aligned_array(
        cfg.feature_path, cfg.feature_index_path, rows, rows)
    targets = load_aligned_array(
        cfg.target_path, cfg.target_index_path, rows, rows).reshape(-1)
    baseline = (
        load_aligned_array(
            cfg.baseline_prediction_path, cfg.baseline_prediction_index_path,
            rows, rows).reshape(-1)
        if cfg.baseline_prediction_path else None)
    if len(features) != len(targets) or len(rows) != len(targets):
        raise ValueError("manifest/features/targets lengths differ")
    if features.ndim != 2 or features.shape[1] != cfg.input_dim:
        raise ValueError("specialist feature shape does not match input_dim")
    positions = {
        split: [
            position for position, row in enumerate(rows)
            if row["cluster_id"] == cfg.cluster_id and row["split"] == split]
        for split in ("train", "validation")}
    if not positions["train"] or not positions["validation"]:
        raise ValueError("selected cluster needs non-empty train and validation")
    if check_config:
        print(json.dumps({
            "cluster": cfg.cluster_id,
            "train": len(positions["train"]),
            "validation": len(positions["validation"]),
            "writes": False}, indent=2))
        return
    torch.manual_seed(cfg.random_seed)
    model = TensorMLPExpert(cfg.input_dim, cfg.hidden_dims, cfg.dropout)
    result = train_tensor_expert(
        model,
        _loader(features, targets, baseline, positions["train"],
                cfg.batch_size, True),
        _loader(features, targets, baseline, positions["validation"],
                cfg.batch_size, False),
        mode=cfg.prediction_mode, epochs=cfg.epochs,
        learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay,
        loss_name=cfg.loss_name, device=cfg.device)
    payload = checkpoint_payload(
        model, cluster_id=cfg.cluster_id,
        prediction_mode=cfg.prediction_mode,
        architecture="tensor_mlp",
        model_config={
            "input_dim": cfg.input_dim,
            "hidden_dims": cfg.hidden_dims, "dropout": cfg.dropout},
        training_config={
            "random_seed": cfg.random_seed, "loss_name": cfg.loss_name,
            "manifest_path": str(Path(cfg.manifest_path).resolve()),
            "manifest_sha256": manifest_fingerprint(rows)},
        epoch=result.best_epoch,
        metrics={"validation_loss": result.best_validation_loss})
    path = save_checkpoint(cfg.output_checkpoint, payload)
    print(f"[PASS] saved independent expert {cfg.cluster_id}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    run(CFG, check_config=args.check_config)

if __name__ == "__main__":
    main()
