#!/usr/bin/env python3
"""Checkpoint-only evaluation for one specialist or the complete routed system."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from scripts.experiments.cluster_specialists.checkpoints import load_checkpoint, manifest_fingerprint
from scripts.experiments.cluster_specialists.config import HistoricalMultiSourceConfig, HistoricalSourceConfig
from scripts.experiments.cluster_specialists.alignment import load_aligned_array
from scripts.experiments.cluster_specialists.evaluation import (
    summarize_paired, write_evaluation)
from scripts.experiments.cluster_specialists.manifests import read_manifest
from scripts.experiments.cluster_specialists.models import TensorMLPExpert
from scripts.experiments.cluster_specialists.routing import load_router

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class EvalConfig:
    evaluation_mode: str = "system"  # A | B | C | system
    prediction_mode: str = "direct"
    manifest_path: str = str(
        ROOT / "runs/cluster_specialists/manifests_seed42/cluster_manifest.csv")
    router_path: str = str(
        ROOT / "runs/cluster_specialists/router_seed42.json")
    router_features_path: str = str(
        ROOT / "data/cluster_router_features.npy")
    specialist_features_path: str = str(
        ROOT / "data/specialist_features.npy")
    targets_path: str = str(ROOT / "data/specialist_targets.npy")
    baseline_predictions_path: str = str(
        ROOT / "data/v2_baseline_predictions.npy")
    router_features_index_path: str = str(ROOT / "data/cluster_router_features_index.csv")
    specialist_features_index_path: str = str(ROOT / "data/specialist_features_index.csv")
    targets_index_path: str = str(ROOT / "data/specialist_targets_index.csv")
    baseline_predictions_index_path: str = str(ROOT / "data/v2_baseline_predictions_index.csv")
    checkpoints: tuple[tuple[str, str], ...] = (
        ("A", str(ROOT / "runs/cluster_specialists/checkpoints/A/best.pt")),
        ("B", str(ROOT / "runs/cluster_specialists/checkpoints/B/best.pt")),
        ("C", str(ROOT / "runs/cluster_specialists/checkpoints/C/best.pt")))
    input_dims: tuple[tuple[str, int], ...] = (
        ("A", 128), ("B", 128), ("C", 128))
    hidden_dims: tuple[tuple[str, tuple[int, ...]], ...] = (
        ("A", (128, 64)), ("B", (128, 64)), ("C", (128, 64)))
    dropouts: tuple[tuple[str, float], ...] = (
        ("A", 0.0), ("B", 0.0), ("C", 0.0))
    split: str = "test"
    batch_size: int = 256
    device: str = "cuda"
    output_dir: str = str(ROOT / "runs/cluster_specialists/evaluation")
    bootstrap_samples: int = 0
    random_seed: int = 42
    historical_source: HistoricalSourceConfig | HistoricalMultiSourceConfig | None = None


CFG = EvalConfig()


def _required_clusters(mode: str) -> tuple[str, ...]:
    if mode == "system":
        return ("A", "B", "C")
    if mode in {"A", "B", "C"}:
        return (mode,)
    raise ValueError("evaluation_mode must be A, B, C or system")


def validate_config(cfg: EvalConfig) -> None:
    if cfg.prediction_mode not in {"direct", "residual"}:
        raise ValueError("prediction_mode must be direct or residual")
    if cfg.split not in {"validation", "test"}:
        raise ValueError("independent evaluation split must be validation or test")
    required = _required_clusters(cfg.evaluation_mode)
    paths = [
        cfg.manifest_path,
        cfg.specialist_features_path, cfg.targets_path,
        cfg.baseline_predictions_path,
        cfg.specialist_features_index_path, cfg.targets_index_path,
        cfg.baseline_predictions_index_path]
    if cfg.historical_source is None:
        paths.extend([cfg.router_path, cfg.router_features_path,
                      cfg.router_features_index_path])
    checkpoint_map = dict(cfg.checkpoints)
    paths.extend(checkpoint_map[cluster] for cluster in required)
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"required evaluation file missing: {missing[0]}")


def run(cfg: EvalConfig, *, check_config: bool = False) -> None:
    validate_config(cfg)
    rows = read_manifest(cfg.manifest_path)
    if cfg.historical_source is not None:
        historical = cfg.historical_source.load()
        if manifest_fingerprint(historical) != manifest_fingerprint(rows):
            raise ValueError("historical labels, identities or splits differ from the complete evaluation manifest")
        rows = historical
    else:
        router = load_router(cfg.router_path)
    required = _required_clusters(cfg.evaluation_mode)
    selected = [
        row for row in rows
        if row["split"] == cfg.split
        and (cfg.evaluation_mode == "system"
             or row["cluster_id"] == cfg.evaluation_mode)]
    if not selected:
        raise ValueError("selected evaluation population is empty")
    # Canonical sample order makes bootstrap resampling reproducible when the
    # manifest is reordered. Identity validation is still required for arrays.
    selected.sort(key=lambda row: row["sample_id"])
    features = load_aligned_array(cfg.specialist_features_path, cfg.specialist_features_index_path, rows, selected)
    targets = load_aligned_array(cfg.targets_path, cfg.targets_index_path, rows, selected)
    baseline = load_aligned_array(cfg.baseline_predictions_path, cfg.baseline_predictions_index_path, rows, selected)
    if targets.size != len(selected) or baseline.size != len(selected):
        raise ValueError("targets and baseline must contain one value per sample")
    targets, baseline = targets.reshape(-1), baseline.reshape(-1)
    saved = [row["cluster_id"] for row in selected]
    if cfg.historical_source is not None:
        routed = saved
    else:
        router_features = load_aligned_array(cfg.router_features_path, cfg.router_features_index_path, rows, selected)
        routed = router.predict(router_features).tolist()
        if routed != saved:
            raise ValueError(
                "router assignments differ from the immutable manifest")
    if check_config:
        print(json.dumps({
            "evaluation_mode": cfg.evaluation_mode, "split": cfg.split,
            "sample_count": len(selected), "loaded_clusters": required,
            "optimizer_created": False, "writes": False}, indent=2))
        return

    device = torch.device(cfg.device)
    checkpoint_map = dict(cfg.checkpoints)
    input_dims = dict(cfg.input_dims)
    hidden_dims = dict(cfg.hidden_dims)
    dropouts = dict(cfg.dropouts)
    models = {}
    for cluster in required:
        model = TensorMLPExpert(
            input_dims[cluster], hidden_dims[cluster], dropouts[cluster])
        load_checkpoint(
            checkpoint_map[cluster], model, expected_cluster=cluster,
            expected_mode=cfg.prediction_mode,
            expected_architecture="tensor_mlp",
            expected_manifest_fingerprint=manifest_fingerprint(rows),
            map_location="cpu")
        model.to(device).eval()
        models[cluster] = model

    predictions = np.empty(len(selected), dtype=np.float64)
    with torch.inference_mode():
        for cluster in required:
            local = [
                offset for offset, value in enumerate(routed)
                if value == cluster]
            if not local:
                continue
            chunks = []
            for start in range(0, len(local), cfg.batch_size):
                batch_positions = local[start:start + cfg.batch_size]
                batch = torch.as_tensor(
                    features[batch_positions], dtype=torch.float32,
                    device=device)
                chunks.extend(
                    models[cluster](batch).reshape(-1).cpu().tolist())
            predictions[local] = np.asarray(chunks)
    if cfg.prediction_mode == "residual":
        predictions += baseline
    result = summarize_paired(
        sample_ids=[row["sample_id"] for row in selected],
        original_indices=[int(row["original_index"]) for row in selected], cluster_ids=routed,
        targets=targets, baseline_predictions=baseline,
        specialist_predictions=predictions,
        bootstrap_samples=cfg.bootstrap_samples, seed=cfg.random_seed)
    for output_row, source_row in zip(result["rows"], selected):
        output_row.update(mp_id=source_row["mp_id"],
                          physical_id=source_row["physical_id"], split=source_row["split"])
        if cfg.historical_source is not None:
            output_row["source_label"] = source_row["source_label"]
            output_row["historical_raw_cluster_id"] = source_row.get("historical_raw_cluster_id", "")
    paths = write_evaluation(
        result, cfg.output_dir,
        prefix=f"{cfg.evaluation_mode}_{cfg.split}_{cfg.prediction_mode}")
    print(json.dumps(paths, ensure_ascii=False, indent=2))



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    run(CFG, check_config=args.check_config)

if __name__ == "__main__":
    main()
