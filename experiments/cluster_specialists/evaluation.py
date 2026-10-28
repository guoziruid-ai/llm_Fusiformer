"""Read-only paired evaluation for specialists and the routed system."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .metrics import (
    improvement_summary, paired_bootstrap_mae_improvement,
    paired_prediction_rows, regression_metrics)


def summarize_paired(
        *, sample_ids: Sequence[str], original_indices: Sequence[int],
        cluster_ids: Sequence[str], targets: Any,
        baseline_predictions: Any, specialist_predictions: Any,
        bootstrap_samples: int = 0, seed: int = 42) -> dict[str, Any]:
    """Compute E0/E1/E2-style global and per-cluster paired metrics."""
    y = np.asarray(targets, dtype=float).reshape(-1)
    baseline = np.asarray(baseline_predictions, dtype=float).reshape(-1)
    specialist = np.asarray(specialist_predictions, dtype=float).reshape(-1)
    clusters = np.asarray(cluster_ids, dtype=str)
    rows = paired_prediction_rows(
        sample_ids=sample_ids, original_indices=original_indices,
        targets=y, baseline_predictions=baseline,
        specialist_predictions=specialist, cluster_ids=clusters)
    result: dict[str, Any] = {
        "coverage": len(rows) / len(y) if len(y) else 0.0,
        "overall": improvement_summary(y, baseline, specialist),
        "clusters": {}}
    for cluster in ("A", "B", "C"):
        selected = clusters == cluster
        if not selected.any():
            result["clusters"][cluster] = {
                "sample_count": 0, "coverage": 0.0}
            continue
        result["clusters"][cluster] = {
            "coverage": float(selected.mean()),
            **improvement_summary(
                y[selected], baseline[selected], specialist[selected])}
    if bootstrap_samples:
        result["paired_bootstrap"] = paired_bootstrap_mae_improvement(
            y, baseline, specialist, samples=bootstrap_samples, seed=seed)
    return {"summary": result, "rows": rows}


def predict_read_only(model: torch.nn.Module, loader: Any, *,
                      device: str | torch.device = "cpu") -> np.ndarray:
    """Inference only: no optimizer, gradients, backward or checkpoint search."""
    device = torch.device(device)
    model.to(device)
    model.eval()
    outputs = []
    with torch.inference_mode():
        for batch in loader:
            inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
            if torch.is_tensor(inputs):
                inputs = inputs.to(device)
            outputs.append(
                model(inputs).detach().float().reshape(-1).cpu().numpy())
    if not outputs:
        raise ValueError("evaluation loader is empty")
    return np.concatenate(outputs)


def write_evaluation(result: dict[str, Any], output_dir: str | Path,
                     *, prefix: str) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    metrics_path = root / f"{prefix}_metrics.json"
    rows_path = root / f"{prefix}_paired_predictions.csv"
    if metrics_path.exists() or rows_path.exists():
        raise FileExistsError(
            "refusing to overwrite existing specialist evaluation outputs")
    metrics_path.write_text(
        json.dumps(result["summary"], ensure_ascii=False, indent=2),
        encoding="utf-8")
    rows = result["rows"]
    with rows_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"metrics": str(metrics_path), "predictions": str(rows_path)}
