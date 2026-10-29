"""Paired regression metrics and quantification helpers."""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


def _vector(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(array) == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be non-empty and finite")
    return array


def regression_metrics(targets: Any,
                       predictions: Any) -> dict[str, float | int]:
    y = _vector(targets, "targets")
    p = _vector(predictions, "predictions")
    if len(y) != len(p):
        raise ValueError(
            "targets and predictions have different lengths")
    error = p - y
    mse = float(np.mean(error ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    return {
        "sample_count": len(y), "mae": float(np.mean(np.abs(error))),
        "mse": mse, "rmse": math.sqrt(mse),
        "r2": (float(1.0 - np.sum(error ** 2) / total)
               if total else float("nan"))}


def relative_improvement(baseline: float, candidate: float) -> float:
    if not math.isfinite(baseline) or not math.isfinite(candidate):
        raise ValueError("metric values must be finite")
    if baseline == 0:
        return 0.0 if candidate == 0 else float("-inf")
    return (baseline - candidate) / baseline * 100.0


def improvement_summary(targets: Any, baseline_predictions: Any,
                        candidate_predictions: Any) -> dict[str, Any]:
    baseline = regression_metrics(targets, baseline_predictions)
    candidate = regression_metrics(targets, candidate_predictions)
    return {
        "baseline": baseline, "candidate": candidate,
        "absolute_change": {
            key: candidate[key] - baseline[key]
            for key in ("mae", "mse", "rmse")},
        "relative_improvement_percent": {
            key: relative_improvement(
                float(baseline[key]), float(candidate[key]))
            for key in ("mae", "mse", "rmse")}}


def paired_prediction_rows(
        *, sample_ids: Sequence[str], original_indices: Sequence[int],
        targets: Any, baseline_predictions: Any,
        specialist_predictions: Any,
        cluster_ids: Sequence[str]) -> list[dict[str, Any]]:
    y = _vector(targets, "targets")
    base = _vector(baseline_predictions, "baseline_predictions")
    expert = _vector(specialist_predictions, "specialist_predictions")
    n = len(y)
    if not all(
            len(values) == n for values in
            (base, expert, sample_ids, original_indices, cluster_ids)):
        raise ValueError("paired result columns have different lengths")
    rows = []
    for index in range(n):
        base_error = abs(float(base[index] - y[index]))
        expert_error = abs(float(expert[index] - y[index]))
        rows.append({
            "sample_id": str(sample_ids[index]),
            "original_index": int(original_indices[index]),
            "target": float(y[index]),
            "baseline_prediction": float(base[index]),
            "specialist_prediction": float(expert[index]),
            "cluster_id": str(cluster_ids[index]),
            "baseline_absolute_error": base_error,
            "specialist_absolute_error": expert_error,
            "absolute_error_change": expert_error - base_error})
    return rows


def paired_bootstrap_mae_improvement(
        targets: Any, baseline_predictions: Any, candidate_predictions: Any,
        *, samples: int = 2000, seed: int = 42,
        confidence: float = 0.95) -> dict[str, float | int | str]:
    y = _vector(targets, "targets")
    base = _vector(baseline_predictions, "baseline_predictions")
    candidate = _vector(candidate_predictions, "candidate_predictions")
    if len(y) != len(base) or len(y) != len(candidate) or samples < 1:
        raise ValueError(
            "paired arrays must align and samples must be positive")
    rng = np.random.default_rng(seed)
    changes = np.empty(samples, dtype=np.float64)
    for draw in range(samples):
        positions = rng.integers(0, len(y), size=len(y))
        changes[draw] = (
            np.mean(np.abs(base[positions] - y[positions]))
            - np.mean(np.abs(candidate[positions] - y[positions])))
    alpha = (1.0 - confidence) / 2.0
    return {
        "unit": "sample", "resamples": samples, "seed": seed,
        "mae_improvement": float(
            np.mean(np.abs(base - y)) - np.mean(np.abs(candidate - y))),
        "ci_low": float(np.quantile(changes, alpha)),
        "ci_high": float(np.quantile(changes, 1.0 - alpha))}
