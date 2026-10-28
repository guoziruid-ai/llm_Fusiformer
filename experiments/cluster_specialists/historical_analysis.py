"""Descriptive analyses on explicitly identity-aligned historical assignments."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .metrics import regression_metrics


def _vector(values: Any, count: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size != count:
        raise ValueError(f"{name} must contain exactly one value per sample")
    array = array.reshape(-1)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    result = regression_metrics(target, prediction)
    return {key: result[key] for key in ("sample_count", "mae", "mse", "rmse")}


def _identity(row: dict[str, Any]) -> dict[str, Any]:
    fields = ("sample_id", "mp_id", "physical_id", "original_index", "split", "cluster_id", "source_label")
    result = {key: row[key] for key in fields}
    result["historical_raw_cluster_id"] = row.get("historical_raw_cluster_id", "")
    return result


def descriptive_analysis(rows: Sequence[dict[str, Any]], targets: Any,
                         baseline_predictions: Any, *, top_k: int = 10) -> dict[str, Any]:
    """Report observable distributions and absolute error, not causal claims."""
    if not rows or top_k < 0:
        raise ValueError("rows must be nonempty and top_k non-negative")
    y = _vector(targets, len(rows), "targets")
    p = _vector(baseline_predictions, len(rows), "baseline_predictions")
    errors = np.abs(p - y)
    labels = np.asarray([row["cluster_id"] for row in rows])
    result: dict[str, Any] = {"analysis_type": "descriptive", "sample_count": len(rows),
                              "clusters": {}, "pairwise_differences": {}}
    for cluster in ("A", "B", "C"):
        selected = labels == cluster
        vals, err = y[selected], errors[selected]
        if len(vals) == 0:
            result["clusters"][cluster] = {"sample_count": 0, "fraction": 0.0}
            continue
        result["clusters"][cluster] = {
            "sample_count": len(vals), "fraction": float(selected.mean()),
            "target": {"mean": float(vals.mean()), "std": float(vals.std()),
                       "min": float(vals.min()), "median": float(np.median(vals)),
                       "max": float(vals.max())},
            "baseline": _metrics(y[selected], p[selected]),
            "absolute_error": {"mean": float(err.mean()),
                               "median": float(np.median(err)),
                               "max": float(err.max())}}
    for left, right in (("A", "B"), ("A", "C"), ("B", "C")):
        a, b = result["clusters"][left], result["clusters"][right]
        if a["sample_count"] and b["sample_count"]:
            result["pairwise_differences"][f"{left}__{right}"] = {
                "target_mean": a["target"]["mean"] - b["target"]["mean"],
                "baseline_mae": a["baseline"]["mae"] - b["baseline"]["mae"]}
    paired = []
    for i, row in enumerate(rows):
        paired.append({**_identity(row),
                       "target": float(y[i]), "baseline_prediction": float(p[i]),
                       "baseline_absolute_error": float(errors[i])})
    result["largest_absolute_errors"] = sorted(
        paired, key=lambda row: (-row["baseline_absolute_error"], row["sample_id"]))[:top_k]
    result["limitations"] = "Descriptive only; no causal or confirmatory inference."
    return {"summary": result, "rows": paired}


def compare_mechanisms(rows: Sequence[dict[str, Any]], targets: Any,
                       predictions: dict[str, Any]) -> dict[str, Any]:
    """Pair existing predictions by identity; no model execution or fitting."""
    if not rows or "baseline" not in predictions or len(predictions) < 2:
        raise ValueError("baseline and at least one named mechanism are required")
    if any(not name or name.strip() != name for name in predictions):
        raise ValueError("mechanism names must be nonempty and trimmed")
    y = _vector(targets, len(rows), "targets")
    vectors = {name: _vector(value, len(rows), name) for name, value in predictions.items()}
    labels = np.asarray([row["cluster_id"] for row in rows])
    base_error = np.abs(vectors["baseline"] - y)
    summary: dict[str, Any] = {"analysis_type": "descriptive_paired_comparison", "overall": {}, "clusters": {}}
    for name, pred in vectors.items():
        summary["overall"][name] = _metrics(y, pred)
    for cluster in ("A", "B", "C"):
        selected = labels == cluster
        summary["clusters"][cluster] = {
            name: _metrics(y[selected], pred[selected]) for name, pred in vectors.items()
        } if selected.any() else {}
    paired = []
    for i, row in enumerate(rows):
        item = _identity(row)
        item["target"] = float(y[i])
        for name, pred in vectors.items():
            error = abs(float(pred[i] - y[i]))
            item[f"{name}_prediction"] = float(pred[i])
            item[f"{name}_absolute_error"] = error
            if name != "baseline":
                item[f"{name}_absolute_error_change"] = error - float(base_error[i])
        paired.append(item)
    summary["limitations"] = "Descriptive paired comparison; no causality or unapproved inferential metric."
    return {"summary": summary, "rows": paired}
