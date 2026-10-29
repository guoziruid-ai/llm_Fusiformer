#!/usr/bin/env python3
"""Read-only cluster quality, baseline-error and boundary diagnostics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.experiments.cluster_specialists.manifests import read_manifest
from scripts.experiments.cluster_specialists.alignment import load_aligned_array
from scripts.experiments.cluster_specialists.metrics import regression_metrics
from scripts.experiments.cluster_specialists.routing import CentroidRouter, load_router

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "runs/cluster_specialists/manifests_seed42/cluster_manifest.csv"
ROUTER_PATH = ROOT / "runs/cluster_specialists/router_seed42.json"
ROUTER_FEATURES_PATH = ROOT / "data/cluster_router_features.npy"
ROUTER_FEATURES_INDEX_PATH = ROOT / "data/cluster_router_features.index.csv"
TARGETS_PATH = ROOT / "data/specialist_targets.npy"
TARGETS_INDEX_PATH = ROOT / "data/specialist_targets.index.csv"
BASELINE_PREDICTIONS_PATH = ROOT / "data/v2_baseline_predictions.npy"
BASELINE_PREDICTIONS_INDEX_PATH = ROOT / "data/v2_baseline_predictions.index.csv"
SPLIT = "test"
BOUNDARY_FRACTION = 0.05
OUTPUT_PATH = ROOT / "runs/cluster_specialists/diagnostics/test_clusters.json"


def main() -> None:
    rows = read_manifest(MANIFEST_PATH)
    router = load_router(ROUTER_PATH)
    selected_rows = [row for row in rows if row["split"] == SPLIT]
    if not selected_rows:
        raise ValueError(f"diagnostic split {SPLIT!r} is empty or invalid")
    features = load_aligned_array(
        ROUTER_FEATURES_PATH, ROUTER_FEATURES_INDEX_PATH, rows, selected_rows)
    targets = load_aligned_array(
        TARGETS_PATH, TARGETS_INDEX_PATH, rows, selected_rows).reshape(-1)
    predictions = load_aligned_array(
        BASELINE_PREDICTIONS_PATH, BASELINE_PREDICTIONS_INDEX_PATH,
        rows, selected_rows).reshape(-1)
    decisions = router.predict_diagnostics(features)
    report = {
        "split": SPLIT, "sample_count": len(selected_rows), "clusters": {},
        "unassigned_count": 0,
        "out_of_domain_count": sum(item.out_of_domain for item in decisions)}
    margins = np.asarray([item.margin for item in decisions], dtype=float)
    threshold = float(np.quantile(
        margins, BOUNDARY_FRACTION)) if len(margins) else None
    report["boundary"] = {
        "definition": f"lowest {BOUNDARY_FRACTION:.1%} routing margins",
        "margin_threshold": threshold,
        "sample_count": int(np.sum(margins <= threshold)) if threshold is not None else 0}
    saved_clusters = np.asarray(
        [row["cluster_id"] for row in selected_rows])
    predicted_clusters = np.asarray(
        [item.cluster_id for item in decisions])
    report["assignment_reproducible"] = bool(
        np.array_equal(saved_clusters, predicted_clusters))
    if not report["assignment_reproducible"]:
        raise ValueError(
            "router assignments differ from the manifest; refusing diagnostics")
    absolute_errors = np.abs(
        predictions - targets)

    def distribution(values):
        values = np.asarray(values, dtype=float)
        if not len(values):
            return None
        return {
            "mean": float(values.mean()), "std": float(values.std()),
            "min": float(values.min()),
            "q50": float(np.quantile(values, 0.50)),
            "q90": float(np.quantile(values, 0.90)),
            "q95": float(np.quantile(values, 0.95)),
            "q99": float(np.quantile(values, 0.99)),
            "max": float(values.max())}

    for cluster in ("A", "B", "C"):
        selected = saved_clusters == cluster
        values = targets[selected]
        report["clusters"][cluster] = {
            "sample_count": int(selected.sum()),
            "fraction": float(selected.mean()) if len(selected) else 0.0,
            "target_mean": float(values.mean()) if len(values) else None,
            "target_std": float(values.std()) if len(values) else None,
            "target_distribution": distribution(values),
            "baseline_absolute_error_distribution": distribution(
                absolute_errors[selected]),
            "baseline_metrics": (
                regression_metrics(
                    values, predictions[selected])
                if len(values) else None)}
    report["pairwise_cluster_differences"] = {}
    for left, right in (("A", "B"), ("A", "C"), ("B", "C")):
        left_stats = report["clusters"][left]
        right_stats = report["clusters"][right]
        report["pairwise_cluster_differences"][f"{left}__{right}"] = {
            "target_mean_difference": (
                left_stats["target_mean"] - right_stats["target_mean"]
                if left_stats["target_mean"] is not None
                and right_stats["target_mean"] is not None else None),
            "baseline_mae_difference": (
                left_stats["baseline_metrics"]["mae"]
                - right_stats["baseline_metrics"]["mae"]
                if left_stats["baseline_metrics"]
                and right_stats["baseline_metrics"] else None)}
    if isinstance(router, CentroidRouter):
        try:
            from sklearn.metrics import (
                calinski_harabasz_score, davies_bouldin_score,
                silhouette_score)
            matrix = np.asarray(features, dtype=float)
            standardized = (matrix - router.mean_) / router.scale_
            numeric = np.asarray([
                {"A": 0, "B": 1, "C": 2}[cluster]
                for cluster in predicted_clusters])
            report["internal_cluster_quality"] = {
                "evaluation_scope": SPLIT,
                "silhouette": float(
                    silhouette_score(standardized, numeric)),
                "calinski_harabasz": float(
                    calinski_harabasz_score(standardized, numeric)),
                "davies_bouldin": float(
                    davies_bouldin_score(standardized, numeric))}
        except ValueError as exc:
            report["internal_cluster_quality"] = {
                "evaluation_scope": SPLIT, "unavailable": str(exc)}
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"refusing to overwrite diagnostic report: {OUTPUT_PATH}")
    OUTPUT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
