#!/usr/bin/env python3
"""Analyze supplied historical assignments and existing paired predictions.

No clustering, inference, split generation, or statistical hypothesis testing.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from scripts.experiments.cluster_specialists.alignment import load_aligned_array
from scripts.experiments.cluster_specialists.config import HistoricalSourceConfig, ROOT
from scripts.experiments.cluster_specialists.historical_analysis import descriptive_analysis


@dataclass(frozen=True)
class Config:
    source: HistoricalSourceConfig = field(default_factory=HistoricalSourceConfig)
    targets_path: str = str(ROOT / "data/specialist_targets.npy")
    targets_index_path: str = str(ROOT / "data/specialist_targets_index.csv")
    baseline_path: str = str(ROOT / "data/v2_baseline_predictions.npy")
    baseline_index_path: str = str(ROOT / "data/v2_baseline_predictions_index.csv")
    split: str = "test"
    output_dir: str = str(ROOT / "runs/cluster_specialists/historical_analysis")
    top_k: int = 10
    plot: bool = False  # optional descriptive plots; requires matplotlib


CFG = Config()


def load_inputs(cfg: Config):
    for path in (cfg.source.master_path, cfg.source.labels_path,
                 cfg.targets_path, cfg.targets_index_path, cfg.baseline_path,
                 cfg.baseline_index_path):
        if not Path(path).is_file():
            raise FileNotFoundError(f"required historical analysis file missing: {path}")
    if cfg.split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    master = cfg.source.load()
    selected = sorted((row for row in master if row["split"] == cfg.split),
                      key=lambda row: row["sample_id"])
    if not selected:
        raise ValueError(f"no samples for requested split: {cfg.split}")
    targets = load_aligned_array(cfg.targets_path, cfg.targets_index_path, master, selected)
    baseline = load_aligned_array(cfg.baseline_path, cfg.baseline_index_path, master, selected)
    return master, selected, targets, baseline


def write_result(result, output_dir: str, *, plot: bool = False) -> None:
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError(f"refusing to reuse output directory: {root}")
    if plot:
        import matplotlib.pyplot as plt
    root.mkdir(parents=True, exist_ok=False)
    (root / "historical_summary.json").write_text(
        json.dumps(result["summary"], indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    with (root / "historical_samples.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["rows"][0]))
        writer.writeheader()
        writer.writerows(result["rows"])
    if plot:
        rows = result["rows"]
        clusters = [name for name in "ABC" if any(row["cluster_id"] == name for row in rows)]
        fig, axes = plt.subplots(1, 2)
        for axis, column, title in ((axes[0], "target", "Target"),
                                    (axes[1], "baseline_absolute_error", "Absolute error")):
            axis.boxplot([[row[column] for row in rows if row["cluster_id"] == name]
                          for name in clusters], labels=clusters)
            axis.set_title(title)
        fig.tight_layout()
        fig.savefig(root / "historical_distributions.png")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    _, rows, targets, baseline = load_inputs(CFG)
    result = descriptive_analysis(rows, targets, baseline, top_k=CFG.top_k)
    if args.check_config:
        print(json.dumps({"split": CFG.split, "sample_count": len(rows), "writes": False}))
        return
    write_result(result, CFG.output_dir, plot=CFG.plot)
    print(json.dumps({"output_dir": CFG.output_dir, "sample_count": len(rows)}))


if __name__ == "__main__":
    main()
