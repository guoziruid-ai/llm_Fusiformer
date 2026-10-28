#!/usr/bin/env python3
"""Compare supplied historical-label paired predictions; never execute models."""
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from experiments.cluster_specialists.alignment import load_aligned_array
from experiments.cluster_specialists.config import HistoricalSourceConfig, ROOT
from experiments.cluster_specialists.historical_analysis import compare_mechanisms


@dataclass(frozen=True)
class Config:
    source: HistoricalSourceConfig = field(default_factory=HistoricalSourceConfig)
    targets_path: str = str(ROOT / "data/specialist_targets.npy")
    targets_index_path: str = str(ROOT / "data/specialist_targets_index.csv")
    # Each NPY requires its own identity sidecar. No predictions are fabricated.
    predictions: tuple[tuple[str, str, str], ...] = (
        ("baseline", str(ROOT / "data/v2_baseline_predictions.npy"), str(ROOT / "data/v2_baseline_predictions_index.csv")),
        ("direct", str(ROOT / "data/direct_predictions.npy"), str(ROOT / "data/direct_predictions_index.csv")),
        ("residual", str(ROOT / "data/residual_predictions.npy"), str(ROOT / "data/residual_predictions_index.csv")),
    )
    split: str = "test"
    output_dir: str = str(ROOT / "runs/cluster_specialists/mechanism_comparison")


CFG = Config()


def load_inputs(cfg: Config):
    names = [item[0] for item in cfg.predictions]
    if len(names) != len(set(names)) or "baseline" not in names or len(names) < 2:
        raise ValueError("unique baseline and at least one comparison prediction are required")
    if any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) for name in names):
        raise ValueError("mechanism names must be simple alphanumeric identifiers")
    required = [cfg.source.master_path, cfg.source.labels_path, cfg.targets_path, cfg.targets_index_path]
    required.extend(path for _, array, index in cfg.predictions for path in (array, index))
    for path in required:
        if not Path(path).is_file():
            raise FileNotFoundError(f"required mechanism input missing: {path}")
    if cfg.split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    master = cfg.source.load()
    selected = sorted((row for row in master if row["split"] == cfg.split),
                      key=lambda row: row["sample_id"])
    if not selected:
        raise ValueError(f"no samples for requested split: {cfg.split}")
    targets = load_aligned_array(cfg.targets_path, cfg.targets_index_path, master, selected)
    predictions = {name: load_aligned_array(array, index, master, selected)
                   for name, array, index in cfg.predictions}
    return selected, targets, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    rows, targets, predictions = load_inputs(CFG)
    result = compare_mechanisms(rows, targets, predictions)
    if args.check_config:
        print(json.dumps({"sample_count": len(rows), "mechanisms": list(predictions), "writes": False}))
        return
    root = Path(CFG.output_dir)
    if root.exists():
        raise FileExistsError(f"refusing to reuse output directory: {root}")
    root.mkdir(parents=True, exist_ok=False)
    (root / "mechanisms_summary.json").write_text(
        json.dumps(result["summary"], ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with (root / "mechanisms_samples.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["rows"][0]))
        writer.writeheader()
        writer.writerows(result["rows"])
    print(json.dumps({"output_dir": str(root), "sample_count": len(rows)}))


if __name__ == "__main__":
    main()
