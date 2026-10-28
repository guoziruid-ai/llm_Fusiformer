"""Compare isolated V2 evaluation outputs with a preserved historical baseline.

Example:
    python -m scripts.diagnostics.verify_evaluation_outputs \
      --baseline-runs /path/to/old-test/runs

Only reads metrics and prediction CSVs. It does not select a checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SPLITS = {
    "validation": "v21_validation_eval",
    "test": "v21_test_eval",
}
NUMERIC_COLUMNS = (
    "original_index", "target", "prediction", "absolute_error", "squared_error",
)


def _read_split(runs: Path, split: str) -> tuple[dict, list[dict[str, str]]]:
    folder = runs / SPLITS[split]
    with (folder / f"{split}_metrics.json").open(encoding="utf-8") as stream:
        metrics = json.load(stream)
    with (folder / f"{split}_predictions.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected_columns = set(NUMERIC_COLUMNS) | {"sample_id"}
    if not rows or set(rows[0]) != expected_columns:
        raise ValueError(f"{split}: missing rows or unexpected CSV columns")
    if metrics["split"] != split or len(rows) != metrics["sample_count"]:
        raise ValueError(f"{split}: metric/CSV sample count mismatch")
    indices = [int(row["original_index"]) for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError(f"{split}: duplicate original_index")
    return metrics, rows


def compare(baseline_runs: Path, candidate_runs: Path) -> dict:
    summary = {}
    index_sets = {}
    for split in SPLITS:
        old_metrics, old_rows = _read_split(baseline_runs, split)
        new_metrics, new_rows = _read_split(candidate_runs, split)
        if old_metrics != new_metrics:
            raise ValueError(f"{split}: aggregate metrics differ from preserved baseline")
        if len(old_rows) != len(new_rows):
            raise ValueError(f"{split}: prediction row count changed")
        for position, (old, new) in enumerate(zip(old_rows, new_rows)):
            if any(old[column] != new[column] for column in NUMERIC_COLUMNS):
                raise ValueError(f"{split}: numeric prediction changed at row {position}")
        missing_ids = sum(not row["sample_id"] for row in new_rows)
        if missing_ids:
            raise ValueError(f"{split}: {missing_ids} sample_id values remain empty")
        index_sets[split] = {int(row["original_index"]) for row in new_rows}
        summary[split] = {
            "sample_count": len(new_rows),
            "filled_sample_id": len(new_rows),
            "baseline_metrics_identical": True,
            "baseline_numeric_rows_identical": True,
        }
    overlap = index_sets["validation"] & index_sets["test"]
    if overlap:
        raise ValueError(f"Validation/test original_index overlap: {len(overlap)}")
    summary["index_overlap"] = 0
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-runs", type=Path, required=True)
    parser.add_argument(
        "--candidate-runs", type=Path,
        default=Path(__file__).resolve().parents[2] / "runs",
    )
    args = parser.parse_args()
    result = compare(args.baseline_runs, args.candidate_runs)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("PASS evaluation outputs match baseline except populated sample_id")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
