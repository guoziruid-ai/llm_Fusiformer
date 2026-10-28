"""Generate nine index-only datasets and prove their split/ID isolation."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import SplitRatios

CLUSTERS = ("A", "B", "C")
SPLITS = ("train", "validation", "test")


def infer_sample_id(record: dict[str, Any], original_index: int) -> str:
    for key in ("mp_id", "material_id", "jid", "id"):
        if record.get(key) not in (None, ""):
            return str(record[key])
    return f"index:{original_index}"


def infer_physical_id(record: dict[str, Any], original_index: int,
                      physical_id_fields: Sequence[str], *,
                      require_explicit: bool = False) -> str:
    for key in physical_id_fields:
        if record.get(key) not in (None, ""):
            return str(record[key])
    if require_explicit:
        raise ValueError(f"missing physical ID for original_index {original_index}; configure physical_id_fields")
    return infer_sample_id(record, original_index)


def _group_order(cluster: str, physical_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{cluster}:{physical_id}".encode()).hexdigest()


def _allocate_groups(groups: list[tuple[str, list[int]]],
                     ratios: SplitRatios) -> dict[int, str]:
    targets = ratios.as_dict()
    total = sum(len(indices) for _, indices in groups)
    desired = {name: total * ratio for name, ratio in targets.items()}
    counts = {name: 0 for name in SPLITS}
    assignment: dict[int, str] = {}
    for _, indices in groups:
        split = max(
            SPLITS,
            key=lambda name: (desired[name] - counts[name], -SPLITS.index(name)))
        for index in indices:
            assignment[index] = split
        counts[split] += len(indices)
    return assignment


def assign_physical_splits(
        records: Sequence[dict[str, Any]], *, ratios: SplitRatios = SplitRatios(),
        seed: int = 42,
        physical_id_fields: Sequence[str] = ("mp_id", "material_id", "jid", "id")) -> list[str]:
    """Fix splits by explicit physical identity before any data-driven fitting."""
    ratios.validate()
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        physical_id = infer_physical_id(
            record, index, physical_id_fields, require_explicit=True)
        grouped[physical_id].append(index)
    ordered = sorted(grouped.items(), key=lambda item: _group_order("all", item[0], seed))
    assigned = _allocate_groups(ordered, ratios)
    return [assigned[index] for index in range(len(records))]


def build_manifest(
        records: Sequence[dict[str, Any]], cluster_ids: Sequence[str], *,
        ratios: SplitRatios = SplitRatios(), seed: int = 42,
        physical_id_fields: Sequence[str] = ("mp_id", "material_id", "jid", "id"),
        target_field: str | None = None,
        splits: Sequence[str] | None = None) -> list[dict[str, Any]]:
    ratios.validate()
    if len(records) != len(cluster_ids):
        raise ValueError("records and cluster_ids have different lengths")
    if any(cluster not in CLUSTERS for cluster in cluster_ids):
        raise ValueError("every sample must have cluster A, B or C")
    source_ids = [
        infer_sample_id(record, index) for index, record in enumerate(records)]
    source_counts = Counter(source_ids)
    # Repeated physical records are allowed and must stay together. The manifest
    # still needs a unique row key, so duplicated source IDs receive a stable
    # original-index suffix while mp_id/physical_id remain unchanged.
    sample_ids = [
        source_id if source_counts[source_id] == 1
        else f"{source_id}#row{index}"
        for index, source_id in enumerate(source_ids)]
    physical_ids = [
        infer_physical_id(record, index, physical_id_fields)
        for index, record in enumerate(records)]
    physical_clusters: dict[str, set[str]] = defaultdict(set)
    for physical_id, cluster in zip(physical_ids, cluster_ids):
        physical_clusters[physical_id].add(cluster)
    conflicts = [
        key for key, values in physical_clusters.items() if len(values) > 1]
    if conflicts:
        raise ValueError(
            f"same physical sample received different clusters: {conflicts[:3]}")

    if splits is not None:
        if len(splits) != len(records) or any(split not in SPLITS for split in splits):
            raise ValueError("splits must contain one valid split per record")
        split_by_index = dict(enumerate(splits))
    else:
        split_by_index: dict[int, str] = {}
        for cluster in CLUSTERS:
            grouped: dict[str, list[int]] = defaultdict(list)
            for index, (physical_id, assigned) in enumerate(
                    zip(physical_ids, cluster_ids)):
                if assigned == cluster:
                    grouped[physical_id].append(index)
            ordered = sorted(
                grouped.items(),
                key=lambda item: _group_order(cluster, item[0], seed))
            split_by_index.update(_allocate_groups(ordered, ratios))

    rows = []
    for index, (record, cluster, sample_id, physical_id) in enumerate(
            zip(records, cluster_ids, sample_ids, physical_ids)):
        row: dict[str, Any] = {
            "sample_id": sample_id, "physical_id": physical_id,
            "mp_id": str(record.get("mp_id", "")), "original_index": index,
            "cluster_id": cluster, "split": split_by_index[index],
        }
        if target_field is not None:
            value = record.get(target_field)
            row["target"] = "" if value is None else value
        rows.append(row)
    validate_manifest(rows, expected_count=len(records))
    return rows


def validate_manifest(rows: Sequence[dict[str, Any]], *,
                      expected_count: int | None = None) -> dict[str, Any]:
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(
            f"manifest has {len(rows)} rows; expected {expected_count}")
    original = []
    samples = []
    for row in rows:
        for field in ("sample_id", "physical_id"):
            value = row.get(field)
            if value is None or not str(value).strip():
                raise ValueError(f"manifest contains empty {field}")
        value = row.get("original_index")
        if value is None or not re.fullmatch(r"[0-9]+", str(value)):
            raise ValueError("manifest original_index must be a non-negative integer")
        original.append(int(value))
        samples.append(str(row["sample_id"]))
    if len(set(original)) != len(rows) or len(set(samples)) != len(rows):
        raise ValueError(
            "manifest contains duplicate original_index or sample_id")
    invalid = [
        row for row in rows
        if row.get("cluster_id") not in CLUSTERS
        or row.get("split") not in SPLITS]
    if invalid:
        raise ValueError("manifest contains invalid cluster or split")
    physical_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        physical_splits[str(row["physical_id"])].add(str(row["split"]))
    leakage = sorted(
        key for key, values in physical_splits.items() if len(values) > 1)
    if leakage:
        raise ValueError(
            f"physical sample leakage across splits: {leakage[:3]}")
    split_sets = {
        split: {
            int(row["original_index"]) for row in rows
            if row["split"] == split}
        for split in SPLITS}
    intersections = {
        f"{left}__{right}": len(split_sets[left] & split_sets[right])
        for position, left in enumerate(SPLITS)
        for right in SPLITS[position + 1:]}
    if any(intersections.values()):
        raise ValueError(
            f"split intersections are non-zero: {intersections}")
    counts = {
        cluster: {
            split: sum(
                row["cluster_id"] == cluster and row["split"] == split
                for row in rows)
            for split in SPLITS}
        for cluster in CLUSTERS}
    cluster_counts = {
        cluster: sum(row["cluster_id"] == cluster for row in rows)
        for cluster in CLUSTERS}
    return {
        "row_count": len(rows), "cluster_counts": cluster_counts,
        "dataset_counts": counts, "split_intersections": intersections,
        "physical_id_leakage_count": 0, "unassigned_count": 0,
        "cluster_fractions": {
            key: value / len(rows) if rows else 0.0
            for key, value in cluster_counts.items()}}


def _distribution(rows: Sequence[dict[str, Any]]) -> dict[str, float | int | None]:
    values = []
    for row in rows:
        try:
            value = float(row.get("target", ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    if not values:
        return {"valid_count": 0, "mean": None, "std": None,
                "min": None, "max": None}
    array = np.asarray(values)
    return {"valid_count": len(array), "mean": float(array.mean()),
            "std": float(array.std()), "min": float(array.min()),
            "max": float(array.max())}


def write_manifests(rows: Sequence[dict[str, Any]],
                    output_dir: str | Path, *, seed: int,
                    ratios: SplitRatios, source: str) -> dict[str, str]:
    """Write one master CSV, nine index CSVs and a validation/statistics JSON."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    filename_split = {"train": "train", "validation": "val", "test": "test"}
    expected = [
        root / "cluster_manifest.csv",
        root / "cluster_split_report.json"] + [
        root / f"{cluster}_{filename_split[split]}.csv"
        for cluster in CLUSTERS for split in SPLITS]
    existing = [path for path in expected if path.exists()]
    if existing:
        raise FileExistsError(
            f"refusing to overwrite manifest outputs: {existing[0]}")
    fieldnames = (
        list(rows[0]) if rows else
        ["sample_id", "physical_id", "mp_id", "original_index",
         "cluster_id", "split"])

    def write_csv(path: Path, selected: Sequence[dict[str, Any]]) -> None:
        with path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(selected)

    write_csv(root / "cluster_manifest.csv", rows)
    paths = {"master": str(root / "cluster_manifest.csv")}
    for cluster in CLUSTERS:
        for split in SPLITS:
            name = f"{cluster}_{filename_split[split]}"
            selected = [
                row for row in rows
                if row["cluster_id"] == cluster and row["split"] == split]
            write_csv(root / f"{name}.csv", selected)
            paths[name] = str(root / f"{name}.csv")
    report = validate_manifest(rows)
    report.update({
        "random_seed": int(seed), "split_ratios": ratios.as_dict(),
        "source": source,
        "target_distributions": {
            f"{cluster}_{filename_split[split]}": _distribution([
                row for row in rows
                if row["cluster_id"] == cluster and row["split"] == split])
            for cluster in CLUSTERS for split in SPLITS}})
    report_path = root / "cluster_split_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["report"] = str(report_path)
    return paths


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    validate_manifest(rows)
    return rows
