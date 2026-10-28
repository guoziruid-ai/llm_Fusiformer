"""Read historical labels through explicit identities, never by row position.

The files supplied here are evidence provided by the caller, not proof of the
history or provenance of a clustering run.  In particular, a historical
``train_position`` or ``val_position`` is not an original dataset index.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

from .manifests import CLUSTERS, validate_manifest


def verify_grouped3_mapping(
        labels_path: str | Path, mapping_path: str | Path, *,
        raw_col: str = "cluster_id", expert_id_col: str = "expert_id",
        expert_name_col: str = "expert_name") -> dict[str, str]:
    """Cross-check actual Grouped-3 route rows against its saved mapping.

    Returns explicit expert-ID to A/B/C mapping for ``load_historical_rows``.
    This checks internal consistency, not run provenance or split membership.
    """
    path = Path(mapping_path)
    if not path.is_file():
        raise FileNotFoundError(f"required Grouped-3 mapping not found: {path}")
    with path.open(encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Grouped-3 mapping must be a JSON object")
    cluster_to_expert = data.get("cluster_to_expert_id")
    expert_names = data.get("expert_id_to_name")
    if not isinstance(cluster_to_expert, list) or len(cluster_to_expert) != 5:
        raise ValueError("Grouped-3 cluster_to_expert_id must contain five raw-cluster assignments")
    if any(type(value) is not int or value not in (0, 1, 2) for value in cluster_to_expert):
        raise ValueError("Grouped-3 cluster_to_expert_id contains an unknown expert ID")
    expected_names = {"expert_A_shared3": "A", "expert_B_hardest": "B", "expert_C_second_hardest": "C"}
    if not isinstance(expert_names, dict) or set(expert_names) != {"0", "1", "2"}:
        raise ValueError("Grouped-3 expert_id_to_name must define IDs 0, 1 and 2")
    if any(not isinstance(name, str) for name in expert_names.values()) or set(expert_names.values()) != set(expected_names):
        raise ValueError("Grouped-3 expert names must uniquely identify the three documented experts")
    explicit_mapping = {key: expected_names[name] for key, name in expert_names.items()}
    if set(cluster_to_expert) != {0, 1, 2}:
        raise ValueError("Grouped-3 cluster mapping must use all three expert IDs")
    # If the artifact additionally provides its inverse, validate that too.
    if "expert_to_clusters" in data:
        inverse = data["expert_to_clusters"]
        expected_inverse = {
            str(expert): [raw for raw, assigned in enumerate(cluster_to_expert) if assigned == expert]
            for expert in range(3)}
        if not isinstance(inverse, dict) or set(inverse) != set(expected_inverse):
            raise ValueError("Grouped-3 expert_to_clusters is inconsistent with the forward mapping")
        for expert, clusters in inverse.items():
            if (not isinstance(clusters, list)
                    or any(type(raw) is not int for raw in clusters)
                    or len(clusters) != len(set(clusters))
                    or sorted(clusters) != expected_inverse[expert]):
                raise ValueError("Grouped-3 expert_to_clusters is inconsistent with the forward mapping")
    labels = _read_csv(labels_path, {raw_col, expert_id_col, expert_name_col})
    for row in labels:
        raw = _key(row, raw_col, "Grouped-3 routes")
        expert = _key(row, expert_id_col, "Grouped-3 routes")
        name = _key(row, expert_name_col, "Grouped-3 routes")
        if raw not in {str(i) for i in range(5)}:
            raise ValueError(f"unknown Grouped-3 raw cluster ID: {raw!r}")
        expected_expert = str(cluster_to_expert[int(raw)])
        if expert != expected_expert or name != expert_names[expected_expert]:
            raise ValueError(f"Grouped-3 route mapping conflict for raw cluster {raw!r}")
    return explicit_mapping


def _read_csv(path: str | Path, required: set[str]) -> list[dict[str, str]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"required historical input not found: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if not fields or len(fields) != len(set(fields)):
            raise ValueError(f"missing or duplicate CSV headers in {path}")
        missing = required - set(fields)
        if missing:
            raise ValueError(f"{path} missing fields: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} has no rows")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"{path} contains malformed CSV rows")
    return rows


def _key(row: Mapping[str, Any], column: str, source: str) -> str:
    value = row.get(column)
    if value is None or not str(value).strip():
        raise ValueError(f"{source} contains missing {column}")
    value = str(value)
    if value != value.strip():
        raise ValueError(f"{source} has whitespace around {column}: {value!r}")
    return value


def _unique(rows: list[dict[str, str]], column: str, source: str) -> dict[str, dict[str, str]]:
    result = {}
    for row in rows:
        key = _key(row, column, source)
        if key in result:
            raise ValueError(f"{source} contains duplicate {column}: {key!r}")
        result[key] = row
    return result


def _same_members(actual: set[str], expected: set[str], source: str) -> None:
    missing, unknown = expected - actual, actual - expected
    if missing or unknown:
        raise ValueError(
            f"{source} ID mismatch: missing={sorted(missing)[:5]} "
            f"unknown={sorted(unknown)[:5]} (counts {len(missing)}/{len(unknown)})")


def _check_identity(row: Mapping[str, str], master: Mapping[str, str], source: str) -> None:
    # If an input carries identity metadata, it must agree; never silently
    # discard an incompatible index, physical ID or split.
    for field in ("sample_id", "physical_id", "mp_id", "original_index", "split"):
        if field in row and row[field] != master[field]:
            raise ValueError(f"{source} {field} mismatch for {master['sample_id']!r}")


def load_historical_rows(
        master_path: str | Path, labels_path: str | Path, *,
        master_id_col: str = "sample_id", label_id_col: str = "sample_id",
        label_col: str = "cluster_id", label_mapping: Mapping[str, str] | None = None,
        position_map_path: str | Path | None = None,
        position_label_col: str | None = None,
        position_map_id_col: str = "sample_id",
        position_map_position_col: str | None = None) -> list[dict[str, str]]:
    """Associate supplied historical labels with a complete explicit master.

    A position-keyed label file needs a trustworthy, caller-supplied position
    mapping from the same run/split. The mapping is validated for exact
    one-to-one coverage but its provenance must be established separately.
    Existing master labels, if present, must match. No file is written.
    """
    if not label_mapping:
        raise ValueError("an explicit source-label to A/B/C label_mapping is required")
    mapping = {str(key): str(value) for key, value in label_mapping.items()}
    if len(mapping) != len(label_mapping) or any(value not in CLUSTERS for value in mapping.values()):
        raise ValueError("label_mapping keys must be distinct strings and values must be A/B/C")
    master = _read_csv(master_path, {master_id_col, "sample_id", "physical_id", "mp_id", "original_index", "split"})
    master_by_id = _unique(master, master_id_col, "master")
    _unique(master, "sample_id", "master")

    if position_map_path is None:
        if position_label_col is not None or position_map_position_col is not None:
            raise ValueError("position columns require an explicit position_map_path")
        labels = _read_csv(labels_path, {label_id_col, label_col})
        label_by_id = _unique(labels, label_id_col, "labels")
        _same_members(set(label_by_id), set(master_by_id), "labels")
        evidence = {key: (row, None) for key, row in label_by_id.items()}
    else:
        if not position_label_col or not position_map_position_col:
            raise ValueError("position-based labels require both explicit position column names")
        labels = _read_csv(labels_path, {position_label_col, label_col})
        position_map = _read_csv(position_map_path, {position_map_position_col, position_map_id_col, "split"})
        if len({row["split"] for row in master}) != 1:
            raise ValueError("position mapping must be scoped to exactly one split")
        labels_by_pos = _unique(labels, position_label_col, "labels")
        map_by_pos = _unique(position_map, position_map_position_col, "position map")
        _unique(position_map, position_map_id_col, "position map")
        _same_members(set(labels_by_pos), set(map_by_pos), "position map positions")
        evidence = {}
        for position, map_row in map_by_pos.items():
            key = _key(map_row, position_map_id_col, "position map")
            evidence[key] = (labels_by_pos[position], map_row)
        _same_members(set(evidence), set(master_by_id), "position map")

    result = []
    physical_cluster: dict[str, str] = {}
    for row in master:
        key = row[master_id_col]
        label_row, map_row = evidence[key]
        _check_identity(label_row, row, "labels")
        if map_row is not None:
            _check_identity(map_row, row, "position map")
        source_label = _key(label_row, label_col, "labels")
        if source_label not in mapping:
            raise ValueError(f"unknown source label {source_label!r}; explicit mapping required")
        cluster = mapping[source_label]
        if row.get("cluster_id", "") and row["cluster_id"] != cluster:
            raise ValueError(f"master cluster_id mismatch for {row['sample_id']!r}")
        if row.get("source_label", "") and row["source_label"] != source_label:
            raise ValueError(f"master source_label mismatch for {row['sample_id']!r}")
        physical_id = _key(row, "physical_id", "master")
        if physical_id in physical_cluster and physical_cluster[physical_id] != cluster:
            raise ValueError(f"physical_id {physical_id!r} has conflicting cluster labels")
        physical_cluster[physical_id] = cluster
        # Grouped-3 route files contain both a five-way raw cluster_id and a
        # three-way expert_id; retain both when routing by the latter.
        combined = {**row, "cluster_id": cluster, "source_label": source_label}
        if "cluster_id" in label_row:
            raw_cluster = _key(label_row, "cluster_id", "labels")
            if row.get("historical_raw_cluster_id", "") and row["historical_raw_cluster_id"] != raw_cluster:
                raise ValueError(f"master historical_raw_cluster_id mismatch for {row['sample_id']!r}")
            combined["historical_raw_cluster_id"] = raw_cluster
        result.append(combined)
    validate_manifest(result, expected_count=len(master))
    return result
