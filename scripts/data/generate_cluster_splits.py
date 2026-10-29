#!/usr/bin/env python3
"""Generate router assignments plus A/B/C x train/validation/test manifests.

Edit the file-internal configuration below. This command writes index manifests
only; it never copies graphs/text caches and never trains a prediction model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.experiments.cluster_specialists.config import SplitRatios
from scripts.experiments.cluster_specialists.manifests import (
    SPLITS, assign_physical_splits, build_manifest, infer_physical_id, write_manifests)
from scripts.experiments.cluster_specialists.routing import (
    CentroidRouter, FixedBoundaryRouter, load_router)

ROOT = Path(__file__).resolve().parents[2]

# =============================================================================
# User configuration
# =============================================================================
DATA_PATH = ROOT / "data/dataset_final_robocrys.json"
ROUTER_FEATURES_PATH = ROOT / "data/cluster_router_features.npy"  # [N,D]
ROUTER_FEATURES_INDEX_PATH = ROOT / "data/cluster_router_features_source_index.csv"  # feature_row,original_index,physical_id
OUTPUT_DIR = ROOT / "runs/cluster_specialists/manifests_seed42"
ROUTER_ARTIFACT_PATH = ROOT / "runs/cluster_specialists/router_seed42.json"
ROUTER_MODE = "load"  # load | fit_centroid | fixed_boundary
PREDECLARED_SPLIT_PATH: Path | None = None  # optional CSV: original_index,physical_id,split
FEATURE_NAMES = ("v2_graph_latent_0",)
FIXED_BOUNDARIES = (-0.5, 0.5)
FIXED_FEATURE_NAME = "baseline_prediction"
TARGET_FIELD = "e_form"
SPLIT_RATIOS = SplitRatios(0.8, 0.1, 0.1)
RANDOM_SEED = 42
PHYSICAL_ID_FIELDS = ("mp_id", "material_id", "jid", "id")  # centroid: choose ONE verified physical ID field


def _align_features(records: list[dict], features: np.ndarray, index_path: Path,
                    physical_id_fields: tuple[str, ...]) -> np.ndarray:
    """Align array rows using a complete explicit identity-checked sidecar."""
    if len(records) != len(features):
        raise ValueError("dataset and router features have different lengths")
    with index_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"feature_row", "original_index", "physical_id"} <= rows[0].keys():
        raise ValueError("feature index CSV must contain feature_row, original_index, physical_id")
    if len(rows) != len(records):
        raise ValueError("feature index CSV count does not match the dataset")
    feature_for_index: dict[int, int] = {}
    seen_feature_rows: set[int] = set()
    for row in rows:
        feature_row, index = int(row["feature_row"]), int(row["original_index"])
        if index < 0 or index >= len(records) or index in feature_for_index:
            raise ValueError(f"duplicate or unknown original_index in feature index: {index}")
        if feature_row < 0 or feature_row >= len(features) or feature_row in seen_feature_rows:
            raise ValueError(f"duplicate or unknown feature_row in feature index: {feature_row}")
        physical_id = infer_physical_id(
            records[index], index, physical_id_fields, require_explicit=True)
        if row["physical_id"] != physical_id:
            raise ValueError(f"physical ID mismatch in feature index at original_index {index}")
        feature_for_index[index] = feature_row
        seen_feature_rows.add(feature_row)
    return np.asarray(features[[feature_for_index[index] for index in range(len(records))]])


def _read_predeclared_splits(path: Path, records: list[dict],
                             physical_id_fields: tuple[str, ...]) -> list[str]:
    """Require a complete, identity-checked upstream split, not just fit indices."""
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"original_index", "physical_id", "split"} <= rows[0].keys():
        raise ValueError("split CSV must contain original_index, physical_id, and split")
    assigned: dict[int, str] = {}
    physical_splits: dict[str, str] = {}
    for row in rows:
        index = int(row["original_index"])
        if index < 0 or index >= len(records) or index in assigned:
            raise ValueError(f"duplicate or unknown original_index in split CSV: {index}")
        physical_id = infer_physical_id(
            records[index], index, physical_id_fields, require_explicit=True)
        if row["physical_id"] != physical_id:
            raise ValueError(f"physical ID mismatch in split CSV at original_index {index}")
        split = row["split"]
        if split not in SPLITS:
            raise ValueError(f"invalid split in CSV: {split!r}")
        if physical_id in physical_splits and physical_splits[physical_id] != split:
            raise ValueError(f"physical sample leakage across splits: {physical_id}")
        physical_splits[physical_id] = split
        assigned[index] = split
    if len(assigned) != len(records):
        raise ValueError("split CSV does not cover every dataset record")
    return [assigned[index] for index in range(len(records))]


def prepare_centroid(records: list[dict], features: np.ndarray, *,
                     ratios: SplitRatios, seed: int, physical_id_fields: tuple[str, ...],
                     feature_names: tuple[str, ...], split_path: Path | None = None,
                     router: CentroidRouter | None = None):
    """Split first, then fit or verify a router tied to that exact training set."""
    if len(records) != len(features):
        raise ValueError("dataset and router features have different lengths")
    if len(physical_id_fields) != 1:
        raise ValueError("centroid mode requires exactly one verified physical ID field")
    splits = (_read_predeclared_splits(split_path, records, physical_id_fields)
              if split_path is not None else assign_physical_splits(
                  records, ratios=ratios, seed=seed, physical_id_fields=physical_id_fields))
    train_indices = [index for index, split in enumerate(splits) if split == "train"]
    matrix = np.asarray(features if features.ndim == 2 else features.reshape(-1, 1))
    split_records = [
        [index, infer_physical_id(record, index, physical_id_fields, require_explicit=True), splits[index]]
        for index, record in enumerate(records)]
    training_matrix = np.ascontiguousarray(matrix[train_indices], dtype="<f8")
    provenance = {
        "version": 1, "split_method": "predeclared" if split_path is not None else "physical_id_hash",
        "split_seed": seed, "split_ratios": ratios.as_dict(),
        "physical_id_fields": list(physical_id_fields),
        "split_sha256": hashlib.sha256(json.dumps(
            split_records, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "train_original_indices": train_indices,
        "train_features_shape": list(training_matrix.shape),
        "train_features_sha256": hashlib.sha256(training_matrix.tobytes()).hexdigest(),
    }
    if router is None:
        router = CentroidRouter(feature_names=feature_names, random_seed=seed)
        router.fit(training_matrix)
        router.fit_provenance_ = provenance
    else:
        if router.feature_names != tuple(feature_names):
            raise ValueError("loaded centroid feature names do not match configuration")
        if router.fit_provenance_ != provenance:
            raise ValueError("loaded centroid fit provenance does not match the current split and training features")
    return router, splits


def validate_configuration() -> tuple[list[dict], np.ndarray]:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"dataset not found: {DATA_PATH}")
    if not ROUTER_FEATURES_PATH.is_file():
        raise FileNotFoundError(
            f"router features not found: {ROUTER_FEATURES_PATH}")
    if not ROUTER_FEATURES_INDEX_PATH.is_file():
        raise FileNotFoundError(
            f"router feature index not found: {ROUTER_FEATURES_INDEX_PATH}")
    records = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    features = np.load(ROUTER_FEATURES_PATH, mmap_mode="r")
    if not isinstance(records, list) or features.ndim not in {1, 2}:
        raise ValueError("dataset must be a list and features must be [N] or [N,D]")
    features = _align_features(records, features, ROUTER_FEATURES_INDEX_PATH, PHYSICAL_ID_FIELDS)
    if ROUTER_MODE == "load" and not ROUTER_ARTIFACT_PATH.is_file():
        raise FileNotFoundError(
            f"router artifact not found: {ROUTER_ARTIFACT_PATH}")
    if PREDECLARED_SPLIT_PATH is not None and not PREDECLARED_SPLIT_PATH.is_file():
        raise FileNotFoundError(f"predeclared split CSV not found: {PREDECLARED_SPLIT_PATH}")
    if ROUTER_MODE not in {"load", "fit_centroid", "fixed_boundary"}:
        raise ValueError("ROUTER_MODE must be load, fit_centroid or fixed_boundary")
    if ROUTER_MODE == "fit_centroid" and len(PHYSICAL_ID_FIELDS) != 1:
        raise ValueError("centroid mode requires exactly one verified physical ID field")
    return records, features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    records, features = validate_configuration()
    if args.check_config:
        print(f"[PASS] {len(records)} identity-aligned samples; no files written")
        return
    splits = None
    if ROUTER_MODE == "load":
        router = load_router(ROUTER_ARTIFACT_PATH)
        if isinstance(router, CentroidRouter):
            router, splits = prepare_centroid(
                records, features, ratios=SPLIT_RATIOS, seed=RANDOM_SEED,
                physical_id_fields=PHYSICAL_ID_FIELDS, feature_names=FEATURE_NAMES,
                split_path=PREDECLARED_SPLIT_PATH, router=router)
        elif PREDECLARED_SPLIT_PATH is not None:
            splits = _read_predeclared_splits(PREDECLARED_SPLIT_PATH, records, PHYSICAL_ID_FIELDS)
    elif ROUTER_MODE == "fixed_boundary":
        router = FixedBoundaryRouter(
            *FIXED_BOUNDARIES, feature_name=FIXED_FEATURE_NAME)
        if PREDECLARED_SPLIT_PATH is not None:
            splits = _read_predeclared_splits(PREDECLARED_SPLIT_PATH, records, PHYSICAL_ID_FIELDS)
    else:
        router, splits = prepare_centroid(
            records, features, ratios=SPLIT_RATIOS, seed=RANDOM_SEED,
            physical_id_fields=PHYSICAL_ID_FIELDS, feature_names=FEATURE_NAMES,
            split_path=PREDECLARED_SPLIT_PATH)
    clusters = router.predict(features)
    rows = build_manifest(
        records, clusters, ratios=SPLIT_RATIOS, seed=RANDOM_SEED,
        physical_id_fields=PHYSICAL_ID_FIELDS, target_field=TARGET_FIELD,
        splits=splits)
    if ROUTER_MODE != "load":
        router.save(ROUTER_ARTIFACT_PATH)
    paths = write_manifests(
        rows, OUTPUT_DIR, seed=RANDOM_SEED, ratios=SPLIT_RATIOS,
        source=str(DATA_PATH.resolve()))
    print(json.dumps(paths, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
