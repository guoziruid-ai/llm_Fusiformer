#!/usr/bin/env python3
"""Recompute saved cluster metrics without importing the attribution implementation.

The default configuration is deliberately inert. Create a separate fixed-config
runner to read the old artifacts and write into a *new* isolated directory.
This does not fit models or amend the historic experiment.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AuditConfig:
    source: str = ""
    output: str = ""
    enabled: bool = False
    expected_evidence_sha256: str = "09d45fcd1f9cc555f3cf860c8fc5738126c0caafa91920b84a09f43e8c58ed62"
    expected_summary_sha256: str = "8486cde3c9af0bf8180c657d8f9831ea225d7e5c9d05a75be7bb09fdeea7cdd4"


CFG = AuditConfig()
SPLITS = ("train", "validation", "test")
BASELINES = tuple(f"{method}_{scope}_seed42_original" for method in ("kmeans", "minibatch") for scope in ("train", "all"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch: {path}: {actual} != {expected}")
    return actual


def read_identity(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    by_vector = {}
    indices, identifiers = set(), set()
    for row in rows:
        vector, original = int(row["vector_row"]), int(row["original_index"])
        identifier, split = row["sample_id"], row["split"]
        if vector in by_vector or original in indices or not identifier or identifier in identifiers:
            raise ValueError("duplicate or missing identity")
        if split not in (*SPLITS, "omitted"):
            raise ValueError(f"unknown split {split}")
        by_vector[vector] = (original, identifier, split)
        indices.add(original)
        identifiers.add(identifier)
    if set(by_vector) != set(range(len(rows))):
        raise ValueError("vector rows do not cover matrix")
    return [by_vector[i] for i in range(len(rows))]


def read_historical(path, rows, k):
    # The full latent was historically extracted in original-index order.
    if any(original != vector for vector, (original, _, _) in enumerate(rows)):
        raise ValueError("full latent row to original-index identity not proven")
    lookup = {original: (i, identifier) for i, (original, identifier, _) in enumerate(rows)}
    result = np.full(len(rows), -1, dtype=np.int64)
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        for record in csv.DictReader(file):
            index = int(record["index"])
            if index not in lookup:
                raise ValueError(f"unknown historical index {index}")
            vector, expected_id = lookup[index]
            if result[vector] != -1 or record["mp_id"] != expected_id:
                raise ValueError(f"duplicate or mismatched historical index {index}")
            label = int(record["fusiformer_cluster"])
            if not 0 <= label < k:
                raise ValueError("invalid historical cluster")
            result[vector] = label
    if np.any(result < 0):
        raise ValueError("missing historical rows")
    return result


def independent_compare(reference, candidate, rows, k=5):
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.shape != (len(rows),) or candidate.shape != reference.shape:
        raise ValueError("label count mismatch")
    if any(np.any((a < 0) | (a >= k) | (a != a.astype(int))) for a in (reference, candidate)):
        raise ValueError("invalid labels")
    masks = {name: np.array([r[2] == name for r in rows], dtype=bool) for name in SPLITS}
    training = np.zeros((k, k), dtype=np.int64)
    for a, b in zip(reference[masks["train"]], candidate[masks["train"]]):
        training[int(a), int(b)] += 1
    if not masks["train"].any():
        raise ValueError("empty training split")
    left, right = linear_sum_assignment(training, maximize=True)
    mapping = np.empty(k, dtype=int)
    mapping[right] = left
    aligned = mapping[candidate.astype(int)]
    splits = {}
    for name, mask in masks.items():
        if not mask.any():
            raise ValueError(f"empty split {name}")
        a, b = reference[mask].astype(int), aligned[mask]
        matrix = np.zeros((k, k), dtype=np.int64)
        for x, y in zip(a, b):
            matrix[x, y] += 1
        splits[name] = {"count": len(a), "ari": float(adjusted_rand_score(a, b)),
                        "nmi": float(normalized_mutual_info_score(a, b)),
                        "matched_agreement": float((a == b).mean()),
                        "migration_fraction": float((a != b).mean()),
                        "migration": matrix.tolist(),
                        "reference_counts": np.bincount(a, minlength=k).tolist(),
                        "candidate_aligned_counts": np.bincount(b, minlength=k).tolist()}
    return {"candidate_to_reference": mapping.tolist(),
            "train_matching_contingency_raw": training.tolist(), "splits": splits}, aligned


def assert_saved_equal(actual, saved, name):
    for key in ("candidate_to_reference", "train_matching_contingency_raw"):
        if actual[key] != saved[key]:
            raise ValueError(f"{name}: saved {key} differs")
    for split in SPLITS:
        for key, value in actual["splits"][split].items():
            old = saved["splits"][split][key]
            if isinstance(value, float):
                if not np.isclose(value, old, rtol=1e-12, atol=1e-12):
                    raise ValueError(f"{name} {split} {key} differs: {value} vs {old}")
            elif value != old:
                raise ValueError(f"{name} {split} {key} differs")


def run(cfg: AuditConfig):
    if not cfg.enabled:
        raise PermissionError("independent audit is disabled; configure a fresh isolated runner")
    root, output = Path(cfg.source).resolve(), Path(cfg.output).resolve()
    if not cfg.source or not cfg.output or output.exists():
        raise ValueError("source and fresh output directory required")
    if root == output or root in output.parents or output in root.parents:
        raise ValueError("new output must not overlap old source")
    evidence_path, identity_path = root / "inputs/evidence.json", root / "inputs/identity.csv"
    summary_path = root / "result/summary.json"
    require_hash(evidence_path, cfg.expected_evidence_sha256)
    require_hash(summary_path, cfg.expected_summary_sha256)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    saved = json.loads(summary_path.read_text(encoding="utf-8"))
    require_hash(identity_path, evidence["identity_sha256"])
    source_hashes = {key: require_hash(item["path"], item["sha256"])
                     for key, item in evidence["identity_verification"]["inputs"].items()}
    manifest = root / "logs/output_sha256_final.stdout.log"
    manifest_hash = sha256(manifest)
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, rel = line.split(maxsplit=1)
        path = (root / rel.strip()).resolve()
        if root not in path.parents:
            raise ValueError("manifest path escapes old source")
        require_hash(path, digest)
        checked += 1
    rows = read_identity(identity_path)
    historical = read_historical(evidence["artifacts"]["full_labels"]["path"], rows, 5)
    dataset = json.loads(Path(evidence["source_dataset_path"]).read_text(encoding="utf-8"))
    if not isinstance(dataset, list) or len(dataset) != len(rows):
        raise ValueError("source JSON row count differs")
    for original, identifier, _ in rows:
        if dataset[original].get("mp_id") != identifier:
            raise ValueError(f"source JSON mp_id differs at original index {original}")
    del dataset
    matrix = np.load(evidence["identity_verification"]["inputs"]["full_latent"]["path"], mmap_mode="r", allow_pickle=False)
    if matrix.shape != (len(rows), 128):
        raise ValueError("full latent dimension differs")
    max_differences = {}
    for key, split in (("train", "train"), ("val", "validation"), ("test", "test")):
        input_rows = np.load(evidence["identity_verification"]["inputs"][key + "_indices"]["path"], allow_pickle=False)
        target = {original for original, _, s in rows if s == split}
        if len(input_rows) != len(target) or set(map(int, input_rows)) != target:
            raise ValueError(f"{split} split indices differ")
        split_vectors = np.load(evidence["identity_verification"]["inputs"][key + "_latent"]["path"], mmap_mode="r", allow_pickle=False)
        if split_vectors.shape != (len(input_rows), matrix.shape[1]):
            raise ValueError(f"{split} latent shape differs")
        maximum = 0.0
        for start in range(0, len(input_rows), 4096):
            reference = matrix[input_rows[start:start + 4096]]
            part = split_vectors[start:start + 4096]
            if not np.allclose(reference, part, atol=1e-5, rtol=1e-5):
                raise ValueError(f"{split} latent rows do not match indexed matrix")
            maximum = max(maximum, float(np.max(np.abs(reference - part))))
        max_differences[split] = maximum
    counts = {split: sum(r[2] == split for r in rows) for split in (*SPLITS, "omitted")}
    if counts != evidence["identity_verification"]["counts"]:
        raise ValueError("split counts differ from identity evidence")
    labels = {"historical_full_labels": historical}
    for name in saved["models"]:
        labels[name] = np.load(root / "result" / (name + "_labels.npy"), allow_pickle=False)
    results = {}
    for name, prior in saved["comparisons"].items():
        result, _ = independent_compare(labels[prior["reference"]], labels[prior["candidate"]], rows)
        assert_saved_equal(result, prior, name)
        results[name] = result
    raw = labels["minibatch_all_seed42_original"]
    raw_match_count = int((raw == historical).sum())
    if raw_match_count != len(rows):
        raise ValueError("historical raw labels do not match")

    # This reads a trusted, hash-verified saved sklearn joblib; it does not fit.
    import joblib
    from sklearn.preprocessing import normalize
    model = joblib.load(evidence["artifacts"]["train_model"]["path"])
    x = normalize(np.asarray(matrix, dtype=np.float32), axis=1)
    historical_train = model.predict(x.astype(model.cluster_centers_.dtype, copy=False))
    historical_comparison, _ = independent_compare(historical, historical_train, rows)
    labels["historical_train_model"] = historical_train

    output.mkdir(parents=True, exist_ok=False)
    aligned = {}
    for name in (*BASELINES, "historical_train_model"):
        _, aligned[name] = independent_compare(historical, labels[name], rows)
    with (output / "sample_migrations.csv").open("x", encoding="utf-8", newline="") as file:
        columns = ["sample_id", "original_index", "vector_row", "split", "historical_full_raw"]
        for name in aligned:
            columns.extend([name + "_raw", name + "_aligned", name + "_changed"])
        writer = csv.writer(file)
        writer.writerow(columns)
        for i, (original, identifier, split) in enumerate(rows):
            record = [identifier, original, i, split, int(historical[i])]
            for name in aligned:
                record.extend([int(labels[name][i]), int(aligned[name][i]), int(aligned[name][i] != historical[i])])
            writer.writerow(record)
    report = {"independent_method": "explicit original_index and mp_id linkage; train-only Hungarian; no prior comparison code imported",
              "source": str(root), "source_hashes": source_hashes, "manifest_sha256": manifest_hash,
              "manifest_files_verified": checked, "source_evidence_sha256": cfg.expected_evidence_sha256,
              "source_summary_sha256": cfg.expected_summary_sha256, "counts": counts,
              "split_latent_max_abs_difference": max_differences,
              "omitted_policy": "excluded from mapping and all split metrics; retained in sample export",
              "raw_historical_match": {"matching_rows": raw_match_count, "rows": len(rows)},
              "saved_comparisons_verified": len(results), "comparisons": results,
              "historical_train_model_vs_historical_labels": historical_comparison}
    (output / "independent_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(CFG)))
