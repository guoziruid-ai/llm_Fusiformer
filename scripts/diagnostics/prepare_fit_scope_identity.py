#!/usr/bin/env python3
"""Verify historical graph-cache identity and create diagnostic sidecar only.

No training, inference, or new clustering.  Dataset/checkpoint metadata bind by
size/mtime, not a historic digest: exact historic file content remains unproved.
All paths are explicit; the default config is deliberately inert.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

try:
    from .compare_latent_fit_scope import sha256
except ImportError:  # direct invocation from an isolated diagnostics directory
    from compare_latent_fit_scope import sha256


@dataclass(frozen=True)
class Config:
    data: str = ""
    checkpoint: str = ""
    full_latent: str = ""
    full_meta: str = ""
    train_latent: str = ""
    train_indices: str = ""
    train_meta: str = ""
    val_latent: str = ""
    val_indices: str = ""
    val_meta: str = ""
    test_latent: str = ""
    test_indices: str = ""
    test_meta: str = ""
    full_labels: str = ""
    test_predictions: str = ""
    train_routes: str = ""
    val_routes: str = ""
    train_model: str = ""
    text_meta: str = ""  # historic text cache metadata, including text_sha256
    output_dir: str = ""  # must not exist
    allow_write: bool = False
    full_model_missing_reason: str = ""
    split_seed: int = 7
    atol: float = 1e-6
    rtol: float = 1e-5


CFG = Config()
FILE_FIELDS = ("data", "checkpoint", "full_latent", "full_meta", "train_latent", "train_indices",
               "train_meta", "val_latent", "val_indices", "val_meta", "test_latent", "test_indices",
               "test_meta", "full_labels", "test_predictions", "train_routes", "val_routes",
               "train_model", "text_meta")


def _csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"missing or duplicate CSV header: {path}")
        return list(reader)


def _signature(meta, key, path):
    item = meta.get(key, {})
    stat = path.stat()
    if (item.get("size"), item.get("mtime_ns")) != (stat.st_size, stat.st_mtime_ns):
        raise ValueError(f"{key} metadata size/mtime mismatch: {path}")
    if not item.get("path"):
        raise ValueError(f"{key} historical path missing")
    return item


def prepare(cfg: Config):
    if not cfg.allow_write:
        raise PermissionError("identity preparation disabled; configure all paths and explicitly enable")
    if not cfg.output_dir or not cfg.full_model_missing_reason:
        raise ValueError("output_dir and the scoped missing full-model reason are required")
    paths = {}
    for key in FILE_FIELDS:
        value = getattr(cfg, key)
        if not value or not Path(value).is_file():
            raise FileNotFoundError(f"required {key}: {value!r}")
        paths[key] = Path(value).resolve()
    output = Path(cfg.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse directory: {output}")
    if any(output == p or output in p.parents for p in paths.values()):
        raise ValueError("output directory cannot contain an input")
    if cfg.split_seed != 7 or cfg.atol < 0 or cfg.rtol < 0:
        raise ValueError("historical split seed must be 7; tolerances must be nonnegative")

    with paths["data"].open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, list) or not data:
        raise ValueError("source data must be a nonempty JSON list")
    sample_ids = []
    targets = []
    text_digest = hashlib.sha256()
    for i, item in enumerate(data):
        sample = item.get("mp_id")
        if not isinstance(sample, str) or not sample.strip():
            raise ValueError(f"missing mp_id at original index {i}")
        sample_ids.append(sample)
        try:
            targets.append(float(item["e_form"]))
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"invalid e_form at original index {i}") from exc
        raw_text = item.get("description", "")
        if raw_text is None:
            raw_text = ""
        if not isinstance(raw_text, str):
            raw_text = str(raw_text)
        text = raw_text.encode("utf-8", errors="replace")
        text_digest.update(i.to_bytes(8, "little"))
        text_digest.update(len(text).to_bytes(8, "little"))
        text_digest.update(text)
    if len(sample_ids) != len(set(sample_ids)) or not np.isfinite(targets).all():
        raise ValueError("mp_id is duplicate or e_form contains a non-finite value")
    text_meta = json.loads(paths["text_meta"].read_text(encoding="utf-8"))
    if text_meta.get("text_sha256") != text_digest.hexdigest():
        raise ValueError("historical text metadata fingerprint mismatch")
    full = np.load(paths["full_latent"], mmap_mode="r", allow_pickle=False)
    meta = json.loads(paths["full_meta"].read_text(encoding="utf-8"))
    n = len(data)
    if full.ndim != 2 or full.shape != (n, meta.get("embed_dim")) or meta.get("num_samples") != n:
        raise ValueError("full graph latent shape/count disagrees with metadata and JSON")
    if meta.get("latent_mode") != "graph_only" or not np.isfinite(full).all():
        raise ValueError("full latent mode or numeric contents invalid")
    historical_data = _signature(meta, "data", paths["data"])
    historical_checkpoint = _signature(meta, "checkpoint", paths["checkpoint"])
    shuffled = list(range(n))
    random.Random(cfg.split_seed).shuffle(shuffled)
    train_n, val_n, test_n = int(.8 * n), int(.1 * n), int(.1 * n)
    expected = {"train": shuffled[:train_n], "val": shuffled[train_n:train_n + val_n],
                "test": shuffled[-test_n:]}
    split_indices = {}
    max_differences = {}
    split_labels = {}
    seen = set()
    model = joblib.load(paths["train_model"])  # trusted local historical artifact only
    if getattr(model, "n_clusters", None) != 5:
        raise ValueError("historic train-only model is not K=5")
    for split in ("train", "val", "test"):
        indices = np.load(paths[split + "_indices"], allow_pickle=False)
        latent = np.load(paths[split + "_latent"], mmap_mode="r", allow_pickle=False)
        sm = json.loads(paths[split + "_meta"].read_text(encoding="utf-8"))
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"{split} indices are not a one-dimensional integer array")
        if indices.tolist() != expected[split]:
            raise ValueError(f"{split} indices do not match historical seed-7 loader split")
        if seen.intersection(indices.tolist()):
            raise ValueError(f"{split} index overlaps another split")
        seen.update(indices.tolist())
        if (sm.get("data") != historical_data or sm.get("base_weights") != historical_checkpoint or
                sm.get("split_name") != split or sm.get("split_size") != len(indices) or
                sm.get("router_mode") != "graph_only_readout"):
            raise ValueError(f"{split} metadata does not match historical source signatures")
        if latent.shape != (len(indices), full.shape[1]) or not np.isfinite(latent).all():
            raise ValueError(f"{split} latent shape/values invalid")
        maximum = 0.0
        for start in range(0, len(indices), 4096):
            part = np.asarray(latent[start:start + 4096])
            reference = np.asarray(full[indices[start:start + 4096]])
            maximum = max(maximum, float(np.max(np.abs(part - reference))))
            if not np.allclose(part, reference, rtol=cfg.rtol, atol=cfg.atol):
                raise ValueError(f"{split} latent differs from indexed full array")
        max_differences[split] = maximum
        split_indices[split] = indices
        if split != "test":
            routes = _csv(paths[split + "_routes"])
            position = split + "_position"
            if len(routes) != len(indices) or [int(r[position]) for r in routes] != list(range(len(indices))):
                raise ValueError(f"{split} routes have missing/duplicate/out-of-order positions")
            normalized = np.asarray(latent, dtype=np.float32).copy()
            norms = np.linalg.norm(normalized, axis=1)
            if np.any(norms == 0):
                raise ValueError(f"{split} has zero vectors")
            normalized /= norms[:, None]
            labels = np.asarray([int(r["cluster_id"]) for r in routes])
            predicted = model.predict(normalized)
            if not np.array_equal(labels, predicted):
                raise ValueError(f"{split} stored-model predictions do not match historic routes")
            split_labels[split] = int(len(labels))

    assignments = _csv(paths["full_labels"])
    if len(assignments) != n:
        raise ValueError("full assignment count mismatch")
    for i, row in enumerate(assignments):
        if (int(row["index"]) != i or row["mp_id"] != sample_ids[i] or
                not np.isclose(float(row["e_form"]), targets[i], rtol=0, atol=1e-10)):
            raise ValueError(f"full assignment index/mp_id/e_form mismatch at {i}")
    predictions = _csv(paths["test_predictions"])
    if len(predictions) != test_n:
        raise ValueError("historical test prediction row count mismatch")
    for position, row in enumerate(predictions):
        i = int(split_indices["test"][position])
        if (int(row["original_index"]) != i or row["mp_id"] != sample_ids[i] or row["split"] != "test" or
                int(row["cluster"]) != int(assignments[i]["fusiformer_cluster"]) or
                not np.isclose(float(row["y_true"]), targets[i], rtol=1e-5, atol=1e-5)):
            raise ValueError(f"historical test prediction identity/target mismatch at {position}")
    hashes = {name: sha256(path) for name, path in paths.items()}
    split_name = {int(index): "validation" if split == "val" else split
                  for split, indices in split_indices.items() for index in indices}
    counts = {split: sum(v == split for v in split_name.values()) for split in ("train", "validation", "test")}
    counts["omitted"] = n - len(split_name)
    output.mkdir(parents=True, exist_ok=False)
    identity = output / "identity.csv"
    with identity.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "original_index", "split", "vector_row"])
        writer.writeheader()
        for i, sample in enumerate(sample_ids):
            writer.writerow({"sample_id": sample, "original_index": i, "split": split_name.get(i, "omitted"), "vector_row": i})
    evidence = {"latent_sha256": hashes["full_latent"], "identity_sha256": sha256(identity),
                "checkpoint_path": str(paths["checkpoint"]), "checkpoint_sha256": hashes["checkpoint"],
                "source_dataset_path": str(paths["data"]), "source_dataset_sha256": hashes["data"],
                "latent_layer": "graph_only_readout (128-D)",
                "preprocessing": "same full graph_out array; diagnostic L2 normalization",
                "identity_mapping_method": "all assignment ID/target rows, seed-7 split indices, split/full vector allclose, model/route match and historical test identities",
                "same_model_and_preprocessing_verified": True, "same_identity_mapping_verified": True,
                "historical_signature_limit": "historical metadata has size/mtime but no historic content digest; exact historical JSON/checkpoint content remains unproved",
                "artifacts": {"train_model": {"path": str(paths["train_model"]), "sha256": hashes["train_model"]},
                              "train_labels": {"path": str(paths["train_routes"]), "sha256": hashes["train_routes"]},
                              "full_model": {"status": "missing", "reason": cfg.full_model_missing_reason},
                              "full_labels": {"path": str(paths["full_labels"]), "sha256": hashes["full_labels"]}},
                "identity_verification": {"counts": counts, "max_abs_difference": max_differences,
                                          "allclose_atol": cfg.atol, "allclose_rtol": cfg.rtol,
                                          "train_val_route_counts": split_labels,
                                          "historical_paths": {"data": historical_data["path"], "checkpoint": historical_checkpoint["path"]},
                                          "text_sha256": text_digest.hexdigest(),
                                          "inputs": {k: {"path": str(paths[k]), "sha256": v} for k, v in hashes.items()}}}
    (output / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "identity_sha256": sha256(identity), "counts": counts,
                      "max_abs_difference": max_differences, "inputs": evidence["identity_verification"]["inputs"]}, ensure_ascii=False))
    return evidence


if __name__ == "__main__":
    prepare(CFG)
