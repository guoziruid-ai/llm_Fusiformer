#!/usr/bin/env python3
"""Describe frozen historical labels, predictions and Grouped-3 routes.

No model execution, clustering, split generation, or inferential tests. Paths and
SHA-256 digests must be explicitly supplied in a caller-owned configuration.
The two raw-cluster spaces are deliberately kept separate.
"""
from __future__ import annotations

import csv
import argparse
import hashlib
import json
import math
import platform
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scripts.experiments.cluster_specialists.historical import verify_grouped3_mapping


@dataclass(frozen=True)
class Config:
    # Required files: identity, assignments, predictions, mapping, router_model,
    # identity_evidence; route train/validation require both CSV and indices.
    paths: dict[str, str] = field(default_factory=dict)
    sha256: dict[str, str] = field(default_factory=dict)
    prediction_split: str = ""
    output_dir: str = ""
    allow_write: bool = False
    plot: bool = True
    require_prediction_routes: bool = False


CFG = Config()  # intentionally inert; edit a separate runner, never historic files
REQUIRED = {"identity", "identity_evidence", "assignments", "predictions", "mapping", "router_model"}
ID_FIELDS = ("sample_id", "original_index", "split", "vector_row")
LIMITATIONS = ("Descriptive only, not causal or confirmatory; multiple cluster comparisons "
               "are uncorrected. Grouped-3 mapping was selected using validation MAE; "
               "validation comparisons have selection bias. History provenance is conditional "
               "on the separately documented evidence. Test results must not select analysis choices.")


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _csv(path, required):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"missing or duplicate CSV headers: {path}")
        if missing := set(required) - set(reader.fieldnames):
            raise ValueError(f"missing fields in {path}: {sorted(missing)}")
        rows = list(reader)
    if not rows or any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError(f"empty or malformed CSV: {path}")
    return rows


def _key(row, name):
    value = row.get(name, "")
    if not value or value != value.strip():
        raise ValueError(f"missing or invalid {name}: {value!r}")
    return value


def _uint(row, name):
    value = _key(row, name)
    if not value.isascii() or not value.isdecimal() or str(int(value)) != value:
        raise ValueError(f"invalid nonnegative integer {name}: {value!r}")
    return int(value)


def _unique(rows, key, source):
    out = {}
    for r in rows:
        value = _key(r, key)
        if value in out:
            raise ValueError(f"duplicate {key} in {source}: {value}")
        out[value] = r
    return out


def _members(actual, expected, source):
    missing, extra = expected - actual, actual - expected
    if missing or extra:
        raise ValueError(f"{source} identity mismatch: missing={sorted(missing)[:5]} "
                         f"unknown={sorted(extra)[:5]} counts={len(missing)}/{len(extra)}")


def _num(row, key):
    try:
        n = float(_key(row, key))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"missing or invalid numeric field {key}") from exc
    if not math.isfinite(n):
        raise ValueError(f"non-finite {key}")
    return n


def _summary(rows):
    if not rows:
        return {"count": 0}
    a = sorted(r["abs_error"] for r in rows)
    return {"count": len(rows), "mae": statistics.fmean(a),
            "rmse": math.sqrt(statistics.fmean(r["residual"] ** 2 for r in rows)),
            "absolute_error": {"min": a[0], "median": statistics.median(a), "max": a[-1],
                               "p90_nearest_rank": a[math.ceil(.9 * len(a)) - 1]}}


def load(cfg):
    if cfg.prediction_split not in ("train", "validation", "test"):
        raise ValueError("prediction_split must be train, validation, or test")
    if not REQUIRED <= cfg.paths.keys() or set(cfg.paths) != set(cfg.sha256):
        raise ValueError("required paths and a SHA-256 for every input must be provided")
    if not any(f"{split}_routes" in cfg.paths for split in ("train", "validation")):
        raise ValueError("at least one verified train or validation routes/indices pair is required")
    for split in ("train", "validation", "test"):
        if (f"{split}_routes" in cfg.paths) != (f"{split}_indices" in cfg.paths):
            raise ValueError(f"{split} routes and indices must be supplied together")
    verified = {}
    for name, value in cfg.paths.items():
        p = Path(value)
        expected = cfg.sha256[name]
        if not p.is_file():
            raise FileNotFoundError(f"missing {name}: {p}")
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError(f"invalid expected SHA-256 for {name}")
        actual = sha256_file(p)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {name}: {p}")
        verified[name] = {"path": str(p.resolve()), "sha256": actual}
    identity = _csv(cfg.paths["identity"], ID_FIELDS)
    by_id = _unique(identity, "sample_id", "identity")
    by_index = _unique(identity, "original_index", "identity")
    _unique(identity, "vector_row", "identity")
    for r in identity:
        _uint(r, "original_index"); _uint(r, "vector_row")
        if r["split"] not in ("train", "validation", "test", "omitted"):
            raise ValueError("invalid identity split")
    # The verified identity sidecar defines sample_id as mp_id. Do not infer
    # this relation from file order: assignments cross-check it explicitly.
    assignments = _csv(cfg.paths["assignments"], ("index", "mp_id", "e_form", "fusiformer_cluster"))
    aindex = _unique(assignments, "index", "assignments")
    _unique(assignments, "mp_id", "assignments")
    _members(set(aindex), set(by_index), "assignments")
    for index, r in aindex.items():
        _uint(r, "index")
        if r["mp_id"] != by_index[index]["sample_id"]:
            raise ValueError(f"assignments mp_id mismatch at {index}")
        if _uint(r, "fusiformer_cluster") not in range(5):
            raise ValueError("unknown historical raw cluster")
        _num(r, "e_form")
    # Identity evidence itself must bind the exact sidecar and source files.
    evidence = json.loads(Path(cfg.paths["identity_evidence"]).read_text(encoding="utf-8"))
    if evidence.get("identity_sha256") != verified["identity"]["sha256"]:
        raise ValueError("identity evidence does not bind identity SHA-256")
    artifacts = evidence.get("artifacts", {})
    for name, evidence_name in (("assignments", "full_labels"), ("router_model", "train_model")):
        if artifacts.get(evidence_name, {}).get("sha256") != verified[name]["sha256"]:
            raise ValueError(f"identity evidence does not bind {name}")
    if evidence.get("same_identity_mapping_verified") is not True:
        raise ValueError("identity evidence has no verified identity mapping")
    prediction = _csv(cfg.paths["predictions"], ("original_index", "mp_id", "split", "cluster", "y_true", "y_pred"))
    pred_index = _unique(prediction, "original_index", "predictions")
    _unique(prediction, "mp_id", "predictions")
    expected = {r["original_index"] for r in identity if r["split"] == cfg.prediction_split}
    if not expected:
        raise ValueError("requested prediction split is empty")
    _members(set(pred_index), expected, "predictions")
    result = []
    for index, p in pred_index.items():
        master, a = by_index[index], aindex[index]
        if p["mp_id"] != master["sample_id"] or p["split"] != master["split"]:
            raise ValueError(f"prediction mp_id or split mismatch at {index}")
        if p["cluster"] != a["fusiformer_cluster"]:
            raise ValueError(f"prediction historical cluster mismatch at {index}")
        y, pred = _num(p, "y_true"), _num(p, "y_pred")
        if not math.isclose(y, _num(a, "e_form"), rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError(f"prediction target mismatch at {index}")
        result.append({**master, "mp_id": master["sample_id"], "historical_cluster": a["fusiformer_cluster"],
                       "target": y, "prediction": pred, "residual": pred-y, "abs_error": abs(pred-y)})
    mapping = json.loads(Path(cfg.paths["mapping"]).read_text(encoding="utf-8"))
    cross, routes_by_index = {}, {}
    for split, prefix in (("train", "train"), ("validation", "val"), ("test", "test")):
        route_key, index_key = f"{split}_routes", f"{split}_indices"
        if route_key not in cfg.paths:
            continue
        verify_grouped3_mapping(cfg.paths[route_key], cfg.paths["mapping"])
        routes = _csv(cfg.paths[route_key], (f"{prefix}_position", "cluster_id", "expert_id", "expert_name"))
        positions = _unique(routes, f"{prefix}_position", route_key)
        indices = np.load(cfg.paths[index_key], allow_pickle=False)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"{index_key} must contain one-dimensional integer indices")
        _members(set(positions), {str(i) for i in range(len(indices))}, route_key + " positions")
        identities = [str(int(i)) for i in indices]
        if len(set(identities)) != len(identities):
            raise ValueError(f"duplicate original index in {index_key}")
        _members(set(identities), {r["original_index"] for r in identity if r["split"] == split}, index_key)
        # Bind actual position-to-original-index mapping to the identity preflight.
        proof_inputs = evidence.get("identity_verification", {}).get("inputs", {})
        proof_split = "val" if split == "validation" else split
        for supplied, evidence_key in ((route_key, proof_split + "_routes"), (index_key, proof_split + "_indices")):
            if proof_inputs.get(evidence_key, {}).get("sha256") != verified[supplied]["sha256"]:
                raise ValueError(f"identity evidence does not bind {supplied}")
        for position, route in positions.items():
            idx = identities[int(position)]
            historical = aindex[idx]["fusiformer_cluster"]
            expert = route["expert_id"]
            raw = route["cluster_id"]
            routes_by_index[idx] = {"router_cluster": raw, "expert_id": expert,
                                    "expert_name": route["expert_name"]}
            key = (split, historical, raw, expert, route["expert_name"])
            cross[key] = cross.get(key, 0) + 1
    for r in result:
        r.update(routes_by_index.get(r["original_index"], {"router_cluster": "", "expert_id": "", "expert_name": ""}))
    matched = sum(bool(r["expert_id"]) for r in result)
    if matched not in (0, len(result)):
        raise ValueError("route identity coverage is partial for selected prediction split")
    if cfg.require_prediction_routes and matched != len(result):
        raise ValueError("missing reliable routes for the prediction split")
    return result, cross, mapping, verified


def run(cfg):
    if not cfg.allow_write or not cfg.output_dir:
        raise PermissionError("explicit output directory and allow_write required")
    output = Path(cfg.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    if any(output == Path(p).resolve() or output in Path(p).resolve().parents for p in cfg.paths.values()):
        raise ValueError("output cannot contain an input")
    rows, cross, mapping, verified = load(cfg)
    if cfg.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    clusters = {str(i): _summary([r for r in rows if r["historical_cluster"] == str(i)]) for i in range(5)}
    experts = {str(i): _summary([r for r in rows if r["expert_id"] == str(i)]) for i in range(3)}
    differences = {}
    for i in range(5):
        for j in range(i+1, 5):
            a, b = clusters[str(i)], clusters[str(j)]
            if a["count"] and b["count"]:
                differences[f"{i}-{j}"] = a["mae"] - b["mae"]
    summary = {"type": "descriptive_only", "split": cfg.prediction_split, "overall": _summary(rows),
               "historical_full_minibatch_clusters": clusters, "baseline_error_by_router_expert": experts,
               "baseline_prediction_route_matched_count": sum(bool(r["expert_id"]) for r in rows),
               "route_stratification_status": "available" if any(r["expert_id"] for r in rows) else
                    "unavailable: no same-split identity-verified route; expert error not computed",
               "expert_model_prediction_status": "not provided; no expert mechanism performance computed",
               "historical_cluster_pairwise_mae_differences": differences,
               "grouped3_cluster_to_expert_id": mapping["cluster_to_expert_id"],
               "limitations": LIMITATIONS}
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    for filename, data, fields in (
        ("samples.csv", rows, list(rows[0])),
        ("historical_vs_router_counts.csv", [{"split": k[0], "historical_cluster": k[1], "router_cluster": k[2],
          "expert_id": k[3], "expert_name": k[4], "count": v} for k, v in sorted(cross.items())],
         ["split", "historical_cluster", "router_cluster", "expert_id", "expert_name", "count"])):
        with (output / filename).open("x", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(data)
    if cfg.plot:
        fig, ax = plt.subplots()
        available = [i for i in range(5) if clusters[str(i)]["count"]]
        ax.boxplot([[r["abs_error"] for r in rows if r["historical_cluster"] == str(i)] for i in available],
                   labels=[str(i) for i in available])
        ax.set(xlabel="Historical full-data MiniBatch raw cluster", ylabel="Absolute prediction error")
        fig.tight_layout(); fig.savefig(output / "absolute_errors.png", dpi=160); plt.close(fig)
    manifest = {"inputs": verified, "prediction_split": cfg.prediction_split,
                "config": {"plot": cfg.plot, "require_prediction_routes": cfg.require_prediction_routes},
                "script_sha256": sha256_file(__file__), "python": platform.python_version(),
                "numpy": np.__version__, "limitations": LIMITATIONS,
                "outputs": {p.name: sha256_file(p) for p in sorted(output.iterdir()) if p.is_file()}}
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def preflight(cfg):
    """Check every input and association, with no output directory or writes."""
    rows, cross, _, verified = load(cfg)
    return {"writes": False, "split": cfg.prediction_split, "prediction_count": len(rows),
            "route_matched_count": sum(bool(r["expert_id"]) for r in rows),
            "cross_count": sum(cross.values()), "inputs": verified}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-config", action="store_true", help="verify inputs and associations without writing")
    args = parser.parse_args()
    print(json.dumps(preflight(CFG) if args.check_config else run(CFG), ensure_ascii=False))
