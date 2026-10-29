#!/usr/bin/env python3
"""Change only the stopping controls of the historical-parameter MiniBatch fit.

Both fits use the same input order, initialization method, seed, number of
initializations, batch size, reassignment and precision.  This retrospective
diagnostic writes to a fresh directory and does not predict test labels.
Call run() from a separate file-internal configuration after provenance review.
"""
from __future__ import annotations

import contextlib
import csv
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

try:
    from scripts.diagnostics.attribute_cluster_drift import compare
    from scripts.diagnostics.audit_minibatch_stopping import _sse
    from scripts.diagnostics.compare_latent_fit_scope import Config, preflight, sha256
    from scripts.diagnostics.compare_saved_cluster_labels import load_full_labels
except ModuleNotFoundError:
    from attribute_cluster_drift import compare
    from audit_minibatch_stopping import _sse
    from compare_latent_fit_scope import Config, preflight, sha256
    from compare_saved_cluster_labels import load_full_labels


@dataclass(frozen=True)
class NativeStopConfig:
    provenance: Config
    baseline_model: str
    baseline_model_sha256: str
    baseline_summary: str
    baseline_summary_sha256: str
    output_dir: str
    allow_diagnostic_fit: bool = False
    bounded_epochs: int = 10


def run(cfg: NativeStopConfig):
    if not cfg.allow_diagnostic_fit:
        raise PermissionError("retrospective fitting is not enabled")
    if cfg.bounded_epochs != 10:
        raise ValueError("the predeclared comparison is limited to 10 epochs")
    vectors, rows, evidence, hashes = preflight(cfg.provenance)
    baseline_path, summary_path = Path(cfg.baseline_model).resolve(), Path(cfg.baseline_summary).resolve()
    if sha256(baseline_path) != cfg.baseline_model_sha256 or sha256(summary_path) != cfg.baseline_summary_sha256:
        raise ValueError("baseline artifact SHA-256 mismatch")
    baseline_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if baseline_summary.get("stage") != "pilot" or baseline_summary.get("input_hashes") != hashes:
        raise ValueError("baseline pilot is not from identical verified inputs")
    baseline_meta = baseline_summary["models"]["historical_parameters_all"]
    if (baseline_meta["algorithm"], baseline_meta["scope"], baseline_meta["order"], baseline_meta["fit_rows"]) != (
            "minibatch", "all", "original", len(vectors)):
        raise ValueError("baseline fit scope, order or algorithm differs")
    import joblib
    import scipy
    import sklearn
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import normalize

    # Joblib loads executable pickle: this path must point to a trusted pilot artifact.
    baseline = joblib.load(baseline_path)
    if not isinstance(baseline, MiniBatchKMeans):
        raise ValueError("baseline is not MiniBatchKMeans")
    params = baseline.get_params(deep=False)
    required = {"n_clusters": 5, "init": "k-means++", "n_init": 20, "max_iter": 500,
                "random_state": 42, "batch_size": 4096, "tol": 1e-4,
                "max_no_improvement": 10, "reassignment_ratio": 0.01,
                "init_size": None, "compute_labels": True, "verbose": 1}
    if any(params.get(key) != value for key, value in required.items()) or params != baseline_meta["params"]:
        raise ValueError("baseline actual and recorded parameters do not match the historical-parameter condition")
    output = Path(cfg.output_dir).resolve()
    inputs = [Path(cfg.provenance.latent_path).resolve(), Path(cfg.provenance.identity_path).resolve(),
              Path(cfg.provenance.evidence_path).resolve(), baseline_path, summary_path]
    inputs += [Path(evidence[key + "_path"]).resolve() for key in ("checkpoint", "source_dataset")]
    inputs += [Path(item["path"]).resolve() for item in evidence["artifacts"].values() if item.get("path")]
    if output.exists() or any(output == p or output in p.parents for p in inputs):
        raise FileExistsError(f"output exists or contains input: {output}")

    x = normalize(np.asarray(vectors, dtype=np.float32), norm="l2", axis=1).astype(np.float32)
    splits = np.asarray([row["split"] for row in rows])
    selected = np.flatnonzero(np.isin(splits, ["train", "validation"]))
    masks = {s: splits[selected] == s for s in ("train", "validation")}
    historical = load_full_labels(Path(evidence["artifacts"]["full_labels"]["path"]), rows,
                                  cfg.provenance.k)[selected]
    baseline_labels = baseline.predict(x[selected])
    modified = params.copy()
    modified.update(max_iter=cfg.bounded_epochs, tol=0.0, max_no_improvement=None)
    candidate = MiniBatchKMeans(**modified)
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    with (output / "bounded.verbose.log").open("x", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log):
            candidate.fit(x)
    seconds = time.perf_counter() - start
    candidate_labels = candidate.predict(x[selected])
    joblib.dump(candidate, output / "bounded.joblib")
    np.save(output / "baseline_centers.npy", baseline.cluster_centers_, allow_pickle=False)
    np.save(output / "bounded_centers.npy", candidate.cluster_centers_, allow_pickle=False)
    labels = {"historical": historical, "baseline": baseline_labels, "bounded": candidate_labels}
    comparisons = {}
    for ref, cand in (("historical", "baseline"), ("baseline", "bounded"), ("historical", "bounded")):
        name = f"{ref}__vs__{cand}"
        values = compare(labels[ref], labels[cand], masks["train"], masks, cfg.provenance.k)
        if ref == "baseline":
            inverse = np.argsort(np.asarray(values["candidate_to_reference"]))
            values["centers_euclidean_shift_by_reference"] = np.linalg.norm(
                baseline.cluster_centers_.astype(np.float64) -
                candidate.cluster_centers_[inverse].astype(np.float64), axis=1).tolist()
        comparisons[name] = values
    raw_match = {split: int(np.sum(historical[mask] == baseline_labels[mask])) for split, mask in masks.items()}
    with (output / "sample_labels.csv").open("x", encoding="utf-8", newline="") as handle:
        fields = ["vector_row", "original_index", "sample_id", "split", *labels]
        fields.extend(f"{name}__{suffix}" for name in comparisons for suffix in ("aligned", "changed"))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pos, row_index in enumerate(selected):
            row = {key: rows[row_index][key] for key in fields[:4]}
            row.update({name: int(values[pos]) for name, values in labels.items()})
            for name, values in comparisons.items():
                ref, cand = name.split("__vs__")
                aligned = values["candidate_to_reference"][int(labels[cand][pos])]
                row[name + "__aligned"] = aligned
                row[name + "__changed"] = int(labels[ref][pos] != aligned)
            writer.writerow(row)
    result = {"purpose": "retrospective early-stop sensitivity, not a production router",
              "config": asdict(cfg), "input_hashes": hashes,
              "semantics": "all normalized original latent rows fit both models; train-only matching; only train and validation predicted",
              "changed_parameters": {"max_iter": [params["max_iter"], modified["max_iter"]],
                                     "tol": [params["tol"], modified["tol"]],
                                     "max_no_improvement": [params["max_no_improvement"], modified["max_no_improvement"]]},
              "baseline": {"params": params, "n_steps": int(baseline.n_steps_), "n_iter": int(baseline.n_iter_),
                           "fit_inertia": float(baseline.inertia_),
                           "common_rows_sse_float64": {s: _sse(x[selected][m], baseline.cluster_centers_) for s, m in masks.items()}},
              "bounded": {"params": modified, "n_steps": int(candidate.n_steps_), "n_iter": int(candidate.n_iter_),
                          "fit_inertia": float(candidate.inertia_), "fit_seconds": seconds,
                          "common_rows_sse_float64": {s: _sse(x[selected][m], candidate.cluster_centers_) for s, m in masks.items()}},
              "historical_baseline_raw_match_counts": raw_match, "comparisons": comparisons,
              "versions": {"python": platform.python_version(), "numpy": np.__version__,
                           "scipy": scipy.__version__, "sklearn": sklearn.__version__, "joblib": joblib.__version__}}
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(f"bounded fit seconds={seconds:.3f} steps={candidate.n_steps_} iterations={candidate.n_iter_} "
          f"inertia={candidate.inertia_:.8f}", flush=True)
    return result
