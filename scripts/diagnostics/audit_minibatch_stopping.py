#!/usr/bin/env python3
"""Bounded retrospective clustering diagnostics; never a production router.

Call run(config, stage) from a separately maintained, explicit file-internal
configuration.  A pilot and full stage write separate fresh directories.  Test
rows are never predicted, scored, or exported by this module.  The all-data
fits are retrospective diagnostics and MUST NOT be used for production.
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
    from scripts.diagnostics.compare_latent_fit_scope import Config, preflight, sha256
    from scripts.diagnostics.compare_saved_cluster_labels import load_full_labels
except ModuleNotFoundError:
    from attribute_cluster_drift import compare
    from compare_latent_fit_scope import Config, preflight, sha256
    from compare_saved_cluster_labels import load_full_labels


@dataclass(frozen=True)
class AuditConfig:
    provenance: Config
    output_root: str
    allow_diagnostic_fit: bool = False
    seed: int = 42
    seeds: tuple[int, ...] = (42, 7, 2026)
    order_seed: int = 20261028
    batches: tuple[int, ...] = (1024, 4096)
    bounded_epochs: int = 10


def _sse(x, centers):
    """Evaluate nearest-center SSE on identical rows in float64, in batches."""
    centers = np.asarray(centers, dtype=np.float64)
    total = 0.0
    for start in range(0, len(x), 2048):
        block = np.asarray(x[start:start + 2048], dtype=np.float64)
        distances = ((block[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        total += np.min(distances, axis=1).sum()
    return float(total)


def run(cfg: AuditConfig, stage: str):
    if stage not in ("pilot", "full"):
        raise ValueError("stage must be pilot or full")
    if not cfg.allow_diagnostic_fit:
        raise PermissionError("retrospective fit requires explicit permission")
    if cfg.bounded_epochs < 1 or len(cfg.batches) != 2 or min(cfg.batches) < cfg.provenance.k:
        raise ValueError("invalid bounded experiment configuration")
    vectors, rows, evidence, hashes = preflight(cfg.provenance)
    output = Path(cfg.output_root).resolve() / stage
    inputs = [Path(cfg.provenance.latent_path).resolve(), Path(cfg.provenance.identity_path).resolve(),
              Path(cfg.provenance.evidence_path).resolve()]
    inputs += [Path(evidence[key + "_path"]).resolve() for key in ("checkpoint", "source_dataset")]
    inputs += [Path(item["path"]).resolve() for item in evidence["artifacts"].values() if item.get("path")]
    if output.exists() or any(output == p or output in p.parents for p in inputs):
        raise FileExistsError(f"output exists or contains input: {output}")
    from sklearn.cluster import KMeans, MiniBatchKMeans, kmeans_plusplus
    from sklearn.preprocessing import normalize
    import joblib
    import scipy
    import sklearn

    x = normalize(np.asarray(vectors, dtype=np.float32), norm="l2", axis=1).astype(np.float32)
    split = np.asarray([row["split"] for row in rows])
    train_indices = np.flatnonzero(split == "train")
    examined = np.flatnonzero(np.isin(split, ["train", "validation"]))
    masks = {s: split[examined] == s for s in ("train", "validation")}
    train_mask = masks["train"]
    historical = load_full_labels(Path(evidence["artifacts"]["full_labels"]["path"]), rows, cfg.provenance.k)[examined]
    full_order = np.arange(len(x))
    shuffled = np.random.RandomState(cfg.order_seed).permutation(len(x))
    orders = {("all", "original"): full_order, ("all", "permuted"): shuffled,
              ("train", "original"): train_indices,
              ("train", "permuted"): shuffled[split[shuffled] == "train"]}
    centers0, picked = kmeans_plusplus(x[train_indices], n_clusters=cfg.provenance.k,
                                      random_state=cfg.seed)
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "shared_initial_centers.npy", centers0, allow_pickle=False)
    labels = {"historical_full": historical}
    models, meta = {}, {}
    print(f"stage={stage} output={output}", flush=True)

    def fit(name, algorithm, scope, seed, order, *, shared, batch=4096, stopped=True, epochs=None):
        if name in models:
            raise ValueError(f"duplicate fit {name}")
        n_iter = cfg.bounded_epochs if epochs is None else epochs
        args = dict(n_clusters=cfg.provenance.k, init=centers0.copy() if shared else "k-means++",
                    n_init=1 if shared else 20, max_iter=n_iter, random_state=seed,
                    tol=1e-4 if stopped else 0.0, verbose=1)
        if algorithm == "kmeans":
            model = KMeans(**args, algorithm="lloyd", copy_x=True)
        elif algorithm == "minibatch":
            model = MiniBatchKMeans(**args, batch_size=batch, max_no_improvement=10 if stopped else None,
                                    reassignment_ratio=0.01, init_size=None, compute_labels=True)
        else:
            raise ValueError("unknown algorithm")
        start = time.perf_counter()
        with (output / (name + ".verbose.log")).open("x", encoding="utf-8") as verbose:
            with contextlib.redirect_stdout(verbose):
                model.fit(x[orders[(scope, order)]])
        duration = time.perf_counter() - start
        predicted = np.asarray(model.predict(x[examined]), dtype=np.int64)
        labels[name], models[name] = predicted, model
        np.save(output / (name + "_centers.npy"), model.cluster_centers_, allow_pickle=False)
        joblib.dump(model, output / (name + ".joblib"))
        params = model.get_params(deep=False)
        if shared:
            params["init"] = "shared_initial_centers.npy"
        meta[name] = {"algorithm": algorithm, "scope": scope, "seed": seed, "order": order,
                      "fit_rows": len(orders[(scope, order)]), "params": params,
                      "seconds": duration, "fit_inertia": float(model.inertia_),
                      "n_iter": int(model.n_iter_),
                      "n_steps": int(model.n_steps_) if hasattr(model, "n_steps_") else None,
                      "common_rows_sse_float64": {s: _sse(x[examined][m], model.cluster_centers_)
                                                 for s, m in masks.items()}}
        print(f"completed {name}: seconds={duration:.3f} steps={meta[name]['n_steps']} "
              f"iter={meta[name]['n_iter']} inertia={meta[name]['fit_inertia']:.8f}", flush=True)

    if stage == "pilot":
        fit("historical_parameters_all", "minibatch", "all", cfg.seed, "original", shared=False,
            batch=4096, stopped=True, epochs=500)
        fit("pilot_default_train_b1024", "minibatch", "train", cfg.seed, "original", shared=True,
            batch=1024, stopped=True)
        fit("pilot_bounded_train_b1024", "minibatch", "train", cfg.seed, "original", shared=True,
            batch=1024, stopped=False)
    else:
        for scope in ("train", "all"):
            fit(f"kmeans_{scope}", "kmeans", scope, cfg.seed, "original", shared=True, epochs=500)
            for batch in cfg.batches:
                for stop in ("default", "bounded"):
                    fit(f"mini_{scope}_b{batch}_{stop}", "minibatch", scope, cfg.seed,
                        "original", shared=True, batch=batch, stopped=stop == "default")
        # Isolate minibatch RNG and order with the same initial centers and budget.
        for batch in cfg.batches:
            for seed in cfg.seeds:
                for order in ("original", "permuted"):
                    if seed == cfg.seed and order == "original":
                        continue
                    fit(f"mini_train_b{batch}_bounded_seed{seed}_{order}", "minibatch", "train",
                        seed, order, shared=True, batch=batch, stopped=False)

    comparisons = {}
    for name in models:
        references = ["historical_full"]
        if stage == "full":
            references.append("kmeans_train")
            if name.startswith("mini_"):
                bits = name.split("_")
                scope, batch = bits[1], bits[2]
                references.append(f"kmeans_{scope}")
                baseline = f"mini_{scope}_{batch}_default"
                if baseline in models and name != baseline:
                    references.append(baseline)
                bounded = f"mini_{scope}_{batch}_bounded"
                if bounded in models and name != bounded:
                    references.append(bounded)
                if scope == "all" and len(bits) == 4:
                    references.append(f"mini_train_{batch}_{bits[3]}")
        for ref in dict.fromkeys(references):
            if ref == name:
                continue
            result = compare(labels[ref], labels[name], train_mask, masks, cfg.provenance.k)
            if ref in models:
                mapping = np.asarray(result["candidate_to_reference"])
                inv = np.argsort(mapping)
                result["centers_euclidean_shift_by_reference"] = np.linalg.norm(
                    models[ref].cluster_centers_.astype(np.float64) -
                    models[name].cluster_centers_[inv].astype(np.float64), axis=1).tolist()
            comparisons[f"{ref}__vs__{name}"] = result

    with (output / "sample_labels.csv").open("x", encoding="utf-8", newline="") as handle:
        comparison_arrays = {}
        for name, result in comparisons.items():
            ref, cand = name.split("__vs__")
            aligned = np.asarray(result["candidate_to_reference"])[labels[cand]]
            comparison_arrays[name + "__aligned"] = aligned
            comparison_arrays[name + "__changed"] = (labels[ref] != aligned).astype(np.int8)
        fields = ["vector_row", "original_index", "sample_id", "split", *labels, *comparison_arrays]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for position, row_index in enumerate(examined):
            row = {key: rows[row_index][key] for key in fields[:4]}
            row.update({name: int(value[position]) for name, value in labels.items()})
            row.update({name: int(value[position]) for name, value in comparison_arrays.items()})
            writer.writerow(row)
    summary = {"purpose": "bounded retrospective diagnostics; no formal router",
               "config": asdict(cfg), "stage": stage, "input_hashes": hashes,
               "split_policy": "train and validation only; test labels never predicted/exported; omitted excluded from metrics but included in all fit",
               "matching": "maximum overlap on train only, fixed for validation",
               "common_sse": "same normalized float32 inputs and centers converted to float64; nearest Euclidean centers",
               "shared_init": {"seed": cfg.seed, "method": "kmeans_plusplus on train original order",
                               "vector_rows": train_indices[picked].tolist(),
                               "sha256": sha256(output / "shared_initial_centers.npy")},
               "versions": {"python": platform.python_version(), "numpy": np.__version__,
                            "scipy": scipy.__version__, "sklearn": sklearn.__version__, "joblib": joblib.__version__},
               "models": meta, "comparisons": comparisons}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary
