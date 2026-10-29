#!/usr/bin/env python3
"""Retrospective attribution, never a production router. Edit CFG before use.

This fits diagnostic clusterers only after the previous identity/provenance
preflight has been independently reviewed. Historical labels remain read-only;
no historical model reconstruction or exact replication is claimed.
"""
from __future__ import annotations

import csv
import itertools
import json
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

try:
    from scripts.diagnostics.compare_latent_fit_scope import Config, preflight, sha256
    from scripts.diagnostics.compare_saved_cluster_labels import load_full_labels
except ModuleNotFoundError:
    from compare_latent_fit_scope import Config, preflight, sha256
    from compare_saved_cluster_labels import load_full_labels


@dataclass(frozen=True)
class AttributionConfig:
    provenance: Config = field(default_factory=Config)
    output_dir: str = ""  # fresh isolated directory; inputs are never written
    allow_diagnostic_fit: bool = False
    seeds: tuple[int, ...] = (42, 7, 2026)  # first is the base condition
    order_seed: int = 20261028  # fixed permutation, independent of fit seed
    batch_size: int = 4096
    max_no_improvement: int = 10
    reassignment_ratio: float = 0.01


CFG = AttributionConfig()
SPLITS = ("train", "validation", "test")


def compare(ref, candidate, train_mask, split_masks, k):
    """Learn candidate->reference assignment on TRAIN only, freeze for held-out."""
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    ref, candidate = np.asarray(ref), np.asarray(candidate)
    if ref.shape != candidate.shape or ref.shape != train_mask.shape:
        raise ValueError("label and mask dimensions differ")
    if not train_mask.any() or np.any(ref < 0) or np.any(ref >= k) or np.any(candidate < 0) or np.any(candidate >= k):
        raise ValueError("invalid cluster labels or empty train")
    contingency = np.zeros((k, k), dtype=np.int64)
    np.add.at(contingency, (ref[train_mask], candidate[train_mask]), 1)
    left, right = linear_sum_assignment(-contingency)
    mapping = np.empty(k, dtype=int)
    mapping[right] = left
    best = int(contingency[left, right].sum())
    optimal_assignments = sum(int(sum(contingency[p[j], j] for j in range(k)) == best)
                              for p in itertools.permutations(range(k)))
    aligned = mapping[candidate]
    splits = {}
    for split, mask in split_masks.items():
        if not mask.any():
            raise ValueError(f"empty split: {split}")
        a, b = ref[mask], aligned[mask]
        mat = np.zeros((k, k), dtype=np.int64)
        np.add.at(mat, (a, b), 1)
        splits[split] = {"count": int(mask.sum()), "ari": float(adjusted_rand_score(a, b)),
            "nmi": float(normalized_mutual_info_score(a, b)),
            "matched_agreement": float(np.mean(a == b)),
            "migration_fraction": float(np.mean(a != b)), "migration": mat.tolist(),
            "reference_counts": np.bincount(a, minlength=k).tolist(),
            "candidate_aligned_counts": np.bincount(b, minlength=k).tolist()}
    return {"matching_scope": "training only", "candidate_to_reference": mapping.tolist(),
            "train_matching_optimal_assignments": optimal_assignments,
            "mapping_ambiguous": optimal_assignments != 1,
            "train_matching_contingency_raw": contingency.tolist(), "splits": splits}


def _params(cfg, algorithm, seed, init):
    common = dict(n_clusters=cfg.provenance.k, init=init, n_init=cfg.provenance.n_init if isinstance(init, str) else 1,
                  max_iter=cfg.provenance.max_iter, tol=cfg.provenance.tol, random_state=seed)
    if algorithm == "kmeans":
        return {**common, "algorithm": "lloyd", "copy_x": True, "verbose": 0}
    if algorithm == "minibatch":
        return {**common, "batch_size": cfg.batch_size, "max_no_improvement": cfg.max_no_improvement,
                "reassignment_ratio": cfg.reassignment_ratio, "init_size": None,
                "compute_labels": True, "verbose": 0}
    raise ValueError(f"unknown algorithm {algorithm}")


def _name(algorithm, scope, seed, order, shared=False, dtype="float32"):
    return f"{algorithm}_{scope}_seed{seed}_{order}" + ("_shared_init" if shared else "") + ("_float64" if dtype == "float64" else "")


def run(cfg: AttributionConfig):
    vectors, identities, evidence, hashes = preflight(cfg.provenance)
    if not cfg.allow_diagnostic_fit:
        raise PermissionError("diagnostic fitting disabled")
    if not cfg.output_dir:
        raise ValueError("output_dir not configured")
    if len(cfg.seeds) < 2 or len(set(cfg.seeds)) != len(cfg.seeds):
        raise ValueError("at least two distinct seeds are required")
    if cfg.batch_size < cfg.provenance.k or cfg.max_no_improvement < 0 or not 0 <= cfg.reassignment_ratio <= 1:
        raise ValueError("invalid MiniBatchKMeans configuration")
    output = Path(cfg.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    inputs = [Path(cfg.provenance.latent_path).resolve(), Path(cfg.provenance.identity_path).resolve(),
              Path(cfg.provenance.evidence_path).resolve()]
    inputs += [Path(evidence[key + "_path"]).resolve() for key in ("checkpoint", "source_dataset")]
    inputs += [Path(item["path"]).resolve() for item in evidence["artifacts"].values() if item.get("path")]
    if any(output == path or output in path.parents for path in inputs):
        raise ValueError("output cannot contain input artifacts")
    historical = load_full_labels(Path(evidence["artifacts"]["full_labels"]["path"]), identities, cfg.provenance.k)

    from sklearn.cluster import KMeans, MiniBatchKMeans, kmeans_plusplus
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize
    import scipy
    import sklearn
    import joblib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Historical preprocess_latent applies sklearn normalize followed by float32.
    # float64 is a separate precision sensitivity control, not a historical fit.
    x = normalize(np.asarray(vectors, dtype=np.float32), norm="l2", axis=1).astype(np.float32)
    x64 = normalize(np.asarray(vectors, dtype=np.float64), norm="l2", axis=1)
    masks = {s: np.asarray([row["split"] == s for row in identities]) for s in SPLITS}
    train_mask = masks["train"]
    full_order = np.arange(len(x))
    perm = np.random.RandomState(cfg.order_seed).permutation(len(x))
    train_order = full_order[train_mask]
    shuffled_train = perm[train_mask[perm]]
    orders = {("train", "original"): train_order, ("train", "permuted"): shuffled_train,
              ("all", "original"): full_order, ("all", "permuted"): perm}
    # The exact same TRAIN-derived centers and same order for all four controls.
    init, init_indices = kmeans_plusplus(x[train_order], n_clusters=cfg.provenance.k,
                                        random_state=cfg.seeds[0])
    labels, metadata = {}, {}
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "shared_initial_centers.npy", init, allow_pickle=False)
    order_metadata = {}
    for (scope, order), indices in orders.items():
        path = output / f"order_{scope}_{order}.npy"
        np.save(path, indices.astype(np.int64), allow_pickle=False)
        order_metadata[f"{scope}_{order}"] = {"path": path.name, "sha256": sha256(path), "rows": len(indices)}

    def fit_one(alg, scope, seed, order, shared=False, dtype="float32"):
        name = _name(alg, scope, seed, order, shared, dtype)
        features = x if dtype == "float32" else x64
        params = _params(cfg, alg, seed, init.copy() if shared else "k-means++")
        model = (KMeans if alg == "kmeans" else MiniBatchKMeans)(**params)
        idx = orders[(scope, order)]
        model.fit(features[idx])
        predicted = np.asarray(model.predict(features), dtype=np.int64)
        if predicted.shape != (len(x),) or np.any(predicted < 0) or np.any(predicted >= cfg.provenance.k):
            raise ValueError(f"invalid predictions: {name}")
        labels[name] = predicted
        parameter_record = model.get_params(deep=False).copy()
        if shared:
            parameter_record["init"] = "shared_initial_centers.npy"
        metadata[name] = {"algorithm": alg, "fit_scope": scope, "fit_rows": len(idx),
                          "fit_order": order, "seed": seed, "shared_init": shared, "input_dtype": dtype,
                          "params": parameter_record, "inertia": float(model.inertia_),
                          "n_iter": int(model.n_iter_),
                          "n_steps": int(model.n_steps_) if hasattr(model, "n_steps_") else None}
        np.save(output / (name + "_labels.npy"), predicted, allow_pickle=False)
        np.save(output / (name + "_centers.npy"), model.cluster_centers_, allow_pickle=False)
        print(f"completed {name}", flush=True)

    for alg in ("kmeans", "minibatch"):
        for scope in ("train", "all"):
            for seed in cfg.seeds:
                for order in ("original", "permuted"):
                    fit_one(alg, scope, seed, order)
            fit_one(alg, scope, cfg.seeds[0], "original", shared=True)
            fit_one(alg, scope, cfg.seeds[0], "original", dtype="float64")

    comparisons = {}
    def add(kind, refname, candname, ref, cand):
        name = f"{kind}__{refname}__vs__{candname}"
        comparisons[name] = {"kind": kind, "reference": refname, "candidate": candname,
                             **compare(ref, cand, train_mask, masks, cfg.provenance.k)}
    for name, candidate in labels.items():
        add("historical", "historical_full_labels", name, historical, candidate)
    base = cfg.seeds[0]
    for seed in cfg.seeds:
        for order in ("original", "permuted"):
            for scope in ("train", "all"):
                a, b = (_name(alg, scope, seed, order) for alg in ("kmeans", "minibatch"))
                add("algorithm", a, b, labels[a], labels[b])
            for alg in ("kmeans", "minibatch"):
                a, b = (_name(alg, scope, seed, order) for scope in ("train", "all"))
                add("fit_scope", a, b, labels[a], labels[b])
    for alg in ("kmeans", "minibatch"):
        for scope in ("train", "all"):
            a = _name(alg, scope, base, "original")
            b = _name(alg, scope, base, "original", dtype="float64")
            add("precision", a, b, labels[a], labels[b])
            for seed in cfg.seeds:
                a, b = (_name(alg, scope, seed, order) for order in ("original", "permuted"))
                add("order", a, b, labels[a], labels[b])
            for order in ("original", "permuted"):
                for seed in cfg.seeds[1:]:
                    a, b = (_name(alg, scope, n, order) for n in (base, seed))
                    add("seed", a, b, labels[a], labels[b])
            a, b = (_name(alg, s, base, "original", True) for s in ("train", "all"))
            # Only once per algorithm.
            if scope == "train":
                add("shared_init_fit_scope", a, b, labels[a], labels[b])
        for scope in ("train", "all"):
            if alg == "kmeans":
                a, b = (_name(q, scope, base, "original", True) for q in ("kmeans", "minibatch"))
                add("shared_init_algorithm", a, b, labels[a], labels[b])

    pca = PCA(n_components=2, svd_solver="full", whiten=False).fit(x[train_mask])
    xy = pca.transform(x)
    joblib.dump(pca, output / "shared_train_pca.joblib")
    for split in SPLITS:
        fig, axes = plt.subplots(1, 5, figsize=(22, 4))
        mask = masks[split]
        axes[0].scatter(xy[mask, 0], xy[mask, 1], c=historical[mask], s=2, cmap="tab10", vmin=0, vmax=4)
        axes[0].set_title("Historical full labels")
        index = 1
        for alg in ("kmeans", "minibatch"):
            for scope in ("train", "all"):
                name = _name(alg, scope, base, "original")
                comp = comparisons[f"historical__historical_full_labels__vs__{name}"]
                aligned = np.asarray(comp["candidate_to_reference"])[labels[name]]
                axes[index].scatter(xy[mask, 0], xy[mask, 1], c=aligned[mask], s=2, cmap="tab10", vmin=0, vmax=4)
                axes[index].set_title(f"{alg} / {scope}")
                index += 1
        fig.suptitle(f"{split}: same train-fitted PCA; colors aligned on train")
        fig.tight_layout()
        fig.savefig(output / f"pca_{split}.png", dpi=130)
        plt.close(fig)
    summary = {"purpose": "retrospective controlled diagnostic; not a router or a historical reproduction",
               "config": asdict(cfg), "matching_scope": "train only, then applied unchanged to validation and test",
               "row_order": {"original": "latent vector_row order", "permuted": "RandomState(order_seed) permutation of all vector rows, filtered for train fit"},
               "fit_order_files": order_metadata,
               "preprocessing": "same latent per group, sklearn rowwise L2 normalize in float32; four explicit float64 normalization and fitting sensitivity controls; Euclidean distance",
               "initialization_caveat": "native n_init mechanisms differ across algorithms; shared centers fix initial values only, not MiniBatch random sampling or reassignment",
               "omitted_policy": "included in all-data fits, excluded from matching and split metrics",
               "shared_init": {"method": "kmeans_plusplus on train in original order", "seed": base,
                               "original_vector_rows": train_order[init_indices].tolist(),
                               "sha256": sha256(output / "shared_initial_centers.npy")},
               "pca": {"fit": "train", "svd_solver": "full", "explained_variance_ratio": pca.explained_variance_ratio_.tolist()},
               "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                            "sklearn": sklearn.__version__, "joblib": joblib.__version__},
               "input_paths": {"latent": str(Path(cfg.provenance.latent_path).resolve()),
                               "identity": str(Path(cfg.provenance.identity_path).resolve()),
                               "evidence": str(Path(cfg.provenance.evidence_path).resolve())},
               "input_evidence": evidence,
               "input_hashes": hashes, "historical_full_model": evidence["artifacts"]["full_model"],
               "models": metadata, "comparisons": comparisons}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    with (output / "metrics.csv").open("x", encoding="utf-8", newline="") as f:
        fields = ["comparison", "kind", "reference", "candidate", "split", "count", "ari", "nmi", "matched_agreement", "migration_fraction"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name, comp in comparisons.items():
            for split, values in comp["splits"].items():
                writer.writerow({"comparison": name, "kind": comp["kind"], "reference": comp["reference"],
                                 "candidate": comp["candidate"], "split": split,
                                 **{key: values[key] for key in fields[5:]}})
    return summary


if __name__ == "__main__":
    result = run(CFG)
    print(json.dumps({"output_dir": CFG.output_dir, "models": len(result["models"]),
                      "comparisons": len(result["comparisons"])}))
