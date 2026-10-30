#!/usr/bin/env python3
"""Controlled train-only versus all-sample KMeans diagnostic, never a router.

Edit CFG explicitly.  This program refuses to fit unless independently verified
historical artifacts and the exact identity/latent hashes are supplied.  Its
results are a new controlled experiment, not reconstructed historical labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Config:
    latent_path: str = ""  # numeric N x D .npy, identical vector space for both fits
    identity_path: str = ""  # CSV: sample_id, original_index, split, vector_row
    evidence_path: str = ""  # JSON evidence contract described in preflight()
    output_dir: str = ""  # NEW directory under an isolated diagnostics root
    allow_controlled_fit: bool = False  # explicit, after independent evidence review
    seed: int = 42
    n_init: int = 10
    max_iter: int = 300
    tol: float = 1e-4
    k: int = 5
    l2_normalize: bool = True


CFG = Config()
SPLITS = ("train", "validation", "test")
ALL_SPLITS = (*SPLITS, "omitted")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file(value: str, name: str) -> Path:
    if not value:
        raise ValueError(f"{name} is not configured")
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path.resolve()


def _integer(value: str, name: str, row: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {name} at CSV line {row}: {value!r}") from exc
    if result < 0:
        raise ValueError(f"negative {name} at CSV line {row}")
    return result


def load_identities(path: Path, n: int) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "original_index", "split", "vector_row"}
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("identity CSV has missing or duplicate column headers")
        if required - set(reader.fieldnames):
            raise ValueError(f"identity CSV missing columns: {sorted(required - set(reader.fieldnames))}")
        rows = []
        for line, item in enumerate(reader, 2):
            sample = (item.get("sample_id") or "").strip()
            split = (item.get("split") or "").strip()
            if not sample or split not in ALL_SPLITS:
                raise ValueError(f"empty sample ID or unknown split at CSV line {line}")
            rows.append({"sample_id": sample, "split": split,
                         "original_index": _integer(item.get("original_index"), "original_index", line),
                         "vector_row": _integer(item.get("vector_row"), "vector_row", line)})
    if len(rows) != n:
        raise ValueError(f"identity/latent count mismatch: {len(rows)} != {n}")
    for key in ("sample_id", "original_index", "vector_row"):
        values = [r[key] for r in rows]
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate {key} in identity CSV")
    if {r["vector_row"] for r in rows} != set(range(n)):
        raise ValueError("vector_row must identify each latent row exactly once; missing or unknown row")
    if not set(SPLITS).issubset({r["split"] for r in rows}):
        raise ValueError("identity CSV must contain train, validation, and test")
    return sorted(rows, key=lambda r: r["vector_row"])


def preflight(cfg: Config):
    """Read-only validation. Assertions in evidence require independent review.

    Evidence JSON: latent_sha256, identity_sha256, checkpoint_path,
    checkpoint_sha256, latent_layer, preprocessing, source_dataset_path,
    source_dataset_sha256, identity_mapping_method,
    same_model_and_preprocessing_verified=true, same_identity_mapping_verified=true,
    and artifacts mapping train_model/train_labels/full_labels to {path, sha256}.
    full_model uses {path, sha256} or {status: "missing", reason: "..."}.
    Missing full_model prevents exact historical reproduction, not a controlled
    refit if the shared feature space and identity are independently verified.
    No filename, title or equal row count establishes provenance.
    """
    latent_path = _file(cfg.latent_path, "latent_path")
    identity_path = _file(cfg.identity_path, "identity_path")
    evidence_path = _file(cfg.evidence_path, "evidence_path")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    for key in ("checkpoint_sha256", "latent_layer", "preprocessing", "source_dataset_sha256", "identity_mapping_method"):
        if not isinstance(evidence.get(key), str) or not evidence[key].strip():
            raise ValueError(f"evidence missing {key}")
    for key in ("checkpoint_sha256", "source_dataset_sha256"):
        if len(evidence[key]) != 64 or any(c not in "0123456789abcdef" for c in evidence[key].lower()):
            raise ValueError(f"evidence {key} is not SHA-256")
    resource_paths = {}
    for name in ("checkpoint", "source_dataset"):
        resource_paths[name] = _file(evidence.get(name + "_path", ""), name + "_path")
        if sha256(resource_paths[name]) != evidence[name + "_sha256"].lower():
            raise ValueError(f"{name} SHA-256 does not match evidence")
    for key in ("same_model_and_preprocessing_verified", "same_identity_mapping_verified"):
        if evidence.get(key) is not True:
            raise ValueError(f"evidence must independently verify {key}")
    digests = {"latent": sha256(latent_path), "identity": sha256(identity_path),
               "evidence": sha256(evidence_path), "checkpoint": evidence["checkpoint_sha256"].lower(),
               "source_dataset": evidence["source_dataset_sha256"].lower()}
    for name in ("latent", "identity"):
        if evidence.get(name + "_sha256", "").lower() != digests[name]:
            raise ValueError(f"{name} SHA-256 does not match evidence")
    artifacts = evidence.get("artifacts", {})
    for role in ("train_model", "train_labels", "full_model", "full_labels"):
        artifact = artifacts.get(role, {})
        if role == "full_model" and artifact.get("status") == "missing":
            if not artifact.get("reason") or artifact.get("path") or artifact.get("sha256"):
                raise ValueError("missing full_model requires a reason and no path/hash")
            digests[role] = None
            continue
        path = _file(artifact.get("path", ""), "artifact " + role)
        digest = sha256(path)
        if digest != artifact.get("sha256", "").lower():
            raise ValueError(f"artifact hash mismatch: {role}: {path}")
        digests[role] = digest
    vectors = np.load(latent_path, mmap_mode="r", allow_pickle=False)
    if vectors.ndim != 2 or vectors.shape[1] < 2 or not np.issubdtype(vectors.dtype, np.number):
        raise ValueError("latent array must be numeric N x D with D >= 2")
    if not np.isfinite(vectors).all():
        raise ValueError("latent array contains non-finite values")
    rows = load_identities(identity_path, len(vectors))
    if cfg.k != 5 or cfg.n_init < 1 or cfg.max_iter < 1 or cfg.tol <= 0 or not cfg.l2_normalize:
        raise ValueError("K must be 5, L2 normalization enabled, and n_init and max_iter positive")
    if sum(r["split"] == "train" for r in rows) < cfg.k:
        raise ValueError("not enough training rows for KMeans")
    if cfg.l2_normalize and np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError("cannot L2-normalize a zero latent vector")
    return vectors, rows, evidence, digests


def match_labels(train_labels, all_labels, k):
    """Global all-sample maximum-overlap alignment, diagnostic not validation."""
    from scipy.optimize import linear_sum_assignment
    matrix = np.zeros((k, k), dtype=int)
    np.add.at(matrix, (train_labels, all_labels), 1)
    left, right = linear_sum_assignment(-matrix)
    return {int(b): int(a) for a, b in zip(left, right)}


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(cfg: Config) -> dict:
    vectors, rows, evidence, hashes = preflight(cfg)
    if not cfg.allow_controlled_fit:
        raise PermissionError("controlled fitting disabled; review provenance then explicitly enable it")
    if not cfg.output_dir:
        raise ValueError("output_dir is not configured")
    output = Path(cfg.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    inputs = [Path(cfg.latent_path).resolve(), Path(cfg.identity_path).resolve(), Path(cfg.evidence_path).resolve()]
    inputs.extend(Path(evidence[name + "_path"]).resolve() for name in ("checkpoint", "source_dataset"))
    inputs.extend(Path(a["path"]).resolve() for a in evidence["artifacts"].values() if a.get("path"))
    if any(output == p or output in p.parents for p in inputs):
        raise ValueError("output directory cannot contain an input artifact")
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    import sklearn
    import scipy
    import joblib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.asarray(vectors, dtype=np.float64).copy()
    if cfg.l2_normalize:
        x /= np.linalg.norm(x, axis=1)[:, None]
    train_mask = np.array([r["split"] == "train" for r in rows])
    opts = dict(n_clusters=cfg.k, random_state=cfg.seed, n_init=cfg.n_init,
                max_iter=cfg.max_iter, tol=cfg.tol, init="k-means++", algorithm="lloyd")
    train_model = KMeans(**opts).fit(x[train_mask])
    all_model = KMeans(**opts).fit(x)
    train_labels = train_model.predict(x)
    all_raw = all_model.predict(x)
    mapping = match_labels(train_labels, all_raw, cfg.k)
    all_labels = np.array([mapping[int(a)] for a in all_raw])
    train_dist = train_model.transform(x)
    all_dist = all_model.transform(x)
    margin_train = np.diff(np.sort(train_dist, axis=1)[:, :2], axis=1)[:, 0]
    margin_all = np.diff(np.sort(all_dist, axis=1)[:, :2], axis=1)[:, 0]
    pca = PCA(n_components=2, svd_solver="full", whiten=False).fit(x[train_mask])
    projection = pca.transform(x)
    center_train = pca.transform(train_model.cluster_centers_)
    center_all = pca.transform(all_model.cluster_centers_)
    by_split = {}
    migration_rows = []
    for split in SPLITS:
        mask = np.array([r["split"] == split for r in rows])
        a, b = train_labels[mask], all_labels[mask]
        migration = np.zeros((cfg.k, cfg.k), dtype=int)
        np.add.at(migration, (a, b), 1)
        by_split[split] = {"count": int(mask.sum()), "ari": float(adjusted_rand_score(a, b)),
                           "nmi": float(normalized_mutual_info_score(a, b)),
                           "matched_agreement": float(np.mean(a == b)),
                           "migration_fraction": float(np.mean(a != b)),
                           "train_fit_counts": np.bincount(a, minlength=cfg.k).tolist(),
                           "all_fit_matched_counts": np.bincount(b, minlength=cfg.k).tolist(),
                           "train_fit_proportions": (np.bincount(a, minlength=cfg.k) / mask.sum()).tolist(),
                           "all_fit_matched_proportions": (np.bincount(b, minlength=cfg.k) / mask.sum()).tolist(),
                           "migration": migration.tolist()}
        for category, select in (("changed", a != b), ("unchanged", a == b)):
            by_split[split]["margin_" + category] = {"count": int(select.sum())}
            for label, margins in (("train_fit", margin_train[mask][select]), ("all_fit", margin_all[mask][select])):
                by_split[split]["margin_" + category][label + "_mean"] = float(margins.mean()) if len(margins) else None
                by_split[split]["margin_" + category][label + "_median"] = float(np.median(margins)) if len(margins) else None
        by_split[split]["major_migration_directions"] = [
            {"from": i, "to": j, "count": int(migration[i, j])}
            for i, j in sorted(((i, j) for i in range(cfg.k) for j in range(cfg.k) if i != j and migration[i, j]),
                               key=lambda ij: (-migration[ij], ij))[:5]]
        for i in range(cfg.k):
            for j in range(cfg.k):
                migration_rows.append({"split": split, "train_cluster": i, "all_matched_cluster": j,
                                       "count": int(migration[i, j]),
                                       "fraction_of_split": float(migration[i, j] / mask.sum())})
    inverse = {v: k for k, v in mapping.items()}
    centers = []
    for c in range(cfg.k):
        other = inverse[c]
        within = train_dist[train_mask & (train_labels == c), c]
        mean_within = float(within.mean()) if len(within) else 0.0
        displacement = float(np.linalg.norm(train_model.cluster_centers_[c] - all_model.cluster_centers_[other]))
        centers.append({"train_cluster": c, "all_raw_cluster": other, "displacement": displacement,
                        "train_mean_within_distance": mean_within,
                        "displacement_over_within": displacement / mean_within if mean_within else None})
    sample_rows = []
    for i, row in enumerate(rows):
        sample_rows.append({**row, "train_fit_cluster": int(train_labels[i]), "all_fit_raw_cluster": int(all_raw[i]),
                            "all_fit_matched_cluster": int(all_labels[i]), "changed": int(train_labels[i] != all_labels[i]),
                            "train_margin": float(margin_train[i]), "all_margin": float(margin_all[i])})
    summary = {"scope": "controlled refit; NOT historical labels or formal router",
               "historical_full_model_missing": evidence["artifacts"]["full_model"].get("status") == "missing",
               "config": {"k": cfg.k, "seed": cfg.seed, "n_init": cfg.n_init, "max_iter": cfg.max_iter,
                          "init": "k-means++", "algorithm": "lloyd", "distance": "euclidean",
                          "l2_normalize": cfg.l2_normalize, "tol": cfg.tol, "input_compute_dtype": "float64"},
               "pca_fit_scope": "train", "pca_config": {"n_components": 2, "svd_solver": "full", "whiten": False,
                                                               "random_state": None, "randomized": False},
               "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
               "matching_scope": "all diagnostic samples, maximum overlap; not predictive evaluation",
               "versions": {"python": platform.python_version(), "numpy": np.__version__,
                            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
               "omitted_count": sum(r["split"] == "omitted" for r in rows),
               "omitted_policy": "included in all-sample fit and global cluster matching, excluded from split metrics",
               "raw_all_to_train_mapping": mapping, "splits": by_split, "centers": centers,
               "input_hashes": hashes, "input_paths": {"latent": str(Path(cfg.latent_path).resolve()),
               "identity": str(Path(cfg.identity_path).resolve()), "evidence": str(Path(cfg.evidence_path).resolve())},
               "evidence": evidence}
    output.mkdir(parents=True, exist_ok=False)
    joblib.dump(train_model, output / "controlled_train_only_kmeans.joblib")
    joblib.dump(all_model, output / "controlled_all_data_kmeans.joblib")
    joblib.dump(pca, output / "controlled_train_pca.joblib")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    _write_csv(output / "samples.csv", sample_rows)
    _write_csv(output / "migration.csv", migration_rows)
    _write_csv(output / "centers.csv", centers)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, labels, title in zip(axes, (train_labels, all_labels), ("train-fit", "all-fit aligned")):
        ax.scatter(projection[:, 0], projection[:, 1], c=labels, s=4, cmap="tab10", vmin=0, vmax=cfg.k-1)
        ax.set_title(title + " (same samples and PCA)")
    fig.tight_layout()
    fig.savefig(output / "shared_pca_all.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    for i, split in enumerate(SPLITS):
        mask = np.array([r["split"] == split for r in rows])
        for j, (labels, title) in enumerate(((train_labels, "train-fit"), (all_labels, "all-fit aligned"))):
            axes[i, j].scatter(projection[mask, 0], projection[mask, 1], c=labels[mask], s=4, cmap="tab10", vmin=0, vmax=cfg.k-1)
            axes[i, j].set_title(f"{split}: {title}")
    fig.tight_layout()
    fig.savefig(output / "shared_pca_by_split.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, split in enumerate(SPLITS):
        axes[i].imshow(by_split[split]["migration"], cmap="Blues")
        for row in range(cfg.k):
            for col in range(cfg.k):
                axes[i].text(col, row, str(by_split[split]["migration"][row][col]), ha="center", va="center", fontsize=7)
        axes[i].set_title(split)
        axes[i].set_xlabel("all-fit aligned")
        axes[i].set_ylabel("train-fit")
    fig.tight_layout()
    fig.savefig(output / "migration.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots()
    for c in range(cfg.k):
        start, end = center_train[c], center_all[inverse[c]]
        ax.annotate(str(c), start)
        ax.arrow(*(start.tolist()), *((end-start).tolist()), head_width=0.01, length_includes_head=True)
    ax.set_title("Center shift in train-fitted PCA (projection only)")
    fig.tight_layout()
    fig.savefig(output / "center_shift_pca.png", dpi=140)
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-config", action="store_true", help="read-only provenance preflight; never fits")
    args = parser.parse_args()
    if args.check_config:
        vectors, rows, _, hashes = preflight(CFG)
        print(json.dumps({"shape": list(vectors.shape), "count": len(rows), "hashes": hashes, "fitted": False}))
    else:
        result = run(CFG)
        print(json.dumps({"output_dir": CFG.output_dir, "splits": result["splits"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
