#!/usr/bin/env python3
"""Compare saved historical labels with predictions from the saved train router.

Edit CFG; every input is required. Never fits a clusterer or replaces a router.
Joblib artifacts must be trusted: loading a pickle can execute arbitrary code.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    from scripts.diagnostics.compare_latent_fit_scope import Config, preflight, match_labels, sha256
except ModuleNotFoundError:
    from compare_latent_fit_scope import Config, preflight, match_labels, sha256


@dataclass(frozen=True)
class SavedConfig:
    provenance: Config = field(default_factory=Config)
    pca_path: str = ""
    pca_sha256: str = ""
    controlled_train_model_path: str = ""
    controlled_train_model_sha256: str = ""
    controlled_all_model_path: str = ""
    controlled_all_model_sha256: str = ""
    output_dir: str = ""


CFG = SavedConfig()


def load_full_labels(path: Path, identities: list[dict], k: int) -> np.ndarray:
    """Join original_index and sample_id explicitly; never use CSV row position."""
    by_index = {r["original_index"]: r for r in identities}
    found = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"index", "mp_id", "fusiformer_cluster"}
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("missing or duplicate historical assignment headers")
        if required - set(reader.fieldnames):
            raise ValueError(f"historical labels missing fields: {sorted(required - set(reader.fieldnames))}")
        for line, row in enumerate(reader, 2):
            try:
                index = int(row["index"])
                label = int(row["fusiformer_cluster"])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"invalid historical index or cluster at line {line}") from exc
            if index in found:
                raise ValueError(f"duplicate historical index: {index}")
            if index not in by_index:
                raise ValueError(f"unknown historical index: {index}")
            if not row["mp_id"] or row["mp_id"] != by_index[index]["sample_id"]:
                raise ValueError(f"mp_id mismatch for original_index={index}")
            if not 0 <= label < k:
                raise ValueError(f"historical cluster out of range at index {index}")
            found[index] = label
    missing = set(by_index) - set(found)
    if missing:
        raise ValueError(f"historical labels missing {len(missing)} identities")
    return np.asarray([found[row["original_index"]] for row in identities], dtype=int)


def run(cfg: SavedConfig):
    vectors, identities, evidence, hashes = preflight(cfg.provenance)
    if not cfg.pca_path or not cfg.pca_sha256:
        raise ValueError("pca_path and pca_sha256 must be configured")
    pca_path = Path(cfg.pca_path).resolve()
    if not pca_path.is_file():
        raise FileNotFoundError(f"PCA not found: {pca_path}")
    if sha256(pca_path) != cfg.pca_sha256.lower():
        raise ValueError("PCA SHA-256 mismatch")
    controlled_paths = {}
    for name in ("controlled_train_model", "controlled_all_model"):
        value = getattr(cfg, name + "_path")
        expected = getattr(cfg, name + "_sha256")
        if not value or not expected:
            raise ValueError(f"{name} path and SHA-256 must be configured")
        path = Path(value).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{name} not found: {path}")
        if sha256(path) != expected.lower():
            raise ValueError(f"{name} SHA-256 mismatch")
        controlled_paths[name] = path
    if not cfg.output_dir:
        raise ValueError("output_dir is not configured")
    output = Path(cfg.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    inputs = [Path(cfg.provenance.latent_path).resolve(), Path(cfg.provenance.identity_path).resolve(),
              Path(cfg.provenance.evidence_path).resolve(), pca_path, *controlled_paths.values()]
    inputs.extend(Path(evidence[key + "_path"]).resolve() for key in ("checkpoint", "source_dataset"))
    inputs.extend(Path(a["path"]).resolve() for a in evidence["artifacts"].values() if a.get("path"))
    if any(output == p or output in p.parents for p in inputs):
        raise ValueError("output cannot contain an input")

    import joblib
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.preprocessing import normalize
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    full_raw = load_full_labels(Path(evidence["artifacts"]["full_labels"]["path"]), identities, cfg.provenance.k)
    model = joblib.load(evidence["artifacts"]["train_model"]["path"])
    pca = joblib.load(pca_path)
    x = normalize(np.asarray(vectors), norm="l2", axis=1)
    if getattr(model, "n_clusters", None) != cfg.provenance.k:
        raise ValueError("historical train model K mismatch")
    historical_x = np.asarray(x, dtype=np.asarray(model.cluster_centers_).dtype)
    train = np.asarray(model.predict(historical_x), dtype=int)
    distances = np.asarray(model.transform(historical_x))
    if distances.shape != (len(x), cfg.provenance.k) or not np.isfinite(distances).all():
        raise ValueError("invalid router distance matrix")
    if train.shape != (len(x),) or np.any(train < 0) or np.any(train >= cfg.provenance.k):
        raise ValueError("invalid train model predictions")
    margin = np.diff(np.sort(distances, axis=1)[:, :2], axis=1)[:, 0]
    mapping = match_labels(train, full_raw, cfg.provenance.k)
    full_aligned = np.asarray([mapping[int(label)] for label in full_raw])
    controlled_labels = {}
    for name, path in controlled_paths.items():
        controlled = joblib.load(path)
        if getattr(controlled, "n_clusters", None) != cfg.provenance.k:
            raise ValueError(f"{name} K mismatch")
        controlled_x = normalize(np.asarray(vectors, dtype=np.asarray(controlled.cluster_centers_).dtype), norm="l2", axis=1)
        labels = np.asarray(controlled.predict(controlled_x), dtype=int)
        if labels.shape != (len(x),) or np.any(labels < 0) or np.any(labels >= cfg.provenance.k):
            raise ValueError(f"{name} predictions invalid")
        controlled_labels[name] = labels
    controls = {}
    for name, left, right in (
        ("historical_train_vs_controlled_train", train, controlled_labels["controlled_train_model"]),
        ("historical_full_vs_controlled_all", full_raw, controlled_labels["controlled_all_model"]),
    ):
        control_map = match_labels(left, right, cfg.provenance.k)
        aligned = np.asarray([control_map[int(value)] for value in right])
        split_stats = {}
        for split in ("train", "validation", "test", "omitted"):
            mask = np.asarray([r["split"] == split for r in identities])
            if mask.any():
                split_stats[split] = {"count": int(mask.sum()),
                    "ari": float(adjusted_rand_score(left[mask], right[mask])),
                    "nmi": float(normalized_mutual_info_score(left[mask], right[mask])),
                    "matched_agreement": float(np.mean(left[mask] == aligned[mask]))}
        controls[name] = {"right_to_left_mapping": control_map, "matching_scope": "all rows including omitted",
                          "splits": split_stats}
    pca_x = normalize(np.asarray(vectors, dtype=np.asarray(pca.components_).dtype), norm="l2", axis=1)
    xy = np.asarray(pca.transform(pca_x))
    if xy.shape != (len(x), 2) or not np.isfinite(xy).all():
        raise ValueError("PCA must produce finite N x 2 coordinates")
    stats = {}
    migration_rows = []
    for split in ("train", "validation", "test", "omitted"):
        mask = np.array([r["split"] == split for r in identities])
        if not mask.any():
            continue
        a, b = train[mask], full_aligned[mask]
        matrix = np.zeros((cfg.provenance.k, cfg.provenance.k), dtype=int)
        np.add.at(matrix, (a, b), 1)
        changed = a != b
        counts_a = np.bincount(a, minlength=cfg.provenance.k)
        counts_b = np.bincount(b, minlength=cfg.provenance.k)
        stats[split] = {"count": int(mask.sum()), "train_counts": counts_a.tolist(),
                        "full_aligned_counts": counts_b.tolist(),
                        "train_proportions": (counts_a / len(a)).tolist(),
                        "full_aligned_proportions": (counts_b / len(a)).tolist(),
                        "ari": float(adjusted_rand_score(a, b)),
                        "nmi": float(normalized_mutual_info_score(a, b)),
                        "matched_agreement": float(np.mean(~changed)),
                        "migration_fraction": float(np.mean(changed)), "migration": matrix.tolist(),
                        "train_margin_changed_mean": float(margin[mask][changed].mean()) if changed.any() else None,
                        "train_margin_unchanged_mean": float(margin[mask][~changed].mean()) if (~changed).any() else None,
                        "major_migrations": [{"from": i, "to": j, "count": int(matrix[i, j])}
                           for i, j in sorted(((i, j) for i in range(cfg.provenance.k) for j in range(cfg.provenance.k)
                                             if i != j and matrix[i, j]), key=lambda ij: (-matrix[ij], ij))[:5]]}
        migration_rows.extend({"split": split, "train_cluster": i, "full_matched_cluster": j,
                               "count": int(matrix[i, j]), "fraction_of_split": float(matrix[i, j] / len(a))}
                              for i in range(cfg.provenance.k) for j in range(cfg.provenance.k))
    summary = {"scope": "Historical saved labels vs saved train-router predictions; descriptive only, not formal evaluation",
               "full_model_status": evidence["artifacts"]["full_model"],
               "full_margin": "unavailable without historical full-data model; labels were not inferred",
               "matching_scope": "all rows including omitted; Hungarian maximum overlap",
               "full_raw_to_train_mapping": mapping, "splits": stats,
               "control_comparisons": controls,
               "controlled_train_order_note": "controlled fit uses full-array row order, historical train fit used split order; seed equality does not imply identical initialization",
               "input_hashes": {**hashes, "pca": sha256(pca_path),
                                **{name: sha256(path) for name, path in controlled_paths.items()}},
               "input_paths": {"latent": cfg.provenance.latent_path, "identity": cfg.provenance.identity_path,
                               "evidence": cfg.provenance.evidence_path, "pca": str(pca_path),
                               **{name: str(path) for name, path in controlled_paths.items()}}}
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    samples = [{**row, "historical_train_cluster": int(train[i]), "historical_full_cluster": int(full_raw[i]),
                "historical_full_aligned": int(full_aligned[i]), "changed": int(train[i] != full_aligned[i]),
                "train_nearest_distance_margin": float(margin[i]),
                "controlled_train_raw_cluster": int(controlled_labels["controlled_train_model"][i]),
                "controlled_all_raw_cluster": int(controlled_labels["controlled_all_model"][i])}
               for i, row in enumerate(identities)]
    for name, records in (("samples.csv", samples), ("migration.csv", migration_rows)):
        with (output / name).open("x", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    for selected in (None, "train", "validation", "test"):
        mask = np.ones(len(x), dtype=bool) if selected is None else np.array([r["split"] == selected for r in identities])
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, labels, title in zip(axes, (train, full_aligned), ("Saved train-only router", "Saved full labels, aligned")):
            ax.scatter(xy[mask, 0], xy[mask, 1], c=labels[mask], s=4, cmap="tab10", vmin=0, vmax=cfg.provenance.k - 1)
            ax.set_title(f"{selected or 'all'}: {title}")
        fig.tight_layout()
        fig.savefig(output / f"historical_pca_{selected or 'all'}.png", dpi=140)
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, split in zip(axes, ("train", "validation", "test")):
        matrix = np.asarray(stats[split]["migration"])
        ax.imshow(matrix, cmap="Blues")
        for i in range(cfg.provenance.k):
            for j in range(cfg.provenance.k):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=7)
        ax.set_title(split)
        ax.set_xlabel("historical full label, aligned")
        ax.set_ylabel("historical train label")
    fig.tight_layout()
    fig.savefig(output / "historical_migration.png", dpi=140)
    plt.close(fig)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(CFG), ensure_ascii=False))
