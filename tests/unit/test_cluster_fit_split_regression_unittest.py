import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.cluster_specialists.config import SplitRatios
from experiments.cluster_specialists.manifests import (
    build_manifest, read_manifest, validate_manifest, write_manifests)
from experiments.cluster_specialists.routing import CentroidRouter, load_router
from scripts.data.generate_cluster_splits import _align_features, prepare_centroid
from scripts.data import generate_cluster_splits as entry


class TrainOnlyFitRegressionTests(unittest.TestCase):
    def setUp(self):
        self.records = [{"mp_id": f"mp-{i}", "material_id": f"physical-{i}"}
                        for i in range(90)]
        # Different clusters are distributed throughout each deterministic split.
        self.features = np.array([[float((i % 3) * 10), float(i % 5) / 20]
                                  for i in range(90)])
        self.kwargs = dict(ratios=SplitRatios(.6, .2, .2), seed=43,
                           physical_id_fields=("material_id",),
                           feature_names=("x", "y"))

    def test_holdout_changes_do_not_change_fit_or_mapping_and_nine_manifests(self):
        first, splits = prepare_centroid(self.records, self.features, **self.kwargs)
        for changed_split in ("validation", "test"):
            modified = self.features.copy()
            for i, split in enumerate(splits):
                if split == changed_split:
                    modified[i] += (1000.0, -300.0)
            second, splits_again = prepare_centroid(self.records, modified, **self.kwargs)
            self.assertEqual(splits, splits_again)
            np.testing.assert_array_equal(first.mean_, second.mean_)
            np.testing.assert_array_equal(first.scale_, second.scale_)
            np.testing.assert_array_equal(first.centers_, second.centers_)
            self.assertEqual(first.raw_to_cluster_, second.raw_to_cluster_)
            self.assertEqual(first.fit_provenance_, second.fit_provenance_)
        train_indices = first.fit_provenance_["train_original_indices"]
        self.assertEqual(train_indices, [i for i, split in enumerate(splits) if split == "train"])
        self.assertTrue(set(train_indices).isdisjoint(
            i for i, split in enumerate(splits) if split in ("validation", "test")))
        np.testing.assert_allclose(first.mean_, self.features[train_indices].mean(axis=0))
        rows = build_manifest(self.records, first.predict(self.features),
                              ratios=self.kwargs["ratios"], seed=self.kwargs["seed"],
                              physical_id_fields=self.kwargs["physical_id_fields"], splits=splits)
        report = validate_manifest(rows, expected_count=len(self.records))
        self.assertTrue(all(value == 0 for value in report["split_intersections"].values()))
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_manifests(rows, tmp, seed=43, ratios=self.kwargs["ratios"], source="synthetic")
            all_indices = []
            for cluster in "ABC":
                for split, name in (("train", "train"), ("validation", "val"), ("test", "test")):
                    selected = read_manifest(paths[f"{cluster}_{name}"])
                    expected = [row for row in rows if row["cluster_id"] == cluster and row["split"] == split]
                    self.assertEqual([int(row["original_index"]) for row in selected],
                                     [row["original_index"] for row in expected])
                    self.assertTrue(selected)
                    all_indices.extend(int(row["original_index"]) for row in selected)
            self.assertEqual(sorted(all_indices), list(range(len(self.records))))
            artifact = first.save(Path(tmp) / "router.json")
            loaded = load_router(artifact)
            prepare_centroid(self.records, modified, router=loaded, **self.kwargs)
            changed_train = self.features.copy()
            changed_train[train_indices[0], 0] += 1
            with self.assertRaisesRegex(ValueError, "provenance"):
                prepare_centroid(self.records, changed_train, router=loaded, **self.kwargs)
            with self.assertRaisesRegex(ValueError, "provenance"):
                prepare_centroid(self.records, self.features, router=CentroidRouter(
                    feature_names=("x", "y")).fit(self.features), **self.kwargs)

    def test_missing_id_and_predeclared_split_are_checked_before_fit(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            prepare_centroid(self.records, self.features, **dict(
                self.kwargs, physical_id_fields=("mp_id", "material_id")))
        missing = list(self.records)
        missing[0] = {"mp_id": "not-the-configured-physical-id"}
        with self.assertRaisesRegex(ValueError, "missing physical ID"):
            prepare_centroid(missing, self.features, **self.kwargs)
        records = list(self.records)
        records[1] = dict(records[1], material_id=records[0]["material_id"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "splits.csv"
            def write(splits, ids=None):
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["original_index", "physical_id", "split"])
                    writer.writeheader()
                    writer.writerows({"original_index": i,
                                      "physical_id": (ids or [r["material_id"] for r in records])[i],
                                      "split": split} for i, split in enumerate(splits))
            splits = ["train" if i < 54 else "validation" if i < 72 else "test"
                      for i in range(len(records))]
            write(splits)
            router, actual = prepare_centroid(records, self.features, split_path=path, **self.kwargs)
            self.assertEqual(actual, splits)
            self.assertEqual(router.fit_provenance_["train_original_indices"], list(range(54)))
            splits[1] = "test"
            write(splits)
            with self.assertRaisesRegex(ValueError, "leakage"):
                prepare_centroid(records, self.features, split_path=path, **self.kwargs)
            splits[1] = "train"
            ids = [r["material_id"] for r in records]
            ids[0] = "wrong"
            write(splits, ids)
            with self.assertRaisesRegex(ValueError, "physical ID mismatch"):
                prepare_centroid(records, self.features, split_path=path, **self.kwargs)

    def test_shuffled_feature_sidecar_aligns_and_rejects_bad_identity(self):
        order = np.random.default_rng(17).permutation(len(self.records))
        shuffled = self.features[order]
        entries = [{"feature_row": row, "original_index": int(index),
                    "physical_id": self.records[index]["material_id"]}
                   for row, index in enumerate(order)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.csv"
            def write(rows):
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["feature_row", "original_index", "physical_id"])
                    writer.writeheader()
                    writer.writerows(rows)
            write(entries[::-1])
            aligned = _align_features(self.records, shuffled, path, ("material_id",))
            np.testing.assert_array_equal(aligned, self.features)
            reference, splits = prepare_centroid(self.records, self.features, **self.kwargs)
            shuffled_router, shuffled_splits = prepare_centroid(self.records, aligned, **self.kwargs)
            self.assertEqual(splits, shuffled_splits)
            self.assertEqual(reference.fit_provenance_, shuffled_router.fit_provenance_)
            np.testing.assert_array_equal(reference.centers_, shuffled_router.centers_)
            write(entries[:-1])
            with self.assertRaisesRegex(ValueError, "count"):
                _align_features(self.records, shuffled, path, ("material_id",))
            duplicate = [dict(row) for row in entries]
            duplicate[-1]["feature_row"] = duplicate[0]["feature_row"]
            write(duplicate)
            with self.assertRaisesRegex(ValueError, "duplicate or unknown feature_row"):
                _align_features(self.records, shuffled, path, ("material_id",))
            duplicate_index = [dict(row) for row in entries]
            duplicate_index[-1]["original_index"] = duplicate_index[0]["original_index"]
            duplicate_index[-1]["physical_id"] = duplicate_index[0]["physical_id"]
            write(duplicate_index)
            with self.assertRaisesRegex(ValueError, "duplicate or unknown original_index"):
                _align_features(self.records, shuffled, path, ("material_id",))
            unknown = [dict(row) for row in entries]
            unknown[-1]["original_index"] = len(self.records)
            write(unknown)
            with self.assertRaisesRegex(ValueError, "duplicate or unknown original_index"):
                _align_features(self.records, shuffled, path, ("material_id",))
            wrong = [dict(row) for row in entries]
            wrong[0]["physical_id"] = "wrong"
            write(wrong)
            with self.assertRaisesRegex(ValueError, "physical ID mismatch"):
                _align_features(self.records, shuffled, path, ("material_id",))

    def test_entry_generates_manifests_from_shuffled_feature_sidecar(self):
        order = np.random.default_rng(5).permutation(len(self.records))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "data.json"
            data_path.write_text(json.dumps(self.records), encoding="utf-8")
            features_path = root / "features.npy"
            np.save(features_path, self.features[order])
            index_path = root / "feature_index.csv"
            with index_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["feature_row", "original_index", "physical_id"])
                writer.writeheader()
                writer.writerows({"feature_row": row, "original_index": int(index),
                                  "physical_id": self.records[index]["material_id"]}
                                 for row, index in enumerate(order))
            router_path = root / "router.json"
            output_dir = root / "manifests"
            with patch.multiple(entry, DATA_PATH=data_path, ROUTER_FEATURES_PATH=features_path,
                                ROUTER_FEATURES_INDEX_PATH=index_path, ROUTER_MODE="fit_centroid",
                                OUTPUT_DIR=output_dir, ROUTER_ARTIFACT_PATH=router_path,
                                PHYSICAL_ID_FIELDS=("material_id",), FEATURE_NAMES=("x", "y"),
                                SPLIT_RATIOS=self.kwargs["ratios"], RANDOM_SEED=self.kwargs["seed"],
                                PREDECLARED_SPLIT_PATH=None), patch("sys.argv", ["generate_cluster_splits.py"]):
                entry.main()
            rows = read_manifest(output_dir / "cluster_manifest.csv")
            router = load_router(router_path)
            self.assertEqual(router.fit_provenance_["train_original_indices"],
                             [int(row["original_index"]) for row in rows if row["split"] == "train"])
            self.assertEqual([row["cluster_id"] for row in rows], router.predict(self.features).tolist())
            self.assertEqual(len(list(output_dir.glob("[ABC]_*.csv"))), 9)


if __name__ == "__main__":
    unittest.main()
