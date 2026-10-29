"""Temporary synthetic data: software checks, not scientific validation."""
import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from scripts.diagnostics.attribute_cluster_drift import AttributionConfig, compare, run
from scripts.diagnostics.compare_latent_fit_scope import Config, sha256


class ClusterAttributionTests(unittest.TestCase):
    def test_mapping_fixed_on_train_even_when_test_prefers_reverse(self):
        ref = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 1])
        cand = np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 1])
        train = np.arange(10) < 4
        masks = {"train": train, "validation": np.arange(10) >= 8,
                 "test": (np.arange(10) >= 4) & (np.arange(10) < 8)}
        result = compare(ref, cand, train, masks, 2)
        self.assertEqual(result["candidate_to_reference"], [0, 1])
        self.assertEqual(result["splits"]["test"]["matched_agreement"], 0.0)
        self.assertEqual(result["splits"]["test"]["ari"], 1.0)
        self.assertFalse(result["mapping_ambiguous"])

    def test_ambiguous_train_mapping_is_explicit(self):
        ref = np.array([0, 1, 0, 1, 0, 1])
        cand = np.array([0, 0, 0, 1, 0, 1])
        masks = {"train": np.array([1, 1, 0, 0, 0, 0], dtype=bool),
                 "validation": np.array([0, 0, 1, 1, 0, 0], dtype=bool),
                 "test": np.array([0, 0, 0, 0, 1, 1], dtype=bool)}
        result = compare(ref, cand, masks["train"], masks, 2)
        self.assertTrue(result["mapping_ambiguous"])

    def test_end_to_end_fixed_centers_and_orders_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            x = np.random.RandomState(2).normal(size=(46, 6))
            latent = root / "latent.npy"
            np.save(latent, x)
            identity = root / "identity.csv"
            labels = root / "assignments.csv"
            rows = [{"sample_id": f"mp-{i}", "original_index": i * 2 + 1,
                     "split": "train" if i < 30 else "validation" if i < 38 else "test" if i < 45 else "omitted",
                     "vector_row": i} for i in range(len(x))]
            with identity.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(reversed(rows))
            with labels.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["index", "mp_id", "fusiformer_cluster"]); w.writeheader()
                for r in reversed(rows):
                    w.writerow({"index": r["original_index"], "mp_id": r["sample_id"], "fusiformer_cluster": r["vector_row"] % 5})
            for name in ("checkpoint", "dataset", "train_model", "train_labels"):
                (root / name).write_text(name)
            evidence = {"checkpoint_path": str(root / "checkpoint"), "checkpoint_sha256": sha256(root / "checkpoint"),
                        "source_dataset_path": str(root / "dataset"), "source_dataset_sha256": sha256(root / "dataset"),
                        "latent_sha256": sha256(latent), "identity_sha256": sha256(identity),
                        "latent_layer": "synthetic", "preprocessing": "synthetic",
                        "identity_mapping_method": "fixture", "same_model_and_preprocessing_verified": True,
                        "same_identity_mapping_verified": True,
                        "artifacts": {"train_model": {"path": str(root / "train_model"), "sha256": sha256(root / "train_model")},
                                      "train_labels": {"path": str(root / "train_labels"), "sha256": sha256(root / "train_labels")},
                                      "full_labels": {"path": str(labels), "sha256": sha256(labels)},
                                      "full_model": {"status": "missing", "reason": "fixture"}}}
            ep = root / "evidence.json"; ep.write_text(json.dumps(evidence))
            provenance = Config(str(latent), str(identity), str(ep), n_init=1, max_iter=10)
            cfg = AttributionConfig(provenance=provenance, output_dir=str(root / "result"),
                                    allow_diagnostic_fit=True, seeds=(42, 7), batch_size=8)
            out = run(cfg)
            self.assertEqual(len(out["models"]), 24)
            shared = [v for v in out["models"].values() if v["shared_init"]]
            self.assertEqual(len(shared), 4)
            self.assertTrue(all(v["params"]["n_init"] == 1 and v["params"]["init"] == "shared_initial_centers.npy" for v in shared))
            shared_init = np.load(root / "result" / "shared_initial_centers.npy")
            from sklearn.preprocessing import normalize
            normalized = normalize(x.astype(np.float32), norm="l2", axis=1).astype(np.float32)
            np.testing.assert_array_equal(shared_init, normalized[out["shared_init"]["original_vector_rows"]])
            self.assertEqual(sum(v["kind"] == "precision" for v in out["comparisons"].values()), 4)
            order = np.load(root / "result" / "order_all_permuted.npy")
            np.testing.assert_array_equal(order, np.random.RandomState(cfg.order_seed).permutation(len(x)))
            np.testing.assert_array_equal(np.load(root / "result" / "order_train_permuted.npy"), order[order < 30])
            self.assertEqual(out["fit_order_files"]["all_permuted"]["sha256"], sha256(root / "result" / "order_all_permuted.npy"))
            with self.assertRaises(FileExistsError):
                run(cfg)
            with self.assertRaises(PermissionError):
                run(replace(cfg, allow_diagnostic_fit=False, output_dir=str(root / "second")))


if __name__ == "__main__":
    unittest.main()
