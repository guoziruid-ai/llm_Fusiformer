"""End-to-end checkpoint-only entry smoke on temporary synthetic data."""
import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from scripts.experiments.cluster_specialists.checkpoints import checkpoint_payload, manifest_fingerprint, save_checkpoint
from scripts.experiments.cluster_specialists.config import SplitRatios
from scripts.experiments.cluster_specialists.manifests import build_manifest, write_manifests
from scripts.experiments.cluster_specialists.models import TensorMLPExpert
from scripts.experiments.cluster_specialists.routing import FixedBoundaryRouter
from scripts.evaluate import evaluate_cluster_specialists as entry


class EvaluationEntrySmokeTests(unittest.TestCase):
    def test_system_loads_checkpoints_and_exports_aligned_validation_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [{"mp_id": f"sample-{i}"} for i in range(30)]
            router_features = np.array([-1., 0., 1.] * 10, dtype=np.float32)
            router = FixedBoundaryRouter(-0.5, 0.5, feature_name="synthetic")
            router_path = router.save(root / "router.json")
            ratios = SplitRatios(0.6, 0.2, 0.2)
            rows = build_manifest(records, router.predict(router_features),
                                  ratios=ratios, seed=42)
            for row in rows:
                row["original_index"] = row["original_index"] * 7 + 11
            manifest = write_manifests(rows, root / "manifest", seed=42,
                                       ratios=ratios, source="synthetic")["master"]
            arrays = {
                "router": router_features,
                "features": np.arange(60, dtype=np.float32).reshape(30, 2),
                "targets": np.arange(30, dtype=np.float32),
                "baseline": np.zeros(30, dtype=np.float32)}
            paths = {}
            index_paths = {}
            for key, value in arrays.items():
                paths[key] = root / f"{key}.npy"
                order = np.random.default_rng(len(key)).permutation(len(rows))
                np.save(paths[key], value[order])
                index_paths[key] = root / f"{key}_index.csv"
                with index_paths[key].open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows[i] for i in order)
            checkpoints = []
            for cluster in "ABC":
                model = TensorMLPExpert(2, (4,), 0.0)
                payload = checkpoint_payload(
                    model, cluster_id=cluster, prediction_mode="direct",
                    architecture="tensor_mlp",
                    model_config={"input_dim": 2, "hidden_dims": (4,)},
                    training_config={"manifest_sha256": manifest_fingerprint(rows)})
                checkpoint = save_checkpoint(root / f"{cluster}.pt", payload)
                checkpoints.append((cluster, str(checkpoint)))
            cfg = replace(
                entry.CFG, evaluation_mode="system", prediction_mode="direct",
                manifest_path=manifest, router_path=str(router_path),
                router_features_path=str(paths["router"]),
                specialist_features_path=str(paths["features"]),
                targets_path=str(paths["targets"]),
                baseline_predictions_path=str(paths["baseline"]),
                router_features_index_path=str(index_paths["router"]),
                specialist_features_index_path=str(index_paths["features"]),
                targets_index_path=str(index_paths["targets"]),
                baseline_predictions_index_path=str(index_paths["baseline"]),
                checkpoints=tuple(checkpoints),
                input_dims=tuple((c, 2) for c in "ABC"),
                hidden_dims=tuple((c, (4,)) for c in "ABC"),
                dropouts=tuple((c, 0.0) for c in "ABC"), split="validation",
                device="cpu", output_dir=str(root / "output"), bootstrap_samples=100)
            with patch.object(entry, "CFG", cfg), patch("sys.argv", ["evaluation"]), \
                    patch("torch.optim.AdamW", side_effect=AssertionError("optimizer created")):
                entry.main()
            output = root / "output" / "system_validation_direct_paired_predictions.csv"
            with output.open(encoding="utf-8", newline="") as handle:
                actual = list(csv.DictReader(handle))
            expected = sorted((row for row in rows if row["split"] == "validation"),
                              key=lambda row: row["sample_id"])
            self.assertEqual([int(row["original_index"]) for row in actual],
                             [row["original_index"] for row in expected])
            self.assertEqual([row["sample_id"] for row in actual],
                             [row["sample_id"] for row in expected])
            for output_row, source_row in zip(actual, expected):
                self.assertEqual(output_row["mp_id"], source_row["mp_id"])
                self.assertEqual(output_row["split"], "validation")
                self.assertEqual(float(output_row["target"]),
                                 float(records.index({"mp_id": source_row["mp_id"]})))
            self.assertEqual({row["cluster_id"] for row in actual}, {"A", "B", "C"})
            self.assertTrue(all(
                np.isfinite(float(row["specialist_prediction"]))
                and np.isfinite(float(row["baseline_prediction"]))
                for row in actual))
            self.assertTrue(
                (root / "output" / "system_validation_direct_metrics.json").is_file())
            # Reordering only the manifest must not change paired predictions or metrics.
            with Path(manifest).open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(reversed(rows))
            cfg2 = replace(cfg, output_dir=str(root / "shuffled"))
            with patch.object(entry, "CFG", cfg2), patch("sys.argv", ["evaluation"]):
                entry.main()
            shuffled = root / "shuffled" / output.name
            with shuffled.open(encoding="utf-8", newline="") as handle:
                shuffled_rows = list(csv.DictReader(handle))
            self.assertEqual({row["sample_id"]: row for row in actual},
                             {row["sample_id"]: row for row in shuffled_rows})
            with (root / "output" / "system_validation_direct_metrics.json").open() as f:
                first_metrics = json.load(f)
            with (root / "shuffled" / "system_validation_direct_metrics.json").open() as f:
                second_metrics = json.load(f)
            self.assertEqual(first_metrics, second_metrics)

            # Different split assignment must not be evaluated with these weights.
            changed_rows = [dict(row) for row in rows]
            changed_rows[0]["split"] = "test" if changed_rows[0]["split"] != "test" else "train"
            self.assertNotEqual(manifest_fingerprint(changed_rows), manifest_fingerprint(rows))
            from scripts.experiments.cluster_specialists.checkpoints import load_checkpoint
            with self.assertRaisesRegex(ValueError, "manifest fingerprint"):
                load_checkpoint(checkpoints[0][1], TensorMLPExpert(2, (4,)),
                                expected_cluster="A", expected_mode="direct",
                                expected_architecture="tensor_mlp",
                                expected_manifest_fingerprint=manifest_fingerprint(changed_rows))
            legacy_path = save_checkpoint(root / "legacy.pt", checkpoint_payload(
                TensorMLPExpert(2, (4,)), cluster_id="A", prediction_mode="direct",
                architecture="tensor_mlp", model_config={"input_dim": 2}))
            with self.assertRaisesRegex(ValueError, "manifest fingerprint"):
                load_checkpoint(legacy_path, TensorMLPExpert(2, (4,)),
                                expected_cluster="A", expected_mode="direct",
                                expected_architecture="tensor_mlp",
                                expected_manifest_fingerprint=manifest_fingerprint(rows))


if __name__ == "__main__":
    unittest.main()
