"""Synthetic interface tests; these do not validate historical data provenance."""
import csv
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from scripts.experiments.cluster_specialists.checkpoints import checkpoint_payload, manifest_fingerprint
from scripts.experiments.cluster_specialists.config import HistoricalMultiSourceConfig, HistoricalSourceConfig
from scripts.experiments.cluster_specialists.models import TensorMLPExpert
from scripts.train import train_cluster_specialist as train
from scripts.evaluate import evaluate_cluster_specialists as evaluate


class HistoricalEntryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rows = []
        for cluster in "ABC":
            for split in ("train", "validation", "test"):
                i = len(self.rows)
                self.rows.append(dict(sample_id=f"id{i}", physical_id=f"id{i}",
                    mp_id=f"id{i}", original_index=str(10 + i * 3),
                    cluster_id=cluster, split=split))
        self.master = self.root / "master.csv"
        self.labels = self.root / "labels.csv"
        self.write_csv(self.master, self.rows)
        self.write_csv(self.labels, [dict(sample_id=r["sample_id"], raw=r["cluster_id"])
                                     for r in reversed(self.rows)])
        self.source = HistoricalSourceConfig(master_path=str(self.master),
            labels_path=str(self.labels), label_col="raw",
            label_mapping=(("A", "A"), ("B", "B"), ("C", "C")))
        self.index = self.root / "index.csv"
        self.write_csv(self.index, [{k: r[k] for k in
            ("sample_id", "physical_id", "mp_id", "original_index", "split")}
            for r in self.rows])
        self.features = self.root / "features.npy"
        self.targets = self.root / "targets.npy"
        self.baseline = self.root / "baseline.npy"
        np.save(self.features, np.arange(18, dtype=np.float32).reshape(9, 2))
        np.save(self.targets, np.arange(9, dtype=np.float32))
        np.save(self.baseline, np.arange(9, dtype=np.float32) / 2)

    @staticmethod
    def write_csv(path, rows):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def training_config(self):
        return replace(train.CFG, historical_source=self.source,
            manifest_path=str(self.master), feature_path=str(self.features),
            feature_index_path=str(self.index), target_path=str(self.targets),
            target_index_path=str(self.index), input_dim=2, hidden_dims=(3,),
            output_checkpoint=str(self.root / "A" / "best.pt"), device="cpu")

    def evaluation_config(self, mode="direct"):
        checkpoints = []
        for cluster in "ABC":
            path = self.root / cluster / "best.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            model = TensorMLPExpert(2, (3,), 0.0)
            torch.save(checkpoint_payload(model, cluster_id=cluster,
                prediction_mode=mode, architecture="tensor_mlp",
                model_config={"input_dim": 2, "hidden_dims": (3,), "dropout": 0.0},
                training_config={"manifest_sha256": manifest_fingerprint(self.source.load())}), path)
            checkpoints.append((cluster, str(path)))
        return replace(evaluate.CFG, historical_source=self.source, prediction_mode=mode,
            manifest_path=str(self.master), specialist_features_path=str(self.features),
            specialist_features_index_path=str(self.index), targets_path=str(self.targets),
            targets_index_path=str(self.index), baseline_predictions_path=str(self.baseline),
            baseline_predictions_index_path=str(self.index), checkpoints=tuple(checkpoints),
            input_dims=tuple((c, 2) for c in "ABC"),
            hidden_dims=tuple((c, (3,)) for c in "ABC"), device="cpu",
            output_dir=str(self.root / "out"))

    def test_training_config_and_checkpoint_fingerprint(self):
        cfg = self.training_config()
        train.run(cfg, check_config=True)
        with patch.object(train, "train_tensor_expert", return_value=SimpleNamespace(
                best_epoch=1, best_validation_loss=0.1)) as fit:
            train.run(cfg)
        self.assertEqual(len(fit.call_args.args[1].dataset), 1)
        self.assertEqual(len(fit.call_args.args[2].dataset), 1)
        payload = torch.load(cfg.output_checkpoint, map_location="cpu", weights_only=False)
        self.assertEqual(payload["training_config"]["manifest_sha256"],
                         manifest_fingerprint(self.source.load()))
        with self.assertRaises(FileExistsError):
            train.run(cfg)

    def test_system_and_single_expert_use_historical_labels_without_router(self):
        cfg = self.evaluation_config()
        with patch.object(evaluate, "load_router", side_effect=AssertionError("router used")):
            evaluate.run(cfg, check_config=True)
            evaluate.run(cfg)
            evaluate.run(replace(cfg, evaluation_mode="A"))
        with (self.root / "out" / "system_test_direct_paired_predictions.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual({(r["sample_id"], r["cluster_id"], r["original_index"])
                          for r in rows}, {("id2", "A", "16"), ("id5", "B", "25"), ("id8", "C", "34")})
        with self.assertRaises(FileExistsError):
            evaluate.run(cfg)

    def test_mismatched_historical_assignment_fails(self):
        cfg = self.training_config()
        altered = [dict(row) for row in self.rows]
        altered[0]["cluster_id"] = "B"
        self.write_csv(self.master, altered)
        with self.assertRaisesRegex(ValueError, "cluster_id mismatch"):
            train.run(cfg, check_config=True)

    def test_different_manifest_split_fails_before_evaluation(self):
        cfg = self.evaluation_config()
        altered = [dict(row) for row in self.rows]
        altered[0]["split"] = "test"
        other = self.root / "other_manifest.csv"
        self.write_csv(other, altered)
        with patch.object(evaluate, "load_router", side_effect=AssertionError("router used")):
            with self.assertRaisesRegex(ValueError, "identities or splits differ"):
                evaluate.run(replace(cfg, manifest_path=str(other)), check_config=True)

    def test_missing_history_fails_closed(self):
        cfg = self.training_config()
        self.write_csv(self.labels, [{"sample_id": "id0", "raw": "A"}])
        with self.assertRaisesRegex(ValueError, "ID mismatch"):
            train.run(cfg, check_config=True)

    def test_residual_mode_retains_baseline_addition(self):
        cfg = self.evaluation_config("residual")
        evaluate.run(cfg)
        with (self.root / "out" / "system_test_residual_paired_predictions.csv").open() as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            cluster = row["cluster_id"]
            model = TensorMLPExpert(2, (3,), 0.0)
            payload = torch.load(dict(cfg.checkpoints)[cluster], map_location="cpu", weights_only=False)
            model.load_state_dict(payload["model_state_dict"])
            index = int(row["sample_id"][2:])
            with torch.inference_mode():
                residual = model(torch.tensor([[float(index * 2), float(index * 2 + 1)]])).item()
            self.assertAlmostEqual(float(row["specialist_prediction"]), residual + index / 2, places=5)

    def test_position_sources_are_joined_per_split_without_router(self):
        sources = []
        for split in ("train", "validation", "test"):
            subset = [row for row in self.rows if row["split"] == split]
            master, labels, positions = (self.root / f"{split}_{suffix}.csv"
                                         for suffix in ("master", "labels", "positions"))
            self.write_csv(master, subset)
            self.write_csv(labels, [{"position": str(i), "raw": row["cluster_id"]}
                                    for i, row in reversed(list(enumerate(subset)))])
            self.write_csv(positions, [{"position": str(i), "sample_id": row["sample_id"],
                                        "split": split} for i, row in enumerate(subset)])
            sources.append(HistoricalSourceConfig(master_path=str(master), labels_path=str(labels),
                label_col="raw", label_mapping=(("A", "A"), ("B", "B"), ("C", "C")),
                position_map_path=str(positions), position_label_col="position",
                position_map_position_col="position"))
        multi = HistoricalMultiSourceConfig(sources=tuple(sources))
        cfg = replace(self.evaluation_config(), historical_source=multi)
        with patch.object(evaluate, "load_router", side_effect=AssertionError("router used")):
            evaluate.run(cfg, check_config=True)
            evaluate.run(cfg)
        with (self.root / "out" / "system_test_direct_paired_predictions.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual({(r["sample_id"], r["cluster_id"]) for r in rows},
                         {("id2", "A"), ("id5", "B"), ("id8", "C")})


if __name__ == "__main__":
    unittest.main()
