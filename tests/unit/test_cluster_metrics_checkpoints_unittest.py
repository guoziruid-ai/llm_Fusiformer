import tempfile
import unittest
import inspect
from pathlib import Path

import numpy as np
import torch

from scripts.experiments.cluster_specialists.checkpoints import (
    checkpoint_payload, load_checkpoint, save_checkpoint)
from scripts.experiments.cluster_specialists.metrics import (
    improvement_summary, paired_bootstrap_mae_improvement,
    regression_metrics)
from scripts.experiments.cluster_specialists.models import TensorMLPExpert
from scripts.evaluate import evaluate_cluster_specialists as evaluation_entry


class ClusterMetricCheckpointTests(unittest.TestCase):
    def test_metrics_and_negative_improvement_are_visible(self):
        targets = [0.0, 1.0, 2.0]
        baseline = [0.0, 1.5, 1.5]
        worse = [1.0, 2.0, 3.0]
        metrics = regression_metrics(targets, baseline)
        self.assertAlmostEqual(metrics["mae"], 1.0 / 3.0)
        summary = improvement_summary(targets, baseline, worse)
        self.assertLess(
            summary["relative_improvement_percent"]["mae"], 0.0)
        ci = paired_bootstrap_mae_improvement(
            targets, worse, baseline, samples=100, seed=3)
        self.assertEqual(ci["unit"], "sample")
        self.assertGreater(ci["mae_improvement"], 0)

    def test_strict_checkpoint_load_and_mismatch(self):
        torch.manual_seed(1)
        model = TensorMLPExpert(3, (4,))
        payload = checkpoint_payload(
            model, cluster_id="A", prediction_mode="direct",
            architecture="tensor_mlp",
            model_config={"input_dim": 3, "hidden_dims": (4,)})
        with tempfile.TemporaryDirectory() as tmp:
            path = save_checkpoint(Path(tmp) / "a.pt", payload)
            restored = TensorMLPExpert(3, (4,))
            loaded = load_checkpoint(
                path, restored, expected_cluster="A",
                expected_mode="direct",
                expected_architecture="tensor_mlp")
            self.assertEqual(loaded["cluster_id"], "A")
            for left, right in zip(model.parameters(), restored.parameters()):
                self.assertTrue(torch.equal(left, right))
            with self.assertRaisesRegex(ValueError, "cluster_id"):
                load_checkpoint(
                    path, restored, expected_cluster="B",
                    expected_mode="direct",
                    expected_architecture="tensor_mlp")
            with self.assertRaises(FileNotFoundError):
                load_checkpoint(
                    Path(tmp) / "missing.pt", restored,
                    expected_cluster="A", expected_mode="direct",
                    expected_architecture="tensor_mlp")

    def test_independent_evaluation_entry_has_no_training_operations(self):
        source = inspect.getsource(evaluation_entry.main)
        self.assertNotIn("torch.optim", source)
        self.assertNotIn(".backward(", source)
        self.assertNotIn(".train(", source)
        self.assertEqual(
            evaluation_entry._required_clusters("A"), ("A",))
        self.assertEqual(
            evaluation_entry._required_clusters("system"),
            ("A", "B", "C"))


if __name__ == "__main__":
    unittest.main()
