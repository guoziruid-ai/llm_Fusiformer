import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.cluster_specialists.evaluation import (
    summarize_paired, write_evaluation)
from experiments.cluster_specialists.features import extract_v2_graph_latents
from experiments.cluster_specialists.models import (
    RoutedSpecialistSystem, SpecialistAdapter, TensorMLPExpert)
from experiments.cluster_specialists.routing import FixedBoundaryRouter
from experiments.cluster_specialists.training import train_tensor_expert


class ConstantExpert(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(float(value)))

    def forward(self, inputs):
        return self.bias.expand(len(inputs), 1)


class FakeV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.readout = nn.Linear(2, 3, bias=False)
        self.head = nn.Linear(3, 1, bias=False)

    def forward(self, inputs):
        return self.head(self.readout(inputs["value"]))


class ClusterSpecialistPipelineTests(unittest.TestCase):
    def test_full_routing_preserves_order_and_exports_paired_results(self):
        router = FixedBoundaryRouter(
            -0.5, 0.5, feature_name="router_feature")
        system = RoutedSpecialistSystem(
            router, {cluster: ConstantExpert(value)
                     for cluster, value in zip("ABC", (1, 2, 3))})
        features = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        output, decisions = system(
            features, np.asarray([[-1.0], [1.0], [0.0], [-2.0]]))
        self.assertEqual(output.reshape(-1).tolist(), [1.0, 3.0, 2.0, 1.0])
        clusters = [item.cluster_id for item in decisions]
        result = summarize_paired(
            sample_ids=["s0", "s1", "s2", "s3"],
            original_indices=[4, 2, 9, 1], cluster_ids=clusters,
            targets=[1, 3, 2, 1], baseline_predictions=[2, 2, 2, 2],
            specialist_predictions=output.detach().reshape(-1).numpy())
        self.assertEqual(result["summary"]["coverage"], 1.0)
        self.assertEqual(result["summary"]["overall"]["candidate"]["mae"], 0.0)
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_evaluation(result, tmp, prefix="system_test")
            self.assertTrue(Path(paths["metrics"]).is_file())
            self.assertTrue(Path(paths["predictions"]).is_file())

    def test_direct_and_residual_interfaces_and_one_expert_training(self):
        direct = SpecialistAdapter(ConstantExpert(2), mode="direct")
        baseline = ConstantExpert(3)
        residual = SpecialistAdapter(
            ConstantExpert(2), mode="residual", baseline=baseline)
        inputs = torch.ones(3, 2)
        self.assertEqual(direct(inputs).reshape(-1).tolist(), [2.0] * 3)
        self.assertEqual(residual(inputs).reshape(-1).tolist(), [5.0] * 3)
        self.assertFalse(any(
            parameter.requires_grad for parameter in baseline.parameters()))

        x = torch.linspace(-1, 1, 24).reshape(12, 2)
        y = x.sum(1, keepdim=True)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y),
            batch_size=4, shuffle=False)
        model = TensorMLPExpert(2, (4,))
        result = train_tensor_expert(
            model, loader, loader, epochs=2, learning_rate=1e-2)
        self.assertEqual(len(result.history), 2)
        self.assertGreaterEqual(result.best_epoch, 1)

    def test_v2_feature_adapter_is_read_only_and_aligned(self):
        model = FakeV2().train()
        loader = [
            ({"value": torch.ones(2, 2)}, torch.zeros(2, 1)),
            ({"value": torch.full((1, 2), 2.0)}, torch.zeros(1, 1))]
        latents, predictions = extract_v2_graph_latents(
            model, loader, device="cpu")
        self.assertEqual(latents.shape, (3, 3))
        self.assertEqual(predictions.shape, (3,))
        self.assertFalse(model.training)
        self.assertTrue(all(
            parameter.grad is None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
