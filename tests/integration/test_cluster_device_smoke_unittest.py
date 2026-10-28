"""Small synthetic CPU/CUDA routed specialist forward/backward smoke tests."""
import unittest

import numpy as np
import torch

from experiments.cluster_specialists.models import (
    RoutedSpecialistSystem, SpecialistAdapter, TensorMLPExpert)
from experiments.cluster_specialists.routing import FixedBoundaryRouter


class DeviceSmokeTests(unittest.TestCase):
    def _exercise(self, device_name):
        device = torch.device(device_name)
        for mode in ("direct", "residual"):
            with self.subTest(device=device_name, mode=mode):
                torch.manual_seed(13)
                experts = {}
                for cluster in "ABC":
                    baseline = torch.nn.Linear(4, 1) if mode == "residual" else None
                    experts[cluster] = SpecialistAdapter(
                        TensorMLPExpert(4, (8,), 0.0), mode=mode, baseline=baseline)
                router = FixedBoundaryRouter(-0.5, 0.5, feature_name="synthetic")
                system = RoutedSpecialistSystem(router, experts).to(device).train()
                features = torch.randn(9, 4, device=device)
                targets = torch.randn(9, 1, device=device)
                routing = np.array([1., -1., 0., -1., 1., 0., 0., 1., -1.])
                optimizer = torch.optim.AdamW(
                    (p for p in system.parameters() if p.requires_grad), lr=0.01)
                before = {name: p.detach().clone() for name, p in system.named_parameters()}
                optimizer.zero_grad(set_to_none=True)
                predictions, decisions = system(features, routing)
                self.assertEqual(predictions.shape, (9, 1))
                self.assertEqual(predictions.device.type, device.type)
                self.assertEqual([d.cluster_id for d in decisions],
                                 ["C", "A", "B", "A", "C", "B", "B", "C", "A"])
                self.assertTrue(torch.isfinite(predictions).all().item())
                for cluster in "ABC":
                    positions = [i for i, d in enumerate(decisions) if d.cluster_id == cluster]
                    torch.testing.assert_close(predictions[positions], experts[cluster](features[positions]))
                loss = torch.nn.functional.mse_loss(predictions, targets)
                self.assertTrue(torch.isfinite(loss).item())
                loss.backward()
                for name, parameter in system.named_parameters():
                    if parameter.requires_grad:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertEqual(parameter.grad.device.type, device.type)
                        self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
                    else:
                        self.assertIsNone(parameter.grad, name)
                optimizer.step()
                for cluster in "ABC":
                    self.assertTrue(any(
                        not torch.equal(parameter, before[name])
                        for name, parameter in system.named_parameters()
                        if name.startswith(f"experts.{cluster}.specialist.")))
                for name, parameter in system.named_parameters():
                    self.assertTrue(torch.isfinite(parameter).all().item(), name)
                    if not parameter.requires_grad:
                        self.assertTrue(torch.equal(parameter, before[name]), name)

    def test_cpu_direct_and_residual(self):
        self._exercise("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_direct_and_residual(self):
        self._exercise("cuda")


if __name__ == "__main__":
    unittest.main()
