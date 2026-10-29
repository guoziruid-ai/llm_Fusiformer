import json
import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.experiments.cluster_specialists.routing import (
    CentroidRouter, FixedBoundaryRouter, load_router)


class ClusterRoutingTests(unittest.TestCase):
    def test_fixed_boundaries_mapping_diagnostics_and_round_trip(self):
        router = FixedBoundaryRouter(
            -1.0, 1.0, feature_name="baseline_prediction",
            observed_range=(-2.0, 2.0))
        values = np.asarray([[-3.0], [-1.0], [0.0], [1.0], [2.0]])
        self.assertEqual(
            router.predict(values).tolist(), ["A", "A", "B", "B", "C"])
        self.assertTrue(router.predict_diagnostics(values)[0].out_of_domain)
        with tempfile.TemporaryDirectory() as tmp:
            path = router.save(Path(tmp) / "router.json")
            loaded = load_router(path)
            self.assertEqual(
                loaded.predict(values).tolist(),
                ["A", "A", "B", "B", "C"])
            with self.assertRaises(FileExistsError):
                router.save(path)

    def test_centroid_fit_is_train_only_serializable_and_reproducible(self):
        features = np.asarray([
            [-4.0, -3.8], [-3.9, -4.1], [-4.2, -4.0],
            [0.0, 0.1], [0.1, -0.1], [-0.2, 0.0],
            [4.0, 4.2], [3.8, 4.0], [4.1, 3.9]])
        router = CentroidRouter(
            feature_names=("latent_0", "latent_1"), random_seed=7)
        router.fit(features)
        labels = router.predict(features)
        self.assertEqual(set(labels.tolist()), {"A", "B", "C"})
        with tempfile.TemporaryDirectory() as tmp:
            path = router.save(Path(tmp) / "router.json")
            loaded = load_router(path)
            np.testing.assert_array_equal(labels, loaded.predict(features))
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["fit_scope"], "training_only")
            self.assertEqual(
                set(payload["cluster_mapping"].values()), {"A", "B", "C"})

    def test_fixed_boundary_rejects_mapping_that_execution_cannot_honor(self):
        router = FixedBoundaryRouter(-0.5, 0.5, feature_name="x")
        with tempfile.TemporaryDirectory() as tmp:
            path = router.save(Path(tmp) / "router.json")
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["cluster_mapping"] = {"0": "C", "1": "B", "2": "A"}
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fixed boundary cluster_mapping"):
                load_router(path)

    def test_fixed_boundary_distance_and_margin_current_behavior(self):
        # Characterization only: this does not endorse margin as a measure of
        # boundary proximity. Its scientific interpretation remains unresolved.
        router = FixedBoundaryRouter(-0.5, 0.5, feature_name="x")
        decisions = router.predict_diagnostics([[-0.5], [0.0], [0.5]])
        np.testing.assert_allclose([item.distance for item in decisions], [0, 0.5, 0])
        np.testing.assert_allclose([item.margin for item in decisions], [1, 0, 1])

    def test_invalid_or_missing_features_are_rejected(self):
        router = FixedBoundaryRouter(0.0, 1.0, feature_name="x")
        with self.assertRaisesRegex(ValueError, "NaN"):
            router.predict([[float("nan")]])
        with self.assertRaisesRegex(ValueError, "dimension"):
            router.predict([[1.0, 2.0]])
        with self.assertRaises(FileNotFoundError):
            load_router("definitely_missing_router.json")

    def test_router_contract_has_no_label_or_error_input(self):
        fit_parameters = set(inspect.signature(CentroidRouter.fit).parameters)
        predict_parameters = set(
            inspect.signature(CentroidRouter.predict).parameters)
        forbidden = {"target", "targets", "label", "labels", "error", "errors"}
        self.assertFalse(fit_parameters & forbidden)
        self.assertFalse(predict_parameters & forbidden)


if __name__ == "__main__":
    unittest.main()
