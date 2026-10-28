"""Diagnostics must use the same identity join as specialist evaluation."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.cluster_specialists.manifests import write_manifests
from experiments.cluster_specialists.config import SplitRatios
from experiments.cluster_specialists.routing import FixedBoundaryRouter
from scripts.diagnostics import analyze_cluster_specialists as diagnostic


class DiagnosticAlignmentTests(unittest.TestCase):
    def test_reordered_manifest_and_independently_shuffled_arrays(self):
        rows = [
            {"sample_id": f"sample-{i}", "physical_id": f"p-{i}",
             "mp_id": f"mp-{i}", "original_index": idx, "cluster_id": cluster,
             "split": split}
            for i, (idx, cluster, split) in enumerate([
                (17, "A", "train"), (93, "B", "train"), (41, "C", "train"),
                (82, "A", "validation"), (22, "B", "validation"),
                (140, "C", "validation"), (5, "A", "test"),
                (64, "B", "test"), (290, "C", "test")])]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = write_manifests(list(reversed(rows)), root / "manifests",
                                    seed=42, ratios=SplitRatios(), source="fixture")
            router = FixedBoundaryRouter(-0.5, 0.5, feature_name="synthetic")
            router_path = router.save(root / "router.json")
            arrays = {
                "features": np.array([-1, 0, 1] * 3, dtype=float),
                "targets": np.arange(9, dtype=float),
                "baseline": np.arange(9, dtype=float) + 1}
            arrays_paths, index_paths = {}, {}
            for offset, (name, values) in enumerate(arrays.items()):
                order = np.roll(np.arange(len(rows)), offset + 2)
                arrays_paths[name] = root / f"{name}.npy"
                index_paths[name] = root / f"{name}.csv"
                np.save(arrays_paths[name], values[order])
                with index_paths[name].open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows[index] for index in order)
            output = root / "report.json"
            config = {
                "MANIFEST_PATH": Path(paths["master"]), "ROUTER_PATH": router_path,
                "ROUTER_FEATURES_PATH": arrays_paths["features"],
                "ROUTER_FEATURES_INDEX_PATH": index_paths["features"],
                "TARGETS_PATH": arrays_paths["targets"],
                "TARGETS_INDEX_PATH": index_paths["targets"],
                "BASELINE_PREDICTIONS_PATH": arrays_paths["baseline"],
                "BASELINE_PREDICTIONS_INDEX_PATH": index_paths["baseline"],
                "SPLIT": "test", "OUTPUT_PATH": output}
            with patch.multiple(diagnostic, **config):
                diagnostic.main()
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["assignment_reproducible"])
            self.assertEqual(report["sample_count"], 3)
            for cluster in "ABC":
                self.assertEqual(report["clusters"][cluster]["baseline_metrics"]["mae"], 1.0)


if __name__ == "__main__":
    unittest.main()
