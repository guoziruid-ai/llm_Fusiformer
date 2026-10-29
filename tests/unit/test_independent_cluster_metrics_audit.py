import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.diagnostics.independent_cluster_metrics_audit import (
    AuditConfig, independent_compare, read_historical, read_identity, run,
)


class IndependentClusterMetricsTests(unittest.TestCase):
    def test_disabled_by_default(self):
        with self.assertRaises(PermissionError):
            run(AuditConfig())

    def test_reordered_records_associate_by_explicit_index_and_id(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            identity = directory / "identity.csv"
            historical = directory / "historical.csv"
            with identity.open("w", newline="", encoding="utf-8") as out:
                writer = csv.writer(out)
                writer.writerow(["sample_id", "original_index", "split", "vector_row"])
                writer.writerows([["mp-2", 1, "validation", 1], ["mp-1", 0, "train", 0]])
            with historical.open("w", newline="", encoding="utf-8") as out:
                writer = csv.writer(out)
                writer.writerow(["index", "mp_id", "fusiformer_cluster"])
                writer.writerows([[1, "mp-2", 1], [0, "mp-1", 0]])
            rows = read_identity(identity)
            np.testing.assert_array_equal(read_historical(historical, rows, 2), [0, 1])
            with historical.open("a", newline="", encoding="utf-8") as out:
                csv.writer(out).writerow([0, "mp-1", 0])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_historical(historical, rows, 2)

    def test_train_only_matching_and_omitted(self):
        rows = [(i, f"id-{i}", split) for i, split in enumerate(
            ["train", "train", "train", "train", "validation", "test", "omitted"])]
        reference = np.array([0, 0, 1, 1, 0, 1, 0])
        candidate = np.array([1, 1, 0, 0, 0, 1, 1])
        result, aligned = independent_compare(reference, candidate, rows, 2)
        self.assertEqual(result["candidate_to_reference"], [1, 0])
        self.assertEqual(result["splits"]["train"]["matched_agreement"], 1)
        self.assertEqual(result["splits"]["validation"]["matched_agreement"], 0)
        self.assertEqual(result["splits"]["test"]["matched_agreement"], 0)
        self.assertEqual(result["splits"]["test"]["migration"], [[0, 0], [1, 0]])
        candidate[-1] = 0
        second, _ = independent_compare(reference, candidate, rows, 2)
        self.assertEqual(result, second)
        self.assertEqual(len(aligned), len(rows))


if __name__ == "__main__":
    unittest.main()
