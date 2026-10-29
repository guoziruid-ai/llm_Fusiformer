import tempfile
import unittest
from pathlib import Path

from scripts.experiments.cluster_specialists.config import SplitRatios
from scripts.experiments.cluster_specialists.manifests import (
    build_manifest, read_manifest, validate_manifest, write_manifests)


class ClusterManifestTests(unittest.TestCase):
    def _records(self):
        records = []
        clusters = []
        for cluster_index, cluster in enumerate("ABC"):
            for index in range(12):
                records.append({
                    "mp_id": f"mp-{cluster}-{index}",
                    "material_id": (
                        f"physical-{cluster}-pair"
                        if index in {0, 1} else f"physical-{cluster}-{index}"),
                    "e_form": cluster_index + index / 100.0})
                clusters.append(cluster)
        return records, clusters

    def test_nine_splits_are_reproducible_and_physical_ids_do_not_leak(self):
        records, clusters = self._records()
        kwargs = {
            "ratios": SplitRatios(0.6, 0.2, 0.2), "seed": 123,
            "physical_id_fields": ("material_id",),
            "target_field": "e_form"}
        first = build_manifest(records, clusters, **kwargs)
        second = build_manifest(records, clusters, **kwargs)
        self.assertEqual(first, second)
        report = validate_manifest(first, expected_count=36)
        self.assertTrue(all(
            count > 0 for per_cluster in report["dataset_counts"].values()
            for count in per_cluster.values()))
        shared = [
            row for row in first
            if row["physical_id"] == "physical-A-pair"]
        self.assertEqual(len({row["split"] for row in shared}), 1)
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_manifests(
                first, tmp, seed=123, ratios=kwargs["ratios"],
                source="synthetic")
            self.assertEqual(len(paths), 11)
            self.assertEqual(len(read_manifest(paths["master"])), 36)

    def test_duplicate_physical_records_stay_together_and_leakage_fails(self):
        records, clusters = self._records()
        records[1]["mp_id"] = records[0]["mp_id"]
        grouped = build_manifest(records, clusters)
        repeated = [row for row in grouped if row["mp_id"] == records[0]["mp_id"]]
        self.assertEqual(len(repeated), 2)
        self.assertEqual(len({row["sample_id"] for row in repeated}), 2)
        self.assertEqual(len({row["split"] for row in repeated}), 1)
        rows = [
            {"sample_id": "a", "physical_id": "same", "mp_id": "a",
             "original_index": 0, "cluster_id": "A", "split": "train"},
            {"sample_id": "b", "physical_id": "same", "mp_id": "b",
             "original_index": 1, "cluster_id": "A", "split": "test"}]
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_manifest(rows)


if __name__ == "__main__":
    unittest.main()
