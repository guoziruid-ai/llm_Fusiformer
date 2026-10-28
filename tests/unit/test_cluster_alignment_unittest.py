"""Regression checks for explicitly identified array rows."""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.cluster_specialists.alignment import IDENTITY_FIELDS, load_aligned_array


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rows = [
            {"sample_id": "mp-1#row0", "original_index": 17, "mp_id": "mp-1",
             "physical_id": "mp-1", "split": "validation", "cluster_id": "A"},
            {"sample_id": "mp-1#row1", "original_index": 32, "mp_id": "mp-1",
             "physical_id": "mp-1", "split": "validation", "cluster_id": "A"},
            {"sample_id": "mp-2", "original_index": 99, "mp_id": "mp-2",
             "physical_id": "mp-2", "split": "test", "cluster_id": "C"},
        ]

    def load(self, sidecar, values, selected=None):
        array = self.root / "values.npy"
        index = self.root / "index.csv"
        np.save(array, values)
        with index.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=IDENTITY_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(sidecar)
        return load_aligned_array(array, index, self.rows, self.rows[:2] if selected is None else selected)

    def test_shuffle_sparse_indices_and_duplicate_mp_ids(self):
        aligned = self.load([self.rows[2], self.rows[0], self.rows[1]], [99, 17, 32])
        np.testing.assert_array_equal(aligned, [17, 32])
        aligned = self.load([self.rows[1], self.rows[0]], [32, 17])
        np.testing.assert_array_equal(aligned, [17, 32])

    def test_missing_duplicate_extra_and_count_mismatch_fail(self):
        unknown = dict(self.rows[2], sample_id="unknown")
        for index, (sidecar, values, message) in enumerate([
            ([self.rows[0]], [17], "missing"),
            ([self.rows[0], self.rows[0]], [17, 17], "duplicate"),
            ([self.rows[0], self.rows[1], unknown], [17, 32, 99], "unknown"),
            (self.rows, [17, 32], "counts"),
            ([self.rows[0], self.rows[1], self.rows[2]], [17, 32, 99, 0], "counts"),
        ]):
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, message):
                self.load(sidecar, values)

    def test_identity_and_split_mismatches_fail(self):
        for field, value in [("split", "test"), ("mp_id", "wrong"),
                             ("original_index", 32), ("physical_id", "wrong")]:
            bad = dict(self.rows[0], **{field: value})
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                self.load([bad, self.rows[1]], [17, 32])
        # A validation-only array cannot supply a test sample.
        with self.assertRaisesRegex(ValueError, "missing"):
            self.load(self.rows[:2], [17, 32], selected=[self.rows[2]])
        # Known but out-of-scope IDs cannot be added as a mixed-split subset.
        with self.assertRaisesRegex(ValueError, "extra"):
            self.load([self.rows[0], self.rows[2]], [17, 99], selected=[self.rows[0]])

    def test_invalid_master_identity_fails_before_alignment(self):
        for field, value, message in [
            ("sample_id", " ", "empty sample_id"),
            ("sample_id", None, "empty sample_id"),
            ("physical_id", "", "empty physical_id"),
            ("physical_id", None, "empty physical_id"),
            ("original_index", -1, "non-negative integer"),
            ("original_index", 1.5, "non-negative integer"),
            ("original_index", None, "non-negative integer"),
        ]:
            original = self.rows[0][field]
            self.rows[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, message):
                self.load(self.rows, [17, 32, 99])
            self.rows[0][field] = original


if __name__ == "__main__":
    unittest.main()
