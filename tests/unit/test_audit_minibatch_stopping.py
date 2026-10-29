import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import numpy as np
    from scripts.diagnostics.audit_minibatch_stopping import AuditConfig, _sse, run
    from scripts.diagnostics.compare_latent_fit_scope import Config
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False


@unittest.skipUnless(HAS_DEPENDENCIES, "NumPy and diagnostic dependencies required")
class StoppingAuditTests(unittest.TestCase):
    def test_common_sse(self):
        x = np.array([[0, 0], [2, 0], [0, 4]], dtype=np.float32)
        centers = np.array([[0, 0], [2, 1]], dtype=np.float32)
        self.assertEqual(_sse(x, centers), 14.0)

    def test_pilot_excludes_test_and_omitted_and_refuses_overwrite(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn unavailable")
        rng = np.random.RandomState(1)
        x = rng.normal(size=(35, 4)).astype(np.float32)
        splits = ["train"] * 25 + ["validation"] * 5 + ["test"] * 4 + ["omitted"]
        rows = [dict(vector_row=i, original_index=100+i, sample_id=f"id{i}", split=s)
                for i, s in enumerate(splits)]
        evidence = {"checkpoint_path": "/nonexistent/checkpoint", "source_dataset_path": "/nonexistent/data",
                    "artifacts": {"full_labels": {"path": "/nonexistent/labels"}}}
        cfg = Config(latent_path="/nonexistent/x", identity_path="/nonexistent/id",
                     evidence_path="/nonexistent/evidence", k=5)
        with tempfile.TemporaryDirectory() as tmp:
            acfg = AuditConfig(provenance=cfg, output_root=tmp, allow_diagnostic_fit=True,
                               batches=(5, 6), bounded_epochs=1)
            with patch("scripts.diagnostics.audit_minibatch_stopping.preflight",
                       return_value=(x, rows, evidence, {"test": "digest"})), \
                 patch("scripts.diagnostics.audit_minibatch_stopping.load_full_labels",
                       return_value=np.arange(len(x)) % 5):
                result = run(acfg, "pilot")
                self.assertEqual(set(result["comparisons"]["historical_full__vs__historical_parameters_all"]["splits"]),
                                 {"train", "validation"})
                with (Path(tmp) / "pilot" / "sample_labels.csv").open(newline="", encoding="utf-8") as handle:
                    output_rows = list(csv.DictReader(handle))
                self.assertEqual(len(output_rows), 30)
                self.assertEqual(set(row["split"] for row in output_rows), {"train", "validation"})
                self.assertEqual(result["models"]["historical_parameters_all"]["fit_rows"], 35)
                with self.assertRaises(FileExistsError):
                    run(acfg, "pilot")
                full = run(acfg, "full")
                self.assertEqual(len(full["models"]), 20)
                key = "mini_train_b5_bounded__vs__mini_train_b5_bounded_seed42_permuted"
                self.assertIn(key, full["comparisons"])
                for result in full["comparisons"].values():
                    self.assertEqual(result["splits"]["train"]["count"], 25)
                    self.assertEqual(result["splits"]["validation"]["count"], 5)
                    self.assertEqual(result["matching_scope"], "training only")
                with (Path(tmp) / "full" / "sample_labels.csv").open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    self.assertIn(key + "__changed", reader.fieldnames)
                    self.assertEqual(sum(1 for _ in reader), 30)


if __name__ == "__main__":
    unittest.main()
