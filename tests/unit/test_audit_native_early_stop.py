import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import numpy as np
    from scripts.diagnostics.audit_native_early_stop import NativeStopConfig, run
    from scripts.diagnostics.compare_latent_fit_scope import Config, sha256
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@unittest.skipUnless(HAS_NUMPY, "NumPy required")
class NativeStopTests(unittest.TestCase):
    def test_stop_comparison_keeps_test_unexamined_and_changes_only_controls(self):
        try:
            import joblib
            from sklearn.cluster import MiniBatchKMeans
            from sklearn.preprocessing import normalize
        except ImportError:
            self.skipTest("scikit-learn/joblib unavailable")
        raw = np.random.RandomState(5).normal(size=(35, 4)).astype(np.float32)
        x = normalize(raw).astype(np.float32)
        split = ["train"] * 25 + ["validation"] * 5 + ["test"] * 4 + ["omitted"]
        rows = [dict(vector_row=i, original_index=100+i, sample_id=f"id{i}", split=s)
                for i, s in enumerate(split)]
        model = MiniBatchKMeans(n_clusters=5, init="k-means++", n_init=20, max_iter=500,
                                random_state=42, batch_size=4096, tol=1e-4,
                                max_no_improvement=10, reassignment_ratio=.01, init_size=None,
                                compute_labels=True, verbose=1).fit(x)
        evidence = {"checkpoint_path": "/nonexistent/checkpoint", "source_dataset_path": "/nonexistent/data",
                    "artifacts": {"full_labels": {"path": "/nonexistent/labels"}}}
        hashes = {"latent": "preflight-verified"}
        provenance = Config(latent_path="/nonexistent/x", identity_path="/nonexistent/identity",
                            evidence_path="/nonexistent/evidence", k=5)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_file, summary_file = root / "baseline.joblib", root / "pilot.json"
            joblib.dump(model, baseline_file)
            summary_file.write_text(json.dumps({"stage": "pilot", "input_hashes": hashes,
                "models": {"historical_parameters_all": {"algorithm": "minibatch", "scope": "all",
                "order": "original", "fit_rows": len(raw), "params": model.get_params(deep=False)}}}), encoding="utf-8")
            cfg = NativeStopConfig(provenance, str(baseline_file), sha256(baseline_file),
                                   str(summary_file), sha256(summary_file), str(root / "output"), True)
            with patch("scripts.diagnostics.audit_native_early_stop.preflight", return_value=(raw, rows, evidence, hashes)), \
                 patch("scripts.diagnostics.audit_native_early_stop.load_full_labels", return_value=model.predict(x)):
                result = run(cfg)
                self.assertEqual(set(result["changed_parameters"]), {"tol", "max_iter", "max_no_improvement"})
                self.assertEqual(result["historical_baseline_raw_match_counts"], {"train": 25, "validation": 5})
                for comp in result["comparisons"].values():
                    self.assertEqual(set(comp["splits"]), {"train", "validation"})
                    self.assertEqual(comp["matching_scope"], "training only")
                with (root / "output" / "sample_labels.csv").open(newline="", encoding="utf-8") as f:
                    output = list(csv.DictReader(f))
                self.assertEqual(len(output), 30)
                self.assertEqual(set(r["split"] for r in output), {"train", "validation"})
                with self.assertRaises(FileExistsError):
                    run(cfg)


if __name__ == "__main__":
    unittest.main()
