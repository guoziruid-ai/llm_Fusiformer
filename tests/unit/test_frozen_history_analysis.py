"""Synthetic identity and read-only workflow tests, not historical validation."""
import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from scripts.diagnostics import analyze_frozen_history as analysis


class FrozenHistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.identity = [dict(sample_id=f"mp-{i}", original_index=str(10+i),
                              vector_row=str(20-i), split="train" if i < 3 else "test")
                         for i in range(6)]
        self.assignments = [dict(index=r["original_index"], mp_id=r["sample_id"],
                                 e_form=str(i), fusiformer_cluster=str(i % 5))
                            for i, r in enumerate(self.identity)]
        self.predictions = [dict(original_index=r["original_index"], mp_id=r["sample_id"],
                                 split=r["split"], cluster=str(i % 5), y_true=str(i),
                                 y_pred=str(i+.5)) for i, r in enumerate(self.identity) if i >= 3]
        self.routes = [dict(train_position=str(i), cluster_id=str(i), expert_id=str(i),
                            expert_name=["expert_A_shared3", "expert_B_hardest", "expert_C_second_hardest"][i])
                       for i in range(3)]
        self.files = {key: self.root / f"{key}.csv" for key in ("identity", "assignments", "predictions", "train_routes")}
        self.files.update({"mapping": self.root / "mapping.json", "router_model": self.root / "model.joblib",
                           "identity_evidence": self.root / "evidence.json", "train_indices": self.root / "indices.npy"})
        self.write_csv("identity", self.identity); self.write_csv("assignments", self.assignments)
        self.write_csv("predictions", self.predictions); self.write_csv("train_routes", self.routes)
        self.files["router_model"].write_bytes(b"synthetic trusted opaque artifact")
        np.save(self.files["train_indices"], np.array([10, 11, 12]))
        self.mapping = {"cluster_to_expert_id": [0, 1, 2, 0, 0], "expert_id_to_name": {
            "0": "expert_A_shared3", "1": "expert_B_hardest", "2": "expert_C_second_hardest"}}
        self.write_mapping()
        self.evidence()

    def write_csv(self, name, rows):
        with self.files[name].open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    def write_mapping(self):
        self.files["mapping"].write_text(json.dumps(self.mapping), encoding="utf-8")

    def evidence(self):
        hashes = {k: analysis.sha256_file(p) for k, p in self.files.items() if p.is_file()}
        data = {"identity_sha256": hashes["identity"], "same_identity_mapping_verified": True,
                "artifacts": {"full_labels": {"sha256": hashes["assignments"]},
                              "train_model": {"sha256": hashes["router_model"]}},
                "identity_verification": {"inputs": {k: {"sha256": hashes[k]}
                   for k in ("train_routes", "train_indices")}}}
        self.files["identity_evidence"].write_text(json.dumps(data), encoding="utf-8")

    def cfg(self, output="out"):
        return analysis.Config(paths={k: str(v) for k, v in self.files.items()},
            sha256={k: analysis.sha256_file(v) for k, v in self.files.items()},
            prediction_split="test", output_dir=str(self.root / output), allow_write=True, plot=False)

    def test_shuffled_identity_mapping_and_cross_space(self):
        result = analysis.run(self.cfg())
        self.assertEqual(result["overall"]["count"], 3)
        self.assertEqual(result["overall"]["mae"], .5)
        self.assertEqual(result["baseline_prediction_route_matched_count"], 0)
        self.assertIn("unavailable", result["route_stratification_status"])
        with (self.root / "out" / "samples.csv").open() as f:
            expected_rows = {r["sample_id"]: r for r in csv.DictReader(f)}
        self.write_csv("identity", list(reversed(self.identity)))
        self.write_csv("assignments", list(reversed(self.assignments)))
        self.write_csv("predictions", list(reversed(self.predictions)))
        self.write_csv("train_routes", list(reversed(self.routes)))
        self.evidence()
        other = analysis.run(self.cfg("other"))
        self.assertEqual(result, other)
        with (self.root / "other" / "samples.csv").open() as f:
            actual_rows = {r["sample_id"]: r for r in csv.DictReader(f)}
        self.assertEqual(expected_rows, actual_rows)
        with (self.root / "other" / "historical_vs_router_counts.csv").open() as f:
            cross = list(csv.DictReader(f))
        self.assertEqual(sum(int(r["count"]) for r in cross), 3)
        self.assertEqual({r["split"] for r in cross}, {"train"})

    def test_prediction_missing_duplicate_unknown_and_split_mismatch(self):
        variants = [self.predictions[:-1], self.predictions + self.predictions[:1],
                    self.predictions[:-1] + [dict(self.predictions[-1], original_index="999")],
                    self.predictions[:-1] + [dict(self.predictions[-1], split="train")],
                    self.predictions[:-1] + [dict(self.predictions[-1], y_pred="")]]
        for rows in variants:
            self.write_csv("predictions", rows)
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                analysis.load(self.cfg())

    def test_duplicate_identity_and_missing_assignment(self):
        self.write_csv("identity", self.identity + self.identity[:1]); self.evidence()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            analysis.load(self.cfg())
        self.write_csv("identity", self.identity)
        self.write_csv("assignments", self.assignments[:-1]); self.evidence()
        with self.assertRaisesRegex(ValueError, "missing"):
            analysis.load(self.cfg())

    def test_route_mapping_and_position_fail_closed(self):
        self.mapping["cluster_to_expert_id"] = [0, 1]
        self.write_mapping()
        with self.assertRaisesRegex(ValueError, "five"):
            analysis.load(self.cfg())
        self.mapping["cluster_to_expert_id"] = [0, 1, 2, 0, 0]; self.write_mapping()
        np.save(self.files["train_indices"], np.array([10, 11, 15])); self.evidence()
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            analysis.load(self.cfg())

    def test_mapping_and_route_resources_are_required(self):
        del self.mapping["cluster_to_expert_id"]
        self.write_mapping()
        with self.assertRaisesRegex(ValueError, "five"):
            analysis.load(self.cfg())
        cfg = self.cfg()
        paths = {k: v for k, v in cfg.paths.items() if not k.startswith("train_")}
        hashes = {k: v for k, v in cfg.sha256.items() if k in paths}
        with self.assertRaisesRegex(ValueError, "at least one verified"):
            analysis.load(replace(cfg, paths=paths, sha256=hashes))

    def test_hash_mismatch_and_no_overwrite(self):
        cfg = self.cfg()
        self.files["router_model"].write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            analysis.run(cfg)
        self.assertFalse((self.root / "out").exists())
        self.files["router_model"].write_bytes(b"synthetic trusted opaque artifact")
        analysis.run(self.cfg())
        with self.assertRaises(FileExistsError):
            analysis.run(self.cfg())

    def test_missing_same_split_routes_can_be_required(self):
        with self.assertRaisesRegex(ValueError, "missing reliable routes"):
            analysis.load(replace(self.cfg(), require_prediction_routes=True))

    def test_same_split_routes_use_verified_position_mapping(self):
        train = [dict(original_index=r["original_index"], mp_id=r["sample_id"],
                      split="train", cluster=str(i % 5), y_true=str(i), y_pred=str(i+1))
                 for i, r in enumerate(self.identity[:3])]
        self.write_csv("predictions", list(reversed(train)))
        cfg = replace(self.cfg(), prediction_split="train", require_prediction_routes=True)
        summary = analysis.run(cfg)
        self.assertEqual(summary["baseline_prediction_route_matched_count"], 3)
        self.assertEqual(summary["baseline_error_by_router_expert"]["1"]["mae"], 1.)
        with (self.root / "out" / "samples.csv").open() as f:
            rows = {r["mp_id"]: r for r in csv.DictReader(f)}
        self.assertEqual(rows["mp-1"]["router_cluster"], "1")
        self.assertEqual(rows["mp-1"]["vector_row"], "19")

    def test_plot_and_manifest(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib not available")
        analysis.run(replace(self.cfg(), plot=True))
        self.assertTrue((self.root / "out" / "absolute_errors.png").is_file())
        manifest = json.loads((self.root / "out" / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["outputs"]["absolute_errors.png"],
                         analysis.sha256_file(self.root / "out" / "absolute_errors.png"))

    def test_unconfigured_defaults_fail(self):
        with self.assertRaises(PermissionError):
            analysis.run(analysis.CFG)

    def test_preflight_writes_nothing(self):
        report = analysis.preflight(self.cfg())
        self.assertFalse(report["writes"])
        self.assertEqual(report["prediction_count"], 3)
        self.assertFalse((self.root / "out").exists())


if __name__ == "__main__":
    unittest.main()
