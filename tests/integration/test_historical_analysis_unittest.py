"""Synthetic historical analysis and mechanism entry-point regressions."""
import csv
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.cluster_specialists.config import HistoricalSourceConfig
from scripts.diagnostics import analyze_historical_clusters as analysis
from scripts.diagnostics import compare_cluster_mechanisms as mechanisms


class HistoricalAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rows = [
            dict(sample_id="s2", mp_id="mp-2", physical_id="p2", original_index="100", split="test", cluster_id="B"),
            dict(sample_id="s0", mp_id="mp-0", physical_id="p0", original_index="9", split="test", cluster_id="A"),
            dict(sample_id="s1", mp_id="mp-1", physical_id="p1", original_index="42", split="test", cluster_id="C"),
        ]
        self.master = self.csv("master.csv", self.rows)
        self.labels = self.csv("labels.csv", [
            dict(sample_id="s1", raw="2"), dict(sample_id="s2", raw="1"), dict(sample_id="s0", raw="0")])
        self.source = HistoricalSourceConfig(master_path=str(self.master), labels_path=str(self.labels),
                                             label_col="raw", label_mapping=(("0", "A"), ("1", "B"), ("2", "C")))

    def csv(self, name, rows):
        path = self.root / name
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def array(self, name, data, order):
        array = self.root / (name + ".npy")
        np.save(array, np.asarray([data[self.rows[i]["sample_id"]] for i in order]))
        fields = ("sample_id", "original_index", "mp_id", "physical_id", "split")
        index = self.csv(name + ".csv", [{k: self.rows[i][k] for k in fields} for i in order])
        return str(array), str(index)

    def inputs(self):
        target = self.array("targets", {"s0": 0., "s1": 1., "s2": 2.}, [1, 0, 2])
        base = self.array("base", {"s0": 1., "s1": 3., "s2": 2.}, [2, 0, 1])
        direct = self.array("direct", {"s0": .5, "s1": 1., "s2": 1.}, [0, 1, 2])
        residual = self.array("residual", {"s0": 0., "s1": 2., "s2": 2.}, [1, 2, 0])
        return target, base, direct, residual

    def test_historical_analysis_entry_and_no_overwrite(self):
        target, base, _, _ = self.inputs()
        cfg = analysis.Config(source=self.source, targets_path=target[0], targets_index_path=target[1],
                              baseline_path=base[0], baseline_index_path=base[1], output_dir=str(self.root / "analysis"))
        with patch.object(analysis, "CFG", cfg), patch.object(sys, "argv", ["analysis", "--check-config"]):
            analysis.main()
        self.assertFalse(Path(cfg.output_dir).exists())
        with patch.object(analysis, "CFG", cfg), patch.object(sys, "argv", ["analysis"]):
            analysis.main()
            with self.assertRaises(FileExistsError):
                analysis.main()
        report = json.loads((Path(cfg.output_dir) / "historical_summary.json").read_text())
        self.assertEqual(report["analysis_type"], "descriptive")
        self.assertEqual(report["clusters"]["A"]["baseline"]["mae"], 1.)
        self.assertEqual(report["largest_absolute_errors"][0]["sample_id"], "s1")
        with (Path(cfg.output_dir) / "historical_samples.csv").open(newline="") as handle:
            rows = {r["sample_id"]: r for r in csv.DictReader(handle)}
        self.assertEqual(rows["s1"]["original_index"], "42")
        self.assertEqual(rows["s1"]["source_label"], "2")
        self.assertEqual(float(rows["s1"]["baseline_prediction"]), 3.)

    def test_mechanism_entry_identity_and_metrics(self):
        target, base, direct, residual = self.inputs()
        cfg = mechanisms.Config(source=self.source, targets_path=target[0], targets_index_path=target[1],
            predictions=(("baseline", *base), ("direct", *direct), ("residual", *residual)),
            output_dir=str(self.root / "compare"))
        with patch.object(mechanisms, "CFG", cfg), patch.object(sys, "argv", ["compare"]):
            mechanisms.main()
            with self.assertRaises(FileExistsError):
                mechanisms.main()
        summary = json.loads((Path(cfg.output_dir) / "mechanisms_summary.json").read_text())
        self.assertEqual(summary["overall"]["baseline"]["mae"], 1.)
        self.assertEqual(summary["clusters"]["A"]["direct"]["mae"], .5)
        with (Path(cfg.output_dir) / "mechanisms_samples.csv").open(newline="") as handle:
            rows = {r["sample_id"]: r for r in csv.DictReader(handle)}
        self.assertEqual(float(rows["s1"]["direct_absolute_error_change"]), -2.)
        self.assertEqual(rows["s1"]["cluster_id"], "C")

    def test_missing_prediction_and_identity_mismatch_fail_before_writes(self):
        target, base, direct, _ = self.inputs()
        cfg = mechanisms.Config(source=self.source, targets_path=target[0], targets_index_path=target[1],
            predictions=(("baseline", *base), ("direct", str(self.root / "missing.npy"), direct[1])),
            output_dir=str(self.root / "missing_out"))
        with self.assertRaisesRegex(FileNotFoundError, "missing.npy"):
            mechanisms.load_inputs(cfg)
        self.assertFalse(Path(cfg.output_dir).exists())
        rows = [{k: self.rows[i][k] for k in ("sample_id", "original_index", "mp_id", "physical_id", "split")} for i in (0, 1, 2)]
        rows[0]["original_index"] = "999"
        self.csv("direct.csv", rows)
        cfg = mechanisms.Config(source=self.source, targets_path=target[0], targets_index_path=target[1],
            predictions=(("baseline", *base), ("direct", *direct)), output_dir=str(self.root / "bad_out"))
        with self.assertRaisesRegex(ValueError, "original_index mismatch"):
            mechanisms.load_inputs(cfg)
        self.assertFalse(Path(cfg.output_dir).exists())

    def test_manifest_and_label_shuffle_preserves_sample_metrics(self):
        target, base, direct, residual = self.inputs()
        cfg = mechanisms.Config(source=self.source, targets_path=target[0], targets_index_path=target[1],
            predictions=(("baseline", *base), ("direct", *direct), ("residual", *residual)))
        rows, targets, predictions = mechanisms.load_inputs(cfg)
        first = mechanisms.compare_mechanisms(rows, targets, predictions)
        self.csv("master.csv", list(reversed(self.rows)))
        self.csv("labels.csv", [dict(sample_id="s0", raw="0"),
                                dict(sample_id="s2", raw="1"), dict(sample_id="s1", raw="2")])
        rows, targets, predictions = mechanisms.load_inputs(cfg)
        second = mechanisms.compare_mechanisms(rows, targets, predictions)
        self.assertEqual(first, second)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib unavailable")
    def test_optional_descriptive_plot_is_created_in_new_output(self):
        target, base, _, _ = self.inputs()
        cfg = analysis.Config(source=self.source, targets_path=target[0], targets_index_path=target[1],
                              baseline_path=base[0], baseline_index_path=base[1],
                              output_dir=str(self.root / "plotted"), plot=True)
        with patch.dict(os.environ, {"MPLBACKEND": "Agg", "MPLCONFIGDIR": str(self.root / "mpl")}):
            with patch.object(analysis, "CFG", cfg), patch.object(sys, "argv", ["analysis"]):
                analysis.main()
        self.assertGreater((Path(cfg.output_dir) / "historical_distributions.png").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
