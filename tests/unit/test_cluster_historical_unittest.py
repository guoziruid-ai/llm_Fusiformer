"""Synthetic identity and historical-label reader regressions (no fitting)."""
import csv
import json
from dataclasses import replace
import tempfile
import unittest
from pathlib import Path

from experiments.cluster_specialists.historical import load_historical_rows, verify_grouped3_mapping
from experiments.cluster_specialists.config import HistoricalSourceConfig


class HistoricalLabelsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.master = [
            dict(sample_id="s1", physical_id="p1", mp_id="mp-1", original_index="91", split="train"),
            dict(sample_id="s2", physical_id="p2", mp_id="mp-2", original_index="7", split="validation"),
            dict(sample_id="s3", physical_id="p3", mp_id="mp-3", original_index="230", split="test"),
        ]
        self.labels = [dict(sample_id="s1", raw="0"), dict(sample_id="s2", raw="1"), dict(sample_id="s3", raw="2")]
        self.mapping = {"0": "C", "1": "A", "2": "B"}

    def write(self, name, rows, fields=None):
        path = self.root / name
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def load(self, master=None, labels=None, **kwargs):
        return load_historical_rows(
            self.write("master.csv", self.master if master is None else master),
            self.write("labels.csv", self.labels if labels is None else labels),
            label_col="raw", label_mapping=self.mapping, **kwargs)

    def test_shuffled_noncontiguous_original_indices(self):
        original = self.load()
        shuffled = self.load(master=list(reversed(self.master)), labels=[self.labels[1], self.labels[2], self.labels[0]])
        identity = lambda rows: {row["sample_id"]: (row["mp_id"], row["original_index"], row["source_label"], row["cluster_id"], row["split"]) for row in rows}
        self.assertEqual(identity(original), identity(shuffled))
        self.assertEqual(identity(original)["s1"], ("mp-1", "91", "0", "C", "train"))

    def test_missing_duplicate_and_unknown_ids(self):
        for labels in (self.labels[:2], self.labels + [self.labels[0]], self.labels[:2] + [dict(sample_id="unknown", raw="2")]):
            with self.subTest(labels=labels), self.assertRaisesRegex(ValueError, "missing|duplicate|unknown"):
                self.load(labels=labels)

    def test_conflicting_existing_labels_and_split(self):
        master = [dict(row, cluster_id=cluster) for row, cluster in zip(self.master, ("A", "A", "B"))]
        with self.assertRaisesRegex(ValueError, "cluster_id mismatch"):
            self.load(master=master)
        labels = [dict(row, split=split) for row, split in zip(self.labels, ("test", "validation", "test"))]
        with self.assertRaisesRegex(ValueError, "split mismatch"):
            self.load(labels=labels)

    def test_identity_conflicts_and_physical_split_leakage(self):
        labels = [dict(row, mp_id=mp_id) for row, mp_id in zip(self.labels, ("mp-wrong", "mp-2", "mp-3"))]
        with self.assertRaisesRegex(ValueError, "mp_id mismatch"):
            self.load(labels=labels)
        master = [dict(row) for row in self.master]
        master[1]["original_index"] = master[0]["original_index"]
        with self.assertRaisesRegex(ValueError, "duplicate original_index"):
            self.load(master=master)
        master = [dict(row) for row in self.master]
        master[1]["physical_id"] = master[0]["physical_id"]
        # First use consistent labels to reach the physical split check.
        labels = [dict(row) for row in self.labels]
        labels[1]["raw"] = "0"
        with self.assertRaisesRegex(ValueError, "physical sample leakage"):
            self.load(master=master, labels=labels)

    def test_missing_required_file_and_field(self):
        master = self.write("master.csv", self.master)
        with self.assertRaisesRegex(FileNotFoundError, "required historical input"):
            load_historical_rows(master, self.root / "absent.csv", label_col="raw", label_mapping=self.mapping)
        labels = self.write("labels.csv", [dict(sample_id="s1")])
        with self.assertRaisesRegex(ValueError, "missing fields.*raw"):
            load_historical_rows(master, labels, label_col="raw", label_mapping=self.mapping)

    def test_explicit_mapping_and_unknown_label_required(self):
        master = self.write("master.csv", self.master)
        labels = self.write("labels.csv", self.labels)
        with self.assertRaisesRegex(ValueError, "explicit.*mapping"):
            load_historical_rows(master, labels, label_col="raw")
        with self.assertRaisesRegex(ValueError, "unknown source label"):
            load_historical_rows(master, labels, label_col="raw", label_mapping={"0": "A"})

    def test_position_requires_explicit_run_mapping(self):
        labels = [dict(train_position="8", raw="0"), dict(train_position="1", raw="1"), dict(train_position="5", raw="2")]
        master = self.write("master.csv", [dict(row, split="train") for row in self.master])
        label_path = self.write("labels.csv", labels)
        with self.assertRaisesRegex(ValueError, "position.*explicit"):
            load_historical_rows(master, label_path, label_col="raw", label_mapping=self.mapping, position_label_col="train_position")
        positions = [dict(pos="5", sample_id="s3", split="train"), dict(pos="8", sample_id="s1", split="train"), dict(pos="1", sample_id="s2", split="train")]
        position_path = self.write("position.csv", positions)
        rows = load_historical_rows(master, label_path, label_col="raw", label_mapping=self.mapping,
                                    position_map_path=position_path, position_label_col="train_position", position_map_position_col="pos")
        self.assertEqual([row["cluster_id"] for row in rows], ["C", "A", "B"])
        positions[0]["split"] = "test"
        self.write("position.csv", positions)
        with self.assertRaisesRegex(ValueError, "split mismatch"):
            load_historical_rows(master, label_path, label_col="raw", label_mapping=self.mapping,
                                 position_map_path=position_path, position_label_col="train_position", position_map_position_col="pos")

    def test_position_map_must_be_bijective(self):
        master = self.write("master.csv", [dict(row, split="train") for row in self.master])
        labels = self.write("labels.csv", [dict(pos="1", raw="0"), dict(pos="2", raw="1"), dict(pos="3", raw="2")])
        for positions in ([dict(pos="1", sample_id="s1"), dict(pos="1", sample_id="s2"), dict(pos="3", sample_id="s3")],
                          [dict(pos="1", sample_id="s1"), dict(pos="2", sample_id="s1"), dict(pos="3", sample_id="s3")],
                          [dict(pos="1", sample_id="s1"), dict(pos="2", sample_id="s2"), dict(pos="3", sample_id="unknown")]):
            positions = [dict(row, split="train") for row in positions]
            with self.subTest(positions=positions), self.assertRaises(ValueError):
                load_historical_rows(master, labels, label_col="raw", label_mapping=self.mapping,
                                     position_map_path=self.write("map.csv", positions), position_label_col="pos", position_map_position_col="pos")

    def test_position_map_rejects_multiple_splits(self):
        master = self.write("master.csv", self.master)
        labels = self.write("labels.csv", [dict(pos="1", raw="0")])
        positions = self.write("map.csv", [dict(pos="1", sample_id="s1", split="train")])
        with self.assertRaisesRegex(ValueError, "exactly one split"):
            load_historical_rows(master, labels, label_col="raw", label_mapping=self.mapping,
                                 position_map_path=positions, position_label_col="pos", position_map_position_col="pos")

    def test_grouped3_actual_shape_and_five_to_three_mapping(self):
        artifact = {
            "cluster_to_expert_id": [2, 0, 1, 0, 0],
            "expert_id_to_name": {"0": "expert_A_shared3", "1": "expert_B_hardest", "2": "expert_C_second_hardest"},
            "expert_to_clusters": {"0": [1, 3, 4], "1": [2], "2": [0]},
        }
        mapping_path = self.root / "grouping.json"
        mapping_path.write_text(json.dumps(artifact), encoding="utf-8")
        routes = [dict(cluster_id=str(raw), expert_id=str(expert), expert_name=artifact["expert_id_to_name"][str(expert)])
                  for raw, expert in enumerate(artifact["cluster_to_expert_id"])]
        labels = self.write("routes.csv", list(reversed(routes)))
        expert_mapping = verify_grouped3_mapping(labels, mapping_path)
        self.assertEqual(expert_mapping, {"0": "A", "1": "B", "2": "C"})
        id_routes = [dict(row, sample_id=sample) for row, sample in zip(routes[:3], ("s1", "s2", "s3"))]
        aligned = load_historical_rows(self.write("master.csv", self.master), self.write("id_routes.csv", id_routes),
                                       label_col="expert_id", label_mapping=expert_mapping)
        self.assertEqual([(row["historical_raw_cluster_id"], row["source_label"], row["cluster_id"]) for row in aligned],
                         [("0", "2", "C"), ("1", "0", "A"), ("2", "1", "B")])
        broken = [dict(row) for row in routes]
        broken[0]["expert_id"] = "0"
        with self.assertRaisesRegex(ValueError, "route mapping conflict"):
            verify_grouped3_mapping(self.write("broken.csv", broken), mapping_path)
        broken[0] = dict(routes[0], expert_name="expert_A_shared3")
        with self.assertRaisesRegex(ValueError, "route mapping conflict"):
            verify_grouped3_mapping(self.write("broken.csv", broken), mapping_path)
        broken[0] = dict(routes[0], cluster_id="5")
        with self.assertRaisesRegex(ValueError, "unknown.*raw"):
            verify_grouped3_mapping(self.write("broken.csv", broken), mapping_path)
        with self.assertRaisesRegex(ValueError, "missing fields"):
            verify_grouped3_mapping(self.write("missing.csv", [dict(cluster_id="0", expert_id="2")]), mapping_path)
        artifact["expert_to_clusters"]["0"] = [1, 3]
        mapping_path.write_text(json.dumps(artifact), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            verify_grouped3_mapping(labels, mapping_path)
        del artifact["cluster_to_expert_id"]
        mapping_path.write_text(json.dumps(artifact), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "cluster_to_expert_id"):
            verify_grouped3_mapping(labels, mapping_path)

    def test_grouped3_source_config_requires_artifact_and_agreement(self):
        artifact = {
            "cluster_to_expert_id": [2, 0, 1, 0, 0],
            "expert_id_to_name": {"0": "expert_A_shared3", "1": "expert_B_hardest", "2": "expert_C_second_hardest"},
            "expert_to_clusters": {"0": [1, 3, 4], "1": [2], "2": [0]},
        }
        mapping_path = self.root / "grouping.json"
        mapping_path.write_text(json.dumps(artifact), encoding="utf-8")
        labels = [dict(sample_id=sample, cluster_id=str(raw), expert_id=str(expert),
                       expert_name=artifact["expert_id_to_name"][str(expert)])
                  for sample, raw, expert in (("s1", 0, 2), ("s2", 1, 0), ("s3", 2, 1))]
        config = HistoricalSourceConfig(master_path=str(self.write("master.csv", self.master)),
                                        labels_path=str(self.write("labels.csv", labels)),
                                        label_col="expert_id", label_mapping=(("0", "A"), ("1", "B"), ("2", "C")))
        with self.assertRaisesRegex(ValueError, "Grouped-3.*mapping"):
            config.load()
        config = replace(config, grouped3_mapping_path=str(mapping_path))
        rows = config.load()
        self.assertEqual([(row["historical_raw_cluster_id"], row["source_label"], row["cluster_id"]) for row in rows],
                         [("0", "2", "C"), ("1", "0", "A"), ("2", "1", "B")])
        with self.assertRaisesRegex(ValueError, "differs from Grouped-3"):
            replace(config, label_mapping=(("0", "C"), ("1", "B"), ("2", "A"))).load()
        with self.assertRaisesRegex(ValueError, "label_col"):
            replace(config, label_col="cluster_id").load()


if __name__ == "__main__":
    unittest.main()
