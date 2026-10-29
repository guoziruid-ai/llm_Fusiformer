"""Training loader must pair features and labels using explicit identities."""
import csv
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts.experiments.cluster_specialists.config import SplitRatios
from scripts.experiments.cluster_specialists.manifests import write_manifests
from scripts.train import train_cluster_specialist as entry


class TrainingAlignmentTests(unittest.TestCase):
    def test_shuffled_arrays_and_sparse_indices_keep_splits_and_pairs(self):
        rows = [
            {"sample_id": f"sample-{i}", "physical_id": f"p-{i}", "mp_id": f"mp-{i}",
             "original_index": 11 + 7 * i, "cluster_id": "A", "split": split}
            for i, split in enumerate(["train", "validation", "train", "test", "validation", "train"])]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_manifests(list(reversed(rows)), root / "manifest",
                                       seed=42, ratios=SplitRatios(), source="fixture")["master"]
            paths = {}
            for shift, name in enumerate(("features", "targets", "baseline"), start=1):
                order = np.roll(np.arange(len(rows)), shift)
                values = np.arange(len(rows), dtype=np.float32)
                if name == "features":
                    values = values.reshape(-1, 1)
                if name == "baseline":
                    values = values + 100
                array_path, index_path = root / f"{name}.npy", root / f"{name}.csv"
                np.save(array_path, values[order])
                with index_path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows[i] for i in order)
                paths[name] = (str(array_path), str(index_path))
            cfg = replace(
                entry.CFG, cluster_id="A", prediction_mode="residual",
                manifest_path=manifest, feature_path=paths["features"][0],
                feature_index_path=paths["features"][1], target_path=paths["targets"][0],
                target_index_path=paths["targets"][1],
                baseline_prediction_path=paths["baseline"][0],
                baseline_prediction_index_path=paths["baseline"][1],
                input_dim=1, hidden_dims=(4,), batch_size=20, device="cpu",
                output_checkpoint=str(root / "never_written.pt"))
            observed = {}

            def inspect_loaders(model, train_loader, validation_loader, **kwargs):
                for name, loader in (("train", train_loader), ("validation", validation_loader)):
                    x, y, baseline = next(iter(loader))
                    pairs = sorted(zip(x.reshape(-1).tolist(), y.reshape(-1).tolist(),
                                       baseline.reshape(-1).tolist()))
                    observed[name] = pairs
                return SimpleNamespace(best_epoch=1, best_validation_loss=0.0)

            with patch.object(entry, "CFG", cfg), patch("sys.argv", ["train"]), \
                    patch.object(entry, "train_tensor_expert", side_effect=inspect_loaders), \
                    patch.object(entry, "save_checkpoint", return_value=root / "mock.pt"):
                entry.main()
            self.assertEqual(observed["train"], [(0.0, 0.0, 100.0),
                                                   (2.0, 2.0, 102.0),
                                                   (5.0, 5.0, 105.0)])
            self.assertEqual(observed["validation"], [(1.0, 1.0, 101.0),
                                                        (4.0, 4.0, 104.0)])


if __name__ == "__main__":
    unittest.main()
