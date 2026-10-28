"""Mocked evaluation tests; real model/data tests require the server environment."""
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.evaluate import validate_fusiformer_v2_modernbert as validation
from scripts.evaluate import test_fusiformer_v2_modernbert as testing
from fusiformer_v2.evaluation import _check_checkpoint, evaluate_loader, select_dataset, validate_settings


class Dataset(torch.utils.data.Dataset):
    original_indices = [17, 4, 23, 9, 1]
    data = [{"jid": f"id-{i}"} for i in original_indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return {"value": torch.tensor([float(i)])}, torch.tensor([float(i + 1)])

    def collect(self):
        def collate(samples):
            return {"value": torch.stack([x["value"] for x, _ in samples])}, torch.stack([y for _, y in samples])
        return collate


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.observed = []

    def forward(self, x):
        self.observed.append((self.training, torch.is_grad_enabled(), x["text_mode"]))
        return x["value"] + 0.5


def fake_v4():
    module = types.ModuleType("train_fusiformer_production_editor_v4")
    module.to_device = lambda x, device: {k: v.to(device) if torch.is_tensor(v) else v for k, v in x.items()}
    module.normalize_targets = lambda x, device: x.to(device)
    module.normalize_outputs = lambda x: x
    module.autocast_context = lambda device, enabled: __import__("contextlib").nullcontext()
    module.empty_metric_accumulator = lambda: {"absolute_error_sum": 0., "squared_error_sum": 0., "element_count": 0.}

    def update(acc, outputs, targets):
        error = outputs.detach().float() - targets.detach().float()
        acc["absolute_error_sum"] += error.abs().sum().item()
        acc["squared_error_sum"] += (error * error).sum().item()
        acc["element_count"] += error.numel()

    module.update_metric_accumulator = update
    module.finalize_metrics = lambda a: {
        "mae": a["absolute_error_sum"] / a["element_count"],
        "mse": a["squared_error_sum"] / a["element_count"],
        "rmse": math.sqrt(a["squared_error_sum"] / a["element_count"]),
    }
    return module


class EvaluationTests(unittest.TestCase):
    def test_split_selection_without_fallback(self):
        train, val, test = [1], [2], [3]
        self.assertIs(select_dataset((train, val, test), "validation"), val)
        self.assertIs(select_dataset((train, val, test), "test"), test)
        with self.assertRaisesRegex(ValueError, "no other split"):
            select_dataset((train, val, None), "test")

    def test_entries_and_required_checkpoint(self):
        self.assertNotEqual(validation.OUTPUT_CONFIG["directory"], testing.OUTPUT_CONFIG["directory"])
        for entry in (validation, testing):
            missing_checkpoint = dict(entry.CHECKPOINT_CONFIG, path=None)
            with self.assertRaisesRegex(ValueError, "CHECKPOINT_CONFIG"):
                validate_settings("validation", entry.MODEL_CONFIG, entry.DATA_CONFIG,
                                  missing_checkpoint, entry.OUTPUT_CONFIG)
        fake = types.ModuleType("fusiformer_v2.evaluation")
        called = []
        fake.run_evaluation = lambda split, *_: called.append(split)
        with patch.dict(sys.modules, {"fusiformer_v2.evaluation": fake}):
            validation.main()
            testing.main()
        self.assertEqual(called, ["validation", "test"])

    def test_no_overwrite_or_checkpoint_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / name for name in ("data.json", "cache.npy", "model.pt")]
            for path in paths:
                path.touch()
            data = dict(validation.DATA_CONFIG, data_path=paths[0], text_cache_path=paths[1])
            checkpoint = dict(validation.CHECKPOINT_CONFIG, path=paths[2])
            output = dict(validation.OUTPUT_CONFIG, directory=root)
            (root / output["metrics_filename"]).touch()
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                validate_settings("validation", validation.MODEL_CONFIG, data, checkpoint, output)
            checkpoint["path"] = root / "missing.pt"
            with self.assertRaisesRegex(FileNotFoundError, "CHECKPOINT_CONFIG.path"):
                validate_settings("validation", validation.MODEL_CONFIG, data, checkpoint, output)

    def test_checkpoint_compatibility(self):
        model, data = validation.MODEL_CONFIG, validation.DATA_CONFIG
        keys = ("target", "stage", "num_queries", "cross_dropout", "model_depth",
                "edge_input_dim", "triplet_input_dim", "embed_dim", "num_heads",
                "text_proj_dim", "text_proj_hidden_dims", "text_proj_dropout", "text_model_path")
        saved = {key: model[key] for key in keys}
        saved.update({key: data[key] for key in ("text_field", "data_split_ratio", "data_split_counts")})
        checkpoint = {"format_version": 4, "model_state_dict": {"weight": torch.ones(1)},
                      "excluded_model_prefixes": [],
                      "training_config": {"fusiformer_version": "V2.1", "independent_user_config": saved}}
        self.assertIs(_check_checkpoint(checkpoint, model, data, validation.CHECKPOINT_CONFIG),
                      checkpoint["model_state_dict"])
        with self.assertRaisesRegex(ValueError, "excludes"):
            _check_checkpoint(dict(checkpoint, excluded_model_prefixes=["cross_modal."]),
                              model, data, validation.CHECKPOINT_CONFIG)
        with self.assertRaisesRegex(ValueError, "format_version"):
            _check_checkpoint(dict(checkpoint, format_version=3), model, data, validation.CHECKPOINT_CONFIG)
        with self.assertRaisesRegex(ValueError, "data_split_ratio"):
            _check_checkpoint(checkpoint, model, dict(data, data_split_ratio=(0.7, 0.2, 0.1)),
                              validation.CHECKPOINT_CONFIG)

    def test_alignment_last_batch_eval_and_no_grad(self):
        dataset = Dataset()
        dataset.data[0] = {"mp_id": "mp-17"}  # Formal JSON identifier; CSV schema remains unchanged.
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False,
                                             drop_last=False, collate_fn=dataset.collect())
        model = Model().train()
        with patch.dict(sys.modules, {"train_fusiformer_production_editor_v4": fake_v4()}):
            metrics, rows = evaluate_loader(model, loader, dataset, torch.device("cpu"),
                                            text_mode="bypass", amp_enabled=False)
        self.assertEqual([row["original_index"] for row in rows], dataset.original_indices)
        self.assertEqual([row["sample_id"] for row in rows],
                         ["mp-17"] + [f"id-{i}" for i in dataset.original_indices[1:]])
        self.assertEqual(len(rows), 5)
        self.assertEqual(metrics["mae"], 0.5)
        self.assertEqual(metrics["mse"], 0.25)
        self.assertTrue(all(item == (False, False, "bypass") for item in model.observed))
        shuffled = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True,
                                               drop_last=False, collate_fn=dataset.collect())
        with patch.dict(sys.modules, {"train_fusiformer_production_editor_v4": fake_v4()}):
            with self.assertRaisesRegex(ValueError, "sequential"):
                evaluate_loader(model, shuffled, dataset, torch.device("cpu"),
                                text_mode="normal", amp_enabled=False)


if __name__ == "__main__":
    unittest.main()
