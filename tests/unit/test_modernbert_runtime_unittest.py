"""CPU-only checks for the shared ModernBERT launcher preflight."""
import json
import io
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.train import train_fusiformer_v2_modernbert as regular
from scripts.train import train_fusiformer_v2_modernbert_independent as independent


class ModernBertRuntimeTests(unittest.TestCase):
    def test_independent_check_config_runs_preflight_without_training(self):
        engine = unittest.mock.Mock()
        runtime = engine.build_runtime_config.return_value
        cache_info = {"cache_shape": (2, 3, 768)}
        with patch.object(independent, "validate_user_config") as validate, \
             patch.object(independent, "inspect_cache", return_value=cache_info) as inspect, \
             patch.object(independent, "setup_training", return_value=engine) as setup, \
             patch.object(independent, "report_configuration") as report, \
             patch.object(sys, "argv", ["train", "--check-config"]), \
             redirect_stdout(io.StringIO()):
            independent.main()
        validate.assert_called_once_with(independent.CFG)
        inspect.assert_called_once_with(independent.CFG)
        setup.assert_called_once_with(independent.CFG)
        engine.validate_config.assert_called_once_with(runtime)
        report.assert_called_once_with(independent.CFG, runtime, cache_info)
        engine.main.assert_not_called()

    def test_launchers_keep_distinct_experiment_defaults(self):
        regular.validate_user_config(regular.CFG)
        independent.validate_user_config(independent.CFG)
        self.assertEqual(regular.CFG.cross_dropout, 0.0)
        self.assertEqual(independent.CFG.cross_dropout, 0.1)
        self.assertEqual(regular.CFG.absolute_lr, 1e-4)
        self.assertIsNone(independent.CFG.absolute_lr)
        self.assertIs(regular.validate_user_config, independent.validate_user_config)
        self.assertIs(regular.inspect_cache, independent.inspect_cache)

    def test_validation_rejects_invalid_stage(self):
        with self.assertRaisesRegex(ValueError, "a1"):
            regular.validate_user_config(replace(regular.CFG, stage="bad"))

    def test_engine_hooks_preserve_each_launcher_metadata(self):
        for launcher in (regular, independent):
            with self.subTest(launcher=launcher.__name__):
                baseline = types.ModuleType("train_fusiformer_production_editor_v6")
                baseline.__getattr__ = lambda name: None
                baseline.engine = types.SimpleNamespace(
                    clone_inputs_without_text=lambda inputs: dict(inputs))
                baseline.build_runtime_config = lambda: types.SimpleNamespace(
                    mode="scratch", text_encoder_name="modernbert",
                    freeze_text_encoder=True, use_precomputed_text_embeddings=True)
                baseline.build_training_config = lambda *args, **kwargs: {"original": True}
                baseline.build_dataloaders = object()
                baseline.parameter_lr_category = lambda name: "backbone"
                dataset = types.ModuleType("fusiformer_v2.dataset")
                dataset.get_dataset_v2 = object()
                model = types.ModuleType("fusiformer_v2.model")
                model.FusiformerV2 = object()
                with patch.dict(sys.modules, {
                    baseline.__name__: baseline,
                    dataset.__name__: dataset,
                    model.__name__: model,
                }):
                    engine = launcher.setup_training(launcher.CFG)
                    runtime = engine.build_runtime_config()
                    payload = engine.build_training_config()
                self.assertIs(engine, baseline.engine)
                self.assertEqual(baseline.SCRIPT_VERSION, launcher.VERSION)
                self.assertEqual(baseline.text_proj_dropout, launcher.CFG.text_proj_dropout)
                self.assertEqual(runtime.v2_stage, launcher.CFG.stage)
                self.assertEqual(runtime.v2_cross_dropout, launcher.CFG.cross_dropout)
                self.assertEqual(payload["independent_script"], Path(launcher.__file__).name)
                self.assertEqual(payload["independent_script_version"], launcher.VERSION)
                self.assertEqual(payload["independent_user_config"]["absolute_lr"], launcher.CFG.absolute_lr)
                self.assertEqual(engine.parameter_lr_category("cross_modal.attn.weight"), "text")
                self.assertEqual(engine.clone_inputs_without_text({"graph_input": 1})["text_mode"], "bypass")

    def test_cache_preflight_checks_shape_metadata_and_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data.json"
            data.write_text("[]", encoding="utf-8")
            model = root / "ModernBERT-base"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            cache = root / "segments.npy"
            mask = root / "segments_mask.npy"
            meta = root / "segments.npy.meta.json"
            np.save(cache, np.zeros((2, 3, 768), dtype=np.float32))
            np.save(mask, np.array([[True, False, False], [True, True, False]]))
            metadata = {
                "cache_format": "fusiformer-v2-segments-v1",
                "encoder_name": "modernbert",
                "text_field": "description",
                "model_name": str(model),
                "text_sha256": "fixture-fingerprint",
                "shape": [2, 3, 768],
                "hidden_size": 768,
            }
            meta.write_text(json.dumps(metadata), encoding="utf-8")
            cfg = replace(regular.CFG, data_path=str(data),
                          text_cache_path=str(cache), text_model_path=str(model))
            for launcher in (regular, independent):
                result = launcher.inspect_cache(cfg)
                self.assertEqual(result["cache_shape"], (2, 3, 768))
                self.assertEqual(result["mask_shape"], (2, 3))
            metadata["shape"] = [2, 4, 768]
            meta.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "metadata.shape"):
                regular.inspect_cache(cfg)
            meta.unlink()
            with self.assertRaises(FileNotFoundError):
                independent.inspect_cache(cfg)


if __name__ == "__main__":
    unittest.main()
