"""Layout and legacy-path regression without importing historical GPU scripts."""
from __future__ import annotations

import importlib.util
import io
import runpy
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from scripts.evaluate import test_fusiformer_v2_modernbert as canonical_test
from scripts.evaluate import validate_fusiformer_v2_modernbert as canonical_validation
from scripts.train import train_fusiformer_v2_modernbert as canonical_train
from scripts.train import train_fusiformer_v2_modernbert_independent as canonical_independent
from scripts.data import check_v2_assets


ROOT = Path(__file__).resolve().parents[2]
LEGACY_MOVES = {
    "analyze_text_effect_results.py": "scripts.experiments.text_effect.analyze_text_effect_results",
    "analyze_text_effect_results_robo_wiki.py": "scripts.experiments.text_effect.analyze_text_effect_results_robo_wiki",
    "experiment_text_effect_editor.py": "scripts.experiments.text_effect.experiment_text_effect_editor",
    "fusiformer_text_vector_effect_v5.py": "scripts.experiments.text_effect.fusiformer_text_vector_effect_v5",
    "fusiformer_text_vector_shuffle_effect_experiment.py": "scripts.experiments.text_effect.fusiformer_text_vector_shuffle_effect_experiment",
    "fusiformer_text_vector_shuffle_effect_experiment_v4.py": "scripts.experiments.text_effect.fusiformer_text_vector_shuffle_effect_experiment_v4",
    "train_fusiformer_production_2Stages_ScratchTraining.py": "scripts.experiments.matscibert.train_fusiformer_production_2Stages_ScratchTraining",
    "train_fusiformer_production_WeghtsMode.py": "scripts.experiments.matscibert.train_fusiformer_production_WeghtsMode",
    "train_fusiformer_production_editor_v2.py": "scripts.experiments.a0_legacy.train_fusiformer_production_editor_v2",
    "train_fusiformer_production_editor_v3.py": "scripts.experiments.a0_legacy.train_fusiformer_production_editor_v3",
    "train_fusiformer_production_editor_v5.py": "scripts.experiments.a0_legacy.train_fusiformer_production_editor_v5",
    "main_crysformer.py": "scripts.experiments.qwen_legacy.main_crysformer",
    "test_dataset_effect_with_logging.py": "scripts.experiments.qwen_legacy.test_dataset_effect_with_logging",
    "debug_dataloader_batch.py": "scripts.diagnostics.legacy.debug_dataloader_batch",
    "debug_engine_to_device.py": "scripts.diagnostics.legacy.debug_engine_to_device",
    "debug_model_forward.py": "scripts.diagnostics.legacy.debug_model_forward",
    "test_qwen_local.py": "scripts.diagnostics.legacy.test_qwen_local",
}


class EntryLayoutTests(unittest.TestCase):
    def test_formal_entries_keep_project_paths_and_canonical_config(self):
        self.assertIsInstance(canonical_train.CFG, canonical_train.Config)
        for module in (canonical_train, canonical_independent, canonical_validation, canonical_test):
            self.assertEqual(module.ROOT, ROOT)
            self.assertTrue(callable(module.main))
        self.assertNotEqual(canonical_validation.OUTPUT_CONFIG["directory"],
                            canonical_test.OUTPUT_CONFIG["directory"])
        for removed in ("train_fusiformer_v2_modernbert.py",
                        "train_fusiformer_v2_modernbert_independent.py",
                        "validate_fusiformer_v2_modernbert.py",
                        "test_fusiformer_v2_modernbert.py"):
            self.assertFalse((ROOT / removed).exists(), removed)

    def test_legacy_modules_have_one_canonical_implementation(self):
        for legacy, target in LEGACY_MOVES.items():
            with self.subTest(legacy=legacy):
                wrapper = ROOT / legacy
                self.assertIsNotNone(importlib.util.find_spec(target))
                if legacy == "main_crysformer.py":
                    self.assertTrue(wrapper.is_file())
                    text = wrapper.read_text(encoding="utf-8")
                    self.assertIn(target, text)
                    self.assertLess(len(text.splitlines()), 15)
                else:
                    self.assertFalse(wrapper.exists(), legacy)

    def test_text_diagnostic_canonical_entry_with_in_memory_json(self):
        self.assertFalse((ROOT / "debug_bad_text_samples.py").exists())
        sample = '[{"description":"ok"},{"description":""},{"description":17},{"unknown":1}]'
        with mock.patch("builtins.open", mock.mock_open(read_data=sample)), \
             redirect_stdout(io.StringIO()) as output:
            runpy.run_module("scripts.diagnostics.legacy.debug_bad_text_samples", run_name="__main__")
        self.assertIn("num records: 4", output.getvalue())
        self.assertIn("bad count: 3", output.getvalue())
        for reason in ("empty_text:description", "bad_type:description", "missing_text_key"):
            self.assertIn(reason, output.getvalue())

    def test_formal_defaults_share_asset_and_checkpoint_contract(self):
        cfg = canonical_train.CFG
        validation = canonical_validation
        testing = canonical_test
        self.assertEqual(validation.MODEL_CONFIG, testing.MODEL_CONFIG)
        self.assertEqual(validation.DATA_CONFIG, testing.DATA_CONFIG)
        self.assertEqual(validation.CHECKPOINT_CONFIG, testing.CHECKPOINT_CONFIG)
        self.assertEqual(Path(cfg.data_path), validation.DATA_CONFIG["data_path"])
        self.assertEqual(Path(cfg.text_cache_path), validation.DATA_CONFIG["text_cache_path"])
        self.assertEqual(Path(cfg.text_model_path), validation.MODEL_CONFIG["text_model_path"])
        self.assertEqual(cfg.target, validation.MODEL_CONFIG["target"])
        self.assertEqual(cfg.text_field, validation.DATA_CONFIG["text_field"])
        self.assertEqual(cfg.stage, validation.MODEL_CONFIG["stage"])
        self.assertEqual(cfg.num_queries, validation.MODEL_CONFIG["num_queries"])
        self.assertEqual(cfg.cross_dropout, validation.MODEL_CONFIG["cross_dropout"])
        self.assertEqual(cfg.data_split_ratio, validation.DATA_CONFIG["data_split_ratio"])
        self.assertEqual(cfg.data_split_counts, validation.DATA_CONFIG["data_split_counts"])
        self.assertEqual(cfg.batch_size, validation.DATA_CONFIG["batch_size"])
        self.assertEqual(cfg.device_name, validation.DATA_CONFIG["device_name"])
        self.assertTrue(validation.CHECKPOINT_CONFIG["strict"])
        self.assertEqual(validation.CHECKPOINT_CONFIG["expected_format_version"], 4)
        self.assertNotEqual(validation.OUTPUT_CONFIG["directory"], testing.OUTPUT_CONFIG["directory"])

    def test_evaluation_entries_route_to_separate_splits(self):
        for entry, split in ((canonical_validation, "validation"), (canonical_test, "test")):
            with self.subTest(split=split):
                with mock.patch("fusiformer_v2.evaluation.run_evaluation", return_value={"ok": True}) as runner:
                    with redirect_stdout(io.StringIO()):
                        entry.main()
                runner.assert_called_once_with(
                    split, entry.MODEL_CONFIG, entry.DATA_CONFIG,
                    entry.CHECKPOINT_CONFIG, entry.OUTPUT_CONFIG,
                )

    def test_asset_check_uses_validation_entry_config(self):
        self.assertIs(check_v2_assets.MODEL_CONFIG, canonical_validation.MODEL_CONFIG)
        self.assertIs(check_v2_assets.DATA_CONFIG, canonical_validation.DATA_CONFIG)
        self.assertIs(check_v2_assets.CHECKPOINT_CONFIG, canonical_validation.CHECKPOINT_CONFIG)


if __name__ == "__main__":
    unittest.main()
