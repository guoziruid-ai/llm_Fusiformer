"""Verify migrated experiments still resolve resources from the project root.

Inspect only the path expression: importing old experiment modules may load GPU
dependencies or have side effects, neither of which is needed for this check.
"""
import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATED_ROOTS = {
    "a0_legacy/train_fusiformer_production_editor_v2.py": "base_path",
    "a0_legacy/train_fusiformer_production_editor_v3.py": "base_path",
    "a0_legacy/train_fusiformer_production_editor_v5.py": "BASE_PATH",
    "cluster_specialists/config.py": "ROOT",
    "matscibert/train_fusiformer_production_2Stages_ScratchTraining.py": "BASE_PATH",
    "matscibert/train_fusiformer_production_WeghtsMode.py": "BASE_PATH",
    "text_effect/experiment_text_effect_editor.py": "base_path",
    "text_effect/fusiformer_text_vector_effect_v5.py": "BASE_PATH",
    "text_effect/fusiformer_text_vector_shuffle_effect_experiment.py": "BASE_PATH",
    "text_effect/fusiformer_text_vector_shuffle_effect_experiment_v4.py": "BASE_PATH",
}


class MigratedExperimentRootTests(unittest.TestCase):
    def test_root_expressions_preserve_original_project_root(self):
        expected_ast = ast.dump(ast.parse(
            "Path(__file__).resolve().parents[3]", mode="eval").body)
        for relative_path, variable in MIGRATED_ROOTS.items():
            with self.subTest(path=relative_path):
                current = PROJECT_ROOT / "scripts" / "experiments" / relative_path
                old = PROJECT_ROOT / "experiments" / relative_path
                tree = ast.parse(current.read_text(encoding="utf-8"), filename=str(current))
                assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                               and any(isinstance(target, ast.Name) and target.id == variable
                                       for target in node.targets)]
                self.assertEqual(len(assignments), 1)
                expression = assignments[0].value
                self.assertEqual(ast.dump(expression), expected_ast)
                value = eval(compile(ast.Expression(expression), str(current), "eval"),
                             {"__builtins__": {}, "Path": Path, "__file__": str(current)})
                self.assertEqual(value, PROJECT_ROOT)
                self.assertEqual(value, old.resolve().parents[2])


if __name__ == "__main__":
    unittest.main()
