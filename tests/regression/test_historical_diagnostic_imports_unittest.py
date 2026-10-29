"""Historical runners can be imported both as modules and from their directory.

These checks import only; they never run the configured diagnostics or main().
"""
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNERS = (
    "compare_native_stopping_to_router_20261028",
    "frozen_train_router_test_20261028",
    "run_attribution_20261028",
    "run_independent_drift_audit_20261028",
    "run_independent_drift_audit_v2_20261028",
    "run_native_stopping_audit_20261028",
    "run_stopping_audit_20261028",
)


class HistoricalDiagnosticImportsTests(unittest.TestCase):
    def check_imports(self, direct):
        code = (
            "import importlib, sys\n"
            + ("sys.path.insert(0, 'scripts/diagnostics')\n" if direct else "")
            + f"names = {RUNNERS!r}\n"
            + f"prefix = {'' if direct else 'scripts.diagnostics.'!r}\n"
            + "for name in names: importlib.import_module(prefix + name)\n"
        )
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT,
                                text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_package_imports_without_running_diagnostics(self):
        self.check_imports(direct=False)

    def test_direct_directory_imports_without_running_diagnostics(self):
        self.check_imports(direct=True)


if __name__ == "__main__":
    unittest.main()
