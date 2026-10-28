"""Read-only Python, dependency and GPU preflight for the existing environment.

Run from the repository root with:
    python -m scripts.diagnostics.check_environment
No packages are installed and no training or checkpoints are touched.
"""
from __future__ import annotations

import importlib
import os
import platform
import sys
from pathlib import Path


REQUIRED_IMPORTS = ("numpy", "torch", "dgl", "jarvis", "transformers")
REQUIRE_CUDA = True


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    print(f"project_root={root}")
    print(f"working_directory={Path.cwd()}")
    print(f"python={sys.executable}")
    print(f"python_version={platform.python_version()}")
    print(f"conda_env={os.environ.get('CONDA_DEFAULT_ENV', '<not activated>')}")
    failures: list[str] = []
    modules = {}
    for name in REQUIRED_IMPORTS:
        try:
            modules[name] = importlib.import_module(name)
            print(f"{name}={getattr(modules[name], '__version__', '<version unavailable>')}")
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    torch = modules.get("torch")
    if torch is not None:
        cuda_available = bool(torch.cuda.is_available())
        print(f"cuda_available={cuda_available}")
        if cuda_available:
            for index in range(torch.cuda.device_count()):
                print(f"gpu[{index}]={torch.cuda.get_device_name(index)}")
        elif REQUIRE_CUDA:
            failures.append("CUDA GPU is unavailable")
    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    if failures:
        return 1
    print("PASS environment preflight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
