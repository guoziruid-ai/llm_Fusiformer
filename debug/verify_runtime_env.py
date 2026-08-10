import os
import sys
import subprocess
import traceback
from pathlib import Path

# ============================================================
# Paths
# ============================================================

DEBUG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEBUG_DIR.parent
QWEN_PATH = Path("/root/models/Qwen2.5-1.5B-Instruct")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def title(name):
    print()
    print("=" * 70)
    print(name)
    print("=" * 70)


def ok(msg):
    print(f"[PASS] {msg}")


def fail(msg):
    print(f"[FAIL] {msg}")


def warn(msg):
    print(f"[WARN] {msg}")


failures = []


# ============================================================
# 1. Basic environment
# ============================================================

title("1. Basic Python / Conda environment")

print("Python executable :", sys.executable)
print("Python version    :", sys.version.replace("\n", " "))
print("Project root      :", PROJECT_ROOT)
print("Conda prefix      :", os.environ.get("CONDA_PREFIX"))
print("Conda env         :", os.environ.get("CONDA_DEFAULT_ENV"))

if os.environ.get("CONDA_DEFAULT_ENV") == "fusiformer":
    ok("Current Conda environment is fusiformer")
else:
    warn("Current Conda environment is not named fusiformer")


# ============================================================
# 2. Core Python packages
# ============================================================

title("2. Core dependency imports")

packages = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("sklearn", "sklearn"),
    ("matplotlib", "matplotlib"),
    ("transformers", "transformers"),
    ("pymatgen", "pymatgen"),
    ("torchdata", "torchdata"),
]

for display_name, module_name in packages:
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "unknown")
        ok(f"{display_name:15s} {version}")
    except Exception as e:
        fail(f"{display_name:15s} {type(e).__name__}: {e}")
        failures.append(display_name)


# ============================================================
# 3. PyTorch / CUDA
# ============================================================

title("3. PyTorch / CUDA")

try:
    import torch

    print("Torch version     :", torch.__version__)
    print("Torch CUDA build  :", torch.version.cuda)
    print("CUDA available    :", torch.cuda.is_available())
    print("CUDA device count :", torch.cuda.device_count())

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() == False")

    print("Current device    :", torch.cuda.current_device())
    print("GPU name          :", torch.cuda.get_device_name(0))

    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x

    print("Test tensor       :", y.device)
    print("Test shape        :", tuple(y.shape))

    ok("PyTorch CUDA matrix multiplication")

    del x, y
    torch.cuda.empty_cache()

except Exception as e:
    fail(f"PyTorch CUDA: {type(e).__name__}: {e}")
    failures.append("torch_cuda")
    traceback.print_exc()


# ============================================================
# 4. NVIDIA driver
# ============================================================

title("4. NVIDIA driver")

try:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.free",
            "--format=csv,noheader"
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    print(result.stdout.strip())
    ok("nvidia-smi")

except Exception as e:
    warn(f"nvidia-smi check failed: {e}")


# ============================================================
# 5. DGL shared-library check
# ============================================================

title("5. DGL shared libraries")

try:
    conda_prefix = Path(os.environ["CONDA_PREFIX"])
    libdgl = (
        conda_prefix
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "dgl"
        / "libdgl.so"
    )

    print("libdgl.so :", libdgl)

    if not libdgl.exists():
        raise FileNotFoundError(libdgl)

    result = subprocess.run(
        ["ldd", str(libdgl)],
        capture_output=True,
        text=True,
        check=True,
    )

    missing = [
        line.strip()
        for line in result.stdout.splitlines()
        if "not found" in line
    ]

    if missing:
        for line in missing:
            print("[MISSING]", line)
        raise RuntimeError("DGL has missing shared libraries")

    ok("DGL shared libraries resolved")

except Exception as e:
    fail(f"DGL shared libraries: {e}")
    failures.append("dgl_shared_lib")


# ============================================================
# 6. DGL CUDA graph + message passing
# ============================================================

title("6. DGL GPU test")

try:
    import dgl
    import dgl.function as fn
    import torch

    print("DGL version       :", dgl.__version__)
    print("Torch version     :", torch.__version__)
    print("Torch CUDA        :", torch.version.cuda)

    src = torch.tensor([0, 1, 2, 0])
    dst = torch.tensor([1, 2, 0, 2])

    g = dgl.graph((src, dst))
    print("Initial graph     :", g.device)

    g = g.to("cuda")
    print("CUDA graph        :", g.device)

    g.ndata["h"] = torch.randn(
        g.num_nodes(),
        16,
        device="cuda",
    )

    g.update_all(
        fn.copy_u("h", "m"),
        fn.sum("m", "h_new"),
    )

    print("Input feature     :", g.ndata["h"].device)
    print("Output feature    :", g.ndata["h_new"].device)
    print("Output shape      :", tuple(g.ndata["h_new"].shape))

    ok("DGL GPU message passing")

except Exception as e:
    fail(f"DGL CUDA: {type(e).__name__}: {e}")
    failures.append("dgl_cuda")
    traceback.print_exc()


# ============================================================
# 7. Qwen local files
# ============================================================

title("7. Qwen local model files")

print("Qwen path         :", QWEN_PATH)

if not QWEN_PATH.exists():
    fail(f"Qwen directory does not exist: {QWEN_PATH}")
    failures.append("qwen_path")

else:
    ok("Qwen directory exists")

    important_files = [
        "config.json",
        "tokenizer_config.json",
    ]

    for filename in important_files:
        path = QWEN_PATH / filename
        if path.exists():
            ok(filename)
        else:
            fail(filename)
            failures.append(f"qwen_{filename}")

    safetensors = list(QWEN_PATH.glob("*.safetensors"))

    print("Safetensor files  :", len(safetensors))

    if safetensors:
        total_size = sum(p.stat().st_size for p in safetensors)
        print(
            "Weights size      :",
            f"{total_size / 1024**3:.2f} GB"
        )
        ok("Qwen weight files found")
    else:
        fail("No *.safetensors Qwen weight files found")
        failures.append("qwen_weights")


# ============================================================
# 8. Qwen config / tokenizer
# ============================================================

title("8. Qwen Transformers test")

if QWEN_PATH.exists():
    try:
        from transformers import AutoConfig, AutoTokenizer

        config = AutoConfig.from_pretrained(
            str(QWEN_PATH),
            trust_remote_code=True,
            local_files_only=True,
        )

        print("Model type        :", config.model_type)
        print(
            "Hidden size       :",
            getattr(config, "hidden_size", None)
        )

        tokenizer = AutoTokenizer.from_pretrained(
            str(QWEN_PATH),
            trust_remote_code=True,
            local_files_only=True,
        )

        print("Tokenizer size    :", len(tokenizer))
        print("Pad token         :", tokenizer.pad_token)
        print("EOS token         :", tokenizer.eos_token)

        test_text = "Crystal structure test."
        tokens = tokenizer(
            test_text,
            return_tensors="pt",
        )

        print("Token shape       :", tuple(tokens["input_ids"].shape))

        ok("Qwen config and tokenizer")

    except Exception as e:
        fail(f"Qwen Transformers: {type(e).__name__}: {e}")
        failures.append("qwen_transformers")
        traceback.print_exc()


# ============================================================
# 9. Project directory structure
# ============================================================

title("9. Fusiformer project files")

required_paths = [
    "engine.py",
    "load_dataset.py",
    "loader",
    "models",
    "tools",
]

for item in required_paths:
    path = PROJECT_ROOT / item

    if path.exists():
        ok(item)
    else:
        fail(item)
        failures.append(f"project_{item}")


# ============================================================
# 10. Project module import smoke test
# ============================================================

title("10. Project module imports")

modules = [
    "engine",
    "load_dataset",
]

for module_name in modules:
    try:
        module = __import__(module_name)
        ok(f"import {module_name}")
    except Exception as e:
        fail(
            f"import {module_name}: "
            f"{type(e).__name__}: {e}"
        )
        failures.append(f"import_{module_name}")
        traceback.print_exc()


# ============================================================
# 11. Final result
# ============================================================

title("FINAL RESULT")

if failures:
    print("Environment verification: FAILED")
    print()
    print("Failed items:")
    for item in failures:
        print(" -", item)

    print()
    print(f"Total failures: {len(failures)}")
    sys.exit(1)

else:
    print("Environment verification: PASS")
    print()
    print("Python           : PASS")
    print("Core packages    : PASS")
    print("PyTorch CUDA     : PASS")
    print("DGL CUDA         : PASS")
    print("Qwen files       : PASS")
    print("Qwen tokenizer   : PASS")
    print("Project imports  : PASS")
    print()
    print("Ready for project-level single-batch testing.")
    sys.exit(0)
