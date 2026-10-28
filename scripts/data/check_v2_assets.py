"""Read-only V2.1 JSON, ModernBERT cache and checkpoint compatibility check.

Edit the canonical validation entry's file-internal MODEL/DATA/CHECKPOINT config,
then run: python -m scripts.data.check_v2_assets
This does not load the full dataset or write results.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fusiformer_v2.evaluation import _check_checkpoint
from fusiformer_v2.modernbert_runtime import inspect_cache
from scripts.evaluate.validate_fusiformer_v2_modernbert import (
    CHECKPOINT_CONFIG, DATA_CONFIG, MODEL_CONFIG,
)


def main() -> int:
    cfg = SimpleNamespace(
        data_path=DATA_CONFIG["data_path"],
        text_cache_path=DATA_CONFIG["text_cache_path"],
        text_model_path=MODEL_CONFIG["text_model_path"],
        text_field=DATA_CONFIG["text_field"],
        data_split_counts=DATA_CONFIG["data_split_counts"],
    )
    cache_info = inspect_cache(cfg)
    checkpoint_path = Path(CHECKPOINT_CONFIG["path"]).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    from train_fusiformer_production_editor_v4 import load_torch_file

    checkpoint = load_torch_file(checkpoint_path, map_location="cpu")
    state = _check_checkpoint(checkpoint, MODEL_CONFIG, DATA_CONFIG, CHECKPOINT_CONFIG)
    print(f"data_path={DATA_CONFIG['data_path']}")
    print(f"cache_shape={cache_info['cache_shape']} dtype={cache_info['cache_dtype']}")
    print(f"checkpoint={checkpoint_path} tensors={len(state)}")
    print("PASS V2.1 asset compatibility; no training or evaluation performed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
