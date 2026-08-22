#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generic Fusiformer text embedding precomputation.

Supports Qwen / MatSciBERT / other AutoModel-compatible encoders through the
same [N, D_text] cache contract.  The saved vectors are pooled encoder outputs
BEFORE the trainable Fusiformer text_proj layer.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.text_encoders import build_text_encoder, resolve_text_encoder_spec


# =============================================================================
# Editable configuration
# =============================================================================
DATA_PATH = "./data/dataset_updated_only.json"
TEXT_FIELD = "description"

# Use "matscibert" or "qwen".  For a custom encoder, set a custom name and
# provide TEXT_MODEL_PATH explicitly.
TEXT_ENCODER_NAME = "matscibert"
TEXT_MODEL_PATH = "/root/models/matscibert"
TEXT_POOLING = "mean"          # "mean" or "cls"
TEXT_PREPROCESSING = "matscibert"  # "matscibert", "none", or "auto"
MAX_LENGTH = 512

OUTPUT_PATH = "./data/text_cache/matscibert_description_meanpool_f32.npy"
BATCH_SIZE = 128
DEVICE = "cuda"
MODEL_DTYPE = "float32"        # float32 | float16 | bfloat16 | auto
CACHE_DTYPE = "float32"        # float32 | float16
ALLOW_REMOTE_MODEL = False
OVERWRITE = False


# =============================================================================
# Utilities
# =============================================================================
def resolve_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def text_fingerprint(data: list[dict]) -> str:
    h = hashlib.sha256()
    for i, item in enumerate(data):
        text = item.get(TEXT_FIELD, "")
        if text is None:
            text = ""
        if not isinstance(text, str):
            text = str(text)
        payload = text.encode("utf-8", errors="replace")
        h.update(i.to_bytes(8, "little", signed=False))
        h.update(len(payload).to_bytes(8, "little", signed=False))
        h.update(payload)
    return h.hexdigest()


def resolve_torch_dtype(name: str) -> torch.dtype | None:
    key = str(name).strip().lower()
    if key in {"auto", "none"}:
        return None
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported MODEL_DTYPE={name!r}")
    return mapping[key]


def resolve_numpy_dtype(name: str):
    key = str(name).strip().lower()
    if key in {"float32", "fp32"}:
        return np.float32
    if key in {"float16", "fp16"}:
        return np.float16
    raise ValueError(f"Unsupported CACHE_DTYPE={name!r}")


def atomic_json_dump(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def validate_configuration() -> None:
    if BATCH_SIZE < 1:
        raise ValueError("BATCH_SIZE must be >= 1")
    if MAX_LENGTH < 1:
        raise ValueError("MAX_LENGTH must be >= 1")
    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("DEVICE is CUDA but torch.cuda.is_available() is False")
    if TEXT_POOLING.strip().lower() not in {
        "mean", "masked_mean", "meanpool", "mean_pool", "cls", "first_token"
    }:
        raise ValueError("TEXT_POOLING must be mean or cls")


def main() -> None:
    validate_configuration()

    data_path = resolve_path(DATA_PATH)
    output_path = resolve_path(OUTPUT_PATH)
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    progress_path = output_path.with_suffix(output_path.suffix + ".progress.json")

    if not data_path.is_file():
        raise FileNotFoundError(f"dataset does not exist: {data_path}")

    spec = resolve_text_encoder_spec(
        TEXT_ENCODER_NAME,
        model_name=TEXT_MODEL_PATH,
        pooling=TEXT_POOLING,
        preprocessing=TEXT_PREPROCESSING,
        max_length=MAX_LENGTH,
    )
    model_name = spec.model_name
    if not ALLOW_REMOTE_MODEL:
        local_model = resolve_path(model_name)
        if not local_model.exists():
            raise FileNotFoundError(
                f"local text model does not exist: {local_model}. "
                "Download MatSciBERT first or set ALLOW_REMOTE_MODEL=True."
            )
        model_name = str(local_model)

    print("=" * 78)
    print("Fusiformer Generic Text Embedding Precompute")
    print("=" * 78)
    print(f"dataset        : {data_path}")
    print(f"encoder        : {spec.name}")
    print(f"model          : {model_name}")
    print(f"pooling        : {spec.pooling}")
    print(f"preprocessing  : {spec.preprocessing}")
    print(f"text field     : {TEXT_FIELD}")
    print(f"max length     : {MAX_LENGTH}")
    print(f"batch size     : {BATCH_SIZE}")
    print(f"device         : {DEVICE}")
    print(f"model dtype    : {MODEL_DTYPE}")
    print(f"cache dtype    : {CACHE_DTYPE}")
    print(f"output         : {output_path}")

    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError("dataset JSON top-level object must be a list")
    n = len(data)
    fingerprint = text_fingerprint(data)
    print(f"samples        : {n}")
    print(f"text sha256    : {fingerprint}")

    model_dtype = resolve_torch_dtype(MODEL_DTYPE)
    encoder = build_text_encoder(
        spec.name,
        model_name=model_name,
        pooling=spec.pooling,
        preprocessing=spec.preprocessing,
        freeze=True,
        local_files_only=not ALLOW_REMOTE_MODEL,
        torch_dtype=model_dtype,
        load_tokenizer=True,
        load_model=True,
    )
    encoder.to(DEVICE)
    encoder.eval()
    hidden_size = int(encoder.hidden_size)
    cache_dtype = resolve_numpy_dtype(CACHE_DTYPE)
    actual_dtype = next(encoder.parameters()).dtype
    print(f"hidden size    : {hidden_size}")
    print(f"actual dtype   : {actual_dtype}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    start_index = 0

    if output_path.exists() and not OVERWRITE:
        mmap = np.lib.format.open_memmap(output_path, mode="r+")
        expected = (n, hidden_size)
        if tuple(mmap.shape) != expected:
            raise ValueError(
                f"existing cache shape={mmap.shape}, expected={expected}; "
                "set OVERWRITE=True to regenerate"
            )
        if mmap.dtype != np.dtype(cache_dtype):
            raise ValueError(
                f"existing cache dtype={mmap.dtype}, expected={np.dtype(cache_dtype)}"
            )
        if progress_path.is_file():
            with progress_path.open("r", encoding="utf-8") as f:
                progress = json.load(f)
            if progress.get("text_sha256") != fingerprint:
                raise ValueError("existing progress file belongs to different text/order")
            if str(progress.get("encoder_name")) != spec.name:
                raise ValueError("existing progress file belongs to another encoder")
            start_index = int(progress.get("next_index", 0))
        elif meta_path.is_file():
            with meta_path.open("r", encoding="utf-8") as f:
                old_meta = json.load(f)
            if old_meta.get("text_sha256") == fingerprint:
                print("Cache already completed; nothing to do.")
                return
        else:
            raise RuntimeError(
                "cache exists but no progress/meta file was found. "
                "Set OVERWRITE=True if it is safe to regenerate."
            )
    else:
        if output_path.exists() and OVERWRITE:
            output_path.unlink()
        if progress_path.exists() and OVERWRITE:
            progress_path.unlink()
        if meta_path.exists() and OVERWRITE:
            meta_path.unlink()
        mmap = np.lib.format.open_memmap(
            output_path,
            mode="w+",
            dtype=cache_dtype,
            shape=(n, hidden_size),
        )

    if start_index < 0 or start_index > n:
        raise ValueError(f"invalid resume start_index={start_index}")
    print(f"resume index   : {start_index}")

    with torch.inference_mode():
        for start in range(start_index, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            texts = []
            for item in data[start:end]:
                text = item.get(TEXT_FIELD, "")
                if text is None:
                    text = ""
                if not isinstance(text, str):
                    text = str(text)
                texts.append(text)

            tokens = encoder.tokenize(
                texts,
                max_length=MAX_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            tokens = {k: v.to(DEVICE, non_blocking=True) for k, v in tokens.items()}
            pooled = encoder.encode_tokens(tokens)
            if tuple(pooled.shape) != (end - start, hidden_size):
                raise RuntimeError(
                    f"unexpected pooled shape={tuple(pooled.shape)} at [{start}:{end}]"
                )
            mmap[start:end] = pooled.detach().float().cpu().numpy().astype(cache_dtype)
            mmap.flush()

            atomic_json_dump(
                {
                    "next_index": end,
                    "num_samples": n,
                    "encoder_name": spec.name,
                    "model_name": model_name,
                    "pooling": spec.pooling,
                    "preprocessing": spec.preprocessing,
                    "hidden_size": hidden_size,
                    "text_field": TEXT_FIELD,
                    "text_max_length": MAX_LENGTH,
                    "text_sha256": fingerprint,
                },
                progress_path,
            )
            if start == start_index or end == n or ((start // BATCH_SIZE) % 25 == 0):
                print(f"[{end:>7d}/{n}] {(100.0 * end / max(n, 1)):6.2f}%")

    del mmap

    metadata = {
        "format_version": 2,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "num_samples": n,
        "encoder_name": spec.name,
        "model_name": model_name,
        "model_type": encoder.model.config.model_type if encoder.model is not None else None,
        "hidden_size": hidden_size,
        "pooling": spec.pooling,
        "preprocessing": spec.preprocessing,
        "text_field": TEXT_FIELD,
        "text_max_length": MAX_LENGTH,
        "text_sha256": fingerprint,
        "padding": "max_length",
        "cache_dtype": str(np.dtype(cache_dtype)),
        "model_dtype": str(actual_dtype).replace("torch.", ""),
        "cache_path": str(output_path),
        # Legacy field lets the old loader validate this cache too.
        "qwen_hidden_size": hidden_size,
    }
    atomic_json_dump(metadata, meta_path)
    if progress_path.exists():
        progress_path.unlink()

    print("=" * 78)
    print("Completed")
    print(f"cache          : {output_path}")
    print(f"metadata       : {meta_path}")
    print(f"shape          : ({n}, {hidden_size})")
    print("=" * 78)


if __name__ == "__main__":
    main()
