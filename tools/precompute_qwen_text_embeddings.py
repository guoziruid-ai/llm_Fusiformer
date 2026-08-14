#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fusiformer - Qwen Text Embedding 预计算脚本（直接修改配置版）

使用方法：
1. 修改下方“用户配置区”
2. 保存到：
   /root/llm_fusiformer/tools/precompute_qwen_text_embeddings.py
3. 直接运行：
   python tools/precompute_qwen_text_embeddings.py

无需任何命令行参数。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer


# =============================================================================
# 用户配置区 —— 只需要修改这里
# =============================================================================

DATA_PATH = "./data/my_dataset_wiki_prompt_robocrys.json"
QWEN_MODEL_PATH = "/root/models/Qwen2.5-1.5B-Instruct"

OUTPUT_PATH = (
    "./data/qwen_cache/"
    "qwen2p5_1p5b_description_meanpool_f32.npy"
)

TEXT_FIELD = "description"
MAX_LENGTH = 512

# "max_length"：与原在线训练 tokenizer 行为保持一致，推荐正式实验使用
# "dynamic"   ：按当前 batch 最长文本补齐，预计算更快
PADDING_MODE = "max_length"

# A800 80GB 建议从 64 开始；OOM 时会自动减半
BATCH_SIZE = 64

DEVICE = "cuda"

# "original" / "float32" / "float16" / "bfloat16"
MODEL_DTYPE = "original"

# "float32" / "float16"
CACHE_DTYPE = "float32"

# False：支持断点续算；已有完整 cache 时直接退出
# True ：删除旧 cache 并从头计算
OVERWRITE = False

LOG_INTERVAL = 20


# =============================================================================
# 以下通常不需要修改
# =============================================================================

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent


def resolve_path(value: str) -> Path:
    p = Path(value).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def resolve_model_dtype(name: str):
    name = str(name).strip().lower()
    if name == "original":
        return None

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    if name not in mapping:
        raise ValueError(
            "MODEL_DTYPE 必须是 original / float32 / float16 / bfloat16"
        )

    return mapping[name]


def validate_config():
    if BATCH_SIZE < 1:
        raise ValueError("BATCH_SIZE 必须 >= 1")

    if MAX_LENGTH < 1:
        raise ValueError("MAX_LENGTH 必须 >= 1")

    if LOG_INTERVAL < 1:
        raise ValueError("LOG_INTERVAL 必须 >= 1")

    if PADDING_MODE not in {"max_length", "dynamic"}:
        raise ValueError(
            "PADDING_MODE 必须是 'max_length' 或 'dynamic'"
        )

    if CACHE_DTYPE not in {"float32", "float16"}:
        raise ValueError(
            "CACHE_DTYPE 必须是 'float32' 或 'float16'"
        )

    resolve_model_dtype(MODEL_DTYPE)

    if not str(OUTPUT_PATH).lower().endswith(".npy"):
        raise ValueError("OUTPUT_PATH 必须以 .npy 结尾")


def get_text(item: dict[str, Any]) -> str:
    text = item.get(TEXT_FIELD, "")

    if text is None:
        return ""

    if not isinstance(text, str):
        return str(text)

    return text


def text_fingerprint(data: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()

    for index, item in enumerate(data):
        text = get_text(item)
        payload = text.encode("utf-8", errors="replace")

        h.update(index.to_bytes(8, "little", signed=False))
        h.update(len(payload).to_bytes(8, "little", signed=False))
        h.update(payload)

    return h.hexdigest()


def save_json_atomic(path: Path, payload: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    os.replace(tmp_path, path)


def masked_mean_pool(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    if attention_mask is None:
        return last_hidden_state.mean(dim=1).float()

    mask = (
        attention_mask
        .unsqueeze(-1)
        .expand_as(last_hidden_state)
        .float()
    )

    pooled = (
        (last_hidden_state * mask).sum(dim=1)
        / mask.sum(dim=1).clamp(min=1e-6)
    )

    return pooled.float()


def main():
    validate_config()

    data_path = resolve_path(DATA_PATH)
    qwen_path = resolve_path(QWEN_MODEL_PATH)
    output_path = resolve_path(OUTPUT_PATH)

    meta_path = output_path.with_suffix(
        output_path.suffix + ".meta.json"
    )

    progress_path = output_path.with_suffix(
        output_path.suffix + ".progress.json"
    )

    print()
    print("=" * 72)
    print("Fusiformer Qwen Text Embedding Precompute")
    print("=" * 72)

    print(f"project root  : {PROJECT_ROOT}")
    print(f"dataset       : {data_path}")
    print(f"qwen model    : {qwen_path}")
    print(f"output        : {output_path}")
    print(f"text field    : {TEXT_FIELD}")
    print(f"max length    : {MAX_LENGTH}")
    print(f"padding mode  : {PADDING_MODE}")
    print(f"batch size    : {BATCH_SIZE}")
    print(f"device        : {DEVICE}")
    print(f"model dtype   : {MODEL_DTYPE}")
    print(f"cache dtype   : {CACHE_DTYPE}")
    print(f"overwrite     : {OVERWRITE}")

    if not data_path.is_file():
        raise FileNotFoundError(
            f"dataset does not exist: {data_path}"
        )

    if not qwen_path.exists():
        raise FileNotFoundError(
            f"Qwen model does not exist: {qwen_path}"
        )

    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "DEVICE='cuda'，但 torch.cuda.is_available() == False"
        )

    if DEVICE.startswith("cuda"):
        print(
            f"GPU           : {torch.cuda.get_device_name(0)}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 72)
    print("Load dataset")
    print("=" * 72)

    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise TypeError(
            "dataset JSON 顶层必须是 list"
        )

    n = len(data)

    print(f"samples       : {n}")
    print("计算文本/顺序 SHA256 ...")

    fingerprint = text_fingerprint(data)

    print(f"text sha256   : {fingerprint}")

    print()
    print("=" * 72)
    print("Load Qwen config")
    print("=" * 72)

    config = AutoConfig.from_pretrained(
        str(qwen_path),
        trust_remote_code=True,
        local_files_only=True,
    )

    hidden_size = int(
        config.hidden_size
    )

    print(
        f"model type    : "
        f"{getattr(config, 'model_type', None)}"
    )

    print(
        f"hidden size   : {hidden_size}"
    )

    cache_np_dtype = (
        np.float32
        if CACHE_DTYPE == "float32"
        else np.float16
    )

    print()
    print("=" * 72)
    print("Prepare cache")
    print("=" * 72)

    start_index = 0

    if output_path.exists() and not OVERWRITE:
        mmap = np.lib.format.open_memmap(
            output_path,
            mode="r+",
        )

        expected_shape = (
            n,
            hidden_size,
        )

        if tuple(mmap.shape) != expected_shape:
            raise ValueError(
                f"已有 cache shape={mmap.shape}，"
                f"期望={expected_shape}"
            )

        if progress_path.exists():
            with progress_path.open(
                "r",
                encoding="utf-8",
            ) as f:
                progress = json.load(f)

            if (
                progress.get("text_sha256")
                != fingerprint
            ):
                raise RuntimeError(
                    "当前数据集文本或顺序与已有 cache 不一致。\n"
                    "如确认要重新生成，请设置 OVERWRITE = True。"
                )

            start_index = int(
                progress.get(
                    "next_index",
                    0,
                )
            )

            print(
                f"发现断点，继续："
                f"{start_index}/{n}"
            )

        elif meta_path.exists():
            print(
                "cache 已经完整生成。"
            )

            print(
                f"cache    : {output_path}"
            )

            print(
                f"metadata : {meta_path}"
            )

            return

        else:
            raise RuntimeError(
                "检测到已有 .npy，但不存在 progress/meta 文件。\n"
                "如需从头生成，请设置 OVERWRITE = True。"
            )

    else:
        if OVERWRITE:
            for p in (
                output_path,
                meta_path,
                progress_path,
            ):
                if p.exists():
                    p.unlink()

        mmap = np.lib.format.open_memmap(
            output_path,
            mode="w+",
            dtype=cache_np_dtype,
            shape=(
                n,
                hidden_size,
            ),
        )

        save_json_atomic(
            progress_path,
            {
                "next_index": 0,
                "num_samples": n,
                "hidden_size": hidden_size,
                "text_sha256": fingerprint,
            },
        )

        print(
            "已创建新的 embedding cache。"
        )

    print()
    print("=" * 72)
    print("Load tokenizer")
    print("=" * 72)

    tokenizer = AutoTokenizer.from_pretrained(
        str(qwen_path),
        trust_remote_code=True,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    print(
        f"pad token     : "
        f"{tokenizer.pad_token!r}"
    )

    print()
    print("=" * 72)
    print("Load Qwen")
    print("=" * 72)

    model_kwargs = {
        "trust_remote_code": True,
        "local_files_only": True,
        "output_hidden_states": False,
    }

    model_dtype = resolve_model_dtype(
        MODEL_DTYPE
    )

    if model_dtype is not None:
        model_kwargs["torch_dtype"] = (
            model_dtype
        )

    model = AutoModel.from_pretrained(
        str(qwen_path),
        **model_kwargs,
    )

    model.eval()
    model.to(
        DEVICE
    )

    actual_model_dtype = (
        next(model.parameters()).dtype
    )

    print(
        f"Qwen dtype    : "
        f"{actual_model_dtype}"
    )

    if DEVICE.startswith("cuda"):
        free_bytes, total_bytes = (
            torch.cuda.mem_get_info()
        )

        gib = 1024 ** 3

        print(
            f"CUDA free     : "
            f"{free_bytes / gib:.2f}/"
            f"{total_bytes / gib:.2f} GiB"
        )

    print()
    print("=" * 72)
    print("Start precompute")
    print("=" * 72)

    current_batch_size = int(
        BATCH_SIZE
    )

    index = start_index
    batch_counter = 0
    started = time.time()

    while index < n:
        current_size = min(
            current_batch_size,
            n - index,
        )

        texts = [
            get_text(item)
            for item
            in data[index:index + current_size]
        ]

        try:
            padding = (
                "max_length"
                if PADDING_MODE == "max_length"
                else True
            )

            tokens = tokenizer(
                texts,
                padding=padding,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )

            tokens = {
                key: value.to(
                    DEVICE,
                    non_blocking=True,
                )
                for key, value
                in tokens.items()
            }

            with torch.inference_mode():
                outputs = model(
                    input_ids=tokens[
                        "input_ids"
                    ],
                    attention_mask=tokens.get(
                        "attention_mask"
                    ),
                    output_hidden_states=False,
                    use_cache=False,
                )

                pooled = masked_mean_pool(
                    outputs.last_hidden_state,
                    tokens.get(
                        "attention_mask"
                    ),
                )

            pooled_np = (
                pooled
                .detach()
                .cpu()
                .numpy()
                .astype(
                    cache_np_dtype,
                    copy=False,
                )
            )

            mmap[
                index:index + current_size
            ] = pooled_np

            mmap.flush()

            index += current_size
            batch_counter += 1

            save_json_atomic(
                progress_path,
                {
                    "next_index": index,
                    "num_samples": n,
                    "hidden_size": hidden_size,
                    "text_sha256": fingerprint,
                    "batch_size": current_batch_size,
                },
            )

            if (
                batch_counter % LOG_INTERVAL
                == 0
                or index == n
            ):
                elapsed = (
                    time.time() - started
                )

                processed = (
                    index - start_index
                )

                speed = (
                    processed
                    / max(
                        elapsed,
                        1e-9,
                    )
                )

                remaining = (
                    n - index
                )

                eta_seconds = (
                    remaining
                    / max(
                        speed,
                        1e-9,
                    )
                )

                print(
                    f"[{index:>7}/{n}] "
                    f"{100.0 * index / max(n, 1):6.2f}%"
                    f" | {speed:7.2f} samples/s"
                    f" | ETA {eta_seconds / 60:7.1f} min"
                    f" | batch={current_batch_size}"
                )

            del (
                outputs,
                pooled,
                pooled_np,
                tokens,
            )

        except torch.OutOfMemoryError:
            if DEVICE.startswith(
                "cuda"
            ):
                torch.cuda.empty_cache()

            if current_batch_size <= 1:
                raise

            current_batch_size = max(
                1,
                current_batch_size // 2,
            )

            print()
            print(
                "CUDA OOM，自动减小 batch size："
                f"{current_batch_size}"
            )

    metadata = {
        "format_version": 3,
        "dataset_path": str(
            data_path
        ),
        "num_samples": n,
        "text_field": TEXT_FIELD,
        "text_max_length": MAX_LENGTH,
        "padding_mode": PADDING_MODE,
        "text_sha256": fingerprint,
        "qwen_model_path": str(
            qwen_path
        ),
        "qwen_model_type": getattr(
            config,
            "model_type",
            None,
        ),
        "qwen_hidden_size": hidden_size,
        "qwen_parameter_dtype": str(
            actual_model_dtype
        ),
        "cache_dtype": CACHE_DTYPE,
        "pooling": (
            "attention_mask_weighted_mean_"
            "before_text_proj"
        ),
        "output_hidden_states": False,
        "use_cache": False,
        "cache_shape": [
            n,
            hidden_size,
        ],
        "created_at_unix": time.time(),
    }

    save_json_atomic(
        meta_path,
        metadata,
    )

    if progress_path.exists():
        progress_path.unlink()

    elapsed_total = (
        time.time() - started
    )

    print()
    print("=" * 72)
    print("Completed")
    print("=" * 72)

    print(
        f"embeddings     : "
        f"{output_path}"
    )

    print(
        f"metadata       : "
        f"{meta_path}"
    )

    print(
        f"shape          : "
        f"({n}, {hidden_size})"
    )

    print(
        f"cache size     : "
        f"{output_path.stat().st_size / (1024 ** 2):.1f} MiB"
    )

    print(
        f"elapsed        : "
        f"{elapsed_total / 60:.1f} min"
    )


if __name__ == "__main__":
    main()
