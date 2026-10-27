#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frozen ModernBERT -> context-aware semantic bank, V2-compatible.

Key properties:
* read large JSON LIST with ijson (never json.load); preserve original row order
* tokenizer offset mapping -> semantic token spans (not decoded/re-tokenized)
* long texts: overlapping context windows + disjoint ownership, NO truncation
* pool contextualized token embeddings AFTER each encoder forward
* disk memmaps + atomic valid metadata + resumable progress checkpoint
* output exactly matches fusiformer_v2.dataset.validate_semantic_cache()

Quick dry-run (does NOT build cache):
 python tools/precompute_modernbert_late_v2.py --dry-run --scan-limit 64

Small smoke test with resumability (incomplete; not trainable until finished):
 python tools/precompute_modernbert_late_v2.py --stop-after 8
 python tools/precompute_modernbert_late_v2.py --resume
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fusiformer_v2.late_chunking import make_windows, semantic_spans

FORMAT = "fusiformer-v2-segments-v1"  # required by native V2 dataset
IMPLEMENTATION = "modernbert-late-windowed-v1"


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/dataset_final_merged_with_robocrys.json")
    p.add_argument("--model", default="/root/models/ModernBERT-base")
    p.add_argument("--output", default="data/v2/modernbert_late_segments.npy")
    p.add_argument("--text-field", default="description")
    p.add_argument("--max-segments", type=int, default=32)
    p.add_argument("--target-segment-tokens", type=int, default=160)
    p.add_argument("--window-size", type=int, default=4096, help="including special tokens; max 8192")
    p.add_argument("--overlap", type=int, default=256, help="overlapping CONTENT tokens per window")
    p.add_argument("--cache-dtype", choices=("float16", "float32"), default="float16")
    p.add_argument("--compute-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    p.add_argument("--attention", choices=("auto", "sdpa"), default="auto")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--checkpoint-every", type=int, default=50)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="inspect token/window counts, no model loaded")
    p.add_argument("--scan-limit", type=int, default=64)
    p.add_argument("--stop-after", type=int, default=0, help="after this many *new* rows save partial state; resume later")
    return p.parse_args(argv)


def stream_items(path: Path):
    try:
        import ijson
    except ImportError as exc:
        raise RuntimeError("Install stream parser first: python -m pip install ijson") from exc
    with path.open("rb") as stream:
        yield from ijson.items(stream, "item")


def normalize_text(value) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def fingerprint_and_count(path: Path, field: str) -> tuple[int, str]:
    """Bit-for-bit compatible with loader.loader_cached._text_fingerprint."""
    hasher = hashlib.sha256()
    count = 0
    for count, item in enumerate(stream_items(path), start=1):
        if not isinstance(item, dict):
            raise ValueError(f"dataset row {count-1} is not an object")
        encoded = normalize_text(item.get(field, "")).encode("utf-8", errors="replace")
        hasher.update((count - 1).to_bytes(8, "little", signed=False))
        hasher.update(len(encoded).to_bytes(8, "little", signed=False))
        hasher.update(encoded)
    if count == 0:
        raise ValueError("dataset JSON array is empty")
    return count, hasher.hexdigest()


def atomic_json(path: Path, payload: dict):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def tokenize_document(text: str, tokenizer):
    if not text.strip():
        return [], []
    batch = tokenizer(text, add_special_tokens=False, truncation=False,
                      return_offsets_mapping=True, return_attention_mask=False)
    ids, offsets = batch["input_ids"], batch["offset_mapping"]
    if len(ids) != len(offsets):
        raise ValueError("token IDs/offset mapping mismatch")
    return ids, offsets


def document_embedding(text: str, tokenizer, model, *, device: torch.device,
                       window_size: int, overlap: int, max_segments: int,
                       target_segment_tokens: int, cache_dtype: np.dtype):
    """Encode one full description; no gradients or silent truncation.

    Window overlap is used as context only. For each token we pool from one
    owning window; hence all tokens contribute ONCE, including long tails.
    """
    ids, offsets = tokenize_document(text, tokenizer)
    hidden_size = int(model.config.hidden_size)
    out = np.zeros((max_segments, hidden_size), dtype=cache_dtype)
    mask = np.zeros((max_segments,), dtype=np.bool_)
    if not ids:
        return out, mask, 0, 0

    spans = semantic_spans(text, offsets, max_segments=max_segments,
                           target_tokens=target_segment_tokens)
    max_index = np.array([end for _, end in spans], dtype=np.int64)
    specials = int(tokenizer.num_special_tokens_to_add(pair=False))
    if window_size <= specials:
        raise ValueError("window-size is smaller than special token budget")
    windows = make_windows(len(ids), content_tokens=window_size-specials, overlap=overlap)

    sums = torch.zeros((len(spans), hidden_size), dtype=torch.float32, device="cpu")
    counts = torch.zeros(len(spans), dtype=torch.long, device="cpu")

    for win in windows:
        chunk_ids = ids[win.input_start:win.input_end]
        # ModernBERT's *fast* tokenizer cannot call get_special_tokens_mask(...,
        # already_has_special_tokens=False). It places [CLS] before the content
        # tokens and [SEP] after them. Verify that template explicitly before
        # indexing hidden states; silently misaligned pooling would corrupt
        # every downstream semantic embedding.
        model_ids = tokenizer.build_inputs_with_special_tokens(chunk_ids)
        if (
            tokenizer.cls_token_id is None
            or tokenizer.sep_token_id is None
            or len(model_ids) != len(chunk_ids) + 2
            or model_ids[0] != tokenizer.cls_token_id
            or model_ids[-1] != tokenizer.sep_token_id
            or model_ids[1:-1] != chunk_ids
        ):
            raise RuntimeError(
                "Unexpected ModernBERT special-token layout. Expected "
                "[CLS] + content IDs + [SEP]; refusing misaligned late pooling."
            )
        content_positions = range(1, len(chunk_ids) + 1)
        encoded_ids = torch.tensor([model_ids], dtype=torch.long, device=device)
        attention = torch.ones_like(encoded_ids)
        with torch.inference_mode():
            result = model(input_ids=encoded_ids, attention_mask=attention,
                           return_dict=True).last_hidden_state[0]
            positions = torch.tensor(content_positions, dtype=torch.long, device=device)
            content_hidden = result.index_select(0, positions)
            own_lo = win.own_start - win.input_start
            own_hi = win.own_end - win.input_start
            owned = content_hidden[own_lo:own_hi].to(torch.float32)
            idx = np.arange(win.own_start, win.own_end)
            seg_ids = np.searchsorted(max_index, idx, side="right").astype(np.int64)
            if np.any(seg_ids >= len(spans)):
                raise AssertionError("segment ownership out of range")
            index = torch.from_numpy(seg_ids).to(device)
            partial = torch.zeros((len(spans), hidden_size), dtype=torch.float32, device=device)
            partial.index_add_(0, index, owned)
            local_counts = torch.bincount(index, minlength=len(spans))
            sums += partial.cpu()
            counts += local_counts.cpu()

    if int(counts.sum()) != len(ids):
        raise AssertionError("some source text tokens were skipped or double-counted")
    values = sums / counts.clamp(min=1).unsqueeze(-1)
    if not torch.isfinite(values).all():
        raise RuntimeError("ModernBERT returned NaN/Inf; aborting cache generation")
    out[:len(spans)] = values.numpy().astype(cache_dtype)
    mask[:len(spans)] = True
    return out, mask, len(ids), len(windows)


def model_files_fingerprint(model_dir: Path) -> str:
    """Hash actual weights AND tokenizer for safe resume/reproducible caches."""
    wanted = [model_dir / "config.json", model_dir / "tokenizer.json",
              model_dir / "tokenizer_config.json"]
    wanted += sorted(model_dir.glob("*.safetensors"))
    wanted += sorted(model_dir.glob("pytorch_model*.bin"))
    if not any(p.suffix in {".safetensors", ".bin"} for p in wanted if p.is_file()):
        raise FileNotFoundError("No checkpoint weights (.safetensors or .bin) found in local model directory")
    h = hashlib.sha256()
    for file in wanted:
        if not file.is_file():
            continue
        h.update(file.name.encode("utf-8"))
        with file.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                h.update(block)
    return h.hexdigest()


def make_signature(args, *, count: int, fingerprint: str, hidden: int, model_revision: str):
    return {
        "implementation": IMPLEMENTATION,
        "num_samples": count,
        "text_field": args.text_field,
        "text_sha256": fingerprint,
        "model_name": str(Path(args.model).resolve()),
        "model_revision": model_revision,
        "shape": [count, args.max_segments, hidden],
        "max_segments": args.max_segments,
        "target_segment_tokens": args.target_segment_tokens,
        "window_size": args.window_size,
        "overlap": args.overlap,
        "cache_dtype": args.cache_dtype,
    }


def main(argv=None):
    args = arguments(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for key in ("max_segments", "target_segment_tokens", "window_size", "checkpoint_every"):
        if getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive")
    if args.overlap < 0 or args.stop_after < 0:
        raise ValueError("overlap and stop-after cannot be negative")
    model_dir = Path(args.model).expanduser().resolve()
    data_path = Path(args.data).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not model_dir.is_dir() or not data_path.is_file():
        raise FileNotFoundError("Expected local ModernBERT model directory and JSON dataset")
    from transformers import AutoConfig, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True,
                                              local_files_only=True, trust_remote_code=False)
    if not tokenizer.is_fast:
        raise ValueError("ModernBERT Late Chunking requires a fast tokenizer with offset mapping")
    config = AutoConfig.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    if config.model_type != "modernbert":
        raise ValueError(f"Not a ModernBERT checkpoint: {config.model_type}")
    hidden = int(config.hidden_size)
    if args.window_size > int(config.max_position_embeddings):
        raise ValueError(f"window-size > model capacity {config.max_position_embeddings}")
    if args.overlap >= args.window_size - tokenizer.num_special_tokens_to_add(pair=False):
        raise ValueError("overlap too large for chosen window")

    if args.dry_run:
        print(f"[DRY RUN] tokenizer={model_dir} max_window={args.window_size}", flush=True)
        samples = []
        for i, item in enumerate(stream_items(data_path)):
            text = normalize_text(item.get(args.text_field, ""))
            ids, offsets = tokenize_document(text, tokenizer)
            spans = semantic_spans(text, offsets, max_segments=args.max_segments,
                                   target_tokens=args.target_segment_tokens)
            windows = make_windows(len(ids), content_tokens=args.window_size-
                                   tokenizer.num_special_tokens_to_add(pair=False), overlap=args.overlap)
            samples.append((len(ids), len(spans), len(windows)))
            if i + 1 >= args.scan_limit:
                break
        if not samples:
            raise ValueError("dataset empty")
        sample_array = np.array(samples, dtype=np.int64)
        print(f"[DRY RUN] sampled={len(samples)} tokens(mean/max)="
              f"{sample_array[:,0].mean():.1f}/{sample_array[:,0].max()} "
              f"segments(max)={sample_array[:,1].max()} "
              f"windows(max)={sample_array[:,2].max()}", flush=True)
        print("[DRY RUN] no cache created; output is not a complete-dataset statistic")
        return

    count, fingerprint = fingerprint_and_count(data_path, args.text_field)
    output.parent.mkdir(parents=True, exist_ok=True)
    mask_path = output.with_name(output.stem + "_mask.npy")
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    unfinished = output.with_name(output.stem + ".incomplete.npy")
    unfinished_mask = mask_path.with_name(mask_path.stem + ".incomplete.npy")
    state_path = output.with_name(output.name + ".progress.json")
    if output.exists() or mask_path.exists() or meta_path.exists():
        raise FileExistsError("Final cache already exists: use a NEW output filename; existing cache is protected")

    # One streaming checksum also detects weights/tokenizer replaced in-place.
    model_digest = model_files_fingerprint(model_dir)
    signature = make_signature(args, count=count, fingerprint=fingerprint,
                               hidden=hidden, model_revision=model_digest)
    cache_dtype = np.float16 if args.cache_dtype == "float16" else np.float32
    if args.resume:
        if not (state_path.is_file() and unfinished.is_file() and unfinished_mask.is_file()):
            raise FileNotFoundError("No valid incomplete files for --resume")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("signature") != signature:
            raise ValueError("Resume parameters/dataset/model mismatch; refusing stale cache")
        start_row = int(state["next_row"])
        vectors = np.lib.format.open_memmap(unfinished, mode="r+")
        masks = np.lib.format.open_memmap(unfinished_mask, mode="r+")
    else:
        if any(p.exists() for p in (state_path, unfinished, unfinished_mask)):
            raise FileExistsError("Incomplete cache exists: use --resume or choose another output path")
        start_row = 0
        vectors = np.lib.format.open_memmap(unfinished, mode="w+", dtype=cache_dtype,
                                             shape=tuple(signature["shape"]))
        masks = np.lib.format.open_memmap(unfinished_mask, mode="w+", dtype=np.bool_,
                                           shape=(count, args.max_segments))
        vectors[:] = 0
        masks[:] = False
        vectors.flush(); masks.flush()
        atomic_json(state_path, {"signature": signature, "next_row": 0})
    if (vectors.shape != tuple(signature["shape"]) or masks.shape != (count, args.max_segments)
            or vectors.dtype != cache_dtype or masks.dtype != np.dtype("bool")):
        raise ValueError("Resumed memmap shape/dtype does not match current configuration")
    if not (0 <= start_row <= count):
        raise ValueError("Invalid resume checkpoint index")
    print(f"[INFO] documents={count:,} rows completed={start_row:,}; cache={output}", flush=True)
    print(f"[INFO] model={model_dir}, frozen=True, device={args.device}, window={args.window_size}", flush=True)

    from transformers import AutoModel
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.compute_dtype]
    if args.device == "cpu" and dtype != torch.float32:
        raise ValueError("Use --compute-dtype float32 on CPU")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    load_kwargs = dict(local_files_only=True, trust_remote_code=False, dtype=dtype)
    if args.attention != "auto":
        load_kwargs["attn_implementation"] = args.attention
    model = AutoModel.from_pretrained(str(model_dir), **load_kwargs).to(args.device).eval()
    model.requires_grad_(False)
    if int(model.config.hidden_size) != hidden:
        raise ValueError("Actual model hidden size differs from checkpoint config")
    device = torch.device(args.device)

    new_rows = 0
    total_tokens = 0
    window_count = 0
    start_time = time.monotonic()
    for i, row in enumerate(stream_items(data_path)):
        if i < start_row:
            continue
        text = normalize_text(row.get(args.text_field, ""))
        result, mask, tokens, windows = document_embedding(
            text, tokenizer, model, device=device, window_size=args.window_size,
            overlap=args.overlap, max_segments=args.max_segments,
            target_segment_tokens=args.target_segment_tokens, cache_dtype=cache_dtype,
        )
        vectors[i] = result
        masks[i] = mask
        new_rows += 1
        total_tokens += tokens
        window_count += windows
        should_save = ((i + 1) % args.checkpoint_every == 0 or i + 1 == count
                       or (args.stop_after and new_rows >= args.stop_after))
        if should_save:
            vectors.flush(); masks.flush()
            atomic_json(state_path, {"signature": signature, "next_row": i + 1})
            elapsed = max(time.monotonic() - start_time, 0.001)
            print(f"[PROGRESS] {i+1:,}/{count:,} | new={new_rows:,} "
                  f"tok={total_tokens:,} windows={window_count:,} "
                  f"{new_rows/elapsed:.2f} docs/s", flush=True)
        if args.stop_after and new_rows >= args.stop_after:
            print("[PAUSED] incomplete cache saved; rerun with --resume. NOT valid for training.")
            return

    vectors.flush(); masks.flush()
    del vectors, masks
    os.replace(unfinished, output)
    os.replace(unfinished_mask, mask_path)
    # Keep exactly the keys expected by the CURRENT native V2 loader.
    meta = dict(signature)
    meta.update({
        "cache_format": FORMAT, "encoder_name": "modernbert", "pooling": "mean",
        "preprocessing": "none", "hidden_size": hidden,
        "window_pooling": "late_chunking_central_ownership",
        "special_tokens_added_per_window": tokenizer.num_special_tokens_to_add(pair=False),
        "completed_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    })
    atomic_json(meta_path, meta)  # FINAL VALIDITY MARKER WRITTEN LAST
    state_path.unlink(missing_ok=True)
    print(f"[SUCCESS] {output} shape={tuple(signature['shape'])} dtype={cache_dtype}", flush=True)
    print(f"[SUCCESS] {mask_path} | {meta_path}", flush=True)


if __name__ == "__main__":
    main()
