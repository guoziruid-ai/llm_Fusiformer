#!/usr/bin/env python3
"""Precompute [N,M,D] semantic text embeddings for the real V2 dataset.

Use the SAME ordered dataset JSON used by production v6; this cache is NOT
interchangeable with the original [N,D] mean-pooled cache. Fully offline if
--model points at a local checkpoint. Writes cache + mask + SHA256 metadata.

Example:
 python tools/precompute_semantic_segments_v2.py \\
   --data data/dataset_final_merged_with_robocrys.json \\
   --model /root/models/matscibert --encoder matscibert \\
   --preprocessing matscibert --output data/v2/matscibert_segments.npy
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fusiformer_v2.segments import split_semantic_segments
from loader.loader_cached import _text_fingerprint
from models.text_encoders import build_text_encoder


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True, help=".npy token-bank cache path")
    p.add_argument("--model", required=True, help="local HF model folder")
    p.add_argument("--encoder", choices=["matscibert", "qwen"], default="matscibert")
    p.add_argument("--preprocessing", default="auto")
    p.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    p.add_argument("--text-field", default="description")
    p.add_argument("--max-segments", type=int, default=24)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64, help="number of segments per forward pass")
    p.add_argument("--overflow", choices=["error", "head_tail"], default="error")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["float16", "float32"], default="float32")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    if a.max_segments < 1 or a.batch_size < 1:
        raise ValueError("max-segments and batch-size must be positive")
    data_path, out_path = Path(a.data).resolve(), Path(a.output).resolve()
    mask_path = out_path.with_name(out_path.stem + "_mask.npy")
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    if not a.overwrite and any(p.exists() for p in (out_path, mask_path, meta_path)):
        raise FileExistsError("V2 cache already exists; choose a new output or use --overwrite")
    if not Path(a.model).is_dir():
        raise FileNotFoundError(f"local model missing: {a.model}")
    with data_path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("dataset JSON must be a list")
    encoder = build_text_encoder(
        a.encoder, model_name=str(Path(a.model).resolve()), pooling=a.pooling,
        preprocessing=a.preprocessing, freeze=True, local_files_only=True,
        load_tokenizer=True, load_model=True,
    ).to(a.device).eval()
    hidden = int(encoder.hidden_size)
    n, m = len(data), a.max_segments
    if n == 0:
        raise ValueError("empty dataset")
    cache_dtype = np.float16 if a.dtype == "float16" else np.float32
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Publish metadata last. Existing metadata is invalidated before an overwrite.
    if a.overwrite and meta_path.exists():
        meta_path.unlink()
    tmp = out_path.with_name(out_path.stem + ".incomplete.npy")
    tmp_mask = mask_path.with_name(mask_path.stem + ".incomplete.npy")
    tokens_out = np.lib.format.open_memmap(tmp, mode="w+", dtype=cache_dtype, shape=(n, m, hidden))
    mask_out = np.lib.format.open_memmap(tmp_mask, mode="w+", dtype=np.bool_, shape=(n, m))
    tokens_out[:] = 0
    mask_out[:] = False
    omitted_total = 0
    omitted_samples = 0
    try:
        for i, item in enumerate(data):
            segments, omitted = split_semantic_segments(
                item.get(a.text_field, ""), encoder.tokenizer,
                max_tokens=a.max_tokens, max_segments=m, overflow=a.overflow,
            )
            omitted_total += omitted
            omitted_samples += int(omitted > 0)
            for segment in segments:
                # Guard against changing token counts after official MatSciBERT
                # preprocessing. Never silently truncate a semantically selected span.
                prepared = encoder.preprocessor(segment)
                actual_ids = encoder.tokenizer.encode(prepared, add_special_tokens=True)
                if len(actual_ids) > a.max_tokens:
                    raise ValueError(
                        f"sample {i} segment exceeds max-tokens after preprocessing: "
                        f"{len(actual_ids)} > {a.max_tokens}. Lower --max-tokens "
                        "or adjust the segmenter."
                    )
            for start in range(0, len(segments), a.batch_size):
                chunk = segments[start:start + a.batch_size]
                batch = encoder.tokenize(
                    chunk, max_length=a.max_tokens, padding=True, truncation=True,
                )
                batch = {k: v.to(a.device) for k, v in batch.items()}
                with torch.inference_mode():
                    vector = encoder.encode_tokens(batch)
                vector = vector.detach().float().cpu().numpy()
                tokens_out[i, start:start + len(chunk)] = vector.astype(cache_dtype)
                mask_out[i, start:start + len(chunk)] = True
            if (i + 1) % 1000 == 0:
                tokens_out.flush(); mask_out.flush()
                print(f"Encoded {i+1}/{n}", flush=True)
        tokens_out.flush(); mask_out.flush()
    except Exception:
        del tokens_out, mask_out
        # Do not publish partial cache as valid.
        raise
    del tokens_out, mask_out
    os.replace(tmp, out_path)
    os.replace(tmp_mask, mask_path)
    metadata = {
        "cache_format": "fusiformer-v2-segments-v1",
        "num_samples": n,
        "shape": [n, m, hidden],
        "hidden_size": hidden,
        "encoder_name": a.encoder,
        "model_name": str(Path(a.model).resolve()),
        "pooling": a.pooling,
        "preprocessing": a.preprocessing,
        "text_field": a.text_field,
        "text_sha256": _text_fingerprint(data, a.text_field),
        "max_tokens": a.max_tokens,
        "max_segments": m,
        "overflow": a.overflow,
        "omitted_segments_total": omitted_total,
        "omitted_samples": omitted_samples,
    }
    tempmeta = meta_path.with_suffix(meta_path.suffix + ".tmp")
    with tempmeta.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    os.replace(tempmeta, meta_path)
    print(f"OK: {out_path} shape={[n,m,hidden]}, omitted samples={omitted_samples}")


if __name__ == "__main__":
    main()
