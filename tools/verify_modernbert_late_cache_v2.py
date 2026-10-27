#!/usr/bin/env python3
"""Low-RAM verification of completed ModernBERT cache and ordered JSON fingerprint.

Unlike the legacy V2 Dataset validator this script NEVER calls json.load() on
all 144k records. It rejects missing metadata, mismatch and NaNs/Inf.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.precompute_modernbert_late_v2 import fingerprint_and_count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/dataset_final_merged_with_robocrys.json")
    parser.add_argument("--cache", default="data/v2/modernbert_late_segments.npy")
    parser.add_argument("--batch-rows", type=int, default=128)
    args = parser.parse_args(argv)
    cache = Path(args.cache)
    mask_path = cache.with_name(cache.stem + "_mask.npy")
    meta_path = cache.with_suffix(cache.suffix + ".meta.json")
    if not (cache.is_file() and mask_path.is_file() and meta_path.is_file()):
        raise FileNotFoundError("Completed cache/mask/metadata absent; .incomplete files cannot train")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    embeddings = np.load(cache, mmap_mode="r")
    masks = np.load(mask_path, mmap_mode="r")
    if embeddings.ndim != 3 or embeddings.shape[:2] != masks.shape or masks.dtype != np.bool_:
        raise ValueError(f"Invalid arrays: cache={embeddings.shape} mask={masks.shape}")
    if meta.get("cache_format") != "fusiformer-v2-segments-v1" or meta.get("encoder_name") != "modernbert":
        raise ValueError("Metadata does not describe a native ModernBERT V2 cache")
    if list(embeddings.shape) != meta.get("shape") or embeddings.shape[2] != meta.get("hidden_size"):
        raise ValueError("Metadata and tensor shapes disagree")
    count, digest = fingerprint_and_count(Path(args.data), str(meta.get("text_field", "description")))
    if count != len(embeddings) or digest != meta.get("text_sha256"):
        raise ValueError("JSON text/order fingerprint mismatch")
    valid_segments = 0
    blank_samples = 0
    for start in range(0, count, args.batch_rows):
        stop = min(count, start + args.batch_rows)
        vec = embeddings[start:stop]
        mask = masks[start:stop]
        if not np.isfinite(vec).all():
            raise ValueError(f"NaN/Inf in cache rows {start}:{stop}")
        if np.any(vec[~mask] != 0):
            raise ValueError(f"Nonzero padding features in rows {start}:{stop}")
        valid_segments += int(mask.sum())
        blank_samples += int((~mask.any(axis=1)).sum())
    print(f"[PASS] documents={count:,} cache_shape={embeddings.shape} dtype={embeddings.dtype}")
    print(f"[PASS] valid semantic segments={valid_segments:,}, empty descriptions={blank_samples:,}")
    print(f"[PASS] matching JSON text/order SHA256={digest}")


if __name__ == "__main__":
    main()
