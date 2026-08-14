#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--qwen-model", required=True)
    p.add_argument("--text-field", default="description")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def normalize_text(x):
    if x is None:
        return ""
    return x if isinstance(x, str) else str(x)


def masked_mean_pool(hidden, mask):
    if mask is None:
        return hidden.mean(dim=1).float()
    expanded = mask.unsqueeze(-1).expand_as(hidden).float()
    return ((hidden * expanded).sum(dim=1) /
            expanded.sum(dim=1).clamp(min=1e-6)).float()


def main():
    args = parse_args()

    data_path = Path(args.data).expanduser().resolve()
    cache_path = Path(args.cache).expanduser().resolve()
    qwen_path = Path(args.qwen_model).expanduser().resolve()

    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    cache = np.load(cache_path, mmap_mode="r")

    if cache.ndim != 2:
        raise ValueError(f"cache must be [N, D], got {cache.shape}")
    if cache.shape[0] != len(data):
        raise ValueError(
            f"cache/data sample mismatch: {cache.shape[0]} != {len(data)}"
        )

    rng = random.Random(args.seed)
    indices = rng.sample(range(len(data)), min(args.samples, len(data)))
    texts = [
        normalize_text(data[i].get(args.text_field, ""))
        for i in indices
    ]

    tokenizer = AutoTokenizer.from_pretrained(
        str(qwen_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokens = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    tokens = {k: v.to(args.device) for k, v in tokens.items()}

    model = AutoModel.from_pretrained(
        str(qwen_path),
        trust_remote_code=True,
        local_files_only=True,
        output_hidden_states=False,
    ).to(args.device)
    model.eval()

    with torch.inference_mode():
        outputs = model(
            input_ids=tokens["input_ids"],
            attention_mask=tokens.get("attention_mask"),
            output_hidden_states=False,
            use_cache=False,
        )
        online = masked_mean_pool(
            outputs.last_hidden_state,
            tokens.get("attention_mask"),
        )

    cached = torch.from_numpy(
        np.array(cache[indices], dtype=np.float32, copy=True)
    ).to(args.device)

    diff = (online - cached).abs()
    cosine = F.cosine_similarity(online, cached, dim=-1)

    print("=" * 72)
    print("Qwen Cache Consistency Check")
    print("=" * 72)
    print("indices        :", indices)
    print("shape          :", tuple(online.shape))
    print("max abs diff   :", f"{diff.max().item():.10e}")
    print("mean abs diff  :", f"{diff.mean().item():.10e}")
    print("min cosine     :", f"{cosine.min().item():.10f}")
    print("mean cosine    :", f"{cosine.mean().item():.10f}")

    if float(cosine.min().item()) < 0.99999:
        raise RuntimeError("CACHE CONSISTENCY: FAIL")

    print("CACHE CONSISTENCY: PASS")


if __name__ == "__main__":
    main()
