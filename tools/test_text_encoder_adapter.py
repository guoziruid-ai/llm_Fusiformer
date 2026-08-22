#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small local smoke test for the modular Fusiformer text encoder."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.text_encoders import build_text_encoder

TEXT_ENCODER_NAME = "matscibert"
TEXT_MODEL_PATH = "/root/models/matscibert"
TEXT_POOLING = "mean"
TEXT_PREPROCESSING = "matscibert"
MAX_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAMPLES = [
    "Fe2O3 crystallizes in a corundum-related structure.",
    "The material contains Fe–O coordination polyhedra and an α phase.",
]


def main():
    encoder = build_text_encoder(
        TEXT_ENCODER_NAME,
        model_name=TEXT_MODEL_PATH,
        pooling=TEXT_POOLING,
        preprocessing=TEXT_PREPROCESSING,
        freeze=True,
        local_files_only=True,
        load_tokenizer=True,
        load_model=True,
    )
    encoder.to(DEVICE)
    encoder.eval()

    print(f"encoder       : {TEXT_ENCODER_NAME}")
    print(f"model         : {TEXT_MODEL_PATH}")
    print(f"hidden_size   : {encoder.hidden_size}")
    print(f"pooling       : {encoder.pooling}")
    print(f"preprocessing : {encoder.preprocessing}")
    print(f"device        : {DEVICE}")

    tokens = encoder.tokenize(
        SAMPLES,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    print("token shapes  :", {k: tuple(v.shape) for k, v in tokens.items()})
    tokens = {k: v.to(DEVICE) for k, v in tokens.items()}

    with torch.inference_mode():
        pooled = encoder.encode_tokens(tokens)

    print(f"pooled shape  : {tuple(pooled.shape)}")
    print(f"pooled dtype  : {pooled.dtype}")
    print(f"finite        : {bool(torch.isfinite(pooled).all())}")
    print(f"mean norm     : {pooled.float().norm(dim=-1).mean().item():.6f}")

    expected = (len(SAMPLES), encoder.hidden_size)
    if tuple(pooled.shape) != expected:
        raise RuntimeError(f"unexpected output shape {tuple(pooled.shape)} != {expected}")
    if not torch.isfinite(pooled).all():
        raise RuntimeError("text encoder produced NaN/Inf")
    print("PASS: modular text encoder adapter is ready.")


if __name__ == "__main__":
    main()
