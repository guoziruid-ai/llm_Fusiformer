from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoConfig, AutoTokenizer

from .huggingface_encoder import HuggingFaceTextEncoder
from .preprocessing import build_text_preprocessor


@dataclass(frozen=True)
class TextEncoderSpec:
    name: str
    model_name: str
    pooling: str
    preprocessing: str
    max_length: int = 512


PRESETS: dict[str, TextEncoderSpec] = {
    "qwen2.5-1.5b": TextEncoderSpec(
        name="qwen2.5-1.5b",
        model_name="/root/models/Qwen2.5-1.5B-Instruct",
        pooling="mean",
        preprocessing="none",
        max_length=512,
    ),
    "qwen": TextEncoderSpec(
        name="qwen",
        model_name="/root/models/Qwen2.5-1.5B-Instruct",
        pooling="mean",
        preprocessing="none",
        max_length=512,
    ),
    "matscibert": TextEncoderSpec(
        name="matscibert",
        model_name="m3rg-iitd/matscibert",
        pooling="mean",
        preprocessing="matscibert",
        max_length=512,
    ),
}


def resolve_text_encoder_spec(
    encoder_name: str,
    *,
    model_name: str | None = None,
    pooling: str | None = None,
    preprocessing: str | None = None,
    max_length: int | None = None,
) -> TextEncoderSpec:
    key = str(encoder_name).strip().lower()
    preset = PRESETS.get(key)
    if preset is None:
        if model_name is None:
            raise ValueError(
                f"Unknown text encoder preset {encoder_name!r}; provide text_model_name"
            )
        preset = TextEncoderSpec(
            name=key or "custom",
            model_name=str(model_name),
            pooling="mean",
            preprocessing="auto",
            max_length=512,
        )
    return TextEncoderSpec(
        name=preset.name,
        model_name=str(model_name or preset.model_name),
        pooling=str(pooling or preset.pooling),
        preprocessing=str(preprocessing or preset.preprocessing),
        max_length=int(max_length or preset.max_length),
    )


def get_text_hidden_size(
    model_name: str,
    *,
    local_files_only: bool = True,
    trust_remote_code: bool = True,
) -> int:
    config = AutoConfig.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    if not hasattr(config, "hidden_size"):
        raise AttributeError(f"Text model config has no hidden_size: {model_name}")
    return int(config.hidden_size)


def build_text_encoder(
    encoder_name: str,
    *,
    model_name: str | None = None,
    pooling: str | None = None,
    preprocessing: str | None = None,
    freeze: bool = True,
    local_files_only: bool = True,
    trust_remote_code: bool = True,
    torch_dtype: torch.dtype | None = None,
    load_tokenizer: bool = True,
    load_model: bool = True,
) -> HuggingFaceTextEncoder:
    spec = resolve_text_encoder_spec(
        encoder_name,
        model_name=model_name,
        pooling=pooling,
        preprocessing=preprocessing,
    )
    return HuggingFaceTextEncoder(
        spec.model_name,
        encoder_name=spec.name,
        pooling=spec.pooling,
        preprocessing=spec.preprocessing,
        freeze=freeze,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        load_tokenizer=load_tokenizer,
        load_model=load_model,
    )


def build_tokenizer_and_preprocessor(
    encoder_name: str,
    *,
    model_name: str,
    preprocessing: str = "auto",
    local_files_only: bool = True,
    trust_remote_code: bool = True,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.sep_token is not None:
            tokenizer.pad_token = tokenizer.sep_token
    preprocessor = build_text_preprocessor(
        preprocessing,
        text_encoder_name=encoder_name,
        text_model_name=model_name,
    )
    return tokenizer, preprocessor
