from __future__ import annotations

from contextlib import nullcontext
from typing import Sequence

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .base import BaseTextEncoder
from .preprocessing import build_text_preprocessor


def masked_mean_pool(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    if attention_mask is None:
        return last_hidden_state.mean(dim=1).float()
    mask = attention_mask.unsqueeze(-1).expand_as(last_hidden_state).float()
    return (
        (last_hidden_state * mask).sum(dim=1)
        / mask.sum(dim=1).clamp(min=1e-6)
    ).float()


def pool_hidden_state(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor | None,
    method: str,
) -> torch.Tensor:
    method = method.strip().lower()
    if method in {"mean", "masked_mean", "meanpool", "mean_pool"}:
        return masked_mean_pool(last_hidden_state, attention_mask)
    if method in {"cls", "first_token"}:
        return last_hidden_state[:, 0, :].float()
    raise ValueError(f"Unsupported text pooling method: {method!r}")


class HuggingFaceTextEncoder(BaseTextEncoder):
    """Generic Hugging Face encoder for Qwen, MatSciBERT and similar models."""

    def __init__(
        self,
        model_name: str,
        *,
        encoder_name: str = "auto",
        pooling: str = "mean",
        preprocessing: str = "auto",
        freeze: bool = True,
        local_files_only: bool = True,
        trust_remote_code: bool = True,
        torch_dtype: torch.dtype | None = None,
        load_tokenizer: bool = True,
        load_model: bool = True,
    ):
        super().__init__()
        self.model_name = str(model_name)
        self.encoder_name = str(encoder_name)
        self.pooling = str(pooling).strip().lower()
        self.preprocessing = str(preprocessing).strip().lower()
        self.freeze = bool(freeze)
        self.local_files_only = bool(local_files_only)
        self.trust_remote_code = bool(trust_remote_code)

        config = AutoConfig.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code,
        )
        if not hasattr(config, "hidden_size"):
            raise AttributeError(
                f"Text model config has no hidden_size: {self.model_name}"
            )
        self._hidden_size = int(config.hidden_size)

        self.preprocessor = build_text_preprocessor(
            self.preprocessing,
            text_encoder_name=self.encoder_name,
            text_model_name=self.model_name,
        )

        self.tokenizer = None
        if load_tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only,
                trust_remote_code=self.trust_remote_code,
            )
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                elif self.tokenizer.sep_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.sep_token
                else:
                    raise ValueError(
                        f"Tokenizer has no pad/eos/sep token: {self.model_name}"
                    )

        self.model = None
        if load_model:
            kwargs = dict(
                local_files_only=self.local_files_only,
                trust_remote_code=self.trust_remote_code,
                output_hidden_states=False,
            )
            if torch_dtype is not None:
                kwargs["torch_dtype"] = torch_dtype
            self.model = AutoModel.from_pretrained(self.model_name, **kwargs)
            if self.freeze:
                for param in self.model.parameters():
                    param.requires_grad = False
                self.model.eval()

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def tokenize(
        self,
        texts: str | Sequence[str],
        *,
        max_length: int = 512,
        padding: str | bool = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer was not loaded for this text encoder")
        if isinstance(texts, str):
            prepared = self.preprocessor(texts)
        else:
            prepared = [self.preprocessor(text) for text in texts]
        return self.tokenizer(
            prepared,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
        )

    def encode_tokens(self, tokens: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Transformer model was not loaded for this text encoder")
        ctx = torch.no_grad() if self.freeze else nullcontext()
        with ctx:
            model_inputs = {
                k: v
                for k, v in tokens.items()
                if k in {"input_ids", "attention_mask", "token_type_ids"}
            }
            outputs = self.model(
                **model_inputs,
                output_hidden_states=False,
                return_dict=True,
            )
            pooled = pool_hidden_state(
                outputs.last_hidden_state,
                tokens.get("attention_mask"),
                self.pooling,
            )
        return pooled

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze and self.model is not None:
            self.model.eval()
        return self
