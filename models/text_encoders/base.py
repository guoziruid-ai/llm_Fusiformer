from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import torch
from torch import nn


class BaseTextEncoder(nn.Module, ABC):
    """Common interface for Fusiformer text encoders.

    The graph model only depends on the pooled [B, D_text] representation.
    Tokenization, model-specific preprocessing and pooling live behind this
    interface.
    """

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def tokenize(
        self,
        texts: str | Sequence[str],
        *,
        max_length: int = 512,
        padding: str | bool = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def encode_tokens(self, tokens: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return pooled embeddings with shape [B, hidden_size]."""
        raise NotImplementedError

    def encode_texts(
        self,
        texts: str | Sequence[str],
        *,
        max_length: int = 512,
        padding: str | bool = True,
        truncation: bool = True,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        tokens = self.tokenize(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors="pt",
        )
        if device is not None:
            tokens = {k: v.to(device) for k, v in tokens.items()}
        return self.encode_tokens(tokens)
