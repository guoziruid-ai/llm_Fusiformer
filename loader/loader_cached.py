#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Encoder-agnostic cache-aware dataset loader for llm_Fusiformer.

Cache convention
----------------
The .npy cache is aligned to the ORIGINAL top-level JSON order:
    cache[i] <-> data[i]
Each row is the pooled text representation BEFORE model.text_proj.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import dgl
import numpy as np
import torch
from jarvis.core.atoms import Atoms

from models.text_encoders import build_tokenizer_and_preprocessor
from .loader import CrystalDataset, CrystalGraphDataset, split_train_val_test
from .transforms import GraphCollecting


class GraphCollectingCached(GraphCollecting):
    """Original GraphCollecting plus propagation of precomputed text embeddings."""

    def __call__(self, crystal):
        inputs, targets = super().__call__(crystal)
        if "text_emb" in crystal:
            inputs["text_emb"] = crystal["text_emb"]
        return inputs, targets


def _default_meta_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".meta.json")


def _text_fingerprint(data, text_field: str) -> str:
    """Fingerprint raw text + order; preprocessing settings live in metadata."""
    h = hashlib.sha256()
    for i, item in enumerate(data):
        text = item.get(text_field, "")
        if text is None:
            text = ""
        if not isinstance(text, str):
            text = str(text)
        payload = text.encode("utf-8", errors="replace")
        h.update(i.to_bytes(8, "little", signed=False))
        h.update(len(payload).to_bytes(8, "little", signed=False))
        h.update(payload)
    return h.hexdigest()


def _validate_cache(
    cache_path: str | Path,
    *,
    data,
    text_field: str,
    text_max_length: int,
    meta_path: str | Path | None = None,
    text_encoder_name: str | None = None,
    text_model_name: str | None = None,
    text_pooling: str | None = None,
    text_preprocessing: str | None = None,
) -> tuple[Path, int, dict]:
    cache_path = Path(cache_path).expanduser().resolve()
    if not cache_path.is_file():
        raise FileNotFoundError(f"text embedding cache does not exist: {cache_path}")

    arr = np.load(cache_path, mmap_mode="r")
    if arr.ndim != 2:
        raise ValueError(
            f"text embedding cache must be rank-2 [N, D], got shape={arr.shape}"
        )
    total_samples = len(data)
    if int(arr.shape[0]) != int(total_samples):
        raise ValueError(
            "text embedding cache sample count mismatch: "
            f"cache={arr.shape[0]}, dataset={total_samples}"
        )
    hidden_size = int(arr.shape[1])
    del arr

    meta = Path(meta_path).expanduser().resolve() if meta_path else _default_meta_path(cache_path)
    info: dict = {}
    if meta.is_file():
        with meta.open("r", encoding="utf-8") as f:
            info = json.load(f)

        if "num_samples" in info and int(info["num_samples"]) != total_samples:
            raise ValueError(
                "cache metadata num_samples mismatch: "
                f"{info['num_samples']} != {total_samples}"
            )

        metadata_hidden = info.get("hidden_size", info.get("qwen_hidden_size"))
        if metadata_hidden is not None and int(metadata_hidden) != hidden_size:
            raise ValueError(
                "cache metadata hidden size mismatch: "
                f"{metadata_hidden} != {hidden_size}"
            )

        if "text_field" in info and str(info["text_field"]) != str(text_field):
            raise ValueError(
                "cache metadata text_field mismatch: "
                f"{info['text_field']!r} != {text_field!r}"
            )
        if "text_max_length" in info and int(info["text_max_length"]) != int(text_max_length):
            raise ValueError(
                "cache metadata text_max_length mismatch: "
                f"{info['text_max_length']} != {text_max_length}"
            )
        if "text_sha256" in info:
            current_fingerprint = _text_fingerprint(data, text_field)
            if str(info["text_sha256"]) != current_fingerprint:
                raise ValueError(
                    "dataset text/order fingerprint does not match the cache. "
                    "Regenerate the text embedding cache."
                )

        expected = {
            "encoder_name": text_encoder_name,
            "pooling": text_pooling,
            "preprocessing": text_preprocessing,
        }
        for key, wanted in expected.items():
            if wanted is None or str(wanted).lower() == "auto" or key not in info:
                continue
            if str(info[key]).strip().lower() != str(wanted).strip().lower():
                raise ValueError(
                    f"cache metadata {key} mismatch: {info[key]!r} != {wanted!r}"
                )

        if text_model_name is not None and "model_name" in info:
            # Paths can legitimately move between machines, so only compare the
            # basename for local paths; exact remote IDs compare directly.
            saved = str(info["model_name"])
            wanted = str(text_model_name)
            saved_base = Path(saved).name
            wanted_base = Path(wanted).name
            if saved != wanted and saved_base != wanted_base:
                raise ValueError(
                    f"cache metadata model_name mismatch: {saved!r} != {wanted!r}"
                )

    return cache_path, hidden_size, info


class CachedCrystalGraphDataset(CrystalGraphDataset):
    """CrystalGraphDataset with mmap text embedding support."""

    def __init__(
        self,
        data,
        transforms,
        *,
        original_indices,
        text_embedding_cache_path: str | Path | None = None,
        cutoff=8,
        max_neighbors=12,
        line_graph=False,
        text_tokenizer=None,
        text_preprocessor=None,
        text_max_length=512,
        text_field="description",
    ):
        super().__init__(
            data=data,
            transforms=transforms,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            line_graph=line_graph,
            text_tokenizer=text_tokenizer,
            text_max_length=text_max_length,
            text_field=text_field,
        )
        if len(data) != len(original_indices):
            raise ValueError(
                f"data/original_indices length mismatch: {len(data)} != {len(original_indices)}"
            )
        self.original_indices = list(original_indices)
        self.text_preprocessor = text_preprocessor
        self.text_embedding_cache_path = (
            None
            if text_embedding_cache_path is None
            else str(Path(text_embedding_cache_path).expanduser().resolve())
        )
        self._text_embedding_mmap = None

    def _get_text_embedding(self, subset_index: int) -> torch.Tensor:
        if self.text_embedding_cache_path is None:
            raise RuntimeError("text_embedding_cache_path is not configured")
        if self._text_embedding_mmap is None:
            self._text_embedding_mmap = np.load(
                self.text_embedding_cache_path,
                mmap_mode="r",
            )
        original_index = int(self.original_indices[subset_index])
        row = np.array(
            self._text_embedding_mmap[original_index],
            dtype=np.float32,
            copy=True,
        )
        return torch.from_numpy(row)

    def __getitem__(self, index):
        info = self.data[index]
        crystal = {
            "info": info,
            "structure": Atoms.from_dict(info["atoms"]),
        }
        crystal["graph"] = self.build_graph(crystal["structure"])
        if self.line_graph:
            crystal["line_graph"] = self.build_line_graph(crystal["graph"])

        if self.text_embedding_cache_path is not None:
            crystal["text_emb"] = self._get_text_embedding(index)
        elif self.text_tokenizer is not None:
            text = info.get(self.text_field, "")
            if text is None:
                text = ""
            if not isinstance(text, str):
                text = str(text)
            if self.text_preprocessor is not None:
                text = self.text_preprocessor(text)
            tokens = self.text_tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.text_max_length,
                return_tensors="pt",
            )
            crystal["text"] = tokens

        inputs, targets = self.transforms(crystal)
        return inputs, targets

    def collect(self):
        def collect_line_graph(samples):
            inputs_list, targets_list = map(list, zip(*samples))
            graphs = [item["graph_input"] for item in inputs_list]
            batched_graph = dgl.batch([g[0] for g in graphs])
            batched_line_graph = dgl.batch([g[1] for g in graphs])

            text_data = [item.get("text") for item in inputs_list]
            if all(text is not None for text in text_data):
                batched_text = {
                    "input_ids": torch.cat([t["input_ids"] for t in text_data], dim=0),
                    "attention_mask": torch.cat(
                        [t["attention_mask"] for t in text_data], dim=0
                    ),
                }
                if all("token_type_ids" in t for t in text_data):
                    batched_text["token_type_ids"] = torch.cat(
                        [t["token_type_ids"] for t in text_data], dim=0
                    )
            else:
                batched_text = None

            text_emb_data = [item.get("text_emb") for item in inputs_list]
            batched_text_emb = (
                torch.stack(text_emb_data, dim=0)
                if all(emb is not None for emb in text_emb_data)
                else None
            )
            batched_inputs = {
                "graph_input": [batched_graph, batched_line_graph],
                "text": batched_text,
                "text_emb": batched_text_emb,
            }
            return batched_inputs, torch.stack(targets_list)

        return collect_line_graph


def get_dataset(
    data_path,
    ratio_train_val_test=None,
    n_train_val_test=None,
    transforms=None,
    graph=False,
    line_graph=False,
    # Generic names
    text_encoder_name="auto",
    text_model_name=None,
    text_pooling="mean",
    text_preprocessing="auto",
    allow_remote_model=False,
    # Legacy name retained for v4 callers
    qwen_model_name=None,
    text_max_length=512,
    text_field="description",
    text_embedding_cache_path=None,
    text_embedding_meta_path=None,
):
    if text_model_name is None:
        text_model_name = qwen_model_name or "/root/models/Qwen2.5-1.5B-Instruct"
    if str(text_encoder_name).lower() == "auto":
        lower_model = str(text_model_name).lower()
        text_encoder_name = "matscibert" if "matscibert" in lower_model else "qwen"

    with open(data_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    print(f"Loading completed. Total samples: {len(data)}")

    train_idx, val_idx, test_idx = split_train_val_test(
        total_size=len(data),
        ratio_train_val_test=ratio_train_val_test,
        n_train_val_test=n_train_val_test,
    )
    train_data = [data[idx] for idx in train_idx]
    val_data = [data[idx] for idx in val_idx] if val_idx else None
    test_data = [data[idx] for idx in test_idx] if test_idx else None

    if isinstance(transforms, list):
        assert len(transforms) <= 3
        if len(transforms) == 3:
            train_transform, val_transform, test_transform = transforms
        elif len(transforms) == 2:
            train_transform = transforms[0]
            val_transform = test_transform = transforms[-1]
        else:
            train_transform = val_transform = test_transform = transforms[0]
    else:
        train_transform = val_transform = test_transform = transforms

    cache_path = None
    tokenizer = None
    preprocessor = None
    if text_embedding_cache_path is not None:
        cache_path, hidden_size, meta = _validate_cache(
            text_embedding_cache_path,
            data=data,
            text_field=text_field,
            text_max_length=text_max_length,
            meta_path=text_embedding_meta_path,
            text_encoder_name=text_encoder_name,
            text_model_name=text_model_name,
            text_pooling=text_pooling,
            text_preprocessing=text_preprocessing,
        )
        encoder_label = meta.get("encoder_name", text_encoder_name) if meta else text_encoder_name
        print(
            "Using precomputed text embeddings: "
            f"encoder={encoder_label} | {cache_path} | shape=({len(data)}, {hidden_size})"
        )
    else:
        tokenizer, preprocessor = build_tokenizer_and_preprocessor(
            text_encoder_name,
            model_name=text_model_name,
            preprocessing=text_preprocessing,
            local_files_only=not allow_remote_model,
        )

    if graph:
        common = dict(
            line_graph=line_graph,
            text_tokenizer=tokenizer,
            text_preprocessor=preprocessor,
            text_max_length=text_max_length,
            text_field=text_field,
            text_embedding_cache_path=cache_path,
        )
        train_dataset = CachedCrystalGraphDataset(
            data=train_data,
            transforms=train_transform,
            original_indices=train_idx,
            **common,
        )
        val_dataset = (
            CachedCrystalGraphDataset(
                data=val_data,
                transforms=val_transform,
                original_indices=val_idx,
                **common,
            )
            if val_data
            else None
        )
        test_dataset = (
            CachedCrystalGraphDataset(
                data=test_data,
                transforms=test_transform,
                original_indices=test_idx,
                **common,
            )
            if test_data
            else None
        )
    else:
        train_dataset = CrystalDataset(data=train_data, transforms=train_transform)
        val_dataset = (
            CrystalDataset(data=val_data, transforms=val_transform) if val_data else None
        )
        test_dataset = (
            CrystalDataset(data=test_data, transforms=test_transform) if test_data else None
        )

    return train_dataset, val_dataset, test_dataset
