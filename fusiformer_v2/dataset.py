"""Native GraphCollectingCached-compatible loader for [N,M,D] semantic caches.

Order is the ORIGINAL dataset JSON order (before split). The parent split
uses split_seed=7; no independent shuffling or new graph preprocessing.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from loader.loader import split_train_val_test
from loader.loader_cached import CachedCrystalGraphDataset, _text_fingerprint


def validate_semantic_cache(cache_path, *, data, text_field="description",
                            text_encoder_name=None, text_model_name=None):
    path = Path(cache_path).expanduser().resolve()
    mask_path = path.with_name(path.stem + "_mask.npy")
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if not (path.is_file() and mask_path.is_file() and meta_path.is_file()):
        raise FileNotFoundError(f"required semantic cache, mask or metadata missing: {path}")
    tokens = np.load(path, mmap_mode="r")
    mask = np.load(mask_path, mmap_mode="r")
    with meta_path.open(encoding="utf-8") as f:
        meta = json.load(f)
    if tokens.ndim != 3 or mask.shape != tokens.shape[:2] or len(data) != tokens.shape[0]:
        raise ValueError(f"cache shape mismatch tokens={tokens.shape}, mask={mask.shape}, samples={len(data)}")
    if mask.dtype != np.bool_:
        raise ValueError("semantic mask must use np.bool_ dtype")
    if meta.get("cache_format") != "fusiformer-v2-segments-v1":
        raise ValueError("cache metadata format/version is missing or incompatible")
    if meta.get("text_field") != text_field or meta.get("text_sha256") != _text_fingerprint(data, text_field):
        raise ValueError("dataset text/order fingerprint mismatch; regenerate semantic cache")
    if list(meta.get("shape", [])) != list(tokens.shape):
        raise ValueError("metadata shape mismatch")
    if int(meta.get("hidden_size", -1)) != int(tokens.shape[2]):
        raise ValueError("metadata hidden_size mismatch")
    if text_encoder_name is not None and meta.get("encoder_name") != text_encoder_name:
        raise ValueError(f"cache encoder {meta.get('encoder_name')!r} != training encoder {text_encoder_name!r}")
    if text_model_name is not None and Path(str(meta.get("model_name", ""))).name != Path(str(text_model_name)).name:
        raise ValueError("cache model checkpoint differs from training configuration")
    return str(path), str(mask_path), meta


class SemanticCrystalGraphDataset(CachedCrystalGraphDataset):
    """Keep existing structure/line-graph preprocessing; add segment embeddings."""

    def __init__(self, *, segment_cache_path, segment_mask_path, **kwargs):
        super().__init__(text_embedding_cache_path=None, text_tokenizer=None, **kwargs)
        self.segment_cache_path = str(segment_cache_path)
        self.segment_mask_path = str(segment_mask_path)
        self._seg_mem = None
        self._mask_mem = None

    def __getitem__(self, index):
        inputs, targets = super().__getitem__(index)
        if self._seg_mem is None:
            self._seg_mem = np.load(self.segment_cache_path, mmap_mode="r")
            self._mask_mem = np.load(self.segment_mask_path, mmap_mode="r")
        original = int(self.original_indices[index])
        inputs["text_segments"] = torch.from_numpy(np.array(self._seg_mem[original], dtype=np.float32, copy=True))
        inputs["text_segment_mask"] = torch.from_numpy(np.array(self._mask_mem[original], dtype=np.bool_, copy=True))
        return inputs, targets

    def collect(self):
        original_collate = super().collect()

        def collate(samples):
            inputs, targets = original_collate(samples)
            inputs["text_segments"] = torch.stack([item["text_segments"] for item, _ in samples])
            inputs["text_segment_mask"] = torch.stack([item["text_segment_mask"] for item, _ in samples])
            return inputs, targets
        return collate


def get_dataset_v2(*, data_path, ratio_train_val_test=None, n_train_val_test=None,
                   transforms=None, graph=True, line_graph=True, text_embedding_cache_path,
                   text_field="description", text_encoder_name=None, text_model_name=None, **_):
    if not graph or not line_graph:
        raise ValueError("Fusiformer-V2 requires graph=True and line_graph=True")
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError("dataset JSON must contain a list")
    cache, mask_cache, _ = validate_semantic_cache(
        text_embedding_cache_path, data=data, text_field=text_field,
        text_encoder_name=text_encoder_name, text_model_name=text_model_name)
    ids = split_train_val_test(len(data), ratio_train_val_test=ratio_train_val_test,
                               n_train_val_test=n_train_val_test)
    if isinstance(transforms, list):
        if len(transforms) == 3:
            ts = transforms
        elif len(transforms) == 2:
            ts = [transforms[0], transforms[1], transforms[1]]
        elif len(transforms) == 1:
            ts = transforms * 3
        else:
            raise ValueError("transforms list must have 1 to 3 elements")
    else:
        ts = [transforms] * 3
    out = []
    for idx, transform in zip(ids, ts):
        if idx is None or len(idx) == 0:
            out.append(None)
            continue
        out.append(SemanticCrystalGraphDataset(
            data=[data[i] for i in idx], original_indices=idx, transforms=transform,
            line_graph=True, text_field=text_field,
            segment_cache_path=cache, segment_mask_path=mask_cache,
        ))
    return tuple(out)
