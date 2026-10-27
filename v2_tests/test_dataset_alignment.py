"""Tests index alignment without importing unavailable DGL/JARVIS packages."""
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


def fingerprint(data, field):
    h = hashlib.sha256()
    for i, item in enumerate(data):
        text = str(item.get(field, "") or "")
        raw = text.encode()
        h.update(i.to_bytes(8, 'little'))
        h.update(len(raw).to_bytes(8, 'little'))
        h.update(raw)
    return h.hexdigest()


class DummyOriginalCachedDataset:
    def __init__(self, *, data, original_indices, **kwargs):
        self.data, self.original_indices = data, original_indices

    def __getitem__(self, idx):
        return {"graph_input": self.data[idx]["description"]}, torch.tensor([float(idx)])

    def collect(self):
        def collate(samples):
            return {"graph_input": [x["graph_input"] for x, _ in samples]}, torch.stack([y for _, y in samples])
        return collate


def test_index_alignment_and_cache_validation(tmp_path):
    data = [{"description": "mat0"}, {"description": "mat1"},
            {"description": "mat2"}, {"description": "mat3"}]
    tokens = np.zeros((4, 3, 8), dtype=np.float32)
    for i in range(4):
        tokens[i] = i
    masks = np.array([[1,0,0], [1,1,0], [1,1,1], [1,0,0]], dtype=bool)
    cache = tmp_path / 'segments.npy'
    np.save(cache, tokens)
    np.save(tmp_path / 'segments_mask.npy', masks)
    (tmp_path / 'segments.npy.meta.json').write_text(json.dumps({
        "cache_format": "fusiformer-v2-segments-v1",
        "shape": [4, 3, 8], "hidden_size": 8,
        "text_field": "description", "text_sha256": fingerprint(data, "description"),
        "encoder_name": "matscibert", "model_name": "/root/models/matscibert",
    }))
    loader_pkg = types.ModuleType('loader')
    loader_pkg.__path__ = []
    loader_module = types.ModuleType('loader.loader')
    loader_module.split_train_val_test = lambda *args, **kwargs: ([2, 0], [3], [1])
    cached_module = types.ModuleType('loader.loader_cached')
    cached_module.CachedCrystalGraphDataset = DummyOriginalCachedDataset
    cached_module._text_fingerprint = fingerprint
    spec = importlib.util.spec_from_file_location(
        'fusiformer_v2.dataset_test_fake',
        Path(__file__).resolve().parents[1] / 'fusiformer_v2' / 'dataset.py')
    dataset_module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'loader': loader_pkg,
                                  'loader.loader': loader_module,
                                  'loader.loader_cached': cached_module}):
        spec.loader.exec_module(dataset_module)
    data_file = tmp_path / 'test.json'
    data_file.write_text(json.dumps(data))
    train, val, test = dataset_module.get_dataset_v2(
        data_path=data_file, ratio_train_val_test=[0.5, 0.25, 0.25],
        text_embedding_cache_path=cache, text_encoder_name='matscibert',
        text_model_name='/root/models/matscibert',
    )
    assert train[0][0]['text_segments'][0, 0].item() == 2
    assert train[1][0]['text_segments'][0, 0].item() == 0
    batch, _ = train.collect()([train[0], train[1]])
    assert batch['text_segments'].shape == (2, 3, 8)
    assert batch['text_segment_mask'].tolist() == masks[[2, 0]].tolist()
    assert val[0][0]['text_segments'][0, 0].item() == 3
    assert test[0][0]['text_segments'][0, 0].item() == 1
