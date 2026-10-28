"""CPU-only tests for deterministic token coverage, late pooling and metadata."""
from __future__ import annotations

import hashlib
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fusiformer_v2.late_chunking import make_windows, semantic_spans
from tools.precompute_modernbert_late_v2 import document_embedding, fingerprint_and_count


class ToyTokenizer:
    is_fast = True
    cls_token_id = 1
    sep_token_id = 2

    def __call__(self, text, add_special_tokens=False, truncation=False,
                 return_offsets_mapping=False, return_attention_mask=False):
        import re
        matches = list(re.finditer(r"\S+", text))
        return {"input_ids": [m.start() + 10 for m in matches],
                "offset_mapping": [(m.start(), m.end()) for m in matches]}

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def build_inputs_with_special_tokens(self, ids):
        return [1, *ids, 2]

    def get_special_tokens_mask(self, ids, already_has_special_tokens=False):
        return [1] + [0] * len(ids) + [1]


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)

    def forward(self, input_ids, attention_mask, return_dict=True):
        x = input_ids.to(torch.float32)
        # First channel remains a token ID, independent of the context window.
        h = torch.stack([x, x * 2, x * 0 + 1, x * 0], -1)
        return SimpleNamespace(last_hidden_state=h)


def test_windows_own_exactly_once():
    for count in (1, 17, 18, 21, 50, 250, 1025):
        windows = make_windows(count, content_tokens=18, overlap=5)
        claimed = []
        for w in windows:
            assert w.input_start <= w.own_start < w.own_end <= w.input_end
            claimed.extend(range(w.own_start, w.own_end))
        assert claimed == list(range(count))


def test_segment_coverage_and_max():
    text = "A Ti bond. " * 250 + "Coordination\n" + "O " * 300
    tokenizer = ToyTokenizer()
    encoded = tokenizer(text, return_offsets_mapping=True)
    segments = semantic_spans(text, encoded["offset_mapping"], max_segments=32, target_tokens=35)
    assert len(segments) <= 32
    assert segments[0][0] == 0
    assert segments[-1][1] == len(encoded["input_ids"])
    assert all(a < b == segments[i+1][0] for i, (a, b) in enumerate(segments[:-1]))


def test_no_truncation_and_exact_late_pool():
    text = " ".join([f"token{i}" for i in range(80)])
    tokenizer = ToyTokenizer()
    model = ToyModel().eval()
    out, mask, n_tokens, n_windows = document_embedding(
        text, tokenizer, model, device=torch.device("cpu"), window_size=20,
        overlap=4, max_segments=8, target_segment_tokens=12, cache_dtype=np.float32,
    )
    assert n_tokens == 80
    assert n_windows > 1
    assert mask.sum() == 7 or mask.sum() == 8
    ids, offsets = (lambda b: (b['input_ids'], b['offset_mapping']))(
        tokenizer(text, return_offsets_mapping=True))
    spans = semantic_spans(text, offsets, max_segments=8, target_tokens=12)
    for i, (start, end) in enumerate(spans):
        assert np.isclose(out[i,0], np.mean(ids[start:end]), atol=1e-5)
        assert np.isclose(out[i,1], 2*np.mean(ids[start:end]), atol=1e-5)
    assert np.isfinite(out).all()


def test_empty_description():
    out, mask, tokens, windows = document_embedding("", ToyTokenizer(), ToyModel(),
        device=torch.device("cpu"), window_size=20, overlap=2,
        max_segments=5, target_segment_tokens=4, cache_dtype=np.float16)
    assert (tokens, windows) == (0, 0)
    assert not mask.any()
    assert not out.any()


def test_fingerprint_matches_v2_loader_format(tmp_path, monkeypatch):
    examples = [{"description": "alpha"}, {"description": None}, {"no_description": True}]
    source = tmp_path / "data.json"
    source.write_text(json.dumps(examples), encoding="utf-8")
    import sys
    fake = SimpleNamespace(items=lambda stream, prefix: iter(json.load(stream)))
    monkeypatch.setitem(sys.modules, "ijson", fake)
    count, digest = fingerprint_and_count(source, "description")
    h = hashlib.sha256()
    for i, row in enumerate(examples):
        value = row.get("description", "")
        text = "" if value is None else str(value)
        blob = text.encode("utf-8")
        h.update(i.to_bytes(8,"little"))
        h.update(len(blob).to_bytes(8,"little"))
        h.update(blob)
    assert count == 3 and digest == h.hexdigest()


def test_complete_streaming_cache_stop_and_resume(tmp_path, monkeypatch):
    """Simulated checkpoint tests actual generator -> native V2 metadata."""
    import sys
    from tools import precompute_modernbert_late_v2 as script
    examples = [{"description": "One Two Three Four Five Six Seven Eight Nine Ten"},
                {"description": ""},
                {"description": "A long example with coordination. Polyhedra share corners."}]
    data = tmp_path / "data.json"
    data.write_text(json.dumps(examples), encoding="utf-8")
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"toy-checkpoint")
    monkeypatch.setitem(sys.modules, "ijson", SimpleNamespace(items=lambda s, prefix: iter(json.load(s))))
    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: ToyTokenizer()),
        AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **kw:
                                   SimpleNamespace(model_type="modernbert", hidden_size=4, max_position_embeddings=128)),
        AutoModel=SimpleNamespace(from_pretrained=lambda *a, **kw: ToyModel()),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    output = tmp_path / "semantic.npy"
    common = ["--data", str(data), "--model", str(tmp_path), "--output", str(output),
              "--device", "cpu", "--compute-dtype", "float32", "--window-size", "14",
              "--overlap", "3", "--max-segments", "6", "--target-segment-tokens", "4",
              "--checkpoint-every", "1"]
    script.main(common + ["--stop-after", "1"])
    assert not output.exists()
    assert (tmp_path / "semantic.npy.progress.json").exists()
    script.main(common + ["--resume"])
    mask_path = tmp_path / "semantic_mask.npy"
    meta_path = tmp_path / "semantic.npy.meta.json"
    tokens = np.load(output)
    mask = np.load(mask_path)
    meta = json.loads(meta_path.read_text())
    assert tokens.shape == (3, 6, 4)
    assert mask.shape == (3, 6)
    assert mask[0].any() and mask[2].any() and not mask[1].any()
    assert meta["cache_format"] == "fusiformer-v2-segments-v1"
    assert meta["encoder_name"] == "modernbert"
    assert meta["hidden_size"] == 4
    assert meta["text_sha256"] == fingerprint_and_count(data, "description")[1]
    assert not (tmp_path / "semantic.npy.progress.json").exists()
