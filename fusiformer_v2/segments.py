"""Conservative, deterministic sentence/segment grouping for Robocrystallographer.

Split on sentence-ending punctuation followed by whitespace; protect decimal
numbers (e.g. 1.95 Å). Long sentences are subdivided by tokenizer IDs.
All segments are reconstructed through the source tokenizer, so specialized
MatSciBERT vocab normalization remains in its established encoder pipeline.
"""
from __future__ import annotations

import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])|\n+")


def split_semantic_segments(text: str, tokenizer, *, max_tokens: int = 256,
                            max_segments: int = 24, overflow: str = "error") -> tuple[list[str], int]:
    if max_tokens < 8 or max_segments < 1:
        raise ValueError("max_tokens must be >=8; max_segments >=1")
    if overflow not in {"error", "head_tail"}:
        raise ValueError("overflow must be error or head_tail")
    text = "" if text is None else str(text).strip()
    if not text:
        return [], 0
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    # tokenizer without special tokens so each segment can later add CLS/SEP
    budget = max_tokens - tokenizer.num_special_tokens_to_add(pair=False)
    if budget < 1:
        raise ValueError("max_tokens insufficient for model special tokens")
    parts = []
    for sentence in sentences:
        ids = tokenizer.encode(sentence, add_special_tokens=False)
        for start in range(0, len(ids), budget):
            chunk = ids[start:start + budget]
            if chunk:
                parts.append(tokenizer.decode(chunk, skip_special_tokens=True))
    # Keep sentence boundaries whenever possible, but don't silently drop spans.
    omitted = max(0, len(parts) - max_segments)
    if omitted:
        if overflow == "error":
            raise ValueError(f"text needs {len(parts)} segments (> max_segments={max_segments}); "
                             "raise max_segments or choose overflow=head_tail explicitly")
        front = (max_segments + 1) // 2
        back = max_segments - front
        parts = parts[:front] + (parts[-back:] if back else [])
    return parts, omitted
