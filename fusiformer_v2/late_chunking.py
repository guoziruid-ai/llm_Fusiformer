"""Token-span planning for frozen ModernBERT Late Chunking.

Pure Python, no GPU/model dependencies. All source tokens belong to exactly
one semantic segment and exactly one window ownership range. Window overlaps
provide context; their tokens are NOT double counted in the pooled embedding.
"""
from __future__ import annotations

import bisect
import math
import re
from dataclasses import dataclass
from typing import Sequence

# Match the original V2 sentence splitting conventions, preserving offsets.
_SENTENCE_GAP = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])|\n+")


@dataclass(frozen=True)
class Window:
    input_start: int
    input_end: int
    own_start: int
    own_end: int


def semantic_spans(
    text: str,
    offsets: Sequence[tuple[int, int]],
    *,
    max_segments: int = 32,
    target_tokens: int = 160,
) -> list[tuple[int, int]]:
    """Partition token indices into <= max_segments contiguous semantic ranges.

    Prefer sentence boundaries near the desired group size; otherwise split by
    token count. A final segment always absorbs any remaining tokens, so no
    sentence/token is silently dropped. `offsets` excludes special tokens.
    """
    if max_segments < 1 or target_tokens < 1:
        raise ValueError("max_segments and target_tokens must be positive")
    n = len(offsets)
    if n == 0:
        return []
    starts = [start for start, _ in offsets]
    candidate_ends = set()
    for match in _SENTENCE_GAP.finditer(text):
        cut = bisect.bisect_left(starts, match.end())
        if 0 < cut < n:
            candidate_ends.add(cut)
    boundaries = sorted(candidate_ends)

    spans: list[tuple[int, int]] = []
    left = 0
    while left < n:
        slots = max_segments - len(spans)
        if slots == 1:
            right = n
        else:
            goal = max(target_tokens, math.ceil((n - left) / slots))
            if n - left <= goal:
                right = n
            else:
                lo = min(n - 1, left + max(1, round(goal * 0.85)))
                hi = min(n - 1, left + max(1, round(goal * 1.25)))
                b0 = bisect.bisect_left(boundaries, lo)
                options = [b for b in boundaries[b0:] if b <= hi]
                right = min(options, key=lambda x: abs(x - (left + goal))) if options else min(n, left + goal)
        if not left < right <= n:
            raise AssertionError("invalid segmentation boundary")
        spans.append((left, right))
        left = right
    if spans[0][0] != 0 or spans[-1][1] != n:
        raise AssertionError("token coverage failure")
    return spans


def make_windows(num_tokens: int, *, content_tokens: int, overlap: int = 256) -> list[Window]:
    """Context windows with disjoint central ownership ranges.

    Pool ONLY owned tokens in each window to avoid inflating overlapped spans.
    """
    if num_tokens < 0 or content_tokens < 1 or not 0 <= overlap < content_tokens:
        raise ValueError("bad window size or overlap")
    if num_tokens == 0:
        return []
    stride = content_tokens - overlap
    bounds = []
    start = 0
    while True:
        end = min(start + content_tokens, num_tokens)
        bounds.append((start, end))
        if end == num_tokens:
            break
        start += stride
    ownership_edges = [0]
    for (a, b), (c, d) in zip(bounds, bounds[1:]):
        if c > b:
            raise AssertionError("context window gap")
        ownership_edges.append((b + c) // 2)
    ownership_edges.append(num_tokens)
    result = [Window(a, b, ownership_edges[i], ownership_edges[i + 1])
              for i, (a, b) in enumerate(bounds)]
    if any(not (w.input_start <= w.own_start < w.own_end <= w.input_end) for w in result):
        raise AssertionError("bad ownership partition")
    return result
