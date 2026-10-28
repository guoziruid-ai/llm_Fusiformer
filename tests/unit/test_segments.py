from fusiformer_v2.segments import split_semantic_segments
import pytest


class FakeTokenizer:
    def num_special_tokens_to_add(self, pair=False):
        return 2

    def encode(self, text, add_special_tokens=False):
        return [ord(x) for x in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


def test_empty():
    assert split_semantic_segments("", FakeTokenizer()) == ([], 0)


def test_sentences():
    parts, dropped = split_semantic_segments("Ti bonds O. Octahedra share corners.", FakeTokenizer(), max_tokens=40)
    assert dropped == 0
    assert parts == ["Ti bonds O.", "Octahedra share corners."]


def test_very_long_sentence_is_chunked():
    parts, dropped = split_semantic_segments("a" * 70, FakeTokenizer(), max_tokens=32, max_segments=4)
    assert len(parts) == 3 and dropped == 0
    assert "".join(parts) == "a" * 70


def test_overflow_explicit():
    with pytest.raises(ValueError):
        split_semantic_segments("x" * 300, FakeTokenizer(), max_tokens=12, max_segments=2)
    parts, omitted = split_semantic_segments("x" * 300, FakeTokenizer(), max_tokens=12, max_segments=2, overflow="head_tail")
    assert len(parts) == 2 and omitted > 0
