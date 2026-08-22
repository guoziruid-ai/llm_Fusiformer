from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

from tokenizers.normalizers import BertNormalizer


class IdentityPreprocessor:
    name = "none"

    def __call__(self, text: str) -> str:
        if text is None:
            return ""
        return text if isinstance(text, str) else str(text)


class MatSciBertPreprocessor:
    """Official MatSciBERT text normalization.

    Mirrors M3RG-IITD/MatSciBERT/normalize_text.py:
      1) BertNormalizer(lowercase=False, strip_accents=True,
         clean_text=True, handle_chinese_chars=True)
      2) per-character mapping from vocab_mappings.txt

    A copy of vocab_mappings.txt is shipped next to this module so the
    preprocessing path also works on an offline training server.
    """

    name = "matscibert"

    def __init__(self, mapping_path: str | Path | None = None):
        if mapping_path is None:
            mapping_path = Path(__file__).with_name("vocab_mappings.txt")
        mapping_path = Path(mapping_path)
        if not mapping_path.is_file():
            raise FileNotFoundError(
                "MatSciBERT vocab mapping file is missing: "
                f"{mapping_path}. Keep vocab_mappings.txt next to preprocessing.py."
            )

        self.normalizer = BertNormalizer(
            lowercase=False,
            strip_accents=True,
            clean_text=True,
            handle_chinese_chars=True,
        )
        mappings: dict[str, str] = {}
        with mapping_path.open("r", encoding="utf-8") as f:
            for raw in f.read().splitlines():
                if not raw:
                    continue
                # Official implementation uses m[0] -> m[2:].
                mappings[raw[0]] = raw[2:] if len(raw) >= 2 else ""
        self.mappings = mappings

    def __call__(self, text: str) -> str:
        if text is None:
            text = ""
        if not isinstance(text, str):
            text = str(text)

        lines = [self.normalizer.normalize_str(s) for s in text.split("\n")]
        out: list[str] = []
        for line in lines:
            out.append("".join(self.mappings.get(c, " ") for c in line))
        return "\n".join(out)


def infer_preprocessing_name(
    text_encoder_name: str | None,
    text_model_name: str | None,
) -> str:
    name = (text_encoder_name or "").strip().lower()
    model = (text_model_name or "").strip().lower()
    if "matscibert" in name or "matscibert" in model:
        return "matscibert"
    return "none"


def build_text_preprocessor(
    name: str | None = "auto",
    *,
    text_encoder_name: str | None = None,
    text_model_name: str | None = None,
    matscibert_mapping_path: str | Path | None = None,
) -> Callable[[str], str]:
    key = (name or "auto").strip().lower()
    if key == "auto":
        key = infer_preprocessing_name(text_encoder_name, text_model_name)

    if key in {"none", "identity", "raw"}:
        return IdentityPreprocessor()
    if key in {"matscibert", "matsci_bert"}:
        return MatSciBertPreprocessor(matscibert_mapping_path)
    raise ValueError(
        f"Unsupported text preprocessing={name!r}. "
        "Supported: auto, none, matscibert"
    )


def preprocess_texts(texts: Iterable[str], preprocessor: Callable[[str], str]) -> list[str]:
    return [preprocessor(text) for text in texts]
