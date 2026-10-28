"""Explicit, validated row identities for externally prepared arrays."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .manifests import validate_manifest

IDENTITY_FIELDS = ("sample_id", "original_index", "mp_id", "physical_id", "split")


def load_aligned_array(array_path: str | Path, sidecar_path: str | Path,
                       master_rows: Sequence[dict[str, Any]],
                       selected_rows: Sequence[dict[str, Any]]) -> np.ndarray:
    """Return array rows in selected manifest order after full identity validation.

    Sidecar CSV rows describe the corresponding array rows, not manifest order.
    Arrays cover the master, one complete selected split, or the selection.
    """
    validate_manifest(master_rows)
    with Path(sidecar_path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not set(IDENTITY_FIELDS).issubset(reader.fieldnames):
            raise ValueError(f"sidecar must contain {IDENTITY_FIELDS}")
        sidecar = list(reader)
    values = np.load(array_path, mmap_mode="r")
    if values.ndim == 0 or len(values) != len(sidecar):
        raise ValueError("array and sidecar row counts differ")
    master = {str(row["sample_id"]): row for row in master_rows}
    if any(not key.strip() for key in master):
        raise ValueError("master manifest contains empty sample_id")
    positions: dict[str, int] = {}
    for position, row in enumerate(sidecar):
        key = row["sample_id"]
        if not key or key in positions:
            raise ValueError(f"sidecar contains empty or duplicate sample_id: {key!r}")
        if key not in master:
            raise ValueError(f"unknown sidecar sample_id: {key!r}")
        expected = master[key]
        for field in IDENTITY_FIELDS:
            if str(row[field]) != str(expected[field]):
                raise ValueError(f"sidecar {field} mismatch for sample_id {key!r}")
        positions[key] = position
    chosen = []
    seen = set()
    for row in selected_rows:
        key = str(row["sample_id"])
        if key in seen or key not in master:
            raise ValueError(f"selected rows contain duplicate or unknown sample_id: {key!r}")
        if any(str(row[field]) != str(master[key][field]) for field in IDENTITY_FIELDS):
            raise ValueError(f"selected row identity mismatch for sample_id {key!r}")
        seen.add(key)
        if key not in positions:
            raise ValueError(f"sidecar missing selected sample_id: {key!r}")
        chosen.append(positions[key])
    selected_splits = {str(row["split"]) for row in selected_rows}
    allowed_sets = [set(master), seen]
    if len(selected_splits) == 1:
        split = next(iter(selected_splits))
        allowed_sets.append({key for key, row in master.items() if row["split"] == split})
    if set(positions) not in allowed_sets:
        raise ValueError("sidecar has missing or extra samples for the requested scope")
    return np.asarray(values[chosen])
