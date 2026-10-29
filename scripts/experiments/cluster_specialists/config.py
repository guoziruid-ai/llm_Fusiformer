"""Small, serializable configuration objects for cluster-specialist runs."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class HistoricalSourceConfig:
    """Explicit immutable labels and identity evidence; never fit a router.

    The defaults are deliberately unconfigured: historical Grouped-3 routes
    use split-local positions, not original indices or crystal IDs.
    """

    master_path: str = str(ROOT / "data/cluster_historical_master.csv")
    labels_path: str = str(ROOT / "data/cluster_historical_labels.csv")
    master_id_col: str = "sample_id"
    label_id_col: str = "sample_id"
    label_col: str = "cluster_id"
    label_mapping: tuple[tuple[str, str], ...] = ()
    position_map_path: str | None = None
    position_label_col: str | None = None
    position_map_id_col: str = "sample_id"
    position_map_position_col: str | None = None
    grouped3_mapping_path: str | None = None
    grouped3_raw_col: str = "cluster_id"
    grouped3_expert_id_col: str = "expert_id"
    grouped3_expert_name_col: str = "expert_name"

    def load(self):
        import csv
        from .historical import load_historical_rows
        if not self.label_mapping:
            raise ValueError("historical label_mapping must be explicitly configured")
        if len(dict(self.label_mapping)) != len(self.label_mapping):
            raise ValueError("historical label_mapping contains duplicate source labels")
        if self.grouped3_mapping_path is None and Path(self.labels_path).is_file():
            with Path(self.labels_path).open(encoding="utf-8-sig", newline="") as handle:
                fields = set(csv.DictReader(handle).fieldnames or ())
            if {self.grouped3_raw_col, self.grouped3_expert_id_col,
                    self.grouped3_expert_name_col} <= fields:
                raise ValueError("Grouped-3 route columns require grouped3_mapping_path")
        if self.grouped3_mapping_path is not None:
            from .historical import verify_grouped3_mapping
            if self.label_col != self.grouped3_expert_id_col:
                raise ValueError("Grouped-3 label_col must be the verified expert ID column")
            expected = verify_grouped3_mapping(
                self.labels_path, self.grouped3_mapping_path,
                raw_col=self.grouped3_raw_col, expert_id_col=self.grouped3_expert_id_col,
                expert_name_col=self.grouped3_expert_name_col)
            if dict(self.label_mapping) != expected:
                raise ValueError("configured mapping differs from Grouped-3 mapping artifact")
        return load_historical_rows(
            self.master_path, self.labels_path, master_id_col=self.master_id_col,
            label_id_col=self.label_id_col, label_col=self.label_col,
            label_mapping=dict(self.label_mapping),
            position_map_path=self.position_map_path,
            position_label_col=self.position_label_col,
            position_map_id_col=self.position_map_id_col,
            position_map_position_col=self.position_map_position_col)


@dataclass(frozen=True)
class HistoricalMultiSourceConfig:
    """Combine individually verified run/split position sources in memory only."""

    sources: tuple[HistoricalSourceConfig, ...] = ()

    def load(self):
        from .manifests import validate_manifest
        if not self.sources:
            raise ValueError("at least one historical source must be configured")
        mapping = dict(self.sources[0].label_mapping)
        mapping_path = self.sources[0].grouped3_mapping_path
        if any(dict(source.label_mapping) != mapping or
               source.grouped3_mapping_path != mapping_path for source in self.sources):
            raise ValueError("historical sources must share the same label mapping and artifact")
        rows = [row for source in self.sources for row in source.load()]
        validate_manifest(rows, expected_count=len(rows))
        clusters = {}
        for row in rows:
            physical_id = row["physical_id"]
            if physical_id in clusters and clusters[physical_id] != row["cluster_id"]:
                raise ValueError(f"physical_id {physical_id!r} has conflicting cluster labels")
            clusters[physical_id] = row["cluster_id"]
        return rows


class ClusterId(str, Enum):
    A = "A"
    B = "B"
    C = "C"


class PredictionMode(str, Enum):
    DIRECT = "direct"
    RESIDUAL = "residual"


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.8
    validation: float = 0.1
    test: float = 0.1

    def validate(self) -> None:
        values = (self.train, self.validation, self.test)
        if any(value < 0 for value in values):
            raise ValueError("split ratios must be non-negative")
        if abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("split ratios must sum to 1")
        if self.train <= 0:
            raise ValueError("training ratio must be positive")

    def as_dict(self) -> dict[str, float]:
        self.validate()
        return {"train": self.train, "validation": self.validation, "test": self.test}
