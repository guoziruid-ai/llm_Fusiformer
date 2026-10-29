"""Strict, non-fallback specialist checkpoint protocol."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from .config import PredictionMode
from .manifests import validate_manifest

CHECKPOINT_FORMAT = "fusiformer-cluster-specialist-v1"


def manifest_fingerprint(rows: Any) -> str:
    """Bind a checkpoint to sample identities, cluster assignments and splits."""
    validate_manifest(rows)
    fields = ("sample_id", "original_index", "mp_id", "physical_id", "cluster_id", "split")
    canonical = [{field: str(row[field]) for field in fields} for row in rows]
    canonical.sort(key=lambda row: row["sample_id"])
    encoded = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def checkpoint_payload(model: torch.nn.Module, *, cluster_id: str,
                       prediction_mode: str | PredictionMode,
                       architecture: str,
                       model_config: Mapping[str, Any],
                       training_config: Mapping[str, Any] | None = None,
                       epoch: int | None = None,
                       metrics: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if cluster_id not in {"A", "B", "C"}:
        raise ValueError("cluster_id must be A, B or C")
    mode = PredictionMode(prediction_mode).value
    if not architecture:
        raise ValueError("architecture name is required")
    return {
        "format": CHECKPOINT_FORMAT, "format_version": 1,
        "cluster_id": cluster_id, "prediction_mode": mode,
        "architecture": architecture, "model_config": dict(model_config),
        "training_config": dict(training_config or {}),
        "epoch": epoch, "metrics": dict(metrics or {}),
        "model_state_dict": model.state_dict()}


def save_checkpoint(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(
            f"refusing to overwrite specialist checkpoint: {target}")
    torch.save(dict(payload), target)
    return target


def load_checkpoint(path: str | Path, model: torch.nn.Module, *,
                    expected_cluster: str,
                    expected_mode: str | PredictionMode,
                    expected_architecture: str,
                    expected_manifest_fingerprint: str | None = None,
                    map_location: Any = "cpu") -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"specialist checkpoint does not exist: {source}")
    try:
        payload = torch.load(
            source, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(source, map_location=map_location)
    if not isinstance(payload, dict):
        raise TypeError("specialist checkpoint must be a dictionary")
    expected = {
        "format": CHECKPOINT_FORMAT, "format_version": 1,
        "cluster_id": expected_cluster,
        "prediction_mode": PredictionMode(expected_mode).value,
        "architecture": expected_architecture}
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"checkpoint {key}={payload.get(key)!r}; expected {value!r}")
    if expected_manifest_fingerprint is not None:
        config = payload.get("training_config")
        saved = config.get("manifest_sha256") if isinstance(config, dict) else None
        if saved != expected_manifest_fingerprint:
            raise ValueError("checkpoint training manifest fingerprint is missing or incompatible")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint has no non-empty model_state_dict")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"specialist checkpoint is incompatible with the model: {exc}") from exc
    return payload
