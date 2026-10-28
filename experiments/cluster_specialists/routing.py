"""Leakage-safe, serializable A/B/C routing.

Routers consume inference-time features only. Labels and prediction errors are
intentionally absent from every fit/predict signature.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import ClusterId

ROUTER_FORMAT = "fusiformer-cluster-router-v1"


@dataclass(frozen=True)
class RouteDecision:
    cluster_id: str
    raw_cluster_id: int
    distance: float
    margin: float
    out_of_domain: bool
    reason: str


def _matrix(features: Any, *, expected_dim: int | None = None) -> np.ndarray:
    array = np.asarray(features, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2 or len(array) == 0:
        raise ValueError("router features must be a non-empty [N,D] matrix")
    if expected_dim is not None and array.shape[1] != expected_dim:
        raise ValueError(f"router feature dimension {array.shape[1]} != expected {expected_dim}")
    if not np.isfinite(array).all():
        raise ValueError("router features contain NaN or infinity")
    return array


class FixedBoundaryRouter:
    """Route a single deployable scalar feature through two fixed boundaries."""

    def __init__(self, low: float, high: float, *, feature_name: str,
                 observed_range: tuple[float, float] | None = None):
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise ValueError("fixed boundaries must be finite and low < high")
        if not feature_name:
            raise ValueError("feature_name is required")
        if observed_range is not None and observed_range[0] > observed_range[1]:
            raise ValueError("observed_range minimum exceeds maximum")
        self.low, self.high = float(low), float(high)
        self.feature_name = feature_name
        self.observed_range = observed_range

    @property
    def feature_dim(self) -> int:
        return 1

    def predict_diagnostics(self, features: Any) -> list[RouteDecision]:
        values = _matrix(features, expected_dim=1)[:, 0]
        decisions = []
        for value in values:
            if value <= self.low:
                cluster, raw = ClusterId.A.value, 0
            elif value <= self.high:
                cluster, raw = ClusterId.B.value, 1
            else:
                cluster, raw = ClusterId.C.value, 2
            distances = sorted((abs(value - self.low), abs(value - self.high)))
            out = bool(self.observed_range and not (
                self.observed_range[0] <= value <= self.observed_range[1]))
            decisions.append(RouteDecision(
                cluster, raw, distances[0], distances[1] - distances[0], out,
                "fixed_boundary_out_of_range" if out else "fixed_boundary"))
        return decisions

    def predict(self, features: Any) -> np.ndarray:
        return np.asarray(
            [item.cluster_id for item in self.predict_diagnostics(features)], dtype="U1")

    def to_payload(self) -> dict[str, Any]:
        return {"format": ROUTER_FORMAT, "router_type": "fixed_boundary",
                "feature_name": self.feature_name, "low": self.low, "high": self.high,
                "observed_range": list(self.observed_range) if self.observed_range else None,
                "cluster_mapping": {"0": "A", "1": "B", "2": "C"}}

    def save(self, path: str | Path) -> Path:
        return _write_payload(path, self.to_payload())


class CentroidRouter:
    """Train-only standardized KMeans router with an explicit stable A/B/C map."""

    def __init__(self, *, feature_names: Iterable[str], random_seed: int = 42,
                 n_init: int = 20, max_iter: int = 500):
        names = tuple(str(name) for name in feature_names)
        if not names or any(not name for name in names):
            raise ValueError("feature_names must be non-empty")
        self.feature_names = names
        self.random_seed = int(random_seed)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.centers_: np.ndarray | None = None
        self.raw_to_cluster_: dict[int, str] | None = None
        self.ood_distance_: float | None = None
        self.fit_provenance_: dict[str, Any] | None = None

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    @property
    def fitted(self) -> bool:
        return self.centers_ is not None

    def fit(self, train_features: Any,
            *, cluster_mapping: dict[int, str] | None = None) -> "CentroidRouter":
        """Fit only on a predeclared training population; no labels are accepted."""
        from sklearn.cluster import KMeans

        self.fit_provenance_ = None
        matrix = _matrix(train_features, expected_dim=self.feature_dim)
        if len(matrix) < 3:
            raise ValueError("at least three training samples are required")
        self.mean_ = matrix.mean(axis=0)
        self.scale_ = matrix.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        standardized = (matrix - self.mean_) / self.scale_
        model = KMeans(n_clusters=3, random_state=self.random_seed,
                       n_init=self.n_init, max_iter=self.max_iter)
        raw = model.fit_predict(standardized)
        self.centers_ = np.asarray(model.cluster_centers_, dtype=np.float64)
        if cluster_mapping is None:
            order = sorted(range(3), key=lambda idx: tuple(self.centers_[idx].tolist()))
            cluster_mapping = {raw_id: "ABC"[rank] for rank, raw_id in enumerate(order)}
        _validate_mapping(cluster_mapping)
        self.raw_to_cluster_ = {
            int(key): str(value) for key, value in cluster_mapping.items()}
        distances = np.linalg.norm(standardized - self.centers_[raw], axis=1)
        self.ood_distance_ = float(np.quantile(distances, 0.995))
        return self

    def _check(self) -> None:
        if (self.mean_ is None or self.scale_ is None or self.centers_ is None
                or self.raw_to_cluster_ is None or self.ood_distance_ is None):
            raise RuntimeError("centroid router has not been fitted or loaded")

    def predict_diagnostics(self, features: Any) -> list[RouteDecision]:
        self._check()
        matrix = _matrix(features, expected_dim=self.feature_dim)
        standardized = (matrix - self.mean_) / self.scale_
        all_distances = np.linalg.norm(
            standardized[:, None, :] - self.centers_[None, :, :], axis=2)
        raw = np.argmin(all_distances, axis=1)
        ordered = np.sort(all_distances, axis=1)
        return [RouteDecision(
            cluster_id=self.raw_to_cluster_[int(raw_id)],
            raw_cluster_id=int(raw_id),
            distance=float(ordered[row, 0]),
            margin=float(ordered[row, 1] - ordered[row, 0]),
            out_of_domain=bool(ordered[row, 0] > self.ood_distance_),
            reason=("nearest_centroid_out_of_domain"
                    if ordered[row, 0] > self.ood_distance_ else "nearest_centroid"),
        ) for row, raw_id in enumerate(raw)]

    def predict(self, features: Any) -> np.ndarray:
        return np.asarray(
            [item.cluster_id for item in self.predict_diagnostics(features)], dtype="U1")

    def to_payload(self) -> dict[str, Any]:
        self._check()
        return {"format": ROUTER_FORMAT, "router_type": "centroid",
                "feature_names": list(self.feature_names),
                "random_seed": self.random_seed, "n_init": self.n_init,
                "max_iter": self.max_iter, "mean": self.mean_.tolist(),
                "scale": self.scale_.tolist(), "centers": self.centers_.tolist(),
                "cluster_mapping": {
                    str(key): value for key, value in self.raw_to_cluster_.items()},
                "ood_distance": self.ood_distance_, "fit_scope": "training_only",
                "fit_provenance": self.fit_provenance_}

    def save(self, path: str | Path) -> Path:
        return _write_payload(path, self.to_payload())

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "CentroidRouter":
        router = cls(
            feature_names=payload["feature_names"],
            random_seed=payload["random_seed"],
            n_init=payload["n_init"], max_iter=payload["max_iter"])
        router.mean_ = np.asarray(payload["mean"], dtype=np.float64)
        router.scale_ = np.asarray(payload["scale"], dtype=np.float64)
        router.centers_ = np.asarray(payload["centers"], dtype=np.float64)
        router.raw_to_cluster_ = {
            int(key): str(value) for key, value in payload["cluster_mapping"].items()}
        _validate_mapping(router.raw_to_cluster_)
        router.ood_distance_ = float(payload["ood_distance"])
        router.fit_provenance_ = payload.get("fit_provenance")
        if router.centers_.shape != (3, router.feature_dim):
            raise ValueError("saved centroid shape is incompatible with feature_names")
        return router


def _validate_mapping(mapping: dict[int, str]) -> None:
    normalized = {int(key): str(value) for key, value in mapping.items()}
    if set(normalized) != {0, 1, 2} or set(normalized.values()) != {"A", "B", "C"}:
        raise ValueError("cluster_mapping must bijectively map raw IDs 0,1,2 to A,B,C")


def _write_payload(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite router artifact: {target}")
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def load_router(path: str | Path) -> FixedBoundaryRouter | CentroidRouter:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"router artifact does not exist: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("format") != ROUTER_FORMAT:
        raise ValueError("router artifact format is missing or incompatible")
    if payload.get("router_type") == "fixed_boundary":
        observed = payload.get("observed_range")
        router = FixedBoundaryRouter(
            payload["low"], payload["high"],
            feature_name=payload["feature_name"],
            observed_range=tuple(observed) if observed else None)
        mapping = {
            int(key): value for key, value in payload["cluster_mapping"].items()}
        _validate_mapping(mapping)
        if mapping != {0: "A", 1: "B", 2: "C"}:
            raise ValueError(
                "fixed boundary cluster_mapping must be 0:A, 1:B, 2:C")
        return router
    if payload.get("router_type") == "centroid":
        return CentroidRouter.from_payload(payload)
    router_type = payload.get("router_type")
    raise ValueError(f"unsupported router_type: {router_type!r}")
