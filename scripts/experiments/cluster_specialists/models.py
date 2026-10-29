"""Independent specialist models and V2-compatible routing adapters."""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn

from .config import PredictionMode


class TensorMLPExpert(nn.Module):
    """Small reference expert used for features, smoke tests and custom studies."""

    def __init__(self, input_dim: int,
                 hidden_dims: Sequence[int] = (64, 32),
                 dropout: float = 0.0):
        super().__init__()
        if input_dim < 1 or any(int(dim) < 1 for dim in hidden_dims):
            raise ValueError("input and hidden dimensions must be positive")
        layers: list[nn.Module] = []
        previous = int(input_dim)
        for hidden in hidden_dims:
            layers.extend([
                nn.Linear(previous, int(hidden)), nn.GELU(),
                nn.Dropout(float(dropout))])
            previous = int(hidden)
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(inputs) or inputs.ndim != 2:
            raise ValueError("TensorMLPExpert expects [B,D] tensor inputs")
        return self.network(inputs)


class SpecialistAdapter(nn.Module):
    """Unify direct and frozen-baseline residual prediction."""

    def __init__(self, specialist: nn.Module, *,
                 mode: str | PredictionMode = PredictionMode.DIRECT,
                 baseline: nn.Module | None = None):
        super().__init__()
        self.specialist = specialist
        self.mode = PredictionMode(mode)
        self.baseline = baseline
        if self.mode is PredictionMode.RESIDUAL and baseline is None:
            raise ValueError("residual mode requires an explicit baseline model")
        if baseline is not None:
            baseline.requires_grad_(False)
            baseline.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.baseline is not None:
            self.baseline.eval()
        return self

    def forward(self, inputs: Any) -> torch.Tensor:
        correction = self.specialist(inputs)
        if self.mode is PredictionMode.DIRECT:
            return correction
        with torch.no_grad():
            baseline = self.baseline(inputs)
        if correction.shape != baseline.shape:
            raise ValueError(
                "baseline and residual prediction shapes do not match")
        return baseline + correction


def build_v2_specialist(*, model_kwargs: Mapping[str, Any],
                        mode: str | PredictionMode = PredictionMode.DIRECT,
                        baseline: nn.Module | None = None) -> SpecialistAdapter:
    """Construct a real FusiformerV2 without duplicating its implementation."""
    from fusiformer_v2.model import FusiformerV2

    return SpecialistAdapter(
        FusiformerV2(**dict(model_kwargs)), mode=mode, baseline=baseline)


def _slice_graph(graph: Any, positions: list[int]) -> Any:
    try:
        import dgl
        return dgl.batch([dgl.unbatch(graph)[position] for position in positions])
    except Exception as exc:
        raise TypeError(
            "graph batch slicing requires a compatible DGL batched graph") from exc


def slice_batch(value: Any, positions: list[int], batch_size: int) -> Any:
    """Select samples from tensors, nested V2 inputs or DGL graph pairs."""
    if torch.is_tensor(value):
        if value.ndim > 0 and len(value) == batch_size:
            return value[positions]
        return value
    if isinstance(value, dict):
        return {
            key: slice_batch(item, positions, batch_size)
            for key, item in value.items()}
    if (isinstance(value, list) and len(value) == 2
            and all(hasattr(item, "batch_num_nodes") for item in value)):
        return [_slice_graph(item, positions) for item in value]
    if isinstance(value, tuple):
        return tuple(slice_batch(item, positions, batch_size) for item in value)
    return value


class RoutedSpecialistSystem(nn.Module):
    """Run A/B/C experts independently and restore original batch order."""

    def __init__(self, router: Any, experts: Mapping[str, nn.Module]):
        super().__init__()
        if set(experts) != {"A", "B", "C"}:
            raise ValueError("experts must contain exactly A, B and C")
        self.router = router
        self.experts = nn.ModuleDict(dict(experts))

    def forward(self, inputs: Any, router_features: Any
                ) -> tuple[torch.Tensor, list[Any]]:
        decisions = self.router.predict_diagnostics(router_features)
        batch_size = len(decisions)
        first_parameter = next(self.parameters(), None)
        device = (first_parameter.device if first_parameter is not None
                  else torch.device("cpu"))
        output = torch.empty((batch_size, 1), device=device)
        for cluster in ("A", "B", "C"):
            positions = [
                index for index, decision in enumerate(decisions)
                if decision.cluster_id == cluster]
            if not positions:
                continue
            selected = slice_batch(inputs, positions, batch_size)
            predicted = self.experts[cluster](selected)
            predicted = predicted.reshape(len(positions), -1)
            if predicted.shape[1] != 1:
                raise ValueError("each specialist must emit one scalar per sample")
            output[positions] = predicted
        return output, decisions
