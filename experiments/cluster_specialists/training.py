"""Shared lightweight training loop for independently constructed experts."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterable

import torch

from .config import PredictionMode


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    best_validation_loss: float
    history: list[dict[str, float | int]]
    state_dict: dict[str, torch.Tensor]


def _prediction_target(targets: torch.Tensor,
                       baseline_predictions: torch.Tensor | None,
                       mode: PredictionMode) -> torch.Tensor:
    targets = targets.reshape(-1, 1)
    if mode is PredictionMode.DIRECT:
        return targets
    if baseline_predictions is None:
        raise ValueError(
            "residual training requires training-split baseline predictions")
    if baseline_predictions.shape != targets.shape:
        raise ValueError(
            "baseline predictions and targets must have matching shapes")
    return targets - baseline_predictions.detach()


def train_tensor_expert(
        model: torch.nn.Module,
        train_loader: Iterable[tuple[torch.Tensor, ...]],
        validation_loader: Iterable[tuple[torch.Tensor, ...]], *,
        mode: str | PredictionMode = PredictionMode.DIRECT,
        epochs: int = 10, learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        loss_name: str = "mse",
        device: str | torch.device = "cpu") -> TrainingResult:
    """Train one expert only; loaders yield (x,y) or (x,y,baseline_pred)."""
    prediction_mode = PredictionMode(mode)
    if epochs < 1 or learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive")
    if loss_name not in {"mse", "mae"}:
        raise ValueError("loss_name must be mse or mae")
    device = torch.device(device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = (torch.nn.MSELoss() if loss_name == "mse"
                 else torch.nn.L1Loss())
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_sum = 0.0
        train_count = 0
        for batch in train_loader:
            if len(batch) not in {2, 3}:
                raise ValueError(
                    "loader batches must be (inputs,targets[,baseline])")
            inputs, targets = batch[0].to(device), batch[1].to(device)
            baseline = batch[2].to(device).reshape(-1, 1) if len(batch) == 3 else None
            expected = _prediction_target(targets, baseline, prediction_mode)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs).reshape(-1, 1)
            loss = criterion(prediction, expected)
            loss.backward()
            optimizer.step()
            train_sum += float(loss.detach()) * len(expected)
            train_count += len(expected)

        model.eval()
        validation_sum = 0.0
        validation_count = 0
        with torch.inference_mode():
            for batch in validation_loader:
                inputs, targets = batch[0].to(device), batch[1].to(device)
                baseline = (
                    batch[2].to(device).reshape(-1, 1)
                    if len(batch) == 3 else None)
                expected = _prediction_target(
                    targets, baseline, prediction_mode)
                loss = criterion(
                    model(inputs).reshape(-1, 1), expected)
                validation_sum += float(loss) * len(expected)
                validation_count += len(expected)
        if not train_count or not validation_count:
            raise ValueError(
                "training and validation loaders must both be non-empty")
        validation_loss = validation_sum / validation_count
        history.append({
            "epoch": epoch, "train_loss": train_sum / train_count,
            "validation_loss": validation_loss})
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    return TrainingResult(
        best_epoch, best_loss, history, copy.deepcopy(best_state))
