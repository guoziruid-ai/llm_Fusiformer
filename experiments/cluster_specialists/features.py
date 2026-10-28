"""Read-only feature extraction adapters for the existing V2 baseline."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if hasattr(value, "to"):
        try:
            return value.to(device, non_blocking=True)
        except TypeError:
            return value.to(device)
    return value


def extract_v2_graph_latents(model: torch.nn.Module, loader: Any, *,
                             device: str | torch.device = "cpu"
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Capture V2 readout vectors and graph-only predictions in loader order.

    The adapter uses a temporary readout hook and `text_mode="bypass"`. It does
    not change FusiformerV2, optimize parameters, or accept target labels.
    """
    if not hasattr(model, "readout"):
        raise TypeError("V2 feature extraction requires model.readout")
    device = torch.device(device)
    model.to(device).eval()
    captured: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    batch_capture: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        if not torch.is_tensor(output) or output.ndim != 2:
            raise ValueError("V2 readout hook expected a [B,D] tensor")
        batch_capture.append(output.detach().float().cpu())

    handle = model.readout.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            for batch in loader:
                inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
                if not isinstance(inputs, dict):
                    raise TypeError("V2 loader must yield an input dictionary")
                inputs = _to_device(dict(inputs), device)
                inputs["text_mode"] = "bypass"
                batch_capture.clear()
                output = model(inputs)
                if len(batch_capture) != 1:
                    raise RuntimeError(
                        "expected exactly one V2 readout call per batch")
                captured.append(batch_capture[0])
                predictions.append(
                    output.detach().float().reshape(-1, 1).cpu())
    finally:
        handle.remove()
    if not captured:
        raise ValueError("feature extraction loader is empty")
    latents = torch.cat(captured).numpy()
    outputs = torch.cat(predictions).reshape(-1).numpy()
    if len(latents) != len(outputs) or not (
            np.isfinite(latents).all() and np.isfinite(outputs).all()):
        raise ValueError("V2 latent/prediction extraction is invalid")
    return latents, outputs
