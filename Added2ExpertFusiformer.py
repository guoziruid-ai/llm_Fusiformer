#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Added-2 Expert Fusiformer
=========================

Architecture used by the Grouped-3 Expert experiment:

    frozen Base depth-6 Fusiformer
                  |
                  +---- Base prediction (optional frozen residual anchor)
                  |
    copied depth-6 shared stem inside depth-8 Expert branch
                  |
          Added Block7 (trainable)
                  |
          Added Block8 (trainable)
                  |
       regression_head (trainable)
                  |
           Expert prediction
                  |
    y = y_base + gate * (y_expert - y_base)      [default]

The depth-8 branch is built by the project's existing
models.TextGuidedFusiformerCached.TextGuidedFusiformer, so the two newly
added graph blocks are inserted by the original model's own forward path
before readout/regression_head. This avoids reimplementing the project's
graph-message-passing internals.

Default stabilization:
- Blocks 1-6: copied from the depth-6 Base checkpoint and frozen.
- Blocks 7-8: initialized by cloning Base Block6.
- regression_head: copied from the Base checkpoint and trainable.
- residual gate: sigmoid parameter initialized to a small positive value.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn

from models.TextGuidedFusiformerCached import TextGuidedFusiformer


def _normalize_outputs(outputs: Any) -> torch.Tensor:
    if torch.is_tensor(outputs):
        return outputs
    if isinstance(outputs, (list, tuple)):
        tensors = [x for x in outputs if torch.is_tensor(x)]
        if not tensors:
            raise TypeError("Model output tuple/list contains no tensors")
        return torch.cat(tensors, dim=-1)
    raise TypeError(f"Unsupported model output type: {type(outputs)}")


def _clone_value(value: Any) -> Any:
    """Best-effort clone so the frozen Base and Expert branch do not mutate
    the same graph container in-place during the two forward passes.
    """
    if isinstance(value, dict):
        return {k: _clone_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(v) for v in value)
    if torch.is_tensor(value):
        return value
    if hasattr(value, "clone"):
        try:
            return value.clone()
        except Exception:
            return value
    return value


def extract_state_dict(checkpoint_or_state: Any) -> dict[str, torch.Tensor]:
    """Accept the Base project's common checkpoint formats."""
    if not isinstance(checkpoint_or_state, Mapping):
        raise TypeError("checkpoint_or_state must be a mapping")

    if "model_state_dict" in checkpoint_or_state:
        state = checkpoint_or_state["model_state_dict"]
    elif "model" in checkpoint_or_state:
        state = checkpoint_or_state["model"]
    elif "state_dict" in checkpoint_or_state:
        state = checkpoint_or_state["state_dict"]
    elif checkpoint_or_state and all(
        torch.is_tensor(v) for v in checkpoint_or_state.values()
    ):
        state = checkpoint_or_state
    else:
        raise KeyError("Checkpoint does not contain a recognizable model state")

    state = dict(state)
    if state and all(name.startswith("module.") for name in state):
        state = {name[len("module."):]: value for name, value in state.items()}
    return state


def _load_compatible_state(
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
) -> tuple[list[str], list[str]]:
    current = model.state_dict()
    compatible = {
        name: value
        for name, value in state.items()
        if name in current and tuple(current[name].shape) == tuple(value.shape)
    }
    incompatible = model.load_state_dict(compatible, strict=False)
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys)


def _build_project_model(
    *,
    depth: int,
    target: str,
    edge_input_dim: int,
    triplet_input_dim: int,
    embed_dim: int,
    num_heads: int,
    text_hidden_size: int,
    text_proj_dim: int,
    text_proj_hidden_dims: tuple[int, ...],
    text_proj_dropout: float,
    text_encoder_name: str,
    text_model_name: str,
    text_pooling: str,
    text_preprocessing: str,
    freeze_text_encoder: bool,
    use_precomputed_text_embeddings: bool,
    allow_remote_model: bool,
) -> TextGuidedFusiformer:
    model = TextGuidedFusiformer(
        targets=[target],
        depth=int(depth),
        edge_input_dim=int(edge_input_dim),
        triplet_input_dim=int(triplet_input_dim),
        embed_dim=int(embed_dim),
        num_heads=int(num_heads),
        text_dim=int(text_hidden_size),
        text_proj_dim=int(text_proj_dim),
        text_proj_hidden_dims=tuple(int(x) for x in text_proj_hidden_dims),
        text_proj_dropout=float(text_proj_dropout),
        text_encoder_name=str(text_encoder_name),
        text_model_name=str(text_model_name),
        text_pooling=str(text_pooling),
        text_preprocessing=str(text_preprocessing),
        freeze_text_encoder=bool(freeze_text_encoder),
        load_text_encoder=not bool(use_precomputed_text_embeddings),
        allow_remote_model=bool(allow_remote_model),
    )
    model.use_precomputed_text_embeddings = bool(use_precomputed_text_embeddings)
    return model


class Added2ExpertFusiformer(nn.Module):
    """Frozen depth-6 Base + trainable added Block7/8 + trainable Base-copied head.

    The Expert branch is an ordinary project TextGuidedFusiformer with
    depth = base_depth + extra_depth. Base checkpoint tensors are copied into all
    matching modules, so blocks 0..base_depth-1 and regression_head start from
    the trained Base.

    When use_residual_gate=True, a frozen depth-6 Base is retained as an anchor:

        y = y_base + sigmoid(residual_logit) * (y_expert - y_base)

    This makes the initial specialist close to the Base prediction while still
    allowing gradients into the newly added graph blocks from step 1.
    """

    def __init__(
        self,
        *,
        base_checkpoint: Any,
        target: str,
        base_depth: int,
        extra_depth: int,
        edge_input_dim: int,
        triplet_input_dim: int,
        embed_dim: int,
        num_heads: int,
        text_hidden_size: int,
        text_proj_dim: int,
        text_proj_hidden_dims: tuple[int, ...],
        text_proj_dropout: float,
        text_encoder_name: str,
        text_model_name: str,
        text_pooling: str,
        text_preprocessing: str,
        freeze_text_encoder: bool,
        use_precomputed_text_embeddings: bool,
        allow_remote_model: bool,
        added_block_init: str = "clone_last",
        use_residual_gate: bool = True,
        residual_gate_init: float = 0.05,
    ) -> None:
        super().__init__()
        if base_depth < 1:
            raise ValueError("base_depth must be >= 1")
        if extra_depth < 1:
            raise ValueError("extra_depth must be >= 1")
        if added_block_init not in {"clone_last", "random"}:
            raise ValueError("added_block_init must be 'clone_last' or 'random'")
        if not (0.0 < residual_gate_init < 1.0):
            raise ValueError("residual_gate_init must be in (0,1)")

        self.base_depth = int(base_depth)
        self.extra_depth = int(extra_depth)
        self.total_depth = self.base_depth + self.extra_depth
        self.added_block_init = str(added_block_init)
        self.use_residual_gate = bool(use_residual_gate)

        common = dict(
            target=target,
            edge_input_dim=edge_input_dim,
            triplet_input_dim=triplet_input_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            text_hidden_size=text_hidden_size,
            text_proj_dim=text_proj_dim,
            text_proj_hidden_dims=text_proj_hidden_dims,
            text_proj_dropout=text_proj_dropout,
            text_encoder_name=text_encoder_name,
            text_model_name=text_model_name,
            text_pooling=text_pooling,
            text_preprocessing=text_preprocessing,
            freeze_text_encoder=freeze_text_encoder,
            use_precomputed_text_embeddings=use_precomputed_text_embeddings,
            allow_remote_model=allow_remote_model,
        )

        state = extract_state_dict(base_checkpoint)

        # Frozen Base anchor used by the graph-only router elsewhere and, when
        # enabled, by the residual prediction gate here.
        self.base_model: Optional[TextGuidedFusiformer]
        if self.use_residual_gate:
            self.base_model = _build_project_model(depth=self.base_depth, **common)
            _load_compatible_state(self.base_model, state)
        else:
            self.base_model = None

        # Actual Expert branch: same trained 6-layer stem + two extra graph blocks.
        self.expert_model = _build_project_model(depth=self.total_depth, **common)
        _load_compatible_state(self.expert_model, state)

        if not hasattr(self.expert_model, "blocks"):
            raise AttributeError("TextGuidedFusiformer has no blocks attribute")
        if len(self.expert_model.blocks) != self.total_depth:
            raise RuntimeError(
                f"Expected {self.total_depth} blocks, got {len(self.expert_model.blocks)}"
            )

        # Stable initialization for the two new graph blocks.
        if self.added_block_init == "clone_last":
            source_state = copy.deepcopy(
                self.expert_model.blocks[self.base_depth - 1].state_dict()
            )
            for idx in range(self.base_depth, self.total_depth):
                self.expert_model.blocks[idx].load_state_dict(source_state, strict=True)

        # The residual gate is deliberately small but non-zero. A zero gate would
        # also zero the new-branch gradient at initialization.
        init_logit = math.log(residual_gate_init / (1.0 - residual_gate_init))
        self.residual_logit = nn.Parameter(
            torch.tensor(init_logit, dtype=torch.float32),
            requires_grad=self.use_residual_gate,
        )

        self.configure_trainable_parameters()
        self.set_trainable_mode()

    @property
    def residual_gate(self) -> torch.Tensor:
        if not self.use_residual_gate:
            return torch.ones((), device=self.residual_logit.device)
        return torch.sigmoid(self.residual_logit)

    def configure_trainable_parameters(self) -> None:
        # Freeze absolutely everything first.
        if self.base_model is not None:
            for p in self.base_model.parameters():
                p.requires_grad = False
        for p in self.expert_model.parameters():
            p.requires_grad = False

        # Only newly added graph blocks and the copied regression head are trained.
        for idx in range(self.base_depth, self.total_depth):
            for p in self.expert_model.blocks[idx].parameters():
                p.requires_grad = True

        if not hasattr(self.expert_model, "regression_head"):
            raise AttributeError("TextGuidedFusiformer has no regression_head")
        for p in self.expert_model.regression_head.parameters():
            p.requires_grad = True

        self.residual_logit.requires_grad = self.use_residual_gate

    def set_trainable_mode(self) -> None:
        # Frozen stem must remain eval so dropout in frozen modules cannot inject
        # noise. Call nn.Module.train directly to avoid recursive eval/train
        # dispatch through this class.
        nn.Module.train(self, False)
        if self.base_model is not None:
            self.base_model.eval()
        self.expert_model.eval()
        for idx in range(self.base_depth, self.total_depth):
            self.expert_model.blocks[idx].train()
        self.expert_model.regression_head.train()

    def train(self, mode: bool = True):
        # Generic training code can safely call model.train()/model.eval().
        if not mode:
            nn.Module.train(self, False)
            return self
        self.set_trainable_mode()
        return self

    def forward(self, inputs: Any) -> torch.Tensor:
        # Preserve a clean copy for the frozen Base anchor before the Expert
        # forward, because graph models may write temporary node/edge features.
        base_inputs = _clone_value(inputs) if self.use_residual_gate else None

        expert_pred = _normalize_outputs(self.expert_model(inputs))
        if not self.use_residual_gate:
            return expert_pred

        if self.base_model is None:
            raise RuntimeError("Residual gate enabled but base_model is missing")
        with torch.no_grad():
            base_pred = _normalize_outputs(self.base_model(base_inputs))
        gate = self.residual_gate.to(dtype=expert_pred.dtype)
        return base_pred + gate * (expert_pred - base_pred)

    def trainable_summary(self) -> dict[str, Any]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        added_blocks = list(range(self.base_depth, self.total_depth))
        return {
            "architecture": "frozen_base_plus_added2_expert",
            "base_depth": self.base_depth,
            "extra_depth": self.extra_depth,
            "total_depth": self.total_depth,
            "added_block_indices_zero_based": added_blocks,
            "added_block_numbers_one_based": [i + 1 for i in added_blocks],
            "added_block_init": self.added_block_init,
            "trainable_head": True,
            "use_residual_gate": self.use_residual_gate,
            "residual_gate_initial_value": float(self.residual_gate.detach().cpu()),
            # Canonical keys used by the training program.
            "total_parameter_count": int(total),
            "trainable_parameter_count": int(trainable),
            "trainable_fraction": float(trainable / max(total, 1)),
            # Runtime aliases kept for diagnostics/backward compatibility.
            "total_parameter_count_runtime": int(total),
            "trainable_fraction_runtime": float(trainable / max(total, 1)),
            "trainable_names": [
                name for name, p in self.named_parameters() if p.requires_grad
            ],
        }

    def compact_state_dict(self) -> dict[str, torch.Tensor]:
        """Save only parameters that differ between Experts.

        The Base checkpoint is stored once by the training/deployment bundle.
        """
        result: dict[str, torch.Tensor] = {}
        for idx in range(self.base_depth, self.total_depth):
            prefix = f"expert_model.blocks.{idx}."
            for name, value in self.expert_model.blocks[idx].state_dict().items():
                result[prefix + name] = value.detach().cpu()

        for name, value in self.expert_model.regression_head.state_dict().items():
            result["expert_model.regression_head." + name] = value.detach().cpu()

        result["residual_logit"] = self.residual_logit.detach().cpu()
        return result

    def load_compact_state_dict(
        self,
        state: Mapping[str, torch.Tensor],
        *,
        strict: bool = True,
    ) -> None:
        full = self.state_dict()
        unknown = [name for name in state if name not in full]
        if unknown and strict:
            raise KeyError(f"Unknown compact-state keys: {unknown[:20]}")
        for name, value in state.items():
            if name in full:
                if tuple(full[name].shape) != tuple(value.shape):
                    raise RuntimeError(
                        f"Shape mismatch for {name}: "
                        f"model={tuple(full[name].shape)} state={tuple(value.shape)}"
                    )
                full[name] = value
        self.load_state_dict(full, strict=True)
        self.configure_trainable_parameters()

    def expert_only_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
