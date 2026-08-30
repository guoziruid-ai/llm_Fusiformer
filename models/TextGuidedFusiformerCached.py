#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cache-aware, encoder-agnostic TextGuidedFusiformer.

Online path:
    tokens -> arbitrary Hugging Face text encoder -> pooling -> text_proj

Cached path:
    precomputed pooled [B, D_text] -> text_proj

The graph/fusion architecture is intentionally unchanged.  Legacy Qwen
constructor names remain accepted so older training utilities do not break.
"""

from __future__ import annotations

from functools import partial

import torch
from dgl.nn import AvgPooling
from torch import nn

from .TextGuidedFusiformer import TextGuidedFusiformerBlock
from .layers import RBFExpansion
from .text_encoders import build_text_encoder


class TextGuidedFusiformer(nn.Module):
    def __init__(
        self,
        targets=None,
        depth=4,
        edge_input_dim=80,
        triplet_input_dim=40,
        embed_dim=128,
        num_heads=4,
        mlp_ratio=2.0,
        use_bias=True,
        norm_layer=None,
        act_layer=None,
        text_dim=1536,
        # Multi-layer text projection
        text_proj_dim=64,
        text_proj_hidden_dims=(256, 128),
        text_proj_dropout=0.1,
        # New generic API
        text_encoder_name="auto",
        text_model_name=None,
        text_pooling="mean",
        text_preprocessing="auto",
        freeze_text_encoder=True,
        load_text_encoder=True,
        allow_remote_model=False,
        # Legacy compatibility API
        qwen_model_name=None,
        freeze_qwen=None,
        load_qwen=None,
    ):
        super().__init__()
        targets = [] if targets is None else targets

        if text_model_name is None:
            text_model_name = qwen_model_name
        if text_model_name is None:
            text_model_name = "/root/models/Qwen2.5-1.5B-Instruct"
        if freeze_qwen is not None:
            freeze_text_encoder = bool(freeze_qwen)
        if load_qwen is not None:
            load_text_encoder = bool(load_qwen)

        if not load_text_encoder and not freeze_text_encoder:
            raise ValueError(
                "load_text_encoder=False is only valid when freeze_text_encoder=True. "
                "A trainable text encoder cannot use a static embedding cache."
            )

        self.targets = targets
        self.embed_dim = int(embed_dim)
        self.text_dim = int(text_dim)
        self.text_proj_dim = int(text_proj_dim)
        self.text_proj_hidden_dims = tuple(int(x) for x in text_proj_hidden_dims)
        self.text_proj_dropout = float(text_proj_dropout)

        if self.text_proj_dim <= 0:
            raise ValueError(f"text_proj_dim must be > 0, got {self.text_proj_dim}")
        if any(x <= 0 for x in self.text_proj_hidden_dims):
            raise ValueError(
                "all text_proj_hidden_dims must be > 0, got "
                f"{self.text_proj_hidden_dims}"
            )
        if not 0.0 <= self.text_proj_dropout < 1.0:
            raise ValueError(
                "text_proj_dropout must be in [0, 1), got "
                f"{self.text_proj_dropout}"
            )

        self.text_encoder_name = str(text_encoder_name)
        self.text_model_name = str(text_model_name)
        self.text_pooling = str(text_pooling)
        self.text_preprocessing = str(text_preprocessing)
        self.freeze_text_encoder = bool(freeze_text_encoder)
        self.load_text_encoder = bool(load_text_encoder)
        self.allow_remote_model = bool(allow_remote_model)

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.ReLU

        self.atom_embedding = nn.Embedding(95, embed_dim)
        self.edge_embedding = nn.Sequential(
            RBFExpansion(vmin=0, vmax=8.0, bins=edge_input_dim),
            nn.Linear(edge_input_dim, embed_dim),
            norm_layer(embed_dim),
            act_layer(),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(vmin=-1, vmax=1.0, bins=triplet_input_dim),
            nn.Linear(triplet_input_dim, embed_dim),
            norm_layer(embed_dim),
            act_layer(),
        )

        # Always trainable in cache mode.
        # Progressive projection: D_text -> hidden dims -> text_proj_dim.
        # Hidden stages use GELU + Dropout; the final stage intentionally
        # omits activation so cross-attention receives signed features.
        projection_dims = [
            self.text_dim,
            *self.text_proj_hidden_dims,
            self.text_proj_dim,
        ]
        projection_layers = []
        for i in range(len(projection_dims) - 1):
            in_dim = projection_dims[i]
            out_dim = projection_dims[i + 1]
            projection_layers.append(nn.Linear(in_dim, out_dim))
            projection_layers.append(norm_layer(out_dim))

            is_last = i == len(projection_dims) - 2
            if not is_last:
                projection_layers.append(nn.GELU())
                if self.text_proj_dropout > 0:
                    projection_layers.append(nn.Dropout(self.text_proj_dropout))

        self.text_proj = nn.Sequential(*projection_layers)

        self.text_encoder = None
        if self.load_text_encoder:
            self.text_encoder = build_text_encoder(
                self.text_encoder_name,
                model_name=self.text_model_name,
                pooling=self.text_pooling,
                preprocessing=self.text_preprocessing,
                freeze=self.freeze_text_encoder,
                local_files_only=not self.allow_remote_model,
                load_tokenizer=False,
                load_model=True,
            )
            if int(self.text_encoder.hidden_size) != self.text_dim:
                raise ValueError(
                    "text_dim does not match text encoder hidden_size: "
                    f"text_dim={self.text_dim}, "
                    f"hidden_size={self.text_encoder.hidden_size}, "
                    f"model={self.text_model_name}"
                )

        self.blocks = nn.ModuleList(
            [
                TextGuidedFusiformerBlock(
                    dim=embed_dim,
                    text_dim=self.text_proj_dim,
                    num_heads=num_heads,
                    use_bias=use_bias,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )

        self.readout = AvgPooling()
        self.regression_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, len(targets) if targets else 1),
        )

    # ------------------------------------------------------------------
    # Legacy compatibility properties.  They are intentionally properties,
    # not duplicate registered modules, so state_dict uses text_encoder.*.
    # ------------------------------------------------------------------
    @property
    def qwen(self):
        return self.text_encoder

    @property
    def qwen_model_name(self):
        return self.text_model_name

    @property
    def freeze_qwen(self):
        return self.freeze_text_encoder

    @freeze_qwen.setter
    def freeze_qwen(self, value):
        self.freeze_text_encoder = bool(value)

    @property
    def load_qwen(self):
        return self.load_text_encoder

    def encode_text(self, text_tokens):
        if self.text_encoder is None:
            raise RuntimeError(
                "Text encoder is not loaded. Provide inputs['text_emb'] from the "
                "precomputed embedding cache, or enable load_text_encoder."
            )
        pooled = self.text_encoder.encode_tokens(text_tokens)
        return self.text_proj(pooled)

    def project_cached_text(self, pooled_text_emb: torch.Tensor) -> torch.Tensor:
        if pooled_text_emb.ndim != 2:
            raise ValueError(
                "cached text embedding must have shape [B, text_dim], "
                f"got {tuple(pooled_text_emb.shape)}"
            )
        expected = int(self.text_proj[0].in_features)
        if int(pooled_text_emb.shape[-1]) != expected:
            raise ValueError(
                "cached text embedding dimension mismatch: "
                f"got {pooled_text_emb.shape[-1]}, expected {expected}. "
                "Do not reuse a Qwen cache with MatSciBERT or vice versa."
            )
        pooled_text_emb = pooled_text_emb.to(dtype=self.text_proj[0].weight.dtype)
        return self.text_proj(pooled_text_emb)

    def forward(self, inputs):
        graph = inputs.get("graph_input")
        if graph is None:
            raise KeyError("inputs is missing 'graph_input'")
        if len(graph) != 2:
            raise ValueError("graph_input must contain graph and line_graph")

        g, lg = graph
        g = g.local_var()
        lg = lg.local_var()

        text_mode = str(inputs.get("text_mode", "normal")).strip().lower()
        valid_text_modes = {"normal", "bypass"}
        if text_mode not in valid_text_modes:
            raise ValueError(
                f"unsupported text_mode={text_mode!r}; choices={sorted(valid_text_modes)}"
            )
        enable_text_guidance = text_mode == "normal"

        batch_num_nodes_g = g.batch_num_nodes()
        batch_num_edges_g = g.batch_num_edges()
        batch_num_nodes_lg = lg.batch_num_nodes()
        batch_num_edges_lg = lg.batch_num_edges()

        if enable_text_guidance:
            cached_text_emb = inputs.get("text_emb")
            text_tokens = inputs.get("text")
            # Cache always wins: an online model is never accidentally executed
            # when both representations are present.
            if cached_text_emb is not None:
                text_emb = self.project_cached_text(cached_text_emb)
            elif text_tokens is not None:
                text_emb = self.encode_text(text_tokens)
            else:
                batch_size = len(batch_num_nodes_g)
                text_emb = torch.zeros(
                    batch_size,
                    self.text_proj_dim,
                    device=g.device,
                    dtype=self.text_proj[0].weight.dtype,
                )
        else:
            text_emb = None

        z = self.angle_embedding(lg.edata.pop("angle"))
        x = self.atom_embedding(g.ndata.pop("atomic_numbers") - 1)
        y = self.edge_embedding(g.edata.pop("distance"))

        for block in self.blocks:
            x, y, z = block(
                g,
                x,
                y,
                lg,
                z,
                text_emb,
                batch_num_nodes_g,
                batch_num_edges_g,
                batch_num_nodes_lg,
                batch_num_edges_lg,
                enable_text_guidance=enable_text_guidance,
            )

        graph_out = self.readout(g, x)
        return self.regression_head(graph_out)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_text_encoder and self.text_encoder is not None:
            self.text_encoder.eval()
        return self
