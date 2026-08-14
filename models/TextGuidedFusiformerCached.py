#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cache-aware TextGuidedFusiformer.

The graph architecture is reused from models/TextGuidedFusiformer.py.
Only the text path is changed:

online:
    tokens -> frozen Qwen -> masked mean pool -> text_proj

cached:
    cached pooled Qwen embedding -> text_proj

The trainable text_proj and all downstream cross-attention layers remain
unchanged, so the cache does NOT freeze text_proj.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from functools import partial

import torch
from dgl.nn import AvgPooling
from torch import nn
from transformers import AutoModel

from .TextGuidedFusiformer import TextGuidedFusiformerBlock
from .layers import RBFExpansion


class TextGuidedFusiformer(nn.Module):
    def __init__(
        self,
        targets=[],
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
        qwen_model_name="/root/models/Qwen2.5-1.5B-Instruct",
        freeze_qwen=True,
        load_qwen=True,
    ):
        super().__init__()

        if not load_qwen and not freeze_qwen:
            raise ValueError(
                "load_qwen=False is only valid when freeze_qwen=True. "
                "Trainable Qwen cannot use a static embedding cache."
            )

        self.targets = targets
        self.freeze_qwen = bool(freeze_qwen)
        self.load_qwen = bool(load_qwen)
        self.qwen_model_name = qwen_model_name
        self.text_dim = int(text_dim)

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

        # This remains trainable in cache mode.
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, embed_dim),
            norm_layer(embed_dim),
            act_layer(),
        )

        self.qwen = None
        if self.load_qwen:
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            self.qwen = AutoModel.from_pretrained(
                qwen_model_name,
                trust_remote_code=True,
                output_hidden_states=False,
                local_files_only=True,
            )

            if self.freeze_qwen:
                for param in self.qwen.parameters():
                    param.requires_grad = False
                self.qwen.eval()
            else:
                self.qwen.train()

        self.blocks = nn.ModuleList(
            [
                TextGuidedFusiformerBlock(
                    dim=embed_dim,
                    text_dim=embed_dim,
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

    @staticmethod
    def _masked_mean_pool(
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if attention_mask is None:
            return last_hidden_state.mean(dim=1)

        # Keep the same pooling semantics as the original project:
        # multiplying by a float32 mask makes the pooled representation float32.
        mask = attention_mask.unsqueeze(-1).expand_as(last_hidden_state).float()
        return (
            (last_hidden_state * mask).sum(dim=1)
            / mask.sum(dim=1).clamp(min=1e-6)
        )

    def encode_text(self, text_tokens):
        """Online fallback: tokenize outside, run Qwen here, then text_proj."""
        if self.qwen is None:
            raise RuntimeError(
                "Qwen is not loaded in this model. "
                "Provide inputs['text_emb'] from the precomputed cache."
            )

        ctx = torch.no_grad() if self.freeze_qwen else nullcontext()
        with ctx:
            outputs = self.qwen(
                input_ids=text_tokens["input_ids"],
                attention_mask=text_tokens.get("attention_mask", None),
                output_hidden_states=False,
                use_cache=False,
            )
            pooled = self._masked_mean_pool(
                outputs.last_hidden_state,
                text_tokens.get("attention_mask", None),
            )

        return self.text_proj(pooled)

    def project_cached_text(self, pooled_text_emb: torch.Tensor) -> torch.Tensor:
        """Project cached [B, text_dim] pooled Qwen embeddings."""
        if pooled_text_emb.ndim != 2:
            raise ValueError(
                "cached text embedding must have shape [B, text_dim], "
                f"got {tuple(pooled_text_emb.shape)}"
            )

        expected = int(self.text_proj[0].in_features)
        if int(pooled_text_emb.shape[-1]) != expected:
            raise ValueError(
                "cached text embedding dimension mismatch: "
                f"got {pooled_text_emb.shape[-1]}, expected {expected}"
            )

        pooled_text_emb = pooled_text_emb.to(
            dtype=self.text_proj[0].weight.dtype,
        )
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
                f"unsupported text_mode={text_mode!r}; "
                f"choices={sorted(valid_text_modes)}"
            )
        enable_text_guidance = text_mode == "normal"

        batch_num_nodes_g = g.batch_num_nodes()
        batch_num_edges_g = g.batch_num_edges()
        batch_num_nodes_lg = lg.batch_num_nodes()
        batch_num_edges_lg = lg.batch_num_edges()

        if enable_text_guidance:
            cached_text_emb = inputs.get("text_emb")
            text_tokens = inputs.get("text")

            # Cache has priority so Qwen is never accidentally executed when
            # both representations are present.
            if cached_text_emb is not None:
                text_emb = self.project_cached_text(cached_text_emb)
            elif text_tokens is not None:
                text_emb = self.encode_text(text_tokens)
            else:
                batch_size = len(batch_num_nodes_g)
                text_emb = torch.zeros(
                    batch_size,
                    self.text_proj[0].out_features,
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
        if self.freeze_qwen and self.qwen is not None:
            self.qwen.eval()
        return self
