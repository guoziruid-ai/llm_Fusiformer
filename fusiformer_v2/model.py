"""Fusiformer-V2 integrated with the real Abstract-Text-Embedding-Module branch.

Reuses coGN/line-graph blocks, atom/edge/angle embeddings, text projection,
AvgPooling, and regression head from models/TextGuidedFusiformerCached.py.
Stages: a1=semantic-token masked mean + legacy text-guidance;
        a2=graph-only coGN + atom<-text cross-attn;
        a3=a2 + material query pooling.

The default is A2. A1 deliberately shares the original fusion architecture
and isolates only the *segment cache* change. No checkpoint is overwritten.
"""
from __future__ import annotations

import torch
from torch import nn
from models.TextGuidedFusiformerCached import TextGuidedFusiformer
from .attention import AtomTextCrossAttention, MaterialQueryPooler, masked_mean


class FusiformerV2(TextGuidedFusiformer):
    def __init__(self, *, stage: str = "a2", num_queries: int = 16,
                 cross_dropout: float = 0.1, **kwargs):
        if stage not in {"a1", "a2", "a3"}:
            raise ValueError("stage must be a1, a2, or a3")
        kwargs["freeze_text_encoder"] = True
        kwargs["load_text_encoder"] = False
        super().__init__(**kwargs)
        self.stage = stage
        if stage in {"a2", "a3"}:
            self.cross_modal = AtomTextCrossAttention(
                dim=self.embed_dim, heads=kwargs.get("num_heads", 4), dropout=cross_dropout)
            self.v2_text_to_graph = nn.Linear(self.text_proj_dim, self.embed_dim)
        if stage == "a3":
            self.material_queries = MaterialQueryPooler(
                dim=self.embed_dim, heads=kwargs.get("num_heads", 4),
                queries=num_queries, dropout=cross_dropout)

    def _get_segments(self, inputs: dict, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        segments = inputs.get("text_segments")
        mask = inputs.get("text_segment_mask")
        if segments is None or mask is None:
            raise KeyError("V2 requires text_segments + text_segment_mask from semantic cache")
        if segments.ndim != 3 or mask.ndim != 2 or segments.shape[:2] != mask.shape:
            raise ValueError("text_segments [B,M,D] / text_segment_mask [B,M] are invalid")
        if int(segments.shape[0]) != batch_size or int(segments.shape[-1]) != self.text_dim:
            raise ValueError("semantic-cache batch size or text_dim mismatch")
        if segments.shape[1] < 1:
            raise ValueError("cache must have at least one segment slot")
        return segments, mask.bool()

    def forward(self, inputs: dict) -> torch.Tensor:
        graph = inputs.get("graph_input")
        if graph is None or len(graph) != 2:
            raise ValueError("graph_input must contain [graph, line_graph]")
        g, lg = graph
        g, lg = g.local_var(), lg.local_var()
        counts_g, counts_e = g.batch_num_nodes(), g.batch_num_edges()
        counts_lg, counts_le = lg.batch_num_nodes(), lg.batch_num_edges()
        batch_size = len(counts_g)
        mode = str(inputs.get("text_mode", "normal")).lower().strip()
        if mode not in {"normal", "bypass"}:
            raise ValueError("text_mode must be normal or bypass")
        use_text = mode == "normal" and inputs.get("text_segments") is not None

        if use_text:
            segments, mask = self._get_segments(inputs, batch_size)
            segments = segments.to(dtype=self.text_proj[0].weight.dtype)
        else:
            segments = None
            mask = None

        if self.stage == "a1":
            # Original blocks remain unchanged; only text aggregation differs.
            inputs_legacy = dict(inputs)
            inputs_legacy["text_mode"] = "normal" if use_text else "bypass"
            inputs_legacy["text_emb"] = masked_mean(segments, mask) if use_text else None
            inputs_legacy["text"] = None
            return super().forward(inputs_legacy)

        if use_text:
            projected = self.text_proj(segments)
            # Explicitly zero pads AFTER projection: Linear and LN have biases.
            projected = projected * mask.unsqueeze(-1)
        else:
            projected = None

        z = self.angle_embedding(lg.edata.pop("angle"))
        x = self.atom_embedding(g.ndata.pop("atomic_numbers") - 1)
        y = self.edge_embedding(g.edata.pop("distance"))
        # Reuse exactly the same coGN + line-graph backbone; switch off the
        # legacy SINGLE-text-vector gate, never graph/line-graph messaging.
        for block in self.blocks:
            x, y, z = block(
                g, x, y, lg, z, None,
                counts_g, counts_e, counts_lg, counts_le,
                enable_text_guidance=False,
            )
        graph_only = self.readout(g, x)
        if not use_text:
            return self.regression_head(graph_only)

        text_graph = self.v2_text_to_graph(projected)
        enriched = self.cross_modal(x, counts_g, text_graph, mask)
        # For samples without text, keep original graph representation.
        fused = self.readout(g, enriched)
        if self.stage == "a3":
            fused = fused + self.material_queries(enriched, counts_g, text_graph, mask) * mask.any(1, keepdim=True)
        return self.regression_head(fused)
