"""Mask-aware, native-DGL compatible fine-grained attention layers.

All masks use True=valid.  The graph is stored in flat DGL node order; a
batch-local dense view is created only for cross-attention/query pooling.
"""
from __future__ import annotations

import torch
from torch import nn


def pack_nodes(nodes: torch.Tensor, counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[sum(N_i), D] -> [B, max(N_i), D] and [B, max(N_i)] valid mask."""
    if nodes.ndim != 2 or counts.ndim != 1:
        raise ValueError("nodes must be [sum_nodes,D], counts must be [B]")
    counts = counts.to(device=nodes.device, dtype=torch.long)
    if (counts < 0).any() or int(counts.sum()) != nodes.size(0):
        raise ValueError("batch counts do not match flattened nodes")
    b = counts.numel()
    max_n = max(int(counts.max().item()) if b else 0, 1)
    offsets = torch.cumsum(counts, 0) - counts
    batch_index = torch.repeat_interleave(torch.arange(b, device=nodes.device), counts)
    positions = torch.arange(nodes.size(0), device=nodes.device) - offsets[batch_index]
    padded = nodes.new_zeros(b, max_n, nodes.size(1))
    valid = torch.arange(max_n, device=nodes.device).unsqueeze(0) < counts.unsqueeze(1)
    padded[batch_index, positions] = nodes
    return padded, valid


def unpack_nodes(padded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Back to concatenated DGL atom order; gradients flow through indexing."""
    return padded[mask]


def masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3 or mask.shape != tokens.shape[:2]:
        raise ValueError("expected tokens [B,M,D] and valid mask [B,M]")
    weights = mask.unsqueeze(-1).to(tokens.dtype)
    return (tokens * weights).sum(1) / weights.sum(1).clamp(min=1.0)


class AtomTextCrossAttention(nn.Module):
    """Atom queries attend to all valid semantic segments of the same crystal.

    This is a REAL [B, heads, N_atoms, M_segments] attention, as opposed to
    the branch's original singleton-text/head-wise gating operation.
    """

    def __init__(self, dim: int = 128, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * dim, dim))
        # Conservative residual init helps preserve pretrained/scratch graph representation.
        self.gain = nn.Parameter(torch.tensor(0.1))

    def forward(self, atoms: torch.Tensor, counts: torch.Tensor,
                segments: torch.Tensor, segment_mask: torch.Tensor) -> torch.Tensor:
        packed, atom_mask = pack_nodes(atoms, counts)
        if segments.ndim != 3 or segments.shape[:2] != segment_mask.shape:
            raise ValueError("segments must be [B,M,D], mask [B,M]")
        if segments.size(0) != packed.size(0) or segments.size(2) != packed.size(2):
            raise ValueError("batch/embedding dimension mismatch")
        has_text = segment_mask.any(dim=1)
        safe_mask = segment_mask.clone()
        safe_mask[~has_text, 0] = True  # avoid softmax over all -inf
        q = self.norm_q(packed)
        update, _ = self.attn(q, segments, segments,
                              key_padding_mask=~safe_mask, need_weights=False)
        update = update * has_text[:, None, None]
        enriched = self.norm1(packed + self.gain * update)
        enriched = self.norm2(enriched + self.gain * self.ffn(enriched))
        # No-text samples must be exactly graph-only (no LN/FFN side effects).
        enriched = torch.where(has_text[:, None, None], enriched, packed)
        return unpack_nodes(enriched, atom_mask)


class MaterialQueryPooler(nn.Module):
    """Learnable latent queries summarize enriched atom + text segment tokens."""

    def __init__(self, dim: int = 128, heads: int = 4, queries: int = 16, dropout: float = 0.1):
        super().__init__()
        if queries < 1:
            raise ValueError("queries must be >=1")
        self.queries = nn.Parameter(torch.empty(1, queries, dim))
        nn.init.normal_(self.queries, std=0.02)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.gain = nn.Parameter(torch.tensor(0.1))

    def forward(self, atoms: torch.Tensor, counts: torch.Tensor,
                segments: torch.Tensor, segment_mask: torch.Tensor) -> torch.Tensor:
        padded, atom_mask = pack_nodes(atoms, counts)
        tokens = torch.cat([padded, segments], dim=1)
        valid = torch.cat([atom_mask, segment_mask], dim=1)
        q = self.queries.expand(tokens.size(0), -1, -1)
        latent, _ = self.attn(q, tokens, tokens, key_padding_mask=~valid, need_weights=False)
        latent = self.norm(q + latent)
        latent = self.norm(latent + self.ffn(latent))
        return latent.mean(dim=1) * self.gain
