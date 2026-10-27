"""Mock the unavailable GPU-only DGL dependency to test the *real* V2 forward.

Integration without changing parent files. Real DGL operations and training
still require validation on the user's actual server.
"""
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn


class FakeGraph:
    def __init__(self, counts, atomic_numbers=None, distance=None, angle=None):
        self._counts = torch.tensor(counts)
        self.ndata = {"atomic_numbers": torch.tensor(atomic_numbers or [1] * sum(counts))}
        self.edata = {}
        if distance is not None:
            self.edata["distance"] = torch.as_tensor(distance).float()
        if angle is not None:
            self.edata["angle"] = torch.as_tensor(angle).float()

    def local_var(self):
        copied = FakeGraph(self._counts.tolist())
        copied.ndata = dict(self.ndata)
        copied.edata = dict(self.edata)
        return copied

    def batch_num_nodes(self):
        return self._counts

    def batch_num_edges(self):
        return self._counts


class FakeBlock(nn.Module):
    def forward(self, g, x, y, lg, z, text, *args, enable_text_guidance):
        return x, y, z


class FakeReadout(nn.Module):
    def forward(self, graph, nodes):
        return torch.stack([t.mean(0) for t in nodes.split(graph.batch_num_nodes().tolist())])


class FakeParent(nn.Module):
    def __init__(self, *, text_dim=8, text_proj_dim=4, embed_dim=16,
                 text_proj_hidden_dims=(6,), **kwargs):
        super().__init__()
        self.text_dim = text_dim
        self.text_proj_dim = text_proj_dim
        self.embed_dim = embed_dim
        self.text_proj_hidden_dims = text_proj_hidden_dims
        self.text_proj_dropout = 0.1
        self.atom_embedding = nn.Embedding(95, embed_dim)
        self.edge_embedding = nn.Linear(1, embed_dim)
        self.angle_embedding = nn.Linear(1, embed_dim)
        self.text_proj = nn.Sequential(nn.Linear(text_dim, text_proj_dim))
        self.blocks = nn.ModuleList([FakeBlock()])
        self.readout = FakeReadout()
        self.regression_head = nn.Linear(embed_dim, 1)

    def forward(self, inputs):
        graph = inputs["graph_input"][0]
        nodes = self.atom_embedding(graph.ndata["atomic_numbers"] - 1)
        return self.regression_head(self.readout(graph, nodes))


def load_with_fake_parent():
    stub = types.ModuleType("models.TextGuidedFusiformerCached")
    stub.TextGuidedFusiformer = FakeParent
    real = Path(__file__).resolve().parents[1] / "fusiformer_v2" / "model.py"
    spec = importlib.util.spec_from_file_location("fusiformer_v2.model_test_fake", real)
    mod = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"models.TextGuidedFusiformerCached": stub}):
        spec.loader.exec_module(mod)
    return mod.FusiformerV2


def test_a2_and_a3_forward_backward_and_bypass():
    model_class = load_with_fake_parent()
    for stage in ("a1", "a2", "a3"):
        model = model_class(stage=stage, text_dim=8, text_proj_dim=4,
                            embed_dim=16, num_heads=4)
        g = FakeGraph([2, 3], atomic_numbers=[1, 2, 3, 4, 5], distance=[[1]]*5)
        lg = FakeGraph([5, 5], angle=[[1.]]*10)
        x = torch.randn(2, 3, 8)
        mask = torch.tensor([[True, True, False], [True, False, False]])
        out = model({"graph_input": [g, lg], "text_segments": x,
                     "text_segment_mask": mask})
        assert out.shape == (2, 1)
        out.sum().backward()
        if stage != "a1":
            assert model.cross_modal.attn.in_proj_weight.grad is not None
            assert model.text_proj[0].weight.grad is not None
        bypass = model({"graph_input": [g, lg], "text_mode": "bypass"})
        assert bypass.shape == (2, 1)
