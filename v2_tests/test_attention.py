"""Torch-only unit tests. Real DGL/JARVIS integration runs on the GPU server."""
import pytest
import torch
from fusiformer_v2.attention import (
    AtomTextCrossAttention, MaterialQueryPooler, pack_nodes, unpack_nodes, masked_mean,
)


def test_pack_roundtrip_and_grad():
    x = torch.randn(7, 12, requires_grad=True)
    padded, valid = pack_nodes(x, torch.tensor([2, 4, 1]))
    assert padded.shape == (3, 4, 12)
    assert valid.sum().item() == 7
    assert torch.allclose(unpack_nodes(padded, valid), x)
    unpack_nodes(padded, valid).square().sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_pack_invalid_counts():
    with pytest.raises(ValueError):
        pack_nodes(torch.zeros(3, 8), torch.tensor([1, 1]))


def test_cross_attention_shapes_and_grad():
    torch.manual_seed(1)
    module = AtomTextCrossAttention(16, 4, dropout=0).train()
    atoms = torch.randn(7, 16, requires_grad=True)
    text = torch.randn(3, 4, 16, requires_grad=True)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0]], dtype=torch.bool)
    out = module(atoms, torch.tensor([2, 4, 1]), text, mask)
    assert out.shape == atoms.shape
    out.square().sum().backward()
    assert atoms.grad.abs().sum() > 0
    assert text.grad.abs().sum() > 0
    assert module.attn.in_proj_weight.grad.abs().sum() > 0


def test_text_padding_invariance():
    torch.manual_seed(2)
    module = AtomTextCrossAttention(16, 4, dropout=0).eval()
    atoms = torch.randn(5, 16)
    texts = torch.randn(2, 3, 16)
    valid = torch.tensor([[True, False, False], [True, True, False]])
    edited = texts.clone()
    edited[~valid] = torch.randn_like(edited[~valid]) * 100
    a = module(atoms, torch.tensor([2, 3]), texts, valid)
    b = module(atoms, torch.tensor([2, 3]), edited, valid)
    assert torch.allclose(a, b, atol=1e-6)


def test_no_text_is_identity():
    module = AtomTextCrossAttention(16, 4, dropout=0).eval()
    atoms = torch.randn(3, 16)
    texts = torch.zeros(2, 2, 16)
    mask = torch.tensor([[False, False], [True, False]])
    out = module(atoms, torch.tensor([2, 1]), texts, mask)
    assert torch.equal(out[:2], atoms[:2])


def test_material_queries_shapes_grad_and_mask():
    torch.manual_seed(3)
    module = MaterialQueryPooler(16, 4, queries=8, dropout=0)
    atoms = torch.randn(5, 16, requires_grad=True)
    texts = torch.randn(2, 3, 16, requires_grad=True)
    mask = torch.tensor([[True, False, False], [True, True, False]])
    output = module(atoms, torch.tensor([2, 3]), texts, mask)
    assert output.shape == (2, 16)
    output.sum().backward()
    assert module.queries.grad.abs().sum() > 0


def test_masked_mean():
    t = torch.tensor([[[1., 3.], [9., 9.]], [[4., 5.], [6., 7.]]])
    m = torch.tensor([[True, False], [False, False]])
    assert masked_mean(t, m).tolist() == [[1., 3.], [0., 0.]]
