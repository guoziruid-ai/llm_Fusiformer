"""Atomistic LIne Graph Neural Network.

A prototype crystal line graph network dgl implementation.
"""
from typing import Tuple, Union

import dgl
import dgl.function as fn
from dgl.nn import AvgPooling
import torch
from torch import nn
from dgl.nn.functional import edge_softmax
from functools import partial
from transformers import LlamaForCausalLM, LlamaConfig
from transformers import BertModel, BertTokenizer

from .layers import RBFExpansion


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act_layer=nn.ReLU):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = act_layer(inplace=True)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class ConditionalAttention(nn.Module):
    def __init__(self, dim, num_heads, use_bias, drop):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv_linear = nn.Linear(dim, dim*3, bias=use_bias)
        self.c_linear = nn.Linear(dim, dim, bias=use_bias)
        # self.gated = nn.ReLU(inplace=True)
        self.gated = nn.Tanh()

        self.h_proj = nn.Linear(dim, dim)
        self.e_proj = nn.Linear(dim, dim)

    def forward(self, g, h, e):
        g = g.local_var()
        qkv = self.qkv_linear(h).reshape(-1, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(1)
        c = self.c_linear(e).reshape(-1, self.num_heads, self.head_dim)
        c = self.gated(c)

        g.ndata['q'] = q
        g.ndata['k'] = k
        g.ndata['v'] = v
        g.apply_edges(fn.u_mul_v('k', 'q', 'score'))

        score = g.edata.pop('score') * c
        attn = score.sum(-1, keepdims=True) * self.scale

        g.edata['attn'] = edge_softmax(g, attn)
        g.update_all(fn.u_mul_e('v', 'attn', 'v'), fn.sum('v', 'h'))

        h_out = self.h_proj(g.ndata.pop('h').reshape(-1, self.dim))
        e_out = self.e_proj(score.reshape(-1, self.dim))

        return h_out, e_out, score.mean(dim=1)


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, use_bias, drop):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.q_linear = nn.Linear(dim, dim, bias=use_bias)
        self.kv_linear = nn.Linear(dim, dim*2, bias=use_bias)
        self.h_proj = nn.Linear(dim, dim)

    def forward(self, g, h, e):
        g = g.local_var()
        q = self.q_linear(h).reshape(-1, self.num_heads, self.head_dim)
        kv = self.kv_linear(e).reshape(-1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(1)

        g.ndata['q'] = q
        g.edata['k'] = k
        g.apply_edges(fn.v_dot_e('q', 'k', 'attn'))

        attn = g.edata.pop('attn') * self.scale
        attn = edge_softmax(g, attn)
        g.edata['v'] = attn * v
        g.update_all(fn.copy_e('v', 'm'), fn.sum('m', 'h'))

        h_out = self.h_proj(g.ndata.pop('h').reshape(-1, self.dim))
        return h_out


class CrysformerLayer(nn.Module):
    def __init__(self, dim, num_heads, use_bias=False, mlp_ratio=2., drop=0.,
                 act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.crossattention = CrossAttention(dim, num_heads, use_bias, drop)
        self.norm1 = norm_layer(dim)

        self.condattention = ConditionalAttention(dim, num_heads, use_bias, drop)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)

        self.mlp1 = MLP(dim, int(dim * mlp_ratio), dim, 2, act_layer)
        self.mlp2 = MLP(dim, int(dim * mlp_ratio), dim, 2, act_layer)
        self.norm4 = norm_layer(dim)
        self.norm5 = norm_layer(dim)

    def forward(self, g, h, e):
        g = g.local_var()
        h = self.norm1(h + self.crossattention(g, h, e))

        h_, e_, score = self.condattention(g, h, e)
        h = self.norm2(h + h_)
        e = self.norm3(e + e_)

        h = self.norm4(h + self.mlp1(h))
        e = self.norm5(e + self.mlp2(e))

        return h, e, score


class DynamicConnection(nn.Module):
    def __init__(self, threshold=1., t=1., act_layer=nn.ReLU):
        super().__init__()
        self.th = threshold
        self.T = t

    def forward(self, g, score, y, lg, z):
        def edge_norm_lt_th(edges):
            return (edges.data.pop('score').norm(dim=-1) / self.T) < self.th
        g.edata['y'] = y
        g.edata['score'] = score
        drop_idx = g.filter_edges(edge_norm_lt_th)
        g.remove_edges(drop_idx)
        y = g.edata.pop('y')
        lg.edata['z'] = z
        lg.remove_nodes(drop_idx)
        z = lg.edata.pop('z')
        return y, z


class CrysformerBlock(nn.Module):
    """Line graph update."""

    def __init__(self, dim, num_heads, use_bias=False, mlp_ratio=2., drop=0.,
                 act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.g_update = CrysformerLayer(dim, num_heads, use_bias, mlp_ratio,
                                        drop, act_layer, norm_layer)
        # self.g_dynamic = DynamicConnection(threshold=0.3, t=1.)

        self.lg_update = CrysformerLayer(dim, num_heads, use_bias, mlp_ratio,
                                        drop, act_layer, norm_layer)
        # self.lg_dynamic = DynamicConnection(threshold=0.3, t=1.)

    def forward(self, g, x, y, lg, z):
        """Node and Edge updates for ALIGNN layer.

        x: node input features
        y: edge input features
        z: edge pair input features
        """
        y, z, score = self.lg_update(lg, y, z)
        x, y, score = self.g_update(g, x, y)

        return x, y, z


class Crysformer(nn.Module):
    def __init__(self, targets=[],
                 depth=4, edge_input_dim=80, triplet_input_dim=40,
                 embed_dim=128, num_heads=4, mlp_ratio=2.,
                 use_bias=True, norm_layer=None, act_layer=None,
                 text_dim=768, fusion_dim=512, llama_layers=4, llama_heads=8):

        super().__init__()

        self.targets = targets
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

        self.blocks = nn.ModuleList([
                CrysformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    use_bias=use_bias,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer=norm_layer
                )
                for _ in range(depth)
            ])

        self.readout = AvgPooling()

        self.graph_proj = nn.Linear(embed_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
        self.llama_config = LlamaConfig(
            vocab_size=32000,
            hidden_size=fusion_dim,
            intermediate_size=fusion_dim*4,
            num_hidden_layers=llama_layers,
            num_attention_heads=llama_heads,
            max_position_embeddings=512
        )
        self.llama = LlamaForCausalLM(self.llama_config)
        
        self.regression_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_dim // 2, len(targets) if targets else 1)
        )

    def forward(
        self, inputs
    ):
        """
        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata)
        z: angle features (lg.edata)
        """    
        graph = inputs.get('graph_input')
        assert len(graph) == 2
        g, lg = graph
        g = g.local_var()
        lg = lg.local_var()
        z = self.angle_embedding(lg.edata.pop("angle"))

        x = self.atom_embedding(g.ndata.pop("atomic_numbers") - 1)
        y = self.edge_embedding(g.edata.pop('distance'))

        for block in self.blocks:
            x, y, z = block(g, x, y, lg, z)

        graph_out = self.readout(g, x)
        text_out = inputs.get('text_emb')

        g_proj = self.graph_proj(graph_out)
        t_proj = self.text_proj(text_out)

        fused = torch.cat([g_proj, t_proj], dim=-1)
        fused = self.fusion_mlp(fused)
        
        input_embeds = fused.unsqueeze(1)
        llama_out = self.llama(
            inputs_embeds=input_embeds,
            output_hidden_states=True
        )

        hidden_state = llama_out.hidden_states[-1][:,0,:]
        pred = self.regression_head(hidden_state)
        
        return pred
        
