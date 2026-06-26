import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from typing import Tuple, Union, Optional

import dgl
import dgl.function as fn
from dgl.nn import AvgPooling
import torch
from torch import nn
from dgl.nn.functional import edge_softmax
from functools import partial
from transformers import AutoModel, AutoConfig  # 改用 AutoModel

from .layers import RBFExpansion


class MLP(nn.Module):
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


class TextCrossAttention(nn.Module):
    """
    Text-guided cross attention: 使用文本特征作为K,V来引导节点/边特征的选择
    """
    def __init__(self, dim, text_dim, num_heads, use_bias=True, drop=0.):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.text_dim = text_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = (dim // num_heads) ** -0.5

        # Q来自节点/边特征
        self.q_linear = nn.Linear(dim, dim, bias=use_bias)
        # K,V来自文本特征
        self.kv_linear = nn.Linear(text_dim, dim * 2, bias=use_bias)
        
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, x, text_emb, batch_num_items=None):
        """
        x: node or edge features [N, dim]
        text_emb: text features [B, text_dim]
        batch_num_items: 每个batch中item的数量（节点数或边数）
        """
        B = text_emb.shape[0]
        
        if batch_num_items is not None:
            # 使用给定的每个batch的item数量
            text_emb_expanded = torch.repeat_interleave(text_emb, batch_num_items, dim=0)
        else:
            # 假设均匀分布
            N = x.shape[0] // B if B > 0 else x.shape[0]
            text_emb_expanded = text_emb.repeat_interleave(N, dim=0)
        
        # Q from x
        q = self.q_linear(x).reshape(-1, self.num_heads, self.head_dim)
        
        # K, V from text
        kv = self.kv_linear(text_emb_expanded).reshape(-1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(1)
        
        # Attention
        attn = torch.einsum('bhd,bhd->bh', q, k) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Weighted sum
        out = torch.einsum('bh,bhd->bhd', attn, v).reshape(-1, self.dim)
        out = self.out_proj(out)
        
        return out


class GraphConditionalAttention(nn.Module):
    def __init__(self, dim, num_heads, use_bias, drop):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv_linear = nn.Linear(dim, dim*3, bias=use_bias)
        self.c_linear = nn.Linear(dim, dim, bias=use_bias)
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


class GraphCrossAttention(nn.Module):
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


class TextGuidedFusiformerLayer(nn.Module):
    def __init__(self, dim, text_dim, num_heads, use_bias=False, mlp_ratio=2., drop=0.,
                 act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super().__init__()

        # 1. Text-CrossAttention: 文本引导节点特征
        self.text_cross_attn_node = TextCrossAttention(dim, text_dim, num_heads, use_bias, drop)
        self.norm_text_node = norm_layer(dim)
        
        # 2. Text-CrossAttention: 文本引导边特征
        self.text_cross_attn_edge = TextCrossAttention(dim, text_dim, num_heads, use_bias, drop)
        self.norm_text_edge = norm_layer(dim)
        
        self.graph_cross_attn = GraphCrossAttention(dim, num_heads, use_bias, drop)
        self.norm_graph = norm_layer(dim)
        
        self.graph_cond_attention = GraphConditionalAttention(dim, num_heads, use_bias, drop)
        self.norm_cond_h = norm_layer(dim)
        self.norm_cond_e = norm_layer(dim)
        
        self.mlp_h = MLP(dim, int(dim * mlp_ratio), dim, 2, act_layer)
        self.mlp_e = MLP(dim, int(dim * mlp_ratio), dim, 2, act_layer)
        self.norm_mlp_h = norm_layer(dim)
        self.norm_mlp_e = norm_layer(dim)

    def forward(self, g, h, e, text_emb, batch_num_nodes=None, batch_num_edges=None):
        """
        g: graph
        h: node features
        e: edge features
        text_emb: text features [B, text_dim]
        batch_num_nodes: 每个batch的节点数量
        batch_num_edges: 每个batch的边数量
        """
        g = g.local_var()
        
        # 1. Text-CrossAttention for nodes
        text_guided_h = self.text_cross_attn_node(h, text_emb, batch_num_nodes)
        h = self.norm_text_node(h + text_guided_h)
        
        # 2. Text-CrossAttention for edges
        text_guided_e = self.text_cross_attn_edge(e, text_emb, batch_num_edges)
        e = self.norm_text_edge(e + text_guided_e)
        
        # 3. Graph-CrossAttention: 节点到边的信息聚合
        h_graph = self.graph_cross_attn(g, h, e)
        h = self.norm_graph(h + h_graph)
        
        # 4. ConditionalAttention: 边特征更新
        h_, e_, score = self.graph_cond_attention(g, h, e)
        h = self.norm_cond_h(h + h_)
        e = self.norm_cond_e(e + e_)
        
        # 5. MLP updates
        h = self.norm_mlp_h(h + self.mlp_h(h))
        e = self.norm_mlp_e(e + self.mlp_e(e))
        
        return h, e, score


class TextGuidedFusiformerBlock(nn.Module):
    def __init__(self, dim, text_dim, num_heads, use_bias=False, mlp_ratio=2., drop=0.,
                 act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.g_update = TextGuidedFusiformerLayer(
            dim, text_dim, num_heads, use_bias, mlp_ratio,
            drop, act_layer, norm_layer
        )
        
        self.lg_update = TextGuidedFusiformerLayer(
            dim, text_dim, num_heads, use_bias, mlp_ratio,
            drop, act_layer, norm_layer
        )

    def forward(self, g, x, y, lg, z, text_emb, 
                batch_num_nodes_g, batch_num_edges_g,
                batch_num_nodes_lg, batch_num_edges_lg):
        """
        x: node features
        y: edge features (also lg node features)
        z: angle features (lg edge features)
        text_emb: text features
        """
        # 先更新线图
        y, z, score_lg = self.lg_update(lg, y, z, text_emb, 
                                        batch_num_nodes_lg, batch_num_edges_lg)
        
        # 再更新主图
        x, y, score_g = self.g_update(g, x, y, text_emb,
                                      batch_num_nodes_g, batch_num_edges_g)
        
        return x, y, z


class TextGuidedFusiformer(nn.Module):
    def __init__(self, targets=[],
                 depth=4, edge_input_dim=80, triplet_input_dim=40,
                 embed_dim=128, num_heads=4, mlp_ratio=2.,
                 use_bias=True, norm_layer=None, act_layer=None,
                 text_dim=1536, qwen_model_name="/root/models/Qwen2.5-1.5B-Instruct",
                 freeze_qwen=True):
        """
        text_dim: Qwen 的隐藏层维度 
        - Qwen2.5-7B: 3584
        - Qwen2.5-14B: 5120
        - Qwen2.5-3B: 2048
        - Qwen2.5-1.5B: 1536
        """
        super().__init__()

        self.targets = targets
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.ReLU

        # 原子嵌入
        self.atom_embedding = nn.Embedding(95, embed_dim)

        # 边嵌入
        self.edge_embedding = nn.Sequential(
            RBFExpansion(vmin=0, vmax=8.0, bins=edge_input_dim),
            nn.Linear(edge_input_dim, embed_dim),
            norm_layer(embed_dim),
            act_layer(),
        )
        
        # 角嵌入
        self.angle_embedding = nn.Sequential(
            RBFExpansion(vmin=-1, vmax=1.0, bins=triplet_input_dim),
            nn.Linear(triplet_input_dim, embed_dim),
            norm_layer(embed_dim),
            act_layer(),
        )

        # 文本投影层
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, embed_dim),
            norm_layer(embed_dim),
            act_layer(),
        )
        
        # 设置环境变量使用国内镜像（可选）
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        
        # 加载预训练的 Qwen 模型
        self.qwen = AutoModel.from_pretrained(
            qwen_model_name,
            trust_remote_code=True,
            output_hidden_states=True,
            local_files_only=True
        )

        self.freeze_qwen = freeze_qwen
        if freeze_qwen:
            for param in self.qwen.parameters():
                param.requires_grad = False
            self.qwen.eval()
        else:
            self.qwen.train()
        
        # 文本引导的 Fusiformer blocks
        self.blocks = nn.ModuleList([
            TextGuidedFusiformerBlock(
                dim=embed_dim,
                text_dim=embed_dim,
                num_heads=num_heads,
                use_bias=use_bias,
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                norm_layer=norm_layer
            )
            for _ in range(depth)
        ])

        # 全局池化
        self.readout = AvgPooling()
        
        # 回归头
        self.regression_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, len(targets) if targets else 1)
        )

    def encode_text(self, text_tokens):
        """
        使用 Qwen 编码文本
        text_tokens: 来自 tokenizer 的输入，包含 'input_ids' 和 'attention_mask'
        """
        ctx = torch.no_grad() if self.freeze_qwen else torch.enable_grad()

        with ctx:
            outputs = self.qwen(
                input_ids=text_tokens['input_ids'],
                attention_mask=text_tokens.get('attention_mask', None),
                output_hidden_states=True
            )

            attention_mask = text_tokens.get('attention_mask', None)
            if attention_mask is not None:
                hidden_states = outputs.last_hidden_state
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                text_emb = (hidden_states * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-6)
            else:
                text_emb = outputs.last_hidden_state.mean(dim=1)

        text_emb = self.text_proj(text_emb)
        return text_emb

    def forward(self, inputs):
        """
        inputs: 包含 'graph_input' 和 'text' 的字典
        """
        # 获取图输入
        graph = inputs.get('graph_input')
        assert len(graph) == 2
        g, lg = graph
        g = g.local_var()
        lg = lg.local_var()
        
        # 获取batch信息
        batch_num_nodes_g = g.batch_num_nodes()
        batch_num_edges_g = g.batch_num_edges()
        batch_num_nodes_lg = lg.batch_num_nodes()
        batch_num_edges_lg = lg.batch_num_edges()
        
        # 编码文本
        text_tokens = inputs.get('text')
        if text_tokens is not None:
            text_emb = self.encode_text(text_tokens)  # [B, embed_dim]
        else:
            # 如果没有文本，使用零向量
            device = g.device
            batch_size = len(batch_num_nodes_g)
            text_emb = torch.zeros(batch_size, self.text_proj[0].out_features).to(device)
        
        # 构建初始特征
        z = self.angle_embedding(lg.edata.pop("angle"))
        x = self.atom_embedding(g.ndata.pop("atomic_numbers") - 1)
        y = self.edge_embedding(g.edata.pop('distance'))
        
        # 通过文本引导的 blocks
        for block in self.blocks:
            x, y, z = block(g, x, y, lg, z, text_emb,
                          batch_num_nodes_g, batch_num_edges_g,
                          batch_num_nodes_lg, batch_num_edges_lg)
        
        # 全局池化
        graph_out = self.readout(g, x)  # [B, embed_dim]
        
        # 回归预测
        pred = self.regression_head(graph_out)
        
        return pred

    def train(self, mode: bool = True):
        super().train(mode)

        if getattr(self, "freeze_qwen", False) and self.qwen is not None:
            self.qwen.eval()

        return self