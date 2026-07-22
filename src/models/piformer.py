"""
PiFormer-style Transformer: cross-channel attention where the Query comes
from the voltage/ICA channels only, and Key/Value come from all combined
channels, followed by a LocalConvFFN instead of the usual dense
feed-forward block.

ASSUMPTION (the user specified "PiFormer-style", not full architecture
details beyond the cross-attention split and LocalConvFFN, so the rest of
the block skeleton follows a standard pre-norm Transformer encoder):
  - Query channels = [V_t, dQdV, dVdQ, dIdV] (sequence_features.
    QUERY_CHANNEL_IDX) -- "voltage/ICA" per the task description.
  - Key/Value channels = all 6 channels ("combined channels").
  - LocalConvFFN = depthwise Conv1d (captures local temporal/voltage-bin
    patterns) -> pointwise Conv1d (channel mixing) -> GELU, replacing the
    usual Linear->GELU->Linear FFN. This is the common "conv-FFN" pattern
    used in several time-series Transformer variants (e.g. Informer's
    distilling conv, Autoformer-style local convs) - a reasonable
    concretization of "LocalConvFFN" given no exact spec was provided.
  - 2 encoder layers, 4 attention heads, d_model=32 (kept small
    deliberately: CPU-only training budget, see OVERNIGHT_LOG.md).
"""

import torch
import torch.nn as nn

from sequence_features import CHANNEL_NAMES, QUERY_CHANNEL_IDX


class LocalConvFFN(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 5):
        super().__init__()
        self.depthwise = nn.Conv1d(d_model, d_model, kernel_size,
                                    padding=kernel_size // 2, groups=d_model)
        self.pointwise = nn.Conv1d(d_model, d_model, 1)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (batch, seq_len, d_model) -> (batch, d_model, seq_len)
        x_t = x.transpose(1, 2)
        out = self.pointwise(self.act(self.depthwise(x_t)))
        return out.transpose(1, 2)


class CrossChannelAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, query, kv):
        out, _ = self.mha(query, kv, kv)
        return out


class PiFormerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = CrossChannelAttention(d_model, n_heads)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = LocalConvFFN(d_model)

    def forward(self, q_embed, kv_embed):
        attn_out = self.attn(self.norm_q(q_embed), self.norm_kv(kv_embed))
        q_embed = q_embed + attn_out
        q_embed = q_embed + self.ffn(self.norm_ffn(q_embed))
        return q_embed


class PiFormer(nn.Module):
    def __init__(self, seq_len: int = 200, n_channels: int = 6, d_model: int = 32,
                 n_heads: int = 4, n_layers: int = 2, n_targets: int = 1):
        super().__init__()
        self.query_idx = QUERY_CHANNEL_IDX
        n_query_channels = len(self.query_idx)

        self.q_proj = nn.Linear(n_query_channels, d_model)
        self.kv_proj = nn.Linear(n_channels, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        self.blocks = nn.ModuleList([
            PiFormerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 16), nn.ReLU(),
            nn.Linear(16, n_targets),
        )

    def forward(self, x):
        # x: (batch, seq_len, n_channels)
        q_in = x[:, :, self.query_idx]
        q_embed = self.q_proj(q_in) + self.pos_embed
        kv_embed = self.kv_proj(x) + self.pos_embed

        for block in self.blocks:
            q_embed = block(q_embed, kv_embed)

        pooled = q_embed.mean(dim=1)
        return self.head(pooled)
