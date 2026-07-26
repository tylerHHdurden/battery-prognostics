"""
Lightweight ICA/DV/DC feature-fusion encoder (no attention - plain
CNN + global-average-pool, per the task's "skip cross-attention, use
simpler concat-fusion" instruction).

Purpose: compress the 3 ICA/DV/DC channels (dQdV, dVdQ, dIdV - channels
3:6 of the sequence_features 6-channel tensor, already per-channel
normalized) into one small fixed-size vector per cycle, trained via a
lightweight SOH regression head so the compressed representation is
actually meaningful rather than a random projection. The head is
discarded after training; `.encode()` returns the pooled embedding for
downstream fusion into XGBoost's feature set and the stacking
meta-learner.

Architecture (deliberately small - this is a fusion side-channel, not a
5th base learner to compare against the other four):
    Conv1d(3->16, k=7) -> ReLU -> Conv1d(16->embed_dim, k=5) -> ReLU
    -> AdaptiveAvgPool1d(1) -> flatten -> [SOH head: Linear(embed_dim,1)]

Shares the exact `forward(x) -> (batch,1)` interface used by
VLSTM/CNNLSTM/PiFormer so it can be trained with the SAME
`train_deep_models.train_one_model` loop without any changes there.
"""

import torch
import torch.nn as nn


class ICAEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, embed_dim: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(16, embed_dim, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(embed_dim, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, in_channels) -> (batch, in_channels, seq_len)
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return self.pool(x).squeeze(-1)  # (batch, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))
