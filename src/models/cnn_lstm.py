"""
CNN-LSTM with a 4-branch multi-kernel-scale front end: four parallel
Conv1d branches with kernel sizes {3, 5, 7, 11} (small/medium/large/very
large receptive field over the 200-step sequence), each followed by
ReLU + BatchNorm, concatenated along the channel dimension, then fed into
a standard LSTM, then a small FC regression head on the final hidden
state.

Input: full 6-channel sequence_features tensor (V_t, I_t, T_t, dQdV,
dVdQ, dIdV), shape (batch, 200, 6) -> permuted to (batch, 6, 200) for
Conv1d.
"""

import torch
import torch.nn as nn


class MultiKernelBranch(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return torch.relu(self.bn(self.conv(x)))


class CNNLSTM(nn.Module):
    def __init__(self, in_channels: int = 6, branch_channels: int = 8,
                 lstm_hidden: int = 32, n_targets: int = 1,
                 kernel_sizes=(3, 5, 7, 11)):
        super().__init__()
        self.branches = nn.ModuleList([
            MultiKernelBranch(in_channels, branch_channels, k) for k in kernel_sizes
        ])
        concat_channels = branch_channels * len(kernel_sizes)
        self.lstm = nn.LSTM(concat_channels, lstm_hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 16), nn.ReLU(), nn.Linear(16, n_targets)
        )

    def forward(self, x):
        # x: (batch, seq_len, in_channels) -> (batch, in_channels, seq_len)
        x = x.transpose(1, 2)
        branch_outs = [b(x) for b in self.branches]
        merged = torch.cat(branch_outs, dim=1)  # (batch, concat_channels, seq_len)
        merged = merged.transpose(1, 2)  # (batch, seq_len, concat_channels)
        _, (h_n, _) = self.lstm(merged)
        return self.head(h_n[-1])
