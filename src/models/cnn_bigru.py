"""
5th base learner (additive - does not touch VLSTM/CNN-LSTM/PiFormer):
CNN-BiGRU hybrid. Same 4-branch multi-kernel-scale 1D-CNN front end as
CNNLSTM (src/models/cnn_lstm.py: kernel sizes {3, 5, 7, 11}, reused
directly via `from models.cnn_lstm import MultiKernelBranch`, not
duplicated) for local pattern extraction, feeding into a Bidirectional
GRU (2 gates - update, reset - vs. LSTM's 3: input, forget, output) for
temporal modeling, then a small FC regression head.

Deliberately reuses CNNLSTM's EXACT front end rather than designing a
fresh one: the point of this comparison is to isolate the ONE
architectural change actually being tested - LSTM vs. bidirectional GRU
as the recurrent core - not to confound it with a different CNN
front-end too. Any RMSE/R2 difference from CNNLSTM below is therefore
attributable to the recurrent core (+ bidirectionality), not to a
different feature-extraction stage.

Input: the same full 6-channel sequence_features tensor (V_t, I_t, T_t,
dQdV, dVdQ, dIdV) that CNNLSTM and PiFormer consume, shape (batch, 200,
6) -> permuted to (batch, 6, 200) for Conv1d. (VLSTM alone, among the
existing 3 deep models, restricts to channel 0 (V_t) only - CNN-BiGRU
matches CNNLSTM/PiFormer's full-tensor convention instead, the majority
convention among the existing base learners.)

Final representation: the GRU's last-timestep hidden state from BOTH
directions concatenated (forward direction's state at t=199, backward
direction's state after processing t=199->0, i.e. "seen" all of t=0) -
the standard way to summarize a bidirectional RNN's output for a
sequence-level regression head.
"""

import torch
import torch.nn as nn

from models.cnn_lstm import MultiKernelBranch


class CNNBiGRU(nn.Module):
    def __init__(self, in_channels: int = 6, branch_channels: int = 8,
                 gru_hidden: int = 32, n_targets: int = 1,
                 kernel_sizes=(3, 5, 7, 11)):
        super().__init__()
        self.branches = nn.ModuleList([
            MultiKernelBranch(in_channels, branch_channels, k) for k in kernel_sizes
        ])
        concat_channels = branch_channels * len(kernel_sizes)
        self.gru = nn.GRU(concat_channels, gru_hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(gru_hidden * 2, 16), nn.ReLU(), nn.Linear(16, n_targets)
        )

    def forward(self, x):
        # x: (batch, seq_len, in_channels) -> (batch, in_channels, seq_len)
        x = x.transpose(1, 2)
        branch_outs = [b(x) for b in self.branches]
        merged = torch.cat(branch_outs, dim=1)  # (batch, concat_channels, seq_len)
        merged = merged.transpose(1, 2)  # (batch, seq_len, concat_channels)
        _, h_n = self.gru(merged)  # h_n: (2, batch, gru_hidden) - [forward, backward]
        h_cat = torch.cat([h_n[0], h_n[1]], dim=-1)  # (batch, 2*gru_hidden)
        return self.head(h_cat)
