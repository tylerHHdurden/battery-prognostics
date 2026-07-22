"""
VLSTM: a peephole-connection LSTM with a coupled forget/input gate,
operating on the univariate discharge-voltage sequence (channel 0 of the
sequence_features tensor — see that module's docstring for why "VLSTM" is
read literally as a voltage-sequence LSTM here).

Peephole connections (Gers & Schmidhuber 2000): gates see the cell state
directly, not just h_{t-1}.
Coupled forget/input gate (Greff et al. 2017 LSTM variants study): i_t =
1 - f_t, removing the separate input-gate weight matrices.

    f_t = sigmoid(W_f x_t + U_f h_{t-1} + V_f * c_{t-1} + b_f)
    i_t = 1 - f_t                                   <- coupled
    g_t = tanh(W_g x_t + U_g h_{t-1} + b_g)
    c_t = f_t * c_{t-1} + i_t * g_t
    o_t = sigmoid(W_o x_t + U_o h_{t-1} + V_o * c_t + b_o)   <- peephole uses c_t
    h_t = o_t * tanh(c_t)

Implemented as an explicit per-timestep recurrence (can't use nn.LSTM,
which has neither peepholes nor gate coupling).
"""

import torch
import torch.nn as nn


class PeepholeCoupledLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_f = nn.Linear(input_size, hidden_size)
        self.U_f = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V_f = nn.Parameter(torch.zeros(hidden_size))  # peephole (diag)

        self.W_g = nn.Linear(input_size, hidden_size)
        self.U_g = nn.Linear(hidden_size, hidden_size, bias=False)

        self.W_o = nn.Linear(input_size, hidden_size)
        self.U_o = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V_o = nn.Parameter(torch.zeros(hidden_size))  # peephole (diag)

    def forward(self, x_t, h_prev, c_prev):
        f_t = torch.sigmoid(self.W_f(x_t) + self.U_f(h_prev) + self.V_f * c_prev)
        i_t = 1.0 - f_t  # coupled forget/input gate
        g_t = torch.tanh(self.W_g(x_t) + self.U_g(h_prev))
        c_t = f_t * c_prev + i_t * g_t
        o_t = torch.sigmoid(self.W_o(x_t) + self.U_o(h_prev) + self.V_o * c_t)
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class VLSTM(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 32, n_targets: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = PeepholeCoupledLSTMCell(input_size, hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 16), nn.ReLU(), nn.Linear(16, n_targets)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        b, seq_len, _ = x.shape
        h = torch.zeros(b, self.hidden_size, device=x.device)
        c = torch.zeros(b, self.hidden_size, device=x.device)
        for t in range(seq_len):
            h, c = self.cell(x[:, t, :], h, c)
        return self.head(h)
