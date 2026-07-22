"""
Joint SOH+RUL model: same 4-branch multi-kernel CNN + LSTM backbone as
CNNLSTM (src/models/cnn_lstm.py), with two separate regression heads
sharing the final LSTM hidden state.

Adaptive loss weighting — IMPORTANT implementation note (read before
treating "alpha/beta trained as parameters" as literal free scalars):
a naive learnable alpha, beta multiplying L_SOH, L_RUL respectively, with
alpha+beta trained by gradient descent on total_loss = alpha*L_SOH +
beta*L_RUL directly, is a degenerate optimization problem — the trivial
global minimum is alpha=beta=0 (any positive loss times zero is zero),
so plain unconstrained learnable weights collapse to nothing during
training. This is why the standard approach (Kendall, Gal & Cipolla 2018,
"Multi-Task Learning Using Uncertainty to Weigh Losses") reparametrizes
each weight via a learnable log-variance term instead:

    alpha = 1 / (2*sigma_soh^2),   beta = 1 / (2*sigma_rul^2)
    total_loss = alpha*L_SOH + beta*L_RUL + log(sigma_soh) + log(sigma_rul)

The log(sigma) regularizer is what prevents the collapse-to-zero solution
(driving sigma to infinity to shrink alpha/beta also grows the log(sigma)
penalty). This is what "adaptive" means in the ablation below — alpha and
beta genuinely move during training, but through this reparametrization,
not as raw free parameters. sigma_soh, sigma_rul are `nn.Parameter`s
(actually log_sigma is the learned parameter, exponentiated for use), so
they are trained jointly with the network by the same optimizer.
"""

import torch
import torch.nn as nn

from models.cnn_lstm import MultiKernelBranch


class JointSOHRULModel(nn.Module):
    def __init__(self, in_channels: int = 6, branch_channels: int = 8,
                 lstm_hidden: int = 32, kernel_sizes=(3, 5, 7, 11)):
        super().__init__()
        self.branches = nn.ModuleList([
            MultiKernelBranch(in_channels, branch_channels, k) for k in kernel_sizes
        ])
        concat_channels = branch_channels * len(kernel_sizes)
        self.lstm = nn.LSTM(concat_channels, lstm_hidden, batch_first=True)
        self.soh_head = nn.Sequential(
            nn.Linear(lstm_hidden, 16), nn.ReLU(), nn.Linear(16, 1)
        )
        self.rul_head = nn.Sequential(
            nn.Linear(lstm_hidden, 16), nn.ReLU(), nn.Linear(16, 1)
        )

    def backbone(self, x):
        x = x.transpose(1, 2)
        branch_outs = [b(x) for b in self.branches]
        merged = torch.cat(branch_outs, dim=1).transpose(1, 2)
        _, (h_n, _) = self.lstm(merged)
        return h_n[-1]

    def forward(self, x):
        h = self.backbone(x)
        return self.soh_head(h), self.rul_head(h)


class AdaptiveLossWeighting(nn.Module):
    """
    Learnable homoscedastic-uncertainty task weighting (see module
    docstring). Clamps log_sigma to prevent the degenerate runaway
    observed empirically: an unclamped run drifted log_sigma to ~-1.41
    (alpha/beta reaching 8.4/8.2), enough for the log(sigma) regularizer
    to dominate the actual prediction-error terms and push total loss
    negative, which underperformed even the naive fixed 50/50 split.

    Bound choice: alpha = 0.5*exp(-2*log_sigma) is EXTREMELY sensitive to
    log_sigma because of that -2x in the exponent - a symmetric clamp of
    [-3, 3] (the naive first guess) would still let alpha reach
    0.5*exp(6) = 201.7, nowhere near "sensible" and wouldn't have
    prevented the observed divergence at all (which only needed
    log_sigma to reach -1.41, well inside [-3,3]). Instead clamped to
    [-0.7, 0.7], which bounds alpha/beta to roughly [0.12, 2.03] - wide
    enough to let the model meaningfully favor one task up to ~4x over
    the fixed 0.5/0.5 split in either direction, but not explode past
    where the log(sigma) term can overwhelm the actual loss signal.
    """

    LOG_SIGMA_CLAMP = 0.7

    def __init__(self):
        super().__init__()
        self.log_sigma_soh = nn.Parameter(torch.zeros(()))
        self.log_sigma_rul = nn.Parameter(torch.zeros(()))

    def forward(self, loss_soh, loss_rul):
        log_sigma_soh = torch.clamp(self.log_sigma_soh, -self.LOG_SIGMA_CLAMP, self.LOG_SIGMA_CLAMP)
        log_sigma_rul = torch.clamp(self.log_sigma_rul, -self.LOG_SIGMA_CLAMP, self.LOG_SIGMA_CLAMP)
        alpha = 0.5 * torch.exp(-2 * log_sigma_soh)
        beta = 0.5 * torch.exp(-2 * log_sigma_rul)
        total = (alpha * loss_soh + log_sigma_soh
                 + beta * loss_rul + log_sigma_rul)
        return total, float(alpha.item()), float(beta.item())
