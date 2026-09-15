"""
BatLiNet (Zhang et al. 2025, Nature Machine Intelligence, "Battery
lifetime prediction across diverse ageing conditions with inter-cell
deep learning") - Stage 5.2.

DISCLOSED ADAPTATION, not a byte-exact paper replication (the paper's
own code was not publicly available at the time of writing - see
DEVELOPMENT_LOG.md Stage 5.2 entry for the search that confirmed this).
Reimplemented from the architecture description recovered from the
paper's arXiv preprint (2310.05052) + its Nature Machine Intelligence
abstract, with two deliberate, disclosed deviations from the source:

1. FEATURE REPRESENTATION: the paper uses a Q-indexed (capacity-domain)
   6-channel 2D image per cell (Vc(Q), Vd(Q), Ic(Q), Id(Q), deltaV(Q),
   R(Q)) fed to a 2D CNN. This project already has its own established,
   verified 6-channel TIME-indexed tensor (V_t, I_t, T_t, dQdV, dVdQ,
   dIdV - see sequence_features.py) used identically by every other
   model in this project (VLSTM/CNN-LSTM/PiFormer/the ICA encoder) and
   already built for every dataset including the 3 new Stage 5.1 ones.
   Reusing it (via a Conv1d encoder, matching ica_encoder.py's own
   pattern, instead of a Conv2d one) keeps this new model on the same,
   already-verified feature pipeline as the rest of the project rather
   than introducing a second, parallel, unverified feature-engineering
   path - a deliberate consistency choice, not an oversight.

2. TARGET VARIABLE: the paper predicts a single scalar (total cycle
   life) per cell from early-cycle data. This project's primary,
   deployed-model metric throughout Stage 0-4 is per-CYCLE SOH (to
   stay comparable to Stage 4's own R2=0.9658 headline number, the
   explicit comparison target this stage was asked to report against).
   Generalized here as: intra-cell branch predicts a cycle's own SOH
   from its own tensor; inter-cell branch predicts the SOH DIFFERENCE
   between two cycles' tensors (possibly from two different cells,
   possibly the same cell at two different points in its life) - the
   same intra/inter-cell DUALITY the paper uses, applied to this
   project's actual per-cycle regression setting instead of the
   paper's one-scalar-per-cell setting.

Architecture (Eqs. 1-9 of the source, as recovered):
    h_theta, h_phi: 2x [Conv1d -> AvgPool1d -> ReLU], hidden_dim=32,
        identical architecture for both branches (as specified).
    f_theta(x)   = w^T h_theta(x)       (intra-cell absolute SOH)
    g_phi(dx)    = w^T h_phi(dx)        (inter-cell SOH difference)
    w: SHARED final linear layer across both heads (as specified) -
        this is what actually couples the two branches into one
        trainable representation, not just two independent heads.
Loss (Eq. 9): sum||f_theta(x_i)-y_i||^2 + lambda*sum||g_phi(x_i-x_j)-(y_i-y_j)||^2
Inference: y_hat = alpha*f_theta(x) + (1-alpha)*median_k[g_phi(x-x_k)+y_k]
    over K sampled reference (cycle, SOH) pairs from the training pool.
"""

import numpy as np
import torch
import torch.nn as nn


class _Branch(nn.Module):
    """Shared architecture for both h_theta (intra) and h_phi (inter) -
    identical to the paper's spec (2 conv+pool+ReLU, hidden_dim=32)."""

    def __init__(self, in_channels: int, hidden_dim: int = 32):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=7, padding=3)
        self.pool1 = nn.AvgPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.pool2 = nn.AvgPool1d(kernel_size=2)
        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, channels) -> (batch, channels, seq_len)
        x = x.transpose(1, 2)
        x = torch.relu(self.pool1(self.conv1(x)))
        x = torch.relu(self.pool2(self.conv2(x)))
        return self.gap(x).squeeze(-1)  # (batch, hidden_dim)


class BatLiNet(nn.Module):
    def __init__(self, in_channels: int = 6, hidden_dim: int = 32):
        super().__init__()
        self.h_theta = _Branch(in_channels, hidden_dim)
        self.h_phi = _Branch(in_channels, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1)  # SHARED final layer (Eq. per paper)

    def forward_intra(self, x: torch.Tensor) -> torch.Tensor:
        return self.w(self.h_theta(x)).squeeze(-1)

    def forward_inter(self, dx: torch.Tensor) -> torch.Tensor:
        return self.w(self.h_phi(dx)).squeeze(-1)

    def predict(self, x_query: torch.Tensor, x_refs: torch.Tensor, y_refs: torch.Tensor,
                alpha: float = 0.5) -> torch.Tensor:
        """Inference per the paper's combination rule: alpha*intra +
        (1-alpha)*median-over-references(inter-cell diff + reference's
        own known SOH). x_query: (B, T, C). x_refs/y_refs: (K, T, C)/(K,)
        - K sampled reference (cycle-tensor, SOH) pairs from the
        training pool, shared across the batch."""
        y_intra = self.forward_intra(x_query)  # (B,)
        B = x_query.shape[0]
        K = x_refs.shape[0]
        dx = x_query.unsqueeze(1) - x_refs.unsqueeze(0)  # (B, K, T, C)
        dx_flat = dx.reshape(B * K, *dx.shape[2:])
        diff_pred = self.forward_inter(dx_flat).reshape(B, K)
        y_cross = diff_pred + y_refs.unsqueeze(0)  # (B, K)
        y_inter = y_cross.median(dim=1).values  # (B,)
        return alpha * y_intra + (1 - alpha) * y_inter
