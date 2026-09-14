"""
Stage 3, Item 3.4: Deep CORAL (Sun & Saenko, "Deep CORAL: Correlation
Alignment for Deep Domain Adaptation", 2016) - second-order covariance
alignment, as a direct replacement for session 13's MMD alignment term.

CORAL aligns the SECOND-ORDER statistics (feature covariance) of the
source and target embedding distributions, rather than MMD's kernel
mean-embedding distance (which compares smoothed versions of the full
distributions via an RBF kernel). It's simpler (no kernel/bandwidth
choice), and, per the motivating MDPI Batteries 2026 12(9),340 paper,
was reported to pair well with a monotonicity-constrained GBM head under
a leave-one-cell-out domain-shift protocol structurally similar to this
project's own CALCE zero-retrain evaluation.

L_CORAL = 1 / (4 * d^2) * || C_source - C_target ||_F^2

where C_source, C_target are the d x d feature covariance matrices of a
minibatch of source/target embeddings (d = embedding dim). Uses the
standard unbiased (n-1) covariance estimator; falls back gracefully for
batch_size=1 (returns 0 - no covariance is estimable from one sample).
"""

import torch


def coral_loss(source, target):
    d = source.shape[1]
    ns, nt = source.shape[0], target.shape[0]
    if ns < 2 or nt < 2:
        return torch.tensor(0.0, device=source.device, dtype=source.dtype)

    source_c = source - source.mean(dim=0, keepdim=True)
    target_c = target - target.mean(dim=0, keepdim=True)
    cov_s = (source_c.T @ source_c) / (ns - 1)
    cov_t = (target_c.T @ target_c) / (nt - 1)

    diff = cov_s - cov_t
    loss = (diff * diff).sum() / (4.0 * d * d)
    return loss
