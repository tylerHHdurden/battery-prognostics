"""
Maximum Mean Discrepancy (MMD) — multi-bandwidth Gaussian-RBF kernel,
the standard unsupervised domain-adaptation regularizer (Gretton et al.
2012; used e.g. by Long et al.'s DAN for deep domain adaptation).

Used by train_fusion_encoder_mmd.py to pull the ICAEncoder's fusion
embedding distribution for CALCE (target, UNLABELED — only its ICA/DV/DC
feature tensors are read, never its SOH/RUL) toward the embedding
distribution for NASA+MIT (source, labeled) during training. This is
standard unsupervised domain adaptation, not label leakage: MMD only
ever sees target *inputs*, never target *targets*.

Biased MMD^2 estimator (Gretton et al. eq. 5):
    MMD^2(X, Y) = mean(K_xx) + mean(K_yy) - 2*mean(K_xy)
for a characteristic kernel K (here, a sum of Gaussian RBFs at several
bandwidths, so the loss isn't sensitive to one arbitrarily-chosen sigma).
"""

import torch


def _pairwise_sq_dists(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b, batched over (n_a, n_b)
    a2 = (A * A).sum(dim=1, keepdim=True)
    b2 = (B * B).sum(dim=1, keepdim=True).t()
    return (a2 + b2 - 2.0 * A @ B.t()).clamp(min=0.0)


def multi_kernel_mmd2(source: torch.Tensor, target: torch.Tensor,
                       sigma_scales=(0.5, 1.0, 2.0, 4.0, 8.0)) -> torch.Tensor:
    """
    source: (n_s, d), target: (n_t, d) — batches of embeddings, same dim.
    Bandwidth is set per-batch via the median-heuristic (median pairwise
    squared distance across the pooled source+target batch), then scaled
    by `sigma_scales` and summed — the standard multi-kernel MMD used in
    DAN, so the result isn't hostage to one hand-picked bandwidth.
    """
    pooled = torch.cat([source, target], dim=0)
    with torch.no_grad():
        d2 = _pairwise_sq_dists(pooled, pooled)
        n = d2.shape[0]
        off_diag = d2[~torch.eye(n, dtype=torch.bool, device=d2.device)]
        median_d2 = off_diag.median().clamp(min=1e-8)

    def kernel_sum(A, B):
        d2_ab = _pairwise_sq_dists(A, B)
        total = torch.zeros_like(d2_ab)
        for scale in sigma_scales:
            bandwidth = median_d2 * scale
            total = total + torch.exp(-d2_ab / (2.0 * bandwidth))
        return total

    k_ss = kernel_sum(source, source).mean()
    k_tt = kernel_sum(target, target).mean()
    k_st = kernel_sum(source, target).mean()
    return k_ss + k_tt - 2.0 * k_st
