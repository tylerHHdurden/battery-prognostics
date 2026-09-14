"""
Stage 3, Item 3.1: Kernel Mean Matching (Huang, Smola, Gretton, Borgwardt
& Scholkopf 2007, "Correcting Sample Selection Bias by Unlabeled Data")
for conformal calibration reweighting, plus a Selective-KMM extension
for limited calibration/target support overlap.

IMPLEMENTATION NOTE, stated explicitly (no QP solver available in this
environment - checked: cvxpy, qpsolvers, quadprog all absent, only
scipy): the standard KMM formulation is
    min_beta  0.5 * beta^T K beta - kappa^T beta
    s.t.      0 <= beta_i <= B,  |sum(beta) - n| <= n*eps
This project solves it via L-BFGS-B (bound-constrained, smooth,
well-suited to this problem size) with the sum-constraint converted to
a SOFT quadratic penalty term rather than a hard linear constraint -
a standard, well-behaved practical simplification for KMM specifically
(the sum constraint only needs to be approximately satisfied; the
literature's own eps-tolerance already treats it as approximate, not
exact). Ridge-regularizes the kernel matrix (K + 1e-6*I) for numerical
stability, since an RBF Gram matrix can be near-singular for large n.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def rbf_kernel(X, Y, gamma):
    d2 = cdist(X, Y, metric="sqeuclidean")
    return np.exp(-gamma * d2)


def median_heuristic_gamma(X, sample_size=500, seed=42):
    """Standard median-heuristic bandwidth: gamma = 1 / (2 * median_pairwise_sqdist)."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), size=min(sample_size, len(X)), replace=False)
    d2 = cdist(X[idx], X[idx], metric="sqeuclidean")
    med = np.median(d2[d2 > 0])
    return 1.0 / (2 * med) if med > 0 else 1.0


def fit_kmm_weights(X_calib, X_target, B=4.0, eps_frac=0.2, gamma=None,
                     per_point_bound=None, penalty=1.0, max_iter=500):
    """
    Standard KMM (per_point_bound=None) or Selective KMM (per_point_bound
    given - a per-calibration-point upper bound array, typically derived
    from each point's similarity to the TARGET set, so points with poor
    target support get a near-zero allowed weight instead of being
    forced to participate in the global moment match).

    Returns (weights, diagnostics dict).
    """
    n, m = len(X_calib), len(X_target)
    if gamma is None:
        gamma = median_heuristic_gamma(np.vstack([X_calib, X_target]))

    K = rbf_kernel(X_calib, X_calib, gamma) + 1e-6 * np.eye(n)
    kappa = (float(n) / m) * rbf_kernel(X_calib, X_target, gamma).sum(axis=1)

    upper = np.full(n, B) if per_point_bound is None else np.minimum(B, per_point_bound)
    bounds = [(0.0, float(u)) for u in upper]

    def obj_and_grad(beta):
        Kb = K @ beta
        obj = 0.5 * beta @ Kb - kappa @ beta
        sum_dev = beta.sum() - n
        obj += 0.5 * penalty * sum_dev ** 2
        grad = Kb - kappa + penalty * sum_dev
        return obj, grad

    beta0 = np.clip(np.ones(n), 0, upper)
    res = minimize(obj_and_grad, beta0, jac=True, bounds=bounds, method="L-BFGS-B",
                    options={"maxiter": max_iter})
    weights = np.clip(res.x, 0, upper)

    ess = float((weights.sum() ** 2) / (weights ** 2).sum()) if (weights ** 2).sum() > 0 else 0.0
    diag = {
        "gamma": gamma, "converged": res.success, "n_iter": res.nit,
        "sum_weights": float(weights.sum()), "n_calib": n,
        "effective_sample_size": ess, "ess_frac": ess / n,
        "weight_min": float(weights.min()), "weight_max": float(weights.max()),
        "frac_at_upper_bound": float((weights >= upper - 1e-6).mean()),
        "frac_near_zero": float((weights < 1e-4).mean()),
    }
    return weights, diag


def selective_bounds(X_calib, X_target, gamma, B=4.0, percentile=50):
    """Per-calibration-point weight bound for Selective KMM: each
    calibration point's max kernel similarity to ANY target point,
    rescaled to [0, B] - points with poor support in the target set
    (low max-similarity to every target point) get a small allowed
    weight regardless of what the global moment-match would otherwise
    assign them, directly addressing "calibration doesn't cover the
    full target support" rather than letting a few far-flung points
    take on extreme weights to compensate."""
    sim = rbf_kernel(X_calib, X_target, gamma)  # (n_calib, n_target)
    max_sim = sim.max(axis=1)  # each calib point's best match to the target set
    max_sim_norm = max_sim / (max_sim.max() + 1e-12)
    ref = np.percentile(max_sim_norm, percentile)
    scale = np.clip(max_sim_norm / (ref + 1e-12), 0, 1)
    return B * scale


def weighted_conformal_interval(calib_resid, w_calib, pred_target, alpha=0.1):
    """Tibshirani et al. 2019 weighted split-conformal quantile (same
    mechanism already used in this project's session-19 weighted-
    conformal script, reused here unchanged) - but with KMM's BOUNDED
    weights instead of a domain-classifier's unbounded odds ratio.
    w_target is implicitly 1 for every target point here (KMM already
    reweights calibration TOWARD the target distribution as a whole,
    rather than needing a per-test-point weight the way the classifier-
    odds approach did) - each target point gets the SAME weighted
    quantile from the (already target-aligned) calibration set."""
    order = np.argsort(calib_resid)
    sorted_resid = calib_resid[order]
    sorted_w = w_calib[order]
    cum_w = np.cumsum(sorted_w)
    W = w_calib.sum()
    threshold = (1 - alpha) * (W + 1.0)  # w_target=1 for every target point
    idx = np.searchsorted(cum_w, threshold, side="left")
    if idx < len(sorted_resid):
        q = sorted_resid[idx]
    else:
        q = np.inf
    lo = pred_target - q
    hi = pred_target + q
    return lo, hi, q
