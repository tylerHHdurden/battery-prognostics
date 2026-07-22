"""
Binary Firefly Algorithm (BFA) for wrapper-based feature selection over the
16 Health Indicators.

Standard S-shaped binary firefly formulation (Yang 2009 firefly algorithm +
the common sigmoid binarization used across binary-metaheuristic feature
selection papers, e.g. Emary et al.'s binary GWO convention for the
fitness function):

  - Each firefly has a continuous position x in R^d (d = n_features).
  - Binary mask b_i = 1 if sigmoid(x_i) > rand() else 0.
  - "Light intensity" = fitness (higher is better) of the binary mask.
  - Dimmer fireflies move toward brighter ones:
        x_i += beta0 * exp(-gamma * r_ij^2) * (x_j - x_i) + alpha * (rand-0.5)
    where r_ij is the Euclidean distance between continuous positions i, j.
  - Fitness = w_acc * (1 - normalized_RMSE) + w_feat * (1 - n_selected/n_total)
    i.e. reward both low prediction error AND fewer selected features — the
    0.99/0.01 accuracy/feature-ratio weighting is the standard convention
    from the wrapper-FS literature (ASSUMPTION: exact weights not specified
    by the user, so the field-standard default is used and logged here).

ASSUMPTION on population/generation size: the user asked for "real
population/generation settings, not reduced" — no exact numbers were
given, so this uses commonly-cited firefly-algorithm defaults from the
literature: n_agents=30, n_iterations=100, alpha=0.25, beta0=1.0, gamma=1.0.
The *inner* fitness evaluation (what gets called 3000 times: 30 agents x
100 generations) is kept deliberately cheap — a Ridge regression under
grouped 3-fold CV — specifically so the *outer* algorithm can run at full
literature-standard scale on a CPU-only laptop within a reasonable time
budget. This is a choice about which part of the pipeline to keep cheap,
not a reduction of the BFA's own iteration/population budget.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _wrapper_rmse(X, y, groups, mask, n_splits=3):
    if mask.sum() == 0:
        return np.inf
    Xs = X[:, mask.astype(bool)]
    n_groups = len(np.unique(groups))
    splits = min(n_splits, n_groups)
    if splits < 2:
        model = Ridge(alpha=1.0).fit(Xs, y)
        return float(np.sqrt(mean_squared_error(y, model.predict(Xs))))
    gkf = GroupKFold(n_splits=splits)
    errs = []
    for tr, te in gkf.split(Xs, y, groups):
        model = Ridge(alpha=1.0).fit(Xs[tr], y[tr])
        pred = model.predict(Xs[te])
        errs.append(mean_squared_error(y[te], pred))
    return float(np.sqrt(np.mean(errs)))


def run_bfa(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
            feature_names: list[str],
            n_agents: int = 30, n_iterations: int = 100,
            alpha: float = 0.25, beta0: float = 1.0, gamma: float = 1.0,
            w_acc: float = 0.99, w_feat: float = 0.01,
            seed: int = 42, log_fn=print):
    rng = np.random.default_rng(seed)
    n_features = X.shape[1]
    X = StandardScaler().fit_transform(X)

    baseline_rmse = _wrapper_rmse(X, y, groups, np.ones(n_features, dtype=bool))
    log_fn(f"[BFA] baseline RMSE (all {n_features} features): {baseline_rmse:.4f}")

    positions = rng.uniform(-4, 4, size=(n_agents, n_features))

    def fitness_of(pos_row):
        mask = (_sigmoid(pos_row) > rng.random(n_features)).astype(int)
        rmse = _wrapper_rmse(X, y, groups, mask)
        norm_rmse = min(rmse / baseline_rmse, 2.0) if baseline_rmse > 0 else 1.0
        acc_term = 1.0 - min(norm_rmse, 1.0)
        feat_term = 1.0 - mask.sum() / n_features
        return w_acc * acc_term + w_feat * feat_term, mask, rmse

    fitness = np.zeros(n_agents)
    masks = np.zeros((n_agents, n_features), dtype=int)
    rmses = np.zeros(n_agents)
    for i in range(n_agents):
        fitness[i], masks[i], rmses[i] = fitness_of(positions[i])

    best_idx = np.argmax(fitness)
    best_fitness = fitness[best_idx]
    best_mask = masks[best_idx].copy()
    best_rmse = rmses[best_idx]

    history = []
    for it in range(n_iterations):
        for i in range(n_agents):
            for j in range(n_agents):
                if fitness[j] > fitness[i]:
                    r2 = np.sum((positions[i] - positions[j]) ** 2)
                    beta = beta0 * np.exp(-gamma * r2)
                    positions[i] += (
                        beta * (positions[j] - positions[i])
                        + alpha * (rng.random(n_features) - 0.5)
                    )
            fitness[i], masks[i], rmses[i] = fitness_of(positions[i])

        gen_best = np.argmax(fitness)
        if fitness[gen_best] > best_fitness:
            best_fitness = fitness[gen_best]
            best_mask = masks[gen_best].copy()
            best_rmse = rmses[gen_best]

        history.append({
            "iteration": it, "best_fitness": float(best_fitness),
            "best_rmse": float(best_rmse), "n_selected": int(best_mask.sum()),
        })
        if it % 10 == 0 or it == n_iterations - 1:
            log_fn(f"[BFA] iter {it:3d}: best_fitness={best_fitness:.4f} "
                   f"best_rmse={best_rmse:.4f} n_selected={best_mask.sum()}/{n_features}")

    selected_names = [f for f, m in zip(feature_names, best_mask) if m]
    log_fn(f"[BFA] DONE. Selected {len(selected_names)}/{n_features} features: {selected_names}")
    return best_mask, selected_names, history
