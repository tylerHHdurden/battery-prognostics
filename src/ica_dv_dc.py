"""
Incremental Capacity Analysis (ICA) / Differential Voltage (DV) / dI/dV
differential tensors, computed per discharge cycle.

ASSUMPTION: capacity Q(t) is computed uniformly across all three datasets
by trapezoidal integration of |I| dt over the discharge phase (Ah), rather
than trusting each dataset's own reported cumulative-capacity field. This
is deliberate — NASA/CALCE/MIT report capacity differently (NASA: one
scalar per cycle; CALCE: cumulative-within-file; MIT: a full per-sample Qd
array) and integrating from raw I ourselves gives one consistent method
everywhere, at the cost of not reusing MIT's own (probably more accurate,
coulomb-counted) Qd trace. Flagged here rather than silently mixed.

For each cycle we output, on a common uniform voltage grid:
    V_grid : (n_bins,)
    dQdV, dVdQ, dIdV : (n_bins,) each, Savitzky-Golay smoothed

Stacking dQdV/dVdQ/dIdV as 3 channels over a uniform voltage axis, across
many cycles, gives the "differential tensor" (n_cycles, n_bins, 3) referred
to in the task — used later as one of the input modalities for the
CNN-LSTM / PiFormer models.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def _dedup_sort_by_v(V: np.ndarray, *others: np.ndarray):
    order = np.argsort(V)
    V_sorted = V[order]
    others_sorted = [o[order] for o in others]
    # average duplicate V's (common with flat plateaus in real cyclers)
    uniq_V, inv = np.unique(V_sorted, return_inverse=True)
    out = []
    for arr in others_sorted:
        summed = np.bincount(inv, weights=arr)
        counts = np.bincount(inv)
        out.append(summed / counts)
    return uniq_V, out


def compute_ica_dv_dc(cycle: dict, n_bins: int = 200,
                       savgol_window: int = 15, savgol_poly: int = 3) -> dict | None:
    dc = cycle["discharge"]
    t, V, I = dc["t"], dc["V"], dc["I"]
    if len(t) < max(savgol_window, 10):
        return None

    # cumulative discharge capacity from raw current (Ah); |I| since I<0
    # during discharge in our sign convention.
    Q = np.concatenate([[0.0], np.cumsum(
        np.abs(I[:-1]) * np.diff(t)
    ) / 3600.0])

    V_u, (Q_u, I_u, t_u) = _dedup_sort_by_v(V, Q, I, t)
    if len(V_u) < max(savgol_window, 10):
        return None

    V_grid = np.linspace(V_u.min(), V_u.max(), n_bins)
    Q_grid = np.interp(V_grid, V_u, Q_u)
    I_grid = np.interp(V_grid, V_u, I_u)

    dQdV = np.gradient(Q_grid, V_grid)
    # Stage 5.1 fix: XJTU's non-constant-current protocols (random-pulse
    # "R2.5"/"R3" discharge, unlike NASA/MIT/CALCE's simple CC discharge)
    # can make Q non-monotonic in V-sorted order, so Q_grid (from
    # interpolating Q against the monotonic V_u/V_grid axis) can itself
    # contain a locally-flat run of duplicate values - np.gradient uses
    # Q_grid as its coordinate array here, so an exact-zero local spacing
    # divides by zero (-> inf/nan), which savgol_filter then rejects.
    # Root-caused on Batch-4/R3_battery-8 cycle 636 (2 of 200 grid points
    # were exact duplicates) before applying this fix - not a blind
    # try/except. Nudging duplicate-run values by a tiny strictly-
    # increasing epsilon (not reordering/dropping anything) keeps
    # np.gradient's output numerically finite everywhere while leaving
    # every real, non-duplicate value bit-identical.
    Q_grid_safe = Q_grid.copy()
    dup = np.diff(Q_grid_safe) == 0
    if dup.any():
        eps = max(np.finfo(np.float64).eps * 10, 1e-12)
        Q_grid_safe = Q_grid_safe + np.concatenate([[0.0], np.cumsum(dup) * eps])
    dVdQ = np.gradient(V_grid, Q_grid_safe)
    dIdV = np.gradient(I_grid, V_grid)

    win = savgol_window if savgol_window < n_bins else (n_bins // 2) * 2 - 1
    win = max(win, savgol_poly + 2 + (savgol_poly % 2 == 0))
    if win % 2 == 0:
        win += 1

    dQdV_s = savgol_filter(dQdV, win, savgol_poly)
    dVdQ_s = savgol_filter(dVdQ, win, savgol_poly)
    dIdV_s = savgol_filter(dIdV, win, savgol_poly)

    return {
        "V_grid": V_grid,
        "I_grid": I_grid,
        "dQdV": dQdV_s,
        "dVdQ": dVdQ_s,
        "dIdV": dIdV_s,
    }


def build_differential_tensor(cycle_records: list[dict], n_bins: int = 200) -> np.ndarray:
    """(n_valid_cycles, n_bins, 3) stacked [dQdV, dVdQ, dIdV] tensor."""
    rows = []
    for c in cycle_records:
        r = compute_ica_dv_dc(c, n_bins=n_bins)
        if r is not None:
            rows.append(np.stack([r["dQdV"], r["dVdQ"], r["dIdV"]], axis=-1))
    if not rows:
        return np.zeros((0, n_bins, 3))
    return np.stack(rows, axis=0)
