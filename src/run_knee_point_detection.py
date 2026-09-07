"""
Knee-point detection on predicted vs. ground-truth SOH curves — a
DERIVED output from the already-computed fusion-ensemble predictions
(ensemble_fusion_test_preds.csv), no new model training, matching the
style of the other post-hoc evaluation scripts (run_early_prediction_test.py,
run_drop_branch_ablation.py, run_homogeneous_bagging.py).

Curvature formula, citation check performed before writing anything to
DEVELOPMENT_LOG.md: the task cited this as matching "the BatteryGPT
reference paper already in the literature survey." This repo has no
literature-survey document (checked: no file matching *literature*/
*survey* anywhere in the tree), so "already in the literature survey"
does not hold for THIS project - logged honestly rather than silently
accepted or silently dropped. The paper itself IS real, though:
"Early prediction of lithium-ion battery degradation with a generative
pre-trained transformer" (Nature Communications, s41467-025-66819-0,
Dec 2025) - confirmed via web search and by fetching its text - and it
does define the knee point exactly as specified here: the point of
maximum curvature kappa = |y''| / (1+y'^2)^1.5 on the SOH-vs-cycle
curve, marking the transition from a linear/gradual ageing rate to
nonlinear/accelerated degradation. The paper's methods text does not
specify how y'/y'' are computed from the discrete, noisy per-cycle SOH
sequence (raw finite differencing twice would mostly amplify measurement
noise into meaningless spikes) - not disclosed in the fetched text, so
this script makes its own documented choice below rather than guessing
at an undisclosed detail.

Derivative computation choice (logged, not hidden): y and its 1st/2nd
derivatives are estimated via a Savitzky-Golay polynomial fit
(window=15, polyorder=3) rather than raw finite differences - the EXACT
SAME window/polyorder this project already uses for smoothing the
dQdV/dVdQ/dIdV curves in ica_dv_dc.py, reused here for consistency
rather than picked freshly. scipy's savgol_filter(deriv=1)/(deriv=2)
gives the derivative of the LOCAL POLYNOMIAL FIT at each point, which is
far better-conditioned on a noisy discrete sequence than differentiating
raw np.diff() twice (that would take a second difference of an already-
noisy first difference, amplifying noise quadratically).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
PRED_DIR = PROC_DIR / "predictions"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

SAVGOL_WINDOW = 15  # matches ica_dv_dc.py's convention exactly
SAVGOL_POLY = 3
# exclude a half-window margin at each end: savgol_filter's boundary
# handling (mode="interp", the default) extrapolates a lower-order fit
# there, which is markedly less reliable for a SECOND derivative than in
# the interior - excluding it avoids the knee search latching onto an
# edge artifact rather than the genuine degradation transition.
EDGE_MARGIN = SAVGOL_WINDOW // 2


def curvature(y: np.ndarray, window: int, poly: int) -> np.ndarray:
    """kappa = |y''| / (1 + y'^2)^1.5, y'/y'' from a Savitzky-Golay local
    polynomial fit (delta=1, since cycle_idx spacing is uniformly 1 for
    every test battery here - verified before writing this script)."""
    win = window if window <= len(y) else (len(y) // 2) * 2 - 1
    win = max(win, poly + 2 + (poly % 2 == 0))  # same odd/min-size guard as ica_dv_dc.py
    y1 = savgol_filter(y, win, poly, deriv=1, delta=1.0)
    y2 = savgol_filter(y, win, poly, deriv=2, delta=1.0)
    return np.abs(y2) / (1 + y1 ** 2) ** 1.5


def knee_cycle(cycle_idx: np.ndarray, y: np.ndarray, window: int, poly: int, margin: int) -> tuple[int, np.ndarray]:
    kappa = curvature(y, window, poly)
    lo, hi = margin, len(kappa) - margin
    if hi <= lo:  # battery too short for the margin - fall back to searching the full range
        lo, hi = 0, len(kappa)
    local_argmax = np.argmax(kappa[lo:hi])
    return int(cycle_idx[lo + local_argmax]), kappa


def main():
    df = pd.read_csv(PRED_DIR / "ensemble_fusion_test_preds.csv")
    print(f"[knee] {df['battery_id'].nunique()} test batteries, "
          f"{len(df)} total cycles, Savitzky-Golay window={SAVGOL_WINDOW} poly={SAVGOL_POLY} "
          f"(same convention as ica_dv_dc.py), edge margin={EDGE_MARGIN} cycles excluded each end")

    rows = []
    for bid, g in df.groupby("battery_id"):
        g = g.sort_values("cycle_idx").reset_index(drop=True)
        cycle_idx = g["cycle_idx"].to_numpy()
        soh_true = g["SOH"].to_numpy()
        soh_pred = g["pred_Stacking_Ridge_fusion"].to_numpy()

        true_knee, kappa_true = knee_cycle(cycle_idx, soh_true, SAVGOL_WINDOW, SAVGOL_POLY, EDGE_MARGIN)
        pred_knee, kappa_pred = knee_cycle(cycle_idx, soh_pred, SAVGOL_WINDOW, SAVGOL_POLY, EDGE_MARGIN)
        offset = pred_knee - true_knee

        rows.append({
            "battery_id": bid, "n_cycles": len(g),
            "true_knee_cycle": true_knee, "pred_knee_cycle": pred_knee,
            "offset_cycles": offset, "abs_offset_cycles": abs(offset),
            "offset_pct_of_lifetime": abs(offset) / len(g) * 100,
        })
        print(f"[knee] {bid} ({len(g)} cycles): true knee=cycle {true_knee}, "
              f"predicted knee=cycle {pred_knee}, offset={offset:+d} cycles "
              f"({abs(offset) / len(g) * 100:.1f}% of lifetime)")

    result_df = pd.DataFrame(rows)
    mean_abs_offset = result_df["abs_offset_cycles"].mean()
    mean_pct_offset = result_df["offset_pct_of_lifetime"].mean()
    print(f"\n[knee] === SUMMARY across {len(result_df)} test batteries ===")
    print(result_df.to_string(index=False))
    print(f"\n[knee] mean absolute cycle-offset error: {mean_abs_offset:.1f} cycles")
    print(f"[knee] mean offset as % of battery lifetime: {mean_pct_offset:.1f}%")

    result_df.to_csv(OUT_DIR / "knee_point_detection.csv", index=False)
    print(f"[knee] saved per-battery results to outputs/knee_point_detection.csv")
    print("[knee] DONE")


if __name__ == "__main__":
    main()
