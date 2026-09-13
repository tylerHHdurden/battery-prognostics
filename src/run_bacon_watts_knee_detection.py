"""
Stage 1, Item 1.7: Bacon-Watts knee detection, replacing session 16's
max-curvature approach. Same data, same evaluation protocol, no new
model training (a DERIVED analysis of ensemble_fusion_test_preds.csv,
exactly like run_knee_point_detection.py).

Model (Bacon-Watts, Fermín-Cueto et al. 2020's formulation - two
linear segments joined by a smooth tanh transition rather than a hard
break, so it's differentiable and fittable by nonlinear least squares):

    y(x) = a0 + a1*(x - x1) + a2*(x - x1)*tanh((x - x1) / gamma)

x1 is the knee-point estimate; gamma controls transition sharpness and
is fit as a free parameter (not fixed), since a fixed gamma risks being
badly mismatched to any one battery's actual transition sharpness.

SCOPE DECISION (stated per the task's own allowance): only the SINGLE
Bacon-Watts model is implemented. Double Bacon-Watts (separately
identifying knee-ONSET and knee-POINT, per Fermín-Cueto et al.'s
extended formulation) was not reached within this stage's time budget -
logged honestly rather than half-implemented. Single Bacon-Watts is
sufficient to test this item's actual hypothesis (whether a segmented-
regression approach is structurally immune to b3c0's curvature-spike
failure mode), which does not require the onset/knee distinction.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
PRED_DIR = PROC_DIR / "predictions"
OUT_DIR = ROOT / "outputs"


def bacon_watts(x, a0, a1, a2, x1, gamma):
    return a0 + a1 * (x - x1) + a2 * (x - x1) * np.tanh((x - x1) / gamma)


def fit_bacon_watts(cycle_idx: np.ndarray, y: np.ndarray):
    x = cycle_idx.astype(float)
    x_range = x.max() - x.min()
    x1_init = x.min() + 0.5 * x_range
    # initial slopes from splitting the range in half - a coarse but
    # reasonable starting point for the nonlinear solver
    mid = len(x) // 2
    a1_init = np.polyfit(x[:mid + 1], y[:mid + 1], 1)[0] if mid >= 1 else 0.0
    a2_init = (np.polyfit(x[mid:], y[mid:], 1)[0] if len(x) - mid >= 2 else 0.0) - a1_init
    a0_init = float(np.interp(x1_init, x, y))
    gamma_init = max(x_range * 0.05, 1.0)

    p0 = [a0_init, a1_init, a2_init, x1_init, gamma_init]
    bounds = (
        [-np.inf, -np.inf, -np.inf, x.min(), 1e-3],
        [np.inf, np.inf, np.inf, x.max() + x_range, x_range * 2 + 1e-3],
        # x1's UPPER bound is deliberately allowed to exceed x.max() - this
        # is exactly the known "knee estimated after end-of-life" failure
        # mode the task asks to check for, not assume away by clamping it.
    )
    try:
        popt, _ = curve_fit(bacon_watts, x, y, p0=p0, bounds=bounds, maxfev=20000)
        return popt, True
    except Exception as e:
        print(f"[bacon-watts] fit FAILED ({type(e).__name__}: {e}) - falling back to p0")
        return np.array(p0), False


def main():
    df = pd.read_csv(PRED_DIR / "ensemble_fusion_test_preds.csv")
    print(f"[bacon-watts] {df['battery_id'].nunique()} test batteries, {len(df)} total cycles")

    # session 16's original results, for direct side-by-side comparison
    orig = pd.read_csv(OUT_DIR / "knee_point_detection.csv").set_index("battery_id")

    rows = []
    for bid, g in df.groupby("battery_id"):
        g = g.sort_values("cycle_idx").reset_index(drop=True)
        cycle_idx = g["cycle_idx"].to_numpy()
        soh_true = g["SOH"].to_numpy()
        soh_pred = g["pred_Stacking_Ridge_fusion"].to_numpy()
        x_max = cycle_idx.max()

        popt_true, ok_true = fit_bacon_watts(cycle_idx, soh_true)
        popt_pred, ok_pred = fit_bacon_watts(cycle_idx, soh_pred)
        true_knee = float(popt_true[3])
        pred_knee = float(popt_pred[3])
        offset = pred_knee - true_knee

        true_past_eol = true_knee > x_max
        pred_past_eol = pred_knee > x_max
        if true_past_eol or pred_past_eol:
            print(f"[bacon-watts] CAVEAT FIRED for {bid}: knee estimated AFTER end-of-life "
                  f"(n_cycles={x_max}) - true_knee={true_knee:.1f} (past_eol={true_past_eol}), "
                  f"pred_knee={pred_knee:.1f} (past_eol={pred_past_eol}). Reporting as-is, not clamped.")

        rows.append({
            "battery_id": bid, "n_cycles": len(g),
            "true_knee_cycle_bw": true_knee, "pred_knee_cycle_bw": pred_knee,
            "offset_cycles_bw": offset, "abs_offset_cycles_bw": abs(offset),
            "offset_pct_of_lifetime_bw": abs(offset) / len(g) * 100,
            "true_fit_converged": ok_true, "pred_fit_converged": ok_pred,
            "true_knee_past_eol": true_past_eol, "pred_knee_past_eol": pred_past_eol,
            "orig_offset_cycles_maxcurv": int(orig.loc[bid, "offset_cycles"]) if bid in orig.index else None,
        })
        print(f"[bacon-watts] {bid} ({len(g)} cycles): true knee={true_knee:.1f}, "
              f"pred knee={pred_knee:.1f}, offset={offset:+.1f} cycles "
              f"({abs(offset)/len(g)*100:.1f}% of lifetime) | max-curvature offset was "
              f"{orig.loc[bid, 'offset_cycles'] if bid in orig.index else 'N/A'}")

    result_df = pd.DataFrame(rows)
    mean_abs_offset_all = result_df["abs_offset_cycles_bw"].mean()
    # exclude b3c0 specifically, for direct comparability with session 16's own
    # "excluding known outliers" 12.0-cycle headline number (which excluded ONLY b3c0)
    excl_b3c0 = result_df[result_df.battery_id != "b3c0"]
    mean_abs_offset_excl_b3c0 = excl_b3c0["abs_offset_cycles_bw"].mean()

    print(f"\n[bacon-watts] === SUMMARY across {len(result_df)} test batteries ===")
    print(result_df.to_string(index=False))
    print(f"\n[bacon-watts] mean absolute cycle-offset error (ALL batteries): {mean_abs_offset_all:.1f} cycles "
          f"(session 16 max-curvature: 161.8 cycles)")
    print(f"[bacon-watts] mean absolute cycle-offset error (excl. b3c0, same exclusion session 16 used "
          f"for its 12.0-cycle headline): {mean_abs_offset_excl_b3c0:.1f} cycles "
          f"(session 16 max-curvature, same exclusion: 12.0 cycles)")

    b3c0_row = result_df[result_df.battery_id == "b3c0"].iloc[0]
    b2c24_row = result_df[result_df.battery_id == "b2c24"].iloc[0]
    print(f"\n[bacon-watts] b3c0 (max-curvature's 911-cycle ground-truth-artifact outlier): "
          f"Bacon-Watts offset={b3c0_row['offset_cycles_bw']:+.1f} vs. max-curvature's "
          f"{b3c0_row['orig_offset_cycles_maxcurv']:+d}")
    print(f"[bacon-watts] b2c24 (max-curvature's 55-cycle prediction-discontinuity outlier): "
          f"Bacon-Watts offset={b2c24_row['offset_cycles_bw']:+.1f} vs. max-curvature's "
          f"{b2c24_row['orig_offset_cycles_maxcurv']:+d}")

    any_past_eol = result_df["true_knee_past_eol"].any() or result_df["pred_knee_past_eol"].any()
    print(f"\n[bacon-watts] Known-caveat check (knee estimated after end-of-life): "
          f"{'FIRED - see per-battery CAVEAT lines above' if any_past_eol else 'did not fire for any battery in this test set'}")

    result_df.to_csv(OUT_DIR / "bacon_watts_knee_detection.csv", index=False)
    print(f"\n[bacon-watts] saved outputs/bacon_watts_knee_detection.csv")
    print("[bacon-watts] DONE")


if __name__ == "__main__":
    main()
