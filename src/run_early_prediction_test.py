"""
Experiment 1/3 (remaining evaluation protocol): early-prediction test.

Uses the already-computed fusion-ensemble predictions
(ensemble_fusion_test_preds.csv, which has a row for EVERY cycle of every
held-out test battery, not just a sampled subset) - no retraining, just
a filtered re-evaluation. For each of the 6 test batteries, keeps only
the first 20% of its logged cycles (by cycle_idx, i.e. early-life data)
and recomputes RMSE/MAE/R2 on that early-only subset, compared against
the full-lifetime test metrics already reported (R2=0.917).

ASSUMPTION: "first 20% of each battery's historical cycles" is applied
per-battery (not 20% of the pooled cycle count), since batteries have
very different total lifetimes (132 to 1230 cycles here) - taking 20%
of each battery's OWN range is what "early in this specific battery's
life" means, and pooling would let long-lived batteries dominate.
Cutoff = max(1, round(0.2 * max_cycle_idx)) per battery.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
PRED_DIR = PROC_DIR / "predictions"
OUT_DIR = ROOT / "outputs"


def metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main():
    df = pd.read_csv(PRED_DIR / "ensemble_fusion_test_preds.csv")

    cutoffs = {}
    early_rows = []
    for bid, g in df.groupby("battery_id"):
        cutoff = max(1, round(0.2 * g["cycle_idx"].max()))
        cutoffs[bid] = (cutoff, g["cycle_idx"].max())
        early_rows.append(g[g["cycle_idx"] <= cutoff])
    early_df = pd.concat(early_rows, ignore_index=True)

    print("[early] per-battery early-life cutoff (first 20% of cycles):")
    for bid, (cutoff, total) in cutoffs.items():
        n_early = (early_df["battery_id"] == bid).sum()
        print(f"    {bid}: cycles 1-{cutoff} of {total} total ({n_early} rows)")
    print(f"[early] total early-life rows: {len(early_df)} (full test set: {len(df)})")

    full_metrics = metrics(df["SOH"], df["pred_Stacking_Ridge_fusion"])
    early_metrics = metrics(early_df["SOH"], early_df["pred_Stacking_Ridge_fusion"])
    early_xgb_metrics = metrics(early_df["SOH"], early_df["pred_XGBoost_fusion"])
    full_xgb_metrics = metrics(df["SOH"], df["pred_XGBoost_fusion"])

    print(f"\n[early] Stacking-Ridge-fusion FULL test-lifetime: {full_metrics}")
    print(f"[early] Stacking-Ridge-fusion EARLY-LIFE (first 20%): {early_metrics}")
    print(f"[early] XGBoost-fusion FULL test-lifetime: {full_xgb_metrics}")
    print(f"[early] XGBoost-fusion EARLY-LIFE (first 20%): {early_xgb_metrics}")

    # per-battery breakdown, since only 6 batteries and early-life SOH
    # range is narrow (all near 100%) - worth seeing individually
    per_battery = []
    for bid, g in early_df.groupby("battery_id"):
        m = metrics(g["SOH"], g["pred_Stacking_Ridge_fusion"]) if len(g) > 1 else \
            {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
        per_battery.append({"battery_id": bid, "n_early_cycles": len(g),
                             "soh_range": f"{g['SOH'].min():.1f}-{g['SOH'].max():.1f}", **m})
    per_battery_df = pd.DataFrame(per_battery)
    print("\n[early] per-battery early-life breakdown:")
    print(per_battery_df.to_string(index=False))

    summary = pd.DataFrame([
        {"model": "Stacking-Ridge-fusion", "regime": "full_lifetime", **full_metrics},
        {"model": "Stacking-Ridge-fusion", "regime": "early_life_20pct", **early_metrics},
        {"model": "XGBoost-fusion", "regime": "full_lifetime", **full_xgb_metrics},
        {"model": "XGBoost-fusion", "regime": "early_life_20pct", **early_xgb_metrics},
    ])
    summary.to_csv(PRED_DIR / "early_prediction_test.csv", index=False)
    per_battery_df.to_csv(PRED_DIR / "early_prediction_per_battery.csv", index=False)
    print("\n[early] DONE")


if __name__ == "__main__":
    main()
