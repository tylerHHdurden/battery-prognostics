"""
Dataset Expansion Phase 1, Step 4: re-runs session 25's second-life
grading on the expanded-pool lean (XGBoost-fusion-expanded) predictions
- specifically to check whether NASA/B0018's known grading failure
(predicted 81.50% vs. true 72.76% at its last test cycle, crossing the
80% line) changed now that NASA has ~6.75x more training representation
(2.7% -> a larger share of training cycles - exact figure reported in
the DEVELOPMENT_LOG entry from Step 1's counts). Same method/thresholds
as run_second_life_grading.py, unchanged - additive, does not touch it.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
PRED_DIR = PROC_DIR / "predictions"
OUT_DIR = ROOT / "outputs"

GRADE_ORDER = ["Recycle only", "Second-life candidate", "Primary EV use"]
GRADE_ORDINAL = {g: i for i, g in enumerate(GRADE_ORDER)}


def grade(soh: float) -> str:
    if soh >= 80:
        return "Primary EV use"
    elif soh >= 50:
        return "Second-life candidate"
    else:
        return "Recycle only"


def main():
    df = pd.read_csv(PRED_DIR / "xgb_fusion_expanded_preds.csv")
    test = df[df["split"] == "test"].copy()
    print(f"[gradelabel-exp] expanded lean XGBoost-fusion test-set rows: {len(test)}, "
          f"{test['battery_id'].nunique()} batteries")

    test["true_grade"] = test["SOH"].apply(grade)
    test["pred_grade"] = test["y_pred_soh_fusion"].apply(grade)
    test["true_ord"] = test["true_grade"].map(GRADE_ORDINAL)
    test["pred_ord"] = test["pred_grade"].map(GRADE_ORDINAL)
    test["misgrade_direction"] = np.select(
        [test["pred_ord"] > test["true_ord"], test["pred_ord"] < test["true_ord"]],
        ["risky (predicted too optimistic)", "conservative (predicted too pessimistic)"],
        default="correct",
    )

    agreement = (test["true_grade"] == test["pred_grade"]).mean()
    print(f"[gradelabel-exp] grading agreement: {agreement*100:.2f}% of {len(test)} test cycles "
          f"(original 32-battery run: 98.75% of 5,208 cycles)")

    last_cycle = test.sort_values("cycle_idx").groupby(["dataset", "battery_id"]).tail(1)
    last_cycle = last_cycle[["dataset", "battery_id", "cycle_idx", "SOH", "y_pred_soh_fusion",
                              "true_grade", "pred_grade"]].sort_values("battery_id")
    print("\n[gradelabel-exp] === per-battery CURRENT-STATUS grade (last test cycle) ===")
    print(last_cycle.to_string(index=False))

    b0018_rows = last_cycle[last_cycle["battery_id"] == "B0018"]
    if len(b0018_rows):
        r = b0018_rows.iloc[0]
        correct = r["true_grade"] == r["pred_grade"]
        print(f"\n[gradelabel-exp] NASA/B0018 specifically (the known original failure case): "
              f"true SOH={r['SOH']:.2f}% pred SOH={r['y_pred_soh_fusion']:.2f}% "
              f"true_grade={r['true_grade']} pred_grade={r['pred_grade']} "
              f"-> {'CORRECT now' if correct else 'STILL WRONG'}")
        print("[gradelabel-exp] ORIGINAL 32-battery run: true=72.76% pred=81.50% "
              "(true=Second-life candidate, pred=Primary EV use, WRONG, 8.7pp overestimate)")
    else:
        print("\n[gradelabel-exp] NOTE: B0018 is not in the expanded test set "
              "(battery_level_split may have placed it in train this time - reported, not hidden)")

    per_batt = test.groupby(["dataset", "battery_id"])["true_grade"].value_counts(normalize=True).unstack(fill_value=0) * 100
    per_batt = per_batt.reindex(columns=GRADE_ORDER, fill_value=0).round(1)

    test.to_csv(PRED_DIR / "second_life_grading_expanded_per_cycle.csv", index=False)
    last_cycle.to_csv(OUT_DIR / "second_life_grading_expanded_current_status.csv", index=False)
    per_batt.to_csv(OUT_DIR / "second_life_grading_expanded_per_battery_distribution.csv")
    print("\n[gradelabel-exp] DONE")


if __name__ == "__main__":
    main()
