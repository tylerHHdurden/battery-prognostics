"""
Stage 4, Step 3: re-run session 25's second-life grading methodology
(unchanged thresholds/logic) against the NEW Stage 4 model's test-set
predictions (xgb_fusion_preds_stage4.csv, from Step 2b) - neither this
nor sensor-noise robustness has been run against a model incorporating
Stage 1+2's full fixes together with the recovered batteries.
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
    df = pd.read_csv(PRED_DIR / "xgb_fusion_preds_stage4.csv")
    test = df[df["split"] == "test"].copy()
    print(f"[stage4-grade] Stage 4 model test-set rows: {len(test)}, "
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
    print(f"\n[stage4-grade] grading agreement: {agreement*100:.2f}% of {len(test)} test cycles "
          f"(prior lean model, session 25: 98.75%)")

    misgrade_counts = test["misgrade_direction"].value_counts()
    print(f"[stage4-grade] misgrade breakdown:")
    for k in ["correct", "risky (predicted too optimistic)", "conservative (predicted too pessimistic)"]:
        n = misgrade_counts.get(k, 0)
        print(f"    {k}: {n} ({n/len(test)*100:.2f}%)")

    risky = test[test["misgrade_direction"].str.startswith("risky")]
    print(f"\n[stage4-grade] RISKY misgrades: {len(risky)}")
    if len(risky):
        print(risky[["dataset", "battery_id", "cycle_idx", "SOH", "y_pred_soh_fusion",
                      "true_grade", "pred_grade"]].to_string(index=False))

    print("\n[stage4-grade] === per-battery CURRENT-STATUS grade (last test cycle per battery) ===")
    last_cycle = test.sort_values("cycle_idx").groupby(["dataset", "battery_id"]).tail(1)
    last_cycle = last_cycle[["dataset", "battery_id", "cycle_idx", "SOH", "y_pred_soh_fusion",
                              "true_grade", "pred_grade"]].sort_values("battery_id")
    print(last_cycle.to_string(index=False))

    b0018 = last_cycle[last_cycle["battery_id"] == "B0018"]
    if len(b0018):
        row = b0018.iloc[0]
        err = row["y_pred_soh_fusion"] - row["SOH"]
        print(f"\n[stage4-grade] B0018 (known mis-certification battery from prior sessions): "
              f"true SOH={row['SOH']:.2f} pred SOH={row['y_pred_soh_fusion']:.2f} error={err:+.2f} "
              f"true_grade={row['true_grade']} pred_grade={row['pred_grade']} "
              f"misgraded={'YES' if row['true_grade'] != row['pred_grade'] else 'no'}")

    test.to_csv(PRED_DIR / "stage4_second_life_grading_per_cycle.csv", index=False)
    last_cycle.to_csv(OUT_DIR / "stage4_second_life_grading_current_status.csv", index=False)
    print(f"\n[stage4-grade] saved outputs/stage4_second_life_grading_current_status.csv")
    print("[stage4-grade] DONE")


if __name__ == "__main__":
    main()
