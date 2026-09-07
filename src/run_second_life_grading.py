"""
Session 25: second-life grading classifier - a pure post-processing/
labeling step on the existing "lean" (session 20) XGBoost-fusion SOH
predictions (`xgb_fusion_preds.csv`, test split). No new model, no
retraining: this only buckets already-computed SOH numbers (both true
and predicted) into 3 industry-standard grades.

Thresholds (per instruction, standard second-life battery literature
convention - e.g. the SOH>=80% "first-life EOL" cutoff is the same one
already used project-wide as the RUL/EOL definition in
`rul_labels.py`):
    SOH >= 80%        -> "Primary EV use"
    50% <= SOH < 80%  -> "Second-life candidate (grid storage/backup)"
    SOH < 50%         -> "Recycle only"

**The point of doing this on BOTH true and predicted SOH, not just
predicted**: grading is a discrete decision with real consequences, so
this session also reports whether the model's prediction ERROR ever
changes what grade a battery would ACTUALLY receive - a small SOH
error near a threshold boundary (e.g. true=79.6%, predicted=80.4%)
flips a real-world disposition decision even though it's an
unremarkable error in RMSE terms. Misgrades are additionally split by
DIRECTION, since they are not symmetric in consequence:
  - "risky" misgrade: predicted grade is MORE optimistic than true
    grade (e.g. true=recycle-only, predicted=second-life-candidate) -
    a safety/reliability-relevant error, since a battery that should be
    recycled would be routed into secondary service instead.
  - "conservative" misgrade: predicted grade is MORE pessimistic than
    true grade (e.g. true=second-life, predicted=recycle-only) - an
    economic-loss error (a usable asset gets scrapped early), not a
    safety error.
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
OUT_DIR.mkdir(exist_ok=True)

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
    df = pd.read_csv(PRED_DIR / "xgb_fusion_preds.csv")
    test = df[df["split"] == "test"].copy()
    print(f"[gradelabel] lean XGBoost-fusion test-set rows: {len(test)}, "
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

    # -------------------------------------------------------------
    # 1. Overall grading distribution (per-cycle, all test cycles pooled)
    # -------------------------------------------------------------
    print("\n[gradelabel] === per-cycle grade distribution across all test cycles ===")
    dist = pd.DataFrame({
        "true_grade": test["true_grade"].value_counts().reindex(GRADE_ORDER, fill_value=0),
        "pred_grade": test["pred_grade"].value_counts().reindex(GRADE_ORDER, fill_value=0),
    })
    dist["true_pct"] = (dist["true_grade"] / len(test) * 100).round(1)
    dist["pred_pct"] = (dist["pred_grade"] / len(test) * 100).round(1)
    print(dist.to_string())

    # -------------------------------------------------------------
    # 2. Grading confusion matrix (true vs predicted)
    # -------------------------------------------------------------
    print("\n[gradelabel] === grading confusion matrix (rows=true, cols=predicted) ===")
    confusion = pd.crosstab(test["true_grade"], test["pred_grade"]).reindex(
        index=GRADE_ORDER, columns=GRADE_ORDER, fill_value=0
    )
    print(confusion.to_string())
    agreement = (test["true_grade"] == test["pred_grade"]).mean()
    print(f"\n[gradelabel] grading agreement (predicted grade == true grade): {agreement*100:.2f}% "
          f"of {len(test)} test cycles")

    misgrade_counts = test["misgrade_direction"].value_counts()
    print(f"\n[gradelabel] misgrade breakdown:")
    for k in ["correct", "risky (predicted too optimistic)", "conservative (predicted too pessimistic)"]:
        n = misgrade_counts.get(k, 0)
        print(f"    {k}: {n} ({n/len(test)*100:.2f}%)")

    risky = test[test["misgrade_direction"].str.startswith("risky")]
    if len(risky):
        print(f"\n[gradelabel] RISKY misgrades detail (predicted more optimistic than true - "
              f"the safety-relevant direction):")
        print(risky[["dataset", "battery_id", "cycle_idx", "SOH", "y_pred_soh_fusion",
                      "true_grade", "pred_grade"]].to_string(index=False))

    # -------------------------------------------------------------
    # 3. Per-battery "current status" grading (last test cycle per
    #    battery - the realistic point at which a real triage decision
    #    would be made, i.e. when a cell is pulled from primary service)
    # -------------------------------------------------------------
    print("\n[gradelabel] === per-battery CURRENT-STATUS grade (last test cycle per battery) ===")
    last_cycle = test.sort_values("cycle_idx").groupby(["dataset", "battery_id"]).tail(1)
    last_cycle = last_cycle[["dataset", "battery_id", "cycle_idx", "SOH", "y_pred_soh_fusion",
                              "true_grade", "pred_grade"]].sort_values("battery_id")
    print(last_cycle.to_string(index=False))

    # -------------------------------------------------------------
    # 4. Per-battery full-life grade distribution (how much of each
    #    battery's OBSERVED test-window life falls in each grade)
    # -------------------------------------------------------------
    print("\n[gradelabel] === per-battery full-life grade distribution (%, true SOH) ===")
    per_batt = test.groupby(["dataset", "battery_id"])["true_grade"].value_counts(normalize=True).unstack(fill_value=0) * 100
    per_batt = per_batt.reindex(columns=GRADE_ORDER, fill_value=0).round(1)
    print(per_batt.to_string())

    test.to_csv(PRED_DIR / "second_life_grading_per_cycle.csv", index=False)
    last_cycle.to_csv(OUT_DIR / "second_life_grading_current_status.csv", index=False)
    per_batt.to_csv(OUT_DIR / "second_life_grading_per_battery_distribution.csv")
    print(f"\n[gradelabel] saved per-cycle grades, current-status summary, "
          f"and per-battery distribution")
    print("[gradelabel] DONE")


if __name__ == "__main__":
    main()
