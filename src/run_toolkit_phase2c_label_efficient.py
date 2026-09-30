"""
Toolkit Phase 2C: label-efficient checkpoints.

Question: at a FIXED, EQUAL label budget per battery (5/10/20/40 labels
- true SOH revealed only at those chosen cycles, the model still
PREDICTS every cycle via `OnlineConformal.interval()` unchanged), which
checkpoint-SELECTION policy gets the best coverage/width tradeoff, and
what's a defensible minimum-labels recommendation?

Reuses this project's own established machinery, not new pipelines:
- `src/online_conformal.py`'s PID recursion (`OnlineConformal`) and its
  new Phase 2C additions (`fixed_every_n_schedule`, `life_stage_schedule`,
  `UncertaintyTriggeredScheduler`, `run_batch_with_schedule`) - the
  no-lookahead property for both the recursion AND the schedules
  themselves is verified in `test_online_conformal.py`, not just here.
- `run_toolkit_phase3a_min_checkpoints.py`'s own dataset-routing
  (`get_target`) and per-battery series construction, applied to the
  SAME 13-source external/BatteryLife scope that script already
  established (CALCE/Oxford/HUST/XJTU + 9 BatteryLife sources - Tongji
  omitted here for direct comparability with Phase 3(a)'s own existing
  numbers, which predate Tongji's Phase 1(b) integration).
- `run_finalpass2_itemA_online_conformal.py`'s own q_src/eta primary
  configuration (PID, eta=0.1, k_burnin=10) - unchanged, not re-tuned
  for this phase.

Three policies compared at each of 4 equal label budgets:
  1. fixed_every_n   - evenly spaced, from the battery's own known
                        total length (not a label value).
  2. life_stage      - denser late in life (gamma=0.5 power-law warp).
  3. uncertainty      - reveals when conformal interval half-width
                        crosses `UNCERTAINTY_WIDTH_THRESHOLD` or an
                        ADWIN drift detector (fed only already-revealed
                        residuals) fires, one reveal guaranteed per
                        budget-slot either way (same total budget as
                        the other two policies, only the redistribution
                        within each slot differs).

Metrics reported per (dataset, budget, policy): coverage (over ALL
cycles, using whatever interval state was current at that cycle, not
just the revealed ones), rolling-20 minimum coverage, late-life
coverage (final 20% of each battery's own length), mean interval
width. A minimum-labels recommendation is DERIVED from the results
(the smallest tested budget whose best-performing policy's pooled
coverage crosses a stated target), not asserted upfront.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from online_conformal import (fixed_every_n_schedule, life_stage_schedule,
                               UncertaintyTriggeredScheduler, run_batch_with_schedule)
from run_finalpass2_itemA_online_conformal import compute_q_src, ETA_PRIMARY, BURNIN_PRIMARY
from researchpass_partA_common import (
    load_base_pool_and_split, load_extended_pool_and_split,
    load_all_heldout_base, load_all_heldout_extended, build_X, OUT_DIR, PROC_DIR,
)

BATTERYLIFE_SOURCES = ["ul_pur", "hnei", "snl", "mich", "mich_exp", "rwth",
                        "stanford", "stanford_2", "isu_ilcc"]
EXTENDED_ROUTED_DATASETS = {"CALCE", "Oxford", "HUST"}
LABEL_BUDGETS = [5, 10, 20, 40]
UNCERTAINTY_WIDTH_THRESHOLD = 2.4  # same value verified to produce genuine, non-trivial,
# non-immediate-every-slot triggering in test_online_conformal.py's own probe of this scheduler
COVERAGE_TARGET = 0.80
SEED = 42


def battery_metrics(y, yhat, lo, hi, revealed) -> dict:
    covered = (y >= lo) & (y <= hi)
    coverage = float(covered.mean())
    roll = pd.Series(covered.astype(float)).rolling(20, min_periods=1).mean()
    rolling_min = float(roll.min())
    n = len(y)
    late_start = int(n * 0.8)
    late_coverage = float(covered[late_start:].mean()) if late_start < n else float("nan")
    mean_width = float(np.mean(hi - lo))
    return {"coverage": coverage, "rolling_min_coverage": rolling_min,
            "late_life_coverage": late_coverage, "mean_width": mean_width,
            "n_revealed": int(revealed.sum())}


def run_policy_for_battery(y, yhat, cyc, q_src, budget, policy) -> dict | None:
    n = len(y)
    if n < max(budget, BURNIN_PRIMARY + 5):
        return None
    eta = max(ETA_PRIMARY * max(np.abs(y[:BURNIN_PRIMARY] - yhat[:BURNIN_PRIMARY])), 1e-6)

    if policy == "fixed_every_n":
        cps = set(fixed_every_n_schedule(n, budget))
        lo, hi, revealed = run_batch_with_schedule(y, yhat, q_src, "PID", checkpoints=cps,
                                                     k_burnin=BURNIN_PRIMARY, eta=eta, cyc=cyc.astype(float))
    elif policy == "life_stage":
        cps = set(life_stage_schedule(n, budget))
        lo, hi, revealed = run_batch_with_schedule(y, yhat, q_src, "PID", checkpoints=cps,
                                                     k_burnin=BURNIN_PRIMARY, eta=eta, cyc=cyc.astype(float))
    elif policy == "uncertainty":
        sched = UncertaintyTriggeredScheduler(n, budget, width_threshold=UNCERTAINTY_WIDTH_THRESHOLD)
        lo, hi, revealed = run_batch_with_schedule(y, yhat, q_src, "PID", checkpoints=None, scheduler=sched,
                                                     k_burnin=BURNIN_PRIMARY, eta=eta, cyc=cyc.astype(float))
    else:
        raise ValueError(policy)

    return battery_metrics(y, yhat, lo, hi, revealed)


def main():
    t0 = time.time()
    print("=== Toolkit Phase 2C: label-efficient checkpoints ===", flush=True)
    print(f"[phase2c] budgets={LABEL_BUDGETS}, policies=[fixed_every_n, life_stage, uncertainty], "
          f"PID eta={ETA_PRIMARY} k_burnin={BURNIN_PRIMARY}, uncertainty width_threshold="
          f"{UNCERTAINTY_WIDTH_THRESHOLD}", flush=True)

    q_src_base, base_cols, base_medians, base_model, q_src_ext, ext_cols, ext_medians, ext_model = compute_q_src()
    _, hi_full_base = load_base_pool_and_split()[:2]
    merged_e, hi_full_ext, hi_full_raw, *_ = load_extended_pool_and_split()
    heldout_base = load_all_heldout_base(hi_full_base)
    heldout_ext = load_all_heldout_extended(hi_full_raw)

    def get_target(name):
        if name in {"CALCE", "Oxford", "HUST", "XJTU"}:
            routed = name in EXTENDED_ROUTED_DATASETS
            df = heldout_ext[name] if routed else heldout_base[name]
            cols, medians, model, q_src = (ext_cols, ext_medians, ext_model, q_src_ext) if routed else (base_cols, base_medians, base_model, q_src_base)
        else:
            df = pd.read_parquet(PROC_DIR / f"batterylife_{name}_merged.parquet")
            cols, medians, model, q_src = base_cols, base_medians, base_model, q_src_base
        return df, cols, medians, model, q_src

    all_datasets = ["CALCE", "Oxford", "HUST", "XJTU"] + BATTERYLIFE_SOURCES
    rows = []

    for name in all_datasets:
        if name in BATTERYLIFE_SOURCES and not (PROC_DIR / f"batterylife_{name}_merged.parquet").exists():
            continue
        df, cols, medians, model, q_src = get_target(name)
        X = build_X(df, cols, medians)
        pred = model.predict(X)
        y_all = df["SOH"].to_numpy()
        bids = df["battery_id"].to_numpy()
        cyc_all = df["cycle_idx"].to_numpy()
        batteries = sorted(set(bids.tolist()))

        for budget in LABEL_BUDGETS:
            for policy in ["fixed_every_n", "life_stage", "uncertainty"]:
                per_battery = []
                for bid in batteries:
                    mask = bids == bid
                    order = np.argsort(cyc_all[mask])
                    yb, predb, cycb = y_all[mask][order], pred[mask][order], cyc_all[mask][order]
                    m = run_policy_for_battery(yb, predb, cycb, q_src, budget, policy)
                    if m is not None:
                        per_battery.append(m)
                if not per_battery:
                    continue
                agg = {
                    "dataset": name, "budget": budget, "policy": policy,
                    "n_batteries": len(per_battery),
                    "mean_coverage": float(np.mean([m["coverage"] for m in per_battery])),
                    "mean_rolling_min_coverage": float(np.mean([m["rolling_min_coverage"] for m in per_battery])),
                    "mean_late_life_coverage": float(np.nanmean([m["late_life_coverage"] for m in per_battery])),
                    "mean_width": float(np.mean([m["mean_width"] for m in per_battery])),
                    "mean_n_revealed": float(np.mean([m["n_revealed"] for m in per_battery])),
                }
                rows.append(agg)
        print(f"[phase2c] {name}: {len(batteries)} batteries processed across all budgets/policies "
              f"({time.time()-t0:.1f}s elapsed)", flush=True)

    result_df = pd.DataFrame(rows)
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    result_df.to_csv(OUT_DIR / "toolkit_phase2c_label_efficient.csv", index=False)

    print("\n=== SUMMARY: mean coverage by budget x policy, pooled (unweighted) across datasets ===", flush=True)
    pooled = result_df.groupby(["budget", "policy"]).agg(
        mean_coverage=("mean_coverage", "mean"),
        mean_rolling_min_coverage=("mean_rolling_min_coverage", "mean"),
        mean_late_life_coverage=("mean_late_life_coverage", "mean"),
        mean_width=("mean_width", "mean"),
    ).reset_index()
    print(pooled.to_string(index=False), flush=True)

    print(f"\n=== Best policy per budget (by pooled mean_coverage) ===", flush=True)
    best_per_budget = {}
    for budget in LABEL_BUDGETS:
        sub = pooled[pooled["budget"] == budget]
        if len(sub):
            best_row = sub.loc[sub["mean_coverage"].idxmax()]
            best_per_budget[budget] = best_row
            print(f"  budget={budget}: best policy={best_row['policy']}, "
                  f"mean_coverage={best_row['mean_coverage']:.3f}, "
                  f"mean_rolling_min_coverage={best_row['mean_rolling_min_coverage']:.3f}", flush=True)

    recommended = None
    for budget in LABEL_BUDGETS:
        if budget in best_per_budget and best_per_budget[budget]["mean_coverage"] >= COVERAGE_TARGET:
            recommended = budget
            break
    if recommended is not None:
        print(f"\n[phase2c] RECOMMENDATION: smallest tested budget reaching pooled mean coverage "
              f">= {COVERAGE_TARGET:.0%} under its best policy = {recommended} labels/battery "
              f"(policy: {best_per_budget[recommended]['policy']})", flush=True)
    else:
        print(f"\n[phase2c] RECOMMENDATION: NONE of the tested budgets ({LABEL_BUDGETS}) reached "
              f"{COVERAGE_TARGET:.0%} pooled mean coverage under any policy - reported honestly, "
              f"not forced to the largest budget as a false positive recommendation.", flush=True)

    print(f"\n[phase2c] TOTAL TIME: {(time.time()-t0)/60:.2f} minutes", flush=True)


if __name__ == "__main__":
    main()
