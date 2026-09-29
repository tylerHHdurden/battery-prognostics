"""
Toolkit Phase 3(a), second half: derive a recommended MINIMUM number of
labeled checkpoints before an online-conformal interval should be
trusted, per dataset and overall, from item A's own PID results
(primary config: eta=0.1, k_burnin=10).

Definition (stated explicitly, not left implicit): for each battery,
the "trustworthy point" is the SMALLEST number of REVEALED cycles t
such that rolling-20 coverage, computed from t onward, NEVER again
drops below a TRUST_THRESHOLD (0.70 - a practical bar clearly below
the 90% target but far above the ~5-20% static-conformal floor this
whole exercise exists to fix) for the rest of that battery's life. A
battery whose rolling coverage never stabilizes above the threshold at
all is EXCLUDED from the median (flagged, counted, and reported
separately) rather than assigned a misleading number.

Uses `src/online_conformal.py`'s own `run_batch` (now this project's
single implementation of the PID/nexCP recursion), reusing the exact
q_src/eta/model-loading machinery `run_finalpass2_itemA_online_
conformal.py` established - not a new pipeline.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from online_conformal import run_batch
from run_finalpass2_itemA_online_conformal import compute_q_src, ETA_PRIMARY, BURNIN_PRIMARY
from researchpass_partA_common import (
    load_base_pool_and_split, load_extended_pool_and_split,
    load_all_heldout_base, load_all_heldout_extended, build_X, OUT_DIR, PROC_DIR,
)

BATTERYLIFE_SOURCES = ["ul_pur", "hnei", "snl", "mich", "mich_exp", "rwth",
                        "stanford", "stanford_2", "isu_ilcc"]
EXTENDED_ROUTED_DATASETS = {"CALCE", "Oxford", "HUST"}
TRUST_THRESHOLD = 0.70


def min_checkpoints_for_battery(y: np.ndarray, yhat: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                                 k_burnin: int) -> int | None:
    n = len(y)
    if n <= k_burnin + 20:
        return None
    covered = (y >= lo) & (y <= hi)
    roll = pd.Series(covered.astype(float)).rolling(20).mean().to_numpy()
    # find the LAST index where roll < TRUST_THRESHOLD (post burn-in) - the
    # trustworthy point is one past that, i.e. never drops below it again
    post = roll[k_burnin:]
    below = np.where(post < TRUST_THRESHOLD)[0]
    if len(below) == 0:
        return k_burnin  # trustworthy from burn-in itself onward
    last_below = below[-1]
    trustworthy_idx = k_burnin + last_below + 1
    if trustworthy_idx >= n - 5:  # never actually stabilizes with meaningful remaining life
        return None
    return int(trustworthy_idx)


def main():
    t0 = time.time()
    print("=== Toolkit Phase 3(a): minimum labeled checkpoints before trusting the online interval ===")
    print(f"[minck] trust threshold: rolling-20 coverage >= {TRUST_THRESHOLD:.0%}, must hold for the "
          f"REST of the battery's life once reached (PID, eta={ETA_PRIMARY}, k_burnin={BURNIN_PRIMARY})")

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
    all_values = []

    for name in all_datasets:
        if name in BATTERYLIFE_SOURCES and not (PROC_DIR / f"batterylife_{name}_merged.parquet").exists():
            continue
        df, cols, medians, model, q_src = get_target(name)
        X = build_X(df, cols, medians)
        pred = model.predict(X)
        y = df["SOH"].to_numpy()
        bids = df["battery_id"].to_numpy()
        cyc = df["cycle_idx"].to_numpy()

        batteries = sorted(set(bids.tolist()))
        vals, n_never = [], 0
        for bid in batteries:
            mask = bids == bid
            order = np.argsort(cyc[mask])
            yb, predb, cycb = y[mask][order], pred[mask][order], cyc[mask][order]
            n = len(yb)
            if n <= BURNIN_PRIMARY:
                continue
            eta = max(ETA_PRIMARY * max(np.abs(yb[:BURNIN_PRIMARY] - predb[:BURNIN_PRIMARY])), 1e-6)
            lo, hi = run_batch(yb, predb, q_src, "PID", k_burnin=BURNIN_PRIMARY, eta=eta, cyc=cycb.astype(float))
            v = min_checkpoints_for_battery(yb, predb, lo, hi, BURNIN_PRIMARY)
            if v is None:
                n_never += 1
            else:
                vals.append(v)
                all_values.append(v)

        if vals:
            median_v = float(np.median(vals))
            mean_v = float(np.mean(vals))
            print(f"[minck] {name}: median={median_v:.0f} mean={mean_v:.0f} cycles "
                  f"(n_batteries_with_stable_point={len(vals)}, n_never_stabilized={n_never})")
        else:
            median_v = mean_v = np.nan
            print(f"[minck] {name}: NO battery reached a stable trustworthy point "
                  f"(n_never_stabilized={n_never})")
        rows.append({"dataset": name, "median_checkpoints": median_v, "mean_checkpoints": mean_v,
                    "n_batteries_stable": len(vals), "n_batteries_never_stabilized": n_never,
                    "n_batteries_total": len(batteries)})

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUT_DIR / "toolkit_phase3a_min_checkpoints.csv", index=False)

    overall_median = float(np.median(all_values)) if all_values else np.nan
    overall_p75 = float(np.percentile(all_values, 75)) if all_values else np.nan
    print(f"\n=== OVERALL: median={overall_median:.0f} cycles, 75th percentile={overall_p75:.0f} cycles "
          f"(pooled across {len(all_values)} batteries that reached a stable trustworthy point)")
    print(result_df.to_string(index=False))

    print(f"\n[minck] TOTAL TIME: {(time.time()-t0)/60:.2f} minutes")


if __name__ == "__main__":
    main()
