"""
Final verification pass, CHECK 2: online conformal audit (item A).

(a) No-lookahead: a real unit test (not just a structural argument) -
    for a synthetic battery trace, runs the PID/nexCP trackers once,
    records the interval at cycle t, then SHUFFLES every label at
    cycles > t (chosen so the shuffle changes the actual residual
    values at those future cycles - verified, not assumed) and reruns
    the SAME trackers, asserting the interval at cycle t is BYTE-
    IDENTICAL in both runs. Also demonstrated on 3 REAL battery traces
    (one per dataset family: CALCE, HUST, isu_ilcc) for a non-synthetic
    check.
(b) Label-free reference in the SAME table as online PID/nexCP - makes
    explicit that PID/nexCP need the target's OWN true SOH labels fed
    back cycle-by-cycle as ground truth arrives (a real operational
    requirement a deployed monitoring system would need to satisfy),
    while the static split-conformal baseline needs NO target labels
    at all, ever.
(c) For the 10/13 datasets where item A's own rolling-20 coverage MIN
    was 0.00 (PID eta=0.1, k_burnin=10): identifies WHICH life stage
    (burn-in-adjacent early / mid / late, tertiles of each battery's
    own post-burn-in cycle range) the zero-coverage window falls in,
    per battery, then reports the aggregate distribution per dataset -
    not just re-asserting the aggregate minimum already known.
(d) Mean interval width reported as a PERCENT of that dataset's own
    true SOH range (max-min) - identifies where intervals are
    uselessly wide vs. genuinely informative, a scale-normalized view
    the raw SOH-point widths in item A's own table did not provide.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_finalpass2_itemA_online_conformal import (
    compute_q_src, pid_tracker, nexcp_tracker, ETA_PRIMARY, BURNIN_PRIMARY, RHO_VALUES,
)
from researchpass_partA_common import (
    load_all_heldout_base, load_all_heldout_extended, build_X, OUT_DIR, PROC_DIR,
)

BATTERYLIFE_SOURCES = ["ul_pur", "hnei", "snl", "mich", "mich_exp", "rwth",
                        "stanford", "stanford_2", "isu_ilcc"]
EXTENDED_ROUTED_DATASETS = {"CALCE", "Oxford", "HUST"}
ALPHA = 0.1
SEED = 42


def part_a_unit_test():
    print("=== (a) No-lookahead unit test ===")
    rng = np.random.default_rng(SEED)
    n = 200
    yhat = 90 - 0.02 * np.arange(n) + rng.normal(0, 0.3, n)
    y = yhat + rng.normal(0, 1.0, n)
    q_src = 1.0
    eta = 0.1
    k_burnin = 10
    t_check = 80

    lo1, hi1, q_used1 = pid_tracker(y, yhat, q_src, eta, k_burnin)
    y_shuffled = y.copy()
    future = y_shuffled[t_check + 1:]
    rng.shuffle(future)
    y_shuffled[t_check + 1:] = future
    assert not np.array_equal(y_shuffled, y), "shuffle produced no change - test is vacuous, fix the RNG/slice"
    lo2, hi2, q_used2 = pid_tracker(y_shuffled, yhat, q_src, eta, k_burnin)

    match_before = np.array_equal(lo1[:t_check + 1], lo2[:t_check + 1]) and np.array_equal(hi1[:t_check + 1], hi2[:t_check + 1])
    changed_after = not (np.array_equal(lo1[t_check + 1:], lo2[t_check + 1:]) and np.array_equal(hi1[t_check + 1:], hi2[t_check + 1:]))
    print(f"[check2a] PID synthetic test: interval[0..{t_check}] IDENTICAL after shuffling labels at "
          f"cycles >{t_check}: {match_before} (must be True)")
    print(f"[check2a] PID synthetic test: interval[{t_check+1}..] CHANGED after the shuffle: "
          f"{changed_after} (expected True - confirms the shuffle actually perturbed downstream cycles, "
          f"i.e. the test is non-vacuous)")
    assert match_before, "LOOKAHEAD BUG: PID interval before t_check changed after shuffling FUTURE labels"

    lo3, hi3 = nexcp_tracker(y, yhat, q_src, 0.95, k_burnin)
    lo4, hi4 = nexcp_tracker(y_shuffled, yhat, q_src, 0.95, k_burnin)
    match_before_nexcp = np.array_equal(lo3[:t_check + 1], lo4[:t_check + 1]) and np.array_equal(hi3[:t_check + 1], hi4[:t_check + 1])
    print(f"[check2a] nexCP synthetic test: interval[0..{t_check}] IDENTICAL after shuffling labels at "
          f"cycles >{t_check}: {match_before_nexcp} (must be True)")
    assert match_before_nexcp, "LOOKAHEAD BUG: nexCP interval before t_check changed after shuffling FUTURE labels"

    print("[check2a] SYNTHETIC TEST: PASSED for both PID and nexCP.\n")

    print("[check2a] Repeating on 3 REAL battery traces (CALCE, HUST, isu_ilcc)...")
    q_src_base, base_cols, base_medians, base_model, q_src_ext, ext_cols, ext_medians, ext_model = compute_q_src()
    from researchpass_partA_common import load_base_pool_and_split, load_extended_pool_and_split
    _, hi_full_base = load_base_pool_and_split()[:2]
    merged_e, hi_full_ext, hi_full_raw, *_ = load_extended_pool_and_split()
    heldout_base = load_all_heldout_base(hi_full_base)
    heldout_ext = load_all_heldout_extended(hi_full_raw)

    real_checks = [
        ("CALCE", heldout_ext["CALCE"], ext_cols, ext_medians, ext_model, q_src_ext),
        ("HUST", heldout_ext["HUST"], ext_cols, ext_medians, ext_model, q_src_ext),
    ]
    isu_path = PROC_DIR / "batterylife_isu_ilcc_merged.parquet"
    if isu_path.exists():
        real_checks.append(("isu_ilcc", pd.read_parquet(isu_path), base_cols, base_medians, base_model, q_src_base))

    all_passed = True
    for name, df, cols, medians, model, q_src in real_checks:
        bid = df["battery_id"].value_counts().index[0]
        battery_df = df[df["battery_id"] == bid].sort_values("cycle_idx")
        if len(battery_df) < 30:
            continue
        X = build_X(battery_df, cols, medians)
        pred = model.predict(X)
        y_real = battery_df["SOH"].to_numpy()
        n_real = len(y_real)
        t_c = n_real // 2
        lo_r1, hi_r1, _ = pid_tracker(y_real, pred, q_src, 0.1, 10)
        y_real_shuf = y_real.copy()
        fut = y_real_shuf[t_c + 1:]
        rng.shuffle(fut)
        y_real_shuf[t_c + 1:] = fut
        lo_r2, hi_r2, _ = pid_tracker(y_real_shuf, pred, q_src, 0.1, 10)
        ok = np.array_equal(lo_r1[:t_c + 1], lo_r2[:t_c + 1]) and np.array_equal(hi_r1[:t_c + 1], hi_r2[:t_c + 1])
        all_passed &= ok
        print(f"[check2a] {name}/{bid} (n={n_real}, t_check={t_c}): no-lookahead holds = {ok}")
        assert ok, f"LOOKAHEAD BUG on real data: {name}/{bid}"

    print(f"[check2a] REAL-DATA TEST: {'ALL PASSED' if all_passed else 'FAILED'}\n")
    return True


def score_with_lifestage(y, yhat, lo, hi, k_burnin):
    n = len(y)
    if n <= k_burnin + 20:
        return None
    sl = slice(k_burnin, n)
    covered = (y[sl] >= lo[sl]) & (y[sl] <= hi[sl])
    n_post = len(covered)
    roll = pd.Series(covered.astype(float)).rolling(20).mean()
    roll_min = float(roll.min())
    if roll_min > 1e-9:
        return {"n_post": n_post, "roll_min": roll_min, "zero_window_lifestage": None}
    zero_positions = np.where(roll.to_numpy() <= 1e-9)[0]
    mid_zero_pos = float(np.median(zero_positions))
    life_frac = mid_zero_pos / n_post
    stage = "early" if life_frac < 1 / 3 else ("mid" if life_frac < 2 / 3 else "late")
    return {"n_post": n_post, "roll_min": roll_min, "zero_window_lifestage": stage, "zero_window_life_frac": life_frac}


def main():
    t0 = time.time()
    part_a_unit_test()

    print("=== (b)+(c)+(d): label-free vs. online table, zero-coverage life-stage, width-as-%-of-range ===")
    q_src_base, base_cols, base_medians, base_model, q_src_ext, ext_cols, ext_medians, ext_model = compute_q_src()
    from researchpass_partA_common import load_base_pool_and_split, load_extended_pool_and_split
    _, hi_full_base = load_base_pool_and_split()[:2]
    merged_e, hi_full_ext, hi_full_raw, *_ = load_extended_pool_and_split()
    heldout_base = load_all_heldout_base(hi_full_base)
    heldout_ext = load_all_heldout_extended(hi_full_raw)

    static_cov = pd.read_csv(OUT_DIR / "finalpass_item5e_conformal_consolidated.csv").set_index("dataset")

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
    lifestage_rows = []

    for name in all_datasets:
        if name in BATTERYLIFE_SOURCES and not (PROC_DIR / f"batterylife_{name}_merged.parquet").exists():
            continue
        df, cols, medians, model, q_src = get_target(name)
        X = build_X(df, cols, medians)
        pred = model.predict(X)
        y = df["SOH"].to_numpy()
        bids = df["battery_id"].to_numpy()
        soh_range = float(y.max() - y.min())

        batteries = sorted(set(bids.tolist()))
        battery_data = {}
        for bid in batteries:
            mask = bids == bid
            order = np.argsort(df["cycle_idx"].to_numpy()[mask])
            battery_data[bid] = (y[mask][order], pred[mask][order])

        method_results = {"PID": [], "nexCP_0.95": [], "nexCP_0.99": []}
        lifestage_results = {"PID": [], "nexCP_0.95": [], "nexCP_0.99": []}
        for bid, (yb, predb) in battery_data.items():
            n = len(yb)
            if n <= BURNIN_PRIMARY:
                continue
            eta = max(ETA_PRIMARY * max(np.abs(yb[:BURNIN_PRIMARY] - predb[:BURNIN_PRIMARY])), 1e-6)
            lo_p, hi_p, _ = pid_tracker(yb, predb, q_src, eta, BURNIN_PRIMARY)
            ls = score_with_lifestage(yb, predb, lo_p, hi_p, BURNIN_PRIMARY)
            if ls:
                lifestage_results["PID"].append(ls)
            sl = slice(BURNIN_PRIMARY, n)
            covered = (yb[sl] >= lo_p[sl]) & (yb[sl] <= hi_p[sl])
            method_results["PID"].append({"coverage": covered.mean(), "width": np.mean(hi_p[sl] - lo_p[sl]), "n": len(covered)})

            for rho in RHO_VALUES:
                lo_n, hi_n = nexcp_tracker(yb, predb, q_src, rho, BURNIN_PRIMARY)
                ls_n = score_with_lifestage(yb, predb, lo_n, hi_n, BURNIN_PRIMARY)
                if ls_n:
                    lifestage_results[f"nexCP_{rho}"].append(ls_n)
                covered_n = (yb[sl] >= lo_n[sl]) & (yb[sl] <= hi_n[sl])
                method_results[f"nexCP_{rho}"].append({"coverage": covered_n.mean(), "width": np.mean(hi_n[sl] - lo_n[sl]), "n": len(covered_n)})

        for method, results in method_results.items():
            if not results:
                continue
            rdf = pd.DataFrame(results)
            w = rdf["n"].to_numpy()
            mean_cov = float(np.average(rdf["coverage"], weights=w))
            mean_width = float(np.average(rdf["width"], weights=w))
            rows.append({"dataset": name, "method": method, "coverage": mean_cov, "mean_width": mean_width,
                        "soh_range": soh_range, "width_pct_of_range": mean_width / soh_range * 100})

        for method, ls_list in lifestage_results.items():
            zero_batteries = [x for x in ls_list if x["zero_window_lifestage"] is not None]
            n_batteries_total = len(ls_list)
            n_zero = len(zero_batteries)
            stage_counts = pd.Series([x["zero_window_lifestage"] for x in zero_batteries]).value_counts().to_dict() if zero_batteries else {}
            lifestage_rows.append({"dataset": name, "method": method, "n_batteries": n_batteries_total,
                                   "n_batteries_with_zero_rolling_coverage": n_zero,
                                   "early_count": stage_counts.get("early", 0), "mid_count": stage_counts.get("mid", 0),
                                   "late_count": stage_counts.get("late", 0)})

        static = static_cov.loc[name] if name in static_cov.index else None
        static_cov_val = float(static["coverage"]) * 100 if static is not None else np.nan
        print(f"[check2] {name}: static(label-free)={static_cov_val:.1f}% | "
              f"PID={next((r['coverage'] for r in rows if r['dataset']==name and r['method']=='PID'), np.nan)*100:.1f}% | "
              f"width%range PID={next((r['width_pct_of_range'] for r in rows if r['dataset']==name and r['method']=='PID'), np.nan):.1f}%")

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUT_DIR / "finalpass3_check2_width_and_coverage.csv", index=False)

    lifestage_df = pd.DataFrame(lifestage_rows)
    lifestage_df.to_csv(OUT_DIR / "finalpass3_check2_lifestage.csv", index=False)

    # (b) consolidated label-free vs online table
    consolidated = []
    for name in all_datasets:
        if name in BATTERYLIFE_SOURCES and not (PROC_DIR / f"batterylife_{name}_merged.parquet").exists():
            continue
        static_pct = float(static_cov.loc[name, "coverage"]) * 100 if name in static_cov.index else np.nan
        sub = result_df[result_df["dataset"] == name]
        row = {"dataset": name, "static_split_conformal_LABEL_FREE_pct": static_pct}
        for method in ["PID", "nexCP_0.95", "nexCP_0.99"]:
            m = sub[sub["method"] == method]
            row[f"{method}_ONLINE_needs_labels_pct"] = float(m["coverage"].iloc[0]) * 100 if len(m) else np.nan
        consolidated.append(row)
    consolidated_df = pd.DataFrame(consolidated)
    consolidated_df.to_csv(OUT_DIR / "finalpass3_check2_labelfree_vs_online.csv", index=False)

    print("\n=== (b) LABEL-FREE (static) vs. ONLINE (needs sequential target labels) coverage ===")
    print(consolidated_df.to_string(index=False))

    print("\n=== (c) Life-stage of the rolling-20 zero-coverage window (PID, k_burnin=10) ===")
    print(lifestage_df[lifestage_df["method"] == "PID"].to_string(index=False))

    print("\n=== (d) Mean width as % of dataset's own SOH range (PID) ===")
    print(result_df[result_df["method"] == "PID"][["dataset", "mean_width", "soh_range", "width_pct_of_range"]].to_string(index=False))

    print(f"\n[check2] TOTAL TIME: {(time.time()-t0)/60:.2f} minutes")


if __name__ == "__main__":
    main()
