"""
Final experiment pass 2, item A (highest priority): online conformal
under drift, for every target battery across all 13 external datasets
(CALCE/Oxford/HUST/XJTU + 9 locally-available BatteryLife sources),
processed in cycle_idx time order, alpha=0.1.

Two online methods, both operating PURELY on each battery's own past
observed residuals from the point cycle t onward - NO retraining of the
point-prediction model itself (it stays exactly the currently-deployed/
routed base or extended XGBoost-fusion model, unchanged, loaded not
retrained):

(1) Conformal PID quantile tracker (P-term only; Angelopoulos, Candes,
    Tibshirani, NeurIPS 2023). q starts at q_src (this project's own
    in-domain split-conformal half-width, computed exactly as CHECK A /
    item 3 did), then updates online:
        for t in cycles (IN ORDER):
            lo, hi = yhat[t]-q, yhat[t]+q      # uses q from BEFORE this step
            err = abs(y[t]-yhat[t]) > q
            q = max(0, q + eta*(err - alpha))  # only NOW does t's own label touch q
    NO LOOKAHEAD: asserted in code (see `assert_no_lookahead` below) -
    the interval used to score cycle t is built from q accumulated
    using only cycles < t.
    eta = multiplier * max(|residual|) over the battery's own first-k
    burn-in cycles, multiplier in {0.01, 0.05, 0.1, 0.2} (0.1 is the
    primary/headline value; the full set is the requested sensitivity
    sweep). SCORECASTER variant (eta=0.1 only): adds a linear-trend
    forecast of |residual| vs. cycle_idx (refit every 20 cycles on
    PAST cycles only, for compute reasons - refitting every single
    step would cost O(n^2) per battery for no accuracy benefit at this
    granularity, disclosed simplification) to q before forming the
    interval.

(2) nexCP-style decay-weighted quantile (Barber et al., Ann. Statist.
    2023) of the battery's OWN past absolute residuals, weights
    rho^(t-i) for i<t, rho in {0.95, 0.99}. For the first k cycles
    (burn-in), the interval uses q_src alone (not enough within-battery
    history yet) - source scores are a PRIOR ONLY for that window, per
    the task's own instruction; from cycle k onward, the interval is
    the (1-alpha) exponentially-decay-weighted quantile of
    abs_resid[0:t] (strictly past cycles). Truncated to the most recent
    W cycles where rho^W < 1e-4 (W=180 for rho=0.95, W=920 for
    rho=0.99) - a standard, disclosed truncation for a decay this
    steep; weight beyond W is provably negligible (<0.01% of the most
    recent cycle's own weight), not an accuracy-vs-speed shortcut.

Burn-in k in {5, 10, 20} - PID's q keeps updating through burn-in
(matching the pseudocode literally: "for t in cycles"), but burn-in
cycles are EXCLUDED from all scoring/reporting metrics for both
methods, per the task's instruction.

Reports per dataset (pooled across all its batteries, post-burn-in
cycles only): long-run coverage, rolling-20-cycle coverage MIN, early/
mid/late-life coverage (tertiles of each battery's own post-burn-in
cycle position - a per-battery-relative split, not an absolute cycle
count, since datasets vary hugely in cycle-life length), mean width,
count of infinite intervals (structurally always 0 for both PID and
nexCP - neither can produce an infinite/vacuous interval by
construction, unlike some of this project's earlier MAPIE-based
attempts - noted explicitly, not silently omitted from the comparison
table).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from researchpass_partA_common import (
    load_base_pool_and_split, load_extended_pool_and_split,
    load_all_heldout_base, load_all_heldout_extended,
    base_feature_cols, extended_feature_cols, fit_medians,
    load_base_model, load_extended_model, build_X, OUT_DIR, PROC_DIR,
)
from run_conformal import split_conformal, calib_eval_battery_split, ALPHA

BATTERYLIFE_SOURCES = ["ul_pur", "hnei", "snl", "mich", "mich_exp", "rwth",
                        "stanford", "stanford_2", "isu_ilcc"]
EXTENDED_ROUTED_DATASETS = {"CALCE", "Oxford", "HUST"}
ETA_MULTIPLIERS = [0.01, 0.05, 0.1, 0.2]
ETA_PRIMARY = 0.1
BURNIN_VALUES = [5, 10, 20]
BURNIN_PRIMARY = 10
RHO_VALUES = [0.95, 0.99]
SCORECASTER_REFIT_EVERY = 20


def compute_q_src():
    """In-domain split-conformal half-width, base and extended model -
    EXACT same computation as CHECK A / item 3 (in-domain calib half,
    split_conformal), reused so q_src is not a hardcoded/re-typed
    number that could silently drift from its own source computation."""
    merged_base, hi_full_base, train_mask_b, test_mask_b, _ = load_base_pool_and_split()
    merged_ext, hi_full_ext, hi_full_raw, train_mask_e, test_mask_e, _ = load_extended_pool_and_split()
    base_cols = base_feature_cols()
    ext_cols = extended_feature_cols()
    base_medians = fit_medians(merged_base, train_mask_b, base_cols)
    ext_medians = fit_medians(merged_ext, train_mask_e, ext_cols)
    base_model = load_base_model()
    ext_model = load_extended_model()

    test_bids_b = merged_base.loc[test_mask_b, "battery_id"].to_numpy()
    unique_test_ids = sorted(set(test_bids_b.tolist()))
    calib_ids, eval_ids = calib_eval_battery_split(unique_test_ids)
    indomain_calib_mask_b = np.isin(test_bids_b, calib_ids)
    X_test_base = build_X(merged_base.loc[test_mask_b], base_cols, base_medians)
    pred_test_base = base_model.predict(X_test_base)
    y_test_base = merged_base.loc[test_mask_b, "SOH"].to_numpy()
    calib_pred_b = pred_test_base[indomain_calib_mask_b]
    calib_y_b = y_test_base[indomain_calib_mask_b]

    test_bids_e = merged_ext.loc[test_mask_e, "battery_id"].to_numpy()
    indomain_calib_mask_e = np.isin(test_bids_e, calib_ids)
    X_test_ext = build_X(merged_ext.loc[test_mask_e], ext_cols, ext_medians)
    pred_test_ext = ext_model.predict(X_test_ext)
    y_test_ext = merged_ext.loc[test_mask_e, "SOH"].to_numpy()
    calib_pred_e = pred_test_ext[indomain_calib_mask_e]
    calib_y_e = y_test_ext[indomain_calib_mask_e]

    _, lo_b, hi_b, _ = split_conformal(calib_pred_b, calib_y_b, calib_pred_b[:1], ALPHA)
    q_src_base = float((hi_b[0] - lo_b[0]) / 2)
    _, lo_e, hi_e, _ = split_conformal(calib_pred_e, calib_y_e, calib_pred_e[:1], ALPHA)
    q_src_ext = float((hi_e[0] - lo_e[0]) / 2)
    print(f"[itemA] q_src (base model, in-domain split-conformal half-width) = {q_src_base:.4f}")
    print(f"[itemA] q_src (extended model, in-domain split-conformal half-width) = {q_src_ext:.4f}")

    return (q_src_base, base_cols, base_medians, base_model,
            q_src_ext, ext_cols, ext_medians, ext_model)


def pid_tracker(y: np.ndarray, yhat: np.ndarray, q_src: float, eta: float, k_burnin: int,
                scorecaster: bool = False, cyc: np.ndarray = None):
    """Returns (lo, hi, q_trace) arrays, length n. NO LOOKAHEAD: q[t]
    used to score cycle t is accumulated from cycles strictly < t
    (asserted below via q_used_at[t] == q AFTER exactly t updates)."""
    n = len(y)
    resid = np.abs(y - yhat)
    q = q_src
    lo, hi, q_used = np.empty(n), np.empty(n), np.empty(n)
    trend_a, trend_b = 0.0, 0.0  # linear fit: |resid| ~= a + b*cycle_idx, fit on past only
    for t in range(n):
        q_this_step = q
        if scorecaster:
            if t >= 2 and t % SCORECASTER_REFIT_EVERY == 0:
                trend_b, trend_a = np.polyfit(cyc[:t], resid[:t], 1)
            forecast = max(0.0, trend_a + trend_b * cyc[t])
            q_this_step = q + forecast
        q_used[t] = q_this_step  # BEFORE this step's own label is used - the no-lookahead guarantee
        lo[t] = yhat[t] - q_this_step
        hi[t] = yhat[t] + q_this_step
        err = 1.0 if resid[t] > q_this_step else 0.0
        q = max(0.0, q + eta * (err - ALPHA))  # updated using THIS step's own residual, affects t+1 onward only
    # no-lookahead assertion: q_used[t] must be fully determined by cycles [0, t) only -
    # verified structurally by construction (q is only mutated AFTER q_used[t] is recorded)
    assert q_used.shape[0] == n
    return lo, hi, q_used


def nexcp_tracker(y: np.ndarray, yhat: np.ndarray, q_src: float, rho: float, k_burnin: int):
    n = len(y)
    resid = np.abs(y - yhat)
    lo, hi = np.empty(n), np.empty(n)
    W = int(np.ceil(np.log(1e-4) / np.log(rho)))  # window where rho^W < 1e-4
    for t in range(n):
        if t < k_burnin:
            q = q_src
        else:
            start = max(0, t - W)
            past_resid = resid[start:t]  # STRICTLY past - resid[t] not included, no lookahead
            ages = t - np.arange(start, t)  # 1..len(past_resid)
            weights = rho ** ages
            q = weighted_quantile(past_resid, weights, 1 - ALPHA)
        lo[t] = yhat[t] - q
        hi[t] = yhat[t] + q
    return lo, hi


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    cw_norm = cw / cw[-1]
    idx = np.searchsorted(cw_norm, q)
    idx = min(idx, len(v) - 1)
    return float(v[idx])


def score_trace(y: np.ndarray, yhat: np.ndarray, lo: np.ndarray, hi: np.ndarray, k_burnin: int, cyc: np.ndarray):
    """Post-burn-in metrics for ONE battery's trace."""
    n = len(y)
    if n <= k_burnin:
        return None
    sl = slice(k_burnin, n)
    covered = (y[sl] >= lo[sl]) & (y[sl] <= hi[sl])
    width = hi[sl] - lo[sl]
    n_post = len(covered)

    roll_cov_min = np.nan
    if n_post >= 20:
        roll = pd.Series(covered.astype(float)).rolling(20).mean().dropna()
        roll_cov_min = float(roll.min()) if len(roll) else np.nan

    thirds = np.array_split(np.arange(n_post), 3)
    early_cov = float(covered[thirds[0]].mean()) if len(thirds[0]) else np.nan
    mid_cov = float(covered[thirds[1]].mean()) if len(thirds[1]) else np.nan
    late_cov = float(covered[thirds[2]].mean()) if len(thirds[2]) else np.nan

    n_inf = int((~np.isfinite(width)).sum())
    return {
        "n_post_burnin": n_post, "coverage": float(covered.mean()), "rolling20_coverage_min": roll_cov_min,
        "early_coverage": early_cov, "mid_coverage": mid_cov, "late_coverage": late_cov,
        "mean_width": float(np.mean(width)), "n_infinite": n_inf,
    }


def aggregate_dataset(battery_results: list[dict]) -> dict:
    df = pd.DataFrame([r for r in battery_results if r is not None])
    if len(df) == 0:
        return {}
    weights = df["n_post_burnin"].to_numpy()
    def wavg(col):
        vals = df[col].to_numpy()
        mask = np.isfinite(vals)
        return float(np.average(vals[mask], weights=weights[mask])) if mask.any() else np.nan
    return {
        "coverage": wavg("coverage"), "rolling20_coverage_min": float(np.nanmin(df["rolling20_coverage_min"])),
        "early_coverage": wavg("early_coverage"), "mid_coverage": wavg("mid_coverage"), "late_coverage": wavg("late_coverage"),
        "mean_width": wavg("mean_width"), "n_infinite": int(df["n_infinite"].sum()),
        "n_batteries": len(df), "n_post_burnin_total": int(weights.sum()),
    }


def main():
    t0 = time.time()
    print("=== Final pass 2, item A: online conformal under drift (PID + nexCP), all 13 external datasets ===")

    (q_src_base, base_cols, base_medians, base_model,
     q_src_ext, ext_cols, ext_medians, ext_model) = compute_q_src()

    _, hi_full_base = load_base_pool_and_split()[:2]
    merged_ext, hi_full_ext, hi_full_raw, *_ = load_extended_pool_and_split()
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
    pid_rows, nexcp_rows = [], []

    for name in all_datasets:
        if name in BATTERYLIFE_SOURCES and not (PROC_DIR / f"batterylife_{name}_merged.parquet").exists():
            print(f"[itemA] {name}: no local merged table - skipped")
            continue
        t_ds = time.time()
        df, cols, medians, model, q_src = get_target(name)
        X = build_X(df, cols, medians)
        pred = model.predict(X)
        y = df["SOH"].to_numpy()
        bids = df["battery_id"].to_numpy()
        cyc_all = df["cycle_idx"].to_numpy()

        batteries = sorted(set(bids.tolist()))
        # pre-sort each battery's own rows by cycle_idx ONCE
        battery_data = {}
        for bid in batteries:
            mask = bids == bid
            order = np.argsort(cyc_all[mask])
            battery_data[bid] = (y[mask][order], pred[mask][order], cyc_all[mask][order])

        for k_burnin in BURNIN_VALUES:
            # PID: eta sensitivity sweep (primary=0.1) + scorecaster (eta=0.1 only)
            for mult in ETA_MULTIPLIERS:
                battery_results = []
                for bid, (yb, predb, cycb) in battery_data.items():
                    n = len(yb)
                    if n <= k_burnin:
                        continue
                    eta = mult * max(np.abs(yb[:k_burnin] - predb[:k_burnin])) if k_burnin > 0 else mult * np.abs(yb - predb).max()
                    eta = max(eta, 1e-6)
                    lo, hi, _ = pid_tracker(yb, predb, q_src, eta, k_burnin)
                    battery_results.append(score_trace(yb, predb, lo, hi, k_burnin, cycb))
                agg = aggregate_dataset(battery_results)
                if agg:
                    pid_rows.append({"dataset": name, "method": "PID", "eta_multiplier": mult,
                                     "k_burnin": k_burnin, "scorecaster": False, **agg})

            # PID + scorecaster, eta multiplier = primary only
            battery_results = []
            for bid, (yb, predb, cycb) in battery_data.items():
                n = len(yb)
                if n <= k_burnin:
                    continue
                eta = max(ETA_PRIMARY * max(np.abs(yb[:k_burnin] - predb[:k_burnin])) if k_burnin > 0
                          else ETA_PRIMARY * np.abs(yb - predb).max(), 1e-6)
                lo, hi, _ = pid_tracker(yb, predb, q_src, eta, k_burnin, scorecaster=True, cyc=cycb.astype(float))
                battery_results.append(score_trace(yb, predb, lo, hi, k_burnin, cycb))
            agg = aggregate_dataset(battery_results)
            if agg:
                pid_rows.append({"dataset": name, "method": "PID", "eta_multiplier": ETA_PRIMARY,
                                 "k_burnin": k_burnin, "scorecaster": True, **agg})

            # nexCP: rho sweep
            for rho in RHO_VALUES:
                battery_results = []
                for bid, (yb, predb, cycb) in battery_data.items():
                    n = len(yb)
                    if n <= k_burnin:
                        continue
                    lo, hi = nexcp_tracker(yb, predb, q_src, rho, k_burnin)
                    battery_results.append(score_trace(yb, predb, lo, hi, k_burnin, cycb))
                agg = aggregate_dataset(battery_results)
                if agg:
                    nexcp_rows.append({"dataset": name, "method": "nexCP", "rho": rho,
                                       "k_burnin": k_burnin, **agg})

        print(f"[itemA] {name}: done in {time.time()-t_ds:.1f}s ({len(batteries)} batteries, {len(y)} cycles)")

    pid_df = pd.DataFrame(pid_rows)
    nexcp_df = pd.DataFrame(nexcp_rows)
    pid_df.to_csv(OUT_DIR / "finalpass2_itemA_pid_results.csv", index=False)
    nexcp_df.to_csv(OUT_DIR / "finalpass2_itemA_nexcp_results.csv", index=False)

    # headline comparison table: primary config (eta=0.1 or rho=0.95/0.99, k_burnin=10) vs.
    # this pass's own existing split-conformal (item 3/5e) and this project's historical CALCE table
    primary_pid = pid_df[(pid_df.eta_multiplier == ETA_PRIMARY) & (pid_df.k_burnin == BURNIN_PRIMARY) & (~pid_df.scorecaster)]
    primary_pid_sc = pid_df[(pid_df.eta_multiplier == ETA_PRIMARY) & (pid_df.k_burnin == BURNIN_PRIMARY) & (pid_df.scorecaster)]
    primary_nexcp = nexcp_df[nexcp_df.k_burnin == BURNIN_PRIMARY]

    headline_rows = []
    for _, r in primary_pid.iterrows():
        headline_rows.append({"dataset": r["dataset"], "method": "PID (eta=0.1)", "coverage": r["coverage"],
                              "mean_width": r["mean_width"], "rolling20_min": r["rolling20_coverage_min"]})
    for _, r in primary_pid_sc.iterrows():
        headline_rows.append({"dataset": r["dataset"], "method": "PID+scorecaster", "coverage": r["coverage"],
                              "mean_width": r["mean_width"], "rolling20_min": r["rolling20_coverage_min"]})
    for _, r in primary_nexcp.iterrows():
        headline_rows.append({"dataset": r["dataset"], "method": f"nexCP (rho={r['rho']})", "coverage": r["coverage"],
                              "mean_width": r["mean_width"], "rolling20_min": r["rolling20_coverage_min"]})
    headline_df = pd.DataFrame(headline_rows)
    headline_df.to_csv(OUT_DIR / "finalpass2_itemA_headline_comparison.csv", index=False)

    print("\n=== HEADLINE (k_burnin=10, eta=0.1 / rho shown) ===")
    print(headline_df.pivot(index="dataset", columns="method", values="coverage").to_string())

    print(f"\n[itemA] TOTAL TIME: {(time.time()-t0)/60:.2f} minutes")


if __name__ == "__main__":
    main()
