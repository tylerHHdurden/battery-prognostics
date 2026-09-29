"""
Part A, item 3: mechanistically-guided residual learning, adapting the
IDEA of Che, Zheng, Rhyu, Guo, Wang, Teodorescu, Braatz, "Mechanistically
guided residual learning for battery state monitoring," Nature
Communications 17:855 (DOI 10.1038/s41467-025-67565-z - verified via web
search before writing this: published online Jan 2026, DOI carries a
2025 suffix consistent with a 2025 acceptance/submission, matching the
task's "Nature Communications 2025" description closely enough to cite
directly, disclosed here rather than silently smoothed over). CONFIRMED
BY DIRECT ABSTRACT-LEVEL SEARCH (nature.com itself returned an auth-wall
redirect to direct WebFetch, so verified via search-engine-indexed
abstract text instead, not fabricated): the paper covers BOTH real-time
SOC estimation (via a Kalman-filter/EKF state estimator with a
mechanistic ECM prior) AND SOH monitoring for EVs, using ML to learn a
RESIDUAL on top of the filter's own physics-based state estimate -
NOT purely an SOC-only paper as the task's framing suggested; that
nuance is disclosed rather than silently assumed away. This item
ADAPTS the residual-on-a-physics-prior IDEA to this project's own
tabular HI+fusion-feature SOH pipeline and its own governing zero-
retrain evaluation protocol - it does not reproduce the paper's own
ECM/EKF architecture, dataset, or results.

DISCLOSED, NECESSARY SIMPLIFICATION OF THE PHYSICS BASELINE: the
paper's own filter tracks an electrochemical (ECM) state from raw
voltage/current using an EKF. This project's tabular pipeline does not
retain per-cycle raw voltage/current in a form this item's scope allows
rebuilding an ECM from (that already exists as a SEPARATE, disclosed-
narrow physics surrogate in Stage 7.3's DeepONet work, a different
project effort) - and this project's SOH ground truth itself IS
directly derived from measured discharge capacity, so a physics
baseline that reads the current cycle's own raw capacity would nearly
reproduce the label directly, defeating the purpose of testing whether
RESIDUAL learning helps (the "baseline" would already explain ~100% of
variance). Instead, this item builds a genuine CAUSAL, one-step-ahead
constant-velocity KALMAN FILTER over each battery's own OBSERVED SOH
sequence (state = [level, fade-rate], a standard, physically-motivated
smooth-degradation-trend prior; F/Q/R below) - a real recursive
Bayesian filter (not a lookup or a smoother that peeks at the label it's
predicting): the reported baseline for cycle n is the filter's own
PREDICT step using ONLY cycles < n of that same battery's history (a
genuine forecast, no leakage of cycle n's own label); the label at
cycle n is only used AFTER prediction, in the filter's measurement-
UPDATE step, to advance the state for predicting cycle n+1. This is a
disclosed, simplified stand-in for the paper's own ECM+EKF SOC/SOH
state estimator, not a literal port of it.

RESIDUAL MODEL: XGBoost (identical hyperparameters to every other
XGBoost-fusion variant in this project, via stage1_common's own
convention) trained to predict r = SOH_true - physics_baseline using
the SAME HI+fusion feature columns the routed deployed model already
uses (base repr for in-domain/XJTU, extended repr for CALCE/Oxford/
HUST, per this project's own already-shipped routing) - final
prediction = physics_baseline + residual_model(features). No monotone
constraint applied to the residual model (disclosed reasoning: Stage
1.5's monotone constraint encodes "SOH itself is non-increasing in
cycle_idx" - the RESIDUAL against a already-trend-following physics
prior has no such guaranteed sign/monotonicity, so constraining it
would be an unjustified, untested assumption, not a neutral carry-
over).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from researchpass_partA_common import (
    load_base_pool_and_split, load_extended_pool_and_split,
    load_all_heldout_base, load_all_heldout_extended,
    base_feature_cols, extended_feature_cols, fusion_cols,
    score, verdict, print_and_save_results, EXTENDED_ROUTED_DATASETS,
    ROUTED_BASELINE, OUT_DIR,
)

SEED = 42
Q_LEVEL = 1e-3   # process noise, SOH-level state (%^2 per cycle) - disclosed, untuned, reasoned default
Q_RATE = 1e-5    # process noise, fade-rate state
R_MEAS = 0.25    # measurement noise variance (0.5%-SOH std) - disclosed, untuned, reasoned default


def kalman_causal_baseline(cycle_idx: np.ndarray, soh: np.ndarray) -> np.ndarray:
    """One battery, sorted by cycle_idx. Returns, for each cycle n, the
    filter's own PREDICT-step estimate using only cycles < n (a genuine
    causal one-step-ahead forecast - cycle n's own true SOH is never
    used to produce baseline[n], only to update the state afterward)."""
    n = len(soh)
    baseline = np.empty(n, dtype=float)
    baseline[0] = soh[0]  # no prior history at all - same "own early cycle" convention used project-wide
    x = np.array([soh[0], 0.0])
    P = np.array([[25.0, 0.0], [0.0, 1.0]])
    H = np.array([1.0, 0.0])
    for i in range(1, n):
        dt = max(float(cycle_idx[i] - cycle_idx[i - 1]), 1e-6)
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = np.array([[Q_LEVEL * dt, 0.0], [0.0, Q_RATE * dt]])
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        baseline[i] = x_pred[0]  # PREDICT-only, reported before any use of soh[i]

        z = soh[i]
        S = float(H @ P_pred @ H.T + R_MEAS)
        K = (P_pred @ H.T) / S
        x = x_pred + K * (z - H @ x_pred)
        P = (np.eye(2) - np.outer(K, H)) @ P_pred
    return baseline


def add_physics_baseline(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["battery_id", "cycle_idx"]).copy()
    out = np.empty(len(df), dtype=float)
    for bid, g in df.groupby("battery_id", sort=False):
        idx = g.index.to_numpy()
        b = kalman_causal_baseline(g["cycle_idx"].to_numpy(dtype=float), g["SOH"].to_numpy(dtype=float))
        out[df.index.get_indexer(idx)] = b
    df["physics_baseline"] = out
    df["residual_target"] = df["SOH"] - df["physics_baseline"]
    return df


def fit_residual_xgb(merged: pd.DataFrame, train_mask: np.ndarray, feature_cols: list[str]):
    cols = feature_cols + fusion_cols()
    X = merged[cols].to_numpy(dtype=float, copy=True)
    X = np.where(np.isinf(X), np.nan, X)
    col_medians = np.nanmedian(X[train_mask], axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_medians, inds[1])
    y = merged["residual_target"].to_numpy(dtype=float)

    model = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03, subsample=0.8,
                          colsample_bytree=0.8, random_state=SEED, n_jobs=-1, reg_lambda=1.0)
    model.fit(X[train_mask], y[train_mask])
    return model, col_medians, cols


def predict_final(model, medians, cols, df: pd.DataFrame) -> np.ndarray:
    X = df[cols].to_numpy(dtype=float, copy=True)
    X = np.where(np.isinf(X), np.nan, X)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(medians, inds[1])
    resid_pred = model.predict(X)
    return df["physics_baseline"].to_numpy(dtype=float) + resid_pred


def main():
    t0 = time.time()
    print("=== Part A, item 3: mechanistically-guided residual learning (Kalman physics prior + XGBoost residual) ===")

    merged_base, hi_full_base, train_mask_b, test_mask_b, _ = load_base_pool_and_split()
    merged_ext, hi_full_ext, hi_full_raw, train_mask_e, test_mask_e, _ = load_extended_pool_and_split()
    base_cols = base_feature_cols()
    ext_cols = extended_feature_cols()

    print("[item3] computing causal Kalman physics baseline for TRAIN+TEST pool (base repr)...")
    merged_base = add_physics_baseline(merged_base)
    print("[item3] computing causal Kalman physics baseline for TRAIN+TEST pool (extended repr)...")
    merged_ext = add_physics_baseline(merged_ext)
    # re-derive masks after sort_values inside add_physics_baseline reordered rows
    from stage1_common import battery_split_masks
    train_mask_b, test_mask_b, _ = battery_split_masks(merged_base)
    train_mask_e, test_mask_e, _ = battery_split_masks(merged_ext)

    physics_only_r2_in = float(np.corrcoef(merged_base.loc[test_mask_b, "SOH"],
                                            merged_base.loc[test_mask_b, "physics_baseline"])[0, 1] ** 2)
    print(f"[item3] sanity check - physics-ONLY baseline in-domain TEST R2 (rough, corr^2): {physics_only_r2_in:.4f} "
          f"(expected far below the full model's ~0.97 - this is a deliberately weak, causal, label-free-at-"
          f"predict-time forecast, not a strong baseline on its own; the point is what the RESIDUAL model adds)")

    print("\n[item3] fitting residual-model, base representation (in-domain/XJTU routing)...")
    model_base, med_base, cols_base = fit_residual_xgb(merged_base, train_mask_b, base_cols)
    print("[item3] fitting residual-model, extended representation (CALCE/Oxford/HUST routing)...")
    model_ext, med_ext, cols_ext = fit_residual_xgb(merged_ext, train_mask_e, ext_cols)

    held_base = load_all_heldout_base(hi_full_base)
    held_ext = load_all_heldout_extended(hi_full_raw)
    for name in held_base:
        held_base[name] = add_physics_baseline(held_base[name])
    for name in held_ext:
        held_ext[name] = add_physics_baseline(held_ext[name])

    results = []
    pred_in = predict_final(model_base, med_base, cols_base, merged_base.loc[test_mask_b])
    y_in = merged_base.loc[test_mask_b, "SOH"].to_numpy(dtype=float)
    r_in = score(y_in, pred_in)
    v_in = verdict(r_in["r2"], "in-domain (fixed split)")
    print(f"[item3] in-domain (fixed split): R2={r_in['r2']:.4f} RMSE={r_in['rmse']:.4f} n={r_in['n']} "
          f"vs routed baseline {ROUTED_BASELINE['in-domain (fixed split)']:.3f} -> {v_in}")
    results.append({"dataset": "in-domain (fixed split)", **r_in,
                     "routed_baseline_r2": ROUTED_BASELINE["in-domain (fixed split)"], "verdict": v_in})

    for name in ["CALCE", "Oxford", "HUST", "XJTU"]:
        use_ext = name in EXTENDED_ROUTED_DATASETS
        df = held_ext[name] if use_ext else held_base[name]
        model, med, cols = (model_ext, med_ext, cols_ext) if use_ext else (model_base, med_base, cols_base)
        pred = predict_final(model, med, cols, df)
        y_true = df["SOH"].to_numpy(dtype=float)
        r = score(y_true, pred)
        v = verdict(r["r2"], name)
        print(f"[item3] {name} (routed repr: {'extended' if use_ext else 'base'}): "
              f"R2={r['r2']:.4f} RMSE={r['rmse']:.4f} n={r['n']} "
              f"vs routed baseline {ROUTED_BASELINE[name]:.3f} -> {v}")
        results.append({"dataset": name, **r, "routed_baseline_r2": ROUTED_BASELINE[name], "verdict": v})

    results_df = print_and_save_results(results, "researchpass_partA_item3_mechanistic_residual.csv",
                                         "Item 3 (mechanistically-guided residual learning)")
    n_wins = int((results_df["verdict"] == "WIN").sum())
    print(f"\n[item3] {n_wins}/5 eval settings beat the routed deployed baseline (RAW, as-built).")

    # CRITICAL DIAGNOSTIC, not a cosmetic add-on: the headline numbers
    # above give this item's Kalman baseline CAUSAL access to each
    # target battery's OWN past true SOH labels, cycle by cycle - an
    # input assumption the DEPLOYED model's task never makes (it scores
    # a single cycle's own features, with no per-battery label history
    # at all). That is a fundamentally different, easier task, not an
    # apples-to-apples "beats deployed baseline" comparison, if most of
    # the apparent gain just comes from having that history rather than
    # from the physics model or the ML residual specifically. Checked
    # directly, not assumed: a trivial PERSISTENCE baseline (predict
    # SOH(n) = SOH(n-1), the simplest possible use of that SAME past-
    # label access, no physics, no ML at all) on the identical eval sets.
    print("\n[item3] DIAGNOSTIC - persistence baseline (SOH(n)=SOH(n-1), same past-label access, "
          "no physics/no ML) - isolates how much of the win above is just from having that access:")
    diag_rows = []
    for name, df, test_mask, merged in [
        ("in-domain (fixed split)", merged_base.loc[test_mask_b], None, None),
        ("CALCE", held_ext["CALCE"], None, None),
        ("Oxford", held_ext["Oxford"], None, None),
        ("HUST", held_ext["HUST"], None, None),
        ("XJTU", held_base["XJTU"], None, None),
    ]:
        d = df.sort_values(["battery_id", "cycle_idx"])
        persisted = d.groupby("battery_id")["SOH"].shift(1)
        persisted = persisted.fillna(d["SOH"])  # first cycle per battery: no prior, use itself (0 error, matches baseline[0] convention)
        r = score(d["SOH"].to_numpy(dtype=float), persisted.to_numpy(dtype=float))
        this_r2 = results_df.loc[results_df["dataset"] == name, "r2"].iloc[0]
        print(f"[item3]   {name}: persistence-only R2={r['r2']:.4f}  |  this item's full result R2={this_r2:.4f}  "
              f"|  routed deployed baseline R2={ROUTED_BASELINE[name]:.3f}")
        diag_rows.append({"dataset": name, "persistence_only_r2": r["r2"], "item3_full_r2": this_r2,
                           "routed_baseline_r2": ROUTED_BASELINE[name]})
    diag_df = pd.DataFrame(diag_rows)
    diag_df.to_csv(OUT_DIR / "researchpass_partA_item3_persistence_diagnostic.csv", index=False)
    print(diag_df.to_string(index=False))
    print(f"\n[item3] TOTAL TIME: {(time.time()-t0)/60:.2f} minutes")


if __name__ == "__main__":
    main()
