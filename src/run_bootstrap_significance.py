"""
Session 21: bootstrap confidence intervals on this project's key
comparison results, to test whether observed differences are
statistically real or within noise at this sample size. Pure analysis
on already-computed predictions - no retraining, no existing file
touched (session 17/20's ablation/lean-vs-full scripts are re-used for
their exact merge logic via import, not modified).

Two bootstrap schemes are reported for every comparison, not just one -
this is NOT part of the original ask, but is added because it's a
real, honesty-relevant methodological issue this project has already
flagged elsewhere (session 11's RUL-coverage battery-count caveat,
session 19's small-battery-count domain-classifier caveat):

  - CYCLE-level bootstrap (the literal request): resample individual
    test-set CYCLES with replacement. This is what's reported as the
    headline CI below, exactly as asked.
  - BATTERY-level (cluster) bootstrap, reported alongside as an honest
    robustness check: resample whole TEST BATTERIES with replacement,
    keeping every cycle from a chosen battery together. The NASA+MIT
    test set is only 6 batteries with hundreds of highly-autocorrelated
    cycles each (a battery's SOH trajectory is smooth - neighboring
    cycles are far from independent draws). A cycle-level bootstrap
    treats ~5208 pseudo-independent draws as if they were genuinely
    independent, which understates true uncertainty; a battery-level
    bootstrap is more honest about the REAL effective sample size (6),
    at the cost of much wider, chunkier intervals (only 2^6-1=63 distinct
    nonempty battery subsets exist, before even considering
    with-replacement duplicates). Both are reported so the reader can
    see how much the conclusion depends on which unit of resampling is
    treated as "independent".
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))

from run_drop_branch_ablation_5branch import load_merged_fusion_5branch, BASE_COLS_5

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
PRED_DIR = PROC_DIR / "predictions"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

N_BOOTSTRAP = 2000
CI = (2.5, 97.5)  # 95% percentile CI
SEED = 42


def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def cycle_bootstrap_indices(n, n_boot=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n, size=n) for _ in range(n_boot)]


def battery_bootstrap_row_indices(battery_ids, n_boot=N_BOOTSTRAP, seed=SEED):
    """Resamples whole BATTERIES with replacement, then returns the
    concatenated row-index arrays for each resample (a battery chosen
    twice contributes its rows twice - the correct way to bootstrap
    clustered/correlated data, Efron & Tibshirani 1993 sec. 8.6)."""
    rng = np.random.default_rng(seed + 1)  # different stream from cycle-level, not reused
    unique_batteries = np.array(sorted(set(battery_ids)))
    battery_ids = np.asarray(battery_ids)
    row_idx_by_battery = {b: np.where(battery_ids == b)[0] for b in unique_batteries}
    n_batteries = len(unique_batteries)
    out = []
    for _ in range(n_boot):
        chosen = rng.choice(unique_batteries, size=n_batteries, replace=True)
        out.append(np.concatenate([row_idx_by_battery[b] for b in chosen]))
    return out


def summarize_ci(deltas, label):
    lo, hi = np.percentile(deltas, CI)
    mean = np.mean(deltas)
    significant = not (lo <= 0 <= hi)
    verdict = "SIGNIFICANT (CI excludes 0)" if significant else "NOT significant (CI includes 0 - indistinguishable from noise)"
    print(f"    {label}: mean_delta={mean:+.5f}  95% CI=[{lo:+.5f}, {hi:+.5f}]  -> {verdict}")
    return {"label": label, "mean_delta": mean, "ci_lo": lo, "ci_hi": hi, "significant": significant}


# ==========================================================================
# 1. Drop-branch ablation (5-branch, incl. CNN-BiGRU): CI on delta R2 for
#    dropping each branch vs. the full ensemble.
# ==========================================================================

def bootstrap_drop_branch():
    print("\n[bootstrap] === 1. Drop-branch ablation (5-branch) - CI on delta R2 per dropped branch ===")
    train_df, fusion_cols = load_merged_fusion_5branch("train")
    test_df, _ = load_merged_fusion_5branch("test")
    y_test = test_df["SOH"].to_numpy()
    battery_ids = test_df["battery_id"].to_numpy()
    n = len(test_df)

    full_cols = BASE_COLS_5 + fusion_cols
    ridge_full = Ridge(alpha=1.0).fit(train_df[full_cols], train_df["SOH"])
    pred_full = ridge_full.predict(test_df[full_cols])

    pred_drop = {}
    for dropped in BASE_COLS_5:
        cols = [c for c in BASE_COLS_5 if c != dropped] + fusion_cols
        ridge = Ridge(alpha=1.0).fit(train_df[cols], train_df["SOH"])
        pred_drop[dropped] = ridge.predict(test_df[cols])

    cycle_idx_sets = cycle_bootstrap_indices(n)
    batt_idx_sets = battery_bootstrap_row_indices(battery_ids)

    results = []
    for dropped, pred_d in pred_drop.items():
        print(f"\n[bootstrap]  branch: {dropped}")
        cycle_deltas = np.array([
            r2(y_test[idx], pred_full[idx]) - r2(y_test[idx], pred_d[idx]) for idx in cycle_idx_sets
        ])
        r_cycle = summarize_ci(cycle_deltas, f"{dropped} CYCLE-level (n_cycles={n})")
        batt_deltas = np.array([
            r2(y_test[idx], pred_full[idx]) - r2(y_test[idx], pred_d[idx]) for idx in batt_idx_sets
        ])
        r_batt = summarize_ci(batt_deltas, f"{dropped} BATTERY-level (n_batteries={len(set(battery_ids))})")
        results.append({"branch": dropped, **{f"cycle_{k}": v for k, v in r_cycle.items()},
                         **{f"battery_{k}": v for k, v in r_batt.items()}})

    pd.DataFrame(results).to_csv(PRED_DIR / "bootstrap_drop_branch_ci.csv", index=False)
    return results


# ==========================================================================
# 2. Base learner comparison: CI on R2 for each of the 5 base learners
#    (non-fusion XGBoost, VLSTM, CNN-LSTM, PiFormer, CNN-BiGRU), on the
#    SAME shared test rows for a fair paired comparison.
# ==========================================================================

def bootstrap_base_learners():
    print("\n[bootstrap] === 2. Base learner comparison - CI on R2 per model, and pairwise delta vs. XGBoost ===")
    xgb = pd.read_csv(PRED_DIR / "xgb_preds.csv")
    xgb = xgb[xgb["split"] == "test"][["dataset", "battery_id", "cycle_idx", "SOH", "y_pred_soh"]]
    deep = pd.read_csv(PRED_DIR / "deep_models_test_preds.csv")
    bigru = pd.read_csv(PRED_DIR / "cnn_bigru_test_preds.csv")[["dataset", "battery_id", "cycle_idx", "y_pred_CNNBiGRU"]]

    merged = xgb.merge(deep[["dataset", "battery_id", "cycle_idx", "y_pred_VLSTM", "y_pred_CNNLSTM", "y_pred_PiFormer"]],
                        on=["dataset", "battery_id", "cycle_idx"], how="inner")
    merged = merged.merge(bigru, on=["dataset", "battery_id", "cycle_idx"], how="inner")
    print(f"[bootstrap] shared test rows across all 5 base learners: {len(merged)} "
          f"(xgb={len(xgb)}, deep={len(deep)}, bigru={len(bigru)})")

    y_test = merged["SOH"].to_numpy()
    battery_ids = merged["battery_id"].to_numpy()
    n = len(merged)
    models = {
        "XGBoost": merged["y_pred_soh"].to_numpy(),
        "VLSTM": merged["y_pred_VLSTM"].to_numpy(),
        "CNNLSTM": merged["y_pred_CNNLSTM"].to_numpy(),
        "PiFormer": merged["y_pred_PiFormer"].to_numpy(),
        "CNNBiGRU": merged["y_pred_CNNBiGRU"].to_numpy(),
    }

    cycle_idx_sets = cycle_bootstrap_indices(n)
    batt_idx_sets = battery_bootstrap_row_indices(battery_ids)

    print("\n[bootstrap] per-model R2 CI:")
    r2_results = []
    for name, pred in models.items():
        cycle_r2 = np.array([r2(y_test[idx], pred[idx]) for idx in cycle_idx_sets])
        lo, hi = np.percentile(cycle_r2, CI)
        print(f"    {name}: point R2={r2(y_test, pred):.4f}  95% CI (cycle-level)=[{lo:.4f}, {hi:.4f}]")
        r2_results.append({"model": name, "point_r2": r2(y_test, pred), "ci_lo": lo, "ci_hi": hi})

    print("\n[bootstrap] pairwise delta R2 vs. XGBoost (does XGBoost's dominance survive resampling?):")
    pred_xgb = models["XGBoost"]
    delta_results = []
    for name, pred in models.items():
        if name == "XGBoost":
            continue
        cycle_deltas = np.array([
            r2(y_test[idx], pred_xgb[idx]) - r2(y_test[idx], pred[idx]) for idx in cycle_idx_sets
        ])
        r_cycle = summarize_ci(cycle_deltas, f"XGBoost - {name} CYCLE-level")
        batt_deltas = np.array([
            r2(y_test[idx], pred_xgb[idx]) - r2(y_test[idx], pred[idx]) for idx in batt_idx_sets
        ])
        r_batt = summarize_ci(batt_deltas, f"XGBoost - {name} BATTERY-level")
        delta_results.append({"vs_model": name, **{f"cycle_{k}": v for k, v in r_cycle.items()},
                               **{f"battery_{k}": v for k, v in r_batt.items()}})

    pd.DataFrame(r2_results).to_csv(PRED_DIR / "bootstrap_base_learner_r2_ci.csv", index=False)
    pd.DataFrame(delta_results).to_csv(PRED_DIR / "bootstrap_base_learner_delta_vs_xgb_ci.csv", index=False)
    return r2_results, delta_results


# ==========================================================================
# 3. Lean vs. full: CI on delta RMSE and delta R2.
# ==========================================================================

def bootstrap_lean_vs_full():
    print("\n[bootstrap] === 3. Lean vs. full - CI on delta RMSE/R2 ===")
    with open(PROC_DIR / "bfa_selected_features.txt") as f:
        BFA_SELECTED = [l.strip() for l in f if l.strip()]
    fusion_cols = [f"fusion_{i}" for i in range(16)]

    hi_df = pd.read_parquet(PROC_DIR / "hi_table.parquet")
    hi_df = hi_df[hi_df["dataset"].isin(["NASA", "MIT"])].reset_index(drop=True)
    fusion_df = pd.read_csv(PROC_DIR / "fusion_embeddings.csv")
    merged = pd.merge(hi_df, fusion_df, on=["dataset", "battery_id", "cycle_idx"], how="inner")

    split = json.loads((PROC_DIR / "battery_split.json").read_text())
    test_mask = merged["battery_id"].isin(split["test_ids"]).to_numpy()

    feature_cols = BFA_SELECTED + fusion_cols
    X = merged[feature_cols].to_numpy(dtype=float, copy=True)
    col_medians = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_medians, inds[1])

    xgb_fusion = XGBRegressor()
    xgb_fusion.load_model(str(ROOT / "models" / "xgb_soh_fusion.json"))
    pred_lean = xgb_fusion.predict(X[test_mask])

    train_df5, fusion_cols5 = load_merged_fusion_5branch("train")
    test_df5, _ = load_merged_fusion_5branch("test")
    full_cols = BASE_COLS_5 + fusion_cols5
    ridge_full = Ridge(alpha=1.0).fit(train_df5[full_cols], train_df5["SOH"])
    pred_full = ridge_full.predict(test_df5[full_cols])

    # LEAN's y_test (merged[test_mask]) and FULL's y_test (test_df5) both
    # derive from the same underlying test battery set, but via different
    # merge chains (hi_table+fusion_embeddings vs. xgb_fusion_preds+deep
    # models+cnn_bigru+fusion_embeddings) - confirm row-for-row identical
    # (dataset,battery_id,cycle_idx) alignment before treating them as a
    # paired sample, rather than assuming it.
    lean_keys = merged.loc[test_mask, ["dataset", "battery_id", "cycle_idx"]].reset_index(drop=True)
    full_keys = test_df5[["dataset", "battery_id", "cycle_idx"]].reset_index(drop=True)
    aligned = lean_keys.equals(full_keys)
    print(f"[bootstrap] LEAN/FULL test-row key alignment identical: {aligned}")
    if not aligned:
        # fall back to an explicit merge-based alignment rather than
        # assuming positional order matches
        lean_df = merged.loc[test_mask, ["dataset", "battery_id", "cycle_idx", "SOH"]].reset_index(drop=True)
        lean_df["pred_lean"] = pred_lean
        full_df = test_df5[["dataset", "battery_id", "cycle_idx", "SOH"]].reset_index(drop=True)
        full_df["pred_full"] = pred_full
        joined = lean_df.merge(full_df, on=["dataset", "battery_id", "cycle_idx", "SOH"], how="inner")
        print(f"[bootstrap] re-aligned via explicit merge: {len(joined)} rows")
        y_test = joined["SOH"].to_numpy()
        pred_lean = joined["pred_lean"].to_numpy()
        pred_full = joined["pred_full"].to_numpy()
        battery_ids = joined["battery_id"].to_numpy()
    else:
        y_test = test_df5["SOH"].to_numpy()
        battery_ids = test_df5["battery_id"].to_numpy()

    n = len(y_test)
    cycle_idx_sets = cycle_bootstrap_indices(n)
    batt_idx_sets = battery_bootstrap_row_indices(battery_ids)

    print("\n[bootstrap] delta RMSE (lean - full; negative = lean is more accurate):")
    cycle_deltas_rmse = np.array([
        rmse(y_test[idx], pred_lean[idx]) - rmse(y_test[idx], pred_full[idx]) for idx in cycle_idx_sets
    ])
    r_rmse_cycle = summarize_ci(cycle_deltas_rmse, "delta_RMSE CYCLE-level")
    batt_deltas_rmse = np.array([
        rmse(y_test[idx], pred_lean[idx]) - rmse(y_test[idx], pred_full[idx]) for idx in batt_idx_sets
    ])
    r_rmse_batt = summarize_ci(batt_deltas_rmse, "delta_RMSE BATTERY-level")

    print("\n[bootstrap] delta R2 (lean - full; positive = lean is more accurate):")
    cycle_deltas_r2 = np.array([
        r2(y_test[idx], pred_lean[idx]) - r2(y_test[idx], pred_full[idx]) for idx in cycle_idx_sets
    ])
    r_r2_cycle = summarize_ci(cycle_deltas_r2, "delta_R2 CYCLE-level")
    batt_deltas_r2 = np.array([
        r2(y_test[idx], pred_lean[idx]) - r2(y_test[idx], pred_full[idx]) for idx in batt_idx_sets
    ])
    r_r2_batt = summarize_ci(batt_deltas_r2, "delta_R2 BATTERY-level")

    out = pd.DataFrame([
        {"metric": "delta_rmse", **{f"cycle_{k}": v for k, v in r_rmse_cycle.items()},
         **{f"battery_{k}": v for k, v in r_rmse_batt.items()}},
        {"metric": "delta_r2", **{f"cycle_{k}": v for k, v in r_r2_cycle.items()},
         **{f"battery_{k}": v for k, v in r_r2_batt.items()}},
    ])
    out.to_csv(PRED_DIR / "bootstrap_lean_vs_full_ci.csv", index=False)
    return out


def main():
    print(f"[bootstrap] N_BOOTSTRAP={N_BOOTSTRAP}, CI={CI}, seed={SEED}")
    bootstrap_drop_branch()
    bootstrap_base_learners()
    bootstrap_lean_vs_full()
    print("\n[bootstrap] DONE - see data/processed/predictions/bootstrap_*_ci.csv")


if __name__ == "__main__":
    main()
