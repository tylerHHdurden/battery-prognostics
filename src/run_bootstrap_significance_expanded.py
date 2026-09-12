"""
Dataset Expansion Phase 1, Step 4: re-runs session 21's bootstrap
significance testing (cycle-level AND battery-level) on the expanded
pool's test set - the direct test of whether more test batteries
(originally 6, now however many the expanded battery_level_split
produces) genuinely improves statistical power, i.e. whether previously
NOT-battery-level-significant findings (XGBoost's edge over VLSTM,
lean-vs-full's accuracy "edge") become significant now.

Mirrors run_bootstrap_significance.py's exact method (same N_BOOTSTRAP,
CI, seed, both resampling schemes) - additive, does not touch the
original script or its outputs.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))

from run_drop_branch_ablation_5branch_expanded import load_merged_fusion_5branch_expanded, BASE_COLS_5

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
PRED_DIR = PROC_DIR / "predictions"
OUT_DIR = ROOT / "outputs"

N_BOOTSTRAP = 2000
CI = (2.5, 97.5)
SEED = 42


def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def cycle_bootstrap_indices(n, n_boot=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n, size=n) for _ in range(n_boot)]


def battery_bootstrap_row_indices(battery_ids, n_boot=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed + 1)
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
    verdict = "SIGNIFICANT (CI excludes 0)" if significant else "NOT significant (CI includes 0)"
    print(f"    {label}: mean_delta={mean:+.5f}  95% CI=[{lo:+.5f}, {hi:+.5f}]  -> {verdict}")
    return {"label": label, "mean_delta": mean, "ci_lo": lo, "ci_hi": hi, "significant": significant}


def bootstrap_drop_branch():
    print("\n[bootstrap-exp] === 1. Drop-branch ablation (5-branch, expanded pool) ===")
    train_df, fusion_cols = load_merged_fusion_5branch_expanded("train")
    test_df, _ = load_merged_fusion_5branch_expanded("test")
    y_test = test_df["SOH"].to_numpy()
    battery_ids = test_df["battery_id"].to_numpy()
    n = len(test_df)
    n_test_batteries = len(set(battery_ids))
    print(f"[bootstrap-exp] test set: {n} cycles, {n_test_batteries} batteries "
          f"(original run had 6 test batteries)")

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
        print(f"\n[bootstrap-exp]  branch: {dropped}")
        cycle_deltas = np.array([
            r2(y_test[idx], pred_full[idx]) - r2(y_test[idx], pred_d[idx]) for idx in cycle_idx_sets
        ])
        r_cycle = summarize_ci(cycle_deltas, f"{dropped} CYCLE-level (n_cycles={n})")
        batt_deltas = np.array([
            r2(y_test[idx], pred_full[idx]) - r2(y_test[idx], pred_d[idx]) for idx in batt_idx_sets
        ])
        r_batt = summarize_ci(batt_deltas, f"{dropped} BATTERY-level (n_batteries={n_test_batteries})")
        results.append({"branch": dropped, **{f"cycle_{k}": v for k, v in r_cycle.items()},
                         **{f"battery_{k}": v for k, v in r_batt.items()}})

    pd.DataFrame(results).to_csv(PRED_DIR / "bootstrap_drop_branch_expanded_ci.csv", index=False)
    return results, n_test_batteries


def bootstrap_base_learners():
    print("\n[bootstrap-exp] === 2. Base learner comparison (expanded pool) ===")
    xgb = pd.read_csv(PRED_DIR / "xgb_expanded_preds.csv")
    xgb = xgb[xgb["split"] == "test"][["dataset", "battery_id", "cycle_idx", "SOH", "y_pred_soh"]]
    deep = pd.read_csv(PRED_DIR / "deep_models_expanded_test_preds.csv")
    bigru = pd.read_csv(PRED_DIR / "cnn_bigru_expanded_test_preds.csv")[
        ["dataset", "battery_id", "cycle_idx", "y_pred_CNNBiGRU"]]

    merged = xgb.merge(deep[["dataset", "battery_id", "cycle_idx", "y_pred_VLSTM", "y_pred_CNNLSTM", "y_pred_PiFormer"]],
                        on=["dataset", "battery_id", "cycle_idx"], how="inner")
    merged = merged.merge(bigru, on=["dataset", "battery_id", "cycle_idx"], how="inner")
    print(f"[bootstrap-exp] shared test rows across all 5 base learners: {len(merged)}")

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

    print("\n[bootstrap-exp] per-model R2 CI:")
    r2_results = []
    for name, pred in models.items():
        cycle_r2 = np.array([r2(y_test[idx], pred[idx]) for idx in cycle_idx_sets])
        lo, hi = np.percentile(cycle_r2, CI)
        print(f"    {name}: point R2={r2(y_test, pred):.4f}  95% CI (cycle-level)=[{lo:.4f}, {hi:.4f}]")
        r2_results.append({"model": name, "point_r2": r2(y_test, pred), "ci_lo": lo, "ci_hi": hi})

    print("\n[bootstrap-exp] pairwise delta R2 vs. XGBoost:")
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

    pd.DataFrame(r2_results).to_csv(PRED_DIR / "bootstrap_base_learner_expanded_r2_ci.csv", index=False)
    pd.DataFrame(delta_results).to_csv(PRED_DIR / "bootstrap_base_learner_expanded_delta_vs_xgb_ci.csv", index=False)
    return r2_results, delta_results


def bootstrap_lean_vs_full():
    print("\n[bootstrap-exp] === 3. Lean vs. full (expanded pool) ===")
    with open(PROC_DIR / "bfa_selected_features_expanded.txt") as f:
        BFA_SELECTED = [l.strip() for l in f if l.strip()]
    fusion_cols = [f"fusion_{i}" for i in range(16)]

    hi_df = pd.read_parquet(PROC_DIR / "hi_table_expanded.parquet")
    hi_df = hi_df[hi_df["dataset"].isin(["NASA", "MIT"])].reset_index(drop=True)
    fusion_df = pd.read_csv(PROC_DIR / "fusion_embeddings_expanded.csv")
    merged = pd.merge(hi_df, fusion_df, on=["dataset", "battery_id", "cycle_idx"], how="inner")

    split = json.loads((PROC_DIR / "battery_split_expanded.json").read_text())
    test_mask = merged["battery_id"].isin(split["test_ids"]).to_numpy()

    feature_cols = BFA_SELECTED + fusion_cols
    X = merged[feature_cols].to_numpy(dtype=float, copy=True)
    X[np.isinf(X)] = np.nan  # see run_bfa_expanded.py's comment: one MIT/b2c30 cycle's VDEDT is inf
    col_medians = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_medians, inds[1])

    xgb_fusion = XGBRegressor()
    xgb_fusion.load_model(str(ROOT / "models" / "xgb_soh_fusion_expanded.json"))
    pred_lean = xgb_fusion.predict(X[test_mask])

    train_df5, fusion_cols5 = load_merged_fusion_5branch_expanded("train")
    test_df5, _ = load_merged_fusion_5branch_expanded("test")
    full_cols = BASE_COLS_5 + fusion_cols5
    ridge_full = Ridge(alpha=1.0).fit(train_df5[full_cols], train_df5["SOH"])
    pred_full = ridge_full.predict(test_df5[full_cols])

    lean_keys = merged.loc[test_mask, ["dataset", "battery_id", "cycle_idx"]].reset_index(drop=True)
    full_keys = test_df5[["dataset", "battery_id", "cycle_idx"]].reset_index(drop=True)
    aligned = lean_keys.equals(full_keys)
    print(f"[bootstrap-exp] LEAN/FULL test-row key alignment identical: {aligned}")
    if not aligned:
        lean_df = merged.loc[test_mask, ["dataset", "battery_id", "cycle_idx", "SOH"]].reset_index(drop=True)
        lean_df["pred_lean"] = pred_lean
        full_df = test_df5[["dataset", "battery_id", "cycle_idx", "SOH"]].reset_index(drop=True)
        full_df["pred_full"] = pred_full
        joined = lean_df.merge(full_df, on=["dataset", "battery_id", "cycle_idx", "SOH"], how="inner")
        print(f"[bootstrap-exp] re-aligned via explicit merge: {len(joined)} rows")
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

    print("\n[bootstrap-exp] delta RMSE (lean - full):")
    cycle_deltas_rmse = np.array([
        rmse(y_test[idx], pred_lean[idx]) - rmse(y_test[idx], pred_full[idx]) for idx in cycle_idx_sets
    ])
    r_rmse_cycle = summarize_ci(cycle_deltas_rmse, "delta_RMSE CYCLE-level")
    batt_deltas_rmse = np.array([
        rmse(y_test[idx], pred_lean[idx]) - rmse(y_test[idx], pred_full[idx]) for idx in batt_idx_sets
    ])
    r_rmse_batt = summarize_ci(batt_deltas_rmse, "delta_RMSE BATTERY-level")

    print("\n[bootstrap-exp] delta R2 (lean - full):")
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
    out.to_csv(PRED_DIR / "bootstrap_lean_vs_full_expanded_ci.csv", index=False)
    return out


def main():
    print(f"[bootstrap-exp] N_BOOTSTRAP={N_BOOTSTRAP}, CI={CI}, seed={SEED}")
    _, n_test_batteries = bootstrap_drop_branch()
    bootstrap_base_learners()
    bootstrap_lean_vs_full()
    print(f"\n[bootstrap-exp] DONE. n_test_batteries={n_test_batteries} "
          f"(original run had 6) - see data/processed/predictions/bootstrap_*_expanded_ci.csv")


if __name__ == "__main__":
    main()
