"""
Research pass 2, item 4: diversity-forced bagged XGBoost ensemble. The
project's own long-standing "stacking does nothing" finding
(Stacking-Ridge-fusion TEST R2=0.9169 vs. standalone XGBoost-fusion's
own 0.9172 on that same, earlier pre-reformulation feature vintage -
data/processed/predictions/ensemble_fusion_metrics.csv /
xgb_fusion_preds.csv, read back here for historical reference only, NOT
used as this item's baseline since it predates the current
reformulated-feature deployed model) has always been diagnosed as "the
4 base learners (XGBoost/VLSTM/CNN-LSTM/PiFormer) aren't diverse
enough" - but every fix attempt so far changed ARCHITECTURE, never
tested diversity within a single model type. This item does that:
trains N=8 XGBoost members, each forced to differ from the others via
(a) a bootstrap resample of TRAIN rows (with replacement, same size as
the full train set) and (b) a random ~70% subset of the 8 canonical HI
feature columns (the 16-dim fusion embedding is kept identical across
all members - it's a single coherent learned representation, not a set
of independent hand-engineered features, so subsetting it arbitrarily
doesn't have the same diversity motivation - disclosed design choice).

BASELINE: the CURRENT deployed XGBoost-fusion model (models/
xgb_soh_fusion.json), evaluated on the SAME feature pipeline/eval
protocol as this pass's other items (stage1_common, reformulated
features) - in-domain TEST R2=0.973 on record (DEPLOYED_REFERENCE,
matches run_stage7_2_selfsupervised_pretrain.py's own reference row).

TWO combination methods reported:
  - simple average across the 8 bagged members (the standard bagging
    combination rule)
  - Ridge-stacked on genuine OUT-OF-BAG predictions per training row
    (each bootstrap draw leaves ~37% of rows out; a row's OOB
    prediction pools only the members that did NOT see it during
    training - avoiding the exact in-sample-leakage problem this
    project's own conformal-calibration work has been careful about
    elsewhere) - the "correct" way to stack on top of bagged members
    without needing extra held-out data.

Evaluated on in-domain TEST and all 4 held-out datasets (zero-retrain),
same protocol as every other item.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_common import (
    load_nasa_mit_pool, battery_split_masks, fusion_cols, canonical_feature_cols,
    build_calce_merged, OUT_DIR, PROC_DIR, ROOT,
)

SEED = 42
N_MEMBERS = 8
FEATURE_SUBSET_FRAC = 0.7
XGB_KWARGS = dict(n_estimators=500, max_depth=6, learning_rate=0.03,
                   subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1)

DEPLOYED_REFERENCE = {
    "in-domain (TEST)": 0.973, "CALCE": 0.740, "Oxford": 0.953, "HUST": 0.800, "XJTU": -1.775,
}


def impute(X, medians):
    X = np.where(np.isinf(X), np.nan, X)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(medians, inds[1])
    return X


def main():
    t0 = time.time()
    print("=== Research pass 2, item 4: diversity-forced bagged XGBoost ensemble ===")
    rng = np.random.default_rng(SEED)

    merged, hi_full = load_nasa_mit_pool(reformulated=True)
    train_mask, test_mask, split = battery_split_masks(merged)
    base_feat_cols = canonical_feature_cols(reformulated=True) + ["cycle_idx"]
    fcols = fusion_cols()
    all_cols = base_feat_cols + fcols

    X_all = merged[all_cols].to_numpy(dtype=float, copy=True)
    train_medians = np.nanmedian(np.where(np.isinf(X_all[train_mask]), np.nan, X_all[train_mask]), axis=0)
    X_all = impute(X_all, train_medians)
    y_all = merged["SOH"].to_numpy(dtype=float)

    train_idx = np.where(train_mask)[0]
    n_train = len(train_idx)
    n_feat_subset = max(3, int(round(FEATURE_SUBSET_FRAC * len(base_feat_cols))))

    print(f"[bagged-xgb] training {N_MEMBERS} members: bootstrap rows (n={n_train} each) + "
          f"{n_feat_subset}/{len(base_feat_cols)} random HI features + all {len(fcols)} fusion dims")

    members = []
    oob_pred_sum = np.zeros(n_train)
    oob_pred_count = np.zeros(n_train)

    for m in range(N_MEMBERS):
        boot_pos = rng.integers(0, n_train, size=n_train)  # bootstrap positions INTO train_idx
        boot_rows = train_idx[boot_pos]
        oob_mask_pos = np.ones(n_train, dtype=bool)
        oob_mask_pos[np.unique(boot_pos)] = False  # positions never drawn this round
        oob_rows = train_idx[oob_mask_pos]

        feat_subset = list(rng.choice(base_feat_cols, size=n_feat_subset, replace=False))
        member_cols = feat_subset + fcols
        col_idx = [all_cols.index(c) for c in member_cols]

        t1 = time.time()
        model = XGBRegressor(random_state=SEED + m, **XGB_KWARGS)
        model.fit(X_all[boot_rows][:, col_idx], y_all[boot_rows])
        print(f"[bagged-xgb] member {m+1}/{N_MEMBERS} trained in {time.time()-t1:.1f}s "
              f"(features: {feat_subset}, oob_rows={len(oob_rows)})")
        members.append((model, col_idx))

        if len(oob_rows) > 0:
            oob_pred = model.predict(X_all[oob_rows][:, col_idx])
            oob_positions = np.where(oob_mask_pos)[0]
            oob_pred_sum[oob_positions] += oob_pred
            oob_pred_count[oob_positions] += 1

    # Ridge meta-learner trained on genuine OOB member predictions (rows with >=2 OOB members only)
    valid_oob_rows = oob_pred_count >= 2
    n_valid = int(valid_oob_rows.sum())
    print(f"[bagged-xgb] {n_valid}/{n_train} train rows have >=2 OOB member votes - using simple "
          f"average of those OOB votes as the meta-learner's TARGET-matched feature for stacking "
          f"(a genuine, if coarser, OOB signal - avoids the complexity of tracking which SPECIFIC "
          f"members were OOB per row for a full per-member design matrix)")
    oob_avg_pred = np.divide(oob_pred_sum, oob_pred_count, out=np.full(n_train, np.nan), where=oob_pred_count > 0)
    ridge = Ridge(alpha=1.0).fit(oob_avg_pred[valid_oob_rows].reshape(-1, 1), y_all[train_idx][valid_oob_rows])
    print(f"[bagged-xgb] Ridge stacker (1-D, on OOB-average): coef={ridge.coef_[0]:.4f} intercept={ridge.intercept_:.4f}")

    def evaluate(name, df):
        X = df[all_cols].to_numpy(dtype=float, copy=True)
        X = impute(X, train_medians)
        y = df["SOH"].to_numpy(dtype=float)
        member_preds = np.stack([model.predict(X[:, col_idx]) for model, col_idx in members], axis=1)
        avg_pred = member_preds.mean(axis=1)
        stacked_pred = ridge.predict(avg_pred.reshape(-1, 1))
        r2_avg = r2_score(y, avg_pred)
        rmse_avg = float(np.sqrt(mean_squared_error(y, avg_pred)))
        r2_stacked = r2_score(y, stacked_pred)
        rmse_stacked = float(np.sqrt(mean_squared_error(y, stacked_pred)))
        ref = DEPLOYED_REFERENCE.get(name)
        print(f"[bagged-xgb] {name}: bagged-avg R2={r2_avg:.4f} RMSE={rmse_avg:.4f} | "
              f"bagged-Ridge-stacked R2={r2_stacked:.4f} RMSE={rmse_stacked:.4f} | deployed reference R2={ref}")
        return {"eval_set": name, "bagged_avg_r2": r2_avg, "bagged_avg_rmse": rmse_avg,
                "bagged_ridge_stacked_r2": r2_stacked, "bagged_ridge_stacked_rmse": rmse_stacked,
                "deployed_reference_r2": ref}

    results = [evaluate("in-domain (TEST)", merged[test_mask])]
    calce_merged = build_calce_merged(hi_full)
    results.append(evaluate("CALCE", calce_merged))
    for name, fname in [("Oxford", "stage5_1_oxford_merged.parquet"),
                         ("HUST", "stage5_1_hust_merged.parquet"),
                         ("XJTU", "stage5_1_xjtu_merged.parquet")]:
        path = PROC_DIR / fname
        if not path.exists():
            print(f"[bagged-xgb] WARNING: {fname} not found, skipping {name}")
            continue
        df_held = pd.read_parquet(path)
        missing = [c for c in all_cols if c not in df_held.columns]
        if missing:
            print(f"[bagged-xgb] WARNING: {name} missing columns {missing}, skipping")
            continue
        results.append(evaluate(name, df_held))

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "researchpass2_item4_bagged_xgb_diversity.csv", index=False)
    print("\n=== SUMMARY ===")
    print(results_df.to_string(index=False))
    for r in results:
        ref = r["deployed_reference_r2"]
        best = max(r["bagged_avg_r2"], r["bagged_ridge_stacked_r2"])
        verdict = "WIN" if (ref is not None and best > ref) else "LOSS"
        print(f"[bagged-xgb] {r['eval_set']}: best bagged R2={best:.4f} vs deployed={ref} -> {verdict}")

    print(f"\n[bagged-xgb] TOTAL TIME: {(time.time()-t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
