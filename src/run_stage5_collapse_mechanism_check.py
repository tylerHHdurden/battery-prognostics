"""
Post-Stage-5 verification, item 2: does the SAME mechanism (protocol-
encoded raw-duration features, session 27's finding) explain all four
zero-retrain collapses (CALCE + Stage 5.1's Oxford/HUST/XJTU), or do
the 3 new datasets collapse for a different reason?

Reuses the EXACT z-score formula from run_b0018_root_cause_analysis.py
(z = (target_mean - train_mean)/train_std, train=NASA+MIT+recovered)
and the exact domain-classifier-AUC method (pooled z-score standardize
-> LogisticRegression -> in-sample AUC vs. train), applied to the
CURRENT canonical (Stage 1.1-reformulated) 8-feature set, so results
are directly comparable to this project's own historical B0018/CALCE
numbers rather than a new, incomparable methodology.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_common import (
    canonical_feature_cols, load_nasa_mit_pool, battery_split_masks,
    fusion_cols, build_calce_merged, OUT_DIR, PROC_DIR,
)

FEATURE_COLS = canonical_feature_cols(reformulated=True)  # 8 canonical duration-reformulated HIs


def zscore_vs_train(train_df: pd.DataFrame, target_df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    # VDEDT can be +-inf (a documented, pre-existing issue - see
    # run_domain_classifier_sanity_check_expanded.py's own comment on
    # MIT/b2c30; found here to also affect at least one XJTU cycle,
    # same known root cause - a zero-diff discharge tail timestamp).
    # Same established fix: treat inf as missing, same as NaN.
    train_df = train_df.copy()
    target_df = target_df.copy()
    for col in cols:
        train_df[col] = train_df[col].replace([np.inf, -np.inf], np.nan)
        target_df[col] = target_df[col].replace([np.inf, -np.inf], np.nan)
    rows = []
    for col in cols:
        tr = train_df[col].dropna()
        tgt = target_df[col].dropna()
        mean, std = tr.mean(), tr.std() + 1e-8
        z = (tgt.mean() - mean) / std
        pct_below = (tr < tgt.mean()).mean() * 100
        rows.append({"feature": col, "train_mean": mean, "train_std": std,
                     "target_mean": tgt.mean(), "zscore": z, "train_pctile_of_target_mean": pct_below})
    return pd.DataFrame(rows).reindex(
        pd.DataFrame(rows)["zscore"].abs().sort_values(ascending=False).index)


def domain_auc_vs_train(X_train: np.ndarray, X_target: np.ndarray, seed: int = 42) -> float:
    # inf -> NaN first (VDEDT can be +-inf, see zscore_vs_train's comment),
    # then median-impute (TRAIN medians only) - same convention as fit_xgb
    # elsewhere in this project. CALCE has no Temperature column, so
    # MATD/MATC/MATDL are NaN for every CALCE cycle (documented,
    # pre-existing limitation, not new to this check).
    X_train = np.where(np.isinf(X_train), np.nan, X_train)
    X_target = np.where(np.isinf(X_target), np.nan, X_target)
    train_medians = np.nanmedian(X_train, axis=0)
    Xt = X_train.copy()
    inds = np.where(np.isnan(Xt))
    Xt[inds] = np.take(train_medians, inds[1])
    Xg = X_target.copy()
    inds2 = np.where(np.isnan(Xg))
    Xg[inds2] = np.take(train_medians, inds2[1])

    Xp = np.concatenate([Xt, Xg])
    mean, std = Xp.mean(0), Xp.std(0) + 1e-8
    Xp_z = (Xp - mean) / std
    y = np.concatenate([np.zeros(len(Xt)), np.ones(len(Xg))])
    clf = LogisticRegression(max_iter=2000, random_state=seed).fit(Xp_z, y)
    return float(roc_auc_score(y, clf.predict_proba(Xp_z)[:, 1]))


def main():
    print("=== Stage 5 collapse-mechanism check ===")
    merged_nm, hi_full = load_nasa_mit_pool(reformulated=True)
    train_mask, test_mask, split = battery_split_masks(merged_nm)
    train_df = merged_nm.loc[train_mask]
    print(f"[collapse-check] NASA+MIT+recovered TRAIN pool: {len(train_df)} cycles, "
          f"{train_df['battery_id'].nunique()} batteries")

    calce_merged = build_calce_merged(hi_full)
    datasets = {
        "CALCE": calce_merged,
        "Oxford": pd.read_parquet(PROC_DIR / "stage5_1_oxford_merged.parquet"),
        "HUST": pd.read_parquet(PROC_DIR / "stage5_1_hust_merged.parquet"),
        "XJTU": pd.read_parquet(PROC_DIR / "stage5_1_xjtu_merged.parquet"),
    }

    fcols = fusion_cols()
    X_train_hi = train_df[FEATURE_COLS].to_numpy(dtype=float)
    X_train_full = train_df[FEATURE_COLS + fcols].to_numpy(dtype=float)

    all_z = []
    auc_rows = []
    for name, df in datasets.items():
        print(f"\n--- {name} ---")
        z_df = zscore_vs_train(train_df, df, FEATURE_COLS)
        z_df.insert(0, "dataset", name)
        all_z.append(z_df)
        print(z_df.to_string(index=False))

        X_target_hi = df[FEATURE_COLS].to_numpy(dtype=float)
        X_target_full = df[FEATURE_COLS + fcols].to_numpy(dtype=float)
        auc_hi = domain_auc_vs_train(X_train_hi, X_target_hi)
        auc_full = domain_auc_vs_train(X_train_full, X_target_full)
        auc_rows.append({"dataset": name, "n_cycles": len(df), "n_cells": df["battery_id"].nunique(),
                          "auc_8hi_only": auc_hi, "auc_8hi_plus_16fusion": auc_full,
                          "max_abs_zscore": z_df["zscore"].abs().max()})
        print(f"[collapse-check] {name}: domain-classifier AUC (8 HI only)={auc_hi:.4f}, "
              f"(8 HI + 16 fusion)={auc_full:.4f}, max|z|={z_df['zscore'].abs().max():.2f}")

    z_all_df = pd.concat(all_z, ignore_index=True)
    z_all_df.to_csv(OUT_DIR / "stage5_collapse_check_zscores.csv", index=False)
    auc_df = pd.DataFrame(auc_rows)
    auc_df.to_csv(OUT_DIR / "stage5_collapse_check_auc.csv", index=False)

    print("\n=== SUMMARY ===")
    print(auc_df.to_string(index=False))
    print("\n[collapse-check] historical reference (pre-Stage-5): B0018 raw-feature z-scores "
          "were 854.7/185.5/419.9 (ICHV/TEVD/TEVI) pre-reformulation, collapsed to |z|<0.25 "
          "post-reformulation. CALCE's own historical domain-classifier AUC (full feature "
          "space, original 6-battery test set): 1.0000; expanded-pool version: 0.936.")


if __name__ == "__main__":
    main()
