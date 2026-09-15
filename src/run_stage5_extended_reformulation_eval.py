"""
Stage 5 follow-on, item 1, full pipeline: apply the SCV/MATD/VIECT
reformulation (stage5_extended_reformulation.py) on top of Stage 1.1's
existing duration reformulation + Stage 1.5's monotone constraints,
recompute the item-2 domain-classifier AUC check, retrain XGBoost-
fusion, and re-evaluate zero-retrain on CALCE/Oxford/HUST/XJTU.

EXPERIMENTAL - a separate model/feature set, NOT written to
models/xgb_soh_fusion.json or any file live_inference.py loads. Only
promoted to deployment if explicitly confirmed after review.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))

from stage1_common import (
    canonical_feature_cols, load_nasa_mit_pool, battery_split_masks,
    fusion_cols, build_calce_merged, fit_xgb, eval_indomain, eval_calce,
    OUT_DIR, PROC_DIR, ROOT,
)
from stage5_extended_reformulation import add_scv_matd_viect_reformulated, extended_canonical_feature_cols


def domain_auc_vs_train(X_train: np.ndarray, X_target: np.ndarray, seed: int = 42) -> float:
    X_train = np.where(np.isinf(X_train), np.nan, X_train)
    X_target = np.where(np.isinf(X_target), np.nan, X_target)
    train_medians = np.nanmedian(X_train, axis=0)
    Xt = X_train.copy(); inds = np.where(np.isnan(Xt)); Xt[inds] = np.take(train_medians, inds[1])
    Xg = X_target.copy(); inds2 = np.where(np.isnan(Xg)); Xg[inds2] = np.take(train_medians, inds2[1])
    Xp = np.concatenate([Xt, Xg])
    mean, std = Xp.mean(0), Xp.std(0) + 1e-8
    Xp_z = (Xp - mean) / std
    y = np.concatenate([np.zeros(len(Xt)), np.ones(len(Xg))])
    clf = LogisticRegression(max_iter=2000, random_state=seed).fit(Xp_z, y)
    return float(roc_auc_score(y, clf.predict_proba(Xp_z)[:, 1]))


def eval_generic(model, medians, feature_cols, df, fcols):
    cols = feature_cols + fcols
    X = df[cols].to_numpy(dtype=float, copy=True)
    X = np.where(np.isinf(X), np.nan, X)
    for j in range(len(feature_cols)):
        nan_mask = np.isnan(X[:, j])
        if nan_mask.any():
            X[nan_mask, j] = medians[j]
    y_true = df["SOH"].to_numpy()
    pred = model.predict(X)
    return {"rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
            "mae": float(mean_absolute_error(y_true, pred)),
            "r2": float(r2_score(y_true, pred)), "pred": pred, "y_true": y_true,
            "battery_id": df["battery_id"].to_numpy()}


def main():
    print("=== Step 1: load pool + apply extended reformulation ===")
    merged_nm, hi_full = load_nasa_mit_pool(reformulated=True)
    hi_full_ext = add_scv_matd_viect_reformulated(hi_full)
    fcols = fusion_cols()
    fusion_df = pd.read_csv(PROC_DIR / "fusion_embeddings.csv")
    nasa_mit_ext = hi_full_ext[hi_full_ext["dataset"].isin(["NASA", "MIT"])]
    merged_ext = pd.merge(nasa_mit_ext, fusion_df, on=["dataset", "battery_id", "cycle_idx"], how="inner")
    train_mask, test_mask, split = battery_split_masks(merged_ext)

    base_cols = canonical_feature_cols(reformulated=True)
    extended_cols = extended_canonical_feature_cols(base_cols)
    feature_cols = extended_cols + ["cycle_idx"]
    print(f"[extended] base reformulated cols: {base_cols}")
    print(f"[extended] EXTENDED cols (SCV/MATD/VIECT swapped): {extended_cols}")

    calce_merged = build_calce_merged(hi_full)  # unreformulated-extended CALCE hi rows
    calce_merged_ext = add_scv_matd_viect_reformulated(calce_merged)

    oxford = add_scv_matd_viect_reformulated(pd.read_parquet(PROC_DIR / "stage5_1_oxford_merged.parquet"))
    hust = add_scv_matd_viect_reformulated(pd.read_parquet(PROC_DIR / "stage5_1_hust_merged.parquet"))
    xjtu = add_scv_matd_viect_reformulated(pd.read_parquet(PROC_DIR / "stage5_1_xjtu_merged.parquet"))

    print("\n=== Step 2: domain-classifier AUC, ORIGINAL vs. EXTENDED feature set ===")
    train_df = merged_ext.loc[train_mask]
    results_auc = []
    for name, df in [("CALCE", calce_merged_ext), ("Oxford", oxford), ("HUST", hust), ("XJTU", xjtu)]:
        X_train_orig = train_df[base_cols].to_numpy(dtype=float)
        X_target_orig = df[base_cols].to_numpy(dtype=float)
        auc_orig = domain_auc_vs_train(X_train_orig, X_target_orig)

        X_train_ext = train_df[extended_cols].to_numpy(dtype=float)
        X_target_ext = df[extended_cols].to_numpy(dtype=float)
        auc_ext = domain_auc_vs_train(X_train_ext, X_target_ext)

        print(f"[extended] {name}: AUC original={auc_orig:.4f} -> AUC extended={auc_ext:.4f} "
              f"(delta={auc_ext-auc_orig:+.4f})")
        results_auc.append({"dataset": name, "auc_original_8feat": auc_orig, "auc_extended_8feat": auc_ext})
    pd.DataFrame(results_auc).to_csv(OUT_DIR / "stage5_extended_reformulation_auc.csv", index=False)

    print("\n=== Step 3: retrain XGBoost-fusion (extended features + 1.5 monotone constraints) ===")
    # EXACT same convention as run_stage4_step2b_xgb_joint.py (verified
    # by reading that script directly before writing this, not assumed):
    # Stage 1.5 constrains ONLY cycle_idx (monotonically non-increasing
    # SOH as cycle_idx rises) - none of the 8 HI features themselves are
    # individually constrained. Swapping SCV/MATD/VIECT for their _rel
    # versions changes nothing about which position gets constrained;
    # applied unchanged, not reinvented.
    monotone = tuple([0] * len(extended_cols) + [-1] + [0] * len(fcols))
    print(f"[extended] monotone_constraints: {monotone} (only cycle_idx constrained, matching Stage 4 exactly)")
    model, medians, cols = fit_xgb(merged_ext, train_mask, feature_cols,
                                    xgb_extra_kwargs={"monotone_constraints": monotone})

    print("\n=== Step 4: evaluate in-domain + zero-retrain on all 4 held-out datasets ===")
    indomain = eval_indomain(model, medians, cols, merged_ext, test_mask)
    print(f"[extended] IN-DOMAIN: R2={indomain['r2']:.4f} RMSE={indomain['rmse']:.4f} "
          f"(Stage 4 baseline: R2=0.9740 fixed-split / 0.9658 GroupKFold-mean)")

    summary_rows = [{"dataset": "in-domain (fixed split)", "n_cycles": len(indomain["pred"]),
                      "r2": indomain["r2"], "rmse": indomain["rmse"]}]
    for name, df in [("CALCE", calce_merged_ext), ("Oxford", oxford), ("HUST", hust), ("XJTU", xjtu)]:
        result = eval_generic(model, medians, feature_cols, df, fcols)
        print(f"[extended] {name}: R2={result['r2']:.4f} RMSE={result['rmse']:.4f} "
              f"(n={len(result['pred'])}, {len(set(result['battery_id']))} cells)")
        summary_rows.append({"dataset": name, "n_cycles": len(result["pred"]),
                              "r2": result["r2"], "rmse": result["rmse"]})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "stage5_extended_reformulation_eval.csv", index=False)
    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))
    print("\n[extended] vs. Stage 5.1 ORIGINAL zero-retrain: CALCE=0.568, Oxford=-2.694, "
          "HUST=-0.152, XJTU=-1.059")
    print("[extended] vs. Stage 4 in-domain: R2=0.9740 (fixed split) / 0.9658 (GroupKFold mean)")

    model.save_model(str(ROOT / "models" / "_experimental_xgb_soh_fusion_extended_reformulation.json"))
    print("\n[extended] saved EXPERIMENTAL model to models/_experimental_xgb_soh_fusion_extended_"
          "reformulation.json - NOT wired into live_inference.py/app.py")


if __name__ == "__main__":
    main()
