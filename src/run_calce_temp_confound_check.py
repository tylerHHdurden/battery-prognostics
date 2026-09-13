"""
Stage 0, Check 0.2: temperature-feature imputation confound on CALCE.

CALCE has zero temperature channel at all (data_adapters.py) - MATC/MATD
(both temperature-based HIs) are 100% NaN for every CALCE cycle, and
run_calce_zero_retrain_eval.py imputes them with NASA+MIT TRAINING
medians (a constant value, not a domain-shift artifact, but also not a
genuine measurement). Since MATC/MATD are 2 of the 7 BFA-selected
features XGBoost-fusion was trained on, this checks how much of the
zero-retrain collapse (R2 0.917->0.304) is attributable to this specific
imputation artifact vs. genuine domain shift.

METHOD CHOICE (per the task's own two options): retraining a
MATC/MATD-free version, not zeroing them out post-hoc at inference. A
tree model has no principled "zero" for a feature it was trained to
split on with real NASA/MIT values around 20-40C - forcing an arbitrary
sentinel value would introduce a NEW artifact, not remove one. Retraining
on the identical NASA+MIT data/split/hyperparameters, just dropping 2 of
23 features, is the clean, valid way to isolate this confound - CALCE
then needs NO imputation at all for these 2 features, because they're
simply not part of the model anymore.

Reused UNCHANGED: the existing ica_encoder.pt (fusion embeddings don't
involve MATC/MATD at all - they're derived from V/I/T tensors, not HI
columns), the existing battery_split.json, the existing CALCE tensor-
building logic from run_calce_zero_retrain_eval.py.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))

from data_adapters import iterate_calce_cycles
from sequence_features import build_dataset_tensors, apply_channel_norm
from models.ica_encoder import ICAEncoder

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
PRED_DIR = PROC_DIR / "predictions"
OUT_DIR = ROOT / "outputs"

CALCE_CELLS = ["CS2_35", "CS2_36", "CS2_37"]
ICA_CHANNEL_SLICE = slice(3, 6)
DROPPED_FEATURES = ["MATC", "MATD"]


def build_calce_tensors():
    all_X, all_soh, all_bid, all_cyc = [], [], [], []
    for cid in CALCE_CELLS:
        cycles = list(iterate_calce_cycles(cid))
        X, soh, rul, idxs, censored = build_dataset_tensors(cycles)
        if X is None:
            continue
        all_X.append(X.astype(np.float32))
        all_soh.append(soh.astype(np.float32))
        all_bid += [cid] * len(soh)
        all_cyc += list(idxs)
    X_all = np.concatenate(all_X)
    soh_all = np.concatenate(all_soh)
    norm_stats = json.loads((PROC_DIR / "channel_norm_stats.json").read_text())
    X_all = apply_channel_norm(X_all, norm_stats)
    return X_all, soh_all, all_bid, all_cyc


def main():
    with open(PROC_DIR / "bfa_selected_features.txt") as f:
        bfa_selected = [l.strip() for l in f if l.strip()]
    reduced_features = [f for f in bfa_selected if f not in DROPPED_FEATURES]
    print(f"[calce-confound] original 7 BFA features: {bfa_selected}")
    print(f"[calce-confound] reduced (MATC/MATD dropped) features: {reduced_features}")

    fusion = pd.read_csv(PROC_DIR / "fusion_embeddings.csv")
    fusion_cols = [c for c in fusion.columns if c.startswith("fusion_")]

    hi_df = pd.read_parquet(PROC_DIR / "hi_table.parquet")
    nasa_mit = hi_df[hi_df["dataset"].isin(["NASA", "MIT"])]
    merged_train_full = pd.merge(nasa_mit, fusion, on=["dataset", "battery_id", "cycle_idx"], how="inner")
    print(f"[calce-confound] NASA+MIT training pool: {len(merged_train_full)} rows")

    split = json.loads((PROC_DIR / "battery_split.json").read_text())
    train_mask = merged_train_full["battery_id"].isin(split["train_ids"]).to_numpy()
    test_mask = ~train_mask

    def fit_and_eval(feature_set, label):
        cols = feature_set + fusion_cols
        X = merged_train_full[cols].to_numpy(dtype=float, copy=True)
        col_medians = np.nanmedian(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_medians, inds[1])
        y = merged_train_full["SOH"].to_numpy(dtype=float)

        model = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                              subsample=0.8, colsample_bytree=0.8, random_state=42,
                              n_jobs=-1, reg_lambda=1.0)
        model.fit(X[train_mask], y[train_mask])
        pred_test = model.predict(X[test_mask])
        rmse = float(np.sqrt(mean_squared_error(y[test_mask], pred_test)))
        mae = float(mean_absolute_error(y[test_mask], pred_test))
        r2 = float(r2_score(y[test_mask], pred_test))
        print(f"[calce-confound] [{label}] NASA+MIT TEST (in-domain): "
              f"RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f} (n_features={len(cols)})")
        return model, col_medians, {"rmse": rmse, "mae": mae, "r2": r2}

    model_full, medians_full, indomain_full = fit_and_eval(bfa_selected, "FULL (incl. MATC/MATD)")
    model_reduced, medians_reduced, indomain_reduced = fit_and_eval(reduced_features, "REDUCED (no MATC/MATD)")

    # --- CALCE evaluation, both versions, same CALCE cycles ---
    print("\n[calce-confound] === Building CALCE tensors + fusion embeddings (shared, unchanged encoder) ===")
    X_calce, soh_calce, bid_calce, cyc_calce = build_calce_tensors()
    encoder = ICAEncoder(in_channels=3, embed_dim=16)
    encoder.load_state_dict(torch.load(ROOT / "models" / "ica_encoder.pt"))
    encoder.eval()
    with torch.no_grad():
        fusion_emb = encoder.encode(torch.tensor(X_calce[:, :, ICA_CHANNEL_SLICE])).numpy()
    print(f"[calce-confound] CALCE cycles: {len(X_calce)}")

    calce_hi = hi_df[hi_df["dataset"] == "CALCE"]
    seq_df = pd.DataFrame({"battery_id": bid_calce, "cycle_idx": cyc_calce})
    for i in range(16):
        seq_df[f"fusion_{i}"] = fusion_emb[:, i]
    calce_merged = pd.merge(calce_hi, seq_df, on=["battery_id", "cycle_idx"], how="inner")
    y_calce = calce_merged["SOH"].to_numpy()
    print(f"[calce-confound] CALCE merged rows: {len(calce_merged)}")

    def predict_calce(model, feature_set, medians, label):
        cols = feature_set + fusion_cols
        X = calce_merged[cols].to_numpy(dtype=float, copy=True)
        for j, col in enumerate(feature_set):
            nan_mask = np.isnan(X[:, j])
            if nan_mask.any():
                X[nan_mask, j] = medians[j]
        n_imputed = np.isnan(calce_merged[feature_set].to_numpy(dtype=float)).sum()
        pred = model.predict(X)
        rmse = float(np.sqrt(mean_squared_error(y_calce, pred)))
        mae = float(mean_absolute_error(y_calce, pred))
        r2 = float(r2_score(y_calce, pred))
        print(f"[calce-confound] [{label}] CALCE (zero-retrain): "
              f"RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f} "
              f"({n_imputed} of {len(feature_set)} feature-columns' NaNs imputed with NASA+MIT medians)")
        return {"rmse": rmse, "mae": mae, "r2": r2}

    calce_full = predict_calce(model_full, bfa_selected, medians_full, "FULL (incl. MATC/MATD, imputed)")
    calce_reduced = predict_calce(model_reduced, reduced_features, medians_reduced, "REDUCED (no MATC/MATD)")

    print(f"\n[calce-confound] === SIDE-BY-SIDE COMPARISON ===")
    print(f"[calce-confound] In-domain (NASA+MIT test) R2:  FULL={indomain_full['r2']:.4f}  "
          f"REDUCED={indomain_reduced['r2']:.4f}  (delta={indomain_reduced['r2']-indomain_full['r2']:+.4f})")
    print(f"[calce-confound] CALCE (zero-retrain)       R2:  FULL={calce_full['r2']:.4f}  "
          f"REDUCED={calce_reduced['r2']:.4f}  (delta={calce_reduced['r2']-calce_full['r2']:+.4f})")
    print(f"[calce-confound] CALCE (zero-retrain)     RMSE:  FULL={calce_full['rmse']:.4f}  "
          f"REDUCED={calce_reduced['rmse']:.4f}  (delta={calce_reduced['rmse']-calce_full['rmse']:+.4f})")

    original_collapse_gap = indomain_full['r2'] - calce_full['r2']
    reduced_collapse_gap = indomain_reduced['r2'] - calce_reduced['r2']
    print(f"\n[calce-confound] Domain-shift GAP (in-domain R2 minus CALCE R2): "
          f"FULL={original_collapse_gap:.4f}  REDUCED={reduced_collapse_gap:.4f}")

    if calce_reduced['r2'] > calce_full['r2'] + 0.01:
        print(f"[calce-confound] FINDING (a): CALCE performance IMPROVES without MATC/MATD - "
              f"part of the collapse IS a data-quality/imputation artifact. Explains "
              f"{(calce_reduced['r2']-calce_full['r2'])/(1-calce_full['r2'])*100:.1f}% of the "
              f"remaining gap to a perfect R2=1.0, in absolute R2 points: "
              f"{calce_reduced['r2']-calce_full['r2']:+.4f}.")
    elif calce_reduced['r2'] < calce_full['r2'] - 0.01:
        print(f"[calce-confound] FINDING: CALCE performance WORSENS without MATC/MATD - "
              f"these features were (surprisingly) net-helpful even under imputation on CALCE. "
              f"The domain-shift interpretation is NOT weakened by this - if anything the "
              f"collapse is slightly UNDER-stated by keeping them.")
    else:
        print(f"[calce-confound] FINDING (b): CALCE performance is ESSENTIALLY UNCHANGED "
              f"({calce_reduced['r2']-calce_full['r2']:+.4f} R2) without MATC/MATD - "
              f"the collapse is NOT meaningfully explained by this confound. The original "
              f"domain-shift interpretation stands as-is.")

    pd.DataFrame([
        {"variant": "FULL (incl. MATC/MATD)", "domain": "NASA+MIT (in-domain)", **indomain_full},
        {"variant": "REDUCED (no MATC/MATD)", "domain": "NASA+MIT (in-domain)", **indomain_reduced},
        {"variant": "FULL (incl. MATC/MATD)", "domain": "CALCE (zero-retrain)", **calce_full},
        {"variant": "REDUCED (no MATC/MATD)", "domain": "CALCE (zero-retrain)", **calce_reduced},
    ]).to_csv(OUT_DIR / "calce_temp_confound_check.csv", index=False)
    print("\n[calce-confound] DONE")


if __name__ == "__main__":
    main()
