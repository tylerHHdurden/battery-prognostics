"""
Stage 3, Item 3.4: CORAL / Deep CORAL replacing MMD (session 13) as the
fusion-embedding domain-alignment mechanism.

CONFIGURATION: CORAL and the no-alignment baseline below both use
Stage 1's 1.1+1.5 canonical XGBoost-fusion model (reformulated duration
features + monotone_constraints on cycle_idx), original 32-battery pool
- the SAME configuration run_stage3_1_kmm_cp.py used, so those two
numbers are directly, apples-to-apples comparable. Session 13's MMD
number (cited below from DEVELOPMENT_LOG.md) used a DIFFERENT, earlier
downstream config (original un-reformulated BFA-selected features, no
monotone_constraints) - flagged explicitly wherever it's compared, since
that comparison mixes the alignment-method change with a downstream-
model-config change, unlike the CORAL-vs-no-alignment comparison which
isolates the alignment mechanism alone.

Pipeline: (1) train the CORAL-aligned ICA encoder (train_fusion_encoder_coral.py,
if not already run), (2) merge its embeddings into the 1.1+1.5 canonical
feature set, (3) fit XGBoost-fusion, (4) evaluate CALCE R2/RMSE and
split-conformal coverage, using the CORAL encoder's own CALCE embeddings
(not the original ica_encoder.pt's).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_common import (
    canonical_feature_cols, add_reformulated_duration_features, fit_xgb,
    eval_indomain, OUT_DIR, PROC_DIR, ROOT, ICA_CHANNEL_SLICE, CALCE_CELLS,
)
from stage1_common import build_calce_tensors  # unlabeled-tensor builder, encoder-agnostic
from run_conformal import calib_eval_battery_split, split_conformal
from models.ica_encoder import ICAEncoder

ALPHA = 0.1


def load_nasa_mit_pool_coral(reformulated=True):
    hi_df = pd.read_parquet(PROC_DIR / "hi_table.parquet")
    if reformulated:
        hi_df = add_reformulated_duration_features(hi_df)
    fusion = pd.read_csv(PROC_DIR / "fusion_embeddings_coral.csv")
    nasa_mit = hi_df[hi_df["dataset"].isin(["NASA", "MIT"])]
    merged = pd.merge(nasa_mit, fusion, on=["dataset", "battery_id", "cycle_idx"], how="inner")
    return merged, hi_df, fusion


def fusion_cols_coral(fusion_df):
    return [c for c in fusion_df.columns if c.startswith("fusion_")]


def build_calce_merged_coral(hi_df_full):
    """CALCE rows merged with the CORAL-trained encoder's OWN embeddings
    (models/ica_encoder_coral.pt) - NOT the original ica_encoder.pt used
    by stage1_common.build_calce_merged, since evaluating CORAL's effect
    requires CORAL's own CALCE-side embeddings, exactly as session 13's
    MMD eval used ica_encoder_mmd.pt's own CALCE embeddings."""
    X_calce, soh_calce, bid_calce, cyc_calce = build_calce_tensors()
    encoder = ICAEncoder(in_channels=3, embed_dim=16)
    encoder.load_state_dict(torch.load(ROOT / "models" / "ica_encoder_coral.pt"))
    encoder.eval()
    with torch.no_grad():
        fusion_emb = encoder.encode(torch.tensor(X_calce[:, :, ICA_CHANNEL_SLICE])).numpy()

    calce_hi = hi_df_full[hi_df_full["dataset"] == "CALCE"]
    seq_df = pd.DataFrame({"battery_id": bid_calce, "cycle_idx": cyc_calce})
    for i in range(16):
        seq_df[f"fusion_{i}"] = fusion_emb[:, i]
    calce_merged = pd.merge(calce_hi, seq_df, on=["battery_id", "cycle_idx"], how="inner")
    return calce_merged


def eval_calce_coral(model, medians, feature_cols, fcols, calce_merged):
    cols = feature_cols + fcols
    X = calce_merged[cols].to_numpy(dtype=float, copy=True)
    for j, col in enumerate(feature_cols):
        nan_mask = np.isnan(X[:, j])
        if nan_mask.any():
            X[nan_mask, j] = medians[j]
    y_true = calce_merged["SOH"].to_numpy()
    pred = model.predict(X)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "mae": float(mean_absolute_error(y_true, pred)),
        "r2": float(r2_score(y_true, pred)),
        "pred": pred, "y_true": y_true,
        "battery_id": calce_merged["battery_id"].to_numpy(),
    }


def main():
    print("[coral-stage34] === Building the Stage 1.1+1.5 canonical model on CORAL embeddings ===")
    merged, hi_full, fusion_df = load_nasa_mit_pool_coral(reformulated=True)
    fcols = fusion_cols_coral(fusion_df)
    split = json.loads((PROC_DIR / "battery_split.json").read_text())
    train_mask = merged["battery_id"].isin(split["train_ids"]).to_numpy()
    test_mask = ~train_mask

    base_features = canonical_feature_cols(reformulated=True)
    feature_cols = base_features + ["cycle_idx"]
    monotone = tuple([0] * len(base_features) + [-1] + [0] * len(fcols))

    cols = feature_cols + fcols
    X = merged[cols].to_numpy(dtype=float, copy=True)
    col_medians = np.nanmedian(X[train_mask], axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_medians, inds[1])
    y = merged["SOH"].to_numpy(dtype=float)

    from xgboost import XGBRegressor
    model = XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        n_jobs=-1, reg_lambda=1.0, monotone_constraints=monotone,
    )
    model.fit(X[train_mask], y[train_mask])
    medians, model_cols = col_medians, cols

    pred_test = model.predict(X[test_mask])
    y_test = y[test_mask]
    indomain = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
        "r2": float(r2_score(y_test, pred_test)),
        "pred": pred_test, "y_true": y_test,
        "battery_id": merged.loc[test_mask, "battery_id"].to_numpy(),
    }
    print(f"[coral-stage34] in-domain R2={indomain['r2']:.4f} RMSE={indomain['rmse']:.4f}")

    calce_merged = build_calce_merged_coral(hi_full)
    calce = eval_calce_coral(model, medians, feature_cols, fcols, calce_merged)
    print(f"[coral-stage34] CALCE R2={calce['r2']:.4f} RMSE={calce['rmse']:.4f}")

    # --- conformal coverage, same calib/target protocol as every other Stage item ---
    calib_ids, eval_ids = calib_eval_battery_split(sorted(set(indomain["battery_id"].tolist())))
    calib_mask = np.isin(indomain["battery_id"], calib_ids)
    calib_pred = indomain["pred"][calib_mask]
    calib_y = indomain["y_true"][calib_mask]

    _, lo, hi, method = split_conformal(calib_pred, calib_y, calce["pred"], ALPHA)
    cov = float(((calce["y_true"] >= lo) & (calce["y_true"] <= hi)).mean())
    width = float(np.mean(hi - lo))
    print(f"[coral-stage34] CALCE split-conformal coverage={cov:.4f} width={width:.4f}")

    print("\n[coral-stage34] === COMPARISON ===")
    print("[coral-stage34] no-alignment 1.1+1.5 baseline (this project, same config, stage3_1 run): "
          "CALCE R2=0.5672 RMSE=14.1669 | coverage=6.73% width=2.332")
    print("[coral-stage34] MMD (session 13, DIFFERENT downstream config - original BFA features, "
          "no monotone_constraints, no 1.1 reformulation): CALCE R2=0.337 RMSE=17.54 | coverage=4.4%")
    print(f"[coral-stage34] CORAL (THIS ITEM, canonical 1.1+1.5 config, same as no-alignment baseline): "
          f"CALCE R2={calce['r2']:.4f} RMSE={calce['rmse']:.4f} | coverage={cov*100:.2f}% width={width:.4f}")

    r2_vs_baseline = "IMPROVES" if calce['r2'] > 0.5672 else ("MATCHES" if abs(calce['r2']-0.5672) < 0.01 else "UNDERPERFORMS")
    cov_vs_baseline = "IMPROVES" if cov > 0.0673 else ("MATCHES" if abs(cov-0.0673) < 0.01 else "UNDERPERFORMS")
    print(f"\n[coral-stage34] CORAL vs no-alignment baseline (apples-to-apples, same 1.1+1.5 config): "
          f"R2 {r2_vs_baseline}, coverage {cov_vs_baseline}")

    pd.DataFrame([
        {"method": "no_alignment_1.1_1.5_baseline", "calce_r2": 0.5672, "calce_rmse": 14.1669, "coverage": 0.0673, "avg_width": 2.3315},
        {"method": "MMD_session13_different_config", "calce_r2": 0.337, "calce_rmse": 17.54, "coverage": 0.044, "avg_width": np.nan},
        {"method": "CORAL_1.1_1.5_config", "calce_r2": calce['r2'], "calce_rmse": calce['rmse'], "coverage": cov, "avg_width": width},
    ]).to_csv(OUT_DIR / "stage3_4_coral_results.csv", index=False)
    print("\n[coral-stage34] saved outputs/stage3_4_coral_results.csv")
    print("[coral-stage34] DONE")


if __name__ == "__main__":
    main()
