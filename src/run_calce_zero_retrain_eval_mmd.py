"""
CALCE zero-retrain evaluation of the MMD-ALIGNED fusion ensemble —
identical procedure to run_calce_zero_retrain_eval.py (same CALCE cells,
same NASA+MIT-only normalization/median-imputation stats, same
calib/eval battery split for conformal recalibration), but every fusion
component is the MMD-aligned one:
    ica_encoder_mmd.pt, xgb_soh_fusion_mmd.json, ridge_meta_fusion_mmd.pkl

Zero-retrain/zero-refit is preserved exactly as before: CALCE is used
here purely for forward-pass inference and for computing empirical
conformal coverage. It is NOT used to fit or recalibrate anything in
this script — the conformal interval is calibrated only on the NASA+MIT
calibration half, exactly as in the non-MMD script, so this is a fair
apples-to-apples comparison of the two encoders' OOD behavior, not a
comparison of an unaligned model against a CALCE-tuned one.

(CALCE's raw discharge curves — never its SOH/RUL labels — WERE used
during training of ica_encoder_mmd.pt itself, by train_fusion_encoder_mmd.py,
for the MMD alignment term. That happened upstream of this script and is
standard unsupervised domain adaptation, not something this evaluation
script does.)

Reports the same two things as the non-MMD script, for direct comparison:
  1. RMSE/MAE/R2 on all CALCE cycles vs. the non-MMD zero-retrain R2=0.31.
  2. Empirical coverage of the (freshly recalibrated on the MMD model's
     own NASA+MIT residuals) 90% interval on CALCE, vs. the non-MMD
     script's 6.1%.
"""

import json
import pickle
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
from train_deep_models import load_all_battery_tensors, make_xy
from models.vlstm import VLSTM
from models.cnn_lstm import CNNLSTM
from models.piformer import PiFormer
from models.ica_encoder import ICAEncoder
from run_conformal import calib_eval_battery_split, evaluate_coverage

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
PRED_DIR = PROC_DIR / "predictions"
OUT_DIR = ROOT / "outputs"

CALCE_CELLS = ["CS2_35", "CS2_36", "CS2_37"]
BFA_SELECTED = None  # loaded in main()
ICA_CHANNEL_SLICE = slice(3, 6)


def build_calce_tensors_and_hi():
    all_X, all_soh, all_rul, all_bid, all_cyc = [], [], [], [], []
    for cid in CALCE_CELLS:
        cycles = list(iterate_calce_cycles(cid))
        X, soh, rul, idxs, censored = build_dataset_tensors(cycles)
        if X is None:
            continue
        all_X.append(X.astype(np.float32))
        all_soh.append(soh.astype(np.float32))
        all_rul.append(rul.astype(np.float32))
        all_bid += [cid] * len(soh)
        all_cyc += list(idxs)
        print(f"[calce-mmd] {cid}: {X.shape[0]} cycles usable for sequence models")

    X_all = np.concatenate(all_X)
    soh_all = np.concatenate(all_soh)
    rul_all = np.concatenate(all_rul)

    norm_stats = json.loads((PROC_DIR / "channel_norm_stats.json").read_text())
    X_all = apply_channel_norm(X_all, norm_stats)

    return X_all, soh_all, rul_all, all_bid, all_cyc


def main():
    global BFA_SELECTED
    with open(PROC_DIR / "bfa_selected_features.txt") as f:
        BFA_SELECTED = [l.strip() for l in f if l.strip()]

    print("[calce-mmd] === building CALCE sequence tensors (zero-retrain, reusing NASA+MIT norm stats) ===")
    X_calce, soh_calce, rul_calce, bid_calce, cyc_calce = build_calce_tensors_and_hi()
    print(f"[calce-mmd] total usable sequence-model cycles: {len(X_calce)}")

    battery_data = load_all_battery_tensors()
    split = json.loads((PROC_DIR / "battery_split.json").read_text())
    train_ids = [b for b in split["train_ids"] if b in battery_data]
    n_val = max(1, len(train_ids) // 5)
    val_ids = sorted(train_ids)[-n_val:]
    fit_ids = [b for b in train_ids if b not in val_ids]
    _, y_fit_ref, _, _, _, _ = make_xy(battery_data, fit_ids)
    y_mean, y_std = float(y_fit_ref.mean()), float(y_fit_ref.std() + 1e-8)
    print(f"[calce-mmd] recomputed de-standardization constants from NASA+MIT fit split: "
          f"y_mean={y_mean:.3f} y_std={y_std:.3f}")

    # --- deep model predictions (unchanged, non-fusion weights - the MMD
    # variant only touches the ICAEncoder/XGBoost-fusion/Ridge-fusion chain) ---
    def load_and_predict(name, model, x_slice=slice(None)):
        model.load_state_dict(torch.load(ROOT / "models" / f"{name}.pt"))
        model.eval()
        with torch.no_grad():
            raw = model(torch.tensor(X_calce[:, :, x_slice])).squeeze(-1).numpy()
        return raw * y_std + y_mean

    pred_vlstm = load_and_predict("vlstm_soh", VLSTM(input_size=1, hidden_size=32, n_targets=1), slice(0, 1))
    pred_cnnlstm = load_and_predict("cnn_lstm_soh", CNNLSTM())
    pred_piformer = load_and_predict("piformer_soh", PiFormer())
    print(f"[calce-mmd] deep model predictions done (VLSTM/CNNLSTM/PiFormer, non-physics weights)")

    # --- MMD-aligned fusion embeddings (ICAEncoder forward pass only) ---
    encoder = ICAEncoder(in_channels=3, embed_dim=16)
    encoder.load_state_dict(torch.load(ROOT / "models" / "ica_encoder_mmd.pt"))
    encoder.eval()
    with torch.no_grad():
        fusion_emb = encoder.encode(torch.tensor(X_calce[:, :, ICA_CHANNEL_SLICE])).numpy()
    print(f"[calce-mmd] MMD-aligned fusion embeddings extracted: {fusion_emb.shape}")

    hi_df = pd.read_parquet(PROC_DIR / "hi_table.parquet")
    hi_df = hi_df[hi_df["dataset"] == "CALCE"]
    seq_df = pd.DataFrame({"battery_id": bid_calce, "cycle_idx": cyc_calce})
    for i in range(16):
        seq_df[f"fusion_{i}"] = fusion_emb[:, i]
    seq_df["pred_VLSTM"] = pred_vlstm
    seq_df["pred_CNNLSTM"] = pred_cnnlstm
    seq_df["pred_PiFormer"] = pred_piformer

    merged = pd.merge(hi_df, seq_df, on=["battery_id", "cycle_idx"], how="inner")
    print(f"[calce-mmd] hi_table CALCE rows={len(hi_df)}, sequence-model rows={len(seq_df)}, "
          f"merged (inner) rows={len(merged)}")

    train_hi_df = pd.read_parquet(PROC_DIR / "hi_table.parquet")
    train_hi_df = train_hi_df[train_hi_df["dataset"].isin(["NASA", "MIT"])]
    train_medians = train_hi_df[BFA_SELECTED].median(numeric_only=True)
    fusion_cols = [f"fusion_{i}" for i in range(16)]
    feature_cols = BFA_SELECTED + fusion_cols

    X_feat = merged[feature_cols].to_numpy(dtype=float, copy=True)
    for j, col in enumerate(BFA_SELECTED):
        col_nan = np.isnan(X_feat[:, j])
        if col_nan.any():
            X_feat[col_nan, j] = train_medians[col]
    n_nan_matc_matd = merged[["MATC", "MATD"]].isna().sum().sum()
    print(f"[calce-mmd] imputed {n_nan_matc_matd} NaN cells (MATC/MATD, 100% missing on CALCE) "
          f"with NASA+MIT training medians")

    xgb_fusion = XGBRegressor()
    xgb_fusion.load_model(str(ROOT / "models" / "xgb_soh_fusion_mmd.json"))
    pred_xgb_fusion = xgb_fusion.predict(X_feat)

    with open(ROOT / "models" / "ridge_meta_fusion_mmd.pkl", "rb") as f:
        ridge_fusion = pickle.load(f)
    meta_X = np.column_stack([
        pred_xgb_fusion, merged["pred_VLSTM"], merged["pred_CNNLSTM"], merged["pred_PiFormer"],
    ] + [merged[c] for c in fusion_cols])
    pred_ensemble = ridge_fusion.predict(meta_X)

    y_true = merged["SOH"].to_numpy()

    def metrics(name, pred):
        rmse = float(np.sqrt(mean_squared_error(y_true, pred)))
        mae = float(mean_absolute_error(y_true, pred))
        r2 = float(r2_score(y_true, pred))
        print(f"[calce-mmd] {name}: RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")
        return {"model": name, "rmse": rmse, "mae": mae, "r2": r2, "n": len(y_true)}

    print("\n[calce-mmd] === ZERO-RETRAIN RESULTS, MMD-ALIGNED (out-of-domain, all 3 CALCE cells) ===")
    results = [
        metrics("XGBoost-fusion-MMD (CALCE, zero-retrain)", pred_xgb_fusion),
        metrics("Stacking-Ridge-fusion-MMD (CALCE, zero-retrain)", pred_ensemble),
    ]
    print("\n[calce-mmd] non-MMD zero-retrain reference: "
          "XGBoost-fusion R2 and Stacking-Ridge-fusion R2 both ~0.31 on CALCE "
          "(see calce_zero_retrain_metrics.csv / DEVELOPMENT_LOG.md)")

    pd.DataFrame(results).to_csv(PRED_DIR / "calce_zero_retrain_mmd_metrics.csv", index=False)
    merged_out = merged[["dataset", "battery_id", "cycle_idx", "SOH", "RUL"]].copy()
    merged_out["pred_XGBoost_fusion_mmd"] = pred_xgb_fusion
    merged_out["pred_Stacking_Ridge_fusion_mmd"] = pred_ensemble
    merged_out.to_csv(PRED_DIR / "calce_zero_retrain_mmd_preds.csv", index=False)

    # --- conformal: RECALIBRATE on the MMD model's own NASA+MIT residuals
    # (this is what the task calls "recalibrated using the MMD-aligned
    # model" - still zero-label-leakage, since calibration uses only the
    # NASA+MIT calib half, never CALCE) ---
    print("\n[calce-mmd] === Conformal interval check (recalibrated on MMD model's NASA+MIT residuals) ===")
    ens_test_mmd = pd.read_csv(PRED_DIR / "ensemble_fusion_mmd_test_preds.csv")
    calib_ids, eval_ids = calib_eval_battery_split(ens_test_mmd["battery_id"].unique().tolist())
    calib_df = ens_test_mmd[ens_test_mmd["battery_id"].isin(calib_ids)]
    eval_df = ens_test_mmd[ens_test_mmd["battery_id"].isin(eval_ids)]

    r_eval, lo_eval, hi_eval = evaluate_coverage(
        "SOH in-domain (NASA+MIT eval, MMD-aligned fusion ensemble)",
        calib_df["pred_Stacking_Ridge_fusion"].to_numpy(), calib_df["SOH"].to_numpy(),
        eval_df["pred_Stacking_Ridge_fusion"].to_numpy(), eval_df["SOH"].to_numpy(),
    )
    half_width_in_domain = float(np.mean(hi_eval - lo_eval)) / 2

    r_calce, lo_calce, hi_calce = evaluate_coverage(
        "SOH out-of-domain (CALCE, zero-retrain, MMD-recalibrated)",
        calib_df["pred_Stacking_Ridge_fusion"].to_numpy(), calib_df["SOH"].to_numpy(),
        pred_ensemble, y_true,
    )
    half_width_calce = float(np.mean(hi_calce - lo_calce)) / 2

    print(f"\n[calce-mmd] in-domain (NASA+MIT eval) half-width={half_width_in_domain:.3f}, "
          f"coverage={r_eval['empirical_coverage']:.3f}")
    print(f"[calce-mmd] out-of-domain (CALCE) half-width={half_width_calce:.3f}, "
          f"coverage={r_calce['empirical_coverage']:.3f}")
    print(f"[calce-mmd] non-MMD reference: in-domain coverage=95.6%, CALCE coverage=6.1% "
          f"(same fixed half-width in both domains)")

    pd.DataFrame([
        {"domain": "NASA+MIT (in-domain)", "half_width": half_width_in_domain,
         "empirical_coverage": r_eval["empirical_coverage"], "target_coverage": 0.9},
        {"domain": "CALCE (out-of-domain, zero-retrain, MMD-aligned)", "half_width": half_width_calce,
         "empirical_coverage": r_calce["empirical_coverage"], "target_coverage": 0.9},
    ]).to_csv(OUT_DIR / "calce_zero_retrain_mmd_conformal.csv", index=False)

    print("\n[calce-mmd] DONE")


if __name__ == "__main__":
    main()
