"""
CALCE zero-retrain evaluation of the final fusion-enabled ensemble
(XGBoost+fusion base learner, Stacking-Ridge+fusion meta-learner - NOT
the physics-informed models, which showed slightly worse in-domain
performance and were kept as a separate, non-adopted result).

Every model used here (ICAEncoder, VLSTM, CNN-LSTM, PiFormer, XGBoost-
fusion, Ridge-meta-fusion) is loaded from its already-trained weights
and used purely for inference - CALCE was never in any train/val/fit
split for any of them (Phase 1's BFA feature selection is the only place
CALCE data was used, for feature selection, not model fitting). The
per-channel normalization stats (channel_norm_stats.json) and HI column-
median imputation values are both reused from NASA+MIT training data,
never recomputed from CALCE - true zero-retrain, zero-refit.

Reports:
  1. RMSE/MAE/R2 on all 2,943 CALCE cycles vs. the known NASA+MIT
     in-domain test performance (R2=0.917 for both fusion models).
  2. Whether the existing NASA+MIT-calibrated 90% split-conformal
     interval (same calib/eval battery split as run_conformal.py)
     "widens" when applied to CALCE, and what CALCE's empirical
     coverage actually is under that fixed interval.
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
    """Returns (X_calce [n,200,6] normalized, soh, rul, battery_id, cycle_idx, hi_df)."""
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
        print(f"[calce] {cid}: {X.shape[0]} cycles usable for sequence models")

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

    print("[calce] === building CALCE sequence tensors (zero-retrain, reusing NASA+MIT norm stats) ===")
    X_calce, soh_calce, rul_calce, bid_calce, cyc_calce = build_calce_tensors_and_hi()
    print(f"[calce] total usable sequence-model cycles: {len(X_calce)}")

    # De-standardization constants: `model.y_mean_`/`model.y_std_` were set
    # as plain Python attributes on the model instance during ORIGINAL
    # training (train_deep_models.py) but are NOT part of state_dict(), so
    # a freshly-loaded model doesn't have them. Recomputed here from the
    # exact same NASA+MIT fit-battery split used at training time (never
    # from CALCE - that would leak test-domain info into "training"
    # statistics and break zero-retrain integrity). All 3 deep models
    # were trained on the same y_fit, so they share one y_mean/y_std pair.
    battery_data = load_all_battery_tensors()
    split = json.loads((PROC_DIR / "battery_split.json").read_text())
    train_ids = [b for b in split["train_ids"] if b in battery_data]
    n_val = max(1, len(train_ids) // 5)
    val_ids = sorted(train_ids)[-n_val:]
    fit_ids = [b for b in train_ids if b not in val_ids]
    _, y_fit_ref, _, _, _, _ = make_xy(battery_data, fit_ids)
    y_mean, y_std = float(y_fit_ref.mean()), float(y_fit_ref.std() + 1e-8)
    print(f"[calce] recomputed de-standardization constants from NASA+MIT fit split: "
          f"y_mean={y_mean:.3f} y_std={y_std:.3f}")

    # --- deep model predictions (forward-pass only, no fitting) ---
    def load_and_predict(name, model, x_slice=slice(None)):
        model.load_state_dict(torch.load(ROOT / "models" / f"{name}.pt"))
        model.eval()
        with torch.no_grad():
            raw = model(torch.tensor(X_calce[:, :, x_slice])).squeeze(-1).numpy()
        return raw * y_std + y_mean

    pred_vlstm = load_and_predict("vlstm_soh", VLSTM(input_size=1, hidden_size=32, n_targets=1), slice(0, 1))
    pred_cnnlstm = load_and_predict("cnn_lstm_soh", CNNLSTM())
    pred_piformer = load_and_predict("piformer_soh", PiFormer())
    print(f"[calce] deep model predictions done (VLSTM/CNNLSTM/PiFormer, non-physics weights)")

    # --- fusion embeddings (ICAEncoder forward pass only) ---
    encoder = ICAEncoder(in_channels=3, embed_dim=16)
    encoder.load_state_dict(torch.load(ROOT / "models" / "ica_encoder.pt"))
    encoder.eval()
    with torch.no_grad():
        fusion_emb = encoder.encode(torch.tensor(X_calce[:, :, ICA_CHANNEL_SLICE])).numpy()
    print(f"[calce] fusion embeddings extracted: {fusion_emb.shape}")

    # --- align with HI table (7 BFA-selected features) ---
    hi_df = pd.read_parquet(PROC_DIR / "hi_table.parquet")
    hi_df = hi_df[hi_df["dataset"] == "CALCE"]
    seq_df = pd.DataFrame({"battery_id": bid_calce, "cycle_idx": cyc_calce})
    for i in range(16):
        seq_df[f"fusion_{i}"] = fusion_emb[:, i]
    seq_df["pred_VLSTM"] = pred_vlstm
    seq_df["pred_CNNLSTM"] = pred_cnnlstm
    seq_df["pred_PiFormer"] = pred_piformer

    merged = pd.merge(hi_df, seq_df, on=["battery_id", "cycle_idx"], how="inner")
    print(f"[calce] hi_table CALCE rows={len(hi_df)}, sequence-model rows={len(seq_df)}, "
          f"merged (inner) rows={len(merged)}")

    # --- XGBoost-fusion prediction ---
    # MATC/MATD are NaN for 100% of CALCE rows (no temperature channel at
    # all - not a domain-shift artifact, a genuine data-availability gap).
    # Imputed with NASA+MIT TRAINING medians (never recomputed from CALCE)
    # to preserve zero-retrain integrity, exactly mirroring how
    # train_xgboost_fusion.py imputes at train time.
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
    print(f"[calce] imputed {n_nan_matc_matd} NaN cells (MATC/MATD, 100% missing on CALCE) "
          f"with NASA+MIT training medians")

    xgb_fusion = XGBRegressor()
    xgb_fusion.load_model(str(ROOT / "models" / "xgb_soh_fusion.json"))
    pred_xgb_fusion = xgb_fusion.predict(X_feat)

    # --- Stacking-Ridge-fusion meta-learner prediction ---
    with open(ROOT / "models" / "ridge_meta_fusion.pkl", "rb") as f:
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
        print(f"[calce] {name}: RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")
        return {"model": name, "rmse": rmse, "mae": mae, "r2": r2, "n": len(y_true)}

    print("\n[calce] === ZERO-RETRAIN RESULTS (out-of-domain, all 3 CALCE cells) ===")
    results = [
        metrics("XGBoost-fusion (CALCE, zero-retrain)", pred_xgb_fusion),
        metrics("Stacking-Ridge-fusion (CALCE, zero-retrain)", pred_ensemble),
    ]
    print("\n[calce] in-domain (NASA+MIT test) reference: "
          "XGBoost-fusion R2=0.917 (RMSE=1.392, MAE=0.959), "
          "Stacking-Ridge-fusion R2=0.917 (RMSE=1.394, MAE=0.966)")

    pd.DataFrame(results).to_csv(PRED_DIR / "calce_zero_retrain_metrics.csv", index=False)
    merged_out = merged[["dataset", "battery_id", "cycle_idx", "SOH", "RUL"]].copy()
    merged_out["pred_XGBoost_fusion"] = pred_xgb_fusion
    merged_out["pred_Stacking_Ridge_fusion"] = pred_ensemble
    merged_out.to_csv(PRED_DIR / "calce_zero_retrain_preds.csv", index=False)

    # --- conformal: apply the EXISTING NASA+MIT-calibrated interval to CALCE ---
    print("\n[calce] === Conformal interval check: does it widen on CALCE? ===")
    ens_test_fusion = pd.read_csv(PRED_DIR / "ensemble_fusion_test_preds.csv")
    calib_ids, eval_ids = calib_eval_battery_split(ens_test_fusion["battery_id"].unique().tolist())
    calib_df = ens_test_fusion[ens_test_fusion["battery_id"].isin(calib_ids)]
    eval_df = ens_test_fusion[ens_test_fusion["battery_id"].isin(eval_ids)]

    # calibrate on NASA+MIT calib half (same battery split as run_conformal.py),
    # get the interval half-width, then check coverage on: (a) the NASA+MIT
    # eval half (in-domain sanity check, should match ~95% from before), and
    # (b) ALL of CALCE (out-of-domain, zero-retrain).
    r_eval, lo_eval, hi_eval = evaluate_coverage(
        "SOH in-domain (NASA+MIT eval, fusion ensemble)",
        calib_df["pred_Stacking_Ridge_fusion"].to_numpy(), calib_df["SOH"].to_numpy(),
        eval_df["pred_Stacking_Ridge_fusion"].to_numpy(), eval_df["SOH"].to_numpy(),
    )
    half_width_in_domain = float(np.mean(hi_eval - lo_eval)) / 2

    r_calce, lo_calce, hi_calce = evaluate_coverage(
        "SOH out-of-domain (CALCE, zero-retrain, SAME calibration)",
        calib_df["pred_Stacking_Ridge_fusion"].to_numpy(), calib_df["SOH"].to_numpy(),
        pred_ensemble, y_true,
    )
    half_width_calce = float(np.mean(hi_calce - lo_calce)) / 2

    print(f"\n[calce] in-domain (NASA+MIT eval) half-width={half_width_in_domain:.3f}, "
          f"coverage={r_eval['empirical_coverage']:.3f}")
    print(f"[calce] out-of-domain (CALCE) half-width={half_width_calce:.3f}, "
          f"coverage={r_calce['empirical_coverage']:.3f}")
    if abs(half_width_calce - half_width_in_domain) < 1e-6:
        print("[calce] FINDING: interval width is IDENTICAL on CALCE vs NASA+MIT eval - "
              "this split-conformal implementation uses a single GLOBAL fixed-width "
              "quantile from calibration, so it does NOT adapt/widen per-input. It does "
              "not 'know' CALCE is out-of-domain.")

    pd.DataFrame([
        {"domain": "NASA+MIT (in-domain)", "half_width": half_width_in_domain,
         "empirical_coverage": r_eval["empirical_coverage"], "target_coverage": 0.9},
        {"domain": "CALCE (out-of-domain, zero-retrain)", "half_width": half_width_calce,
         "empirical_coverage": r_calce["empirical_coverage"], "target_coverage": 0.9},
    ]).to_csv(OUT_DIR / "calce_zero_retrain_conformal.csv", index=False)

    print("\n[calce] DONE")


if __name__ == "__main__":
    main()
