"""
Dataset Expansion Phase 1, Step 4 (the actual hypothesis test): CALCE
zero-retrain evaluation of the EXPANDED-pool fusion ensemble - the
direct test of whether more NASA+MIT training data (32 -> ~219
batteries) improves generalization to CALCE's genuinely different cell
chemistry/format/protocol, and whether the fixed-width conformal
interval's coverage collapse (95.6% -> 6.1% on the original 32-battery
model) is any different here.

Mirrors run_calce_zero_retrain_eval.py's exact method, swapped to the
expanded-pool models/embeddings/split - additive, does not touch the
original script or its outputs. CALCE itself is UNCHANGED (still the
same 3 cells, never part of the expanded training pool) - only the
model being evaluated against it has more training data now.
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
from train_deep_models import make_xy
from train_deep_models_expanded import load_all_battery_tensors_expanded
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
        print(f"[calce-exp] {cid}: {X.shape[0]} cycles usable for sequence models")

    X_all = np.concatenate(all_X)
    soh_all = np.concatenate(all_soh)
    rul_all = np.concatenate(all_rul)

    norm_stats = json.loads((PROC_DIR / "channel_norm_stats_expanded.json").read_text())
    X_all = apply_channel_norm(X_all, norm_stats)
    return X_all, soh_all, rul_all, all_bid, all_cyc


def main():
    with open(PROC_DIR / "bfa_selected_features_expanded.txt") as f:
        bfa_selected = [l.strip() for l in f if l.strip()]

    print("[calce-exp] === building CALCE sequence tensors (zero-retrain, expanded-pool norm stats) ===")
    X_calce, soh_calce, rul_calce, bid_calce, cyc_calce = build_calce_tensors_and_hi()
    print(f"[calce-exp] total usable sequence-model cycles: {len(X_calce)}")

    battery_data = load_all_battery_tensors_expanded()
    split = json.loads((PROC_DIR / "battery_split_expanded.json").read_text())
    train_ids = [b for b in split["train_ids"] if b in battery_data]
    n_val = max(1, len(train_ids) // 5)
    val_ids = sorted(train_ids)[-n_val:]
    fit_ids = [b for b in train_ids if b not in val_ids]
    _, y_fit_ref, _, _, _, _ = make_xy(battery_data, fit_ids)
    y_mean, y_std = float(y_fit_ref.mean()), float(y_fit_ref.std() + 1e-8)
    print(f"[calce-exp] de-standardization constants (expanded fit split): "
          f"y_mean={y_mean:.3f} y_std={y_std:.3f}")

    def load_and_predict(name, model, x_slice=slice(None)):
        model.load_state_dict(torch.load(ROOT / "models" / f"{name}.pt"))
        model.eval()
        with torch.no_grad():
            raw = model(torch.tensor(X_calce[:, :, x_slice])).squeeze(-1).numpy()
        return raw * y_std + y_mean

    pred_vlstm = load_and_predict("vlstm_soh_expanded", VLSTM(input_size=1, hidden_size=32, n_targets=1), slice(0, 1))
    pred_cnnlstm = load_and_predict("cnn_lstm_soh_expanded", CNNLSTM())
    pred_piformer = load_and_predict("piformer_soh_expanded", PiFormer())
    print("[calce-exp] deep model predictions done (expanded-pool weights)")

    encoder = ICAEncoder(in_channels=3, embed_dim=16)
    encoder.load_state_dict(torch.load(ROOT / "models" / "ica_encoder_expanded.pt"))
    encoder.eval()
    with torch.no_grad():
        fusion_emb = encoder.encode(torch.tensor(X_calce[:, :, ICA_CHANNEL_SLICE])).numpy()
    print(f"[calce-exp] fusion embeddings extracted: {fusion_emb.shape}")

    hi_df = pd.read_parquet(PROC_DIR / "hi_table_expanded.parquet")
    hi_df = hi_df[hi_df["dataset"] == "CALCE"]
    seq_df = pd.DataFrame({"battery_id": bid_calce, "cycle_idx": cyc_calce})
    for i in range(16):
        seq_df[f"fusion_{i}"] = fusion_emb[:, i]
    seq_df["pred_VLSTM"] = pred_vlstm
    seq_df["pred_CNNLSTM"] = pred_cnnlstm
    seq_df["pred_PiFormer"] = pred_piformer

    merged = pd.merge(hi_df, seq_df, on=["battery_id", "cycle_idx"], how="inner")
    print(f"[calce-exp] hi_table CALCE rows={len(hi_df)}, sequence-model rows={len(seq_df)}, "
          f"merged rows={len(merged)}")

    train_hi_df = pd.read_parquet(PROC_DIR / "hi_table_expanded.parquet")
    train_hi_df = train_hi_df[train_hi_df["dataset"].isin(["NASA", "MIT"])]
    train_medians = train_hi_df[bfa_selected].median(numeric_only=True)
    fusion_cols = [f"fusion_{i}" for i in range(16)]
    feature_cols = bfa_selected + fusion_cols

    X_feat = merged[feature_cols].to_numpy(dtype=float, copy=True)
    for j, col in enumerate(bfa_selected):
        col_nan = np.isnan(X_feat[:, j])
        if col_nan.any():
            X_feat[col_nan, j] = train_medians[col]

    xgb_fusion = XGBRegressor()
    xgb_fusion.load_model(str(ROOT / "models" / "xgb_soh_fusion_expanded.json"))
    pred_xgb_fusion = xgb_fusion.predict(X_feat)

    with open(ROOT / "models" / "ridge_meta_fusion_expanded.pkl", "rb") as f:
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
        print(f"[calce-exp] {name}: RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")
        return {"model": name, "rmse": rmse, "mae": mae, "r2": r2, "n": len(y_true)}

    print("\n[calce-exp] === ZERO-RETRAIN RESULTS (expanded-pool model, out-of-domain CALCE) ===")
    results = [
        metrics("XGBoost-fusion-expanded (CALCE, zero-retrain)", pred_xgb_fusion),
        metrics("Stacking-Ridge-fusion-expanded (CALCE, zero-retrain)", pred_ensemble),
    ]
    print("\n[calce-exp] ORIGINAL 32-battery reference: XGBoost-fusion R2=0.304, "
          "Stacking-Ridge-fusion R2=0.314 (both RMSE ~17.8-18.0) - THIS is the number "
          "being directly tested for improvement.")

    pd.DataFrame(results).to_csv(PRED_DIR / "calce_zero_retrain_expanded_metrics.csv", index=False)
    merged_out = merged[["dataset", "battery_id", "cycle_idx", "SOH", "RUL"]].copy()
    merged_out["pred_XGBoost_fusion_expanded"] = pred_xgb_fusion
    merged_out["pred_Stacking_Ridge_fusion_expanded"] = pred_ensemble
    merged_out.to_csv(PRED_DIR / "calce_zero_retrain_expanded_preds.csv", index=False)

    print("\n[calce-exp] === Conformal interval check on the expanded-pool model ===")
    ens_test_fusion = pd.read_csv(PRED_DIR / "ensemble_fusion_expanded_test_preds.csv")
    calib_ids, eval_ids = calib_eval_battery_split(ens_test_fusion["battery_id"].unique().tolist())
    calib_df = ens_test_fusion[ens_test_fusion["battery_id"].isin(calib_ids)]
    eval_df = ens_test_fusion[ens_test_fusion["battery_id"].isin(eval_ids)]
    print(f"[calce-exp] conformal calib battery count={len(calib_ids)}, eval battery count={len(eval_ids)} "
          f"(original 32-battery run had 3 calib / 3 eval)")

    r_eval, lo_eval, hi_eval = evaluate_coverage(
        "SOH in-domain (NASA+MIT eval, expanded fusion ensemble)",
        calib_df["pred_Stacking_Ridge_fusion_expanded"].to_numpy(), calib_df["SOH"].to_numpy(),
        eval_df["pred_Stacking_Ridge_fusion_expanded"].to_numpy(), eval_df["SOH"].to_numpy(),
    )
    half_width_in_domain = float(np.mean(hi_eval - lo_eval)) / 2

    r_calce, lo_calce, hi_calce = evaluate_coverage(
        "SOH out-of-domain (CALCE, zero-retrain, expanded-pool model, SAME calibration)",
        calib_df["pred_Stacking_Ridge_fusion_expanded"].to_numpy(), calib_df["SOH"].to_numpy(),
        pred_ensemble, y_true,
    )
    half_width_calce = float(np.mean(hi_calce - lo_calce)) / 2

    print(f"\n[calce-exp] in-domain (NASA+MIT eval) half-width={half_width_in_domain:.3f}, "
          f"coverage={r_eval['empirical_coverage']:.3f}")
    print(f"[calce-exp] out-of-domain (CALCE) half-width={half_width_calce:.3f}, "
          f"coverage={r_calce['empirical_coverage']:.3f}")
    print("[calce-exp] ORIGINAL 32-battery reference: in-domain half-width=2.367 "
          "coverage=0.956; CALCE half-width=2.367 (IDENTICAL) coverage=0.061")

    pd.DataFrame([
        {"domain": "NASA+MIT (in-domain, expanded pool)", "half_width": half_width_in_domain,
         "empirical_coverage": r_eval["empirical_coverage"], "target_coverage": 0.9},
        {"domain": "CALCE (out-of-domain, zero-retrain, expanded pool)", "half_width": half_width_calce,
         "empirical_coverage": r_calce["empirical_coverage"], "target_coverage": 0.9},
    ]).to_csv(OUT_DIR / "calce_zero_retrain_expanded_conformal.csv", index=False)

    print("\n[calce-exp] DONE")


if __name__ == "__main__":
    main()
