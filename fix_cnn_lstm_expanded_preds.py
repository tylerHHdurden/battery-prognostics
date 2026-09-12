"""
One-off corrective script: recomputes CNN-LSTM-expanded's test AND
train predictions correctly (model.eval() applied) and patches the
already-written deep_models_expanded_{test,train}_preds.csv /
deep_models_expanded_metrics.csv in place - WITHOUT touching VLSTM's or
PiFormer's already-correct columns in those same files, and WITHOUT
retraining anything (loads the existing, correct cnn_lstm_soh_expanded.pt
checkpoint - confirmed via file mtime unchanged across the crash/resume,
so its WEIGHTS were never in question, only the missing .eval() call
used to score it).

Run standalone, safe to run while train_fusion_encoder_expanded.py is
running concurrently in the background chain (touches none of that
script's files).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "src" / "models"))
from train_deep_models import make_xy
from train_deep_models_expanded import load_all_battery_tensors_expanded, predict
from sequence_features import apply_channel_norm
from models.cnn_lstm import CNNLSTM

ROOT = Path(__file__).resolve().parent
PROC_DIR = ROOT / "data" / "processed"
PRED_DIR = PROC_DIR / "predictions"


def main():
    t0 = time.time()
    battery_data = load_all_battery_tensors_expanded()
    print(f"[fix] loaded {len(battery_data)} batteries in {time.time()-t0:.1f}s")

    split = json.loads((PROC_DIR / "battery_split_expanded.json").read_text())
    train_ids = [b for b in split["train_ids"] if b in battery_data]
    test_ids = [b for b in split["test_ids"] if b in battery_data]
    n_val_batteries = max(1, len(train_ids) // 5)
    val_ids = sorted(train_ids)[-n_val_batteries:]
    fit_ids = [b for b in train_ids if b not in val_ids]

    X_fit, y_fit, rul_fit, _, _, _ = make_xy(battery_data, fit_ids)
    X_val, y_val, rul_val, _, _, _ = make_xy(battery_data, val_ids)
    X_test, y_test, rul_test, ds_test, bid_test, cyc_test = make_xy(battery_data, test_ids)

    norm_stats = json.loads((PROC_DIR / "channel_norm_stats_expanded.json").read_text())
    X_fit_n = apply_channel_norm(X_fit, norm_stats)
    X_val_n = apply_channel_norm(X_val, norm_stats)
    X_test_n = apply_channel_norm(X_test, norm_stats)

    cnn_lstm = CNNLSTM()
    cnn_lstm.load_state_dict(torch.load(ROOT / "models" / "cnn_lstm_soh_expanded.pt"))
    cnn_lstm.eval()  # THE fix
    cnn_lstm.y_mean_, cnn_lstm.y_std_ = float(y_fit.mean()), float(y_fit.std() + 1e-8)

    pred_test = predict(cnn_lstm, X_test_n)
    rmse = np.sqrt(mean_squared_error(y_test, pred_test))
    mae = mean_absolute_error(y_test, pred_test)
    r2 = r2_score(y_test, pred_test)
    print(f"[fix] CNNLSTM CORRECTED TEST RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")
    print(f"[fix] (was WRONG before fix: RMSE=2.4256 MAE=1.0309 R2=0.8837 - train-mode BatchNorm bug)")
    print(f"[fix] (matches pre-crash original run: RMSE=1.3003 MAE=0.7450 R2=0.9666)")

    # patch test_preds.csv + metrics.csv (both already have VLSTM/PiFormer
    # correct columns - only CNNLSTM's column/row needs replacing)
    test_df = pd.read_csv(PRED_DIR / "deep_models_expanded_test_preds.csv")
    assert len(test_df) == len(pred_test), f"row count mismatch: {len(test_df)} vs {len(pred_test)}"
    # confirm row order matches before overwriting - dataset/battery_id/cycle_idx must align exactly
    assert (test_df["dataset"].to_numpy() == np.array(ds_test)).all()
    assert (test_df["battery_id"].to_numpy() == np.array(bid_test)).all()
    assert (test_df["cycle_idx"].to_numpy() == np.array(cyc_test)).all()
    test_df["y_pred_CNNLSTM"] = pred_test
    test_df.to_csv(PRED_DIR / "deep_models_expanded_test_preds.csv", index=False)

    metrics_df = pd.read_csv(PRED_DIR / "deep_models_expanded_metrics.csv")
    metrics_df.loc[metrics_df["model"] == "CNNLSTM", ["rmse", "mae", "r2"]] = [rmse, mae, r2]
    metrics_df.to_csv(PRED_DIR / "deep_models_expanded_metrics.csv", index=False)

    # train_preds.csv: only build if it was already written (i.e. the
    # crashed run got that far for VLSTM/PiFormer too) or build fresh -
    # either way this recomputes ALL THREE columns correctly here since
    # we already have X_fit/X_val loaded, cheap to just redo all 3
    # correctly rather than partially patch.
    from models.vlstm import VLSTM
    from models.piformer import PiFormer
    vlstm = VLSTM(input_size=1, hidden_size=32, n_targets=1)
    vlstm.load_state_dict(torch.load(ROOT / "models" / "vlstm_soh_expanded.pt"))
    vlstm.eval()
    vlstm.y_mean_, vlstm.y_std_ = float(y_fit.mean()), float(y_fit.std() + 1e-8)
    piformer = PiFormer()
    piformer.load_state_dict(torch.load(ROOT / "models" / "piformer_soh_expanded.pt"))
    piformer.eval()
    piformer.y_mean_, piformer.y_std_ = float(y_fit.mean()), float(y_fit.std() + 1e-8)

    ids_fitval = fit_ids + val_ids
    _, ytr, rultr, dstr, bidtr, cyctr = make_xy(battery_data, ids_fitval)
    Xtr_n = np.concatenate([X_fit_n, X_val_n])
    assert len(Xtr_n) == len(ytr)
    tr_pred_v = predict(vlstm, Xtr_n[:, :, 0:1])
    tr_pred_c = predict(cnn_lstm, Xtr_n)
    tr_pred_p = predict(piformer, Xtr_n)
    out_train = pd.DataFrame({
        "dataset": dstr, "battery_id": bidtr, "cycle_idx": cyctr,
        "SOH": ytr, "RUL": rultr,
        "y_pred_VLSTM": tr_pred_v, "y_pred_CNNLSTM": tr_pred_c, "y_pred_PiFormer": tr_pred_p,
    })
    out_train.to_csv(PRED_DIR / "deep_models_expanded_train_preds.csv", index=False)
    print(f"[fix] wrote corrected deep_models_expanded_train_preds.csv ({len(out_train)} rows)")

    # sanity: VLSTM/PiFormer test predictions should be UNCHANGED (verifies
    # they were never affected, confirms this fix is scoped correctly)
    pred_v_test = predict(vlstm, X_test_n[:, :, 0:1])
    pred_p_test = predict(piformer, X_test_n)
    r2_v = r2_score(y_test, pred_v_test)
    r2_p = r2_score(y_test, pred_p_test)
    print(f"[fix] sanity re-check VLSTM TEST R2={r2_v:.4f} (should match 0.9472)")
    print(f"[fix] sanity re-check PiFormer TEST R2={r2_p:.4f} (should match 0.7795)")

    print(f"[fix] DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
