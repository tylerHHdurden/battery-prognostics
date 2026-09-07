"""
5th base learner (additive): standalone training of CNN-BiGRU
(src/models/cnn_bigru.py) on the SAME NASA+MIT sequence tensors, SAME
battery-level split, SAME channel normalization, and SAME
train_deep_models.train_one_model training loop (reused unchanged, not
duplicated) as VLSTM/CNN-LSTM/PiFormer - so its RMSE/MAE/R2 is directly
comparable to deep_models_metrics.csv's existing 3 rows.

Fully additive: does not touch train_deep_models.py, its 3 existing
model weights, or deep_models_metrics.csv/deep_models_test_preds.csv -
writes its own parallel `cnn_bigru_*` files.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
from train_deep_models import load_all_battery_tensors, make_xy, train_one_model
from sequence_features import apply_channel_norm
from models.cnn_bigru import CNNBiGRU

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
(PROC_DIR / "predictions").mkdir(exist_ok=True, parents=True)

torch.manual_seed(42)
np.random.seed(42)


def main():
    t0 = time.time()
    battery_data = load_all_battery_tensors()
    split = json.loads((PROC_DIR / "battery_split.json").read_text())
    train_ids = [b for b in split["train_ids"] if b in battery_data]
    test_ids = [b for b in split["test_ids"] if b in battery_data]
    n_val_batteries = max(1, len(train_ids) // 5)
    val_ids = sorted(train_ids)[-n_val_batteries:]
    fit_ids = [b for b in train_ids if b not in val_ids]
    print(f"[cnn-bigru] fit batteries: {len(fit_ids)}, val batteries: {len(val_ids)}, "
          f"test batteries: {len(test_ids)}")

    X_fit, y_fit, rul_fit, _, _, _ = make_xy(battery_data, fit_ids)
    X_val, y_val, rul_val, _, _, _ = make_xy(battery_data, val_ids)
    X_test, y_test, rul_test, ds_test, bid_test, cyc_test = make_xy(battery_data, test_ids)
    print(f"[cnn-bigru] fit cycles={len(X_fit)}, val cycles={len(X_val)}, test cycles={len(X_test)}")

    # REUSES the norm stats already saved by train_deep_models.py (same
    # fit_ids battery set, same computation) rather than recomputing -
    # identical transform as every other deep model in this project.
    norm_stats = json.loads((PROC_DIR / "channel_norm_stats.json").read_text())
    X_fit = apply_channel_norm(X_fit, norm_stats)
    X_val = apply_channel_norm(X_val, norm_stats)
    X_test = apply_channel_norm(X_test, norm_stats)

    def predict(model, X):
        with torch.no_grad():
            raw = model(torch.tensor(X)).squeeze(-1).numpy()
        return raw * model.y_std_ + model.y_mean_

    # --- CNN-BiGRU: all 6 channels, same as CNN-LSTM/PiFormer ---
    cnn_bigru = CNNBiGRU()
    cnn_bigru, hist = train_one_model("CNNBiGRU", cnn_bigru, X_fit, y_fit, X_val, y_val)
    pred_test = predict(cnn_bigru, X_test)
    torch.save(cnn_bigru.state_dict(), ROOT / "models" / "cnn_bigru_soh.pt")
    pd.DataFrame(hist).to_csv(PROC_DIR / "predictions" / "cnn_bigru_history.csv", index=False)

    rmse = np.sqrt(mean_squared_error(y_test, pred_test))
    mae = mean_absolute_error(y_test, pred_test)
    r2 = r2_score(y_test, pred_test)
    print(f"[cnn-bigru] TEST RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")

    # print alongside the existing 4 base learners' numbers for direct
    # comparison, without touching deep_models_metrics.csv itself
    existing = pd.read_csv(PROC_DIR / "predictions" / "deep_models_metrics.csv")
    xgb_metrics = pd.read_csv(PROC_DIR / "predictions" / "xgb_metrics.csv") \
        if (PROC_DIR / "predictions" / "xgb_metrics.csv").exists() else None
    print("\n[cnn-bigru] === base learner comparison (existing 4 + CNN-BiGRU) ===")
    print(existing.to_string(index=False))
    if xgb_metrics is not None:
        print(xgb_metrics.to_string(index=False))
    print(f"      CNNBiGRU  rmse={rmse:.4f}  mae={mae:.4f}  r2={r2:.4f}")

    out = pd.DataFrame({
        "dataset": ds_test, "battery_id": bid_test, "cycle_idx": cyc_test,
        "SOH": y_test, "RUL": rul_test, "y_pred_CNNBiGRU": pred_test,
    })
    out.to_csv(PROC_DIR / "predictions" / "cnn_bigru_test_preds.csv", index=False)
    pd.DataFrame([{"model": "CNNBiGRU", "rmse": rmse, "mae": mae, "r2": r2}]).to_csv(
        PROC_DIR / "predictions" / "cnn_bigru_metrics.csv", index=False
    )

    # also save train-set predictions (needed for the 5-branch drop-branch
    # ablation's meta-learner refit, same rationale as
    # train_deep_models.py's out_train block)
    ids_fitval = fit_ids + val_ids
    _, ytr, rultr, dstr, bidtr, cyctr = make_xy(battery_data, ids_fitval)
    Xtr = np.concatenate([X_fit, X_val])
    assert len(Xtr) == len(ytr), "fit+val concatenation order must match make_xy's own order"
    tr_pred = predict(cnn_bigru, Xtr)
    out_train = pd.DataFrame({
        "dataset": dstr, "battery_id": bidtr, "cycle_idx": cyctr,
        "SOH": ytr, "RUL": rultr, "y_pred_CNNBiGRU": tr_pred,
    })
    out_train.to_csv(PROC_DIR / "predictions" / "cnn_bigru_train_preds.csv", index=False)

    print(f"\n[cnn-bigru] ALL DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
