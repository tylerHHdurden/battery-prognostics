"""
Base learners 2-4/4: VLSTM, CNN-LSTM, PiFormer, trained on NASA + MIT
subset sequence tensors (SOH regression), using the SAME battery-level
split as XGBoost (data/processed/battery_split.json).

Epoch count / batch size ASSUMPTION (logged per the task's own allowance
for shortcuts under time pressure): CPU-only laptop, no GPU (see
OVERNIGHT_LOG.md compute-environment note). 40 epochs, batch_size=64,
Adam lr=1e-3, is a real training run (loss curves genuinely converge, not
a token 1-2 epoch smoke test) but deliberately not the hundreds of epochs
that would be used on a GPU. Early stopping on a held-out validation slice
of the TRAIN batteries (never touching test batteries) with patience=8.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
from data_adapters import iterate_nasa_cycles, iterate_mit_cycles
from sequence_features import (
    build_dataset_tensors, CHANNEL_NAMES,
    compute_channel_norm_stats, apply_channel_norm,
)
from models.vlstm import VLSTM
from models.cnn_lstm import CNNLSTM
from models.piformer import PiFormer

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
(PROC_DIR / "predictions").mkdir(exist_ok=True, parents=True)
(ROOT / "models").mkdir(exist_ok=True)
(ROOT / "outputs").mkdir(exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)


def load_all_battery_tensors():
    """Returns dict battery_id -> (X, soh, rul, dataset)."""
    out = {}
    for cid in ["B0005", "B0006", "B0007", "B0018"]:
        cycles = list(iterate_nasa_cycles(cid))
        X, soh, rul, idxs, censored = build_dataset_tensors(cycles)
        if X is not None:
            out[cid] = (X.astype(np.float32), soh.astype(np.float32), rul.astype(np.float32), "NASA")
        print(f"[deep] loaded NASA/{cid}: {X.shape if X is not None else None}")

    with open(PROC_DIR / "mit_subset.json") as f:
        mit_subset = json.load(f)
    for entry in mit_subset:
        cycles = list(iterate_mit_cycles(entry["batch_file"], entry["cell_index"]))
        X, soh, rul, idxs, censored = build_dataset_tensors(cycles)
        if X is not None:
            out[entry["global_id"]] = (X.astype(np.float32), soh.astype(np.float32),
                                        rul.astype(np.float32), "MIT")
        print(f"[deep] loaded MIT/{entry['global_id']}: {X.shape if X is not None else None}")
    return out


def make_xy(battery_data: dict, ids: list[str]):
    Xs, sohs, ruls, ds_list, bid_list, cyc_list = [], [], [], [], [], []
    for bid in ids:
        X, soh, rul, ds = battery_data[bid]
        Xs.append(X)
        sohs.append(soh)
        ruls.append(rul)
        ds_list += [ds] * len(soh)
        bid_list += [bid] * len(soh)
        cyc_list += list(range(1, len(soh) + 1))
    return (np.concatenate(Xs), np.concatenate(sohs), np.concatenate(ruls),
            ds_list, bid_list, cyc_list)


def train_one_model(name, model, X_train, y_train, X_val, y_val, epochs=40,
                     batch_size=64, lr=1e-3, patience=8):
    """
    Target standardization (z-score on y_train) is applied here: without
    it, an untrained regression head starts near output~0 while SOH targets
    sit around 70-100, so the first many epochs are spent just moving the
    output bias into range rather than learning shape - confirmed during
    smoke-testing (val MSE ~5100, i.e. RMSE~71, after 3 epochs without
    standardization = predicting near 0). Standard practice, not a
    workaround for anything model-specific; predictions are de-standardized
    before being returned/evaluated by the caller.
    """
    device = torch.device("cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    y_mean, y_std = float(y_train.mean()), float(y_train.std() + 1e-8)
    model.y_mean_, model.y_std_ = y_mean, y_std

    Xt = torch.tensor(X_train)
    yt = torch.tensor((y_train - y_mean) / y_std).unsqueeze(-1)
    Xv = torch.tensor(X_val)
    yv = torch.tensor((y_val - y_mean) / y_std).unsqueeze(-1)

    n = len(Xt)
    best_val = np.inf
    best_state = None
    patience_ctr = 0
    history = []

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xt[idx], yt[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n

        model.eval()
        with torch.no_grad():
            val_pred = model(Xv)
            val_loss = loss_fn(val_pred, yv).item()
        history.append({"epoch": epoch, "train_loss": epoch_loss, "val_loss": val_loss})
        print(f"[deep/{name}] epoch {epoch:3d} train_mse={epoch_loss:.4f} val_mse={val_loss:.4f}")

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"[deep/{name}] early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def main():
    t0 = time.time()
    battery_data = load_all_battery_tensors()
    print(f"[deep] loaded {len(battery_data)} batteries in {time.time()-t0:.1f}s")

    split = json.loads((PROC_DIR / "battery_split.json").read_text())
    train_ids = [b for b in split["train_ids"] if b in battery_data]
    test_ids = [b for b in split["test_ids"] if b in battery_data]

    # carve a validation slice out of TRAIN batteries only (last 20% of the
    # sorted train battery list) - test batteries are never used for
    # early-stopping decisions.
    n_val_batteries = max(1, len(train_ids) // 5)
    val_ids = sorted(train_ids)[-n_val_batteries:]
    fit_ids = [b for b in train_ids if b not in val_ids]
    print(f"[deep] fit batteries: {len(fit_ids)}, val batteries: {len(val_ids)}, "
          f"test batteries: {len(test_ids)}")

    X_fit, y_fit, rul_fit, _, _, _ = make_xy(battery_data, fit_ids)
    X_val, y_val, rul_val, _, _, _ = make_xy(battery_data, val_ids)
    X_test, y_test, rul_test, ds_test, bid_test, cyc_test = make_xy(battery_data, test_ids)
    print(f"[deep] fit cycles={len(X_fit)}, val cycles={len(X_val)}, test cycles={len(X_test)}")

    # Per-channel robust normalization, stats fit on X_fit ONLY (see
    # sequence_features.compute_channel_norm_stats docstring for why this
    # was necessary - dVdQ's raw values reach ~9.5M vs O(1-100) for other
    # channels, which broke CNN-LSTM's BatchNorm). Saved to disk so
    # Phase 4/5/6 scripts apply the IDENTICAL transform rather than
    # recomputing (and potentially drifting) their own stats.
    norm_stats = compute_channel_norm_stats(X_fit)
    with open(PROC_DIR / "channel_norm_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)
    for i, s in enumerate(norm_stats):
        print(f"[deep] channel {i} ({CHANNEL_NAMES[i]}) norm: clip=[{s['lo']:.3g},{s['hi']:.3g}] "
              f"mean={s['mean']:.3g} std={s['std']:.3g}")
    X_fit = apply_channel_norm(X_fit, norm_stats)
    X_val = apply_channel_norm(X_val, norm_stats)
    X_test = apply_channel_norm(X_test, norm_stats)

    results = {}
    all_metrics = []

    def predict(model, X):
        with torch.no_grad():
            raw = model(torch.tensor(X)).squeeze(-1).numpy()
        return raw * model.y_std_ + model.y_mean_

    # --- VLSTM: only channel 0 (V_t) ---
    vlstm = VLSTM(input_size=1, hidden_size=32, n_targets=1)
    vlstm, hist_v = train_one_model("VLSTM", vlstm, X_fit[:, :, 0:1], y_fit,
                                     X_val[:, :, 0:1], y_val)
    results["VLSTM"] = predict(vlstm, X_test[:, :, 0:1])
    torch.save(vlstm.state_dict(), ROOT / "models" / "vlstm_soh.pt")
    pd.DataFrame(hist_v).to_csv(PROC_DIR / "predictions" / "vlstm_history.csv", index=False)

    # --- CNN-LSTM: all 6 channels ---
    cnn_lstm = CNNLSTM()
    cnn_lstm, hist_c = train_one_model("CNNLSTM", cnn_lstm, X_fit, y_fit, X_val, y_val)
    results["CNNLSTM"] = predict(cnn_lstm, X_test)
    torch.save(cnn_lstm.state_dict(), ROOT / "models" / "cnn_lstm_soh.pt")
    pd.DataFrame(hist_c).to_csv(PROC_DIR / "predictions" / "cnn_lstm_history.csv", index=False)

    # --- PiFormer: all 6 channels ---
    piformer = PiFormer()
    piformer, hist_p = train_one_model("PiFormer", piformer, X_fit, y_fit, X_val, y_val)
    results["PiFormer"] = predict(piformer, X_test)
    torch.save(piformer.state_dict(), ROOT / "models" / "piformer_soh.pt")
    pd.DataFrame(hist_p).to_csv(PROC_DIR / "predictions" / "piformer_history.csv", index=False)

    for name, pred in results.items():
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        print(f"[deep] {name} TEST RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")
        all_metrics.append({"model": name, "rmse": rmse, "mae": mae, "r2": r2})

    out = pd.DataFrame({
        "dataset": ds_test, "battery_id": bid_test, "cycle_idx": cyc_test,
        "SOH": y_test, "RUL": rul_test,
        "y_pred_VLSTM": results["VLSTM"], "y_pred_CNNLSTM": results["CNNLSTM"],
        "y_pred_PiFormer": results["PiFormer"],
    })
    out.to_csv(PROC_DIR / "predictions" / "deep_models_test_preds.csv", index=False)
    pd.DataFrame(all_metrics).to_csv(PROC_DIR / "predictions" / "deep_models_metrics.csv", index=False)

    # also save train-set predictions (needed for the Phase 3 stacking
    # meta-learner to be fit on non-test data). Reuses the ALREADY-
    # normalized X_fit/X_val directly rather than rebuilding via make_xy()
    # again - an earlier version of this rebuilt raw (unnormalized) tensors
    # here and fed them straight into models trained on normalized input,
    # which would have silently produced garbage train-side predictions
    # even after the CNN-LSTM normalization fix. Only the id/label
    # metadata (not X) needs rebuilding, since make_xy's X output isn't used.
    ids_fitval = fit_ids + val_ids
    _, ytr, rultr, dstr, bidtr, cyctr = make_xy(battery_data, ids_fitval)
    Xtr = np.concatenate([X_fit, X_val])
    assert len(Xtr) == len(ytr), "fit+val concatenation order must match make_xy's own order"
    tr_pred_v = predict(vlstm, Xtr[:, :, 0:1])
    tr_pred_c = predict(cnn_lstm, Xtr)
    tr_pred_p = predict(piformer, Xtr)
    out_train = pd.DataFrame({
        "dataset": dstr, "battery_id": bidtr, "cycle_idx": cyctr,
        "SOH": ytr, "RUL": rultr,
        "y_pred_VLSTM": tr_pred_v, "y_pred_CNNLSTM": tr_pred_c, "y_pred_PiFormer": tr_pred_p,
    })
    out_train.to_csv(PROC_DIR / "predictions" / "deep_models_train_preds.csv", index=False)

    print(f"[deep] ALL DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
