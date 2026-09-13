"""
Stage 0, Check 0.1: out-of-fold stacking verification.

CONFIRMED BEFORE WRITING ANY NEW CODE (inspected train_ensemble.py,
train_xgboost.py, train_deep_models.py directly - not inferred):
Phase 3's Ridge meta-learner WAS fit on in-sample base-learner
predictions. train_xgboost.py's "train" predictions are
`model.predict(X[train_mask])` - the same rows XGBoost was just fit on.
train_deep_models.py's "train" predictions are computed from
`Xtr = np.concatenate([X_fit, X_val])` - the exact fit+val data each
deep model was trained/early-stopped on. train_ensemble.py's
`load_merged("train")` reads these two files directly as the Ridge
meta-learner's training data. This IS the leakage class the task
description hypothesizes, on the original 32-battery pool, for the
project's original (non-fusion) Phase 3 ensemble specifically - the
one whose coefficients are quoted in this check's own instructions.

THIS SCRIPT re-fits genuine out-of-fold meta-features: GroupKFold
(grouped by battery, so no cycle from any battery ever appears in both
a fold's "training" and "held-out" side) over the 26 TRAIN battery IDs
only. For each fold, all 4 base learners are retrained from scratch on
the OTHER folds' batteries (with their own inner fit/val carve-out for
the 3 deep models' early stopping, identical 80/20 rule to the
original), then used to predict on the held-out fold - genuinely
never-seen-by-those-specific-models data. Concatenating all 5 folds'
held-out predictions gives an honest meta-learner training set.

TEST-set predictions are NOT recomputed - the existing xgb_preds.csv/
deep_models_test_preds.csv test-split rows were already genuinely
out-of-sample (from models trained only on the 26 TRAIN batteries,
evaluated on the 6 held-out TEST batteries) - reusing them is correct,
not a shortcut, and saves real compute.

Resilience: per-fold checkpointing (skip an already-completed fold on
restart) PLUS per-epoch checkpointing within each fold/model (this
project's now-standard resumable-training pattern, verified
bit-for-bit correct in earlier sessions) - this is a multi-hour job
(5 folds x 3 deep models each) run on a machine with a documented
history of external session teardowns mid-training.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
from train_deep_models import load_all_battery_tensors, make_xy
from sequence_features import compute_channel_norm_stats, apply_channel_norm
from models.vlstm import VLSTM
from models.cnn_lstm import CNNLSTM
from models.piformer import PiFormer

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
PRED_DIR = PROC_DIR / "predictions"
OUT_DIR = ROOT / "outputs"

N_FOLDS = 5
BASE_COLS = ["pred_XGBoost", "pred_VLSTM", "pred_CNNLSTM", "pred_PiFormer"]
_OOF_FOLD_DIR = PROC_DIR / "_oof_check_folds"
_OOF_FOLD_DIR.mkdir(exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)


def train_one_model_resumable(name, model, X_train, y_train, X_val, y_val, ckpt_path,
                               epochs=40, batch_size=64, lr=1e-3, patience=8):
    """Identical logic/hyperparameters to train_deep_models.train_one_model
    (which stays untouched, still used everywhere else) - per-epoch
    checkpoint/resume added, same verified pattern as
    train_cnn_bigru_expanded.py / train_piformer_huber_expanded.py."""
    device = torch.device("cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["opt_state"])
        torch.set_rng_state(ckpt["torch_rng_state"])
        y_mean, y_std = ckpt["y_mean"], ckpt["y_std"]
        best_val, best_state, patience_ctr = ckpt["best_val"], ckpt["best_state"], ckpt["patience_ctr"]
        history = ckpt["history"]
        start_epoch = ckpt["epoch"] + 1
        print(f"[oof/{name}] RESUMING from epoch {start_epoch} ({ckpt_path.name})")
    else:
        y_mean, y_std = float(y_train.mean()), float(y_train.std() + 1e-8)
        best_val, best_state, patience_ctr, history, start_epoch = np.inf, None, 0, [], 0

    model.y_mean_, model.y_std_ = y_mean, y_std
    Xt = torch.tensor(X_train)
    yt = torch.tensor((y_train - y_mean) / y_std).unsqueeze(-1)
    Xv = torch.tensor(X_val)
    yv = torch.tensor((y_val - y_mean) / y_std).unsqueeze(-1)
    n = len(Xt)

    for epoch in range(start_epoch, epochs):
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
        print(f"[oof/{name}] epoch {epoch:3d} train_mse={epoch_loss:.4f} val_mse={val_loss:.4f}")

        stop = False
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"[oof/{name}] early stopping at epoch {epoch}")
                stop = True

        torch.save({
            "model_state": model.state_dict(), "opt_state": opt.state_dict(),
            "torch_rng_state": torch.get_rng_state(), "y_mean": y_mean, "y_std": y_std,
            "best_val": best_val, "best_state": best_state, "patience_ctr": patience_ctr,
            "history": history, "epoch": epoch,
        }, ckpt_path)
        if stop:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt_path.unlink(missing_ok=True)
    return model


def predict_deep(model, X):
    model.eval()
    with torch.no_grad():
        raw = model(torch.tensor(X)).squeeze(-1).numpy()
    return raw * model.y_std_ + model.y_mean_


def run_fold(fold_idx, other_ids, held_ids, battery_data, hi_df, bfa_selected):
    fold_out_path = _OOF_FOLD_DIR / f"fold_{fold_idx}_preds.csv"
    if fold_out_path.exists():
        print(f"[oof] fold {fold_idx}: already completed, loading from disk")
        return pd.read_csv(fold_out_path)

    t0 = time.time()
    other_ids_sorted = sorted(other_ids)
    n_inner_val = max(1, len(other_ids_sorted) // 5)
    inner_val_ids = other_ids_sorted[-n_inner_val:]
    inner_fit_ids = [b for b in other_ids_sorted if b not in inner_val_ids]
    print(f"\n[oof] === fold {fold_idx}: held-out={sorted(held_ids)} ===")
    print(f"[oof] fold {fold_idx}: inner_fit={len(inner_fit_ids)} battteries, "
          f"inner_val={len(inner_val_ids)} batteries")

    # --- XGBoost: fit on ALL "other" rows (original script never used a
    # separate val split for XGBoost - no early stopping there either) ---
    hi_other = hi_df[hi_df["battery_id"].isin(other_ids)]
    hi_held = hi_df[hi_df["battery_id"].isin(held_ids)]
    X_hi_other = hi_other[bfa_selected].to_numpy(dtype=float, copy=True)
    col_medians = np.nanmedian(X_hi_other, axis=0)
    inds = np.where(np.isnan(X_hi_other))
    X_hi_other[inds] = np.take(col_medians, inds[1])
    y_hi_other = hi_other["SOH"].to_numpy(dtype=float)

    X_hi_held = hi_held[bfa_selected].to_numpy(dtype=float, copy=True)
    inds_h = np.where(np.isnan(X_hi_held))
    X_hi_held[inds_h] = np.take(col_medians, inds_h[1])

    xgb = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                        subsample=0.8, colsample_bytree=0.8, random_state=42,
                        n_jobs=-1, reg_lambda=1.0)
    xgb.fit(X_hi_other, y_hi_other)
    pred_xgb_held = xgb.predict(X_hi_held)
    print(f"[oof] fold {fold_idx}: XGBoost trained+predicted ({time.time()-t0:.1f}s so far)")

    # --- Deep models: inner fit/val carve, then predict on held-out fold ---
    X_fit, y_fit, _, _, _, _ = make_xy(battery_data, inner_fit_ids)
    X_val, y_val, _, _, _, _ = make_xy(battery_data, inner_val_ids)
    X_held, y_held, rul_held, ds_held, bid_held, cyc_held = make_xy(battery_data, sorted(held_ids))

    norm_stats = compute_channel_norm_stats(X_fit)  # fit on THIS fold's inner_fit only
    X_fit = apply_channel_norm(X_fit, norm_stats)
    X_val = apply_channel_norm(X_val, norm_stats)
    X_held = apply_channel_norm(X_held, norm_stats)

    vlstm = VLSTM(input_size=1, hidden_size=32, n_targets=1)
    vlstm = train_one_model_resumable(
        f"fold{fold_idx}-VLSTM", vlstm, X_fit[:, :, 0:1], y_fit, X_val[:, :, 0:1], y_val,
        _OOF_FOLD_DIR / f"_ckpt_fold{fold_idx}_vlstm.pt")
    pred_vlstm_held = predict_deep(vlstm, X_held[:, :, 0:1])

    cnn_lstm = CNNLSTM()
    cnn_lstm = train_one_model_resumable(
        f"fold{fold_idx}-CNNLSTM", cnn_lstm, X_fit, y_fit, X_val, y_val,
        _OOF_FOLD_DIR / f"_ckpt_fold{fold_idx}_cnnlstm.pt")
    pred_cnnlstm_held = predict_deep(cnn_lstm, X_held)

    piformer = PiFormer()
    piformer = train_one_model_resumable(
        f"fold{fold_idx}-PiFormer", piformer, X_fit, y_fit, X_val, y_val,
        _OOF_FOLD_DIR / f"_ckpt_fold{fold_idx}_piformer.pt")
    pred_piformer_held = predict_deep(piformer, X_held)

    out = pd.DataFrame({
        "dataset": ds_held, "battery_id": bid_held, "cycle_idx": cyc_held, "SOH": y_held,
        "pred_XGBoost": np.nan, "pred_VLSTM": pred_vlstm_held,
        "pred_CNNLSTM": pred_cnnlstm_held, "pred_PiFormer": pred_piformer_held,
    })
    # XGBoost predictions merged in on (battery_id, cycle_idx) since hi_table's
    # cycle indexing can differ slightly in row count from make_xy's - inner join
    xgb_df = hi_held[["battery_id", "cycle_idx"]].copy()
    xgb_df["pred_XGBoost"] = pred_xgb_held
    out = out.drop(columns=["pred_XGBoost"])
    out = pd.merge(out, xgb_df, on=["battery_id", "cycle_idx"], how="inner")

    out.to_csv(fold_out_path, index=False)
    print(f"[oof] fold {fold_idx}: DONE in {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min), "
          f"{len(out)} held-out rows saved to {fold_out_path.name}")
    return out


def metrics(y_true, y_pred):
    return {"rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred))}


def main():
    t_start = time.time()
    with open(PROC_DIR / "bfa_selected_features.txt") as f:
        bfa_selected = [l.strip() for l in f if l.strip()]
    hi_df = pd.read_parquet(PROC_DIR / "hi_table.parquet")
    hi_df = hi_df[hi_df["dataset"].isin(["NASA", "MIT"])].reset_index(drop=True)

    split = json.loads((PROC_DIR / "battery_split.json").read_text())
    train_ids = split["train_ids"]
    test_ids = split["test_ids"]
    print(f"[oof] {len(train_ids)} train batteries, {len(test_ids)} test batteries (original 32-battery pool)")

    battery_data = load_all_battery_tensors()

    groups = np.array(sorted(train_ids))
    gkf = GroupKFold(n_splits=N_FOLDS)
    dummy_X = np.zeros((len(groups), 1))
    fold_frames = []
    for fold_idx, (other_idx, held_idx) in enumerate(gkf.split(dummy_X, groups=groups)):
        other_ids = set(groups[other_idx])
        held_ids = set(groups[held_idx])
        fold_frames.append(run_fold(fold_idx, other_ids, held_ids, battery_data, hi_df, bfa_selected))

    oof_df = pd.concat(fold_frames, ignore_index=True)
    print(f"\n[oof] === ALL {N_FOLDS} FOLDS DONE in {(time.time()-t_start)/60:.1f} min total ===")
    print(f"[oof] genuine out-of-fold training rows: {len(oof_df)}")

    # --- Load EXISTING (already legitimately out-of-sample) test predictions ---
    xgb_test = pd.read_csv(PRED_DIR / "xgb_preds.csv")
    xgb_test = xgb_test[xgb_test["split"] == "test"][
        ["dataset", "battery_id", "cycle_idx", "SOH", "y_pred_soh"]
    ].rename(columns={"y_pred_soh": "pred_XGBoost"})
    deep_test = pd.read_csv(PRED_DIR / "deep_models_test_preds.csv").rename(columns={
        "y_pred_VLSTM": "pred_VLSTM", "y_pred_CNNLSTM": "pred_CNNLSTM", "y_pred_PiFormer": "pred_PiFormer",
    })
    test_df = pd.merge(xgb_test, deep_test[["dataset", "battery_id", "cycle_idx",
                        "pred_VLSTM", "pred_CNNLSTM", "pred_PiFormer"]],
                        on=["dataset", "battery_id", "cycle_idx"], how="inner")
    print(f"[oof] reusing existing test predictions (already genuinely out-of-sample): {len(test_df)} rows")

    # --- Refit Ridge on genuine OOF meta-features ---
    X_oof, y_oof = oof_df[BASE_COLS].to_numpy(), oof_df["SOH"].to_numpy()
    X_test, y_test = test_df[BASE_COLS].to_numpy(), test_df["SOH"].to_numpy()

    ridge_oof = Ridge(alpha=1.0).fit(X_oof, y_oof)
    pred_oof_test = ridge_oof.predict(X_test)
    m_oof = metrics(y_test, pred_oof_test)
    print(f"\n[oof] === NEW (genuine out-of-fold) Ridge meta-learner ===")
    print(f"[oof] coefficients: {dict(zip(BASE_COLS, ridge_oof.coef_.round(4)))}, "
          f"intercept={ridge_oof.intercept_:.4f}")
    print(f"[oof] TEST metrics: {m_oof}")

    # --- Original (in-sample) Ridge, refit here identically for a clean side-by-side ---
    orig_train = pd.read_csv(PRED_DIR / "xgb_preds.csv")
    orig_train = orig_train[orig_train["split"] == "train"][
        ["dataset", "battery_id", "cycle_idx", "SOH", "y_pred_soh"]
    ].rename(columns={"y_pred_soh": "pred_XGBoost"})
    orig_deep_train = pd.read_csv(PRED_DIR / "deep_models_train_preds.csv").rename(columns={
        "y_pred_VLSTM": "pred_VLSTM", "y_pred_CNNLSTM": "pred_CNNLSTM", "y_pred_PiFormer": "pred_PiFormer",
    })
    orig_train_df = pd.merge(orig_train, orig_deep_train[["dataset", "battery_id", "cycle_idx",
                              "pred_VLSTM", "pred_CNNLSTM", "pred_PiFormer"]],
                              on=["dataset", "battery_id", "cycle_idx"], how="inner")
    ridge_insample = Ridge(alpha=1.0).fit(orig_train_df[BASE_COLS], orig_train_df["SOH"])
    pred_insample_test = ridge_insample.predict(X_test)
    m_insample = metrics(y_test, pred_insample_test)
    print(f"\n[oof] === ORIGINAL (in-sample) Ridge meta-learner, re-confirmed here ===")
    print(f"[oof] coefficients: {dict(zip(BASE_COLS, ridge_insample.coef_.round(4)))}, "
          f"intercept={ridge_insample.intercept_:.4f}")
    print(f"[oof] TEST metrics: {m_insample}")

    # --- Drop-branch ablation, BOTH versions, side by side ---
    print(f"\n[oof] === DROP-BRANCH ABLATION (plain 4-branch, non-fusion) ===")
    ablation_rows = []
    for label, train_meta, ridge_kind in [
        ("in-sample (original)", orig_train_df, "insample"),
        ("out-of-fold (corrected)", oof_df, "oof"),
    ]:
        full_ridge = Ridge(alpha=1.0).fit(train_meta[BASE_COLS], train_meta["SOH"])
        full_pred = full_ridge.predict(test_df[BASE_COLS])
        m_full = metrics(y_test, full_pred)
        ablation_rows.append({"stacking": label, "variant": "full_4_branch", "dropped": "none", **m_full})
        for dropped in BASE_COLS:
            cols = [c for c in BASE_COLS if c != dropped]
            r = Ridge(alpha=1.0).fit(train_meta[cols], train_meta["SOH"])
            p = r.predict(test_df[cols])
            m = metrics(y_test, p)
            ablation_rows.append({"stacking": label, "variant": f"drop_{dropped}",
                                   "dropped": dropped, **m})
        print(f"[oof] [{label}] full: {m_full}")

    ablation_df = pd.DataFrame(ablation_rows)
    print("\n" + ablation_df.to_string(index=False))

    # --- explicit side-by-side summary ---
    print(f"\n[oof] === SIDE-BY-SIDE SUMMARY ===")
    print(f"[oof] Original in-sample coefficients:  {dict(zip(BASE_COLS, ridge_insample.coef_.round(4)))}")
    print(f"[oof] New out-of-fold coefficients:      {dict(zip(BASE_COLS, ridge_oof.coef_.round(4)))}")
    print(f"[oof] Original in-sample TEST R2:  {m_insample['r2']:.4f}")
    print(f"[oof] New out-of-fold TEST R2:      {m_oof['r2']:.4f}")

    oof_df.to_csv(PRED_DIR / "oof_stacking_check_meta_features.csv", index=False)
    ablation_df.to_csv(OUT_DIR / "oof_stacking_check_ablation.csv", index=False)
    pd.DataFrame([
        {"stacking": "in-sample (original)", **{f"coef_{k}": v for k, v in zip(BASE_COLS, ridge_insample.coef_)},
         "intercept": ridge_insample.intercept_, **m_insample},
        {"stacking": "out-of-fold (corrected)", **{f"coef_{k}": v for k, v in zip(BASE_COLS, ridge_oof.coef_)},
         "intercept": ridge_oof.intercept_, **m_oof},
    ]).to_csv(OUT_DIR / "oof_stacking_check_coefficients.csv", index=False)

    print(f"\n[oof] TOTAL WALL TIME: {(time.time()-t_start)/60:.1f} min")
    print("[oof] DONE")


if __name__ == "__main__":
    main()
