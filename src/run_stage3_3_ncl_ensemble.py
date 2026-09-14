"""
Stage 3, Item 3.3: Negative Correlation Learning (NCL, Liu & Yao 1999) on
the ensemble's 3 deep base learners (VLSTM, CNN-LSTM, PiFormer).

CONTEXT (Stage 0 Check 0.1, re-confirmed directly here before writing
any new training code - see the freshly-recomputed correlation matrix
below, not a remembered figure): the 4 base learners' genuinely
out-of-fold predictions are highly inter-correlated. Recomputed directly
from data/processed/predictions/oof_stacking_check_meta_features.csv
(Check 0.1's own artifact):
    XGB-VLSTM=0.8355  XGB-CNNLSTM=0.7224  XGB-PiFormer=0.8554
    VLSTM-CNNLSTM=0.7887  VLSTM-PiFormer=0.9198  CNNLSTM-PiFormer=0.8199
All 6 pairwise correlations fall in [0.72, 0.92] - confirms the task's
own cited range exactly, computed fresh rather than trusted from memory.
NCL targets the 3 DEEP learners specifically (VLSTM/CNNLSTM/PiFormer,
pairwise range [0.7887, 0.9198] among themselves) - XGBoost is a
fundamentally different model family (gradient-boosted trees vs. all 3
deep sequence models) and is not part of the correlation-reduction
target; it is retrained unchanged, exactly as Check 0.1 did.

MECHANISM: standard NCL ambiguity-decomposition loss. For each deep
member i, with f_ens = mean of all 3 members' CURRENT predictions
(including i) on this batch:
    L_i = MSE(f_i, y) - lambda * mean((f_i - f_ens)^2)
Summed over the 3 members and backpropagated through all 3 models
TOGETHER in one joint optimizer step per batch (this is the essential
change from Check 0.1's independent per-model training - NCL requires
JOINT training since each member's loss depends on the OTHER members'
current predictions via f_ens). lambda=0.3 (conservative; smoke-tested
on toy data first - confirmed stable, no NaN/exploding loss, diversity
term grows gradually rather than degenerately - before running on real
data). Early stopping/model selection still uses PLAIN validation MSE
(no NCL term), matching Check 0.1's own criterion, so this is judged on
the same bar as the original independent training.

METHODOLOGY: reuses Check 0.1's own GroupKFold(5)-over-26-TRAIN-battery-
IDs out-of-fold protocol EXACTLY (same fold splits via the same
GroupKFold(n_splits=5) call, same inner fit/val carve, same TEST-set
reuse) - "proper out-of-fold stacking per Stage 0.1's established
correct methodology," not the original in-sample-leakage version. Only
the deep-learner TRAINING PROCEDURE differs (joint+NCL vs. independent).

Checkpointed per-fold AND per-epoch-within-fold (same resumable pattern
as run_oof_stacking_check.py) - explicit real-first-attempt risk
acknowledged per instruction: this is the first time this project has
trained 3 deep models jointly with a shared, non-independent loss term.
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
NCL_LAMBDA = 0.3
BASE_COLS = ["pred_XGBoost", "pred_VLSTM", "pred_CNNLSTM", "pred_PiFormer"]
_NCL_FOLD_DIR = PROC_DIR / "_ncl_check_folds"
_NCL_FOLD_DIR.mkdir(exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)


def train_ncl_joint_resumable(vlstm, cnn_lstm, piformer, X_train, y_train, X_val, y_val,
                               ckpt_path, tag, epochs=40, batch_size=64, lr=1e-3, patience=8):
    opt = torch.optim.Adam(
        list(vlstm.parameters()) + list(cnn_lstm.parameters()) + list(piformer.parameters()), lr=lr)

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=False)
        vlstm.load_state_dict(ckpt["vlstm_state"])
        cnn_lstm.load_state_dict(ckpt["cnnlstm_state"])
        piformer.load_state_dict(ckpt["piformer_state"])
        opt.load_state_dict(ckpt["opt_state"])
        torch.set_rng_state(ckpt["torch_rng_state"])
        y_mean, y_std = ckpt["y_mean"], ckpt["y_std"]
        best_val, best_states, patience_ctr = ckpt["best_val"], ckpt["best_states"], ckpt["patience_ctr"]
        history = ckpt["history"]
        start_epoch = ckpt["epoch"] + 1
        print(f"[ncl/{tag}] RESUMING from epoch {start_epoch} ({ckpt_path.name})")
    else:
        y_mean, y_std = float(y_train.mean()), float(y_train.std() + 1e-8)
        best_val, best_states, patience_ctr, history, start_epoch = np.inf, None, 0, [], 0

    Xt = torch.tensor(X_train)
    yt = torch.tensor((y_train - y_mean) / y_std).unsqueeze(-1)
    Xv = torch.tensor(X_val)
    yv = torch.tensor((y_val - y_mean) / y_std).unsqueeze(-1)
    n = len(Xt)

    for epoch in range(start_epoch, epochs):
        vlstm.train(); cnn_lstm.train(); piformer.train()
        perm = torch.randperm(n)
        ep_loss, ep_div = 0.0, 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xt[idx], yt[idx]
            opt.zero_grad()
            f1 = vlstm(xb[:, :, 0:1])
            f2 = cnn_lstm(xb)
            f3 = piformer(xb)
            f_ens = (f1 + f2 + f3) / 3.0
            mse1 = ((f1 - yb) ** 2).mean()
            mse2 = ((f2 - yb) ** 2).mean()
            mse3 = ((f3 - yb) ** 2).mean()
            div1 = ((f1 - f_ens) ** 2).mean()
            div2 = ((f2 - f_ens) ** 2).mean()
            div3 = ((f3 - f_ens) ** 2).mean()
            total = (mse1 - NCL_LAMBDA * div1) + (mse2 - NCL_LAMBDA * div2) + (mse3 - NCL_LAMBDA * div3)
            total.backward()
            opt.step()
            ep_loss += total.item() * len(idx)
            ep_div += (div1.item() + div2.item() + div3.item()) / 3.0 * len(idx)
        ep_loss /= n
        ep_div /= n

        vlstm.eval(); cnn_lstm.eval(); piformer.eval()
        with torch.no_grad():
            v1, v2, v3 = vlstm(Xv[:, :, 0:1]), cnn_lstm(Xv), piformer(Xv)
            val_mse = (((v1 - yv) ** 2).mean() + ((v2 - yv) ** 2).mean() + ((v3 - yv) ** 2).mean()).item() / 3.0
        history.append({"epoch": epoch, "train_total_loss": ep_loss, "train_avg_diversity": ep_div, "val_mse": val_mse})
        print(f"[ncl/{tag}] epoch {epoch:3d} train_total={ep_loss:.4f} avg_div={ep_div:.4f} val_mse={val_mse:.4f}")

        stop = False
        if val_mse < best_val - 1e-4:
            best_val = val_mse
            best_states = {
                "vlstm": {k: v.clone() for k, v in vlstm.state_dict().items()},
                "cnnlstm": {k: v.clone() for k, v in cnn_lstm.state_dict().items()},
                "piformer": {k: v.clone() for k, v in piformer.state_dict().items()},
            }
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"[ncl/{tag}] early stopping at epoch {epoch}")
                stop = True

        torch.save({
            "vlstm_state": vlstm.state_dict(), "cnnlstm_state": cnn_lstm.state_dict(),
            "piformer_state": piformer.state_dict(), "opt_state": opt.state_dict(),
            "torch_rng_state": torch.get_rng_state(), "y_mean": y_mean, "y_std": y_std,
            "best_val": best_val, "best_states": best_states, "patience_ctr": patience_ctr,
            "history": history, "epoch": epoch,
        }, ckpt_path)
        if stop:
            break

    if best_states is not None:
        vlstm.load_state_dict(best_states["vlstm"])
        cnn_lstm.load_state_dict(best_states["cnnlstm"])
        piformer.load_state_dict(best_states["piformer"])
    ckpt_path.unlink(missing_ok=True)
    return vlstm, cnn_lstm, piformer, y_mean, y_std


def predict_deep(model, X, y_mean, y_std, is_vlstm=False):
    model.eval()
    with torch.no_grad():
        Xin = torch.tensor(X[:, :, 0:1]) if is_vlstm else torch.tensor(X)
        raw = model(Xin).squeeze(-1).numpy()
    return raw * y_std + y_mean


def run_fold(fold_idx, other_ids, held_ids, battery_data, hi_df, bfa_selected):
    fold_out_path = _NCL_FOLD_DIR / f"fold_{fold_idx}_preds.csv"
    if fold_out_path.exists():
        print(f"[ncl] fold {fold_idx}: already completed, loading from disk")
        return pd.read_csv(fold_out_path)

    t0 = time.time()
    other_ids_sorted = sorted(other_ids)
    n_inner_val = max(1, len(other_ids_sorted) // 5)
    inner_val_ids = other_ids_sorted[-n_inner_val:]
    inner_fit_ids = [b for b in other_ids_sorted if b not in inner_val_ids]
    print(f"\n[ncl] === fold {fold_idx}: held-out={sorted(held_ids)} ===")

    # --- XGBoost: unchanged, independent, not an NCL target (per this item's own scoping) ---
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
    print(f"[ncl] fold {fold_idx}: XGBoost trained+predicted ({time.time()-t0:.1f}s so far)")

    # --- 3 deep learners: JOINT NCL training, then predict on held-out fold ---
    X_fit, y_fit, _, _, _, _ = make_xy(battery_data, inner_fit_ids)
    X_val, y_val, _, _, _, _ = make_xy(battery_data, inner_val_ids)
    X_held, y_held, rul_held, ds_held, bid_held, cyc_held = make_xy(battery_data, sorted(held_ids))

    norm_stats = compute_channel_norm_stats(X_fit)
    X_fit = apply_channel_norm(X_fit, norm_stats)
    X_val = apply_channel_norm(X_val, norm_stats)
    X_held = apply_channel_norm(X_held, norm_stats)

    vlstm = VLSTM(input_size=1, hidden_size=32, n_targets=1)
    cnn_lstm = CNNLSTM()
    piformer = PiFormer()
    vlstm, cnn_lstm, piformer, y_mean, y_std = train_ncl_joint_resumable(
        vlstm, cnn_lstm, piformer, X_fit, y_fit, X_val, y_val,
        _NCL_FOLD_DIR / f"_ckpt_fold{fold_idx}_ncl.pt", f"fold{fold_idx}")

    pred_vlstm_held = predict_deep(vlstm, X_held, y_mean, y_std, is_vlstm=True)
    pred_cnnlstm_held = predict_deep(cnn_lstm, X_held, y_mean, y_std)
    pred_piformer_held = predict_deep(piformer, X_held, y_mean, y_std)

    out = pd.DataFrame({
        "dataset": ds_held, "battery_id": bid_held, "cycle_idx": cyc_held, "SOH": y_held,
        "pred_VLSTM": pred_vlstm_held, "pred_CNNLSTM": pred_cnnlstm_held, "pred_PiFormer": pred_piformer_held,
    })
    xgb_df = hi_held[["battery_id", "cycle_idx"]].copy()
    xgb_df["pred_XGBoost"] = pred_xgb_held
    out = pd.merge(out, xgb_df, on=["battery_id", "cycle_idx"], how="inner")

    out.to_csv(fold_out_path, index=False)
    print(f"[ncl] fold {fold_idx}: DONE in {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min), "
          f"{len(out)} held-out rows saved")
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
    print(f"[ncl] {len(train_ids)} train batteries, {len(test_ids)} test batteries (original 32-battery pool)")
    print(f"[ncl] NCL_LAMBDA={NCL_LAMBDA}")

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
    print(f"\n[ncl] === ALL {N_FOLDS} FOLDS DONE in {(time.time()-t_start)/60:.1f} min total ===")

    # --- pairwise OOF correlation matrix: NCL vs. original (Check 0.1) ---
    print("\n[ncl] === PAIRWISE OOF CORRELATION MATRIX ===")
    corr_ncl = oof_df[BASE_COLS].corr()
    print("[ncl] NCL-trained (this run):")
    print(corr_ncl.round(4).to_string())
    orig_oof = pd.read_csv(PRED_DIR / "oof_stacking_check_meta_features.csv")
    corr_orig = orig_oof[BASE_COLS].corr()
    print("[ncl] ORIGINAL (Check 0.1, independent training), re-confirmed fresh:")
    print(corr_orig.round(4).to_string())

    deep_cols = ["pred_VLSTM", "pred_CNNLSTM", "pred_PiFormer"]
    pairs = [("VLSTM", "CNNLSTM"), ("VLSTM", "PiFormer"), ("CNNLSTM", "PiFormer")]
    print("\n[ncl] === DEEP-LEARNER PAIRWISE CORRELATION: NCL vs ORIGINAL ===")
    corr_deltas = []
    for a, b in pairs:
        orig_r = corr_orig.loc[f"pred_{a}", f"pred_{b}"]
        ncl_r = corr_ncl.loc[f"pred_{a}", f"pred_{b}"]
        print(f"[ncl] {a}-{b}: original={orig_r:.4f} -> NCL={ncl_r:.4f} (delta={ncl_r-orig_r:+.4f})")
        corr_deltas.append({"pair": f"{a}-{b}", "original": orig_r, "ncl": ncl_r, "delta": ncl_r - orig_r})
    mean_orig = np.mean([d["original"] for d in corr_deltas])
    mean_ncl = np.mean([d["ncl"] for d in corr_deltas])
    print(f"[ncl] MEAN deep-learner pairwise correlation: original={mean_orig:.4f} -> NCL={mean_ncl:.4f}")
    correlation_dropped = mean_ncl < mean_orig - 0.02  # requires a real, non-noise-level drop

    # --- Load EXISTING (already legitimately out-of-sample) test predictions ---
    xgb_test = pd.read_csv(PRED_DIR / "xgb_preds.csv")
    xgb_test = xgb_test[xgb_test["split"] == "test"][
        ["dataset", "battery_id", "cycle_idx", "SOH", "y_pred_soh"]
    ].rename(columns={"y_pred_soh": "pred_XGBoost"})

    # NCL deep-learner TEST predictions: need a model trained on the FULL 26-battery
    # train pool (not just one fold) - train once more, jointly, NCL, on ALL train data,
    # predict on the 6 TEST batteries (standard "final refit on all training data" step,
    # same convention as every other final-model script in this project).
    print("\n[ncl] === Final refit: NCL-trained deep learners on FULL 26-battery train pool ===")
    train_ids_sorted = sorted(train_ids)
    n_val_final = max(1, len(train_ids_sorted) // 5)
    val_ids_final = train_ids_sorted[-n_val_final:]
    fit_ids_final = [b for b in train_ids_sorted if b not in val_ids_final]
    X_fit_f, y_fit_f, _, _, _, _ = make_xy(battery_data, fit_ids_final)
    X_val_f, y_val_f, _, _, _, _ = make_xy(battery_data, val_ids_final)
    X_test_f, y_test_f, rul_test_f, ds_test_f, bid_test_f, cyc_test_f = make_xy(battery_data, sorted(test_ids))
    norm_stats_f = compute_channel_norm_stats(X_fit_f)
    X_fit_f = apply_channel_norm(X_fit_f, norm_stats_f)
    X_val_f = apply_channel_norm(X_val_f, norm_stats_f)
    X_test_f = apply_channel_norm(X_test_f, norm_stats_f)

    vlstm_f = VLSTM(input_size=1, hidden_size=32, n_targets=1)
    cnn_lstm_f = CNNLSTM()
    piformer_f = PiFormer()
    vlstm_f, cnn_lstm_f, piformer_f, y_mean_f, y_std_f = train_ncl_joint_resumable(
        vlstm_f, cnn_lstm_f, piformer_f, X_fit_f, y_fit_f, X_val_f, y_val_f,
        _NCL_FOLD_DIR / "_ckpt_final_ncl.pt", "final")

    pred_vlstm_test = predict_deep(vlstm_f, X_test_f, y_mean_f, y_std_f, is_vlstm=True)
    pred_cnnlstm_test = predict_deep(cnn_lstm_f, X_test_f, y_mean_f, y_std_f)
    pred_piformer_test = predict_deep(piformer_f, X_test_f, y_mean_f, y_std_f)

    torch.save(vlstm_f.state_dict(), ROOT / "models" / "vlstm_ncl.pt")
    torch.save(cnn_lstm_f.state_dict(), ROOT / "models" / "cnn_lstm_ncl.pt")
    torch.save(piformer_f.state_dict(), ROOT / "models" / "piformer_ncl.pt")

    deep_test = pd.DataFrame({
        "dataset": ds_test_f, "battery_id": bid_test_f, "cycle_idx": cyc_test_f,
        "pred_VLSTM": pred_vlstm_test, "pred_CNNLSTM": pred_cnnlstm_test, "pred_PiFormer": pred_piformer_test,
    })
    test_df = pd.merge(xgb_test, deep_test, on=["dataset", "battery_id", "cycle_idx"], how="inner")
    print(f"[ncl] TEST predictions ready: {len(test_df)} rows")

    # --- Refit Ridge on genuine NCL OOF meta-features ---
    X_oof, y_oof = oof_df[BASE_COLS].to_numpy(), oof_df["SOH"].to_numpy()
    X_test, y_test = test_df[BASE_COLS].to_numpy(), test_df["SOH"].to_numpy()

    ridge_ncl = Ridge(alpha=1.0).fit(X_oof, y_oof)
    pred_ncl_test = ridge_ncl.predict(X_test)
    m_ncl = metrics(y_test, pred_ncl_test)
    print(f"\n[ncl] === NCL-trained ensemble, genuine OOF-refit Ridge meta-learner ===")
    print(f"[ncl] coefficients: {dict(zip(BASE_COLS, ridge_ncl.coef_.round(4)))}, intercept={ridge_ncl.intercept_:.4f}")
    print(f"[ncl] TEST metrics: {m_ncl}")

    # --- Original (Check 0.1's own OOF, independent training) Ridge, re-confirmed here ---
    ridge_orig = Ridge(alpha=1.0).fit(orig_oof[BASE_COLS], orig_oof["SOH"])
    orig_test_deep = pd.read_csv(PRED_DIR / "deep_models_test_preds.csv").rename(columns={
        "y_pred_VLSTM": "pred_VLSTM", "y_pred_CNNLSTM": "pred_CNNLSTM", "y_pred_PiFormer": "pred_PiFormer"})
    orig_test_df = pd.merge(xgb_test, orig_test_deep[["dataset", "battery_id", "cycle_idx",
                             "pred_VLSTM", "pred_CNNLSTM", "pred_PiFormer"]],
                             on=["dataset", "battery_id", "cycle_idx"], how="inner")
    pred_orig_test = ridge_orig.predict(orig_test_df[BASE_COLS])
    m_orig = metrics(orig_test_df["SOH"], pred_orig_test)
    print(f"\n[ncl] === ORIGINAL (Check 0.1, independent training) Ridge meta-learner, re-confirmed ===")
    print(f"[ncl] TEST metrics: {m_orig}")

    # --- Drop-branch ablation: does any deep learner now show genuine marginal value? ---
    print(f"\n[ncl] === DROP-BRANCH ABLATION: NCL-trained ensemble ===")
    ablation_rows = []
    full_pred = ridge_ncl.predict(test_df[BASE_COLS])
    m_full = metrics(y_test, full_pred)
    ablation_rows.append({"stacking": "NCL (this item)", "variant": "full_4_branch", "dropped": "none", **m_full})
    for dropped in BASE_COLS:
        cols = [c for c in BASE_COLS if c != dropped]
        r = Ridge(alpha=1.0).fit(oof_df[cols], oof_df["SOH"])
        p = r.predict(test_df[cols])
        mm = metrics(y_test, p)
        delta = mm["r2"] - m_full["r2"]
        ablation_rows.append({"stacking": "NCL (this item)", "variant": f"drop_{dropped}",
                               "dropped": dropped, "delta_r2_vs_full": delta, **mm})
        print(f"[ncl] drop {dropped}: R2={mm['r2']:.4f} (delta={delta:+.4f} vs full)")

    ablation_df = pd.DataFrame(ablation_rows)

    any_genuine_value = any(
        row["dropped"] in ("pred_VLSTM", "pred_CNNLSTM", "pred_PiFormer") and row.get("delta_r2_vs_full", 0) < -0.01
        for row in ablation_rows
    )

    print(f"\n[ncl] === FINAL SUMMARY ===")
    print(f"[ncl] mean deep-learner pairwise correlation: original={mean_orig:.4f} -> NCL={mean_ncl:.4f} "
          f"({'DROPPED' if correlation_dropped else 'DID NOT MEANINGFULLY DROP'})")
    print(f"[ncl] any deep learner shows genuine ablation value (delta R2 < -0.01 when dropped)? "
          f"{'YES' if any_genuine_value else 'NO'}")
    if correlation_dropped and any_genuine_value:
        print("[ncl] OUTCOME: diversity achieved AND translates to ablation value - a genuine win.")
    elif correlation_dropped and not any_genuine_value:
        print("[ncl] OUTCOME: (per instruction) correlation dropped but no ablation value - "
              "diversity is achievable but doesn't translate to value.")
    elif not correlation_dropped:
        print("[ncl] OUTCOME: (per instruction) correlation did NOT meaningfully drop even with an "
              "explicit NCL penalty - stronger evidence of a feature/architecture ceiling.")
    print(f"[ncl] overall ensemble TEST R2: original={m_orig['r2']:.4f} -> NCL={m_ncl['r2']:.4f}")

    oof_df.to_csv(PRED_DIR / "ncl_oof_meta_features.csv", index=False)
    ablation_df.to_csv(OUT_DIR / "stage3_3_ncl_ablation.csv", index=False)
    pd.DataFrame(corr_deltas).to_csv(OUT_DIR / "stage3_3_ncl_correlation_deltas.csv", index=False)
    pd.DataFrame([
        {"stacking": "original_independent_training", **{f"coef_{k}": v for k, v in zip(BASE_COLS, ridge_orig.coef_)},
         "intercept": ridge_orig.intercept_, **m_orig},
        {"stacking": "NCL_joint_training", **{f"coef_{k}": v for k, v in zip(BASE_COLS, ridge_ncl.coef_)},
         "intercept": ridge_ncl.intercept_, **m_ncl},
    ]).to_csv(OUT_DIR / "stage3_3_ncl_coefficients.csv", index=False)

    print(f"\n[ncl] TOTAL WALL TIME: {(time.time()-t_start)/60:.1f} min")
    print("[ncl] DONE")


if __name__ == "__main__":
    main()
