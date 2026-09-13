"""
Stage 1 final closeout, Part C: isolates the confound in session 41
Part B's RUL result (R2 0.432->0.666). That result mixed TWO changes:
(1) 1.1's specific reformulated duration features, and (2) giving
JointSOHRULModelFusion access to ANY version of the 8 BFA HI features
at all, which it structurally never had before that session.

This is the CONTROL run: identical architecture (JointSOHRULModelFusion,
reused unchanged from run_stage1_followup_partB_joint_rul.py), identical
25-epoch budget/32-battery pool/adaptive-clamped loss weighting, but
with the ORIGINAL, UNREFORMULATED duration features (raw ICHV/TEVD/TEVI)
instead of 1.1's reformulated (_rel) set - everything else held
identical to session 41 Part B's run.
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
from train_deep_models import load_all_battery_tensors, make_xy
from sequence_features import apply_channel_norm
from models.joint_model import AdaptiveLossWeighting
from stage1_common import CANONICAL_FEATURES
from run_stage1_followup_partB_joint_rul import JointSOHRULModelFusion, build_hi_features, standardize

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"

EPOCHS = 25
BATCH_SIZE = 64
_CKPT_PATH = PROC_DIR / "_joint_fusion_control_raw_epoch_checkpoint.pt"


def main():
    t0 = time.time()
    battery_data = load_all_battery_tensors()
    split = json.loads((PROC_DIR / "battery_split.json").read_text())
    train_ids = [b for b in split["train_ids"] if b in battery_data]
    test_ids = [b for b in split["test_ids"] if b in battery_data]
    n_val = max(1, len(train_ids) // 5)
    val_ids = sorted(train_ids)[-n_val:]
    fit_ids = [b for b in train_ids if b not in val_ids]
    print(f"[joint-fusion-control] fit={len(fit_ids)} val={len(val_ids)} test={len(test_ids)} batteries")

    X_fit, soh_fit, rul_fit, _, bid_fit, cyc_fit = make_xy(battery_data, fit_ids)
    X_val, soh_val, rul_val, _, bid_val, cyc_val = make_xy(battery_data, val_ids)
    X_test, soh_test, rul_test, _, bid_test, cyc_test = make_xy(battery_data, test_ids)
    print(f"[joint-fusion-control] fit={len(X_fit)} val={len(X_val)} test={len(X_test)} cycles "
          f"(loaded in {time.time()-t0:.1f}s)")

    norm_stats = json.loads((PROC_DIR / "channel_norm_stats.json").read_text())
    X_fit = apply_channel_norm(X_fit, norm_stats)
    X_val = apply_channel_norm(X_val, norm_stats)
    X_test = apply_channel_norm(X_test, norm_stats)

    # --- RAW, UNREFORMULATED canonical HI features (the control - no 1.1 reformulation applied) ---
    hi_full = pd.read_parquet(PROC_DIR / "hi_table.parquet")
    feature_cols = list(CANONICAL_FEATURES)  # raw ICHV/SCV/VDEDT/VIECT/MATD/MET/TEVD/TEVI
    print(f"[joint-fusion-control] RAW (unreformulated) canonical feature set: {feature_cols}")

    fit_key_df = pd.DataFrame({"battery_id": bid_fit, "cycle_idx": cyc_fit}).merge(
        hi_full[["battery_id", "cycle_idx"] + feature_cols], on=["battery_id", "cycle_idx"], how="left")
    train_medians = fit_key_df[feature_cols].median(numeric_only=True).to_numpy()

    HI_fit = build_hi_features(bid_fit, cyc_fit, hi_full, feature_cols, train_medians)
    HI_val = build_hi_features(bid_val, cyc_val, hi_full, feature_cols, train_medians)
    HI_test = build_hi_features(bid_test, cyc_test, hi_full, feature_cols, train_medians)
    hi_mean, hi_std = HI_fit.mean(axis=0), HI_fit.std(axis=0) + 1e-8
    HI_fit = (HI_fit - hi_mean) / hi_std
    HI_val = (HI_val - hi_mean) / hi_std
    HI_test = (HI_test - hi_mean) / hi_std

    model = JointSOHRULModelFusion(n_hi_features=len(feature_cols))
    soh_fit_z, soh_mean, soh_std = standardize(soh_fit)
    rul_fit_z, rul_mean, rul_std = standardize(rul_fit)
    soh_val_z = (soh_val - soh_mean) / soh_std
    rul_val_z = (rul_val - rul_mean) / rul_std

    adaptive = AdaptiveLossWeighting()
    params = list(model.parameters()) + list(adaptive.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    mse = nn.MSELoss()

    Xt, HIt = torch.tensor(X_fit), torch.tensor(HI_fit)
    soh_t = torch.tensor(soh_fit_z).unsqueeze(-1)
    rul_t = torch.tensor(rul_fit_z).unsqueeze(-1)
    Xv, HIv = torch.tensor(X_val), torch.tensor(HI_val)
    soh_v = torch.tensor(soh_val_z).unsqueeze(-1)
    rul_v = torch.tensor(rul_val_z).unsqueeze(-1)

    start_epoch, history = 0, []
    if _CKPT_PATH.exists():
        ckpt = torch.load(_CKPT_PATH, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        adaptive.load_state_dict(ckpt["adaptive_state"])
        opt.load_state_dict(ckpt["opt_state"])
        torch.set_rng_state(ckpt["torch_rng_state"])
        start_epoch = ckpt["epoch"] + 1
        history = ckpt["history"]
        print(f"[joint-fusion-control] RESUMING from epoch {start_epoch} (checkpoint found)")

    n = len(Xt)
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        perm = torch.randperm(n)
        ep_loss = 0.0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            pred_soh, pred_rul = model(Xt[idx], HIt[idx])
            l_soh = mse(pred_soh, soh_t[idx])
            l_rul = mse(pred_rul, rul_t[idx])
            total, a, b = adaptive(l_soh, l_rul)
            total.backward()
            opt.step()
            with torch.no_grad():
                adaptive.log_sigma_soh.clamp_(-adaptive.LOG_SIGMA_CLAMP, adaptive.LOG_SIGMA_CLAMP)
                adaptive.log_sigma_rul.clamp_(-adaptive.LOG_SIGMA_CLAMP, adaptive.LOG_SIGMA_CLAMP)
            ep_loss += total.item() * len(idx)
        ep_loss /= n

        model.eval()
        with torch.no_grad():
            vp_soh, vp_rul = model(Xv, HIv)
            v_l_soh = mse(vp_soh, soh_v).item()
            v_l_rul = mse(vp_rul, rul_v).item()
        history.append({"epoch": epoch, "train_loss": ep_loss,
                         "val_loss_soh": v_l_soh, "val_loss_rul": v_l_rul, "alpha": a, "beta": b})
        print(f"[joint-fusion-control] epoch {epoch:2d} train={ep_loss:.4f} val_soh={v_l_soh:.4f} "
              f"val_rul={v_l_rul:.4f} alpha={a:.3f} beta={b:.3f}")

        torch.save({"model_state": model.state_dict(), "adaptive_state": adaptive.state_dict(),
                    "opt_state": opt.state_dict(), "torch_rng_state": torch.get_rng_state(),
                    "epoch": epoch, "history": history}, _CKPT_PATH)

    _CKPT_PATH.unlink(missing_ok=True)

    model.eval()
    with torch.no_grad():
        pred_soh_z, pred_rul_z = model(torch.tensor(X_test), torch.tensor(HI_test))
    pred_soh = pred_soh_z.squeeze(-1).numpy() * soh_std + soh_mean
    pred_rul = pred_rul_z.squeeze(-1).numpy() * rul_std + rul_mean

    def m(y_true, y_pred):
        return {"rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "mae": float(mean_absolute_error(y_true, y_pred)), "r2": float(r2_score(y_true, y_pred))}
    soh_metrics = m(soh_test, pred_soh)
    rul_metrics = m(rul_test, pred_rul)
    print(f"\n[joint-fusion-control] TEST SOH: {soh_metrics}")
    print(f"[joint-fusion-control] TEST RUL: {rul_metrics}")

    print(f"\n[joint-fusion-control] === COMPARISON ===")
    print(f"[joint-fusion-control] fixed_balanced (session 4):  SOH R2=0.416, RUL R2=0.428")
    print(f"[joint-fusion-control] adaptive-clamped (session 4): SOH R2=0.344, RUL R2=0.432")
    print(f"[joint-fusion-control] adaptive-clamped + RAW HI features (this CONTROL run): "
          f"SOH R2={soh_metrics['r2']:.4f}, RUL R2={rul_metrics['r2']:.4f}")
    print(f"[joint-fusion-control] adaptive-clamped + 1.1's REFORMULATED HI features (session 41 Part B): "
          f"SOH R2=0.9241, RUL R2=0.6657")

    torch.save(model.state_dict(), ROOT / "models" / "joint_adaptive_fusion_control_raw.pt")
    pd.DataFrame([{"variant": "adaptive_fusion_control_raw", "target": "SOH", **soh_metrics},
                  {"variant": "adaptive_fusion_control_raw", "target": "RUL", **rul_metrics}]).to_csv(
        OUT_DIR / "stage1_final_partC_control_results.csv", index=False)
    pd.DataFrame(history).to_csv(PROC_DIR / "predictions" / "joint_fusion_control_raw_history.csv", index=False)
    print(f"\n[joint-fusion-control] saved outputs/stage1_final_partC_control_results.csv")
    print(f"[joint-fusion-control] ALL DONE in {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
