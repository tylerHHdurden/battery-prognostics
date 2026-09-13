"""
Follow-up Part B: does Stage 1's canonical (1.1-reformulated) feature
set help the joint SOH+RUL model, the way it helped XGBoost-fusion?

ARCHITECTURAL MISMATCH - ROOT-CAUSED AND STATED PLAINLY BEFORE WRITING
ANY TRAINING CODE, per this project's own standard: the task's B.1
instruction asks to retrain the joint-adaptive model "on Stage 1's
canonical feature set... same architecture... as its original
training." Checked `models/joint_model.py` and `train_joint_adaptive.py`
directly first: JointSOHRULModel is a CNN+LSTM operating on 6-channel
RAW SEQUENCE TENSORS (200 timesteps of V_t/I_t/T_t/dQdV/dVdQ/dIdV) -
it has NEVER consumed the 8 BFA HI features (ICHV/SCV/VDEDT/etc.) at
all, unlike XGBoost-fusion. "Same architecture" AND "Stage 1's
canonical feature set" are mutually exclusive as literally written -
the joint model has no input slot for 8 static per-cycle features to
begin with.

RESOLUTION, stated explicitly rather than silently picking one: rather
than either (a) silently do a no-op "retrain" that changes nothing
(the sequence tensors are identical either way, so the model would be
byte-for-byte the same modulo RNG), or (b) refuse the item entirely,
this implements the SMALLEST faithful extension that lets the actual
question ("does 1.1's reformulated features help RUL") be tested at
all: a new, clearly-flagged, ADDITIVE `JointSOHRULModelFusion` variant
that concatenates the 8 canonical (1.1-reformulated) HI features onto
the LSTM's final hidden state before the two regression heads -
directly analogous to how "XGBoost-fusion" itself is named/built (HI
features + a learned embedding, concatenated, then a final regressor).
This is NOT "the same architecture" (a real, disclosed deviation) - it
is the closest architecture that can actually receive the requested
features. Everything else (backbone, training budget, split, epochs,
adaptive-clamped loss weighting - the SAME variant that produced the
best recorded RUL result, RUL R2=0.432) matches the original exactly.

Scope, per B.3: only the "adaptive" (clamped) loss-weighting variant is
retrained here - the one directly comparable to the best recorded RUL
baseline - not the full 4-variant ablation, and no monotone_constraints
or sample weighting are attempted (out of this item's stated scope).

Pool: the ORIGINAL 32-battery pool (battery_split.json), matching
where Stage 1's canonical feature set was actually validated - NOT the
204-battery expanded pool, per this item's own instruction not to
conflate the two.
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
from models.joint_model import JointSOHRULModel, AdaptiveLossWeighting
from stage1_common import canonical_feature_cols, add_reformulated_duration_features

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"

EPOCHS = 25  # identical to train_joint_adaptive.py's original budget
BATCH_SIZE = 64
_CKPT_PATH = PROC_DIR / "_joint_fusion_epoch_checkpoint.pt"


class JointSOHRULModelFusion(JointSOHRULModel):
    """Additive subclass - backbone/branches/lstm all inherited unchanged.
    Only the heads differ: they take [lstm_hidden ++ n_hi_features]
    instead of [lstm_hidden] alone."""

    def __init__(self, n_hi_features: int, **kwargs):
        super().__init__(**kwargs)
        lstm_hidden = self.lstm.hidden_size
        self.soh_head = nn.Sequential(
            nn.Linear(lstm_hidden + n_hi_features, 16), nn.ReLU(), nn.Linear(16, 1)
        )
        self.rul_head = nn.Sequential(
            nn.Linear(lstm_hidden + n_hi_features, 16), nn.ReLU(), nn.Linear(16, 1)
        )

    def forward(self, x, hi_feats):
        h = self.backbone(x)
        h_fused = torch.cat([h, hi_feats], dim=1)
        return self.soh_head(h_fused), self.rul_head(h_fused)


def standardize(arr):
    mean, std = float(arr.mean()), float(arr.std() + 1e-8)
    return (arr - mean) / std, mean, std


def build_hi_features(bid_list, cyc_list, hi_reformulated, feature_cols, train_medians):
    """Aligns the canonical (reformulated) HI features to each sequence-
    tensor row by (battery_id, cycle_idx) - same median-imputation
    convention (train-only medians) as every XGBoost-fusion script in
    this project."""
    key_df = pd.DataFrame({"battery_id": bid_list, "cycle_idx": cyc_list, "_row_order": range(len(bid_list))})
    merged = key_df.merge(hi_reformulated[["battery_id", "cycle_idx"] + feature_cols],
                           on=["battery_id", "cycle_idx"], how="left")
    merged = merged.sort_values("_row_order")
    X = merged[feature_cols].to_numpy(dtype=np.float32, copy=True)
    for j, col in enumerate(feature_cols):
        nan_mask = np.isnan(X[:, j])
        if nan_mask.any():
            X[nan_mask, j] = train_medians[j]
            print(f"[joint-fusion] {int(nan_mask.sum())} rows missing {col} - imputed with train median {train_medians[j]:.4f}")
    return X


def main():
    t0 = time.time()
    battery_data = load_all_battery_tensors()
    split = json.loads((PROC_DIR / "battery_split.json").read_text())
    train_ids = [b for b in split["train_ids"] if b in battery_data]
    test_ids = [b for b in split["test_ids"] if b in battery_data]
    n_val = max(1, len(train_ids) // 5)
    val_ids = sorted(train_ids)[-n_val:]
    fit_ids = [b for b in train_ids if b not in val_ids]
    print(f"[joint-fusion] fit={len(fit_ids)} val={len(val_ids)} test={len(test_ids)} batteries")

    X_fit, soh_fit, rul_fit, _, bid_fit, cyc_fit = make_xy(battery_data, fit_ids)
    X_val, soh_val, rul_val, _, bid_val, cyc_val = make_xy(battery_data, val_ids)
    X_test, soh_test, rul_test, _, bid_test, cyc_test = make_xy(battery_data, test_ids)
    print(f"[joint-fusion] fit={len(X_fit)} val={len(X_val)} test={len(X_test)} cycles "
          f"(loaded in {time.time()-t0:.1f}s)")

    norm_stats = json.loads((PROC_DIR / "channel_norm_stats.json").read_text())
    X_fit = apply_channel_norm(X_fit, norm_stats)
    X_val = apply_channel_norm(X_val, norm_stats)
    X_test = apply_channel_norm(X_test, norm_stats)

    # --- Stage 1's canonical (1.1-reformulated) HI features, aligned per-cycle ---
    hi_full = pd.read_parquet(PROC_DIR / "hi_table.parquet")
    hi_reformulated = add_reformulated_duration_features(hi_full)
    feature_cols = canonical_feature_cols(reformulated=True)
    print(f"[joint-fusion] canonical reformulated feature set: {feature_cols}")

    fit_key_df = pd.DataFrame({"battery_id": bid_fit, "cycle_idx": cyc_fit}).merge(
        hi_reformulated[["battery_id", "cycle_idx"] + feature_cols], on=["battery_id", "cycle_idx"], how="left")
    train_medians = fit_key_df[feature_cols].median(numeric_only=True).to_numpy()

    HI_fit = build_hi_features(bid_fit, cyc_fit, hi_reformulated, feature_cols, train_medians)
    HI_val = build_hi_features(bid_val, cyc_val, hi_reformulated, feature_cols, train_medians)
    HI_test = build_hi_features(bid_test, cyc_test, hi_reformulated, feature_cols, train_medians)
    # standardize HI features (z-score, fit-only stats) - same convention as
    # the sequence channels' own normalization, needed since raw HI scales
    # vary wildly (durations vs. voltages vs. energies) and would otherwise
    # dominate/vanish relative to the LSTM's own hidden-state scale.
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
        print(f"[joint-fusion] RESUMING from epoch {start_epoch} (checkpoint found)")

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
        print(f"[joint-fusion] epoch {epoch:2d} train={ep_loss:.4f} val_soh={v_l_soh:.4f} "
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
    print(f"\n[joint-fusion] TEST SOH: {soh_metrics}")
    print(f"[joint-fusion] TEST RUL: {rul_metrics}")

    print(f"\n[joint-fusion] === COMPARISON vs. recorded baselines ===")
    print(f"[joint-fusion] fixed_balanced (session 4):  SOH R2=0.416, RUL R2=0.428")
    print(f"[joint-fusion] adaptive-clamped (session 4): SOH R2=0.344, RUL R2=0.432 (best recorded RUL)")
    print(f"[joint-fusion] adaptive-clamped + 1.1's canonical reformulated features (this run): "
          f"SOH R2={soh_metrics['r2']:.4f}, RUL R2={rul_metrics['r2']:.4f}")

    torch.save(model.state_dict(), ROOT / "models" / "joint_adaptive_fusion_canonical.pt")
    pd.DataFrame([{"variant": "adaptive_fusion_canonical", "target": "SOH", **soh_metrics},
                  {"variant": "adaptive_fusion_canonical", "target": "RUL", **rul_metrics}]).to_csv(
        OUT_DIR / "stage1_followup_partB_joint_rul_results.csv", index=False)
    pd.DataFrame(history).to_csv(PROC_DIR / "predictions" / "joint_fusion_canonical_history.csv", index=False)
    print(f"\n[joint-fusion] saved outputs/stage1_followup_partB_joint_rul_results.csv")
    print(f"[joint-fusion] ALL DONE in {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
