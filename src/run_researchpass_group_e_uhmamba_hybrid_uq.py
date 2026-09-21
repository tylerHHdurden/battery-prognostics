"""
Research pass Group E, item 16: U-H-Mamba's hybrid-UQ pattern (MC
Dropout combined with conformal prediction) applied to this project's
existing conformal machinery. Reports whether combining beats either
alone.

DISCLOSED SCOPE: MC Dropout needs a neural net with Dropout layers
sampled stochastically at inference time. None of this project's
existing deep models (VLSTM/CNN-LSTM/PiFormer, models/*.py) have any
Dropout layers, and the DEPLOYED SOH point model is XGBoost (no
dropout concept at all) - so this item trains ONE NEW small dropout-
enabled MLP (Dropout(p=0.2) x2, tabular HI+fusion features - the SAME
feature set as the deployed XGBoost model, not the raw cycle-tensor
sequence models, which would need the ~30-minute CNN-LSTM training
pipeline for a UQ-method comparison that doesn't need that scale) as
the base regressor this item's three UQ configurations are all built
on top of, so the comparison is apples-to-apples ACROSS the three UQ
methods even though it isn't literally the deployed XGBoost model
itself.

THREE CONFIGURATIONS compared on CALCE (same zero-retrain convention,
same coverage+width reporting as items 8-9, same PRIOR_ATTEMPTS table
this pass has used throughout - NOTE the new MLP's OWN plain-conformal
row is a new baseline specific to this base model, not literally
re-measuring the prior "Plain split-conformal (do-nothing floor)" row,
which used the XGBoost model - disclosed, not conflated):
  (a) MC-Dropout ALONE: 50 stochastic forward passes -> predictive
      mean + std -> naive Gaussian interval (mean +/- 1.645*std for a
      90% two-sided interval), NO conformal calibration at all.
  (b) Split-conformal ALONE on this MLP's (dropout-off, deterministic)
      point prediction - the standard fixed-width split-conformal this
      project has used throughout, no local-uncertainty scaling.
  (c) HYBRID: split-conformal calibration where the local interval
      SCALE is the model's own MC-Dropout predictive std (same
      "locally-adaptive normalized conformal" family as items 8-9,
      but using the model's native epistemic-uncertainty signal
      instead of a separately-fit isotonic/GPR scale function) - this
      is the actual U-H-Mamba-style combination being tested.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_common import (
    load_nasa_mit_pool, battery_split_masks, fusion_cols, canonical_feature_cols,
    build_calce_merged, OUT_DIR,
)

SEED = 42
ALPHA = 0.1
N_MC_SAMPLES = 50
EPOCHS = 60
BATCH_SIZE = 64
LR = 1e-3
DROPOUT_P = 0.2
Z_90 = 1.645

PRIOR_ATTEMPTS = [
    {"method": "Domain-alignment (MMD)", "coverage": 0.044, "width": 4.43},
    {"method": "Reweighted calibration (domain classifier)", "coverage": 0.044, "width": 4.38},
    {"method": "More training data alone (32->204 batteries)", "coverage": 0.074, "width": 2.309},
    {"method": "Normalized CP (GBR local-difficulty)", "coverage": 0.193, "width": 6.545},
    {"method": "CQR", "coverage": 0.213, "width": 18.267},
    {"method": "Jackknife+/CV+ resampling", "coverage": 0.371, "width": 11.89},
    {"method": "KMM-CP", "coverage": 0.340, "width": 9.924},
    {"method": "Domain-alignment (CORAL)", "coverage": 0.032, "width": 2.347},
    {"method": "Rescaled Jackknife+ (difficulty-aware)", "coverage": 0.820, "width": 58.905},
    {"method": "Plain split-conformal (XGBoost, do-nothing floor)", "coverage": 0.067, "width": 2.332},
]


class DropoutMLP(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, p: float = DROPOUT_P):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def mc_dropout_predict(model, X_t, n_samples=N_MC_SAMPLES):
    model.train()  # keep dropout ACTIVE
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds.append(model(X_t).numpy())
    preds = np.stack(preds, axis=0)  # (n_samples, n)
    return preds.mean(axis=0), preds.std(axis=0)


def main():
    t0 = time.time()
    print("=== Research pass Group E, item 16: U-H-Mamba hybrid UQ (MC Dropout + conformal) ===")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    merged, hi_full = load_nasa_mit_pool(reformulated=True)
    train_mask, test_mask, split = battery_split_masks(merged)
    feature_cols = canonical_feature_cols(reformulated=True) + ["cycle_idx"]
    fcols = fusion_cols()
    cols = feature_cols + fcols

    X_all = merged[cols].to_numpy(dtype=float, copy=True)
    X_all = np.where(np.isinf(X_all), np.nan, X_all)
    train_medians = np.nanmedian(X_all[train_mask], axis=0)
    inds = np.where(np.isnan(X_all))
    X_all[inds] = np.take(train_medians, inds[1])
    y_all = merged["SOH"].to_numpy(dtype=float)

    scaler = StandardScaler().fit(X_all[train_mask])
    X_all_scaled = scaler.transform(X_all)

    rng = np.random.default_rng(SEED)
    train_idx = np.where(train_mask)[0]
    perm = rng.permutation(len(train_idx))
    n_val = max(1, int(0.15 * len(train_idx)))
    val_pos, fit_pos = perm[:n_val], perm[n_val:]
    fit_idx, val_idx = train_idx[fit_pos], train_idx[val_pos]

    X_fit_t = torch.tensor(X_all_scaled[fit_idx], dtype=torch.float32)
    y_fit_t = torch.tensor(y_all[fit_idx], dtype=torch.float32)
    X_val_t = torch.tensor(X_all_scaled[val_idx], dtype=torch.float32)
    y_val_t = torch.tensor(y_all[val_idx], dtype=torch.float32)

    model = DropoutMLP(n_features=X_all_scaled.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    best_val, best_state, patience, bad = float("inf"), None, 8, 0
    n_fit = len(fit_idx)
    print(f"[hybrid-uq] training DropoutMLP: fit={n_fit} rows, val={len(val_idx)} rows, {X_all_scaled.shape[1]} features")
    for epoch in range(EPOCHS):
        model.train()
        idx = rng.permutation(n_fit)
        ep_losses = []
        for s in range(0, n_fit, BATCH_SIZE):
            b = idx[s:s + BATCH_SIZE]
            opt.zero_grad()
            pred = model(X_fit_t[b])
            loss = loss_fn(pred, y_fit_t[b])
            loss.backward(); opt.step()
            ep_losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = loss_fn(val_pred, y_val_t).item()
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(f"[hybrid-uq] epoch {epoch+1}/{EPOCHS}: train_mse={np.mean(ep_losses):.4f} val_mse={val_loss:.4f}")
        if val_loss < best_val:
            best_val, best_state, bad = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[hybrid-uq] early stop at epoch {epoch+1}")
                break
    model.load_state_dict(best_state)

    from run_conformal import calib_eval_battery_split
    test_ids = sorted(merged.loc[test_mask, "battery_id"].unique())
    calib_ids, eval_ids = calib_eval_battery_split(test_ids)
    calib_mask = merged["battery_id"].isin(calib_ids).to_numpy() & test_mask
    print(f"[hybrid-uq] calibration batteries: {calib_ids}, in-domain eval batteries: {eval_ids}")

    X_calib_t = torch.tensor(X_all_scaled[calib_mask], dtype=torch.float32)
    y_calib = y_all[calib_mask]
    model.eval()
    with torch.no_grad():
        calib_point_pred = model(X_calib_t).numpy()
    calib_mc_mean, calib_mc_std = mc_dropout_predict(model, X_calib_t)
    calib_mc_std = np.clip(calib_mc_std, 1e-3, None)

    calce_merged = build_calce_merged(hi_full)
    X_calce = calce_merged[cols].to_numpy(dtype=float, copy=True)
    X_calce = np.where(np.isinf(X_calce), np.nan, X_calce)
    inds_c = np.where(np.isnan(X_calce))
    X_calce[inds_c] = np.take(train_medians, inds_c[1])
    X_calce_scaled = scaler.transform(X_calce)
    X_calce_t = torch.tensor(X_calce_scaled, dtype=torch.float32)
    y_calce = calce_merged["SOH"].to_numpy(dtype=float)

    with torch.no_grad():
        calce_point_pred = model(X_calce_t).numpy()
    calce_mc_mean, calce_mc_std = mc_dropout_predict(model, X_calce_t)
    calce_mc_std = np.clip(calce_mc_std, 1e-3, None)

    def coverage_width(lo, hi, y):
        covered = (y >= lo) & (y <= hi)
        return float(covered.mean()), float(np.mean(hi - lo))

    # (a) MC-Dropout ALONE: naive Gaussian interval from dropout std, no conformal calibration
    lo_a, hi_a = calce_mc_mean - Z_90 * calce_mc_std, calce_mc_mean + Z_90 * calce_mc_std
    cov_a, width_a = coverage_width(lo_a, hi_a, y_calce)
    print(f"[hybrid-uq] (a) MC-Dropout ALONE: coverage={cov_a:.4f} ({cov_a:.1%}) width={width_a:.3f}")

    # (b) Split-conformal ALONE on this MLP's deterministic point prediction
    calib_resid_b = np.abs(y_calib - calib_point_pred)
    n_b = len(calib_resid_b)
    q_level_b = min(np.ceil((n_b + 1) * (1 - ALPHA)) / n_b, 1.0)
    q_b = np.quantile(calib_resid_b, q_level_b, method="higher")
    lo_b, hi_b = calce_point_pred - q_b, calce_point_pred + q_b
    cov_b, width_b = coverage_width(lo_b, hi_b, y_calce)
    print(f"[hybrid-uq] (b) Split-conformal ALONE (this MLP): coverage={cov_b:.4f} ({cov_b:.1%}) width={width_b:.3f}")

    # (c) HYBRID: conformal calibration, local scale = MC-Dropout predictive std
    calib_resid_c = np.abs(y_calib - calib_mc_mean)
    normalized_resid_c = calib_resid_c / calib_mc_std
    n_c = len(normalized_resid_c)
    q_level_c = min(np.ceil((n_c + 1) * (1 - ALPHA)) / n_c, 1.0)
    q_c = np.quantile(normalized_resid_c, q_level_c, method="higher")
    half_width_c = q_c * calce_mc_std
    lo_c, hi_c = calce_mc_mean - half_width_c, calce_mc_mean + half_width_c
    cov_c, width_c = coverage_width(lo_c, hi_c, y_calce)
    print(f"[hybrid-uq] (c) HYBRID (MC-Dropout scale + conformal calibration): "
          f"coverage={cov_c:.4f} ({cov_c:.1%}) width={width_c:.3f}")

    results = PRIOR_ATTEMPTS + [
        {"method": "MC-Dropout ALONE (naive Gaussian, this item's MLP)", "coverage": cov_a, "width": width_a},
        {"method": "Split-conformal ALONE (this item's MLP)", "coverage": cov_b, "width": width_b},
        {"method": "HYBRID: MC-Dropout scale + conformal (THIS ITEM, U-H-Mamba-style)", "coverage": cov_c, "width": width_c},
    ]
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "researchpass_groupE16_uhmamba_hybrid_uq.csv", index=False)
    print("\n=== CONSOLIDATED COMPARISON ===")
    print(results_df.to_string(index=False))

    better_than_both = (cov_c >= max(cov_a, cov_b)) and (width_c <= min(width_a, width_b))
    if better_than_both and not (cov_a == cov_c and width_a == width_c):
        print(f"\n[hybrid-uq] VERDICT: WIN - hybrid combination beats BOTH MC-Dropout-alone and "
              f"conformal-alone on this base model (coverage>=both, width<=both)")
    elif cov_c > max(cov_a, cov_b):
        print(f"\n[hybrid-uq] VERDICT: PARTIAL - hybrid reaches higher coverage than either alone, "
              f"but not strictly narrower too")
    else:
        print(f"\n[hybrid-uq] VERDICT: LOSS/INCONCLUSIVE - combining did not clearly beat both components alone")
    print(f"\n[hybrid-uq] TOTAL TIME: {(time.time()-t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
