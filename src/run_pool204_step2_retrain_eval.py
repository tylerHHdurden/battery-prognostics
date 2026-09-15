"""
Data-expansion pass, Part A, Step 2+3: retrain the deployable pipeline
(channel_norm_stats, ICA fusion encoder, XGBoost-fusion with the
CURRENT canonical Stage 1.1 reformulated-feature + Stage 1.5 monotone-
constraint config) on the full 204-battery pool, then evaluate
in-domain (fixed split + GroupKFold(5)) and zero-retrain on all four
held-out sets (CALCE/Oxford/HUST/XJTU - NEVER trained on, same
convention as every other stage). Reports directly against the
currently-deployed 42-battery model's own numbers for an honest,
apples-to-apples before/after comparison.

Writes ONLY to its own, separate files (channel_norm_stats_pool204.
json, ica_encoder_pool204.pt, fusion_embeddings_pool204.csv,
xgb_soh_fusion_pool204.json) - does NOT touch any deployed file
(channel_norm_stats.json, ica_encoder.pt, fusion_embeddings.csv,
xgb_soh_fusion.json, live_inference.py, app.py). VLSTM is NOT retrained
here (SHAP-explainability role only, not needed for this comparison,
scoped out to keep this already-large pass bounded).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
from pool204_tensors import load_battery_tensors_pool204
from train_deep_models import make_xy
from sequence_features import compute_channel_norm_stats, apply_channel_norm
from models.ica_encoder import ICAEncoder
from stage1_common import (
    canonical_feature_cols, add_reformulated_duration_features,
    OUT_DIR, PROC_DIR, ROOT, ALPHA,
)
from run_stage5_1_new_datasets_eval import (
    oxford_cell_ids, iterate_oxford_cycles, hust_cell_ids, iterate_hust_cycles,
    xjtu_cell_ids_soh_valid, iterate_xjtu_cycles,
)
from data_adapters import iterate_calce_cycles
from sequence_features import build_dataset_tensors
from health_indicators import compute_health_indicators
from rul_labels import soh_per_cycle

EMBED_DIM = 16
ICA_CHANNEL_SLICE = slice(3, 6)
CALCE_CELLS = ["CS2_35", "CS2_36", "CS2_37"]


def train_encoder(X_fit, y_fit, X_val, y_val, epochs=25, patience=6, batch_size=64, lr=1e-3):
    encoder = ICAEncoder(in_channels=3, embed_dim=EMBED_DIM)
    opt = torch.optim.Adam(encoder.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    y_mean, y_std = float(y_fit.mean()), float(y_fit.std() + 1e-8)
    Xt = torch.tensor(X_fit); yt = torch.tensor((y_fit - y_mean) / y_std).unsqueeze(-1)
    Xv = torch.tensor(X_val); yv = torch.tensor((y_val - y_mean) / y_std).unsqueeze(-1)
    n = len(Xt)
    best_val, best_state, patience_ctr = np.inf, None, 0
    history = []
    for epoch in range(epochs):
        encoder.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            pred = encoder(Xt[idx])
            loss = loss_fn(pred, yt[idx])
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n
        encoder.eval()
        with torch.no_grad():
            val_loss = loss_fn(encoder(Xv), yv).item()
        history.append({"epoch": epoch, "train_loss": epoch_loss, "val_loss": val_loss})
        print(f"[pool204-2] encoder epoch {epoch:3d} train_mse={epoch_loss:.4f} val_mse={val_loss:.4f}")
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in encoder.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"[pool204-2] encoder early stopping at epoch {epoch}")
                break
    if best_state is not None:
        encoder.load_state_dict(best_state)
    return encoder, history


def fit_xgb_local(merged, train_mask, feature_cols, fcols, xgb_extra_kwargs=None):
    cols = feature_cols + fcols
    X = merged[cols].to_numpy(dtype=float, copy=True)
    # bug found and fixed: the 204-pool has real inf values (VDEDT, the
    # same known issue as elsewhere in this project) - eval_xgb_local
    # already handled this but this training path didn't, crashing
    # XGBoost's QuantileDMatrix on the FIRST retrain attempt ("Input
    # data contains `inf`..."). Fixed here rather than worked around.
    X = np.where(np.isinf(X), np.nan, X)
    col_medians = np.nanmedian(X[train_mask], axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_medians, inds[1])
    y = merged["SOH"].to_numpy(dtype=float)
    kwargs = dict(n_estimators=500, max_depth=6, learning_rate=0.03,
                  subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, reg_lambda=1.0)
    if xgb_extra_kwargs:
        kwargs.update(xgb_extra_kwargs)
    model = XGBRegressor(**kwargs)
    model.fit(X[train_mask], y[train_mask])
    return model, col_medians, cols


def eval_xgb_local(model, medians, feature_cols, fcols, df):
    cols = feature_cols + fcols
    X = df[cols].to_numpy(dtype=float, copy=True)
    X = np.where(np.isinf(X), np.nan, X)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(medians, inds[1])
    y_true = df["SOH"].to_numpy(dtype=float)
    pred = model.predict(X)
    return {"rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
            "mae": float(mean_absolute_error(y_true, pred)),
            "r2": float(r2_score(y_true, pred)), "pred": pred, "y_true": y_true,
            "battery_id": df["battery_id"].to_numpy()}


def build_new_dataset_merged(name, cell_ids, iterate_fn, norm_stats, encoder):
    hi_rows = []
    Xs, bids, cycs = [], [], []
    for cid in cell_ids:
        cycles = list(iterate_fn(cid))
        if len(cycles) < 5:
            continue
        soh_map = soh_per_cycle(cycles)
        for c in cycles:
            his = compute_health_indicators(c)
            row = {"dataset": name, "battery_id": cid, "cycle_idx": c["cycle_idx"],
                   "SOH": soh_map[c["cycle_idx"]]}
            row.update(his)
            hi_rows.append(row)
        X, soh, rul, idxs, censored = build_dataset_tensors(cycles)
        if X is None:
            continue
        Xs.append(X.astype(np.float32)); bids += [cid] * len(idxs); cycs += list(idxs)
    hi_df = pd.DataFrame(hi_rows)
    X_all = np.concatenate(Xs).astype(np.float32)
    X_norm = apply_channel_norm(X_all, norm_stats)
    with torch.no_grad():
        fusion_emb = encoder.encode(torch.tensor(X_norm[:, :, ICA_CHANNEL_SLICE])).numpy()
    hi_reformulated = add_reformulated_duration_features(hi_df)
    seq_df = pd.DataFrame({"battery_id": bids, "cycle_idx": cycs})
    for i in range(16):
        seq_df[f"fusion_{i}"] = fusion_emb[:, i]
    merged = pd.merge(hi_reformulated, seq_df, on=["battery_id", "cycle_idx"], how="inner")
    return merged


def main():
    t_start = time.time()
    print("=== Part A, Step 2: retrain on the 204-battery pool ===")
    battery_data = load_battery_tensors_pool204()

    split = json.loads((PROC_DIR / "battery_split_expanded_b0018pinned.json").read_text())
    train_ids = [b for b in split["train_ids"] if b in battery_data]
    test_ids = [b for b in split["test_ids"] if b in battery_data]
    n_val = max(1, len(train_ids) // 5)
    val_ids = sorted(train_ids)[-n_val:]
    fit_ids = [b for b in train_ids if b not in val_ids]
    print(f"[pool204-2] fit={len(fit_ids)} val={len(val_ids)} test={len(test_ids)} batteries "
          f"({len(battery_data)} total loaded)")

    all_ids = fit_ids + val_ids + test_ids
    X_all, y_all, rul_all, ds_all, bid_all, cyc_all = make_xy(battery_data, all_ids)
    n_fit = sum(len(battery_data[b][1]) for b in fit_ids)
    n_val_cyc = sum(len(battery_data[b][1]) for b in val_ids)
    X_fit_raw = X_all[:n_fit]

    print("\n[pool204-2] === refitting channel_norm_stats (own file, NOT the deployed one) ===")
    norm_stats = compute_channel_norm_stats(X_fit_raw)
    with open(PROC_DIR / "channel_norm_stats_pool204.json", "w") as f:
        json.dump(norm_stats, f, indent=2)

    X_all_norm = apply_channel_norm(X_all, norm_stats)
    X_fit_ica = X_all_norm[:n_fit, :, ICA_CHANNEL_SLICE]
    X_val_ica = X_all_norm[n_fit:n_fit + n_val_cyc, :, ICA_CHANNEL_SLICE]
    y_fit, y_val = y_all[:n_fit], y_all[n_fit:n_fit + n_val_cyc]

    print("\n[pool204-2] === training ICA fusion encoder (own file) ===")
    encoder, hist = train_encoder(X_fit_ica, y_fit, X_val_ica, y_val)
    torch.save(encoder.state_dict(), ROOT / "models" / "ica_encoder_pool204.pt")
    pd.DataFrame(hist).to_csv(OUT_DIR / "pool204_encoder_history.csv", index=False)

    print("\n[pool204-2] === generating fusion embeddings for the full pool ===")
    encoder.eval()
    with torch.no_grad():
        embeddings = encoder.encode(torch.tensor(X_all_norm[:, :, ICA_CHANNEL_SLICE])).numpy()
    fusion_df = pd.DataFrame({"dataset": ds_all, "battery_id": bid_all, "cycle_idx": cyc_all})
    for i in range(EMBED_DIM):
        fusion_df[f"fusion_{i}"] = embeddings[:, i]
    fusion_df.to_csv(PROC_DIR / "fusion_embeddings_pool204.csv", index=False)
    fcols = [f"fusion_{i}" for i in range(EMBED_DIM)]

    print("\n[pool204-2] === building HI-reformulated + fusion merged frame ===")
    hi_full = pd.read_parquet(PROC_DIR / "hi_table_pool204.parquet")
    hi_reformulated = add_reformulated_duration_features(hi_full)
    nasa_mit = hi_reformulated[hi_reformulated["dataset"].isin(["NASA", "MIT"])]
    merged = pd.merge(nasa_mit, fusion_df, on=["dataset", "battery_id", "cycle_idx"], how="inner")
    print(f"[pool204-2] merged: {len(merged)} rows ({len(nasa_mit)} HI rows, "
          f"{len(nasa_mit) - len(merged)} lost to merge)")

    train_mask = merged["battery_id"].isin(train_ids).to_numpy()
    test_mask = merged["battery_id"].isin(test_ids).to_numpy()

    feature_cols = canonical_feature_cols(reformulated=True) + ["cycle_idx"]
    monotone = tuple([0] * (len(feature_cols) - 1) + [-1] + [0] * len(fcols))

    print("\n[pool204-2] === retraining XGBoost-fusion (canonical 1.1+1.5 config) ===")
    model, medians, cols = fit_xgb_local(merged, train_mask, feature_cols, fcols,
                                          xgb_extra_kwargs={"monotone_constraints": monotone})
    model.save_model(str(ROOT / "models" / "xgb_soh_fusion_pool204.json"))

    indomain = eval_xgb_local(model, medians, feature_cols, fcols, merged[test_mask])
    print(f"[pool204-2] IN-DOMAIN (fixed split): R2={indomain['r2']:.4f} RMSE={indomain['rmse']:.4f} "
          f"(Stage 4 42-battery: R2=0.9740 RMSE=0.7805)")

    print("\n[pool204-2] === GroupKFold(5) CV ===")
    gkf = GroupKFold(n_splits=5)
    unique_batteries = merged["battery_id"].unique()
    fold_rows = []
    for fold_i, (train_bidx, test_bidx) in enumerate(gkf.split(unique_batteries, groups=unique_batteries)):
        tb = set(unique_batteries[train_bidx]); eb = set(unique_batteries[test_bidx])
        tm = merged["battery_id"].isin(tb).to_numpy()
        em = merged["battery_id"].isin(eb).to_numpy()
        m2, med2, c2 = fit_xgb_local(merged, tm, feature_cols, fcols,
                                      xgb_extra_kwargs={"monotone_constraints": monotone})
        r = eval_xgb_local(m2, med2, feature_cols, fcols, merged[em])
        print(f"[pool204-2] fold {fold_i}: R2={r['r2']:.4f} RMSE={r['rmse']:.4f} n={em.sum()}")
        fold_rows.append({"fold": fold_i, "r2": r["r2"], "rmse": r["rmse"], "n_test": int(em.sum())})
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(OUT_DIR / "pool204_groupkfold.csv", index=False)
    print(f"[pool204-2] GroupKFold mean R2={fold_df['r2'].mean():.4f} (std {fold_df['r2'].std():.4f}) "
          f"mean RMSE={fold_df['rmse'].mean():.4f} (std {fold_df['rmse'].std():.4f})")
    print(f"[pool204-2] vs. Stage 4 42-battery GroupKFold: R2=0.9658 (std 0.0207) RMSE=1.1388 (std 0.5673)")

    print("\n=== Part A, Step 3: zero-retrain eval on all 4 held-out datasets ===")
    calce_rows = []
    for cid in CALCE_CELLS:
        cycles = list(iterate_calce_cycles(cid))
        soh_map = soh_per_cycle(cycles)
        for c in cycles:
            his = compute_health_indicators(c)
            row = {"dataset": "CALCE", "battery_id": cid, "cycle_idx": c["cycle_idx"], "SOH": soh_map[c["cycle_idx"]]}
            row.update(his)
            calce_rows.append(row)
    calce_hi = pd.DataFrame(calce_rows)
    calce_hi_ref = add_reformulated_duration_features(calce_hi)
    calce_fusion = fusion_df[fusion_df["dataset"] == "CALCE"]
    calce_merged = pd.merge(calce_hi_ref, calce_fusion, on=["battery_id", "cycle_idx"], how="inner")

    results = {"CALCE": eval_xgb_local(model, medians, feature_cols, fcols, calce_merged)}
    summary_rows = [{"dataset": "in-domain (fixed split)", "n_cycles": len(indomain["pred"]),
                      "r2": indomain["r2"], "rmse": indomain["rmse"]},
                     {"dataset": "CALCE", "n_cycles": len(results["CALCE"]["pred"]),
                      "r2": results["CALCE"]["r2"], "rmse": results["CALCE"]["rmse"]}]

    for name, ids, iterate_fn in [
        ("Oxford", oxford_cell_ids(), iterate_oxford_cycles),
        ("HUST", hust_cell_ids(), iterate_hust_cycles),
        ("XJTU", xjtu_cell_ids_soh_valid(), iterate_xjtu_cycles),
    ]:
        m = build_new_dataset_merged(name, ids, iterate_fn, norm_stats, encoder)
        r = eval_xgb_local(model, medians, feature_cols, fcols, m)
        print(f"[pool204-3] {name}: R2={r['r2']:.4f} RMSE={r['rmse']:.4f} n={len(r['pred'])}")
        results[name] = r
        summary_rows.append({"dataset": name, "n_cycles": len(r["pred"]), "r2": r["r2"], "rmse": r["rmse"]})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "pool204_zero_retrain_eval.csv", index=False)
    print("\n=== SUMMARY (204-battery pool, vs. Stage 4's 42-battery deployed model) ===")
    print(summary.to_string(index=False))
    print("\nStage 4 (42-battery, deployed) reference: in-domain R2=0.9740/0.9658(CV), "
          "CALCE=0.568, Oxford=-2.694, HUST=-0.152, XJTU=-1.059")
    print(f"\n[pool204-2/3] TOTAL TIME: {(time.time()-t_start)/60:.1f} minutes")


if __name__ == "__main__":
    main()
