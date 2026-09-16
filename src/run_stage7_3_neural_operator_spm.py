"""
Stage 7.3: DeepONet (src/models/deeponet.py - see its docstring for the
DeepONet-vs-FNO choice) as a real SPM/SPMe-style electrochemical-
dynamics surrogate, trained on this project's own raw discharge curves:
learns the OPERATOR mapping discharge current I(t) -> discharge voltage
V(t), a genuinely different, physically-grounded task from point SOH
regression.

Two-part evaluation, per the task's own instruction:
1. Does the neural operator ITSELF work? Reconstruction accuracy of
   V(t) from I(t) alone, on in-pool held-out cycles, the TEST battery
   split, and zero-retrain on CALCE/Oxford/HUST/XJTU.
2. Integration: does the trained branch net's own embedding, added as
   a NEW feature alongside the existing fusion embeddings, help the
   canonical Stage-6 XGBoost-fusion pipeline (in-domain + zero-retrain),
   directly compared against (a) the pipeline WITHOUT this embedding
   (an honest ablation) and (b) session 4's own monotonicity-penalty
   result (a "shallow physics constraint" vs. "genuine physics
   surrogate" comparison, as the task explicitly asks for).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
from stage7_common import (
    load_pool_train_test_keyed, load_heldout_keyed, OUT_DIR, MODEL_DIR, PROC_DIR,
)
from stage1_common import (
    load_nasa_mit_pool, add_reformulated_duration_features, canonical_feature_cols,
    fusion_cols, battery_split_masks, build_calce_merged, ROOT,
)
from stage5_extended_reformulation import add_scv_matd_viect_reformulated, extended_canonical_feature_cols
from models.deeponet import DeepONet

SEED = 42
N_SENSORS = 200
BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 15
EMBED_DIM = 16
DEVICE = torch.device("cpu")

XGB_KWARGS = dict(n_estimators=500, max_depth=6, learning_rate=0.03,
                   subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, reg_lambda=1.0)

# Session 4's own physics-informed-loss result (monotonicity penalty),
# same pool-family precedent this comparison is asked to reference
# (see DEVELOPMENT_LOG.md "Follow-up session 4"). NOTE: that result was
# on the 21-battery pool that predates Stage 4's 42-battery pool, not
# literally the same pool as this comparison - reported alongside, not
# pretending it's the identical evaluation set.
SESSION4_REFERENCE = {
    "VLSTM": {"rmse_baseline": 2.131, "rmse_physics": 2.234, "r2_baseline": 0.806, "r2_physics": 0.787},
    "CNN-LSTM": {"rmse_baseline": 3.948, "rmse_physics": 4.261, "r2_baseline": 0.334, "r2_physics": 0.224},
    "PiFormer": {"rmse_baseline": 2.993, "rmse_physics": 3.043, "r2_baseline": 0.617, "r2_physics": 0.604},
}


def fit_channel_stats(V: np.ndarray, I: np.ndarray):
    return {"V_mean": float(V.mean()), "V_std": float(V.std() + 1e-8),
            "I_mean": float(I.mean()), "I_std": float(I.std() + 1e-8)}


def build_arrays(pool_5tuple: dict, is_keyed_pool=True):
    """pool_5tuple: {bid: (X, soh, rul, idxs, ds)} or {bid: (X, soh, rul, idxs)}
    for held-out. Returns V (N,200), I (N,200), plus per-row keys."""
    Vs, Is, dss, bids, cycs = [], [], [], [], []
    for bid, tup in pool_5tuple.items():
        if is_keyed_pool:
            X, soh, rul, idxs, ds = tup
        else:
            X, soh, rul, idxs = tup
            ds = None
        Vs.append(X[:, :, 0]); Is.append(X[:, :, 1])
        bids += [bid] * len(idxs); cycs += list(idxs)
        dss += [ds] * len(idxs)
    return np.concatenate(Vs).astype(np.float32), np.concatenate(Is).astype(np.float32), dss, bids, cycs


def train_deeponet(V_train, I_train, stats):
    Vn = (V_train - stats["V_mean"]) / stats["V_std"]
    In = (I_train - stats["I_mean"]) / stats["I_std"]
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(Vn))
    n_val = max(1, int(0.15 * len(Vn)))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    query_grid = torch.tensor(np.linspace(0, 1, N_SENSORS), dtype=torch.float32).reshape(1, -1, 1)
    model = DeepONet(n_sensors=N_SENSORS, p=EMBED_DIM, hidden=64).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    In_t, Vn_t = torch.tensor(In), torch.tensor(Vn)
    best_val, best_state = float("inf"), None
    for epoch in range(EPOCHS):
        model.train()
        rng.shuffle(tr_idx)
        tr_losses = []
        for s in range(0, len(tr_idx), BATCH_SIZE):
            b = tr_idx[s:s + BATCH_SIZE]
            q = query_grid.expand(len(b), -1, -1)
            opt.zero_grad()
            pred = model(In_t[b], q)
            loss = nn.functional.mse_loss(pred, Vn_t[b])
            loss.backward(); opt.step()
            tr_losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            q_val = query_grid.expand(len(val_idx), -1, -1)
            val_pred = model(In_t[val_idx], q_val)
            val_loss = nn.functional.mse_loss(val_pred, Vn_t[val_idx]).item()
        print(f"[deeponet] epoch {epoch+1}/{EPOCHS}: train_mse={np.mean(tr_losses):.5f} val_mse={val_loss:.5f}")
        if not np.isfinite(val_loss):
            print("[deeponet] NON-CONVERGENCE: val loss NaN/Inf - stopping early.")
            break
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, val_idx, tr_idx


def eval_reconstruction(model, V, I, stats, label):
    Vn = (V - stats["V_mean"]) / stats["V_std"]
    In = (I - stats["I_mean"]) / stats["I_std"]
    query_grid = torch.tensor(np.linspace(0, 1, N_SENSORS), dtype=torch.float32).reshape(1, -1, 1)
    model.eval()
    with torch.no_grad():
        q = query_grid.expand(len(In), -1, -1)
        pred_n = model(torch.tensor(In), q).numpy()
    pred = pred_n * stats["V_std"] + stats["V_mean"]
    r2 = r2_score(V.flatten(), pred.flatten())
    rmse = float(np.sqrt(mean_squared_error(V.flatten(), pred.flatten())))
    print(f"[deeponet-recon] {label}: R2={r2:.4f} RMSE={rmse:.4f} (V, volts) n_cycles={len(V)}")
    return {"eval_set": label, "r2": r2, "rmse": rmse, "n_cycles": len(V)}


def extract_embeddings(model, I: np.ndarray, stats):
    In = (I - stats["I_mean"]) / stats["I_std"]
    model.eval()
    with torch.no_grad():
        emb = model.branch_embed(torch.tensor(In)).numpy()
    return emb


def fit_eval_xgb(train_df, test_df, feat_cols, label):
    X_train = train_df[feat_cols].to_numpy(dtype=float)
    y_train = train_df["SOH"].to_numpy(dtype=float)
    col_medians = np.nanmedian(X_train, axis=0)
    inds = np.where(np.isnan(X_train))
    X_train = X_train.copy(); X_train[inds] = np.take(col_medians, inds[1])
    model = XGBRegressor(**XGB_KWARGS)
    model.fit(X_train, y_train)

    X_test = test_df[feat_cols].to_numpy(dtype=float).copy()
    inds2 = np.where(np.isnan(X_test))
    X_test[inds2] = np.take(col_medians, inds2[1])
    pred = model.predict(X_test)
    y_test = test_df["SOH"].to_numpy(dtype=float)
    r2 = r2_score(y_test, pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    print(f"[xgb-ablation:{label}] R2={r2:.4f} RMSE={rmse:.4f} n={len(y_test)}")
    return {"label": label, "r2": r2, "rmse": rmse, "n": len(y_test)}, model, col_medians


def main():
    t0 = time.time()
    print("=== Stage 7.3: DeepONet neural-operator SPM surrogate ===")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train, test = load_pool_train_test_keyed()
    print(f"[7.3] pool: {len(train)} train batteries, {len(test)} test batteries")
    V_train, I_train, ds_tr, bid_tr, cyc_tr = build_arrays(train)
    V_test, I_test, ds_te, bid_te, cyc_te = build_arrays(test)
    print(f"[7.3] train cycles: {len(V_train)}, test cycles: {len(V_test)}")

    stats = fit_channel_stats(V_train, I_train)
    model, val_idx, tr_idx = train_deeponet(V_train, I_train, stats)
    torch.save(model.state_dict(), MODEL_DIR / "_experimental_deeponet_spm_surrogate.pt")

    recon_rows = []
    recon_rows.append(eval_reconstruction(model, V_train[val_idx], I_train[val_idx], stats,
                                           "in-pool held-out cycles (val split)"))
    recon_rows.append(eval_reconstruction(model, V_test, I_test, stats, "in-domain (TEST batteries)"))

    held = {}
    for name in ["CALCE", "Oxford", "HUST", "XJTU"]:
        held[name] = load_heldout_keyed(name)
        Vh, Ih, _, _, _ = build_arrays(held[name], is_keyed_pool=False)
        recon_rows.append(eval_reconstruction(model, Vh, Ih, stats, name))

    recon_df = pd.DataFrame(recon_rows)
    recon_df.to_csv(OUT_DIR / "stage7_3_deeponet_reconstruction_results.csv", index=False)
    print("\n=== DeepONet standalone reconstruction (does the operator itself work) ===")
    print(recon_df.to_string(index=False))

    # === Part 2: integration ablation (with vs without DeepONet embedding) ===
    print("\n[7.3] building DeepONet embeddings for the XGBoost-fusion ablation...")
    emb_tr = extract_embeddings(model, I_train, stats)
    emb_te = extract_embeddings(model, I_test, stats)
    deeponet_cols = [f"deeponet_{i}" for i in range(EMBED_DIM)]

    def make_emb_df(ds_list, bid_list, cyc_list, emb):
        df = pd.DataFrame({"dataset": ds_list, "battery_id": bid_list, "cycle_idx": cyc_list})
        for i in range(EMBED_DIM):
            df[deeponet_cols[i]] = emb[:, i]
        return df

    emb_pool_df = pd.concat([make_emb_df(ds_tr, bid_tr, cyc_tr, emb_tr),
                              make_emb_df(ds_te, bid_te, cyc_te, emb_te)], ignore_index=True)

    merged_nm, hi_full = load_nasa_mit_pool(reformulated=True)
    hi_full_ext = add_scv_matd_viect_reformulated(hi_full)
    nasa_mit_ext = hi_full_ext[hi_full_ext["dataset"].isin(["NASA", "MIT"])]
    fusion_df = pd.read_csv(PROC_DIR / "fusion_embeddings.csv")
    merged = pd.merge(nasa_mit_ext, fusion_df, on=["dataset", "battery_id", "cycle_idx"], how="inner")
    n_before = len(merged)
    merged = pd.merge(merged, emb_pool_df, on=["dataset", "battery_id", "cycle_idx"], how="inner")
    print(f"[7.3] 42-pool merge check: {n_before} rows before DeepONet-embedding merge, "
          f"{len(merged)} rows after (should match closely - large drop would signal a key mismatch)")
    assert len(merged) > 0.9 * n_before, "DeepONet-embedding merge lost >10% of rows - key mismatch, not trusting result"

    train_mask, test_mask, split = battery_split_masks(merged)
    base_cols = canonical_feature_cols(reformulated=True)
    extended_cols = extended_canonical_feature_cols(base_cols)
    fcols = fusion_cols()

    train_df, test_df = merged.loc[train_mask], merged.loc[test_mask]
    ablation_rows = []
    r_base, model_base, med_base = fit_eval_xgb(train_df, test_df, extended_cols + fcols,
                                                 "without DeepONet embedding, in-domain")
    ablation_rows.append({"variant": "without_deeponet_embedding", "eval_set": "in-domain (fixed split)", **{
        k: v for k, v in r_base.items() if k != "label"}})
    r_with, model_with, med_with = fit_eval_xgb(train_df, test_df, extended_cols + fcols + deeponet_cols,
                                                 "WITH DeepONet embedding, in-domain")
    ablation_rows.append({"variant": "with_deeponet_embedding", "eval_set": "in-domain (fixed split)", **{
        k: v for k, v in r_with.items() if k != "label"}})

    # zero-retrain on the 3 held-out sets that already have precomputed
    # HI+fusion merges from Stage 5.1 (Oxford/HUST/XJTU), plus CALCE via
    # stage1_common's own established build_calce_merged path.
    for name, parquet_name in [("Oxford", "stage5_1_oxford_merged.parquet"),
                                ("HUST", "stage5_1_hust_merged.parquet"),
                                ("XJTU", "stage5_1_xjtu_merged.parquet")]:
        held_df = pd.read_parquet(PROC_DIR / parquet_name)
        held_df = add_scv_matd_viect_reformulated(held_df)
        Vh, Ih, _, bid_h, cyc_h = build_arrays(held[name], is_keyed_pool=False)
        emb_h = extract_embeddings(model, Ih, stats)
        emb_h_df = pd.DataFrame({"battery_id": bid_h, "cycle_idx": cyc_h})
        for i in range(EMBED_DIM):
            emb_h_df[deeponet_cols[i]] = emb_h[:, i]
        n_before_h = len(held_df)
        held_merged = pd.merge(held_df, emb_h_df, on=["battery_id", "cycle_idx"], how="inner")
        print(f"[7.3] {name} merge check: {n_before_h} rows before, {len(held_merged)} after")
        if len(held_merged) < 0.5 * n_before_h:
            print(f"[7.3] WARNING: {name} DeepONet-embedding merge lost >50% of rows - "
                  f"reporting honestly as unreliable, not silently trusting it.")
            continue

        for label, cols, med, xgb_model in [("without_deeponet_embedding", extended_cols + fcols, med_base, model_base),
                                             ("with_deeponet_embedding", extended_cols + fcols + deeponet_cols, med_with, model_with)]:
            X = held_merged[cols].to_numpy(dtype=float).copy()
            inds = np.where(np.isnan(X))
            X[inds] = np.take(med, inds[1])
            pred = xgb_model.predict(X)
            y_true = held_merged["SOH"].to_numpy(dtype=float)
            r2 = r2_score(y_true, pred)
            rmse = float(np.sqrt(mean_squared_error(y_true, pred)))
            print(f"[xgb-ablation:{label}] {name}: R2={r2:.4f} RMSE={rmse:.4f} n={len(y_true)}")
            ablation_rows.append({"variant": label, "eval_set": name, "r2": r2, "rmse": rmse, "n": len(y_true)})

    # CALCE via the established build_calce_merged path
    calce_merged = build_calce_merged(hi_full_ext)
    Vc, Ic, _, bid_c, cyc_c = build_arrays(held["CALCE"], is_keyed_pool=False)
    emb_c = extract_embeddings(model, Ic, stats)
    emb_c_df = pd.DataFrame({"battery_id": bid_c, "cycle_idx": cyc_c})
    for i in range(EMBED_DIM):
        emb_c_df[deeponet_cols[i]] = emb_c[:, i]
    n_before_c = len(calce_merged)
    calce_merged_emb = pd.merge(calce_merged, emb_c_df, on=["battery_id", "cycle_idx"], how="inner")
    print(f"[7.3] CALCE merge check: {n_before_c} rows before, {len(calce_merged_emb)} after")
    if len(calce_merged_emb) >= 0.5 * n_before_c:
        for label, cols, med, xgb_model in [("without_deeponet_embedding", extended_cols + fcols, med_base, model_base),
                                             ("with_deeponet_embedding", extended_cols + fcols + deeponet_cols, med_with, model_with)]:
            X = calce_merged_emb[cols].to_numpy(dtype=float).copy()
            inds = np.where(np.isnan(X))
            X[inds] = np.take(med, inds[1])
            pred = xgb_model.predict(X)
            y_true = calce_merged_emb["SOH"].to_numpy(dtype=float)
            r2 = r2_score(y_true, pred)
            rmse = float(np.sqrt(mean_squared_error(y_true, pred)))
            print(f"[xgb-ablation:{label}] CALCE: R2={r2:.4f} RMSE={rmse:.4f} n={len(y_true)}")
            ablation_rows.append({"variant": label, "eval_set": "CALCE", "r2": r2, "rmse": rmse, "n": len(y_true)})
    else:
        print("[7.3] WARNING: CALCE DeepONet-embedding merge lost >50% of rows - skipping honestly.")

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(OUT_DIR / "stage7_3_deeponet_xgb_ablation_results.csv", index=False)
    print("\n=== XGBoost-fusion ablation: with vs without DeepONet embedding ===")
    print(ablation_df.to_string(index=False))

    print("\n=== Session 4 reference (shallow physics constraint - monotonicity penalty) ===")
    for k, v in SESSION4_REFERENCE.items():
        print(f"  {k}: RMSE {v['rmse_baseline']}->{v['rmse_physics']}, R2 {v['r2_baseline']}->{v['r2_physics']}")

    print(f"\n[7.3] TOTAL TIME: {(time.time()-t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
