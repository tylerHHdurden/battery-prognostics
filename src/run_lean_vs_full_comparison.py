"""
Session 20: "lean" deployment variant vs. the full 5-branch ensemble.

Motivation, stated plainly: across sessions 9 (drop-branch ablation),
17 (CNN-BiGRU added as a 5th branch), and every base-learner comparison
in between, this project has repeatedly found that the Ridge meta-
learner's reliance on ANY of the 4 deep sequence models
(VLSTM/CNN-LSTM/PiFormer/CNN-BiGRU) is negligible-to-negative once
XGBoost-fusion + the 16-dim fusion embedding are present. This session
asks the practical question that evidence implies: if the deep models
add ~nothing, what do you actually save by shipping without them - and
does the "lean" (XGBoost-fusion only) variant genuinely match the full
ensemble's accuracy, confirmed directly rather than assumed?

No base learner retrained. The "full 5-branch" Ridge meta-learner
(pred_XGBoost_fusion + pred_VLSTM + pred_CNNLSTM + pred_PiFormer +
pred_CNNBiGRU + 16 fusion dims) was fit in-memory-only during session
17's ablation and never persisted to disk - refit here (a one-line
Ridge.fit on already-computed predictions, NOT a retrain of any base
learner) and saved to models/ridge_meta_fusion_5branch.pkl so "full" has
a genuine, complete file set to measure model size against, exactly
matching session 17's already-reported full_5_branch numbers (R2=0.916884).
"""

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))

from train_deep_models import load_all_battery_tensors, make_xy
from sequence_features import apply_channel_norm
from run_drop_branch_ablation_5branch import load_merged_fusion_5branch, BASE_COLS_5
from models.vlstm import VLSTM
from models.cnn_lstm import CNNLSTM
from models.piformer import PiFormer
from models.cnn_bigru import CNNBiGRU
from models.ica_encoder import ICAEncoder

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
PRED_DIR = PROC_DIR / "predictions"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

N_LATENCY_SAMPLES = 300  # single-instance timing repetitions, not a full-test-set batch


def metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# --------------------------------------------------------------------------
# (1) Accuracy - recomputed fresh from the already-trained models, not
# assumed from prior sessions' CSVs (per the task's explicit instruction).
# --------------------------------------------------------------------------

def accuracy_comparison():
    with open(PROC_DIR / "bfa_selected_features.txt") as f:
        BFA_SELECTED = [l.strip() for l in f if l.strip()]
    fusion_cols = [f"fusion_{i}" for i in range(16)]

    # --- LEAN: XGBoost-fusion alone, freshly re-predicted ---
    hi_df = pd.read_parquet(PROC_DIR / "hi_table.parquet")
    hi_df = hi_df[hi_df["dataset"].isin(["NASA", "MIT"])].reset_index(drop=True)
    fusion_df = pd.read_csv(PROC_DIR / "fusion_embeddings.csv")  # non-MMD, the shipped fusion embedding
    merged = pd.merge(hi_df, fusion_df, on=["dataset", "battery_id", "cycle_idx"], how="inner")

    split = json.loads((PROC_DIR / "battery_split.json").read_text())
    test_mask = merged["battery_id"].isin(split["test_ids"]).to_numpy()

    feature_cols = BFA_SELECTED + fusion_cols
    X = merged[feature_cols].to_numpy(dtype=float, copy=True)
    col_medians = np.nanmedian(X, axis=0)  # same imputation convention as train_xgboost_fusion.py
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_medians, inds[1])
    y_test = merged.loc[test_mask, "SOH"].to_numpy()

    xgb_fusion = XGBRegressor()
    xgb_fusion.load_model(str(ROOT / "models" / "xgb_soh_fusion.json"))
    pred_lean = xgb_fusion.predict(X[test_mask])
    m_lean = metrics(y_test, pred_lean)
    print(f"[lean-vs-full] LEAN (XGBoost-fusion only), freshly recomputed: {m_lean}")

    # cross-check against the already-saved metrics file from when this
    # model was originally trained - should match almost exactly (same
    # deterministic model, same input features), confirming this is a
    # genuine re-derivation and not a copy-pasted number
    saved = pd.read_csv(PRED_DIR / "xgb_fusion_metrics.csv").iloc[0]
    print(f"[lean-vs-full] (cross-check vs. originally-saved xgb_fusion_metrics.csv: "
          f"rmse={saved['rmse']:.4f} mae={saved['mae']:.4f} r2={saved['r2']:.4f})")

    # --- FULL: 5-branch Ridge meta-learner, refit here (Ridge only, no
    # base learner retrained) since session 17 never persisted it ---
    train_df, fusion_cols_check = load_merged_fusion_5branch("train")
    test_df, _ = load_merged_fusion_5branch("test")
    assert fusion_cols_check == fusion_cols

    full_cols = BASE_COLS_5 + fusion_cols
    ridge_full = Ridge(alpha=1.0).fit(train_df[full_cols], train_df["SOH"])
    pred_full = ridge_full.predict(test_df[full_cols])
    m_full = metrics(test_df["SOH"], pred_full)
    print(f"[lean-vs-full] FULL (5-branch ensemble: XGBoost-fusion + VLSTM + CNNLSTM + "
          f"PiFormer + CNNBiGRU, Ridge meta-learner): {m_full}")
    print(f"[lean-vs-full] (cross-check vs. session 17's full_5_branch: rmse=1.39433 r2=0.916884)")

    with open(ROOT / "models" / "ridge_meta_fusion_5branch.pkl", "wb") as f:
        pickle.dump(ridge_full, f)
    print(f"[lean-vs-full] persisted models/ridge_meta_fusion_5branch.pkl "
          f"(previously in-memory-only in session 17)")

    delta_rmse = m_lean["rmse"] - m_full["rmse"]
    delta_r2 = m_lean["r2"] - m_full["r2"]
    print(f"[lean-vs-full] LEAN vs FULL delta: delta_rmse={delta_rmse:+.4f}, delta_r2={delta_r2:+.4f}")

    return m_lean, m_full, delta_rmse, delta_r2


# --------------------------------------------------------------------------
# (2) Inference latency - single-instance (batch_size=1) timing, the
# realistic unit for a live one-cycle-at-a-time prediction (matching how
# app.py/live_inference.py actually serves a prediction), not a batched
# throughput number. Warm-up pass excluded from timing (standard
# benchmarking practice - first call pays one-off allocation/cache costs
# that a real deployed service wouldn't repeat every request).
# --------------------------------------------------------------------------

def latency_comparison():
    battery_data = load_all_battery_tensors()
    split = json.loads((PROC_DIR / "battery_split.json").read_text())
    test_ids = [b for b in split["test_ids"] if b in battery_data]
    train_ids = [b for b in split["train_ids"] if b in battery_data]
    n_val = max(1, len(train_ids) // 5)
    val_ids = sorted(train_ids)[-n_val:]
    fit_ids = [b for b in train_ids if b not in val_ids]

    X_test, y_test, _, _, _, _ = make_xy(battery_data, test_ids)
    norm_stats = json.loads((PROC_DIR / "channel_norm_stats.json").read_text())
    X_test = apply_channel_norm(X_test, norm_stats)

    _, y_fit_ref, _, _, _, _ = make_xy(battery_data, fit_ids)
    y_mean, y_std = float(y_fit_ref.mean()), float(y_fit_ref.std() + 1e-8)

    encoder = ICAEncoder(in_channels=3, embed_dim=16)
    encoder.load_state_dict(torch.load(ROOT / "models" / "ica_encoder.pt"))
    encoder.eval()

    xgb_fusion = XGBRegressor()
    xgb_fusion.load_model(str(ROOT / "models" / "xgb_soh_fusion.json"))

    vlstm = VLSTM(input_size=1, hidden_size=32, n_targets=1)
    vlstm.load_state_dict(torch.load(ROOT / "models" / "vlstm_soh.pt"))
    vlstm.eval()
    cnn_lstm = CNNLSTM()
    cnn_lstm.load_state_dict(torch.load(ROOT / "models" / "cnn_lstm_soh.pt"))
    cnn_lstm.eval()
    piformer = PiFormer()
    piformer.load_state_dict(torch.load(ROOT / "models" / "piformer_soh.pt"))
    piformer.eval()
    cnn_bigru = CNNBiGRU()
    cnn_bigru.load_state_dict(torch.load(ROOT / "models" / "cnn_bigru_soh.pt"))
    cnn_bigru.eval()

    with open(ROOT / "models" / "ridge_meta_fusion_5branch.pkl", "rb") as f:
        ridge_full = pickle.load(f)
    full_cols = BASE_COLS_5 + [f"fusion_{i}" for i in range(16)]  # exact column names/order ridge_full was fit on

    with open(PROC_DIR / "bfa_selected_features.txt") as f:
        BFA_SELECTED = [l.strip() for l in f if l.strip()]
    hi_df = pd.read_parquet(PROC_DIR / "hi_table.parquet")
    hi_medians = hi_df[hi_df["dataset"].isin(["NASA", "MIT"])][BFA_SELECTED].median(numeric_only=True)
    # generic placeholder HI vector (median values) - latency timing only
    # cares about the SHAPE/cost of the XGBoost predict call, not the
    # exact HI values (those are computed upstream by health_indicators.py
    # identically for both variants, so are NOT part of what differs
    # between lean and full and are correctly excluded from this timing).
    hi_vec = hi_medians.to_numpy(dtype=float).reshape(1, -1)

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X_test), size=min(N_LATENCY_SAMPLES, len(X_test)), replace=False)

    def lean_predict(x_single):
        with torch.no_grad():
            emb = encoder.encode(torch.tensor(x_single[None, :, 3:6])).numpy()
        feat = np.concatenate([hi_vec, emb], axis=1)
        return xgb_fusion.predict(feat)

    def full_predict(x_single):
        with torch.no_grad():
            emb = encoder.encode(torch.tensor(x_single[None, :, 3:6])).numpy()
            p_v = (vlstm(torch.tensor(x_single[None, :, 0:1])).item()) * y_std + y_mean
            p_c = (cnn_lstm(torch.tensor(x_single[None])).item()) * y_std + y_mean
            p_p = (piformer(torch.tensor(x_single[None])).item()) * y_std + y_mean
            p_g = (cnn_bigru(torch.tensor(x_single[None])).item()) * y_std + y_mean
        feat = np.concatenate([hi_vec, emb], axis=1)
        p_x = xgb_fusion.predict(feat)[0]
        # DataFrame with the exact column names/order ridge_full was fit
        # on (train_df[full_cols]) - a plain ndarray works identically
        # numerically but sklearn warns on every call otherwise, since
        # this Ridge was fit on named columns.
        meta_x = pd.DataFrame([[p_x, p_v, p_c, p_p, p_g] + list(emb[0])], columns=full_cols)
        return ridge_full.predict(meta_x)

    # warm-up (excluded from timing)
    for i in sample_idx[:5]:
        lean_predict(X_test[i])
        full_predict(X_test[i])

    lean_times, full_times = [], []
    for i in sample_idx:
        t0 = time.perf_counter()
        lean_predict(X_test[i])
        lean_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        full_predict(X_test[i])
        full_times.append(time.perf_counter() - t0)

    lean_times, full_times = np.array(lean_times), np.array(full_times)
    print(f"\n[lean-vs-full] latency over {len(sample_idx)} single-instance predictions "
          f"(batch_size=1, warm-up excluded):")
    print(f"[lean-vs-full] LEAN: mean={lean_times.mean()*1000:.3f}ms median={np.median(lean_times)*1000:.3f}ms "
          f"std={lean_times.std()*1000:.3f}ms")
    print(f"[lean-vs-full] FULL: mean={full_times.mean()*1000:.3f}ms median={np.median(full_times)*1000:.3f}ms "
          f"std={full_times.std()*1000:.3f}ms")
    speedup = full_times.mean() / lean_times.mean()
    print(f"[lean-vs-full] LEAN is {speedup:.2f}x faster than FULL, on average")

    return lean_times, full_times, speedup


# --------------------------------------------------------------------------
# (3) Model size on disk
# --------------------------------------------------------------------------

def model_size_comparison():
    lean_files = ["xgb_soh_fusion.json", "ica_encoder.pt"]
    full_files = lean_files + ["vlstm_soh.pt", "cnn_lstm_soh.pt", "piformer_soh.pt",
                                "cnn_bigru_soh.pt", "ridge_meta_fusion_5branch.pkl"]

    def total_size(files):
        rows = []
        total = 0
        for fn in files:
            p = ROOT / "models" / fn
            size = p.stat().st_size
            rows.append((fn, size))
            total += size
        return total, rows

    lean_total, lean_rows = total_size(lean_files)
    full_total, full_rows = total_size(full_files)

    print(f"\n[lean-vs-full] model size on disk:")
    print(f"[lean-vs-full] LEAN ({len(lean_files)} files): {lean_total/1024:.1f} KB")
    for fn, size in lean_rows:
        print(f"    {fn}: {size/1024:.1f} KB")
    print(f"[lean-vs-full] FULL ({len(full_files)} files): {full_total/1024:.1f} KB")
    for fn, size in full_rows:
        print(f"    {fn}: {size/1024:.1f} KB")
    print(f"[lean-vs-full] LEAN is {full_total/lean_total:.1f}x smaller than FULL")

    return lean_total, full_total, lean_rows, full_rows


# --------------------------------------------------------------------------
# (4) Pipeline complexity - number of distinct trained-model invocations
# needed to go from one cycle's features to one final SOH prediction.
# --------------------------------------------------------------------------

def complexity_comparison():
    lean_steps = ["ICAEncoder.encode", "XGBoost-fusion.predict"]
    full_steps = lean_steps + ["VLSTM.forward", "CNN-LSTM.forward", "PiFormer.forward",
                                "CNN-BiGRU.forward", "Ridge-meta.predict"]
    print(f"\n[lean-vs-full] pipeline complexity (model invocations per prediction):")
    print(f"[lean-vs-full] LEAN: {len(lean_steps)} steps: {lean_steps}")
    print(f"[lean-vs-full] FULL: {len(full_steps)} steps: {full_steps}")
    return lean_steps, full_steps


def main():
    m_lean, m_full, delta_rmse, delta_r2 = accuracy_comparison()
    lean_times, full_times, speedup = latency_comparison()
    lean_size, full_size, lean_rows, full_rows = model_size_comparison()
    lean_steps, full_steps = complexity_comparison()

    summary = pd.DataFrame([
        {"variant": "lean_xgboost_fusion", "rmse": m_lean["rmse"], "mae": m_lean["mae"], "r2": m_lean["r2"],
         "mean_latency_ms": float(lean_times.mean() * 1000), "model_size_kb": lean_size / 1024,
         "n_model_invocations": len(lean_steps)},
        {"variant": "full_5branch_ensemble", "rmse": m_full["rmse"], "mae": m_full["mae"], "r2": m_full["r2"],
         "mean_latency_ms": float(full_times.mean() * 1000), "model_size_kb": full_size / 1024,
         "n_model_invocations": len(full_steps)},
    ])
    summary.to_csv(OUT_DIR / "lean_vs_full_comparison.csv", index=False)

    print("\n[lean-vs-full] === FINAL SUMMARY TABLE ===")
    print(summary.to_string(index=False))
    print(f"\n[lean-vs-full] accuracy delta (lean - full): delta_rmse={delta_rmse:+.4f}, delta_r2={delta_r2:+.4f}")
    print(f"[lean-vs-full] speedup: {speedup:.2f}x, size reduction: {full_size/lean_size:.1f}x, "
          f"invocation reduction: {len(full_steps)}->{len(lean_steps)}")
    print("[lean-vs-full] DONE")


if __name__ == "__main__":
    main()
