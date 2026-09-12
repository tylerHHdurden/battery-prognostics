"""
Targeted improvement pass, Part 3, item 2: does training-time noise
injection close the robustness-margin gap?

SCOPE CORRECTION, found and stated explicitly rather than silently
carried forward: run_sensor_noise_robustness_expanded.py (and this
session's earlier Dataset Expansion report, which called its result
"the fusion ensemble's" noise robustness) actually evaluates ONLY the
LEAN pipeline (ICAEncoder + xgb_soh_fusion_expanded.json) - VLSTM/CNN-
LSTM/PiFormer are never loaded there at all. That was the correct scope
for THAT script (matching session 26's own original design), but it
means Part 3's noise-augmented VLSTM/CNN-LSTM/PiFormer retraining
cannot move that specific number even in principle - the deep models
never touch it. Rather than report a null result against the wrong
metric, this script builds the metric that actually exercises the
retrained models: the FULL 4-branch Stacking-Ridge ensemble (XGBoost-
fusion + VLSTM + CNN-LSTM + PiFormer), evaluated under the SAME 3 noise
levels on the SAME 6 test batteries, run identically for BOTH the
clean-trained deep models (built fresh here, as the correct "before"
reference - no prior run of the FULL ensemble under noise exists to
reuse) and the noise-augmented deep models (the "after"). This is a
genuinely new, more complete metric than the lean-only one, not a
retrofit of an old one - reported as such.

Both conditions' Ridge meta-learners are refit on their own TRAIN
predictions (XGBoost-fusion's TRAIN predictions are UNCHANGED between
conditions - it is never retrained here) - matching
train_ensemble_fusion_expanded.py's exact recipe (4 base preds + 16
fusion-embedding dims, Ridge alpha=1.0).
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))

from data_adapters import iterate_nasa_cycles, iterate_mit_cycles
from health_indicators import compute_health_indicators
from rul_labels import soh_per_cycle
from sequence_features import get_cycle_tensor, apply_channel_norm
from models.ica_encoder import ICAEncoder
from models.vlstm import VLSTM
from models.cnn_lstm import CNNLSTM
from models.piformer import PiFormer
from train_ensemble_fusion_expanded import load_merged_fusion_expanded

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
PRED_DIR = PROC_DIR / "predictions"
OUT_DIR = ROOT / "outputs"

ICA_CHANNEL_SLICE = slice(3, 6)
SEED = 42

NOISE_LEVELS = [
    {"name": "clean (baseline)", "sigma_V": 0.0, "sigma_I": 0.0, "sigma_T": 0.0},
    {"name": "1x BMS-grade", "sigma_V": 0.001, "sigma_I": 0.01, "sigma_T": 0.5},
    {"name": "2x BMS-grade", "sigma_V": 0.002, "sigma_I": 0.02, "sigma_T": 1.0},
    {"name": "5x BMS-grade (stress)", "sigma_V": 0.005, "sigma_I": 0.05, "sigma_T": 2.5},
]

TEST_BATTERIES = [("NASA", "B0018"), ("MIT", "b1c4"), ("MIT", "b2c24"),
                   ("MIT", "b3c0"), ("MIT", "b3c35"), ("MIT", "b4c38")]


def load_cycles(dataset: str, battery_id: str) -> list[dict]:
    if dataset == "NASA":
        return list(iterate_nasa_cycles(battery_id))
    with open(PROC_DIR / "mit_subset.json") as f:
        mit_subset = json.load(f)
    entry = next(e for e in mit_subset if e["global_id"] == battery_id)
    return list(iterate_mit_cycles(entry["batch_file"], entry["cell_index"]))


def add_noise(cycle: dict, sigma_V: float, sigma_I: float, sigma_T: float, rng) -> dict:
    if sigma_V == 0 and sigma_I == 0 and sigma_T == 0:
        return cycle
    out = {"cycle_idx": cycle["cycle_idx"], "discharge_capacity": cycle["discharge_capacity"]}
    for phase in ("charge", "discharge"):
        d = cycle[phase]
        out[phase] = {
            "t": d["t"],
            "V": d["V"] + rng.normal(0, sigma_V, size=d["V"].shape) if sigma_V else d["V"],
            "I": d["I"] + rng.normal(0, sigma_I, size=d["I"].shape) if sigma_I else d["I"],
            "T": d["T"] + rng.normal(0, sigma_T, size=d["T"].shape) if sigma_T else d["T"],
        }
    return out


def fit_ensemble_ridge(deep_train_csv: str) -> tuple[Ridge, list[str]]:
    """Rebuilds the exact train_ensemble_fusion_expanded.py recipe,
    swapping in whichever deep-model TRAIN predictions file is given
    (clean-trained vs noise-augmented) - XGBoost-fusion's own TRAIN
    predictions are unchanged either way."""
    xgb = pd.read_csv(PRED_DIR / "xgb_fusion_expanded_preds.csv")
    xgb = xgb[xgb["split"] == "train"][
        ["dataset", "battery_id", "cycle_idx", "SOH", "RUL", "y_pred_soh_fusion"]
    ].rename(columns={"y_pred_soh_fusion": "pred_XGBoost_fusion"})
    deep = pd.read_csv(PRED_DIR / deep_train_csv).rename(columns={
        "y_pred_VLSTM": "pred_VLSTM", "y_pred_CNNLSTM": "pred_CNNLSTM", "y_pred_PiFormer": "pred_PiFormer",
    })
    fusion = pd.read_csv(PROC_DIR / "fusion_embeddings_expanded.csv")
    fusion_cols = [c for c in fusion.columns if c.startswith("fusion_")]
    merged = pd.merge(xgb, deep[["dataset", "battery_id", "cycle_idx", "pred_VLSTM", "pred_CNNLSTM", "pred_PiFormer"]],
                       on=["dataset", "battery_id", "cycle_idx"], how="inner")
    merged = pd.merge(merged, fusion, on=["dataset", "battery_id", "cycle_idx"], how="inner")
    meta_cols = ["pred_XGBoost_fusion", "pred_VLSTM", "pred_CNNLSTM", "pred_PiFormer"] + fusion_cols
    ridge = Ridge(alpha=1.0).fit(merged[meta_cols].to_numpy(), merged["SOH"].to_numpy())
    return ridge, meta_cols


def evaluate_condition(condition_name: str, vlstm_ckpt: str, cnnlstm_ckpt: str, piformer_ckpt: str,
                        ridge: Ridge, meta_cols: list[str],
                        bfa_selected: list[str], train_medians, norm_stats, encoder):
    xgb_fusion = XGBRegressor()
    xgb_fusion.load_model(str(ROOT / "models" / "xgb_soh_fusion_expanded.json"))

    vlstm = VLSTM(input_size=1, hidden_size=32, n_targets=1)
    vlstm.load_state_dict(torch.load(ROOT / "models" / vlstm_ckpt))
    vlstm.eval()
    cnn_lstm = CNNLSTM()
    cnn_lstm.load_state_dict(torch.load(ROOT / "models" / cnnlstm_ckpt))
    cnn_lstm.eval()
    piformer = PiFormer()
    piformer.load_state_dict(torch.load(ROOT / "models" / piformer_ckpt))
    piformer.eval()

    # y_mean_/y_std_ for de-standardization - recomputed identically to
    # every other resumed script in this project (same fit_ids, same
    # convention, not stored in state_dict()).
    with open(PROC_DIR / "battery_split_expanded.json") as f:
        split = json.load(f)
    from train_deep_models_expanded import load_all_battery_tensors_expanded
    from train_deep_models import make_xy
    battery_data = load_all_battery_tensors_expanded()
    train_ids = [b for b in split["train_ids"] if b in battery_data]
    n_val = max(1, len(train_ids) // 5)
    val_ids = sorted(train_ids)[-n_val:]
    fit_ids = [b for b in train_ids if b not in val_ids]
    _, y_fit_ref, _, _, _, _ = make_xy(battery_data, fit_ids)
    y_mean, y_std = float(y_fit_ref.mean()), float(y_fit_ref.std() + 1e-8)
    for m in (vlstm, cnn_lstm, piformer):
        m.y_mean_, m.y_std_ = y_mean, y_std

    feature_cols = bfa_selected + [f"fusion_{i}" for i in range(16)]
    rng = np.random.default_rng(SEED)
    level_results = []
    per_battery_results = []

    print(f"\n[noiseaug-eval] loading raw test-battery cycles ONCE for condition={condition_name}...")
    all_cycles = {(ds, bid): load_cycles(ds, bid) for ds, bid in TEST_BATTERIES}

    for level in NOISE_LEVELS:
        y_true_all, y_pred_ensemble_all = [], []
        for dataset, battery_id in TEST_BATTERIES:
            cycles = all_cycles[(dataset, battery_id)]
            soh_map = soh_per_cycle(cycles)
            noisy_cycles = [add_noise(c, level["sigma_V"], level["sigma_I"], level["sigma_T"], rng)
                             for c in cycles]

            hi_rows, tensors, y_true = [], [], []
            for c in noisy_cycles:
                his = compute_health_indicators(c)
                tensor = get_cycle_tensor(c)
                if tensor is None:
                    continue
                hi_rows.append(his)
                tensors.append(tensor)
                y_true.append(soh_map[c["cycle_idx"]])

            hi_df_noisy = pd.DataFrame(hi_rows)[bfa_selected]
            for col in bfa_selected:
                hi_df_noisy[col] = hi_df_noisy[col].fillna(train_medians[col])

            X_all = apply_channel_norm(np.stack(tensors).astype(np.float32), norm_stats)
            with torch.no_grad():
                emb = encoder.encode(torch.tensor(X_all[:, :, ICA_CHANNEL_SLICE])).numpy()
                pred_vlstm_raw = vlstm(torch.tensor(X_all[:, :, 0:1])).squeeze(-1).numpy()
                pred_cnnlstm_raw = cnn_lstm(torch.tensor(X_all)).squeeze(-1).numpy()
                pred_piformer_raw = piformer(torch.tensor(X_all)).squeeze(-1).numpy()
            pred_vlstm = pred_vlstm_raw * y_std + y_mean
            pred_cnnlstm = pred_cnnlstm_raw * y_std + y_mean
            pred_piformer = pred_piformer_raw * y_std + y_mean

            feat_df = hi_df_noisy.reset_index(drop=True)
            for i in range(16):
                feat_df[f"fusion_{i}"] = emb[:, i]
            pred_xgb_fusion = xgb_fusion.predict(feat_df[feature_cols].to_numpy(dtype=float))

            meta_X = np.column_stack(
                [pred_xgb_fusion, pred_vlstm, pred_cnnlstm, pred_piformer] + [emb[:, i] for i in range(16)]
            )
            pred_ensemble = ridge.predict(meta_X)

            y_true_arr, pred_arr = np.array(y_true), np.array(pred_ensemble)
            batt_r2 = float(r2_score(y_true_arr, pred_arr)) if len(y_true_arr) > 1 else float("nan")
            batt_rmse = float(np.sqrt(mean_squared_error(y_true_arr, pred_arr)))
            per_battery_results.append({"condition": condition_name, "noise_level": level["name"],
                                         "dataset": dataset, "battery_id": battery_id,
                                         "n": len(y_true_arr), "rmse": batt_rmse, "r2": batt_r2})
            y_true_all.extend(y_true)
            y_pred_ensemble_all.extend(pred_ensemble.tolist())

        y_true_all, y_pred_ensemble_all = np.array(y_true_all), np.array(y_pred_ensemble_all)
        rmse = float(np.sqrt(mean_squared_error(y_true_all, y_pred_ensemble_all)))
        mae = float(mean_absolute_error(y_true_all, y_pred_ensemble_all))
        r2 = float(r2_score(y_true_all, y_pred_ensemble_all))
        print(f"[noiseaug-eval] [{condition_name}] {level['name']}: n={len(y_true_all)} "
              f"RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")
        level_results.append({"condition": condition_name, "noise_level": level["name"],
                               "n": len(y_true_all), "rmse": rmse, "mae": mae, "r2": r2})

    return pd.DataFrame(level_results), pd.DataFrame(per_battery_results)


def main():
    with open(PROC_DIR / "bfa_selected_features_expanded.txt") as f:
        bfa_selected = [l.strip() for l in f if l.strip()]
    norm_stats = json.loads((PROC_DIR / "channel_norm_stats_expanded.json").read_text())
    hi_df = pd.read_parquet(PROC_DIR / "hi_table_expanded.parquet")
    train_medians = hi_df[hi_df["dataset"].isin(["NASA", "MIT"])][bfa_selected].median(numeric_only=True)
    encoder = ICAEncoder(in_channels=3, embed_dim=16)
    encoder.load_state_dict(torch.load(ROOT / "models" / "ica_encoder_expanded.pt"))
    encoder.eval()

    print("[noiseaug-eval] === Refitting Ridge for CLEAN-trained deep models (the 'before' "
          "reference - this exact full-4-branch-under-noise metric has no prior run to reuse) ===")
    ridge_clean, meta_cols = fit_ensemble_ridge("deep_models_expanded_train_preds.csv")
    print("[noiseaug-eval] === Refitting Ridge for NOISE-AUGMENTED deep models (the 'after') ===")
    ridge_noiseaug, _ = fit_ensemble_ridge("deep_models_noiseaug_expanded_train_preds.csv")

    level_clean, batt_clean = evaluate_condition(
        "clean-trained", "vlstm_soh_expanded.pt", "cnn_lstm_soh_expanded.pt", "piformer_soh_expanded.pt",
        ridge_clean, meta_cols, bfa_selected, train_medians, norm_stats, encoder,
    )
    level_noiseaug, batt_noiseaug = evaluate_condition(
        "noise-augmented", "vlstm_soh_noiseaug_expanded.pt", "cnn_lstm_soh_noiseaug_expanded.pt",
        "piformer_soh_noiseaug_expanded.pt",
        ridge_noiseaug, meta_cols, bfa_selected, train_medians, norm_stats, encoder,
    )

    combined = pd.concat([level_clean, level_noiseaug], ignore_index=True)
    combined_batt = pd.concat([batt_clean, batt_noiseaug], ignore_index=True)
    print("\n[noiseaug-eval] === SUMMARY: clean-trained vs. noise-augmented, "
          "full 4-branch ensemble, same 6 test batteries ===")
    print(combined.to_string(index=False))

    pivot = combined.pivot(index="noise_level", columns="condition", values="r2")
    pivot = pivot.reindex([lvl["name"] for lvl in NOISE_LEVELS])
    print("\n[noiseaug-eval] R2 by noise level, clean-trained vs. noise-augmented:")
    print(pivot.to_string())
    margin_clean = pivot["clean-trained"].iloc[0] - pivot["clean-trained"].iloc[-1]
    margin_noiseaug = pivot["noise-augmented"].iloc[0] - pivot["noise-augmented"].iloc[-1]
    print(f"\n[noiseaug-eval] Robustness margin (clean R2 - 5x-stress R2): "
          f"clean-trained={margin_clean:.4f}, noise-augmented={margin_noiseaug:.4f} - "
          f"{'NARROWER (improved)' if margin_noiseaug < margin_clean else 'WIDER (did not help)'} "
          f"margin with training-time noise injection.")

    print("\n[noiseaug-eval] NASA/B0018 specifically:")
    b18 = combined_batt[combined_batt.battery_id == "B0018"]
    print(b18.to_string(index=False))

    combined.to_csv(OUT_DIR / "sensor_noise_robustness_noiseaug_expanded_summary.csv", index=False)
    combined_batt.to_csv(OUT_DIR / "sensor_noise_robustness_noiseaug_expanded_per_battery.csv", index=False)
    print("\n[noiseaug-eval] DONE")


if __name__ == "__main__":
    main()
