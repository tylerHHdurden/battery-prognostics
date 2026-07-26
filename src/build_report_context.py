"""
Assembles the structured context an LLM report-writer needs for one
specific (dataset, battery_id, cycle_idx): SOH + RUL point predictions
and conformal intervals from the FUSION-enabled ensemble (NOT the
physics-informed models - explicitly excluded per instruction), plus
per-instance explainability (TreeSHAP on the fusion feature vector +
DeepSHAP voltage-region localization on VLSTM) computed FOR THAT SPECIFIC
ROW, not a global/average explanation.

Two different models supply the two predictions, and that's disclosed
rather than blurred:
  - SOH: Stacking-Ridge-fusion (the fusion ensemble's actual output).
  - RUL: the Phase 4 joint-adaptive model's RUL head - the ONLY trained
    RUL predictor in this project (the fusion ensemble was never built
    to predict RUL; train_ensemble_fusion.py is SOH-only). Its conformal
    interval is the one already calibrated and reported in Phase 6.

Conformal interval widths are the FIXED half-widths already calibrated
in prior sessions (both are GLOBAL, non-adaptive split-conformal per the
CALCE zero-retrain finding - the width does not change per-input):
  SOH: +/- 2.367 (fusion-ensemble calibration, from the CALCE session)
  RUL: +/- 435.7 (joint-adaptive calibration, from Phase 6; 871.417/2)
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))

from data_adapters import iterate_nasa_cycles, iterate_mit_cycles
from sequence_features import build_dataset_tensors, apply_channel_norm, get_cycle_tensor
from ica_dv_dc import compute_ica_dv_dc
from health_indicators import HI_NAMES
from models.vlstm import VLSTM

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
PRED_DIR = PROC_DIR / "predictions"

SOH_CONFORMAL_HALF_WIDTH = 2.367
RUL_CONFORMAL_HALF_WIDTH = 871.417 / 2

HI_DESCRIPTIONS = {
    "ICHV": "time spent charging near peak voltage (high-voltage/CV-tail duration)",
    "SCV": "average capacity-vs-voltage slope during discharge",
    "VDEDT": "rate of voltage collapse near the end of discharge",
    "VIECT": "voltage reached at a fixed elapsed time into charging",
    "MATC": "mean cell temperature during charging",
    "MATD": "mean cell temperature during discharging",
    "TEVI": "time spent traversing the mid-range of the discharge voltage curve",
}


def _load_battery_cycles(dataset: str, battery_id: str, mit_subset: dict):
    if dataset == "NASA":
        return list(iterate_nasa_cycles(battery_id))
    elif dataset == "MIT":
        entry = mit_subset[battery_id]
        return list(iterate_mit_cycles(entry["batch_file"], entry["cell_index"]))
    else:
        raise ValueError(f"unsupported dataset for report context: {dataset}")


def _tree_shap_top_features(dataset, battery_id, cycle_idx, n_top=3):
    """Per-instance TreeSHAP on the XGBoost-fusion model's 23 features
    (7 BFA HIs + 16 fusion embedding dims) for this exact row."""
    with open(PROC_DIR / "bfa_selected_features.txt") as f:
        selected = [l.strip() for l in f if l.strip()]

    hi_df = pd.read_parquet(PROC_DIR / "hi_table.parquet")
    row = hi_df[(hi_df["dataset"] == dataset) & (hi_df["battery_id"] == battery_id)
                & (hi_df["cycle_idx"] == cycle_idx)]
    if row.empty:
        raise ValueError(f"no hi_table row for {dataset}/{battery_id}/cycle {cycle_idx}")

    fusion = pd.read_csv(PROC_DIR / "fusion_embeddings.csv")
    fusion_row = fusion[(fusion["dataset"] == dataset) & (fusion["battery_id"] == battery_id)
                        & (fusion["cycle_idx"] == cycle_idx)]
    fusion_cols = [c for c in fusion.columns if c.startswith("fusion_")]

    train_hi = hi_df[hi_df["dataset"].isin(["NASA", "MIT"])]
    train_medians = train_hi[selected].median(numeric_only=True)

    feature_cols = selected + fusion_cols
    x = row[selected].to_numpy(dtype=float, copy=True)[0]
    for j, col in enumerate(selected):
        if np.isnan(x[j]):
            x[j] = train_medians[col]
    x_fusion = fusion_row[fusion_cols].to_numpy(dtype=float)[0]
    x_full = np.concatenate([x, x_fusion]).reshape(1, -1)

    model = XGBRegressor()
    model.load_model(str(ROOT / "models" / "xgb_soh_fusion.json"))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_full)[0]

    ranked = sorted(zip(feature_cols, shap_values), key=lambda t: -abs(t[1]))
    top = []
    for name, val in ranked[:n_top]:
        desc = HI_DESCRIPTIONS.get(name, "a learned discharge-curve-shape feature (from the ICA/DV/DC fusion encoder)")
        top.append({"feature": name, "shap_value": float(val), "description": desc})
    return top


def _vlstm_voltage_region(dataset, battery_id, cycle_idx, mit_subset, n_background=20):
    """Per-instance DeepSHAP on VLSTM's voltage-sequence input for this
    exact cycle, mapped back to a real voltage range via that cycle's
    own V(t) curve (same method as Phase 5's global voltage-region
    check, applied to a single instance instead of averaged)."""
    cycles = _load_battery_cycles(dataset, battery_id, mit_subset)
    target_cycle = next((c for c in cycles if c["cycle_idx"] == cycle_idx), None)
    if target_cycle is None:
        return None

    norm_stats = json.loads((PROC_DIR / "channel_norm_stats.json").read_text())

    # background: a handful of other cycles from the SAME battery, earlier in life
    background_cycles = [c for c in cycles if c["cycle_idx"] != cycle_idx][:n_background]
    if len(background_cycles) < 5:
        return None

    X_target, *_ = build_dataset_tensors([target_cycle])
    X_bg, *_ = build_dataset_tensors(background_cycles)
    if X_target is None or X_bg is None or len(X_target) == 0:
        return None

    raw_v_t = X_target[0, :, 0].copy()  # real volts, before normalization
    X_target_n = apply_channel_norm(X_target.astype(np.float32), norm_stats)
    X_bg_n = apply_channel_norm(X_bg.astype(np.float32), norm_stats)

    vlstm = VLSTM(input_size=1, hidden_size=32, n_targets=1)
    vlstm.load_state_dict(torch.load(ROOT / "models" / "vlstm_soh.pt"))
    vlstm.eval()

    bg = torch.tensor(X_bg_n[:, :, 0:1], dtype=torch.float32)
    test_sample = torch.tensor(X_target_n[:, :, 0:1], dtype=torch.float32)
    explainer = shap.DeepExplainer(vlstm, bg)
    shap_values = explainer.shap_values(test_sample, check_additivity=False)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.array(shap_values).reshape(200)

    # top-20% most-attributed time bins -> their real voltage values
    n_top_bins = max(1, len(shap_values) // 5)
    top_idx = np.argsort(-np.abs(shap_values))[:n_top_bins]
    top_voltages = raw_v_t[top_idx]
    v_lo, v_hi = float(np.percentile(top_voltages, 10)), float(np.percentile(top_voltages, 90))
    frac_mass = float(np.abs(shap_values[top_idx]).sum() / (np.abs(shap_values).sum() + 1e-9))
    return {"v_lo": round(v_lo, 2), "v_hi": round(v_hi, 2), "frac_of_attribution": round(frac_mass, 3)}


def get_report_context(dataset: str, battery_id: str, cycle_idx: int) -> dict:
    with open(PROC_DIR / "mit_subset.json") as f:
        mit_subset = {e["global_id"]: e for e in json.load(f)}

    ens = pd.read_csv(PRED_DIR / "ensemble_fusion_test_preds.csv")
    row = ens[(ens["dataset"] == dataset) & (ens["battery_id"] == battery_id)
              & (ens["cycle_idx"] == cycle_idx)]
    if row.empty:
        raise ValueError(f"no fusion-ensemble prediction for {dataset}/{battery_id}/cycle {cycle_idx}")
    row = row.iloc[0]
    soh_pred = float(row["pred_Stacking_Ridge_fusion"])
    soh_true = float(row["SOH"])

    rul_df = pd.read_csv(PRED_DIR / "conformal_rul_eval_preds.csv")
    rul_row = rul_df[(rul_df["dataset"] == dataset) & (rul_df["battery_id"] == battery_id)
                      & (rul_df["cycle_idx"] == cycle_idx)]
    if rul_row.empty:
        raise ValueError(f"no RUL prediction for {dataset}/{battery_id}/cycle {cycle_idx} "
                          f"(only available for eval battery subset: b1c4, b3c0, b4c38)")
    rul_row = rul_row.iloc[0]
    rul_pred = float(rul_row["RUL_pred"])
    rul_true = float(rul_row["RUL"])

    top_features = _tree_shap_top_features(dataset, battery_id, cycle_idx)
    voltage_region = _vlstm_voltage_region(dataset, battery_id, cycle_idx, mit_subset)

    return {
        "dataset": dataset, "battery_id": battery_id, "cycle_idx": int(cycle_idx),
        "soh_pred": round(soh_pred, 1), "soh_true": round(soh_true, 1),
        "soh_conformal_lo": round(soh_pred - SOH_CONFORMAL_HALF_WIDTH, 1),
        "soh_conformal_hi": round(soh_pred + SOH_CONFORMAL_HALF_WIDTH, 1),
        "rul_pred": round(rul_pred), "rul_true": round(rul_true),
        "rul_conformal_lo": max(0, round(rul_pred - RUL_CONFORMAL_HALF_WIDTH)),
        "rul_conformal_hi": round(rul_pred + RUL_CONFORMAL_HALF_WIDTH),
        "top_features": top_features,
        "voltage_region": voltage_region,
    }


if __name__ == "__main__":
    import pprint
    ctx = get_report_context("MIT", "b1c4", 674)
    pprint.pprint(ctx)
