"""
Generic, live (no-lookup) inference pipeline for the Digital Twin
dashboard: given ONE cycle record (in the data_adapters convention -
{"charge": {t,V,I,T}, "discharge": {t,V,I,T}, "discharge_capacity",
"cycle_idx"}), runs the complete fusion-enabled ensemble (SOH) + the
joint-adaptive model (RUL) + per-instance SHAP explanation + OC-SVM
anomaly check + out-of-domain determination, entirely from already-
trained weights - no retraining, ever (matches "a full pipeline rerun
per new upload is fine, skip live incremental meta-learner updating").

Deliberately NOT physics-informed models - the fusion ensemble and the
plain joint-adaptive model, per every instruction in this project since
they diverged from the physics-informed variant.

Uses `data/processed/destandardization_constants.json` and
`data/processed/shap_background.npy` (both precomputed once by
`precompute_app_constants.py`) so this module - and the Streamlit app
built on top of it - never needs to reload the full NASA+MIT battery set
(a multi-minute operation) to answer a single prediction request.
"""

import json
import pickle
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

from health_indicators import compute_health_indicators, HI_NAMES
from sequence_features import get_cycle_tensor, apply_channel_norm, CHANNEL_NAMES
from models.vlstm import VLSTM
from models.cnn_lstm import CNNLSTM
from models.piformer import PiFormer
from models.ica_encoder import ICAEncoder
from models.joint_model import JointSOHRULModel

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

HI_DESCRIPTIONS = {
    "ICHV": "time spent charging near peak voltage (high-voltage/CV-tail duration)",
    "SCV": "average capacity-vs-voltage slope during discharge",
    "VDEDT": "rate of voltage collapse near the end of discharge",
    "VIECT": "voltage reached at a fixed elapsed time into charging",
    "MATC": "mean cell temperature during charging",
    "MATD": "mean cell temperature during discharging",
    "TEVI": "time spent traversing the mid-range of the discharge voltage curve",
}


def load_resources() -> dict:
    """Loads every trained model + precomputed constant ONCE. Callers
    (the Streamlit app) should wrap this in st.cache_resource."""
    with open(PROC_DIR / "bfa_selected_features.txt") as f:
        bfa_selected = [l.strip() for l in f if l.strip()]
    norm_stats = json.loads((PROC_DIR / "channel_norm_stats.json").read_text())
    constants = json.loads((PROC_DIR / "destandardization_constants.json").read_text())
    background = np.load(PROC_DIR / "shap_background.npy")

    hi_df = pd.read_parquet(PROC_DIR / "hi_table.parquet")
    train_hi = hi_df[hi_df["dataset"].isin(["NASA", "MIT"])]
    train_medians = train_hi[bfa_selected].median(numeric_only=True).to_dict()

    vlstm = VLSTM(input_size=1, hidden_size=32, n_targets=1)
    vlstm.load_state_dict(torch.load(MODELS_DIR / "vlstm_soh.pt"))
    vlstm.eval()
    cnn_lstm = CNNLSTM()
    cnn_lstm.load_state_dict(torch.load(MODELS_DIR / "cnn_lstm_soh.pt"))
    cnn_lstm.eval()
    piformer = PiFormer()
    piformer.load_state_dict(torch.load(MODELS_DIR / "piformer_soh.pt"))
    piformer.eval()
    encoder = ICAEncoder(in_channels=3, embed_dim=16)
    encoder.load_state_dict(torch.load(MODELS_DIR / "ica_encoder.pt"))
    encoder.eval()
    joint = JointSOHRULModel()
    joint.load_state_dict(torch.load(MODELS_DIR / "joint_adaptive.pt"))
    joint.eval()

    xgb_fusion = XGBRegressor()
    xgb_fusion.load_model(str(MODELS_DIR / "xgb_soh_fusion.json"))
    with open(MODELS_DIR / "ridge_meta_fusion.pkl", "rb") as f:
        ridge_fusion = pickle.load(f)
    with open(MODELS_DIR / "ocsvm_model.pkl", "rb") as f:
        ocsvm = pickle.load(f)
    with open(MODELS_DIR / "ocsvm_scaler.pkl", "rb") as f:
        ocsvm_scaler = pickle.load(f)
    ocsvm_feature_cols = json.loads((PROC_DIR / "ocsvm_feature_cols.json").read_text())

    return {
        "bfa_selected": bfa_selected, "norm_stats": norm_stats, "constants": constants,
        "background": background, "train_medians": train_medians,
        "vlstm": vlstm, "cnn_lstm": cnn_lstm, "piformer": piformer,
        "encoder": encoder, "joint": joint,
        "xgb_fusion": xgb_fusion, "ridge_fusion": ridge_fusion,
        "ocsvm": ocsvm, "ocsvm_scaler": ocsvm_scaler, "ocsvm_feature_cols": ocsvm_feature_cols,
    }


def _tree_shap_top_features(feat_vector, xgb_fusion, feature_cols, n_top=3):
    explainer = shap.TreeExplainer(xgb_fusion)
    shap_values = explainer.shap_values(feat_vector.reshape(1, -1))[0]
    ranked = sorted(zip(feature_cols, shap_values), key=lambda t: -abs(t[1]))
    top = []
    for name, val in ranked[:n_top]:
        desc = HI_DESCRIPTIONS.get(name, "a learned discharge-curve-shape feature (from the ICA/DV/DC fusion encoder)")
        top.append({"feature": name, "shap_value": float(val), "description": desc})
    return top


def _vlstm_voltage_region(x_raw, x_norm, vlstm, background):
    """Per-instance DeepSHAP on VLSTM's voltage channel, mapped back to
    real volts via this cycle's own (unnormalized) V(t) curve."""
    raw_v_t = x_raw[:, 0].copy()
    bg = torch.tensor(background[:, :, 0:1], dtype=torch.float32)
    test_sample = torch.tensor(x_norm[None, :, 0:1], dtype=torch.float32)
    try:
        explainer = shap.DeepExplainer(vlstm, bg)
        shap_values = explainer.shap_values(test_sample, check_additivity=False)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = np.array(shap_values).reshape(200)
    except Exception as e:
        return None, str(e)

    n_top_bins = max(1, len(shap_values) // 5)
    top_idx = np.argsort(-np.abs(shap_values))[:n_top_bins]
    top_voltages = raw_v_t[top_idx]
    v_lo, v_hi = float(np.percentile(top_voltages, 10)), float(np.percentile(top_voltages, 90))
    frac_mass = float(np.abs(shap_values[top_idx]).sum() / (np.abs(shap_values).sum() + 1e-9))
    return {"v_lo": round(v_lo, 2), "v_hi": round(v_hi, 2), "frac_of_attribution": round(frac_mass, 3)}, None


def predict_and_explain(cycle: dict, res: dict) -> dict:
    """cycle: single cycle record (data_adapters convention). Returns a
    full context dict for display, or {"error": ...} if the cycle is too
    short/malformed for the sequence models."""
    bfa_selected = res["bfa_selected"]
    train_medians = res["train_medians"]

    # 1. Health Indicators (always computable, even for very short cycles)
    his = compute_health_indicators(cycle)
    hi_vector = np.array([
        his[c] if (c in his and not (his[c] != his[c])) else train_medians[c]  # NaN check without pandas
        for c in bfa_selected
    ], dtype=float)
    no_temperature = cycle["discharge"]["T"] is None

    # 2. sequence tensor (needed for VLSTM/CNN-LSTM/PiFormer/fusion/joint/RUL)
    x_raw = get_cycle_tensor(cycle, n_bins=200)
    if x_raw is None:
        return {"error": "This cycle is too short (or its ICA computation failed) for the "
                          "sequence models. Health-Indicator-only analysis isn't supported in "
                          "this dashboard - try a different cycle."}
    x_norm = apply_channel_norm(x_raw[None].astype(np.float32), res["norm_stats"])[0]

    soh_mean, soh_std = res["constants"]["soh_mean"], res["constants"]["soh_std"]
    rul_mean, rul_std = res["constants"]["rul_mean"], res["constants"]["rul_std"]

    def destd(t, mean, std):
        return float(t.detach().numpy().reshape(-1)[0]) * std + mean

    with torch.no_grad():
        pred_vlstm = destd(res["vlstm"](torch.tensor(x_norm[None, :, 0:1])), soh_mean, soh_std)
        pred_cnnlstm = destd(res["cnn_lstm"](torch.tensor(x_norm[None])), soh_mean, soh_std)
        pred_piformer = destd(res["piformer"](torch.tensor(x_norm[None])), soh_mean, soh_std)
        fusion_emb = res["encoder"].encode(torch.tensor(x_norm[None, :, 3:6])).numpy()[0]
        _, pred_rul_z = res["joint"](torch.tensor(x_norm[None]))
        pred_rul = destd(pred_rul_z, rul_mean, rul_std)

    feat_vector = np.concatenate([hi_vector, fusion_emb])
    pred_xgb_fusion = float(res["xgb_fusion"].predict(feat_vector.reshape(1, -1))[0])

    meta_vector = np.concatenate([[pred_xgb_fusion, pred_vlstm, pred_cnnlstm, pred_piformer], fusion_emb])
    pred_soh = float(res["ridge_fusion"].predict(meta_vector.reshape(1, -1))[0])

    soh_half = res["constants"]["soh_conformal_half_width"]
    rul_half = res["constants"]["rul_conformal_half_width"]

    # OC-SVM anomaly check (same 23-feature vector, in the order saved by train_ocsvm.py)
    ocsvm_feat = feat_vector.reshape(1, -1)  # feature_cols == bfa_selected + fusion_cols, same order
    ocsvm_scaled = res["ocsvm_scaler"].transform(ocsvm_feat)
    anomaly_flag = bool(res["ocsvm"].predict(ocsvm_scaled)[0] == -1)

    out_of_domain = no_temperature or anomaly_flag
    domain_reasons = []
    if no_temperature:
        domain_reasons.append("no temperature channel present (this cell's data can't confirm "
                               "it was measured the way NASA/MIT training cells were)")
    if anomaly_flag:
        domain_reasons.append("the One-Class SVM flags this cycle's feature vector as unlike "
                               "anything in the NASA+MIT training data")

    top_features = _tree_shap_top_features(feat_vector, res["xgb_fusion"],
                                            bfa_selected + [f"fusion_{i}" for i in range(16)])
    voltage_region, voltage_region_error = _vlstm_voltage_region(
        x_raw, x_norm, res["vlstm"], res["background"]
    )

    return {
        "soh_pred": round(pred_soh, 1),
        "soh_conformal_lo": round(pred_soh - soh_half, 1),
        "soh_conformal_hi": round(pred_soh + soh_half, 1),
        # clip to >=0 for display - the raw regression output can go
        # slightly negative for deeply past-EOL cycles (e.g. -15 observed
        # on NASA B0005's last logged cycle, true RUL=0), which is a
        # faithful regression residual but a meaningless thing to show a
        # dashboard user ("-15 cycles remaining" isn't interpretable).
        "rul_pred": max(0, round(pred_rul)),
        "rul_conformal_lo": max(0, round(pred_rul - rul_half)),
        "rul_conformal_hi": round(pred_rul + rul_half),
        "top_features": top_features,
        "voltage_region": voltage_region,
        "voltage_region_error": voltage_region_error,
        "anomaly_flag": anomaly_flag,
        "out_of_domain": out_of_domain,
        "domain_reasons": domain_reasons,
        "cycle_idx": cycle["cycle_idx"],
        "true_soh": None,  # filled in by the caller if ground truth is available
        "true_rul": None,
    }
