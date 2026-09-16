"""
Generic, live (no-lookup) inference pipeline for the Digital Twin
dashboard: given ONE cycle record (in the data_adapters convention -
{"charge": {t,V,I,T}, "discharge": {t,V,I,T}, "discharge_capacity",
"cycle_idx"}), runs XGBoost-fusion (SOH, lean deployment) + the RUL
joint-fusion model + per-instance SHAP explanation + OC-SVM anomaly
check + out-of-domain determination, entirely from already-trained
weights - no retraining, ever.

STAGE 4 REWIRE (session: Stage 4 promotion), stated explicitly since
this changes what was actually running before: prior to this pass the
deployed app used the FULL 4-branch ensemble (VLSTM/CNN-LSTM/PiFormer +
XGBoost-fusion, combined via a Ridge meta-learner) for SOH, and the
OLD, non-fusion `JointSOHRULModel` for RUL - despite session 20's lean
decision and session 41's fusion joint model both being recommended
long before this. This module now implements what was actually decided:
- SOH: XGBoost-fusion ALONE (lean, session 20 + Stage 3.3's NCL result -
  no ensemble reconfiguration warranted). CNN-LSTM/PiFormer/the Ridge
  meta-learner are no longer loaded at all.
- RUL: JointSOHRULModelFusion (session 41's HI-fused architecture),
  replacing the old bare JointSOHRULModel.
- VLSTM stays loaded, but ONLY for its SHAP voltage-region
  explainability feature - not part of either prediction anymore.
- Features: Stage 1's canonical 1.1-reformulated 8-feature set (+
  cycle_idx, 1.5's monotone-constrained feature) instead of the older,
  pre-Stage-1 7-feature bfa_selected_features.txt set.

Reformulated (`_rel`) duration features need a per-battery BASELINE (
that battery's own cycle-10 raw value) - `predict_and_explain` accepts
an optional `baseline_his` dict for this (the caller looks it up once
per battery selection, not on every cycle - see app.py). Without it,
the function falls back to using the CURRENT cycle's own raw value as
its baseline (ratio=1.0) - a defensive, disclosed degradation for
contexts with no other cycle available, not a silent wrong answer.

Uses `data/processed/destandardization_constants.json` and
`data/processed/shap_background.npy` (both precomputed once by
`precompute_app_constants.py`) so this module - and the Streamlit app
built on top of it - never needs to reload the full NASA+MIT battery
set (a multi-minute operation) to answer a single prediction request.
"""

from __future__ import annotations

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
from models.ica_encoder import ICAEncoder
from stage1_common import canonical_feature_cols, fusion_cols as _fusion_cols, DURATION_FEATURES, BASELINE_CYCLE
from run_stage1_followup_partB_joint_rul import JointSOHRULModelFusion

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

CANONICAL_RAW = canonical_feature_cols(reformulated=False)   # e.g. ICHV, SCV, ...
CANONICAL_REL = canonical_feature_cols(reformulated=True)    # e.g. ICHV_rel, SCV, ...
FEATURE_COLS = CANONICAL_REL + ["cycle_idx"]

HI_DESCRIPTIONS = {
    "ICHV_rel": "time spent charging near peak voltage (high-voltage/CV-tail duration), relative to this battery's own cycle-10 baseline",
    "SCV": "average capacity-vs-voltage slope during discharge",
    "VDEDT": "rate of voltage collapse near the end of discharge",
    "VIECT": "voltage reached at a fixed elapsed time into charging",
    "MATD": "mean cell temperature during discharging",
    "MET": "mean energy during test (average of charge and discharge energy)",
    "TEVD_rel": "time elapsed until discharge voltage first falls to 50%, relative to this battery's own cycle-10 baseline",
    "TEVI_rel": "time spent traversing the mid-range of the discharge voltage curve, relative to this battery's own cycle-10 baseline",
    "cycle_idx": "cycle number (monotone-constrained: the model is constrained to never predict higher SOH for a later cycle)",
}


def load_resources() -> dict:
    """Loads every trained model + precomputed constant ONCE. Callers
    (the Streamlit app) should wrap this in st.cache_resource."""
    norm_stats = json.loads((PROC_DIR / "channel_norm_stats.json").read_text())
    constants = json.loads((PROC_DIR / "destandardization_constants.json").read_text())
    background = np.load(PROC_DIR / "shap_background.npy")
    # the RUL joint-fusion model's OWN HI feature z-score stats (8
    # canonical _rel-ratio features, NO cycle_idx - see build_hi_features/
    # run_stage1_followup_partB_joint_rul.py; this is a DIFFERENT, smaller
    # feature vector than XGBoost-fusion's own 9-feature (+cycle_idx) one,
    # and is separately z-scored where XGBoost's is not - both facts
    # caught by a real shape-mismatch crash in this project's own
    # pre-promotion smoke test, not assumed).
    joint_hi_norm = json.loads((PROC_DIR / "joint_hi_norm_stats.json").read_text())

    hi_df = pd.read_parquet(PROC_DIR / "hi_table.parquet")
    train_hi = hi_df[hi_df["dataset"].isin(["NASA", "MIT"])]
    train_medians = train_hi[CANONICAL_RAW].median(numeric_only=True).to_dict()

    vlstm = VLSTM(input_size=1, hidden_size=32, n_targets=1)
    vlstm.load_state_dict(torch.load(MODELS_DIR / "vlstm_soh.pt"))
    vlstm.eval()
    encoder = ICAEncoder(in_channels=3, embed_dim=16)
    encoder.load_state_dict(torch.load(MODELS_DIR / "ica_encoder.pt"))
    encoder.eval()
    joint_fusion = JointSOHRULModelFusion(n_hi_features=len(CANONICAL_REL))
    joint_fusion.load_state_dict(torch.load(MODELS_DIR / "joint_adaptive_fusion.pt"))
    joint_fusion.eval()

    xgb_fusion = XGBRegressor()
    xgb_fusion.load_model(str(MODELS_DIR / "xgb_soh_fusion.json"))

    with open(MODELS_DIR / "ocsvm_model.pkl", "rb") as f:
        ocsvm = pickle.load(f)
    with open(MODELS_DIR / "ocsvm_scaler.pkl", "rb") as f:
        ocsvm_scaler = pickle.load(f)
    ocsvm_feature_cols = json.loads((PROC_DIR / "ocsvm_feature_cols.json").read_text())

    return {
        "norm_stats": norm_stats, "constants": constants, "joint_hi_norm": joint_hi_norm,
        "background": background, "train_medians": train_medians,
        "vlstm": vlstm, "encoder": encoder, "joint_fusion": joint_fusion,
        "xgb_fusion": xgb_fusion,
        "ocsvm": ocsvm, "ocsvm_scaler": ocsvm_scaler, "ocsvm_feature_cols": ocsvm_feature_cols,
        # kept for StreamingDigitalTwin's constructor API (digital_twin_streaming.py) -
        # the canonical reformulated names now, not the old pre-Stage-1 set; no longer
        # used for feature-BUILDING directly (build_reformulated_hi_vector handles that)
        "bfa_selected": CANONICAL_REL,
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
    real volts via this cycle's own (unnormalized) V(t) curve. VLSTM is
    used HERE ONLY, as an explainability tool - not part of either the
    SOH or RUL prediction (see module docstring)."""
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


def build_reformulated_hi_vector(his: dict, train_medians: dict, baseline_his: dict | None) -> np.ndarray:
    """Builds the canonical 1.1-reformulated feature vector (8 features,
    NOT including cycle_idx - added separately by the caller) for ONE
    cycle's raw HI dict.

    baseline_his: this SAME battery's own cycle-10 (BASELINE_CYCLE) raw
    HI dict, as returned by compute_health_indicators - looked up ONCE
    per battery selection by the caller (app.py), not recomputed here.
    If None (no other cycle available in this context), falls back to
    using THIS cycle's own raw value as its baseline (ratio=1.0 for
    every duration feature) - a disclosed degradation, not silently
    wrong: this only fires when the caller genuinely has no other cycle
    to offer (e.g. a single isolated cycle with no battery history)."""
    vec = np.zeros(len(CANONICAL_REL), dtype=float)
    for i, col in enumerate(CANONICAL_REL):
        raw_feat = col[:-4] if col.endswith("_rel") else col
        raw_val = his.get(raw_feat)
        if raw_val is None or raw_val != raw_val:  # NaN check without pandas
            raw_val = train_medians.get(raw_feat, 0.0)

        if col.endswith("_rel"):
            base_val = None
            if baseline_his is not None:
                base_val = baseline_his.get(raw_feat)
                if base_val is not None and (base_val != base_val or abs(base_val) < 1e-6):
                    base_val = None
            if base_val is None:
                base_val = raw_val if abs(raw_val) >= 1e-6 else train_medians.get(raw_feat, 1.0)
            vec[i] = raw_val / base_val if abs(base_val) >= 1e-6 else 1.0
        else:
            vec[i] = raw_val
    return vec


PRECOMPUTED_HELDOUT_PARQUETS = {
    "Oxford": "stage5_1_oxford_merged.parquet",
    "HUST": "stage5_1_hust_merged.parquet",
    "XJTU": "stage5_1_xjtu_merged.parquet",
}


def _calce_fusion_embeddings(battery_id: str, encoder) -> np.ndarray | None:
    """CALCE has no precomputed fusion_embeddings.csv row (unlike NASA/
    MIT/Oxford/HUST/XJTU), but DOES have its 3-channel ICA/DV/DC
    differential tensor cached (data/processed/differential_tensors/
    CALCE_{id}.npy, shape (n_cycles, 200, 3) - exactly the encoder's own
    input) from earlier preprocessing - so the encoder can still run
    live on this, genuinely computing fusion embeddings, without any
    raw V/I/T curve. Row order verified (not assumed) to align 1:1 with
    hi_table.parquet's own CALCE rows sorted by cycle_idx (same
    original preprocessing pass built both, in the same cycle order)."""
    path = PROC_DIR / "differential_tensors" / f"CALCE_{battery_id}.npy"
    if not path.exists():
        return None
    tensor = np.load(path)  # (n_cycles, 200, 3)
    with torch.no_grad():
        return encoder.encode(torch.tensor(tensor, dtype=torch.float32)).numpy()


def available_precomputed_cycles(dataset: str) -> dict[str, list[int]]:
    """{battery_id: sorted [cycle_idx, ...]} of cycles this dataset can
    serve WITHOUT any raw cycle data (data/raw/) - used to populate
    battery/cycle selection whenever raw data isn't available locally
    (e.g. this project's own Streamlit Cloud deploy, which never has
    data/raw/ - see README's own documented reason). Covers every
    dataset this dashboard supports, not just NASA/MIT."""
    if dataset in ("NASA", "MIT"):
        hi_df = pd.read_parquet(PROC_DIR / "hi_table.parquet")
        fusion_df = pd.read_csv(PROC_DIR / "fusion_embeddings.csv")
        have_fusion = set(zip(fusion_df["battery_id"], fusion_df["cycle_idx"]))
        sub = hi_df[hi_df["dataset"] == dataset]
        out: dict[str, list[int]] = {}
        for bid, cyc in zip(sub["battery_id"], sub["cycle_idx"]):
            if (bid, int(cyc)) in have_fusion:
                out.setdefault(bid, []).append(int(cyc))
        return {k: sorted(v) for k, v in out.items()}
    if dataset == "CALCE":
        hi_df = pd.read_parquet(PROC_DIR / "hi_table.parquet")
        sub = hi_df[hi_df["dataset"] == "CALCE"]
        out = {}
        for bid, cyc in zip(sub["battery_id"], sub["cycle_idx"]):
            tensor_path = PROC_DIR / "differential_tensors" / f"CALCE_{bid}.npy"
            if tensor_path.exists():
                out.setdefault(bid, []).append(int(cyc))
        return {k: sorted(v) for k, v in out.items()}
    if dataset in PRECOMPUTED_HELDOUT_PARQUETS:
        df = pd.read_parquet(PROC_DIR / PRECOMPUTED_HELDOUT_PARQUETS[dataset])
        out = {}
        for bid, cyc in zip(df["battery_id"], df["cycle_idx"]):
            out.setdefault(bid, []).append(int(cyc))
        return {k: sorted(v) for k, v in out.items()}
    return {}


def load_precomputed_battery_series(dataset: str, battery_id: str, res: dict) -> list[dict]:
    """Full per-cycle series for ONE battery, sorted by cycle_idx, built
    from the SAME precomputed artifacts predict_and_explain_precomputed
    uses for a single cycle - {"cycle_idx", "his", "fusion_emb",
    "true_soh"} per entry. This is what makes a Streaming Digital Twin
    replay possible without raw cycle data (Priority 2, "Start
    streaming simulation" was previously raw-data-only with no
    fallback - see PrecomputedStreamingTwin below, which consumes this
    series cycle by cycle through the SAME online-correction/ACI/drift
    machinery the raw-data twin uses, unchanged)."""
    out = []
    if dataset in ("NASA", "MIT"):
        hi_df = pd.read_parquet(PROC_DIR / "hi_table.parquet")
        fusion_df = pd.read_csv(PROC_DIR / "fusion_embeddings.csv")
        sub = hi_df[(hi_df["dataset"] == dataset) & (hi_df["battery_id"] == battery_id)].sort_values("cycle_idx")
        femb = fusion_df[fusion_df["battery_id"] == battery_id].set_index("cycle_idx")
        for _, row in sub.iterrows():
            cyc = int(row["cycle_idx"])
            if cyc not in femb.index:
                continue
            frow = femb.loc[cyc]
            out.append({"cycle_idx": cyc, "his": row.to_dict(), "true_soh": float(row["SOH"]),
                        "fusion_emb": frow[[f"fusion_{i}" for i in range(16)]].to_numpy(dtype=float)})
    elif dataset == "CALCE":
        hi_df = pd.read_parquet(PROC_DIR / "hi_table.parquet")
        sub = hi_df[(hi_df["dataset"] == "CALCE") & (hi_df["battery_id"] == battery_id)].sort_values("cycle_idx").reset_index(drop=True)
        all_emb = _calce_fusion_embeddings(battery_id, res["encoder"])
        if all_emb is None:
            return []
        for pos, row in sub.iterrows():
            if pos >= len(all_emb):
                break
            out.append({"cycle_idx": int(row["cycle_idx"]), "his": row.to_dict(),
                        "true_soh": float(row["SOH"]), "fusion_emb": all_emb[pos]})
    elif dataset in PRECOMPUTED_HELDOUT_PARQUETS:
        df = pd.read_parquet(PROC_DIR / PRECOMPUTED_HELDOUT_PARQUETS[dataset])
        sub = df[df["battery_id"] == battery_id].sort_values("cycle_idx")
        for _, row in sub.iterrows():
            out.append({"cycle_idx": int(row["cycle_idx"]), "his": row.to_dict(),
                        "true_soh": float(row["SOH"]),
                        "fusion_emb": row[[f"fusion_{i}" for i in range(16)]].to_numpy(dtype=float)})
    return out


class PrecomputedStreamingTwin:
    """Same online-correction/ACI/drift-detection behavior as
    StreamingDigitalTwinRiver, but consuming a precomputed per-cycle
    series (see load_precomputed_battery_series) instead of raw cycle
    curves - built by composition (wraps a real StreamingDigitalTwinRiver
    and overrides only its raw-prediction step), not by duplicating the
    online-learning logic, so a fix to the real corrector automatically
    applies here too.

    "cycle" arguments to .step() here are plain {"cycle_idx": N} dicts,
    not real cycle records - the wrapped twin's own _raw_predict is
    replaced (not called) precisely because that is the ONE method that
    needs a raw curve; every other piece of StreamingDigitalTwinRiver
    (the online corrector, ACI, OC-SVM check, drift detector) only ever
    consumes the (raw_pred, feat) pair _raw_predict returns, so
    substituting that pair from precomputed data is a correct, narrow
    substitution, not a reimplementation."""

    def __init__(self, series: list[dict], res: dict, train_medians: dict):
        from digital_twin_streaming_river import StreamingDigitalTwinRiver
        self._twin = StreamingDigitalTwinRiver(
            res["xgb_fusion"], res["encoder"], res["ocsvm"], res["ocsvm_scaler"],
            res["ocsvm_feature_cols"], res["bfa_selected"], train_medians, res["norm_stats"],
            res["constants"]["soh_conformal_half_width"],
        )
        self._by_cycle = {e["cycle_idx"]: e for e in series}
        self.train_medians = train_medians
        baseline_entry = self._by_cycle.get(BASELINE_CYCLE)
        self._baseline_his = baseline_entry["his"] if baseline_entry else (series[0]["his"] if series else None)

        def _precomputed_raw_predict(cycle: dict):
            entry = self._by_cycle.get(cycle["cycle_idx"])
            if entry is None:
                return None, None
            hi_rel_vec = build_reformulated_hi_vector(entry["his"], self.train_medians, self._baseline_his)
            hi_vec = np.concatenate([hi_rel_vec, [float(cycle["cycle_idx"])]])
            feat = np.concatenate([hi_vec, entry["fusion_emb"]])
            raw_pred = float(res["xgb_fusion"].predict(feat.reshape(1, -1))[0])
            return raw_pred, feat

        self._twin._raw_predict = _precomputed_raw_predict

    def step(self, cycle: dict, true_soh: float | None = None) -> dict:
        return self._twin.step(cycle, true_soh=true_soh)

    @property
    def drift_events(self):
        return self._twin.drift_events


def predict_and_explain_precomputed(dataset: str, battery_id: str, cycle_idx: int, res: dict) -> dict:
    """Same output CONTRACT as predict_and_explain (identical dict keys,
    so every existing render_*_tab function works unchanged regardless
    of which path built ctx) - but sourced entirely from this project's
    own precomputed, git-tracked artifacts instead of a raw per-cycle
    V/I/T curve: hi_table.parquet + fusion_embeddings.csv (NASA/MIT),
    hi_table.parquet + a cached ICA/DV/DC tensor (CALCE), or the
    Stage 5.1 merged parquets that already carry HI + fusion columns
    together (Oxford/HUST/XJTU). Runs XGBoost-fusion + TreeSHAP + the
    OC-SVM check GENUINELY LIVE on this reconstructed 25-dim feature
    vector - this is a real prediction, not a cached lookup of a saved
    number, for every cycle these artifacts cover (the full 42-battery
    pool, all of CALCE, and all of Oxford/HUST/XJTU).

    RUL and the VLSTM voltage-region SHAP explanation are honestly
    reported unavailable (not faked, not silently dropped) - both
    genuinely need the raw per-cycle V/I/T curve, which isn't cached
    anywhere for any dataset (only the derived ICA/DV/DC tensor is,
    for CALCE)."""
    train_medians = res["train_medians"]
    unavailable_msg = ("this cycle's raw voltage/current curve isn't available in this "
                        "deployment (see the note at the top of this page) - only "
                        "precomputed Health-Indicator and fusion-embedding data is, which "
                        "is enough for a real SOH prediction and SHAP explanation but not "
                        "for RUL or the voltage-region explanation, which both need the "
                        "raw curve directly")

    if dataset in ("NASA", "MIT"):
        hi_df = pd.read_parquet(PROC_DIR / "hi_table.parquet")
        row = hi_df[(hi_df["dataset"] == dataset) & (hi_df["battery_id"] == battery_id)
                     & (hi_df["cycle_idx"] == cycle_idx)]
        if row.empty:
            return {"error": f"No precomputed data for {dataset}/{battery_id} cycle {cycle_idx}."}
        his = row.iloc[0].to_dict()
        true_soh = float(his.get("SOH", float("nan")))

        base_row = hi_df[(hi_df["dataset"] == dataset) & (hi_df["battery_id"] == battery_id)
                          & (hi_df["cycle_idx"] == BASELINE_CYCLE)]
        baseline_his = base_row.iloc[0].to_dict() if not base_row.empty else None

        fusion_df = pd.read_csv(PROC_DIR / "fusion_embeddings.csv")
        frow = fusion_df[(fusion_df["battery_id"] == battery_id) & (fusion_df["cycle_idx"] == cycle_idx)]
        if frow.empty:
            return {"error": f"No precomputed fusion embedding for {dataset}/{battery_id} cycle {cycle_idx}."}
        fusion_emb = frow.iloc[0][[f"fusion_{i}" for i in range(16)]].to_numpy(dtype=float)
        no_temperature = dataset == "CALCE"  # never true here, kept for symmetry

    elif dataset == "CALCE":
        hi_df = pd.read_parquet(PROC_DIR / "hi_table.parquet")
        sub = hi_df[(hi_df["dataset"] == "CALCE") & (hi_df["battery_id"] == battery_id)].sort_values("cycle_idx").reset_index(drop=True)
        match = sub[sub["cycle_idx"] == cycle_idx]
        if match.empty:
            return {"error": f"No precomputed data for CALCE/{battery_id} cycle {cycle_idx}."}
        pos = match.index[0]
        his = match.iloc[0].to_dict()
        true_soh = float(his.get("SOH", float("nan")))
        baseline_match = sub[sub["cycle_idx"] == BASELINE_CYCLE]
        baseline_his = baseline_match.iloc[0].to_dict() if not baseline_match.empty else None

        all_emb = _calce_fusion_embeddings(battery_id, res["encoder"])
        if all_emb is None or pos >= len(all_emb):
            return {"error": f"No cached ICA tensor available for CALCE/{battery_id}."}
        fusion_emb = all_emb[pos]
        no_temperature = True  # CALCE has no temperature channel - documented throughout this project

    elif dataset in PRECOMPUTED_HELDOUT_PARQUETS:
        df = pd.read_parquet(PROC_DIR / PRECOMPUTED_HELDOUT_PARQUETS[dataset])
        row = df[(df["battery_id"] == battery_id) & (df["cycle_idx"] == cycle_idx)]
        if row.empty:
            return {"error": f"No precomputed data for {dataset}/{battery_id} cycle {cycle_idx}."}
        r = row.iloc[0]
        his = r.to_dict()
        true_soh = float(his.get("SOH", float("nan")))
        base_row = df[(df["battery_id"] == battery_id) & (df["cycle_idx"] == BASELINE_CYCLE)]
        baseline_his = base_row.iloc[0].to_dict() if not base_row.empty else None
        fusion_emb = r[[f"fusion_{i}" for i in range(16)]].to_numpy(dtype=float)
        no_temperature = dataset != "HUST"  # HUST has a temperature channel; Oxford/XJTU's raw pipeline does not carry one into this table

    else:
        return {"error": f"Precomputed fallback not implemented for dataset {dataset}."}

    hi_rel_vector = build_reformulated_hi_vector(his, train_medians, baseline_his)
    hi_vector = np.concatenate([hi_rel_vector, [float(cycle_idx)]])
    feat_vector = np.concatenate([hi_vector, fusion_emb])

    pred_soh = float(res["xgb_fusion"].predict(feat_vector.reshape(1, -1))[0])
    soh_half = res["constants"]["soh_conformal_half_width"]

    ocsvm_feat = feat_vector.reshape(1, -1)
    ocsvm_scaled = res["ocsvm_scaler"].transform(ocsvm_feat)
    anomaly_flag = bool(res["ocsvm"].predict(ocsvm_scaled)[0] == -1)

    out_of_domain = no_temperature or anomaly_flag or dataset not in ("NASA", "MIT")
    domain_reasons = []
    if dataset not in ("NASA", "MIT"):
        domain_reasons.append(f"{dataset} was never part of this model's NASA/MIT training data - "
                               f"this is a genuine zero-retrain (out-of-domain) evaluation")
    if no_temperature:
        domain_reasons.append("no temperature channel present in this dataset")
    if anomaly_flag:
        domain_reasons.append("the One-Class SVM flags this cycle's feature vector as unlike "
                               "anything in the NASA+MIT training data")

    top_features = _tree_shap_top_features(feat_vector, res["xgb_fusion"],
                                            FEATURE_COLS + [f"fusion_{i}" for i in range(16)])

    return {
        "soh_pred": round(pred_soh, 1),
        "soh_conformal_lo": round(pred_soh - soh_half, 1),
        "soh_conformal_hi": round(pred_soh + soh_half, 1),
        "rul_pred": None,
        "rul_conformal_lo": None,
        "rul_conformal_hi": None,
        "rul_unavailable_reason": unavailable_msg,
        "top_features": top_features,
        "voltage_region": None,
        "voltage_region_error": unavailable_msg,
        "anomaly_flag": anomaly_flag,
        "out_of_domain": out_of_domain,
        "domain_reasons": domain_reasons,
        "cycle_idx": int(cycle_idx),
        "true_soh": round(true_soh, 1) if true_soh == true_soh else None,
        "true_rul": None,
        "precomputed_fallback": True,
    }


def predict_and_explain(cycle: dict, res: dict, baseline_his: dict | None = None) -> dict:
    """cycle: single cycle record (data_adapters convention). baseline_his:
    optional - this battery's own cycle-10 raw HI dict (see
    build_reformulated_hi_vector). Returns a full context dict for
    display, or {"error": ...} if the cycle is too short/malformed for
    the sequence models."""
    train_medians = res["train_medians"]

    # 1. Health Indicators (always computable, even for very short cycles)
    his = compute_health_indicators(cycle)
    hi_rel_vector = build_reformulated_hi_vector(his, train_medians, baseline_his)  # 8-dim, no cycle_idx
    hi_vector = np.concatenate([hi_rel_vector, [float(cycle["cycle_idx"])]])  # 9-dim, XGBoost's own feature set (1.5's cycle_idx)
    no_temperature = cycle["discharge"]["T"] is None

    # RUL joint-fusion model's own input: the SAME 8 _rel-ratio features
    # (no cycle_idx - it was never part of that model's feature set),
    # z-scored with ITS OWN fit-split mean/std (a different normalization
    # than XGBoost's, which uses raw values - see load_resources).
    jn = res["joint_hi_norm"]
    hi_rel_z = (hi_rel_vector - np.array(jn["hi_mean"])) / np.array(jn["hi_std"])

    # 2. sequence tensor (needed for VLSTM's SHAP explanation, the
    # fusion embedding, and RUL's joint model)
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
        fusion_emb = res["encoder"].encode(torch.tensor(x_norm[None, :, 3:6])).numpy()[0]
        _, pred_rul_z = res["joint_fusion"](torch.tensor(x_norm[None]), torch.tensor(hi_rel_z[None], dtype=torch.float32))
        pred_rul = destd(pred_rul_z, rul_mean, rul_std)

    feat_vector = np.concatenate([hi_vector, fusion_emb])
    pred_soh = float(res["xgb_fusion"].predict(feat_vector.reshape(1, -1))[0])

    soh_half = res["constants"]["soh_conformal_half_width"]
    rul_half = res["constants"]["rul_conformal_half_width"]

    # OC-SVM anomaly check (same feature vector, in the order saved by train_ocsvm.py)
    ocsvm_feat = feat_vector.reshape(1, -1)
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
                                            FEATURE_COLS + [f"fusion_{i}" for i in range(16)])
    voltage_region, voltage_region_error = _vlstm_voltage_region(
        x_raw, x_norm, res["vlstm"], res["background"]
    )

    return {
        "soh_pred": round(pred_soh, 1),
        "soh_conformal_lo": round(pred_soh - soh_half, 1),
        "soh_conformal_hi": round(pred_soh + soh_half, 1),
        # clip to >=0 for display - the raw regression output can go
        # slightly negative for deeply past-EOL cycles, which is a
        # faithful regression residual but a meaningless thing to show a
        # dashboard user ("-15 cycles remaining" isn't interpretable).
        "rul_pred": max(0, round(pred_rul)),
        "rul_conformal_lo": max(0, round(pred_rul - rul_half)),
        "rul_conformal_hi": round(pred_rul + rul_half),
        "rul_unavailable_reason": None,
        "top_features": top_features,
        "voltage_region": voltage_region,
        "voltage_region_error": voltage_region_error,
        "anomaly_flag": anomaly_flag,
        "out_of_domain": out_of_domain,
        "domain_reasons": domain_reasons,
        "cycle_idx": cycle["cycle_idx"],
        "true_soh": None,  # filled in by the caller if ground truth is available
        "true_rul": None,
        "precomputed_fallback": False,
    }
