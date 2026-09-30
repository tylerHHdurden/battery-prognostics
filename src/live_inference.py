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
import encoder_provenance as ep
from stage1_common import canonical_feature_cols, fusion_cols as _fusion_cols, DURATION_FEATURES, BASELINE_CYCLE, ICA_CHANNEL_SLICE
from stage5_extended_reformulation import extended_canonical_feature_cols
from run_stage1_followup_partB_joint_rul import JointSOHRULModelFusion

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

CANONICAL_RAW = canonical_feature_cols(reformulated=False)   # e.g. ICHV, SCV, ...
CANONICAL_REL = canonical_feature_cols(reformulated=True)    # e.g. ICHV_rel, SCV, ...
FEATURE_COLS = CANONICAL_REL + ["cycle_idx"]

# Dataset-aware routing (added after this session's own baseline audit
# found the deployed model's zero-retrain numbers were much worse than
# assumed on Oxford/HUST/CALCE - see DEVELOPMENT_LOG.md). Stage 5's
# extended reformulation (SCV/MATD/VIECT/MET swapped for _rel versions)
# genuinely beats the base model on these 3 datasets but REGRESSES
# XJTU - twice-investigated, twice-declined for a BLANKET promotion
# (also in DEVELOPMENT_LOG.md). Routing sidesteps that all-or-nothing
# choice: each dataset gets whichever already-validated model actually
# wins on IT, decided by dataset identity alone, not a new trained
# model. XJTU (and NASA/MIT, never part of this routing's scope) stay
# on the original Stage 4 model, completely unchanged.
EXTENDED_ROUTED_DATASETS = {"CALCE", "Oxford", "HUST"}
EXTENDED_CANONICAL_REL = extended_canonical_feature_cols(CANONICAL_REL)  # ICHV_rel,SCV_rel,VDEDT,VIECT_rel,MATD_rel,MET_rel,TEVD_rel,TEVI_rel
EXTENDED_FEATURE_COLS = EXTENDED_CANONICAL_REL + ["cycle_idx"]
EXT_RATIO_RAW_FEATURES = {"ICHV", "TEVD", "TEVI", "SCV", "MET"}   # ratio-reformulated (Stage 1.1's own 3 + Stage 5's SCV/MET)
EXT_DELTA_RAW_FEATURES = {"MATD", "VIECT"}                        # delta-reformulated (Stage 5's own choice - additive-offset quantities)

# ---------------------------------------------------------------------
# MULTI-SOURCE CANDIDATE MODEL FLAG (toolkit pass, Phase 2 -> decision).
# ---------------------------------------------------------------------
# Trained on ALL 16 sources (NASA/MIT/CALCE/Oxford/HUST/XJTU + 9
# BatteryLife sources + Tongji) - its own ICA encoder, its own XGBoost-
# fusion. Gate evaluation (battery-level split WITHIN every source)
# found it beats the current deployed routed model on 13/16 sources,
# often by a wide margin; NASA and MIT are the exception - it loses to
# the current deployed model on both (a real, diagnosed "small source
# crowded out by pooling" effect, not fixed here - see DEVELOPMENT_LOG.
# md's "Phase 2 decision" entry). A source-balanced retrain (equal
# per-source sample weight) was tried specifically to close that NASA/
# MIT gap and instead collapsed catastrophically on 15/16 sources - not
# adopted.
#
# ROUTING (revised 2026-10-01 so the live app matches the submitted report and PAPER_RESULTS.md):
#   * The SIX APP DATASETS (NASA, MIT, CALCE, Oxford, HUST, XJTU) use the VALIDATED routing: extended reformulation for
#     CALCE/Oxford/HUST, the base XGBoost-fusion model for XJTU, NASA and MIT. Validation = PAPER_RESULTS.md sec 1 (five-seed
#     zero-retrain numbers: CALCE R2 0.749, Oxford 0.940, HUST 0.795, XJTU -1.037) and outputs/finalpass_item5a_5seed_aggregate.csv;
#     that routing was itself selected on held-out data, so it is a deployment choice, not an unbiased result.
#   * The multi-source candidate is used only where it was validated: batteries from the nine BatteryLife sources + Tongji and any
#     "Uploaded"/unknown battery. What validated it: the Phase 2 gate table (outputs/toolkit_phase2_gate_table.csv), a battery-level split
#     WITHIN each source in which the candidate beat the routed baseline on all ten of those sources. Limits, stated plainly: those
#     sources' training batteries were in the candidate's training pool, so this is not a zero-retrain test, and the leave-one-source-out
#     rerun (outputs/toolkit_lodo_family_holdout_rerun_corrected.csv) shows the candidate's transfer to a truly unseen source is much
#     weaker - which is exactly what the distribution-shift message in the app says.
#   * Why not the candidate for the six app datasets: the report and PAPER_RESULTS.md quote the routed numbers, and the candidate was
#     never compared with them under the same zero-retrain protocol. (Gate-table wins for the candidate on CALCE/HUST/XJTU exist but
#     are same-source battery-level splits.)
#
# Toggle this to False to revert to the pre-candidate routing for everything (Stage-5 extended reformulation for CALCE/Oxford/HUST,
# base model otherwise) with ZERO other code changes required.
USE_MULTISOURCE_CANDIDATE = True
SIX_APP_DATASETS = ("NASA", "MIT", "CALCE", "Oxford", "HUST", "XJTU")  # validated routing, never the candidate (see the comment above)
CANDIDATE_FEATURE_COLS = CANONICAL_REL + ["cycle_idx"]  # SAME 8 HI + cycle_idx as the base model - only the fusion embedding and the XGBoost model object differ


def _use_candidate(dataset: str | None) -> bool:
    """True iff the multi-source candidate should be used for this
    dataset, per the routing decision above. `dataset=None` (no
    selection made yet) and `dataset="Uploaded"` both correctly route
    to the candidate (every "uploaded/unknown battery" case) since
    neither equals "NASA" or "MIT"."""
    return USE_MULTISOURCE_CANDIDATE and dataset not in SIX_APP_DATASETS

HI_DESCRIPTIONS = {
    "ICHV_rel": "time spent charging near peak voltage (high-voltage/CV-tail duration), relative to this battery's own cycle-10 baseline",
    "SCV": "average capacity-vs-voltage slope during discharge",
    "SCV_rel": "average capacity-vs-voltage slope during discharge, relative to this battery's own cycle-10 baseline",
    "VDEDT": "rate of voltage collapse near the end of discharge",
    "VIECT": "voltage reached at a fixed elapsed time into charging",
    "VIECT_rel": "voltage reached at a fixed elapsed time into charging, relative to this battery's own cycle-10 baseline (a delta, not a ratio - voltage level is an additive offset, not a multiplicative scale)",
    "MATD": "mean cell temperature during discharging",
    "MATD_rel": "mean cell temperature during discharging, relative to this battery's own cycle-10 baseline (a delta, not a ratio - temperature is an additive offset, not a multiplicative scale)",
    "MET": "mean energy during test (average of charge and discharge energy)",
    "MET_rel": "mean energy during test, relative to this battery's own cycle-10 baseline",
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

    # Dataset-aware routing's second model (CALCE/Oxford/HUST only - see
    # EXTENDED_ROUTED_DATASETS above). ext_medians is precomputed and
    # saved (precompute_ext_reformulation_medians.py) - the TRAIN-pool,
    # post-reformulation column medians fit_xgb used when this model was
    # trained; recomputing it live on every app start would mean
    # reloading the full NASA+MIT pool (a multi-second operation) for a
    # constant that never changes unless the model itself is retrained.
    xgb_fusion_extended = XGBRegressor()
    xgb_fusion_extended.load_model(str(MODELS_DIR / "_experimental_xgb_soh_fusion_extended_reformulation.json"))
    ext_medians = json.loads((PROC_DIR / "ext_reformulation_medians.json").read_text())

    with open(MODELS_DIR / "ocsvm_model.pkl", "rb") as f:
        ocsvm = pickle.load(f)
    with open(MODELS_DIR / "ocsvm_scaler.pkl", "rb") as f:
        ocsvm_scaler = pickle.load(f)
    ocsvm_feature_cols = json.loads((PROC_DIR / "ocsvm_feature_cols.json").read_text())

    # --- multi-source candidate (see USE_MULTISOURCE_CANDIDATE above) ---
    xgb_candidate = candidate_encoder = candidate_norm_stats = candidate_medians = candidate_fusion_lookup = None
    candidate_fusion_range = None
    if USE_MULTISOURCE_CANDIDATE:
        xgb_candidate = XGBRegressor()
        xgb_candidate.load_model(str(MODELS_DIR / "_candidate_multisource.json"))
        candidate_encoder = ICAEncoder(in_channels=3, embed_dim=16)
        candidate_encoder.load_state_dict(torch.load(MODELS_DIR / "_candidate_ica_encoder.pt"))
        candidate_encoder.eval()
        candidate_norm_stats = json.loads((PROC_DIR / "candidate_channel_norm_stats.json").read_text())
        candidate_fusion_range = json.loads((PROC_DIR / "candidate_fusion_train_range.json").read_text())
        _cand_med = json.loads((PROC_DIR / "candidate_multisource_medians.json").read_text())
        candidate_medians = dict(zip(_cand_med["cols"], _cand_med["medians"]))
        # precomputed fusion embeddings for every (dataset, battery_id, cycle_idx)
        # this candidate encoder has already scored (Phase 2's own build) - used
        # by the PRECOMPUTED-fallback path so it never needs to re-run the
        # encoder live; the LIVE raw-cycle path (predict_and_explain) always
        # re-encodes on the fly instead, using candidate_encoder/candidate_norm_stats
        # directly, since an uploaded/live cycle has no precomputed row to look up.
        _cand_fusion_df = pd.read_csv(PROC_DIR / "fusion_embeddings_multisource.csv")
        _cand_fusion_vals = _cand_fusion_df[[f"fusion_{i}" for i in range(16)]].to_numpy(dtype=float)
        candidate_fusion_lookup = {
            (ds, bid, int(cyc)): _cand_fusion_vals[i]
            for i, (ds, bid, cyc) in enumerate(zip(_cand_fusion_df["dataset"], _cand_fusion_df["battery_id"], _cand_fusion_df["cycle_idx"]))
        }

    # Phase 3 OC-SVM correction: the candidate OC-SVM (fit on the same 16-source pool the
    # multisource candidate was trained on) is a genuinely discriminating INPUT-SANITY check
    # (100% detection of swapped V/I columns, capacity-x10, truncated cycles, 0.6% in-domain
    # false-flag rate - see DEVELOPMENT_LOG.md's "OC-SVM sanity check" entries) - unlike the
    # DEPLOYED ocsvm above, which was never used for this and stays loaded/unchanged for its
    # existing role. Used specifically for the upload "does this data look malformed" check,
    # not for domain/familiarity (that's the nearest-source trust report's job).
    with open(MODELS_DIR / "_candidate_ocsvm.pkl", "rb") as f:
        candidate_ocsvm = pickle.load(f)
    with open(MODELS_DIR / "_candidate_ocsvm_scaler.pkl", "rb") as f:
        candidate_ocsvm_scaler = pickle.load(f)

    # Encoder-provenance guard (2026-09-30): fail LOUDLY if any model is paired with the other encoder's embeddings.
    # Old models <-> old encoder (+ its norm stats); candidate models <-> candidate encoder + the candidate CSV lookup.
    for _m in ["xgb_soh_fusion.json", "_experimental_xgb_soh_fusion_extended_reformulation.json",
               "ocsvm_model.pkl", "ocsvm_scaler.pkl"]:
        ep.assert_encoder_match(MODELS_DIR / _m, MODELS_DIR / "ica_encoder.pt", context=f"load_resources[{_m}]")
    if USE_MULTISOURCE_CANDIDATE:
        for _m in ["_candidate_multisource.json", "_candidate_ocsvm.pkl", "_candidate_ocsvm_scaler.pkl"]:
            ep.assert_encoder_match(MODELS_DIR / _m, MODELS_DIR / "_candidate_ica_encoder.pt", context=f"load_resources[{_m}]")
        ep.assert_encoder_match(MODELS_DIR / "_candidate_multisource.json", PROC_DIR / "fusion_embeddings_multisource.csv",
                                context="load_resources[candidate fusion lookup]")
        ep.assert_encoder_match(MODELS_DIR / "_candidate_multisource.json", PROC_DIR / "candidate_fusion_train_range.json",
                                context="load_resources[candidate fusion range]")

    return {
        "norm_stats": norm_stats, "constants": constants, "joint_hi_norm": joint_hi_norm,
        "background": background, "train_medians": train_medians,
        "vlstm": vlstm, "encoder": encoder, "joint_fusion": joint_fusion,
        "xgb_fusion": xgb_fusion,
        "xgb_fusion_extended": xgb_fusion_extended, "ext_medians": ext_medians,
        "ocsvm": ocsvm, "ocsvm_scaler": ocsvm_scaler, "ocsvm_feature_cols": ocsvm_feature_cols,
        "xgb_candidate": xgb_candidate, "candidate_encoder": candidate_encoder,
        "candidate_norm_stats": candidate_norm_stats, "candidate_medians": candidate_medians,
        "candidate_fusion_lookup": candidate_fusion_lookup, "candidate_fusion_range": candidate_fusion_range,
        "candidate_ocsvm": candidate_ocsvm, "candidate_ocsvm_scaler": candidate_ocsvm_scaler,
        # kept for StreamingDigitalTwin's constructor API (digital_twin_streaming.py) -
        # the canonical reformulated names now, not the old pre-Stage-1 set; no longer
        # used for feature-BUILDING directly (build_reformulated_hi_vector handles that)
        "bfa_selected": CANONICAL_REL,
    }


RUL_TRAINING_SOURCES = {"NASA", "MIT"}  # the only sources the joint RUL model was ever trained on
_BUILTIN_TRUST = None


def lookup_builtin_trust(dataset: str | None, battery_id: str | None) -> dict | None:
    """Precomputed nearest-source trust verdict (src/build_builtin_battery_trust_table.py ->
    models/_builtin_battery_trust.csv) for a BUILT-IN battery, or None if not in the table."""
    global _BUILTIN_TRUST
    if _BUILTIN_TRUST is None:
        path = MODELS_DIR / "_builtin_battery_trust.csv"
        _BUILTIN_TRUST = {}
        if path.exists():
            df = pd.read_csv(path)
            for r in df.to_dict("records"):
                _BUILTIN_TRUST[(str(r["dataset"]), str(r["battery_id"]))] = r
    if dataset is None or battery_id is None:
        return None
    return _BUILTIN_TRUST.get((str(dataset), str(battery_id)))


# ---- Out-of-domain operating point (approved 2026-09-30) -------------------------------------------------------------
# A battery is FLAGGED when its nearest-source Gaussian negative log-likelihood score `nll_min` (trust_report's
# `nll_min`, computed on the battery's median 25-dim feature vector) is >= OOD_NLL_THRESHOLD.
# How the number was chosen and validated (src/trust_operating_point.py, src/validate_trust_threshold_leave_source_out.py):
#   NOVEL = every battery of a source scored against the profiles of all OTHER sources (its whole sibling family removed);
#   KNOWN = each source's held-out test batteries scored against all 16 profiles. The threshold is the one with the lowest
#   known-source false-alarm rate among those flagging >= 80% of novel batteries (ROC AUC of nll_min: 0.896).
#   Leave-source-out check (threshold chosen WITHOUT the source it is applied to), pooled over 16 sources:
#   390/476 = 81.9% of novel-source batteries flagged, 10/91 = 11.0% of known-source batteries falsely flagged.
#   Per-source table: outputs/toolkit_phase3_trust_threshold_leave_source_out.csv. The old rule (anything not "familiar")
#   flagged 47.5% (226/476). A hybrid rule (OR with the trust level / ratio score / logistic regression) was tested and
#   rejected: it raised detection only by raising false alarms (outputs/toolkit_phase3_hybrid_rule_summary.txt).
OOD_SCORE = "nll_min"
OOD_NLL_THRESHOLD = -5.521444398006403
OOD_THRESHOLD_DATE = "2026-09-30"
OOD_VALIDATION_TABLE = "outputs/toolkit_phase3_trust_threshold_leave_source_out.csv"
OOD_NOVEL_DETECTED = 0.819          # 390/476, leave-source-out
OOD_KNOWN_FALSE_ALARM = 0.110       # 10/91, leave-source-out
OOD_WEAK_SOURCES = ("mich", "NASA", "snl")  # fewer than half of these sources' novel batteries were flagged in validation
OOD_NOT_DETECTED_LINE = ("About 1 in 5 unfamiliar batteries are not detected; check against a measured capacity when possible.")
_TYPICAL_UNSEEN_MAE = None


def typical_unseen_source_mae() -> float | None:
    """Median over sources of the corrected family-holdout MAE (SOH points): the typical error on a source the model
    has never seen. From outputs/toolkit_lodo_family_holdout_rerun_corrected.csv (rerun on corrected embeddings)."""
    global _TYPICAL_UNSEEN_MAE
    if _TYPICAL_UNSEEN_MAE is None:
        p = ROOT / "outputs" / "toolkit_lodo_family_holdout_rerun_corrected.csv"
        _TYPICAL_UNSEEN_MAE = float(pd.read_csv(p)["family_lodo_mae_corrected"].median()) if p.exists() else float("nan")
    return _TYPICAL_UNSEEN_MAE if _TYPICAL_UNSEEN_MAE == _TYPICAL_UNSEEN_MAE else None


def domain_verdict(trust: dict | None) -> dict:
    """out_of_domain (= FLAGGED) is decided ONLY by the ROC rule above: nll_min >= OOD_NLL_THRESHOLD. The deployed OC-SVM
    is still computed and returned as `anomaly_flag` for transparency but decides nothing here (it stays an input-sanity
    check). Not flagged is NEVER presented as "trusted": the message says only that no distribution shift was detected,
    plus the measured miss rate. RUL is shown only when the nearest source is NASA or MIT (the only sources the RUL model was
    trained on; RUL R2=-566 on CALCE) AND the battery is not flagged."""
    if trust is None:
        return {"out_of_domain": True,
                "domain_reasons": ["no nearest-source familiarity check was available for this battery, so it cannot be judged"],
                "rul_hidden": True, "rul_hidden_reason": "no familiarity check was available for this battery",
                "flagged": True, "errors": None}
    nearest = trust["nearest_source"]
    score = trust.get("nll_min")
    flagged = True if score is None or score != score else bool(score >= OOD_NLL_THRESHOLD)
    gate = trust.get("gate_table_mae")
    unseen = typical_unseen_source_mae()
    errors = {"in_domain_mae": gate, "unseen_source_typical_mae": unseen, "nearest_source": nearest,
              "nearest_unseen_mae": trust.get("lodo_family_mae")}
    reasons = []
    if flagged:
        sc = "unavailable" if score is None or score != score else f"{score:.1f}"
        reasons.append(f"its nearest-source score ({OOD_SCORE} = {sc}) is at or above the flag threshold "
                       f"({OOD_NLL_THRESHOLD:.1f}); nearest known source: {nearest}")
    rul_hidden = flagged or nearest not in RUL_TRAINING_SOURCES
    if flagged:
        rr = "; ".join(reasons)
    elif nearest not in RUL_TRAINING_SOURCES:
        rr = (f"its nearest known source is {nearest}, which the RUL model was never trained on (it only saw NASA+MIT)")
    else:
        rr = None
    return {"out_of_domain": flagged, "flagged": flagged, "domain_reasons": reasons, "rul_hidden": rul_hidden,
            "rul_hidden_reason": rr, "errors": errors}


FUSION_RANGE_TOLERANCE = 0.05  # fraction of each dim's training span allowed beyond [min, max]; measured on
# 72,729 held-out embeddings: 0 false alarms at 5% (exact-range alone false-alarms 0.078%, 5.35% on stanford)


def sanitize_raw_tensor_for_candidate(x_raw: np.ndarray) -> np.ndarray:
    """The candidate encoder was TRAINED on tensors with NaN/+-Inf replaced by 0.0 BEFORE normalization
    (run_toolkit_phase2_multisource_retrain.py). Skipping this at inference is a train/inference mismatch:
    np.clip passes NaN straight through (the whole embedding becomes NaN, then is silently replaced
    downstream by per-dim medians) and clips +Inf to the channel's `hi` instead of the 0.0 training saw.
    Found on 3 of 1,118 sampled live cycles (all stanford, non-finite dVdQ)."""
    return np.nan_to_num(x_raw, nan=0.0, posinf=0.0, neginf=0.0)


def fusion_range_check(fusion_emb: np.ndarray, res: dict) -> tuple[bool, str | None]:
    """Runtime guard: (ok, reason). False if any candidate fusion value is non-finite or outside the
    candidate's TRAINING range (per-dim min/max over Phase 2 train rows, +-FUSION_RANGE_TOLERANCE*span).
    The caller then shows "prediction unreliable for this cycle" instead of a number."""
    rng = res.get("candidate_fusion_range")
    f = np.asarray(fusion_emb, dtype=float)
    if not np.all(np.isfinite(f)):
        return False, "the encoder produced a non-finite value for this cycle"
    if rng is None:
        return True, None
    lo, hi, span = np.array(rng["min"]), np.array(rng["max"]), np.array(rng["span"])
    bad = (f < lo - FUSION_RANGE_TOLERANCE * span) | (f > hi + FUSION_RANGE_TOLERANCE * span)
    if bad.any():
        return False, (f"encoder output dimension(s) {np.where(bad)[0].tolist()} fall outside the range seen "
                       f"in training")
    return True, None


def _candidate_soh_prediction(hi_rel_vector: np.ndarray, cycle_idx: float,
                               fusion_emb_candidate: np.ndarray, res: dict):
    """Shared by both predict_and_explain_precomputed and predict_and_
    explain: builds the multi-source candidate's 25-dim feature vector
    (the SAME 8 HI + cycle_idx as the base model - CANDIDATE_FEATURE_
    COLS == FEATURE_COLS - only the fusion embedding and the XGBoost
    model object differ) and scores it. Returns (pred_soh, shap_vector,
    shap_model, shap_cols) in the same shape every other branch's SHAP
    call site already expects."""
    hi_vector = np.concatenate([hi_rel_vector, [float(cycle_idx)]])
    feat_vector = np.concatenate([hi_vector, fusion_emb_candidate]).astype(float)
    nan_mask = np.isnan(feat_vector)
    if nan_mask.any():
        cols = CANDIDATE_FEATURE_COLS + [f"fusion_{i}" for i in range(16)]
        for i in np.where(nan_mask)[0]:
            feat_vector[i] = res["candidate_medians"].get(cols[i], 0.0)
    pred_soh = float(res["xgb_candidate"].predict(feat_vector.reshape(1, -1))[0])
    return pred_soh, feat_vector, res["xgb_candidate"], CANDIDATE_FEATURE_COLS


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


def build_extended_reformulated_hi_vector(his: dict, baseline_his: dict | None,
                                           battery_medians: dict, ext_medians: dict) -> np.ndarray:
    """Extended analogue of build_reformulated_hi_vector, for the
    dataset-aware-routed model (CALCE/Oxford/HUST - see
    EXTENDED_ROUTED_DATASETS). RATIO for {ICHV,TEVD,TEVI,SCV,MET},
    DELTA for {MATD,VIECT} - Stage 5's own established convention
    (stage5_extended_reformulation.py), VDEDT passed through raw.

    Mechanically faithful to how this exact model was TRAINED
    (fit_xgb fills NaN with POST-reformulation, TRAIN-pool column
    medians - NOT raw-feature medians, a materially different
    statistic once a feature has been ratio/delta-transformed) -
    verified end-to-end against the batch-validated numbers
    (0.740/0.953/0.800 for CALCE/Oxford/HUST) before this function was
    ever wired into live routing (see DEVELOPMENT_LOG.md and
    src/verify_extended_live_feature_parity.py, whose own exact logic
    this ports unchanged, not reimplemented from scratch).

    battery_medians: {"SCV": ..., "MET": ...} - that SAME battery's own
    median across ALL its cycles, used ONLY as the near-zero-baseline
    ratio guard's fallback (mirrors stage5_extended_reformulation.
    add_scv_matd_viect_reformulated's own per-battery guard exactly) -
    in practice this guard essentially never fires (0 batteries needed
    it across the entire TRAIN+CALCE pool when ext_medians was computed
    - see that script's own log), but kept faithful to the validated
    logic rather than assumed safe to drop.

    ext_medians: TRAIN-pool, POST-reformulation column medians (see
    precompute_ext_reformulation_medians.py) - fills any value this
    function itself leaves as NaN (near-zero/missing baseline with no
    battery_medians fallback either, or a genuinely missing raw
    feature), matching fit_xgb's own training-time imputation exactly."""
    vec = np.full(len(EXTENDED_CANONICAL_REL), np.nan, dtype=float)
    for i, col in enumerate(EXTENDED_CANONICAL_REL):
        raw_feat = col[:-4] if col.endswith("_rel") else col
        raw_val = his.get(raw_feat)
        raw_val = float(raw_val) if raw_val is not None else float("nan")

        if raw_feat in ("ICHV", "TEVD", "TEVI"):
            # UNCHANGED existing convention (matches build_reformulated_hi_vector exactly)
            if raw_val != raw_val:
                raw_val = 0.0
            base_val = None
            if baseline_his is not None:
                base_val = baseline_his.get(raw_feat)
                if base_val is not None and (base_val != base_val or abs(base_val) < 1e-6):
                    base_val = None
            if base_val is None:
                base_val = raw_val if abs(raw_val) >= 1e-6 else 1.0
            vec[i] = raw_val / base_val if abs(base_val) >= 1e-6 else 1.0

        elif raw_feat in EXT_RATIO_RAW_FEATURES:  # SCV, MET
            base_val = baseline_his.get(raw_feat) if baseline_his is not None else None
            if base_val is None or base_val != base_val or abs(base_val) < 1e-6:
                base_val = battery_medians.get(raw_feat)
            if base_val is not None and np.isfinite(base_val) and abs(base_val) >= 1e-6 and raw_val == raw_val:
                vec[i] = raw_val / base_val
            # else leave NaN -> filled by ext_medians below

        elif raw_feat in EXT_DELTA_RAW_FEATURES:  # MATD, VIECT
            base_val = baseline_his.get(raw_feat) if baseline_his is not None else None
            if base_val is not None and base_val == base_val and raw_val == raw_val:
                vec[i] = raw_val - base_val
            # else leave NaN -> filled by ext_medians below

        else:  # VDEDT, raw passthrough
            vec[i] = raw_val

    nan_mask = np.isnan(vec)
    if nan_mask.any():
        for i in np.where(nan_mask)[0]:
            vec[i] = ext_medians[EXTENDED_CANONICAL_REL[i]]
    return vec


PRECOMPUTED_HELDOUT_PARQUETS = {
    "Oxford": "stage5_1_oxford_merged.parquet",
    "HUST": "stage5_1_hust_merged.parquet",
    "XJTU": "stage5_1_xjtu_merged.parquet",
}


def _calce_fusion_embeddings(battery_id: str, encoder, norm_stats: list[dict]) -> np.ndarray | None:
    """CALCE has no precomputed fusion_embeddings.csv row (unlike NASA/
    MIT/Oxford/HUST/XJTU), but DOES have its 3-channel ICA/DV/DC
    differential tensor cached (data/processed/differential_tensors/
    CALCE_{id}.npy, shape (n_cycles, 200, 3) - exactly the encoder's own
    input) from earlier preprocessing - so the encoder can still run
    live on this, genuinely computing fusion embeddings, without any
    raw V/I/T curve. Row order verified (not assumed) to align 1:1 with
    hi_table.parquet's own CALCE rows sorted by cycle_idx (same
    original preprocessing pass built both, in the same cycle order).

    REAL BUG, FOUND AND FIXED (not present when this function was first
    written; caught while verifying a separate dataset-routing feature,
    then fixed as its own standalone production fix - see
    DEVELOPMENT_LOG.md): this cached tensor stores RAW, unnormalized
    dQdV/dVdQ/dIdV values, but `ica_encoder.pt` was TRAINED on
    channel-normalized input (train_fusion_encoder.py's own
    `apply_channel_norm(X_all, norm_stats)` call, applied BEFORE slicing
    to these 3 channels - stage1_common.build_calce_tensors, the batch
    pipeline that produced every "CALCE" number this project has ever
    reported, does the exact same thing). Differential-tensor channels
    can blow up to O(1e5-1e6) near flat-capacity regions (documented
    elsewhere in this project as the same reason CNN-LSTM needed
    BatchNorm-input clipping) - feeding that raw into an encoder trained
    on clipped/z-scored input produced embeddings wildly outside what
    the encoder (and everything downstream of it) was ever trained to
    expect. `norm_stats` is the SAME 6-channel `channel_norm_stats.json`
    already loaded for every other tensor in this module - sliced to
    channels 3:6 (dQdV/dVdQ/dIdV) via `ICA_CHANNEL_SLICE`, which is
    mathematically identical to normalizing the full 6-channel tensor
    first and slicing after (apply_channel_norm operates per-channel,
    independently of what other channels are present) - matches
    build_calce_tensors' own convention exactly, not a new one."""
    path = PROC_DIR / "differential_tensors" / f"CALCE_{battery_id}.npy"
    if not path.exists():
        return None
    tensor = np.load(path)  # (n_cycles, 200, 3), RAW - channels 3-5 of the project's own 6-channel convention
    tensor = apply_channel_norm(tensor, norm_stats[ICA_CHANNEL_SLICE])
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
        all_emb = _calce_fusion_embeddings(battery_id, res["encoder"], res["norm_stats"])
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


def predict_and_explain_precomputed(dataset: str, battery_id: str, cycle_idx: int, res: dict,
                                     trust: dict | None = None) -> dict:
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

        ep.assert_encoder_match(MODELS_DIR / "xgb_soh_fusion.json", PROC_DIR / "fusion_embeddings.csv",
                                context="predict_and_explain_precomputed[NASA/MIT]")
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
        battery_medians = {"SCV": sub["SCV"].median(), "MET": sub["MET"].median()}

        all_emb = _calce_fusion_embeddings(battery_id, res["encoder"], res["norm_stats"])
        if all_emb is None or pos >= len(all_emb):
            return {"error": f"No cached ICA tensor available for CALCE/{battery_id}."}
        fusion_emb = all_emb[pos]
        no_temperature = True  # CALCE has no temperature channel - documented throughout this project

    elif dataset in PRECOMPUTED_HELDOUT_PARQUETS:
        ep.assert_encoder_match(MODELS_DIR / "xgb_soh_fusion.json", PROC_DIR / PRECOMPUTED_HELDOUT_PARQUETS[dataset],
                                context=f"predict_and_explain_precomputed[{dataset}]")
        df = pd.read_parquet(PROC_DIR / PRECOMPUTED_HELDOUT_PARQUETS[dataset])
        row = df[(df["battery_id"] == battery_id) & (df["cycle_idx"] == cycle_idx)]
        if row.empty:
            return {"error": f"No precomputed data for {dataset}/{battery_id} cycle {cycle_idx}."}
        r = row.iloc[0]
        his = r.to_dict()
        true_soh = float(his.get("SOH", float("nan")))
        battery_df = df[df["battery_id"] == battery_id].sort_values("cycle_idx")
        base_row = battery_df[battery_df["cycle_idx"] == BASELINE_CYCLE]
        if not base_row.empty:
            baseline_his = base_row.iloc[0].to_dict()
        else:
            # REAL BUG, found and fixed here (not present in the original
            # narrower build_reformulated_hi_vector degradation path, which
            # is a disclosed, deliberate fallback for a genuinely isolated
            # cycle with no battery context - this branch DOES have full
            # battery context, it just wasn't using it). Oxford has ZERO
            # batteries with a real cycle_idx==10 row (confirmed directly -
            # its own HI table samples every ~20-30 cycles, not every
            # cycle) - falling back to None here fired for EVERY Oxford
            # row, not a rare edge case, and produced degraded ICHV_rel/
            # TEVD_rel/TEVI_rel (base model) and SCV_rel/MET_rel/MATD_rel/
            # VIECT_rel (routed model) for all of them. Matches
            # stage5_extended_reformulation._baseline_map's own convention
            # exactly: that battery's own FIRST available cycle, not None -
            # the SAME fix already verified end-to-end in
            # verify_extended_live_feature_parity.py before being ported
            # here (see DEVELOPMENT_LOG.md).
            baseline_his = battery_df.iloc[0].to_dict()
        battery_medians = {"SCV": battery_df["SCV"].median(), "MET": battery_df["MET"].median()}
        fusion_emb = r[[f"fusion_{i}" for i in range(16)]].to_numpy(dtype=float)
        no_temperature = dataset != "HUST"  # HUST has a temperature channel; Oxford/XJTU's raw pipeline does not carry one into this table

    else:
        return {"error": f"Precomputed fallback not implemented for dataset {dataset}."}

    # ORIGINAL (Stage 4/base-model) feature vector - ALWAYS built, regardless of
    # routing: the OC-SVM anomaly check below was fit on THIS feature
    # representation and must never see the extended one (different scale,
    # different columns - routing only ever swaps the SOH point-prediction
    # model, never this diagnostic).
    hi_rel_vector = build_reformulated_hi_vector(his, train_medians, baseline_his)
    hi_vector = np.concatenate([hi_rel_vector, [float(cycle_idx)]])
    feat_vector = np.concatenate([hi_vector, fusion_emb])

    model_variant = None
    fusion_ok, fusion_reason = True, None
    if _use_candidate(dataset) and res.get("xgb_candidate") is not None:
        cand_key = (dataset, battery_id, int(cycle_idx))
        cand_fusion = res["candidate_fusion_lookup"].get(cand_key)
        if cand_fusion is not None:
            fusion_ok, fusion_reason = fusion_range_check(cand_fusion, res)
            pred_soh, shap_vector, shap_model, shap_cols = _candidate_soh_prediction(
                hi_rel_vector, cycle_idx, cand_fusion, res)
            model_variant = "multisource_candidate"
        else:
            # Disclosed fallback, not a silent one: this specific (dataset,
            # battery, cycle) was never scored by the candidate encoder
            # (e.g. a BatteryLife source never wired into PRECOMPUTED_
            # HELDOUT_PARQUETS, or a genuinely missing row) - falls through
            # to the SAME base-model branch every dataset used before this
            # flag existed, exactly as if USE_MULTISOURCE_CANDIDATE were False.
            pred_soh = float(res["xgb_fusion"].predict(feat_vector.reshape(1, -1))[0])
            shap_vector, shap_model, shap_cols = feat_vector, res["xgb_fusion"], FEATURE_COLS
            model_variant = "base_candidate_fallback_no_embedding"
    elif dataset in EXTENDED_ROUTED_DATASETS:
        ext_hi_rel_vector = build_extended_reformulated_hi_vector(
            his, baseline_his, battery_medians, res["ext_medians"])
        ext_hi_vector = np.concatenate([ext_hi_rel_vector, [float(cycle_idx)]])
        ext_feat_vector = np.concatenate([ext_hi_vector, fusion_emb])
        pred_soh = float(res["xgb_fusion_extended"].predict(ext_feat_vector.reshape(1, -1))[0])
        shap_vector, shap_model, shap_cols = ext_feat_vector, res["xgb_fusion_extended"], EXTENDED_FEATURE_COLS
        model_variant = "extended_reformulation"
    else:
        pred_soh = float(res["xgb_fusion"].predict(feat_vector.reshape(1, -1))[0])
        shap_vector, shap_model, shap_cols = feat_vector, res["xgb_fusion"], FEATURE_COLS
        model_variant = "base"

    # NOTE, disclosed limitation: the conformal interval half-width below is
    # the BASE model's own calibrated figure - this routing feature swaps the
    # point-prediction model per dataset, but does not (yet) separately
    # recalibrate conformal intervals for the extended model. Out of this
    # item's scope (would need its own held-out calibration/eval battery
    # split for the extended model) - the point prediction is what's been
    # verified to match the batch-validated numbers, not the interval width.
    soh_half = res["constants"]["soh_conformal_half_width"]

    ocsvm_feat = feat_vector.reshape(1, -1)
    ocsvm_scaled = res["ocsvm_scaler"].transform(ocsvm_feat)
    anomaly_flag = bool(res["ocsvm"].predict(ocsvm_scaled)[0] == -1)

    if trust is None:
        trust = lookup_builtin_trust(dataset, battery_id)
    verdict = domain_verdict(trust)
    out_of_domain, domain_reasons = verdict["out_of_domain"], verdict["domain_reasons"]
    domain_notes = []
    if no_temperature:
        domain_notes.append("no temperature channel present in this dataset (informational; does not decide "
                            "out-of-domain)")
    if anomaly_flag:
        domain_notes.append("the deployed One-Class SVM flags this cycle (informational only - it no longer "
                            "decides out-of-domain)")

    top_features = _tree_shap_top_features(shap_vector, shap_model,
                                            shap_cols + [f"fusion_{i}" for i in range(16)])

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
        "domain_reasons": domain_reasons, "domain_info": verdict,
        "cycle_idx": int(cycle_idx),
        "true_soh": round(true_soh, 1) if true_soh == true_soh else None,
        "true_rul": None,
        "precomputed_fallback": True,
        "trust": trust, "domain_notes": domain_notes,
        "rul_hidden": verdict["rul_hidden"], "rul_hidden_reason": verdict["rul_hidden_reason"],
        "model_variant": model_variant,
        "fusion_unreliable": not fusion_ok, "fusion_unreliable_reason": fusion_reason,
    }


def build_candidate_feature_vector_for_cycle(cycle: dict, res: dict, baseline_his: dict | None) -> np.ndarray | None:
    """The candidate model's own 25-dim feature vector (8 reformulated HI
    + cycle_idx + 16 CANDIDATE fusion-embedding dims) for ONE cycle -
    the same construction `predict_and_explain`'s own candidate branch
    uses inline, factored out here (a SEPARATE, standalone copy, not a
    refactor of that function itself - deliberately, to avoid touching
    already-verified prediction-path behavior for this new, additive
    use: the Phase 3 nearest-source trust report and the upload
    input-sanity check, both of which need this same vector for
    POTENTIALLY MANY cycles of an uploaded battery, not just the one
    currently on screen). Returns None if the cycle's raw tensor can't
    be built (too short/malformed), matching predict_and_explain's own
    "error" convention for that case."""
    train_medians = res["train_medians"]
    his = compute_health_indicators(cycle)
    hi_rel_vector = build_reformulated_hi_vector(his, train_medians, baseline_his)
    hi_vector = np.concatenate([hi_rel_vector, [float(cycle["cycle_idx"])]])

    x_raw = get_cycle_tensor(cycle, n_bins=200)
    if x_raw is None:
        return None
    x_norm_candidate = apply_channel_norm(
        sanitize_raw_tensor_for_candidate(x_raw)[None].astype(np.float32), res["candidate_norm_stats"])[0]
    with torch.no_grad():
        fusion_emb_candidate = res["candidate_encoder"].encode(
            torch.tensor(x_norm_candidate[None, :, ICA_CHANNEL_SLICE])).numpy()[0]
    return np.concatenate([hi_vector, fusion_emb_candidate])


def build_battery_trust_query_vector(cycles: list[dict], res: dict, baseline_his: dict | None,
                                      max_cycles_sampled: int = 30) -> np.ndarray | None:
    """This battery's own representative 25-dim vector for the trust
    report: the MEDIAN (not mean - same robustness reasoning as
    build_source_trust_profiles.py's own battery_mean_vectors, which
    found a handful of individual cycles with clearly-diverged encoder
    outputs) across up to `max_cycles_sampled` evenly-spaced cycles -
    sampled, not every cycle, so this stays fast for a battery with
    hundreds/thousands of cycles (this is a UI-latency-sensitive path,
    unlike the offline profile-building script). Returns None if too
    few cycles produce a usable vector to form a meaningful median."""
    if not cycles:
        return None
    n = len(cycles)
    step = max(1, n // max_cycles_sampled)
    sampled = cycles[::step][:max_cycles_sampled]
    vecs = [v for c in sampled if (v := build_candidate_feature_vector_for_cycle(c, res, baseline_his)) is not None]
    if len(vecs) < 3:
        return None
    return np.median(np.stack(vecs), axis=0)


def candidate_ocsvm_malformed_check(cycles: list[dict], res: dict, baseline_his: dict | None,
                                     max_cycles_sampled: int = 30) -> dict:
    """Input-sanity check for uploads, using the CANDIDATE OC-SVM (see
    load_resources' own comment for why this one, not the deployed
    ocsvm - it's a genuinely discriminating malformed-data detector,
    100% detection of swapped V/I columns / capacity-x10 / truncated
    cycles at 0.6% in-domain false-flag rate). Uses the SAME 25-dim
    candidate feature vector as the trust report (built once per
    sampled cycle, shared between both checks by the caller rather than
    computed twice) - flags per-cycle, then reports the flagged
    FRACTION across sampled cycles, not a single pass/fail off one
    cycle, since a genuinely malformed file usually corrupts most/all
    of its own rows, not just one."""
    if not cycles:
        return {"available": False, "reason": "no cycles to check"}
    n = len(cycles)
    step = max(1, n // max_cycles_sampled)
    sampled = cycles[::step][:max_cycles_sampled]
    vecs = [v for c in sampled if (v := build_candidate_feature_vector_for_cycle(c, res, baseline_his)) is not None]
    if len(vecs) < 3:
        return {"available": False, "reason": "too few usable cycles to check"}
    X = np.stack(vecs)
    X_scaled = res["candidate_ocsvm_scaler"].transform(X)
    flags = res["candidate_ocsvm"].predict(X_scaled) == -1
    frac_flagged = float(flags.mean())
    return {
        "available": True, "n_checked": len(vecs), "n_flagged": int(flags.sum()),
        "frac_flagged": frac_flagged,
        # matches this project's own established OC-SVM calibration read (0.6% in-domain false-
        # flag rate; structural corruptions caught at ~100%) - a THIRD or more of sampled cycles
        # flagged is a real, disclosed threshold choice, not a formula derived from a target FPR.
        "likely_malformed": frac_flagged >= 0.34,
    }


def predict_and_explain(cycle: dict, res: dict, baseline_his: dict | None = None,
                         dataset: str | None = None, battery_medians: dict | None = None,
                         trust: dict | None = None, battery_id: str | None = None) -> dict:
    """cycle: single cycle record (data_adapters convention). baseline_his:
    optional - this battery's own cycle-10 raw HI dict (see
    build_reformulated_hi_vector). Returns a full context dict for
    display, or {"error": ...} if the cycle is too short/malformed for
    the sequence models.

    dataset: optional - enables dataset-aware routing (see
    EXTENDED_ROUTED_DATASETS) for the SOH point prediction ONLY, exactly
    as predict_and_explain_precomputed does. RUL (joint_fusion) is
    NEVER routed - it's a separate model, entirely outside this
    routing's scope, unaffected regardless of dataset. Defaults to None
    (base model, today's existing behavior) for backward compatibility
    with any caller that doesn't pass it.

    battery_medians: optional {"SCV":..., "MET":...} - this battery's
    own median across all its cycles, for the extended model's ratio
    near-zero-baseline guard (see build_extended_reformulated_hi_vector).
    This "live" single-cycle path has no natural full-battery context
    the way predict_and_explain_precomputed's callers do (they already
    loop a whole battery's rows) - if the caller doesn't supply it, the
    near-zero guard falls straight through to ext_medians instead, a
    disclosed simplification for this path only. In practice this guard
    essentially never fires (0 batteries needed it across the entire
    TRAIN+CALCE pool when ext_medians was computed), so this only
    matters for a hypothetical future battery with a genuinely
    near-zero SCV/MET baseline."""
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

    feat_vector = np.concatenate([hi_vector, fusion_emb])  # ORIGINAL vector - OC-SVM always uses this, unchanged

    model_variant = None
    fusion_ok, fusion_reason = True, None
    if _use_candidate(dataset) and res.get("xgb_candidate") is not None:
        # Live path: no precomputed lookup possible (an uploaded/streamed
        # cycle has no saved row) - re-normalize the SAME raw tensor with
        # the CANDIDATE's own channel_norm_stats and re-encode with the
        # CANDIDATE's own encoder (never reuses x_norm/fusion_emb above,
        # which are the OLD deployed encoder's - a different normalization
        # and a different network, mixing them would be a real train/
        # inference mismatch, not just a stylistic choice).
        x_norm_candidate = apply_channel_norm(
            sanitize_raw_tensor_for_candidate(x_raw)[None].astype(np.float32), res["candidate_norm_stats"])[0]
        with torch.no_grad():
            fusion_emb_candidate = res["candidate_encoder"].encode(
                torch.tensor(x_norm_candidate[None, :, ICA_CHANNEL_SLICE])).numpy()[0]
        fusion_ok, fusion_reason = fusion_range_check(fusion_emb_candidate, res)
        pred_soh, shap_vector, shap_model, shap_cols = _candidate_soh_prediction(
            hi_rel_vector, cycle["cycle_idx"], fusion_emb_candidate, res)
        model_variant = "multisource_candidate"
    elif dataset in EXTENDED_ROUTED_DATASETS:
        ext_hi_rel_vector = build_extended_reformulated_hi_vector(
            his, baseline_his, battery_medians or {}, res["ext_medians"])
        ext_hi_vector = np.concatenate([ext_hi_rel_vector, [float(cycle["cycle_idx"])]])
        ext_feat_vector = np.concatenate([ext_hi_vector, fusion_emb])
        pred_soh = float(res["xgb_fusion_extended"].predict(ext_feat_vector.reshape(1, -1))[0])
        shap_vector, shap_model, shap_cols = ext_feat_vector, res["xgb_fusion_extended"], EXTENDED_FEATURE_COLS
        model_variant = "extended_reformulation"
    else:
        pred_soh = float(res["xgb_fusion"].predict(feat_vector.reshape(1, -1))[0])
        shap_vector, shap_model, shap_cols = feat_vector, res["xgb_fusion"], FEATURE_COLS
        model_variant = "base"

    # Same disclosed limitation as predict_and_explain_precomputed: the
    # conformal half-width is the base model's own calibrated figure,
    # not separately recalibrated for the extended model.
    soh_half = res["constants"]["soh_conformal_half_width"]
    rul_half = res["constants"]["rul_conformal_half_width"]

    # OC-SVM anomaly check (same feature vector, in the order saved by train_ocsvm.py)
    ocsvm_feat = feat_vector.reshape(1, -1)
    ocsvm_scaled = res["ocsvm_scaler"].transform(ocsvm_feat)
    anomaly_flag = bool(res["ocsvm"].predict(ocsvm_scaled)[0] == -1)

    if trust is None:
        trust = lookup_builtin_trust(dataset, battery_id)
    verdict = domain_verdict(trust)
    out_of_domain, domain_reasons = verdict["out_of_domain"], verdict["domain_reasons"]
    domain_notes = []
    if no_temperature:
        domain_notes.append("no temperature channel present (informational; does not decide out-of-domain)")
    if anomaly_flag:
        domain_notes.append("the deployed One-Class SVM flags this cycle (informational only - it no longer "
                            "decides out-of-domain)")

    top_features = _tree_shap_top_features(shap_vector, shap_model,
                                            shap_cols + [f"fusion_{i}" for i in range(16)])
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
        "domain_reasons": domain_reasons, "domain_info": verdict,
        "cycle_idx": cycle["cycle_idx"],
        "true_soh": None,  # filled in by the caller if ground truth is available
        "true_rul": None,
        "precomputed_fallback": False,
        "trust": trust, "domain_notes": domain_notes,
        "rul_hidden": verdict["rul_hidden"], "rul_hidden_reason": verdict["rul_hidden_reason"],
        "model_variant": model_variant,
        "fusion_unreliable": not fusion_ok, "fusion_unreliable_reason": fusion_reason,
    }
