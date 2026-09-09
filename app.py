"""
Battery Digital Twin Dashboard (Streamlit).

Uses the FUSION-enabled ensemble (XGBoost+fusion, Stacking-Ridge+fusion)
for SOH, the joint-adaptive model for RUL - NOT the physics-informed
model variants, per instruction. Every prediction is computed LIVE via
src/live_inference.py (no lookup tables) - "a full pipeline rerun per
new upload is fine" per instruction, so this app does not attempt any
incremental/cached meta-learner updating.

UI is organized into 7 tabs (Showcase / Prediction / Explainability /
Health Report / Streaming Digital Twin / Model Validation / Full
Results Archive) purely for presentation - no change to any underlying
computation, model, or data versus the single-page layout this
replaced. Functional over polished: plain Streamlit widgets, matplotlib
and Plotly plots, no custom theming beyond native `st.metric`/colored-
markdown/status-container idioms plus one consistent Plotly palette on
the Showcase tab only.

Session 30 (time-boxed to 1h) added the "Showcase" tab as the new
DEFAULT landing view (first tab): a REPLAY of session 28/29's already-
recorded, already-verified per-cycle results (NOT live recomputation -
a deliberate scope cut, stated in the tab itself), paced on a timer so
it visually looks live. Every existing tab is unchanged.

Session 28 added the "Streaming Digital Twin" tab: a genuine, additive
INCREMENTAL/ONLINE-UPDATE mode (src/digital_twin_streaming.py) that
replays an existing NASA/MIT test battery's cycles one at a time to
simulate live streaming (NOT real hardware) and updates a lightweight
per-battery correction term as it goes - a real change from the "no
incremental updating" scope every other tab still deliberately keeps
(this app's original session-7 write-up in DEVELOPMENT_LOG.md explicitly
scoped that decision; session 28 revisits it as a new, separate, opt-in
mode rather than changing the existing one-shot tabs' behavior).
"""

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from data_adapters import (
    iterate_nasa_cycles, iterate_mit_cycles, iterate_calce_cycles,
    nasa_data_available, mit_data_available, calce_data_available,
)
from live_inference import load_resources, predict_and_explain
from generate_health_report import build_prompt, call_llm
from digital_twin_streaming import StreamingDigitalTwin

ROOT = Path(__file__).resolve().parent
PROC_DIR = ROOT / "data" / "processed"
PRED_DIR = PROC_DIR / "predictions"
OUT_DIR = ROOT / "outputs"

NASA_CELLS = ["B0005", "B0006", "B0007", "B0018"]
CALCE_CELLS = ["CS2_35", "CS2_36", "CS2_37"]

st.set_page_config(page_title="Battery Digital Twin", layout="wide")

# --------------------------------------------------------------------------
# Priority 4 visual pass (session 31, time-boxed): app-wide font pairing +
# restyled alert boxes + restyled buttons. Deliberately scoped to exactly
# these 3 changes, per instruction - no hero redesign, no per-chart Plotly
# theming, no sidebar/tab restyling attempted in this pass.
# Accent color: #2166ac (the same blue already used in the Showcase tab's
# gauge/trend chart, session 30) - kept consistent rather than introducing
# a second, competing accent.
# --------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif;
}
h1, h2, h3, h4, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
}

/* Restyled alert boxes (st.info/st.warning/st.error/st.success) - one
   consistent card look instead of the default muddy yellow/olive fills.
   Streamlit still applies its own per-type background tint underneath;
   this layers a consistent shape/border/shadow on top of all of them. */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-left: 5px solid #2166ac !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.12) !important;
    padding: 0.9rem 1.1rem !important;
}

/* Restyled buttons - accent color + hover state, replacing the default
   Streamlit gray button. */
.stButton > button {
    background-color: #2166ac !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: background-color 0.15s ease-in-out, transform 0.1s ease-in-out;
}
.stButton > button:hover {
    background-color: #14528a !important;
    color: #ffffff !important;
    transform: translateY(-1px);
}
.stButton > button:active {
    background-color: #0f3f6b !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_resources():
    return load_resources()


@st.cache_data(show_spinner=False)
def get_mit_subset():
    with open(PROC_DIR / "mit_subset.json") as f:
        return {e["global_id"]: e for e in json.load(f)}


@st.cache_data(show_spinner=False)
def load_streaming_replay(filename: str) -> pd.DataFrame:
    """Loads a session-28/29 per-cycle streaming record AS-IS - every
    number in the returned frame was already computed and verified when
    those sessions ran (DEVELOPMENT_LOG.md). Nothing here recomputes
    anything."""
    return pd.read_csv(PRED_DIR / filename)


@st.cache_data(show_spinner="Loading battery cycles (CALCE cells take ~1-2 min - xlsx parsing)...")
def load_battery_cycles(dataset: str, battery_id: str):
    if dataset == "NASA":
        return list(iterate_nasa_cycles(battery_id))
    elif dataset == "MIT":
        entry = get_mit_subset()[battery_id]
        return list(iterate_mit_cycles(entry["batch_file"], entry["cell_index"]))
    elif dataset == "CALCE":
        return list(iterate_calce_cycles(battery_id))
    raise ValueError(dataset)


def parse_uploaded_csv(df: pd.DataFrame) -> list[dict]:
    """
    Expected columns: cycle_idx, phase (charge/discharge), time_s,
    voltage_v, current_a, [temperature_c optional]. Current sign
    convention: positive during charge, negative during discharge
    (same as everywhere else in this project) - if your data uses the
    opposite convention, flip the sign of current_a before uploading.
    """
    required = {"cycle_idx", "phase", "time_s", "voltage_v", "current_a"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    has_temp = "temperature_c" in df.columns and df["temperature_c"].notna().any()

    records = []
    for cyc_idx, g in df.groupby("cycle_idx"):
        charge = g[g["phase"] == "charge"].sort_values("time_s")
        discharge = g[g["phase"] == "discharge"].sort_values("time_s")
        if len(charge) < 2 or len(discharge) < 2:
            continue

        def phase_dict(sub):
            return {
                "t": sub["time_s"].to_numpy(float),
                "V": sub["voltage_v"].to_numpy(float),
                "I": sub["current_a"].to_numpy(float),
                "T": sub["temperature_c"].to_numpy(float) if has_temp else None,
            }

        dis_cap = float(np.trapezoid(np.abs(discharge["current_a"]), discharge["time_s"]) / 3600)
        records.append({
            "cycle_idx": int(cyc_idx),
            "discharge_capacity": dis_cap,
            "charge": phase_dict(charge),
            "discharge": phase_dict(discharge),
        })
    return sorted(records, key=lambda r: r["cycle_idx"])


def soh_band(soh_pred: float) -> tuple[str, str, str]:
    """Returns (emoji, streamlit-markdown-color, label) for the 3 SOH health bands."""
    if soh_pred > 80:
        return "🟢", "green", "Healthy"
    elif soh_pred >= 50:
        return "🟡", "orange", "Degraded"
    else:
        return "🔴", "red", "Critical"


def render_about_section():
    st.markdown(
        "**About this dashboard**: this tool estimates a lithium-ion battery's current "
        "**State of Health (SOH)** - its remaining capacity as a percentage of what it "
        "could hold when new - and its **Remaining Useful Life (RUL)** - roughly how many "
        "more charge/discharge cycles it can complete before reaching end of life "
        "(conventionally, 80% of its original capacity). Both predictions come with a "
        "**90% conformal interval**: a statistically-calibrated range built so that, for "
        "batteries similar to the ones this model was trained on, the true value falls "
        "inside that range about 90% of the time - it is a genuine calibrated uncertainty "
        "estimate, not just an arbitrary error bar, though (as this dashboard will warn "
        "you explicitly) that guarantee can break down for batteries unlike the training data."
    )
    st.divider()


def render_prediction_tab(ctx: dict, true_soh, true_rul):
    st.caption("Live predictions for the currently-selected cycle, computed fresh from "
               "trained model weights (no retraining happens in this app).")

    col1, col2 = st.columns(2)
    with col1:
        delta = None if true_soh is None else round(ctx["soh_pred"] - true_soh, 1)
        st.metric("Predicted SOH", f"{ctx['soh_pred']}%",
                   delta=(f"{delta:+.1f} vs. true {true_soh}%" if delta is not None else None))
        emoji, color, label = soh_band(ctx["soh_pred"])
        st.markdown(f"{emoji} :{color}[**{label}**] "
                    f"(bands: green >80% healthy, yellow 50-80% degraded, red <50% critical)")
        st.caption(f"90% conformal interval: {ctx['soh_conformal_lo']}% – {ctx['soh_conformal_hi']}%"
                   + (" ⚠️ unreliable, see banner above" if ctx["out_of_domain"] else ""))
    with col2:
        delta = None if true_rul is None else round(ctx["rul_pred"] - true_rul)
        st.metric("Predicted RUL", f"{ctx['rul_pred']} cycles",
                   delta=(f"{delta:+d} vs. true {true_rul}" if delta is not None else None))
        st.caption(f"90% conformal interval: {ctx['rul_conformal_lo']} – {ctx['rul_conformal_hi']} cycles"
                   + (" ⚠️ unreliable, see banner above" if ctx["out_of_domain"] else ""))
        st.caption("_RUL comes from the Phase 4 joint-adaptive model, the only trained RUL "
                   "predictor in this project - the fusion ensemble itself is SOH-only._")

    st.divider()
    if ctx["anomaly_flag"]:
        st.error("🚨 **Anomaly flag**: the One-Class SVM considers this cycle's feature "
                  "vector unlike the NASA+MIT training distribution.")
    else:
        st.success("✅ No anomaly flagged (One-Class SVM) - this cycle's feature vector "
                    "looks consistent with the NASA+MIT training distribution.")
    st.caption("The anomaly detector is trained only on NASA+MIT data, so it doubles as an "
               "early signal of out-of-domain data alongside the missing-temperature check.")


def render_explainability_tab(ctx: dict):
    st.caption("Per-instance explanation for THIS specific cycle's prediction - not an "
               "average over many cycles.")

    st.subheader("Top contributing features (TreeSHAP)")
    feat_df = pd.DataFrame(ctx["top_features"])
    st.dataframe(feat_df[["feature", "description", "shap_value"]], hide_index=True,
                 width="stretch")
    st.caption("Ranked by |SHAP value| - how much each feature pushed this cycle's SOH "
               "prediction up or down, computed via TreeSHAP on the fusion feature vector.")

    st.subheader("Voltage region driving this prediction (VLSTM DeepSHAP)")
    if ctx["voltage_region"]:
        vr = ctx["voltage_region"]
        st.write(f"**{vr['v_lo']}V – {vr['v_hi']}V** "
                 f"(~{round(vr['frac_of_attribution']*100)}% of attribution)")
        st.caption("This is the voltage window where VLSTM's own gradient-based explanation "
                   "(DeepSHAP) concentrates most of its attention when predicting this "
                   "specific cycle's SOH - computed fresh per cycle, not a fixed constant.")
    elif ctx["voltage_region_error"]:
        st.caption(f"(voltage-region explanation unavailable: {ctx['voltage_region_error']})")


def render_health_report_tab(ctx: dict, dataset: str, battery_id: str):
    st.caption("A plain-English summary of the two panels above, generated by an LLM from "
               "the exact structured numbers shown there (no numbers are invented).")

    context_for_llm = dict(ctx)
    context_for_llm["battery_id"] = battery_id
    context_for_llm["dataset"] = dataset
    prompt = build_prompt(context_for_llm)

    with st.spinner("Generating plain-English report (Gemini, falling back to Groq if needed)..."):
        report, provider = call_llm(prompt)

    if report.startswith("NO_API_KEY") or report.startswith("API_ERROR"):
        st.info(f"Both Gemini and Groq were unavailable ({report}) - showing the structured "
                f"data the LLM would have used instead of a generated narrative:")
        st.json({
            "soh_pred": ctx["soh_pred"], "soh_interval": [ctx["soh_conformal_lo"], ctx["soh_conformal_hi"]],
            "rul_pred": ctx["rul_pred"], "rul_interval": [ctx["rul_conformal_lo"], ctx["rul_conformal_hi"]],
            "top_features": ctx["top_features"], "voltage_region": ctx["voltage_region"],
        })
    else:
        st.write(report)
        st.caption(f"_Generated via {provider}._")


def render_evaluation_protocol_section():
    """
    Surfaces the 3 evaluation-protocol experiments (early-prediction test,
    drop-one-branch ablation, homogeneous-bagging baseline) that were run
    as standalone analyses (src/run_early_prediction_test.py,
    run_drop_branch_ablation.py, run_homogeneous_bagging.py) against the
    fixed test set - NOT re-run live per battery/cycle like the rest of
    this dashboard. Read-only display of already-computed CSVs; see
    DEVELOPMENT_LOG.md for the full narrative and caveats (especially the
    R2-vs-near-zero-target-variance caveat on the early-prediction table).
    """
    st.caption("These 3 experiments evaluate the FIXED TEST SET as a whole - static research "
               "validation, unrelated to whichever battery/cycle is selected in the sidebar. "
               "See DEVELOPMENT_LOG.md for full discussion.")

    with st.expander("📊 Evaluation protocol: early-prediction / drop-branch / bagging experiments",
                      expanded=True):
        st.markdown("**1. Early-prediction test** (first 20% of each battery's cycles)")
        st.warning("R² goes negative here (early-life SOH has almost no variance to "
                   "explain), but RMSE/MAE actually *improve* - use RMSE/MAE, not R², "
                   "to judge this table.")
        early_df = pd.read_csv(PRED_DIR / "early_prediction_test.csv")
        st.dataframe(early_df, hide_index=True, width="stretch")
        early_battery_df = pd.read_csv(PRED_DIR / "early_prediction_per_battery.csv")
        st.dataframe(early_battery_df, hide_index=True, width="stretch")

        st.markdown("**2. Drop-one-branch ablation** (Ridge meta-learner refit without each base learner)")
        drop_df = pd.read_csv(PRED_DIR / "drop_branch_ablation.csv")
        st.dataframe(drop_df, hide_index=True, width="stretch")
        st.caption("Dropping XGBoost collapses performance; dropping any deep model changes "
                   "almost nothing - the ensemble's accuracy is carried almost entirely by "
                   "XGBoost + the fusion embedding.")

        st.markdown("**3. Homogeneous-bagging baseline** (5 XGBoost seeds averaged)")
        bag_df = pd.read_csv(PRED_DIR / "homogeneous_bagging_comparison.csv")
        st.dataframe(bag_df, hide_index=True, width="stretch")
        st.caption("Averaging 5 same-model seeds underperforms both the single best seed "
                   "and the heterogeneous ensemble here - bagging smooths noise without "
                   "adding useful diversity for this dataset.")


def _safe_image(path, caption: str):
    """st.image wrapped so a missing/unreadable file shows a small note
    instead of crashing the whole archive tab."""
    try:
        st.image(str(path))
        st.caption(caption)
    except Exception:
        st.info(f"_(plot not available: `{path.name}`)_")


def _safe_table(path, caption: str, head: int | None = None):
    """st.dataframe wrapped the same way, with an optional row-count preview."""
    try:
        df = pd.read_csv(path)
        shown = df.head(head) if head else df
        st.dataframe(shown, hide_index=True, width="stretch")
        note = f" (previewing first {head} of {len(df):,} rows)" if head and len(df) > head else ""
        st.caption(caption + note)
    except Exception:
        st.info(f"_(table not available: `{path.name}`)_")


_KEYCAP_DIGIT = {str(i): f"{i}️⃣" for i in range(10)}


def _section_num(n: int) -> str:
    """Emoji keycap section number, continuing this tab's existing
    1️⃣.../🔟/1️⃣1️⃣-style numbering past 12 without switching styles
    partway through the tab."""
    if n == 10:
        return "\U0001F51F"  # 🔟
    return "".join(_KEYCAP_DIGIT[d] for d in str(n))


def render_full_results_archive_tab():
    """
    Read-only browse of every result artifact already sitting on disk in
    outputs/ and data/processed/predictions/ (plus the saved LLM health-
    report examples) - no new computation, just surfacing what earlier
    pipeline phases already produced, for a reviewer to browse phase by
    phase. Collapsed by default (st.expander(expanded=False)) so the tab
    loads as a browsable list, not one giant scroll.
    """
    st.caption("Every plot and metrics table already produced by the pipeline, organized by "
               "phase and collapsed by default - expand whichever phase you want to inspect. "
               "Nothing here is recomputed; this is a read-only view of files on disk.")

    with st.expander("1️⃣ BFA Feature Selection"):
        _safe_image(OUT_DIR / "phase1_bfa_convergence.png",
                     "BFA (Butterfly-inspired) feature-selection convergence: best fitness "
                     "and number of selected features vs. iteration, 30 agents × 100 "
                     "iterations. Converged on 7 of 16 candidate Health Indicators.")

    with st.expander("2️⃣ SOH Fade Examples & ICA/DV/DC Example"):
        _safe_image(OUT_DIR / "phase1_soh_fade_examples.png",
                     "Sample SOH-vs-cycle fade curves for NASA, CALCE, and MIT cells, with "
                     "the 80% end-of-life threshold marked.")
        _safe_image(OUT_DIR / "phase1_ica_dv_dc_example.png",
                     "dQ/dV, dV/dQ, and dI/dV curves vs. voltage bin at several cycles of "
                     "NASA B0005, showing how capacity fade shifts these curves over life - "
                     "the basis for this project's ICA/DV/DC-derived Health Indicators.")

    with st.expander("3️⃣ Base Learner Training (incl. the CNN-LSTM root-cause fix)"):
        st.error(
            "**Original run: CNN-LSTM did not learn (R²=-0.071, worse than predicting the "
            "mean).** Root-caused, not left as a known issue: the model's raw `dVdQ` input "
            "channel reached **~9.5 MILLION** in places (every other channel sits at "
            "O(1-100)) and was never normalized, poisoning CNN-LSTM's BatchNorm running "
            "statistics (`running_var` reached ~1e14-1e15). VLSTM (no BatchNorm) and "
            "PiFormer (LayerNorm, no persistent running stat) were architecturally immune - "
            "only CNN-LSTM's global running-average BatchNorm was positioned to break this "
            "way. Fixed with per-channel percentile-clip + z-score normalization, fit on "
            "TRAIN data only."
        )
        st.markdown("**Before → after the fix (same 40-epoch/patience-8 budget):**")
        st.dataframe(pd.DataFrame([
            {"model": "VLSTM", "RMSE before": 2.694, "RMSE after": 2.131, "R² before": 0.690, "R² after": 0.806},
            {"model": "CNN-LSTM", "RMSE before": 5.006, "RMSE after": 3.948, "R² before": -0.071, "R² after": 0.334},
            {"model": "PiFormer", "RMSE before": 2.491, "RMSE after": 2.993, "R² before": 0.735, "R² after": 0.617},
        ]), hide_index=True, width="stretch")
        st.caption("CNN-LSTM: fixed and confirmed by the numbers, but still the weakest of "
                   "the three post-fix. PiFormer's slight regression (0.735→0.617) is "
                   "reported as-is, attributed to ordinary run-to-run training variance, not "
                   "hidden because it doesn't fit the 'the fix helped' narrative.")
        _safe_image(OUT_DIR / "phase2_deep_model_training_curves.png",
                     "Train/validation loss curves for the three deep sequence models "
                     "(VLSTM, CNN-LSTM, PiFormer), post-fix.")
        _safe_image(OUT_DIR / "phase2_cnn_bigru_training_curves.png",
                     "CNN-BiGRU (5th base learner, session 17) training curves - added "
                     "later, same 4-branch CNN front end as CNN-LSTM feeding a "
                     "bidirectional GRU instead of an LSTM.")
        _safe_table(PRED_DIR / "ensemble_comparison.csv",
                    "RMSE/MAE/R² for all 4 base learners plus both stacking meta-learners "
                    "on the held-out test set (post-fix) - XGBoost is the strongest "
                    "individual model.")
        _safe_table(PRED_DIR / "cnn_bigru_metrics.csv",
                    "CNN-BiGRU standalone test metrics (session 17) - lands 4th of 5 base "
                    "learners, beating only CNN-LSTM.")

    with st.expander("4️⃣ Stacking Ensemble"):
        _safe_image(OUT_DIR / "phase3_stacking_parity_plot.png",
                     "Predicted vs. true SOH scatter for the Stacking-Ridge ensemble on the "
                     "test set - points near the diagonal are accurate predictions.")
        _safe_table(PRED_DIR / "ensemble_comparison.csv",
                    "Same comparison table as above, repeated here for the ensemble-vs-"
                    "individual-learner comparison this phase is about.")
        _safe_table(PRED_DIR / "ensemble_test_preds.csv",
                    "Per-cycle predictions from every base learner and both meta-learners "
                    "on the test set.", head=20)

    with st.expander("5️⃣ Feature Fusion"):
        _safe_table(PRED_DIR / "xgb_fusion_metrics.csv",
                    "XGBoost with the ICA/DV/DC fusion embedding added (23 features total) - "
                    "RMSE improves from 1.478 (no fusion) to 1.392.")
        _safe_table(PRED_DIR / "ensemble_fusion_metrics.csv",
                    "Stacking-Ridge with the same fusion embedding added to its meta-"
                    "features - the fusion-enabled ensemble this dashboard's Prediction tab "
                    "actually serves.")

    with st.expander("6️⃣ Physics-Informed Loss Experiment"):
        _safe_table(PRED_DIR / "deep_models_physics_metrics.csv",
                    "The 3 deep models retrained with an added monotonicity-penalty loss "
                    "term (λ=0.1).")
        st.info("ℹ️ **Negative result, kept for documentation only - not adopted into the "
                "final pipeline.** The physics-informed variants did not outperform their "
                "plain counterparts; the dashboard uses the plain (non-physics) models "
                "throughout.")

    with st.expander("7️⃣ Joint SOH+RUL Ablation (incl. the log_sigma clamp fix, session 2)"):
        st.info(
            "**Divergence found and fixed (session 2).** The unconstrained "
            "homoscedastic-uncertainty `adaptive` variant's learned α/β weights grew "
            "unbounded from ~0.7 each at epoch 0 to **8.4 and 8.2 by epoch 24**, and "
            "training loss went **negative** (-1.83) - the `log(sigma)` regularizer term "
            "was numerically dominating the actual prediction-error terms. Fixed by "
            "clamping `log_sigma` to **[-0.7, 0.7]** (bounding α/β to ~[0.12, 2.03]), worked "
            "from the loss math rather than the naive `[-3,3]` suggestion, which wouldn't "
            "have stopped this specific divergence. After the fix, α and β climbed to "
            "exactly **2.028 (the clamp ceiling) by epoch 5** and stayed pinned there - "
            "moving *together*, not asymmetrically, meaning this parametrization mostly "
            "expresses \"how confident overall\" rather than \"how to trade off SOH vs. RUL\"."
        )
        _safe_image(OUT_DIR / "phase4_joint_ablation_curves.png",
                     "Validation SOH loss and validation RUL loss vs. epoch, all 4 loss-"
                     "weighting variants overlaid (fixed_balanced, soh_only, rul_only, "
                     "adaptive).")
        _safe_image(OUT_DIR / "phase4_joint_ablation_bars.png",
                     "Final test RMSE bar charts (SOH and RUL) across the 4 ablation "
                     "variants - single-task variants collapse on whichever target they "
                     "weren't trained on.")
        _safe_image(OUT_DIR / "phase4_adaptive_alpha_beta.png",
                     "Learned α (SOH weight) and β (RUL weight) vs. epoch for the adaptive "
                     "variant - both climb together and pin at the 2.028 clamp ceiling, "
                     "rather than settling on an asymmetric trade-off.")
        _safe_table(PRED_DIR / "joint_ablation.csv",
                    "Final SOH/RUL RMSE, MAE, and R² for all 5 loss-weighting variants "
                    "(fixed_balanced / soh_only / rul_only / adaptive / adaptive_softmax - "
                    "the last added in session 14, see below).")
        st.warning(
            "**Genuine split decision, not a clean win.** `fixed_balanced` still wins "
            "SOH (R²=0.416 vs. adaptive's 0.344); clamped `adaptive` wins RUL, barely "
            "(R²=0.432 vs. fixed_balanced's 0.428). Single-task collapse is textbook-clean "
            "in both directions (soh_only RUL R²=0.030; rul_only SOH R²=-0.040, actually "
            "worse than predicting the mean)."
        )

    with st.expander("8️⃣ SHAP Explainability"):
        _safe_image(OUT_DIR / "phase5_shap_xgboost_ranking.png",
                     "Mean |SHAP| bar chart for XGBoost's 7 BFA-selected Health Indicator "
                     "features - SCV, VIECT, and TEVI dominate.")
        _safe_image(OUT_DIR / "phase5_shap_meta_ranking.png",
                     "Mean |SHAP| bar chart for the 4 base-learner inputs to the Stacking-"
                     "XGBoost meta-learner - over 99.9% of the ensemble's prediction is "
                     "XGBoost alone.")
        _safe_image(OUT_DIR / "phase5_shap_voltage_region.png",
                     "Fraction of |SHAP| attribution mass falling in the 3.55-3.8V window, "
                     "per deep model (DeepSHAP) - VLSTM concentrates 60% of its attribution "
                     "in this narrow, physically-meaningful plateau region.")
        _safe_table(OUT_DIR / "shap_xgboost_base_ranking.csv",
                    "XGBoost's Health-Indicator SHAP ranking, numeric values behind the "
                    "first plot above.")
        _safe_table(OUT_DIR / "shap_meta_ranking.csv",
                    "Meta-learner base-learner SHAP ranking, numeric values behind the "
                    "second plot above.")
        _safe_table(OUT_DIR / "shap_deep_models_summary.csv",
                    "Voltage-region attribution fractions, numeric values behind the third "
                    "plot above.")

    with st.expander("9️⃣ Split-Conformal Prediction (incl. the 27.1% calibration bug)"):
        st.error(
            "**A real methodological bug, caught by checking the numbers, not just running "
            "the code.** The first draft calibrated on the TRAIN split's own residuals (the "
            "same rows the meta-learner was fit on) - ran without error, but empirical "
            "coverage came back **27.1% against a 90% target**. A from-scratch manual "
            "reimplementation gave the identical 27.1%, ruling out a MAPIE-wrapper bug: the "
            "real cause was calibration/test residual medians differing ~6.7x because "
            "calibration rows were in-sample. **Fixed** by splitting the 6 held-out TEST "
            "batteries in half (`calib=[B0018,b2c24,b3c35]`, `eval=[b1c4,b3c0,b4c38]`) so "
            "both halves are genuinely unseen by every fitted model - after the fix, SOH "
            "coverage is 95.1%."
        )
        _safe_image(OUT_DIR / "phase6_conformal_soh.png",
                     "SOH point predictions with 90% conformal interval band vs. true "
                     "values, evaluation batteries.")
        _safe_image(OUT_DIR / "phase6_conformal_rul.png",
                     "RUL point predictions with 90% conformal interval band vs. true "
                     "values, evaluation batteries.")
        _safe_table(OUT_DIR / "conformal_coverage.csv",
                    "Empirical coverage and average interval width for both targets - SOH "
                    "slightly over-covers (95.1%), RUL over-covers too (93.0%, corrected - "
                    "see DEVELOPMENT_LOG.md for the RUL conformal investigation).")

    with st.expander("🔟 CALCE Zero-Retrain Evaluation"):
        _safe_table(PRED_DIR / "calce_zero_retrain_metrics.csv",
                    "The fusion ensemble's SOH accuracy on CALCE cells with ZERO retraining "
                    "- a genuine out-of-domain test (different chemistry/format, no "
                    "temperature channel).")
        _safe_table(OUT_DIR / "calce_zero_retrain_conformal.csv",
                    "The key finding: the SAME fixed conformal interval width looks "
                    "identically confident in-domain and out-of-domain, but empirical "
                    "coverage collapses from 95.6% (NASA+MIT) to just 6.1% (CALCE) - the "
                    "motivation for this dashboard's out-of-domain warning banner.")

    with st.expander("1️⃣1️⃣ Session 7 — Dashboard v1 (OC-SVM + negative-RUL bugs)"):
        st.markdown("No standalone metrics CSV for this session - both findings summarized "
                    "here in text, matching how they were logged in DEVELOPMENT_LOG.md.")
        st.info(
            "**Bug 1 - class-imbalance, found and fixed.** The first-pass One-Class SVM was "
            "trained on all fit-split cycles as-is: NASA (~470 cycles) vs. MIT (~14,400 "
            "cycles), a ~30:1 imbalance. Sanity-checking it against its own training data "
            "found **83.9% of NASA's own cycles flagged \"anomalous\"** vs. only 2.2% of "
            "MIT's - meaning a perfectly legitimate NASA battery would trigger the "
            "dashboard's out-of-domain warning almost every time. Fixed by capping each fit "
            "battery to 200 cycles before fitting; NASA's false-flag rate dropped to 24.4% "
            "(MIT stayed ~2.3%) - a large improvement, though not perfectly balanced, and "
            "left documented as a residual limitation rather than claimed as fully solved."
        )
        st.info(
            "**Bug 2 - negative RUL predictions displayed to the user, found by `AppTest` "
            "on the very first automated run.** NASA B0005's last logged cycle (already past "
            "its true EOL) produced a raw RUL prediction of **-15 cycles** - mathematically "
            "a faithful regression residual, but meaningless to show a dashboard user. Fixed "
            "by clipping the displayed value to `max(0, ...)`."
        )
        st.caption("Verified via `streamlit.testing.v1.AppTest` across 4 paths, zero "
                   "exceptions: NASA/B0005 (SOH 71.9% vs. true 71.8%), MIT/b1c17 (SOH 82.6% "
                   "vs. true 82.3%), CALCE/CS2_35 (SOH 66.0% vs. true 26.7% - a huge miss, "
                   "out-of-domain warning correctly triggered), and a synthetic uploaded CSV "
                   "(correctly flagged anomalous and out-of-domain).")

    with st.expander("1️⃣2️⃣ Health Report Examples (sessions 6 & 8 — Claude → Gemini)"):
        st.caption("5 saved example reports from `outputs/health_reports_examples.json` - "
                   "generated by the same prompt/LLM chain the Health Report tab uses live, "
                   "shown here as readable text rather than raw JSON.")
        st.info(
            "**Session 8 - switched the LLM provider from Anthropic to Gemini** (user-"
            "provided `GEMINI_API_KEY`). Real finding, not a code bug: the requested model "
            "`gemini-2.5-flash` returned **HTTP 404** (\"no longer available to new users\") "
            "despite appearing in the same key's own model listing - reproduced "
            "independently via raw `curl`. `gemini-2.0-flash` hit a separate 429 rate limit. "
            "`gemini-flash-latest` confirmed working and substituted as the default, "
            "documented as a deviation rather than silently swapped. All 5 example reports "
            "below were then re-run live through the real Gemini API (session 8), after "
            "originally being Claude-authored (session 6, when no API key existed) - every "
            "SOH/RUL figure verified correct in every report, zero invented numbers."
        )
        try:
            examples = json.loads((OUT_DIR / "health_reports_examples.json").read_text())
            for i, ex in enumerate(examples, 1):
                ctx = ex.get("context", {})
                provider = ex.get("provider", "not recorded (saved before the Groq fallback was added)")
                st.markdown(
                    f"**Example {i}: {ctx.get('battery_id', '?')} "
                    f"({ctx.get('dataset', '?')}, cycle {ctx.get('cycle_idx', '?')})** "
                    f"— _via {provider}_"
                )
                st.write(ex.get("report", "_(no report text saved)_"))
                st.divider()
        except Exception:
            st.info("_(health_reports_examples.json not available)_")

    with st.expander(f"{_section_num(13)} Session 9 — 3 evaluation-protocol experiments"):
        st.caption("Same 3 experiments as the 🧪 Model Validation tab, included here too so "
                   "the archive is a complete, standalone record - all evaluation-only, no "
                   "base learner retrained.")
        st.markdown("**1. Early-prediction test** (first 20% of each battery's cycles)")
        st.warning("R² goes negative here (early-life SOH has almost no variance to "
                   "explain), but RMSE/MAE actually *improve* - use RMSE/MAE, not R², "
                   "to judge this table. b2c24's R²=-52.3 is not a real failure - its true-"
                   "SOH range is only 0.5 points wide in this window.")
        _safe_table(PRED_DIR / "early_prediction_test.csv", "Pooled early-life vs. full-lifetime metrics.")
        _safe_table(PRED_DIR / "early_prediction_per_battery.csv", "Per-battery early-life breakdown.")
        st.markdown("**2. Drop-one-branch ablation** (5-branch, incl. CNN-BiGRU, session 17)")
        _safe_table(PRED_DIR / "drop_branch_ablation_5branch.csv",
                    "Dropping XGBoost-fusion collapses performance (ΔR²=-0.097); dropping "
                    "any deep model changes almost nothing (≤0.0002 R²) - dropping "
                    "CNN-BiGRU is very slightly the single BEST-performing ablation row.")
        st.markdown("**3. Homogeneous-bagging baseline** (5 XGBoost seeds averaged)")
        _safe_table(PRED_DIR / "homogeneous_bagging_comparison.csv",
                    "Averaging 5 same-model seeds underperforms both the single best seed "
                    "and the heterogeneous ensemble - bagging smooths noise without adding "
                    "useful diversity for this dataset.")

    with st.expander(f"{_section_num(14)} Session 11 — RUL conformal coverage investigation"):
        st.info(
            "**Finding 1: the 88.9% coverage figure was stale, not a live bug.** Re-running "
            "the calibration script unmodified against the documented split now gives "
            "**93.0% coverage** - the RUL model had been retrained again (session 2's "
            "log_sigma clamp) after 88.9% was measured, but conformal calibration was never "
            "re-run against the new checkpoint."
        )
        st.warning(
            "**Finding 2: the real issue has no small fix.** Only 6 test batteries means "
            "single-split coverage is inherently high-variance. Evaluating all 20 possible "
            "3-battery-calib/3-battery-eval partitions of the 6 test batteries gave "
            "**coverage ranging 64.7% to 99.6%** (mean 87.4%, std 10 points) - purely a "
            "function of which batteries land in calib vs. eval, since per-battery RUL RMSE "
            "varies ~13x across the 6 test batteries. Documented as a structural, "
            "not-fixable-here limitation of this dataset's battery count."
        )
        _safe_table(OUT_DIR / "conformal_coverage.csv", "Current (corrected, 93.0%) RUL coverage, alongside SOH's.")

    with st.expander(f"{_section_num(15)} Session 13 — MMD domain adaptation"):
        st.caption("Maximum Mean Discrepancy alignment on the fusion embedding, retrained "
                   "against CALCE's UNLABELED inputs only (zero label leakage) - does it "
                   "fix the CALCE collapse from session 5?")
        st.error(
            "**Bug caught before trusting any result: λ=1.0 silently broke training** "
            "(validation loss got worse every epoch, early-stopping kept an essentially "
            "untrained encoder). Swept down to **λ=0.1**, which trained cleanly and reached "
            "a slightly BETTER validation loss than the non-MMD baseline."
        )
        _safe_table(PRED_DIR / "xgb_fusion_mmd_metrics.csv", "XGBoost-fusion-MMD in-domain metrics.")
        _safe_table(PRED_DIR / "ensemble_fusion_mmd_metrics.csv", "Stacking-Ridge-fusion-MMD in-domain metrics.")
        _safe_table(PRED_DIR / "calce_zero_retrain_mmd_metrics.csv",
                    "CALCE R² improves modestly and genuinely: 0.304→0.337 (XGBoost), "
                    "0.314→0.347 (Ridge) - still >12x the in-domain RMSE.")
        _safe_table(OUT_DIR / "calce_zero_retrain_mmd_conformal.csv",
                    "**Conformal coverage gets slightly WORSE, not better: 6.1%→4.4%.** A "
                    "narrower recalibrated interval (2.367→2.217) covers *less* of a still-"
                    "catastrophically-wrong prediction distribution - MMD does not fix, and "
                    "here slightly worsens, the conformal miscalibration problem.")

    with st.expander(f"{_section_num(16)} Session 14 — softmax-normalized adaptive loss weighting"):
        st.caption("Constrains (α,β) = 2·softmax(s_α, s_β), pinning α+β=2 so one weight can "
                   "only rise at the other's direct expense - does this fix the α=β=2.028 "
                   "collapse from the log_sigma-clamped `adaptive` variant above?")
        st.success("**Yes - α/β are now genuinely asymmetric**: 0.993/1.007 at epoch 0 → "
                   "**0.527/1.473 by epoch 24**, steadily diverging rather than moving together.")
        st.error(
            "**But the model is worse than BOTH baselines on BOTH tasks.** "
            "`adaptive_softmax`: SOH R²=0.091 (vs. fixed_balanced's 0.416, adaptive's "
            "0.344); RUL R²=0.244 (vs. fixed_balanced's 0.428, adaptive's 0.432). β rises "
            "monotonically with no regularizer opposing the drift, runaway-starving the SOH "
            "head - **structurally forcing asymmetry does not guarantee the asymmetry found "
            "is a GOOD one.**"
        )
        _safe_table(PRED_DIR / "joint_ablation.csv",
                    "Same table as section 7️⃣, now including the adaptive_softmax row.")

    with st.expander(f"{_section_num(17)} Session 15 — LIME cross-validation of TreeSHAP"):
        st.caption("For 5 sampled test-set predictions per model, does an entirely "
                   "independent explanation method (LIME's local-linear surrogate) agree "
                   "with TreeSHAP's exact game-theoretic attribution?")
        _safe_table(OUT_DIR / "lime_shap_comparison.csv",
                    "8/10 instances reached full 3/3 top-3 agreement; overall mean top-3 "
                    "overlap 93.3%. XGBoost-base: 100% (5/5 full match). XGBoost-meta: "
                    "86.7% (3/5 full, 2/5 at 2/3) - disagreements are on 2nd/3rd-ranked "
                    "features only, never on which base learner dominates (pred_XGBoost was "
                    "in every single top-3 from both methods, 10/10).")

    with st.expander(f"{_section_num(18)} Session 16 — knee-point detection"):
        st.caption("Curvature-based knee detection (Savitzky-Golay derivatives, "
                   "κ=|y''|/(1+y'²)^1.5) on predicted vs. true SOH curves, matching the "
                   "BatteryGPT reference definition.")
        _safe_table(OUT_DIR / "knee_point_detection.csv",
                    "Literal mean absolute offset = 161.8 cycles (17.0% of lifetime) across "
                    "6 batteries.")
        st.warning(
            "**Both outliers root-caused, not just noted.** b3c0's 911-cycle offset is a "
            "GROUND-TRUTH-side artifact: its true SOH rises slightly above 100% for its "
            "first ~10 cycles (a real formation/break-in effect), producing the single "
            "highest curvature value in the whole true curve - cycle 8 genuinely IS the "
            "global-argmax-curvature point by this definition. b2c24's 55-cycle offset is a "
            "PREDICTION-side artifact: a real single-cycle discontinuity in the predicted "
            "curve around cycle 204 produces a spurious curvature spike ~8x every other "
            "value. Excluding the ground-truth artifact (b3c0) gives a mean of **12.0 "
            "cycles** across the remaining 5 batteries - both numbers reported, neither "
            "picked to look better."
        )

    with st.expander(f"{_section_num(19)} Session 17 — CNN-BiGRU as a 5th base learner"):
        st.caption("Same 4-branch CNN front end as CNN-LSTM, feeding a Bidirectional GRU "
                   "instead of a unidirectional LSTM - a genuine test, reported either way.")
        _safe_table(PRED_DIR / "cnn_bigru_metrics.csv",
                    "Standalone: R²=0.572, 4th of 5 base learners (beats only CNN-LSTM).")
        _safe_table(PRED_DIR / "drop_branch_ablation_5branch.csv",
                    "Ensemble contribution: adding CNN-BiGRU as a 5th branch makes the "
                    "ensemble marginally WORSE overall (R² 0.916947→0.916884) - dropping "
                    "its own column is the single best-performing ablation row. A better "
                    "standalone recurrent core did not translate into a better ensemble "
                    "branch.")
        _safe_image(OUT_DIR / "phase2_cnn_bigru_training_curves.png", "CNN-BiGRU training curves.")

    with st.expander(f"{_section_num(20)} Session 18 — consolidated convergence comparison"):
        _safe_image(OUT_DIR / "phase8_convergence_comparison.png",
                     "Training-loss-vs-epoch overlay, all 4 deep models on one chart.")
        st.caption("PiFormer converges fastest (best epoch 12 of 40) but stops earliest "
                   "(mild early overfitting); VLSTM converges slowest (epoch 34) but reaches "
                   "the lowest overall validation loss (0.269); CNN-BiGRU trains most "
                   "smoothly (smallest transient loss spikes, +0.0035 max) without being "
                   "fastest or best; CNN-LSTM is both slower than PiFormer/CNN-BiGRU AND "
                   "the least stable (+0.0359 max spike), consistent with remaining the "
                   "weakest base learner throughout.")

    with st.expander(f"{_section_num(21)} Session 19 — domain-shift-aware conformal prediction"):
        st.caption("Weighted split-conformal (Tibshirani et al. 2019): calibration "
                   "residuals reweighted by a covariate-shift density ratio from a "
                   "logistic-regression domain classifier - does this fix CALCE's coverage?")
        st.error(
            "**First attempt (full 7-HI+16-fusion feature space) broke something that "
            "wasn't broken.** The domain classifier hit AUC=1.0000 for calib-vs-CALCE, but "
            "ALSO AUC=0.9021 for the supposed in-domain sanity check (should be ~0.5 with "
            "only 3-vs-3 batteries) - in-domain coverage collapsed from 94.6% to **43.6%**, "
            "and 100% of CALCE points got a vacuous infinite-width interval."
        )
        _safe_table(OUT_DIR / "domain_shift_conformal_summary.csv", "Full-feature-space result (degenerate).")
        _safe_table(OUT_DIR / "domain_shift_conformal_summary_fusion_only.csv",
                    "Better-behaved fusion-only (16-dim) variant: in-domain coverage 69.6% "
                    "(down from 94.6%), CALCE coverage still just **4.4%** - statistically "
                    "identical to session 13's fixed-width result. A genuine partial "
                    "result, not a fix: the interval now differs by domain, but does not "
                    "meaningfully improve CALCE coverage.")

    with st.expander(f"{_section_num(22)} Session 20 — \"lean\" deployment vs. the full 5-branch ensemble"):
        _safe_table(OUT_DIR / "lean_vs_full_comparison.csv",
                    "LEAN (XGBoost-fusion only) matches or slightly beats FULL (5-branch + "
                    "Ridge meta) on accuracy (R² 0.91715 vs. 0.91688), while being "
                    "**~52x faster** per prediction (3.9ms vs. 201.9ms, batch=1) and "
                    "requiring 3.5x fewer model invocations (2 vs. 7). Only 1.1x smaller on "
                    "disk - `xgb_soh_fusion.json` alone is 99.7% of LEAN's size, so latency "
                    "and complexity are where the real savings are, not size. "
                    "**Recommendation: ship LEAN.**")

    with st.expander(f"{_section_num(23)} Session 21 — bootstrap confidence intervals"):
        st.caption("2,000-resample percentile bootstrap CIs at BOTH cycle-level (literal "
                   "request, but pseudo-replicated - ~5,208 autocorrelated cycles treated "
                   "as independent) and battery-level (cluster bootstrap over the 6 test "
                   "batteries - the honest resampling unit).")
        st.warning(
            "**XGBoost's dominance is large in point-estimate terms but is NOT "
            "battery-level statistically significant** with only 6 test batteries "
            "(CI [-0.054, +0.239] includes zero) - a genuine statistical-power limitation, "
            "not evidence the effect is fake. Its edge over the 3 WEAKER deep models IS "
            "battery-level significant; its edge over VLSTM specifically is not. Session "
            "20's \"LEAN is better\" edge does NOT survive battery-level resampling - LEAN "
            "and FULL are statistically indistinguishable in accuracy, which if anything "
            "strengthens the case for shipping lean (no accuracy trade-off at all)."
        )
        _safe_table(PRED_DIR / "bootstrap_drop_branch_ci.csv", "Drop-branch ablation CIs, cycle- and battery-level.")
        _safe_table(PRED_DIR / "bootstrap_base_learner_r2_ci.csv", "Base-learner R² point estimate + cycle-level CI.")
        _safe_table(PRED_DIR / "bootstrap_base_learner_delta_vs_xgb_ci.csv", "XGBoost vs. each deep model, both CI levels.")
        _safe_table(PRED_DIR / "bootstrap_lean_vs_full_ci.csv", "Lean vs. full deltas, both CI levels.")

    with st.expander(f"{_section_num(24)} Session 22 — NASA EIS features as candidate Health Indicators"):
        st.caption("NASA's .mat files carry already-fitted equivalent-circuit impedance "
                   "parameters (Re, Rct) - tested honestly against BFA's existing 7 features.")
        try:
            eis_selected = (PROC_DIR / "bfa_selected_features_with_eis.txt").read_text().strip()
            st.markdown(f"**19-candidate BFA re-run (7 original + 3 EIS + 9 other candidates) selected:** `{eis_selected}`")
        except Exception:
            st.info("_(bfa_selected_features_with_eis.txt not available)_")
        st.warning(
            "**None of the 3 EIS-derived features were selected.** Plausibly explained by "
            "severe data-availability missingness, not by EIS being uninformative in "
            "principle: EIS is NASA-only, so **97.6% of all pooled rows (26,360/26,996) are "
            "NaN** for these columns, imputed with the NASA-only median - a feature that's "
            "only ever real for ~2% of a pooled, battery-grouped-CV wrapper search has very "
            "little room to prove its value. Read as \"EIS didn't help THIS pooled, "
            "EIS-sparse dataset,\" not \"EIS doesn't matter for battery SOH.\""
        )

    with st.expander(f"{_section_num(25)} Session 23 — degradation-mode analysis (dV/dQ peak-tracking)"):
        st.caption("Inspired by DVA degradation-mode literature (Bloom et al. 2005; "
                   "Dubarry et al. 2012) - peak position shift ↔ LLI, height loss ↔ LAM. "
                   "Explicitly NOT a validated LLI/LAM decomposition (no half-cell reference "
                   "data available in any of the 3 datasets) - a qualitative leaning only.")
        _safe_table(OUT_DIR / "degradation_mode_summary.csv",
                    "NASA/B0018 shows a mixed LLI+LAM-leaning signature (the most degraded "
                    "of the 3 checked); both MIT cells show pure LAM-leaning signatures.")
        _safe_table(PRED_DIR / "degradation_mode_peak_tracks.csv",
                    "Per-cycle tracked peak position/height (NaN where tracking was lost, "
                    "not silently interpolated).", head=20)
        st.caption("Two mid-analysis corrections logged rather than hidden: raw peak VALUE "
                   "was numerically unstable (367502→6318→9119→7632 swings for a position-"
                   "stable peak) - switched to peak PROMINENCE; cycle 1 was a reproducible "
                   "outlier on every battery (an SG-filter boundary artifact) - fixed by "
                   "baselining height on the median of the first/last 5 tracked cycles.")

    with st.expander(f"{_section_num(26)} Session 24 — model quantization / TinyML feasibility"):
        _safe_table(OUT_DIR / "model_quantization_summary.csv",
                    "FP16 (5.90KB) actually beats INT8 (6.12KB) in absolute size for this "
                    "tiny 1,665-parameter encoder - INT8's per-channel calibration metadata "
                    "overhead eats most of its theoretical 4x storage win at this scale. "
                    "Accuracy changes at either precision are negligible (≤0.0003 R²).")
        st.error(
            "**Honest headline: quantizing the encoder was almost beside the point.** "
            "ICAEncoder is 9.15KB of a 2,854.9KB total lean-pipeline size (0.3%) - "
            "`xgb_soh_fusion.json` (2,845.7KB) is 99.7% of the total. Even best-case "
            "quantized (~1,987KB), this pipeline is **3.9x to 62x OVER** a typical small "
            "BMS microcontroller's flash budget (32KB-512KB) - almost entirely XGBoost's "
            "size, not the encoder's. **This pipeline, even after quantization, is not "
            "feasible on typical microcontroller-class BMS hardware.**"
        )

    with st.expander(f"{_section_num(27)} Session 25 — second-life grading classifier"):
        st.caption("Pure post-processing on lean-pipeline SOH predictions: ≥80% Primary EV "
                   "use, 50-80% Second-life candidate, <50% Recycle only. Overall grading "
                   "agreement: 98.75% of 5,208 test cycles.")
        _safe_table(OUT_DIR / "second_life_grading_current_status.csv",
                    "Per-battery LAST-cycle grading - the realistic triage moment.")
        st.error(
            "**Most operationally important finding: at the exact moment a real "
            "disposition decision would be made for NASA/B0018 today, this model would "
            "incorrectly certify it fit for continued primary EV use** (predicted 81.50% "
            "vs. true 72.76%, an 8.7-point overestimate crossing the 80% line) - a "
            "**sustained** systematic overestimate: 41 of B0018's last 56 cycles are risky "
            "misgrades, not a boundary-noise blip. Every other test battery grades "
            "correctly at its last cycle."
        )
        _safe_table(OUT_DIR / "second_life_grading_per_battery_distribution.csv",
                    "Per-battery full-life grade distribution - B0018 is the only battery "
                    "spending a substantial fraction of its life (42.4%) in the "
                    "second-life bracket rather than staying almost entirely primary-use.")

    with st.expander(f"{_section_num(28)} Session 26 — sensor-noise robustness"):
        st.caption("Gaussian noise on every raw V/I/T sample, 3 levels (1x/2x/5x BMS-grade). "
                   "Ground-truth SOH left unperturbed to isolate prediction degradation.")
        _safe_table(OUT_DIR / "sensor_noise_robustness_summary.csv",
                    "Pooled result looks like near-total noise immunity (R² even ticks "
                    "slightly UP at 5x stress) - not accepted at face value.")
        st.warning(
            "**Per-battery breakdown reveals the pooled number is HIDING a real, "
            "asymmetric weakness.** NASA/B0018 degrades MONOTONICALLY at every noise level "
            "(R² 0.576→0.565→0.525→**0.513** at 5x); the other MIT batteries show mild "
            "improvement, masking B0018's real degradation in the pooled average. The same "
            "battery session 25 already flagged as this pipeline's weakest case, now "
            "confirmed by a second, unrelated stress-test."
        )
        _safe_table(PRED_DIR / "sensor_noise_robustness_per_cycle.csv",
                    "Per-cycle predictions at every noise level.", head=20)

    with st.expander(f"{_section_num(29)} Session 27 — root-causing NASA/B0018's weak point"):
        st.caption("Four angles converging on one root cause: NASA's cycling protocol and "
                   "training representation differ fundamentally from MIT's.")
        _safe_table(OUT_DIR / "b0018_rootcause_lifetime.csv",
                    "B0018 fades 4.9x faster per cycle than even the fastest-fading MIT "
                    "test battery.")
        _safe_table(OUT_DIR / "b0018_rootcause_domain_auc.csv",
                    "B0018 is the ONLY test battery to hit a perfect AUC=1.0000 vs. "
                    "MIT-train - categorically beyond the already-elevated MIT range.")
        _safe_table(OUT_DIR / "b0018_rootcause_feature_zscores.csv",
                    "The mechanistic explanation: ICHV and TEVI (raw wall-clock-time "
                    "durations) are z=855 and z=420 away from the MIT-train mean for B0018 "
                    "- every single one of 18,341 MIT training cycles has a LOWER value, a "
                    "complete non-overlap caused by NASA's slow protocol vs. MIT's "
                    "fast-charging protocol.", head=10)
        _safe_table(OUT_DIR / "b0018_rootcause_degradation_mode.csv",
                    "B0018 is the only one of all 6 test batteries whose peak-position "
                    "shift crosses the 5% LLI-leaning threshold; every MIT battery shows a "
                    "pure LAM-leaning signature.")
        st.error(
            "**Synthesis: NASA training representation is only 2.7% of training CYCLES "
            "(vs. 97.3% MIT)** despite being 11.5% of training batteries - B0018 is a mild, "
            "within-project echo of the exact CALCE domain-shift problem, just with SOME "
            "(rather than zero) training representation, which is presumably why it's "
            "\"merely\" this pipeline's weakest test case rather than a CALCE-scale collapse."
        )

    with st.expander(f"{_section_num(30)} Session 28 — streaming Digital Twin (online learning)"):
        st.caption("Simulation-stage only - replays already-recorded test-battery cycles "
                   "with an artificial per-cycle delay. NOT connected to real hardware. "
                   "See the 🌊 Streaming Digital Twin tab for the live version of this.")
        st.error(
            "**A genuine bug caught by \"test it actually updates\" before it ever reached "
            "the app.** With unscaled raw SOH predictions (~70-100) and a fixed SGD "
            "learning rate, the online corrector's coefficients exploded to ~1e11 and "
            "predictions reached the **TRILLIONS** "
            "(`corrected_pred=7,159,720,237,468.60` observed on B0018's stream). Fixed via "
            "centered/scaled corrector inputs + `learning_rate=\"invscaling\"` + a "
            "defense-in-depth ±25pp correction clamp."
        )
        st.dataframe(pd.DataFrame([
            {"battery": "NASA/B0018", "overall MAE raw→corrected": "4.641 → 4.426",
             "final-cycle err raw→corrected": "8.74 → 6.89", "interval half-width first→last": "8.16 → 7.57 (narrowed)"},
            {"battery": "MIT/b3c35", "overall MAE raw→corrected": "0.439 → 0.133",
             "final-cycle err raw→corrected": "1.22 → 0.03", "interval half-width first→last": "0.029 → 0.143 (widened, 5x)"},
        ]), hide_index=True, width="stretch")
        st.caption("Online correction genuinely helps on both batteries, and converges to a "
                   "BETTER final prediction than the frozen one-shot pipeline. Honest "
                   "exception, not hidden: b3c35's conformal interval WIDENS rather than "
                   "narrowing - diagnosed as an artifact of unusually small early residuals "
                   "producing an artificially tiny initial half-width.")
        _safe_table(PRED_DIR / "streaming_dt_NASA_B0018.csv", "Per-cycle streaming record, NASA/B0018.", head=15)
        _safe_table(PRED_DIR / "streaming_dt_MIT_b3c35.csv", "Per-cycle streaming record, MIT/b3c35.", head=15)

    with st.expander(f"{_section_num(31)} Session 29 — Adaptive Conformal Inference (ACI)"):
        st.caption("Replaces session 28's fixed-alpha sliding-window conformal mechanism "
                   "with ACI (Gibbs & Candès 2021) - the SGDRegressor corrector itself is "
                   "completely UNCHANGED (confirmed by identical MAE numbers). Addresses "
                   "the broken exchangeability assumption: the corrector keeps updating "
                   "online, so calibration is a moving target.")
        st.dataframe(pd.DataFrame([
            {"battery": "NASA/B0018", "coverage: sliding-window (session 28)": "82.0%", "coverage: ACI (session 29)": "85.9%",
             "max half-width: sliding-window": 9.060, "max half-width: ACI": 9.849},
            {"battery": "MIT/b3c35", "coverage: sliding-window (session 28)": "84.2%", "coverage: ACI (session 29)": "87.2%",
             "max half-width: sliding-window": 1.040, "max half-width: ACI": 4.547},
        ]), hide_index=True, width="stretch")
        st.warning(
            "**Genuinely mixed, not a clean win.** On coverage - the metric that actually "
            "matters for conformal validity - ACI improves on BOTH batteries (82.0%→85.9%, "
            "84.2%→87.2%). But on raw half-width volatility, ACI does NOT make b3c35 "
            "smoother - it makes it **4.4x MORE volatile** (max half-width 4.547 vs. "
            "1.040). Neither battery reaches the 90% coverage target within its observed "
            "stream - stated honestly as ACI's guarantee being a long-run average property "
            "these short streams may not fully converge within."
        )
        st.caption("The final per-cycle CSVs in the session above (streaming_dt_*.csv) "
                   "already reflect this ACI-based conformal mechanism, not session 28's "
                   "original sliding-window version, which was superseded on disk.")

    st.markdown("**Sessions not duplicated here**: session 10 (surfacing the eval-protocol "
                "experiments in the dashboard) is the 🧪 **Model Validation** tab and this "
                "archive tab themselves; session 12 (graceful degradation when raw data is "
                "unavailable) and session 31 (visual polish pass) are behavioral/styling "
                "fixes to this app with no standalone result artifact; session 30 (the "
                "Digital Twin Showcase) is the 🎬 **Showcase** tab. See "
                "`PRESENTATION_SUMMARY.md` at the repo root for the complete session-by-"
                "session narrative these tables summarize.")


# --------------------------------------------------------------------------
# Showcase tab (session 30, time-boxed to 1h) - the new DEFAULT landing
# view (first tab). REPLAY of already-recorded, already-verified session
# 28/29 per-cycle results, NOT live recomputation - a deliberate scope
# cut for the time limit, stated plainly in the UI itself below, not
# glossed over. Every number shown comes straight from
# `data/processed/predictions/streaming_dt_{NASA_B0018,MIT_b3c35}.csv`
# (session 28/29's own saved output) - this function only paces the
# display on a timer so it LOOKS live; it computes nothing new.
# --------------------------------------------------------------------------

_SHOWCASE_BATTERIES = {
    "NASA/B0018 — the clean win": ("NASA", "B0018", "streaming_dt_NASA_B0018.csv"),
    "MIT/b3c35 — the hard case": ("MIT", "b3c35", "streaming_dt_MIT_b3c35.csv"),
}


def _showcase_verdict(dataset: str, df: pd.DataFrame) -> str:
    """Every figure quoted here is computed live from the loaded CSV
    (session 28/29's real recorded numbers), not hardcoded - so this
    verdict can never silently drift from the data it's describing."""
    raw_mae = df["raw_abs_err"].mean()
    corr_mae = df["corrected_abs_err"].mean()
    coverage = df["covered"].mean(skipna=True) * 100
    final = df.iloc[-1]
    if dataset == "NASA":
        return (
            f"✅ **Clean win**: online correction cut mean absolute error from "
            f"**{raw_mae:.2f}pp** (frozen pipeline) to **{corr_mae:.2f}pp** (digital twin) "
            f"over the whole stream. Final-cycle error: {final['raw_abs_err']:.2f}pp (frozen) "
            f"vs. {final['corrected_abs_err']:.2f}pp (twin). Empirical ACI coverage: "
            f"{coverage:.1f}% (target 90%). — Session 28/29, DEVELOPMENT_LOG.md."
        )
    max_hw, old_max_hw = df["half_width"].max(), df["old_half_width"].max()
    return (
        f"⚠️ **Hard case — reported honestly, not softened**: accuracy improved sharply "
        f"(MAE **{raw_mae:.2f}pp → {corr_mae:.2f}pp**), but the ACI conformal interval "
        f"itself got **more volatile, not smoother**, than the original sliding-window "
        f"mechanism it replaced (max half-width **{max_hw:.2f}pp vs. {old_max_hw:.2f}pp**). "
        f"Empirical coverage: {coverage:.1f}% (target 90%). — Session 29's own documented "
        f"limitation, unchanged here."
    )


def _showcase_gauge(row: pd.Series, battery_id: str, last_cycle: int) -> go.Figure:
    # Priority 2 fix (session 31): the SOH number was previously rendered
    # with no explicit font color against a transparent paper_bgcolor -
    # it inherited whatever dark text Plotly defaults to, which nearly
    # vanished against a dark app background. Fixed by giving the card
    # its OWN fixed opaque background (not transparent, so contrast is
    # guaranteed regardless of the surrounding Streamlit theme) plus an
    # explicit bright/white number color.
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(row["corrected_pred"]),
        number={"suffix": "%", "valueformat": ".1f", "font": {"color": "#ffffff", "size": 48}},
        title={"text": f"{battery_id} — Digital Twin SOH — cycle {int(row['cycle_idx'])} of {last_cycle}",
               "font": {"color": "#ffffff", "size": 16}},
        gauge={
            "axis": {"range": [0, 105], "tickfont": {"color": "#ffffff"}},
            "bar": {"color": "#4ea1e8"},
            "bgcolor": "#1a1a2e",
            "steps": [
                {"range": [0, 50], "color": "#7a2e2e"},
                {"range": [50, 80], "color": "#7a6a1e"},
                {"range": [80, 105], "color": "#2e6a3e"},
            ],
            "threshold": {"line": {"color": "#ff6b6b", "width": 4}, "value": float(row["true_soh"])},
        },
    ))
    fig.update_layout(height=280, margin=dict(l=30, r=30, t=60, b=10),
                       font=dict(color="#ffffff"), paper_bgcolor="#1a1a2e")
    return fig


def _showcase_trend(seen: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    band_x = pd.concat([seen["cycle_idx"], seen["cycle_idx"][::-1]])
    band_y = pd.concat([seen["corrected_pred"] + seen["half_width"],
                         (seen["corrected_pred"] - seen["half_width"])[::-1]])
    fig.add_trace(go.Scatter(x=band_x, y=band_y, fill="toself", fillcolor="rgba(33,102,172,0.15)",
                              line=dict(width=0), name="ACI conformal band"))
    fig.add_trace(go.Scatter(x=seen["cycle_idx"], y=seen["true_soh"], name="true SOH",
                              line=dict(color="#1a1a2e", dash="dot", width=1.5)))
    fig.add_trace(go.Scatter(x=seen["cycle_idx"], y=seen["raw_pred"], name="frozen pipeline (raw)",
                              line=dict(color="#999999", width=1.5)))
    fig.add_trace(go.Scatter(x=seen["cycle_idx"], y=seen["corrected_pred"], name="digital twin",
                              line=dict(color="#2166ac", width=2.5)))
    anomalies = seen[seen["anomaly"].astype(bool)]
    if len(anomalies):
        fig.add_trace(go.Scatter(x=anomalies["cycle_idx"], y=anomalies["corrected_pred"], mode="markers",
                                  marker=dict(color="#d62728", symbol="x", size=9), name="anomaly flagged"))
    fig.update_layout(height=360, xaxis_title="cycle", yaxis_title="SOH (%)",
                       margin=dict(l=30, r=20, t=20, b=40),
                       legend=dict(orientation="h", yanchor="bottom", y=1.02),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f7f7fb")
    return fig


def render_showcase_tab():
    st.markdown("## 🔋 Digital Twin Showcase")
    st.warning(
        "🎬 **This is a REPLAY of already-recorded, already-verified results (sessions "
        "28-29), not live recomputation.** Every number below is read straight from the "
        "per-cycle CSVs those sessions produced and independently verified - this view only "
        "paces the display on a timer so it visually looks live. See DEVELOPMENT_LOG.md "
        "sessions 28/29 for exactly how the online correction and conformal interval were "
        "computed, including the honest limitations. For genuinely live inference on any "
        "cycle you pick, use the 🔮 Prediction or 🌊 Streaming Digital Twin tabs."
    )

    choice = st.radio("Battery", list(_SHOWCASE_BATTERIES.keys()), horizontal=True, key="showcase_choice")
    dataset, battery_id, fname = _SHOWCASE_BATTERIES[choice]
    df = load_streaming_replay(fname)
    last_cycle = int(df["cycle_idx"].iloc[-1])

    st.info(_showcase_verdict(dataset, df))

    # Stride long streams (b3c35 has 1091 recorded cycles) down to ~150
    # animation frames for a snappy replay - still exclusively real
    # recorded rows, just not literally every single one animated frame
    # by frame; the trend chart still accumulates every strided point.
    stride = max(1, len(df) // 150)
    play_df = df.iloc[::stride]
    if play_df["cycle_idx"].iloc[-1] != last_cycle:
        play_df = pd.concat([play_df, df.iloc[[-1]]])
    play_df = play_df.reset_index(drop=True)

    col_speed, col_play = st.columns([3, 1])
    with col_speed:
        speed = st.slider("Replay speed (seconds/frame)", 0.01, 0.15, 0.03, step=0.01, key="showcase_speed")
    with col_play:
        st.write("")
        play = st.button("▶ Play replay", key="showcase_play", use_container_width=True)

    gauge_ph = st.empty()
    numbers_ph = st.empty()
    chart_ph = st.empty()

    def draw(i: int):
        row = play_df.iloc[i]
        seen = play_df.iloc[:i + 1]
        gauge_ph.plotly_chart(_showcase_gauge(row, battery_id, last_cycle), use_container_width=True)
        c1, c2, c3 = numbers_ph.columns(3)
        c1.metric("Frozen pipeline prediction", f"{row['raw_pred']:.1f}%")
        c2.metric("Digital-Twin prediction", f"{row['corrected_pred']:.1f}%")
        c3.metric("True SOH", f"{row['true_soh']:.1f}%")
        chart_ph.plotly_chart(_showcase_trend(seen), use_container_width=True)

    if play:
        for i in range(len(play_df)):
            draw(i)
            time.sleep(speed)
    else:
        draw(len(play_df) - 1)


# --------------------------------------------------------------------------
# Streaming Digital Twin tab (session 28) - a NEW mode, additive alongside
# the one-shot Prediction tab above (which it does not replace or modify).
# --------------------------------------------------------------------------

_STREAM_TEST_BATTERIES = {
    "NASA/B0018": ("NASA", "B0018"), "MIT/b1c4": ("MIT", "b1c4"),
    "MIT/b2c24": ("MIT", "b2c24"), "MIT/b3c0": ("MIT", "b3c0"),
    "MIT/b3c35": ("MIT", "b3c35"), "MIT/b4c38": ("MIT", "b4c38"),
}


def render_streaming_twin_tab(res: dict):
    st.caption(
        "🔬 **Simulation-stage digital twin** - not live hardware. This replays an "
        "already-recorded NASA/MIT TEST battery's cycles one at a time (from the exact "
        "same raw data every other tab uses) with a short artificial delay, to emulate "
        "data streaming in live. Unlike the 🔮 Prediction tab (a fresh, independent "
        "full-pipeline rerun for whichever single cycle you pick - no memory between "
        "cycles), this mode keeps a small **online-learning corrector** that updates, "
        "cycle by cycle, using ONLY cycles already streamed in so far - a genuine "
        "incremental model, not a lookup table replaying precomputed numbers. "
        "**What's frozen**: the XGBoost-fusion SOH model, the ICA fusion encoder, the "
        "RUL model, and the anomaly detector - none of these are retrained here. "
        "**What updates online**: a lightweight residual-correction term (see "
        "`src/digital_twin_streaming.py` and DEVELOPMENT_LOG.md session 28 for exactly "
        "how, and an honest report of whether it actually helps)."
    )

    choice = st.selectbox(
        "Battery to stream (restricted to this project's 6 held-out TEST batteries - "
        "genuinely unseen by every frozen model here, for an honest demo)",
        list(_STREAM_TEST_BATTERIES.keys()), key="stream_battery_choice",
    )
    dataset, battery_id = _STREAM_TEST_BATTERIES[choice]

    col_a, col_b = st.columns(2)
    with col_a:
        max_cycles = st.slider("Max cycles to stream (after the first 5, which load instantly "
                                "as already-available warm-start history)", 20, 300, 80, step=10)
    with col_b:
        delay = st.slider("Simulated per-cycle arrival delay (seconds)", 0.0, 0.2, 0.02, step=0.01)

    if st.button("▶ Start streaming simulation", key="stream_start"):
        # Same graceful-degradation pattern as the sidebar's "Browse
        # existing battery" path (session 12): check availability FIRST
        # with a clear, specific reason, rather than attempting to load
        # and surfacing a raw scipy/h5py exception to the user.
        dataset_available = {"NASA": nasa_data_available, "MIT": mit_data_available}[dataset]()
        if not dataset_available:
            st.warning(
                f"⚠️ {dataset}'s raw data isn't available in this environment - the "
                f"NASA/CALCE/MIT research datasets aren't bundled with this app (size + "
                f"third-party redistribution terms), so this replay-free streaming demo only "
                f"works where they've been downloaded locally (see README). Try the 🎬 "
                f"Showcase tab instead - it replays session 28/29's already-recorded results "
                f"for these same batteries and needs no raw data at all."
            )
            return
        try:
            cycles = load_battery_cycles(dataset, battery_id)
        except (FileNotFoundError, OSError, KeyError) as e:
            st.error(f"Could not load {dataset}/{battery_id}'s raw data: {e}. It may be "
                     f"missing or incomplete locally - try a different battery, or use the "
                     f"🎬 Showcase tab instead (no raw data needed).")
            return

        hi_df = pd.read_parquet(PROC_DIR / "hi_table.parquet")
        soh_lookup = hi_df[(hi_df["dataset"] == dataset) & (hi_df["battery_id"] == battery_id)] \
            .set_index("cycle_idx")["SOH"].to_dict()

        twin = StreamingDigitalTwin(
            res["xgb_fusion"], res["encoder"], res["ocsvm"], res["ocsvm_scaler"],
            res["ocsvm_feature_cols"], res["bfa_selected"], res["train_medians"],
            res["norm_stats"], res["constants"]["soh_conformal_half_width"],
        )

        n_stream = min(max_cycles, max(0, len(cycles) - 5))
        stream_cycles = cycles[:5 + n_stream]

        chart_placeholder = st.empty()
        status_placeholder = st.empty()
        rows = []
        for i, c in enumerate(stream_cycles):
            true_soh = soh_lookup.get(c["cycle_idx"])
            result = twin.step(c, true_soh=true_soh)
            if "error" in result:
                continue
            rows.append(result)

            if i >= 5:  # only the genuinely "streamed" portion gets the simulated-arrival delay - the first 5 are already-available warm-start history, shown instantly
                time.sleep(delay)

            df_so_far = pd.DataFrame(rows)
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(df_so_far["cycle_idx"], df_so_far["true_soh"], "k--", linewidth=1, label="true SOH")
            ax.plot(df_so_far["cycle_idx"], df_so_far["raw_pred"], color="tab:gray", alpha=0.7,
                    label="frozen pipeline (raw, no correction)")
            ax.plot(df_so_far["cycle_idx"], df_so_far["corrected_pred"], color="tab:blue",
                    label="online-corrected twin")
            ax.fill_between(df_so_far["cycle_idx"],
                             df_so_far["corrected_pred"] - df_so_far["half_width"],
                             df_so_far["corrected_pred"] + df_so_far["half_width"],
                             color="tab:blue", alpha=0.15, label="online conformal band")
            anomalies = df_so_far[df_so_far["anomaly"]]
            if len(anomalies):
                ax.scatter(anomalies["cycle_idx"], anomalies["corrected_pred"], color="red",
                           marker="x", s=70, zorder=5, label="anomaly flagged")
            ax.set_xlabel("cycle"); ax.set_ylabel("SOH (%)")
            ax.set_title(f"{dataset}/{battery_id} — streaming twin "
                         f"(cycle {c['cycle_idx']} of {stream_cycles[-1]['cycle_idx']})")
            ax.legend(fontsize=8, loc="lower left")
            chart_placeholder.pyplot(fig)
            plt.close(fig)

            last = rows[-1]
            bits = [f"cycle **{last['cycle_idx']}**", f"raw={last['raw_pred']:.1f}%",
                    f"corrected={last['corrected_pred']:.1f}%", f"±{last['half_width']:.1f}pp",
                    f"revealed history: {last['n_revealed_so_far']} cycles"]
            if last["anomaly"]:
                bits.append("🚨 **ANOMALY FLAGGED**")
            status_placeholder.markdown(" | ".join(bits))

        df_final = pd.DataFrame(rows)
        stream_only = df_final.iloc[5:] if len(df_final) > 5 else df_final
        stream_only = stream_only.dropna(subset=["true_soh"])
        raw_mae = (stream_only["raw_pred"] - stream_only["true_soh"]).abs().mean()
        corrected_mae = (stream_only["corrected_pred"] - stream_only["true_soh"]).abs().mean()

        st.markdown("---")
        st.subheader("Honest summary: did online updating actually help?")
        c1, c2, c3 = st.columns(3)
        c1.metric("Frozen-pipeline MAE (whole stream)", f"{raw_mae:.2f} pp")
        c2.metric("Online-corrected MAE (whole stream)", f"{corrected_mae:.2f} pp",
                  delta=f"{corrected_mae - raw_mae:+.2f} pp vs. frozen", delta_color="inverse")
        final_row = df_final.iloc[-1]
        c3.metric("Final cycle: corrected vs. true",
                  f"{final_row['corrected_pred']:.1f}% vs {final_row['true_soh']:.1f}%")
        st.caption(
            "Compare the final streamed cycle's numbers above against this SAME battery/cycle "
            "in the 🔮 Prediction tab (the frozen one-shot pipeline, computed independently) to "
            "see how closely the online-updating twin converges to it. Whether online updating "
            "helps, and by how much, is reported honestly (both directions, not cherry-picked) "
            "in DEVELOPMENT_LOG.md session 28 - it is NOT assumed to always improve on the "
            "frozen pipeline."
        )
        st.caption(
            "⚠️ Reminder: this is a SIMULATION of streaming, replaying already-recorded test "
            "data with an artificial delay - not a connection to live hardware or a real BMS."
        )


def main():
    st.title("🔋 Battery Digital Twin Dashboard")
    st.caption("Fusion-enabled ensemble (XGBoost+fusion / Stacking-Ridge+fusion) for SOH, "
               "joint-adaptive model for RUL.")
    render_about_section()

    with st.sidebar:
        st.header("Battery selection")
        mode = st.radio("Data source", ["Browse existing battery", "Upload your own cycle data"])

        cycles = None
        dataset = battery_id = None

        if mode == "Browse existing battery":
            dataset = st.selectbox("Dataset", ["NASA", "MIT", "CALCE"])
            dataset_available = {
                "NASA": nasa_data_available, "MIT": mit_data_available, "CALCE": calce_data_available,
            }[dataset]()

            if not dataset_available:
                st.warning(
                    f"⚠️ {dataset}'s raw data isn't available in this environment - the "
                    f"NASA/CALCE/MIT research datasets aren't bundled with this app (size + "
                    f"third-party redistribution terms), so pre-loaded browsing only works "
                    f"where they've been downloaded locally (see README). Try a different "
                    f"dataset, or use 'Upload your own cycle data' below - it works fully "
                    f"without any of them."
                )
            else:
                if dataset == "NASA":
                    battery_id = st.selectbox("Battery", NASA_CELLS)
                elif dataset == "MIT":
                    battery_id = st.selectbox("Battery", sorted(get_mit_subset().keys()))
                else:
                    battery_id = st.selectbox("Battery", CALCE_CELLS)
                    st.caption("⚠️ CALCE has no temperature channel and is a different cell "
                               "chemistry/format than NASA+MIT training data - expect the "
                               "out-of-domain warning to trigger.")
                try:
                    cycles = load_battery_cycles(dataset, battery_id)
                except (FileNotFoundError, OSError, KeyError) as e:
                    st.error(f"Could not load {dataset}/{battery_id}'s raw data: {e}. "
                             f"It may be missing or incomplete locally - try a different "
                             f"battery, or use 'Upload your own cycle data' below.")
        else:
            st.caption("CSV columns required: `cycle_idx, phase (charge/discharge), "
                       "time_s, voltage_v, current_a`. Optional: `temperature_c`.")
            uploaded = st.file_uploader("Upload cycle data CSV", type="csv")
            if uploaded is not None:
                try:
                    df = pd.read_csv(uploaded)
                    cycles = parse_uploaded_csv(df)
                    dataset, battery_id = "Uploaded", uploaded.name
                    st.success(f"Parsed {len(cycles)} usable cycles.")
                except Exception as e:
                    st.error(f"Could not parse upload: {e}")

        if cycles:
            idx = st.slider("Cycle", 1, len(cycles), value=len(cycles),
                             help="Defaults to the most recent cycle (current battery state).")
            selected_cycle = cycles[idx - 1]
        else:
            selected_cycle = None

    ctx = None
    true_soh = true_rul = None

    if selected_cycle is None:
        st.info("👈 Select a battery (or upload a CSV) in the sidebar to begin.")
    else:
        st.header(f"{battery_id} — cycle {selected_cycle['cycle_idx']} of {len(cycles)}")

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(selected_cycle["discharge"]["t"], selected_cycle["discharge"]["V"])
        ax.set_xlabel("time (s)")
        ax.set_ylabel("voltage (V)")
        ax.set_title("Discharge voltage curve for this cycle")
        st.pyplot(fig)
        st.caption("Raw voltage-vs-time trace for the selected cycle's discharge phase - the "
                   "signal every prediction below is ultimately derived from.")

        res = get_resources()
        with st.spinner("Running fusion ensemble + joint-adaptive model + SHAP explanation..."):
            ctx = predict_and_explain(selected_cycle, res)

        if "error" in ctx:
            st.error(ctx["error"])
            ctx = None
        else:
            if dataset in ("NASA", "MIT", "CALCE"):
                hi_df = pd.read_parquet(PROC_DIR / "hi_table.parquet")
                row = hi_df[(hi_df["dataset"] == dataset) & (hi_df["battery_id"] == battery_id)
                            & (hi_df["cycle_idx"] == selected_cycle["cycle_idx"])]
                if not row.empty:
                    true_soh = round(float(row.iloc[0]["SOH"]), 1)
                    true_rul = int(row.iloc[0]["RUL"])

            if ctx["out_of_domain"]:
                st.error(
                    "🚨 **OUT-OF-DOMAIN — conformal interval reliability NOT guaranteed.** 🚨\n\n"
                    "Reasons: " + "; ".join(ctx["domain_reasons"]) + ".\n\n"
                    "The CALCE zero-retrain evaluation found the exact failure mode this warning "
                    "exists to prevent: the conformal interval shown in the Prediction tab looked "
                    "**identically confident** in-domain and out-of-domain (same fixed width, "
                    "±2.37 SOH points either way), while actual empirical coverage collapsed from "
                    "95.6% (NASA/MIT) to just 6.1% (CALCE). Treat any interval below as decorative, "
                    "not a real confidence guarantee, for this battery."
                )

    tab_showcase, tab_prediction, tab_explain, tab_report, tab_stream, tab_validation, tab_archive = st.tabs(
        ["🎬 Showcase", "🔮 Prediction", "🔍 Explainability", "📝 Health Report",
         "🌊 Streaming Digital Twin", "🧪 Model Validation", "📁 Full Results Archive"]
    )
    with tab_showcase:
        render_showcase_tab()
    # Priority 3 (session 31): the 3 tabs below no longer repeat their own
    # "select a battery" placeholder - the single shared message at line
    # ~843 (`if selected_cycle is None: st.info(...)`), which renders
    # above the tabs regardless of which one is active, already covers
    # this. Previously each tab additionally showed its own near-
    # identical copy, stacking 2 duplicate messages on top of each other
    # for every one of these 3 tabs whenever no battery was selected
    # (including the "data unavailable" case Priority 1 fixed above).
    with tab_prediction:
        if ctx is not None:
            render_prediction_tab(ctx, true_soh, true_rul)
    with tab_explain:
        if ctx is not None:
            render_explainability_tab(ctx)
    with tab_report:
        if ctx is not None:
            render_health_report_tab(ctx, dataset, battery_id)
    with tab_stream:
        render_streaming_twin_tab(get_resources())
    with tab_validation:
        render_evaluation_protocol_section()
    with tab_archive:
        render_full_results_archive_tab()


if __name__ == "__main__":
    main()
