"""
Battery Digital Twin Dashboard (Streamlit).

Uses the FUSION-enabled ensemble (XGBoost+fusion, Stacking-Ridge+fusion)
for SOH, the joint-adaptive model for RUL - NOT the physics-informed
model variants, per instruction. Every prediction is computed LIVE via
src/live_inference.py (no lookup tables) - "a full pipeline rerun per
new upload is fine" per instruction, so this app does not attempt any
incremental/cached meta-learner updating.

UI is organized into 5 tabs (Prediction / Explainability / Health Report
/ Model Validation / Full Results Archive) purely for presentation - no
change to any underlying computation, model, or data versus the
single-page layout this replaced. Functional over polished: plain
Streamlit widgets, one matplotlib plot, no custom theming beyond native
`st.metric`/colored-markdown/status-container idioms.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from data_adapters import (
    iterate_nasa_cycles, iterate_mit_cycles, iterate_calce_cycles,
    nasa_data_available, mit_data_available, calce_data_available,
)
from live_inference import load_resources, predict_and_explain
from generate_health_report import build_prompt, call_llm

ROOT = Path(__file__).resolve().parent
PROC_DIR = ROOT / "data" / "processed"
PRED_DIR = PROC_DIR / "predictions"
OUT_DIR = ROOT / "outputs"

NASA_CELLS = ["B0005", "B0006", "B0007", "B0018"]
CALCE_CELLS = ["CS2_35", "CS2_36", "CS2_37"]

st.set_page_config(page_title="Battery Digital Twin", layout="wide")


@st.cache_resource
def get_resources():
    return load_resources()


@st.cache_data(show_spinner=False)
def get_mit_subset():
    with open(PROC_DIR / "mit_subset.json") as f:
        return {e["global_id"]: e for e in json.load(f)}


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

    with st.expander("3️⃣ Base Learner Training"):
        _safe_image(OUT_DIR / "phase2_deep_model_training_curves.png",
                     "Train/validation loss curves for the three deep sequence models "
                     "(VLSTM, CNN-LSTM, PiFormer).")
        _safe_table(PRED_DIR / "ensemble_comparison.csv",
                    "RMSE/MAE/R² for all 4 base learners plus both stacking meta-learners "
                    "on the held-out test set - XGBoost is the strongest individual model.")

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

    with st.expander("7️⃣ Joint SOH+RUL Ablation"):
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
                    "Final SOH/RUL RMSE, MAE, and R² for all 4 loss-weighting variants.")

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

    with st.expander("9️⃣ Split-Conformal Prediction"):
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

    with st.expander("1️⃣1️⃣ OC-SVM Anomaly Detector"):
        st.markdown("No standalone metrics CSV for this component - result summarized here "
                    "in text instead of a plot/table that doesn't exist on disk.")
        st.info(
            "**Class-imbalance bug, found and fixed.** The first-pass One-Class SVM was "
            "trained on all fit-split cycles as-is: NASA (~470 cycles) vs. MIT (~14,400 "
            "cycles), a ~30:1 imbalance. Sanity-checking it against its own training data "
            "found **83.9% of NASA's own cycles flagged \"anomalous\"** vs. only 2.2% of "
            "MIT's - meaning a perfectly legitimate NASA battery would trigger the "
            "dashboard's out-of-domain warning almost every time. Fixed by capping each fit "
            "battery to 200 cycles before fitting; NASA's false-flag rate dropped to 24.4% "
            "(MIT stayed ~2.3%) - a large improvement, though not perfectly balanced, and "
            "left documented as a residual limitation rather than claimed as fully solved."
        )

    with st.expander("1️⃣2️⃣ Health Report Examples (LLM-generated)"):
        st.caption("5 saved example reports from `outputs/health_reports_examples.json` - "
                   "generated by the same prompt/LLM chain the Health Report tab uses live, "
                   "shown here as readable text rather than raw JSON.")
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

    st.markdown("**Evaluation-protocol experiments** (early-prediction test, drop-one-branch "
                "ablation, homogeneous-bagging baseline) are already covered in full in the "
                "🧪 **Model Validation** tab - not duplicated here.")


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

    tab_prediction, tab_explain, tab_report, tab_validation, tab_archive = st.tabs(
        ["🔮 Prediction", "🔍 Explainability", "📝 Health Report", "🧪 Model Validation",
         "📁 Full Results Archive"]
    )
    with tab_prediction:
        if ctx is not None:
            render_prediction_tab(ctx, true_soh, true_rul)
        else:
            st.info("👈 Select a battery (or upload a CSV) in the sidebar to see predictions.")
    with tab_explain:
        if ctx is not None:
            render_explainability_tab(ctx)
        else:
            st.info("👈 Select a battery (or upload a CSV) in the sidebar first.")
    with tab_report:
        if ctx is not None:
            render_health_report_tab(ctx, dataset, battery_id)
        else:
            st.info("👈 Select a battery (or upload a CSV) in the sidebar first.")
    with tab_validation:
        render_evaluation_protocol_section()
    with tab_archive:
        render_full_results_archive_tab()


if __name__ == "__main__":
    main()
