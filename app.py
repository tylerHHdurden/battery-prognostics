"""
Battery Digital Twin Dashboard (Streamlit).

Uses the FUSION-enabled ensemble (XGBoost+fusion, Stacking-Ridge+fusion)
for SOH, the joint-adaptive model for RUL - NOT the physics-informed
model variants, per instruction. Every prediction is computed LIVE via
src/live_inference.py (no lookup tables) - "a full pipeline rerun per
new upload is fine" per instruction, so this app does not attempt any
incremental/cached meta-learner updating.

UI is organized into 4 tabs (Prediction / Explainability / Health Report
/ Model Validation) purely for presentation - no change to any
underlying computation, model, or data versus the single-page layout
this replaced. Functional over polished: plain Streamlit widgets, one
matplotlib plot, no custom theming beyond native `st.metric`/colored-
markdown/status-container idioms.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from data_adapters import iterate_nasa_cycles, iterate_mit_cycles, iterate_calce_cycles
from live_inference import load_resources, predict_and_explain
from generate_health_report import build_prompt, call_llm

ROOT = Path(__file__).resolve().parent
PROC_DIR = ROOT / "data" / "processed"
PRED_DIR = PROC_DIR / "predictions"

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
            if dataset == "NASA":
                battery_id = st.selectbox("Battery", NASA_CELLS)
            elif dataset == "MIT":
                battery_id = st.selectbox("Battery", sorted(get_mit_subset().keys()))
            else:
                battery_id = st.selectbox("Battery", CALCE_CELLS)
                st.caption("⚠️ CALCE has no temperature channel and is a different cell "
                           "chemistry/format than NASA+MIT training data - expect the "
                           "out-of-domain warning to trigger.")
            cycles = load_battery_cycles(dataset, battery_id)
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

    if selected_cycle is None:
        st.info("👈 Select a battery (or upload a CSV) in the sidebar to begin.")
        return

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
        return

    true_soh = true_rul = None
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

    tab_prediction, tab_explain, tab_report, tab_validation = st.tabs(
        ["🔮 Prediction", "🔍 Explainability", "📝 Health Report", "🧪 Model Validation"]
    )
    with tab_prediction:
        render_prediction_tab(ctx, true_soh, true_rul)
    with tab_explain:
        render_explainability_tab(ctx)
    with tab_report:
        render_health_report_tab(ctx, dataset, battery_id)
    with tab_validation:
        render_evaluation_protocol_section()


if __name__ == "__main__":
    main()
