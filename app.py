"""
Battery Digital Twin Dashboard (Streamlit).

Uses the FUSION-enabled ensemble (XGBoost+fusion, Stacking-Ridge+fusion)
for SOH, the joint-adaptive model for RUL - NOT the physics-informed
model variants, per instruction. Every prediction is computed LIVE via
src/live_inference.py (no lookup tables) - "a full pipeline rerun per
new upload is fine" per instruction, so this app does not attempt any
incremental/cached meta-learner updating.

Functional over polished: plain Streamlit widgets, one matplotlib plot,
no custom theming.
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


def render_context(ctx: dict, dataset: str, battery_id: str, true_soh=None, true_rul=None):
    if "error" in ctx:
        st.error(ctx["error"])
        return

    if ctx["out_of_domain"]:
        st.warning(
            "**OUT-OF-DOMAIN — conformal interval reliability not guaranteed.**\n\n"
            "Reasons: " + "; ".join(ctx["domain_reasons"]) + ".\n\n"
            "Tonight's CALCE zero-retrain evaluation found the exact failure mode this "
            "warning exists to prevent: the conformal interval below looked **identically "
            "confident** in-domain and out-of-domain (same fixed width, ±2.37 SOH points "
            "either way), while actual empirical coverage collapsed from 95.6% (NASA/MIT) "
            "to just 6.1% (CALCE). The interval is shown below for reference, but treat it "
            "as decorative, not a real confidence guarantee, for this battery."
        )

    col1, col2 = st.columns(2)
    with col1:
        delta = None if true_soh is None else round(ctx["soh_pred"] - true_soh, 1)
        st.metric("Predicted SOH", f"{ctx['soh_pred']}%",
                   delta=(f"{delta:+.1f} vs. true {true_soh}%" if delta is not None else None))
        st.caption(f"90% conformal interval: {ctx['soh_conformal_lo']}% – {ctx['soh_conformal_hi']}%"
                   + (" ⚠️ unreliable (see warning above)" if ctx["out_of_domain"] else ""))
    with col2:
        delta = None if true_rul is None else round(ctx["rul_pred"] - true_rul)
        st.metric("Predicted RUL", f"{ctx['rul_pred']} cycles",
                   delta=(f"{delta:+d} vs. true {true_rul}" if delta is not None else None))
        st.caption(f"90% conformal interval: {ctx['rul_conformal_lo']} – {ctx['rul_conformal_hi']} cycles"
                   + (" ⚠️ unreliable (see warning above)" if ctx["out_of_domain"] else ""))
        st.caption("_RUL comes from the Phase 4 joint-adaptive model, the only trained RUL "
                   "predictor in this project - the fusion ensemble itself is SOH-only._")

    if ctx["anomaly_flag"]:
        st.error("🚨 **Anomaly flag**: the One-Class SVM considers this cycle's feature "
                  "vector unlike the NASA+MIT training distribution.")
    else:
        st.success("✅ No anomaly flagged (One-Class SVM).")

    st.subheader("Why this prediction? (per-instance SHAP)")
    feat_df = pd.DataFrame(ctx["top_features"])
    st.dataframe(feat_df[["feature", "description", "shap_value"]], hide_index=True,
                 width="stretch")
    if ctx["voltage_region"]:
        vr = ctx["voltage_region"]
        st.write(f"**Voltage region driving this prediction (VLSTM DeepSHAP):** "
                 f"{vr['v_lo']}V – {vr['v_hi']}V "
                 f"(~{round(vr['frac_of_attribution']*100)}% of attribution)")
    elif ctx["voltage_region_error"]:
        st.caption(f"(voltage-region explanation unavailable: {ctx['voltage_region_error']})")

    st.subheader("Plain-English health report")
    context_for_llm = dict(ctx)
    context_for_llm["battery_id"] = battery_id
    context_for_llm["dataset"] = dataset
    prompt = build_prompt(context_for_llm)
    report = call_llm(prompt)
    if report.startswith("NO_API_KEY") or report.startswith("API_ERROR"):
        reason = ("No `GEMINI_API_KEY` configured" if report.startswith("NO_API_KEY")
                   else f"Gemini API call failed ({report})")
        st.info(f"{reason} - showing the structured data the LLM would have used instead "
                f"of a generated narrative:")
        st.json({
            "soh_pred": ctx["soh_pred"], "soh_interval": [ctx["soh_conformal_lo"], ctx["soh_conformal_hi"]],
            "rul_pred": ctx["rul_pred"], "rul_interval": [ctx["rul_conformal_lo"], ctx["rul_conformal_hi"]],
            "top_features": ctx["top_features"], "voltage_region": ctx["voltage_region"],
        })
    else:
        st.write(report)


def main():
    st.title("🔋 Battery Digital Twin Dashboard")
    st.caption("Fusion-enabled ensemble (XGBoost+fusion / Stacking-Ridge+fusion) for SOH, "
               "joint-adaptive model for RUL. Every prediction below is computed live from "
               "trained weights - no retraining happens in this app.")

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
        st.info("Select a battery (or upload a CSV) in the sidebar to begin.")
        return

    st.header(f"{battery_id} — cycle {selected_cycle['cycle_idx']} of {len(cycles)}")

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(selected_cycle["discharge"]["t"], selected_cycle["discharge"]["V"])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("voltage (V)")
    ax.set_title("Discharge voltage curve for this cycle")
    st.pyplot(fig)

    res = get_resources()
    with st.spinner("Running fusion ensemble + joint-adaptive model + SHAP..."):
        ctx = predict_and_explain(selected_cycle, res)

    true_soh = true_rul = None
    if dataset in ("NASA", "MIT", "CALCE"):
        hi_df = pd.read_parquet(PROC_DIR / "hi_table.parquet")
        row = hi_df[(hi_df["dataset"] == dataset) & (hi_df["battery_id"] == battery_id)
                    & (hi_df["cycle_idx"] == selected_cycle["cycle_idx"])]
        if not row.empty:
            true_soh = round(float(row.iloc[0]["SOH"]), 1)
            true_rul = int(row.iloc[0]["RUL"])

    render_context(ctx, dataset, battery_id, true_soh, true_rul)


if __name__ == "__main__":
    main()
