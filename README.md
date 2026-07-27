# battery-prognostics

A battery cycle-life / capacity-fade prognostics pipeline built on three
public li-ion cycling datasets — NASA PCoE, CALCE (CS2), and MIT/Stanford/TRI
(Severson et al., Nature Energy 2019) — with a live Streamlit dashboard on
top.

## Pipeline

1. **Health Indicator extraction** — 16 candidate HIs per cycle (voltage-curve
   shape, ICA/DV/DC features, etc.) from raw charge/discharge data.
2. **BFA feature selection** — a Butterfly-inspired metaheuristic selects 7 of
   the 16 HIs.
3. **4 base learners** — XGBoost, VLSTM, CNN-LSTM, PiFormer — trained to
   predict SOH.
4. **Stacking ensemble** — Ridge and XGBoost meta-learners over the 4 base
   learners.
5. **Fusion module** — an embedding of the ICA/DV/DC curves fused into the
   XGBoost/stacking feature set (`train_fusion_encoder.py`,
   `train_xgboost_fusion.py`, `train_ensemble_fusion.py`) — this fusion-enabled
   ensemble is what the dashboard actually serves.
6. **Joint SOH+RUL adaptive-loss model** — a separate multi-task model
   (`train_joint_adaptive.py`) that also predicts Remaining Useful Life; it is
   the only trained RUL predictor in the project.
7. **SHAP explainability** — TreeSHAP (XGBoost/fusion) and DeepSHAP (the
   sequence models), including per-instance voltage-region localization.
8. **Split-conformal prediction** — calibrated 90% intervals for both SOH and
   RUL via MAPIE.

Consolidated final numbers for every stage are in **`FINAL_SUMMARY.md`**; the
full chronological narrative — every assumption, every bug found and fixed,
every intermediate/pre-fix number — is in **`DEVELOPMENT_LOG.md`**.

## Structure

```
data/
  raw/
    nasa/    - NASA PCoE .mat files (B0005, B0006, B0007, B0018)
    calce/   - CALCE CS2 cell .xlsx sessions (CS2_35, CS2_36, CS2_37)
    mit/     - MIT/BEEP quickstart example + full-dataset batch .mat files
  processed/ - HI tables, BFA selection, fusion embeddings, predictions, etc.
src/         - pipeline stages (loaders, feature extraction, training,
               SHAP, conformal calibration) and the dashboard's supporting
               modules (data_adapters.py, live_inference.py,
               generate_health_report.py)
notebooks/   - exploratory analysis
outputs/     - figures, reports
models/      - trained model artifacts
app.py       - Streamlit dashboard
.venv/       - Python 3.14 virtualenv
requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
```

## Running the loaders

```bash
python src/load_nasa.py
python src/load_calce.py
python src/load_mit.py
```

See `STATUS.md` for current dataset status, sources, and known caveats.

## Running the dashboard

```bash
streamlit run app.py
```

Lets you pick a battery (NASA/MIT/CALCE) or upload your own cycle-data CSV,
and get a live SOH/RUL prediction with conformal intervals, SHAP-based
explanations, an LLM-generated plain-English health report (Gemini, falling
back to Groq), an out-of-domain warning, and the evaluation-protocol
experiments used to validate the pipeline.
