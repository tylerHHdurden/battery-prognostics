# CellSense

CellSense is a research prototype that estimates the state of health (SOH) of a lithium-ion cell from its
charge/discharge cycling data, shows a 90% interval and an explanation, and tells you when a battery looks
unlike the ones it was built on.

**Purpose:** screen cycling data from lab-tested Li-ion cells and estimate SOH (discharge capacity relative to the
first cycles, in percent) with an uncertainty interval and a SHAP explanation.

**What it cannot do:** it is not a certified battery passport, not a safety or warranty tool, and not a
predictor for arbitrary cells. On batteries unlike its training data its errors are large and its interval is not
reliable, and its remaining-useful-life (RUL) output fails outside the data it was trained on.

## Try it

- Live app: <https://bat-pro.streamlit.app>
- Locally:

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows; Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Python 3.14 is what the project was developed and deployed on. Pick a built-in battery and cycle, or upload your
own cycle data. The app shows the SOH estimate, the interval, the top SHAP features, a familiarity (trust) check,
and an exportable research passport (JSON/PDF) that is labelled as not certified.

## Headline results (honest version)

All numbers are from `PAPER_RESULTS.md` (Section 1, 5 seeds, mean +/- std; SOH is on a 0-100 scale; errors are in
SOH percentage points). Status words follow that file: **VERIFIED** means the value is unchanged after the
2026-09-30 rerun that fixed a BatteryLife embedding bug.

| Data | R2 | RMSE | MAE | Status |
|---|---|---|---|---|
| In-domain (held-out NASA + MIT test batteries) | 0.978 +/- 0.003 | 0.716 +/- 0.050 | 0.281 +/- 0.014 | VERIFIED |
| CALCE (never trained on) | 0.749 +/- 0.012 | 10.791 +/- 0.265 | 6.256 +/- 0.255 | VERIFIED |
| Oxford (never trained on) | 0.940 +/- 0.030 | 1.639 +/- 0.409 | 1.431 +/- 0.434 | VERIFIED |
| HUST (never trained on) | 0.795 +/- 0.022 | 3.336 +/- 0.178 | 2.643 +/- 0.138 | VERIFIED |
| XJTU (never trained on) | -1.037 +/- 0.216 | 8.572 +/- 0.460 | 6.457 +/- 0.213 | VERIFIED |

- Accuracy is excellent in-domain and drops sharply on other cell types; on XJTU the model is worse than
  predicting the mean (negative R2).
- **RUL fails across domains.** The RUL model was trained only on NASA and MIT cells. On CALCE its R2 is -566.35
  (MAE 421.5 cycles, MAPE 1395.9%), on Oxford -1.31, HUST -0.45, XJTU -78.46 (MAE in cycles: 2520.0, 543.3, 1017.9).
  R2 is not comparable across datasets here, so read MAE/MAPE. The app therefore shows RUL only when the nearest
  known source is NASA or MIT and the battery is not flagged. (VERIFIED)
- **Conformal coverage is an in-domain guarantee only.** The 90% interval covers 97.9% of in-domain cycles but only
  4.3% (CALCE), 5.4% (Oxford), 8.5% (HUST) and 11.0% (XJTU) on other cell types. (VERIFIED)
- **Trust check.** A flag based on a nearest-source likelihood score detects 81.9% of batteries from unseen sources
  (390/476) with 11.0% false alarms on known sources (10/91), measured leave-one-source-out over 16 sources. It is
  weak for the `mich`, `NASA` and `snl` sources (fewer than half of their batteries were flagged) and gives no
  guarantee. Not being flagged does not mean a battery is trustworthy; about 1 in 5 unfamiliar batteries are missed.
- The dataset-aware routing (which SOH model handles which dataset) was selected on held-out data, so it is a
  deployment choice, not an unbiased test result.

## Limits

See `release/KNOWN_LIMITATIONS.md` and `release/MODEL_CARD.md`. In short: trained on two public datasets (NASA and
MIT), cross-dataset transfer is poor for some cell types, intervals do not transfer, RUL does not transfer,
raw data are not redistributed, and this is a research prototype.

## How to cite

See `CITATION.cff` (a draft is in `release/CITATION.cff`; copy it to the repository root when publishing). Please
also cite the original datasets, see `DATA_AVAILABILITY.md`.

## Reproduce

See `REPRODUCIBILITY.md` (methods table, battery-level split confirmation, how to re-run, Python version notes) and
`DATA_AVAILABILITY.md` (where to download each dataset). Quick checks: `python -m pytest tests -q`. Full
narrative, including every negative result: `DEVELOPMENT_LOG.md`.

## Repository layout

```
app.py                     Streamlit app
src/                       pipeline stages, live inference (live_inference.py), passport (battery_passport.py),
                           trust check (trust_report.py), conformal, SHAP, run_* experiment scripts
src/models/                network definitions (VLSTM, ICA encoder, ...)
models/                    trained model files and encoder-provenance sidecars (*.meta.json)
data/raw/                  raw datasets (git-ignored; download per DATA_AVAILABILITY.md)
data/processed/            derived feature tables and embeddings
outputs/                   result CSVs behind every number in PAPER_RESULTS.md
scripts/reproduce_paper.sh re-runs the final-pass tables
tests/                     quick smoke tests
release/                   open-release drafts (licence, citation, model/data cards, checklist)
PAPER_RESULTS.md           final tables and status words
DEVELOPMENT_LOG.md         chronological log of every experiment
REPRODUCIBILITY.md         how to reproduce
DATA_AVAILABILITY.md       data sources and licences
requirements.txt           pinned dependencies (requirements-lock.txt: full freeze)
```

## Licence

Code: MIT, see `LICENSE` (a draft is in `release/LICENSE`; copy it to the repository root when publishing). The
datasets keep their own licences; see `DATA_AVAILABILITY.md`.
