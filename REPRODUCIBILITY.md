# Reproducibility Package

Companion to `PAPER_RESULTS.md` and `DEVELOPMENT_LOG.md`. This file
covers item F of the final pre-submission experiment pass: a table of
every distinct method tried across this project's full history, a
written confirmation of battery-level (not cycle-level) splitting, a
requirements lock file, a script that regenerates every
`PAPER_RESULTS.md` table, and a data-availability statement.

## Methods table

Every method that reached the point of a real, run experiment across
this project's entire history (compiled from `DEVELOPMENT_LOG.md`'s own
section headers and results tables - not recalled from memory without
a source). "Tuning budget" of 0 means this project's own standing
practice: fixed hyperparameters, reused project-wide, no grid/random
search performed for that method (the norm for nearly every XGBoost
variant tried). Seeds are 42 unless noted (this project's own
project-wide default). Verdict uses this project's own established
vocabulary (WIN/LOSS/TIE/SPLIT/PARTIAL/BLOCKED/PROMOTED).

| # | Method | Stage/pass | Hyperparameters (key ones) | Tuning budget | Seed(s) | Verdict |
|---|---|---|---|---|---|---|
| 1 | Binary Firefly Algorithm (BFA) feature selection | Phase 1 / Stage 0 correction | 30 agents x 100 iterations, GroupKFold-CV Ridge wrapper fitness | search procedure itself (30x100=3000 evals) | 42 | PROMOTED (corrected NASA+MIT-only version is canonical) |
| 2 | XGBoost base learner | Phase 2 | n_estimators=500, max_depth=6, lr=0.03, subsample=0.8, colsample=0.8, reg_lambda=1.0 | 0 (fixed, reused everywhere) | 42 | PROMOTED (base of every later XGBoost variant) |
| 3 | VLSTM / CNN-LSTM / PiFormer (5 base learners incl. CNN-BiGRU) | Phase 2 / session 17 | ~40 epoch, patience-8 early stop | 0 | 42 | MIXED - none beat XGBoost-fusion standalone |
| 4 | Stacking ensemble | Phase 3 | linear meta-learner over base-learner OOF preds | 0 | 42 | LOSS (does not beat standalone XGBoost) |
| 5 | Adaptive vs. fixed joint SOH+RUL loss weighting | Phase 4 | log_sigma-clamped uncertainty weighting | 0 | 42 | SPLIT (genuine, reported honestly) |
| 6 | ICA/DV/DC feature fusion embedding | session 3 | 16-dim encoder, no attention | 0 | 42 | PROMOTED (16 fusion dims used by every model since) |
| 7 | Physics-informed monotonicity loss penalty | session 4 | soft penalty term, deep models | 0 | 42 | LOSS (every deep model got worse) |
| 8 | MMD domain adaptation (fusion encoder alignment) | session 13 | kernel mean-embedding alignment loss | 0 | 42 | LOSS (CALCE coverage worse, 4.4% vs 6.1% do-nothing) |
| 9 | Weighted split-conformal (logistic domain classifier) | session 19 | LogisticRegression density-ratio reweighting | 0 | 42 | LOSS (never improved coverage; broke in-domain coverage as side effect) |
| 10 | Adaptive Conformal Inference (ACI) | session 29 | alpha_t clipped to [0.01,0.5] | 0 | 42 | MIXED (genuine long-run improvement, does not fix short-run instability) |
| 11 | Normalized conformal prediction (local-scale GBR) | session 35 Part 4 | secondary GBR predicts sigma(x), train-only | 0 | 42 | PARTIAL (19.3% CALCE coverage, real but far short) |
| 12 | CQR (quantile-regression conformal) | session 35 Part 4 | two GBR quantile models (5%/95%) | 0 | 42 | PARTIAL (21.3% coverage, bought via width inflation) |
| 13 | Jackknife+/CV+ (plain) | Stage 1 follow-up | K=3 leave-battery-group-out | 0 | 42 | WIN (37.1% CALCE coverage, largest prior gain, config-specific) |
| 14 | KMM-CP (kernel mean matching conformal) | Stage 3.1 | bounded-weight kernel reweighting | 0 | 42 | WIN (34.0% coverage, best coverage-per-width tradeoff pre-this-pass) |
| 15 | CORAL (2nd-order covariance alignment) | Stage 3.4 | direct MMD replacement | 0 | 42 | LOSS (worst of every method tried at the time, 3.2%) |
| 16 | Rescaled Jackknife+/CV+ | Stage 3.5 | CV+ residuals rescaled by train-only difficulty regressor | 0 | 42 | WIN but flagged (82.0% coverage, width approaches vacuous) |
| 17 | Stage 1.1 duration-feature reformulation (ratio vs. cycle-10 baseline) | Stage 1 | ICHV/TEVD/TEVI -> _rel | 0 | 42 | PROMOTED |
| 18 | Stage 1.2 per-battery sample weighting | Stage 1 | - | 0 | 42 | NOT carried forward (real CALCE R2 tradeoff) |
| 19 | Stage 1.5 XGBoost monotone_constraints on cycle_idx | Stage 1 | constraint=-1 on cycle_idx only | 0 | 42 | PROMOTED (deployed model's own recipe) |
| 20 | Stage 5 extended reformulation (SCV/MATD/VIECT/MET) | Stage 5 | ratio (SCV/MET) + delta (MATD/VIECT) reformulation | 0 | 42 | PROMOTED via routing (CALCE/Oxford/HUST only, not blanket - regresses XJTU) |
| 21 | Dataset-aware routing (positive-list) | post-Stage-5 | EXTENDED_ROUTED_DATASETS={CALCE,Oxford,HUST} | 0 (label-based selection, disclosed) | 42 | PROMOTED, currently deployed |
| 22 | Severson variance-model baseline | Stage 6.1 | 1 feature (log_var_dq), ElasticNetCV | 5-fold CV over l1_ratio grid [.1,.5,.7,.9,.95,.99,1] | 42 | LOSS vs. deployed (real, disclosed baseline) |
| 23 | Attia-style rich-feature baseline | Stage 6.1 | 4 features, ElasticNetCV | same as #22 | 42 | LOSS vs. deployed |
| 24 | TabPFN / DeepHPM | Stage 6.2 | - | 0 | 42 | (see Stage 6.2 entry) |
| 25 | Symbolic regression (SISSO-inspired SIS+Lasso, Stage 6.5 + group B retry) | Stage 6.5 / 18-item pass item 6 | genuine SISSO unavailable (broken build chain), SIS+Lasso approximation used | 0 | 42 | LOSS vs. deployed both times; retry WIN vs. its own prior degenerate attempt |
| 26 | Counterfactual explanations | Stage 6.6 | - | 0 | 42 | (diagnostic, not a predictive-accuracy method) |
| 27 | World Model (Stage 7.1) | Stage 7.1 | raw sequence tensors, no HI features | 0 | 42 | (see Stage 7.1 entry) |
| 28 | Self-supervised pretraining (order-ranking + ACCEPT-style physics-simulated) | Stage 7.2 / 18-item pass item 15 | first-order ECM perturbation (genuine SPM out of scope) | 0 | 42 | LOSS, dramatically below deployed on every eval set |
| 29 | Neural-operator SPM surrogate | Stage 7.3 | - | 0 | 42 | (see Stage 7.3 entry) |
| 30 | Isolation Forest vs. OC-SVM (anomaly detection) | 18-item pass item 1 | default sklearn params | 0 | 42 | LOSS (worse false-flag rate) |
| 31 | CoV / EMA adaptive loss weighting | 18-item pass items 2-3 | - | 0 | 42 | LOSS both |
| 32 | Cox Proportional Hazards survival analysis (RUL) | 18-item pass item 5 | lifelines (scikit-survival unavailable) | 0 | 42 | LOSS |
| 33 | Reference-anchored peak-tracking | 18-item pass item 7 | - | 0 | 42 | TIE |
| 34 | Isotonic-calibrated / composite-kernel-GPR conformal | 18-item pass items 8-9 | GPR on 800-row subsample (O(n^3) cap) | 0 | 42 | LOSS both |
| 35 | DFC-DGGate domain-difference gating | 18-item pass item 14 | 3-branch gated ensemble | 0 | 42 | SPLIT, not promotable |
| 36 | U-H-Mamba hybrid MC-Dropout + conformal UQ | 18-item pass item 16 | purpose-built dropout MLP | 0 | 42 | LOSS/INCONCLUSIVE |
| 37 | GroupNorm swap (CNN-LSTM BatchNorm replacement) | 18-item pass item 17 | - | 0 | 42 | WIN in-domain, LOSS zero-retrain (closed out this pass) |
| 38 | Few-shot k-shot affine/continued-training adaptation | research pass 2 item 1 | K in {5,10} | 0 | 42 | (per-dataset mixed, see entry) |
| 39 | DANN (domain-adversarial neural network) | research pass 2 item 2 | - | 0 | 42 | (see entry) |
| 40 | PyBaMM physics pretraining | research pass 2 item 3 | - | 0 | 42 | (see entry) |
| 41 | Bagged XGBoost diversity ensemble | research pass 2 item 4 | same XGB_KWARGS as #2, bagged | 0 | 42 (+ bag seeds) | (see entry) |
| 42 | Part A: 5 external-method upgrades | Part A | - | 0 | 42 | 0/5 valid wins vs. routed baseline |
| 43 | Part B: 204-battery-pool BatteryLife retrain | Part B | same recipe as #19/#20 | 0 | 42 | 2/5 mixed, not promoted |
| 44 | Item 1: label-free routing (domain-classifier AUC rule) | this pass, item 1 | AUC tie-margin=0.01 | 0 | 42 | LOSS (ties naive baseline, 8/13) |
| 45 | Item 2: cycle_idx ablation (drop / equivalent-full-cycles) | this pass, item 2 | same XGB recipe as #19 | 0 | 42 | NEUTRAL (not a shortcut, no promotion) |
| 46 | Item 3: few-shot conformal calibration | this pass, item 3 | k in {0,5,10,20,50} | 0 | 42 | LOSS (coverage stays <14% even at k=50) |
| 47 | Item 4: relaxation-voltage features | this pass, item 4 | - | 0 | - | STOPPED at feasibility gate (5/13 datasets) |
| 48 | Item A: online conformal (PID + nexCP) | pass 2, item A | eta multiplier {.01,.05,.1,.2}, rho {.95,.99}, burn-in {5,10,20} | small grid, not a search (sensitivity sweep) | 42 | WIN, strongest positive conformal result in project history |
| 49 | Item B: leave-one-dataset-out (LODO) | pass 2, item B | same XGB recipe as #19 | 0 | 42 | see item B's own results |
| 50 | Item D: honest routing selection | pass 2, item D | 4 candidate strategies | 0 (candidate set fixed, not searched) | 42 | neither direction reproduces oracle routing |

## Battery-level split confirmation

**Every train/test/calibration split in this project operates on whole
batteries, never on individual cycles within a battery** - confirmed by
direct code reading, not assumed:

- `src/split_utils.py`'s `battery_level_split()` - the canonical
  train/test split (`data/processed/battery_split.json` and its
  expanded-pool counterpart) - splits on `sorted(set(battery_ids))`,
  per-dataset stratified, with an explicit `pinned_test_ids` mechanism
  (added after a real near-miss: B0018 silently moving from test to
  train when the pool expanded - documented, fixed, not hidden).
- `src/run_conformal.py`'s `calib_eval_battery_split()` - the
  calibration/evaluation split used by every conformal-prediction item
  in this project (including items 3, 5e, A, E of this pass) - splits
  on unique battery IDs (even/odd by sorted position), never on rows.
- `GroupKFold(groups=battery_id)` - used in Stage 6.1's Severson/Attia
  baselines and `run_groupkfold_cv.py` - guarantees no battery's cycles
  appear in both a fold's train and test split.
- Every battery-level bootstrap CI in this project (item 5a, item B,
  this pass's checks) resamples **battery IDs** with replacement, then
  takes all of that battery's rows - never resamples individual rows.
- LODO (item B) holds out entire **sources** (all of a dataset's
  batteries at once) - a coarser, a fortiori battery-level split.

**One known, historical exception - found, corrected, and disclosed,
not a currently-live leak**: Check 0.3 (Stage 0) found that the
ORIGINAL Binary Firefly Algorithm feature-selection run (`run_bfa.py`)
included CALCE's cycles in its GroupKFold-cross-validated fitness
function pool, alongside NASA+MIT - a DATASET-VISIBILITY leak (CALCE
was supposed to be a never-trained-on holdout), not a cycle-within-
battery leak. Corrected via a NASA+MIT-only BFA re-run; the corrected
8-feature set (`bfa_selected_features_nasa_mit_only.txt`) has been
canonical for every model in this project since Stage 1 (confirmed:
`stage1_common.py`'s own module docstring states this explicitly). The
original leaked run's own numbers were never used for any promotion
decision after this was found.

## Files

- `requirements.txt` - the direct dependencies, pinned (what Streamlit Cloud installs). Every third-party
  package imported by `app.py` and the modules it reaches (`src/live_inference.py`, `src/battery_passport.py`, ...)
  is listed, including `pyarrow` (needed by `pandas.read_parquet`; it is also pulled in by Streamlit).
  It also lists some packages the app does not import (for example `lime`, `dice_ml`, `gplearn`, `tabpfn`, `beep`)
  that are only used by research scripts.
- `requirements-lock.txt` - the full `pip freeze` of the environment used (adds Streamlit's own dependencies).
- **Python version**: developed and deployed on Python 3.14 (the `.venv` is 3.14.2). `runtime.txt` says
  `python-3.11`, which is stale: Streamlit Cloud takes the Python version from the app's settings, not from this
  file, and the pinned versions (numpy 2.4, pandas 3.0, torch 2.13) need a recent Python. Use 3.14.
- `tests/` - quick smoke tests: `python -m pytest tests -q` (about 10-35 s).
- `scripts/reproduce_paper.sh` - re-runs the final-pass scripts (`src/run_finalpass*.py`) that regenerate the
  `outputs/finalpass*.csv` tables in `PAPER_RESULTS.md` sections 1-12. It does **not** regenerate the later "toolkit"
  tables (trust threshold, label-efficient checkpoints, minimum checkpoints); those come from
  `src/run_toolkit_*.py`, `src/trust_operating_point.py` and `src/validate_trust_threshold_leave_source_out.py`
  (see the "Rerun status log" at the end of `PAPER_RESULTS.md`). It also needs the raw datasets in
  `data/raw/` (git-ignored; see `DATA_AVAILABILITY.md`) and the processed files already in the repository.
- `src/fresh_clone_smoke_test.py`, `src/verify_sidecars_vs_git_blobs.py` - checks intended for a fresh clone
  (models load, one prediction per built-in dataset, passport export, encoder-provenance hashes).
- `DATA_AVAILABILITY.md` - every dataset source, access method, and license/terms, as documented at the point
  each was integrated into this project.

Quick start (from a fresh clone):

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt   # or requirements-lock.txt for the exact full environment
python -m pytest tests -q
streamlit run app.py
```
