# Overnight run log — 2026-07-21 22:4x IST to 2026-07-22 12:0x IST (~14h incl. follow-up fix session)

Running fully unattended per user instruction. This file is a live log:
every assumption, shortcut, and workaround is recorded here as it happens,
in chronological order.

## TL;DR — read this first (updated after the CNN-LSTM root-cause fix)

The original overnight run (ending ~04:22) shipped with a real bug: the
CNN-LSTM base learner never learned (R2=-0.071), which also silently
crippled Phase 4's joint model (same backbone). The user asked for this
to be root-caused rather than left as a known issue. It was: **the raw
`dVdQ` input channel reaches ~9.5 MILLION in places (vs O(1-100) for
every other channel) and was never normalized**, which poisoned CNN-LSTM's
BatchNorm running statistics. Fixed with per-channel robust
normalization; Phases 2, 3, 4, 5, and 6 were all re-run against the fix.
Numbers below are POST-FIX (final); the original numbers are preserved
inline further down, in the sections where each bug was found.

| # | phase | status | headline result (post-fix) |
|---|---|---|---|
| 1 | 16 HIs + BFA FS + RUL labels + ICA/DV/DC | done, unaffected by the fix | BFA kept 7/16 HIs (ICHV,SCV,VDEDT,VIECT,MATC,MATD,TEVI), RMSE 3.90->2.86 |
| 2 | 4 base learners (XGBoost, VLSTM, CNN-LSTM, PiFormer) | done | XGB R2=0.907, VLSTM R2=0.806, PiFormer R2=0.617, **CNN-LSTM R2=0.334 (was -0.071, now genuinely learning though still weakest)** |
| 3 | Stacking ensemble (Ridge + XGBoost meta) | done | Still ≈ XGBoost alone (RMSE 1.481 vs 1.478) - now for a legitimate reason: even the fixed CNN-LSTM is still well behind XGBoost, not because it's broken |
| 4 | Joint SOH+RUL, adaptive loss ablation | done, incl. log_sigma-clamp fix | **Single-task collapse textbook-clean both directions** (soh_only RUL R2=0.030, rul_only SOH R2=-0.040). Adaptive's runaway alpha/beta (was 0.7->8.4) fixed via clamping to [-0.7,0.7] on log_sigma - **result is a genuine split decision**: adaptive now wins RUL (R2=0.432 vs fixed_balanced's 0.428) but still loses SOH (0.344 vs 0.416); alpha/beta converged to the SAME value (2.028) rather than an asymmetric trade-off, explaining why |
| 5 | TreeSHAP + DeepSHAP | done | XGBoost/meta rankings unchanged (as expected); CNN-LSTM's voltage-region SHAP concentration went from 0.000 (meaningless, broken model) to 0.354 (meaningful) |
| 6 | Split-conformal (MAPIE) | done | SOH: 95.1% coverage (unchanged); RUL: **88.9%** (up from 83.3%, closer to 90% target as a side effect of the better model, not actively fixed per instruction) |

**Bugs caught and fixed across the full run (6 total, all detailed
inline at the point each was found)**: a pandas read-only-array crash in
BFA; a battery-split stratification bug that zeroed out NASA from the
first XGBoost test set; a conformal-calibration exchangeability
violation (in-sample residuals made SOH coverage look like 27% instead
of 95%); the CNN-LSTM input-normalization root cause itself; a second,
related bug where retrained-model train-set predictions were rebuilt
from raw unnormalized tensors; and an OneDrive-sync-lag trap where a
downstream script briefly read a stale prediction file despite the
writing script having exited cleanly (caught by noticing suspiciously
identical output, not by an error).

**What's now a genuinely open, unresolved item** (neither the CNN-LSTM
convergence issue nor the adaptive-weighting divergence - both fixed):
even after clamping log_sigma to stop alpha/beta from diverging,
`adaptive` still only wins a split decision against `fixed_balanced`
(beats it on RUL, loses on SOH) rather than clearly winning outright.
Root cause identified but not further pursued: alpha and beta converged
to the SAME value (2.028, the clamp ceiling) rather than an asymmetric
split, meaning this parametrization mostly expresses "how confident
overall" rather than "how to trade off SOH against RUL" - a real
limitation of this specific weighting scheme, not a bug. Also open: RUL
conformal coverage (88.9% vs 90% target) and PiFormer's slight
regression (0.735->0.617, plausibly ordinary run-to-run variance) -
explicitly left alone per instruction, since it's weaker than VLSTM but
not broken.

**Where everything lives**: `OVERNIGHT_LOG.md` (this file, narrative,
read top-to-bottom for the full chronological story including the
original bugs and the follow-up fix session), `STATUS.md` (dataset
acquisition notes from the earlier data-loading session), `src/*.py`
(all pipeline code), `data/processed/` (hi_table.parquet, RUL labels,
differential tensors, all model predictions - now post-fix),
`models/` (every trained model, post-fix), `outputs/` (14 PNG plots +
SHAP/conformal CSV summaries, regenerated post-fix), `logs/` (raw
stdout from every phase's run, both pre- and post-fix, suffixed `_v2`
for the re-runs).

## Compute environment (read before judging any scope decisions below)

- CPU: Intel Core 7 150U, 12 logical cores. **No GPU** (`nvidia-smi` not
  found). All deep learning runs on CPU via PyTorch.
- RAM: ~16 GB total, ~2.8 GB free at start of run. This is tight — MIT batch
  files are 2-3 GB each as HDF5, so they're read lazily per-cell via h5py
  rather than loaded whole into memory.
- Implication logged up front: "real epoch counts" for the deep models
  (VLSTM/CNN-LSTM/PiFormer) will mean *real but modest* — small hidden
  dims, few layers, tens of epochs, not hundreds — because this is a
  laptop CPU, not a training server. This is the single biggest scope
  compromise in the whole run and it's made once, here, rather than
  re-litigated at every step below.

## Plan (6 phases from the user's message, tracked via TodoWrite too)

1. 16 Health Indicators (all datasets) + BFA feature selection + RUL labels
   + ICA/DV/DC differential tensors
2. Train 4 base learners (XGBoost, VLSTM, CNN-LSTM, PiFormer-Transformer)
   on NASA + MIT subset, split by battery
3. Stacking ensemble (ridge + XGBoost meta-learners)
4. Joint SOH+RUL with adaptive loss weighting + ablation
5. TreeSHAP / DeepSHAP(→KernelSHAP fallback) + HI-region check
6. Split-conformal prediction (MAPIE) for SOH and RUL

---

## [Phase 1] Core modules built

- `src/data_adapters.py` — normalizes NASA/CALCE/MIT into common per-cycle
  records (charge/discharge {t,V,I,T}, discharge_capacity). Sign convention:
  I>0 charge, I<0 discharge everywhere (NASA discharge current is flipped
  to match, since it's logged positive natively).
- `src/health_indicators.py` — all 16 HIs (CDECT, ICHV, UVP, SCV, VDEDT,
  VIECT, LVP, MATC, MATD, MATDL, MET, TCCC, TCVC, TECD, TEVD, TEVI). Every
  acronym expansion is MY interpretation (none were given formal
  definitions) — see the module docstring for the full reasoning. Bug
  caught + fixed during testing: the initial CC-phase detector broke on a
  1-sample startup current transient (e.g. B0005 cycle 1 charge current:
  [-0.0012, -4.03, 1.51, 1.51, ...]) — fixed by referencing the median of
  the *middle* half of the array instead of the first few samples.
  Verified on NASA B0005: HIs trend monotonically with known capacity
  fade (e.g. TECD drops from 3311s at cycle 1 to 2427s at cycle 151, in
  step with capacity dropping 1.86->1.36 Ah).
- `src/rul_labels.py` — EOL = first cycle where capacity <= 80% of the
  median of the first 3 cycles' capacity (robust "initial capacity").
  Verified NASA EOL cycles: B0005=102, B0006=61, B0007=125, B0018=77 — all
  consistent with published NASA PCoE fade curves, none censored.
- `src/ica_dv_dc.py` — dQ/dV, dV/dQ, dI/dV on a common 200-point voltage
  grid per cycle, Savitzky-Golay smoothed (window 15, poly 3). Capacity Q
  is integrated from raw current for all 3 datasets uniformly (not reusing
  each dataset's own reported capacity fields) — see module docstring.
- `src/bfa_feature_selection.py` — S-shaped binary firefly algorithm,
  wrapper fitness = 0.99*accuracy + 0.01*feature-ratio (standard wrapper-FS
  convention), n_agents=30, n_iterations=100 (literature-standard scale,
  not reduced) with a cheap Ridge-regression surrogate for the 3000 inner
  fitness evaluations this implies.

## [Phase 1] MIT subset selection: 28 cells (not all 185)

Picked 7 cells per batch (28 total across the 4 batches) by taking 7
evenly-spaced points along each batch's *sorted cycle_life percentile*
range, so the subset spans short-life to long-life cells rather than
being randomly or arbitrarily chosen. 2 of 46 cells in the 20180412 batch
have NaN cycle_life (a data-quality artifact in that HDF5 file, not a bug
in our reader) and were excluded from consideration.

Why 28 and not e.g. 20 or 30: it's within the user's suggested 20-30
range, and picking a fixed 7-per-batch keeps all 4 charge-policy families
equally represented rather than skewing toward whichever batch happens to
have more cells.

Selected cells + cycle_life logged to `data/processed/mit_subset.json`.
Cycle_life range covered: 148 (short-life outlier, batch 2) to 1935
(longest-life cell, batch 3) — genuinely representative of the dataset's
spread, not just the median.

## [Phase 1] Full feature extraction: COMPLETE

35/35 batteries processed, 26,996 total cycles (NASA 636, CALCE 2,943,
MIT 23,417). `data/processed/hi_table.parquet` (22 cols: dataset,
battery_id, cycle_idx, discharge_capacity, SOH, RUL, 16 HIs) and
`data/processed/rul_summary.csv` (per-battery EOL/censored) both written.
35 ICA/DV/DC tensors in `data/processed/differential_tensors/`.

**Finding worth flagging**: most MIT batch-1 and several batch-3 cells
come back `censored=True` under our EOL rule (discharge capacity never
crosses 80% of the median-of-first-3-cycles "initial capacity" within the
logged cycles) — e.g. b1c4 logs 1225 cycles and still never crosses 80%.
Yet MIT's own HDF5 `cycle_life` field reports a finite number for these
same cells (e.g. b1c4: cycle_life=1227). This is a genuine methodological
difference, not a bug: MIT's `cycle_life` appears to use a different EOL
reference (likely nameplate/rated capacity, or a noise-filtered fit,
rather than "first crossing of 80% of this cell's own first few logged
cycles"). Both labelings are kept — ours in `rul_summary.csv`, theirs
recoverable from the raw HDF5 `cycle_life` field if needed later. Not
treated as ground truth in either direction here.

Runtime: NASA ~0.6s/battery, CALCE ~130s/battery (930-1040 xlsx-derived
cycles each, openpyxl parsing is the bottleneck, not the HI math), MIT
2.7-30s/battery depending on cycle_life. Total wall time ~13 minutes.

## [Phase 1] BFA feature selection: COMPLETE

One bug caught immediately: `df[cols].to_numpy(dtype=float)` returned a
read-only array (pandas 3.0 + pyarrow-backed parquet dtypes) — in-place
NaN imputation threw `ValueError: assignment destination is read-only`.
Fixed with `to_numpy(..., copy=True)`.

Ran at full literature-standard scale: 30 agents x 100 iterations = 3000
wrapper-fitness evaluations (Ridge regression, 3-fold GroupKFold by
battery), on the imputed, pooled NASA+CALCE+MIT HI table (26,996 rows).
Baseline RMSE (SOH%, all 16 features) = 3.897. Converged by iteration ~60
to RMSE=2.864 using 7/16 features — **better** than using all 16, which is
the expected/correct outcome for a working wrapper FS run (fewer, more
relevant features generalizing better across batteries than the full set).

**Selected 7 features: ICHV, SCV, VDEDT, VIECT, MATC, MATD, TEVI.**
Notably keeps both temperature HIs (MATC, MATD) despite CALCE's 8,829
imputed NaN cells for those two columns — the wrapper still found them
worth their imputation noise, i.e. NASA/MIT's real temperature signal
outweighed the CALCE imputation noise in cross-validated fitness. Also
notable: BOTH TCCC/TCVC (charge-timing) and TECD/TEVD (discharge-timing
duration) were dropped in favor of TEVI (voltage-interval duration) and
VDEDT/VIECT (voltage-level features) — i.e. the wrapper preferred
voltage-shape indicators over raw CC/CV timing for this pooled dataset.
Selected feature list saved to `data/processed/bfa_selected_features.txt`,
full fitness-per-iteration history to `data/processed/bfa_history.csv`.

This 7-feature subset is what Phase 2's XGBoost base learner uses (per the
task: BFA ran on the full pooled dataset, but the resulting feature
subset feeds only the NASA+MIT XGBoost training, matching "train all four
base learners ... on NASA + MIT").

## [Phase 2] XGBoost base learner: COMPLETE (with a split bug caught + fixed)

First run's battery-level split (`src/split_utils.py`, "every 5th battery
in sorted order") put **zero NASA batteries in the test set** — NASA
sorts first alphabetically (B0005...B0018) and has exactly 4 batteries,
so the stride-5 slice starting at index 4 landed entirely inside the MIT
block. First-run numbers (RMSE=0.954, R2=0.955) were real but not
representative of cross-dataset generalization. Fixed by stratifying the
split per-dataset (NASA and MIT each contribute their own held-out slice)
before rerunning — this is now the split every Phase 2+ script uses.

**Final XGBoost result** (26 train / 6 test batteries, test set: B0018 +
5 MIT cells): **RMSE=1.479, MAE=0.990, R2=0.907** on held-out SOH(%).
Lower than the buggy first run, as expected — NASA's chemistry/fade
pattern genuinely differs from MIT's fast-charge A123 cells, so making
the model actually generalize across datasets is a harder and more honest
test. Predictions saved to `data/processed/predictions/xgb_preds.csv`,
model to `models/xgb_soh.json`.

## [Phase 2] VLSTM / CNN-LSTM / PiFormer: COMPLETE (~65 min wall time)

Target-standardization bug caught during smoke-testing (see module
docstring in train_deep_models.py) and fixed before the real run: without
z-scoring the SOH target, val MSE sat around 5100 (RMSE~71) after 3
epochs because the untrained output head starts near 0 while SOH~70-100 -
essentially all early training was wasted moving the bias into range.
Fixed, verified on a 3-battery smoke test, then launched full run: 21 fit
/ 5 val / 6 test batteries, 14,872 fit cycles, 40-epoch budget with
early-stopping patience=8 (see compute-environment note for why 40 and
not more).

**Results (test set, SOH%):**
| model    | RMSE  | MAE   | R2     | epochs run |
|----------|-------|-------|--------|------------|
| VLSTM    | 2.694 | 1.634 | 0.690  | 40 (no early stop, still improving) |
| CNN-LSTM | 5.006 | 3.952 | -0.071 | 8 (early-stopped, never improved) |
| PiFormer | 2.491 | 1.571 | 0.735  | 28 (early-stopped) |

**CNN-LSTM did not learn** - val MSE at epoch 0 was 2.84 and never beat
that across 8 epochs before early-stopping fired, despite train MSE
dropping normally (0.885 -> 0.674). R2=-0.071 means it's worse than
predicting the mean SOH for every test row. VLSTM and PiFormer both
converged cleanly by comparison, so this isn't a data or pipeline
problem - it's specific to the 4-branch multi-kernel CNN-LSTM's
optimization at these settings. Working hypothesis (not verified further
to stay on schedule): BatchNorm1d in the 4 conv branches, combined with
batch_size=64 over very heterogeneous NASA+MIT cycle counts per battery,
may be giving the model unstable running statistics between train() and
eval() mode - a common source of exactly this "train loss fine, val loss
stuck/worse" signature. **Flagged for follow-up, not fixed here**: a
retry with GroupNorm instead of BatchNorm, or a lower learning rate,
would be the first things to try. Proceeding with the ensemble AS-IS
(including the underperforming CNN-LSTM) since the stacking meta-learners
can in principle learn to downweight a bad base learner - itself an
informative result either way.

All 4 base learners' predictions saved: `xgb_preds.csv`,
`deep_models_test_preds.csv`, `deep_models_train_preds.csv`. Models saved
to `models/{vlstm,cnn_lstm,piformer}_soh.pt`.

## [Phase 3] Stacking ensemble: COMPLETE

**Comparison table (test set, SOH%, sorted by RMSE):**
| model | RMSE | MAE | R2 |
|---|---|---|---|
| XGBoost | 1.478 | 0.990 | 0.907 |
| Stacking-Ridge | 1.480 | 0.989 | 0.906 |
| Stacking-XGBoost | 1.490 | 0.992 | 0.905 |
| PiFormer | 2.491 | 1.571 | 0.735 |
| VLSTM | 2.694 | 1.634 | 0.690 |
| CNN-LSTM | 5.006 | 3.952 | -0.071 |

**Honest finding, not a bug: stacking did not beat the single best base
learner.** The Ridge meta-learner's fitted coefficients are
`{XGBoost: 1.011, VLSTM: -0.007, CNNLSTM: -0.036, PiFormer: -0.004}` -
it essentially learned to ignore all three deep models and reproduce
XGBoost's prediction almost verbatim. This is the correct behavior for a
linear meta-learner facing one base learner (XGBoost, tabular HI
features) that's dramatically more accurate than the other three
(sequence-based deep models, trained far fewer effective "epochs" worth
of gradient signal than XGBoost's 500 boosting rounds, on a CPU budget) -
there's no diverse-but-comparably-strong signal for the meta-learner to
usefully combine. Stacking earning its keep would need base learners of
more comparable strength; that's a real limitation of this run's compute
budget, not of the stacking implementation. Full table + per-cycle
predictions in `data/processed/predictions/ensemble_comparison.csv` and
`ensemble_test_preds.csv`.

## [Phase 5] SHAP analysis: COMPLETE

**A real bug caught in code review before it ran**: my first draft of the
KernelSHAP fallback wrapper never actually referenced the real test
instance being explained — it perturbed between two different background
samples instead of between "real instance" and "baseline", which would
have produced meaningless attributions. Caught and fixed while re-reading
my own code (not by a failed run), rewritten as
`_grouped_kernelshap_one_instance`: explains ONE real instance at a time,
mask=1 keeps that instance's real (channel, coarse-bin) group value,
mask=0 replaces it with the background mean — the correct present/absent
semantics KernelSHAP expects.

**In practice, the fallback was never needed**: DeepSHAP succeeded on
all 3 deep models in this environment's SHAP version (0.52.0), including
CNN-LSTM (contains `nn.LSTM`) and PiFormer (contains
`nn.MultiheadAttention` + `nn.LayerNorm`) — layers DeepSHAP is
historically flaky with. SHAP emitted one harmless warning
("unrecognized nn.Module: LayerNorm") for PiFormer but still completed.
Logged per the task's requirement to say which method was used per
model: **DeepSHAP for VLSTM, CNN-LSTM, and PiFormer, all three** (no
KernelSHAP fallback triggered, though that code path was independently
unit-tested and works).

**TreeSHAP, XGBoost base learner** (mean|SHAP| over the 7 BFA-selected
features): SCV=2.266, VIECT=1.934, TEVI=0.905, ICHV=0.244, MATC=0.199,
MATD=0.136, VDEDT=0.066. Top-3 by SHAP (SCV, VIECT, TEVI) are exactly the
voltage-shape features BFA's own history showed converging on early —
**SHAP importance ranking agrees with BFA's revealed preference**,
cross-validating that BFA's selection wasn't a fluke of its wrapper
fitness function.

**TreeSHAP, Stacking-XGBoost meta-learner**: pred_XGBoost=4.139,
pred_PiFormer=0.0026, pred_VLSTM=0.0007, pred_CNNLSTM=0.0000 — confirms
the Ridge coefficients finding from Phase 3 independently: the ensemble
is, in SHAP's own accounting, >99.9% "just XGBoost."

**Voltage-region check (does SHAP mass concentrate in 3.55-3.8V?)**,
measured as fraction of total |SHAP| mass (V_t channel mapped by its own
time-to-voltage curve, dQdV/dVdQ/dIdV channels mapped by their voltage
grid) landing in that window:
| model | fraction in [3.55,3.8]V |
|---|---|
| VLSTM | 0.518 |
| CNN-LSTM | 0.000 |
| PiFormer | 0.040 |

VLSTM genuinely concentrates over half its attribution mass in that
narrow ~0.25V window — a real, specific finding (this is the region
where these NASA cells' discharge voltage plateau sits, consistent with
it being diagnostically rich). CNN-LSTM's ~0 is not a finding about
voltage regions, it's the same broken/non-learning model showing near-
zero gradients everywhere (consistent with Phase 2). PiFormer's low 4%
concentration is itself informative: its cross-channel attention appears
to spread importance across the ICA channels and the full voltage range
rather than localizing narrowly — a genuinely different explanation
pattern from VLSTM's, not a failure of the analysis.

Full outputs: `outputs/shap_xgboost_base_ranking.csv`,
`outputs/shap_meta_ranking.csv`, `outputs/shap_deep_models_summary.csv`.

## [Phase 6] Conformal prediction: a real methodological bug caught by checking the numbers, not just running the code

First draft calibrated MAPIE's split-conformal wrapper on the TRAIN
split's own residuals (the same rows the Ridge/XGBoost meta-learners
were fit on). This ran without error and LOOKED done, but the empirical
coverage came back **27.1% against a 90% target** - a massive miss. Before
writing that off as "conformal prediction is just imperfect," checked
whether it was a MAPIE-wrapper bug: reimplemented split-conformal by
hand (absolute-residual quantile, no MAPIE) and got the **identical**
27.1% coverage. That ruled out a wrapper bug and pointed at the real
cause: calibration residuals (median 0.115 SOH%) vs. test residuals
(median 0.777 SOH%) differ by ~6.7x, because the meta-learner was FIT on
those exact "calibration" rows, so its in-sample error is nowhere near
its genuine held-out error. Split-conformal's coverage guarantee requires
calibration/test exchangeability, which in-sample residuals badly
violate.

**Fix**: the 6 held-out TEST batteries (never used to fit anything) are
split in half by sorted battery_id - `calib=[B0018, b2c24, b3c35]`,
`eval=[b1c4, b3c0, b4c38]` - one half calibrates the conformal interval,
the other half's coverage is what gets reported. Both halves are
genuinely unseen by every fitted model. Cost, logged rather than hidden:
final coverage is now measured on 3 batteries/3,462 cycles instead of 6
batteries/5,208 - real loss of statistical power, and B0018 (the only
NASA battery in the whole test set) landed in the calibration half, so
**the final reported coverage numbers below are validated on MIT cells
only**, not on NASA.

**SOH (Stacking-Ridge) result after the fix**: target coverage 90%,
**empirical coverage 95.1%**, avg interval width 4.69 SOH percentage
points, n=3,462. Slightly conservative (over-covers) rather than
under-covers, which is the expected and safe direction for finite-sample
split-conformal - a legitimate, trustworthy result, in sharp contrast to
the pre-fix 27.1%.

## [Phase 4] Joint SOH+RUL ablation results: ran clean, but confounded by the same backbone problem CNN-LSTM had in Phase 2

All 4 variants (fixed_balanced, soh_only, rul_only, adaptive) trained for
their full 25-epoch budget without errors (1308s total). Raw test
results:

| variant | RUL RMSE | SOH RMSE | RUL R2 | SOH R2 |
|---|---|---|---|---|
| adaptive | **335.65** | 4.891 | -0.002 | -0.023 |
| fixed_balanced | 337.03 | 4.944 | -0.010 | -0.045 |
| rul_only | 337.69 | **4.850** | -0.014 | -0.006 |
| soh_only | 350.34 | 4.874 | -0.091 | -0.016 |

**Honest read, not the clean story the task asked to demonstrate**: the
`JointSOHRULModel` shares its backbone (4-branch multi-kernel CNN + LSTM)
with Phase 2's CNN-LSTM, which already failed to learn (R2=-0.071, flagged
above). Here, `val_soh` sits at 2.8-3.0 essentially flat across ALL FOUR
variants for the entire 25 epochs (compare fixed_balanced epoch 0 val_soh
2.845 vs epoch 24 val_soh 2.856 - no net progress) - the backbone isn't
extracting a useful SOH signal regardless of loss weighting, so every
variant's SOH R2 is negative (worse than predicting the mean) and RUL R2
is roughly zero. This is a shared-architecture ceiling problem, not a
property of the loss-weighting scheme being compared.

**What DOES partially survive as real signal despite that ceiling:**
- `soh_only` has the clearly worst RUL RMSE (350.3, vs 335-338 for the
  other three) - consistent with the expected "RUL head collapses when it
  never receives gradient" story, since alpha=1/beta=0 means the RUL head's
  loss term is exactly zero and its weights never update from their random
  init.
- `adaptive` achieves the best RUL RMSE and 2nd-best SOH RMSE of the four
  variants - consistent with "adaptive wins," though the margin over
  `fixed_balanced` is small (335.65 vs 337.03 RUL, both far from good in
  absolute terms).
- `rul_only`'s SOH RMSE does NOT show the mirror collapse (4.850, actually
  the single best SOH number of the four) - this is the part that breaks
  the clean narrative. Expected: with alpha=0, the SOH head's weights
  should also be frozen at random init, giving it the worst SOH score.
  Instead it's the best, most likely because when EVERY variant's SOH
  head is already near-random-quality (backbone ceiling), which specific
  random init "wins" becomes noise rather than signal.

**Conclusion logged honestly**: the ablation demonstrates the *mechanism*
correctly (adaptive alpha/beta genuinely move during training - see raw
log, alpha drifts 0.559->0.998 and beta 0.533->1.081 over the adaptive
run, both pulling upward together rather than either collapsing to 0,
confirming the log-variance reparametrization is working as designed) and
shows a directionally-consistent partial pattern (soh_only's RUL collapse,
adaptive's best-RUL/near-best-SOH result), but does NOT cleanly show
"forced-single-task collapse on the untrained target" for both directions
simultaneously, because the shared backbone's own convergence problems
(same root cause as CNN-LSTM's Phase 2 failure) suppress the signal-to-
noise ratio needed to see it clearly. **Flagged for follow-up alongside
the CNN-LSTM fix**: fixing the backbone's optimization issue (GroupNorm
swap or lower LR, per the Phase 2 note) would very likely also clean up
this ablation's story, since both symptoms trace to the same architecture.
Full history in `data/processed/predictions/joint_ablation_history.csv`,
summary table in `joint_ablation.csv`.

## [Phase 6] Conformal prediction: COMPLETE

**RUL (Phase 4 joint-adaptive model) result**: target coverage 90%,
**empirical coverage 83.3%**, avg interval width 961.2 cycles, n=3,462
(same calib=[B0018,b2c24,b3c35] / eval=[b1c4,b3c0,b4c38] split as SOH).
Undershoots the 90% target this time (unlike SOH's 95.1% over-coverage) -
plausible given the RUL point-estimate itself comes from the Phase 4
joint-adaptive model, whose RUL R2 was -0.002 (essentially uninformative,
see above) and whose calibration set is only 3 batteries, so the tail
behavior of RUL residuals is poorly estimated from so few batteries.
Reported honestly rather than tuned to hit 90% - a 6.7-point undershoot on
a weak underlying point-estimate, with a small calibration set, is a
believable and explicable result, not a red flag to hide.

**Final coverage table** (`outputs/conformal_coverage.csv`):
| target | method | target coverage | empirical coverage | avg width | n |
|---|---|---|---|---|---|
| SOH (Stacking-Ridge) | MAPIE.SplitConformalRegressor | 90% | 95.1% | 4.69 (SOH%) | 3,462 |
| RUL (joint-adaptive) | MAPIE.SplitConformalRegressor | 90% | 83.3% | 961.2 (cycles) | 3,462 |

Note: MAPIE emitted a harmless "Estimator does not appear fitted" warning
on both calls (our `PrefitLookup` shim doesn't expose sklearn's usual
fitted-attribute markers) but completed correctly both times - verified
independently against a from-scratch manual split-conformal
implementation for SOH (identical numbers), so the warning doesn't affect
correctness, just cosmetics.

---

# Follow-up session — CNN-LSTM root-cause fix (2026-07-22, later)

User asked to investigate the CNN-LSTM failure (R2=-0.071) with a specific
BatchNorm/eval-mode hypothesis, since it also plausibly explained why the
Phase 4 joint-model ablation showed near-zero R2 across all 4 variants
(same backbone).

## Root cause found: NOT an eval-mode bug, but genuinely unnormalized inputs

Checked all three of the user's specific hypotheses:
1. **Is `.eval()` called before val/test inference?** Yes - confirmed in
   `train_one_model`, `model.eval()` + `torch.no_grad()` precede every
   validation/test forward pass. Not the bug.
2. **Are validation batch sizes small enough to destabilize BatchNorm?**
   Not applicable the way hypothesized - in eval() mode, BatchNorm1d uses
   frozen `running_mean`/`running_var`, not batch statistics, so eval
   batch size can't be the mechanism (this only matters in train mode).
3. **Are `running_mean`/`running_var` actually updating?** Yes,
   `num_batches_tracked=233` confirmed they were updating - but the
   VALUES they converged to were the actual smoking gun:
   `running_var` was on the order of **1e14 to 1e15**, and
   `running_mean` in the hundreds/thousands - wildly, numerically
   unstable magnitudes for anything that's supposed to normalize toward
   unit variance.

Traced this to the source: the model's 6th input channel, **dVdQ**
(differential voltage, from `src/ica_dv_dc.py`), blows up wherever dQ is
near zero (flat-capacity plateaus) - measured raw values up to
**~9.5 million** for MIT cells (std ~325,000), while every other channel
(V_t, I_t, T_t, dQdV, dIdV) sits at O(1-100). This ~5-6 order-of-magnitude
scale mismatch was never normalized before being fed into the models -
an oversight from Phase 2's original build, not caught until now.

**Why only CNN-LSTM broke, and VLSTM/PiFormer looked fine**: this is the
part that makes it a genuinely satisfying diagnosis rather than a vague
"numerical issues" hand-wave. VLSTM has no BatchNorm at all (custom
peephole cell, plain Linear/sigmoid/tanh) - architecturally immune.
PiFormer uses LayerNorm, which normalizes per-SAMPLE at inference time
(no persistent running average to poison) - also immune. CNN-LSTM is the
ONLY one of the three using BatchNorm1d with a global running-average
statistic, which is exactly the mechanism that gets dominated by rare
extreme dVdQ spikes across many training batches and then generalizes
terribly at eval time (train-mode per-batch normalization masks the
problem locally; eval-mode global running stats don't). VLSTM's R2=0.690
and PiFormer's R2=0.735 weren't evidence the pipeline was fine - they
were evidence the bug was architecture-specific, and CNN-LSTM was the
one architecture positioned to expose it.

## Fix: per-channel robust normalization, fit on TRAIN data only

Added `compute_channel_norm_stats`/`apply_channel_norm` to
`sequence_features.py`: clip each channel to its [1st, 99th] percentile
(computed from the FIT battery split only, to avoid any leakage) before
z-scoring, so extreme dVdQ/dIdV spikes near-zero-dV regions can't
dominate the scale statistics the way raw mean/std would let them.

**Also caught and fixed a second, related bug while wiring this in**:
`train_deep_models.py`'s "save train-set predictions" block rebuilt a
completely RAW (unnormalized) tensor via a fresh `make_xy()` call and fed
it straight into the newly-normalization-expecting models - would have
produced silently garbage train-side predictions even after the main fix.
Fixed by reusing the already-normalized `X_fit`/`X_val` arrays directly
instead of rebuilding them, with an `assert` guarding the row-order
alignment between the reused arrays and the freshly-rebuilt id/label
metadata.

Normalization stats are computed ONCE (by `train_deep_models.py`, from
the fit-battery split) and saved to
`data/processed/channel_norm_stats.json`, then loaded and reused
identically by `train_joint_adaptive.py`, `run_conformal.py`, and
`run_shap_analysis.py` - all four scripts now apply the exact same
transform rather than each computing (and potentially drifting from) its
own. `run_shap_analysis.py` needed special care: it still needs REAL
volts (not z-scores) for the 3.55-3.8V region check, so it keeps a
separate raw copy of the V_t channel/V_grid for that specific analysis
while feeding the (now-normalized) tensor to the models for SHAP itself.

**Verified on a quick NASA-only smoke test before committing to the full
retrain**: post-normalization, CNN-LSTM's val MSE dropped cleanly from
0.554 (epoch 0) to 0.078 (epoch 8) - actual learning, compared to the
original run being stuck at ~2.84-2.90 for its entire (early-stopped)
8-epoch run. BatchNorm `running_var` after the fix: ~0.11-0.36 (vs.
~1e14-1e15 before) - back in a sane range.

## Full retrain results (all 3 deep models, same 40-epoch/patience-8 budget as before)

| model | RMSE (before) | RMSE (after) | R2 (before) | R2 (after) |
|---|---|---|---|---|
| VLSTM | 2.694 | **2.131** | 0.690 | **0.806** |
| CNN-LSTM | 5.006 | **3.948** | -0.071 | **0.334** |
| PiFormer | 2.491 | **2.993** | 0.735 | **0.617** |

**CNN-LSTM: fixed, confirmed by the numbers, but not fully "comparable to
VLSTM" as targeted.** R2 went from -0.071 (worse than predicting the
mean) to +0.334 (genuinely predictive) - a decisive confirmation that the
normalization diagnosis was correct, not a partial/lucky improvement.
It's still the weakest of the three deep models, though, not on par with
VLSTM's 0.806. Plausible remaining gap causes (not investigated further,
flagged for next follow-up rather than chased now): the 4-branch
multi-kernel conv front-end has more parameters/capacity than VLSTM or
PiFormer, so it may need more epochs or a lower learning rate to fully
converge post-normalization; BatchNorm can also just be intrinsically
noisier to optimize than LayerNorm/no-norm on a dataset this small
(14,872 fit cycles), independent of the input-scale bug that's now fixed.

**Side effects on the other two models, both plausible and both
logged rather than cherry-picked**: VLSTM improved further (0.690->0.806)
- normalization helps gradient-based optimization generally, not just
BatchNorm-specific numerical stability, so this is a believable genuine
gain, not noise. PiFormer got slightly worse (0.735->0.617) - a single
run's difference on a moderately-sized validation set; LayerNorm was
already handling PiFormer's scale robustness reasonably well pre-fix, so
this is more likely ordinary run-to-run training variance (different
random init interacting with the now-different loss landscape) than a
real regression, but reported as-is rather than re-run repeatedly to
cherry-pick a better seed.

Predictions and models regenerated: `deep_models_test_preds.csv`,
`deep_models_train_preds.csv`, `models/{vlstm,cnn_lstm,piformer}_soh.pt`
all overwritten with the new (normalized-input) versions.

## Yet another bug, caught by a suspiciously-identical number: OneDrive sync lag masked the first ensemble re-run

Re-ran `train_ensemble.py` immediately after retraining finished and got
**Ridge meta-learner coefficients IDENTICAL to the pre-fix run**
(`{XGBoost: 1.011, VLSTM: -0.007, CNNLSTM: -0.036, PiFormer: -0.004}`,
intercept 3.346, matching to 3 decimals) - despite CNN-LSTM's test R2
having just changed from -0.071 to +0.334. That's not a plausible
coincidence for a closed-form Ridge fit on genuinely different input
data, so before trusting the table, checked
`data/processed/predictions/deep_models_train_preds.csv`'s modification
time: **03:54 AM - untouched since the ORIGINAL first-session run**,
despite `train_deep_models.py` finishing (exit 0, "ALL DONE" printed) at
~10:33 AM and definitely executing the code that writes that exact file.
Confirmed via content, not just timestamp: the stale file's CNN-LSTM
column had almost zero variance (std=0.057, all values ~95.22) - the
fingerprint of the OLD broken constant-output model, not the newly
retrained one.

This project lives inside a synced `OneDrive\Desktop\...` folder;
working hypothesis is OneDrive's background sync/placeholder
materialization delayed the write becoming visible on disk, rather than
a code bug (the script had already returned exit 0 with no exception, so
the Python-level file write itself must have been issued correctly).
Re-checked the file a few minutes later: now correctly timestamped
~10:37 AM with real per-row variance (std=6.73). **Re-ran the ensemble
comparison again** on the confirmed-fresh file - see updated numbers
below. Lesson logged for future overnight runs on synced folders: verify
output file CONTENT (not just "did the script exit 0") before trusting a
downstream script's read of it, especially immediately after a long
background job finishes.

## Phase 3 re-run: ensemble STILL doesn't beat standalone XGBoost, but the story behind why is now cleaner

| model | RMSE | MAE | R2 |
|---|---|---|---|
| XGBoost | 1.478 | 0.990 | 0.907 |
| Stacking-Ridge | 1.481 | 0.998 | 0.906 |
| Stacking-XGBoost | 1.491 | 0.994 | 0.905 |
| VLSTM | 2.131 | 1.564 | 0.806 |
| PiFormer | 2.993 | 1.928 | 0.617 |
| CNN-LSTM | 3.948 | 2.926 | 0.334 |

**Answering the user's question directly: no, stacking still does not
beat standalone XGBoost**, even with CNN-LSTM genuinely contributing now
instead of being broken. New Ridge coefficients:
`{XGBoost: 1.010, VLSTM: 0.007, CNNLSTM: -0.011, PiFormer: -0.006}`,
intercept -0.042 - essentially unchanged in spirit from before (still
>99% weight on XGBoost), but now for a legitimate reason rather than a
broken-model artifact: **even the best deep model (VLSTM, R2=0.806) is
still meaningfully behind XGBoost (R2=0.907)**, and all three deep models
sit further below that. A linear meta-learner has no incentive to blend
in a systematically weaker, correlated-error predictor - this is the
statistically correct behavior of stacking, not a limitation of the
implementation. Full table in `ensemble_comparison.csv`, per-cycle
predictions in `ensemble_test_preds.csv`.

## Phase 4 re-run: single-task collapse is now textbook-clean, but "adaptive wins" does NOT hold - a real, diagnosed finding, not forced into the hoped-for shape

| variant | SOH RMSE | SOH R2 | RUL RMSE | RUL R2 |
|---|---|---|---|---|
| **fixed_balanced** | **3.695** | **0.416** | **253.73** | **0.428** |
| soh_only | 4.759 | 0.032 | 330.29 | 0.030 |
| rul_only | 4.932 | **-0.040** | 263.83 | 0.381 |
| adaptive | 4.563 | 0.110 | 284.73 | 0.279 |

**Half the expected story now shows up cleanly, confirmed by the fix**:
- `soh_only` (alpha=1, beta=0 - RUL head never gets gradient) collapses
  on RUL exactly as predicted: R2=0.030, essentially uninformative,
  vs. 0.428/0.381/0.279 for the other three variants that all train the
  RUL head.
- `rul_only` (alpha=0, beta=1 - SOH head never gets gradient) collapses
  on SOH exactly as predicted, even more starkly: **R2=-0.040, actually
  negative** (worse than predicting the mean), vs. 0.416/0.032/0.110 for
  the other three. This is the textbook "forced-single-task collapse on
  the untrained target" signature the task asked to demonstrate, and
  with the backbone fix in place, it's now unambiguous in both
  directions - it was NOT visible in the pre-fix run (see above), where
  every variant was too broken to show any real pattern.

**The other half does not hold: `fixed_balanced` wins on BOTH targets,
beating `adaptive`.** This is a genuine result, not a bug I'm papering
over - checked the training log before accepting it. Adaptive's alpha
and beta (the learned task weights) grew from ~0.7 each at epoch 0 to
**8.4 and 8.2 by epoch 24** - unbounded growth, not convergence to a
sensible balance point - and `adaptive`'s TRAINING LOSS went **negative**
(-1.83 by epoch 24). Given the loss is
`alpha*L_soh + log(sigma_soh) + beta*L_rul + log(sigma_rul)` (see
`models/joint_model.py`), a negative total loss means the `log(sigma)`
regularization terms (which shrink as alpha/beta grow) are numerically
dominating the actual prediction-error terms rather than balancing them
- the optimizer found a way to make the log-variance penalty very
negative faster than it could actually reduce prediction error, a known
degenerate-optimization risk of the unconstrained Kendall-et-al.
uncertainty-weighting formula when alpha/beta aren't clipped or
otherwise regularized. **Diagnosed, not fixed**: the straightforward
next step would be clamping `log_sigma` to a bounded range (e.g.
[-3, 3]) to prevent alpha/beta from running away, but that's a real code
change left for follow-up rather than done under this ablation's time
budget.

**Bottom line, stated plainly**: fixing CNN-LSTM's normalization bug
turned this ablation from "everything clustered near R2=0, no usable
signal" into a genuinely informative result - just not the specific
"adaptive wins" result hoped for. What it actually shows: (1) forced
single-task training measurably breaks the untrained head, confirmed in
both directions now that the backbone works: soh_only R2 drops to
1/14th of fixed_balanced's, rul_only's SOH R2 goes negative; (2) this
particular adaptive-weighting implementation has a real, diagnosed
failure mode (unbounded alpha/beta growth) that made it underperform the
much simpler fixed 50/50 split here. Both are legitimate things to know
before using this joint model for anything real. Full history in
`joint_ablation_history.csv`, summary in `joint_ablation.csv`.

## Phase 6 re-run: RUL coverage improved as a side effect (not actively fixed, per instruction)

| target | before fix | after fix |
|---|---|---|
| SOH (Stacking-Ridge) | 95.1% coverage, width 4.69 | 95.1% coverage, width 4.64 (essentially unchanged - expected, SOH's point estimate didn't change) |
| RUL (joint-adaptive) | 83.3% coverage, width 961.2 | **88.9% coverage, width 871.4** |

SOH is essentially unchanged, as expected (its point estimate, Stacking-
Ridge, doesn't depend on the CNN-LSTM fix at all). RUL's coverage moved
from 83.3% to 88.9% - much closer to the 90% target - purely as a side
effect of the underlying joint-adaptive model being far more accurate
now (RUL R2 -0.002 -> 0.279). **Per instruction, this gap was not
actively worked on** - no calibration-method changes were made here,
this improvement is entirely downstream of the Phase 2/4 model fix.
Still not fully at 90%, so still logged as a known open item, just a
smaller one than before.

## Follow-up session 2 — log_sigma clamping fix for adaptive loss weighting

User asked specifically to fix the adaptive-weighting divergence
(alpha/beta running away to 8.4/8.2) since "adaptive beats fixed" is
part of Experiment 3 in their evaluation protocol.

**Bound choice, not the naive suggestion**: the user's example bound was
"clamp log_sigma to [-3,3] or similar." Worked the math before
implementing: `alpha = 0.5*exp(-2*log_sigma)` is extremely sensitive to
log_sigma because of that `-2x` in the exponent - at log_sigma=-3, alpha
would still reach 0.5*exp(6)=201.7, nowhere near "sensible," and
wouldn't have prevented the ORIGINAL divergence at all (which only
reached log_sigma=-1.41, well inside [-3,3]). Used **[-0.7, 0.7]**
instead, which bounds alpha/beta to roughly [0.12, 2.03] - wide enough to
let the model meaningfully favor one task up to ~4x over the fixed
50/50 split, but not explode past where the log(sigma) regularizer term
can overwhelm the actual prediction-error terms.

Implemented as (1) an in-forward `torch.clamp` on log_sigma before it's
used to compute alpha/beta - this alone gives zero gradient outside the
bound, which should stop further movement - PLUS (2) an explicit
in-place `.clamp_()` on the raw parameter after every optimizer step, as
belt-and-suspenders against Adam's momentum carrying the raw parameter
slightly past the boundary even when the clamped-value's gradient there
is exactly zero.

Re-ran ONLY the `adaptive` variant (per the user's instruction - the
other 3 don't depend on this parameter at all, so re-running them would
have been pure wasted compute). Added mode-subset support to
`train_joint_adaptive.py` (`python train_joint_adaptive.py adaptive`)
that merges the new row into the existing `joint_ablation.csv` by
variant name rather than requiring a full 4-variant re-run or clobbering
the untouched rows.

**Mechanism confirmed working**: alpha/beta climbed from their 0.5 init
to exactly **2.028 (the clamp ceiling) by epoch 5, and stayed pinned
there for the rest of training** - the clamp is doing its job,
preventing the unbounded growth seen before. Training loss going
slightly negative (-0.87 onward) is now benign/expected, not a sign of
degenerate optimization: at the clamp boundary the constant
`log_sigma_soh + log_sigma_rul = -1.4` regularizer offset can exceed the
(now well-behaved) weighted prediction-error terms without indicating
anything is broken - unlike before, where the negative loss was
correlated with alpha/beta still actively diverging.

**One more genuinely interesting observation, not previously visible**:
alpha and beta converged to the EXACT SAME value (2.028) and moved
together throughout training, rather than diverging from each other to
reflect a genuine SOH-vs-RUL uncertainty asymmetry. This means the
model, freed to move within the [0.12, 2.03] band, chose to scale both
tasks' weights up by the same ~4x factor rather than finding an
asymmetric balance - i.e. even bounded, this particular parametrization
still primarily expresses "how confident am I overall" rather than "how
should I trade off SOH against RUL," which is a real, useful thing to
know about this weighting scheme's behavior, independent of the final
R2 numbers.

## Final adaptive-vs-fixed comparison: a genuine split decision, reported honestly

| variant | SOH RMSE | SOH R2 | RUL RMSE | RUL R2 |
|---|---|---|---|---|
| fixed_balanced | **3.695** | **0.416** | 253.73 | 0.428 |
| adaptive (clamped) | 3.918 | 0.344 | **252.80** | **0.432** |

**The clamp fix closed most of the gap and flipped the RUL result in
adaptive's favor** (RUL R2 0.279->0.432, now marginally beating
fixed_balanced's 0.428; RUL RMSE 284.73->252.80, now marginally BETTER
than fixed_balanced's 253.73). **But adaptive still loses on SOH**
(R2 0.344 vs fixed_balanced's 0.416) - a real, if smaller, gap than
before the fix (was 0.110 vs 0.416).

**Honest verdict, as instructed**: adaptive weighting's theoretical
benefit partially materialized here - it now wins on RUL (barely) after
losing badly on both targets pre-clamp - but it does not clearly beat
fixed_balanced overall, since it still trails meaningfully on SOH. Given
the alpha=beta=2.028 observation above (both weights moved together
rather than finding an asymmetric split), this isn't surprising: uniformly
upweighting BOTH loss terms by the same factor doesn't change their
RELATIVE balance versus a fixed 50/50 split, so any difference from
fixed_balanced here is coming from the overall gradient-scale change
(larger effective learning-rate-like effect from alpha=beta=2.03x)
rather than genuine adaptive task-rebalancing. **This is a legitimate
limitation to flag rather than a bug to keep chasing**: the specific
homoscedastic-uncertainty parametrization used here, even correctly
bounded, converged to a solution that scales both tasks' weights
together rather than trading off between them - a different tuning of
the clamp bounds, a different weighting scheme entirely (e.g. GradNorm,
or a learned SOFTMAX-normalized alpha+beta=1 constraint that forces an
actual trade-off), or simply more epochs, are the natural next things to
try, but are left for future follow-up rather than pursued further here,
per the instruction to move to final consolidation after this result.

## Phase 5 re-run: CNN-LSTM's SHAP values are now meaningful (were pure noise before)

| model | fraction in [3.55,3.8]V (before) | fraction (after fix) |
|---|---|---|
| VLSTM | 0.518 | 0.601 |
| CNN-LSTM | 0.000 | **0.354** |
| PiFormer | 0.040 | 0.117 |

XGBoost's TreeSHAP ranking and the meta-learner's ranking are both
essentially unchanged (expected - XGBoost itself wasn't retrained, and
the meta-learner still weights it at >99%). The interesting change is
CNN-LSTM: its voltage-region concentration went from 0.000 (meaningless -
a broken, near-constant-output model has no real gradient signal for
DeepSHAP to attribute anywhere) to **0.354**, a genuinely interpretable
value in the same ballpark as VLSTM's. This is independent confirmation,
via a completely different analysis method, that CNN-LSTM is now a real,
learning model rather than the collapsed one from before. VLSTM and
PiFormer's concentration both increased too (0.518->0.601,
0.040->0.117) - plausibly because normalized inputs give cleaner, less
noisy gradients for DeepSHAP to attribute through generally, not specific
to any one model.
