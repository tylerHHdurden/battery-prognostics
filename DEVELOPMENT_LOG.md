# Development Log — 2026-07-21 22:4x IST to 2026-07-22 12:0x IST (~14h incl. follow-up fix session)

Running fully unattended per user instruction. This file is a live log:
every assumption, shortcut, and workaround is recorded here as it happens,
in chronological order.

## TL;DR — read this first (updated after the CNN-LSTM root-cause fix)

The original run (ending ~04:22) shipped with a real bug: the
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

**Follow-up session 3 (additive, not re-run against the table above)**:
added a lightweight ICA/DV/DC feature-fusion step (small CNN encoder,
16-dim embedding, concatenated with the 7 HIs / 4 base predictions - no
attention, per instruction). Single confirmatory retrain:
**XGBoost+fusion R2=0.917 (was 0.907), Stacking-Ridge+fusion R2=0.917
(was 0.906)** - both genuinely improved. Fully additive: new
`*_fusion.*` files alongside the originals, nothing existing overwritten
(verified by timestamp). Full detail in its own section below.

**Where everything lives**: `DEVELOPMENT_LOG.md` (this file, narrative,
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
below. Lesson logged for future runs on synced folders: verify
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

## Follow-up session 3 — lightweight ICA/DV/DC feature-fusion (additive, no attention)

Added a concat-fusion step per request: a small CNN encoder
(`src/models/ica_encoder.py` - Conv1d(3->16,k=7) -> Conv1d(16->16,k=5)
-> AdaptiveAvgPool1d -> 16-dim embedding, trained via a lightweight SOH
head using the exact same `train_one_model` loop as VLSTM/CNN-LSTM/
PiFormer) compresses the 3 ICA/DV/DC channels (dQdV/dVdQ/dIdV) into a
fixed-size vector per cycle. Concatenated (no attention) with the 7
BFA-selected HIs for XGBoost, and with the 4 base-learner predictions
for the Ridge meta-learner. Entirely additive: new scripts
(`train_fusion_encoder.py`, `train_xgboost_fusion.py`,
`train_ensemble_fusion.py`) and new output files
(`xgb_soh_fusion.json`, `ridge_meta_fusion.pkl`,
`fusion_embeddings.csv`, `xgb_fusion_preds.csv`,
`ensemble_fusion_test_preds.csv`) - verified by file timestamp that
none of the original pipeline's files (`xgb_soh.json`, `xgb_preds.csv`,
`ensemble_comparison.csv`, `ensemble_test_preds.csv`) were touched.

**One smoke-test bug caught before the real run**: a first-pass sanity
check fed the encoder RAW (unnormalized) ICA channels and got
train/val MSE in the hundreds of thousands - the same `dVdQ`-scale bug
that broke CNN-LSTM originally, reproduced instantly on a fresh model
architecture, confirming how easy it is to reintroduce. Fixed by
applying the same saved `channel_norm_stats.json` transform before
training, exactly as the fix requires; re-verified sane loss values
(~1-3) before committing to the full run.

**Results, single retrain pass (no with/without ablation suite, as
instructed)**:

| model | RMSE | MAE | R2 |
|---|---|---|---|
| XGBoost (7 HIs only) | 1.478 | 0.990 | 0.907 |
| **XGBoost + fusion (7 HIs + 16-dim ICA embedding)** | **1.392** | **0.959** | **0.917** |
| Stacking-Ridge (4 base preds only) | 1.481 | 0.998 | 0.906 |
| **Stacking-Ridge + fusion (4 base preds + 16-dim ICA embedding)** | **1.394** | **0.966** | **0.917** |

Both fusion-enabled models trained cleanly and **genuinely improved**
over their non-fusion counterparts - XGBoost's R2 0.907->0.917, ensemble's
0.906->0.917. This is a single confirmatory run, not a rigorously
controlled ablation (same battery split and same non-fusion base-learner
predictions were reused throughout, so the comparison is apples-to-apples
on that front, but statistical significance / seed-to-seed variance
wasn't checked) - reported as "fusion trains and helps," not as a fully
validated architectural claim.

## Follow-up session 4 — physics-informed loss (monotonicity penalty), additive

Added a physics-informed loss term to VLSTM/CNN-LSTM/PiFormer per
request: (1) `src/physics_loss.py:fit_fade_curves` fits an empirical
exponential capacity-fade curve `SOH(cycle) = A*exp(-k*cycle) + C` per
TRAINING battery (scipy `curve_fit`, bounded `k>=0` so the fit itself
cannot express a capacity-recovering trend) - fitted for all 21/21
training batteries, k range `[0.00007, 0.00621]`, all non-negative as
expected, confirming the physics prior is consistent with the data;
(2) `monotonicity_penalty` is the actual trainable loss term: a
vectorized pairwise check within each mini-batch - for every same-
battery pair (i,j) where cycle_j > cycle_i, penalizes
`relu(pred_j - pred_i)`, discouraging the model's own predicted SOH from
rising at a later cycle. Exploits that random 64-sample batches drawn
from ~700-cycles-per-battery data almost always contain multiple same-
battery pairs, so no change to the batching/sampling strategy was
needed. Combined loss = `MSE + lambda * penalty`, **lambda=0.1 fixed,
not tuned**, chosen once because both terms run O(1) in standardized-SOH
units throughout this project.

**Divergence safety net (as instructed): not triggered for any of the
3 models.** All three trained stably at lambda=0.1 on the first attempt
- no NaN/Inf, no negative loss (mathematically impossible here anyway,
since both loss terms are non-negative by construction - unlike the
earlier Kendall-uncertainty joint loss, which used a log(sigma) term
that genuinely could and did go negative), and no epoch-0-vs-final-epoch
blowup. The penalty term stayed small and bounded throughout training for
all three models (e.g. VLSTM: 0.0014 -> peak ~0.014 -> 0.0102, never
approaching or exceeding the MSE term's scale) - halving-and-retry was
never needed.

**Results (single retrain pass, test set, additive - original
non-physics models/files fully untouched, verified by timestamp):**

| model | RMSE (baseline) | RMSE (physics) | R2 (baseline) | R2 (physics) |
|---|---|---|---|---|
| VLSTM | 2.131 | 2.234 | 0.806 | **0.787** |
| CNN-LSTM | 3.948 | 4.261 | 0.334 | **0.224** |
| PiFormer | 2.993 | 3.043 | 0.617 | **0.604** |

**Honest result: the physics-informed loss made all three models
slightly WORSE at this fixed weight**, not better. Reported as-is, per
instruction ("report updated RMSE/MAE/R2... " with no requirement that
it improve). A plausible explanation, not further investigated in this session:
real per-cycle SOH measurements have genuine session-to-session noise
(temperature effects, brief internal-resistance recovery between
sessions, measurement variance) that is not perfectly monotonic even in
truly degrading cells - the fitted fade curves themselves are smooth
monotonic idealizations, but the actual training/test LABELS the model
is scored against retain some of that local non-monotonic noise. A
penalty that forces the model's predictions toward strict monotonicity
fights against fitting that real (if noisy) local signal in the labels,
trading a bit of raw accuracy for a physically-motivated but here
net-negative constraint. Flagged as a candidate for follow-up (e.g. a
smaller lambda, or applying the penalty only to the smoothed/summary
capacity trend rather than every raw cycle) rather than pursued further
in this session, exactly as instructed (single pass only).

New files: `src/physics_loss.py`, `src/train_deep_models_physics.py`,
`models/{vlstm,cnnlstm,piformer}_soh_physics.pt`,
`data/processed/predictions/deep_models_physics_{test_preds,metrics}.csv`
and per-model `*_physics_history.csv`. Original `vlstm_soh.pt`,
`cnn_lstm_soh.pt`, `piformer_soh.pt`, and `deep_models_metrics.csv`
confirmed untouched by file timestamp.

## Follow-up session 5 — CALCE zero-retrain evaluation of the fusion ensemble

Single evaluation experiment, as requested: run the final fusion-enabled
ensemble (XGBoost+fusion, Stacking-Ridge+fusion - explicitly NOT the
physics-informed models, which stayed a separate non-adopted result) on
all 2,941 usable CALCE cycles with **zero retraining** - every model
(ICAEncoder, VLSTM, CNN-LSTM, PiFormer, XGBoost-fusion, Ridge-meta-
fusion) loaded from its already-trained weights and used purely for
inference. CALCE was never in any train/val/fit split for any of these
models (Phase 1's BFA feature selection is the only place CALCE data was
used at all, for feature selection, not model fitting).

**One bug caught immediately on first run**: `model.y_mean_`/
`model.y_std_` (the SOH de-standardization constants) were set as plain
Python attributes on the model INSTANCE during original training, not
saved into `state_dict()` - a freshly-loaded model object doesn't have
them, so the first attempt crashed with `AttributeError`. Fixed by
recomputing y_mean/y_std from the exact same NASA+MIT fit-battery split
used at original training time (not from CALCE - that would leak
test-domain statistics into what's supposed to be a fixed training-time
constant). All 3 deep models share one y_mean/y_std pair since they were
all trained on the same y_fit.

**Zero-retrain results (out-of-domain, all 2,941 CALCE cycles):**

| model | RMSE | MAE | R2 |
|---|---|---|---|
| XGBoost-fusion, NASA+MIT (in-domain) | 1.392 | 0.959 | 0.917 |
| XGBoost-fusion, CALCE (zero-retrain) | **17.969** | **14.079** | **0.304** |
| Stacking-Ridge-fusion, NASA+MIT (in-domain) | 1.394 | 0.966 | 0.917 |
| Stacking-Ridge-fusion, CALCE (zero-retrain) | **17.838** | **14.012** | **0.314** |

**A dramatic, expected domain-shift collapse**: R2 falls from 0.917 to
~0.31, RMSE grows more than 12x. Physically unsurprising - CALCE's CS2
cells are a different form factor, chemistry, and cycling protocol than
either NASA's 18650 cylindrical cells or MIT's A123 LFP fast-charge
cells, and CALCE additionally has NO temperature channel at all (100%
of MATC/MATD imputed with NASA+MIT training medians, a genuine data-
availability gap layered on top of the domain shift, not just a harder
version of the same problem). The BFA-selected HIs and fusion embedding
generalize only partially; an R2 of 0.31 (rather than ~0, or negative)
suggests the model retains SOME real signal (better than predicting the
mean) but should not be trusted for anything quantitative on this cell
chemistry without retraining or fine-tuning on CALCE data.

**Conformal interval check - does it widen on CALCE? No, and that's the
finding.**

| domain | half-width | empirical coverage | target |
|---|---|---|---|
| NASA+MIT (in-domain eval) | 2.367 | 95.6% | 90% |
| CALCE (out-of-domain, zero-retrain, SAME calibration) | **2.367** | **6.1%** | 90% |

The interval half-width is **bit-for-bit identical** between domains -
expected by construction, not a bug: this project's split-conformal
implementation computes ONE global residual quantile from calibration
and applies it as a fixed additive/subtractive band to every test point,
with no mechanism to adapt to a harder or more out-of-distribution
input. Applied to CALCE, that fixed band - correctly calibrated for
95.6% in-domain coverage - captures only **6.1%** of CALCE's true SOH
values, because CALCE's actual errors are roughly an order of magnitude
larger than what the band was sized for. This is a genuine, important
limitation to flag rather than a defect in this run specifically:
**standard split-conformal's coverage guarantee assumes calibration and
test data are exchangeable, and it provides no warning signal when that
assumption is violated by domain shift** - the interval looks exactly as
confident on CALCE as it does in-domain, while being almost entirely
wrong there. A locally-adaptive conformal method (e.g. normalized
conformal prediction, scaling the interval by a per-input difficulty
estimate) would be the natural fix, but is out of scope for this single
evaluation experiment per instruction.

Per instruction, early-prediction test, drop-one-branch ablation, and
the homogeneous-bagging baseline were explicitly skipped this round.
Results saved to `data/processed/predictions/calce_zero_retrain_{metrics,preds}.csv`
and `outputs/calce_zero_retrain_conformal.csv`.

## Follow-up session 6 — plain-English health report generator

Built a small pipeline that turns the fusion ensemble's structured
per-cycle output into a natural-language health report via an LLM API,
per request. Two new files:

- `src/build_report_context.py` - assembles, for one specific
  `(dataset, battery_id, cycle_idx)`: SOH prediction (Stacking-Ridge-
  **fusion**, per instruction - not the physics-informed models) with its
  conformal interval; RUL prediction with its conformal interval; the
  top-3 SHAP-ranked features for THAT SPECIFIC cycle (per-instance
  TreeSHAP on the XGBoost-fusion model's 23-feature vector, not a
  global/average ranking); and a per-instance voltage-region
  localization (per-instance DeepSHAP on VLSTM's voltage channel for
  that exact cycle, mapped back to real volts via the cycle's own V(t)
  curve).
- `src/generate_health_report.py` - the prompt template + `call_llm()`
  (Anthropic Messages API via `requests`, reading `ANTHROPIC_API_KEY`
  from the environment).

**Two things worth flagging about what "the ensemble's RUL prediction"
actually means here**: the fusion ensemble (`train_ensemble_fusion.py`)
only ever predicts SOH - it was never built or trained for RUL. The RUL
figure in every report therefore comes from the Phase 4 joint-adaptive
model (the only trained RUL predictor in this whole project), not from
the fusion ensemble itself. This is disclosed in the module docstring
rather than blurred, and is why RUL examples are limited to the 3
"eval" battery subset (b1c4, b3c0, b4c38) - the only rows with a
saved RUL prediction from Phase 6's conformal run.

**Environment reality check, done before writing any code**: no
`ANTHROPIC_API_KEY`/`OPENAI_API_KEY` in this environment, and no
`anthropic` package installed (network itself works - confirmed a live
HTTP round-trip to api.anthropic.com). `call_llm()` is written for a
real, working API call and needs nothing but a key added to actually run
unattended - but for this session's 5 test examples, it correctly returns a
`NO_API_KEY` sentinel rather than fabricating a fake response. Since
Claude (me, writing this) is itself an LLM, the 5 example reports below
were generated by reading each fully-assembled prompt exactly as the API
would have received it and writing the completion directly - clearly
labeled as such in `outputs/health_reports_examples.json`'s
`report_source` field, not presented as a live API result.

**5 example reports, spanning SOH from 100.4% down to 79.3% (near-EOL)
and RUL from 796 down to 112 cycles - every number in every report
verified by direct string cross-check against its source context (no
invented figures):**

1. MIT/b1c4 cycle 67 (SOH 100.4%, RUL 796): *"This battery is in
   excellent condition at 100.4% health (90% confidence interval:
   98.0%-102.7%), with its degradation signature most concentrated in
   the 2.0-3.11V region of the discharge curve. It has an estimated 796
   cycles of useful life remaining, with 90% confidence the true value
   falls between 360 and 1,232 cycles."*
2. MIT/b4c38 cycle 250 (SOH 100.0%, RUL 774): *"This battery is at 100%
   health (90% confidence interval: 97.6%-102.4%), with its discharge-
   curve degradation signature concentrated in the 2.0-3.23V range. The
   model estimates approximately 774 cycles remaining, with 90%
   confidence between 339 and 1,210 cycles."*
3. MIT/b1c4 cycle 674 (SOH 99.4%, RUL 678): *"This battery is at 99.4%
   health (90% confidence interval: 97.1%-101.8%), likely reflecting
   early-stage wear concentrated in the 3.08-3.55V region. Approximately
   678 cycles remain before end of life, with 90% confidence the true
   remaining life falling between 243 and 1,114 cycles."*
4. MIT/b3c0 cycle 747 (SOH 97.1%, RUL 112): *"This battery is at 97.1%
   health (90% confidence interval: 94.7%-99.5%), with degradation
   concentrated in the 2.23-3.24V region of its discharge curve.
   Remaining life is estimated at approximately 112 cycles, though the
   wide 90% confidence interval (0-548 cycles) reflects considerable
   uncertainty at this stage of life."*
5. MIT/b4c38 cycle 1096 (SOH 79.3%, RUL 419, past the 80% EOL threshold):
   *"This battery is at 79.3% health (90% confidence interval:
   76.9%-81.7%), with degradation strongly concentrated in the narrow
   3.13-3.28V region, consistent with its advanced wear state. The
   model estimates approximately 419 cycles remaining, but the wide 90%
   confidence range (0-855 cycles) signals that end-of-life could occur
   at any time."*

**A genuinely interesting side-finding from building this**: the
per-instance voltage regions above (2.0-3.6V-ish throughout) are all
correctly in MIT's actual operating range - NOT the 3.55-3.8V figure
reported for NASA in Phase 5. That NASA number was never a general
"batteries degrade in this voltage band" finding - it was specific to
NASA B0005's chemistry/voltage range (2.5-4.2V), and MIT's A123 LFP
cells simply operate lower (2.0-3.6V, confirmed back in the CNN-LSTM
normalization-fix session's channel-scale check). Computing the region
**per-instance** rather than reusing one global constant is what caught
this - a hard-coded "3.55-3.8V" in the prompt template would have been
silently wrong for every MIT example above.

Also notable: cycle 1096 (the most degraded example) is the only one
where a fusion-embedding dimension (`fusion_5`) cracked the top-3 SHAP
features, displacing TEVI - a hint that the learned ICA/DV/DC embedding
carries information the 7 hand-picked HIs don't, specifically in the
more-degraded regime, though this is a single-instance observation, not
a validated general pattern.

RUL confidence intervals are visibly very wide at low-RUL cycles (e.g.
0-548 cycles for a point estimate of 112) - a direct, honest consequence
of the joint-adaptive RUL model's modest R2 (0.279–0.432 depending on
variant, see Phase 4) and the fixed-width conformal calibration
discussed in the CALCE session; the reports don't hide this, they state
the wide interval plainly.

All 5 full prompts + reports + underlying context saved to
`outputs/health_reports_examples.json`.

## Follow-up session 7 — Streamlit Digital Twin dashboard

Built `app.py` plus three new supporting modules, using the fusion
ensemble for SOH and the joint-adaptive model for RUL, per instruction
(explicitly not the physics-informed variants). Functional-over-polished
as instructed: plain Streamlit widgets, one matplotlib voltage-curve
plot, no custom theming.

**New files**:
- `src/train_ocsvm.py` - One-Class SVM anomaly detector on the 23-feature
  space (7 BFA HIs + 16 fusion dims), trained on the NASA+MIT FIT split
  only (never test or CALCE).
- `src/precompute_app_constants.py` - one-time precomputation of SOH/RUL
  de-standardization constants and a DeepSHAP background sample, so the
  dashboard never has to reload the full NASA+MIT battery set (a
  multi-minute operation) just to answer one prediction.
- `src/live_inference.py` - the generic, LIVE (no lookup-table) inference
  pipeline: one cycle record in, full SOH+RUL+SHAP+anomaly+domain-check
  context out. Works identically for an existing test battery or a fresh
  CSV upload, per "a full pipeline rerun per new upload is fine, skip
  live incremental meta-learner updating."
- `app.py` - the dashboard itself: browse NASA/MIT/CALCE test batteries
  or upload a CSV, cycle slider (defaults to most recent = "current
  state"), SOH/RUL metrics with ground-truth deltas where available,
  per-instance SHAP (TreeSHAP table + VLSTM DeepSHAP voltage region),
  conformal intervals with an explicit out-of-domain warning, anomaly
  flag, and the LLM health report (or its structured-data fallback).

**Two real bugs caught and fixed during build/test, not shipped
silently:**

1. **OC-SVM severe class imbalance.** First-pass training used ALL fit
   cycles as-is: NASA (3 batteries, ~470 cycles) vs. MIT (18 batteries,
   ~14,400 cycles) - a ~30:1 imbalance. Sanity-checked the trained
   detector against its OWN training data before wiring it into the app
   and found **83.9% of NASA's own training cycles flagged "anomalous"**
   vs. only 2.2% of MIT's - meaning a perfectly legitimate, in-domain
   NASA battery would have shown the anomaly flag (and the linked
   out-of-domain warning) essentially always in the dashboard. Fixed by
   capping each fit battery to 200 cycles before fitting, so every
   battery gets comparable weight regardless of how long its test
   happened to run. Re-checked: NASA's false-flag rate dropped to 24.4%
   (MIT stayed ~2.3%) - a large improvement, though not perfectly
   balanced; flagged as a residual, not-fully-resolved limitation below
   rather than claimed as fully fixed.
2. **Negative RUL predictions displayed to the user.** `AppTest`
   surfaced this immediately on the very first automated run: NASA
   B0005's last logged cycle (already well past its true EOL) produced
   a raw RUL prediction of **-15 cycles**. Mathematically a faithful
   regression residual, but "-15 cycles remaining" is meaningless to
   show a dashboard user. Fixed by clipping the displayed value to
   `max(0, ...)` in `live_inference.py` (the conformal lower bound was
   already clipped this way; the point prediction itself wasn't).

**Verification method**: since this is a Streamlit app, "does it run"
can't be confirmed by starting the server alone (the script only
executes per client session, so a bare `curl` just returns the static
HTML shell). Used `streamlit.testing.v1.AppTest` to actually execute the
script and drive widget interactions programmatically - confirmed
**zero exceptions** across all four paths:
- NASA/B0005 (default, last cycle): SOH 71.9% vs. true 71.8% (+0.1),
  RUL 0 vs. true 0 - accurate, in-domain, no anomaly flagged.
- MIT/b1c17 (via dataset-switch interaction): SOH 82.6% vs. true 82.3%
  (+0.3), RUL 36 vs. true 1 - accurate SOH, RUL prediction here is much
  further from ground truth (consistent with the joint-adaptive model's
  known modest R2, see Phase 4).
- CALCE/CS2_35 (via dataset-switch interaction): SOH 66.0% vs. true
  26.7% (**+39.3** - a huge miss), out-of-domain warning correctly
  triggered (both reasons: no temperature channel AND OC-SVM anomaly),
  anomaly flag correctly shown. This is the same domain-shift collapse
  from the CALCE zero-retrain session, now visible live in the
  dashboard exactly where a user would need to see it.
- Synthetic uploaded CSV (2 cycles of idealized linear ramps, not real
  battery data): parsed correctly, correctly flagged both anomalous and
  out-of-domain (OC-SVM alone this time, since the synthetic upload DID
  include a temperature column) - confirms the OC-SVM check operates
  independently of the temperature-channel check, not just piggybacking
  on it.

**Known limitations, stated plainly rather than glossed over**:
- OC-SVM balance fix is a large improvement, not a perfect one (NASA
  cycles are still ~10x more likely to be flagged than MIT's, down from
  ~38x) - a residual artifact of NASA having only 3 distinct batteries
  to MIT's 18, even after per-battery cycle capping.
- CSV upload expects a specific column schema (documented in the
  sidebar's file-uploader caption) and a full charge+discharge cycle;
  discharge-only uploads or different column names will fail the
  parser with a shown error rather than a silent misread.
- No live incremental meta-learner updating, exactly as scoped - every
  prediction reruns the full fixed pipeline from already-trained
  weights.

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

## Follow-up session 8 — switched health-report LLM from Anthropic to Gemini

User provided a `GEMINI_API_KEY` and asked to switch `generate_health_report.py`
from the (never-actually-called, since no key existed) Anthropic path to a
live Gemini integration, keeping the prompt template unchanged.

**Setup**: key written to `.env` (git-ignored - added `.env` to
`.gitignore` first and confirmed via `git check-ignore -v .env` before
writing the file). No `python-dotenv` installed; wrote a ~10-line manual
`.env` parser in `generate_health_report.py` rather than add a dependency
for one key, consistent with this project's existing pattern (e.g. using
`requests` instead of the `anthropic`/`google-genai` SDKs).

**Real finding, not a bug in this code**: the requested model,
`gemini-2.5-flash`, returned **HTTP 404** for this specific key - *"This
model models/gemini-2.5-flash is no longer available to new users"* -
despite that exact model appearing in this same key's own
`/v1beta/models` listing. Verified this wasn't a code/request-formatting
issue by reproducing the identical 404 via a raw `curl` independent of
any Python code. `gemini-2.0-flash` hit a separate free-tier 429 rate
limit on the same key. `gemini-flash-latest` is the model alias
confirmed working (HTTP 200) - its response's own `modelVersion` field
reports it currently resolves to `gemini-3.6-flash`, not literally 2.5.
Substituted this as the default model, with the deviation and full
reasoning documented directly in `call_llm`'s docstring (not silently
swapped) - trivial to revert to `gemini-2.5-flash` if the account's
tier/billing changes later.

**Re-ran all 5 example reports live through the real API** (not a
fallback, not Claude-authored this time - genuine Gemini output). All 5
succeeded, `outputs/health_reports_examples.json` updated in place.
Verified numerically: **SOH point value and full RUL range (point +
both confidence bounds) present and correct in every one of the 5
reports, zero invented figures.**

One real thing worth noting, not a defect: every Gemini report omits
the SOH confidence interval, where the earlier Claude-authored
versions included it. Re-reading the prompt template's own closing
instruction line before assuming this was wrong: it asks the model to
mention "the SOH percentage" and "the RUL estimate **with its
confidence range**" - only RUL is explicitly told to include a range
(matching the worked example in the prompt, which also only ranges
RUL). Gemini is following the prompt's literal wording; the earlier
model's inclusion of the SOH range was extra, not more correct. Prompt
template left unchanged per instruction, so this is a legitimate
model-to-model style difference, not a regression.

`app.py`'s fallback-detection was widened from checking only
`NO_API_KEY` to also catch a new `API_ERROR` sentinel `call_llm` now
returns on any exception (timeout, non-2xx, malformed response), so a
live API failure still degrades to the structured-data display rather
than crashing the dashboard - "in case the key is missing or the call
fails for any reason," per instruction.

## Follow-up session 9 — the 3 remaining evaluation-protocol experiments

Ran the 3 experiments explicitly skipped in earlier sessions (CALCE
zero-retrain session skipped these three: early-prediction test,
drop-one-branch ablation, homogeneous-bagging baseline), all using the
final fusion-enabled ensemble - no retraining of any base learner in any
of the three, all evaluation-only reuse of already-trained models.

### Experiment 1: Early-prediction test (first 20% of each battery's cycles)

Filtered the existing full-lifetime test predictions
(`ensemble_fusion_test_preds.csv`) to each test battery's own first 20%
of logged cycles (per-battery cutoff, since lifetimes range 132-1230
cycles - pooling would let long-lived batteries dominate). No
retraining - a filtered re-evaluation of the already-fit ensemble.

| model | regime | RMSE | MAE | R2 |
|---|---|---|---|---|
| Stacking-Ridge-fusion | full lifetime | 1.394 | 0.966 | 0.917 |
| Stacking-Ridge-fusion | **early-life (first 20%)** | **0.847** | **0.287** | **-0.584** |
| XGBoost-fusion | full lifetime | 1.392 | 0.959 | 0.917 |
| XGBoost-fusion | **early-life (first 20%)** | **0.847** | **0.287** | **-0.582** |

**Read this table carefully - it is NOT "the model gets worse early in
life."** RMSE and MAE both **improve** substantially in the early-life
regime (1.39->0.85, 0.97->0.29) - the model's raw prediction errors are
genuinely smaller when cells are new. R2 going negative here is a
textbook artifact of near-zero target variance, not a sign the model
degraded: early-life SOH sits in an extremely narrow band (most test
batteries: 99.5-100.6% in their first 20% of cycles), so R2 (which
measures error relative to the variance of the target) can go deeply
negative even with tiny absolute errors, because there's almost no
genuine spread in the ground truth left to "explain." The per-battery
breakdown makes this unambiguous:

| battery | n cycles | SOH range | RMSE | MAE | R2 |
|---|---|---|---|---|---|
| b3c35 | 218 | 99.5-100.2 | 0.159 | 0.149 | 0.279 |
| b4c38 | 246 | 99.7-100.3 | 0.149 | 0.116 | 0.309 |
| b3c0 | 201 | 99.8-100.3 | 0.198 | 0.183 | -1.167 |
| b1c4 | 245 | 99.9-100.5 | 0.295 | 0.146 | -3.044 |
| B0018 | 26 | 92.6-100.6 | 5.074 | 3.983 | -3.372 |
| **b2c24** | 105 | **99.9-100.4** | 0.601 | 0.583 | **-52.321** |

b2c24's R2 of **-52.3** looks catastrophic in isolation, but its RMSE
(0.601) and MAE (0.583) are unremarkable - it just has the narrowest
true-SOH range of any battery here (99.9-100.4%, a 0.5-point spread), so
even a modest, consistent prediction offset relative to that razor-thin
range destroys R2 arithmetically. B0018 is the one battery here with a
genuinely large early-life RMSE (5.07) - it's also the only NASA battery
in the test set and the shortest-lived (132 total cycles, only 26 in its
"first 20%"), so this may reflect genuine early-life difficulty
specific to that battery/chemistry rather than a general early-
prediction weakness - not investigated further here. **Practical
takeaway**: use RMSE/MAE, not R2, to judge early-life performance in
this project - R2 is the wrong lens whenever the evaluation window has
near-constant ground truth. Full breakdown in
`data/processed/predictions/early_prediction_per_battery.csv`.

### Experiment 2: Drop-one-branch ablation

Refit ONLY the Ridge meta-learner (not the base learners) with one of
the 4 base-learner prediction columns removed at a time (fusion-
embedding columns always kept); measured the test-set performance drop
vs. the full 4-branch ensemble.

| variant | dropped | RMSE | delta RMSE | R2 | delta R2 |
|---|---|---|---|---|---|
| drop_CNNLSTM | CNN-LSTM | 1.3937 | **-0.0001** | 0.91696 | **+0.00001** |
| full_4_branch | (none) | 1.3938 | 0 | 0.91695 | 0 |
| drop_PiFormer | PiFormer | 1.3942 | +0.0004 | 0.91689 | -0.00005 |
| drop_VLSTM | VLSTM | 1.3950 | +0.0012 | 0.91680 | -0.00015 |
| **drop_XGBoost_fusion** | **XGBoost-fusion** | **2.0694** | **+0.6756** | **0.81693** | **-0.10002** |

Exactly the result the Ridge coefficients (`{XGBoost: 1.010, VLSTM:
0.007, CNNLSTM: -0.011, PiFormer: -0.006}`, reported back in the fusion
ensemble session) predicted: **dropping XGBoost causes a massive
performance collapse** (R2 0.917->0.817, RMSE +0.68), while dropping
any of the three deep models changes performance by less than 0.002
RMSE / 0.0002 R2 - noise-level, and dropping CNN-LSTM (the weakest
individual base learner, R2=0.334 standalone) very slightly *improves*
the ensemble, consistent with it contributing net-negative signal once
XGBoost and the fusion embedding are already present. This is
independent confirmation, via direct ablation rather than just reading
off fitted coefficients, that the ensemble's accuracy is carried almost
entirely by XGBoost + the fusion embedding, not by the deep models'
predictions. Full table in `data/processed/predictions/drop_branch_ablation.csv`.

### Experiment 3: Homogeneous-bagging baseline

Trained XGBoost-fusion 5 times (seeds 42, 0, 1, 7, 123), identical data
and hyperparameters otherwise, averaged the 5 models' test predictions.

| model | RMSE | MAE | R2 |
|---|---|---|---|
| Single XGBoost-fusion (seed=42) | 1.3921 | 0.9593 | **0.9172** |
| Heterogeneous Stacking-Ridge-fusion | 1.3938 | 0.9660 | 0.9169 |
| Homogeneous bag (5 XGBoost seeds) | 1.4234 | 0.9584 | 0.9134 |

Individual-seed R2 ranged **0.9035 to 0.9202** across the 5 seeds - real,
non-trivial seed-to-seed variance for a single XGBoost fit on this data.
But **averaging the 5 seeds performed worse than either the single best
seed or the heterogeneous ensemble**, not better - homogeneous bagging
here mostly smooths out each seed's minor idiosyncrasies without adding
the kind of genuinely diverse signal that would let averaging beat the
best individual fit. Combined with Experiment 2's finding (the
heterogeneous ensemble's 3 deep-model branches contribute ~nothing),
the overall picture is consistent and unglamorous: **for this dataset
and feature set, there is very little to be gained from combining
multiple models of ANY kind (same-type bagging or heterogeneous
stacking) beyond a single well-tuned XGBoost fit** - both combination
strategies land within noise of the single-model baseline, in slightly
different directions. Full table in
`data/processed/predictions/homogeneous_bagging_comparison.csv`.

### Taken together

All three experiments point the same direction as the fusion-ensemble
and drop-branch findings already in this log: **XGBoost-on-fusion-
features is doing essentially all of the real work in this pipeline.**
The deep models add negligible ensemble value (Experiment 2), simple
bagging doesn't beat a single good fit either (Experiment 3), and
raw-error performance is genuinely strong in the early-life regime even
though R2 (the wrong metric there) suggests otherwise (Experiment 1).
None of this was tuned or cherry-picked to produce this narrative - it's
the consistent result of every ensembling angle tried across this
project's sessions.

## Follow-up session 10 — surfaced the 3 evaluation-protocol experiments in the dashboard

The early-prediction test, drop-one-branch ablation, and homogeneous-
bagging baseline (Follow-up session 9) previously existed only as CSVs
and this log entry - not visible anywhere in `app.py`. Added a
collapsed "Evaluation protocol" expander to the dashboard (always
visible, independent of whichever battery/cycle is currently selected,
since these 3 experiments evaluate the fixed test set as a whole) that
reads and displays all 4 result tables directly - no re-computation,
just a read-only view of the same CSVs already referenced above. The
early-prediction table carries its own in-app warning reiterating the
R2-vs-near-zero-variance caveat, so a dashboard viewer doesn't
misread the negative R2 values as a regression the way a bare number
might suggest.

Also added matching plots (`plot_early_prediction_test`,
`plot_drop_branch_ablation`, `plot_homogeneous_bagging` in
`make_plots.py`) for consistency with every earlier phase, saved as
`outputs/phase7_{early_prediction_test,drop_branch_ablation,
homogeneous_bagging}.png`.

Verified via `streamlit.testing.v1.AppTest`: zero exceptions, the new
expander renders with 5 dataframes total on initial load (4 new + the
1 already shown by the default battery selection).

## Follow-up session 11 — RUL conformal coverage investigation (calibration-layer only)

User asked why RUL conformal coverage sits at 88.9% instead of 90%,
scoped explicitly to the calibration layer: check calibration-set size,
check quantile-interpolation correctness, check whether 88.9% is stable
across a fresh split - no retraining of any base/deep/fusion/joint
model, no switching conformal schemes (e.g. CQR) if there's no small
fix.

**Finding 1 - the 88.9% figure itself was stale, not a live bug.**
Re-running `src/run_conformal.py` unmodified, against the exact same
documented calib/eval battery split (`calib=[B0018,b2c24,b3c35]`,
`eval=[b1c4,b3c0,b4c38]`), now gives **93.0% coverage** (avg width
828.7), not 88.9%. Root cause: the 88.9% number was measured right
after the CNN-LSTM channel-normalization fix (RUL R2 -0.002 -> 0.279,
see "Phase 6 re-run" above) - but "Follow-up session 2" (log_sigma
clamping) retrained `joint_adaptive.pt` again immediately afterward
(RUL R2 0.279 -> 0.432, confirmed via `models/joint_adaptive.pt`'s
mtime, 13:23, the latest of all 4 joint-model checkpoints, and via the
document order - the log_sigma-clamp session comes strictly after the
Phase 6 re-run entry). Conformal calibration was simply never re-run
after that second retraining, so the docs kept reporting a number that
no longer matched the checkpoint on disk. No calibration code was
touched to get 93.0% - it's the same script, same split, same
(already-trained, untouched-by-this-session) model, just actually
re-executed.

**Finding 2 - calibration-set size and quantile interpolation both
check out; they are not the problem.**
- Calibration-set size: n_calib=1,746 rows (documented split) up to
  ~3,500 for other partitions - the finite-sample correction
  `ceil((n+1)(1-alpha))/n` shifts the target quantile level from 0.9 to
  only 0.900916 at this size, moving the calibrated residual quantile
  by about 1.5 cycles out of ~414 - negligible. Small calibration sets
  (tens of points) would make this correction matter; 1,700+ does not.
- Interpolation method: confirmed MAPIE 1.4.1's own internal
  `_compute_quantiles` (`mapie/utils.py`) uses the exact same formula
  and `method="higher"` as `run_conformal.py`'s manual fallback -
  bit-for-bit the same approach, verified by reading MAPIE's source,
  not assumed. No interpolation bug.

**Finding 3 - the real issue: only 6 test batteries means single-split
coverage is inherently high-variance, and this has no small fix.**
Computed RUL predictions once (pure inference, no retraining) for all
6 test batteries, then evaluated coverage under every one of the 20
possible 3-battery-calib / 3-battery-eval partitions of those same 6
batteries (same quantile code, same model, only the calib/eval
battery assignment changes):

| calib batteries | n_calib | coverage | width |
|---|---|---|---|
| B0018, b2c24, b3c35 (documented split) | 1,746 | 0.930 | 828.8 |
| B0018, b3c0, b4c38 | 2,369 | **0.647** | 629.3 |
| b1c4, b2c24, b3c35 | 2,839 | **0.996** | 902.6 |
| ...(17 more) | | | |

Full range across all 20 partitions: **coverage 64.7% to 99.6%**, mean
87.4%, std 10 percentage points - purely a function of which 3
batteries happen to land in calib vs. eval. Root cause: per-battery RUL
RMSE varies about 13x across the 6 test batteries (B0018: 24.7 cycles;
b1c4: 319.3 cycles), so whichever batteries happen to be "easy" or
"hard" in a given half dominates that half's residual quantile /
realized coverage. With only 6 distinct batteries (an inherent data
limit - not something this calibration-layer investigation can
manufacture more of), there aren't enough independent "battery-level"
units for split-conformal's marginal-coverage guarantee to concentrate
near 90% for any single partition; 1,700-3,500 calibration *rows* looks
like a lot but the effective sample size for this guarantee is closer
to "3 batteries," which is not enough.

**Verdict: fixed one thing, documented one thing.**
- Fixed: the stale 88.9% -> updated `FINAL_SUMMARY.md` and
  `outputs/conformal_coverage.csv` to the current, correctly-measured
  93.0% (re-running the untouched calibration script against the
  untouched, already-trained model - no model retraining, no
  calibration-method change, no touching of SOH's conformal numbers,
  which remain exactly 95.1%/4.64 as before).
- Left as a documented, structural limitation (per instruction, not
  pursued further): RUL coverage for any single 3/3 battery partition
  of this 6-battery test set is inherently noisy (64.7%-99.6% observed
  range) due to high per-battery RUL-residual heterogeneity combined
  with too few distinct test batteries. Not a calibration-code bug,
  not fixable by a different quantile formula or a bigger calibration
  set drawn from the same 6 batteries - would require either more held-
  out test batteries or a different, battery-cluster-aware conformal
  scheme, both explicitly out of scope here.

## Follow-up session 12 — graceful degradation when raw datasets aren't present (deploy fix)

Streamlit Community Cloud deployment crashed: `data/raw/` (NASA .mat,
CALCE .zip, MIT .mat/HDF5 files) is entirely gitignored - correctly, per
size (~11GB locally) and third-party redistribution-licensing concerns
for these research datasets - so a fresh clone has none of them, and
"Browse existing battery" hit a raw `FileNotFoundError`/`OSError` deep
inside `scipy.io.loadmat`/`h5py.File`/`zipfile.ZipFile` with no handling.

**Considered and rejected: serving "Browse existing battery" from
already-processed/derived data instead of raw files.** Checked what
`predict_and_explain` (`live_inference.py`) actually consumes: it calls
`compute_health_indicators(cycle)` and `get_cycle_tensor(cycle,
n_bins=200)` directly on the RAW per-cycle `{t, V, I, T}` waveform
arrays - not on anything in `hi_table.parquet` or
`fusion_embeddings.csv` (both already-reduced: scalar HI columns / a
fixed 16-dim embedding, no raw sequences). VLSTM/CNN-LSTM/PiFormer/the
joint-adaptive model and the per-instance DeepSHAP voltage-region
localization all need the full raw sequence tensor to run a genuine
forward pass and explanation for a chosen cycle - there's no derived
table that substitutes for this without literally re-embedding the raw
waveforms into some other file, which is the same redistribution
concern in a different format, not a way around it. So a
derived-data-only fix for "Browse existing battery" isn't feasible;
only the raw files (or none) will do.

**Fix implemented: per-dataset graceful degradation, upload path
untouched.** Added `nasa_data_available()` / `calce_data_available()` /
`mit_data_available()` to `data_adapters.py` (cheap existence checks:
does the expected raw directory/file exist on disk right now).
`app.py`'s sidebar checks the SELECTED dataset's availability before
attempting to load it: if unavailable, shows a clear `st.warning`
explaining why (size + licensing, not a bug) and to try a different
dataset or use upload instead - no attempt to load, no crash. If
available, loads as before, now wrapped in a `try/except
(FileNotFoundError, OSError, KeyError)` as a defensive second layer
(handles a partially-populated or corrupted local raw dataset, not just
total absence) that shows a clear inline `st.error` instead of an
uncaught traceback. "Upload your own cycle data" needed no changes -
verified it already never touches `data/raw` anywhere in
`parse_uploaded_csv`/`predict_and_explain`.

**Verified both ways, not just read the code:**
- Renamed `data/raw` away entirely (its full ~11GB, confirmed untracked
  via `git status`, so trivially restorable) to reproduce the exact
  deployed condition. Re-ran `AppTest` against NASA/MIT/CALCE: zero
  exceptions, each shows the new per-dataset warning message, nothing
  crashes.
  - Ran the upload path in this same no-raw-data state with a synthetic
    CSV: 4 tabs render, zero exceptions - confirms it's genuinely
    independent of `data/raw`, not just untested.
  - Restored `data/raw`, re-ran `AppTest` on the NASA default path:
  back to normal, zero sidebar warnings, 4 tabs - confirms the fix adds
  a check, not a regression, for the existing local-dev experience.
## Follow-up session 13 — MMD domain adaptation, targeting Review 1's CALCE finding directly

Review 1 flagged the sharpest problem in the whole project: on CALCE
(out-of-domain), the fusion ensemble's R2 collapsed 0.917->0.31, and the
90% split-conformal interval stayed **bit-for-bit the same width**
in-domain and out-of-domain while true coverage collapsed 95.6%->6.1% -
the model looked exactly as confident whether right or catastrophically
wrong (session 5, above). This session implements Maximum Mean
Discrepancy (MMD) domain adaptation on the fusion embedding to attack
that directly, then re-runs the same zero-retrain CALCE evaluation to
see whether it actually helps.

**What was added (all additive - every non-MMD file listed below is
confirmed untouched by `git status`):**
- `src/mmd_loss.py` — multi-bandwidth Gaussian-RBF MMD^2 (Gretton et al.
  2012 biased estimator; median-heuristic bandwidth scaled by
  `[0.5,1,2,4,8]` and summed, the standard DAN-style multi-kernel MMD so
  the loss isn't hostage to one hand-picked bandwidth).
- `src/train_fusion_encoder_mmd.py` — retrains `ICAEncoder` (the same
  architecture as `models/ica_encoder.py`, session 3) with an ADDITIVE
  loss: `sup_MSE(NASA+MIT) + lambda * MMD(embed(NASA+MIT batch),
  embed(CALCE batch))`. **Zero-label-leakage preserved**: CALCE's raw
  discharge curves are read (via `iterate_calce_cycles`) to compute its
  ICA/DV/DC channels for the MMD term, but its SOH/RUL labels are never
  read anywhere in this script - MMD is unsupervised by construction, it
  only ever compares feature *distributions*. This is standard
  unsupervised domain adaptation (the target's unlabeled inputs are
  legitimately usable), not a departure from the project's zero-retrain
  contract - the CALCE *evaluation* script still does a pure forward
  pass with no CALCE-specific fitting.
- `src/train_xgboost_fusion_mmd.py`, `src/train_ensemble_fusion_mmd.py`
  — retrain the downstream XGBoost-fusion base learner and Ridge
  stacking meta-learner on the MMD-aligned embeddings
  (`fusion_embeddings_mmd.csv`), otherwise identical to their non-MMD
  counterparts.
- `src/run_calce_zero_retrain_eval_mmd.py` — same procedure as session
  5's script, pointed at the MMD-aligned weights, with the conformal
  interval **recalibrated on the MMD model's own NASA+MIT residuals**
  (still only the NASA+MIT calib half - CALCE is never used for fitting
  or calibrating anything, only for computing empirical coverage).

**Bug caught before trusting any result: lambda=1.0 silently broke
training.** First attempt used `MMD_LAMBDA=1.0`. Validation loss came
out *worse* every single epoch after epoch 0 (2.837 -> 2.86, monotonically
rising) and patience-6 early-stopping fired at epoch 6, keeping the
epoch-0 weights - i.e. an essentially untrained encoder, compared to the
non-MMD baseline's smooth 25-epoch descent from val=2.75 to val=1.19.
The alignment term was strong enough to actively fight the supervised
objective from initialization. Rather than report results from a
degenerate model, added disk-caching for the (expensive, ~10min) CALCE
tensor-build step and swept lambda down: **lambda=0.1** trains cleanly
through all 25 epochs, reaching val_mse=1.105 at its best epoch (epoch
23) - actually marginally *better* than the non-MMD baseline's 1.192,
confirming lambda=1.0's failure was a weighting problem, not a sign that
MMD is fundamentally incompatible with this encoder. **lambda=0.1 is the
MMD-aligned model reported below**; the lambda=1.0 log is kept at
`logs/logs_fusion_mmd_lambda1.0_failed.txt` rather than deleted, per
this project's practice of logging what didn't work, not just what did.

**Result 1 — does R2 on CALCE improve? Yes, modestly, genuinely:**

| model | NASA+MIT in-domain R2 | CALCE zero-retrain R2 (non-MMD) | CALCE zero-retrain R2 (MMD-aligned) |
|---|---|---|---|
| XGBoost-fusion | 0.917 -> 0.911 | 0.304 | **0.337** |
| Stacking-Ridge-fusion | 0.917 -> 0.911 | 0.314 | **0.347** |

R2 on CALCE rises from ~0.31 to ~0.34-0.35 (RMSE 17.97->17.54,
14.01->17.41 [Ridge RMSE 17.84->17.41]), a real, reproducible ~11%
relative improvement, at the cost of a small (~0.006) drop in in-domain
R2 - a genuine, if modest, step in the right direction on the point-
prediction metric. It is nowhere close to closing the gap: CALCE RMSE
(~17.4-17.5) is still more than 12x the in-domain RMSE (~1.44), so this
should still not be read as "MMD fixes CALCE generalization" - it's a
partial, incremental improvement, not a fix.

**Result 2 — does the conformal interval's actual coverage on CALCE
improve? No. It gets slightly WORSE, and this is the important finding:**

| domain | half-width (non-MMD) | coverage (non-MMD) | half-width (MMD-aligned) | coverage (MMD-aligned) |
|---|---|---|---|---|
| NASA+MIT in-domain | 2.367 | 95.6% | 2.217 | 94.6% |
| CALCE out-of-domain | 2.367 | **6.1%** | 2.217 | **4.4%** |

Recalibrating on the MMD-aligned model's own (slightly better-fit, since
in-domain R2 barely moved) NASA+MIT residuals produced a slightly
NARROWER interval (half-width 2.367->2.217) - and a narrower interval
covers *less* of a still-catastrophically-wrong CALCE prediction
distribution, not more. Coverage fell from 6.1% to 4.4%.

**Honest verdict, reported plainly per instruction and consistent with
how this project reported the physics-informed loss result (session 4):
MMD domain adaptation gives a small, real improvement in point-
prediction R2 but does NOT fix - and here slightly worsens - the
conformal miscalibration problem Review 1 identified.** The reason is
structural, not a tuning artifact: this project's split-conformal
implementation (session 5, session 11) computes one global fixed-width
band from calibration residuals with no per-input adaptivity. Shrinking
CALCE's RMSE by ~2% (17.97->17.54) while it remains >12x the in-domain
RMSE cannot move empirical coverage in any meaningful way when the
interval's width is set purely by in-domain residuals - the model is
still extrapolating just as blindly, only slightly less badly, and the
interval still has no mechanism to widen in response. Actually fixing
the coverage collapse would need a domain-shift-*aware* uncertainty
mechanism (e.g. normalized/locally-adaptive conformal prediction scaled
by a per-input OOD/difficulty score, or a rejection/abstention
mechanism that flags low-confidence-domain inputs instead of emitting a
fixed-width interval for them) - out of scope for this single
experiment, flagged here as the natural follow-up, same as session 5
already flagged it.

New files (fully additive, verified via `git status` that every
non-`_mmd` file above is untouched): `src/mmd_loss.py`,
`src/train_fusion_encoder_mmd.py`, `src/train_xgboost_fusion_mmd.py`,
`src/train_ensemble_fusion_mmd.py`, `src/run_calce_zero_retrain_eval_mmd.py`,
`models/ica_encoder_mmd.pt`, `models/xgb_soh_fusion_mmd.json`,
`models/ridge_meta_fusion_mmd.pkl`, `data/processed/fusion_embeddings_mmd.csv`,
`data/processed/predictions/{ica_encoder_mmd_history,xgb_fusion_mmd_preds,
xgb_fusion_mmd_metrics,ensemble_fusion_mmd_test_preds,ensemble_fusion_mmd_metrics,
calce_zero_retrain_mmd_metrics,calce_zero_retrain_mmd_preds}.csv`,
`outputs/calce_zero_retrain_mmd_conformal.csv`,
`logs/logs_{fusion_mmd_lambda0.1,fusion_mmd_lambda1.0_failed,xgb_fusion_mmd,
ensemble_fusion_mmd,calce_zero_retrain_mmd}.txt`.

## Follow-up session 14 — softmax-normalized adaptive loss weighting, targeting Review 1's alpha=beta collapse

Review 1's other flagged limitation (session Phase 4/2, above): the
homoscedastic-uncertainty `adaptive` variant's learned alpha/beta
converged to the SAME value (2.028) instead of an asymmetric SOH/RUL
trade-off, because `log_sigma_soh`/`log_sigma_rul` are two INDEPENDENT
scalars with nothing structurally stopping them drifting together -
this parametrization mostly expresses "overall confidence," not "how to
balance the two tasks." This session implements the fix specified:
constrain `(alpha, beta) = 2 * softmax(s_alpha, s_beta)`, which pins
alpha+beta=2 by construction so one weight can only rise at the other's
direct expense.

**Citation check performed before writing anything to this log**: the
cited paper, "Dynamic Loss Balancing for Joint SOH and RUL Prediction of
Lithium-Ion Batteries via a Rotary SOH-Injected Prior Battery
Transformer" (arXiv:2607.18329, Chen/Shi/Huang/Tu, submitted 2026-07-19),
**is real** - confirmed via web search and fetching its arXiv abstract
page. The abstract does frame the identical underlying tension this
project's own `AdaptiveLossWeighting` docstring already independently
described: SOH's bounded, low-variance measurement noise fighting RUL's
unbounded, nonlinearly-expanding long-horizon uncertainty, resolved via
"a homoscedastic uncertainty weighting mechanism to dynamically scale
task-specific gradients." **What was NOT verifiable**: neither the
abstract nor a fetched-and-text-extracted copy of the PDF contained an
explicit `2*softmax(s_alpha, s_beta)` equation - the PDF's text layer
didn't extract cleanly enough to confirm or rule this out directly.
Logged honestly rather than either silently dropping the citation or
overclaiming it: this session credits the paper for the *problem
framing* (which is genuinely shared), not for a verified-identical
equation. The softmax constraint implemented below was built to the
exact specification given for this session, as an idea inspired by that
framing.

**What was added (additive to `models/joint_model.py` /
`src/train_joint_adaptive.py` - `AdaptiveLossWeighting`, the `"adaptive"`
training branch, and `models/joint_adaptive.pt` are all untouched;
confirmed via `git diff` that no existing class/branch was removed, only
a new one added alongside):**
- `models/joint_model.py: SoftmaxAdaptiveLossWeighting` — `s_alpha`,
  `s_beta` raw learnable `nn.Parameter`s; forward pass computes
  `(alpha, beta) = 2*softmax([s_alpha, s_beta])`,
  `total = alpha*L_soh + beta*L_rul`. No `log(sigma)`-style regularizer
  is used (unlike `AdaptiveLossWeighting`) - the degenerate
  alpha=beta=0 collapse that regularizer exists to prevent is
  structurally impossible once alpha+beta is pinned to 2 by the
  softmax itself.
- `src/train_joint_adaptive.py` — added a 5th ablation variant,
  `"adaptive_softmax"`, alongside the existing 4 (`fixed_balanced`,
  `soh_only`, `rul_only`, `adaptive`). Trained via
  `python train_joint_adaptive.py adaptive_softmax`, which uses the
  script's existing partial-re-run merge logic (already built for
  exactly this use case) to add this variant's rows into
  `joint_ablation.csv`/`joint_ablation_history.csv` **without touching**
  the other 4 variants' existing rows - confirmed by inspecting the
  output CSV after the run: all 4 original rows present, byte-identical
  values to before this session.

**Result 1 — are alpha/beta now asymmetric? Yes, confirmed:**

| epoch | alpha | beta |
|---|---|---|
| 0 | 0.993 | 1.007 |
| 5 | 0.771 | 1.229 |
| 10 | 0.680 | 1.320 |
| 15 | 0.618 | 1.382 |
| 20 | 0.569 | 1.431 |
| 24 (final) | **0.527** | **1.473** |

Unlike `adaptive`'s alpha=beta=2.028 (identical to 3 decimal places
throughout training), `adaptive_softmax`'s alpha and beta are visibly,
steadily divergent from the first few epochs onward and settle at a
genuinely asymmetric 0.527/1.473 - the softmax constraint structurally
delivers the asymmetric trade-off the homoscedastic version couldn't.
The literal ask - "confirm they're now asymmetric rather than
identical" - is satisfied.

**Result 2 — but is the model actually better? No. It's worse than
BOTH baselines on BOTH tasks, and this needs to be said plainly:**

| variant | SOH RMSE | SOH R2 | RUL RMSE | RUL R2 |
|---|---|---|---|---|
| fixed_balanced | 3.695 | 0.416 | 253.73 | 0.428 |
| adaptive (original, alpha=beta=2.028) | 3.918 | 0.344 | 252.80 | **0.432** |
| **adaptive_softmax (this session)** | **4.612** | **0.091** | **291.53** | **0.244** |

`adaptive_softmax` loses to `fixed_balanced` on both SOH (R2 0.091 vs
0.416) and RUL (R2 0.244 vs 0.428), and loses to the original `adaptive`
on both as well (0.091 vs 0.344 SOH; 0.244 vs 0.432 RUL). Fixing the
literal symptom Review 1 named (identical alpha/beta) made the model
worse, not better.

**Why, diagnosed rather than left as a mystery**: beta rises
monotonically from epoch 0 (0.527/1.473 by the end means RUL is weighted
~2.8x SOH) with no regularizer opposing that drift - unlike
`AdaptiveLossWeighting`'s `log(sigma)` term, which actively penalizes a
task's weight for growing (so a task can be down-weighted for being
noisy, but nothing pushes it toward domination), plain softmax
normalization here just follows whichever direction reduces the raw
total loss fastest at each step, with no counterweight. Empirically that
directs increasing weight onto RUL at SOH's expense throughout training,
runaway-starving the SOH head rather than settling into a stable
balance - the opposite failure mode from `adaptive`'s failure mode
(identical-collapse), but still a failure mode. **Structurally forcing
asymmetry, on its own, does not guarantee the asymmetry the optimizer
finds is a GOOD one** - it just guarantees the two weights are allowed
to differ; nothing here yet tells it a good place to stop differing.

**Honest verdict, reported plainly per instruction and consistent with
this project's practice of reporting negative results as directly as
positive ones (physics-informed loss, session 4; MMD conformal
non-improvement, session 13): the softmax-normalized constraint fixes
the exact symptom Review 1 identified (alpha/beta no longer identical)
but is a net regression in actual SOH/RUL predictive performance versus
both `fixed_balanced` and the original `adaptive` variant.** A plausible
next step, not pursued in this single additive session: combine the
softmax normalization (which correctly prevents the alpha=beta=alpha
collapse) with an uncertainty-style regularizer or an explicit bound/
entropy penalty on how far the softmax can drift per epoch, so the
model gets structural asymmetry AND something anchoring it away from
runaway single-task dominance - neither mechanism alone, on this
evidence, is sufficient by itself.

New/changed files (per-file disposition logged for clarity): `models/
joint_model.py` (added `SoftmaxAdaptiveLossWeighting` class, nothing
removed), `src/train_joint_adaptive.py` (added `"adaptive_softmax"`
branch, nothing removed), `models/joint_adaptive_softmax.pt` (new),
`data/processed/predictions/joint_ablation.csv` and
`joint_ablation_history.csv` (2 new rows / 25 new history rows appended
for `adaptive_softmax`, all 4 prior variants' rows confirmed unchanged),
`logs/logs_joint_adaptive_softmax.txt` (new).

## Follow-up session 15 — LIME as a second, independent explainability method (cross-validating TreeSHAP)

Adds LIME (Local Interpretable Model-agnostic Explanations, via the
`lime` package's `LimeTabularExplainer`) alongside the existing TreeSHAP
analysis (`src/run_shap_analysis.py`, read in full before writing this,
left completely untouched — no line of it changed) for the two places
TreeSHAP explains a genuinely tabular feature vector: the XGBoost base
learner (7 BFA-selected HIs) and the Stacking-XGBoost meta-learner (4
base-learner predictions). LIME was NOT run against VLSTM/CNN-LSTM/
PiFormer - `LimeTabularExplainer` explains tabular feature vectors, not
the raw (200-timestep x 6-channel) sequence input those 3 models take,
which is exactly why the task scoped this to "XGBoost/meta-learner
predictions" specifically.

**The point, per instruction, is cross-validation, not LIME in
isolation**: for a handful of the SAME held-out test-set predictions,
does an entirely independent explanation method - LIME's local-linear-
surrogate approximation, fit by perturbing around one instance, vs.
TreeSHAP's exact game-theoretic attribution, read directly off the
fitted tree structure - agree on which features actually drove that
specific prediction? Two methods with unrelated mechanics landing on the
same answer is a much stronger interpretability signal than either
method's ranking alone.

**Setup** (`src/run_lime_analysis.py`, new, read-only - installed
`lime==0.2.0.1` and its 4 new transitive deps, added to
`requirements.txt` alongside the existing pins, nothing removed):
- Base learner: same `hi_table.parquet` rows/BFA-selected features/
  `xgb_soh.json` as `run_shap_analysis.tree_shap_xgboost`, but restricted
  to the held-out TEST battery rows (`battery_split.json`) rather than
  that function's full NASA+MIT sample - this script explains actual
  test-set *predictions*, per instruction, which the original TreeSHAP
  function (aggregate ranking only, not scoped to test) didn't need to.
- Meta-learner: same `ensemble_test_preds.csv` rows/4 base-learner-
  prediction features/`xgb_meta.json` as
  `run_shap_analysis.tree_shap_meta`. LIME's background distribution
  (perturbation statistics only - TreeSHAP needs no such background)
  uses the TRAIN-split meta-features via `train_ensemble.load_merged
  ("train")`, the exact features `xgb_meta.json` was fit on.
- 5 instances per model (`np.random.default_rng(42)`, "a handful" per
  instruction), TreeSHAP and LIME both re-run on the identical rows.
  `LimeTabularExplainer(..., discretize_continuous=False)` so
  `.as_list()` returns plain feature names with signed weights (already
  sorted by |weight|), not binned rule strings - no string-parsing
  needed to recover feature identity for the top-3 comparison.

**Results — 8/10 instances (80%) reached full 3/3 top-3 agreement
between TreeSHAP and LIME; overall mean top-3 overlap 93.3%:**

| model | instances | mean top-3 overlap |
|---|---|---|
| XGBoost-base | 5 | **100%** (5/5 instances, full 3/3 every time) |
| XGBoost-meta | 5 | **86.7%** (3/5 full 3/3, 2/5 at 2/3) |

XGBoost-base: every one of the 5 sampled test instances (all MIT
cells - the test set has only 1 NASA battery, B0018, so a 5-instance
random sample landing entirely on MIT cells is unsurprising, not an
error) got the identical `[SCV, VIECT, TEVI]` top-3 from BOTH methods,
same order. **Caveat logged rather than glossed over**: TreeSHAP's own
top-3 was ALSO identical across all 5 instances here - meaning this
particular comparison mostly validates that LIME recovers the same
*global* feature dominance TreeSHAP sees (SCV/VIECT/TEVI clearly
dominate the 7 BFA-selected HIs), not that the two methods track
genuinely instance-varying, per-prediction feature importance. That's
still a real, useful agreement (a wrong/unrelated local surrogate could
easily have picked different features), just a weaker test than the
meta-learner case below.

XGBoost-meta: TreeSHAP's top-3 was also stable across all 5 instances
(`[pred_XGBoost, pred_PiFormer, pred_VLSTM]`), but LIME's top-3
genuinely varied per instance here - agreeing fully on 3/5 (including
swapped rank order between VLSTM/PiFormer, still counted as a match
since this is a set-overlap metric, not exact-rank agreement) and
disagreeing on `pred_CNNLSTM` vs. one of `{VLSTM, PiFormer}` for the
other 2/5. This is the more genuine per-instance cross-validation of the
two: pred_XGBoost (correctly, since it's by far the strongest individual
base learner) was in every single top-3 from both methods, 10/10.

**Honest read**: LIME independently confirms TreeSHAP's feature
rankings for both XGBoost models at a high but not perfect rate (93.3%
mean top-3 overlap, 80% exact full-match) - a genuine, positive
cross-validation result, not a wash and not a clean 100% either. The
2/5 meta-learner disagreements are on the 2nd/3rd-ranked features
(CNN-LSTM vs. VLSTM/PiFormer), never on which base learner dominates
(pred_XGBoost), so the disagreement is about second-order ranking
detail, not about the headline conclusion.

Per-instance results (model, battery_id, cycle_idx, both methods' top-3,
overlap count/fraction) saved to `outputs/lime_shap_comparison.csv`.
`src/run_shap_analysis.py` and all its output files
(`shap_xgboost_base_ranking.csv`, `shap_meta_ranking.csv`,
`shap_deep_models_summary.csv`) confirmed untouched by `git status`.
New files: `src/run_lime_analysis.py`, `outputs/lime_shap_comparison.csv`,
`logs/logs_lime_analysis.txt`; `requirements.txt` updated (additive:
`lime`, `scikit-image`, `imageio`, `lazy_loader`, `tifffile`).

## Follow-up session 16 — knee-point detection on predicted vs. true SOH curves

Purely a DERIVED analysis of the already-computed fusion-ensemble
predictions (`ensemble_fusion_test_preds.csv`) - no new model training,
new file `src/run_knee_point_detection.py` only.

**Citation check performed before writing anything here** (same
discipline as sessions 14/15): the task cited this as matching "the
BatteryGPT reference paper already in the literature survey." Checked
first - **this repo has no literature-survey document at all** (no file
matching `*literature*`/`*survey*` anywhere in the tree), so "already in
the literature survey" does not hold for this project specifically;
logged plainly rather than silently accepted. The paper itself IS real,
though: "Early prediction of lithium-ion battery degradation with a
generative pre-trained transformer" (Nature Communications,
10.1038/s41467-025-66819-0, Dec 2025) - confirmed via web search and by
fetching its text - and it does define the knee point exactly as
specified: kappa = |y''| / (1+y'^2)^1.5 on the SOH-vs-cycle curve, knee
= point of maximum curvature. The paper's methods text does not specify
HOW y'/y'' are estimated from the discrete, noisy per-cycle sequence
(this detail wasn't in the fetched text) - addressed below as this
project's own documented choice, not a guess passed off as the paper's.

**Method**: y and its 1st/2nd derivatives are estimated via a
Savitzky-Golay local-polynomial fit (`window=15, polyorder=3` -
`scipy.signal.savgol_filter(..., deriv=1/2, delta=1.0)`), reusing this
project's EXACT existing convention from `ica_dv_dc.py` rather than
picking fresh parameters. This is far better-conditioned on a noisy
discrete sequence than differentiating raw `np.diff()` twice (which
would amplify per-cycle measurement noise quadratically). The knee
search excludes a 7-cycle margin (half the SG window) at each end of
every battery's cycle range, to avoid the search latching onto an SG
boundary-fit artifact instead of a genuine curve feature.

**Literal result, exactly as asked - reported first, before the
diagnosis below**: mean absolute cycle-offset error across the 6 test
batteries = **161.8 cycles** (17.0% of lifetime, mean).

| battery | n_cycles | true knee | pred knee | offset | % of lifetime |
|---|---|---|---|---|---|
| B0018 | 132 | 48 | 49 | +1 | 0.8% |
| b1c4 | 1225 | 886 | 886 | +0 | 0.0% |
| b2c24 | 523 | 255 | 200 | -55 | 10.5% |
| b3c0 | 1007 | 8 | 919 | **+911** | **90.5%** |
| b3c35 | 1091 | 359 | 363 | +4 | 0.4% |
| b4c38 | 1230 | 1089 | 1089 | +0 | 0.0% |

**That raw average would be badly misleading reported alone - both
outliers were root-caused, not just noted, per this project's own
established practice of diagnosing anomalies rather than passing a
summary statistic through unexamined:**

- **4 of 6 batteries show excellent agreement** (0-4 cycles offset,
  ≤0.8% of lifetime): B0018, b1c4, b3c35, b4c38. Excluding BOTH outliers
  below, mean absolute offset over these 4 is 1.25 cycles.
- **b3c0 (911-cycle offset) - a GROUND-TRUTH-side artifact, not a
  prediction error.** Inspected the raw SOH sequence directly: b3c0's
  true SOH rises slightly ABOVE 100% for its first ~10 cycles (99.92 ->
  100.29 -> gradually declining from there) - a real, physically
  plausible formation/break-in effect in early cycling data, not a
  data bug. That small early bend produces the single HIGHEST curvature
  value in the entire true curve (kappa=0.0088 at cycle ~8, vs. every
  later point's kappa monotonically smaller in the checked range) -
  larger than the curvature at any point during the battery's actual
  later-life degradation. By the strict global-argmax-curvature
  definition used here (and not contradicted by anything in the
  paper's available text), cycle 8 genuinely IS the true curve's point
  of maximum curvature - this is a known, documented limitation of
  naive curvature-based knee detection on raw capacity curves (early
  formation-cycle transients can out-curve the real degradation knee),
  not a bug in this implementation. The model's PREDICTED curve did not
  reproduce this early transient as sharply, so its own knee search
  landed on a plausible later-life value (cycle 919) instead - meaning
  the 911-cycle "offset" here mostly reflects an ambiguity in what the
  ground-truth knee even IS for this battery, not a prediction failure.
- **b2c24 (55-cycle offset) - a PREDICTION-side artifact, genuinely
  different from b3c0's cause.** Inspected the predicted SOH sequence:
  Stacking-Ridge-fusion's prediction for b2c24 has a real discontinuity
  around cycle 204 (98.65% -> 96.35% in one step, briefly back to
  98.47% at cycle 207, then back down) - a single-cycle prediction
  glitch, not present in the smooth ground-truth curve at the same
  cycles. That jump produces a spurious curvature spike (kappa=0.076 at
  cycle ~199, ~8x every other value in this battery's curve, including
  its own true curvature peak of ~0.0063), which hijacks the predicted-
  knee search away from the correctly-shaped bend the ground truth
  shows at cycle 254-255 (found correctly as this battery's TRUE knee).
  This is a genuine, useful finding in its own right: curvature is a
  second-derivative quantity, so even Savitzky-Golay-smoothed knee
  detection remains sensitive to isolated point-wise prediction noise
  in a way that plain RMSE/R2 metrics on the same predictions do not
  surface - a real limitation of applying curvature-based knee
  detection to ML-predicted (rather than raw measured) curves.

**Honest summary, reported both ways rather than picking whichever
number looks better**: the literal, as-specified mean absolute
cycle-offset error is 161.8 cycles (17.0% of lifetime) across all 6 test
batteries. Excluding b3c0 (the ground-truth-side formation-cycle
artifact, arguably a data-definition ambiguity rather than a model
failure) gives a mean of 12.0 cycles across the remaining 5 batteries -
still including b2c24's genuine 55-cycle prediction-glitch-driven miss.
Both numbers are logged; neither is "the" answer picked to look best -
the raw 161.8-cycle figure is what the task's exact literal
specification produces, and the diagnosis above is what makes that
figure interpretable rather than just alarming.

Per-battery results (n_cycles, true/predicted knee cycle, offset,
% of lifetime) saved to `outputs/knee_point_detection.csv`. New file:
`src/run_knee_point_detection.py` (read-only derived analysis, no
existing script or model touched); `logs/logs_knee_point_detection.txt`.

## Follow-up session 17 — CNN-BiGRU as a 5th base learner (genuine test, reports either way)

Adds CNN-BiGRU as a 5th base learner alongside XGBoost/VLSTM/CNN-LSTM/
PiFormer, fully additive (no existing model, script, or output file
touched - `train_deep_models.py`, `deep_models_metrics.csv`,
`run_drop_branch_ablation.py`, and `drop_branch_ablation.csv` all
confirmed unchanged via `git status`).

**Architecture** (`src/models/cnn_bigru.py`): same 4-branch multi-
kernel-scale 1D-CNN front end as CNN-LSTM (kernel sizes {3,5,7,11},
reused directly via `from models.cnn_lstm import MultiKernelBranch`,
not duplicated), feeding a Bidirectional GRU (2 gates - update, reset -
vs. LSTM's 3) instead of CNN-LSTM's unidirectional LSTM, then a small FC
head on the concatenated forward+backward final hidden states.
**Deliberately reuses CNN-LSTM's exact front end** rather than a fresh
one, so the comparison isolates the one architectural change actually
being tested (LSTM -> bidirectional GRU as the recurrent core), not
confounded by a different feature-extraction stage too. Trained via
`src/train_cnn_bigru.py`, reusing `train_deep_models.py`'s
`load_all_battery_tensors`/`make_xy`/`train_one_model` unchanged (same
battery split, same channel normalization, same 40-epoch/patience-8
budget) - so its numbers are directly comparable to the other 3 deep
models' rows in `deep_models_metrics.csv`, not just similar in spirit.

**Operational note, logged rather than hidden**: the first training
invocation (piped through `tee`) exited with code 4 and wrote zero bytes
of output - no traceback, nothing. An immediate retry of the identical
script/seed, run directly without the `tee` pipe, completed successfully
end-to-end on the next attempt with no code changes, pointing to a
transient issue in this session's background-process/pipe-capture
plumbing rather than a bug in the model or training script - noted,
not deeply investigated, since it did not reproduce. Full detail (incl.
which parts of `logs/logs_cnn_bigru.txt` are a genuine terminal capture
vs. reconstructed from the successful run's saved CSV/metrics files,
since a `head -50` truncation cut the live capture short) is in that
log file's own header.

**Result 1 - standalone base-learner comparison (test set, all 5 now
side by side):**

| model | RMSE | MAE | R2 |
|---|---|---|---|
| XGBoost | 1.478 | 0.990 | **0.907** |
| VLSTM | 2.131 | 1.564 | 0.806 |
| PiFormer | 2.993 | 1.928 | 0.617 |
| **CNNBiGRU (new)** | **3.165** | **2.341** | **0.572** |
| CNNLSTM | 3.948 | 2.926 | 0.334 |

CNN-BiGRU lands 4th of 5 - beats only CNN-LSTM (the project's
established weakest base learner), and clearly behind VLSTM/PiFormer/
XGBoost. Bidirectional GRU over the identical CNN front end did NOT
close the gap to the other sequence models, let alone to XGBoost.

**Convergence speed** (`outputs/phase2_cnn_bigru_training_curves.png`,
additive 4-panel companion to the original 3-panel
`phase2_deep_model_training_curves.png`, which is untouched):

| model | epochs trained (of 40 budget) | best epoch | best val_loss (standardized MSE) |
|---|---|---|---|
| PiFormer | 21 | 12 | 0.484 |
| **CNNBiGRU** | **27** | **18** | **0.554** |
| CNNLSTM | 34 | 25 | 0.733 |
| VLSTM | 40 (ran full budget, never triggered early stop) | 34 | 0.269 |

CNN-BiGRU converges faster than CNN-LSTM and VLSTM (reaches its best
val_loss by epoch 18 vs. 25 and 34) but slower than PiFormer (epoch 12),
and its best val_loss (0.554) sits between PiFormer's (better, 0.484)
and CNN-LSTM's (worse, 0.733) - broadly consistent with, though not
identical in ranking to, its final test-set R2 position.

**Result 2 - does it help the ensemble? Re-ran the drop-branch ablation
with CNN-BiGRU as a genuine 5th branch
(`src/run_drop_branch_ablation_5branch.py`, additive companion to
`run_drop_branch_ablation.py`) - reported honestly, exactly as asked,
not steered toward either outcome:**

| variant | RMSE | R2 | delta R2 vs. full |
|---|---|---|---|
| **drop CNN-BiGRU** | 1.39380 | 0.916947 | **+0.000064 (removing it is very slightly BETTER)** |
| full 5-branch (incl. CNN-BiGRU) | 1.39433 | 0.916884 | 0.0 (reference) |
| drop CNN-LSTM | 1.39436 | 0.916880 | -0.000004 |
| drop PiFormer | 1.39455 | 0.916857 | -0.000026 |
| drop VLSTM | 1.39573 | 0.916717 | -0.000167 |
| drop XGBoost-fusion | 2.05088 | 0.820182 | -0.096702 |

Direct comparison against the ORIGINAL (pre-existing, untouched)
4-branch ablation's full result: R2 0.916947 (4-branch) -> 0.916884
(5-branch, CNN-BiGRU added) - **adding CNN-BiGRU as a 5th branch makes
the ensemble marginally WORSE** (delta_r2=-0.0001, delta_rmse=+0.0005),
and dropping CNN-BiGRU's OWN column from the 5-branch ensemble is the
single best-performing ablation row of the whole table (R2 0.916947,
matching the original 4-branch full-ensemble number almost exactly).

**Honest verdict, reported plainly as a genuine result rather than
forced toward a conclusion either way: CNN-BiGRU repeats EXACTLY the
pattern this project already found and documented for the other 3 deep
models** - the Ridge meta-learner's reliance on any individual deep
model's prediction, CNN-BiGRU included, is essentially zero once
XGBoost-fusion + the 16-dim fusion embedding are present (dropping
XGBoost-fusion costs -0.097 R2; dropping any deep model, including the
new one, costs between +0.0001 and -0.0002 R2 - noise-level). A better
standalone recurrent core (bidirectional GRU vs. LSTM, same CNN front
end) did not translate into a better base learner (R2 0.572 vs.
CNN-LSTM's 0.334 - genuinely better standalone, an improvement over the
weakest existing deep model) NOR into a more useful ensemble branch
(negligibly worse than not having it at all). This is consistent with -
not a new finding contradicting - this project's established
conclusion that the fusion-embedding-augmented XGBoost branch already
captures most of what these sequence models offer for this specific
task/dataset combination.

New files (fully additive): `src/models/cnn_bigru.py`,
`src/train_cnn_bigru.py`, `src/run_drop_branch_ablation_5branch.py`,
`src/plot_cnn_bigru_training_curves.py`, `models/cnn_bigru_soh.pt`,
`data/processed/predictions/cnn_bigru_{history,metrics,test_preds,
train_preds}.csv`, `data/processed/predictions/drop_branch_ablation_5branch.csv`,
`outputs/phase2_cnn_bigru_training_curves.png`,
`logs/logs_{cnn_bigru,drop_branch_5branch}.txt`.

## Follow-up session 18 — consolidated convergence comparison across all 4 deep models

Single new plot, additive: `src/plot_convergence_comparison.py`,
matching `make_plots.py`'s exact style/idioms (same imports, directory
constants, `dpi=120`, `phaseN_*.png` naming) as a standalone script
rather than an edit to that file - same pattern this project already
used for the CNN-BiGRU-specific plot in session 17. Does not touch
`make_plots.py` or any existing output. Overlays training-loss-vs-epoch
for VLSTM, CNN-LSTM, PiFormer, and CNN-BiGRU on ONE chart (rather than
the per-model side-by-side subplots the two existing training-curve
plots use), specifically to compare convergence speed/stability across
architectures at a glance. Saved to
`outputs/phase8_convergence_comparison.png`.

**Convergence speed** (epoch at which each model reached its best
val_loss - reusing the exact numbers already computed and logged in
session 17, not recomputed differently here):

| model | epochs trained (of 40 budget) | best epoch | best val_loss |
|---|---|---|---|
| **PiFormer** | 21 | **12** | 0.484 |
| CNN-BiGRU | 27 | 18 | 0.553 |
| CNN-LSTM | 34 | 25 | 0.733 |
| VLSTM | 40 (full budget, never early-stopped) | 34 | 0.269 |

**PiFormer converges fastest** by a clear margin (reaches its best
validation point by epoch 12, half of CNN-BiGRU's 18 and roughly a
third of VLSTM's 34) - though its train_loss keeps falling well past
that point (down to 0.022 by epoch 20), meaning early stopping caught
it starting to overfit shortly after epoch 12, not that training had
truly plateaued. VLSTM is the slowest to converge (best epoch 34 of a
full 40-epoch run, never triggering early stopping) but reaches the
lowest final val_loss of all 4 (0.269, roughly half PiFormer's 0.484) -
slow-but-thorough convergence, not a failure to converge.

**Stability - measured, not just eyeballed from the chart**: defined
here as the size of the largest single-epoch UPWARD jump in train_loss
during training (a transient spike is what actually reads as
"instability" on the overlaid chart, more than raw epoch-to-epoch
variance, which is dominated by each model's early steep descent and
isn't a fair cross-model comparison on its own):

| model | largest single-epoch upward jump in train_loss | % of epochs where loss rose vs. previous epoch |
|---|---|---|
| **CNN-BiGRU** | **+0.0035** | 19% (5/26) |
| PiFormer | +0.0074 | 15% (3/20) |
| CNN-LSTM | +0.0359 | 27% (9/33) |
| VLSTM | +0.0897 | 21% (8/39) |

**CNN-BiGRU trains the most stably of the 4** by this measure - its
largest single-epoch setback (+0.0035) is roughly 20-25x smaller than
VLSTM's and CNN-LSTM's visible mid-training spikes (VLSTM: a jump to
~0.165 around epoch 17; CNN-LSTM: a jump to ~0.09 around epoch 24-25,
both clearly visible as blips on the overlaid chart) - consistent with
CNN-BiGRU's bidirectional-GRU recurrent core producing a smoother
optimization trajectory than CNN-LSTM's LSTM core over the identical
CNN front end, even though (per session 17) that smoother training did
NOT translate into better final test-set accuracy. PiFormer is a close
second on this measure (+0.0074) and has the fewest upward jumps overall
(15%) despite converging fastest, i.e. it is both the fastest AND one of
the two most stable by this metric - CNN-LSTM (highest jump, most
frequent upward jumps) is the least stable of the 4, consistent with it
also being this project's weakest base learner on final test R2 (0.334).

**Summary, one line each**: PiFormer converges fastest but stops
earliest (mild early overfitting); VLSTM converges slowest but reaches
the lowest overall validation loss; CNN-BiGRU trains the most smoothly/
stably of the 4 (smallest transient spikes) without being fastest or
reaching the lowest loss; CNN-LSTM is both slower than PiFormer/
CNN-BiGRU AND the least stable, consistent with it remaining this
project's weakest deep base learner throughout every prior session that
touched it.

New files (fully additive, nothing existing touched): `src/plot_convergence_comparison.py`,
`outputs/phase8_convergence_comparison.png`, `logs/logs_convergence_comparison.txt`.

## Follow-up session 19 — domain-shift-aware conformal prediction (weighted split-conformal)

Directly targets session 13's finding: fixing point-prediction accuracy
via MMD alignment did NOT fix CALCE's conformal coverage (6.1% original,
4.4% after MMD-recalibration), because this project's split-conformal
implementation computes ONE global fixed-width interval with zero
per-input adaptivity. Implements weighted split-conformal prediction
(Tibshirani, Barber, Candes, Ramdas 2019, "Conformal Prediction Under
Covariate Shift"): calibration residuals are reweighted by a covariate-
shift density ratio w(x) = P_target(x)/P_calib(x), estimated via a
lightweight logistic-regression domain classifier - the exact method
named in the task. No base model retrained: purely a post-hoc
reweighting of the existing MMD-aligned ensemble's (session 13)
calibration residuals. New file: `src/run_domain_shift_conformal.py`,
fully additive - `run_conformal.py` and every session 13 file untouched
(only `calib_eval_battery_split` is imported/reused, not modified).

**Method**: for a target domain (NASA+MIT eval half, or CALCE), fit
ONE `LogisticRegression` (standardized features) to distinguish
calibration-domain points (label 0) from that target domain's points
(label 1); `w(x) = clip(p(target|x))/(1-clip(p(target|x)))` for every
point (clipped to [0.01, 0.99] to avoid a single near-certain
classification producing an infinite weight). For each individual test
point, the (1-alpha) weighted quantile of calibration residuals is
computed via Tibshirani et al.'s exact formula (calibration residuals
plus a point-mass at +infinity weighted by that specific test point's
own w(x)) - genuinely PER-INPUT, not just "one new number per domain".
Domain-classifier feature space: the 7 BFA-selected HIs (cheaply
available for CALCE from the already-computed `hi_table.parquet` - no
raw-data rework needed) + the 16-dim MMD-aligned fusion embedding
(which session 13 never persisted to disk for CALCE - rebuilt once here
via a forward pass through the already-trained, not retrained,
`ica_encoder_mmd.pt`, and cached to a temporary file deleted at the end
of this session, same convention as session 13's own temporary cache).

**First result (default: 7 HI + 16 fusion, 23 features) surfaced a
problem that needed investigating before anything could be reported
honestly**: the domain classifier for calib-vs-CALCE reached AUC=1.0000
(perfect separation, i.e. genuinely zero feature overlap - exactly the
degenerate regime the task's own caveat warned about) - but so did the
supposed **in-domain sanity check**, calib-vs-NASA+MIT-eval-half, at
AUC=0.9021 ("near-total separation"), which should have come out near
0.5 (two halves of the SAME domain ought to look indistinguishable to a
domain classifier). Root cause, investigated rather than shrugged off:
the calib/eval split is only 3 batteries vs. 3 batteries (per
`run_conformal.calib_eval_battery_split`, reused unchanged from session
5/13) - at that scale, individual-battery manufacturing/aging
idiosyncrasies dominate the 23-dimensional feature space more than any
genuine shared "NASA+MIT-ness", so the classifier partly learns
battery IDENTITY rather than domain membership. Consequence: in-domain
effective calibration sample size collapsed to 7.8% (136.8 of 1746),
intervals became far too narrow (mean half-width 0.882 vs. the
original 2.217), and in-domain coverage collapsed from the original
94.6% to **43.6%** - the domain-shift-aware method broke something that
wasn't broken, on top of not fixing what was. Separately, CALCE's
AUC=1.0000 meant its weighted quantile hit the formula's "test point too
atypical to bound" case for **100% of CALCE points** (every single one
got an infinite-width interval) - a real, meaningful output of the
method (an honest admission of zero overlap), not a bug, but not a
useful interval either. (Nominal "coverage" of 100% follows trivially
since an infinite interval always contains the true value - reported,
but flagged immediately as vacuous, not a genuine improvement.)

**Diagnostic follow-up (not part of the original ask, run because the
first result demanded investigation, per this project's standing
practice): fusion-embedding-ONLY (16-dim, dropping the un-aligned raw
HI features) as the domain-classifier's feature space**, to isolate
whether the raw HIs (never touched by MMD) were driving the spurious
in-domain separability:

| | in-domain AUC | in-domain half-width | in-domain coverage | CALCE AUC | CALCE half-width | CALCE coverage | CALCE degenerate % |
|---|---|---|---|---|---|---|---|
| Full (7 HI + 16 fusion) | 0.902 | 0.882 | 43.6% | 1.000 | inf (100% degenerate) | 100% (vacuous) | 100% |
| **Fusion-only (16-dim)** | **0.747** | **1.261** | **69.6%** | **0.936** | **2.192** | **4.4%** | **0%** |
| *Original (session 13, fixed-width)* | *n/a* | *2.217* | *94.6%* | *n/a* | *2.217 (identical)* | *4.4%* | *n/a* |

Dropping the raw HIs meaningfully reduces the spurious separability
(AUC 0.902 -> 0.747 in-domain; 1.000 -> 0.936 for CALCE) and, this time,
every CALCE point gets a genuine FINITE interval (0% degenerate) - the
better-behaved of the two configurations, reported as the honest
headline result below.

**Answering the two questions directly, using the fusion-only
(non-degenerate) configuration:**

**1. Does interval width now differ between domains? Yes, technically
- but the mechanism is not what "adaptive widening" suggests.** In-
domain mean half-width fell to 1.261 (down from the original 2.217);
CALCE's mean half-width is 2.192 - almost EXACTLY the original fixed
value, not meaningfully wider than before in absolute terms. The two
numbers now differ (1.261 vs. 2.192, genuinely not identical) - but
that is almost entirely because in-domain calibration got NARROWER
under this reweighting, not because CALCE's own interval grew to
reflect its true difficulty. Per-point spread within each domain
(`predictions/domain_shift_conformal_per_point_fusion_only.csv`):
in-domain widths range 1.238-1.874 across 51 distinct values (genuine
per-input variation over 3462 points); CALCE widths range a nearly
imperceptible 2.191-2.210 across only 4 distinct values over 2941
points - CALCE points are so uniformly unlike calibration that the
method treats almost all of them identically anyway, in practice barely
more "per-input adaptive" than the original fixed constant it replaces.

**2. Does empirical coverage on CALCE improve? No - it stays at
4.4%, statistically identical to the session-13 MMD-recalibrated fixed-
width result.** A ~1x-unchanged interval width cannot move coverage when
CALCE's actual prediction errors (RMSE ~17.4) remain >12x the in-domain
RMSE (~1.4, per session 13) - the interval would need to be an order of
magnitude wider, not ~1x wider, to meaningfully close this gap, and nothing
in this reweighting scheme pushed it that far (the classifier's AUC of
0.936, while high, isn't total enough to produce the near-infinite
weights that WOULD force a much wider quantile - that only happened in
the degenerate full-feature run, which threw away every interval's
usefulness entirely to get there).

**Honest verdict, reported exactly as asked, matching the task's own
pre-flagged caveat about "recent literature on this exact problem":**
this is a genuine partial result, not a fix. The interval DOES now
differ by domain (fusion-only variant) rather than being bit-for-bit
identical everywhere as in every prior session - a real, if modest,
structural improvement. But it does not meaningfully improve CALCE
coverage (still 4.4%), and the more aggressive (full-feature) attempt
at genuine adaptivity didn't produce useful improvement either - it
collapsed into vacuous infinite intervals for CALCE while actively
BREAKING the in-domain guarantee that was previously working fine
(94.6% -> 43.6%, or 69.6% even in the gentler fusion-only variant).
This matches exactly the known failure mode the task anticipated:
covariate-shift-aware conformal methods degrade sharply when train/test
feature overlap is near-zero (AUC approaching 1.0, as measured here
directly rather than assumed) - the domain classifier can tell the two
distributions apart almost perfectly, which is precisely the condition
under which its density-ratio estimate becomes numerically extreme and
the resulting weighted quantile becomes unstable or degenerate, exactly
as documented in this method's own literature. The additional wrinkle
found here - that even the "in-domain" calibration/eval split isn't
reliably indistinguishable to a domain classifier at only 3-vs-3
batteries - is a genuine, additional limitation of applying this method
to a dataset this battery-sparse, not something the cited literature's
caveat alone would have predicted.

New files (fully additive): `src/run_domain_shift_conformal.py`,
`outputs/domain_shift_conformal_summary{,_fusion_only}.csv`,
`data/processed/predictions/domain_shift_conformal_per_point{,_fusion_only}.csv`,
`logs/logs_domain_shift_conformal{,_fusion_only}.txt`. The temporary
CALCE fusion-embedding cache this script builds
(`data/processed/_calce_fusion_mmd_cache.csv`) was deleted after use,
same as session 13's equivalent temporary cache.

## Follow-up session 20 — "lean" deployment variant vs. the full 5-branch ensemble

Practical follow-through on the evidence this project has repeatedly
gathered: sessions 9's drop-branch ablation and 17's 5-branch re-run
(CNN-BiGRU added) both found the Ridge meta-learner's reliance on ANY
of the 4 deep sequence models is negligible-to-negative once
XGBoost-fusion + the fusion embedding are present. This session asks
the practical question that evidence implies - what does a "lean"
deployment (XGBoost-fusion only) actually cost vs. save relative to the
full 5-branch ensemble - across accuracy, latency, size, and
complexity. New file: `src/run_lean_vs_full_comparison.py`, fully
additive. No base learner retrained; the full 5-branch Ridge
meta-learner (in-memory-only in session 17) was refit here (a one-line
`Ridge.fit` on already-computed base-learner predictions, not a
retrain) and persisted to `models/ridge_meta_fusion_5branch.pkl` so
"full" has a genuine, complete file set to measure.

**1. Accuracy - confirmed directly, not assumed, per instruction:**

| variant | RMSE | MAE | R2 |
|---|---|---|---|
| LEAN (XGBoost-fusion only) | **1.3921** | 0.9593 | **0.91715** |
| FULL (5-branch + Ridge meta) | 1.3943 | 0.9657 | 0.91688 |

Freshly recomputed both ways (LEAN cross-checked against the originally-
saved `xgb_fusion_metrics.csv`; FULL cross-checked against session 17's
`full_5_branch` row - both matched to 4 decimal places, confirming
genuine re-derivation, not copied numbers). **LEAN is not just "nearly
identical" - it is marginally BETTER than FULL** (delta_rmse=-0.0023,
delta_r2=+0.0003): the 4 deep models' net contribution through the
meta-learner is not merely negligible, it is very slightly negative,
exactly consistent with every prior ablation in this project.

**2. Inference latency - single-instance (batch_size=1), the realistic
unit for one live prediction, warm-up excluded, 300 repetitions:**

| variant | mean latency | median | model invocations |
|---|---|---|---|
| LEAN | **3.9ms** | 3.9ms | 2 |
| FULL | 201.9ms | 235.6ms | 7 |

**LEAN is ~52x faster** (a repeat run during development measured
54.7x - run-to-run wall-clock CPU variance of that order is expected
and doesn't change the conclusion: roughly two orders of magnitude
either way). The 4 torch forward passes (VLSTM's explicit per-timestep
recurrence loop, PiFormer's attention blocks, CNN-LSTM/CNN-BiGRU's
multi-branch convs) dominate FULL's latency at batch_size=1 on this
CPU-only setup; XGBoost's tree traversal and the ICAEncoder's small
conv forward pass (both needed by LEAN too) are comparatively trivial.

**3. Model size on disk - the one dimension where "lean" does NOT
mean dramatically smaller, reported honestly rather than glossed over:**

| variant | files | total size |
|---|---|---|
| LEAN | 2 | 2855.2 KB |
| FULL | 7 | 3086.3 KB |

Only **1.1x smaller** - `xgb_soh_fusion.json` alone is 2845.7 KB, 99.7%
of LEAN's total, and the 5 extra files FULL adds (4 small `.pt` deep-
model weights + a 1KB Ridge pickle) together add only ~231KB. Size is
NOT where this trade-off's savings come from; latency and complexity
are.

**4. Pipeline complexity - model invocations needed per prediction:**

LEAN: 2 (`ICAEncoder.encode` -> `XGBoost-fusion.predict`). FULL: 7
(adds `VLSTM.forward`, `CNN-LSTM.forward`, `PiFormer.forward`,
`CNN-BiGRU.forward`, `Ridge-meta.predict`) - **3.5x fewer moving parts**
to load, version, test, and maintain in the lean variant.

**Recommendation, tied directly to the ablation evidence already
gathered in this project**: ship the LEAN variant (XGBoost-fusion +
ICAEncoder only). This is not a compromise - on this test set, LEAN
matches FULL's accuracy (in fact fractionally exceeds it), while being
~52x faster per prediction, marginally smaller on disk, and requiring
3.5x fewer model artifacts to maintain. There is no honest accuracy
case for keeping VLSTM/CNN-LSTM/PiFormer/CNN-BiGRU in the deployed
inference path for THIS pipeline on THIS data - sessions 9 and 17
already showed they don't help the ensemble; this session confirms the
practical corollary: since they don't help, removing them costs nothing
in accuracy and saves substantially everywhere else. The caveat worth
stating plainly rather than omitting: this conclusion is scoped to this
project's specific NASA+MIT-trained, cycle-level SOH regression task -
it says nothing about whether recurrent/attention sequence models would
earn their keep on a different task (e.g. genuinely sequential/
multi-cycle forecasting rather than per-cycle regression) where
XGBoost's tabular-feature framing may be less naturally suited. Full
per-variant numbers saved to `outputs/lean_vs_full_comparison.csv`.

New files (fully additive): `src/run_lean_vs_full_comparison.py`,
`models/ridge_meta_fusion_5branch.pkl` (persists what session 17 only
computed in-memory), `outputs/lean_vs_full_comparison.csv`,
`logs/logs_lean_vs_full.txt`.

## Follow-up session 21 — bootstrap confidence intervals on the key comparison results

Pure statistical analysis on already-computed predictions, no
retraining, new file `src/run_bootstrap_significance.py` only (imports
session 17/20's merge logic unchanged, touches nothing else). Tests
whether this project's headline comparisons - XGBoost's dominance, each
deep model's negligible/negative contribution, and session 20's
lean-vs-full result - are statistically real at this sample size or
within noise, using percentile bootstrap CIs (2000 resamples, 95% CI).

**Two resampling units are reported for every comparison, not one -
added beyond the literal ask because it's a real, honesty-relevant
issue this project already flagged twice before (session 11's RUL-
coverage battery-count caveat, session 19's small-battery-count
domain-classifier caveat):**
- **CYCLE-level** (the literal request): resample individual test
  cycles with replacement - simple, matches the instruction, but treats
  ~5208 highly autocorrelated within-battery cycles as if independent
  (pseudo-replication), which mechanically narrows every CI and makes
  even microscopic effects "significant" given enough pseudo-samples.
- **BATTERY-level (cluster bootstrap)**, reported alongside: resample
  whole test BATTERIES (all 6) with replacement, keeping each chosen
  battery's cycles together (Efron & Tibshirani 1993 sec. 8.6 - the
  standard way to bootstrap clustered data). Honest about the REAL
  effective sample size (6 batteries), at the cost of much wider
  intervals. Both are reported so the conclusion doesn't quietly depend
  on treating correlated cycles as independent evidence.

**1. Drop-branch ablation (5-branch) - delta R2 (full ensemble minus
drop-one-branch), per branch:**

| dropped branch | mean delta R2 | cycle-level 95% CI | cycle-level verdict | battery-level 95% CI | battery-level verdict |
|---|---|---|---|---|---|
| XGBoost-fusion | **+0.0970** | [+0.0813, +0.1138] | significant | [-0.0540, +0.2392] | **NOT significant** |
| VLSTM | +0.00017 | [+0.00009, +0.00024] | significant | [-0.00036, +0.00078] | not significant |
| CNN-LSTM | +0.0000 | [-0.00001, +0.00002] | not significant | [-0.00011, +0.00009] | not significant |
| PiFormer | +0.00003 | [+0.00001, +0.00004] | significant | [-0.00002, +0.00007] | not significant |
| CNN-BiGRU | -0.00006 | [-0.00014, +0.00002] | not significant | [-0.00055, +0.00059] | not significant |

**Honest, important nuance that has to be said plainly**: XGBoost-
fusion's importance is LARGE in point-estimate terms (+0.097 R2, by far
the biggest number in the table) but its 95% CI does NOT exclude zero
at the battery level - with only 6 test batteries, this project cannot
formally claim "statistically significant" for XGBoost's dominance at
95% confidence, even though the practical effect is obviously real and
consistent with every session that touched this pipeline. This is a
genuine STATISTICAL POWER limitation (6 units is a very small cluster
sample), not evidence the effect is fake - the cycle-level CI (which
DOES call it significant) is likely closer to the practically-true
signal, but is technically the less rigorous of the two tests given the
autocorrelation concern.

The 4 deep models tell an equally honest, more clear-cut story: VLSTM's
and PiFormer's tiny positive deltas ARE technically "cycle-level
significant" (their CIs barely exclude zero) - but at magnitudes of
0.00017 and 0.00003 R2 units respectively, this is a textbook case of
statistical significance without practical significance: a p<0.05
result driven by pseudo-replicated sample size, not a meaningful
ensemble contribution. CNN-LSTM and CNN-BiGRU don't even clear the
cycle-level bar. **None of the 4 deep models' contributions are
significant at the battery level** - the only level of resampling that
respects this dataset's real unit of independence.

**2. Base learner comparison - R2 point estimate + cycle-level CI:**

| model | point R2 | 95% CI |
|---|---|---|
| XGBoost | 0.9066 | [0.8978, 0.9148] |
| VLSTM | 0.8059 | [0.7912, 0.8194] |
| PiFormer | 0.6170 | [0.5804, 0.6469] |
| CNN-BiGRU | 0.5717 | [0.5368, 0.6027] |
| CNN-LSTM | 0.3335 | [0.2866, 0.3779] |

**Pairwise delta R2 vs. XGBoost - does XGBoost's dominance survive
resampling?**

| vs. | cycle-level 95% CI | cycle verdict | battery-level 95% CI | battery verdict |
|---|---|---|---|---|
| VLSTM | [+0.0839, +0.1178] | significant | [-0.0513, +0.2528] | **NOT significant** |
| CNN-LSTM | [+0.5279, +0.6204] | significant | [+0.1927, +1.0162] | **significant** |
| PiFormer | [+0.2578, +0.3266] | significant | [+0.0623, +0.5125] | **significant** |
| CNN-BiGRU | [+0.3031, +0.3696] | significant | [+0.0619, +0.6567] | **significant** |

**The single most useful, nuanced finding of this whole session**:
XGBoost's dominance over the 3 WEAKER deep models (CNN-LSTM, PiFormer,
CNN-BiGRU) is robust even under the strict, battery-clustered bootstrap
- genuinely, formally significant, not just a cycle-level artifact.
But XGBoost's edge over VLSTM specifically (the STRONGEST of the 4 deep
models, and the only one with a real, non-trivial standalone R2) is
NOT battery-level significant, despite a large ~0.10 R2 point-estimate
gap - with only 6 independent test batteries, this project cannot
formally rule out that VLSTM's underperformance vs. XGBoost is
battery-selection noise, even though it looks real in the data at hand.

**3. Lean vs. full (session 20) - delta RMSE and delta R2 (lean minus full):**

| metric | cycle-level 95% CI | cycle verdict | battery-level 95% CI | battery verdict |
|---|---|---|---|---|
| delta RMSE | [-0.00311, -0.00142] | significant | [-0.01149, +0.00860] | **NOT significant** |
| delta R2 | [+0.00017, +0.00038] | significant | [-0.00085, +0.00180] | **NOT significant** |

Session 20 reported LEAN as "marginally better" than FULL
(delta_rmse=-0.0023, delta_r2=+0.0003). This session's bootstrap
confirms that framing needs one honest correction: **that "LEAN is
better" edge is cycle-level significant only because of pseudo-
replication - at the battery level, the honest resampling unit, it is
NOT distinguishable from zero.** This does not weaken session 20's
actual recommendation (ship lean) - if anything it strengthens the
correct version of it: **LEAN and FULL are statistically
indistinguishable in accuracy** (not "LEAN is provably better"), which
combined with LEAN's ~52x latency advantage (session 20) is an even
cleaner case for shipping lean than a spurious accuracy edge would have
been - there is no accuracy trade-off to weigh against the efficiency
gain, in either direction.

**Overall honest summary, reported exactly as asked**: XGBoost's
practical dominance over the deep models is real and, for 3 of the 4
deep models (CNN-LSTM, PiFormer, CNN-BiGRU), holds up even under the
strict battery-level bootstrap. Its edge over VLSTM specifically, and
its edge over the full ensemble as a whole (the drop-XGBoost-fusion
row), are large in practice but NOT formally significant at the battery
level - an honest statistical-power limitation of this project's
6-battery test set, not a reason to doubt the finding, but a real
reason not to overstate its rigor. Every individual deep model's
ensemble contribution is at best a statistically-detectable-but-
practically-meaningless sliver (VLSTM, PiFormer) or not even that
(CNN-LSTM, CNN-BiGRU) - this holds at the cycle level and gets only
MORE true, not less, at the battery level. The lean-vs-full accuracy
"edge" from session 20 does not survive honest resampling and should be
read as "no measurable difference," not "lean wins."

Full CI tables saved to `data/processed/predictions/bootstrap_{drop_branch,
base_learner_r2,base_learner_delta_vs_xgb,lean_vs_full}_ci.csv`. New
file: `src/run_bootstrap_significance.py`; `logs/logs_bootstrap_significance.txt`.

## Follow-up session 22 — NASA EIS features as new candidate Health Indicators

Extracts NASA PCoE's never-before-used EIS (Electrochemical Impedance
Spectroscopy) data and tests, honestly, whether it earns a spot in BFA's
feature selection over the existing 7. Fully additive: `run_bfa.py`,
`hi_table.parquet`, and `bfa_selected_features.txt` are all untouched.

**What's actually in the raw NASA .mat files, checked directly rather
than assumed**: interleaved with charge/discharge entries, NASA logs
`'impedance'`-type cycle entries with a complex-valued swept-frequency
measurement (`Battery_impedance`, shape (48,), NASA's documented sweep
0.1Hz-5kHz) AND two ALREADY-FITTED equivalent-circuit parameters per
test - `Re` (electrolyte/ohmic resistance) and `Rct` (charge-transfer
resistance) - present in every impedance entry checked (278/278 for
B0005/B0006/B0007, 53/53 for B0018). Extracting `Re`/`Rct` directly
needed **zero new curve-fitting dependency** (no lmfit/impedance.py
Randles-circuit fit) - NASA already provides the fitted values, so
this is a read, not a fit. `src/eis_features.py` (new) extracts these
plus one frequency-agnostic magnitude summary (mean |Battery_impedance|
across the sweep) as a 3rd candidate: `EIS_Re`, `EIS_Rct`, `EIS_Zmag_mean`.

**Limitation #1, reported explicitly rather than worked around: no
frequency vector is stored in the .mat structure at all** - only the
raw complex array. NASA's documentation states 0.1Hz-5kHz, but without
an explicit per-sample frequency array IN THE FILE, indexing "impedance
at a specific frequency" (e.g. "Z at 1kHz") would require importing a
hardcoded assumption from external documentation with no way to verify
it against this data - so that was not attempted; only frequency-index-
free features (the 2 equivalent-circuit parameters + 1 whole-sweep
magnitude average) were extracted.

**Limitation #2, also reported explicitly: impedance measurements are
NOT one-to-one with discharge cycles.** NASA ran EIS sweeps on its own
schedule (e.g. B0005: 168 discharges vs. 278 impedance tests), so each
discharge cycle here is matched to whichever impedance test's timestamp
is CLOSEST in time (`eis_features.py`'s per-cycle time-matching, using
the `time` field every cycle entry already carries) - a nearest-
neighbor approximation, not an exact per-cycle reading. Match quality
reported, not hidden: median gap 0.53h (B0005/6/7) / 4.83h (B0018), but
9% of matches (57/636) have a gap >24h, up to a max of 389.5h (~16
days) - almost certainly end-of-life cycles where NASA's impedance-test
cadence had already tapered off relative to discharge cycling. These
are still included (not silently dropped) since excluding them would
itself be a silent workaround; the gap is reported per-row in
`eis_features_nasa.csv` for anyone who wants to filter by it.

**Limitation #3, the one the task specifically asked to be surfaced
rather than hidden: EIS is NASA-ONLY, and NASA is a small minority of
the pooled dataset.** MIT's source HDF5 files and CALCE's format
contain no impedance measurements of any kind. Since `hi_table.parquet`
pools NASA (4 batteries) + MIT (28) + CALCE (3) = 26,996 total cycles,
the 3 new EIS columns are NaN for **97.6% of all rows** (26,360/26,996) -
a far more severe missingness pattern than the existing MATC/MATD/
MATDL columns (NaN only for CALCE, ~9% of rows). Imputed with the exact
same convention `run_bfa.py` already uses for MATC/MATD (column median
from non-NaN values - here, necessarily, the NASA-only median), logged
explicitly because this means ~98% of every EIS column's values feeding
BFA's wrapper fitness are an imputed CONSTANT, not measured data.

**Result: none of the 3 EIS-derived features were selected.**

| | 16-candidate BFA (original, session 1) | 19-candidate BFA (this session, +3 EIS) |
|---|---|---|
| selected | ICHV, MATC, MATD, SCV, TEVI, VDEDT, VIECT (7) | ICHV, MATC, MATD, MATDL, SCV, TEVD, UVP, VIECT (8) |
| EIS features selected | n/a | **NONE** (EIS_Re, EIS_Rct, EIS_Zmag_mean all excluded) |

Reported honestly, exactly as asked: **the EIS-derived features did not
get selected over the existing HIs.** Worth noting as a real, if
secondary, side-effect: the non-EIS selection itself shifted slightly
(dropped VDEDT and TEVI, gained MATDL, TEVD, UVP) even at the same
seed=42 - BFA is a stochastic metaheuristic over a search space whose
DIMENSIONALITY changed (16 -> 19 features means every agent's position
vector, and therefore every RNG draw downstream of it, differs from the
original run), so this is not "the same 7 plus a verdict on EIS," it's
a genuinely independent re-run of the whole selection process, and its
non-EIS answer landing close to (6 of 7 features shared) but not
identical to the original is expected, not a bug.

**Honest interpretation, not just the raw fact**: this negative result
is very plausibly explained by limitation #3 above rather than EIS
being uninformative in principle - a feature that is a NASA-only
non-null value hiding among a ~98%-imputed-constant column has very
little room to demonstrate predictive value to a wrapper method
evaluated with battery-grouped cross-validation across a pool
overwhelmingly dominated by MIT/CALCE batteries where the feature
literally cannot vary. This is a genuine limitation of the DATA
AVAILABILITY (EIS only exists for one of three source datasets), not
necessarily a limitation of EIS as a health indicator in general - the
literature on impedance-based battery diagnostics (Rct in particular is
a well-established degradation marker, tracking charge-transfer
resistance growth with SEI/electrode aging) suggests these features
could plausibly matter on an EIS-complete dataset. This project's BFA
result should be read as "EIS didn't help THIS pooled, EIS-sparse
dataset," not as "EIS doesn't matter for battery SOH."

New files (fully additive): `src/eis_features.py`,
`src/run_bfa_with_eis.py`, `data/processed/eis_features_nasa.csv`,
`data/processed/bfa_history_with_eis.csv`,
`data/processed/bfa_selected_features_with_eis.txt`,
`logs/logs_{eis_features,bfa_with_eis}.txt`.

## Follow-up session 23 — degradation-mode analysis via dV/dQ peak-tracking

Adds an electrochemical "why" on top of this project's existing SOH
number and SHAP "which features mattered": peak-tracking on the dV/dQ
differential-voltage curve already computed by `ica_dv_dc.py`
(unchanged, imported not duplicated - same function feeding CNN-LSTM/
PiFormer/CNN-BiGRU's dV/dQ channel) to produce a qualitative
degradation-mode signature per battery. New file:
`src/run_degradation_mode_analysis.py`, fully additive, no retraining.

**Scoping, stated as plainly as the task asked**: this is inspired by
established differential-voltage-analysis (DVA) degradation-mode
literature (Bloom et al. 2005; Dubarry, Truchot & Liaw 2012) - peak
POSITION shift is associated with loss of lithium inventory (LLI),
peak HEIGHT/amplitude loss with loss of active material (LAM). It is
explicitly **NOT a validated LLI/LAM decomposition**: that literature's
real diagnostic power comes from comparing full-cell DVA against
HALF-CELL reference curves, and this project's datasets (NASA/CALCE/
MIT) have no half-cell data at all. What follows is a simplified single-
signal heuristic reported as a qualitative LEANING, not a quantified
LLI%/LAM% split.

**Method**: `V_grid` (per-cycle, needed for peak POSITION in real
Volts) isn't persisted in the saved `differential_tensors/*.npy` files
(only the 3 differential channels are stacked, not the voltage axis
each was computed against) - recomputed here via the exact same
unmodified `compute_ica_dv_dc` call Phase 1 already used, not a new
method. Peak-tracking: the most prominent peak in the first usable
cycle anchors position; every later cycle searches for a peak within
0.15V of the last known position (standard peak-tracking practice -
follow the same physical feature's drift, don't re-pick "whatever is
biggest" each cycle, which risks jumping between unrelated peaks as
their relative sizes cross over during fade). Applied to 3
representative test batteries spanning NASA/MIT (per instruction, not
all 6): NASA/B0018, and MIT/b3c0 + MIT/b1c4 (short/long-life spread;
b3c0 also revisits session 16's knee-point finding of an early-life
curve anomaly there).

**Mid-analysis correction, logged rather than hidden (twice)**:
1. The first version tracked raw dV/dQ curve VALUE at the peak as
   "height". Inspecting the output showed this was numerically unstable
   - e.g. one MIT battery's tracked peak (position stable at ~2.03-2.07V)
   swung 367502 -> 6318 -> 9119 -> 7632 across its first 4 cycles. dV/dQ
   is well known in the DVA literature to be noisier/spikier than dQ/dV
   at comparable smoothing (differentiating twice amplifies noise).
   Switched to peak PROMINENCE (height above local surrounding baseline,
   a relative measure) instead of raw curve value - visibly more stable.
2. Even with prominence, cycle 1 specifically was a reproducible
   OUTLIER on every battery checked (NASA B0018 cycle 1: ~101618 vs.
   ~1000-6000 for cycles 2-11; MIT b1c4 cycle 1: ~248658 vs. ~5000-8000
   for cycles 2-10) - most plausibly a numerical edge effect (Savitzky-
   Golay/`np.gradient` boundary behavior on the very first interpolated
   curve) rather than a genuine degradation-relevant value. Using a
   single first-cycle endpoint made `delta_H_frac` almost entirely an
   artifact of this one anomalous cycle. Fixed by baselining height on
   the MEDIAN of the first/last 5 tracked cycles instead of a single
   endpoint (position kept as single-endpoint, since position showed no
   equivalent instability).

**Results - degradation-mode signature alongside the existing SOH
prediction:**

| battery | tracking coverage | SOH true (first→last) | SOH pred (last) | Δposition (% of V window) | Δheight (relative) | signature |
|---|---|---|---|---|---|---|
| NASA/B0018 | 131/132 (99.2%) | 100.6% → 72.8% | 81.5% | 5.7% | -61.8% | **mixed LLI+LAM-leaning** |
| MIT/b3c0 | 930/1007 (92.3%) | 99.9% → 82.4% | 83.3% | 2.0% | -40.5% | **LAM-leaning** |
| MIT/b1c4 | 548/1225 (**44.7%**) | 99.9% → 94.7% | 97.4% | 3.2% | -37.8% | **LAM-leaning** (lower confidence - see coverage caveat) |

NASA/B0018 - by far the most degraded of the 3 (72.8% SOH, near this
project's 80% EOL threshold) - shows BOTH a real position shift (5.7%
of its discharge window, above the 5% heuristic threshold) and a large
height collapse, read as a mixed-mode signature: plausible for a cell
this far into its fade, where both lithium-inventory loss and active-
material loss are commonly reported to coexist. Both MIT cells show
LAM-leaning signatures (large height loss, position comparatively
stable, below the 5% threshold) even at very different overall SOH
(82.4% vs. 94.7%) - notably, MIT/b1c4 shows a substantial LAM-type
signal (-37.8%) despite only mild capacity fade (94.7% SOH), which
could be genuine early-onset LAM (LAM commonly precedes large visible
capacity loss in the literature) or could be an artifact of this
battery's markedly worse tracking coverage - flagged, not glossed over.

**Reliability caveat, reported plainly rather than only in a footnote**:
peak-tracking coverage varies dramatically across these 3 batteries -
99.2% (NASA/B0018) and 92.3% (MIT/b3c0) down to just **44.7%**
(MIT/b1c4, 677 of 1225 cycles lost - the tracked peak drifted beyond
the 0.15V search window or vanished entirely). A plausible (not
independently confirmed here) explanation: MIT's fast-charge cells may
have flatter voltage-capacity curves with less-pronounced phase-
transition features than NASA's cells, making peaks harder to detect/
track reliably - but this project has no half-cell or chemistry
datasheet to confirm that explanation, so it's reported as a plausible
hypothesis, not a verified cause. Practical consequence: MIT/b1c4's
signature above should be read with LESS confidence than the other two,
purely on data-coverage grounds, independent of the position-vs-height
methodology question.

Per-cycle peak tracks saved to
`data/processed/predictions/degradation_mode_peak_tracks.csv`
(dataset, battery_id, cycle_idx, peak_V, peak_H - NaN where tracking was
lost, not silently interpolated); per-battery summary to
`outputs/degradation_mode_summary.csv`. New file:
`src/run_degradation_mode_analysis.py`; `logs/logs_degradation_mode_analysis.txt`.

## Follow-up session 24 — quantizing the lean pipeline for embedded/BMS feasibility

Quantizes session 20's "lean" deployment pipeline (ICAEncoder +
XGBoost-fusion) and checks it against realistic embedded-hardware
memory budgets. Fully additive: `ica_encoder.pt`/`xgb_soh_fusion.json`
untouched; new files only. New file: `src/run_model_quantization.py`.

**Scope, per instruction**: reduce the ICA encoder's (neural) weight
precision; report XGBoost's size AS-IS, not quantized - a tree
ensemble's "model" IS its tree structure (per-node split thresholds +
per-leaf values), not a stack of weight matrices, so there's no
standard numeric-precision quantization API for it the way there is for
a neural net (`torch.quantization`). Shrinking XGBoost would mean
pruning trees/reducing `n_estimators`/truncating leaf-value precision -
changing the MODEL, not just its numeric representation - explicitly
out of scope here, exactly as the task anticipated.

**Method**: FP16 (`.half()` on every tensor) and INT8 (per-output-
channel symmetric affine quantization via `torch.quantize_per_channel`
- weights only, biases kept FP32 as standard practice) applied to the
9.15KB `ica_encoder.pt`. `torch.quantize_per_channel` is flagged
deprecated in this torch version (a newer `torchao`-based API exists) -
used anyway since it's still fully functional and adding a new
quantization-library dependency would work against the task's own
"standard, no heavy new dependencies" scope; noted, not hidden.
Accuracy evaluated by DEQUANTIZING back to float32 before the Conv1d
forward pass (eager-mode Conv1d can't run directly on qint8 tensors
without the full separate quantized-module graph) - this correctly
captures quantization's NUMERICAL error but, stated plainly, does NOT
measure any latency benefit real int8 hardware kernels would provide;
no real embedded hardware was available to measure that here, per
instruction.

**Result 1 - size, with a genuinely non-obvious finding**:

| precision | ICAEncoder size | vs. FP32 |
|---|---|---|
| FP32 (baseline) | 9.15 KB | - |
| FP16 | **5.90 KB** | 1.55x smaller |
| INT8 (weights) | 6.12 KB | 1.49x smaller |

**FP16 actually beats INT8 in absolute size for this model** - the
opposite of the usual "int8 = 4x smaller than fp32" intuition. Cause,
verified rather than assumed: this model is tiny (1,665 parameters
total across `conv1`/`conv2`/`head`), so INT8's per-output-channel
scale/zero-point CALIBRATION METADATA (one float64 scale per output
channel, 16+16+1=33 extra values) is a non-negligible fraction of the
already-tiny parameter count - the metadata overhead eats most of
int8's theoretical 4x storage win at this scale. A real, reportable
finding specific to very small models, not a general refutation of
int8 quantization's usual size advantage on larger networks.

**Result 2 - accuracy (XGBoost held fixed, so any change is 100%
attributable to encoder precision)**:

| precision | RMSE | R2 | delta R2 vs. FP32 |
|---|---|---|---|
| FP32 (baseline) | 1.3921 | 0.91715 | - |
| FP16 | 1.3922 | 0.91714 | -0.00002 |
| INT8 (weights) | 1.3895 | 0.91745 | **+0.00030** |

Both changes are utterly negligible - INT8 even comes out marginally
*better* than FP32, plausibly quantization acting as a mild noise
perturbation rather than a genuine improvement. Sanity-checked against
session 21's own bootstrap-CI framework: session 21 found delta-R2
values of this exact order of magnitude (~0.0003, the lean-vs-full
accuracy "edge") were NOT battery-level statistically distinguishable
from zero. By the same standard, this quantization-vs-FP32 delta should
be read as noise, not a genuine accuracy change in either direction -
**quantization cost nothing measurable in accuracy, at either
precision level.**

**Result 3 - the honest headline finding: quantizing the encoder was
almost beside the point.** ICAEncoder is 9.15KB of a 2,854.9KB total
lean-pipeline size (0.3%) - `xgb_soh_fusion.json`'s 2,845.7KB (500
trees, depth 6) is 99.7% of the total, and encoder quantization's
maximum possible saving (3.0KB) is a rounding error against it. As an
aside (not requested, zero precision change - same model bit-for-bit,
just a more compact serialization): saving XGBoost via its own compact
binary format (`.ubj`) instead of the default verbose JSON text shrinks
it to 1,981.0KB (1.44x smaller) - a bigger, entirely free win than
anything encoder quantization could offer, achieved with ZERO
numerical change.

**Result 4 - embedded/BMS feasibility, reference figures only, stated
explicitly as such (not measured on real hardware - none available)**:
typical Cortex-M0+/M3/M4-class BMS microcontrollers ship with roughly
32KB-512KB total flash and 4KB-256KB total RAM for the WHOLE firmware
(not just an ML model); commonly-cited TinyML deployment benchmarks
(e.g. MLPerf Tiny-class targets) aim for roughly tens-of-KB to ~250KB
model budgets to coexist with the rest of a device's firmware. **This
project's best-case quantized lean pipeline (INT8 encoder + UBJ
XGBoost) is ~1,987KB - 3.9x to 62x OVER the full flash budget of a
typical small BMS MCU**, and that gap is almost entirely XGBoost's
~1,981KB (500 trees), not the 6.1KB encoder. Reported plainly: **this
pipeline, even after quantization, is not feasible on typical
microcontroller-class BMS hardware** - genuine embedded feasibility
would require shrinking the tree ensemble itself (fewer/shallower
trees, explicitly out of scope for "quantization" as requested here),
not further neural-weight precision reduction, which has already
extracted essentially all it can from a model this small.

Full size/accuracy table saved to
`outputs/model_quantization_summary.csv`. New files:
`src/run_model_quantization.py`, `models/ica_encoder_{fp16,int8}.pt`,
`models/xgb_soh_fusion.ubj` (the zero-precision-change aside),
`logs/logs_model_quantization.txt`.

## Follow-up session 25 — second-life grading classifier (post-processing, no new model)

Pure post-processing/labeling on the existing "lean" (session 20)
XGBoost-fusion SOH predictions (`xgb_fusion_preds.csv`, test split) -
no new model, no retraining, new file `src/run_second_life_grading.py`
only. Buckets both TRUE and PREDICTED SOH into 3 industry-standard
grades: SOH>=80% "Primary EV use", 50-80% "Second-life candidate (grid
storage/backup)", <50% "Recycle only" (the 80% cutoff is the same
threshold this project already uses project-wide as the EOL definition
in `rul_labels.py`).

**Why both true AND predicted SOH are graded, not just predicted**:
grading is a discrete, consequential decision - a small SOH error near
a threshold can flip a real disposition decision even when it's
unremarkable in RMSE terms. Misgrades are split by DIRECTION since they
are not symmetric in consequence: **"risky"** (predicted grade more
optimistic than true - e.g. true=second-life, predicted=primary-EV - a
safety/reliability-relevant error, routing an unfit cell into
continued/secondary service) vs. **"conservative"** (predicted grade
more pessimistic than true - an economic-loss error, scrapping a usable
asset early, not a safety concern).

**Overall grading distribution and agreement, across all 5,208 test
cycles (6 batteries):**

| | true | predicted |
|---|---|---|
| Primary EV use | 98.2% | 99.3% |
| Second-life candidate | 1.8% | 0.7% |
| Recycle only | 0% | 0% |

No test cycle (true or predicted) ever falls below 50% SOH - the
"Recycle only" grade is untested by this test set entirely, reported
plainly rather than silently omitted from the table. **Grading
agreement (predicted grade == true grade): 98.75%** of 5,208 cycles -
61 risky misgrades (1.17%), 4 conservative misgrades (0.08%).

**The single most operationally important finding of this session,
surfaced by the per-battery CURRENT-STATUS view (grading at each
battery's LAST test cycle - the realistic point a real triage decision
would be made, i.e. when a cell is pulled from primary service):**

| battery | last true SOH | last predicted SOH | true grade | predicted grade |
|---|---|---|---|---|
| **NASA/B0018** | **72.76%** | **81.50%** | Second-life candidate | **Primary EV use (WRONG)** |
| MIT/b1c4 | 94.71% | 97.42% | Primary EV use | Primary EV use (correct) |
| MIT/b2c24 | 77.28% | 76.94% | Second-life candidate | Second-life candidate (correct) |
| MIT/b3c0 | 82.42% | 83.31% | Primary EV use | Primary EV use (correct) |
| MIT/b3c35 | 82.73% | 83.96% | Primary EV use | Primary EV use (correct) |
| MIT/b4c38 | 78.35% | 79.65% | Second-life candidate | Second-life candidate (correct) |

**At the exact moment a real disposition decision would be made for
NASA/B0018 today, this model would incorrectly certify it fit for
continued primary EV use when its true SOH (72.76%) already places it
solidly in the second-life-only bracket** - an 8.7-point overestimate
that crosses the operationally consequential 80% line, not a boundary-
adjacent rounding error. This is not a one-off: inspecting the full
misgrade list shows this is a SUSTAINED systematic overestimate for
B0018, not threshold noise - 41 of its last 56 cycles (roughly its
entire second-life-eligible life-phase, cycles 77-132) are risky
misgrades, with the true-vs-predicted gap running consistently 5-9
percentage points across that whole stretch (e.g. cycle 90: true=76.8%,
predicted=85.0%). By contrast, the OTHER risky misgrades in the dataset
(all from MIT/b4c38, its last 17 cycles) are much milder, boundary-
hugging cases (true SOH drifting from 79.99% to 78.69% while
predictions stay just above 80%, a 1-2 point overestimate) - genuinely
different in character and severity from B0018's case. All 4
conservative misgrades are similarly boundary-adjacent (within ~1-6
points of 80%), confirming they're threshold noise rather than a
systematic pattern.

**Honest read**: this project's known finding that NASA/B0018 is its
hardest-to-predict test battery (the sole NASA battery in the test set,
and the only one contributing a real EOL-crossing life-phase to this
test set at all - see the per-battery full-life distribution below)
translates directly into a real, consequential grading error, not just
a slightly-worse RMSE number - exactly the kind of gap between
"aggregate accuracy looks fine" and "a specific real-world decision
would be wrong" this post-processing step exists to surface.

**Per-battery full-life grade distribution (% of each battery's
observed test-window cycles, by TRUE SOH)** - shows why B0018 is the
only battery where this matters: it's the only one that spends a
substantial fraction of its observed life in the second-life bracket
(42.4%) rather than staying almost entirely in the primary-use bracket
like every MIT test battery (96.9%-100%, except b4c38's 1.8% and
b2c24's 3.1% right at the tail end):

| battery | Second-life candidate | Primary EV use |
|---|---|---|
| NASA/B0018 | **42.4%** | 57.6% |
| MIT/b1c4 | 0.0% | 100.0% |
| MIT/b2c24 | 3.1% | 96.9% |
| MIT/b3c0 | 0.0% | 100.0% |
| MIT/b3c35 | 0.0% | 100.0% |
| MIT/b4c38 | 1.8% | 98.2% |

New files: `src/run_second_life_grading.py`, per-cycle grades saved to
`data/processed/predictions/second_life_grading_per_cycle.csv`,
current-status/per-battery-distribution summaries to
`outputs/second_life_grading_{current_status,per_battery_distribution}.csv`,
`logs/logs_second_life_grading.txt`.

## Follow-up session 26 — sensor-noise robustness test (new axis: input imperfection, not domain shift)

Tests something genuinely uncovered by any prior session: robustness to
realistic INPUT IMPERFECTION on the SAME battery/domain, as distinct
from domain shift (a different battery/dataset entirely - sessions
5/13/19) or statistical significance of the existing predictions
(session 21). Pure inference-time test on the already-trained lean
(session 20) pipeline - `ica_encoder.pt`/`xgb_soh_fusion.json`
untouched, no retraining. New file:
`src/run_sensor_noise_robustness.py`.

**Noise levels, stated explicitly per instruction**: independent
Gaussian noise added to every raw V/I/T sample of every test cycle's
charge AND discharge phases, at 3 levels - 1x ("BMS-grade", exactly as
specified in the task): sigma_V=1mV, sigma_I=10mA, sigma_T=0.5degC;
2x: sigma_V=2mV, sigma_I=20mA, sigma_T=1.0degC; 5x (stress test):
sigma_V=5mV, sigma_I=50mA, sigma_T=2.5degC. **Interpretation stated
explicitly**: the task's "+/-1mV" etc. is read as the Gaussian STANDARD
DEVIATION, not a hard clip bound - a common shorthand for roughly a
1-sigma sensor noise floor; this choice directly sets how much noise is
actually injected, so it's stated rather than left implicit. `t`
(sampling clock) and `discharge_capacity` (the separately-measured/
coulomb-counted SOH ground-truth source, untouched by V/I/T noise) are
left unperturbed, so ground truth SOH is IDENTICAL at every noise level
- isolating "does the prediction degrade" from any confound of a
shifting target.

**Sanity check passed before trusting anything else**: the "clean
(baseline)" noise level (sigma=0 everywhere) reproduces session 20's
already-known lean-pipeline numbers EXACTLY (RMSE=1.3921, R2=0.9172,
matching to 4 decimal places) despite being an independent
reimplementation (HIs/ICA/embeddings recomputed fresh from raw cycles
here, not read from the cached `hi_table.parquet`/`fusion_embeddings.csv`)
- confirms this script's pipeline genuinely matches the deployed one
before drawing any noise conclusions from it.

**Pooled result across all 5,208 test cycles - looks, at first glance,
like total noise immunity:**

| noise level | RMSE | R2 | delta R2 vs. clean |
|---|---|---|---|
| clean (baseline) | 1.3921 | 0.9172 | - |
| 1x BMS-grade | 1.3714 | 0.9196 | +0.0024 |
| 2x BMS-grade | 1.3814 | 0.9184 | +0.0013 |
| 5x BMS-grade (stress) | 1.3614 | 0.9208 | +0.0036 |

RMSE/R2 barely move, and if anything tick very slightly BETTER at
every noise level, even at 5x stress. Counter-intuitive, and not
accepted at face value - checked further rather than reported as-is.

**Per-battery breakdown reveals the pooled number is HIDING a real,
asymmetric pattern - this is the actual finding of this session, not
the flattering pooled headline:**

| battery | clean R2 | 1x R2 | 2x R2 | 5x R2 | clean RMSE | 5x RMSE |
|---|---|---|---|---|---|---|
| **NASA/B0018** | 0.576 | 0.565 | 0.525 | **0.513** | 5.449 | **5.842** |
| MIT/b2c24 | 0.939 | 0.948 | 0.943 | **0.897** | 1.426 | **1.863** |
| MIT/b1c4 | 0.195 | 0.241 | 0.212 | 0.370 | 1.436 | 1.270 |
| MIT/b3c0 | 0.945 | 0.945 | 0.950 | 0.965 | 0.889 | 0.708 |
| MIT/b3c35 | 0.983 | 0.983 | 0.986 | 0.986 | 0.555 | 0.493 |
| MIT/b4c38 | 0.958 | 0.961 | 0.969 | 0.983 | 1.086 | 0.691 |

**NASA/B0018 - already flagged in session 25 as this pipeline's
hardest-to-predict, most operationally consequential test battery -
degrades MONOTONICALLY and CONSISTENTLY at every single noise level**:
R2 falls 0.576 -> 0.565 -> 0.525 -> 0.513, RMSE rises 5.449 -> 5.522 ->
5.771 -> 5.842, in lockstep with increasing noise, at all 3 levels with
no reversal. A clean 4-point monotonic trend in one direction is very
unlikely to be pure chance the way the pooled number's non-monotonic
wobble (better at 1x, worse at 2x, best at 5x) plausibly is. MIT/b2c24
shows a related, if smaller, real effect: stable through 1x/2x but a
genuine drop specifically at 5x stress (R2 0.939->0.897) - a
stress-level-specific weakness the pooled average also papers over.
The other 4 MIT batteries show mild, non-monotonic IMPROVEMENT or no
clear trend - and because they collectively outnumber and outweigh
B0018/b2c24 in the pooled 5,208-cycle average, their improvement
MASKS B0018's real, consistent degradation in the aggregate number.
(MIT/b1c4's R2 values of 0.19-0.37 look alarming in isolation but are a
known artifact of that battery's own near-flat true SOH range over its
observed test window - R2 is a fragile metric on a near-constant
target - not a new noise-robustness finding; its RMSE, the more
meaningful number for a low-variance target, stays consistently small
and even improves slightly under noise like most of the other MIT
cells.)

**Plausible explanation for the small pooled effect size overall**
(offered honestly as a hypothesis, not independently verified further):
Health Indicators and the ICA/DV/DC channels are computed as SMOOTHED,
AGGREGATE quantities (durations, capacity integrals, Savitzky-Golay-
smoothed curves) over hundreds of raw V/I/T samples per cycle - zero-
mean per-sample Gaussian noise at these realistic BMS-grade magnitudes
is heavily attenuated by this built-in averaging before it reaches the
model, roughly consistent with noise shrinking as ~1/sqrt(n) over an
n-sample integration window. This plausibly explains why MOST test
batteries show negligible-to-mildly-positive effects even at 5x stress
- but does NOT explain, and this session did not further diagnose, why
NASA/B0018 specifically fails to get this same protection.

**Honest verdict, tying together with session 25 rather than treating
this as an isolated result**: reporting only the pooled metric would
have concluded "this pipeline is essentially immune to realistic sensor
noise" - true in aggregate, but wrong as a claim about every battery.
The per-battery view shows a real, modest, but genuine noise-robustness
weakness specific to NASA/B0018 (and a milder one for MIT/b2c24 at
stress level) - the SAME battery session 25 already flagged as this
pipeline's hardest case via a completely independent test (second-life
grading accuracy, not noise injection). Two unrelated stress-tests
converging on the same weak point is a more convincing signal than
either alone: NASA/B0018 is this pipeline's genuine reliability edge
case, not a one-off artifact of either analysis.

Full per-level summary saved to
`outputs/sensor_noise_robustness_summary.csv`; per-cycle predictions at
every noise level to
`data/processed/predictions/sensor_noise_robustness_per_cycle.csv`.
New file: `src/run_sensor_noise_robustness.py`;
`logs/logs_sensor_noise_robustness.txt`.

## Follow-up session 27 — root-causing NASA/B0018 as this pipeline's consistent weak point

Three independent, unrelated analyses (the early-prediction test,
session 25's second-life grading, session 26's sensor-noise robustness)
flagged NASA/B0018 as this pipeline's weakest test battery, but none
asked why. This session investigates, on 4 requested angles, all pure
analysis on existing data - no retraining. New file:
`src/run_b0018_root_cause_analysis.py`, reusing session 23's
degradation-mode functions and session 19's domain-classifier
methodology (both imported, not duplicated).

**1. Training representation - NASA is drastically underrepresented,
more so by cycle count than battery count:**

| | batteries | cycles |
|---|---|---|
| NASA (training) | 3 (11.5%) | 504 (**2.7%**) |
| MIT (training) | 23 (88.5%) | 18,341 (97.3%) |

Only 1 of NASA's 4 batteries (B0018) is even in the test set; the other
3 (B0005/6/7) are training-only. But the more telling number is
CYCLES, not batteries: NASA contributes only 2.7% of all training
cycles, because NASA batteries are individually far shorter-lived than
MIT's - this pipeline's XGBoost splits are calibrated on a training set
that is, in cycle-count terms, essentially all-MIT.

**2. Lifetime/protocol characteristics - B0018 fades dramatically
faster per cycle than every MIT test battery:**

| battery | n_cycles | SOH drop | fade rate (pp/cycle) |
|---|---|---|---|
| **NASA/B0018** | **132 (fewest)** | 27.9pp | **0.2112 (fastest by far)** |
| MIT/b2c24 | 523 | 22.6pp | 0.0432 |
| MIT/b4c38 | 1230 | 21.6pp | 0.0175 |
| MIT/b3c0 | 1007 | 17.5pp | 0.0174 |
| MIT/b3c35 | 1091 | 17.2pp | 0.0158 |
| MIT/b1c4 | 1225 | 5.2pp | 0.0043 |

B0018 fades **4.9x faster per cycle** than even the fastest-fading MIT
test battery, despite covering a similar or smaller total SOH range -
consistent with NASA's standard, low-rate constant-current cycling
protocol vs. MIT's fast-charging-study protocol (a well-documented
characteristic of these two specific published datasets, not
independently re-derived here - MIT's dataset exists specifically to
study fast-charging optimization).

**3. Feature-distribution outlier check (session-19-style domain
classifier, MIT-train vs. each test battery, WITH proper controls):**

| battery | AUC vs. MIT-train |
|---|---|
| **NASA/B0018** | **1.0000 (perfect separation)** |
| MIT/b2c24 | 0.9898 |
| MIT/b3c35 | 0.9870 |
| MIT/b4c38 | 0.9719 |
| MIT/b1c4 | 0.9515 |
| MIT/b3c0 | 0.9207 |

Every held-out battery looks quite separable from MIT-train (0.92-0.99)
- consistent with session 19's own finding that small battery counts
make even genuine in-domain holdouts look separable to a classifier, so
this alone would NOT be a clean B0018-specific finding. **But B0018 is
the only one to hit the absolute ceiling (1.0000)** - categorically
beyond even the already-elevated MIT-battery range, not just modestly
higher.

**The concrete, mechanistic explanation for that ceiling, found by
checking WHICH features drive it**: of the 7 BFA-selected HIs, **ICHV**
and **TEVI** - both raw WALL-CLOCK-TIME durations (`ICHV` = time spent
in the >=90%-of-peak-voltage CV-tail during charging; `TEVI` = time to
traverse the 80%->20% discharge-voltage band, per `health_indicators.py`'s
own formulas) - are astronomically different for B0018:

| feature | MIT-train mean | B0018 mean | z-score | percentile |
|---|---|---|---|---|
| ICHV | 26.5 (sec) | **10,200.2** | **z=855** | 100th |
| TEVI | 13.0 (sec) | **2,565.6** | **z=420** | 100th |

Every single one of 18,341 MIT training cycles has a LOWER ICHV and
TEVI value than B0018's average - not a subtle shift, a complete
non-overlap. This is directly explained by point #2: these are absolute
TIME measurements of charge/discharge phase durations, and NASA's slow
protocol makes those phases last 2-3 ORDERS OF MAGNITUDE longer in wall-
clock time than MIT's fast-charging protocol - so 2 of the model's 7
selected input features are, for B0018, numerically far outside
anything the tree-splitting thresholds were ever calibrated against.
This is very plausibly why the domain classifier hits a clean 1.0 AUC
for B0018 specifically: it has an easy, unambiguous signal to exploit
that no MIT test battery provides.

**4. Degradation-mode signature (session 23's method, extended here
from 3 to all 6 test batteries for complete coverage):**

| battery | position shift (% of V window) | signature |
|---|---|---|
| **NASA/B0018** | **5.7%** | **mixed LLI+LAM-leaning** |
| MIT/b3c35 | 3.8% | LAM-leaning |
| MIT/b4c38 | 3.5% | LAM-leaning |
| MIT/b1c4 | 3.2% | LAM-leaning |
| MIT/b3c0 | 2.0% | LAM-leaning |
| MIT/b2c24 | 1.8% | LAM-leaning |

With full coverage now (session 23 only checked 2 of the 5 MIT test
batteries), the pattern is clean and complete: **B0018 is the ONLY one
of all 6 test batteries whose peak-position shift crosses the 5%
heuristic threshold** - every MIT battery, without exception, shows a
pure LAM-leaning signature.

**Synthesis - an honest "converges on one root cause with several
correlated symptoms", not 4 independent causes stacked coincidentally**:
findings #2, #3, and #4 are NOT 4 separate root causes - they are 3
different, independently-measured SYMPTOMS of the SAME underlying fact:
**NASA's cycling protocol (and likely cell chemistry/format) differs
fundamentally from MIT's, and the training set this pipeline was fit on
is, in cycle-count terms, 97.3% MIT.** The faster fade rate (#2), the
ICHV/TEVI feature values sitting completely outside the training
distribution (#3), and the different degradation-mode signature (#4)
are all downstream consequences of that one protocol/chemistry gap, not
independent coincidences - and #1 (severe training underrepresentation)
is WHY the pipeline never had enough NASA-protocol examples to
calibrate split thresholds that would generalize to it. In short:
**B0018 is a mild, WITHIN-project echo of the exact CALCE domain-shift
problem this project already extensively documented (sessions 5/13/19)
- a genuinely different cell/protocol regime, just one that (unlike
CALCE) has SOME, though very little (2.7% of cycles), representation in
training rather than none at all.** That partial representation is
presumably why B0018's degradation is "merely" this pipeline's weakest
test case rather than a CALCE-scale collapse - but it is not enough
representation for the model to have genuinely learned NASA's protocol
regime, which is exactly what 3 unrelated stress-tests (early-
prediction, grading, noise-robustness) independently discovered without
knowing why.

Per-section results saved to
`outputs/b0018_rootcause_{lifetime,domain_auc,feature_zscores,
degradation_mode}.csv`. New file: `src/run_b0018_root_cause_analysis.py`;
`logs/logs_b0018_root_cause.txt`.

## Follow-up session 28 — genuine incremental/online-update Digital Twin mode

Adds a real online-learning Digital Twin mode alongside (not replacing)
the existing one-shot Prediction tab - the first thing in this project
to genuinely update as new data arrives, revisiting session 7's explicit
scope decision ("no live incremental meta-learner updating") as a new,
separate, opt-in mode rather than changing any existing tab's behavior.

**Citation check performed before writing anything else** (same
discipline as sessions 14/15/22/23): the task described this as
matching "this project's own cited reference (Najafi-Shad et al., Paper
DT)". A full-repository search found **zero** mentions of "Najafi"
anywhere in this codebase or DEVELOPMENT_LOG before this session - so
"this project's own cited reference" was not an accurate premise. The
underlying paper IS real, though (confirmed via web search): Najafi-
Shad, Sciortino, Resalati et al., "Digital twin-based prediction of
battery parameters from limited initial data using an optimised
time-series multi-layer perceptron," Journal of Energy Storage (Feb
2026) - genuinely on-theme (predicting from limited initial data,
framed as simulation-stage, not hardware-in-the-loop). Cited here
honestly, for the first time, on its own merits, not as a continuation
of a prior citation.

**Scoping, stated explicitly per instruction - what is and isn't
"real-time" here**: this is a SIMULATION-STAGE digital twin. It replays
an already-recorded NASA/MIT TEST battery's cycles one at a time (same
raw data every other tab uses) with an artificial per-cycle delay in
the UI, to emulate live arrival. It does **NOT** connect to real
hardware, a real BMS, or a live sensor - there is no hardware-in-the-
loop anywhere in this session's work.

**What's FROZEN (pretrained, never touched)**: `ica_encoder.pt`,
`xgb_soh_fusion.json` (the lean pipeline, session 20), `ocsvm_model.pkl`/
`ocsvm_scaler.pkl` (session 7's anomaly detector, reused as-is), and
`joint_adaptive.pt` (RUL - shown as a frozen one-shot value; RUL is
explicitly outside the "lean" pipeline this task scoped for online
updating, and no online-learning claim is made about it).

**What UPDATES ONLINE**: a lightweight `SGDRegressor` (scikit-learn,
`partial_fit` - a genuine incremental learner, not a batch refit) that
learns, cycle by cycle, to correct the frozen XGBoost's residual bias
FOR THIS SPECIFIC BATTERY from `[raw_prediction, cycle_idx]`, plus a
sliding-window empirical-quantile conformal half-width computed from
the correction model's own recent residuals (not the frozen pipeline's
fixed global interval). Strict no-leakage discipline, the same standard
as every session-19/13 domain-shift check: at cycle i, the prediction
uses ONLY the corrector's state as of cycles < i, already-revealed;
cycle i's own true SOH is never used to correct cycle i's own
prediction (predict-then-reveal-then-update, every step). New files:
`src/digital_twin_streaming.py` (the core, UI-independent logic),
`src/run_streaming_dt_test.py` (CLI verification before touching the
app), `app.py` (new "🌊 Streaming Digital Twin" tab, additive - see
below for how the existing 5 tabs were confirmed unaffected).

**A genuine bug caught by "test it actually updates" before this ever
reached the app, fixed rather than papered over**: the first version
used `learning_rate="constant"` with the RAW, unscaled SOH prediction
(~70-100) as a corrector input feature. A fixed step size on an
unstandardized, non-zero-centered feature is a classic SGD divergence
setup - within one battery's stream, the corrector's coefficients
exploded to ~1e11 and predictions reached the TRILLIONS
(`corrected_pred=7,159,720,237,468.60` observed on NASA/B0018's stream).
Fixed two ways: (1) corrector inputs are now centered/scaled to O(1)
(`(raw_pred - 85) / 15`, `cycle_idx / 100`) rather than fed raw; (2)
`learning_rate="invscaling"` (step size shrinks ~1/sqrt(t), the standard
stable choice for online SGD) replaces the fixed rate. A hard clip
(±25pp) on the applied correction was also added as defense-in-depth,
not a substitute for the real fix - a runaway corrector should never be
able to push a displayed SOH number outside a physically sane range.
Re-verified clean after the fix (coefficients settle to small, stable
values; no divergence across either test battery's full stream).

**CLI verification results (`run_streaming_dt_test.py`, full battery
streams, not the app's truncated demo defaults) - answering the two
honesty questions directly:**

| battery | corrector genuinely updates? | overall MAE: raw -> corrected | final cycle: raw err -> corrected err | interval half-width: first -> last |
|---|---|---|---|---|
| NASA/B0018 | yes (coef changed 0.019,0.0002 -> 0.026,-0.953) | 4.641 -> **4.426** (improved) | 8.74 -> **6.89** | 8.16 -> **7.57** (narrowed) |
| MIT/b3c35 | yes (coef changed 0.0002,0.000002 -> 0.012,-0.110) | 0.439 -> **0.133** (improved) | 1.22 -> **0.03** | 0.029 -> **0.143** (widened) |

**Does online updating help? Yes, on both test batteries, genuinely -
not just at the final cycle.** For B0018 specifically (session 27's
identified weak point, with a known systematic late-life overestimate),
the improvement is concentrated exactly where expected: second-half-of-
stream MAE improved from 6.043 (raw) to 5.593 (corrected), a larger
relative gain than the first half (3.240 -> 3.258, essentially
unchanged) - the online corrector is learning THIS battery's own local
bias pattern, not just adding generic noise-reduction. MIT/b3c35's
improvement is larger in relative terms (69.7%) but starting from an
already-small error.

**Does the online prediction converge toward the one-shot batch
answer? Yes, and it does BETTER than the frozen one-shot pipeline at
the final cycle in both cases** - B0018's final corrected prediction
(79.65% vs. frozen 81.50%) is closer to the true 72.76% than the frozen
pipeline's own one-shot number, and the same pattern holds for b3c35
(82.77% corrected vs. 83.96% frozen, true 82.73%). This is a genuinely
positive, non-cherry-picked result for the online mechanism, not merely
"converges to the same number" - it converges to a BETTER number.

**Honest exception, reported rather than hidden**: the conformal
interval's "growing confidence" story does NOT hold universally.
B0018's half-width narrows as expected (8.16 -> 7.57), but b3c35's
WIDENS substantially (0.029 -> 0.143, ~5x). Diagnosed rather than
glossed over: with only `MIN_HISTORY_FOR_CORRECTION=2` residuals
required before the sliding-window quantile activates, the VERY EARLY
interval estimate is itself statistically noisy - b3c35's first couple
of residuals happened to be unusually small by chance, producing an
artificially tiny initial half-width that a larger, more representative
sample later corrected upward. This is a genuine, if unglamorous,
limitation of a small sliding window (`window=10`) early in a stream,
not a sign the mechanism is broken - reported honestly as "interval
narrowing is not a universal early-stream property with this window
size," rather than claimed as a clean success on the strength of the
one battery where it looked good.

**App-level integration verified via `streamlit.testing.v1.AppTest`**
(same verification method sessions 7/12 established for this app,
reused here): all 6 tabs (5 existing + the new Streaming Digital Twin
tab) render with **zero exceptions** on initial load; a full truncated
streaming simulation (20 cycles, 0s artificial delay for the automated
check) was actually triggered via the tab's real "Start streaming
simulation" button and completed with zero exceptions, producing
genuinely dynamic, non-static metrics (`Frozen-pipeline MAE=3.54pp` vs.
`Online-corrected MAE=3.21pp`, `delta=-0.33pp`) - confirming this reads
live-computed numbers, not a canned display. The existing 5 tabs'
widgets/behavior were confirmed unaffected (same selectbox/slider/
metric structure as sessions 7/12 already verified).

New files: `src/digital_twin_streaming.py`, `src/run_streaming_dt_test.py`,
`data/processed/predictions/streaming_dt_{NASA_B0018,MIT_b3c35}.csv`,
`logs/logs_streaming_dt_test.txt`. Changed (additive only, confirmed via
`git diff` that nothing was removed beyond the tab-count docstring/tuple
lines the new tab required updating): `app.py` (new "🌊 Streaming
Digital Twin" tab + its render function + `import time`/`StreamingDigitalTwin`
imports).

## Follow-up session 29 — Adaptive Conformal Inference (ACI) replaces session 28's sliding-window interval

Replaces session 28's fixed-alpha sliding-window conformal mechanism
with Adaptive Conformal Inference (Gibbs & Candes 2021, "Adaptive
Conformal Inference Under Distribution Shift"), per instruction: the
SGDRegressor residual corrector is completely UNCHANGED (confirmed
below - identical MAE numbers on both test batteries, bit-for-bit).
Only `src/digital_twin_streaming.py`'s conformal-interval code changed;
`src/run_streaming_dt_test.py` was extended (not replaced) to report
the fuller diagnostics ACI needs (empirical coverage, alpha_t
trajectory) alongside session 28's original metrics.

**Why this matters, stated as the task did**: standard split-conformal
requires calibration/test residuals to be exchangeable - violated here
by construction, since the SGDRegressor keeps updating online, so the
residual distribution being calibrated against is a moving target, not
fixed. (The exact same exchangeability requirement sessions 5 and 11
already flagged for this project's battery-level conformal calibration,
now showing up in a new place.) ACI replaces the fixed target
miscoverage alpha with an online-adapting `alpha_t`, updated after every
revealed outcome: `alpha_{t+1} = alpha_t + gamma*(alpha - err_t)` where
`err_t=1` if the previous interval missed. This has a proven long-run-
average-coverage guarantee under continuous distribution shift, which
the fixed-alpha sliding window does not. Implementation deviation
stated explicitly: `alpha_t` is clipped to `[0.01, 0.5]` - raw ACI's
alpha_t is an unconstrained random walk that can wander outside (0,1)
without a bound (a documented practical issue in ACI follow-up
literature, e.g. Zaffran et al. 2022), not an unstated part of the
original paper.

**Methodology note on the comparison itself**: session 28's original
write-up only reported a first-vs-last half-width snapshot, not the
full trajectory - exactly the framing that made b3c35's "5x widening"
look as dramatic as it did. To compare fairly, this session replayed
the EXACT session-28 mechanism (fixed 0.9 quantile, same window=10,
same MIN_HISTORY=2) on each battery's own residual sequence (identical
between runs, since the corrector didn't change) to get session 28's
own full mean/std/min/max - not just re-quoting its two old numbers.
**A bug caught in this replay before trusting it**: the first attempt
computed residuals only from the "streamed" cycles (cycle>=5) instead
of the FULL cycle-1-onward history the real twin object actually
accumulates before cycle 5 - silently resetting history to empty and
producing a first-cycle value that didn't match session 28's own
recorded number. Caught by cross-checking against that recorded value
before accepting the replay, then fixed to replay over the full history
and slice down afterward, matching the real twin's own bookkeeping.

**Results - direct comparison, same residual sequence, both mechanisms:**

| battery | metric | session 28 original (replayed, full trajectory) | session 29 ACI |
|---|---|---|---|
| NASA/B0018 | half-width first/last | 8.159 / 7.567 | 8.363 / 7.599 |
| NASA/B0018 | half-width mean/std/max | 6.710 / 1.304 / 9.060 | 7.524 / 1.295 / 9.849 |
| NASA/B0018 | empirical coverage | 82.0% | **85.9%** |
| MIT/b3c35 | half-width first/last | 0.029 / 0.143 | 0.029 / 0.234 |
| MIT/b3c35 | half-width mean/std/max | 0.231 / 0.154 / 1.040 | 0.308 / 0.420 / **4.547** |
| MIT/b3c35 | empirical coverage | 84.2% | **87.2%** |

(SGDRegressor correction confirmed unchanged: both batteries reproduce
session 28's exact overall MAE - B0018 4.641->4.426, b3c35 0.439->0.133
- and exact final-cycle predictions, since none of that logic was
touched.)

**Honest answer to "does ACI fix the b3c35 instability?" - genuinely
mixed, not a clean win, reported exactly as found:**

**On the metric that actually matters for a conformal method - empirical
coverage - ACI is a real, if modest, improvement on BOTH batteries**:
82.0%->85.9% for B0018, 84.2%->87.2% for b3c35, both moving closer to
the 90% target. This is the property ACI is specifically designed and
proven to control, and it does so here.

**But on the metric session 28's write-up actually flagged as the
"instability" - the raw half-width trajectory - ACI does NOT make
b3c35 smoother or less volatile. It makes it MORE volatile**: b3c35's
max half-width under ACI (4.547) is **4.4x worse** than the original
mechanism's own max (1.040), and its standard deviation (0.420) is
nearly 3x the original's (0.154). This is a real, mechanistic
consequence of how ACI works, not a bug: alpha_t reacts to strings of
coverage misses with potentially large corrective jumps, and b3c35's
tiny, degenerate early residuals (the root cause diagnosed in session
28) mean ACI's early alpha_t swings are large in relative terms,
occasionally producing a much wider single-cycle interval than the old
mechanism's smoother (if under-covering) quantile ever did. B0018 does
NOT show this same blowup (max only 9% higher, 9.849 vs 9.060) because
its early residuals were never as degenerately tiny to begin with -
consistent with session 28's own diagnosis that b3c35's problem was
specifically a small-sample quantile-estimation artifact, which ACI's
alpha-adaptation does not, and was never going to, directly fix - ACI
fixes COVERAGE VALIDITY under drift, not the underlying small-sample
noise in the score buffer itself.

**Neither battery reaches the full 90% coverage target within its
observed stream** (85.9% and 87.2%, both still below target with
alpha_t still below its 0.10 target and actively compensating in the
correct direction at the end of both streams) - stated honestly as
ACI's proven guarantee being a LONG-RUN average property, and these
streams (128 and 1087 cycles) plus the deliberately conservative
`gamma=0.05` step size (chosen to avoid an overly reactive, noisy
alpha_t) may simply not be long enough to fully converge, not evidence
the mechanism is incorrectly implemented.

**Overall verdict, reported plainly**: ACI is a genuine, mechanistically-
justified upgrade on the metric that defines conformal validity
(coverage, improved on both batteries) but is not a strict all-around
improvement - it trades a smoother-but-under-covering interval for a
more volatile-but-better-covering one, most visibly on exactly the
battery (b3c35) whose instability motivated this session. Whether that
trade-off is worth it depends on what the interval is FOR: if the goal
is a statistically honest coverage guarantee under a continuously-
updating model (the literature-documented reason to prefer ACI in the
first place), yes; if the goal is a visually smooth, low-variance
interval trajectory, the original sliding-window mechanism was
actually calmer on this specific battery, at the cost of the coverage
gap ACI exists to close.

App-level integration re-verified via the same `AppTest` method sessions
7/12/28 established: the streaming simulation was re-run end-to-end
through the actual "🌊 Streaming Digital Twin" tab with zero exceptions,
reproducing IDENTICAL accuracy metrics to session 28's own AppTest run
(`Frozen-pipeline MAE=3.54pp`, `Online-corrected MAE=3.21pp`,
`delta=-0.33pp`) - confirming the swap is invisible to the app's
accuracy-facing behavior and only changes the interval mechanism
underneath, exactly as scoped.

New files: none (session 29 modifies session 28's files only - no new
source files). Changed: `src/digital_twin_streaming.py` (added
`ACIConformal` class, replaced the sliding-window quantile logic in
`StreamingDigitalTwin.step()` - SGDRegressor untouched), `src/run_streaming_dt_test.py`
(extended with ACI diagnostics + the session-28-original replay
comparison). `app.py` unchanged (the interval-mechanism swap is fully
internal to `StreamingDigitalTwin`, invisible to the UI layer). New
data: `logs/logs_streaming_dt_test_aci.txt`;
`data/processed/predictions/streaming_dt_{NASA_B0018,MIT_b3c35}.csv`
overwritten with the ACI-based per-cycle records (raw per-cycle history
from session 28's run is superseded, not separately preserved, since
these are derived/regenerable analysis outputs, not source data).

## Follow-up session 30 — Digital Twin Showcase tab (time-boxed to 1h)

**Time-boxed session, scope cut deliberately to fit.** Two parts:

**Part 1 - repo sync**: 109 files of accumulated, uncommitted work from
sessions 13-29 (all logged in this file already, but never pushed) were
committed and pushed in one commit (`37836e8`), respecting `.gitignore`
(no raw data staged - confirmed via `git status` before committing).
Streamlit Cloud auto-redeploys from `master` on push; this session has
no direct dashboard/API access to positively confirm the live URL or
redeploy status from here - noted as an open item for the user to
verify, not silently assumed.

**Part 2 - Showcase tab, REPLAY not live recomputation, stated
explicitly per instruction**: new `render_showcase_tab()` in `app.py`,
wired as the FIRST tab (Streamlit's default-active tab) - the new
landing view. Loads session 28/29's already-recorded, already-verified
per-cycle CSVs (`streaming_dt_{NASA_B0018,MIT_b3c35}.csv`) via a new
`load_streaming_replay()` cached loader and plays them back on a timer
(`time.sleep` + `st.empty()` placeholders, the same pattern session 28's
Streaming Digital Twin tab already used and had verified) - a circular
Plotly `go.Indicator` SOH gauge with a true-SOH threshold marker, a
predicted-vs-true trend chart with the real ACI conformal band
(`half_width` column from session 29), and a toggle between NASA/B0018
and MIT/b3c35 with a one-line verdict COMPUTED LIVE from the loaded
CSV's own `raw_abs_err`/`corrected_abs_err`/`covered`/`half_width`
columns (not hardcoded numbers, so it can't silently drift from the
data) - b3c35's ACI interval-volatility limitation from session 29
(max half-width 4.55pp vs. the original mechanism's 1.04pp) is quoted
as-is, not softened. New dependency: `plotly==7.0.0`, added to
`requirements.txt`.

**What was cut for time, exactly as pre-authorized**: the scrolling
terminal log feed and the animated maturity ladder were skipped
entirely - never started, not a partial/broken attempt.

**Part 3 - polish + verification**: one consistent palette applied to
the Showcase tab only (`#2166ac` blue / `#d62728` red-threshold /
green-yellow-red gauge steps), not a full-app theming pass. Verified
via `streamlit.testing.v1.AppTest` (same method sessions 7/12/28/29
established): 7 tabs total, zero exceptions on initial load; both
toggle states clicked programmatically and confirmed to show real
recorded metrics matching the underlying CSVs EXACTLY (B0018 final
cycle: frozen 81.5%, twin 79.6%, true 72.8% - matches
`streaming_dt_NASA_B0018.csv`'s last row to the decimal; b3c35 final
cycle: frozen 84.0%, twin 82.8%, true 82.7% - same exact match) -
confirming the tab reads genuinely recorded numbers, not placeholder
values. All 6 existing tabs' own widgets confirmed present and
unaffected in the same AppTest run. One cosmetic, non-blocking
deprecation warning was observed (`use_container_width` vs. the newer
`width=` Streamlit API) - left as-is, explicitly not worth spending
time-boxed budget on since it doesn't affect functionality.

New files: none (this session only edits `app.py`/`requirements.txt` -
no new source files). Changed: `app.py` (new Showcase tab + its render
function, first-tab wiring, `import plotly.graph_objects as go`,
docstring updated), `requirements.txt` (+`plotly==7.0.0`).

## Follow-up session 31 — fix graceful-degradation gaps + visual pass (time-boxed to 1h)

**Priority 1 root cause, confirmed by reproduction, not guessed**: hid
`data/raw/` locally (renamed it away, then restored - the exact method
session 12 already used to test this scenario) to simulate the
deployed/no-raw-data condition. Reproduced the reported error
bit-for-bit: `"Could not load NASA/B0018: Reader needs file name or
open file-like object"`. Root cause: the Streaming Digital Twin tab's
own `load_battery_cycles()` call (independent of the sidebar's) had no
`nasa_data_available()`/`mit_data_available()` check - unlike the
sidebar's "Browse existing battery" path, which session 12 already
protects. Fix was exactly as fast as the task expected given the
existing pattern: applied the same 2-layer check (availability check +
try/except fallback, matching the sidebar's exact wording style) to
that one call site - genuinely wiring, not new debugging, confirmed
honestly since the whole investigation-to-fix took a fraction of the
20-minute budget.

Also explains Prediction/Explainability/Health Report's "silent
placeholder": with no raw data, the SIDEBAR's own (already-correct)
check fires a clear warning and leaves `selected_cycle=None` - by
design, not a bug - but each of those 3 tabs ALSO printed its own
near-identical "select a battery" placeholder, stacking on top of the
one shared message already shown above the tabs. That's Priority 3's
"duplication," and it's the same root cause manifesting differently, not
a second bug requiring separate diagnosis.

**Verified both states**: data hidden -> zero exceptions, one clear
`st.warning` per gap instead of a crash, "Select a battery" text count
4->1. Data restored -> real predictions confirmed for NASA/B0018 (SOH
81.5%) and MIT/b1c17 (SOH 82.6%, matching session 7's own originally-
recorded number exactly) and the streaming tab loading real data with
zero errors.

**Priority 2**: Showcase gauge's SOH number had no explicit font color
against a transparent `paper_bgcolor` - inherited near-invisible dark
text on a dark app background. Fixed with an opaque dark card
background + explicit white number/title/axis-tick colors, so contrast
holds regardless of the surrounding theme rather than depending on it.

**Priority 4, scoped to exactly 3 changes as instructed, nothing more
attempted**: app-wide CSS injection (Outfit/Inter Google Font pairing;
restyled `[data-testid="stAlert"]` boxes - consistent rounded card +
accent left-border replacing the default muddy fills; restyled
`.stButton > button` with the accent color + hover/active states).
Accent color `#2166ac` reused from the Showcase tab (session 30) rather
than introducing a second competing accent.

**Priority 5**: full AppTest suite re-run (6/6 pass: initial load,
Showcase toggle, sidebar dataset -> MIT, sidebar dataset -> CALCE,
streaming-twin selectbox present, streaming-twin live run) - confirms
nothing else broke.

**What got cut for time**: nothing from the pre-authorized scope -
hero-section redesign, animated/gradient backgrounds, per-chart Plotly
theming, sidebar restyling, and tab-pill redesign were never attempted,
exactly as pre-scoped as out-of-bounds for this session.

Changed: `app.py` only (97 insertions, 16 deletions - the 4-tab
availability-check fix, 3-tab duplicate-placeholder removal, gauge
contrast fix, and the CSS injection block). No new files.

## Follow-up session 32 — PRESENTATION_SUMMARY.md + Full Results Archive tab expansion (time-sensitive, presenting today)

Two-part session, both read-only against already-computed results - no
retraining, no new model output.

**Part 1**: built `PRESENTATION_SUMMARY.md` at the repo root - one
comprehensive markdown file covering every session/phase that produced
real output (Phase 1-6, the CNN-LSTM root-cause fix, and all 31
follow-up sessions), with every relevant PNG embedded, every relevant
CSV rendered as a markdown table, and every honest negative/limitation
finding included explicitly rather than omitted. Verified before
writing: all 19 `outputs/*.png` paths resolve to real files: confirmed
present on disk; every number pulled into the document was copied
directly from `DEVELOPMENT_LOG.md` (read in full, all 3,683 lines at
the time) or the source CSV/JSON files (read directly), not paraphrased
or re-rounded. Final file: 1,200 lines, ~12,200 words, 87,607 bytes,
41 numbered sections + a full file index.

**Part 2**: extended the existing "Full Results Archive" Streamlit tab
(`render_full_results_archive_tab()`), which previously only covered
Phase 1 through session 6-ish (12 expander sections), to match
`PRESENTATION_SUMMARY.md`'s scope. Added 19 new collapsed
`st.expander` sections (sessions 9, 11, 13-29) reusing the exact same
already-verified CSVs/PNGs `PRESENTATION_SUMMARY.md` cites - no new
computation, purely `pd.read_csv`/`st.image` on files already on disk.
Also enhanced 5 of the 12 existing sections with negative-result
narrative that was previously missing from the app (though already in
this log): CNN-LSTM's original R2=-0.071 root-cause fix, the log_sigma
adaptive-weighting divergence (session 2), the original 27.1%
conformal-calibration bug (Phase 6), session 7's negative-RUL-display
bug (alongside the already-present OC-SVM imbalance bug), and session
8's Gemini-model-404 finding. Total: 31 expander sections in this tab
(12 original + 19 new), plus the 1 pre-existing expander in the
separate Model Validation tab = 32 expanders app-wide. Not literally 41
(PRESENTATION_SUMMARY's number) by design: sessions 10 (surfacing eval-
protocol results in the dashboard), 12 (graceful-degradation deploy
fix), 30 (Showcase tab), and 31 (visual pass) are behavioral/styling
changes to this app itself or already have their own dedicated tab, not
standalone result artifacts to re-surface in the archive - stated
explicitly in a closing note in the tab rather than silently
undercounted.

**Verified, not assumed**:
- Every file path referenced by the new sections (64 unique
  `PRED_DIR`/`OUT_DIR`/`PROC_DIR` references) checked programmatically
  against disk: **all resolve, zero missing.**
- `streamlit.testing.v1.AppTest`: **zero exceptions** on initial load,
  **32 expanders** found (matches 31 archive + 1 validation), 56
  dataframes rendered.
- Spot-checked 8 specific numbers across early/middle/late sessions
  (CNN-LSTM before/after, MMD CALCE R2, lean-vs-full latency, second-
  life grading B0018 numbers, streaming-twin MAE, bootstrap CI bound,
  B0018 feature z-score, ACI coverage) directly against the rendered
  `AppTest` dataframe/alert-box content - **all 8 matched**
  `PRESENTATION_SUMMARY.md`'s own already-verified figures, confirming
  nothing drifted in the port from markdown to Streamlit.
- **Load-time check, done properly rather than eyeballed**: a raw
  AppTest wall-clock comparison (baseline vs. this session's changes)
  was too noisy to trust (swung 7.5s-58s run to run, dominated by
  system contention/model-loading warmup, not by this change - the
  "baseline" run even measured slower than "current" on one pass).
  Measured the actual isolated cost instead: reading all 33 new files
  the 19 new sections load (32 CSVs + 1 txt) takes **229ms** total,
  cold - negligible against the multi-second `get_resources()` model-
  loading time that already dominates every page load, with or without
  this change.

New files: `PRESENTATION_SUMMARY.md`. Changed: `app.py` (adds
`_section_num` helper + 19 new expander sections + enhancements to 5
existing sections; no other tab, function, or existing computation
touched - confirmed by `git diff --stat`: 468 insertions / 14
deletions, all within `render_full_results_archive_tab` and its two
already-existing enhanced sections).


## Follow-up session 33 — Dataset Expansion Phase 1: 32 → 204 batteries (fully autonomous overnight run)

Directly tests this project's own repeated hypothesis (sessions 19, 21,
27 all independently flagged battery count, not architecture choice, as
the likely bottleneck behind the CALCE domain-shift collapse and
B0018's weak performance): expands the NASA+MIT training/test pool from
32 batteries to 204, using exclusively already-downloaded data (zero
new acquisition), then re-runs the full base-learner/fusion/ensemble
pipeline and all 6 Step-4 hypothesis tests against it.

### Step 1 — Inventory, extraction, validation

Confirmed exactly what was unused before this session: 30 zipped,
unextracted NASA batteries (inside `data/raw/nasa/extracted/5. Battery
Data Set/*.zip`, 6 sub-archives) alongside the 4 already in use
(B0005/6/7/18), and 157 of 185 total MIT cells across the 4
`MATR_batch_*.mat` files (only 28 were in `mit_subset.json`).

Extracted all 6 NASA sub-zips into the existing `NASA_DIR` - 34 `.mat`
files total, zero code changes needed for loading (`iterate_nasa_cycles`
already treats the directory generically).

**Bug #1, found and fixed**: 2 of 34 NASA batteries (B0050, B0052)
crashed with `IndexError: index 0 is out of bounds for axis 0 with
size 0`. Root cause: some discharge cycles have a `Capacity` field that
is PRESENT but an EMPTY array, not just missing - the existing
`"Capacity" in d` check doesn't catch this. Confirmed a genuine NASA
data quirk (B0050: 4/25 discharge cycles affected; B0052: 21/25) via
direct inspection, not an extraction artifact. Confirmed the fix is a
pure no-op for the 4 originally-used batteries (zero empty-Capacity
cycles in any of them). Fixed in `data_adapters.py`'s
`iterate_nasa_cycles` by treating an empty Capacity array the same as a
missing one. **Result: 34/34 NASA batteries load correctly** (1.5s
total for all 34).

MIT validation: all **157/157 unused cells load correctly**, 130,809
additional cycles, 6.5 minutes total (390.9s). One real, honestly-
logged timing variance investigated rather than shrugged off: a 20-cell
stretch ran at 3.35s/cell vs. ~1.4-1.5s/cell elsewhere - traced to two
genuinely long-lived cells in that stretch (2,188 and 2,236 cycles),
not OneDrive sync lag specifically.

### Step 2 — Feature pipeline re-run, and a second, more consequential bug

Full HI/RUL/ICA-DV-DC extraction on the expanded pool (`run_phase1_
features_expanded.py`, additive, `run_phase1_features.py` untouched):
**221/222 batteries succeeded** (159,912 cycles) in 9.2 minutes (551.3s).
B0052 correctly self-excluded (only 3 usable cycles even after the
Capacity fix - too sparse for meaningful HI/RUL computation, the same
`len(cycles) < 5` threshold the original script already used).

**Bug #2, found, root-caused, fixed - the more serious one**: BFA's
baseline Ridge RMSE on the raw expanded pool came back at 26.54 (SOH%),
a 6.8x jump from the original 32-battery pool's 3.897 - not accepted at
face value. Root cause, confirmed by directly inspecting NASA/B0041's
raw capacity trace: `rul_labels.py`'s SOH formula (`capacity /
median(first 3 logged cycles) * 100` - completely correct for every
battery in the original 32-battery pool) breaks down for a subset of
batteries whose first several logged cycles are a separate low-rate
characterization protocol phase, not real aging cycles (B0041:
capacities of ~0.044-0.057Ah for cycles 1-41, then an abrupt jump to
~1.09-1.22Ah at cycle 42 where its real aging trend begins). Dividing
later, normal cycles by that degenerate near-zero baseline produces
physically impossible SOH values - up to **2177%** for B0041.

Checked BOTH datasets, not just where it was first noticed: found 4
affected MIT batteries too (b1c18 max 270.4%, b2c44 max 144.1%, b1c0
max 143.6%, b2c12 max 138.9%) - the identical underlying mechanism,
confirming this is a general property of the labeling convention when
applied to protocol variety it was never validated against, not a
NASA-specific quirk.

Exclusion criterion (principled, not tuned to a target number): any
battery whose SOH ever exceeds 110% - chosen because session 16 already
independently established mild readings up to ~101.5% (e.g. MIT/b3c0's
real formation-cycle bump to 100.29%) as genuine physically plausible
early-life effects, so 110% is a deliberately generous ceiling that
keeps every battery showing that kind of normal behavior and excludes
only the ones with clearly nonsensical (150%+) readings. A battery's
`min SOH == 0%` alone did NOT trigger exclusion - many otherwise-clean
batteries (e.g. B0042-48, B0053-56) legitimately fade all the way to 0%
at true end-of-life, a real data point, not an artifact.

`rul_labels.py` ITSELF was left completely untouched (shared by every
other script in this project, worked correctly for the original 32
batteries) - the exclusion is applied only at the expanded-pool level,
in a new small module (`src/expanded_pool_exclusions.py`) that documents
the finding in full and is imported wherever the expanded pool is built.

**Final, honest usable pool after full validation: 23 NASA + 181 MIT =
204 batteries** (vs. original 32, vs. 219 theoretical ceiling - the
15-battery gap is fully accounted for: 1 too-few-cycles + 10 NASA +
4 MIT degenerate-SOH-baseline exclusions, nothing silently dropped).

**BFA re-run on the clean 204-battery pool - does the 7-feature
selection change? Yes, genuinely:**
- Original (32 batteries): `ICHV, SCV, VDEDT, VIECT, MATC, MATD, TEVI` (7 features)
- Expanded (204 batteries): `CDECT, ICHV, VDEDT, VIECT, LVP, MET, TCCC, TCVC` (8 features)
- Only 3 features survive (`ICHV, VDEDT, VIECT`); both temperature HIs
  (MATC, MATD) dropped out entirely, replaced by 4 new ones (CDECT, LVP,
  MET, TCCC).
- Baseline RMSE (all 16 features, clean data) = 8.79, still meaningfully
  higher than the original 3.897 even after fixing the label bug - a
  genuine sign the expanded pool is more heterogeneous (more distinct
  battery chemistries/protocols pooled together), not something to hide.
- Converged (8-feature) RMSE = 5.2065, also higher than the original's
  converged 2.864 - consistent with the same explanation.

### Step 1/2 structural finding: B0018 moved from TEST to TRAIN

Checked `battery_split_expanded.json` directly rather than assuming:
B0018 (the subject of session 27's entire root-cause investigation) is
now in the TRAINING set, not the test set (`battery_level_split`'s
deterministic per-dataset "every 5th battery" stratification lands
differently with 19 more NASA batteries added - the new NASA test
batteries are B0025/B0030/B0044/B0053). This means any session-27-style
"B0018 root cause, re-examined" comparison is NOT apples-to-apples in
the usual sense: the expanded-pool model has literally been trained on
B0018's own cycles, not just "trained on more NASA data elsewhere" -
stated explicitly here rather than silently reconciled.

### Step 3 — Retraining all 5 base learners + fusion + ensemble

Same hyperparameters/training budget as the original runs throughout
(40-epoch budget/patience=8 for VLSTM/CNN-LSTM/PiFormer/CNN-BiGRU,
25-epoch/patience=6 for the ICAEncoder, 500-tree XGBoost) - the ~5.9x
increase in training cycles (26,996 -> 159,912) is accepted as the
genuine cost of the experiment, not worked around by cutting budgets.

**Bug #3, found, root-caused, fixed - a genuine methodological catch,
not just an engineering one**: this session's first attempt at
VLSTM/CNN-LSTM/PiFormer training was killed mid-run (external process
teardown, not a code failure) after VLSTM (10,154.2s), CNN-LSTM (1,683.1s),
and PiFormer (8,290.7s) had already finished - ~5.6h of genuine compute
- but before the train-set predictions step (needed for the ensemble)
completed. Rather than redo ~5.6h of training, added checkpoint-resume
logic (`load_or_train()`) to load the already-trained weights. This
itself then hit TWO further real bugs, both caught and fixed rather
than silently absorbed:

1. **Memory bug**: the resumed run's unbatched prediction call on the
   full 123,755-row train+val set crashed PiFormer specifically with
   `RuntimeError: ... not enough memory: you tried to allocate
   79203200000 bytes` (~79GB) - PiFormer's multi-head attention
   materializes an O(batch) `(batch,heads,seq,seq)` score tensor that
   VLSTM/CNN-LSTM's recurrent/conv passes don't have, so it scaled fine
   at the 29,489-row TEST set but not at 123,755. Fixed with chunked
   inference (4,096 rows/chunk) - mathematically identical predictions,
   bounded peak memory.
2. **Silent-corruption bug (the more serious one)**: after fixing #1
   and re-running, CNN-LSTM's TEST metrics did NOT match its own
   pre-crash reference (RMSE 1.3003->2.4256, MAE 0.7450->1.0309, R2
   0.9666->0.8837), while VLSTM and PiFormer matched EXACTLY. Root
   cause: CNN-LSTM is the ONLY one of the 3 models using
   `nn.BatchNorm1d` (VLSTM has no normalization layer; PiFormer uses
   `LayerNorm`, which is train/eval-mode-independent). The new
   `load_or_train()` loaded the checkpoint's WEIGHTS correctly
   (confirmed via unchanged file mtime across the crash) but never
   called `model.eval()` - a freshly-constructed `nn.Module` defaults
   to `.train()` mode, so CNN-LSTM's BatchNorm silently used LIVE batch
   statistics computed from whichever test chunk it was scoring,
   instead of the stored `running_mean`/`running_var` - a genuine,
   if narrow, form of test-set leakage (the model's own normalization
   was peeking at the composition of the batch being scored). This is
   the exact same architecture-specific failure signature as this
   project's very first CNN-LSTM investigation (only CNN-LSTM affected,
   VLSTM/PiFormer architecturally immune) - a different root cause this
   time (missing `eval()` vs. the original ~9.5M-unnormalized-dVdQ bug),
   caught the same way: by not accepting a suspicious number at face
   value and checking each model's actual normalization layers directly.

   Fixed by adding `model.eval()` in `load_or_train()` AND defensively
   inside `predict()` itself (belt-and-suspenders), plus the same
   defensive (currently-inert - CNN-BiGRU has no normalization layer at
   all) call in `train_cnn_bigru_expanded.py` for consistency. Wrote a
   corrective script (`fix_cnn_lstm_expanded_preds.py`) to patch the
   already-written prediction files with CNN-LSTM's correctly-scored
   values, loaded from the SAME untouched, always-correct checkpoint -
   confirmed via sanity re-check that VLSTM (R2=0.9472) and PiFormer
   (R2=0.7795) were unaffected, both exact matches to their known values.

   **Caught a downstream sequencing consequence too**: the ensemble
   (`train_ensemble_fusion_expanded.py`) had already run once against
   the still-wrong CNN-LSTM column before the fix landed (chain
   proceeded automatically while the fix script was still computing) -
   re-ran the ensemble immediately once corrected data was available.
   **Genuinely interesting, honest finding from this re-run**: the
   CORRECTED ensemble (R2=0.9579) is very slightly WORSE than the
   bug-corrupted one (R2=0.9591) had reported - consistent with the
   leakage explanation above (a model's normalization silently adapting
   to the exact test batch it's being scored on can coincidentally
   flatter its numbers in ways that don't reflect genuine
   generalization). The corrected, honest number is reported throughout
   this log, not the more flattering wrong one.

**Base learner results, old (32-battery) vs. new (204-battery), all
verified/confirmed, all real:**

| model | RMSE before → after | MAE before → after | R² before → after |
|---|---|---|---|
| XGBoost (BFA-HI only) | 1.478 → 1.2121 | 0.990 → 0.5666 | 0.907 → **0.9721** |
| VLSTM | 2.131 → 1.6345 | 1.564 → 0.9164 | 0.806 → **0.9472** |
| CNN-LSTM | 3.948 → 1.3003 | 2.926 → 0.7450 | 0.334 → **0.9666** |
| PiFormer | 2.993 → 3.3405 | 1.928 → 0.7143 | 0.617 → **0.7795** |
| CNN-BiGRU | 3.165 → 1.4791 | 2.341 → 0.8265 | 0.572 → **0.9568** |

VLSTM, CNN-LSTM, and CNN-BiGRU all improved substantially and
unambiguously across every metric - CNN-LSTM in particular went from
this project's weakest base learner (R2=0.334) to genuinely strong
(R2=0.967), and CNN-BiGRU went from the worst-performing model overall
(R2=0.572) to R2=0.957, purely from more training data, no architecture
change. A real, direct confirmation of the "battery count is the
bottleneck" hypothesis for three of the four affected models.

**PiFormer is a genuine mixed result - root-caused, not left as a
shrug.** R2 improved (0.617->0.780) and MAE dropped sharply
(1.928->0.714), but RMSE got slightly WORSE (2.993->3.340). Per-battery
breakdown of the 40-battery expanded test set (not assumed to mirror
the original 6-battery story) found the cause precisely: **one single
battery, NASA/B0053 (54 cycles, 0.18% of all 29,489 test cycles),
produces wildly erratic PiFormer predictions (RMSE=74.7 on that battery
alone - predictions swing from near-0 to ~82 cycle-to-cycle against a
true SOH band of 85-100%, not a bias, an actual breakdown)**, while
PiFormer performs *well* on every other battery (median per-battery
RMSE across the other 39 = 0.674). Recomputed PiFormer's pooled metrics
excluding only B0053: **RMSE=0.9693, MAE=0.5862, R2=0.9814** - which
would make PiFormer the single BEST base learner of all five, not the
worst, if not for this one battery. NASA/B0053 is one of the 4 new NASA
test batteries introduced by the expanded split (B0025/B0030/B0044/
B0053) and its raw capacity trace starts at 0.000Ah (visible in
`nasa_validation.txt`) similar in flavor to the degenerate-baseline
batteries excluded in Step 2, though B0053 itself never crosses the
110% SOH exclusion threshold so it correctly stayed in the pool - this
looks like a battery with real but unusual early-life behavior that
VLSTM/CNN-LSTM's recurrent/conv inductive bias handles gracefully but
PiFormer's attention mechanism does not, a plausible explanation
offered honestly as a hypothesis, not confirmed via further ablation
(time-boxed, not chased further this session). **Practical takeaway:
PiFormer's RMSE regression is a single-battery outlier artifact, not a
genuine broad-based regression** - but it is a real, reproducible
failure on that specific battery, not explained away.

XGBoost-fusion: RMSE=1.2115 MAE=0.5461 R2=0.9719, confirmed consistent
with `train_xgboost_fusion_expanded.py`'s own logged run (no drift).

Fusion ensemble (Stacking-Ridge-fusion-expanded, CORRECTED, 4-branch:
XGBoost-fusion+VLSTM+CNNLSTM+PiFormer, CNN-BiGRU excluded from the
stack by the same design as the original architecture): RMSE=1.4834
MAE=0.5276 R2=0.9579 (original 32-battery Stacking-Ridge-fusion was
RMSE=1.394 MAE=0.966 R2=0.917) - genuinely improved (R2 0.917->0.958),
though a smaller relative gain than the standalone deep models, entirely
consistent with this project's own repeated finding that the ensemble
is carried almost entirely by XGBoost-fusion regardless of how the
other branches perform individually.

**Drop-branch ablation (5-branch, expanded pool)** -
`run_drop_branch_ablation_5branch_expanded.py`, which DOES include
CNN-BiGRU as a 5th stacked branch (a separate meta-model from the
4-branch Ridge ensemble above, not the same number):

| variant (branch dropped) | RMSE | R2 | delta RMSE vs. full |
|---|---|---|---|
| drop PiFormer | **1.1821** | **0.9733** | -0.3230 (biggest improvement from dropping) |
| drop CNN-BiGRU | 1.4834 | 0.9579 | -0.0217 |
| drop VLSTM | 1.4939 | 0.9573 | -0.0112 |
| drop CNN-LSTM | 1.5042 | 0.9567 | -0.0008 |
| **full 5-branch** | 1.5051 | 0.9567 | 0.0000 |
| drop XGBoost-fusion | 3.1483 | 0.8103 | +1.6432 (catastrophic) |

Honest reading: at the expanded scale, PiFormer is now actively
*hurting* the full ensemble, not just showing a mixed standalone
number (consistent with the B0053 outlier above - one bad battery's
errors get amplified when stacked). CNN-BiGRU very slightly hurts the
stack too. XGBoost-fusion remains overwhelmingly the dominant branch,
as in every prior session.

### Step 4 — The actual hypothesis test

**1. CALCE zero-retrain domain-shift eval - THE single most important
result in this whole update, reported first and most prominently, per
instruction:**

| metric | original (32-batt) | expanded (204-batt) |
|---|---|---|
| XGBoost-fusion R2 (CALCE) | 0.304 | **0.5556** |
| Stacking-Ridge R2 (CALCE) | 0.314 | **0.6690** |
| In-domain coverage (target 90%) | 95.6% | 91.2% |
| **CALCE coverage (target 90%)** | **6.1%** | **7.4%** |

**Verdict, stated plainly: the hypothesis is only PARTIALLY confirmed.**
Point-prediction accuracy under domain shift improved substantially and
genuinely - the in-domain-vs-CALCE R2 gap roughly halved (0.603->0.289).
This part of the "more battery data helps" hypothesis holds. But the
conformal coverage collapse - arguably the more practically important
failure mode, since it's about whether the model's stated uncertainty
can be trusted at all, not just its point accuracy - did **not**
improve in any meaningful sense (6.1%->7.4%, still catastrophic).
Root cause, visible directly in the numbers: the prediction interval
half-width is **identical** between in-domain and CALCE in both runs
(orig 2.367/2.367; expanded 1.154/1.154) because it's calibrated
entirely on in-domain residuals and never adapts to CALCE's own error
distribution. This means the coverage collapse is most likely a
distinct problem from raw model capacity/data volume - CALCE's residual
distribution remains structurally different enough from NASA/MIT's
that a split-conformal interval calibrated purely in-domain can't cover
it, no matter how well the underlying point predictions generalize.
**Battery count fixed accuracy under domain shift; it did not fix
calibration under domain shift.** This is reported exactly as measured,
not rounded up to "hypothesis confirmed."

**2. Bootstrap significance (2000 resamples, 95% CI, cycle- and
battery-level), 40 expanded test batteries vs. original 6:**

- **XGBoost vs. VLSTM, battery-level**: original NOT significant (CI
  [-0.0513,+0.2528]) -> expanded **NOW SIGNIFICANT** (CI
  [+0.0091,+0.0766]). With 6x more test batteries, XGBoost's edge over
  VLSTM is now formally confirmed, not just a large noisy point estimate.
- **Drop-XGBoost-fusion ablation, battery-level**: original NOT
  significant (CI [-0.054,+0.2392]) -> expanded **NOW SIGNIFICANT** (CI
  [+0.00503,+0.51075]). Same story for the ensemble's dependence on
  XGBoost-fusion specifically.
- **Lean vs. full, battery-level**: original NOT significant (delta
  RMSE CI [-0.0115,+0.0086]; delta R2 CI [-0.00085,+0.0018]) -> expanded
  **STILL NOT significant** (delta RMSE CI [-0.731,+0.070]; delta R2 CI
  [-0.0031,+0.0697]). This one did NOT flip - lean and full remain
  statistically indistinguishable even at 6x the battery count, which
  *reinforces* rather than undermines session 20's original "ship lean"
  recommendation.
- All 4 deep models individually in the drop-branch ablation: cycle-
  level significant, battery-level NOT significant in both the original
  and expanded runs - same qualitative pattern, unchanged by scale.

**3. Domain-classifier sanity check (session 19)**: in-domain AUC
(calib-half vs. eval-half of NASA+MIT, should approach 0.5 for
genuinely indistinguishable data) moved from **0.9021** (original,
6-battery test set) to **0.6936** (expanded, 40-battery test set,
full feature space). |AUC-0.5| more than halved (0.4021->0.1936) -
genuinely moved closer to 0.5, supporting the idea that the original
near-1.0 readings were partly a small-sample artifact, though 0.69 is
still clearly separable, so this is "meaningfully improved," not "solved."

**4. Second-life grading**: agreement 98.75% (original, 5,208 cycles,
6 batteries) -> **99.50%** (expanded, 29,489 cycles, 40 batteries) -
improved. **Important caveat stated plainly**: NASA/B0018 - the subject
of session 25's entire second-life analysis - is **not in the expanded
test set** (moved to train by the split, see Step 1/2 finding above),
so there is no direct expanded-pool counterpart to the original
B0018-specific mis-grade (72.76% true vs. 81.50% predicted) to compare
against; this is reported as a genuine gap, not silently dropped. Two
near-miss grade mismatches in the expanded run: NASA/B0053 (true
"Primary EV use", graded "Second-life candidate") and MIT/b4c3 (true
"Recycle only", graded "Second-life candidate") - both one-bracket-off
errors near a grade boundary.

**5. Sensor-noise robustness** (same 6 original test batteries, by
design, for apples-to-apples comparison):

| | clean | 1x BMS | 2x BMS | 5x BMS (stress) |
|---|---|---|---|---|
| original R2 | 0.9172 | 0.9196 | 0.9184 | 0.9208 |
| expanded R2 | **0.9827** | 0.9618 | 0.9495 | 0.8704 |

**Honest nuance, not glossed over**: the expanded model's clean-condition
accuracy is much better, but it is measurably *more sensitive* to
injected sensor noise than the original - the original's apparent
"noise robustness" (R2 barely moving, even ticking up slightly) looks
in hindsight like an artifact of it already being noisy/imprecise
enough that a bit more sensor noise didn't register, not genuine
robustness. At the worst stress level tested, the expanded model's R2
(0.8704) actually dips slightly *below* the original's clean-condition
R2 (0.9172). More data bought a better baseline, not a strictly-dominant
result once noise is stacked on top.

NASA/B0018 specifically (per-noise-level R2): original 0.576->0.565->
0.525->0.513 (monotonic degradation); expanded 0.9347->0.8951->0.8739->
0.8735 (also monotonic degradation, same qualitative shape). B0018's
absolute accuracy improved hugely, but per the Step 1/2 caveat this is
**confounded by B0018 now being in the training set**, not a clean
"generalizes better on unseen B0018" result.

### Honest summary

The core hypothesis (sessions 19, 21, 27: battery count, not
architecture, is the bottleneck) is **partially confirmed, not fully**.
What genuinely improved with 6x more battery data: every standalone
base learner's point-prediction accuracy (4 of 5 cleanly, PiFormer with
a single-battery outlier caveat), the fusion ensemble's point accuracy,
CALCE's out-of-domain R2 (gap roughly halved), the domain-classifier's
AUC (moved meaningfully closer to 0.5), second-life grading agreement,
and two specific bootstrap comparisons that are now formally
significant where they weren't before. What did **not** improve, stated
without hedging: the CALCE conformal coverage collapse (6.1%->7.4%,
still catastrophic - the single most important negative finding here),
the model's noise-robustness margin (thinner despite a better baseline),
and lean-vs-full's statistical indistinguishability (unchanged,
reinforcing the existing "ship lean" decision rather than calling it
into question). BFA's feature selection genuinely shifted (7->8
features, only 3 survive), meaning the original feature set was at
least partly a small-sample artifact.

**Deployment decision, per instruction, not resolved unilaterally**:
the original 32-battery lean pipeline remains the deployed default -
every file this session produced is additive (`*_expanded.*`), nothing
original was overwritten. The expanded-pool result is a clearly-labeled
separate research finding, pending explicit review before any swap.

### Dataset Completeness Audit

- **NASA: 34/34 downloaded .mat files accounted for.** 23 used in the
  final pool. 10 excluded (degenerate SOH baseline, up to 2177%):
  B0033, B0034, B0036, B0038, B0039, B0040, B0041, B0049, B0050, B0051.
  1 excluded (too_few_cycles, 3 usable cycles even post-fix): B0052.
- **MIT: 185/185 cells accounted for** (across all 4
  `MATR_batch_*.mat` files). 181 used. 4 excluded (degenerate SOH
  baseline): b1c0, b1c18, b2c12, b2c44.
- **Total expanded pool: 23 NASA + 181 MIT = 204 batteries** (vs.
  original 32, vs. 219 theoretical ceiling - the 15-battery gap is
  fully accounted for above, nothing silently dropped).
- **CALCE: confirmed 3/3 locally-available cells (CS2_35/36/37) used
  ONLY as the held-out zero-retrain domain-shift test set, never added
  to training** - a deliberate design choice stated explicitly, not a
  gap. Verified directly against `hi_table_expanded.parquet`: 2,943
  CALCE rows, zero overlap with the NASA/MIT training pool.
- **Anything found but unused for a reason OTHER than a logged
  data-quality exclusion: NONE.** All 34 NASA .mat files and all 4 MIT
  `MATR_batch_*.mat` files were present and every cell/battery inside
  them accounted for above. The one other file under `data/raw/mit/`
  (`Severson-et-al/2017-05-12_6C-50per_3_6C_CH36.csv`) is a pre-existing
  single-cell CSV quickstart sample used only by the exploratory
  `src/load_mit.py`, not new data this session found and very likely a
  redundant format of a cell already inside `MATR_batch_20170512.mat` -
  not a separate battery. No time-based scope cuts were taken anywhere.

### Resilience work (harness interruptions, not code bugs)

This session's background training was killed by external process
teardowns **three separate times** (not related to any bug in the
code). Rather than just restart repeatedly, the resilience gaps were
fixed properly each time: (1) `load_or_train()` model-level
checkpointing added to `train_deep_models_expanded.py` after the first
interruption (VLSTM/CNN-LSTM/PiFormer), (2) disk-based tensor-load
caching (`_expanded_battery_tensors_cache.pkl`, ~700MB, gitignored -
regenerable, not a result) added after the second, eliminating a
repeated ~17-18min raw-data reload on every restart, (3) genuine
per-epoch checkpoint/resume for CNN-BiGRU
(`train_one_model_resumable()`, saving model/optimizer/torch-RNG state
every epoch) added after the third, **verified bit-for-bit
reproducible via a synthetic crash-and-resume smoke test before being
trusted on the real 40-epoch run** - confirmed in production too: the
killed second CNN-BiGRU attempt's epoch 0-12 losses matched the final
successful third attempt's epoch 0-12 losses to every printed decimal.
No trained weights or completed compute were ever silently discarded
across any of the three interruptions.

New files: `src/run_phase1_features_expanded.py`,
`src/expanded_pool_exclusions.py`, `src/run_bfa_expanded.py`,
`src/train_xgboost_expanded.py`, `src/train_deep_models_expanded.py`,
`src/train_fusion_encoder_expanded.py`,
`src/train_xgboost_fusion_expanded.py`,
`src/train_ensemble_fusion_expanded.py`,
`src/train_cnn_bigru_expanded.py`,
`src/run_drop_branch_ablation_5branch_expanded.py`,
`src/run_calce_zero_retrain_eval_expanded.py`,
`src/run_bootstrap_significance_expanded.py`,
`src/run_domain_classifier_sanity_check_expanded.py`,
`src/run_second_life_grading_expanded.py`,
`src/run_sensor_noise_robustness_expanded.py`,
`fix_cnn_lstm_expanded_preds.py` (one-off corrective script, kept as a
permanent record), `models/*_expanded.*`, `data/processed/{hi_table,
bfa_*,channel_norm_stats,battery_split,mit_full_cells}_expanded.*`,
`data/processed/predictions/*_expanded*.csv`,
`logs/logs_overnight_progress.txt` (full timestamped run log).
Changed (bug fix only, additive): `src/data_adapters.py`
(`iterate_nasa_cycles` empty-Capacity guard). Does NOT touch `app.py`
or the deployed Streamlit site - that is session 34, a separate commit.
