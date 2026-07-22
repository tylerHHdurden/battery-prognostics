# Battery Prognostics Pipeline — Final Summary

Consolidated results from the full 6-phase pipeline (16 Health Indicators
→ BFA feature selection → 4 base learners → stacking ensemble → joint
SOH+RUL adaptive-loss ablation → SHAP explainability → split-conformal
prediction), run on NASA PCoE + CALCE CS2 + a 28-cell MIT/Stanford/TRI
subset, followed by a targeted root-cause fix session. Full chronological
narrative, every assumption, and every intermediate (including the
pre-fix numbers) is in `OVERNIGHT_LOG.md`; this file is the consolidated,
final-numbers view for review.

---

## 1. Base learner comparison (final, post-fix)

Trained on NASA + a 28-cell MIT subset, split by battery (not by
shuffled cycles), test set = 6 held-out batteries (1 NASA + 5 MIT).
Target: SOH (%).

| model | RMSE | MAE | R² |
|---|---|---|---|
| **XGBoost** | **1.478** | **0.990** | **0.907** |
| VLSTM | 2.131 | 1.564 | 0.806 |
| PiFormer | 2.993 | 1.928 | 0.617 |
| CNN-LSTM | 3.948 | 2.926 | 0.334 |

XGBoost (trained on 7 BFA-selected Health Indicators) is the strongest
model by a clear margin. Among the three deep sequence models, VLSTM >
PiFormer > CNN-LSTM. CNN-LSTM's R² of 0.334 reflects a genuine fix (it
started at **-0.071**, worse than predicting the mean — see Debugging
Highlights below); it is still the weakest of the four, which is a
separate, unresolved fact about its capacity/optimization at this scale,
not a residual bug.

## 2. Ensemble vs. individual: stacking does not beat XGBoost — legitimately

| model | RMSE | MAE | R² |
|---|---|---|---|
| XGBoost (best individual) | 1.478 | 0.990 | 0.907 |
| Stacking-Ridge | 1.481 | 0.998 | 0.906 |
| Stacking-XGBoost | 1.491 | 0.994 | 0.905 |

Ridge meta-learner coefficients: `{XGBoost: 1.010, VLSTM: 0.007,
CNN-LSTM: -0.011, PiFormer: -0.006}`, intercept -0.042 — over 99% of the
ensemble's prediction is XGBoost alone.

**Why this is a legitimate result, not a failure of stacking**: even
after CNN-LSTM was fixed and genuinely learning, every deep model is
still meaningfully behind XGBoost (best deep model R²=0.806 vs XGBoost's
0.907). A linear meta-learner correctly declines to blend in
systematically weaker, correlated-error predictors — stacking only earns
its keep when base learners are of comparable strength, which isn't the
case here. This was checked twice: the ensemble was re-run after the
CNN-LSTM fix specifically to see whether a genuinely-contributing fourth
learner would change the outcome. It didn't.

## 3. Joint SOH+RUL ablation: clean single-task collapse; adaptive-vs-fixed is an honest split decision

| variant | SOH RMSE | SOH R² | RUL RMSE | RUL R² |
|---|---|---|---|---|
| fixed_balanced (α=β=0.5, fixed) | 3.695 | **0.416** | 253.73 | 0.428 |
| soh_only (α=1, β=0) | 4.759 | 0.032 | 330.29 | **0.030** |
| rul_only (α=0, β=1) | 4.932 | **-0.040** | 263.83 | 0.381 |
| adaptive (learned, clamped) | 3.918 | 0.344 | 252.80 | **0.432** |

**Single-task collapse is textbook-clean in both directions**:
- `soh_only` — RUL head never receives gradient (β=0) — collapses on
  RUL: R²=0.030, essentially uninformative, vs. 0.428/0.381/0.432 for
  every variant that actually trains the RUL head.
- `rul_only` — SOH head never receives gradient (α=0) — collapses on
  SOH even more starkly: **R²=-0.040, actually negative** (worse than
  predicting the mean), vs. 0.416/0.032/0.344 for the other three.

**Adaptive weighting: a genuine split decision, not a clean win.**
After fixing the initial runaway-divergence bug (see Debugging
Highlights), adaptive now **wins on RUL** (R²=0.432 vs fixed_balanced's
0.428) but **still loses on SOH** (0.344 vs 0.416).

**Mechanistic explanation — alpha=beta=2.028**: the learned α and β
converged to the *exact same value* (2.028, the clamp ceiling) rather
than an asymmetric split reflecting genuine SOH-vs-RUL uncertainty. This
means the homoscedastic-uncertainty parametrization used here mostly
learned "how confident am I overall" (scaling both task weights up
together by the same ~4× factor) rather than "how should I trade off
SOH against RUL." That's why the result is a wash rather than a clean
win: uniformly upweighting both losses changes the effective gradient
scale, which happened to help RUL slightly and hurt SOH slightly, but
it isn't the asymmetric task-rebalancing the method is meant to provide.

## 4. SHAP physics validation

**TreeSHAP, XGBoost base learner** (mean|SHAP| over the 7 BFA-selected
HIs): SCV=2.266, VIECT=1.934, TEVI=0.905, ICHV=0.244, MATC=0.199,
MATD=0.136, VDEDT=0.066. Top-3 (SCV, VIECT, TEVI) are exactly the
voltage-shape features BFA's own convergence history favored —
independent cross-validation that BFA's selection wasn't a fluke.

**TreeSHAP, Stacking-XGBoost meta-learner**: pred_XGBoost=4.138,
pred_PiFormer=0.0043, pred_VLSTM≈0, pred_CNNLSTM≈0 — confirms the Ridge
coefficients finding independently: >99.9% "just XGBoost."

**Voltage-region concentration** (fraction of total |SHAP| mass falling
in the 3.55–3.8V window — the discharge voltage plateau region — using
DeepSHAP for all three deep models, no KernelSHAP fallback needed):

| model | fraction of |SHAP| mass in [3.55, 3.8]V |
|---|---|
| VLSTM | **0.601** |
| CNN-LSTM | 0.354 |
| PiFormer | 0.117 |

VLSTM concentrates 60% of its attribution in a ~0.25V window — a real,
physically-grounded finding (this is the diagnostically-rich plateau
region for these cells). CNN-LSTM's 0.354 is itself a validation result:
pre-fix, this number was 0.000 (a broken, near-constant model has no real
gradient signal for DeepSHAP to attribute anywhere) — post-fix, it's a
meaningful, interpretable value in the same ballpark as VLSTM's,
independent confirmation (via a completely different method) that
CNN-LSTM is now a genuinely learning model. PiFormer's low 12%
concentration reflects its cross-channel attention spreading importance
across the full voltage range and all ICA channels rather than
localizing narrowly — a different but equally legitimate explanation
pattern, not a failure of the analysis.

## 5. Split-conformal prediction coverage

| target | method | target coverage | empirical coverage | avg interval width | n |
|---|---|---|---|---|---|
| SOH (Stacking-Ridge) | MAPIE split-conformal | 90% | **95.1%** | 4.64 (SOH %) | 3,462 |
| RUL (joint-adaptive) | MAPIE split-conformal | 90% | **88.9%** | 871.4 (cycles) | 3,462 |

SOH slightly over-covers (the safe direction for finite-sample
split-conformal). RUL slightly under-covers, improved from 83.3% to
88.9% as a side effect of the CNN-LSTM/joint-model fix (not from any
change to the calibration method itself).

---

## Known Limitations

1. **PiFormer underperforms VLSTM** (R²=0.617 vs 0.806) among the
   converged deep models. Not broken — it trains and converges cleanly,
   just with lower final accuracy at this model scale/epoch budget.
   Left as-is per explicit instruction; not investigated further.
2. **RUL conformal coverage (88.9%) is still slightly under the 90%
   target.** Improved substantially as a side effect of the CNN-LSTM
   fix, but not independently investigated or corrected — a real,
   open calibration gap.
3. **Adaptive loss weighting's homoscedastic parametrization has a
   structural limitation**: even correctly bounded (log_sigma clamped
   to [-0.7, 0.7]), it converged to α=β rather than an asymmetric
   trade-off, so it only partially delivers the task-rebalancing benefit
   the method is meant to provide. A structurally different scheme
   (e.g. GradNorm, or a softmax-normalized α+β=1 constraint that forces
   an actual trade-off) is the natural next step — explicitly out of
   scope for this round.

## Debugging Highlights

Three real bugs were caught and fixed during this project, each of which
would have silently produced wrong conclusions if shipped as originally
written:

1. **CNN-LSTM's unnormalized `dV/dQ` input channel.** Root-caused (not
   guessed) via a specific diagnostic trail: `.eval()` was confirmed
   correctly called before validation; BatchNorm running stats were
   confirmed to be updating; but their converged *values* were
   nonsensical (`running_var` ~1e14–1e15). Traced to the raw `dV/dQ`
   channel reaching **~9.5 million** in places (division-by-near-zero
   near flat-capacity plateaus) while every other channel sat at
   O(1–100) — a 5–6 order-of-magnitude scale mismatch that was never
   normalized. VLSTM (no BatchNorm) and PiFormer (LayerNorm, per-sample)
   were architecturally immune, which is exactly why only CNN-LSTM
   broke despite all three seeing the same data. Fixed with per-channel
   percentile-clipped z-scoring, fit on training data only. Result:
   CNN-LSTM's R² went from **-0.071 to +0.334**.
2. **A conformal-calibration exchangeability violation.** The first
   conformal run calibrated on the training split's own residuals — but
   the meta-learner had been *fit* on those exact rows, so its in-sample
   residuals were ~6.7× smaller than genuine held-out residuals. This
   produced an empirical coverage of **27% against a 90% target**.
   Verified it wasn't a MAPIE-library bug by reimplementing
   split-conformal from scratch and getting the identical 27% — then
   fixed by splitting the held-out test batteries into a genuine
   calibration half and evaluation half, bringing coverage to a
   legitimate **95.1%**.
3. **An OneDrive sync-lag trap.** After retraining the deep models, a
   downstream ensemble script read a prediction file whose Ridge
   meta-learner coefficients came back numerically *identical* to the
   pre-fix run — implausible for a closed-form fit on genuinely
   different data. Investigated the file's timestamp and content rather
   than trusting the script's exit code: the file was still the
   pre-retrain version, despite the writing script having exited
   cleanly minutes earlier (a background sync/materialization delay on
   the OneDrive-hosted project folder). Caught by content inspection,
   not by any error message, and re-verified fresh before trusting
   downstream numbers.

## Plots (all in `outputs/`)

| file | shows |
|---|---|
| `phase1_bfa_convergence.png` | BFA best-fitness and selected-feature-count vs. iteration (30 agents × 100 iterations) |
| `phase1_soh_fade_examples.png` | Sample SOH-vs-cycle fade curves for NASA, CALCE, and MIT cells, with the 80% EOL threshold marked |
| `phase1_ica_dv_dc_example.png` | dQ/dV, dV/dQ, dI/dV curves vs. voltage bin at several cycles of NASA B0005, showing fade-driven curve shift |
| `phase2_deep_model_training_curves.png` | Train/val loss curves for VLSTM, CNN-LSTM, and PiFormer (post-fix) |
| `phase3_ensemble_comparison.png` | RMSE bar chart, all 4 base learners vs. both stacking meta-learners |
| `phase3_stacking_parity_plot.png` | Predicted vs. true SOH scatter for the Stacking-Ridge ensemble on the test set |
| `phase4_joint_ablation_curves.png` | Validation SOH loss and validation RUL loss vs. epoch, all 4 loss-weighting variants overlaid |
| `phase4_joint_ablation_bars.png` | Final test RMSE bar charts (SOH and RUL) across the 4 ablation variants |
| `phase4_adaptive_alpha_beta.png` | Learned α (SOH weight) and β (RUL weight) vs. epoch for the adaptive variant — shows both climbing together and pinning at the 2.028 clamp ceiling |
| `phase5_shap_xgboost_ranking.png` | Mean |SHAP| bar chart for XGBoost's 7 BFA-selected Health Indicator features |
| `phase5_shap_meta_ranking.png` | Mean |SHAP| bar chart for the 4 base-learner inputs to the Stacking-XGBoost meta-learner |
| `phase5_shap_voltage_region.png` | Fraction of |SHAP| attribution mass falling in the 3.55–3.8V window, per deep model |
| `phase6_conformal_soh.png` | SOH point predictions with 90% conformal interval band vs. true values, evaluation batteries |
| `phase6_conformal_rul.png` | RUL point predictions with 90% conformal interval band vs. true values, evaluation batteries |

---

*No further experiments are planned. GradNorm and softmax-normalized
adaptive-weighting alternatives are noted as future work only, per
explicit instruction not to pursue them now. Full chronological detail,
every dataset/HI-definition assumption, and the pre-fix numbers for
comparison are preserved in `OVERNIGHT_LOG.md`.*
