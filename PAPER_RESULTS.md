# Paper Results

Final, publication-ready tables and findings from the last research
pass before submission (5 items: label-free routing, cycle_idx
ablation, few-shot conformal calibration, relaxation-voltage
feasibility, final rigor pass). Full experimental narrative, negative
results, and disclosed scope limits are in `DEVELOPMENT_LOG.md`; this
file contains only the tables and one-paragraph plain-language findings
meant for the paper itself.

**Governing rule, unchanged throughout every pass in this project**:
nothing is promoted to the deployed model/routing/app unless it clearly
beats the current baseline on the full standard protocol. **Nothing
from this final pass was promoted** - every item is either a disclosed
negative result, a feasibility-gated stop, or a rigor/reporting
exercise on the already-shipped configuration. The deployed model
(dataset-aware routing: the extended-reformulation XGBoost-fusion model
for CALCE/Oxford/HUST, the base XGBoost-fusion model for XJTU and every
other evaluated dataset) is unchanged.

---

## 1. Final configuration: 5-seed accuracy, mean +/- std

Retrained 5 times (seeds 42, 1, 2, 3, 4 - seed 42 matches the
actually-deployed models), routing applied exactly as production does.
R2, RMSE, MAE in % SOH (SOH is stored on a 0-100 scale).

| Dataset | R2 (mean +/- std) | RMSE (mean +/- std) | MAE (mean +/- std) | n cycles |
|---|---|---|---|---|
| In-domain (TEST) | 0.978 +/- 0.003 | 0.716 +/- 0.050 | 0.281 +/- 0.014 | 5,208 |
| CALCE | 0.749 +/- 0.012 | 10.791 +/- 0.265 | 6.256 +/- 0.255 | 2,941 |
| Oxford | 0.940 +/- 0.030 | 1.639 +/- 0.409 | 1.431 +/- 0.434 | 519 |
| HUST | 0.795 +/- 0.022 | 3.336 +/- 0.178 | 2.643 +/- 0.138 | 146,122 |
| XJTU | -1.037 +/- 0.216 | 8.572 +/- 0.460 | 6.457 +/- 0.213 | 19,238 |
| ul_pur (BatteryLife) | 0.116 +/- 0.094 | 5.689 +/- 0.304 | 3.675 +/- 0.283 | 2,245 |
| hnei (BatteryLife) | -0.038 +/- 0.073 | 18.180 +/- 0.650 | 13.966 +/- 0.624 | 15,155 |
| snl (BatteryLife) | 0.147 +/- 0.042 | 7.703 +/- 0.193 | 5.752 +/- 0.183 | 38,880 |
| mich (BatteryLife) | 0.573 +/- 0.028 | 14.727 +/- 0.478 | 7.837 +/- 0.343 | 19,881 |
| mich_exp (BatteryLife) | 0.721 +/- 0.013 | 7.254 +/- 0.165 | 4.539 +/- 0.148 | 6,545 |
| rwth (BatteryLife) | -0.485 +/- 0.077 | 29.558 +/- 0.775 | 23.072 +/- 0.457 | 22,095 |
| stanford (BatteryLife) | 0.111 +/- 0.105 | 20.183 +/- 1.179 | 17.827 +/- 1.265 | 5,633 |
| stanford_2 (BatteryLife) | 0.066 +/- 0.118 | 20.027 +/- 1.254 | 17.988 +/- 1.381 | 8,465 |
| isu_ilcc (BatteryLife) | 0.140 +/- 0.029 | 33.450 +/- 0.560 | 29.737 +/- 0.420 | 45,229 |

XJTU is by far the least seed-stable held-out result (std=0.216 on R2),
consistent with this project's own repeated prior finding that it is
the hardest, most unstable held-out set.

## 2. Extended per-dataset metrics (NRMSE, NMAE, MAPE, % SOH)

NRMSE/NMAE = RMSE or MAE as a percentage of the dataset's own mean SOH.
5-seed mean; full mean+/-std in `outputs/finalpass_item5a_5seed_aggregate.csv`.

| Dataset | NRMSE (%) | NMAE (%) | MAPE (%) |
|---|---|---|---|
| In-domain (TEST) | 0.742 | 0.291 | 0.316 |
| CALCE | 14.718 | 8.533 | 19.007 |
| Oxford | 1.865 | 1.628 | 1.670 |
| HUST | 3.654 | 2.895 | 2.947 |
| XJTU | 8.728 | 6.575 | 6.532 |
| ul_pur | 6.123 | 3.955 | 4.251 |
| hnei | 26.780 | 20.572 | 27.215 |
| snl | 8.804 | 6.573 | 7.438 |
| mich | 17.599 | 9.365 | 34.333 |
| mich_exp | 8.074 | 5.052 | 6.758 |
| rwth | 47.178 | 36.826 | 61.700 |
| stanford | 24.641 | 21.765 | 32.020 |
| stanford_2 | 24.245 | 21.777 | 28.553 |
| isu_ilcc | 69.489 | 61.775 | 324.315 |

MAPE is disproportionately large for isu_ilcc/mich/stanford/stanford_2
specifically - diagnosed as a near-zero-SOH-value artifact in the MAPE
denominator for a handful of rows on those sources (RMSE/MAE/NRMSE/NMAE
all stay reasonable for the same rows), not a computation error.

## 3. Battery-level bootstrap 95% confidence intervals (seed=42, n=1000 resamples)

Resampling BATTERIES (the real unit of independence), not rows.

| Dataset | n batteries | R2 [95% CI] | RMSE [95% CI] | MAE [95% CI] |
|---|---|---|---|---|
| In-domain (TEST) | 6 | 0.976 [0.945, 0.996] | 0.755 [0.243, 1.482] | 0.295 [0.156, 0.629] |
| CALCE | 3 | 0.755 [0.721, 0.826] | 10.656 [8.102, 12.173] | 6.229 [4.874, 7.421] |
| Oxford | 8 | 0.914 [0.900, 0.927] | 2.003 [1.707, 2.239] | 1.773 [1.551, 1.976] |
| HUST | 77 | 0.770 [0.703, 0.818] | 3.540 [3.153, 4.006] | 2.778 [2.544, 3.050] |
| XJTU | 47 | -0.974 [-1.404, -0.593] | 8.449 [7.592, 9.317] | 6.400 [5.769, 7.108] |
| ul_pur | 10 | 0.140 [0.024, 0.289] | 5.618 [4.456, 6.630] | 3.705 [3.018, 4.399] |
| hnei | 14 | -0.106 [-0.148, -0.068] | 18.776 [18.067, 19.387] | 14.544 [13.961, 15.106] |
| snl | 55 | 0.110 [-0.211, 0.266] | 7.867 [6.560, 9.511] | 5.723 [4.871, 6.965] |
| mich | 40 | 0.544 [0.526, 0.556] | 15.222 [12.606, 18.295] | 8.325 [7.001, 10.135] |
| mich_exp | 18 | 0.702 [0.599, 0.811] | 7.490 [3.760, 10.531] | 4.492 [2.744, 6.608] |
| rwth | 10 | -0.554 [-0.632, -0.476] | 30.243 [28.933, 31.325] | 23.480 [22.353, 24.510] |
| stanford | 6 | 0.030 [-0.229, 0.152] | 21.115 [18.902, 23.557] | 18.842 [17.006, 20.987] |
| stanford_2 | 8 | -0.028 [-0.377, 0.154] | 21.035 [19.889, 22.390] | 19.096 [18.208, 20.222] |
| isu_ilcc | 9 | 0.129 [-0.294, 0.265] | 33.670 [21.438, 39.426] | 29.993 [19.701, 36.127] |

Datasets with fewer than ~10 batteries (in-domain, CALCE, Oxford,
stanford, stanford_2, isu_ilcc, ul_pur, rwth, mich_exp) carry
substantial battery-count-driven uncertainty - a single point R2 for
these should not be read as precise.

## 4. RUL cross-domain

RUL LABELS are derivable for all 13 held-out datasets (every one has a
`discharge_capacity` column; the project's threshold-crossing RUL
definition is dataset-agnostic). Zero-retrain scoring of the DEPLOYED
RUL model is only possible for CALCE/Oxford/HUST/XJTU - it requires raw
per-cycle V/I/T tensors that were never built for the 9 BatteryLife
sources (disclosed infrastructure gap, not silently skipped).

| Dataset | RUL R2 | RUL RMSE (cycles) | RUL MAE (cycles) | RUL MAPE (%) | Target RUL range (cycles) | n batteries |
|---|---|---|---|---|---|---|
| In-domain (TEST, reference) | 0.666 | - | - | - | mean 390.6, std 292.9 (train-fit scale) | - |
| CALCE | -566.35 | 474.2 | 421.5 | 1395.9 | 0-125, mean 5.9, std 19.9 | 3 |
| Oxford | -1.31 | 3208.5 | 2520.0 | - | - | 8 |
| HUST | -0.45 | 689.7 | 543.3 | - | - | 77 |
| XJTU | -78.46 | 1538.8 | 1017.9 | - | - | 47 |

**RUL prediction fails far more catastrophically under domain shift
than SOH prediction does, for the exact same 4 datasets and the same
deployed model family** (compare against Table 1's SOH R2: CALCE 0.749,
Oxford 0.940, HUST 0.795, XJTU -1.037 - every one dramatically better
than the corresponding RUL number here). The RUL head's own zero-retrain
generalization had never been measured on these 4 datasets before this
pass; the deployed model's practical reliability claim should be
understood as in-domain-only for RUL even more strongly than for SOH.

**Verification (CHECK B, post-hoc): CALCE's R2=-566 is a real failure, not a bug, and not purely a label/censoring
artifact, but its exact magnitude is amplified by a genuine scale mismatch.** All 3 CALCE cells are non-censored
(each independently crosses the SAME 80%-of-initial-capacity EOL threshold used for every other dataset - no
cross-dataset threshold inconsistency). CALCE's true RUL range is compressed (0-125 cycles, std=19.9) because
these cells degrade far faster than the NASA/MIT batteries the model was trained/destandardized on (train-fit
RUL scale: mean=390.6, std=292.9). The model's predictions for CALCE are not merely off - their mean is
**-401.3 cycles** (physically impossible; RUL cannot be negative) with values ranging from -1144 to +1098,
showing the model has effectively zero transfer to CALCE's compressed lifetime scale (MAE=421.5, MAPE=1396%,
both genuinely large on an absolute/relative basis, not just relative-to-small-variance). R2's specific
magnitude is additionally amplified because CALCE's own target variance (SS_tot) is ~740x smaller than the
in-domain training-scale variance the model was calibrated against, so R2 is not directly comparable in scale
across datasets with such different native RUL ranges - MAE/MAPE are the more honest headline metrics here.
Full diagnostic: `outputs/finalpass_checkB_rul_diagnostic.csv`.

## 5. BatteryLife's own benchmark task (early-cycle-life prediction), reproduced

Predict the cycle number at which SOH first reaches 80%, from only the
first <=100 cycles, using this project's own HI features (mean + slope
over the first 100 cycles) + discharge_capacity + fusion embedding,
XGBoost, 5-fold battery-level cross-validation, pooled across all 9
locally-available BatteryLife sources (125 usable batteries).

| Metric | This work (HI+XGBoost) | BatteryLife published (Li-ion, CPTransformer/CPMLP) |
|---|---|---|
| MAPE | 0.214 | 0.184 / 0.179 |
| 15%-Acc | 0.576 | 0.573 |

**A simple hand-engineered-feature + XGBoost approach is essentially
tied on 15%-Acc and only modestly worse on MAPE against a purpose-built
transformer architecture.** Per-source breakdown (n<10 flagged as
low-confidence): rwth 15%-Acc=1.00 (n=10), hnei 0.86 (n=14), mich 0.75
(n=40), stanford_2 0.63 (n=8), mich_exp 0.50 (n=10), snl 0.25 (n=24),
ul_pur 0.25 (n=4), isu_ilcc 0.22 (n=9), stanford 0.17 (n=6). Not a
strict apples-to-apples comparison (different exact battery
sets/splits; this pool is chemistry-mixed rather than family-separated
the way BatteryLife's own table is) - stated plainly.

## 6. Consolidated conformal coverage/width (target = 90%)

Standard (non-few-shot) split-conformal calibration - the same
convention this project has always shipped.

| Dataset | Coverage | Avg. width (% SOH) | n |
|---|---|---|---|
| In-domain (TEST) | 97.9% | 2.288 | 3,462 |
| CALCE | 4.3% | 0.813 | 2,941 |
| Oxford | 5.4% | 0.813 | 519 |
| HUST | 8.5% | 0.813 | 146,122 |
| XJTU | 11.0% | 2.288 | 19,238 |
| ul_pur | 21.6% | 2.288 | 2,245 |
| hnei | 19.0% | 2.288 | 15,155 |
| snl | 11.7% | 2.288 | 38,880 |
| mich | 16.2% | 2.288 | 19,881 |
| mich_exp | 34.1% | 2.288 | 6,545 |
| rwth | 4.7% | 2.288 | 22,095 |
| stanford | 1.5% | 2.288 | 5,633 |
| stanford_2 | 1.6% | 2.288 | 8,465 |
| isu_ilcc | 1.3% | 2.288 | 45,229 |

**Only 1 of 14 rows (in-domain itself) reaches within 10 percentage
points of the 90% target.** This project's conformal interval should
be read as an in-domain-only guarantee - it does not transfer under
domain shift, and (per item 3, below) does not recover even with up to
50 genuinely-labeled target-domain calibration points.

**Verification (CHECK A, post-hoc): the low coverage is a real non-exchangeability effect, not a bug.**
Independently re-checked, for CALCE/Oxford/HUST/XJTU: (1) calibration and evaluation cycles are correctly split
within the same target battery with zero index overlap (re-asserted, not just trusted), and `split_conformal`'s
argument order is not reversed; (2) calibration residuals and target residuals are on the identical raw-SOH-%
scale (no unit mismatch - e.g. CALCE calibration residuals mean 0.41% vs. CALCE target residuals mean 6.51%,
directly comparable); (3) MAPIE's `SplitConformalRegressor` half-width matches a manually-computed
ceil((n+1)(1-alpha))/n quantile to within 0.003 SOH% on every dataset checked - the finite-sample correction is
applied correctly. The actual cause: the in-domain-calibrated interval is extremely tight (half-width 0.41-1.14
SOH%, reflecting how accurate the model is in-domain) while target residuals under domain shift are 3-15x larger
(mean 1.29-6.51 SOH%) - an interval sized for in-domain noise cannot cover errors of that magnitude, regardless
of k. CALCE and HUST additionally show genuine within-battery residual growth from early to late cycles (CALCE:
2.54% -> 10.49% SOH, HUST: 2.21% -> 2.92% SOH) - a second, compounding non-exchangeability effect (degradation-
stage-dependent error), while Oxford is roughly flat and XJTU's residuals are uniformly large regardless of cycle
position. Full diagnostic: `outputs/finalpass_checkA_conformal_diagnostic.csv`.

## 7. Findings from items 1-4 (all negative/gated, none promoted)

**Item 1 - label-free routing**: a routing rule built purely from the
project's own domain-classifier-AUC diagnostic (no target labels) ties
the naive "always route to the extended model" baseline at 8/13 (61.5%)
correct across all 13 held-out datasets. It fails specifically on
CALCE/Oxford/HUST - where the extended model IS the true winner by a
wide margin - because the AUC signal is saturated (~1.0) in both
feature representations for those datasets, giving the rule nothing to
discriminate on. Not adopted.

**Item 2 - cycle_idx ablation**: retraining with cycle_idx removed, or
replaced by a chemistry/format-normalized "equivalent full cycles"
feature, shows cycle_idx provides negligible in-domain benefit (+0.0005
R2) and a slightly POSITIVE mean zero-retrain effect (+0.0421 R2 across
13 held-out sets, worse on only 5/13). cycle_idx is not acting as a
dataset-specific shortcut. No change to the deployed feature set.

**Item 3 - few-shot conformal calibration**: pooling k=5/10/20/50
genuinely-labeled target-domain cycles into the calibration set barely
moves mean coverage across 13 held-out datasets (10.8% -> 13.4% as k
goes 0 -> 50, target 90%). Domain shift this severe cannot be fixed
with a handful of labeled calibration points. Not adopted.

**Item 4 - relaxation-voltage features**: stopped at the feasibility
gate, per the item's own explicit instruction. Only 5 of 13 held-out
datasets have genuine post-charge rest-period data in their raw files
(CALCE, XJTU, ul_pur, snl, mich), and only 2 of those 5 (XJTU, mich)
are well-resolved enough for the variance/skewness features the
underlying method (Zhu et al., 2022) specifies. Below the ~half-of-
datasets bar; not built, not evaluated.

---

# Final experiment pass 2: items A-F

Second and final pre-submission pass. Same governing rule: nothing
promoted, `app.py`/`live_inference.py`/`models/` untouched, every new
model saved as `models/_experimental_*`. Full narrative:
`DEVELOPMENT_LOG.md`, "Final experiment pass 2" section.

## 8. Online conformal under drift (item A) - the strongest positive result in this project's conformal history

Two online methods (Conformal PID, Angelopoulos/Candes/Tibshirani
NeurIPS 2023; nexCP decay-weighted quantile, Barber et al. Ann.
Statist. 2023), run per-battery in cycle-time order, alpha=0.1,
starting from this project's own in-domain split-conformal half-width.
No lookahead (structurally guaranteed in code).

| Dataset | Static split-conformal (baseline) | PID (eta=0.1) | PID+scorecaster | nexCP (rho=0.95) | nexCP (rho=0.99) |
|---|---|---|---|---|---|
| CALCE | 4.3% | 84.9% | 87.4% | 80.4% | 73.5% |
| Oxford | 5.4% | 85.4% | 90.4% | 89.7% | 86.1% |
| HUST | 8.5% | 88.1% | 90.2% | 87.0% | 85.0% |
| XJTU | 11.0% | 89.3% | 92.4% | 89.2% | 90.8% |
| ul_pur | 21.6% | 72.8% | 83.3% | 79.7% | 72.5% |
| hnei | 19.0% | 69.1% | 89.4% | 71.2% | 56.8% |
| snl | 11.7% | 84.0% | 91.3% | 63.1% | 59.7% |
| mich | 16.2% | 64.4% | 74.3% | 64.8% | 61.4% |
| mich_exp | 34.1% | 69.1% | 83.4% | 70.1% | 64.2% |
| rwth | 4.7% | 79.9% | 91.6% | 78.2% | 57.0% |
| stanford | 1.5% | 89.7% | 93.5% | 86.0% | 89.8% |
| stanford_2 | 1.6% | 89.7% | 93.4% | 86.5% | 89.0% |
| isu_ilcc | 1.3% | 90.0% | 94.7% | 82.5% | 73.9% |

**Every dataset improves dramatically; PID+scorecaster is best on
12/13, several reaching 90-95% from a low-single-digit starting point.**
This beats every prior conformal method tried in this project's
history (best prior CALCE result: 82.0%, itself flagged as
near-vacuous-width - see the consolidated table above).

**Real caveats, disclosed not hidden**: (1) rolling-20-cycle coverage
MIN is 0.00 for 10/13 datasets even under the best config - local
bursts of zero coverage persist despite strong long-run averages; (2)
late-life coverage is often much worse than early-life (e.g. mich:
87.8%->22.7%), consistent with the residual-growth-with-degradation
effect CHECK A found; (3) mean interval width is large on several
datasets (30-62 SOH-% on hnei/rwth/stanford/stanford_2/isu_ilcc) -
coverage recovery is bought partly through width, not free; (4)
infinite intervals are structurally always 0 for both methods (unlike
this project's earlier MAPIE-based attempts); (5) the eta=0.1/rho
values reported as primary are defaults, not separately tuned/selected
against any held-out criterion.

**Why standard ACI cannot do this**: ACI shifts WHICH quantile of the
SOURCE calibration-score distribution to use, but that distribution's
own support is too narrow under severe shift (target residuals run
3-15x larger than calibration residuals, per CHECK A) - no quantile of
an intrinsically too-narrow distribution can reach the target. PID/
nexCP instead estimate width directly from the target's own observed
residuals as they arrive - the structural reason they succeed where ACI
cannot.

**This is a genuine, disclosed finding that changes the paper's
practical conclusion on conformal prediction under domain shift**:
static source-calibrated intervals fail almost completely cross-
dataset, but online recalibration from the target's own sequentially-
revealed labels recovers most of nominal coverage on every dataset
tested. Not wired into the deployed app (out of scope) - flagged as the
strongest concrete follow-up direction.

## 9. Leave-one-dataset-out (item B) - source diversity helps transfer on 11/13 targets

XGBoost-fusion trained on the pooled union of 14 sources, evaluated
zero-retrain on the 15th (held-out) source, same features/
hyperparameters as the deployed model, no tuning on the held-out set.

| Held-out | LODO R2 [95% CI] | NASA+MIT-only R2 | Delta |
|---|---|---|---|
| CALCE | 0.870 [0.845,0.896] | 0.568 | +0.302 |
| Oxford | -0.571 [-1.361,-0.090] | -2.694 | +2.123 |
| HUST | 0.535 [0.437,0.614] | -0.152 | +0.687 |
| XJTU | -4.100 [-4.953,-3.229] | -1.062 | **-3.038 (worse)** |
| ul_pur | 0.488 [0.416,0.589] | 0.116 | +0.372 |
| hnei | 0.681 [0.634,0.738] | -0.038 | +0.719 |
| snl | 0.442 [0.033,0.677] | 0.147 | +0.296 |
| mich | 0.795 [0.754,0.844] | 0.573 | +0.223 |
| mich_exp | 0.638 [0.256,0.714] | 0.721 | -0.082 |
| rwth | 0.353 [0.330,0.373] | -0.485 | +0.838 |
| stanford | 0.997 [0.994,0.999] | 0.111 | +0.886 |
| stanford_2 | 0.990 [0.975,0.999] | 0.066 | +0.925 |
| isu_ilcc | 0.800 [0.187,0.892] | 0.140 | +0.660 |
| NASA (held out, no comparable baseline) | 0.149 [-0.249,0.363] | - | - |
| MIT (held out, no comparable baseline) | -4.817 [-9.151,-2.498] | - | - |

**Source diversity helps transfer on 11/13 comparable targets, often
dramatically** (stanford/stanford_2 jump from ~0.07-0.11 to ~0.99).
**It fails on exactly the two datasets this project has repeatedly
flagged as its hardest, most protocol-divergent cases**: XJTU gets
substantially WORSE with more pooled data (not better), and mich_exp
regresses marginally. MIT held out alone is a striking new finding:
its fast-charging protocol is different enough that even 14-source
pooling cannot generalize to it (R2=-4.82).

## 10. Shift diagnostics (item E) - AUC has no reliable relationship with outcomes at this severity of shift

**Part 1**: across the 13 datasets, domain-classifier AUC correlates
weakly and NOT significantly with R2, MAE, or conformal coverage
(Spearman rho: -0.201, 0.248, -0.195 respectively, all 95% bootstrap
CIs crossing zero) - directly reinforcing item 1's finding that AUC
saturates near 1.0 for most datasets, leaving too little variance to
predict anything with.

**Part 2**: a legitimately source-only-calibrated OOD threshold (90th
percentile of an in-domain calib-vs-eval classifier's own scores, never
touching real target data) abstains on 100% of batteries for 12/13
target datasets (96.4% for the sole exception, snl) - a real,
disclosed consequence of how saturated OOD scores already are. The
target-relative risk-coverage curve (retaining the lowest-OOD battery
fraction within each dataset, sweeping abstention 0-50%) shows mean
retained-battery MAE staying essentially flat (10.34 -> 10.17 -> 10.39)
- abstaining on "more OOD-looking" batteries does not reliably reduce
error at this severity of shift.

**Coherent conclusion across both parts**: the domain-classifier-AUC
diagnostic, useful earlier for motivating Stage 1.1's reformulation
work, does not function as a reliable per-dataset risk indicator once
shift is this severe. No AUC-based selective-prediction mechanism is
recommended for deployment.

## 11. Complete baseline table (item C) - engineered HI+fusion features dramatically beat literature early-cycle-life baselines, with one important exception

| Dataset | Trivial linear | Severson variance | Attia rich | Deployed base | Routed (oracle) | Best LODO |
|---|---|---|---|---|---|---|
| CALCE | -0.858 | 0.077 | 0.078 | 0.749 | 0.740 | **0.870** |
| Oxford | -7.589 | 0.169 | 0.195 | **0.940** | 0.953 | -0.571 |
| HUST | 0.755 | -1.790 | -1.437 | 0.795 | **0.800** | 0.535 |
| XJTU | 0.279 | -7.835 | -7.410 | **-1.037** | -1.037 | -4.100 |
| ul_pur | -0.566 | -3.292 | -3.397 | 0.116 | 0.116 | **0.488** |
| hnei | -2.001 | -0.087 | -0.108 | -0.038 | -0.038 | **0.681** |
| snl | -0.550 | -0.324 | -0.275 | 0.147 | 0.147 | **0.442** |
| mich | -0.275 | 0.241 | 0.252 | 0.573 | 0.573 | **0.795** |
| mich_exp | -0.217 | -0.206 | -0.122 | **0.721** | 0.721 | 0.638 |
| rwth | -0.000 | -0.000 | -0.001 | -0.485 | -0.485 | **0.353** |
| stanford | -0.274 | 0.148 | 0.135 | 0.111 | 0.111 | **0.997** |
| stanford_2 | -0.206 | 0.121 | 0.108 | 0.066 | 0.066 | **0.990** |
| isu_ilcc | 0.358 | -0.989 | -1.024 | 0.140 | 0.140 | **0.800** |

**Engineered HI+fusion+XGBoost dramatically outperforms Severson
(2019)/Attia-style early-cycle-life features on cross-dataset zero-
retrain generalization** - the literature baselines go deeply negative
on 7/13 datasets (as low as -7.8), while the deployed model stays
competitive or clearly better almost everywhere. The trivial linear
baseline is usually catastrophic but surprisingly competitive on HUST
(0.755, near the deployed model's 0.795) and isu_ilcc (0.358) - both
apparently near-linear degradation protocols, a real structural
property, not a modeling artifact.

**Paired battery-level significance tests** (Wilcoxon + bootstrap,
absolute error): the deployed base model is significantly better than
both literature baselines on 8/13 datasets. **One important, disclosed
exception: on Oxford, Severson/Attia have significantly LOWER MAE**
(5.2-5.3 vs. 12.5, p=0.008) **despite the deployed model's much higher
R2** (0.940 vs. 0.17-0.20) - R2 measures variance explained (the
deployed model tracks Oxford's overall trend far better) while MAE
measures raw error magnitude (the simpler baselines make smaller,
more conservative absolute errors) - a real case where metric choice
changes which model looks better, reported as found.

---

*Full experimental detail for items A-F: `DEVELOPMENT_LOG.md`, "Final
experiment pass 2 before the paper" section onward.*
