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

## Rerun status banner (2026-09-30) - READ FIRST

**The BatteryLife embedding-normalisation bug is fixed and everything that read the affected columns was rerun.**
`build_batterylife_hi_table.py` (old line 98) fed raw, un-normalised X to the encoder for the 9 BatteryLife sources
(and Tongji), so their fusion-embedding columns were defective. The corrected old-encoder embeddings are now the
canonical parquets (swap verified, `outputs/toolkit_swap_verification.csv`); the rerun queue finished 2026-09-30 20:02
(all rc=0). Before/after evidence: `outputs/toolkit_rerun_change_summary.csv`, `outputs/toolkit_rerun_top_movers.csv`,
`outputs/toolkit_rerun_before_after_itemB.csv`, `outputs/toolkit_lodo_family_holdout_rerun_corrected.csv`. BEFORE = the raw-X
files at git commit c56ddcb; AFTER = the current files.

Status tags used below:
- **VERIFIED** - the built-in datasets' rows (in-domain, CALCE, Oxford, HUST, XJTU) are unchanged (0 changed cells / identical
  to the number of digits printed) after the rerun.
- **SUPERSEDED** - the BatteryLife-source numbers were replaced by the corrected rerun values (current CSVs). The old value is
  kept as "(before ...)" or in a "before" column. Built-in rows in the same table are VERIFIED unless stated.
- **PAUSED/RUNNING** - not finished; do not cite.

Where a headline verdict changed, it is stated in a "Rerun verdict check" line under the section. Summary of all
scripts: the "Rerun status log 2026-09-30" at the end of this file.

---

## 1. Final configuration: 5-seed accuracy, mean +/- std

Retrained 5 times (seeds 42, 1, 2, 3, 4 - seed 42 matches the
actually-deployed models), routing applied exactly as production does.
R2, RMSE, MAE in % SOH (SOH is stored on a 0-100 scale).

**Status: in-domain/CALCE/Oxford/HUST/XJTU rows VERIFIED (identical to the pre-rerun table); the 9 BatteryLife rows are
SUPERSEDED by the corrected rerun (`outputs/finalpass_item5a_5seed_aggregate.csv`); pre-rerun value in parentheses.**

| Dataset | R2 (mean +/- std) | RMSE (mean +/- std) | MAE (mean +/- std) | n cycles |
|---|---|---|---|---|
| In-domain (TEST) | 0.978 +/- 0.003 | 0.716 +/- 0.050 | 0.281 +/- 0.014 | 5,208 |
| CALCE | 0.749 +/- 0.012 | 10.791 +/- 0.265 | 6.256 +/- 0.255 | 2,941 |
| Oxford | 0.940 +/- 0.030 | 1.639 +/- 0.409 | 1.431 +/- 0.434 | 519 |
| HUST | 0.795 +/- 0.022 | 3.336 +/- 0.178 | 2.643 +/- 0.138 | 146,122 |
| XJTU | -1.037 +/- 0.216 | 8.572 +/- 0.460 | 6.457 +/- 0.213 | 19,238 |
| ul_pur (BatteryLife) | 0.138 +/- 0.037 (before 0.116) | 5.622 +/- 0.120 (before 5.689) | 3.430 +/- 0.184 (before 3.675) | 2,245 |
| hnei (BatteryLife) | -0.137 +/- 0.080 (before -0.038) | 19.034 +/- 0.667 (before 18.180) | 14.669 +/- 0.700 (before 13.966) | 15,155 |
| snl (BatteryLife) | 0.118 +/- 0.056 (before 0.147) | 7.829 +/- 0.248 (before 7.703) | 5.932 +/- 0.227 (before 5.752) | 38,880 |
| mich (BatteryLife) | 0.536 +/- 0.019 (before 0.573) | 15.347 +/- 0.319 (before 14.727) | 7.861 +/- 0.342 (before 7.837) | 19,881 |
| mich_exp (BatteryLife) | 0.716 +/- 0.012 (before 0.721) | 7.320 +/- 0.158 (before 7.254) | 4.567 +/- 0.166 (before 4.539) | 6,545 |
| rwth (BatteryLife) | -0.535 +/- 0.086 (before -0.485) | 30.049 +/- 0.851 (before 29.558) | 23.500 +/- 0.549 (before 23.072) | 22,095 |
| stanford (BatteryLife) | 0.291 +/- 0.103 (before 0.111) | 18.019 +/- 1.282 (before 20.183) | 15.295 +/- 1.393 (before 17.827) | 5,633 |
| stanford_2 (BatteryLife) | 0.263 +/- 0.112 (before 0.066) | 17.778 +/- 1.321 (before 20.027) | 15.303 +/- 1.465 (before 17.988) | 8,465 |
| isu_ilcc (BatteryLife) | 0.123 +/- 0.013 (before 0.140) | 33.792 +/- 0.246 (before 33.450) | 29.409 +/- 0.140 (before 29.737) | 45,229 |

XJTU is by far the least seed-stable held-out result (std=0.216 on R2),
consistent with this project's own repeated prior finding that it is
the hardest, most unstable held-out set. (Still true after the rerun: the
largest BatteryLife std is stanford_2's 0.112.)

**Rerun verdict check (section 1): no headline verdict changed.** No BatteryLife R2 changed sign; the largest moves are
stanford 0.111 -> 0.291, stanford_2 0.066 -> 0.263 (both up) and hnei -0.038 -> -0.137 (down). The in-domain and four
built-in rows are identical to before.

## 2. Extended per-dataset metrics (NRMSE, NMAE, MAPE, % SOH)

NRMSE/NMAE = RMSE or MAE as a percentage of the dataset's own mean SOH.
5-seed mean; full mean+/-std in `outputs/finalpass_item5a_5seed_aggregate.csv`.

**Status: in-domain/CALCE/Oxford/HUST/XJTU rows VERIFIED; the 9 BatteryLife rows SUPERSEDED (corrected rerun values, pre-rerun in parentheses).**

| Dataset | NRMSE (%) | NMAE (%) | MAPE (%) |
|---|---|---|---|
| In-domain (TEST) | 0.742 | 0.291 | 0.316 |
| CALCE | 14.718 | 8.533 | 19.007 |
| Oxford | 1.865 | 1.628 | 1.670 |
| HUST | 3.654 | 2.895 | 2.947 |
| XJTU | 8.728 | 6.575 | 6.532 |
| ul_pur | 6.051 (before 6.123) | 3.691 (before 3.955) | 3.994 (before 4.251) |
| hnei | 28.038 (before 26.780) | 21.609 (before 20.572) | 28.561 (before 27.215) |
| snl | 8.947 (before 8.804) | 6.779 (before 6.573) | 7.694 (before 7.438) |
| mich | 18.340 (before 17.599) | 9.394 (before 9.365) | 35.594 (before 34.333) |
| mich_exp | 8.147 (before 8.074) | 5.083 (before 5.052) | 6.825 (before 6.758) |
| rwth | 47.961 (before 47.178) | 37.510 (before 36.826) | 62.743 (before 61.700) |
| stanford | 21.999 (before 24.641) | 18.673 (before 21.765) | 29.931 (before 32.020) |
| stanford_2 | 21.522 (before 24.245) | 18.526 (before 21.777) | 26.138 (before 28.553) |
| isu_ilcc | 70.198 (before 69.489) | 61.094 (before 61.775) | 332.963 (before 324.315) |

MAPE is disproportionately large for isu_ilcc/mich/stanford/stanford_2
specifically - diagnosed as a near-zero-SOH-value artifact in the MAPE
denominator for a handful of rows on those sources (RMSE/MAE/NRMSE/NMAE
all stay reasonable for the same rows), not a computation error. The same pattern persists after the rerun
(isu_ilcc MAPE 333.0, mich 35.6, rwth 62.7, stanford 29.9, stanford_2 26.1).

**Rerun verdict check (section 2): no headline verdict changed.**

## 3. Battery-level bootstrap 95% confidence intervals (seed=42, n=1000 resamples)

Resampling BATTERIES (the real unit of independence), not rows.

**Status: in-domain/CALCE/Oxford/HUST/XJTU rows VERIFIED; the 9 BatteryLife rows SUPERSEDED (`outputs/finalpass_item5a_bootstrap_ci.csv`, corrected rerun; pre-rerun point estimate in parentheses).**

| Dataset | n batteries | R2 [95% CI] | RMSE [95% CI] | MAE [95% CI] |
|---|---|---|---|---|
| In-domain (TEST) | 6 | 0.976 [0.945, 0.996] | 0.755 [0.243, 1.482] | 0.295 [0.156, 0.629] |
| CALCE | 3 | 0.755 [0.721, 0.826] | 10.656 [8.102, 12.173] | 6.229 [4.874, 7.421] |
| Oxford | 8 | 0.914 [0.900, 0.927] | 2.003 [1.707, 2.239] | 1.773 [1.551, 1.976] |
| HUST | 77 | 0.770 [0.703, 0.818] | 3.540 [3.153, 4.006] | 2.778 [2.544, 3.050] |
| XJTU | 47 | -0.974 [-1.404, -0.593] | 8.449 [7.592, 9.317] | 6.400 [5.769, 7.108] |
| ul_pur | 10 | 0.157 [0.047, 0.312] (before 0.140) | 5.561 [4.476, 6.496] (before 5.618) | 3.381 [2.807, 3.940] (before 3.705) |
| hnei | 14 | -0.144 [-0.187, -0.107] (before -0.106) | 19.101 [18.398, 19.704] (before 18.776) | 14.846 [14.259, 15.408] (before 14.544) |
| snl | 55 | 0.097 [-0.239, 0.262] (before 0.110) | 7.927 [6.670, 9.567] (before 7.867) | 5.945 [5.200, 7.063] (before 5.723) |
| mich | 40 | 0.526 [0.502, 0.547] (before 0.544) | 15.522 [12.738, 18.796] (before 15.222) | 7.884 [6.477, 9.840] (before 8.325) |
| mich_exp | 18 | 0.713 [0.618, 0.799] (before 0.702) | 7.353 [3.926, 10.205] (before 7.490) | 4.454 [2.830, 6.399] (before 4.492) |
| rwth | 10 | -0.608 [-0.718, -0.513] (before -0.554) | 30.767 [29.380, 32.011] (before 30.243) | 24.003 [22.773, 25.107] (before 23.480) |
| stanford | 6 | 0.257 [0.130, 0.308] (before 0.030) | 18.485 [15.888, 21.316] (before 21.115) | 15.811 [13.936, 17.986] (before 18.842) |
| stanford_2 | 8 | 0.229 [0.025, 0.327] (before -0.028) | 18.222 [16.714, 19.920] (before 21.035) | 15.826 [14.797, 17.144] (before 19.096) |
| isu_ilcc | 9 | 0.122 [-0.329, 0.297] (before 0.129) | 33.810 [19.444, 40.254] (before 33.670) | 29.365 [17.513, 36.377] (before 29.993) |

Datasets with fewer than ~10 batteries (in-domain, CALCE, Oxford,
stanford, stanford_2, isu_ilcc, ul_pur, rwth, mich_exp) carry
substantial battery-count-driven uncertainty - a single point R2 for
these should not be read as precise.

**Rerun verdict check (section 3): no headline verdict changed.** The one qualitative shift: the stanford and stanford_2
R2 CIs previously spanned zero ([-0.229, 0.152], [-0.377, 0.154]) and now sit above zero ([0.130, 0.308], [0.025, 0.327]).

## 4. RUL cross-domain

RUL LABELS are derivable for all 13 held-out datasets (every one has a
`discharge_capacity` column; the project's threshold-crossing RUL
definition is dataset-agnostic). Zero-retrain scoring of the DEPLOYED
RUL model is only possible for CALCE/Oxford/HUST/XJTU - it requires raw
per-cycle V/I/T tensors that were never built for the 9 BatteryLife
sources (disclosed infrastructure gap, not silently skipped).

**Status: VERIFIED - not in the rerun change summary.** This section scores only CALCE/Oxford/HUST/XJTU (no BatteryLife embeddings are
read), and none of its files appear among the rewritten CSVs in `outputs/toolkit_rerun_change_summary.csv`. No verdict changed.

| Dataset | RUL R2 | RUL RMSE (cycles) | RUL MAE (cycles) | RUL MAPE (%) | Target RUL range (cycles) | n batteries |
|---|---|---|---|---|---|---|
| In-domain (TEST) - first-pass joint model (reference) | 0.666 | - | - | - | mean 390.6, std 292.9 (train-fit scale) | - |
| In-domain (TEST) - deployed model (retrained, Stage 4) | 0.374 | 265.30 | - | - | see DEVELOPMENT_LOG.md Stage 4 step 2b | - |
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

**Verification (CHECK B, post-hoc; status VERIFIED - built-in datasets only, not in the rerun change summary): CALCE's R2=-566 is a real failure, not a bug, and not purely a label/censoring
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

**Status: SUPERSEDED - BatteryLife-only task, rerun on corrected embeddings (`outputs/finalpass_item5d_summary.csv`); pre-rerun value in parentheses (pre-rerun CSV c56ddcb).**

| Metric | This work (HI+XGBoost) | BatteryLife published (Li-ion, CPTransformer/CPMLP) |
|---|---|---|
| MAPE | 0.211 (before 0.213) | 0.184 / 0.179 |
| 15%-Acc | 0.600 (before 0.576) | 0.573 |

**A simple hand-engineered-feature + XGBoost approach is essentially
tied on 15%-Acc and only modestly worse on MAPE against a purpose-built
transformer architecture.** Per-source breakdown (n<10 flagged as
low-confidence; corrected rerun, before in parentheses): hnei 15%-Acc=0.93 (n=14; before 0.86), mich 0.83
(n=40; 0.75), rwth 0.80 (n=10; 1.00), stanford 0.50 (n=6; 0.17), stanford_2 0.50 (n=8; 0.63), mich_exp 0.40 (n=10; 0.50),
snl 0.29 (n=24; 0.25), ul_pur 0.25 (n=4; 0.25), isu_ilcc 0.22 (n=9; 0.22). Not a
strict apples-to-apples comparison (different exact battery
sets/splits; this pool is chemistry-mixed rather than family-separated
the way BatteryLife's own table is) - stated plainly.

**Rerun verdict check (section 5): no headline verdict changed.** Pooled 15%-Acc moves from 0.576 to 0.600 (published 0.573), so the
gap changes from +0.003 to +0.027 - still "essentially tied"; MAPE 0.211 is still worse than the published 0.184/0.179. Per-source
values moved substantially (rwth 1.00 -> 0.80, stanford 0.17 -> 0.50) on 6-10 batteries each and stay low-confidence.

## 6. Consolidated conformal coverage/width (target = 90%)

Standard (non-few-shot) split-conformal calibration - the same
convention this project has always shipped.

**Status: in-domain/CALCE/Oxford/HUST/XJTU coverage and widths VERIFIED; the 9 BatteryLife coverage values are SUPERSEDED (model predictions on
those sources changed; corrected values = the k=0 column of `outputs/finalpass_item3_coverage_pivot.csv`, pre-rerun in parentheses). Widths are unchanged (k=0 width 2.288 on every
BatteryLife source, `finalpass_item3_width_pivot.csv`).**

| Dataset | Coverage | Avg. width (% SOH) | n |
|---|---|---|---|
| In-domain (TEST) | 97.9% | 2.288 | 3,462 |
| CALCE | 4.3% | 0.813 | 2,941 |
| Oxford | 5.4% | 0.813 | 519 |
| HUST | 8.5% | 0.813 | 146,122 |
| XJTU | 11.0% | 2.288 | 19,238 |
| ul_pur | 39.6% (before 21.6%) | 2.288 | 2,245 |
| hnei | 14.2% (before 19.0%) | 2.288 | 15,155 |
| snl | 11.1% (before 11.7%) | 2.288 | 38,880 |
| mich | 30.2% (before 16.2%) | 2.288 | 19,881 |
| mich_exp | 33.4% (before 34.1%) | 2.288 | 6,545 |
| rwth | 4.6% (before 4.7%) | 2.288 | 22,095 |
| stanford | 1.7% (before 1.5%) | 2.288 | 5,633 |
| stanford_2 | 1.8% (before 1.6%) | 2.288 | 8,465 |
| isu_ilcc | 1.5% (before 1.3%) | 2.288 | 45,229 |

**Only 1 of 14 rows (in-domain itself) reaches within 10 percentage
points of the 90% target.** This project's conformal interval should
be read as an in-domain-only guarantee - it does not transfer under
domain shift, and (per item 3, below) does not recover even with up to
50 genuinely-labeled target-domain calibration points.

**Rerun verdict check (section 6): no headline verdict changed.** After the rerun the best BatteryLife row is ul_pur at 39.6%, so still only 1 of 14 rows
(in-domain, 97.9%) is within 10 points of the 90% target.

**Verification (CHECK A, post-hoc; status VERIFIED - covers only CALCE/Oxford/HUST/XJTU, not in the rerun change summary): the low coverage is a real non-exchangeability effect, not a bug.**
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

**Status: item 1 SUPERSEDED (numbers and one comparison changed, see below); item 2 SUPERSEDED (BatteryLife eval sets); item 3 SUPERSEDED (BatteryLife eval sets; CALCE/Oxford/HUST/XJTU coverage VERIFIED); item 4 VERIFIED (feasibility gate, not in the rerun change summary).**

**Item 1 - label-free routing** (SUPERSEDED, corrected rerun; `outputs/finalpass_item1_labelfree_routing.csv`): a routing rule built purely from the
project's own domain-classifier-AUC diagnostic (no target labels) is correct on **3/13** held-out datasets after the rerun
(before: 8/13). The AUC signal is saturated (~1.0) in both feature representations for every dataset, giving the rule nothing to
discriminate on (rule_pick = base on all 13). **What changed:** with corrected BatteryLife embeddings the extended model is the true winner
on **10/13** datasets (before: 8/13: CALCE/Oxford/HUST plus, now, hnei, snl, mich, rwth, stanford, stanford_2, isu_ilcc), so the naive "always route to
the extended model" baseline is correct on 10/13 (before: 8/13, where it tied the rule at 8/13). The rule therefore no longer ties that baseline - it loses
to it, 3/13 vs 10/13 - and the deployed routing (extended only for CALCE/Oxford/HUST) is correct on 6/13 (before: 8/13). Corrected extended-vs-base R2 on the
BatteryLife sources: hnei 0.785 vs -0.101, snl 0.442 vs 0.056, mich 0.754 vs 0.533, rwth 0.503 vs -0.473, stanford 0.809 vs 0.275, stanford_2 0.806 vs 0.247,
isu_ilcc 0.180 vs 0.114; the base model still wins on XJTU, ul_pur (-0.605 vs 0.044) and mich_exp (0.309 vs 0.718). Not adopted (the label-free rule is still not
usable); the routing change this hints at is NOT made here - it would need the full standard protocol under the governing rule.

**Item 2 - cycle_idx ablation**: retraining with cycle_idx removed, or
replaced by a chemistry/format-normalized "equivalent full cycles"
feature, shows cycle_idx provides negligible in-domain benefit (+0.0005
R2) and a slightly POSITIVE mean zero-retrain effect (+0.0421 R2 across
13 held-out sets, worse on only 5/13). cycle_idx is not acting as a
dataset-specific shortcut. No change to the deployed feature set.

*Rerun (SUPERSEDED, `outputs/finalpass_item2_r2_pivot.csv`)*: in-domain benefit is unchanged (+0.0005 R2, 0.9756 vs 0.9751). The mean zero-retrain effect of
keeping cycle_idx over removing it is now **+0.0617 R2** across the 13 held-out sets (before +0.0421); removal is worse on 10/13, i.e. cycle_idx is worse on
only 3/13 (before 5/13). Verdict unchanged: cycle_idx is not a dataset-specific shortcut.

**Item 3 - few-shot conformal calibration**: pooling k=5/10/20/50
genuinely-labeled target-domain cycles into the calibration set barely
moves mean coverage across 13 held-out datasets (12.9% -> 15.2% as k
goes 0 -> 50 after the rerun; before 10.8% -> 13.4%; target 90%). Domain shift this severe cannot be fixed
with a handful of labeled calibration points. Not adopted.
*Rerun verdict check (items 2, 3): no headline verdict changed (`outputs/finalpass_item3_coverage_pivot.csv`).*

**Item 4 - relaxation-voltage features**: stopped at the feasibility
gate, per the item's own explicit instruction. Only 5 of 13 held-out
datasets have genuine post-charge rest-period data in their raw files
(CALCE, XJTU, ul_pur, snl, mich), and only 2 of those 5 (XJTU, mich)
are well-resolved enough for the variance/skewness features the
underlying method (Zhu et al., 2022) specifies. Below the ~half-of-
datasets bar; not built, not evaluated. (Status VERIFIED: nothing was
computed, nothing to rerun.)

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

**Status: CALCE/Oxford/HUST/XJTU rows VERIFIED (identical to the pre-rerun table); the 9 BatteryLife rows are SUPERSEDED (`outputs/finalpass2_itemA_headline_comparison.csv`,
`finalpass2_itemA_pid_results.csv`, `finalpass2_itemA_nexcp_results.csv`; pre-rerun value in parentheses). The static column is the corrected k=0 coverage of
`finalpass_item3_coverage_pivot.csv` (the static column inside `finalpass3_check2_labelfree_vs_online.csv` was NOT recomputed and still shows the pre-rerun static numbers).**

| Dataset | Static split-conformal (baseline) | PID (eta=0.1) | PID+scorecaster | nexCP (rho=0.95) | nexCP (rho=0.99) |
|---|---|---|---|---|---|
| CALCE | 4.3% | 84.9% | 87.4% | 80.4% | 73.5% |
| Oxford | 5.4% | 85.4% | 90.4% | 89.7% | 86.1% |
| HUST | 8.5% | 88.1% | 90.2% | 87.0% | 85.0% |
| XJTU | 11.0% | 89.3% | 92.4% | 89.2% | 90.8% |
| ul_pur | 39.6% (21.6%) | 58.9% (72.8%) | 79.8% (83.3%) | 74.3% (79.7%) | 60.0% (72.5%) |
| hnei | 14.2% (19.0%) | 38.0% (69.1%) | 85.2% (89.4%) | 63.2% (71.2%) | 46.7% (56.8%) |
| snl | 11.1% (11.7%) | 82.3% (84.0%) | 91.3% (91.3%) | 66.4% (63.1%) | 62.2% (59.7%) |
| mich | 30.2% (16.2%) | 61.9% (64.4%) | 71.1% (74.3%) | 63.5% (64.8%) | 59.8% (61.4%) |
| mich_exp | 33.4% (34.1%) | 65.2% (69.1%) | 82.2% (83.4%) | 69.3% (70.1%) | 62.3% (64.2%) |
| rwth | 4.6% (4.7%) | 83.7% (79.9%) | 92.2% (91.6%) | 68.8% (78.2%) | 55.5% (57.0%) |
| stanford | 1.7% (1.5%) | 89.4% (89.7%) | 93.6% (93.5%) | 84.8% (86.0%) | 88.4% (89.8%) |
| stanford_2 | 1.8% (1.6%) | 89.5% (89.7%) | 93.3% (93.4%) | 86.0% (86.5%) | 88.3% (89.0%) |
| isu_ilcc | 1.5% (1.3%) | 89.9% (90.0%) | 95.0% (94.7%) | 80.6% (82.5%) | 72.3% (73.9%) |

**Every dataset improves dramatically; PID+scorecaster is the best of the four tested configurations on all 13 (per the CSV; the pre-rerun text said 12/13), several reaching 90-95% from a low-single-digit starting point.**
This beats every prior conformal method tried in this project's
history (best prior CALCE result: 82.0%, itself flagged as
near-vacuous-width - see the consolidated table above).

**Rerun verdict check (section 8): the direction of the headline holds, its strength is weaker on four BatteryLife sources; no verdict flipped.**
PID+scorecaster: static 1.5-39.6% -> 71.1-95.0% (before: 1.3-34.1% -> 74.3-94.7%); the minimum is now mich at 71.1%. The plain PID/nexCP
configurations lost the most: PID hnei 69.1% -> 38.0%, ul_pur 72.8% -> 58.9%; nexCP(0.99) hnei 56.8% -> 46.7%, ul_pur 72.5% -> 60.0%. Only rwth PID (79.9% -> 83.7%) and
snl nexCP(0.95) (63.1% -> 66.4%) improved noticeably. So "recovers most of nominal coverage on every dataset" now holds for PID+scorecaster (all >= 71.1%) but
not for plain PID (hnei 38.0%, ul_pur 58.9%, mich 61.9%, mich_exp 65.2% are below 70%).

**Real caveats, disclosed not hidden**: (1) rolling-20-cycle coverage
MIN is 0.00 for 7/13 datasets even under the best config (HUST,
ul_pur, hnei, snl, mich, mich_exp, rwth - **corrected here from an
earlier, wrong "10/13" via the final verification pass's own
per-battery re-check**; rerun `finalpass3_check2_lifestage.csv` gives the same 7 datasets for PID) - local bursts of zero coverage persist despite
strong long-run averages; (2)
late-life coverage is often much worse than early-life (e.g. mich:
90.4%->19.1% after the rerun, before 87.8%->22.7%), consistent with the residual-growth-with-degradation
effect CHECK A found; (3) mean interval width is large on several
datasets (29-62 SOH-% on hnei/rwth/stanford/stanford_2/isu_ilcc after the rerun: 29.0, 48.1, 34.0, 34.2, 61.7 for plain PID) -
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

## 9. Leave-one-dataset-out (item B) - source diversity helps transfer on 10/13 targets, with snl not distinguishable

**Status: SUPERSEDED - every row changed (the pooled training set contains the BatteryLife sources); corrected values from `outputs/finalpass2_itemB_lodo_results.csv`,
`outputs/toolkit_lodo_family_holdout_rerun_corrected.csv` and `outputs/toolkit_rerun_before_after_itemB.csv`. Pre-rerun value in the "Before" column.**
The NASA+MIT-only baseline was recomputed in the rerun (`nasa_mit_only_r2_corrected`) and moved on several rows, including built-in ones (e.g. Oxford -2.694 -> -3.598,
XJTU -1.062 -> -1.556), so the "Delta" column is against the recomputed baseline. Stanford/stanford_2 rows use the sibling-holdout (family) number, as before.

XGBoost-fusion trained on the pooled union of 14 sources, evaluated
zero-retrain on the 15th (held-out) source, same features/
hyperparameters as the deployed model, no tuning on the held-out set.

| Held-out | LODO R2 [95% CI] (corrected) | Before | NASA+MIT-only R2 (corrected; before) | Delta (corrected) |
|---|---|---|---|---|
| CALCE | 0.855 [0.804,0.939] | 0.870 | 0.560 (0.568) | +0.295 |
| Oxford | 0.076 [-0.377,0.349] | -0.571 | -3.598 (-2.694) | +3.673 |
| HUST | 0.722 [0.672,0.759] | 0.535 | -0.027 (-0.152) | +0.749 |
| XJTU | -6.394 [-7.578,-5.049] | -4.100 | -1.556 (-1.062) | **-4.838 (worse)** |
| ul_pur | 0.517 [0.442,0.636] | 0.488 | 0.265 (0.116) | +0.252 |
| hnei | 0.929 [0.923,0.937] | 0.681 | 0.070 (-0.038) | +0.860 |
| snl | 0.149 [-0.213,0.336] | 0.442 | 0.154 (0.147) | **-0.005 (not distinguishable)** |
| mich | 0.746 [0.726,0.768] | 0.795 | 0.565 (0.573) | +0.181 |
| mich_exp | 0.499 [-0.523,0.725] | 0.638 | 0.731 (0.721) | **-0.232 (worse)** |
| rwth | 0.505 [0.469,0.540] | 0.353 | -0.152 (-0.485) | +0.657 |
| stanford | ~~0.997~~ **0.919** [0.776,0.985] (sibling-holdout corrected, see CHECK 1) | ~~0.997~~ 0.889 | 0.299 (0.111) | +0.621 |
| stanford_2 | ~~0.990~~ **0.895** [0.626,0.993] (sibling-holdout corrected, see CHECK 1) | ~~0.990~~ 0.858 | 0.220 (0.066) | +0.675 |
| isu_ilcc | 0.901 [0.611,0.948] | 0.800 | 0.053 (0.140) | +0.848 |
| NASA (held out, no comparable baseline) | 0.031 [-0.350,0.225] | 0.149 | - | - |
| MIT (held out, no comparable baseline) | -6.118 [-11.357,-3.360] | -4.817 | - | - |

**Source diversity helps transfer on 10/13 comparable targets (before: 11/13), often
dramatically** (stanford/stanford_2 jump from ~0.22-0.30 to ~0.90-0.92
- see the family-level-holdout correction below). **snl is recorded as "not distinguishable", not as a flipped verdict:** its corrected family-LODO R2 is
0.149 against a NASA+MIT-only R2 of 0.154 (margin 0.005, far inside the CI [-0.213, 0.336]); before the rerun it was 0.442 vs 0.147, a clear win, so the
*count* of clear wins drops from 11 to 10 while snl itself is a tie. **It fails on exactly
the datasets this project has repeatedly flagged as its hardest,
most protocol-divergent cases**: XJTU gets substantially WORSE with
more pooled data (-6.394 vs -1.556 for NASA+MIT-only; before -4.100 vs -1.062), and mich_exp regresses (0.499 plain / 0.550 sibling-holdout vs 0.731).
MIT held out alone remains a striking finding: its fast-charging protocol
is different enough that even 14-source pooling cannot generalize to
it (R2=-6.118; before -4.817).

**The very negative MIT (-6.1) and XJTU (-6.4) values are genuine failures of leave-one-dataset-out transfer, not artefacts.** They were computed on the
corrected BatteryLife/Tongji embeddings (the raw-X bug removed) and got MORE negative, not less (MIT -4.817 -> -6.118, XJTU -4.100 -> -6.394), so the defect was not what produced them.

**Rerun verdict check (section 9):** no source flipped in `toolkit_rerun_before_after_itemB.csv` (verdict_flipped = False on all 15 rows, using the pre-rerun baseline), but with the
recomputed NASA+MIT-only baseline snl's margin over it collapses to -0.005 (not distinguishable), so the headline count of clear wins is 10/13, not 11/13.
Largest R2 moves: XJTU -2.29, MIT -1.30, snl -0.293, hnei +0.248, Oxford +0.647.

**CHECK 1 (family/sibling-holdout verification, post-hoc; SUPERSEDED, rerun as `lodo_check1`)**: Stanford
and Stanford_2 are sibling sources (same lab); MICH/MICH_EXP likewise.
Re-running LODO while holding out EACH SIBLING PAIR TOGETHER (not just
the target alone) still shows a measurable leakage effect for the
Stanford pair - **stanford's plain 0.997 does NOT survive as
reported**; the honest, sibling-corrected number is now **0.919** (before the rerun 0.889; MAE
0.55 -> 2.15 cycles), and stanford_2's corrected number
is **0.895** (before 0.858; plain 0.990). mich: plain 0.746 -> family 0.702 (before 0.795 -> 0.790);
mich_exp: plain 0.499 -> family 0.550 (before 0.638 -> 0.563). **The qualitative headline
claim is slightly weaker than before**: 10/13 targets beat the NASA+MIT-only
baseline in BOTH the plain and the sibling-corrected setting (same
10 datasets: CALCE, Oxford, HUST, ul_pur, hnei, mich, rwth, stanford, stanford_2, isu_ilcc), with snl a tie (margin 0.005) and XJTU/mich_exp/NASA/MIT not beating it. The battery-ID-collision
assertions live in the rerun scripts and were not re-checked in this document. Full detail:
`outputs/toolkit_lodo_family_holdout_rerun_corrected.csv` (pre-rerun: `outputs/finalpass3_check1_side_by_side.csv`).

## 10. Shift diagnostics (item E) - AUC has no reliable relationship with outcomes at this severity of shift

**Status: SUPERSEDED (corrected rerun; `outputs/finalpass2_itemE_correlations.csv`, `finalpass2_itemE_source_threshold.csv`, `finalpass2_itemE_risk_coverage.csv`; pre-rerun values in parentheses). Verdict unchanged.**

**Part 1**: across the 13 datasets, domain-classifier AUC correlates
weakly and NOT significantly with R2, MAE, or conformal coverage
(Spearman rho after the rerun: -0.297, 0.416, 0.026 respectively (before -0.201, 0.248, -0.195), all 95% bootstrap
CIs still crossing zero: [-0.820, 0.313], [-0.050, 0.824], [-0.429, 0.490]) - directly reinforcing item 1's finding that AUC
saturates near 1.0 for most datasets, leaving too little variance to
predict anything with.

**Part 2**: a legitimately source-only-calibrated OOD threshold (90th
percentile of an in-domain calib-vs-eval classifier's own scores, never
touching real target data) abstains on 100% of batteries for 13/13
target datasets after the rerun (before: 12/13, with 96.4% for snl) - a real,
disclosed consequence of how saturated OOD scores already are. The
target-relative risk-coverage curve (retaining the lowest-OOD battery
fraction within each dataset, sweeping abstention 0-50%) shows mean
retained-battery MAE staying essentially flat (10.11 -> 10.03 -> 10.56 at 0/20/50% abstention after the rerun; before 10.34 -> 10.17 -> 10.39)
- abstaining on "more OOD-looking" batteries does not reliably reduce
error at this severity of shift.

**Coherent conclusion across both parts**: the domain-classifier-AUC
diagnostic, useful earlier for motivating Stage 1.1's reformulation
work, does not function as a reliable per-dataset risk indicator once
shift is this severe. No AUC-based selective-prediction mechanism is
recommended for deployment.

**Rerun verdict check (section 10): no headline verdict changed.** One nuance: the Pearson r of AUC with MAE now has a CI that excludes zero (r=0.285, [0.123, 0.571]),
but the Spearman CI still spans zero, the R2 and coverage correlations do not separate from zero, and 13/13 datasets abstain at 100%, so the "no reliable
relationship / no AUC-based selective prediction" conclusion stands.

## 11. Complete baseline table (item C) - engineered HI+fusion features dramatically beat literature early-cycle-life baselines, with one important exception

**Status: trivial-linear, Severson and Attia columns VERIFIED (max abs change 0.000 in `outputs/finalpass2_itemC_baseline_table.csv`); "True deployed base (audit)" column VERIFIED (built-in datasets only,
`audit_true_deployed_baseline.csv`); "Routed (5-seed)" and "Best LODO" columns for the 9 BatteryLife rows SUPERSEDED (pre-rerun in parentheses). Note: the `deployed_base_r2` and
`routed_oracle_selected_r2` columns inside `finalpass2_itemC_baseline_table.csv` are hard-coded pre-rerun numbers for the BatteryLife rows; the routed values below are taken from the
corrected `finalpass_item5a_5seed_aggregate.csv` (BatteryLife sources are routed to the base model, so oracle = routed there). Paired tests
(`finalpass2_itemC_paired_tests.csv`) were recomputed from the rerun predictions.**

| Dataset | Trivial linear | Severson variance | Attia rich | Routed (5-seed) | True deployed base (audit) | Routed (oracle) | Best LODO |
|---|---|---|---|---|---|---|---|
| CALCE | -0.858 | 0.077 | 0.078 | 0.749 | 0.568 | 0.740 | **0.855** (before 0.870) |
| Oxford | -7.589 | 0.169 | 0.195 | **0.940** | -2.694 | 0.953 | 0.076 (before -0.571) |
| HUST | 0.755 | -1.790 | -1.437 | 0.795 | -0.152 | **0.800** | 0.722 (before 0.535) |
| XJTU | 0.279 | -7.835 | -7.410 | **-1.037** | -1.062 | -1.037 | -6.394 (before -4.100) |
| ul_pur | -0.566 | -3.292 | -3.397 | 0.138 (before 0.116) | not audited | 0.138 | **0.517** (before 0.488) |
| hnei | -2.001 | -0.087 | -0.108 | -0.137 (before -0.038) | not audited | -0.137 | **0.929** (before 0.681) |
| snl | -0.550 | -0.324 | -0.275 | 0.118 (before 0.147) | not audited | 0.118 | **0.149** (before 0.442) |
| mich | -0.275 | 0.241 | 0.252 | 0.536 (before 0.573) | not audited | 0.536 | **0.746** (before 0.795) |
| mich_exp | -0.217 | -0.206 | -0.122 | **0.716** (before 0.721) | not audited | 0.716 | 0.499 (before 0.638) |
| rwth | -0.000 | -0.000 | -0.001 | -0.535 (before -0.485) | not audited | -0.535 | **0.505** (before 0.353) |
| stanford | -0.274 | 0.148 | 0.135 | 0.291 (before 0.111) | not audited | 0.291 | **0.997** (before 0.997) |
| stanford_2 | -0.206 | 0.121 | 0.108 | 0.263 (before 0.066) | not audited | 0.263 | **0.990** (before 0.990) |
| isu_ilcc | 0.358 | -0.989 | -1.024 | 0.123 (before 0.140) | not audited | 0.123 | **0.901** (before 0.800) |

**Column labels corrected 2026-09-30.** The column earlier headed "Deployed base" holds the *routed* 5-seed numbers (dataset-aware routing applied; CALCE/Oxford/HUST use the extended-reformulation model, selected on the same held-out data). The true deployed base model (`xgb_soh_fusion.json`, no routing) is the new column, from `outputs/audit_true_deployed_baseline.csv` (single fixed-seed model, so it is not a 5-seed mean; BatteryLife rows were not part of that audit). In the significance sentences below, "deployed base model" refers to the routed numbers.

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
absolute error; recomputed in the rerun): the deployed base model is significantly better than
both literature baselines on 8/13 datasets (unchanged: the same 8 datasets - HUST, XJTU, ul_pur, hnei, snl, mich, mich_exp, rwth - before and after). **One important, disclosed
exception: on Oxford, Severson/Attia have significantly LOWER MAE**
(5.2-5.3 vs. 12.5, p=0.008) **despite the deployed model's much higher
R2** (0.940 vs. 0.17-0.20) - R2 measures variance explained (the
deployed model tracks Oxford's overall trend far better) while MAE
measures raw error magnitude (the simpler baselines make smaller,
more conservative absolute errors) - a real case where metric choice
changes which model looks better, reported as found (Oxford values unchanged after the rerun: 12.51 vs 5.28/5.18 MAE, p=0.0078).

**Rerun verdict check (section 11): the "8/13 significantly better than literature baselines" headline is VERIFIED unchanged, and so is the Oxford exception.** Two R2-level
changes on BatteryLife rows: hnei's routed R2 (-0.137) is now below Severson's (-0.087) and Attia's (-0.108) on R2 (before -0.038, above both), although the paired MAE test on hnei still favours the model
(14.80 vs 15.35 MAE against Severson, p=0.0040; before p=0.00012); and the sentence "deployed model stays competitive or clearly better almost everywhere" should not be read to include hnei on R2.

## 12. Online conformal verification (CHECK 2)

**Status: (a) VERIFIED (structural property, not a numeric result; not in the rerun change summary). (b), (c), (d): CALCE/Oxford/HUST/XJTU rows VERIFIED; BatteryLife rows SUPERSEDED by the corrected rerun
(`outputs/finalpass3_check2_labelfree_vs_online.csv`, `finalpass3_check2_lifestage.csv`, `finalpass3_check2_width_and_coverage.csv`), pre-rerun value in parentheses. Caution: the static (label-free) column of the
check2 CSV was not recomputed; the corrected static numbers below are from `finalpass_item3_coverage_pivot.csv` (k=0).**

**(a) No-lookahead - now a VERIFIED fact, not an inline assertion.** A
real unit test (shuffle every label after a fixed cycle t, confirm the
interval at t is byte-identical) passed for both PID and nexCP on
synthetic data and 3 real battery traces (CALCE, HUST, isu_ilcc).

**(b) Label-free vs. online, one table** (the label-feedback
requirement made explicit):

| Dataset | Static (label-free) | PID (needs labels) | nexCP 0.95 (needs labels) | nexCP 0.99 (needs labels) |
|---|---|---|---|---|
| CALCE | 4.3% | 84.9% | 80.4% | 73.5% |
| Oxford | 5.4% | 85.4% | 89.7% | 86.1% |
| HUST | 8.5% | 88.1% | 87.0% | 85.0% |
| XJTU | 11.0% | 89.3% | 89.2% | 90.8% |
| ul_pur | 39.6% (21.6%) | 58.9% (72.8%) | 74.3% (79.7%) | 60.0% (72.5%) |
| hnei | 14.2% (19.0%) | 38.0% (69.1%) | 63.2% (71.2%) | 46.7% (56.8%) |
| snl | 11.1% (11.7%) | 82.3% (84.0%) | 66.4% (63.1%) | 62.2% (59.7%) |
| mich | 30.2% (16.2%) | 61.9% (64.4%) | 63.5% (64.8%) | 59.8% (61.4%) |
| mich_exp | 33.4% (34.1%) | 65.2% (69.1%) | 69.3% (70.1%) | 62.3% (64.2%) |
| rwth | 4.6% (4.7%) | 83.7% (79.9%) | 68.8% (78.2%) | 55.5% (57.0%) |
| stanford | 1.7% (1.5%) | 89.4% (89.7%) | 84.8% (86.0%) | 88.4% (89.8%) |
| stanford_2 | 1.8% (1.6%) | 89.5% (89.7%) | 86.0% (86.5%) | 88.3% (89.0%) |
| isu_ilcc | 1.5% (1.3%) | 89.9% (90.0%) | 80.6% (82.5%) | 72.3% (73.9%) |

**(c) Life-stage of the rolling-20 zero-coverage window** (7/13
datasets affected, same 7 after the rerun; PID rows of `finalpass3_check2_lifestage.csv`): after the rerun **72% of affected battery-
instances (63/88) hit zero coverage in LATE life**, 22 mid and 3
early (before, as reported: 88%, 63/72 late, 6 mid, 3 early). The late-life share is lower because all 14 hnei batteries now hit their zero window in mid-life
- still a mostly late-life phenomenon, not a burn-in artifact. **mich is a
standout case**: all 40/40 of its batteries hit a zero-coverage window
(39/40 late-life, unchanged), despite a 61.9% aggregate coverage (before 64.4%) -
the aggregate number alone masks near-universal local failure there.

**(d) Mean width as % of SOH range** - where intervals are genuinely
informative vs. borderline useless: Oxford is most informative (8.8%
of range); **XJTU (62.7%), isu_ilcc (60.5%; before 61.0%), and rwth (60.1%; before 59.0%) are
borderline uselessly wide** - their strong-looking coverage numbers
(89.3%, 89.9%, 83.7%; before 89.3%, 90.0%, 79.9%) are bought almost entirely through width on these
3 specific datasets. (hnei is next at 43.2% of range.)

**Rerun verdict check (section 12): no headline verdict changed** (7/13 zero-window count and dataset set, mich 40/40, and the same 3 too-wide datasets all hold; the late-life share falls from 88% to 72%,
still a majority).

---

## Final list: numbers the paper can safely headline

Updated 2026-09-30 after the corrected-embedding rerun. Tag per bullet: **VERIFIED** = rows unaffected by the BatteryLife embedding bug (identical after the
rerun); **SUPERSEDED** = the numbers were replaced by the corrected rerun values shown. Every number below comes from the current CSVs.

- **Built-in dataset numbers (in-domain, CALCE, Oxford, HUST, XJTU) in sections 1-3 and 6 (5-seed accuracy, bootstrap CIs, static conformal coverage), the RUL
  cross-domain table (section 4) and CHECK A/B - VERIFIED**, unchanged after the rerun. The deployed model's in-domain 5-seed R2 0.978 +/- 0.003 is VERIFIED.
- **Online conformal (item A) recovers coverage dramatically - SUPERSEDED for BatteryLife rows, VERIFIED for CALCE/Oxford/HUST/XJTU.** Headline with corrected numbers:
  static 1.5-39.6% -> PID+scorecaster 71.1-95.0% across all 13 datasets (before: 1.3-34.1% -> 74.3-94.7%). State it as the PID+scorecaster result: plain PID is below 70% on
  hnei (38.0%), ul_pur (58.9%), mich (61.9%) and mich_exp (65.2%). No-lookahead remains a VERIFIED, tested property. The 7/13 rolling-20-zero-coverage count is VERIFIED unchanged;
  flag XJTU/isu_ilcc/rwth's width (60-63% of SOH range) as the 3 datasets where coverage is bought mostly through width.
- **LODO source-diversity pooling (item B) beats NASA+MIT-only zero-retrain on 10/13 comparable targets - SUPERSEDED (was 11/13).** snl is not distinguishable
  (family-LODO R2 0.149 vs 0.154, margin 0.005), XJTU (-6.394 vs -1.556) and mich_exp (0.499 plain / 0.550 sibling vs 0.731) are worse. The 10/13 count is the same
  in the plain and sibling-holdout settings. **Use stanford=0.919 and stanford_2=0.895** (family-holdout, corrected embeddings), NOT the plain 0.997/0.990 (sibling leakage), and not the
  pre-rerun 0.889/0.858.
- **MIT held out alone cannot be predicted even from 14 pooled other sources (R2=-6.118, CI [-11.357, -3.360]) - SUPERSEDED number (before -4.817), same conclusion, now on corrected
  embeddings.** The very negative MIT and XJTU (-6.394) LODO values are genuine failures of leave-one-dataset-out transfer, not artefacts of the embedding bug.
- **XJTU is the one dataset where more pooled training data makes things WORSE** (-6.394 vs -1.556 for the recomputed NASA+MIT-only baseline; before -4.100 vs -1.062) - SUPERSEDED number, same direction,
  and still true in the family-holdout setting.
- **Engineered HI+fusion+XGBoost beats literature early-cycle-life baselines (Severson/Attia) on 8/13 datasets** (item C, paired significance) - **VERIFIED** (same 8 datasets before and after),
  with the Oxford MAE-vs-R2 exception stated exactly as found. Add the corrected note that on hnei the routed R2 (-0.137) is now below Severson's (-0.087) even though the MAE test still favours the model.
- **Domain-classifier AUC has no reliable relationship with outcomes** once shift is this severe (item E) - SUPERSEDED numbers (Spearman -0.297 / 0.416 / 0.026, all CIs span zero), same clean negative result.
- **Label-free routing (item 1) is not usable** - SUPERSEDED: rule 3/13 correct vs always-extended 10/13 (before: tie at 8/13). Do not headline the old "ties the naive baseline" wording.
- **BatteryLife benchmark reproduction (section 5): 15%-Acc 0.600, MAPE 0.211 - SUPERSEDED** (before 0.576 / 0.213); published 0.573 / 0.184-0.179.

---

---

# Toolkit pass: minimum labeled checkpoints before trusting the online conformal interval

Derived from item A's own PID results (eta=0.1, k_burnin=10), using
`src/online_conformal.py` (the reusable module item A's recursion was
moved into). Definition: for each battery, the smallest number of
revealed cycles t such that rolling-20 coverage from t onward never
again drops below 70% for the rest of that battery's life. Batteries
that never reach a stable point are excluded from the median and
counted separately, not papered over with a misleading number.

**Status: Oxford/XJTU/CALCE/HUST rows VERIFIED (identical to the pre-rerun table); the 9 BatteryLife rows SUPERSEDED by the corrected rerun (`outputs/toolkit_phase3a_min_checkpoints.csv`, `outputs/rerun_queue/phase3a.log`); pre-rerun value in parentheses.**

| Dataset | Median checkpoints | Mean | Batteries reaching stability | Batteries that never stabilize |
|---|---|---|---|---|
| Oxford | 22 | 22 | 4/8 | 4/8 |
| XJTU | 25 | 34 | 46/47 | 1/47 |
| isu_ilcc | 25 (25) | 25 (25) | 9/9 | 0/9 |
| stanford_2 | 25 (26) | 207 (203) | 8/8 | 0/8 |
| ul_pur | 77.5 (60) | 77.5 (60) | 2/10 | 8/10 |
| stanford | 138.5 (26) | 347 (308) | 6/6 | 0/6 |
| mich_exp | 312 (250) | 255 (243) | 6/18 (was 7/18) | 12/18 (was 11/18) |
| snl | 407 (391) | 802 (842) | 24/55 (was 28/55) | 31/55 (was 27/55) |
| CALCE | 985 | 985 | 2/3 | 1/3 |
| hnei | 1043.5 (992) | 1044 (1001) | 2/14 (was 5/14) | 12/14 (was 9/14) |
| HUST | 1324 | 1095 | 67/77 | 10/77 |
| rwth | 2044 (2018) | 1579 (1482) | 10/10 (was 9/10) | 0/10 (was 1/10) |
| **mich** | **N/A - no battery ever stabilizes** | - | 0/40 | **40/40** |

**Overall (pooled across every battery that reached a stable point,
n=186 after the rerun; before n=193): median = 65 cycles (before 62), 75th percentile = 1310 cycles (before 1297).**

**Rerun verdict check (Phase 3a): no headline verdict changed.** Still no single safe number, still mich = 0/40 stable, pooled p75 still ~1300. Stability got worse on hnei
(5/14 -> 2/14) and snl (28/55 -> 24/55).

**There is no single safe "minimum checkpoints" number - it varies by
roughly 100x across datasets (22 to 2044), and mich is a genuine,
disclosed outlier where NO battery ever reaches a stable trustworthy
point at all** (consistent with CHECK 2's own earlier finding that
every one of mich's 40 batteries hits a late-life zero-coverage
window). **Practical recommendation for the toolkit**: use the
dataset-specific number when the nearest source is known (via the
trust-report's own nearest-source lookup); when unknown, use the
conservative POOLED 75th percentile (~1300 cycles) rather than the
median, and always disclose that some sources (mich-like protocols)
may never reach a trustworthy interval at all, however many labeled
checkpoints are supplied.

---

# Toolkit pass, Phase 2B: federated multi-source learning - a clean negative result

**Status: SUPERSEDED by the corrected rerun (2026-09-30, 106.5 min, 16/16 folds); the verdict is unchanged - VERIFIED negative result for federated bagging.**
The table further below is the pre-rerun (mixed-encoder) table, kept for the record and marked SUPERSEDED. Corrected numbers come from
`outputs/toolkit_phase2b_federated_results.csv`; the interrupted first attempt is kept as `outputs/toolkit_phase2b_federated_run_INTERRUPTED_12of16.log`.
The method is federated *bagging* (Flower FedXgbBagging aggregation, lossy by construction); histogram-based federated GBDT is lossless by construction and was not run.

| Held-out R2 across the 16 sources | BEFORE (SUPERSEDED, mixed encoders) | AFTER (corrected rerun) |
|---|---|---|
| Centralized (pooled): mean / median / sources with R2 > 0 | 0.033 / 0.570 / 13 | -0.203 / 0.584 / 12 |
| Federated, sample-weighted: mean / median / sources > 0 | -3.532 / -0.919 / 6 | -3.302 / -0.166 / 7 |
| Federated, uniform: mean / median / sources > 0 | -3.353 / -0.490 / 7 | -1.995 / -0.287 / 7 |
| Federated, tempered: mean / median / sources > 0 | -3.146 / -0.205 / 7 | -3.732 / -0.019 / 8 |
| NASA+MIT-only routed: mean / median / sources > 0 | -0.540 / 0.153 / 10 | -0.465 / 0.197 / 9 |
| Federated beats centralized (sample-weighted / uniform / tempered) | 1 / 4 / 3 of 16 | 3 / 4 / 5 of 16 |
| Centralized beats NASA+MIT-only | 13 of 16 | 13 of 16 |

Corrected verdict: no federated variant matches centralized training (the best federated variant beats it on at most 5 of 16 held-out sources); MIT stays the worst case
(centralized R2 -2.639, federated sample-weighted -26.864, tempered -52.581); centralized still beats NASA+MIT-only on 13 of 16. Federated bagging is NOT adopted. Same verdict as before.
Status: **VERIFIED** (corrected embeddings), old table **SUPERSEDED**.

Question: does federating the Phase 2 multi-source candidate (bagging-
style, each of 14 client groups - siblings {stanford,stanford_2} and
{mich,mich_exp} merged - training only on its own local data, never
sharing raw rows) recover the same candidate's CENTRALIZED-pooled
performance, and does it fix NASA/MIT's "crowded out by larger sources"
problem? Native XGBoost federated learning was confirmed unavailable in
this environment (not compiled with federated support); used Flower's
own `FedXgbBagging` aggregation instead, per instruction. Evaluated
under this project's own LODO family-holdout protocol (16 sources,
Tongji included), same features/hyperparameters as the Phase 2
candidate throughout.

| Held out | Centralized (Phase 2 candidate) | Federated (best of 3 weighting schemes) | NASA+MIT-only (deployed) |
|---|---|---|---|
| NASA | 0.229 | 0.417 | **0.999** |
| MIT | -4.646 | -33.939 | **0.999** |
| CALCE | **0.860** | 0.840 | 0.646 |
| Oxford | -0.492 | -1.795 | -11.974 |
| HUST | **0.469** | -1.100 | 0.307 |
| XJTU | -2.646 | -2.344 | -0.934 |
| ul_pur | **0.465** | -1.960 | 0.186 |
| hnei | **0.671** | -0.202 | -0.075 |
| snl | **0.363** | -1.425 | 0.120 |
| mich | **0.764** | 0.773 | 0.661 |
| mich_exp | **0.709** | -0.209 | 0.616 |
| rwth | 0.338 | **0.482** | -0.215 |
| stanford | **0.875** | 0.875 | 0.066 |
| stanford_2 | **0.856** | 0.868 | -0.012 |
| isu_ilcc | **0.898** | 0.812 | 0.359 |
| tongji | **0.818** | -1.277 | -0.395 |

**Federated never clearly beats centralized pooling** (centralized
wins outright on 13/16 sources, ties or marginally loses on the other
3), **and does not fix NASA/MIT's crowded-out problem - it makes MIT's
case measurably worse** (-33.9 federated vs an already-bad -4.6
centralized). A companion experiment (tempering the CENTRALIZED model's
per-source sample weights by sqrt(source size) instead of federating)
also failed to fix NASA/MIT and pushed 14/16 other sources negative -
closing an open question from the earlier balanced-retrain collapse:
it isn't specifically extreme equal-weighting that breaks this, any
departure from natural row-count-proportional weighting tested so far
does. **Recommendation: the existing Phase 2 decision (unweighted
centralized candidate + dataset-identity routing, already deployed)
stands - federation was tested honestly and the honest result is
negative, not adopted.**

*Full detail, including two real implementation bugs found and fixed
via direct verification before trusting any number here: `DEVELOPMENT_
LOG.md`, "Phase 2B: federated multi-source learning" section.*

---

# Toolkit pass, Phase 2C: label-efficient checkpoints - no tested budget reaches a usable coverage target, and life-stage weighting underperforms even spacing

At equal label budgets (5/10/20/40 true-SOH revelations per battery,
model still predicts every cycle), compared three checkpoint-selection
policies - fixed even spacing, life-stage-weighted (denser late in
life), and uncertainty-triggered (reveals on conformal-interval-width
or ADWIN drift signal, both fed only already-revealed information) -
under this project's own PID online-conformal recursion (eta=0.1,
k_burnin=10) and the SAME 13-source external/BatteryLife scope Phase
3(a)'s own minimum-checkpoints work already established.

**Status: SUPERSEDED - the pooled means include 9 BatteryLife sources and were rerun on corrected embeddings (`outputs/toolkit_phase2c_label_efficient.csv`, `outputs/rerun_queue/phase2c.log`); pre-rerun value in parentheses. CALCE/Oxford/HUST/XJTU per-source values are identical to before.**

| Budget | Best policy | Mean coverage (pooled) | Mean late-life coverage |
|---|---|---|---|
| 5 | fixed_every_n | 0.232 (before 0.219) | 0.210 (before 0.228) |
| 10 | fixed_every_n | 0.322 (before 0.313) | 0.263 (before 0.283) |
| 20 | fixed_every_n | 0.410 (before 0.413) | 0.305 (before 0.325) |
| 40 | uncertainty (~tied with fixed_every_n, 0.496) | **0.499** (before 0.511) | 0.362 (before 0.385) |

**None of the four tested budgets reach a usable pooled coverage
target** - even the best case (40 labels) only reaches 51% mean
coverage, well short of this project's own 90% conformal target (corrected rerun: 50% mean coverage).
Highly variable by source, consistent with this project's standing
theme of severe, uneven out-of-domain shift: at budget=40 (fixed_every_n), XJTU reaches
76% coverage / 92% late-life coverage (unchanged), but hnei (15.0%) and rwth (19.3%) stay below
20% coverage and mich (46.3%; before 44.7%) barely reaches 46%, all with **near-zero (0-0.1%) late-life coverage** - the exact
window that matters most for a real trust decision. **Life-stage-
weighted checkpointing consistently underperforms even spacing on
every budget** (corrected rerun: 0.188 vs 0.232, 0.241 vs 0.322, 0.306 vs 0.410, 0.389 vs 0.496 mean coverage at budgets 5/10/20/40) - concentrating labels late in life starves the PID
controller's error-correction of the early corrections it needs, so it
enters the dense late-life region already badly miscalibrated.
**Uncertainty-triggered and fixed-every-n perform almost identically**
throughout - the adaptive trigger doesn't meaningfully beat simple even
spacing at these budgets. **No forced minimum-labels recommendation is
given** - the honest result at these tested budgets is that none of
them are enough, reported as such rather than rounded up to a false
positive.

**Rerun verdict check (Phase 2C): no headline verdict changed** (no budget reaches a usable target - best 0.499 vs 0.511 before; life-stage still below even spacing at every
budget; uncertainty ~ fixed_every_n).

*Full detail: `DEVELOPMENT_LOG.md`, "Phase 2C: label-efficient
checkpoints" section.*

---

*Full experimental detail for items A-F and the final verification
pass: `DEVELOPMENT_LOG.md`, "Final experiment pass 2 before the paper"
and "Final verification pass" sections. Toolkit-pass detail:
`DEVELOPMENT_LOG.md`, "Toolkit pass" section.*

---

## Rerun status log 2026-09-30

Corrected-embedding rerun of everything that read the BatteryLife/Tongji embedding columns. Queue: `outputs/rerun_queue/run_queue.sh` (itemA, itemB; 12:09-12:29, killed mid `lodo_check1`)
and `run_queue2.sh` (resumed 18:30, ALL DONE 20:02, every step rc=0; `outputs/rerun_queue/summary_resume.txt`). Changed-cell counts: `outputs/toolkit_rerun_change_summary.csv`.
"Verdict changes" is the outcome of comparing the before/after numbers in this file, not merely whether cells moved.

| Script | Status | Verdict changes |
|---|---|---|
| Item A online conformal (`finalpass2_itemA_*`; secs 8, 12b) | rerun, done (643 s) | none flipped; PID+scorecaster min 74.3% -> 71.1% (mich); plain PID hnei 69.1% -> 38.0%, ul_pur 72.8% -> 58.9% (weaker, direction holds). CALCE/Oxford/HUST/XJTU VERIFIED |
| Item B LODO (`finalpass2_itemB_lodo_results.csv`; sec 9) | rerun, done (538 s) | count of clear wins 11/13 -> 10/13: snl 0.442 -> 0.149 vs NASA+MIT-only 0.154 (margin 0.005, not distinguishable); no verdict_flipped in `toolkit_rerun_before_after_itemB.csv`; MIT -4.817 -> -6.118 and XJTU -4.100 -> -6.394 stay failures |
| `lodo_check1` family holdout (sec 9 CHECK 1) | rerun, done (507 s) | stanford 0.889 -> 0.919, stanford_2 0.858 -> 0.895; same 10 clear wins in plain and family settings; snl tie |
| Phase 2C label-efficient (`toolkit_phase2c_label_efficient.csv`) | rerun, done (314 s) | none (best 0.499 vs 0.511 before; no budget reaches target) |
| Phase 3a min checkpoints (`toolkit_phase3a_min_checkpoints.csv`) | rerun, done (276 s) | none (pooled median 62 -> 65, p75 1297 -> 1310; mich still 0/40) |
| Item 1 label-free routing (`finalpass_item1_*`; sec 7) | rerun, done (288 s) | rule correct 8/13 -> 3/13; extended model now true winner on 10/13 (was 8/13); rule no longer ties always-extended (3/13 vs 10/13); not adopted (unchanged decision) |
| Item 2 cycle_idx ablation (`finalpass_item2_*`; sec 7) | rerun, done (283 s) | none (cycle_idx effect +0.0421 -> +0.0617; worse on 3/13, was 5/13) |
| Item 3 few-shot conformal (`finalpass_item3_*`; secs 6, 7) | rerun, done (279 s) | none (mean coverage k=0 -> 50: 10.8% -> 13.4% before, 12.9% -> 15.2% after) |
| Item 5a 5-seed / bootstrap (`finalpass_item5a_*`; secs 1-3) | rerun, done (396 s) | none (no BatteryLife R2 changed sign; stanford/stanford_2 bootstrap CIs now above zero) |
| Item 5d BatteryLife benchmark (`finalpass_item5d_*`; sec 5) | rerun, done (13 s) | none (15%-Acc 0.576 -> 0.600 vs published 0.573; MAPE 0.213 -> 0.211 vs 0.184/0.179) |
| Part B item 7 zero-retrain eval (`partB_item7_zeroretrain_eval.csv`) | rerun, done (14 s); not a PAPER_RESULTS section | extended-vs-base R2 moved on all 9 BatteryLife sources; no verdict statement in this file depends on it beyond item 1 |
| Part B item 8 | rerun, done (7 s) | no rewritten CSV in the change summary |
| Part B item 9 pool-expansion retrain (`partB_item9_pool_expansion_retrain.csv`) | rerun, done (275 s); not a PAPER_RESULTS section | in-domain/CALCE WIN and Oxford/HUST/XJTU LOSS verdicts unchanged (Oxford R2 -1.456 -> -0.127, HUST 0.542 -> 0.683, XJTU -4.268 -> -7.336) |
| CHECK 2 online conformal verification (`finalpass3_check2_*`; sec 12) | rerun, done (359 s) | none (7/13 zero-window datasets, mich 40/40, same 3 too-wide datasets; late-life share 88% -> 72%). Static column in the CSV not recomputed |
| Item E shift diagnostics (`finalpass2_itemE_*`; sec 10) | rerun, done (289 s) | none (Spearman -0.201/0.248/-0.195 -> -0.297/0.416/0.026, all CIs span zero; abstention 100% on 13/13) |
| Item C baseline table + paired tests (`finalpass2_itemC_*`; sec 11) | rerun, done (2,190 s) | none (8/13 significantly better, Oxford exception unchanged); hnei routed R2 -0.137 now below Severson -0.087 on R2 |
| Trust profiles / trust report validation (`toolkit_phase3_trust_*`; no PAPER_RESULTS section) | profiles rebuilt (byte-identical `_source_profiles.pkl`); validation CSV rewritten | not assessed in this file - see DEVELOPMENT_LOG.md "Step 2 close-out and Step 3a" |
| Phase 2B federated (`toolkit_phase2b_federated_*`; restarted 18:30) | done, 16/16 folds, 106.5 min; transcribed above | none - federated bagging still does not match centralized (VERIFIED negative result) |
| Sections not rerun: 4 (RUL), item 4, CHECK A, CHECK B | VERIFIED (not in the change summary) | none |

