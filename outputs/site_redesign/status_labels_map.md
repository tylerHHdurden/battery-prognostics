# Status labels map: Model Validation tab, Full Results Archive tab, Showcase headline metrics

Authority: `PAPER_RESULTS.md` (rerun banner dated 2026-09-30). Read-only audit of `app.py` (3283 lines); no code changed.
Line numbers are approximate (start of the expander / call in `app.py`).

## Key findings

1. Every in-domain number in the Model Validation tab and in archive sections 3-23 is from the FIRST-PASS 32-battery pool (6 test batteries, single seed, in-domain R2 0.917). The authority's in-domain number is the 5-seed deployed-config R2 0.978 +/- 0.003 (PAPER_RESULTS s1, VERIFIED). So these are SUPERSEDED as headlines; they remain valid as labelled first-pass experiments.
2. Nothing in either tab reads BatteryLife or Tongji embeddings (the tabs use only NASA/MIT/CALCE/Oxford/HUST/XJTU and the 204-battery NASA+MIT pool). So **no app number is INVALIDATED** by the embedding-normalisation bug. The bug only touches PAPER_RESULTS BatteryLife rows, which the app never shows.
3. The app's "Reformulation / Zero-retrain / Severson" tables (archive s33, s34, s36) use the single-seed Stage-4 figures (CALCE 0.740, Oxford 0.953, HUST 0.800, XJTU -1.775, in-domain 0.973). These differ from the authority's 5-seed routed numbers (0.749 / 0.940 / 0.795 / -1.037 / 0.978). XJTU differs most, because the app uses the extended-reformulation model's -1.775 while deployed routing sends XJTU to the base model (-1.037 five-seed, -1.062 single-seed).
4. The app never shows PAPER_RESULTS s8 (online conformal PID+scorecaster, CALCE 87.4%), so the archive's "no method works on CALCE" story (s35) and the Showcase "6.1% coverage" headline are incomplete against the authority.

Status codes: VERIFIED (final, matches PAPER_RESULTS), SUPERSEDED (replaced; replacing value named), INVALIDATED (affected by embedding bug; none in app), HISTORICAL-EXPERIMENT (session narrative, not a headline). "NO-AUTHORITY" marks a real session result that PAPER_RESULTS simply does not cover (counted as HISTORICAL-EXPERIMENT).

## A. Showcase headline metrics (`render_showcase_tab` ~2577, `render_headline_numbers` ~2555)

| App location | Number(s) shown | Status | Source / replacing value |
|---|---|---|---|
| render_headline_numbers, card "Ensemble R2" (~2565) | 0.917 | SUPERSEDED | First-pass 32-battery, 6-test-battery, single-seed R2 (`data/processed/predictions/ensemble_fusion_metrics.csv` 0.9169, `xgb_fusion_metrics.csv` 0.9172; DEVELOPMENT_LOG ~858-860). Replaced by in-domain 5-seed R2 0.978 +/- 0.003 (PAPER_RESULTS s1; bootstrap 0.976 [0.945, 0.996], s3). **CONFLICT C1** |
| render_headline_numbers, card "Lean pipeline speedup" (~2566) | 52x (3.9 ms vs 201.9 ms), "faster than the full 5-branch ensemble" | HISTORICAL-EXPERIMENT (NO-AUTHORITY) | `outputs/lean_vs_full_comparison.csv`; DEVELOPMENT_LOG 2426 (session 20). Not in PAPER_RESULTS. DEVELOPMENT_LOG "Stage 4" notes the deployed app ran the full 4-branch ensemble before the Stage 4 retrain, so "lean vs full" is a session-20 comparison of 32-battery models, not a statement about the current routed deployment |
| render_headline_numbers, card "CALCE conformal coverage" (~2567) | 6.1% of a 90% target | SUPERSEDED | First-pass 32-battery value (`outputs/calce_zero_retrain_conformal.csv`, 0.0612). Replaced by 4.3% (PAPER_RESULTS s6 and s8 static baseline, 5-seed routed, n=2,941). **CONFLICT C2**. Direction (catastrophic undercoverage) is unchanged; PAPER_RESULTS s8 also shows online PID+scorecaster recovers CALCE to 87.4% (not shown in app) |
| render_headline_numbers, card "B0018 MAE, online-corrected" (~2568) | 2.86 pp (from 4.64 pp raw); earlier linear corrector 4.43 pp | HISTORICAL-EXPERIMENT (NO-AUTHORITY) | DEVELOPMENT_LOG Stage 6.3 (~10613-10618): River HoeffdingAdaptiveTree 2.858 vs SGD 4.426, B0018 only. Single-battery replay, not in PAPER_RESULTS, not affected by the embedding bug. Raw 4.64 is the pre-Stage-4 frozen pipeline (4.641 in archive s30) |
| render_showcase_tab, warning box (~2581) | "REPLAY of already-recorded, already-verified results (sessions 28-29)" | HISTORICAL-EXPERIMENT | Per-cycle CSVs `streaming_dt_NASA_B0018.csv`, `streaming_dt_MIT_b3c35.csv`; label is accurate but "verified" means session-level verification, not PAPER_RESULTS |
| render_showcase_tab, per-cycle metrics (raw / digital-twin / true SOH), `_showcase_verdict` | live from CSV rows | HISTORICAL-EXPERIMENT | same CSVs; B0018 (NASA) and b3c35 (MIT) |

## B. Model Validation tab (`main()` ~3273: `render_evaluation_protocol_section` ~890, then "Battery comparison mode" ~3276)

All three tables are read from `data/processed/predictions/`, the first-pass 32-battery, 6-test-battery, single-seed pipeline.

| App location | Number(s) shown | Status | Source / replacing value |
|---|---|---|---|
| render_evaluation_protocol_section, "1. Early-prediction test" (~905) | pooled full-life R2 0.917 (XGB-fusion), 0.917 (Stacking-Ridge); early-life (first 20%) R2 -0.58, RMSE 0.847, MAE 0.287 | SUPERSEDED (full-life rows) / HISTORICAL-EXPERIMENT (early-life rows) | `early_prediction_test.csv`. Full-life R2 replaced by 0.978 +/- 0.003 (PAPER_RESULTS s1). Early-life regime not in PAPER_RESULTS. **CONFLICT C3** |
| same, per-battery early table (~911) | b2c24 R2 -52.3; B0018 -3.37; b1c4 -3.04; b3c0 -1.17; b3c35 0.279; b4c38 0.309 | HISTORICAL-EXPERIMENT | `early_prediction_per_battery.csv` (b2c24 -52.32 confirmed; soh_range 99.9-100.4). Not in PAPER_RESULTS |
| same, warning text "R2 goes negative ... use RMSE/MAE" (~907) | qualitative | HISTORICAL-EXPERIMENT | text matches CSV |
| "2. Drop-one-branch ablation (Ridge meta-learner refit without each base learner)" (~916) | reads `drop_branch_ablation.csv` (4-branch); caption "Dropping XGBoost collapses performance" | SUPERSEDED / HISTORICAL-EXPERIMENT | First-pass 4-branch table. The archive uses the 5-branch file (`drop_branch_ablation_5branch.csv`: dropping XGB-fusion dR2 -0.0967, R2 0.820 vs 0.9169). Not in PAPER_RESULTS; deployed 5-seed config is XGBoost-fusion |
| "3. Homogeneous-bagging baseline (5 XGBoost seeds averaged)" (~925) | single XGB R2 0.9172, hetero stack 0.9169, 5-seed bag 0.9134 (RMSE 1.423) | HISTORICAL-EXPERIMENT | `homogeneous_bagging_comparison.csv`; not in PAPER_RESULTS |
| main(), "Battery comparison mode" / render_battery_comparison_section (~985, 3276) | live predictions, SHAP, conformal interval for two chosen cycles | not a results section | Live inference, no static numbers. Banner should say the OOD interval is an in-domain-only guarantee (PAPER_RESULTS s6: only 1 of 14 rows within 10 points of 90%) |

## C. Full Results Archive tab (`render_full_results_archive_tab` ~1145-2440)

Number in first column is the section number in the expander title.

| # | App location / heading (approx line) | Number(s) shown | Status | Source / replacing value |
|---|---|---|---|---|
| - | pipeline diagram, search box, caption (~1160) | none | n/a | - |
| 1 | BFA Feature Selection (~1201) | 7 of 16 HIs selected | HISTORICAL-EXPERIMENT | 32-battery BFA (DEVELOPMENT_LOG line 22). Later 204-pool BFA selected 8 (s32); deployed set is the reformulated 8-feature set (Stage 4) |
| 2 | SOH Fade Examples & ICA/DV/DC Example (~1208) | images only | HISTORICAL-EXPERIMENT | figures |
| 3 | Base Learner Training (~1217), CNN-LSTM table | VLSTM RMSE 2.694->2.131, R2 0.690->0.806; CNN-LSTM R2 -0.071->0.334, RMSE 5.006->3.948; PiFormer R2 0.735->0.617 | HISTORICAL-EXPERIMENT | `ensemble_comparison.csv` (VLSTM 2.131/0.806, PiFormer 2.993/0.617, CNNLSTM 3.948/0.334 match). Not in PAPER_RESULTS |
| 3 | same, ensemble_comparison.csv and cnn_bigru_metrics.csv tables | XGBoost R2 0.9066, Stacking-Ridge 0.9063, Stacking-XGB 0.9049; CNN-BiGRU R2 0.572 | SUPERSEDED (as headline) | 32-battery; replaced by XGB-fusion 5-seed 0.978 (s1) |
| 3 | same, "32 -> 204 batteries" table (~1262) | XGB R2 0.907->0.9721; VLSTM 0.806->0.9472; CNN-LSTM 0.334->0.9666; PiFormer 0.617->0.7795 (RMSE 2.993->3.3405); CNN-BiGRU 0.572->0.9568; PiFormer excl. B0053 RMSE 0.9693, R2 0.9814 | HISTORICAL-EXPERIMENT | Session 33 research finding (DEVELOPMENT_LOG 3971). Not in PAPER_RESULTS; 204-pool was not deployed (Stage 4 used 32 + recovered batteries) |
| 4 | Stacking Ensemble (~1285) | parity plot, per-cycle preds | HISTORICAL-EXPERIMENT | first-pass |
| 5 | Feature Fusion (~1296) | XGB RMSE 1.478->1.392 (23 features); XGB-fusion R2 0.9172; Ridge-fusion 0.9169 | SUPERSEDED | level replaced by 0.978 +/- 0.003 (s1); the 0.907->0.917 fusion gain is first-pass |
| 6 | Physics-Informed Loss Experiment (~1305) | negative result, lambda=0.1 | HISTORICAL-EXPERIMENT | `deep_models_physics_metrics.csv`; not adopted |
| 7 | Joint SOH+RUL Ablation (~1314) | alpha/beta 8.4/8.2 by epoch 24, loss -1.83, clamp [-0.7,0.7], alpha=beta=2.028; fixed_balanced SOH R2 0.416, adaptive 0.344; adaptive RUL 0.432 vs 0.428; soh_only RUL 0.030; rul_only SOH -0.040 | HISTORICAL-EXPERIMENT (RUL levels SUPERSEDED) | `joint_ablation.csv` (0.4164, 0.3437, 0.4277, 0.0302, -0.0398 confirmed). Deployed RUL: 0.374 in-domain for the retrained Stage-4 model, 0.666 first-pass fusion joint (PAPER_RESULTS s4). **CONFLICT C9** |
| 8 | SHAP Explainability (~1352) | XGB 7 HIs, SCV/VIECT/TEVI dominate; meta >99.9% XGBoost; VLSTM 60% in 3.55-3.8 V | HISTORICAL-EXPERIMENT | `shap_*.csv`; 7-feature first-pass HI set, not the current 8-feature set |
| 9 | Split-Conformal Prediction incl. the 27.1% bug (~1375) | 27.1% (buggy) -> SOH 95.1%, RUL 93.0%, SOH width 4.64 | SUPERSEDED (95.1%) / HISTORICAL-EXPERIMENT (27.1% bug) | `conformal_coverage.csv` (0.9515, 0.9304). In-domain coverage now 97.9% (width 2.288, n=3,462; PAPER_RESULTS s6). Bug narrative is a valid process story |
| 10 | CALCE Zero-Retrain Evaluation (~1400), CSV tables | XGB-fusion CALCE R2 0.304, Ridge-fusion 0.314 (RMSE 17.97/17.84); conformal 95.6% -> 6.1%, half-width 2.367 | SUPERSEDED | replaced by CALCE R2 0.749 +/- 0.012 (s1) and coverage 4.3% (s6). **CONFLICT C4** |
| 10 | same, "Does more data fix this?" metrics (~1420-1428) | 0.314 -> 0.669 (+0.355); coverage 6.1% -> 7.4% (+1.3 pts); in-domain 95.6% / 91.2%; R2 gap 0.603 -> 0.289 | HISTORICAL-EXPERIMENT (levels SUPERSEDED) | DEVELOPMENT_LOG 4056 (0.6690). Session 33; CALCE now 0.749 5-seed, 4.3% coverage. **CONFLICT C4** |
| 11 | Dashboard v1 (OC-SVM + Negative-RUL Bugs) (~1446) | 83.9% NASA flagged; 24.4% after cap; MIT 2.2-2.3%; RUL -15 cycles; AppTest SOH 71.9/71.8, 82.6/82.3, CALCE 66.0 vs 26.7 | HISTORICAL-EXPERIMENT | early dashboard bugs. In-app OC-SVM flag later replaced by the trust report (DEVELOPMENT_LOG "OC-SVM sanity check ... REPLACE") |
| 12 | Health Report Examples (Claude -> Gemini) (~1473) | 5 saved reports, HTTP 404/429 story | HISTORICAL-EXPERIMENT | `health_reports_examples.json`; no PAPER_RESULTS number |
| 13 | 3 Evaluation-Protocol Experiments (~1504) | early-life R2 negative; b2c24 -52.3; drop XGB-fusion dR2 -0.097, deep models <= 0.0002; bagging | SUPERSEDED (levels) | same CSVs as Model Validation tab (5-branch ablation R2 0.9169 vs 0.8202 confirmed). **CONFLICT C3** |
| 14 | RUL Conformal Coverage Investigation (~1526) | 88.9% stale -> 93.0%; 20 partitions 64.7-99.6%, mean 87.4%, std 10 pts | HISTORICAL-EXPERIMENT | `conformal_coverage.csv` 0.9304 confirmed. PAPER_RESULTS s4 shows RUL fails OOD (CALCE R2 -566) |
| 15 | MMD Domain Adaptation (~1545) | lambda 1.0 broken, 0.1 works; CALCE R2 0.304->0.337 (XGB), 0.314->0.347 (Ridge); coverage 6.1% -> 4.4%; half-width 2.367 -> 2.217 | HISTORICAL-EXPERIMENT | DEVELOPMENT_LOG 7702 (4.4%). Negative result, not promoted |
| 16 | Softmax-Normalized Adaptive Loss Weighting (~1566) | alpha/beta 0.993/1.007 -> 0.527/1.473; SOH R2 0.091; RUL 0.244 | HISTORICAL-EXPERIMENT | `joint_ablation.csv` (adaptive_softmax row) |
| 17 | LIME Cross-Validation of TreeSHAP (~1583) | 8/10 full 3/3 agreement; mean overlap 93.3%; XGB-base 100%; meta 86.7% | HISTORICAL-EXPERIMENT | `lime_shap_comparison.csv` |
| 18 | Knee-Point Detection (~1595) | mean offset 161.8 cycles (17.0%); 12.0 cycles excl. b3c0; b3c0 911, b2c24 55 | HISTORICAL-EXPERIMENT | `knee_point_detection.csv` (911 and 55 confirmed) |
| 19 | CNN-BiGRU as a 5th Base Learner (~1615) | R2 0.572, 4th of 5; ensemble 0.916947 -> 0.916884 | HISTORICAL-EXPERIMENT | `cnn_bigru_metrics.csv`, `drop_branch_ablation_5branch.csv` (0.91695 / 0.91688 confirmed) |
| 20 | Consolidated Convergence Comparison (~1628) | PiFormer best epoch 12/40; VLSTM epoch 34, val loss 0.269; spikes +0.0035, +0.0359 | HISTORICAL-EXPERIMENT | training curves |
| 21 | Domain-Shift-Aware Conformal Prediction (~1639) | AUC 1.0000 / 0.9021; in-domain coverage 94.6% -> 43.6%; fusion-only 69.6%; CALCE 4.4%; AUC 0.9021 -> 0.6936 (204-pool) | HISTORICAL-EXPERIMENT | `domain_shift_conformal_summary*.csv`; negative result |
| 22 | "Lean" Deployment vs. the Full 5-Branch Ensemble (~1667) | R2 0.91715 vs 0.91688; 52x (3.9 vs 201.9 ms); 2 vs 7 invocations; 1.1x smaller; "Recommendation: ship LEAN" | HISTORICAL-EXPERIMENT (NO-AUTHORITY) | `lean_vs_full_comparison.csv` confirmed. Not in PAPER_RESULTS |
| 23 | Bootstrap Confidence Intervals (~1677) | 2,000 resamples; XGB vs VLSTM CI [-0.054,+0.239] incl. 0 (32 batt), [+0.009,+0.077] (204); drop XGB-fusion [+0.005,+0.511]; Lean vs Full not significant | SUPERSEDED (framing) | PAPER_RESULTS s3 uses 1,000 battery-level resamples, seed 42 (in-domain R2 0.976 [0.945, 0.996], n=6). The 2,000-resample tables are session-20/33 |
| 24 | NASA EIS Features (~1718) | 97.6% NaN (26,360/26,996); 0 of 3 EIS features selected | HISTORICAL-EXPERIMENT | `bfa_selected_features_with_eis.txt` |
| 25 | Degradation-Mode Analysis (~1736) | B0018 LLI+LAM, MIT cells LAM | HISTORICAL-EXPERIMENT | qualitative, flagged not validated |
| 26 | Model Quantization / TinyML (~1753) | FP16 5.90 KB, INT8 6.12 KB, encoder 9.15 KB of 2,854.9 KB; 3.9-62x over 32-512 KB flash; R2 change <= 0.0003 | HISTORICAL-EXPERIMENT | `model_quantization_summary.csv`; not in PAPER_RESULTS |
| 27 | Second-Life Grading Classifier (~1769) | 98.75% of 5,208 cycles; B0018 last cycle 81.50% vs true 72.76% (8.7 pts); 41 of last 56 misgraded; 42.4% in second-life bracket | HISTORICAL-EXPERIMENT | `second_life_grading_*.csv` (first-pass 32-battery test set, n=5,208) |
| 28 | Sensor-Noise Robustness (~1789) | B0018 R2 0.576 -> 0.565 -> 0.525 -> 0.513 | HISTORICAL-EXPERIMENT | `sensor_noise_robustness_summary.csv` |
| 29 | B0018 Root-Cause Analysis (~1806) | 4.9x faster fade; AUC 1.0000; ICHV z=855, TEVI z=420; 18,341 MIT cycles; NASA 2.7% of cycles vs 11.5% of batteries; expanded 0.935 -> 0.895 -> 0.874 -> 0.874 | HISTORICAL-EXPERIMENT | `b0018_rootcause_*.csv`; expanded-pool B0018 is confounded by training exposure (the app says so) |
| 30 | Streaming Digital Twin (Online Learning) (~1857) | B0018 MAE 4.641->4.426, final err 8.74->6.89, half-width 8.16->7.57; b3c35 0.439->0.133, 1.22->0.03, 0.029->0.143; corrector blow-up 7,159,720,237,468.60; River 2.86 vs 4.43 | HISTORICAL-EXPERIMENT | `streaming_dt_*.csv`; DEVELOPMENT_LOG Stage 6.3 (River 2.858 / SGD 4.426). Not in PAPER_RESULTS |
| 31 | Adaptive Conformal Inference (ACI) (~1889) | B0018 82.0% -> 85.9%; b3c35 84.2% -> 87.2%; max half-width 9.060 -> 9.849 and 1.040 -> 4.547 (4.4x) | SUPERSEDED (framing) | PAPER_RESULTS s8: PID+scorecaster / PID reach 85-95% on built-in sets; standard ACI cannot fix severe shift. The ACI numbers are a single-battery session result |
| 32 | Dataset Expansion Phase 1: 32 -> 204 Batteries (~1915) | 23 NASA + 181 MIT of 219; bugs: 2177% SOH, 14 excluded (10 NASA + 4 MIT); BFA 7 -> 8 features, 3 shared; "32-battery lean pipeline remains the deployed default" | HISTORICAL-EXPERIMENT (deployment sentence SUPERSEDED) | DEVELOPMENT_LOG "Data-expansion pass" (line 9853). PAPER_RESULTS banner: deployed model is dataset-aware routing (extended XGBoost-fusion for CALCE/Oxford/HUST, base for XJTU and others). **CONFLICT C7** |
| 33 | The Reformulation Fix: Protocol-Invariant Features (~1989) | CALCE 0.568->0.740; Oxford -2.694->0.953; HUST -0.152->0.800; XJTU -1.059->-1.775; in-domain 0.974->0.973 | SUPERSEDED (after-values) | "before" CALCE 0.568 / Oxford -2.694 / HUST -0.152 match PAPER_RESULTS s11 "true deployed base"; XJTU before -1.059 vs -1.062 there. "After" replaced by 5-seed routed 0.749, 0.940, 0.795, -1.037 and in-domain 0.978 (s1). **CONFLICT C5** |
| 34 | Zero-Retrain Generalization: Four Datasets (~2044) | same table: CALCE 0.568/0.740, Oxford -2.694/0.953, HUST -0.152/0.800, XJTU -1.059/-1.775 | SUPERSEDED (after-values) | as s33. **CONFLICT C5**. "XJTU is the one unresolved weak point" is VERIFIED (-1.037 +/- 0.216, least seed-stable) |
| 35 | CALCE Conformal Coverage: Every Attempt (~2077) | "about 6-7%" static; 9 methods: MMD 4.4%, reweighted 4.4%, more data 7.4%, normalized 19.3%, CQR 21.3%, Jackknife+ 37.1%, KMM-CP 34.0%, CORAL 3.2%, rescaled Jackknife+ 82.0% (width ~29); reseed swing 2.69-5.47% | SUPERSEDED (incomplete) | Static level replaced by 4.3% (s6). PAPER_RESULTS s8: online PID 84.9%, PID+scorecaster 87.4%, nexCP 80.4/73.5% on CALCE (needs label feedback). The claim "none reached 90% at usable width" still holds, but the online result is not mentioned. **CONFLICT C8** |
| 36 | Comparing Against Published Methods (Severson, Attia) (~2122) | Severson/Attia: in-domain 0.57-0.58, CALCE 0.08, Oxford 0.17-0.20, HUST -1.4 to -1.8, XJTU -7.4 to -7.8; this model 0.973, 0.740, 0.953, 0.800, -1.775; fairer baseline in-domain 0.57 -> 0.91, Oxford 0.92, XJTU -1.47 | VERIFIED (literature columns) / SUPERSEDED (this-model column) | Literature columns match PAPER_RESULTS s11 (CALCE 0.077/0.078, Oxford 0.169/0.195, HUST -1.790/-1.437, XJTU -7.835/-7.410). This-model column replaced by 0.978/0.749/0.940/0.795/-1.037 (s1). The "XJTU exception (-1.78 vs -1.47)" is stale for deployed routing (-1.037 beats -1.47). PAPER_RESULTS s11 keeps the Oxford MAE-vs-R2 exception (MAE 12.51 vs 5.28/5.18, p=0.0078), not shown in app. **CONFLICTS C5, C6** |
| 37 | BatLiNet: An Experimental Deep-Learning Alternative (~2170) | initial R2 ~0.47 vs 0.97; single-cycle branch R2 ~0.52; after clamp CALCE 0.506 (vs 0.568), in-domain 0.55 -> 0.69 | HISTORICAL-EXPERIMENT | Stage 5; explicitly "never deployed"; the 0.568 comparator is the old base (PAPER_RESULTS s11 "true deployed base") |
| 38 | Does a Dataset's Similarity to Training Data Predict How It Responds? (~2224) | AUC CALCE 0.988, HUST 0.999, XJTU 0.9999, Oxford 0.9999; Oxford R2 -12.05 (NASA-heavy); 8-cell HUST vs 77-cell | HISTORICAL-EXPERIMENT (framing SUPERSEDED) | Session-level. PAPER_RESULTS s10 (item E, rerun): AUC has no reliable relationship with outcomes (Spearman -0.297 / 0.416 / 0.026, all CIs span zero). The app's "perfect correlation" (n=4) is an earlier, partly contradicted claim. **CONFLICT C10** |
| 39 | Three Speculative Architectures (~2287) | World Model ties/loses 4 of 5, wins Oxford; self-supervised 93.4%, helps 3 of 5, hurts 2; DeepONet R2 0.98 in-domain, hurts 3 of 5 | HISTORICAL-EXPERIMENT | Stage 7; none deployed |
| 40 | World Model (~2329) | pointer to its own tab | n/a | - |
| 41 | The Oxford Pattern (~2336) | Oxford -2.69 -> 0.95; R2 0.92 -> 0.99 with physics surrogate; 8 Oxford cells, 77 HUST | HISTORICAL-EXPERIMENT (Oxford 0.95 SUPERSEDED by 0.940 +/- 0.030) | Stage 7 closeout. n=8 Oxford and n=77 HUST match PAPER_RESULTS s3 |
| 42 | A Data-Quality Detective Story: B0044, B0045, B0053 (~2373) | one 0.00% reading at cycle 6; NASA ~1% of cycles vs >11% of batteries; B0053 final 0.00 Ah | HISTORICAL-EXPERIMENT | DEVELOPMENT_LOG sessions 33-46. s29 says NASA is 2.7% of training cycles, s42 says ~1%; different pools, not reconciled in the app |
| - | Session-history browser (raw DEVELOPMENT_LOG.md) (~2441) | raw log | HISTORICAL-EXPERIMENT | `render_session_history_browser` ~335 |

## D. Counts

Rows in tables A-C: Showcase 6, Validation 6, Archive 46 (42 sections, several with multiple rows, plus diagram row and history browser).

By primary status (one per row; rows with a split status are counted under the first-named status):

| Status | Showcase | Validation | Archive | Total |
|---|---|---|---|---|
| VERIFIED (matches PAPER_RESULTS) | 0 | 0 | 1 (s36 literature columns) | 1 |
| SUPERSEDED | 2 (R2 0.917; coverage 6.1%) | 2 (early-prediction full-life; drop-branch 4-branch) | 13 | 17 |
| INVALIDATED (embedding bug) | 0 | 0 | 0 | 0 |
| HISTORICAL-EXPERIMENT | 4 | 3 | 29 | 36 |
| not a results row / pointer | 0 | 1 | 3 | 4 |

## E. CONFLICTS (app number disagrees with PAPER_RESULTS.md)

| ID | App location | App says | PAPER_RESULTS says |
|---|---|---|---|
| C1 | Showcase "Ensemble R2" (~2565) | 0.917 | in-domain 5-seed R2 0.978 +/- 0.003 (s1); bootstrap 0.976 [0.945, 0.996] (s3) |
| C2 | Showcase "CALCE conformal coverage" (~2567) | 6.1% | 4.3% (s6, s8) |
| C3 | Model Validation "Early-prediction test" full-life rows (~905) and archive s3/s5/s13/s19/s22 | R2 0.917 (range 0.9066-0.9172) as the in-domain accuracy | 0.978 +/- 0.003 |
| C4 | Archive s10 (~1400-1428) | CALCE R2 0.314 (0.304 XGB), 0.669 expanded; coverage 6.1% / 7.4% | CALCE R2 0.749 +/- 0.012; coverage 4.3% |
| C5 | Archive s33, s34, s36 "after" / "this model" column | CALCE 0.740, Oxford 0.953, HUST 0.800, XJTU -1.775, in-domain 0.973 | 0.749, 0.940, 0.795, -1.037, 0.978 (s1; s11 "Routed (oracle)" reproduces 0.740/0.953/0.800/-1.037). The app's XJTU -1.775 is the extended model; deployed routing sends XJTU to the base model |
| C6 | Archive s36 (~2160) | "XJTU: this project loses to the fairer baseline (-1.78 vs -1.47)" | with deployed routing XJTU is -1.037 (s1), better than -1.47; s11 says the model is significantly better than the literature baselines on XJTU (8/13). The Oxford MAE exception (s11) is not shown |
| C7 | Archive s32 (~1915) | "original 32-battery lean pipeline remains the deployed default" | deployed model is dataset-aware routing (extended XGBoost-fusion for CALCE/Oxford/HUST, base elsewhere), retrained in Stage 4 (DEVELOPMENT_LOG ~8298) |
| C8 | Archive s35 (~2077) | CALCE static "about 6-7%"; best ever 82.0% (rescaled Jackknife+) | static 4.3%; online PID+scorecaster 87.4% and PID 84.9% on CALCE (s8, VERIFIED for built-in rows) |
| C9 | Archive s7 / s14 (RUL levels) | adaptive RUL R2 0.432, RUL coverage 93.0% | deployed retrained RUL in-domain R2 0.374 (RMSE 265.30), first-pass fusion joint 0.666, and RUL fails OOD (CALCE -566.35) (s4) |
| C10 | Archive s38 (~2237) | domain-classifier AUC rank "perfectly" predicts direction of change (n=4) | s10: no reliable relationship; Spearman -0.297 / 0.416 / 0.026, CIs span zero |

Not conflicts, but unverifiable against PAPER_RESULTS: 52x speedup, B0018 2.86 pp / 4.64 pp / 4.43 pp, and the quantization, second-life, knee-point, LIME, SHAP, ACI and streaming numbers. These are session results (DEVELOPMENT_LOG / outputs CSVs); where a CSV exists they were checked against it above (early-prediction, drop-branch, bagging, ensemble, fusion, joint ablation, conformal, CALCE conformal, lean-vs-full, knee-point all matched).

## F. Banner sentences (machine-readable)

Keys are `function_name:heading text`. Heading text is the expander/section title without the keycap-number emoji (archive section numbers 1-42 are in table C).

```json
{
"render_showcase_tab:Digital Twin Showcase": "Status: replay of session 28-29 streaming runs; headline R2 0.917 and CALCE 6.1% are first-pass (final: R2 0.978, CALCE coverage 4.3%, PAPER_RESULTS).",
"render_evaluation_protocol_section:Evaluation protocol: early-prediction / drop-branch / bagging experiments": "Status: first-pass 32-battery, 6-battery, single-seed experiments; in-domain R2 0.917 is superseded by 0.978 +/- 0.003 (PAPER_RESULTS s1).",
"render_battery_comparison_section:Battery comparison mode": "Status: live inference, no stored results; intervals are an in-domain-only guarantee (static coverage 1.5-39.6% out of domain, PAPER_RESULTS s6).",
"render_full_results_archive_tab:Full Results Archive": "Status: session-by-session archive of first-pass 32-battery results; final numbers are the 5-seed tables in PAPER_RESULTS.md (2026-09-30).",
"render_full_results_archive_tab:BFA Feature Selection": "Status: historical 32-battery run (7 features); the deployed set is the later reformulated 8-feature set (DEVELOPMENT_LOG Stage 4).",
"render_full_results_archive_tab:SOH Fade Examples & ICA/DV/DC Example": "Status: illustrative figures from the first-pass pipeline; not a results claim.",
"render_full_results_archive_tab:Base Learner Training (incl. the CNN-LSTM root-cause fix)": "Status: historical first-pass 32-battery results (XGBoost R2 0.907); the final in-domain 5-seed R2 is 0.978 (PAPER_RESULTS s1).",
"render_full_results_archive_tab:Stacking Ensemble": "Status: historical first-pass 32-battery ensemble (R2 0.906); superseded by the 5-seed XGBoost-fusion R2 0.978 (PAPER_RESULTS s1).",
"render_full_results_archive_tab:Feature Fusion": "Status: first-pass R2 0.917 with fusion; superseded by the 5-seed in-domain R2 0.978 +/- 0.003 (PAPER_RESULTS s1).",
"render_full_results_archive_tab:Physics-Informed Loss Experiment": "Status: historical negative result, not adopted into the final pipeline.",
"render_full_results_archive_tab:Joint SOH+RUL Ablation (incl. the log_sigma Clamp Fix)": "Status: historical 32-battery ablation; deployed RUL is R2 0.374 in-domain and fails out of domain (PAPER_RESULTS s4).",
"render_full_results_archive_tab:SHAP Explainability": "Status: historical attribution on the first-pass 7-feature model; not re-run for the final 8-feature deployed model.",
"render_full_results_archive_tab:Split-Conformal Prediction (incl. the 27.1% calibration bug)": "Status: first-pass in-domain coverage 95.1%; final in-domain coverage is 97.9% (PAPER_RESULTS s6); out-of-domain coverage is far lower.",
"render_full_results_archive_tab:CALCE Zero-Retrain Evaluation": "Status: first-pass CALCE R2 0.314 and coverage 6.1% are SUPERSEDED by R2 0.749 and coverage 4.3% (PAPER_RESULTS s1, s6).",
"render_full_results_archive_tab:Dashboard v1 (OC-SVM + Negative-RUL Bugs)": "Status: historical bug-fix narrative; the OC-SVM flag was later replaced by the trust report.",
"render_full_results_archive_tab:Health Report Examples (Claude → Gemini)": "Status: saved example reports from sessions 6-8; illustrative, not a results table.",
"render_full_results_archive_tab:3 Evaluation-Protocol Experiments": "Status: first-pass 32-battery experiments (R2 0.917); superseded by the five-seed tables and not re-run for the final model (PAPER_RESULTS s1).",
"render_full_results_archive_tab:RUL Conformal Coverage Investigation": "Status: historical 6-battery RUL coverage study (93.0%); RUL fails out of domain (CALCE R2 -566, PAPER_RESULTS s4).",
"render_full_results_archive_tab:MMD Domain Adaptation": "Status: historical negative result (CALCE coverage 6.1% to 4.4%); final static CALCE coverage is 4.3% (PAPER_RESULTS s6).",
"render_full_results_archive_tab:Softmax-Normalized Adaptive Loss Weighting": "Status: historical negative result on the first-pass joint model; not adopted.",
"render_full_results_archive_tab:LIME Cross-Validation of TreeSHAP": "Status: historical first-pass check; not re-run for the final model.",
"render_full_results_archive_tab:Knee-Point Detection": "Status: historical 6-battery analysis; not in PAPER_RESULTS.",
"render_full_results_archive_tab:CNN-BiGRU as a 5th Base Learner": "Status: historical first-pass ablation (R2 0.9169 vs 0.9168); the final model is XGBoost-fusion (PAPER_RESULTS s1).",
"render_full_results_archive_tab:Consolidated Convergence Comparison": "Status: historical training-curve comparison for the first-pass deep models.",
"render_full_results_archive_tab:Domain-Shift-Aware Conformal Prediction": "Status: historical negative result; online PID+scorecaster reaches 71-95% coverage on all 13 sets (PAPER_RESULTS s8).",
"render_full_results_archive_tab:\"Lean\" Deployment vs. the Full 5-Branch Ensemble": "Status: session-20 first-pass comparison (52x faster, R2 0.917); not in PAPER_RESULTS, whose final model is 5-seed XGBoost-fusion.",
"render_full_results_archive_tab:Bootstrap Confidence Intervals": "Status: session-20/33 bootstrap; final battery-level CIs are in PAPER_RESULTS s3 (in-domain R2 0.976 [0.945, 0.996]).",
"render_full_results_archive_tab:NASA EIS Features as Candidate Health Indicators": "Status: historical negative result (EIS is NASA-only, 97.6% missing); not in PAPER_RESULTS.",
"render_full_results_archive_tab:Degradation-Mode Analysis (dV/dQ Peak-Tracking)": "Status: historical qualitative analysis, not a validated LLI/LAM decomposition.",
"render_full_results_archive_tab:Model Quantization / TinyML Feasibility": "Status: historical feasibility check on the lean 32-battery pipeline; not in PAPER_RESULTS.",
"render_full_results_archive_tab:Second-Life Grading Classifier": "Status: historical 6-battery result on the first-pass model (98.75% agreement); not in PAPER_RESULTS.",
"render_full_results_archive_tab:Sensor-Noise Robustness": "Status: historical first-pass stress test on the 6 test batteries; not in PAPER_RESULTS.",
"render_full_results_archive_tab:B0018 Root-Cause Analysis": "Status: historical root-cause analysis; 204-pool B0018 numbers are confounded by training exposure.",
"render_full_results_archive_tab:Streaming Digital Twin (Online Learning)": "Status: session 28-29 replay results (B0018 and b3c35 only); not in PAPER_RESULTS, see its s8 for the online-conformal result.",
"render_full_results_archive_tab:Adaptive Conformal Inference (ACI)": "Status: historical single-battery ACI result (82-87%); online PID+scorecaster supersedes it at 71-95% across 13 datasets (PAPER_RESULTS s8).",
"render_full_results_archive_tab:Dataset Expansion Phase 1: 32 → 204 Batteries": "Status: session-33 research finding, not the deployed model; deployed is dataset-aware routing (PAPER_RESULTS banner, DEVELOPMENT_LOG Stage 4).",
"render_full_results_archive_tab:The Reformulation Fix: Protocol-Invariant Features": "Status: single-seed numbers SUPERSEDED by 5-seed CALCE 0.749, Oxford 0.940, HUST 0.795, XJTU -1.037, in-domain 0.978 (PAPER_RESULTS s1).",
"render_full_results_archive_tab:Zero-Retrain Generalization: Four Datasets, Never Trained On": "Status: single-seed table SUPERSEDED by 5-seed 0.749 / 0.940 / 0.795 / -1.037 (PAPER_RESULTS s1); XJTU -1.775 was the extended model.",
"render_full_results_archive_tab:CALCE Conformal Coverage: Every Attempt, Consolidated": "Status: static CALCE coverage is now 4.3%; online PID+scorecaster reaches 87.4% with label feedback (PAPER_RESULTS s6, s8).",
"render_full_results_archive_tab:Comparing Against Published Methods (Severson, Attia)": "Status: literature columns VERIFIED; this-model column SUPERSEDED by 5-seed numbers, XJTU now -1.037 (PAPER_RESULTS s1, s11).",
"render_full_results_archive_tab:BatLiNet: An Experimental Deep-Learning Alternative": "Status: historical experiment, never deployed; not in PAPER_RESULTS.",
"render_full_results_archive_tab:Does a Dataset's Similarity to Training Data Predict How It Responds to Change?": "Status: early n=4 finding; rerun shift diagnostics show no reliable AUC-to-outcome relationship (PAPER_RESULTS s10).",
"render_full_results_archive_tab:Three Speculative Architectures, Reported Honestly": "Status: historical Stage-7 negative-to-mixed results; none deployed or in PAPER_RESULTS.",
"render_full_results_archive_tab:World Model: Forecasting a Future, Not Just a Point": "Status: pointer to the World Model tab; historical experiment, not deployed.",
"render_full_results_archive_tab:The Oxford Pattern: One Dataset, Four Independent Surprises": "Status: historical analysis; Oxford is now 0.940 +/- 0.030 five-seed, n=8 cells (PAPER_RESULTS s1).",
"render_full_results_archive_tab:A Data-Quality Detective Story: B0044, B0045, and B0053": "Status: historical data-quality investigation of three NASA batteries; not a headline result.",
"render_full_results_archive_tab:Session-history browser (raw DEVELOPMENT_LOG.md, all sections)": "Status: raw development log; contains superseded numbers, PAPER_RESULTS.md (2026-09-30) is the authority."
}
```
