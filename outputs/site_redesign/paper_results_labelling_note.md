# PAPER_RESULTS.md labelling note (2026-10-01)

Only `PAPER_RESULTS.md` was edited (nothing committed or pushed). No table number was changed.

## What was added
- Every table (15 existing + 1 new headline-status table in the "Final list" section = 16) now has: a bold "Table status" line directly above it,
  a per-row `Status` column (VERIFIED / SUPERSEDED, taken from the existing 2026-09-30 section tags; mixed tables are labelled per row),
  a `Source file` column (names relative to `outputs/`), and a "Source: outputs/..." line under it.
- Legend at the top: INVALIDATED and PAUSED added as labels; none currently assigned (rerun queue finished, old mixed-encoder Phase 2B table kept as SUPERSEDED, as the file already said).
- "Final list" section: every bullet already stated a status; added a Toolkit-pass (3a/2B/2C) bullet and a status + source table for all headlines.

## Verification (numbers checked against CSVs)
About 260 cells checked by script, 0 mismatches except the one CONFLICT below: all of section 1 (R2/RMSE/MAE mean and std, n; 14 rows), section 2 sample (12 cells),
section 3 sample (12), section 4 (3 + checkB), section 5 (2), section 6 coverage sample (6) and k=0 / k=50 means (12.9 / 15.2), section 8 all 52 PID/PID+sc/nexCP cells,
section 9 LODO R2 (13) plus family/NASA+MIT-only values, section 10 (Spearman, risk-coverage means 10.11/10.03/10.56), section 11 (trivial/Severson/Attia/oracle/best-LODO/audit columns),
section 12 (width % of range, 7/13 and 88/63/22/3 life-stage counts), Phase 3a (all rows), Phase 2B AFTER and BEFORE summary rows and per-source old table,
Phase 2C pooled means (budgets 5-40, both policies) and budget-40 per-source values.

## CONFLICT (number left as is)
- Phase 2B OLD (mixed-encoder) per-source table, row `rwth`, "Federated (best of 3 weighting schemes)": file says 0.482; `toolkit_phase2b_federated_results.MIXED_ENCODER_CONTAMINATED.csv`
  best of the three schemes = 0.540. The table is SUPERSEDED/do-not-cite, so no headline is affected. Flagged in the table's source cell and under it.

## Source not recorded (1)
- Section 4 row "In-domain (TEST) - first-pass joint model (reference)", RUL R2 0.666: only in `DEVELOPMENT_LOG.md` (session 41 Part B, line ~8376); no outputs/ file holds it.
  (Its "mean 390.6, std 292.9" scale is in `finalpass_checkB_rul_diagnostic.csv`. The deployed 0.374/265.30 row is sourced: `stage4_step2b_summary.csv`.)

## Caveats found (not number conflicts)
- `finalpass_item5e_conformal_consolidated.csv` still holds PRE-rerun BatteryLife coverage (the "before" values); corrected section 6/8/12 static values come only from `finalpass_item3_coverage_pivot.csv` column 0.
  Likewise the static column of `finalpass3_check2_labelfree_vs_online.csv` and the deployed/oracle columns of `finalpass2_itemC_baseline_table.csv` (BatteryLife rows) are stale; the file already said so.
- Section 11 "Best LODO" shows the PLAIN stanford/stanford_2 values (0.997/0.990, matching `finalpass2_itemB_lodo_results.csv`); the paper should use the sibling-holdout 0.919/0.895 (section 9). Noted under the table, cell untouched.
- Section 7 has no table (text only); its per-item statuses were already present.
- "Before" values are traceable only via git commit c56ddcb (and `toolkit_rerun_before_after_itemB.csv` for item B); no separate before-CSV exists for the other tables.
