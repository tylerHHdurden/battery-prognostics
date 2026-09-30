# Extension sources selectable in "Browse existing battery"

## What changed (working tree only, nothing committed)
- `src/live_inference.py`: added `EXTENSION_SOURCES` (ul_pur, hnei, snl, mich, mich_exp, rwth, stanford, stanford_2, isu_ilcc, tongji) and
  extended `PRECOMPUTED_HELDOUT_PARQUETS` with `batterylife_<src>_merged.parquet`. New helper `_precomputed_df(dataset)` reads the parquet and,
  for extension sources only, prefixes battery_id as `<src>::<raw>` (the id format used by `fusion_embeddings_multisource.csv` and
  `models/_builtin_battery_trust.csv`; parquets hold the raw id). The three existing `pd.read_parquet(...PRECOMPUTED_HELDOUT_PARQUETS[dataset])`
  calls now go through it; Oxford/HUST/XJTU are unchanged (helper is a plain read for them). The existing `_use_candidate` branch is what scores these
  sources (they are not in `SIX_APP_DATASETS`). Encoder provenance assert passes (parquets are old_v1, same as xgb_soh_fusion.json; the candidate
  model uses the candidate-encoder lookup CSV).
- `app.py`: `_dataset_options` (sidebar and A/B comparison picker) += `EXTENSION_SOURCES`; sidebar caption for extension sources:
  "Extension source: predicted by the 16-source candidate model, which saw batteries of this source in training; leave-one-source-out transfer to a
  truly unseen source is much weaker (see the trust message)." It replaces the "genuine zero-retrain evaluation" caption for these sources only
  (that claim would be false for them).
- Checks added: `outputs/site_redesign/check_extension_sources.py` (AppTest) and `check_extension_direct.py` (direct inference + passport).

## Per-source results
AppTest (default battery, last cycle): 0 exceptions, Predicted SOH shown, result card present, extension caption shown, RUL "not available" for all 10.
Direct check over ALL batteries (last cycle): every battery has a trust row and a candidate-lookup row (0 missing), model_variant =
multisource_candidate everywhere, RUL never shown, passport builds (JSON serialises; model = multi-source candidate; RUL not shown).

| source | batteries | flagged (last cycle) | nearest source (first battery) | MAE at last cycle vs true SOH (pts) |
|---|---|---|---|---|
| ul_pur | 10 | 0/10 | ul_pur | 1.89 |
| hnei | 14 | 0/14 | hnei | 1.98 |
| snl | 55 | 18/55 | snl | 3.67 |
| mich | 40 | 1/40 | mich | 3.54 |
| mich_exp | 18 | 0/18 | mich_exp | 5.37 |
| rwth | 10 | 1/10 | rwth | 1.33 |
| stanford | 6 | 0/6 | stanford | 3.38 |
| stanford_2 | 8 | 0/8 | stanford (not stanford_2) | 3.90 |
| isu_ilcc | 9 | 9/9 | isu_ilcc | 0.89 |
| tongji | 130 | 3/130 | tongji | 0.81 |

The MAE column is on batteries the candidate saw in training (in-sample for most), so it is NOT a generalisation figure. Nearest source is almost
always the battery's own source, so the RUL rule (nearest source NASA/MIT) keeps RUL hidden; no green/"trusted" wording (existing banner unchanged:
neutral "no distribution shift detected" note or amber warning).

## Regression
- `pytest tests -q`: 25 passed.
- `ood_matrix_test.py agentcheck` vs `toolkit_ood_matrix_routed.json`: soh_metric and rul_metric identical for all 7 cases
  (NASA 71.7, MIT 82.5, CALCE 54.0, Oxford 73.5, HUST 79.3, XJTU 95.2, HNEI upload 97.3).

## Caveats / not wired
- All 10 sources wired; none skipped.
- Streaming digital twin / comparison tab for these sources use `load_precomputed_battery_series` with the old-encoder fusion columns and the base
  model (existing twin behaviour), not the candidate. Not changed.
- The sidebar "informational" no-temperature note is based on dataset != HUST, as before; it may be inaccurate for some extension sources (informational only).
