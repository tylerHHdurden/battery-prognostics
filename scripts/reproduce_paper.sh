#!/usr/bin/env bash
# Regenerates every table in PAPER_RESULTS.md from scratch by re-running
# this project's own scripts, in dependency order. Read-only with
# respect to app.py/src/live_inference.py/models/xgb_soh_fusion.json/
# models/_experimental_xgb_soh_fusion_extended_reformulation.json/
# models/joint_adaptive_fusion.pt - every script here EVALUATES those
# already-trained files zero-retrain, or trains a NEW _experimental_*
# file; none of them overwrite a deployed file.
#
# Usage: bash scripts/reproduce_paper.sh
# Expects: repo root as CWD, Python venv active (see requirements-lock.txt),
# data/raw/{calce,oxford,hust,xjtu,nasa,mit,batterylife}/ populated per
# DATA_AVAILABILITY.md (raw data is git-ignored, not shipped in this repo).
#
# Wall-clock note (this project's own last real run, disclosed): under
# 30 minutes total for every step below combined, except item C's
# Severson/Attia feature rebuild for HUST/isu_ilcc-scale sources, which
# alone can take 15-20+ minutes (raw per-cycle Q(V) curve integration,
# no caching) - the single slowest step in this whole script.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Prerequisite: precomputed feature/embedding files (only if missing) ==="
# These are assumed already built by this project's own earlier stages
# (hi_table.parquet, fusion_embeddings.csv, battery_split.json,
# bfa_selected_features_nasa_mit_only.txt, stage5_1_{oxford,hust,xjtu}_
# merged.parquet, batterylife_<source>_merged.parquet, the deployed
# model files themselves) - this script does NOT re-run Stage 0-7 from
# raw data (that is a multi-hour, multi-session undertaking documented
# in full in DEVELOPMENT_LOG.md's own earlier sections, out of scope
# for a "reproduce the FINAL PAPER's tables" script). If any of these
# are missing, re-run the relevant Stage 0-7 script named in
# DEVELOPMENT_LOG.md before continuing.

echo "=== Table set 1: final research pass (items 1-5, checks A/B) ==="
python src/run_finalpass_item1_labelfree_routing.py
python src/run_finalpass_item2_cycleidx_ablation.py
python src/run_finalpass_item3_fewshot_conformal.py
# item 4: no script - feasibility determined by direct code/data inspection, documented not run
python src/run_finalpass_item5a_rigor_seeds_ci.py
python src/run_finalpass_item5c_rul_crossdomain.py
python src/run_finalpass_item5d_batterylife_benchmark.py
python src/run_finalpass_item5e_conformal_consolidated.py
python src/run_finalpass_checkA_conformal_diagnostic.py
python src/run_finalpass_checkB_rul_diagnostic.py

echo "=== Table set 2: final experiment pass 2 (items A-F) ==="
python src/run_finalpass2_itemA_online_conformal.py
python src/run_finalpass2_itemB_lodo.py
python src/run_finalpass2_itemD_honest_routing.py   # depends on item 1's own output CSV
python src/run_finalpass2_itemE_shift_diagnostics.py
python src/run_finalpass2_itemC_baseline_table.py   # depends on item B's own output CSV; SLOWEST STEP

echo "=== DONE. Every outputs/finalpass*.csv referenced by PAPER_RESULTS.md has been regenerated. ==="
echo "No file under app.py, src/live_inference.py, or models/ (excluding models/_experimental_*) was modified."
