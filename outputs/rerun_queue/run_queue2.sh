#!/bin/bash
# Queue order per user decision 2026-09-30: paper-deciding results first.
cd /c/Users/manoj/OneDrive/Desktop/battery-prognostics
PY=.venv/Scripts/python.exe
RUNNER=src/run_with_corrected_batterylife_embeddings.py
SUMMARY=outputs/rerun_queue/summary_resume.txt
run () { local name=$1; shift; local t0=$(date +%s)
  echo "[queue] START $name $(date +%H:%M:%S)" | tee -a $SUMMARY
  "$@" > outputs/rerun_queue/$name.log 2>&1; local rc=$?; local t1=$(date +%s)
  echo "[queue] END   $name rc=$rc secs=$((t1-t0)) $(date +%H:%M:%S)" | tee -a $SUMMARY; }
run lodo_check1  $PY src/rerun_lodo_family_holdout_corrected.py                 # family-holdout (check1 protocol)
run phase2c      $PY $RUNNER src/run_toolkit_phase2c_label_efficient.py
run phase3a      $PY $RUNNER src/run_toolkit_phase3a_min_checkpoints.py
run item1        $PY $RUNNER src/run_finalpass_item1_labelfree_routing.py
run item2        $PY $RUNNER src/run_finalpass_item2_cycleidx_ablation.py
run item3        $PY $RUNNER src/run_finalpass_item3_fewshot_conformal.py
run item5a       $PY $RUNNER src/run_finalpass_item5a_rigor_seeds_ci.py
run item5d       $PY $RUNNER src/run_finalpass_item5d_batterylife_benchmark.py
run partB7       $PY $RUNNER src/run_partB_item7_zeroretrain_eval.py
run partB8       $PY $RUNNER src/run_partB_item8_external_baseline_comparison.py   # reads partB7's output
run partB9       $PY $RUNNER src/run_partB_item9_pool_expansion_retrain.py
run check2       $PY $RUNNER src/run_finalpass3_check2_online_conformal_audit.py
run itemE        $PY $RUNNER src/run_finalpass2_itemE_shift_diagnostics.py
run itemC        $PY $RUNNER src/run_finalpass2_itemC_baseline_table.py
echo "[queue2] ALL DONE $(date +%H:%M:%S)" | tee -a $SUMMARY
