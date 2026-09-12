#!/bin/bash
# Resume of run_expanded_pipeline_chain.sh after the previous session was
# torn down mid-run. train_xgboost_expanded.py already completed and its
# outputs (models/xgb_soh_expanded.json, predictions/xgb_expanded_*.csv)
# are confirmed on disk and untouched - skipped here to avoid redoing
# already-good work. train_deep_models_expanded.py got to VLSTM epoch 23/40
# with nothing checkpointed (no partial-epoch save in that script), so it
# restarts from epoch 0 - unavoidable, not a redundant re-run of something
# already saved.
set -e
cd "/c/Users/manoj/OneDrive/Desktop/battery-prognostics"

echo "=== [resume] $(date) starting train_deep_models_expanded.py (restart from epoch 0 - previous run was killed at VLSTM epoch 23/40 with no checkpoint) ==="
python -u src/train_deep_models_expanded.py

echo "=== [resume] $(date) starting train_fusion_encoder_expanded.py ==="
python -u src/train_fusion_encoder_expanded.py

echo "=== [resume] $(date) starting train_xgboost_fusion_expanded.py ==="
python -u src/train_xgboost_fusion_expanded.py

echo "=== [resume] $(date) starting train_ensemble_fusion_expanded.py ==="
python -u src/train_ensemble_fusion_expanded.py

echo "=== [resume] $(date) starting train_cnn_bigru_expanded.py ==="
python -u src/train_cnn_bigru_expanded.py

echo "=== [resume] $(date) starting run_drop_branch_ablation_5branch_expanded.py ==="
python -u src/run_drop_branch_ablation_5branch_expanded.py

echo "=== [resume] $(date) starting run_calce_zero_retrain_eval_expanded.py ==="
python -u src/run_calce_zero_retrain_eval_expanded.py

echo "=== [resume] $(date) starting run_bootstrap_significance_expanded.py ==="
python -u src/run_bootstrap_significance_expanded.py

echo "=== [resume] $(date) starting run_domain_classifier_sanity_check_expanded.py ==="
python -u src/run_domain_classifier_sanity_check_expanded.py

echo "=== [resume] $(date) starting run_second_life_grading_expanded.py ==="
python -u src/run_second_life_grading_expanded.py

echo "=== [resume] $(date) starting run_sensor_noise_robustness_expanded.py ==="
python -u src/run_sensor_noise_robustness_expanded.py

echo "=== [resume] $(date) ALL STAGES COMPLETE ==="
