#!/bin/bash
# Dataset Expansion Phase 1: chains Step 2 (BFA already run separately)
# through Step 3 (all retraining) as one unattended background run.
# `set -e` so a failure in any stage stops the chain rather than letting
# a downstream script silently consume a broken/missing upstream file.
set -e
cd "/c/Users/manoj/OneDrive/Desktop/battery-prognostics"

echo "=== [chain] $(date) starting train_xgboost_expanded.py ==="
python -u src/train_xgboost_expanded.py

echo "=== [chain] $(date) starting train_deep_models_expanded.py (the long one) ==="
python -u src/train_deep_models_expanded.py

echo "=== [chain] $(date) starting train_fusion_encoder_expanded.py ==="
python -u src/train_fusion_encoder_expanded.py

echo "=== [chain] $(date) starting train_xgboost_fusion_expanded.py ==="
python -u src/train_xgboost_fusion_expanded.py

echo "=== [chain] $(date) starting train_ensemble_fusion_expanded.py ==="
python -u src/train_ensemble_fusion_expanded.py

echo "=== [chain] $(date) starting train_cnn_bigru_expanded.py ==="
python -u src/train_cnn_bigru_expanded.py

echo "=== [chain] $(date) starting run_drop_branch_ablation_5branch_expanded.py ==="
python -u src/run_drop_branch_ablation_5branch_expanded.py

echo "=== [chain] $(date) starting run_calce_zero_retrain_eval_expanded.py ==="
python -u src/run_calce_zero_retrain_eval_expanded.py

echo "=== [chain] $(date) starting run_bootstrap_significance_expanded.py ==="
python -u src/run_bootstrap_significance_expanded.py

echo "=== [chain] $(date) starting run_domain_classifier_sanity_check_expanded.py ==="
python -u src/run_domain_classifier_sanity_check_expanded.py

echo "=== [chain] $(date) starting run_second_life_grading_expanded.py ==="
python -u src/run_second_life_grading_expanded.py

echo "=== [chain] $(date) starting run_sensor_noise_robustness_expanded.py ==="
python -u src/run_sensor_noise_robustness_expanded.py

echo "=== [chain] $(date) ALL STAGES COMPLETE ==="
