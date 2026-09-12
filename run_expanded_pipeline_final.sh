#!/bin/bash
# Final remaining stages after 2 prior interruptions. Everything through
# the (corrected) ensemble is done and confirmed safe on disk:
#   XGBoost, VLSTM, CNN-LSTM (corrected), PiFormer, fusion encoder,
#   XGBoost-fusion, ensemble (re-run with corrected CNN-LSTM data).
# Only CNN-BiGRU (now genuinely resumable, verified bit-for-bit via a
# synthetic smoke test) and the 6 Step-4 evaluation scripts remain.
set -e
cd "/c/Users/manoj/OneDrive/Desktop/battery-prognostics"

echo "=== [final] $(date) starting train_cnn_bigru_expanded.py (now checkpoint-resumable per-epoch) ==="
python -u src/train_cnn_bigru_expanded.py

echo "=== [final] $(date) starting run_drop_branch_ablation_5branch_expanded.py ==="
python -u src/run_drop_branch_ablation_5branch_expanded.py

echo "=== [final] $(date) starting run_calce_zero_retrain_eval_expanded.py ==="
python -u src/run_calce_zero_retrain_eval_expanded.py

echo "=== [final] $(date) starting run_bootstrap_significance_expanded.py ==="
python -u src/run_bootstrap_significance_expanded.py

echo "=== [final] $(date) starting run_domain_classifier_sanity_check_expanded.py ==="
python -u src/run_domain_classifier_sanity_check_expanded.py

echo "=== [final] $(date) starting run_second_life_grading_expanded.py ==="
python -u src/run_second_life_grading_expanded.py

echo "=== [final] $(date) starting run_sensor_noise_robustness_expanded.py ==="
python -u src/run_sensor_noise_robustness_expanded.py

echo "=== [final] $(date) ALL STAGES COMPLETE ==="
