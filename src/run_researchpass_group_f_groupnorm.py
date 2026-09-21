"""
Research pass Group F, item 17: GroupNorm swap for CNN-LSTM's
BatchNorm1d layers. "Literature is genuinely split... a negative
result here is a real, expected, valid outcome" - task's own framing.

BASELINE (already on record, not re-run): CNNLSTM (BatchNorm) on the
EXPANDED NASA+MIT pool, data/processed/predictions/deep_models_expanded_metrics.csv:
TEST RMSE=1.3003 MAE=0.7450 R2=0.9666.

Reuses the EXACT same expanded-pool tensor cache, battery split
(battery_split_expanded.json), and channel-norm stats
(channel_norm_stats_expanded.json) as that baseline run, so the only
difference between this run and the baseline is BatchNorm1d ->
GroupNorm(num_groups=4) inside MultiKernelBranch (branch_channels=8,
so 4 groups of 2 channels each - untuned, disclosed, the most
"vanilla" divisor choice rather than a searched hyperparameter).
Same 40 epochs/batch=64/lr=1e-3/patience=8 training loop
(train_one_model, imported unchanged from train_deep_models.py).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
from sequence_features import CHANNEL_NAMES, apply_channel_norm
from train_deep_models import make_xy, train_one_model
from train_deep_models_expanded import load_all_battery_tensors_expanded, predict

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"

BASELINE = {"rmse": 1.3002859094637986, "mae": 0.7450214624404907, "r2": 0.9665936231613159}

torch.manual_seed(42)
np.random.seed(42)


class MultiKernelBranchGN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, num_groups: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.gn = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        return torch.relu(self.gn(self.conv(x)))


class CNNLSTM_GroupNorm(nn.Module):
    """Identical to models.cnn_lstm.CNNLSTM except BatchNorm1d -> GroupNorm."""
    def __init__(self, in_channels: int = 6, branch_channels: int = 8,
                 lstm_hidden: int = 32, n_targets: int = 1,
                 kernel_sizes=(3, 5, 7, 11)):
        super().__init__()
        self.branches = nn.ModuleList([
            MultiKernelBranchGN(in_channels, branch_channels, k) for k in kernel_sizes
        ])
        concat_channels = branch_channels * len(kernel_sizes)
        self.lstm = nn.LSTM(concat_channels, lstm_hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 16), nn.ReLU(), nn.Linear(16, n_targets)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        branch_outs = [b(x) for b in self.branches]
        merged = torch.cat(branch_outs, dim=1)
        merged = merged.transpose(1, 2)
        _, (h_n, _) = self.lstm(merged)
        return self.head(h_n[-1])


def main():
    t0 = time.time()
    print("=== Research pass Group F, item 17: GroupNorm swap for CNN-LSTM ===")
    battery_data = load_all_battery_tensors_expanded()
    print(f"[groupnorm] loaded {len(battery_data)} batteries in {time.time()-t0:.1f}s")

    split = json.loads((PROC_DIR / "battery_split_expanded.json").read_text())
    train_ids = [b for b in split["train_ids"] if b in battery_data]
    test_ids = [b for b in split["test_ids"] if b in battery_data]
    n_val_batteries = max(1, len(train_ids) // 5)
    val_ids = sorted(train_ids)[-n_val_batteries:]
    fit_ids = [b for b in train_ids if b not in val_ids]

    X_fit, y_fit, _, _, _, _ = make_xy(battery_data, fit_ids)
    X_val, y_val, _, _, _, _ = make_xy(battery_data, val_ids)
    X_test, y_test, _, ds_test, bid_test, cyc_test = make_xy(battery_data, test_ids)
    print(f"[groupnorm] fit cycles={len(X_fit)}, val cycles={len(X_val)}, test cycles={len(X_test)}")

    norm_stats = json.loads((PROC_DIR / "channel_norm_stats_expanded.json").read_text())
    print("[groupnorm] reusing existing channel_norm_stats_expanded.json (identical to baseline run)")
    X_fit = apply_channel_norm(X_fit, norm_stats)
    X_val = apply_channel_norm(X_val, norm_stats)
    X_test = apply_channel_norm(X_test, norm_stats)

    model = CNNLSTM_GroupNorm()
    t1 = time.time()
    model, hist = train_one_model("CNNLSTM-GroupNorm", model, X_fit, y_fit, X_val, y_val)
    train_time = time.time() - t1
    print(f"[groupnorm] CNNLSTM-GroupNorm trained in {train_time:.1f}s")
    torch.save(model.state_dict(), ROOT / "models" / "_experimental_cnn_lstm_groupnorm_expanded.pt")
    pd.DataFrame(hist).to_csv(PROC_DIR / "predictions" / "_experimental_cnn_lstm_groupnorm_history.csv", index=False)

    pred_test = predict(model, X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred_test)))
    mae = float(mean_absolute_error(y_test, pred_test))
    r2 = float(r2_score(y_test, pred_test))
    print(f"[groupnorm] TEST RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")
    print(f"[groupnorm] BASELINE (BatchNorm): RMSE={BASELINE['rmse']:.4f} MAE={BASELINE['mae']:.4f} R2={BASELINE['r2']:.4f}")

    verdict = "WIN" if r2 > BASELINE["r2"] else "LOSS"
    print(f"\n[groupnorm] VERDICT: {verdict} (GroupNorm R2={r2:.4f} vs. BatchNorm baseline R2={BASELINE['r2']:.4f})")

    results = pd.DataFrame([
        {"variant": "BatchNorm (DEPLOYED baseline, on record)", "rmse": BASELINE["rmse"], "mae": BASELINE["mae"], "r2": BASELINE["r2"]},
        {"variant": "GroupNorm(num_groups=4) (THIS ITEM)", "rmse": rmse, "mae": mae, "r2": r2},
    ])
    results.to_csv(OUT_DIR / "researchpass_groupF17_groupnorm.csv", index=False)
    print(f"\n[groupnorm] TOTAL TIME: {(time.time()-t0)/60:.1f} minutes (training alone: {train_time/60:.1f} min)")


if __name__ == "__main__":
    main()
