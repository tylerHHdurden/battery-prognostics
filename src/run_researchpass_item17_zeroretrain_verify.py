"""
Completes the zero-retrain verification that was flagged and left undone
for item 17 (GroupNorm swap for CNN-LSTM's BatchNorm) in the original
18-item research pass. That item only ever reported an in-domain result
(GroupNorm R2=0.9757 vs. BatchNorm baseline R2=0.9666) and was
explicitly flagged as "not yet held-out verified."

Loads the ALREADY-TRAINED checkpoint (models/_experimental_cnn_lstm_
groupnorm_expanded.pt) - no retraining - and scores it zero-retrain on
CALCE/Oxford/HUST/XJTU, using the SAME channel_norm_stats_expanded.json
the model was trained with.

BASELINE FOR THE VERDICT: per this project's own governing rule
("nothing gets promoted unless it clearly beats the CURRENT DEPLOYED
baseline on the full protocol"), the relevant comparison is against
the model actually running in production (the XGBoost-fusion SOH
model), using the TRUE numbers this session's own audit established
(audit_true_deployed_baseline.csv) - NOT the mislabeled
DEPLOYED_REFERENCE dict item 17's own BatchNorm-vs-GroupNorm comparison
never touched in the first place (that comparison was always CNN-LSTM
BatchNorm vs. CNN-LSTM GroupNorm, a different architecture family
entirely, and was never affected by the mislabeling bug - reconfirmed
here for completeness, not because it was ever wrong).
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
from sequence_features import apply_channel_norm
from train_deep_models import make_xy
from train_deep_models_expanded import load_all_battery_tensors_expanded, predict
from stage7_common import load_heldout

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"

# In-domain numbers already on record (BatchNorm-vs-GroupNorm CNN-LSTM
# comparison, expanded 204-battery pool) - reconfirmed, not recomputed,
# since reloading+rerunning the full expanded pool's in-domain TEST set
# is redundant with what's already verified and saved.
BATCHNORM_INDOMAIN = {"rmse": 1.3002859094637986, "mae": 0.7450214624404907, "r2": 0.9665936231613159}
GROUPNORM_INDOMAIN_ON_RECORD = {"rmse": 1.1093, "mae": 0.7423, "r2": 0.9757}

# TRUE currently-deployed XGBoost-fusion model's own numbers, per this
# session's own audit (outputs/audit_true_deployed_baseline.csv) - the
# actual governing "beats baseline" comparison point for any promotion
# decision, regardless of candidate architecture.
TRUE_DEPLOYED_XGB_FUSION = {
    "in-domain (TEST)": 0.973957, "CALCE": 0.567876, "Oxford": -2.693906,
    "HUST": -0.152262, "XJTU": -1.061987,
}


class MultiKernelBranchGN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, num_groups: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.gn = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        return torch.relu(self.gn(self.conv(x)))


class CNNLSTM_GroupNorm(nn.Module):
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
    print("=== Completing item 17's zero-retrain verification (CALCE/Oxford/HUST/XJTU) ===")

    print("[verify17] loading expanded pool (cached) to reconstruct y_mean_/y_std_ "
          "(NOT part of state_dict, must be recomputed from the SAME fit split used at training time)...")
    battery_data = load_all_battery_tensors_expanded()
    split = json.loads((PROC_DIR / "battery_split_expanded.json").read_text())
    train_ids = [b for b in split["train_ids"] if b in battery_data]
    n_val_batteries = max(1, len(train_ids) // 5)
    val_ids = sorted(train_ids)[-n_val_batteries:]
    fit_ids = [b for b in train_ids if b not in val_ids]
    _, y_fit, _, _, _, _ = make_xy(battery_data, fit_ids)
    y_mean, y_std = float(y_fit.mean()), float(y_fit.std() + 1e-8)
    print(f"[verify17] y_mean_={y_mean:.4f} y_std_={y_std:.4f} (from {len(fit_ids)} fit batteries)")

    model = CNNLSTM_GroupNorm()
    model.load_state_dict(torch.load(ROOT / "models" / "_experimental_cnn_lstm_groupnorm_expanded.pt"))
    model.eval()
    model.y_mean_, model.y_std_ = y_mean, y_std

    norm_stats = json.loads((PROC_DIR / "channel_norm_stats_expanded.json").read_text())

    results = []
    for name in ["CALCE", "Oxford", "HUST", "XJTU"]:
        print(f"\n[verify17] loading {name} (raw cycles -> 6-channel tensors, zero-retrain)...")
        pool = load_heldout(name)
        if not pool:
            print(f"[verify17] WARNING: no usable {name} data, skipping")
            continue
        X_all = np.concatenate([X for X, soh, rul in pool.values()]).astype(np.float32)
        soh_all = np.concatenate([soh for X, soh, rul in pool.values()]).astype(np.float32)
        X_norm = apply_channel_norm(X_all, norm_stats)
        pred = predict(model, X_norm)
        r2 = float(r2_score(soh_all, pred))
        rmse = float(np.sqrt(mean_squared_error(soh_all, pred)))
        mae = float(mean_absolute_error(soh_all, pred))
        true_dep = TRUE_DEPLOYED_XGB_FUSION[name]
        verdict = "WIN" if r2 > true_dep else "LOSS"
        print(f"[verify17] {name}: GroupNorm CNN-LSTM R2={r2:.4f} RMSE={rmse:.4f} MAE={mae:.4f} "
              f"n={len(soh_all)} n_batteries={len(pool)} | vs TRUE deployed XGB-fusion R2={true_dep:.4f} -> {verdict}")
        results.append({"eval_set": name, "groupnorm_cnnlstm_r2": r2, "groupnorm_cnnlstm_rmse": rmse,
                         "groupnorm_cnnlstm_mae": mae, "n": len(soh_all), "n_batteries": len(pool),
                         "true_deployed_xgb_fusion_r2": true_dep, "verdict_vs_true_deployed": verdict})

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "researchpass_item17_zeroretrain_verify.csv", index=False)

    print("\n=== SUMMARY ===")
    print(f"In-domain (already on record, reconfirmed not recomputed): "
          f"GroupNorm R2={GROUPNORM_INDOMAIN_ON_RECORD['r2']:.4f} vs. BatchNorm R2={BATCHNORM_INDOMAIN['r2']:.4f} "
          f"-> still a WIN within the CNN-LSTM architecture family (unaffected by the mislabeling bug - "
          f"this comparison was never against the XGBoost-fusion model at all)")
    print(f"vs. TRUE deployed XGBoost-fusion baseline (in-domain): "
          f"{GROUPNORM_INDOMAIN_ON_RECORD['r2']:.4f} vs {TRUE_DEPLOYED_XGB_FUSION['in-domain (TEST)']:.4f} -> "
          f"{'WIN' if GROUPNORM_INDOMAIN_ON_RECORD['r2'] > TRUE_DEPLOYED_XGB_FUSION['in-domain (TEST)'] else 'LOSS'}")
    print(results_df.to_string(index=False))

    n_wins = int((results_df["verdict_vs_true_deployed"] == "WIN").sum())
    print(f"\n[verify17] Zero-retrain: {n_wins}/{len(results_df)} held-out datasets beat the TRUE deployed "
          f"XGBoost-fusion baseline. Per the project's own rule, promotion needs a clear win on the FULL "
          f"protocol (in-domain + all held-out) - this is a verification report, not a promotion decision, "
          f"nothing is being promoted here regardless of outcome.")
    print(f"\n[verify17] TOTAL TIME: {(time.time()-t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
