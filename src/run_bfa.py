"""
Runs the Binary Firefly Algorithm over the pooled NASA+CALCE+MIT HI table
to select a feature subset for the XGBoost base learner (Phase 2).

Target: SOH (%). Groups: battery_id (so CV folds never split a battery's
cycles across train/test — required for a meaningful wrapper fitness).

ASSUMPTION: CALCE's MATC/MATD/MATDL are NaN (no temperature column, see
data_adapters.py). Rather than drop those 3 columns dataset-wide (which
would throw away real temperature information from NASA/MIT), NaNs are
median-imputed per column using only the non-NaN (NASA+MIT) values. This
is logged because it means CALCE rows contribute imputed, not measured,
values for those 3 of 16 features during BFA fitness evaluation.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bfa_feature_selection import run_bfa
from health_indicators import HI_NAMES

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"


def main():
    df = pd.read_parquet(PROC_DIR / "hi_table.parquet")
    print(f"[bfa] loaded hi_table: {df.shape}")

    X = df[HI_NAMES].to_numpy(dtype=float, copy=True)
    n_nan_before = np.isnan(X).sum()
    col_medians = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_medians, inds[1])
    print(f"[bfa] imputed {n_nan_before} NaN cells (median-per-column, "
          f"mostly CALCE MATC/MATD/MATDL)")

    y = df["SOH"].to_numpy(dtype=float)
    groups = df["battery_id"].to_numpy()

    best_mask, selected_names, history = run_bfa(
        X, y, groups, HI_NAMES,
        n_agents=30, n_iterations=100,
        seed=42, log_fn=print,
    )

    pd.DataFrame(history).to_csv(PROC_DIR / "bfa_history.csv", index=False)
    with open(PROC_DIR / "bfa_selected_features.txt", "w") as f:
        f.write("\n".join(selected_names))
    print(f"[bfa] DONE. Selected features saved to bfa_selected_features.txt")


if __name__ == "__main__":
    main()
