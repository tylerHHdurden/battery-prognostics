"""
Stage 0, Check 0.3: BFA feature-selection leakage check.

CONFIRMED WITH DIRECT EVIDENCE (not inference) before writing this
script: src/run_bfa.py's own docstring states "Runs the Binary Firefly
Algorithm over the pooled NASA+CALCE+MIT HI table" and its code loads
`hi_table.parquet` with ZERO dataset filter (`df = pd.read_parquet(...)`,
no `.isin(["NASA","MIT"])` the way train_xgboost.py explicitly has).
Phase 1's own log line "35/35 batteries, 26,996 total cycles" already
named this (35 = 4 NASA + 28 MIT + 3 CALCE), but was never previously
flagged as a leakage concern. CALCE's SOH labels WERE visible to BFA's
GroupKFold wrapper-fitness function (Ridge-regression cross-validated
RMSE) that selected the original 7-feature set every base learner since
Phase 2 has used - a genuine methodological leak against CALCE's role
everywhere else in this project as a never-trained-on holdout.

This script re-runs BFA identically (same run_bfa() call, same
n_agents/n_iterations/seed) with CALCE excluded from the pool entirely -
NASA+MIT only, matching train_xgboost.py's own filter exactly.
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
    df_full = pd.read_parquet(PROC_DIR / "hi_table.parquet")
    df_nasa_mit = df_full[df_full["dataset"].isin(["NASA", "MIT"])].reset_index(drop=True)
    n_calce = len(df_full) - len(df_nasa_mit)
    print(f"[bfa-nomcalce] hi_table full (incl. CALCE): {df_full.shape}, "
          f"NASA+MIT only: {df_nasa_mit.shape} ({n_calce} CALCE rows excluded)")
    print(f"[bfa-nomcalce] batteries: "
          f"{df_nasa_mit['battery_id'].nunique()} (was {df_full['battery_id'].nunique()} incl. CALCE)")

    X = df_nasa_mit[HI_NAMES].to_numpy(dtype=float, copy=True)
    n_nan_before = np.isnan(X).sum()
    col_medians = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_medians, inds[1])
    print(f"[bfa-nomcalce] imputed {n_nan_before} NaN cells (median-per-column, "
          f"NASA+MIT only - no CALCE MATC/MATD/MATDL NaNs to impute anymore)")

    y = df_nasa_mit["SOH"].to_numpy(dtype=float)
    groups = df_nasa_mit["battery_id"].to_numpy()

    best_mask, selected_names, history = run_bfa(
        X, y, groups, HI_NAMES,
        n_agents=30, n_iterations=100,
        seed=42, log_fn=print,
    )

    pd.DataFrame(history).to_csv(PROC_DIR / "bfa_history_nasa_mit_only.csv", index=False)
    with open(PROC_DIR / "bfa_selected_features_nasa_mit_only.txt", "w") as f:
        f.write("\n".join(selected_names))

    with open(PROC_DIR / "bfa_selected_features.txt") as f:
        original_with_calce = [l.strip() for l in f if l.strip()]
    print(f"\n[bfa-nomcalce] === COMPARISON ===")
    print(f"[bfa-nomcalce] ORIGINAL (leaked, incl. CALCE, 32-battery pool): {original_with_calce}")
    print(f"[bfa-nomcalce] CORRECTED (NASA+MIT only, 32-battery pool):      {selected_names}")
    overlap = set(original_with_calce) & set(selected_names)
    print(f"[bfa-nomcalce] Overlap: {sorted(overlap)} ({len(overlap)}/{len(original_with_calce)})")

    try:
        with open(PROC_DIR / "bfa_selected_features_expanded.txt") as f:
            expanded_204 = [l.strip() for l in f if l.strip()]
        print(f"[bfa-nomcalce] Session 33's 204-battery reselection (NASA+MIT only, "
              f"already never included CALCE): {expanded_204}")
        overlap2 = set(selected_names) & set(expanded_204)
        print(f"[bfa-nomcalce] Overlap (corrected-32 vs. 204-battery): {sorted(overlap2)} "
              f"({len(overlap2)}/{len(selected_names)})")
    except FileNotFoundError:
        pass

    print("[bfa-nomcalce] DONE")


if __name__ == "__main__":
    main()
