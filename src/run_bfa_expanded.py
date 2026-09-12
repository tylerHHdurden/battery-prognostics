"""
Dataset Expansion Phase 1: re-runs BFA feature selection on the expanded
NASA+CALCE+MIT pool (hi_table_expanded.parquet, ~219 NASA+MIT batteries
+ 3 CALCE cells, vs. the original 32-battery pool's 35 total). Additive:
the original run_bfa.py / bfa_selected_features.txt / bfa_history.csv
are completely untouched.

Same method as run_bfa.py, unchanged: same 16-feature HI_NAMES set,
same median-imputation-per-column for CALCE's missing MATC/MATD/MATDL,
same literature-standard 30 agents x 100 iterations, same seed=42, same
battery_id GroupKFold grouping (now over ~222 groups instead of 35 -
more groups per fold, if anything a MORE reliable CV signal, not less).

Whether the resulting 7(ish)-feature selection changes at this larger
scale is itself the thing being tested here - reported either way,
not steered toward matching the original 7.
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
    df = pd.read_parquet(PROC_DIR / "hi_table_expanded.parquet")
    print(f"[bfa-exp] loaded hi_table_expanded: {df.shape}, "
          f"{df['battery_id'].nunique()} unique batteries")

    X = df[HI_NAMES].to_numpy(dtype=float, copy=True)
    # Bug caught on this expanded-pool run (not present in the original
    # 32-battery run): one single cycle (MIT/b2c30, cycle_idx=487, out
    # of 159,912 total) has a duplicate raw timestamp in its final
    # discharge samples, making VDEDT's dV/dt denominator exactly zero -
    # an actual `inf`, not just a large finite outlier (VDEDT already had
    # huge legitimate variance pre-expansion, see session 27's
    # mit_train_std=354437 finding). `nanmedian`/nan-imputation below
    # silently pass inf through untouched (they only handle NaN), which
    # crashed sklearn's StandardScaler downstream with "Input X contains
    # infinity" - treating inf as another form of "not a usable value"
    # and letting the SAME median-imputation already used for NaN handle
    # it too, rather than writing a second imputation path.
    n_inf = np.isinf(X).sum()
    if n_inf:
        print(f"[bfa-exp] {n_inf} inf value(s) found (e.g. VDEDT dV/dt division-by-zero "
              f"on a duplicate-timestamp cycle) - treated as missing, median-imputed below")
        X[np.isinf(X)] = np.nan
    n_nan_before = np.isnan(X).sum()
    col_medians = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_medians, inds[1])
    print(f"[bfa-exp] imputed {n_nan_before} NaN cells (median-per-column, "
          f"mostly CALCE MATC/MATD/MATDL)")

    y = df["SOH"].to_numpy(dtype=float)
    groups = df["battery_id"].to_numpy()

    best_mask, selected_names, history = run_bfa(
        X, y, groups, HI_NAMES,
        n_agents=30, n_iterations=100,
        seed=42, log_fn=print,
    )

    pd.DataFrame(history).to_csv(PROC_DIR / "bfa_history_expanded.csv", index=False)
    with open(PROC_DIR / "bfa_selected_features_expanded.txt", "w") as f:
        f.write("\n".join(selected_names))

    original = (PROC_DIR / "bfa_selected_features.txt").read_text().strip().split("\n")
    same = set(original) == set(selected_names)
    print(f"[bfa-exp] DONE. Selected ({len(selected_names)}): {selected_names}")
    print(f"[bfa-exp] Original 32-battery selection ({len(original)}): {original}")
    print(f"[bfa-exp] IDENTICAL to original selection: {same}")


if __name__ == "__main__":
    main()
