"""
Session 22: re-runs BFA feature selection with the 3 new EIS-derived
candidate HIs (eis_features.py: EIS_Re, EIS_Rct, EIS_Zmag_mean) added
alongside the existing 16, to see honestly whether they get selected
over the current 7. Fully additive: does NOT touch run_bfa.py,
hi_table.parquet, or bfa_selected_features.txt - reads hi_table.parquet,
merges the EIS features in-memory onto a COPY, and writes its own
parallel `*_with_eis` output files.

Same wrapper-fitness BFA (bfa_feature_selection.run_bfa, unchanged,
imported not duplicated), same battery-grouped CV, same 30 agents x 100
iterations, same seed=42 - the ONLY difference from run_bfa.py is the
input feature matrix (19 candidates instead of 16).

**Missingness, reported plainly per instruction rather than worked
around silently**: EIS_Re/EIS_Rct/EIS_Zmag_mean are NASA-ONLY - MIT and
CALCE have zero EIS data (confirmed: MIT's source HDF5 files and
CALCE's format contain no impedance measurements of any kind, unlike
NASA's raw .mat files, which log 'impedance'-type cycle entries
alongside charge/discharge). This means the 3 new candidates are NaN
for ~94% of hi_table's rows (NASA is 4 of ~35 total batteries pooled
into hi_table.parquet) - a FAR more severe missingness pattern than the
existing MATC/MATD/MATDL columns (NaN only for CALCE's 3 of ~35
batteries, i.e. ~91% present vs. EIS's ~6% present). Median-imputed
here with the exact same convention run_bfa.py already uses for MATC/
MATD (impute NaN with the column median computed from non-NaN, i.e.
NASA-only, values) - logged explicitly because ~94% of each EIS
column's values going into the wrapper fitness are therefore an
imputed CONSTANT, not measured data, which is expected to (and, per the
result below, does) make it far harder for BFA's wrapper accuracy
signal to reward these features relative to the existing 16.
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

EIS_NAMES = ["EIS_Re", "EIS_Rct", "EIS_Zmag_mean"]


def main():
    df = pd.read_parquet(PROC_DIR / "hi_table.parquet")
    print(f"[bfa-eis] loaded hi_table: {df.shape}")

    eis = pd.read_csv(PROC_DIR / "eis_features_nasa.csv")
    df = df.copy()
    df = pd.merge(df, eis[["battery_id", "cycle_idx"] + EIS_NAMES],
                   on=["battery_id", "cycle_idx"], how="left")
    n_nasa_rows = (df["dataset"] == "NASA").sum()
    n_eis_present = df["EIS_Re"].notna().sum()
    print(f"[bfa-eis] merged EIS features: {n_eis_present} of {len(df)} rows have EIS data "
          f"({n_eis_present/len(df)*100:.1f}%) - all from NASA ({n_nasa_rows} NASA rows total, "
          f"{n_eis_present} of which got a nearest-impedance match)")
    print(f"[bfa-eis] MIT rows with EIS data: {(df[df['dataset']=='MIT']['EIS_Re'].notna()).sum()} "
          f"(expected 0 - MIT has no EIS data at all)")
    print(f"[bfa-eis] CALCE rows with EIS data: {(df[df['dataset']=='CALCE']['EIS_Re'].notna()).sum()} "
          f"(expected 0 - CALCE has no EIS data at all)")

    all_names = HI_NAMES + EIS_NAMES
    X = df[all_names].to_numpy(dtype=float, copy=True)
    n_nan_before = np.isnan(X).sum()
    col_medians = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_medians, inds[1])
    print(f"[bfa-eis] imputed {n_nan_before} NaN cells total across {len(all_names)} candidate "
          f"features (median-per-column, same convention as run_bfa.py)")
    for name in EIS_NAMES:
        col_idx = all_names.index(name)
        n_nan_col = np.isnan(df[name].to_numpy(dtype=float)).sum()
        print(f"    {name}: {n_nan_col}/{len(df)} NaN ({n_nan_col/len(df)*100:.1f}%), "
              f"imputed with NASA-only median={col_medians[col_idx]:.4f}")

    y = df["SOH"].to_numpy(dtype=float)
    groups = df["battery_id"].to_numpy()

    best_mask, selected_names, history = run_bfa(
        X, y, groups, all_names,
        n_agents=30, n_iterations=100,
        seed=42, log_fn=print,
    )

    pd.DataFrame(history).to_csv(PROC_DIR / "bfa_history_with_eis.csv", index=False)
    with open(PROC_DIR / "bfa_selected_features_with_eis.txt", "w") as f:
        f.write("\n".join(selected_names))

    original_selected = set(Path(PROC_DIR / "bfa_selected_features.txt").read_text().split())
    eis_selected = [n for n in selected_names if n in EIS_NAMES]
    print(f"\n[bfa-eis] === RESULT ===")
    print(f"[bfa-eis] original 7 selected (16-candidate BFA, session 1): {sorted(original_selected)}")
    print(f"[bfa-eis] new selection (19-candidate BFA, this session): {sorted(selected_names)}")
    print(f"[bfa-eis] EIS features selected: {eis_selected if eis_selected else 'NONE'}")
    print(f"[bfa-eis] DONE")


if __name__ == "__main__":
    main()
