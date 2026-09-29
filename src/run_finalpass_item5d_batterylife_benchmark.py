"""
Final research pass, item 5 (rigor pass), sub-item 4: BatteryLife's OWN
benchmark task (established via WebFetch of arXiv:2502.18807 in Part
B's own item 8: predict the CYCLE NUMBER at which SOH first reaches 80%
- 90% for CALB, none of the 9 locally-available sources are CALB - from
ONLY the first S<=100 cycles of a battery, scored by MAPE and "15%-Acc"
fraction of predictions within 15% relative error), reproduced here
with THIS project's own features + XGBoost, so item 8's disclosed
task/metric mismatch becomes a genuine, literal, same-protocol
comparison against BatteryLife's own published numbers.

Per-BATTERY regression (one row per battery, not one row per cycle -
a different granularity than every other item in this project):
  - TARGET: EOL cycle number (SOH first <=80% of that battery's own
    initial capacity), computed from the battery's FULL cycle history
    (same threshold-crossing rule as rul_labels.compute_eol_and_rul).
    Batteries that never cross 80% (right-censored) or whose EOL is
    <=100 (the task's own input horizon - predicting a past event from
    a future cutoff is degenerate) are EXCLUDED, disclosed by count.
  - FEATURES: aggregated from ONLY that battery's own cycles with
    cycle_idx<=100 - mean AND linear slope (vs. cycle_idx) of the 8
    canonical raw HI features (ICHV/SCV/VDEDT/VIECT/MATD/MET/TEVD/
    TEVI) and discharge_capacity, plus the mean 16-dim fusion
    embedding, plus n_cycles_used - this project's own established
    feature vocabulary, aggregated into a single per-battery snapshot
    rather than scored per-cycle.
  - MODEL: XGBoost regressor, 5-fold battery-level cross-validation
    (KFold, shuffled, seed=42 - no leakage risk since each row IS
    already one battery), pooled across all 9 locally-available
    BatteryLife sources (more data for a fundamentally small-n task -
    ~100-200 batteries total vs. BatteryLife's own hundreds-per-
    chemistry-family pretraining set).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_common import canonical_feature_cols, OUT_DIR, PROC_DIR

BATTERYLIFE_SOURCES = ["ul_pur", "hnei", "snl", "mich", "mich_exp", "rwth",
                        "stanford", "stanford_2", "isu_ilcc"]
EOL_FRACTION = 0.8
INPUT_HORIZON = 100
N_FOLDS = 5
SEED = 42

RAW_HI = canonical_feature_cols(reformulated=False)  # 8 raw HI names, no cycle_idx
FUSION_COLS = [f"fusion_{i}" for i in range(16)]


def eol_cycle_for_battery(g: pd.DataFrame):
    g = g.sort_values("cycle_idx")
    caps = g["discharge_capacity"].to_numpy(dtype=float)
    idxs = g["cycle_idx"].to_numpy(dtype=int)
    if len(caps) == 0 or not np.isfinite(caps[:3]).any():
        return None, True
    initial_cap = float(np.nanmedian(caps[:min(3, len(caps))]))
    if not np.isfinite(initial_cap) or abs(initial_cap) < 1e-9:
        return None, True
    threshold = EOL_FRACTION * initial_cap
    below = np.where(caps <= threshold)[0]
    if len(below) > 0:
        return int(idxs[below[0]]), False
    return int(idxs[-1]) + 1, True  # censored


def battery_feature_row(g: pd.DataFrame):
    early = g[g["cycle_idx"] <= INPUT_HORIZON].sort_values("cycle_idx")
    if len(early) < 5:
        return None
    row = {"n_cycles_used": len(early)}
    x = early["cycle_idx"].to_numpy(dtype=float)
    for col in RAW_HI + ["discharge_capacity"]:
        y = early[col].to_numpy(dtype=float)
        y = np.where(np.isinf(y), np.nan, y)
        finite = np.isfinite(y)
        row[f"{col}_mean"] = float(np.nanmean(y)) if finite.any() else np.nan
        if finite.sum() >= 2:
            slope = float(np.polyfit(x[finite], y[finite], 1)[0])
        else:
            slope = np.nan
        row[f"{col}_slope"] = slope
    for col in FUSION_COLS:
        row[f"{col}_mean"] = float(early[col].mean())
    return row


def main():
    t0 = time.time()
    print("=== Final pass item 5d: BatteryLife's own early-cycle-life-prediction benchmark, reproduced ===")

    all_rows = []
    n_censored, n_too_short_eol, n_too_short_input = 0, 0, 0
    for source in BATTERYLIFE_SOURCES:
        path = PROC_DIR / f"batterylife_{source}_merged.parquet"
        if not path.exists():
            print(f"[item5d] {source}: no local merged table - skipped")
            continue
        df = pd.read_parquet(path)
        for bid, g in df.groupby("battery_id"):
            eol, censored = eol_cycle_for_battery(g)
            if eol is None or censored:
                n_censored += 1
                continue
            if eol <= INPUT_HORIZON:
                n_too_short_eol += 1
                continue
            feat_row = battery_feature_row(g)
            if feat_row is None:
                n_too_short_input += 1
                continue
            feat_row.update({"source": source, "battery_id": bid, "eol_cycle": eol})
            all_rows.append(feat_row)

    battery_df = pd.DataFrame(all_rows)
    print(f"\n[item5d] {len(battery_df)} usable batteries across {battery_df['source'].nunique() if len(battery_df) else 0} sources "
          f"(excluded: {n_censored} right-censored/no-EOL, {n_too_short_eol} EOL<={INPUT_HORIZON}, "
          f"{n_too_short_input} <5 cycles in first {INPUT_HORIZON})")
    print(battery_df.groupby("source")["battery_id"].count().to_string() if len(battery_df) else "[item5d] NO USABLE BATTERIES")

    if len(battery_df) < 10:
        print("[item5d] Too few usable batteries for a meaningful cross-validated benchmark - STOPPING, not fabricating a result.")
        return

    feature_cols = [c for c in battery_df.columns if c not in {"source", "battery_id", "eol_cycle"}]
    X = battery_df[feature_cols].to_numpy(dtype=float)
    X = np.where(np.isinf(X), np.nan, X)
    y = battery_df["eol_cycle"].to_numpy(dtype=float)
    sources = battery_df["source"].to_numpy()
    battery_ids = battery_df["battery_id"].to_numpy()

    kf = KFold(n_splits=min(N_FOLDS, len(battery_df)), shuffle=True, random_state=SEED)
    oof_pred = np.full(len(y), np.nan)
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        col_medians = np.nanmedian(X[train_idx], axis=0)
        col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
        X_train = X[train_idx].copy(); inds = np.where(np.isnan(X_train)); X_train[inds] = np.take(col_medians, inds[1])
        X_test = X[test_idx].copy(); inds2 = np.where(np.isnan(X_test)); X_test[inds2] = np.take(col_medians, inds2[1])

        model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, reg_lambda=1.0)
        model.fit(X_train, y[train_idx])
        oof_pred[test_idx] = model.predict(X_test)
        print(f"[item5d] fold {fold+1}/{kf.get_n_splits()}: train={len(train_idx)} test={len(test_idx)}")

    ape = np.abs(oof_pred - y) / np.abs(y)
    mape = float(np.mean(ape))
    acc15 = float(np.mean(ape <= 0.15))
    print(f"\n=== OVERALL (pooled, 5-fold out-of-fold): MAPE={mape:.4f} 15%-Acc={acc15:.4f} n={len(y)} ===")

    result_df = battery_df[["source", "battery_id", "eol_cycle"]].copy()
    result_df["pred_eol_cycle"] = oof_pred
    result_df["ape"] = ape
    result_df.to_csv(OUT_DIR / "finalpass_item5d_battery_predictions.csv", index=False)

    per_source_rows = [{"source": "OVERALL (pooled)", "n_batteries": len(y), "mape": mape, "acc15pct": acc15}]
    for source in sorted(set(sources)):
        mask = sources == source
        if mask.sum() < 3:
            print(f"[item5d] {source}: only {mask.sum()} batteries - MAPE/Acc reported but LOW-N, not reliable on its own")
        s_mape = float(np.mean(ape[mask]))
        s_acc15 = float(np.mean(ape[mask] <= 0.15))
        per_source_rows.append({"source": source, "n_batteries": int(mask.sum()), "mape": s_mape, "acc15pct": s_acc15})
        print(f"[item5d] {source}: n={mask.sum()} MAPE={s_mape:.4f} 15%-Acc={s_acc15:.4f}")

    summary_df = pd.DataFrame(per_source_rows)
    summary_df.to_csv(OUT_DIR / "finalpass_item5d_summary.csv", index=False)

    print("\n[item5d] BatteryLife's OWN published numbers (Part B item 8, arXiv:2502.18807 Table 3, "
          "aggregated by CHEMISTRY FAMILY not source): Li-ion MAPE=0.184/0.179 15%-Acc=0.573, "
          "Zn-ion MAPE=0.515, Na-ion MAPE=0.255, CALB MAPE=0.149 15%-Acc=0.672.")
    print("[item5d] This IS now a genuinely comparable number (same task, same metric, this project's own "
          "features+XGBoost instead of their CPTransformer/CPMLP) - still not a strict apples-to-apples "
          "per-source comparison (their splits/exact battery sets are not reproduced, only the task/metric "
          "protocol), and the local pool is chemistry-mixed rather than family-separated the way their table "
          "is, stated plainly rather than implied away.")

    print(f"\n[item5d] TOTAL TIME: {(time.time()-t0)/60:.2f} minutes")


if __name__ == "__main__":
    main()
