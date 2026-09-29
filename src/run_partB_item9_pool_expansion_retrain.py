"""
Part B, item 9: retest the training-pool-expansion pattern that worked
in Stage 5 (32->204 batteries) - fold the new BatteryLife batteries into
the training pool and retrain XGBoost-fusion, following Stage 5's own
methodology.

DISCLOSED, NARROWER SCOPE THAN STAGE 5's OWN 204-POOL VERSION (per the
task's own explicit permission: "if you find reason to also retrain a
deep-learning component, disclose that scope decision explicitly rather
than silently expanding it"): Stage 5's 204-pool retrain
(`run_pool204_step2_retrain_eval.py`) ALSO retrained the ICA fusion
encoder itself on the expanded pool (its own `ica_encoder_pool204.pt`,
`channel_norm_stats_pool204.json`). This item does NOT retrain the
encoder - it reuses the SAME already-trained `models/ica_encoder.pt`
every other dataset's fusion embeddings already come from (the same one
`build_batterylife_hi_table.py` used to build the new sources' own
fusion columns) - found no specific reason this pass's time budget
justified retraining a second deep-learning component, so kept to
EXACTLY what the task named ("retrained XGBoost-fusion specifically").

TRAIN/TEST SPLIT FOR THE NEW DATA, disclosed and reasoned: the new
BatteryLife batteries cannot be BOTH folded into training AND reported
as "zero-retrain" - that would be a direct contradiction. Instead: a
FIXED-SEED (42) 80/20 battery-level split of the new BatteryLife pool
into `new_train` (folded into the expanded training pool) and
`new_test` (held out, evaluated as a genuine, if non-standard, new-
domain generalization check - NOT part of the standard protocol, kept
clearly separate from it). The ORIGINAL NASA+MIT `battery_split.json`
train/test split is UNCHANGED - the "in-domain (fixed split)" test
battery set is IDENTICAL to every other item's own in-domain number in
this pass, directly comparable to the routed baseline's own 0.974.

HEADLINE COMPARISON (per the project's own governing rule): does this
retrained model beat the CURRENT ROUTED deployed baseline on the
STANDARD protocol (original in-domain test + CALCE/Oxford/HUST/XJTU
zero-retrain)? The new-BatteryLife-sources' own `new_test` split is
reported as a secondary, bonus data point, explicitly NOT counted
toward that headline verdict (those datasets are not part of the
standard protocol this project's governing rule is defined against).
"""
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from researchpass_partA_common import (
    load_base_pool_and_split, load_all_heldout_base,
    base_feature_cols, fusion_cols, score, OUT_DIR, PROC_DIR, ROOT,
)

SEED = 42
SOURCES = ["ul_pur", "hnei", "snl", "mich", "mich_exp", "rwth", "stanford", "stanford_2", "isu_ilcc"]
ROUTED_BASELINE = {"in-domain (fixed split)": 0.974, "CALCE": 0.740, "Oxford": 0.953, "HUST": 0.800, "XJTU": -1.062}


def load_new_batterylife_pool():
    parts = []
    per_source_batteries = {}
    for source in SOURCES:
        path = PROC_DIR / f"batterylife_{source}_merged.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df["battery_id"] = source + "::" + df["battery_id"].astype(str)  # namespace to avoid cross-source collisions
        parts.append(df)
        per_source_batteries[source] = sorted(df["battery_id"].unique())
    if not parts:
        return None, {}
    return pd.concat(parts, ignore_index=True), per_source_batteries


def main():
    t0 = time.time()
    print("=== Part B, item 9: pool-expansion retrain (Stage 5 pattern, XGBoost-fusion only) ===")

    new_df, per_source = load_new_batterylife_pool()
    if new_df is None:
        print("[item9] STOPPING: no local BatteryLife merged tables found at all - nothing to fold in.")
        return
    all_new_ids = sorted(new_df["battery_id"].unique())
    rng = random.Random(SEED)
    shuffled = all_new_ids[:]
    rng.shuffle(shuffled)
    n_test = max(1, int(round(0.2 * len(shuffled))))
    new_test_ids = set(shuffled[:n_test])
    new_train_ids = set(shuffled[n_test:])
    print(f"[item9] new BatteryLife pool: {len(all_new_ids)} batteries across {len(per_source)} sources "
          f"-> {len(new_train_ids)} folded into TRAIN, {len(new_test_ids)} held out as new_test (80/20, seed={SEED})")
    for s, ids in per_source.items():
        print(f"[item9]   {s}: {len(ids)} batteries")

    print("\n[item9] loading original NASA+MIT pool (Stage 1.1 duration-reformulated, UNCHANGED split)...")
    merged_base, hi_full_base, orig_train_mask, orig_test_mask, split = load_base_pool_and_split()
    base_cols = base_feature_cols()
    fcols = fusion_cols()
    all_cols = base_cols + fcols

    # Align new_df's columns to the exact same feature set (it already
    # has them all, built by build_batterylife_hi_table.py using the
    # identical reformulation code).
    missing = [c for c in all_cols + ["SOH", "battery_id"] if c not in new_df.columns]
    if missing:
        print(f"[item9] STOPPING: new BatteryLife table missing required columns: {missing}")
        return

    orig_train_df = merged_base.loc[orig_train_mask, all_cols + ["SOH", "battery_id"]].copy()
    orig_test_df = merged_base.loc[orig_test_mask, all_cols + ["SOH", "battery_id"]].copy()

    new_train_df = new_df[new_df["battery_id"].isin(new_train_ids)][all_cols + ["SOH", "battery_id"]].copy()
    new_test_df = new_df[new_df["battery_id"].isin(new_test_ids)][all_cols + ["SOH", "battery_id"]].copy()

    expanded_train_df = pd.concat([orig_train_df, new_train_df], ignore_index=True)
    print(f"[item9] expanded TRAIN pool: {len(orig_train_df)} original rows ({orig_train_mask.sum()} orig batteries) "
          f"+ {len(new_train_df)} new BatteryLife rows ({len(new_train_ids)} new batteries) "
          f"= {len(expanded_train_df)} rows, {expanded_train_df['battery_id'].nunique()} batteries total")

    X_train = expanded_train_df[all_cols].to_numpy(dtype=float, copy=True)
    X_train = np.where(np.isinf(X_train), np.nan, X_train)
    col_medians = np.nanmedian(X_train, axis=0)
    inds = np.where(np.isnan(X_train))
    X_train[inds] = np.take(col_medians, inds[1])
    y_train = expanded_train_df["SOH"].to_numpy(dtype=float)

    cyc_pos = base_cols.index("cycle_idx")
    monotone = tuple([-1 if i == cyc_pos else 0 for i in range(len(base_cols))] + [0] * len(fcols))
    print(f"[item9] monotone_constraints (only cycle_idx constrained, matching Stage 1.5/Stage 5 exactly): {monotone}")

    print("\n[item9] retraining XGBoost-fusion on the EXPANDED pool (same hyperparameters as every other "
          "XGBoost-fusion variant in this project)...")
    model = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03, subsample=0.8,
                          colsample_bytree=0.8, random_state=SEED, n_jobs=-1, reg_lambda=1.0,
                          monotone_constraints=monotone)
    model.fit(X_train, y_train)
    model.save_model(str(ROOT / "models" / "_experimental_xgb_soh_fusion_batterylife_expanded.json"))
    print("[item9] saved to models/_experimental_xgb_soh_fusion_batterylife_expanded.json - "
          "NOT wired into live_inference.py/app.py")

    def evaluate(df, label):
        X = df[all_cols].to_numpy(dtype=float, copy=True)
        X = np.where(np.isinf(X), np.nan, X)
        i2 = np.where(np.isnan(X))
        X[i2] = np.take(col_medians, i2[1])
        pred = model.predict(X)
        r = score(df["SOH"].to_numpy(dtype=float), pred)
        return r

    print("\n=== HEADLINE: standard protocol, vs. CURRENT ROUTED deployed baseline ===")
    results = []
    r = evaluate(orig_test_df, "in-domain")
    v = "WIN" if r["r2"] > ROUTED_BASELINE["in-domain (fixed split)"] else "LOSS"
    print(f"[item9] in-domain (fixed split, UNCHANGED test battery set): R2={r['r2']:.4f} RMSE={r['rmse']:.4f} "
          f"n={r['n']} | vs routed baseline {ROUTED_BASELINE['in-domain (fixed split)']:.3f} -> {v}")
    results.append({"eval_set": "in-domain (fixed split)", "part_of_standard_protocol": True, **r,
                     "routed_baseline_r2": ROUTED_BASELINE["in-domain (fixed split)"], "verdict": v})

    held = load_all_heldout_base(hi_full_base)
    for name in ["CALCE", "Oxford", "HUST", "XJTU"]:
        df = held[name][all_cols + ["SOH"]].copy() if all(c in held[name].columns for c in all_cols) else None
        if df is None:
            print(f"[item9] {name}: missing required columns in its own held-out table - skipped "
                  f"(should not happen; flagged if it does)")
            continue
        r = evaluate(df, name)
        v = "WIN" if r["r2"] > ROUTED_BASELINE[name] else "LOSS"
        print(f"[item9] {name} (zero-retrain, UNCHANGED - never in the expanded train pool): "
              f"R2={r['r2']:.4f} RMSE={r['rmse']:.4f} n={r['n']} | vs routed baseline {ROUTED_BASELINE[name]:.3f} -> {v}")
        results.append({"eval_set": name, "part_of_standard_protocol": True, **r,
                         "routed_baseline_r2": ROUTED_BASELINE[name], "verdict": v})

    print("\n=== SECONDARY, BONUS: held-out 20% of the NEW BatteryLife pool itself (NOT part of the standard protocol) ===")
    if len(new_test_df):
        r = evaluate(new_test_df, "new_batterylife_held_out_20pct")
        print(f"[item9] new BatteryLife held-out batteries ({len(new_test_ids)} batteries, {r['n']} cycles): "
              f"R2={r['r2']:.4f} RMSE={r['rmse']:.4f} - NOT counted toward the headline verdict")
        results.append({"eval_set": "new_batterylife_held_out_20pct", "part_of_standard_protocol": False, **r,
                         "routed_baseline_r2": None, "verdict": "n/a (not standard protocol)"})
    else:
        print("[item9] no new_test rows - skipped")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "partB_item9_pool_expansion_retrain.csv", index=False)
    print("\n=== Item 9 FULL RESULTS ===")
    print(results_df.to_string(index=False))

    n_wins_standard = int((results_df[results_df["part_of_standard_protocol"] == True]["verdict"] == "WIN").sum())
    n_standard = int((results_df["part_of_standard_protocol"] == True).sum())
    print(f"\n[item9] STANDARD PROTOCOL: {n_wins_standard}/{n_standard} eval settings beat the routed deployed "
          f"baseline. Per project rule: no promotion regardless of outcome.")
    print(f"[item9] TOTAL TIME: {(time.time()-t0)/60:.2f} minutes")


if __name__ == "__main__":
    main()
