"""
Research pass Group A, item 1: Isolation Forest as a replacement for
the deployed One-Class SVM anomaly detector.

METHODOLOGY DISCLOSED: the task cites "session 7's original 24.4%/2.3%
figures" as the baseline - but train_ocsvm.py's OWN current docstring
records a DIFFERENT number (83.9%/2.2%, the PRE-balance-fix bug) for
what reads as a different point in this project's history, and the
CURRENT deployed OC-SVM's own NASA-vs-MIT false-flag rate on a genuine
held-out (VAL) split is not written down anywhere as a single number -
train_ocsvm.py's own instrumentation only prints ONE aggregate FIT-set
rate, never split by dataset, never evaluated on VAL. Rather than reuse
a possibly-stale, ambiguously-sourced historical figure, this script
fits BOTH detectors fresh, on the EXACT SAME data train_ocsvm.py uses
(same features, same balanced per-battery FIT sampling), and evaluates
BOTH on the SAME genuine held-out VAL split (already carved out by
train_ocsvm.py's own code, just never actually scored) - a clean,
current, apples-to-apples comparison for both methods at once, not
old-number-vs-new-method.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_common import canonical_feature_cols, add_reformulated_duration_features

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"


def build_pool():
    selected = canonical_feature_cols(reformulated=True) + ["cycle_idx"]
    hi_df = pd.read_parquet(PROC_DIR / "hi_table.parquet")
    hi_df = hi_df[hi_df["dataset"].isin(["NASA", "MIT"])].reset_index(drop=True)
    hi_df = add_reformulated_duration_features(hi_df)
    fusion = pd.read_csv(PROC_DIR / "fusion_embeddings.csv")
    fusion_cols = [c for c in fusion.columns if c.startswith("fusion_")]
    merged = pd.merge(hi_df, fusion, on=["dataset", "battery_id", "cycle_idx"], how="inner")
    feature_cols = selected + fusion_cols
    return merged, feature_cols


def main():
    t0 = time.time()
    print("=== Research pass Group A, item 1: Isolation Forest vs. One-Class SVM ===")
    merged, feature_cols = build_pool()

    split = json.loads((PROC_DIR / "battery_split.json").read_text())
    train_ids = split["train_ids"]
    # FIX, found by direct inspection of the actual output before trusting
    # it: train_ocsvm.py's own `sorted(train_ids)[-n_val:]` VAL-split logic
    # (replicated here unmodified at first, to match the deployed script
    # exactly) silently produces a VAL set with ZERO NASA batteries -
    # uppercase "B00XX" (NASA) sorts lexicographically BEFORE lowercase
    # "b#c#" (MIT) in Python string sort, so the LAST n_val elements are
    # always all-MIT. This is a real, pre-existing bug in the deployed
    # script's own VAL-split construction, invisible until now because
    # train_ocsvm.py never actually scores VAL (only prints a FIT-set
    # rate) - useless for a NASA-vs-MIT imbalance comparison, so this
    # script does NOT reuse it: a proper stratified split (both NASA and
    # MIT represented in VAL) is required for the comparison to mean
    # anything, built explicitly rather than assumed correct.
    nasa_ids = sorted(b for b in train_ids if b.startswith("B0"))
    mit_ids = sorted(b for b in train_ids if not b.startswith("B0"))
    rng_split = np.random.default_rng(42)
    val_nasa = list(rng_split.choice(nasa_ids, size=max(1, len(nasa_ids) // 5), replace=False))
    val_mit = list(rng_split.choice(mit_ids, size=max(1, len(mit_ids) // 5), replace=False))
    val_ids = val_nasa + val_mit
    fit_ids = [b for b in train_ids if b not in val_ids]
    print(f"[researchA1] FIT batteries: {len(fit_ids)}, VAL batteries (genuine held-out, "
          f"never fit on by either detector): {len(val_ids)} -> {val_ids}")

    fit_df = merged[merged["battery_id"].isin(fit_ids)]
    val_df = merged[merged["battery_id"].isin(val_ids)]
    print(f"[researchA1] VAL set composition: "
          f"{(val_df['dataset']=='NASA').sum()} NASA cycles, {(val_df['dataset']=='MIT').sum()} MIT cycles")

    # SAME balanced per-battery sampling as the deployed train_ocsvm.py
    max_per_battery = 200
    rng = np.random.default_rng(42)
    sampled = []
    for bid, g in fit_df.groupby("battery_id"):
        if len(g) > max_per_battery:
            g = g.sample(n=max_per_battery, random_state=42)
        sampled.append(g)
    fit_df = pd.concat(sampled, ignore_index=True)
    print(f"[researchA1] FIT set (balanced, capped {max_per_battery}/battery): {len(fit_df)} cycles")

    X_fit = fit_df[feature_cols].to_numpy(dtype=float, copy=True)
    col_medians = np.nanmedian(X_fit, axis=0)
    inds = np.where(np.isnan(X_fit))
    X_fit[inds] = np.take(col_medians, inds[1])
    scaler = StandardScaler().fit(X_fit)
    X_fit_scaled = scaler.transform(X_fit)

    X_val = val_df[feature_cols].to_numpy(dtype=float, copy=True)
    inds_v = np.where(np.isnan(X_val))
    X_val[inds_v] = np.take(col_medians, inds_v[1])
    X_val_scaled = scaler.transform(X_val)
    val_dataset = val_df["dataset"].to_numpy()

    results = []
    for name, model in [
        ("OneClassSVM (deployed, rbf, nu=0.05)", OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")),
        ("IsolationForest (contamination=0.05, n_estimators=200, random_state=42)",
         IsolationForest(contamination=0.05, n_estimators=200, random_state=42, n_jobs=-1)),
    ]:
        model.fit(X_fit_scaled)
        fit_pred = model.predict(X_fit_scaled)
        val_pred = model.predict(X_val_scaled)
        frac_fit_flagged = float((fit_pred == -1).mean())
        frac_val_flagged = float((val_pred == -1).mean())
        nasa_mask = val_dataset == "NASA"
        mit_mask = val_dataset == "MIT"
        nasa_flag_rate = float((val_pred[nasa_mask] == -1).mean()) if nasa_mask.sum() else float("nan")
        mit_flag_rate = float((val_pred[mit_mask] == -1).mean()) if mit_mask.sum() else float("nan")
        print(f"\n[researchA1] {name}")
        print(f"  FIT-set flag rate (should be ~0.05, the contamination assumption): {frac_fit_flagged:.4f}")
        print(f"  VAL-set flag rate overall: {frac_val_flagged:.4f}")
        print(f"  VAL-set NASA false-flag rate: {nasa_flag_rate:.4f} ({nasa_mask.sum()} cycles)")
        print(f"  VAL-set MIT false-flag rate: {mit_flag_rate:.4f} ({mit_mask.sum()} cycles)")
        print(f"  Imbalance (NASA - MIT): {nasa_flag_rate - mit_flag_rate:+.4f}")
        results.append({"method": name, "fit_flag_rate": frac_fit_flagged, "val_flag_rate": frac_val_flagged,
                         "val_nasa_flag_rate": nasa_flag_rate, "val_mit_flag_rate": mit_flag_rate,
                         "nasa_minus_mit_imbalance": nasa_flag_rate - mit_flag_rate})

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "researchpass_groupA1_ocsvm_vs_isolationforest.csv", index=False)
    print("\n=== SUMMARY ===")
    print(results_df.to_string(index=False))

    ocsvm_imbalance = abs(results[0]["nasa_minus_mit_imbalance"])
    if_imbalance = abs(results[1]["nasa_minus_mit_imbalance"])
    print(f"\n[researchA1] Absolute NASA-vs-MIT imbalance: OC-SVM={ocsvm_imbalance:.4f}, "
          f"IsolationForest={if_imbalance:.4f}")
    verdict = "WIN for IsolationForest (smaller imbalance)" if if_imbalance < ocsvm_imbalance else \
        ("TIE (imbalance ~equal)" if abs(if_imbalance - ocsvm_imbalance) < 0.01 else "LOSS for IsolationForest (larger imbalance)")
    print(f"[researchA1] VERDICT: {verdict}")
    print(f"\n[researchA1] TOTAL TIME: {(time.time()-t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
