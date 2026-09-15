"""
Battery-level cluster bootstrap significance check on the three-way
pool comparison (42-battery deployed vs. 204-battery vs. NASA-heavy),
on all four held-out datasets. Exact same methodology as session 21/
Stage 0.4 (battery_bootstrap_row_indices - resample whole BATTERIES
with replacement, not cycles, avoiding the pseudo-replication problem
this project already root-caused once; N_BOOTSTRAP=2000, 95%
percentile CI). Row-level alignment across all 3 pools' saved per-
cycle predictions verified exact (same battery_id order, same y_true)
before trusting any paired delta below.

No retraining - pure analysis on the already-saved per-cycle
predictions (run_save_percycle_predictions.py).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
PRED_DIR = PROC_DIR / "predictions"
OUT_DIR = ROOT / "outputs"

N_BOOTSTRAP = 2000
CI = (2.5, 97.5)
SEED = 42

POOLS = {"42-battery (deployed)": "deployed", "204-battery": "pool204", "NASA-heavy": "nasaheavy"}
DATASETS = ["CALCE", "Oxford", "HUST", "XJTU"]


def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)


def battery_bootstrap_row_indices(battery_ids, n_boot=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed + 1)
    unique_batteries = np.array(sorted(set(battery_ids)))
    battery_ids = np.asarray(battery_ids)
    row_idx_by_battery = {b: np.where(battery_ids == b)[0] for b in unique_batteries}
    n_batteries = len(unique_batteries)
    out = []
    for _ in range(n_boot):
        chosen = rng.choice(unique_batteries, size=n_batteries, replace=True)
        out.append(np.concatenate([row_idx_by_battery[b] for b in chosen]))
    return out, n_batteries


def main():
    r2_ci_rows = []
    delta_rows = []

    for dataset in DATASETS:
        print(f"\n=== {dataset} ===")
        data = {}
        for pool_label, suffix in POOLS.items():
            df = pd.read_csv(PRED_DIR / f"percycle_{dataset.lower()}_{suffix}.csv")
            data[pool_label] = df

        battery_ids = data["42-battery (deployed)"]["battery_id"].to_numpy()
        y_true = data["42-battery (deployed)"]["y_true"].to_numpy()
        idx_sets, n_batteries = battery_bootstrap_row_indices(battery_ids)
        n_distinct_subsets = 2 ** n_batteries - 1
        print(f"[bootstrap] {dataset}: {n_batteries} batteries, {len(battery_ids)} cycles, "
              f"{n_distinct_subsets} distinct nonempty battery subsets exist (before replacement)")

        # per-pool R2 CI
        preds = {}
        for pool_label, df in data.items():
            pred = df["pred"].to_numpy()
            preds[pool_label] = pred
            point_r2 = r2(y_true, pred)
            boot_r2 = np.array([r2(y_true[idx], pred[idx]) for idx in idx_sets])
            lo, hi = np.percentile(boot_r2, CI)
            width = hi - lo
            print(f"    {pool_label}: point R2={point_r2:+.4f}  95% CI=[{lo:+.4f}, {hi:+.4f}]  width={width:.4f}")
            r2_ci_rows.append({"dataset": dataset, "pool": pool_label, "n_batteries": n_batteries,
                                "point_r2": point_r2, "ci_lo": lo, "ci_hi": hi, "ci_width": width})

        # pairwise deltas, paired bootstrap (same resampled rows for both pools)
        pairs = [("42-battery (deployed)", "204-battery"),
                 ("42-battery (deployed)", "NASA-heavy"),
                 ("204-battery", "NASA-heavy")]
        for pool_a, pool_b in pairs:
            pred_a, pred_b = preds[pool_a], preds[pool_b]
            point_delta = r2(y_true, pred_b) - r2(y_true, pred_a)
            boot_deltas = np.array([
                r2(y_true[idx], pred_b[idx]) - r2(y_true[idx], pred_a[idx]) for idx in idx_sets
            ])
            lo, hi = np.percentile(boot_deltas, CI)
            significant = not (lo <= 0 <= hi)
            verdict = "SIGNIFICANT (CI excludes 0)" if significant else "NOT significant (CI includes 0)"
            print(f"    delta[{pool_b} - {pool_a}]: point={point_delta:+.4f}  "
                  f"95% CI=[{lo:+.4f}, {hi:+.4f}]  -> {verdict}")
            delta_rows.append({"dataset": dataset, "pool_a": pool_a, "pool_b": pool_b,
                                "point_delta": point_delta, "ci_lo": lo, "ci_hi": hi,
                                "significant": significant, "n_batteries": n_batteries})

    r2_ci_df = pd.DataFrame(r2_ci_rows)
    delta_df = pd.DataFrame(delta_rows)
    r2_ci_df.to_csv(OUT_DIR / "pool_comparison_bootstrap_r2_ci.csv", index=False)
    delta_df.to_csv(OUT_DIR / "pool_comparison_bootstrap_deltas.csv", index=False)

    print("\n\n=== FULL SUMMARY: R2 CI per pool/dataset ===")
    print(r2_ci_df.to_string(index=False))
    print("\n=== FULL SUMMARY: pairwise deltas ===")
    print(delta_df.to_string(index=False))


if __name__ == "__main__":
    main()
