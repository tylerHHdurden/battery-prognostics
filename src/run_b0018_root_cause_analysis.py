"""
Session 27: root-cause investigation for NASA/B0018 - flagged as this
pipeline's consistent reliability edge case by 3 independent, unrelated
analyses (the early-prediction test, session 25's second-life grading,
session 26's sensor-noise robustness), none of which asked WHY. Pure
analysis on existing data/predictions/scripts - no retraining. Reuses
session 23's degradation-mode functions (`run_degradation_mode_analysis`,
imported not duplicated) and session 19's domain-classifier methodology
(reimplemented here on train-vs-B0018 rather than train-vs-CALCE, since
session 19's script is specific to the MMD-aligned CALCE pipeline).

4 independent angles requested, each answered from existing project
data with no new model training:
  1. Training representation (battery_split.json + hi_table.parquet)
  2. Lifetime/protocol characteristics (hi_table.parquet)
  3. Feature-distribution outlier check (hi_table.parquet + fusion_embeddings.csv,
     session-19-style logistic domain classifier, WITH proper controls -
     every MIT test battery checked the same way, not just B0018, since
     session 19 already established that even in-domain held-out
     batteries look somewhat separable at this small battery count)
  4. Degradation-mode signature (session 23's peak-tracking, extended
     here to ALL 5 MIT test batteries - session 23 only covered 2 of
     them - for a complete, not partial, comparison)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_degradation_mode_analysis import load_cycles, analyze_battery, classify_degradation_mode
from ica_dv_dc import compute_ica_dv_dc

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
PRED_DIR = PROC_DIR / "predictions"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

TEST_BATTERIES = [("NASA", "B0018"), ("MIT", "b1c4"), ("MIT", "b2c24"),
                   ("MIT", "b3c0"), ("MIT", "b3c35"), ("MIT", "b4c38")]


def section1_representation(split, hi):
    print("\n[rootcause] === 1. Training representation: NASA vs MIT ===")
    train_ids = split["train_ids"]
    bat_ds_map = dict(zip(hi.drop_duplicates(["dataset", "battery_id"])["battery_id"],
                           hi.drop_duplicates(["dataset", "battery_id"])["dataset"]))
    train_ds = [bat_ds_map.get(b, "?") for b in train_ids]
    n_nasa_train_bat = train_ds.count("NASA")
    n_mit_train_bat = train_ds.count("MIT")
    hi_train = hi[hi["battery_id"].isin(train_ids)]
    n_nasa_cyc = (hi_train["dataset"] == "NASA").sum()
    n_mit_cyc = (hi_train["dataset"] == "MIT").sum()
    print(f"[rootcause] training batteries: {n_nasa_train_bat} NASA, {n_mit_train_bat} MIT "
          f"({n_nasa_train_bat/(n_nasa_train_bat+n_mit_train_bat)*100:.1f}% NASA by battery count)")
    print(f"[rootcause] training CYCLES: {n_nasa_cyc} NASA, {n_mit_cyc} MIT "
          f"({n_nasa_cyc/(n_nasa_cyc+n_mit_cyc)*100:.1f}% NASA by cycle count - "
          f"NASA is far MORE underrepresented by cycle count than by battery count, "
          f"since NASA batteries are individually much shorter-lived)")
    return {"n_nasa_train_batteries": n_nasa_train_bat, "n_mit_train_batteries": n_mit_train_bat,
            "n_nasa_train_cycles": int(n_nasa_cyc), "n_mit_train_cycles": int(n_mit_cyc)}


def section2_lifetime(hi):
    print("\n[rootcause] === 2. Lifetime/fade-rate characteristics per test battery ===")
    rows = []
    for dataset, bid in TEST_BATTERIES:
        sub = hi[hi["battery_id"] == bid].sort_values("cycle_idx")
        n = len(sub)
        drop = sub["SOH"].iloc[0] - sub["SOH"].iloc[-1]
        rate = drop / n
        rows.append({"dataset": dataset, "battery_id": bid, "n_cycles": n,
                      "SOH_first": sub["SOH"].iloc[0], "SOH_last": sub["SOH"].iloc[-1],
                      "SOH_drop_pp": drop, "fade_rate_pp_per_cycle": rate})
        print(f"[rootcause] {dataset}/{bid}: n_cycles={n}, SOH {sub['SOH'].iloc[0]:.1f}%->"
              f"{sub['SOH'].iloc[-1]:.1f}% ({drop:.1f}pp), fade_rate={rate:.4f}pp/cycle")
    df = pd.DataFrame(rows)
    b0018_rate = df[df.battery_id == "B0018"]["fade_rate_pp_per_cycle"].iloc[0]
    fastest_mit_rate = df[df.dataset == "MIT"]["fade_rate_pp_per_cycle"].max()
    print(f"[rootcause] B0018 fades {b0018_rate/fastest_mit_rate:.1f}x FASTER per cycle than "
          f"even the fastest-fading MIT test battery ({b0018_rate:.4f} vs {fastest_mit_rate:.4f} pp/cycle) "
          f"despite having by far the FEWEST total cycles ({df[df.battery_id=='B0018']['n_cycles'].iloc[0]}) "
          f"of any test battery - consistent with NASA's standard low-rate cycling protocol vs. "
          f"MIT's fast-charging-study protocol (well-documented characteristic of these specific "
          f"published datasets, not independently re-derived here)")
    return df


def section3_feature_outliers(hi, split):
    print("\n[rootcause] === 3. Feature-distribution outlier check (session-19-style domain classifier) ===")
    with open(PROC_DIR / "bfa_selected_features.txt") as f:
        BFA = [l.strip() for l in f if l.strip()]
    fusion = pd.read_csv(PROC_DIR / "fusion_embeddings.csv")
    merged = pd.merge(hi, fusion, on=["dataset", "battery_id", "cycle_idx"], how="inner")
    fusion_cols = [f"fusion_{i}" for i in range(16)]
    feat_cols = BFA + fusion_cols

    mit_train = merged[(merged["dataset"] == "MIT") & (merged["battery_id"].isin(split["train_ids"]))]
    X_train = mit_train[feat_cols].to_numpy(dtype=float)

    print(f"[rootcause] MIT training rows (the reference distribution): {len(mit_train)}")
    print(f"[rootcause] AUC(MIT-train vs. X) for EVERY test battery, not just B0018 - "
          f"session 19 already showed held-out battery separability is elevated even IN-domain "
          f"at this battery count, so B0018 needs to be checked against that baseline, not zero:")
    auc_rows = []
    for dataset, bid in TEST_BATTERIES:
        X_target = merged[merged["battery_id"] == bid][feat_cols].to_numpy(dtype=float)
        Xp = np.concatenate([X_train, X_target])
        mean, std = Xp.mean(0), Xp.std(0) + 1e-8
        Xp_z = (Xp - mean) / std
        y = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_target))])
        clf = LogisticRegression(max_iter=2000).fit(Xp_z, y)
        auc = roc_auc_score(y, clf.predict_proba(Xp_z)[:, 1])
        auc_rows.append({"dataset": dataset, "battery_id": bid, "n": len(X_target), "auc_vs_mit_train": auc})
        print(f"    {dataset}/{bid}: n={len(X_target)}, AUC={auc:.4f}")
    auc_df = pd.DataFrame(auc_rows)

    b0018 = merged[merged["battery_id"] == "B0018"]
    print(f"\n[rootcause] per-feature z-score of B0018's MEAN vs. MIT-train distribution "
          f"(top 5 by |z|, identifying WHICH features drive the separation):")
    zrows = []
    for col in feat_cols:
        tr = mit_train[col].dropna()
        tgt = b0018[col].dropna()
        mean, std = tr.mean(), tr.std() + 1e-8
        z = (tgt.mean() - mean) / std
        pct_below = (tr < tgt.mean()).mean() * 100
        zrows.append({"feature": col, "mit_train_mean": mean, "mit_train_std": std,
                       "b0018_mean": tgt.mean(), "zscore": z, "mit_train_pctile_of_b0018_mean": pct_below})
    z_df = pd.DataFrame(zrows).reindex(pd.DataFrame(zrows)["zscore"].abs().sort_values(ascending=False).index)
    for _, r in z_df.head(5).iterrows():
        print(f"    {r['feature']}: MIT-train mean={r['mit_train_mean']:.3f}, "
              f"B0018 mean={r['b0018_mean']:.3f}, z={r['zscore']:.1f} "
              f"({r['mit_train_pctile_of_b0018_mean']:.1f}th percentile of MIT-train)")
    return auc_df, z_df


def section4_degradation_mode():
    print("\n[rootcause] === 4. Degradation-mode signature - completing session 23's coverage to all 6 ===")
    rows = []
    for dataset, bid in TEST_BATTERIES:
        cycles = load_cycles(dataset, bid)
        first_r = None
        for c in cycles:
            first_r = compute_ica_dv_dc(c)
            if first_r is not None:
                break
        V_window = float(first_r["V_grid"].max() - first_r["V_grid"].min()) if first_r is not None else np.nan
        peak_df = analyze_battery(dataset, bid, cycles)
        result = classify_degradation_mode(peak_df, V_window)
        print(f"[rootcause] {dataset}/{bid}: delta_V_frac={result.get('delta_V_frac_of_window', float('nan'))*100:.1f}%, "
              f"delta_H_frac={result.get('delta_H_frac', float('nan')):+.3f} -> {result['signature']}")
        rows.append({"dataset": dataset, "battery_id": bid, **result})
    return pd.DataFrame(rows)


def main():
    split = json.loads((PROC_DIR / "battery_split.json").read_text())
    hi = pd.read_parquet(PROC_DIR / "hi_table.parquet")

    rep = section1_representation(split, hi)
    lifetime_df = section2_lifetime(hi)
    auc_df, z_df = section3_feature_outliers(hi, split)
    degmode_df = section4_degradation_mode()

    lifetime_df.to_csv(OUT_DIR / "b0018_rootcause_lifetime.csv", index=False)
    auc_df.to_csv(OUT_DIR / "b0018_rootcause_domain_auc.csv", index=False)
    z_df.to_csv(OUT_DIR / "b0018_rootcause_feature_zscores.csv", index=False)
    degmode_df.to_csv(OUT_DIR / "b0018_rootcause_degradation_mode.csv", index=False)

    print("\n[rootcause] === SYNTHESIS ===")
    print("[rootcause] See DEVELOPMENT_LOG.md session 27 for the full synthesized explanation.")
    print("[rootcause] DONE")


if __name__ == "__main__":
    main()
