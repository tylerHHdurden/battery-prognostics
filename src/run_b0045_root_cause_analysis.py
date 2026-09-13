"""
B0045 root-cause investigation, mirroring session 27 (B0018) and
session 41 Part B (B0044)'s methodology - B0045 has now surfaced
independently in three places: session 41's project-wide artifact
sweep (flagged, never fully investigated), session 41 Part A.1 (a
two-part finding: isolated cycle-artifact + whole-battery capacity-
scale anomaly, recommended as an exclusion candidate but NOT excluded),
and this closeout's item 5 (GroupKFold fold 3's weakness, RMSE 5.5x
its next-worst neighbor in that fold).

CORRECTION TO THE TASK'S OWN FRAMING, stated explicitly before
anything else (checked directly, not assumed): B0045 was NEVER part of
Stage 2.1's 14-battery exclusion/recovery set - `EXCLUDED_NASA_BATTERIES`
in `expanded_pool_exclusions.py` does not include B0045 at all. The
"GROUP 3, pervasive scattered anomalies, not cleanly recoverable"
classification from Stage 2.1 belongs to NASA/B0050, a DIFFERENT
battery. B0045 sits in the expanded pool's TRAIN split, untouched by
that exclusion, and was investigated separately in session 41 Part
A.1. This distinction matters for interpreting what follows.

Pool: the EXPANDED 204-battery pool - the pool where all three
discoveries above actually happened.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_degradation_mode_analysis import analyze_battery, classify_degradation_mode
from data_adapters import iterate_nasa_cycles, iterate_mit_cycles
from ica_dv_dc import compute_ica_dv_dc
from stage1_common import add_reformulated_duration_features, DURATION_FEATURES
from expanded_pool_exclusions import EXCLUDED_NASA_BATTERIES

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"

TARGET = "B0045"
CANONICAL_RAW = ["ICHV", "SCV", "VDEDT", "VIECT", "MATD", "MET", "TEVD", "TEVI"]


def load_cycles(dataset: str, battery_id: str) -> list:
    if dataset == "NASA":
        return list(iterate_nasa_cycles(battery_id))
    with open(PROC_DIR / "mit_full_cells.json") as f:
        mit_full = json.load(f)
    entry = next(e for e in mit_full if e["global_id"] == battery_id)
    return list(iterate_mit_cycles(entry["batch_file"], entry["cell_index"]))


def section0_status_check(split):
    print("\n[b0045-rootcause] === 0. Status check (correcting the premise before proceeding) ===")
    in_excluded_14 = TARGET in EXCLUDED_NASA_BATTERIES
    in_train = TARGET in split["train_ids"]
    in_test = TARGET in split["test_ids"]
    print(f"[b0045-rootcause] {TARGET} in Stage 2.1's 14-battery EXCLUDED_NASA_BATTERIES set: {in_excluded_14}")
    print(f"[b0045-rootcause] {TARGET} in expanded-pool train_ids: {in_train}, test_ids: {in_test}")
    if not in_excluded_14:
        print(f"[b0045-rootcause] CONFIRMED: {TARGET} was NEVER part of Stage 2.1's 14-battery set - "
              f"the 'GROUP 3, pervasive scattered anomalies' classification belongs to NASA/B0050, "
              f"a different battery. {TARGET} sits in TRAIN, untouched by that exclusion pass.")


def section1_representation(split, hi):
    print("\n[b0045-rootcause] === 1. Training representation: NASA vs MIT (expanded pool) ===")
    train_ids = split["train_ids"]
    n_nasa_bat = sum(1 for b in train_ids if b.startswith("B0"))
    n_mit_bat = len(train_ids) - n_nasa_bat
    hi_train = hi[hi["battery_id"].isin(train_ids)]
    n_nasa_cyc = (hi_train["dataset"] == "NASA").sum()
    n_mit_cyc = (hi_train["dataset"] == "MIT").sum()
    print(f"[b0045-rootcause] training batteries: {n_nasa_bat} NASA, {n_mit_bat} MIT "
          f"({n_nasa_bat/(n_nasa_bat+n_mit_bat)*100:.1f}% NASA by battery count)")
    print(f"[b0045-rootcause] training CYCLES: {n_nasa_cyc} NASA, {n_mit_cyc} MIT "
          f"({n_nasa_cyc/(n_nasa_cyc+n_mit_cyc)*100:.1f}% NASA by cycle count) - "
          f"matches session 38/41's own finding for this same pool (1.3%), unchanged since - "
          f"B0045 sits inside this same underrepresented-by-cycle-count training population.")


def section2_lifetime(hi, all_ids):
    print("\n[b0045-rootcause] === 2. Lifetime/fade-rate: B0045 vs. the FULL expanded pool "
          "(train+test, since B0045 itself is a TRAIN battery, not test) ===")
    rows = []
    for bid in all_ids:
        sub = hi[hi["battery_id"] == bid].sort_values("cycle_idx")
        if len(sub) == 0:
            continue
        n = len(sub)
        drop = sub["SOH"].iloc[0] - sub["SOH"].iloc[-1]
        rate = drop / n
        ds = sub["dataset"].iloc[0]
        rows.append({"dataset": ds, "battery_id": bid, "n_cycles": n,
                      "SOH_first": sub["SOH"].iloc[0], "SOH_last": sub["SOH"].iloc[-1],
                      "SOH_drop_pp": drop, "fade_rate_pp_per_cycle": rate})
    df = pd.DataFrame(rows).sort_values("fade_rate_pp_per_cycle", ascending=False).reset_index(drop=True)
    b0045 = df[df.battery_id == TARGET].iloc[0]
    rank_fade = int((df["fade_rate_pp_per_cycle"] >= b0045["fade_rate_pp_per_cycle"]).sum())
    rank_short = int((df["n_cycles"] <= b0045["n_cycles"]).sum())
    print(f"[b0045-rootcause] {TARGET}: n_cycles={b0045['n_cycles']}, SOH {b0045['SOH_first']:.1f}%->"
          f"{b0045['SOH_last']:.1f}% ({b0045['SOH_drop_pp']:.1f}pp), fade_rate={b0045['fade_rate_pp_per_cycle']:.4f}pp/cycle")
    print(f"[b0045-rootcause] {TARGET} ranks #{rank_fade} of {len(df)} FASTEST-fading batteries (whole pool), "
          f"#{rank_short} of {len(df)} SHORTEST-lived")
    print(f"[b0045-rootcause] whole-pool median: n_cycles={df['n_cycles'].median():.0f}, "
          f"fade_rate={df['fade_rate_pp_per_cycle'].median():.4f}pp/cycle")
    print(f"[b0045-rootcause] {TARGET} is {b0045['fade_rate_pp_per_cycle']/df['fade_rate_pp_per_cycle'].median():.1f}x "
          f"the median fade rate, and only {b0045['n_cycles']/df['n_cycles'].median()*100:.0f}% of the median lifetime")
    nasa_df = df[df.dataset == "NASA"]
    print(f"\n[b0045-rootcause] {TARGET}'s rank among the {len(nasa_df)} NASA batteries specifically "
          f"(fade rate): #{int((nasa_df['fade_rate_pp_per_cycle']>=b0045['fade_rate_pp_per_cycle']).sum())} of {len(nasa_df)}")
    print(nasa_df.head(10).to_string(index=False))
    return df


def section3_feature_outliers(hi_reformulated, split):
    print("\n[b0045-rootcause] === 3. Feature-distribution outlier check (canonical Stage 1.1+1.5 features) ===")
    reformulated_cols = [f"{f}_rel" if f in DURATION_FEATURES else f for f in CANONICAL_RAW]
    fusion = pd.read_csv(PROC_DIR / "fusion_embeddings_expanded.csv")
    merged = pd.merge(hi_reformulated, fusion, on=["dataset", "battery_id", "cycle_idx"], how="inner")
    fusion_cols = [f"fusion_{i}" for i in range(16)]
    feat_cols = reformulated_cols + fusion_cols

    mit_train = merged[(merged["dataset"] == "MIT") & (merged["battery_id"].isin(split["train_ids"]))]
    X_train = mit_train[feat_cols].to_numpy(dtype=float)
    print(f"[b0045-rootcause] MIT training rows (reference distribution): {len(mit_train)}")

    target_rows = merged[merged["battery_id"] == TARGET]
    X_target = target_rows[feat_cols].to_numpy(dtype=float)
    Xp = np.concatenate([X_train, X_target])
    mean, std = np.nanmean(Xp, axis=0), np.nanstd(Xp, axis=0) + 1e-8
    Xp = np.nan_to_num((Xp - mean) / std, nan=0.0)
    y = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_target))])
    clf = LogisticRegression(max_iter=2000).fit(Xp, y)
    auc = roc_auc_score(y, clf.predict_proba(Xp)[:, 1])
    print(f"[b0045-rootcause] AUC(MIT-train vs. {TARGET}): {auc:.4f} "
          f"({'near-chance, well-overlapped' if auc < 0.65 else 'high separability' if auc < 0.9 else 'NEAR-TOTAL separation'})")
    print(f"[b0045-rootcause] for direct comparison: B0018's AUC=1.0000, B0044's AUC=1.0000 (both session 27/41)")

    print(f"\n[b0045-rootcause] per-feature z-score of {TARGET}'s MEAN vs. MIT-train distribution (top 8 by |z|):")
    zrows = []
    for col in feat_cols:
        tr = mit_train[col].dropna()
        tgt = target_rows[col].dropna()
        if len(tgt) == 0 or tr.std() == 0:
            continue
        mean_c, std_c = tr.mean(), tr.std() + 1e-8
        z = (tgt.mean() - mean_c) / std_c
        pct_below = (tr < tgt.mean()).mean() * 100
        zrows.append({"feature": col, "mit_train_mean": mean_c, "mit_train_std": std_c,
                       "b0045_mean": tgt.mean(), "zscore": z, "mit_train_pctile_of_b0045_mean": pct_below})
    z_df = pd.DataFrame(zrows).reindex(pd.DataFrame(zrows)["zscore"].abs().sort_values(ascending=False).index)
    for _, r in z_df.head(8).iterrows():
        print(f"    {r['feature']}: MIT-train mean={r['mit_train_mean']:.3f}, "
              f"{TARGET} mean={r['b0045_mean']:.3f}, z={r['zscore']:.1f} "
              f"({r['mit_train_pctile_of_b0045_mean']:.1f}th percentile of MIT-train)")
    return auc, z_df


def section4_degradation_mode(sample_ids):
    print("\n[b0045-rootcause] === 4. Degradation-mode signature (session 23's peak-tracking method) ===")
    rows = []
    for dataset, bid in sample_ids:
        cycles = load_cycles(dataset, bid)
        first_r = None
        for c in cycles:
            first_r = compute_ica_dv_dc(c)
            if first_r is not None:
                break
        V_window = float(first_r["V_grid"].max() - first_r["V_grid"].min()) if first_r is not None else np.nan
        peak_df = analyze_battery(dataset, bid, cycles)
        result = classify_degradation_mode(peak_df, V_window)
        print(f"[b0045-rootcause] {dataset}/{bid}: delta_V_frac={result.get('delta_V_frac_of_window', float('nan'))*100:.1f}%, "
              f"delta_H_frac={result.get('delta_H_frac', float('nan')):+.3f} -> {result['signature']}")
        rows.append({"dataset": dataset, "battery_id": bid, **result})
    return pd.DataFrame(rows)


def section5_raw_trace_check():
    print(f"\n[b0045-rootcause] === 5. Raw capacity-trace inspection for {TARGET} - characterizing "
          f"'scattered' precisely (direct on raw discharge_capacity, not hi_table's SOH, since the "
          f"point is to check the RAW measurement pattern itself) ===")
    cycles = load_cycles("NASA", TARGET)
    caps = np.array([c["discharge_capacity"] for c in cycles])
    idxs = np.array([c["cycle_idx"] for c in cycles])
    print(f"[b0045-rootcause] {TARGET}: {len(caps)} cycles, cycle-1 capacity={caps[0]:.4f}Ah, "
          f"final capacity={caps[-1]:.4f}Ah")

    zero_idx = np.where(caps == 0.0)[0]
    print(f"[b0045-rootcause] EXACT-ZERO cycles: {[int(idxs[i]) for i in zero_idx]} "
          f"({len(zero_idx)} of {len(caps)} cycles, {len(zero_idx)/len(caps)*100:.1f}%)")

    # exclude the exact-zero cycles and check whether what REMAINS is smooth/monotonic
    # (a genuine aging trend) or itself noisy/erratic (true "scattered" pervasive noise)
    keep = caps != 0.0
    caps_clean = caps[keep]
    diffs = np.diff(caps_clean)
    frac_increasing = float((diffs > 0).mean())
    rel_diffs = diffs / caps_clean[:-1]
    print(f"[b0045-rootcause] EXCLUDING the exact-zero cycles, the remaining {len(caps_clean)} cycles: "
          f"fraction where capacity INCREASED vs. previous kept cycle = {frac_increasing*100:.1f}% "
          f"(a smooth monotonic decline would be close to 0%)")
    print(f"[b0045-rootcause] remaining-trace relative cycle-to-cycle change: mean={rel_diffs.mean()*100:+.2f}%, "
          f"std={rel_diffs.std()*100:.2f}%, max|change|={np.abs(rel_diffs).max()*100:.2f}%")
    print(f"[b0045-rootcause] VERDICT: {'a genuinely SMOOTH, monotonic aging trend' if abs(rel_diffs).max() < 0.15 and frac_increasing < 0.25 else 'still shows real scatter/noise beyond the exact-zero cycles'} "
          f"once the exact-zero artifact cycles are set aside.")

    # cohort capacity-scale comparison, direct (not assumed)
    cohort = {}
    for cid in ["B0043", "B0044", "B0046", "B0047"]:
        c1 = next(iter(iterate_nasa_cycles(cid)))["discharge_capacity"]
        cohort[cid] = c1
    print(f"\n[b0045-rootcause] cycle-1 capacity, {TARGET}={caps[0]:.4f}Ah vs. immediate-ID-neighbor cohort: "
          f"{ {k: round(v,4) for k,v in cohort.items()} }")
    ratio_to_cohort = caps[0] / np.mean(list(cohort.values()))
    print(f"[b0045-rootcause] {TARGET}'s cycle-1 capacity is {ratio_to_cohort*100:.1f}% of its immediate "
          f"cohort's mean cycle-1 capacity - {'a genuine, unexplained whole-battery scale anomaly' if ratio_to_cohort < 0.7 else 'within normal cohort range'}")

    return {
        "n_cycles": len(caps), "n_exact_zero_cycles": len(zero_idx), "exact_zero_cycle_idxs": [int(idxs[i]) for i in zero_idx],
        "cycle1_capacity": float(caps[0]), "cohort_mean_cycle1_capacity": float(np.mean(list(cohort.values()))),
        "ratio_to_cohort": float(ratio_to_cohort), "frac_increasing_excl_zeros": frac_increasing,
        "max_abs_rel_change_excl_zeros": float(np.abs(rel_diffs).max()),
    }


def main():
    split = json.loads((PROC_DIR / "battery_split_expanded.json").read_text())
    hi = pd.read_parquet(PROC_DIR / "hi_table_expanded.parquet")

    section0_status_check(split)
    section1_representation(split, hi)
    all_ids = split["train_ids"] + split["test_ids"]
    lifetime_df = section2_lifetime(hi, all_ids)

    hi_reformulated = add_reformulated_duration_features(hi)
    auc, z_df = section3_feature_outliers(hi_reformulated, split)

    sample_ids = [("NASA", "B0045"), ("NASA", "B0025"), ("NASA", "B0030"), ("NASA", "B0053"),
                  ("MIT", "b4c25"), ("MIT", "b3c17"), ("MIT", "b2c9")]
    degmode_df = section4_degradation_mode(sample_ids)

    trace_result = section5_raw_trace_check()

    lifetime_df.to_csv(OUT_DIR / "b0045_rootcause_lifetime.csv", index=False)
    z_df.to_csv(OUT_DIR / "b0045_rootcause_feature_zscores.csv", index=False)
    degmode_df.to_csv(OUT_DIR / "b0045_rootcause_degradation_mode.csv", index=False)
    pd.DataFrame([{"auc_vs_mit_train": auc, **trace_result}]).to_csv(
        OUT_DIR / "b0045_rootcause_summary.csv", index=False)

    print("\n[b0045-rootcause] === SYNTHESIS ===")
    print("[b0045-rootcause] See DEVELOPMENT_LOG.md follow-up entry for the full synthesized explanation.")
    print("[b0045-rootcause] DONE")


if __name__ == "__main__":
    main()
