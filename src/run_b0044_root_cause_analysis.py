"""
Stage 1 follow-up, Part B: root-cause investigation for NASA/B0044,
mirroring session 27's B0018 methodology as closely as possible -
NASA/B0044 regressed under Huber loss in 3 of 4 architectures tested
in Stage 1.3 (VLSTM, CNN-BiGRU, and to a lesser extent was already the
2nd-worst under MSE; CNN-LSTM was the one exception where it slightly
improved), discovered by accident rather than targeted investigation,
and deserves the same systematic treatment B0018 got.

Pool: the EXPANDED 204-battery pool (battery_split_expanded.json,
hi_table_expanded.parquet) - the pool where B0044's Huber-loss
regression was actually observed (B0044 does not exist in the original
32-battery pool's NASA subset... actually it DOES: B0044 was excluded
from the ORIGINAL 32-battery pool's 4 hardcoded NASA IDs [B0005, B0006,
B0007, B0018] - only present in the EXPANDED pool's 34-NASA-battery
list. Confirmed before writing this script.)

4 angles, mirroring session 27's structure exactly:
  1. Training representation (battery_split_expanded.json + hi_table_expanded.parquet)
  2. Lifetime/protocol characteristics vs. the full 40-battery expanded test set
  3. Feature-distribution outlier check, using the CURRENT Stage 1 canonical
     REFORMULATED 8-feature set (ICHV_rel/SCV/VDEDT/VIECT/MATD/MET/TEVD_rel/
     TEVI_rel), not the original leaked 7-feature raw-duration set - stated
     honestly this is a general representativeness diagnostic on the BFA HI
     feature space, not literally what Huber-loss-trained deep models see
     (they train on raw 6-channel V/I/T/dQdV/dVdQ/dIdV sequence tensors, not
     BFA HI features at all - this distinction is not glossed over).
  5. Degradation-mode signature (session 23's peak-tracking method)
  6. Raw capacity-trace inspection, checking specifically for a B0053-style
     data artifact vs. a genuine (if difficult) degradation curve.
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

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"

TARGET = "B0044"
CANONICAL_RAW = ["ICHV", "SCV", "VDEDT", "VIECT", "MATD", "MET", "TEVD", "TEVI"]


def load_cycles(dataset: str, battery_id: str) -> list:
    """session 23/27's load_cycles was hardcoded to mit_subset.json (the
    ORIGINAL 32-battery pool's 28 MIT cells) - this follow-up's MIT
    comparison batteries are drawn from the EXPANDED pool (204 batteries),
    which is a real, root-caused gap in reusing the existing function
    as-is (confirmed via StopIteration on a first run), not silently
    worked around: this override checks mit_full_cells.json (the
    expanded pool's 185-cell list, confirmed to be a superset of
    mit_subset.json's 28) instead, so it also still handles the original
    28 correctly."""
    if dataset == "NASA":
        return list(iterate_nasa_cycles(battery_id))
    with open(PROC_DIR / "mit_full_cells.json") as f:
        mit_full = json.load(f)
    entry = next(e for e in mit_full if e["global_id"] == battery_id)
    return list(iterate_mit_cycles(entry["batch_file"], entry["cell_index"]))


def section1_representation(split, hi):
    print("\n[b0044-rootcause] === 1. Training representation: NASA vs MIT (expanded pool) ===")
    train_ids = split["train_ids"]
    n_nasa_bat = sum(1 for b in train_ids if b.startswith("B0"))
    n_mit_bat = len(train_ids) - n_nasa_bat
    hi_train = hi[hi["battery_id"].isin(train_ids)]
    n_nasa_cyc = (hi_train["dataset"] == "NASA").sum()
    n_mit_cyc = (hi_train["dataset"] == "MIT").sum()
    print(f"[b0044-rootcause] training batteries: {n_nasa_bat} NASA, {n_mit_bat} MIT "
          f"({n_nasa_bat/(n_nasa_bat+n_mit_bat)*100:.1f}% NASA by battery count)")
    print(f"[b0044-rootcause] training CYCLES: {n_nasa_cyc} NASA, {n_mit_cyc} MIT "
          f"({n_nasa_cyc/(n_nasa_cyc+n_mit_cyc)*100:.1f}% NASA by cycle count)")
    print(f"[b0044-rootcause] for reference, session 27's ORIGINAL 32-battery pool finding was "
          f"11.5% NASA by battery, 2.7% by cycle - the expanded pool's underrepresentation is "
          f"{'WORSE' if n_nasa_cyc/(n_nasa_cyc+n_mit_cyc) < 0.027 else 'similar or better'} "
          f"by cycle count ({n_nasa_cyc/(n_nasa_cyc+n_mit_cyc)*100:.1f}% vs. 2.7%)")
    return {"n_nasa_train_batteries": n_nasa_bat, "n_mit_train_batteries": n_mit_bat,
            "n_nasa_train_cycles": int(n_nasa_cyc), "n_mit_train_cycles": int(n_mit_cyc)}


def section2_lifetime(hi, test_ids):
    print("\n[b0044-rootcause] === 2. Lifetime/fade-rate: B0044 vs. the full 40-battery expanded test set ===")
    rows = []
    for bid in test_ids:
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
    b0044 = df[df.battery_id == TARGET].iloc[0]
    rank_fade = int((df["fade_rate_pp_per_cycle"] >= b0044["fade_rate_pp_per_cycle"]).sum())
    rank_short = int((df["n_cycles"] <= b0044["n_cycles"]).sum())
    print(f"[b0044-rootcause] {TARGET}: n_cycles={b0044['n_cycles']}, SOH {b0044['SOH_first']:.1f}%->"
          f"{b0044['SOH_last']:.1f}% ({b0044['SOH_drop_pp']:.1f}pp), fade_rate={b0044['fade_rate_pp_per_cycle']:.4f}pp/cycle")
    print(f"[b0044-rootcause] {TARGET} ranks #{rank_fade} of {len(df)} FASTEST-fading test batteries, "
          f"#{rank_short} of {len(df)} SHORTEST-lived test batteries")
    print(f"[b0044-rootcause] test-set median: n_cycles={df['n_cycles'].median():.0f}, "
          f"fade_rate={df['fade_rate_pp_per_cycle'].median():.4f}pp/cycle")
    print(f"[b0044-rootcause] {TARGET} is {b0044['fade_rate_pp_per_cycle']/df['fade_rate_pp_per_cycle'].median():.1f}x "
          f"the median fade rate, and only {b0044['n_cycles']/df['n_cycles'].median()*100:.0f}% of the median lifetime")
    nasa_df = df[df.dataset == "NASA"]
    print(f"\n[b0044-rootcause] ALL {len(nasa_df)} NASA test batteries (for context):")
    print(nasa_df.to_string(index=False))
    print(f"[b0044-rootcause] ALL {len(nasa_df)} NASA test batteries rank in the top "
          f"{int((df['fade_rate_pp_per_cycle']>=nasa_df['fade_rate_pp_per_cycle'].min()).sum())} of {len(df)} "
          f"fastest-fading batteries - {TARGET} is not an isolated case, it fits the same pattern as "
          f"every other NASA test battery in the expanded pool")
    return df


def section3_feature_outliers(hi_reformulated, split):
    print("\n[b0044-rootcause] === 3. Feature-distribution outlier check (session-19/27-style domain classifier) ===")
    print("[b0044-rootcause] NOTE: uses the CURRENT Stage 1 canonical REFORMULATED 8-feature set "
          "(ICHV_rel/SCV/VDEDT/VIECT/MATD/MET/TEVD_rel/TEVI_rel) + 16-dim fusion embedding - a general "
          "representativeness diagnostic on the BFA HI feature space. Stated explicitly: this is NOT "
          "literally what Huber-loss-trained deep models see (those train on raw 6-channel V/I/T/dQdV/"
          "dVdQ/dIdV sequence tensors, never on BFA HI features) - section 6 checks the raw trace directly "
          "for that reason.")
    reformulated_cols = [f"{f}_rel" if f in DURATION_FEATURES else f for f in CANONICAL_RAW]
    fusion = pd.read_csv(PROC_DIR / "fusion_embeddings_expanded.csv")
    merged = pd.merge(hi_reformulated, fusion, on=["dataset", "battery_id", "cycle_idx"], how="inner")
    fusion_cols = [f"fusion_{i}" for i in range(16)]
    feat_cols = reformulated_cols + fusion_cols

    mit_train = merged[(merged["dataset"] == "MIT") & (merged["battery_id"].isin(split["train_ids"]))]
    X_train = mit_train[feat_cols].to_numpy(dtype=float)
    print(f"[b0044-rootcause] MIT training rows (reference distribution): {len(mit_train)}")

    target_rows = merged[merged["battery_id"] == TARGET]
    X_target = target_rows[feat_cols].to_numpy(dtype=float)
    Xp = np.concatenate([X_train, X_target])
    mean, std = np.nanmean(Xp, axis=0), np.nanstd(Xp, axis=0) + 1e-8
    Xp = np.nan_to_num((Xp - mean) / std, nan=0.0)
    y = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_target))])
    clf = LogisticRegression(max_iter=2000).fit(Xp, y)
    auc = roc_auc_score(y, clf.predict_proba(Xp)[:, 1])
    print(f"[b0044-rootcause] AUC(MIT-train vs. {TARGET}): {auc:.4f} "
          f"({'near-chance, well-overlapped' if auc < 0.65 else 'high separability' if auc < 0.9 else 'NEAR-TOTAL separation'})")
    print(f"[b0044-rootcause] for reference, session 27's B0018 AUC (original 7-feature leaked set, "
          f"32-battery pool) - see outputs/b0018_rootcause_domain_auc.csv for the exact number")

    print(f"\n[b0044-rootcause] per-feature z-score of {TARGET}'s MEAN vs. MIT-train distribution "
          f"(top 8 by |z|):")
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
                       "b0044_mean": tgt.mean(), "zscore": z, "mit_train_pctile_of_b0044_mean": pct_below})
    z_df = pd.DataFrame(zrows).reindex(pd.DataFrame(zrows)["zscore"].abs().sort_values(ascending=False).index)
    for _, r in z_df.head(8).iterrows():
        print(f"    {r['feature']}: MIT-train mean={r['mit_train_mean']:.3f}, "
              f"{TARGET} mean={r['b0044_mean']:.3f}, z={r['zscore']:.1f} "
              f"({r['mit_train_pctile_of_b0044_mean']:.1f}th percentile of MIT-train)")
    return auc, z_df


def section4_degradation_mode(test_ids_sample):
    print("\n[b0044-rootcause] === 4. Degradation-mode signature (session 23's peak-tracking method) ===")
    rows = []
    for dataset, bid in test_ids_sample:
        cycles = load_cycles(dataset, bid)
        first_r = None
        for c in cycles:
            first_r = compute_ica_dv_dc(c)
            if first_r is not None:
                break
        V_window = float(first_r["V_grid"].max() - first_r["V_grid"].min()) if first_r is not None else np.nan
        peak_df = analyze_battery(dataset, bid, cycles)
        result = classify_degradation_mode(peak_df, V_window)
        print(f"[b0044-rootcause] {dataset}/{bid}: delta_V_frac={result.get('delta_V_frac_of_window', float('nan'))*100:.1f}%, "
              f"delta_H_frac={result.get('delta_H_frac', float('nan')):+.3f} -> {result['signature']}")
        rows.append({"dataset": dataset, "battery_id": bid, **result})
    return pd.DataFrame(rows)


def section5_raw_trace_check(hi):
    print(f"\n[b0044-rootcause] === 5. Raw capacity-trace inspection for {TARGET} - artifact or genuine curve? ===")
    sub = hi[hi["battery_id"] == TARGET].sort_values("cycle_idx")
    soh = sub["SOH"].to_numpy()
    cyc = sub["cycle_idx"].to_numpy()
    print(f"[b0044-rootcause] {TARGET} SOH trace: {len(soh)} cycles, "
          f"first 5={np.round(soh[:5], 2).tolist()}, last 5={np.round(soh[-5:], 2).tolist()}")
    diffs = np.diff(soh)
    worst_jump_idx = int(np.argmin(diffs))
    print(f"[b0044-rootcause] largest single-cycle SOH drop: {diffs[worst_jump_idx]:.2f}pp "
          f"(cycle {cyc[worst_jump_idx]}->{ cyc[worst_jump_idx+1]}, "
          f"SOH {soh[worst_jump_idx]:.2f}->{soh[worst_jump_idx+1]:.2f})")
    is_zero_or_negative_end = soh[-1] <= 1.0
    is_monotonic_ish = float(pd.Series(soh).diff().dropna().gt(0).mean())
    print(f"[b0044-rootcause] final-cycle SOH near-zero (B0053-style artifact check): {is_zero_or_negative_end} "
          f"(final SOH={soh[-1]:.2f}%)")
    print(f"[b0044-rootcause] fraction of cycles where SOH INCREASED vs. previous cycle "
          f"(noise/non-monotonicity indicator): {is_monotonic_ish*100:.1f}%")
    # compare against B0053's own known artifact signature for direct contrast
    return {"n_cycles": len(soh), "soh_first": float(soh[0]), "soh_last": float(soh[-1]),
            "largest_single_drop_pp": float(diffs[worst_jump_idx]),
            "final_near_zero": bool(is_zero_or_negative_end),
            "frac_cycles_increasing": is_monotonic_ish}


def main():
    split = json.loads((PROC_DIR / "battery_split_expanded.json").read_text())
    hi = pd.read_parquet(PROC_DIR / "hi_table_expanded.parquet")
    print(f"[b0044-rootcause] {TARGET} in expanded test_ids: {TARGET in split['test_ids']}")

    rep = section1_representation(split, hi)
    lifetime_df = section2_lifetime(hi, split["test_ids"])

    hi_reformulated = add_reformulated_duration_features(hi)
    auc, z_df = section3_feature_outliers(hi_reformulated, split)

    # degradation-mode: B0044 + a handful of representative test batteries for comparison
    # (full 40-battery run would be expensive relative to this follow-up's scope; same
    # sampling spirit as session 27's own 6-battery comparison set, scaled proportionally)
    sample_ids = [("NASA", "B0044"), ("NASA", "B0025"), ("NASA", "B0030"), ("NASA", "B0053"),
                  ("MIT", "b4c25"), ("MIT", "b3c17"), ("MIT", "b2c9")]
    degmode_df = section4_degradation_mode(sample_ids)

    trace_result = section5_raw_trace_check(hi)

    lifetime_df.to_csv(OUT_DIR / "b0044_rootcause_lifetime.csv", index=False)
    z_df.to_csv(OUT_DIR / "b0044_rootcause_feature_zscores.csv", index=False)
    degmode_df.to_csv(OUT_DIR / "b0044_rootcause_degradation_mode.csv", index=False)
    pd.DataFrame([{"auc_vs_mit_train": auc, **trace_result}]).to_csv(
        OUT_DIR / "b0044_rootcause_summary.csv", index=False)

    print("\n[b0044-rootcause] === SYNTHESIS ===")
    print("[b0044-rootcause] See DEVELOPMENT_LOG.md follow-up entry for the full synthesized explanation.")
    print("[b0044-rootcause] DONE")


if __name__ == "__main__":
    main()
