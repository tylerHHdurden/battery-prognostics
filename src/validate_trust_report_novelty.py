"""
Two validations the trust report never had (added 2026-09-30, after the HNEI upload came back "nearest source snl,
familiar"): the existing validation only scores held-out batteries of sources that HAVE a profile, using
whole-battery vectors. That says nothing about the two cases the app actually needs the report for.

A. LEAVE-SOURCE-OUT (novelty): for every source S, score all of S's batteries against the profiles of every OTHER
   source (S's whole sibling family removed from the candidate set - e.g. mich also removes mich_exp). A battery
   from a source the report has never seen SHOULD come back "somewhat familiar"/"unfamiliar"; "familiar" here is a
   false reassurance. (Profile parameters/thresholds are the saved ones; the pooled standardization scale did see S -
   a small, disclosed leak that cannot make novelty detection look worse.)
B. PARTIAL HISTORY: score held-out batteries using only their FIRST N cycles (median over those cycles), as an
   upload of a short recording would be, against all 16 profiles: nearest-source accuracy and trust level.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_toolkit_phase2b_federated import load_pooled_data, ALL_SOURCES, FAMILIES
from build_source_trust_profiles import split_with_fallback
import trust_report as tr

ROOT = Path(__file__).resolve().parent.parent
FIRST_N = [30, 100]


def verdict(x_std, profiles, exclude):
    cands = {n: p for n, p in profiles.items() if n not in exclude}
    nll = {n: tr._neg_log_likelihood(x_std, p["mean"], p["precision"], p["logdet_cov"]) for n, p in cands.items()}
    nearest = min(nll, key=nll.get)
    p = cands[nearest]
    d = tr._mahalanobis(x_std, p["mean"], p["precision"])
    lvl = "familiar" if d <= p["threshold_familiar"] else ("somewhat_familiar" if d <= p["threshold_somewhat_familiar"] else "unfamiliar")
    verdict.last_ratio = d / p["threshold_familiar"]  # distance in units of the matched source's own "familiar" limit
    return nearest, lvl, d


def main():
    data = tr._load()
    profiles = data["profiles"]
    sources, all_cols, _ = load_pooled_data()
    med, sm, ss = data["col_medians"], data["scale_mean"], data["scale_std"]

    def vec(df_rows):
        v = np.median(df_rows[all_cols].to_numpy(float), axis=0)
        return tr._prepare_vector(v, med, sm, ss)

    # ---------- A: leave-source-out ----------
    rows = []
    for S in ALL_SOURCES:
        fam = FAMILIES[S]
        for bid, sub in sources[S].groupby("battery_id"):
            nearest, lvl, d = verdict(vec(sub), profiles, exclude=fam)
            rows.append({"held_out_source": S, "battery_id": bid, "nearest_remaining": nearest, "trust_level": lvl,
                         "distance": d, "ratio": verdict.last_ratio})
    a = pd.DataFrame(rows)
    a["false_reassurance"] = a["trust_level"] == "familiar"
    a.to_csv(ROOT / "outputs" / "toolkit_phase3_trust_leave_source_out.csv", index=False)
    sa = a.groupby("held_out_source").agg(n=("battery_id", "size"), familiar=("false_reassurance", "sum"),
                                          frac_flagged_novel=("false_reassurance", lambda s: 1 - s.mean()))
    print("=== A. LEAVE-SOURCE-OUT: fraction of a NOVEL source's batteries flagged somewhat-familiar/unfamiliar ===")
    print(sa.round(3).to_string())
    print(f"OVERALL: {1 - a['false_reassurance'].mean():.1%} of {len(a)} novel-source batteries flagged; "
          f"{int(a['false_reassurance'].sum())} falsely 'familiar'\n")

    # ---------- B: partial history ----------
    brow = []
    for S in ALL_SOURCES:
        bids = sorted(sources[S]["battery_id"].unique())
        _, test_ids = split_with_fallback(bids, test_every=5)
        for bid in test_ids:
            sub = sources[S][sources[S]["battery_id"] == bid].sort_values("cycle_idx")
            for n in FIRST_N:
                part = sub.head(n)
                if len(part) < 3:
                    continue
                nearest, lvl, d = verdict(vec(part), profiles, exclude=set())
                brow.append({"true_source": S, "battery_id": bid, "first_n": n, "n_cycles_used": len(part),
                             "nearest": nearest, "correct": nearest == S, "trust_level": lvl})
    b = pd.DataFrame(brow)
    b.to_csv(ROOT / "outputs" / "toolkit_phase3_trust_partial_history.csv", index=False)
    print("=== B. PARTIAL HISTORY (held-out batteries, first N cycles only) ===")
    for n, g in b.groupby("first_n"):
        print(f"first {n} cycles: nearest-source accuracy {g['correct'].mean():.1%} ({int(g['correct'].sum())}/{len(g)}); "
              f"trust levels: {g['trust_level'].value_counts().to_dict()}")
    hn = b[(b["true_source"] == "hnei") & (b["first_n"] == 30)]
    print("hnei first-30 rows:", hn[["battery_id", "nearest", "trust_level"]].to_string(index=False))

    # ---------- C: threshold trade-off (flag if distance > c x matched source's own familiar limit) ----------
    known = []
    for S in ALL_SOURCES:
        bids = sorted(sources[S]["battery_id"].unique())
        _, test_ids = split_with_fallback(bids, test_every=5)
        for bid in test_ids:
            sub = sources[S][sources[S]["battery_id"] == bid]
            verdict(vec(sub), profiles, exclude=set())
            known.append(verdict.last_ratio)
    known = np.array(known); novel = a["ratio"].to_numpy()
    print("
=== C. TRADE-OFF: flag as not-familiar if distance > c x (matched source's own 95th-pct limit) ===")
    print(f"{'c':>5} | novel-source batteries flagged (want high) | known-source held-out batteries flagged (false alarms, want low)")
    rows_c = []
    for c in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25]:
        rows_c.append({"c": c, "novel_flagged": float((novel > c).mean()), "known_false_alarm": float((known > c).mean())})
        print(f"{c:>5} | {(novel > c).mean():>40.1%} | {(known > c).mean():>50.1%}")
    pd.DataFrame(rows_c).to_csv(ROOT / "outputs" / "toolkit_phase3_trust_threshold_tradeoff.csv", index=False)


if __name__ == "__main__":
    main()
