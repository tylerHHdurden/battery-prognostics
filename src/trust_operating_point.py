"""
Step 3a (2026-09-30): ROC of "novel source vs known source" for the trust report.

NOVEL   = every battery of a source, scored against the profiles of every OTHER source (its whole sibling family
          removed) - a battery from a source the report has never seen (leave-source-out; profiles' pooled standardization
          scale did see the source, a small disclosed leak that cannot make novelty detection look worse).
KNOWN   = each source's held-out test batteries (split_with_fallback, test_every=5), scored against all 16 profiles.
Several candidate novelty scores are compared by AUC; the best is used. Operating point = the threshold with the lowest
false-alarm rate among those flagging >= 80% of novel-source batteries. If that costs too many false alarms the report
says so and the app uses graded messaging only.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_toolkit_phase2b_federated import load_pooled_data, ALL_SOURCES, FAMILIES
from build_source_trust_profiles import split_with_fallback
import trust_report as tr
import encoder_provenance as ep

ROOT = Path(__file__).resolve().parent.parent
TARGET_TPR = 0.80
MAX_ACCEPTABLE_FPR = 0.35  # stated cut: above this, "reach 80% detection" is declared impractical -> graded messaging only


def scores(x_std, profiles, exclude):
    cands = {n: p for n, p in profiles.items() if n not in exclude}
    nll = {n: tr._neg_log_likelihood(x_std, p["mean"], p["precision"], p["logdet_cov"]) for n, p in cands.items()}
    maha = {n: tr._mahalanobis(x_std, p["mean"], p["precision"]) for n, p in cands.items()}
    nearest = min(nll, key=nll.get)
    p = cands[nearest]
    d = maha[nearest]
    return {"nearest": nearest, "d_nearest": d, "ratio_familiar": d / p["threshold_familiar"],
            "ratio_somewhat": d / p["threshold_somewhat_familiar"], "min_maha": min(maha.values()), "nll_min": nll[nearest],
            "level": ("familiar" if d <= p["threshold_familiar"] else ("somewhat_familiar" if d <= p["threshold_somewhat_familiar"] else "unfamiliar"))}


def main():
    data = tr._load()
    profiles = data["profiles"]
    sources, all_cols, _ = load_pooled_data()
    med, sm, ss = data["col_medians"], data["scale_mean"], data["scale_std"]
    vec = lambda df: tr._prepare_vector(np.median(df[all_cols].to_numpy(float), axis=0), med, sm, ss)

    rows = []
    for S in ALL_SOURCES:
        bids = sorted(sources[S]["battery_id"].unique())
        _, test_ids = split_with_fallback(bids, test_every=5)
        for bid, sub in sources[S].groupby("battery_id"):
            x = vec(sub)
            r = scores(x, profiles, exclude=FAMILIES[S])
            rows.append({"set": "novel", "source": S, "battery_id": bid, **r})
            if bid in set(test_ids):
                r2 = scores(x, profiles, exclude=set())
                rows.append({"set": "known", "source": S, "battery_id": bid, "correct": r2["nearest"] == S, **r2})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "outputs" / "toolkit_phase3_trust_scores_novel_vs_known.csv", index=False)

    y = (df["set"] == "novel").astype(int).to_numpy()
    print(f"novel batteries: {int(y.sum())}, known held-out batteries: {int((1 - y).sum())}")
    aucs = {}
    for s in ["ratio_familiar", "ratio_somewhat", "d_nearest", "min_maha", "nll_min"]:
        aucs[s] = float(roc_auc_score(y, df[s].to_numpy()))
        print(f"  AUC {s:16s} = {aucs[s]:.3f}")
    best = max(aucs, key=aucs.get)
    fpr, tpr, thr = roc_curve(y, df[best].to_numpy())
    roc = pd.DataFrame({"threshold": thr, "tpr_novel_flagged": tpr, "fpr_known_false_alarm": fpr})
    roc.to_csv(ROOT / "outputs" / "toolkit_phase3_trust_roc.csv", index=False)
    print(f"\nbest score by AUC: {best} ({aucs[best]:.3f}). ROC excerpt:")
    for target in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        ok = roc[roc["tpr_novel_flagged"] >= target]
        r0 = ok.iloc[0]
        print(f"  TPR>={target:.2f}: threshold {r0['threshold']:.3f} -> novel flagged {r0['tpr_novel_flagged']:.1%}, known false-alarm {r0['fpr_known_false_alarm']:.1%}")
    ok = roc[roc["tpr_novel_flagged"] >= TARGET_TPR].iloc[0]
    t = float(ok["threshold"])
    flag = df[best] > t if False else df[best] >= t   # sklearn thresholds: flag when score >= threshold
    novel, known = df["set"] == "novel", df["set"] == "known"
    tpr_o, fpr_o = float(flag[novel].mean()), float(flag[known].mean())
    graded_only = fpr_o > MAX_ACCEPTABLE_FPR
    print(f"\nOPERATING POINT: flag if {best} >= {t:.3f}: novel-source batteries flagged {tpr_o:.1%}, "
          f"known-source held-out false-alarm {fpr_o:.1%} ({int(flag[known].sum())}/{int(known.sum())}); "
          f"{'GRADED MESSAGING ONLY (false alarms above the stated %.0f%% cap)' % (MAX_ACCEPTABLE_FPR*100) if graded_only else 'usable'}")
    per = df.assign(flag=flag).groupby(["set", "source"])["flag"].agg(["sum", "size", "mean"]).reset_index()
    print("\nPer source (novel: fraction flagged; known: false-alarm fraction):")
    print(per.pivot(index="source", columns="set", values="mean").round(3).to_string())
    per.to_csv(ROOT / "outputs" / "toolkit_phase3_trust_operating_point_per_source.csv", index=False)
    op = {"score": best, "threshold": t, "novel_flagged": tpr_o, "known_false_alarm": fpr_o, "auc": aucs[best],
          "all_aucs": aucs, "graded_only": bool(graded_only), "target_tpr": TARGET_TPR, "max_acceptable_fpr": MAX_ACCEPTABLE_FPR}
    (ROOT / "models" / "_trust_operating_point.json").write_text(json.dumps(op, indent=1))
    ep.write_meta(ROOT / "models" / "_trust_operating_point.json", "embeddings", ep.CAND)
    print("saved models/_trust_operating_point.json")


if __name__ == "__main__":
    main()
