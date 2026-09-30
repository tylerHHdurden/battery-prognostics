"""Step 3 validation: is the ROC operating point (nll_min >= t, >= 80% novel flagged) honest on HELD-OUT sources?
For each source S the threshold t is chosen WITHOUT S (both its novel-source rows and its known held-out rows removed),
using the same rule as src/trust_operating_point.py (lowest false-alarm rate among thresholds flagging >= 80% of novel batteries),
then applied to S. Also evaluates the CURRENT deployed-style rule (flag only when trust level == 'unfamiliar') on the same rows.
Reads outputs/toolkit_phase3_trust_scores_novel_vs_known.csv (written by trust_operating_point.py). Writes new CSVs only."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "outputs" / "toolkit_phase3_trust_scores_novel_vs_known.csv")
df["novel"] = (df["set"] == "novel").astype(int)
SCORE, TARGET = "nll_min", 0.80


def pick_threshold(train):
    fpr, tpr, thr = roc_curve(train["novel"], train[SCORE])
    ok = np.where(tpr >= TARGET)[0][0]
    return float(thr[ok])


rows = []
for S in sorted(df["source"].unique()):
    train, test = df[df["source"] != S], df[df["source"] == S]
    t = pick_threshold(train)
    flag = test[SCORE] >= t
    nov, kn = test["novel"] == 1, test["novel"] == 0
    cur = test["level"] == "unfamiliar"
    rows.append({"held_out_source": S, "threshold_chosen_without_source": round(t, 3),
                 "n_novel": int(nov.sum()), "novel_flagged_roc": int(flag[nov].sum()), "n_known": int(kn.sum()), "known_false_alarm_roc": int(flag[kn].sum()),
                 "novel_flagged_current_rule": int(cur[nov].sum()), "known_false_alarm_current_rule": int(cur[kn].sum())})
out = pd.DataFrame(rows)
out.to_csv(ROOT / "outputs" / "toolkit_phase3_trust_threshold_leave_source_out.csv", index=False)
tot = out[["n_novel", "novel_flagged_roc", "n_known", "known_false_alarm_roc", "novel_flagged_current_rule", "known_false_alarm_current_rule"]].sum()
pd.set_option("display.width", 250)
print(out.to_string(index=False))
print("\nPOOLED (thresholds chosen leave-source-out):")
print(f"  ROC rule    : novel flagged {tot.novel_flagged_roc}/{tot.n_novel} = {tot.novel_flagged_roc/tot.n_novel:.1%}; known false alarm {tot.known_false_alarm_roc}/{tot.n_known} = {tot.known_false_alarm_roc/tot.n_known:.1%}")
print(f"  current rule: novel flagged {tot.novel_flagged_current_rule}/{tot.n_novel} = {tot.novel_flagged_current_rule/tot.n_novel:.1%}; known false alarm {tot.known_false_alarm_current_rule}/{tot.n_known} = {tot.known_false_alarm_current_rule/tot.n_known:.1%}")
in_sample = out.assign(x=1)
# sources where the leave-source-out ROC rule flags fewer than half of their novel batteries
weak = out[out["novel_flagged_roc"] / out["n_novel"] < 0.5]["held_out_source"].tolist()
print("  sources where < 50% of their novel batteries are flagged by the ROC rule:", weak)
