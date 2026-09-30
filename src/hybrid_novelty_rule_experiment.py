"""Hybrid novelty rule experiment (leave-source-out). Reads the trust scores CSV, writes only the new hybrid CSV + summary txt.
Threshold(s) always chosen without the held-out source: lowest known false alarm subject to >=80% novel flagged (ties -> higher detection)."""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "outputs" / "toolkit_phase3_trust_scores_novel_vs_known.csv")
df["novel"] = (df["set"] == "novel").astype(int)
TARGET = 0.80
FEATS = ["nll_min", "min_maha", "ratio_familiar", "d_nearest"]

def grid(x, n=100000):  # n large = every unique score is a candidate threshold (exact, like sklearn roc_curve); the first version used 120 quantiles, which made the ROC baseline 389/476 instead of 390/476
    u = np.unique(x)
    if len(u) > n: u = np.unique(np.quantile(x, np.linspace(0, 1, n)))
    return np.append(u, np.inf)

def best(cands):  # cands: list of (params, tpr, fpr)
    ok = [c for c in cands if c[1] >= TARGET]
    if not ok: return max(cands, key=lambda c: c[1])[0]
    return min(ok, key=lambda c: (c[2], -c[1]))[0]

def eval_flags(flag, y): return flag[y == 1].mean(), flag[y == 0].mean()

def fit_roc(tr):
    y = tr.novel.values; c = [(t, *eval_flags(tr.nll_min.values >= t, y)) for t in grid(tr.nll_min.values)]
    return best(c)
def app_roc(p, te): return (te.nll_min >= p).values

def fit_a(tr):
    y = tr.novel.values; nf = (tr.level != "familiar").values
    c = [(t, *eval_flags((tr.nll_min.values >= t) | nf, y)) for t in grid(tr.nll_min.values)]
    return best(c)
def app_a(p, te): return ((te.nll_min >= p) | (te.level != "familiar")).values

def fit_b(tr):
    y = tr.novel.values; g1, g2 = grid(tr.nll_min.values, 80), grid(tr.ratio_familiar.values, 80)
    a1, a2 = tr.nll_min.values, tr.ratio_familiar.values
    c = [((t1, t2), *eval_flags((a1 >= t1) | (a2 >= t2), y)) for t1 in g1 for t2 in g2]
    return best(c)
def app_b(p, te): return ((te.nll_min >= p[0]) | (te.ratio_familiar >= p[1])).values

def mk(): return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000))
def fit_c(tr):
    # inner leave-source-out out-of-fold scores -> threshold; final model on all train
    oof = np.zeros(len(tr)); srcs = tr.source.values
    for s in np.unique(srcs):
        m = srcs == s; oof[m] = mk().fit(tr.loc[~m, FEATS], tr.novel[~m]).decision_function(tr.loc[m, FEATS])
    y = tr.novel.values; t = best([(t, *eval_flags(oof >= t, y)) for t in grid(oof)])
    return (mk().fit(tr[FEATS], tr.novel), t)
def app_c(p, te): return p[0].decision_function(te[FEATS]) >= p[1]

METHODS = {"roc": (fit_roc, app_roc), "a_or_level": (fit_a, app_a), "b_or_ratio": (fit_b, app_b), "c_logreg": (fit_c, app_c)}
rows = []
for S in sorted(df.source.unique()):
    tr, te = df[df.source != S], df[df.source == S]
    r = {"held_out_source": S, "n_novel": int((te.novel == 1).sum()), "n_known": int((te.novel == 0).sum())}
    for k, (f, a) in METHODS.items():
        fl = a(f(tr), te)
        r[f"{k}_novel_flagged"] = int(fl[te.novel.values == 1].sum()); r[f"{k}_known_fa"] = int(fl[te.novel.values == 0].sum())
    rows.append(r)
out = pd.DataFrame(rows); out.to_csv(ROOT / "outputs" / "toolkit_phase3_hybrid_rule_leave_source_out.csv", index=False)

N, K = out.n_novel.sum(), out.n_known.sum()
lines = ["Hybrid novelty rule vs ROC rule, leave-source-out (thresholds chosen without held-out source; >=80% novel target)", "",
         f"{'rule':<12}{'novel flagged':>18}{'known false alarm':>22}"]
for k in METHODS:
    nf, fa = out[f"{k}_novel_flagged"].sum(), out[f"{k}_known_fa"].sum()
    lines.append(f"{k:<12}{f'{nf}/{N} = {nf/N:.1%}':>18}{f'{fa}/{K} = {fa/K:.1%}':>22}")
lines += ["", "Per-source (novel flagged/n_novel ; known FA/n_known) for all sources:"]
for _, r in out.iterrows():
    lines.append(f"{r.held_out_source:<10} nov {r.n_novel:>3} kn {r.n_known:>3} | " + " | ".join(f"{k}: {r[f'{k}_novel_flagged']}/{r.n_novel} FA {r[f'{k}_known_fa']}" for k in METHODS))
roc_nf, roc_fa = out.roc_novel_flagged.sum(), out.roc_known_fa.sum()
lines.append("")
for k in list(METHODS)[1:]:
    nf, fa = out[f"{k}_novel_flagged"].sum(), out[f"{k}_known_fa"].sum()
    clear = nf > roc_nf and fa <= roc_fa
    lines.append(f"{k}: {'BEATS' if clear else 'does NOT beat'} ROC (pooled dNovel={nf-roc_nf:+d}, dFA={fa-roc_fa:+d})")
txt = "\n".join(lines); (ROOT / "outputs" / "toolkit_phase3_hybrid_rule_summary.txt").write_text(txt); print(txt)
