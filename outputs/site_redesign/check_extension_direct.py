import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import live_inference as li
from battery_passport import build_passport, passport_json
res = li.load_resources()
out = []
for s in li.EXTENSION_SOURCES:
    cm = li.available_precomputed_cycles(s)
    nb = len(cm); miss_trust = miss_lookup = 0; variants = set()
    for b, cyc in cm.items():
        if li.lookup_builtin_trust(s, b) is None: miss_trust += 1
        if (s, b, cyc[-1]) not in res["candidate_fusion_lookup"]: miss_lookup += 1
    b0 = sorted(cm)[0]; c = cm[b0][-1]
    ctx = li.predict_and_explain_precomputed(s, b0, c, res)
    p = build_passport(ctx, s, b0, len(cm[b0]))
    passport_json(p)
    # flagged share and variants over all batteries (last cycle)
    flagged = 0; rul_any = 0; errs = []
    for b, cyc in cm.items():
        x = li.predict_and_explain_precomputed(s, b, cyc[-1], res)
        variants.add(x.get("model_variant")); flagged += bool(x["out_of_domain"])
        rul_any += (not x["rul_hidden"]) and x["rul_pred"] is not None
        if x["true_soh"] is not None: errs.append(abs(x["soh_pred"] - x["true_soh"]))
    row = dict(source=s, n_batteries=nb, missing_trust=miss_trust, missing_cand_lookup=miss_lookup, variants=sorted(variants),
               first_battery=b0, cycle=c, soh=ctx["soh_pred"], true=ctx["true_soh"], flagged_first=ctx["out_of_domain"],
               nearest=(ctx["trust"] or {}).get("nearest_source"), flagged_batteries=f"{flagged}/{nb}", rul_shown_any=rul_any,
               passport_model=p["model_provenance"]["model"], passport_rul_shown=p["expected_remaining_life"].get("shown", True),
               mae_last_cycle=round(sum(errs)/len(errs), 2))
    print(row, flush=True); out.append(row)
json.dump(out, open(ROOT / "outputs/site_redesign/_extension_direct.json", "w"), indent=1, default=str)
