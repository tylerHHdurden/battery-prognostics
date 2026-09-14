"""
End-to-end smoke test of the Stage-4-rewired live_inference.py - the
exact pipeline app.py actually calls (load_resources + predict_and_
explain), not a proxy for it. Run before declaring the deployed
pipeline verified working, and worth rerunning after any future change
to the deployed model/feature pipeline as a permanent regression test.

Caught two real bugs before they reached the live app when first run:
(1) the RUL joint-fusion model's own HI feature vector must NOT include
cycle_idx and MUST be z-scored with its own fit-split stats - a
mismatch crashed with a matrix-shape error rather than silently
producing wrong output; (2) confirmed the canonical _rel-ratio features
are highly sensitive to which baseline is used (test 2's SOH prediction
differs from test 1's by >15 points using only the current-cycle
fallback baseline instead of the real cycle-10 one) - validating why
baseline_his must be threaded through every app.py call site, not an
optional nicety.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
from data_adapters import iterate_nasa_cycles, iterate_mit_cycles, iterate_calce_cycles
from health_indicators import compute_health_indicators
from stage1_common import BASELINE_CYCLE
from live_inference import load_resources, predict_and_explain
import json

print("=== loading resources (as app.py's get_resources() would) ===")
res = load_resources()
print("loaded keys:", sorted(res.keys()))

print("\n=== test 1: known NASA battery, mid-life cycle, WITH baseline_his ===")
cycles = list(iterate_nasa_cycles("B0018"))
baseline_cycle = next((c for c in cycles if c["cycle_idx"] == BASELINE_CYCLE), cycles[0])
baseline_his = compute_health_indicators(baseline_cycle)
test_cycle = cycles[60]
ctx = predict_and_explain(test_cycle, res, baseline_his=baseline_his)
assert "error" not in ctx, ctx.get("error")
print(f"SOH pred={ctx['soh_pred']} interval=[{ctx['soh_conformal_lo']},{ctx['soh_conformal_hi']}]")
print(f"RUL pred={ctx['rul_pred']} interval=[{ctx['rul_conformal_lo']},{ctx['rul_conformal_hi']}]")
print(f"anomaly_flag={ctx['anomaly_flag']} out_of_domain={ctx['out_of_domain']}")
print(f"top_features={[f['feature'] for f in ctx['top_features']]}")
print(f"voltage_region={ctx['voltage_region']} (error={ctx['voltage_region_error']})")

print("\n=== test 2: same cycle, WITHOUT baseline_his (fallback path) ===")
ctx2 = predict_and_explain(test_cycle, res, baseline_his=None)
assert "error" not in ctx2, ctx2.get("error")
print(f"SOH pred={ctx2['soh_pred']} (fallback baseline - should differ somewhat from test 1)")

print("\n=== test 3: MIT battery (different chemistry-adjacent profile) ===")
mit_subset = json.load(open('data/processed/mit_subset.json'))
entry = next(e for e in mit_subset if e['global_id'] == 'b3c35')
mit_cycles = list(iterate_mit_cycles(entry['batch_file'], entry['cell_index']))
mit_baseline = next((c for c in mit_cycles if c["cycle_idx"] == BASELINE_CYCLE), mit_cycles[0])
mit_baseline_his = compute_health_indicators(mit_baseline)
ctx3 = predict_and_explain(mit_cycles[500], res, baseline_his=mit_baseline_his)
assert "error" not in ctx3, ctx3.get("error")
print(f"SOH pred={ctx3['soh_pred']} RUL pred={ctx3['rul_pred']} anomaly={ctx3['anomaly_flag']}")

print("\n=== test 4: CALCE cycle (expect out-of-domain / no temperature) ===")
calce_cycles = list(iterate_calce_cycles("CS2_35"))
calce_baseline = next((c for c in calce_cycles if c["cycle_idx"] == BASELINE_CYCLE), calce_cycles[0])
calce_baseline_his = compute_health_indicators(calce_baseline)
ctx4 = predict_and_explain(calce_cycles[300], res, baseline_his=calce_baseline_his)
assert "error" not in ctx4, ctx4.get("error")
print(f"SOH pred={ctx4['soh_pred']} out_of_domain={ctx4['out_of_domain']} reasons={ctx4['domain_reasons']}")
assert ctx4["out_of_domain"] is True, "CALCE should trigger out-of-domain (no temperature channel at minimum)"

print("\n=== ALL SMOKE TESTS PASSED ===")
