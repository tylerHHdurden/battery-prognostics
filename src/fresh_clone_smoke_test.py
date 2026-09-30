"""Smoke test meant to run inside a FRESH CLONE with a clean virtualenv built from requirements-lock.txt (no data/raw, nothing outside the repo).
1. load_resources() (all model + sidecar + profile + CSV assertions run), 2. one real Prediction per dataset (NASA, MIT, CALCE, Oxford, HUST, XJTU)
through the production precomputed path, 3. the full app (AppTest) for each dataset plus one BatteryLife (HNEI) upload, 4. a passport build + JSON/PDF export.
Fails loudly on any exception; prints a JSON summary; asserts every resolved path is inside the repo."""
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import live_inference as li  # noqa: E402
from battery_passport import build_passport, passport_json, passport_pdf  # noqa: E402

assert str(li.ROOT.resolve()) == str(ROOT.resolve()), (li.ROOT, ROOT)
summary = {"repo_root": str(ROOT), "raw_data_present": (ROOT / "data" / "raw").exists()}
res = li.load_resources()
summary["load_resources"] = "ok"
rows = []
for ds in ["NASA", "MIT", "CALCE", "Oxford", "HUST", "XJTU"]:
    avail = li.available_precomputed_cycles(ds)
    bid = sorted(avail)[0]
    cyc = avail[bid][len(avail[bid]) // 2]
    ctx = li.predict_and_explain_precomputed(ds, bid, int(cyc), res)
    p = build_passport(ctx, ds, bid, len(avail[bid]))
    assert passport_json(p) and len(passport_pdf(p)) > 10000
    rows.append({"dataset": ds, "battery": bid, "cycle": int(cyc), "soh_pred": ctx["soh_pred"], "true_soh": ctx.get("true_soh"),
                 "candidate_used": li._use_candidate(ds), "out_of_domain": ctx["out_of_domain"], "passport_ok": True})
    print(rows[-1], flush=True)
summary["precomputed_predictions"] = rows

from streamlit.testing.v1 import AppTest  # noqa: E402

app = []
for ds in ["NASA", "MIT", "CALCE", "Oxford", "HUST", "XJTU"]:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
    at.query_params["dataset"] = ds
    at.run()
    soh = [m.value for m in at.metric if m.label == "Predicted SOH"]
    app.append({"dataset": ds, "exceptions": len(at.exception), "predicted_soh": soh[:1]})
    print(app[-1], flush=True)
    assert len(at.exception) == 0 and soh, ds
csv = ROOT / "outputs" / "_step3_real_hnei_upload.csv"
at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
at.query_params["mode"] = "Upload your own cycle data"
at.run()
at.get("file_uploader")[0].set_value([(csv.name, csv.read_bytes(), "text/csv")])
at.run()
soh = [m.value for m in at.metric if m.label == "Predicted SOH"]
app.append({"dataset": "BatteryLife upload (HNEI)", "exceptions": len(at.exception), "predicted_soh": soh[:1],
            "banner": [e.value[:60] for e in list(at.warning) + list(at.info) if e.value and ("unfamiliar" in e.value or "No distribution shift" in e.value)][:1]})
print(app[-1], flush=True)
assert len(at.exception) == 0 and soh
summary["app_runs"] = app
summary["result"] = "ALL PASSED"
print(json.dumps(summary, indent=1, ensure_ascii=False))
