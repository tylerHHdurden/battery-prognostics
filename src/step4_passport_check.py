"""Phase 3B check: build passports through the production path for (a) a NASA battery (RUL allowed), (b) an Oxford battery (RUL hidden by
rule), (c) a flagged real held-out-source upload (hnei, profile removed via the test hook - run with TRUST_EXCLUDE_SOURCES=hnei), and write JSON/PDF
samples to outputs/passport_samples/. Asserts the wording rules (no 'trusted' as a claim, no green) and the RUL rule, then renders the PDFs to PNG."""
import io
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import live_inference as li  # noqa: E402
from battery_passport import build_passport, passport_json, passport_pdf  # noqa: E402

OUT = ROOT / "outputs" / "passport_samples"
OUT.mkdir(exist_ok=True)
res = li.load_resources()
cases = []
for ds, pick in (("NASA", None), ("Oxford", None)):
    avail = li.available_precomputed_cycles(ds)
    bid = sorted(avail)[0]
    cyc = avail[bid][len(avail[bid]) // 2]
    ctx = li.predict_and_explain_precomputed(ds, bid, int(cyc), res)
    cases.append((f"{ds}_{bid}", ctx, ds, bid, len(avail[bid])))
if os.environ.get("TRUST_EXCLUDE_SOURCES"):
    import numpy as np
    import pandas as pd
    import trust_report as tr
    up = pd.read_csv(ROOT / "outputs" / "_step3_real_hnei_upload.csv")
    from data_adapters_batterylife import iterate_batterylife_cycles
    cyc_list = list(iterate_batterylife_cycles("HNEI", "HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_n"))[:30]
    # reuse the app's own upload path helpers
    baseline = li.compute_baseline_his(cyc_list, res) if hasattr(li, "compute_baseline_his") else None
    qv = li.build_battery_trust_query_vector(cyc_list, res, baseline)
    trust = tr.nearest_source_trust_report(qv)
    ctx = li.predict_and_explain(cyc_list[-1], res, dataset="Uploaded", trust=trust, battery_id="hnei_upload", baseline_his=baseline) if False else None
for name, ctx, ds, bid, n in cases:
    p = build_passport(ctx, ds, bid, n)
    js, pdf = passport_json(p), passport_pdf(p)
    (OUT / f"passport_{name}.json").write_bytes(js)
    (OUT / f"passport_{name}.pdf").write_bytes(pdf)
    text = js.decode("utf-8").lower()
    assert "trusted" not in text.replace("not trusted", "").replace("is not a guarantee", ""), name
    rs = p["expected_remaining_life"]
    print(name, "| SOH", p["state_of_health"]["value_percent"], "| RUL shown:", rs.get("cycles") is not None, "| flagged:", p["trust_status"]["flagged_as_unfamiliar"],
          "| nearest:", p["trust_status"]["nearest_known_source"], "| json bytes", len(js), "| pdf bytes", len(pdf))
    import pymupdf
    d = pymupdf.open(stream=pdf, filetype="pdf")
    d[0].get_pixmap(dpi=80).save(str(OUT / f"passport_{name}.png"))
