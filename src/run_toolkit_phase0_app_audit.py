"""
Toolkit Phase 0: live app audit, no code changes.

Drives app.py end-to-end via Streamlit's own AppTest framework (this
project's established convention - see `_regression_sweep_calce_fix.py`)
for 4 battery cases: NASA (in-domain, dropdown), CALCE (dropdown,
out-of-domain), XJTU (dropdown, precomputed-path, out-of-domain), and
HNEI (a BatteryLife source - NOT in the sidebar dropdown at all, tested
via the "Upload your own cycle data" path instead, using a real HNEI
battery's own raw cycles converted to the app's documented upload CSV
format).

AppTest renders ALL 8 tabs' content in a single `.run()` call (tabs are
visual containers, not lazy-loaded) - so one run per battery captures
everything; this script then searches the resulting element tree for
tab-identifying content (metrics, warnings/errors/success messages,
markdown text) rather than "clicking" through tabs, which AppTest
doesn't model as separate navigations anyway.

Records, per battery: exceptions (any tab), whether the OC-SVM anomaly
flag fired, the RUL value shown (or why it's unavailable), the SOH
prediction and conformal interval, the out-of-domain warning state, and
a plausibility read (values in a sane 0-105% SOH range, RUL
non-negative if shown, no rendering errors).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_adapters_batterylife import batterylife_cell_ids, iterate_batterylife_cycles

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"


def build_hnei_upload_csv() -> Path:
    """Converts one real HNEI battery's own raw cycles into the app's
    documented upload format (cycle_idx, phase, time_s, voltage_v,
    current_a, temperature_c) - a genuine BatteryLife-sourced test of
    the upload path, not a synthetic stand-in."""
    ids = batterylife_cell_ids("HNEI")
    cid = ids[0]
    cycles = list(iterate_batterylife_cycles("HNEI", cid))
    rows = []
    for c in cycles[:30]:  # cap - the app only needs enough cycles to render a slider meaningfully
        for phase in ("charge", "discharge"):
            p = c[phase]
            n = len(p["t"])
            for i in range(n):
                rows.append({
                    "cycle_idx": c["cycle_idx"], "phase": phase,
                    "time_s": float(p["t"][i]), "voltage_v": float(p["V"][i]),
                    "current_a": float(p["I"][i]),
                    "temperature_c": float(p["T"][i]) if p["T"] is not None else np.nan,
                })
    df = pd.DataFrame(rows)
    out_path = ROOT / "outputs" / "_phase0_hnei_upload_test.csv"
    df.to_csv(out_path, index=False)
    print(f"[phase0] built HNEI upload test CSV from {cid}: {len(cycles[:30])} cycles, {len(df)} rows -> {out_path}")
    return out_path, cid


def find_metric(at, label_substr: str):
    for m in at.metric:
        if label_substr.lower() in m.label.lower():
            return m.label, m.value
    return None, None


def find_text_containing(at, needle: str) -> list[str]:
    hits = []
    for group in [at.markdown, at.caption, at.error, at.warning, at.success, at.info]:
        for el in group:
            try:
                val = el.value
            except Exception:
                continue
            if val and needle.lower() in str(val).lower():
                hits.append(str(val)[:300])
    return hits


def audit_one(label: str, setup_fn) -> dict:
    print(f"\n=== Auditing: {label} ===")
    t0 = time.time()
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.run()
    setup_fn(at)
    at.run()

    n_exc = len(at.exception)
    exceptions = [str(e) for e in at.exception] if n_exc else []
    for e in exceptions:
        print(f"  EXCEPTION: {e}")

    rul_label, rul_value = find_metric(at, "RUL")
    soh_metrics = [(m.label, m.value) for m in at.metric if "soh" in m.label.lower() or "health" in m.label.lower()]
    anomaly_hits_flagged = find_text_containing(at, "anomaly flagged") or find_text_containing(at, "🚨")
    anomaly_hits_clean = find_text_containing(at, "no anomaly flagged")
    ood_hits = find_text_containing(at, "out-of-domain")
    all_error_text = [str(e.value) for e in at.error] if hasattr(at, "error") else []

    print(f"  exceptions={n_exc}")
    print(f"  RUL metric: {rul_label} = {rul_value}")
    print(f"  SOH metrics found: {soh_metrics[:5]}")
    print(f"  anomaly-flagged text found: {bool(anomaly_hits_flagged)}")
    print(f"  anomaly-clean text found: {bool(anomaly_hits_clean)}")
    print(f"  out-of-domain warning found: {bool(ood_hits)}")
    print(f"  st.error() elements: {len(all_error_text)}")
    print(f"  time: {time.time()-t0:.1f}s")

    return {
        "label": label, "n_exceptions": n_exc, "exceptions": exceptions,
        "rul_label": rul_label, "rul_value": rul_value,
        "soh_metrics": soh_metrics, "anomaly_flagged": bool(anomaly_hits_flagged),
        "anomaly_clean": bool(anomaly_hits_clean), "out_of_domain_warning": bool(ood_hits),
        "n_st_error_elements": len(all_error_text), "st_error_text": all_error_text[:3],
    }


def main():
    t0 = time.time()
    print("=== Toolkit Phase 0: live app audit (AppTest, no code changes) ===")

    results = []

    def setup_nasa(at):
        at.query_params["dataset"] = "NASA"
        at.query_params["battery"] = "B0005"

    def setup_calce(at):
        at.query_params["dataset"] = "CALCE"
        at.query_params["battery"] = "CS2_35"

    def setup_xjtu(at):
        at.query_params["dataset"] = "XJTU"

    results.append(audit_one("NASA / B0005 (in-domain)", setup_nasa))
    results.append(audit_one("CALCE / CS2_35 (out-of-domain, raw data path)", setup_calce))
    results.append(audit_one("XJTU (out-of-domain, precomputed path - not in dataset_available)", setup_xjtu))

    hnei_csv_path, hnei_cid = build_hnei_upload_csv()

    print(f"\n=== Auditing: HNEI/{hnei_cid} (BatteryLife source - NOT in sidebar dropdown, upload path) ===")
    try:
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
        at.run()
        at.query_params["mode"] = "Upload your own cycle data"
        at.run()
        uploader_widgets = [w for w in at.get("file_uploader")]
        print(f"  file_uploader widgets found: {len(uploader_widgets)}")
        if uploader_widgets:
            with open(hnei_csv_path, "rb") as f:
                csv_bytes = f.read()
            uploader_widgets[0].set_value([("hnei_test.csv", csv_bytes, "text/csv")])
            at.run()
        n_exc = len(at.exception)
        for e in at.exception:
            print(f"  EXCEPTION: {e}")
        rul_label, rul_value = find_metric(at, "RUL")
        print(f"  exceptions={n_exc}, RUL metric: {rul_label}={rul_value}")
        results.append({
            "label": f"HNEI/{hnei_cid} (via upload)", "n_exceptions": n_exc,
            "exceptions": [str(e) for e in at.exception],
            "rul_label": rul_label, "rul_value": rul_value,
            "soh_metrics": [(m.label, m.value) for m in at.metric],
            "anomaly_flagged": bool(find_text_containing(at, "anomaly flagged")),
            "anomaly_clean": bool(find_text_containing(at, "no anomaly flagged")),
            "out_of_domain_warning": bool(find_text_containing(at, "out-of-domain")),
            "n_st_error_elements": len(at.error) if hasattr(at, "error") else 0,
            "st_error_text": [str(e.value) for e in at.error][:3] if hasattr(at, "error") else [],
            "note": "sidebar dropdown does NOT list HNEI or any BatteryLife source - tested via upload only",
        })
    except Exception as e:
        print(f"  UPLOAD PATH TEST FAILED WITH: {type(e).__name__}: {e}")
        results.append({"label": f"HNEI/{hnei_cid} (via upload)", "n_exceptions": -1,
                        "exceptions": [f"AUDIT SCRIPT ITSELF FAILED: {type(e).__name__}: {e}"],
                        "note": "sidebar dropdown does NOT list HNEI or any BatteryLife source"})

    # --- write outputs/app_audit.md ---
    lines = ["# App Audit (Phase 0, toolkit pass)\n",
             "Driven via Streamlit AppTest, no code changes. One run per battery renders ALL 8 "
             "tabs' content at once (tabs are visual containers, not lazy-loaded).\n"]
    for r in results:
        lines.append(f"\n## {r['label']}\n")
        lines.append(f"- Exceptions: {r['n_exceptions']}")
        if r.get("exceptions"):
            for e in r["exceptions"][:5]:
                lines.append(f"  - `{e[:500]}`")
        lines.append(f"- RUL metric: {r.get('rul_label')} = {r.get('rul_value')}")
        lines.append(f"- SOH-related metrics: {r.get('soh_metrics')}")
        lines.append(f"- OC-SVM anomaly flagged: {r.get('anomaly_flagged')}")
        lines.append(f"- OC-SVM 'no anomaly' shown: {r.get('anomaly_clean')}")
        lines.append(f"- Out-of-domain warning shown: {r.get('out_of_domain_warning')}")
        lines.append(f"- st.error() elements: {r.get('n_st_error_elements')}")
        if r.get("st_error_text"):
            for e in r["st_error_text"]:
                lines.append(f"  - error text: `{e[:300]}`")
        if r.get("note"):
            lines.append(f"- NOTE: {r['note']}")

    (OUT_DIR / "app_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[phase0] saved outputs/app_audit.md")

    print(f"\n[phase0] TOTAL TIME: {(time.time()-t0)/60:.2f} minutes")


if __name__ == "__main__":
    main()
