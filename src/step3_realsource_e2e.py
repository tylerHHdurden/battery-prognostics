"""Step 3 end-to-end test with REAL held-out sources (not a distorted upload).
For a chosen real battery we REMOVE its own source's profile (and its sibling family) from the trust-profile set via the
TRUST_EXCLUDE_SOURCES test hook in trust_report.py, upload the battery's first 30 cycles through the real app upload path
(load_resources -> predict_and_explain -> trust report -> domain_verdict -> render_domain_banner), and record nll_min, flagged,
nearest source and RUL visibility, then screenshot the app.
Usage: python src/step3_realsource_e2e.py <mich|hnei>   (run once per case; the env hook must be set before app start)."""
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
CASES = {
    "mich": {"raw": "MICH", "cid": "MICH_BLForm4_pouch_NMC_45C_0-100_1-1C_d", "exclude": "mich,mich_exp",
             "expect": "NOT flagged (leave-source-out table: mich 1/40 flagged; this battery's whole-history nll_min = -52.8)"},
    "hnei": {"raw": "HNEI", "cid": "HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_n", "exclude": "hnei",
             "expect": "flagged (leave-source-out table: hnei 14/14 flagged; this battery's whole-history nll_min = -4.67)"},
}
case = sys.argv[1]
cfg = CASES[case]
os.environ["TRUST_EXCLUDE_SOURCES"] = cfg["exclude"]
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from data_adapters_batterylife import iterate_batterylife_cycles  # noqa: E402

cycles = list(iterate_batterylife_cycles(cfg["raw"], cfg["cid"]))[:30]
rows = []
for c in cycles:
    for phase in ("charge", "discharge"):
        p = c[phase]
        for i in range(len(p["t"])):
            rows.append({"cycle_idx": c["cycle_idx"], "phase": phase, "time_s": float(p["t"][i]), "voltage_v": float(p["V"][i]),
                         "current_a": float(p["I"][i]), "temperature_c": float(p["T"][i]) if p["T"] is not None else np.nan})
csv = ROOT / "outputs" / f"_step3_real_{case}_upload.csv"
pd.DataFrame(rows).to_csv(csv, index=False)

from streamlit.testing.v1 import AppTest  # noqa: E402

at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
at.query_params["mode"] = "Upload your own cycle data"
at.run()
at.get("file_uploader")[0].set_value([(csv.name, csv.read_bytes(), "text/csv")])
at.run()
trust = None
try:
    trust = at.session_state[f"upload_trust::{csv.name}::{len(cycles)}"][1]
except Exception as e:  # key format guess failed: fall back to parsing the message
    print("session_state key lookup failed:", repr(e)[:120])
warn = [e.value for e in at.warning if e.value and "looks unfamiliar" in e.value]
info = [e.value for e in at.info if e.value and "No distribution shift detected" in e.value]
rul = [m.value for m in at.metric if m.label == "Predicted RUL"]
soh = [m.value for m in at.metric if m.label == "Predicted SOH"]
rec = [e.value for e in list(at.info) + list(at.warning) + list(at.error) + list(at.success) if e.value and "Continue normal use" in e.value or (e.value and "Monitor closely" in e.value)]
result = {"case": case, "battery": cfg["cid"], "profiles_removed": cfg["exclude"], "expectation": cfg["expect"], "cycles_uploaded": len(cycles),
          "exceptions": len(at.exception),
          "nll_min_30_cycle_median": None if trust is None else round(float(trust["nll_min"]), 2),
          "threshold": -5.521444398006403,
          "flagged": bool(warn), "neutral_message_shown": bool(info),
          "nearest_source": None if trust is None else trust["nearest_source"],
          "predicted_soh": soh[:1], "predicted_rul": rul[:1], "rul_visible": bool(rul and rul[0] != "not available"),
          "recommendation_box": [r[:200] for r in rec][:2]}
print(json.dumps(result, ensure_ascii=False, indent=1), flush=True)
(ROOT / "outputs" / f"step3_realsource_e2e_{case}.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")

# screenshot through the real server with the same env hook
from playwright.sync_api import sync_playwright  # noqa: E402

port = 8770 + (1 if case == "hnei" else 0)
srv = subprocess.Popen([str(ROOT / ".venv" / "Scripts" / "python.exe"), "-m", "streamlit", "run", str(ROOT / "app.py"), "--server.port", str(port),
                        "--server.headless", "true", "--browser.gatherUsageStats", "false"], cwd=str(ROOT), env=os.environ.copy(),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    time.sleep(12)
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=True)
        page = b.new_page(viewport={"width": 1400, "height": 1700})
        page.goto(f"http://localhost:{port}/?mode=Upload+your+own+cycle+data", wait_until="networkidle", timeout=120000)
        page.wait_for_selector("input[type=file]", state="attached", timeout=120000)
        page.set_input_files("input[type=file]", str(csv))
        key = "looks unfamiliar" if result["flagged"] else "No distribution shift detected"
        page.get_by_text(key).first.wait_for(timeout=300000)
        page.get_by_text(key).first.scroll_into_view_if_needed()
        time.sleep(3)
        page.screenshot(path=str(ROOT / "outputs" / f"step3_real_{case}_screenshot.png"))
        b.close()
finally:
    srv.terminate()
print("screenshot saved")
