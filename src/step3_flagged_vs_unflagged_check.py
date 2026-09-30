"""Step 3 check: exercise BOTH branches of the new distribution-shift message in the app through AppTest.
UNFLAGGED = the HNEI test upload (its own source is among the 16 profiles). FLAGGED = the same file with distorted physical
channels (voltage x1.4, current x3, temperature 65 C), which lands far from every source profile. Prints the messages shown
and whether any green/'trusted' element appears. Writes outputs/step3_flagged_vs_unflagged_check.json."""
import io
import json
import sys
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
src = pd.read_csv(ROOT / "outputs" / "_phase0_hnei_upload_test.csv")
bad = src.copy()
bad["voltage_v"] = bad["voltage_v"] * 1.4
bad["current_a"] = bad["current_a"] * 3.0
bad["temperature_c"] = 65.0
bad_path = ROOT / "outputs" / "_step3_distorted_upload_test.csv"
bad.to_csv(bad_path, index=False)


def run(label, path):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
    at.query_params["mode"] = "Upload your own cycle data"
    at.run()
    at.get("file_uploader")[0].set_value([(path.name, path.read_bytes(), "text/csv")])
    at.run()
    warn = [e.value for e in at.warning if e.value and "looks unfamiliar" in e.value]
    info = [e.value for e in at.info if e.value and "No distribution shift detected" in e.value]
    errors = [e.value for e in at.error]
    green = [e.value for e in at.success]
    rul = [m.value for m in at.metric if m.label == "Predicted RUL"]
    soh = [m.value for m in at.metric if m.label == "Predicted SOH"]
    r = {"case": label, "exceptions": len(at.exception), "amber_warning": warn[:1], "neutral_info": info[:1],
         "st_success_elements": green, "st_error_elements": [e[:160] for e in errors],
         "predicted_soh": soh[:1], "predicted_rul": rul[:1]}
    print(json.dumps(r, ensure_ascii=False, indent=1), flush=True)
    return r


rows = [run("unflagged (HNEI upload)", ROOT / "outputs" / "_phase0_hnei_upload_test.csv"), run("flagged (distorted upload)", bad_path)]
(ROOT / "outputs" / "step3_flagged_vs_unflagged_check.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
