"""
out_of_domain replacement test matrix (2026-09-30): for each of the 6 built-in datasets plus the HNEI upload,
drive the REAL app via AppTest and record: which battery/cycle the app shows by default, whether the
OUT-OF-DOMAIN banner is shown (= ctx["out_of_domain"]), what the Prediction tab's RUL metric displays, and
the stated reason. Run once before and once after the change; label via argv[1] ("old"/"new").
"""
import io
import json
import sys
import time
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def snapshot(at, label):
    banner = [e.value for e in list(at.error) + list(at.warning) if e.value and ("OUT-OF-DOMAIN" in e.value or "looks unfamiliar" in e.value)]
    neutral = [e.value for e in at.info if e.value and "No distribution shift detected" in e.value]
    green = [e.value for e in list(at.success) if e.value and ("trusted" in e.value.lower() or "in-domain" in e.value.lower())]
    unreliable = [e.value for e in at.error if e.value and "unreliable for this cycle" in e.value]
    rul = [m.value for m in at.metric if m.label == "Predicted RUL"]
    soh = [m.value for m in at.metric if m.label == "Predicted SOH"]
    hdr = [h.value for h in at.header][:1]
    reason = [c.value for c in at.caption if c.value and ("RUL is hidden" in c.value or "hidden rather than" in c.value)]
    return {"case": label, "header": hdr[0] if hdr else None, "exceptions": len(at.exception),
            "out_of_domain_banner": bool(banner), "neutral_no_shift_note": bool(neutral),
            "not_detected_line_shown": any("1 in 5 unfamiliar batteries are not detected" in n for n in neutral),
            "any_green_trusted_message": bool(green),
            "banner_reason": (banner[0].split("Reasons:")[1].split("\n")[0].strip()[:220] if banner and "Reasons:" in banner[0] else None),
            "banner_has_both_error_numbers": bool(banner and "Two error numbers" in banner[0]),
            "soh_metric": soh[0] if soh else None, "rul_metric": rul[0] if rul else None,
            "rul_hidden_reason": (reason[0][:260] if reason else None),
            "unreliable_banner": bool(unreliable)}


def main(tag):
    rows = []
    for ds in ["NASA", "MIT", "CALCE", "Oxford", "HUST", "XJTU"]:
        t0 = time.time()
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=400)
        at.query_params["dataset"] = ds
        at.run()
        r = snapshot(at, ds)
        r["secs"] = round(time.time() - t0, 1)
        rows.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
    t0 = time.time()
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=400)
    at.query_params["mode"] = "Upload your own cycle data"
    at.run()
    up = at.get("file_uploader")[0]
    up.set_value([("hnei_test.csv", (ROOT / "outputs" / "_phase0_hnei_upload_test.csv").read_bytes(), "text/csv")])
    at.run()
    r = snapshot(at, "HNEI upload")
    r["secs"] = round(time.time() - t0, 1)
    rows.append(r)
    print(json.dumps(r, ensure_ascii=False), flush=True)
    (ROOT / "outputs" / f"toolkit_ood_matrix_{tag}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "run")
