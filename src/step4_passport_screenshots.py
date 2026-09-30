"""Phase 3B screenshots + real download test: drives the running app with Playwright (Edge), opens the Prediction tab, screenshots the passport section
and clicks both download buttons (saving the JSON and PDF the app actually serves).
Usage: python src/step4_passport_screenshots.py <nasa|mich|hnei>   (mich/hnei: real upload with the source's own profile removed via TRUST_EXCLUDE_SOURCES)."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
case = sys.argv[1]
CFG = {"nasa": {"upload": None, "exclude": ""},
       "mich": {"upload": "_step3_real_mich_upload.csv", "exclude": "mich,mich_exp"},
       "hnei": {"upload": "_step3_real_hnei_upload.csv", "exclude": "hnei"}}[case]
env = os.environ.copy()
env["TRUST_EXCLUDE_SOURCES"] = CFG["exclude"]
port = {"nasa": 8780, "mich": 8781, "hnei": 8782}[case]
out = ROOT / "outputs" / "passport_samples"
out.mkdir(exist_ok=True)
srv = subprocess.Popen([str(ROOT / ".venv" / "Scripts" / "python.exe"), "-m", "streamlit", "run", str(ROOT / "app.py"), "--server.port", str(port),
                        "--server.headless", "true", "--browser.gatherUsageStats", "false"], cwd=str(ROOT), env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
from playwright.sync_api import sync_playwright  # noqa: E402

try:
    time.sleep(12)
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=True)
        page = b.new_page(viewport={"width": 1400, "height": 1500}, accept_downloads=True)
        if CFG["upload"]:
            page.goto(f"http://localhost:{port}/?mode=Upload+your+own+cycle+data", wait_until="networkidle", timeout=120000)
            page.wait_for_selector("input[type=file]", state="attached", timeout=120000)
            page.set_input_files("input[type=file]", str(ROOT / "outputs" / CFG["upload"]))
            key = "looks unfamiliar" if case == "hnei" else "No distribution shift detected"
            page.get_by_text(key).first.wait_for(timeout=400000)  # the upload run must finish before the tab is used
            time.sleep(5)
        else:
            page.goto(f"http://localhost:{port}/?dataset=NASA", wait_until="networkidle", timeout=120000)
        page.get_by_role("tab", name="Prediction").first.wait_for(timeout=300000)
        page.get_by_role("tab", name="Prediction").first.click()
        page.get_by_text("Battery passport (research prototype)").first.wait_for(timeout=300000)
        page.get_by_text("Battery passport (research prototype)").first.scroll_into_view_if_needed()
        time.sleep(2)
        page.screenshot(path=str(out / f"passport_app_{case}.png"))
        with page.expect_download(timeout=120000) as dj:
            page.get_by_text("Download passport (JSON)").first.click()
        dj.value.save_as(str(out / f"passport_app_{case}_downloaded.json"))
        with page.expect_download(timeout=120000) as dp:
            page.get_by_text("Download passport (printable PDF)").first.click()
        dp.value.save_as(str(out / f"passport_app_{case}_downloaded.pdf"))
        b.close()
    js = json.loads((out / f"passport_app_{case}_downloaded.json").read_text(encoding="utf-8"))
    print(case, "SOH", js["state_of_health"]["value_percent"], "| RUL", js["expected_remaining_life"].get("cycles"), "| flagged", js["trust_status"]["flagged_as_unfamiliar"],
          "| nearest", js["trust_status"]["nearest_known_source"], "| pdf bytes", (out / f"passport_app_{case}_downloaded.pdf").stat().st_size, flush=True)
finally:
    srv.terminate()
