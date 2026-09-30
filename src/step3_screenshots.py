"""Screenshots of the new distribution-shift messages in the running app (Streamlit + Edge via Playwright).
Uploads the unflagged (HNEI) and the flagged (distorted) CSVs and saves outputs/step3_screenshot_{unflagged,flagged}.png."""
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
PORT = 8765
srv = subprocess.Popen([str(ROOT / ".venv" / "Scripts" / "python.exe"), "-m", "streamlit", "run", str(ROOT / "app.py"), "--server.port", str(PORT),
                        "--server.headless", "true", "--browser.gatherUsageStats", "false"], cwd=str(ROOT),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    time.sleep(12)
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=True)
        for label, csv in (("unflagged", "_phase0_hnei_upload_test.csv"), ("flagged", "_step3_distorted_upload_test.csv")):
            page = b.new_page(viewport={"width": 1400, "height": 1500})
            page.goto(f"http://localhost:{PORT}/?mode=Upload+your+own+cycle+data", wait_until="networkidle", timeout=120000)
            page.wait_for_selector("input[type=file]", state="attached", timeout=120000)
            page.set_input_files("input[type=file]", str(ROOT / "outputs" / csv))
            key = "No distribution shift detected" if label == "unflagged" else "looks unfamiliar"
            page.get_by_text(key).first.wait_for(timeout=300000)
            page.get_by_text(key).first.scroll_into_view_if_needed()
            time.sleep(2)
            page.screenshot(path=str(ROOT / "outputs" / f"step3_screenshot_{label}.png"), full_page=False)
            print("saved", label, flush=True)
            page.close()
        b.close()
finally:
    srv.terminate()
