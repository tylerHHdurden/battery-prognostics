"""Visit a deployed app, report exception text (full traceback text as rendered) and whether the 'build probe' caption is present. Usage: python src/staging_probe_check.py <url> <screenshot name>"""
import sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright
URL, NAME = sys.argv[1].rstrip("/"), sys.argv[2]
OUT = Path(__file__).resolve().parent.parent / "outputs" / "live_smoke"
with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=True); pg = b.new_page(viewport={"width": 1300, "height": 1400})
    pg.goto(f"{URL}/?dataset=NASA", wait_until="domcontentloaded", timeout=180000)
    t0 = time.time(); state = "timeout"
    while time.time() - t0 < 420:
        btn = pg.get_by_role("button", name="Yes, get this app back up!")
        if btn.count(): btn.first.click(); print("woke app", flush=True); time.sleep(15)
        root = pg.frame_locator('iframe[title="streamlitApp"]') if pg.locator('iframe[title="streamlitApp"]').count() else pg
        try:
            if root.locator('[data-testid="stException"]').count(): state = "EXCEPTION"; break
            if root.get_by_role("tab", name="Prediction").count(): state = "TABS"; break
        except Exception: pass
        time.sleep(5)
    root = pg.frame_locator('iframe[title="streamlitApp"]') if pg.locator('iframe[title="streamlitApp"]').count() else pg
    body = root.locator("body").inner_text(timeout=30000)
    if state == "TABS":
        root.get_by_role("tab", name="Prediction").first.click(); time.sleep(4); body = root.locator("body").inner_text(timeout=30000)
    print("STATE:", state, "| probe caption present:", "build probe: probe-2026-10-01" in body)
    if state == "EXCEPTION": print(root.locator('[data-testid="stException"]').first.inner_text()[:1500])
    pg.screenshot(path=str(OUT / f"{NAME}.png")); b.close()
