"""Visual + functional check of the redesigned site at desktop and mobile widths. Usage: python src/site_visual_check.py <base_url> <out_dir>
For each viewport: built-in Oxford (every tab opened, exceptions counted), then a real upload (flagged distorted + unflagged HNEI): full-page screenshots of the top
(result card) and of each tab. Also checks: no green (stAlertContentSuccess) elements, sample-file download works, mobile has no horizontal page scroll."""
import json, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright
BASE, OUT = sys.argv[1].rstrip("/"), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
ROOT = Path(__file__).resolve().parent.parent
TABS = ["Showcase", "Prediction", "Explainability", "Health Report", "Streaming Digital Twin", "World Model", "Model Validation", "Full Results Archive"]
res = {"runs": []}


def settle(pg, secs=4):
    time.sleep(secs)
    pg.wait_for_function("() => !document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=180000) if False else None


with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=True)
    for vp_name, vp in (("desktop", {"width": 1400, "height": 1000}), ("mobile", {"width": 390, "height": 844})):
        for case, url, upload in (("oxford", "/?dataset=Oxford", None), ("upload_unflagged", "/?mode=Upload+your+own+cycle+data", "_phase0_hnei_upload_test.csv"),
                                  ("upload_flagged", "/?mode=Upload+your+own+cycle+data", "_step3_distorted_upload_test.csv")):
            row = {"viewport": vp_name, "case": case}
            try:
                pg = b.new_page(viewport=vp, accept_downloads=True)
                pg.goto(BASE + url, wait_until="domcontentloaded", timeout=180000)
                pg.get_by_role("tab", name="Showcase").first.wait_for(timeout=240000)
                if upload:
                    pg.locator("input[type=file]").first.set_input_files(str(ROOT / "outputs" / upload))
                    pg.get_by_text("Result").first.wait_for(timeout=600000)
                    pg.get_by_text("Predicted SOH").first.wait_for(timeout=600000)
                time.sleep(4)
                pg.screenshot(path=str(OUT / f"{vp_name}_{case}_top.png"), full_page=False)
                row["horizontal_scroll"] = pg.evaluate("() => document.documentElement.scrollWidth > document.documentElement.clientWidth + 2")
                exc = {}
                for t in TABS:
                    tab = pg.get_by_role("tab", name=t).first
                    tab.scroll_into_view_if_needed(); tab.click(); time.sleep(6)
                    exc[t] = pg.locator('[data-testid="stException"]').count()
                    if t in ("Prediction", "Full Results Archive"):
                        pg.screenshot(path=str(OUT / f"{vp_name}_{case}_{t.replace(' ', '_')}.png"), full_page=False)
                row["exceptions_by_tab"] = exc
                row["green_elements"] = pg.locator('[data-testid="stAlertContentSuccess"]').count()
                row["result_card_visible"] = pg.get_by_text("Predicted SOH").count() > 0
                if case == "upload_unflagged" and vp_name == "desktop":
                    with pg.expect_download(timeout=60000) as d:
                        pg.get_by_text("Download a sample file").first.click()
                    d.value.save_as(str(OUT / "downloaded_sample.csv")); row["sample_download_bytes"] = (OUT / "downloaded_sample.csv").stat().st_size
                row["pass"] = sum(exc.values()) == 0 and row["green_elements"] == 0 and row["result_card_visible"] and not (vp_name == "mobile" and row["horizontal_scroll"])
                pg.close()
            except Exception as e:
                row.update({"error": repr(e)[:300], "pass": False})
            res["runs"].append(row); print(row, flush=True)
    b.close()
res["all_passed"] = all(r.get("pass") for r in res["runs"])
(OUT / "site_visual_result.json").write_text(json.dumps(res, indent=1)); print("ALL PASSED" if res["all_passed"] else "SOME FAILED")
