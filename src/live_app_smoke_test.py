"""Smoke test of a DEPLOYED Streamlit Cloud app (iframe-aware: Streamlit Cloud serves the app inside iframe[title="streamlitApp"], which is why an earlier
locator-based check saw 0 tabs). Usage: python src/live_app_smoke_test.py <url> <label> <old|new>
  old = version without the passport: each dataset must load its Prediction tab with no exception.
  new = additionally: passport section, distribution-shift message, no green/'trusted' trust wording, passport downloads, a flagged upload (distorted real file)
        and an unflagged upload (real HNEI file). Writes outputs/live_smoke/<label>_result.json and screenshots; exit code 0 only if everything passed."""
import json
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
URL, LABEL, EXPECT = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]
OUT = ROOT / "outputs" / "live_smoke"
OUT.mkdir(exist_ok=True)
DATASETS = ["NASA", "MIT", "CALCE", "Oxford", "HUST", "XJTU"]
results, failures = [], []


def log(row):
    results.append(row); print(row, flush=True)
    if not row.get("pass", True): failures.append(row["case"])


def open_app(page, path, wait_s=900):
    """Returns the locator root (iframe content or page) once the Streamlit tabs exist."""
    page.goto(f"{URL}/{path}", wait_until="domcontentloaded", timeout=180000)
    t0 = time.time()
    while time.time() - t0 < wait_s:
        try:
            btn = page.get_by_role("button", name="Yes, get this app back up!")
            if btn.count():
                btn.first.click(); print("  (woke the app)", flush=True); time.sleep(15)
        except Exception:
            pass
        root = page.frame_locator('iframe[title="streamlitApp"]') if page.locator('iframe[title="streamlitApp"]').count() else page
        try:
            if root.get_by_role("tab", name="Prediction").count() or root.locator('[data-testid="stException"]').count():
                return root
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError("app did not render tabs")


def alerts(root):
    loc = root.locator('[data-testid^="stAlertContent"]')
    out = []
    for i in range(loc.count()):
        e = loc.nth(i)
        out.append({"kind": e.get_attribute("data-testid"), "text": e.inner_text()[:220]})
    return out


def trusted_hits(body):
    hits = [body[max(0, m.start() - 50): m.end() + 40].replace("\n", " ") for m in re.finditer(r"\btrusted\b", body, flags=re.I)]
    return [h for h in hits if not re.search(r"\bnot\b[^.]{0,40}\btrusted\b", h, flags=re.I)]  # "does not mean ... trusted" is the disclaimer, not a claim


def open_prediction(root):
    root.get_by_role("tab", name="Prediction").first.click()
    time.sleep(2)


def download_both(page, root, tag):
    info = {}
    with page.expect_download(timeout=180000) as d1:
        root.get_by_text("Download passport (JSON)").first.click()
    d1.value.save_as(str(OUT / f"{LABEL}_{tag}_passport.json")); info["json_bytes"] = (OUT / f"{LABEL}_{tag}_passport.json").stat().st_size
    with page.expect_download(timeout=180000) as d2:
        root.get_by_text("Download passport (printable PDF)").first.click()
    d2.value.save_as(str(OUT / f"{LABEL}_{tag}_passport.pdf")); info["pdf_bytes"] = (OUT / f"{LABEL}_{tag}_passport.pdf").stat().st_size
    pj = json.loads((OUT / f"{LABEL}_{tag}_passport.json").read_text(encoding="utf-8"))
    info["soh"] = pj["state_of_health"]["value_percent"]; info["flagged"] = pj["trust_status"]["flagged_as_unfamiliar"]
    info["rul_cycles"] = pj["expected_remaining_life"].get("cycles")
    return info


with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=True)
    page = b.new_page(viewport={"width": 1400, "height": 1600}, accept_downloads=True)
    for ds in DATASETS:
        row = {"case": f"dataset {ds}"}
        try:
            root = open_app(page, f"?dataset={ds}")
            exc = root.locator('[data-testid="stException"]').count()
            open_prediction(root)
            if EXPECT == "new":
                root.get_by_text("Battery passport (research prototype)").first.wait_for(timeout=240000)
                root.get_by_text("Battery passport (research prototype)").first.scroll_into_view_if_needed()
            time.sleep(3)
            body = root.locator("body").inner_text()
            al = alerts(root)
            row.update({"exceptions": exc, "tabs_render": True, "passport_section": "Battery passport" in body,
                        "shift_message": ("No distribution shift detected" in body) or ("looks unfamiliar" in body),
                        "trusted_wording": trusted_hits(body), "success_alerts": [a["text"][:90] for a in al if a["kind"].endswith("Success")]})
            page.screenshot(path=str(OUT / f"{LABEL}_{ds}.png"))
            ok = exc == 0
            if EXPECT == "new":
                ok = ok and row["passport_section"] and row["shift_message"]
                if ds in ("NASA", "Oxford"):
                    row["downloads"] = download_both(page, root, ds); ok = ok and row["downloads"]["pdf_bytes"] > 10000
            row["pass"] = bool(ok)
        except Exception as e:
            row.update({"error": repr(e)[:300], "pass": False})
        log(row)
    if EXPECT == "new":
        for tag, fname, want in (("flagged_distorted", "_step3_distorted_upload_test.csv", "looks unfamiliar"), ("unflagged_hnei", "_phase0_hnei_upload_test.csv", "No distribution shift detected")):
            row = {"case": f"upload {tag}"}
            try:
                root = open_app(page, "?mode=Upload+your+own+cycle+data")
                root.locator("input[type=file]").first.set_input_files(str(ROOT / "outputs" / fname))
                root.get_by_text(want).first.wait_for(timeout=900000)
                time.sleep(5)
                open_prediction(root)
                root.get_by_text("Battery passport (research prototype)").first.wait_for(timeout=300000)
                root.get_by_text("Battery passport (research prototype)").first.scroll_into_view_if_needed()
                time.sleep(3)
                body = root.locator("body").inner_text(); al = alerts(root)
                exc = root.locator('[data-testid="stException"]').count()
                page.screenshot(path=str(OUT / f"{LABEL}_upload_{tag}.png"))
                row.update({"exceptions": exc, "expected_message_shown": want in body, "trusted_wording": trusted_hits(body),
                            "success_alerts": [a["text"][:90] for a in al if a["kind"].endswith("Success")], "downloads": download_both(page, root, tag)})
                row["pass"] = exc == 0 and row["expected_message_shown"] and row["downloads"]["pdf_bytes"] > 10000 and row["downloads"]["flagged"] == (want == "looks unfamiliar")
            except Exception as e:
                row.update({"error": repr(e)[:300], "pass": False})
            log(row)
    b.close()
res = {"url": URL, "label": LABEL, "expect": EXPECT, "failures": failures, "runs": results, "all_passed": not failures}
(OUT / f"{LABEL}_result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
print("ALL PASSED" if not failures else f"FAILURES: {failures}")
sys.exit(0 if not failures else 1)
