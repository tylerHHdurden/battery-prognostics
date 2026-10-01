"""Full staging/live test of the redesigned app (iframe-aware). Usage: python src/staging_full_test.py <url> <label> [fast]
Phases: A desktop - six datasets + sample of extension sources in Browse mode (render time, result card, every tab, exceptions, green elements, trusted wording,
extension caution caption, RUL hidden for extension sources); B downloads - passport JSON/PDF on NASA, Oxford and one extension source; C uploads - flagged (distorted real file)
and unflagged (real HNEI file): result card + message, sample-file download, passport downloads; D mobile (390x844) - Oxford, one extension source, unflagged upload, every tab,
no horizontal scroll. Writes outputs/staging_test/<label>/result.json + screenshots; exit 0 only if all pass."""
import json
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL, LABEL = sys.argv[1].rstrip("/"), sys.argv[2]
FAST = len(sys.argv) > 3
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "staging_test" / LABEL
OUT.mkdir(parents=True, exist_ok=True)
SIX = ["NASA", "MIT", "CALCE", "Oxford", "HUST", "XJTU"]
EXT = ["ul_pur", "hnei", "snl", "mich", "tongji", "isu_ilcc"] if not FAST else ["snl"]
TABS = ["Showcase", "Prediction", "Explainability", "Health Report", "Streaming Digital Twin", "World Model", "Model Validation", "Full Results Archive"]
DESK, MOB = {"width": 1400, "height": 1000}, {"width": 390, "height": 844}
results, failures = [], []


def log(row):
    results.append(row)
    print({k: (v if not isinstance(v, (list, dict)) or len(str(v)) < 140 else str(v)[:140]) for k, v in row.items()}, flush=True)
    if not row.get("pass", True):
        failures.append(row["case"])


def root_of(pg):
    return pg.frame_locator('iframe[title="streamlitApp"]') if pg.locator('iframe[title="streamlitApp"]').count() else pg


def open_app(b, path, vp, wait_s=600):
    pg = b.new_page(viewport=vp, accept_downloads=True)
    t0 = time.time()
    pg.goto(f"{URL}/{path}", wait_until="domcontentloaded", timeout=180000)
    while time.time() - t0 < wait_s:
        btn = pg.get_by_role("button", name="Yes, get this app back up!")
        if btn.count():
            btn.first.click()
            time.sleep(15)
        root = root_of(pg)
        try:
            if root.get_by_role("tab", name="Showcase").count() or root.locator('[data-testid="stException"]').count():
                break
        except Exception:
            pass
        time.sleep(1)
    return pg, root_of(pg), round(time.time() - t0, 1)


def trusted(body):
    hits = [body[max(0, m.start() - 50): m.end() + 40] for m in re.finditer(r"\btrusted\b", body, flags=re.I)]
    return [h.replace("\n", " ") for h in hits if not re.search(r"\bnot\b[^.]{0,40}\btrusted\b", h, flags=re.I)]


def tabs_pass(pg, root, tag, shot_tabs=("Prediction",)):
    exc = {}
    for t in TABS:
        tab = root.get_by_role("tab", name=t).first
        tab.scroll_into_view_if_needed()
        tab.click()
        time.sleep(3 if FAST else 5)
        exc[t] = root.locator('[data-testid="stException"]').count()
        if t in shot_tabs:
            pg.screenshot(path=str(OUT / f"{tag}_{t.replace(' ', '_')}.png"))
    return exc


def passport_downloads(pg, root, tag):
    root.get_by_role("tab", name="Prediction").first.click()
    root.get_by_text("Battery passport (research prototype)").first.wait_for(timeout=300000)
    root.get_by_text("Battery passport (research prototype)").first.scroll_into_view_if_needed()
    time.sleep(2)
    with pg.expect_download(timeout=180000) as d1:
        root.get_by_text("Download passport (JSON)").first.click()
    d1.value.save_as(str(OUT / f"{tag}_passport.json"))
    with pg.expect_download(timeout=180000) as d2:
        root.get_by_text("Download passport (printable PDF)").first.click()
    d2.value.save_as(str(OUT / f"{tag}_passport.pdf"))
    pj = json.loads((OUT / f"{tag}_passport.json").read_text(encoding="utf-8"))
    return {"json_bytes": (OUT / f"{tag}_passport.json").stat().st_size, "pdf_bytes": (OUT / f"{tag}_passport.pdf").stat().st_size,
            "soh": pj["state_of_health"]["value_percent"], "flagged": pj["trust_status"]["flagged_as_unfamiliar"],
            "rul": pj["expected_remaining_life"].get("cycles")}


with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=True)
    # ---- A + B: datasets (desktop), passport downloads on three of them
    for ds in SIX + EXT:
        row = {"case": f"A dataset {ds}", "viewport": "desktop"}
        try:
            pg, root, t_render = open_app(b, f"?dataset={ds}", DESK)
            row["first_render_s"] = t_render
            root.get_by_text("Predicted SOH").first.wait_for(timeout=300000)
            time.sleep(2)
            pg.screenshot(path=str(OUT / f"A_{ds}_top.png"))
            body = root.locator("body").inner_text()
            row["result_card"] = True
            row["rul_not_available"] = ("not available" in body) if ds in EXT else None
            row["extension_caution"] = ("Extension source" in body) if ds in EXT else None
            exc = tabs_pass(pg, root, f"A_{ds}")
            row["exceptions_by_tab"] = exc
            row["green"] = root.locator('[data-testid="stAlertContentSuccess"]').count()
            row["trusted_wording"] = trusted(root.locator("body").inner_text())
            ok = sum(exc.values()) == 0 and row["green"] == 0 and not row["trusted_wording"]
            if ds in EXT:
                ok = ok and bool(row["extension_caution"]) and bool(row["rul_not_available"])
            if ds in ("NASA", "Oxford") or ds == EXT[0]:
                row["downloads"] = passport_downloads(pg, root, f"B_{ds}")
                ok = ok and row["downloads"]["pdf_bytes"] > 10000 and row["downloads"]["json_bytes"] > 1500
                if ds in EXT or ds == "Oxford":
                    ok = ok and row["downloads"]["rul"] is None
            row["pass"] = bool(ok)
            pg.close()
        except Exception as e:
            row.update({"error": repr(e)[:300], "pass": False})
        log(row)
    # ---- C: uploads (desktop)
    for tag, fname, want, flag in (("flagged_distorted", "_step3_distorted_upload_test.csv", "Unfamiliar battery", True),
                                   ("unflagged_hnei", "_phase0_hnei_upload_test.csv", "No distribution shift detected", False)):
        row = {"case": f"C upload {tag}", "viewport": "desktop"}
        try:
            pg, root, t_render = open_app(b, "?mode=Upload+your+own+cycle+data", DESK)
            if tag == "unflagged_hnei":
                with pg.expect_download(timeout=120000) as d:
                    root.get_by_text("Download a sample file").first.click()
                d.value.save_as(str(OUT / "sample_download.csv"))
                row["sample_download_bytes"] = (OUT / "sample_download.csv").stat().st_size
            root.locator("input[type=file]").first.wait_for(state="attached", timeout=180000)
            root.locator("input[type=file]").first.set_input_files(str(ROOT / "outputs" / fname))
            root.get_by_text("Predicted SOH").first.wait_for(timeout=600000)
            time.sleep(3)
            pg.screenshot(path=str(OUT / f"C_{tag}_top.png"))
            body = root.locator("body").inner_text()
            row["message_shown"] = want in body
            exc = tabs_pass(pg, root, f"C_{tag}")
            row["exceptions_by_tab"] = exc
            row["green"] = root.locator('[data-testid="stAlertContentSuccess"]').count()
            row["trusted_wording"] = trusted(root.locator("body").inner_text())
            row["downloads"] = passport_downloads(pg, root, f"C_{tag}")
            row["pass"] = bool(row["message_shown"] and sum(exc.values()) == 0 and row["green"] == 0 and not row["trusted_wording"]
                               and row["downloads"]["flagged"] == flag and row["downloads"]["pdf_bytes"] > 10000
                               and (tag != "unflagged_hnei" or row["sample_download_bytes"] > 1_000_000))
            pg.close()
        except Exception as e:
            row.update({"error": repr(e)[:300], "pass": False})
        log(row)
    # ---- D: mobile
    for case, path, upload in (("oxford", "?dataset=Oxford", None), ("ext_" + EXT[0], f"?dataset={EXT[0]}", None),
                               ("upload_unflagged", "?mode=Upload+your+own+cycle+data", "_phase0_hnei_upload_test.csv")):
        row = {"case": f"D mobile {case}", "viewport": "mobile"}
        try:
            pg, root, t_render = open_app(b, path, MOB)
            row["first_render_s"] = t_render
            if upload:
                root.locator("input[type=file]").first.wait_for(state="attached", timeout=180000)
                root.locator("input[type=file]").first.set_input_files(str(ROOT / "outputs" / upload))
            root.get_by_text("Predicted SOH").first.wait_for(timeout=600000)
            time.sleep(3)
            pg.screenshot(path=str(OUT / f"D_{case}_top.png"))
            row["horizontal_scroll"] = root.locator("body").evaluate("() => document.documentElement.scrollWidth > document.documentElement.clientWidth + 2")
            exc = tabs_pass(pg, root, f"D_{case}")
            row["exceptions_by_tab"] = exc
            row["green"] = root.locator('[data-testid="stAlertContentSuccess"]').count()
            row["pass"] = bool(sum(exc.values()) == 0 and row["green"] == 0 and not row["horizontal_scroll"])
            pg.close()
        except Exception as e:
            row.update({"error": repr(e)[:300], "pass": False})
        log(row)
    b.close()
res = {"url": URL, "label": LABEL, "failures": failures, "all_passed": not failures, "runs": results}
(OUT / "result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
print("ALL PASSED" if not failures else f"FAILURES: {failures}")
sys.exit(0 if not failures else 1)
