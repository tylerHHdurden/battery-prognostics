import sys, json
from pathlib import Path
from streamlit.testing.v1 import AppTest
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import live_inference as li
SRC = list(li.EXTENSION_SOURCES)
rows = []
for s in SRC:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.run()
    at.sidebar.selectbox[[b.label for b in at.sidebar.selectbox].index("Dataset")].set_value(s).run()
    exc = [e.value for e in at.exception]
    txt = " ".join([m.value for m in at.markdown] + [c.value for c in at.caption] + [w.value for w in at.warning] + [i.value for i in at.info])
    cap = any("Extension source" in c.value for c in at.sidebar.caption)
    flagged = "looks unfamiliar" in txt
    rul_shown = any("remaining" in str(m.label).lower() and m.value not in ("n/a", "hidden", "None") for m in at.metric if "RUL" in str(m.label) or "remaining" in str(m.label).lower())
    bat = at.sidebar.selectbox[[b.label for b in at.sidebar.selectbox].index("Battery")].value
    ctx = li.predict_and_explain_precomputed(s, bat, int(at.sidebar.select_slider[0].value), li.load_resources()) if False else None
    rows.append(dict(source=s, battery=bat, exceptions=exc, ext_caption=cap, flagged=flagged,
                     soh_metrics=[(m.label, m.value) for m in at.metric][:6],
                     downloads=len([b for b in at.get("download_button")]) if hasattr(at, "get") else None,
                     passport_header=any("Battery passport" in str(x.value) for x in at.subheader)))
    print(rows[-1], flush=True)
json.dump(rows, open(ROOT / "outputs/site_redesign/_extension_apptest.json", "w"), indent=1, default=str)
