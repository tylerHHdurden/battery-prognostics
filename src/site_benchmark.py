"""Cold/warm render time of the app through AppTest (no browser): fresh process -> AppTest.run() for a dataset (cold: imports, cache_resource loads, first render),
then a second run in the same process (warm). Usage: python src/site_benchmark.py <label> [dataset]  -> outputs/site_redesign/bench_<label>.json"""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
label = sys.argv[1]; ds = sys.argv[2] if len(sys.argv) > 2 else "Oxford"
t0 = time.time()
from streamlit.testing.v1 import AppTest
res = {"label": label, "dataset": ds}
for k in ("cold_s", "warm_s"):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600); at.query_params["dataset"] = ds
    t = time.time(); at.run(); res[k] = round(time.time() - t, 1); res[k.replace("_s", "_exceptions")] = len(at.exception)
    res[k.replace("_s", "_elements")] = sum(len(getattr(at, n)) for n in ("markdown", "caption", "metric", "dataframe", "image", "expander", "subheader"))
(ROOT / "outputs" / "site_redesign" / f"bench_{label}.json").write_text(json.dumps(res, indent=1)); print(res)
