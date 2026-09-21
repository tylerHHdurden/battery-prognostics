"""
Regression sweep after the CALCE embedding-normalization fix: base app
load + all 6 dataset selections (NASA/MIT/CALCE/Oxford/HUST/XJTU),
confirming zero exceptions. Uses the project's own established AppTest
convention (see app.py's own "Bug 2" note: "Verified via
streamlit.testing.v1.AppTest across 4 paths, zero exceptions").

NOTE on what this DOES and does NOT re-verify: NASA/MIT/CALCE raw data
IS present locally (data/raw/), so `dataset_available` is True for all
3 here and the app takes the LIVE raw-cycle path (predict_and_explain),
NOT the precomputed-fallback path (predict_and_explain_precomputed)
this fix touches - this sweep's job for those 3 is confirming the app
still loads/renders cleanly, not re-testing the fix's own accuracy
(already done directly and thoroughly in verify_calce_embedding_fix.py,
scoring all 2941 CALCE rows through the actual production function).
Oxford/HUST/XJTU are never in the `dataset_available` dict at all (see
app.py) - they ALWAYS take the precomputed path, in every environment,
so this sweep DOES exercise that path for them, using the SAME
`res["norm_stats"]`/load_resources() plumbing the CALCE fix changed.
"""
import time
from streamlit.testing.v1 import AppTest

DATASETS = ["NASA", "MIT", "CALCE", "Oxford", "HUST", "XJTU"]


def main():
    t0 = time.time()
    print("=== Regression sweep: base load + 6 dataset selections, post-CALCE-fix ===")

    at = AppTest.from_file("app.py", default_timeout=120)
    at.run()
    print(f"[sweep] base load: exceptions={len(at.exception)}")
    if at.exception:
        for e in at.exception:
            print(f"  EXCEPTION: {e}")

    all_clean = len(at.exception) == 0
    for ds in DATASETS:
        t1 = time.time()
        at.query_params["dataset"] = ds
        at.run()
        n_exc = len(at.exception)
        all_clean = all_clean and n_exc == 0
        status = "OK" if n_exc == 0 else "FAILED"
        print(f"[sweep] dataset={ds}: exceptions={n_exc} ({time.time()-t1:.1f}s) -> {status}")
        if at.exception:
            for e in at.exception:
                print(f"  EXCEPTION: {e}")

    print(f"\n[sweep] TOTAL TIME: {(time.time()-t0)/60:.1f} minutes")
    print("[sweep] ALL CLEAN - zero exceptions across base load + all 6 datasets" if all_clean
          else "[sweep] FAILURES FOUND - see above")
    return all_clean


if __name__ == "__main__":
    main()
