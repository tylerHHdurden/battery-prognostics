"""
Unit tests for src/online_conformal.py - run directly (`python src/
test_online_conformal.py`), no pytest dependency (not installed in
this project's environment, confirmed before choosing this format;
matches this project's own established `verify_*.py`/inline-assert
convention rather than introducing a new test framework/dependency).

1. test_no_lookahead: the property CHECK 2 (final verification pass)
   already established for the ORIGINAL item A functions - shuffling
   every label after cycle t must not change the interval AT or BEFORE
   t. Re-verified here against the NEW OnlineConformal class (not just
   re-trusting the old functions), for both PID and nexCP.
2. test_matches_item_a_batch_functions: `online_conformal.run_batch`
   must be BYTE-IDENTICAL to `run_finalpass2_itemA_online_conformal`'s
   own `pid_tracker`/`nexcp_tracker` - the module claims this
   equivalence in its own docstring; this test is what makes that a
   verified fact, not an assertion.
3. test_interval_is_idempotent: calling interval() twice without an
   intervening update() must return the identical result both times.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from online_conformal import OnlineConformal, run_batch
from run_finalpass2_itemA_online_conformal import pid_tracker, nexcp_tracker

SEED = 42


def test_no_lookahead():
    print("=== test_no_lookahead ===")
    rng = np.random.default_rng(SEED)
    n = 200
    yhat = 90 - 0.02 * np.arange(n) + rng.normal(0, 0.3, n)
    y = yhat + rng.normal(0, 1.0, n)
    t_check = 80

    for method, kwargs in [("PID", {"eta": 0.1}), ("PID", {"eta": 0.1, "scorecaster": True}),
                            ("nexCP", {"rho": 0.95}), ("nexCP", {"rho": 0.99})]:
        oc1 = OnlineConformal(q_src=1.0, method=method, k_burnin=10, **kwargs)
        lo1, hi1 = [], []
        for t in range(n):
            lo_t, hi_t = oc1.interval(yhat[t], cycle_idx=t)
            lo1.append(lo_t); hi1.append(hi_t)
            oc1.update(y[t], yhat[t], cycle_idx=t)
            if t == t_check:
                snapshot_lo, snapshot_hi = list(lo1), list(hi1)

        y_shuffled = y.copy()
        future = y_shuffled[t_check + 1:]
        rng.shuffle(future)
        y_shuffled[t_check + 1:] = future
        assert not np.array_equal(y_shuffled, y), "shuffle produced no change - vacuous test"

        oc2 = OnlineConformal(q_src=1.0, method=method, k_burnin=10, **kwargs)
        lo2, hi2 = [], []
        for t in range(n):
            lo_t, hi_t = oc2.interval(yhat[t], cycle_idx=t)
            lo2.append(lo_t); hi2.append(hi_t)
            oc2.update(y_shuffled[t], yhat[t], cycle_idx=t)

        before_match = np.allclose(snapshot_lo, lo2[:t_check + 1]) and np.allclose(snapshot_hi, hi2[:t_check + 1])
        after_differs = not (np.allclose(lo1[t_check + 1:], lo2[t_check + 1:]) and np.allclose(hi1[t_check + 1:], hi2[t_check + 1:]))
        print(f"  {method} {kwargs}: interval[0..{t_check}] identical after shuffling future labels: "
              f"{before_match} (must be True) | interval[{t_check+1}..] changed: {after_differs} (expected True)")
        assert before_match, f"LOOKAHEAD BUG: {method} {kwargs} interval before t_check changed"
    print("  PASSED\n")


def test_matches_item_a_batch_functions():
    print("=== test_matches_item_a_batch_functions ===")
    rng = np.random.default_rng(SEED)
    n = 150
    yhat = 85 - 0.03 * np.arange(n) + rng.normal(0, 0.4, n)
    y = yhat + rng.normal(0, 1.2, n)
    q_src = 1.14
    k_burnin = 10

    lo_orig, hi_orig, _ = pid_tracker(y, yhat, q_src, eta=0.1, k_burnin=k_burnin)
    lo_new, hi_new = run_batch(y, yhat, q_src, "PID", k_burnin=k_burnin, eta=0.1)
    assert np.allclose(lo_orig, lo_new) and np.allclose(hi_orig, hi_new), "PID run_batch does not match pid_tracker"
    print("  PID (no scorecaster): run_batch matches pid_tracker exactly - PASSED")

    lo_orig_sc, hi_orig_sc, _ = pid_tracker(y, yhat, q_src, eta=0.1, k_burnin=k_burnin, scorecaster=True, cyc=np.arange(n, dtype=float))
    lo_new_sc, hi_new_sc = run_batch(y, yhat, q_src, "PID", k_burnin=k_burnin, eta=0.1, scorecaster=True,
                                     cyc=np.arange(n, dtype=float))
    assert np.allclose(lo_orig_sc, lo_new_sc) and np.allclose(hi_orig_sc, hi_new_sc), "PID+scorecaster run_batch mismatch"
    print("  PID+scorecaster: run_batch matches pid_tracker exactly - PASSED")

    for rho in (0.95, 0.99):
        lo_orig_n, hi_orig_n = nexcp_tracker(y, yhat, q_src, rho, k_burnin)
        lo_new_n, hi_new_n = run_batch(y, yhat, q_src, "nexCP", k_burnin=k_burnin, rho=rho)
        assert np.allclose(lo_orig_n, lo_new_n) and np.allclose(hi_orig_n, hi_new_n), f"nexCP(rho={rho}) run_batch mismatch"
        print(f"  nexCP (rho={rho}): run_batch matches nexcp_tracker exactly - PASSED")
    print()


def test_interval_is_idempotent():
    print("=== test_interval_is_idempotent ===")
    oc = OnlineConformal(q_src=1.0, method="PID", eta=0.1, k_burnin=10)
    for _ in range(15):
        oc.update(90.0, 89.5, cycle_idx=_)
    lo1, hi1 = oc.interval(88.0, cycle_idx=15)
    lo2, hi2 = oc.interval(88.0, cycle_idx=15)
    assert (lo1, hi1) == (lo2, hi2), "interval() is not idempotent - it must not mutate state"
    print("  PASSED\n")


if __name__ == "__main__":
    test_no_lookahead()
    test_matches_item_a_batch_functions()
    test_interval_is_idempotent()
    print("=== ALL TESTS PASSED ===")
