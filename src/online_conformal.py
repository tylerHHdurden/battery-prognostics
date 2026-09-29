"""
Reusable online conformal prediction module - the PID (Angelopoulos,
Candes, Tibshirani, NeurIPS 2023) and nexCP (Barber et al., Ann.
Statist. 2023) trackers originally written for the final research
pass's item A (`run_finalpass2_itemA_online_conformal.py`), moved here
unchanged in their core recursion so this project's own item A results
remain reproducible byte-for-byte, and exposed behind a clean, stateful
streaming API for reuse (the "Analyse your battery" page, and any
future caller).

USAGE (matches the no-lookahead contract every caller must respect):

    oc = OnlineConformal(q_src=1.14, method="PID", eta=0.1, k_burnin=10)
    for t in cycles:
        lo, hi = oc.interval(yhat[t])   # uses ONLY state from cycles < t
        # ... show (lo, hi) to the user ...
        oc.update(y_true[t], yhat[t])   # NOW t's own label is allowed to affect state

Calling `interval()` twice in a row without an intervening `update()`
returns the SAME interval (idempotent) - it never mutates state itself.
`update()` is the only method that advances internal state, and it may
only be called with a label the caller has ALREADY observed for that
cycle - the module has no way to enforce this from outside, which is
exactly why the no-lookahead unit test (`test_online_conformal.py`)
exists: it verifies the RECURSION ITSELF has no lookahead, given a
correctly-ordered call sequence.
"""
from __future__ import annotations

import numpy as np

ALPHA_DEFAULT = 0.1


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    cw_norm = cw / cw[-1]
    idx = np.searchsorted(cw_norm, q)
    idx = min(idx, len(v) - 1)
    return float(v[idx])


class OnlineConformal:
    """Stateful online conformal interval tracker. method="PID" or
    "nexCP" - see module docstring for the required call order."""

    def __init__(self, q_src: float, method: str, alpha: float = ALPHA_DEFAULT, k_burnin: int = 10, **params):
        if method not in ("PID", "nexCP"):
            raise ValueError(f"method must be 'PID' or 'nexCP', got {method!r}")
        self.method = method
        self.alpha = alpha
        self.k_burnin = k_burnin
        self.q_src = float(q_src)
        self.t = 0

        if method == "PID":
            if "eta" not in params:
                raise ValueError("PID requires eta=... (recommended: 0.1 * max(|residual|) over a burn-in window)")
            self.eta = float(params["eta"])
            self.scorecaster = bool(params.get("scorecaster", False))
            self.scorecaster_refit_every = int(params.get("scorecaster_refit_every", 20))
            self.q = self.q_src
            self._cyc_history: list[float] = []
            self._resid_history: list[float] = []
            self._trend_a, self._trend_b = 0.0, 0.0
        else:  # nexCP
            self.rho = float(params.get("rho", 0.95))
            if not (0.0 < self.rho < 1.0):
                raise ValueError("rho must be in (0, 1)")
            self.window = int(np.ceil(np.log(1e-4) / np.log(self.rho)))  # rho**window < 1e-4
            self._resid_history: list[float] = []

    def interval(self, yhat: float, cycle_idx: float | None = None) -> tuple[float, float]:
        """Returns (lo, hi) using ONLY state accumulated from previous
        `update()` calls - never mutates state, safe to call repeatedly."""
        q = self._current_q(cycle_idx)
        return float(yhat) - q, float(yhat) + q

    def update(self, y_true: float, yhat: float, cycle_idx: float | None = None) -> None:
        """Advances internal state using THIS cycle's own (now-observed)
        label. Must be called with cycle_idx values in non-decreasing
        order across the object's lifetime (the caller's own
        responsibility - see the no-lookahead unit test for what this
        module itself guarantees given a correct call order)."""
        resid = abs(float(y_true) - float(yhat))
        if self.method == "PID":
            q_used = self._current_q(cycle_idx)
            err = 1.0 if resid > q_used else 0.0
            self.q = max(0.0, self.q + self.eta * (err - self.alpha))
            self._resid_history.append(resid)
            self._cyc_history.append(cycle_idx if cycle_idx is not None else self.t)
        else:  # nexCP
            self._resid_history.append(resid)
        self.t += 1

    def _current_q(self, cycle_idx: float | None) -> float:
        if self.method == "PID":
            q = self.q
            if self.scorecaster:
                # Refit HERE (inside the read path, called by interval()
                # BEFORE update() for this same cycle) using only PAST
                # history (self._resid_history/_cyc_history hold exactly
                # cycles < self.t at this point) - matches the original
                # pid_tracker's own same-iteration refit-then-use timing
                # and its resid[:t] (strictly past) slicing exactly; a
                # refit inside update() instead would use one extra,
                # just-observed residual and apply it one cycle late -
                # verified as a real mismatch by this module's own test
                # suite before this fix.
                if self.t >= 2 and self.t % self.scorecaster_refit_every == 0:
                    cyc_arr = np.array(self._cyc_history, dtype=float)
                    resid_arr = np.array(self._resid_history, dtype=float)
                    self._trend_b, self._trend_a = np.polyfit(cyc_arr, resid_arr, 1)
                cyc = cycle_idx if cycle_idx is not None else self.t
                forecast = max(0.0, self._trend_a + self._trend_b * cyc)
                q = q + forecast
            return q
        # nexCP
        if self.t < self.k_burnin or not self._resid_history:
            return self.q_src
        past = np.array(self._resid_history[-self.window:], dtype=float)
        ages = np.arange(len(past), 0, -1)
        weights = self.rho ** ages
        return weighted_quantile(past, weights, 1 - self.alpha)

    def reset(self) -> None:
        """Reinitializes to the same state as a freshly-constructed
        object with the same q_src/params - for reuse across multiple
        batteries without rebuilding the object."""
        self.t = 0
        if self.method == "PID":
            self.q = self.q_src
            self._cyc_history = []
            self._resid_history = []
            self._trend_a, self._trend_b = 0.0, 0.0
        else:
            self._resid_history = []


def run_batch(y: np.ndarray, yhat: np.ndarray, q_src: float, method: str,
              alpha: float = ALPHA_DEFAULT, k_burnin: int = 10, cyc: np.ndarray | None = None,
              **params) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper for scoring a whole pre-computed (y, yhat)
    battery trace at once - byte-for-byte equivalent to this project's
    own item A `pid_tracker`/`nexcp_tracker` functions (verified by
    `test_online_conformal.py::test_matches_item_a_batch_functions`),
    used by every finalpass* script and the toolkit's own gate/report
    scripts so there is exactly ONE implementation of this recursion
    in the codebase from this point forward."""
    n = len(y)
    oc = OnlineConformal(q_src, method, alpha=alpha, k_burnin=k_burnin, **params)
    lo, hi = np.empty(n), np.empty(n)
    for t in range(n):
        c = cyc[t] if cyc is not None else t
        lo[t], hi[t] = oc.interval(yhat[t], cycle_idx=c)
        oc.update(y[t], yhat[t], cycle_idx=c)
    return lo, hi
