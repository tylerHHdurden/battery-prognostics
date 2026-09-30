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


# ---------------------------------------------------------------------------
# Toolkit Phase 2C: label-efficient checkpoints - the model predicts every
# cycle (via `interval()`, unchanged), but the TRUE label is only ever
# revealed to `update()` at chosen checkpoints, not every cycle. Three
# schedule policies, all built to respect the same no-lookahead contract
# `OnlineConformal` itself already keeps (see module docstring): a
# schedule's decision for cycle t may use only information available up
# to and including t, never a later cycle's true label.
# ---------------------------------------------------------------------------

def fixed_every_n_schedule(battery_length: int, budget: int) -> list[int]:
    """Reveal `budget` labels spaced as evenly as possible across the
    battery's own KNOWN TOTAL LENGTH. Precomputed once, upfront, from a
    quantity (total cycle count) this project's evaluation setting
    already knows in advance - not from any LABEL VALUE - so this is not
    a no-lookahead violation in the sense that property is about (using
    a future measurement's value early), the same way a lab knowing a
    planned test's total duration upfront isn't "cheating" the way
    peeking at a future SOH reading would be."""
    if budget <= 0 or battery_length <= 0:
        return []
    step = battery_length / budget
    return sorted(set(min(battery_length - 1, int(round(i * step))) for i in range(budget)))


def life_stage_schedule(battery_length: int, budget: int, gamma: float = 0.5) -> list[int]:
    """Same upfront-only information as `fixed_every_n_schedule` (total
    length, never a label), but concentrated late in life via gamma<1
    (a concave power-law warp of the [0,1] budget index - checkpoint
    spacing shrinks as cycle index grows, since SOH changes faster and
    matters more for RUL near end-of-life)."""
    if budget <= 0 or battery_length <= 0:
        return []
    denom = max(1, budget - 1)
    fracs = [(k / denom) ** gamma for k in range(budget)]
    return sorted(set(min(battery_length - 1, int(round(f * (battery_length - 1)))) for f in fracs))


class UncertaintyTriggeredScheduler:
    """Chooses which cycles get a label revealed using ONLY the current
    conformal interval width and an ADWIN drift detector fed exclusively
    with ALREADY-REVEALED residuals - never a future label. Guarantees
    exactly `budget` reveals total (for a fair, equal-budget comparison
    against the other two policies) by partitioning the battery into
    `budget` equal-length slots upfront (same upfront-only "known total
    length" information the other two schedules use) and revealing once
    per slot: at the FIRST cycle in that slot where the interval
    half-width exceeds `width_threshold` OR ADWIN signals drift,
    otherwise at the slot's own last cycle (so every slot contributes
    exactly one reveal even if neither trigger ever fires)."""

    def __init__(self, battery_length: int, budget: int, width_threshold: float, adwin_delta: float = 0.002):
        from river.drift import ADWIN
        edges = np.linspace(0, battery_length, budget + 1)
        self.slots = [(int(edges[i]), int(edges[i + 1]) - 1) for i in range(budget)]
        self.width_threshold = width_threshold
        self.adwin = ADWIN(delta=adwin_delta)
        self._slot_idx = 0
        self._max_cycle_seen = -1  # no-lookahead guard: cycles must arrive in non-decreasing order

    def should_reveal(self, cycle_idx: int, current_interval_width: float) -> bool:
        """Call once per cycle, BEFORE that cycle's true label is known -
        current_interval_width must come from `OnlineConformal.interval()`
        (already causal) computed at this same cycle."""
        assert cycle_idx >= self._max_cycle_seen, \
            f"no-lookahead violation: cycle {cycle_idx} arrived after cycle {self._max_cycle_seen}"
        self._max_cycle_seen = cycle_idx
        if self._slot_idx >= len(self.slots):
            return False
        lo, hi = self.slots[self._slot_idx]
        if cycle_idx < lo:
            return False
        if cycle_idx > hi:
            self._slot_idx += 1  # missed this slot's window - force a reveal at the next opportunity
            return True
        triggered = current_interval_width > self.width_threshold or self.adwin.drift_detected
        if cycle_idx == hi or triggered:
            self._slot_idx += 1
            return True
        return False

    def observe_residual(self, resid: float) -> None:
        """Call ONLY immediately after a label was actually revealed at
        the current cycle (never for an unrevealed cycle) - feeds ADWIN
        the residual just observed, so its own drift state is built
        exclusively from labels already seen, same causal guarantee as
        everything else in this module."""
        self.adwin.update(resid)


def run_batch_with_schedule(y: np.ndarray, yhat: np.ndarray, q_src: float, method: str,
                             checkpoints: set[int] | None, alpha: float = ALPHA_DEFAULT,
                             k_burnin: int = 10, cyc: np.ndarray | None = None,
                             scheduler: "UncertaintyTriggeredScheduler | None" = None,
                             **params) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same per-cycle interval() contract as `run_batch`, but `update()`
    (the ONLY method that reveals a label to the tracker) is called ONLY
    at chosen checkpoints - every other cycle gets a real, causal
    interval from whatever state was last updated, and nothing more.

    Exactly one of `checkpoints` (a precomputed set, for the fixed/
    life-stage schedules) or `scheduler` (for the uncertainty-triggered
    schedule, decided cycle-by-cycle) must be given. Returns (lo, hi,
    revealed_mask) - revealed_mask lets the caller verify the actual
    label budget spent matches what was intended."""
    assert (checkpoints is None) != (scheduler is None), \
        "run_batch_with_schedule needs exactly one of checkpoints or scheduler"
    n = len(y)
    oc = OnlineConformal(q_src, method, alpha=alpha, k_burnin=k_burnin, **params)
    lo, hi = np.empty(n), np.empty(n)
    revealed = np.zeros(n, dtype=bool)
    for t in range(n):
        c = cyc[t] if cyc is not None else t
        lo[t], hi[t] = oc.interval(yhat[t], cycle_idx=c)
        if checkpoints is not None:
            reveal = t in checkpoints
        else:
            reveal = scheduler.should_reveal(t, hi[t] - lo[t])
        if reveal:
            oc.update(y[t], yhat[t], cycle_idx=c)
            revealed[t] = True
            if scheduler is not None:
                scheduler.observe_residual(abs(float(y[t]) - float(yhat[t])))
    return lo, hi, revealed
