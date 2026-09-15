"""
Stage 6.3: swaps the streaming Digital Twin's linear-only
sklearn.SGDRegressor online corrector (session 28) for a River-based
non-linear online learner, plus adds standalone concept-drift
detection - a genuine capability upgrade to an already-working
component, not a fix to something broken. Subclasses StreamingDigitalTwin
(digital_twin_streaming.py) and overrides ONLY the corrector - every
other piece (frozen model inference, ACI conformal interval, OC-SVM
anomaly check) is reused unchanged via inheritance, not duplicated.

Model choice, stated explicitly: river.tree.HoeffdingAdaptiveTreeRegressor
- a SINGLE adaptive Hoeffding tree, not a full Adaptive Random Forest
ensemble. Reasoning: the corrector's own input is tiny (2 features -
scaled raw prediction, scaled cycle_idx), the same scale of problem
session 28's own SGDRegressor was solving; an ensemble of many trees
would be a heavier, slower, less interpretable choice for a problem
this small, with no evidence more capacity is needed (matches this
project's own repeated finding elsewhere - e.g. Stage 5.2's BatLiNet
work - that a bigger model is not automatically a better one for a
small, well-scoped correction task). HoeffdingAdaptiveTreeRegressor
specifically (not the plain, non-adaptive HoeffdingTreeRegressor) was
chosen because it has ADWIN-based drift detection BUILT IN to its own
node-replacement logic - directly matching this item's own "built-in
concept-drift awareness" framing, not bolted on separately for the
corrector itself (a SEPARATE, standalone ADWIN instance is also wired
in below, monitoring the raw-prediction residual stream directly, for
explicit, reportable drift events - the two serve different purposes:
the tree's internal ADWIN adapts model structure; the standalone one
reports human-readable drift flags).
"""
import numpy as np
from river import tree, drift

from digital_twin_streaming import StreamingDigitalTwin, MIN_HISTORY_FOR_CORRECTION


class StreamingDigitalTwinRiver(StreamingDigitalTwin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.corrector = tree.HoeffdingAdaptiveTreeRegressor(seed=42)
        self._corrector_fitted = False
        # standalone drift detector on the RAW-prediction residual
        # stream (true_soh - raw_pred) - reports genuine drift events
        # as its own capability, separate from the tree's internal
        # adaptation.
        self.drift_detector = drift.ADWIN()
        self.drift_events: list[dict] = []

    @staticmethod
    def _corrector_features_dict(raw_pred: float, cycle_idx: int) -> dict:
        # same centering/scaling convention as the SGD version's
        # _corrector_features, just as a dict (River's own input format)
        return {"raw_pred_scaled": (raw_pred - 85.0) / 15.0, "cycle_frac": cycle_idx / 100.0}

    def step(self, cycle: dict, true_soh: float | None = None) -> dict:
        raw_pred, feat = self._raw_predict(cycle)
        if raw_pred is None:
            return {"error": "cycle too short for the sequence/fusion pipeline", "cycle_idx": cycle["cycle_idx"]}

        ocsvm_scaled = self.ocsvm_scaler.transform(feat.reshape(1, -1))
        anomaly = bool(self.ocsvm.predict(ocsvm_scaled)[0] == -1)

        x_dict = self._corrector_features_dict(raw_pred, cycle["cycle_idx"])
        if self._corrector_fitted and self.n_revealed >= MIN_HISTORY_FOR_CORRECTION:
            correction = float(self.corrector.predict_one(x_dict))
            correction = float(np.clip(correction, -self.MAX_CORRECTION_PP, self.MAX_CORRECTION_PP))
        else:
            correction = 0.0
        corrected_pred = raw_pred + correction

        half_width = self.aci.current_half_width(self.fallback_half_width)

        record = {
            "cycle_idx": cycle["cycle_idx"], "raw_pred": raw_pred, "correction": correction,
            "corrected_pred": corrected_pred, "half_width": half_width, "anomaly": anomaly,
            "true_soh": true_soh, "n_revealed_so_far": self.n_revealed,
            "alpha_t": self.aci.alpha_t, "drift_detected": False,
        }

        if true_soh is not None:
            residual_target = true_soh - raw_pred
            self.corrector.learn_one(x_dict, residual_target)
            self._corrector_fitted = True

            covered = (corrected_pred - half_width) <= true_soh <= (corrected_pred + half_width)
            score = abs(true_soh - corrected_pred)
            self.aci.update(covered, score)
            self.n_revealed += 1

            # standalone drift detection on the raw-prediction residual
            abs_residual = abs(residual_target)
            self.drift_detector.update(abs_residual)
            if self.drift_detector.drift_detected:
                record["drift_detected"] = True
                self.drift_events.append({"cycle_idx": cycle["cycle_idx"], "abs_residual": abs_residual,
                                           "n_revealed": self.n_revealed})

        self.history.append(record)
        return record
