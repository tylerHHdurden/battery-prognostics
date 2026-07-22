"""
Fixed-length multi-channel per-cycle tensors, shared input representation
for all three deep models (VLSTM, CNN-LSTM, PiFormer-Transformer).

ASSUMPTION (logged clearly because it's a real modeling compromise, not
an incidental detail): each cycle is represented as a (n_bins, 6) array
with channels:
    [0] V_t   - discharge voltage, resampled to n_bins uniform TIME steps
    [1] I_t   - discharge current, resampled to n_bins uniform TIME steps
    [2] T_t   - discharge temperature, resampled to n_bins uniform TIME
                steps (zero-filled for CALCE, which has no temperature
                column — see data_adapters.py)
    [3] dQdV  - incremental capacity, resampled to n_bins uniform VOLTAGE
                steps (from ica_dv_dc)
    [4] dVdQ  - differential voltage, uniform VOLTAGE steps
    [5] dIdV  - current derivative, uniform VOLTAGE steps

Channels 0-2 are aligned by TIME progress (0->1 through discharge);
channels 3-5 are aligned by VOLTAGE progress (V_min->V_max). They are
concatenated by ARRAY POSITION only, not by shared physical meaning per
position. This is a deliberate simplification to get one fixed-length
tensor per cycle without building a full irregular-sequence / masking
pipeline — good enough for the modest CPU-scale models in this run, but
flagged so it isn't mistaken for a physically-aligned multi-sensor tensor.

VLSTM specifically consumes ONLY channel 0 (V_t) — "VLSTM" is read
literally as a voltage-sequence LSTM, per the module's own name.
CNN-LSTM and PiFormer consume all 6 channels.
"""

from __future__ import annotations

import numpy as np

from ica_dv_dc import compute_ica_dv_dc


def _resample_time(t: np.ndarray, x: np.ndarray, n_bins: int) -> np.ndarray:
    if len(t) < 2:
        return np.zeros(n_bins)
    t_norm = (t - t[0]) / (t[-1] - t[0] + 1e-12)
    grid = np.linspace(0, 1, n_bins)
    return np.interp(grid, t_norm, x)


def compute_channel_norm_stats(X: np.ndarray, clip_percentile: float = 1.0) -> list[dict]:
    """
    Per-channel robust normalization stats, fit on TRAIN data only.

    Root-cause fix for the CNN-LSTM training failure (R2=-0.071):
    dVdQ has raw values up to ~9.5 MILLION (blows up wherever dQ~0, i.e.
    flat-capacity regions) while V_t/I_t/T_t/dQdV/dIdV are all O(1-100).
    That 5-6 order-of-magnitude scale mismatch, fed raw into Conv1d +
    BatchNorm1d, produced BatchNorm running_var on the order of 1e14-1e15
    - numerically unstable, and dominated by rare extreme dVdQ spikes
    rather than representative of typical inputs. VLSTM (no BatchNorm,
    custom cell) and PiFormer (LayerNorm, normalizes per-sample not via a
    global running average) were architecturally immune to this specific
    failure mode, which is why only CNN-LSTM broke despite all three
    seeing the same unnormalized data.

    Fix: clip each channel to its [clip_percentile, 100-clip_percentile]
    range (computed from TRAIN data) before z-scoring, so a handful of
    extreme dVdQ/dIdV spikes near-zero-dV regions can't dominate the
    scale statistics the way raw mean/std would let them.
    """
    n_channels = X.shape[-1]
    stats = []
    for c in range(n_channels):
        vals = X[:, :, c].flatten()
        lo, hi = np.percentile(vals, [clip_percentile, 100 - clip_percentile])
        clipped = np.clip(vals, lo, hi)
        stats.append({
            "lo": float(lo), "hi": float(hi),
            "mean": float(clipped.mean()), "std": float(clipped.std() + 1e-8),
        })
    return stats


def apply_channel_norm(X: np.ndarray, stats: list[dict]) -> np.ndarray:
    X_norm = np.empty_like(X, dtype=np.float32)
    for c, s in enumerate(stats):
        X_norm[:, :, c] = (np.clip(X[:, :, c], s["lo"], s["hi"]) - s["mean"]) / s["std"]
    return X_norm


def get_cycle_tensor(cycle: dict, n_bins: int = 200) -> np.ndarray | None:
    dc = cycle["discharge"]
    t, V, I, T = dc["t"], dc["V"], dc["I"], dc["T"]

    ica = compute_ica_dv_dc(cycle, n_bins=n_bins)
    if ica is None:
        return None

    V_t = _resample_time(t, V, n_bins)
    I_t = _resample_time(t, I, n_bins)
    T_t = _resample_time(t, T, n_bins) if T is not None else np.zeros(n_bins)

    return np.stack([V_t, I_t, T_t, ica["dQdV"], ica["dVdQ"], ica["dIdV"]], axis=-1)


CHANNEL_NAMES = ["V_t", "I_t", "T_t", "dQdV", "dVdQ", "dIdV"]
VLSTM_CHANNEL_IDX = 0
QUERY_CHANNEL_IDX = [0, 3, 4, 5]  # V_t + ICA channels ("voltage/ICA")


def build_dataset_tensors(cycle_records: list[dict], n_bins: int = 200):
    """
    Returns (X, soh, rul, cycle_idx) for one battery, X shape
    (n_valid_cycles, n_bins, 6), aligned with rul_labels' SOH/RUL maps
    computed by the caller (kept separate here to avoid recompute).
    """
    from rul_labels import compute_eol_and_rul, soh_per_cycle

    eol_cycle, censored, rul_map = compute_eol_and_rul(cycle_records)
    soh_map = soh_per_cycle(cycle_records)

    tensors, sohs, ruls, idxs = [], [], [], []
    for c in cycle_records:
        x = get_cycle_tensor(c, n_bins=n_bins)
        if x is None:
            continue
        tensors.append(x)
        sohs.append(soh_map[c["cycle_idx"]])
        ruls.append(rul_map[c["cycle_idx"]])
        idxs.append(c["cycle_idx"])

    if not tensors:
        return None, None, None, None, censored
    return (np.stack(tensors), np.array(sohs), np.array(ruls),
            np.array(idxs), censored)
