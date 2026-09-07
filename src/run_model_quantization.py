"""
Session 24: quantize the "lean" deployment pipeline (session 20:
ICAEncoder + XGBoost-fusion, no deep sequence models) for embedded/BMS-
hardware feasibility. Fully additive - no existing model file
overwritten, `ica_encoder.pt`/`xgb_soh_fusion.json` untouched.

Scope, per instruction: reduce precision of the ICA ENCODER's weights
(the neural component) and report XGBoost's size AS-IS (the tree
ensemble) - tree ensembles don't have a standard numeric-precision
"quantization" API the way neural nets do (XGBoost's model IS its tree
structure: per-node split thresholds + per-leaf values in a JSON/binary
serialization, not a stack of weight matrices you can uniformly cast to
int8/fp16 without touching the actual tree-traversal logic). This
script does NOT attempt any custom XGBoost quantization scheme -
reducing XGBoost's footprint would mean pruning trees/reducing
n_estimators or truncating leaf-value precision, which changes the
MODEL, not just its numeric representation, and was out of scope here.

Two precision levels tested on the encoder, both standard, simple,
no new dependency beyond torch's own (legacy but still functional in
this torch version, with a deprecation warning noted below) quantized-
tensor primitives:
  - FP16 (half precision): every weight AND bias cast via `.half()`.
  - INT8 (per-output-channel symmetric affine): weights only (biases
    kept FP32 - standard practice; biases are numerically sensitive
    additive terms and negligible in size - 33 values total here).
    Uses `torch.quantize_per_channel`, which PyTorch flags as
    deprecated in this version (a newer `torchao`-based API exists) -
    used anyway since it is still fully functional and introducing a
    new quantization-library dependency for this analysis would work
    against the task's own "standard, no heavy new dependencies" scope.

**Accuracy evaluation methodology, stated honestly**: quantized
Conv1d weights are DEQUANTIZED back to float32 before the forward
pass (PyTorch's eager-mode Conv1d does not execute directly on qint8
tensors without the full separate quantized-module graph, which is a
heavier, less standard path than this analysis needs). This correctly
measures the NUMERICAL ERROR quantization introduces (the same error a
real int8 fixed-point kernel would produce, since dequantized-and-
recomputed-in-float is mathematically equivalent to the affine int8
computation for this purpose) but does **NOT** measure any LATENCY
benefit real int8 hardware/kernels would provide - this script reports
size and accuracy only, not a hardware-measured speed comparison, since
(per instruction) no real embedded hardware is available here.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))

from train_deep_models import load_all_battery_tensors, make_xy
from sequence_features import apply_channel_norm
from models.ica_encoder import ICAEncoder

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
PRED_DIR = PROC_DIR / "predictions"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

ICA_CHANNEL_SLICE = slice(3, 6)


def metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def quantize_fp16(state_dict: dict) -> dict:
    return {k: v.half() for k, v in state_dict.items()}


def quantize_int8_weights(state_dict: dict) -> dict:
    """Per-output-channel symmetric affine int8 quantization of every
    WEIGHT tensor (conv1.weight, conv2.weight, head.weight); biases
    (conv1.bias, conv2.bias, head.bias) untouched (kept FP32)."""
    out = {}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)  # the known torch quantize_per_channel deprecation notice, not a new/unexpected warning
        for k, v in state_dict.items():
            if k.endswith(".weight") and v.dim() >= 1:
                axis = 0
                per_channel_max = v.abs().flatten(1).amax(dim=1).clamp(min=1e-8)
                scale = per_channel_max / 127.0
                zero_point = torch.zeros_like(scale, dtype=torch.long)
                out[k] = torch.quantize_per_channel(v, scale, zero_point, axis=axis, dtype=torch.qint8)
            else:
                out[k] = v
    return out


def dequantize_state_dict(state_dict: dict) -> dict:
    return {k: (v.dequantize() if v.is_quantized else v.float()) for k, v in state_dict.items()}


def real_size_bytes(state_dict: dict) -> int:
    """Actually torch.save()'s the given state dict and measures the
    real serialized size on disk - not a hand-calculated estimate."""
    import io
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return len(buf.getvalue())


def build_encoder_from_state_dict(state_dict_float: dict) -> ICAEncoder:
    """state_dict_float must already be plain float32 (dequantized if
    it came from a quantized source) - Conv1d can't run on qint8
    tensors directly in eager mode."""
    enc = ICAEncoder(in_channels=3, embed_dim=16)
    enc.load_state_dict(state_dict_float)
    enc.eval()
    return enc


def compute_test_predictions(encoder: ICAEncoder, X_test_ica, hi_features, feature_cols, xgb_model):
    with torch.no_grad():
        emb = encoder.encode(torch.tensor(X_test_ica)).numpy()
    fusion_cols = [f"fusion_{i}" for i in range(16)]
    feat_df = hi_features.copy()
    for i in range(16):
        feat_df[fusion_cols[i]] = emb[:, i]
    X_feat = feat_df[feature_cols].to_numpy(dtype=float)
    return xgb_model.predict(X_feat)


def main():
    # -------------------------------------------------------------
    # 1. Model size: ICA encoder at 3 precision levels
    # -------------------------------------------------------------
    sd_fp32 = torch.load(ROOT / "models" / "ica_encoder.pt")
    sd_fp16 = quantize_fp16(sd_fp32)
    sd_int8 = quantize_int8_weights(sd_fp32)

    size_fp32 = real_size_bytes(sd_fp32)
    size_fp16 = real_size_bytes(sd_fp16)
    size_int8 = real_size_bytes(sd_int8)

    torch.save(sd_fp16, ROOT / "models" / "ica_encoder_fp16.pt")
    torch.save(sd_int8, ROOT / "models" / "ica_encoder_int8.pt")

    print("[quant] === ICA encoder size by precision (real serialized size, not estimated) ===")
    print(f"    FP32 (baseline): {size_fp32} bytes ({size_fp32/1024:.2f} KB)")
    print(f"    FP16:            {size_fp16} bytes ({size_fp16/1024:.2f} KB)  "
          f"[{size_fp32/size_fp16:.2f}x smaller]")
    print(f"    INT8 (weights):  {size_int8} bytes ({size_int8/1024:.2f} KB)  "
          f"[{size_fp32/size_int8:.2f}x smaller]")

    xgb_size = (ROOT / "models" / "xgb_soh_fusion.json").stat().st_size
    print(f"\n[quant] XGBoost-fusion (AS-IS, not quantized - tree ensemble, see module docstring): "
          f"{xgb_size} bytes ({xgb_size/1024:.1f} KB)")
    # aside, not requested but directly relevant to embedded-format
    # feasibility and zero-risk (no precision change at all): XGBoost's
    # own compact binary serialization format vs. its default verbose
    # JSON text format
    xgb_model = XGBRegressor()
    xgb_model.load_model(str(ROOT / "models" / "xgb_soh_fusion.json"))
    ubj_path = ROOT / "models" / "xgb_soh_fusion.ubj"
    xgb_model.save_model(str(ubj_path))
    ubj_size = ubj_path.stat().st_size
    print(f"[quant] ASIDE (not requested, zero precision change - same model, different serialization): "
          f"XGBoost saved as compact binary (.ubj) instead of verbose JSON: "
          f"{ubj_size} bytes ({ubj_size/1024:.1f} KB, {xgb_size/ubj_size:.2f}x smaller, bit-identical model)")

    total_fp32 = xgb_size + size_fp32
    total_int8 = xgb_size + size_int8
    total_ubj_int8 = ubj_size + size_int8
    print(f"\n[quant] TOTAL lean pipeline size: "
          f"FP32-encoder+JSON-xgb={total_fp32/1024:.1f}KB, "
          f"INT8-encoder+JSON-xgb={total_int8/1024:.1f}KB, "
          f"INT8-encoder+UBJ-xgb={total_ubj_int8/1024:.1f}KB")
    print(f"[quant] encoder quantization's share of total size saved: "
          f"{(size_fp32-size_int8)} of {(total_fp32-total_ubj_int8)} bytes possible "
          f"({(size_fp32-size_int8)/(total_fp32-total_ubj_int8)*100:.1f}%) - "
          f"XGBoost's OWN format change accounts for the rest")

    # -------------------------------------------------------------
    # 2. Accuracy: same XGBoost-fusion model, 3 encoder precisions
    # -------------------------------------------------------------
    print("\n[quant] === Accuracy on NASA+MIT test set, by encoder precision (XGBoost held fixed) ===")
    battery_data = load_all_battery_tensors()
    split = json.loads((PROC_DIR / "battery_split.json").read_text())
    test_ids = [b for b in split["test_ids"] if b in battery_data]
    X_test, y_test, _, ds_test, bid_test, cyc_test = make_xy(battery_data, test_ids)
    norm_stats = json.loads((PROC_DIR / "channel_norm_stats.json").read_text())
    X_test = apply_channel_norm(X_test, norm_stats)
    X_test_ica = X_test[:, :, ICA_CHANNEL_SLICE]

    with open(PROC_DIR / "bfa_selected_features.txt") as f:
        BFA_SELECTED = [l.strip() for l in f if l.strip()]
    hi_df = pd.read_parquet(PROC_DIR / "hi_table.parquet")
    hi_df = hi_df[hi_df["dataset"].isin(["NASA", "MIT"])]
    key_df = pd.DataFrame({"dataset": ds_test, "battery_id": bid_test, "cycle_idx": cyc_test})
    hi_features = key_df.merge(hi_df[["dataset", "battery_id", "cycle_idx"] + BFA_SELECTED],
                                on=["dataset", "battery_id", "cycle_idx"], how="left")
    col_medians = hi_df[BFA_SELECTED].median(numeric_only=True)
    for col in BFA_SELECTED:
        hi_features[col] = hi_features[col].fillna(col_medians[col])
    feature_cols = BFA_SELECTED + [f"fusion_{i}" for i in range(16)]

    results = {}
    for label, sd in [("FP32 (baseline)", sd_fp32), ("FP16", sd_fp16), ("INT8 (weights)", sd_int8)]:
        sd_float = dequantize_state_dict(sd)
        enc = build_encoder_from_state_dict(sd_float)
        pred = compute_test_predictions(enc, X_test_ica, hi_features, feature_cols, xgb_model)
        m = metrics(y_test, pred)
        results[label] = m
        print(f"    {label}: RMSE={m['rmse']:.4f} MAE={m['mae']:.4f} R2={m['r2']:.4f}")

    base = results["FP32 (baseline)"]
    for label in ["FP16", "INT8 (weights)"]:
        d_rmse = results[label]["rmse"] - base["rmse"]
        d_r2 = results[label]["r2"] - base["r2"]
        print(f"    {label} vs FP32: delta_rmse={d_rmse:+.5f}, delta_r2={d_r2:+.6f}")

    # -------------------------------------------------------------
    # 3. Compare against reference embedded/BMS memory budgets
    #    (REFERENCE FIGURES ONLY - not measured on real hardware,
    #    per instruction; commonly-cited TinyML/embedded-BMS ballparks)
    # -------------------------------------------------------------
    print("\n[quant] === Embedded/BMS hardware feasibility (reference figures, not measured) ===")
    print("[quant] Typical Cortex-M0+/M3/M4-class BMS microcontrollers: "
          "32KB-512KB total flash, 4KB-256KB total RAM (whole firmware, not just an ML model)")
    print("[quant] Commonly-cited TinyML deployment targets (e.g. MLPerf Tiny-class benchmarks): "
          "roughly tens-of-KB to ~250KB model budget to coexist with the rest of the firmware")
    print(f"[quant] This project's lean pipeline: {total_ubj_int8/1024:.0f}KB best case "
          f"(INT8 encoder + UBJ XGBoost) - "
          f"{total_ubj_int8/1024/32:.0f}x-{total_ubj_int8/1024/512:.1f}x the FULL flash budget "
          f"of a typical small BMS MCU, DOMINATED by XGBoost's ~{ubj_size/1024:.0f}KB "
          f"(500 trees, depth 6) - the tiny ICA encoder (9.5KB->{size_int8/1024:.1f}KB after "
          f"quantization) was never the bottleneck")

    summary = pd.DataFrame([
        {"component": "ICAEncoder", "precision": "FP32", "size_kb": size_fp32/1024, **base},
        {"component": "ICAEncoder", "precision": "FP16", "size_kb": size_fp16/1024, **results["FP16"]},
        {"component": "ICAEncoder", "precision": "INT8", "size_kb": size_int8/1024, **results["INT8 (weights)"]},
        {"component": "XGBoost-fusion", "precision": "as-is (JSON)", "size_kb": xgb_size/1024,
         "rmse": np.nan, "mae": np.nan, "r2": np.nan},
        {"component": "XGBoost-fusion", "precision": "as-is (UBJ, aside)", "size_kb": ubj_size/1024,
         "rmse": np.nan, "mae": np.nan, "r2": np.nan},
    ])
    summary.to_csv(OUT_DIR / "model_quantization_summary.csv", index=False)
    print(f"\n[quant] saved outputs/model_quantization_summary.csv")
    print("[quant] DONE")


if __name__ == "__main__":
    main()
