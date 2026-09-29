"""
Verification check B (user-requested, post-item-5c): RUL R2=-566.35 on
CALCE looks like a collapse. This script reproduces item 5c's EXACT
CALCE pipeline unchanged (same model, same tensors, same predictions)
and additionally reports what item 5c's own aggregate-only CSV never
saved: target variance/range/units, MAE, MAPE, per-battery EOL/censored
status, and a plain real-vs-artifact verdict.

R2 = 1 - SS_res/SS_tot is scale-free in the numerator but NOT in the
denominator - if a dataset's own RUL target variance (SS_tot) is small,
even a moderate absolute error produces an enormous negative R2. This
script checks exactly that hypothesis directly against CALCE's actual
label distribution, rather than assuming it.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
from stage1_common import canonical_feature_cols, build_calce_merged, load_nasa_mit_pool, PROC_DIR, ROOT
from stage7_common import load_heldout_keyed, CALCE_CELLS
from sequence_features import apply_channel_norm
from run_stage1_followup_partB_joint_rul import JointSOHRULModelFusion
from data_adapters import iterate_calce_cycles
from rul_labels import compute_eol_and_rul


def main():
    print("=== CHECK B: RUL R2=-566 on CALCE - target scale/variance, MAE/MAPE, censoring ===\n")

    canonical_rel = canonical_feature_cols(reformulated=True)
    norm_stats = json.loads((PROC_DIR / "channel_norm_stats.json").read_text())
    joint_hi_norm = json.loads((PROC_DIR / "joint_hi_norm_stats.json").read_text())
    destd = json.loads((PROC_DIR / "destandardization_constants.json").read_text())
    hi_mean = np.array(joint_hi_norm["hi_mean"])
    hi_std = np.array(joint_hi_norm["hi_std"])
    rul_mean, rul_std = destd["rul_mean"], destd["rul_std"]
    print(f"[checkB] RUL model's own training destandardization constants: rul_mean={rul_mean:.2f} cycles, "
          f"rul_std={rul_std:.2f} cycles (fit on the ORIGINAL 42-battery NASA+MIT train split)")

    joint_fusion = JointSOHRULModelFusion(n_hi_features=len(canonical_rel))
    joint_fusion.load_state_dict(torch.load(ROOT / "models" / "joint_adaptive_fusion.pt"))
    joint_fusion.eval()

    print("\n--- Per-CALCE-cell EOL/censoring diagnosis (rul_labels.compute_eol_and_rul, eol_fraction=0.8 default) ---")
    for cid in CALCE_CELLS:
        cycles = list(iterate_calce_cycles(cid))
        eol_cycle, censored, rul_map = compute_eol_and_rul(cycles)
        caps = np.array([c["discharge_capacity"] for c in cycles], dtype=float)
        initial_cap = float(np.median(caps[:3]))
        final_cap = float(caps[-1])
        n_cycles = len(cycles)
        censor_note = "(never crossed 80% threshold within recorded data)" if censored else "(crossed 80% threshold within recorded data)"
        print(f"[checkB] {cid}: n_cycles={n_cycles}, initial_cap={initial_cap:.4f}Ah, final_cap={final_cap:.4f}Ah "
              f"({100*final_cap/initial_cap:.1f}% of initial), eol_cycle={eol_cycle}, censored={censored} {censor_note}")

    _, hi_full = load_nasa_mit_pool(reformulated=True)
    calce_hi = build_calce_merged(hi_full)[["battery_id", "cycle_idx"] + canonical_rel]
    calce_hi = calce_hi.drop_duplicates(subset=["battery_id", "cycle_idx"])

    pool = load_heldout_keyed("CALCE")
    key_rows = []
    for bid, (X, soh, rul, idxs) in pool.items():
        for i, cyc in enumerate(idxs):
            key_rows.append((bid, int(cyc), bid, i))
    key_df = pd.DataFrame(key_rows, columns=["battery_id", "cycle_idx", "battery_key", "pos_idx"])
    merged_keys = key_df.merge(calce_hi, on=["battery_id", "cycle_idx"], how="inner")

    bkeys = merged_keys["battery_key"].to_numpy()
    positions = merged_keys["pos_idx"].to_numpy()
    X_arr = np.stack([pool[bk][0][p] for bk, p in zip(bkeys, positions)]).astype(np.float32)
    rul_arr = np.array([pool[bk][2][p] for bk, p in zip(bkeys, positions)], dtype=float)
    hi_arr = merged_keys[canonical_rel].to_numpy(dtype=float)
    hi_arr = np.where(np.isinf(hi_arr), np.nan, hi_arr)
    with np.errstate(invalid="ignore"):
        col_medians = np.nanmedian(hi_arr, axis=0)
    col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
    inds = np.where(np.isnan(hi_arr))
    hi_arr[inds] = np.take(col_medians, inds[1])

    X_norm = apply_channel_norm(X_arr, norm_stats)
    X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)
    hi_z = (hi_arr - hi_mean) / hi_std
    hi_z = np.nan_to_num(hi_z, nan=0.0, posinf=0.0, neginf=0.0)

    with torch.no_grad():
        _, pred_rul_z = joint_fusion(torch.tensor(X_norm), torch.tensor(hi_z, dtype=torch.float32))
    pred_rul = pred_rul_z.numpy().reshape(-1) * rul_std + rul_mean

    valid = np.isfinite(pred_rul) & np.isfinite(rul_arr)
    rul_arr, pred_rul = rul_arr[valid], pred_rul[valid]
    battery_keys_valid = bkeys[valid]

    print(f"\n--- CALCE RUL target distribution (n={len(rul_arr)} cycles, {len(set(battery_keys_valid))} batteries) ---")
    print(f"[checkB] target RUL: mean={rul_arr.mean():.2f}, std={rul_arr.std():.2f}, var={rul_arr.var():.2f}, "
          f"min={rul_arr.min():.2f}, max={rul_arr.max():.2f} cycles")
    print(f"[checkB] predicted RUL: mean={pred_rul.mean():.2f}, std={pred_rul.std():.2f}, "
          f"min={pred_rul.min():.2f}, max={pred_rul.max():.2f} cycles")

    resid = pred_rul - rul_arr
    abs_resid = np.abs(resid)
    mae = float(abs_resid.mean())
    rmse = float(np.sqrt((resid ** 2).mean()))
    y_safe = np.where(np.abs(rul_arr) < 1e-6, np.nan, rul_arr)
    mape = float(np.nanmean(abs_resid / y_safe) * 100)
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((rul_arr - rul_arr.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    print(f"\n[checkB] MAE={mae:.2f} cycles, RMSE={rmse:.2f} cycles, MAPE={mape:.2f}%")
    print(f"[checkB] SS_res={ss_res:.1f}, SS_tot={ss_tot:.1f}, ratio SS_res/SS_tot={ss_res/ss_tot:.2f} -> R2={r2:.2f}")
    print(f"[checkB] For comparison, in-domain (NASA+MIT TEST) RUL: rul_mean={rul_mean:.1f}, rul_std={rul_std:.1f} "
          f"(the SAME scale the model was trained/destandardized on)")
    print(f"[checkB] CALCE target RUL std ({rul_arr.std():.1f}) vs. in-domain train-fit RUL std ({rul_std:.1f}): "
          f"ratio={rul_arr.std()/rul_std:.3f} -> CALCE's own RUL range is "
          f"{'MUCH NARROWER' if rul_arr.std() < 0.3*rul_std else 'comparable'} than what the model was calibrated for")

    print("\n--- VERDICT ---")
    if rul_arr.std() < 0.3 * rul_std:
        print(f"[checkB] CALCE's true RUL range is narrow ({rul_arr.min():.0f}-{rul_arr.max():.0f} cycles, "
              f"std={rul_arr.std():.1f}) relative to the model's own training-scale RUL std ({rul_std:.1f}). "
              f"A SMALL absolute RUL error (MAE={mae:.1f} cycles) against this NARROW target range is enough to "
              f"produce a very large negative R2, by construction (R2's denominator SS_tot is small). This is a "
              f"REAL prediction failure (MAE/MAPE are still large in absolute/relative terms), but R2's specific "
              f"magnitude (-566) is amplified by CALCE's narrow target range - a genuine SCALE artifact of the R2 "
              f"metric itself, not evidence the model is 566x worse than it is on a normal-variance target. "
              f"MAE/MAPE are the more honest headline metrics for this specific dataset.")
    else:
        print(f"[checkB] CALCE's RUL target range/variance is NOT unusually narrow - the R2=-566 reflects a "
              f"genuinely large absolute error (MAE={mae:.1f}, MAPE={mape:.1f}%), not a metric-scale artifact.")

    pd.DataFrame([{
        "dataset": "CALCE", "n": len(rul_arr), "n_batteries": len(set(battery_keys_valid)),
        "target_mean": rul_arr.mean(), "target_std": rul_arr.std(), "target_var": rul_arr.var(),
        "target_min": rul_arr.min(), "target_max": rul_arr.max(),
        "pred_mean": pred_rul.mean(), "pred_std": pred_rul.std(),
        "mae": mae, "rmse": rmse, "mape_pct": mape, "r2": r2,
        "training_rul_mean": rul_mean, "training_rul_std": rul_std,
    }]).to_csv(ROOT / "outputs" / "finalpass_checkB_rul_diagnostic.csv", index=False)
    print("\n[checkB] saved outputs/finalpass_checkB_rul_diagnostic.csv")


if __name__ == "__main__":
    main()
