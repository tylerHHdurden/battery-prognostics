"""
Stage 4, Step 2b: retrain XGBoost-fusion (1.1 reformulated features +
1.5 monotone constraints) and the RUL joint model (JointSOHRULModelFusion,
session 41's architecture) on the new pool; GroupKFold in-domain SOH
CV (Stage 2.3 protocol); Jackknife+/CV+ SOH calibration refit; RUL
split-conformal (plain, NOT Jackknife+/CV+ - a disclosed, reasoned
deviation, see below); CALCE zero-retrain eval (plain baseline +
KMM-CP documented-but-not-adopted context).

DEVIATION FROM A LITERAL "Jackknife+/CV+ for SOH/RUL" READING, disclosed
per instruction rather than silently applied: RUL's base model is a
deep joint network, and genuine Jackknife+/CV+ would require K-fold
RETRAINING that deep model K times - the exact cost concern session
47's own RUL conformal work already investigated and scoped DOWN to
plain split-conformal for this reason (Stage 1.6 itself never attempted
Jackknife+/CV+ for RUL either, only for SOH via cheap-to-refit
estimators - Ridge/XGBoost). This follows that same, most-recent
established precedent: Jackknife+/CV+ (genuinely K-fold, cheap since
XGBoost retrains in seconds) for SOH; plain split-conformal for RUL.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor
from mapie.regression import CrossConformalRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
from stage1_common import (
    canonical_feature_cols, load_nasa_mit_pool, battery_split_masks,
    fit_xgb, eval_indomain, build_calce_merged, eval_calce, fusion_cols,
    add_reformulated_duration_features, OUT_DIR, PROC_DIR, ROOT, ALPHA,
)
from run_conformal import calib_eval_battery_split, split_conformal
from kmm_utils import fit_kmm_weights, weighted_conformal_interval, median_heuristic_gamma
from stage4_pool import load_all_battery_tensors_stage4
from sequence_features import apply_channel_norm
from run_stage1_followup_partB_joint_rul import JointSOHRULModelFusion, build_hi_features, standardize

_CKPT_JOINT = PROC_DIR / "_stage4_ckpt_joint.pt"


def backup_if_needed(rel_path):
    backup_dir = PROC_DIR / "_pre_stage4_backup"
    backup_dir.mkdir(exist_ok=True)
    src = ROOT / rel_path
    if not src.exists():
        return
    dst = backup_dir / Path(rel_path).name
    if not dst.exists():
        import shutil
        shutil.copy2(src, dst)
        print(f"[stage4-2b] backed up {rel_path} -> _pre_stage4_backup/{Path(rel_path).name}")


def main():
    t0 = time.time()
    for rel in ["models/xgb_soh_fusion.json"]:
        backup_if_needed(rel)

    # =================================================================
    # PART 1: XGBoost-fusion retrain (canonical 1.1+1.5)
    # =================================================================
    print("[stage4-2b] === retraining XGBoost-fusion (1.1+1.5 canonical) on the NEW pool ===")
    merged, hi_full = load_nasa_mit_pool(reformulated=True)
    train_mask, test_mask, split = battery_split_masks(merged)
    base_features = canonical_feature_cols(reformulated=True)
    feature_cols = base_features + ["cycle_idx"]
    fcols = fusion_cols()
    monotone = tuple([0] * len(base_features) + [-1] + [0] * len(fcols))
    print(f"[stage4-2b] pool: {len(merged)} rows, {merged['battery_id'].nunique()} batteries "
          f"({train_mask.sum()} train rows, {test_mask.sum()} test rows)")

    model, medians, cols = fit_xgb(merged, train_mask, feature_cols, xgb_extra_kwargs={"monotone_constraints": monotone})
    indomain = eval_indomain(model, medians, cols, merged, test_mask)
    print(f"[stage4-2b] in-domain (fixed test-battery split) SOH R2={indomain['r2']:.4f} RMSE={indomain['rmse']:.4f}")
    model.save_model(str(ROOT / "models" / "xgb_soh_fusion.json"))
    print("[stage4-2b] saved models/xgb_soh_fusion.json (OVERWRITES the prior 32-battery/stale-feature-set model)")

    # save per-cycle test predictions (needed by Step 3's second-life
    # grading re-run - same xgb_fusion_preds.csv shape/columns session
    # 25's script already expects, so it's reused unchanged)
    X_train_for_pred = merged.loc[train_mask, cols].to_numpy(dtype=float, copy=True)
    inds = np.where(np.isnan(X_train_for_pred))
    X_train_for_pred[inds] = np.take(medians, inds[1])
    pred_train = model.predict(X_train_for_pred)
    out_preds = merged[["dataset", "battery_id", "cycle_idx", "SOH"]].copy()
    out_preds["y_pred_soh_fusion"] = np.nan
    out_preds.loc[train_mask, "y_pred_soh_fusion"] = pred_train
    out_preds.loc[test_mask, "y_pred_soh_fusion"] = indomain["pred"]
    out_preds["split"] = np.where(train_mask, "train", "test")
    out_preds.to_csv(PROC_DIR / "predictions" / "xgb_fusion_preds_stage4.csv", index=False)
    print("[stage4-2b] saved data/processed/predictions/xgb_fusion_preds_stage4.csv "
          "(per-cycle train+test predictions, for Step 3's grading re-run)")

    # =================================================================
    # PART 2: GroupKFold in-domain CV (Stage 2.3 protocol)
    # =================================================================
    print("\n[stage4-2b] === GroupKFold(5) in-domain SOH CV (Stage 2.3 protocol) ===")
    X_all = merged[cols].to_numpy(dtype=float, copy=True)
    y_all = merged["SOH"].to_numpy(dtype=float)
    groups = merged["battery_id"].to_numpy()
    gkf = GroupKFold(n_splits=5)
    fold_results = []
    for fold_idx, (tr_idx, te_idx) in enumerate(gkf.split(X_all, y_all, groups=groups)):
        Xtr, Xte = X_all[tr_idx].copy(), X_all[te_idx].copy()
        ytr, yte = y_all[tr_idx], y_all[te_idx]
        col_med = np.nanmedian(Xtr, axis=0)
        for arr in (Xtr, Xte):
            inds = np.where(np.isnan(arr))
            arr[inds] = np.take(col_med, inds[1])
        m = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03, subsample=0.8,
                          colsample_bytree=0.8, random_state=42, n_jobs=-1, reg_lambda=1.0,
                          monotone_constraints=monotone)
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)
        r2 = float(r2_score(yte, pred))
        rmse = float(np.sqrt(mean_squared_error(yte, pred)))
        n_test_b = len(set(groups[te_idx]))
        print(f"[stage4-2b] fold {fold_idx}: {n_test_b} test batteries, {len(te_idx)} rows, "
              f"R2={r2:.4f} RMSE={rmse:.4f}")
        fold_results.append({"fold": fold_idx, "r2": r2, "rmse": rmse, "n_test_batteries": n_test_b})
    fold_df = pd.DataFrame(fold_results)
    print(f"[stage4-2b] GroupKFold mean R2={fold_df.r2.mean():.4f} (std={fold_df.r2.std():.4f}), "
          f"mean RMSE={fold_df.rmse.mean():.4f} (std={fold_df.rmse.std():.4f})")

    # =================================================================
    # PART 3: Jackknife+/CV+ SOH calibration (genuine K-fold, cheap for XGBoost)
    # =================================================================
    print("\n[stage4-2b] === Jackknife+/CV+ SOH calibration (Stage 1.6's method) ===")
    calib_ids, eval_ids = calib_eval_battery_split(sorted(set(indomain["battery_id"].tolist())))
    print(f"[stage4-2b] calibration batteries: {calib_ids}")
    calib_mask_full = merged["battery_id"].isin(calib_ids).to_numpy() & test_mask
    X_calib_all = merged[cols].to_numpy(dtype=float, copy=True)
    inds = np.where(np.isnan(X_calib_all))
    X_calib_all[inds] = np.take(medians, inds[1])
    X_calib_cv = X_calib_all[calib_mask_full]
    y_calib_cv = merged.loc[calib_mask_full, "SOH"].to_numpy()
    groups_cv = merged.loc[calib_mask_full, "battery_id"].to_numpy()
    n_calib_batteries = len(set(groups_cv))

    eval_mask_full = merged["battery_id"].isin(eval_ids).to_numpy() & test_mask
    X_eval_cv = X_calib_all[eval_mask_full]
    y_eval_cv = merged.loc[eval_mask_full, "SOH"].to_numpy()

    base_est = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03, subsample=0.8,
                             colsample_bytree=0.8, random_state=42, n_jobs=-1, reg_lambda=1.0,
                             monotone_constraints=monotone)
    cv_loo = GroupKFold(n_splits=n_calib_batteries)
    mapie_jk = CrossConformalRegressor(estimator=base_est, confidence_level=1 - ALPHA, method="plus", cv=cv_loo)
    mapie_jk.fit_conformalize(X_calib_cv, y_calib_cv, groups=groups_cv)
    pred_jk, interval_jk = mapie_jk.predict_interval(X_eval_cv)
    lo_jk, hi_jk = interval_jk[:, 0, 0], interval_jk[:, 1, 0]
    covered_jk = (y_eval_cv >= lo_jk) & (y_eval_cv <= hi_jk)
    soh_jk_coverage = float(covered_jk.mean())
    soh_jk_width = float(np.mean(hi_jk - lo_jk))
    print(f"[stage4-2b] SOH Jackknife+/CV+ (in-domain eval battery split): coverage={soh_jk_coverage:.4f} "
          f"avg_width={soh_jk_width:.4f}")

    # =================================================================
    # PART 4: RUL joint model retrain (JointSOHRULModelFusion, session 41 arch)
    # =================================================================
    print("\n[stage4-2b] === retraining RUL joint model (JointSOHRULModelFusion) on the NEW pool ===")
    battery_data = load_all_battery_tensors_stage4()
    split_json = json.loads((PROC_DIR / "battery_split.json").read_text())
    train_ids = [b for b in split_json["train_ids"] if b in battery_data]
    test_ids = [b for b in split_json["test_ids"] if b in battery_data]
    n_val = max(1, len(train_ids) // 5)
    val_ids = sorted(train_ids)[-n_val:]
    fit_ids = [b for b in train_ids if b not in val_ids]

    from train_deep_models import make_xy
    X_fit, soh_fit, rul_fit, _, bid_fit, cyc_fit = make_xy(battery_data, fit_ids)
    X_val, soh_val, rul_val, _, bid_val, cyc_val = make_xy(battery_data, val_ids)
    X_test, soh_test, rul_test, _, bid_test, cyc_test = make_xy(battery_data, test_ids)
    print(f"[stage4-2b] joint model: fit={len(X_fit)} val={len(X_val)} test={len(X_test)} cycles")

    norm_stats = json.loads((PROC_DIR / "channel_norm_stats.json").read_text())
    X_fit = apply_channel_norm(X_fit, norm_stats)
    X_val = apply_channel_norm(X_val, norm_stats)
    X_test = apply_channel_norm(X_test, norm_stats)

    hi_reformulated = add_reformulated_duration_features(pd.read_parquet(PROC_DIR / "hi_table.parquet"))
    joint_feature_cols = canonical_feature_cols(reformulated=True)
    fit_key_df = pd.DataFrame({"battery_id": bid_fit, "cycle_idx": cyc_fit}).merge(
        hi_reformulated[["battery_id", "cycle_idx"] + joint_feature_cols], on=["battery_id", "cycle_idx"], how="left")
    train_medians_hi = fit_key_df[joint_feature_cols].median(numeric_only=True).to_numpy()

    HI_fit = build_hi_features(bid_fit, cyc_fit, hi_reformulated, joint_feature_cols, train_medians_hi)
    HI_val = build_hi_features(bid_val, cyc_val, hi_reformulated, joint_feature_cols, train_medians_hi)
    HI_test = build_hi_features(bid_test, cyc_test, hi_reformulated, joint_feature_cols, train_medians_hi)
    hi_mean, hi_std = HI_fit.mean(axis=0), HI_fit.std(axis=0) + 1e-8
    HI_fit = (HI_fit - hi_mean) / hi_std
    HI_val = (HI_val - hi_mean) / hi_std
    HI_test = (HI_test - hi_mean) / hi_std

    # persisted so live_inference.py can apply the IDENTICAL z-score to a
    # live single-cycle HI vector at inference time - completeness fix
    # caught by this project's own end-to-end smoke test (see
    # run_stage4_smoke_test.py / run_stage4_save_joint_hi_norm_stats.py's
    # docstrings): the first version of this script computed these stats
    # in-memory only and never saved them, so live inference had no way
    # to reproduce the exact normalization the model was trained under.
    with open(PROC_DIR / "joint_hi_norm_stats.json", "w") as f:
        json.dump({"feature_order": joint_feature_cols, "hi_mean": hi_mean.tolist(),
                    "hi_std": hi_std.tolist(), "train_medians": train_medians_hi.tolist()}, f, indent=2)
    print("[stage4-2b] saved data/processed/joint_hi_norm_stats.json (RUL joint model's own HI z-score stats)")

    from models.joint_model import AdaptiveLossWeighting
    joint_model = JointSOHRULModelFusion(n_hi_features=len(joint_feature_cols))
    soh_fit_z, soh_mean, soh_std = standardize(soh_fit)
    rul_fit_z, rul_mean, rul_std = standardize(rul_fit)
    soh_val_z = (soh_val - soh_mean) / soh_std
    rul_val_z = (rul_val - rul_mean) / rul_std

    adaptive = AdaptiveLossWeighting()
    params = list(joint_model.parameters()) + list(adaptive.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    mse = nn.MSELoss()
    Xt, HIt = torch.tensor(X_fit), torch.tensor(HI_fit)
    soh_t = torch.tensor(soh_fit_z).unsqueeze(-1)
    rul_t = torch.tensor(rul_fit_z).unsqueeze(-1)
    Xv, HIv = torch.tensor(X_val), torch.tensor(HI_val)
    soh_v = torch.tensor(soh_val_z).unsqueeze(-1)
    rul_v = torch.tensor(rul_val_z).unsqueeze(-1)

    start_epoch, history = 0, []
    EPOCHS, BATCH_SIZE = 25, 64
    if _CKPT_JOINT.exists():
        ckpt = torch.load(_CKPT_JOINT, weights_only=False)
        joint_model.load_state_dict(ckpt["model_state"])
        adaptive.load_state_dict(ckpt["adaptive_state"])
        opt.load_state_dict(ckpt["opt_state"])
        torch.set_rng_state(ckpt["torch_rng_state"])
        start_epoch = ckpt["epoch"] + 1
        history = ckpt["history"]
        print(f"[stage4-2b/joint] RESUMING from epoch {start_epoch}")

    n = len(Xt)
    for epoch in range(start_epoch, EPOCHS):
        joint_model.train()
        perm = torch.randperm(n)
        ep_loss = 0.0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            pred_soh, pred_rul = joint_model(Xt[idx], HIt[idx])
            l_soh = mse(pred_soh, soh_t[idx])
            l_rul = mse(pred_rul, rul_t[idx])
            total, a, b = adaptive(l_soh, l_rul)
            total.backward()
            opt.step()
            with torch.no_grad():
                adaptive.log_sigma_soh.clamp_(-adaptive.LOG_SIGMA_CLAMP, adaptive.LOG_SIGMA_CLAMP)
                adaptive.log_sigma_rul.clamp_(-adaptive.LOG_SIGMA_CLAMP, adaptive.LOG_SIGMA_CLAMP)
            ep_loss += total.item() * len(idx)
        ep_loss /= n

        joint_model.eval()
        with torch.no_grad():
            vp_soh, vp_rul = joint_model(Xv, HIv)
            v_l_soh = mse(vp_soh, soh_v).item()
            v_l_rul = mse(vp_rul, rul_v).item()
        history.append({"epoch": epoch, "train_loss": ep_loss, "val_loss_soh": v_l_soh,
                         "val_loss_rul": v_l_rul, "alpha": a, "beta": b})
        print(f"[stage4-2b/joint] epoch {epoch:2d} train={ep_loss:.4f} val_soh={v_l_soh:.4f} "
              f"val_rul={v_l_rul:.4f} alpha={a:.3f} beta={b:.3f}")

        torch.save({"model_state": joint_model.state_dict(), "adaptive_state": adaptive.state_dict(),
                    "opt_state": opt.state_dict(), "torch_rng_state": torch.get_rng_state(),
                    "epoch": epoch, "history": history}, _CKPT_JOINT)

    _CKPT_JOINT.unlink(missing_ok=True)

    joint_model.eval()
    with torch.no_grad():
        pred_soh_z, pred_rul_z = joint_model(torch.tensor(X_test), torch.tensor(HI_test))
    pred_soh_joint = pred_soh_z.squeeze(-1).numpy() * soh_std + soh_mean
    pred_rul_joint = pred_rul_z.squeeze(-1).numpy() * rul_std + rul_mean

    def m(y_true, y_pred):
        return {"rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "mae": float(mean_absolute_error(y_true, y_pred)), "r2": float(r2_score(y_true, y_pred))}
    soh_metrics_joint = m(soh_test, pred_soh_joint)
    rul_metrics_joint = m(rul_test, pred_rul_joint)
    print(f"\n[stage4-2b] joint model TEST SOH: {soh_metrics_joint}")
    print(f"[stage4-2b] joint model TEST RUL: {rul_metrics_joint}")

    torch.save(joint_model.state_dict(), ROOT / "models" / "joint_adaptive_fusion.pt")
    print("[stage4-2b] saved models/joint_adaptive_fusion.pt (NEW file - deployed app will be "
          "repointed to this in Step 4, replacing the stale joint_adaptive.pt)")

    # RUL split-conformal (plain - see module docstring for why not Jackknife+/CV+)
    print("\n[stage4-2b] === RUL split-conformal calibration ===")
    calib_ids_rul, eval_ids_rul = calib_eval_battery_split(sorted(set(bid_test)))
    bid_test_arr = np.array(bid_test)
    calib_mask_rul = np.isin(bid_test_arr, calib_ids_rul)
    eval_mask_rul = np.isin(bid_test_arr, eval_ids_rul)
    _, lo_rul, hi_rul, _ = split_conformal(pred_rul_joint[calib_mask_rul], rul_test[calib_mask_rul],
                                            pred_rul_joint[eval_mask_rul], ALPHA)
    rul_coverage = float(((rul_test[eval_mask_rul] >= lo_rul) & (rul_test[eval_mask_rul] <= hi_rul)).mean())
    rul_width = float(np.mean(hi_rul - lo_rul))
    print(f"[stage4-2b] RUL split-conformal coverage={rul_coverage:.4f} avg_width={rul_width:.4f}")

    # =================================================================
    # PART 4b: compressed model (session 35 Part 5's n=100/depth=3 config),
    # retrained on the NEW pool/canonical features - embedded/BMS-feasible
    # variant deployed ALONGSIDE the full model, not replacing it
    # =================================================================
    print("\n[stage4-2b] === compressed model (n=100, depth=3 - session 35 Part 5's config) ===")
    X_train_full = merged.loc[train_mask, cols].to_numpy(dtype=float, copy=True)
    inds = np.where(np.isnan(X_train_full))
    X_train_full[inds] = np.take(medians, inds[1])
    y_train_full = merged.loc[train_mask, "SOH"].to_numpy(dtype=float)
    compressed = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.03, subsample=0.8,
                               colsample_bytree=0.8, random_state=42, n_jobs=-1, reg_lambda=1.0,
                               monotone_constraints=monotone)
    compressed.fit(X_train_full, y_train_full)
    X_test_full = merged.loc[test_mask, cols].to_numpy(dtype=float, copy=True)
    inds = np.where(np.isnan(X_test_full))
    X_test_full[inds] = np.take(medians, inds[1])
    pred_compressed = compressed.predict(X_test_full)
    y_test_true = merged.loc[test_mask, "SOH"].to_numpy(dtype=float)
    compressed_r2 = float(r2_score(y_test_true, pred_compressed))
    compressed_rmse = float(np.sqrt(mean_squared_error(y_test_true, pred_compressed)))
    compressed.save_model(str(ROOT / "models" / "xgb_soh_fusion_compressed.json"))
    compressed_path_ubj = ROOT / "models" / "xgb_soh_fusion_compressed.ubj"
    compressed.save_model(str(compressed_path_ubj))
    ubj_size_kb = compressed_path_ubj.stat().st_size / 1024
    print(f"[stage4-2b] compressed model: R2={compressed_r2:.4f} RMSE={compressed_rmse:.4f} "
          f"UBJ size={ubj_size_kb:.1f}KB (session 35 Part 5's own number for this config: "
          f"114.9KB, R2=0.9119 - on the OLD 32-battery/stale-feature-set model)")
    print(f"[stage4-2b] saved models/xgb_soh_fusion_compressed.{{json,ubj}} "
          f"(NEW files - the embedded/BMS-feasible variant, deployed ALONGSIDE the full model)")

    # =================================================================
    # PART 5: CALCE zero-retrain eval - plain baseline (primary) + KMM-CP (context)
    # =================================================================
    print("\n[stage4-2b] === CALCE zero-retrain evaluation ===")
    calce_merged = build_calce_merged(hi_full)
    calce = eval_calce(model, medians, feature_cols, calce_merged)
    print(f"[stage4-2b] CALCE point accuracy: R2={calce['r2']:.4f} RMSE={calce['rmse']:.4f}")

    calib_ids_c, eval_ids_c = calib_eval_battery_split(sorted(set(indomain["battery_id"].tolist())))
    calib_mask_c = merged["battery_id"].isin(calib_ids_c).to_numpy() & test_mask
    calib_pred_c = indomain["pred"][np.isin(indomain["battery_id"], calib_ids_c)]
    calib_y_c = indomain["y_true"][np.isin(indomain["battery_id"], calib_ids_c)]
    _, lo_plain, hi_plain, _ = split_conformal(calib_pred_c, calib_y_c, calce["pred"], ALPHA)
    cov_plain = float(((calce["y_true"] >= lo_plain) & (calce["y_true"] <= hi_plain)).mean())
    width_plain = float(np.mean(hi_plain - lo_plain))
    print(f"[stage4-2b] CALCE PLAIN split-conformal (PRIMARY, deployed number): "
          f"coverage={cov_plain:.4f} avg_width={width_plain:.4f}")

    print(f"[stage4-2b] KMM-CP context (documented-but-NOT-adopted, Stage 3.1 result on the "
          f"PRIOR 32-battery model): coverage=34.04% avg_width=9.924 - NOT recomputed against "
          f"this new model/pool in this pass; reported here only as documented context per "
          f"instruction, not as this deployment's actual calibration.")

    # =================================================================
    # Save everything
    # =================================================================
    fold_df.to_csv(OUT_DIR / "stage4_groupkfold_cv.csv", index=False)
    pd.DataFrame([
        {"metric": "soh_indomain_r2", "value": indomain["r2"]},
        {"metric": "soh_indomain_rmse", "value": indomain["rmse"]},
        {"metric": "soh_groupkfold_mean_r2", "value": fold_df.r2.mean()},
        {"metric": "soh_groupkfold_mean_rmse", "value": fold_df.rmse.mean()},
        {"metric": "soh_jackknife_coverage", "value": soh_jk_coverage},
        {"metric": "soh_jackknife_width", "value": soh_jk_width},
        {"metric": "rul_joint_r2", "value": rul_metrics_joint["r2"]},
        {"metric": "rul_joint_rmse", "value": rul_metrics_joint["rmse"]},
        {"metric": "rul_split_conformal_coverage", "value": rul_coverage},
        {"metric": "rul_split_conformal_width", "value": rul_width},
        {"metric": "calce_r2", "value": calce["r2"]},
        {"metric": "calce_rmse", "value": calce["rmse"]},
        {"metric": "calce_plain_coverage", "value": cov_plain},
        {"metric": "calce_plain_width", "value": width_plain},
        {"metric": "joint_soh_test_r2", "value": soh_metrics_joint["r2"]},
        {"metric": "joint_soh_test_rmse", "value": soh_metrics_joint["rmse"]},
    ]).to_csv(OUT_DIR / "stage4_step2b_summary.csv", index=False)

    print(f"\n[stage4-2b] saved outputs/stage4_groupkfold_cv.csv, outputs/stage4_step2b_summary.csv")
    print(f"[stage4-2b] ALL DONE in {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
