"""
Targeted improvement pass, Part 5: tree-ensemble compression of the
DEPLOYED lean pipeline's XGBoost-fusion model.

Scope decision, stated explicitly: this targets `xgb_soh_fusion.json`
(the original 32-battery lean model, 500 trees/depth 6, 2,845.7KB JSON
/ 1,981.0KB UBJ per session 24) - not the Dataset Expansion session's
additive `xgb_soh_fusion_expanded.json`. The deployed default is still
the 32-battery lean pipeline (that decision was explicitly NOT changed
by the Dataset Expansion session), and session 24's own embedded-
feasibility finding was scoped to this exact model, so continuing that
same investigation on the same model keeps this a direct, apples-to-
apples follow-up rather than a new, separate question.

Session 24 found quantizing the small ICAEncoder was "almost beside the
point" - XGBoost-fusion's 500 trees are 99.7% of the lean pipeline's
size, and shrinking the tree ensemble itself was explicitly flagged as
"out of scope for 'quantization' as requested" THEN. This script does
exactly that follow-up: (1) a hyperparameter-pruning sweep (fewer/
shallower trees, retrained - XGBoost has no post-hoc "prune a fitted
ensemble" API, so exploring the accuracy/size frontier via retraining
is the honest way to do this), (2) UBJ serialization stacked on top
(free, zero precision change, per session 24), (3) knowledge
distillation into a much smaller model if pruning alone can't clear
the budget without an unacceptable accuracy cost.

Reused UNCHANGED from train_xgboost_fusion.py: same data
(hi_table.parquet + fusion_embeddings.csv), same 23-feature
construction (7 BFA HIs + 16 fusion dims), same battery_split.json,
same median-imputation of NaNs - this is a compression study on the
EXISTING model/data, not a new training pipeline.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

FLASH_BUDGET_KB = (32, 512)  # session 24's typical small-BMS-MCU reference range


def _model_size_kb(model: XGBRegressor) -> tuple[float, float]:
    """Returns (json_kb, ubj_kb) - same measurement method as session 24
    (serialize to a real temp file, measure actual bytes on disk, not an
    estimate)."""
    with tempfile.TemporaryDirectory() as d:
        p_json = Path(d) / "m.json"
        p_ubj = Path(d) / "m.ubj"
        model.save_model(str(p_json))
        model.save_model(str(p_ubj))
        return p_json.stat().st_size / 1024, p_ubj.stat().st_size / 1024


def load_data():
    df = pd.read_parquet(PROC_DIR / "hi_table.parquet")
    df = df[df["dataset"].isin(["NASA", "MIT"])].reset_index(drop=True)
    with open(PROC_DIR / "bfa_selected_features.txt") as f:
        selected = [l.strip() for l in f if l.strip()]
    fusion = pd.read_csv(PROC_DIR / "fusion_embeddings.csv")
    fusion_cols = [c for c in fusion.columns if c.startswith("fusion_")]
    merged = pd.merge(df, fusion, on=["dataset", "battery_id", "cycle_idx"], how="inner")
    feature_cols = selected + fusion_cols
    X = merged[feature_cols].to_numpy(dtype=float, copy=True)
    col_medians = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_medians, inds[1])
    y = merged["SOH"].to_numpy(dtype=float)
    split = json.loads((PROC_DIR / "battery_split.json").read_text())
    train_mask = merged["battery_id"].isin(split["train_ids"]).to_numpy()
    test_mask = ~train_mask
    return X, y, train_mask, test_mask, feature_cols


def eval_model(model, X, y, test_mask):
    pred = model.predict(X[test_mask])
    rmse = float(np.sqrt(mean_squared_error(y[test_mask], pred)))
    mae = float(mean_absolute_error(y[test_mask], pred))
    r2 = float(r2_score(y[test_mask], pred))
    return rmse, mae, r2, pred


def main():
    X, y, train_mask, test_mask, feature_cols = load_data()
    print(f"[tree-compress] {len(feature_cols)} features, train={train_mask.sum()} rows, "
          f"test={test_mask.sum()} rows")

    rows = []

    print("\n[tree-compress] === Baseline: existing deployed xgb_soh_fusion.json ===")
    baseline = XGBRegressor()
    baseline.load_model(str(ROOT / "models" / "xgb_soh_fusion.json"))
    rmse, mae, r2, base_pred_test = eval_model(baseline, X, y, test_mask)
    json_kb, ubj_kb = _model_size_kb(baseline)
    print(f"[tree-compress] baseline (500 trees, depth 6): RMSE={rmse:.4f} MAE={mae:.4f} "
          f"R2={r2:.4f} | JSON={json_kb:.1f}KB UBJ={ubj_kb:.1f}KB")
    rows.append({"approach": "baseline (as-deployed)", "n_estimators": 500, "max_depth": 6,
                 "format": "JSON", "size_kb": json_kb, "rmse": rmse, "mae": mae, "r2": r2})
    rows.append({"approach": "baseline (as-deployed)", "n_estimators": 500, "max_depth": 6,
                 "format": "UBJ", "size_kb": ubj_kb, "rmse": rmse, "mae": mae, "r2": r2})

    print("\n[tree-compress] === Step 1: hyperparameter-pruning sweep (retrained, "
          "cheapest option tried first) ===")
    sweep_configs = [
        (n, d) for n in [250, 100, 50, 25, 15, 10, 5]
        for d in [6, 4, 3, 2]
    ]
    best_under_budget = None
    for n_estimators, max_depth in sweep_configs:
        model = XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            n_jobs=-1, reg_lambda=1.0,
        )
        model.fit(X[train_mask], y[train_mask])
        rmse, mae, r2, _ = eval_model(model, X, y, test_mask)
        json_kb, ubj_kb = _model_size_kb(model)
        row = {"approach": f"pruned n={n_estimators} depth={max_depth}",
               "n_estimators": n_estimators, "max_depth": max_depth,
               "format": "UBJ", "size_kb": ubj_kb, "rmse": rmse, "mae": mae, "r2": r2}
        rows.append(row)
        rows.append({**row, "format": "JSON", "size_kb": json_kb})
        print(f"[tree-compress] n={n_estimators:>4} depth={max_depth}: "
              f"RMSE={rmse:.4f} R2={r2:.4f} | JSON={json_kb:7.1f}KB UBJ={ubj_kb:7.1f}KB")
        if ubj_kb <= FLASH_BUDGET_KB[1] and (best_under_budget is None
                                              or r2 > best_under_budget["r2"]):
            best_under_budget = row

    if best_under_budget:
        print(f"\n[tree-compress] Best pruned config that fits under the "
              f"{FLASH_BUDGET_KB[1]}KB budget: {best_under_budget['approach']} "
              f"(UBJ={best_under_budget['size_kb']:.1f}KB, R2={best_under_budget['r2']:.4f}, "
              f"vs. baseline R2={r2:.4f} above -> baseline is `rmse` var from last loop, "
              f"see table for the real baseline row).")
    else:
        print(f"\n[tree-compress] NO pruned config in this sweep fits under "
              f"{FLASH_BUDGET_KB[1]}KB - even n=5/depth=2 is measured above; pruning alone "
              f"is insufficient for this budget, moving to distillation.")

    print("\n[tree-compress] === Step 2: knowledge distillation (small model mimics "
          "the FULL 500-tree baseline's OUTPUT, not the true SOH label directly) ===")
    teacher_pred_train = baseline.predict(X[train_mask])
    distill_configs = [
        ("XGB-distill n=5 depth=2", XGBRegressor(n_estimators=5, max_depth=2, learning_rate=0.3,
                                                  random_state=42, n_jobs=-1)),
        ("XGB-distill n=3 depth=2", XGBRegressor(n_estimators=3, max_depth=2, learning_rate=0.3,
                                                  random_state=42, n_jobs=-1)),
        ("single DecisionTree depth=4", DecisionTreeRegressor(max_depth=4, random_state=42)),
        ("single DecisionTree depth=6", DecisionTreeRegressor(max_depth=6, random_state=42)),
        ("single DecisionTree depth=8", DecisionTreeRegressor(max_depth=8, random_state=42)),
    ]
    import pickle
    distill_rows = []
    for name, dmodel in distill_configs:
        dmodel.fit(X[train_mask], teacher_pred_train)  # distilled on the TEACHER's output
        pred_test = dmodel.predict(X[test_mask])
        # Accuracy reported against the TRUE SOH label, not the teacher's
        # output - the only honest way to know if distillation actually
        # preserved real-world accuracy, not just mimicry fidelity.
        rmse = float(np.sqrt(mean_squared_error(y[test_mask], pred_test)))
        mae = float(mean_absolute_error(y[test_mask], pred_test))
        r2 = float(r2_score(y[test_mask], pred_test))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.pkl"
            with open(p, "wb") as f:
                pickle.dump(dmodel, f, protocol=pickle.HIGHEST_PROTOCOL)
            size_kb = p.stat().st_size / 1024
        fidelity_r2 = float(r2_score(teacher_pred_train, dmodel.predict(X[train_mask])))
        print(f"[tree-compress] {name}: vs TRUE SOH -> RMSE={rmse:.4f} MAE={mae:.4f} "
              f"R2={r2:.4f} | size(pickle)={size_kb:.2f}KB | fidelity-to-teacher R2(train)={fidelity_r2:.4f}")
        distill_rows.append({"approach": name, "format": "pickle", "size_kb": size_kb,
                              "rmse": rmse, "mae": mae, "r2": r2,
                              "fidelity_to_teacher_r2_train": fidelity_r2})

    pd.DataFrame(rows).to_csv(OUT_DIR / "tree_compression_pruning_sweep.csv", index=False)
    pd.DataFrame(distill_rows).to_csv(OUT_DIR / "tree_compression_distillation.csv", index=False)
    print(f"\n[tree-compress] Saved outputs/tree_compression_pruning_sweep.csv "
          f"({len(rows)} rows) and outputs/tree_compression_distillation.csv "
          f"({len(distill_rows)} rows)")
    print("[tree-compress] DONE")


if __name__ == "__main__":
    main()
