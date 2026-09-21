"""
Research pass Group E, item 14 ("largest build in this pass"):
DFC-DGGate-style domain-difference gating architecture - 3 branches
(local nonlinear, global trend, robust/Huber) fused by a domain-
difference-aware gate. Compared on in-domain TEST and all 4 held-out
datasets (CALCE, Oxford, HUST, XJTU) against the deployed XGBoost-
fusion model's own on-record numbers.

BRANCHES (disclosed, concrete choices for each named role):
  - local_nonlinear: XGBRegressor on the full canonical HI + fusion
    feature set (identical hyperparameters to the deployed model's own
    stage1_common.fit_xgb) - captures fine-grained nonlinear feature
    interactions, the same family as the currently-deployed point model.
  - global_trend: plain linear regression on [cycle_idx, cycle_idx^2]
    ONLY (deliberately low-capacity, no other features) - a smooth,
    monotonic-ish global degradation-trend model, the same spirit as
    this pass's own symbolic-regression item's "explain the smooth
    global trend" framing.
  - robust_huber: sklearn HuberRegressor (robust loss, downweights
    outliers) on the full standardized feature set - the "robust to
    noisy/shifted points" branch, expected to matter most on
    held-out domains where the feature distribution has shifted.

GATE: a small MLP mapping a 2-D "domain-difference" signal ->
softmax weights over the 3 branches:
  - feature 1: per-sample normalized squared distance from the TRAIN
    population mean (diagonal-covariance Mahalanobis approximation,
    i.e. mean of per-feature z^2) - literally "how far does this point
    look from the training domain," the concrete, disclosed stand-in
    for DFC-DGGate's own domain-difference signal.
  - feature 2: cycle_idx normalized by the TRAIN max - a cheap
    within-domain "how deep into degradation" signal, since branches'
    relative reliability plausibly shifts across a battery's own life,
    not just across datasets.

STACKING DISCIPLINE (to avoid the gate overfitting to branches' own
in-sample near-perfect fits, XGBoost especially): TRAIN batteries are
split battery-level into branch_fit_ids / gate_fit_ids (70/30). Phase 1
fits the 3 branches on branch_fit_ids ONLY, generates their predictions
on the held-out gate_fit_ids, and trains the gate on THOSE (genuinely
out-of-sample to the branches) predictions. Phase 2 then refits all 3
branches on the FULL train set (branch_fit + gate_fit batteries) for
final deployment-scale capacity, matching this project's own established
stacking convention (fit base learners to get out-of-fold predictions
for the meta-learner, then refit base learners on all available data) -
disclosed as a design choice, not hidden.

BASELINE, already on record (run_stage7_2_selfsupervised_pretrain.py's
own DEPLOYED_REFERENCE dict, itself sourced from Stage 6/6-closeout):
in-domain=0.973, CALCE=0.740, Oxford=0.953, HUST=0.800, XJTU=-1.775.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_common import (
    load_nasa_mit_pool, battery_split_masks, fusion_cols, canonical_feature_cols,
    build_calce_merged, fit_xgb, OUT_DIR, PROC_DIR,
)

SEED = 42
GATE_EPOCHS = 200
GATE_LR = 5e-3

DEPLOYED_REFERENCE = {
    "in-domain (TEST)": 0.973, "CALCE": 0.740, "Oxford": 0.953, "HUST": 0.800, "XJTU": -1.775,
}


class Gate(nn.Module):
    def __init__(self, n_in: int = 2, hidden: int = 16, n_branches: int = 3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, n_branches))

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


def domain_diff_features(X: np.ndarray, cycle_idx: np.ndarray, train_mean: np.ndarray,
                          train_std: np.ndarray, cycle_max: float) -> np.ndarray:
    z = (X - train_mean) / train_std
    mahal_approx = np.mean(z ** 2, axis=1)
    cyc_norm = np.clip(cycle_idx / max(cycle_max, 1.0), 0, 3.0)
    return np.stack([mahal_approx, cyc_norm], axis=1).astype(np.float32)


def fit_branches(Xdf: pd.DataFrame, y: np.ndarray, feature_cols: list[str], fcols: list[str]):
    cols = feature_cols + fcols
    X = Xdf[cols].to_numpy(dtype=float, copy=True)
    X = np.where(np.isinf(X), np.nan, X)
    medians = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(medians, inds[1])

    xgb_model, _, _ = fit_xgb(Xdf.assign(SOH=y), np.ones(len(Xdf), dtype=bool), feature_cols)

    cyc = Xdf["cycle_idx"].to_numpy(dtype=float)
    Xtrend = np.stack([cyc, cyc ** 2], axis=1)
    from sklearn.linear_model import LinearRegression
    trend_model = LinearRegression().fit(Xtrend, y)

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    huber_model = HuberRegressor(max_iter=500).fit(Xs, y)

    return {"xgb": xgb_model, "trend": trend_model, "huber": huber_model,
            "medians": medians, "scaler": scaler, "cols": cols}


def branch_predict(branches: dict, df: pd.DataFrame) -> np.ndarray:
    cols = branches["cols"]
    X = df[cols].to_numpy(dtype=float, copy=True)
    X = np.where(np.isinf(X), np.nan, X)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(branches["medians"], inds[1])

    pred_xgb = branches["xgb"].predict(X)
    cyc = df["cycle_idx"].to_numpy(dtype=float)
    Xtrend = np.stack([cyc, cyc ** 2], axis=1)
    pred_trend = branches["trend"].predict(Xtrend)
    Xs = branches["scaler"].transform(X)
    pred_huber = branches["huber"].predict(Xs)
    return np.stack([pred_xgb, pred_trend, pred_huber], axis=1), X  # (n, 3), raw X for domain-diff


def main():
    t0 = time.time()
    print("=== Research pass Group E, item 14: DFC-DGGate domain-difference gating ===")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    merged, hi_full = load_nasa_mit_pool(reformulated=True)
    train_mask, test_mask, split = battery_split_masks(merged)
    feature_cols = canonical_feature_cols(reformulated=True) + ["cycle_idx"]
    fcols = fusion_cols()

    train_ids = sorted(merged.loc[train_mask, "battery_id"].unique())
    perm = rng.permutation(len(train_ids))
    n_gate = max(1, int(0.3 * len(train_ids)))
    gate_fit_ids = set(np.array(train_ids)[perm[:n_gate]])
    branch_fit_ids = set(np.array(train_ids)[perm[n_gate:]])
    print(f"[dfc-dggate] TRAIN batteries: {len(train_ids)} total -> "
          f"{len(branch_fit_ids)} branch_fit / {len(gate_fit_ids)} gate_fit")

    df_branch_fit = merged[merged["battery_id"].isin(branch_fit_ids)].reset_index(drop=True)
    df_gate_fit = merged[merged["battery_id"].isin(gate_fit_ids)].reset_index(drop=True)
    df_train_full = merged[train_mask].reset_index(drop=True)
    df_test = merged[test_mask].reset_index(drop=True)

    print("[dfc-dggate] Phase 1: fitting 3 branches on branch_fit_ids only...")
    branches_p1 = fit_branches(df_branch_fit, df_branch_fit["SOH"].to_numpy(), feature_cols, fcols)

    print("[dfc-dggate] generating out-of-sample branch predictions on gate_fit_ids...")
    preds_gate_fit, X_gate_fit = branch_predict(branches_p1, df_gate_fit)
    y_gate_fit = df_gate_fit["SOH"].to_numpy()

    Xbf = df_branch_fit[branches_p1["cols"]].to_numpy(dtype=float, copy=True)
    Xbf = np.where(np.isinf(Xbf), np.nan, Xbf)
    inds_bf = np.where(np.isnan(Xbf))
    Xbf[inds_bf] = np.take(branches_p1["medians"], inds_bf[1])
    train_mean = Xbf.mean(axis=0)
    train_std = Xbf.std(axis=0) + 1e-8
    cycle_max = float(df_branch_fit["cycle_idx"].max())

    gate_feat_train = domain_diff_features(X_gate_fit, df_gate_fit["cycle_idx"].to_numpy(),
                                            train_mean, train_std, cycle_max)

    print("[dfc-dggate] training gate MLP (softmax over 3 branches, minimizing weighted MSE on gate_fit)...")
    gate = Gate()
    opt = torch.optim.Adam(gate.parameters(), lr=GATE_LR)
    preds_t = torch.tensor(preds_gate_fit, dtype=torch.float32)
    y_t = torch.tensor(y_gate_fit, dtype=torch.float32)
    feat_t = torch.tensor(gate_feat_train, dtype=torch.float32)
    for epoch in range(GATE_EPOCHS):
        opt.zero_grad()
        w = gate(feat_t)  # (n, 3)
        combined = (w * preds_t).sum(dim=1)
        loss = nn.functional.mse_loss(combined, y_t)
        loss.backward(); opt.step()
        if epoch % 40 == 0 or epoch == GATE_EPOCHS - 1:
            print(f"[dfc-dggate] gate epoch {epoch+1}/{GATE_EPOCHS}: mse={loss.item():.4f}")
    with torch.no_grad():
        w_final = gate(feat_t).mean(dim=0).numpy()
    print(f"[dfc-dggate] gate mean branch weights on gate_fit (xgb, trend, huber): {w_final}")

    print("[dfc-dggate] Phase 2: refitting all 3 branches on FULL train set for deployment-scale eval...")
    branches_final = fit_branches(df_train_full, df_train_full["SOH"].to_numpy(), feature_cols, fcols)
    Xtf = df_train_full[branches_final["cols"]].to_numpy(dtype=float, copy=True)
    Xtf = np.where(np.isinf(Xtf), np.nan, Xtf)
    inds_tf = np.where(np.isnan(Xtf))
    Xtf[inds_tf] = np.take(branches_final["medians"], inds_tf[1])
    train_mean_final = Xtf.mean(axis=0)
    train_std_final = Xtf.std(axis=0) + 1e-8
    cycle_max_final = float(df_train_full["cycle_idx"].max())

    def evaluate(name: str, df: pd.DataFrame):
        preds, X = branch_predict(branches_final, df)
        y_true = df["SOH"].to_numpy()
        feat = domain_diff_features(X, df["cycle_idx"].to_numpy(), train_mean_final, train_std_final, cycle_max_final)
        with torch.no_grad():
            w = gate(torch.tensor(feat, dtype=torch.float32)).numpy()
        combined = (w * preds).sum(axis=1)
        r2_gated = r2_score(y_true, combined)
        rmse_gated = float(np.sqrt(mean_squared_error(y_true, combined)))
        r2_xgb = r2_score(y_true, preds[:, 0])
        r2_trend = r2_score(y_true, preds[:, 1])
        r2_huber = r2_score(y_true, preds[:, 2])
        print(f"[dfc-dggate] {name}: GATED R2={r2_gated:.4f} RMSE={rmse_gated:.4f} | "
              f"branch-only R2: xgb={r2_xgb:.4f} trend={r2_trend:.4f} huber={r2_huber:.4f} | "
              f"mean gate weights={w.mean(axis=0)}")
        return {"eval_set": name, "gated_r2": r2_gated, "gated_rmse": rmse_gated,
                "branch_xgb_r2": r2_xgb, "branch_trend_r2": r2_trend, "branch_huber_r2": r2_huber,
                "mean_weight_xgb": float(w[:, 0].mean()), "mean_weight_trend": float(w[:, 1].mean()),
                "mean_weight_huber": float(w[:, 2].mean())}

    results = [evaluate("in-domain (TEST)", df_test)]
    calce_merged = build_calce_merged(hi_full)
    results.append(evaluate("CALCE", calce_merged))
    for name, fname in [("Oxford", "stage5_1_oxford_merged.parquet"),
                         ("HUST", "stage5_1_hust_merged.parquet"),
                         ("XJTU", "stage5_1_xjtu_merged.parquet")]:
        path = PROC_DIR / fname
        if not path.exists():
            print(f"[dfc-dggate] WARNING: {fname} not found, skipping {name}")
            continue
        df_held = pd.read_parquet(path)
        missing = [c for c in branches_final["cols"] if c not in df_held.columns]
        if missing:
            print(f"[dfc-dggate] WARNING: {name} missing columns {missing}, skipping")
            continue
        results.append(evaluate(name, df_held))

    results_df = pd.DataFrame(results)
    for r in results:
        r["deployed_reference_r2"] = DEPLOYED_REFERENCE.get(r["eval_set"])
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "researchpass_groupE14_dfc_dggate.csv", index=False)
    print("\n=== SUMMARY (gated R2 vs. deployed reference R2) ===")
    print(results_df[["eval_set", "gated_r2", "deployed_reference_r2"]].to_string(index=False))
    for r in results:
        ref = r["deployed_reference_r2"]
        verdict = "WIN" if (ref is not None and r["gated_r2"] > ref) else "LOSS"
        print(f"[dfc-dggate] {r['eval_set']}: gated R2={r['gated_r2']:.4f} vs deployed={ref} -> {verdict}")

    print(f"\n[dfc-dggate] TOTAL TIME: {(time.time()-t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
