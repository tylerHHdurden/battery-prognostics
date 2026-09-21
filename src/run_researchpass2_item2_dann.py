"""
Research pass 2, item 2: domain-adversarial training (DANN, Ganin et
al. 2016) - MMD and CORAL (both post-hoc distribution-alignment
methods, already on record: outputs/stage3_4_coral_results.csv) have
been tried; DANN is a genuinely different mechanism - a gradient-
reversal layer (GRL) trained END-TO-END so the SAME feature extractor
that produces the SOH prediction is also adversarially pushed to make
its embeddings domain-invariant, rather than aligning distributions as
a separate pre/post-processing step.

ARCHITECTURE (small MLP on the existing canonical HI + fusion tabular
feature pipeline - the SAME features items 8/9/14/16 used, not raw
sequences, for tractable training time):
  - feature extractor: Linear(n_feat->64) -> ReLU -> Linear(64->32) -> ReLU
  - SOH regression head: Linear(32->16) -> ReLU -> Linear(16->1),
    trained on SOURCE (TRAIN, labeled) embeddings only
  - domain classifier head: Linear(32->16) -> ReLU -> Linear(16->1)
    (logit: P(target)), trained on BOTH source and target embeddings
    passed through the GRL - standard GRL: identity in the forward
    pass, negates and scales the gradient in the backward pass, so the
    SAME backward step that trains the domain classifier to tell
    source/target apart simultaneously trains the feature extractor to
    make that classification HARDER.
  - lambda (GRL strength) ramped per Ganin et al.'s own published
    schedule: lambda_p = 2/(1+exp(-10*p)) - 1, p=training progress
    in [0,1] - not a tuned hyperparameter, the paper's own default.

ONE SEPARATE DANN MODEL PER TARGET DOMAIN (CALCE/Oxford/HUST/XJTU) -
DANN's own mechanism NEEDS unlabeled target-domain features during
training to align against (this is true of MMD/CORAL too, already on
record) - disclosed explicitly: this is NOT the same as the strict
zero-visibility zero-retrain protocol used for the deployed model
itself. Target SOH LABELS are never used during training, only X.

BASELINE COMPARISON: the deployed model's own DEPLOYED_REFERENCE R2 per
dataset (same reference used throughout both research passes), AND
(CALCE only - the only dataset MMD/CORAL were ever evaluated against on
record) direct comparison to MMD (R2=0.337) and CORAL (R2=0.529) vs.
the no-alignment baseline's own R2=0.567 - notably, both prior
alignment methods made CALCE R2 WORSE than doing nothing, a real
pattern this item's result should be read against.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_common import (
    load_nasa_mit_pool, battery_split_masks, fusion_cols, canonical_feature_cols,
    build_calce_merged, OUT_DIR, PROC_DIR,
)

SEED = 42
EPOCHS = 60
BATCH_SIZE = 64
LR = 1e-3
DEVICE = torch.device("cpu")

DEPLOYED_REFERENCE = {
    # TRUE currently-deployed xgb_soh_fusion.json numbers, per this
    # research pass's own item 5 finding, NOT the inflated
    # Stage-5-extended-reformulation figures used in earlier passes.
    "CALCE": 0.568, "Oxford": -2.694, "HUST": -0.152, "XJTU": -1.062,
}
CORAL_MMD_CALCE_REFERENCE = {"no_alignment_baseline": 0.567, "MMD": 0.337, "CORAL": 0.529}


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x, lambda_):
    return GradReverse.apply(x, lambda_)


class DANN(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(),
        )
        self.soh_head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
        self.domain_head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x, lambda_=0.0):
        z = self.extractor(x)
        soh = self.soh_head(z).squeeze(-1)
        domain_logit = self.domain_head(grad_reverse(z, lambda_)).squeeze(-1)
        return soh, domain_logit

    def predict_soh(self, x):
        z = self.extractor(x)
        return self.soh_head(z).squeeze(-1)


def impute(X, medians):
    X = np.where(np.isinf(X), np.nan, X)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(medians, inds[1])
    return X


def train_dann(X_source, y_source, X_target, seed=SEED):
    torch.manual_seed(seed)
    n_feat = X_source.shape[1]
    model = DANN(n_feat).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    Xs_t = torch.tensor(X_source, dtype=torch.float32)
    ys_t = torch.tensor(y_source, dtype=torch.float32)
    Xt_t = torch.tensor(X_target, dtype=torch.float32)
    n_s, n_t = len(Xs_t), len(Xt_t)
    rng = np.random.default_rng(seed)

    n_steps_per_epoch = max(1, n_s // BATCH_SIZE)
    total_steps = EPOCHS * n_steps_per_epoch
    step = 0
    for epoch in range(EPOCHS):
        idx_s = rng.permutation(n_s)
        soh_losses, dom_losses = [], []
        for b in range(n_steps_per_epoch):
            p = step / max(1, total_steps - 1)
            lambda_p = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

            bs = idx_s[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            bt = rng.integers(0, n_t, size=len(bs))

            soh_pred, dom_logit_s = model(Xs_t[bs], lambda_p)
            _, dom_logit_t = model(Xt_t[bt], lambda_p)

            soh_loss = nn.functional.mse_loss(soh_pred, ys_t[bs])
            dom_labels = torch.cat([torch.zeros(len(bs)), torch.ones(len(bt))])
            dom_logits = torch.cat([dom_logit_s, dom_logit_t])
            dom_loss = nn.functional.binary_cross_entropy_with_logits(dom_logits, dom_labels)

            loss = soh_loss + dom_loss
            opt.zero_grad(); loss.backward(); opt.step()
            soh_losses.append(soh_loss.item()); dom_losses.append(dom_loss.item())
            step += 1
        if epoch % 15 == 0 or epoch == EPOCHS - 1:
            with torch.no_grad():
                dom_acc = ((torch.sigmoid(dom_logits) > 0.5).float() == dom_labels).float().mean().item()
            print(f"    epoch {epoch+1}/{EPOCHS}: soh_mse={np.mean(soh_losses):.4f} "
                  f"dom_loss={np.mean(dom_losses):.4f} dom_acc={dom_acc:.3f} lambda={lambda_p:.3f}")
    return model


def main():
    t0 = time.time()
    print("=== Research pass 2, item 2: domain-adversarial training (DANN) ===")
    merged, hi_full = load_nasa_mit_pool(reformulated=True)
    train_mask, test_mask, split = battery_split_masks(merged)
    feature_cols = canonical_feature_cols(reformulated=True) + ["cycle_idx"]
    fcols = fusion_cols()
    cols = feature_cols + fcols

    X_train_full = merged.loc[train_mask, cols].to_numpy(dtype=float, copy=True)
    train_medians = np.nanmedian(np.where(np.isinf(X_train_full), np.nan, X_train_full), axis=0)
    X_source = impute(X_train_full.copy(), train_medians)
    y_source = merged.loc[train_mask, "SOH"].to_numpy(dtype=float)

    # StandardScaler (fit on SOURCE/TRAIN only) - without this, raw
    # feature scales (e.g. cycle_idx in the hundreds vs. other features
    # O(1)) caused genuine training-instability soh_mse blowups (as
    # high as 1e11 in an initial unscaled run) that swamped any real
    # signal about DANN's own merits - fixed before trusting the
    # method's own result, not reported as a "DANN doesn't work" finding.
    feat_scaler = StandardScaler().fit(X_source)
    X_source = feat_scaler.transform(X_source)

    calce_merged = build_calce_merged(hi_full)
    target_frames = {"CALCE": calce_merged}
    for name, fname in [("Oxford", "stage5_1_oxford_merged.parquet"),
                         ("HUST", "stage5_1_hust_merged.parquet"),
                         ("XJTU", "stage5_1_xjtu_merged.parquet")]:
        path = PROC_DIR / fname
        if path.exists():
            target_frames[name] = pd.read_parquet(path)
        else:
            print(f"[dann] WARNING: {fname} not found, skipping {name}")

    results = []
    for name, df in target_frames.items():
        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(f"[dann] {name}: missing columns {missing}, skipping")
            continue
        print(f"\n--- training DANN for target domain: {name} ---")
        X_target = impute(df[cols].to_numpy(dtype=float, copy=True), train_medians)
        X_target = feat_scaler.transform(X_target)
        y_target = df["SOH"].to_numpy(dtype=float)

        t1 = time.time()
        model = train_dann(X_source, y_source, X_target)
        train_time = time.time() - t1

        model.eval()
        with torch.no_grad():
            pred_target = model.predict_soh(torch.tensor(X_target, dtype=torch.float32)).numpy()
        r2 = float(r2_score(y_target, pred_target))
        rmse = float(np.sqrt(mean_squared_error(y_target, pred_target)))
        ref = DEPLOYED_REFERENCE.get(name)
        verdict = "WIN" if (ref is not None and r2 > ref) else "LOSS"
        print(f"[dann] {name}: R2={r2:.4f} RMSE={rmse:.4f} (trained in {train_time:.1f}s) | "
              f"deployed reference R2={ref} -> {verdict}")
        results.append({"target_domain": name, "dann_r2": r2, "dann_rmse": rmse,
                         "deployed_reference_r2": ref, "verdict": verdict, "train_time_s": train_time})

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "researchpass2_item2_dann.csv", index=False)
    print("\n=== SUMMARY ===")
    print(results_df.to_string(index=False))

    calce_row = results_df[results_df["target_domain"] == "CALCE"]
    if not calce_row.empty:
        dann_calce_r2 = float(calce_row["dann_r2"].iloc[0])
        print(f"\n[dann] CALCE vs. prior domain-alignment methods on record: "
              f"no-alignment baseline R2={CORAL_MMD_CALCE_REFERENCE['no_alignment_baseline']:.3f}, "
              f"MMD R2={CORAL_MMD_CALCE_REFERENCE['MMD']:.3f}, CORAL R2={CORAL_MMD_CALCE_REFERENCE['CORAL']:.3f}, "
              f"DANN (this item) R2={dann_calce_r2:.4f}")

    print(f"\n[dann] TOTAL TIME: {(time.time()-t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
