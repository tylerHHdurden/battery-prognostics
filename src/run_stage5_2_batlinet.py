"""
Stage 5.2: BatLiNet (Zhang et al. 2025, Nature Machine Intelligence) -
train the intra/inter-cell dual-branch model (src/models/batlinet.py,
see its docstring for the 2 disclosed adaptations from the paper) on
the full available pool EXCLUDING CALCE: original 32 + 10 recovered +
Stage 5.1's Oxford/HUST/XJTU. MIT Fast-Charging (Attia et al. 2020) is
NOT separately added here - per Stage 5.1's own finding, it is already
the existing b4c* batteries inside the "original 32" MIT pool, not a
new addition (see DEVELOPMENT_LOG.md Stage 5.1 entry).

TRAIN-POOL SCOPE DECISION, disclosed: HUST alone contributes 146,122
cycles vs. 29,705 for the entire existing NASA+MIT+recovered pool and
519/~20-30k for Oxford/XJTU - training on every raw cycle would let
HUST's sheer row-count dominate the learned representation by simple
cycle-count imbalance, not because it's more informative. Each
battery's cycles are also highly autocorrelated (adjacent cycles are
near-duplicates), so this is mostly redundant signal, not more real
information. FIX: per-battery cycle subsampling cap for TRAINING ONLY
(evenly-spaced, MAX_CYCLES_PER_BATTERY) - evaluation (GroupKFold CV,
CALCE zero-retrain) still uses every real cycle, unaffected.

TRAIN/EVAL PROTOCOL, disclosed (the task's own item 2 vs. item 3 read
as in tension - item 2 says train on Oxford/HUST/XJTU, item 3 says
"zero-retrain on ... 5.1's new held-out datasets"; both cannot be
literally true for the same datasets at once). Resolved as: item 2 is
followed literally (Oxford/HUST/XJTU ARE in the training pool, CALCE
is the only held-out set - "held out exactly as always" is CALCE's own
long-established, unambiguous role in this project). In-domain
performance is measured via GroupKFold(5) (Stage 2.3's protocol,
directly comparable to Stage 4's 0.9658), broken down BOTH pooled and
per-dataset-family so Oxford/HUST/XJTU's in-domain fit (now that
they're real training data) is visible separately from the
NASA/MIT/recovered family's. CALCE remains the one genuine zero-
retrain cross-dataset check, unchanged in meaning from every earlier
stage.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))

from data_adapters import (
    oxford_cell_ids, iterate_oxford_cycles,
    hust_cell_ids, iterate_hust_cycles,
    xjtu_cell_ids, iterate_xjtu_cycles,
    iterate_calce_cycles,
)
from run_stage5_1_new_datasets_eval import xjtu_cell_ids_soh_valid  # excludes Batch-6/Sim_satellite - see that module for the root-caused reason
from sequence_features import build_dataset_tensors, apply_channel_norm
from stage4_pool import load_all_battery_tensors_stage4
from models.batlinet import BatLiNet
from stage1_common import OUT_DIR, PROC_DIR, ROOT

MAX_CYCLES_PER_BATTERY = 150  # training-only cap, see module docstring
RNG_SEED = 42
CALCE_CELLS = ["CS2_35", "CS2_36", "CS2_37"]


def build_full_pool():
    """Returns X_all (N,200,6) float32, soh_all (N,), bid_all (list[N]),
    dataset_family_all (list[N] - 'NASA_MIT' or 'Oxford'/'HUST'/'XJTU'),
    for the full non-CALCE pool (every real cycle, no subsampling -
    subsampling is applied later, only for the training subset)."""
    rng = np.random.default_rng(RNG_SEED)
    Xs, sohs, bids, fams = [], [], [], []

    print("[stage5-2] loading NASA+MIT+recovered (Stage 4 pool)...")
    base = load_all_battery_tensors_stage4()
    for bid, (X, soh, rul, dataset) in base.items():
        Xs.append(X); sohs.append(soh)
        bids += [bid] * len(soh); fams += ["NASA_MIT"] * len(soh)
    print(f"[stage5-2] NASA+MIT+recovered: {len(base)} batteries, {sum(len(v[1]) for v in base.values())} cycles")

    for fam_name, ids, iterate_fn in [
        ("Oxford", oxford_cell_ids(), iterate_oxford_cycles),
        ("HUST", hust_cell_ids(), iterate_hust_cycles),
        ("XJTU", xjtu_cell_ids_soh_valid(), iterate_xjtu_cycles),
    ]:
        t0 = time.time()
        n_cyc = 0
        for cid in ids:
            cycles = list(iterate_fn(cid))
            if len(cycles) < 5:
                continue
            X, soh, rul, idxs, censored = build_dataset_tensors(cycles)
            if X is None:
                continue
            Xs.append(X.astype(np.float32)); sohs.append(soh)
            bids += [f"{fam_name}:{cid}"] * len(idxs); fams += [fam_name] * len(idxs)
            n_cyc += len(idxs)
        print(f"[stage5-2] {fam_name}: {len(ids)} cells, {n_cyc} cycles ({time.time()-t0:.1f}s)")

    X_all = np.concatenate(Xs).astype(np.float32)
    soh_all = np.concatenate(sohs).astype(np.float32)
    bid_all = np.array(bids)
    fam_all = np.array(fams)
    print(f"[stage5-2] FULL POOL (no CALCE): {len(X_all)} cycles, {len(set(bid_all))} batteries, "
          f"families={dict(zip(*np.unique(fam_all, return_counts=True)))}")
    return X_all, soh_all, bid_all, fam_all


def build_calce_pool():
    Xs, sohs, bids = [], [], []
    for cid in CALCE_CELLS:
        cycles = list(iterate_calce_cycles(cid))
        X, soh, rul, idxs, censored = build_dataset_tensors(cycles)
        if X is None:
            continue
        Xs.append(X.astype(np.float32)); sohs.append(soh); bids += [cid] * len(idxs)
    return np.concatenate(Xs).astype(np.float32), np.concatenate(sohs).astype(np.float32), np.array(bids)


def train_batlinet(X_train, soh_train, norm_stats, n_epochs=12, batch_size=256, lr=3e-3,
                    lam=1.0, alpha=0.5, K_ref=32, seed=RNG_SEED, log_prefix="[stage5-2]"):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X_norm = apply_channel_norm(X_train, norm_stats)
    X_t = torch.tensor(X_norm, dtype=torch.float32)
    y_t = torch.tensor(soh_train, dtype=torch.float32)
    n = len(y_t)

    model = BatLiNet(in_channels=6, hidden_dim=32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n_steps = max(1, n // batch_size)
    for epoch in range(n_epochs):
        perm = rng.permutation(n)
        epoch_loss_intra, epoch_loss_inter = 0.0, 0.0
        for step in range(n_steps):
            idx = perm[step * batch_size:(step + 1) * batch_size]
            if len(idx) < 2:
                continue
            xb, yb = X_t[idx], y_t[idx]

            opt.zero_grad()
            pred_intra = model.forward_intra(xb)
            loss_intra = torch.mean((pred_intra - yb) ** 2)

            # random cross-pairing within the batch (paper: "randomly
            # assign reference cells" - approximated here by pairing
            # each batch element with a random OTHER batch element,
            # cheaper than all N(N-1) pairs over the full pool every
            # step while still exposing the inter-cell branch to a
            # fresh random pairing every step across the whole pool
            # over the course of training)
            perm2 = torch.randperm(len(idx))
            xb2, yb2 = xb[perm2], yb[perm2]
            dx = xb - xb2
            dy = yb - yb2
            pred_inter = model.forward_inter(dx)
            loss_inter = torch.mean((pred_inter - dy) ** 2)

            loss = loss_intra + lam * loss_inter
            loss.backward()
            opt.step()
            epoch_loss_intra += loss_intra.item()
            epoch_loss_inter += loss_inter.item()

        print(f"{log_prefix} epoch {epoch+1}/{n_epochs}: intra_mse={epoch_loss_intra/n_steps:.4f} "
              f"inter_mse={epoch_loss_inter/n_steps:.4f}")

    # fixed reference set for inference (sampled once from the training pool)
    ref_idx = rng.choice(n, size=min(K_ref, n), replace=False)
    x_refs = X_t[ref_idx]
    y_refs = y_t[ref_idx]
    return model, x_refs, y_refs


def evaluate(model, x_refs, y_refs, X_eval, soh_eval, norm_stats, alpha=0.5, batch_size=512):
    X_norm = apply_channel_norm(X_eval, norm_stats)
    X_t = torch.tensor(X_norm, dtype=torch.float32)
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            xb = X_t[i:i + batch_size]
            pred = model.predict(xb, x_refs, y_refs, alpha=alpha)
            preds.append(pred.numpy())
    model.train()
    pred_all = np.concatenate(preds)
    r2 = r2_score(soh_eval, pred_all)
    rmse = float(np.sqrt(mean_squared_error(soh_eval, pred_all)))
    return r2, rmse, pred_all


def capped_train_indices(bid_all, candidate_mask, cap, rng):
    """Evenly-spaced per-battery subsample of candidate_mask's True
    positions, capped at `cap` cycles/battery - TRAINING ONLY, see
    module docstring."""
    out = []
    cand_idx = np.where(candidate_mask)[0]
    for bid in np.unique(bid_all[cand_idx]):
        battery_idx = cand_idx[bid_all[cand_idx] == bid]
        if len(battery_idx) <= cap:
            out.append(battery_idx)
        else:
            chosen = np.round(np.linspace(0, len(battery_idx) - 1, cap)).astype(int)
            out.append(battery_idx[chosen])
    return np.concatenate(out)


def main():
    t_start = time.time()
    print("=== Stage 5.2: BatLiNet - inter-cell deep learning ===")
    X_all, soh_all, bid_all, fam_all = build_full_pool()

    print("\n[stage5-2] fitting FRESH channel_norm_stats on this new, larger, "
          "more diverse pool (own file, does NOT touch the deployed "
          "data/processed/channel_norm_stats.json - disclosed reasoning "
          "in module docstring: reusing Stage 4's NASA+MIT-only clip "
          "bounds here would badly saturate Oxford/HUST/XJTU's channels).")
    from sequence_features import compute_channel_norm_stats
    norm_stats = compute_channel_norm_stats(X_all)
    import json
    with open(PROC_DIR / "channel_norm_stats_batlinet.json", "w") as f:
        json.dump(norm_stats, f, indent=2)

    rng = np.random.default_rng(RNG_SEED)
    unique_batteries = np.unique(bid_all)
    print(f"\n[stage5-2] {len(unique_batteries)} unique batteries across the full non-CALCE pool")

    # === GroupKFold(5) in-domain CV, Stage 2.3's protocol ===
    print("\n=== GroupKFold(5) in-domain CV ===")
    gkf = GroupKFold(n_splits=5)
    # split the UNIQUE battery-id array itself (each id its own singleton
    # group) - equivalent to a plain battery-level 5-fold split, reusing
    # GroupKFold's implementation rather than hand-rolling one.
    fold_rows = []
    per_family_preds = {"NASA_MIT": [], "Oxford": [], "HUST": [], "XJTU": []}
    for fold_i, (train_bidx, test_bidx) in enumerate(gkf.split(unique_batteries, groups=unique_batteries)):
        train_batteries = set(unique_batteries[train_bidx])
        test_batteries = set(unique_batteries[test_bidx])
        train_mask = np.isin(bid_all, list(train_batteries))
        test_mask = np.isin(bid_all, list(test_batteries))

        train_idx = capped_train_indices(bid_all, train_mask, MAX_CYCLES_PER_BATTERY, rng)
        print(f"\n[stage5-2] fold {fold_i}: {len(train_batteries)} train batteries "
              f"({len(train_idx)} training cycles after cap), {len(test_batteries)} test batteries "
              f"({test_mask.sum()} eval cycles)")

        model, x_refs, y_refs = train_batlinet(
            X_all[train_idx], soh_all[train_idx], norm_stats,
            n_epochs=12, log_prefix=f"[stage5-2 fold{fold_i}]")
        r2, rmse, pred = evaluate(model, x_refs, y_refs, X_all[test_mask], soh_all[test_mask], norm_stats)
        print(f"[stage5-2] fold {fold_i}: R2={r2:.4f} RMSE={rmse:.4f}")
        fold_rows.append({"fold": fold_i, "r2": r2, "rmse": rmse, "n_test": int(test_mask.sum())})

        for fam in per_family_preds:
            fam_mask_in_test = test_mask & (fam_all == fam)
            if fam_mask_in_test.sum() > 0:
                _, _, fam_pred = evaluate(model, x_refs, y_refs, X_all[fam_mask_in_test],
                                           soh_all[fam_mask_in_test], norm_stats)
                per_family_preds[fam].append((soh_all[fam_mask_in_test], fam_pred))

    import pandas as pd
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(OUT_DIR / "stage5_2_batlinet_groupkfold.csv", index=False)
    print("\n[stage5-2] GroupKFold summary:")
    print(fold_df.to_string(index=False))
    print(f"[stage5-2] mean R2={fold_df['r2'].mean():.4f} (std {fold_df['r2'].std():.4f}) "
          f"mean RMSE={fold_df['rmse'].mean():.4f} (std {fold_df['rmse'].std():.4f})")
    print(f"[stage5-2] vs. Stage 4 XGBoost-fusion GroupKFold: R2=0.9658 (std 0.0207) RMSE=1.1388 (std 0.5673)")

    fam_summary = []
    for fam, pairs in per_family_preds.items():
        if not pairs:
            continue
        y_true = np.concatenate([p[0] for p in pairs])
        y_pred = np.concatenate([p[1] for p in pairs])
        fam_summary.append({"family": fam, "n": len(y_true), "r2": r2_score(y_true, y_pred),
                             "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred)))})
    fam_df = pd.DataFrame(fam_summary)
    fam_df.to_csv(OUT_DIR / "stage5_2_batlinet_groupkfold_per_family.csv", index=False)
    print("\n[stage5-2] per-family in-domain GroupKFold breakdown (each battery only in its OWN held-out fold):")
    print(fam_df.to_string(index=False))

    # === Final model (all non-CALCE data) for the CALCE zero-retrain check ===
    print("\n=== training FINAL model (full non-CALCE pool) for the CALCE zero-retrain check ===")
    full_mask = np.ones(len(bid_all), dtype=bool)
    train_idx_final = capped_train_indices(bid_all, full_mask, MAX_CYCLES_PER_BATTERY, rng)
    print(f"[stage5-2] final training set: {len(train_idx_final)} cycles (of {len(bid_all)} total)")
    final_model, x_refs_final, y_refs_final = train_batlinet(
        X_all[train_idx_final], soh_all[train_idx_final], norm_stats,
        n_epochs=15, log_prefix="[stage5-2 FINAL]")

    torch.save(final_model.state_dict(), ROOT / "models" / "batlinet_experimental.pt")
    print("[stage5-2] saved models/batlinet_experimental.pt (EXPERIMENTAL - not wired into live_inference.py/app.py)")

    print("\n=== CALCE zero-retrain eval ===")
    X_calce, soh_calce, bid_calce = build_calce_pool()
    r2_calce, rmse_calce, pred_calce = evaluate(final_model, x_refs_final, y_refs_final,
                                                  X_calce, soh_calce, norm_stats)
    print(f"[stage5-2] CALCE: R2={r2_calce:.4f} RMSE={rmse_calce:.4f} (n={len(soh_calce)}, "
          f"{len(set(bid_calce))} cells)")
    print(f"[stage5-2] vs. Stage 4 XGBoost-fusion CALCE: R2=0.5679 RMSE=14.155")

    summary = {
        "batlinet_groupkfold_mean_r2": float(fold_df["r2"].mean()),
        "batlinet_groupkfold_std_r2": float(fold_df["r2"].std()),
        "batlinet_groupkfold_mean_rmse": float(fold_df["rmse"].mean()),
        "batlinet_groupkfold_std_rmse": float(fold_df["rmse"].std()),
        "batlinet_calce_r2": r2_calce,
        "batlinet_calce_rmse": rmse_calce,
        "xgb_fusion_groupkfold_mean_r2": 0.9657628399971541,
        "xgb_fusion_groupkfold_mean_rmse": 1.1387732065275267,
        "xgb_fusion_calce_r2": 0.5678764726067165,
        "xgb_fusion_calce_rmse": 14.155331877704725,
    }
    pd.DataFrame([summary]).to_csv(OUT_DIR / "stage5_2_batlinet_summary.csv", index=False)
    print("\n[stage5-2] summary saved to outputs/stage5_2_batlinet_summary.csv")
    print(f"[stage5-2] TOTAL WALL TIME: {(time.time()-t_start)/60:.1f} minutes")


if __name__ == "__main__":
    main()
