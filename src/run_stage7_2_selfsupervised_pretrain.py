"""
Stage 7.2: degradation-aligned self-supervised pretraining (cycle-order-
ranking pretext task, see src/models/degradation_pretrain.py for the
architecture + the disclosed scope note on what "unlabeled" actually
means for this project's data).

Protocol:
1. PRETRAIN a CycleCNNEncoder + OrderRankingHead on random (cycle_i,
   cycle_j) pairs sampled from the SAME battery, TRAIN split only (no
   SOH labels used, no held-out data touched - preserves zero-retrain
   validity for step 3).
2. FINE-TUNE two copies of the SAME architecture (PretrainableSOHModel)
   on the actual per-cycle SOH regression task, same TRAIN split, same
   hyperparameters/epochs: (a) starting from the pretrained encoder
   weights, (b) starting from a fresh random initialization - isolating
   whether pretraining itself helps.
3. Evaluate both in-domain (TEST split) and zero-retrain (CALCE/Oxford/
   HUST/XJTU), reported side by side, plus a reference row of the
   already-established deployed XGBoost-fusion model's own numbers for
   context (a different architecture entirely - not a literal apples-
   to-apples comparison, disclosed here rather than implied).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
from stage7_common import load_pool_train_test, load_heldout, fit_norm_stats, norm_pool, OUT_DIR, MODEL_DIR
from models.degradation_pretrain import OrderRankingHead, PretrainableSOHModel
from models.world_model import CycleCNNEncoder

SEED = 42
PAIRS_PER_BATTERY = 300
PRETRAIN_EPOCHS = 20
FINETUNE_EPOCHS = 20
BATCH_SIZE = 64
LR = 1e-3
DEVICE = torch.device("cpu")

# Reference numbers, already established and reported in Stage 6/6-closeout
# (this project's own deployed XGBoost-fusion pipeline) - for context only.
DEPLOYED_REFERENCE = {
    "in-domain (fixed split)": 0.973, "CALCE": 0.740, "Oxford": 0.953,
    "HUST": 0.800, "XJTU": -1.775,
}


def sample_pairs(pool: dict, pairs_per_battery=PAIRS_PER_BATTERY, rng=None):
    rng = rng or np.random.default_rng(SEED)
    Xi, Xj, order_y, gap_y = [], [], [], []
    for bid, (X, soh, rul) in pool.items():
        n = len(X)
        if n < 2:
            continue
        n_pairs = min(pairs_per_battery, n * (n - 1) // 2)
        idx_i = rng.integers(0, n, size=n_pairs)
        idx_j = rng.integers(0, n, size=n_pairs)
        keep = idx_i != idx_j
        idx_i, idx_j = idx_i[keep], idx_j[keep]
        for i, j in zip(idx_i, idx_j):
            Xi.append(X[i]); Xj.append(X[j])
            order_y.append(1.0 if j > i else 0.0)
            gap_y.append(np.log1p(abs(int(j) - int(i))))
    return (np.stack(Xi).astype(np.float32), np.stack(Xj).astype(np.float32),
            np.array(order_y, dtype=np.float32), np.array(gap_y, dtype=np.float32))


def pretrain_encoder(train_n: dict):
    print("[ssl-pretrain] sampling order-ranking pairs from TRAIN battery pool only...")
    rng = np.random.default_rng(SEED)
    Xi, Xj, order_y, gap_y = sample_pairs(train_n, rng=rng)
    print(f"[ssl-pretrain] {len(Xi)} pairs sampled from {len(train_n)} batteries")
    n_val = max(1, int(0.15 * len(Xi)))
    perm = rng.permutation(len(Xi))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    encoder = CycleCNNEncoder(in_channels=6, embed_dim=32).to(DEVICE)
    head = OrderRankingHead(embed_dim=32).to(DEVICE)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=LR)

    Xi_t, Xj_t = torch.tensor(Xi), torch.tensor(Xj)
    order_t, gap_t = torch.tensor(order_y), torch.tensor(gap_y)

    def run_epoch(idx, train: bool):
        encoder.train(train); head.train(train)
        losses, order_correct, n_seen = [], 0, 0
        rng.shuffle(idx) if train else None
        for s in range(0, len(idx), BATCH_SIZE):
            b = idx[s:s + BATCH_SIZE]
            zi, zj = encoder(Xi_t[b]), encoder(Xj_t[b])
            order_logit, gap_pred = head(zi, zj)
            loss = nn.functional.binary_cross_entropy_with_logits(order_logit, order_t[b]) + \
                nn.functional.mse_loss(gap_pred, gap_t[b])
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
            order_correct += ((order_logit > 0).float() == order_t[b]).sum().item()
            n_seen += len(b)
        return float(np.mean(losses)), order_correct / n_seen

    for epoch in range(PRETRAIN_EPOCHS):
        tr_loss, tr_acc = run_epoch(tr_idx.copy(), train=True)
        with torch.no_grad():
            val_loss, val_acc = run_epoch(val_idx.copy(), train=False)
        print(f"[ssl-pretrain] epoch {epoch+1}/{PRETRAIN_EPOCHS}: train_loss={tr_loss:.4f} "
              f"order_acc={tr_acc:.3f} | val_loss={val_loss:.4f} val_order_acc={val_acc:.3f}")
        if not np.isfinite(val_loss):
            print("[ssl-pretrain] NON-CONVERGENCE: val loss NaN/Inf - stopping pretraining early.")
            break

    print(f"[ssl-pretrain] final val order-ranking accuracy: {val_acc:.3f} "
          f"(0.5=random guessing, 1.0=perfect - {'clearly above chance, encoder learned real degradation-order signal' if val_acc > 0.6 else 'close to chance, pretext task may not have taught much'})")
    return encoder.state_dict(), val_acc


def finetune(pool_train: dict, pretrained_encoder_state, label: str):
    model = PretrainableSOHModel(embed_dim=32).to(DEVICE)
    if pretrained_encoder_state is not None:
        model.encoder.load_state_dict(pretrained_encoder_state)

    X_all, soh_all = [], []
    for bid, (X, soh, rul) in pool_train.items():
        X_all.append(X); soh_all.append(soh)
    X_all = np.concatenate(X_all).astype(np.float32)
    soh_all = np.concatenate(soh_all).astype(np.float32)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(X_all))
    n_val = max(1, int(0.15 * len(X_all)))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    X_t, soh_t = torch.tensor(X_all), torch.tensor(soh_all)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best_val, best_state = float("inf"), None
    for epoch in range(FINETUNE_EPOCHS):
        model.train()
        idx = tr_idx.copy(); rng.shuffle(idx)
        tr_losses = []
        for s in range(0, len(idx), BATCH_SIZE):
            b = idx[s:s + BATCH_SIZE]
            opt.zero_grad()
            pred = model(X_t[b])
            loss = nn.functional.mse_loss(pred, soh_t[b])
            loss.backward(); opt.step()
            tr_losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            val_pred = model(X_t[val_idx])
            val_loss = nn.functional.mse_loss(val_pred, soh_t[val_idx]).item()
        print(f"[ft:{label}] epoch {epoch+1}/{FINETUNE_EPOCHS}: train_mse={np.mean(tr_losses):.4f} val_mse={val_loss:.4f}")
        if not np.isfinite(val_loss):
            print(f"[ft:{label}] NON-CONVERGENCE: val loss NaN/Inf - stopping early.")
            break
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def eval_model(model, pool: dict, label: str):
    model.eval()
    X_all, soh_all = [], []
    for bid, (X, soh, rul) in pool.items():
        X_all.append(X); soh_all.append(soh)
    if not X_all:
        return None
    X_all = np.concatenate(X_all).astype(np.float32)
    soh_all = np.concatenate(soh_all).astype(np.float32)
    with torch.no_grad():
        pred = model(torch.tensor(X_all)).numpy()
    r2 = r2_score(soh_all, pred)
    rmse = float(np.sqrt(mean_squared_error(soh_all, pred)))
    print(f"[eval] {label}: R2={r2:.4f} RMSE={rmse:.4f} n={len(soh_all)}")
    return {"eval_set": label, "r2": r2, "rmse": rmse, "n": len(soh_all)}


def main():
    t0 = time.time()
    print("=== Stage 7.2: degradation-aligned self-supervised pretraining ===")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train, test = load_pool_train_test()
    stats = fit_norm_stats(train)
    train_n = norm_pool(train, stats)
    test_n = norm_pool(test, stats)
    print(f"[7.2] pool: {len(train)} train batteries, {len(test)} test batteries")

    pretrained_state, order_acc = pretrain_encoder(train_n)
    torch.save(pretrained_state, MODEL_DIR / "_experimental_degradation_pretrained_encoder.pt")

    results = []
    for label, enc_state in [("pretrained", pretrained_state), ("random_init", None)]:
        print(f"\n--- fine-tuning: {label} ---")
        model = finetune(train_n, enc_state, label)
        torch.save(model.state_dict(), MODEL_DIR / f"_experimental_ssl_soh_{label}.pt")
        r = eval_model(model, test_n, "in-domain (fixed split)")
        if r: results.append({"variant": label, **r})
        for name in ["CALCE", "Oxford", "HUST", "XJTU"]:
            held = load_heldout(name)
            held_n = norm_pool(held, stats)
            r = eval_model(model, held_n, name)
            if r: results.append({"variant": label, **r})

    for eval_set, ref_r2 in DEPLOYED_REFERENCE.items():
        results.append({"variant": "deployed_xgb_fusion_REFERENCE", "eval_set": eval_set, "r2": ref_r2,
                         "rmse": None, "n": None})

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "stage7_2_selfsupervised_pretrain_results.csv", index=False)
    print("\n=== FULL RESULTS ===")
    print(results_df.to_string(index=False))
    print(f"\n[7.2] pretext-task final order-ranking val accuracy: {order_acc:.3f}")
    print(f"[7.2] TOTAL TIME: {(time.time()-t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
