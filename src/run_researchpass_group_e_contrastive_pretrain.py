"""
Research pass Group E, item 15: ACCEPT-style physics-simulated
contrastive pretraining, compared directly against Stage 7.2's already-
recorded self-supervised (order-ranking) pretraining result.

DISCLOSED SCOPE, stated plainly rather than overclaimed: ACCEPT's own
published pretext task pairs a REAL cycle with a physics-SIMULATED
counterpart (from a full electrochemical/equivalent-circuit simulator)
as a positive pair for contrastive (InfoNCE/NT-Xent) learning. This
project does not have a full SPM/electrochemical simulator wired up as
an on-the-fly augmentation source (run_stage7_3_neural_operator_spm.py
is a separate, heavy, offline pipeline, not built for this). What IS
implemented is a genuine, disclosed, SIMPLIFIED physics-motivated
augmentation via a first-order equivalent-circuit perturbation applied
directly to each real cycle tensor:
  V' = V - delta_r * I        (Ohm's-law impedance-growth perturbation:
                                a higher internal resistance means a
                                larger voltage sag at the same current -
                                a real, standard ECM effect, not an
                                arbitrary augmentation)
  dQdV', dVdQ' rescaled by a small random factor (simulating a small
                                capacity-fade shift in the differential
                                curves' magnitude)
This is a real physics-motivated simulation of "what would this SAME
cycle look like on a slightly more degraded cell," just via a first-
order ECM approximation rather than a full simulator - the same class
of disclosed substitution already used elsewhere in this pass (SISSO-
inspired, Cox instead of DeepHit).

PRETEXT TASK: standard InfoNCE contrastive loss (NT-Xent, temperature
0.1) between each real cycle's embedding and its own ECM-perturbed
positive view, with all other cycles in the batch as negatives -
encoder must learn a representation where physically-plausible
"more degraded" perturbations of the SAME cycle stay close, and
different cycles stay apart.

FINE-TUNING/EVAL: byte-for-byte the SAME finetune()/eval_model()
protocol as Stage 7.2 (imported directly, not reimplemented), same
train/test/held-out pools, same architecture (PretrainableSOHModel) -
so results land in the exact same table and are directly comparable to
both Stage 7.2 rows (pretrained order-ranking, random_init) and the
deployed XGBoost-fusion reference.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
from stage7_common import load_pool_train_test, load_heldout, fit_norm_stats, norm_pool, OUT_DIR, MODEL_DIR
from models.world_model import CycleCNNEncoder
from run_stage7_2_selfsupervised_pretrain import finetune, eval_model, DEPLOYED_REFERENCE

SEED = 42
BATCH_SIZE = 64
PRETRAIN_EPOCHS = 20
LR = 1e-3
TEMPERATURE = 0.1
DEVICE = torch.device("cpu")

# Stage 7.2's own already-recorded results, read back for direct
# side-by-side comparison rather than re-run (identical protocol,
# no reason to re-pay the ~20-epoch pretrain + 2x20-epoch finetune cost).
STAGE7_2_RESULTS_PATH = Path(__file__).resolve().parents[1] / "outputs" / "stage7_2_selfsupervised_pretrain_results.csv"


def ecm_perturb(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """First-order equivalent-circuit-model perturbation - see module
    docstring. X: (n, 200, 6), channel order V_t,I_t,T_t,dQdV,dVdQ,dIdV."""
    Xp = X.copy()
    n = len(X)
    delta_r = rng.uniform(0.01, 0.08, size=(n, 1)).astype(np.float32)  # small IR-growth draw per sample
    Xp[:, :, 0] = Xp[:, :, 0] - delta_r * Xp[:, :, 1]  # V' = V - delta_r * I
    fade_scale = rng.uniform(0.90, 1.0, size=(n, 1)).astype(np.float32)
    Xp[:, :, 3] = Xp[:, :, 3] * fade_scale  # dQdV
    Xp[:, :, 4] = Xp[:, :, 4] * fade_scale  # dVdQ
    return Xp


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = TEMPERATURE) -> torch.Tensor:
    """Standard NT-Xent over a batch of (real, ECM-perturbed) pairs -
    each real embedding's positive is its own perturbed view; all other
    2N-2 embeddings in the batch (real+perturbed of OTHER cycles) are
    negatives."""
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    z = torch.cat([z1, z2], dim=0)  # (2N, D)
    sim = z @ z.T / temperature  # (2N, 2N)
    n = z1.size(0)
    mask = torch.eye(2 * n, dtype=torch.bool)
    sim = sim.masked_fill(mask, float("-inf"))
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)])
    return F.cross_entropy(sim, targets)


def pretrain_contrastive(train_n: dict):
    print("[contrastive-pretrain] building ECM-perturbed positive pairs from TRAIN battery pool only...")
    rng = np.random.default_rng(SEED)
    X_all = []
    for bid, (X, soh, rul) in train_n.items():
        X_all.append(X)
    X_all = np.concatenate(X_all).astype(np.float32)
    print(f"[contrastive-pretrain] {len(X_all)} real cycles pooled from {len(train_n)} batteries")

    encoder = CycleCNNEncoder(in_channels=6, embed_dim=32).to(DEVICE)
    proj_head = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32)).to(DEVICE)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(proj_head.parameters()), lr=LR)

    n_val = max(1, int(0.15 * len(X_all)))
    perm = rng.permutation(len(X_all))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    def run_epoch(idx, train: bool):
        encoder.train(train); proj_head.train(train)
        losses = []
        idx = idx.copy()
        if train:
            rng.shuffle(idx)
        for s in range(0, len(idx), BATCH_SIZE):
            b = idx[s:s + BATCH_SIZE]
            if len(b) < 2:
                continue
            Xr = X_all[b]
            Xp = ecm_perturb(Xr, rng)
            z1 = proj_head(encoder(torch.tensor(Xr)))
            z2 = proj_head(encoder(torch.tensor(Xp)))
            loss = info_nce_loss(z1, z2)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        return float(np.mean(losses)) if losses else float("nan")

    for epoch in range(PRETRAIN_EPOCHS):
        tr_loss = run_epoch(tr_idx, train=True)
        with torch.no_grad():
            val_loss = run_epoch(val_idx, train=False)
        print(f"[contrastive-pretrain] epoch {epoch+1}/{PRETRAIN_EPOCHS}: train_nce={tr_loss:.4f} val_nce={val_loss:.4f}")
        if not np.isfinite(val_loss):
            print("[contrastive-pretrain] NON-CONVERGENCE: val loss NaN/Inf - stopping early.")
            break

    return encoder.state_dict(), val_loss


def main():
    t0 = time.time()
    print("=== Research pass Group E, item 15: ACCEPT-style physics-simulated contrastive pretraining ===")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train, test = load_pool_train_test()
    stats = fit_norm_stats(train)
    train_n = norm_pool(train, stats)
    test_n = norm_pool(test, stats)
    print(f"[contrastive-pretrain] pool: {len(train)} train batteries, {len(test)} test batteries")

    pretrained_state, final_val_nce = pretrain_contrastive(train_n)
    torch.save(pretrained_state, MODEL_DIR / "_experimental_contrastive_pretrained_encoder.pt")

    print("\n--- fine-tuning: contrastive_pretrained ---")
    model = finetune(train_n, pretrained_state, "contrastive_pretrained")
    torch.save(model.state_dict(), MODEL_DIR / "_experimental_ssl_soh_contrastive_pretrained.pt")

    results = []
    r = eval_model(model, test_n, "in-domain (fixed split)")
    if r: results.append({"variant": "contrastive_pretrained", **r})
    for name in ["CALCE", "Oxford", "HUST", "XJTU"]:
        held = load_heldout(name)
        held_n = norm_pool(held, stats)
        r = eval_model(model, held_n, name)
        if r: results.append({"variant": "contrastive_pretrained", **r})

    results_df = pd.DataFrame(results)

    if STAGE7_2_RESULTS_PATH.exists():
        stage7_2_df = pd.read_csv(STAGE7_2_RESULTS_PATH)
        combined = pd.concat([stage7_2_df, results_df], ignore_index=True)
    else:
        print("[contrastive-pretrain] WARNING: stage7_2 results file not found, reporting this item alone")
        combined = pd.concat([results_df, pd.DataFrame(
            [{"variant": "deployed_xgb_fusion_REFERENCE", "eval_set": k, "r2": v, "rmse": None, "n": None}
             for k, v in DEPLOYED_REFERENCE.items()])], ignore_index=True)

    combined.to_csv(OUT_DIR / "researchpass_groupE15_contrastive_pretrain.csv", index=False)
    print("\n=== FULL COMPARISON (Stage 7.2 order-ranking rows + this item's contrastive row + deployed reference) ===")
    print(combined.to_string(index=False))

    indomain_this = results_df[results_df["eval_set"] == "in-domain (fixed split)"]["r2"]
    if STAGE7_2_RESULTS_PATH.exists() and not indomain_this.empty:
        stage72_pretrained_indomain = stage7_2_df[(stage7_2_df["variant"] == "pretrained") &
                                                    (stage7_2_df["eval_set"] == "in-domain (fixed split)")]["r2"]
        if not stage72_pretrained_indomain.empty:
            this_r2 = float(indomain_this.iloc[0])
            s72_r2 = float(stage72_pretrained_indomain.iloc[0])
            verdict = "WIN" if this_r2 > s72_r2 else "LOSS"
            print(f"\n[contrastive-pretrain] in-domain R2 vs. Stage 7.2 order-ranking pretrained "
                  f"(R2={s72_r2:.4f}): {verdict} ({this_r2:.4f})")

    print(f"\n[contrastive-pretrain] final val NT-Xent loss: {final_val_nce:.4f}")
    print(f"[contrastive-pretrain] TOTAL TIME: {(time.time()-t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
