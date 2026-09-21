"""
Research pass 2, item 3: PyBaMM-based physics-simulated pretraining.
Item 15 of the PRIOR research pass used a first-order equivalent-
circuit-model (ECM) voltage-sag approximation because no real simulator
was wired up - explicitly flagged as a scope limit there, not a dead
end. This item closes that gap: PyBaMM (pip-installable, no license
issue) installed CLEANLY in this environment (verified: `pip install
pybamm` succeeded with no errors, `pybamm.lithium_ion.SPM()` solves a
basic simulation immediately) and SPM+SEI-aging cycling simulations run
FAST here (30 simulated charge/discharge/hold cycles solve in ~1.5s) -
neither of item 15's two disclosed risk scenarios (won't install /
computationally infeasible) materialized, so this item proceeds with
genuine physics simulation, not a substitute proxy.

SIMULATION: PyBaMM's SPM with SEI growth enabled ("ec reaction
limited"), Chen2020's default parameter set as the base, PERTURBED per
simulated trajectory (SEI kinetic rate constant and negative electrode
particle radius, each independently log-uniform-jittered +/-40% around
the default - disclosed, untuned choice meant only to produce VARIED
capacity-fade trajectories, not calibrated to any real cell) - 12
trajectories x 60 cycles each ("Discharge at 1C until 3.0V" / "Charge
at 1C until 4.2V" / "Hold at 4.2V until C/50" per cycle). Each cycle's
DISCHARGE step's raw (t, V, I) is extracted directly from PyBaMM's own
solution object; current is NEGATED to match this project's own sign
convention (discharge current < 0, per ica_dv_dc.py's own documented
convention); temperature is a constant ambient placeholder (298.15 K)
since SPM has no active thermal submodel here - disclosed, not
fabricated as a measured channel. Each cycle's discharge capacity is
computed by the SAME trapezoidal |I|dt integration this project uses
everywhere (ica_dv_dc.py's own method, reused via sequence_features.
get_cycle_tensor which calls it internally) - genuine per-cycle SOH
from real simulated physics, not a synthetic label.

PRETEXT TASK: InfoNCE contrastive loss (same NT-Xent machinery as item
15, temperature=0.1) where the POSITIVE pair for each simulated cycle
is the NEXT cycle in the SAME simulated trajectory (temporally
adjacent, small degradation-state gap - the physics-simulated analogue
of item 15's ECM-perturbed "slightly more degraded view" positive
pair, except now genuinely simulated forward in time by the physics
model itself, not a hand-perturbed proxy); negatives are every other
cycle in the batch (other trajectories, or far-apart cycles in the
same trajectory).

FINE-TUNE/EVAL: byte-for-byte the SAME finetune()/eval_model()
protocol as Stage 7.2 and item 15 (imported directly, not
reimplemented) - same real TRAIN/TEST/held-out pools, same
PretrainableSOHModel architecture - so results land in the same table,
directly comparable to Stage 7.2's rows, item 15's ECM-contrastive row,
AND (per this research pass's own item 5 finding) the TRUE currently-
deployed model's real numbers, not the inflated reference figure used
in earlier passes.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pybamm
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
from stage7_common import load_pool_train_test, load_heldout, fit_norm_stats, norm_pool, OUT_DIR, MODEL_DIR
from models.world_model import CycleCNNEncoder
from run_stage7_2_selfsupervised_pretrain import finetune, eval_model
from sequence_features import get_cycle_tensor

SEED = 42
N_TRAJECTORIES = 12
N_CYCLES_PER_TRAJ = 60
BATCH_SIZE = 64
PRETRAIN_EPOCHS = 20
LR = 1e-3
TEMPERATURE = 0.1
AMBIENT_T_K = 298.15
DEVICE = torch.device("cpu")

# TRUE currently-deployed model's own numbers, per THIS research pass's
# own item 5 finding (models/xgb_soh_fusion.json, scored fresh, NOT the
# Stage-5-extended-reformulation model previously mislabeled "deployed"
# in Stage 7.2/item 15's own reference dicts).
TRUE_DEPLOYED_REFERENCE = {
    "in-domain (fixed split)": 0.974, "CALCE": 0.568, "Oxford": -2.694, "HUST": -0.152, "XJTU": -1.062,
}
STAGE7_2_RESULTS_PATH = Path(__file__).resolve().parents[1] / "outputs" / "stage7_2_selfsupervised_pretrain_results.csv"
ITEM15_RESULTS_PATH = Path(__file__).resolve().parents[1] / "outputs" / "researchpass_groupE15_contrastive_pretrain.csv"


def simulate_trajectory(traj_idx: int, rng: np.random.Generator) -> dict:
    sei_mult = float(np.exp(rng.uniform(np.log(0.6), np.log(1.4))))
    radius_mult = float(np.exp(rng.uniform(np.log(0.6), np.log(1.4))))

    model = pybamm.lithium_ion.SPM(options={"SEI": "ec reaction limited"})
    param = pybamm.ParameterValues("Chen2020")
    base_sei_k = param["SEI kinetic rate constant [m.s-1]"]
    base_radius = param["Negative particle radius [m]"]
    param["SEI kinetic rate constant [m.s-1]"] = base_sei_k * sei_mult
    param["Negative particle radius [m]"] = base_radius * radius_mult

    experiment = pybamm.Experiment(
        [("Discharge at 1C until 3.0V", "Charge at 1C until 4.2V", "Hold at 4.2V until C/50")]
        * N_CYCLES_PER_TRAJ
    )
    sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param)
    sol = sim.solve()

    cycles = []
    cap_first = None
    for cyc in sol.cycles:
        if len(cyc.steps) == 0:
            continue
        step = cyc.steps[0]  # discharge step
        t = step["Time [s]"].data
        V = step["Terminal voltage [V]"].data
        I = -step["Current [A]"].data  # negate to match this project's discharge<0 convention
        if len(t) < 15:
            continue
        Q = float(np.trapezoid(np.abs(I), t) / 3600.0)
        if cap_first is None:
            cap_first = Q
        cell = {"discharge": {"t": t, "V": V, "I": I, "T": np.full_like(t, AMBIENT_T_K)}}
        tensor = get_cycle_tensor(cell)
        if tensor is None:
            continue
        soh = 100.0 * Q / max(cap_first, 1e-9)
        cycles.append((tensor, soh))

    if len(cycles) < 5:
        return None
    X = np.stack([c[0] for c in cycles]).astype(np.float32)
    soh_arr = np.array([c[1] for c in cycles], dtype=np.float32)
    print(f"[pybamm-pretrain] trajectory {traj_idx}: sei_mult={sei_mult:.3f} radius_mult={radius_mult:.3f} "
          f"-> {len(cycles)} usable cycles, SOH {soh_arr[0]:.1f} -> {soh_arr[-1]:.1f}")
    return {"X": X, "soh": soh_arr}


def build_simulated_pool() -> dict:
    print(f"[pybamm-pretrain] simulating {N_TRAJECTORIES} SPM+SEI-aging trajectories "
          f"({N_CYCLES_PER_TRAJ} cycles each)...")
    rng = np.random.default_rng(SEED)
    pool = {}
    t0 = time.time()
    for i in range(N_TRAJECTORIES):
        traj = simulate_trajectory(i, rng)
        if traj is not None:
            pool[f"sim_{i}"] = traj
    print(f"[pybamm-pretrain] {len(pool)}/{N_TRAJECTORIES} trajectories usable, "
          f"simulation total time: {time.time()-t0:.1f}s")
    return pool


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = TEMPERATURE) -> torch.Tensor:
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    z = torch.cat([z1, z2], dim=0)
    sim = z @ z.T / temperature
    n = z1.size(0)
    mask = torch.eye(2 * n, dtype=torch.bool)
    sim = sim.masked_fill(mask, float("-inf"))
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)])
    return F.cross_entropy(sim, targets)


def build_adjacent_pairs(sim_pool: dict):
    Xi, Xj = [], []
    for bid, d in sim_pool.items():
        X = d["X"]
        for k in range(len(X) - 1):
            Xi.append(X[k]); Xj.append(X[k + 1])
    return np.stack(Xi).astype(np.float32), np.stack(Xj).astype(np.float32)


def pretrain_physics_contrastive(sim_pool: dict):
    Xi, Xj = build_adjacent_pairs(sim_pool)
    print(f"[pybamm-pretrain] {len(Xi)} temporally-adjacent simulated-cycle pairs built for InfoNCE pretraining")

    # normalize channels using the SIMULATED pool's OWN stats (this
    # pretraining phase never touches real data - the encoder is fully
    # fine-tuned on real, REAL-normalized data afterward anyway, so
    # this is a disclosed, self-consistent choice, not a leak)
    flat = np.concatenate([Xi, Xj], axis=0)
    mean = flat.mean(axis=(0, 1), keepdims=True)
    std = flat.std(axis=(0, 1), keepdims=True) + 1e-6
    Xi_n = (Xi - mean) / std
    Xj_n = (Xj - mean) / std

    rng = np.random.default_rng(SEED)
    n = len(Xi_n)
    n_val = max(1, int(0.15 * n))
    perm = rng.permutation(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    encoder = CycleCNNEncoder(in_channels=6, embed_dim=32).to(DEVICE)
    proj_head = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32)).to(DEVICE)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(proj_head.parameters()), lr=LR)

    Xi_t, Xj_t = torch.tensor(Xi_n), torch.tensor(Xj_n)

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
            z1 = proj_head(encoder(Xi_t[b]))
            z2 = proj_head(encoder(Xj_t[b]))
            loss = info_nce_loss(z1, z2)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        return float(np.mean(losses)) if losses else float("nan")

    val_loss = float("nan")
    for epoch in range(PRETRAIN_EPOCHS):
        tr_loss = run_epoch(tr_idx, train=True)
        with torch.no_grad():
            val_loss = run_epoch(val_idx, train=False)
        print(f"[pybamm-pretrain] epoch {epoch+1}/{PRETRAIN_EPOCHS}: train_nce={tr_loss:.4f} val_nce={val_loss:.4f}")
        if not np.isfinite(val_loss):
            print("[pybamm-pretrain] NON-CONVERGENCE: val loss NaN/Inf - stopping early.")
            break

    return encoder.state_dict(), val_loss


def main():
    t0 = time.time()
    print("=== Research pass 2, item 3: PyBaMM-based physics-simulated contrastive pretraining ===")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    sim_pool = build_simulated_pool()
    if len(sim_pool) < 3:
        print("[pybamm-pretrain] STOPPING: fewer than 3 usable simulated trajectories - "
              "reporting this plainly rather than proceeding on too little simulated data.")
        return

    pretrained_state, final_val_nce = pretrain_physics_contrastive(sim_pool)
    torch.save(pretrained_state, MODEL_DIR / "_experimental_pybamm_pretrained_encoder.pt")

    print("\n[pybamm-pretrain] loading REAL labeled data for fine-tuning/eval (unchanged Stage 7.2 pipeline)...")
    train, test = load_pool_train_test()
    stats = fit_norm_stats(train)
    train_n = norm_pool(train, stats)
    test_n = norm_pool(test, stats)

    print("\n--- fine-tuning: pybamm_physics_pretrained ---")
    model = finetune(train_n, pretrained_state, "pybamm_physics_pretrained")
    torch.save(model.state_dict(), MODEL_DIR / "_experimental_ssl_soh_pybamm_pretrained.pt")

    results = []
    r = eval_model(model, test_n, "in-domain (fixed split)")
    if r: results.append({"variant": "pybamm_physics_pretrained", **r})
    for name in ["CALCE", "Oxford", "HUST", "XJTU"]:
        held = load_heldout(name)
        held_n = norm_pool(held, stats)
        r = eval_model(model, held_n, name)
        if r: results.append({"variant": "pybamm_physics_pretrained", **r})

    results_df = pd.DataFrame(results)

    combined_parts = [results_df]
    if STAGE7_2_RESULTS_PATH.exists():
        combined_parts.append(pd.read_csv(STAGE7_2_RESULTS_PATH))
    if ITEM15_RESULTS_PATH.exists():
        item15_df = pd.read_csv(ITEM15_RESULTS_PATH)
        item15_df = item15_df[item15_df.get("variant", "") == "contrastive_pretrained"] if "variant" in item15_df.columns else item15_df
        combined_parts.append(item15_df)
    for eval_set, r2 in TRUE_DEPLOYED_REFERENCE.items():
        combined_parts.append(pd.DataFrame([{"variant": "TRUE_deployed_xgb_fusion_REFERENCE (item5-verified)",
                                              "eval_set": eval_set, "r2": r2, "rmse": None, "n": None}]))
    combined = pd.concat(combined_parts, ignore_index=True)
    combined.to_csv(OUT_DIR / "researchpass2_item3_pybamm_pretrain.csv", index=False)

    print("\n=== FULL COMPARISON (this item + Stage 7.2 rows + item 15's ECM-contrastive row + TRUE deployed reference) ===")
    print(combined.to_string(index=False))

    indomain_this = results_df[results_df["eval_set"] == "in-domain (fixed split)"]["r2"]
    if not indomain_this.empty:
        this_r2 = float(indomain_this.iloc[0])
        print(f"\n[pybamm-pretrain] in-domain R2 (this item, PyBaMM physics pretrained): {this_r2:.4f}")
        if STAGE7_2_RESULTS_PATH.exists():
            s72 = pd.read_csv(STAGE7_2_RESULTS_PATH)
            s72_pretrained = s72[(s72["variant"] == "pretrained") & (s72["eval_set"] == "in-domain (fixed split)")]["r2"]
            if not s72_pretrained.empty:
                print(f"[pybamm-pretrain] vs Stage 7.2 order-ranking pretrained R2={float(s72_pretrained.iloc[0]):.4f}: "
                      f"{'WIN' if this_r2 > float(s72_pretrained.iloc[0]) else 'LOSS'}")
        if ITEM15_RESULTS_PATH.exists():
            item15_indomain = pd.read_csv(ITEM15_RESULTS_PATH)
            item15_indomain = item15_indomain[item15_indomain["eval_set"] == "in-domain (fixed split)"]["r2"] if "eval_set" in item15_indomain.columns else pd.Series([], dtype=float)
            if not item15_indomain.empty:
                print(f"[pybamm-pretrain] vs item 15 ECM-contrastive R2={float(item15_indomain.iloc[-1]):.4f}: "
                      f"{'WIN' if this_r2 > float(item15_indomain.iloc[-1]) else 'LOSS'}")

    print(f"\n[pybamm-pretrain] TOTAL TIME: {(time.time()-t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
