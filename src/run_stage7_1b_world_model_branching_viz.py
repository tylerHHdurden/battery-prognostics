"""
Stage 7 website overhaul, Phase 1 item 8a: generates the data behind the
World Model's new "branching future trajectories" visualization on the
dashboard - genuinely different from every other chart on the site
(which show one predicted line vs. one true line).

The trained World Model (models/_experimental_world_model.pt) is
DETERMINISTIC - a fixed window produces one fixed forecast. To show
several "plausible" future paths without retraining it as a genuinely
stochastic model, this script perturbs the model's own initial latent
state (h0, the output of its CNN-encoder + PatchTST-lite context
summary) with small Gaussian noise before each of N rollouts, then lets
the SAME deterministic dynamics equation roll each perturbed state
forward independently - a standard, disclosed way to visualize a
deterministic forecaster's own uncertainty (the site's own copy states
this plainly, not implied as genuine stochastic sampling).

Picks ONE real TEST-split battery, one real window of W=10 past cycles,
generates 12 branching 15-cycle-ahead trajectories from that exact
point, alongside the battery's OWN true SOH for the same future cycles
(for visual comparison, not used to bias the forecast).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
from stage7_common import load_pool_train_test, fit_norm_stats, norm_pool, OUT_DIR, ROOT
from models.world_model import WorldModel

SEED = 42
WINDOW = 10
HORIZON = 15
N_TRAJECTORIES = 12
NOISE_STD = 0.35  # perturbation scale on h0, in the transformer's own d_model=64 latent units


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    train, test = load_pool_train_test()
    stats = fit_norm_stats(train)
    test_n = norm_pool(test, stats)

    # Pick the TEST battery with the most cycles, so there's a long real
    # future trajectory to plot alongside the forecast branches - not
    # cherry-picked for a flattering result, just for a visually useful
    # amount of context on both sides of the "current cycle" marker.
    battery_id = max(test_n, key=lambda k: len(test_n[k][1]))
    X, soh, rul = test_n[battery_id]
    print(f"[branching-viz] selected battery: {battery_id} ({len(soh)} cycles)")

    start = len(soh) // 3  # a point with real history before AND after it
    if start + WINDOW + HORIZON > len(soh):
        start = max(0, len(soh) - WINDOW - HORIZON)
    window_X = X[start:start + WINDOW]
    window_soh = soh[start:start + WINDOW]
    true_future = soh[start + WINDOW:start + WINDOW + HORIZON]
    current_cycle = start + WINDOW - 1

    model = WorldModel(embed_dim=32, window=WINDOW, patch_len=2, d_model=64)
    model.load_state_dict(torch.load(ROOT / "models" / "_experimental_world_model.pt"))
    model.eval()

    Xw = torch.tensor(window_X, dtype=torch.float32).unsqueeze(0)
    soh_w = torch.tensor(window_soh, dtype=torch.float32).unsqueeze(0)

    rows = []
    # cycles -WINDOW+1 .. 0 relative to "current cycle" = real observed history
    for i, s in enumerate(window_soh):
        rows.append({"battery_id": battery_id, "rel_cycle": i - WINDOW + 1, "soh": float(s),
                      "series": "observed history", "trajectory_id": -1})
    # true future, if available (may be shorter than HORIZON near end of life)
    for i, s in enumerate(true_future):
        rows.append({"battery_id": battery_id, "rel_cycle": i + 1, "soh": float(s),
                      "series": "true future (for comparison, not used by the model)", "trajectory_id": -1})

    with torch.no_grad():
        h0 = model.encode_context(Xw, soh_w)  # (1, d_model)
        soh_last = soh_w[:, -1]
        for traj in range(N_TRAJECTORIES):
            noise = torch.randn_like(h0) * NOISE_STD
            h0_perturbed = h0 + noise
            preds = model.rollout(h0_perturbed, soh_last, HORIZON).squeeze(0).numpy()
            for i, p in enumerate(preds):
                rows.append({"battery_id": battery_id, "rel_cycle": i + 1, "soh": float(p),
                              "series": f"forecast branch {traj+1}", "trajectory_id": traj})

    df = pd.DataFrame(rows)
    out_path = OUT_DIR / "stage7_1_world_model_branching_trajectories.csv"
    df.to_csv(out_path, index=False)
    print(f"[branching-viz] saved {len(df)} rows to {out_path}")
    print(f"[branching-viz] current cycle (absolute, within {battery_id}'s own sequence): {current_cycle}")
    spread_at_h1 = df[(df.trajectory_id >= 0) & (df.rel_cycle == 1)]["soh"].std()
    spread_at_hN = df[(df.trajectory_id >= 0) & (df.rel_cycle == HORIZON)]["soh"].std()
    print(f"[branching-viz] branch spread (std dev across the {N_TRAJECTORIES} trajectories): "
          f"horizon 1 = {spread_at_h1:.4f}, horizon {HORIZON} = {spread_at_hN:.4f} "
          f"({'widens as expected' if spread_at_hN > spread_at_h1 else 'does NOT widen - reported honestly either way'})")


if __name__ == "__main__":
    main()
