"""
Phase 5: TreeSHAP on XGBoost (base + meta-learner), DeepSHAP-with-
KernelSHAP-fallback on VLSTM/CNN-LSTM/PiFormer.

Per the task: try DeepSHAP first; if it errors out for a model, fall back
to KernelSHAP immediately and log which method was actually used for
which model, rather than burning time forcing DeepSHAP to work.

KernelSHAP tractability note: our deep models take (200 timesteps x 6
channels) = 1200 scalar inputs. Full per-scalar KernelSHAP over 1200
"features" is computationally infeasible on CPU (KernelSHAP cost scales
with n_features x nsamples). When DeepSHAP fails, this falls back to
GROUPED KernelSHAP: the 200 positions per channel are pooled into 10
coarse bins x 6 channels = 60 groups, each perturbed as a unit (masked
group -> replaced with the background mean for those positions). This is
the standard way to make KernelSHAP tractable on structured/sequence
input and is what makes the fallback actually finish in reasonable time,
at the cost of losing single-timestep resolution (kept: which ~1/10th of
the voltage/time range matters, lost: which exact sample).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
PRED_DIR = PROC_DIR / "predictions"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

N_GROUPS_PER_CHANNEL = 10
SEQ_LEN = 200
GROUP_SIZE = SEQ_LEN // N_GROUPS_PER_CHANNEL


# --------------------------------------------------------------------------
# TreeSHAP: XGBoost base learner (7 BFA-selected HIs) + meta-learner
# --------------------------------------------------------------------------

def tree_shap_xgboost():
    from health_indicators import HI_NAMES

    df = pd.read_parquet(PROC_DIR / "hi_table.parquet")
    df = df[df["dataset"].isin(["NASA", "MIT"])].reset_index(drop=True)
    with open(PROC_DIR / "bfa_selected_features.txt") as f:
        selected = [l.strip() for l in f if l.strip()]

    X = df[selected].to_numpy(dtype=float, copy=True)
    col_medians = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_medians, inds[1])

    model = XGBRegressor()
    model.load_model(str(ROOT / "models" / "xgb_soh.json"))

    explainer = shap.TreeExplainer(model)
    sample_idx = np.random.default_rng(42).choice(len(X), size=min(2000, len(X)), replace=False)
    shap_values = explainer.shap_values(X[sample_idx])
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranking = sorted(zip(selected, mean_abs), key=lambda t: -t[1])
    print("[shap] XGBoost (base learner) TreeSHAP mean|SHAP| ranking of BFA-selected features:")
    for name, val in ranking:
        print(f"    {name}: {val:.4f}")

    pd.DataFrame(ranking, columns=["feature", "mean_abs_shap"]).to_csv(
        OUT_DIR / "shap_xgboost_base_ranking.csv", index=False
    )
    return ranking


def tree_shap_meta():
    meta = XGBRegressor()
    meta.load_model(str(ROOT / "models" / "xgb_meta.json"))
    test_df = pd.read_csv(PRED_DIR / "ensemble_test_preds.csv")
    base_cols = ["pred_XGBoost", "pred_VLSTM", "pred_CNNLSTM", "pred_PiFormer"]
    X = test_df[base_cols].to_numpy()

    explainer = shap.TreeExplainer(meta)
    shap_values = explainer.shap_values(X)
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranking = sorted(zip(base_cols, mean_abs), key=lambda t: -t[1])
    print("[shap] Stacking-XGBoost meta-learner TreeSHAP ranking of base learners:")
    for name, val in ranking:
        print(f"    {name}: {val:.4f}")
    pd.DataFrame(ranking, columns=["base_learner", "mean_abs_shap"]).to_csv(
        OUT_DIR / "shap_meta_ranking.csv", index=False
    )
    return ranking


# --------------------------------------------------------------------------
# Deep models: DeepSHAP with KernelSHAP fallback
# --------------------------------------------------------------------------

def _grouped_kernelshap_one_instance(model, x0, bg_mean, n_channels, nsamples=200):
    """
    Explains ONE real test instance x0 (200, n_channels) against a
    background/baseline of bg_mean, using (channel, coarse-time/voltage-
    bin) groups as the perturbation unit. mask=1 for a group means "keep
    x0's real value there"; mask=0 means "replace with baseline mean" -
    the standard present/absent semantics KernelSHAP expects, made
    explicit here because a first draft of this function got it backwards
    (perturbed toward OTHER background samples instead of toward the
    actual x0 being explained, which wouldn't have explained anything
    about x0's own prediction). Caught in code review before ever running.
    """
    n_groups_total = n_channels * N_GROUPS_PER_CHANNEL

    def predict(mask_batch):
        n_samples = mask_batch.shape[0]
        full = np.tile(bg_mean[None, :, :], (n_samples, 1, 1)).copy()
        mask_r = mask_batch.reshape(n_samples, n_channels, N_GROUPS_PER_CHANNEL)
        present = mask_r > 0.5
        for c in range(n_channels):
            for g in range(N_GROUPS_PER_CHANNEL):
                lo, hi = g * GROUP_SIZE, (g + 1) * GROUP_SIZE
                rows = present[:, c, g]
                if rows.any():
                    full[rows, lo:hi, c] = x0[lo:hi, c]
        with torch.no_grad():
            out = model(torch.tensor(full, dtype=torch.float32))
        return out.squeeze(-1).numpy()

    background = np.zeros((1, n_groups_total))  # "all absent" reference coalition
    explainer = shap.KernelExplainer(predict, background)
    group_shap = explainer.shap_values(np.ones((1, n_groups_total)), nsamples=nsamples, silent=True)
    return np.array(group_shap).reshape(n_channels, N_GROUPS_PER_CHANNEL)


def shap_for_deep_model(name, model, X_test, X_background, n_channels, channel_names,
                         n_explain=5):
    model.eval()
    try:
        bg = torch.tensor(X_background[:30], dtype=torch.float32)
        test_sample = torch.tensor(X_test[:20], dtype=torch.float32)
        explainer = shap.DeepExplainer(model, bg)
        shap_values = explainer.shap_values(test_sample, check_additivity=False)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = np.array(shap_values).reshape(len(test_sample), SEQ_LEN, n_channels)
        print(f"[shap] {name}: DeepSHAP succeeded.")
        mean_abs = np.abs(shap_values).mean(axis=0)
        return "DeepSHAP", mean_abs
    except Exception as e:
        print(f"[shap] {name}: DeepSHAP failed ({type(e).__name__}: {e}). "
              f"Falling back to grouped KernelSHAP ({n_explain} test instances, "
              f"{N_GROUPS_PER_CHANNEL} bins x {n_channels} channels = "
              f"{N_GROUPS_PER_CHANNEL * n_channels} groups).")
        bg_mean = X_background.mean(axis=0)
        per_instance = []
        for i in range(min(n_explain, len(X_test))):
            g_shap = _grouped_kernelshap_one_instance(model, X_test[i], bg_mean, n_channels)
            per_instance.append(g_shap)
        group_shap = np.stack(per_instance, axis=0)  # (n_explain, n_channels, n_groups)
        shap_values = np.repeat(group_shap, GROUP_SIZE, axis=2).transpose(0, 2, 1)  # (n_explain, SEQ_LEN, n_channels)
        mean_abs = np.abs(shap_values).mean(axis=0)
        return "KernelSHAP(grouped)", mean_abs


def voltage_region_concentration(mean_abs_by_bin_channel, V_grid_typical, v_t_typical,
                                  channel_names, low=3.55, high=3.8):
    """
    channel 0 (V_t) is TIME-indexed; map its bins to voltage via v_t_typical.
    channels 3-5 (dQdV/dVdQ/dIdV) are VOLTAGE-indexed via V_grid_typical.
    Returns fraction of total |SHAP| mass (summed over these 4 voltage-
    interpretable channels) landing in [low, high] volts.
    """
    total_mass = 0.0
    in_region_mass = 0.0
    for ci, cname in enumerate(channel_names):
        col = mean_abs_by_bin_channel[:, ci]
        if cname == "V_t":
            voltages = v_t_typical
        elif cname in ("dQdV", "dVdQ", "dIdV"):
            voltages = V_grid_typical
        else:
            continue
        total_mass += col.sum()
        in_region_mass += col[(voltages >= low) & (voltages <= high)].sum()
    frac = in_region_mass / total_mass if total_mass > 0 else np.nan
    return frac


def main():
    from data_adapters import iterate_nasa_cycles
    from sequence_features import build_dataset_tensors, CHANNEL_NAMES, get_cycle_tensor
    from ica_dv_dc import compute_ica_dv_dc
    from models.vlstm import VLSTM
    from models.cnn_lstm import CNNLSTM
    from models.piformer import PiFormer

    print("\n[shap] === TreeSHAP: XGBoost base learner ===")
    tree_shap_xgboost()

    print("\n[shap] === TreeSHAP: Stacking-XGBoost meta-learner ===")
    tree_shap_meta()

    print("\n[shap] === Deep models: DeepSHAP (fallback KernelSHAP) ===")
    cycles = list(iterate_nasa_cycles("B0005"))
    X_all_raw, soh_all, rul_all, idxs, censored = build_dataset_tensors(cycles)
    X_all_raw = X_all_raw.astype(np.float32)
    n = len(X_all_raw)
    split = int(n * 0.7)

    # typical V_grid / V_t for the voltage-region mapping MUST come from
    # the RAW (real-volts) tensor, not the normalized one fed to the
    # models below - z-scored V_t values aren't volts anymore.
    ica_ref = compute_ica_dv_dc(cycles[split])
    V_grid_typical = ica_ref["V_grid"]
    v_t_typical = X_all_raw[split:, :, 0].mean(axis=0)  # avg voltage-vs-time-bin curve

    # models were trained on channel-normalized input (see
    # sequence_features.compute_channel_norm_stats) - apply the same saved
    # transform before feeding SHAP background/test data to them, or every
    # attribution below is meaningless (wrong input distribution entirely).
    from sequence_features import apply_channel_norm
    norm_stats = json.loads((PROC_DIR / "channel_norm_stats.json").read_text())
    X_all = apply_channel_norm(X_all_raw, norm_stats)
    X_background, X_test = X_all[:split], X_all[split:]

    methods_used = {}

    vlstm = VLSTM(input_size=1, hidden_size=32, n_targets=1)
    vlstm.load_state_dict(torch.load(ROOT / "models" / "vlstm_soh.pt"))
    method, mean_abs_v = shap_for_deep_model(
        "VLSTM", vlstm, X_test[:, :, 0:1], X_background[:, :, 0:1], 1, ["V_t"]
    )
    methods_used["VLSTM"] = method
    frac_v = voltage_region_concentration(mean_abs_v, V_grid_typical, v_t_typical, ["V_t"])
    print(f"[shap] VLSTM: fraction of |SHAP| mass in [3.55,3.8]V = {frac_v:.3f}")

    cnn_lstm = CNNLSTM()
    cnn_lstm.load_state_dict(torch.load(ROOT / "models" / "cnn_lstm_soh.pt"))
    method, mean_abs_c = shap_for_deep_model(
        "CNNLSTM", cnn_lstm, X_test, X_background, 6, CHANNEL_NAMES
    )
    methods_used["CNNLSTM"] = method
    frac_c = voltage_region_concentration(mean_abs_c, V_grid_typical, v_t_typical, CHANNEL_NAMES)
    print(f"[shap] CNNLSTM: fraction of |SHAP| mass in [3.55,3.8]V = {frac_c:.3f}")

    piformer = PiFormer()
    piformer.load_state_dict(torch.load(ROOT / "models" / "piformer_soh.pt"))
    method, mean_abs_p = shap_for_deep_model(
        "PiFormer", piformer, X_test, X_background, 6, CHANNEL_NAMES
    )
    methods_used["PiFormer"] = method
    frac_p = voltage_region_concentration(mean_abs_p, V_grid_typical, v_t_typical, CHANNEL_NAMES)
    print(f"[shap] PiFormer: fraction of |SHAP| mass in [3.55,3.8]V = {frac_p:.3f}")

    summary = pd.DataFrame([
        {"model": "VLSTM", "shap_method": methods_used["VLSTM"], "frac_mass_3.55_3.8V": frac_v},
        {"model": "CNNLSTM", "shap_method": methods_used["CNNLSTM"], "frac_mass_3.55_3.8V": frac_c},
        {"model": "PiFormer", "shap_method": methods_used["PiFormer"], "frac_mass_3.55_3.8V": frac_p},
    ])
    summary.to_csv(OUT_DIR / "shap_deep_models_summary.csv", index=False)
    print("\n[shap] === METHOD USED PER MODEL ===")
    print(summary.to_string(index=False))
    print("[shap] DONE")


if __name__ == "__main__":
    main()
