"""
Generates and saves every summary plot referenced in OVERNIGHT_LOG.md /
STATUS reporting, per the task's "save every plot and metrics table as
files" instruction. Run after all 6 phases complete. All figures saved
to outputs/ as PNG.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
PRED_DIR = PROC_DIR / "predictions"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def plot_bfa_convergence():
    hist = pd.read_csv(PROC_DIR / "bfa_history.csv")
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(hist["iteration"], hist["best_fitness"], color="tab:blue", label="best fitness")
    ax1.set_xlabel("BFA iteration")
    ax1.set_ylabel("best fitness", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(hist["iteration"], hist["n_selected"], color="tab:orange", label="n_selected")
    ax2.set_ylabel("# features selected", color="tab:orange")
    plt.title("Binary Firefly Algorithm convergence (30 agents x 100 iterations)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase1_bfa_convergence.png", dpi=120)
    plt.close(fig)


def plot_capacity_fade_examples():
    df = pd.read_parquet(PROC_DIR / "hi_table.parquet")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, ds in zip(axes, ["NASA", "CALCE", "MIT"]):
        sub = df[df["dataset"] == ds]
        for bid in sub["battery_id"].unique()[:5]:
            b = sub[sub["battery_id"] == bid].sort_values("cycle_idx")
            ax.plot(b["cycle_idx"], b["SOH"], label=bid, linewidth=1)
        ax.axhline(80, color="red", linestyle="--", linewidth=0.8, label="80% EOL")
        ax.set_title(f"{ds}: SOH fade (sample cells)")
        ax.set_xlabel("cycle")
        ax.set_ylabel("SOH (%)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase1_soh_fade_examples.png", dpi=120)
    plt.close(fig)


def plot_ica_example():
    tensor = np.load(PROC_DIR / "differential_tensors" / "NASA_B0005.npy")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    labels = ["dQ/dV", "dV/dQ", "dI/dV"]
    for i, ax in enumerate(axes):
        for cyc_idx in [0, 50, 100, 150]:
            if cyc_idx < tensor.shape[0]:
                ax.plot(tensor[cyc_idx, :, i], label=f"cycle {cyc_idx+1}", linewidth=1)
        ax.set_title(f"NASA B0005: {labels[i]} vs voltage bin")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase1_ica_dv_dc_example.png", dpi=120)
    plt.close(fig)


def plot_deep_model_curves():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, name in zip(axes, ["vlstm", "cnn_lstm", "piformer"]):
        hist = pd.read_csv(PRED_DIR / f"{name}_history.csv")
        ax.plot(hist["epoch"], hist["train_loss"], label="train")
        ax.plot(hist["epoch"], hist["val_loss"], label="val")
        ax.set_title(f"{name.upper()} training curve")
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE (standardized SOH)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase2_deep_model_training_curves.png", dpi=120)
    plt.close(fig)


def plot_ensemble_comparison():
    comp = pd.read_csv(PRED_DIR / "ensemble_comparison.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["tab:red" if "Stacking" in m else "tab:blue" for m in comp["model"]]
    ax.barh(comp["model"], comp["rmse"], color=colors)
    ax.set_xlabel("RMSE (SOH %)")
    ax.set_title("Base learners vs. stacking ensemble (test set)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase3_ensemble_comparison.png", dpi=120)
    plt.close(fig)

    pred_df = pd.read_csv(PRED_DIR / "ensemble_test_preds.csv")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pred_df["SOH"], pred_df["pred_Stacking_Ridge"], s=4, alpha=0.3)
    lims = [pred_df["SOH"].min(), pred_df["SOH"].max()]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("True SOH (%)")
    ax.set_ylabel("Predicted SOH (%) [Stacking-Ridge]")
    ax.set_title("Stacking-Ridge: predicted vs true SOH (test)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase3_stacking_parity_plot.png", dpi=120)
    plt.close(fig)


def plot_joint_ablation():
    hist = pd.read_csv(PRED_DIR / "joint_ablation_history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for variant in hist["variant"].unique():
        sub = hist[hist["variant"] == variant]
        axes[0].plot(sub["epoch"], sub["val_loss_soh"], label=variant)
        axes[1].plot(sub["epoch"], sub["val_loss_rul"], label=variant)
    axes[0].set_title("Joint model: val SOH loss by variant")
    axes[1].set_title("Joint model: val RUL loss by variant")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE (standardized)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase4_joint_ablation_curves.png", dpi=120)
    plt.close(fig)

    ablation = pd.read_csv(PRED_DIR / "joint_ablation.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, target in zip(axes, ["SOH", "RUL"]):
        sub = ablation[ablation["target"] == target]
        ax.bar(sub["variant"], sub["rmse"])
        ax.set_title(f"Ablation: {target} RMSE by loss-weighting variant")
        ax.set_ylabel("RMSE")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase4_joint_ablation_bars.png", dpi=120)
    plt.close(fig)

    adaptive_hist = hist[hist["variant"] == "adaptive"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(adaptive_hist["epoch"], adaptive_hist["alpha"], label="alpha (SOH weight)")
    ax.plot(adaptive_hist["epoch"], adaptive_hist["beta"], label="beta (RUL weight)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("learned weight")
    ax.set_title("Adaptive loss weighting: alpha/beta evolution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase4_adaptive_alpha_beta.png", dpi=120)
    plt.close(fig)


def plot_shap_rankings():
    xgb_rank = pd.read_csv(OUT_DIR / "shap_xgboost_base_ranking.csv")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(xgb_rank["feature"], xgb_rank["mean_abs_shap"])
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("XGBoost base learner: TreeSHAP feature importance (BFA-selected HIs)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase5_shap_xgboost_ranking.png", dpi=120)
    plt.close(fig)

    meta_rank = pd.read_csv(OUT_DIR / "shap_meta_ranking.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(meta_rank["base_learner"], meta_rank["mean_abs_shap"])
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("Stacking-XGBoost meta-learner: TreeSHAP base-learner importance")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase5_shap_meta_ranking.png", dpi=120)
    plt.close(fig)

    deep_summary = pd.read_csv(OUT_DIR / "shap_deep_models_summary.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(deep_summary["model"], deep_summary["frac_mass_3.55_3.8V"])
    ax.set_ylabel("fraction of |SHAP| mass in [3.55, 3.8]V")
    ax.set_title("Deep models: SHAP mass concentration in voltage plateau region")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase5_shap_voltage_region.png", dpi=120)
    plt.close(fig)


def plot_conformal():
    soh = pd.read_csv(PRED_DIR / "conformal_soh_eval_preds.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    order = soh.sort_values("SOH_pred").reset_index(drop=True)
    x = np.arange(len(order))
    ax.fill_between(x, order["SOH_lo90"], order["SOH_hi90"], alpha=0.3, label="90% conformal interval")
    ax.scatter(x, order["SOH"], s=3, color="black", label="true SOH", alpha=0.5)
    ax.plot(x, order["SOH_pred"], color="red", linewidth=1, label="point prediction")
    ax.set_title("SOH split-conformal intervals (evaluation batteries, sorted by prediction)")
    ax.set_xlabel("sample (sorted)")
    ax.set_ylabel("SOH (%)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase6_conformal_soh.png", dpi=120)
    plt.close(fig)

    rul = pd.read_csv(PRED_DIR / "conformal_rul_eval_preds.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    order = rul.sort_values("RUL_pred").reset_index(drop=True)
    x = np.arange(len(order))
    ax.fill_between(x, order["RUL_lo90"], order["RUL_hi90"], alpha=0.3, color="tab:orange",
                     label="90% conformal interval")
    ax.scatter(x, order["RUL"], s=3, color="black", label="true RUL", alpha=0.5)
    ax.plot(x, order["RUL_pred"], color="red", linewidth=1, label="point prediction")
    ax.set_title("RUL split-conformal intervals (evaluation batteries, sorted by prediction)")
    ax.set_xlabel("sample (sorted)")
    ax.set_ylabel("RUL (cycles)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase6_conformal_rul.png", dpi=120)
    plt.close(fig)


def main():
    plot_bfa_convergence()
    print("[plots] BFA convergence saved")
    plot_capacity_fade_examples()
    print("[plots] SOH fade examples saved")
    plot_ica_example()
    print("[plots] ICA/DV/DC example saved")
    plot_deep_model_curves()
    print("[plots] deep model training curves saved")
    plot_ensemble_comparison()
    print("[plots] ensemble comparison saved")
    plot_joint_ablation()
    print("[plots] joint ablation plots saved")
    plot_shap_rankings()
    print("[plots] SHAP rankings saved")
    plot_conformal()
    print("[plots] conformal plots saved")
    print("[plots] ALL DONE")


if __name__ == "__main__":
    main()
