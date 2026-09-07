"""
Consolidated convergence comparison: training loss vs. epoch for ALL 4
deep models built in this project (VLSTM, CNN-LSTM, PiFormer,
CNN-BiGRU - session 17), overlaid on ONE chart rather than the
per-model side-by-side subplots make_plots.plot_deep_model_curves and
plot_cnn_bigru_training_curves.py already use, for direct visual
comparison of convergence speed/stability across architectures.

Additive, standalone script matching make_plots.py's exact style
(imports, directory constants, figure idioms, dpi=120, phaseN_*.png
naming) rather than editing that file - same convention this project's
other single-plot follow-up sessions already used (e.g.
plot_cnn_bigru_training_curves.py in session 17). Does not touch
make_plots.py or any existing output file.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
PRED_DIR = PROC_DIR / "predictions"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

MODELS = [
    ("vlstm", "VLSTM", "tab:blue"),
    ("cnn_lstm", "CNN-LSTM", "tab:orange"),
    ("piformer", "PiFormer", "tab:green"),
    ("cnn_bigru", "CNN-BiGRU", "tab:red"),
]


def main():
    fig, ax = plt.subplots(figsize=(9, 6))
    summary_rows = []
    for fname, label, color in MODELS:
        hist = pd.read_csv(PRED_DIR / f"{fname}_history.csv")
        ax.plot(hist["epoch"], hist["train_loss"], label=label, color=color, linewidth=1.5)
        best_row = hist.loc[hist["val_loss"].idxmin()]
        summary_rows.append({
            "model": label, "epochs_trained": len(hist),
            "best_epoch": int(best_row["epoch"]), "best_val_loss": float(best_row["val_loss"]),
            "final_train_loss": float(hist["train_loss"].iloc[-1]),
        })

    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss (MSE, standardized SOH)")
    ax.set_title("Deep model convergence comparison: training loss vs. epoch")
    ax.legend()
    fig.tight_layout()
    out_path = OUT_DIR / "phase8_convergence_comparison.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[plot] saved {out_path}")

    summary_df = pd.DataFrame(summary_rows).sort_values("best_epoch")
    print("\n[plot] convergence summary (sorted by epoch of best val_loss, i.e. convergence speed):")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
