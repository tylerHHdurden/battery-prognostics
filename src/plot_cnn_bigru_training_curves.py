"""
Training convergence plot for CNN-BiGRU, matching the exact style of
make_plots.plot_deep_model_curves (which produces
outputs/phase2_deep_model_training_curves.png for VLSTM/CNN-LSTM/
PiFormer) - additive, does NOT modify make_plots.py or regenerate that
original 3-panel file. Saves a separate 4-panel figure (the original 3
plus CNN-BiGRU) so convergence speed can be compared side by side
without touching the existing output.
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


def main():
    models = [
        ("vlstm", "VLSTM"),
        ("cnn_lstm", "CNNLSTM"),
        ("piformer", "PiFormer"),
        ("cnn_bigru", "CNNBiGRU"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for ax, (fname, title) in zip(axes, models):
        hist = pd.read_csv(PRED_DIR / f"{fname}_history.csv")
        ax.plot(hist["epoch"], hist["train_loss"], label="train")
        ax.plot(hist["epoch"], hist["val_loss"], label="val")
        ax.set_title(f"{title} training curve")
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE (standardized SOH)")
        ax.legend()
    fig.tight_layout()
    out_path = OUT_DIR / "phase2_cnn_bigru_training_curves.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[plot] saved {out_path}")

    # convergence-speed summary: epoch of best val_loss for each model,
    # and best val_loss itself, printed for the DEVELOPMENT_LOG write-up
    print("\n[plot] convergence speed comparison (epoch of best val_loss):")
    for fname, title in models:
        hist = pd.read_csv(PRED_DIR / f"{fname}_history.csv")
        best_row = hist.loc[hist["val_loss"].idxmin()]
        print(f"    {title}: best val_loss={best_row['val_loss']:.4f} at epoch "
              f"{int(best_row['epoch'])} (of {len(hist)} epochs trained before early stop/budget)")


if __name__ == "__main__":
    main()
