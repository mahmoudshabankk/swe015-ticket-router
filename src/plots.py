"""Charts for the presentation, from results/results.csv (measured by evaluate.py)."""
import matplotlib.pyplot as plt
import pandas as pd

from config import RESULTS_DIR

COL = {"tf-idf + LogReg": "#d97706", "FFN on avg. GloVe": "#16a34a", "BERT-base (teacher)": "#7c3aed",
       "BERT-base INT8 (ONNX)": "#a78bfa", "BERT-small distilled": "#db2777",
       "BERT-small distilled INT8 (PyTorch)": "#f9a8d4", "BERT-small distilled INT8 (ONNX)": "#9d174d"}


def main():
    df = pd.read_csv(RESULTS_DIR / "results.csv")
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), sharey=True)
    names = df["model"].str.replace(" (teacher)", "").str.replace("distilled ", "dist. ").str.replace(
        "FFN on avg. GloVe", "FFN GloVe").str.replace("tf-idf + LogReg", "tf-idf LR").str.replace(
        "BERT-small", "BERT-sm")
    colors = [COL[m] for m in df["model"]]
    panels = [("accuracy", "Test accuracy (%)", 100, False), ("oos_recall", "OOS recall (%)", 100, False),
              ("latency_ms", "Latency per message (ms, log)", 1, True), ("size_mb", "Size on disk (MB, log)", 1, True)]
    for ax, (col, title, mul, log) in zip(axes, panels):
        vals = df[col] * mul
        ax.barh(names, vals, color=colors)
        for i, v in enumerate(vals):
            ax.text(v * (1.08 if log else 1) + (0 if log else 0.8), i, f"{v:.1f}" if v >= 1 else f"{v:.2f}",
                    va="center", fontsize=9)
        ax.set_title(title)
        if log:
            ax.set_xscale("log")
            ax.set_xlim(0.1, vals.max() * 4)
        else:
            ax.set_xlim(0, 110)
    axes[0].invert_yaxis()
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "results_chart.png", dpi=160)
    print("saved results_chart.png")


if __name__ == "__main__":
    main()
