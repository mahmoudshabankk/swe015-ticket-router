"""Draws results/architecture.png - the architecture diagram (matches app.py / src/)."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from config import RESULTS_DIR

C = {"ui": "#dbeafe", "B": "#fef3c7", "A": "#dcfce7", "C": "#ede9fe", "D": "#fce7f3", "out": "#e5e7eb"}


def box(ax, x, y, w, h, title, sub, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="#374151", lw=1.2))
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=10.5, weight="bold")
    ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=8.2, color="#374151")


def arrow(ax, x1, y1, x2, y2, label=""):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color="#374151", lw=1.3))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, label, ha="center", fontsize=7.5, color="#6b7280")


def main():
    fig, ax = plt.subplots(figsize=(15, 7.6))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 7.6)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.15, 0.15), 14.7, 6.6, boxstyle="round,pad=0.02", fc="none",
                                ec="#9ca3af", ls="--"))
    ax.text(0.35, 6.55, "Runs locally on CPU (AMD Ryzen 7 3700X) - Python 3.12, PyTorch + ONNX Runtime, Gradio server :7860",
            fontsize=9, color="#6b7280")

    box(ax, 0.4, 4.6, 2.1, 1.3, "Gradio UI", "user types message\n(browser)", C["ui"])
    box(ax, 3.0, 4.6, 2.4, 1.3, "Regex normalizer", "lowercase, URLs, e-mail,\nelongation, model codes", C["B"])
    box(ax, 5.9, 4.6, 2.4, 1.3, "Spelling corrector", "min. edit distance vs.\ntrain vocab + products", C["B"])
    box(ax, 8.8, 4.6, 2.4, 1.3, "ELIZA rules", "regex small-talk\npatterns -> reply", C["B"])
    box(ax, 11.7, 4.6, 2.9, 1.3, "BERT-small distilled", "INT8, ONNX Runtime\n(4 layers, 29 MB) - PRIMARY",
        C["D"])

    box(ax, 0.4, 1.0, 2.9, 1.5, "tf-idf + LogReg", "uni+bigram tf-idf\nfp64 - baseline", C["B"])
    box(ax, 3.7, 1.0, 2.9, 1.5, "FFN on avg. GloVe", "100-d GloVe -> 512 -> 512\n-> 151, SGD, fp32", C["A"])
    box(ax, 7.0, 1.0, 2.9, 1.5, "BERT-base teacher", "fine-tuned, 12 layers\nfp32", C["C"])
    box(ax, 11.7, 2.2, 2.9, 1.5, "Router / post-processing", "conf < tau or oos -> human\nintent -> domain -> team",
        C["out"])
    box(ax, 11.7, 0.35, 2.9, 1.3, "Output in UI", "team, intent, confidence,\nsteps, 4-model table", C["ui"])

    arrow(ax, 2.5, 5.25, 3.0, 5.25, "text")
    arrow(ax, 5.4, 5.25, 5.9, 5.25)
    arrow(ax, 8.3, 5.25, 8.8, 5.25)
    arrow(ax, 11.2, 5.25, 11.7, 5.25)
    ax.text(11.45, 4.35, "always runs", ha="center", fontsize=7.5, color="#6b7280")
    arrow(ax, 13.15, 4.6, 13.15, 3.7, "probs")
    arrow(ax, 13.15, 2.2, 13.15, 1.65)
    arrow(ax, 10.0, 4.6, 11.7, 1.2)
    ax.text(10.55, 3.0, "rule reply\n(small talk)", fontsize=7.5, color="#6b7280")
    for x in (1.85, 5.15, 8.45):
        arrow(ax, 7.1, 4.6, x, 2.5)
    arrow(ax, 9.9, 1.75, 11.7, 2.6)
    ax.text(10.1, 2.25, "all 3 models\n(comparison)", fontsize=7.5, color="#6b7280")
    ax.text(8.45, 0.65, "knowledge distillation (T=2) -> BERT-small -> ONNX -> INT8", ha="center", fontsize=8,
            color="#9d174d")
    ax.text(3.3, 3.4, "corrected text is also sent to the 3 comparison models", fontsize=8, color="#6b7280")

    legend = [("Area A: neural net", C["A"]), ("Area B: classical NLP", C["B"]),
              ("Area C: Transformer", C["C"]), ("Area D: efficient model", C["D"])]
    for k, (t, c) in enumerate(legend):
        ax.add_patch(FancyBboxPatch((0.4 + k * 2.6, 6.95), 0.35, 0.3, boxstyle="round,pad=0.01", fc=c, ec="#374151"))
        ax.text(0.85 + k * 2.6, 7.1, t, va="center", fontsize=9)
    fig.savefig(RESULTS_DIR / "architecture.png", dpi=160, bbox_inches="tight")
    print("saved", RESULTS_DIR / "architecture.png")


if __name__ == "__main__":
    main()
