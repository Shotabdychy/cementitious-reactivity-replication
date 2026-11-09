"""Extraction-to-prediction workflow diagram for the final report (plain, print-friendly)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]

fig, ax = plt.subplots(figsize=(6.5, 1.9))
ax.set_xlim(0, 100)
ax.set_ylim(0, 30)
ax.axis("off")
boxes = [
    (2, "Table text\n(Dataset 2;\nnotebook-4\nprompt)", "#ffffff"),
    (22, "Oxide extraction\n(11 oxides;\nClaude Opus 5\nreplaces GPT-3.5)", "#f0f0f0"),
    (42, "Descriptor table\n(notebook-10\nconstants; LightGBM\nimputation)", "#ffffff"),
    (62, "Network ensemble\n(11 seeds; tuned\n512–256–32\nnetwork)", "#f0f0f0"),
    (82, "Predictions:\nheat release,\nCa(OH)₂ consumption,\nbound water", "#ffffff"),
]
w, h, y = 16, 20, 5
for x, txt, fc in boxes:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.2",
                                fc=fc, ec="#222222", lw=0.9))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=6.0, linespacing=1.25)
for x, _, _ in boxes[:-1]:
    ax.annotate("", xy=(x + w + 3.6, y + h / 2), xytext=(x + w + 0.6, y + h / 2),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#222222"))
ax.text(50, 1.0, "Material type (notebook-8 categories) sets the assumed d50 and silica-fume "
        "overrides in the descriptor table", ha="center", va="center", fontsize=6.0,
        style="italic", color="#333333")
out = os.path.join(ROOT, "figures", "fig_workflow.png")
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print("wrote", out)
