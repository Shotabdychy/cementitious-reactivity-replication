"""Two summary figures for the README, drawn from the result files.

  figures/fig_summary_r2.png      published vs reproduced test R^2, per target
  figures/fig_rock_ablation.png   rock-screening count for the eight pipeline variants

Colours are the first slots of the documented reference categorical palette; every bar carries its
value so the figure does not rely on colour alone.

usage: python fig_summary.py
"""
import os
import re
import glob
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

INK, INK2, GRID = "#1b1b1b", "#55555a", "#e3e2de"
C_PAPER, C_FAITH, C_MOD, C_GROUP = "#6b6b70", "#eb6834", "#2a78d6", "#1baf7a"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": GRID, "axes.unicode_minus": False,
                     "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": "white", "savefig.facecolor": "white"})


def J(*p):
    with open(os.path.join(RES, *p), encoding="utf-8") as f:
        return json.load(f)


T = ("heat release", "CH consumption", "Bound water")
TITLE = {"heat release": "Heat release", "CH consumption": "Ca(OH)₂ consumption",
         "Bound water": "Bound water"}
PAPER = {"heat release": 0.93, "CH consumption": 0.85, "Bound water": 0.89}

faith = [J(os.path.basename(p)) for p in glob.glob(os.path.join(RES, "nb9_faithful_seed*.json"))
         if re.fullmatch(r"nb9_faithful_seed\d+\.json", os.path.basename(p))]
tuned, grouped = J("nb9_sound_tuned.json"), J("nb9_grouped_material.json")

# ------------------------------------------------------------------ figure 1
def draw_summary(path, figsize, k, dpi, titles):
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    for ax, t in zip(axes, T):
        f = np.array([r["test"][t]["R2"] for r in faith])
        e, g = tuned["ensemble_test"][t], grouped["ensemble_test"][t]
        bars = [("Paper", PAPER[t], None, C_PAPER),
                ("Released\ncode", f.mean(), (f.std(ddof=1), f.std(ddof=1)), C_FAITH),
                ("Modified\nprotocol", e["R2"], (e["R2"] - e["R2_lo"], e["R2_hi"] - e["R2"]), C_MOD),
                ("New\nmaterials", g["R2"], (g["R2"] - g["R2_lo"], g["R2_hi"] - g["R2"]), C_GROUP)]
        for i, (lab, v, err, col) in enumerate(bars):
            ax.bar(i, v, color=col, width=0.62, zorder=3)
            if err:
                ax.errorbar(i, v, yerr=[[err[0]], [err[1]]], color=INK, capsize=3, lw=1, zorder=4)
            top = v + (err[1] if err else 0)
            bottom = v - (err[0] if err else 0)
            y = top + 0.03 if v >= 0 else bottom - 0.03
            ax.text(i, y, f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=8.5 * k, fontweight="bold")
        ax.axhline(0, color=INK2, lw=0.8, zorder=2)
        if titles:
            ax.set_xticks(range(4))
            ax.set_xticklabels([b[0] for b in bars], fontsize=8.6 * k)
        else:
            ax.set_xticks([])          # compact version: a colour legend replaces the tick labels
        n = tuned["labels_test"][t]
        ttl = f"{TITLE[t]}  (test labels: {n})" if titles else f"{TITLE[t]} (n = {n})"
        ax.set_title(ttl, fontsize=9.5 * k, fontweight="bold")
        ax.set_ylim(-0.75, 1.12)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    axes[0].set_ylabel("test R²")
    if titles:
        fig.suptitle("Published vs reproduced accuracy (test R²)", fontsize=10.5 * k, fontweight="bold", y=0.995)
    if titles:
      fig.text(0.5, 0.012, "Released code = notebook 9 as published, mean of 11 runs (bar: SD).   "
             "Modified protocol = 11-seed ensemble with the fixes described in the report (bar: 95% bootstrap CI).\n"
             "New materials = same protocol, but no material is shared between train and test.",
             ha="center", fontsize=8 * k, color=INK2, linespacing=1.5)
    if not titles:
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(color=c, label=l.replace("\n", " ")) for l, _, _, c in bars],
                   loc="lower center", ncol=4, frameon=False, fontsize=8 * k, handlelength=1.2,
                   columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.09, 1, 0.95) if titles else (0, 0.09, 1, 1))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)



draw_summary(os.path.join(FIG, "fig_summary_r2.png"), (10.2, 3.6), 1.0, 200, True)
draw_summary(os.path.join(FIG, "fig_summary_report.png"), (6.3, 2.3), 0.85, 300, False)

# ------------------------------------------------------------------ figure 2
a = pd.read_csv(os.path.join(RES, "nb11_sound_tuned_ablation.csv"))
a = a[(a.model == "ensemble") & (a.row_filter == "all rows")].set_index("variant")
order = ["base", "d5", "fe", "d5+fe", "mix", "d5+mix", "fe+mix", "d5+fe+mix"]
label = {"base": "As released", "d5": "+ Dataset 5", "fe": "+ total iron", "d5+fe": "+ Dataset 5 + total iron",
         "mix": "+ R³ standard mix", "d5+mix": "+ Dataset 5 + R³ mix",
         "fe+mix": "+ total iron + R³ mix", "d5+fe+mix": "+ all three"}
vals = [int(a.loc[v, "reactive"]) for v in order]
fig, ax = plt.subplots(figsize=(7.4, 3.5))
y = np.arange(len(order))[::-1]
cols = [C_MOD if "mix" not in v else "#9db8e3" for v in order]
ax.barh(y, vals, color=cols, height=0.62, zorder=3)
ax.axvline(50569, color=C_FAITH, lw=1.6, ls=(0, (5, 3)), zorder=4)
ax.text(50569 + 9000, -0.62, "paper: 50,569", color=C_FAITH, fontsize=8.5, fontweight="bold", va="center")
ax.set_ylim(-0.95, len(order) - 0.45)
for yi, v in zip(y, vals):
    ax.text(v + 6000, yi, f"{v:,}  ({100 * (v - 50569) / 50569:+.0f}%)".replace("-", "−"),
            va="center", fontsize=8.3)
ax.set_yticks(y)
ax.set_yticklabels([label[v] for v in order], fontsize=8.5)
ax.set_xlim(0, 760000)
ax.set_xlabel("rock samples predicted reactive (> 200 J/g), of 1,022,092")
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
ax.set_title("Natural-rock screening: how the count depends on three pipeline choices\n"
             "(dark blue: notebook mix constants; light blue: R³ standard mix)", fontsize=9.2,
             fontweight="bold", loc="left")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_rock_ablation.png"), dpi=200)
plt.close(fig)
print("wrote figures/fig_summary_r2.png and figures/fig_rock_ablation.png")
