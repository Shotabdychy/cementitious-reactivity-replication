"""PART 3 -- figures, each paired with the paper's reported numbers.

The paper is CC BY-NC-ND, so its own figures are not redrawn or copied here. Every reproduced
figure carries the paper's corresponding values as an annotation or in a table beside it, and the
numbers behind each panel are also written to results/ as CSV.

Colours are the first three slots of the documented reference categorical palette
(blue #2a78d6, orange #eb6834, aqua #1baf7a), used unchanged; that three-slot subset is the one
published as clearing the all-pairs colour-vision gates in light mode, which is what scatter
panels need. (Node is not installed on this machine, so the palette validator could not be
re-run; no palette value was altered.) Every panel also ships its numbers as a CSV table, which
satisfies the relief rule for the low-contrast slot.

usage:
  python figures.py --models sound_tuned --parts parity,importance,shap,literature,rocks
"""
import os
import sys
import glob
import pickle
import argparse

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (TARGETS, FEATURES, SHORT, PAPER, RESULTS, FIGURES, ROOT, load_r3, make_split,
                    prepare, CUSTOM_OBJECTS, unstandardize, masked_metrics, jdump)

ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True)
ap.add_argument("--tag", default=None)
ap.add_argument("--nb10-tag", default=None)
ap.add_argument("--nb11-tag", default=None)
ap.add_argument("--parts", default="parity,importance,shap,literature,rocks")
args = ap.parse_args()
TAG = args.tag or args.models
NB10 = args.nb10_tag or TAG
NB11 = args.nb11_tag or TAG
PARTS = [p.strip() for p in args.parts.split(",")]
MODELDIR = os.path.join(ROOT, "models", args.models)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e3e2de"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 9, "axes.titlesize": 10, "axes.linewidth": 0.8,
    "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 150,
    "font.family": "sans-serif",
})


def tidy(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


UNITS = {'heat release': 'J/g', 'CH consumption': 'g/100 g', 'Bound water': 'g/100 g'}

# ---------------------------------------------------------------- 3.1 parity plots
if "parity" in PARTS:
    p = os.path.join(RESULTS, f"nb9_{TAG}_predictions.csv")
    df = pd.read_csv(p)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))
    tbl = []
    for ax, t in zip(axes, TARGETS):
        s = SHORT[t]
        sub = df[df[f'y_{s}'].notna()]
        for split, col, lab in (("train", C1, "train"), ("test", C2, "test")):
            d = sub[sub.split == split]
            ax.scatter(d[f'y_{s}'], d[f'pred_{s}'], s=22, c=col, alpha=0.55,
                       linewidths=0.5, edgecolors=SURFACE, label=lab, zorder=3)
            m = masked_metrics(d[f'y_{s}'].values, d[f'pred_{s}'].values)
            tbl.append(dict(target=t, split=split, n=m["n"], R2=m["R2"], RMSE=m["RMSE"]))
        lo = float(min(sub[f'y_{s}'].min(), sub[f'pred_{s}'].min()))
        hi = float(max(sub[f'y_{s}'].max(), sub[f'pred_{s}'].max()))
        pad = 0.05 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=INK2, lw=1,
                ls=(0, (4, 3)), zorder=2)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        tr = [r for r in tbl if r["target"] == t and r["split"] == "train"][0]
        te = [r for r in tbl if r["target"] == t and r["split"] == "test"][0]
        ax.set_title(f"{t} ({UNITS[t]})")
        ax.set_xlabel("measured")
        ax.set_ylabel("predicted (ensemble)")
        ax.text(0.03, 0.97,
                f"train  R$^2$ {tr['R2']:.3f}   RMSE {tr['RMSE']:.2f}   n={tr['n']}\n"
                f"test   R$^2$ {te['R2']:.3f}   RMSE {te['RMSE']:.2f}   n={te['n']}\n"
                f"paper  R$^2$ {PAPER[t]['R2']:.2f}   RMSE {PAPER[t]['RMSE']:.2f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=7.5, color=INK2,
                linespacing=1.5)
        tidy(ax)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"Figure 2a-c reproduced: parity of the {TAG} ensemble, with the paper's "
                 f"reported values", fontsize=10, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(FIGURES, f"fig2_parity_{TAG}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(tbl).to_csv(os.path.join(RESULTS, f"fig2_parity_{TAG}.csv"), index=False)
    print(f"wrote {out}")

# ---------------------------------------------------------------- shared model/data for 3.2
if "importance" in PARTS or "shap" in PARTS:
    bundle = pickle.load(open(os.path.join(MODELDIR, "bundle.pkl"), "rb"))
    r3 = load_r3()
    tr, te, _ = make_split(r3, grouped=None if bundle["grouped"] == "none" else bundle["grouped"])
    X, mask, y_std, mu, sd, _ = prepare(r3, tr, log=lambda m: None, impute=bundle["impute"])
    paths = sorted(glob.glob(os.path.join(MODELDIR, "seed*.keras")))
    models = [tf.keras.models.load_model(p, custom_objects=CUSTOM_OBJECTS) for p in paths]
    print(f"loaded {len(models)} models for importance/SHAP")

    def predict_ens(Xa, Ma):
        acc = None
        for m in models:
            o = m([tf.convert_to_tensor(Xa), tf.convert_to_tensor(Ma)], training=False)
            o = np.stack([np.asarray(a).flatten() for a in o])
            acc = o if acc is None else acc + o
        return acc / len(models)

# ---------------------------------------------------------------- 3.2a permutation importance
if "importance" in PARTS:
    rng = np.random.default_rng(0)
    base = predict_ens(X[te], mask[te])
    base_mse = {}
    for i, t in enumerate(TARGETS):
        yv = y_std[t].values[te]
        ok = ~np.isnan(yv)
        base_mse[t] = float(np.mean((yv[ok] - base[i][ok]) ** 2))
    rows = []
    REP = 10
    for j, f in enumerate(FEATURES):
        inc = {t: [] for t in TARGETS}
        for r in range(REP):
            Xp, Mp = X[te].copy(), mask[te].copy()
            perm = rng.permutation(len(te))
            Xp[:, j] = Xp[perm, j]
            Mp[:, j] = Mp[perm, j]          # descriptor and its missingness flag move together
            p = predict_ens(Xp, Mp)
            for i, t in enumerate(TARGETS):
                yv = y_std[t].values[te]
                ok = ~np.isnan(yv)
                inc[t].append(float(np.mean((yv[ok] - p[i][ok]) ** 2)) - base_mse[t])
        for t in TARGETS:
            rows.append(dict(feature=f, target=t, delta_mse=float(np.mean(inc[t])),
                             delta_mse_sd=float(np.std(inc[t], ddof=1)),
                             rel_increase=float(np.mean(inc[t]) / base_mse[t])))
        print(f"  permuted {f}")
    imp = pd.DataFrame(rows)
    imp.to_csv(os.path.join(RESULTS, f"fig2_permutation_importance_{TAG}.csv"), index=False)

    PAPER_TOP = ['age', 'Al2O3', 'CaCO3', 'Fe2O3', 'S gravity', 'SiO2', 'CaO']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    overlap = {}
    for ax, t in zip(axes, TARGETS):
        d = imp[imp.target == t].nlargest(10, "delta_mse").iloc[::-1]
        ax.barh(d.feature, d.delta_mse, color=C1, height=0.68, zorder=3)
        ax.set_title(f"{t}")
        ax.set_xlabel("increase in masked MSE (standardized)")
        tidy(ax)
        ov = [f for f in d.feature if f in PAPER_TOP]
        overlap[t] = ov
        ax.text(0.98, 0.04, f"of the paper's named top descriptors,\n{len(ov)} of "
                            f"{len(PAPER_TOP)} appear here:\n{', '.join(ov) if ov else 'none'}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=INK2,
                linespacing=1.4)
    fig.suptitle("Figure 2d-f reproduced: permutation importance on the test rows, top 10 per "
                 "target (descriptor and its missingness flag permuted together)", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(FIGURES, f"fig2_importance_{TAG}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    jdump(dict(paper_named_top_descriptors=PAPER_TOP, overlap_with_my_top10=overlap),
          os.path.join(RESULTS, f"fig2_importance_overlap_{TAG}.json"))
    print(f"wrote {out}")

# ---------------------------------------------------------------- 3.2b SHAP
if "shap" in PARTS:
    import shap
    from tensorflow.keras.layers import Input, Lambda
    from tensorflow.keras.models import Model
    n = len(FEATURES)
    XM = np.hstack([X, mask]).astype('float32')
    bg = XM[np.random.default_rng(0).choice(tr, size=min(100, len(tr)), replace=False)]
    N_EXPLAIN, N_BG, NSAMPLES, N_MEMBERS = 60, 40, 200, 3
    sample_idx = te[:N_EXPLAIN]
    # shap.GradientExplainer refuses this network ("the model output to be explained must be a
    # single tensor"), so the model-agnostic KernelExplainer is used instead and said so in the
    # caption. Results are cached: re-rendering the figure does not recompute them.
    cache = os.path.join(RESULTS, f"fig2_shap_values_{TAG}.npz")
    EXPLAINER = f"KernelExplainer, {N_BG} background rows, nsamples={NSAMPLES}"
    if os.path.exists(cache):
        z = np.load(cache)
        sv = {t: z[SHORT[t]] for t in TARGETS}
        sample_idx = z["sample_idx"]
        print(f"  SHAP values loaded from {cache}")
    else:
        sv = {}
        for i, t in enumerate(TARGETS):
            outs = []
            for m in models[:N_MEMBERS]:
                inp = Input(shape=(2 * n,))
                f = Lambda(lambda z: z[:, :n])(inp)
                mk = Lambda(lambda z: z[:, n:])(inp)
                w = Model(inp, m([f, mk])[i])
                ex = shap.KernelExplainer(lambda z: w.predict(z, verbose=0).flatten(),
                                          shap.sample(bg, N_BG))
                vals = ex.shap_values(XM[sample_idx], nsamples=NSAMPLES, silent=True)
                vals = vals[0] if isinstance(vals, list) else vals
                outs.append(np.asarray(vals).reshape(len(vals), -1))
            sv[t] = np.mean(outs, axis=0)
            print(f"  SHAP done for {t}: {sv[t].shape}")
        np.savez(cache, sample_idx=sample_idx, **{SHORT[t]: sv[t] for t in TARGETS})

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6))
    rows = []
    for ax, t in zip(axes, TARGETS):
        v = sv[t][:, :n]                       # attribution of the value channels
        order = np.argsort(np.abs(v).mean(axis=0))[::-1][:10][::-1]
        xs = XM[sample_idx][:len(v), :n]
        for k, j in enumerate(order):
            vals = v[:, j]
            col = xs[:, j]
            rngc = np.ptp(col)
            cn = (col - col.min()) / rngc if rngc > 0 else np.zeros_like(col)
            jit = (np.random.default_rng(j).random(len(vals)) - 0.5) * 0.34
            ax.scatter(vals, np.full(len(vals), k) + jit, c=cn, cmap=
                       matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ),
                       s=12, alpha=0.75, linewidths=0, zorder=3)
            rows.append(dict(target=t, feature=FEATURES[j],
                             mean_abs_shap=float(np.abs(vals).mean())))
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([FEATURES[j] for j in order])
        ax.axvline(0, color=INK2, lw=0.8, zorder=2)
        ax.set_title(t)
        ax.set_xlabel("SHAP value (standardized target)")
        tidy(ax)
    sm = plt.cm.ScalarMappable(cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "seq", SEQ), norm=matplotlib.colors.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes, fraction=0.016, pad=0.015)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["low", "high"])
    cb.set_label("descriptor value", fontsize=8)
    cb.outline.set_visible(False)
    fig.suptitle(f"Figure 2g-i reproduced: SHAP summary, top 10 descriptors per target "
                 f"(mean over {N_MEMBERS} ensemble members; {EXPLAINER}; "
                 f"{len(sample_idx)} test rows)", fontsize=9.5)
    out = os.path.join(FIGURES, f"fig2_shap_{TAG}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).groupby(["target", "feature"], as_index=False).mean().to_csv(
        os.path.join(RESULTS, f"fig2_shap_mean_abs_{TAG}.csv"), index=False)
    print(f"wrote {out}")

# ---------------------------------------------------------------- 3.3 literature materials
if "literature" in PARTS:
    p = os.path.join(RESULTS, f"nb10_{NB10}_predictions.csv")
    df = pd.read_csv(p)
    top = df.Category.value_counts().head(8).index.tolist()
    xlo, xhi = np.percentile(df['heat release'], [0.2, 99.8])
    ylo, yhi = np.percentile(df['CH consumption'], [0.2, 99.8])
    xlo, ylo = min(xlo, 0), min(ylo, 0)
    off = int(((df['heat release'] < xlo) | (df['heat release'] > xhi) |
               (df['CH consumption'] < ylo) | (df['CH consumption'] > yhi)).sum())
    fig, axes = plt.subplots(3, 3, figsize=(11, 10), sharex=True, sharey=True)
    axl = axes.ravel()
    rows = []
    for ax, cat in zip(axl, top + ["all other categories"]):
        d = df if cat == "all other categories" else df[df.Category == cat]
        if cat == "all other categories":
            d = df[~df.Category.isin(top)]
        ax.scatter(df['heat release'], df['CH consumption'], s=5, c="#d8d7d2", linewidths=0,
                   zorder=2)
        ax.scatter(d['heat release'], d['CH consumption'], s=14, c=C1, alpha=0.6, linewidths=0,
                   zorder=4)
        ax.axvline(120, color=C2, lw=1.2, ls=(0, (4, 3)), zorder=3)
        ax.axhline(50, color=C3, lw=1.2, ls=(0, (4, 3)), zorder=3)
        ax.set_title(f"{cat}  (n={len(d):,})", fontsize=9)
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        tidy(ax)
        rows.append(dict(category=cat, n=len(d),
                         heat_median=float(d['heat release'].median()),
                         CH_median=float(d['CH consumption'].median()),
                         pct_heat_above_120=float((d['heat release'] > 120).mean() * 100),
                         pct_CH_above_50=float((d['CH consumption'] > 50).mean() * 100)))
    for ax in axes[-1]:
        ax.set_xlabel("predicted heat release (J/g)")
    for ax in axes[:, 0]:
        ax.set_ylabel("predicted Ca(OH)$_2$ consumption (g/100 g)")
    fig.suptitle(f"Figure 3a/3b reproduced: the {len(df):,} literature-extracted materials, each "
                 f"panel highlighting one category (blue) against all of them (grey).\n"
                 f"Dashed lines: 120 J/g (orange, inert threshold) and 50 g/100 g (aqua, "
                 f"pozzolanic threshold). Axes clipped to the 0.2-99.8 percentile range; "
                 f"{off:,} of {len(df):,} points fall outside and are not drawn.", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(FIGURES, f"fig3_literature_{NB10}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, f"fig3_literature_{NB10}.csv"), index=False)
    print(f"wrote {out}")

# ---------------------------------------------------------------- 3.4 rock types
if "rocks" in PARTS:
    p = os.path.join(RESULTS, f"nb11_{NB11}_by_rock_name.csv")
    g = pd.read_csv(p)
    PAPER_NAMED = ['anorthosite', 'ignimbrite', 'porphyry', 'tuff', 'rhyolite', 'andesite',
                   'dacite', 'basalt', 'granite', 'shale']
    PAPER_RATIO = {'anorthosite': 0.25, 'ignimbrite': 0.25, 'porphyry': 0.22, 'tuff': 0.21,
                   'rhyolite': 0.12, 'andesite': 0.12, 'dacite': 0.12}
    d = g[g.rock_name.isin(PAPER_NAMED)].set_index('rock_name').reindex(PAPER_NAMED).dropna(
        subset=['n']).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0), sharey=True,
                             gridspec_kw=dict(width_ratios=[1, 1], wspace=0.08))
    yy = np.arange(len(d))[::-1]
    axes[0].barh(yy, d.ratio, color=C1, height=0.6, zorder=3, label="reproduced")
    pr = [PAPER_RATIO.get(nm, np.nan) for nm in d.rock_name]
    axes[0].scatter(pr, yy, marker="D", s=34, c=C2, zorder=5,
                    label="paper (reported, where given)")
    axes[0].set_yticks(yy)
    axes[0].set_yticklabels(d.rock_name)
    hi = max(float(d.ratio.max()), max(PAPER_RATIO.values()))
    axes[0].set_xlim(0, hi * 2.05)          # room at the right for the legend
    axes[0].set_xlabel("reactive samples / total samples")
    axes[0].set_title("reactive-to-total ratio", fontsize=9.5)
    tidy(axes[0])
    axes[1].barh(yy, d.n, color=C3, height=0.6, zorder=3)
    axes[1].set_xscale("log")
    axes[1].set_xlim(max(1, d.n.min() * 0.5), d.n.max() * 3.2)
    axes[1].set_xlabel("total samples in the database (log scale)")
    axes[1].set_title("samples per rock type", fontsize=9.5)
    axes[1].tick_params(axis="y", length=0)
    for y, v in zip(yy, d.n):
        axes[1].text(v * 1.15, y, f"{int(v):,}", va="center", fontsize=7.5, color=INK2)
    tidy(axes[1])
    axes[0].legend(loc="center right", fontsize=8)
    fig.suptitle("Figure 4a reproduced: rock types named in the paper, using the RAW rock_name "
                 "values of the Gard et al. database.\nThe paper's own name grouping is not "
                 "documented, so no grouping was applied; the two sets of names need not match.",
                 fontsize=9.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    out = os.path.join(FIGURES, f"fig4_rocks_{NB11}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    d.assign(paper_ratio=pr).to_csv(os.path.join(RESULTS, f"fig4_rocks_{NB11}.csv"), index=False)
    print(f"wrote {out}")

print("figures done")
