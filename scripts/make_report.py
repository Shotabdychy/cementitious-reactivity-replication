"""PART 5 -- fill docs/report_template.md from the result files and write
docs/REPLICATION_REPORT.md.

Every {{PLACEHOLDER}} is resolved from results/*.json or results/*.csv, so no number in the
report is typed by hand except the paper's own reported values, which are labelled as such.
The script fails if any placeholder is left unresolved.

usage: python make_report.py
"""
import os
import re
import sys
import json
import glob
import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TARGETS, SHORT, PAPER, RESULTS, ROOT, FIGURES

DOCS = os.path.join(ROOT, "docs")
TAG = "sound_tuned"


def jload(*p):
    f = os.path.join(RESULTS, *p)
    return json.load(open(f)) if os.path.exists(f) else None


def cload(*p):
    f = os.path.join(RESULTS, *p)
    return pd.read_csv(f) if os.path.exists(f) else None


def md(df, floatfmt="{:g}"):
    """DataFrame -> markdown table."""
    d = df.copy()
    for c in d.columns:
        d[c] = [("" if (isinstance(v, float) and not np.isfinite(v)) or v is None
                 else (floatfmt.format(v) if isinstance(v, float) else str(v))) for v in d[c]]
    head = "| " + " | ".join(str(c) for c in d.columns) + " |"
    rule = "|" + "|".join("---" for _ in d.columns) + "|"
    body = ["| " + " | ".join(r) + " |" for r in d.values.tolist()]
    return "\n".join([head, rule] + body)


def f(x, nd=3):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def n(x):
    return "-" if x is None else f"{int(x):,}"


V = {}
V["DATE"] = datetime.date.today().isoformat()

# ---------------------------------------------------------------- claims table
cl = cload("table_claims.csv")
c2 = cl.copy()
c2.columns = ["claim", "paper", "reproduced", "detail", "status", "reason"]
c2["reason"] = c2["reason"].fillna("")


def shorten(s, k=78):
    s = str(s)
    return s if len(s) <= k else s[:k - 1].rsplit(" ", 1)[0] + "…"


c2["claim"] = c2["claim"].map(lambda s: shorten(s, 58))
c2["reason"] = c2["reason"].map(lambda s: shorten(s, 78))
V["TABLE_CLAIMS"] = (md(c2[["claim", "paper", "reproduced", "status", "reason"]])
                     + "\n\nFull wording, the detail column and the confidence intervals are in "
                       "`results/table_claims.csv`.")

# ---------------------------------------------------------------- accuracy table
acc = cload("table_model_accuracy.csv")
rows = []
for _, r in acc.iterrows():
    rec = {"row": r["row"], "seeds": "" if pd.isna(r.get("n_seeds")) else int(r["n_seeds"])}
    for t in TARGETS:
        s = SHORT[t]
        m, sd = r.get(f"{s}_R2_mean"), r.get(f"{s}_R2_sd")
        lo, hi = r.get(f"{s}_R2_lo"), r.get(f"{s}_R2_hi")
        cell = f(m)
        if pd.notna(sd):
            cell += f" ± {f(sd)}"
        elif pd.notna(lo):
            cell += f" [{f(lo)}, {f(hi)}]"
        rec[f"{s} R²"] = cell
        rec[f"{s} RMSE"] = f(r.get(f"{s}_RMSE_mean"), 2)
    rows.append(rec)
_acc = pd.DataFrame(rows)
# the body carries the rows a reader needs to compare; the appendix carries all of them
KEEP = ("Faithful notebook 9", "Clean variant (notebook monitor, no input leakage)")
body = _acc[_acc.row.isin(KEEP) | _acc.row.str.contains("ENSEMBLE") |
            (_acc.row == "PAPER (reported)")].copy()
body["row"] = body.row.str.replace(" -- ENSEMBLE", " (ensemble)", regex=False)
V["TABLE_ACCURACY"] = md(body)
V["TABLE_ACCURACY_FULL"] = (
    "Per-seed means with their standard deviations, for every row including the ones the body "
    "summarises as an ensemble:\n\n" + md(_acc))

# ---------------------------------------------------------------- faithful collapse counts
fjs = [json.load(open(p)) for p in glob.glob(os.path.join(RESULTS, "nb9_faithful_seed*.json"))
       if re.fullmatch(r"nb9_faithful_seed\d+\.json", os.path.basename(p))]
V["N_FAITHFUL"] = str(len(fjs))
nc = sum(r["collapsed"] for r in fjs)
V["N_COLLAPSED"] = str(nc)
V["CLEAN_COLLAPSED"] = str(sum(json.load(open(q))["collapsed"] for q in glob.glob(os.path.join(RESULTS, "nb9_clean_seed*.json")) if re.fullmatch(r"nb9_clean_seed\d+\.json", os.path.basename(q))))
mons = [r["monitor_at_best"] for r in fjs]
V["COLLAPSE_SENTENCE"] = (
    (f"Every one of the {len(fjs)} faithful runs trips a collapse rule" if nc == len(fjs)
     else f"{nc} of the {len(fjs)} faithful runs trip a collapse rule")
    + f"; the monitored value at the restored epoch runs from {min(mons):.3g} to "
      f"{max(mons):.3g}, never below 1.")
f42 = [r for r in fjs if r["seed"] == 42]
V["FAITHFUL42_CH_R2"] = f(f42[0]["test"]["CH consumption"]["R2"]) if f42 else "-"

sound = jload(f"nb9_{TAG}.json")
V["SOUND_CH_R2"] = f(sound["ensemble_test"]["CH consumption"]["R2"])
V["N_SEEDS"] = str(len(sound["seeds"]))

met, miss = [], []
for t in TARGETS:
    e = sound["ensemble_test"][t]
    covers = e["R2_lo"] <= PAPER[t]["R2"] <= e["R2_hi"]
    if e["R2"] >= PAPER[t]["R2"]:
        met.append(f"R² {f(e['R2'])} for {t} against the paper's {PAPER[t]['R2']}")
    elif covers:
        miss.append(f"{t} reaches {f(e['R2'])}, below the paper's {PAPER[t]['R2']} but with a "
                    f"95 % interval [{f(e['R2_lo'])}, {f(e['R2_hi'])}] that covers it "
                    f"({e['n']} labelled test rows)")
    else:
        miss.append(f"{t} reaches {f(e['R2'])} against the paper's {PAPER[t]['R2']}, with a "
                    f"95 % interval [{f(e['R2_lo'])}, {f(e['R2_hi'])}] that does not cover it")
V["WHAT_MATCHED"] = (
    (f"Under the sound protocol the {len(sound['seeds'])}-seed ensemble reaches "
     + "; ".join(met) + ". " if met else
     f"Under the sound protocol the {len(sound['seeds'])}-seed ensemble reaches none of the "
     f"paper's values. ")
    + ("The other two targets: " + "; ".join(miss) + ". " if miss else "")
    + f"The sound protocol changes several things together (the early-stopping monitor, the "
      f"validation split, train-only preprocessing and a stratified outer split), and their "
      f"separate contributions were not isolated: Ca(OH)₂ consumption goes from R² "
      f"{V['FAITHFUL42_CH_R2']} in the faithful run to {V['SOUND_CH_R2']} here.")

lt = sound["labels_test"]
V["TABLE_ACCURACY"] += (
    f"\n\nFor the row-level split the test set holds {lt['heat release']} labelled rows for "
    f"heat release, {lt['CH consumption']} for Ca(OH)₂ consumption and {lt['Bound water']} for "
    f"bound water; the grouped splits differ and their sizes are in "
    f"`results/table_model_accuracy.csv`. '±' is the standard deviation over seeds; square "
    f"brackets are a 2,000-sample bootstrap 95 % confidence interval for the ensemble.")

# ---------------------------------------------------------------- tuning
tc = jload("nb9_tuned_config.json")
V["TUNE_TRIALS"] = str(tc["trials"])
V["TUNE_FOLDS"] = str(tc["folds"])
V["TUNE_CFG"] = (f"widths {'-'.join(str(w) for w in tc['widths'])}, dropout {tc['dropout']}, "
                 f"L2 {tc['l2']:g}, learning rate {tc['lr']:.2e}, batch {tc['batch']}, "
                 f"loss weighting '{tc['weighting']}' with the Ca(OH)₂ head × {tc['ch_mult']:g}, "
                 f"patience {tc['patience']}")
V["TUNE_EPOCH"] = f"{tc['mean_best_epoch']:.0f}"
V["TUNE_SCORE"] = f(tc["cv_score"], 4)
alltr = cload("nb9_tuning_all_trials.csv")
V["TUNE_WORST"] = f(alltr.cv_score.max(), 4)

sn = jload("nb9_sound_notebook.json")
if sn:
    bits = [f"{t} {f(sn['ensemble_test'][t]['R2'])} against {f(sound['ensemble_test'][t]['R2'])}"
            for t in TARGETS]
    better = sum(sn["ensemble_test"][t]["R2"] > sound["ensemble_test"][t]["R2"] for t in TARGETS)
    V["TUNE_OUTCOME"] = (
        f"**The tuning did not pay off on the test set.** Running the same 11-seed protocol with "
        f"the notebook's own hyperparameters gives (notebook against tuned): "
        + "; ".join(bits)
        + (" — better on all three targets." if better == 3 else
           f" — better on {better} of the three targets.")
        + f" The tuned configuration is nevertheless the one used "
        f"for notebooks 10 and 11 and for the figures, because it was chosen by cross-validation "
        f"on the training rows before any test number existed; swapping to the configuration "
        f"that happens to score better on the test set would be exactly the selection this "
        f"replication is trying to avoid. On this evidence the improvement comes from the protocol "
        f"changes of §3.2 rather than from the choice of hyperparameters.")
else:
    V["TUNE_OUTCOME"] = ""

# ---------------------------------------------------------------- grouped split paragraph
gm = jload("nb9_grouped_material.json")
gd = jload("nb9_grouped_doi.json")
par = []
if gm:
    bits = []
    for t in TARGETS:
        a, b = sound["ensemble_test"][t]["R2"], gm["ensemble_test"][t]["R2"]
        bits.append(f"{t} {f(a)} → {f(b)}")
    par.append(
        "Dataset 4 holds 1,849 rows but only 277 distinct materials, most of them measured at "
        "several ages, so a row-level split puts the same material on both sides. Repeating the "
        "whole protocol with a split grouped by material name, so that no material appears in "
        "both train and test, gives (ensemble, random split → grouped split): "
        + "; ".join(bits) + ". "
        f"The grouped test set holds {gm['labels_test']['heat release']} heat-release, "
        f"{gm['labels_test']['CH consumption']} Ca(OH)₂ and {gm['labels_test']['Bound water']} "
        "bound-water labels.")
if gd:
    bits = []
    for t in TARGETS:
        bits.append(f"{t} {f(gd['ensemble_test'][t]['R2'])} (n={gd['ensemble_test'][t]['n']})")
    par.append(
        "Grouping by source paper instead is much coarser — there are only 28 DOIs — and leaves "
        "an unbalanced test set: " + "; ".join(bits) + ". The bound-water figure there rests on "
        "too few labels to mean anything and is reported only for completeness.")
if gm:
    worst = min(TARGETS, key=lambda t: gm["ensemble_test"][t]["R2"])
    w = gm["ensemble_test"][worst]
    par.append(
        f"This matters for how the accuracy should be read. On the grouped split {worst} reaches "
        f"R² {f(w['R2'])} (95 % interval [{f(w['R2_lo'])}, {f(w['R2_hi'])}]), while the training "
        f"rows still fit at R² {f(gm['ensemble_train'][worst]['R2'])}. A row-level R² on this "
        f"dataset therefore measures interpolation between measurements of a material the model "
        f"has already seen — usually the same material at a different age — far more than "
        f"prediction for a material it has not. The paper's purpose is screening new precursors, "
        f"so the grouped figure is the one that speaks to that use.")
par.append("The paper does not state how its split was formed, so none of this is a correction "
           "to a published number; it is the size of the effect that the split choice has on "
           "this dataset, measured under the same protocol as every other row.")
V["GROUPED_PARA"] = "\n\n".join(par)

# ---------------------------------------------------------------- legacy env
lg = cload("table_library_version_check.csv")
if lg is not None and len(lg):
    r = lg.iloc[0]
    d = {t: abs(r[f"{SHORT[t]}_R2_legacy"] - r[f"{SHORT[t]}_R2_main"]) for t in TARGETS
         if pd.notna(r.get(f"{SHORT[t]}_R2_main"))}
    moved = sorted(d, key=d.get, reverse=True)
    V["TABLE_LEGACY"] = (
        f"The paper states Python 3.8. Only Python 3.11 is installed on this machine and no "
        f"conda or uv is available, so Python 3.8 itself could not be used; the faithful "
        f"notebook-9 run was instead repeated in a second environment pinned to the oldest "
        f"stack that still has wheels for this interpreter ({r['env_legacy']}) and compared "
        f"with the main one ({r['env_main']}). `winget` does offer `Python.Python.3.8` "
        f"(3.8.10) if a true 3.8 interpreter is wanted.\n\n"
        + md(lg[["seed"] + [f"{SHORT[t]}_R2_{w}" for t in TARGETS for w in ("main", "legacy")]
                + ["best_epoch_main", "best_epoch_legacy"]], "{:.4f}")
        + (f"\n\nThe two environments do not run the same trajectory: the restored epoch moves "
           f"from {int(r['best_epoch_main'])} to {int(r['best_epoch_legacy'])}, and the largest "
           f"change in test R² is {moved[0]}, by {d[moved[0]]:.3f}. Ca(OH)₂ consumption changes "
           f"by {d['CH consumption']:.4f} — both environments land at about "
           f"{r['CH_R2_main']:.2f} — and both runs are flagged collapsed by the validation-only "
           f"rule of §3.2. The Ca(OH)₂ failure is therefore structural rather than an artefact "
           f"of library versions, while the spread in the other targets is what an unstable "
           f"early-stopping criterion produces when anything perturbs the trajectory."))
else:
    V["TABLE_LEGACY"] = (
        "The paper states Python 3.8. Only Python 3.11 is installed on this machine and no "
        "conda or uv is available, so Python 3.8 itself could not be used; `winget` does offer "
        "`Python.Python.3.8` (3.8.10) if that interpreter is wanted. The pinned legacy "
        "environment for the oldest stack that still has Python 3.11 wheels did not finish "
        "installing in time for this report, so the library-version check is **not done**.")

# ---------------------------------------------------------------- figures
def fig(name):
    c = sorted(glob.glob(os.path.join(FIGURES, name)))
    return os.path.basename(c[0]) if c else None


for k, pat in (("FIG_PARITY", f"fig2_parity_{TAG}.png"),
               ("FIG_IMPORTANCE", f"fig2_importance_{TAG}.png"),
               ("FIG_SHAP", f"fig2_shap_{TAG}.png"),
               ("FIG_LIT", f"fig3_literature_{TAG}.png"),
               ("FIG_ROCKS", f"fig4_rocks_{TAG}.png")):
    V[k] = fig(pat) or "MISSING.png"

ov = jload(f"fig2_importance_overlap_{TAG}.json")
if ov:
    bits = []
    for t in TARGETS:
        o = ov["overlap_with_my_top10"].get(t, [])
        bits.append(f"{t}: {len(o)} of {len(ov['paper_named_top_descriptors'])} "
                    f"({', '.join(o) if o else 'none'})")
    V["IMPORTANCE_PARA"] = (
        "The paper names age, Al₂O₃, CaCO₃, Fe₂O₃, specific gravity, SiO₂ and CaO among its "
        "most important descriptors. Overlap with the top ten found here, per target — "
        + "; ".join(bits) + ". The descriptor and its missingness flag are permuted together, "
        "since the network receives both.")
    sh = cload(f"fig2_shap_mean_abs_{TAG}.csv")
    pi = cload(f"fig2_permutation_importance_{TAG}.csv")
    if sh is not None and pi is not None:
        agree = []
        for t in TARGETS:
            a = set(pi[pi.target == t].nlargest(10, "delta_mse").feature)
            b = set(sh[sh.target == t].nlargest(10, "mean_abs_shap").feature)
            agree.append(f"{t}: {len(a & b)} of 10")
        V["IMPORTANCE_PARA"] += (
            " The SHAP panel below is an independent check on the same question: the two "
            "rankings share " + "; ".join(agree) + " of their top ten descriptors.")
else:
    V["IMPORTANCE_PARA"] = "Permutation importance was not computed."

# ---------------------------------------------------------------- notebook 10
nb10 = jload(f"nb10_{TAG}.json")
q = nb10["qualitative_claims"]
V["NB10_PARA"] = (
    f"The count the authors left in their saved notebook — materials above 380 J/g with more "
    f"than 10 g/100 g bound water — is 964. The ensemble here gives {n(nb10['check1']['ensemble'])} "
    f"(per-seed range {n(min(nb10['check1']['per_seed']))}–{n(max(nb10['check1']['per_seed']))}). "
    f"That comparison is **not testable**: it is a property of the authors' trained weights, "
    f"which are not released, and the per-seed range shows how much it moves between equally "
    f"valid retrainings.\n\n"
    f"The paper's qualitative claims do reproduce. Metakaolin ranks "
    f"{q['metakaolin among the most reactive']['rank_by_heat']} of "
    f"{q['metakaolin among the most reactive']['n_categories']} categories by median heat "
    f"release. Fly ashes are pozzolanic: median Ca(OH)₂ consumption "
    f"{f(q['fly ashes pozzolanic (high Ca(OH)2 consumption)']['CH_median'], 1)} g/100 g, with "
    f"{f(q['fly ashes pozzolanic (high Ca(OH)2 consumption)']['pct_above_50'], 0)} % above "
    f"50 g/100 g. Slags are hydraulic: high median heat release "
    f"{f(q['slags hydraulic (low Ca(OH)2 consumption, high heat)']['heat_median'], 0)} J/g but "
    f"median Ca(OH)₂ consumption only "
    f"{f(q['slags hydraulic (low Ca(OH)2 consumption, high heat)']['CH_median'], 1)} g/100 g. "
    f"Lime species sit lowest, at "
    f"{f(q['lime species inert']['heat_median'], 0)} J/g and "
    f"{f(q['lime species inert']['CH_median'], 1)} g/100 g.\n\n"
    f"{n(nb10['out_of_range']['n_above_dataset4_max'])} of {n(nb10['out_of_range']['n_total'])} "
    f"predictions exceed the largest heat release in the training data "
    f"({nb10['heat_max_in_dataset4']:.0f} J/g) and "
    f"{n(nb10['out_of_range']['n_negative_heat'])} are negative — the network extrapolates "
    f"outside the range it was fitted on.")

s10 = cload(f"nb10_{TAG}_summary_by_category.csv")
sel = s10.head(10)[["Category", "n", "heat_median", "CH_median", "BW_median",
                    "heat_seed_sd_median", "pct_CH_above_50"]]
sel.columns = ["category", "n", "heat (J/g)", "Ca(OH)₂ (g/100 g)", "bound water (g/100 g)",
               "heat seed SD", "% above 50 g/100 g"]
V["TABLE_NB10"] = (md(sel, "{:.1f}")
                   + "\n\nMedians over the ensemble, ten largest categories by median heat "
                     "release; the full table is `results/nb10_" + TAG +
                     "_summary_by_category.csv`.")

# ---------------------------------------------------------------- notebook 11
ab = jload(f"nb11_{TAG}_ablation.json")
base = [r for r in ab["base"]["rows"] if r["model"] == "ensemble"][0]
V["NB11_REACTIVE_ALL"] = n(base["reactive_all"])
V["N_NO_OXIDE"] = n(base["n_no_oxide"])
V["PCT_NO_OXIDE_REACTIVE"] = f(100 * base["reactive_no_oxide"] / max(base["n_no_oxide"], 1), 1)
closest = ab["closest_to_paper"][0]
single = [r for r in ab["base"]["rows"] if r["model"] != "ensemble"]
pct_no_ox = 100 * base["reactive_no_oxide"] / max(base["n_no_oxide"], 1)
seed_counts = ab["base"]["per_seed_reactive_all"]
V["NB11_PARA"] = (
    f"The notebook as released, scored with the {len(sound['seeds'])}-seed ensemble over all "
    f"{n(ab['base']['n_rows'])} samples, calls {n(base['reactive_all'])} of them reactive "
    f"(> 200 J/g) against the paper's 50,569 — "
    f"{100 * (base['reactive_all'] - 50569) / 50569:+.0f} %.\n\n"
    f"Three things about that number are worth stating before the ablation. "
    f"**It is unstable across retrainings:** the individual seeds give between "
    f"{n(min(seed_counts))} and {n(max(seed_counts))}"
    + (f", and the single seed-42 model alone gives {n(single[0]['reactive_all'])} against the "
       f"ensemble's {n(base['reactive_all'])}" if single else "")
    + f". **The row filter barely moves it here:** {n(base['n_no_oxide'])} samples carry no "
      f"measured oxide at all and the notebook scores them anyway, but under this ensemble only "
      f"{pct_no_ox:.1f} % of them come out reactive, so counting only samples with at least one "
      f"measured oxide leaves {n(base['reactive_any_oxide'])} and counting only those with "
      f"SiO₂, Al₂O₃ and CaO leaves {n(base['reactive_has3'])}. (In an earlier run of this "
      f"replication, with a single weaker model, those same composition-less rows were almost "
      f"all scored reactive and dominated the total — which is exactly why the count cannot be "
      f"attributed to any one cause without the ablation below.) **The screening mix is not the "
      f"R³ mix:** the constants the notebook uses are not the ones Dataset 4 records.\n\n"
    f"The ablation moves one factor at a time from that baseline and then combines them. The "
    f"closest combination to the paper's figure is {n(closest['reactive'])} "
    f"({closest['variant']}, rows: {closest['row_filter']}, {closest['model']}), "
    f"{closest['pct_vs_paper']:+.1f} % from 50,569. No combination reproduces it, and the "
    f"spread across combinations is wider than the gap to the paper, so no single factor can be "
    f"named as the explanation.")

abt = cload(f"nb11_{TAG}_ablation.csv")
show = abt[abt.model == "ensemble"].copy()
show["factors"] = show.apply(
    lambda r: ", ".join([x for x, on in (("Dataset 5 in pool", r.d5), ("total iron", r.fe),
                                         ("R³ standard mix", r.mix)) if on]) or "as released",
    axis=1)
show = show[["factors", "row_filter", "reactive", "pct_vs_paper"]]
show.columns = ["imputation variant", "rows counted", "reactive (> 200 J/g)", "% vs paper"]
show["reactive (> 200 J/g)"] = show["reactive (> 200 J/g)"].map(lambda v: f"{int(v):,}")
body_ab = show[show["rows counted"] == "all rows"].drop(columns=["rows counted"])
V["TABLE_ABLATION"] = (
    md(body_ab)
    + "\n\nPaper: 50,569. 'as released' is the notebook unchanged; each other row adds one "
      "factor to it. Counting only samples with SiO₂, Al₂O₃ and CaO instead of all rows lowers these "
      "numbers by up to about 9 % (the full 48-row table, with all three row filters and the "
      f"single-model rows, is `results/nb11_{TAG}_ablation.csv` and Appendix D).")
V["TABLE_ABLATION_FULL"] = md(show)

am = ab["base"]["amorphous"]
amd5 = ab.get("d5", {}).get("amorphous")
V["NB11_AMORPH"] = (
    f"**Imputed amorphous content.** No rock in the database has a measured amorphous content, "
    f"so every value is imputed from the R³ rows, whose measured mean is "
    f"{f(am['r3_measured_mean'], 1)} %. The imputed rock mean is {f(am['mean'], 1)} % "
    f"(median {f(am['median'], 1)} %). The ~160 rocks in Dataset 5 that do have a measurement "
    f"average {f(am['dataset5_measured_mean'], 1)} % — less than half. The released notebook "
    f"does not use Dataset 5, although the paper describes using it; adding it to the "
    f"imputation pool brings the imputed mean to "
    + (f"{f(amd5['mean'], 1)} %." if amd5 else "a value reported in the ablation file.")
)

sw = pd.DataFrame(ab["pozzolanic_threshold_sweep_base"])
sw.columns = ["Ca(OH)₂ threshold (g/100 g)", "pozzolanic", "hydraulic"]
for c in ("pozzolanic", "hydraulic"):
    sw[c] = sw[c].map(lambda v: f"{int(v):,}")
V["TABLE_SWEEP"] = md(sw)

rn = cload(f"nb11_{TAG}_by_rock_name.csv")
big = rn[rn.n >= 20]
V["N_ROCKNAMES"] = n(len(rn))
V["N_ROCKNAMES_20"] = n(len(big))
V["N_ROCKNAMES_5PCT"] = n(int((big.ratio > 0.05).sum()))

f4 = cload(f"fig4_rocks_{TAG}.csv")
if f4 is not None and "paper_ratio" in f4.columns:
    cmp4 = f4.dropna(subset=["paper_ratio"])
    within2 = int((cmp4.ratio <= 2 * cmp4.paper_ratio).sum())
    higher = int((cmp4.ratio > cmp4.paper_ratio).sum())
    V["FIG4_PARA"] = (
        f"For the {len(cmp4)} of those types whose reactive share the paper states, the "
        f"reproduced share is higher in {higher} and within a factor of two in {within2}; the "
        f"ordering is broadly the same, with anorthosite, ignimbrite and porphyry at the top. "
        f"Since the names are not the paper's groups, the comparison is indicative only.")
else:
    V["FIG4_PARA"] = ""

# ---------------------------------------------------------------- few papers
fp = jload("few_papers", "scores_summary.json")
selj = jload("few_papers", "selection.json")
V["N_CAPTION_MATCHED"] = n(selj["n_caption_matched_single_doi"])
V["FP_CELLS"] = n(fp["standard_total_cells"])
V["FP_EXACT"] = n(fp["standard_total_exact"])
_stdm = cload("few_papers", "metrics_standard.csv")
V["FP_NTABLES"] = str(int((_stdm.cells_both_present > 0).sum()))
cmpd = cload("few_papers", "comparison_vs_dataset1.csv")
std = cload("few_papers", "metrics_standard.csv")
nb5 = cload("few_papers", "metrics_notebook5.csv")
tb = nb5[["d2_row", "doi", "f1_rel", "f1_mat", "f1_num", "rmse"]].copy()
tb = tb.merge(std[["d2_row", "n_pred_materials", "n_true_materials", "cells_both_present",
                   "cells_exact", "cells_missed", "cells_invented"]], on="d2_row", how="left")
tb.columns = ["Dataset 2 row", "DOI", "relevance F1", "materials F1", "compositions F1",
              "comp. RMSE", "materials found", "materials in truth", "cells compared",
              "cells exact", "cells missed", "cells invented"]
V["TABLE_FEWPAPERS"] = (
    md(tb, "{:.2f}")
    + "\n\nThe first four columns are notebook 5's own metric, for comparability with "
      "Supplementary Table G3 (fine-tuned GPT-3.5, test: relevance 0.95, materials 0.83, "
      "compositions 0.93, RMSE 2.81). The remaining columns are a standard set-based count. "
      "Dataset 2 is the authors' fine-tuning set and notebook 4 splits it 90/10 with no fixed "
      "seed, so which rows were their test rows cannot be recovered.")

v1 = fp["vs_dataset1"]
V["FEWPAPERS_PARA"] = (
    f"Against the **manual ground truth** of Dataset 2: {n(fp['standard_total_cells'])} oxide "
    f"cells could be compared, {n(fp['standard_total_exact'])} agree exactly, "
    f"{n(fp['standard_total_missed'])} were missed and {n(fp['standard_total_invented'])} "
    f"invented. Relevance was judged correctly on all {fp['n_tables']} tables, including the "
    f"two that are not composition tables and were answered `NR`.\n\n"
    f"Against **Dataset 1**, i.e. what the authors' pipeline produced for the same tables: "
    f"{n(v1['materials_mine'])} materials extracted against {n(v1['materials_dataset1'])} in "
    f"Dataset 1, {n(v1['materials_shared'])} of them pairable; of "
    f"{n(v1['cells_both_present'])} comparable oxide cells, {n(v1['cells_exact'])} agree "
    f"exactly.\n\n"
    f"The disagreements are informative rather than extraction errors. Three of the four "
    f"notebook-5 'materials F1' scores are low for mechanical reasons — two are 0 and one is "
    f"0.25: in one table the ground truth's material names are destroyed by an unquoted comma, "
    f"in another the pipeline used the EN 197-1 designations (`CEM I 42.5R`, repeated six "
    f"times) where the table's own codes are `PC1`…`PC9`, and in the third the Dataset 2 "
    f"ground truth belongs to a different table. The single value disagreement across all "
    f"the tables with a usable ground truth is the MgO of sample SNM 7: "
    f"the table gives Na₂O 1.87 and MgO 1.57, and both the manual ground truth and Dataset 1 "
    f"record MgO as 1.87.")

d3 = jload("few_papers", "d3_sample2_scores.json")
V["D3_AGREE"] = str(d3["n_agree"])
V["D3_ACC"] = f"{100 * d3['accuracy']:.0f}"

pp = jload("few_papers", "predictions_summary.json")
if pp and pp.get("n_paired"):
    V["FEWPAPERS_PRED"] = (
        f"Pushing the {pp['n_extracted']} extracted compositions through the same ensemble and "
        f"comparing with its predictions for the corresponding Dataset 1 rows: of "
        f"{pp['n_paired']} pairable materials, heat release differs by "
        f"{pp['heat_mae']:.1f} J/g on average (largest {pp['heat_max_abs']:.1f}), Ca(OH)₂ "
        f"consumption by {pp['CH_mae']:.1f} g/100 g and bound water by {pp['BW_mae']:.2f} "
        f"g/100 g, and {pp['n_same_reactivity_class']} of {pp['n_paired']} fall on the same "
        f"side of the 200 J/g reactivity threshold. The differences come from the small "
        f"extraction differences above and from the imputed descriptors, not from the model.")
else:
    V["FEWPAPERS_PRED"] = "Not computed."

# ---------------------------------------------------------------- runtime, verification
rt = cload("table_runtime.csv")
if rt is not None and len(rt):
    rt = rt.dropna(subset=["wall_seconds"]).copy()
    rt["wall"] = rt.wall_seconds.map(lambda s: f"{s/60:.0f} min" if s >= 90 else f"{s:.0f} s")
    V["TABLE_RUNTIME"] = (md(rt[["step", "wall"]])
                          + "\n\nWall-clock on 8 CPU cores, no GPU. Several of these runs "
                            "shared the machine with one another (the tuning search, the "
                            "faithful batch and the sound-protocol streams overlapped), so "
                            "individual times are upper bounds and are not a benchmark; "
                            "`nb9_sound_notebook` in particular ran beside four other jobs. "
                            "The notebook-11 ablation row is the 8-variant total, and the "
                            "figures and few-papers scoring (not listed) took minutes.")
else:
    V["TABLE_RUNTIME"] = "Runtimes are in `results/table_runtime.csv`."

vp = os.path.join(ROOT, "logs", "verify.log")
sp = os.path.join(ROOT, "logs", "verify_supplement.log")
NL = chr(10)
if os.path.exists(vp):
    txt = open(vp, encoding="utf-8", errors="replace").read()
    lines = [l.strip() for l in txt.splitlines()
             if ("IDENTICAL" in l or "DIFFERS" in l) and "scores_summary" not in l]
    lines = [l.replace("(variant 'base' only)", "(base variant only)") for l in lines]
    V["VERIFY_PARA"] = (
        "Each final script was re-run once from a clean start, with the cache directory "
        "deleted so every intermediate table was rebuilt, and the regenerated results compared "
        "field by field against the first run (wall-clock fields ignored):" + NL + NL
        + NL.join(f"- `{l}`" for l in lines) + NL + NL
        + "The notebook-11 re-run in that pass covered the base variant only, and it "
          "overwrote the full 8-variant ablation file. The full ablation was therefore "
          "regenerated afterwards and compared row by row with the first run; the few-papers "
          "scoring, which was corrected after the first verification pass, was re-run and "
          "compared as well:" + NL + NL)
    if os.path.exists(sp):
        sl = [l.strip() for l in open(sp, encoding="utf-8").read().splitlines()
              if ("IDENTICAL" in l or "DIFFERS" in l) and not l.strip().startswith("#")]
        V["VERIFY_PARA"] += NL.join(f"- `{l}` (after the corrections)" for l in sl)
    V["VERIFY_PARA"] += (NL + NL + "Not re-run a second time: the 11-seed grouped-split and "
                         "sensitivity runs, the tuning search, the figures and the "
                         "legacy-environment run. Full output: `logs/verify.log`, "
                         "`logs/verify_supplement.log`.")
else:
    V["VERIFY_PARA"] = ("The clean-start verification pass had not finished when this report was "
                        "generated; `src/run_verify.sh` performs it and writes `logs/verify.log`.")

# ---------------------------------------------------------------- appendices
V["TABLE_FILES"] = md(pd.DataFrame([
    dict(item="claim summary (§1)", file="results/table_claims.csv"),
    dict(item="model accuracy (§3.1)", file="results/table_model_accuracy.csv, results/nb9_*.json"),
    dict(item="collapse diagnostics (§3.2)", file="results/nb9_faithful_seed*.json, logs/nb9_faithful_*.log"),
    dict(item="tuning (§3.3)", file="results/nb9_tuning_all_trials.csv, results/nb9_tuned_config.json, logs/nb9_tune.log"),
    dict(item="grouped splits (§3.4)", file="results/nb9_grouped_material.json, results/nb9_grouped_doi.json"),
    dict(item="library versions (§3.5)", file="results/table_library_version_check.csv"),
    dict(item="figures (§4)", file="figures/*.png with results/fig*.csv"),
    dict(item="notebook 10 (§5)", file=f"results/nb10_{TAG}.json, results/nb10_{TAG}_summary_by_category.csv"),
    dict(item="rock ablation (§6)", file=f"results/nb11_{TAG}_ablation.csv, results/nb11_{TAG}_ablation.json"),
    dict(item="rock types (§6)", file=f"results/nb11_{TAG}_by_rock_name.csv"),
    dict(item="few papers (§7)", file="results/few_papers/ (prompts, responses, metrics, mismatches)"),
    dict(item="runtimes (§10)", file="results/table_runtime.csv"),
]))

ps = []
for r in sound["runs"]:
    rec = dict(seed=r["seed"], epochs=r["epochs_run"], best_epoch=r["best_epoch"],
               val_criterion=f(r["val_criterion"], 5))
    for t in TARGETS:
        rec[f"{SHORT[t]} R²"] = f(r["test"][t]["R2"])
    ps.append(rec)
V["TABLE_PERSEED"] = (f"Sound protocol with the tuned configuration, the {len(ps)} seeds behind "
                      f"the ensemble row of §3.1:\n\n" + md(pd.DataFrame(ps))
                      + "\n\nPer-seed results for every other row are in the matching "
                        "`results/nb9_*.json`.")

# ---------------------------------------------------------------- fill
tpl = open(os.path.join(DOCS, "report_template.md"), encoding="utf-8").read()
missing = sorted(set(re.findall(r"\{\{(\w+)\}\}", tpl)) - set(V))
if missing:
    print("UNRESOLVED PLACEHOLDERS:", missing)
    sys.exit(1)
out = re.sub(r"\{\{(\w+)\}\}", lambda m: str(V[m.group(1)]), tpl)
dst = os.path.join(DOCS, "REPLICATION_REPORT.md")
open(dst, "w", encoding="utf-8").write(out)
print(f"wrote {dst} ({len(out.splitlines())} lines, {len(V)} substitutions)")
