"""Build REPLICATION_REPORT_FINAL.docx (A4, Times New Roman 11 pt, justified, 1.15 spacing).

Every number is read from results/*.json or results/*.csv; the only values typed here are the
paper's own reported figures (marked as such) and fixed facts about the original repository.

usage: python make_docx.py [--out docs/REPLICATION_REPORT_FINAL.docx]
"""
import os
import re
import sys
import glob
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Cm, RGBColor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(ROOT, "report", "REPLICATION_REPORT_FINAL.docx"))
ap.add_argument("--author", default=None, help="author name; a placeholder is printed if omitted")
ap.add_argument("--repl-url", default="https://github.com/Shotabdychy/cementitious-reactivity-replication",
                help="URL of the replication repository (supplied by the report author)")
args = ap.parse_args()

ORIG_URL = "https://github.com/olivettigroup/Cementitious-reactivity-prediction-and-discovery"
ORIG_COMMIT = "2340ec87eb3a8f5d564528f08a85f509887c811c"
PAPER_URL = "https://www.nature.com/articles/s43246-025-00820-4"
PAPER_DOI = "10.1038/s43246-025-00820-4"
REPL_URL = args.repl_url
AUTHOR = args.author or "[Author name]"
REPL_PLACEHOLDER = "[replication repository URL: to be added after publication]"
PAPER = {"heat": (0.93, 28.20), "CH": (0.85, 12.17), "BW": (0.89, 1.47)}


def J(*p):
    with open(os.path.join(RES, *p), encoding="utf-8") as f:
        return json.load(f)


def C(*p):
    return pd.read_csv(os.path.join(RES, *p))


T = ("heat release", "CH consumption", "Bound water")
S = {"heat release": "heat", "CH consumption": "CH", "Bound water": "BW"}

sound = J("nb9_sound_tuned.json")
snb = J("nb9_sound_notebook.json")
grp = J("nb9_grouped_material.json")
nb10 = J("nb10_sound_tuned.json")
ab = J("nb11_sound_tuned_ablation.json")
fp = J("few_papers", "scores_summary.json")
d3 = J("few_papers", "d3_sample2_scores.json")
pp = J("few_papers", "predictions_summary.json")
tc = J("nb9_tuned_config.json")
WE = J("few_papers", "worked_example.json")
std = C("few_papers", "metrics_standard.csv")
cmp1 = C("few_papers", "comparison_vs_dataset1.csv")
cat = C("nb10_sound_tuned_summary_by_category.csv").set_index("Category")
faith = [J(os.path.basename(p)) for p in glob.glob(os.path.join(RES, "nb9_faithful_seed*.json"))
         if re.fullmatch(r"nb9_faithful_seed\d+\.json", os.path.basename(p))]
clean = [J(os.path.basename(p)) for p in glob.glob(os.path.join(RES, "nb9_clean_seed*.json"))
         if re.fullmatch(r"nb9_clean_seed\d+\.json", os.path.basename(p))]
legacy = J("nb9_faithful_seed42_legacy.json")
main42 = J("nb9_faithful_seed42.json")
base = [r for r in ab["base"]["rows"] if r["model"] == "ensemble"][0]


def f3(x):
    return f"{x:.3f}".replace("-", "−")


def f2(x):
    return f"{x:.2f}".replace("-", "−")


def n(x):
    return f"{int(x):,}"


# ============================================================ document plumbing
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(sec, side, Cm(2.54))
sec.footer_distance = Cm(1.25)

FONT = "Times New Roman"


def set_font(style, size, bold=None, italic=None):
    style.font.name = FONT
    style.font.size = Pt(size)
    style.font.color.rgb = RGBColor(0, 0, 0)
    if bold is not None:
        style.font.bold = bold
    if italic is not None:
        style.font.italic = italic
    rpr = style.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.append(rf)
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf.set(qn(a), FONT)
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        if rf.get(qn(a)) is not None:
            del rf.attrib[qn(a)]


normal = doc.styles["Normal"]
set_font(normal, 11)
pf = normal.paragraph_format
pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
pf.line_spacing = 1.15
pf.space_before, pf.space_after = Pt(0), Pt(4)
pf.widow_control = True          # never leave a single line of a paragraph alone at a page edge

for name, size, before, after in (("Heading 1", 13, 6, 3), ("Heading 2", 11, 4, 2)):
    st = doc.styles[name]
    set_font(st, size, bold=True, italic=False)
    st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    st.paragraph_format.space_before, st.paragraph_format.space_after = Pt(before), Pt(after)
    st.paragraph_format.keep_with_next = True
    st.paragraph_format.line_spacing = 1.15

cap = doc.styles["Caption"]
set_font(cap, 9.5, bold=False, italic=False)
cap.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
cap.paragraph_format.space_before, cap.paragraph_format.space_after = Pt(2), Pt(5)
cap.paragraph_format.line_spacing = 1.0


def add_field(run, instr):
    for typ, txt in (("begin", None), (None, instr), ("end", None)):
        if typ:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), typ)
        else:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = txt
        run._r.append(el)


fp_ = sec.footer.paragraphs[0]
fp_.alignment = WD_ALIGN_PARAGRAPH.CENTER
r_ = fp_.add_run()
r_.font.name, r_.font.size = FONT, Pt(10)
add_field(r_, "PAGE")


def hyperlink(par, url, text, size=None):
    rid = par.part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
                             is_external=True)
    h = OxmlElement("w:hyperlink")
    h.set(qn("r:id"), rid)
    r = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    rf = OxmlElement("w:rFonts")
    for a in ("w:ascii", "w:hAnsi", "w:cs"):
        rf.set(qn(a), FONT)
    rpr.append(rf)
    col = OxmlElement("w:color")
    col.set(qn("w:val"), "1F3A93")
    rpr.append(col)
    u = OxmlElement("w:u")
    u.set(qn("w:val"), "single")
    rpr.append(u)
    if size:
        sz = OxmlElement("w:sz")
        sz.set(qn("w:val"), str(int(size * 2)))
        rpr.append(sz)
    r.append(rpr)
    t = OxmlElement("w:t")
    t.text = text
    t.set(qn("xml:space"), "preserve")
    r.append(t)
    h.append(r)
    par._p.append(h)


TOKEN = re.compile(r"(\*\*.+?\*\*|\*.+?\*|\[\[.+?\|.+?\]\])")


DASHES = str.maketrans({"—": "-", "–": "-", "−": "-"})


def lead_bold_only(t):
    """Keep bold only for a label at the very start of a paragraph; drop it in the middle."""
    if t.startswith("**") and "**" in t[2:]:
        j = t.index("**", 2) + 2
        return t[:j] + t[j:].replace("**", "")
    return t.replace("**", "")


def rich(par, text, size=None, bold=False):
    """Mini-markup: **bold**, *italic*, [[url|label]]. Every dash is written as the plain keyboard hyphen."""
    text = text.translate(DASHES)
    for part in TOKEN.split(text):
        if not part:
            continue
        if part.startswith("**"):
            r = par.add_run(part[2:-2])
            r.bold = True
        elif part.startswith("[["):
            url, label = part[2:-2].split("|", 1)
            hyperlink(par, url, label, size)
            continue
        elif part.startswith("*"):
            r = par.add_run(part[1:-1])
            r.italic = True
        else:
            r = par.add_run(part)
            r.bold = bold or None
        if size:
            r.font.size = Pt(size)
    return par


KEEP_LIMIT = 520     # characters; paragraphs shorter than this are never split across pages


def P(text, keep=None, **kw):
    p = doc.add_paragraph()
    text = lead_bold_only(text)
    rich(p, text, **kw)
    short = len(text) < KEEP_LIMIT if keep is None else keep
    p.paragraph_format.keep_together = short
    return p


def H1(t):
    return doc.add_heading(t.translate(DASHES), level=1)


def H2(t):
    return doc.add_heading(t.translate(DASHES), level=2)


def bullets(items, size=None):
    for it in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(1.5)
        p.paragraph_format.left_indent = Cm(0.6)
        p.paragraph_format.first_line_indent = Cm(-0.4)
        p.paragraph_format.line_spacing = 1.1
        p.paragraph_format.keep_together = True
        for r in p.runs:
            r.font.name = FONT
        rich(p, lead_bold_only(it), size=size)


def caption(text, kind, num):
    p = doc.add_paragraph(style="Caption")
    p.paragraph_format.keep_with_next = (kind == "Table")
    p.paragraph_format.keep_together = True
    r = p.add_run(f"{kind} {num}. ")
    r.bold = True
    rich(p, text)
    return p


def border(cell, **edges):
    tcPr = cell._tc.get_or_add_tcPr()
    b = tcPr.find(qn("w:tcBorders"))
    if b is None:
        b = OxmlElement("w:tcBorders")
        tcPr.append(b)
    for edge, val in edges.items():
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), val[0])
        el.set(qn("w:sz"), str(val[1]))
        el.set(qn("w:color"), "000000")
        b.append(el)


def table(header, rows, widths_cm, size=8.5, bold_last=False, left_cols=1, shade_rows=()):
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    tblPr = t._tbl.tblPr
    lay = OxmlElement("w:tblLayout")
    lay.set(qn("w:type"), "fixed")
    tblPr.append(lay)
    mar = OxmlElement("w:tblCellMar")
    for side, w in (("top", 20), ("bottom", 20), ("left", 60), ("right", 60)):
        e = OxmlElement(f"w:{side}")
        e.set(qn("w:w"), str(w))
        e.set(qn("w:type"), "dxa")
        mar.append(e)
    tblPr.append(mar)
    for ri, row in enumerate([header] + rows):
        trPr = t.rows[ri]._tr.get_or_add_trPr()
        cs = OxmlElement("w:cantSplit")
        trPr.append(cs)
        for ci, val in enumerate(row):
            c = t.cell(ri, ci)
            c.width = Cm(widths_cm[ci])
            p = c.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if ci < left_cols else WD_ALIGN_PARAGRAPH.CENTER
            rich(p, str(val), size=size, bold=(ri == 0))
            p.paragraph_format.keep_with_next = (ri < len(rows))
            if ri == 0:
                border(c, top=("single", 8), bottom=("single", 6))
            if ri == len(rows):
                border(c, bottom=("single", 8))
            if (ri - 1) in shade_rows:
                shd = OxmlElement("w:shd")
                shd.set(qn("w:val"), "clear")
                shd.set(qn("w:fill"), "EDEDED")
                c._tc.get_or_add_tcPr().append(shd)
    return t


def picture(path, width_cm):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(path, width=Cm(width_cm))


# ============================================================ compact figures
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 7, "axes.linewidth": 0.6, "axes.unicode_minus": False})
pred = C("nb9_sound_tuned_predictions.csv")
fig, axes = plt.subplots(1, 3, figsize=(6.6, 1.95))
for ax, t in zip(axes, T):
    s_ = S[t]
    d = pred[pred[f"y_{s_}"].notna()]
    for split, col, lab in (("train", "#7f9fc9", "train"), ("test", "#c0392b", "test")):
        q = d[d.split == split]
        ax.scatter(q[f"y_{s_}"], q[f"pred_{s_}"], s=6, c=col, alpha=0.6, linewidths=0, label=lab)
    lo = min(d[f"y_{s_}"].min(), d[f"pred_{s_}"].min())
    hi = max(d[f"y_{s_}"].max(), d[f"pred_{s_}"].max())
    ax.plot([lo, hi], [lo, hi], color="#333333", lw=0.6, ls="--")
    e = sound["ensemble_test"][t]
    unit = "J/g" if t == "heat release" else "g/100 g"
    ax.set_title(f"{t} ({unit})", fontsize=7.5)
    ax.set_xlabel("measured")
    ax.text(0.04, 0.96, f"test $R^2$ = {e['R2']:.3f}\nRMSE = {e['RMSE']:.2f}\n$n$ = {e['n']}",
            transform=ax.transAxes, va="top", fontsize=6.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
axes[0].set_ylabel("predicted (ensemble mean)")
axes[0].legend(loc="lower right", frameon=False, fontsize=6.5, markerscale=1.5)
fig.tight_layout(pad=0.4)
parity_png = os.path.join(FIG, "fig_parity_report.png")
fig.savefig(parity_png, dpi=300, facecolor="white")
plt.close(fig)

# ============================================================ numbers used in the text
e = {t: sound["ensemble_test"][t] for t in T}
g = {t: grp["ensemble_test"][t] for t in T}
nbk = {t: snb["ensemble_test"][t] for t in T}
fa = {t: [r["test"][t]["R2"] for r in faith] for t in T}
far = {t: [r["test"][t]["RMSE"] for r in faith] for t in T}
mon = [r["monitor_at_best"] for r in faith]
n_coll = sum(r["collapsed"] for r in faith)
cl_coll = sum(r["collapsed"] for r in clean)
cl_ch = np.mean([r["test"]["CH consumption"]["R2"] for r in clean])
cl_ch_sd = np.std([r["test"]["CH consumption"]["R2"] for r in clean], ddof=1)
lt = sound["labels_test"]
single = [r for r in ab["base"]["rows"] if r["model"] != "ensemble"][0]
seedc = ab["base"]["per_seed_reactive_all"]
amo = ab["base"]["amorphous"]
n_tab_gt = int((std.cells_both_present > 0).sum())
tot_extr = int(cmp1.n_mine.sum())
v1 = fp["vs_dataset1"]
rows_ab = {r["variant"]: r for r in []}
abl = C("nb11_sound_tuned_ablation.csv")
abl_e = abl[(abl.model == "ensemble") & (abl.row_filter == "all rows")].set_index("variant")


def abl_n(v):
    return n(abl_e.loc[v, "reactive"])


# ============================================================ PAGE 1: title, abstract, introduction
tp = doc.add_paragraph()
tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
tp.paragraph_format.space_after = Pt(3)
r = tp.add_run("Replicating a Data-Driven Screening Framework for Secondary and Natural "
               "Cementitious Precursors: Code Execution, Modified Training Protocol and "
               "Reproducibility Assessment")
r.bold = True
r.font.size = Pt(14)
lp = doc.add_paragraph()
lp.alignment = WD_ALIGN_PARAGRAPH.CENTER
lp.paragraph_format.space_after = Pt(1)
rich(lp, "**Original study:** Mahjoubi et al. (2025), ", size=10)
hyperlink(lp, "https://doi.org/" + PAPER_DOI, "doi:" + PAPER_DOI, 10)
lp1 = doc.add_paragraph()
lp1.alignment = WD_ALIGN_PARAGRAPH.CENTER
lp1.paragraph_format.space_after = Pt(1)
rich(lp1, "**Original code:** ", size=10)
hyperlink(lp1, ORIG_URL, ORIG_URL, 10)
lp2 = doc.add_paragraph()
lp2.alignment = WD_ALIGN_PARAGRAPH.CENTER
lp2.paragraph_format.space_after = Pt(5)
rich(lp2, "**Replication repository:** ", size=10)
if REPL_URL:
    hyperlink(lp2, REPL_URL, REPL_URL, 10)
else:
    rich(lp2, REPL_PLACEHOLDER, size=10)

H1("Abstract")
P(f"This report replicates the computational workflow of Mahjoubi et al. (2025), who predict the "
  f"reactivity of cementitious precursors (R³ heat release, Ca(OH)₂ consumption and bound "
  f"water) with a multi-headed neural network and use it to screen literature-extracted materials "
  f"and about 10⁶ natural rock samples. The released notebooks 9–11 were executed as "
  f"published and again under a modified training protocol (whole-validation-set early stopping, "
  f"training-only preprocessing, cross-validated tuning, 11-seed ensembles). Because the authors’ "
  f"input tables and trained weights are not released, inputs were reconstructed from the "
  f"supplementary datasets, and the language-model extraction step was reproduced on a small sample "
  f"with a substitute model. As released, the code produced unstable models (Ca(OH)₂ test "
  f"*R*² = {f3(np.mean(fa['CH consumption']))} ± {f3(np.std(fa['CH consumption'], ddof=1))} "
  f"over 11 seeds). Under the modified protocol the ensemble reached *R*² = {f3(e['heat release']['R2'])}, "
  f"{f3(e['CH consumption']['R2'])} and {f3(e['Bound water']['R2'])} for heat release, Ca(OH)₂ "
  f"consumption and bound water (paper: 0.93, 0.85, 0.89), the latter two resting on only "
  f"{lt['CH consumption']} and {lt['Bound water']} test labels. Accuracy fell sharply when whole "
  f"materials were held out (Ca(OH)₂ *R*² = {f3(g['CH consumption']['R2'])}). The published "
  f"rock-screening count was not reproduced, and every comparison that depends on the authors’ "
  f"trained weights is untestable.", size=10.5)

H1("1. Introduction")
P("Reducing clinker content requires new supplementary cementitious materials, and screening "
  "candidates by measured reactivity is slow. Mahjoubi et al. [1] combine fine-tuned language models "
  "that mine oxide compositions from the literature (notebooks 1–8 of their repository [2]) with a "
  "multi-task network that predicts three R³ outcomes from oxide composition, specific gravity, "
  "amorphous content, median particle size and test conditions (notebook 9). The network is then "
  "applied to 14,429 literature-extracted materials (notebook 10) and to the Gard et al. whole-rock "
  "database [3] (notebook 11).")
P("**Objective and scope.** The aim was to run the released code, obtain its outputs and check the "
  "published results within a limited compute budget. Notebooks 9–11 were executed in full. Notebooks "
  "1–8 depend on a private paper corpus and on the authors’ fine-tuned endpoints and could not be run; "
  "their extraction and classification steps were demonstrated on a small sample of tables (Section 4). "
  "Analyses that the paper does not contain (a modified training protocol, held-out-material "
  "validation, and an ablation of the rock-screening pipeline) were added independently and are "
  "labelled as such. The authors’ code [2] and this replication’s repository [5] are separate; see "
  "Code and Data Availability.")

# ============================================================ PAGE 2: methodology
H1("2. Methodology and Computational Implementation")
H2("2.1 Source code and reconstructed inputs")
P("The original repository contains 14 notebooks. Five files they load are absent: *R3.csv*, "
  "*natural_rocks.csv*, *df_imputed.pkl*, and the trained models *90andupnewones.keras* (notebook 10) "
  "and *90andupnewonesch.keras* (notebook 11). Substitutes: (i) *R3.csv* was rebuilt from supplementary "
  "Dataset 4 (1,849 of 1,875 rows have a usable material mass; the mix columns, stored in grams, were "
  "divided by it); (ii) *df_imputed.pkl* was rebuilt with the notebook’s LightGBM imputation of the "
  "R³ features; (iii) *natural_rocks.csv* was rebuilt by merging the Gard et al. tables, keeping all "
  "1,022,092 samples; (iv) both models were retrained, so comparisons with values the authors left in "
  "saved cells are reported as untestable, not as failures.")
H2("2.2 Faithful execution versus modified protocol")
P("**Faithful runs** copy notebook 9 cell by cell, including its early-stopping monitor "
  "(a normalised RMSE for the Ca(OH)₂ head averaged over validation batches of 50 rows), "
  "its validation split (the last 15 % of the training array) and its preprocessing, which fits "
  "the scaler and target standardisation on all rows before splitting. They were run with 11 seeds "
  "(42, 1–10). A **clean variant** keeps the monitor but removes the leakage and draws the "
  "validation rows at random. The **modified protocol** (an independent addition) changes four things "
  "at once: early stopping on the masked validation loss over the whole validation set; a random "
  "validation split stratified on which target is labelled; scaling, standardisation and imputation fitted "
  "on training rows only; and an outer split stratified on target availability. These changes were not "
  "evaluated separately.")
H2("2.3 Training, validation and evaluation")
P(f"Two network configurations were used, and Table 1 states which applies to each row. The "
  f"**released architecture** has three NaN-masked regression heads on 256–256–16 ReLU layers, "
  f"dropout 0.1 after the first layer, Adam at 10⁻³, batch 50, patience 200 and the notebook’s loss "
  f"weights (Ca(OH)₂ head ×5); it is used by the faithful and clean runs and by the "
  f"“notebook hyperparameters” rows. The **tuned architecture** was selected by a KerasTuner [4] random search of "
  f"{tc['trials']} trials with {tc['folds']}-fold cross-validation inside the training rows, scored on a "
  f"criterion fixed across trials: widths {'–'.join(str(w) for w in tc['widths'])}, dropout "
  f"{tc['dropout']}, L2 {tc['l2']:g}, learning rate {tc['lr'] * 1e3:.2f} × 10⁻³, batch {tc['batch']}, equal loss "
  f"weights with the Ca(OH)₂ head ×{tc['ch_mult']:g}, patience {tc['patience']}. All results in "
  f"Sections 3–4 and the figures use the tuned ensemble unless stated otherwise. The test rows were "
  f"never read during tuning, early stopping or model selection. Each configuration was trained with "
  f"11 seeds and the ensemble is the mean of the seed predictions. The fixed test set holds {sound['n_test']} "
  f"rows with {lt['heat release']}, {lt['CH consumption']} and {lt['Bound water']} labels for heat "
  f"release, Ca(OH)₂ consumption and bound water. Because so few rows carry the second and third "
  f"labels, 95 % confidence intervals were estimated by a 2,000-sample bootstrap over test rows. "
  f"To estimate performance on unseen materials, the whole protocol was repeated with a split grouped "
  f"by material name (277 groups).")
H2("2.4 Software environment")
P("Python 3.11.9, TensorFlow 2.15.1, pandas 3.0.6, scikit-learn 1.9.1 and LightGBM 4.7.0 on Windows 11 "
  "(8 CPU cores, no GPU), pinned in *requirements.txt*. The paper states Python 3.8, which was not "
  "used; one faithful run was repeated with older libraries (TensorFlow 2.12.0, pandas 1.5.3; "
  "Section 5). Seeds and commands are in the repository README.")

# ============================================================ PAGE 3: model results
H1("3. Model Replication Results")
P(f"**Faithful runs.** All {len(faith)} faithful runs satisfy a collapse rule defined on validation "
  f"behaviour only (monitored value non-finite, restored epoch ≤ 3, or normalised RMSE > 1): the "
  f"monitored value at the restored epoch ranges from {min(mon):.3g} to {max(mon):.3g}. Test *R*² "
  f"for Ca(OH)₂ consumption ranged from {f3(min(fa['CH consumption']))} to "
  f"{f3(max(fa['CH consumption']))} across seeds, and heat release from {f3(min(fa['heat release']))} "
  f"to {f3(max(fa['heat release']))}.")
P(f"**Modified protocol.** The tuned ensemble exceeds the published heat-release *R*² and is below "
  f"the published Ca(OH)₂ and bound-water values. For those two targets the bootstrap intervals "
  f"([{f3(e['CH consumption']['R2_lo'])}, {f3(e['CH consumption']['R2_hi'])}] and "
  f"[{f3(e['Bound water']['R2_lo'])}, {f3(e['Bound water']['R2_hi'])}]) include the published values, "
  f"but this shows only that the test sets are too small to separate the two, not that they are "
  f"equivalent; the paper reports no intervals and does not state its split. The released "
  f"hyperparameters scored slightly higher on test than the tuned ones (Table 1); the "
  f"cross-validation-selected configuration was retained, because choosing by test score would be "
  f"selection on the test set. The improvement over the faithful runs comes from the combined "
  f"protocol changes (Section 2.2), whose separate effects were not isolated. The clean "
  f"variant, which keeps the original monitor, remains unstable ({cl_coll} of {len(clean)} runs "
  f"collapse; Ca(OH)₂ *R*² = {f3(cl_ch)} ± {f3(cl_ch_sd)}), which is consistent with the "
  f"monitor being an important contributor but does not prove it.")

caption("Test-set accuracy of the reactivity model: published values and reproduced values. "
        "Ensembles are the mean of 11 seeds with bootstrap 95 % confidence intervals in brackets; "
        "the faithful row is the mean ± standard deviation over 11 single-seed runs. “Released” "
        "architecture: 256–256–16; “tuned”: " + "–".join(str(w) for w in tc["widths"]) + " (Section 2.3). "
        "RMSE in J/g (heat release) and g/100 g (other targets).", "Table", 1)


def cell(rm, rmse):
    return f"{f3(rm)} / {f2(rmse)}"


rows = [
    ["Published (Mahjoubi et al.)", "0.930 / 28.20", "0.850 / 12.17", "0.890 / 1.47"],
    ["Faithful notebook 9 (released arch.), 11 seeds",
     *[f"{f3(np.mean(fa[t]))} ± {f3(np.std(fa[t], ddof=1))} / {f2(np.mean(far[t]))}" for t in T]],
    ["Modified protocol, released arch., ensemble",
     *[f"{f3(nbk[t]['R2'])} / {f2(nbk[t]['RMSE'])}" for t in T]],
    ["Modified protocol, tuned arch., ensemble",
     *[f"{f3(e[t]['R2'])} / {f2(e[t]['RMSE'])}" for t in T]],
    ["  95 % CI of the tuned ensemble",
     *[f"[{f3(e[t]['R2_lo'])}, {f3(e[t]['R2_hi'])}] / [{f2(e[t]['RMSE_lo'])}, {f2(e[t]['RMSE_hi'])}]" for t in T]],
    ["Modified protocol, tuned arch., grouped split",
     *[f"{f3(g[t]['R2'])} / {f2(g[t]['RMSE'])}" for t in T]],
    ["Test labels (row-level / grouped)",
     *[f"{sound['labels_test'][t]} / {grp['labels_test'][t]}" for t in T]],
]
table(["Row", "Heat release\n*R*² / RMSE", "Ca(OH)₂ consumption\n*R*² / RMSE",
       "Bound water\n*R*² / RMSE"], rows, [5.6, 3.5, 3.7, 3.1], size=8, shade_rows=(3,))
doc.add_paragraph().paragraph_format.space_after = Pt(0)
picture(os.path.join(FIG, "fig_summary_report.png"), 15.9)
caption("Published and reproduced test R² for the three targets (n = labelled test rows). Released code: "
        "notebook 9 as published, mean of 11 runs (bar: SD). Modified protocol: tuned 11-seed ensemble "
        "(bar: 95 % bootstrap CI). New materials: the same protocol with no material shared between "
        "training and test.", "Figure", 1)
# ============================================================ PAGE 4: literature + screening
H1("4. Literature Extraction and Material Screening")
H2("4.1 Extraction demonstration with a substitute model")
P("The authors’ fine-tuned GPT-3.5 endpoints are inaccessible, so **Claude Opus 5** was used as a "
  "substitute extractor with the notebook-4 prompt verbatim. Dataset 2 supplies the table text and a "
  "**manual ground truth**; Dataset 1 supplies the **authors’ pipeline output**, with DOI and caption. "
  "Dataset 2 has no DOI, so tables were linked to Dataset 1 by normalised caption (123 of 200 match "
  "with a single DOI). Six tables were processed (four composition tables and two non-composition "
  "tables); the replies were written before any ground truth was read, and relevance was judged "
  "correctly on all six.")
P(f"**Two comparisons, two table sets.** Against the *manual ground truth* (Dataset 2), "
  f"{fp['standard_total_exact']} of {fp['standard_total_cells']} cells agree exactly, with none missed or "
  f"invented; this uses rows 191, 107 and 1. Against the *authors’ pipeline output* (Dataset 1), "
  f"{v1['cells_exact']} of {v1['cells_both_present']} cells agree; this uses rows 191, 105 and 1, because "
  f"row 107 has no pairable rows (Dataset 1 collapsed its 14 cements into five designations). "
  f"The two counts therefore overlap in rows 191 and 1 only and should not be compared with each other. "
  f"The single disagreement in either is the MgO of sample SNM 7: the source table gives 1.57, while "
  f"both the manual ground truth and Dataset 1 record 1.87. Material typing agreed with the manual "
  f"top-level type on {d3['n_agree']} of {d3['n']} unseen Dataset 3 rows ({100 * d3['accuracy']:.0f} %; "
  f"paper: 97 %), indicative only because Dataset 3 labels are free text "
  f"({d3['n_distinct_full_labels']} distinct values), not the predefined types.")
caption("Extraction of the four composition tables with the substitute model. “Cells” are (material, "
        "oxide) pairs with a value on both sides; the two comparisons use different tables (see text).",
        "Table", 2)
lab = {191: "Mortar samples", 107: "Normal cements", 105: "Cement kiln dust", 1: "Cement and fly ashes"}
r2 = []
for rid in (191, 107, 105, 1):
    s_ = std[std.d2_row == rid].iloc[0]
    c_ = cmp1[cmp1.d2_row == rid].iloc[0]
    gt = ("not scorable†" if rid == 105 else f"{int(s_.cells_exact)}/{int(s_.cells_both_present)}")
    r2.append([f"Row {rid}: {lab[rid]}", int(s_.n_pred_materials), gt,
               f"{int(c_.n_shared)} of {int(c_.n_dataset1)}", f"{int(c_.cells_exact)}/{int(c_.cells_both_present)}"])
r2.append(["Total", tot_extr, f"{fp['standard_total_exact']}/{fp['standard_total_cells']}",
           f"{v1['materials_shared']} of {v1['materials_dataset1']}", f"{v1['cells_exact']}/{v1['cells_both_present']}"])
table(["Table (Dataset 2 row)", "Materials extracted", "Exact cells vs. manual ground truth",
       "Pairable with authors’ output", "Exact cells vs. authors’ output"], r2,
      [4.4, 2.2, 3.2, 2.9, 3.2], size=8.5, shade_rows=(4,))
P("† The Dataset 2 ground truth for row 105 describes a different table (a single “Cement” row).", size=9)
P(f"**Worked example.** Table text and manual values are from Dataset 2 (row 1, which has no DOI); the "
  f"DOI, category and the authors’ output are from the five Dataset 1 rows with the same caption, which "
  f"cite Tokyay [6]; title and journal are from Crossref. The author confirmed that the table appears "
  f"in that paper. **HL1** is followed through Tables 4–5 using existing records: all nine measured "
  f"oxides match Dataset 2 and Dataset 1 exactly, and the assigned type (fly ashes) matches Dataset 1.")
caption("Worked example: HL1, a high-lime fly ash, Tokyay [6]; DOI (from Dataset 1) ", "Table", 4)
cp_ = doc.paragraphs[-1]
hyperlink(cp_, "https://doi.org/" + WE["doi"], WE["doi"].upper(), 9.5)
rich(cp_, ". Oxide contents in wt %; – = not reported in the table; italic = imputed.")
fmt = lambda v: "–" if v is None else f"{v:g}"
ox_head = ["SiO₂", "Al₂O₃", "Fe₂O₃", "CaO", "MgO", "SO₃", "TiO₂", "MnO", "K₂O", "Na₂O", "LOI"]
OXK = ["SiO2", "Al2O3", "Fe2O3", "CaO", "MgO", "SO3", "TiO2", "MnO", "K2O", "Na2O", "LOI"]
mi = WE["model_input"] if "model_input" in WE else WE["model_inputs"]
meas = WE["measured"]
model_row = [(f"{mi[o]:.2f}".rstrip("0").rstrip(".") if meas[o] else f"*{mi[o]:.2f}*") for o in OXK]
rows4 = [["Extracted (Claude Opus 5)", *[fmt(WE["extracted"][o]) for o in OXK]],
         ["Manual ground truth (Dataset 2)", *[fmt(WE["dataset2_manual"][o]) for o in OXK]],
         ["Authors’ output (Dataset 1)", *[fmt(WE["dataset1"][o]) for o in OXK]],
         ["Model input after imputation", *model_row]]
table(["Source"] + ox_head, rows4, [4.5] + [1.04] * 11, size=8)
caption("Assumptions, imputation and predictions for HL1. Parentheses: the same ensemble’s prediction for "
        "the authors’ Dataset 1 row (identical measured inputs; the small difference comes from imputing "
        "within a different pool of rows).", "Table", 5)
pm, pd1 = WE["predictions"]["mine"], WE["predictions"]["dataset1_row"]
rows5 = [
    ["Material type", f"{WE['my_category']} (Dataset 1: {WE['dataset1_category']})",
     "assigned from name, caption and descriptor before any label was read"],
    ["Specific gravity", f"{mi['S gravity']:.2f} g/cm³ (imputed)",
     f"not extracted; the table lists {WE['table_density_g_cm3_not_extracted']} g/cm³, unused"],
    ["Amorphous content", f"{mi['Amorphous content']:.1f} % (imputed)", "absent from the table; LightGBM imputation"],
    ["Fixed inputs", f"d50 {mi['d50']:g} μm; Ca(OH)₂ 3, CaCO₃ 0, water 3.6, K₂SO₄ 0, KOH 0.151 per g; 50 °C; 10 d",
     "notebook-10 constants, not the table"],
    ["Predicted heat release, Ca(OH)₂ consumption, bound water",
     f"{pm['heat']:.1f} J/g, {pm['CH']:.1f} g/100 g, {pm['BW']:.1f} g/100 g "
     f"({pd1['heat']:.1f}, {pd1['CH']:.1f}, {pd1['BW']:.1f})", "11-seed tuned ensemble mean"],
]
table(["Item", "Value", "Basis"], rows5, [4.3, 6.6, 5.0], size=8, left_cols=3)
picture(os.path.join(FIG, "fig_workflow.png"), 12.5)
caption("Extraction-to-prediction workflow used in the demonstration; Tables 4-5 follow HL1 through it.", "Figure", 2)
H2("4.2 Literature materials and natural-rock screening")
P(f"For the 14,429 literature materials the ensemble reproduces the qualitative ordering reported by "
  f"the paper: metakaolin has the highest median heat release ({cat.loc['Metakaolin','heat_median']:.0f} J/g; "
  f"median Ca(OH)₂ consumption {cat.loc['Metakaolin','CH_median']:.0f} g/100 g), fly ashes are pozzolanic "
  f"(median {cat.loc['Fly ashes','CH_median']:.0f} g/100 g; {cat.loc['Fly ashes','pct_CH_above_50']:.0f} % above "
  f"50 g/100 g), slags and cements show lower Ca(OH)₂ consumption (medians "
  f"{cat.loc['Slags','CH_median']:.0f} and {cat.loc['Cements','CH_median']:.0f} g/100 g), and lime species "
  f"have the lowest median heat release among the categories discussed ({cat.loc['Lime species','heat_median']:.0f} J/g). The authors’ count of materials above 380 J/g with "
  f"bound water above 10 g/100 g (964) is untestable without their weights; here it is "
  f"{n(nb10['check1']['ensemble'])}, ranging from {n(min(nb10['check1']['per_seed']))} to "
  f"{n(max(nb10['check1']['per_seed']))} across seeds. {nb10['out_of_range']['n_above_dataset4_max']} "
  f"predictions exceed the largest training heat release.")
P(f"The notebook as released, scored with the ensemble over all 1,022,092 samples, calls "
  f"{n(base['reactive_all'])} samples reactive (> 200 J/g) against the paper’s 50,569, varying from "
  f"{n(min(seedc))} to {n(max(seedc))} between seeds. In the ablation (Table 3) none of the eight "
  f"combinations of pipeline choices reproduces the paper’s figure, and the pozzolanic/hydraulic "
  f"split depends on a Ca(OH)₂ threshold that the released code does not define.")
caption("Rock samples predicted reactive (> 200 J/g) by the 11-seed ensemble over all 1,022,092 samples, "
        "for all eight combinations of three pipeline choices (Dataset 5 in the imputation pool, "
        "Fe₂O₃ as total iron, R³ standard mix instead of the notebook constants). The paper reports "
        "50,569. Counts restricted to samples with SiO₂, Al₂O₃ and CaO, and single-model counts, are in "
        "the repository (results/nb11_sound_tuned_ablation.csv).", "Table", 3)
labs = [("As released (none)", "base"), ("Dataset 5", "d5"), ("Total iron", "fe"),
        ("Dataset 5 + total iron", "d5+fe"), ("R³ standard mix", "mix"),
        ("Dataset 5 + R³ mix", "d5+mix"), ("Total iron + R³ mix", "fe+mix"),
        ("Dataset 5 + total iron + R³ mix", "d5+fe+mix")]
rr = []
for lbl, v in labs:
    cnt = abl_e.loc[v, "reactive"]
    rr.append([lbl, n(cnt), f"{100 * (cnt - 50569) / 50569:+.0f} %".replace("-", "−")])
table(["Change to the notebook", "Reactive samples", "vs. paper"], rr, [8.6, 3.6, 2.8], size=8.5)

# ============================================================ PAGE 5: reproducibility assessment
H1("5. Reproducibility Assessment and Discussion")
bullets([
    "**Early stopping.** The notebook monitors a normalised RMSE for the Ca(OH)₂ head, averaged over "
    "validation batches of 50 rows. With 192 labels in 1,849 rows most batches hold none or one label, "
    "so the metric is NaN or of order 10⁶ and the restored epoch is not meaningful.",
    "**Splitting and preprocessing.** The stratification key is built on unstandardised targets but "
    "applied to standardised ones (six classes), the scaler and target standardisation are fitted on all "
    "rows before the split, and validation is the last 15 % of the training array.",
    "**Provenance.** The saved notebook records about 1,050 executions run out of order, and the "
    "model-selection protocol is not specified; Keras Tuner is cited but no tuner call is present.",
    "**Rock screening.** Notebook 11 scores every row (including 328,466 with no measured oxide), has no "
    "pozzolanic/hydraulic rule or rock-name grouping, and does not use Dataset 5; its mix constants "
    "differ from the R³ mix in Dataset 4.",
    "**Published datasets.** Dataset 2 row 105 has a ground truth from another table, row 191 has "
    "unquoted commas in material names, Dataset 3 labels are free text, and the row counts of Datasets 3 "
    "and 4 differ from the Supplementary Information.",
], size=10.5)
P(f"**Published versus reproduced.** Heat release is reproduced and the other two targets are "
  f"partially reproduced under the modified protocol; no faithful run reaches the published "
  f"Ca(OH)₂ value. Counts that depend on the authors’ weights (964 materials; 50,569 rocks) cannot "
  f"be tested. In the rock ablation, adding Dataset 5 to the imputation pool lowers the imputed "
  f"amorphous content of rocks from {amo['mean']:.1f} % to {ab['d5']['amorphous']['mean']:.1f} % (the "
  f"measured rocks average {amo['dataset5_measured_mean']:.1f} %) and is the single change that lowers "
  f"the count most; the R³ mix multiplies it by about "
  f"{abl_e.loc['mix','reactive'] / abl_e.loc['base','reactive']:.0f}. These show the count’s "
  f"sensitivity, not the authors’ procedure.")
P(f"**Grouped validation.** When no material appears in both training and test, the ensemble gives "
  f"*R*² = {f3(g['heat release']['R2'])} (heat release), {f3(g['CH consumption']['R2'])} "
  f"(Ca(OH)₂ consumption, 95 % CI [{f3(g['CH consumption']['R2_lo'])}, "
  f"{f3(g['CH consumption']['R2_hi'])}]) and {f3(g['Bound water']['R2'])} (bound water), while training "
  f"rows are still fitted at *R*² = {f3(grp['ensemble_train']['CH consumption']['R2'])} for Ca(OH)₂. "
  f"Dataset 4 has 1,849 rows but only 277 materials, mostly measured at several ages, so a row-level "
  f"split mainly measures interpolation between ages of materials already seen. The paper does not "
  f"state its split, so this measures how much the split matters, not an error in a published number.")
P(f"**Limitations.** (i) The effects of the individual protocol changes were not isolated. (ii) The "
  f"Ca(OH)₂ and bound-water test sets are small, so single *R*² values are noisy. (iii) The extractor "
  f"is a substitute and was run interactively, so no timing comparison is made. (iv) Python 3.8 was not "
  f"used; the library check changed heat-release *R*² from {f3(main42['test']['heat release']['R2'])} to "
  f"{f3(legacy['test']['heat release']['R2'])} but left Ca(OH)₂ unchanged "
  f"({f3(main42['test']['CH consumption']['R2'])} versus {f3(legacy['test']['CH consumption']['R2'])}). "
  f"(v) A clean-start second run reproduced the headline outputs identically; not every run was repeated.")

# ============================================================ PAGE 6: conclusions
H1("6. Conclusions and Future Research")
P(f"The released code runs, but single runs of notebook 9 are not reliable: its early-stopping "
  f"criterion collapsed in all {len(faith)} faithful runs. With a protocol that evaluates validation "
  f"loss over the whole validation set and avoids preprocessing leakage, the model reaches the "
  f"published heat-release accuracy and approaches the published Ca(OH)₂ and bound-water accuracy "
  f"within the uncertainty of very small test sets. That accuracy does not carry over to materials "
  f"absent from training, and the published rock-screening totals could not be reproduced from the "
  f"released files. The literature-extraction step transferred well to a substitute model on the "
  f"tables that could be scored.")
P("**Potential future research.** The following are directions suggested by this work, not experiments "
  "that were carried out.")
bullets([
    "Screen alternative precursors with validation on held-out material families, reporting grouped "
    "alongside random-split accuracy.",
    "Predict reactivity from oxide composition and descriptors with uncertainty estimates, flagging "
    "out-of-range extrapolations.",
    "Add phase-assemblage descriptors (e.g. QXRD phase fractions) to study how mineralogy and chemistry "
    "jointly control hydration performance.",
    "Rank industrial by-products and natural rocks as supplementary cementitious materials once "
    "composition coverage, amorphous content and the pozzolanic/hydraulic criterion are documented.",
    "Combine the model with experimental characterisation, using model disagreement to choose which "
    "materials to test next.",
], size=10.5)
H1("Code and Data Availability")
P("The original implementation is available at the Olivetti Group GitHub repository: [[" + ORIG_URL + "|"
  + ORIG_URL + "]] (replication used commit " + ORIG_COMMIT[:7] + "). The scripts, modified implementations, "
  "computational outputs and supporting documentation developed for this replication study are "
  + ("available in the accompanying replication repository: [[" + REPL_URL + "|" + REPL_URL + "]]. "
     if REPL_URL else "prepared for the accompanying replication repository, whose URL is "
     + REPL_PLACEHOLDER + " (not yet published at the time of writing). ")
  + "Certain original datasets, trained model weights and private extraction resources were "
  "unavailable; their limitations and the corresponding reconstruction or substitution procedures "
  "are documented in this report. The datasets themselves are not redistributed; the README gives "
  "their sources.")
H1("References")
refs = [
    "Mahjoubi, S., Venugopal, V., Manav, I. B., Azarijafari, H., Kirchain, R. & Olivetti, E. Data-driven "
    "material screening of secondary and natural cementitious precursors. *Commun. Mater.* **6**, 99 (2025). "
    "[[" + PAPER_URL + "|" + PAPER_URL + "]]",
    "Olivetti Group. *Cementitious-reactivity-prediction-and-discovery* (GitHub repository, commit "
    + ORIG_COMMIT[:7] + "). [[" + ORIG_URL + "|" + ORIG_URL + "]]",
    "Gard, M., Hasterok, D. & Halpin, J. A. Global whole-rock geochemical database compilation. "
    "*Earth Syst. Sci. Data* **11**, 1553–1566 (2019).",
    "O’Malley, T. et al. *KerasTuner* (2019). [[https://github.com/keras-team/keras-tuner|"
    "https://github.com/keras-team/keras-tuner]]",
    "Replication repository (this work): " + (f"[[{REPL_URL}|{REPL_URL}]]" if REPL_URL else REPL_PLACEHOLDER) + ".",
    "Tokyay, M. Strength prediction of fly ash concretes by accelerated testing. *Cem. Concr. Res.* "
    "**29**, 1737–1741 (1999). [[https://doi.org/" + WE["doi"] + "|doi:" + WE["doi"].upper() + "]] "
    "",
]
for i, rtxt in enumerate(refs, 1):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.7)
    p.paragraph_format.first_line_indent = Cm(-0.7)
    p.paragraph_format.space_after = Pt(1.5)
    p.paragraph_format.line_spacing = 1.0
    rich(p, f"[{i}] " + rtxt, size=9.5)

doc.core_properties.title = "Replication of Mahjoubi et al. (2025): code execution and reproducibility assessment"
doc.core_properties.author = ""
doc.save(args.out)
print("wrote", args.out)
