"""Worked example for the report: one material from one processed table, end to end.

Uses only records that already exist: the substitute extractor's reply, the Dataset 2 manual
ground truth, the Dataset 1 row, the model inputs saved by few_papers_predict.py, and the ensemble
predictions. Nothing is re-extracted or re-predicted here.

The bibliographic record (title, author, journal, volume, pages) is NOT in any dataset; it was
read from the public Crossref record of the DOI and is stored below as constants with that source
named, so it can be checked.

usage: python worked_example.py [--row 1 --material HL1]
"""
import os
import io
import re
import sys
import json
import argparse

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, RESULTS, FEATURES, OXIDES, jdump, key_name

ap = argparse.ArgumentParser()
ap.add_argument("--row", type=int, default=1)
ap.add_argument("--material", default="HL1")
args = ap.parse_args()

# Crossref record of the DOI (retrieved 2026-11-10); not part of the supplementary datasets.
REFERENCES = {
    "10.1016/s0008-8846(99)00160-x": dict(
        authors="M. Tokyay", year=1999,
        title="Strength prediction of fly ash concretes by accelerated testing",
        journal="Cement and Concrete Research", volume="29", issue="11", pages="1737–1741",
        source="Crossref record of the DOI, retrieved 2026-11-10"),
}

OUT = os.path.join(RESULTS, "few_papers")
sel = json.load(open(os.path.join(OUT, "selection.json")))
rec = [r for r in sel["selected"] if r["d2_row"] == args.row][0]
doi, caption = rec["doi"], rec["caption"]
name = args.material
norm = lambda s: re.sub(r"\s+", " ", str(s)).strip()

# ---- extracted (substitute extractor)
txt = open(os.path.join(OUT, "responses", f"d2row{args.row:03d}.csv"), encoding="utf-8").read()
mine = pd.read_csv(io.StringIO(txt.replace("null", "")))
mine.columns = [c.strip() for c in mine.columns]
mine["Material Name"] = mine["Material Name"].astype(str).str.strip()
for _o in OXIDES:
    mine[_o] = pd.to_numeric(mine[_o], errors="coerce")
mrow = mine[mine["Material Name"] == name].iloc[0]
extracted = {o: (None if pd.isna(mrow[o]) else float(mrow[o])) for o in OXIDES}

# ---- Dataset 2 manual ground truth for the same table
d2 = pd.read_csv(os.path.join(DATA, "Dataset 2.csv"))
w = str(d2.loc[args.row, "Winner"]).replace("\\n", "\n").replace("NaN", "")
truth = pd.read_csv(io.StringIO(w))
truth.columns = [c.strip() for c in truth.columns]
truth["Material Name"] = truth["Material Name"].astype(str).str.strip()
for _o in OXIDES:
    truth[_o] = pd.to_numeric(truth[_o], errors="coerce")
trow = truth[truth["Material Name"] == name].iloc[0]
manual = {o: (None if pd.isna(trow[o]) else float(trow[o])) for o in OXIDES}

# ---- Dataset 1 row (what the authors' pipeline produced)
d1 = pd.read_csv(os.path.join(DATA, "Dataset 1.csv"), low_memory=False)
d1["cap_n"] = [norm(s) for s in d1.Caption]
sub = d1[(d1.cap_n == caption) & (d1.DOI == doi) & (d1["Material Name"].astype(str).str.strip() == name)]
assert len(sub) == 1, len(sub)
drow = sub.iloc[0]
dataset1 = {o: (None if pd.isna(pd.to_numeric(drow[o], errors="coerce")) else float(drow[o])) for o in OXIDES}
d1_category = str(drow["Category"])

# ---- my category (assigned before any Category label was read)
cats = pd.read_csv(os.path.join(OUT, "my_categories.csv"))
my_cat = cats[(cats.d2_row == args.row) & (cats.material_name == name)].iloc[0].my_category

# ---- model inputs after imputation, and which were measured
desc = pd.read_csv(os.path.join(OUT, "imputed_descriptors.csv"))
dr = desc[(desc.d2_row == args.row) & (desc["Material Name"] == name)].iloc[0]
inputs = {f: float(dr[f]) for f in FEATURES}
measured = {f: bool(dr[f"measured_{f}"]) for f in FEATURES}

# ---- predictions
pr = pd.read_csv(os.path.join(OUT, "predictions_vs_dataset1_rows.csv"))
prow = pr[(pr.d2_row == args.row) & (pr.material == name)].iloc[0]
preds = dict(mine=dict(heat=float(prow.mine_heat), CH=float(prow.mine_CH), BW=float(prow.mine_BW)),
             dataset1_row=dict(heat=float(prow.d1_heat), CH=float(prow.d1_CH), BW=float(prow.d1_BW)))

# the table's own density row, quoted from the Dataset 2 prompt (not extracted by the pipeline)
m = re.search(r"\['Density \(g/cm3\)'((?:, '[^']*')+)\]", rec["table_text"])
density = None
if m:
    vals = re.findall(r"'([^']*)'", m.group(1))
    cols = re.findall(r"\['Oxide \(%\)'((?:, '[^']*')+)\]", rec["table_text"])
    header = re.findall(r"'([^']*)'", cols[0]) if cols else []
    if name in header:
        density = float(vals[header.index(name)])

ex = dict(provenance=dict(table_text_and_manual_values="Dataset 2 (no DOI column)", doi_category_authors_output="Dataset 1 rows with the same caption", bibliographic_details="Crossref record of the DOI", table_presence_in_paper="confirmed by the report author against the paper"), d2_row=args.row, material=name, doi=doi, reference=REFERENCES[doi], table_caption=caption,
          extracted=extracted, dataset2_manual=manual, dataset1=dataset1,
          exact_vs_manual=all(extracted[o] == manual[o] for o in OXIDES),
          exact_vs_dataset1=all(extracted[o] == dataset1[o] for o in OXIDES),
          my_category=my_cat, dataset1_category=d1_category,
          model_inputs=inputs, measured=measured, predictions=preds,
          table_density_g_cm3_not_extracted=density,
          constants=dict(d50_um_by_category="8.88 for 'Fly ashes' (notebook 10)",
                         mix="Ca(OH)2 3, CaCO3 0, water 3.6, K2SO4 0, KOH 0.151 per g; 50 C; 10 days"))
jdump(ex, os.path.join(OUT, "worked_example.json"))
print(json.dumps(ex, indent=1, default=str)[:2400])
