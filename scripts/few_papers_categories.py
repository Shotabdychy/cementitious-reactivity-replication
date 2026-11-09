"""PART 4.3 (second half) -- compare my material-type classification with the Category column
of Dataset 1, i.e. with the output of the authors' notebook-8 classifier.

My labels in results/few_papers/my_categories.csv were written from the notebook-8 prompt inputs
only (material name, table caption, and the Dataset 1 descriptor produced by notebook 7), with
the 21 category names that occur in Dataset 1 as the label set, before the Category column was
read.

usage: python few_papers_categories.py
"""
import os
import re
import sys
import json

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, RESULTS, jdump, key_name, pair_materials

OUT = os.path.join(RESULTS, "few_papers")
norm = lambda s: re.sub(r'\s+', ' ', str(s)).strip()
key = key_name

d1 = pd.read_csv(os.path.join(DATA, "Dataset 1.csv"), low_memory=False)
d1["cap_n"] = [norm(s) for s in d1.Caption]
sel = json.load(open(os.path.join(OUT, "selection.json")))
mine = pd.read_csv(os.path.join(OUT, "my_categories.csv"))

cats = sorted(d1.Category.dropna().unique())
print(f"category names present in Dataset 1: {len(cats)}")
print("  " + ", ".join(cats))
print("(the Supplementary Information describes 19 predefined cementitious material types)\n")

rows = []
for rec in sel["selected"]:
    if rec["doi"] is None:
        continue
    sub = d1[(d1.cap_n == rec["caption"]) & (d1.DOI == rec["doi"])].copy()
    sub["_k"] = [key(x) for x in sub["Material Name"]]
    mysub = mine[mine.d2_row == rec["d2_row"]].copy()
    mysub["_k"] = [key(x) for x in mysub.material_name]
    pairs, n_strict = pair_materials(list(dict.fromkeys([k for k in sub._k if k])),
                                     list(dict.fromkeys([k for k in mysub._k if k])))
    pmap = {b: a for a, b in pairs}
    strict = {a for a, b in pairs[:n_strict]}
    for _, m in mysub.iterrows():
        tk = pmap.get(m._k)
        if tk is None:
            rows.append(dict(d2_row=rec["d2_row"], material=m.material_name,
                             my_category=m.my_category, dataset1_category=None,
                             paired="no", agree=None))
            continue
        hit = sub[sub._k == tk].iloc[0]
        rows.append(dict(d2_row=rec["d2_row"], material=m.material_name,
                         my_category=m.my_category, dataset1_category=hit.Category,
                         paired="exact name" if tk in strict else "containment",
                         agree=bool(m.my_category == hit.Category)))

cmp = pd.DataFrame(rows)
cmp.to_csv(os.path.join(OUT, "category_comparison.csv"), index=False)
paired = cmp[cmp.paired != "no"]
print(cmp.to_string(index=False))
acc = float(paired.agree.mean()) if len(paired) else float("nan")
print(f"\npaired materials: {len(paired)} of {len(cmp)}")
print(f"agreement with the Dataset 1 Category column: {int(paired.agree.sum())}/{len(paired)} "
      f"= {100 * acc:.0f}%   (paper reports 97% classification accuracy)")

# table-level comparison, for the table whose material names could not be paired
tl = []
for rec in sel["selected"]:
    if rec["doi"] is None:
        continue
    sub = d1[(d1.cap_n == rec["caption"]) & (d1.DOI == rec["doi"])]
    mysub = mine[mine.d2_row == rec["d2_row"]]
    tl.append(dict(d2_row=rec["d2_row"], doi=rec["doi"],
                   dataset1_categories=sorted(sub.Category.dropna().unique().tolist()),
                   my_categories=sorted(mysub.my_category.unique().tolist()),
                   same_set=sorted(sub.Category.dropna().unique().tolist())
                   == sorted(mysub.my_category.unique().tolist())))
print("\nper-table category sets:")
for t in tl:
    print(f"  d2row {t['d2_row']}: Dataset 1 {t['dataset1_categories']} | mine "
          f"{t['my_categories']}  -> {'same' if t['same_set'] else 'different'}")

jdump(dict(n_categories_in_dataset1=len(cats), categories=cats,
           n_paired=len(paired), n_agree=int(paired.agree.sum()) if len(paired) else 0,
           agreement=acc, per_table=tl,
           paper_classification_accuracy=0.97,
           note=("The Supplementary Information states 19 predefined material types; the "
                 "Category column of Dataset 1 contains 21 distinct names.")),
      os.path.join(OUT, "category_scores.json"))
