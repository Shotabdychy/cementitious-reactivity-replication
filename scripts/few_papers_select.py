"""PART 4.1 -- choose the tables for the 'few papers' extraction demonstration.

Dataset 2 holds the table text that the authors' pipeline fed to the LLM, and Dataset 1 holds
what came out of it, with a DOI and a caption. Dataset 2 has no DOI column, so the two are
linked here on the normalized table caption: the caption is the text of the Dataset 2 prompt
before the cell list, and Dataset 1 has the same caption for every material extracted from that
table. 123 of the 200 Dataset 2 tables match a Dataset 1 caption with a single DOI.

Working from the Dataset 2 table text means the extractor receives exactly the input the
notebook-4 pipeline receives, which the paper corpus (a private MongoDB) would otherwise be
needed for.

This script writes the prompts only. It deliberately does not write or print the Dataset 2
'Winner' column (the manual ground truth) or the Dataset 1 oxide values, so that the extraction
step cannot see the answer. Scoring happens afterwards, in few_papers_score.py.

usage: python few_papers_select.py
"""
import os
import re
import sys
import json

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, RESULTS

OUT = os.path.join(RESULTS, "few_papers")
os.makedirs(OUT, exist_ok=True)
PROMPTS = os.path.join(OUT, "prompts")
os.makedirs(PROMPTS, exist_ok=True)

# chosen in advance: the four caption-matched tables with the most materials per table, plus two
# Dataset 2 tables with no Dataset 1 match (Part 4.5, no replication claim possible for those).
MAIN = [191, 107, 105, 1]
EXTRA_WANTED = 2

norm = lambda s: re.sub(r'\s+', ' ', str(s)).strip()


def caption_of(prompt):
    s = str(prompt)
    i = s.find('[[')
    return norm(s[:i]) if i > 0 else norm(s[:120])


d2 = pd.read_csv(os.path.join(DATA, "Dataset 2.csv"))
d1 = pd.read_csv(os.path.join(DATA, "Dataset 1.csv"), low_memory=False)
d2["caption"] = [caption_of(p) for p in d2.Prompt]
d1["cap_n"] = [norm(s) for s in d1.Caption]

PROMPT_TEMPLATE = ("""CSV Format Requirements: Ensure the CSV includes columns for Material Name, SiO2, Al2O3, Fe2O3, CaO, MgO, SO3, TiO2, MnO, K2O, Na2O, LOI.
    Handling Missing Data: Use 'null' for any missing data in these columns.
    Non-relevant Data: If the table's content is not related to the specified chemicals, respond with 'NR' only.

    Unstructured HTML table: {table}""")
SYSTEM = "You are an assistant trained to parse unstructured HTML tables into CSV format."

matched = {}
for i, c in d2.caption.items():
    if not c:
        continue
    hit = d1[d1.cap_n == c]
    if len(hit) and hit.DOI.nunique() == 1:
        matched[i] = (c, str(hit.DOI.iloc[0]), len(hit))

unmatched = [i for i, c in d2.caption.items() if c and not len(d1[d1.cap_n == c])]
extra = [i for i in unmatched if 600 < len(str(d2.loc[i, "Prompt"])) < 1800][:EXTRA_WANTED]

records = []
for i in MAIN + extra:
    is_main = i in MAIN
    cap, doi, nmat = matched.get(i, (d2.caption[i], None, 0))
    rec = dict(d2_row=int(i), caption=cap, doi=doi, n_dataset1_materials=nmat,
               group="main" if is_main else "extra_no_dataset1_ground_truth",
               table_text=str(d2.loc[i, "Prompt"]))
    records.append(rec)
    with open(os.path.join(PROMPTS, f"d2row{i:03d}.txt"), "w", encoding="utf-8") as f:
        f.write("### SYSTEM\n" + SYSTEM + "\n\n### USER\n"
                + PROMPT_TEMPLATE.format(table=rec["table_text"]) + "\n")

json.dump(dict(prompt_system=SYSTEM, prompt_user_template=PROMPT_TEMPLATE,
               n_dataset2=len(d2), n_caption_matched_single_doi=len(matched),
               n_dataset2_without_dataset1_match=len(unmatched),
               selected=records),
          open(os.path.join(OUT, "selection.json"), "w"), indent=1)

print(f"Dataset 2 tables: {len(d2)}")
print(f"  caption-matched to Dataset 1 with a single DOI: {len(matched)}")
print(f"  no Dataset 1 caption match at all: {len(unmatched)}")
print(f"\nselected {len(records)} tables; prompts written to {PROMPTS}")
for r in records:
    print(f"  d2row {r['d2_row']:3d} [{r['group']}] doi={r['doi']} "
          f"dataset1_materials={r['n_dataset1_materials']} chars={len(r['table_text'])}")
    print(f"      caption: {r['caption'][:90]}")
