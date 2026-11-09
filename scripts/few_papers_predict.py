"""PART 4.4 -- push the compositions I extracted through the notebook-9 ensemble and compare the
predicted reactivity with the predictions for the corresponding Dataset 1 rows.

The descriptor table is built with the notebook-10 constants, including its per-category d50, so
the only difference between my row and the authors' row for the same material is the extracted
composition itself (and, where the pipeline named the material differently, the row pairing).

usage: python few_papers_predict.py --models sound_tuned
"""
import os
import re
import io
import sys
import glob
import json
import pickle
import argparse

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (TARGETS, FEATURES, SHORT, DATA, RESULTS, ROOT, load_r3, impute_notebook,
                    CUSTOM_OBJECTS, unstandardize, jdump, key_name, pair_materials)

ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True)
ap.add_argument("--nb10-tag", default=None)
args = ap.parse_args()
NB10 = args.nb10_tag or args.models
MODELDIR = os.path.join(ROOT, "models", args.models)
OUT = os.path.join(RESULTS, "few_papers")

OX = ['SiO2', 'Al2O3', 'Fe2O3', 'CaO', 'MgO', 'SO3', 'TiO2', 'MnO', 'K2O', 'Na2O', 'LOI']
norm = lambda s: re.sub(r'\s+', ' ', str(s)).strip()
key = key_name

sel = json.load(open(os.path.join(OUT, "selection.json")))
cats = pd.read_csv(os.path.join(OUT, "my_categories.csv"))

rows = []
for rec in sel["selected"]:
    if rec["doi"] is None:
        continue
    i = rec["d2_row"]
    txt = open(os.path.join(OUT, "responses", f"d2row{i:03d}.csv"), encoding="utf-8").read()
    if txt.strip().upper() == "NR":
        continue
    d = pd.read_csv(io.StringIO(txt.replace("null", "")))
    d.columns = [c.strip() for c in d.columns]
    for c in d.columns:
        if c != "Material Name":
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d["d2_row"] = i
    d["doi"] = rec["doi"]
    d["caption"] = rec["caption"]
    rows.append(d)
mine = pd.concat(rows, ignore_index=True)
mine["Material Name"] = [norm(x) for x in mine["Material Name"]]
mine = mine.merge(cats[["d2_row", "material_name", "my_category"]],
                  left_on=["d2_row", "Material Name"], right_on=["d2_row", "material_name"],
                  how="left")
print(f"{len(mine)} extracted materials from {mine.d2_row.nunique()} tables")
print(mine.my_category.value_counts().to_string())

# ---------------------------------------------------------------- notebook-10 descriptor table
D50 = {'Cements': 14.11, 'Clinkers': 20, 'Fly ashes': 8.88, 'Slags': 6.15, 'Glasses': 24.99}


def build(df, category):
    t = pd.DataFrame(index=df.index)
    for c in OX:
        t[c] = df[c] if c in df.columns else np.nan
    t['S gravity'] = np.nan
    t['Amorphous content'] = np.nan
    t['d50'] = 10.0
    t['Ca(OH)2'] = 3
    t['CaCO3'] = 0
    t['Water'] = 3.6
    t['K2SO4'] = 0
    t['KOH'] = 0.151
    t['curing temp'] = 50
    t['age'] = 10
    for cat, v in D50.items():
        t.loc[category.values == cat, 'd50'] = v
    sf = category.values == 'Silica fume'
    t.loc[sf, 'd50'] = 0.2
    t.loc[sf, 'S gravity'] = 2.16
    t.loc[sf, 'LOI'] = 6.04
    return t[FEATURES]


mytab = build(mine, mine.my_category)
mymask = mytab.notnull().astype('float32').values

d1 = pd.read_csv(os.path.join(DATA, "Dataset 1.csv"), low_memory=False)
d1tab = pd.DataFrame({c: pd.to_numeric(d1[c], errors='coerce') if c in d1.columns else np.nan
                      for c in OX})
d1tab['S gravity'] = pd.to_numeric(d1['S gravity'], errors='coerce') if 'S gravity' in d1.columns else np.nan
d1tab['Amorphous content'] = np.nan
r3 = load_r3()
pool = pd.concat([mytab,
                  build(d1tab.assign(**{c: d1tab[c] for c in OX}), d1['Category']),
                  r3[FEATURES]], ignore_index=True)
for c in pool.columns:
    pool[c] = pd.to_numeric(pool[c], errors='coerce')
print(f"imputing {len(pool)} rows (my {len(mytab)} + Dataset 1 {len(d1tab)} + R3 {len(r3)})")
pool = impute_notebook(pool, log=lambda m: None)
pool[pool < 0] = 0
myimp = pool.iloc[:len(mytab)][FEATURES]

# keep the model inputs after imputation, and which of them were measured (present in the table)
_desc = pd.concat([mine[["d2_row", "Material Name", "my_category"]].reset_index(drop=True),
                   myimp.reset_index(drop=True)], axis=1)
for _i, _c in enumerate(FEATURES):
    _desc[f"measured_{_c}"] = mymask[:, _i].astype(int)
_desc.to_csv(os.path.join(OUT, "imputed_descriptors.csv"), index=False)

bundle = pickle.load(open(os.path.join(MODELDIR, "bundle.pkl"), "rb"))
scaler, mu, sd = bundle["scaler"], bundle["mu"], bundle["sd"]
Xs = scaler.transform(myimp).astype('float32')
paths = sorted(glob.glob(os.path.join(MODELDIR, "seed*.keras")))
acc = None
for p in paths:
    m = tf.keras.models.load_model(p, custom_objects=CUSTOM_OBJECTS)
    o = m.predict([Xs, mymask], verbose=0)
    o = np.stack([np.asarray(a).flatten() for a in o])
    acc = o if acc is None else acc + o
    tf.keras.backend.clear_session()
ens = acc / len(paths)
for i, t in enumerate(TARGETS):
    mine[f'pred_{SHORT[t]}'] = unstandardize(ens[i], mu[t], sd[t])

# ---------------------------------------------------------------- compare with the Dataset 1 rows
nb10 = pd.read_csv(os.path.join(RESULTS, f"nb10_{NB10}_predictions.csv"))
nb10["_k"] = [key(x) for x in nb10["Material Name"]]
d1cap = [norm(s) for s in d1.Caption]
nb10["cap_n"] = d1cap
mine["_k"] = [key(x) for x in mine["Material Name"]]

cmp_rows = []
for (d2row, caption, doi), grp in mine.groupby(["d2_row", "caption", "doi"], sort=False):
    sub = nb10[(nb10.cap_n == caption) & (nb10.DOI == doi)]
    pairs, n_strict = pair_materials(list(dict.fromkeys([k for k in sub._k if k])),
                                     list(dict.fromkeys([k for k in grp._k if k])))
    pmap = {b: a for a, b in pairs}
    strict = {a for a, b in pairs[:n_strict]}
    for _, r in grp.iterrows():
        tk = pmap.get(r._k)
        how = "unpaired" if tk is None else ("exact name" if tk in strict else "containment")
        row = dict(d2_row=r.d2_row, material=r["Material Name"], category=r.my_category,
                   paired=how, mine_heat=r.pred_heat, mine_CH=r.pred_CH, mine_BW=r.pred_BW)
        if tk is not None:
            h = sub[sub._k == tk].iloc[0]
            row.update(d1_heat=h['heat release'], d1_CH=h['CH consumption'],
                       d1_BW=h['Bound water'],
                       d_heat=r.pred_heat - h['heat release'],
                       d_CH=r.pred_CH - h['CH consumption'],
                       d_BW=r.pred_BW - h['Bound water'])
        cmp_rows.append(row)
cmp = pd.DataFrame(cmp_rows)
cmp.to_csv(os.path.join(OUT, "predictions_vs_dataset1_rows.csv"), index=False)
print("\npredicted reactivity of the materials I extracted, against the same ensemble's "
      "predictions for the Dataset 1 rows:")
print(cmp.round(1).to_string(index=False))

p = cmp[cmp.paired != "unpaired"].dropna(subset=["d_heat"])
summary = dict(n_extracted=len(cmp), n_paired=len(p),
               heat_mae=float(p.d_heat.abs().mean()) if len(p) else None,
               heat_max_abs=float(p.d_heat.abs().max()) if len(p) else None,
               CH_mae=float(p.d_CH.abs().mean()) if len(p) else None,
               BW_mae=float(p.d_BW.abs().mean()) if len(p) else None,
               n_same_reactivity_class=int(((p.mine_heat > 200) == (p.d1_heat > 200)).sum())
               if len(p) else None)
if len(p):
    print(f"\npaired {len(p)} materials: heat release MAE {summary['heat_mae']:.1f} J/g "
          f"(max {summary['heat_max_abs']:.1f}), Ca(OH)2 MAE {summary['CH_mae']:.1f}, "
          f"bound water MAE {summary['BW_mae']:.2f}; "
          f"{summary['n_same_reactivity_class']}/{len(p)} fall on the same side of 200 J/g")
jdump(summary, os.path.join(OUT, "predictions_summary.json"))
