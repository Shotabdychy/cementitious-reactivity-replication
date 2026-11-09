"""PART 2.1 - 2.2 -- notebook 10: reactivity of the literature-extracted materials (Dataset 1),
scored with the seed ensemble produced by nb9_final.py.

The descriptor table is built exactly as in notebook 10 (same constants, same per-category d50,
same silica-fume overrides, same mask-before-coercion order). Only the model and the scaler
change: predictions are the mean over the ensemble seeds, and the scaler / target statistics come
from the training rows of the notebook-9 split (never from test rows).

usage: python nb10_run.py --models sound_tuned [--tag sound_tuned]
"""
import os
import sys
import time
import glob
import pickle
import argparse

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (TARGETS, FEATURES, SHORT, DATA, RESULTS, ROOT, CACHE, load_r3,
                    impute_notebook, CUSTOM_OBJECTS, unstandardize, jdump)

ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True, help="folder name under models/")
ap.add_argument("--tag", default=None)
args = ap.parse_args()
TAG = args.tag or args.models
MODELDIR = os.path.join(ROOT, "models", args.models)
t0 = time.time()


def log(m):
    print(f"[{TAG} | {time.time() - t0:5.0f}s] {m}", flush=True)


bundle = pickle.load(open(os.path.join(MODELDIR, "bundle.pkl"), "rb"))
scaler, mu, sd = bundle["scaler"], bundle["mu"], bundle["sd"]
paths = sorted(glob.glob(os.path.join(MODELDIR, "seed*.keras")))
log(f"{len(paths)} ensemble members from models/{args.models}; "
    f"target statistics from the {len(bundle['tr'])} training rows")

# ---------------------------------------------------------------- notebook 10 cells 2-8
df = pd.read_csv(os.path.join(DATA, "Dataset 1.csv"), encoding="utf-8-sig", low_memory=False)
desired_columns = ['SiO2', 'Al2O3', 'Fe2O3', 'CaO', 'MgO', 'SO3', 'TiO2', 'MnO', 'K2O', 'Na2O',
                   'LOI', 'S gravity', 'Amorphous content']
new_df = pd.DataFrame()
src = {}
for column in desired_columns:
    if column in df.columns:
        new_df[column] = df[column]
        src[column] = column
    else:
        matches = [c for c in df.columns if c.lower() == column.lower()]
        new_df[column] = df[matches[0]] if matches else np.nan
        src[column] = matches[0] if matches else None
log(f"column mapping: {src}")
for c in new_df.columns:                      # pandas 3: keep object dtype for mixed columns
    if not pd.api.types.is_float_dtype(new_df[c]):
        new_df[c] = new_df[c].astype(object)
new_df['d50'] = 10.0
new_df['Ca(OH)2'] = 3
new_df['CaCO3'] = 0
new_df['Water'] = 3.6
new_df['K2SO4'] = 0
new_df['KOH'] = 0.151
new_df['curing temp'] = 50
new_df['age'] = 10
for cat, v in {'Cements': 14.11, 'Clinkers': 20, 'Fly ashes': 8.88, 'Slags': 6.15,
               'Glasses': 24.99}.items():
    new_df.loc[df['Category'] == cat, 'd50'] = v
sf = df['Category'] == 'Silica fume'
new_df.loc[sf, 'd50'] = 0.2
new_df.loc[sf, 'S gravity'] = 2.16
new_df.loc[sf, 'LOI'] = 6.04
input_mask = new_df.notnull().astype(int).values      # BEFORE coercion, as in the notebook

# ---------------------------------------------------------------- cells 9-14
cache = os.path.join(CACHE, "nb10_imputed.pkl")
if os.path.exists(cache):
    imputed_df = pickle.load(open(cache, "rb"))
    log(f"imputed literature table loaded from cache ({len(imputed_df)} rows)")
else:
    r3 = load_r3()
    Updated = pd.concat([new_df, r3[FEATURES]], ignore_index=True)
    for column in Updated.columns:
        Updated[column] = pd.to_numeric(Updated[column], errors='coerce')
    log(f"imputing {len(Updated)} rows ({len(df)} literature + {len(r3)} R3)")
    Updated = impute_notebook(Updated, log=lambda m: None)
    Updated[Updated < 0] = 0
    imputed_df = Updated.iloc[:len(df)][FEATURES]
    pickle.dump(imputed_df, open(cache, "wb"))
    log("imputation done and cached")

X_scaled = scaler.transform(imputed_df).astype('float32')
mask32 = input_mask.astype('float32')
r3 = load_r3()
heat_max_train = r3['heat release'].max()

# ---------------------------------------------------------------- predict, per seed and ensemble
per_seed = []
for p in paths:
    m = tf.keras.models.load_model(p, custom_objects=CUSTOM_OBJECTS)
    out = m.predict([X_scaled, mask32], batch_size=8192, verbose=0)
    per_seed.append([np.asarray(a).flatten() for a in out])
    tf.keras.backend.clear_session()
    log(f"  scored with {os.path.basename(p)}")
ens = [np.mean([s[i] for s in per_seed], axis=0) for i in range(3)]
hr, ch, bw = (unstandardize(ens[i], mu[t], sd[t]) for i, t in enumerate(TARGETS))

res = df[['Material Name', 'DOI', 'Category']].copy()
res['heat release'], res['CH consumption'], res['Bound water'] = hr, ch, bw
for i, t in enumerate(TARGETS):
    per = np.array([unstandardize(s[i], mu[t], sd[t]) for s in per_seed])
    res[f'{SHORT[t]}_seed_sd'] = per.std(axis=0, ddof=1) if len(per) > 1 else 0.0
res['heat_above_train_max'] = hr > heat_max_train
res.to_csv(os.path.join(RESULTS, f"nb10_{TAG}_predictions.csv"), index=False)

AUTH_COUNT = 964
AUTH_ROWS = {3: ('HAC', 527.6, 77.8, 39.4), 38: ('MK', 418.6, 101.1, 26.5),
             50: ('Residuals', 513.6, 123.8, 23.5), 55: ('Slag', 392.3, 42.3, 14.0),
             80: ('Granulated BFS', 399.7, 40.3, 14.0), 14393: ('MK [%]', 470.3, 109.6, 28.9),
             14409: ('Al-ULCC', 867.3, 136.8, 46.1), 14417: ('T75', 410.3, 18.3, 27.4),
             14420: ('T25', 381.8, 21.7, 24.1), 14421: ('Alumina A1000-SG', 780.6, 187.0, 50.0)}

summary = dict(tag=TAG, n_materials=len(df), n_models=len(paths),
               heat_max_in_dataset4=float(heat_max_train))

sel = int(((hr > 380) & (bw > 10)).sum())
per_seed_sel = [int(((unstandardize(s[0], mu[TARGETS[0]], sd[TARGETS[0]]) > 380) &
                     (unstandardize(s[2], mu[TARGETS[2]], sd[TARGETS[2]]) > 10)).sum())
                for s in per_seed]
log(f"\nCHECK 1  materials with heat release > 380 J/g and bound water > 10 g/100 g")
log(f"  ensemble {sel} | per-seed min {min(per_seed_sel)} max {max(per_seed_sel)} "
    f"mean {np.mean(per_seed_sel):.0f} | authors' saved output {AUTH_COUNT} | "
    f"ensemble/authors {100 * (sel - AUTH_COUNT) / AUTH_COUNT:+.1f}%")
summary["check1"] = dict(ensemble=sel, per_seed=per_seed_sel, authors=AUTH_COUNT)

log("\nrow-level comparison with the values left in the authors' saved notebook 10 "
    "(informative only: it depends on the authors' trained weights, which are not released)")
rows = []
for i, (nm, ah, ac, ab) in AUTH_ROWS.items():
    name_ok = str(df.loc[i, 'Material Name']) == nm
    rows.append(dict(index=i, name=nm, name_matches=bool(name_ok),
                     heat_mine=float(hr[i]), heat_authors=ah,
                     CH_mine=float(ch[i]), CH_authors=ac,
                     BW_mine=float(bw[i]), BW_authors=ab))
    log(f"  {i:6d} {nm:17s} name {'ok' if name_ok else 'MISMATCH'} | heat {hr[i]:7.1f}/{ah:6.1f} "
        f"| CH {ch[i]:6.1f}/{ac:6.1f} | BW {bw[i]:5.1f}/{ab:5.1f}")
summary["authors_rows"] = rows

# ---------------------------------------------------------------- category summary
res['reactive'] = res['heat release'] > 200
g = res.groupby('Category')
summ = pd.DataFrame({
    'n': g.size(),
    'heat_median': g['heat release'].median().round(1),
    'CH_median': g['CH consumption'].median().round(1),
    'BW_median': g['Bound water'].median().round(1),
    'heat_seed_sd_median': g['heat_seed_sd'].median().round(1),
    'CH_seed_sd_median': g['CH_seed_sd'].median().round(1),
    'pct_CH_above_50': (g['CH consumption'].apply(lambda s: (s > 50).mean()) * 100).round(0),
    'pct_heat_above_train_max': (g['heat_above_train_max'].mean() * 100).round(1),
}).sort_values('heat_median', ascending=False)
log("\ncategory medians (ensemble; seed_sd columns show the seed-to-seed spread):")
log(summ.to_string())
summ.to_csv(os.path.join(RESULTS, f"nb10_{TAG}_summary_by_category.csv"))

# ---------------------------------------------------------------- Part 2.2 qualitative claims
def med(cat, col):
    s = res.loc[res.Category == cat, col]
    return float(s.median()) if len(s) else float('nan')


claims = {}
claims["metakaolin among the most reactive"] = dict(
    rank_by_heat=int(summ.index.get_loc('Metakaolin') + 1) if 'Metakaolin' in summ.index else None,
    n_categories=len(summ),
    heat_median=med('Metakaolin', 'heat release'), CH_median=med('Metakaolin', 'CH consumption'))
claims["fly ashes pozzolanic (high Ca(OH)2 consumption)"] = dict(
    CH_median=med('Fly ashes', 'CH consumption'),
    pct_above_50=float((res.loc[res.Category == 'Fly ashes', 'CH consumption'] > 50).mean() * 100))
claims["slags hydraulic (low Ca(OH)2 consumption, high heat)"] = dict(
    CH_median=med('Slags', 'CH consumption'), heat_median=med('Slags', 'heat release'),
    pct_above_50=float((res.loc[res.Category == 'Slags', 'CH consumption'] > 50).mean() * 100))
claims["lime species inert"] = dict(
    heat_median=med('Lime species', 'heat release'), CH_median=med('Lime species', 'CH consumption'),
    pct_reactive=float((res.loc[res.Category == 'Lime species', 'heat release'] > 200).mean() * 100))
log("\nPart 2.2 qualitative claims of the paper:")
for k, v in claims.items():
    log(f"  {k}: { {kk: (round(vv, 1) if isinstance(vv, float) else vv) for kk, vv in v.items()} }")
summary["qualitative_claims"] = claims
summary["out_of_range"] = dict(
    n_above_dataset4_max=int(res['heat_above_train_max'].sum()),
    n_negative_heat=int((res['heat release'] < 0).sum()),
    n_total=len(res))
log(f"\npredictions above the maximum heat release in Dataset 4 ({heat_max_train:.0f} J/g): "
    f"{summary['out_of_range']['n_above_dataset4_max']} of {len(res)}; "
    f"negative heat predictions: {summary['out_of_range']['n_negative_heat']}")
summary["secs"] = time.time() - t0
jdump(summary, os.path.join(RESULTS, f"nb10_{TAG}.json"))
log("done")
