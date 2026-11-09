"""PART 2.3 - 2.5 -- notebook 11: natural rocks, with a one-factor-at-a-time ablation.

The authors' data/natural_rocks.csv is not in the repository, so the equivalent table is rebuilt
by merging the Gard et al. (2019) whole-rock database on its own keys, keeping all 1,022,092
samples, and the notebook's own column matching is then applied.

Ablation factors (baseline = the notebook as released):
  d5    : add Dataset 5 (measured amorphous content of ~160 rocks) to the imputation pool
  fe    : take Fe2O3 from 'fe2o3_tot' (total iron) instead of the notebook's exact match 'fe2o3'
  mix   : use the standard R3 mix of Dataset 4 instead of the notebook's constants
  rows  : which scored rows are counted (all / at least one measured oxide / SiO2+Al2O3+CaO)
  model : ensemble mean over seeds, or one single seed
'rows' and 'model' are applied at scoring time, so all of their combinations are reported for
every imputation variant.

usage:
  python nb11_ablation.py --models sound_tuned                 # all 8 imputation variants
  python nb11_ablation.py --models sound_tuned --only base     # just the baseline
"""
import os
import sys
import gc
import time
import glob
import pickle
import argparse
import itertools

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (TARGETS, FEATURES, OXIDES, SHORT, DATA, GARD, RESULTS, ROOT, CACHE,
                    load_r3, impute_notebook, CUSTOM_OBJECTS, unstandardize, jdump)

ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True)
ap.add_argument("--tag", default=None)
ap.add_argument("--only", default=None, help="comma-separated variant names, e.g. base,fe")
ap.add_argument("--paper-total", type=int, default=50569)
ap.add_argument("--limit", type=int, default=0,
                help="smoke test only: keep the first N rock samples instead of all 1,022,092")
args = ap.parse_args()
TAG = args.tag or args.models
MODELDIR = os.path.join(ROOT, "models", args.models)
PRED = os.path.join(CACHE, f"nb11_pred_{TAG}")
os.makedirs(PRED, exist_ok=True)
t0 = time.time()


def log(m):
    print(f"[{TAG} | {time.time() - t0:6.0f}s] {m}", flush=True)


# notebook 11 constants, and the standard R3 mix of Dataset 4 (per gram of SCM)
MIX_NOTEBOOK = dict(d50=10, **{'Ca(OH)2': 3, 'CaCO3': 0, 'Water': 3.6, 'K2SO4': 0, 'KOH': 0.151},
                    **{'curing temp': 50, 'age': 10})
MIX_R3STD = dict(d50=10, **{'Ca(OH)2': 3.0, 'CaCO3': 0.5, 'Water': 5.4, 'K2SO4': 0.108,
                            'KOH': 0.0216}, **{'curing temp': 40, 'age': 7})

VARIANTS = {}
for d5, fe, mix in itertools.product([False, True], [False, True], [False, True]):
    name = "base" if not (d5 or fe or mix) else "+".join(
        [n for n, f in (("d5", d5), ("fe", fe), ("mix", mix)) if f])
    VARIANTS[name] = dict(d5=d5, fe=fe, mix=mix)
if args.only:
    keep = [s.strip() for s in args.only.split(",")]
    VARIANTS = {k: v for k, v in VARIANTS.items() if k in keep}
log(f"variants to run: {list(VARIANTS)}")

# ---------------------------------------------------------------- merged rock table (cached)
raw_cache = os.path.join(CACHE, "gard_merged.pkl")
if os.path.exists(raw_cache):
    df = pickle.load(open(raw_cache, "rb"))
    log(f"merged Gard table from cache: {len(df)} samples")
else:
    s = pd.read_csv(os.path.join(GARD, "sample.csv"), encoding="latin-1", low_memory=False,
                    usecols=["sample_id", "rock_name", "latitude", "longitude", "density_model",
                             "major_id", "rgroup_id", "country_id", "ref_id"])
    m = pd.read_csv(os.path.join(GARD, "major.csv"), encoding="latin-1")
    rg = pd.read_csv(os.path.join(GARD, "rockgroup.csv"), encoding="latin-1")
    ct = pd.read_csv(os.path.join(GARD, "country.csv"), encoding="latin-1")
    ref = pd.read_csv(os.path.join(GARD, "reference.csv"), encoding="latin-1",
                      usecols=["ref_id", "author", "year", "data_source"])
    for nm, t, k in (("major", m, "major_id"), ("rockgroup", rg, "rgroup_id"),
                     ("country", ct, "country_id"), ("reference", ref, "ref_id")):
        assert t[k].is_unique, f"{nm}.{k} not unique"
    df = (s.merge(m, on="major_id", how="left").merge(rg, on="rgroup_id", how="left")
          .merge(ct, on="country_id", how="left").merge(ref, on="ref_id", how="left"))
    assert len(df) == len(s), (len(df), len(s))
    pickle.dump(df, open(raw_cache, "wb"))
    log(f"merged Gard table: {len(df)} samples, {df.shape[1]} columns (cached)")
if args.limit:
    df = df.head(args.limit).reset_index(drop=True)
    log(f"SMOKE TEST: limited to the first {len(df)} samples -- counts are not comparable "
        f"with the paper")
N = len(df)
log(f"first rows: {df.head(2)[['sample_id','rock_name','data_source']].to_dict('records')}")
log(f"last rows:  {df.tail(2)[['sample_id','rock_name','author','year']].to_dict('records')}")

bundle = pickle.load(open(os.path.join(MODELDIR, "bundle.pkl"), "rb"))
scaler, mu, sd = bundle["scaler"], bundle["mu"], bundle["sd"]
paths = sorted(glob.glob(os.path.join(MODELDIR, "seed*.keras")))
SINGLE = [p for p in paths if p.endswith("seed42.keras")] or paths[:1]
log(f"{len(paths)} ensemble members; single-model row uses {os.path.basename(SINGLE[0])}")

d5df = pd.read_csv(os.path.join(DATA, "Dataset 5.csv"), encoding="utf-8-sig")
d5df.columns = [c.strip() for c in d5df.columns]
D5_MAP = {'SiO2': 'SiO2', 'Al2O3': 'Al2O3', 'Fe2O3': 'Fe2O3', 'CaO': 'CaO', 'MgO': 'MgO',
          'SO3': 'SO3', 'TiO2': 'TiO2', 'MnO': 'MnO', 'K2O': 'K2O', 'Na2O': 'Na2O', 'LOI': 'LOI',
          'S gravity': 'S gravity', 'Amorphous content': 'Amorphous content'}
d5_meas = pd.to_numeric(d5df['Amorphous content'], errors='coerce')
log(f"Dataset 5: {len(d5df)} rows, {int(d5_meas.notna().sum())} with a measured amorphous content, "
    f"mean {d5_meas.mean():.1f}%")

r3 = load_r3()
r3_feats = r3[FEATURES]
log(f"R3 rows appended to the imputation pool: {len(r3_feats)}; measured amorphous content "
    f"mean {r3['Amorphous content'].mean():.1f}%")


def build_features(v):
    """notebook 11 cells 3-11, with the ablation switches applied."""
    new_df = pd.DataFrame(index=df.index)
    src = {}
    for column in FEATURES:
        if column == 'Fe2O3' and v["fe"]:
            new_df[column] = df['fe2o3_tot']
            src[column] = 'fe2o3_tot'
            continue
        if column in df.columns:
            new_df[column] = df[column]
            src[column] = column
        else:
            matches = [c for c in df.columns if c.lower() == column.lower()]
            new_df[column] = df[matches[0]] if matches else np.nan
            src[column] = matches[0] if matches else None
    new_df['S gravity'] = df['density_model'] / 1000
    src['S gravity'] = 'density_model/1000'
    mix = MIX_R3STD if v["mix"] else MIX_NOTEBOOK
    for k, val in mix.items():
        new_df[k] = val
        src[k] = f"constant {val}"
    new_df = new_df.clip(lower=0)
    for column in new_df.columns:
        new_df[column] = pd.to_numeric(new_df[column], errors='coerce')
    return new_df[FEATURES], src


summary = {}
for name, v in VARIANTS.items():
    vt = time.time()
    log(f"\n################ variant '{name}'  {v}")
    feats, src = build_features(v)
    log("  feature <- source: " + ", ".join(f"{k}<-{s}" for k, s in src.items()
                                            if not str(s).startswith("constant")))
    mask = feats.notnull().astype(np.uint8).values
    ox_idx = [FEATURES.index(c) for c in OXIDES]
    n_ox = mask[:, ox_idx].sum(axis=1)
    any_ox = n_ox > 0
    has3 = mask[:, [FEATURES.index(c) for c in ('SiO2', 'Al2O3', 'CaO')]].all(axis=1)
    log(f"  rows with no measured oxide: {int((~any_ox).sum())}; with >=1: {int(any_ox.sum())}; "
        f"with SiO2+Al2O3+CaO: {int(has3.sum())}; mean oxides present {n_ox.mean():.2f}")

    pool = [feats, r3_feats]
    if v["d5"]:
        extra = pd.DataFrame({k: pd.to_numeric(d5df[c], errors='coerce')
                              for k, c in D5_MAP.items()})
        for k in FEATURES:
            if k not in extra.columns:
                extra[k] = np.nan
        pool.append(extra[FEATURES])
        log(f"  Dataset 5 added to the imputation pool ({len(extra)} rows)")
    allrows = pd.concat(pool, ignore_index=True).clip(lower=0)
    log(f"  imputing {len(allrows)} rows")
    allrows = impute_notebook(allrows, log=lambda m: None)
    imputed = allrows.iloc[:N][FEATURES]
    am = imputed['Amorphous content']
    amorph = dict(mean=float(am.mean()), median=float(am.median()),
                  p10=float(am.quantile(.1)), p90=float(am.quantile(.9)),
                  r3_measured_mean=float(r3['Amorphous content'].mean()),
                  dataset5_measured_mean=float(d5_meas.mean()))
    log(f"  imputed rock amorphous content: mean {amorph['mean']:.1f}% median {amorph['median']:.1f}% "
        f"(R3 measured {amorph['r3_measured_mean']:.1f}%, Dataset-5 measured "
        f"{amorph['dataset5_measured_mean']:.1f}%)")
    X = scaler.transform(imputed).astype('float32')
    del allrows, imputed
    gc.collect()
    maskf = mask.astype('float32')

    acc = [np.zeros(N, dtype=np.float64) for _ in range(3)]
    per_seed_counts = []
    single_pred = None
    for p in paths:
        m = tf.keras.models.load_model(p, custom_objects=CUSTOM_OBJECTS)
        out = m.predict([X, maskf], batch_size=16384, verbose=0)
        out = [np.asarray(a).flatten() for a in out]
        for i in range(3):
            acc[i] += out[i]
        hr_i = unstandardize(out[0], mu[TARGETS[0]], sd[TARGETS[0]])
        per_seed_counts.append(int((hr_i > 200).sum()))
        if p in SINGLE:
            single_pred = [unstandardize(out[i], mu[t], sd[t]) for i, t in enumerate(TARGETS)]
        tf.keras.backend.clear_session()
        del out
        gc.collect()
    ens = [unstandardize(acc[i] / len(paths), mu[t], sd[t]) for i, t in enumerate(TARGETS)]
    np.save(os.path.join(PRED, f"{name}_ens.npy"),
            np.vstack([a.astype('float32') for a in ens]))
    del X, maskf, acc
    gc.collect()

    def counts(pred, label):
        hr, ch = pred[0], pred[1]
        react = hr > 200
        row = dict(
            model=label,
            reactive_all=int(react.sum()),
            reactive_any_oxide=int((react & any_ox).sum()),
            reactive_has3=int((react & has3).sum()),
            reactive_no_oxide=int((react & ~any_ox).sum()),
            n_no_oxide=int((~any_ox).sum()),
            inert=int((hr <= 120).sum()),
            low=int(((hr > 120) & (hr <= 200)).sum()),
            pozz_CH50=int((react & (ch > 50)).sum()),
            hydr_CH50=int((react & (ch <= 50)).sum()),
            pozz_CH25=int((react & (ch > 25)).sum()),
            pozz_CH75=int((react & (ch > 75)).sum()),
            heat_mean=float(hr.mean()), heat_median=float(np.median(hr)),
            CH_mean=float(ch.mean()),
        )
        return row

    rows = [counts(ens, "ensemble")]
    if single_pred is not None and len(paths) > 1:
        rows.append(counts(single_pred, "single seed42"))
    for r in rows:
        log(f"  {r['model']:14s} reactive >200 J/g: all {r['reactive_all']:>7,} | "
            f">=1 oxide {r['reactive_any_oxide']:>7,} | SiO2+Al2O3+CaO {r['reactive_has3']:>7,} | "
            f"no oxide {r['reactive_no_oxide']:>7,}/{r['n_no_oxide']:,}  (paper {args.paper_total:,})")
    summary[name] = dict(factors=v, amorphous=amorph, rows=rows,
                         per_seed_reactive_all=per_seed_counts,
                         n_rows=N, n_any_oxide=int(any_ox.sum()), n_has3=int(has3.sum()),
                         secs=time.time() - vt)

    if name == "base":
        # ---- Part 2.5 threshold sweep and Fig-4 input, on the baseline variant
        hr, ch = ens[0], ens[1]
        sweep = []
        for thr in (0, 10, 25, 50, 75, 100):
            sweep.append(dict(CH_threshold=thr,
                              pozzolanic=int(((hr > 200) & (ch > thr)).sum()),
                              hydraulic=int(((hr > 200) & (ch <= thr)).sum())))
        summary["pozzolanic_threshold_sweep_base"] = sweep
        log("  Part 2.5 pozzolanic/hydraulic split of the reactive rocks "
            "(my rule, not in the released code); paper ~46,700 / ~3,800:")
        for s in sweep:
            log(f"    CH > {s['CH_threshold']:3d} g/100 g -> {s['pozzolanic']:>7,} pozzolanic / "
                f"{s['hydraulic']:>7,} hydraulic")
        rt = pd.DataFrame({'rock_name': df.rock_name, 'rock_group': df.rock_group,
                           'heat': hr, 'CH': ch, 'any_oxide': any_ox, 'has3': has3})
        rt['reactive'] = rt.heat > 200
        g = rt.groupby('rock_name').agg(n=('reactive', 'size'), n_reactive=('reactive', 'sum'),
                                        n_any_oxide=('any_oxide', 'sum'),
                                        heat_median=('heat', 'median'), heat_max=('heat', 'max'))
        g['ratio'] = g.n_reactive / g.n
        g.sort_values('n_reactive', ascending=False).to_csv(
            os.path.join(RESULTS, f"nb11_{TAG}_by_rock_name.csv"))
        big = g[g.n >= 20]
        log(f"  rock names: {len(g)} raw; {len(big)} with n>=20; of those "
            f"{int((big.ratio > 0.05).sum())} have more than 5% reactive "
            f"(paper: 25-38 types, grouping undocumented)")
        named = ['anorthosite', 'ignimbrite', 'porphyry', 'rhyolite', 'andesite', 'dacite',
                 'tuff', 'basalt', 'granite', 'shale']
        log("  types named in the paper, raw database names:\n" +
            g.reindex(named)[['n', 'n_reactive', 'ratio', 'n_any_oxide']].round(3).to_string())
        summary["rock_types_named"] = g.reindex(named).round(4).reset_index().to_dict("records")

    del ens, mask
    gc.collect()

# ---------------------------------------------------------------- ablation table
tab = []
for name, s in summary.items():
    if not isinstance(s, dict) or "rows" not in s:
        continue
    for r in s["rows"]:
        for rowfilter, k in (("all rows", "reactive_all"),
                             (">=1 measured oxide", "reactive_any_oxide"),
                             ("SiO2+Al2O3+CaO", "reactive_has3")):
            tab.append(dict(variant=name, **s["factors"], model=r["model"],
                            row_filter=rowfilter, reactive=r[k],
                            paper=args.paper_total,
                            pct_vs_paper=round(100 * (r[k] - args.paper_total) / args.paper_total, 1)))
tabdf = pd.DataFrame(tab)
tabdf.to_csv(os.path.join(RESULTS, f"nb11_{TAG}_ablation.csv"), index=False)
log("\n==================== ablation table (reactive = heat release > 200 J/g) ====================")
log(tabdf.to_string(index=False))
closest = tabdf.iloc[(tabdf.reactive - args.paper_total).abs().argsort()[:5]]
log("\nclosest five combinations to the paper's 50,569:")
log(closest.to_string(index=False))
summary["closest_to_paper"] = closest.to_dict("records")
summary["secs"] = time.time() - t0
jdump(summary, os.path.join(RESULTS, f"nb11_{TAG}_ablation.json"))
log("done")
