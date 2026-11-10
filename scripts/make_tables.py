"""PART 5.1 / 5.2 -- build the report tables from the result files.

Every number written here is read back out of results/*.json or results/*.csv; nothing is typed
in by hand except the paper's own reported values, which are marked as such.

usage: python make_tables.py
"""
import os
import sys
import json
import glob

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TARGETS, SHORT, PAPER, RESULTS, ROOT



def jload(p):
    return json.load(open(p)) if os.path.exists(p) else None


# ---------------------------------------------------------------- model accuracy table
rows = []


def faithful_block(variant, label):
    import re as _re
    fs = sorted(f for f in glob.glob(os.path.join(RESULTS, f"nb9_{variant}_seed*.json"))
                if _re.fullmatch(rf"nb9_{variant}_seed\d+\.json", os.path.basename(f)))
    if not fs:
        return
    runs = [json.load(open(f)) for f in fs]
    for only_good in (False, True):
        sub = [r for r in runs if (not r["collapsed"]) or not only_good]
        if only_good and len(sub) == len(runs):
            continue
        if not sub:
            continue
        rec = dict(row=label + (" (non-collapsed only)" if only_good else ""),
                   protocol="notebook as released", n_seeds=len(sub),
                   n_collapsed=sum(r["collapsed"] for r in runs) if not only_good else 0,
                   selection="validation only", split="row-level, notebook key, random_state 42")
        for t in TARGETS:
            r2 = np.array([r["test"][t]["R2"] for r in sub], dtype=float)
            rm = np.array([r["test"][t]["RMSE"] for r in sub], dtype=float)
            rec[f"{SHORT[t]}_R2_mean"] = float(np.nanmean(r2))
            rec[f"{SHORT[t]}_R2_sd"] = float(np.nanstd(r2, ddof=1)) if len(r2) > 1 else np.nan
            rec[f"{SHORT[t]}_R2_min"] = float(np.nanmin(r2))
            rec[f"{SHORT[t]}_R2_max"] = float(np.nanmax(r2))
            rec[f"{SHORT[t]}_RMSE_mean"] = float(np.nanmean(rm))
            rec[f"{SHORT[t]}_n_test"] = sub[0]["test"][t]["n"]
        rows.append(rec)


faithful_block("faithful", "Faithful notebook 9")
faithful_block("clean", "Clean variant (notebook monitor, no input leakage)")


def sound_block(tag, label, protocol):
    s = jload(os.path.join(RESULTS, f"nb9_{tag}.json"))
    if not s:
        return
    ps = s["per_seed_mean_sd"]
    rec = dict(row=label, protocol=protocol, n_seeds=len(s["seeds"]),
               n_collapsed=s["n_collapsed"], selection="validation only", split=s["split_how"])
    for t in TARGETS:
        rec[f"{SHORT[t]}_R2_mean"] = ps[t]["R2_mean"]
        rec[f"{SHORT[t]}_R2_sd"] = ps[t]["R2_sd"]
        rec[f"{SHORT[t]}_R2_min"] = ps[t]["R2_min"]
        rec[f"{SHORT[t]}_R2_max"] = ps[t]["R2_max"]
        rec[f"{SHORT[t]}_RMSE_mean"] = ps[t]["RMSE_mean"]
        rec[f"{SHORT[t]}_n_test"] = s["labels_test"][t]
    rows.append(rec)
    e = s["ensemble_test"]
    rec2 = dict(row=label + " -- ENSEMBLE", protocol=protocol, n_seeds=len(s["seeds"]),
                n_collapsed=s["n_collapsed"], selection="validation only", split=s["split_how"])
    for t in TARGETS:
        rec2[f"{SHORT[t]}_R2_mean"] = e[t]["R2"]
        rec2[f"{SHORT[t]}_R2_lo"] = e[t].get("R2_lo")
        rec2[f"{SHORT[t]}_R2_hi"] = e[t].get("R2_hi")
        rec2[f"{SHORT[t]}_RMSE_mean"] = e[t]["RMSE"]
        rec2[f"{SHORT[t]}_RMSE_lo"] = e[t].get("RMSE_lo")
        rec2[f"{SHORT[t]}_RMSE_hi"] = e[t].get("RMSE_hi")
        rec2[f"{SHORT[t]}_n_test"] = e[t]["n"]
    rows.append(rec2)


sound_block("sound_notebook", "Sound protocol, notebook hyperparameters",
            "whole-set validation monitor, random stratified validation split, train-only "
            "scaling and imputation")
sound_block("sound_tuned", "Sound protocol, tuned hyperparameters (5-fold CV on train)",
            "as above, hyperparameters from nb9_tune.py")
sound_block("grouped_material", "Grouped split by material name",
            "as sound_tuned, but no material appears in both train and test")
sound_block("grouped_doi", "Grouped split by source paper (DOI)",
            "as sound_tuned, but no paper appears in both train and test")
sound_block("sens_impute_all", "Sensitivity: notebook imputation over all rows",
            "as sound_tuned, feature imputation fitted on all rows")

paper_row = dict(row="PAPER (reported)", protocol="not reproducible from the released files",
                 n_seeds=None, n_collapsed=None, selection="not specified", split="not specified")
for t in TARGETS:
    paper_row[f"{SHORT[t]}_R2_mean"] = PAPER[t]["R2"]
    paper_row[f"{SHORT[t]}_RMSE_mean"] = PAPER[t]["RMSE"]
rows.append(paper_row)

acc = pd.DataFrame(rows)
acc.to_csv(os.path.join(RESULTS, "table_model_accuracy.csv"), index=False)

cols = ["row", "n_seeds", "n_collapsed"]
for t in TARGETS:
    cols += [f"{SHORT[t]}_R2_mean", f"{SHORT[t]}_R2_sd", f"{SHORT[t]}_RMSE_mean",
             f"{SHORT[t]}_n_test"]
print("=============== model accuracy (test set) ===============")
print(acc[[c for c in cols if c in acc.columns]].round(3).to_string(index=False))


# ---------------------------------------------------------------- summary / claim table
def fmt(x, nd=2):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:,.{nd}f}"


claims = []
sound = jload(os.path.join(RESULTS, "nb9_sound_tuned.json"))
grp = jload(os.path.join(RESULTS, "nb9_grouped_material.json"))
nb10 = jload(os.path.join(RESULTS, "nb10_sound_tuned.json"))
ab = jload(os.path.join(RESULTS, "nb11_sound_tuned_ablation.json"))
fp = jload(os.path.join(RESULTS, "few_papers", "scores_summary.json"))
fpc = jload(os.path.join(RESULTS, "few_papers", "d3_sample2_scores.json"))

if sound:
    for t in TARGETS:
        e = sound["ensemble_test"][t]
        covers = e["R2_lo"] <= PAPER[t]["R2"] <= e["R2_hi"]
        status = ("reproduced" if e["R2"] >= PAPER[t]["R2"] - 0.05 else
                  "partially reproduced" if covers or e["R2"] >= PAPER[t]["R2"] - 0.20 else
                  "not reproduced")
        claims.append(dict(
            claim=f"test R2 for {t}", paper=PAPER[t]["R2"],
            reproduced=round(e["R2"], 3),
            detail=f"ensemble of {len(sound['seeds'])} seeds, 95% CI "
                   f"[{e['R2_lo']:.3f}, {e['R2_hi']:.3f}], n={e['n']}",
            status=status,
            reason="" if status == "reproduced" else
                   (f"the 95% interval covers the paper's value; only {e['n']} labelled test "
                    f"rows" if covers else
                    f"only {e['n']} labelled test rows for this target" if e["n"] < 100 else
                    "see the model accuracy table")))
        claims.append(dict(
            claim=f"test RMSE for {t}", paper=PAPER[t]["RMSE"],
            reproduced=round(e["RMSE"], 2),
            detail=f"95% CI [{e['RMSE_lo']:.2f}, {e['RMSE_hi']:.2f}]",
            status=("reproduced" if e["RMSE"] <= PAPER[t]["RMSE"] * 1.15 else
                    "partially reproduced" if e["RMSE"] <= PAPER[t]["RMSE"] * 1.6 else
                    "not reproduced"), reason=""))
if grp and sound:
    for t in TARGETS:
        drop = sound["ensemble_test"][t]["R2"] - grp["ensemble_test"][t]["R2"]
        claims.append(dict(
            claim=f"{t}: R2 when no material is shared between train and test",
            paper=None, reproduced=round(grp["ensemble_test"][t]["R2"], 3),
            detail=f"random-split ensemble {sound['ensemble_test'][t]['R2']:.3f} -> grouped "
                   f"{grp['ensemble_test'][t]['R2']:.3f} (drop {drop:.3f})",
            status="not testable",
            reason="the paper does not state how its split was formed; reported because the "
                   "stated purpose is screening materials the model has not seen"))
if nb10:
    c1 = nb10["check1"]
    claims.append(dict(claim="materials with heat release > 380 J/g and bound water > 10",
                       paper=c1["authors"], reproduced=c1["ensemble"],
                       detail=f"per-seed range {min(c1['per_seed'])}-{max(c1['per_seed'])}",
                       status="not reproduced",
                       reason="depends on the authors' trained weights, which are not released"))
if ab:
    base = [r for r in ab["base"]["rows"] if r["model"] == "ensemble"][0]
    claims.append(dict(claim="natural rock samples predicted reactive (> 200 J/g)",
                       paper=50569, reproduced=base["reactive_all"],
                       detail=f"notebook as released, all {ab['base']['n_rows']:,} rows; "
                              f"closest ablation combination "
                              f"{ab['closest_to_paper'][0]['reactive']:,} "
                              f"({ab['closest_to_paper'][0]['variant']}, "
                              f"{ab['closest_to_paper'][0]['row_filter']})",
                       status="not reproduced",
                       reason="the row filter, the rock-name grouping and the "
                              "pozzolanic/hydraulic rule are not in the released code"))
if fp:
    SUB = "substitute extractor (Claude Opus 5); the authors' fine-tuned endpoint is private"
    claims.append(dict(claim="table-extraction relevance detection (Table G3, F1)",
                       paper=0.95, reproduced=round(fp["notebook5_mean_f1_relevance"], 2),
                       detail=f"{fp['n_tables']} tables, notebook-5 metric",
                       status="reproduced with a substitute model", reason=SUB))
    claims.append(dict(claim="oxide values extracted from the table text",
                       paper=None,
                       reproduced=f"{fp['standard_total_exact']}/{fp['standard_total_cells']}",
                       detail="cells agreeing exactly with the Dataset 2 manual ground truth (3 of the 4 composition tables; row 105's ground truth belongs to another table)",
                       status="reproduced with a substitute model", reason=SUB))
    v = fp.get("vs_dataset1", {})
    if v:
        claims.append(dict(claim="agreement with the Dataset 1 rows for the same tables",
                           paper=None,
                           reproduced=f"{v['cells_exact']}/{v['cells_both_present']}",
                           detail=f"{v['materials_shared']} materials paired out of "
                                  f"{v['materials_mine']} extracted",
                           status="reproduced with a substitute model", reason=SUB))
if fpc:
    claims.append(dict(claim="material-type classification accuracy",
                       paper=0.97, reproduced=round(fpc["accuracy"], 2),
                       detail=f"{fpc['n_agree']}/{fpc['n']} unseen Dataset 3 rows, top-level type",
                       status="partially reproduced",
                       reason="substitute classifier, and Dataset 3 labels are hierarchical free "
                              f"text ({fpc['n_distinct_full_labels']} distinct labels, "
                              f"{fpc['n_distinct_top_level']} distinct top-level strings), not "
                              "the 19 predefined types the Supplementary Information describes"))

cl = pd.DataFrame(claims)
cl.to_csv(os.path.join(RESULTS, "table_claims.csv"), index=False)
print("\n=============== claim-by-claim summary ===============")
print(cl.to_string(index=False, max_colwidth=46))

# ---------------------------------------------------------------- Part 1.9 library-version check
leg = []
for f in sorted(glob.glob(os.path.join(RESULTS, "nb9_faithful_seed*_legacy.json"))):
    s = json.load(open(f))
    base = os.path.join(RESULTS, f"nb9_faithful_seed{s['seed']}.json")
    b = jload(base)
    row = dict(seed=s["seed"],
               env_legacy=f"py {s.get('python')} / tf {s.get('tensorflow')} / pandas {s.get('pandas')}",
               env_main=f"py {b.get('python')} / tf {b.get('tensorflow')} / pandas {b.get('pandas')}"
               if b else None)
    for t in TARGETS:
        row[f"{SHORT[t]}_R2_legacy"] = round(s["test"][t]["R2"], 4)
        row[f"{SHORT[t]}_R2_main"] = round(b["test"][t]["R2"], 4) if b else None
    row["best_epoch_legacy"] = s["best_epoch"]
    row["best_epoch_main"] = b["best_epoch"] if b else None
    row["collapsed_legacy"] = s["collapsed"]
    row["collapsed_main"] = b["collapsed"] if b else None
    leg.append(row)
if leg:
    lg = pd.DataFrame(leg)
    lg.to_csv(os.path.join(RESULTS, "table_library_version_check.csv"), index=False)
    print("\n=============== Part 1.9: library-version check (faithful notebook 9) ===============")
    print(lg.to_string(index=False))

# ---------------------------------------------------------------- runtime table
rt = []
for p in sorted(glob.glob(os.path.join(RESULTS, "nb9_*.json")) +
                glob.glob(os.path.join(RESULTS, "nb10_*.json")) +
                glob.glob(os.path.join(RESULTS, "nb11_*.json"))):
    s = jload(p)
    if isinstance(s, dict) and "secs" in s:
        rt.append(dict(step=os.path.basename(p).replace(".json", ""),
                       wall_seconds=round(s["secs"], 1)))
tj = jload(os.path.join(RESULTS, "nb9_tuned_config.json"))
if tj:
    import re as _re2
    secs = None
    lp = os.path.join(ROOT, "logs", "nb9_tune.log")
    if os.path.exists(lp):
        m = _re2.findall(r"total (\d+)s", open(lp, encoding="utf-8", errors="replace").read())
        secs = float(m[-1]) if m else None
    rt.append(dict(step=f"nb9_tune ({tj['trials']} trials x {tj['folds']} folds)",
                   wall_seconds=secs))
pd.DataFrame(rt).to_csv(os.path.join(RESULTS, "table_runtime.csv"), index=False)
print(f"\nwrote results/table_model_accuracy.csv, results/table_claims.csv, "
      f"results/table_runtime.csv")
