"""PART 4.3 -- score the 'few papers' extraction.

Two comparisons, kept separate:

  A. against Dataset 2's 'Winner' column (the authors' manual ground truth for the same table),
     using the metric code of notebook 5 re-implemented line for line, so the numbers are
     comparable with Supplementary Table G3 -- plus a standard set-based metric, because the
     notebook's metric has properties that are easy to misread (see the report).

  B. against the Dataset 1 rows for the same DOI and caption, i.e. what the authors' pipeline
     actually produced for that table: how many materials, how many oxide cells agree exactly,
     how many within 0.1 wt%, and a reason for every mismatch.

usage: python few_papers_score.py
"""
import os
import re
import io
import sys
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, RESULTS, jdump, key_name, pair_materials

OUT = os.path.join(RESULTS, "few_papers")
EXPECTED = ['Material Name', 'SiO2', 'Al2O3', 'Fe2O3', 'CaO', 'MgO', 'SO3', 'TiO2', 'MnO',
            'K2O', 'Na2O', 'LOI']
LOWER = [c.lower() for c in EXPECTED]
NUMCOLS = LOWER[1:]

sel = json.load(open(os.path.join(OUT, "selection.json")))
d2 = pd.read_csv(os.path.join(DATA, "Dataset 2.csv"))
d1 = pd.read_csv(os.path.join(DATA, "Dataset 1.csv"), low_memory=False)
norm = lambda s: re.sub(r'\s+', ' ', str(s)).strip()
d1["cap_n"] = [norm(s) for s in d1.Caption]


# ---------------------------------------------------------------- notebook 5 helpers (verbatim)
def parse_response_to_df(response, expected_columns):
    cleaned_response = str(response).replace("null", "NaN")
    cleaned_response = ''.join(map(str, cleaned_response))
    cleaned_response = (cleaned_response.strip('[').strip(']').strip('""').strip('''''')
                        .strip('\'"').replace('\\n', '\n').replace('\\', ''))
    expected_columns_lower = [col.lower() for col in expected_columns]
    new_df = pd.DataFrame(columns=expected_columns_lower)
    try:
        df = pd.read_csv(io.StringIO(cleaned_response))
        df.columns = [col.lower() for col in df.columns]
        df.columns = df.columns.str.strip()
        for exp_col in expected_columns_lower:
            new_df[exp_col] = df[exp_col] if exp_col in df.columns else pd.NA
    except Exception as e:
        print("  parse error:", e)
    return new_df


def check_relevance(resp):
    if isinstance(resp, list):
        resp = resp[0]
    resp = str(resp).strip('""/\\[]')
    return 0 if resp.lower().strip() == 'nr' else 1


def calculate_precision_recall_f1(true_labels, predicted_labels):
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels,
                                                               zero_division=0)
    return precision[0], recall[0], f1[0]


def safe_lower(x):
    try:
        return x.lower()
    except AttributeError:
        return x


def notebook5_metrics(Winner, Response):
    """Exactly the computation of notebook 5 cell 8, for one table."""
    out = dict(rel_true=check_relevance(Winner), rel_pred=check_relevance(Response))
    out["acc_rel"] = accuracy_score([out["rel_true"]], [out["rel_pred"]])
    p, r, f = calculate_precision_recall_f1([out["rel_true"]], [out["rel_pred"]])
    out.update(prec_rel=p, recall_rel=r, f1_rel=f)
    if not (out["rel_true"] and out["rel_pred"]):
        return out
    rdf = parse_response_to_df(Response, EXPECTED)
    wdf = parse_response_to_df(Winner, EXPECTED)
    for col in NUMCOLS:
        rdf[col] = pd.to_numeric(rdf[col], errors='coerce')
        wdf[col] = pd.to_numeric(wdf[col], errors='coerce')
    mt = [safe_lower(x) for x in wdf['material name'].tolist()]
    mp = [safe_lower(x) for x in rdf['material name'].tolist()]
    le = LabelEncoder().fit(mt + mp)
    et, ep = le.transform(mt), le.transform(mp)
    L = max(len(et), len(ep))
    et = np.pad(et, (0, L - len(et)), mode='constant')
    ep = np.pad(ep, (0, L - len(ep)), mode='constant')
    out["acc_mat"] = accuracy_score(et.tolist(), ep.tolist())
    p, r, f = calculate_precision_recall_f1(et.tolist(), ep.tolist())
    out.update(prec_mat=p, recall_mat=r, f1_mat=f)
    tv = wdf[NUMCOLS].fillna(0).values.flatten().tolist()
    pv = rdf[NUMCOLS].fillna(0).values.flatten().tolist()
    L = max(len(tv), len(pv))
    tv = np.pad(tv, (0, L - len(tv)), mode='constant').tolist()
    pv = np.pad(pv, (0, L - len(pv)), mode='constant').tolist()
    le = LabelEncoder().fit(tv + pv)
    out["acc_num"] = accuracy_score(le.transform(tv).tolist(), le.transform(pv).tolist())
    p, r, f = calculate_precision_recall_f1(le.transform(tv).tolist(), le.transform(pv).tolist())
    out.update(prec_num=p, recall_num=r, f1_num=f)
    out["rmse"] = float(np.sqrt(mean_squared_error(tv, pv)))
    out["n_true_materials"] = int(len(mt))
    out["n_pred_materials"] = int(len(mp))
    return out


# ---------------------------------------------------------------- standard, set-based metrics
def standard_metrics(Winner, Response):
    """Set-based metrics, stated separately from the notebook's.

    materials : precision/recall/F1 over the SET of material names, strict and after the
                containment fallback above.
    cells     : a cell is a (material, oxide) pair of a paired material. Counted as agreeing
                when both values are present and within 0.1 wt%; 'missed' when the ground truth
                has a value and the extraction does not; 'invented' for the reverse.
    """
    rdf = parse_response_to_df(Response, EXPECTED)
    wdf = parse_response_to_df(Winner, EXPECTED)
    for col in NUMCOLS:
        rdf[col] = pd.to_numeric(rdf[col], errors='coerce')
        wdf[col] = pd.to_numeric(wdf[col], errors='coerce')
    wdf['_k'] = [key_name(x) for x in wdf['material name']]
    rdf['_k'] = [key_name(x) for x in rdf['material name']]
    tn = [k for k in wdf._k.tolist() if k]
    pn = [k for k in rdf._k.tolist() if k]
    pairs, n_strict = pair_materials(list(dict.fromkeys(tn)), list(dict.fromkeys(pn)))
    prec = len(pairs) / len(set(pn)) if pn else 0.0
    rec = len(pairs) / len(set(tn)) if tn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    w = wdf.drop_duplicates('_k').set_index('_k')
    r = rdf.drop_duplicates('_k').set_index('_k')
    tp = fp = fn = exact = within = 0
    diffs = []
    for kt, kp in pairs:
        for c in NUMCOLS:
            a, b = w.loc[kt, c], r.loc[kp, c]
            ap, bp = pd.notna(a), pd.notna(b)
            if ap and bp:
                tp += 1
                if float(a) == float(b):
                    exact += 1
                if abs(float(a) - float(b)) <= 0.1:
                    within += 1
                diffs.append(float(b) - float(a))
            elif ap and not bp:
                fn += 1
            elif bp and not ap:
                fp += 1
    cp = tp / (tp + fp) if tp + fp else 0.0
    cr = tp / (tp + fn) if tp + fn else 0.0
    cf = 2 * cp * cr / (cp + cr) if cp + cr else 0.0
    return dict(mat_precision=prec, mat_recall=rec, mat_f1=f1,
                n_true_materials=len(set(tn)), n_pred_materials=len(set(pn)),
                n_paired=len(pairs), n_paired_strict=n_strict,
                cell_precision=cp, cell_recall=cr, cell_f1=cf,
                cells_both_present=tp, cells_missed=fn, cells_invented=fp,
                cells_exact=exact, cells_within_0p1=within,
                cell_rmse=float(np.sqrt(np.mean(np.square(diffs)))) if diffs else 0.0,
                cell_max_abs_diff=float(np.max(np.abs(diffs))) if diffs else 0.0)


def winner_integrity(Winner):
    """Checks on the ground-truth string itself, independent of any extraction.

    field_count : a data line of the Winner CSV should hold 12 fields. An unquoted comma inside
                  the material name produces 13. Parsed against the 12-field header, pandas
                  takes the first field as the row index, so the oxide values still land in the
                  right columns but the material name becomes whatever followed the comma
                  (for Dataset 2 row 191, 'mortar SNM, 1' is read as index 'mortar SNM' with
                  material name '1'). Any material-name metric on such a table is meaningless.
    """
    s = str(Winner).replace('\\n', '\n')
    lines = [ln for ln in s.split('\n') if ln.strip()]
    if not lines:
        return dict(n_lines=0)
    counts = [ln.count(',') + 1 for ln in lines[1:]]
    return dict(n_data_lines=len(counts),
                field_counts=sorted(set(counts)),
                n_lines_with_extra_field=int(sum(1 for c in counts if c > len(EXPECTED))),
                header_fields=lines[0].count(',') + 1)


# ---------------------------------------------------------------- run
rows_nb5, rows_std, d1_cmp, mismatch_rows, integrity = [], [], [], [], []
for rec in sel["selected"]:
    i = rec["d2_row"]
    resp = open(os.path.join(OUT, "responses", f"d2row{i:03d}.csv"), encoding="utf-8").read().strip()
    winner = d2.loc[i, "Winner"]
    print(f"\n=== d2 row {i} [{rec['group']}] {rec['caption'][:70]}")
    wi_chk = winner_integrity(winner)
    wi_chk.update(d2_row=i)
    integrity.append(wi_chk)
    if wi_chk.get("n_lines_with_extra_field"):
        print(f"  ground-truth check: {wi_chk['n_lines_with_extra_field']} of "
              f"{wi_chk['n_data_lines']} Dataset-2 data lines hold more than {len(EXPECTED)} "
              f"fields (field counts {wi_chk['field_counts']}): the material name contains an "
              f"unquoted comma, so the name is lost on parsing (the oxide values stay aligned)")
    m5 = notebook5_metrics(winner, resp)
    m5.update(d2_row=i, group=rec["group"], doi=rec["doi"])
    rows_nb5.append(m5)
    print(f"  notebook-5 metrics: relevance f1 {m5['f1_rel']:.2f}" +
          (f" | materials f1 {m5.get('f1_mat', float('nan')):.2f} "
           f"| compositions f1 {m5.get('f1_num', float('nan')):.2f} "
           f"| RMSE {m5.get('rmse', float('nan')):.2f}" if 'f1_mat' in m5 else "  (one side 'NR')"))
    if m5["rel_true"] and m5["rel_pred"]:
        st = standard_metrics(winner, resp)
        st.update(d2_row=i, group=rec["group"], doi=rec["doi"])
        rows_std.append(st)
        print(f"  standard metrics:   materials {st['n_pred_materials']}/{st['n_true_materials']} "
              f"F1 {st['mat_f1']:.2f} | cells both present {st['cells_both_present']} "
              f"(exact {st['cells_exact']}, within 0.1 {st['cells_within_0p1']}), "
              f"missed {st['cells_missed']}, invented {st['cells_invented']}, "
              f"RMSE {st['cell_rmse']:.3f}")
    else:
        print(f"  relevance: ground truth {'relevant' if m5['rel_true'] else 'NR'}, "
              f"mine {'relevant' if m5['rel_pred'] else 'NR'}")

    # ---------------- B: against the Dataset 1 rows for this table
    if rec["doi"] is None:
        continue
    sub = d1[(d1.cap_n == rec["caption"]) & (d1.DOI == rec["doi"])]
    rdf = parse_response_to_df(resp, EXPECTED)
    for col in NUMCOLS:
        rdf[col] = pd.to_numeric(rdf[col], errors='coerce')
    rdf["_k"] = [key_name(x) for x in rdf['material name']]
    d1k = sub.copy()
    d1k["_k"] = [key_name(x) for x in d1k['Material Name']]
    pairs, n_strict = pair_materials(list(dict.fromkeys([k for k in d1k._k if k])),
                                     list(dict.fromkeys([k for k in rdf._k if k])))
    shared = pairs
    ex = wi = both = miss = inv = 0
    for ka, kb in shared:
        a_row = d1k[d1k._k == ka].iloc[0]
        b_row = rdf[rdf._k == kb].iloc[0]
        k = ka
        for c in NUMCOLS:
            a = pd.to_numeric(pd.Series([a_row[[q for q in d1.columns if q.lower() == c][0]]]),
                              errors='coerce').iloc[0]
            b = b_row[c]
            ap, bp = pd.notna(a), pd.notna(b)
            if ap and bp:
                both += 1
                if float(a) == float(b):
                    ex += 1
                if abs(float(a) - float(b)) <= 0.1:
                    wi += 1
                else:
                    mismatch_rows.append(dict(d2_row=i, doi=rec["doi"], material=k, oxide=c,
                                              dataset1=float(a), mine=float(b),
                                              diff=float(b) - float(a)))
            elif ap and not bp:
                miss += 1
                mismatch_rows.append(dict(d2_row=i, doi=rec["doi"], material=k, oxide=c,
                                          dataset1=float(a), mine=np.nan, diff=np.nan))
            elif bp and not ap:
                inv += 1
                mismatch_rows.append(dict(d2_row=i, doi=rec["doi"], material=k, oxide=c,
                                          dataset1=np.nan, mine=float(b), diff=np.nan))
    only_d1 = sorted(set(d1k._k) - {a for a, _ in shared})
    only_me = sorted(set(rdf._k) - {b for _, b in shared})
    d1_cmp.append(dict(d2_row=i, doi=rec["doi"], n_dataset1=len(sub), n_mine=len(rdf),
                       n_shared=len(shared), n_shared_strict=n_strict,
                       only_in_dataset1=only_d1, only_in_mine=only_me,
                       cells_both_present=both, cells_exact=ex, cells_within_0p1=wi,
                       cells_in_d1_not_mine=miss, cells_in_mine_not_d1=inv))
    print(f"  vs Dataset 1: materials {len(rdf)} mine / {len(sub)} theirs, {len(shared)} shared; "
          f"cells both present {both} (exact {ex}, within 0.1 {wi}); "
          f"only in Dataset 1 {only_d1 or '-'}; only in mine {only_me or '-'}")

pd.DataFrame(rows_nb5).to_csv(os.path.join(OUT, "metrics_notebook5.csv"), index=False)
pd.DataFrame(rows_std).to_csv(os.path.join(OUT, "metrics_standard.csv"), index=False)
pd.DataFrame(d1_cmp).to_csv(os.path.join(OUT, "comparison_vs_dataset1.csv"), index=False)
pd.DataFrame(mismatch_rows).to_csv(os.path.join(OUT, "mismatches_vs_dataset1.csv"), index=False)
pd.DataFrame(integrity).to_csv(os.path.join(OUT, "dataset2_integrity.csv"), index=False)

main5 = [r for r in rows_nb5 if r["group"] == "main"]
allrel = [r for r in rows_nb5]
agg = dict(
    n_tables=len(rows_nb5), n_main=len(main5),
    notebook5_mean_f1_relevance=float(np.mean([r["f1_rel"] for r in allrel])),
    notebook5_relevance_accuracy=float(np.mean([r["acc_rel"] for r in allrel])),
    notebook5_mean_f1_materials=float(np.mean([r["f1_mat"] for r in allrel if "f1_mat" in r])),
    notebook5_mean_f1_compositions=float(np.mean([r["f1_num"] for r in allrel if "f1_num" in r])),
    notebook5_mean_rmse=float(np.mean([r["rmse"] for r in allrel if "rmse" in r])),
    standard_mean_material_f1=float(np.mean([r["mat_f1"] for r in rows_std])),
    standard_mean_cell_f1=float(np.mean([r["cell_f1"] for r in rows_std])),
    standard_total_cells=int(np.sum([r["cells_both_present"] for r in rows_std])),
    standard_total_exact=int(np.sum([r["cells_exact"] for r in rows_std])),
    standard_total_within_0p1=int(np.sum([r["cells_within_0p1"] for r in rows_std])),
    standard_total_missed=int(np.sum([r["cells_missed"] for r in rows_std])),
    standard_total_invented=int(np.sum([r["cells_invented"] for r in rows_std])),
    paper_table_G3_ft_gpt35=dict(relevance_train=0.92, relevance_test=0.95,
                                 materials_train=1.00, materials_test=0.83,
                                 compositions_train=0.99, compositions_test=0.93,
                                 rmse_train=1.49, rmse_test=2.81, avg_time_s=6.87),
    note=("Dataset 2 is the authors' fine-tuning set. Notebook 4 splits it 90/10 with no fixed "
          "seed, so which of the 200 tables were their test rows is not recoverable; the Table "
          "G3 train and test columns are both shown for reference."),
)
d1tot = pd.DataFrame(d1_cmp)
if len(d1tot):
    agg["vs_dataset1"] = dict(
        materials_mine=int(d1tot.n_mine.sum()), materials_dataset1=int(d1tot.n_dataset1.sum()),
        materials_shared=int(d1tot.n_shared.sum()),
        cells_both_present=int(d1tot.cells_both_present.sum()),
        cells_exact=int(d1tot.cells_exact.sum()),
        cells_within_0p1=int(d1tot.cells_within_0p1.sum()),
        cells_in_d1_not_mine=int(d1tot.cells_in_d1_not_mine.sum()),
        cells_in_mine_not_d1=int(d1tot.cells_in_mine_not_d1.sum()))
jdump(agg, os.path.join(OUT, "scores_summary.json"))

print("\n==================== aggregate ====================")
for k, v in agg.items():
    print(f"  {k}: {v}")
