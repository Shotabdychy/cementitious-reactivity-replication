# Input data

No datasets are redistributed in this repository. The supplementary material of the paper is published under CC BY-NC-ND, and the Gard et al. database is a third-party compilation; obtain both from their original sources.

## Supplementary datasets (put in `data/supplementary/`, or set `REPL_DATA`)

From the Supplementary Information of Mahjoubi et al. (2025), <https://www.nature.com/articles/s43246-025-00820-4>:

| File name expected | Content | Used by |
|---|---|---|
| `Dataset 1.csv` | 14,429 literature-extracted materials with DOI, caption, descriptor and category | `nb10_run.py`, `few_papers_*.py` |
| `Dataset 2.csv` | 200 tables with the manually curated extraction (`Prompt`, `Winner`) | `few_papers_*.py` |
| `Dataset 3.csv` | 318 material descriptions with manual categories | `few_papers_*.py` |
| `Dataset 4.csv` | 1,875 R³ test rows (the stand-in for the authors' `R3.csv`) | `nb9_*.py`, all downstream |
| `Dataset 5.csv` | About 160 rocks with measured amorphous content | `nb11_ablation.py` |

## Rock database (put in `data/gard/`, or set `REPL_GARD`)

Gard, M., Hasterok, D. & Halpin, J. A. *Global whole-rock geochemical database compilation*, Earth Syst. Sci. Data 11, 1553–1566 (2019); use the data release cited in that paper. Required tables: `sample.csv`, `major.csv`, `rockgroup.csv`, `country.csv`, `reference.csv` (1,022,092 samples after merging on their own keys).

## Files the original repository does not contain, and how they were reconstructed

| Missing file | Reconstruction | Caveat |
|---|---|---|
| `R3.csv` | `Dataset 4.csv`, mix columns divided by `SCM (g)`; 1,849 of 1,875 rows kept | The authors' file had 1,848 rows |
| `df_imputed.pkl` | The notebook's LightGBM imputation of the R³ features, clipped at 0 | Whether the authors' file also held imputed targets is unknown |
| `natural_rocks.csv` | Merge of the Gard tables on their keys, all samples kept | Row order matches the authors' first and last rows |
| `90andupnewones.keras`, `90andupnewonesch.keras` | Retrained: `nb9_faithful.py` (as released) and `nb9_final.py` (modified protocol) | Comparisons with values the authors saved are untestable |
| Pozzolanic/hydraulic rule, rock-name grouping, row filter for rocks with no composition | Not reconstructable; the report states which rules are mine | Reported as sensitivity analyses only |

The authors' private paper corpus (notebooks 1–3, 6) and fine-tuned OpenAI endpoints (notebooks 4, 7, 8) cannot be reconstructed; see the report, Section 4.1.
