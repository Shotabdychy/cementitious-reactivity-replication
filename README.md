# cementitious-reactivity-replication

A replication study of **Mahjoubi, Venugopal, Manav, Azarijafari, Kirchain & Olivetti, "Data-driven material screening of secondary and natural cementitious precursors", *Communications Materials* 6, 99 (2025)** ([article](https://www.nature.com/articles/s43246-025-00820-4)).

| | |
|---|---|
| **Report** | [`REPLICATION_REPORT_FINAL.docx`](REPLICATION_REPORT_FINAL.docx) (PDF: [`REPLICATION_REPORT_FINAL.pdf`](REPLICATION_REPORT_FINAL.pdf)) |
| **Short summary** | [`docs/SUMMARY.md`](docs/SUMMARY.md) |
| **Longer technical report** | [`docs/REPLICATION_REPORT_TECHNICAL.pdf`](docs/REPLICATION_REPORT_TECHNICAL.pdf) |
| **Original code (authors')** | <https://github.com/olivettigroup/Cementitious-reactivity-prediction-and-discovery>, commit `2340ec87eb3a8f5d564528f08a85f509887c811c` (27 Oct 2024) |
| **This repository** | <https://github.com/Shotabdychy/cementitious-reactivity-replication> |

> **Scope.** Everything needed to inspect the computational work is here. Large or third-party material (trained models, datasets, row-level predictions) is intentionally not committed; see `data/README.md` and the commands below to regenerate it.

## Original code versus this replication

- **The authors' notebooks are not copied here.** They are MIT-licensed and available at the link above; clone them at the commit shown to compare line by line.
- **Everything in `scripts/` was written for this replication.** `nb9_faithful.py` re-implements notebook 9 cell by cell and reuses its loss functions, architecture and loss weights *as published* (including their quirks); that logic is the authors', re-typed. Every other script is new work and is labelled as such in its docstring.
- Results from the notebook-faithful scripts and from the modified protocol are kept in separate files and separate rows of every table.

## What was and was not run

| Stage | Notebooks | Status |
|---|---|---|
| Literature filtering, table and sentence retrieval | 1–3, 6 | Not runnable: needs the group's private MongoDB corpus |
| Table extraction and material typing | 4, 5, 7, 8 | Reproduced on 6 tables / 40 rows with a **substitute model (Claude Opus 5)**; the authors' fine-tuned endpoints are private |
| Reactivity network | 9 | Re-run: faithful, "clean", and modified protocol (tuning, 11 seeds, ensemble, grouped splits) |
| Literature materials | 10 | Re-run with the ensemble |
| Natural rocks | 11 | Re-run with the ensemble plus an 8-variant ablation |

## Layout

```
README.md
REPLICATION_REPORT_FINAL.docx / .pdf   the 6-page report
scripts/        all code written for this replication (see the table below)
environment/    requirements.txt (main), requirements_legacy.txt (library-version check)
data/README.md  where each input comes from and how missing files were reconstructed
results/        JSON/CSV the report quotes (flat; see "Results index")
  few_papers/   substitute-extractor replies, scores, mismatches
figures/        PNG figures
logs/           stdout of every run
docs/           SUMMARY.md/.pdf, technical report, report template
```

Not included: trained models (`models/`, about 100 MB; regenerate with the commands below), caches, the supplementary datasets and the Gard et al. database (third-party; see `data/README.md`), row-level prediction tables, and any file embedding third-party table text (the extraction prompts, the Dataset 2 ground truth, Dataset 3 captions). Folders are not created empty to match a template.

## Scripts and the notebooks they correspond to

| Script | Role |
|---|---|
| `common.py` | Shared code: data loading, losses, model, splits, metrics, name matching |
| `nb9_faithful.py` | Notebook 9, faithful and "clean" variants (`python nb9_faithful.py faithful 42`) |
| `nb9_tune.py` | Keras Tuner random search, 5-fold CV on training rows only |
| `nb9_final.py` | Modified protocol: N seeds, ensemble, bootstrap CIs, grouped splits |
| `nb10_run.py` | Notebook 10 on the literature materials (Dataset 1) |
| `nb11_ablation.py` | Notebook 11 on the Gard database, with the ablation |
| `figures.py`, `fig_workflow.py` | Figures 2–4 equivalents and the workflow diagram |
| `few_papers_*.py`, `worked_example.py` | Extraction demonstration: select tables, score, classify, predict; assemble the single-material worked example |
| `make_tables.py`, `make_report.py`, `make_pdf.py`, `make_docx.py` | Build tables and the reports from the result files |
| `compare_runs.py`, `run_verify.sh` | Clean-start verification |
| `run_*.sh`, `docx_to_pdf.ps1` | Batch runners; Word-based PDF export |

## Environment

Python 3.11.9, TensorFlow 2.15.1, pandas 3.0.6, scikit-learn 1.9.1, LightGBM 4.7.0, KerasTuner 1.4.7, SHAP 0.49.1 (Windows 11, 8 CPU cores, no GPU).

```bash
pip install -r environment/requirements.txt
```

The paper states Python 3.8. That interpreter was **not** used: only 3.11 was available. One faithful run was repeated on an older stack (`environment/requirements_legacy.txt`: TensorFlow 2.12.0, pandas 1.5.3, scikit-learn 1.2.2, LightGBM 3.3.5) on Python 3.11, which is a library-version check, not a Python 3.8 test.

## Inputs

See [`data/README.md`](data/README.md). Put the supplementary datasets in `data/supplementary/` and the Gard et al. tables in `data/gard/`, or point `REPL_DATA` and `REPL_GARD` at them.

## Reproducing each result

Run from `scripts/`. Seeds are `42, 1, 2, … 10`; the sound-protocol outer split uses `SPLIT_SEED = 0`; bootstrap seed 7.

```bash
for s in 42 1 2 3 4 5 6 7 8 9 10; do python nb9_faithful.py faithful $s; done
```

```bash
for s in 42 1 2 3 4 5 6 7 8 9 10; do python nb9_faithful.py clean $s; done
```

```bash
python nb9_tune.py --trials 18 --folds 5 --max-epochs 500
```

```bash
python nb9_final.py --config notebook --seeds 11 --tag sound_notebook
```

```bash
python nb9_final.py --config tuned --seeds 11 --tag sound_tuned
```

```bash
python nb9_final.py --config tuned --seeds 11 --grouped material --tag grouped_material
```

```bash
python nb10_run.py --models sound_tuned
```

```bash
python nb11_ablation.py --models sound_tuned
```

```bash
python figures.py --models sound_tuned
```

```bash
python few_papers_select.py && python few_papers_score.py && python few_papers_categories.py && python few_papers_predict.py --models sound_tuned && python worked_example.py
```

Expected headline outputs (from the logs in this folder): faithful Ca(OH)₂ test R² 0.087 ± 0.130 over 11 seeds, all 11 runs flagged as collapsed; tuned ensemble test R² 0.947 / 0.764 / 0.830 (heat release / Ca(OH)₂ / bound water) on 264 / 38 / 56 labels; material-grouped ensemble 0.645 / −0.061 / 0.548; 141,548 rock samples reactive as released, 64,239 with Dataset 5 and total iron; 240 of 241 oxide cells exact on 3 tables.

Building the report: to put an author name on the title line, run `python make_docx.py --author "Name"` and then `powershell docx_to_pdf.ps1` (Word is required for the PDF). Without those arguments the report prints clearly marked placeholders. `make_docx.py` reads the row-level prediction files that `nb9_final.py` writes (they are not committed), so run the training commands first.

Step 4 of the extraction demonstration (the replies in `results/few_papers/responses/`) was produced interactively by Claude Opus 5 from the prompts that `few_papers_select.py` writes; it is not a scripted API call. See `results/few_papers/EXTRACTOR_NOTE.md`.

## Results index

| Report item | Files |
|---|---|
| Model accuracy | `results/nb9_*.json`, `results/table_model_accuracy.csv` |
| Collapse diagnostics | `results/nb9_faithful_seed*.json`, `logs/nb9_faithful_seed*.log` |
| Tuning | `results/nb9_tuned_config.json`, `results/nb9_tuning_all_trials.csv`, `logs/nb9_tune.log` |
| Literature materials | `results/nb10_sound_tuned.json`, `results/nb10_sound_tuned_summary_by_category.csv` |
| Rock ablation | `results/nb11_sound_tuned_ablation.csv/.json`, `results/nb11_sound_tuned_by_rock_name.csv` |
| Extraction demonstration | `results/few_papers/` |
| Verification | `logs/verify.log`, `logs/verify_supplement.log` |

## Limitations

Individual protocol changes were not isolated; the Ca(OH)₂ and bound-water test sets are small; the extractor is a substitute; Python 3.8 was not used; and not every run was repeated in the verification pass. The scripts carry no licence yet: choose one before publishing. Details are in the report, Section 5.
