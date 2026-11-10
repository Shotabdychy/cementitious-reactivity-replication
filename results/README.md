# results/

Every number in the report and the README comes from a file in this folder. Files are written flat by the scripts
(so the commands in the main README reproduce them in place); this index groups them by topic.

| Topic | Files | Produced by |
|---|---|---|
| **Model accuracy** (notebook 9) | `nb9_faithful_seed*.json`, `nb9_clean_seed*.json` (11 seeds each); `nb9_sound_notebook.json`, `nb9_sound_tuned.json`, `nb9_grouped_material.json`, `nb9_grouped_doi.json`, `nb9_sens_impute_all.json` (11-seed ensembles; per-seed results, ensemble metrics, bootstrap CIs); `nb9_faithful_seed42_legacy.json` (older libraries) | `nb9_faithful.py`, `nb9_final.py` |
| **Tuning** | `nb9_tuned_config.json` (chosen configuration), `nb9_tuning_all_trials.csv` (all 18 trials) | `nb9_tune.py` |
| **Literature materials** (notebook 10) | `nb10_sound_tuned.json`, `nb10_sound_tuned_summary_by_category.csv` | `nb10_run.py` |
| **Rock screening** (notebook 11) | `nb11_sound_tuned_ablation.csv` / `.json` (8 variants x 3 row filters x ensemble/single model), `nb11_sound_tuned_by_rock_name.csv` | `nb11_ablation.py` |
| **Figure data** | `fig2_*.csv`, `fig2_importance_overlap_sound_tuned.json`, `fig3_literature_sound_tuned.csv`, `fig4_rocks_sound_tuned.csv` | `figures.py` |
| **Summary tables** | `table_model_accuracy.csv`, `table_claims.csv`, `table_library_version_check.csv`, `table_runtime.csv` | `make_tables.py` |
| **Extraction demonstration** | `few_papers/` (see below) | `few_papers_*.py`, `worked_example.py` |

`few_papers/`: `responses/` (the substitute extractor's replies, one CSV per table), `metrics_standard.csv` and
`metrics_notebook5.csv` (scores), `comparison_vs_dataset1.csv`, `mismatches_vs_dataset1.csv`, `scores_summary.json`,
`my_categories.csv`, `d3_*` (material-type classification), `predictions_*` and `imputed_descriptors.csv` (model inputs and
outputs for the extracted materials), `worked_example.json` (the HL1 example), `dataset2_integrity.csv`, `EXTRACTOR_NOTE.md`.

Not committed (regenerate with the main README): trained models, row-level prediction tables, and any file that embeds
third-party table text.
