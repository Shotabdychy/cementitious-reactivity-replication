# logs/

Standard output of every run, kept as evidence for the numbers in the report.

| Files | What |
|---|---|
| `nb9_faithful_seed*.log`, `nb9_clean_seed*.log`, `nb9_faithful_seed42_legacy.log` | notebook 9 as published / "clean" variant, one log per seed (the legacy log is the older-library run) |
| `nb9_tune.log` | the 18-trial, 5-fold tuning search |
| `nb9_sound_notebook.log`, `nb9_sound_tuned.log`, `nb9_grouped_*.log`, `nb9_sens_impute_all.log` | 11-seed ensembles under the modified protocol, grouped splits and the sensitivity run |
| `nb10_sound_tuned.log`, `nb11_ablation.log` | notebooks 10 and 11 |
| `figures*.log`, `few_papers_*.log` | figures; extraction demonstration |
| `verify.log`, `verify_supplement.log` | clean-start verification: re-run outputs compared with the first run (IDENTICAL / DIFFERS) |
