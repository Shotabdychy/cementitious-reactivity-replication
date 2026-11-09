#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
PY=${PY:-python}
export OMP_NUM_THREADS=3 TF_NUM_INTRAOP_THREADS=3 TF_NUM_INTEROP_THREADS=1
"$PY" nb9_final.py --config notebook --seeds 11 --tag sound_notebook      > ../logs/nb9_sound_notebook.log 2>&1
"$PY" nb9_final.py --config tuned --seeds 11 --grouped doi --tag grouped_doi > ../logs/nb9_grouped_doi.log 2>&1
"$PY" nb9_final.py --config tuned --seeds 3 --impute all --tag sens_impute_all > ../logs/nb9_sens_impute_all.log 2>&1
echo "STREAM B DONE"
