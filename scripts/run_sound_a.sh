#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
PY=${PY:-python}
export OMP_NUM_THREADS=3 TF_NUM_INTRAOP_THREADS=3 TF_NUM_INTEROP_THREADS=1
"$PY" nb9_final.py --config tuned --seeds 11 --tag sound_tuned            > ../logs/nb9_sound_tuned.log 2>&1
"$PY" nb9_final.py --config tuned --seeds 11 --grouped material --tag grouped_material > ../logs/nb9_grouped_material.log 2>&1
echo "STREAM A DONE"
