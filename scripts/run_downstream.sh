#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
PY=${PY:-python}
export OMP_NUM_THREADS=3 TF_NUM_INTRAOP_THREADS=3 TF_NUM_INTEROP_THREADS=1
"$PY" nb10_run.py --models sound_tuned                > ../logs/nb10_sound_tuned.log 2>&1
echo "NB10 DONE"
"$PY" few_papers_predict.py --models sound_tuned      > ../logs/few_papers_predict.log 2>&1
echo "FEWPAPERS PREDICT DONE"
"$PY" nb11_ablation.py --models sound_tuned           > ../logs/nb11_ablation.log 2>&1
echo "NB11 ABLATION DONE"
"$PY" figures.py --models sound_tuned                 > ../logs/figures.log 2>&1
echo "FIGURES DONE"
