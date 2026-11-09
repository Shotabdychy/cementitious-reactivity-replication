#!/usr/bin/env bash
# Verification pass: re-run the final scripts once from a clean start (empty cache) and diff the
# regenerated results against the first run. Prints IDENTICAL or DIFFERS for each.
#
# The ablation and the 11-seed sweeps are verified on their headline configuration only
# (notebook 9: sound_tuned; notebook 11: the 'base' variant), which is stated in the report.
set -u
cd "$(dirname "$0")"
PY=${PY:-python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6} TF_NUM_INTRAOP_THREADS=${TF_NUM_INTRAOP_THREADS:-6}
export TF_NUM_INTEROP_THREADS=1

R=../results
B=../results/verify_baseline
mkdir -p "$B" ../logs

FILES=(
  "nb9_faithful_seed42.json"
  "nb9_clean_seed42.json"
  "nb9_sound_tuned.json"
  "nb10_sound_tuned_summary_by_category.csv"
  "nb11_sound_tuned_ablation.csv"
  "few_papers/scores_summary.json"
)

echo "### saving the first run as the baseline"
for f in "${FILES[@]}"; do
  mkdir -p "$B/$(dirname "$f")"
  [ -f "$R/$f" ] && cp "$R/$f" "$B/$f" && echo "  saved $f"
done

echo "### clearing the cache so every table is rebuilt"
rm -rf ../data_cache
mkdir -p ../data_cache

echo "### re-running"
"$PY" nb9_faithful.py faithful 42 > ../logs/verify_nb9_faithful42.log 2>&1
"$PY" nb9_faithful.py clean 42    > ../logs/verify_nb9_clean42.log 2>&1
"$PY" nb9_final.py --config tuned --seeds 11 --tag sound_tuned > ../logs/verify_nb9_sound_tuned.log 2>&1
"$PY" nb10_run.py --models sound_tuned > ../logs/verify_nb10.log 2>&1
# --tag verify keeps this base-only re-run out of the full ablation's result files
"$PY" nb11_ablation.py --models sound_tuned --tag verify --only base > ../logs/verify_nb11_base.log 2>&1
"$PY" few_papers_score.py > ../logs/verify_few_papers.log 2>&1

echo "### comparison (wall-clock and environment fields ignored)"
for f in "${FILES[@]}"; do
  if [ "$f" = "nb11_sound_tuned_ablation.csv" ]; then
    "$PY" compare_runs.py "$B/$f" "$R/nb11_verify_ablation.csv" --only-variant base
  else
    "$PY" compare_runs.py "$B/$f" "$R/$f"
  fi
done
echo "VERIFY DONE"
