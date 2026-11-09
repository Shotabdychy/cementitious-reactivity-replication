#!/usr/bin/env bash
# Part 1.6 / baseline: the released notebook 9 and the 'clean' variant, 11 seeds each.
set -u
cd "$(dirname "$0")"
PY=${PY:-python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-3}
export TF_NUM_INTRAOP_THREADS=${TF_NUM_INTRAOP_THREADS:-3}
export TF_NUM_INTEROP_THREADS=1
for v in faithful clean; do
  for s in 42 1 2 3 4 5 6 7 8 9 10; do
    echo "=== $v seed $s ==="
    "$PY" nb9_faithful.py "$v" "$s" > "../logs/nb9_${v}_seed${s}.log" 2>&1
    tail -4 "../logs/nb9_${v}_seed${s}.log" | grep -E "R2|COLLAPSE" || true
  done
done
echo "ALL FAITHFUL RUNS DONE"
