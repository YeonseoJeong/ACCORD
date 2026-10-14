#!/bin/bash
# Train and evaluate methods x seeds, JOBS at a time (2 torch threads each).
#
#   ./run_batch.sh "accord es_only es_cio0 ippo team central" "1 2 3"
#   PREFIX=k30   BASE="K=30"            ./run_batch.sh "accord es_only" "1 2 3"
#   PREFIX=vmax2 BASE="v_max=0.02"      ./run_batch.sh "accord es_cio0" "1 2 3"
#   PREFIX=d25   PRESET=hotspot_tight   ./run_batch.sh "accord es_cio0" "1 2 3"
#
# Each run goes to runs/<PREFIX>_<method>_s<seed>/ (log.csv, final.pt, eval_stochastic_*.json
# and eval_greedy_final.json);
# runs that already have final.pt are only re-evaluated.
cd "$(dirname "$0")"
mkdir -p runs
METHODS=${1:-"accord es_only es_cio0 ippo team central"}
SEEDS=${2:-"1 2 3"}
export PRESET=${PRESET:-hotspot} BASE=${BASE:-} PREFIX=${PREFIX:-main} ITERS=${ITERS:-500}
JOBS=${JOBS:-8}
one() {
  m=$1; s=$2; out=runs/${PREFIX}_${m}_s${s}
  if [ ! -f "$out/final.pt" ]; then
    .venv/bin/python trainer.py --method "$m" --preset "$PRESET" ${BASE:+--base "$BASE"} \
      --iters "$ITERS" --envs 16 --seed "$s" --threads 2 --save_every 100 --out "$out" \
      > "$out.log" 2>&1
  fi
  # sampled actions (Algorithm 2) are the reported execution; the mode is kept for the gap column
  .venv/bin/python evaluate.py "$out" --stochastic --threads 1 > "$out.evalstoch.txt" 2>&1
  case $m in accord*|ippo|team|central)
    .venv/bin/python evaluate.py "$out" --stochastic --mlb_off --threads 1 > "$out.evalstochoff.txt" 2>&1;; esac
  .venv/bin/python evaluate.py "$out" --threads 1 > "$out.eval.txt" 2>&1
}
export -f one
for m in $METHODS; do for s in $SEEDS; do echo "$m $s"; done; done | xargs -P "$JOBS" -L 1 bash -c 'one $0 $1'
echo "batch $PREFIX finished"
