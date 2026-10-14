#!/usr/bin/env bash
# Full population re-evaluation with the ORIGINAL gen_eval.py (no modifications).
# Run at the repository root, alongside gen_eval.py and plot_population_paper.py.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
ROOT="${ROOT:-runs_dynamic_ue_50_100}"
FIXED_DIR="${FIXED_DIR:-complexity_1000/ippo_s1}"
ACCORD_DIR="${ACCORD_DIR:-${ROOT}/ippo_s1}"
ESONLY_DIR="${ESONLY_DIR:-${ROOT}/es_only_s1}"
OUT="${ROOT}/population_eval_stochastic"
JSON_DIR="${OUT}/gen_eval_json"
ORIGINAL_JSON_DIR="results/gen_load"
NS=(50 75 100 125 150)

for d in "$FIXED_DIR" "$ACCORD_DIR" "$ESONLY_DIR"; do
    if [[ ! -f "$d/config.json" || ! -f "$d/final.pt" ]]; then
        echo "ERROR: Cannot find checkpoint config.json/final.pt in '$d'." >&2
        echo "Set FIXED_DIR, ACCORD_DIR, or ESONLY_DIR to the actual trained run folder." >&2
        exit 1
    fi
done
mkdir -p "$JSON_DIR"

# Original gen_eval always writes results/gen_load/<run-basename>_N<N>.json.
# Copy after EACH call: both the fixed and dynamic ACCORD checkpoints are often
# named ippo_s1, so otherwise one would overwrite the other's result.
run_and_copy() {
    local dest_key="$1"; shift
    local n="$1"; shift
    local kind="$1"; shift
    local run_dir="${1:-}"
    local src
    if [[ "$kind" == "learned" ]]; then
        "$PYTHON_BIN" gen_eval.py run learned "$n" "$run_dir" --episodes 150
        src="${ORIGINAL_JSON_DIR}/$(basename "$run_dir")_N${n}.json"
    else
        "$PYTHON_BIN" gen_eval.py run "$kind" "$n" --episodes 150
        src="${ORIGINAL_JSON_DIR}/${kind}_N${n}.json"
    fi
    cp "$src" "$JSON_DIR/${dest_key}_N${n}.json"
}

for N in "${NS[@]}"; do
    echo "======== N=$N, 30 eval seeds x 5 envs = 150 episodes ========"
    run_and_copy allon   "$N" allon
    run_and_copy rule    "$N" rule
    run_and_copy fixed   "$N" learned "$FIXED_DIR"
    run_and_copy dynamic "$N" learned "$ACCORD_DIR"
    run_and_copy esonly  "$N" learned "$ESONLY_DIR"
done

"$PYTHON_BIN" build_population_table.py --json-dir "$JSON_DIR" --out "$OUT" --episodes 150
"$PYTHON_BIN" plot_population_paper.py \
    --csv "$OUT/population_table.csv" \
    --out "$OUT/population_paper"
echo "Finished: $OUT/population_table.csv and population_paper.{pdf,png,svg}"
