#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-.venv/bin/python}
PREFIX=${PREFIX:-complexity}
ITERS=${ITERS:-1000}
ENVS=${ENVS:-16}
THREADS=${THREADS:-2}
SEEDS=${SEEDS:-"1 2 3 4 5"}
METHODS=${METHODS:-"ippo central"}
OVERWRITE=${OVERWRITE:-0}

if [ ! -x /usr/bin/time ]; then
    echo "/usr/bin/time is required."
    exit 1
fi

mkdir -p runs results

echo "============================================================"
echo "Training complexity experiment"
echo "methods : $METHODS"
echo "seeds   : $SEEDS"
echo "iters   : $ITERS"
echo "envs    : $ENVS"
echo "threads : $THREADS"
echo "prefix  : $PREFIX"
echo "============================================================"

for method in $METHODS; do
    for seed in $SEEDS; do
        out="runs/${PREFIX}_${method}_s${seed}"

        if [ -e "$out" ]; then
            if [ "$OVERWRITE" = "1" ]; then
                rm -rf "$out"
            else
                echo "ERROR: $out already exists."
                echo "Delete it, change PREFIX, or rerun with OVERWRITE=1."
                exit 1
            fi
        fi

        mkdir -p "$out"

        echo
        echo ">>> START method=$method seed=$seed"

        /usr/bin/time -v             -o "$out/time.txt"             "$PYTHON" trainer.py                 --method "$method"                 --preset hotspot                 --iters "$ITERS"                 --envs "$ENVS"                 --seed "$seed"                 --threads "$THREADS"                 --save_every 100                 --out "$out"             > "$out/train_stdout.log" 2>&1

        echo "<<< DONE  method=$method seed=$seed"
        tail -n 1 "$out/train_stdout.log" || true
        grep "Maximum resident set size" "$out/time.txt" || true
    done
done

echo
echo "All training runs finished."
echo "Now run:"
echo "  $PYTHON complexity_report.py --prefix $PREFIX"
