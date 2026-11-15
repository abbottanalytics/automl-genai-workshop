#!/usr/bin/env bash
# The two quick AutoGluon runs for the live session (about 1 minute each on a laptop).
# Run from any folder; uses the AutoGluon environment's python (set AUTOGLUON_PYTHON to override).
# Results -> results/<run-name>.json, .log, _preds.csv (the shipped results are overwritten).
cd "$(dirname "$0")"
PY=${AUTOGLUON_PYTHON:-python}
TITANIC_DROP=(name ticket cabin embarked boat body home.dest)     # keep pclass, sex, age, sibsp, parch, fare
run() { name=$1; shift; echo "=== $name $(date)"; "$PY" automl_harness_autogluon.py "$@" --run-name "$name" 2>&1 | tee "results/$name.log" | tail -3; }
run titanic_autogluon_default --data ../data/titanic3.csv --target survived --drop "${TITANIC_DROP[@]}" --no-time-limit
run kdd98_autogluon_default --data ../data/cup98lrn_automl_prepped_with_split.tsv --split-col is_test --id-col RECORD_ID \
    --target TARGET_B --gift-col TARGET_D --no-time-limit
