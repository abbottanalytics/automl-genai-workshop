#!/usr/bin/env bash
# Every auto-sklearn run in the workshop tables: about 10 minutes (Titanic) and 20 minutes (each KDD98 run).
# Uses the auto-sklearn environment's python (set AUTOSKLEARN_PYTHON to override). See README.md for
# installing auto-sklearn, and for memory and disk use on large runs.
cd "$(dirname "$0")"
PY=${AUTOSKLEARN_PYTHON:-python}
TITANIC_DROP=(name ticket cabin embarked boat body home.dest)
KDD=(--data ../data/cup98lrn_automl_prepped_with_split.tsv --split-col is_test --id-col RECORD_ID --target TARGET_B --gift-col TARGET_D)
run() { name=$1; shift; echo "=== $name $(date)"; "$PY" -W ignore automl_harness_autosklearn.py "$@" --run-name "$name" > "results/$name.log" 2>&1; echo "exit $? $(date)"; }
run titanic_autosklearn_default --data ../data/titanic3.csv --target survived --drop "${TITANIC_DROP[@]}" --time-limit 600 --n-jobs 8
run kdd98_autosklearn_default   "${KDD[@]}" --time-limit 900 --n-jobs 2 --max-models-on-disc 10
run kdd98_autosklearn_auc       "${KDD[@]}" --time-limit 900 --n-jobs 2 --max-models-on-disc 10 --metric roc_auc
