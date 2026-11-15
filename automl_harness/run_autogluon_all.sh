#!/usr/bin/env bash
# Every AutoGluon run in the workshop tables. The two best_quality runs are long:
# Titanic about 10 minutes, KDD98 about 80 minutes (1 hour search + gift model).
cd "$(dirname "$0")"
PY=${AUTOGLUON_PYTHON:-python}
TITANIC_DROP=(name ticket cabin embarked boat body home.dest)
KDD=(--data ../data/cup98lrn_automl_prepped_with_split.tsv --split-col is_test --id-col RECORD_ID --target TARGET_B --gift-col TARGET_D)
run() { name=$1; shift; echo "=== $name $(date)"; "$PY" automl_harness_autogluon.py "$@" --run-name "$name" > "results/$name.log" 2>&1; echo "exit $? $(date)"; }
run titanic_autogluon_default  --data ../data/titanic3.csv --target survived --drop "${TITANIC_DROP[@]}" --no-time-limit
run titanic_autogluon_best_auc --data ../data/titanic3.csv --target survived --drop "${TITANIC_DROP[@]}" --presets best_quality --eval-metric roc_auc --time-limit 600
run kdd98_autogluon_default    "${KDD[@]}" --no-time-limit
run kdd98_autogluon_best_auc   "${KDD[@]}" --presets best_quality --eval-metric roc_auc --time-limit 3600
