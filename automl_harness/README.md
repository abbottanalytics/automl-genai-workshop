# AutoML harness: AutoGluon vs. auto-sklearn

The code and results behind the deck's AutoGluon and auto-sklearn slides ("Automating Building of Predictive Models", Abbott Analytics, ML Week Europe, November 2026).

Both tools are run the same way. The data is loaded the same way and split the same way (the same stratified train/test split), and the results are scored with the same metrics:

* **Speed**: search (fit) time and prediction time.
* **Accuracy, using each tool's defaults**: `predict()` labels at the tool's default 0.5 threshold, plus ROC AUC, log loss and lift in the top 10% from its probabilities.
* **KDD Cup 1998 business metric**: cumulative net revenue (gift − $0.68 mail cost, summed down the file sorted by score), its peak and the depth of the peak, and the net revenue of what the tool's own default decision would mail. Also a responder-only gift model, then expected value = P(respond) × predicted gift (Zadrozny & Elkan 2001).

Neither tool can optimize cumulative net revenue directly. The harness shows what their default metrics do to that business number.

A step-by-step GenAI version of the same experiment is in `../prompts/automl_experiments_prompts_v1.md`.

## In class vs. on your own

**In class, these analyses are run directly in a GenAI assistant (ChatGPT or Claude), using the prompt files in `../prompts/`.** This folder is not the main way the workshop runs. It is here for two reasons:

1. **To run the experiments on your own after the workshop**, outside a chat window, on your own machine and your own data.
2. **As a backup during the workshop**, in case an assistant fails or stalls on a step.

What it contains:

* `run_in_class_fast.sh` runs the two quick AutoGluon runs (Titanic and KDD98 with default settings, about 1 minute each). These are the runs to use if a live demo needs a fallback.
* **The longer runs** (AutoGluon `best_quality`: 10 minutes for Titanic, 80 minutes for KDD98; auto-sklearn: 10–20 minutes per run) are already done. Their results are in `results/`, so you can read them without running anything. `run_autogluon_all.sh` and `run_autosklearn_all.sh` rerun them.

## Results (from `results/comparison.md`)

**Titanic** (393 test passengers, 30% stratified hold-out; baseline accuracy 0.618)

| run | minutes | accuracy | ROC AUC |
|---|---|---|---|
| AutoGluon, defaults | 0.7 | 0.812 | 0.873 |
| AutoGluon, best_quality + AUC | 10.1 | 0.837 | 0.894 |
| auto-sklearn, defaults | 10.1 | 0.837 | 0.890 |

**KDD Cup 1998** (47,706 test donors, 5.1% respond; "nobody responds" is 94.9% accurate; mailing everyone nets $5,405)

| run | minutes (incl. gift model) | accuracy | ROC AUC | net revenue, tool's own decision | net revenue, EV, depth chosen on training (% mailed) |
|---|---|---|---|---|---|
| AutoGluon, defaults (accuracy) | 1 | 0.949 | 0.625 | $0 | $6,736 (84%) |
| auto-sklearn, defaults (accuracy) | 20 | 0.949 | 0.500 | $0 | $5,951 (93%) |
| AutoGluon, best_quality + AUC | 80 | 0.949 | 0.634 | $0 | $8,238 (60%) |
| auto-sklearn + AUC | 20 | 0.949 | 0.626 | $0 | $6,468 (80%) |

Both tools optimize accuracy by default, and with 5% responders every model ties "nobody responds". auto-sklearn then returns its dummy model (the same probability for everyone, AUC 0.500), and neither tool's `predict()` mails anyone. The fixes: tell the tool what matters (`eval_metric` / `metric = roc_auc`), and make the mailing decision yourself, with a cutoff chosen on training data.

Times are on an Apple M1 laptop (16 GB), AutoGluon 1.6.3, auto-sklearn 0.15.0.

## Data

Both datasets are in the repository's `data/` folder.

* **Titanic:** `titanic3.csv`, using six inputs: pclass, sex, age, sibsp, parch, fare. The run scripts drop the other columns, including the leakage columns `boat` and `body`.
* **KDD Cup 1998:** `cup98lrn_automl_prepped_with_split.tsv`. This is a different preparation of the same 95,412 records from the one the GenAI prompts use (`cup98lrn_prepped_allrecords_w_TargetD.tsv`):
  * 38 inputs, including the raw categorical fields (STATE, DOMAIN, RFA_2, RFA_3, GENDER, and others) left for the AutoML tools to encode themselves;
  * no variable selection;
  * `is_test` is the 50/50 train/test split used for every KDD98 run, and `RECORD_ID` identifies the record (never an input).

  So the AutoML numbers are not a like-for-like comparison with the GenAI notebook's models, which use the 34-column file with fewer, prepared inputs.

## Files

| file | what it does |
|---|---|
| `automl_harness_common.py` | data loading, split, metrics, result files (shared) |
| `automl_harness_autogluon.py` | AutoGluon-Tabular run (`--presets`, `--eval-metric` optional) |
| `automl_harness_autosklearn.py` | auto-sklearn run (`--metric roc_auc` optional; default accuracy) |
| `automl_harness_autogluon_results_view.py` | for notebook use after `predictor.fit(...)`: a compact model table and a full-width plot of validation score vs. fit time, in place of `fit_summary(show_plot=True)` |
| `automl_harness_compare.py` | builds `results/comparison.md` and the comparison CSVs from all runs in `results/` |
| `automl_harness_bootstrap.py` | paired bootstrap of the KDD98 net-revenue results from the saved predictions |
| `automl_harness_ask_mem_watchdog.sh` | optional memory guard for auto-sklearn on macOS (see below) |
| `run_in_class_fast.sh` | the two quick AutoGluon runs |
| `run_autogluon_all.sh`, `run_autosklearn_all.sh` | every run in the tables above |
| `results/` | for each run: `<run>.json` (settings, timings, metrics), `<run>.log`, `<run>_preds.csv` (test predictions) and, for KDD98, `<run>_cutoff_preds.csv` (out-of-fold training predictions); plus `comparison.md`, `comparison_*.csv`, `bootstrap_*.csv` |

Rerunning a script overwrites that run's files in `results/`. In the shipped JSON and log files, the data path was changed to the repository's file (the deck's runs read an identical local copy of the same data), and temporary-folder paths are shown as `<tmp>`.

## Setup

AutoGluon and auto-sklearn need separate Python environments.

**AutoGluon** (macOS, Linux, Windows):

```bash
conda create -n autogluon16 python=3.11
```

```bash
conda run -n autogluon16 pip install "autogluon.tabular[all]==1.6.3"
```

**auto-sklearn:** see the auto-sklearn section below.

## Run

From the `automl_harness/` folder, with the matching environment's python:

```bash
AUTOGLUON_PYTHON=$(conda run -n autogluon16 which python) ./run_in_class_fast.sh
```

```bash
AUTOGLUON_PYTHON=$(conda run -n autogluon16 which python) ./run_autogluon_all.sh
```

```bash
AUTOSKLEARN_PYTHON=$(conda run -n autosklearn which python) ./run_autosklearn_all.sh
```

Then rebuild the comparison tables:

```bash
python automl_harness_compare.py
```

A single run, for example AutoGluon's default on Titanic:

```bash
python automl_harness_autogluon.py --data ../data/titanic3.csv --target survived --drop name ticket cabin embarked boat body home.dest --no-time-limit --run-name titanic_autogluon_default
```

## Mailing cutoff chosen on TRAIN (KDD98)

With `--gift-col`, each run also chooses a mailing cutoff where cumulative net revenue peaks on **out-of-fold TRAIN predictions**, and applies it to TEST as a score threshold and as a share of the file (depth):

* AutoGluon with bagging (`best_quality`): `predict_proba_oof()` gives them directly.
* Otherwise (AutoGluon default, auto-sklearn): the whole run, gift model included, is cross-fit `--cutoff-cv` times (default 5; 0 turns it off). This multiplies run time by about 6. The fold models' TEST predictions are also averaged (`cvavg_*` results).
* `--cutoff-holdout 0.2` instead fits on 80% of TRAIN and chooses the cutoff on the other 20%.

For cross-fit runs the fold models can be calibrated differently from the final model, so the depth version is the safer one to report.

Bootstrap intervals (no refits; about 10 minutes):

```bash
python automl_harness_bootstrap.py --reps 2000
```

## auto-sklearn

auto-sklearn's last release is 0.15.0 (September 2022). It needs Python 3.9 and scikit-learn 0.24, and conda-forge only publishes it for Linux. Windows is not supported at all.

**Linux (or WSL2 on Windows):**

```bash
conda create -n autosklearn -c conda-forge python=3.9 auto-sklearn=0.15.0
```

**macOS (tested on Apple M1):** install the compiled pieces from conda-forge, then auto-sklearn itself with pip:

```bash
conda create -n autosklearn -c conda-forge python=3.9 scikit-learn=0.24.2 "numpy<1.24" "scipy<1.11" "pandas<1.6" pyrfr swig dask distributed pyyaml liac-arff threadpoolctl joblib distro typing_extensions psutil
```

```bash
conda run -n autosklearn pip install "pynisher>=0.6.3,<0.7" "ConfigSpace>=0.4.21,<0.5" "smac>=1.2,<1.3" emcee "setuptools<81"
```

```bash
conda run -n autosklearn pip install --no-deps --no-build-isolation auto-sklearn==0.15.0
```

Check that scikit-learn is still 0.24.2. If a later pip install upgrades it, auto-sklearn fails on import with `cannot import name '_approximate_mode'`:

```bash
conda run -n autosklearn python -c "import sklearn, autosklearn.classification; print(sklearn.__version__)"
```

**Disk and memory.** auto-sklearn saves candidate models to a temp folder and keeps up to 50 (`max_models_on_disc=50`). On the KDD98 training half (47,706 rows) on a 16 GB M1, a 30-minute, 4-job run pushed swap up until free disk fell below 3 GB. The KDD98 runs here therefore use 15 minutes, `--n-jobs 2` and `--max-models-on-disc 10`. If a run is killed, remove the leftover `autosk_*` folder in your temp directory and any orphaned `python` workers.

On macOS the harness runs auto-sklearn with `memory_limit=None`. With any memory limit, every model crashes ("current limit exceeds maximum limit"), because macOS cannot set the memory limit auto-sklearn relies on. `automl_harness_ask_mem_watchdog.sh <logfile>`, started alongside a run, does the limit's job: it kills any auto-sklearn worker above 3 GB (`LIMIT_KB`), which auto-sklearn records as a crashed model. The KDD98 auto-sklearn + AUC run used it.

Why the install is this awkward is part of the story: auto-sklearn still benchmarks well (highest F1 in a 2025 Scientific Reports comparison of 16 frameworks), but it has not been maintained since its authors moved on, largely to TabPFN.
