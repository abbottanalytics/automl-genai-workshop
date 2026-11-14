# Backup code for the Titanic GenAI workshop prompts (v1)

The code an assistant would generate for `../prompts/titanic_workshop_modeling_v1.md`, one cell per step (Steps 0–12 plus a final timing report). Use it if a live step fails or runs long, or as a reference answer.

| file | what it is |
|---|---|
| `titanic_genai_v1_backup.ipynb` | the notebook, already run, with all outputs. Open it in Jupyter and run one cell per step, or Run All. |
| `titanic_genai_v1_backup.py` | the same code as a script with `# %%` cells |
| `reports/` | everything each step printed (`stepNN_log.txt`), every table (`stepNN_<name>.csv`), and `test_predictions_all_models.csv` |
| `reports/plots/` | every plot (`stepNN_<name>.png`) |

**Data path.** The notebook reads `../data/titanic3.csv` (the repository's `data/` folder). If you move the notebook, it also looks in `data/`, in its own folder, and in your home folder. It prints the path it used.

**Step 0 (environment setup).** Checks and prints package versions, sets the global seed, starts the logs, and loads the data file once. It keeps the data in memory and saves a Parquet copy (`reports/step00_data_copy.parquet`, made when you run it; not included in the repository). Step 1 reuses that data instead of reading the file again.

**Problem parameters.** The top cell holds the same parameters as the prompt file (`TARGET`, leakage and ID columns, `CAPTURE_TARGET = 0.80`, test fraction, seed). The derived features in Step 2 and the preparation in Step 4 are Titanic-specific.

**Python.** Anaconda base (scikit-learn 1.4, xgboost 2.0, pandas, matplotlib). Step 10 uses the input shuffling package (github.com/abbottanalytics/input_shuffling, licensed AGPL-3.0) if it is cloned next to this repository or `INPUT_SHUFFLING_SRC` points to its `src/` folder; otherwise it uses the short notebook-native version included here.

**Runtime** (M1, 14 Nov 2026): about 40 seconds in total. Step 0 takes 1 s; Step 8 (tuning, 21 combinations × 5 folds) takes 22 s.

**Rerunning.** Rerunning a cell is safe: models and predictions that already exist are not refit or overwritten. To start fresh, restart the kernel and Run All.

**Key results from this run** (70/30 stratified split, seed 42; TEST = 393 passengers, 150 survivors; base rate 0.382):
- Depth to capture 80% of survivors on TEST: XGBoost and the neural network 39.2%, random forest 39.7%, logistic regression 39.9%, Naive Bayes 45.3%. Random ranking needs 80% and a perfect ranking needs 30.5%.
- The paired bootstrap (1,000 resamples) cannot separate the top four models: their depths differ by about 1 point, and the bootstrap SD of each is about 3–4 points. So the recommendation is the simplest model, **logistic regression**.
- The rankings disagree. By AUC, random forest is first (0.898); by percent correct at the base-rate threshold, the neural network is first (0.827); by depth to capture 80%, XGBoost and the neural network tie.
- Training-chosen operating point (out-of-fold on TRAIN): act on the top 46.9%. On TEST this captured 85.3% of survivors, more than the 80% target. The 5-fold models trained on 80% of TRAIN rank less well than the final model, and TEST happened to be easier than TRAIN (TEST AUCs are higher), so the training-chosen depth is conservative here.
- Percent correct at 0.5 vs at the base rate: Naive Bayes is 61.3% at 0.5 (barely above "everyone died", 61.8%) and 77.1% at the base rate.
- Tuning (CV inside TRAIN, scored on depth@80) did not beat the noise for XGBoost or the neural network, so the untuned logistic regression stays in production.
- Input shuffling on the logistic regression ranks the inputs sex, title, pclass, sibsp, is_alone, age. `cabin_known` and `family_size` show 0 because the linear model drops them as overlapping encodings.
- Association rules (mined on TRAIN, checked on TEST): 112 two-item SURVIVED rules, of which 91 are confirmed on TEST (e.g. sex=female & pclass=1: 97% survived on TRAIN, 96% on TEST), and 60 two-item DIED rules (51 confirmed).

An assistant running the prompts live will make different choices (features, model settings, bins), so its numbers will not match these exactly.
