# KDD Cup 1998 AutoML Experiments
## What do AutoML engines optimize, and what does that do to the business result?

Optional companion to the workshop "AutoML and GenAI" (Abbott Analytics, MLWEU Nov 2026).

Use these prompts one step at a time with an LLM coding assistant, or run the matching scripts in the repository's `automl_harness/` folder (`automl_harness_*.py`). The harness does Steps 3–8 for you. The prompts let you see each decision being made.

The prompts use `data/cup98lrn_prepped_allrecords_w_TargetD.tsv` (34 columns). The harness results in the deck used a different preparation of the same 95,412 records (`data/cup98lrn_automl_prepped_with_split.tsv`, 38 inputs with the raw categorical fields), so numbers from these prompts will not match the deck's AutoML tables exactly.

---

# GLOBAL PYTHON INSTRUCTIONS

For every step:

1) Generate complete, commented Python code for the requested step only. Do not proceed beyond the current step.

2) Assume the notebook is stateful and earlier cells have run. Do not rerun expensive fits unless asked.

3) Use explicit random seeds for every random operation (splits, model seeds, AutoML seeds). Log each seed in a `random_seed_log` list of dicts (step, purpose, seed).

4) Time every model fit and every prediction call. Keep a `run_log_df` with one row per AutoML run: tool, version, settings, fit seconds, predict seconds.

5) Keep predictions in dataframes indexed by RECORD_ID. Never overwrite earlier predictions: name them by tool and setting (e.g. `p_ag_default`, `p_ask_auc`).

6) For linear and logistic regression, always use N-1 one-hot encoding (drop a reference level). Full-N dummies are perfectly collinear with the intercept. Also check for overlapping encodings of the same source field (for example the combined RFA_2 dummies and the RFA_2A dummies) and keep only one.

7) Never use TARGET_B or TARGET_D, or anything derived from them, as an input.

8) The test set is used once, for the final numbers. Choices (thresholds, mailing depth, which model to keep) are made on training data only.

9) After the code, explain briefly what it does, the important outputs, and the expected runtime.

---

# MASTER INITIAL PROMPT

The file `cup98lrn_prepped_allrecords_w_TargetD.tsv` will be uploaded. It has predictor variables and two targets:

- TARGET_B = responded to the mailing (1) or not (0)
- TARGET_D = donation amount (0 for non-responders)

Mailing one record costs $0.68.

We will compare two AutoML engines, AutoGluon-Tabular and auto-sklearn, against each other and against individually built models. The question is not just "which is most accurate" but "what does each engine optimize by default, and what does that do to cumulative net revenue (CNR)?"

CNR: sort the records by a model score, highest first; walk down the list adding (actual donation − $0.68) for each record mailed. The CNR curve peaks at the best mailing depth.

Work one step at a time. Wait until prompted to move on.

---

# STEP 0 — ENVIRONMENTS

Give the commands to create two separate Python environments:

1) AutoGluon 1.6.x (Python 3.11): `pip install "autogluon.tabular[all]"`.

2) auto-sklearn 0.15.0, the last release (Sept 2022). It needs Python 3.9 and scikit-learn 0.24, and conda-forge publishes it for Linux only (on Windows use WSL2). For macOS, follow the recipe in `automl_harness/README.md`.

Explain why they cannot share one environment.

**Questions**
- auto-sklearn has had no release since 2022. What happened to the project? Should that affect whether you use it in production?
- What does it cost you, in time and risk, to depend on an unmaintained package?

---

# STEP 1 — LOAD AND SPLIT

1) Load the file. Create RECORD_ID if it is missing.

2) Split 50/50 into TRAIN and TEST, stratified on TARGET_B, seed 22. Store `split_flag` in the master dataframe.

3) Assign every TRAIN row to one of 5 stratified folds (seed 42) and store `cv_fold` (TEST rows get none). The folds are used to get out-of-fold TRAIN predictions, which choose the mailing cutoff in Step 7. Every final model is still fit on all of TRAIN.

4) Report the rows and the TARGET_B rate in each part, and the mean TARGET_D among responders.

5) Report two baselines on TEST:
   - majority-class accuracy (predict "no response" for everyone)
   - CNR if we mail everyone

**Questions**
- What accuracy does predicting "nobody responds" achieve? Keep that number in mind for the rest of the exercise.
- Why must the mailing cutoff be chosen on out-of-fold TRAIN predictions, and not on TEST or on in-sample TRAIN predictions?
- An alternative is to hold out 20% of TRAIN for the cutoff. What does that cost, and what does cross-fitting cost instead?

---

# STEP 2 — INDIVIDUAL REFERENCE MODELS

Build two individual models for TARGET_B on all TRAIN rows:

1) Logistic regression (N-1 one-hot, scaled numeric inputs, seed 22).
2) A gradient boosting model (XGBoost or LightGBM, default settings, seed 22).

Also fit a responder-only TARGET_D model for each family (linear regression with N-1 encoding, and the boosted regressor), trained on TRAIN rows with TARGET_B = 1.

Also get out-of-fold TRAIN predictions for both models with the Step 1 folds (`cross_val_predict` with those folds, or a loop).

Report AUC, log loss, mean predicted probability vs the actual rate, and lift in the top 10% on TEST. Store the out-of-fold TRAIN predictions and the TEST predictions.

**Questions**
- Is the mean predicted probability close to the actual response rate? Why does that matter for an expected-value rule?

---

# STEP 3 — AUTOGLUON WITH DEFAULTS

Fit `TabularPredictor(label="TARGET_B")` on all TRAIN rows with no other settings (default preset, default metric, no time limit). Show the leaderboard.

Report on TEST:
- `predict()` accuracy, and the % of records it predicts as responders
- AUC, log loss, mean predicted probability
- fit and predict time

**Questions**
- What metric did AutoGluon optimize? (Look at `predictor.eval_metric`.)
- How does its accuracy compare with the Step 1 baseline? How many people would its default decision mail?
- AUC is well above 0.5. How can a model rank donors well and still mail nobody?
- Look at the leaderboard's validation scores. How many models tie? Which one did AutoGluon keep, and would it keep the same one on a different sample?

Then get out-of-fold TRAIN predictions: this run is not bagged, so repeat the whole fit (classifier and, after Step 6, gift model) on 4 of the 5 Step 1 folds and predict the fifth. Record the model AutoGluon keeps in each fold.

---

# STEP 4 — AUTO-SKLEARN WITH DEFAULTS

Fit `AutoSklearnClassifier` on all TRAIN rows with a 15-minute budget (`time_left_for_this_task=900`), seed 42, and otherwise default settings. On large data, use `n_jobs=2` and `max_models_on_disc=10` to limit disk and memory. On macOS, `memory_limit=None`.

Print `sprint_statistics()` and `leaderboard()`. Report the same TEST metrics as Step 3.

**Questions**
- What is the best validation score in `sprint_statistics()`? Compare it with the Step 1 baseline.
- What is in the final ensemble? What does `predict_proba` return for every record?
- auto-sklearn's ensemble builder keeps a model only if its validation loss is strictly better than a dummy model's. Under accuracy at a 5% response rate, what happens?
- `predict()` takes the class with the larger probability, which means a 0.5 cutoff, and auto-sklearn has no setting to change it. What cutoff would make sense for a 5% response rate?

Cross-fit this run the same way as Step 3 to get out-of-fold TRAIN predictions (this takes 5 more 15-minute runs; skip it if short of time and use only Rule 1 and Rule 2 in Step 7). Record each fold's ensemble members.

---

# STEP 5 — CHANGE THE METRIC

Refit both tools on all TRAIN rows with ROC AUC as the metric:

1) auto-sklearn: `metric=autosklearn.metrics.roc_auc`, same budget as Step 4.
2) AutoGluon: `eval_metric="roc_auc"`, `presets="best_quality"`, `time_limit=3600` (about an hour; use 600 for a quicker look).

Report the same TEST metrics, plus the ensemble members.

Out-of-fold TRAIN predictions: `best_quality` bags its models, so `predictor.predict_proba_oof()` already gives them. auto-sklearn needs the Step 4 cross-fit.

**Questions**
- What changed in auto-sklearn's ensemble once the metric could tell models apart?
- Compare mean predicted probability with the actual response rate for each run. Which runs are well calibrated?
- AUC is computed from the ranking alone. What does it not measure that a dollar decision needs?

---

# STEP 6 — GIFT MODELS AND EXPECTED VALUE

For each AutoML run, fit a responder-only TARGET_D regressor with the same tool on TRAIN rows with TARGET_B = 1 (AutoGluon `problem_type="regression"`; auto-sklearn `AutoSklearnRegressor`, 5-minute budget).

Compute EV = P(TARGET_B) × max(predicted gift, 0) for TEST rows and for the out-of-fold TRAIN rows (out-of-fold P × out-of-fold gift). Do the same for the Step 2 pairs (logistic × linear, boosted × boosted).

**Questions**
- Why fit the gift model on responders only?
- Which metric does each tool optimize for the gift model by default (RMSE, R²)? Are those the right ones for ranking donors?

---

# STEP 7 — THE MAILING DECISION

For every model (Step 2–6) and for two scores (P alone, and EV), compute CNR on TEST under these rules:

1) **Tool default:** mail the records `predict()` labels 1 (P only).
2) **Fixed EV rule:** mail when EV > $0.68 (EV only).
3) **Cutoff chosen on training data:** on the out-of-fold TRAIN rows, sort by the score and find where CNR peaks. Apply it to TEST two ways:
   - **threshold:** mail every TEST record scoring at or above the score at the peak;
   - **depth:** mail the same share of TEST as the share of TRAIN mailed at the peak.
4) **Hindsight peak on TEST:** the CNR peak on TEST itself. Label it clearly as an optimistic upper bound. It is not a result.

Plot CNR curves on TEST for the AutoML runs and the Step 2 models, with each rule's mailing point marked.

**Questions**
- Rule 2 assumes P and the predicted gift are calibrated in dollars. Which models break that assumption, and in which direction?
- Why is Rule 4 not an honest result? When would it be acceptable (for example, averaged over many test samples)?
- How far is Rule 3 from Rule 4 for each model? What does that gap tell you?
- Compare the mean out-of-fold P with the mean TEST P for each run. For the cross-fit runs, the out-of-fold scores come from five different fold models. Does a dollar threshold chosen on them still mean the same thing for the final model? Which version of Rule 3 is safer, and why?

---

# STEP 8 — COMPARISON TABLE

Build one table with a row per model and these columns:

- tool, settings, fit + predict minutes
- default-decision accuracy and the baseline
- AUC, log loss, mean predicted vs actual rate
- CNR: tool default, fixed EV rule, training-chosen cutoff (P and EV), mail-all; hindsight peak shown separately

Sort by CNR from the training-chosen EV depth.

Then bootstrap the CNR results (2,000 paired replicates, resampling TEST rows; a second version also resamples the out-of-fold TRAIN rows and re-chooses the cutoff). Report a 90% interval for each model and for the difference between the best run and each other run.

**Questions**
- Does the ranking by AUC match the ranking by CNR? Where does it differ, and why?
- How much of the gap between the best and worst runs comes from the engine, and how much from the metric you told it to optimize?
- Was the hour-long AutoGluon run worth it compared with its two-minute default run?
- Which differences are larger than the bootstrap intervals? Which would you report as real?

---

# STEP 9 (OPTIONAL) — TABULAR FOUNDATION MODELS

AutoGluon 1.6's `extreme` preset adds tabular foundation models (TabICL, TabPFN, TabDPT) to the ensemble. These models predict from training rows given as context, without a model search. A GPU is strongly recommended.

If you have a GPU, rerun Step 5's AutoGluon fit with `presets="extreme"` and add it to the Step 8 table.

**Questions**
- Do foundation models help on a 5% response, noisy, 38-feature problem like this one?
- What do they cost in prediction time compared with a boosted tree?

---

# DISCUSSION QUESTIONS

1. AutoML engines default to accuracy for classification. Why is that a poor default for rare-event problems such as response modeling?
2. Neither engine can optimize cumulative net revenue directly. What is the closest metric you can give them, and why?
3. Why can two models with the same AUC produce very different revenue under an expected-value rule?
4. Where should the mailing decision live: inside the AutoML engine or in your own code? Why?
5. When is an unmaintained AutoML package still worth using?

---

# KEY CONCEPTS

| Concept | Meaning |
|---|---|
| Default metric | What the engine optimizes if you say nothing (accuracy for both engines) |
| Better-than-dummy filter | auto-sklearn drops models that do not beat a dummy predictor on the chosen metric |
| ROC AUC | Ranking quality over the whole population; ignores calibration |
| Calibration | Whether predicted probabilities match actual rates; needed for dollar rules |
| Expected value | P(respond) × predicted gift |
| Out-of-fold predictions | TRAIN predictions from models that did not see that row (cross-fitting or bagging) |
| Training-chosen cutoff | Mailing threshold or depth picked where CNR peaks on out-of-fold TRAIN predictions |
| Paired bootstrap | Resample rows once per replicate and score every model on the same rows |
| Hindsight peak | CNR peak on TEST; an upper bound, not a result |
