# Titanic Predictive Modeling Workshop
## Statistical Metrics vs a Capacity Metric (no dollar value)

Companion to `kddcup1998_workshop_modeling_v8.md`. Both files follow the same step-by-step pattern and the same global standards. The difference is the business metric:

- **KDD Cup 1998** has a dollar value per success (the gift) and a cost per action ($0.68 per mailing), so models are judged on **cumulative net revenue**.
- **Titanic** has no dollar value. The question becomes one of **capacity**: how few passengers do we have to act on to find most of the survivors? Models are judged on the **depth needed to capture 80% of survivors** (lower is better).

---

# HOW TO REUSE THIS FILE FOR YOUR OWN DATA

Every step below refers to the PROBLEM PARAMETERS block, not to Titanic column names. To point this workshop at your own data:

1) Edit the PROBLEM PARAMETERS block. That is the only required edit.

2) Fill in the column lists (ID, leakage, high-cardinality text) from your own data dictionary. If you don't know them yet, leave them empty: Steps 1 and 2 will ask the assistant to propose them, and you confirm.

3) Pick the metric section that matches your problem:

| Your problem | Business metric | Use |
|---|---|---|
| Binary outcome, no dollar value (this file) | depth needed to capture X% of positives; or positives captured at a fixed capacity | this file |
| Binary outcome with a cost per action and a value per success | cumulative net revenue = value of successes − cost of actions | the KDD Cup 1998 file (Steps 7–8) |
| Probability × amount (respond, then how much) | expected value = P(success) × predicted amount | the KDD Cup 1998 file (Steps 5–8) |

4) Do NOT set cost = 1 and value = 1 to make a dollar metric work. Net value per success is then 0 and per failure −1, so the best decision is to act on nobody. Only the ratio of value to cost matters (act when P > cost / value). If you have no real dollar values, use a capacity metric as this file does.

5) Choose the secondary accuracy threshold. Percent correct at a 0.5 threshold assumes both errors cost the same. This file thresholds at the prior (the training positive rate), so a record is predicted positive when its probability is above the base rate.

6) Rare outcomes (below about 10% positive): percent correct is not useful, because predicting "no" for everyone is already more than 90% correct. Keep ROC AUC and the capture-depth metric, and drop percent correct as a ranking metric.

7) Data size: this file assumes a small dataset (about 1,300 rows), so it cross-validates inside TRAIN for tuning and cutoffs. On large data, the runtime limits in the GLOBAL NOTEBOOK EXECUTION STANDARDS take over (sampling, fewer folds).

---

# PROBLEM PARAMETERS

```yaml
dataset_file: titanic3.csv
unit_of_analysis: one passenger
decision: which passengers to prioritize (e.g., where to send limited rescue or follow-up resources)
outcome_pattern: single binary outcome
target: survived            # 1 = survived, 0 = died
positive_class: 1
id_column: RECORD_ID        # create it if missing
known_leakage_columns: [boat, body]    # recorded after the outcome was known
id_like_columns: [name, ticket]        # identifiers; use only for derived features
high_cardinality_text: [cabin, home.dest]
cost_of_action: none        # no dollar value
value_per_success: none
primary_metric: depth to capture 80% of positives (fraction of the ranked file; lower is better)
capture_target: 0.80
secondary_metric: percent correct, thresholding at the prior (training positive rate)
also_report: [ROC AUC, F1, precision, recall, lift at decile 3, depth to capture 50% and 95%]
test_fraction: 0.30
random_state: 42
```

Every step that says `target`, `capture_target`, `known_leakage_columns`, etc. means the values in this block.

---

# GLOBAL PYTHON INSTRUCTIONS

For every step:

1) Generate complete Python code for the requested task.

2) Use Python libraries appropriate for machine learning and data science including:
- pandas
- numpy
- scikit-learn
- xgboost if available
- matplotlib
- seaborn if needed

3) Preserve variables and outputs from earlier steps.

4) Do NOT rerun earlier expensive computations unless necessary.

5) Use reproducible random seeds. Every random action (train/test splits, cross-validation folds, random record sampling, model random_state parameters, random initialization, bootstrap resampling, or any other operation that draws on randomness) must use an explicitly specified seed. See GLOBAL RANDOM SEED STANDARDS below for documentation requirements.

6) Clearly comment all Python code.

7) Separate:
- data preparation
- model building
- prediction
- evaluation
- visualization

into logical code sections.

8) Store model predictions in reusable dataframes.

9) Create reusable functions whenever appropriate. In particular, write the metric functions once (Step 5) and reuse them in every later step:
- `depth_to_capture(scores, y, capture=CAPTURE_TARGET)`: sort records by score, descending; return the smallest fraction of records whose positives reach `capture` of all positives. Break ties in the score at random with a logged seed, or report how many records are tied, because tied scores (Naive Bayes, small random forests) make depth ambiguous.
- `capture_at_depth(scores, y, depth)`: fraction of all positives found in the top `depth` of the ranked records.
- `gains_table(scores, y)`: cumulative % of records vs cumulative % of positives, by decile and at every record.

10) Use consistent naming conventions across all notebook cells.

11) Assume the notebook is stateful and prior code cells have already been executed successfully.

12) Do not proceed beyond the current requested step.

13) Explain the purpose of the code before generating it.

14) After generating code, briefly explain:
- what the code does
- important outputs
- expected runtime if relevant

15) Never overwrite previously generated predictions or model objects.

16) Persist predictions in reusable dataframes indexed by RECORD_ID.

17) Persist the final feature list used for each model family and each modeling variation.

17a) For linear and logistic regression (and any other model with an intercept and no regularization), always use N-1 one-hot encoding: drop one reference level per categorical variable. A full set of N dummies is perfectly collinear with the intercept. Also check for overlapping encodings of the same source field and keep only one encoding per field. In this file, for example, `cabin_known` and the deck dummies overlap (deck = "U" exactly when the cabin is unknown), and `family_size` = sibsp + parch + 1 is an exact linear combination of two other inputs. For linear and logistic models, keep only one of each overlapping set; tree models can keep them all. Report which columns were dropped and why.

Examples:
- lr_features
- nb_features
- gbm_features
- nn_features
- rf_features

Store:
- variable names
- transformed versions
- excluded variables
- leakage removals
- encoding decisions if applicable

Do not overwrite feature lists once created.

18) Maintain a reusable dataframe called model_metadata_df that stores metadata for every model built during the workshop.

Use ONE ROW PER MODEL.

The dataframe should contain fields such as:
- model_name
- model_family
- target_variable
- feature_list_name
- feature_count
- prediction_column_name
- hyperparameters
- training metrics
- testing metrics
- depth_to_capture_80 (the primary metric)
- pct_correct_at_prior (the secondary metric)
- train_chosen_cutoff and its actual capture on TEST
- notes

Append new rows as models are created.
Do not overwrite prior metadata.

19) Persist prediction column names for all models.

Examples:
- pred_lr
- pred_nb
- pred_gbm
- pred_nn
- pred_rf
- pred_gbm_tuned

Use consistent prediction naming conventions throughout the notebook.
Never overwrite prediction columns once created.

20) Create reusable helper functions as early as possible and reuse them throughout the notebook.

21) Persist trained model objects using consistent naming conventions.

Examples:
- model_lr
- model_gbm
- model_rf
- model_gbm_tuned

Never overwrite trained model objects once created.

22) Preserve the original untransformed variables throughout the notebook, so response profiles (Step 11) and association rules (Step 12) can be read in original units.

23) Save every report to files as well as displaying it: tables as CSV in a `reports/` folder and plots as PNG in `reports/plots/`, named `stepNN_<name>`. Create the folders if needed.

--------------------

# GLOBAL RANDOM SEED STANDARDS

A single master random seed may be reused across the notebook (for example, RANDOM_STATE defined once at notebook initialization), but every individual random action must still be explicitly documented — not merely inherited implicitly.

---

## RANDOM SEED LOG REQUIREMENTS

At notebook initialization create:
- random_seed_log
- random_seed_df

Example structure:
- step_number
- action_name
- seed_value
- function_or_parameter (e.g. train_test_split(random_state=...), StratifiedKFold(random_state=...), model random_state, np.random.default_rng(), sample())
- notes

Append a new row to random_seed_log every time a random action is performed.

Examples of random actions requiring a logged seed:
- train/test splitting
- cross-validation fold assignment (Steps 5 and 8)
- any random row/record sampling
- model random_state parameters (LogisticRegression, RandomForest, GBM, MLP, etc.)
- random tie-breaking of scores
- bootstrap or resampling procedures (Step 7)
- shuffling operations (including input shuffling in Step 10)
- any np.random / random module calls

Do not overwrite prior rows in random_seed_log.

---

## RANDOM SEED REPORTING REQUIREMENTS

At the end of each step that performs a random action, print or display the relevant rows of random_seed_log for that step.

At the final notebook step, include a full random_seed_df summary table alongside the final timing report, so every random action in the notebook is traceable to a documented seed.

--------------------

# GLOBAL NOTEBOOK EXECUTION STANDARDS

All notebook steps generated for this project should follow these standards.

---

## STEP EXECUTION TIMING REQUIREMENTS

Every notebook step must include:

1) standardized timing instrumentation

2) standardized start/end reporting

3) timing persistence into:
- notebook_timing_log

4) timing summary reporting at the end of each step

5) elapsed runtime reporting in:
- seconds
- minutes

6) notebook-safe timing behavior after kernel restarts

7) runtime reporting for major within-step operations that take significant time

Examples:
- model training
- cross-validation inside TRAIN
- hyperparameter tuning
- bootstrap resampling
- input shuffling
- association rule counting
- large plotting operations

---

## NOTEBOOK INITIALIZATION REQUIREMENTS

At notebook initialization create:

- notebook_timing_log
- notebook_timing_df

Example structure:

- step_number
- step_name
- start_timestamp
- end_timestamp
- elapsed_seconds
- elapsed_minutes
- major_runtime_notes

Persist notebook_timing_df throughout the notebook.

Do not overwrite prior timing rows.

---

## STANDARD STEP TIMING VARIABLES

Use the following standardized variables in every step:

- step_start_time
- step_end_time
- step_elapsed_seconds
- step_elapsed_minutes
- step_start_timestamp
- step_end_timestamp

---

## STANDARD STEP START REPORTING

At the beginning of every step generate standardized reporting such as:

- clear visual step banners
- step number
- step description
- start timestamp

Examples:
- RUNNING STEP 5 — BUILD CLASSIFICATION MODELS
- RUNNING STEP 6 — CAPACITY EVALUATION

---

## STANDARD STEP COMPLETION REPORTING

At the end of every step report:

- step completion banner
- elapsed runtime
- major objects created
- important outputs generated
- runtime observations if relevant

---

## WITHIN-STEP RUNTIME REPORTING

For computationally expensive within-step operations, generate additional timing instrumentation.

Examples:
- model fitting runtime
- model scoring runtime
- cross-validation runtime
- hyperparameter tuning runtime
- bootstrap runtime
- input shuffling runtime
- association rule counting runtime
- response profile runtime

Persist these timings into reusable runtime summary tables where appropriate.

Examples:
- model_runtime_summary_df
- tuning_runtime_summary_df
- association_rule_runtime_df

---

## MODEL TRAINING RUNTIME REQUIREMENTS

For all model-building steps:

1) measure runtime for each model separately

2) report:
- algorithm name
- training runtime
- scoring runtime if feasible
- total runtime

3) create runtime comparison tables

4) append runtime metrics into:
- model_metadata_df

Suggested fields:
- train_runtime_seconds
- scoring_runtime_seconds
- total_runtime_seconds

---

## LONG-RUNNING OPERATION REPORTING

If any operation exceeds approximately:
- 60 seconds

then:
1) explicitly report the runtime
2) explain what operation consumed the time
3) include the runtime in summary tables
4) discuss workshop/runtime implications if appropriate

On a dataset this size (about 1,300 rows), nothing should take that long. If something does, say so: it usually means a loop that should be vectorized.

---

## PERFORMANCE COMPARISON REPORTING

Where appropriate, compare:
- runtime
- statistical performance
- capacity performance

Examples:
- ROC AUC vs runtime
- depth to capture 80% vs runtime

Discuss operational tradeoffs where relevant.

---

## TIMING PERSISTENCE REQUIREMENTS

Append timing metadata into:
- notebook_timing_log

Then periodically regenerate:
- notebook_timing_df

Persist timing dataframes throughout the notebook.

Do not overwrite prior timing history.

---

## FINAL NOTEBOOK TIMING REPORT

At the final notebook step generate:

1) notebook-wide runtime summary tables

2) step-by-step elapsed runtime tables

3) longest-running steps

4) longest-running algorithms

5) cumulative notebook runtime

6) runtime comparison visualizations if feasible

Persist:
- notebook_timing_df
- model_runtime_summary_df
- tuning_runtime_summary_df if created

---

## NOTEBOOK RESTART SAFETY

All timing instrumentation should safely support:
- notebook reruns
- kernel restarts
- partial reruns
- interrupted execution

Avoid hidden timing-state dependencies.

---

## PERFORMANCE MANAGEMENT REQUIREMENTS

Long-running steps should:
- report progress where feasible
- use progressive sampling if appropriate
- report computational tradeoffs used
- avoid unnecessary recomputation
- reuse cached objects where possible

Favor workshop practicality over exhaustive computation.

--------------------

# GLOBAL BINNING AND PANDAS DTYPE STANDARDS

These rules apply whenever values are binned with pd.cut() or pd.qcut() — decile and gains tables, response profiles (Step 11), and association rule binning (Step 12).

1. Convert interval bins to explicit float columns before sorting, aggregating, or plotting:
   - bin_left, bin_right, bin_midpoint (float64)
   - an integer bin_order column

2. Sort and plot by bin_midpoint or bin_order, never by the interval category itself (avoids alphabetical interval sorting).

3. Never apply mean(), median(), or other numeric reductions to categorical or interval dtype columns. Before any groupby aggregation, coerce the value columns with pd.to_numeric(..., errors="coerce").

4. Initialize placeholder numeric columns as float64 (for example np.nan with dtype float64).

5. Use observed=True in groupby calls on categorical columns so empty categories do not create phantom rows.

6. If numeric coercion fails, use NaN and continue; do not terminate the notebook.

7. Small data: with about 400 test rows, deciles hold about 40 records each. Report the record count in every bin.

The generated code must avoid:
- TypeError: category dtype does not support aggregation 'mean'

--------------------

# PREDICTION STORAGE STANDARDS

Persist all predictions in standardized dataframes:
- predictions_train
- predictions_test
- predictions_oof_train (out-of-fold predictions on TRAIN from cross-validation, Step 5)

Each prediction dataframe should contain:
- RECORD_ID
- actual target
- model name
- predicted probability of the positive class
- dataset indicator (TRAIN, TEST, OOF)
- model family

Never overwrite prediction dataframes.
Append new predictions instead.

--------------------

# MASTER INITIAL PROMPT

A dataset called `titanic3.csv` will be uploaded (see PROBLEM PARAMETERS). Each row is one passenger on the Titanic. The target variable is:

- survived = 1 if the passenger survived, 0 if not

Do not use any external datasets or information beyond the uploaded file. Wait until the data is uploaded before beginning the analysis.

This workshop will proceed one step at a time. Do not proceed to the next step until prompted.

The primary objective of this workshop is to compare statistical error metrics with a capacity metric for predictive modeling, when the outcome has no dollar value.

Ultimately, models will be evaluated on hold-out data by the depth needed to capture 80% of survivors: rank passengers by predicted probability of survival, and find the smallest share of the list that contains 80% of all survivors. Lower is better.

The secondary metric is percent correct, where a passenger is predicted to survive when the predicted probability is above the base rate (the survival rate in the training data), not above 0.5.

There is no cost per action and no value per success. Do not invent dollar values.

Use best practices throughout.

Algorithms should include at least:
- Logistic Regression
- Naive Bayes
- Gradient Boosting Machines
- Neural Networks
- Random Forests

Use reproducible random seeds where appropriate.

---

# STEP 1 — LOAD DATA AND AUDIT

Load the dataset named in PROBLEM PARAMETERS. If no variable named RECORD_ID is in the data, create it (1 … n, in file order) and persist it for all data frames.

Generate a "data audit" report for the following:

1) Number of rows and columns
2) Variable names
3) Variable data types
4) Distribution of the target (counts and rate); this rate is the base rate used throughout
5) Number of unique values for categorical and text variables
6) Variables with missing values: count and percent missing for each. Treat empty strings and whitespace-only strings as missing.
7) For numeric data, a table containing the min, max, mean, median, std. dev, 1st Quartile and 3rd Quartile
8) Survival rate by each categorical variable with 20 or fewer levels, with record counts
9) Survival rate for records with and without a missing value, for each variable that has missing values (missingness can itself predict the outcome)

10) Leakage screen. For every variable, report whether it could only be known after the outcome. Flag the columns in `known_leakage_columns` and propose any others. For each flagged variable, show the target rate when it is present vs missing. In this file, `boat` (lifeboat number) is filled almost only for survivors and `body` (body identification number) only for victims: both are recorded after the outcome.

11) Identifier screen. Flag columns whose values are nearly unique per row (`id_like_columns` and any others found), which a model would memorize rather than learn from.

Do not build any models yet.

REUSE NOTE: on your own data, the leakage and identifier screens are where domain knowledge matters most. The assistant can only propose candidates; you decide.

Generate Python code that performs this step, including standardized timing instrumentation and runtime summary reporting.

---

# STEP 2 — NON-LEARNED DATA PREPARATION AND FEATURE ENGINEERING

Prepare the data with transformations that do NOT learn anything from the data. Anything that learns from the data (imputation values, rare-level grouping, scaling) waits until after the split (Step 4), so the test data never influences the preparation.

Name the prepared master dataframe `modeling_prepared_df`. It holds every row, RECORD_ID, the target and all prepared inputs, and later steps refer to it by this exact name.

Tasks:

1) Remove the leakage columns (`known_leakage_columns`) from the candidate inputs. Keep them out of every model, profile and rule. Report what was removed and why.

2) Derive features from identifier and text columns, then drop the raw columns from the inputs:
- `title` from `name` (the word before the period: Mr, Mrs, Miss, Master, Dr, Rev, ...). Report the counts. Do not group rare titles yet (that is learned; see Step 4).
- `deck` = first letter of `cabin`, with "U" (unknown) when the cabin is missing
- `cabin_known` = 1 if `cabin` is present, else 0
- `family_size` = sibsp + parch + 1, and `is_alone` = 1 if family_size == 1
- `ticket_group_size` = number of passengers in the file who share the same ticket. This uses no target information, so counting over the whole file is acceptable here (the passenger list is known before the outcome); say so in the report.
- Drop `name`, `ticket`, `cabin` and `home.dest` from the inputs after deriving features. Keep them in the dataframe for reference, not as inputs.

3) Set data types:
- target: integer 0/1
- `sex`, `embarked`, `title`, `deck`: categorical
- `pclass`: keep numeric (1, 2, 3) for tree models, and record that linear models may use it either as numeric or as N-1 dummies (decided in Step 4)

4) Report:
- the list of candidate input variables, with type and source column
- missing values per candidate input (they are imputed in Step 4, not here)
- near-zero-variance inputs
- skewed numeric inputs (exclude binary 0/1 variables)
- survival rate by each new derived feature, with record counts

Do not impute, group, scale or encode yet. Do not build models yet.

REUSE NOTE: the derived features in task 2 are specific to Titanic. On your own data, replace them with features from your own identifier, text and date columns (for example, months since a reference date), and keep the rule: nothing that learns from the data happens before the split.

Generate Python code that performs this step, including standardized timing instrumentation and runtime summary reporting.

---

# STEP 3 — TRAIN / HOLD-OUT SPLIT

Create training and hold-out datasets using best practices.

Requirements:

1) Use stratified random sampling on the target, with the test fraction from PROBLEM PARAMETERS (0.30).

2) Set a random seed for reproducibility and log it in random_seed_log per the GLOBAL RANDOM SEED STANDARDS.

3) Report:
- training sample size
- testing sample size
- number of positives in each
- target rate in training and in testing
- percentage split used

4) Ensure all future models use exactly the same split for fair comparison. Persist the train/test assignment as a variable in the master dataframe.

Example:
split_flag = TRAIN or TEST

5) Store the training base rate as `BASE_RATE_TRAIN`. It is the classification threshold for the secondary metric in every later step. Never recompute it from TEST.

6) Point out how small the test set is (about 393 passengers and 150 survivors). One survivor is about 0.7% of all test survivors, so small differences between models in later steps may be noise. Step 7 measures this with a bootstrap.

Do not regenerate splits after Step 3.

Do not build models yet.

Generate Python code that performs this step, including standardized timing instrumentation and runtime summary reporting.

---

# STEP 4 — FITTED PREPARATION (LEARNED ON TRAIN ONLY)

Fit every data-dependent preparation on TRAIN, then apply the fitted values unchanged to TEST. Never fit anything on TEST.

Tasks:

1) Rare-level grouping: first map spelling variants to their common title (Mlle and Ms to Miss, Mme to Mrs) and say so; this is domain knowledge, not learned. Then group `title` levels with fewer than 10 TRAIN records into "Rare" (frequency grouping). Apply the same mapping to TEST; a level never seen in TRAIN also maps to "Rare". Do the same for `deck` if it has levels with fewer than 10 TRAIN records. Report the mapping.

2) Imputation (values learned from TRAIN):
- `age`: median within `title` group (titles carry age information: Master = boys, Miss = mostly young women). Add `age_missing` = 1 where age was imputed, and report the TRAIN survival rate with and without age.
- `fare`: median within `pclass`
- `embarked`: most frequent TRAIN value
Report every fill value and how many records it filled in TRAIN and TEST.

3) Fare: report the number of fare = 0 records and their class and survival rate. Create `fare_log` = log10(fare + 1) for the models that are sensitive to skew (logistic regression, Naive Bayes, neural network). Tree models use the raw fare.

4) Encoding:
- Tree models (GBM, random forest): one-hot all categorical inputs, or use native categorical support if available.
- Logistic regression and neural network: N-1 one-hot encoding (rule 17a): drop one reference level per categorical variable, drop overlapping encodings (`cabin_known` vs deck = "U", `family_size` vs sibsp + parch), and report the reference levels and every dropped column.
- Decide whether `pclass` enters linear models as a numeric 1/2/3 or as N-1 dummies; explain the choice.

5) Scaling: standardize numeric inputs for logistic regression and the neural network, with means and standard deviations fit on TRAIN.

6) Wrap the fitted preparation in reusable functions or sklearn Pipelines / ColumnTransformers, so that Steps 5, 8 and 10 can apply exactly the same preparation to any dataframe of original inputs (Step 10 shuffles original inputs and must re-derive the prepared columns).

7) Persist the feature list for each model family (lr_features, nb_features, gbm_features, nn_features, rf_features) and the fitted preparation objects.

Do not build models yet.

Generate Python code that performs this step, including standardized timing instrumentation and runtime summary reporting.

---

# STEP 5 — BUILD CLASSIFICATION MODELS

Build classification models to predict the target.

Use at least:
- Logistic Regression
- Naive Bayes
- Gradient Boosting Machines
- Neural Networks
- Random Forests

Use best judgment for parameter settings, and keep them modest: the training set is under 1,000 rows, so deep trees and large networks will overfit.

Requirements:

1) Fit each model on TRAIN using the Step 4 preparation for its family.

2) Save predicted probabilities of the positive class for:
- training data
- testing data
- out-of-fold training data: 5-fold stratified cross-validation inside TRAIN (seed logged), where each TRAIN record is scored by a model that did not see it. Use the same model settings and the same fitted-preparation logic, refit inside each fold.

3) Calibrate probabilities where appropriate (isotonic or Platt scaling, with cross-validation inside TRAIN only). Note that calibration does not change the ranking, so it does not change ROC AUC or the depth metric; it does change which records clear the base-rate threshold.

4) Write the metric helpers once (GLOBAL PYTHON INSTRUCTIONS rule 9): `depth_to_capture`, `capture_at_depth`, `gains_table`.

5) Compute the following metrics separately for training and testing data:
- baseline target rate
- ROC AUC
- percent correct, F1, precision and recall, all with the threshold at `BASE_RATE_TRAIN` (predict positive when P > BASE_RATE_TRAIN), NOT 0.5
- for reference only, percent correct at 0.5
- lift at the 3rd decile
- depth to capture 50%, 80% and 95% of positives

6) Training-chosen operating point (this is what you would use in production):
- On the out-of-fold TRAIN predictions, find the depth that captures `capture_target` (80%) of TRAIN positives, and the probability cutoff at that depth.
- Apply both to TEST: (a) act on the top share of TEST equal to that depth; (b) act on every TEST record with P at or above that cutoff.
- Report for each: the share of TEST acted on and the share of TEST positives actually captured. Compare with the 80% target.
- If the out-of-fold probabilities sit on a different scale from the final model's (compare the mean predicted probability), say so: the depth rule carries over and the probability cutoff does not.

7) Create:
- decile lift tables (with record counts per decile)
- gains charts (cumulative % of positives vs % of records), with a horizontal line at 80% and the depth marked for each model
- ROC curves

8) Create a parameter report for all algorithms.

9) Initialize and populate model_metadata_df with metadata and evaluation metrics for each model.

10) Persist the top 10 predictors for each model in a dict called `top_features` (model name → list of variable names). Use built-in importances for tree models, absolute standardized coefficients for logistic regression, and permutation importance on TRAIN where neither exists (Naive Bayes, neural network). Step 12 uses this dict.

Do not rank the models yet (Step 6 does that).

Generate Python code that performs this step, including standardized timing instrumentation and runtime summary reporting.

---

# STEP 6 — CAPACITY EVALUATION

Evaluate every model on the primary metric: the depth needed to capture 80% of positives.

Requirements:

1) Reuse the stored TEST predictions and the helpers from Step 5. Do not refit or rescore.

2) For each model on TEST, report:
- depth to capture 50%, 80% and 95% of positives (as % of records and as number of passengers)
- capture at fixed depths of 10%, 20%, 30%, 40% and 50%
- the random baseline (depth to capture 80% with no model = 80% of the file) and the perfect model (depth = 80% × base rate, about 31%)
- the number of tied scores near the 80% point, if any

3) Plot all models' gains curves on one chart with the random and perfect lines, and a horizontal line at 80% capture. Zoom in on the region from 40% to 60% depth, where the models differ.

4) Rank the models by depth to capture 80% (ascending), with ROC AUC and percent correct at the base-rate threshold beside it.

5) Read the depth metric as a decision: "to find 80% of survivors, we must act on the top X% of passengers". Translate X% into a number of passengers for the TEST set and for the full file of 1,309.

6) Note: depth to capture 80% measured on TEST is a fair evaluation of each model's ranking. It is NOT the depth you would choose in production, because you cannot see TEST outcomes in advance. That is what Step 5's training-chosen operating point is for. Show both side by side.

Generate Python code that performs this step, including standardized timing instrumentation and runtime summary reporting.

---

# STEP 7 — FINAL MODEL COMPARISON

Compare all models on statistical and capacity metrics, and say which one you would use.

Requirements:

1) One comparison table, one row per model:
- ROC AUC (TEST)
- percent correct at the base-rate threshold (TEST)
- percent correct at 0.5 (TEST, reference only)
- lift at the 3rd decile
- depth to capture 80% (TEST)
- training-chosen depth, and the actual TEST capture at that depth
- training runtime

2) Rank the models separately by ROC AUC, by percent correct at the base-rate threshold, and by depth to capture 80%. Show where the rankings disagree.

3) Paired bootstrap on TEST: resample TEST records with replacement (1,000 resamples, seed logged), and for each resample compute every model's depth to capture 80% and ROC AUC on the same resampled records. Report:
- a 90% interval for each model's depth to capture 80%
- for each pair of models, the 90% interval of the difference in depth, and the share of resamples in which the first model is better
- which differences are larger than the noise

4) Recommend a production model. Use depth to capture 80% as the primary criterion, prefer the simpler model when the bootstrap cannot separate two models, and state the training-chosen depth you would use to act.

5) Explain why percent correct at 0.5 and percent correct at the base-rate threshold rank the models differently, and what each threshold assumes about the cost of the two errors.

6) Store `production_model_name` and `production_depth` (the training-chosen depth).

Generate Python code that performs this step, including standardized timing instrumentation and runtime summary reporting.

---

# STEP 8 — HYPERPARAMETER TUNING

Tune the two best model families from Step 7 for the primary metric.

Requirements:

1) Tune with cross-validation inside TRAIN only (5-fold stratified, seed logged). The scoring function is depth to capture 80%, computed on each validation fold (lower is better; pass it to the search as a negative or with greater_is_better=False). Do NOT use TEST to choose settings.

2) Keep the search small: no more than 20 combinations per family (grid or seeded random search). Report the search space.

3) Refit the best settings on all of TRAIN. Score TEST once, and report the same metrics as Step 7 for the tuned models next to the untuned ones.

4) Repeat the training-chosen operating point from Step 5 for the tuned models (out-of-fold depth that captures 80% on TRAIN, applied to TEST).

5) Decide whether tuning helped: compare the change in depth to capture 80% with the bootstrap noise from Step 7. If the gain is within the noise, keep the untuned model as the production model and say so.

6) Persist tuned models and predictions under new names (model_gbm_tuned, pred_gbm_tuned, ...). Update `production_model_name` and `production_depth` only if the tuned model is chosen, and record the decision in model_metadata_df.

Note the difference from the KDD Cup workshop: there, tuning was scored on TEST for teaching purposes. Here it uses cross-validation inside TRAIN, which is how tuning should be done in production. TEST is used only once, to report the result.

Generate Python code that performs this step, including standardized timing instrumentation and runtime summary reporting.

---

# STEP 9 — MEMORY OPTIMIZATION AND OBJECT CLEANUP

Report the notebook's memory use and remove intermediate objects that later steps do not need.

Requirements:

1) List the largest objects in memory (name, type, approximate size).

2) Keep everything needed later: `modeling_prepared_df`, the fitted preparation objects, every model object, the prediction dataframes, feature lists, model_metadata_df, `top_features`, `production_model_name`, `production_depth`, the logs.

3) Delete only temporary objects (cross-validation fold models, search objects, plot buffers) and run garbage collection. Report memory before and after.

4) Do not delete or overwrite any model, prediction or metadata object.

On a dataset this size the notebook uses little memory; the point of this step is the habit, which matters on large data.

Generate Python code that performs this step, including standardized timing instrumentation and runtime summary reporting.

---

# STEP 10 — INPUT SHUFFLING VARIABLE IMPORTANCE

Measure how much the production model's scores depend on each input variable, using the INPUT SHUFFLING algorithm defined below.

This step computes variable importance only. Plots of how the score changes with each variable are produced in Step 11.

---

## INPUT SHUFFLING ALGORITHM (AUTHORITATIVE DEFINITION)

Follow this definition exactly. Do NOT substitute:
- sklearn.inspection.permutation_importance
- drop-column importance
- SHAP values
- model-native feature importances (tree gain, coefficients)

Input shuffling measures prediction instability: when an input's values are scrambled, how much does each record's score move? It does not need the target variable.

### Definitions

- S = the production model's score function from Step 7 or 8: predicted probability of the positive class, computed from ORIGINAL input variables
- A = the analysis set (the testing partition; it is small enough to use in full)
- s_i = baseline score of record i in A
- M = number of shuffles per variable (N_SHUFFLES)

### Algorithm

1. Baseline. Score every record in A with S → s_i.

2. Shuffle and rescore. For each input variable X_j, for m = 1 … M:
   - randomly permute the values of X_j across ALL records in A
   - hold every other column fixed
   - rescore with S → s_i^(j,m)
   - compute the per-record delta d_i^(j,m) = s_i^(j,m) − s_i

3. Per-record dispersion. For each record i and variable j, compute the dispersion of its M deltas:
   - IQR_ij = 75th percentile − 25th percentile of d_i^(j,1..M)  (primary)
   - SD_ij  = standard deviation (ddof=1) of d_i^(j,1..M)         (complementary diagnostic)

4. Variable influence. Average the per-record dispersion over records:
   - mean_record_iqr_j = mean over i of IQR_ij   ← RANKING STATISTIC
   - also report median_record_iqr, mean_record_std, mean_abs_delta (mean of |d| pooled over all records and shuffles), and fraction_abs_delta_gt_eps (fraction of |d| > 0.01, "instability frequency")

5. Rank variables by mean_record_iqr, descending. Rank 1 = most influential.

### Top-group importance

The top group is the records whose BASELINE score s_i is in the top `production_depth` share of A (the share you would act on; about the top 40–50%). Compute top-group importance by reusing the per-record IQR_ij from step 3, averaged over top-group records only.

Do NOT reshuffle within the top group. Every shuffle draws values from the full analysis set, so a top-group record can receive any value seen in the population — which is what happens when its true value is unknown.

### Secondary measure: capacity degradation

For each shuffle (j, m), also compute, against the actual target on A:
- depth to capture 80% of positives
- ROC AUC
- capture at the baseline production depth

Degradation_j = mean over m of the shuffled metric − baseline metric (for depth, a positive number means more passengers must be acted on).

This is the classic "metric drop" view. Report it alongside the instability ranking; it never replaces mean_record_iqr as the ranking statistic.

### Which columns are shuffled

- Shuffle each ORIGINAL input variable of the production model, one at a time (for example `age`, `fare`, `sex`, `title`, `pclass`), not its encoded or transformed columns.
- Build a scoring function that takes a dataframe of original inputs and reapplies the Step 4 fitted preparation (imputation, grouping, log, encoding, scaling) before scoring. Shuffling `fare` must therefore also change `fare_log` wherever the model uses it; shuffling `title` changes all of its dummies together.
- Derived features count as their own inputs (`family_size`, `deck`, ...); say in the report that shuffling `sibsp` does not change `family_size` unless the scorer re-derives it, and choose one approach consistently.
- Never shuffle RECORD_ID, split_flag, the target, or prediction columns.

---

## REFERENCE IMPLEMENTATION

The reference implementation is Dean Abbott's input_shuffling_v2 package:

https://github.com/abbottanalytics/input_shuffling (licensed AGPL-3.0)

Its README.md ("The algorithm" section) describes the same algorithm. If the repository is available, clone it and call the engine directly from Python:

```python
import sys
INPUT_SHUFFLING_REPO_DIR = "input_shuffling"   # path to the cloned repository
sys.path.insert(0, f"{INPUT_SHUFFLING_REPO_DIR}/src")

from input_shuffling_v2.in_shuff_engine import run_engine
from input_shuffling_v2.in_shuff_importance import build_feature_summary, record_stat_matrix

in_shuff_cfg = {
    "run": {"seed": RANDOM_STATE},
    "shuffling": {"n_shuffles": N_SHUFFLES},
    "importance": {"eps": 0.01, "ranking_stat": "mean_record_iqr"},
    "performance": {"shuffle_batch": 10, "n_jobs": 1},   # n_jobs must be 1 with an in-memory scorer
    "schema": {"prediction_columns": ["production_score"]},
}

X_base = shuffle_analysis_df[shuffle_feature_list].reset_index(drop=True)
baseline_scores = production_scorer(X_base)          # callable: DataFrame -> 1-D numpy array
feature_items = [(f, i) for i, f in enumerate(shuffle_feature_list)]

shuffle_results = run_engine(
    X_base, feature_items, production_scorer, baseline_scores, in_shuff_cfg,
    metric_fns={"depth_to_capture_80": depth_80_fn},  # fn(scores) -> float, aligned to X_base rows
)
overall_importance_df = build_feature_summary(shuffle_results, in_shuff_cfg)
record_iqr_df = record_stat_matrix(shuffle_results, "iqr")   # rows = records, columns = variables
```

Notes on the engine:
- It seeds every shuffle deterministically from (seed, feature index, shuffle index), so results do not depend on batching.
- `metric_fns` values are returned per shuffle in each result's `metric_values`; use them for capacity degradation.
- Top-group importance = mean of `record_iqr_df` rows where baseline_scores is in the top `production_depth` share.

Use notebook-native execution only. Do NOT use shell scripts, subprocess calls, or the package's command-line interface.

If the repository is not available, implement the algorithm above directly in the notebook. The implementation must produce the same quantities: per-record IQR across M shuffles, mean_record_iqr ranking, and top-group importance from the same shuffles. Seed each permutation with np.random.default_rng([RANDOM_STATE, feature_index, shuffle_index]).

---

## BASELINE PRODUCTION SCORE

Regenerate the production model's scores directly from the preserved model and preparation objects.

The generated code must:
1. build `production_scorer` from the preserved objects (tuned or untuned, per Step 8)
2. regenerate baseline scores on the analysis set with `production_scorer`
3. verify the regenerated scores match the stored production prediction column (report the max absolute difference; warn if it exceeds 1e-6)

Persist the regenerated scores as `production_score_test` (indexed by RECORD_ID) for reuse in Step 11.

---

## CONFIGURATION AND RUNTIME

Define these constants at the top of the step:

```python
N_SHUFFLES = 50                 # M; the reference implementation's default (the data is small)
TOP_GROUP_FRACTION = production_depth
```

Runtime is roughly (number of variables) × N_SHUFFLES × (time to score A once); on about 400 rows this is seconds. Time one scoring pass first and report the estimate.

Log the shuffle seeds in random_seed_log. Report shuffling runtime per variable and in total.

---

## OUTPUT REQUIREMENTS

1. Overall importance table: all variables ranked by mean_record_iqr, with median_record_iqr, mean_record_std, mean_abs_delta, fraction_abs_delta_gt_eps, number of records, and N_SHUFFLES.

2. Top-group importance table: the same columns computed on top-group records.

3. Rank comparison table: overall rank, top-group rank, rank shift (overall rank − top-group rank), and the change in mean_record_iqr. Name the variables that become more and less important in the top group.

4. Capacity degradation table for every variable: depth-to-capture-80% degradation, AUC degradation, capture degradation at the production depth.

5. Plots:
   - horizontal bar chart of the top 10 overall variables by mean_record_iqr
   - the same for the top group
   - side-by-side comparison of overall vs top-group ranks

---

## OBJECT CREATION REQUIREMENTS

Persist:
- production_scorer
- production_score_test
- shuffle_feature_list
- overall_importance_df
- top_group_importance_df
- importance_rank_shift_df
- record_iqr_df
- capacity_degradation_df
- input_shuffling_discussion_df

---

## DISCUSSION REQUIREMENTS

Generate concise discussion summaries covering:
1. why variable importance changes in the top group
2. why capacity degradation may rank variables differently from prediction instability
3. what happens to the importance of `sex` and `title`, which carry overlapping information (shuffling one leaves the other intact)
4. how input shuffling differs from permutation importance, SHAP, and model-native importances

Generate Python code that performs this step, including standardized timing instrumentation and runtime summary reporting.

---

# STEP 11 — RESPONSE PROFILES FOR TOP VARIABLES

Show how the production score changes as each of the most important variables changes, for the full testing population and for the top group (the top `production_depth` share by production_score_test).

Reuse from Step 10: production_score_test, overall_importance_df, top_group_importance_df. Do NOT retrain or rescore models.

## VARIABLES AND POPULATIONS

Plot the union of the top 3 overall variables and the top 3 top-group variables, without duplicates. For each, compare the overall testing population with the top group.

## WHAT TO PLOT

- x-axis = the original (untransformed) input variable, binned: numeric variables in quantile bins (default 5, because TEST has only about 400 rows; duplicates dropped); categorical variables one bin per level, ordered by record count
- y-axis = mean production score in each bin
- also plot the actual survival rate in each bin for the overall population, so the score can be checked against what happened

Label these plots "response profiles". Call them partial dependence plots only if the code actually rescores a controlled grid of values while holding other variables fixed.

## BINNING AND SPARSE BINS

Follow the GLOBAL BINNING AND PANDAS DTYPE STANDARDS, plus:
1. Create bin edges once, from the overall testing population, and apply the same edges to the top group.
2. MIN_BIN_RECORDS_OVERALL = 20, MIN_BIN_RECORDS_TOP_GROUP = 10. Flag bins below these as sparse; do not draw lines through sparse bins; list them in a diagnostic table.
3. Show record counts per bin.
4. Do not smooth across bins.

## OBJECT CREATION REQUIREMENTS

Persist:
- response_profile_variables
- response_profile_bin_edges
- response_profile_plot_dataframes
- response_profile_sparse_bins_df
- response_profile_discussion_df

## DISCUSSION REQUIREMENTS

1. how the score responds to each top variable (direction, shape, thresholds), for example age near childhood, fare, class
2. where the score and the actual survival rate disagree, and what that says about the model
3. why relationships may differ in the top group
4. where sparse bins limit what can be concluded

Generate Python code that performs this step, including standardized timing instrumentation and runtime summary reporting.

---

# STEP 12 — SUPERVISED ASSOCIATION RULE DISCOVERY

Build supervised association rules to find combinations of INPUT characteristics associated with each outcome:

INPUT VARIABLES  --->  TARGET OUTCOME

Examples:
- sex = female + pclass = 1 or 2  ---> SURVIVED = 1
- title = Master + family_size small ---> SURVIVED = 1
- sex = male + pclass = 3          ---> DIED = 1

Use Python code only. Do NOT use shell scripts, subprocess calls, or command-line execution. Do NOT rebuild predictive models.

---

## DATA SOURCE: MINE ON TRAIN, CHECK ON TEST

The data is small, and TEST has already been used to evaluate (Steps 5–7), tune-check (Step 8) and explain (Steps 10–11) the models. So:
- mine the rules on TRAIN (`modeling_prepared_df[split_flag == "TRAIN"]`, original units, with the Step 4 imputation and grouping applied)
- then re-apply every kept rule to TEST and report its TEST support, confidence and lift next to the TRAIN values

The master dataframe is `modeling_prepared_df`. Do NOT search for alternative dataframe names.

---

## RULE STRUCTURE REQUIREMENTS

1) Antecedents contain ONLY input variables. Never the target, leakage columns (`known_leakage_columns`), RECORD_ID, split_flag, predictions, scores, or anything derived from the target.

2) Consequents contain ONLY outcome flags:
- OUTCOME_SURVIVED: target == 1
- OUTCOME_DIED: target == 0

3) Exactly ONE consequent per rule; at most 2 antecedent items per rule.

---

## DATA PREPARATION FOR RULE MINING

1) Candidate inputs: all model inputs from Step 4 in original form (about 12–15 variables, so no pre-selection is needed). Add any variables in `top_features` that are missing.

2) Numeric variables: 4 quantile bins with pd.qcut on TRAIN (duplicates="drop"), with "missing" as its own bin if any remain. Save the edges and apply the same edges to TEST. Follow the GLOBAL BINNING AND PANDAS DTYPE STANDARDS. Variables with few distinct values (sibsp, parch, family_size, pclass) use their values or small groups (0, 1, 2, 3+) instead of quantiles.

3) Categorical variables: one item per level (after the Step 4 grouping).

4) Binary 0/1 inputs: create only the "=1" item.

5) One-hot encode with prefix "ANT::" for antecedents and "CONS::" for outcomes. Drop antecedent items with TRAIN support below 0.02 (about 18 passengers) or above 0.95.

---

## ASSOCIATION RULE MINING

Count the rules directly with matrix products. No itemset-mining library is needed:

```python
X = transaction_matrix_train[ant_cols].to_numpy(dtype=np.float32)    # rows x antecedent items (0/1)
n_A   = X.sum(axis=0)                  # records matching each single item
n_AB  = X.T @ X                        # records matching each pair of items
for outcome in outcome_cols:
    c = transaction_matrix_train[outcome].to_numpy(dtype=np.float32)
    n_AC  = X.T @ c                    # single item AND outcome
    n_ABC = X.T @ (X * c[:, None])     # pair of items AND outcome
```

From these counts:
- support(A) = n_A / n_rows, confidence = n_AC / n_A, lift = confidence / outcome rate (same for pairs, using n_AB and n_ABC)
- records_with_outcome = n_AC (or n_ABC)

Keep a rule only if:
- the antecedent matches at least 2% of TRAIN rows (`MIN_ANTECEDENT_SUPPORT = 0.02`), and
- records_with_outcome >= 10 (`MIN_OUTCOME_RECORDS = 10`).

Ignore the diagonal of n_AB and pairs of items from the same source variable.

Do NOT use mlxtend fpgrowth plus association_rules(); the matrix version computes every 1- and 2-item rule exactly in well under a second.

---

## RULE FILTERING AND VALIDATION

1) Lift >= 1.10 on TRAIN.
2) Keep 1-antecedent rules in their own table for reference; focus reporting on 2-antecedent rules.
3) A 2-antecedent rule must beat both of its 1-antecedent parents: its lift must be at least 5% higher than the better parent's lift for the same outcome. Otherwise it is not an interaction.
4) TEST check: for every kept rule, compute TEST records matched, TEST confidence and TEST lift. Flag rules whose TEST lift is below 1.0, or that match fewer than 10 TEST records, as not confirmed.

Create separate rule sets:
- association_rules_survived
- association_rules_died

---

## RULE PROFILING (NO DOLLAR VALUES)

For each rule report:
- TRAIN and TEST records matched
- TRAIN and TEST outcome rate (confidence)
- TRAIN and TEST lift
- share of all TRAIN positives the rule captures (for SURVIVED rules), which links rules to the capacity metric: a segment that is 10% of passengers and holds 25% of survivors is a high-priority segment
- confirmed on TEST (yes/no)

---

## REPORTING

Generate:
- top 20 two-antecedent rules for each outcome, sorted by TRAIN lift then records matched
- the 1-antecedent reference table
- support vs confidence scatter plot (bubble size = lift) for each rule set, marking rules not confirmed on TEST
- TRAIN lift vs TEST lift scatter plot for all kept rules, with the diagonal

---

## COMPARISON WITH MODEL RESULTS

Compare the top rule antecedents with the Step 10 input shuffling importance and with `top_features`. Discuss:
1) whether the important model variables also appear in strong rules
2) which interactions the rules find (for example sex × class) and whether a main-effects model such as logistic regression can represent them
3) how many rules hold up on TEST, and what the drop from TRAIN lift to TEST lift says about mining many rules on a small sample (multiple testing)
4) how the rules could be used as a simple, explainable prioritization list, and how its depth to capture 80% would compare with the production model

---

## OBJECT PERSISTENCE

Persist:
- ar_train_df, ar_test_df
- ar_bin_edges
- transaction_matrix_train, transaction_matrix_test
- supervised_rules_all
- association_rules_survived
- association_rules_died
- association_rule_runtime_df

Generate Python code that performs this step, including standardized timing instrumentation and runtime summary reporting.

Finally, generate the FINAL NOTEBOOK TIMING REPORT and the full random_seed_df summary.

---

# DISCUSSION QUESTIONS

## Discussion Question 1

Why might the model with the best ROC AUC not need the smallest depth to capture 80% of survivors?

---

## Discussion Question 2

Percent correct at a 0.5 threshold and percent correct at the base-rate threshold can rank models differently. What does each threshold assume about the cost of a missed survivor vs a false alarm?

---

## Discussion Question 3

Why does setting cost = 1 and value = 1 not turn this into a profit problem? When is a capacity metric (depth to capture X% of positives) the right business metric?

---

## Discussion Question 4

The depth measured on TEST and the depth chosen on TRAIN (out-of-fold) differ. Which one would you put into production, and how much of the 80% target did it actually capture on TEST?

---

## Discussion Question 5

What would happen to ROC AUC and to the depth metric if `boat` were left in the data? How would you notice leakage on your own data if no one told you which columns were recorded after the outcome?

---

## Discussion Question 6

With about 400 test passengers, how large must a difference between two models be before you believe it? What did the bootstrap in Step 7 say?

---

## Discussion Question 7

Why were the association rules mined on TRAIN and checked on TEST, when the KDD Cup workshop mined on TEST?

---

# KEY CONCEPTS

| Concept | Meaning |
|---|---|
| ROC AUC | Global ranking quality |
| Percent correct at the base rate | Classification accuracy when a record is called positive if P > the training positive rate |
| Lift | Targeting quality at a depth |
| Depth to capture 80% | Smallest share of the ranked list that holds 80% of positives: the capacity needed |
| Training-chosen depth | The depth chosen on out-of-fold TRAIN predictions, used to act on new data |
| Fitted preparation | Imputation, grouping and scaling learned on TRAIN only and applied unchanged to TEST |
| Leakage | Inputs recorded after the outcome (boat, body), which make a model look far better than it can be |
| Input Shuffling | Variable influence measured as per-record score instability when an input is permuted |
| Supervised association rules | Interpretable 2-way segments linked to an outcome, mined on TRAIN and checked on TEST |

---

# OVERALL WORKSHOP OBJECTIVE

The goal of this workshop is to demonstrate that:

- the best statistical model is not necessarily the best model for the decision
- when there is no dollar value, a capacity metric (how much of the list must we act on to find most of the positives?) states the business question directly
- the classification threshold is a business choice: 0.5 is a default, not a decision
- every learned preparation step and every operating point must be chosen on training data, or hold-out results will be optimistic
- the same step-by-step process works on any binary-outcome dataset once the PROBLEM PARAMETERS are filled in
