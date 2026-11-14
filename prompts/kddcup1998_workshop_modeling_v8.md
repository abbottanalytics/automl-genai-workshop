# KDD Cup 1998 Predictive Modeling Workshop
## Business Metrics vs Statistical Metrics

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

5) Use reproducible random seeds. Every random action (train/test splits, random record sampling, model random_state parameters, random initialization, bootstrap resampling, or any other operation that draws on randomness) must use an explicitly specified seed. See GLOBAL RANDOM SEED STANDARDS below for documentation requirements.

6) Clearly comment all Python code.

7) Separate:
- data preparation
- model building
- prediction
- evaluation
- visualization

into logical code sections.

8) Store model predictions in reusable dataframes.

9) Create reusable functions whenever appropriate.

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

17a) For linear and logistic regression (and any other model with an intercept and no regularization), always use N-1 one-hot encoding: drop one reference level per categorical variable. A full set of N dummies is perfectly collinear with the intercept. Also check for overlapping encodings of the same source field and keep only one encoding per field. In this file, the combined RFA_2 dummies (L1F_RFA_2, L3D_RFA_2, ...) overlap with RFA_2F and the RFA_2A dummies (E_RFA_2A, F_RFA_2A, G_RFA_2A). In fact E_RFA_2A + F_RFA_2A + G_RFA_2A + L3D_RFA_2 + L4D_RFA_2 = 1 for every row, which is exactly collinear with the intercept; using them all gives coefficients in the billions and infinite scores later. For linear and logistic models, drop the combined L*_RFA_2 dummies; tree models can keep them. Report which columns were dropped and why.

Examples:
- target_b_logistic_features
- target_b_gbm_features
- target_d_rf_features
- target_d_log_features

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
- transformed_target_flag
- feature_list_name
- feature_count
- prediction_column_name
- hyperparameters
- training metrics
- testing metrics
- cumulative_net_revenue
- optimal_depth
- notes

Append new rows as models are created.
Do not overwrite prior metadata.

19) Persist prediction column names for all models.

Examples:
- pred_lr_target_b
- pred_gbm_target_b
- pred_rf_target_d
- pred_log_target_d
- pred_ev_gbm_rf

Use consistent prediction naming conventions throughout the notebook.
Never overwrite prediction columns once created.

20) Create reusable helper functions as early as possible and reuse them throughout the notebook.

21) Persist trained model objects using consistent naming conventions.

Examples:
- model_lr_target_b
- model_gbm_target_b
- model_rf_target_d
- model_gbm_log_target_d

Never overwrite trained model objects once created.

22) Preserve original untransformed variables and targets throughout the notebook for comparison and inverse transformation purposes.

Append new rows as models are created.
Do not overwrite prior metadata.

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
- function_or_parameter (e.g. train_test_split(random_state=...), model random_state, np.random.seed(), sample())
- notes

Append a new row to random_seed_log every time a random action is performed.

Examples of random actions requiring a logged seed:
- train/test splitting
- stratified sampling
- any random row/record sampling (e.g. for PDP, plotting, or runtime-reduction subsampling)
- model random_state parameters (LogisticRegression, RandomForest, GBM, MLP, etc.)
- bootstrap or resampling procedures
- SMOTE or other synthetic sampling if used
- shuffling operations (including input shuffling in Step 12)
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
- hyperparameter tuning
- neural network fitting
- random forest fitting
- cumulative revenue calculations
- input shuffling
- association rule mining
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
- RUNNING STEP 4 — BUILD TARGET_B CLASSIFICATION MODELS
- RUNNING STEP 7 — CUMULATIVE NET REVENUE EVALUATION

---

## STANDARD STEP COMPLETION REPORTING

At the end of every step report:

- step completion banner
- elapsed runtime
- major objects created
- important outputs generated
- runtime observations if relevant

Examples:
- total elapsed seconds
- total elapsed minutes
- expensive operations encountered
- major runtime bottlenecks

---

## WITHIN-STEP RUNTIME REPORTING

For computationally expensive within-step operations, generate additional timing instrumentation.

Examples:
- model fitting runtime
- model scoring runtime
- hyperparameter tuning runtime
- neural network runtime
- random forest runtime
- GBM runtime
- cumulative revenue runtime
- association rule mining runtime
- input shuffling runtime
- PDP / response profile runtime

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

Examples:
- logistic regression runtime
- GBM runtime
- neural network runtime
- random forest runtime

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

---

## PERFORMANCE COMPARISON REPORTING

Where appropriate, compare:
- runtime
- statistical performance
- business performance

Examples:
- ROC AUC vs runtime
- cumulative net revenue vs runtime
- neural network accuracy vs runtime
- GBM performance vs runtime

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

These rules apply whenever values are binned with pd.cut() or pd.qcut() — decile tables, response profiles (Step 13), and association rule binning (Step 14).

1. Convert interval bins to explicit float columns before sorting, aggregating, or plotting:
   - bin_left, bin_right, bin_midpoint (float64)
   - an integer bin_order column

2. Sort and plot by bin_midpoint or bin_order, never by the interval category itself (avoids alphabetical interval sorting).

3. Never apply mean(), median(), or other numeric reductions to categorical or interval dtype columns. Before any groupby aggregation, coerce the value columns with pd.to_numeric(..., errors="coerce").

4. Initialize placeholder numeric columns as float64 (for example np.nan with dtype float64).

5. Use observed=True in groupby calls on categorical columns so empty categories do not create phantom rows.

6. If numeric coercion fails, use NaN and continue; do not terminate the notebook.

The generated code must avoid:
- TypeError: category dtype does not support aggregation 'mean'

--------------------

# PREDICTION STORAGE STANDARDS

Persist all predictions in standardized dataframes.

For classification models:
predictions_target_b_train
predictions_target_b_test

For donation models:
predictions_target_d_train
predictions_target_d_test

Each prediction dataframe should contain:
- RECORD_ID
- actual target
- model name
- prediction
- dataset indicator
- model family
- transformation indicator if applicable

Never overwrite prediction dataframes.
Append new predictions instead.

--------------------

# MASTER INITIAL PROMPT

A dataset called `cup98lrn_prepped_allrecords_w_TargetD.tsv` will be uploaded. It contains predictor variables along with two target variables:

- TARGET_B = binary response variable (1=responded, 0=did not respond)
- TARGET_D = donation amount

Do not use any external datasets or information beyond the uploaded file. Wait until the data is uploaded before beginning the analysis.

This workshop will proceed one step at a time. Do not proceed to the next step until prompted.

The primary objective of this workshop is to compare statistical error metrics versus business metrics for predictive modeling.

Ultimately, models will be evaluated using cumulative net revenue on hold-out data.

Assume mailing cost per record is $0.68 unless instructed otherwise.

Use best practices throughout.

Algorithms should include at least:
- Logistic Regression
- Naive Bayes
- Gradient Boosting Machines
- Neural Networks
- Random Forests if feasible

Use reproducible random seeds where appropriate.

---

# STEP 0 — ENVIRONMENT SETUP (no analysis yet)

Prepare the environment only. Do not audit, summarize, or model the data in this step.

1) Read and acknowledge the global standards above (Python, random seeds, notebook execution and timing, binning and dtypes, prediction storage). Do not restate them; confirm in one or two lines that they will be followed.

2) Check that these packages import: pandas, numpy, scikit-learn, xgboost, scipy, matplotlib, pyarrow. Print each version. If one is missing, try to install it. If it cannot be installed, say so and name the substitute you will use for the rest of the workshop (for example scikit-learn's HistGradientBoostingClassifier in place of xgboost). Do not change that choice later without telling me.

3) Set the global random seed, start the seed log and the step timing log, and create the output folders for reports, tables and plots.

4) Locate the uploaded file cup98lrn_prepped_allrecords_w_TargetD.tsv and load it once with pandas (tab-separated). Report only the file path used, the number of rows and columns, and the load time. Keep the dataframe in memory and save a copy in Parquet format, so later steps reuse it instead of reading the TSV again. Do not create RECORD_ID yet; that is part of Step 1.

5) Finish with the standard step-completion report and wait. Do not start Step 1.

---

# STEP 1 — LOAD DATA

Use the dataframe loaded in Step 0 (reload it from the Parquet copy or the uploaded file only if it is no longer in memory). If no variable named RECORD_ID is in the data, create it and persist it for all data frames

Generate a "data audit" report for the following:

1) Number of rows and columns
2) Variable names
3) Variable data types
4) Distribution of TARGET_B
5) Summary statistics for TARGET_D
6) Number of unique values for categorical variables
7) Variables with missing values and count of missing values for each variable
8) For Numeric data, a table containing the min, max, mean, median, std. dev, 1st Quartile and 3rd Quartile

Do not build any models yet.

Generate Python code that performs this step, including standardized timing instrumentation and runtime summary reporting.

---

# STEP 2 — DATA PREPARATION

Prepare the data for modeling.

Name the prepared master dataframe `modeling_prepared_df`. It holds every row, RECORD_ID, the targets and all prepared inputs, and later steps (including Step 14) refer to it by this exact name.

Tasks:

1) Convert TARGET_B into a categorical/string target suitable for binary classification.

2) Replace missing numeric values with the median.

3) Handle missing categorical values using best practices and clearly explain the method used.

4) Report:
- missing values per variable
- percent missing
- variables with near-zero variance
- highly skewed numeric variables (exclude binary 0/1 variables: a rare dummy always looks "skewed", and transforming it is meaningless)
- suspicious variables that may represent leakage

5) Identify the monetary variables. This prepared file already contains log10 versions of the monetary variables (LASTGIFT_log10, AVGGIFT_log10, MINRAMNT_log10, and others ending in _log10) and has no missing values. Do not assume raw names such as LASTGIFT or RAMNT exist: list the monetary columns that are actually present and say whether each is already transformed.

6) Create a table showing skewness values for non-binary numeric variables.

Do not build models yet.

Generate Python code that performs this step, including standardized timing instrumentation and runtime summary reporting.


---

# STEP 3 — TRAIN / HOLD-OUT SPLIT

Create training and hold-out datasets using best practices.

Requirements:

1) Use stratified random sampling based on TARGET_B if feasible.

2) Set a random seed for reproducibility and log it in random_seed_log per the GLOBAL RANDOM SEED STANDARDS.

3) Report:
- training sample size
- testing sample size
- response rate in training
- response rate in testing
- percentage split used

4) Ensure all future models use exactly the same split for fair comparison.  Persist train/test assignment as a variable in the master dataframe.

Example:
split_flag = TRAIN or TEST

Do not regenerate splits after Step 3.

Do not build models yet.

Generate Python code that performs this step, including standardized timing instrumentation and runtime summary reporting.

---

# STEP 4 — BUILD TARGET_B CLASSIFICATION MODELS

Build classification models to predict TARGET_B.

Use at least:
- Logistic Regression
- Naive Bayes
- Gradient Boosting Machines
- Neural Networks
- Random Forests if feasible

Use best judgment for parameter settings.

Requirements:

1) Save predicted probabilities for:
- training data
- testing data

2) calibrate probabilities using isotonic calibration or Platt scaling where appropriate

3) Use the overall TARGET_B response rate as the probability threshold for classification accuracy rather than 0.5.

4) Compute the following metrics separately for training and testing data:
- baseline response rate
- ROC AUC
- F1
- classification accuracy
- precision
- recall
- lift at the 3rd decile

5) Create:
- decile lift tables
- gains charts if possible
- ROC curves if possible

6) Rank models by:
- lift at the 3rd decile
- ROC AUC secondary

7) Create a parameter report for all algorithms.

8) Persist the final feature list used for each classification model.

9) Initialize and populate model_metadata_df with metadata and evaluation metrics for each TARGET_B model.

10) Persist the top 10 predictors for each classification model in a dict called `target_b_top_features` (model name → list of variable names). Use built-in importances for tree models, absolute standardized coefficients for logistic regression, and permutation importance on a sample of the testing data where neither exists (Naive Bayes, neural network). Step 14 uses this dict.

Do not build TARGET_D models yet.

Generate Python code that performs this step, including standardized timing instrumentation and runtime summary reporting.

---

# STEP 5 — BUILD TARGET_D REGRESSION MODELS

Build regression models to predict TARGET_D.

Use ONLY records where TARGET_B = 1 in BOTH training and testing data.

Use at least:
- Linear Regression
- Gradient Boosting Regression
- Neural Network Regression
- Random Forest Regression if feasible

Requirements:

1) Save predictions for:
- training data
- testing data

2) Compute the following metrics separately for training and testing data:
- RMSE
- MAE
- R-squared
- lift at the 3rd decile based on donation amount

3) Create:
- predicted vs actual plots
- donation decile tables

4) Rank models by:
- donation lift at 3rd decile
- RMSE, secondary if there is a tie

5) Report parameter settings for all models.

6) Persist the final feature list used for each regression model.

7) Append TARGET_D model results to model_metadata_df.

Do not build transformed models yet.

Generate Python code that performs this step, including standardized timing instrumentation and runtime summary reporting.

---

# STEP 6 — BUILD TRANSFORMED TARGET_D MODELS

Build additional TARGET_D regression models using transformed variables.

Tasks:

1) Create a transformed target variable:

log(1 + TARGET_D)

2) Identify highly skewed numeric variables and apply appropriate transformations where justified. Exclude binary 0/1 variables.

Focus especially on:
- monetary variables
- gift amount variables
- lifetime donation variables

The monetary inputs in this file are already log10-transformed (names ending in _log10). Do not transform them again; report them as "already transformed".

3) Clearly report:
- variables transformed
- transformation applied
- skewness before transformation
- skewness after transformation

4) Rebuild the TARGET_D regression models using the transformed target.

Use the same algorithms and same train/test split as before.

5) Save predictions for:
- training data
- testing data

6) Report:
- RMSE
- MAE
- R-squared
- donation lift at 3rd decile

7) Compare transformed versus non-transformed models.

8) Persist the final feature list used for each regression model.

9) Append transformed TARGET_D model results to model_metadata_df.

After scoring transformed models, inverse-transform predictions back to original dollar scale before evaluating cumulative net revenue.

Do not compute cumulative net revenue yet.

Generate Python code that performs this step, including standardized timing instrumentation and runtime summary reporting.

---

## STEP 7 — COMPUTE CUMULATIVE NET REVENUE FOR ALL MODELS

Using the testing dataset only, compute cumulative net revenue curves for ALL previously built models.
### IMPORTANT
Do NOT rebuild any models.

Use:
Net Revenue = actual donation amount - mailing cost
Assume mailing cost = $0.68 per mailed record.
---
### REQUIREMENTS

### 1. Score ALL testing records for all models.
---
### 2. For TARGET_D models:
- use all testing records, not just testing records with TARGET_B = 1
- non-responders contribute:
0 - 0.68 = -0.68 net revenue
---
## IMPORTANT ARCHITECTURAL REQUIREMENT FOR TARGET_D MODELS
Although TARGET_D models were TRAINED only on responder records
(TARGET_B = 1), ALL TARGET_D models must SCORE the FULL testing dataset,
including non-responders.

This means:

1. Generate TARGET_D predictions for EVERY testing record.

2. Do NOT filter the testing dataset to responders only during revenue evaluation.

3. Do NOT remove or filter testing rows because predictions are null. Instead, generate missing predictions before evaluation.

4. If TARGET_D prediction columns currently exist only for responder rows, generate predictions for the remaining testing rows before revenue evaluation.

5. Non-responders must remain in the ranked mailing file and contribute:

0 donation - mailing cost

Example:

0 - 0.68 = -0.68 net revenue

6. Revenue curves for TARGET_D models must therefore include:
- responders
- non-responders
- profitable mailings
- unprofitable mailings

7. A response rate of 1.0 at optimal depth would indicate an invalid evaluation.

8. TARGET_D prediction columns must contain predictions for ALL testing rows before cumulative revenue calculations begin.

9. TARGET_D models must therefore be rescored against the full testing population even if they were originally trained only on responder rows.
---
### 3. Sort records descending by model score.
---
### 4. Compute:
- cumulative donations
- cumulative mailing costs
- cumulative net revenue
---
### 5. Generate:
- cumulative net revenue tables
- cumulative net revenue plots
- cumulative response curves if possible
---
### 6. Report for each model:
- maximum cumulative net revenue: label it "TEST peak (optimistic)" in all tables and plots
- depth at maximum cumulative net revenue
- number of mailed records at optimum
- ROI at optimum

IMPORTANT: The TEST peak is chosen by looking at the testing data itself, so it is an optimistic upper bound, not what the model would earn in practice. In production, the mailing threshold or depth must be set on training data (for example, where cumulative net revenue peaks on out-of-fold training predictions) and then applied unchanged to new data. State this in a note under the results table.
---
### 7. Rank ALL models by:
- maximum cumulative net revenue descending (TEST peak, optimistic)
---
### 8. Update model_metadata_df with cumulative net revenue metrics.
---
Do not yet combine TARGET_B and TARGET_D scores.

Generate Python code that performs this step, including standardized timing instrumentation and runtime summary reporting.

---

# STEP 8 — BUILD EXPECTED VALUE SCORES

Create expected value scores using:

Expected Donation Score =
Predicted Probability TARGET_B
*
Predicted TARGET_D

(Multiply the two predictions. Clip negative TARGET_D predictions at 0 before multiplying.)

Tasks:

1) Use:
- TARGET_B probability predictions
- TARGET_D predictions
- transformed TARGET_D predictions if applicable

2) Create expected value scores for all compatible model combinations.

3) Score ALL testing records.

4) Compute cumulative net revenue curves using:

Net Revenue = actual donation amount - mailing cost

5) Report:
- maximum cumulative net revenue, labeled "TEST peak (optimistic)"
- optimal mailing depth (TEST peak)
- ROI
- cumulative donation capture

Repeat the Step 7 note: the TEST peak is an optimistic upper bound; in production the threshold or depth is set on training data.

6) Compare:
- TARGET_B-only models
- TARGET_D-only models
- transformed TARGET_D models
- expected value models

7) Rank all approaches by:
- maximum cumulative net revenue descending (TEST peak, optimistic)

8) Append expected value model combinations and results to model_metadata_df.

Do not perform hyperparameter tuning yet.

Generate Python code that performs this step, including standardized timing instrumentation and runtime summary reporting.

---

# STEP 9 — FINAL MODEL COMPARISON

Create a final comparison report summarizing all modeling approaches.

---

## IMPORTANT REUSE REQUIREMENT

Do NOT:
- retrain models
- rebuild models
- regenerate predictions
- recreate train/test splits

Reuse all previously-created:
- prediction columns
- evaluation tables
- metadata tables
- cumulative revenue tables
- expected value tables

Step 9 is strictly a reporting, comparison, and recommendation step.

---

## COLUMN VALIDATION REQUIREMENTS

Before merging dataframes:
- validate join keys exist
- validate prediction_column_name uniqueness where required
- avoid accidental column overwrites during merges

If duplicate columns occur after merging:
- explicitly resolve them
- do not silently overwrite columns

---

## METRIC EXTRACTION REQUIREMENTS

Metrics stored in nested dictionaries or JSON fields must be extracted safely.

If a metric does not exist for a model family:
- return NaN
- do not raise exceptions

Examples:
- ROC AUC may not exist for regression models
- RMSE may not exist for classification models
- lift metrics may not exist for expected value models

Use safe parsing and extraction logic throughout.

---

## INCLUDE THE FOLLOWING STANDARD ML METRICS

### Classification Metrics
- ROC AUC
- F1
- Accuracy
- Precision
- Recall

### Regression Metrics
- RMSE
- MAE
- R-squared

---

## INCLUDE THE FOLLOWING BUSINESS METRICS

- lift at 3rd decile
- donation lift at 3rd decile
- maximum cumulative net revenue (label as "TEST peak (optimistic)"; add the Step 7 production note under the final table)
- ROI
- optimal mailing depth
- optimal mail count
- cumulative donation capture

---

## CANONICAL APPROACH TYPE LABELS

Use the following exact canonical values for the variable:

approach_type

Do NOT invent alternative labels.

Required canonical values:
- target_b_classification
- target_d_regression
- transformed_target_d_regression
- expected_value

If prior steps created different labels, normalize them to these canonical labels before creating final comparison tables.

All filtering, grouping, ranking, and summary tables in Step 9 must use these canonical labels consistently.

---

## CREATE SUMMARY TABLES COMPARING

1. TARGET_B classification models
2. TARGET_D regression models
3. transformed TARGET_D regression models
4. expected value models

Create:
- individual summary tables
- combined summary tables
- rankings by statistical metrics
- rankings by business metrics

---

## BEST MODEL SELECTION REQUIREMENTS

Before selecting best models using:
- .iloc[0]
- .head(1)
- indexing operations
- ranking logic

validate that the filtered dataframe is non-empty.

If a category contains no rows:
- print a warning
- continue processing remaining categories
- do not raise exceptions
- do not terminate the notebook

---

## DISCUSSION REQUIREMENTS

Generate concise discussion sections covering:

1. Why statistical metrics and business metrics differ

2. Why some models rank differently under:
- ROC AUC
- RMSE
- lift
- cumulative net revenue

3. Why expected value models may outperform:
- TARGET_B-only models
- TARGET_D-only models

4. Operational implications for campaign targeting

5. Why optimal mailing depth matters operationally

6. Risks of selecting models using only statistical metrics

---

## RECOMMENDATION REQUIREMENTS

Recommend:
- best statistical TARGET_B model
- best statistical TARGET_D model
- best business model
- recommended deployment strategy

Recommendations should explicitly reference:
- statistical metrics
- business metrics
- operational tradeoffs

---

## FINAL REPORTING REQUIREMENTS

Generate:
- final combined comparison dataframe
- business ranking tables
- statistical ranking tables
- recommendation summary tables
- discussion summary tables

Sort the final combined comparison table by:
- maximum cumulative net revenue descending (TEST peak, optimistic)

---

Do not perform hyperparameter tuning yet.

Generate Python code that performs this step, including standardized timing instrumentation and runtime summary reporting.

---

# STEP 10 — HYPERPARAMETER TUNING

Using the modeling strategy that produced the highest cumulative net revenue, perform hyperparameter tuning.

---

## IMPORTANT REUSE REQUIREMENT

Do NOT:
- recreate train/test splits
- rebuild preprocessing logic from scratch
- regenerate feature engineering logic
- regenerate calibration logic unless explicitly required

Reuse all previously-created:
- train/test splits
- preprocessing pipelines
- calibrated model structures
- feature lists
- transformed variables
- prediction frameworks
- cumulative revenue evaluation logic

Use the SAME training/testing split from prior steps.

---

## OPTIMIZATION OBJECTIVE

Optimize for:
- maximum cumulative net revenue on testing data

Use:

Net Revenue = actual donation amount - mailing cost

Assume mailing cost = $0.68 per mailed record.

---

## IMPORTANT VALIDATION NOTE

This workshop intentionally optimizes cumulative net revenue using the testing dataset because the workshop objective is instructional.

However, production workflows should:
- tune on validation data
- preserve the test set for final evaluation
- avoid leaderboard-style overfitting

Include comments in the Python code explaining this distinction.

---

## NESTED ESTIMATOR / PIPELINE REQUIREMENTS

Some models may be wrapped inside higher-level sklearn estimators such as:
- CalibratedClassifierCV
- pipelines
- nested estimators
- probability calibration wrappers

Hyperparameter tuning code must correctly support nested estimator parameter namespaces.

Examples:

A plain pipeline may use:
- model__n_estimators
- model__max_depth

A calibrated classifier wrapping a pipeline may require:
- estimator__model__n_estimators
- estimator__model__max_depth

The tuning code must:
1. detect nested estimator structures dynamically
2. apply parameter prefixes correctly
3. avoid hardcoding assumptions about estimator depth
4. support both calibrated and non-calibrated models safely

Do not assume all models are simple sklearn pipelines.

---

## SAFE PARAMETER APPLICATION REQUIREMENTS

Before applying tuned parameters:
- validate parameter names against estimator.get_params()

If invalid parameter names are detected:
- print a warning
- adjust parameter prefixes if appropriate
- do not terminate the notebook unnecessarily

The notebook should continue processing safely whenever possible.

---

## MODEL CLONING REQUIREMENTS

When tuning models:
- clone existing trained model architectures
- preserve preprocessing pipelines
- preserve calibration structure
- preserve feature engineering structure

Only modify hyperparameters being tuned.

Do not overwrite original trained model objects.

Persist tuned models separately.

Examples:
- tuned_model_gbm_target_b
- tuned_model_rf_log_target_d

---

## HYPERPARAMETER TUNING REQUIREMENTS

Tune parameters appropriate to each algorithm.

Possible examples include:
- number of trees
- learning rate
- max depth
- minimum samples per leaf
- regularization parameters
- hidden layer sizes if neural networks are used

Use reasonable tuning grids suitable for workshop runtime constraints.

Avoid excessively large tuning grids.

---

## EXPECTED VALUE MODEL REQUIREMENTS

If the best-performing business strategy is an expected value model:

Expected Value =
Predicted Probability TARGET_B
*
Predicted TARGET_D

then:

1. Tune the underlying component models separately.

2. Recompute expected value scores after tuning.

3. Recompute cumulative net revenue curves after tuning.

4. Compare tuned versus untuned expected value performance.

---

## PERFORMANCE COMPARISON REQUIREMENTS

Compare:
- tuned versus untuned models
- business metrics
- statistical metrics where applicable

Report:
- parameter settings tested
- best parameter settings
- performance improvements
- degradation if tuning hurts performance
- evidence of overfitting if present

---

## OVERFITTING ASSESSMENT REQUIREMENTS

Discuss:
- why tuning on testing data can overfit
- why small improvements may not generalize
- why simpler untuned models may sometimes be preferable

Generate concise summary discussion tables for:
- tuning gains
- tuning risks
- deployment considerations

---

## FINAL REVENUE REPORTING REQUIREMENTS

Create:
- cumulative net revenue tables
- cumulative net revenue plots
- tuned vs untuned comparison tables
- final business ranking tables for tuned models

Report:
- maximum cumulative net revenue
- optimal mailing depth
- optimal mail count
- ROI at optimum

---

## FINAL RECOMMENDATION REQUIREMENTS

Report the final recommended production model.

Include:
- model type
- tuned parameters
- expected value strategy if applicable
- deployment recommendation
- operational considerations
- validation caveats if tuning used testing data

---

## OBJECT CREATION REQUIREMENTS

Persist:
- tuning result tables
- tuned model objects
- tuned revenue tables
- tuned vs untuned comparison tables
- final recommendation tables
- Do not overwrite original untuned models or results.

## RUNTIME MANAGEMENT REQUIREMENTS

Because the dataset is large:

- avoid excessively large parameter grids
- prefer small, targeted tuning grids
- limit the total number of tuning combinations
- reuse previously-generated predictions where possible
- avoid recomputing expensive preprocessing steps unnecessarily
- Limit tuning to no more than 20 total parameter combinations unless instructed otherwise.

Favor workshop runtime practicality over exhaustive search.

Do not overwrite original untuned models or results.

Generate Python code that performs this step, including standardized timing instrumentation and runtime summary reporting.

---

# STEP 11 — MEMORY OPTIMIZATION AND OBJECT CLEANUP

Optimize notebook memory usage by removing intermediate objects that are no longer needed after hyperparameter tuning.

---

## IMPORTANT PRESERVATION REQUIREMENTS

Do NOT delete or overwrite final production artifacts.

Preserve:
- all training datasets
- all testing datasets
- all train/test split assignments
- all split_flag variables
- all final prediction dataframes
- all final expected value score tables
- all business ranking tables
- all tuning results
- all final recommendation tables
- all final feature lists
- all cumulative revenue tables needed for future analysis
- model_metadata_df
- reusable helper functions
- reusable scoring infrastructure
- reusable evaluation infrastructure
- reusable plotting infrastructure
- memory profiling helper variables
- memory profiling helper functions
- psutil imports if loaded
- runtime configuration constants
- cleanup helper functions
- cleanup helper configuration variables

---

## FINAL MODEL PROTECTION REQUIREMENTS

Before deleting any model-related object:

1. Identify the final recommended production model from Step 10.

2. Explicitly preserve:
- the final selected production model object
- the final tuned winning model object
- the final untuned comparison model if applicable
- the final selected production prediction column
- the final selected production feature list
- the final selected preprocessing pipeline
- the final selected calibration wrapper if present
- the final selected transformed-variable framework if applicable
- the final selected expected value scoring components if applicable
- all component models required for expected value scoring
- all production scoring logic required to generate future predictions

3. Preserve all objects required to:
- generate future predictions
- rerun cumulative net revenue evaluation
- run Step 12 input shuffling
- run Step 13 response profiles
- run Step 14 association rule discovery

4. Print the preserved final production model names before cleanup begins.

Do NOT delete any object required for future interpretation or scoring.

---

## TRAINING / TESTING DATA PROTECTION REQUIREMENTS

Preserve:
- all training dataframes
- all testing dataframes
- all responder-only training subsets
- all transformed-target training subsets
- all prediction-aligned testing subsets
- all feature-engineered datasets
- all transformed-variable datasets
- all expected value scoring datasets

Preserve all datasets needed for:
- input shuffling
- interaction detection
- rescoring
- future validation
- business metric recomputation

Do NOT regenerate these datasets later unless explicitly instructed.

---

## SAFE GLOBAL OBJECT ITERATION REQUIREMENTS

When iterating through globals():

NEVER iterate directly over:
- globals().items()
- globals().keys()

because globals() may change size during iteration.

Instead:
- always iterate over static copies such as:
  - list(globals().items())
  - list(globals().keys())

This requirement applies to:
- helper function detection
- object scanning
- object deletion
- preservation logic
- cleanup reporting

The generated code must avoid:
- RuntimeError: dictionary changed size during iteration

---

## CLEANUP INFRASTRUCTURE PROTECTION REQUIREMENTS

Do NOT delete objects required by the cleanup process itself.

Examples include:
- MEMORY_PROFILING_AVAILABLE
- psutil
- gc
- helper functions
- memory measurement utilities
- object size estimation utilities
- cleanup reporting utilities
- cleanup configuration constants

The cleanup step must preserve all objects required until:
- memory reporting
- cleanup reporting
- final summary reporting

have completed successfully.

Only delete temporary objects after:
- memory reporting
- cleanup reporting
- final summary reporting

are complete.

The generated code must avoid:
- NameError caused by deleting cleanup infrastructure too early

---

## MEMORY CLEANUP REQUIREMENTS

Identify and remove intermediate objects that are no longer needed.

Examples may include:
- temporary transformed dataframes
- temporary tuning iteration artifacts
- temporary plotting objects
- duplicated matrices
- temporary revenue tables
- obsolete cloned models
- temporary preprocessing artifacts
- temporary hyperparameter search objects
- temporary shuffled prediction objects
- temporary diagnostic tables

Use:
- del
- gc.collect()

where appropriate.

---

## SAFE CLEANUP REQUIREMENTS

Before deleting objects:
1. verify they are not required by later steps
2. avoid deleting final production artifacts
3. avoid deleting reusable helper functions
4. avoid deleting reusable scoring infrastructure
5. avoid deleting preserved train/test datasets
6. avoid deleting preserved production model objects
7. avoid deleting cleanup infrastructure objects before reporting is complete

If uncertain whether an object is safe to remove:
- preserve it
- print a warning instead

The notebook should continue safely even if some cleanup operations fail.

---

## PYTHON MEMORY MANAGEMENT REQUIREMENTS

Use best practices where feasible:
- remove large unused pandas objects
- remove duplicated matrices
- remove temporary tuning artifacts
- invoke garbage collection
- remove obsolete intermediate objects
- minimize unnecessary dataframe duplication

Avoid:
- excessive copying
- preserving obsolete temporary structures
- deleting reusable production infrastructure
- deleting cleanup infrastructure prematurely

---

## MEMORY REPORTING REQUIREMENTS

Generate:
- memory usage before cleanup if feasible
- memory usage after cleanup if feasible
- estimated memory reclaimed
- deleted object list
- preserved object list
- protected production artifact list

If exact memory measurement is not feasible:
- provide approximate reporting
- do not fail the notebook

---

## FINAL CLEANUP REPORT REQUIREMENTS

Generate:
- cleanup summary dataframe
- preserved object summary
- deleted object summary
- protected production artifact summary
- estimated memory savings summary

Persist final cleanup reporting tables.

Do not overwrite preserved final artifacts.

---

## OBJECT CREATION REQUIREMENTS

Persist:
- cleanup_summary_df
- preserved_object_summary_df
- deleted_object_summary_df
- protected_production_artifacts_df
- memory_usage_summary_df if feasible

Do not overwrite final production artifacts.

---

## EXECUTION SAFETY REQUIREMENTS

The generated cleanup code must:
- execute safely inside a stateful Jupyter notebook
- preserve all objects needed for subsequent workshop steps
- preserve all input shuffling infrastructure
- preserve all interaction detection infrastructure
- preserve all scoring pipelines
- preserve all tuned production models
- preserve all train/test datasets

Avoid aggressive cleanup that risks breaking later notebook steps.

Favor workflow continuity over maximal memory reclamation.

---

Generate Python code that performs this step, including standardized timing instrumentation and runtime summary reporting.


---

# STEP 12 — INPUT SHUFFLING VARIABLE IMPORTANCE

Measure how much the final production model's scores depend on each input variable, using the INPUT SHUFFLING algorithm defined below.

This step computes variable importance only. Plots of how the score changes with each variable are produced in Step 13.

---

## INPUT SHUFFLING ALGORITHM (AUTHORITATIVE DEFINITION)

Follow this definition exactly. Do NOT substitute:
- sklearn.inspection.permutation_importance
- drop-column importance
- SHAP values
- model-native feature importances (tree gain, coefficients)

Input shuffling measures prediction instability: when an input's values are scrambled, how much does each record's score move? It does not need the target variable.

### Definitions

- S = the final production score function from Step 10 (for an expected value model, S = P(TARGET_B) × predicted TARGET_D, scored as one combined function)
- A = the analysis set (testing partition, or a seeded sample of it — see CONFIGURATION)
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

   (Dispersion of the deltas equals dispersion of the shuffled scores, since s_i is a constant per record.)

4. Variable influence. Average the per-record dispersion over records:
   - mean_record_iqr_j = mean over i of IQR_ij   ← RANKING STATISTIC
   - also report median_record_iqr, mean_record_std, mean_abs_delta (mean of |d| pooled over all records and shuffles), and fraction_abs_delta_gt_eps (fraction of |d| > 0.01, "instability frequency")

5. Rank variables by mean_record_iqr, descending. Rank 1 = most influential.

### Top-3-decile importance

The top 3 deciles are the records whose BASELINE score s_i is in the highest 30% of A.

Compute top-decile importance by reusing the per-record IQR_ij from step 3, averaged over top-3-decile records only.

Do NOT reshuffle within the top-decile subset. Every shuffle draws values from the full analysis set, so a top-decile record can receive any value seen in the population — which is what happens when its true value is unknown.

### Secondary measure: business degradation

For each shuffle (j, m), also compute the business metrics of the shuffled scores s^(j,m) against the actual TARGET_D on A:
- maximum cumulative net revenue (mailing cost $0.68)
- ROI at the optimum
- optimal mailing depth
- donation capture at the baseline optimal depth

Business degradation_j = baseline metric − mean over m of the shuffled metric.

This is the classic "metric drop" view. Report it alongside the instability ranking; it never replaces mean_record_iqr as the ranking statistic.

### Which columns are shuffled

- Shuffle each ORIGINAL input variable of the production model, one at a time.
- Build a scoring function that takes a dataframe of original input variables and recomputes any derived columns (log transforms, encodings, scaling) before scoring. Shuffling LASTGIFT must therefore also change LASTGIFT_log wherever a model uses it.
- For expected value models, shuffle the union of both component models' original inputs; the scoring function routes the right columns to each component and multiplies the results.
- Never shuffle RECORD_ID, split_flag, targets, or prediction columns.

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
    metric_fns={"max_cum_net_revenue": max_cum_net_revenue_fn},  # fn(scores) -> float, aligned to X_base rows
)
overall_importance_df = build_feature_summary(shuffle_results, in_shuff_cfg)
record_iqr_df = record_stat_matrix(shuffle_results, "iqr")   # rows = records, columns = variables
```

Notes on the engine:
- It seeds every shuffle deterministically from (seed, feature index, shuffle index), so results do not depend on batching.
- `metric_fns` values are returned per shuffle in each result's `metric_values`; use them for business degradation.
- Top-decile importance = mean of `record_iqr_df` rows where baseline_scores is in the top 30%.

Use notebook-native execution only. Do NOT use shell scripts, subprocess calls, or the package's command-line interface.

If the repository is not available, implement the algorithm above directly in the notebook. The implementation must produce the same quantities: per-record IQR across M shuffles, mean_record_iqr ranking, and top-3-decile importance from the same shuffles. Seed each permutation with np.random.default_rng([RANDOM_STATE, feature_index, shuffle_index]).

---

## BASELINE PRODUCTION SCORE

Identify the final recommended production model from Step 10 and regenerate its scores directly from the preserved production model objects.

The generated code must:
1. detect whether the production strategy is tuned or untuned, and whether it is an expected value model
2. build `production_scorer` from the preserved model objects
3. regenerate baseline scores on the analysis set with `production_scorer`
4. verify the regenerated scores match the stored production prediction column (report the max absolute difference; warn if it exceeds 1e-6)

Do NOT rank records or define the top 3 deciles from stale prediction columns.

For an expected value model, interpret the combined expected value score. Do NOT interpret only one component model unless explicitly instructed.

Persist the regenerated scores as `production_score_test` (indexed by RECORD_ID) for reuse in Step 13.

---

## REUSE REQUIREMENTS

Reuse the existing train/test split, preprocessing pipelines, transformed-variable logic, feature lists, and cumulative net revenue functions.

Do NOT retrain models, recreate preprocessing pipelines, or regenerate train/test splits.

---

## CONFIGURATION AND RUNTIME

Define these constants at the top of the step:

```python
N_SHUFFLES = 20                 # M; the reference implementation's default is 50
INPUT_SHUFFLING_SAMPLE_N = None # None = full testing partition
TOP_DECILE_FRACTION = 0.30
```

Runtime is roughly (number of variables) × N_SHUFFLES × (time to score A once).

Before the full run:
1. time one scoring pass of A
2. report the estimated total runtime
3. if the estimate exceeds about 10 minutes, set INPUT_SHUFFLING_SAMPLE_N (for example 10,000), draw a seeded random sample of the testing partition, and report the sample size used

Reduce the sample size before reducing N_SHUFFLES below 10.

Log the shuffle seeds and any sampling seed in random_seed_log. Report shuffling runtime per variable and in total.

---

## OUTPUT REQUIREMENTS

1. Overall importance table: all variables ranked by mean_record_iqr, with median_record_iqr, mean_record_std, mean_abs_delta, fraction_abs_delta_gt_eps, number of records, and N_SHUFFLES.

2. Top-3-decile importance table: the same columns computed on top-3-decile records.

3. Rank comparison table: overall rank, top-decile rank, rank shift (overall rank − top-decile rank), and the change in mean_record_iqr. Name the variables that become more important and less important in the top deciles.

4. Business degradation tables for (a) the top 10 overall variables and (b) the top 10 top-decile variables, each with:
   - feature name
   - overall rank and top-decile rank
   - mean_record_iqr
   - cumulative net revenue degradation
   - ROI degradation
   - optimal depth degradation
   - donation capture degradation
   - records used

5. Plots:
   - horizontal bar chart of the top 10 overall variables by mean_record_iqr
   - the same for the top 3 deciles
   - side-by-side comparison of overall vs top-decile ranks for the union of both top-10 lists

---

## OBJECT CREATION REQUIREMENTS

Persist:
- production_scorer
- production_score_test
- shuffle_feature_list
- overall_importance_df
- top_decile_importance_df
- importance_rank_shift_df
- record_iqr_df
- business_degradation_df
- overall_business_degradation_top10_df
- top_decile_business_degradation_top10_df
- input_shuffling_summary_tables
- input_shuffling_discussion_df

Do not overwrite final production model objects.

---

## DISCUSSION REQUIREMENTS

Generate concise discussion summaries covering:
1. why variable importance changes in the top mailing deciles
2. why business degradation may rank variables differently from prediction instability
3. operational implications for campaign targeting
4. how input shuffling differs from permutation importance, SHAP, and model-native importances
5. why per-record instability (rather than a single metric drop) is a useful view of a model

---

Generate Python code that performs this step, including standardized timing instrumentation and runtime summary reporting.

---

# STEP 13 — RESPONSE PROFILES FOR TOP VARIABLES

Show how the production score changes as each of the most important variables changes, for the full testing population and for the top 3 deciles.

---

## IMPORTANT REUSE REQUIREMENTS

Reuse from Step 12:
- production_score_test (do NOT rescore and do NOT use older prediction columns)
- overall_importance_df
- top_decile_importance_df

Use the same top-3-decile definition as Step 12: the highest 30% of production_score_test.

Do NOT retrain or rescore models.

---

## VARIABLES AND POPULATIONS

Plot the union of:
- the top 3 overall importance variables
- the top 3 top-decile importance variables

Do not duplicate plots when the lists overlap.

For each variable, compare two populations:
- overall testing population
- top 3 decile population

---

## WHAT TO PLOT

- x-axis = the raw (untransformed) input variable, binned
- y-axis = mean production score in each bin

Label these plots "response profiles". They show observed average scores by bin. Call them partial dependence plots only if the code actually rescores a controlled grid of values while holding other variables fixed.

---

## SHARED BINNING REQUIREMENTS

Follow the GLOBAL BINNING AND PANDAS DTYPE STANDARDS, plus:

1. Create bin edges once, from the overall testing population.
2. Apply those same edges to the top-3-decile population. Do NOT create separate qcut bins for each population.
3. Numeric variables: quantile bins (default 10, duplicates dropped), ordered by numeric midpoint.
4. Categorical variables: one bin per level, ordered by overall record count.
5. Report record counts per bin for both populations.

---

## SPARSE BIN REQUIREMENTS

Applying overall bins to the top 3 deciles can leave some bins nearly empty.

1. Define configurable thresholds:
   - MIN_BIN_RECORDS_OVERALL = 50
   - MIN_BIN_RECORDS_TOP_DECILE = 30
2. Flag every bin below its threshold as sparse.
3. Do not draw connecting lines through sparse bins. Show sparse top-decile bins as isolated, visibly marked points, or suppress them.
4. List sparse bins in a diagnostic table and mention them in the discussion.
5. Continue safely when bins are sparse or empty.

---

## VISUALIZATION REQUIREMENTS

1. Overall population: connected line where bins meet the threshold.
2. Top-decile population: points, connected only between adjacent non-sparse bins.
3. Overlay both populations on one chart only because they share identical bins.
4. Show record counts per bin (secondary axis bars or annotations).
5. Legend distinguishes overall and top-decile.
6. Keep numeric x-axis ordering; fall back to ordered categorical plotting if midpoints are unavailable.
7. Do not smooth across bins. Noisy profiles are informative and should be shown as they are.

---

## RUNTIME REQUIREMENTS

Response profiles use stored scores only, so this step should be fast. If plotting is slow, sample the testing population with a logged seed and report the sample size.

---

## OBJECT CREATION REQUIREMENTS

Persist:
- response_profile_variables
- response_profile_bin_edges
- response_profile_plot_dataframes
- response_profile_sparse_bins_df
- response_profile_plot_objects
- response_profile_discussion_df

---

## DISCUSSION REQUIREMENTS

Generate concise discussion summaries covering:
1. how the score responds to each top variable (direction, shape, thresholds)
2. why relationships may differ between the overall population and the top operational mailing population
3. where sparse bins limit what can be concluded
4. whether the shapes are consistent with the Step 12 importance rankings

---

Generate Python code that performs this step, including standardized timing instrumentation and runtime summary reporting.


---

# STEP 14 — SUPERVISED ASSOCIATION RULE DISCOVERY

Build supervised association rules using the KDD Cup 1998 dataset to identify combinations of INPUT characteristics associated with desirable TARGET outcomes.

The purpose is interpretable segment discovery and operational targeting insight — finding rules of the form:

INPUT VARIABLES  --->  TARGET OUTCOME

Examples:
- high recency + high last gift  ---> TARGET_B = 1
- high frequency + wealth rating ---> HIGH_DONATION = 1
- long tenure + high giving      ---> TOP_DECILE_DONOR = 1

Profitability is judged per segment in RULE PROFILING below (net revenue of everyone a rule matches), not as a record-level outcome: every gift in this file is at least $1, more than the $0.68 mail cost, so "profitable record" is identical to TARGET_B = 1.

Use Python code only. Do NOT use shell scripts, subprocess calls, or command-line execution.

Reuse the existing train/test split, transformed variables, prediction tables, model_metadata_df, and cumulative net revenue framework. Do NOT rebuild predictive models.

---

## DATA SOURCE

Use the testing partition only:

```python
ar_source_df = modeling_prepared_df[
    modeling_prepared_df["split_flag"] == "TEST"
].copy()
```

The master dataframe throughout this notebook is `modeling_prepared_df`. Do NOT search for alternative dataframe names. Do NOT use `df`, `df_model`, `df_prepared`, or any other name.

---

## RULE STRUCTURE REQUIREMENTS

1) Antecedents must contain ONLY input predictor variables — donor characteristics, behavioral, demographic, engineered, or binned predictors.

2) Antecedents must NEVER contain:
- TARGET_B, TARGET_D
- donation decile indicators
- expected value scores, model predictions, probability scores
- net revenue, cumulative revenue, lift, ROI, or decile variables
- any downstream target-derived variable

3) Consequents must contain ONLY target outcome variables:
- TARGET_B = 1
- HIGH_DONATION = 1
- TOP_DECILE_DONOR = 1

4) Each rule must have exactly ONE consequent.

5) Limit antecedents to a maximum of 2 per rule. This is an interaction detection exercise: we are looking for 2-way input combinations associated with target outcomes. Rules with a single antecedent are less interesting but acceptable. Rules with 3 or more antecedents should be excluded.

---

## VARIABLE PRE-SELECTION

The KDD Cup dataset has hundreds of input variables. Mining association rules on all inputs causes combinatorial explosion. Pre-select a manageable candidate set before building the transaction matrix.

### Step A — Exclude forbidden variables

Exclude from antecedent candidates:
- TARGET_B, TARGET_D
- RECORD_ID, split_flag
- Any column whose name contains: PRED, PROB, SCORE, EXPECTED, EV_, NET_REVENUE, CUM_, DECILE, LIFT, ROI, OUTCOME_

### Step B — Rank remaining inputs by univariate association with TARGET_B

Use chi-square tests for categorical variables and ANOVA F-statistics for numeric variables (after binning into 4 quantile bins). Compute a test statistic and p-value for each candidate input against TARGET_B.

Sort candidates by chi-square or F-statistic descending. Report this ranking.

### Step C — Supplement with top model predictors

Append any variables that appear in `target_b_top_features` (the top-10 predictors per Step 4 classification model). If the dict is missing, print a warning and continue with the univariate ranking only. These variables are already validated as strong predictors and should be included even if they rank lower on univariate tests.

Deduplicate the combined list.

### Step D — Select final candidate set

Retain the top N candidates after ranking, where N is a configurable parameter:

```python
MAX_ANTECEDENT_INPUTS = 50  # configurable
```

Report:
- variable name
- test statistic
- p-value
- source (chi-square, F-statistic, or model importance)

---

## DATA PREPARATION FOR RULE MINING

1) For numeric variables in the candidate set: create 4 quantile bins using pd.qcut with duplicates="drop". Follow the GLOBAL BINNING AND PANDAS DTYPE STANDARDS. Label bins with their interval string. Handle missing values as a separate "missing" bin.

2) For categorical variables: use as-is if cardinality <= 20. Exclude higher-cardinality categoricals.

2a) For binary 0/1 inputs (most columns in this file are dummies): do not bin them, and create only the "=1" item. Drop the "=0" item: it is true for almost everyone and produces filler rules such as "RFA_2F=4 & L1F_RFA_2=0".

3) One-hot encode all binned/categorical inputs using pd.get_dummies. Prefix antecedent columns with "ANT::". Prefix target outcome columns with "CONS::".

4) Create CONSEQUENT target outcome flags separately:
- OUTCOME_TARGET_B_1: TARGET_B == 1
- OUTCOME_HIGH_DONATION_TOP30: TARGET_D >= 70th percentile of positive donations
- OUTCOME_TOP_DECILE_DONOR: TARGET_D >= 90th percentile of positive donations

Report the base rate of each outcome. They are rare (about 5%, 1.5% and 0.5% of TEST rows), which is why the mining below works on antecedents first and counts outcomes directly.

5) Combine antecedent and consequent matrices into a single boolean transaction matrix.

6) Filter out antecedent items with support < 0.005 or > 0.95.

---

## ASSOCIATION RULE MINING

Count the rules directly with matrix products. No itemset-mining library is needed. With 1- and 2-item antecedents and one outcome per rule, every rule's counts come from one matrix product per outcome:

```python
X = transaction_matrix_supervised[ant_cols].to_numpy(dtype=np.float32)    # rows x antecedent items (0/1)
n_A   = X.sum(axis=0)                  # records matching each single item
n_AB  = X.T @ X                        # records matching each pair of items
for outcome in outcome_cols:
    c = transaction_matrix_supervised[outcome].to_numpy(dtype=np.float32)
    n_AC  = X.T @ c                    # single item AND outcome
    n_ABC = X.T @ (X * c[:, None])     # pair of items AND outcome
```

From these counts:
- support(A) = n_A / n_rows, confidence = n_AC / n_A, lift = confidence / outcome rate (same for pairs, using n_AB and n_ABC)
- records_with_outcome = n_AC (or n_ABC)

Keep a rule only if:
- the antecedent matches at least 0.5% of rows (`MIN_ANTECEDENT_SUPPORT = 0.005`), and
- records_with_outcome >= 20 (`MIN_OUTCOME_RECORDS = 20`).

Ignore the diagonal of n_AB (an item paired with itself) and items from the same source variable (two bins of one variable cannot both be true).

Do NOT use mlxtend fpgrowth plus association_rules() here. In a test run that took 15–19 minutes per support level plus over 30 minutes for association_rules(), and the rare outcomes produced no rules at all. The matrix version computes every 1- and 2-item rule exactly and takes seconds.

## RULE FILTERING AND VALIDATION

After mining, filter rules to enforce supervised structure:

1) Antecedents must contain ONLY ANT:: items.
2) Consequents must contain ONLY CONS:: items.
3) Exactly 1 consequent per rule.
4) Antecedent count must be exactly 2 (for interaction detection). Keep 1-antecedent rules in a separate table for reference but focus reporting on 2-antecedent rules.
5) Lift >= 1.10 minimum.
6) A 2-antecedent rule must beat both of its 1-antecedent parents: its lift must be at least 5% higher than the better parent's lift for the same outcome. Otherwise the second item adds nothing, and the rule is not an interaction.

Create separate filtered rule sets for each consequent type:
- association_rules_response (TARGET_B = 1)
- association_rules_high_donation
- association_rules_top_decile_donor

---

## RULE PROFILING

For each rule set, re-apply antecedent conditions back to the original rows in ar_source_df and compute:
- records_matched
- response_rate
- avg_donation
- total_donation
- mailing_cost (records_matched * 0.68)
- net_revenue
- net_revenue_per_record

Sort profiled rules by net_revenue descending. A rule's segment is profitable when its net_revenue > 0; this is how profitable segments are identified.

---

## REPORTING

Generate:
- variable pre-selection ranking table (top 50 by test statistic)
- top 20 two-antecedent rules for each consequent, sorted by chi-square statistic then confidence
- profiled rule tables with business metrics
- support vs. confidence scatter plot (bubble size = lift) for each rule set
- bar chart of net revenue by rule antecedent (top 15 rules)
- operational segment ranking table combining all rule sets, sorted by net_revenue descending

---

## COMPARISON WITH MODEL RESULTS

Compare the top association rule antecedents with:
- input shuffling importance from Step 12 (overall_importance_df and top_decile_importance_df)
- top predictors from Step 4 classification models

Discuss:
1) Whether important predictive variables also appear in strong rules
2) Whether rules identify nonlinear interactions missed by main-effects models
3) Whether rules provide interpretability advantages over black-box models
4) Which donor segments appear most profitable or responsive
5) Risks of overinterpreting rules (multiple testing, overfitting to test data). Note explicitly that the testing partition has now been reused across cumulative net revenue evaluation (Step 7), hyperparameter tuning (Step 10), input shuffling (Step 12), and association rule mining (this step) — each additional round of pattern-fitting against the same holdout increases the risk that findings reflect noise in this specific test partition rather than generalizable structure. Discuss the implications for production validation.

---

## OBJECT PERSISTENCE

Persist:
- ar_source_df
- variable_preselection_ranking_df
- antecedent_binned_df
- transaction_matrix_supervised
- frequent_itemsets_supervised
- supervised_rules_all
- association_rules_response
- association_rules_high_donation
- association_rules_top_decile_donor
- profile_response_rules
- profile_high_donation_rules
- profile_top_decile_donor_rules
- operational_segments_ranked

Do not overwrite final production model objects or prior notebook state.

---

## RUNTIME MANAGEMENT

Variable pre-selection (Step B above) is mandatory — do not skip it. Use MAX_ANTECEDENT_INPUTS = 50 as the default. Report the runtime of the matrix counting separately; it should take seconds.

Generate Python code that performs this step, including standardized timing instrumentation and runtime summary reporting.

---

# DISCUSSION QUESTIONS

## Discussion Question 1

Why might a model with the best ROC AUC fail to produce the highest profit?

---

## Discussion Question 2

Why is lift often more operationally useful than overall classification accuracy in direct marketing?

---

## Discussion Question 3

Why might predicting donation amount alone produce poor campaign performance?

---

## Discussion Question 4

Why can transformed variables improve predictive stability for highly skewed monetary data?

---

## Discussion Question 5

Why might the best statistical model differ from the best business model?

---

## Discussion Question 6

Steps 7–9 report the TEST peak: the best mailing depth found by looking at the test data itself. Why is that optimistic? How would you set the mailing threshold or depth in production, using only training data?

---

# KEY CONCEPTS

| Concept | Meaning |
|---|---|
| ROC AUC | Global ranking quality |
| Lift | Targeting quality |
| Cumulative Net Revenue | Business value generated |
| TARGET_B Models | Predict response probability |
| TARGET_D Models | Predict donation amount |
| Expected Value Models | Predict economic value |
| Input Shuffling | Variable influence measured as per-record score instability when an input is permuted |

---

# OVERALL WORKSHOP OBJECTIVE

The goal of this workshop is to demonstrate that:

- the best statistical model is not necessarily the best business model
- business objectives should determine model evaluation strategy
- predictive modeling should ultimately optimize operational outcomes
- ranking quality is often more important than raw classification accuracy
- expected economic value can outperform pure response prediction
