# Data Leakage Exercise: Prompts
## Can a GenAI assistant tell when a model is too good to be true?

Companion to the workshop "AutoML and GenAI" (Abbott Analytics, ML Week Europe, November 2026).

**Data:** `data/workshopdataset_1_final.csv` (1,309 rows × 14 columns) and its dictionary `data/workshopdataset_1_final_dictionary.csv`. The variable names and descriptions are deliberately generic.

**How to use this file**

1. Start a new chat with memory off: Claude's Incognito chat, or ChatGPT's Temporary Chat. The assistant should know only what you give it.
2. Paste the MAIN PROMPT below, then upload the data file and the dictionary.
3. Paste one step at a time (Steps 1–7). Read each result before you move on.
4. Then paste the FOLLOW-UP PROMPTS, one at a time.

Before you start the follow-ups, look at the test results yourself. What would you expect for real-world data? Which inputs does the model rely on, and could each one be known at the time a prediction would be made?

The prompt is reproduced as it was used in the workshop, with only spelling fixed, so you can compare your results with the slides.

---

# MAIN PROMPT

For all the analyses to be conducted in this chat, do not use any information in any other chat, and do not use any information from the internet. Use only principles about predictive modeling you already know.

A dataset called workshopdataset_1_final.csv will be uploaded. I'll also upload a dictionary file to explain the variables.

The target variable is called "target", a binary target. Do not use any other data except what I upload for this analysis, so wait to proceed until I upload the data.

Make target a categorical variable. Build a machine learning model to predict target most accurately, where accuracy means highest lift for the top 30% of probability scores. Use at least logistic regression, GBMs, Naive Bayes, and if possible Random Forests.

Here is the entire analysis to do. Please only do them one step at a time. I'll prompt you to go to the next step.

1) Load the data.

2) Replace missing values with the median. Report how many missing values each variable had.

3) Use best practices for train/hold-out split and report the split.

4) Build the models. Use best judgement to determine the parameters for each model.

5) For each algorithm, create a table showing baseline rate, lift at the 3rd decile, ROC AUC, F1, and classification accuracy. Rank order the list by most accurate algorithm at the 3rd decile. Create separate tables for training and testing results.

6) Also generate a report on what parameters were used for each algorithm as follows:
- all models: what were the candidate inputs
- logistic: algorithm used for optimizing weights
- naive bayes: all non-defaults
- RF (if done): terminal node min size; # trees; splitting metric; variable selection method (sqrt(M)?)
- GBM: # trees, learning rate, # terminal nodes per tree or # splits per tree

7) For each model, create a list of the top 3 variables as determined by permutation importance. Create a table showing this list.

---

# STEP PROMPTS

Paste these one at a time, after the main prompt and the uploads.

- Go ahead with step 1.
- Go to step 2.
- Go to step 3.
- Go to step 4.
- Go to step 5.
- Go to step 6.
- Go to step 7.

---

# FOLLOW-UP PROMPTS

**Follow-up 1: does the assistant notice?**

Is there anything surprising or striking about these models?

**Follow-up 2: tuning (as in the workshop)**

Can you improve the random forests model by reducing overfit? Try increasing the minimum number of records in the terminal node.

Did tuning fix the problem? Why or why not?

**Follow-up 3: the leakage audit**

For each input variable, use the data dictionary to decide whether its value would be known at the time a prediction is made, or only after the outcome. Make a table: variable, description, known before the outcome (yes / no / unclear), and your reason. For any variable marked "no" or "unclear", show the target rate for each of its values (or for missing vs. present), and say what that pattern suggests.

**Follow-up 4: rebuild without leakage**

Remove every input you marked "no" in the audit, along with pure identifiers, and rebuild all of the models with the same split and settings. Report the same training and testing tables as in step 5, and compare them with the original results. Which results would you trust, and why?

---

# DISCUSSION QUESTIONS

1. What first made you suspicious: the size of the AUCs, the agreement across very different algorithms, or the variables that came out on top?

2. Which variables are identifiers? Which ones are recorded after the outcome? Can a variable be both?

3. Would cross-validation, a different train/test split, or more tuning have revealed the problem? Why or why not?

4. The assistant was told not to use outside information. How does that change what it can and cannot conclude about the meaning of each variable?

5. Where else does leakage hide in practice: dates, status codes, fields that are updated after the event, or IDs assigned in a sequence?
