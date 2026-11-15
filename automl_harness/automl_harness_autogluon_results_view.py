#######################################################
# automl_harness_autogluon_results_view.py
#
# A cleaner view of AutoGluon results, in place of predictor.fit_summary(show_plot=True):
#   - a compact table (model, validation score, fit and predict seconds, stack level)
#     that does not wrap in a notebook
#   - a full-width plot of validation score vs. fit time, every model labeled
#
# Use it in the notebook right after predictor.fit(...): paste it into the next cell,
# or run it there with  %run automl_harness_autogluon_results_view.py -i
# It expects a fitted TabularPredictor named `predictor`.
#
# The y-axis is labeled ROC AUC (the workshop runs use eval_metric='roc_auc');
# change the label if you use another metric.
# Abbott Analytics, Nov 2026.
#######################################################
import matplotlib.pyplot as plt

lb = predictor.leaderboard()
# good_quality / best_quality refit each model on all of TRAIN at the end ("..._FULL" models).
# The refit copies have no validation score, so leave them out of the table and the plot.
lb = lb[~lb["model"].str.endswith("_FULL")]

# Compact results table: one row per model, five columns
table = (lb[["model", "score_val", "fit_time", "pred_time_val", "stack_level"]]
         .rename(columns={"score_val": "AUC (validation)", "fit_time": "fit sec",
                          "pred_time_val": "predict sec", "stack_level": "level"})
         .round(3))
try:
    display(table)                     # Jupyter
except NameError:
    print(table.to_string(index=False))  # plain Python

# Full-width plot: validation score vs. fit time, every model labeled
fig, ax = plt.subplots(figsize=(14, 7))
ax.scatter(lb["fit_time"], lb["score_val"], s=120)
for _, r in lb.iterrows():
    ax.annotate(r["model"], (r["fit_time"], r["score_val"]),
                xytext=(6, 4), textcoords="offset points", fontsize=11)
ax.set_xlabel("fit time (seconds)", fontsize=12)
ax.set_ylabel("ROC AUC on validation data", fontsize=12)
ax.set_title("Models produced during fit()", fontsize=14)
ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
