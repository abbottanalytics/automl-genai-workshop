# AutoML and GenAI Workshop

Slides, AI prompts, data files, and sample output from Dean Abbott's workshop on AutoML and generative AI for predictive modeling (Abbott Analytics; first given at ML Week Europe, November 2026).

The theme: **the best statistical model is not necessarily the best business model.** The prompt files walk a GenAI assistant (ChatGPT, Claude, or similar) through a full modeling project one step at a time, and evaluate the models on the business question as well as on statistical metrics.

## Download

Click the green **Code** button above, then **Download ZIP**. Unzip it anywhere. You don't need a GitHub account.

## What's here

| folder | contents |
|---|---|
| `slides/` | the workshop deck as PDF: a slides version and a notes version |
| `prompts/` | the step-by-step prompt files: `kddcup1998_workshop_modeling_v8.md` (KDD Cup 1998, net revenue) and `titanic_workshop_modeling_v1.md` (Titanic, a capacity metric when there is no dollar value) |
| `data/` | the two datasets the prompts use, with sources and terms in `data/README.md` |
| `genai_backup_v8/` | finished notebook for the KDD Cup prompt file: the code for every step, already run, plus every report table, log and plot in `reports/` |
| `genai_backup_titanic_v1/` | the same for the Titanic prompt file |

## Two ways to use it

**1. With a GenAI assistant (as in the workshop).** Open a prompt file in any text editor. Paste the global sections and the MASTER INITIAL PROMPT into a new chat, upload the data file from `data/`, then paste one step at a time. Run the code the assistant writes in a Jupyter notebook, look at the results, and ask questions before moving on. The assistant will make its own choices, so its numbers won't match the backup notebooks exactly.

**2. Without an assistant.** Open the backup notebook (`.ipynb`) in Jupyter. It has already been run, so you can read every step's output without running anything. To run it yourself, keep the folder layout as it is (the notebooks read `../data/`) and Run All. The KDD notebook takes about 1.5 minutes on a laptop, and the Titanic notebook about 40 seconds.

**Reusing the prompts on your own data.** The Titanic prompt file starts with a section called "HOW TO REUSE THIS FILE FOR YOUR OWN DATA" and a PROBLEM PARAMETERS block: edit that block (data file, target, leakage and ID columns, metric), and pick the KDD or Titanic file depending on whether your outcome has a dollar value.

## Python

Python 3.10 or later with pandas, numpy, scikit-learn, xgboost, scipy, matplotlib and pyarrow (an Anaconda installation has them all), plus Jupyter.

The input shuffling steps (KDD Step 12, Titanic Step 10) can use Dean Abbott's `input_shuffling` package (https://github.com/abbottanalytics/input_shuffling) if you clone it next to this folder. Without it, the notebooks use a short built-in version of the same algorithm.

## License

The code in this repository (notebooks and scripts) is licensed under the MIT License; see `LICENSE`.

The optional `input_shuffling` package is a separate repository, licensed separately under **AGPL-3.0**. It is not included here.

The datasets are not covered by this repository's license; see `data/README.md` for their sources and terms.

## Contact

Dean Abbott, Abbott Analytics.
