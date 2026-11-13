# Backup code for the GenAI workshop prompts (v8)

The code an assistant would generate for `../prompts/kddcup1998_workshop_modeling_v8.md`, one cell per step (Steps 1–14 plus a final timing report). Use it if a live step fails or runs long.

| file | what it is |
|---|---|
| `kdd98_genai_v8_backup.ipynb` | the notebook, already run, with all outputs. Open it in Jupyter and run one cell per step, or Run All. |
| `kdd98_genai_v8_backup.py` | the same code as a script with `# %%` cells (VS Code / Jupyter can run it cell by cell) |
| `reports/` | everything each step printed (`stepNN_log.txt`), every table (`stepNN_<name>.csv`), and `test_predictions_all_models.parquet` |
| `reports/plots/` | every plot (`stepNN_<name>.png`) |

**Data path.** The notebook reads `../data/cup98lrn_prepped_allrecords_w_TargetD.tsv` (the repository's `data/` folder). If you move the notebook, it also looks in `data/`, in its own folder, and in your home folder. It prints the path it used.

**Python.** Anaconda base (scikit-learn 1.4, xgboost 2.0, scipy, pandas, matplotlib, pyarrow). Step 12 uses the input shuffling package (github.com/abbottanalytics/input_shuffling, licensed AGPL-3.0) if it is cloned next to this repository or `INPUT_SHUFFLING_SRC` points to its `src/` folder; otherwise it uses the short notebook-native version included here.

**Runtime** (M1, 13 Nov 2026): about 1.5 minutes in total. Step 12 (input shuffling, 20 shuffles × 32 inputs × 28,624 test rows) is 48 s; Step 4 is 15 s; Step 14 counts all association rules in 0.1 s.

**Rerunning.** Rerunning a cell is safe: models, predictions and metadata that already exist are not refit or overwritten. To start fresh, restart the kernel and Run All; the `reports/` files are overwritten.

**Key results from this run** (70/30 split, seed 42, 28,624 test rows; mail everyone = $2,933; all peaks are TEST peaks, optimistic):
- Best response model by lift: XGBoost (AUC 0.626). Best gift model: random forest / linear regression (very close).
- Best business model: EV = XGBoost P(respond) × linear-regression gift, TEST peak $4,751 at 36% mailed.
- Tuning (8 combinations, scored on TEST) gained $16, within noise, so the untuned model stays the production model.
- Linear models drop the combined `L*_RFA_2` dummies: `E_RFA_2A + F_RFA_2A + G_RFA_2A + L3D_RFA_2 + L4D_RFA_2 = 1` on every row, which is collinear with the intercept. Largest linear-regression coefficient is now 4.7 (it was ±5.8e11).
- Step 14: 4,236 rules counted, 499 kept, including 100 two-item rules for the rare top-decile-donor outcome (the v7 approach found none).

An assistant running the v8 prompts live will make different choices (split, model settings, bins), so its numbers will not match these exactly.
