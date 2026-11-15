# Data

| file | rows × columns | what it is |
|---|---|---|
| `cup98lrn_prepped_allrecords_w_TargetD.tsv` | 95,412 × 34 (tab-separated) | KDD Cup 1998 learning file, a prepared subset by Dean Abbott: no missing values, RFA codes as dummy variables, monetary fields log10-transformed (`_log10`). Targets: `TARGET_B` (responded, about 5%) and `TARGET_D` (gift amount). Used by `prompts/kddcup1998_workshop_modeling_v8.md`. |
| `cup98lrn_automl_prepped_with_split.tsv` | 95,412 × 42 (tab-separated) | The same KDD Cup 1998 records, prepared differently, for the AutoML runs in `automl_harness/`: 38 inputs, including the raw categorical fields (STATE, DOMAIN, RFA_2, RFA_3, GENDER and others) for the AutoML tools to encode themselves, and no variable selection. `TARGET_D` is 0 for non-responders. `is_test` (0/1) is the 50/50 train/test split used for every AutoML run in the deck; `RECORD_ID` matches the row order of the file above and is never an input. |
| `Cup98Dic edited 3.pdf` | 8 pages | Data dictionary for the KDD Cup 1998 fields (the original competition dictionary, edited by Dean Abbott in 2014). The prepared file uses a subset of these fields; `_log10` columns are log10 versions of the monetary fields. |
| `titanic3.csv` | 1,309 × 14 | Titanic passenger list with survival (`survived`, 38% survived). Used by `prompts/titanic_workshop_modeling_v1.md`. Note that `boat` and `body` were recorded after the outcome (leakage); the prompt file removes them. |
| `workshopdataset_1_final.csv` | 1,309 × 14 | A disguised dataset for the data-leakage exercise. The variable names and descriptions are deliberately generic. Your task: find out which inputs could only be known after the outcome. |
| `workshopdataset_1_final_dictionary.csv` | 14 variables | Data dictionary for `workshopdataset_1_final.csv`: variable name, type, description. |

These datasets are not covered by this repository's license. Their original terms apply.

**KDD Cup 1998.** Parsa, I. (1998). *KDD Cup 1998 Data* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5401H. The UCI repository lists it under CC BY 4.0. Both prepared KDD files here are derivatives of the original learning file (`cup98lrn`).

**Titanic (titanic3).** Compiled by Frank E. Harrell Jr. and Thomas Cason from Encyclopedia Titanica. Courtesy of the Vanderbilt University Department of Biostatistics: https://hbiostat.org/data

**workshopdataset_1_final.** Derived from a public dataset by Dean Abbott for teaching; the source is revealed during the workshop.
