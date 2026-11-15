# AutoML harness comparison

## Headline

**Titanic**

```
                        run  minutes accuracy (baseline)   AUC
 titanic_autogluon_best_auc     10.1       0.837 (0.618) 0.894
  titanic_autogluon_default      0.7       0.812 (0.618) 0.873
titanic_autosklearn_default     10.1       0.837 (0.618) 0.890
```


**KDD Cup 1998** (47,706 test rows; 38-input prep and 50/50 split in data/cup98lrn_automl_prepped_with_split.tsv; models fit on all of TRAIN; cutoffs chosen where CNR peaks on out-of-fold TRAIN predictions (bagged OOF for best_quality, otherwise a 5-fold cross-fit of the whole run); mean response rate 0.0508)

```
                      run  minutes (incl. gift model) accuracy (baseline)  % mailed by tool default   AUC  mean P CNR, P cutoff (train) CNR, EV > $0.68 CNR, EV cutoff (train)  % mailed, EV cutoff CNR, EV depth (train)  % mailed, EV depth CNR, EV cutoff, fold-average models  OOF mean P EV hindsight peak (upper bound) CNR mail all
 kdd98_autogluon_best_auc                        80.1     0.9492 (0.9492)                       0.0 0.634  0.0506                $5,341          $8,337                 $8,183                 61.4                $8,238                60.4                                          0.0506                          $8,456       $5,405
  kdd98_autogluon_default                         1.2     0.9492 (0.9492)                       0.0 0.625  0.0504                $5,408          $7,712                 $5,877                 94.9                $6,736                84.3                              $5,420      0.1001                          $8,087       $5,405
    kdd98_autosklearn_auc                        20.0     0.9492 (0.9492)                       0.0 0.626  0.0607                    $0          $6,485                   $591                  0.2                $6,468                79.7                              $6,150      0.4504                          $6,556       $5,405
kdd98_autosklearn_default                        19.9     0.9492 (0.9492)                       0.0 0.500  0.5000                $5,405          $5,405                 $5,405                100.0                $5,951                93.2                              $5,405      0.4010                          $6,474       $5,405
```


## Titanic (survived), 30% stratified hold-out

```
                   run_name         tool tool_version                                                                          settings  n_train  fit_seconds  predict_seconds  accuracy_default  accuracy_baseline  roc_auc  log_loss  best_val_score
 titanic_autogluon_best_auc    AutoGluon        1.6.3                         presets=best_quality, eval_metric=roc_auc, time_limit=600      916     604.4186           0.6046            0.8372             0.6183   0.8937    0.3935          0.8690
  titanic_autogluon_default    AutoGluon        1.6.3           presets=default (medium_quality), eval_metric=accuracy, time_limit=None      916      44.7087           0.0087            0.8117             0.6183   0.8727    0.4249          0.8315
titanic_autosklearn_default auto-sklearn       0.15.0 defaults (metric=accuracy), time_limit=600s, n_jobs=8, memory_limit=None, seed=42      916     603.8068           0.3589            0.8372             0.6183   0.8897    0.4748             NaN
```


## KDD Cup 1998 (TARGET_B; CNR uses TARGET_D, $0.68 mail cost)

```
                 run_name         tool                                                                                                 settings  n_train  n_test  fit_seconds  gift_fit_seconds  predict_seconds  accuracy_default  accuracy_baseline  pct_predicted_positive  roc_auc  log_loss  lift_top10  cnr_peak  cnr_peak_depth_pct  cnr_top10  cnr_mail_all  cnr_default_decision  mailed_pct_default_decision  gift_rmse_resp  gift_spearman_resp  ev_rule_cnr  ev_rule_mailed_pct  ev_cnr_peak  ev_cnr_peak_depth_pct  p_cut_threshold  p_cut_cnr  p_cut_mailed_pct  p_cut_cnr_at_depth  ev_cut_threshold  ev_cut_train_depth_pct  ev_cut_cnr  ev_cut_mailed_pct  ev_cut_cnr_at_depth                                    cutoff_method  cutoff_cv_seconds  oof_mean_pred  cvavg_roc_auc  cvavg_mean_pred  cvavg_ev_cut_cnr  cvavg_ev_cut_mailed_pct  cvavg_ev_cut_cnr_at_depth
 kdd98_autogluon_best_auc    AutoGluon                                               presets=best_quality, eval_metric=roc_auc, time_limit=3600    47706   47706    3609.1131         1198.1070          39.1535            0.9492             0.9492                     0.0   0.6336    0.1950      2.1757   5628.96              0.9790    1276.72       5404.92                   0.0                          0.0          9.7538              0.7929      8337.00              0.5780      8455.80                 0.5369           0.0203    5341.28            0.9973             5426.96            0.6645                  0.6042     8183.16             0.6139              8238.32 bagged out-of-fold (AutoGluon predict_proba_oof)             0.0000         0.0506            NaN              NaN               NaN                      NaN                        NaN
  kdd98_autogluon_default    AutoGluon                                  presets=default (medium_quality), eval_metric=accuracy, time_limit=None    47706   47706      54.1223           18.9815           0.1663            0.9492             0.9492                     0.0   0.6248    0.1955      2.0766   5468.32              0.9973    1405.72       5404.92                   0.0                          0.0          9.2468              0.7893      7712.08              0.5664      8087.08                 0.3850           0.0167    5407.64            0.9999             4933.68            0.4303                  0.8429     5877.44             0.9494              6735.84          5-fold cross-fit of the full AutoML run           328.6077         0.1001         0.6045           0.0999           5419.92                   0.9990                    6556.84
    kdd98_autosklearn_auc auto-sklearn             metric=roc_auc, time_limit=900s, n_jobs=2, memory_limit=None, max_models_on_disc=10, seed=42    47706   47706     905.3777          295.2382          19.5961            0.9492             0.9492                     0.0   0.6262    0.1999      2.1138   5747.52              0.9592    1218.72       5404.92                   0.0                          0.0         10.1315              0.7826      6484.80              0.7995      6556.00                 0.5141           0.3705       0.00            0.0000             5431.60            4.9335                  0.7967      591.00             0.0016              6467.60          5-fold cross-fit of the full AutoML run          6033.7792         0.4504         0.6261           0.4504           6150.12                   0.8074                    6056.60
kdd98_autosklearn_default auto-sklearn defaults (metric=accuracy), time_limit=900s, n_jobs=2, memory_limit=None, max_models_on_disc=10, seed=42    47706   47706     896.7532          294.8573           0.0523            0.9492             0.9492                     0.0   0.5000    0.6931      0.9619   5406.96              0.9999     415.72       5404.92                   0.0                          0.0         11.7377              0.7886      5404.92              1.0000      6473.72                 0.7792           0.0629    5404.92            1.0000             4964.36            1.1757                  0.9325     5404.92             1.0000              5951.20          5-fold cross-fit of the full AutoML run          6025.4628         0.4010         0.5928           0.4009           5404.92                   1.0000                    5898.20
```

