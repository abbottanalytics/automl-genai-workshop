#######################################################
# automl_harness_autosklearn.py
#
# Run auto-sklearn 0.15 (the last release, Sept 2022) with its DEFAULTS on a
# binary target and score it on a hold-out test set. Optional for KDD Cup
# 1998: a responder-only gift model, then expected value P x gift and
# cumulative net revenue.
#
# auto-sklearn's default classification metric is accuracy, and the default
# regression metric is R^2. Neither knows about mail cost or donations.
#
# Setup (macOS arm64 or Linux; Windows is not supported):
#   conda create -n autosklearn -c conda-forge python=3.9 auto-sklearn=0.15.0
#
#   (macOS: see README.md)
#
# Titanic (run from automl_harness/ with the autosklearn environment's python; see run_*.sh):
#   python automl_harness_autosklearn.py --data ../data/titanic3.csv --target survived \
#       --drop name ticket cabin embarked boat body home.dest --time-limit 600 \
#       --run-name titanic_autosklearn_default
# KDD Cup 1998 (with cumulative net revenue):
#   python automl_harness_autosklearn.py --data ../data/cup98lrn_automl_prepped_with_split.tsv \
#       --split-col is_test --id-col RECORD_ID --target TARGET_B --gift-col TARGET_D \
#       --time-limit 900 --n-jobs 2 --max-models-on-disc 10 --run-name kdd98_autosklearn_default
#######################################################
from __future__ import annotations

import argparse
import shutil
import tempfile

from pathlib import Path

import numpy as np
import pandas as pd

import automl_harness_common as H


def main():
    ap = H.add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--gift-time-limit", type=int, default=None, help="seconds for the gift model (default max(300, time_limit/3))")
    ap.add_argument("--metric", default=None,
                    help="classification metric name from autosklearn.metrics (e.g. roc_auc, log_loss); default accuracy")
    ap.add_argument("--max-models-on-disc", type=int, default=50,
                    help="models auto-sklearn keeps on disk (its default 50); lower it on big data to save disk")
    ap.add_argument("--memory-limit", type=int, default=None,
                    help="MB per worker. Default: 3072 on Linux, none on macOS (macOS cannot set the rlimit "
                         "auto-sklearn uses, so any limit crashes every run)")
    args = ap.parse_args()
    import sys
    if args.memory_limit is None and not sys.platform.startswith("darwin"):
        args.memory_limit = 3072
    if sys.platform.startswith("darwin") and args.memory_limit:
        print("macOS: ignoring --memory-limit (RLIMIT_AS is not supported); using memory_limit=None")
        args.memory_limit = None

    import autosklearn
    import autosklearn.classification
    import autosklearn.metrics
    import autosklearn.regression

    df = H.load_table(args.data)
    train, test = H.split_data(df, args.target, args.test_size, args.seed, args.split_col, args.test_data)
    drop = list(args.drop) + ([args.gift_col] if args.gift_col else []) + ([args.id_col] if args.id_col else [])
    feats = H.feature_columns(train, args.target, drop)
    train, test = H.as_categories(train, test, feats)
    hold = None
    if args.gift_col and args.cutoff_holdout > 0:   # KDD98: keep part of TRAIN aside to choose the mailing cutoff
        train, hold = H.holdout_split(train, args.target, args.cutoff_holdout, args.seed)
    y_tr = train[args.target].astype(int).to_numpy()
    y_te = test[args.target].astype(int).to_numpy()
    print(f"train {len(train):,}  test {len(test):,}  features {len(feats)}  rate {y_tr.mean():.4f}")

    tmp = tempfile.mkdtemp(prefix="autosk_")

    def fit_models(part, path):
        """Fit the TARGET_B classifier and (KDD98) the responder-only gift model on `part`."""
        c = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=args.time_limit, seed=args.seed, n_jobs=args.n_jobs,
            memory_limit=args.memory_limit, tmp_folder=f"{path}/clf", delete_tmp_folder_after_terminate=True,
            max_models_on_disc=args.max_models_on_disc,
            metric=getattr(autosklearn.metrics, args.metric) if args.metric else None)
        with H.Timer() as t_c:
            c.fit(part[feats], part[args.target].astype(int).to_numpy())
        r, t_r = None, None
        if args.gift_col and not args.no_gift_model:
            resp = part[part[args.target].astype(int) == 1]
            r = autosklearn.regression.AutoSklearnRegressor(
                time_left_for_this_task=args.gift_time_limit or max(300, args.time_limit // 3), seed=args.seed, n_jobs=args.n_jobs,
                memory_limit=args.memory_limit, tmp_folder=f"{path}/reg", delete_tmp_folder_after_terminate=True,
                max_models_on_disc=args.max_models_on_disc)
            with H.Timer() as t_r:
                r.fit(resp[feats], resp[args.gift_col].to_numpy(float))
        return c, r, t_c, t_r

    clf, reg, t_fit, t_reg = fit_models(train, f"{tmp}/final")
    with H.Timer() as t_pred:
        p = clf.predict_proba(test[feats])[:, 1]
        yhat = clf.predict(test[feats])   # argmax of predict_proba: a fixed 0.5 cutoff; auto-sklearn has no threshold setting

    gift = test[args.gift_col].fillna(0).to_numpy(float) if args.gift_col else None
    result = {"tool": "auto-sklearn", "tool_version": autosklearn.__version__, "data": args.data,
              "target": args.target, "settings": (f"metric={args.metric}" if args.metric else "defaults (metric=accuracy)") + f", time_limit={args.time_limit}s, "
                                                 f"n_jobs={args.n_jobs}, memory_limit={args.memory_limit}, max_models_on_disc={args.max_models_on_disc}, seed={args.seed}",
              "n_train": len(train), "n_features": len(feats),
              "fit_seconds": t_fit.seconds, "predict_seconds": t_pred.seconds,
              **H.classification_scores(y_te, p, yhat, gift, args.mail_cost)}
    try:
        lb = clf.leaderboard()
        result["n_models_in_ensemble"] = int(len(lb))
        result["ensemble_members"] = lb["type"].value_counts().to_dict()
    except Exception as e:  # leaderboard API changed across versions
        result["leaderboard_error"] = str(e)
    result["sprint_statistics"] = clf.sprint_statistics()
    preds = pd.DataFrame({"row": test[args.id_col].to_numpy() if args.id_col else np.arange(len(test)), "y": y_te, "p": p, "yhat": yhat})

    # ---- KDD98: responder-only gift model, then expected value ----
    g = None
    if reg is not None:
        g = reg.predict(test[feats])
        result["gift_fit_seconds"] = t_reg.seconds
        result.update(H.regression_scores(gift, y_te, g))
        result.update(H.expected_value_scores(p, g, gift, args.mail_cost))
        preds["gift_pred"] = g
        preds["gift"] = gift

    # ---- KDD98: choose the mailing cutoff on TRAIN, apply it to TEST ----
    if args.gift_col and hold is None and args.cutoff_cv > 0:
        # auto-sklearn exposes no out-of-fold predictions for its ensemble, so cross-fit the whole run
        gift_tr = train[args.gift_col].fillna(0).to_numpy(float)
        p_oof = np.zeros(len(train)); g_oof = np.zeros(len(train)) if reg is not None else None
        fold_info = []
        p_avg = np.zeros(len(test)); g_avg = np.zeros(len(test)) if reg is not None else None   # fold-average test scores
        with H.Timer() as t_cv:
            for k, (tr_i, va_i) in enumerate(H.crossfit_folds(train, args.target, args.cutoff_cv, args.seed)):
                print(f"cross-fit fold {k + 1}/{args.cutoff_cv}", flush=True)
                c_k, r_k, _, _ = fit_models(train.iloc[tr_i], f"{tmp}/cv{k}")
                p_oof[va_i] = c_k.predict_proba(train.iloc[va_i][feats])[:, 1]
                try:
                    members = c_k.leaderboard()["type"].value_counts().to_dict()
                except Exception as e:
                    members = str(e)
                fold_info.append({"fold": k + 1, "ensemble_members": members,
                                  "oof_auc": float(H.roc_auc_score(train.iloc[va_i][args.target].astype(int), p_oof[va_i])),
                                  "oof_mean_p": float(p_oof[va_i].mean())})
                p_avg += c_k.predict_proba(test[feats])[:, 1] / args.cutoff_cv
                if r_k is not None:
                    g_oof[va_i] = r_k.predict(train.iloc[va_i][feats])
                    g_avg += r_k.predict(test[feats]) / args.cutoff_cv
                del c_k, r_k
                shutil.rmtree(f"{tmp}/cv{k}", ignore_errors=True)
        result.update({"cutoff_method": f"{args.cutoff_cv}-fold cross-fit of the full AutoML run",
                       "cutoff_cv_seconds": t_cv.seconds, "n_cutoff_rows": len(train), "cutoff_folds": fold_info})
        result.update(H.train_cutoff_scores(p_oof, g_oof, gift_tr, p, g, gift, args.mail_cost))
        # same cutoffs applied to the average of the K fold models (the models the cutoff was chosen on)
        result.update(H.train_cutoff_scores(p_oof, g_oof, gift_tr, p_avg, g_avg, gift, args.mail_cost, prefix="cvavg_"))
        result["cvavg_roc_auc"] = float(H.roc_auc_score(y_te, p_avg)); result["cvavg_mean_pred"] = float(p_avg.mean())
        preds["p_cvavg"] = p_avg
        if g_avg is not None:
            preds["gift_pred_cvavg"] = g_avg
        H.write_cutoff_preds(args.out, args.run_name, y_tr, p_oof, gift_tr, g_oof,
                             train[args.id_col].to_numpy() if args.id_col else None)
    elif hold is not None:   # single hold-out alternative
        gift_hold = hold[args.gift_col].fillna(0).to_numpy(float)
        p_hold = clf.predict_proba(hold[feats])[:, 1]
        g_hold = reg.predict(hold[feats]) if reg is not None else None
        result.update({"cutoff_method": f"{args.cutoff_holdout:.0%} hold-out of TRAIN", "n_cutoff_rows": len(hold)})
        result.update(H.train_cutoff_scores(p_hold, g_hold, gift_hold, p, g, gift, args.mail_cost))
        H.write_cutoff_preds(args.out, args.run_name, hold[args.target].astype(int).to_numpy(), p_hold, gift_hold, g_hold,
                             hold[args.id_col].to_numpy() if args.id_col else None)
    shutil.rmtree(tmp, ignore_errors=True)
    H.write_result(args.out, args.run_name, result, preds)


if __name__ == "__main__":
    main()
