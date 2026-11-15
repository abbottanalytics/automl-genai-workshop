#######################################################
# automl_harness_autogluon.py
#
# Run AutoGluon-Tabular on a binary target and score it on a hold-out test
# set. Same data handling and metrics as automl_harness_autosklearn.py, so
# results compare directly. Optional for KDD Cup 1998: a responder-only gift
# model, then expected value P x gift and cumulative net revenue.
#
# --presets / --eval-metric default to AutoGluon's own defaults
# (medium_quality, accuracy), which is what you get from
#   TabularPredictor(label=...).fit(data)
#
# Setup:
#   conda create -n autogluon16 python=3.11
#   conda run -n autogluon16 pip install "autogluon.tabular[all]==1.6.3"
#
# Titanic, defaults vs. the workshop settings (run from automl_harness/; see run_*.sh for all runs):
#   python automl_harness_autogluon.py --data ../data/titanic3.csv --target survived \
#       --drop name ticket cabin embarked boat body home.dest --no-time-limit \
#       --run-name titanic_autogluon_default
#   python automl_harness_autogluon.py --data ../data/titanic3.csv --target survived \
#       --drop name ticket cabin embarked boat body home.dest \
#       --presets best_quality --eval-metric roc_auc --time-limit 600 \
#       --run-name titanic_autogluon_best_auc
#######################################################
from __future__ import annotations

import argparse
import shutil
import tempfile

from pathlib import Path

import numpy as np
import pandas as pd

import automl_harness_common as H


def fit_models(train, feats, args, path, time_limit, verbosity=2):
    """Fit the TARGET_B classifier and (KDD98) the responder-only gift model on `train`."""
    from autogluon.tabular import TabularPredictor
    fit_kw = {"time_limit": time_limit}
    if args.presets:
        fit_kw["presets"] = args.presets
    if args.excluded_model_types:
        fit_kw["excluded_model_types"] = args.excluded_model_types
    pred_kw = {"label": args.target, "problem_type": "binary", "path": f"{path}/clf", "verbosity": verbosity}
    if args.eval_metric:
        pred_kw["eval_metric"] = args.eval_metric
    clf = TabularPredictor(**pred_kw)
    with H.Timer() as t_fit:
        clf.fit(train[feats + [args.target]], **fit_kw)
    reg, t_reg = None, None
    if args.gift_col and not args.no_gift_model:
        resp = train[train[args.target] == 1]
        reg = TabularPredictor(label=args.gift_col, problem_type="regression", path=f"{path}/reg", verbosity=1)
        with H.Timer() as t_reg:
            reg.fit(resp[feats + [args.gift_col]],
                    **({**fit_kw, "time_limit": None if time_limit is None else max(300, time_limit // 3)}))
    return clf, reg, t_fit, t_reg


def ensemble_members(clf):
    """Models the best (weighted-ensemble) model actually uses."""
    try:
        return [m for m in clf._trainer.get_minimum_model_set(clf.model_best) if m != clf.model_best]
    except Exception as e:   # internal API; may change across versions
        return [f"unavailable: {e}"]


def p_of(clf, X):
    return clf.predict_proba(X)[clf.positive_class].to_numpy()


def main():
    ap = H.add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--presets", default=None, help="e.g. medium_quality (default), best_quality, extreme")
    ap.add_argument("--eval-metric", default=None, help="e.g. roc_auc, log_loss (default: accuracy)")
    ap.add_argument("--no-time-limit", action="store_true", help="AutoGluon default: no time limit")
    ap.add_argument("--excluded-model-types", nargs="*", default=None)
    args = ap.parse_args()

    import autogluon.tabular as agt

    df = H.load_table(args.data)
    train, test = H.split_data(df, args.target, args.test_size, args.seed, args.split_col, args.test_data)
    drop = list(args.drop) + ([args.gift_col] if args.gift_col else []) + ([args.id_col] if args.id_col else [])
    feats = H.feature_columns(train, args.target, drop)
    train, test = H.as_categories(train, test, feats)
    hold = None
    if args.gift_col and args.cutoff_holdout > 0:   # optional: keep part of TRAIN aside to choose the cutoff
        train, hold = H.holdout_split(train, args.target, args.cutoff_holdout, args.seed)
    train[args.target] = train[args.target].astype(int)
    y_te = test[args.target].astype(int).to_numpy()
    print(f"train {len(train):,}  test {len(test):,}  features {len(feats)}  rate {train[args.target].mean():.4f}")

    tmp = tempfile.mkdtemp(prefix="ag_")
    time_limit = None if args.no_time_limit else args.time_limit
    predictor, reg, t_fit, t_reg = fit_models(train, feats, args, f"{tmp}/final", time_limit)
    with H.Timer() as t_pred:
        p = p_of(predictor, test[feats])
        yhat = predictor.predict(test[feats]).astype(int).to_numpy()

    gift = test[args.gift_col].fillna(0).to_numpy(float) if args.gift_col else None
    lb = predictor.leaderboard(silent=True)
    result = {"tool": "AutoGluon", "tool_version": agt.__version__, "data": args.data, "target": args.target,
              "settings": f"presets={args.presets or 'default (medium_quality)'}, "
                          f"eval_metric={predictor.eval_metric.name}, time_limit={time_limit}",
              "n_train": len(train), "n_features": len(feats),
              "fit_seconds": t_fit.seconds, "predict_seconds": t_pred.seconds,
              "best_model": predictor.model_best, "ensemble_members": ensemble_members(predictor), "best_val_score": float(lb["score_val"].iloc[0]),
              "n_models_trained": int(len(lb)), "models": lb["model"].tolist(),
              **H.classification_scores(y_te, p, yhat, gift, args.mail_cost)}
    preds = pd.DataFrame({"row": test[args.id_col].to_numpy() if args.id_col else np.arange(len(test)), "y": y_te, "p": p, "yhat": yhat})

    g = None
    if reg is not None:
        g = reg.predict(test[feats]).to_numpy()
        result["gift_fit_seconds"] = t_reg.seconds
        result["gift_eval_metric"] = reg.eval_metric.name
        result.update(H.regression_scores(gift, y_te, g))
        result.update(H.expected_value_scores(p, g, gift, args.mail_cost))
        preds["gift_pred"] = g
        preds["gift"] = gift

    # ---- KDD98: choose the mailing cutoff on TRAIN, apply it to TEST ----
    if args.gift_col and hold is None and args.cutoff_cv > 0:
        gift_tr = train[args.gift_col].fillna(0).to_numpy(float)
        if predictor._trainer.bagged_mode:
            # bagged models already have out-of-fold predictions for every TRAIN row
            method = "bagged out-of-fold (AutoGluon predict_proba_oof)"
            p_oof = predictor.predict_proba_oof()[predictor.positive_class].loc[train.index].to_numpy()
            g_oof = None
            if reg is not None:
                g_oof = pd.Series(reg.predict(train[feats]).to_numpy(), index=train.index)  # non-responders: out of sample
                if reg._trainer.bagged_mode:
                    g_oof.loc[reg.predict_oof().index] = reg.predict_oof()            # responders: out of fold
                g_oof = g_oof.to_numpy()
            t_cv_s = 0.0
        else:
            # cross-fit: refit the whole AutoML run K times, each time predicting the left-out fold
            method = f"{args.cutoff_cv}-fold cross-fit of the full AutoML run"
            p_oof = np.zeros(len(train)); g_oof = np.zeros(len(train)) if reg is not None else None
            fold_info = []
            p_avg = np.zeros(len(test)); g_avg = np.zeros(len(test)) if reg is not None else None
            with H.Timer() as t_cv:
                for k, (tr_i, va_i) in enumerate(H.crossfit_folds(train, args.target, args.cutoff_cv, args.seed)):
                    print(f"cross-fit fold {k + 1}/{args.cutoff_cv}")
                    c_k, r_k, _, _ = fit_models(train.iloc[tr_i], feats, args, f"{tmp}/cv{k}", time_limit, verbosity=1)
                    p_oof[va_i] = p_of(c_k, train.iloc[va_i][feats])
                    fold_info.append({"fold": k + 1, "ensemble_members": ensemble_members(c_k),
                                      "oof_auc": float(H.roc_auc_score(train.iloc[va_i][args.target], p_oof[va_i])),
                                      "oof_mean_p": float(p_oof[va_i].mean())})
                    p_avg += p_of(c_k, test[feats]) / args.cutoff_cv
                    if r_k is not None:
                        g_oof[va_i] = r_k.predict(train.iloc[va_i][feats]).to_numpy()
                        g_avg += r_k.predict(test[feats]).to_numpy() / args.cutoff_cv
                    shutil.rmtree(f"{tmp}/cv{k}", ignore_errors=True)
            t_cv_s = t_cv.seconds
            result["cutoff_folds"] = fold_info
        result.update({"cutoff_method": method, "cutoff_cv_seconds": t_cv_s, "n_cutoff_rows": len(train)})
        result.update(H.train_cutoff_scores(p_oof, g_oof, gift_tr, p, g, gift, args.mail_cost))
        if not predictor._trainer.bagged_mode:   # same cutoffs on the average of the K fold models
            result.update(H.train_cutoff_scores(p_oof, g_oof, gift_tr, p_avg, g_avg, gift, args.mail_cost, prefix="cvavg_"))
            result["cvavg_roc_auc"] = float(H.roc_auc_score(y_te, p_avg)); result["cvavg_mean_pred"] = float(p_avg.mean())
            preds["p_cvavg"] = p_avg
            if g_avg is not None:
                preds["gift_pred_cvavg"] = g_avg
        H.write_cutoff_preds(args.out, args.run_name, train[args.target].to_numpy(), p_oof, gift_tr, g_oof,
                             train[args.id_col].to_numpy() if args.id_col else None)
    elif hold is not None:   # single hold-out alternative
        gift_hold = hold[args.gift_col].fillna(0).to_numpy(float)
        p_hold = p_of(predictor, hold[feats])
        g_hold = reg.predict(hold[feats]).to_numpy() if reg is not None else None
        result.update({"cutoff_method": f"{args.cutoff_holdout:.0%} hold-out of TRAIN", "n_cutoff_rows": len(hold)})
        result.update(H.train_cutoff_scores(p_hold, g_hold, gift_hold, p, g, gift, args.mail_cost))
        H.write_cutoff_preds(args.out, args.run_name, hold[args.target].astype(int).to_numpy(), p_hold, gift_hold, g_hold,
                             hold[args.id_col].to_numpy() if args.id_col else None)
    shutil.rmtree(tmp, ignore_errors=True)
    H.write_result(args.out, args.run_name, result, preds)


if __name__ == "__main__":
    main()
