#######################################################
# automl_harness_common.py
#
# Shared pieces for the AutoML workshop harness (AutoGluon and auto-sklearn):
#   - load a tab-separated, CSV or parquet file and make a reproducible
#     stratified train/test split (or use an existing split column)
#   - score a classifier with the tool's DEFAULT decision (predict) and with
#     its probabilities: accuracy, ROC AUC, log loss, lift in the top 10%
#   - KDD Cup 1998 business metric: cumulative net revenue (gift - mail cost,
#     summed down the file sorted by score), its peak and the depth of the peak
#   - expected value P(respond) x predicted gift (Zadrozny & Elkan 2001):
#     mail when EV > mail cost
#   - write one JSON result file per run
#
# Works on Python 3.9 (auto-sklearn) and 3.11 (AutoGluon).
# Abbott Analytics, Nov 2026.
#######################################################
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

MAIL_COST = 0.68


# ---------------------------------------------------------------- data
def load_table(path: str) -> pd.DataFrame:
    p = Path(path).expanduser()
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix == ".csv":
        return pd.read_csv(p, low_memory=False)
    return pd.read_csv(p, sep="\t", low_memory=False)


def split_data(df: pd.DataFrame, target: str, test_size: float, seed: int,
               split_col: str | None = None, test_path: str | None = None):
    """Return (train, test). Priority: separate test file > split column > new split."""
    if test_path:
        return df.reset_index(drop=True), load_table(test_path).reset_index(drop=True)
    if split_col:
        is_test = df[split_col].astype(bool)
        return (df[~is_test].drop(columns=[split_col]).reset_index(drop=True),
                df[is_test].drop(columns=[split_col]).reset_index(drop=True))
    tr, te = train_test_split(df, test_size=test_size, stratify=df[target], random_state=seed)
    return tr.reset_index(drop=True), te.reset_index(drop=True)


def feature_columns(df: pd.DataFrame, target: str, drop: list[str]) -> list[str]:
    return [c for c in df.columns if c != target and c not in drop]


def as_categories(train: pd.DataFrame, test: pd.DataFrame, feats: list[str]):
    """Turn text columns into pandas category with one shared coding (both tools accept this)."""
    for c in feats:
        if not pd.api.types.is_numeric_dtype(train[c]):
            cats = pd.Index(sorted(set(train[c].astype(str)) | set(test[c].astype(str))))
            train[c] = pd.Categorical(train[c].astype(str), categories=cats)
            test[c] = pd.Categorical(test[c].astype(str), categories=cats)
    return train, test


# ---------------------------------------------------------------- metrics
def lift_top(y: np.ndarray, p: np.ndarray, frac: float) -> float:
    n = int(round(len(y) * frac))
    idx = np.argsort(-p, kind="stable")[:n]
    return float(y[idx].mean() / y.mean())


def cnr_curve(score: np.ndarray, gift: np.ndarray, mail_cost: float = MAIL_COST) -> np.ndarray:
    """Cumulative net revenue down the file sorted by score (highest first)."""
    order = np.argsort(-score, kind="stable")
    return np.cumsum(gift[order] - mail_cost)


def cnr_summary(score: np.ndarray, gift: np.ndarray, mail_cost: float = MAIL_COST) -> dict:
    cum = cnr_curve(score, gift, mail_cost)
    i = int(np.argmax(cum))
    n = len(cum)
    return {"cnr_peak": float(cum[i]), "cnr_peak_depth_pct": (i + 1) / n,
            "cnr_top10": float(cum[int(round(0.10 * n)) - 1]),
            "cnr_top50": float(cum[int(round(0.50 * n)) - 1]),
            "cnr_mail_all": float(cum[-1])}


def classification_scores(y: np.ndarray, p: np.ndarray, yhat: np.ndarray,
                          gift: np.ndarray | None = None, mail_cost: float = MAIL_COST) -> dict:
    """y: 0/1 truth; p: P(class 1); yhat: the tool's default predict() labels."""
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    out = {"n_test": int(len(y)), "true_rate": float(y.mean()),
           "accuracy_default": float(accuracy_score(y, yhat)),
           "accuracy_baseline": float(max(y.mean(), 1 - y.mean())),
           "pct_predicted_positive": float(np.mean(yhat == 1)),
           "roc_auc": float(roc_auc_score(y, p)), "log_loss": float(log_loss(y, pc)),
           "mean_pred": float(p.mean()), "lift_top10": lift_top(y, p, 0.10)}
    if gift is not None:
        out.update(cnr_summary(p, gift, mail_cost))
        mailed = yhat == 1   # what the tool's own default decision would mail
        out["cnr_default_decision"] = float((gift[mailed] - mail_cost).sum())
        out["mailed_pct_default_decision"] = float(mailed.mean())
    return out


def expected_value_scores(p: np.ndarray, gift_pred: np.ndarray, gift: np.ndarray,
                          mail_cost: float = MAIL_COST) -> dict:
    ev = p * np.clip(gift_pred, 0, None)
    mail = ev > mail_cost
    out = {"ev_rule_cnr": float((gift[mail] - mail_cost).sum()), "ev_rule_mailed_pct": float(mail.mean())}
    out.update({f"ev_{k}": v for k, v in cnr_summary(ev, gift, mail_cost).items()})
    return out


def regression_scores(gift: np.ndarray, b: np.ndarray, pred: np.ndarray) -> dict:
    m = b == 1
    err = pred[m] - gift[m]
    return {"gift_mae_resp": float(np.abs(err).mean()), "gift_rmse_resp": float(np.sqrt((err ** 2).mean())),
            "gift_spearman_resp": float(pd.Series(pred[m]).rank().corr(pd.Series(gift[m]).rank()))}


# ---------------------------------------------------------------- mailing cutoff chosen on training data
def holdout_split(train: pd.DataFrame, target: str, frac: float, seed: int):
    """Split the TRAINING data into a fit part and a cutoff hold-out part (stratified).
    The models never see the hold-out; it is used only to choose the mailing cutoff."""
    fit, hold = train_test_split(train, test_size=frac, random_state=seed, stratify=train[target])
    return fit.copy(), hold.copy()


def crossfit_folds(train: pd.DataFrame, target: str, k: int, seed: int):
    """Stratified K folds over TRAIN (positions). Each AutoML run is refit on K-1 folds
    and predicts the left-out fold, giving out-of-fold scores for every TRAIN row."""
    from sklearn.model_selection import StratifiedKFold
    return list(StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(train, train[target]))


def write_cutoff_preds(out_dir, run_name, y, p, gift, gift_pred=None, ids=None):
    """Save the TRAIN-side scores the cutoff was chosen on (for checking and bootstrapping)."""
    d = pd.DataFrame({"row": ids if ids is not None else np.arange(len(y)), "y": y, "p": p, "gift": gift})
    if gift_pred is not None:
        d["gift_pred"] = gift_pred
    out = Path(out_dir).expanduser(); out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / f"{run_name}_cutoff_preds.csv", index=False)


def choose_cutoff(score: np.ndarray, gift: np.ndarray, mail_cost: float = MAIL_COST):
    """Sort by score; the cutoff is the score at the row where cumulative net revenue peaks.
    Returns (threshold, depth share, peak $) on the data it was chosen on."""
    order = np.argsort(-score, kind="stable")
    cum = np.cumsum(gift[order] - mail_cost)
    i = int(np.argmax(cum))
    return float(score[order[i]]), (i + 1) / len(score), float(cum[i])


def apply_cutoff(score: np.ndarray, gift: np.ndarray, threshold: float, depth: float,
                 mail_cost: float = MAIL_COST) -> dict:
    """Net revenue on new data from mailing score >= threshold, and from mailing the same top share."""
    mail = score >= threshold
    order = np.argsort(-score, kind="stable")
    k = int(round(depth * len(score)))
    return {"cnr": float((gift[mail] - mail_cost).sum()), "mailed_pct": float(mail.mean()),
            "cnr_at_depth": float((gift[order[:k]] - mail_cost).sum())}


def train_cutoff_scores(p_tr, g_tr, gift_tr, p_test, g_test, gift_test,
                        mail_cost: float = MAIL_COST, prefix: str = "") -> dict:
    """Cutoffs chosen on TRAIN-side scores (out-of-fold or hold-out), applied to test.
    Two scores: P(respond) alone, and expected value P x predicted gift."""
    out = {}
    for name, sh, st in [("p", p_tr, p_test),
                         ("ev", None if g_tr is None else p_tr * np.clip(g_tr, 0, None),
                          None if g_test is None else p_test * np.clip(g_test, 0, None))]:
        if sh is None:
            continue
        thr, depth, peak = choose_cutoff(sh, gift_tr, mail_cost)
        res = apply_cutoff(st, gift_test, thr, depth, mail_cost)
        name = prefix + name
        out.update({f"{name}_cut_threshold": thr, f"{name}_cut_train_depth_pct": depth,
                    f"{name}_cut_train_peak": peak, f"{name}_cut_cnr": res["cnr"],
                    f"{name}_cut_mailed_pct": res["mailed_pct"], f"{name}_cut_cnr_at_depth": res["cnr_at_depth"]})
    return out


# ---------------------------------------------------------------- bookkeeping
class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.seconds = time.time() - self.t0


def write_result(out_dir: str, run_name: str, result: dict, preds: pd.DataFrame | None = None) -> Path:
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    result = {"run_name": run_name, "python": platform.python_version(),
              "machine": f"{platform.system()} {platform.machine()}",
              "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"), **result}
    path = out / f"{run_name}.json"
    path.write_text(json.dumps(result, indent=2, default=str))
    if preds is not None:
        preds.to_csv(out / f"{run_name}_preds.csv", index=False)
    print(json.dumps(result, indent=2, default=str))
    return path


def add_common_args(ap):
    ap.add_argument("--data", required=True, help="tab-separated file (or .parquet) with predictors + target")
    ap.add_argument("--target", required=True, help="binary 0/1 target column, e.g. survived or TARGET_B")
    ap.add_argument("--gift-col", default=None, help="KDD98: donation column (TARGET_D). Enables cumulative net revenue")
    ap.add_argument("--drop", nargs="*", default=[], help="columns never used as inputs (IDs, leaks)")
    ap.add_argument("--test-data", default=None, help="optional separate test file (else split --data)")
    ap.add_argument("--split-col", default=None, help="optional existing 0/1 test-flag column")
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--time-limit", type=int, default=600, help="seconds per model search")
    ap.add_argument("--mail-cost", type=float, default=MAIL_COST)
    ap.add_argument("--cutoff-cv", type=int, default=5,
                    help="KDD98: choose the mailing cutoff (max cumulative net revenue) on out-of-fold TRAIN scores. "
                         "Bagged AutoGluon runs use their own OOF predictions; otherwise the whole AutoML run is "
                         "cross-fit this many times. 0 = off")
    ap.add_argument("--cutoff-holdout", type=float, default=0.0,
                    help="alternative to --cutoff-cv: hold out this share of TRAIN for the cutoff (models fit on the rest)")
    ap.add_argument("--no-gift-model", action="store_true", help="skip the responder-only gift model (EV)")
    ap.add_argument("--id-col", default=None, help="optional record ID column, kept in the predictions file (never an input)")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
    return ap
