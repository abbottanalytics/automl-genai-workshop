#######################################################
# automl_harness_compare.py
#
# Collect results/*.json from the AutoGluon and auto-sklearn harness runs into
# comparison tables (speed, default-metric accuracy, KDD98 cumulative net
# revenue). Optionally add reference rows from the TabICL study
# (same KDD98 prep, split and metric definitions).
#
#   python automl_harness_compare.py [--tabicl-results <dir>]
# Writes results/comparison_titanic.csv, comparison_kdd98.csv and
# results/comparison.md
#######################################################
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"


def load():
    rows = []
    for f in sorted(RES.glob("*.json")):
        d = json.loads(f.read_text())
        if "tool" in d:
            cp = RES / f"{d['run_name']}_cutoff_preds.csv"
            if cp.exists():   # out-of-fold TRAIN predictions the cutoff was chosen on
                c = pd.read_csv(cp, usecols=["y", "p"])
                d["oof_mean_pred"] = float(c.p.mean())
            rows.append(d)
    return pd.DataFrame(rows)


def as_table(df):
    """Markdown if tabulate is available, else a fixed-width text table."""
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```\n" + df.to_string(index=False) + "\n```"


def fmt_money(x):
    return "" if pd.isna(x) else f"${x:,.0f}"


def tabicl_reference(r: Path) -> pd.DataFrame:
    """LR, XGBoost and TabICL rows from the TabICL study, scored with train-chosen cutoffs."""
    b = pd.read_csv(r / "metrics_TARGET_B.csv")
    th = pd.read_csv(r / "oof_thresholds.csv")
    th = th[th.target == "TARGET_B"]
    rows = []
    for model, n, label in [("lr", 47706, "Logistic/Ridge all"), ("xgb", 47706, "XGBoost all"), ("tabicl", 5000, "TabICL 5k")]:
        m = b[(b.model == model) & (b.n_context == n)].iloc[0]
        single = th[(th.kind == "single") & (th.p_model == label)].iloc[0]
        fam = label.rsplit(" ", 1)[0]
        pair = th[(th.kind == "pairing") & (th.p_model == label) & (th.gift_model == fam)].iloc[0]
        rows.append({"model": f"{label} (gift: {fam})", "n_train": n, "AUC": round(m.auc, 3), "log loss": round(m.log_loss, 4),
                     "mean P": round(m.mean_pred, 4), "predict s": round(m.predict_s, 2),
                     "CNR, P cutoff (train)": fmt_money(single.test_profit),
                     "CNR, EV > $0.68": fmt_money(pair.ev_rule_profit),
                     "CNR, EV cutoff (train)": fmt_money(pair.test_profit),
                     "% mailed, EV cutoff": round(100 * pair.test_mailed, 1),
                     "EV hindsight peak (upper bound)": fmt_money(pair.test_hindsight_peak)})
    return pd.DataFrame(rows)


def headline(df, args):
    """One short table per dataset: speed, default-metric accuracy, and (KDD98) net revenue."""
    out = ["## Headline\n"]
    d = df.copy()
    d["minutes"] = (d.total_seconds / 60).round(1)
    t = d[d.target == "survived"]
    if len(t):
        h = pd.DataFrame({"run": t.run_name, "minutes": t.minutes,
                          "accuracy (baseline)": [f"{a:.3f} ({b:.3f})" for a, b in zip(t.accuracy_default, t.accuracy_baseline)],
                          "AUC": t.roc_auc.round(3)})
        out += ["**Titanic**\n", as_table(h), "\n"]
    k = d[(d.target == "TARGET_B") & d.get("cnr_peak", pd.Series(index=d.index)).notna()]
    if len(k):
        g = lambda c: k[c] if c in k else pd.Series(index=k.index, dtype=float)
        h = pd.DataFrame({"run": k.run_name, "minutes (incl. gift model)": k.minutes,
                          "accuracy (baseline)": [f"{a:.4f} ({b:.4f})" for a, b in zip(k.accuracy_default, k.accuracy_baseline)],
                          "% mailed by tool default": (100 * k.mailed_pct_default_decision).round(1),
                          "AUC": k.roc_auc.round(3), "mean P": k.mean_pred.round(4),
                          "CNR, P cutoff (train)": g("p_cut_cnr").map(fmt_money),
                          "CNR, EV > $0.68": k.ev_rule_cnr.map(fmt_money),
                          "CNR, EV cutoff (train)": g("ev_cut_cnr").map(fmt_money),
                          "% mailed, EV cutoff": (100 * g("ev_cut_mailed_pct")).round(1),
                          "CNR, EV depth (train)": g("ev_cut_cnr_at_depth").map(fmt_money),
                          "% mailed, EV depth": (100 * g("ev_cut_train_depth_pct")).round(1),
                          "CNR, EV cutoff, fold-average models": g("cvavg_ev_cut_cnr").map(fmt_money),
                          "OOF mean P": g("oof_mean_pred").round(4),
                          "EV hindsight peak (upper bound)": k.ev_cnr_peak.map(fmt_money),
                          "CNR mail all": k.cnr_mail_all.map(fmt_money)})
        out += ["**KDD Cup 1998** (47,706 test rows; 38-input prep and 50/50 split in data/cup98lrn_automl_prepped_with_split.tsv; models fit on all of TRAIN; cutoffs chosen where CNR peaks on out-of-fold TRAIN predictions (bagged OOF for best_quality, otherwise a 5-fold cross-fit of the whole run); mean response rate 0.0508)\n", as_table(h), "\n"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tabicl-results", default=None)
    args = ap.parse_args()
    df = load()
    if df.empty:
        print("no results yet")
        return
    gift_s = df["gift_fit_seconds"].fillna(0) if "gift_fit_seconds" in df else 0
    df["total_seconds"] = df.fit_seconds + gift_s
    md = ["# AutoML harness comparison\n"]

    t = df[df.target == "survived"]
    if len(t):
        tt = t[["run_name", "tool", "tool_version", "settings", "n_train", "fit_seconds", "predict_seconds",
                "accuracy_default", "accuracy_baseline", "roc_auc", "log_loss", "best_val_score"]].copy() \
            if "best_val_score" in t else t
        tt.to_csv(RES / "comparison_titanic.csv", index=False)
        md += ["## Titanic (survived), 30% stratified hold-out\n", as_table(tt.round(4)), "\n"]

    k = df[df.target == "TARGET_B"].copy()
    if len(k):
        cols = ["run_name", "tool", "settings", "n_train", "n_test", "fit_seconds", "gift_fit_seconds", "predict_seconds",
                "accuracy_default", "accuracy_baseline", "pct_predicted_positive", "roc_auc", "log_loss", "lift_top10",
                "cnr_peak", "cnr_peak_depth_pct", "cnr_top10", "cnr_mail_all", "cnr_default_decision",
                "mailed_pct_default_decision", "gift_rmse_resp", "gift_spearman_resp", "ev_rule_cnr",
                "ev_rule_mailed_pct", "ev_cnr_peak", "ev_cnr_peak_depth_pct", "n_cutoff_holdout",
                "p_cut_threshold", "p_cut_cnr", "p_cut_mailed_pct", "p_cut_cnr_at_depth",
                "ev_cut_threshold", "ev_cut_train_depth_pct", "ev_cut_cnr", "ev_cut_mailed_pct", "ev_cut_cnr_at_depth",
                "cutoff_method", "cutoff_cv_seconds", "oof_mean_pred", "cvavg_roc_auc", "cvavg_mean_pred",
                "cvavg_ev_cut_cnr", "cvavg_ev_cut_mailed_pct", "cvavg_ev_cut_cnr_at_depth"]
        kk = k[[c for c in cols if c in k]]
        kk.to_csv(RES / "comparison_kdd98.csv", index=False)
        md += ["## KDD Cup 1998 (TARGET_B; CNR uses TARGET_D, $0.68 mail cost)\n", as_table(kk.round(4)), "\n"]
        if args.tabicl_results:
            ref = tabicl_reference(Path(args.tabicl_results))
            ref.to_csv(RES / "comparison_kdd98_with_tabicl_study.csv", index=False)
            md += ["## Reference: individual models and TabICL from the TabICL study (same prep and test rows)\n",
                   "Cutoffs chosen on 5-fold out-of-fold TRAIN predictions, models refit on all 47,706 TRAIN rows; "
                   "EV pairs a response model with the gift model of the same family.\n",
                   as_table(ref), "\n"]
    md = md[:1] + headline(df, args) + md[1:]
    (RES / "comparison.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
