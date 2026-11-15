#######################################################
# automl_harness_bootstrap.py
#
# How much would the KDD98 net-revenue results move on a different sample?
# Uses the saved predictions only (no refits):
#   results/<run>_cutoff_preds.csv  out-of-fold TRAIN scores (the cutoff is chosen here)
#   results/<run>_preds.csv         TEST scores
#
# Each arm = a score (P alone, or EV = P x predicted gift) plus a decision rule:
#   cut      mail TEST rows scoring at or above the TRAIN cutoff (score at the
#            out-of-fold CNR peak)
#   depth    mail the same share of TEST as the TRAIN peak depth
#   cvavg    'cut', with TEST scored by the average of the K cross-fit fold
#            models (the models the cutoff was chosen on); cross-fit runs only
#   rule068  EV > $0.68 (no TRAIN data used)
#
# Two bootstraps, both paired (every arm sees the same resampled rows):
#   test   cutoff fixed as chosen on all TRAIN rows; resample the 47,706 TEST
#          rows. Test-sample variation only.
#   joint  also resample the TRAIN out-of-fold rows and re-choose the cutoff
#          in every replicate. Adds cutoff variation.
# Neither refits the models (an AutoML run takes 20-80 minutes), so variation
# from refitting is not included.
#
# Optional --tabicl-results adds the TabICL study's logistic, XGBoost and
# TabICL models (same prep, split and out-of-fold method), each paired with the
# gift model of its own family.
#
# Writes results/bootstrap_arms.csv and results/bootstrap_diffs.csv.
#   python automl_harness_bootstrap.py --reps 2000 [--tabicl-results <dir>]
#######################################################
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
MC = 0.68


def cutoff(score, gift):
    """Threshold and depth share at the cumulative-net-revenue peak."""
    order = np.argsort(-score, kind="stable")
    cum = np.cumsum(gift[order] - MC)
    i = int(np.argmax(cum))
    return score[order[i]], (i + 1) / len(score)


def cnr_threshold(score, gift, thr):
    return float((gift[score >= thr] - MC).sum())


def cnr_depth(score, gift, depth):
    k = int(round(depth * len(score)))
    if k == 0:
        return 0.0
    top = np.argpartition(-score, k - 1)[:k]
    return float((gift[top] - MC).sum())


def load_arms(tabicl_dir: Path | None):
    """Return (train_ids, test_ids, gift_tr, gift_te, arms); every arm holds TRAIN and TEST scores."""
    arms, ids_tr, ids_te, gtr, gte = [], None, None, None, None
    for f in sorted(RES.glob("kdd98_*_cutoff_preds.csv")):
        run = f.name.replace("_cutoff_preds.csv", "")
        tr = pd.read_csv(f).set_index("row").sort_index()
        te = pd.read_csv(RES / f"{run}_preds.csv").set_index("row").sort_index()
        if ids_tr is None:
            ids_tr, ids_te = tr.index, te.index
            gtr, gte = tr.gift.to_numpy(float), te.gift.to_numpy(float)
        assert tr.index.equals(ids_tr) and te.index.equals(ids_te), f"{run}: rows differ"
        ev = lambda p, g: p * np.clip(g, 0, None)
        arms.append(dict(arm=f"{run} | P", run=run, rule="cut", s_tr=tr.p.to_numpy(), s_te=te.p.to_numpy()))
        if "gift_pred" in tr:
            s_tr, s_te = ev(tr.p.to_numpy(), tr.gift_pred.to_numpy()), ev(te.p.to_numpy(), te.gift_pred.to_numpy())
            for rule in ("cut", "depth", "rule068"):
                arms.append(dict(arm=f"{run} | EV {rule}", run=run, rule=rule, s_tr=s_tr, s_te=s_te))
            if "p_cvavg" in te:
                arms.append(dict(arm=f"{run} | EV cvavg", run=run, rule="cut", s_tr=s_tr,
                                 s_te=ev(te.p_cvavg.to_numpy(), te.gift_pred_cvavg.to_numpy())))
    if ids_tr is None:
        raise SystemExit("no results/kdd98_*_cutoff_preds.csv yet: run the harness with --cutoff-cv first")
    if tabicl_dir is not None:
        def pair(prefix, model, n):
            a = pd.read_parquet(tabicl_dir / "oof" / f"oof_{prefix}{model}_ctx{n}_seed0.parquet").set_index("RECORD_ID")
            b = pd.read_parquet(tabicl_dir / f"preds_{prefix}{model}_ctx{n}_seed0.parquet").set_index("RECORD_ID")
            return a.loc[ids_tr, "pred"].to_numpy(float), b.loc[ids_te, "pred"].to_numpy(float)
        for model, n, label in [("lr", 47706, "TabICL study: logistic"), ("xgb", 47706, "TabICL study: XGBoost"),
                                ("tabicl", 5000, "TabICL study: TabICL 5k")]:
            p_tr, p_te = pair("", model, n)
            g_tr, g_te = pair("TARGET_D_", model, 47706)
            arms.append(dict(arm=f"{label} | P", run=label, rule="cut", s_tr=p_tr, s_te=p_te))
            s_tr, s_te = p_tr * np.clip(g_tr, 0, None), p_te * np.clip(g_te, 0, None)
            for rule in ("cut", "depth", "rule068"):
                arms.append(dict(arm=f"{label} | EV {rule}", run=label, rule=rule, s_tr=s_tr, s_te=s_te))
    return ids_tr, ids_te, gtr, gte, arms


def score_arm(a, gift_tr, gift_te, i_tr=None, i_te=None, fixed=None):
    """CNR on (resampled) TEST for one arm; cutoff from (resampled) TRAIN unless `fixed` is given."""
    s_te = a["s_te"] if i_te is None else a["s_te"][i_te]
    g_te = gift_te if i_te is None else gift_te[i_te]
    if a["rule"] == "rule068":
        return cnr_threshold(s_te, g_te, np.nextafter(MC, np.inf))   # EV > 0.68
    if fixed is not None:
        thr, depth = fixed
    else:
        s_tr = a["s_tr"] if i_tr is None else a["s_tr"][i_tr]
        g_tr = gift_tr if i_tr is None else gift_tr[i_tr]
        thr, depth = cutoff(s_tr, g_tr)
    return cnr_depth(s_te, g_te, depth) if a["rule"] == "depth" else cnr_threshold(s_te, g_te, thr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tabicl-results", default=None)
    ap.add_argument("--reference", default="kdd98_autogluon_best_auc | EV cut",
                    help="arm every other arm is compared with in bootstrap_diffs.csv")
    args = ap.parse_args()
    ids_tr, ids_te, gtr, gte, arms = load_arms(Path(args.tabicl_results) if args.tabicl_results else None)
    n_tr, n_te = len(gtr), len(gte)
    fixed = [None if a["rule"] == "rule068" else cutoff(a["s_tr"], gtr) for a in arms]
    actual = np.array([score_arm(a, gtr, gte, fixed=f) for a, f in zip(arms, fixed)])
    rng = np.random.default_rng(args.seed)
    boot_test = np.zeros((args.reps, len(arms)))
    boot_joint = np.zeros((args.reps, len(arms)))
    for b in range(args.reps):
        i_te = rng.integers(0, n_te, n_te)
        i_tr = rng.integers(0, n_tr, n_tr)
        for j, a in enumerate(arms):
            boot_test[b, j] = score_arm(a, gtr, gte, i_te=i_te, fixed=fixed[j])
            boot_joint[b, j] = score_arm(a, gtr, gte, i_tr=i_tr, i_te=i_te)
        if (b + 1) % 250 == 0:
            print(f"  {b + 1}/{args.reps} replicates", flush=True)

    def q(x, p):
        return np.percentile(x, p, axis=0)
    out = pd.DataFrame({"arm": [a["arm"] for a in arms], "actual_test_cnr": actual,
                        "test_boot_mean": boot_test.mean(0), "test_boot_p05": q(boot_test, 5), "test_boot_p95": q(boot_test, 95),
                        "joint_boot_mean": boot_joint.mean(0), "joint_boot_p05": q(boot_joint, 5), "joint_boot_p95": q(boot_joint, 95),
                        "joint_boot_sd": boot_joint.std(0)})
    out.to_csv(RES / "bootstrap_arms.csv", index=False)
    names = [a["arm"] for a in arms]
    rows = []
    if args.reference in names:
        r = names.index(args.reference)
        for j, nm in enumerate(names):
            if j == r:
                continue
            for mode, B in (("test", boot_test), ("joint", boot_joint)):
                d = B[:, r] - B[:, j]
                rows.append({"reference": args.reference, "arm": nm, "bootstrap": mode, "actual_diff": actual[r] - actual[j],
                             "mean_diff": d.mean(), "p05": np.percentile(d, 5), "p95": np.percentile(d, 95),
                             "share_reference_better": float((d > 0).mean())})
    diffs = pd.DataFrame(rows)
    diffs.to_csv(RES / "bootstrap_diffs.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.round(0).to_string(index=False))
    if len(diffs):
        print(diffs[diffs.bootstrap == "joint"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
