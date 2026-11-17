# -*- coding: utf-8 -*-
"""
Summary statistics reported in the article, computed from the run outputs.

    python analysis.py [results_dir] [sensitivity_json]

results_dir       directory with all_records.json (default: triage_results)
sensitivity_json  optional, Stage-2 sweep output
                  (default: triage_sensitivity/sensitivity_results.json)

Prints and writes <results_dir>/paper_statistics.json with:
  * Table 3  paired effect sizes of TRIAGE vs every baseline (attack-class F1):
             mean difference, 95% percentile bootstrap CI (20,000 resamples),
             Cohen's d_z, rank-biserial correlation, wins, Wilcoxon p
  * relative reduction in missed intrusions vs Stage 1 alone, with CI
  * Table 4  risk-control validity (escalation, calibration/test risk)
  * Table 5  per-class conformal coverage, marginal vs Mondrian
  * Section 7 (if the sweep file exists): one-way ANOVA omega^2 over the
             Stage-2 configurations, and Spearman trend of worst-class coverage
"""
import json
import os
import sys
import warnings

import numpy as np
from scipy import stats

N_BOOT, BOOT_SEED = 20_000, 0
warnings.filterwarnings("ignore", category=RuntimeWarning)   # single-seed runs


def boot_ci(x, rng):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    means = x[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def rank_biserial(d):
    d = np.asarray(d, float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def mean_sd(x):
    x = np.asarray(x, float)
    return {"mean": float(x.mean()), "sd": float(x.std())}   # population sd, as reported


def effect_sizes(recs, rng):
    out = {}
    for b in recs[0]["baselines"]:
        t = np.array([r["triage"]["f1_attack"] for r in recs])
        v = np.array([r["baselines"][b]["f1_attack"] for r in recs])
        d = t - v
        sd = d.std(ddof=1)
        p = (float(stats.wilcoxon(t, v).pvalue)
             if len(d) > 1 and np.any(d != 0) else None)
        out[b] = {"delta": float(d.mean()), "ci95": boot_ci(d, rng),
                  "d_z": float(d.mean() / sd) if sd > 0 else None,
                  "r_rb": rank_biserial(d), "wins": f"{int((d > 0).sum())}/{len(d)}",
                  "wilcoxon_p": p}
    fnr_t = np.array([r["triage"]["fnr"] for r in recs])
    fnr_1 = np.array([r["baselines"]["Stage1-only"]["fnr"] for r in recs])
    red = 1 - fnr_t / fnr_1
    out["_fnr_reduction_vs_stage1"] = {"mean": float(red.mean()), "ci95": boot_ci(red, rng)}
    return out


def risk_validity(recs):
    return {"escalation_cal": mean_sd([r["rcps_cal_escalation"] for r in recs]),
            "escalation_test": mean_sd([r["escalation_rate"] for r in recs]),
            "risk_cal": mean_sd([r["rcps_cal_risk"] for r in recs]),
            "risk_test": mean_sd([r["test_stage1_risk"] for r in recs]),
            "target_met": f"{sum(bool(r['risk_respected']) for r in recs)}/{len(recs)}"}


def class_coverage(recs):
    names = recs[0]["class_names"]
    rows = {}
    for c in names:
        m = [r["conformal_marginal"]["per_class"][c]["coverage"] for r in recs
             if c in r["conformal_marginal"]["per_class"]]
        k = [r["conformal_class_conditional"]["per_class"][c]["coverage"] for r in recs
             if c in r["conformal_class_conditional"]["per_class"]]
        n = [r["conformal_marginal"]["per_class"][c]["n"] for r in recs
             if c in r["conformal_marginal"]["per_class"]]
        if m:
            rows[c] = {"n": float(np.mean(n)), "marginal": float(np.mean(m)),
                       "mondrian": float(np.mean(k)), "delta": float(np.mean(k) - np.mean(m))}
    summ, per_seed_worst = {}, {}
    for key, lab in (("conformal_marginal", "marginal"),
                     ("conformal_class_conditional", "mondrian")):
        per_seed_worst[lab] = np.array([min(v["coverage"] for v in r[key]["per_class"].values())
                                        for r in recs])
        summ[lab] = {
            "coverage": float(np.mean([r[key]["marginal_coverage"] for r in recs])),
            "mean_set_size": float(np.mean([r[key]["mean_set_size"] for r in recs])),
            # Table 5 'worst class' row: lowest of the seed-averaged class coverages
            "worst_class": float(min(v[lab] for v in rows.values())),
            # per-seed worst class, averaged over seeds (used for the paired gain)
            "worst_class_per_seed_mean": float(per_seed_worst[lab].mean())}
    d = per_seed_worst["mondrian"] - per_seed_worst["marginal"]
    summ["worst_class_gain_paired"] = {
        "delta": float(d.mean()), "ci95": boot_ci(d, np.random.default_rng(BOOT_SEED)),
        "positive_seeds": f"{int((d > 0).sum())}/{len(d)}",
        "wilcoxon_p": float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else None}
    return {"per_class": rows, "summary": summ}


def omega_sq(groups):
    allv = np.concatenate(groups)
    k, n = len(groups), len(allv)
    grand = allv.mean()
    ssb = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups)
    ssw = sum(((np.asarray(g) - np.mean(g)) ** 2).sum() for g in groups)
    msw = ssw / (n - k)
    w2 = (ssb - (k - 1) * msw) / (ssb + ssw + msw)
    f, p = stats.f_oneway(*groups)
    return {"omega_sq": float(max(w2, 0.0)), "F": float(f), "p": float(p),
            "df": [k - 1, n - k]}


def sensitivity(sens):
    keys = sorted({(r["group"], r["name"]) for r in sens})
    def groups(f):
        return [np.array([f(r) for r in sens if (r["group"], r["name"]) == k]) for k in keys]
    out = {"anova": {
        "mean_set_size": omega_sq(groups(lambda r: r["conformal_marginal"]["mean_set_size"])),
        "cascade_f1": omega_sq(groups(lambda r: r["cascade"]["f1_attack"])),
        "coverage": omega_sq(groups(lambda r: r["conformal_marginal"]["coverage"]))}}
    for cal in ("conformal_marginal", "conformal_mondrian"):
        leaves = [r for r in sens if r["group"] == "num_leaves"]
        if leaves:
            x = [int(r["name"].split("=")[1]) for r in leaves]
            y = [r[cal]["worst_class_coverage"] for r in leaves]
            rho, p = stats.spearmanr(x, y)
            out[f"spearman_worst_vs_leaves_{cal}"] = {"rho": float(rho), "p": float(p)}
    return out


def main():
    res = sys.argv[1] if len(sys.argv) > 1 else "triage_results"
    sens_path = (sys.argv[2] if len(sys.argv) > 2
                 else os.path.join("triage_sensitivity", "sensitivity_results.json"))
    R = json.load(open(os.path.join(res, "all_records.json")))
    rng = np.random.default_rng(BOOT_SEED)
    out = {}
    for key in sorted({(r["dataset"], r["split_mode"]) for r in R}):
        recs = sorted([r for r in R if (r["dataset"], r["split_mode"]) == key],
                      key=lambda r: r["seed"])
        name = f"{key[0]}|{key[1]}"
        out[name] = {"n_seeds": len(recs), "effect_sizes_f1": effect_sizes(recs, rng),
                     "risk_validity": risk_validity(recs),
                     "class_coverage": class_coverage(recs)}
    if os.path.exists(sens_path):
        out["stage2_sensitivity"] = sensitivity(json.load(open(sens_path)))
    with open(os.path.join(res, "paper_statistics.json"), "w") as f:
        json.dump(out, f, indent=2)

    for name, e in out.items():
        if name == "stage2_sensitivity":
            continue
        print(f"\n=== {name} ({e['n_seeds']} seeds)")
        rv = e["risk_validity"]
        print(f"  risk: cal {rv['risk_cal']['mean']:.5f}±{rv['risk_cal']['sd']:.5f}  "
              f"test {rv['risk_test']['mean']:.5f}±{rv['risk_test']['sd']:.5f}  "
              f"escalation(test) {rv['escalation_test']['mean']:.3f}  met {rv['target_met']}")
        for b, s in e["effect_sizes_f1"].items():
            if b.startswith("_"):
                print(f"  missed-intrusion reduction vs Stage 1: {s['mean']:.1%} "
                      f"[{s['ci95'][0]:.1%}, {s['ci95'][1]:.1%}]")
                continue
            dz = "n/a" if s["d_z"] is None else f"{s['d_z']:.2f}"
            print(f"  vs {b:<26} Δ={s['delta']:+.4f} CI[{s['ci95'][0]:+.4f}, "
                  f"{s['ci95'][1]:+.4f}] d_z={dz} r_rb={s['r_rb']:+.2f} wins={s['wins']}")
        cs = e["class_coverage"]["summary"]
        print(f"  coverage marginal {cs['marginal']['coverage']:.4f} (worst {cs['marginal']['worst_class']:.3f}, "
              f"|C| {cs['marginal']['mean_set_size']:.3f})  Mondrian {cs['mondrian']['coverage']:.4f} "
              f"(worst {cs['mondrian']['worst_class']:.3f}, |C| {cs['mondrian']['mean_set_size']:.3f})")
        g = cs["worst_class_gain_paired"]
        print(f"  worst-class gain (per seed, Mondrian - marginal): {g['delta']:+.3f} "
              f"[{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}] positive {g['positive_seeds']} p={g['wilcoxon_p']}")
        for c, v in sorted(e["class_coverage"]["per_class"].items(), key=lambda kv: -kv[1]["n"]):
            print(f"    {c:<16} n={v['n']:>8.0f}  marginal {v['marginal']:.3f}  "
                  f"Mondrian {v['mondrian']:.3f}  Δ {v['delta']:+.3f}")
    if "stage2_sensitivity" in out:
        s = out["stage2_sensitivity"]
        print("\n=== Stage-2 sensitivity")
        for k, v in s["anova"].items():
            print(f"  {k:<14} omega^2={v['omega_sq']:.3f}  F({v['df'][0]},{v['df'][1]})={v['F']:.2f}  p={v['p']:.2g}")
        for k, v in s.items():
            if k.startswith("spearman"):
                print(f"  {k}: rho={v['rho']:+.2f} p={v['p']:.2g}")
    print(f"\nwritten: {os.path.join(res, 'paper_statistics.json')}")


if __name__ == "__main__":
    main()
