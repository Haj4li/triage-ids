# -*- coding: utf-8 -*-
"""
Redraw the article's result figures (Figs. 2-6) from the run outputs.

    python make_figures.py [results_dir] [sensitivity_json] [out_dir]

results_dir       directory with all_records.json (default: triage_results)
sensitivity_json  Stage-2 sweep output, needed only for Fig. 6
                  (default: triage_sensitivity/sensitivity_results.json)
out_dir           where the PNG/PDF files go (default: <results_dir>/paper_figures)
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

BLUE, ORANGE, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#555555"
plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "axes.axisbelow": True,
                     "savefig.dpi": 600, "savefig.bbox": "tight"})


def _save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)
    print("  ->", name)


def _recs(R, ds, mode):
    return sorted([r for r in R if r["dataset"] == ds and r["split_mode"] == mode],
                  key=lambda r: r["seed"])


def fig2_risk_validity(R, out):
    fig, axs = plt.subplots(1, 2, figsize=(3.35, 2.1), sharey=True)
    for ax, mode, title in ((axs[0], "iid", "Exchangeable split"),
                            (axs[1], "official", "Official partition")):
        rs = _recs(R, "UNSW-NB15", mode)
        if not rs:
            continue
        cal = [r["rcps_cal_risk"] for r in rs]
        te = [r["test_stage1_risk"] for r in rs]
        x, w = np.arange(len(rs)), .38
        ax.bar(x - w / 2, cal, w, color=BLUE, label="calibration risk")
        ax.bar(x + w / 2, te, w, color=ORANGE, label="test risk")
        ax.axhline(0.01, color="k", ls="--", lw=.9, label=r"target $\delta=0.01$")
        ax.set_yscale("log"); ax.set_ylim(3e-3, 3e-1)
        ax.set_xticks(x); ax.set_xticklabels([r["seed"] for r in rs])
        ax.set_xlabel("seed"); ax.set_title(title, fontsize=8)
        met = sum(bool(r["risk_respected"]) for r in rs)
        ax.text(.5, .96, f"target met: {met}/{len(rs)}", transform=ax.transAxes,
                ha="center", va="top", fontsize=7,
                bbox=dict(fc="white", ec="0.7", lw=.5, pad=1.5))
    axs[0].set_ylabel("Stage-1 risk (log scale)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=6.8,
               bbox_to_anchor=(.5, -.02))
    fig.tight_layout(rect=(0, .08, 1, 1))
    _save(fig, out, "fig2_risk_validity")


def fig3_operating_curve(R, out):
    series = [("UNSW-NB15", "iid", "UNSW-NB15 (exch.)", BLUE),
              ("UNSW-NB15", "official", "UNSW-NB15 (official)", ORANGE),
              ("CIC-IDS2017", "iid", "CIC-IDS2017 (exch.)", GREEN)]
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.3))
    for ds, mode, lab, col in series:
        rs = _recs(R, ds, mode)
        if not rs or "delta_sweep" not in rs[0]:
            continue
        ds_keys = sorted(rs[0]["delta_sweep"], key=float)
        d = np.array([float(k) for k in ds_keys])
        m = lambda f: np.array([np.mean([r["delta_sweep"][k][f] for r in rs]) for k in ds_keys])
        axs[0].plot(d, 100 * m("escalation_rate"), "o-", color=col, ms=3, label=lab)
        axs[1].plot(d, m("test_risk"), "o-", color=col, ms=3)
        axs[2].plot(d, m("f1_attack"), "o-", color=col, ms=3)
    lim = axs[1].get_xlim()
    axs[1].plot(lim, lim, "k--", lw=.8, label=r"validity line $R=\delta$")
    for ax in axs:
        ax.set_xscale("log"); ax.set_xlabel(r"risk target $\delta$")
    axs[1].set_yscale("log")
    axs[0].set_ylabel("escalation rate (%)"); axs[0].set_title("(a) Cost of a stricter target")
    axs[1].set_ylabel("achieved test risk"); axs[1].set_title("(b) Validity")
    axs[2].set_ylabel(r"$F_1$ (attack)"); axs[2].set_title("(c) Detection quality")
    h0, l0 = axs[0].get_legend_handles_labels(); h1, l1 = axs[1].get_legend_handles_labels()
    fig.legend(h0 + h1, l0 + l1, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(.5, -.04))
    fig.tight_layout(rect=(0, .07, 1, 1))
    _save(fig, out, "fig3_operating_curve")


def fig4_mondrian(R, out):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0), sharex=True)
    for ax, mode, title in ((axs[0], "iid", "Exchangeable split"),
                            (axs[1], "official", "Official partition")):
        rs = _recs(R, "UNSW-NB15", mode)
        if not rs:
            continue
        cls = list(rs[0]["conformal_marginal"]["per_class"])
        n = {c: np.mean([r["conformal_marginal"]["per_class"][c]["n"] for r in rs]) for c in cls}
        cls = sorted(cls, key=lambda c: n[c])          # rarest at the bottom
        for i, c in enumerate(cls):
            a = np.mean([r["conformal_marginal"]["per_class"][c]["coverage"] for r in rs])
            b = np.mean([r["conformal_class_conditional"]["per_class"][c]["coverage"] for r in rs])
            ax.annotate("", xy=(b, i), xytext=(a, i),
                        arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2))
            ax.plot(a, i, "o", color=BLUE, ms=4)
            ax.plot(b, i, "s", color=ORANGE, ms=4)
        ax.axvline(.95, color="k", ls="--", lw=.8)
        ax.set_yticks(range(len(cls)))
        ax.set_yticklabels([f"{c} ({n[c]:,.0f})" for c in cls], fontsize=7)
        ax.set_title(title); ax.set_xlabel("empirical coverage")
    axs[0].plot([], [], "o", color=BLUE, label="marginal calibration")
    axs[0].plot([], [], "s", color=ORANGE, label="Mondrian calibration")
    axs[0].legend(loc="lower left", fontsize=7, frameon=False)
    fig.tight_layout()
    _save(fig, out, "fig4_mondrian")


def fig5_budget(R, out):
    rs = _recs(R, "UNSW-NB15", "iid")
    if not rs or "budget_sweep" not in rs[0]:
        return
    keys = sorted(rs[0]["budget_sweep"], key=float)
    B = 100 * np.array([float(k) for k in keys])
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.2))
    for ax, f, col, lab, title, scale in (
            (axs[0], "f1_attack", BLUE, r"$F_1$ (attack)", "(a) Detection quality", 1),
            (axs[1], "fnr", ORANGE, "false-negative rate", "(b) Missed intrusions", 1),
            (axs[2], "cost", GREEN, r"operational cost ($\times10^3$)", "(c) Total cost", 1e-3)):
        v = np.array([[r["budget_sweep"][k][f] * scale for k in keys] for r in rs])
        mu, sd = v.mean(0), v.std(0)
        ax.plot(B, mu, "o-", color=col, ms=3)
        ax.fill_between(B, mu - sd, mu + sd, color=col, alpha=.2)
        ax.set_xlabel(r"analyst budget $B$ (%)"); ax.set_ylabel(lab); ax.set_title(title)
    fig.tight_layout()
    _save(fig, out, "fig5_budget")


def fig6_sensitivity(sens, out):
    keys = list(dict.fromkeys((r["group"], r["name"]) for r in sens))
    def omega(f):
        g = [np.array([f(r) for r in sens if (r["group"], r["name"]) == k]) for k in keys]
        allv = np.concatenate(g); k, n = len(g), len(allv)
        ssb = sum(len(x) * (x.mean() - allv.mean()) ** 2 for x in g)
        ssw = sum(((x - x.mean()) ** 2).sum() for x in g)
        msw = ssw / (n - k)
        return max((ssb - (k - 1) * msw) / (ssb + ssw + msw), 0.0), stats.f_oneway(*g).pvalue
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.3))
    items = [("coverage", lambda r: r["conformal_marginal"]["coverage"], GREY),
             (r"cascade $F_1$", lambda r: r["cascade"]["f1_attack"], GREEN),
             (r"set size $|C|$", lambda r: r["conformal_marginal"]["mean_set_size"], ORANGE)]
    for i, (lab, f, col) in enumerate(items):
        w, p = omega(f)
        axs[0].barh(i, 100 * w, color=col)
        axs[0].text(100 * w + 2, i, f"{100*w:.0f}%  (p={p:.1g})", va="center", fontsize=7)
    axs[0].set_yticks(range(3)); axs[0].set_yticklabels([x[0] for x in items])
    axs[0].set_xlim(0, 140); axs[0].set_xlabel("variance explained by Stage-2 choice (%)")
    axs[0].set_title("What the choice controls")
    groups = list(dict.fromkeys(r["group"] for r in sens))
    cols = dict(zip(groups, [BLUE, ORANGE, GREEN, "#CC79A7", GREY]))
    for g in groups:
        rr = [r for r in sens if r["group"] == g]
        axs[1].scatter([r["stage2_alone"]["ece"] for r in rr],
                       [r["conformal_marginal"]["mean_set_size"] for r in rr],
                       s=8, color=cols[g], label=g.replace("_", " "))
    axs[1].set_xlabel("Stage-2 calibration error (ECE)"); axs[1].set_ylabel("mean set size")
    axs[1].legend(fontsize=6, frameon=False); axs[1].set_title("Miscalibration widens sets")
    leaves = [r for r in sens if r["group"] == "num_leaves"]
    if leaves:
        L = sorted({int(r["name"].split("=")[1]) for r in leaves})
        for cal, col, lab, mk in (("conformal_marginal", BLUE, "marginal", "o"),
                                  ("conformal_mondrian", ORANGE, "Mondrian", "s")):
            v = [[r[cal]["worst_class_coverage"] for r in leaves
                  if int(r["name"].split("=")[1]) == x] for x in L]
            mu, sd = np.array([np.mean(a) for a in v]), np.array([np.std(a) for a in v])
            axs[2].plot(range(len(L)), mu, marker=mk, color=col, label=lab, ms=3)
            axs[2].fill_between(range(len(L)), mu - sd, mu + sd, color=col, alpha=.2)
        axs[2].axhline(.95, color="k", ls="--", lw=.8)
        axs[2].set_xticks(range(len(L))); axs[2].set_xticklabels(L)
        axs[2].set_xlabel("Stage-2 capacity (leaves per tree)")
        axs[2].set_ylabel("worst-class coverage"); axs[2].legend(fontsize=7, frameon=False)
        axs[2].set_title("Capacity abandons rare classes")
    fig.tight_layout()
    _save(fig, out, "fig6_sensitivity")


def main():
    res = sys.argv[1] if len(sys.argv) > 1 else "triage_results"
    sens_path = (sys.argv[2] if len(sys.argv) > 2
                 else os.path.join("triage_sensitivity", "sensitivity_results.json"))
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(res, "paper_figures")
    os.makedirs(out, exist_ok=True)
    R = json.load(open(os.path.join(res, "all_records.json")))
    if not any(r["dataset"] == "UNSW-NB15" for r in R):
        print("  no UNSW-NB15 records (smoke test?) -- only Fig. 3 is drawn")
        fig3_operating_curve(R, out)
        return
    fig2_risk_validity(R, out)
    fig3_operating_curve(R, out)
    fig4_mondrian(R, out)
    fig5_budget(R, out)
    if os.path.exists(sens_path):
        fig6_sensitivity(json.load(open(sens_path)), out)
    else:
        print(f"  (Fig. 6 skipped: {sens_path} not found)")


if __name__ == "__main__":
    main()
