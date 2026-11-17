# -*- coding: utf-8 -*-
"""
================================================================================
TRIAGE -- Stage-2 sensitivity study
================================================================================

Stage-2 sensitivity study. Answers two questions:

  (C4) "All experiments use LightGBM. How much do the reported results depend on
        that particular choice? Evaluate at least one or two alternative
        classifiers (e.g. XGBoost, CatBoost) or provide a sensitivity analysis."

  (C5) "Only the main LightGBM setting is reported. No sensitivity analysis with
        respect to parameters governing model complexity (max_depth, num_leaves),
        which can affect probability quality and therefore conformal set
        behaviour, escalation rate and Stage-2 performance."

DESIGN NOTE -- why this is fast.
Stage 1 and the risk-controlled threshold lambda_hat depend on the fast filter
ALONE. They are trained and calibrated once per seed and then held fixed while
Stage 2 is swept. This is not merely an optimisation: it means the escalation
rate is *structurally independent* of the Stage-2 choice, and the study
demonstrates that rather than assuming it. What can vary is the quality of the
Stage-2 posterior, and therefore the conformal sets, the deferral rate and the
end-to-end numbers.

RUN
    !pip -q install lightgbm xgboost catboost

    python stage2_sensitivity.py        (or the last section of the notebook)

The script prints a running ETA. It also RESUMES: every finished configuration
is written to disk immediately, and re-running skips whatever is already there.
So if Colab disconnects, just run the same cell again -- nothing is repeated.
One seed already answers both comments; three lets us report a mean and a range.

Output:  <sens_out_dir>/sensitivity_results.json
================================================================================
"""

from triage_ids import *            # noqa: F401,F403
from triage_ids import CONFIG
import os
import json
import time
import warnings
from typing import Dict, List, Any, Optional

import numpy as np

warnings.filterwarnings("ignore")


from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from lightgbm import LGBMClassifier

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False
try:
    from catboost import CatBoostClassifier
    _HAS_CAT = True
except Exception:
    _HAS_CAT = False

CONFIG.setdefault('sens_out_dir', 'triage_sensitivity')
CONFIG.setdefault('sens_seeds', [0, 1, 2])
# ==============================================================================
#  Stage-2 configurations under test
# ==============================================================================

def _lgbm(**kw):
    base = dict(n_estimators=CONFIG["s2_n_estimators"], max_depth=8,
                learning_rate=0.06, num_leaves=63, subsample=0.9,
                colsample_bytree=0.9, n_jobs=-1, verbose=-1)
    base.update(kw)
    return lambda seed, cw: LGBMClassifier(random_state=seed, class_weight=cw,
                                           **base)


def _xgb(**kw):
    base = dict(n_estimators=CONFIG["s2_n_estimators"], max_depth=8,
                learning_rate=0.06, subsample=0.9, colsample_bytree=0.9,
                n_jobs=-1, tree_method="hist", verbosity=0)
    base.update(kw)

    # XGBoost exposes no class_weight argument.  Rather than silently leaving
    # the binary head unbalanced -- which would confound "XGBoost is worse"
    # with "this head was not balanced" -- we tag the estimator and apply the
    # equivalent balanced sample weights at fit time.
    def _f(seed, cw):
        m = XGBClassifier(random_state=seed, **base)
        m._triage_needs_sw = (cw == "balanced")
        return m
    return _f


def _cat(**kw):
    base = dict(iterations=CONFIG["s2_n_estimators"], depth=8,
                learning_rate=0.06, verbose=0, allow_writing_files=False)
    base.update(kw)
    return lambda seed, cw: CatBoostClassifier(
        random_seed=seed,
        auto_class_weights="Balanced" if cw == "balanced" else None, **base)


def _rf(**kw):
    base = dict(n_estimators=300, max_depth=16, n_jobs=-1)
    base.update(kw)
    return lambda seed, cw: RandomForestClassifier(random_state=seed,
                                                   class_weight=cw, **base)


def _et(**kw):
    base = dict(n_estimators=300, max_depth=16, n_jobs=-1)
    base.update(kw)
    return lambda seed, cw: ExtraTreesClassifier(random_state=seed,
                                                 class_weight=cw, **base)


def build_configs() -> List[Dict[str, Any]]:
    """(C4) alternative classifiers, then (C5) complexity sweep."""
    cfgs: List[Dict[str, Any]] = []

    # ---- C4: classifier family ------------------------------------------
    cfgs.append(dict(group="classifier", name="LightGBM (paper default)",
                     factory=_lgbm()))
    if _HAS_XGB:
        cfgs.append(dict(group="classifier", name="XGBoost", factory=_xgb()))
    else:
        log("xgboost not installed -- skipping", 1)
    if _HAS_CAT:
        cfgs.append(dict(group="classifier", name="CatBoost", factory=_cat()))
    else:
        log("catboost not installed -- skipping", 1)
    cfgs.append(dict(group="classifier", name="Random forest", factory=_rf()))
    cfgs.append(dict(group="classifier", name="Extra trees", factory=_et()))

    # ---- C5: complexity -- depth ----------------------------------------
    for d in [3, 5, 8, 12, -1]:
        lbl = "unlimited" if d == -1 else str(d)
        cfgs.append(dict(group="max_depth", name=f"max_depth = {lbl}",
                         factory=_lgbm(max_depth=d)))

    # ---- C5: complexity -- leaves ---------------------------------------
    for nl in [15, 31, 63, 127, 255]:
        cfgs.append(dict(group="num_leaves", name=f"num_leaves = {nl}",
                         factory=_lgbm(num_leaves=nl, max_depth=-1)))

    # ---- C5: complexity -- capacity via n_estimators ---------------------
    for ne in [50, 150, 250, 500]:
        cfgs.append(dict(group="n_estimators", name=f"n_estimators = {ne}",
                         factory=_lgbm(n_estimators=ne)))

    return cfgs


# ==============================================================================
#  A Stage-2 that accepts an arbitrary estimator factory
# ==============================================================================

class SweptStage2:
    """Drop-in replacement for ``Stage2Analyzer`` whose estimator family and
    hyper-parameters are supplied by a factory.

    The two heads are weighted exactly as in the paper's Stage 2, so that the
    "LightGBM (paper default)" row of every sweep reproduces the main
    experiment rather than a near-miss of it:

      * binary head      -- class-balanced (the operational decision)
      * multi-class head -- ``CONFIG['s2_multi_class_weight']``
    """

    def __init__(self, factory, seed: int, n_classes: int):
        self.bin_model = factory(seed, "balanced")
        self.multi_model = factory(seed + 1000,
                                   CONFIG.get("s2_multi_class_weight", None))
        self.classes_: Optional[np.ndarray] = None
        self.fit_seconds = 0.0

    @staticmethod
    def _fit_one(model, X, y, balanced: bool):
        """Fit, applying balanced sample weights only where the estimator
        cannot express class balancing itself (XGBoost)."""
        if balanced and getattr(model, "_triage_needs_sw", False):
            from sklearn.utils.class_weight import compute_sample_weight
            model.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
        else:
            model.fit(X, y)

    def fit(self, X, y_bin, y_multi):
        t0 = time.perf_counter()
        # XGBoost requires integer-encoded, contiguous class labels
        self._lut = np.unique(y_multi)
        y_m = np.searchsorted(self._lut, y_multi)
        self._fit_one(self.bin_model, X, y_bin, True)
        self._fit_one(self.multi_model, X, y_m,
                      CONFIG.get("s2_multi_class_weight", None) == "balanced")
        self.classes_ = self._lut
        self.fit_seconds = time.perf_counter() - t0
        return self

    def predict_proba_bin(self, X):
        return np.asarray(self.bin_model.predict_proba(X))

    def predict_proba_multi(self, X):
        return np.asarray(self.multi_model.predict_proba(X))


# ==============================================================================
#  Evaluation of one configuration
# ==============================================================================

def eval_config(cfg, seed, Xtr, ytr_b, ytr_m, Xc2, yc2_m,
                Xte, yte_b, yte_m, class_names,
                u_test, lam, escalate, budget) -> Dict[str, Any]:
    n_classes = len(class_names)
    s2 = SweptStage2(cfg["factory"], seed, n_classes).fit(Xtr, ytr_b, ytr_m)

    # ---- conformal calibration, marginal and Mondrian -------------------
    p_cal = s2.predict_proba_multi(Xc2)
    classes = np.asarray(s2.classes_)
    sc = CONFIG.get("conformal_score", "lac")
    aps = APSConformal(CONFIG["alpha_aps"], False, False, seed, sc)
    aps.fit(p_cal, yc2_m, classes)
    apc = APSConformal(CONFIG["alpha_aps"], True, False, seed, sc)
    apc.fit(p_cal, yc2_m, classes)

    p_te = s2.predict_proba_multi(Xte)
    cls_pos = {int(c): i for i, c in enumerate(classes)}
    valid = np.array([int(v) in cls_pos for v in yte_m])
    y_idx = np.array([cls_pos.get(int(v), 0) for v in yte_m])
    lbl = [class_names[int(c)] for c in classes]

    sets_m = aps.predict_sets(p_te)
    sets_c = apc.predict_sets(p_te)
    cov_m = evaluate_conformal_sets(sets_m[valid], y_idx[valid], lbl)
    cov_c = evaluate_conformal_sets(sets_c[valid], y_idx[valid], lbl)

    def worst(cv):
        pc = {k: v["coverage"] for k, v in cv["per_class"].items()
              if v["n"] >= 20}
        return (min(pc.values()), min(pc, key=pc.get)) if pc else (float("nan"), "")

    wm, wm_name = worst(cov_m)
    wc, wc_name = worst(cov_c)

    # ---- Stage-2 quality in isolation ------------------------------------
    pb = s2.predict_proba_bin(Xte)[:, 1]
    m_all = core_metrics(yte_b, (pb > 0.5).astype(int), pb)
    top1 = float((classes[np.argmax(p_te, axis=1)] == yte_m)[valid].mean())

    # ---- end-to-end cascade, Stage 1 and lambda held fixed ---------------
    final = u_test["pred"].astype(np.int8).copy()
    score = u_test["score"].astype(float).copy()
    esc_idx = np.where(escalate)[0]
    set_size = np.ones(len(Xte), dtype=int)
    if len(esc_idx):
        pbe = s2.predict_proba_bin(Xte[esc_idx])[:, 1]
        final[esc_idx] = (pbe > 0.5).astype(np.int8)
        score[esc_idx] = pbe
        set_size[esc_idx] = sets_m[esc_idx].sum(axis=1)
    amb = np.zeros(len(Xte), bool)
    amb[esc_idx] = set_size[esc_idx] > 1
    amb_idx = np.where(amb)[0]
    pri = set_size[amb_idx] + u_test["u_epistemic"][amb_idx]
    deferred = amb_idx[np.argsort(-pri)][:budget]
    final[deferred] = yte_b[deferred]
    score[deferred] = yte_b[deferred].astype(float)
    m_casc = core_metrics(yte_b, final, score)

    return {
        "group": cfg["group"], "name": cfg["name"], "seed": seed,
        "fit_seconds": round(s2.fit_seconds, 2),
        "escalation_rate": float(escalate.mean()),
        "deferral_rate": float(len(deferred) / len(Xte)),
        "stage2_alone": {k: m_all[k] for k in
                         ["f1_attack", "fnr", "mcc", "roc_auc", "ece"]},
        "multiclass_top1": top1,
        "conformal_marginal": {
            "coverage": cov_m["marginal_coverage"],
            "mean_set_size": cov_m["mean_set_size"],
            "singleton_rate": cov_m["singleton_rate"],
            "worst_class_coverage": wm, "worst_class": wm_name},
        "conformal_mondrian": {
            "coverage": cov_c["marginal_coverage"],
            "mean_set_size": cov_c["mean_set_size"],
            "singleton_rate": cov_c["singleton_rate"],
            "worst_class_coverage": wc, "worst_class": wc_name},
        "cascade": {k: m_casc[k] for k in
                    ["f1_attack", "fnr", "mcc", "accuracy"]},
    }


# ==============================================================================
def sensitivity_main():
    _main_out = CONFIG["out_dir"]
    CONFIG["out_dir"] = CONFIG["sens_out_dir"]
    try:
        _sens_main()
    finally:
        CONFIG["out_dir"] = _main_out


def _sens_main():
    banner("TRIAGE -- Stage-2 sensitivity study")
    seeds = CONFIG.get("sens_seeds", [0, 1, 2])
    log(f"seeds={seeds}  xgboost={_HAS_XGB}  catboost={_HAS_CAT}")

    bundle = load_dataset("unsw", seed=0)
    log(str(bundle), 1)
    cfgs = build_configs()
    log(f"{len(cfgs)} configurations x {len(seeds)} seeds "
        f"= {len(cfgs)*len(seeds)} Stage-2 fits", 1)

    # Resume support: anything already recorded is not recomputed, so the run
    # can be interrupted and restarted without losing work.
    out: List[Dict[str, Any]] = []
    prev = None
    try:
        prev = load_json("sensitivity_results.json")
    except Exception:
        prev = None
    if isinstance(prev, list) and prev:
        out = prev
        log(f"resuming: {len(out)} records already on disk", 1)
    done = {(r["group"], r["name"], r["seed"]) for r in out}

    t_start = time.perf_counter()
    n_total = len(cfgs) * len(seeds)
    n_done = 0
    n_cached = 0

    for seed in seeds:
        banner(f"seed {seed}", "-")
        tr, cal1, cal2, te = make_splits(bundle, seed, "iid")
        pre = Preprocessor().fit(bundle.X.iloc[tr])
        Xtr = pre.transform(bundle.X.iloc[tr])
        Xc1 = pre.transform(bundle.X.iloc[cal1])
        Xc2 = pre.transform(bundle.X.iloc[cal2])
        Xte = pre.transform(bundle.X.iloc[te])
        ytr_b, ytr_m = bundle.y_bin[tr], bundle.y_multi[tr]
        yte_b, yte_m = bundle.y_bin[te], bundle.y_multi[te]

        # ---- Stage 1 and lambda: trained ONCE, then frozen ---------------
        s1 = Stage1Detector(seed).fit(Xtr, ytr_b)
        u1 = s1.predict_uncertainty(Xc1)
        miss = ((bundle.y_bin[cal1] == 1) & (u1["pred"] == 0)).astype(float)
        info = select_threshold_rcps(u1["u_total"], miss,
                                     CONFIG["delta_stage1"], CONFIG["gamma"])
        lam = info["lambda"]
        u_test = s1.predict_uncertainty(Xte)
        escalate = u_test["u_total"] > lam
        budget = int(len(te) * CONFIG["analyst_budget"])
        log(f"Stage 1 frozen: lambda={lam:.4f}  escalation={escalate.mean():.4f}"
            f"  budget={budget}", 1)

        for i, cfg in enumerate(cfgs, 1):
            if (cfg["group"], cfg["name"], seed) in done:
                n_done += 1; n_cached += 1
                log(f"[{i:>2}/{len(cfgs)}] {cfg['name']:<26} cached", 1)
                continue
            try:
                t0 = time.perf_counter()
                rec = eval_config(cfg, seed, Xtr, ytr_b, ytr_m, Xc2,
                                  bundle.y_multi[cal2], Xte, yte_b, yte_m,
                                  bundle.class_names, u_test, lam, escalate,
                                  budget)
                out.append(rec)
                cm, mo = rec["conformal_marginal"], rec["conformal_mondrian"]
                log(f"[{i:>2}/{len(cfgs)}] {cfg['name']:<26} "
                    f"F1={rec['cascade']['f1_attack']:.4f}  "
                    f"cov={cm['coverage']:.4f}/{mo['coverage']:.4f}  "
                    f"|C|={cm['mean_set_size']:.2f}/{mo['mean_set_size']:.2f}  "
                    f"worst={cm['worst_class_coverage']:.3f}/"
                    f"{mo['worst_class_coverage']:.3f}  "
                    f"({time.perf_counter()-t0:.0f}s)", 1)
                n_done += 1
                el = time.perf_counter() - t_start
                if n_done > n_cached and n_done < n_total:
                    eta = el / (n_done - n_cached) * (n_total - n_done)
                    log(f"     elapsed {el/60:.0f} min, "
                        f"~{eta/60:.0f} min remaining", 1)
            except Exception as e:
                import traceback
                log(f"[{i:>2}/{len(cfgs)}] {cfg['name']}: FAILED -- {e}", 1)
                traceback.print_exc()
            save_json(out, "sensitivity_results.json")

    # ------------------------------------------------------------ summary
    banner("SUMMARY (mean over seeds)")
    for grp in ["classifier", "max_depth", "num_leaves", "n_estimators"]:
        rows = [r for r in out if r["group"] == grp]
        if not rows:
            continue
        log(f"\n--- {grp} ---")
        log(f"  {'config':<26}{'casc F1':>9}{'esc':>8}{'cov_m':>8}{'cov_M':>8}"
            f"{'|C|_m':>7}{'|C|_M':>7}{'worst_m':>9}{'worst_M':>9}{'ECE':>8}")
        for name in dict.fromkeys(r["name"] for r in rows):
            g = [r for r in rows if r["name"] == name]
            f = lambda p, k: np.mean([r[p][k] for r in g])
            log(f"  {name:<26}{f('cascade','f1_attack'):>9.4f}"
                f"{np.mean([r['escalation_rate'] for r in g]):>8.4f}"
                f"{f('conformal_marginal','coverage'):>8.4f}"
                f"{f('conformal_mondrian','coverage'):>8.4f}"
                f"{f('conformal_marginal','mean_set_size'):>7.2f}"
                f"{f('conformal_mondrian','mean_set_size'):>7.2f}"
                f"{f('conformal_marginal','worst_class_coverage'):>9.3f}"
                f"{f('conformal_mondrian','worst_class_coverage'):>9.3f}"
                f"{f('stage2_alone','ece'):>8.4f}")

    save_json(out, "sensitivity_results.json")
    banner("DONE")
    log(f"{len(out)} records written to "
        f"{os.path.join(CONFIG['out_dir'], 'sensitivity_results.json')}")


if __name__ == "__main__":
    from triage_ids import prepare_data
    CONFIG["datasets"] = ["unsw"]
    prepare_data()
    sensitivity_main()
