# -*- coding: utf-8 -*-
"""
================================================================================
TRIAGE  --  Two-stage Risk-controlled Intrusion Analysis with Guaranteed
            Escalation
================================================================================

A conformally-calibrated, uncertainty-decomposed, human-in-the-loop cascade for
network intrusion detection.

Code accompanying the article "TRIAGE: Risk-Controlled Escalation for
Human-in-the-Loop Network Intrusion Detection".

--------------------------------------------------------------------------------
WHAT THIS IMPLEMENTS
--------------------------------------------------------------------------------
 1. Strict three-way split  train / CALIBRATION / test.  No threshold is ever
    tuned on test data.

 2. Uncertainty DECOMPOSITION.  Stage 1 is a probabilistic ensemble; predictive
    entropy is split into aleatoric (E_t H[p_t]) and epistemic (mutual
    information I(y;theta) = H[E_t p_t] - E_t H[p_t]) components, following
    Kendall & Gal (2017) and Huellermeier & Waegeman (2021).  The two components
    trigger different operational responses.

 3. DISTRIBUTION-FREE RISK CONTROL.  The Stage-1 escalation threshold is chosen
    by Risk-Controlling Prediction Sets / Learn-then-Test (Bates et al. JACM
    2021; Angelopoulos et al. AoAS 2025) using a Hoeffding-Bentkus upper
    confidence bound and fixed-sequence testing.  Guarantee:

        P( FNR among Stage-1 auto-cleared flows  <=  delta )  >=  1 - gamma

 4. ADAPTIVE PREDICTION SETS.  Stage 2 emits a conformal set of candidate attack
    classes with a coverage guarantee (Romano, Sesia & Candes, NeurIPS 2020),
    replacing the ungrounded "top-5" list.  Class-conditional (Mondrian)
    variant included (Sadinle et al., JASA 2019) because marginal coverage hides
    failure on rare attack classes.

 5. Deferral to human is a BUDGETED resource with a cost model, not a free
    action, and the simulated analyst is allowed to be WRONG (sensitivity
    analysis over analyst error rate).

 6. BATCHED inference.  No per-sample Python loop.

 7. Real BASELINES at matched escalation rates, multiple SEEDS, mean +/- std,
    and paired significance tests.

 8. MEASURED explainability (descriptive accuracy via deletion curves, and
    stability across seeds), per Warnecke et al. (EuroS&P 2020), instead of one
    illustrative bar chart.

 9. TWO datasets (UNSW-NB15 + CIC-IDS2017) through one unified loader, with the
    documented CIC-IDS2017 label errors acknowledged and the corrected-label
    path supported (Engelen et al., WTMC 2021).

10. EXCHANGEABILITY STUDY.  Conformal guarantees require calibration and test to
    be exchangeable.  The code runs both an i.i.d. pooled split (guarantee
    holds) and the official partition (guarantee expected to degrade).  The
    degradation is a RESULT to report, not a bug to hide.

--------------------------------------------------------------------------------
HOW TO RUN
--------------------------------------------------------------------------------
Notebook (Google Colab or Jupyter): open triage_ids.ipynb and run all cells.
Script:   pip install -r requirements.txt && python triage_ids.py
Both datasets are downloaded and verified automatically (see prepare_data).

Quick self-test with synthetic data (no downloads, ~1 minute):
    CONFIG['smoke_test'] = True

Everything is checkpointed to CONFIG['out_dir'].  If the run is interrupted,
run it again and it resumes.
================================================================================
"""

from __future__ import annotations

import os
import re
import gc
import io
import sys
import json
import time
import glob
import math
import pickle
import hashlib
import warnings
import platform
from typing import Dict, List, Tuple, Optional, Any, Sequence

import numpy as np
import pandas as pd

import matplotlib
if os.environ.get("TRIAGE_HEADLESS") or not os.environ.get("DISPLAY"):
    try:
        get_ipython  # noqa: F821  -- in a notebook, keep the default backend
    except NameError:
        matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import binom

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score, confusion_matrix,
    matthews_corrcoef, balanced_accuracy_score, classification_report,
)

import joblib

warnings.filterwarnings("ignore")

# Optional heavy deps -- degrade gracefully so the script still runs.
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False

try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="paper")
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False


# ==============================================================================
# CONFIGURATION
# ==============================================================================

CONFIG: Dict[str, Any] = {

    # ---- experiment scope -----------------------------------------------
    "smoke_test":        False,      # True -> synthetic data, tiny, ~1 min
    "datasets":          ["unsw", "cicids2017"],
    "seeds":             [0, 1, 2, 3, 4, 5],   # v2: six seeds. With n=3 the
    # smallest attainable two-sided Wilcoxon p-value is 0.25; n=6 reaches
    # p=0.031.
    "split_modes":       ["iid", "official"],   # 'official' only for unsw

    # ---- paths -----------------------------------------------------------
    "unsw_dir":          "data/UNSW-NB15",     # filled in by prepare_data()
    "cic_dir":           "data/CIC-IDS2017",   # filled in by prepare_data()
    "out_dir":           "triage_results",

    # ---- data handling ---------------------------------------------------
    "max_rows_per_dataset": 400_000,  # stratified subsample cap (Colab RAM)
    "cal_frac":          0.20,        # of the non-test pool
    "test_frac":         0.20,        # of everything (iid mode)
    "clip_quantile":     0.999,       # winsorise heavy-tailed features
    "drop_leaky_cic":    True,        # drop Flow ID / IPs / Timestamp

    # ---- risk-control targets (THE headline knobs) -----------------------
    "delta_stage1":      0.01,   # max FNR contributed by Stage-1 auto-clear
    "gamma":             0.10,   # 1-gamma confidence on the risk bound
    "alpha_aps":         0.05,   # 1-alpha coverage for Stage-2 class sets
    "analyst_budget":    0.02,   # fraction of flows a human can review

    # ---- Stage 1 (fast filter) -------------------------------------------
    "s1_n_estimators":   40,
    "s1_max_depth":      9,
    "s1_min_samples_leaf": 25,   # KEEP LEAVES IMPURE -> aleatoric term is real

    # ---- Stage 2 (deep analyzer) -----------------------------------------
    "s2_n_estimators":   250,
    "s2_max_depth":      8,
    "s2_learning_rate":  0.06,
    "use_smote":         False,  # class_weight is cheaper and less distorting
    "s2_multi_class_weight": "balanced",  # v3: REVERTED. Removing balancing in
    # v2 made the 15-class head overconfident (p_max ~ 1). APS cumulative sums
    # then pile up against 1.0, qhat lands in a numerically degenerate region,
    # every set collapses to a singleton, and coverage decays to plain top-1
    # accuracy. On CIC-IDS2017 that is ~0.745 -- the share of the Normal class.
    "aps_randomized":    False,      # v2: deterministic verdicts, consistent rule
    "conformal_score":   "lac",      # v3: "lac" (default) | "aps".  LAC/THR uses
    # s = 1 - p(y|x) -- no cumulative sum, so it does not degenerate when the
    # model is overconfident, and it provably yields the smallest average set
    # size (Sadinle et al., JASA 2019). APS is retained for the ablation.
    "lambda_grid_max":   1.0,        # v2: pre-specified LTT grid upper bound

    # ---- active learning --------------------------------------------------
    "al_rounds":         4,
    "al_budget_per_round": 1500,
    "analyst_error_rates": [0.0, 0.05, 0.10, 0.20],

    # ---- XAI --------------------------------------------------------------
    "shap_n_samples":    250,
    "deletion_steps":    12,

    # ---- misc -------------------------------------------------------------
    "latency_probe_n":   2000,
    "figure_dpi":        300,
    "verbose":           True,
}


# ==============================================================================
# 0.  UTILITIES
# ==============================================================================

def log(msg: str = "", level: int = 0) -> None:
    if CONFIG["verbose"]:
        print(("  " * level) + str(msg), flush=True)


def banner(title: str, char: str = "=") -> None:
    log()
    log(char * 78)
    log(title)
    log(char * 78)


class Timer:
    def __init__(self, name: str = ""):
        self.name = name
    def __enter__(self):
        self.t0 = time.perf_counter(); return self
    def __exit__(self, *a):
        self.dt = time.perf_counter() - self.t0
        if self.name:
            log(f"[{self.name}] {self.dt:.2f}s", 1)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def ckpt_path(name: str) -> str:
    return os.path.join(ensure_dir(CONFIG["out_dir"]), name)


def save_json(obj: Any, name: str) -> None:
    def _default(o):
        if isinstance(o, (np.integer,)):  return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.ndarray,)):  return o.tolist()
        if isinstance(o, (np.bool_,)):    return bool(o)
        return str(o)
    with open(ckpt_path(name), "w") as f:
        json.dump(obj, f, indent=2, default=_default)


def load_json(name: str) -> Optional[Any]:
    p = ckpt_path(name)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def entropy(p: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """Shannon entropy in bits."""
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log2(p), axis=axis)


# np.trapz was removed in NumPy 2.0 in favour of np.trapezoid.  Colab's NumPy
# version drifts, so bind whichever exists.
_TRAPZ = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def trapz(y, x):
    return float(_TRAPZ(y, x))


# ==============================================================================
# 1.  DATA LOADING  --  unified interface over UNSW-NB15 and CIC-IDS2017
# ==============================================================================

class DatasetBundle:
    """Container for one loaded, label-harmonised dataset."""

    def __init__(self, name: str, X: pd.DataFrame, y_bin: np.ndarray,
                 y_multi: np.ndarray, class_names: List[str],
                 official_split: Optional[np.ndarray] = None):
        self.name = name
        self.X = X
        self.y_bin = y_bin.astype(np.int8)
        self.y_multi = y_multi.astype(np.int16)
        self.class_names = class_names
        # official_split: 0 = official-train, 1 = official-test; None if N/A
        self.official_split = official_split

    def __repr__(self):
        return (f"<DatasetBundle {self.name}: {self.X.shape[0]:,} rows x "
                f"{self.X.shape[1]} feats, {len(self.class_names)} classes, "
                f"attack rate {self.y_bin.mean():.3f}>")


def _stratified_cap(X: pd.DataFrame, y_bin: np.ndarray, y_multi: np.ndarray,
                    cap: int, seed: int = 0,
                    extra: Optional[np.ndarray] = None):
    """Stratified subsample to `cap` rows, preserving multi-class proportions."""
    n = len(X)
    if n <= cap:
        return X, y_bin, y_multi, extra
    log(f"Subsampling {n:,} -> {cap:,} rows (stratified on attack class)", 1)
    idx = np.arange(n)
    keep, _ = train_test_split(
        idx, train_size=cap, random_state=seed,
        stratify=y_multi if len(np.unique(y_multi)) > 1 else None)
    keep = np.sort(keep)
    return (X.iloc[keep].reset_index(drop=True), y_bin[keep], y_multi[keep],
            None if extra is None else extra[keep])


# ------------------------------------------------------------------ UNSW-NB15

def load_unsw(seed: int = 0) -> DatasetBundle:
    """
    UNSW-NB15 (Moustafa & Slay, MilCIS 2015).

    NOTE ON THE PARTITION FILES.  The files are used by NAME: the file named
    UNSW_NB15_training-set.csv is the official TRAIN partition and the file
    named UNSW_NB15_testing-set.csv the official TEST partition.  In the
    distributed copies the training-set file holds 82,332 records and the
    testing-set file 175,341 (the reverse of the dataset documentation);
    prepare_data() verifies these counts.
    """
    d = CONFIG["unsw_dir"]
    f_train = os.path.join(d, "UNSW_NB15_training-set.csv")
    f_test = os.path.join(d, "UNSW_NB15_testing-set.csv")
    for f in (f_train, f_test):
        if not os.path.exists(f):
            raise FileNotFoundError(f"UNSW-NB15 file not found: {f}")

    df_a = pd.read_csv(f_train)
    df_b = pd.read_csv(f_test)
    log(f"UNSW official partitions: {df_a.shape} and {df_b.shape}", 1)

    df_a["_official"] = 0
    df_b["_official"] = 1
    df = pd.concat([df_a, df_b], ignore_index=True)

    df["attack_cat"] = (df["attack_cat"].fillna("Normal")
                        .astype(str).str.strip().replace({"": "Normal"}))
    # UNSW has a well-known "Backdoors"/"Backdoor" inconsistency
    df["attack_cat"] = df["attack_cat"].replace({"Backdoors": "Backdoor"})

    y_bin = df["label"].to_numpy()
    class_names = sorted(df["attack_cat"].unique().tolist())
    cmap = {c: i for i, c in enumerate(class_names)}
    y_multi = df["attack_cat"].map(cmap).to_numpy()
    official = df["_official"].to_numpy()

    X = df.drop(columns=[c for c in ["id", "label", "attack_cat", "_official"]
                         if c in df.columns])

    X, y_bin, y_multi, official = _stratified_cap(
        X, y_bin, y_multi, CONFIG["max_rows_per_dataset"], seed, official)

    return DatasetBundle("UNSW-NB15", X, y_bin, y_multi, class_names, official)


# --------------------------------------------------------------- CIC-IDS2017

_CIC_DROP = ["Flow ID", "Source IP", "Src IP", "Destination IP", "Dst IP",
             "Source Port", "Src Port", "Timestamp", "SimillarHTTP",
             "Unnamed: 0", "Fwd Header Length.1"]


def _clean_cic_label(s: str) -> str:
    s = str(s).strip()
    # v2: latin-1 decoding of the non-UTF-8 dash yields U+FFFD; strip every
    # variant so labels do not reach the paper as mojibake.
    for bad in ("\ufffd", "\x96", "\u2013", "\u2014", "\xc3\xa2"):
        s = s.replace(bad, "-")
    s = re.sub(r"\s*-\s*", " - ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if s.upper() == "BENIGN":
        return "Normal"
    return s


def load_cicids2017(seed: int = 0) -> DatasetBundle:
    """
    CIC-IDS2017 (Sharafaldin et al., ICISSP 2018).

    IMPORTANT: this dataset has documented label and feature-extraction errors
    (Engelen et al., WTMC 2021; Lanvin et al., CRiSIS 2022).  If you have the
    corrected CSVs from https://github.com/GintsEngelen/WTMC2021-Code, point
    CONFIG['cic_dir'] at them -- the loader is agnostic.  Say which version you
    used in the paper.
    """
    d = CONFIG["cic_dir"]
    files = sorted(glob.glob(os.path.join(d, "**", "*.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No CSVs under {d}")
    log(f"CIC-IDS2017: found {len(files)} CSV files", 1)

    frames = []
    for f in files:
        try:
            sub = pd.read_csv(f, low_memory=False, encoding="latin-1")
        except Exception as e:
            log(f"  skip {os.path.basename(f)}: {e}", 2)
            continue
        sub.columns = [c.strip() for c in sub.columns]
        frames.append(sub)
    df = pd.concat(frames, ignore_index=True)
    del frames; gc.collect()
    log(f"CIC-IDS2017 raw: {df.shape}", 1)

    label_col = next((c for c in df.columns if c.strip().lower() == "label"),
                     None)
    if label_col is None:
        raise ValueError("No 'Label' column found in CIC-IDS2017 CSVs")

    df[label_col] = df[label_col].map(_clean_cic_label)

    if CONFIG["drop_leaky_cic"]:
        drop = [c for c in _CIC_DROP if c in df.columns]
        if drop:
            log(f"Dropping identifier/leaky columns: {drop}", 1)
            df = df.drop(columns=drop)

    y_txt = df[label_col].to_numpy()
    X = df.drop(columns=[label_col])

    # Numeric coercion + inf/NaN handling (CICFlowMeter emits inf freely).
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    all_nan = X.columns[X.isna().all()].tolist()
    if all_nan:
        X = X.drop(columns=all_nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)

    # Drop zero-variance columns (CIC-IDS2017 has several constant columns).
    nun = X.nunique()
    const = nun[nun <= 1].index.tolist()
    if const:
        log(f"Dropping {len(const)} constant columns", 1)
        X = X.drop(columns=const)

    class_names = sorted(pd.unique(y_txt).tolist())
    cmap = {c: i for i, c in enumerate(class_names)}
    y_multi = np.array([cmap[v] for v in y_txt])
    y_bin = (y_txt != "Normal").astype(np.int8)

    X = X.astype(np.float32)
    X, y_bin, y_multi, _ = _stratified_cap(
        X, y_bin, y_multi, CONFIG["max_rows_per_dataset"], seed)

    return DatasetBundle("CIC-IDS2017", X, y_bin, y_multi, class_names, None)


# ------------------------------------------------------------------ synthetic

def load_synthetic(seed: int = 0, n: int = 40_000) -> DatasetBundle:
    """Synthetic UNSW-shaped data so the pipeline can be smoke-tested offline."""
    rng = np.random.default_rng(seed)
    n_feat = 30
    class_names = ["Normal", "DoS", "Exploits", "Fuzzers", "Generic",
                   "Reconnaissance", "Worms"]
    probs = np.array([0.45, 0.14, 0.16, 0.12, 0.09, 0.035, 0.005])
    y_multi = rng.choice(len(class_names), size=n, p=probs)

    # Overlapping class centres and heavy noise on purpose: if the synthetic
    # task is separable the cascade never escalates and the smoke test exercises
    # none of the interesting code paths.
    centres = rng.normal(0, 0.75, size=(len(class_names), n_feat))
    X = centres[y_multi] + rng.normal(0, 1.6, size=(n, n_feat))
    # a chunk of genuinely ambiguous flows (label noise -> aleatoric mass)
    amb = rng.random(n) < 0.12
    X[amb] = rng.normal(0, 1.6, size=(amb.sum(), n_feat))
    # heavy tail + a near-constant feature, mimicking real NIDS features
    X[:, 0] = np.exp(np.abs(X[:, 0]))
    X[:, 1] = (rng.random(n) < 0.002).astype(float) * rng.integers(1, 5, n)

    cols = [f"f{i}" for i in range(n_feat)]
    Xdf = pd.DataFrame(X.astype(np.float32), columns=cols)
    Xdf["proto"] = rng.choice(["tcp", "udp", "icmp", "rare"], n,
                              p=[.6, .3, .09, .01])
    Xdf["service"] = rng.choice(["-", "dns", "http", "ftp"], n,
                                p=[.5, .2, .25, .05])
    Xdf["state"] = rng.choice(["FIN", "CON", "INT"], n, p=[.6, .25, .15])

    y_bin = (y_multi != 0).astype(np.int8)
    official = (rng.random(n) < 0.35).astype(int)
    return DatasetBundle("SYNTHETIC", Xdf, y_bin, y_multi, class_names,
                         official)


def load_dataset(key: str, seed: int = 0) -> DatasetBundle:
    if CONFIG["smoke_test"]:
        return load_synthetic(seed)
    return {"unsw": load_unsw, "cicids2017": load_cicids2017}[key](seed)


# ==============================================================================
# 2.  PREPROCESSING
# ==============================================================================

class Preprocessor:
    """
    Fit on TRAIN ONLY.  Handles:
      * categorical columns with an explicit <UNK> bucket for unseen values
        (the old code mapped unseen -> -1 and then scaled it, silently creating
        out-of-distribution values)
      * winsorisation of heavy-tailed features before scaling (the old code let
        rare extreme values in `is_ftp_login` blow the scaler up)
      * standardisation
    """

    def __init__(self):
        self.cat_cols: List[str] = []
        self.num_cols: List[str] = []
        self.cat_maps: Dict[str, Dict[str, int]] = {}
        self.lo: Optional[np.ndarray] = None
        self.hi: Optional[np.ndarray] = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        self.cat_cols = [c for c in X.columns
                         if X[c].dtype == object or str(X[c].dtype) == "category"]
        self.num_cols = [c for c in X.columns if c not in self.cat_cols]

        for c in self.cat_cols:
            vals = X[c].astype(str).value_counts()
            keep = vals[vals >= 5].index.tolist()      # rare levels -> <UNK>
            self.cat_maps[c] = {v: i for i, v in enumerate(keep)}
            self.cat_maps[c]["<UNK>"] = len(keep)

        Xn = self._encode(X)
        q = CONFIG["clip_quantile"]
        self.lo = np.nanquantile(Xn.values, 1 - q, axis=0)
        self.hi = np.nanquantile(Xn.values, q, axis=0)
        Xc = np.clip(Xn.values, self.lo, self.hi)
        self.scaler.fit(Xc)
        self.feature_names = Xn.columns.tolist()
        return self

    def _encode(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for c in self.cat_cols:
            m = self.cat_maps[c]
            unk = m["<UNK>"]
            out[c] = out[c].astype(str).map(lambda v: m.get(v, unk)).astype(
                np.float32)
        for c in self.num_cols:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out[self.cat_cols + self.num_cols].astype(np.float32)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        Xn = self._encode(X)
        Xc = np.clip(Xn.values, self.lo, self.hi)
        return self.scaler.transform(Xc).astype(np.float32)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)


def make_splits(bundle: DatasetBundle, seed: int, mode: str):
    """
    Three-way split.

    mode='iid'      : pooled stratified split.  Calibration and test are
                      exchangeable, so the conformal guarantee is valid.
    mode='official' : the dataset's own train/test partition.  Exchangeability
                      is VIOLATED by construction.  Running this deliberately
                      and reporting the degradation is a contribution, not an
                      error.
    """
    n = len(bundle.X)
    idx = np.arange(n)
    strat = bundle.y_multi

    if mode == "official" and bundle.official_split is not None:
        te = idx[bundle.official_split == 1]
        pool = idx[bundle.official_split == 0]
    else:
        pool, te = train_test_split(
            idx, test_size=CONFIG["test_frac"], random_state=seed,
            stratify=strat)

    # guard: a class present in pool but with <2 members breaks stratification
    strat_pool = strat[pool]
    counts = pd.Series(strat_pool).value_counts()
    use_strat = strat_pool if (counts.min() >= 2) else None

    tr, cal = train_test_split(
        pool, test_size=CONFIG["cal_frac"], random_state=seed,
        stratify=use_strat)

    # Calibration is split in two disjoint halves: cal1 calibrates the Stage-1
    # threshold, cal2 calibrates the Stage-2 conformal sets.  Disjointness is
    # what keeps the two guarantees from interfering -- it removes the
    # multiplicity problem without needing a correction.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(cal))
    cal1 = cal[perm[: len(cal) // 2]]
    cal2 = cal[perm[len(cal) // 2:]]

    return np.sort(tr), np.sort(cal1), np.sort(cal2), np.sort(te)


# ==============================================================================
# 3.  MODELS
# ==============================================================================

class Stage1Detector:
    """
    Fast probabilistic filter.

    A Random Forest is used NOT because it is the strongest classifier but
    because it is (a) fast enough for line rate and (b) an ensemble, so the
    disagreement between its members gives a genuine epistemic signal.

    Critically, `min_samples_leaf` is kept large so that individual trees emit
    SOFT leaf distributions.  With fully-grown trees every member returns a
    near-degenerate 0/1 vector, the aleatoric term collapses to zero, and the
    decomposition becomes meaningless -- which is exactly the failure mode of
    naive `predict_proba` entropy.
    """

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.model = RandomForestClassifier(
            n_estimators=CONFIG["s1_n_estimators"],
            max_depth=CONFIG["s1_max_depth"],
            min_samples_leaf=CONFIG["s1_min_samples_leaf"],
            class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed, bootstrap=True,
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_uncertainty(self, X: np.ndarray,
                            batch: int = 100_000) -> Dict[str, np.ndarray]:
        """
        Returns mean probabilities and the entropy decomposition.

            total      H[ E_t p_t ]           predictive entropy
            aleatoric  E_t H[ p_t ]           irreducible / data ambiguity
            epistemic  total - aleatoric      = mutual information I(y; theta)

        (Huellermeier & Waegeman 2021, eq. 22; Kendall & Gal 2017.)
        """
        n = X.shape[0]
        K = len(self.model.classes_)
        trees = self.model.estimators_
        T = len(trees)

        mean_p = np.zeros((n, K), dtype=np.float64)
        mean_H = np.zeros(n, dtype=np.float64)

        for start in range(0, n, batch):
            sl = slice(start, min(start + batch, n))
            Xb = X[sl]
            acc_p = np.zeros((Xb.shape[0], K))
            acc_H = np.zeros(Xb.shape[0])
            for t in trees:
                pt = t.predict_proba(Xb)
                if pt.shape[1] != K:                # defensive
                    full = np.zeros((pt.shape[0], K)); full[:, :pt.shape[1]] = pt
                    pt = full
                acc_p += pt
                acc_H += entropy(pt, axis=1)
            mean_p[sl] = acc_p / T
            mean_H[sl] = acc_H / T

        total = entropy(mean_p, axis=1)
        aleatoric = mean_H
        epistemic = np.maximum(total - aleatoric, 0.0)

        return {
            "proba": mean_p,
            "pred": np.argmax(mean_p, axis=1).astype(np.int8),
            "score": mean_p[:, 1] if K == 2 else mean_p.max(axis=1),
            "u_total": total,
            "u_aleatoric": aleatoric,
            "u_epistemic": epistemic,
            "u_margin": 1.0 - mean_p.max(axis=1),   # softmax-response baseline
        }


def _make_gbm(seed: int, n_classes: int, class_weight: Optional[str] = "balanced"):
    """Prefer LightGBM, fall back to XGBoost, fall back to RF."""
    if _HAS_LGB:
        return LGBMClassifier(
            n_estimators=CONFIG["s2_n_estimators"],
            max_depth=CONFIG["s2_max_depth"],
            learning_rate=CONFIG["s2_learning_rate"],
            num_leaves=63, subsample=0.9, colsample_bytree=0.9,
            class_weight=class_weight, random_state=seed,
            n_jobs=-1, verbose=-1,
        )
    if _HAS_XGB:
        return XGBClassifier(
            n_estimators=CONFIG["s2_n_estimators"],
            max_depth=CONFIG["s2_max_depth"],
            learning_rate=CONFIG["s2_learning_rate"],
            subsample=0.9, colsample_bytree=0.9,
            objective="multi:softprob" if n_classes > 2 else "binary:logistic",
            num_class=n_classes if n_classes > 2 else None,
            random_state=seed, n_jobs=-1, tree_method="hist",
            eval_metric="mlogloss" if n_classes > 2 else "logloss",
        )
    return RandomForestClassifier(
        n_estimators=200, max_depth=14, class_weight=class_weight,
        n_jobs=-1, random_state=seed)


class Stage2Analyzer:
    """
    Deep analyzer.  Two heads:
      * binary  -- attack vs. normal, used for the operational decision
      * multi   -- attack category, wrapped in a conformal prediction set

    Kept as gradient-boosted trees rather than a neural network so the whole
    pipeline fits comfortably in Colab free-tier memory and TreeSHAP stays exact.
    """

    def __init__(self, seed: int = 0, n_classes: int = 2):
        self.seed = seed
        self.n_classes = n_classes
        # The binary head benefits from balancing.  The multi-class head must
        # NOT be balanced: its job is to emit a calibrated posterior for
        # conformal set construction, and balancing deliberately decalibrates
        # it.  On a 15-class problem with n=1 classes this flattens the
        # posterior badly enough to destroy top-1 accuracy on the majority
        # class.  (v2 fix; see Section "conformal anomaly".)
        self.bin_model = _make_gbm(seed, 2, class_weight="balanced")
        self.multi_model = _make_gbm(seed + 1000, n_classes,
                                     class_weight=CONFIG.get(
                                         "s2_multi_class_weight", None))
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X, y_bin, y_multi, sample_weight=None):
        if CONFIG["use_smote"]:
            try:
                from imblearn.over_sampling import SMOTE
                Xb, yb = SMOTE(random_state=self.seed).fit_resample(X, y_bin)
            except Exception as e:
                log(f"SMOTE unavailable/failed ({e}); using class weights", 1)
                Xb, yb = X, y_bin
            self.bin_model.fit(Xb, yb)
        else:
            self.bin_model.fit(X, y_bin, sample_weight=sample_weight)

        self.multi_model.fit(X, y_multi, sample_weight=sample_weight)
        self.classes_ = np.array(self.multi_model.classes_)
        return self

    def predict_proba_bin(self, X):
        return self.bin_model.predict_proba(X)

    def predict_proba_multi(self, X):
        return self.multi_model.predict_proba(X)


# ==============================================================================
# 4.  CONFORMAL CALIBRATION AND RISK CONTROL
# ==============================================================================

def hoeffding_bentkus_pvalue(r_hat: float, n: int, delta: float) -> float:
    """
    Hoeffding-Bentkus p-value for H0: R > delta, given empirical risk r_hat on
    n i.i.d. bounded-[0,1] losses.  Bates et al., JACM 68(6), 2021, Prop. 6.

    Smaller p  =>  stronger evidence that the true risk is at or below delta.
    """
    if n <= 0:
        return 1.0
    r_hat = float(np.clip(r_hat, 0.0, 1.0))
    delta = float(np.clip(delta, 1e-12, 1 - 1e-12))
    if r_hat >= delta:
        return 1.0

    # Hoeffding term via the KL / Chernoff bound
    def _kl(a, b):
        a = np.clip(a, 1e-12, 1 - 1e-12); b = np.clip(b, 1e-12, 1 - 1e-12)
        return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))
    p_hoef = float(np.exp(-n * _kl(r_hat, delta)))

    # Bentkus term
    p_bent = float(math.e * binom.cdf(math.ceil(n * r_hat), n, delta))

    return float(min(1.0, p_hoef, p_bent))


def select_threshold_rcps(scores: np.ndarray,
                          loss_indicator: np.ndarray,
                          delta: float,
                          gamma: float,
                          n_grid: int = 300) -> Dict[str, Any]:
    """
    Choose the largest escalation threshold lambda such that

        P( R(lambda) <= delta ) >= 1 - gamma

    where R(lambda) = E[ loss_indicator * 1{score <= lambda} ].

    `scores` is the Stage-1 uncertainty (escalate when score > lambda), and
    `loss_indicator` is 1 for calibration points that WOULD be a costly error if
    auto-cleared (attack predicted benign).  R is monotone non-decreasing in
    lambda, so we can use fixed-sequence testing along an ordered grid: walk
    from the most conservative lambda upward and stop at the first threshold
    whose null cannot be rejected.  Fixed-sequence testing controls the
    family-wise error rate at gamma with no multiplicity correction
    (Angelopoulos et al., 'Learn then Test', AoAS 2025).
    """
    n = len(scores)
    # v2: the grid must be PRE-SPECIFIED, not derived from the calibration
    # scores.  Learn-then-Test assumes a fixed candidate set Lambda; building it
    # from quantiles of the same data used to test is a (mild) violation of the
    # theory.  Binary Shannon entropy in bits is bounded on [0,1], and the other
    # supported signals are bounded likewise, so a fixed lattice is available at
    # no cost.
    hi_bound = float(CONFIG.get("lambda_grid_max", 1.0))
    grid = np.linspace(-1e-9, hi_bound + 1e-9, n_grid)

    trace = []
    lam_hat = grid[0]
    any_admissible = False
    for lam in grid:
        cleared = scores <= lam
        risk = float(np.mean(loss_indicator * cleared))   # unconditional risk
        p = hoeffding_bentkus_pvalue(risk, n, delta)
        trace.append((float(lam), risk, float(cleared.mean()), p))
        if p <= gamma:
            lam_hat = float(lam)     # null rejected -> lambda is admissible
            any_admissible = True
        else:
            break                    # fixed sequence: stop at first failure

    # If not even the most conservative threshold is certifiable, the target
    # delta is INFEASIBLE at this calibration-set size: a risk of delta simply
    # cannot be certified at confidence 1-gamma from n points, regardless of how
    # good the model is.  The procedure then escalates everything, which is the
    # correct conservative behaviour -- but you must report it as infeasibility
    # rather than as an operating point.  Roughly, certifying delta needs
    # n >~ log(1/gamma)/delta calibration samples.
    n_needed = int(np.ceil(math.log(1.0 / max(gamma, 1e-12)) / max(delta, 1e-12)))

    cleared = scores <= lam_hat
    return {
        "lambda": lam_hat,
        "cal_risk": float(np.mean(loss_indicator * cleared)),
        "cal_escalation_rate": float(1.0 - cleared.mean()),
        "n_cal": int(n),
        "delta": delta,
        "gamma": gamma,
        "feasible": bool(any_admissible),
        "approx_n_cal_needed": n_needed,
        "trace": trace,
    }


class APSConformal:
    """
    Adaptive Prediction Sets (Romano, Sesia & Candes, NeurIPS 2020).

    Produces class sets with marginal coverage >= 1 - alpha.  Set with
    class_conditional=True for the Mondrian / class-conditional variant
    (Sadinle et al., JASA 2019), which is what you must report for rare attack
    classes -- marginal coverage can look excellent while Worms is covered 40%
    of the time.
    """

    def __init__(self, alpha: float = 0.05, class_conditional: bool = False,
                 randomized: bool = False, seed: int = 0, score: str = "lac"):
        self.score = score   # v3: "lac" (numerically robust) or "aps"
        # NOTE (v2): `randomized` now defaults to False and is applied
        # CONSISTENTLY at calibration and prediction.  In v1 the calibration
        # score subtracted a uniform jitter term while the prediction rule did
        # not; that mismatch lowers qhat without widening the sets and causes
        # systematic UNDER-coverage.  It is the root cause of the degenerate
        # CIC-IDS2017 seed (marginal coverage 0.516, every set a singleton).
        self.alpha = alpha
        self.class_conditional = class_conditional
        self.randomized = randomized
        self.rng = np.random.default_rng(seed)
        self.qhat: Any = None
        self.classes_: Optional[np.ndarray] = None

    @staticmethod
    def _aps_scores(proba: np.ndarray, y: np.ndarray, rng, randomized: bool):
        order = np.argsort(-proba, axis=1)
        sorted_p = np.take_along_axis(proba, order, axis=1)
        cum = np.cumsum(sorted_p, axis=1)
        rank_of_true = np.argmax(order == y[:, None], axis=1)
        rows = np.arange(len(y))
        s = cum[rows, rank_of_true]
        if randomized:
            u = rng.random(len(y))
            s = s - u * sorted_p[rows, rank_of_true]
        return s

    @staticmethod
    def _lac_scores(proba: np.ndarray, y: np.ndarray):
        """LAC / THR nonconformity: s = 1 - p(true class). No cumulative sum,
        so overconfident posteriors cannot push every score against 1.0."""
        return 1.0 - proba[np.arange(len(y)), y]

    def fit(self, proba_cal: np.ndarray, y_cal: np.ndarray,
            classes: np.ndarray) -> "APSConformal":
        self.classes_ = np.asarray(classes)
        y_idx = np.searchsorted(self.classes_, y_cal)
        if self.score == "lac":
            s = self._lac_scores(proba_cal, y_idx)
        else:
            s = self._aps_scores(proba_cal, y_idx, self.rng, self.randomized)

        if not self.class_conditional:
            n = len(s)
            lvl = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / max(n, 1))
            self.qhat = float(np.quantile(s, lvl, method="higher"))
        else:
            self.qhat = {}
            for k in range(len(self.classes_)):
                m = (y_idx == k)
                nk = int(m.sum())
                if nk < 10:                    # too few to calibrate: fall back
                    self.qhat[k] = None
                    continue
                lvl = min(1.0, np.ceil((nk + 1) * (1 - self.alpha)) / nk)
                self.qhat[k] = float(np.quantile(s[m], lvl, method="higher"))
            valid = [v for v in self.qhat.values() if v is not None]
            self._fallback_q = float(np.mean(valid)) if valid else 1.0
        return self

    def predict_sets(self, proba: np.ndarray) -> np.ndarray:
        """
        Boolean membership matrix (n, n_classes).

        Classes are added in descending probability order until the cumulative
        mass first reaches the calibrated threshold; the class that crosses is
        included.  If the top class alone already reaches the threshold the set
        is a SINGLETON -- which is the whole point, because singleton vs.
        non-singleton is what drives the deferral decision.  (The non-randomised
        rule is used at prediction time; it is conservative, so coverage is
        still at least 1 - alpha.)
        """
        n, K = proba.shape
        if self.score == "lac":
            # include every class whose probability clears the calibrated level
            if not self.class_conditional:
                keep = (1.0 - proba) <= self.qhat
            else:
                qv = np.array([self.qhat.get(k) if self.qhat.get(k) is not None
                               else self._fallback_q for k in range(K)], float)
                keep = (1.0 - proba) <= qv[None, :]
            # never emit an empty set: fall back to the arg-max
            empty = ~keep.any(axis=1)
            if empty.any():
                keep[empty, np.argmax(proba[empty], axis=1)] = True
            return keep
        order = np.argsort(-proba, axis=1)
        sorted_p = np.take_along_axis(proba, order, axis=1)
        cum = np.cumsum(sorted_p, axis=1)

        if not self.class_conditional:
            thr = np.full((n, K), float(self.qhat))
        else:
            qv = np.array([self.qhat.get(k) if self.qhat.get(k) is not None
                           else self._fallback_q for k in range(K)],
                          dtype=float)
            thr = qv[order]                    # threshold follows the class

        if self.randomized:
            # must mirror _aps_scores exactly, or coverage is not guaranteed
            u = self.rng.random((n, 1))
            reached = (cum - u * sorted_p) >= thr
        else:
            reached = cum >= thr
        any_reached = reached.any(axis=1)
        crossed = np.where(any_reached, np.argmax(reached, axis=1), K - 1)

        ranks = np.arange(K)[None, :]
        keep_sorted = ranks <= crossed[:, None]

        out = np.zeros((n, K), dtype=bool)
        np.put_along_axis(out, order, keep_sorted, axis=1)
        return out


def evaluate_conformal_sets(sets: np.ndarray, y_true_idx: np.ndarray,
                            class_names: List[str]) -> Dict[str, Any]:
    rows = np.arange(len(y_true_idx))
    covered = sets[rows, y_true_idx]
    sizes = sets.sum(axis=1)
    per_class = {}
    for k, name in enumerate(class_names):
        m = y_true_idx == k
        if m.sum() == 0:
            continue
        per_class[name] = {
            "n": int(m.sum()),
            "coverage": float(covered[m].mean()),
            "mean_set_size": float(sizes[m].mean()),
        }
    return {
        "marginal_coverage": float(covered.mean()),
        "mean_set_size": float(sizes.mean()),
        "singleton_rate": float((sizes == 1).mean()),
        "per_class": per_class,
    }


# ==============================================================================
# 5.  THE TRIAGE CASCADE
# ==============================================================================

class TriageCascade:
    """
    Tier 1  Fast probabilistic filter.  Auto-clears flows whose Stage-1
            uncertainty is below a conformally-chosen threshold lambda1, with a
            distribution-free bound on the false negatives this creates.

    Tier 2  Deep analyzer on escalated flows.  Emits a binary decision plus a
            conformal set of candidate attack classes.

    Tier 3  Human analyst.  Receives flows whose Stage-2 conformal set is
            ambiguous (|C(x)| > 1), subject to a hard budget.  When the budget
            is exhausted the system degrades to Stage-2's point prediction, and
            this is recorded rather than hidden.
    """

    def __init__(self, seed: int = 0, n_classes: int = 2):
        self.seed = seed
        self.n_classes = n_classes
        self.stage1 = Stage1Detector(seed)
        self.stage2 = Stage2Analyzer(seed, n_classes)
        self.lambda1: Optional[float] = None
        self.rcps_info: Dict[str, Any] = {}
        self.aps: Optional[APSConformal] = None
        self.aps_cc: Optional[APSConformal] = None
        self.s1_signal = "u_total"

    # -------------------------------------------------------------- training
    def fit(self, Xtr, ytr_bin, ytr_multi):
        with Timer("Stage 1 fit"):
            self.stage1.fit(Xtr, ytr_bin)
        with Timer("Stage 2 fit"):
            self.stage2.fit(Xtr, ytr_bin, ytr_multi)
        return self

    # ----------------------------------------------------------- calibration
    def calibrate(self, Xcal1, ycal1_bin, Xcal2, ycal2_multi,
                  signal: str = "u_total"):
        """
        cal1 -> Stage-1 risk-controlled escalation threshold (RCPS / LTT)
        cal2 -> Stage-2 conformal prediction sets (APS, marginal + Mondrian)

        The two calibration sets are DISJOINT.  That is what makes the two
        guarantees simultaneously valid without a multiplicity correction.
        """
        self.s1_signal = signal

        u1 = self.stage1.predict_uncertainty(Xcal1)
        scores = u1[signal]
        # loss: an attack that Stage 1 would call benign
        miss = ((ycal1_bin == 1) & (u1["pred"] == 0)).astype(float)

        # retained so the operating point can be re-derived for any delta
        # without re-scoring the calibration set
        self._cal_scores = scores
        self._cal_miss = miss

        self.rcps_info = select_threshold_rcps(
            scores, miss, CONFIG["delta_stage1"], CONFIG["gamma"])
        self.lambda1 = self.rcps_info["lambda"]
        log(f"RCPS: lambda1={self.lambda1:.4f}  "
            f"cal risk={self.rcps_info['cal_risk']:.5f} "
            f"(target delta={CONFIG['delta_stage1']})  "
            f"cal escalation={self.rcps_info['cal_escalation_rate']:.3%}", 1)
        if not self.rcps_info["feasible"]:
            log(f"  !! delta={CONFIG['delta_stage1']} is INFEASIBLE with "
                f"{self.rcps_info['n_cal']:,} calibration points at "
                f"confidence {1 - CONFIG['gamma']:.0%}. You need roughly "
                f"{self.rcps_info['approx_n_cal_needed']:,}. The procedure is "
                f"escalating everything, which is correct but not a useful "
                f"operating point -- raise delta or enlarge the calibration "
                f"set.", 1)

        p2 = self.stage2.predict_proba_multi(Xcal2)
        classes = self.stage2.classes_
        rnd = CONFIG.get("aps_randomized", False)
        sc = CONFIG.get("conformal_score", "lac")
        self.aps = APSConformal(CONFIG["alpha_aps"], False, rnd,
                                self.seed, sc).fit(p2, ycal2_multi, classes)
        self.aps_cc = APSConformal(CONFIG["alpha_aps"], True, rnd,
                                   self.seed, sc).fit(p2, ycal2_multi, classes)
        log(f"APS calibrated: qhat={self.aps.qhat:.4f} "
            f"(alpha={CONFIG['alpha_aps']})", 1)
        return self

    def lambda_for_delta(self, delta: float) -> Dict[str, Any]:
        """
        Re-derive the escalation threshold for a different risk target using the
        stored calibration scores.  This is what produces the operating curve
        (delta -> escalation rate -> cost) that lets an operator choose a
        working point instead of accepting one hard-coded number.
        """
        return select_threshold_rcps(self._cal_scores, self._cal_miss,
                                     delta, CONFIG["gamma"])

    # ------------------------------------------------------------- inference
    def run(self, X: np.ndarray, y_bin: np.ndarray, y_multi: np.ndarray,
            analyst_budget: Optional[float] = None,
            analyst_error: float = 0.0,
            u1: Optional[Dict[str, np.ndarray]] = None,
            lambda1: Optional[float] = None) -> Dict[str, Any]:
        """
        Fully batched.  Three vectorised passes, no Python loop over samples.

        `u1` may be passed in to avoid re-scoring Stage 1 when sweeping budgets,
        analyst error rates, or risk targets over the same test set.
        """
        n = X.shape[0]
        budget = int(n * (CONFIG["analyst_budget"] if analyst_budget is None
                          else analyst_budget))
        lam = self.lambda1 if lambda1 is None else lambda1
        # v2: derive the stream from (seed, epsilon) so that the eps-sweep does
        # not corrupt overlapping flow sets, which made the curve artificially
        # smooth in v1.
        rng = np.random.default_rng(
            (self.seed + 1) * 100003 + int(round(analyst_error * 1e6)))

        # ---- Tier 1 -------------------------------------------------------
        t0 = time.perf_counter()
        if u1 is None:
            u1 = self.stage1.predict_uncertainty(X)
        t_s1 = time.perf_counter() - t0

        escalate = u1[self.s1_signal] > lam
        esc_idx = np.where(escalate)[0]

        final_bin = u1["pred"].astype(np.int8).copy()
        final_score = u1["score"].astype(np.float64).copy()
        tier = np.ones(n, dtype=np.int8)          # 1 / 2 / 3
        set_size = np.ones(n, dtype=np.int16)
        cand_sets = np.zeros((n, self.n_classes), dtype=bool)
        cand_sets[np.arange(n), 0] = True

        # ---- Tier 2 -------------------------------------------------------
        t0 = time.perf_counter()
        if len(esc_idx) > 0:
            Xe = X[esc_idx]
            pb = self.stage2.predict_proba_bin(Xe)
            pm = self.stage2.predict_proba_multi(Xe)
            sets = self.aps.predict_sets(pm)

            final_bin[esc_idx] = (pb[:, 1] > 0.5).astype(np.int8)
            final_score[esc_idx] = pb[:, 1]
            tier[esc_idx] = 2
            set_size[esc_idx] = sets.sum(axis=1)
            cand_sets[esc_idx] = sets
        t_s2 = time.perf_counter() - t0

        # ---- Tier 3: defer ambiguous conformal sets, under budget ---------
        ambiguous = np.zeros(n, dtype=bool)
        if len(esc_idx) > 0:
            ambiguous[esc_idx] = set_size[esc_idx] > 1

        amb_idx = np.where(ambiguous)[0]
        # priority: largest conformal set first, tie-broken by epistemic
        # uncertainty -- flows the model has least evidence about go to a human
        # before flows that are merely ambiguous.
        if len(amb_idx) > 0:
            pri = (set_size[amb_idx].astype(float)
                   + u1["u_epistemic"][amb_idx])
            order = amb_idx[np.argsort(-pri)]
            deferred = order[:budget]
            overflow = order[budget:]
        else:
            deferred = np.array([], dtype=int)
            overflow = np.array([], dtype=int)

        if len(deferred) > 0:
            truth = y_bin[deferred]
            if analyst_error > 0:
                flip = rng.random(len(deferred)) < analyst_error
                truth = np.where(flip, 1 - truth, truth)
            final_bin[deferred] = truth
            final_score[deferred] = truth.astype(float)
            tier[deferred] = 3

        return {
            "final_bin": final_bin,
            "final_score": final_score,
            "tier": tier,
            "u": u1,
            "escalated": esc_idx,
            "deferred": deferred,
            "overflow": overflow,
            "set_size": set_size,
            "cand_sets": cand_sets,
            "budget": budget,
            "timing": {"stage1_s": t_s1, "stage2_s": t_s2,
                       "total_s": t_s1 + t_s2,
                       "throughput_fps": n / max(t_s1 + t_s2, 1e-9)},
        }


# ==============================================================================
# 6.  BASELINES  (all compared at MATCHED escalation rates)
# ==============================================================================

def cascade_with_fixed_rate(stage1: Stage1Detector, stage2: Stage2Analyzer,
                            X, u1: Dict[str, np.ndarray], signal: str,
                            rate: float, seed: int = 0) -> np.ndarray:
    """
    Generic cascade: escalate the top `rate` fraction by `signal`, take Stage 2's
    answer there and Stage 1's elsewhere.  This is how every prior two-stage IDS
    picks its threshold -- by choosing a rate, not by controlling a risk.
    """
    n = X.shape[0]
    k = int(round(rate * n))
    if signal == "random":
        rng = np.random.default_rng(seed)
        s = rng.random(n)
    else:
        s = u1[signal]
    esc = np.argsort(-s)[:k]

    pred = u1["pred"].astype(np.int8).copy()
    score = u1["score"].astype(np.float64).copy()
    if k > 0:
        pb = stage2.predict_proba_bin(X[esc])
        pred[esc] = (pb[:, 1] > 0.5).astype(np.int8)
        score[esc] = pb[:, 1]
    return pred, score, esc


def risk_coverage_curve(scores_unc: np.ndarray, correct: np.ndarray,
                        n_points: int = 200):
    """
    Selective risk vs. coverage (Geifman & El-Yaniv, NeurIPS 2017).
    Sort by increasing uncertainty; at coverage c, accept the c most certain
    fraction and measure error among them.  AURC = area under this curve
    (lower is better).
    """
    order = np.argsort(scores_unc)
    err = (~correct.astype(bool))[order].astype(float)
    cum_err = np.cumsum(err)
    ks = np.linspace(1, len(err), n_points).astype(int)
    cov = ks / len(err)
    risk = cum_err[ks - 1] / ks
    aurc = trapz(risk, cov)
    return cov, risk, aurc


# ==============================================================================
# 7.  METRICS
# ==============================================================================

def expected_calibration_error(y: np.ndarray, p: np.ndarray,
                               n_bins: int = 15) -> float:
    conf = np.maximum(p, 1 - p)
    pred = (p > 0.5).astype(int)
    acc = (pred == y).astype(float)
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.mean()) * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def core_metrics(y: np.ndarray, pred: np.ndarray,
                 score: Optional[np.ndarray] = None) -> Dict[str, float]:
    out = {
        "accuracy":     float(accuracy_score(y, pred)),
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
        "f1_attack":    float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "f1_macro":     float(f1_score(y, pred, average="macro",
                                       zero_division=0)),
        "mcc":          float(matthews_corrcoef(y, pred)),
    }
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out.update({
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "fpr": float(fp / max(fp + tn, 1)),
        "fnr": float(fn / max(fn + tp, 1)),
        "precision": float(tp / max(tp + fp, 1)),
        "recall":    float(tp / max(tp + fn, 1)),
    })
    if score is not None and len(np.unique(y)) > 1:
        out["roc_auc"] = float(roc_auc_score(y, score))
        out["pr_auc"] = float(average_precision_score(y, score))
        out["ece"] = expected_calibration_error(y, np.clip(score, 0, 1))
    return out


def operational_cost(m: Dict[str, float], n_reviewed: int,
                     c_fp: float = 1.0, c_fn: float = 50.0,
                     c_human: float = 5.0) -> float:
    """
    Asymmetric cost.  c_fn >> c_fp because a missed intrusion is far more
    expensive than a wasted investigation; c_human prices analyst time.  Sweep
    the ratio in the paper rather than defending one number.
    """
    return c_fp * m["fp"] + c_fn * m["fn"] + c_human * n_reviewed


def measure_latency(stage1: Stage1Detector, X: np.ndarray,
                    n: int = 2000, micro_batch: int = 256) -> Dict[str, float]:
    """
    Latency for Stage 1 (the component that must keep up with line rate).
    Reported in the style of Kitsune (NDSS 2018): batch throughput alone is not
    a latency claim.

    THREE numbers, because only reporting the flattering one is exactly the kind
    of thing reviewers catch:

      * single_flow_*   one flow at a time through scikit-learn.  This is
                        dominated by Python/NumPy call overhead, not by the
                        model, and is the pessimistic bound.
      * microbatch_*    amortised per-flow latency at a realistic micro-batch
                        size, which is how a real collector feeds a detector.
      * batched_*       full-batch throughput, the optimistic bound.

    Be explicit in the paper that a production deployment would compile the
    forest (Treelite / ONNX Runtime / a C++ inference path); the Python numbers
    characterise the algorithm's cost, not the achievable ceiling.
    """
    n = min(n, len(X))
    sub = X[:n]

    single = np.empty(min(n, 500))
    for i in range(len(single)):
        t0 = time.perf_counter()
        stage1.model.predict_proba(sub[i:i + 1])
        single[i] = (time.perf_counter() - t0) * 1e3          # ms

    mb = []
    for start in range(0, n - micro_batch, micro_batch):
        t0 = time.perf_counter()
        stage1.model.predict_proba(sub[start:start + micro_batch])
        mb.append((time.perf_counter() - t0) * 1e3 / micro_batch)
    mb = np.array(mb) if mb else np.array([np.nan])

    t0 = time.perf_counter()
    stage1.model.predict_proba(sub)
    batch_s = time.perf_counter() - t0

    return {
        "single_flow_p50_ms": float(np.percentile(single, 50)),
        "single_flow_p95_ms": float(np.percentile(single, 95)),
        "single_flow_p99_ms": float(np.percentile(single, 99)),
        "microbatch_size": int(micro_batch),
        "microbatch_p50_ms_per_flow": float(np.nanpercentile(mb, 50)),
        "microbatch_p99_ms_per_flow": float(np.nanpercentile(mb, 99)),
        "microbatch_throughput_fps": float(
            1000.0 / max(np.nanmedian(mb), 1e-9)),
        "batched_throughput_fps": float(n / max(batch_s, 1e-9)),
    }


# ==============================================================================
# 8.  ACTIVE LEARNING  (with a replay buffer -- no catastrophic forgetting)
# ==============================================================================

def active_learning_study(Xtr, ytr_bin, ytr_multi, Xpool, ypool_bin,
                          ypool_multi, Xte, yte_bin, yte_multi,
                          n_classes: int, seed: int,
                          uncertainty_pool: np.ndarray) -> Dict[str, Any]:
    """
    Multi-round active learning, evaluated on held-out test after EVERY round,
    against two baselines:
        * random sampling with the same budget
        * no retraining at all

    Critically, each round retrains on ORIGINAL TRAIN + all acquired samples
    (a replay buffer), not on the acquired samples alone.  Retraining on the
    hard-negative buffer alone is the catastrophic-forgetting failure mode
    documented by Chen et al. (USENIX Security 2023).
    """
    rng = np.random.default_rng(seed)
    B = CONFIG["al_budget_per_round"]
    R = CONFIG["al_rounds"]
    results = {"uncertainty": [], "random": [], "none": []}

    base = Stage2Analyzer(seed, n_classes).fit(Xtr, ytr_bin, ytr_multi)
    p0 = base.predict_proba_bin(Xte)[:, 1]
    m0 = core_metrics(yte_bin, (p0 > 0.5).astype(int), p0)
    for k in ("uncertainty", "random"):
        results[k].append({"round": 0, "n_labels": 0, **m0})
    # "none" = never retrain; flat reference line across the whole budget axis
    for r in range(R + 1):
        results["none"].append({"round": r, "n_labels": int(r * B), **m0})

    for strategy in ("uncertainty", "random"):
        acc_idx = np.array([], dtype=int)
        remaining = np.arange(len(Xpool))
        for r in range(1, R + 1):
            if len(remaining) == 0:
                break
            take = min(B, len(remaining))
            if strategy == "uncertainty":
                pick = remaining[np.argsort(-uncertainty_pool[remaining])[:take]]
            else:
                pick = rng.choice(remaining, size=take, replace=False)
            acc_idx = np.concatenate([acc_idx, pick])
            remaining = np.setdiff1d(remaining, pick)

            # replay buffer: original training data + everything acquired
            Xa = np.vstack([Xtr, Xpool[acc_idx]])
            ya_b = np.concatenate([ytr_bin, ypool_bin[acc_idx]])
            ya_m = np.concatenate([ytr_multi, ypool_multi[acc_idx]])
            # upweight acquired hard samples, but do not let them dominate
            w = np.concatenate([np.ones(len(Xtr)),
                                np.full(len(acc_idx), 3.0)])

            mdl = Stage2Analyzer(seed + r, n_classes).fit(Xa, ya_b, ya_m,
                                                          sample_weight=w)
            p = mdl.predict_proba_bin(Xte)[:, 1]
            m = core_metrics(yte_bin, (p > 0.5).astype(int), p)
            results[strategy].append({"round": r, "n_labels": int(len(acc_idx)),
                                      **m})
            log(f"AL[{strategy}] round {r}: n={len(acc_idx)} "
                f"F1={m['f1_attack']:.4f} FNR={m['fnr']:.4f}", 1)

    return results


# ==============================================================================
# 9.  EXPLAINABILITY  --  measured, not illustrated
# ==============================================================================

def shap_deletion_curve(model, X_bg: np.ndarray, X_sub: np.ndarray,
                        feature_names: List[str],
                        steps: int = 12) -> Dict[str, Any]:
    """
    Descriptive accuracy (Warnecke et al., EuroS&P 2020).

    Progressively replace the k highest-|SHAP| features with the training
    median and measure the drop in the predicted probability of the originally
    predicted class.  A faithful explanation causes a steep drop.  Compared
    against a random-ordering control -- without that control the curve means
    nothing.
    """
    if not _HAS_SHAP:
        return {"available": False,
                "reason": "shap not installed"}

    try:
        expl = shap.TreeExplainer(model)
        sv = expl.shap_values(X_sub)
    except Exception as e:
        return {"available": False, "reason": str(e)}

    p_full = model.predict_proba(X_sub)
    cls = np.argmax(p_full, axis=1)
    rows = np.arange(len(X_sub))
    p0 = p_full[rows, cls]

    # normalise SHAP output shape across shap/lightgbm/xgboost versions
    if isinstance(sv, list):
        contrib = np.stack([sv[c][i] for i, c in enumerate(cls)])
    elif np.ndim(sv) == 3:
        contrib = np.stack([sv[i, :, c] for i, c in enumerate(cls)])
    else:
        contrib = np.asarray(sv)

    med = np.median(X_bg, axis=0)
    order_shap = np.argsort(-np.abs(contrib), axis=1)
    rng = np.random.default_rng(0)
    order_rand = np.stack([rng.permutation(X_sub.shape[1])
                           for _ in range(len(X_sub))])

    ks = np.unique(np.linspace(0, min(steps, X_sub.shape[1]),
                               steps + 1).astype(int))
    curves = {}
    for tag, order in (("shap", order_shap), ("random", order_rand)):
        vals = []
        for k in ks:
            Xm = X_sub.copy()
            if k > 0:
                idx = order[:, :k]
                for j in range(k):
                    Xm[rows, idx[:, j]] = med[idx[:, j]]
            pk = model.predict_proba(Xm)[rows, cls]
            vals.append(float(np.mean(pk / np.maximum(p0, 1e-9))))
        curves[tag] = vals

    auc_shap = trapz(curves["shap"], ks) / max(int(ks[-1]), 1)
    auc_rand = trapz(curves["random"], ks) / max(int(ks[-1]), 1)

    # global ranking for the cross-seed stability check
    global_rank = np.argsort(-np.abs(contrib).mean(axis=0))

    return {
        "available": True,
        "k": ks.tolist(),
        "curve_shap": curves["shap"],
        "curve_random": curves["random"],
        "auc_shap": auc_shap,
        "auc_random": auc_rand,
        # lower AUC = faster probability collapse = more faithful ranking
        "faithfulness_gain": float(auc_rand - auc_shap),
        "global_top20": [feature_names[i] for i in global_rank[:20]],
        "global_rank": global_rank.tolist(),
    }


def explanation_stability(rank_lists: List[List[int]]) -> Dict[str, float]:
    """Mean pairwise Spearman correlation of global SHAP rankings across seeds."""
    if len(rank_lists) < 2:
        return {"mean_spearman": float("nan"), "n_pairs": 0}
    rhos = []
    for i in range(len(rank_lists)):
        for j in range(i + 1, len(rank_lists)):
            a, b = rank_lists[i], rank_lists[j]
            L = min(len(a), len(b))
            pa = np.empty(L); pa[np.array(a[:L])] = np.arange(L)
            pb = np.empty(L); pb[np.array(b[:L])] = np.arange(L)
            rhos.append(stats.spearmanr(pa, pb).correlation)
    return {"mean_spearman": float(np.nanmean(rhos)),
            "std_spearman": float(np.nanstd(rhos)),
            "n_pairs": len(rhos)}


# ==============================================================================
# 10.  FIGURES
# ==============================================================================

def _save(fig, name: str):
    p = ckpt_path(name)
    fig.savefig(p, dpi=CONFIG["figure_dpi"], bbox_inches="tight")
    fig.savefig(p.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"figure -> {os.path.basename(p)}", 1)


def fig_risk_coverage(curves: Dict[str, Tuple], dataset: str):
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for name, (cov, risk, aurc) in curves.items():
        ax.plot(cov, risk, label=f"{name} (AURC={aurc:.4f})", linewidth=1.8)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective risk (error rate on accepted flows)")
    ax.set_title(f"Risk–coverage trade-off — {dataset}")
    ax.legend(fontsize=8, frameon=True)
    ax.grid(alpha=.3)
    _save(fig, f"fig_risk_coverage_{dataset}.pdf")


def fig_uncertainty_decomposition(u: Dict[str, np.ndarray], y: np.ndarray,
                                  dataset: str):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, key, title in zip(
            axes, ["u_total", "u_aleatoric", "u_epistemic"],
            ["Total predictive entropy", "Aleatoric (data ambiguity)",
             "Epistemic (model ignorance)"]):
        for lab, name, c in ((0, "Normal", "#4c72b0"), (1, "Attack", "#c44e52")):
            v = u[key][y == lab]
            if len(v) == 0:
                continue
            ax.hist(v, bins=60, alpha=.55, density=True, label=name, color=c)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("bits")
        ax.grid(alpha=.25)
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Uncertainty decomposition at Stage 1 — {dataset}",
                 fontsize=11)
    _save(fig, f"fig_uncertainty_decomposition_{dataset}.pdf")


def fig_coverage_validity(records: List[Dict], dataset: str):
    """The money figure: does the empirical risk actually respect delta?"""
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    modes = sorted({r["split_mode"] for r in records})
    for i, mode in enumerate(modes):
        rs = [r["test_stage1_risk"] for r in records if r["split_mode"] == mode]
        ax.scatter([i] * len(rs), rs, s=45, zorder=3,
                   label=f"{mode} (n={len(rs)} seeds)")
    ax.axhline(CONFIG["delta_stage1"], color="crimson", ls="--",
               label=f"target δ = {CONFIG['delta_stage1']}")
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels(modes)
    ax.set_ylabel("Empirical Stage-1 risk on test")
    ax.set_title(f"Risk control validity — {dataset}\n"
                 "(exchangeable vs. non-exchangeable calibration)",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=.3, axis="y")
    _save(fig, f"fig_risk_validity_{dataset}.pdf")


def fig_class_conditional_coverage(cc: Dict[str, Any], dataset: str):
    per = cc.get("per_class", {})
    if not per:
        return
    names = list(per.keys())
    covs = [per[n]["coverage"] for n in names]
    ns = [per[n]["n"] for n in names]
    order = np.argsort(ns)
    names = [names[i] for i in order]; covs = [covs[i] for i in order]
    ns = [ns[i] for i in order]

    fig, ax = plt.subplots(figsize=(6.4, 0.35 * len(names) + 1.6))
    cols = ["#c44e52" if c < 1 - CONFIG["alpha_aps"] else "#55a868"
            for c in covs]
    ax.barh(range(len(names)), covs, color=cols)
    ax.axvline(1 - CONFIG["alpha_aps"], color="k", ls="--", lw=1,
               label=f"target 1−α = {1 - CONFIG['alpha_aps']:.2f}")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{n}  (n={c:,})" for n, c in zip(names, ns)],
                       fontsize=8)
    ax.set_xlabel("Empirical coverage")
    ax.set_xlim(0, 1.05)
    ax.set_title(f"Class-conditional coverage — {dataset}\n"
                 "marginal coverage hides rare-class failure", fontsize=10)
    ax.legend(fontsize=8)
    _save(fig, f"fig_class_coverage_{dataset}.pdf")


def fig_active_learning(al: Dict[str, List[Dict]], dataset: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for strat, c in (("uncertainty", "#c44e52"), ("random", "#4c72b0"),
                     ("none", "#8c8c8c")):
        rows = al.get(strat, [])
        if not rows:
            continue
        x = [r["n_labels"] for r in rows]
        axes[0].plot(x, [r["f1_attack"] for r in rows], "o-", color=c,
                     label=strat, linewidth=1.7)
        axes[1].plot(x, [r["fnr"] for r in rows], "o-", color=c,
                     label=strat, linewidth=1.7)
    axes[0].set_xlabel("Analyst labels acquired"); axes[0].set_ylabel("F1 (attack)")
    axes[1].set_xlabel("Analyst labels acquired"); axes[1].set_ylabel("FNR")
    for a in axes:
        a.legend(fontsize=8); a.grid(alpha=.3)
    fig.suptitle(f"Active learning under an analyst labelling budget — {dataset}",
                 fontsize=11)
    _save(fig, f"fig_active_learning_{dataset}.pdf")


def fig_xai(x: Dict[str, Any], dataset: str):
    if not x.get("available"):
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.plot(x["k"], x["curve_shap"], "o-", color="#c44e52",
            label=f"SHAP order (AUC={x['auc_shap']:.3f})")
    ax.plot(x["k"], x["curve_random"], "s--", color="#8c8c8c",
            label=f"random order (AUC={x['auc_random']:.3f})")
    ax.set_xlabel("Number of top features neutralised (k)")
    ax.set_ylabel("Retained probability of predicted class")
    ax.set_title(f"Explanation descriptive accuracy — {dataset}", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    _save(fig, f"fig_xai_deletion_{dataset}.pdf")


def fig_operating_curve(delta_sweep: Dict[str, Dict], dataset: str):
    """
    The operator-facing figure: pick a risk target, read off the escalation rate
    and the cost.  Prior cascaded IDSs cannot draw this plot, because their
    threshold is a number with no risk semantics attached.
    """
    ds = [k for k in sorted(delta_sweep.keys(), key=float)
          if delta_sweep[k].get("feasible", True)]
    if len(ds) < 2:
        log("operating curve skipped: fewer than 2 feasible risk targets", 1)
        return
    d = [float(k) for k in ds]
    esc = [delta_sweep[k]["escalation_rate"] for k in ds]
    risk = [delta_sweep[k]["test_risk"] for k in ds]
    f1 = [delta_sweep[k]["f1_attack"] for k in ds]
    cost = [delta_sweep[k]["cost"] for k in ds]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.5))
    axes[0].plot(d, esc, "o-", color="#4c72b0")
    axes[0].set_xscale("log"); axes[0].set_xlabel("risk target δ")
    axes[0].set_ylabel("Stage-1 escalation rate")
    axes[0].set_title("Cost of a stricter guarantee", fontsize=10)

    axes[1].plot(d, risk, "o-", color="#c44e52", label="achieved test risk")
    axes[1].plot(d, d, "k--", lw=1, label="target δ (validity line)")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("risk target δ"); axes[1].set_ylabel("achieved risk")
    axes[1].set_title("Validity: achieved ≤ target", fontsize=10)
    axes[1].legend(fontsize=8)

    ax2 = axes[2]
    ax2.plot(d, f1, "o-", color="#55a868", label="F1 (attack)")
    ax2.set_xscale("log"); ax2.set_xlabel("risk target δ")
    ax2.set_ylabel("F1 (attack)", color="#55a868")
    ax3 = ax2.twinx()
    ax3.plot(d, cost, "s--", color="#8172b2", label="operational cost")
    ax3.set_ylabel("operational cost", color="#8172b2")
    ax2.set_title("Detection quality vs. cost", fontsize=10)

    for a in axes:
        a.grid(alpha=.3)
    fig.suptitle(f"TRIAGE operating curve — {dataset}", fontsize=11, y=1.06)
    fig.tight_layout()
    _save(fig, f"fig_operating_curve_{dataset}.pdf")


def fig_analyst_error(sens: Dict[str, Dict], dataset: str):
    ks = sorted(sens.keys(), key=float)
    x = [float(k) for k in ks]
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.plot(x, [sens[k]["f1_attack"] for k in ks], "o-", color="#c44e52",
            label="F1 (attack)")
    ax.plot(x, [sens[k]["accuracy"] for k in ks], "s-", color="#4c72b0",
            label="accuracy")
    ax.set_xlabel("Simulated analyst error rate ε")
    ax.set_ylabel("Score")
    ax.set_title(f"Sensitivity to an imperfect human analyst — {dataset}",
                 fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    _save(fig, f"fig_analyst_error_{dataset}.pdf")


def fig_tier_distribution(tier: np.ndarray, dataset: str):
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    labels = ["Tier 1\n(auto-clear)", "Tier 2\n(deep analysis)",
              "Tier 3\n(human review)"]
    vals = [(tier == 1).sum(), (tier == 2).sum(), (tier == 3).sum()]
    cols = ["#55a868", "#4c72b0", "#c44e52"]
    keep = [i for i, v in enumerate(vals) if v > 0]
    ax.pie([vals[i] for i in keep], labels=[labels[i] for i in keep],
           colors=[cols[i] for i in keep], autopct="%1.2f%%", startangle=90,
           wedgeprops=dict(width=0.38, edgecolor="w"), textprops={"fontsize": 9})
    ax.set_title(f"Traffic routing across tiers — {dataset}\n"
                 f"(n={len(tier):,} flows)", fontsize=10)
    _save(fig, f"fig_tier_distribution_{dataset}.pdf")


# ==============================================================================
# 11.  EXPERIMENT DRIVER

# ==============================================================================
# DATA: download from public mirrors and verify (no Google Drive needed)
# ==============================================================================
#
# UNSW-NB15  (Moustafa & Slay, MilCIS 2015). Official source:
#   https://research.unsw.edu.au/projects/unsw-nb15-dataset  (SharePoint folder,
#   not script-accessible). Mirror used here: Hugging Face dataset
#   "Mireu-Lab/UNSW-NB15" (train.csv / test.csv = the official partition CSVs).
# CIC-IDS2017 (Sharafaldin et al., ICISSP 2018). Official source:
#   https://www.unb.ca/cic/datasets/ids-2017.html  (MachineLearningCSV.zip).
#   Mirror used here: Hugging Face dataset "c01dsnap/CIC-IDS2017"
#   (the eight MachineLearningCSV files, original file names).
#
# Every file is checked (row counts, column counts, required columns) and its
# SHA-256 is written to <out_dir>/data_manifest.json, so a reader can confirm
# they used byte-identical inputs. If you already have the official files,
# put them in DATA_DIR/UNSW-NB15 and DATA_DIR/CIC-IDS2017 and nothing is
# downloaded.

DATA_DIR = os.environ.get("TRIAGE_DATA_DIR", "data")

UNSW_REPO, UNSW_REV = "Mireu-Lab/UNSW-NB15", "3eb02c4a0a29866b7abcb5ef77c45cf4fcc8f6b0"
CIC_REPO, CIC_REV = "c01dsnap/CIC-IDS2017", "4e9edfa727f800b6f595bc2ee7420527156562b2"

# The UNSW-NB15 partition files as distributed: the file NAMED training-set
# holds 82,332 records and the file NAMED testing-set holds 175,341 (the
# reverse of the dataset documentation). The pipeline uses the files by name,
# exactly as in the reported experiments.
UNSW_EXPECTED = {"UNSW_NB15_training-set.csv": 82_332,
                 "UNSW_NB15_testing-set.csv": 175_341}
UNSW_REQUIRED_COLS = {"id", "dur", "proto", "service", "state", "sttl",
                      "sbytes", "dbytes", "smean", "attack_cat", "label"}

CIC_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]
CIC_EXPECTED_ROWS, CIC_EXPECTED_COLS = 2_830_743, 79


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hf_get(repo: str, rev: str, filename: str) -> str:
    """Download one file from a public Hugging Face dataset repo. Tries the
    pinned revision first, then the default branch."""
    from huggingface_hub import hf_hub_download
    token = os.environ.get("HF_TOKEN") or None      # optional; repos are public
    last = None
    for r in (rev, None):
        try:
            return hf_hub_download(repo_id=repo, repo_type="dataset",
                                   filename=filename, revision=r, token=token)
        except Exception as e:                       # noqa: BLE001
            last = e
            if r is not None:
                log(f"pinned revision unavailable for {filename}; "
                    f"trying the default branch", 2)
    raise RuntimeError(f"Could not download {filename} from {repo}: {last}")


def _count_rows(path: str) -> int:
    with open(path, "rb") as f:
        return max(sum(1 for _ in f) - 1, 0)


def prepare_unsw(data_dir: str = DATA_DIR) -> str:
    d = ensure_dir(os.path.join(data_dir, "UNSW-NB15"))
    targets = {os.path.join(d, n): n for n in UNSW_EXPECTED}
    if not all(os.path.exists(p) for p in targets):
        log("Downloading UNSW-NB15 partition files ...", 1)
        import shutil
        got = [_hf_get(UNSW_REPO, UNSW_REV, f) for f in ("train.csv", "test.csv")]
        # Assign by row count, so the mapping cannot be silently swapped. Files
        # are copied byte for byte; they are only rewritten if the mirror added
        # an index column, which would otherwise become a spurious feature.
        by_rows = {_count_rows(g): g for g in got}
        for p, name in targets.items():
            n = UNSW_EXPECTED[name]
            if n not in by_rows:
                raise ValueError(
                    f"UNSW-NB15 mirror check failed: no file with {n:,} rows "
                    f"(found {sorted(by_rows)}). Download the official CSVs from "
                    f"https://research.unsw.edu.au/projects/unsw-nb15-dataset and "
                    f"put them in {d}.")
            src = by_rows[n]
            head = pd.read_csv(src, nrows=2)
            if any(str(c).startswith("Unnamed") for c in head.columns):
                df = pd.read_csv(src, low_memory=False)
                df.drop(columns=[c for c in df.columns
                                 if str(c).startswith("Unnamed")]).to_csv(p, index=False)
                log(f"{name}: removed an index column added by the mirror", 2)
            else:
                shutil.copyfile(src, p)
    for p, name in targets.items():
        df = pd.read_csv(p, nrows=5)
        missing = UNSW_REQUIRED_COLS - set(c.strip() for c in df.columns)
        rows = _count_rows(p)
        if rows != UNSW_EXPECTED[name] or len(df.columns) != 45 or missing:
            raise ValueError(f"{name}: {rows:,} rows x {len(df.columns)} cols, "
                             f"missing {sorted(missing)}; expected "
                             f"{UNSW_EXPECTED[name]:,} x 45")
        log(f"OK {name}: {rows:,} rows x 45 columns", 1)
    return d


def prepare_cic(data_dir: str = DATA_DIR) -> str:
    d = ensure_dir(os.path.join(data_dir, "CIC-IDS2017"))
    total = 0
    for name in CIC_FILES:
        p = os.path.join(d, name)
        if not os.path.exists(p):
            log(f"Downloading {name} ...", 1)
            src = _hf_get(CIC_REPO, CIC_REV, name)
            if os.path.abspath(src) != os.path.abspath(p):
                import shutil
                shutil.copyfile(src, p)
        head = pd.read_csv(p, nrows=5, encoding="latin-1")
        ncol = len(head.columns)
        rows = _count_rows(p)
        if ncol != CIC_EXPECTED_COLS:
            raise ValueError(f"{name}: {ncol} columns, expected {CIC_EXPECTED_COLS}")
        if not any(str(c).strip().lower() == "label" for c in head.columns):
            raise ValueError(f"{name}: no Label column")
        total += rows
        log(f"OK {name}: {rows:,} rows", 1)
    if total != CIC_EXPECTED_ROWS:
        raise ValueError(f"CIC-IDS2017 has {total:,} rows in total, expected "
                         f"{CIC_EXPECTED_ROWS:,} (MachineLearningCSV).")
    log(f"OK CIC-IDS2017: {total:,} rows in 8 files", 1)
    return d


def prepare_data(data_dir: str = DATA_DIR) -> None:
    """Download (if needed) and verify both datasets, point CONFIG at them and
    record SHA-256 checksums in <out_dir>/data_manifest.json."""
    banner("DATA")
    manifest = {}
    if "unsw" in CONFIG["datasets"]:
        CONFIG["unsw_dir"] = prepare_unsw(data_dir)
        for n in UNSW_EXPECTED:
            manifest[f"UNSW-NB15/{n}"] = _sha256(os.path.join(CONFIG["unsw_dir"], n))
    if "cicids2017" in CONFIG["datasets"]:
        CONFIG["cic_dir"] = prepare_cic(data_dir)
        for n in CIC_FILES:
            manifest[f"CIC-IDS2017/{n}"] = _sha256(os.path.join(CONFIG["cic_dir"], n))
    manifest["_sources"] = {"UNSW-NB15": f"hf://datasets/{UNSW_REPO}@{UNSW_REV}",
                            "CIC-IDS2017": f"hf://datasets/{CIC_REPO}@{CIC_REV}"}
    save_json(manifest, "data_manifest.json")
    log(f"checksums -> {ckpt_path('data_manifest.json')}", 1)


# ==============================================================================

def run_single(bundle: DatasetBundle, seed: int, split_mode: str) -> Dict:
    set_seed(seed)
    tag = f"{bundle.name}_seed{seed}_{split_mode}"
    banner(f"RUN  {tag}", "-")

    tr, cal1, cal2, te = make_splits(bundle, seed, split_mode)
    log(f"split sizes  train={len(tr):,}  cal1={len(cal1):,}  "
        f"cal2={len(cal2):,}  test={len(te):,}", 1)

    pre = Preprocessor().fit(bundle.X.iloc[tr])
    Xtr = pre.transform(bundle.X.iloc[tr])
    Xc1 = pre.transform(bundle.X.iloc[cal1])
    Xc2 = pre.transform(bundle.X.iloc[cal2])
    Xte = pre.transform(bundle.X.iloc[te])

    ytr_b, ytr_m = bundle.y_bin[tr], bundle.y_multi[tr]
    yc1_b = bundle.y_bin[cal1]
    yc2_m = bundle.y_multi[cal2]
    yte_b, yte_m = bundle.y_bin[te], bundle.y_multi[te]

    n_classes = len(bundle.class_names)

    cascade = TriageCascade(seed, n_classes).fit(Xtr, ytr_b, ytr_m)
    cascade.calibrate(Xc1, yc1_b, Xc2, yc2_m, signal="u_total")

    # Score Stage 1 over the test set ONCE and reuse it for every sweep below.
    u_test = cascade.stage1.predict_uncertainty(Xte)

    out = cascade.run(Xte, yte_b, yte_m, u1=u_test)
    m_triage = core_metrics(yte_b, out["final_bin"], out["final_score"])

    # --- validity check: is the Stage-1 risk actually <= delta on test? -----
    u = out["u"]
    cleared = u["u_total"] <= cascade.lambda1
    test_risk = float(np.mean((yte_b == 1) & (u["pred"] == 0) & cleared))

    esc_rate = len(out["escalated"]) / len(te)
    def_rate = len(out["deferred"]) / len(te)

    log(f"TRIAGE: acc={m_triage['accuracy']:.4f} "
        f"F1={m_triage['f1_attack']:.4f} FNR={m_triage['fnr']:.4f}", 1)
    log(f"routing: escalated={esc_rate:.2%}  deferred={def_rate:.2%}  "
        f"overflow={len(out['overflow']):,}", 1)
    log(f"stage-1 test risk={test_risk:.5f} (target δ={CONFIG['delta_stage1']}) "
        f"-> {'RESPECTED' if test_risk <= CONFIG['delta_stage1'] else 'VIOLATED'}",
        1)

    # --- baselines at matched escalation rate ------------------------------
    baselines = {}
    for sig, name in (("u_total", "Cascade-Entropy"),
                      ("u_margin", "Cascade-MaxProb"),
                      ("u_epistemic", "Cascade-Epistemic"),
                      ("random", "Cascade-Random")):
        p, s, _ = cascade_with_fixed_rate(cascade.stage1, cascade.stage2,
                                          Xte, u, sig, esc_rate, seed)
        baselines[name] = core_metrics(yte_b, p, s)

    p1 = u["pred"]
    baselines["Stage1-only"] = core_metrics(yte_b, p1, u["score"])
    pb_all = cascade.stage2.predict_proba_bin(Xte)[:, 1]
    baselines["Stage2-only (all traffic)"] = core_metrics(
        yte_b, (pb_all > 0.5).astype(int), pb_all)

    # --- conformal set evaluation ------------------------------------------
    pm_te = cascade.stage2.predict_proba_multi(Xte)
    classes = np.asarray(cascade.stage2.classes_)
    # Safe label -> column mapping.  A class present in test but absent from
    # train would otherwise get a silently wrong index from searchsorted.
    cls_pos = {int(c): i for i, c in enumerate(classes)}
    unseen = sorted({int(v) for v in np.unique(yte_m)} - set(cls_pos))
    if unseen:
        log(f"WARNING: {len(unseen)} attack class(es) in test are absent from "
            f"train: {[bundle.class_names[c] for c in unseen]}. They cannot be "
            f"covered by any conformal set and are excluded from coverage "
            f"statistics.", 1)
    valid_mask = np.array([int(v) in cls_pos for v in yte_m])
    yte_idx = np.array([cls_pos.get(int(v), 0) for v in yte_m])
    sets_marg = cascade.aps.predict_sets(pm_te)
    sets_cc = cascade.aps_cc.predict_sets(pm_te)
    cls_lbl = [bundle.class_names[c] for c in classes]
    cov_marg = evaluate_conformal_sets(sets_marg[valid_mask],
                                       yte_idx[valid_mask], cls_lbl)
    cov_cc = evaluate_conformal_sets(sets_cc[valid_mask],
                                     yte_idx[valid_mask], cls_lbl)
    log(f"APS marginal coverage={cov_marg['marginal_coverage']:.4f} "
        f"(target {1 - CONFIG['alpha_aps']:.2f})  "
        f"mean|C|={cov_marg['mean_set_size']:.2f}  "
        f"singleton rate={cov_marg['singleton_rate']:.2%}", 1)

    # --- analyst-error sensitivity -----------------------------------------
    sens = {}
    for eps in CONFIG["analyst_error_rates"]:
        o = cascade.run(Xte, yte_b, yte_m, analyst_error=eps, u1=u_test)
        mm = core_metrics(yte_b, o["final_bin"], o["final_score"])
        sens[str(eps)] = {**mm,
                          "cost": operational_cost(mm, len(o["deferred"]))}

    # --- budget sweep -------------------------------------------------------
    budget_sweep = {}
    for b in [0.0, 0.005, 0.01, 0.02, 0.05, 0.10]:
        o = cascade.run(Xte, yte_b, yte_m, analyst_budget=b, u1=u_test)
        mm = core_metrics(yte_b, o["final_bin"], o["final_score"])
        budget_sweep[str(b)] = {**mm, "n_reviewed": int(len(o["deferred"])),
                                "cost": operational_cost(mm,
                                                         len(o["deferred"]))}

    # --- risk-target (delta) sweep: the operating curve ---------------------
    # This is what an operator actually tunes.  It also protects you from the
    # degenerate case where Stage 1 is already good enough that the chosen delta
    # implies zero escalation -- you can see the whole frontier instead of one
    # point.
    delta_sweep = {}
    for dlt in [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]:
        info = cascade.lambda_for_delta(dlt)
        o = cascade.run(Xte, yte_b, yte_m, u1=u_test, lambda1=info["lambda"])
        mm = core_metrics(yte_b, o["final_bin"], o["final_score"])
        cleared_d = u["u_total"] <= info["lambda"]
        risk_d = float(np.mean((yte_b == 1) & (u["pred"] == 0) & cleared_d))
        delta_sweep[str(dlt)] = {
            "lambda": info["lambda"],
            "cal_risk": info["cal_risk"],
            "feasible": info["feasible"],
            "approx_n_cal_needed": info["approx_n_cal_needed"],
            "test_risk": risk_d,
            "respected": bool(risk_d <= dlt),
            "escalation_rate": float(len(o["escalated"]) / len(te)),
            "deferral_rate": float(len(o["deferred"]) / len(te)),
            **mm,
            "cost": operational_cost(mm, len(o["deferred"])),
        }
    log("delta sweep (target -> escalation / test risk / F1):", 1)
    for k, v in delta_sweep.items():
        flag = ("INFEASIBLE: needs n_cal>~"
                f"{v['approx_n_cal_needed']:,}" if not v["feasible"]
                else ("OK  " if v["respected"] else "VIOLATED"))
        log(f"  δ={k:<7} esc={v['escalation_rate']:6.2%}  "
            f"risk={v['test_risk']:.5f}  F1={v['f1_attack']:.4f}  {flag}", 1)

    # --- risk-coverage ------------------------------------------------------
    rc = {}
    correct1 = (u["pred"] == yte_b)
    for sig, name in (("u_total", "Total entropy"),
                      ("u_epistemic", "Epistemic only"),
                      ("u_aleatoric", "Aleatoric only"),
                      ("u_margin", "Softmax response")):
        cov, risk, aurc = risk_coverage_curve(u[sig], correct1)
        rc[name] = (cov, risk, aurc)

    # --- per-attack-class detection recall (the table reviewers look for) ---
    per_class_recall = {}
    for c in np.unique(yte_m):
        name = bundle.class_names[int(c)]
        m = yte_m == c
        if m.sum() == 0:
            continue
        if name == "Normal":
            per_class_recall[name] = {
                "n": int(m.sum()),
                "specificity": float((out["final_bin"][m] == 0).mean()),
                "tier1_frac": float((out["tier"][m] == 1).mean()),
                "tier3_frac": float((out["tier"][m] == 3).mean()),
            }
        else:
            per_class_recall[name] = {
                "n": int(m.sum()),
                "recall": float((out["final_bin"][m] == 1).mean()),
                "recall_stage1_only": float((u["pred"][m] == 1).mean()),
                "tier1_frac": float((out["tier"][m] == 1).mean()),
                "tier3_frac": float((out["tier"][m] == 3).mean()),
            }
    log("per-class detection (recall | fraction auto-cleared at tier 1):", 1)
    for k, v in sorted(per_class_recall.items(),
                       key=lambda kv: kv[1]["n"]):
        r = v.get("recall", v.get("specificity"))
        log(f"  {k:<18} n={v['n']:>7,}  {r:.4f}   "
            f"tier1={v['tier1_frac']:.2%}  tier3={v['tier3_frac']:.2%}", 1)

    latency = measure_latency(cascade.stage1, Xte, CONFIG["latency_probe_n"])
    log(f"latency  single-flow p50={latency['single_flow_p50_ms']:.3f}ms "
        f"p99={latency['single_flow_p99_ms']:.3f}ms | "
        f"micro-batch({latency['microbatch_size']}) "
        f"{latency['microbatch_p50_ms_per_flow']:.4f}ms/flow "
        f"({latency['microbatch_throughput_fps']:,.0f} f/s) | "
        f"full-batch {latency['batched_throughput_fps']:,.0f} f/s", 1)

    # --- XAI ----------------------------------------------------------------
    xai = {"available": False}
    if len(out["deferred"]) > 5:
        sub = Xte[out["deferred"][:CONFIG["shap_n_samples"]]]
        xai = shap_deletion_curve(cascade.stage2.multi_model, Xtr, sub,
                                  pre.feature_names, CONFIG["deletion_steps"])
        if xai.get("available"):
            log(f"XAI faithfulness gain vs random ordering: "
                f"{xai['faithfulness_gain']:+.4f}", 1)

    # --- figures (first seed only, to keep runtime sane) -------------------
    if seed == CONFIG["seeds"][0]:
        fig_risk_coverage(rc, f"{bundle.name}-{split_mode}")
        fig_uncertainty_decomposition(u, yte_b, f"{bundle.name}-{split_mode}")
        fig_class_conditional_coverage(cov_cc, f"{bundle.name}-{split_mode}")
        fig_tier_distribution(out["tier"], f"{bundle.name}-{split_mode}")
        fig_xai(xai, f"{bundle.name}-{split_mode}")
        fig_operating_curve(delta_sweep, f"{bundle.name}-{split_mode}")
        fig_analyst_error(sens, f"{bundle.name}-{split_mode}")

    return {
        "dataset": bundle.name,
        "seed": seed,
        "split_mode": split_mode,
        "n_train": int(len(tr)), "n_cal": int(len(cal1) + len(cal2)),
        "n_test": int(len(te)),
        "n_classes": n_classes,
        "lambda1": float(cascade.lambda1),
        "rcps_cal_risk": cascade.rcps_info["cal_risk"],
        "rcps_cal_escalation": cascade.rcps_info["cal_escalation_rate"],
        "test_stage1_risk": test_risk,
        "risk_respected": bool(test_risk <= CONFIG["delta_stage1"]),
        "escalation_rate": float(esc_rate),
        "deferral_rate": float(def_rate),
        "overflow": int(len(out["overflow"])),
        "triage": m_triage,
        "triage_cost": operational_cost(m_triage, len(out["deferred"])),
        "baselines": baselines,
        "conformal_marginal": cov_marg,
        "conformal_class_conditional": cov_cc,
        "per_class_detection": per_class_recall,
        "analyst_error_sensitivity": sens,
        "budget_sweep": budget_sweep,
        "delta_sweep": delta_sweep,
        "aurc": {k: v[2] for k, v in rc.items()},
        "latency": latency,
        "timing": out["timing"],
        "xai": {k: v for k, v in xai.items() if k != "global_rank"},
        "_xai_rank": xai.get("global_rank"),
        "class_names": bundle.class_names,
        "feature_names": pre.feature_names,
    }


def aggregate(records: List[Dict]) -> Dict[str, Any]:
    """Mean +/- std across seeds, plus paired Wilcoxon vs. each baseline."""
    agg: Dict[str, Any] = {}
    keyf = lambda r: (r["dataset"], r["split_mode"])
    for k in sorted({keyf(r) for r in records}):
        rs = [r for r in records if keyf(r) == k]
        name = f"{k[0]}|{k[1]}"
        entry: Dict[str, Any] = {"n_seeds": len(rs)}

        for m in ["accuracy", "balanced_acc", "f1_attack", "f1_macro", "mcc",
                  "fpr", "fnr", "roc_auc", "pr_auc", "ece"]:
            v = [r["triage"].get(m) for r in rs if r["triage"].get(m) is not None]
            if v:
                entry[f"triage_{m}"] = {"mean": float(np.mean(v)),
                                        "std": float(np.std(v))}

        for m in ["escalation_rate", "deferral_rate", "test_stage1_risk",
                  "lambda1", "triage_cost"]:
            v = [r[m] for r in rs]
            entry[m] = {"mean": float(np.mean(v)), "std": float(np.std(v))}

        entry["risk_respected_fraction"] = float(
            np.mean([r["risk_respected"] for r in rs]))

        bl_names = list(rs[0]["baselines"].keys())
        entry["baselines"] = {}
        for b in bl_names:
            row = {}
            for m in ["accuracy", "f1_attack", "fnr", "mcc"]:
                v = [r["baselines"][b].get(m) for r in rs
                     if r["baselines"][b].get(m) is not None]
                if v:
                    row[m] = {"mean": float(np.mean(v)), "std": float(np.std(v))}
            a = [r["triage"]["f1_attack"] for r in rs]
            c = [r["baselines"][b]["f1_attack"] for r in rs]
            if len(a) >= 3 and not np.allclose(a, c):
                try:
                    row["wilcoxon_p_vs_triage_f1"] = float(
                        stats.wilcoxon(a, c).pvalue)
                except Exception:
                    row["wilcoxon_p_vs_triage_f1"] = None
            entry["baselines"][b] = row

        cm = [r["conformal_marginal"]["marginal_coverage"] for r in rs]
        entry["conformal_marginal_coverage"] = {"mean": float(np.mean(cm)),
                                                "std": float(np.std(cm))}
        ss = [r["conformal_marginal"]["mean_set_size"] for r in rs]
        entry["conformal_mean_set_size"] = {"mean": float(np.mean(ss)),
                                            "std": float(np.std(ss))}

        ranks = [r["_xai_rank"] for r in rs if r.get("_xai_rank")]
        entry["xai_stability"] = explanation_stability(ranks)

        agg[name] = entry
    return agg


def results_to_latex(agg: Dict[str, Any]) -> str:
    """Emit a LaTeX table you can paste straight into the paper."""
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{TRIAGE vs.\ baselines at matched escalation rate. "
             r"Mean $\pm$ std over seeds.}",
             r"\begin{tabular}{llcccc}", r"\toprule",
             r"Dataset & Method & Acc. & F1$_{atk}$ & FNR & MCC \\",
             r"\midrule"]
    for name, e in agg.items():
        ds = name.replace("|", " / ").replace("_", r"\_")
        t = e
        lines.append(
            f"{ds} & \\textbf{{TRIAGE}} & "
            f"{t['triage_accuracy']['mean']:.4f} & "
            f"{t['triage_f1_attack']['mean']:.4f} & "
            f"{t['triage_fnr']['mean']:.4f} & "
            f"{t['triage_mcc']['mean']:.4f} \\\\")
        for b, row in e["baselines"].items():
            if not row:
                continue
            lines.append(
                f" & {b.replace('_', ' ')} & "
                f"{row.get('accuracy', {}).get('mean', float('nan')):.4f} & "
                f"{row.get('f1_attack', {}).get('mean', float('nan')):.4f} & "
                f"{row.get('fnr', {}).get('mean', float('nan')):.4f} & "
                f"{row.get('mcc', {}).get('mean', float('nan')):.4f} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\label{tab:main}", r"\end{table}"]
    return "\n".join(lines)


def main():
    banner("TRIAGE — Two-stage Risk-controlled Intrusion Analysis "
           "with Guaranteed Escalation")
    ensure_dir(CONFIG["out_dir"])
    log(f"python {platform.python_version()}  numpy {np.__version__}  "
        f"pandas {pd.__version__}")
    log(f"lightgbm={_HAS_LGB}  xgboost={_HAS_XGB}  shap={_HAS_SHAP}")
    log(f"output dir: {os.path.abspath(CONFIG['out_dir'])}")
    save_json(CONFIG, "config.json")

    ds_keys = ["synthetic"] if CONFIG["smoke_test"] else CONFIG["datasets"]
    all_records: List[Dict] = []

    for key in ds_keys:
        banner(f"DATASET: {key}")
        try:
            bundle = load_dataset(key, seed=0)
        except FileNotFoundError as e:
            log(f"SKIPPING {key}: {e}")
            continue
        log(str(bundle), 1)
        log(f"classes: {bundle.class_names}", 1)

        modes = [m for m in CONFIG["split_modes"]
                 if m == "iid" or bundle.official_split is not None]

        for mode in modes:
            for seed in CONFIG["seeds"]:
                ck = f"rec_{bundle.name}_{mode}_s{seed}.json"
                cached = load_json(ck)
                if cached is not None:
                    log(f"[checkpoint] reusing {ck}", 1)
                    all_records.append(cached)
                    continue
                try:
                    rec = run_single(bundle, seed, mode)
                    save_json(rec, ck)
                    all_records.append(rec)
                except Exception as e:
                    import traceback
                    log(f"RUN FAILED ({bundle.name}, {mode}, seed {seed}): {e}")
                    traceback.print_exc()

        # ---- risk-validity figure across seeds and split modes ------------
        recs = [r for r in all_records if r["dataset"] == bundle.name]
        if recs:
            fig_coverage_validity(recs, bundle.name)

        # ---- active learning (one seed; it is the expensive part) ---------
        al_ck = f"al_{bundle.name}.json"
        al = load_json(al_ck)
        if al is None:
            try:
                banner(f"ACTIVE LEARNING — {bundle.name}", "-")
                seed = CONFIG["seeds"][0]
                tr, cal1, cal2, te = make_splits(bundle, seed, "iid")
                pre = Preprocessor().fit(bundle.X.iloc[tr])
                Xtr, Xte = pre.transform(bundle.X.iloc[tr]), pre.transform(
                    bundle.X.iloc[te])
                Xpool = pre.transform(bundle.X.iloc[cal1])
                s1 = Stage1Detector(seed).fit(Xtr, bundle.y_bin[tr])
                up = s1.predict_uncertainty(Xpool)["u_total"]
                al = active_learning_study(
                    Xtr, bundle.y_bin[tr], bundle.y_multi[tr],
                    Xpool, bundle.y_bin[cal1], bundle.y_multi[cal1],
                    Xte, bundle.y_bin[te], bundle.y_multi[te],
                    len(bundle.class_names), seed, up)
                save_json(al, al_ck)
            except Exception as e:
                import traceback
                log(f"AL failed: {e}"); traceback.print_exc(); al = None
        if al:
            fig_active_learning(al, bundle.name)

        del bundle; gc.collect()

    if not all_records:
        log("\nNo runs completed. Check your dataset paths in CONFIG.")
        return

    banner("AGGREGATED RESULTS")
    agg = aggregate(all_records)
    save_json(agg, "aggregate.json")
    save_json(all_records, "all_records.json")

    for name, e in agg.items():
        log()
        log(f"### {name}   ({e['n_seeds']} seeds)")
        log(f"  TRIAGE  acc={e['triage_accuracy']['mean']:.4f}"
            f"±{e['triage_accuracy']['std']:.4f}  "
            f"F1={e['triage_f1_attack']['mean']:.4f}"
            f"±{e['triage_f1_attack']['std']:.4f}  "
            f"FNR={e['triage_fnr']['mean']:.4f}  "
            f"MCC={e['triage_mcc']['mean']:.4f}")
        log(f"  escalation={e['escalation_rate']['mean']:.3%}  "
            f"deferral={e['deferral_rate']['mean']:.3%}")
        log(f"  stage-1 test risk={e['test_stage1_risk']['mean']:.5f} "
            f"(δ={CONFIG['delta_stage1']}) "
            f"respected in {e['risk_respected_fraction']:.0%} of seeds")
        log(f"  APS coverage={e['conformal_marginal_coverage']['mean']:.4f} "
            f"(target {1 - CONFIG['alpha_aps']:.2f})  "
            f"mean|C|={e['conformal_mean_set_size']['mean']:.2f}")
        log(f"  XAI stability (Spearman across seeds)="
            f"{e['xai_stability'].get('mean_spearman', float('nan')):.3f}")
        log("  baselines:")
        for b, row in e["baselines"].items():
            if not row:
                continue
            log(f"    {b:<28} F1={row.get('f1_attack', {}).get('mean', float('nan')):.4f}"
                f"  FNR={row.get('fnr', {}).get('mean', float('nan')):.4f}"
                f"  p={row.get('wilcoxon_p_vs_triage_f1')}", 1)

    tex = results_to_latex(agg)
    with open(ckpt_path("table_main.tex"), "w") as f:
        f.write(tex)

    banner("DONE")
    log(f"All artefacts written to {os.path.abspath(CONFIG['out_dir'])}")


if __name__ == "__main__":
    if not CONFIG["smoke_test"]:
        prepare_data()
    main()
