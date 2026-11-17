# TRIAGE: Risk-Controlled Escalation for Human-in-the-Loop Network Intrusion Detection

Code to reproduce the results of the article *TRIAGE: Risk-Controlled Escalation for Human-in-the-Loop Network Intrusion Detection*.

TRIAGE is a three-tier intrusion detection cascade. A fast random-forest filter resolves most flows. Its escalation threshold is chosen by a distribution-free risk-control procedure (Hoeffding–Bentkus bound with fixed-sequence testing), so the false-negative rate it introduces is bounded in finite samples. Escalated flows go to a gradient-boosted classifier that returns a conformal set of candidate attack classes, and ambiguous flows are sent to a human analyst under an explicit review budget.

## Quick start

**Google Colab (recommended):** open `triage_ids.ipynb` in Colab and choose *Runtime → Run all*. Nothing else is needed: the notebook installs its packages and downloads both datasets itself. No Google Drive is used.

**Locally:**

```bash
pip install -r requirements.txt
python triage_ids.py                 # downloads the data, then runs every experiment
python analysis.py triage_results    # effect sizes and coverage tables reported in the article
python make_figures.py triage_results  # redraws Figs. 2-6 into triage_results/paper_figures/
```

**Fast checks:** in the notebook, set `SMOKE_TEST = True` (synthetic data, about 1 minute) or `QUICK = True` (real data, one seed instead of six).

## Data

Both datasets are public. `prepare_data()` downloads them, checks row counts, column counts and required columns, and writes SHA-256 checksums of every input file to `triage_results/data_manifest.json`.

| Dataset | Official source | Mirror used for automatic download | Expected size |
|---|---|---|---|
| UNSW-NB15 (official partition CSVs) | https://research.unsw.edu.au/projects/unsw-nb15-dataset | Hugging Face `Mireu-Lab/UNSW-NB15` | training-set 82,332 rows, testing-set 175,341 rows, 45 columns |
| CIC-IDS2017 (MachineLearningCSV, 8 files) | https://www.unb.ca/cic/datasets/ids-2017.html | Hugging Face `c01dsnap/CIC-IDS2017` | 2,830,743 rows, 79 columns |

The official UNSW-NB15 download is a SharePoint folder that scripts cannot reach, so a mirror is used, pinned to a fixed revision. If you already have the official files, put them in `data/UNSW-NB15/` and `data/CIC-IDS2017/` (same file names) and nothing is downloaded. They are verified in the same way.

**UNSW-NB15 partition files.** The files are used by name: `UNSW_NB15_training-set.csv` is the official training partition and `UNSW_NB15_testing-set.csv` the official test partition. In the distributed copies the training-set file holds 82,332 records and the testing-set file 175,341, the reverse of the dataset documentation. The loader checks these counts.

Please cite the dataset authors if you use the data:
- N. Moustafa and J. Slay, "UNSW-NB15: A Comprehensive Data Set for Network Intrusion Detection Systems", MilCIS 2015, doi:10.1109/MilCIS.2015.7348942
- I. Sharafaldin, A.H. Lashkari, and A.A. Ghorbani, "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization", ICISSP 2018, doi:10.5220/0006639801080116

## What reproduces what

| Article | Numbers from | Figure redrawn by `make_figures.py` |
|---|---|---|
| Table 2 (detection performance) | `aggregate.json`, `table_main.tex` (main run) | |
| Table 3 (paired effect sizes), missed-intrusion reduction | `analysis.py` → `paper_statistics.json` | |
| Table 4, Fig. 2 (risk-control validity) | `analysis.py` | `fig2_risk_validity` |
| Fig. 3 (operating curve, δ sweep) | `delta_sweep` in `all_records.json` | `fig3_operating_curve` |
| Table 5, Fig. 4 (per-class coverage, Mondrian) | `analysis.py` | `fig4_mondrian` |
| Fig. 5 (analyst budget), analyst-error sweep | `budget_sweep`, `analyst_error_sensitivity` in `all_records.json` | `fig5_budget` |
| Explanations, latency, AURC, active learning (Sec. 6) | `all_records.json`, `al_*.json` | |
| Table 6, Fig. 6 (Stage-2 sweep) | `stage2_sensitivity.py` (or the notebook's last section, off by default), then `analysis.py` | `fig6_sensitivity` |

The main run also writes its own diagnostic figures (`fig_*.pdf`) to `triage_results/`.

All settings used in the article are the defaults in `CONFIG` (top of `triage_ids.py`): δ = 0.01, γ = 0.10, α = 0.05, analyst budget 2%, six seeds (0–5), at most 400,000 flows per dataset (stratified subsample).

## Runtime and environment

On a free Colab CPU runtime the full main run takes several hours (6 seeds × 2 datasets, plus the official UNSW-NB15 partition, active learning and SHAP). The Stage-2 sweep adds about 3 hours. Every finished run is checkpointed to `triage_results/`; if a session is interrupted, run it again and it resumes.

The article's results were produced on Google Colab with Python 3.13, NumPy 2.1, pandas 2.2, LightGBM, XGBoost and SHAP (see `requirements.txt`). Small numerical differences can appear with other library versions. Bootstrap intervals computed by `analysis.py` can differ from the article in the third decimal because of the resampling random state.

## Repository contents

| File | Purpose |
|---|---|
| `triage_ids.ipynb` | Self-contained notebook: install, download and verify data, run, package results |
| `triage_ids.py` | The same code as a script (`python triage_ids.py`) |
| `stage2_sensitivity.py` | Optional Stage-2 sensitivity sweep (Table 6, Fig. 6) |
| `analysis.py` | Effect sizes, risk-validity, per-class coverage and ANOVA statistics reported in the article |
| `make_figures.py` | Redraws the article's Figs. 2–6 from the run outputs |
| `requirements.txt` | Python dependencies |

## License

Code: MIT (see `LICENSE`). The datasets are not redistributed here and remain under their owners' terms.
