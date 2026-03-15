# Cross-Regional Credit Risk: PD Model Transferability Study

> Can a US-trained credit default model generalise to Singapore?
> An end-to-end study on model transferability across financial ecosystems.

---

## Overview

This project investigates whether Probability of Default (PD) models
trained on US lending data (Lending Club) can transfer to a
Singapore/Asia credit context — and what adaptations are needed when
they don't.

Built as part of **IS5126 (Hands-on with Applied Analytics)** at NUS.

**Key techniques:**
- Logistic Regression / XGBoost / LightGBM for credit scoring
- WoE encoding & IV-based feature selection
- LLM-assisted feature engineering (381K job titles → 15 semantic categories)
- SHAP-based model explainability
- Population Stability Index (PSI) for distribution drift detection
- Copula-based synthetic data generation anchored to public statistics
- Cross-regional model recalibration

---

## Project Phases

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | US Baseline PD Model — EDA, Feature Engineering, Modeling & SHAP | ✅ Complete |
| 2 | Singapore Synthetic Data Generation (Copula + MAS/SingStat anchoring) | 🔄 In Progress |
| 3 | Cross-Regional Transfer Analysis — Direct transfer, PSI drift, adaptation | ⬜ Upcoming |
| 4 | Streamlit Demo & Final Presentation | ⬜ Upcoming |

---

## Phase 1 Results (US Baseline)

| Model | AUC | KS | Gini |
|---|---|---|---|
| Logistic Regression | 0.6973 | 0.288 | 0.395 |
| **XGBoost (tuned) ★** | **0.7195** | **0.318** | **0.439** |
| LightGBM (tuned) | 0.6760 | 0.251 | 0.352 |

*All results from Experiment B (no grade features) — the transfer-ready configuration.*

**Top SHAP features:** `is_long_term`, `installment_to_income`, `fico_score`, `installment`, `dti`

---

## Repository Structure

```
is5126-credit-risk/
├── notebooks/
│   ├── 01_data_download_and_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_modeling.ipynb
│   ├── 04_sg_synthetic_data.ipynb          # Phase 2
│   └── 05_transfer_analysis.ipynb          # Phase 3
│
├── src/
│   ├── features.py          # WoE/IV calculation, derived ratios, job title classification
│   ├── evaluation.py        # AUC, KS, Gini, Brier, calibration metrics
│   └── drift.py             # PSI computation, feature distribution comparison
│
├── app.py                   # Streamlit demo (Phase 4)
├── data/
│   └── processed/
├── models/
├── figures/
├── requirements.txt
└── .gitignore
```

**Design principle:** Notebooks drive the analysis narrative; `src/` holds reusable
functions extracted from the notebooks to avoid duplication across phases.
Notebooks import from `src/` via `sys.path.append('..')`.

---

## Setup

```bash
git clone https://github.com/niguang220/is5126-credit-risk.git
cd is5126-credit-risk
pip install -r requirements.txt
```

Or open any notebook directly in Google Colab (data is stored on Google Drive under `is5126/`).

---

## Tech Stack

`Python` `scikit-learn` `XGBoost` `LightGBM` `SHAP` `SDV` `Pandas` `OpenAI API` `Streamlit`

---

## Author

**Darren** — Master of Computing (General Track), NUS
[GitHub](https://github.com/niguang220)
