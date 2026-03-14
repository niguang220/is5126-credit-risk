# Cross-Regional Credit Risk: PD Model Transferability Study

> Can a US-trained credit default model generalise to Singapore? 
> An end-to-end study on model transferability across financial ecosystems.

---

## Overview

This project investigates whether Probability of Default (PD) models 
trained on US lending data (Lending Club) can transfer to a 
Singapore/Asia credit context — and what adaptations are needed when 
they don't.

Built as part of IS5126 (Applied Machine Learning for Business) at NUS, 
with the full modeling pipeline, feature engineering, and infrastructure 
developed independently.

**Key techniques covered:**
- Logistic Regression / XGBoost / LightGBM for credit scoring
- LLM-assisted feature extraction
- SHAP-based model explainability
- Population Stability Index (PSI) for distribution drift detection
- Cross-regional model recalibration

---

## Project Phases

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | US Baseline PD Model (Lending Club) — EDA, Feature Engineering, Modeling, Evaluation | ✅ Complete |
| 2 | Singapore/Asia Data Acquisition | 🔄 In Progress |
| 3 | Cross-Regional Transfer Analysis + Drift | ⬜ Upcoming |
| 4 | Localisation, Demo & Presentation | ⬜ Upcoming |

---

## Repository Structure
```
is5126-credit-risk/
├── notebooks/
│   ├── 01_data_download_and_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_modeling.ipynb
│   ├── 04_model_evaluation.ipynb
│   ├── 05_sg_data_acquisition.ipynb
│   ├── 06_cross_regional_transfer.ipynb
│   └── 07_localization.ipynb
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── llm_features.py
│   ├── modeling.py
│   ├── evaluation.py
│   └── drift_monitoring.py
├── models/
├── reports/figures/
├── requirements.txt
└── .gitignore
```

---

## Setup

\`\`\`bash
git clone https://github.com/niguang220/is5126-credit-risk.git
cd is5126-credit-risk
pip install -r requirements.txt
\`\`\`

Or open any notebook directly in Google Colab.

---

## Tech Stack

`Python` `scikit-learn` `XGBoost` `LightGBM` `SHAP` `Pandas` `Streamlit`

---

## Author

**Darren** — Master of Computing (General Track), NUS  
[GitHub](https://github.com/niguang220)
