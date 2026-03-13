# IS5126 Cross-Regional PD Model Transferability Study

> Can US credit default models predict defaults in Singapore? A study on model transferability across financial ecosystems.

## Project Structure

```
is5126-credit-risk/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── 01_data_download_and_eda.ipynb      # Data acquisition + exploratory analysis
│   ├── 02_feature_engineering.ipynb         # Standard + LLM-assisted features
│   ├── 03_baseline_modeling.ipynb           # LR / XGBoost / LightGBM
│   ├── 04_model_evaluation.ipynb            # AUC, KS, Gini, SHAP analysis
│   ├── 05_sg_data_acquisition.ipynb         # Singapore/Asia data collection
│   ├── 06_cross_regional_transfer.ipynb     # Transfer test + drift analysis
│   └── 07_localization.ipynb                # Feature adaptation + recalibration
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py                   # Reusable cleaning/preprocessing functions
│   ├── feature_engineering.py               # Feature engineering pipeline
│   ├── llm_features.py                      # LLM-assisted feature extraction
│   ├── modeling.py                          # Model training utilities
│   ├── evaluation.py                        # Evaluation metrics + SHAP
│   └── drift_monitoring.py                  # PSI / CSI calculations
│
├── data/                                    # ⚠️ Add to .gitignore
│   ├── raw/
│   ├── processed/
│   └── external/                            # SG/Asia data
│
├── models/                                  # Saved model artifacts
│
├── reports/                                 # Generated analysis reports
│   └── figures/
│
└── app/
    └── streamlit_app.py                     # Phase 4 demo
```

## Setup

### Google Colab
Open any notebook in `notebooks/` directly in Colab. Data will be stored in Google Drive.

### Local
```bash
git clone https://github.com/YOUR_USERNAME/is5126-credit-risk.git
cd is5126-credit-risk
pip install -r requirements.txt
```

## Team

| Role | Member | Focus |
|------|--------|-------|
| Domain Expert | TBD | Feature review, industry alignment |
| ML Engineer | TBD | Modeling pipeline, LLM features |
| Data & Research | TBD | Data collection, EDA, documentation |
| Demo & Presentation | TBD | Streamlit app, report, slides |

## Project Phases

- **Phase 1** (Now → Late Feb): US Baseline PD Model on Lending Club ⭐
- **Phase 2** (Early Mar → Mid Mar): Singapore/Asia Data Acquisition
- **Phase 3** (Mid Mar → Early Apr): Cross-Regional Transfer Analysis
- **Phase 4** (April): Synthesis, Demo & Presentation
