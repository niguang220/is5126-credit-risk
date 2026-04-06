# LLM-Enhanced Credit Risk Assessment

> **Does semantic analysis of borrower descriptions improve credit scoring?**
> An end-to-end study on LLM integration in credit default prediction.

**IS5126 Hands-on Applied Analytics | Group 7 | NUS School of Computing | April 2026**

🔗 **[Live Demo](https://is5126-credit-risk-gow2bpzskbfbsjyzjvrzeq.streamlit.app/)** | **[Final Report](final_report_complete.pdf)**

---

## Overview

Traditional credit risk models rely entirely on structured numeric features — FICO score, DTI, income, employment history. But two applicants with identical numbers can have vastly different risk profiles depending on *why* they need the loan — a signal only visible in the borrower's own description.

This project systematically investigates two paradigms for integrating LLMs into a credit risk pipeline:

1. **Pre-stage (Semantic Feature Extraction):** `desc text → LLM → risk score → XGBoost feature`
2. **Post-stage (Boundary Case Correction):** `ML borderline case → LLM reasoning → label correction`

The study uses the public **LendingClub dataset** (1.3M loans, 2007–2018) and compares three LLM configurations — no LLM, naive BERT, and Qwen3-Max — against a strong traditional XGBoost baseline.

---

## Key Results

### Part 1 — Traditional ML Baseline (Full Dataset, 95 features)

| Model | AUC | KS | Gini |
|-------|-----|-----|------|
| Logistic Regression (no grade) | 0.697 | 0.288 | 0.395 |
| **XGBoost (no grade) ← Primary** | **0.719** | **0.318** | **0.438** |
| XGBoost (with grade) | 0.722 | 0.324 | 0.444 |

*Grade features excluded — assigned post-application, proxies the target variable.*

### Part 2 — LLM Semantic Features (3K sample)

| Pipeline | AUC | KS | BACC | Recall |
|----------|-----|-----|------|--------|
| A': Traditional baseline | 0.649 | 0.236 | 0.618 | 0.576 |
| B': + Naive BERT | 0.660 | 0.277 | 0.638 | 0.533 |
| **C: + Qwen3-Max** | 0.637 | **0.286** | **0.643** | **0.609** |

Qwen3-Max achieves the highest KS (+0.050 vs baseline) and Recall (+0.033), improving separation at the operational threshold — at the cost of lower global AUC due to discretised output values.

### Part 3 — LLM Correction on Structured Profiles (200 samples)

| Condition | Accuracy | BACC |
|-----------|----------|------|
| ML alone (borderline cases) | **0.600** | **0.600** |
| ML + LLM correction (borderline) | 0.550 | 0.550 |

LLM correction **hurts** accuracy by 5pp on borderline cases. LLM value is tied to text, not tabular reasoning.

---

## Architecture

```
Loan Application
       │
       ├── Structured Features ──────────────────────────────┐
       │   (FICO, DTI, income, employment, ...)               │
       │                                                       ▼
       └── desc text (optional, ~15% of loans)          XGBoost Model
               │                                              │
               ▼                                              ▼
          Qwen3-Max                                    Risk Score (0–1)
          ↓ default_probability                              │
          (single scalar feature) ────────────────────────── ┘
                                                             │
                                              ┌──────────────┼──────────────┐
                                              ▼              ▼              ▼
                                           Approve     Manual Review    Decline
                                          (< 0.25)     (0.25–0.52)     (> 0.52)
                                                             │
                                              SHAP attribution + Qwen reasoning
```

---

## Repository Structure

```
is5126-credit-risk/
├── notebooks/
│   ├── 01_eda.ipynb                        # EDA on full LC dataset
│   ├── 02_feature_engineering.ipynb        # WoE, derived features, imputation
│   ├── 03_part1_ml_baseline.ipynb          # Part 1: XGBoost baseline
│   ├── 04_part2_bert_pipeline.ipynb        # Part 2: BERT pipelines (A, B, A', B')
│   ├── 05_part2_qwen_pipeline.ipynb        # Part 2: Qwen pipeline (C) + evaluation
│   └── 06_part3_llm_correction.ipynb       # Part 3: LLM correction experiment
│
├── scripts/
│   ├── run_qwen_3000.py                    # Qwen API calls for 3K scoring
│   └── run_qwen_correction.py             # Qwen API calls for Part 3
│
├── src/
│   ├── features.py                         # Feature engineering, WoE, job classification
│   ├── evaluation.py                       # AUC, KS, Gini, bootstrap DeLong
│   └── config.py                           # Shared constants
│
├── demo/
│   ├── app.py                              # Streamlit demo (3-tab interactive app)
│   ├── pipeline.py                         # Full scoring pipeline (OOD → score → SHAP)
│   ├── config.py                           # Demo paths and thresholds
│   ├── data/                               # Bundled model + data files (~5MB)
│   ├── ood_bounds.json                     # Precomputed OOD percentile bounds
│   ├── requirements.txt
│   └── SETUP.md                            # Setup guide for teammates
│
├── final_report_complete.pdf
└── README.md
```

---

## Demo

A Streamlit prototype is deployed at:
**https://is5126-credit-risk-gow2bpzskbfbsjyzjvrzeq.streamlit.app/**

**Three tabs:**
- **Overview** — Pipeline architecture and core results from all three parts
- **Case Studies** — Side-by-side analysis of where LLMs succeed (Case A) and fail (Case B)
- **Interactive Demo** — Select any of 3,000 real applicants; see OOD warnings, dual ML-Only vs ML+Qwen gauge, SHAP top factors, Qwen reasoning, and adjustable risk thresholds

### Run locally

```bash
git clone https://github.com/niguang220/is5126-credit-risk.git
cd is5126-credit-risk
pip install -r demo/requirements.txt
python -m streamlit run demo/app.py
```

Data files are bundled in `demo/data/` — no additional setup needed.
See [`demo/SETUP.md`](demo/SETUP.md) for custom data directory configuration.

---

## Decision Thresholds

| Zone | Score | Basis |
|------|-------|-------|
| ✅ Approve | < 0.25 | Actual default rate 6.6% (vs 26.3% average) |
| 🔶 Manual Review | 0.25 – 0.52 | Model uncertainty zone (~25% of applications) |
| ❌ Decline | > 0.52 | KS-optimal threshold (KS = 0.318) |

---

## Regulatory Context

This architecture aligns with Singapore's **MAS FEAT Framework** (2019):

| Principle | Implementation |
|-----------|---------------|
| **Fairness** | Bias analysis: Qwen scores vs actual default rates by borrower group |
| **Ethics** | No proxy discrimination; decisions grounded in legitimate signals |
| **Accountability** | SHAP attribution for every prediction |
| **Transparency** | Qwen natural language reasoning per applicant |

---

## Tech Stack

`Python` · `XGBoost` · `LightGBM` · `SHAP` · `Streamlit` · `Pandas` · `scikit-learn` · `Qwen3-Max (iFlow API)` · `BERT (HuggingFace)`

---

## Team — Group 7

| Name | Student ID | Contribution |
|------|-----------|--------------|
| He Yufan | A0331801U | GitHub Repo, Demo, Slides |
| Zhang Ruijia | A0215695W | Methodology Design, Report Structure |
| Yiwei Luo | A0328749L | Report Parts 1, 2, 3 |
| Yang Enjian | A0327980U | Report Part 5 |
| Yang Haoran | A0332268A | Report Part 4 |
