# Demo Setup Guide

## Prerequisites

Python 3.10+ with the following packages:

```bash
pip install -r demo/requirements.txt
```

---

## Step 1 — Get the data files

The demo requires model and data files that are too large for GitHub.
Download the shared Google Drive folder and place it (or sync it) somewhere on your machine.

You need this folder structure:
```
is5126/
├── data/
│   └── processed/
│       ├── sample_3000.parquet
│       ├── train.parquet
│       ├── woe_maps.json
│       └── imputation_values.json
├── models/
│   ├── demo_model_with_qwen.joblib
│   ├── XGB_B_no_grade.joblib
│   ├── demo_model_features.json
│   ├── qwen_cache_3000.json
│   ├── model_results.json
│   ├── part2_table1_bert_results.csv
│   ├── part2_table2_qwen_results.csv
│   └── part3_correction_results.csv
└── figures/
    └── (optional — for Overview tab images)
```

---

## Step 2 — Set the data directory

Tell the demo where your `is5126` folder is by setting an environment variable.

**Windows (Command Prompt):**
```cmd
set IS5126_DATA_DIR=C:\path\to\is5126
```

**Windows (PowerShell):**
```powershell
$env:IS5126_DATA_DIR = "C:\path\to\is5126"
```

**Mac / Linux:**
```bash
export IS5126_DATA_DIR=/path/to/is5126
```

> If your Google Drive is already mounted at `G:\我的云端硬盘\is5126`, you can skip this step — that's the default path.

---

## Step 3 — Run the demo

```bash
python -m streamlit run demo/app.py
```

Then open http://localhost:8501 in your browser.

---

## Troubleshooting

**`FileNotFoundError` on startup**
→ Check that `IS5126_DATA_DIR` points to the right folder and that all files in Step 1 are present.

**`ModuleNotFoundError`**
→ Make sure you installed requirements: `pip install -r demo/requirements.txt`

**Port 8501 already in use**
→ Kill the existing process or run on a different port:
```bash
python -m streamlit run demo/app.py --server.port 8502
```
