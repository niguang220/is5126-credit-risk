"""
Demo configuration — paths, thresholds, constants.

Path resolution order:
  1. IS5126_DATA_DIR environment variable (explicit override)
  2. demo/data/ directory bundled inside the repo (for Streamlit Cloud / teammates)
  3. Default Google Drive mount path (original dev machine)
"""
import os
from pathlib import Path

_REPO_DATA = Path(__file__).parent / "data"  # demo/data/ bundled in repo

if "IS5126_DATA_DIR" in os.environ:
    DRIVE_DIR = Path(os.environ["IS5126_DATA_DIR"])
    DATA_DIR  = DRIVE_DIR / "data/processed"
    MODEL_DIR = DRIVE_DIR / "models"
elif _REPO_DATA.exists():
    DATA_DIR  = _REPO_DATA / "processed"
    MODEL_DIR = _REPO_DATA / "models"
    DRIVE_DIR = _REPO_DATA.parent  # not used directly
else:
    DRIVE_DIR = Path("G:/我的云端硬盘/is5126")
    DATA_DIR  = DRIVE_DIR / "data/processed"
    MODEL_DIR = DRIVE_DIR / "models"
FIGURE_DIR  = DRIVE_DIR / "figures"

# Data files
SAMPLE_3000_PATH       = DATA_DIR / "sample_3000.parquet"
TRAIN_PATH             = DATA_DIR / "train.parquet"  # only needed if ood_bounds.json is absent
WOE_MAPS_PATH          = DATA_DIR / "woe_maps.json"
IMPUTATION_PATH        = DATA_DIR / "imputation_values.json"

# Precomputed OOD bounds (eliminates train.parquet dependency for deployment)
OOD_BOUNDS_PATH        = Path(__file__).parent / "ood_bounds.json"

# Model files
XGB_MODEL_PATH           = MODEL_DIR / "demo_model_with_qwen.joblib"
XGB_MODEL_BASELINE_PATH  = MODEL_DIR / "XGB_B_no_grade.joblib"
DEMO_MODEL_FEATURES_PATH = MODEL_DIR / "demo_model_features.json"
QWEN_CACHE_PATH        = MODEL_DIR / "qwen_cache_3000.json"
MODEL_RESULTS_PATH     = MODEL_DIR / "model_results.json"
TABLE1_PATH            = MODEL_DIR / "part2_table1_bert_results.csv"
TABLE2_PATH            = MODEL_DIR / "part2_table2_qwen_results.csv"
TABLE3_PATH            = MODEL_DIR / "part3_correction_results.csv"

# Figure files (Part 2 & 3 — already generated)
FIG_PART2_ROC_BERT     = FIGURE_DIR / "part2_roc_bert.png"
FIG_PART2_ROC_COMP     = FIGURE_DIR / "part2_roc_comparison.png"
FIG_PART2_METRICS      = FIGURE_DIR / "part2_metrics_comparison.png"
FIG_PART2_CONSISTENCY  = FIGURE_DIR / "part2_consistency.png"
FIG_PART2_BIAS         = FIGURE_DIR / "part2_bias.png"
FIG_PART2_QWEN_DIST    = FIGURE_DIR / "part2_qwen_distribution.png"
FIG_PART3_RESULTS      = FIGURE_DIR / "part3_correction_results.png"

# ── Decision thresholds (data-driven from KS analysis) ────────
# KS-optimal threshold = 0.5184, KS = 0.3180 on test set
# Approve < THRESH_APPROVE  →  Approve
# THRESH_APPROVE ≤ score ≤ THRESH_DECLINE  →  Manual Review
# score > THRESH_DECLINE  →  Decline
THRESH_APPROVE  = 0.25   # actual default rate < 6.6% below this — business-acceptable risk
THRESH_DECLINE  = 0.52   # KS-optimal threshold (KS=0.318, max TPR-FPR separation)

# ── OOD detection ─────────────────────────────────────────────
# Features to check for out-of-distribution values
OOD_FEATURES = [
    "fico_score", "dti", "annual_inc", "loan_amnt",
    "installment", "revol_util", "inq_last_6mths",
]
OOD_PERCENTILE_LOW  = 5    # below this → unusual
OOD_PERCENTILE_HIGH = 95   # above this → unusual
OOD_FLAG_THRESHOLD  = 3    # flag if ≥ this many features are OOD

# ── Case study indices (in sample_3000) ───────────────────────
CASE_STUDY_1_IDX = 2869   # Startup + debt consolidation — ML approves, Qwen correctly warns
CASE_STUDY_2_IDX = 2543   # Template text — Qwen fails due to low-info desc

# ── Model feature list (95 features, XGB_B_no_grade) ──────────
# Exact order as trained — DO NOT reorder
MODEL_FEATURES = [
    "acc_now_delinq", "acc_open_past_24mths", "acc_open_past_24mths_missing",
    "addr_state_woe", "annual_inc", "application_type_woe",
    "avg_account_bal", "avg_cur_bal", "avg_cur_bal_missing",
    "bc_util", "bc_util_missing", "chargeoff_within_12_mths",
    "collections_12_mths_ex_med", "credit_history_months",
    "delinq_2yrs", "delinq_amnt", "delinq_per_year", "dti",
    "emp_length_num", "emp_length_num_missing", "fico_score",
    "home_ownership_woe", "initial_list_status_woe", "inq_last_6mths",
    "inq_per_open_acc", "installment", "installment_to_income",
    "is_long_term", "job_category_woe", "loan_amnt", "loan_to_income",
    "mo_sin_old_il_acct", "mo_sin_old_il_acct_missing",
    "mo_sin_old_rev_tl_op", "mo_sin_old_rev_tl_op_missing",
    "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_rev_tl_op_missing",
    "mo_sin_rcnt_tl", "mo_sin_rcnt_tl_missing",
    "monthly_inc", "mort_acc", "mort_acc_missing",
    "mths_since_last_delinq", "mths_since_last_delinq_missing",
    "mths_since_recent_bc", "mths_since_recent_bc_missing",
    "mths_since_recent_inq", "mths_since_recent_inq_missing",
    "mths_since_recent_revol_delinq", "mths_since_recent_revol_delinq_missing",
    "new_acc_rate",
    "num_accts_ever_120_pd", "num_accts_ever_120_pd_missing",
    "num_actv_bc_tl", "num_actv_bc_tl_missing",
    "num_actv_rev_tl", "num_actv_rev_tl_missing",
    "num_bc_sats", "num_bc_sats_missing",
    "num_bc_tl", "num_bc_tl_missing",
    "num_il_tl", "num_il_tl_missing",
    "num_rev_accts", "num_rev_accts_missing",
    "num_tl_120dpd_2m", "num_tl_120dpd_2m_missing",
    "num_tl_30dpd", "num_tl_30dpd_missing",
    "num_tl_90g_dpd_24m", "num_tl_90g_dpd_24m_missing",
    "num_tl_op_past_12m", "num_tl_op_past_12m_missing",
    "open_acc", "pct_tl_nvr_dlq", "pct_tl_nvr_dlq_missing",
    "percent_bc_gt_75", "percent_bc_gt_75_missing",
    "pub_rec", "pub_rec_bankruptcies", "purpose_woe",
    "revol_bal", "revol_bal_to_income", "revol_to_total_bal",
    "revol_util", "tax_liens", "term",
    "tot_coll_amt", "tot_coll_amt_missing",
    "tot_cur_bal", "tot_cur_bal_missing",
    "total_acc", "total_rev_hi_lim", "total_rev_hi_lim_missing",
    "verification_status_woe",
]

# Features that need missing indicators computed
MISSING_INDICATOR_FEATURES = [
    "acc_open_past_24mths", "avg_cur_bal", "bc_util", "emp_length_num",
    "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op",
    "mo_sin_rcnt_tl", "mort_acc", "mths_since_last_delinq",
    "mths_since_recent_bc", "mths_since_recent_inq",
    "mths_since_recent_revol_delinq", "num_accts_ever_120_pd",
    "num_actv_bc_tl", "num_actv_rev_tl", "num_bc_sats", "num_bc_tl",
    "num_il_tl", "num_rev_accts", "num_tl_120dpd_2m", "num_tl_30dpd",
    "num_tl_90g_dpd_24m", "num_tl_op_past_12m", "pct_tl_nvr_dlq",
    "percent_bc_gt_75", "tot_coll_amt", "tot_cur_bal", "total_rev_hi_lim",
]

# ── UI colours (matching existing project palette) ────────────
COLORS = {
    "approve":  "#28A745",
    "review":   "#FFC107",
    "decline":  "#DC3545",
    "ml":       "#2E86AB",
    "qwen":     "#E74C3C",
    "bert":     "#9B59B6",
    "neutral":  "#6C757D",
}
