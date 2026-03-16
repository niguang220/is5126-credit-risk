"""
Central configuration for IS5126 Credit Risk project.
All tunable parameters in one place — no magic numbers in notebooks.
"""

# ── Paths ────────────────────────────────────────────────────
DRIVE_DIR = '/content/drive/MyDrive/is5126'
DATA_DIR = f'{DRIVE_DIR}/data/processed'
MODEL_DIR = f'{DRIVE_DIR}/models'
FIGURE_DIR = 'figures'

# ── Time Split ───────────────────────────────────────────────
TIME_SPLIT_DATE = '2017-01-01'

# ── Target ───────────────────────────────────────────────────
TARGET_COL = 'default'

# ── Feature Groups ───────────────────────────────────────────
GRADE_RELATED = ['grade_num', 'sub_grade_num', 'int_rate']

WOE_CATEGORICALS = [
    'home_ownership', 'verification_status', 'purpose',
    'initial_list_status', 'application_type', 'addr_state',
]

DERIVED_FEATURES = [
    'monthly_inc', 'installment_to_income', 'loan_to_income',
    'revol_to_total_bal', 'avg_account_bal', 'delinq_per_year',
    'inq_per_open_acc', 'revol_bal_to_income', 'new_acc_rate',
    'is_long_term',
]

# ── Model Hyperparameters ────────────────────────────────────
XGB_PARAMS = {
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'auc',
    'early_stopping_rounds': 50,
}

LGB_PARAMS = {
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.02,
    'num_leaves': 63,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'is_unbalance': True,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}

LR_PARAMS = {
    'max_iter': 1000,
    'C': 0.1,
    'class_weight': 'balanced',
    'solver': 'lbfgs',
    'random_state': 42,
    'n_jobs': -1,
}

# ── Feature Selection ────────────────────────────────────────
IV_THRESHOLD = 0.02        # minimum IV to keep a feature
IV_SUSPICIOUS = 0.50       # IV above this suggests leakage

# ── PSI Thresholds (Phase 3) ────────────────────────────────
PSI_STABLE = 0.10          # < 0.10 = no drift
PSI_MODERATE = 0.25        # 0.10-0.25 = investigate
                           # > 0.25 = significant drift

# ── SG Synthetic Data Anchors (Phase 2) ─────────────────────
SG_ANCHORS = {
    'home_ownership_rate': 0.90,       # SingStat
    'median_income_sgd': 54000,        # SingStat
    'savings_rate': 0.30,              # CPF contribution
    'credit_score_range': (1000, 2000),  # CBS
    'default_rate': 0.055,             # MAS charge-off rate
}

# ── Visualization ────────────────────────────────────────────
COLORS = {
    'primary': '#1F4E79',
    'accent': '#2E86AB',
    'good': '#28A745',
    'bad': '#DC3545',
    'neutral': '#6C757D',
    'lr': '#E67E22',
    'xgb': '#2E86AB',
    'lgb': '#28A745',
}
