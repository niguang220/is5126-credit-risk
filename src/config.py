"""
Central configuration for IS5126 Credit Risk project.
All tunable parameters in one place — no magic numbers in notebooks.
"""

# ── Paths ────────────────────────────────────────────────────
DRIVE_DIR = '/content/drive/MyDrive/is5126'
DATA_DIR = f'{DRIVE_DIR}/data/processed'
MODEL_DIR = f'{DRIVE_DIR}/models'
FIGURE_DIR = 'figures'

# Part 2 / Part 3 paths
DESC_SUBSET_PATH = f'{DATA_DIR}/desc_subset.parquet'
SAMPLE_3000_PATH = f'{DATA_DIR}/sample_3000.parquet'
SAMPLE_200_PATH  = f'{DATA_DIR}/sample_200.parquet'
BERT_EMBED_PATH  = f'{DATA_DIR}/bert_embeddings.npy'
BERT_INDEX_PATH  = f'{DATA_DIR}/bert_embeddings_index.npy'
BERT_PCA_PATH    = f'{DATA_DIR}/bert_pca_features.parquet'
QWEN_CACHE_3000  = f'{MODEL_DIR}/qwen_cache_3000.json'
QWEN_CACHE_200   = f'{MODEL_DIR}/qwen_cache_200.json'

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

# ── PSI Thresholds ───────────────────────────────────────────
PSI_STABLE = 0.10          # < 0.10 = no drift
PSI_MODERATE = 0.25        # 0.10-0.25 = investigate
                           # > 0.25 = significant drift

# ── BERT Config ──────────────────────────────────────────────
BERT_MODEL_NAME = 'bert-base-uncased'
BERT_MAX_LENGTH = 128
BERT_BATCH_SIZE = 64
BERT_PCA_COMPONENTS = 50   # reduce 768-dim → 50-dim before feeding to XGB

# ── Qwen Config ──────────────────────────────────────────────
QWEN_MODEL = 'qwen3-max'
QWEN_BASE_URL = 'https://apis.iflow.cn/v1'
QWEN_API_KEY_SECRET = 'IFLOW_API_KEY'   # Colab Secrets key name
QWEN_REQUEST_DELAY = 0.5               # seconds between API calls (rate limit)
# ⚠️ iFlow API shuts down 2026-04-17 — all LLM experiments must finish before then

# ── Part 3: LLM Correction ───────────────────────────────────
BORDERLINE_LOW  = 0.3      # ML prob below this = confident non-default
BORDERLINE_HIGH = 0.7      # ML prob above this = confident default
                           # [0.3, 0.7] = borderline → send to LLM
PART3_SAMPLE_N  = 200
PART3_BORDERLINE_N = 100   # how many of the 200 are borderline cases

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
    'bert': '#9B59B6',
    'qwen': '#E74C3C',
}
