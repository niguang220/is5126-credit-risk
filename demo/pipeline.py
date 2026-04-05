"""
Demo pipeline: feature engineering → OOD detection → scoring → decision.

All data loaded from Google Drive (local sync). No live API calls — Qwen
results served entirely from qwen_cache_3000.json.
"""
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Allow importing from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features import create_derived_features, classify_job_title_rules
from demo.config import (
    TRAIN_PATH, WOE_MAPS_PATH, IMPUTATION_PATH,
    XGB_MODEL_PATH, XGB_MODEL_BASELINE_PATH, QWEN_CACHE_PATH, DEMO_MODEL_FEATURES_PATH,
    MODEL_FEATURES, MISSING_INDICATOR_FEATURES,
    OOD_FEATURES, OOD_PERCENTILE_LOW, OOD_PERCENTILE_HIGH, OOD_FLAG_THRESHOLD,
    THRESH_APPROVE, THRESH_DECLINE,
)


# ── Data loading (call once at startup) ───────────────────────

def load_artifacts():
    """Load all static artifacts. Call once and cache in Streamlit."""
    model = joblib.load(XGB_MODEL_PATH)
    model_baseline = joblib.load(XGB_MODEL_BASELINE_PATH)
    with open(DEMO_MODEL_FEATURES_PATH) as f:
        model_feature_list = json.load(f)

    with open(WOE_MAPS_PATH) as f:
        woe_maps = json.load(f)

    with open(IMPUTATION_PATH) as f:
        imp = json.load(f)
    numeric_medians = imp["numeric_medians"]

    with open(QWEN_CACHE_PATH) as f:
        qwen_cache = json.load(f)

    # OOD bounds: computed from training data (all OOD_FEATURES are raw columns)
    train_df = pd.read_parquet(TRAIN_PATH, columns=OOD_FEATURES)
    ood_bounds = {
        feat: (
            float(np.nanpercentile(train_df[feat], OOD_PERCENTILE_LOW)),
            float(np.nanpercentile(train_df[feat], OOD_PERCENTILE_HIGH)),
        )
        for feat in OOD_FEATURES if feat in train_df.columns
    }

    return model, model_baseline, woe_maps, numeric_medians, qwen_cache, ood_bounds, model_feature_list


# ── Feature engineering ───────────────────────────────────────

def engineer_features(raw: pd.Series, woe_maps: dict, numeric_medians: dict,
                      feature_list: list[str] | None = None) -> pd.DataFrame:
    """
    Apply full feature engineering to a single raw loan row.
    Returns a 1-row DataFrame with exactly feature_list columns (defaults to MODEL_FEATURES).
    qwen_score should be added to the row BEFORE calling if it is in feature_list.
    """
    if feature_list is None:
        feature_list = MODEL_FEATURES
    df = raw.to_frame().T.copy()
    df = df.reset_index(drop=True)

    # term: ensure numeric (may be "36 months" string or int 36)
    if "term" in df.columns:
        try:
            df["term"] = pd.to_numeric(df["term"], errors="coerce")
            if df["term"].isna().all():  # was a string like "36 months"
                df["term"] = df["term"].astype(str).str.extract(r"(\d+)").astype(float)
        except Exception:
            df["term"] = df["term"].astype(str).str.extract(r"(\d+)").astype(float)

    # Missing indicators (before imputation so we capture real NaNs)
    for feat in MISSING_INDICATOR_FEATURES:
        if feat in df.columns:
            df[f"{feat}_missing"] = df[feat].isna().astype(int)
        else:
            df[f"{feat}_missing"] = 1  # feature entirely absent → treat as missing

    # Imputation
    for feat, median in numeric_medians.items():
        if feat in df.columns:
            df[feat] = pd.to_numeric(df[feat], errors="coerce").fillna(median)

    # Derived ratio features
    df = create_derived_features(df)

    # Job category + WoE
    if "emp_title" in df.columns:
        df["job_category"] = df["emp_title"].apply(classify_job_title_rules)
    else:
        df["job_category"] = "Unknown"

    # WoE encoding for all categoricals
    for col, woe_map in woe_maps.items():
        src_col = col  # e.g. "home_ownership"
        dst_col = f"{col}_woe"
        if src_col in df.columns:
            df[dst_col] = df[src_col].astype(str).map(woe_map).fillna(0.0)
        else:
            df[dst_col] = 0.0

    # Fill any remaining missing model features with 0
    for feat in feature_list:
        if feat not in df.columns:
            df[feat] = 0.0

    return df[feature_list].astype(float)


# ── OOD detection ─────────────────────────────────────────────

def check_ood(raw: pd.Series, ood_bounds: dict) -> dict:
    """
    Check how many key features fall outside training distribution.
    Returns dict with flag, count, and per-feature details.
    """
    details = {}
    out_of_range = 0

    # Need derived features for installment_to_income
    df_tmp = raw.to_frame().T.copy()
    df_tmp = create_derived_features(df_tmp)

    for feat, (lo, hi) in ood_bounds.items():
        val = None
        if feat in df_tmp.columns:
            val = df_tmp[feat].iloc[0]
        elif feat in raw.index:
            val = raw[feat]

        if val is None or (isinstance(val, float) and np.isnan(val)):
            details[feat] = {"value": None, "lo": lo, "hi": hi, "ood": False}
            continue

        is_ood = float(val) < lo or float(val) > hi
        if is_ood:
            out_of_range += 1
        details[feat] = {"value": float(val), "lo": lo, "hi": hi, "ood": is_ood}

    return {
        "is_ood": out_of_range >= OOD_FLAG_THRESHOLD,
        "n_ood": out_of_range,
        "details": details,
    }


# ── Qwen cache lookup ─────────────────────────────────────────

def get_qwen_result(idx: int, qwen_cache: dict) -> dict | None:
    """Look up Qwen result by sample_3000 index. Returns None if not cached."""
    entry = qwen_cache.get(str(idx))
    if entry is None or entry.get("parse_error"):
        return None
    return entry


def desc_quality(desc: str | None) -> str:
    """Classify desc quality: 'good' / 'short' / 'missing'."""
    if not desc or pd.isna(desc):
        return "missing"
    clean = re.sub(r"Borrower added on \d{2}/\d{2}/\d{2}\s*>?\s*", "", str(desc)).strip()
    return "good" if len(clean) > 80 else "short"


# ── Scoring ───────────────────────────────────────────────────

def score(
    raw: pd.Series,
    idx: int,
    model,
    woe_maps: dict,
    numeric_medians: dict,
    qwen_cache: dict,
    ood_bounds: dict,
    model_feature_list: list[str] | None = None,
    model_baseline=None,
) -> dict:
    """
    Run the full 5-step pipeline for one loan application.

    Returns a result dict with all intermediate outputs.
    """
    result = {}

    # Step 1 — desc quality pre-screen
    desc = raw.get("desc", None)
    dq = desc_quality(desc)
    result["desc"] = desc
    result["desc_quality"] = dq  # "good" | "short" | "missing"

    # Step 2 — OOD detection
    ood = check_ood(raw, ood_bounds)
    result["ood"] = ood

    # Step 4 — Qwen lookup (done before feature engineering so qwen_score can be injected)
    qwen = None
    if dq == "good":
        qwen = get_qwen_result(idx, qwen_cache)
    result["qwen"] = qwen

    # Step 3 — Feature engineering + ML score
    # Inject qwen_score into raw so engineer_features can include it
    raw_with_qwen = raw.copy()
    if qwen is not None:
        raw_with_qwen["qwen_score"] = float(qwen["score"])
    else:
        raw_with_qwen["qwen_score"] = 0.0  # fallback: no Qwen signal

    X = engineer_features(raw_with_qwen, woe_maps, numeric_medians, model_feature_list)
    ml_score = float(model.predict_proba(X)[0, 1])
    result["ml_score"] = ml_score
    result["feature_df"] = X  # for SHAP

    # Baseline score: ML only, no qwen_score (uses Part 1 model, 95 features)
    if model_baseline is not None:
        X_base = engineer_features(raw, woe_maps, numeric_medians, MODEL_FEATURES)
        result["ml_baseline_score"] = float(model_baseline.predict_proba(X_base)[0, 1])
    else:
        result["ml_baseline_score"] = None

    # Step 5 — Decision
    result["decision"] = _make_decision(ml_score, THRESH_APPROVE, THRESH_DECLINE)

    return result


def _make_decision(score: float, thresh_approve: float, thresh_decline: float) -> str:
    if score < thresh_approve:
        return "Approve"
    elif score > thresh_decline:
        return "Decline"
    return "Manual Review"


# ── SHAP ─────────────────────────────────────────────────────

def compute_shap(model, X: pd.DataFrame, top_n: int = 10) -> list[dict]:
    """
    Compute SHAP values for a single row.
    Returns top_n features sorted by |shap_value|.
    """
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # XGBoost TreeExplainer returns array of shape (1, n_features)
    sv = shap_values[0] if shap_values.ndim == 2 else shap_values

    feature_names = X.columns.tolist()
    impacts = sorted(
        zip(feature_names, sv, X.iloc[0].tolist()),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    return [
        {"feature": name, "shap": float(sv), "value": float(val)}
        for name, sv, val in impacts[:top_n]
    ]
