"""
Feature engineering utilities for credit risk modeling.
Extracted from notebooks 02 (feature_engineering) to enable reuse across phases.
"""

import pandas as pd
import numpy as np
import re


# ── WoE / IV ────────────────────────────────────────────────

def calculate_woe_iv(df, feature, target='default'):
    """Calculate Weight of Evidence and Information Value for a categorical feature.
    
    Must be fit on TRAINING data only to prevent leakage.
    Returns: (woe_map dict, iv_total float, crosstab DataFrame)
    """
    crosstab = pd.crosstab(df[feature], df[target])
    crosstab.columns = ['good', 'bad']
    crosstab['good_pct'] = (crosstab['good'] / crosstab['good'].sum()).clip(lower=0.0001)
    crosstab['bad_pct'] = (crosstab['bad'] / crosstab['bad'].sum()).clip(lower=0.0001)
    crosstab['woe'] = np.log(crosstab['good_pct'] / crosstab['bad_pct'])
    crosstab['iv'] = (crosstab['good_pct'] - crosstab['bad_pct']) * crosstab['woe']
    return crosstab['woe'].to_dict(), crosstab['iv'].sum(), crosstab


def calculate_numeric_iv(df, feature, target='default', n_bins=10):
    """Calculate IV for a numeric feature by binning into quantiles."""
    try:
        df_temp = df[[feature, target]].dropna()
        if df_temp[feature].nunique() <= n_bins:
            df_temp['bin'] = df_temp[feature]
        else:
            df_temp['bin'] = pd.qcut(df_temp[feature], q=n_bins, duplicates='drop')
        crosstab = pd.crosstab(df_temp['bin'], df_temp[target])
        crosstab.columns = ['good', 'bad']
        crosstab['good_pct'] = (crosstab['good'] / crosstab['good'].sum()).clip(lower=0.0001)
        crosstab['bad_pct'] = (crosstab['bad'] / crosstab['bad'].sum()).clip(lower=0.0001)
        crosstab['woe'] = np.log(crosstab['good_pct'] / crosstab['bad_pct'])
        crosstab['iv'] = (crosstab['good_pct'] - crosstab['bad_pct']) * crosstab['woe']
        return crosstab['iv'].sum()
    except Exception:
        return 0


def apply_woe_encoding(df_train, df_test, features, target='default'):
    """Fit WoE on train, apply to both train and test. Returns woe_maps dict."""
    woe_maps = {}
    iv_results = []
    for feat in features:
        if feat not in df_train.columns:
            continue
        woe_map, iv, _ = calculate_woe_iv(df_train, feat, target)
        woe_maps[feat] = {str(k): float(v) for k, v in woe_map.items()}
        iv_results.append({'feature': feat, 'IV': round(iv, 4)})
        df_train[f'{feat}_woe'] = df_train[feat].map(woe_map).fillna(0)
        df_test[f'{feat}_woe'] = df_test[feat].map(woe_map).fillna(0)
    return woe_maps, pd.DataFrame(iv_results).sort_values('IV', ascending=False)


# ── Derived ratio features ──────────────────────────────────

def create_derived_features(df):
    """Create domain-driven ratio features. Deterministic — safe for train and test."""
    df = df.copy()
    df['monthly_inc'] = df['annual_inc'] / 12
    df['installment_to_income'] = df['installment'] / (df['monthly_inc'] + 1)
    df['loan_to_income'] = df['loan_amnt'] / (df['annual_inc'] + 1)
    if 'tot_cur_bal' in df.columns:
        df['revol_to_total_bal'] = df['revol_bal'] / (df['tot_cur_bal'] + 1)
        df['avg_account_bal'] = df['tot_cur_bal'] / (df['total_acc'] + 1)
    df['delinq_per_year'] = df['delinq_2yrs'] / (df['credit_history_months'] / 12 + 1)
    df['inq_per_open_acc'] = df['inq_last_6mths'] / (df['open_acc'] + 1)
    df['revol_bal_to_income'] = df['revol_bal'] / (df['annual_inc'] + 1)
    if 'acc_open_past_24mths' in df.columns:
        df['new_acc_rate'] = df['acc_open_past_24mths'] / (df['total_acc'] + 1)
    df['is_long_term'] = (df['term'] == 60).astype(int)
    return df


# ── Job title classification (rule-based stage) ─────────────

def classify_job_title_rules(title):
    """Rule-based job title classification. Handles ~59% of emp_title values.
    Remaining 'Other' titles can be sent to LLM for semantic categorization."""
    if pd.isna(title):
        return 'Unknown'
    t = str(title).lower().strip()

    patterns = [
        (r'\b(rn|lpn|cna|nurse|nursing|doctor|physician|surgeon|dentist|pharmacist|therapist|paramedic|emt|medical|clinical|health|hospital|dental|optom|chiro|psych|radiol|anesthes|patholog|midwife|phlebotom|caregiver|aide)\b', 'Healthcare'),
        (r'\b(teacher|professor|principal|instructor|tutor|coach|school|education|academic|faculty|librarian|dean|superintendent|counselor)\b', 'Education'),
        (r'\b(software|developer|programmer|engineer.*software|web dev|data scien|data analy|data engineer|machine learn|devops|sysadmin|system admin|network admin|it |it$|information tech|cyber|cloud|database|dba|full.?stack|front.?end|back.?end|sre|qa engineer|test engineer|ux|ui designer)\b', 'Technology'),
        (r'\b(account|cpa|auditor|bookkeeper|finance|financial|banker|loan officer|underwriter|actuary|tax |treasurer|controller|cfo|comptroller|credit analyst|investment|broker|trader)\b', 'Finance_Accounting'),
        (r'\b(attorney|lawyer|paralegal|legal|judge|solicitor|counsel|litigation|law clerk|barrister)\b', 'Legal'),
        (r'\b(military|army|navy|marine|air force|sergeant|colonel|captain|lieutenant|officer.*police|police|sheriff|trooper|detective|federal|government|postal|firefighter|fire fighter|corrections|border patrol|tsa|fbi|cia|dea)\b', 'Government_Military'),
        (r'\b(ceo|coo|cto|cfo|cio|president|vice president|vp |director|executive|partner|founder|owner|chief|gm |general manager)\b', 'Management_Executive'),
        (r'\b(engineer|architect|scientist|researcher|chemist|biologist|physicist|geologist|environmental|civil eng|mechanical eng|electrical eng|chemical eng|aerospace|structural|surveyor|drafter)\b', 'Science_Engineering'),
        (r'\b(sales|realtor|real estate|marketing|advertising|brand|merchandis|buyer|purchasing|business develop|account exec|territory|retail.*manager)\b', 'Sales_Marketing'),
        (r'\b(admin|secretary|receptionist|clerk|office manage|assistant|coordinator|scheduler|dispatcher|hr |human resource|recruiter|payroll)\b', 'Admin_Clerical'),
        (r'\b(driver|truck|mechanic|electrician|plumber|carpenter|welder|machin|warehouse|forklift|construction|hvac|painter|roofer|landscap|janitor|custodian|maintenance|technician|installer|assembl|factory|manufactur|production|operator|laborer)\b', 'Trades_Labor'),
        (r'\b(server|bartender|cook|chef|restaurant|hotel|hospitality|retail|cashier|barista|waitress|waiter|housekeeper|stylist|barber|beautician|customer service|call center|flight attendant)\b', 'Service_Hospitality'),
        (r'\b(manager|supervisor|superintendent|lead|foreman|team lead)\b', 'Management_Executive'),
    ]
    for pattern, category in patterns:
        if re.search(pattern, t):
            return category
    return 'Other'
