"""
Distribution drift monitoring for cross-regional model transfer.
Used in Phase 3 to quantify feature distribution shift between US and SG data.
"""

import numpy as np
import pandas as pd


def calculate_psi(expected, actual, bins=10):
    """Population Stability Index — quantifies distribution shift between two datasets.

    PSI interpretation (industry standard):
        < 0.10  — insignificant shift
        0.10–0.25 — moderate shift, investigate
        > 0.25 — significant shift, action required

    Args:
        expected: array-like, baseline distribution (e.g. US training data)
        actual: array-like, new distribution (e.g. SG data)
        bins: number of bins for discretization

    Returns:
        float: PSI value
    """
    expected = np.array(expected).flatten()
    actual = np.array(actual).flatten()

    # Create bins from expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    breakpoints = np.unique(breakpoints)

    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    # Avoid division by zero
    expected_pct = np.clip(expected_pct, 1e-6, None)
    actual_pct = np.clip(actual_pct, 1e-6, None)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def psi_report(df_expected, df_actual, feature_cols, bins=10):
    """Compute PSI for all features and return a sorted report.

    Returns:
        DataFrame with columns: feature, PSI, status
    """
    results = []
    for col in feature_cols:
        if col not in df_expected.columns or col not in df_actual.columns:
            continue
        exp = df_expected[col].dropna()
        act = df_actual[col].dropna()
        if len(exp) == 0 or len(act) == 0:
            continue
        psi = calculate_psi(exp, act, bins=bins)
        if psi < 0.10:
            status = 'Stable'
        elif psi < 0.25:
            status = 'Moderate drift'
        else:
            status = 'Significant drift'
        results.append({'feature': col, 'PSI': round(psi, 4), 'status': status})

    return pd.DataFrame(results).sort_values('PSI', ascending=False).reset_index(drop=True)


def classify_features_by_drift(psi_df, threshold_universal=0.10, threshold_regional=0.25):
    """Classify features as universal vs region-specific based on PSI.

    Returns:
        dict with keys 'universal', 'moderate', 'region_specific', each a list of feature names
    """
    return {
        'universal': psi_df[psi_df['PSI'] < threshold_universal]['feature'].tolist(),
        'moderate': psi_df[(psi_df['PSI'] >= threshold_universal) &
                           (psi_df['PSI'] < threshold_regional)]['feature'].tolist(),
        'region_specific': psi_df[psi_df['PSI'] >= threshold_regional]['feature'].tolist(),
    }
