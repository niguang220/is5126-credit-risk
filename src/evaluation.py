"""
Model evaluation utilities for credit risk PD models.
Extracted from notebook 03 (baseline_modeling).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, brier_score_loss,
)
from sklearn.calibration import calibration_curve


def calc_ks(y_true, y_prob):
    """Kolmogorov-Smirnov statistic — max separation between TPR and FPR."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(max(tpr - fpr))


def evaluate(y_true, y_prob, name='Model'):
    """Full credit risk evaluation suite: AUC, KS, Gini, Brier, AP."""
    auc = roc_auc_score(y_true, y_prob)
    ks = calc_ks(y_true, y_prob)
    gini = 2 * auc - 1
    brier = brier_score_loss(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    return {
        'model': name,
        'AUC': round(auc, 4),
        'KS': round(ks, 4),
        'Gini': round(gini, 4),
        'Brier': round(brier, 4),
        'AP': round(ap, 4),
    }


def print_metrics(metrics):
    """Pretty-print a metrics dict from evaluate()."""
    print(f'  AUC: {metrics["AUC"]:.4f} | KS: {metrics["KS"]:.4f} '
          f'| Gini: {metrics["Gini"]:.4f} | Brier: {metrics["Brier"]:.4f}')


def compare_models(results_list):
    """Create a comparison DataFrame from a list of evaluate() outputs."""
    df = pd.DataFrame(results_list).set_index('model')
    return df.sort_values('AUC', ascending=False)


def get_rejection_reasons(model, feature_names, X_single, shap_values_single, top_n=3):
    """Return top-N SHAP-based reasons for a high-risk prediction.

    Args:
        model: trained model (unused, kept for interface consistency)
        feature_names: list of feature names
        X_single: single row of features (1D array)
        shap_values_single: SHAP values for that row (1D array)
        top_n: number of top reasons to return

    Returns:
        list of (feature_name, shap_value, feature_value) tuples
    """
    feature_impact = list(zip(feature_names, shap_values_single, X_single))
    feature_impact.sort(key=lambda x: -abs(x[1]))
    return [(name, shap_val, feat_val) for name, shap_val, feat_val in feature_impact[:top_n]]
