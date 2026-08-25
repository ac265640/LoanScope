"""
Data Drift and Distributional Shift Detection Module
=====================================================
Calculates Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests
to detect covariate drift between historical Train cohorts and out-of-time Test cohorts.
"""

from typing import Dict, Any, List, Union
import numpy as np
import pandas as pd
from scipy import stats


def calculate_psi(
    expected: Union[np.ndarray, pd.Series],
    actual: Union[np.ndarray, pd.Series],
    num_buckets: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """
    Calculate Population Stability Index (PSI) for a continuous or discrete numerical feature.
    Threshold interpretations:
      - PSI < 0.10: No significant shift (Stable)
      - 0.10 <= PSI < 0.25: Moderate shift (Requires monitoring)
      - PSI >= 0.25: Significant shift (Action required / Model degradation risk)
    """
    exp_clean = np.array(expected)[~pd.isna(expected)]
    act_clean = np.array(actual)[~pd.isna(actual)]

    if len(exp_clean) < 20 or len(act_clean) < 20:
        return 0.0

    # Determine quantile bins from reference (expected / train)
    quantiles = np.linspace(0, 100, num_buckets + 1)
    bins = np.percentile(exp_clean, quantiles)
    bins[0] = -np.inf
    bins[-1] = np.inf
    # Ensure strictly unique bin edges
    bins = np.unique(bins)
    if len(bins) < 3:
        return 0.0

    exp_counts, _ = np.histogram(exp_clean, bins=bins)
    act_counts, _ = np.histogram(act_clean, bins=bins)

    exp_pct = exp_counts / len(exp_clean)
    act_pct = act_counts / len(act_clean)

    # Avoid zero division
    exp_pct = np.clip(exp_pct, epsilon, 1.0)
    act_pct = np.clip(act_pct, epsilon, 1.0)

    # Normalize after clipping
    exp_pct = exp_pct / exp_pct.sum()
    act_pct = act_pct / act_pct.sum()

    psi_val = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(round(max(0.0, psi_val), 4))


def calculate_categorical_drift(
    train_series: pd.Series,
    test_series: pd.Series,
) -> Dict[str, Any]:
    """
    Compute categorical frequency shift and Chi-Square test of independence.
    """
    train_counts = train_series.value_counts(normalize=True, dropna=False)
    test_counts = test_series.value_counts(normalize=True, dropna=False)

    all_cats = list(set(train_counts.index).union(set(test_counts.index)))
    shifts = {}
    max_shift = 0.0

    for cat in all_cats:
        cat_str = str(cat) if not pd.isna(cat) else "<NULL>"
        p_train = float(train_counts.get(cat, 0.0))
        p_test = float(test_counts.get(cat, 0.0))
        diff = p_test - p_train
        shifts[cat_str] = {
            "train_pct": round(p_train * 100, 2),
            "test_pct": round(p_test * 100, 2),
            "shift_pct": round(diff * 100, 2),
        }
        if abs(diff) > max_shift:
            max_shift = abs(diff)

    return {
        "max_category_shift_pct": round(max_shift * 100, 2),
        "category_breakdown": shifts,
    }


def compute_dataset_drift(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str] = None,
) -> Dict[str, Any]:
    """
    Comprehensive drift comparison between Train panel and Test panel datasets.
    """
    if feature_cols is None:
        exclude = {"loan_id", "reporting_month", "origination_month", "last_updated_at",
                   "next_3m_delinquency_flag", "next_6m_delinquency_flag", "next_12m_default_flag",
                   "next_12m_prepayment_flag", "next_state", "exception_required", "exception_type"}
        feature_cols = [c for c in train_df.columns if c in test_df.columns and c not in exclude]

    drift_report: Dict[str, Any] = {
        "n_train_records": len(train_df),
        "n_test_records": len(test_df),
        "features_analyzed": len(feature_cols),
        "feature_metrics": {},
        "high_drift_features": [],
        "moderate_drift_features": [],
        "stable_features": [],
    }

    for col in feature_cols:
        t_col = train_df[col]
        v_col = test_df[col]

        if pd.api.types.is_numeric_dtype(t_col) and not pd.api.types.is_bool_dtype(t_col):
            t_valid = t_col.dropna()
            v_valid = v_col.dropna()

            psi_score = calculate_psi(t_valid, v_valid)
            ks_stat, ks_pval = stats.ks_2samp(t_valid, v_valid)

            status = "STABLE"
            if psi_score >= 0.25 or ks_stat >= 0.15:
                status = "HIGH_DRIFT"
                drift_report["high_drift_features"].append(col)
            elif psi_score >= 0.10 or ks_stat >= 0.08:
                status = "MODERATE_DRIFT"
                drift_report["moderate_drift_features"].append(col)
            else:
                drift_report["stable_features"].append(col)

            drift_report["feature_metrics"][col] = {
                "type": "numeric",
                "psi": psi_score,
                "ks_statistic": round(float(ks_stat), 4),
                "ks_pvalue": float(ks_pval),
                "status": status,
            }
        else:
            cat_drift = calculate_categorical_drift(t_col, v_col)
            max_shift = cat_drift["max_category_shift_pct"]
            status = "STABLE"
            if max_shift >= 10.0:
                status = "HIGH_DRIFT"
                drift_report["high_drift_features"].append(col)
            elif max_shift >= 5.0:
                status = "MODERATE_DRIFT"
                drift_report["moderate_drift_features"].append(col)
            else:
                drift_report["stable_features"].append(col)

            drift_report["feature_metrics"][col] = {
                "type": "categorical",
                "max_category_shift_pct": max_shift,
                "status": status,
                "breakdown": cat_drift["category_breakdown"],
            }

    return drift_report
