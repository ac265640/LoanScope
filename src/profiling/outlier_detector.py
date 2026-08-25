"""
Outlier Detection Module
========================
Provides univariate (IQR, modified Z-score) and multivariate (Isolation Forest)
outlier detection across financial and delinquency features with statistical justification.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_outliers_iqr(series: pd.Series, factor: float = 1.5) -> Dict[str, Any]:
    """
    Detect outliers using Tukey's Interquartile Range (IQR) rule.
    """
    valid = series.dropna()
    if len(valid) == 0:
        return {"n_outliers": 0, "pct_outliers": 0.0, "lower_bound": None, "upper_bound": None}

    q25 = float(valid.quantile(0.25))
    q75 = float(valid.quantile(0.75))
    iqr = q75 - q25
    lower_bound = q25 - factor * iqr
    upper_bound = q75 + factor * iqr

    outliers = valid[(valid < lower_bound) | (valid > upper_bound)]
    return {
        "method": f"IQR (factor={factor})",
        "q25": round(q25, 4),
        "q75": round(q75, 4),
        "iqr": round(iqr, 4),
        "lower_bound": round(lower_bound, 4),
        "upper_bound": round(upper_bound, 4),
        "n_outliers": int(len(outliers)),
        "pct_outliers": round(len(outliers) / len(series) * 100, 3),
        "extreme_min_outlier": float(outliers.min()) if not outliers.empty else None,
        "extreme_max_outlier": float(outliers.max()) if not outliers.empty else None,
    }


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> Dict[str, Any]:
    """
    Detect outliers using standard Z-score deviations (> threshold std from mean).
    """
    valid = series.dropna()
    if len(valid) < 2 or valid.std() == 0:
        return {"n_outliers": 0, "pct_outliers": 0.0}

    mean = float(valid.mean())
    std = float(valid.std())
    z_scores = np.abs((valid - mean) / std)
    outliers = valid[z_scores > threshold]

    return {
        "method": f"Z-Score (threshold={threshold})",
        "mean": round(mean, 4),
        "std": round(std, 4),
        "n_outliers": int(len(outliers)),
        "pct_outliers": round(len(outliers) / len(series) * 100, 3),
    }


def detect_multivariate_outliers(
    df: pd.DataFrame,
    features: List[str] = None,
    contamination: float = 0.01,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Perform multivariate outlier detection using Isolation Forest on key financial variables.
    Justification: Isolation Forest effectively captures non-linear interactions across
    multidimensional continuous spaces without assuming normality.
    """
    if features is None:
        features = [
            col for col in ["current_balance", "original_balance", "interest_rate", "days_past_due", "loan_age_months"]
            if col in df.columns
        ]

    sub_df = df[features].dropna()
    if len(sub_df) < 50:
        return {"status": "insufficient_data"}

    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
        n_jobs=-1,
    )
    preds = iso.fit_predict(sub_df)
    scores = -iso.decision_function(sub_df)  # higher score = more anomalous

    n_anom = int((preds == -1).sum())
    return {
        "method": "Isolation Forest (Multivariate)",
        "features_evaluated": features,
        "contamination": contamination,
        "n_anomalies_flagged": n_anom,
        "pct_anomalies": round(n_anom / len(sub_df) * 100, 3),
        "score_min": round(float(scores.min()), 4),
        "score_median": round(float(np.median(scores)), 4),
        "score_max": round(float(scores.max()), 4),
    }


def run_comprehensive_outlier_audit(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run full outlier inspection across relevant numerical columns in panel dataset.
    """
    numeric_cols = ["interest_rate", "current_balance", "original_balance", "days_past_due", "remaining_term_months"]
    target_cols = [c for c in numeric_cols if c in df.columns]

    audit: Dict[str, Any] = {
        "univariate_iqr": {},
        "univariate_zscore": {},
        "multivariate_isolation_forest": {},
    }

    for col in target_cols:
        audit["univariate_iqr"][col] = detect_outliers_iqr(df[col])
        audit["univariate_zscore"][col] = detect_outliers_zscore(df[col])

    audit["multivariate_isolation_forest"] = detect_multivariate_outliers(df)
    return audit
